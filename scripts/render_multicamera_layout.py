from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Batch

from smart.datasets.scalable_dataset import MultiDataset
from smart.model import (
    SMART,
    SMARTAutoregressiveDiffusion,
    SMARTCausalDiffusion,
    SMARTCausalFlowMatching,
    SMARTDiffusion,
    SMARTEmbeddedLanguageFlow,
    SMARTHybridDiffusion,
    SMARTActionChunkDiffusion,
    SMARTContinuousActionDiffusion,
    SMARTDiscreteDiffusionPolicy,
)
from smart.transforms import WaymoTargetBuilder
from smart.utils.config import load_config_act
from smart.utils.log import Logging
from smart.utils.torch_compat import register_checkpoint_safe_globals


PREDICTORS = {
    "smart": SMART,
    "smart_diffusion": SMARTDiffusion,
    "smart_ar_diffusion": SMARTAutoregressiveDiffusion,
    "smart_causal_diffusion": SMARTCausalDiffusion,
    "smart_causal_flow_matching": SMARTCausalFlowMatching,
    "smart_elf": SMARTEmbeddedLanguageFlow,
    "smart_hybrid_diffusion": SMARTHybridDiffusion,
    "smart_action_chunk_diffusion": SMARTActionChunkDiffusion,
    "smart_continuous_action_diffusion": SMARTContinuousActionDiffusion,
    "smart_discrete_diffusion_policy": SMARTDiscreteDiffusionPolicy,
}


@dataclass(frozen=True)
class CameraSpec:
    name: str
    yaw_deg: float
    width: int = 640
    height: int = 360
    fx: float = 560.0
    fy: float = 560.0
    cx: float = 320.0
    cy: float = 180.0
    height_m: float = 1.6
    min_depth: float = 0.5


def default_camera_specs() -> list[CameraSpec]:
    return [
        CameraSpec("CAM_FRONT", 0.0),
        CameraSpec("CAM_FRONT_LEFT", 55.0),
        CameraSpec("CAM_FRONT_RIGHT", -55.0),
        CameraSpec("CAM_BACK", 180.0),
        CameraSpec("CAM_BACK_LEFT", 125.0),
        CameraSpec("CAM_BACK_RIGHT", -125.0),
    ]


def _to_float_tensor(value, shape_last=None):
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if shape_last is not None and tensor.shape[-1] != shape_last:
        raise ValueError(f"Expected last dim {shape_last}, got {tuple(tensor.shape)}")
    return tensor


def box_corners_3d(center_xy, heading, size_lwh):
    center_xy = _to_float_tensor(center_xy, 2)
    size_lwh = _to_float_tensor(size_lwh)
    length = max(float(size_lwh[0].item()), 0.2)
    width = max(float(size_lwh[1].item()), 0.2)
    height = max(float(size_lwh[2].item()) if size_lwh.numel() >= 3 else 1.5, 0.2)
    heading = float(heading)
    forward = torch.tensor([math.cos(heading), math.sin(heading)], dtype=torch.float32)
    lateral = torch.tensor([-math.sin(heading), math.cos(heading)], dtype=torch.float32)
    footprint = torch.stack(
        [
            center_xy + 0.5 * length * forward + 0.5 * width * lateral,
            center_xy + 0.5 * length * forward - 0.5 * width * lateral,
            center_xy - 0.5 * length * forward - 0.5 * width * lateral,
            center_xy - 0.5 * length * forward + 0.5 * width * lateral,
        ],
        dim=0,
    )
    bottom = torch.cat([footprint, torch.zeros(4, 1)], dim=-1)
    top = torch.cat([footprint, torch.full((4, 1), height)], dim=-1)
    return torch.cat([bottom, top], dim=0)


def project_points(points_world, ego_xy, ego_heading, camera: CameraSpec):
    points_world = _to_float_tensor(points_world)
    if points_world.dim() != 2:
        raise ValueError("points_world must have shape [N,2] or [N,3].")
    if points_world.shape[-1] == 2:
        points_world = torch.cat([points_world, torch.zeros(points_world.shape[0], 1)], dim=-1)
    ego_xy = _to_float_tensor(ego_xy, 2)
    yaw = float(ego_heading) + math.radians(camera.yaw_deg)
    forward = torch.tensor([math.cos(yaw), math.sin(yaw)], dtype=torch.float32)
    right = torch.tensor([-math.sin(yaw), math.cos(yaw)], dtype=torch.float32)
    rel_xy = points_world[:, :2] - ego_xy.unsqueeze(0)
    depth = rel_xy.matmul(forward)
    horizontal = rel_xy.matmul(right)
    vertical = points_world[:, 2] - float(camera.height_m)
    safe_depth = depth.clamp_min(float(camera.min_depth))
    u = float(camera.cx) + float(camera.fx) * horizontal / safe_depth
    v = float(camera.cy) - float(camera.fy) * vertical / safe_depth
    pixels = torch.stack([u, v], dim=-1)
    visible = depth > float(camera.min_depth)
    return pixels, visible


def build_synthetic_layout_scene(num_frames=2):
    ego_xy = torch.zeros(num_frames, 2)
    ego_heading = torch.zeros(num_frames)
    agent_xy = torch.stack(
        [torch.tensor([10.0 + 1.0 * idx, 0.5]) for idx in range(num_frames)],
        dim=0,
    )
    return {
        "scenario_id": "synthetic",
        "ego_xy": ego_xy,
        "ego_heading": ego_heading,
        "agents": [
            {
                "track_id": 1,
                "type": 0,
                "xy": agent_xy,
                "heading": torch.zeros(num_frames),
                "size_lwh": torch.tensor([4.5, 1.9, 1.6]),
                "valid": torch.ones(num_frames, dtype=torch.bool),
            }
        ],
        "map_polylines": [
            torch.tensor([[0.0, -3.5], [25.0, -3.5]], dtype=torch.float32),
            torch.tensor([[0.0, 3.5], [25.0, 3.5]], dtype=torch.float32),
        ],
    }


def _camera_image_path(output_dir: Path, frame_index: int, camera_name: str) -> Path:
    return output_dir / f"frame_{frame_index:04d}" / f"{camera_name}.png"


def _draw_projected_polyline(ax, polyline, ego_xy, ego_heading, camera, color, linewidth, alpha):
    pixels, visible = project_points(polyline, ego_xy, ego_heading, camera)
    for start in range(max(0, pixels.shape[0] - 1)):
        if not bool(visible[start] and visible[start + 1]):
            continue
        segment = pixels[start:start + 2]
        ax.plot(
            segment[:, 0].tolist(),
            segment[:, 1].tolist(),
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            zorder=2,
        )


def _draw_projected_box(ax, agent, frame_index, ego_xy, ego_heading, camera):
    if not bool(agent["valid"][frame_index]):
        return False
    corners = box_corners_3d(
        center_xy=agent["xy"][frame_index],
        heading=float(agent["heading"][frame_index].item()),
        size_lwh=agent["size_lwh"],
    )
    pixels, visible = project_points(corners, ego_xy, ego_heading, camera)
    if not bool(visible.any()):
        return False
    edges = (
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    )
    color_by_type = {
        0: "#39a0ff",
        1: "#f2c94c",
        2: "#6fcf97",
    }
    color = color_by_type.get(int(agent.get("type", 0)), "#ffffff")
    drawn = False
    for src, dst in edges:
        if not bool(visible[src] and visible[dst]):
            continue
        ax.plot(
            [float(pixels[src, 0]), float(pixels[dst, 0])],
            [float(pixels[src, 1]), float(pixels[dst, 1])],
            color=color,
            linewidth=1.6,
            alpha=0.95,
            zorder=5,
        )
        drawn = True
    if drawn:
        label_xy = pixels[4:8].mean(dim=0)
        if 0 <= float(label_xy[0]) <= camera.width and 0 <= float(label_xy[1]) <= camera.height:
            ax.text(
                float(label_xy[0]),
                float(label_xy[1]),
                str(agent.get("track_id", "")),
                color=color,
                fontsize=7,
                ha="center",
                va="center",
                zorder=6,
            )
    return drawn


def render_camera_layout(
    scene,
    camera: CameraSpec,
    frame_index: int,
    output_path: Path,
    draw_map_polylines: bool = False,
):
    ego_xy = scene["ego_xy"][frame_index]
    ego_heading = float(scene["ego_heading"][frame_index].item())
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dpi = 100
    fig = plt.figure(
        figsize=(camera.width / dpi, camera.height / dpi),
        dpi=dpi,
        facecolor="#101114",
    )
    ax = fig.add_axes([0, 0, 1, 1], facecolor="#101114")
    ax.set_xlim(0, camera.width)
    ax.set_ylim(camera.height, 0)
    ax.axis("off")

    if draw_map_polylines:
        for polyline in scene.get("map_polylines", []):
            polyline_3d = torch.cat(
                [_to_float_tensor(polyline, 2), torch.zeros(len(polyline), 1)],
                dim=-1,
            )
            _draw_projected_polyline(
                ax,
                polyline_3d,
                ego_xy,
                ego_heading,
                camera,
                color="#b8c0cc",
                linewidth=1.0,
                alpha=0.8,
            )

    projected_agents = 0
    for agent in scene.get("agents", []):
        projected_agents += int(
            _draw_projected_box(ax, agent, frame_index, ego_xy, ego_heading, camera)
        )

    ax.text(
        8,
        16,
        f"{camera.name} | {scene.get('scenario_id', 'unknown')} | f={frame_index}",
        color="#ffffff",
        fontsize=8,
        ha="left",
        va="top",
        bbox={"facecolor": "#101114", "alpha": 0.65, "edgecolor": "none", "pad": 2.0},
    )
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return projected_agents


def render_layout_sequence(scene, output_dir, cameras=None, frame_indices=None, draw_map_polylines=False):
    output_dir = Path(output_dir)
    cameras = default_camera_specs() if cameras is None else list(cameras)
    if frame_indices is None:
        frame_indices = list(range(int(scene["ego_xy"].shape[0])))
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "scenario_id": scene.get("scenario_id", "unknown"),
        "camera_names": [camera.name for camera in cameras],
        "cameras": [asdict(camera) for camera in cameras],
        "draw_map_polylines": bool(draw_map_polylines),
        "frames": [],
    }
    for frame_index in frame_indices:
        images = {}
        projected_counts = {}
        for camera in cameras:
            image_path = _camera_image_path(output_dir, int(frame_index), camera.name)
            projected_counts[camera.name] = render_camera_layout(
                scene,
                camera,
                int(frame_index),
                image_path,
                draw_map_polylines=draw_map_polylines,
            )
            images[camera.name] = str(image_path.relative_to(output_dir))
        manifest["frames"].append({
            "frame_index": int(frame_index),
            "images": images,
            "projected_agent_counts": projected_counts,
        })
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return {
        "output_dir": str(output_dir),
        "num_frames": len(manifest["frames"]),
        "num_cameras": len(cameras),
        "manifest": str(output_dir / "manifest.json"),
    }


def _scenario_id(graph) -> str:
    scenario_id = getattr(graph, "scenario_id", None)
    if scenario_id is None and hasattr(graph, "get"):
        scenario_id = graph.get("scenario_id", None)
    if scenario_id is None and hasattr(graph, "__contains__") and "scenario_id" in graph:
        scenario_id = graph["scenario_id"]
    if scenario_id is None:
        return "unknown"
    return str(scenario_id).replace("/", "_")


def _load_dataset(config):
    data_config = config.Dataset
    return MultiDataset(
        root=data_config.root,
        split="val",
        raw_dir=data_config.val_raw_dir,
        processed_dir=data_config.val_processed_dir,
        transform=WaymoTargetBuilder(
            config.Model.num_historical_steps,
            config.Model.decoder.num_future_steps,
        ),
        token_size=int(
            getattr(
                data_config,
                "token_size",
                getattr(config.Model.decoder, "token_size", 512),
            )
        ),
    )


def _load_model(config, ckpt_path, device):
    predictor = PREDICTORS[config.Model.predictor]
    model = predictor(config.Model)
    if ckpt_path:
        logger = Logging().log(level="DEBUG")
        model.load_params_from_file(ckpt_path, logger=logger, to_cpu=device.type == "cpu")
    model.to(device)
    model.eval()
    return model


def _prepare_batch(model, batch: Batch) -> Batch:
    if hasattr(model, "_prepare_batch"):
        return model._prepare_batch(batch)
    data = model.match_token_map(batch)
    data = model.sample_pt_pred(data)
    if isinstance(data, Batch):
        data["agent"]["av_index"] += data["agent"]["ptr"][:-1]
    return data


def _prediction_valid_mask(data, prediction, hist_steps, num_frames):
    if "pred_valid_mask" in prediction:
        return prediction["pred_valid_mask"][:, :num_frames].bool()
    return data["agent"]["valid_mask"][:, hist_steps:hist_steps + num_frames].bool()


def _map_polylines_from_data(data):
    if ("map_point", "to", "map_polygon") not in data.edge_types:
        return []
    edge_index = data[("map_point", "to", "map_polygon")]["edge_index"]
    polygon_ids = edge_index[1].long()
    point_indices = edge_index[0].long()
    point_position = data["map_point"]["position"][:, :2].float().cpu()
    polylines = []
    for polygon_id in torch.unique(polygon_ids).tolist():
        indices = point_indices[polygon_ids == polygon_id]
        if indices.numel() >= 2:
            polylines.append(point_position.index_select(0, indices.cpu()))
    return polylines


def layout_scene_from_prediction(data, prediction, scenario_id, num_frames, max_agents=48):
    data = data.cpu()
    prediction = {
        key: value.detach().cpu() if torch.is_tensor(value) else value
        for key, value in prediction.items()
    }
    hist_steps = data["agent"]["valid_mask"].shape[1] - prediction["gt"].shape[1]
    current_step = hist_steps - 1
    num_frames = min(int(num_frames), int(prediction["pred_traj"].shape[1]))
    av_index = int(data["agent"]["av_index"])

    ego_xy = prediction["gt"][av_index, :num_frames].clone()
    ego_valid = data["agent"]["valid_mask"][av_index, hist_steps:hist_steps + num_frames].bool()
    fallback_xy = data["agent"]["position"][av_index, current_step, :2].float()
    ego_xy = torch.where(ego_valid.unsqueeze(-1), ego_xy, fallback_xy.unsqueeze(0))
    ego_heading = data["agent"]["heading"][av_index, hist_steps:hist_steps + num_frames].float().clone()
    fallback_heading = data["agent"]["heading"][av_index, current_step].float()
    ego_heading = torch.where(ego_valid, ego_heading, fallback_heading)

    pred_valid = _prediction_valid_mask(data, prediction, hist_steps, num_frames)
    current_valid = data["agent"]["valid_mask"][:, current_step].bool()
    candidate_mask = current_valid & (data["agent"]["type"].long() != 3)
    candidate_mask[av_index] = False
    agent_indices = torch.nonzero(candidate_mask, as_tuple=False).squeeze(-1)
    if max_agents > 0 and agent_indices.numel() > max_agents:
        distance = torch.norm(
            data["agent"]["position"][agent_indices, current_step, :2].float() - fallback_xy,
            dim=-1,
        )
        agent_indices = agent_indices[torch.argsort(distance)[:max_agents]]

    shapes = data["agent"]["shape"]
    if shapes.dim() == 3:
        shapes = shapes[:, current_step]
    agents = []
    for agent_index in agent_indices.tolist():
        shape = shapes[agent_index].float()
        if shape.numel() < 3:
            shape = torch.cat([shape[:2], shape.new_tensor([1.6])])
        agents.append({
            "track_id": int(agent_index),
            "type": int(data["agent"]["type"][agent_index].item()),
            "xy": prediction["pred_traj"][agent_index, :num_frames].float().clone(),
            "heading": prediction["pred_head"][agent_index, :num_frames].float().clone(),
            "size_lwh": shape[:3].float().clone(),
            "valid": pred_valid[agent_index, :num_frames].bool().clone(),
        })

    return {
        "scenario_id": scenario_id,
        "ego_xy": ego_xy.float(),
        "ego_heading": ego_heading.float(),
        "agents": agents,
        "map_polylines": _map_polylines_from_data(data),
    }


def encode_camera_videos(output_dir: Path, cameras: Iterable[CameraSpec], fps: float):
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise RuntimeError("ffmpeg not found in PATH")
    for camera in cameras:
        cmd = [
            ffmpeg_path,
            "-y",
            "-framerate",
            f"{fps:.8f}",
            "-i",
            str(output_dir / "frame_%04d" / f"{camera.name}.png"),
            "-c:v",
            "mpeg4",
            "-q:v",
            "2",
            "-pix_fmt",
            "yuv420p",
            str(output_dir / f"{camera.name}.mp4"),
        ]
        subprocess.run(cmd, check=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/validation/validation_scalable_hybrid_diffusion.yaml")
    parser.add_argument("--ckpt", default="")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output-dir", default="outputs/multicamera_layout")
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--max-agents", type=int, default=48)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--draw-map-polylines", action="store_true")
    parser.add_argument("--encode-video", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    register_checkpoint_safe_globals()
    config = load_config_act(args.config)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    dataset = _load_dataset(config)
    model = _load_model(config, args.ckpt, device)
    graph = dataset[int(args.index)]
    batch = Batch.from_data_list([graph]).to(device)
    with torch.no_grad():
        prepared = _prepare_batch(model, batch)
        prediction = model.inference(prepared)
    if prediction is None:
        raise RuntimeError("Model inference returned None; cannot render layout.")
    scene = layout_scene_from_prediction(
        prepared,
        prediction,
        scenario_id=_scenario_id(graph),
        num_frames=args.num_frames,
        max_agents=args.max_agents,
    )
    cameras = default_camera_specs()
    output_dir = Path(args.output_dir)
    summary = render_layout_sequence(
        scene,
        output_dir,
        cameras=cameras,
        draw_map_polylines=args.draw_map_polylines,
    )
    if args.encode_video:
        encode_camera_videos(output_dir, cameras, args.fps)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
