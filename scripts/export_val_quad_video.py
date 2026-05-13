from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import math
import shutil
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
import torch
from torch_geometric.data import Batch

from smart.datasets.scalable_dataset import MultiDataset
from smart.model import SMART, SMARTDiffusion
from smart.transforms import WaymoTargetBuilder
from smart.utils.config import load_config_act
from smart.utils.log import Logging
from smart.utils.torch_compat import register_checkpoint_safe_globals


PREDICTORS = {
    "smart": SMART,
    "smart_diffusion": SMARTDiffusion,
}

SHIFT = 5
BASE_RADIUS = 28.0
FOLLOW_SMOOTH_WINDOW = 7


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/validation/validation_scalable.yaml")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--output", type=str, default="outputs/val_quad_15s.mp4")
    parser.add_argument("--indices", type=int, nargs="*", default=None)
    parser.add_argument("--num-scenes", type=int, default=4)
    parser.add_argument("--search-window", type=int, default=32)
    parser.add_argument("--duration-seconds", type=float, default=15.0)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--max-agents", type=int, default=48)
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--camera-mode", choices=["global", "fixed_follow"], default="global")
    parser.add_argument("--fixed-radius", type=float, default=28.0)
    parser.add_argument("--scale-annotations", action="store_true")
    return parser.parse_args()


def load_model(config, ckpt_path, device):
    predictor = PREDICTORS[config.Model.predictor]
    model = predictor(config.Model)
    logger = Logging().log(level="DEBUG")
    model.load_params_from_file(filename=ckpt_path, logger=logger, to_cpu=device.type == "cpu")
    model = model.to(device)
    model.eval()
    return model


def load_dataset(config):
    return MultiDataset(
        root=config.Dataset.root,
        split="val",
        raw_dir=config.Dataset.val_raw_dir,
        processed_dir=config.Dataset.val_processed_dir,
        transform=WaymoTargetBuilder(config.Model.num_historical_steps, config.Model.decoder.num_future_steps),
        token_size=int(getattr(config.Dataset, "token_size", getattr(config.Model.decoder, "token_size", 512))),
    )


def choose_scene_indices(dataset, num_scenes, search_window, default_indices):
    if default_indices:
        if len(default_indices) != num_scenes:
            raise ValueError(f"Expected {num_scenes} indices, got {len(default_indices)}")
        return default_indices

    candidates = []
    limit = min(len(dataset), max(search_window, num_scenes))
    current_step = 10
    for index in range(limit):
        graph = dataset[index]
        valid_now = graph["agent"]["valid_mask"][:, current_step] & (graph["agent"]["type"] != 3)
        score = int(valid_now.sum().item())
        candidates.append((score, index))
    candidates.sort(reverse=True)
    return [index for _, index in candidates[:num_scenes]]


def scenario_id_from_graph(graph):
    scenario_id = getattr(graph, "scenario_id", None)
    if scenario_id is None and hasattr(graph, "get"):
        scenario_id = graph.get("scenario_id", None)
    if scenario_id is None and hasattr(graph, "__contains__") and "scenario_id" in graph:
        scenario_id = graph["scenario_id"]
    if scenario_id is None:
        return "unknown"
    return str(scenario_id).replace("/", "_")


def pad_time_tensor(agent, key, extra_steps, pad_value=0):
    if extra_steps <= 0:
        return
    value = agent[key]
    pad_shape = list(value.shape)
    pad_shape[1] = extra_steps
    pad = value.new_full(pad_shape, pad_value)
    agent[key] = torch.cat([value, pad], dim=1)



def extend_graph_for_rollout(graph, target_future_steps, hist_steps):
    graph = deepcopy(graph)
    agent = graph["agent"]
    total_steps = hist_steps + target_future_steps
    current_step = hist_steps - 1
    old_total_steps = agent["position"].shape[1]
    extra_steps = total_steps - old_total_steps
    if extra_steps < 0:
        raise ValueError(f"Target total steps {total_steps} is shorter than source steps {old_total_steps}")

    current_valid = agent["valid_mask"][:, current_step].clone()

    pad_time_tensor(agent, "valid_mask", extra_steps, False)
    pad_time_tensor(agent, "predict_mask", extra_steps, False)
    pad_time_tensor(agent, "position", extra_steps, 0)
    pad_time_tensor(agent, "heading", extra_steps, 0)
    pad_time_tensor(agent, "velocity", extra_steps, 0)
    pad_time_tensor(agent, "shape", extra_steps, 0)

    agent["valid_mask"][:, hist_steps:total_steps] = current_valid[:, None]
    agent["predict_mask"][:, hist_steps:total_steps] = current_valid[:, None]
    agent["shape"][:, hist_steps:total_steps] = agent["shape"][:, current_step : current_step + 1]

    token_steps_old = agent["token_pos"].shape[1]
    token_steps_new = math.floor((total_steps - (SHIFT + 1)) / SHIFT) + 1
    token_extra = token_steps_new - token_steps_old
    if token_extra < 0:
        raise ValueError(f"Target token steps {token_steps_new} is shorter than source token steps {token_steps_old}")

    for key in ["token_idx", "token_pos", "token_heading", "agent_valid_mask", "token_velocity"]:
        pad_time_tensor(agent, key, token_extra, 0)

    if token_extra > 0:
        contour = agent["token_contour"]
        pad_shape = list(contour.shape)
        pad_shape[1] = token_extra
        contour_pad = contour.new_zeros(pad_shape)
        agent["token_contour"] = torch.cat([contour, contour_pad], dim=1)

    token_hist_idx = current_step // SHIFT
    token_valid_now = agent["agent_valid_mask"][:, token_hist_idx].clone()
    agent["agent_valid_mask"][:, token_hist_idx:] = token_valid_now[:, None]

    return graph



def prepare_batch(model, graph):
    batch = Batch.from_data_list([graph]).to(model.device)
    if hasattr(model, "_prepare_batch"):
        return model._prepare_batch(batch)
    data = model.match_token_map(batch)
    data = model.sample_pt_pred(data)
    if isinstance(data, Batch):
        data["agent"]["av_index"] += data["agent"]["ptr"][:-1]
    return data



def pick_agent_indices(data, max_agents, hist_steps):
    current_step = hist_steps - 1
    valid_agents = data["agent"]["valid_mask"][:, current_step] & (data["agent"]["type"] != 3)
    if valid_agents.any():
        agent_indices = torch.nonzero(valid_agents, as_tuple=False).squeeze(-1)
    else:
        agent_indices = torch.arange(data["agent"]["num_nodes"])

    av_index = int(data["agent"]["av_index"])
    if max_agents > 0 and agent_indices.numel() > max_agents:
        anchor = data["agent"]["position"][av_index, current_step, :2]
        distance = torch.norm(data["agent"]["position"][agent_indices, current_step, :2] - anchor, dim=-1)
        keep = torch.argsort(distance)[:max_agents]
        agent_indices = agent_indices[keep]
        if av_index not in agent_indices.tolist():
            agent_indices = torch.cat([torch.tensor([av_index]), agent_indices[:-1]])

    return agent_indices, av_index



def compute_view_radius(data, prediction, agent_indices, hist_steps, av_index):
    anchor = data["agent"]["position"][av_index, hist_steps - 1, :2]
    distances = [torch.tensor(20.0)]
    for agent_index in agent_indices.tolist():
        history_mask = data["agent"]["valid_mask"][agent_index, :hist_steps]
        if history_mask.any():
            history = data["agent"]["position"][agent_index, :hist_steps, :2][history_mask]
            distances.append(torch.norm(history - anchor, dim=-1).max())
        future_mask = data["agent"]["valid_mask"][agent_index, hist_steps:]
        if future_mask.any():
            pred = prediction["pred_traj"][agent_index][future_mask]
            if pred.numel() > 0:
                distances.append(torch.norm(pred - anchor, dim=-1).max())
    radius = float(torch.stack(distances).max().item()) + 12.0
    return max(45.0, min(radius, 130.0))


def smooth_camera_track(track, window_size=FOLLOW_SMOOTH_WINDOW):
    if track.ndim != 2 or track.shape[0] == 0:
        return track

    window_size = max(1, int(window_size))
    if window_size == 1 or track.shape[0] < 3:
        return track.clone()

    radius = window_size // 2
    smoothed = track.clone()
    for index in range(track.shape[0]):
        start = max(0, index - radius)
        end = min(track.shape[0], index + radius + 1)
        smoothed[index] = track[start:end].mean(dim=0)
    return smoothed



def prepare_scene(model, graph, max_agents, target_future_steps, hist_steps):
    extended_graph = extend_graph_for_rollout(graph, target_future_steps=target_future_steps, hist_steps=hist_steps)
    with torch.no_grad():
        data = prepare_batch(model, extended_graph)
        prediction = model.inference(data)
    data_cpu = data.cpu()
    prediction_cpu = {key: value.detach().cpu() if torch.is_tensor(value) else value for key, value in prediction.items()}
    agent_indices, av_index = pick_agent_indices(data_cpu, max_agents, hist_steps)
    follow_anchor_track = smooth_camera_track(prediction_cpu["pred_traj"][av_index, :target_future_steps])
    return {
        "scenario_id": scenario_id_from_graph(graph),
        "data": data_cpu,
        "prediction": prediction_cpu,
        "agent_indices": agent_indices,
        "hist_steps": hist_steps,
        "av_index": av_index,
        "global_radius": compute_view_radius(data_cpu, prediction_cpu, agent_indices, hist_steps, av_index),
        "follow_anchor_track": follow_anchor_track,
        "target_future_steps": target_future_steps,
    }



def get_camera_radius(scene, camera_mode, fixed_radius):
    if camera_mode == "fixed_follow":
        return float(fixed_radius)
    return float(scene["global_radius"])



def get_camera_anchor(scene, future_step, camera_mode):
    data = scene["data"]
    prediction = scene["prediction"]
    hist_steps = scene["hist_steps"]
    av_index = scene["av_index"]
    base_anchor = data["agent"]["position"][av_index, hist_steps - 1, :2]
    if camera_mode != "fixed_follow" or future_step <= 0:
        return base_anchor
    future_index = min(future_step - 1, prediction["pred_traj"].shape[1] - 1)
    follow_anchor_track = scene.get("follow_anchor_track")
    if follow_anchor_track is not None and future_index < follow_anchor_track.shape[0]:
        return follow_anchor_track[future_index]
    return prediction["pred_traj"][av_index, future_index]



def get_annotation_scale(scene, camera_mode, camera_radius, scale_annotations):
    if not scale_annotations:
        return 1.0
    reference = BASE_RADIUS
    radius = camera_radius if camera_mode == "fixed_follow" else float(scene["global_radius"])
    scale = reference / max(radius, reference)
    return max(0.3, min(1.0, float(scale)))



def style_params(scene, camera_mode, fixed_radius, scale_annotations):
    camera_radius = get_camera_radius(scene, camera_mode, fixed_radius)
    scale = get_annotation_scale(scene, camera_mode, camera_radius, scale_annotations)
    return {
        "camera_radius": camera_radius,
        "scale": scale,
        "lane_linewidth": max(0.35, 0.9 * scale),
        "lane_alpha": 0.2,
        "marker_size": max(20.0, 140.0 * scale * scale),
        "marker_edge_width": max(0.6, 1.4 * scale),
        "shape_linewidth": max(0.5, 1.0 * scale),
        "arrow_linewidth": max(0.5, 1.0 * scale),
        "ego_traj_linewidth": max(0.8, 2.2 * scale),
        "other_traj_linewidth": max(0.6, 1.3 * scale),
    }



def draw_light_markers(ax, data, point_to_polygon, polygon_ids, point_position, marker_size, marker_edge_width):
    if "light_type" not in data["map_polygon"]:
        return

    light_type = data["map_polygon"]["light_type"]
    state_colors = {
        0: "#d62728",
        1: "#2ca02c",
        2: "#f1c40f",
    }

    for polygon_id in torch.unique(polygon_ids).tolist():
        state = int(light_type[polygon_id].item())
        if state not in state_colors:
            continue
        indices = point_to_polygon[0, polygon_ids == polygon_id]
        if indices.numel() == 0:
            continue
        marker_xy = point_position[indices[0]]
        ax.scatter(
            marker_xy[0].item(),
            marker_xy[1].item(),
            s=marker_size,
            c=state_colors[state],
            edgecolors="white",
            linewidths=marker_edge_width,
            alpha=0.95,
            zorder=2,
        )



def draw_map(ax, data, lane_linewidth, lane_alpha, marker_size, marker_edge_width):
    point_to_polygon = data[("map_point", "to", "map_polygon")]["edge_index"]
    polygon_ids = point_to_polygon[1].long()
    point_position = data["map_point"]["position"][:, :2]
    for polygon_id in torch.unique(polygon_ids).tolist():
        indices = point_to_polygon[0, polygon_ids == polygon_id]
        if indices.numel() < 2:
            continue
        points = point_position.index_select(0, indices)
        ax.plot(points[:, 0], points[:, 1], color="#000000", linewidth=lane_linewidth, alpha=lane_alpha, zorder=1)
    draw_light_markers(ax, data, point_to_polygon, polygon_ids, point_position, marker_size, marker_edge_width)



def oriented_box(center_xy, heading, length, width):
    cos = torch.cos(torch.tensor(heading, dtype=torch.float32))
    sin = torch.sin(torch.tensor(heading, dtype=torch.float32))
    forward = torch.tensor([cos, sin]) * (length * 0.5)
    lateral = torch.tensor([-sin, cos]) * (width * 0.5)
    corners = [
        center_xy + forward + lateral,
        center_xy + forward - lateral,
        center_xy - forward - lateral,
        center_xy - forward + lateral,
    ]
    return [(float(point[0].item()), float(point[1].item())) for point in corners]



def heading_triangle(center_xy, heading, length, width):
    direction = torch.tensor(
        [torch.cos(torch.tensor(heading, dtype=torch.float32)), torch.sin(torch.tensor(heading, dtype=torch.float32))]
    )
    lateral = torch.tensor([-direction[1], direction[0]])
    tip = center_xy + direction * (0.5 * length)
    base_center = center_xy
    base_left = base_center + lateral * (0.35 * width)
    base_right = base_center - lateral * (0.35 * width)
    points = [tip, base_right, base_left]
    return [(float(point[0].item()), float(point[1].item())) for point in points]



def draw_agent_shape(ax, center_xy, heading, shape, agent_type, face_color, edge_color, shape_linewidth, arrow_linewidth):
    length = max(float(shape[0].item()), 0.6)
    width = max(float(shape[1].item()), 0.4)
    face_rgba = matplotlib.colors.to_rgba(face_color, alpha=0.28)

    if agent_type == 1:
        patch = Circle(
            (center_xy[0].item(), center_xy[1].item()),
            radius=0.3 * max(length, width),
            facecolor=face_rgba,
            edgecolor=edge_color,
            linewidth=shape_linewidth,
            zorder=6,
        )
        ax.add_patch(patch)
    else:
        patch = Polygon(
            oriented_box(center_xy, heading, length, width),
            closed=True,
            facecolor=face_rgba,
            edgecolor=edge_color,
            linewidth=shape_linewidth,
            zorder=6,
        )
        ax.add_patch(patch)

    arrow = Polygon(
        heading_triangle(center_xy, heading, length, width),
        closed=True,
        facecolor="none",
        edgecolor=edge_color,
        linewidth=arrow_linewidth,
        alpha=0.95,
        zorder=7,
    )
    ax.add_patch(arrow)



def draw_scene(ax, scene, frame_index, fps, camera_mode="global", fixed_radius=BASE_RADIUS, scale_annotations=False):
    data = scene["data"]
    prediction = scene["prediction"]
    agent_indices = scene["agent_indices"]
    hist_steps = scene["hist_steps"]
    av_index = scene["av_index"]
    future_step = min(frame_index + 1, scene["target_future_steps"])
    style = style_params(scene, camera_mode, fixed_radius, scale_annotations)

    draw_map(
        ax,
        data,
        lane_linewidth=style["lane_linewidth"],
        lane_alpha=style["lane_alpha"],
        marker_size=style["marker_size"],
        marker_edge_width=style["marker_edge_width"],
    )

    for agent_index in agent_indices.tolist():
        is_ego = agent_index == av_index
        history_color = "#7a4bc2" if is_ego else "#6ea8de"
        edge_color = "#7a4bc2" if is_ego else "#000000"
        linewidth = style["ego_traj_linewidth"] if is_ego else style["other_traj_linewidth"]

        history_mask = data["agent"]["valid_mask"][agent_index, :hist_steps]
        history = data["agent"]["position"][agent_index, :hist_steps, :2][history_mask]
        if history.numel() > 0:
            ax.plot(
                history[:, 0],
                history[:, 1],
                color=history_color,
                linestyle=":",
                linewidth=linewidth,
                alpha=0.95,
                zorder=3,
            )

        pred_mask = data["agent"]["valid_mask"][agent_index, hist_steps : hist_steps + future_step]
        pred = prediction["pred_traj"][agent_index, :future_step][pred_mask]
        if pred.numel() > 0:
            ax.plot(pred[:, 0], pred[:, 1], color="#e15759", linestyle="-", linewidth=linewidth, alpha=0.95, zorder=5)

        if pred_mask.numel() > 0 and bool(pred_mask[-1]):
            current_xy = prediction["pred_traj"][agent_index, future_step - 1]
            current_heading = float(prediction["pred_head"][agent_index, future_step - 1].item())
            shape = data["agent"]["shape"][agent_index, hist_steps - 1, :2]
            agent_type = int(data["agent"]["type"][agent_index].item())
            draw_agent_shape(
                ax,
                current_xy,
                current_heading,
                shape,
                agent_type,
                history_color,
                edge_color,
                shape_linewidth=style["shape_linewidth"],
                arrow_linewidth=style["arrow_linewidth"],
            )

    anchor = get_camera_anchor(scene, future_step, camera_mode)
    radius = style["camera_radius"]
    ax.set_xlim(float(anchor[0].item() - radius), float(anchor[0].item() + radius))
    ax.set_ylim(float(anchor[1].item() - radius), float(anchor[1].item() + radius))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_color("#d9d9d9")
        spine.set_linewidth(0.8)

    time_seconds = future_step / fps
    ax.set_title(f"idx={scene['index']} | {scene['scenario_id']}", fontsize=10, pad=8)
    ax.text(
        0.02,
        0.98,
        f"t=+{time_seconds:.1f}s",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 2.0},
    )



def render_frames(scenes, output_dir, dpi, total_frames, fps, camera_mode="global", fixed_radius=BASE_RADIUS, scale_annotations=False):
    for frame_index in range(total_frames):
        fig, axes = plt.subplots(2, 2, figsize=(14, 14), constrained_layout=False)
        fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.96, wspace=0.04, hspace=0.06)
        for ax, scene in zip(axes.flat, scenes):
            draw_scene(
                ax,
                scene,
                frame_index,
                fps,
                camera_mode=camera_mode,
                fixed_radius=fixed_radius,
                scale_annotations=scale_annotations,
            )
        frame_path = output_dir / f"frame_{frame_index:04d}.png"
        fig.savefig(frame_path, dpi=dpi)
        plt.close(fig)
    return total_frames



def encode_video(frame_dir, output_path, fps):
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise RuntimeError("ffmpeg not found in PATH")
    cmd = [
        ffmpeg_path,
        "-y",
        "-framerate",
        f"{fps:.8f}",
        "-i",
        str(frame_dir / "frame_%04d.png"),
        "-c:v",
        "mpeg4",
        "-q:v",
        "2",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)



def main():
    args = parse_args()
    register_checkpoint_safe_globals()
    config = load_config_act(args.config)

    target_future_steps = int(round(args.duration_seconds * args.fps))
    if target_future_steps % SHIFT != 0:
        raise ValueError(
            f"duration_seconds * fps must be divisible by {SHIFT} for autoregressive rollout; got {target_future_steps} steps"
        )

    hist_steps = int(config.Model.num_historical_steps)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    dataset = load_dataset(config)
    chosen_indices = choose_scene_indices(dataset, args.num_scenes, args.search_window, args.indices)
    model = load_model(config, args.ckpt, device)

    scenes = []
    for index in chosen_indices:
        graph = dataset[index]
        scene = prepare_scene(model, graph, args.max_agents, target_future_steps, hist_steps)
        scene["index"] = index
        scenes.append(scene)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(prefix="smart_val_quad_") as temp_dir:
        frame_dir = Path(temp_dir)
        total_frames = render_frames(
            scenes,
            frame_dir,
            args.dpi,
            target_future_steps,
            args.fps,
            camera_mode=args.camera_mode,
            fixed_radius=args.fixed_radius,
            scale_annotations=args.scale_annotations,
        )
        encode_video(frame_dir, output_path, args.fps)

    print(f"Saved video to {output_path}")
    print(f"Scene indices: {chosen_indices}")
    print(f"Frames: {total_frames}")
    print(f"FPS: {args.fps:.6f}")
    print(f"Future steps: {target_future_steps}")
    print(f"Camera mode: {args.camera_mode}")
    print(f"Fixed radius: {args.fixed_radius:.2f}")
    print(f"Scale annotations: {args.scale_annotations}")


if __name__ == "__main__":
    main()
