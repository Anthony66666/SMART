import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image


class MultiCameraLayoutProjectionTest(unittest.TestCase):
    def test_front_camera_projects_forward_box_inside_image(self):
        from scripts.render_multicamera_layout import (
            box_corners_3d,
            default_camera_specs,
            project_points,
        )

        camera = default_camera_specs()[0]
        corners = box_corners_3d(
            center_xy=torch.tensor([12.0, 0.0]),
            heading=0.0,
            size_lwh=torch.tensor([4.0, 2.0, 1.6]),
        )

        pixels, visible = project_points(
            points_world=corners,
            ego_xy=torch.zeros(2),
            ego_heading=0.0,
            camera=camera,
        )

        self.assertTrue(bool(visible.all()))
        self.assertTrue(torch.all(pixels[:, 0] >= 0))
        self.assertTrue(torch.all(pixels[:, 0] <= camera.width))
        self.assertTrue(torch.all(pixels[:, 1] >= 0))
        self.assertTrue(torch.all(pixels[:, 1] <= camera.height))

    def test_front_camera_rejects_points_behind_camera(self):
        from scripts.render_multicamera_layout import default_camera_specs, project_points

        camera = default_camera_specs()[0]
        pixels, visible = project_points(
            points_world=torch.tensor([[-5.0, 0.0, 0.0]]),
            ego_xy=torch.zeros(2),
            ego_heading=0.0,
            camera=camera,
        )

        self.assertEqual(tuple(pixels.shape), (1, 2))
        self.assertFalse(bool(visible.item()))


class MultiCameraLayoutRenderTest(unittest.TestCase):
    def test_default_render_does_not_draw_map_polylines(self):
        from scripts.render_multicamera_layout import (
            build_synthetic_layout_scene,
            default_camera_specs,
            render_camera_layout,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("scripts.render_multicamera_layout._draw_projected_polyline") as draw_polyline:
                render_camera_layout(
                    scene=build_synthetic_layout_scene(num_frames=1),
                    camera=default_camera_specs()[0],
                    frame_index=0,
                    output_path=Path(tmpdir) / "camera.png",
                )

        draw_polyline.assert_not_called()

    def test_debug_render_can_draw_map_polylines(self):
        from scripts.render_multicamera_layout import (
            build_synthetic_layout_scene,
            default_camera_specs,
            render_camera_layout,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("scripts.render_multicamera_layout._draw_projected_polyline") as draw_polyline:
                render_camera_layout(
                    scene=build_synthetic_layout_scene(num_frames=1),
                    camera=default_camera_specs()[0],
                    frame_index=0,
                    output_path=Path(tmpdir) / "camera.png",
                    draw_map_polylines=True,
                )

        self.assertGreater(draw_polyline.call_count, 0)

    def test_render_synthetic_layout_writes_camera_pngs_and_manifest(self):
        from scripts.render_multicamera_layout import (
            build_synthetic_layout_scene,
            default_camera_specs,
            render_layout_sequence,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            summary = render_layout_sequence(
                scene=build_synthetic_layout_scene(num_frames=2),
                output_dir=output_dir,
                cameras=default_camera_specs()[:2],
                frame_indices=[0, 1],
            )

            self.assertEqual(summary["num_frames"], 2)
            self.assertEqual(summary["num_cameras"], 2)
            manifest_path = output_dir / "manifest.json"
            self.assertTrue(manifest_path.exists())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["camera_names"], ["CAM_FRONT", "CAM_FRONT_LEFT"])
            for rel_path in manifest["frames"][0]["images"].values():
                image_path = output_dir / rel_path
                self.assertTrue(image_path.exists())
                with Image.open(image_path) as image:
                    self.assertEqual(image.size, (640, 360))


if __name__ == "__main__":
    unittest.main()
