"""Lazy WiLoR model runtime used by the live USB input path."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
import sys
from typing import Iterator

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WILOR_ROOT = PROJECT_ROOT / "third_party" / "WiLoR"


@dataclass(frozen=True)
class WilorAssets:
    root: Path
    detector_checkpoint: Path
    wilor_checkpoint: Path
    model_config: Path
    mano_model: Path
    mano_mean_params: Path


@dataclass(frozen=True)
class WilorDetection:
    """One WiLoR hand detection retained by the live input and tuning recorder."""

    is_right: bool
    bbox_xyxy: np.ndarray
    joints_mano: np.ndarray
    vertices_mano: np.ndarray
    global_orient: np.ndarray
    hand_pose: np.ndarray
    betas: np.ndarray
    pred_cam: np.ndarray
    camera_translation: np.ndarray


def validate_wilor_assets(wilor_root: Path = DEFAULT_WILOR_ROOT) -> WilorAssets:
    root = wilor_root.expanduser().resolve()
    required_paths = {
        "WiLoR source": root / "wilor" / "models" / "__init__.py",
        "detector.pt": root / "pretrained_models" / "detector.pt",
        "wilor_final.ckpt": root / "pretrained_models" / "wilor_final.ckpt",
        "model_config.yaml": root / "pretrained_models" / "model_config.yaml",
        "MANO_RIGHT.pkl": root / "mano_data" / "MANO_RIGHT.pkl",
        "mano_mean_params.npz": root / "mano_data" / "mano_mean_params.npz",
    }
    for label, path in required_paths.items():
        if not path.is_file():
            raise FileNotFoundError(
                f"missing {label}: {path}. Install the WiLoR assets documented in README.md"
            )
    return WilorAssets(
        root=root,
        detector_checkpoint=required_paths["detector.pt"],
        wilor_checkpoint=required_paths["wilor_final.ckpt"],
        model_config=required_paths["model_config.yaml"],
        mano_model=required_paths["MANO_RIGHT.pkl"],
        mano_mean_params=required_paths["mano_mean_params.npz"],
    )


@contextmanager
def working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


@contextmanager
def trusted_checkpoint_load(torch_module) -> Iterator[None]:
    """Load verified legacy WiLoR checkpoints under its required policy."""
    original_load = torch_module.load

    def compatible_load(*arguments, **keyword_arguments):
        keyword_arguments.setdefault("weights_only", False)
        return original_load(*arguments, **keyword_arguments)

    torch_module.load = compatible_load
    try:
        yield
    finally:
        torch_module.load = original_load


class WiLoRRunner:
    """Lazy bridge to the untouched upstream WiLoR inference code."""

    def __init__(
        self,
        assets: WilorAssets,
        device_name: str,
        batch_size: int,
        confidence: float,
        fast: bool,
    ) -> None:
        if str(assets.root) not in sys.path:
            sys.path.insert(0, str(assets.root))

        import torch
        from ultralytics import YOLO
        from wilor.datasets.vitdet_dataset import ViTDetDataset
        from wilor.models import load_wilor
        from wilor.utils import recursive_to
        from wilor.utils.renderer import cam_crop_to_full

        self.torch = torch
        self.ViTDetDataset = ViTDetDataset
        self.recursive_to = recursive_to
        self.cam_crop_to_full = cam_crop_to_full
        self.batch_size = batch_size
        self.confidence = confidence
        self.fast = fast
        self.device = self._resolve_device(device_name)

        with trusted_checkpoint_load(torch):
            with working_directory(assets.root):
                self.model, self.model_cfg = load_wilor(
                    checkpoint_path=str(assets.wilor_checkpoint),
                    cfg_path=str(assets.model_config),
                )
            self.detector = YOLO(str(assets.detector_checkpoint))
        if fast:
            if self.device.type == "cpu":
                raise ValueError("WiLoR fast mode requires CUDA")
            torch.set_float32_matmul_precision("high")
            self.model = self.model.half()
            self.model.backbone.skip_blocks = True
        self.model = self.model.to(self.device).eval()
        self.detector = self.detector.to(self.device)
        self.faces = np.asarray(self.model.mano.faces, dtype=np.int32)

    def _resolve_device(self, device_name: str):
        cuda_available = self.torch.cuda.is_available()
        if device_name == "cuda" and not cuda_available:
            raise RuntimeError("--device cuda was requested, but Torch reports no accelerator")
        if device_name == "cpu" or not cuda_available:
            return self.torch.device("cpu")
        return self.torch.device("cuda")

    def infer(self, image_bgr: np.ndarray) -> list[WilorDetection]:
        detector_output = self.detector(image_bgr, conf=self.confidence, verbose=False)[0]
        if len(detector_output.boxes) == 0:
            return []

        boxes = detector_output.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
        right = detector_output.boxes.cls.detach().cpu().numpy().astype(np.float32)
        dataset = self.ViTDetDataset(
            self.model_cfg, image_bgr, boxes, right, rescale_factor=2.0, fp16=self.fast
        )
        loader = self.torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False, num_workers=0
        )
        detections: list[WilorDetection] = []
        for batch in loader:
            batch = self.recursive_to(batch, self.device)
            with self.torch.no_grad():
                output = self.model(batch)

            multiplier = 2 * batch["right"] - 1
            pred_cam = output["pred_cam"].clone()
            pred_cam[:, 1] *= multiplier
            image_size = batch["img_size"].float()
            scaled_focal = (
                self.model_cfg.EXTRA.FOCAL_LENGTH / self.model_cfg.MODEL.IMAGE_SIZE
                * image_size.max()
            )
            camera_translation = self.cam_crop_to_full(
                pred_cam,
                batch["box_center"].float(),
                batch["box_size"].float(),
                image_size,
                scaled_focal,
            )
            parameters = output["pred_mano_params"]
            for index in range(batch["img"].shape[0]):
                is_right = bool(batch["right"][index].detach().cpu().item())
                mirror = 1.0 if is_right else -1.0
                joints = output["pred_keypoints_3d"][index].detach().cpu().numpy()
                vertices = output["pred_vertices"][index].detach().cpu().numpy()
                joints[:, 0] *= mirror
                vertices[:, 0] *= mirror
                detections.append(
                    WilorDetection(
                        is_right=is_right,
                        bbox_xyxy=boxes[len(detections)],
                        joints_mano=joints,
                        vertices_mano=vertices,
                        global_orient=parameters["global_orient"][index]
                        .detach().cpu().numpy().reshape(3, 3),
                        hand_pose=parameters["hand_pose"][index]
                        .detach().cpu().numpy().reshape(15, 3, 3),
                        betas=parameters["betas"][index].detach().cpu().numpy().reshape(10),
                        pred_cam=pred_cam[index].detach().cpu().numpy(),
                        camera_translation=camera_translation[index].detach().cpu().numpy(),
                    )
                )
        return detections
