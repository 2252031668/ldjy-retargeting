#!/usr/bin/env python3
"""Standalone MANO hand viewer with real-time parameter sliders.

Displays the MANO hand model in MuJoCo with a semi-transparent mesh
and 21 keypoints as colored spheres. Provides a PySide6 GUI with sliders
for all MANO parameters: betas (10), hand_pose (15x3=45), global_orient (3),
translation (3), and scale (1).

Usage:
    uv run --extra wilor --extra gui --extra tuning python example/mano_viewer.py
"""

import sys
import tempfile
import shutil
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer

from ldjy_retargeting.mano_conventions import (
    MANO_HAND_POSE_FINGER_ORDER,
    mano_native_to_mediapipe_keypoints,
)

try:
    import smplx
    import torch
    from smplx.lbs import batch_rodrigues
except ImportError:
    print("This script requires smplx and torch.")
    print("Install with: uv sync --extra wilor")
    sys.exit(1)

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSlider, QDoubleSpinBox, QLabel, QGroupBox, QScrollArea, QPushButton,
    QSizePolicy,
)
from PySide6.QtCore import Qt, QTimer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANO_MODEL_PATH = PROJECT_ROOT / "third_party" / "WiLoR" / "mano_data" / "MANO_RIGHT.pkl"

# 15 hand_pose joints use MANO's native kinematic-tree order.
FINGER_JOINT_NAMES = {
    "Thumb":  ["CMC", "MCP", "IP"],
    "Index":  ["MCP", "PIP", "DIP"],
    "Middle": ["MCP", "PIP", "DIP"],
    "Ring":   ["MCP", "PIP", "DIP"],
    "Pinky":  ["MCP", "PIP", "DIP"],
}
FINGER_ORDER = list(MANO_HAND_POSE_FINGER_ORDER)

# 21 public keypoints use MediaPipe/WiLoR ordering.
JOINT_FINGER = [
    "Wrist",
    "Thumb", "Thumb", "Thumb", "Thumb",
    "Index", "Index", "Index", "Index",
    "Middle", "Middle", "Middle", "Middle",
    "Ring", "Ring", "Ring", "Ring",
    "Pinky", "Pinky", "Pinky", "Pinky",
]

FINGER_COLORS = {
    "Wrist":  np.array([0.5, 0.5, 0.5, 1.0]),
    "Thumb":  np.array([1.0, 0.3, 0.3, 1.0]),
    "Index":  np.array([0.3, 1.0, 0.3, 1.0]),
    "Middle": np.array([0.3, 0.5, 1.0, 1.0]),
    "Ring":   np.array([1.0, 1.0, 0.3, 1.0]),
    "Pinky":  np.array([1.0, 0.3, 1.0, 1.0]),
}

# Tip joint vertex indices in the MANO mesh
TIP_VERTEX_IDS = {
    "Thumb":  744,
    "Index":  320,
    "Middle": 443,
    "Ring":   554,
    "Pinky":  671,
}

# Finger-pad vertex indices (fingerprint-center vert of each distal segment,
# located on v_template via palmar-side normal filter + DIP proximity).
# All have y<0 (palmar side) and strong palmar normals.
# Calibrated via the interactive pad-selection system in MANOViewer.
# Some fingers use 3-point centroids (PAD_3PT_VERTEX_IDS) for better surface fit.
PAD_VERTEX_IDS = {
    "Thumb":  763,
    "Index":  355,
    "Middle": 438,
    "Ring":   573,
    "Pinky":  690,
}

# 3-point centroid defaults for fingers that use triangle-center pad.
# Each key maps to [vid_a, vid_b, vid_c] defining the pad surface triangle.
PAD_3PT_VERTEX_IDS = {
    "Index":  [328, 343, 350],
    "Middle": [438, 455, 439],
}

PAD_COLOR   = np.array([1.0, 0.5, 0.0, 1.0])   # orange pad keypoints
ARROW_COLOR = np.array([1.0, 0.9, 0.2, 1.0])   # yellow pad-normal vectors
ARROW_RADIUS = 0.0008                           # 0.8mm capsule radius
DEFAULT_ARROW_LENGTH = 0.01                     # 1cm

# --- Pad-selection (interactive) constants ---
CAND_RADIUS = 0.0015                             # 1.5mm candidate sphere (clickable)
CAND_COLOR_DEFAULT = np.array([1.0, 1.0, 1.0, 0.6])   # semi-transparent white
CAND_COLOR_SELECT = np.array([1.0, 0.2, 0.2, 1.0])    # red = currently selected
CAND_COLOR_A = np.array([0.0, 1.0, 1.0, 1.0])         # cyan = A marker
CAND_COLOR_B = np.array([1.0, 0.0, 1.0, 1.0])         # magenta = B marker
CAND_COLOR_C = np.array([0.0, 1.0, 0.0, 1.0])         # green = C marker
# Geodesic radius from tip to find candidates per finger (meters)
CAND_GEO_RADIUS = 0.022
# Max candidates per finger
MAX_CAND_PER_FINGER = 28
# Palmar-normal dot threshold for candidates (must face palm side)
CAND_PALMAR_DOT = 0.05
# MCP approximation for each finger (v_template coordinates) — for axis/cylinder
CAND_MCP_APPROX = {
    "Index":  [-0.035,  0.0,   0.025],
    "Middle": [-0.040,  0.005, -0.005],
    "Ring":   [-0.038, -0.005, -0.030],
    "Pinky":  [-0.030, -0.010, -0.055],
    "Thumb":  [ 0.020, -0.020,  0.060],
}
MCP_CAND_RADIUS_M = 0.006  # find actual MCP vertex within 6mm of the above
# Axial range from tip along finger axis within which candidates are selected
CAND_AXIAL_BACK_M = 0.034   # 34mm back from tip (covers Middle/Ring auto pads)
CAND_PERP_RADIUS_M = 0.009  # 9mm around the axis (finger width approx)

PAD_FINGER_KEYS = {
    "1": "Thumb", "2": "Index", "3": "Middle", "4": "Ring", "5": "Pinky",
}
ABC_KEYS = {"A": 0, "B": 1, "C": 2}

TIP_ORDER = ["Thumb", "Index", "Middle", "Ring", "Pinky"]


class MANOModel:
    """Wrapper around smplx.MANOLayer with 21-joint output."""

    def __init__(self, model_path: str):
        self.mano = smplx.MANOLayer(
            model_path=model_path,
            gender="neutral",
            num_betas=10,
        )
        self.faces = self.mano.faces  # (F, 3) int32
        self.n_verts = int(self.mano.v_template.shape[0])  # 778
        self.n_faces = int(self.faces.shape[0])  # 1538

        # Default parameters
        self.default_betas = np.zeros(10, dtype=np.float64)
        self.default_hand_pose = np.zeros((15, 3), dtype=np.float64)
        self.default_global_orient = np.zeros(3, dtype=np.float64)
        self.default_translation = np.zeros(3, dtype=np.float64)
        self.default_scale = 0.001  # mm -> m

        # Pre-compute default output for mesh initialization
        self._default_vertices, self._default_joints, self._default_pads = self.compute(
            self.default_betas,
            self.default_hand_pose,
            self.default_global_orient,
            self.default_translation,
            1.0,
        )

        # Compute candidate vertex IDs per finger (on v_template rest pose)
        self._candidates_per_finger = self._compute_candidate_vertices()
        self._all_candidate_vids = np.concatenate(
            list(self._candidates_per_finger.values())
        ).astype(np.int64)
        # (cand_global_idx -> (finger_idx, finger_cand_idx)) for reverse lookup
        self._cand_to_finger = {}
        for fi, name in enumerate(TIP_ORDER):
            for ci, vid in enumerate(self._candidates_per_finger[name]):
                self._cand_to_finger[int(vid)] = (fi, ci)

    @property
    def default_vertices(self):
        return self._default_vertices

    @property
    def default_joints(self):
        return self._default_joints

    @property
    def default_pads(self):
        return self._default_pads

    @property
    def all_candidate_vids(self):
        """Flat array of candidate vertex indices (on original 778-vert mesh)."""
        return self._all_candidate_vids

    @property
    def candidates_per_finger(self):
        """Dict TIP_ORDER -> (n_candidates_in_finger,) int64 vertex ids."""
        return self._candidates_per_finger

    @property
    def n_candidates_total(self):
        return int(len(self._all_candidate_vids))

    def _compute_candidate_vertices(self):
        """Compute candidate mesh-vertex indices per finger (on v_template).

        For each finger:
          1. Use tip vertex + MCP (actual vert nearest to CAND_MCP_APPROX)
             to get finger axis.
          2. Filter mesh verts: in cylinder around axis (from tip backward
             CAND_AXIAL_BACK_M meters, radius CAND_PERP_RADIUS_M).
          3. Keep only palmar-side verts (normal·palm_normal > CAND_PALMAR_DOT).
          4. Sort candidates along the finger axis (by distance from tip) and
             select up to MAX_CAND_PER_FINGER evenly distributed.
        """
        vtpl = np.asarray(self.mano.v_template).astype(np.float64)
        faces = np.asarray(self.faces).astype(np.int64)
        n = vtpl.shape[0]

        # rest-pose vertex normals
        normals = np.zeros((n, 3))
        for f in faces:
            v0, v1, v2 = vtpl[f]
            fn = np.cross(v1 - v0, v2 - v0)
            nn = np.linalg.norm(fn)
            if nn > 1e-12:
                fn = fn / nn
            normals[f] += fn
        for i in range(n):
            nn = np.linalg.norm(normals[i])
            if nn > 1e-12:
                normals[i] /= nn

        # palm normal (palm region verts near wrist within 5cm)
        wrist = vtpl[0]
        palm_region = np.linalg.norm(vtpl - wrist, axis=1) < 0.05
        palm_n = normals[palm_region].mean(0)
        palm_n /= (np.linalg.norm(palm_n) + 1e-12)

        result = {}
        for name in TIP_ORDER:
            tip_vid = TIP_VERTEX_IDS[name]
            tip = vtpl[tip_vid]
            mcp_approx = np.asarray(CAND_MCP_APPROX[name], dtype=np.float64)
            mcp_vid = int(np.argmin(np.linalg.norm(vtpl - mcp_approx, axis=1)))
            mcp = vtpl[mcp_vid]

            finger = tip - mcp
            finger_hat = finger / (np.linalg.norm(finger) + 1e-12)

            # Cylinder around axis
            rel = vtpl - tip
            axial = rel @ finger_hat  # negative = toward MCP
            perp = rel - np.outer(axial, finger_hat)
            perp_norm = np.linalg.norm(perp, axis=1)
            mask = (perp_norm < CAND_PERP_RADIUS_M) \
                & (axial > -CAND_AXIAL_BACK_M) & (axial < 0.004)

            # Palmar side
            dots = normals @ palm_n
            mask = mask & (dots > CAND_PALMAR_DOT)

            # Remove tip vertex itself (dorsal side normally)
            # Keep tip only if it happens to pass palmar filter
            ids = np.where(mask)[0]
            if len(ids) == 0:
                # Fallback: relax palmar filter to > 0
                mask2 = (perp_norm < CAND_PERP_RADIUS_M) \
                    & (axial > -CAND_AXIAL_BACK_M) & (axial < 0.004) \
                    & (dots > 0.0)
                ids = np.where(mask2)[0]

            # Sort by axial distance from tip (ascending)
            order = np.argsort(axial[ids])
            ids_sorted = ids[order]

            # Pick evenly-spaced subset up to MAX_CAND_PER_FINGER
            if len(ids_sorted) > MAX_CAND_PER_FINGER:
                picks = np.linspace(0, len(ids_sorted) - 1, MAX_CAND_PER_FINGER)
                picks = picks.round().astype(int)
                ids_picked = np.unique(ids_sorted[picks])
            else:
                ids_picked = ids_sorted

            # Guarantee the current auto pad vertex for this finger is in the
            # candidate list, regardless of filters — users may want to keep
            # the auto-selected point as a clickable baseline.
            pad_vid = int(PAD_VERTEX_IDS.get(name, -1))
            if pad_vid >= 0 and pad_vid not in ids_picked:
                ids_picked = np.concatenate(
                    [ids_picked, np.array([pad_vid], dtype=np.int64)]
                )

            result[name] = ids_picked.astype(np.int64)

        return result

    def compute(self, betas, hand_pose, global_orient, translation, scale):
        """Compute MANO vertices, 21 joints, and 5 finger-pad points.

        Args:
            betas: (10,) shape PCA coefficients
            hand_pose: (15, 3) axis-angle rotations
            global_orient: (3,) axis-angle rotation of the wrist
            translation: (3,) world-space translation in meters
            scale: float, additional scale multiplier

        Returns:
            vertices: (778, 3) in meters
            joints: (21, 3) in meters
            pads: (5, 3) finger-pad surface points in meters
        """
        with torch.no_grad():
            betas_t = torch.tensor(betas, dtype=torch.float32).unsqueeze(0)

            # Convert axis-angle to rotation matrices (batch_rodrigues expects (N, 3))
            hand_pose_aa = torch.tensor(hand_pose, dtype=torch.float32)
            hand_pose_rot = batch_rodrigues(hand_pose_aa.reshape(-1, 3))
            hand_pose_rot = hand_pose_rot.reshape(1, 15, 3, 3)

            global_orient_aa = torch.tensor(global_orient, dtype=torch.float32).reshape(1, 3)
            global_orient_rot = batch_rodrigues(global_orient_aa).unsqueeze(0)

            transl_t = torch.tensor(translation, dtype=torch.float32).unsqueeze(0)
            scale_val = self.default_scale * float(scale)
            scale_t = torch.tensor([scale_val], dtype=torch.float32)

            output = self.mano(
                betas=betas_t,
                hand_pose=hand_pose_rot,
                global_orient=global_orient_rot,
                transl=transl_t,
                scale=scale_t,
            )

        vertices = output.vertices[0].cpu().numpy().astype(np.float64)
        base_joints = output.joints[0].cpu().numpy().astype(np.float64)  # (16, 3)

        fingertip_points = np.asarray(
            [vertices[TIP_VERTEX_IDS[finger_name]] for finger_name in TIP_ORDER]
        )
        joints = mano_native_to_mediapipe_keypoints(base_joints, fingertip_points)

        # 5 finger-pad surface points (deformed with the mesh)
        pads = []
        for name in TIP_ORDER:
            pt3 = PAD_3PT_VERTEX_IDS.get(name)
            if pt3 is not None:
                # 3-point centroid
                vs = [vertices[int(v)] for v in pt3]
                pads.append(np.mean(vs, axis=0))
            else:
                pads.append(vertices[int(PAD_VERTEX_IDS[name])])
        pads = np.array(pads, dtype=np.float64)

        return vertices, joints, pads


class MANOViewer:
    """MuJoCo-based passive viewer for the MANO hand model."""

    KEYPOINT_RADIUS = 0.003

    def __init__(self, mano_model: MANOModel):
        self.mano_model = mano_model
        self.show_keypoints = True
        self.mesh_rgba = np.array([0.3, 0.6, 1.0, 0.5], dtype=np.float64)
        self._tmpdir = None
        self._kp_radius = self.KEYPOINT_RADIUS

        # Build expanded mesh for flat (polyhedron) shading:
        # each face gets 3 unique vertices so per-vertex normals = face normals.
        # n_expanded = n_faces * 3 = 1538 * 3 = 4614
        orig_faces = mano_model.faces  # (1538, 3) indices into 778 verts
        self._expanded_to_orig = orig_faces.reshape(-1).astype(np.int64)  # (4614,)
        self._expanded_faces = np.arange(
            len(self._expanded_to_orig), dtype=np.int32
        ).reshape(-1, 3)  # (1538, 3) referencing 0..4613
        self._n_expanded_verts = len(self._expanded_to_orig)

        # Create MuJoCo model (writes temp OBJ + MJCF with keypoint geoms)
        self.model = self._create_model(
            mano_model.default_vertices, mano_model.default_joints,
            mano_model.default_pads,
        )
        self.data = mujoco.MjData(self.model)

        # Read MuJoCo's mesh reference transform (applied during rendering).
        # rendered_vertex = mesh_scale * (R @ mesh_vert) + mesh_pos
        # We store T_ref_inv so we can write vertices that, after T_ref is
        # applied by the renderer, end up at the desired MANO-world position.
        self._mesh_pos = self.model.mesh_pos[0].copy()
        self._mesh_quat = self.model.mesh_quat[0].copy()
        self._mesh_scale = self.model.mesh_scale[0].copy()

        # Compute rotation matrix from quaternion [w, x, y, z]
        R_flat = np.zeros(9, dtype=np.float64)
        mujoco.mju_quat2Mat(R_flat, self._mesh_quat)
        self._mesh_R = R_flat.reshape(3, 3)

        # Pre-compute inverse scale (element-wise)
        self._mesh_scale_inv = np.where(
            np.abs(self._mesh_scale) > 1e-10,
            1.0 / self._mesh_scale,
            0.0,
        )

        # Pre-compute keypoint body and geom IDs for fast lookup
        self._kp_body_ids = []
        self._kp_geom_ids = []
        self._kp_colors = []
        for i in range(21):
            bid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, f"kp_{i}"
            )
            gid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, f"kp_{i}_geom"
            )
            self._kp_body_ids.append(bid)
            self._kp_geom_ids.append(gid)
            self._kp_colors.append(FINGER_COLORS[JOINT_FINGER[i]].copy())

        # Pad-keypoint and pad-normal-arrow body/geom IDs (5 each)
        self._pad_body_ids = []
        self._pad_geom_ids = []
        self._arrow_body_ids = []
        self._arrow_geom_ids = []
        for i in range(5):
            pbid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"pad_{i}")
            pgid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"pad_{i}_geom")
            abid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"arrow_{i}")
            agid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"arrow_{i}_geom")
            self._pad_body_ids.append(pbid)
            self._pad_geom_ids.append(pgid)
            self._arrow_body_ids.append(abid)
            self._arrow_geom_ids.append(agid)

        # Pre-compute adjacent-face indices for each pad (on original
        # 778-vert mesh) so we can average face normals at runtime.
        # Supports both single-vertex and 3-point centroid defaults.
        orig_faces = mano_model.faces.astype(np.int64)  # (1538, 3)
        self._pad_vids = np.array(
            [PAD_VERTEX_IDS[name] for name in TIP_ORDER], dtype=np.int64
        )
        self._pad_adj_faces = []
        for i, name in enumerate(TIP_ORDER):
            pt3 = PAD_3PT_VERTEX_IDS.get(name)
            if pt3 is not None:
                # 3-point centroid: union of adjacent faces from all 3 verts
                adj_list = [np.where((orig_faces == int(v)).any(axis=1))[0]
                            for v in pt3]
                adj = np.unique(np.concatenate(adj_list))
            else:
                # Single vertex
                vid = int(PAD_VERTEX_IDS[name])
                adj = np.where((orig_faces == vid).any(axis=1))[0]
            self._pad_adj_faces.append(adj)

        # Arrow (pad-normal vector) length, controlled by a GUI slider
        self._arrow_length = DEFAULT_ARROW_LENGTH

        # Pad keypoint visibility and radius (controlled by GUI)
        self._pad_visible = True
        self._pad_radius = self.KEYPOINT_RADIUS

        # ======================================================
        # Interactive pad-selection state machine
        # ======================================================
        n_cand = mano_model.n_candidates_total
        # Candidate body/geom ids (one body per candidate sphere)
        self._cand_body_ids = []
        self._cand_geom_ids = []
        for i in range(n_cand):
            cbid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, f"cand_{i}"
            )
            cgid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, f"cand_{i}_geom"
            )
            self._cand_body_ids.append(cbid)
            self._cand_geom_ids.append(cgid)

        # Candidate vertex -> MANO-world pos computed each frame
        # (derived from deformed mesh at the candidate vid)
        self._cand_vids = mano_model.all_candidate_vids  # (n_cand,)

        # Selection state:
        #   _sel_cand: current candidate id highlighted (red) or -1
        #   _abc: [vid, vid, vid] (or -1 for unused) - the three marked ABC vertices
        #   _custom_pad_vids: {"Thumb": vid|None, ...} — per-finger selected vertex
        #                     or None if "use 3-point center"
        #   _custom_pad_pos: if 3-point center used for a finger, store position here
        self._sel_cand = -1
        self._prev_perturb_select = 0   # last viewer.perturb.select value
        self._abc = [-1, -1, -1]            # mesh vertex ids (in 778-vid space)
        self._custom_pad_vids = {k: None for k in TIP_ORDER}
        self._custom_pad_3pt_vertices = {k: None for k in TIP_ORDER}  # list of 3 vids
        self._candidates_visible = False  # candidates hidden by default
        self._last_vertices = mano_model.default_vertices.copy()
        # Status text (shown in PySide6 UI)
        self._selection_status = "双击候选球选中，然后按 1..5 设置指腹点"

        # Initialize candidate positions + selection colors
        self._set_candidate_positions_and_colors(
            mano_model.default_vertices, pad_ids_overridden=None
        )

        # Hide candidates by default
        for ci in range(len(self._cand_body_ids)):
            cgid = self._cand_geom_ids[ci]
            if cgid >= 0:
                self.model.geom_rgba[cgid][3] = 0.0

        # Apply initial mesh and keypoint positions
        self._apply_mesh_and_joints(
            mano_model.default_vertices, mano_model.default_joints,
            mano_model.default_pads,
        )

        # Initialize forward dynamics so data.geom_xpos is populated
        mujoco.mj_forward(self.model, self.data)

        # Launch viewer with keyboard callback.
        # Mouse picking uses MuJoCo's built-in double-click selection
        # (viewer.perturb.select), polled in refresh().
        self.viewer = mujoco.viewer.launch_passive(
            self.model, self.data,
            key_callback=self._on_key_press,
        )

        # Setup camera for a lateral view of the hand
        self.viewer.cam.azimuth = 180
        self.viewer.cam.elevation = -25
        self.viewer.cam.distance = 0.6
        self.viewer.cam.lookat[:] = [0.0, 0.08, 0.0]

        # Print interactive pad-selection key bindings
        self._print_keybindings()

    @staticmethod
    def _print_keybindings():
        """Print key binding instructions to stdout on startup."""
        print("=" * 60)
        print("  交互式指腹标定 — 按键说明 (Interactive Pad Selection)")
        print("=" * 60)
        print("  [选点] 在 MuJoCo 窗口中:")
        print("    双击候选球       → 选中该球 (高亮为红色)")
        print("    Esc              → 清除当前选中")
        print()
        print("  [标定] 选好点后按以下键:")
        print("    1  →  Thumb  (拇指)   设为当前选中点")
        print("    2  →  Index  (食指)")
        print("    3  →  Middle (中指)")
        print("    4  →  Ring   (无名指)")
        print("    5  →  Pinky  (小指)")
        print()
        print("  [三点重心模式] 如果指纹中心不在单个顶点上:")
        print("    A / B / C    将当前选中点标记为三角形的三个顶点")
        print("    再按 1..5    把 A/B/C 三点重心设为该手指的 pad")
        print()
        print("  [其他]")
        print("    S            导出 PAD_VERTEX_IDS 字典 (打印+剪贴板)")
        print("    R            重置所有自定义选点 (回到自动)")
        print("    ! @ # $ %   (Shift+1..5) 清除单个手指的自定义选点")
        print()
        print("  候选球颜色:")
        print("    白色半透明 = 普通 | 红色 = 当前选中 | 橙色 = 已设为 pad")
        print("    青色=A  品红=B  绿色=C")
        print("=" * 60)
        print()

    def _check_perturb_select(self):
        """Poll viewer.perturb.select to detect double-click on a candidate.

        MuJoCo's passive viewer sets perturb.select to the body id of the
        geom under the cursor on double-click. We map that body id back to
        a candidate index and update _sel_cand accordingly.
        """
        # Only allow picking when candidates are visible
        if not self._candidates_visible:
            return

        cur = int(self.viewer.perturb.select)
        if cur == self._prev_perturb_select:
            return  # no change

        self._prev_perturb_select = cur

        # cur == 0 means "no selection" (world body) — don't clear our
        # selection, the user may have just clicked empty space.
        if cur == 0:
            return

        # Check if the selected body is one of our candidate bodies
        for ci, cbid in enumerate(self._cand_body_ids):
            if cbid == cur:
                self._sel_cand = ci
                vid = int(self._cand_vids[ci])
                self._selection_status = (
                    f"[PICK] candidate #{ci} (v{vid})"
                )
                # Disable perturbation so the sphere doesn't get dragged
                self.viewer.perturb.active = 0
                self.viewer.perturb.active2 = 0
                return

    def _current_selected_vid(self):
        """Return currently selected 778-mesh vertex id or -1."""
        if self._sel_cand >= 0 and self._sel_cand < len(self._cand_vids):
            return int(self._cand_vids[self._sel_cand])
        return -1

    def _on_key_press(self, code: int):
        """Keyboard callback for MuJoCo viewer.

        Key mappings:
          1..5  : Save selected candidate (or A/B/C center) as pad for
                  Thumb/Index/Middle/Ring/Pinky.
          A, B, C : Mark current selection as ABC points (for 3-point center).
          Shift+1..5 : Clear custom pad for that finger (back to auto).
          S     : Export current PAD_VERTEX_IDS dictionary (print+clipboard).
          R     : Reset all custom pad selections (back to auto).
          Esc   : Clear current selection.
        """
        try:
            ch = chr(code & 0x7F) if (code & 0x7F) >= 32 else None
        except (ValueError, OverflowError):
            ch = None

        # --- Finger pad keys 1..5 (only when candidates are visible) ---
        if ch and ch in PAD_FINGER_KEYS:
            if not self._candidates_visible:
                self._selection_status = (
                    "请先点击 'Show Candidates' 开启候选球模式"
                )
                return
            fname = PAD_FINGER_KEYS[ch]
            # If ABC are all set (3 points), use 3-point center mode.
            n_abc = sum(1 for v in self._abc if v >= 0)
            sel_vid = self._current_selected_vid()
            if n_abc == 3:
                # Use ABC vertex triplet as the 3-point center
                self._custom_pad_3pt_vertices[fname] = list(self._abc)
                self._custom_pad_vids[fname] = None
                self._selection_status = (
                    f"[OK] {fname} pad = centroid of A(v{self._abc[0]}) "
                    f"B(v{self._abc[1]}) C(v{self._abc[2]})"
                )
                print(f"[pad_select] {fname} = triangle center of v{self._abc}")
            elif sel_vid >= 0:
                self._custom_pad_vids[fname] = sel_vid
                self._custom_pad_3pt_vertices[fname] = None
                self._selection_status = f"[OK] {fname} pad = single vertex v{sel_vid}"
                print(f"[pad_select] {fname} = v{sel_vid}")
            else:
                self._selection_status = (
                    "[WARN] 先双击候选球选中，或用 A/B/C 标记三个点"
                )
            return

        # --- ABC marker keys (only when candidates are visible) ---
        if ch and ch.upper() in ABC_KEYS:
            if not self._candidates_visible:
                self._selection_status = (
                    "请先点击 'Show Candidates' 开启候选球模式"
                )
                return
            abi = ABC_KEYS[ch.upper()]
            is_shift = (code & 0x8000) != 0 or ch.isupper()  # heuristic shift
            sel_vid = self._current_selected_vid()
            if sel_vid >= 0:
                self._abc[abi] = sel_vid
                mark = ["A", "B", "C"][abi]
                self._selection_status = f"[MARK] {mark} = v{sel_vid}"
                print(f"[pad_select] {mark} = v{sel_vid}")
            else:
                self._selection_status = "[WARN] Select a candidate first before marking."
            return

        # --- Shift 1..5: Clear custom selection for specific finger ---
        if ch and ch in "!@#$%":  # Shift+1..5 on US keyboard
            idx = {"!": 0, "@": 1, "#": 2, "$": 3, "%": 4}[ch]
            fname = TIP_ORDER[idx]
            self._custom_pad_vids[fname] = None
            self._custom_pad_3pt_vertices[fname] = None
            self._selection_status = f"[RESET] {fname} back to auto."
            return

        # --- S = export ---
        if ch and ch.upper() == "S":
            lines = ["PAD_VERTEX_IDS = {"]
            for fname in TIP_ORDER:
                vid = self._custom_pad_vids.get(fname)
                if vid is not None:
                    lines.append(f'    "{fname}": {int(vid)},')
                elif self._custom_pad_3pt_vertices.get(fname) is not None:
                    v3 = self._custom_pad_3pt_vertices[fname]
                    v3s = [str(int(v)) for v in v3]
                    lines.append(f'    # "{fname}": 3pt center v{v3s},')
                else:
                    # Use current auto
                    lines.append(
                        f'    "{fname}": {PAD_VERTEX_IDS[fname]},  # auto'
                    )
            lines.append("}")
            block = "\n".join(lines)
            self._selection_status = f"[EXPORT] see stdout\n{block}"
            print("=" * 40)
            print(block)
            print("=" * 40)
            try:
                import subprocess
                subprocess.run(
                    ["xclip", "-selection", "clipboard"],
                    input=block, text=True, check=False,
                    timeout=3,
                )
            except Exception:
                pass
            return

        # --- R = reset all ---
        if ch and ch.upper() == "R":
            for fname in TIP_ORDER:
                self._custom_pad_vids[fname] = None
                self._custom_pad_3pt_vertices[fname] = None
            self._abc = [-1, -1, -1]
            self._sel_cand = -1
            self._selection_status = "[RESET] All pads back to auto; ABC cleared."
            return

        # --- Esc = clear current sel ---
        if code == 0x1000000 or (ch is not None and ord(ch) == 27):
            self._sel_cand = -1
            self._selection_status = "Click a candidate sphere to select it."
            return

    def _to_mesh_vert_space(self, points):
        """Transform points from MANO-world to mesh_vert storage space.

        The renderer applies: rendered = scale * (R @ mesh_vert) + pos
        So mesh_vert = R^T @ ((point - pos) / scale)
        In row-vector form: mesh_vert = (point - pos) / scale @ R
        """
        centered = points - self._mesh_pos
        scaled = centered * self._mesh_scale_inv
        return scaled @ self._mesh_R

    def _apply_mesh_and_joints(self, vertices, joints, pads):
        """Update mesh vertices, keypoint positions, pad points and pad arrows.

        vertices: (778, 3) raw MANO-world coordinates
        joints: (21, 3) raw MANO-world coordinates
        pads: (5, 3) raw MANO-world finger-pad surface points

        For flat (polyhedron) shading, each of the 1538 faces gets 3 unique
        vertices (4614 total). Per-vertex normals are set to the face normal,
        giving a faceted appearance.

        Vertices are transformed to mesh_vert storage space (inverse of
        MuJoCo's reference transform) so that after the renderer applies
        the forward transform, they appear at the raw MANO positions.

        Keypoint / pad body positions are set directly in world space (= raw
        MANO coordinates), since bodies are children of worldbody.

        Pad-normal arrows: capsule geom along z-axis, oriented so z aligns
        with the averaged face-normal at the pad vertex; body placed at the
        capsule midpoint (pad_pos + normal * length/2).
        """
        n_exp = self._n_expanded_verts

        # Expand: map 778 original verts -> 4614 expanded verts (3 per face)
        expanded = vertices[self._expanded_to_orig]  # (4614, 3)

        # Transform to mesh_vert storage space
        expanded_mesh = self._to_mesh_vert_space(expanded)
        self.model.mesh_vert[0:n_exp] = expanded_mesh.astype(np.float32)

        # Compute per-face normals (flat shading): each face's 3 verts share
        # the same normal = the face normal.
        f = self._expanded_faces
        v0 = expanded_mesh[f[:, 0]]
        v1 = expanded_mesh[f[:, 1]]
        v2 = expanded_mesh[f[:, 2]]
        face_normals = np.cross(v1 - v0, v2 - v0)
        norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
        norms = np.where(norms > 1e-12, norms, 1.0)
        face_normals = face_normals / norms
        # Assign face normal to each of the face's 3 expanded verts
        expanded_normals = np.repeat(face_normals, 3, axis=0)  # (4614, 3)
        self.model.mesh_normal[0:n_exp] = expanded_normals.astype(np.float32)

        # Set keypoint body positions in world space (raw MANO coordinates)
        for i in range(21):
            bid = self._kp_body_ids[i]
            if bid >= 0:
                self.model.body_pos[bid] = joints[i]

        # Set pad-keypoint body positions
        for i in range(5):
            bid = self._pad_body_ids[i]
            if bid >= 0:
                self.model.body_pos[bid] = pads[i]

        # Pad normals: average of adjacent face normals on the ORIGINAL 778-vert
        # mesh (smooth normal perpendicular to the local pad surface).
        orig_faces = self.mano_model.faces.astype(np.int64)
        of0 = vertices[orig_faces[:, 0]]
        of1 = vertices[orig_faces[:, 1]]
        of2 = vertices[orig_faces[:, 2]]
        orig_face_n = np.cross(of1 - of0, of2 - of0)
        nrm = np.linalg.norm(orig_face_n, axis=1, keepdims=True)
        nrm = np.where(nrm > 1e-12, nrm, 1.0)
        orig_face_n = orig_face_n / nrm  # (1538, 3) in MANO-world

        L = self._arrow_length
        fname = TIP_ORDER  # ["Thumb", "Index", "Middle", "Ring", "Pinky"]
        for i in range(5):
            name = fname[i]
            # Determine which vertices define this pad's surface normal
            custom_vid = self._custom_pad_vids.get(name)
            v3 = self._custom_pad_3pt_vertices.get(name)
            v3_count = sum(1 for v in (v3 or []) if v is not None and int(v) >= 0)

            if custom_vid is not None:
                # Single custom vertex — use its adjacent faces
                adj = np.where((orig_faces == int(custom_vid)).any(axis=1))[0]
            elif v3_count == 3:
                # 3-point centroid — average normals from all 3 vertices' faces
                adj_list = [np.where((orig_faces == int(v)).any(axis=1))[0]
                            for v in v3]
                adj = np.unique(np.concatenate(adj_list))
            else:
                # Default pad vertex
                adj = self._pad_adj_faces[i]

            nvec = orig_face_n[adj].mean(axis=0)
            nn = np.linalg.norm(nvec)
            if nn > 1e-12:
                nvec = nvec / nn
            else:
                nvec = np.array([0.0, -1.0, 0.0])

            # Arrow body at capsule midpoint, z-axis aligned with nvec.
            abid = self._arrow_body_ids[i]
            agid = self._arrow_geom_ids[i]
            if abid >= 0:
                mid = pads[i] + nvec * (L / 2.0)
                self.model.body_pos[abid] = mid
                # quat that rotates +z to nvec
                quat = np.zeros(4)
                mujoco.mju_quatZ2Vec(quat, nvec.astype(np.float64))
                self.model.body_quat[abid] = quat
            if agid >= 0:
                self.model.geom_size[agid] = [ARROW_RADIUS, L / 2.0, 0.0]

    def _create_model(self, default_vertices, default_joints, default_pads):
        """Create a MuJoCo model with MANO mesh + 21 keypoint spheres +
        5 pad keypoints + 5 pad-normal arrow capsules.

        The OBJ is written with EXPANDED vertices (4614 = 1538 faces x 3)
        for flat shading: each face references its own 3 unique vertices.
        """
        # Expand default vertices: 778 -> 4614 (3 per face)
        expanded_verts = default_vertices[self._expanded_to_orig]

        # Write temp OBJ file with expanded vertices
        self._tmpdir = tempfile.mkdtemp(prefix="mano_viewer_")
        obj_path = Path(self._tmpdir) / "mano_mesh.obj"
        with open(obj_path, "w") as f:
            f.write("o mano_mesh\n")
            for v in expanded_verts:
                f.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
            for fc in self._expanded_faces:
                f.write(f"f {fc[0] + 1} {fc[1] + 1} {fc[2] + 1}\n")

        r, g, b, a = self.mesh_rgba
        rad = self.KEYPOINT_RADIUS
        pad_rad = self.KEYPOINT_RADIUS  # same size as joint keypoints
        pr, pg, pb, pa = PAD_COLOR
        ar, ag, ab, aa = ARROW_COLOR
        half_len = DEFAULT_ARROW_LENGTH / 2.0

        # Place keypoint bodies at raw MANO joint positions
        kp_xml = ""
        for i in range(21):
            color = FINGER_COLORS[JOINT_FINGER[i]]
            kr, kg, kb, ka = color
            x, y, z = default_joints[i]
            kp_xml += (
                f'        <body name="kp_{i}" pos="{x:.8f} {y:.8f} {z:.8f}">\n'
                f'          <geom name="kp_{i}_geom" type="sphere" size="{rad}"\n'
                f'                rgba="{kr} {kg} {kb} {ka}"\n'
                f'                contype="0" conaffinity="0" mass="0"/>\n'
                f"        </body>\n"
            )

        # 5 pad-keypoint bodies (orange spheres)
        pad_xml = ""
        for i in range(5):
            x, y, z = default_pads[i]
            pad_xml += (
                f'        <body name="pad_{i}" pos="{x:.8f} {y:.8f} {z:.8f}">\n'
                f'          <geom name="pad_{i}_geom" type="sphere" size="{pad_rad}"\n'
                f'                rgba="{pr} {pg} {pb} {pa}"\n'
                f'                contype="0" conaffinity="0" mass="0"/>\n'
                f"        </body>\n"
            )

        # 5 pad-normal arrow bodies (yellow capsules along +z, repositioned
        # and reoriented every frame in _apply_mesh_and_joints).
        arrow_xml = ""
        for i in range(5):
            arrow_xml += (
                f'        <body name="arrow_{i}" pos="0 0 0">\n'
                f'          <geom name="arrow_{i}_geom" type="capsule"'
                f' size="{ARROW_RADIUS} {half_len}"\n'
                f'                rgba="{ar} {ag} {ab} {aa}"\n'
                f'                contype="0" conaffinity="0" mass="0"/>\n'
                f"        </body>\n"
            )

        # Candidate-sphere bodies (interactive pad selection)
        cand_xml = ""
        cr, cg, cb, ca = CAND_COLOR_DEFAULT
        cand_rad = CAND_RADIUS
        # Default positions will be overwritten at runtime; just need valid xml
        for i in range(self.mano_model.n_candidates_total):
            cand_xml += (
                f'        <body name="cand_{i}" pos="0 0 0">\n'
                f'          <geom name="cand_{i}_geom" type="sphere" size="{cand_rad}"\n'
                f'                rgba="{cr} {cg} {cb} {ca}"\n'
                f'                contype="0" conaffinity="0" mass="0"/>\n'
                f"        </body>\n"
            )

        mjcf = f"""
        <mujoco>
          <worldbody>
            <body name="mano_body" pos="0 0 0">
              <geom name="mano_geom" type="mesh" mesh="mano_mesh"
                    rgba="{r} {g} {b} {a}"
                    contype="0" conaffinity="0" mass="0.001" priority="1"/>
            </body>
{kp_xml}{pad_xml}{arrow_xml}{cand_xml}          </worldbody>
          <asset>
            <mesh name="mano_mesh" file="{obj_path}"/>
          </asset>
        </mujoco>
        """

        model = mujoco.MjModel.from_xml_string(mjcf)
        return model

    def set_mesh_rgba(self, rgba):
        """Update mesh RGBA color and opacity."""
        self.mesh_rgba = np.array(rgba, dtype=np.float64)
        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "mano_geom")
        if geom_id >= 0:
            self.model.geom_rgba[geom_id] = self.mesh_rgba

    def set_keypoints_visible(self, visible):
        """Toggle keypoint visibility by setting geom alpha."""
        self.show_keypoints = visible
        for i, gid in enumerate(self._kp_geom_ids):
            if gid >= 0:
                if visible:
                    self.model.geom_rgba[gid] = self._kp_colors[i]
                else:
                    self.model.geom_rgba[gid] = np.array([0, 0, 0, 0])

    def set_keypoint_radius(self, radius):
        """Update keypoint sphere sizes."""
        self._kp_radius = radius
        for gid in self._kp_geom_ids:
            if gid >= 0:
                self.model.geom_size[gid] = [radius, 0, 0]

    def set_pad_keypoints_visible(self, visible):
        """Toggle pad-keypoint (orange spheres) visibility."""
        self._pad_visible = visible
        pad_color = PAD_COLOR
        arrow_color = ARROW_COLOR
        alpha = pad_color[3] if visible else 0.0
        for gid in self._pad_geom_ids:
            if gid >= 0:
                c = pad_color.copy()
                c[3] = alpha
                self.model.geom_rgba[gid] = c
        # Also show/hide arrow capsules
        arrow_alpha = arrow_color[3] if visible else 0.0
        for gid in self._arrow_geom_ids:
            if gid >= 0:
                c = arrow_color.copy()
                c[3] = arrow_alpha
                self.model.geom_rgba[gid] = c

    def set_pad_keypoint_radius(self, radius):
        """Update pad-keypoint sphere sizes."""
        self._pad_radius = radius
        for gid in self._pad_geom_ids:
            if gid >= 0:
                self.model.geom_size[gid] = [radius, 0, 0]

    def set_arrow_length(self, length):
        """Set pad-normal arrow length (meters). Applied on next refresh."""
        self._arrow_length = float(length)

    def refresh(self, betas, hand_pose, global_orient, translation, scale):
        """Update visualization with current MANO parameters.

        Returns:
            True if viewer is still running, False if user closed the window.
        """
        if not self.viewer.is_running():
            return False

        # Compute MANO output in raw MANO coordinate frame
        vertices, joints, _pads_auto = self.mano_model.compute(
            betas, hand_pose, global_orient, translation, scale
        )
        self._last_vertices = vertices

        # Apply custom pad selection (vertex override or 3-point center)
        pads = self._resolve_pad_positions(vertices)

        # Write vertices (inverse-transformed to mesh_vert space) and
        # keypoint / pad positions (raw MANO world coordinates) + pad arrows
        self._apply_mesh_and_joints(vertices, joints, pads)

        # Detect double-click selection (poll perturb.select)
        self._check_perturb_select()

        # Candidate positions + colors
        self._set_candidate_positions_and_colors(
            vertices, pad_ids_overridden=self._custom_pad_vids
        )

        # If candidates are hidden, zero out their alphas
        if not self._candidates_visible:
            for ci in range(len(self._cand_body_ids)):
                cgid = self._cand_geom_ids[ci]
                if cgid >= 0:
                    self.model.geom_rgba[cgid][3] = 0.0

        # Force GPU re-upload of mesh data
        self.viewer.update_mesh(0)

        # Recompute forward dynamics so data.geom_xpos reflects new positions
        mujoco.mj_forward(self.model, self.data)

        self.viewer.sync()
        return True

    def _resolve_pad_positions(self, vertices):
        """Compute 5 pad positions using custom selection, falling back to auto.

        Returns (5, 3) array in MANO-world coordinates.
        """
        pads = []
        for fi, name in enumerate(TIP_ORDER):
            v3 = self._custom_pad_3pt_vertices.get(name)
            v3_count = sum(1 for v in (v3 or []) if v is not None and int(v) >= 0)
            if self._custom_pad_vids.get(name) is not None:
                # Single vertex override
                pads.append(vertices[int(self._custom_pad_vids[name])])
            elif v3_count == 3:
                # 3-point centroid from user's custom selection
                vs = [vertices[int(v)] for v in v3]
                pads.append(np.mean(vs, axis=0))
            elif PAD_3PT_VERTEX_IDS.get(name) is not None:
                # Default 3-point centroid
                pt3 = PAD_3PT_VERTEX_IDS[name]
                vs = [vertices[int(v)] for v in pt3]
                pads.append(np.mean(vs, axis=0))
            else:
                # Default single vertex
                pads.append(vertices[int(PAD_VERTEX_IDS[name])])
        return np.asarray(pads, dtype=np.float64)

    def _set_candidate_positions_and_colors(self, vertices, pad_ids_overridden):
        """Update candidate geom positions + colors per selection state.

        vertices: (778, 3) current deformed mesh in MANO-world coordinates.
        pad_ids_overridden: dict TIP_ORDER -> vid | None (per-finger custom pad)
            If a finger has an override, mark that candidate (if in candidate
            list) with PAD_COLOR.
        """
        n_cand = len(self._cand_vids)
        cand_world = vertices[self._cand_vids]  # (n_cand, 3)

        # Update body positions
        for ci in range(n_cand):
            cbid = self._cand_body_ids[ci]
            if cbid >= 0:
                self.model.body_pos[cbid] = cand_world[ci]

        # Default colors
        default_c = CAND_COLOR_DEFAULT
        sel_c = CAND_COLOR_SELECT
        abc_colors = [CAND_COLOR_A, CAND_COLOR_B, CAND_COLOR_C]

        # Map mesh vid -> candidate indices
        vid_to_cands = {}
        for ci in range(n_cand):
            v = int(self._cand_vids[ci])
            vid_to_cands.setdefault(v, []).append(ci)

        for ci in range(n_cand):
            cgid = self._cand_geom_ids[ci]
            if cgid < 0:
                continue
            color = default_c.copy()
            # ABC marker
            for abi in range(3):
                if ci < n_cand and self._abc[abi] >= 0 \
                        and self._cand_vids[ci] == self._abc[abi]:
                    color = abc_colors[abi]
            # Current selection (overrides ABC, last wins)
            if self._sel_cand == ci:
                color = sel_c.copy()
            # Custom pad marker: if this candidate is the per-finger selected pad
            if pad_ids_overridden is not None:
                this_vid = int(self._cand_vids[ci])
                for fi, fname in enumerate(TIP_ORDER):
                    ovr = pad_ids_overridden[fname]
                    if ovr is not None and this_vid == ovr:
                        color = PAD_COLOR.copy()
            self.model.geom_rgba[cgid] = color

    def set_candidates_visible(self, visible):
        """Show/hide all candidate spheres.

        When hiding: set all candidate geom alphas to 0.
        When showing: re-run _set_candidate_positions_and_colors to restore.
        """
        if visible:
            self._set_candidate_positions_and_colors(
                self._last_vertices, pad_ids_overridden=self._custom_pad_vids
            )
        else:
            for ci in range(len(self._cand_body_ids)):
                cgid = self._cand_geom_ids[ci]
                if cgid < 0:
                    continue
                self.model.geom_rgba[cgid][3] = 0.0

    def close(self):
        self.viewer.close()
        if self._tmpdir is not None:
            shutil.rmtree(self._tmpdir, ignore_errors=True)
            self._tmpdir = None


class SliderWithSpinbox(QWidget):
    """A slider paired with a spinbox for precise numeric input."""

    def __init__(self, label, minimum, maximum, value, step, parent=None):
        super().__init__(parent)
        self._value = float(value)
        self._minimum = minimum
        self._maximum = maximum
        self._step = step
        self._listeners = []

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.label = QLabel(label)
        self.label.setMinimumWidth(110)
        layout.addWidget(self.label, 1)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 1000)
        self.slider.setValue(self._to_slider(value))
        self.slider.valueChanged.connect(self._on_slider_changed)
        layout.addWidget(self.slider, 3)

        self.spinbox = QDoubleSpinBox()
        self.spinbox.setRange(minimum, maximum)
        self.spinbox.setSingleStep(step)
        self.spinbox.setDecimals(3)
        self.spinbox.setValue(value)
        self.spinbox.valueChanged.connect(self._on_spinbox_changed)
        self.spinbox.setFixedWidth(85)
        layout.addWidget(self.spinbox, 1)

    def _to_slider(self, value):
        """Map float value to slider integer range."""
        if self._maximum == self._minimum:
            return 500
        ratio = (value - self._minimum) / (self._maximum - self._minimum)
        return int(ratio * 1000)

    def _from_slider(self, slider_val):
        """Map slider integer back to float value."""
        ratio = slider_val / 1000.0
        return self._minimum + ratio * (self._maximum - self._minimum)

    def _on_slider_changed(self, value):
        float_val = self._from_slider(value)
        self.spinbox.blockSignals(True)
        self.spinbox.setValue(float_val)
        self.spinbox.blockSignals(False)
        self._value = float_val
        self._notify()

    def _on_spinbox_changed(self, value):
        self.slider.blockSignals(True)
        self.slider.setValue(self._to_slider(value))
        self.slider.blockSignals(False)
        self._value = float(value)
        self._notify()

    def _notify(self):
        for listener in self._listeners:
            listener(self._value)

    def value(self):
        return self._value

    def set_value(self, value):
        self._value = float(value)
        self.slider.blockSignals(True)
        self.slider.setValue(self._to_slider(self._value))
        self.slider.blockSignals(False)
        self.spinbox.blockSignals(True)
        self.spinbox.setValue(self._value)
        self.spinbox.blockSignals(False)
        self._notify()

    def add_listener(self, callback):
        self._listeners.append(callback)


class MainWindow(QMainWindow):
    """Main application window with MANO parameter sliders."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("MANO Hand Viewer — Parameter Explorer")
        self.setMinimumWidth(420)

        # Load MANO model
        if not MANO_MODEL_PATH.exists():
            self._show_error(f"MANO model not found at: {MANO_MODEL_PATH}")
            sys.exit(1)

        self.mano_model = MANOModel(str(MANO_MODEL_PATH))

        # Create MuJoCo viewer (non-blocking)
        self.viewer = MANOViewer(self.mano_model)

        # Build GUI
        self._build_ui()

        # Set initial parameter values
        self._init_parameters()

        # Setup refresh timer (30 Hz)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(33)  # ~30 fps

        self.status_label.setText(
            f"MANO loaded: {self.mano_model.n_verts} verts, "
            f"{self.mano_model.n_faces} faces, 21 joints. Ready."
        )

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)

        # --- Top: action buttons ---
        btn_layout = QHBoxLayout()
        self.reset_btn = QPushButton("Reset to Defaults")
        self.reset_btn.clicked.connect(self._reset)
        btn_layout.addWidget(self.reset_btn)

        self.keypoints_btn = QPushButton("Hide Keypoints")
        self.keypoints_btn.setCheckable(True)
        self.keypoints_btn.toggled.connect(self._toggle_keypoints)
        btn_layout.addWidget(self.keypoints_btn)

        btn_layout.addStretch()
        root_layout.addLayout(btn_layout)

        # --- Scroll area for sliders ---
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        # Beta sliders
        self.beta_sliders = []
        beta_group = QGroupBox("Betas (Shape, 10 dims)")
        beta_form = QVBoxLayout(beta_group)
        for i in range(10):
            slider = SliderWithSpinbox(
                f"beta_{i:02d}", -10.0, 10.0, 0.0, 0.01
            )
            self.beta_sliders.append(slider)
            beta_form.addWidget(slider)
        scroll_layout.addWidget(beta_group)

        # Global orient sliders
        self.global_orient_sliders = []
        go_group = QGroupBox("Global Orient (Wrist Rotation, axis-angle rad)")
        go_form = QVBoxLayout(go_group)
        for i, axis in enumerate(["X", "Y", "Z"]):
            slider = SliderWithSpinbox(
                f"axis_{axis}", -3.14159, 3.14159, 0.0, 0.01
            )
            self.global_orient_sliders.append(slider)
            go_form.addWidget(slider)
        scroll_layout.addWidget(go_group)

        # Hand pose sliders (15 joints x 3 axis-angle)
        self.hand_pose_sliders = []  # [finger_idx][joint_idx][axis_idx]
        hp_group = QGroupBox("Hand Pose (15 joints x 3 axis-angle rad)")
        hp_layout = QVBoxLayout(hp_group)

        for finger_idx, finger_name in enumerate(FINGER_ORDER):
            finger_joints = FINGER_JOINT_NAMES[finger_name]
            finger_group = QGroupBox(finger_name)
            finger_layout = QVBoxLayout(finger_group)
            finger_sliders = []
            for joint_idx, joint_name in enumerate(finger_joints):
                joint_sliders = []
                for axis_idx, axis in enumerate(["X", "Y", "Z"]):
                    slider = SliderWithSpinbox(
                        f"{joint_name} {axis}", -3.14159, 3.14159, 0.0, 0.01
                    )
                    joint_sliders.append(slider)
                    finger_layout.addWidget(slider)
                finger_sliders.append(joint_sliders)
            self.hand_pose_sliders.append(finger_sliders)
            hp_layout.addWidget(finger_group)

        scroll_layout.addWidget(hp_group)

        # Translation sliders
        self.translation_sliders = []
        trans_group = QGroupBox("Translation (World Position, meters)")
        trans_form = QVBoxLayout(trans_group)
        for i, axis in enumerate(["X", "Y", "Z"]):
            slider = SliderWithSpinbox(
                f"pos_{axis}", -0.5, 0.5, 0.0, 0.001
            )
            self.translation_sliders.append(slider)
            trans_form.addWidget(slider)
        scroll_layout.addWidget(trans_group)

        # Scale slider
        scale_group = QGroupBox("Scale (Global Multiplier, 0.001 = mm->m baseline)")
        scale_form = QVBoxLayout(scale_group)
        self.scale_slider = SliderWithSpinbox("scale", 0.1, 3.0, 1.0, 0.01)
        scale_form.addWidget(self.scale_slider)
        scroll_layout.addWidget(scale_group)

        # Mesh color controls
        color_group = QGroupBox("Mesh Appearance")
        color_form = QVBoxLayout(color_group)
        self.color_r_slider = SliderWithSpinbox("Red", 0.0, 1.0, 0.3, 0.01)
        self.color_g_slider = SliderWithSpinbox("Green", 0.0, 1.0, 0.6, 0.01)
        self.color_b_slider = SliderWithSpinbox("Blue", 0.0, 1.0, 1.0, 0.01)
        self.color_a_slider = SliderWithSpinbox("Alpha", 0.0, 1.0, 0.5, 0.01)
        for s in [self.color_r_slider, self.color_g_slider, self.color_b_slider, self.color_a_slider]:
            color_form.addWidget(s)
        self.color_r_slider.add_listener(lambda _: self._update_mesh_color())
        self.color_g_slider.add_listener(lambda _: self._update_mesh_color())
        self.color_b_slider.add_listener(lambda _: self._update_mesh_color())
        self.color_a_slider.add_listener(lambda _: self._update_mesh_color())
        scroll_layout.addWidget(color_group)

        # Keypoint radius control
        kp_group = QGroupBox("Keypoint Appearance")
        kp_form = QVBoxLayout(kp_group)
        self.kp_radius_slider = SliderWithSpinbox(
            "Sphere Radius", 0.001, 0.02, 0.003, 0.0005
        )
        self.kp_radius_slider.add_listener(
            lambda v: self.viewer.set_keypoint_radius(v)
        )
        kp_form.addWidget(self.kp_radius_slider)
        scroll_layout.addWidget(kp_group)

        # Pad-normal arrow length control
        pad_group = QGroupBox("Finger Pad Arrows")
        pad_form = QVBoxLayout(pad_group)
        self.arrow_len_slider = SliderWithSpinbox(
            "Arrow Length", 0.005, 0.03, DEFAULT_ARROW_LENGTH, 0.001
        )
        self.arrow_len_slider.add_listener(
            lambda v: self.viewer.set_arrow_length(v)
        )
        pad_form.addWidget(self.arrow_len_slider)
        scroll_layout.addWidget(pad_group)

        # Pad keypoint appearance (separate from joint keypoints)
        pad_kp_group = QGroupBox("Pad Keypoint Appearance")
        pad_kp_form = QVBoxLayout(pad_kp_group)

        # Visibility toggle
        self.pad_kp_btn = QPushButton("Hide Pad Keypoints")
        self.pad_kp_btn.setCheckable(True)
        self.pad_kp_btn.setChecked(False)  # visible by default
        self.pad_kp_btn.toggled.connect(self._toggle_pad_keypoints)
        pad_kp_form.addWidget(self.pad_kp_btn)

        # Pad keypoint radius slider
        self.pad_kp_radius_slider = SliderWithSpinbox(
            "Pad Sphere Radius", 0.001, 0.02, self.viewer.KEYPOINT_RADIUS, 0.0005
        )
        self.pad_kp_radius_slider.add_listener(
            lambda v: self.viewer.set_pad_keypoint_radius(v)
        )
        pad_kp_form.addWidget(self.pad_kp_radius_slider)
        scroll_layout.addWidget(pad_kp_group)

        # ===== Interactive Pad-Selection Panel =====
        select_group = QGroupBox("Pad Selection (Interactive)")
        sel_form = QVBoxLayout(select_group)

        # Per-finger status labels
        self.finger_status_labels = {}
        fingers_row = QVBoxLayout()
        for name in TIP_ORDER:
            row = QHBoxLayout()
            lbl_fname = QLabel(f"<b>{name}</b>")
            lbl_fname.setMinimumWidth(55)
            lbl_val = QLabel("auto")
            lbl_val.setStyleSheet("color: #666;")
            lbl_val.setTextInteractionFlags(Qt.TextSelectableByMouse)
            row.addWidget(lbl_fname)
            row.addWidget(lbl_val, 1)
            btn_clear = QPushButton("X")
            btn_clear.setFixedWidth(30)
            btn_clear.setToolTip(f"Clear {name} custom selection (back to auto)")
            btn_clear.clicked.connect(
                lambda _=False, n=name, lv=lbl_val: self._clear_finger_selection(n, lv)
            )
            row.addWidget(btn_clear)
            fingers_row.addLayout(row)
            self.finger_status_labels[name] = lbl_val
        sel_form.addLayout(fingers_row)

        # ABC status
        abc_row = QHBoxLayout()
        self.abc_status_labels = [QLabel("A: -"), QLabel("B: -"), QLabel("C: -")]
        for i, lbl in enumerate(self.abc_status_labels):
            lbl.setMinimumWidth(55)
            abc_row.addWidget(lbl)
        btn_clear_abc = QPushButton("Clear A/B/C")
        btn_clear_abc.clicked.connect(self._clear_abc)
        abc_row.addWidget(btn_clear_abc)
        abc_row.addStretch()
        sel_form.addLayout(abc_row)

        # Action buttons
        btn_row1 = QHBoxLayout()
        btn_export = QPushButton("Export (S)")
        btn_export.setToolTip("Save current PAD_VERTEX_IDS dict (printed to stdout and clipboard)")
        btn_export.clicked.connect(self._export_pad_ids)
        btn_reset = QPushButton("Reset All (R)")
        btn_reset.setToolTip("Revert all pads to auto; clear A/B/C")
        btn_reset.clicked.connect(self._reset_all_pads)
        btn_cand = QPushButton("Show Candidates")
        btn_cand.setCheckable(True)
        btn_cand.setChecked(True)  # checked = "hidden" state for toggle
        btn_cand.clicked.connect(self._toggle_candidates)
        btn_row1.addWidget(btn_export)
        btn_row1.addWidget(btn_reset)
        btn_row1.addWidget(btn_cand)
        sel_form.addLayout(btn_row1)

        # Instructions + status
        self._candidates_visible = False
        instr = QLabel(
            "使用说明:\n"
            "  • 点击 'Show Candidates' 开启候选球模式\n"
            "  • 在 MuJoCo 窗口中双击候选球进行选中\n"
            "  • 选中后按 1..5 将其设为 Thumb/Index/Middle/Ring/Pinky 的指腹点\n"
            "  • 选中 3 个候选球后按 A, B, C 标记三角点，再按 1..5 使用重心\n"
            "  • S = 导出字典, R = 全部重置, Esc = 清除当前选中"
        )
        instr.setStyleSheet("color: #555; font-size: 10pt;")
        instr.setWordWrap(True)
        sel_form.addWidget(instr)

        scroll_layout.addWidget(select_group)

        scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        root_layout.addWidget(scroll, 1)

        # Status bar
        self.status_label = QLabel("Ready")
        root_layout.addWidget(self.status_label)

    def _poll_selection_status(self):
        """Called periodically to keep the PySide6 status labels in sync."""
        v = self.viewer
        # Per-finger
        for name in TIP_ORDER:
            vid = v._custom_pad_vids.get(name)
            v3 = v._custom_pad_3pt_vertices.get(name)
            lbl = self.finger_status_labels[name]
            if vid is not None:
                lbl.setText(f"vertex v{int(vid)}")
                lbl.setStyleSheet("color: #c0392b; font-weight: bold;")
            elif v3 is not None and sum(1 for x in v3 if x is not None and int(x) >= 0) == 3:
                vs = [str(int(x)) for x in v3]
                lbl.setText(f"3pt {','.join(vs)}")
                lbl.setStyleSheet("color: #27ae60; font-weight: bold;")
            elif PAD_3PT_VERTEX_IDS.get(name) is not None:
                pt3 = PAD_3PT_VERTEX_IDS[name]
                vs = [str(int(x)) for x in pt3]
                lbl.setText(f"3pt {','.join(vs)}")
                lbl.setStyleSheet("color: #27ae60;")
            else:
                lbl.setText(f"v{PAD_VERTEX_IDS[name]}")
                lbl.setStyleSheet("color: #666;")
        # A/B/C
        letters = ["A", "B", "C"]
        for i in range(3):
            lbl = self.abc_status_labels[i]
            vid = v._abc[i]
            if vid >= 0:
                lbl.setText(f"{letters[i]}: v{vid}")
                lbl.setStyleSheet("color: #2980b9; font-weight: bold;")
            else:
                lbl.setText(f"{letters[i]}: -")
                lbl.setStyleSheet("color: #666;")
        # Viewer status line
        new_text = v._selection_status or ""
        if self.status_label.text() != new_text:
            self.status_label.setText(new_text)

    def _clear_finger_selection(self, name, label):
        """Reset a single finger back to auto."""
        self.viewer._custom_pad_vids[name] = None
        self.viewer._custom_pad_3pt_vertices[name] = None
        self.viewer._selection_status = f"[RESET] {name} back to auto."
        label.setText("auto")
        label.setStyleSheet("color: #666;")

    def _clear_abc(self):
        self.viewer._abc = [-1, -1, -1]
        self.viewer._selection_status = "[RESET] A/B/C markers cleared."

    def _reset_all_pads(self):
        # Invoke the same code as pressing 'R'
        for fname in TIP_ORDER:
            self.viewer._custom_pad_vids[fname] = None
            self.viewer._custom_pad_3pt_vertices[fname] = None
        self.viewer._abc = [-1, -1, -1]
        self.viewer._sel_cand = -1
        self.viewer._selection_status = "[RESET] All pads back to auto; ABC cleared."
        self._poll_selection_status()

    def _export_pad_ids(self):
        """Duplicate the 'S' key export logic (usable from button)."""
        lines = ["PAD_VERTEX_IDS = {"]
        lines3pt = ["PAD_3PT_VERTEX_IDS = {"]
        has_3pt_defaults = False
        for fname in TIP_ORDER:
            vid = self.viewer._custom_pad_vids.get(fname)
            if vid is not None:
                lines.append(f'    "{fname}": {int(vid)},')
                pt3 = self.viewer._custom_pad_3pt_vertices.get(fname)
                if pt3 is not None and sum(1 for v in pt3 if v is not None and int(v) >= 0) == 3:
                    v3s = [str(int(v)) for v in pt3]
                    lines3pt.append(f'    "{fname}": [{", ".join(v3s)}],')
                    has_3pt_defaults = True
            elif self.viewer._custom_pad_3pt_vertices.get(fname) is not None:
                v3 = self.viewer._custom_pad_3pt_vertices[fname]
                v3s = [str(int(v)) for v in v3]
                lines3pt.append(f'    "{fname}": [{", ".join(v3s)}],')
                has_3pt_defaults = True
                lines.append(f'    # "{fname}": 3pt center v{v3s},')
            else:
                pt3_default = PAD_3PT_VERTEX_IDS.get(fname)
                if pt3_default is not None:
                    v3s = [str(int(v)) for v in pt3_default]
                    lines3pt.append(f'    "{fname}": [{", ".join(v3s)}],')
                    has_3pt_defaults = True
                    lines.append(f'    # "{fname}": 3pt center v{v3s},')
                else:
                    lines.append(
                        f'    "{fname}": {PAD_VERTEX_IDS[fname]},'
                    )
        lines.append("}")
        lines3pt.append("}")
        block = "\n".join(lines)
        if has_3pt_defaults:
            block += "\n\n" + "\n".join(lines3pt)
        self.viewer._selection_status = f"[EXPORT] see stdout"
        print("=" * 40)
        print(block)
        print("=" * 40)
        try:
            import subprocess
            subprocess.run(
                ["xclip", "-selection", "clipboard"],
                input=block, text=True, check=False, timeout=3,
            )
        except Exception:
            pass
        self._poll_selection_status()

    def _toggle_candidates(self, checked):
        visible = not checked
        self.viewer.set_candidates_visible(visible)
        self.viewer._candidates_visible = visible
        sender = self.sender()
        try:
            sender.setText("Show Candidates" if checked else "Hide Candidates")
        except Exception:
            pass
        self._candidates_visible = visible

    def _init_parameters(self):
        """Set all sliders to default values."""
        defaults = self.mano_model
        for i, slider in enumerate(self.beta_sliders):
            slider.set_value(defaults.default_betas[i])
        for i, slider in enumerate(self.global_orient_sliders):
            slider.set_value(defaults.default_global_orient[i])
        for fi in range(5):
            for ji in range(3):
                for ai in range(3):
                    slider = self.hand_pose_sliders[fi][ji][ai]
                    slider.set_value(defaults.default_hand_pose[fi * 3 + ji, ai])
        for i, slider in enumerate(self.translation_sliders):
            slider.set_value(defaults.default_translation[i])
        self.scale_slider.set_value(1.0)

    def _reset(self):
        """Reset all sliders to defaults."""
        self._init_parameters()
        self.status_label.setText("Reset to defaults")

    def _toggle_keypoints(self, checked):
        """Toggle keypoint visibility."""
        self.viewer.set_keypoints_visible(not checked)
        self.keypoints_btn.setText("Show Keypoints" if checked else "Hide Keypoints")

    def _toggle_pad_keypoints(self, checked):
        """Toggle pad-keypoint (orange spheres + arrows) visibility."""
        self.viewer.set_pad_keypoints_visible(not checked)
        self.pad_kp_btn.setText(
            "Show Pad Keypoints" if checked else "Hide Pad Keypoints"
        )

    def _update_mesh_color(self):
        """Update mesh RGBA from color sliders."""
        rgba = [
            self.color_r_slider.value(),
            self.color_g_slider.value(),
            self.color_b_slider.value(),
            self.color_a_slider.value(),
        ]
        self.viewer.set_mesh_rgba(rgba)

    def _get_parameters(self):
        """Read current parameters from all sliders."""
        betas = np.array([s.value() for s in self.beta_sliders], dtype=np.float64)
        global_orient = np.array([s.value() for s in self.global_orient_sliders], dtype=np.float64)
        translation = np.array([s.value() for s in self.translation_sliders], dtype=np.float64)
        scale = self.scale_slider.value()

        hand_pose = np.zeros((15, 3), dtype=np.float64)
        for fi in range(5):
            for ji in range(3):
                for ai in range(3):
                    hand_pose[fi * 3 + ji, ai] = \
                        self.hand_pose_sliders[fi][ji][ai].value()

        return betas, hand_pose, global_orient, translation, scale

    def _refresh(self):
        """Timer callback: read parameters and update viewer."""
        if not self.viewer.refresh(*self._get_parameters()):
            self._timer.stop()
            self.close()
            return
        # Keep PySide6 selection panel up-to-date
        self._poll_selection_status()

    def closeEvent(self, event):
        """Handle window close."""
        self._timer.stop()
        self.viewer.close()
        super().closeEvent(event)

    def _show_error(self, message):
        """Display error message."""
        self.status_label.setText(f"Error: {message}")


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
