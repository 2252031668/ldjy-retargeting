# OpenArm Hand Assets

`assets/robots/openarm_hand` contains a fixed-base OpenArm bimanual model with
two 20-DOF LDJY hands. `example/teleop_sim.py --robot openarm` uses it for
simulation: the selected arm is held in a measured raised home pose and only
that hand's 20 joints receive retargeted commands.

The 7-DOF arms are not part of the retargeting optimizer and this mode does not
control physical OpenArm hardware.

## Coordinate contract

The generated `urdf/openarm_bimanual_mano.urdf` preserves the original OpenArm
world root and all 54 movable joints: 7 left-arm + 7 right-arm + 20 joints per
hand. Each hand mount is split without changing its source zero-pose palm:

```text
hand_adapter -> {side}_retarget_wrist -> {side}_palm
```

`{side}_retarget_wrist` uses the same MANO wrist convention as `ldjy_hand`.
Each distal link has a fixed `{side}_{finger}_tip` frame. The tip frame is
derived from the source link-4 mesh extremity during generation, not a hard
coded finger-length constant.

## Build

Run both generators from the repository root after changing source meshes or
the source URDF:

```bash
uv run --no-sync python tools/build_openarm_hand_urdf.py
uv run --no-sync python tools/build_openarm_hand_mjcf.py
uv run --no-sync python -m unittest tests.test_openarm_asset_generation -v
```

The generated MJCF retains a fixed root and creates one position actuator for
every original movable joint, for a total of 54. Arm and hand controls are
therefore addressed by joint name, not by a fragile qpos offset.

The generated URDF replaces every source collision geometry with its authored
visual surface before import, so MuJoCo uses visual meshes as collision meshes.
The original simplified collision mesh files remain vendored only as source
provenance and are not referenced by the generated URDF or MJCF. Because visual
CAD surfaces overlap at several assembled joints, the MJCF automatically
excludes body pairs already penetrating at the zero pose; all other visual mesh
contacts, including floor and non-assembly contacts, remain enabled.

The MJCF is intended to be directly viewable and dynamically stable at zero
control. It uses an `implicitfast` integrator, hand `armature=0.002` and
`damping=0.02`, and `kp=3, kv=0.4` for hand joints. The source hand torque
limits are preserved. `teleop_sim.py` schedules its control/viewer loop at
120 Hz and advances this 2 ms physics model by four or five substeps per tick,
preserving a 500 Hz average physics rate.

Arm joints use a shared left/right per-joint table, calibrated in three
representative gravity-loaded poses rather than one uniform gain:

| Joint | kp | kv | damping | armature |
| --- | ---: | ---: | ---: | ---: |
| J1 | 300 | 30 | 2 | 0.02 |
| J2 | 300 | 30 | 2 | 0.02 |
| J3 | 150 | 12 | 2 | 0.02 |
| J4 | 300 | 18 | 2 | 0.02 |
| J5 | 150 | 6 | 2 | 0.02 |
| J6 | 100 | 15 | 5 | 0.02 |
| J7 | 100 | 12 | 5 | 0.02 |

A checker floor is placed 1 mm below the fixed pedestal's zero-pose lowest mesh point.
The scene also adds a brighter viewer headlight plus fixed key, fill, and rim
lights so the source dark-red material remains legible without changing its
authored color.

## Source provenance

The source asset was imported from the user-provided local directory
`/home/wxx/下载/openarm_hand_description`. Its license was not supplied with the
directory. Confirm redistribution rights before packaging or publishing the
embedded meshes outside this repository.
