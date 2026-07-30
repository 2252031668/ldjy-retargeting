# OpenArm Bimanual Hand Asset

This directory contains an independently styled, fixed-base OpenArm robot with
two 20-DOF LDJY hands. The generated URDF and MJCF preserve the source red,
dark-red, and gold mesh appearance.

The output files are generated, not hand edited:

```bash
uv run --no-sync python tools/build_openarm_hand_urdf.py
uv run --no-sync python tools/build_openarm_hand_mjcf.py
```

`urdf/openarm_bimanual_mano.urdf` keeps all 54 original movable joints and
adds `left/right_retarget_wrist` plus five `{finger}_tip` task frames per hand.
`mjcf/openarm_bimanual_mano.xml` uses the same fixed root and exposes one
position actuator for every movable joint. It uses the authored visual meshes
as collision meshes; the source simplified collision meshes are replaced.

Source meshes and the original bimanual URDF came from the user-provided
`/home/wxx/下载/openarm_hand_description` directory. Confirm their license before
redistributing them.
