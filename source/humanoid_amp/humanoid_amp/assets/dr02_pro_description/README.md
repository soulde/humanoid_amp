# DR02-Pro model

The URDF geometry and meshes originate from `soulde_robot_zoo/robots/dr02_pro`.
The complete mesh set is mirrored from the corresponding robot description used
by `robot_lab` because the zoo MJCF references several mesh files that are not
present in its `meshes` directory.

The two neck joints are fixed in this copy to match the 29 actuated degrees of
freedom in the finetuned `dr02_pos.xml` model at `soulde_robot_zoo` commit
`57def56`.

Converted USD files are cached below `$TMPDIR/IsaacLab/dr02_pro`. `TMPDIR` is
required, so converter output cannot silently fall back to a global cache.
