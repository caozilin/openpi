# Task/null feedback ablation and B10 replay — 2026-10-20

Original-dataset scope correction (2026-10-21): these replay targets were decoded
from OpenVLA-regenerated states, not the original collection states. A separate
archived-XML/state/action audit proves original Goal02/demo_10/75 and
Long08/demo_17/373 pose targets unreachable by a global shoulder/wrist necessary
bound. Original Spatial04/demo_22/105 is reachable (verified joint witnesses),
while its regenerated counterpart is unreachable. Do not attribute all three
replay failures to original demonstration targets. See
[original-sample diagnosis](ORIGINAL_DATASET_DIAGNOSIS.md) and
`results/original_dataset_target_audit_v2` for certificates and original images.

The new `osc_task_joint_null` option is experimental. The default remains `native`.
It retains the full IPOPT joint reference explicitly: FK(q_d) defines the task pose;
q_d defines the projected redundant posture target. This is local task/null
configuration tracking, not the original seven independent joint-PD dynamics or a global convergence guarantee.

All physics, gripper actuation, clipping and success evaluation still run through native LIBERO `env.step`.
The original 20 Hz reconstructed physical action HDF5 is unchanged. TaskTol/IPOPT,
objective and tolerance implementation remain unchanged. No VLA checkpoint is connected.

## Controller factors

Use native robosuite `opspace_matrices`, `orientation_error` and `nullspace_torques`:

```text
u_q = 150 (q_d - q) - 2 sqrt(150) qdot
u_x = K_x [p_goal-p; native_cross_column_orientation_error] - D_x ee_velocity
N = I - M^-1 J.T Lambda_full J
tau = J.T Lambda_split u_task + N.T M u_null + bias
```

| Mode | Task input | Null target | Null kp |
| --- | --- | --- | ---: |
| A10 | J u_q | q_d | 10 |
| B10 | native OSC u_x | q_d | 10 |
| C10 | J u_q | native initial_joint | 10 |
| D10 | native OSC u_x | native initial_joint | 10 |
| old_A150 | J u_q | q_d | 150 |

D10 is checked against an additional unmodified native OSC branch. The balanced
2×2 uses the same null gains so reference changes are not confounded with stiffness.
The old inertia correction is an extra baseline, not A10.

## Same-state short forks

Final evidence: `results/factorized_short_forks_recorded_q_v2/analysis.json`.
Four tasks (Spatial00, Goal05, Long00, Long06), 32 snapshots, six branches,
25 native physics steps per branch, 50 ms each. Events include initial/early motion,
gripper close, first movable contact and additional Long00 divergence windows.

Forks copy native integration state (including warm-start), matching physical model
parameters, native controller state including initial_joint, gripper accumulator,
and control-step/time fields. No custom physics loop is used. Source capture is
read-only: all four OSC source joint and EEF trajectories match the established
source replay bit for bit. D10 recovers native OSC. All 288 factor/recovery checks pass.
All 192 branch q_d values are bitwise identical to the full replay's q_d at those frames.
64 source event images and per-branch physical/contact/object traces are retained.

| Mode | Mean EEF position difference after 50 ms | Maximum |
| --- | ---: | ---: |
| A10 | 0.393838 mm | 1.064480 mm |
| B10 | 0.004140 mm | 0.033961 mm |
| C10 | 0.393666 mm | 1.055996 mm |
| D10 | numerical precision | numerical precision |
| old_A150 | 0.704470 mm | 3.487241 mm |

Mean absolute difference in integrated robot/external normal contact impulse:
A10 0.0222384 Ns, B10 0.00114741 Ns. These are short-window observations, not
proof of equal long-horizon contact behavior or task success.

Audit history: pilot_v1 is invalid because an instrumentation method override
was copied into controller state; excluded. A guard and regression test prevent it.
The earlier `factorized_short_forks_v1` used freshly solved q_d at sampled frames.
Its q_d differs from the full replay by up to 0.05836 rad due to IK history; it is
valid only as a within-snapshot factor comparison. Final evidence above instead
uses the exact q_d from the complete canonical IPOPT replay. Do not mix the two results.

## Complete 40-task replay

Same fixed selected demos, kp150 task gain, tolerance off; errors use common 6441
executed frames against the same HDF5-driven native OSC replay. 6444 attempted
frames, 3 rejected IPOPT targets. No infeasible output is executed.

| Controller | Final success | Ever success | Mean position difference | Median | p95 | Maximum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Native OSC | 39/40 | 39/40 | — | — | — | — |
| IPOPT + native joint | 34/40 | 36/40 | 4.073 mm | 3.021 mm | 10.543 mm | 45.541 mm |
| Old inertia correction | 32/40 | 33/40 | 2.029 mm | 1.297 mm | 5.370 mm | 60.549 mm |
| B10 | 35/40 | 35/40 | 1.650 mm | 0.158 mm | 2.015 mm | 163.848 mm |

Full q_d trajectories are bitwise identical across B10 and native-joint execution
for all 40 cases. Thus the observed differences here arise from execution changes,
not from a changed accepted joint-reference sequence.

B10 gains final success on Spatial06 and Goal05, loses Long03 relative to native-joint
execution. Long03 also fails with native OSC. All B10 successes persist to the final
recorded frame, but its ever-success count is lower than the native-joint baseline.
It is not promoted as a replacement based on a one-task final-success improvement.

Three solver rejections remain Spatial04/105 (CPU limit), Goal02/75 (local
infeasibility) and Long08/371 (iteration limit). The two executed B10 task failures
are Long00 and Long03. Do not combine solver reliability and execution failures.

## Long00 divergence audit

Read-only 500 Hz traces reproduce the main B10 and OSC trajectories exactly.
B10 clips torque at 841/7125 calls, native OSC at 0/7125. However:

- EEF difference first exceeds 1 mm at reference frame124, t=6.2 s.
- It exceeds 10 mm at t=6.41 s.
- First clipping is frame196, t=9.8 s; EEF difference is already 31.37 mm.
- First saturation is joint6, requested -12.226 Nm, limited to -12 Nm.

Therefore clipping cannot be called the initiating cause. Images near frames124–128
show divergence during the first can insertion into the basket with the gripper
command still closed. The exact causal contact/force mechanism remains unverified.
Same-index later torques belong to different states and must not be presented as
same-state controller comparisons.

Evidence: `results/factorized_B10_long00_diagnosis_v1.json`,
`results/tracking_factorized_B10_long00_v1`, `results/tracking_saved_osc_long00_v1`,
`results/factorized_B10_long00_event_images`.

## Validation and source

20 relevant native-environment tests pass; one unrelated generic-model test is
deselected for the pinned MuJoCo version. compileall and Ruff F/E9 pass.
40-task pairing: 720/720 checks. 44 complete replay videos (40 main + 4 initial
cases) and two additional instrumented Long00 videos fully decode; process images
are saved. No HTML report is generated.

Shared implementation: `franka_mujoco/tasktol/simulation/libero_joint_controller.py`
and `libero.py`; canonical replay CLI gains `--joint-control-mode osc_task_joint_null`.
The new mode uses native absolute OSC pose/gripper command transport plus an
explicit seven-dimensional q_d setter, not the default 8D joint-action encoding.
Every frame verifies the controller-held q_d equals the IPOPT output. Statistics
verify the factorized controller runs at every native physical step.

Full results: `results/ipopt_factorized_B10_comparison_v1.json`.
Reproduction inputs and unchanged source fingerprints are retained per batch.
