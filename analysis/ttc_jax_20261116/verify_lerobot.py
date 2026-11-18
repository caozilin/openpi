"""Numerical comparison using only three inspected definitions from pinned LeRobot source."""
import ast
import enum
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch
from openpi.models import rtc

root = Path(__file__).resolve().parent
namespace = {"enum": enum, "torch": torch, "Tensor": torch.Tensor}
for filename, definitions in [
    ("lerobot_common_flow_matching.py", {"FlowConvention", "make_flow_matching_inputs"}),
    ("lerobot_pi05_modeling_pi05.py", {"_reduce_training_rtc_loss"}),
]:
    parsed = ast.parse((root / filename).read_text())
    selected = [node for node in parsed.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                and node.name in definitions]
    assert {node.name for node in selected} == definitions
    code = ast.Module(body=selected, type_ignores=[])
    exec(compile(code, filename, "exec"), namespace)

rng = np.random.default_rng(42)
results = []
for dtype, torch_dtype, tolerance in [(jnp.float32, torch.float32, 1e-6), (jnp.bfloat16, torch.bfloat16, 0.02)]:
    for dim in (7, 32):
        for lengths in ([0, 0, 0, 0], [0, 1, 2, 3], [3, 3, 3, 3]):
            actions, noise, predictions = [rng.normal(size=(4, 8, dim)).astype(np.float32) for _ in range(3)]
            times = np.asarray([.125, .25, .5, .875], np.float32)
            prefix_mask = np.arange(8)[None] < np.asarray(lengths)[:, None]
            a, n, p = [torch.from_numpy(value).to(torch_dtype) for value in (actions, noise, predictions)]
            tx, tv, tt = namespace["make_flow_matching_inputs"](
                a, n, torch.from_numpy(times).to(torch_dtype),
                convention=namespace["FlowConvention"].NOISE_AT_ONE,
                prefix_mask=torch.from_numpy(prefix_mask))
            x, v, token_time, valid = jax.jit(rtc.flow_inputs)(
                jnp.asarray(actions, dtype), jnp.asarray(noise, dtype),
                jnp.asarray(times, dtype), jnp.asarray(lengths))
            maximum = 0.0
            for actual, expected in [(x, tx), (v, tv), (token_time, tt)]:
                left = np.asarray(actual, np.float32)
                right = expected.float().numpy()
                np.testing.assert_allclose(left, right, rtol=tolerance, atol=tolerance)
                maximum = max(maximum, float(np.max(np.abs(left-right))))
            torch_loss = namespace["_reduce_training_rtc_loss"](
                (p-tv).float().square(), torch.from_numpy(prefix_mask), "mean")
            jax_loss = rtc.mask_loss(jnp.mean(jnp.square((jnp.asarray(predictions, dtype)-v).astype(jnp.float32)), axis=-1), valid).mean()
            np.testing.assert_allclose(float(jax_loss), float(torch_loss), rtol=tolerance, atol=tolerance)
            results.append({"dtype": str(dtype), "action_dim": dim, "delays": lengths,
                            "max_flow_absolute_difference": maximum,
                            "jax_loss": float(jax_loss), "lerobot_loss": float(torch_loss), "passed": True})
(root / "lerobot_comparison.json").write_text(json.dumps(results, indent=2) + "\n")
print(json.dumps(results, indent=2))
