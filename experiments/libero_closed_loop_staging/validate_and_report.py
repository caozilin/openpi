"""Audit recorded native LIBERO/IPOPT evidence and build a local report."""

from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import shutil
import subprocess


def main():
    root = Path(__file__).resolve().parent
    project = root.parents[1]
    checks = []
    cases = []

    def check(name, ok):
        checks.append(dict(name=name, passed=bool(ok)))

    batches = ("native_nominal40", "native_tolerance5_four")
    for batch in batches:
        summary = json.loads((root / "results" / batch / "batch_summary.json").read_text())
        check(batch + ":complete", summary["complete"])
        for entry in summary["results"]:
            path = Path(entry["report"])
            report = json.loads(path.read_text())
            frames = json.loads(path.with_name("trajectory.json").read_text())["frames"]
            key = f"{batch}/{entry['suite']}/{entry['task_id']:02d}"
            check(key + ":native_backend", report["execution_backend"] == "native_LIBERO_env.step")
            check(
                key + ":one_solver_per_frame",
                all(f["optimizer"]["backend"] == "ipopt" and f["optimizer"]["solver_called"] for f in frames),
            )
            check(key + ":frame_count", len(frames) == report["frames"])
            executed = [f for f in frames if f["physics_steps"] > 0]
            check(key + ":native_steps", len(executed) == report["native_env_steps"])
            check(
                key + ":physics_timing",
                all(f["physics_steps"] == 25 for f in executed)
                and report["control_dt"] == 0.05
                and report["physics_dt"] == 0.002,
            )
            check(key + ":exact_joint_targets", all(f["native_joint_goal_error_rad"] < 1e-10 for f in executed))
            check(key + ":default_gains", report["native_controller_kp"] == [50.0] * 7)
            check(
                key + ":strict_transitions",
                all(f["effective_mask"] == "000" for f in frames if f["phase"] in ("ENG", "DIS", "outside_subtask")),
            )
            check(
                key + ":masked_axes",
                all(all(a == "0" or b == "1" for a, b in zip(f["effective_mask"], f["semantic_mask"])) for f in frames),
            )
            check(
                key + ":frozen_prediction",
                report["annotation_sha256"] == "84d383ce65d853a91507a9d56ae28a22d1a6bc4319ed9fa5406944c2e71086c9",
            )
            if "tolerance" in batch:
                check(
                    key + ":planned_tolerance_bounds",
                    all(
                        max(f.get("predicted_tolerance_violation_rad", [0])) <= 1e-7
                        for f in frames
                        if f["optimizer"]["feasible"]
                    ),
                )
            ffprobe = shutil.which("ffprobe")
            ffmpeg = shutil.which("ffmpeg")
            video = path.with_name("main_rgb.mp4")
            probe = json.loads(
                subprocess.check_output(
                    [
                        ffprobe,
                        "-v",
                        "error",
                        "-select_streams",
                        "v:0",
                        "-show_entries",
                        "stream=width,height,avg_frame_rate,nb_frames",
                        "-of",
                        "json",
                        str(video),
                    ]
                )
            )["streams"][0]
            check(
                key + ":video_metadata",
                probe["width"] == 256
                and probe["height"] == 256
                and probe["avg_frame_rate"] == "20/1"
                and int(probe["nb_frames"]) == len(frames),
            )
            decoded = subprocess.run([ffmpeg, "-v", "error", "-i", str(video), "-f", "null", "-"], capture_output=True)
            check(key + ":video_decode", decoded.returncode == 0)
            cases.append(
                dict(
                    batch=batch,
                    relative=str(path.parent.relative_to(root)),
                    report=report,
                    failure_reason=None
                    if entry["success"]
                    else (
                        "optimizer_rejected"
                        if any(not f["optimizer"]["feasible"] for f in frames)
                        else "task_failed_after_execution"
                    ),
                )
            )
    original = root / "artifacts/original_sources/tasktol/simulation/preset_runtime.py"
    check(
        "original_preset_backend_restored",
        (project / "tasktol/simulation/preset_runtime.py").read_bytes() == original.read_bytes(),
    )
    source = (project / "tasktol/simulation/libero.py").read_text()
    check(
        "no_custom_physics_or_torque",
        all(
            x not in source
            for x in ("mj_step(", ".run_controller(", ".data.ctrl[", "PresetSimulationBackend", "LiberoJointActuation")
        ),
    )
    check("native_env_step_called", "self.env.step(native_action.tolist())" in source)
    validation = dict(
        checks=len(checks),
        passed=sum(x["passed"] for x in checks),
        failed=[x for x in checks if not x["passed"]],
        details=checks,
    )
    (root / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    rows = []
    for batch in batches:
        subset = [c for c in cases if c["batch"] == batch]
        rows.append(
            dict(
                batch=batch,
                cases=len(subset),
                successes=sum(c["report"]["task_success"] for c in subset),
                frames=sum(c["report"]["frames"] for c in subset),
                solver_calls=sum(c["report"]["solver_calls"] for c in subset),
                native_env_steps=sum(c["report"]["native_env_steps"] for c in subset),
                failure_reasons=dict(Counter(c["failure_reason"] for c in subset if c["failure_reason"])),
            )
        )
    (root / "summary.json").write_text(
        json.dumps(
            dict(
                batches=rows,
                validation=validation["failed"],
                cases=[
                    dict(
                        batch=c["batch"],
                        path=c["relative"],
                        task=c["report"]["task"],
                        success=c["report"]["task_success"],
                        reason=c["failure_reason"],
                    )
                    for c in cases
                ],
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    lines = [
        "# LIBERO 原生闭环迁移验证 · 2026-10-20",
        "",
        "执行链：独立名义目标 → 原有 TaskTol/IPOPT → 绝对关节目标 → 原生 JOINT_POSITION 输入编码 → LIBERO env.step → 原生观测与成功判定。",
        "",
        "原 TaskTol preset_runtime.py 已恢复。当前 LIBERO 路径没有自定义物理循环、额外速度前馈、重力补偿或手工力矩写入。原生关节控制器的算法、夹爪和步进流程保持原实现；其动作输出缩放配置为各关节跨度，保证绝对 IPOPT 目标可无截断编码。正式批次保持原生默认 kp=50、阻尼比 1。",
        "",
        "| 批次 | 任务成功 | 录制帧 / IPOPT 调用 | 原生 env.step 调用 |",
        "|---|---:|---:|---:|",
    ]
    lines += [
        f"| {r['batch']} | {r['successes']}/{r['cases']} | {r['frames']} / {r['solver_calls']} | {r['native_env_steps']} |"
        for r in rows
    ]
    lines += [
        "",
        f"验证检查：{validation['passed']}/{validation['checks']} 通过。失败轨迹保留用于诊断，不计入成功示范。",
        "",
        "5° 是显式诊断边界，用于验证阶段容差接口，未经过数值容差搜索，不是真值或预测精度评估。ENG/DIS 以及无标注区间维持 000。阶段边界沿用源代表示范的冻结标注，尚未重新视觉标注 IPOPT 实际接触事件。",
        "",
        "名义输入来自已处理示范机器人状态与原始 OSC 动作离线解码后冻结的目标序列。执行时不调用 OSC，不把实际 FK 反馈重新写回独立名义分支。零旋转命令的初始保持方向以首个 FK 近似；源示范没有保存全部控制器内部变量，因此不宣称逐帧还原原始控制器历史。",
        "",
        "本次是示范参考驱动的 IPOPT 执行闭环，未连接训练模型服务器。客户端代码位于 tasktol/benchmark/libero.py；可复用的单步接口位于 tasktol/simulation/libero.py。",
        "",
        "## 默认配置下失败的名义示范",
        "",
        "| 任务集 / task_id | 任务 | 分类 | 日志 |",
        "|---|---|---|---|",
    ]
    for c in cases:
        if c["batch"] != "native_nominal40" or c["report"]["task_success"]:
            continue
        r = c["report"]
        lines.append(
            f"| {r['task']['suite']} / {r['task']['task_id']} | {r['task']['name']} | {c['failure_reason']} | [report]({c['relative']}/report.json) |"
        )
    lines += [
        "",
        "## 验证范围",
        "",
        "原 TaskTol 环境回归：37 passed、1 skipped（跳过锁定 3.2.3 的 LIBERO 集成项）；LIBERO 隔离环境原生集成：1 passed、1 deselected。集成检查包括原生步进调用次数、关节目标一致性、工具坐标、20Hz 时序、单侧容差、ENG 门控及关闭容差后的任务清理。",
        "",
        "MuJoCo 3.2.3、robosuite 1.4.1、cyipopt 1.7.0。资产直接引用原 openpi/third_party/libero 路径。全部轨迹、逐帧求解日志、初始/终止 PNG 和主视角 20Hz MP4 位于 results/native_*。旧 400 个 OSC 视频与冻结 mask 均未覆盖。",
        "",
        "[逐任务视频报告](report.html) · [机器可读汇总](summary.json) · [完整验证记录](validation.json)",
        "",
    ]
    (root / "RESULTS.md").write_text("\n".join(lines))
    cards = []
    for batch in batches:
        parts = [f"<h2>{html.escape(batch)}</h2>"]
        for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
            parts.append(f"<h3>{suite}</h3>")
            for c in sorted(
                (c for c in cases if c["batch"] == batch and c["report"]["task"]["suite"] == suite),
                key=lambda c: c["report"]["task"]["task_id"],
            ):
                r = c["report"]
                path = c["relative"]
                good = r["task_success"]
                cls = "good" if good else "bad"
                parts.append(
                    f'<details><summary class="{cls}">{r["task"]["task_id"]:02d} · {html.escape(r["task"]["name"])} · {"成功" if good else "失败"}</summary><p>{r["frames"]} 帧；IPOPT 可行 {r["feasible_frames"]} 帧；原生步进 {r["native_env_steps"]} 次；关节目标编码误差 {r["native_joint_goal_max_error_rad"]:.2e} rad。</p><video controls preload="none" poster="{path}/initial.png" src="{path}/main_rgb.mp4"></video><p><a href="{path}/report.json">报告</a> · <a href="{path}/trajectory.json">完整轨迹</a> · <a href="{path}/solver_trace.jsonl">逐帧求解</a> · <a href="{path}/final.png">终止图像</a></p></details>'
                )
        cards.extend(parts)
    page = (
        '<!doctype html><html lang="zh"><meta charset="utf-8"><title>LIBERO × IPOPT 原生闭环</title><style>body{font:15px system-ui;background:#10151e;color:#e4eaf4;margin:32px auto;max-width:1150px;padding:0 20px}h1,h2{color:#9ac8ff}p{line-height:1.7}details{background:#1b2432;margin:10px 0;padding:14px;border-radius:8px}summary{cursor:pointer;overflow-wrap:anywhere}.good{color:#84deae}.bad{color:#ffad9c}video{width: min(100%,512px);background:black}a{color:#9ac8ff}</style><h1>LIBERO × TaskTol/IPOPT</h1><p>IPOPT 仅生成关节目标；执行完整复用 LIBERO 原生 JOINT_POSITION 与 env.step。默认名义批次 33/40 成功；5° 诊断边界批次见下方实际结果。失败保留，不混入成功示范。</p><p>5° 不作为数值容差真值；源阶段时间线尚未重新视觉标注。<a href="RESULTS.md">详细说明</a> · <a href="validation.json">验证记录</a></p>'
        + "".join(cards)
        + "</html>"
    )
    (root / "report.html").write_text(page)
    print(
        json.dumps(
            dict(batches=rows, checks=validation["checks"], passed=validation["passed"], failed=validation["failed"]),
            ensure_ascii=False,
            indent=2,
        )
    )
    if validation["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
