"use strict";

// This extension reads the raw, already recorded comparison trace. Older
// viewer payloads contain default-zero conversions and omit directional bounds.
// Never use those conversions as evidence of an accepted release.
(function () {
  const panel = document.createElement('section');
  panel.className = 'panel dataset-release-panel';
  panel.setAttribute('aria-label', '数据集实时容差释放');
  panel.innerHTML = `<div class="panel-title-row"><div><span class="section-index">SO(3)</span><h2>实时容差释放</h2></div><span id="datasetReleaseFrame" class="dataset-release-caption"></span></div>
    <div class="dataset-release-content"><div><p id="datasetReleaseContext" class="dataset-release-caption"></p><div id="datasetReleaseBars" class="tolerance-bars dataset-release-bars"></div></div>
    <div class="dataset-release-key"><p>右侧回放的已接受优化释放 · 固定XYZ/RPY · deg</p><p>阴影为有效轴边界；严格轴可继承已有偏差。</p><p id="datasetReleaseStatus" role="status"></p></div></div>
    <div class="chart-card dataset-release-chart"><div><strong>容差释放时序</strong><span>Rx / Ry / Rz · 实线为已接受值，虚线为有效边界 · 点击跳转</span></div><canvas id="datasetReleaseChart" aria-label="容差释放时序曲线"></canvas></div>`;
  $('.video-panel').after(panel);

  const cache = new Map();
  let data = [], loading = false, error = '', generation = 0, controller;
  let source = null;
  const mode = () => usingRecovery() ? '30°＋松爪姿态恢复' : '原30°容差';

  function drawReleaseChart() {
    const canvas = $('#datasetReleaseChart');
    if (!state.frames.length) return;
    const series = [];
    for (let axis = 0; axis < 3; axis++) {
      series.push({values: state.frames.map((_, i) => data[i]?.degrees[axis] ?? null), color: COLORS[axis], width: 1.8});
      for (const direction of [0, 1]) series.push({
        values: state.frames.map((_, i) => data[i]?.boundsAvailable ? data[i].limits[axis][direction] * (direction ? 1 : -1) : null),
        color: COLORS[axis], width: 1, dash: [3, 4], alpha: .42,
      });
    }
    drawLineChart(canvas, series);
  }

  function renderRelease() {
    const fps = Number(state.dataset?.frequency_hz || 20);
    const index = Math.min(state.frameIndex, data.length - 1);
    const value = data[index];
    const stopped = state.frameIndex >= data.length && data.length > 0;
    const showValue = source && !loading && !error && value;
    $('#datasetReleaseFrame').textContent = showValue ? `FRAME ${index} · ${formatTime(index / fps)}${stopped ? ' · 右侧记录已结束' : ''}` : '';
    $('#datasetReleaseContext').textContent = !source ? '本条暂无容差回放' : `${mode()}${showValue ? ` · ${value.phase} · 子任务 ${value.semantic_subtask_index ?? '—'} · mask ${value.semantic} → 有效 ${value.effective}${value.heading_guarded ? ' · heading退化保护' : ''}` : ''}`;
    const bars = $('#datasetReleaseBars');
    bars.replaceChildren();
    const scale = Math.max(45, ...data.flatMap(v => [...v.degrees.filter(Number.isFinite).map(Math.abs), ...v.limits.flat()])) * 1.05;
    ['xF', 'yF', 'zF'].forEach((axis, i) => {
      const row = document.createElement('div'); row.className = `tol-row ${['x', 'y', 'z'][i]}`;
      const label = document.createElement('span'); label.textContent = axis;
      const track = document.createElement('div'); track.className = 'tol-track';
      const interval = document.createElement('i');
      interval.style.left = `${50 - (showValue ? value.limits[i][0] : 0) / scale * 48}%`;
      interval.style.width = `${(showValue ? value.limits[i][0] + value.limits[i][1] : 0) / scale * 48}%`;
      const marker = document.createElement('b');
      marker.hidden = !showValue || !value.available;
      marker.style.left = `${clamp(50 + (showValue ? value.degrees[i] ?? 0 : 0) / scale * 48, 1, 99)}%`;
      track.append(interval, marker);
      const number = document.createElement('span'); number.className = 'dataset-release-number';
      number.textContent = showValue && value.available ? `${value.degrees[i] >= 0 ? '+' : ''}${format(value.degrees[i], 2)}°` : '—';
      const boundary = document.createElement('small');
      boundary.textContent = !showValue || !value.boundsAvailable ? '无边界记录' : value.effective[i] === '1' ? `[-${format(value.limits[i][0], 0)}, +${format(value.limits[i][1], 0)}]°` : '严格 · 不新增释放';
      row.append(label, track, number, boundary); bars.append(row);
    });
    const status = $('#datasetReleaseStatus');
    status.textContent = loading ? '正在读取本条已保存的求解记录…' : error ? `释放记录加载失败：${error}` : !source ? '原始图像和无容差基线不产生容差释放。' : !value ? '未提供已接受释放记录。' : value.rejected ? `求解拒绝 · 候选未执行${value.available ? ' · 保留上一接受释放' : ' · 无上一接受释放记录'}${stopped ? '；右侧视频已停止。' : ''}` : stopped ? '右侧视频已停止，保留最后已接受释放；后续曲线留空。' : value.available ? '已接受优化释放 · 与右侧视频同一动作帧' : '本帧无已接受释放记录，显示—。';
    status.classList.toggle('dataset-release-held', Boolean(showValue && value.rejected));
  }

  async function loadRelease() {
    const ticket = ++generation;
    controller?.abort();
    source = activeComparison(); data = []; error = ''; loading = Boolean(source);
    renderRelease(); drawReleaseChart();
    if (!source) return;
    const url = source.trajectory;
    try {
      if (cache.has(url)) data = cache.get(url);
      else {
        controller = new AbortController();
        const response = await fetch(url, {signal: controller.signal});
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const rows = await response.json();
        if (!Array.isArray(rows)) throw new Error('轨迹不是逐帧数组');
        const parsed = DatasetRelease.series(rows);
        if (ticket !== generation) return;
        cache.set(url, parsed);
        if (cache.size > 6) cache.delete(cache.keys().next().value);
        data = parsed;
      }
    } catch (e) {
      if (ticket !== generation || e.name === 'AbortError') return;
      error = e.message;
    }
    if (ticket !== generation) return;
    loading = false; renderRelease(); drawReleaseChart();
  }

  const previousConfigure = configureVideos;
  configureVideos = function () { previousConfigure(); void loadRelease(); };
  const previousInspector = renderInspector;
  renderInspector = function () { previousInspector(); renderRelease(); };
  const previousCharts = drawCharts;
  drawCharts = function () { previousCharts(); drawReleaseChart(); };
  $('#datasetReleaseChart').addEventListener('pointerdown', event => {
    const canvas = event.currentTarget;
    if (!canvas._plot || !state.frames.length) return;
    const rect = canvas.getBoundingClientRect();
    const ratio = clamp((event.clientX - rect.left - canvas._plot.left) / canvas._plot.width, 0, 1);
    seekFrame(Math.round(ratio * (state.frames.length - 1)));
  });
  renderRelease();
  if (state.episode) void loadRelease();
})();
