"use strict";

// The three signed gauges and fixed-XYZ/RPY chart follow the MuJoCo Studio.
// Values come from the solver's saved accepted release, not tracking error.
const releaseBuildDerived = buildDerivedFrames;
buildDerivedFrames = function () {
  releaseBuildDerived();
  const enabled=Boolean(state.payload?.episode?.benchmark_result?.tolerance_enabled);
  state.frames.forEach(f=>{f._release=LiberoRelease.values(f,enabled);});
  const values=state.frames.map(f=>f._release);
  state.chartCache.toleranceError=[0,1,2].map(i=>values.map(v=>v.degrees[i]));
  state.chartCache.tolerancePositive=[0,1,2].map(i=>values.map(v=>enabled ? v.limits[i][1] : null));
  state.chartCache.toleranceNegative=[0,1,2].map(i=>values.map(v=>enabled ? -v.limits[i][0] : null));
  $('.release-chart').hidden=!enabled;
};

const releaseInspector=renderInspector;
renderInspector=function () {
  releaseInspector();
  const f=state.frames[state.frameIndex];if(!f)return;
  const r=state.payload?.episode?.benchmark_result||{}, v=f._release;
  $('#releaseFrame').textContent=`FRAME ${String(state.frameIndex).padStart(3,'0')} · ${formatTime(state.frameIndex/20)}`;
  $('#releaseContext').textContent=r.tolerance_enabled ? `${f.semantic_subtask_label||'任务'} · ${f.phase} · mask ${v.semantic} → 有效 ${v.effective}${f.heading_guarded ? ' · heading退化保护' : ''}` : '本回合容差关闭';
  const bars=$('#releaseBars');bars.replaceChildren();
  ['x','y','z'].forEach((axis,i)=>{
    const row=document.createElement('div');row.className=`tol-row ${axis}`;
    const label=document.createElement('span');label.textContent=`R${axis}`;
    const track=document.createElement('div');track.className='tol-track';
    const interval=document.createElement('i');
    // 45 degrees per half-track keeps the30-degree box readable and leaves
    // room for inherited stage offsets outside currently strict axes.
    interval.style.left=`${50-v.limits[i][0]/45*48}%`;
    interval.style.width=`${(v.limits[i][0]+v.limits[i][1])/45*48}%`;
    const marker=document.createElement('b');marker.style.left=`${clamp(50+(v.degrees[i]??0)/45*48,1,99)}%`;
    marker.hidden=!v.available;
    track.append(interval,marker);
    const text=document.createElement('span');text.className='release-number';
    text.textContent=v.available ? `${v.degrees[i]>=0?'+':''}${format(v.degrees[i],2)}°` : '—';
    const mode=document.createElement('small');
    mode.textContent=v.effective[i]==='1' ? `[-${format(v.limits[i][0],0)}, +${format(v.limits[i][1],0)}]°` : '严格 · 不新增释放';
    row.append(label,track,text,mode);bars.append(row);
  });
  const status=$('#releaseStatus');
  status.textContent=!r.tolerance_enabled ? '无释放记录' : v.held ? `保持上一接受参考 · 连续失败 ${f.consecutive_solver_failures}/${f.max_consecutive_solver_failures}${v.rejected?' · 本帧失败候选未采用':''}` : v.available ? '已接受优化释放 · 与视频同一动作帧' : '本帧无已接受释放记录';
  status.className=v.held?'release-held':'';
};
