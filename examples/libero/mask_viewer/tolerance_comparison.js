"use strict";
// A second executed rollout accompanies the fixed no-tolerance master timeline.
const toleranceVideo=$('#toleranceVideo');
const nominalWristElement=$('#nominalWristVideo'),toleranceWristElement=$('#toleranceWristVideo');
function syncWristVideos(force=false){
  LiberoVideoSync.sync(nominalWristElement,mainVideo,state.speed,force);
  LiberoVideoSync.sync(toleranceWristElement,mainVideo,state.speed,force);
}
function loadWrist(video,path){
  video.pause();video.removeAttribute('poster');
  if(path)video.src=path;else video.removeAttribute('src');
  video.playbackRate=state.speed;video.load();
}
const recoveryLabel=document.createElement('label');
recoveryLabel.innerHTML='<input id="useRecovery" type="checkbox" checked> 右侧优先显示姿态恢复复测';
$('.comparison-toolbar').appendChild(recoveryLabel);
function comparisonFor(episode){return episode&&$('#useRecovery').checked&&episode.recovery?episode.recovery:episode?.tolerance;}
function activeComparison(){return comparisonFor(state.episode);}
function usingRecovery(){return !!state.episode?.recovery&&activeComparison()===state.episode.recovery;}
function activeComparisonFrame(frame){return usingRecovery()?frame.recovery:frame.tolerance;}

function toleranceAvailable(){return !!activeComparison();}
function syncToleranceVideo(force=false){
  if(!toleranceAvailable())return;
  toleranceVideo.playbackRate=state.speed;
  const t=activeComparison(),stopped=mainVideo.currentTime>=t.frames/20;
  LiberoVideoSync.sync(toleranceVideo,mainVideo,state.speed,force);
  if(stopped)toleranceVideo.pause();
  $('#toleranceVideoState').textContent=stopped&&!t.result.success?`已停止 · ${t.result.execution_rejected?'求解拒绝':'任务未完成'}`:'';
}
const pairConfigureVideos=configureVideos;
configureVideos=function(){
  pairConfigureVideos();toleranceVideo.pause();
  const t=activeComparison();
  loadWrist(nominalWristElement,state.episode.wrist_video_path);
  loadWrist(toleranceWristElement,t?.wrist_video);
  $('#toleranceWristCaption').textContent=t?'标注容差 · 腕部视角 · 20 Hz':'本条无容差回放 · 腕部视角';
  if(t){toleranceVideo.src=t.video;toleranceVideo.poster=t.poster;toleranceVideo.load();}
  else{toleranceVideo.removeAttribute('src');toleranceVideo.removeAttribute('poster');toleranceVideo.load();}
  $('#toleranceCaption').textContent=t?`${usingRecovery()?'30°＋松爪恢复':'原30°容差'} · ${t.result.success?'成功':t.result.execution_rejected?'求解拒绝':'任务未完成'} · ${t.result.executed_frames}/${t.result.planned_frames}帧`:'30°容差 · 尚未完成';
  $('#episodeScope').textContent='从左到右：无容差主视角、容差主视角、无容差腕部、容差腕部。恢复只修改旋转增量，保留平移、原夹爪、20Hz原速和默认IPOPT。复测范围为此前失败的52条长任务。';
  $('#comparisonStatus').textContent=`${state.episode.path} · 四视频同帧号 / 20Hz · 提前停止时容差双视角保留最后画面`;
};
const pairPauseVideos=pauseVideos;
pauseVideos=function(){pairPauseVideos();toleranceVideo.pause();nominalWristElement.pause();toleranceWristElement.pause();};
const pairSeekFrame=seekFrame;
seekFrame=function(i){pairSeekFrame(i);syncToleranceVideo(true);syncWristVideos(true);};
togglePlayback=async function(){
  if(!state.frames.length)return;if(!mainVideo.paused){pauseVideos();return;}
  if(mainVideo.ended)seekFrame(0);syncToleranceVideo(true);syncWristVideos(true);
  const videos=[mainVideo,nominalWristElement,...(toleranceAvailable()&&mainVideo.currentTime<activeComparison().frames/20?[toleranceVideo,toleranceWristElement]:[])].filter(v=>v.getAttribute('src'));
  try{await Promise.all(videos.map(v=>{v.playbackRate=state.speed;return v.play();}));$('#playButton').textContent='Ⅱ';}
  catch(e){pauseVideos();showError(`四视频播放失败：${e.message}`);}
};
// The existing scheduled animation callback resolves this global on its next tick.
animationLoop=function(){
  if(state.frames.length&&!mainVideo.paused){syncToleranceVideo();syncWristVideos();updateFrame(videoFrameIndex(mainVideo.currentTime));$('#timeline').value=mainVideo.currentTime;$('#currentTime').textContent=formatTime(mainVideo.currentTime);}
  requestAnimationFrame(animationLoop);
};
mainVideo.addEventListener('seeking',()=>{syncToleranceVideo(true);syncWristVideos(true);});
mainVideo.addEventListener('seeked',()=>{syncToleranceVideo(true);syncWristVideos(true);});
mainVideo.addEventListener('pause',()=>{toleranceVideo.pause();nominalWristElement.pause();toleranceWristElement.pause();});
mainVideo.addEventListener('ratechange',()=>{toleranceVideo.playbackRate=nominalWristElement.playbackRate=toleranceWristElement.playbackRate=mainVideo.playbackRate;});
for(const video of [nominalWristElement,toleranceWristElement]){
  video.addEventListener('loadedmetadata',()=>syncWristVideos(true));
  video.addEventListener('error',()=>{if(video.getAttribute('src'))showError('腕部视频加载失败。');});
}
toleranceVideo.addEventListener('loadedmetadata',()=>syncToleranceVideo(true));
toleranceVideo.addEventListener('error',()=>{if(toleranceVideo.getAttribute('src'))showError('容差视频加载失败。');});
renderSharedPrediction=function(){
  const t=activeComparison(),m=state.dataset.confirmed_masks||[];
  $('#sharedScope').textContent='确认语义 mask · 上限30°';
  $('#sharedMaskSummary').textContent=m.map((s,i)=>`子任务${i+1}：PRE ${s.APP} / POST ${s.EXEC}`).join('；')+(t?`。本条容差 ${t.result.success?'成功':t.result.execution_rejected?'求解拒绝':'任务未完成'}；无容差成功。`:'。本条容差尚未完成。');
  if(state.episode.recovery)$('#sharedMaskSummary').textContent+=` 原容差失败 → 恢复复测${state.episode.recovery.result.success?'成功':'失败'}。`;
  renderReasoning();
};
const reconstructionReasoning=renderReasoning;
renderReasoning=function(){
  reconstructionReasoning();const p=document.createElement('p');
  p.textContent='原容差回放保留同一冻结动作，不再次修正参考。语义子任务身份采用已保存的逐示范计划；阶段由当前手指反馈在20Hz下判定，连续3帧稳定。ENG/DIS禁止新增释放；POST000仍可继承PRE偏差。Goal/03开抽屉直至抓碗归PRE010，POST000。';$('#reasoningContent').prepend(p);
  const recovery=state.episode.recovery?.result.rotation_recovery;
  if(recovery){const note=document.createElement('p');note.textContent='松爪确认后，以45°/s纠偏上限和45°/s合成上限，用平滑SO(3)曲线恢复原始名义姿态；参考阈值0.5°，2秒仅提示；下一次闭爪即停止。原动作已超速的帧不附加恢复、不截断原动作。';$('#reasoningContent').appendChild(note);
    const table=document.createElement('table');table.innerHTML='<thead><tr><th>循环</th><th>起始偏差/°</th><th>耗时/s</th><th>参考恢复</th><th>下次闭爪残差/°</th></tr></thead><tbody></tbody>';
    for(const e of recovery.events){const tr=document.createElement('tr');for(const v of [e.cycle,e.initial_offset_deg.toFixed(2),format(e.elapsed_s,2),e.completed?'完成':e.interrupted_by?'闭爪中断':e.rejected_during_recovery?'求解拒绝':'未完成',format(e.next_grasp_accepted_residual_deg,2)]){const td=document.createElement('td');td.textContent=v;tr.appendChild(td);}table.tBodies[0].appendChild(tr);}$('#reasoningContent').appendChild(table);}

};
const baselineInspector=renderInspector;
renderInspector=function(){
  baselineInspector();const f=state.frames[state.frameIndex];if(!f)return;
  const t=activeComparisonFrame(f),done=activeComparison();
  const values=v=>v.map(x=>format(x,2)).join(', ');
  if(!t){$('#tolerancePose').textContent=done?'容差轨迹已停止；当前仅继续显示无容差双视角。':'容差 rollout 尚未完成。';return;}
  $('#tolerancePose').textContent=`30°容差 · ${t.phase} · 语义子任务 ${t.semantic_subtask_index===null?'外部':t.semantic_subtask_index+1}\nmask ${t.semantic_mask} → 有效 ${t.effective_mask}${t.aborted?' · 求解拒绝，未步进':''}\n优化释放 XYZ [°]：${values(t.rotation_release_deg)}\n实际阶段坐标 XYZ [°]：${values(t.actual_rotation_release_deg)}\n启用轴实际越界 XYZ [°]：${values(t.actual_violation_deg)}\n对参考位置 XYZ [mm]：${values(t.position_error_xyz_mm)}\n对参考旋转 XYZ [°]：${values(t.rotation_error_xyz_deg)}\n对原state：${format(t.source_next_position_error_mm,3)} mm / ${format(t.source_next_rotation_error_deg,3)}°\n实际→优化姿态：${format(t.tracking_optimized_rotation_error_deg,3)}°\n求解 ${format(t.optimizer.elapsed_ms,2)} ms · ${t.optimizer.status}`;
};
const comparisonInspector=renderInspector;
renderInspector=function(){comparisonInspector();const f=state.frames[state.frameIndex],t=f&&activeComparisonFrame(f),r=t?.rotation_recovery;
  if(r)$('#tolerancePose').textContent+=`\n姿态恢复：${r.active?'执行中':'未激活'}${r.progress===null?'':` · ${(r.progress*100).toFixed(1)}%`}\n本帧纠偏 ${format(r.correction_angle_deg,3)}° · 耗时 ${format(r.elapsed_s,2)}s\n对原始名义姿态：优化 ${format(r.accepted_original_error_after_deg,2)}° / 实际 ${format(r.actual_original_error_after_deg,2)}°${r.over_warning_time?'\n已超过2秒提示阈值':''}`;
};
const baselineMetadata=renderMetadata;
renderMetadata=function(){
  baselineMetadata();const c=$('#episodeMetadata'),t=activeComparison();
  for(const item of c.querySelectorAll('.metadata-item')){
    const label=item.querySelector('span'),value=item.querySelector('strong');
    if(label?.textContent==='容差')value.textContent=usingRecovery()?'第1、3格基线关闭；第2、4格30°＋松爪恢复':'第1、3格基线关闭；第2、4格原30°';
    if(label?.textContent==='右侧原始视频')label.textContent='左侧原始视频';
    if(label?.textContent==='双视频对齐')label.textContent='视频时间轴';
  }
  c.appendChild(metadataItem('视频顺序',`第1、3格无容差；第2、4格${usingRecovery()?'30°＋松爪姿态恢复':'原30°容差'}；前两格主视角，后两格腕部视角。轨迹图默认显示无容差基线，容差逐帧详情和CSV单独展示。`));
  c.appendChild(metadataItem('右列容差',t?`${t.result.success?'成功':'失败'}；执行${t.result.executed_frames}/${t.result.planned_frames}帧；30°是边界上限，非强制偏转角度。`:'等待本条运行完成'));
  if(t){const r=t.result;c.appendChild(metadataItem('容差对原state均值',`${format(r.tracking.source_position_mm?.mean,3)} mm / ${format(r.tracking.source_rotation_deg?.mean,3)}°`));
    const a=document.createElement('a');a.href=t.report;a.textContent='容差执行报告';a.target='_blank';c.appendChild(a);
    const b=document.createElement('a');b.href=t.trajectory;b.textContent='容差逐帧轨迹';b.target='_blank';c.appendChild(b);}
};
const baselineRenderFolders=renderFolderList;
renderFolderList=function(){baselineRenderFolders();const c=window.LIBERO_CATALOG.counts;$('#folderSummary').textContent=`原30°：${c.tolerance_success||0}/400成功 · 失败长任务恢复复测：${c.recovery_completed||0}/52 · 新增成功${c.recovery_success||0}`;};
const baselineSelect=selectEpisode;
selectEpisode=async function(i){await baselineSelect(i);if(state.episode){const t=activeComparison();$('#syncStatus').textContent=t?`${usingRecovery()?'恢复复测':'原30°'} · ${t.result.success?'SUCCESS':t.result.execution_rejected?'REJECTED':'FAILED'}`:'30° · PENDING';$('#resultValue').textContent=t?`基线成功 / 容差${t.result.success?'成功':'失败'}`:'基线成功 / 容差待运行';$('#resultValue').className=t&&!t.result.success?'failure':'success';}};
renderEpisodeList=function(){
  const c=$('#episodeList'),episodes=state.dataset?.episodes||[];c.innerHTML='';$('#episodeCount').textContent=episodes.length;
  episodes.forEach((e,i)=>{const t=comparisonFor(e),ok=t?.result.success,b=document.createElement('button');b.className=`episode-item${state.episode===e?' active':''}`;b.dataset.demo=e.path;
    b.innerHTML=`<span class="episode-number">${String(i+1).padStart(2,'0')}</span><span class="episode-copy"><strong>${escapeHtml(e.path)}</strong><span>基线成功 · 容差${t?(ok?'成功':t.result.execution_rejected?'求解拒绝':'任务失败'):'待运行'}${e.recovery&&$('#useRecovery').checked?' · 恢复复测':''}<br>${e.frames}帧 · ${format(e.frames/20,1)}s</span></span><i class="episode-result" style="background:${t?(ok?'#48d9aa':'#ff6b72'):'#65717c'}"></i>`;
    b.onclick=()=>selectEpisode(i);c.appendChild(b);});
};
$('#downloadCsv').textContent='导出基线 CSV';
const toleranceCsv=document.createElement('button');toleranceCsv.textContent='导出容差 CSV';
toleranceCsv.onclick=()=>{
  const fields=['frame','time_s','phase','semantic_subtask','semantic_mask','effective_mask','aborted','source_position_mm','source_rotation_deg','reference_position_mm',
    'release_x_deg','release_y_deg','release_z_deg','actual_release_x_deg','actual_release_y_deg','actual_release_z_deg',
    'position_error_x_mm','position_error_y_mm','position_error_z_mm','rotation_error_x_deg','rotation_error_y_deg','rotation_error_z_deg',
    'qtarget1','qtarget2','qtarget3','qtarget4','qtarget5','qtarget6','qtarget7','solver_ms'];
  const rows=state.frames.filter(f=>activeComparisonFrame(f)).map(f=>{const t=activeComparisonFrame(f);return[f.index,f.time_s,t.phase,t.semantic_subtask_index,t.semantic_mask,t.effective_mask,t.aborted,
    t.source_next_position_error_mm,t.source_next_rotation_error_deg,t.tracking_position_error_mm,...t.rotation_release_deg,...t.actual_rotation_release_deg,
    ...t.position_error_xyz_mm,...t.rotation_error_xyz_deg,...t.joint_target_rad,t.optimizer.elapsed_ms].join(',');});
  const u=URL.createObjectURL(new Blob([fields.join(',')+'\n'+rows.join('\n')+'\n'],{type:'text/csv;charset=utf-8'})),a=document.createElement('a');a.href=u;a.download=`${state.episode.key.replaceAll('/','_')}_${usingRecovery()?'release_recovery30':'tolerance30'}.csv`;a.click();setTimeout(()=>URL.revokeObjectURL(u),1000);
};
$('#downloadCsv').after(toleranceCsv);
$('#useRecovery').addEventListener('change',()=>{pauseVideos();const index=state.frameIndex;configureVideos();seekFrame(index);renderSharedPrediction();renderEpisodeList();renderMetadata();renderInspector();const t=activeComparison();$('#syncStatus').textContent=`${usingRecovery()?'恢复复测':'原30°'} · ${t?.result.success?'SUCCESS':'FAILED'}`;$('#resultValue').textContent=`基线成功 / ${usingRecovery()?'恢复':'原容差'}${t?.result.success?'成功':'失败'}`;$('#resultValue').className=t?.result.success?'success':'failure';});

loadDatasets();
