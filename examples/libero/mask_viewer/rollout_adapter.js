"use strict";
// Current reconstructed shared-servo data only. Historical recordings never load.
renderFolderList=function(){
  const suites=[...new Set(state.datasets.map(d=>d.suite))],select=$('#folderSelect');select.innerHTML='';
  for(const suite of suites){const tasks=state.datasets.filter(d=>d.suite===suite),o=document.createElement('option');o.value=suite;o.textContent=`${suite} · ${tasks.reduce((n,d)=>n+d.episodes.length,0)}/100 成功`;select.appendChild(o);}
  select.value=state.folder||suites[0];const c=window.LIBERO_CATALOG.counts;
  $('#folderSummary').textContent=`重构 Rollout · ${c.episodes}/400 成功 · ${c.tasks_complete}/40 任务已满`;
};
renderDatasetList=function(){
  const c=$('#datasetList');c.innerHTML='';const query=$('#taskSearch').value.trim().toLowerCase();
  for(const suite of [...new Set(state.datasets.map(d=>d.suite))]){
    const items=state.datasets.map((d,i)=>({d,i})).filter(({d})=>d.suite===suite&&(!query||`${d.task_name} ${d.display_name} ${d.suite}`.toLowerCase().includes(query)));
    if(!items.length)continue;const group=document.createElement('details');group.className='suite-tree';group.open=!!query||suite===state.folder;
    const heading=document.createElement('summary');heading.textContent=suite;const sub=document.createElement('span');sub.textContent=`${items.reduce((n,{d})=>n+d.episodes.length,0)} 条成功轨迹`;heading.appendChild(sub);group.appendChild(heading);
    for(const{d,i}of items){const b=document.createElement('button');b.className=`dataset-item${state.dataset===d?' active':''}`;b.dataset.taskIndex=i;
      const title=document.createElement('strong');title.textContent=`${String(d.task_id).padStart(2,'0')} · ${d.display_name}`;
      const count=document.createElement('span');count.textContent=`${d.episodes.length}/10 成功${d.episodes.length?` · 已完成${d.completed_attempts}条重构`:d.completed_attempts?' · 已尝试，未成功':' · 尚未开始录制'}`;
      b.append(title,count);b.onclick=()=>selectDataset(i);group.appendChild(b);}c.appendChild(group);
  }
};
const populatedSelectDataset=selectDataset;
selectDataset=async function(index,requestedEpisode=null){
  const d=state.datasets[index];if(!d)return;
  if(d.episodes.length){await populatedSelectDataset(index,requestedEpisode);return;}
  ++state.loadGeneration;pauseVideos();mainVideo.removeAttribute('src');mainVideo.load();
  state.dataset=d;state.folder=d.suite;state.episode=null;state.payload=null;state.frames=[];state.chartCache=null;
  renderFolderList();renderDatasetList();renderEpisodeList();$('#content').classList.add('hidden');$('#emptyState').classList.remove('hidden');
  $('#emptyState h2').textContent=`${d.suite} / ${String(d.task_id).padStart(2,'0')} · ${d.completed_attempts?'尚无成功录制':'尚未开始录制'}`;
  $('#emptyState p').textContent=d.completed_attempts?`已完成 ${d.completed_attempts} 条来源示范的重构尝试，暂无完整成功轨迹。批量录制当前已停止。`:'该任务尚未进行重构录制（已完成尝试0条）。此前批量任务中途停止，尚未处理到此任务；这不表示执行失败。';$('#taskName').textContent=d.task_name;$('#instructionText').textContent=d.display_name;
  $('#sceneName').textContent='RECONSTRUCTED ROLLOUT · PENDING';for(const id of ['seedValue','frameCountValue','resultValue'])$('#'+id).textContent='—';
  const u=new URL(location.href);u.searchParams.set('dataset',d.relative_path);u.searchParams.delete('episode');history.replaceState(null,'',u);
};
const recordedConfigureVideos=configureVideos;
configureVideos=function(){recordedConfigureVideos();$('#episodeScope').textContent='重构成功 Rollout · 默认 IPOPT → 共享关节插值 / Tracker → Panda 位置伺服。20Hz 原速、原夹爪、容差关闭。视频为动作前观测，误差为动作后状态。';};
const recordedSelectEpisode=selectEpisode;
selectEpisode=async function(index){await recordedSelectEpisode(index);if(state.episode)$('#syncStatus').textContent='RECONSTRUCTED · SUCCESS';};
const recordedRenderHeader=renderHeader;
renderHeader=function(){recordedRenderHeader();$('#sceneName').textContent=`重构 Rollout / ${state.dataset.suite} / TASK ${state.dataset.task_id} / ${state.episode.path}`;};
renderSharedPrediction=function(){
  const p=state.payload,r=p.episode.result;$('#sharedScope').textContent='共享关节伺服 · 容差 OFF';
  $('#sharedMaskSummary').textContent=`所选 ${state.episode.selected_trial} · 对原数据末端：${format(r.tracking.source_position_mm.mean,3)} mm / ${format(r.tracking.source_rotation_deg.mean,3)}° · 对补偿参考：${format(r.tracking.nominal_position_mm.mean,3)} mm`;
  renderReasoning();
};
renderReasoning=function(){
  const p=state.payload,c=$('#reasoningContent');c.innerHTML='';
  const note=document.createElement('p');note.textContent='离线测量实际末端相对原始下一状态的误差，平滑并提前一帧修正参考，再做无关节姿态惩罚的 FK 投影和动作差分。在线仅执行冻结动作；默认 IPOPT 不变。';c.appendChild(note);
  const table=document.createElement('table');table.innerHTML='<thead><tr><th>轮次</th><th>增益</th><th>任务</th><th>接受</th><th>原state位置/mm</th></tr></thead><tbody></tbody>';
  for(const t of p.reconstruction_trials){const row=document.createElement('tr');for(const value of[t.iteration,t.gain??'初始',t.success?'成功':'失败',t.accepted?'是':'否',format(t.tracking?.source_position_mm?.mean,3)]){const cell=document.createElement('td');cell.textContent=value;row.appendChild(cell);}table.tBodies[0].appendChild(row);}c.appendChild(table);
};
renderMetadata=function(){
  const p=state.payload,e=state.episode,r=p.episode.result,c=$('#episodeMetadata');c.innerHTML='';
  const values=[['数据批次',window.LIBERO_CATALOG.run],['结果',`完整成功 · ${e.selected_trial}`],['执行/计划帧',`${r.executed_frames}/${r.planned_frames}`],
    ['执行链','默认 IPOPT → 共享关节插值 / JointReferenceTracker → Panda位置伺服 + 偏置补偿'],['求解设置','默认设置和目标函数；20迭代 / 95ms CPU预算'],
    ['时序','20Hz动作 / 500Hz物理；原速；原夹爪；无额外等待'],['容差','关闭'],['零空间控制','无'],
    ['阶段','录制时在线共享分类器结果；20Hz逐帧，连续3帧稳定判定'],['轨迹误差','原state误差与补偿后参考误差分别显示'],
    ['关节惩罚','离线候选评分不惩罚原始关节轨迹'],['观测对齐','视频和state为pre-action；跟踪误差为post-action'],['保存路径',e.selected_root]];
  values.forEach(([k,v])=>c.appendChild(metadataItem(k,v)));
  const links=document.createElement('div');links.className='video-links';
  for(const[label,path]of[['初始截图',e.poster_path],['终态截图',e.final_path],['执行轨迹',e.raw_trace_path],['报告',e.report_path]]){const a=document.createElement('a');a.textContent=label;a.href=path;a.target='_blank';links.appendChild(a);}c.appendChild(links);
};
const recordedInspector=renderInspector;
renderInspector=function(){recordedInspector();const f=state.frames[state.frameIndex];if(f)$('#tolerancePose').textContent=`相对同期原始下一状态：\n位置 ${format(f.source_next_position_error_mm,3)} mm\n旋转 ${format(f.source_next_rotation_error_deg,3)}°\n默认 IPOPT · 共享关节伺服 · 容差 OFF`;};
// Rebind the original listener, which captured the legacy list renderer.
const oldSearch=$('#taskSearch'),newSearch=oldSearch.cloneNode(true);oldSearch.replaceWith(newSearch);newSearch.addEventListener('input',renderDatasetList);
// Legacy source comparison is only initialized when its optional DOM exists.
if(document.getElementById("sourceVideo")){
// Optional side-by-side inspection of the original stored observations.
const sourceVideo=$('#sourceVideo');
function sourceEnabled(){return !!state.episode?.source_video_path&&$('#compareSource').checked;}
function syncSourceVideo(force=false){
  if(!sourceVideo||!sourceEnabled())return;
  sourceVideo.playbackRate=state.speed;
  if(sourceVideo.readyState>=1&&!sourceVideo.seeking&&(force||Math.abs(sourceVideo.currentTime-mainVideo.currentTime)>.025)){
    sourceVideo.currentTime=Math.max(0,Math.min(mainVideo.currentTime,sourceVideo.duration-.001));
  }
}
const singleConfigureVideos=configureVideos;
configureVideos=function(){
  singleConfigureVideos();sourceVideo.pause();
  sourceVideo.src=state.episode.source_video_path;sourceVideo.poster=state.episode.source_poster_path;sourceVideo.playbackRate=state.speed;sourceVideo.load();
  $('#openSourceVideo').href=state.episode.source_video_path;
  $('#comparisonStatus').textContent=`${state.episode.path} · 原始帧 i ↔ 重构帧 i · 20Hz`;
};
const singlePauseVideos=pauseVideos;
pauseVideos=function(){singlePauseVideos();sourceVideo.pause();};
const singleSeekFrame=seekFrame;
seekFrame=function(index){singleSeekFrame(index);syncSourceVideo(true);};
togglePlayback=async function(){
  if(!state.frames.length)return;
  if(!mainVideo.paused){pauseVideos();return;}
  if(mainVideo.ended)seekFrame(0);
  mainVideo.playbackRate=sourceVideo.playbackRate=state.speed;syncSourceVideo(true);
  try{await Promise.all([mainVideo.play(),...(sourceEnabled()?[sourceVideo.play()]:[])]);$('#playButton').textContent='Ⅱ';}
  catch(e){pauseVideos();showError(`视频无法同步播放：${e.message}`);}
};
animationLoop=function(){
  if(state.frames.length&&!mainVideo.paused){syncSourceVideo();updateFrame(videoFrameIndex(mainVideo.currentTime));$('#timeline').value=mainVideo.currentTime;$('#currentTime').textContent=formatTime(mainVideo.currentTime);}
  requestAnimationFrame(animationLoop);
};
mainVideo.addEventListener('seeking',()=>syncSourceVideo(true));
mainVideo.addEventListener('seeked',()=>syncSourceVideo(true));
mainVideo.addEventListener('pause',()=>sourceVideo.pause());
mainVideo.addEventListener('ratechange',()=>{sourceVideo.playbackRate=mainVideo.playbackRate;});
sourceVideo.addEventListener('loadedmetadata',()=>syncSourceVideo(true));
sourceVideo.addEventListener('seeked',()=>{if(mainVideo.paused)syncSourceVideo();});
sourceVideo.addEventListener('error',()=>{if(sourceVideo.getAttribute('src'))showError('原始数据视频加载失败；请检查 source_rgb.mp4。');});
$('#compareSource').addEventListener('change',async()=>{
  $('.rollout-comparison').classList.toggle('source-hidden',!$('#compareSource').checked);
  if(sourceEnabled()){syncSourceVideo(true);if(!mainVideo.paused)try{await sourceVideo.play();}catch(e){showError(e.message);}}
  else sourceVideo.pause();
});
const singleMetadata=renderMetadata;
renderMetadata=function(){singleMetadata();const s=state.episode.source_comparison;if(s){
  $('#episodeMetadata').appendChild(metadataItem('右侧原始视频',`${s.source_hdf5} / ${s.image_dataset}；原始${s.resolution_hw.join('×')}保存图像直接导出，未重录`));
  $('#episodeMetadata').appendChild(metadataItem('双视频对齐',`同 demo、同帧号、20Hz。原始${s.frame_count}帧；rollout${state.frames.length}帧。原始额外末帧可在独立视频查看。`));
}};
}
requestAnimationFrame(animationLoop);
loadDatasets();
