"use strict";

const suiteNames = {libero_spatial:"Spatial",libero_object:"Object",libero_goal:"Goal",libero_10:"Long"};
const terminationNames = {task_success:"成功",ik_infeasible:"求解失败",ik_failure_streak:"连续20帧求解失败",step_limit:"达到步数上限"};
let liveBusy = false;
const payloadCache = new Map();
const originalFetchJSON = fetchJSON;
fetchJSON = function (url) {
  if(!url.startsWith('/api/episode?'))return originalFetchJSON(url);
  if(payloadCache.has(url))return payloadCache.get(url);
  const pending=originalFetchJSON(url).catch(error=>{payloadCache.delete(url);throw error;});
  payloadCache.set(url,pending);
  while(payloadCache.size>24)payloadCache.delete(payloadCache.keys().next().value);
  return pending;
};
function episodeURL(dataset,path){return `/api/episode?${encodeQuery({source:'benchmark',dataset:dataset.relative_path,episode:path})}`;}
function videoURL(dataset,path){return `/media?${encodeQuery({dataset:dataset.relative_path,episode:path,stream:'main_rgb.mp4'})}`;}
function warmEpisode(dataset,path,video=false){
  fetchJSON(episodeURL(dataset,path)).catch(()=>{});
  if(video)fetch(videoURL(dataset,path),{cache:'force-cache',priority:'low'}).then(r=>{if(r.ok)return r.arrayBuffer();}).catch(()=>{});
}

// Rendering charts below the viewport must not compete with the player or clicks.
const visibleCanvases = new Set();
const studioDrawLineChart = drawLineChart, studioDrawSpatial = drawSpatial;
drawLineChart = function (canvas,...args) {
  if(visibleCanvases.has(canvas))studioDrawLineChart(canvas,...args);
};
drawSpatial = function () {
  if(visibleCanvases.has(spatialCanvas))studioDrawSpatial();
};
const chartVisibility = new IntersectionObserver(entries=>{
  entries.forEach(e=>{if(e.isIntersecting)visibleCanvases.add(e.target);else visibleCanvases.delete(e.target);});
  if(state.frames.length){drawSpatial();drawCharts();}
});
$$('canvas').forEach(c=>chartVisibility.observe(c));

// Keep the original Studio's player, canvases, seeking and selection functions.
// The navigation shows tasks and saved seeds; recording refreshes preserve playback.
renderFolderList = function () {
  const select=$('#folderSelect');
  const folders=datasetFolders().map(f=>{
    const d=state.datasets.find(d=>folderForDataset(d)===f.name);
    return {...f,label:d?.eval_label||f.label,priority:d?.view_priority||0};
  }).sort((a,b)=>b.priority-a.priority);
  const signature=folders.map(f=>`${f.name}:${f.label}`).join('|');
  if(select.dataset.signature!==signature){
    select.replaceChildren();select.dataset.signature=signature;
    folders.forEach(f=>{const o=document.createElement('option');o.value=f.name;o.textContent=f.label;o.title=f.title;select.append(o);});
  }
  select.value=state.folder||'';
  $('#folderSummary').textContent='';
};
renderDatasetList = function () {
  const container = $("#datasetList");
  const query = $("#benchmarkSearch").value.trim().toLowerCase();
  let visible = 0;
  const retained = new Set();
  state.datasets.forEach((dataset, index) => {
    if (folderForDataset(dataset) !== state.folder || !`${dataset.task_name} ${dataset.suite}`.toLowerCase().includes(query)) return;
    visible++;
    retained.add(dataset.relative_path);
    let button=[...container.children].find(b=>b.dataset.path===dataset.relative_path);
    if(!button){
      button=document.createElement('button');button.dataset.path=dataset.relative_path;
      button.append(document.createElement('strong'),document.createElement('span'));container.append(button);
      button.onclick=()=>selectDataset(state.datasets.findIndex(d=>d.relative_path===button.dataset.path));
      button.onpointerenter=()=>{
        const d=state.datasets.find(d=>d.relative_path===button.dataset.path);
        const e=d?.episodes.find(e=>e.path===state.episode?.path)||d?.episodes[0];if(e)warmEpisode(d,e.path);
      };
    }
    button.className = `dataset-item${state.dataset === dataset ? " active" : ""}`;
    const title = button.children[0], detail = button.children[1];
    const titleText=dataset.task_name;
    if(title.textContent!==titleText)title.textContent=titleText;
    title.title = dataset.task_name;
    const detailText=`${suiteNames[dataset.suite]} · task ${String(dataset.task_id).padStart(2,"0")}`;
    if(detail.textContent!==detailText)detail.textContent=detailText;
  });
  [...container.children].forEach(b=>{if(!retained.has(b.dataset.path))b.remove();});
  $(".dataset-heading > span").textContent = `LIBERO任务 · ${visible}`;
};

renderEpisodeList = function () {
  const container=$('#episodeList'),episodes=filteredEpisodes(state.dataset);
  const signature=episodes.map(e=>`${e.path}:${e.frames}:${e.success}`).join('|');
  if(container.dataset.signature!==signature){
    container.replaceChildren();container.dataset.signature=signature;
    episodes.forEach(e=>{
      const b=document.createElement('button');b.dataset.path=e.path;b.className='episode-item';
      b.innerHTML=`<span class="episode-copy"><strong>seed ${e.seed}</strong><span>${e.frames} frames</span></span><i class="episode-result ${e.success?'success':''}"></i>`;
      b.title=terminationNames[e.termination]||e.termination||'';
      b.onclick=()=>selectEpisode(state.dataset.episodes.findIndex(x=>x.path===b.dataset.path));
      b.onpointerenter=()=>warmEpisode(state.dataset,b.dataset.path,true);container.append(b);
    });
  }
  [...container.children].forEach(b=>b.classList.toggle('active',b.dataset.path===state.episode?.path));
  $('#episodeCount').textContent=episodes.length;
};

renderMetadata = function () {
  const container = $("#episodeMetadata");container.replaceChildren();
  const r = state.payload?.episode?.benchmark_result || {};
  const items = [
    ["任务指令",r.language], ["任务集 / seed",`${suiteNames[r.suite]} / ${r.seed}`],
    ["结果",terminationNames[r.termination] || r.termination], [r.post_success_seconds === undefined ? "步数 / 上限" : "步数 / 成功前上限",`${r.steps} / ${r.max_steps}`],
    ["仿真频率",`${r.control_hz} Hz / ${r.physics_hz} Hz physics`], ["重规划",`${r.replan_steps}步 · 预测10步`],
    ["推理调用",r.inference_calls], ["回合耗时",`${format(r.elapsed_seconds,1)} s`],
    ["控制器",r.controller], ["旋转容差",r.tolerance_enabled ? (r.tolerance_bounds_policy === 'per_task_subtask_phase' ? '按任务/子任务/阶段分轴角度 · xF/yF/zF' : `±${r.tolerance_bounds_deg?.[0] ?? 30}° · ${r.semantic_mask_policy === 'user_confirmed_task_phase_masks' ? '对应任务/阶段mask' : `mask ${r.semantic_mask}`}`) : "关闭"], ["输入state",r.state_input_enabled ? "开启" : "关闭"],
    ["初始状态",`官方索引 ${r.initial_state_index}`], ["模型checkpoint",r.server_metadata?.checkpoint],
    ["完成时间",r.finished_at], ["记录来源",state.payload?.episode?.benchmark_source?.run],
  ];
  if (r.post_success_seconds !== undefined) items.push(
    ["成功判据", "首次原生成功命中；末态单独记录"],
    ["首次成功", r.first_success_frame === null ? "未命中" : `帧 ${r.first_success_frame} · ${format(r.first_success_time_s,2)} s`],
    ["成功后继续", `${format(r.post_success_elapsed_seconds,2)} / ${format(r.post_success_seconds,2)} s${r.post_success_interrupted_reason ? ` · 提前结束：${r.post_success_interrupted_reason}` : ''}`],
    ["结束时原生成功", r.final_native_success ? "满足" : "不满足"]
  );
  items.forEach(([label,value]) => container.append(metadataItem(label,value ?? "—")));
  const query = encodeQuery({dataset:state.dataset.relative_path,episode:state.episode.path});
  const links = document.createElement("div");links.className="benchmark-downloads";
  [["原始双相机MP4","main_rgb.mp4"],["逐步轨迹JSON","trajectory.json"],["回合结果JSON","result.json"]].forEach(([label,stream])=>{
    const a=document.createElement("a");a.textContent=label;a.href=`/media?${query}&stream=${stream}`;a.target="_blank";a.rel="noopener";links.append(a);
  });container.append(links);
};

const studioInspector = renderInspector;
renderInspector = function () {
  studioInspector();
  const f=state.frames[state.frameIndex];if(!f)return;
  const r=state.payload?.episode?.benchmark_result;
  const angles=f.semantic_tolerance_deg || (Array.isArray(f.bounds_rad) ? [0,1,2].map(i=>f.semantic_mask?.[i]==='1' ? f.bounds_rad[2*i]*180/Math.PI : 0) : null);
  const limits=r?.tolerance_bounds_policy==='per_task_subtask_phase' ? (angles ? `±[${angles.map(v=>format(v,0)).join(', ')}]°` : '分轴角度缺失') : `±${r?.tolerance_bounds_deg?.[0] ?? 30}°`;
  $("#phaseBadge").textContent = f.aborted ? "求解失败 · 终止帧" : f.reference_held ? `保持上一参考 · 连续失败 ${f.consecutive_solver_failures}/${f.max_consecutive_solver_failures}` : r?.tolerance_enabled ? `${f.phase} · ${limits} · mask ${f.semantic_mask ?? r.semantic_mask} · 有效 ${f.effective_mask ?? '—'}` : "无容差执行";
  $("#optimizerValue").textContent = `${f.optimizer?.iterations ?? "—"} iter · ${f.optimizer?.feasible ? "可行" : "不可行"}`;
};

const studioClearEpisode = clearEpisode;
clearEpisode = function () {
  studioClearEpisode();
  delete $('.video-panel').dataset.record;
  $('.video-panel').setAttribute('aria-busy','false');
  $('#playButton').disabled=false;
  $('#emptyState h2').textContent='该任务暂无回放';
  $('#emptyState p').textContent='已保存的 seed 会自动出现在右侧列表，也可选择其他任务。';
};

selectDataset = async function (index,requestedEpisode=null) {
  if(index<0||!state.datasets[index])return;
  const previousSeed=requestedEpisode||state.episode?.path;
  state.dataset=state.datasets[index];state.folder=folderForDataset(state.dataset);
  renderFolderList();renderDatasetList();renderEpisodeList();
  const episodes=filteredEpisodes(state.dataset);
  if(!episodes.length){clearEpisode();return;}
  const chosen=episodes.find(e=>e.path===previousSeed)||episodes[0];
  await selectEpisode(state.dataset.episodes.indexOf(chosen));
};

configureVideos = function () {
  const url=videoURL(state.dataset,state.episode.path);
  [mainVideo,wristVideo].forEach(v=>{
    v.preload='auto';v.playbackRate=state.speed;
    if(v.getAttribute('src')!==url){v.src=url;v.load();}
    else v.currentTime=0;
  });
  const duration=state.frames.length/20;
  $('#timeline').max=duration;$('#timeline').value=0;$('#durationTime').textContent=formatTime(duration);
  $('#playButton').textContent='▶';
};

selectEpisode = async function (index) {
  const dataset=state.dataset,episode=dataset?.episodes[index];if(!episode)return;
  const key=`${dataset.relative_path}/${episode.path}`;
  if($('.video-panel').dataset.record===key){
    state.episodeRequest++;state.episode=episode;renderEpisodeList();
    $('.video-panel').setAttribute('aria-busy','false');$('#playButton').disabled=false;return;
  }
  const started=performance.now(),request=++state.episodeRequest;
  pauseVideos();state.episode=episode;renderEpisodeList();
  $('#playButton').disabled=true;
  $('.video-panel').setAttribute('aria-busy','true');
  try{
    const payload=await fetchJSON(episodeURL(dataset,episode.path));
    if(request!==state.episodeRequest)return;
    state.dataset=dataset;state.episode=episode;state.payload=payload;state.frames=payload.trajectory||[];
    state.annotations=payload.stage_annotations||[];state.frameIndex=0;state.renderedFrame=-1;resetSpatialView();
    if(payload._chartCache)state.chartCache=payload._chartCache;
    else{buildDerivedFrames();payload._chartCache=state.chartCache;}
    configureVideos();renderHeader();$('#taskName').title=dataset.task_name;
    $('#emptyState').classList.add('hidden');$('#content').classList.remove('hidden');
    const panel=$('.video-panel');panel.dataset.record=key;panel.dataset.selectionMs=(performance.now()-started).toFixed(1);
    const url=new URL(location.href);url.searchParams.set('dataset',dataset.relative_path);url.searchParams.set('episode',episode.path);url.searchParams.set('source','benchmark');history.replaceState(null,'',url);
    requestAnimationFrame(()=>{if(request===state.episodeRequest){updateFrame(0,true);renderMetadata();}});
    setTimeout(()=>{
      if(request!==state.episodeRequest)return;
      const nextSeed=dataset.episodes[index+1];if(nextSeed)warmEpisode(dataset,nextSeed.path,true);
      const nextTask=state.datasets[state.datasets.findIndex(d=>d.relative_path===dataset.relative_path)+1];
      const nextEpisode=nextTask?.episodes.find(e=>e.path===episode.path)||nextTask?.episodes[0];
      if(nextEpisode)warmEpisode(nextTask,nextEpisode.path);
    },200);
  }catch(error){if(request===state.episodeRequest)showError(error.message);}
  finally{if(request===state.episodeRequest){$('.video-panel').setAttribute('aria-busy','false');$('#playButton').disabled=false;}}
};

async function refreshLive() {
  if(liveBusy)return;liveBusy=true;
  try {
    const snapshot=await fetchJSON("/api/datasets");
    const oldPath=state.dataset?.relative_path,oldEpisode=state.episode?.path;
    state.datasets=snapshot.datasets;
    const selected=state.datasets.find(d=>d.relative_path===oldPath);
    if(selected){
      state.dataset=selected;state.episode=selected.episodes.find(e=>e.path===oldEpisode)||state.episode;
      renderDatasetList();renderEpisodeList();
      if(!state.episode&&selected.episodes.length)await selectEpisode(0);
    }else if(state.datasets.length){
      const first=state.datasets.findIndex(d=>d.episodes.length);
      await selectDataset(Math.max(0,first));
    }
    if(state.dataset)$("#taskName").title=state.dataset.task_name;
  }catch(error){showError(`记录刷新失败：${error.message}`);}
  finally{liveBusy=false;}
}

$("#benchmarkSearch").addEventListener("input",renderDatasetList);
setInterval(refreshLive,3000);
loadDatasets();
