"use strict";
// Data/semantic adapter for the copied MuJoCo Dataset Studio rendering engine.
// No fetch API is needed: local classic-script payloads also work from file://.
window.LIBERO_EPISODES = window.LIBERO_EPISODES || {};
state.reasonIndex = 0;
state.followRepresentative = true;
state.loadGeneration = 0;
const LIBERO_PHASE_LABEL = {
  pregrasp:"APP / PRE", grasp:"ENG / 严格过渡", postgrasp:"EXEC / POST", release:"DIS / 严格过渡",
  outside_subtask:"子任务外 / 撤离", gripper_open_stable:"开口稳定 · 事件候选",
  gripper_closing:"闭合过渡 · 事件候选", gripper_closed_stable:"闭口稳定 · 事件候选",
  gripper_opening:"张开过渡 · 事件候选",
};
const LIBERO_PHASE_COLORS = {
  pregrasp:"#48d9dc",grasp:"#ffb15a",postgrasp:"#c397ff",release:"#ff6b72",outside_subtask:"#65717c",
  gripper_open_stable:"#55747c",gripper_closing:"#b88b4e",gripper_closed_stable:"#8a72ae",gripper_opening:"#b2757a",
};
function escapeHtml(value){return String(value??"").replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
function format(value,digits=3){return value!==null&&value!==undefined&&Number.isFinite(Number(value))?Number(value).toFixed(digits):"—";}
function maskText(value){return Array.isArray(value)?value.map(v=>v===null?'?':v).join(''):value??'未知';}
function activeTask(){return state.dataset?.shared_annotation;}
function activeSubtask(){return activeTask()?.subtasks?.[state.reasonIndex]||null;}
function folderForDataset(dataset){return dataset.suite;}
function renderFolderList(){
  const suites=[...new Set(state.datasets.map(d=>d.suite))];
  const select=$("#folderSelect");select.innerHTML="";
  for(const suite of suites){const o=document.createElement('option');o.value=suite;o.textContent=`${suite} · 10任务 / 100 MP4`;select.appendChild(o);}
  select.value=state.folder||suites[0]||'';
  $("#folderSummary").textContent=`4任务集 · 40任务 · 400视频 · 当前 ${state.folder||'—'}`;
}
function renderDatasetList(){
  const container=$("#datasetList");container.innerHTML='';
  const query=($("#taskSearch").value||'').trim().toLowerCase();
  for(const suite of [...new Set(state.datasets.map(d=>d.suite))]){
    const items=state.datasets.map((d,i)=>({d,i})).filter(({d})=>d.suite===suite&&(!query||`${d.task_name} ${d.display_name} ${d.suite}`.toLowerCase().includes(query)));
    if(!items.length)continue;
    const group=document.createElement('details');group.className='suite-tree';group.open=!!query||suite===state.folder;
    const heading=document.createElement('summary');heading.textContent=suite;
    const detail=document.createElement('span');detail.textContent=`${items.length}任务 · 各10个MP4`;heading.appendChild(detail);group.appendChild(heading);
    for(const {d,i} of items){
      const b=document.createElement('button');b.className=`dataset-item${state.dataset===d?' active':''}`;b.dataset.taskIndex=i;
      const title=document.createElement('strong');title.textContent=`${String(d.task_id).padStart(2,'0')} · ${d.display_name}`;title.title=d.task_name;
      const small=document.createElement('span');small.textContent=`10 MP4 · ${d.shared_annotation.subtasks.length}子任务 · 共享预测`;
      b.append(title,small);b.addEventListener('click',()=>selectDataset(i));group.appendChild(b);
    }
    container.appendChild(group);
  }
  const heading=document.querySelector('.dataset-heading > span');if(heading)heading.textContent=query?'任务搜索结果':'任务集 → 各10任务';
}
async function selectFolder(folder){
  state.folder=folder;renderFolderList();renderDatasetList();
  const index=state.datasets.findIndex(d=>d.suite===folder);if(index>=0)await selectDataset(index);
}
async function loadDatasets(){
  try{
    state.datasets=window.LIBERO_CATALOG.datasets;
    const q=new URLSearchParams(location.search);
    const selected=Math.max(0,state.datasets.findIndex(d=>d.relative_path===q.get('dataset')));
    state.folder=state.datasets[selected].suite;renderFolderList();
    await selectDataset(selected,q.get('episode'));
  }catch(e){showError(e.message);$("#emptyState h2").textContent='未能读取本地目录';$("#emptyState p").textContent=e.message;}
}
async function selectDataset(index,requestedEpisode=null){
  const selected=state.datasets[index];if(!selected)return;
  state.dataset=selected;state.folder=selected.suite;state.reasonIndex=0;state.followRepresentative=true;
  renderFolderList();renderDatasetList();
  const i=Math.max(0,selected.episodes.findIndex(e=>e.path===requestedEpisode));
  await selectEpisode(i);
}
function renderEpisodeList(){
  const c=$("#episodeList");c.innerHTML='';const episodes=state.dataset?.episodes||[];$("#episodeCount").textContent=episodes.length;
  episodes.forEach((episode,i)=>{
    const b=document.createElement('button');b.className=`episode-item${state.episode===episode?' active':''}`;b.dataset.demo=episode.path;
    b.innerHTML=`<span class="episode-number">${String(i+1).padStart(2,'0')}</span><span class="episode-copy"><strong>${escapeHtml(episode.path)}</strong><span>${episode.representative?'原标注代表':'补充成功示范'}<br>${episode.frames}帧 · ${format(episode.frames/20,1)}s</span></span><i class="episode-result success"></i>`;
    b.addEventListener('click',()=>selectEpisode(i));c.appendChild(b);
  });
}
function readEpisodeScript(episode){
  if(window.LIBERO_EPISODES[episode.key])return Promise.resolve(window.LIBERO_EPISODES[episode.key]);
  return new Promise((resolve,reject)=>{
    const script=document.createElement('script');script.src=episode.trace_script_path;
    script.onload=()=>{const data=window.LIBERO_EPISODES[episode.key];script.remove();data?resolve(data):reject(new Error('轨迹文件缺少对应demo数据'));};
    script.onerror=()=>{script.remove();reject(new Error(`无法加载 ${episode.trace_script_path}；请保留HTML旁的assets/trajectories目录。`));};
    document.body.appendChild(script);
  });
}
async function selectEpisode(index){
  const dataset=state.dataset,episode=dataset?.episodes[index];if(!episode)return;
  const token=++state.loadGeneration;pauseVideos();state.episode=episode;renderEpisodeList();
  $("#syncStatus").textContent='LOADING';
  try{
    const payload=await readEpisodeScript(episode);if(token!==state.loadGeneration)return;
    state.payload=payload;state.frames=payload.trajectory||[];state.annotations=[];state.frameIndex=0;state.renderedFrame=-1;
    resetSpatialView();state.view.mode='FREE 3D · not camera calibration';
    buildDerivedFrames();configureVideos();renderHeader();renderMetadata();renderSharedPrediction();renderPhaseTrack();
    $("#emptyState").classList.add('hidden');$("#content").classList.remove('hidden');
    updateFrame(0,true);$("#syncStatus").textContent=episode.representative?'REVIEWED DEMO':'EXTRA DEMO';
    try{const u=new URL(location.href);u.searchParams.set('dataset',dataset.relative_path);u.searchParams.set('episode',episode.path);u.searchParams.delete('source');history.replaceState(null,'',u);}catch(_e){}
  }catch(e){if(token===state.loadGeneration){showError(e.message);$("#syncStatus").textContent='LOAD ERROR';}}
}
function configureVideos(){
  mainVideo.src=state.episode.video_path;mainVideo.poster=state.episode.poster_path;mainVideo.playbackRate=state.speed;mainVideo.load();
  const duration=state.frames.length/20;$("#timeline").max=duration;$("#timeline").step=1/20;$("#timeline").value=0;
  $("#durationTime").textContent=formatTime(duration);$("#playButton").textContent='▶';
  $("#openVideo").href=state.episode.video_path;$("#traceDownload").href=state.episode.trace_path;
  const rep=state.episode.representative,scope=$("#episodeScope");scope.classList.toggle('unreviewed',!rep);
  scope.textContent=rep?`原代表示范 ${state.episode.path}：图像核对过的阶段边界与有效mask。当前显示观测先于动作；完整post-action终态未记录。`:`补充示范 ${state.episode.path}：夹爪颜色条仅为事件候选，未分配视觉子任务或逐帧mask。任务共享预测的证据来自 ${state.dataset.representative_demo_id}。`;
}
function renderHeader(){
  $("#sceneName").textContent=`LIBERO / ${state.dataset.suite} / TASK ${state.dataset.task_id} / ${state.episode.path}`;
  $("#taskName").textContent=state.dataset.task_name;$("#instructionText").textContent=state.dataset.display_name;
  $("#seedValue").textContent=state.episode.path;$("#frameCountValue").textContent=state.frames.length;$("#resultValue").textContent='SUCCESS';$("#resultValue").className='success';
}
function renderMetadata(){
  const c=$("#episodeMetadata");c.innerHTML='';const p=state.payload,m=p.metrics;
  const values=[['任务集 / task id',`${state.dataset.suite} / ${state.dataset.task_id}`],['示范 / 保存结果',`${state.episode.path} · 成功记录`],
    ['视频与观测',`${state.frames.length}帧 · 20Hz · ${(state.frames.length/20).toFixed(2)}s`],['预测来源',`${state.dataset.representative_demo_id} · 所选模式`],
    ['观测时序','pre-action；10次settling后时间归零'],['保留帧之间的源帧空洞',`${m.retained_source_gaps}；不在重放时间内插入停顿`],
    ['末端观测路径长度',`${format(m.observed_ee_path_length_m,3)} m`],['平均观测关节速度范数',`${format(m.mean_observed_joint_speed_rad_s,3)} rad/s · 20Hz差分`],
    ['轨迹来源',p.source.processed_hdf5],['状态对齐字段','tasktol/pre_action_robot_states；不使用legacy states[0]'],
    ['控制动作单位','归一化OSC_POSE输入；非测得位移/角度'],['六方向容差 / 标称参考','未生成数值边界；未重建独立传播标称参考'],
    ['重力参考诊断','F_obs使用标定后的物理闭合线；观测代理不改写原预测'],['测度范围','速度由观测关节差分；不是优化命令轨迹的benchmark指标'],
    ['原标注冻结SHA256',window.LIBERO_CATALOG.annotation_sha256],['证据范围','所有判定图像只来自原代表示范主视角']];
  values.forEach(([k,v])=>c.appendChild(metadataItem(k,v)));
}
function renderSharedPrediction(){
  const task=activeTask(),sub=activeSubtask();if(!sub)return;
  $("#sharedScope").textContent=`共享给10个demo · 依据 ${task.demo_id}`;
  $("#sharedMaskSummary").innerHTML=`<div class="mask-pair"><span>APP / PRE<b>${sub.APP.effective_mask}</b></span><span>EXEC / POST<b>${sub.EXEC.effective_mask}</b></span></div><p><strong>s${sub.index} · ${escapeHtml(sub.object_or_mechanism)} → ${escapeHtml(sub.target)}</strong><br>接触部位：${escapeHtml(sub.contact_part)}<br>ENG / DIS 执行门控 000 · 当前子任务${sub.uncertainties.length?'含保守置零待确认轴':'有局部视觉依据'}</p>`;
  const tabs=$("#subtaskTabs");tabs.innerHTML='';
  task.subtasks.forEach((s,i)=>{const b=document.createElement('button');b.classList.toggle('active',i===state.reasonIndex);b.textContent=`s${i} ${s.object_or_mechanism} · ${s.APP.effective_mask}/${s.EXEC.effective_mask}`;b.onclick=()=>{state.reasonIndex=i;state.followRepresentative=false;renderSharedPrediction();};tabs.appendChild(b);});
  const label=document.createElement('label');label.className='follow-control';const checkbox=document.createElement('input');checkbox.type='checkbox';checkbox.checked=state.followRepresentative;checkbox.onchange=()=>{state.followRepresentative=checkbox.checked;renderInspector();};label.append(checkbox,document.createTextNode('随代表视频切换子任务'));tabs.appendChild(label);
  renderReasoning();
}
function evidenceLink(id,sub){
  const f=sub.keyframes.find(f=>f.id===id);if(f)return `<a href="${escapeHtml(f.path)}" target="_blank" rel="noopener">${escapeHtml(id)}</a>`;
  const m=id.match(/:p(\d+)$/);return m?`<a href="keyframes/${escapeHtml(state.dataset.suite)}/${escapeHtml(state.dataset.task_name)}/frame_${m[1].padStart(5,'0')}.png" target="_blank" rel="noopener">${escapeHtml(id)}</a>`:escapeHtml(id);
}
function renderReasoning(){
  const sub=activeSubtask(),task=activeTask();if(!sub)return;
  let h=`<div class="reason-section"><h3>图像观察与分段</h3><p>${escapeHtml(sub.visual_observations)}</p><p>实际接触：${escapeHtml(sub.contact_part)}；操作：${escapeHtml(sub.operation)}；目标：${escapeHtml(sub.target)}。</p><p>${escapeHtml(task.boundary_notes)}</p><table><tr><th>阶段</th><th>处理帧半开区间</th><th>重放秒</th></tr>`;
  for(const span of sub.phase_spans)h+=`<tr><td>${span.phase}</td><td>[${span.processed_interval.join(', ')})</td><td>${span.replay_time_interval_s.map(v=>format(v,2)).join(' – ')}</td></tr>`;
  h+='</table></div>';
  for(const phase of ['APP','EXEC']){
    const d=sub[phase];h+=`<section class="reason-section"><h3>${phase} · 判定摘要</h3>`;
    if(phase==='EXEC')h+=`<div class="formula">物体 ${maskText(d.object_mask)} ∧ 操作 ${maskText(d.operation_mask)} ∧ 目标 ${maskText(d.goal_mask)} = 候选 ${maskText(d.semantic_candidate_mask)}</div>`;
    h+=`<div class="formula">候选 ${maskText(d.semantic_candidate_mask)} ∧ 框架 ${d.frame_support_mask} ∧ 任务点 ${d.task_point_support_mask} = 有效 ${d.effective_mask}</div><p>${escapeHtml(d.semantic_scope)} · ${escapeHtml(d.semantic_scope_basis)}</p><p class="reason-links">参考：${d.reference_task_ids.map(r=>`<a href="reference_25.md" target="_blank" rel="noopener">${escapeHtml(r)}</a>`).join(' · ')}</p>`;
    for(const [axis,v]of Object.entries(d.axes))h+=`<div class="reason-axis"><strong>${axis} = ${v.value}</strong> · ${escapeHtml(v.status)} · ${escapeHtml(v.confidence)}<p>${escapeHtml(v.reason)}</p><div class="reason-links">证据：${v.evidence_ids.map(id=>evidenceLink(id,sub)).join(' · ')}<br>原因：${escapeHtml(v.reason_codes.join(', ')||'未增加框架/任务点否决')}</div></div>`;
    h+='</section>';
  }
  h+='<div class="reason-section"><h3>实际查看过的主视角证据</h3><p>点击截图会切回原代表示范并定位该帧。</p><div class="evidence-grid">';
  for(const f of sub.keyframes)h+=`<figure><button data-evidence-frame="${f.processed_frame}"><img loading="lazy" src="${escapeHtml(f.path)}" alt="${escapeHtml(f.id)}"></button><figcaption>${escapeHtml(f.event_name)}<br>p=${f.processed_frame} / src=${f.source_frame} · ${format(f.replay_time_s,2)}s<br>pre-action · 已实际查看</figcaption></figure>`;
  h+='</div></div>';
  h+=`<p class="warning">${sub.missing_events.map(x=>escapeHtml(x.reason)).join('；')||'数值方向边界未生成。'} 原标注没有独立真值；000可能表示证据不足或框架不支持。</p>`;
  h+=`<p class="reason-links"><a href="review/${escapeHtml(task.suite)}_${String(task.task_id).padStart(2,'0')}.png" target="_blank">全程关键帧拼图</a> · <a href="review/${escapeHtml(task.suite)}_${String(task.task_id).padStart(2,'0')}_expanded.png" target="_blank">接触±0.6秒补帧</a> · <a href="decision_records/${escapeHtml(task.suite)}/${String(task.task_id).padStart(2,'0')}_${escapeHtml(task.task_name)}.md" target="_blank">完整逐任务记录</a> · <a href="uncertainties.md" target="_blank">所有未知项</a> · <a href="run_manifest.json" target="_blank">原实验访问协议与偏差</a></p>`;
  $("#reasoningContent").innerHTML=h;
  $$('#reasoningContent button[data-evidence-frame]').forEach(b=>b.onclick=async()=>{
    const index=state.dataset.episodes.findIndex(e=>e.representative);if(!state.episode.representative)await selectEpisode(index);seekFrame(Number(b.dataset.evidenceFrame));
  });
}
function renderPhaseTrack(){
  const bar=$("#phaseTrack");bar.innerHTML='';const f=state.frames;if(!f.length)return;
  let start=0;
  for(let i=1;i<=f.length;i++)if(i===f.length||f[i].phase!==f[start].phase||f[i].subtask_index!==f[start].subtask_index){
    const phase=f[start].phase,b=document.createElement('button');b.style.width=`${100*(i-start)/f.length}%`;b.style.background=LIBERO_PHASE_COLORS[phase]||'#65717c';
    b.title=`${LIBERO_PHASE_LABEL[phase]||phase} · [${start},${i}) · ${format(start/20,2)}–${format(i/20,2)}s`;const target=start;b.onclick=()=>seekFrame(target);bar.appendChild(b);start=i;
  }
}
function buildDerivedFrames(){
  const frames=state.frames;
  state.chartCache={
    positionActual:[0,1,2].map(i=>frames.map(f=>f.ee_state.position_m[i])),
    rotationActual:[0,1,2].map(i=>frames.map(f=>f.ee_state.rotation_vector_base_rad[i])),
    masks:[0,1,2].map(i=>frames.map(f=>f.execution_mask===null?null:Number(f.execution_mask[i]))),
    joints:[0,1,2,3,4,5,6].map(i=>frames.map(f=>f.joint_state.position_rad[i])),
    velocities:[0,1,2,3,4,5,6].map(i=>frames.map(f=>f.joint_state.velocity_rad_s[i])),
    actionsT:[0,1,2].map(i=>frames.map(f=>f.action[i])),actionsR:[3,4,5].map(i=>frames.map(f=>f.action[i])),
    actionGripper:[frames.map(f=>f.action[6])],aperture:[frames.map(f=>f.gripper_state.aperture_m*1000)],
    source:[frames.map(f=>f.source_frame-f.index)],
  };
}
function renderPose(container,pose){
  const values=[...(pose?.position_m||[null,null,null]),...(pose?.rotation_vector_base_rad||[null,null,null])];
  vectorCells(container,['X','Y','Z','rvX','rvY','rvZ'],values,'');
  [...container.children].forEach((cell,i)=>cell.querySelector('strong').textContent+=i<3?' m':' rad');
}
function renderInspector(){
  const f=state.frames[state.frameIndex];if(!f)return;
  if(state.episode.representative&&state.followRepresentative&&f.subtask_index!==null&&f.subtask_index!==state.reasonIndex){state.reasonIndex=f.subtask_index;renderSharedPrediction();}
  const badge=$("#phaseBadge");badge.textContent=`${f.subtask_index===null?'':`s${f.subtask_index} · `}${LIBERO_PHASE_LABEL[f.phase]||f.phase}`;badge.className=`phase-badge ${PHASE_CLASS[f.phase]||''}`;
  renderPose($("#actualPose"),f.ee_state);
  $("#nominalPose").textContent='未记录独立传播的标称参考；不把观测姿态冒充标称目标。';
  $("#stagePose").textContent=state.episode.representative?`原代表视觉边界 · ${LIBERO_PHASE_LABEL[f.phase]||f.phase} · 有效mask ${f.execution_mask??'不适用'}`:'该补充示范未作视觉分段；事件候选不作为语义子任务或逐帧mask。';
  const g=f.observed_gravity_frame;
  $("#tolerancePose").className='frame-facts';
  $("#tolerancePose").textContent=g.rotation_matrix?`F_obs · 校准后的物理闭合线\nyF [${g.rotation_matrix.map(r=>format(r[1],3)).join(', ')}]\nzF [0, 0, 1] · h=${format(g.h,6)}\n观测代理，不是动作积分标称F`:'F_obs投影退化：不换轴；诊断保护000。实际标称F退化仍未测量。';
  const bars=$("#toleranceLimits");bars.innerHTML='';
  ['xF','yF','zF'].forEach((axis,i)=>{const value=f.execution_mask===null?null:Number(f.execution_mask[i]);const row=document.createElement('div');row.className=`mask-row${value===null?' unknown':''}`;row.innerHTML=`<span>${axis}</span><div class="mask-track"><i style="width:${value===1?100:0}%"></i></div><span>${value===null?'逐帧mask未知':`有效 ${value}`}</span>`;bars.appendChild(row);});
  vectorCells($("#actionVector"),['ΔX raw','ΔY raw','ΔZ raw','ΔRx raw','ΔRy raw','ΔRz raw','Grip'],f.action,'');
  $("#gripperValue").textContent=`${format(f.gripper_state.aperture_m*1000,2)} mm · q左−q右`;
  $("#optimizerValue").textContent=`src ${f.source_frame} · p ${f.index} · pre-action`;
}
function drawSpatial(){
  if(!state.frames.length)return;
  const {ctx,width,height}=canvasContext(spatialCanvas);ctx.clearRect(0,0,width,height);
  // Reuse the MuJoCo projection/orbit engine with only observed points.
  const points=state.frames.map(f=>f.ee_state.position_m);
  const mins=[0,1,2].map(i=>Math.min(...points.map(p=>p[i]))-.06),maxs=[0,1,2].map(i=>Math.max(...points.map(p=>p[i]))+.06);
  const center=mins.map((v,i)=>(v+maxs[i])/2),viewed=points.map(p=>viewPoint(subtract(p,center)));
  const spanU=Math.max(.12,Math.max(...viewed.map(p=>p[0]))-Math.min(...viewed.map(p=>p[0]))),spanV=Math.max(.12,Math.max(...viewed.map(p=>p[1]))-Math.min(...viewed.map(p=>p[1])));
  const scale=Math.min((width-100)/spanU,(height-85)/spanV)*state.view.zoom;
  const projection={mins,maxs,center,project(p){const v=viewPoint(subtract(p,center));return[width/2+v[0]*scale,height/2-v[1]*scale,v[2]];}};
  const step=.05,x0=Math.floor(mins[0]/step)*step,x1=Math.ceil(maxs[0]/step)*step,y0=Math.floor(mins[1]/step)*step,y1=Math.ceil(maxs[1]/step)*step,floor=mins[2];
  for(let x=x0;x<=x1+.001;x+=step)stroke3D(ctx,projection,[[x,y0,floor],[x,y1,floor]],'rgba(91,116,128,.14)',1);
  for(let y=y0;y<=y1+.001;y+=step)stroke3D(ctx,projection,[[x0,y,floor],[x1,y,floor]],'rgba(91,116,128,.14)',1);
  drawFrame3D(ctx,projection,[x0,y0,floor],[[1,0,0],[0,1,0],[0,0,1]],.05,.65,'WORLD XYZ');
  stroke3D(ctx,projection,points,'rgba(72,217,220,.24)',1.2);stroke3D(ctx,projection,points.slice(0,state.frameIndex+1),'#48d9dc',2.4);
  const f=state.frames[state.frameIndex],position=f.ee_state.position_m;
  drawFrame3D(ctx,projection,position,rotvecMatrix(f.ee_state.rotation_vector_base_rad),.045,.8,'TOOL observed');
  if(f.observed_gravity_frame.rotation_matrix)drawFrame3D(ctx,projection,position,f.observed_gravity_frame.rotation_matrix,.075,.95,'F_obs');
  const [x,y]=projection.project(position);ctx.beginPath();ctx.arc(x,y,5,0,Math.PI*2);ctx.fillStyle='#ffb15a';ctx.fill();
  $("#axisReadout").textContent=`WORLD [m]\nx ${format(position[0],4)}\ny ${format(position[1],4)}\nz ${format(position[2],4)}\n\np ${f.index} / src ${f.source_frame}\nF_obs: ${f.observed_gravity_frame.rotation_matrix?'observed proxy':'DEGENERATE / no axis swap'}\nview ${state.view.mode}`;
}
function drawCharts(){
  const c=state.chartCache;if(!c)return;
  drawLineChart($("#positionChart"),axisSeries(c.positionActual));drawLineChart($("#rotationChart"),axisSeries(c.rotationActual));
  drawLineChart($("#toleranceChart"),axisSeries(c.masks),{min:0,max:1,padding:false});
  if(!state.episode.representative){const {ctx,width,height}=canvasContext($("#toleranceChart"));ctx.fillStyle='#b9a27d';ctx.font='12px sans-serif';ctx.fillText('补充demo未逐帧标注 · 任务共享预测见上方',50,height/2);}
  drawLineChart($("#jointChart"),c.joints.map((values,i)=>({values,color:COLORS[i],width:1.15,alpha:.82})));
  drawLineChart($("#jointVelocityChart"),c.velocities.map((values,i)=>({values,color:COLORS[i],width:1.15,alpha:.82})));
  drawLineChart($("#apertureChart"),[{values:c.aperture[0],color:COLORS[4]}]);drawLineChart($("#sourceChart"),[{values:c.source[0],color:COLORS[3]}]);
  drawLineChart($("#actionTranslationChart"),axisSeries(c.actionsT));drawLineChart($("#actionRotationChart"),axisSeries(c.actionsR));
  drawLineChart($("#actionGripperChart"),[{values:c.actionGripper[0],color:COLORS[4],width:1.8}],{min:-1.05,max:1.05,padding:false});
}
function updateFrame(index,force=false){
  if(!state.frames.length)return;state.frameIndex=clamp(Math.round(index),0,state.frames.length-1);
  if(!force&&state.renderedFrame===state.frameIndex)return;state.renderedFrame=state.frameIndex;
  const t=state.frameIndex/20;$("#timeline").value=t;$("#currentTime").textContent=formatTime(t);$("#currentFrame").textContent=String(state.frameIndex).padStart(3,'0');
  renderInspector();drawSpatial();drawCharts();
}
function seekFrame(index){
  if(!state.frames.length)return;index=clamp(Math.round(index),0,state.frames.length-1);mainVideo.currentTime=index/20;updateFrame(index,true);
}
function pauseVideos(){mainVideo.pause();$("#playButton").textContent='▶';}
async function togglePlayback(){
  if(!state.frames.length)return;
  if(mainVideo.paused){if(mainVideo.ended)seekFrame(0);mainVideo.playbackRate=state.speed;try{await mainVideo.play();$("#playButton").textContent='Ⅱ';}catch(e){showError(`无法播放MP4：${e.message}；可使用“打开MP4”。`);}}
  else pauseVideos();
}
function animationLoop(){
  if(state.frames.length&&!mainVideo.paused){updateFrame(videoFrameIndex(mainVideo.currentTime));$("#timeline").value=mainVideo.currentTime;$("#currentTime").textContent=formatTime(mainVideo.currentTime);}
  requestAnimationFrame(animationLoop);
}
function exportTraceCsv(){
  if(!state.frames.length)return;
  const header=['processed_frame','source_frame','replay_time_s','phase','subtask','execution_mask','x_world_m','y_world_m','z_world_m','rx_axisangle_rad','ry_axisangle_rad','rz_axisangle_rad',...Array.from({length:7},(_,i)=>`q${i+1}_rad`),...Array.from({length:7},(_,i)=>`dq${i+1}_rad_s`),'gripper_aperture_m',...Array.from({length:7},(_,i)=>`raw_action_${i}`)];
  const rows=state.frames.map(f=>[f.index,f.source_frame,f.time_s,f.phase,f.subtask_index??'',f.execution_mask??'',...f.ee_state.position_m,...f.ee_state.rotation_vector_base_rad,...f.joint_state.position_rad,...f.joint_state.velocity_rad_s.map(v=>v??''),f.gripper_state.aperture_m,...f.action].join(','));
  const blob=new Blob([header.join(',')+'\n'+rows.join('\n')+'\n'],{type:'text/csv;charset=utf-8'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=`${state.dataset.suite}_task${state.dataset.task_id}_${state.episode.path}.csv`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}
$("#taskSearch").addEventListener('input',renderDatasetList);$("#downloadCsv").addEventListener('click',exportTraceCsv);
for(const id of ['jointVelocityChart','apertureChart','sourceChart'])$("#"+id).addEventListener('pointerdown',e=>{const c=e.currentTarget;if(!c._plot)return;const r=c.getBoundingClientRect(),ratio=clamp((e.clientX-r.left-c._plot.left)/c._plot.width,0,1);seekFrame(Math.round(ratio*(state.frames.length-1)));});
mainVideo.addEventListener('seeked',()=>{if(state.frames.length)updateFrame(videoFrameIndex(mainVideo.currentTime),true);});
mainVideo.addEventListener('loadedmetadata',()=>{if(state.frames.length)updateFrame(state.frameIndex,true);});
mainVideo.addEventListener('error',()=>{if(mainVideo.getAttribute('src'))showError('MP4加载失败。请保留videos目录；“打开MP4”可单独检查视频。');});
requestAnimationFrame(animationLoop);
loadDatasets();
