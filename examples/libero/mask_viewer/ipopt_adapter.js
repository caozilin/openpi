"use strict";
// Extend the existing local MuJoCo Studio adapter with recorded IPOPT semantics.
LIBERO_PHASE_LABEL.solver_rejected='求解拒绝 · 未执行';
LIBERO_PHASE_COLORS.solver_rejected='#ff6b72';
Object.assign(LIBERO_PHASE_LABEL,{pregrasp:'APP / PRE · 开口稳定',grasp:'ENG / GRASP · 闭合过渡',postgrasp:'EXEC / POST · 闭口稳定',release:'DIS / RELEASE · 张开过渡'});
const legacyPhaseTrack=renderPhaseTrack;
renderPhaseTrack=function(){legacyPhaseTrack();const groups=[];let start=0;const f=state.frames;
  for(let i=1;i<=f.length;i++)if(i===f.length||f[i].phase!==f[start].phase||f[i].observed_subtask_index!==f[start].observed_subtask_index){groups.push([start,i]);start=i;}
  const bar=$('#phaseTrack');bar.innerHTML='';for(const[a,b]of groups){const p=f[a].phase,button=document.createElement('button');button.style.width=`${100*(b-a)/f.length}%`;button.style.background=LIBERO_PHASE_COLORS[p]||'#65717c';button.title=`观测循环 ${f[a].observed_subtask_index} · ${LIBERO_PHASE_LABEL[p]} · [${a},${b}) · ${format(a/20,2)}–${format(b/20,2)}s`;button.onclick=()=>seekFrame(a);bar.appendChild(button);}
};
const originalFolderList=renderFolderList;
renderFolderList=function(){originalFolderList();const c=window.LIBERO_CATALOG.counts;$('#folderSummary').textContent=`IPOPT · ${c.tasks}任务 / ${c.episodes}视频 · ${c.success}成功 / ${c.failures}失败`;};
const originalDatasetList=renderDatasetList;
renderDatasetList=function(){originalDatasetList();$$('#datasetList button[data-task-index]').forEach(b=>{const d=state.datasets[Number(b.dataset.taskIndex)];b.querySelector('span').textContent=`${d.success_count}/10 成功 · ${d.shared_annotation.subtasks.length}子任务 · 共享预测`;});};
renderEpisodeList=function(){
  const c=$('#episodeList');c.innerHTML='';const episodes=state.dataset?.episodes||[];$('#episodeCount').textContent=episodes.length;
  episodes.forEach((e,i)=>{const b=document.createElement('button');b.className=`episode-item${state.episode===e?' active':''}`;b.dataset.demo=e.path;
    b.innerHTML=`<span class="episode-number">${String(i+1).padStart(2,'0')}</span><span class="episode-copy"><strong>${escapeHtml(e.path)}</strong><span>${e.success?'成功':e.execution_rejected?'求解拒绝':'任务未完成'} · ${escapeHtml(e.selected_trial)}<br>${e.frames}帧 · ${format(e.frames/20,1)}s</span></span><i class="episode-result ${e.success?'success':'failure'}" style="background:${e.success?'#48d9aa':'#ff6b72'}"></i>`;
    b.onclick=()=>selectEpisode(i);c.appendChild(b);});
};
const originalSelectEpisode=selectEpisode;
selectEpisode=async function(index){await originalSelectEpisode(index);if(state.episode)$('#syncStatus').textContent=state.episode.success?'IPOPT · SUCCESS':state.episode.execution_rejected?'IPOPT · REJECTED':'IPOPT · FAILED';};
configureVideos=function(){
  const e=state.episode;mainVideo.src=e.video_path;mainVideo.poster=e.poster_path;mainVideo.playbackRate=state.speed;mainVideo.load();
  $('#timeline').max=state.frames.length/20;$('#timeline').step=.05;$('#timeline').value=0;$('#durationTime').textContent=formatTime(state.frames.length/20);$('#playButton').textContent='▶';
  $('#openVideo').href=e.video_path;$('#traceDownload').href=e.trace_path;
  $('#episodeScope').classList.toggle('unreviewed',!e.success);
  $('#episodeScope').textContent=`IPOPT ${e.success?'成功':e.execution_rejected?'求解拒绝':'任务未完成'} · ${e.selected_trial} · 阶段复用MuJoCo分类器：20Hz逐帧因果观测，连续3帧稳定判定。动作前视频／动作后误差。原OSC语义边界未迁移；容差关闭。`;
};
renderHeader=function(){
  $('#sceneName').textContent=`LIBERO / ${state.dataset.suite} / TASK ${state.dataset.task_id} / ${state.episode.path} / IPOPT`;
  $('#taskName').textContent=state.dataset.task_name;$('#instructionText').textContent=state.dataset.display_name;
  $('#seedValue').textContent=state.episode.path;$('#frameCountValue').textContent=state.frames.length;
  $('#resultValue').textContent=state.episode.success?'SUCCESS':state.episode.execution_rejected?'REJECTED':'FAILED';$('#resultValue').className=state.episode.success?'success':'failure';
};
renderMetadata=function(){
  const c=$('#episodeMetadata');c.innerHTML='';const p=state.payload,e=state.episode,r=p.episode.result,t=p.metrics.terms;
  const values=[['录制批次','task_pose_default_ipopt400_fixed_v2 · 最终选定结果'],['结果 / 候选',`${e.success?'成功':e.execution_rejected?'求解拒绝':'任务未完成'} · ${e.selected_trial}`],
    ['执行步数 / 计划',`${r.executed_frames} / ${r.expected_frames} · 视频含拒绝尝试帧`],['采样 / 物理步长','20 Hz动作；每步25个2ms物理步；原速'],
    ['求解器','默认 IPOPT · 20迭代 / 95ms CPU预算 · 未修改'],['执行控制','完整关节参考 · OSC任务控制 + 关节零空间 · B10'],['容差执行','关闭；任务共享mask仅供检查'],
    ['重构动作','累积独立参考的物理增量 [m, rad]；夹爪沿用原指令'],['初始化','0次额外settling；保留等待动作；缺下一state的末条动作不执行'],
    ['同期源状态位置均值',`${format((t.ee_position_m.mean_norm||0)*1000,3)} mm`],['同期源状态旋转均值',`${format((t.ee_rotation_rad.mean_norm||0)*180/Math.PI,3)}°`],
    ['阶段判定','直接复用MuJoCo PhaseClassifier；20Hz逐帧，开口变化阈值0.2mm，3个稳定样本；稳定阶段锁存到指令改变'],
    ['阶段范围','基于历史观测重算；未用于当时执行。观测循环编号不等同于共享mask的语义子任务'],
    ['视频/误差对齐','图像和实际姿态为pre-action；6维误差为post-action与当前命令参考；拒绝帧为空'],['原生观测缓存','原生观测可能落后物理终态一个2ms tick；沿用录制记录'],
    ['最终候选路径',e.selected_root],['源示范文件',e.source_hdf5],['共享语义证据',`原OSC ${state.dataset.representative_demo_id}；不转移时序标注`],['原标注SHA256',window.LIBERO_CATALOG.annotation_sha256]];
  values.forEach(([k,v])=>c.appendChild(metadataItem(k,v)));
  const links=document.createElement('div');links.className='video-links';
  for(const [label,path] of [['初始截图',e.poster_path],['执行后终态截图',e.final_path],['原始轨迹 JSON',e.raw_trace_path],['执行报告',e.report_path]]){const a=document.createElement('a');a.textContent=label;a.href=path;a.target='_blank';links.appendChild(a);}c.appendChild(links);
};
const originalReasoning=renderReasoning;
renderReasoning=function(){originalReasoning();const c=$('#reasoningContent');
  c.insertAdjacentHTML('afterbegin','<p class="warning">以下分段帧号和截图均属于原OSC代表示范，作为任务语义依据保留。截图按钮打开原证据图片，不跳转当前IPOPT视频；这些阶段未转移到重构轨迹。</p>');
  c.querySelectorAll('button[data-evidence-frame]').forEach(b=>{b.onclick=()=>window.open(b.querySelector('img').src,'_blank','noopener');b.title='打开原OSC证据截图';});
  const note=[...c.querySelectorAll('p')].find(p=>p.textContent==='点击截图会切回原代表示范并定位该帧。');if(note)note.textContent='点击截图打开原OSC证据；帧号不属于当前IPOPT视频。';
};
const originalShared=renderSharedPrediction;
renderSharedPrediction=function(){originalShared();$('#sharedScope').textContent=`任务语义证据 · 原OSC ${activeTask().demo_id} · 容差OFF`;const label=$('#subtaskTabs .follow-control');if(label)label.remove();};
buildDerivedFrames=function(){const frames=state.frames;state.chartCache={
  positionActual:[0,1,2].map(i=>frames.map(f=>f.ee_state.position_m[i])),rotationActual:[0,1,2].map(i=>frames.map(f=>f.ee_state.rotation_vector_base_rad[i])),
  positionNominal:[0,1,2].map(i=>frames.map(f=>f.nominal_target_pose.position_m[i])),rotationNominal:[0,1,2].map(i=>frames.map(f=>f.nominal_target_pose.rotation_vector_base_rad[i])),
  joints:[0,1,2,3,4,5,6].map(i=>frames.map(f=>f.joint_state.position_rad[i])),velocities:[0,1,2,3,4,5,6].map(i=>frames.map(f=>f.joint_state.velocity_rad_s[i])),
  actionsT:[0,1,2].map(i=>frames.map(f=>f.action[i])),actionsR:[3,4,5].map(i=>frames.map(f=>f.action[i])),actionGripper:[frames.map(f=>f.action[6])],aperture:[frames.map(f=>f.gripper_state.aperture_m*1000)],
  postP:[0,1,2].map(i=>frames.map(f=>f.post_target_error_position_mm?.[i]??null)),postR:[0,1,2].map(i=>frames.map(f=>f.post_target_error_rotation_deg?.[i]??null)),
  solver:frames.map(f=>f.optimizer.elapsed_ms),sourceError:frames.map(f=>f.source_next_position_error_mm)};};
renderInspector=function(){const f=state.frames[state.frameIndex];if(!f)return;
  const b=$('#phaseBadge');b.textContent=`观测循环 ${f.observed_subtask_index} · ${LIBERO_PHASE_LABEL[f.phase]||f.phase}${f.aborted?' · 求解拒绝／未执行':''}`;b.className='phase-badge';
  renderPose($('#actualPose'),f.ee_state);renderPose($('#nominalPose'),f.nominal_target_pose);
  $('#stagePose').className='frame-facts';$('#stagePose').textContent=f.aborted?'该帧求解拒绝，未执行物理步；执行后误差不适用。':`位置 XYZ [mm]：${f.post_target_error_position_mm.map(v=>format(v,3)).join(', ')}\n旋转 XYZ [deg]：${f.post_target_error_rotation_deg.map(v=>format(v,3)).join(', ')}\nLog(R实际 R参考ᵀ) · 世界系`;
  $('#tolerancePose').className='frame-facts';$('#tolerancePose').textContent=f.aborted?'未执行，无下一状态误差。':`同期下一源状态：\n位置 ${format(f.source_next_position_error_mm,3)} mm\n旋转 ${format(f.source_next_rotation_error_deg,3)}°\n容差关闭；未套用原OSC阶段`;
  $('#toleranceLimits').innerHTML='';
  vectorCells($('#actionVector'),['ΔX m','ΔY m','ΔZ m','ΔRx rad','ΔRy rad','ΔRz rad','Grip'],f.action,'');
  $('#gripperValue').textContent=`${format(f.gripper_state.aperture_m*1000,2)} mm · q左−q右`;
  $('#optimizerValue').textContent=`${f.optimizer.iterations} iter · ${format(f.optimizer.elapsed_ms,2)} ms · ${f.optimizer.feasible?'可行':'不可行'}${f.aborted?' · 拒绝':''}`;
  $('#optimizerValue').title=f.optimizer.status;
};
drawSpatial=function(){if(!state.frames.length)return;const{ctx,width,height}=canvasContext(spatialCanvas);ctx.clearRect(0,0,width,height);
  const actual=state.frames.map(f=>f.ee_state.position_m),target=state.frames.map(f=>f.nominal_target_pose.position_m),all=[...actual,...target];
  const mins=[0,1,2].map(i=>Math.min(...all.map(p=>p[i]))-.04),maxs=[0,1,2].map(i=>Math.max(...all.map(p=>p[i]))+.04),center=mins.map((v,i)=>(v+maxs[i])/2),viewed=all.map(p=>viewPoint(subtract(p,center)));
  const su=Math.max(.12,Math.max(...viewed.map(p=>p[0]))-Math.min(...viewed.map(p=>p[0]))),sv=Math.max(.12,Math.max(...viewed.map(p=>p[1]))-Math.min(...viewed.map(p=>p[1]))),scale=Math.min((width-90)/su,(height-80)/sv)*state.view.zoom;
  const projection={mins,maxs,center,project(p){const v=viewPoint(subtract(p,center));return[width/2+v[0]*scale,height/2-v[1]*scale,v[2]];}};
  drawFrame3D(ctx,projection,mins,[[1,0,0],[0,1,0],[0,0,1]],.05,.65,'WORLD XYZ');
  stroke3D(ctx,projection,target,'rgba(255,177,90,.55)',1.2);stroke3D(ctx,projection,actual,'rgba(72,217,220,.25)',1);stroke3D(ctx,projection,actual.slice(0,state.frameIndex+1),'#48d9dc',2.4);
  const f=state.frames[state.frameIndex],p=f.ee_state.position_m;drawFrame3D(ctx,projection,p,rotvecMatrix(f.ee_state.rotation_vector_base_rad),.045,.8,'ACTUAL');
  const q=f.nominal_target_pose.position_m;stroke3D(ctx,projection,[p,q],'#ffb15a',1.8);
  for(const [point,color]of [[p,'#48d9dc'],[q,'#ffb15a']]){const[x,y]=projection.project(point);ctx.beginPath();ctx.arc(x,y,4,0,Math.PI*2);ctx.fillStyle=color;ctx.fill();}
  $('#axisReadout').textContent=`WORLD [m]\n青色：实际（动作前）\n橙色：当前命令参考\n\n帧 ${f.index} / src ${f.source_frame}\n${f.aborted?'REJECTED':'20 Hz · IPOPT'}\nview ${state.view.mode}`;
};
drawCharts=function(){const c=state.chartCache;if(!c)return;
  drawLineChart($('#positionChart'),[...axisSeries(c.positionActual),...axisSeries(c.positionNominal,true,.48)]);
  drawLineChart($('#rotationChart'),[...axisSeries(c.rotationActual),...axisSeries(c.rotationNominal,true,.48)]);
  drawLineChart($('#toleranceChart'),[{values:c.solver,color:COLORS[4]}]);
  drawLineChart($('#jointChart'),c.joints.map((values,i)=>({values,color:COLORS[i],width:1.15,alpha:.82})));
  drawLineChart($('#jointVelocityChart'),c.velocities.map((values,i)=>({values,color:COLORS[i],width:1.15,alpha:.82})));
  drawLineChart($('#apertureChart'),[{values:c.aperture[0],color:COLORS[4]}]);drawLineChart($('#sourceChart'),[{values:c.sourceError,color:COLORS[3]}]);
  drawLineChart($('#actionTranslationChart'),axisSeries(c.actionsT));drawLineChart($('#actionRotationChart'),axisSeries(c.actionsR));
  drawLineChart($('#actionGripperChart'),[{values:c.actionGripper[0],color:COLORS[4]}],{min:-1.05,max:1.05,padding:false});
  drawLineChart($('#postPositionErrorChart'),axisSeries(c.postP));drawLineChart($('#postRotationErrorChart'),axisSeries(c.postR));
};
exportTraceCsv=function(){const fields=['frame','source_frame','time_s','rejected','x_m','y_m','z_m','rvx_rad','rvy_rad','rvz_rad','target_x_m','target_y_m','target_z_m','target_rvx_rad','target_rvy_rad','target_rvz_rad','post_dx_mm','post_dy_mm','post_dz_mm','post_rx_deg','post_ry_deg','post_rz_deg','source_next_position_error_mm','source_next_rotation_error_deg','solver_ms','solver_iterations',...Array.from({length:7},(_,i)=>`q${i+1}_rad`),...Array.from({length:7},(_,i)=>`action_${i}`)];
  const rows=state.frames.map(f=>[f.index,f.source_frame,f.time_s,f.aborted,...f.ee_state.position_m,...f.ee_state.rotation_vector_base_rad,...f.nominal_target_pose.position_m,...f.nominal_target_pose.rotation_vector_base_rad,...(f.post_target_error_position_mm||['','','']),...(f.post_target_error_rotation_deg||['','','']),f.source_next_position_error_mm??'',f.source_next_rotation_error_deg??'',f.optimizer.elapsed_ms,f.optimizer.iterations,...f.joint_state.position_rad,...f.action].join(','));
  const blob=new Blob([fields.join(',')+'\n'+rows.join('\n')+'\n'],{type:'text/csv;charset=utf-8'}),u=URL.createObjectURL(blob),a=document.createElement('a');a.href=u;a.download=`${state.episode.key.replaceAll('/','_')}_ipopt.csv`;a.click();setTimeout(()=>URL.revokeObjectURL(u),1000);
};
// Original click listener captured the old export function; replace it explicitly.
const oldDownload=$('#downloadCsv'),newDownload=oldDownload.cloneNode(true);oldDownload.replaceWith(newDownload);newDownload.onclick=()=>exportTraceCsv();
for(const id of ['postPositionErrorChart','postRotationErrorChart'])$('#'+id).addEventListener('pointerdown',e=>{const c=e.currentTarget;if(!c._plot)return;const r=c.getBoundingClientRect(),ratio=clamp((e.clientX-r.left-c._plot.left)/c._plot.width,0,1);seekFrame(Math.round(ratio*(state.frames.length-1)));});
requestAnimationFrame(animationLoop);
loadDatasets();
