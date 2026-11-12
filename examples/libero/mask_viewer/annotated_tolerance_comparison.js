"use strict";
// Display-only adaptation for the new per-demo annotated dataset.
if(window.LIBERO_CATALOG.tolerance_kind==='offline_annotated_signed_bounds'){
  recoveryLabel.classList.add('hidden');
  const annotatedConfigure=configureVideos;
  configureVideos=function(){
    annotatedConfigure();const t=activeComparison();
    $('#toleranceCaption').textContent=t?`标注容差 · ${t.result.success?'成功':t.result.execution_rejected?'求解拒绝':'任务未完成'} · ${t.result.executed_frames}/${t.result.planned_frames}帧`:'本条无有效标签，未生成容差回放';
    $('#episodeScope').textContent='从左到右：无容差主视角、标注容差主视角、无容差腕部、标注容差腕部。沿原始源帧使用阶段、子任务及六方向搜索边界，原动作与初态不变。';
  };
  renderSharedPrediction=function(){
    const t=activeComparison();$('#sharedScope').textContent='逐demo搜索标注 · 六方向边界';
    $('#sharedMaskSummary').textContent=t?(t.label_subtasks||[]).map((s,i)=>`子任务${i+1}：PRE [${s.pregrasp.map(x=>format(x,2)).join(', ')}]° / POST [${s.postgrasp.map(x=>format(x,2)).join(', ')}]°`).join('；'):'无有效标注；名义轨迹仍保留。';
    renderReasoning();
  };
  renderReasoning=function(){
    const c=$('#reasoningContent');c.textContent='每条使用已审计的搜索边界，沿原始源帧及阶段对齐；严格轴保持零，ENG/DIS不释放。物理执行连续运行，包含任务物体；离线运动学可解不等于任务一定成功。所有成败结果均保留。';
  };
  const annotatedInspector=renderInspector;
  renderInspector=function(){annotatedInspector();$('#tolerancePose').textContent=$('#tolerancePose').textContent.replace('30°容差','标注容差');};
  renderMetadata=function(){
    baselineMetadata();const c=$('#episodeMetadata'),t=activeComparison();
    c.appendChild(metadataItem('容差来源','最新离线搜索标注，逐源帧应用；原固定30°及姿态恢复组已替换。'));
    if(t){c.appendChild(metadataItem('容差回放',`${t.result.success?'成功':t.result.execution_rejected?'求解拒绝':'任务未完成'}；执行${t.result.executed_frames}/${t.result.planned_frames}帧`));
      for(const [url,title] of [[t.report,'容差执行报告'],[t.trajectory,'容差逐帧轨迹']]){const a=document.createElement('a');a.href=url;a.textContent=title;a.target='_blank';c.appendChild(a);}}
  };
  renderFolderList=function(){baselineRenderFolders();const c=window.LIBERO_CATALOG.counts;$('#folderSummary').textContent=`标注容差 ${c.tolerance_completed||0}/${window.LIBERO_CATALOG.tolerance_target}条 · 成功${c.tolerance_success||0} · 无标签24条保留名义视频`;};
  const annotatedSelect=selectEpisode;
  selectEpisode=async function(i){await annotatedSelect(i);const t=activeComparison();$('#syncStatus').textContent=t?`标注容差 · ${t.result.success?'SUCCESS':t.result.execution_rejected?'REJECTED':'FAILED'}`:'无有效标签';};
}
