"use strict";
// Full audited reconstruction catalog. Annotation availability never filters demos.
let annotationSnapshot={episodes:{},selected:0};
let trainingSnapshot=null;
const trainingQuery=new URLSearchParams(location.search).get("training")==="1";
function trainingSelected(d,e){return LiberoTrainingSelection.selected(trainingSnapshot,d,e);}
const annotationNames={labeled:'标签已生成',no_feasible_label:'未找到合格标签',adapter_error:'适配错误',baseline_unavailable:'零容差基线不可用',running:'搜索中',pending:'待标注'};
document.querySelector('.brand span').textContent='LIBERO · 40任务 · 全量demo';
document.querySelector('.episode-heading > span').textContent='Demo列表';
document.querySelector('.sidebar-foot').textContent=`本地只读 · 完整重构数据集 · ${window.LIBERO_CATALOG.counts.episodes}条成功demo`;
function annotationKey(d,e){return `${d.suite}/${String(d.task_id).padStart(2,'0')}/${e.path}`;}
const filterBox=document.createElement('div');filterBox.className='full-demo-filter';
filterBox.innerHTML='<small>训练入选范围 / 离线标注状态</small><label>Demo范围 <select id="annotationFilter"><option value="all">全部重构demo</option><option value="training" disabled>训练入选容差（加载中）</option><option value="selected">已列入离线标注</option><option value="no_feasible_label">未找到合格标签</option><option value="adapter_error">适配错误</option></select></label><input id="demoSearch" type="search" placeholder="搜索 demo 编号" aria-label="搜索demo编号">';
$('#episodeList').before(filterBox);
if(trainingQuery)$('#annotationFilter').value='training';
const labelStyle=document.createElement('style');labelStyle.textContent='.full-demo-filter{grid-column:2;grid-row:2/6;align-self:start;min-width:0;display:grid;gap:8px;padding:8px 6px;font-size:10px}.libero-viewer .episode-list{grid-row:6/8}.full-demo-filter label{display:grid;gap:5px}.full-demo-filter small{white-space:normal;overflow-wrap:anywhere;min-width:0;font-size:9px;line-height:1.5;color:#91a9ba}.full-demo-filter select,.full-demo-filter input{width:100%;min-width:0;max-width:100%;background:#131c25;color:inherit;border:1px solid #334252;border-radius:6px;padding:7px}.annotation-tag{font-style:normal;font-size:11px;color:#91a9ba}.annotation-tag.no_feasible_label{color:#ffbb72}.annotation-tag.adapter_error{color:#ff8888}.annotation-tag.labeled{color:#78d5a1}';document.head.appendChild(labelStyle);
renderFolderList=function(){
 const select=$('#folderSelect');select.innerHTML='';
 for(const suite of [...new Set(state.datasets.map(d=>d.suite))]){const tasks=state.datasets.filter(d=>d.suite===suite),option=document.createElement('option');option.value=suite;option.textContent=`${suite} · ${tasks.reduce((n,d)=>n+d.episodes.length,0)}条demo`;select.appendChild(option);}
 select.value=state.folder||state.datasets[0]?.suite;const c=window.LIBERO_CATALOG.counts;
 $('#folderSummary').textContent=`完整重构数据集 · ${c.episodes}条成功demo · 40任务 · 训练入选容差${trainingSnapshot?trainingSnapshot.selected:'加载中'}条`;
};
renderDatasetList=function(){
 const container=$('#datasetList');container.innerHTML='';const query=$('#taskSearch').value.trim().toLowerCase();
 for(const suite of [...new Set(state.datasets.map(d=>d.suite))]){
  const items=state.datasets.map((d,i)=>({d,i})).filter(({d})=>d.suite===suite&&(!query||`${d.task_id} ${d.display_name} ${d.task_name} ${d.suite}`.toLowerCase().includes(query)));
  if(!items.length)continue;const group=document.createElement('details');group.className='suite-tree';group.open=!!query||suite===state.folder;
  const title=document.createElement('summary');title.textContent=suite;const note=document.createElement('span');note.textContent=`${items.reduce((n,{d})=>n+d.episodes.length,0)}条demo`;title.append(note);group.append(title);
  for(const {d,i} of items){const button=document.createElement('button');button.className=`dataset-item${state.dataset===d?' active':''}`;button.dataset.taskIndex=i;const h=document.createElement('strong');h.textContent=`${String(d.task_id).padStart(2,'0')} · ${d.display_name}`;const count=document.createElement('span');count.textContent=`${d.episodes.length}条重构demo · 训练入选${d.episodes.filter(e=>trainingSelected(d,e)).length}条容差`;button.append(h,count);button.onclick=()=>selectDataset(i);group.append(button);}container.append(group);
 }
};
renderEpisodeList=function(){
 const container=$('#episodeList');container.innerHTML='';const episodes=state.dataset?.episodes||[];
 const filter=$('#annotationFilter').value,search=$('#demoSearch').value.trim().toLowerCase();let visible=0;
 episodes.forEach((episode,index)=>{const a=annotationSnapshot.episodes[annotationKey(state.dataset,episode)];
  if(search&&!episode.path.toLowerCase().includes(search))return;
  if(filter==='training'&&!trainingSelected(state.dataset,episode))return;
  if(filter==='selected'&&!a)return;if(!['all','selected','training'].includes(filter)&&a?.status!==filter)return;
  visible++;const button=document.createElement('button');button.className=`episode-item${state.episode===episode?' active':''}`;button.dataset.demo=episode.path;
  const number=document.createElement('span');number.className='episode-number';number.textContent=episode.path.replace('demo_','');
  const body=document.createElement('span');body.className='episode-copy';const name=document.createElement('strong');name.textContent=episode.path;
  const info=document.createElement('span');info.textContent=`${episode.frames}帧 · ${(episode.frames/20).toFixed(1)}s`;
  const status=document.createElement('em');status.className=`annotation-tag ${a?.status||''}`;status.textContent=(trainingSelected(state.dataset,episode)?'训练入选容差 · ':'')+(a?annotationNames[a.status]||a.status:annotationSnapshot.updated_at?'未选入本次标注':'标注状态加载中');
  body.append(name,info,document.createElement('br'),status);button.append(number,body);button.onclick=()=>selectEpisode(index);container.append(button);
 });
 $('#episodeCount').textContent=filter==='all'&&!search?episodes.length:`${visible}/${episodes.length}`;
 if(!visible){const note=document.createElement('p');note.textContent='当前筛选没有demo；原始列表仍完整保留。';container.append(note);}
};
async function applyTrainingFilter(){
 const on=$('#annotationFilter').value==='training',u=new URL(location.href);
 if(on)u.searchParams.set('training','1');else u.searchParams.delete('training');
 history.replaceState(null,'',u);renderDatasetList();renderEpisodeList();
 if(on&&state.dataset&&(!state.episode||!trainingSelected(state.dataset,state.episode))){
  const index=state.dataset.episodes.findIndex(e=>trainingSelected(state.dataset,e));
  if(index>=0)await selectEpisode(index);
 }
}
$('#annotationFilter').addEventListener('change',applyTrainingFilter);$('#demoSearch').addEventListener('input',renderEpisodeList);
const trainingSelectDataset=selectDataset;
selectDataset=async function(index,requestedEpisode=null){
 const dataset=state.datasets[index];
 if(dataset&&$('#annotationFilter').value==='training'&&trainingSnapshot){
  const requested=dataset.episodes.find(e=>e.path===requestedEpisode);
  if(!requested||!trainingSelected(dataset,requested))requestedEpisode=dataset.episodes.find(e=>trainingSelected(dataset,e))?.path||null;
 }
 return trainingSelectDataset(index,requestedEpisode);
};
async function loadTrainingSelection(){
 const option=$('#annotationFilter').querySelector('option[value="training"]');
 try{
  const response=await fetch('training_selection.json',{cache:'no-store'});if(!response.ok)throw new Error('Missing selection');
  trainingSnapshot=await response.json();option.disabled=false;option.textContent=`训练入选容差（${trainingSnapshot.selected}条）`;
  renderFolderList();await applyTrainingFilter();
 }catch(_error){option.textContent='训练入选清单不可用';option.disabled=true;if(trainingQuery){$('#annotationFilter').value='all';renderEpisodeList();}}
}
loadTrainingSelection();
// Earlier adapters registered their renderer directly, so bind the full-catalog renderer.
const previousTaskSearch=$('#taskSearch'),fullTaskSearch=previousTaskSearch.cloneNode(true);previousTaskSearch.replaceWith(fullTaskSearch);fullTaskSearch.addEventListener('input',renderDatasetList);
async function refreshAnnotations(){
 try{const response=await fetch(`annotation_status.json?t=${Date.now()}`,{cache:'no-store'});if(!response.ok)return;const next=await response.json();
  if(next.updated_at!==annotationSnapshot.updated_at){annotationSnapshot=next;renderEpisodeList();}}
 catch(_error){/* Static catalog/video remains usable if status bridge is unavailable. */}
}
refreshAnnotations();setInterval(refreshAnnotations,3000);
const fullCatalogConfigureVideos=configureVideos;
configureVideos=function(){fullCatalogConfigureVideos();
 if(typeof activeComparison==='function'&&!activeComparison()){
  $('#toleranceCaption').textContent='本条没有历史容差回放';
  $('#episodeScope').textContent='第1、3格显示无容差主视角与腕部视角；本条没有容差回放，第2、4格留空。离线标注状态单独显示在demo列表。';
 }
};
renderFolderList();renderDatasetList();renderEpisodeList();

if(state.payload)configureVideos();
