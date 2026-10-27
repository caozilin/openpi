"""Adapt the original MuJoCo Dataset Studio document without replacing its UI."""
import re

def adapt_layout(source: str) -> str:
    source=source.replace('<title>TaskTol Dataset Studio</title>','<title>LIBERO-40 · TaskTol Dataset Studio</title>')
    source=source.replace('href="/styles.css"','href="assets/mujoco-studio.css"')
    source=source.replace('</head>','<link rel="stylesheet" href="assets/adapter.css">\n</head>')
    source=source.replace('<body>','<body class="libero-viewer">')
    source=source.replace('Dataset Studio</span>','LIBERO · 4 × 10 × 10</span>')
    source=source.replace('<span>目录</span>','<span>任务集 · 共 400 段视频</span>')
    source=source.replace('Choose a dataset or benchmark run folder','选择 LIBERO 任务集')
    source=source.replace('<div id="datasetList" class="dataset-list"></div>','<div class="task-search"><input id="taskSearch" type="search" placeholder="搜索40个任务 / 指令" aria-label="搜索任务"></div><div id="datasetList" class="dataset-list"></div>')
    source=source.replace('<span>Episodes</span>','<span>10 MP4</span>')
    source=source.replace('本地只读 · Dataset / Benchmark Rollout','本地只读 · 每任务1个共享预测 / 10个成功示范')
    source=source.replace('<span>SEED</span>','<span>DEMO</span>').replace('<span>RESULT</span>','<span>SAVED</span>')
    source=source.replace('轨迹、容差与视频将在同一时间轴上同步显示','视频、观测轨迹和语义预测在同一时间轴上检查')
    source=source.replace('选择左侧 episode 后即可检查双视角视频、末端轨迹与阶段容差。','按任务集 → 任务 → demo 选择。保留原代表视频，每任务补充9条成功示范。')
    source=source.replace('<div id="content" class="content hidden">','''<div id="content" class="content hidden">
<section class="panel shared-mask-panel"><div class="panel-title-row"><div><span class="section-index">00</span><h2>任务共享预测 mask 与判定依据</h2></div><span id="sharedScope" class="status-chip"></span></div>
<div id="sharedMaskSummary" class="shared-summary"></div><div id="subtaskTabs" class="subtask-tabs"></div>
<div class="scope-note">共享预测来自原代表示范；新增9条仅补充视频与轨迹，未逐条视觉标注。标称参考和六方向数值容差未提供。</div>
<details id="reasoningDetails"><summary>展开当前子任务的逐轴判定、参考案例与图片证据</summary><div id="reasoningContent"></div></details></section>''')
    source=source.replace('<h2>同步视觉流</h2>','<h2>主视角视频 · 20 Hz</h2>')
    source=re.sub(r'<figure><video id="wristVideo".*?</figure>','',source,flags=re.S)
    source=source.replace('Main RGB · D435','agentview · 观测先于动作')
    source=source.replace('<div class="transport">','<div id="episodeScope" class="episode-scope"></div><div id="phaseTrack" class="phase-track"></div><div class="transport">')
    source=source.replace('</section>\n\n        <section class="analysis-grid">','''<div class="video-links"><a id="openVideo" target="_blank" rel="noopener">打开 MP4</a><a id="traceDownload" download>轨迹 JSON</a><button id="downloadCsv">导出轨迹 CSV</button><span>空格播放 · ← → 逐帧 · 点击曲线跳转</span></div></section>
<section class="analysis-grid">''')
    source=source.replace('空间轨迹与容差坐标系','空间观测轨迹与重力参考')
    source=source.replace('<span><i class="line nominal"></i>名义目标</span>','')
    source=source.replace('当前帧</span>','当前观测</span>').replace('容差系</span>','F_obs 参考</span>')
    source=source.replace('EE actual · base','EE observed · world / grip_site')
    source=source.replace('Nominal target · base','标称参考可用性')
    source=source.replace('Stage target · base','当前帧的语义标注状态')
    source=source.replace('Tolerance frame · fixed XYZ/RPY (Rz Ry Rx)','重力对齐 F_obs · 观测代理')
    source=source.replace('Action Δxyz / Δrot / grip','原始 OSC_POSE 归一化控制 / grip')
    source=source.replace('<span>Optimizer</span>','<span>源帧 / 对齐</span>')
    source=source.replace('actual + nominal · m','observed world position · m')
    source=source.replace('actual + stage · rad','observed axis-angle vector · rad')
    source=source.replace('任务容差坐标</strong><span>固定轴 XYZ/RPY（Rz·Ry·Rx）与 ±limit · deg','当前代表轨迹的有效 mask</strong><span>二值 xF / yF / zF；补充demo不伪造逐帧标签')
    source=source.replace('<div class="chart-card wide action-card">','''<div class="chart-card wide"><div><strong>关节速度</strong><span>dq/dt · retained observations · rad/s</span></div><canvas id="jointVelocityChart"></canvas></div>
<div class="chart-card"><div><strong>夹爪开口</strong><span>q_left − q_right · mm</span></div><canvas id="apertureChart"></canvas></div>
<div class="chart-card"><div><strong>源帧映射</strong><span>source_frame − processed_frame · filtered-action gaps</span></div><canvas id="sourceChart"></canvas></div>
<div class="chart-card wide action-card">''')
    source=source.replace('VLA / scripted action','LIBERO OSC_POSE controls').replace('三个独立坐标系 · 共享时间游标','原始归一化控制值 · 共享时间游标')
    source=source.replace('3D 平移</strong><span>ΔX / ΔY / ΔZ · mm','平移控制</strong><span>normalized ΔX / ΔY / ΔZ')
    source=source.replace('3D 旋转</strong><span>ΔRx / ΔRy / ΔRz · deg','旋转控制</strong><span>normalized ΔRx / ΔRy / ΔRz')
    source=source.replace('<script src="/app.js"></script>','''<script src="assets/catalog.js"></script><script src="assets/mujoco-studio.js"></script><script src="assets/adapter.js"></script>''')
    return source
