"use strict";

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const COLORS = ["#ff6b72", "#73db9b", "#6ca7ff", "#ffb15a", "#c397ff", "#43d8d5", "#d6df75"];
const PHASE_CLASS = {pregrasp: "pregrasp", grasp: "grasp", postgrasp: "postgrasp", release: "release"};
const PHASE_NAME = {pregrasp: "PRE-GRASP", grasp: "GRASP", postgrasp: "POST-GRASP", release: "RELEASE"};
const FALLBACK_MAIN_CAMERA = {
  right_world: [0, 1, 0],
  up_world: [-0.480000205, 0, 0.877268376],
  forward_world: [-0.877268376, 0, -0.480000205],
};
// Start close to the canonical RGB camera, but lower its downward pitch and
// add a small side angle so planar motion still reads as a 3D trajectory.
const DEFAULT_SPATIAL_YAW_RAD = 15 * Math.PI / 180;
const DEFAULT_SPATIAL_PITCH_RAD = 12 * Math.PI / 180;

const state = {
  datasets: [], folder: null, dataset: null, episode: null, payload: null, frames: [], annotations: [],
  frameIndex: 0, renderedFrame: -1, speed: 1, dragging: false, toleranceFilter: "all", episodeRequest: 0,
  view: {...FALLBACK_MAIN_CAMERA, zoom: 1.0, mode: "FRONT_VIEW"}, chartCache: null,
};

const mainVideo = $("#mainVideo");
const wristVideo = $("#wristVideo");
const spatialCanvas = $("#spatialCanvas");

async function fetchJSON(url) {
  const response = await fetch(url, {priority: url.startsWith('/api/episode?') ? 'high' : 'auto'});
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
  return payload;
}

function showError(message) {
  const toast = $("#toast");
  toast.textContent = message;
  toast.classList.add("show");
  clearTimeout(showError.timer);
  showError.timer = setTimeout(() => toast.classList.remove("show"), 4500);
}

function encodeQuery(values) { return new URLSearchParams(values).toString(); }
function clamp(value, low, high) { return Math.max(low, Math.min(high, value)); }
function degrees(value) { return value * 180 / Math.PI; }
function format(value, digits = 3) { return Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : "—"; }
function formatTime(seconds) {
  const value = Math.max(0, Number(seconds) || 0);
  return `${String(Math.floor(value / 60)).padStart(2, "0")}:${(value % 60).toFixed(1).padStart(4, "0")}`;
}

function matVec(matrix, vector) {
  return matrix.map(row => row[0] * vector[0] + row[1] * vector[1] + row[2] * vector[2]);
}
function transpose(matrix) { return matrix[0].map((_, column) => matrix.map(row => row[column])); }
function matMul(a, b) { return a.map(row => transpose(b).map(column => row.reduce((sum, value, i) => sum + value * column[i], 0))); }
function add(a, b) { return a.map((value, index) => value + b[index]); }
function subtract(a, b) { return a.map((value, index) => value - b[index]); }
function scaleVector(a, scalar) { return a.map(value => value * scalar); }
function norm(a) { return Math.hypot(...a); }
function dot(a, b) { return a.reduce((sum, value, index) => sum + value * b[index], 0); }
function normalized(a) { const length = norm(a); return length > 1e-12 ? scaleVector(a, 1/length) : [0,0,0]; }
function rotateAroundAxis(vector, axis, angle) {
  const unit = normalized(axis), c = Math.cos(angle), s = Math.sin(angle);
  const cross = [
    unit[1]*vector[2]-unit[2]*vector[1],
    unit[2]*vector[0]-unit[0]*vector[2],
    unit[0]*vector[1]-unit[1]*vector[0],
  ];
  return add(add(scaleVector(vector,c),scaleVector(cross,s)),scaleVector(unit,dot(unit,vector)*(1-c)));
}

function resetSpatialView() {
  const camera = state.payload?.viewer?.default_main_camera || FALLBACK_MAIN_CAMERA;
  const worldZ = [0,0,1];
  const rightYaw = normalized(rotateAroundAxis(
    camera.right_world, worldZ, DEFAULT_SPATIAL_YAW_RAD,
  ));
  const upYaw = rotateAroundAxis(
    camera.up_world, worldZ, DEFAULT_SPATIAL_YAW_RAD,
  );
  const forwardYaw = rotateAroundAxis(
    camera.forward_world, worldZ, DEFAULT_SPATIAL_YAW_RAD,
  );
  state.view = {
    right_world: rightYaw,
    up_world: normalized(rotateAroundAxis(
      upYaw, rightYaw, DEFAULT_SPATIAL_PITCH_RAD,
    )),
    forward_world: normalized(rotateAroundAxis(
      forwardYaw, rightYaw, DEFAULT_SPATIAL_PITCH_RAD,
    )),
    zoom: 1.0,
    mode: "FRONT_VIEW + 3D",
  };
}

function rotvecMatrix(vector) {
  const angle = norm(vector);
  if (angle < 1e-10) return [[1,0,0],[0,1,0],[0,0,1]];
  const [x, y, z] = vector.map(value => value / angle);
  const c = Math.cos(angle), s = Math.sin(angle), d = 1 - c;
  return [
    [c + x*x*d, x*y*d - z*s, x*z*d + y*s],
    [y*x*d + z*s, c + y*y*d, y*z*d - x*s],
    [z*x*d - y*s, z*y*d + x*s, c + z*z*d],
  ];
}

function toleranceFrameMatrix(annotation) {
  if (annotation?.viewer_tolerance_rotation_matrix) return annotation.viewer_tolerance_rotation_matrix;
  const nominal = rotvecMatrix(annotation.stage_target_pose.rotation_vector_base_rad);
  let heading = [nominal[0][1], nominal[1][1]];
  if (Math.hypot(...heading) < 1e-8) heading = [nominal[0][0], nominal[1][0]];
  const length = Math.hypot(...heading);
  const c = heading[0] / length, s = heading[1] / length;
  return [[s,c,0],[-c,s,0],[0,0,1]];
}

function fixedXyzAngles(matrix) {
  // Principal [roll, pitch, yaw] for R = Rz(yaw) Ry(pitch) Rx(roll).
  const horizontal = Math.hypot(matrix[0][0], matrix[1][0]);
  const pitch = Math.atan2(-matrix[2][0], horizontal);
  if (horizontal > 1e-10) {
    return [
      Math.atan2(matrix[2][1], matrix[2][2]),
      pitch,
      Math.atan2(matrix[1][0], matrix[0][0]),
    ];
  }
  const roll = matrix[2][0] < 0
    ? Math.atan2(matrix[0][1], matrix[0][2])
    : Math.atan2(-matrix[0][1], -matrix[0][2]);
  return [roll, pitch, 0];
}

function annotationFor(frame) {
  if (frame.viewer_tolerance_annotation) return frame.viewer_tolerance_annotation;
  if (Array.isArray(frame.rotation_tolerance_rad)) return {
    stage_target_pose: frame.nominal_target_pose,
    rotation_tolerance_rad: frame.rotation_tolerance_rad,
  };
  return state.annotations.find(item => Number(item.id) === Number(frame.stage_annotation_id)) || null;
}

function limitsFor(annotation) {
  if (!annotation) return [0, 0, 0, 0, 0, 0];
  if (Array.isArray(annotation.rotation_tolerance_rad)) return annotation.rotation_tolerance_rad;
  return [0, 0, 0, 0, 0, 0];
}

function limitsForFrame(frame, annotation) {
  const limits = limitsFor(annotation);
  if (annotation?.viewer_respect_runtime_ranged_axes === false) return limits;
  const ranged = frame.tolerance?.ranged_axes;
  if (!Array.isArray(ranged)) return limits;
  return limits.map((value, index) => ranged[Math.floor(index / 2)] ? value : 0);
}

function toleranceError(frame, annotation) {
  if (Array.isArray(frame.tolerance?.actual_task_error_rad)) {
    return frame.tolerance.actual_task_error_rad;
  }
  if (!annotation) return [0, 0, 0];
  const actual = rotvecMatrix(frame.ee_state.rotation_vector_base_rad);
  const target = rotvecMatrix(annotation.stage_target_pose.rotation_vector_base_rad);
  const tolerance = toleranceFrameMatrix(annotation);
  const deltaTolerance = matMul(
    matMul(matMul(transpose(tolerance), actual), transpose(target)),
    tolerance,
  );
  return fixedXyzAngles(deltaTolerance);
}

function buildDerivedFrames() {
  state.frames.forEach(frame => {
    const annotation = annotationFor(frame);
    frame._annotation = annotation;
    frame._limits = annotation ? limitsForFrame(frame, annotation) : null;
    frame._toleranceError = annotation ? toleranceError(frame, annotation) : null;
  });
  state.chartCache = {
    positionActual: [0,1,2].map(i => state.frames.map(frame => frame.ee_state.position_m[i])),
    positionNominal: [0,1,2].map(i => state.frames.map(frame => frame.nominal_target_pose.position_m[i])),
    rotationActual: [0,1,2].map(i => state.frames.map(frame => frame.ee_state.rotation_vector_base_rad[i])),
    rotationStage: [0,1,2].map(i => state.frames.map(frame => frame._annotation?.stage_target_pose.rotation_vector_base_rad[i] ?? null)),
    toleranceError: [0,1,2].map(i => state.frames.map(frame => frame._toleranceError ? degrees(frame._toleranceError[i]) : null)),
    tolerancePositive: [0,1,2].map(i => state.frames.map(frame => frame._limits ? degrees(frame._limits[2*i+1]) : null)),
    toleranceNegative: [0,1,2].map(i => state.frames.map(frame => frame._limits ? -degrees(frame._limits[2*i]) : null)),
    joints: Array.from(
      {length: state.frames[0]?.joint_state?.position_rad?.length || 0},
      (_, i) => state.frames.map(frame => frame.joint_state.position_rad[i]),
    ),
    actionTranslationMm: [0,1,2].map(i => state.frames.map(frame => frame.action[i] * 1000)),
    actionRotationDeg: [3,4,5].map(i => state.frames.map(frame => degrees(frame.action[i]))),
    actionGripper: [state.frames.map(frame => frame.action[6])],
  };
}

async function loadDatasets() {
  try {
    const payload = await fetchJSON("/api/datasets");
    state.datasets = payload.datasets;
    if (!state.datasets.length) {
      state.folder = null;
      renderFolderList();
      renderDatasetList();
      $("#emptyState h2").textContent = "未发现 TaskTol 数据集";
      $("#emptyState p").textContent = "指定的目录中需要包含 manifest.json 与 task_metadata.json。";
      return;
    }
    const query = new URLSearchParams(location.search);
    const filter = query.get("execution");
    state.toleranceFilter = ["nominal", "tolerance"].includes(filter) ? filter : "all";
    $("#toleranceFilter").value = state.toleranceFilter;
    const requested = query.get("dataset");
    const requestedSource = query.get("source") || "dataset";
    const index = Math.max(0, state.datasets.findIndex(item => (
      item.relative_path === requested && item.source_type === requestedSource
    )));
    state.folder = folderForDataset(state.datasets[index]);
    renderFolderList();
    await selectDataset(index, query.get("episode"));
  } catch (error) { showError(error.message); }
}

function folderForDataset(dataset) {
  const parts = String(dataset.relative_path || "").split("/").filter(Boolean);
  if (dataset.source_type === "benchmark") {
    const tasksIndex = parts.indexOf("tasks");
    const jobsIndex = parts.indexOf("jobs");
    const runParts = tasksIndex > 0 ? parts.slice(0, tasksIndex)
      : jobsIndex > 0 ? parts.slice(0, jobsIndex + 2) : parts;
    return `benchmark:${runParts.join("/")}`;
  }
  const robot = String(dataset.robot || "other");
  const robotIndex = parts.indexOf(robot);
  const directory = robotIndex >= 0 ? parts.slice(0, robotIndex + 1) : [robot];
  return `dataset:${directory.join("/")}`;
}

function episodeExecutionType(episode, dataset) {
  const variant = episode.execution_variant || episode.trajectory_type;
  if (variant === "nominal") return "nominal";
  if (["tolerance", "boundary_tolerance"].includes(variant)) return "tolerance";
  if (typeof episode.trajectory_is_tolerance === "boolean") {
    return episode.trajectory_is_tolerance ? "tolerance" : "nominal";
  }
  if (dataset.source_type === "benchmark") {
    return (episode.tolerance_mode || dataset.tolerance_mode || "off") === "off" ? "nominal" : "tolerance";
  }
  // Legacy nominal manifests predate execution flags; labels do not imply release.
  return "nominal";
}

function filteredEpisodes(dataset) {
  return (dataset?.episodes || []).filter(episode => (
    state.toleranceFilter === "all" || episodeExecutionType(episode, dataset) === state.toleranceFilter
  ));
}

function clearEpisode() {
  state.episodeRequest += 1;
  pauseVideos();
  state.episode = null;
  state.payload = null;
  state.frames = [];
  state.annotations = [];
  state.chartCache = null;
  mainVideo.removeAttribute("src");
  wristVideo.removeAttribute("src");
  mainVideo.load(); wristVideo.load();
  $("#content").classList.add("hidden");
  $("#emptyState").classList.remove("hidden");
  $("#emptyState h2").textContent = "没有符合筛选条件的 episode";
  $("#emptyState p").textContent = "切换执行类型或选择其他数据集。";
  $("#sceneName").textContent = state.dataset?.scene || "NO DATASET";
  $("#taskName").textContent = state.dataset?.display_name || "选择一个数据集";
  $("#instructionText").textContent = "没有符合筛选条件的 episode";
  ["#seedValue", "#frameCountValue", "#resultValue"].forEach(selector => {
    $(selector).textContent = "—";
    $(selector).classList.remove("success");
  });
  const url = new URL(location.href);
  url.searchParams.delete("episode");
  if (state.dataset) {
    url.searchParams.set("dataset", state.dataset.relative_path);
    url.searchParams.set("source", state.dataset.source_type || "dataset");
  }
  history.replaceState(null, "", url);
}

async function selectToleranceFilter(filter) {
  state.toleranceFilter = filter;
  const url = new URL(location.href);
  if (filter === "all") url.searchParams.delete("execution");
  else url.searchParams.set("execution", filter);
  history.replaceState(null, "", url);
  renderFolderList();
  renderDatasetList();
  renderEpisodeList();
  if (state.episode && filteredEpisodes(state.dataset).includes(state.episode)) return;
  const current = state.datasets.indexOf(state.dataset);
  const index = filteredEpisodes(state.dataset).length ? current : state.datasets.findIndex(dataset => (
    folderForDataset(dataset) === state.folder && filteredEpisodes(dataset).length
  ));
  if (index >= 0) await selectDataset(index);
  else { clearEpisode(); renderEpisodeList(); }
}

function folderDetails(key) {
  const separator = key.indexOf(":");
  const source = key.slice(0, separator);
  const path = key.slice(separator + 1);
  const parts = path.split("/").filter(Boolean);
  if (source === "benchmark") {
    return {
      source, path,
      group: parts.slice(0, -1).join(" / ") || "Benchmark",
      label: parts.at(-1) || path,
      title: path,
    };
  }
  return {
    source, path, group: "Datasets",
    label: parts.join(" / ") || path, title: path,
  };
}

function datasetFolders() {
  const folders = new Map();
  const directoryTimes = new Map();
  state.datasets.forEach(dataset => {
    const name = folderForDataset(dataset);
    const entry = folders.get(name) || {
      name, ...folderDetails(name), datasets: 0, episodes: 0,
    };
    const parts = entry.path.split("/").filter(Boolean);
    let sourceTime = 0;
    parts.forEach((_, index) => {
      const prefix = parts.slice(0, index + 1).join("/");
      const value = Number(dataset.directory_times?.[prefix]) || 0;
      const key = `${entry.source}:${prefix}`;
      directoryTimes.set(key, Math.max(directoryTimes.get(key) || 0, value));
      sourceTime = Math.max(sourceTime, value);
    });
    directoryTimes.set(entry.source, Math.max(directoryTimes.get(entry.source) || 0, sourceTime));
    entry.datasets += 1;
    entry.episodes += filteredEpisodes(dataset).length;
    folders.set(name, entry);
  });
  return [...folders.values()].sort((a, b) => {
    const left = [a.source, ...a.path.split("/").filter(Boolean)];
    const right = [b.source, ...b.path.split("/").filter(Boolean)];
    for (let index = 0; index < Math.min(left.length, right.length); index += 1) {
      if (left[index] === right[index]) continue;
      const key = (parts) => index === 0 ? parts[0] : `${parts[0]}:${parts.slice(1, index + 1).join("/")}`;
      return (directoryTimes.get(key(right)) || 0) - (directoryTimes.get(key(left)) || 0)
        || right[index].localeCompare(left[index], undefined, {numeric: true});
    }
    return left.length - right.length;
  });
}

function renderFolderList() {
  const select = $("#folderSelect");
  const folders = datasetFolders();
  select.innerHTML = "";
  const groups = new Map();
  folders.forEach(folder => {
    const items = groups.get(folder.group) || [];
    items.push(folder);
    groups.set(folder.group, items);
  });
  groups.forEach((items, label) => {
    const optgroup = document.createElement("optgroup");
    optgroup.label = label;
    items.forEach(folder => {
      const option = document.createElement("option");
      option.value = folder.name;
      option.textContent = `${folder.label} (${folder.datasets} items / ${folder.episodes} eps)`;
      option.title = folder.title;
      optgroup.appendChild(option);
    });
    select.appendChild(optgroup);
  });
  select.value = state.folder || "";
  const active = folders.find(folder => folder.name === state.folder);
  $("#folderSummary").textContent = active
    ? `${active.title} · ${active.datasets} items · ${active.episodes} episodes`
    : "No folder selected";
  $("#folderSummary").title = active?.title || "";
}

async function selectFolder(folder) {
  state.folder = folder;
  state.dataset = null;
  state.episode = null;
  renderFolderList();
  renderDatasetList();
  renderEpisodeList();
  const index = state.datasets.findIndex(dataset => folderForDataset(dataset) === folder);
  if (index >= 0) await selectDataset(index);
}

function renderDatasetList() {
  const container = $("#datasetList");
  container.innerHTML = "";
  let visibleCount = 0;
  state.datasets.forEach((dataset, index) => {
    if (folderForDataset(dataset) !== state.folder) return;
    visibleCount += 1;
    const button = document.createElement("button");
    button.className = `dataset-item${state.dataset === dataset ? " active" : ""}`;
    button.innerHTML = `<strong></strong><span></span>`;
    const title = dataset.display_name || dataset.task_name || dataset.task_id;
    const robotNames = {panda: "Franka", ur5e: "UR5e", xarm7: "xArm7"};
    const robot = robotNames[dataset.robot] || dataset.robot;
    const source = dataset.source_type === "benchmark" ? "rollout" : "dataset";
    const count = `${filteredEpisodes(dataset).length} / ${dataset.episode_count} episodes`;
    button.querySelector("strong").textContent = title;
    button.querySelector("strong").title = title;
    button.querySelector("span").textContent = `${robot} ${source} · ${count}`
      + (dataset.source_type === "benchmark" ? ` · ${dataset.tolerance_mode}` : "");
    button.addEventListener("click", () => selectDataset(index));
    container.appendChild(button);
  });
  const heading = document.querySelector(".dataset-heading > span");
  if (heading) heading.textContent = `Folder contents · ${visibleCount}`;
}

async function selectDataset(index, requestedEpisode = null) {
  state.dataset = state.datasets[index];
  clearEpisode();
  state.folder = folderForDataset(state.dataset);
  renderFolderList();
  renderDatasetList();
  renderEpisodeList();
  const episodes = filteredEpisodes(state.dataset);
  if (!episodes.length) return;
  const episode = episodes.find(item => item.path === requestedEpisode) || episodes[0];
  await selectEpisode(state.dataset.episodes.indexOf(episode));
}

function renderEpisodeList() {
  const container = $("#episodeList");
  container.innerHTML = "";
  const episodes = filteredEpisodes(state.dataset);
  $("#episodeCount").textContent = episodes.length;
  episodes.forEach(episode => {
    const index = state.dataset.episodes.indexOf(episode);
    const button = document.createElement("button");
    button.className = `episode-item${state.episode === episode ? " active" : ""}`;
    const status = episode.success ? "success" : "";
    const execution = episodeExecutionType(episode, state.dataset) === "tolerance" ? "容差" : "无容差";
    button.innerHTML = `<span class="episode-number">${String(index).padStart(2,"0")}</span><span class="episode-copy"><strong>seed ${episode.seed}</strong><span>${episode.frames} frames · ${execution}</span></span><i class="episode-result ${status}"></i>`;
    button.addEventListener("click", () => selectEpisode(index));
    container.appendChild(button);
  });
}

async function selectEpisode(index) {
  pauseVideos();
  const request = ++state.episodeRequest;
  state.episode = state.dataset.episodes[index];
  renderEpisodeList();
  try {
    const query = encodeQuery({
      source: state.dataset.source_type || "dataset",
      dataset: state.dataset.relative_path,
      episode: state.episode.path,
    });
    const payload = await fetchJSON(`/api/episode?${query}`);
    if (request !== state.episodeRequest) return;
    state.payload = payload;
    state.frames = state.payload.trajectory || [];
    state.annotations = state.payload.stage_annotations || [];
    state.frameIndex = 0;
    state.renderedFrame = -1;
    resetSpatialView();
    buildDerivedFrames();
    configureVideos();
    renderHeader();
    renderMetadata();
    updateFrame(0, true);
    $("#emptyState").classList.add("hidden");
    $("#content").classList.remove("hidden");
    const url = new URL(location.href);
    url.searchParams.set("dataset", state.dataset.relative_path);
    url.searchParams.set("episode", state.episode.path);
    url.searchParams.set("source", state.dataset.source_type || "dataset");
    history.replaceState(null, "", url);
  } catch (error) { if (request === state.episodeRequest) showError(error.message); }
}

function configureVideos() {
  const base = {
    source: state.dataset.source_type || "dataset",
    dataset: state.dataset.relative_path,
    episode: state.episode.path,
  };
  mainVideo.src = `/media?${encodeQuery({...base, stream: "main_rgb.mp4"})}`;
  wristVideo.src = `/media?${encodeQuery({...base, stream: "wrist_rgb.mp4"})}`;
  mainVideo.playbackRate = state.speed;
  wristVideo.playbackRate = state.speed;
  mainVideo.load(); wristVideo.load();
  const duration = state.frames.length / Number(state.dataset.frequency_hz || 10);
  $("#timeline").max = duration;
  $("#timeline").value = 0;
  $("#durationTime").textContent = formatTime(duration);
  $("#playButton").textContent = "▶";
}

function instructionForEpisode() {
  if (state.payload?.episode?.instruction) return state.payload.episode.instruction;
  const source = state.payload?.dataset?.instructions;
  const list = Array.isArray(source) ? source : source?.instructions;
  if (!Array.isArray(list) || !list.length) return state.dataset.task_name;
  const index = Number(state.payload.episode?.instruction_index || 0) % list.length;
  return list[index];
}

function renderHeader() {
  const source = state.dataset.source_type === "benchmark" ? "ROLLOUT" : "DATASET";
  $("#sceneName").textContent = `${source} / ${state.dataset.scene} / ${state.episode.path}`.toUpperCase();
  $("#taskName").textContent = state.dataset.task_name;
  $("#instructionText").textContent = instructionForEpisode();
  $("#seedValue").textContent = state.episode.seed;
  $("#frameCountValue").textContent = state.frames.length;
  const result = state.payload.episode?.result || {};
  $("#resultValue").textContent = result.success ? "SUCCESS" : (result.termination_reason || "FAILED").toUpperCase();
  $("#resultValue").className = result.success ? "success" : "";
}

function metadataItem(label, value) {
  const div = document.createElement("div");
  div.className = "metadata-item";
  const span = document.createElement("span"); span.textContent = label;
  const strong = document.createElement("strong"); strong.textContent = value;
  div.append(span, strong); return div;
}

function renderMetadata() {
  const container = $("#episodeMetadata"); container.innerHTML = "";
  const episode = state.payload.episode || {};
  const random = episode.domain_randomization || {};
  const result = episode.result || {};
  const duration = state.frames.length / Number(state.dataset.frequency_hz || 10);
  const collision = result.collision ? JSON.stringify(result.collision) : "None";
  const values = [
    ["Instruction", instructionForEpisode()],
    ["Duration / frequency", `${duration.toFixed(1)} s · ${state.dataset.frequency_hz} Hz`],
    ["Termination", result.termination_reason || "—"],
    ["Collision", collision],
    ["Table offset [m]", (random.table_offset_m || [0,0,0]).map(v => format(v,3)).join(", ")],
    ["Camera Δxyz [m]", (random.front_camera_local_translation_m || [0,0,0]).map(v => format(v,3)).join(", ")],
    ["Camera Δangle [deg]", (random.front_camera_local_angles_deg || [0,0,0]).map(v => format(v,2)).join(", ")],
    ["Light brightness", (random.lighting_brightness || [1,1]).map(v => format(v,2)).join(", ")],
    ["Stage annotations", String(state.annotations.length)],
    ["Final XY error", result.final_block_plate_xy_error_m == null ? "—" : `${format(result.final_block_plate_xy_error_m * 1000,2)} mm`],
    ["Subgoals", Object.entries(result.subgoals || {}).filter(([,v]) => v).map(([k]) => k).join(" · ") || "—"],
    ["Object placements", String(random.object_layout?.placements?.length || 0)],
  ];
  values.forEach(([label,value]) => container.appendChild(metadataItem(label,value)));
}

function videoFrameIndex(time) {
  const fps = Number(state.dataset?.frequency_hz || 10);
  return clamp(Math.floor(time * fps + 0.04), 0, Math.max(0, state.frames.length - 1));
}

function seekFrame(index) {
  if (!state.frames.length) return;
  const fps = Number(state.dataset.frequency_hz || 10);
  const time = clamp(index / fps, 0, state.frames.length / fps);
  mainVideo.currentTime = time;
  wristVideo.currentTime = time;
  updateFrame(index, true);
}

function updateFrame(index, force = false) {
  state.frameIndex = clamp(index, 0, Math.max(0, state.frames.length - 1));
  if (!force && state.renderedFrame === state.frameIndex) return;
  state.renderedFrame = state.frameIndex;
  const fps = Number(state.dataset?.frequency_hz || 10);
  const time = state.frameIndex / fps;
  $("#timeline").value = mainVideo.readyState ? mainVideo.currentTime : time;
  $("#currentTime").textContent = formatTime(mainVideo.readyState ? mainVideo.currentTime : time);
  $("#currentFrame").textContent = String(state.frameIndex).padStart(3, "0");
  renderInspector();
  drawSpatial();
  drawCharts();
}

function animationLoop() {
  if (state.frames.length && !mainVideo.paused) {
    if (Math.abs(wristVideo.currentTime - mainVideo.currentTime) > 0.075) wristVideo.currentTime = mainVideo.currentTime;
    updateFrame(videoFrameIndex(mainVideo.currentTime));
    $("#timeline").value = mainVideo.currentTime;
    $("#currentTime").textContent = formatTime(mainVideo.currentTime);
  }
  requestAnimationFrame(animationLoop);
}

async function togglePlayback() {
  if (!state.frames.length || $('.video-panel').getAttribute('aria-busy') === 'true') return;
  const request = state.episodeRequest;
  if (mainVideo.paused) {
    if (mainVideo.ended) seekFrame(0);
    wristVideo.currentTime = mainVideo.currentTime;
    mainVideo.playbackRate = wristVideo.playbackRate = state.speed;
    try {
      await Promise.all([mainVideo.play(), wristVideo.play()]);
      if (request === state.episodeRequest) $("#playButton").textContent = "Ⅱ";
    } catch (error) {
      if (request === state.episodeRequest && error.name !== 'AbortError') showError(`视频无法播放：${error.message}`);
    }
  } else pauseVideos();
}

function pauseVideos() {
  mainVideo.pause(); wristVideo.pause();
  $("#playButton").textContent = "▶";
}

function vectorCells(container, labels, values, unit, transform = value => value) {
  container.innerHTML = "";
  labels.forEach((label, index) => {
    const cell = document.createElement("div"); cell.className = "vector-cell";
    const key = document.createElement("span"); key.textContent = label;
    const value = document.createElement("strong"); value.textContent = `${format(transform(values?.[index], index), 3)}${unit}`;
    cell.append(key, value); container.appendChild(cell);
  });
}

function renderPose(container, pose) {
  const values = [...(pose?.position_m || [NaN,NaN,NaN]), ...(pose?.rotation_vector_base_rad || [NaN,NaN,NaN])];
  vectorCells(container, ["X","Y","Z","Rx","Ry","Rz"], values, "", (value, index) => index < 3 ? value : degrees(value));
  [...container.children].forEach((cell, index) => cell.querySelector("strong").textContent += index < 3 ? " m" : "°");
}

function renderInspector() {
  const frame = state.frames[state.frameIndex]; if (!frame) return;
  const annotation = frame._annotation;
  const phase = String(frame.phase || "pregrasp").toLowerCase();
  const badge = $("#phaseBadge");
  const stageText = frame.stage_annotation_id == null ? "" : ` · STAGE ${frame.stage_annotation_id}`;
  const subtaskText = frame.subtask_index == null ? "" : `SUBTASK ${Number(frame.subtask_index) + 1} · `;
  badge.textContent = `${subtaskText}${PHASE_NAME[phase] || phase}${stageText}`;
  badge.className = `phase-badge ${PHASE_CLASS[phase] || ""}`;
  renderPose($("#actualPose"), frame.ee_state);
  renderPose($("#nominalPose"), frame.nominal_target_pose);
  if (annotation) {
    renderPose($("#stagePose"), annotation.stage_target_pose);
    $("#tolerancePose").textContent = "坐标系由名义姿态实时构造";
  } else {
    $("#stagePose").innerHTML = "";
    $("#tolerancePose").textContent = state.payload.tolerance_label_status === "unlabeled" ? "容差待标注" : "";
  }

  const action = frame.action || [];
  vectorCells($("#actionVector"), ["ΔX","ΔY","ΔZ","ΔRx","ΔRy","ΔRz","Grip"], action, "");
  const cells = [...$("#actionVector").children];
  cells.forEach((cell, index) => {
    const raw = action[index];
    cell.querySelector("strong").textContent = index < 3 ? `${format(raw * 1000,2)} mm` : index < 6 ? `${format(degrees(raw),2)}°` : format(raw,1);
  });

  const limits = frame._limits;
  const errors = frame._toleranceError;
  const bars = $("#toleranceLimits"); bars.innerHTML = "";
  if (limits && errors) ["x","y","z"].forEach((axis, i) => {
    const row = document.createElement("div"); row.className = `tol-row ${axis}`;
    const negativeDeg = degrees(limits[2*i]), positiveDeg = degrees(limits[2*i+1]), errorDeg = degrees(errors[i]);
    const negativePercent = Math.min(48, negativeDeg / 90 * 48);
    const positivePercent = Math.min(48, positiveDeg / 90 * 48);
    const markerPercent = clamp(50 + errorDeg / 90 * 48, 1, 99);
    row.innerHTML = `<span>R${axis}</span><div class="tol-track"><i style="left:${50-negativePercent}%;width:${negativePercent+positivePercent}%"></i><b style="left:${markerPercent}%"></b></div><span>${errorDeg >= 0 ? "+" : ""}${format(errorDeg,1)}° / [-${format(negativeDeg,0)}, +${format(positiveDeg,0)}]°</span>`;
    bars.appendChild(row);
  });
  $("#gripperValue").textContent = `${format(frame.gripper_state?.aperture_m * 1000, 2)} mm aperture`;
  $("#optimizerValue").textContent = `${frame.optimizer?.iterations ?? "—"} iter · cost ${format(frame.optimizer?.cost,4)}`;
}

function canvasContext(canvas) {
  const ratio = Math.min(2, window.devicePixelRatio || 1);
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(1, Math.round(rect.width * ratio));
  const height = Math.max(1, Math.round(rect.height * ratio));
  if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  return {ctx, width: rect.width, height: rect.height};
}

function viewPoint(point) {
  return [
    dot(state.view.right_world, point),
    dot(state.view.up_world, point),
    dot(state.view.forward_world, point),
  ];
}

function spatialProjection(width, height) {
  const points = [];
  state.frames.forEach(frame => {
    points.push(frame.ee_state.position_m, frame.nominal_target_pose.position_m);
  });
  state.annotations.forEach(item => points.push(item.stage_target_pose.position_m));
  const mins = [0,1,2].map(i => Math.min(...points.map(p => p[i])));
  const maxs = [0,1,2].map(i => Math.max(...points.map(p => p[i])));
  const padding = [0.07,0.07,0.05];
  for (let i=0; i<3; i++) { mins[i] -= padding[i]; maxs[i] += padding[i]; }
  const center = mins.map((value,i) => (value + maxs[i]) / 2);
  const viewed = points.map(point => viewPoint(subtract(point, center)));
  const minU = Math.min(...viewed.map(p => p[0])), maxU = Math.max(...viewed.map(p => p[0]));
  const minV = Math.min(...viewed.map(p => p[1])), maxV = Math.max(...viewed.map(p => p[1]));
  const fit = Math.min((width - 100) / Math.max(.12, maxU-minU), (height - 80) / Math.max(.12, maxV-minV));
  const scale = fit * state.view.zoom;
  return {
    mins, maxs, center,
    project(point) {
      const viewedPoint = viewPoint(subtract(point, center));
      return [width/2 + viewedPoint[0]*scale, height/2 - viewedPoint[1]*scale, viewedPoint[2]];
    }
  };
}

function stroke3D(ctx, projection, points, style, width = 1, dash = []) {
  if (points.length < 2) return;
  ctx.beginPath();
  points.forEach((point,index) => {
    const [x,y] = projection.project(point);
    if (index) ctx.lineTo(x,y); else ctx.moveTo(x,y);
  });
  ctx.strokeStyle = style; ctx.lineWidth = width; ctx.setLineDash(dash); ctx.stroke(); ctx.setLineDash([]);
}

function drawFrame3D(ctx, projection, position, rotation, length, alpha = 1, label = "") {
  const colors = [COLORS[0], COLORS[1], COLORS[2]];
  [0,1,2].forEach(axis => {
    const direction = rotation.map(row => row[axis]);
    const end = add(position, scaleVector(direction, length));
    stroke3D(ctx, projection, [position,end], `${colors[axis]}${Math.round(alpha*255).toString(16).padStart(2,"0")}`, axis === 2 ? 2.3 : 1.8);
    const [x,y] = projection.project(end);
    ctx.fillStyle = colors[axis]; ctx.font = "9px ui-monospace, Consolas"; ctx.fillText(["x","y","z"][axis],x+3,y-2);
  });
  if (label) { const [x,y] = projection.project(position); ctx.fillStyle = `rgba(224,236,240,${alpha})`; ctx.font = "9px ui-monospace, Consolas"; ctx.fillText(label,x+6,y-7); }
}

function rotateLocalAroundAxis(axis, angle) {
  const c = Math.cos(angle), s = Math.sin(angle);
  if (axis === 0) return [0,c,s];
  if (axis === 1) return [s,0,c];
  return [c,s,0];
}

function drawToleranceArc(ctx, projection, position, toleranceRotation, axis, negativeLimit, positiveLimit) {
  if (negativeLimit + positiveLimit <= 1e-8) return;
  const points = [];
  for (let step=0; step<=30; step++) {
    const angle = -negativeLimit + (negativeLimit+positiveLimit)*step/30;
    const local = scaleVector(rotateLocalAroundAxis(axis,angle), .052 + axis*.005);
    points.push(add(position, matVec(toleranceRotation, local)));
  }
  stroke3D(ctx, projection, points, COLORS[axis], 2.2);
  const [x,y] = projection.project(points[points.length-1]);
  ctx.fillStyle = COLORS[axis]; ctx.font = "9px ui-monospace, Consolas";
  ctx.fillText(`[-${format(degrees(negativeLimit),0)},+${format(degrees(positiveLimit),0)}]°`,x+4,y);
}

function drawSpatial() {
  if (!state.frames.length) return;
  const {ctx,width,height} = canvasContext(spatialCanvas);
  ctx.clearRect(0,0,width,height);
  const projection = spatialProjection(width,height);
  const floor = projection.mins[2];
  const step = .05;
  const x0 = Math.floor(projection.mins[0]/step)*step, x1 = Math.ceil(projection.maxs[0]/step)*step;
  const y0 = Math.floor(projection.mins[1]/step)*step, y1 = Math.ceil(projection.maxs[1]/step)*step;
  for (let x=x0; x<=x1+.001; x+=step) stroke3D(ctx,projection,[[x,y0,floor],[x,y1,floor]],"rgba(91,116,128,.14)",1);
  for (let y=y0; y<=y1+.001; y+=step) stroke3D(ctx,projection,[[x0,y,floor],[x1,y,floor]],"rgba(91,116,128,.14)",1);
  drawFrame3D(ctx,projection,[x0,y0,floor],[[1,0,0],[0,1,0],[0,0,1]],.055,.65,"BASE XYZ");

  const actual = state.frames.map(frame => frame.ee_state.position_m);
  const nominal = state.frames.map(frame => frame.nominal_target_pose.position_m);
  stroke3D(ctx,projection,nominal,"rgba(171,187,195,.38)",1,[4,5]);
  stroke3D(ctx,projection,actual,"rgba(72,217,220,.25)",1.3);
  stroke3D(ctx,projection,actual.slice(0,state.frameIndex+1),"#48d9dc",2.4);

  state.annotations.forEach((annotation,index) => {
    const point = annotation.stage_target_pose.position_m;
    const [x,y] = projection.project(point);
    ctx.beginPath(); ctx.arc(x,y,4,0,Math.PI*2); ctx.fillStyle = index === Number(state.frames[state.frameIndex].stage_annotation_id) ? "#ffb15a" : "rgba(255,177,90,.35)"; ctx.fill();
    ctx.fillStyle = "rgba(255,190,111,.7)"; ctx.font = "9px ui-monospace, Consolas"; ctx.fillText(`S${annotation.id}`,x+6,y+3);
  });

  const frame = state.frames[state.frameIndex], annotation = frame._annotation;
  const position = frame.ee_state.position_m;
  const nominalPosition = frame.nominal_target_pose.position_m;
  stroke3D(ctx,projection,[position,nominalPosition],"rgba(255,255,255,.45)",1,[2,3]);
  const actionPosition = add(position, scaleVector(frame.action.slice(0,3),8));
  stroke3D(ctx,projection,[position,actionPosition],"#ffb15a",2);
  drawFrame3D(ctx,projection,position,rotvecMatrix(frame.ee_state.rotation_vector_base_rad),.045,1,"EE");
  drawFrame3D(ctx,projection,nominalPosition,rotvecMatrix(frame.nominal_target_pose.rotation_vector_base_rad),.033,.52,"NOM");
  if (annotation) {
    const targetPosition = annotation.stage_target_pose.position_m;
    drawFrame3D(ctx,projection,targetPosition,rotvecMatrix(annotation.stage_target_pose.rotation_vector_base_rad),.035,.45,"GOAL");
    const toleranceRotation = toleranceFrameMatrix(annotation);
    drawFrame3D(ctx,projection,targetPosition,toleranceRotation,.065,.95,"TOL");
    [0,1,2].forEach(axis => drawToleranceArc(ctx,projection,targetPosition,toleranceRotation,axis,frame._limits[2*axis],frame._limits[2*axis+1]));
  }
  const [cx,cy] = projection.project(position);
  ctx.beginPath(); ctx.arc(cx,cy,5,0,Math.PI*2); ctx.fillStyle = "#ffb15a"; ctx.shadowBlur=12; ctx.shadowColor="#ffb15a"; ctx.fill(); ctx.shadowBlur=0;
  const forward = state.view.forward_world;
  $("#axisReadout").textContent = `EE BASE [m]\nx ${format(position[0],4)}\ny ${format(position[1],4)}\nz ${format(position[2],4)}\n\nview ${state.view.mode}\nforward ${format(forward[0],2)}, ${format(forward[1],2)}, ${format(forward[2],2)}`;
}

function drawLineChart(canvas, series, options = {}) {
  const {ctx,width,height} = canvasContext(canvas);
  ctx.clearRect(0,0,width,height);
  const margin = {left: 43,right: 12,top: 13,bottom: 22};
  const plotW = width-margin.left-margin.right, plotH = height-margin.top-margin.bottom;
  const finite = series.flatMap(item => item.values.filter(Number.isFinite));
  let minimum = options.min ?? Math.min(...finite), maximum = options.max ?? Math.max(...finite);
  if (!Number.isFinite(minimum) || !Number.isFinite(maximum)) { minimum=-1; maximum=1; }
  if (Math.abs(maximum-minimum) < 1e-9) { minimum-=1; maximum+=1; }
  const pad = options.padding === false ? 0 : (maximum-minimum)*.08; minimum-=pad; maximum+=pad;
  const xAt = index => margin.left + index / Math.max(1,state.frames.length-1)*plotW;
  const yAt = value => margin.top + (maximum-value)/(maximum-minimum)*plotH;

  let runStart=0;
  for (let i=1;i<=state.frames.length;i++) {
    if (i===state.frames.length || state.frames[i].phase !== state.frames[runStart].phase) {
      const phase = String(state.frames[runStart].phase || "pregrasp").toLowerCase();
      const color = phase === "grasp" ? "rgba(255,177,90,.035)" : phase === "postgrasp" ? "rgba(195,151,255,.035)" : phase === "release" ? "rgba(255,107,114,.03)" : "rgba(72,217,220,.02)";
      ctx.fillStyle=color; ctx.fillRect(xAt(runStart),margin.top,xAt(Math.max(runStart,i-1))-xAt(runStart)+1,plotH); runStart=i;
    }
  }
  ctx.strokeStyle="rgba(91,116,128,.18)"; ctx.lineWidth=1; ctx.fillStyle="#70818b"; ctx.font="8px ui-monospace, Consolas";
  for(let i=0;i<=4;i++) { const y=margin.top+plotH*i/4; ctx.beginPath();ctx.moveTo(margin.left,y);ctx.lineTo(width-margin.right,y);ctx.stroke(); const value=maximum-(maximum-minimum)*i/4;ctx.fillText(format(value,2),3,y+3); }
  if (minimum<0 && maximum>0) { ctx.strokeStyle="rgba(190,204,210,.23)";ctx.beginPath();ctx.moveTo(margin.left,yAt(0));ctx.lineTo(width-margin.right,yAt(0));ctx.stroke(); }
  series.forEach(item => {
    ctx.beginPath(); let drawing=false;
    item.values.forEach((value,index) => {
      if (!Number.isFinite(value)) { drawing=false; return; }
      const x=xAt(index), y=yAt(value); if(drawing)ctx.lineTo(x,y);else{ctx.moveTo(x,y);drawing=true;}
    });
    ctx.strokeStyle=item.color;ctx.globalAlpha=item.alpha ?? 1;ctx.lineWidth=item.width ?? 1.4;ctx.setLineDash(item.dash||[]);ctx.stroke();ctx.setLineDash([]);ctx.globalAlpha=1;
  });
  const cursor=xAt(state.frameIndex);ctx.strokeStyle="#ffb15a";ctx.lineWidth=1.4;ctx.beginPath();ctx.moveTo(cursor,margin.top);ctx.lineTo(cursor,margin.top+plotH);ctx.stroke();
  const fps=Number(state.dataset?.frequency_hz||10);ctx.fillStyle="#70818b";ctx.fillText("0 s",margin.left,height-7);const end=`${format(state.frames.length/fps,1)} s`;ctx.fillText(end,width-margin.right-ctx.measureText(end).width,height-7);
  canvas._plot={left:margin.left,width:plotW};
}

function axisSeries(values, dashed=false, alpha=1) { return values.map((data,index)=>({values:data,color:COLORS[index],dash:dashed?[4,4]:[],alpha,width:dashed?1:1.6})); }

function drawCharts() {
  if (!state.chartCache) return; const c=state.chartCache;
  drawLineChart($("#positionChart"), [...axisSeries(c.positionActual),...axisSeries(c.positionNominal,true,.42)]);
  drawLineChart($("#rotationChart"), [...axisSeries(c.rotationActual),...axisSeries(c.rotationStage,true,.42)]);
  const toleranceSeries=[];
  [0,1,2].forEach(i=>{
    toleranceSeries.push({values:c.toleranceError[i],color:COLORS[i],width:1.8});
    toleranceSeries.push({values:c.tolerancePositive[i],color:COLORS[i],width:1,dash:[3,4],alpha:.48});
    toleranceSeries.push({values:c.toleranceNegative[i],color:COLORS[i],width:1,dash:[3,4],alpha:.48});
  });
  drawLineChart($("#toleranceChart"),toleranceSeries);
  drawLineChart($("#jointChart"),c.joints.map((values,index)=>({values,color:COLORS[index],width:1.15,alpha:.82})));
  drawLineChart($("#actionTranslationChart"),axisSeries(c.actionTranslationMm));
  drawLineChart($("#actionRotationChart"),axisSeries(c.actionRotationDeg));
  drawLineChart($("#actionGripperChart"),[
    {values:c.actionGripper[0],color:COLORS[4],width:1.8,alpha:.95},
  ],{min:-1.05,max:1.05,padding:false});
}

$("#playButton").addEventListener("click",togglePlayback);
$("#folderSelect").addEventListener("change", event => selectFolder(event.target.value));
$("#toleranceFilter").addEventListener("change", event => selectToleranceFilter(event.target.value));
mainVideo.addEventListener("ended",pauseVideos);
$("#timeline").addEventListener("input", event => {
  const time=Number(event.target.value);mainVideo.currentTime=time;wristVideo.currentTime=time;updateFrame(videoFrameIndex(time),true);
});
$$('.speed-control button').forEach(button=>button.addEventListener("click",()=>{
  state.speed=Number(button.dataset.speed);mainVideo.playbackRate=wristVideo.playbackRate=state.speed;
  $$('.speed-control button').forEach(item=>item.classList.toggle("active",item===button));
}));
$("#refreshButton").addEventListener("click",loadDatasets);
$("#resetView").addEventListener("click",()=>{resetSpatialView();drawSpatial();});
document.addEventListener("keydown",event=>{
  if(event.target.matches("input,textarea,select"))return;
  if(event.code==="Space"){event.preventDefault();togglePlayback();}
  if(event.code==="ArrowRight")seekFrame(state.frameIndex+1);
  if(event.code==="ArrowLeft")seekFrame(state.frameIndex-1);
});

let dragStart=null;
spatialCanvas.addEventListener("pointerdown",event=>{dragStart={x:event.clientX,y:event.clientY,right:[...state.view.right_world],up:[...state.view.up_world],forward:[...state.view.forward_world]};spatialCanvas.setPointerCapture(event.pointerId);});
spatialCanvas.addEventListener("pointermove",event=>{
  if(!dragStart)return;
  const yaw=-(event.clientX-dragStart.x)*.008, pitch=(event.clientY-dragStart.y)*.006;
  const worldZ=[0,0,1];
  const rightYaw=normalized(rotateAroundAxis(dragStart.right,worldZ,yaw));
  const upYaw=rotateAroundAxis(dragStart.up,worldZ,yaw);
  const forwardYaw=rotateAroundAxis(dragStart.forward,worldZ,yaw);
  state.view.right_world=rightYaw;
  state.view.up_world=normalized(rotateAroundAxis(upYaw,rightYaw,pitch));
  state.view.forward_world=normalized(rotateAroundAxis(forwardYaw,rightYaw,pitch));
  state.view.mode="ORBIT";
  drawSpatial();
});
spatialCanvas.addEventListener("pointerup",()=>{dragStart=null;});
spatialCanvas.addEventListener("wheel",event=>{event.preventDefault();state.view.zoom=clamp(state.view.zoom*Math.exp(-event.deltaY*.001),.55,2.4);drawSpatial();},{passive:false});

[$("#positionChart"),$("#rotationChart"),$("#toleranceChart"),$("#jointChart"),$("#actionTranslationChart"),$("#actionRotationChart"),$("#actionGripperChart")].forEach(canvas=>canvas.addEventListener("pointerdown",event=>{
  if(!canvas._plot)return;const rect=canvas.getBoundingClientRect();const x=event.clientX-rect.left;const ratio=clamp((x-canvas._plot.left)/canvas._plot.width,0,1);seekFrame(Math.round(ratio*(state.frames.length-1)));
}));

window.addEventListener("resize",()=>{drawSpatial();drawCharts();});
requestAnimationFrame(animationLoop);
