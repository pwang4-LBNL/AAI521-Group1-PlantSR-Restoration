"use strict";

const $ = (id) => document.getElementById(id);
const TASK_HINTS = {
  denoising: "Remove noise",
  super_resolution: "Upscale 4×",
  colorization: "Color a grayscale photo",
  inpainting: "Fill missing regions",
};
const KIND_LABELS = { finetuned: "Fine-tuned on PlantSR", pretrained: "Original pretrained", baseline: "Simple baselines" };
const METRIC_LABELS = {
  delta_e00: ["Colour error ΔE00 (lower is better)", (v) => v.toFixed(2)],
  masked_psnr: ["Missing-region PSNR", (v) => `${v.toFixed(2)} dB`],
  masked_ssim: ["Missing-region SSIM", (v) => v.toFixed(4)],
  boundary_l1: ["Boundary L1 (lower is better)", (v) => v.toFixed(4)],
  psnr: ["PSNR", (v) => `${v.toFixed(2)} dB`],
  ssim: ["SSIM", (v) => v.toFixed(4)],
};

const state = {
  tasks: [], task: null, source: "upload", file: null, exampleId: null, examples: [],
  loaded: false, tool: "brush", drawing: false, last: null, maskLocked: false,
  coverage: 0, busy: false, jobId: null, pollTimer: null, submitted: null,
  session: (window.crypto && crypto.randomUUID ? crypto.randomUUID() : String(Math.random()).slice(2)).slice(0, 12),
  rating: 0,
  mode: "single",
  pipeline: [],  // [{task, method}] in run order
  dragIndex: null,
};
const DEFAULT_PIPELINE = ["denoising", "colorization", "inpainting"];

function storage(key, value) {  // localStorage can be unavailable; never let it break the page
  try {
    if (value === undefined) return window.localStorage.getItem(key) || "";
    window.localStorage.setItem(key, value);
  } catch (err) { /* ignore */ }
  return value || "";
}

const imageCanvas = $("imageCanvas");
const maskCanvas = $("maskCanvas");
const imageCtx = imageCanvas.getContext("2d");
const maskCtx = maskCanvas.getContext("2d", { willReadFrequently: true });

const dims = (size) => (size && size[0] ? `${size[0]}×${size[1]}` : "—");

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const type = response.headers.get("content-type") || "";
  const body = type.includes("application/json") ? await response.json() : null;
  if (!response.ok) throw new Error((body && body.error) || `Request failed (${response.status})`);
  return body;
}

function setStatus(text, kind = "", spinning = false) {
  const el = $("status");
  el.className = `status ${kind}`;
  el.textContent = "";
  if (spinning) {
    const s = document.createElement("span");
    s.className = "spinner";
    el.appendChild(s);
  }
  el.appendChild(document.createTextNode(text));
}

// ---------------------------------------------------------------- tasks and methods
function currentTask() {
  return state.tasks.find((t) => t.task === state.task);
}

function currentMethod() {
  const task = currentTask();
  return task && task.methods.find((m) => m.method === $("methodSelect").value);
}

function taskInfo(name) {
  return state.tasks.find((t) => t.task === name);
}

// Does the job about to run need a mask? (one inpainting task, or a pipeline with an inpainting step)
function needsMask() {
  if (state.mode === "pipeline") return state.pipeline.some((s) => taskInfo(s.task) && taskInfo(s.task).needs_mask);
  return Boolean(currentTask() && currentTask().needs_mask);
}

function fillMethodOptions(select, task, value) {
  select.textContent = "";
  for (const kind of ["finetuned", "pretrained", "baseline"]) {
    const methods = task.methods.filter((m) => (m.kind || "baseline") === kind);
    if (!methods.length) continue;
    const group = document.createElement("optgroup");
    group.label = KIND_LABELS[kind];
    for (const m of methods) {
      const option = document.createElement("option");
      option.value = m.method;
      option.textContent = m.label + (m.method === task.default_method ? " (default)" : "");
      group.appendChild(option);
    }
    select.appendChild(group);
  }
  select.value = value || task.default_method;
}

function refreshMaskTools() {
  $("maskTools").hidden = !needsMask();
  maskCanvas.hidden = !needsMask() || !state.loaded;
}

function renderTasks() {
  const list = $("taskList");
  list.textContent = "";
  for (const task of state.tasks) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "task";
    button.dataset.task = task.task;
    button.setAttribute("role", "radio");
    button.textContent = task.label;
    const small = document.createElement("small");
    small.textContent = TASK_HINTS[task.task] || "";
    button.appendChild(small);
    button.addEventListener("click", () => selectTask(task.task));
    list.appendChild(button);
  }
}

function selectTask(name) {
  if (state.busy) return;
  state.task = name;
  const task = currentTask();
  document.querySelectorAll(".task").forEach((b) => {
    const active = b.dataset.task === name;
    b.classList.toggle("active", active);
    b.setAttribute("aria-checked", String(active));
  });
  fillMethodOptions($("methodSelect"), task);
  selectMethod();
  const missing = task.unavailable || [];
  $("methodMissing").hidden = !missing.length;
  $("methodMissing").textContent = missing.length
    ? `Not installed on this server: ${missing.map((u) => u.method).join(", ")}.` : "";
  refreshMaskTools();
  if (state.source === "example" && state.exampleId) loadExample(state.exampleId);
  updateRunButton();
}

function selectMethod() {
  const method = currentMethod();
  $("methodDesc").textContent = method ? method.description : "";
  updateSeedRow();
}

function updateSeedRow() {
  let generative;
  if (state.mode === "pipeline") {
    generative = state.pipeline.some((s) => {
      const task = taskInfo(s.task);
      const m = task && task.methods.find((x) => x.method === s.method);
      return m && m.generative;
    });
  } else {
    const method = currentMethod();
    generative = method && method.generative;
  }
  $("seedRow").hidden = !generative;
}

// ---------------------------------------------------------------- pipeline
function setMode(mode) {
  if (state.busy || mode === state.mode) return;
  state.mode = mode;
  document.querySelectorAll("[data-mode]").forEach((b) => {
    const active = b.dataset.mode === mode;
    b.classList.toggle("active", active);
    b.setAttribute("aria-selected", String(active));
  });
  $("singlePane").hidden = mode !== "single";
  $("pipelinePane").hidden = mode !== "pipeline";
  const exampleTab = document.querySelector('[data-source="example"]');
  exampleTab.disabled = mode === "pipeline";
  exampleTab.title = mode === "pipeline" ? "Benchmark examples score one task at a time" : "";
  if (mode === "pipeline" && state.source === "example") setSource("upload");
  if (mode === "pipeline" && !state.pipeline.length) {
    state.pipeline = DEFAULT_PIPELINE.filter((t) => taskInfo(t))
      .map((t) => ({ task: t, method: taskInfo(t).default_method }));
  }
  renderPipeline();
  refreshMaskTools();
  updateSeedRow();
  updateRunButton();
}

function moveStep(from, to) {
  if (state.busy || to < 0 || to >= state.pipeline.length || from === to) return;
  const [step] = state.pipeline.splice(from, 1);
  state.pipeline.splice(to, 0, step);
  renderPipeline();
}

function renderPipeline() {
  const list = $("pipelineList");
  list.textContent = "";
  if (!state.pipeline.length) {
    const empty = document.createElement("li");
    empty.className = "pipeline-empty";
    empty.textContent = "No steps yet. Add a task below.";
    list.appendChild(empty);
  }
  state.pipeline.forEach((step, index) => {
    const task = taskInfo(step.task);
    const li = document.createElement("li");
    li.draggable = !state.busy;
    li.dataset.index = String(index);

    const handle = document.createElement("span");
    handle.className = "handle";
    handle.textContent = "⠿";
    handle.title = "Drag to reorder";
    const badge = document.createElement("span");
    badge.className = "badge";
    badge.textContent = String(index + 1);

    const body = document.createElement("div");
    body.className = "step-body";
    const title = document.createElement("strong");
    title.textContent = task.label;
    const select = document.createElement("select");
    select.setAttribute("aria-label", `Method for step ${index + 1}, ${task.label}`);
    fillMethodOptions(select, task, step.method);
    const showTitle = () => { select.title = select.selectedOptions[0] ? select.selectedOptions[0].textContent : ""; };
    showTitle();
    select.addEventListener("change", () => { step.method = select.value; showTitle(); updateSeedRow(); });
    body.append(title, select);

    const actions = document.createElement("div");
    actions.className = "step-actions";
    const button = (text, label, onClick, disabled) => {
      const b = document.createElement("button");
      b.type = "button";
      b.textContent = text;
      b.setAttribute("aria-label", label);
      b.title = label;
      b.disabled = disabled;
      b.addEventListener("click", onClick);
      actions.appendChild(b);
    };
    button("↑", `Move ${task.label} up`, () => moveStep(index, index - 1), index === 0);
    button("↓", `Move ${task.label} down`, () => moveStep(index, index + 1), index === state.pipeline.length - 1);
    button("✕", `Remove ${task.label}`, () => {
      if (state.busy) return;
      state.pipeline.splice(index, 1);
      renderPipeline();
    }, false);

    li.addEventListener("dragstart", (e) => {
      if (state.busy) { e.preventDefault(); return; }
      state.dragIndex = index;
      li.classList.add("dragging");
      e.dataTransfer.effectAllowed = "move";
      e.dataTransfer.setData("text/plain", String(index));
    });
    li.addEventListener("dragend", () => {
      state.dragIndex = null;
      list.querySelectorAll("li").forEach((x) => x.classList.remove("dragging", "drop-before", "drop-after"));
    });
    li.addEventListener("dragover", (e) => {
      if (state.dragIndex === null) return;
      e.preventDefault();
      const after = e.clientY > li.getBoundingClientRect().top + li.offsetHeight / 2;
      li.classList.toggle("drop-after", after);
      li.classList.toggle("drop-before", !after);
    });
    li.addEventListener("dragleave", () => li.classList.remove("drop-before", "drop-after"));
    li.addEventListener("drop", (e) => {
      e.preventDefault();
      const from = state.dragIndex;
      if (from === null) return;
      const after = e.clientY > li.getBoundingClientRect().top + li.offsetHeight / 2;
      let to = index + (after ? 1 : 0);
      if (from < to) to -= 1;
      moveStep(from, to);
    });

    li.append(handle, badge, body, actions);
    list.appendChild(li);
  });

  // Tasks not in the pipeline yet can be added (each task at most once).
  const add = $("addStepSelect");
  add.textContent = "";
  const remaining = state.tasks.filter((t) => !state.pipeline.some((s) => s.task === t.task));
  for (const t of remaining) {
    const option = document.createElement("option");
    option.value = t.task;
    option.textContent = t.label;
    add.appendChild(option);
  }
  add.disabled = $("addStepBtn").disabled = !remaining.length || state.busy;

  const srIndex = state.pipeline.findIndex((s) => s.task === "super_resolution");
  const warn = $("pipelineWarn");
  warn.hidden = srIndex < 0 || srIndex === state.pipeline.length - 1;
  warn.textContent = warn.hidden ? "" : "Super-resolution makes the image 4× larger, so the steps after it take "
    + "longer (and inpainting resizes your mask to match). It usually works best as the last step.";
  refreshMaskTools();
  updateSeedRow();
  updateRunButton();
}

function addStep() {
  const name = $("addStepSelect").value;
  if (state.busy || !name || state.pipeline.some((s) => s.task === name)) return;
  state.pipeline.push({ task: name, method: taskInfo(name).default_method });
  renderPipeline();
}

function pipelineLabel(steps) {
  return steps.map((s) => (taskInfo(s.task) || { label: s.task }).label).join(" → ");
}

// ---------------------------------------------------------------- image loading
function showImage(img) {
  const w = img.naturalWidth || img.width;
  const h = img.naturalHeight || img.height;
  for (const c of [imageCanvas, maskCanvas]) {
    c.width = w;
    c.height = h;
  }
  imageCtx.drawImage(img, 0, 0, w, h);
  maskCtx.clearRect(0, 0, w, h);
  state.loaded = true;
  $("inputPlaceholder").hidden = true;
  imageCanvas.hidden = false;
  maskCanvas.hidden = !needsMask();
  $("inputDims").textContent = `${w}×${h}`;
  fitCanvas();
  updateCoverage();
}

function fitCanvas() {
  if (!state.loaded) return;
  const stage = $("inputStage");
  const maxW = stage.clientWidth - 2;
  const maxH = Math.max(240, window.innerHeight * 0.7);
  const scale = Math.min(maxW / imageCanvas.width, maxH / imageCanvas.height);
  for (const c of [imageCanvas, maskCanvas]) {
    c.style.width = `${Math.round(imageCanvas.width * scale)}px`;
    c.style.height = `${Math.round(imageCanvas.height * scale)}px`;
    c.classList.toggle("pixelated", scale >= 2);
  }
}

function loadImageFromUrl(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("The image could not be displayed."));
    img.src = url;
  });
}

async function loadFile(file) {
  if (!file || state.busy) return;
  if (!/^image\/(jpeg|png)$/.test(file.type) && !/\.(jpe?g|png)$/i.test(file.name)) {
    setStatus("Please choose a JPG or PNG image.", "error");
    return;
  }
  try {
    const url = URL.createObjectURL(file);
    const img = await loadImageFromUrl(url);
    state.file = file;
    state.exampleId = null;
    setMaskLocked(false);
    showImage(img);
    URL.revokeObjectURL(url);
    $("imageInfo").textContent =
      `${file.name} · ${img.naturalWidth}×${img.naturalHeight} · ${(file.size / 1048576).toFixed(2)} MB`;
    setStatus("");
    resetOutput();
  } catch (err) {
    setStatus(err.message, "error");
  }
}

async function loadExample(id) {
  if (!id || state.busy) return;
  state.exampleId = id;
  const task = state.task || "inpainting";
  try {
    const base = `api/examples/${encodeURIComponent(id)}`;
    const img = await loadImageFromUrl(`${base}/input.png?task=${task}`);
    showImage(img);
    const info = state.examples.find((e) => e.sample_id === id) || {};
    let text = `${id} · ${img.naturalWidth}×${img.naturalHeight} · ${task.replace("_", "-")} input`;
    if (currentTask() && currentTask().needs_mask) {
      const mask = await loadImageFromUrl(`${base}/mask.png?task=inpainting`);
      paintMaskFromImage(mask);
      text += ` · ${info.mask_type || ""} mask`;
    }
    setMaskLocked(true);
    $("imageInfo").textContent = text;
    resetOutput();
  } catch (err) {
    setStatus(`Could not load example: ${err.message}`, "error");
  }
}

function setSource(source) {
  if (state.busy) return;
  state.source = source;
  document.querySelectorAll("[data-source]").forEach((b) => {
    const active = b.dataset.source === source;
    b.classList.toggle("active", active);
    b.setAttribute("aria-selected", String(active));
  });
  $("uploadPane").hidden = source !== "upload";
  $("examplePane").hidden = source !== "example";
  if (source === "example") {
    if (!state.examples.length) {
      setStatus("No benchmark examples are configured on this server.", "error");
      return;
    }
    loadExample($("exampleSelect").value);
  } else if (state.file) {
    loadFile(state.file);
  } else {
    clearInput();
  }
}

function clearInput() {
  state.loaded = false;
  state.exampleId = null;
  imageCanvas.hidden = true;
  maskCanvas.hidden = true;
  $("inputPlaceholder").hidden = false;
  $("inputDims").textContent = "";
  $("imageInfo").textContent = "";
  setMaskLocked(false);
  updateRunButton();
}

// ---------------------------------------------------------------- mask editing
function setMaskLocked(locked) {
  state.maskLocked = locked;
  maskCanvas.classList.toggle("locked", locked);
  $("maskTools").classList.toggle("locked", locked);
  document.querySelectorAll("[data-tool], #clearMask, #maskFile").forEach((el) => { el.disabled = locked; });
  $("maskNote").textContent = locked ? "· fixed benchmark mask" : "";
}

function canvasPoint(event) {
  const rect = maskCanvas.getBoundingClientRect();
  return {
    x: ((event.clientX - rect.left) * maskCanvas.width) / rect.width,
    y: ((event.clientY - rect.top) * maskCanvas.height) / rect.height,
    scale: maskCanvas.width / rect.width,
  };
}

function strokeTo(point) {
  maskCtx.globalCompositeOperation = state.tool === "eraser" ? "destination-out" : "source-over";
  maskCtx.strokeStyle = maskCtx.fillStyle = "rgb(255, 48, 48)";
  maskCtx.lineCap = maskCtx.lineJoin = "round";
  maskCtx.lineWidth = Number($("brushSize").value) * point.scale;
  maskCtx.beginPath();
  const from = state.last || point;
  maskCtx.moveTo(from.x, from.y);
  maskCtx.lineTo(point.x + 0.01, point.y);
  maskCtx.stroke();
  state.last = point;
}

maskCanvas.addEventListener("pointerdown", (event) => {
  if (state.maskLocked || !state.loaded || state.busy) return;
  state.drawing = true;
  state.last = null;
  maskCanvas.setPointerCapture(event.pointerId);
  strokeTo(canvasPoint(event));
});
maskCanvas.addEventListener("pointermove", (event) => {
  if (state.drawing) strokeTo(canvasPoint(event));
});
for (const name of ["pointerup", "pointercancel"]) {
  maskCanvas.addEventListener(name, () => {
    if (!state.drawing) return;
    state.drawing = false;
    state.last = null;
    updateCoverage();
  });
}

function paintMaskFromImage(img) {
  const w = maskCanvas.width;
  const h = maskCanvas.height;
  const tmp = document.createElement("canvas");
  tmp.width = w;
  tmp.height = h;
  const ctx = tmp.getContext("2d");
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(img, 0, 0, w, h);
  const src = ctx.getImageData(0, 0, w, h).data;
  const out = maskCtx.createImageData(w, h);
  // Same rule as the server: a mask whose brightest value is at most 1 is a 0/1 mask
  // (above 0.5 = fill); otherwise it is 0/255 (above 127 = fill). Transparent pixels are kept.
  const value = (i) => (src[i + 3] > 0 ? Math.max(src[i], src[i + 1], src[i + 2]) : 0);
  let peak = 0;
  for (let i = 0; i < src.length; i += 4) peak = Math.max(peak, value(i));
  const threshold = peak <= 1 ? 0.5 : 127;
  for (let i = 0; i < src.length; i += 4) {
    if (value(i) > threshold) {
      out.data[i] = 255; out.data[i + 1] = 48; out.data[i + 2] = 48; out.data[i + 3] = 255;
    }
  }
  maskCtx.globalCompositeOperation = "source-over";
  maskCtx.putImageData(out, 0, 0);
  updateCoverage();
}

function updateCoverage() {
  if (!state.loaded || !maskCanvas.width) {
    state.coverage = 0;
  } else {
    const data = maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height).data;
    let count = 0;
    for (let i = 3; i < data.length; i += 4) if (data[i] > 0) count += 1;
    state.coverage = count / (maskCanvas.width * maskCanvas.height);
  }
  $("coverage").textContent = `${(state.coverage * 100).toFixed(2)}%`;
  updateRunButton();
}

function maskBlob() {
  const w = maskCanvas.width;
  const h = maskCanvas.height;
  const src = maskCtx.getImageData(0, 0, w, h).data;
  const tmp = document.createElement("canvas");
  tmp.width = w;
  tmp.height = h;
  const ctx = tmp.getContext("2d");
  const out = ctx.createImageData(w, h);
  for (let i = 0; i < src.length; i += 4) {
    const v = src[i + 3] > 0 ? 255 : 0;
    out.data[i] = out.data[i + 1] = out.data[i + 2] = v;
    out.data[i + 3] = 255;
  }
  ctx.putImageData(out, 0, 0);
  return new Promise((resolve) => tmp.toBlob(resolve, "image/png"));
}

// ---------------------------------------------------------------- jobs
// Lock everything that defines a job (image, task, method, seed, mask) while it runs,
// so the finished result always matches what was submitted.
function setBusy(busy) {
  state.busy = busy;
  document.body.classList.toggle("busy", busy);
  const controls = document.querySelectorAll(
    ".task, [data-source], [data-mode], #fileInput, #exampleSelect, #methodSelect, #seedInput, [data-tool], "
    + "#clearMask, #maskFile, #pipelineList button, #pipelineList select, #addStepSelect, #addStepBtn");
  controls.forEach((el) => {
    if (busy) {
      el.dataset.wasDisabled = el.disabled ? "1" : "";
      el.disabled = true;
    } else {
      el.disabled = el.dataset.wasDisabled === "1";
      delete el.dataset.wasDisabled;
    }
  });
  $("dropzone").classList.toggle("disabled", busy);
  document.querySelectorAll("#pipelineList li").forEach((li) => { li.draggable = !busy; });
  updateRunButton();
}

function updateRunButton() {
  const pipeline = state.mode === "pipeline";
  const ready = pipeline ? state.pipeline.length > 0 : Boolean(currentTask());
  const maskMissing = needsMask() && state.source === "upload" && state.coverage <= 0;
  $("runBtn").disabled = state.busy || !state.loaded || !ready || maskMissing;
  $("runBtn").textContent = pipeline
    ? `Run pipeline (${state.pipeline.length} step${state.pipeline.length === 1 ? "" : "s"})` : "Restore image";
  $("runBtn").title = maskMissing ? "Paint the area to fill first" : pipeline && !ready ? "Add at least one step" : "";
}

function resetOutput() {
  $("outputImg").hidden = true;
  $("outputImg").removeAttribute("src");
  $("outputPlaceholder").hidden = false;
  $("outputDims").textContent = "";
  $("resultBox").hidden = true;
  $("stepStrip").hidden = true;
  $("stepStrip").textContent = "";
}

async function runJob() {
  if (state.mode === "pipeline") return runPipeline();
  const form = new FormData();
  form.append("task", state.task);
  form.append("method", $("methodSelect").value);
  form.append("seed", $("seedInput").value || "42");
  form.append("source", state.source);
  form.append("session", state.session);
  form.append("tester", $("testerId").value.trim());
  if (state.source === "example") {
    form.append("example_id", state.exampleId);
  } else {
    form.append("image", state.file, state.file.name);
    if (currentTask().needs_mask) form.append("mask", await maskBlob(), "mask.png");
  }
  const method = currentMethod();
  state.submitted = {
    task: currentTask(), method, seed: $("seedInput").value || "42",
    input: state.source === "example" ? state.exampleId : state.file.name,
  };
  setBusy(true);
  resetOutput();
  setStatus("Uploading…", "", true);
  try {
    const job = await api("api/jobs", { method: "POST", body: form });
    state.jobId = job.id;
    poll();
  } catch (err) {
    setBusy(false);
    setStatus(err.message, "error");
  }
}

async function runPipeline() {
  const steps = state.pipeline.map((s) => ({ task: s.task, method: s.method }));
  const form = new FormData();
  form.append("task", "pipeline");
  form.append("steps", JSON.stringify(steps));
  form.append("seed", $("seedInput").value || "42");
  form.append("source", "upload");
  form.append("session", state.session);
  form.append("tester", $("testerId").value.trim());
  form.append("image", state.file, state.file.name);
  if (needsMask()) form.append("mask", await maskBlob(), "mask.png");
  state.submitted = { steps, input: state.file.name };
  setBusy(true);
  resetOutput();
  setStatus("Uploading…", "", true);
  try {
    const job = await api("api/jobs", { method: "POST", body: form });
    state.jobId = job.id;
    poll();
  } catch (err) {
    setBusy(false);
    setStatus(err.message, "error");
  }
}

async function poll() {
  clearTimeout(state.pollTimer);
  try {
    const job = await api(`api/jobs/${state.jobId}`);
    if (job.status === "queued") {
      setStatus(`Queued${job.queue_position > 1 ? ` (position ${job.queue_position})` : ""}…`, "", true);
    } else if (job.status === "running") {
      const seconds = (Date.now() / 1000 - job.started_at).toFixed(1);
      const p = job.progress;
      const label = p ? (taskInfo(p.task) || { label: p.task }).label : "";
      setStatus(p ? `Step ${p.step} of ${p.total}: ${label}… ${seconds}s` : `Processing… ${seconds}s`, "", true);
    }
    if (job.status === "done" || job.status === "error") {
      refreshHealth();  // update the queue count and loaded models right away
      setBusy(false);
      return finish(job);
    }
  } catch (err) {
    setStatus(`Lost contact with the server: ${err.message}`, "error");
    setBusy(false);
    return;
  }
  state.pollTimer = setTimeout(poll, 700);
}

function showOutput(url, size) {
  const img = $("outputImg");
  img.onload = () => {
    const shown = Math.min(img.clientWidth / img.naturalWidth, img.clientHeight / img.naturalHeight);
    img.classList.toggle("pixelated", shown >= 2);
    $("outputDims").textContent = size ? dims(size) : `${img.naturalWidth}×${img.naturalHeight}`;
  };
  img.src = `${url}?t=${Date.now()}`;
  img.hidden = false;
  $("outputPlaceholder").hidden = true;
}

function infoRows(rows) {
  const table = $("infoTable");
  table.textContent = "";
  for (const [label, value, cls] of rows) {
    const tr = table.insertRow();
    if (cls) tr.className = cls;
    const th = document.createElement("th");
    th.textContent = label;
    tr.appendChild(th);
    tr.insertCell().textContent = value;
  }
}

function finishPipeline(job) {
  const r = job.result || {};
  const steps = r.steps || [];
  const urls = job.step_urls || [];
  const strip = $("stepStrip");
  strip.textContent = "";
  urls.forEach((url, i) => {
    const step = steps[i] || {};
    const fig = document.createElement("figure");
    const thumb = document.createElement("button");
    thumb.type = "button";
    thumb.className = `thumb${i === urls.length - 1 ? " active" : ""}`;
    thumb.title = "Show this step's result";
    const img = document.createElement("img");
    img.src = url;
    img.alt = `Result after step ${i + 1}`;
    thumb.appendChild(img);
    thumb.addEventListener("click", () => {
      strip.querySelectorAll(".thumb").forEach((t) => t.classList.toggle("active", t === thumb));
      showOutput(url, step.output_size);
    });
    const cap = document.createElement("figcaption");
    const name = document.createElement("span");
    name.textContent = `${i + 1}. ${(taskInfo(step.task) || { label: step.task }).label}`;
    const link = document.createElement("a");
    link.href = `${url}?download=1`;
    link.textContent = "PNG";
    link.setAttribute("download", "");
    cap.append(name, link);
    fig.append(thumb, cap);
    strip.appendChild(fig);
  });
  strip.hidden = !urls.length;

  const wait = job.started_at - job.created_at;
  if (job.status === "error") {
    setStatus(job.error || "The pipeline failed.", "error");
  } else {
    setStatus(`Done: ${steps.length} steps in ${r.elapsed_seconds.toFixed(2)}s`
      + `${wait > 0.5 ? ` (waited ${wait.toFixed(1)}s in queue)` : ""}`, "done");
  }
  if (urls.length) showOutput(urls[urls.length - 1], steps[urls.length - 1] && steps[urls.length - 1].output_size);

  const rows = [
    ["Task", "Pipeline"],
    ["Steps", pipelineLabel(job.steps || [])],
    ["Input", state.submitted ? state.submitted.input : "—"],
    ["Received", dims(r.input_size)],
    ["Output", job.status === "done" ? dims(r.output_size) : "—"],
    ["Total inference time", `${(r.elapsed_seconds || 0).toFixed(2)} s`],
  ];
  if ((job.steps || []).some((s) => s.needs_mask)) rows.push(["Seed", String(job.seed)]);
  (r.notes || []).forEach((note) => rows.push(["Note", note]));
  steps.forEach((step, i) => {
    const task = taskInfo(step.task) || { label: step.task, methods: [] };
    const method = task.methods.find((m) => m.method === step.method);
    rows.push([`Step ${i + 1} · ${task.label}`, step.success ? "" : "failed", "section"]);
    rows.push(["Method", method ? method.label : step.method]);
    if (step.success) {
      rows.push(["Model version", step.model_version]);
      rows.push(["Size", `${dims(step.input_size)} → processed ${dims(step.processed_size)} → ${dims(step.output_size)}`]);
      rows.push(["Time", `${step.elapsed_seconds.toFixed(2)} s`]);
      if (step.mask_coverage !== undefined) rows.push(["Mask coverage", `${(step.mask_coverage * 100).toFixed(2)}%`]);
      (step.notes || []).forEach((note) => rows.push(["Note", note]));
    } else {
      rows.push(["Error", step.error || "failed"]);
    }
  });
  infoRows(rows);
  const download = $("downloadBtn");
  download.hidden = job.status !== "done";
  if (job.status === "done") {
    download.href = `${job.output_url}?download=1`;
    download.setAttribute("download", `plantsr_pipeline_${job.id}.png`);
  }
  $("resultBox").hidden = false;
  resetFeedback();
  $("feedbackForm").hidden = job.status !== "done";
}

function finish(job) {
  const r = job.result || {};
  if (job.task === "pipeline") return finishPipeline(job);
  $("downloadBtn").hidden = false;
  if (job.status === "error") {
    setStatus(job.error || "The restoration failed.", "error");
    return;
  }
  const wait = job.started_at - job.created_at;
  setStatus(`Done in ${r.elapsed_seconds.toFixed(2)}s${wait > 0.5 ? ` (waited ${wait.toFixed(1)}s in queue)` : ""}`, "done");
  const img = $("outputImg");
  img.onload = () => {
    const shown = Math.min(img.clientWidth / img.naturalWidth, img.clientHeight / img.naturalHeight);
    img.classList.toggle("pixelated", shown >= 2);
  };
  img.src = `${job.output_url}?t=${Date.now()}`;
  img.hidden = false;
  $("outputPlaceholder").hidden = true;
  $("outputDims").textContent = dims(r.output_size);
  // Describe the job that ran (server record), not whatever is selected now.
  const task = state.tasks.find((t) => t.task === job.task) || { label: job.task, methods: [] };
  const method = task.methods.find((m) => m.method === job.method);
  const rows = [
    ["Task", task.label],
    ["Method", method ? method.label : job.method],
    ["Input", state.submitted ? state.submitted.input : "—"],
    ["Model version", r.model_version],
    ["Received", dims(r.input_size)],
    ["Processed at", dims(r.processed_size)],
    ["Output", dims(r.output_size)],
    ["Inference time", `${r.elapsed_seconds.toFixed(2)} s`],
  ];
  if (method && method.generative) rows.push(["Seed", String(job.seed)]);
  if (r.mask_coverage !== undefined) rows.push(["Mask coverage", `${(r.mask_coverage * 100).toFixed(2)}%`]);
  (r.notes || []).forEach((note) => rows.push(["Note", note]));
  const table = $("infoTable");
  table.textContent = "";
  const addRow = (label, value, cls) => {
    const tr = table.insertRow();
    if (cls) tr.className = cls;
    const th = document.createElement("th");
    th.textContent = label;
    tr.appendChild(th);
    const td = tr.insertCell();
    td.textContent = value;
  };
  rows.forEach(([k, v]) => addRow(k, v));
  if (job.metrics) {
    addRow("Benchmark scores", "vs. the known reference", "section");
    for (const [key, value] of Object.entries(job.metrics)) {
      if (key === "note") addRow("Note", value);
      else if (METRIC_LABELS[key]) addRow(METRIC_LABELS[key][0], METRIC_LABELS[key][1](value));
    }
  }
  const download = $("downloadBtn");
  download.href = `${job.output_url}?download=1`;
  download.setAttribute("download", `plantsr_${job.task}_${job.method}_${job.id}.png`);
  $("resultBox").hidden = false;
  resetFeedback();
}

// ---------------------------------------------------------------- user-testing feedback
function resetFeedback() {
  state.rating = 0;
  document.querySelectorAll("[data-rating]").forEach((b) => b.classList.remove("active"));
  $("feedbackComment").value = "";
  $("feedbackSend").disabled = true;
  $("feedbackStatus").textContent = "";
  $("feedbackForm").hidden = false;
}

async function sendFeedback(event) {
  event.preventDefault();
  if (!state.rating || !state.jobId) return;
  $("feedbackSend").disabled = true;
  try {
    await api("api/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ job_id: state.jobId, rating: state.rating, comment: $("feedbackComment").value,
        tester: $("testerId").value.trim(), session: state.session }),
    });
    $("feedbackStatus").textContent = "Thanks, your feedback was saved.";
  } catch (err) {
    $("feedbackStatus").textContent = err.message;
    $("feedbackSend").disabled = false;
  }
}

// ---------------------------------------------------------------- server status
async function refreshHealth() {
  const pill = $("health");
  try {
    const h = await api("api/health");
    pill.classList.remove("offline");
    const where = h.gpu ? `GPU · ${h.gpu}` : "CPU";
    pill.textContent = `${where} · ${h.pending_jobs} in queue`;
    const missing = (h.unavailable || []).map((u) => `${u.task}/${u.method}`);
    pill.title = (h.loaded_models.length ? `Loaded: ${h.loaded_models.join(", ")}` : "No large model loaded")
      + (missing.length ? `\nNot installed: ${missing.join(", ")}` : "");
  } catch (err) {
    pill.classList.add("offline");
    pill.textContent = "Server offline";
  }
}

// ---------------------------------------------------------------- wiring
async function init() {
  $("fileInput").addEventListener("change", (e) => loadFile(e.target.files[0]));
  const drop = $("dropzone");
  drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("over"));
  drop.addEventListener("drop", (e) => {
    e.preventDefault();
    drop.classList.remove("over");
    loadFile(e.dataTransfer.files[0]);
  });
  document.querySelectorAll("[data-source]").forEach((b) => b.addEventListener("click", () => setSource(b.dataset.source)));
  document.querySelectorAll("[data-tool]").forEach((b) => b.addEventListener("click", () => {
    state.tool = b.dataset.tool;
    document.querySelectorAll("[data-tool]").forEach((x) => x.classList.toggle("active", x === b));
  }));
  $("clearMask").addEventListener("click", () => {
    maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
    updateCoverage();
  });
  $("maskFile").addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (!file || !state.loaded) return;
    const url = URL.createObjectURL(file);
    try {
      paintMaskFromImage(await loadImageFromUrl(url));
    } catch (err) {
      setStatus(err.message, "error");
    } finally {
      URL.revokeObjectURL(url);
      e.target.value = "";
    }
  });
  $("showMask").addEventListener("change", (e) => { maskCanvas.style.opacity = e.target.checked ? "" : "0"; });
  $("methodSelect").addEventListener("change", selectMethod);
  $("exampleSelect").addEventListener("change", (e) => loadExample(e.target.value));
  $("runBtn").addEventListener("click", runJob);
  document.querySelectorAll("[data-mode]").forEach((b) => b.addEventListener("click", () => setMode(b.dataset.mode)));
  $("addStepBtn").addEventListener("click", addStep);
  document.querySelectorAll("[data-rating]").forEach((b) => b.addEventListener("click", () => {
    state.rating = Number(b.dataset.rating);
    document.querySelectorAll("[data-rating]").forEach((x) => x.classList.toggle("active", x === b));
    $("feedbackSend").disabled = false;
  }));
  $("feedbackForm").addEventListener("submit", sendFeedback);
  // Observers can open the app as ...?tester=T3 so every job of that tester is labelled in logs/jobs.csv.
  const fromUrl = new URLSearchParams(window.location.search).get("tester");
  if (fromUrl) storage("plantsr-tester", fromUrl.trim().slice(0, 40));
  $("testerId").value = storage("plantsr-tester");
  $("testerId").addEventListener("change", (e) => storage("plantsr-tester", e.target.value.trim()));

  try {
    state.tasks = await api("api/tasks");
    renderTasks();
    const preferred = state.tasks.find((t) => t.task === "inpainting") || state.tasks[0];
    selectTask(preferred.task);
    state.examples = await api("api/examples");
    const select = $("exampleSelect");
    for (const ex of state.examples) {
      const option = document.createElement("option");
      option.value = ex.sample_id;
      option.textContent = `${ex.sample_id} · ${ex.mask_type} · ${(ex.mask_coverage * 100).toFixed(1)}%`;
      select.appendChild(option);
    }
    document.querySelector('[data-source="example"]').disabled = !state.examples.length;
  } catch (err) {
    setStatus(`Could not load the task list: ${err.message}`, "error");
  }
  window.addEventListener("resize", fitCanvas);
  refreshHealth();
  setInterval(refreshHealth, 10000);
}

init();
