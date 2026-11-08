const $ = (id) => document.getElementById(id);
let csrfToken = '';
const uploadState = { base: null, target: null, seed: null };
const uploadInProgress = { base: false, target: false, seed: false };
let lastLogId = 0;
let logGeneration = null;
let totalLogEvents = 0;
let logEntries = [];
let pollTimer = null;
let toastTimer = null;
let currentStatus = 'idle';

const defaults = {
  max_depth: 2, target_hours: 0.5, max_chains: 0, rng_seed: '', bloom_mb: 0, device: '', output_name: 'rulest_output.txt',
  bloom_no_shard: false, allow_reject_rules: false, no_builtin_seeds: false,
  token_strip: false, token_strip_min_stem: 4, token_strip_max_prefix: 4,
  token_strip_max_suffix: 4, token_strip_min_leet_amb: 3, token_strip_workers: 0, token_strip_chunk_size: 0, token_strip_bloom_mb: 0,
  genetic: false, genetic_generations: 50, genetic_pop: 200, genetic_elite: 0.15,
  select_mode: 'frequency', select_budget: 0, select_budgets: '', select_cost_alpha: 0, exact_recovery: false,
  debug: false, profile: false,
};

function toast(message, isError = false) {
  const el = $('toast');
  el.textContent = message;
  el.classList.toggle('error', isError);
  el.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove('show'), 3200);
}
function humanSize(bytes) {
  if (!Number.isFinite(bytes)) return '—';
  if (bytes < 1024) return `${bytes} B`;
  const units = ['KB', 'MB', 'GB', 'TB'];
  let val = bytes / 1024, index = 0;
  while (val >= 1024 && index < units.length - 1) { val /= 1024; index++; }
  return `${val.toFixed(val >= 100 ? 0 : 1)} ${units[index]}`;
}
function fmtTime(seconds) {
  if (seconds == null || !Number.isFinite(seconds)) return '00:00';
  seconds = Math.floor(seconds);
  const h = Math.floor(seconds / 3600), m = Math.floor((seconds % 3600) / 60), s = seconds % 60;
  return h ? `${String(h).padStart(2,'0')}:${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}` : `${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}`;
}
function toggleStage(name, enabled) {
  const options = $(`${name}-options`);
  const note = $(`${name}-note`);
  const state = $(`${name}-state`);
  const panel = $(`${name}-stage-panel`);
  if (options) options.classList.toggle('disabled', !enabled);
  if (note) note.classList.toggle('hidden', enabled);
  if (state) {
    state.textContent = enabled ? 'ENABLED' : 'DISABLED';
    state.dataset.enabled = String(enabled);
  }
  if (panel) panel.dataset.enabled = String(enabled);
}
function initDepthOverrides() {
  const grid = $('depth-grid');
  for (let depth = 2; depth <= 10; depth++) {
    const wrap = document.createElement('div');
    wrap.className = 'depth-field';
    const label = document.createElement('label');
    label.textContent = `DEPTH ${depth}`;
    label.htmlFor = `depth${depth}_chains`;
    const input = document.createElement('input');
    input.type = 'number'; input.min = '0'; input.placeholder = 'Auto'; input.id = `depth${depth}_chains`;
    input.setAttribute('aria-label', `Depth ${depth} chains override`);
    wrap.append(label, input); grid.append(wrap);
  }
}
function setDefaults() {
  Object.entries(defaults).forEach(([key, value]) => {
    const el = $(key);
    if (!el) return;
    if (el.type === 'checkbox') el.checked = value;
    else el.value = value;
  });
  for (let d = 2; d <= 10; d++) $(`depth${d}_chains`).value = '';
  toggleStage('token', false);
  toggleStage('genetic', false);
}
function collectConfig() {
  const config = {};
  Object.keys(defaults).forEach((key) => {
    const el = $(key);
    if (!el) return;
    config[key] = el.type === 'checkbox' ? el.checked : el.value;
  });
  for (let d = 2; d <= 10; d++) config[`depth${d}_chains`] = $(`depth${d}_chains`).value;
  return config;
}
function setUploadCard(field, file) {
  const stateEl = $(`${field}-state`);
  stateEl.classList.remove('error');
  stateEl.classList.add('uploaded');
  stateEl.innerHTML = '';
  const dot = document.createElement('span'); dot.className = 'file-state-dot';
  const text = document.createElement('span'); text.textContent = `${file.name} · ${humanSize(file.size)}`;
  stateEl.append(dot, text);
}
function setUploadError(field, message) {
  const el = field === 'seed' ? $('seed-state') : $(`${field}-state`);
  el.classList.add('error'); el.textContent = message;
}
function clearUploadField(field) {
  uploadState[field] = null;
  const stateEl = field === 'seed' ? $('seed-state') : $(`${field}-state`);
  stateEl.classList.remove('uploaded', 'error');
  if (field === 'seed') {
    stateEl.textContent = 'No custom seed rules uploaded';
  } else {
    stateEl.innerHTML = '';
    const dot = document.createElement('span'); dot.className = 'file-state-dot';
    const text = document.createElement('span'); text.textContent = 'No file selected';
    stateEl.append(dot, text);
  }
  const progress = $(`${field}-progress`);
  if (progress) progress.style.width = '0%';
  const input = $(`${field}-file`);
  if (input) input.value = '';
}
function uploadFile(field, file) {
  if (!file) return;
  uploadInProgress[field] = true;
  const progress = $(`${field}-progress`);
  const stateEl = field === 'seed' ? $('seed-state') : $(`${field}-state`);
  if (field !== 'seed') {
    stateEl.classList.remove('uploaded', 'error');
    stateEl.textContent = `Uploading ${file.name}…`;
  } else stateEl.textContent = `Uploading ${file.name}…`;
  if (progress) progress.style.width = '0%';
  const xhr = new XMLHttpRequest();
  xhr.open('POST', `/api/upload/${field}?filename=${encodeURIComponent(file.name)}`);
  xhr.setRequestHeader('Content-Type', 'application/octet-stream');
  xhr.setRequestHeader('X-CSRF-Token', csrfToken);
  xhr.upload.onprogress = (event) => {
    if (event.lengthComputable && progress) progress.style.width = `${Math.round(event.loaded / event.total * 100)}%`;
  };
  xhr.onload = () => {
    uploadInProgress[field] = false;
    let result;
    try { result = JSON.parse(xhr.responseText); } catch { result = { error: 'Invalid server response.' }; }
    if (xhr.status === 401) { window.location.replace('/'); return; }
    if (xhr.status >= 200 && xhr.status < 300 && result.ok) {
      uploadState[field] = { name: result.name, size: result.size };
      if (field === 'seed') {
        stateEl.classList.remove('error'); stateEl.textContent = `✓ ${result.name} · ${humanSize(result.size)}`;
      } else setUploadCard(field, file);
      if (progress) progress.style.width = '100%';
      toast(`${field === 'base' ? 'BASE' : field === 'target' ? 'TARGET' : 'Seed rules'} uploaded successfully.`);
    } else {
      setUploadError(field, result.error || 'Upload failed.');
      if (progress) progress.style.width = '0%';
      toast(result.error || 'Upload failed.', true);
    }
  };
  xhr.onerror = () => { uploadInProgress[field] = false; setUploadError(field, 'Network error during upload.'); toast('Upload failed due to a network error.', true); };
  xhr.send(file);
}
function configureFileInputs() {
  ['base','target','seed'].forEach((field) => {
    const input = $(`${field}-file`);
    input.addEventListener('change', (e) => uploadFile(field, e.target.files[0]));
    if (field !== 'seed') {
      const drop = document.querySelector(`[data-upload="${field}"] .drop-zone`);
      ['dragenter','dragover'].forEach(type => drop.addEventListener(type, (e) => { e.preventDefault(); drop.classList.add('dragover'); }));
      ['dragleave','drop'].forEach(type => drop.addEventListener(type, (e) => { e.preventDefault(); drop.classList.remove('dragover'); }));
      drop.addEventListener('drop', (e) => { const file = e.dataTransfer.files[0]; if (file) uploadFile(field, file); });
    }
  });
}
function addLog(entry) {
  logEntries.push(entry.text);
  totalLogEvents++;
  const body = $('log-output');
  const nearBottom = body.scrollHeight - body.scrollTop - body.clientHeight < 70;
  const placeholder = body.querySelector('.terminal-placeholder');
  if (placeholder) placeholder.remove();
  const row = document.createElement('div');
  row.className = 'log-line';
  const lower = entry.text.toLowerCase();
  if (lower.includes('[error]') || lower.includes('error:') || lower.includes('traceback')) row.classList.add('error');
  else if (lower.includes('[warn]') || lower.includes('warning')) row.classList.add('warn');
  else if (lower.includes('done') || lower.includes('finished') || lower.includes('[out]')) row.classList.add('success');
  else if (/\[(?:s[0-3]|seed|celf)\]|stage [0-3s]/i.test(entry.text)) row.classList.add('info');
  const time = new Date(entry.time * 1000).toLocaleTimeString([], { hour12: false });
  const stamp = document.createElement('span'); stamp.className = 'log-time'; stamp.textContent = time;
  const text = document.createElement('span'); text.className = 'log-message'; text.textContent = entry.text;
  row.append(stamp, text); body.append(row);
  if (nearBottom) body.scrollTop = body.scrollHeight;
  $('log-line-count').textContent = `${totalLogEvents.toLocaleString()} EVENTS`;
}

function renderOutputs(outputs = []) {
  $('metric-outputs').textContent = outputs.length;
  $('output-count').textContent = outputs.length;
  const root = $('output-list'); root.innerHTML = '';
  if (!outputs.length) {
    const empty = document.createElement('div'); empty.className = 'empty-output';
    empty.innerHTML = '<span>▤</span><b>No outputs yet</b><small>Generated rulesets will appear here.</small>';
    root.append(empty); return;
  }
  outputs.forEach((file) => {
    const row = document.createElement('div'); row.className = 'output-item';
    const icon = document.createElement('span'); icon.className = 'output-file-icon'; icon.textContent = file.name.toLowerCase().endsWith('.log') ? 'LOG' : 'TXT';
    const meta = document.createElement('div'); meta.className = 'output-meta';
    const name = document.createElement('b'); name.textContent = file.name;
    const size = document.createElement('small'); size.textContent = humanSize(file.size);
    const link = document.createElement('a'); link.className = 'download-output'; link.href = `/api/download/${encodeURIComponent(file.name)}`; link.textContent = '↓'; link.title = 'Download output';
    meta.append(name, size); row.append(icon, meta, link); root.append(row);
  });
}
function renderStatus(data) {
  currentStatus = data.status;
  const running = data.status === 'running' || data.status === 'stopping';
  const paused = !!data.paused;
  let statusText = data.status.charAt(0).toUpperCase() + data.status.slice(1);
  if (paused) statusText = 'Paused';
  if (data.status === 'idle') statusText = 'Idle';
  $('metric-status').textContent = statusText;
  $('status-note').textContent = paused ? 'Paused at safe checkpoint' : data.pause_requested ? 'Pause requested · checkpoint pending' : running ? 'Engine process is active' : data.status === 'completed' ? 'Run finished successfully' : data.status === 'stopped' ? 'Run stopped · partial results preserved' : data.status === 'failed' ? 'Check engine output below' : 'Ready to configure';
  $('metric-stage').textContent = data.stage || '—';
  $('current-stage').textContent = data.stage || 'Waiting for a run';
  $('current-stage-detail').textContent = data.stage_detail || (running ? 'Waiting for the next engine activity update…' : 'Start a run to see the active processing phase.');
  $('current-stage-detail').title = data.stage_detail || '';
  $('metric-elapsed').textContent = fmtTime(data.elapsed);
  $('monitor-runtime').textContent = fmtTime(data.elapsed);
  $('monitor-pid').textContent = data.pid || '—';
  $('live-pill').classList.toggle('running', running && !paused);
  $('live-pill').classList.toggle('paused', paused);
  $('live-pill').innerHTML = `<span></span> ${paused ? 'PAUSED' : data.pause_requested ? 'PAUSE PENDING' : running ? (data.status === 'stopping' ? 'STOPPING' : 'RUNNING') : data.status === 'completed' ? 'COMPLETE' : data.status === 'stopped' ? 'STOPPED' : data.status === 'failed' ? 'FAILED' : 'STANDBY'}`;
  const pct = data.progress;
  $('progress-label').textContent = pct == null ? (running ? 'LIVE' : '—') : `${pct}%`;
  $('phase-progress').style.width = pct == null ? '0%' : `${pct}%`;
  $('phase-progress').classList.toggle('indeterminate', running && pct == null);
  $('start-run').disabled = running;
  $('pause-run').disabled = !running || paused || !!data.pause_requested || data.status === 'stopping';
  $('resume-run').disabled = !running || (!paused && !data.pause_requested) || data.status === 'stopping';
  $('stop-run').disabled = !running || data.status === 'stopping';
  renderOutputs(data.outputs || []);
}
function resetLogView(title = 'Log history cleared', subtitle = 'New engine events will appear here.') {
  logEntries = [];
  totalLogEvents = 0;
  $('log-output').innerHTML = `<div class="terminal-placeholder"><span>⌁</span><b>${title}</b><small>${subtitle}</small></div>`;
  $('log-line-count').textContent = '0 EVENTS';
}
async function pollStatus() {
  try {
    const generationParam = logGeneration === null ? -1 : logGeneration;
    const response = await fetch(`/api/status?since=${lastLogId}&generation=${generationParam}`, { cache: 'no-store' });
    if (response.status === 401) { window.location.replace('/'); return; }
    const data = await response.json();
    if (typeof data.log_generation === 'number') {
      if (logGeneration !== null && data.log_generation < logGeneration) return; // Ignore stale in-flight polls.
      if (logGeneration === null) {
        logGeneration = data.log_generation;
      } else if (data.log_generation !== logGeneration) {
        logGeneration = data.log_generation;
        lastLogId = 0;
        resetLogView('Log history reset', 'Showing events from the current log session.');
      }
    }
    (data.logs || []).forEach((entry) => { addLog(entry); lastLogId = Math.max(lastLogId, entry.id); });
    if (data.log_seq > lastLogId) lastLogId = data.log_seq;
    if (data.uploads) {
      ['base','target','seed'].forEach((field) => {
        const upload = data.uploads[field];
        if (!upload && !uploadInProgress[field] && uploadState[field]) {
          clearUploadField(field);
        } else if (upload && !uploadState[field]) {
          uploadState[field] = upload;
          if (field === 'seed') $('seed-state').textContent = `✓ ${upload.name} · ${humanSize(upload.size)}`;
          else {
            const stateEl = $(`${field}-state`); stateEl.classList.add('uploaded');
            stateEl.innerHTML = ''; const dot = document.createElement('span'); dot.className = 'file-state-dot';
            const text = document.createElement('span'); text.textContent = `${upload.name} · ${humanSize(upload.size)}`; stateEl.append(dot, text);
          }
        }
      });
    }
    renderStatus(data);
    if (data.error && data.status === 'failed') $('status-note').textContent = data.error;
  } catch (error) {
    $('status-note').textContent = 'Web interface disconnected';
  }
}
async function apiPost(path, payload) {
  const response = await fetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json', 'X-CSRF-Token': csrfToken }, body: JSON.stringify(payload) });
  const data = await response.json();
  if (response.status === 401 && data.auth_required) { window.location.replace('/'); throw new Error('Your session expired. Please sign in again.'); }
  if (!response.ok) throw new Error(data.error || 'Request failed.');
  if (data.csrf_token) csrfToken = data.csrf_token;
  return data;
}
async function startRun() {
  if (!uploadState.base || !uploadState.target) { toast('Upload both BASE and TARGET wordlists first.', true); return; }
  const config = collectConfig();
  if (!config.output_name.trim()) { toast('Enter an output filename.', true); return; }
  const rngSeedText = String(config.rng_seed ?? '').trim();
  if (rngSeedText !== '' && (!/^[0-9]{1,10}$/.test(rngSeedText) || !Number.isSafeInteger(Number(rngSeedText)) || Number(rngSeedText) > 4294967295)) {
    toast('RNG seed must be an integer from 0 to 4294967295, or left blank for a random run.', true);
    return;
  }
  const numericRanges = [
    ['target_hours', 0.000001, Infinity, 'Target runtime must be greater than zero.'],
    ['max_chains', 0, Infinity, 'Maximum chains cannot be negative.'],
    ['bloom_mb', 0, Infinity, 'Bloom filter size cannot be negative.'],
    ['genetic_generations', 1, Infinity, 'Generations must be at least 1.'],
    ['genetic_pop', 1, Infinity, 'Population size must be at least 1.'],
    ['select_budget', 0, Infinity, 'Selection budget cannot be negative.'],
    ['select_cost_alpha', 0, Infinity, 'Cost alpha cannot be negative.'],
    ['token_strip_min_stem', 1, Infinity, 'Minimum stem length must be at least 1.'],
    ['token_strip_max_prefix', 0, Infinity, 'Maximum prefix length cannot be negative.'],
    ['token_strip_max_suffix', 0, Infinity, 'Maximum suffix length cannot be negative.'],
    ['token_strip_min_leet_amb', 0, Infinity, 'Minimum leet ambiguity cannot be negative.'],
    ['token_strip_workers', 0, Infinity, 'Worker processes cannot be negative.'],
    ['token_strip_chunk_size', 0, Infinity, 'Chunk size cannot be negative.'],
  ];
  const tokenBloomMb = Number(config.token_strip_bloom_mb);
  if (!Number.isInteger(tokenBloomMb) || tokenBloomMb < -1) { toast('Base Bloom size must be -1 (auto), 0 (exact), or a positive number of MB.', true); return; }
  if (Number(config.max_depth) < 1 || Number(config.max_depth) > 31) { toast('Maximum depth must be between 1 and 31.', true); return; }
  if (config.genetic && Number(config.max_depth) < 2) { toast('Genetic Algorithm requires Maximum depth ≥ 2. Increase Maximum depth or switch Genetic off.', true); return; }
  if (Number(config.genetic_elite) <= 0 || Number(config.genetic_elite) >= 1 || !Number.isFinite(Number(config.genetic_elite))) { toast('Elite ratio must be between 0 and 1.', true); return; }
  for (const [key, min, max, message] of numericRanges) {
    const value = Number(config[key]);
    if (!Number.isFinite(value) || value < min || value > max) { toast(message, true); return; }
  }
  for (let d = 2; d <= 10; d++) {
    const value = config[`depth${d}_chains`];
    if (value !== '' && (!Number.isInteger(Number(value)) || Number(value) < 0)) { toast(`Depth ${d} chain override must be a non-negative integer.`, true); return; }
  }
  if (config.select_budgets.trim()) {
    const budgets = config.select_budgets.split(',').map(value => value.trim());
    if (budgets.some(value => !/^\d+$/.test(value) || Number(value) <= 0)) { toast('Budget sweep must contain positive integers separated by commas.', true); return; }
  }
  $('start-run').disabled = true;
  resetLogView('Starting a new run', 'Engine output will appear here as soon as processing begins.');
  lastLogId = 0;
  try {
    await apiPost('/api/run', config);
    toast('Rulest engine launched.');
    await pollStatus();
  } catch (error) {
    toast(error.message, true);
    $('start-run').disabled = false;
  }
}
async function controlRun(action) {
  try { await apiPost('/api/control', { action }); toast(action === 'stop' ? 'Partial-save exit requested.' : `${action[0].toUpperCase()}${action.slice(1)} requested.`); await pollStatus(); }
  catch (error) { toast(error.message, true); }
}
function renderGpuDiagnostics(data) {
  const diag = data.diagnostics || {};
  const gpuDevices = data.gpu_devices || (data.devices || []).filter((item) => item.type === 'GPU');
  const state = gpuDevices.length ? 'ready' : (!diag.pyopencl_available ? 'error' : 'warning');
  const summary = $('gpu-diagnostic-summary');
  const action = $('gpu-diagnostic-action');
  const badge = $('gpu-count-badge');
  const panel = $('gpu-diagnostic');
  const icon = $('gpu-diagnostic-icon');
  panel.dataset.state = state;
  summary.textContent = data.message || diag.summary || 'GPU detection completed.';
  action.textContent = diag.recommended_action || 'Check the detailed report below.';
  icon.textContent = gpuDevices.length ? '✓' : (state === 'error' ? '!' : '⚠');
  badge.textContent = gpuDevices.length ? `${gpuDevices.length} GPU${gpuDevices.length === 1 ? '' : 'S'} READY` : 'GPU NOT READY';

  const lines = [
    `System: ${diag.system || 'Unknown'} ${diag.system_release || ''} (${diag.machine || 'unknown architecture'})`,
    `Python version: ${diag.python_version || 'Unknown'}`,
    `Python executable: ${diag.python_executable || 'Unknown'}`,
    `PyOpenCL: ${diag.pyopencl_available ? (diag.pyopencl_version || 'available') : 'NOT AVAILABLE'}`,
    `OpenCL platforms: ${(diag.opencl_platforms || []).length}`,
    `GPU devices: ${diag.gpu_count ?? gpuDevices.length}`,
    `CPU OpenCL devices: ${diag.cpu_count ?? 0}`,
  ];
  if (diag.pyopencl_error) lines.push(`PyOpenCL import error: ${diag.pyopencl_error}`);
  if (diag.opencl_error) lines.push(`OpenCL enumeration error: ${diag.opencl_error}`);
  (diag.opencl_platforms || []).forEach((item) => {
    lines.push(`Platform ${item.index}: ${item.name} | ${item.vendor} | ${item.version}`);
    (item.device_query_errors || []).forEach((error) => lines.push(`  Device query warning: ${error}`));
  });
  (diag.devices || []).forEach((device) => {
    const memory = device.global_memory_bytes ? ` | ${(device.global_memory_bytes / (1024 ** 3)).toFixed(1)} GB` : '';
    lines.push(`Device ${device.index}: ${device.name} (${device.type}) | ${device.vendor}${memory}`);
  });
  Object.entries(diag.probes || {}).forEach(([name, probe]) => {
    lines.push(`${name}: ${probe.available ? (probe.output || 'Available') : 'Not found in PATH'}`);
  });
  lines.push('', `RECOMMENDED ACTION: ${diag.recommended_action || 'Run detection again.'}`);
  $('gpu-diagnostic-report').textContent = lines.join('\n');
}

async function detectDevices(silent = false) {
  const button = $('detect-devices'); button.disabled = true; button.textContent = 'Detecting…';
  try {
    const response = await fetch('/api/devices', { cache: 'no-store' });
    if (response.status === 401) { window.location.replace('/'); return; }
    const data = await response.json();
    if (!response.ok && !data.diagnostics) throw new Error(data.message || `HTTP ${response.status}`);
    const list = $('device-options');
    list.innerHTML = '';
    if ((data.gpu_devices || []).length > 1) {
      const allOption = document.createElement('option'); allOption.value = 'all'; allOption.label = 'All detected GPUs (parallel)'; list.append(allOption);
    }
    (data.gpu_devices || []).forEach((device) => {
      const option = document.createElement('option');
      option.value = device.index;
      option.label = `${device.index}: ${device.name} (${device.vendor || device.type})`;
      list.append(option);
    });
    renderGpuDiagnostics(data);
    if (!silent) {
      const gpuCount = (data.gpu_devices || []).length;
      toast(gpuCount ? `Detected ${gpuCount} OpenCL GPU(s).` : (data.message || 'No OpenCL GPU detected.'), !gpuCount);
    }
  } catch (error) {
    const message = `Device detection failed: ${error.message}`;
    renderGpuDiagnostics({ message, diagnostics: { summary: message, recommended_action: 'Check that the Control Center server is running, then retry GPU detection.' } });
    if (!silent) toast(message, true);
  } finally { button.disabled = false; button.textContent = '⌁ Detect GPUs'; }
}
async function copyGpuReport() {
  try {
    await navigator.clipboard.writeText($('gpu-diagnostic-report').textContent);
    toast('GPU diagnostic report copied.');
  } catch (error) { toast('Clipboard access is unavailable in this browser.', true); }
}

function downloadLog() {
  const blob = new Blob([logEntries.join('\n') + (logEntries.length ? '\n' : '')], { type: 'text/plain;charset=utf-8' });
  const url = URL.createObjectURL(blob); const a = document.createElement('a');
  a.href = url; a.download = `rulest-${new Date().toISOString().replace(/[:.]/g,'-')}.log`; a.click(); URL.revokeObjectURL(url);
}
async function clearLogHistoryFromUi() {
  if (!window.confirm('Clear the displayed log history and permanently truncate the server-side engine.log file? This cannot be undone. Generated ruleset files will be kept.')) return;
  const button = $('clear-log');
  button.disabled = true;
  try {
    const result = await apiPost('/api/logs/clear', {});
    logGeneration = result.log_generation;
    lastLogId = result.log_seq || 0;
    resetLogView('Log history cleared', 'New engine events will continue to appear here.');
    toast('Log history cleared. New engine events will continue to stream.');
  } catch (error) {
    toast(`Could not clear logs: ${error.message}`, true);
  } finally {
    button.disabled = false;
  }
}
function initializeApp() {
  initDepthOverrides(); setDefaults(); configureFileInputs();
  $('token_strip').addEventListener('change', e => toggleStage('token', e.target.checked));
  $('genetic').addEventListener('change', e => toggleStage('genetic', e.target.checked));
  $('reset-defaults').addEventListener('click', () => { setDefaults(); toast('All configuration values reset to defaults.'); });
  $('advanced-trigger').addEventListener('click', () => {
    const btn = $('advanced-trigger'), open = btn.getAttribute('aria-expanded') !== 'true';
    btn.setAttribute('aria-expanded', String(open)); $('advanced-content').classList.toggle('open', open);
  });
  $('start-run').addEventListener('click', startRun);
  $('pause-run').addEventListener('click', () => controlRun('pause'));
  $('resume-run').addEventListener('click', () => controlRun('resume'));
  $('stop-run').addEventListener('click', () => controlRun('stop'));
  $('detect-devices').addEventListener('click', () => detectDevices(false));
  $('copy-gpu-report').addEventListener('click', copyGpuReport);
  $('logout-button').addEventListener('click', async () => {
    try { await apiPost('/api/auth/logout', {}); window.location.replace('/'); }
    catch (error) { toast(error.message, true); }
  });
  $('change-password-button').addEventListener('click', () => {
    $('password-change-error').textContent = ''; $('password-dialog').showModal();
  });
  $('cancel-password-change').addEventListener('click', () => $('password-dialog').close());
  $('password-form').addEventListener('submit', submitPasswordChange);
  $('download-log').addEventListener('click', downloadLog);
  $('clear-log').addEventListener('click', clearLogHistoryFromUi);
  $('wrap-log').addEventListener('click', () => {
    const body = $('log-output');
    const wrapOn = body.classList.toggle('no-wrap') === false;
    $('wrap-log').textContent = `Wrap: ${wrapOn ? 'ON' : 'OFF'}`;
    $('wrap-log').setAttribute('aria-pressed', String(wrapOn));
  });
  $('expand-log').addEventListener('click', () => {
    const expanded = $('terminal-window').classList.toggle('expanded');
    $('expand-log').textContent = expanded ? 'Collapse' : 'Expand';
    $('expand-log').setAttribute('aria-expanded', String(expanded));
  });
  const navLinks = Array.from(document.querySelectorAll('.nav-item'));
  const setActiveNav = (hash) => {
    navLinks.forEach((link) => link.classList.toggle('active', link.getAttribute('href') === (hash || '#workspace')));
  };
  navLinks.forEach((link) => link.addEventListener('click', (event) => {
    const href = link.getAttribute('href');
    const target = href && href.startsWith('#') ? document.getElementById(href.slice(1)) : null;
    if (!target) return;
    event.preventDefault();
    if (window.location.hash !== href) history.pushState(null, '', href);
    setActiveNav(href);
    target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    // Move keyboard focus as well, so this navigation works for keyboard users.
    if (!target.hasAttribute('tabindex')) target.setAttribute('tabindex', '-1');
    target.focus({ preventScroll: true });
    target.classList.add('nav-target-highlight');
    window.setTimeout(() => target.classList.remove('nav-target-highlight'), 1400);
  }));
  window.addEventListener('hashchange', () => setActiveNav(window.location.hash));
  window.addEventListener('popstate', () => setActiveNav(window.location.hash));
  setActiveNav(window.location.hash);
  detectDevices(true);
  pollStatus(); pollTimer = setInterval(pollStatus, 1000);
}
async function submitPasswordChange(event) {
  event.preventDefault();
  const current = $('current-password').value;
  const next = $('new-password').value;
  const confirm = $('confirm-new-password').value;
  const error = $('password-change-error');
  error.textContent = '';
  if (next.length < 12) { error.textContent = 'New password must contain at least 12 characters.'; return; }
  if (next !== confirm) { error.textContent = 'The new passwords do not match.'; return; }
  $('submit-password-change').disabled = true;
  try {
    await apiPost('/api/auth/change-password', { current_password: current, new_password: next });
    $('password-dialog').close();
    $('password-form').reset();
    toast('Password changed. Other sessions have been signed out.');
  } catch (requestError) {
    error.textContent = requestError.message;
  } finally { $('submit-password-change').disabled = false; }
}

async function boot() {
  try {
    const response = await fetch('/api/auth/me', { cache: 'no-store' });
    const auth = await response.json();
    if (!auth.authenticated || !auth.csrf_token) { window.location.replace('/'); return; }
    csrfToken = auth.csrf_token;
    initializeApp();
  } catch (error) {
    window.location.replace('/');
  }
}
document.addEventListener('DOMContentLoaded', boot);
