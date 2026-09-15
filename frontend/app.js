'use strict';

const $ = (id) => document.getElementById(id);

const state = {
  voices: [],
  voiceKeys: [],       // ["#1", "#2", "ИВАН"] — голоса, встреченные в тексте
  voiceMeta: {},       // key -> {key, label, slot}
  configs: {},         // key -> {voice_id, speed, cfg_strength, nfe_step}
  preview: {},         // voice_id -> {speed, cfg_strength, nfe_step, text}
  modelReady: false,
  jobId: null,
  pollTimer: null,
  statusTimer: null,
  previewJob: null,    // {jobId, voiceId}
  previewTimer: null,
  textJobId: null,     // задача режима «Сплошной текст»
  textPollTimer: null,
};

const NFE_OPTIONS = [8, 16, 32];
const PREVIEW_TEXT = 'Привет! Так звучит этот голос в диалоге.';
const DROP_HINT = 'Перетащите сюда аудио (wav/mp3/m4a) или кликните';

// --- утилиты ------------------------------------------------------------------
const esc = (value) =>
  String(value).replace(/[&<>"']/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

async function api(url, options) {
  let response;
  try {
    response = await fetch(url, options);
  } catch (_) {
    // fetch отклоняется только на сетевом сбое: сервер не запущен или упал.
    // Без этой ветки пользователь видел невнятное «Failed to fetch».
    throw new Error('Бэкенд недоступен: сервер не отвечает. Запустите ./run.sh и обновите страницу.');
  }
  if (!response.ok) {
    let detail = `Ошибка ${response.status}`;
    try {
      const body = await response.json();
      if (body.detail) detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail);
    } catch (_) { /* тело не JSON — оставляем текст статуса */ }
    throw new Error(detail);
  }
  return response.status === 204 ? null : response.json();
}

function showAlert(el, message, kind = 'error') {
  if (!message) { el.classList.remove('show'); el.textContent = ''; return; }
  el.className = `alert ${kind} show`;
  el.textContent = message;
}

// Опции NFE одинаковы в карточке голоса и в карточке спикера диалога.
const nfeOptions = (selected) => NFE_OPTIONS
  .map((n) => `<option value="${n}"${n === selected ? ' selected' : ''}>${n}${n === 32 ? ' (максимум качества)' : ''}</option>`)
  .join('');

function defaultConfig() {
  // Голос не подставляем: одинаковый голос на все слоты — это ровно та ошибка,
  // из-за которой диалог читается одним голосом. Выбор должен быть осознанным.
  return { voice_id: '', speed: 1.0, cfg_strength: 2.0, nfe_step: 32 };
}

function voiceTags(voice) {
  const label = voice.gender === 'male' ? 'муж.' : voice.gender === 'female' ? 'жен.' : '—';
  const tags = [`<span class="tag">${label}</span>`];
  if (voice.f0_hz) {
    tags.push(`<span class="tag" title="Измеренная высота тона референса">~${Math.round(voice.f0_hz)} Гц</span>`);
  }
  if (voice.is_demo) {
    const source = voice.demo_source ? `: ${esc(voice.demo_source)}` : '';
    tags.push(`<span class="tag warn" title="Копия демо-файла F5-TTS${source} — не пользовательская запись">demo</span>`);
  }
  if (voice.gender_warning) {
    tags.push(`<span class="tag warn" title="${esc(voice.gender_warning)}">⚠ проверьте пол</span>`);
  }
  if (voice.ref_text_warning) {
    tags.push(`<span class="tag warn" title="${esc(voice.ref_text_warning)}">⚠ расшифровка</span>`);
  }
  return tags.join('');
}

// --- вкладки ------------------------------------------------------------------
function switchTab(name) {
  document.querySelectorAll('.tab').forEach((tab) => {
    tab.classList.toggle('active', tab.dataset.tab === name);
  });
  $('tab-voices').hidden = name !== 'voices';
  $('tab-dialogue').hidden = name !== 'dialogue';
  $('tab-text').hidden = name !== 'text';
}

// --- голоса -------------------------------------------------------------------
async function loadVoices() {
  const data = await api('/api/voices');
  state.voices = data.voices;
  renderVoiceCards();
  renderVoiceConfigs();
  renderTextVoiceOptions();
  updateGenerateButton();
}

function previewState(voiceId) {
  if (!state.preview[voiceId]) {
    state.preview[voiceId] = { speed: 1.0, cfg_strength: 2.0, nfe_step: 32, text: PREVIEW_TEXT };
  }
  return state.preview[voiceId];
}

function previewCard(voiceId) {
  return document.querySelector(`#voice-cards [data-voice-id="${voiceId}"]`);
}

function renderVoiceCards() {
  const box = $('voice-cards');
  if (!state.voices.length) {
    box.innerHTML = '<div class="muted" style="padding:6px 0">Голосов пока нет — загрузите референс-аудио справа.</div>';
    return;
  }
  box.innerHTML = state.voices.map((voice) => {
    const cfg = previewState(voice.id);
    return `
      <div class="card" data-voice-id="${esc(voice.id)}">
        <div class="card-title">
          <span class="swatch"></span>${esc(voice.name)}
          ${voiceTags(voice)}
          ${voice.ref_text ? '' : '<span class="tag warn" title="Без референс-текста синтез невозможен">нет текста</span>'}
          <button class="tiny ghost danger" data-role="delete" style="margin-left:auto">удалить</button>
        </div>
        ${voice.ref_text ? `<div class="muted ref-text">в референсе: «${esc(voice.ref_text)}»</div>` : ''}

        <div class="grid-2">
          <label class="field">
            <span class="slider-head">Скорость речи <b data-role="speed-label">${cfg.speed.toFixed(2)}x</b></span>
            <input type="range" data-role="speed" min="0.5" max="2" step="0.05" value="${cfg.speed}" />
          </label>
          <label class="field">
            <span class="slider-head">CFG strength <b data-role="cfg-label">${cfg.cfg_strength.toFixed(1)}</b></span>
            <input type="range" data-role="cfg" min="1" max="4" step="0.1" value="${cfg.cfg_strength}" />
          </label>
        </div>

        <label class="field">
          <span>NFE steps — быстрее ↔ качественнее</span>
          <select data-role="nfe">
            ${nfeOptions(cfg.nfe_step)}
          </select>
        </label>

        <label class="field">
          <span>Фраза для прослушивания</span>
          <input type="text" data-role="preview-text" value="${esc(cfg.text)}" />
        </label>

        <div class="row">
          <button class="tiny primary" data-role="preview">Прослушать</button>
          <span class="muted" data-role="preview-status"></span>
        </div>
        <audio data-role="player" controls hidden></audio>
      </div>`;
  }).join('');
}

function setPreviewStatus(card, message, isError = false) {
  const el = card.querySelector('[data-role="preview-status"]');
  el.textContent = message;
  el.classList.toggle('err', isError);
}

async function previewVoice(card) {
  const voiceId = card.dataset.voiceId;
  const cfg = state.preview[voiceId];
  const text = card.querySelector('[data-role="preview-text"]').value.trim();
  showAlert($('voice-error'), '');

  if (!text) { setPreviewStatus(card, 'введите фразу для прослушивания'); return; }
  if (!state.modelReady) { setPreviewStatus(card, 'модель ещё загружается'); return; }
  if (state.previewJob) { setPreviewStatus(card, 'дождитесь окончания текущей генерации'); return; }

  setPreviewStatus(card, 'ставлю в очередь…');
  try {
    const job = await api('/api/preview', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        voice_id: voiceId,
        text,
        speed: cfg.speed,
        cfg_strength: cfg.cfg_strength,
        nfe_step: cfg.nfe_step,
      }),
    });
    state.previewJob = { jobId: job.job_id, voiceId };
    state.previewTimer = setInterval(pollPreview, 1500);
  } catch (error) {
    // Ошибку показываем у самой кнопки: иначе сетевой сбой читается как «кнопка не работает»
    setPreviewStatus(card, error.message, true);
  }
}

async function pollPreview() {
  const active = state.previewJob;
  if (!active) return;
  const card = previewCard(active.voiceId);
  if (!card) {  // карточку перерисовали или голос удалили — следить больше не за чем
    clearInterval(state.previewTimer);
    state.previewTimer = null;
    state.previewJob = null;
    return;
  }
  try {
    const job = await api(`/api/jobs/${active.jobId}`);
    if (job.status === 'queued') { setPreviewStatus(card, 'в очереди…'); return; }
    if (job.status === 'processing') { setPreviewStatus(card, 'генерирую…'); return; }
    clearInterval(state.previewTimer);
    state.previewTimer = null;
    state.previewJob = null;
    if (job.status === 'error') {
      setPreviewStatus(card, job.error || 'Не удалось синтезировать голос', true);
      return;
    }
    setPreviewStatus(card, `${job.duration_sec} с`);
    const player = card.querySelector('[data-role="player"]');
    player.src = `${job.audio_url}?t=${Date.now()}`;
    player.hidden = false;
    player.play().catch(() => { /* автоплей может быть заблокирован — плеер виден */ });
  } catch (error) {
    clearInterval(state.previewTimer);
    state.previewTimer = null;
    state.previewJob = null;
    setPreviewStatus(card, error.message, true);
  }
}

async function recognizeRefText() {
  const file = $('new-voice-file').files[0];
  const status = $('recognize-status');
  const button = $('btn-recognize-voice');
  showAlert($('voice-error'), '');
  if (!file) { status.textContent = 'сначала выберите аудиофайл'; return; }

  const form = new FormData();
  form.append('file', file);
  status.textContent = 'распознаю речь… это занимает до минуты';
  button.disabled = true;
  try {
    const result = await api('/api/voices/transcribe', { method: 'POST', body: form });
    $('new-voice-ref-text').value = result.ref_text;
    const trimmed = result.full_sec - result.effective_sec;
    const used = `использовано ${result.effective_sec.toFixed(1)} с из ${result.full_sec.toFixed(1)} с`;
    status.textContent = trimmed > 0.5
      ? `распознано: ${used} — в модель уйдёт только начало записи`
      : `распознано: ${used}`;
  } catch (error) {
    status.textContent = '';
    showAlert($('voice-error'), error.message);
  } finally {
    button.disabled = false;
  }
}

async function createVoice() {
  const file = $('new-voice-file').files[0];
  const status = $('new-voice-status');
  const refText = $('new-voice-ref-text').value.trim();
  showAlert($('voice-error'), '');
  if (!file) { status.textContent = 'выберите аудиофайл'; return; }

  const form = new FormData();
  form.append('name', $('new-voice-name').value.trim() || 'Новый голос');
  form.append('gender', $('new-voice-gender').value);
  form.append('ref_text', refText);
  form.append('file', file);

  // Пустое поле бэкенд заполняет расшифровкой сам — это десятки секунд.
  status.textContent = refText
    ? 'сверяю расшифровку с записью и сохраняю…'
    : 'распознаю речь и сохраняю… это занимает до минуты';
  try {
    const voice = await api('/api/voices', { method: 'POST', body: form });
    status.textContent = 'голос сохранён';
    $('new-voice-file').value = '';
    $('new-voice-name').value = '';
    $('new-voice-ref-text').value = '';
    $('recognize-status').textContent = '';
    resetDropZone();
    await loadVoices();
    const notes = [];
    if (voice.gender_warning) notes.push(voice.gender_warning);
    if (voice.ref_text_warning) notes.push(voice.ref_text_warning);
    if (voice.is_demo) notes.push(`Загружена копия демо-файла F5-TTS (${voice.demo_source}) — это не пользовательская запись.`);
    if (notes.length) showAlert($('voice-error'), notes.join('\n'), 'info');
  } catch (error) {
    status.textContent = '';
    showAlert($('voice-error'), error.message);
  }
}

function resetDropZone() {
  const drop = $('new-voice-drop');
  drop.textContent = DROP_HINT;
  drop.classList.remove('has-file', 'over');
}

// --- голоса диалога (слоты из маркеров и имена спикеров) ----------------------
function renderVoiceConfigs() {
  const box = $('voice-configs');
  if (!state.voiceKeys.length) {
    box.innerHTML = '<span class="muted">Вставьте диалог и нажмите «Определить голоса».</span>';
    return;
  }
  box.innerHTML = state.voiceKeys.map((key) => {
    const cfg = state.configs[key];
    const meta = state.voiceMeta[key] || { label: key, slot: null };
    const options = state.voices
      .map((v) => `<option value="${esc(v.id)}"${v.id === cfg.voice_id ? ' selected' : ''}>${esc(v.name)}</option>`)
      .join('');
    const markerTag = meta.slot === null
      ? ''
      : `<span class="tag">маркер (${meta.slot})</span>`;
    return `
      <div class="card" data-voice="${esc(key)}">
        <div class="card-title"><span class="swatch"></span>${esc(meta.label)} ${markerTag}</div>

        <label class="field">
          <span>Голос</span>
          <select data-role="voice">
            <option value="">— выберите голос —</option>
            ${options}
          </select>
        </label>

        <label class="field">
          <span class="slider-head">Скорость речи <b data-role="speed-label">${cfg.speed.toFixed(2)}x</b></span>
          <input type="range" data-role="speed" min="0.5" max="2" step="0.05" value="${cfg.speed}" />
        </label>
        <label class="field">
          <span class="slider-head">CFG strength — стабильность ↔ выразительность <b data-role="cfg-label">${cfg.cfg_strength.toFixed(1)}</b></span>
          <input type="range" data-role="cfg" min="1" max="4" step="0.1" value="${cfg.cfg_strength}" />
        </label>
        <label class="field" style="margin-bottom:0">
          <span>NFE steps — быстрее ↔ качественнее</span>
          <select data-role="nfe">
            ${nfeOptions(cfg.nfe_step)}
          </select>
        </label>
      </div>`;
  }).join('');
}

// Панель голосов должна отражать текущий текст: пользователь правит диалог, а не
// жмёт каждый раз «Определить голоса». Без этого новый слот («(2)», «(3)») уходил
// на бэкенд без голоса и задача падала с «Не назначен голос для: …».
async function syncVoices({ silent = false } = {}) {
  if (!silent) showAlert($('parse-error'), '');
  let parsed;
  try {
    parsed = await api('/api/parse', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ dialogue_text: $('dialogue').value }),
    });
  } catch (error) {
    if (!silent) showAlert($('parse-error'), error.message);
    return false;
  }

  if (!parsed.replicas.length && !silent) {
    showAlert($('parse-error'), 'Текст пуст — вставьте диалог или хотя бы одну фразу.');
  }

  const keys = parsed.voices.map((v) => v.key);
  const meta = {};
  parsed.voices.forEach((v) => { meta[v.key] = v; });
  // Подпись состава голосов: если она не изменилась, панель не перерисовываем,
  // иначе на каждой правке текста сбрасывались бы положения ползунков.
  const signature = (list, map) =>
    list.map((key) => `${key}|${map[key] ? map[key].label : ''}|${map[key] ? map[key].slot : ''}`).join('\n');
  const changed = signature(keys, meta) !== signature(state.voiceKeys, state.voiceMeta);

  // Настроенные параметры сохраняем и для голосов, исчезнувших из текста при
  // правке: пользователь может вернуть их обратно, настройки не потеряются.
  const configs = { ...state.configs };
  keys.forEach((key) => { configs[key] = configs[key] || defaultConfig(); });
  state.voiceKeys = keys;
  state.voiceMeta = meta;
  state.configs = configs;

  const overrides = parsed.override_count ? ` · параметров из текста: ${parsed.override_count}` : '';
  $('parse-summary').textContent =
    `${parsed.replicas.length} реплик · голосов: ${parsed.voices.length}${overrides}`;
  if (changed) renderVoiceConfigs();
  updateGenerateButton();
  return true;
}

async function detectVoices() {
  await syncVoices({ silent: false });
}

// --- работа с карточками голосов ----------------------------------------------
function readCardConfigs() {
  document.querySelectorAll('#voice-configs .card').forEach((card) => {
    const key = card.dataset.voice;
    const cfg = state.configs[key];
    cfg.voice_id = card.querySelector('[data-role="voice"]').value;
    cfg.speed = parseFloat(card.querySelector('[data-role="speed"]').value);
    cfg.cfg_strength = parseFloat(card.querySelector('[data-role="cfg"]').value);
    cfg.nfe_step = parseInt(card.querySelector('[data-role="nfe"]').value, 10);
  });
}

function updateGenerateButton() {
  const assigned = state.voiceKeys.length > 0 &&
    state.voiceKeys.every((key) => state.configs[key] && state.configs[key].voice_id);
  const button = $('btn-generate');
  button.disabled = !(state.modelReady && assigned) || Boolean(state.jobId);
  if (!state.modelReady) button.title = 'Модель ещё загружается';
  else if (!state.voiceKeys.length) button.title = 'Сначала определите голоса';
  else if (!assigned) button.title = 'Выберите голос для каждого участника диалога';
  else button.title = '';
}

// --- генерация диалога --------------------------------------------------------
async function generate() {
  // Текст мог измениться после последней синхронизации — сверяем список голосов
  // до отправки, иначе задача упадёт на бэкенде с «Не назначен голос для: …».
  if (!(await syncVoices({ silent: false }))) return;
  readCardConfigs();
  showAlert($('job-error'), '');

  const unassigned = state.voiceKeys.filter((key) => !state.configs[key].voice_id);
  if (unassigned.length) {
    const names = unassigned
      .map((key) => `«${(state.voiceMeta[key] || {}).label || key}»`)
      .join(', ');
    showAlert($('job-error'), `Выберите голос для: ${names}`);
    updateGenerateButton();
    return;
  }

  const payload = {
    dialogue_text: $('dialogue').value,
    speakers: state.configs,
    pause_ms: parseInt($('pause').value, 10),
    cross_fade_duration: parseFloat($('crossfade').value),
    auto_accent: $('auto-accent').checked,
    output_format: $('output-format').value,
  };

  $('btn-generate').disabled = true;
  $('player').hidden = true;
  $('download-link').hidden = true;
  setProgress(0);

  try {
    const job = await api('/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    state.jobId = job.job_id;
    $('job-status').textContent = `задача ${job.job_id}: в очереди`;
    state.pollTimer = setInterval(pollJob, 1500);
  } catch (error) {
    showAlert($('job-error'), error.message);
    state.jobId = null;
    updateGenerateButton();
  }
}

function setProgress(ratio) {
  $('progress-bar').style.width = `${Math.round(ratio * 100)}%`;
}

async function pollJob() {
  try {
    const job = await api(`/api/jobs/${state.jobId}`);
    setProgress(job.progress);
    if (job.status === 'queued') {
      $('job-status').textContent = 'в очереди…';
      return;
    }
    if (job.status === 'processing') {
      const eta = job.eta_sec ? `, осталось ~${Math.round(job.eta_sec)} с` : '';
      $('job-status').textContent = `${job.message}${eta}`;
      return;
    }
    clearInterval(state.pollTimer);
    state.pollTimer = null;
    state.jobId = null;
    if (job.status === 'error') {
      $('job-status').textContent = 'ошибка';
      showAlert($('job-error'), job.error || 'Неизвестная ошибка генерации');
      setProgress(0);
    } else {
      $('job-status').textContent = `готово · ${job.duration_sec} с аудио`;
      setProgress(1);
      const url = `${job.audio_url}?t=${Date.now()}`;
      $('player').src = url;
      $('player').hidden = false;
      const link = $('download-link');
      link.href = `${job.audio_url}?download=true`;
      link.download = `dialogue.${job.output_format}`;
      link.hidden = false;
    }
    updateGenerateButton();
  } catch (error) {
    clearInterval(state.pollTimer);
    state.pollTimer = null;
    state.jobId = null;
    showAlert($('job-error'), error.message);
    updateGenerateButton();
  }
}

// --- сплошной текст (один голос на весь файл) ---------------------------------
function renderTextVoiceOptions() {
  const select = $('text-voice');
  const current = select.value;
  select.innerHTML = '<option value="">— выберите голос —</option>' +
    state.voices.map((v) => `<option value="${esc(v.id)}">${esc(v.name)}</option>`).join('');
  const fallback = state.voices.length ? state.voices[0].id : '';
  select.value = state.voices.some((v) => v.id === current) ? current : fallback;
  updateTextButton();
}

function updateTextSummary() {
  const chars = $('text-body').value.trim().length;
  $('text-summary').textContent = chars ? `${chars} символов` : '';
}

function updateTextButton() {
  const button = $('btn-render-text');
  const hasVoice = Boolean($('text-voice').value);
  const hasText = Boolean($('text-body').value.trim());
  button.disabled = !(state.modelReady && hasVoice && hasText) || Boolean(state.textJobId);
  if (!state.modelReady) button.title = 'Модель ещё загружается';
  else if (!hasVoice) button.title = 'Выберите голос';
  else if (!hasText) button.title = 'Вставьте текст или загрузите файл';
  else button.title = '';
}

function loadTextFile(file) {
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    $('text-body').value = String(reader.result || '');
    const drop = $('text-drop');
    drop.textContent = file.name;
    drop.classList.add('has-file');
    updateTextSummary();
    updateTextButton();
  };
  reader.onerror = () => showAlert($('text-error'), `Не удалось прочитать файл ${file.name}`);
  reader.readAsText(file, 'utf-8');
}

function setTextProgress(ratio) {
  $('text-progress-bar').style.width = `${Math.round(ratio * 100)}%`;
}

async function renderText() {
  showAlert($('text-job-error'), '');
  const payload = {
    text: $('text-body').value,
    voice_id: $('text-voice').value,
    speed: parseFloat($('text-speed').value),
    cfg_strength: parseFloat($('text-cfg').value),
    nfe_step: parseInt($('text-nfe').value, 10),
    pause_ms: parseInt($('text-pause').value, 10),
    cross_fade_duration: parseFloat($('text-crossfade').value),
    auto_accent: $('text-auto-accent').checked,
    output_format: $('text-output-format').value,
  };

  $('btn-render-text').disabled = true;
  $('text-player').hidden = true;
  $('text-download-link').hidden = true;
  setTextProgress(0);

  try {
    const job = await api('/api/render-text', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    state.textJobId = job.job_id;
    $('text-job-status').textContent = `задача ${job.job_id}: кусков ${job.total_replicas}`;
    state.textPollTimer = setInterval(pollTextJob, 1500);
  } catch (error) {
    showAlert($('text-job-error'), error.message);
    state.textJobId = null;
    updateTextButton();
  }
}

async function pollTextJob() {
  try {
    const job = await api(`/api/jobs/${state.textJobId}`);
    setTextProgress(job.progress);
    if (job.status === 'queued') {
      $('text-job-status').textContent = 'в очереди…';
      return;
    }
    if (job.status === 'processing') {
      const eta = job.eta_sec ? `, осталось ~${Math.round(job.eta_sec)} с` : '';
      $('text-job-status').textContent = `кусок ${job.current_replica + 1} из ${job.total_replicas}${eta}`;
      return;
    }
    clearInterval(state.textPollTimer);
    state.textPollTimer = null;
    state.textJobId = null;
    if (job.status === 'error') {
      $('text-job-status').textContent = 'ошибка';
      showAlert($('text-job-error'), job.error || 'Неизвестная ошибка генерации');
      setTextProgress(0);
    } else {
      $('text-job-status').textContent = `готово · ${job.duration_sec} с аудио`;
      setTextProgress(1);
      $('text-player').src = `${job.audio_url}?t=${Date.now()}`;
      $('text-player').hidden = false;
      const link = $('text-download-link');
      link.href = `${job.audio_url}?download=true`;
      link.download = `text.${job.output_format}`;
      link.hidden = false;
    }
    updateTextButton();
  } catch (error) {
    clearInterval(state.textPollTimer);
    state.textPollTimer = null;
    state.textJobId = null;
    showAlert($('text-job-error'), error.message);
    updateTextButton();
  }
}

// --- статус модели ------------------------------------------------------------
async function refreshStatus() {
  const dot = $('status-dot');
  const text = $('status-text');
  try {
    const status = await api('/api/status');
    state.modelReady = status.model_loaded;

    // Устройство известно только после загрузки модели: не показываем «на null».
    const model = status.model_loaded
      ? `модель готова · ${status.device || 'устройство неизвестно'}`
      : `загружаю модель${status.device ? ` на ${status.device}` : ''}…`;
    const accent = {
      ready: 'ударения: вкл',
      failed: 'ударения: ОШИБКА',
      loading: 'ударения: загружаются',
      idle: 'ударения: загружаются',
    }[status.accentizer_state] || `ударения: ${status.accentizer_state}`;
    text.textContent = `${model} · ${accent}`;

    if (status.accentizer_state === 'failed') {
      dot.className = 'dot err';
      text.title = status.accentizer_error
        ? `RUAccent не поднялся, синтез идёт без ударений: ${status.accentizer_error}`
        : 'RUAccent не поднялся, синтез идёт без ударений';
    } else {
      dot.className = `dot ${status.model_loaded ? 'on' : 'busy'}`;
      text.title = '';
    }

    // Прекращаем опрос, только когда устоялось и то, и другое: иначе статус
    // ударений замирал на «загружаются» — модель-то уже готова.
    const accentSettled = status.accentizer_state === 'ready' || status.accentizer_state === 'failed';
    if (status.model_loaded && accentSettled) {
      clearInterval(state.statusTimer);
      state.statusTimer = null;
    }
  } catch (error) {
    dot.className = 'dot err';
    text.textContent = 'бэкенд недоступен';
    text.title = '';
  }
  updateGenerateButton();
  updateTextButton();
}

// --- события ------------------------------------------------------------------
function bindEvents() {
  document.querySelectorAll('.tab').forEach((tab) => {
    tab.addEventListener('click', () => switchTab(tab.dataset.tab));
  });

  $('btn-detect').addEventListener('click', detectVoices);
  $('btn-generate').addEventListener('click', generate);
  $('btn-reload-voices').addEventListener('click', () => loadVoices().catch((e) => alert(e.message)));
  $('btn-save-voice').addEventListener('click', createVoice);
  $('btn-recognize-voice').addEventListener('click', recognizeRefText);

  $('pause').addEventListener('input', (e) => { $('pause-value').textContent = `${e.target.value} мс`; });
  $('crossfade').addEventListener('input', (e) => {
    $('crossfade-value').textContent = `${parseFloat(e.target.value).toFixed(2)} с`;
  });

  // Правка текста меняет состав голосов (новые маркеры, новые имена) — панель
  // должна успевать за ней сама, без нажатия «Определить голоса». Пауза нужна,
  // чтобы не дёргать разбор на каждом символе.
  let dialogueSyncTimer = null;
  $('dialogue').addEventListener('input', () => {
    clearTimeout(dialogueSyncTimer);
    dialogueSyncTimer = setTimeout(() => { syncVoices({ silent: true }); }, 600);
  });

  // --- вкладка «Сплошной текст» ---
  const textDrop = $('text-drop');
  const textFile = $('text-file');
  textDrop.addEventListener('click', () => textFile.click());
  textFile.addEventListener('change', () => loadTextFile(textFile.files[0]));
  textDrop.addEventListener('dragover', (event) => {
    event.preventDefault();
    textDrop.classList.add('over');
  });
  textDrop.addEventListener('dragleave', () => textDrop.classList.remove('over'));
  textDrop.addEventListener('drop', (event) => {
    event.preventDefault();
    textDrop.classList.remove('over');
    loadTextFile(event.dataTransfer.files[0]);
  });

  $('text-body').addEventListener('input', () => { updateTextSummary(); updateTextButton(); });
  $('text-voice').addEventListener('change', updateTextButton);
  $('btn-render-text').addEventListener('click', renderText);
  $('text-pause').addEventListener('input', (e) => { $('text-pause-value').textContent = `${e.target.value} мс`; });
  $('text-crossfade').addEventListener('input', (e) => {
    $('text-crossfade-value').textContent = `${parseFloat(e.target.value).toFixed(2)} с`;
  });
  $('text-speed').addEventListener('input', (e) => {
    $('text-speed-label').textContent = `${parseFloat(e.target.value).toFixed(2)}x`;
  });
  $('text-cfg').addEventListener('input', (e) => {
    $('text-cfg-label').textContent = parseFloat(e.target.value).toFixed(1);
  });

  // --- форма нового голоса ---
  const drop = $('new-voice-drop');
  const fileInput = $('new-voice-file');
  drop.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', () => {
    const file = fileInput.files[0];
    if (!file) return;
    drop.textContent = file.name;
    drop.classList.add('has-file');
  });
  drop.addEventListener('dragover', (event) => {
    event.preventDefault();
    drop.classList.add('over');
  });
  drop.addEventListener('dragleave', () => drop.classList.remove('over'));
  drop.addEventListener('drop', (event) => {
    event.preventDefault();
    drop.classList.remove('over');
    const file = event.dataTransfer.files[0];
    if (!file) return;
    fileInput.files = event.dataTransfer.files;
    drop.textContent = file.name;
    drop.classList.add('has-file');
  });

  // --- карточки голосов ---
  const voiceCards = $('voice-cards');
  voiceCards.addEventListener('input', (event) => {
    const card = event.target.closest('.card');
    if (!card) return;
    const cfg = state.preview[card.dataset.voiceId];
    const role = event.target.dataset.role;
    if (role === 'speed') {
      cfg.speed = parseFloat(event.target.value);
      card.querySelector('[data-role="speed-label"]').textContent = `${cfg.speed.toFixed(2)}x`;
    }
    if (role === 'cfg') {
      cfg.cfg_strength = parseFloat(event.target.value);
      card.querySelector('[data-role="cfg-label"]').textContent = cfg.cfg_strength.toFixed(1);
    }
    if (role === 'preview-text') cfg.text = event.target.value;
  });

  voiceCards.addEventListener('change', (event) => {
    const card = event.target.closest('.card');
    if (!card) return;
    if (event.target.dataset.role === 'nfe') {
      state.preview[card.dataset.voiceId].nfe_step = parseInt(event.target.value, 10);
    }
  });

  voiceCards.addEventListener('click', async (event) => {
    const card = event.target.closest('.card');
    const button = event.target.closest('button');
    if (!card || !button) return;
    if (button.dataset.role === 'preview') previewVoice(card);
    if (button.dataset.role === 'delete') {
      const voiceId = card.dataset.voiceId;
      if (!confirm('Удалить голос? Файл референса будет стёрт.')) return;
      try {
        await api(`/api/voices/${voiceId}`, { method: 'DELETE' });
        delete state.preview[voiceId];
        await loadVoices();
      } catch (error) {
        showAlert($('voice-error'), error.message);
      }
    }
  });

  // --- карточки голосов диалога ---
  const voiceConfigs = $('voice-configs');
  voiceConfigs.addEventListener('input', (event) => {
    const card = event.target.closest('.card');
    if (!card) return;
    const role = event.target.dataset.role;
    if (role === 'speed') card.querySelector('[data-role="speed-label"]').textContent = `${parseFloat(event.target.value).toFixed(2)}x`;
    if (role === 'cfg') card.querySelector('[data-role="cfg-label"]').textContent = parseFloat(event.target.value).toFixed(1);
    readCardConfigs();
    updateGenerateButton();
  });

  voiceConfigs.addEventListener('change', (event) => {
    if (event.target.dataset.role === 'voice') {
      readCardConfigs();
      updateGenerateButton();
    }
  });
}

// --- старт --------------------------------------------------------------------
async function init() {
  bindEvents();
  // Таймер создаём до первого опроса: если всё уже готово, refreshStatus его снимет.
  state.statusTimer = setInterval(refreshStatus, 4000);
  await refreshStatus();
  try {
    await loadVoices();
  } catch (error) {
    showAlert($('voice-error'), error.message);
  }
}

init();
