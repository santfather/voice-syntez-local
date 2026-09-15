'use strict';

const $ = (id) => document.getElementById(id);

const state = {
  voices: [],
  engines: {},         // id -> паспорт движка из /api/engines
  engineTouched: false, // пользователь сам выбрал движок в форме нового голоса
  voiceKeys: [],       // ["#1", "#2", "ИВАН"] — голоса, встреченные в тексте
  voiceMeta: {},       // key -> {key, label, slot}
  configs: {},         // key -> {voice_id, speed, cfg_strength, nfe_step, engine_params}
  preview: {},         // voice_id -> {speed, cfg_strength, nfe_step, engine_params, text}
  textEngineParams: {}, // ручки движка в режиме «Сплошной текст»
  textEngineVoice: '',  // голос, к которому относятся эти ручки
  modelReady: false,
  jobId: null,
  pollTimer: null,
  doneJobId: null,      // id последней готовой задачи — по нему перегенерация реплик
  regenTimer: null,
  regenIndex: null,
  variantKey: null,     // "индекс:вариант" того варианта, что играет в плеере вариантов
  statusTimer: null,
  previewJob: null,    // {jobId, voiceId}
  previewTimer: null,
  textJobId: null,     // задача режима «Сплошной текст»
  textPollTimer: null,
  record: {            // запись голоса с микрофона
    recorder: null,
    stream: null,
    chunks: [],
    suffix: 'webm',
    file: null,
    url: null,
    startedAt: 0,
    peak: 0,
    analyser: null,
    ctx: null,
    raf: null,
    autoStop: null,
    tick: null,
  },
};

const NFE_OPTIONS = [8, 16, 32];
const PREVIEW_TEXT = 'Привет! Так звучит этот голос в диалоге.';
const DROP_HINT = 'Перетащите сюда аудио (wav/mp3/m4a) или кликните';

// Идентификаторы движков совпадают с backend/engines/base.py. Нужны здесь только
// для подсказки по полу: паспорта всех движков приходят из /api/engines.
const ENGINE_F5 = 'f5';
const ENGINE_XTTS = 'xtts';

// Фразы для референса. F5-TTS клонирует голос по паре «запись + её дословная
// расшифровка», поэтому текст известен заранее — в отличие от загруженного файла,
// где его приходится распознавать после записи. Фразы разные по составу звуков и
// интонации: модель копирует то, что слышит, и на однотипных утверждениях
// клонирует слишком узкий кусок голосового диапазона.
const RECORD_PHRASES = [
  {
    label: 'Вопрос и утверждение',
    text: 'Добрый вечер. Мы договаривались встретиться у метро, но я вас так и не увидел. Вы точно получили моё сообщение?',
  },
  {
    label: 'Восклицания, много шипящих',
    text: 'Осторожно, здесь очень скользко! Я чуть не упал, когда выбегал из подъезда. И это уже третий раз за неделю.',
  },
  {
    label: 'Спокойная просьба (короче)',
    text: 'Дайте пройти, пожалуйста. Я вас совсем не знаю, и мне нечего вам сказать.',
  },
  {
    label: 'Только вопросы',
    text: 'Ты уверен, что мы правильно свернули? Кажется, этот поворот был раньше. Может, спросим дорогу у кого-нибудь?',
  },
  {
    label: 'Сложные сочетания согласных',
    text: 'Съешь ещё этих мягких французских булочек, а потом расскажешь, как прошла твоя поездка в Ярославль.',
  },
];

// MediaRecorder отдаёт то, что умеет браузер: Chrome — webm/opus, Safari — mp4/aac.
// Все три формата бэкенд принимает и читает через ffmpeg.
const RECORD_FORMATS = [
  ['audio/webm;codecs=opus', 'webm'],
  ['audio/webm', 'webm'],
  ['audio/mp4', 'm4a'],
  ['audio/ogg;codecs=opus', 'ogg'],
];
// За фразы из списка длиннее 12 секунд брать не нужно: F5-TTS всё равно обрежет
// референс. Потолок нужен, чтобы забытая запись не тянулась минутами.
const RECORD_MAX_MS = 20000;
// Порог перегрузки — тот же, что в audio_analysis.CLIPPING_LEVEL: индикатор должен
// показывать красное ровно там, где бэкенд потом скажет «запись перегружена».
const CLIPPING_LEVEL = 0.99;

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

// --- движки синтеза -----------------------------------------------------------
// Движок выбирается отдельно для каждого голоса, поэтому один и тот же диалог
// может озвучиваться несколькими моделями: бэкенд держит их поднятыми в одном
// процессе и ходит в них по очереди (см. backend/engines, audio_pipeline).
// Подписи ручек и их границы объявлены на бэкенде (/api/engines) — здесь только
// отрисовка того, что он прислал: иначе фронт и модель разошлись бы в допустимых
// значениях, и ползунок обещал бы одно, а бэкенд молча обрезал значение.

const voiceById = (voiceId) => state.voices.find((v) => v.id === voiceId) || null;

function engineInfo(engineId) {
  return state.engines[engineId] || null;
}

function engineLabel(engineId) {
  const info = engineInfo(engineId);
  return info ? info.label : engineId;
}

function engineOptionsHtml(selected) {
  return Object.values(state.engines)
    .map((info) => `<option value="${esc(info.id)}"${info.id === selected ? ' selected' : ''}>${esc(info.label)}</option>`)
    .join('');
}

// Подсказка движка по полу повторяет default_engine_for_gender из backend/engines/base.py
// (F5 обучен на разметке ударений и ровнее на мужских голосах, базовая XTTS — наоборот).
// Это предзаполнение формы, а не правило.
function suggestedEngine(gender) {
  return gender === 'female' ? ENGINE_XTTS : ENGINE_F5;
}

// Короткая строка про движок голоса: что за модель и работает ли с ней RUAccent.
function engineNote(engineId) {
  const info = engineInfo(engineId);
  if (!info) return 'движок не выбран';
  return info.supports_accents
    ? `${info.label} · ударения расставляются автоматически`
    : `${info.label} · ударения не поддерживаются — текст идёт как есть`;
}

const fmtByStep = (value, step) =>
  (step >= 1 ? String(Math.round(value)) : value.toFixed(step < 0.1 ? 2 : 1));

// Значения ручек движка: то, что уже сохранено у голоса или набрано в карточке,
// иначе — дефолт из паспорта. Ручки чужого движка отбрасываются: у XTTS нет
// nfe_step, а у F5 нет temperature, и тащить их через запрос незачем.
function engineParamsFor(engineId, values) {
  const info = engineInfo(engineId);
  const result = {};
  if (!info) return result;
  info.params.forEach((param) => {
    const raw = values ? values[param.name] : undefined;
    result[param.name] = typeof raw === 'number' ? raw : param.default;
  });
  return result;
}

function engineParamsHtml(engineId, values) {
  const info = engineInfo(engineId);
  if (!info || !info.params.length) return '';
  const current = engineParamsFor(engineId, values);
  return `<div class="grid-2">${info.params.map((param) => `
    <label class="field">
      <span class="slider-head">${esc(param.label)}
        <b data-param-label="${esc(param.name)}">${fmtByStep(current[param.name], param.step)}</b></span>
      <input type="range" data-engine-param="${esc(param.name)}"
             min="${param.min}" max="${param.max}" step="${param.step}" value="${current[param.name]}" />
      ${param.hint ? `<span class="muted">${esc(param.hint)}</span>` : ''}
    </label>`).join('')}</div>`;
}

function readEngineParams(container) {
  const result = {};
  container.querySelectorAll('[data-engine-param]').forEach((input) => {
    result[input.dataset.engineParam] = parseFloat(input.value);
  });
  return result;
}

function updateEngineParamLabel(container, input) {
  const label = container.querySelector(`[data-param-label="${input.dataset.engineParam}"]`);
  if (label) label.textContent = fmtByStep(parseFloat(input.value), parseFloat(input.step));
}

async function loadEngines() {
  const data = await api('/api/engines');
  state.engines = {};
  data.engines.forEach((info) => { state.engines[info.id] = info; });
  const select = $('new-voice-engine');
  select.innerHTML = engineOptionsHtml(select.value || suggestedEngine($('new-voice-gender').value));
  updateNewVoiceEngineNote();
}

function updateNewVoiceEngineNote() {
  const info = engineInfo($('new-voice-engine').value);
  $('new-voice-engine-note').textContent = info
    ? `${info.description}${info.note ? ` ${info.note}` : ''}`
    : '';
}

function defaultConfig() {
  // Голос не подставляем: одинаковый голос на все слоты — это ровно та ошибка,
  // из-за которой диалог читается одним голосом. Выбор должен быть осознанным.
  return {
    voice_id: '',
    speed: 1.0,
    cfg_strength: 2.0,
    nfe_step: 32,
    target_rms: 0.1,
    gain_db: 0,
    pitch_semitones: 0,
    pause_override_ms: null,  // null — пауза общая для всего диалога
    engine_params: {},        // ручки движка, переопределённые для этого слота
  };
}

// Подписи «ещё настроек» спикера: одинаковы при отрисовке и при движении ползунка.
const fmtGain = (value) => `${value.toFixed(1)} дБ`;
const fmtPitch = (value) => `${value.toFixed(1)} пт`;
const fmtRms = (value) => value.toFixed(2);
const fmtPause = (value) => (value === null ? 'общая' : `${value} мс`);

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
  if (voice.band_warning) {
    tags.push(`<span class="tag warn" title="${esc(voice.band_warning)}">⚠ узкая полоса</span>`);
  }
  return tags.join('');
}

// Монограмма в карточке: первые буквы первых двух слов имени (Марго → МА).
// Служебные символы выкидываются, иначе у имени вида «[demo] REF_EN» в квадрат
// попадала бы скобка.
function initials(name) {
  const words = String(name || '').replace(/[^\p{L}\p{N}]+/gu, ' ').trim().split(/\s+/).filter(Boolean);
  return words.length ? words.slice(0, 2).map((word) => word[0]).join('').toUpperCase() : '??';
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
    const voice = voiceById(voiceId);
    state.preview[voiceId] = {
      speed: 1.0,
      cfg_strength: 2.0,
      nfe_step: 32,
      // Ручки движка начинаются с сохранённых у голоса — тех же, что уйдут в диалог.
      engine_params: voice ? { ...voice.engine_params } : {},
      text: PREVIEW_TEXT,
    };
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
    const isF5 = voice.engine === ENGINE_F5;
    // Движок читается по монограмме и тегу справа, поэтому подпись движка в
    // подзаголовке короткая — расшифровка «ударения/не поддерживаются» ушла в
    // подсказку тега (engineNote).
    const tags = voiceTags(voice)
      + (voice.ref_text ? '' : '<span class="tag warn" title="Без референс-текста синтез невозможен">нет текста</span>');
    return `
      <div class="card" data-voice-id="${esc(voice.id)}">
        <div class="voice-top">
          <span class="avatar${isF5 ? '' : ' violet'}">${esc(initials(voice.name))}</span>
          <div class="voice-name">
            <h3>${esc(voice.name)}</h3>
            <p data-role="engine-note" title="${esc(engineNote(voice.engine))}">${esc(engineLabel(voice.engine))}</p>
          </div>
          <span class="engine-tag${isF5 ? '' : ' violet'}" title="${esc(engineNote(voice.engine))}">${esc(voice.engine.toUpperCase())}</span>
        </div>
        <div class="tag-row">
          ${tags}
          <button class="tiny ghost danger" data-role="delete" style="margin-left:auto">удалить</button>
        </div>
        ${voice.ref_text ? `<div class="ref-text">в референсе: «${esc(voice.ref_text)}»</div>` : ''}

        <label class="field">
          <span>Движок синтеза</span>
          <select data-role="engine">${engineOptionsHtml(voice.engine)}</select>
        </label>

        <div class="grid-2">
          <label class="field">
            <span class="slider-head">Скорость речи <b data-role="speed-label">${cfg.speed.toFixed(2)}x</b></span>
            <input type="range" data-role="speed" min="0.5" max="2" step="0.05" value="${cfg.speed}" />
          </label>
          <label class="field" data-role="f5-params" ${isF5 ? '' : 'hidden'}>
            <span class="slider-head">CFG strength <b data-role="cfg-label">${cfg.cfg_strength.toFixed(1)}</b></span>
            <input type="range" data-role="cfg" min="1" max="4" step="0.1" value="${cfg.cfg_strength}" />
          </label>
        </div>

        <label class="field" data-role="f5-params" ${isF5 ? '' : 'hidden'}>
          <span>NFE steps — быстрее ↔ качественнее</span>
          <select data-role="nfe">
            ${nfeOptions(cfg.nfe_step)}
          </select>
        </label>

        <div data-role="engine-params">${engineParamsHtml(voice.engine, cfg.engine_params)}</div>

        <label class="field">
          <span>Фраза для прослушивания</span>
          <input type="text" data-role="preview-text" value="${esc(cfg.text)}" />
        </label>

        <div class="test-row">
          <button class="tiny primary" data-role="preview">Прослушать</button>
          <span class="muted preview-status" data-role="preview-status"></span>
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
  // Движок голоса может подниматься прямо сейчас — следим за его состоянием в шапке.
  watchEngines();
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
        engine_params: cfg.engine_params,
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

// Смена движка у готового голоса: выбор сохраняется на бэкенде, запись референса
// не трогается. Ручки прошлого движка к новому не относятся, поэтому бэкенд их
// сбрасывает (voices_store.update) — перечитываем голоса, чтобы карточка показала
// набор настроек нового движка.
async function changeVoiceEngine(card, engineId) {
  const voiceId = card.dataset.voiceId;
  showAlert($('voice-error'), '');
  try {
    const voice = await api(`/api/voices/${voiceId}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ engine: engineId }),
    });
    delete state.preview[voiceId];
    await loadVoices();
    showAlert($('voice-error'), `Голос «${voice.name}» переведён на ${engineLabel(voice.engine)}`, 'info');
  } catch (error) {
    // Селект уже показывает новый движок, которого на бэкенде нет — возвращаем как было.
    showAlert($('voice-error'), error.message);
    renderVoiceCards();
  }
}

// Ручки движка сохраняются у голоса и работают его настройками по умолчанию:
// карточка слота в диалоге может их переопределить на одну генерацию.
async function saveVoiceEngineParams(voiceId, params) {
  try {
    await api(`/api/voices/${voiceId}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ engine_params: params }),
    });
  } catch (error) {
    showAlert($('voice-error'), error.message);
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
  const verify = $('new-voice-verify').checked;
  const denoise = $('new-voice-denoise').checked;
  showAlert($('voice-error'), '');
  if (!file) { status.textContent = 'выберите аудиофайл или запишите голос'; return; }

  const form = new FormData();
  form.append('name', $('new-voice-name').value.trim() || 'Новый голос');
  form.append('gender', $('new-voice-gender').value);
  // Движок сохраняется вместе с голосом: все диалоги этим голосом пойдут через него.
  form.append('engine', $('new-voice-engine').value);
  form.append('ref_text', refText);
  // Без сверки бэкенд берёт расшифровку как есть: для записанного голоса текст
  // прочитан с экрана, и распознавание здесь было бы лишними десятками секунд.
  form.append('verify_ref_text', verify ? '1' : '0');
  // Очистка референса — опциональная зависимость: бэкенд чистит запись до всех
  // проверок, чтобы F0, полоса и расшифровка считались по той же записи, что уйдёт в модель.
  form.append('denoise', denoise ? '1' : '0');
  form.append('file', file);

  const cleaning = denoise ? 'чищу запись от шума, ' : '';
  if (!verify) status.textContent = `${cleaning}сохраняю голос…`;
  else if (refText) status.textContent = `${cleaning}сверяю расшифровку с записью и сохраняю…`;
  else status.textContent = `${cleaning}распознаю речь и сохраняю… это занимает до минуты`;
  try {
    const voice = await api('/api/voices', { method: 'POST', body: form });
    status.textContent = 'голос сохранён';
    $('new-voice-file').value = '';
    $('new-voice-name').value = '';
    $('new-voice-ref-text').value = '';
    $('recognize-status').textContent = '';
    // Очистка — разовая операция при сохранении, а не свойство голоса: снимаем галочку.
    $('new-voice-denoise').checked = false;
    resetDropZone();
    resetRecording();
    // Форма обнуляется вместе с выбором движка: подсказка по полу снова в силе.
    state.engineTouched = false;
    $('new-voice-engine').value = suggestedEngine($('new-voice-gender').value);
    updateNewVoiceEngineNote();
    await loadVoices();
    const notes = [];
    if (voice.gender_warning) notes.push(voice.gender_warning);
    if (voice.ref_text_warning) notes.push(voice.ref_text_warning);
    if (voice.band_warning) notes.push(voice.band_warning);
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

// --- запись голоса с микрофона -------------------------------------------------
function recordFormat() {
  if (typeof MediaRecorder === 'undefined') return null;
  return RECORD_FORMATS.find(([mime]) => MediaRecorder.isTypeSupported(mime)) || null;
}

function showRecordPhrase() {
  const index = parseInt($('record-phrase').value || '0', 10);
  const phrase = RECORD_PHRASES[index] || RECORD_PHRASES[0];
  $('record-phrase-text').textContent = phrase.text;
}

function renderRecordPhrases() {
  $('record-phrase').innerHTML = RECORD_PHRASES
    .map((phrase, index) => `<option value="${index}">${esc(phrase.label)}</option>`)
    .join('');
  showRecordPhrase();
}

function setRecordStatus(message) {
  $('record-status').textContent = message;
}

// Встроенный браузер IDE (Electron) до микрофона не допускается: разрешение выдаёт
// оболочка приложения, а не страница, поэтому запрос даже не показывается — вызов
// отклоняется сразу. Отличить его от обычного браузера можно по UA, иначе
// пользователь ищет настройку в браузере, которого перед ним нет.
const isEmbeddedBrowser = () => /Electron/i.test(navigator.userAgent || '');

function recordErrorMessage(error) {
  switch (error.name) {
    case 'NotAllowedError':
    case 'SecurityError':
      if (isEmbeddedBrowser()) {
        return 'Микрофон недоступен: дашборд открыт во встроенном браузере IDE, а он доступа '
          + 'к микрофону не имеет. Разрешите IDE доступ в «Системные настройки → '
          + 'Конфиденциальность и безопасность → Микрофон» и перезапустите её — либо откройте '
          + `${location.origin} в Safari или Chrome, там запрос появится сам.`;
      }
      return 'Доступ к микрофону запрещён. Разрешите его для этой страницы в настройках браузера и нажмите «Записать» снова.';
    case 'NotFoundError':
    case 'OverconstrainedError':
      return 'Микрофон не найден. Подключите его или загрузите готовый файл слева.';
    case 'NotReadableError':
      return 'Микрофон занят другим приложением — закройте его и попробуйте снова.';
    default:
      return `Не удалось начать запись: ${error.message || error.name}`;
  }
}

async function startRecording() {
  showAlert($('record-error'), '');
  showAlert($('record-notes'), '');

  const format = recordFormat();
  if (!format) {
    showAlert($('record-error'), 'Браузер не умеет записывать звук. Загрузите готовый файл.');
    return;
  }
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    // getUserMedia работает только в защищённом контексте: localhost тоже подходит.
    showAlert($('record-error'), 'Запись с микрофона доступна только на localhost или по https.');
    return;
  }

  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (error) {
    showAlert($('record-error'), recordErrorMessage(error));
    return;
  }

  const record = state.record;
  const recorder = new MediaRecorder(stream, { mimeType: format[0] });
  record.stream = stream;
  record.recorder = recorder;
  record.chunks = [];
  record.suffix = format[1];
  record.startedAt = Date.now();
  record.peak = 0;

  recorder.addEventListener('dataavailable', (event) => {
    if (event.data.size) record.chunks.push(event.data);
  });
  recorder.addEventListener('stop', finishRecording);
  recorder.start();

  // Индикатор уровня: без него не видно ни «говорю в тишину», ни перегрузки.
  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  const analyser = ctx.createAnalyser();
  analyser.fftSize = 2048;
  ctx.createMediaStreamSource(stream).connect(analyser);
  record.ctx = ctx;
  record.analyser = analyser;
  $('record-meter').hidden = false;
  drawRecordMeter();

  // Остановка по таймеру — страховка от забытой записи: распознавать минуту
  // молчания всё равно нечего, а референс длиннее 12 секунд модель обрежет.
  record.autoStop = setTimeout(stopRecording, RECORD_MAX_MS);
  record.tick = setInterval(() => {
    setRecordStatus(`идёт запись · ${((Date.now() - record.startedAt) / 1000).toFixed(1)} с`);
  }, 200);

  $('btn-record').textContent = 'Стоп';
  setRecordStatus('идёт запись · 0.0 с');
}

function stopRecording() {
  const record = state.record;
  if (!record.recorder) return;
  clearTimeout(record.autoStop);
  clearInterval(record.tick);
  record.autoStop = null;
  record.tick = null;
  if (record.recorder.state !== 'inactive') record.recorder.stop();
}

function drawRecordMeter() {
  const record = state.record;
  const canvas = $('record-meter');
  if (!record.analyser || canvas.hidden) return;
  const data = new Uint8Array(record.analyser.fftSize);
  record.analyser.getByteTimeDomainData(data);
  let peak = 0;
  for (const sample of data) peak = Math.max(peak, Math.abs(sample - 128) / 128);
  // Пик держим с затуханием: мгновенное значение мелькает быстрее, чем глаз успевает
  // заметить перегрузку.
  record.peak = Math.max(peak, record.peak * 0.9);
  drawMeterBar(canvas, record.peak);
  record.raf = requestAnimationFrame(drawRecordMeter);
}

function drawMeterBar(canvas, peak) {
  const ctx = canvas.getContext('2d');
  const { width, height } = canvas;
  const barHeight = height - 14;
  const level = Math.min(peak, 1);
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = '#12151c';
  ctx.fillRect(0, 0, width, barHeight);
  ctx.fillStyle = peak >= CLIPPING_LEVEL ? '#e2585f' : peak >= 0.7 ? '#e8a33d' : '#34c38f';
  ctx.fillRect(0, 0, level * width, barHeight);
  ctx.fillStyle = '#2c3444';
  ctx.fillRect(CLIPPING_LEVEL * width, 0, Math.max(width - CLIPPING_LEVEL * width, 1), barHeight);
  ctx.font = '11px -apple-system, sans-serif';
  ctx.fillStyle = peak >= CLIPPING_LEVEL ? '#ffb9bd' : '#8d99ab';
  ctx.fillText(
    peak >= CLIPPING_LEVEL ? 'перегрузка — убавьте усиление микрофона' : 'уровень записи',
    2,
    height - 2,
  );
}

function finishRecording() {
  const record = state.record;
  record.recorder = null;
  // Дорожку и аудиоконтекст закрываем всегда: иначе браузер продолжает считать,
  // что идёт захват микрофона, и не гасит индикатор записи.
  if (record.stream) record.stream.getTracks().forEach((track) => track.stop());
  record.stream = null;
  if (record.raf) cancelAnimationFrame(record.raf);
  record.raf = null;
  if (record.ctx) record.ctx.close();
  record.ctx = null;
  record.analyser = null;
  $('record-meter').hidden = true;
  $('btn-record').textContent = 'Записать заново';

  const blob = new Blob(record.chunks, { type: record.chunks.length ? record.chunks[0].type : '' });
  record.chunks = [];
  if (!blob.size) {
    setRecordStatus('—');
    showAlert($('record-error'), 'Запись получилась пустой — проверьте, что выбран верный микрофон.');
    return;
  }

  const file = new File([blob], `recording.${record.suffix}`, { type: blob.type });
  const player = $('record-player');
  if (record.url) URL.revokeObjectURL(record.url);
  record.url = URL.createObjectURL(file);
  player.src = record.url;
  player.hidden = false;

  // Записанный файл кладём в тот же input, что и загруженный: дальше сохранение,
  // кнопка «распознать» и все проверки работают без отдельной ветки кода.
  const transfer = new DataTransfer();
  transfer.items.add(file);
  $('new-voice-file').files = transfer.files;
  record.file = file;
  const drop = $('new-voice-drop');
  drop.textContent = `${file.name} · запись с микрофона`;
  drop.classList.add('has-file');

  // Текст прочитан с экрана и известен точно, поэтому сверку через Whisper
  // (десятки секунд) по умолчанию выключаем — включить её можно галочкой ниже.
  $('new-voice-ref-text').value = $('record-phrase-text').textContent;
  $('new-voice-verify').checked = false;

  analyzeRecording(file);
}

async function analyzeRecording(file) {
  setRecordStatus('проверяю запись…');
  const form = new FormData();
  form.append('file', file);
  form.append('gender', $('new-voice-gender').value);
  try {
    const report = await api('/api/voices/analyze', { method: 'POST', body: form });
    const tone = report.f0_hz ? ` · тон ~${Math.round(report.f0_hz)} Гц` : '';
    setRecordStatus(`записано ${report.duration_sec.toFixed(1)} с${tone}`);
    if (report.warnings.length) {
      showAlert($('record-notes'), report.warnings.join('\n\n'));
    } else {
      showAlert(
        $('record-notes'),
        `Запись в порядке: ${report.duration_sec.toFixed(1)} с${tone}. Текст взят из эталонной `
          + 'фразы, поэтому сверка через Whisper выключена — включите её, если хотите подстраховаться.',
        'info',
      );
    }
  } catch (error) {
    setRecordStatus('запись готова, но проверить её не удалось');
    showAlert($('record-notes'), error.message);
  }
}

function resetRecording() {
  const record = state.record;
  if (record.url) URL.revokeObjectURL(record.url);
  record.url = null;
  record.file = null;
  $('record-player').hidden = true;
  $('record-player').removeAttribute('src');
  $('btn-record').textContent = 'Записать';
  $('new-voice-verify').checked = true;
  setRecordStatus('до 20 секунд');
  showAlert($('record-error'), '');
  showAlert($('record-notes'), '');
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
    // Набор настроек диктует движок выбранного голоса: у XTTS вместо CFG и NFE
    // температура и штраф за повторы. Движок можно смешивать в одном диалоге —
    // каждый слот считает свой набор.
    const voice = voiceById(cfg.voice_id);
    // Слот стартует с ручек, сохранённых у голоса: иначе он отправил бы дефолты
    // движка и молча перекрыл бы настройки голоса (см. audio_pipeline._engine_params).
    if (voice && !Object.keys(cfg.engine_params).length) {
      cfg.engine_params = engineParamsFor(voice.engine, voice.engine_params);
    }
    const note = voice
      ? engineNote(voice.engine)
      : 'выберите голос — набор настроек появится под движок этого голоса';
    return `
      <div class="card" data-voice="${esc(key)}">
        <div class="voice-top">
          <span class="avatar${voice && voice.engine !== ENGINE_F5 ? ' violet' : ''}">${esc(initials(meta.label))}</span>
          <div class="voice-name">
            <h3>${esc(meta.label)}</h3>
            <p data-role="engine-note">${esc(note)}</p>
          </div>
          ${voice ? `<span class="engine-tag${voice.engine === ENGINE_F5 ? '' : ' violet'}" title="${esc(engineNote(voice.engine))}">${esc(voice.engine.toUpperCase())}</span>` : ''}
        </div>
        ${markerTag ? `<div class="tag-row">${markerTag}</div>` : ''}

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
        <div data-role="engine-params">${voice ? engineParamsHtml(voice.engine, cfg.engine_params) : ''}</div>
        <div data-role="f5-params" ${voice && voice.engine === ENGINE_F5 ? '' : 'hidden'}>
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
        </div>

        <details class="advanced">
          <summary>Ещё настройки голоса</summary>
          <label class="field">
            <span class="slider-head">Громкость <b data-role="gain-label">${fmtGain(cfg.gain_db)}</b></span>
            <input type="range" data-role="gain" min="-20" max="20" step="0.5" value="${cfg.gain_db}" />
          </label>
          <label class="field">
            <span class="slider-head">Питч-шифт <b data-role="pitch-label">${fmtPitch(cfg.pitch_semitones)}</b></span>
            <input type="range" data-role="pitch" min="-12" max="12" step="0.5" value="${cfg.pitch_semitones}" />
            <span class="muted">экспериментально: меняет высоту тона, а не пол голоса</span>
          </label>
          <label class="field">
            <span class="slider-head">Целевая громкость куска (RMS) <b data-role="rms-label">${fmtRms(cfg.target_rms)}</b></span>
            <input type="range" data-role="rms" min="0.02" max="0.3" step="0.01" value="${cfg.target_rms}" />
          </label>
          <div class="field" style="margin-bottom:0">
            <span class="slider-head">Пауза перед репликами <b data-role="pause-label">${fmtPause(cfg.pause_override_ms)}</b></span>
            <input type="range" data-role="pause-override" min="0" max="5000" step="50"
                   value="${cfg.pause_override_ms ?? 400}" ${cfg.pause_override_ms === null ? 'disabled' : ''} />
            <label class="toggle">
              <input type="checkbox" data-role="pause-inherit" ${cfg.pause_override_ms === null ? 'checked' : ''} />
              как в общих настройках
            </label>
          </div>
        </details>
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
      body: JSON.stringify({
        dialogue_text: $('dialogue').value,
        chunk_strategy: $('chunk-strategy').value,
      }),
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
    cfg.gain_db = parseFloat(card.querySelector('[data-role="gain"]').value);
    cfg.pitch_semitones = parseFloat(card.querySelector('[data-role="pitch"]').value);
    cfg.target_rms = parseFloat(card.querySelector('[data-role="rms"]').value);
    cfg.pause_override_ms = card.querySelector('[data-role="pause-inherit"]').checked
      ? null
      : parseInt(card.querySelector('[data-role="pause-override"]').value, 10);
    // Ручки движка (temperature у XTTS и т.п.) — в слот идут как переопределение
    // настроек голоса; F5-специфичные cfg/nfe остаются отдельными полями.
    cfg.engine_params = readEngineParams(card);
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
    chunk_strategy: $('chunk-strategy').value,
  };

  $('btn-generate').disabled = true;
  $('player').hidden = true;
  $('download-link').hidden = true;
  // Список реплик относится к прошлому файлу — до готовности нового он неактуален.
  state.doneJobId = null;
  stopVariant();
  $('replicas-block').hidden = true;
  $('replica-list').innerHTML = '';
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
    watchEngines();
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
      state.doneJobId = job.job_id;
      renderReplicas(job.replicas);
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

// --- перегенерация отдельной реплики ------------------------------------------
function renderReplicas(replicas) {
  const block = $('replicas-block');
  const box = $('replica-list');
  if (!replicas || !replicas.length) {
    block.hidden = true;
    box.innerHTML = '';
    return;
  }
  // Разметка строк пересобирается целиком, поэтому прослушивание варианта,
  // начатое до обновления статуса, надо остановить — иначе кнопка осталась бы
  // в состоянии «стоп» у строки, которой уже нет.
  stopVariant();
  // Текст реплики длинный — в строке он обрезается CSS, полный виден в подсказке.
  box.innerHTML = replicas.map((replica) => `
    <div class="replica-row" data-index="${replica.index}">
      <span class="replica-label">${esc(replica.label)}</span>
      <span class="replica-text" title="${esc(replica.text)}">${esc(replica.text)}</span>
      <span class="replica-seed muted">${seedText(replica.seed)}</span>
      <button class="tiny" data-role="regen">заново</button>
    </div>
    ${renderVariants(replica)}`).join('');
  block.hidden = false;
}

// Список вариантов появляется после первой перегенерации: пока звучание одно,
// список из одного пункта был бы шумом. Активный вариант — то, что в файле.
function renderVariants(replica) {
  const variants = replica.variants || [];
  if (variants.length < 2) return '';
  const rows = variants.map((variant) => `
    <div class="variant-row" data-index="${replica.index}" data-variant="${esc(variant.id)}">
      <button class="tiny" data-role="variant-play">слушать</button>
      <span class="variant-label">${esc(variant.label)}${variant.active ? ' · в файле' : ''}</span>
      <span class="muted">${variant.duration_sec} с · ${seedText(variant.seed)}</span>
      ${variant.active ? '' : '<button class="tiny" data-role="variant-pick">поставить</button>'}
    </div>`).join('');
  return `<div class="variant-list">${rows}</div>`;
}

function seedText(seed) {
  return seed === null || seed === undefined ? 'без сида' : `сид ${seed}`;
}

function setRegenBusy(busy) {
  document.querySelectorAll('#replica-list button').forEach((button) => {
    button.disabled = busy;
  });
  if (!busy) {
    document.querySelectorAll('#replica-list [data-role="regen"]').forEach((button) => {
      button.textContent = 'заново';
    });
  }
}

async function regenerateReplica(index, button) {
  const jobId = state.doneJobId;
  if (!jobId || state.regenTimer) return;
  showAlert($('job-error'), '');
  setRegenBusy(true);
  button.textContent = 'генерирую…';
  try {
    await api(`/api/jobs/${jobId}/replicas/${index}/regenerate`, { method: 'POST' });
  } catch (error) {
    setRegenBusy(false);
    showAlert($('job-error'), error.message);
    return;
  }
  startRegenPoll(jobId, index);
}

async function selectVariant(index, variantId, button) {
  const jobId = state.doneJobId;
  if (!jobId || state.regenTimer) return;
  showAlert($('job-error'), '');
  setRegenBusy(true);
  button.textContent = 'ставлю…';
  try {
    await api(`/api/jobs/${jobId}/replicas/${index}/variants/${variantId}`, { method: 'POST' });
  } catch (error) {
    setRegenBusy(false);
    showAlert($('job-error'), error.message);
    return;
  }
  startRegenPoll(jobId, index);
}

// Опрос один на перегенерацию и на выбор варианта: для интерфейса это одно
// состояние — «реплика занята, файл скоро обновится».
function startRegenPoll(jobId, index) {
  state.regenIndex = index;
  state.regenTimer = setInterval(() => pollRegenerate(jobId), 1500);
}

async function pollRegenerate(jobId) {
  try {
    const job = await api(`/api/jobs/${jobId}`);
    // Бэкенд снимает флаг в самом конце — и при успехе, и при ошибке.
    if (job.regenerating_replica !== null) return;
    clearInterval(state.regenTimer);
    state.regenTimer = null;
    setRegenBusy(false);
    if (job.regen_error) {
      showAlert($('job-error'), `Не удалось пересобрать реплику: ${job.regen_error}`);
      return;
    }
    $('job-status').textContent = `готово · ${job.duration_sec} с аудио`;
    // Варианты и сид изменились вместе с файлом — список реплик берём из статуса,
    // а не правим на месте: так он не разойдётся с тем, что лежит на диске.
    if (job.replicas) renderReplicas(job.replicas);
    // Тот же URL, но содержимое файла новое — без метки времени браузер отдал бы кэш.
    $('player').src = `${job.audio_url}?t=${Date.now()}`;
  } catch (error) {
    clearInterval(state.regenTimer);
    state.regenTimer = null;
    setRegenBusy(false);
    showAlert($('job-error'), error.message);
  }
}

// --- прослушивание вариантов реплики ------------------------------------------
function variantKey(index, variantId) {
  return `${index}:${variantId}`;
}

function toggleVariant(index, variantId) {
  const player = $('variant-player');
  const key = variantKey(index, variantId);
  if (state.variantKey === key && !player.paused) {
    player.pause();
    return;
  }
  state.variantKey = key;
  player.src = `/api/jobs/${state.doneJobId}/replicas/${index}/variants/${variantId}/audio`;
  // play() отказывается промисом, а не исключением: молча оставить это нельзя —
  // кнопка показывала бы «стоп» у варианта, который не звучит.
  player.play().catch((error) => {
    state.variantKey = null;
    markVariantPlaying(false);
    showAlert($('job-error'), `Не удалось воспроизвести вариант: ${error.message}`);
  });
}

function markVariantPlaying(playing) {
  document.querySelectorAll('#replica-list [data-role="variant-play"]').forEach((button) => {
    const row = button.closest('.variant-row');
    const key = variantKey(row.dataset.index, row.dataset.variant);
    button.textContent = playing && key === state.variantKey ? 'стоп' : 'слушать';
  });
}

function stopVariant() {
  const player = $('variant-player');
  player.pause();
  player.removeAttribute('src');
  state.variantKey = null;
}

// --- сплошной текст (один голос на весь файл) ---------------------------------
function renderTextVoiceOptions() {
  const select = $('text-voice');
  const current = select.value;
  select.innerHTML = '<option value="">— выберите голос —</option>' +
    state.voices.map((v) => `<option value="${esc(v.id)}">${esc(v.name)}</option>`).join('');
  const fallback = state.voices.length ? state.voices[0].id : '';
  select.value = state.voices.some((v) => v.id === current) ? current : fallback;
  renderTextEngineParams();
  updateTextButton();
}

// Набор настроек здесь тоже диктует движок голоса: CFG и NFE есть только у F5,
// у XTTS — температура и штраф за повторы.
function renderTextEngineParams() {
  const voice = voiceById($('text-voice').value);
  const box = $('text-engine-params');
  if (!voice) {
    box.innerHTML = '';
    $('text-engine-note').textContent = '';
    $('text-f5-params').hidden = true;
    state.textEngineVoice = '';
    state.textEngineParams = {};
    return;
  }
  // Правки в блоке относятся к конкретному голосу: при смене голоса берём
  // сохранённые ручки нового, а не значения предыдущего. Перерисовка блока по
  // другой причине (обновился список голосов) значения не сбрасывает.
  const sameVoice = state.textEngineVoice === voice.id;
  state.textEngineVoice = voice.id;
  state.textEngineParams = engineParamsFor(
    voice.engine,
    sameVoice ? readEngineParams(box) : voice.engine_params,
  );
  box.innerHTML = engineParamsHtml(voice.engine, state.textEngineParams);
  $('text-engine-note').textContent = engineNote(voice.engine);
  $('text-f5-params').hidden = voice.engine !== ENGINE_F5;
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
    engine_params: readEngineParams($('text-engine-params')),
    gain_db: parseFloat($('text-gain').value),
    pitch_semitones: parseFloat($('text-pitch').value),
    target_rms: parseFloat($('text-rms').value),
    pause_ms: parseInt($('text-pause').value, 10),
    cross_fade_duration: parseFloat($('text-crossfade').value),
    auto_accent: $('text-auto-accent').checked,
    output_format: $('text-output-format').value,
    chunk_strategy: $('text-chunk-strategy').value,
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
    watchEngines();
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
// Состояние движков видно в шапке: XTTS поднимается 20–40 секунд при первом
// обращении к ней, и без этого в интерфейсе не отличить «ещё грузится» от «упало».
const ENGINE_STATES = {
  idle: 'не загружен',
  loading: 'загружается',
  ready: 'готов',
  failed: 'ОШИБКА',
};

// Опрос статуса прекращается, когда F5 и RUAccent устоялись, — но движок,
// выбранный у голоса, поднимается позже, уже во время генерации. Поэтому перед
// запуском задачи опрос включаем снова.
function watchEngines() {
  if (!state.statusTimer) state.statusTimer = setInterval(refreshStatus, 4000);
  refreshStatus();
}

async function refreshStatus() {
  const dot = $('status-dot');
  const text = $('status-text');
  try {
    const status = await api('/api/status');
    state.modelReady = status.model_loaded;
    const engines = status.engines || [];
    const failedEngine = engines.find((e) => e.state === 'failed');
    const loadingEngines = engines.filter((e) => e.state === 'loading');

    // Устройство известно только после загрузки модели: не показываем «на null».
    const model = status.model_loaded
      ? `модель готова · ${status.device || 'устройство неизвестно'}`
      : `загружаю модель${status.device ? ` на ${status.device}` : ''}…`;
    const accent = {
      ready: 'вкл',
      failed: 'ОШИБКА',
      loading: 'загружается',
      idle: 'загружается',
    }[status.accentizer_state] || status.accentizer_state;
    const ready = status.model_loaded && !loadingEngines.length;

    // Чипы в шапке показывают то же состояние, что раньше умещалось в одну строку,
    // но по отдельным полям: подробности (модель, ошибки движка и ударений) уходят
    // в подсказки, чтобы шапка не разрасталась.
    text.textContent = failedEngine ? 'ошибка' : ready ? 'готов' : 'загрузка';
    $('status-device').textContent = status.device || '—';
    $('status-engines').textContent = engines.length
      ? engines.map((e) => `${e.id.toUpperCase()} ${ENGINE_STATES[e.state] || e.state}`).join(' · ')
      : 'не поднят';
    $('status-accent').textContent = accent;
    // Очистка референса — опциональная зависимость (DeepFilterNet). Нет её — гасим
    // тумблер, а не роняем сохранение голоса ошибкой уже после нажатия кнопки.
    const denoiseReady = status.denoise_available !== false;
    $('new-voice-denoise').disabled = !denoiseReady;
    $('new-voice-denoise-block').title = denoiseReady
      ? ''
      : 'Не установлен DeepFilterNet: ./venv/bin/pip install --no-deps -r requirements-denoise.txt';
    if (!denoiseReady) $('new-voice-denoise').checked = false;
    if (typeof status.rss_mb === 'number') {
      $('status-mem').textContent = `${Math.round(status.rss_mb)} МБ`;
      // Полоса показывает занятость памяти всей системы, а число рядом — RSS этого
      // процесса: вместе видно и «сколько съели мы», и «есть ли ещё запас».
      $('status-mem-bar').style.width = `${Math.min(100, status.system_mem_percent || 0)}%`;
      $('status-mem-bar').closest('.memory').title =
        `RSS процесса: ${status.rss_mb} МБ · системная память занята на ${status.system_mem_percent ?? '?'}%`;
    }

    if (failedEngine) {
      dot.className = 'online-dot err';
      text.title = `Движок «${failedEngine.label}» не поднялся: ${failedEngine.error || 'причина неизвестна'}`;
    } else if (status.accentizer_state === 'failed') {
      dot.className = 'online-dot err';
      text.title = status.accentizer_error
        ? `RUAccent не поднялся, синтез идёт без ударений: ${status.accentizer_error}`
        : 'RUAccent не поднялся, синтез идёт без ударений';
    } else {
      dot.className = `online-dot${ready ? '' : ' busy'}`;
      text.title = `${model} · ударения: ${accent}`;
    }

    // Прекращаем опрос, только когда устоялось и то, и другое: иначе статус
    // ударений замирал на «загружаются» — модель-то уже готова.
    const accentSettled = status.accentizer_state === 'ready' || status.accentizer_state === 'failed';
    if (status.model_loaded && accentSettled && !loadingEngines.length) {
      clearInterval(state.statusTimer);
      state.statusTimer = null;
    }
  } catch (error) {
    dot.className = 'online-dot err';
    text.textContent = 'нет связи';
    text.title = '';
    $('status-device').textContent = '—';
    $('status-engines').textContent = '—';
    $('status-accent').textContent = '—';
    $('status-mem').textContent = '—';
    $('status-mem-bar').style.width = '0';
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
  $('replica-list').addEventListener('click', (event) => {
    const row = event.target.closest('.replica-row');
    const regen = event.target.closest('button[data-role="regen"]');
    if (regen && row) {
      regenerateReplica(parseInt(row.dataset.index, 10), regen);
      return;
    }
    const variant = event.target.closest('.variant-row');
    if (!variant) return;
    const index = parseInt(variant.dataset.index, 10);
    const play = event.target.closest('button[data-role="variant-play"]');
    if (play) {
      toggleVariant(index, variant.dataset.variant);
      return;
    }
    const pick = event.target.closest('button[data-role="variant-pick"]');
    if (pick) selectVariant(index, variant.dataset.variant, pick);
  });
  const variantPlayer = $('variant-player');
  variantPlayer.addEventListener('play', () => markVariantPlaying(true));
  variantPlayer.addEventListener('pause', () => markVariantPlaying(false));
  variantPlayer.addEventListener('ended', () => markVariantPlaying(false));
  // Основной трек и отдельный вариант одновременно звучать не должны.
  $('player').addEventListener('play', stopVariant);
  $('btn-reload-voices').addEventListener('click', () => loadVoices().catch((e) => alert(e.message)));
  $('btn-save-voice').addEventListener('click', createVoice);
  $('btn-recognize-voice').addEventListener('click', recognizeRefText);

  $('pause').addEventListener('input', (e) => { $('pause-value').textContent = `${e.target.value} мс`; });
  // Смена стратегии меняет число реплик — сводка должна обновиться сама.
  $('chunk-strategy').addEventListener('change', () => { syncVoices({ silent: true }); });
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
  $('text-voice').addEventListener('change', () => { renderTextEngineParams(); updateTextButton(); });
  $('text-engine-params').addEventListener('input', (event) => {
    if (event.target.dataset.engineParam) updateEngineParamLabel($('text-engine-params'), event.target);
  });
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
  $('text-gain').addEventListener('input', (e) => {
    $('text-gain-label').textContent = fmtGain(parseFloat(e.target.value));
  });
  $('text-pitch').addEventListener('input', (e) => {
    $('text-pitch-label').textContent = fmtPitch(parseFloat(e.target.value));
  });
  $('text-rms').addEventListener('input', (e) => {
    $('text-rms-label').textContent = fmtRms(parseFloat(e.target.value));
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
    // У загруженного файла расшифровку проверить больше нечем — возвращаем сверку.
    $('new-voice-verify').checked = true;
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
    $('new-voice-verify').checked = true;
  });

  // --- запись голоса с микрофона ---
  $('record-phrase').addEventListener('change', showRecordPhrase);
  $('btn-record').addEventListener('click', () => {
    // Одна кнопка на весь цикл: записать → стоп → записать заново.
    if (state.record.recorder) stopRecording();
    else startRecording();
  });
  // Пол проверяется по измеренному тону, поэтому смена пола пересчитывает проверку:
  // иначе на карточке предупреждение будет, а до сохранения его не покажут.
  $('new-voice-gender').addEventListener('change', () => {
    if (state.record.file) analyzeRecording(state.record.file);
    // Движок подсказывается полом, пока пользователь не выбрал его сам.
    if (!state.engineTouched) {
      $('new-voice-engine').value = suggestedEngine($('new-voice-gender').value);
      updateNewVoiceEngineNote();
    }
  });
  $('new-voice-engine').addEventListener('change', () => {
    state.engineTouched = true;
    updateNewVoiceEngineNote();
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
    // Ручки движка проверяются прослушиванием, поэтому на движении ползунка их
    // только запоминаем: запись на бэкенд происходит по отпусканию (change).
    if (event.target.dataset.engineParam) {
      updateEngineParamLabel(card, event.target);
      cfg.engine_params = readEngineParams(card);
    }
  });

  voiceCards.addEventListener('change', (event) => {
    const card = event.target.closest('.card');
    if (!card) return;
    const voiceId = card.dataset.voiceId;
    const role = event.target.dataset.role;
    if (role === 'nfe') {
      state.preview[voiceId].nfe_step = parseInt(event.target.value, 10);
    }
    if (role === 'engine') {
      changeVoiceEngine(card, event.target.value);
    }
    if (event.target.dataset.engineParam) {
      // Значения ручек сохраняются у голоса — это его настройки по умолчанию,
      // карточка слота в диалоге может переопределить их на одну генерацию.
      const params = readEngineParams(card);
      state.preview[voiceId].engine_params = params;
      saveVoiceEngineParams(voiceId, params);
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
    const value = parseFloat(event.target.value);
    if (role === 'speed') card.querySelector('[data-role="speed-label"]').textContent = `${value.toFixed(2)}x`;
    if (role === 'cfg') card.querySelector('[data-role="cfg-label"]').textContent = value.toFixed(1);
    if (role === 'gain') card.querySelector('[data-role="gain-label"]').textContent = fmtGain(value);
    if (role === 'pitch') card.querySelector('[data-role="pitch-label"]').textContent = fmtPitch(value);
    if (role === 'rms') card.querySelector('[data-role="rms-label"]').textContent = fmtRms(value);
    if (role === 'pause-override') {
      card.querySelector('[data-role="pause-label"]').textContent = fmtPause(parseInt(event.target.value, 10));
    }
    if (event.target.dataset.engineParam) updateEngineParamLabel(card, event.target);
    readCardConfigs();
    updateGenerateButton();
  });

  voiceConfigs.addEventListener('change', (event) => {
    const role = event.target.dataset.role;
    if (role === 'pause-inherit') {
      const card = event.target.closest('.card');
      const range = card.querySelector('[data-role="pause-override"]');
      range.disabled = event.target.checked;
      card.querySelector('[data-role="pause-label"]').textContent = event.target.checked
        ? fmtPause(null)
        : fmtPause(parseInt(range.value, 10));
      readCardConfigs();
      return;
    }
    if (role === 'voice') {
      readCardConfigs();
      // Набор ручек диктует движок голоса: значения прошлого движка к новому не
      // относятся, поэтому блок пересобирается с сохранёнными ручками нового голоса.
      state.configs[event.target.closest('.card').dataset.voice].engine_params = {};
      renderVoiceConfigs();
      updateGenerateButton();
    }
  });
}

// --- старт --------------------------------------------------------------------
async function init() {
  renderRecordPhrases();
  bindEvents();
  // Таймер создаём до первого опроса: если всё уже готово, refreshStatus его снимет.
  state.statusTimer = setInterval(refreshStatus, 4000);
  await refreshStatus();
  try {
    // Паспорта движков — до голосов: и карточки, и форма рисуют по ним набор настроек.
    await loadEngines();
    await loadVoices();
  } catch (error) {
    showAlert($('voice-error'), error.message);
  }
}

init();
