'use strict';

const $ = (id) => document.getElementById(id);

const state = {
  voices: [],
  engines: {},         // id -> паспорт движка из /api/engines
  engineTouched: false, // пользователь сам выбрал движок в форме нового голоса
  project: null,       // проект диалога: исходный текст, спикеры, реплики и варианты
  sourceDirty: false,  // исходный текст правили после последнего разбора
  openDetails: {},     // индекс реплики -> раскрыт ли блок «ещё настройки»
  speakerDetails: {},  // ключ спикера -> раскрыт ли блок «ещё настройки голоса»
  replicaBusy: null,   // индекс реплики, которая синтезируется прямо сейчас
  replicaJobId: null,  // задача этого пересинтеза — по ней работает отмена
  replicaTimer: null,
  takeKey: null,       // "индекс:take_id" варианта, играющего в плеере реплик
  timeline: null,      // последний ответ /timeline: дорожки спикеров и сегменты реплик
  timelineBusy: false, // запрос таймлайна в полёте — кнопка «обновить» заблокирована
  timelineError: null, // текст ошибки таймлайна (показывается под дорожками)
  preview: {},         // voice_id -> {speed, cfg_strength, nfe_step, engine_params, text}
  dictionary: [],      // правила словаря произношения: [{id, source, target, ...}]
  dictionaryLoaded: false, // словарь уже запрашивался (чтобы не мигать «загружаю»)
  dictionaryEditing: null, // id правила, которое правится в форме (null — новое)
  suggestions: [],     // кандидаты в словарь для текста вкладки 03
  replicaPreview: {},  // индекс реплики -> {loading} | {data} | {error} для «что услышит модель»
  benchmark: {},       // voice_id -> состояние панели «сравнить движки»
  models: null,        // последний ответ /api/models: {models, disk}
  modelsTimer: null,   // опрос прогресса скачивания; null — ни одна модель не качается
  textEngineParams: {}, // ручки движка в режиме «Сплошной текст»
  textEngineVoice: '',  // голос, к которому относятся эти ручки
  modelReady: false,
  exportBusy: false,   // идёт выгрузка/загрузка файла: кнопка экспорта заблокирована
  jobId: null,
  pollTimer: null,
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
// Фраза для сравнения движков: достаточно длинная, чтобы услышать тембр и
// интонацию, и с разными типами звуков — на односложном «привет» F5 и XTTS
// почти неразличимы, а сравнивают именно звучание.
const BENCHMARK_TEXT = 'Сегодня хорошая погода, и мы наконец можем спокойно обсудить одно небольшое дело.';
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
  renderDictionaryEngineOptions();
}

// Список движков для проверки словаря: от движка зависит только одно — уйдут ли
// ему знаки «+». Синтеза здесь нет, поэтому подпись честно говорит об этом.
function renderDictionaryEngineOptions() {
  const select = $('dict-preview-engine');
  if (!select) return;
  const current = select.value;
  select.innerHTML = '<option value="">как для F5 — с ударениями</option>'
    + Object.values(state.engines).map((info) => {
      const suffix = info.supports_accents ? 'ударения есть' : 'без ударений';
      return `<option value="${esc(info.id)}">${esc(info.label)} — ${suffix}</option>`;
    }).join('');
  select.value = current;
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
  $('tab-dictionary').hidden = name !== 'dictionary';
  $('tab-models').hidden = name !== 'models';
  // Словарь грузится при первом открытии вкладки: он не нужен для озвучки, и
  // запрашивать его на каждом старте дашборда незачем.
  if (name === 'dictionary') {
    loadDictionary().catch((error) => showAlert($('dictionary-error'), error.message));
  }
  // Модели — так же: список весов нужен только тому, кто пришёл его смотреть,
  // а обход каталогов с гигабайтными файлами на старте дашборда незачем.
  if (name === 'models') {
    loadModels().catch((error) => showAlert($('models-error'), error.message));
  } else {
    stopModelsPolling();
  }
}

// --- менеджер моделей ---------------------------------------------------------
// Вкладка показывает состояние весов на диске и, если чего-то не хватает, умеет
// их скачать. Скачивание идёт мимо очереди синтеза (это сеть и диск), но удаление
// запрещено, пока движок модели занят — отказ приходит из бэкенда текстом 409.
const MODELS_POLL_MS = 1500;

function fmtBytes(bytes) {
  const value = Number(bytes);
  if (!Number.isFinite(value) || value <= 0) return '—';
  const units = ['Б', 'КБ', 'МБ', 'ГБ', 'ТБ'];
  let size = value;
  let unit = 0;
  while (size >= 1024 && unit < units.length - 1) { size /= 1024; unit += 1; }
  return `${size >= 100 || unit === 0 ? Math.round(size) : size.toFixed(1)} ${units[unit]}`;
}

const DOWNLOAD_LABELS = {
  idle: '',
  downloading: 'скачивание',
  done: 'скачано',
  error: 'ошибка скачивания',
  interrupted: 'скачивание прервано',
};

// Скачивается ли хоть что-то прямо сейчас. По этому признаку живёт опрос:
// закончились загрузки — таймер снимается, и вкладка перестаёт дёргать бэкенд.
const isDownloading = (model) => model.download && model.download.state === 'downloading';

function modelStatusTags(model) {
  const tags = [];
  tags.push(model.installed
    ? '<span class="tag ok">установлена</span>'
    : '<span class="tag warn">не установлена</span>');
  if (model.loaded) tags.push('<span class="tag">в памяти</span>');
  // `idle` у уже созданного движка означает именно выгрузку: движок либо поднят
  // синтезом, либо отпущен кнопкой/по простою. Никогда не созданный движок даёт
  // `engine_state: null` — его выгруженным называть нечестно.
  else if (model.engine_state === 'idle') tags.push('<span class="tag">выгружен</span>');
  if (model.engine_in_use) tags.push('<span class="tag warn">движок занят</span>');
  if (model.kind === 'cache') tags.push('<span class="tag">кеш HF</span>');
  if (model.download && model.download.state !== 'idle') {
    tags.push(`<span class="tag">${esc(DOWNLOAD_LABELS[model.download.state] || model.download.state)}</span>`);
  }
  return tags.join('');
}

function modelActions(model) {
  const buttons = [];
  const downloading = isDownloading(model);
  const canDownload = model.kind === 'local';
  if (canDownload && !model.installed) {
    buttons.push(`<button class="tiny primary" data-model-action="download" data-model-id="${esc(model.id)}"`
      + `${downloading ? ' disabled' : ''}>${downloading ? 'Скачивается…' : 'Скачать'}</button>`);
  }
  // Выгрузка/загрузка — только у моделей с движком синтеза (у Whisper его нет) и
  // только когда файлы на месте. Занятый движок кнопку не гасит: отказ 409 с
  // понятным текстом честнее, чем молча неактивная кнопка.
  if (model.engine_id && model.installed && !downloading) {
    if (model.loaded) {
      buttons.push(`<button class="tiny ghost" data-model-action="unload" data-model-id="${esc(model.id)}"`
        + '>Выгрузить</button>');
    } else {
      buttons.push(`<button class="tiny" data-model-action="load" data-model-id="${esc(model.id)}"`
        + '>Загрузить</button>');
    }
  }
  if (canDownload && model.installed && !downloading) {
    buttons.push(`<button class="tiny ghost danger" data-model-action="delete" data-model-id="${esc(model.id)}"`
      + '>Удалить</button>');
  }
  return buttons.join('');
}

function renderModels() {
  const container = $('models-list');
  const data = state.models;
  if (!data) return;
  if (!data.models.length) {
    container.innerHTML = '<p class="muted">Модели не объявлены.</p>';
    return;
  }
  container.innerHTML = data.models.map((model) => {
    const download = model.download || {};
    const showProgress = download.state === 'downloading';
    const missing = model.missing_files && model.missing_files.length
      ? `<p class="muted warn">не хватает: ${esc(model.missing_files.join(', '))}</p>` : '';
    const error = download.error
      ? `<div class="alert error show">${esc(download.error)}</div>` : '';
    const progress = showProgress
      ? `<div class="progress"><div style="width:${Math.round((download.progress || 0) * 100)}%"></div></div>
         <span class="muted job-line">${fmtBytes(download.bytes_downloaded)} из ${fmtBytes(download.bytes_total)}`
         + ` · ${Math.round((download.progress || 0) * 100)}%</span>`
      : '';
    return `
      <div class="model-card" data-model-id="${esc(model.id)}">
        <div class="model-top">
          <div class="voice-name">
            <h3>${esc(model.label)}</h3>
            <p>${esc(model.description)}</p>
          </div>
          <b class="muted">${fmtBytes(model.size_bytes)}</b>
        </div>
        <div class="tag-row">${modelStatusTags(model)}</div>
        ${missing}
        ${progress}
        ${error}
        <p class="model-path" title="${esc(model.path)}">${esc(model.path)}</p>
        <div class="row">
          ${modelActions(model)}
          <span class="muted model-hint">${esc(model.repo_id)} · ожидается ~${fmtBytes(model.approx_size_bytes)}</span>
        </div>
      </div>`;
  }).join('');
  const disk = data.disk || {};
  $('models-disk-used').textContent = fmtBytes(disk.models_bytes);
  $('models-disk-free').textContent = fmtBytes(disk.free_bytes);
}

async function loadModels() {
  state.models = await api('/api/models');
  showAlert($('models-error'), '');
  renderModels();
  // Опрос живёт ровно столько, сколько идёт скачивание: после завершения
  // (успех, ошибка или прерывание) он снимается сам.
  if (state.models.models.some(isDownloading)) startModelsPolling();
  else stopModelsPolling();
}

function startModelsPolling() {
  if (state.modelsTimer) return;
  state.modelsTimer = setInterval(() => {
    loadModels().catch((error) => {
      stopModelsPolling();
      showAlert($('models-error'), error.message);
    });
  }, MODELS_POLL_MS);
}

function stopModelsPolling() {
  if (!state.modelsTimer) return;
  clearInterval(state.modelsTimer);
  state.modelsTimer = null;
}

async function downloadModel(modelId) {
  const model = (state.models?.models || []).find((item) => item.id === modelId);
  try {
    await api(`/api/models/${encodeURIComponent(modelId)}/download`, { method: 'POST' });
  } catch (error) {
    showAlert($('models-error'), error.message);
  }
  await loadModels().catch((error) => showAlert($('models-error'), error.message));
  if (model && model.kind === 'local') {
    showAlert(
      $('models-note'),
      `Скачивание «${model.label}» идёт в фоне: страницу можно закрыть, прогресс виден здесь.`,
      'info',
    );
  }
}

async function deleteModel(modelId) {
  const model = (state.models?.models || []).find((item) => item.id === modelId);
  if (!model) return;
  const ok = window.confirm(
    `Удалить файлы модели «${model.label}» (${fmtBytes(model.size_bytes)})?\n`
    + 'Веса придётся скачивать заново.',
  );
  if (!ok) return;
  try {
    await api(`/api/models/${encodeURIComponent(modelId)}`, { method: 'DELETE' });
    showAlert($('models-note'), `Файлы модели «${model.label}» удалены.`, 'info');
  } catch (error) {
    // 409 — движок занят: показываем текст бэкенда как есть, он объясняет, что делать.
    showAlert($('models-error'), error.message);
  }
  await loadModels().catch((error) => showAlert($('models-error'), error.message));
}

// Выгрузка и загрузка движка — возврат памяти без перезапуска приложения.
// Отказ 409 приходит понятным текстом (движок синтезирует или очередь занята) и
// показывается как есть; после любой попытки список перечитывается, чтобы вкладка
// не показывала устаревшее «в памяти».
async function unloadModelEngine(modelId) {
  const model = (state.models?.models || []).find((item) => item.id === modelId);
  showAlert($('models-note'), 'Выгружаю движок и возвращаю память…', 'info');
  try {
    const result = await api(`/api/engines/${encodeURIComponent(modelId)}/unload`, { method: 'POST' });
    showAlert($('models-note'), result.message || `Движок «${model?.label || modelId}» выгружен.`, 'info');
  } catch (error) {
    showAlert($('models-error'), error.message);
  }
  await loadModels().catch((error) => showAlert($('models-error'), error.message));
}

async function loadModelEngine(modelId) {
  const model = (state.models?.models || []).find((item) => item.id === modelId);
  showAlert(
    $('models-note'),
    `Поднимаю «${model?.label || modelId}»: это может занять до минуты, страница остаётся отзывчивой.`,
    'info',
  );
  try {
    const result = await api(`/api/engines/${encodeURIComponent(modelId)}/load`, { method: 'POST' });
    showAlert($('models-note'), result.message || `Движок «${model?.label || modelId}» поднят.`, 'info');
  } catch (error) {
    showAlert($('models-error'), error.message);
  }
  await loadModels().catch((error) => showAlert($('models-error'), error.message));
}

function bindModelsEvents() {
  const reload = $('btn-reload-models');
  if (reload) reload.addEventListener('click', () => {
    loadModels().catch((error) => showAlert($('models-error'), error.message));
  });
  const list = $('models-list');
  if (!list) return;
  list.addEventListener('click', (event) => {
    const button = event.target.closest('[data-model-action]');
    if (!button) return;
    const modelId = button.dataset.modelId;
    const action = button.dataset.modelAction;
    if (action === 'download') {
      button.disabled = true;
      downloadModel(modelId).catch((error) => showAlert($('models-error'), error.message));
    } else if (action === 'delete') {
      deleteModel(modelId).catch((error) => showAlert($('models-error'), error.message));
    } else if (action === 'unload' || action === 'load') {
      // Блокируем кнопку до перерисовки списка: так видно, что запрос в работе.
      button.disabled = true;
      const task = action === 'unload' ? unloadModelEngine(modelId) : loadModelEngine(modelId);
      task.catch((error) => showAlert($('models-error'), error.message));
    }
  });
}

// --- словарь произношения -----------------------------------------------------
// Словарь глобальный: правило, добавленное здесь, применяется во всех проектах.
// Проверка текста идёт тем же эндпоинтом, что и стадии синтеза, поэтому фронт не
// повторяет порядок шагов и не может показать одно, а отправить в модель другое.

async function loadDictionary() {
  if (!state.dictionaryLoaded) {
    $('dictionary-list').innerHTML = '<div class="muted" style="padding:6px 0">Загружаю словарь…</div>';
  }
  const data = await api('/api/pronunciation');
  state.dictionary = data.entries;
  state.dictionaryLoaded = true;
  renderDictionary();
}

function renderDictionary() {
  const box = $('dictionary-list');
  if (!state.dictionary.length) {
    box.innerHTML = '<div class="muted" style="padding:6px 0">Словарь пуст. Добавьте первое правило '
      + 'справа — например <b>OpenAI → оупен эй-ай</b>. Оно сразу начнёт работать во всех проектах.</div>';
    return;
  }
  box.innerHTML = state.dictionary.map((entry) => `
    <div class="card dict-rule${entry.enabled ? '' : ' off'}" data-entry-id="${entry.id}">
      <div class="row between">
        <div class="dict-pair">
          <b>${esc(entry.source)}</b><span class="dict-arrow">→</span><span>${esc(entry.target)}</span>
        </div>
        <div class="row">
          <button class="tiny ghost" data-role="edit">правка</button>
          <button class="tiny ghost danger" data-role="delete">удалить</button>
        </div>
      </div>
      <div class="tag-row" style="margin:10px 0 0">
        <span class="tag">${entry.case_sensitive ? 'регистр' : 'без регистра'}</span>
        <span class="tag${entry.whole_word ? '' : ' warn'}">${entry.whole_word ? 'целое слово' : 'подстрока'}</span>
        ${entry.note ? `<span class="tag">${esc(entry.note)}</span>` : ''}
      </div>
      <div class="grid-2" style="margin-top:10px">
        <label class="toggle" style="margin:0">
          <input type="checkbox" data-role="enabled" ${entry.enabled ? 'checked' : ''} /> включено
        </label>
        <label class="toggle" style="margin:0">
          <input type="checkbox" data-role="case" ${entry.case_sensitive ? 'checked' : ''} /> учитывать регистр
        </label>
      </div>
    </div>`).join('');
}

function dictionaryEntry(id) {
  return state.dictionary.find((entry) => entry.id === id) || null;
}

function editDictionaryEntry(id) {
  const entry = dictionaryEntry(id);
  if (!entry) return;
  state.dictionaryEditing = id;
  $('dict-form-title').textContent = 'ПРАВКА ПРАВИЛА';
  $('dict-source').value = entry.source;
  $('dict-target').value = entry.target;
  $('dict-note').value = entry.note || '';
  $('dict-whole-word').checked = entry.whole_word;
  $('dict-case-sensitive').checked = entry.case_sensitive;
  $('dict-enabled').checked = entry.enabled;
  $('btn-reset-dict-form').hidden = false;
  showAlert($('dictionary-error'), '');
}

function resetDictionaryForm() {
  state.dictionaryEditing = null;
  $('dict-form-title').textContent = 'НОВОЕ ПРАВИЛО';
  $('dict-source').value = '';
  $('dict-target').value = '';
  $('dict-note').value = '';
  $('dict-whole-word').checked = true;
  $('dict-case-sensitive').checked = false;
  $('dict-enabled').checked = true;
  $('btn-reset-dict-form').hidden = true;
}

async function saveDictionaryEntry() {
  const source = $('dict-source').value.trim();
  const target = $('dict-target').value.trim();
  showAlert($('dictionary-error'), '');
  if (!source || !target) {
    showAlert($('dictionary-error'), 'Заполните источник и замену — пустое правило ничего не меняет');
    return;
  }
  const payload = {
    source,
    target,
    note: $('dict-note').value.trim(),
    whole_word: $('dict-whole-word').checked,
    case_sensitive: $('dict-case-sensitive').checked,
    enabled: $('dict-enabled').checked,
  };
  // Правка отправляется целиком, как и создание: правило small, и частичный
  // PATCH здесь только запутал бы — что именно ушло, видно в форме.
  const editing = state.dictionaryEditing;
  const button = $('btn-save-dict-entry');
  button.disabled = true;
  try {
    await api(editing === null ? '/api/pronunciation' : `/api/pronunciation/${editing}`, {
      method: editing === null ? 'POST' : 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    resetDictionaryForm();
    await loadDictionary();
  } catch (error) {
    showAlert($('dictionary-error'), error.message);
  } finally {
    button.disabled = false;
  }
}

async function updateDictionaryEntry(id, patch) {
  showAlert($('dictionary-error'), '');
  try {
    await api(`/api/pronunciation/${id}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(patch),
    });
  } catch (error) {
    showAlert($('dictionary-error'), error.message);
  }
  // Список перечитывается и при ошибке: переключатель не должен остаться в
  // состоянии, которого нет в базе.
  await loadDictionary();
}

async function deleteDictionaryEntry(id) {
  const entry = dictionaryEntry(id);
  if (!entry) return;
  if (!confirm(`Удалить правило «${entry.source} → ${entry.target}»?`)) return;
  showAlert($('dictionary-error'), '');
  try {
    await api(`/api/pronunciation/${id}`, { method: 'DELETE' });
    if (state.dictionaryEditing === id) resetDictionaryForm();
    await loadDictionary();
  } catch (error) {
    showAlert($('dictionary-error'), error.message);
  }
}

async function previewDictionary() {
  const text = $('dict-preview-text').value.trim();
  showAlert($('dictionary-error'), '');
  if (!text) {
    showAlert($('dictionary-error'), 'Введите текст для проверки');
    return;
  }
  const status = $('dict-preview-status');
  status.textContent = 'проверяю…';
  try {
    const data = await api('/api/pronunciation/preview', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, engine: $('dict-preview-engine').value || null }),
    });
    $('dict-preview-normalized').textContent = data.normalized;
    $('dict-preview-final').textContent = data.result;
    $('dict-preview-matches').innerHTML = data.matches.length
      ? '<p class="eyebrow" style="margin:16px 0 6px">СРАБОТАЛИ ПРАВИЛА</p>'
        + `<div class="tag-row">${data.matches.map((match) => (
          `<span class="tag">${esc(match.source)} → ${esc(match.target)} ×${match.count}</span>`
        )).join('')}</div>`
      : '<p class="hint muted">Ни одно правило не сработало — текст уйдёт как есть.</p>';
    $('dict-preview-result').hidden = false;
  } catch (error) {
    $('dict-preview-result').hidden = true;
    showAlert($('dictionary-error'), error.message);
  } finally {
    status.textContent = '';
  }
}

// --- «Что услышит модель» -----------------------------------------------------
// Один компонент на две вкладки: панель — это разметка плюс функции, а не два
// набора кода. Проверку делает бэкенд ровно тем же путём, каким готовит текст к
// синтезу (/api/text/preview), поэтому фронт ничего не пересчитывает и не может
// показать одно, а отправить в модель другое. Синтез здесь не запускается и ни
// одна модель TTS не поднимается — только читается текст.
//
// Тумблер ударений начинается со значения панели настроек той же вкладки: preview
// обязан показывать то, что произойдёт при текущих настройках, а не собственный
// дефолт, который разошёлся бы с рендером.
const PREVIEW_MOUNTS = [
  { id: 'dialogue-preview', accent: 'auto-accent', fromTextarea: null },
  { id: 'text-preview', accent: 'text-auto-accent', fromTextarea: 'text-body' },
];

function previewPanelHtml(config) {
  return `
    <label class="field">
      <span>Текст</span>
      <textarea data-role="text" spellcheck="false" style="min-height:90px"
        placeholder="Вставьте фразу — покажу, что из неё услышит модель."></textarea>
    </label>
    <div class="grid-2">
      <label class="field">
        <span>Голос</span>
        <select data-role="voice"><option value="">— без голоса —</option></select>
      </label>
      <label class="field">
        <span>Движок, если голос не выбран</span>
        <select data-role="engine"></select>
      </label>
    </div>
    <label class="toggle">
      <input type="checkbox" data-role="auto-accent" />
      Расставлять ударения (RUAccent) — по умолчанию как в панели настроек
    </label>
    <div class="row" style="margin-top:6px">
      <button class="outline-button" data-role="run">Показать</button>
      ${config.fromTextarea ? '<button class="tiny ghost" data-role="from-text">взять текст вкладки</button>' : ''}
      <span class="muted preview-status" data-role="status"></span>
    </div>
    <div class="alert error" data-role="error"></div>
    <p class="hint muted" data-role="empty">
      Пока ничего не проверяли. Здесь появятся стадии: исходный текст → нормализация
      → восстановление «ё» → словарь произношения → ударения → итог. Синтез не запускается.
    </p>
    <div data-role="result" hidden></div>`;
}

const previewEl = (mount, role) => mount.querySelector(`[data-role="${role}"]`);

// Панель монтируется один раз (введённый текст не должен пропадать при
// обновлении списка голосов), а списки голосов и движков обновляются отдельно.
function renderPreviewPanels() {
  PREVIEW_MOUNTS.forEach((config) => {
    const mount = $(config.id);
    if (!mount) return;
    if (!mount.dataset.ready) {
      mount.innerHTML = previewPanelHtml(config);
      mount.dataset.ready = '1';
      previewEl(mount, 'auto-accent').checked = $(config.accent).checked;
      mount.addEventListener('click', (event) => {
        if (event.target.closest('[data-role="run"]')) { runPreviewPanel(config); return; }
        if (config.fromTextarea && event.target.closest('[data-role="from-text"]')) {
          previewEl(mount, 'text').value = $(config.fromTextarea).value;
        }
      });
    }

    const voice = previewEl(mount, 'voice');
    const voiceCurrent = voice.value;
    voice.innerHTML = '<option value="">— без голоса —</option>'
      + state.voices.map((item) => (
        `<option value="${esc(item.id)}">${esc(item.name)} · ${esc(engineLabel(item.engine))}</option>`
      )).join('');
    voice.value = voiceCurrent;

    const engine = previewEl(mount, 'engine');
    const engineCurrent = engine.value;
    engine.innerHTML = '<option value="">— выберите движок —</option>'
      + Object.values(state.engines).map((info) => (
        `<option value="${esc(info.id)}">${esc(info.label)} — ${info.supports_accents ? 'ударения есть' : 'без ударений'}</option>`
      )).join('');
    engine.value = engineCurrent;
  });
}

// Тумблер панели настроек — источник по умолчанию: пока пользователь не тронул
// preview отдельно, они не должны расходиться.
function syncPreviewAccent(mountId, accentId) {
  const mount = $(mountId);
  if (!mount || !mount.dataset.ready) return;
  previewEl(mount, 'auto-accent').checked = $(accentId).checked;
}

function previewStagesHtml(data) {
  const stage = (title, body) => `
    <p class="eyebrow" style="margin:16px 0 6px">${title}</p>
    <blockquote class="ref-phrase">${esc(body)}</blockquote>`;
  const matches = data.matches.length
    ? `<div class="tag-row" style="margin-top:6px">${data.matches.map((match) => (
      `<span class="tag">${esc(match.source)} → ${esc(match.target)} ×${match.count}</span>`
    )).join('')}</div>`
    : '<p class="hint muted" style="margin-top:6px">Ни одно правило не сработало — стадия ничего не изменила.</p>';

  // Стадия ударений показывается только у движка, который их понимает. Для
  // остальных честно сказано, почему её нет, — «пусто» и «не поддерживается»
  // не должны выглядеть одинаково.
  let accents = '';
  if (!data.supports_accents) {
    accents = `<p class="hint muted" style="margin-top:6px">${esc(data.engine_label)} знак «+» `
      + 'не понимает — стадия пропущена, текст уходит как после словаря.</p>';
  } else if (!data.accents_applied) {
    accents = '<p class="hint muted" style="margin-top:6px">Ударения выключены — '
      + 'текст уходит как после словаря.</p>';
  } else if (data.accentizer && data.accentizer.error) {
    // Ошибка RUAccent видна рядом со своей стадией, а не молчаливым отсутствием
    // ударений: иначе пользователь искал бы причину в тексте, а не в модели.
    accents = `<div class="alert error show" style="margin-top:6px">RUAccent не поднялся: `
      + `${esc(data.accentizer.error)} — текст уйдёт без ударений.</div>`;
  }
  const accentStage = data.supports_accents
    ? stage('УДАРЕНИЯ (RUACCENT)', data.accentized) + accents
    : accents;

  // Стадия «ё» показывается отдельно: «е» и «ё» — разные символы, и по равенству
  // с нормализацией видно, сработал шаг или нет. Неоднозначные «е/ё» он не
  // трогает намеренно — их уточняет панель предложений.
  const yoChanged = data.yo !== data.normalized;
  const yoNote = yoChanged
    ? '<p class="hint muted" style="margin-top:6px">Бесспорные слова получили «ё»: '
      + '«е» на месте «ё» модель прочитала бы как «е».</p>'
    : '<p class="hint muted" style="margin-top:6px">Шаг ничего не изменил: слов '
      + 'с гарантированной «ё» не нашлось (неоднозначные «е/ё» здесь не меняются).</p>';

  return stage('ИСХОДНЫЙ ТЕКСТ', data.original)
    + stage('НОРМАЛИЗАЦИЯ', data.normalized)
    + stage('ВОССТАНОВЛЕНИЕ Ё', data.yo) + yoNote
    + stage('СЛОВАРЬ ПРОИЗНОШЕНИЯ', data.dictionary) + matches
    + accentStage
    + stage(`ИТОГ: ЧТО УСЛЫШИТ МОДЕЛЬ · ${esc(data.engine_label)}`, data.final);
}

function setPreviewBusy(mount, busy, note) {
  previewEl(mount, 'run').disabled = busy;
  previewEl(mount, 'status').textContent = note;
}

async function runPreviewPanel(config) {
  const mount = $(config.id);
  const errorBox = previewEl(mount, 'error');
  const text = previewEl(mount, 'text').value;
  const voiceId = previewEl(mount, 'voice').value;
  const engine = previewEl(mount, 'engine').value;
  showAlert(errorBox, '');
  if (!text.trim()) {
    showAlert(errorBox, 'Введите текст для проверки');
    return;
  }
  if (!voiceId && !engine) {
    showAlert(errorBox, 'Выберите голос или движок — иначе неясно, ставить ли ударения');
    return;
  }
  setPreviewBusy(mount, true, 'смотрю…');
  try {
    const data = await api('/api/text/preview', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text,
        voice_id: voiceId || null,
        engine: engine || null,
        auto_accent: previewEl(mount, 'auto-accent').checked,
      }),
    });
    previewEl(mount, 'result').innerHTML = previewStagesHtml(data);
    previewEl(mount, 'result').hidden = false;
    previewEl(mount, 'empty').hidden = true;
  } catch (error) {
    previewEl(mount, 'result').hidden = true;
    previewEl(mount, 'empty').hidden = false;
    showAlert(errorBox, error.message);
  } finally {
    setPreviewBusy(mount, false, '');
  }
}

// Preview конкретной реплики: текст и голос берутся на бэкенде по project_id и
// индексу, поэтому показанное — это то, что уйдёт в модель для этой реплики,
// включая её собственный голос поверх голоса спикера.
function replicaPreviewHtml(replica) {
  const entry = state.replicaPreview[replica.index];
  if (!entry) return '';
  if (entry.loading) {
    return '<div class="replica-preview"><span class="muted preview-status">смотрю…</span></div>';
  }
  if (entry.error) {
    return `<div class="replica-preview"><div class="alert error show">${esc(entry.error)}</div></div>`;
  }
  return `<div class="replica-preview">${previewStagesHtml(entry.data)}</div>`;
}

async function previewReplica(index) {
  if (!state.project) return;
  state.replicaPreview[index] = { loading: true };
  renderReplicaCards();
  try {
    const data = await api('/api/text/preview', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        project_id: state.project.id,
        replica_index: index,
        auto_accent: $('auto-accent').checked,
      }),
    });
    state.replicaPreview[index] = { data };
  } catch (error) {
    state.replicaPreview[index] = { error: error.message };
  }
  renderReplicaCards();
}

// --- предложения для словаря --------------------------------------------------
// Кандидатов считает бэкенд тем же путём, что и preview («/api/pronunciation/
// suggestions»), поэтому фронт ничего не пересчитывает. Подтверждение и отклонение
// — обычные записи словаря через уже существующий CRUD: отдельного хранилища у
// предложений нет. Отклонение создаёт выключенное правило с пометкой — оно
// остаётся на вкладке 04 и работает памятью об отказе, поэтому слово больше
// не предлагается (фильтр на бэкенде смотрит и на выключенные правила).
const SUGGESTION_KINDS = {
  yo_homograph: 'е/ё',
  yo_low_confidence: 'е/ё · модель',
  stress_homograph: 'ударение',
  rare: 'редкое слово',
};

const CONFIRMED_NOTE = 'подтверждено в предложениях';
const REJECTED_NOTE = 'отклонено в предложениях';

function suggestionsEngineId() {
  const voice = voiceById($('text-voice').value);
  return voice ? voice.engine : null;
}

function resetSuggestions() {
  state.suggestions = [];
  $('suggestions-list').innerHTML = '';
  $('btn-add-all-suggestions').hidden = true;
}

function renderSuggestions() {
  const box = $('suggestions-list');
  const addAll = $('btn-add-all-suggestions');
  if (!state.suggestions.length) {
    box.innerHTML = '';
    addAll.hidden = true;
    return;
  }
  addAll.hidden = false;
  box.innerHTML = state.suggestions.map((item, index) => `
    <div class="card suggestion" data-index="${index}">
      <div class="row between">
        <div class="dict-pair"><b>${esc(item.word)}</b><span class="dict-arrow">→</span></div>
        <span class="tag">${esc(SUGGESTION_KINDS[item.kind] || item.kind)}</span>
      </div>
      <label class="field" style="margin:10px 0 0">
        <span>Предлагаемая замена</span>
        <input type="text" data-role="target" value="${esc(item.target)}"
          placeholder="${item.target ? '' : 'впишите ударение, например «звон+ит»'}" />
      </label>
      <p class="hint muted" style="margin:8px 0 0">${esc(item.reason)}</p>
      ${item.alternatives.length
        ? `<div class="tag-row" style="margin:8px 0 0">${item.alternatives
          .map((alt) => `<span class="tag">${esc(alt)}</span>`).join('')}</div>`
        : ''}
      <div class="row" style="margin-top:10px">
        <button class="tiny" data-role="add">Добавить</button>
        <button class="tiny ghost" data-role="reject">Отклонить</button>
      </div>
    </div>`).join('');
}

// Правки в полях живут в DOM: перед действием переносим их в состояние, иначе
// «Добавить все» отправило бы первоначальные варианты, а не вписанные.
function syncSuggestionTargets() {
  document.querySelectorAll('#suggestions-list .suggestion').forEach((card) => {
    const index = Number(card.dataset.index);
    const input = card.querySelector('[data-role="target"]');
    const item = state.suggestions[index];
    if (input && item) state.suggestions[index] = { ...item, target: input.value.trim() };
  });
}

async function findSuggestions() {
  const text = $('text-body').value.trim();
  const errorBox = $('suggestions-error');
  showAlert(errorBox, '');
  if (!text) {
    resetSuggestions();
    $('suggestions-empty').hidden = false;
    showAlert(errorBox, 'Введите текст — искать слова не в чем');
    return;
  }
  const button = $('btn-find-suggestions');
  const status = $('suggestions-status');
  button.disabled = true;
  status.textContent = 'ищу…';
  $('suggestions-empty').hidden = true;
  $('suggestions-list').innerHTML = '<div class="muted" style="padding:6px 0">Прогоняю текст по стадиям…</div>';
  try {
    const data = await api('/api/pronunciation/suggestions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, engine: suggestionsEngineId() }),
    });
    state.suggestions = data.candidates;
    renderSuggestions();
    if (!state.suggestions.length) {
      $('suggestions-empty').hidden = false;
      $('suggestions-empty').textContent =
        `Слов для уточнения не найдено: просмотрено ${data.considered} — все либо уже `
        + 'в словаре, либо однозначны.';
    }
  } catch (error) {
    resetSuggestions();
    $('suggestions-empty').hidden = true;
    showAlert(errorBox, error.message);
  } finally {
    button.disabled = false;
    status.textContent = '';
  }
}

// Общее для подтверждения, отклонения и «добавить все»: запись идёт в тот же
// словарь, что и вкладка 04, поэтому после неё обновляются и словарь, и список.
async function saveSuggestion(payload) {
  await api('/api/pronunciation', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ whole_word: true, case_sensitive: false, ...payload }),
  });
}

async function confirmSuggestion(index) {
  syncSuggestionTargets();
  const item = state.suggestions[index];
  if (!item) return;
  showAlert($('suggestions-error'), '');
  if (!item.target) {
    showAlert($('suggestions-error'), 'Впишите замену — пустое правило ничего не меняет');
    return;
  }
  const button = document.querySelector(`.suggestion[data-index="${index}"] [data-role="add"]`);
  if (button) button.disabled = true;
  try {
    await saveSuggestion({
      source: item.word, target: item.target, enabled: true, note: CONFIRMED_NOTE,
    });
  } catch (error) {
    showAlert($('suggestions-error'), error.message);
    if (button) button.disabled = false;
    return;
  }
  await loadDictionary().catch(() => {});
  await findSuggestions();
}

async function rejectSuggestion(index) {
  syncSuggestionTargets();
  const item = state.suggestions[index];
  if (!item) return;
  if (!confirm(`Отклонить «${item.word}»? Оно больше не будет предлагаться, `
    + 'но останется в словаре выключенным правилом — его можно включить на вкладке «Словарь».')) {
    return;
  }
  showAlert($('suggestions-error'), '');
  try {
    // Замена обязана быть непустой: у редкого слова её может не быть — берём само слово.
    await saveSuggestion({
      source: item.word,
      target: item.target || item.word,
      enabled: false,
      note: REJECTED_NOTE,
    });
  } catch (error) {
    showAlert($('suggestions-error'), error.message);
    return;
  }
  await loadDictionary().catch(() => {});
  await findSuggestions();
}

async function addAllSuggestions() {
  syncSuggestionTargets();
  const ready = state.suggestions.filter((item) => item.target);
  const skipped = state.suggestions.length - ready.length;
  const errorBox = $('suggestions-error');
  showAlert(errorBox, '');
  if (!ready.length) {
    showAlert(errorBox, 'Ни у одного предложения нет готовой замены — заполните их вручную');
    return;
  }
  const button = $('btn-add-all-suggestions');
  const status = $('suggestions-status');
  button.disabled = true;
  status.textContent = 'добавляю…';
  try {
    // Последовательно, без промежуточной перерисовки: иначе индексы карточек
    // разъехались бы с обновлённым списком.
    for (const item of ready) {
      await saveSuggestion({
        source: item.word, target: item.target, enabled: true, note: CONFIRMED_NOTE,
      });
    }
    await loadDictionary().catch(() => {});
    await findSuggestions();
    if (skipped > 0) {
      showAlert(
        errorBox,
        `Пропущено предложений без замены: ${skipped} — у редких слов её нужно вписать вручную`,
        'info',
      );
    }
  } catch (error) {
    showAlert(errorBox, error.message);
    await loadDictionary().catch(() => {});
  } finally {
    button.disabled = false;
    status.textContent = '';
  }
}

// --- голоса -------------------------------------------------------------------
async function loadVoices() {
  const data = await api('/api/voices');
  state.voices = data.voices;
  renderVoiceCards();
  renderVoiceConfigs();
  // Карточки реплик показывают имя и движок голоса: после переименования или
  // удаления голоса они должны обновиться, а не остаться со старым названием.
  renderReplicaCards();
  renderTextVoiceOptions();
  renderPreviewPanels();
  updateGenerateButton();
}

function previewState(voiceId) {
  if (!state.preview[voiceId]) {
    const voice = voiceById(voiceId);
    // Прослушивание начинается с пресета голоса: это те же настройки, что
    // достанутся новому диалогу, поэтому подобранное здесь можно сохранить
    // обратно в голос и получить их по умолчанию.
    const preset = (voice && voice.preset) || {};
    state.preview[voiceId] = {
      speed: preset.speed ?? PARAM_DEFAULTS.speed,
      cfg_strength: preset.cfg_strength ?? PARAM_DEFAULTS.cfg_strength,
      nfe_step: preset.nfe_step ?? PARAM_DEFAULTS.nfe_step,
      // Ручки движка начинаются с сохранённых у голоса — тех же, что уйдут в диалог.
      engine_params: voice ? { ...voice.engine_params } : {},
      text: PREVIEW_TEXT,
    };
  }
  return state.preview[voiceId];
}

// Что сохранено у голоса. Показываем списком, а не подписью «настройки заданы»:
// пользователь должен видеть, что именно получит новый диалог.
function presetText(voice) {
  const preset = voice.preset || {};
  const parts = [];
  if (preset.speed !== undefined) parts.push(`скорость ${Number(preset.speed).toFixed(2)}x`);
  if (preset.cfg_strength !== undefined) parts.push(`CFG ${Number(preset.cfg_strength).toFixed(1)}`);
  if (preset.nfe_step !== undefined) parts.push(`NFE ${Math.round(preset.nfe_step)}`);
  return parts.length ? `настройки голоса: ${parts.join(' · ')}` : 'настройки голоса: по умолчанию движка';
}

// Настройки, подобранные в «Прослушать», становятся пресетом голоса: с них
// начинается и следующий диалог, и это же значение наследуют слоты и реплики.
async function saveVoicePreset(voiceId) {
  const cfg = state.preview[voiceId];
  if (!cfg) return;
  showAlert($('voice-error'), '');
  try {
    await api(`/api/voices/${voiceId}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        preset: { speed: cfg.speed, cfg_strength: cfg.cfg_strength, nfe_step: cfg.nfe_step },
      }),
    });
    await loadVoices();
    showAlert($('voice-error'), 'Настройки сохранены у голоса: новый диалог начнётся с них', 'info');
  } catch (error) {
    showAlert($('voice-error'), error.message);
  }
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
          <button class="tiny" data-role="save-preset"
                  title="Записать подобранные настройки в голос — новый диалог начнётся с них">сохранить настройки</button>
          <button class="tiny ghost danger" data-role="preview-cancel" hidden>Отменить</button>
          <span class="muted preview-status" data-role="preview-status"></span>
        </div>
        <p class="muted preset-state" data-role="preset-state">${esc(presetText(voice))}</p>
        <button class="tiny ghost compare-button" data-role="compare">⇄ Сравнить движки</button>
        <div class="benchmark-panel" data-role="benchmark" hidden></div>
        <audio data-role="player" controls hidden></audio>
      </div>`;
  }).join('');
  // Открытые панели сравнения пересобираются: список голосов перечитывается после
  // правок (например, после выбора движка), и терять результаты на этом нельзя.
  Object.keys(state.benchmark).forEach((voiceId) => {
    if (state.benchmark[voiceId].open) renderBenchmarkPanel(voiceId);
  });
}

function setPreviewStatus(card, message, isError = false) {
  const el = card.querySelector('[data-role="preview-status"]');
  el.textContent = message;
  el.classList.toggle('err', isError);
}

// Отменяемость показывается у задачи, которая ещё ждёт или уже синтезируется:
// у завершённой отменять нечего, и кнопка только сбивала бы с толку.
async function cancelJob(jobId) {
  return api(`/api/jobs/${jobId}/cancel`, { method: 'POST' });
}

function setCancelButton(button, jobId, visible) {
  if (!button) return;
  button.hidden = !visible;
  button.dataset.jobId = visible ? jobId : '';
  button.disabled = false;
  button.textContent = 'Отменить';
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
    setCancelButton(card.querySelector('[data-role="preview-cancel"]'), job.job_id, true);
    state.previewTimer = setInterval(pollPreview, 1500);
  } catch (error) {
    // Ошибку показываем у самой кнопки: иначе сетевой сбой читается как «кнопка не работает»
    setCancelButton(card.querySelector('[data-role="preview-cancel"]'), '', false);
    setPreviewStatus(card, error.message, true);
  }
}

// Отмена прослушивания: эндпоинт сразу переводит ожидающую задачу в `cancelled`,
// а идущую просит остановиться на безопасной точке. Опрос не бросаем: статус
// `cancelled` придёт ответом на ближайший же запрос, и его покажет pollPreview.
async function cancelPreview(card) {
  const active = state.previewJob;
  if (!active) return;
  const button = card.querySelector('[data-role="preview-cancel"]');
  if (button) { button.disabled = true; button.textContent = 'Отменяю…'; }
  try {
    await cancelJob(active.jobId);
    setPreviewStatus(card, 'отменяю…');
  } catch (error) {
    if (button) { button.disabled = false; button.textContent = 'Отменить'; }
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
    const button = card.querySelector('[data-role="preview-cancel"]');
    if (job.status === 'queued') {
      setPreviewStatus(card, 'в очереди…');
      setCancelButton(button, active.jobId, true);
      return;
    }
    if (job.status === 'processing') {
      setPreviewStatus(card, job.cancel_requested ? 'останавливаю…' : 'генерирую…');
      setCancelButton(button, active.jobId, true);
      return;
    }
    clearInterval(state.previewTimer);
    state.previewTimer = null;
    state.previewJob = null;
    setCancelButton(button, '', false);
    if (job.status === 'cancelled') {
      setPreviewStatus(card, 'отменено');
      return;
    }
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

// --- сравнение движков --------------------------------------------------------
// Одна фраза и один reference, разные модели: система показывает время синтеза и
// WER как справку и ничего не ранжирует — движок выбирает человек кнопкой.
function benchmarkState(voiceId) {
  if (!state.benchmark[voiceId]) {
    state.benchmark[voiceId] = {
      open: false,
      text: BENCHMARK_TEXT,
      engines: [],   // пусто — «все объявленные»; заполняется при открытии панели
      qa: 'off',
      runId: null,
      data: null,
      busy: false,
      error: '',
      timer: null,
    };
  }
  return state.benchmark[voiceId];
}

function stopBenchmarkPoll(voiceId) {
  const cfg = state.benchmark[voiceId];
  if (cfg && cfg.timer) {
    clearInterval(cfg.timer);
    cfg.timer = null;
  }
}

function benchmarkSeconds(value) {
  return value === null || value === undefined ? '—' : `${Number(value).toFixed(1)} сек`;
}

function benchmarkWer(result) {
  const wer = result.qa ? result.qa.wer : null;
  return wer === null || wer === undefined ? 'WER: —' : `WER: ${Number(wer).toFixed(2)}`;
}

function benchmarkRowHtml(voice, result) {
  const label = esc(result.engine_label || result.engine);
  if (result.status === 'done') {
    const chosen = Boolean(voice) && voice.engine === result.engine;
    return `
      <div class="benchmark-row">
        <b class="benchmark-engine">${label}</b>
        <button class="tiny" data-role="benchmark-play" data-engine="${esc(result.engine)}"
                data-url="${esc(result.audio_url)}">Play</button>
        <span class="muted">Render: ${benchmarkSeconds(result.render_sec)}</span>
        <span class="muted">${benchmarkWer(result)}</span>
        <button class="tiny${chosen ? '' : ' primary'}" data-role="benchmark-select"
                data-engine="${esc(result.engine)}" ${chosen ? 'disabled' : ''}>
          ${chosen ? 'движок голоса' : 'Выбрать этот движок'}
        </button>
      </div>`;
  }
  // Ошибка движка — строка с её текстом, а не сломанная панель: сравнение
  // продолжается, и остальные строки остаются на месте.
  const note = result.error || 'движок не ответил';
  const kind = result.status === 'skipped' ? 'warn' : 'err';
  const prefix = result.status === 'skipped' ? 'пропущен' : 'ошибка';
  return `
    <div class="benchmark-row">
      <b class="benchmark-engine">${label}</b>
      <span class="muted ${kind}">${prefix}: ${esc(note)}</span>
    </div>`;
}

function benchmarkResultsHtml(voiceId) {
  const cfg = benchmarkState(voiceId);
  const voice = voiceById(voiceId);
  if (cfg.error && !cfg.busy) return `<div class="alert error show">${esc(cfg.error)}</div>`;
  if (!cfg.data) {
    return cfg.busy
      ? '<p class="muted">Ставлю сравнение в очередь…</p>'
      : '<p class="muted">Отметьте движки, впишите фразу и нажмите «Сравнить».</p>';
  }
  if (!cfg.data.results.length) {
    return `<p class="muted">${cfg.busy ? 'Синтезирую первый движок…' : 'Ни один движок ещё не ответил.'}</p>`;
  }
  return `<div class="benchmark-rows">${cfg.data.results
    .map((result) => benchmarkRowHtml(voice, result)).join('')}</div>`;
}

function benchmarkPanelHtml(voiceId) {
  const cfg = benchmarkState(voiceId);
  const boxes = Object.values(state.engines).map((info) => `
    <label class="toggle benchmark-engine-toggle">
      <input type="checkbox" data-role="benchmark-engine" data-engine="${esc(info.id)}"
             ${cfg.engines.includes(info.id) ? 'checked' : ''} />
      ${esc(info.label)}
    </label>`).join('');
  const total = Math.max(cfg.engines.length, 1);
  const done = cfg.data ? cfg.data.results.length : 0;
  const percent = Math.min(100, Math.round((done / total) * 100));
  const status = cfg.busy
    ? `обработано ${done} из ${cfg.engines.length}`
    : 'фраза и reference общие — отличаются только движки';
  return `
    <div class="benchmark-head">СРАВНИТЬ ДВИЖКИ</div>
    <label class="field">
      <span>Тестовая фраза</span>
      <input type="text" data-role="benchmark-text" value="${esc(cfg.text)}" />
    </label>
    <div class="benchmark-engines">${boxes}</div>
    <label class="field">
      <span>Проверка качества</span>
      <select data-role="benchmark-qa">
        <option value="off"${cfg.qa === 'off' ? ' selected' : ''}>выключена</option>
        <option value="smart"${cfg.qa === 'smart' ? ' selected' : ''}>Smart — по подозрительным</option>
        <option value="strict"${cfg.qa === 'strict' ? ' selected' : ''}>Strict — каждый движок</option>
      </select>
    </label>
    <div class="test-row">
      <button class="tiny primary" data-role="benchmark-run" ${cfg.busy ? 'disabled' : ''}>
        ${cfg.busy ? 'сравниваю…' : 'Сравнить'}
      </button>
      <span class="muted benchmark-status">${status}</span>
    </div>
    <div class="progress"><div style="width:${percent}%"></div></div>
    <div data-role="benchmark-results">${benchmarkResultsHtml(voiceId)}</div>
    <p class="hint muted">
      WER и время — справка, а не приговор: система не выбирает лучший движок сама.
      Прослушайте takes и нажмите «Выбрать этот движок» — reference и настройки голоса
      при этом не меняются.
    </p>
    <audio data-role="benchmark-player" controls hidden></audio>`;
}

function renderBenchmarkPanel(voiceId) {
  const card = previewCard(voiceId);
  if (!card) return;
  const box = card.querySelector('[data-role="benchmark"]');
  if (!box) return;
  const cfg = benchmarkState(voiceId);
  if (!cfg.open) {
    box.innerHTML = '';
    box.hidden = true;
    delete box.dataset.ready;
    return;
  }
  if (!box.dataset.ready) {
    box.innerHTML = benchmarkPanelHtml(voiceId);
    box.dataset.ready = '1';
  }
  box.hidden = false;
  syncBenchmarkPanel(voiceId);
}

// Обновление хода сравнения без пересборки панели: опрос статуса не должен
// пересоздавать плеер и сбрасывать прослушивание и правки в форме.
function syncBenchmarkPanel(voiceId) {
  const card = previewCard(voiceId);
  const box = card ? card.querySelector('[data-role="benchmark"]') : null;
  if (!box) return;
  const cfg = benchmarkState(voiceId);
  const button = box.querySelector('[data-role="benchmark-run"]');
  if (button) {
    button.disabled = cfg.busy;
    button.textContent = cfg.busy ? 'сравниваю…' : 'Сравнить';
  }
  const done = cfg.data ? cfg.data.results.length : 0;
  const percent = Math.min(100, Math.round((done / Math.max(cfg.engines.length, 1)) * 100));
  const bar = box.querySelector('.progress > div');
  if (bar) bar.style.width = `${percent}%`;
  const status = box.querySelector('.benchmark-status');
  if (status) {
    status.textContent = cfg.busy
      ? `обработано ${done} из ${cfg.engines.length}`
      : 'фраза и reference общие — отличаются только движки';
  }
  const results = box.querySelector('[data-role="benchmark-results"]');
  if (results) results.innerHTML = benchmarkResultsHtml(voiceId);
}

function toggleBenchmark(voiceId) {
  const cfg = benchmarkState(voiceId);
  cfg.open = !cfg.open;
  // По умолчанию сравниваем на всём, что установлено: отмечать руками каждый
  // движок в самом частом сценарии — лишний шаг.
  if (cfg.open && !cfg.engines.length) cfg.engines = Object.keys(state.engines);
  renderBenchmarkPanel(voiceId);
}

async function runBenchmark(voiceId) {
  const cfg = benchmarkState(voiceId);
  const text = cfg.text.trim();
  if (!cfg.engines.length) { cfg.error = 'Отметьте хотя бы один движок'; renderBenchmarkPanel(voiceId); return; }
  if (!text) { cfg.error = 'Введите фразу для сравнения'; renderBenchmarkPanel(voiceId); return; }

  cfg.error = '';
  cfg.busy = true;
  cfg.data = null;
  renderBenchmarkPanel(voiceId);
  // Движок сравнения поднимается впервые и может грузиться десятки секунд —
  // состояние моделей видно в шапке.
  watchEngines();
  try {
    const started = await api(`/api/voices/${voiceId}/benchmark`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, engines: cfg.engines, qa: cfg.qa, auto_accent: true }),
    });
    cfg.runId = started.benchmark_id;
    stopBenchmarkPoll(voiceId);
    cfg.timer = setInterval(() => pollBenchmark(voiceId), 1500);
  } catch (error) {
    cfg.busy = false;
    cfg.error = error.message;
    renderBenchmarkPanel(voiceId);
  }
}

async function pollBenchmark(voiceId) {
  const cfg = state.benchmark[voiceId];
  if (!cfg || !cfg.runId) return;
  if (!previewCard(voiceId)) {  // карточку перерисовали или голос удалили
    stopBenchmarkPoll(voiceId);
    return;
  }
  try {
    const data = await api(`/api/benchmarks/${cfg.runId}`);
    cfg.data = data;
    if (data.status === 'done' || data.status === 'error') {
      stopBenchmarkPoll(voiceId);
      cfg.busy = false;
      if (data.status === 'error') cfg.error = data.error || 'Сравнение не удалось';
    }
    renderBenchmarkPanel(voiceId);
  } catch (error) {
    stopBenchmarkPoll(voiceId);
    cfg.busy = false;
    cfg.error = error.message;
    renderBenchmarkPanel(voiceId);
  }
}

function playBenchmarkTake(voiceId, url) {
  const card = previewCard(voiceId);
  if (!card || !url) return;
  const player = card.querySelector('[data-role="benchmark-player"]');
  player.src = `${url}?t=${Date.now()}`;
  player.hidden = false;
  player.play().catch(() => { /* автоплей может быть заблокирован — плеер виден */ });
}

async function selectBenchmarkEngine(voiceId, engine) {
  const cfg = benchmarkState(voiceId);
  if (!cfg.runId) return;
  try {
    const voice = await api(`/api/benchmarks/${cfg.runId}/select`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ engine }),
    });
    // Карточка перечитывается: движок голоса изменился, и это видно в её теге.
    // Панель сравнения остаётся открытой — результаты ещё нужны для прослушивания.
    await loadVoices();
    renderBenchmarkPanel(voiceId);
    showAlert(
      $('voice-error'),
      `Голос «${voice.name}» переведён на ${engineLabel(voice.engine)}: reference и настройки не тронуты`,
      'info',
    );
  } catch (error) {
    cfg.error = error.message;
    renderBenchmarkPanel(voiceId);
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

// --- проект диалога -----------------------------------------------------------
// Вкладка работает с проектом, а не с разовой задачей: исходный текст, голоса
// спикеров, реплики и их варианты лежат в базе. Поэтому правки переживают
// перезагрузку страницы, а пересинтез одной реплики не трогает остальные.

// Значения по умолчанию — те же числа, что и в backend/config.py. Нужны только
// как запасной вариант: пока голос не выбран, разрешать слои не из чего, а поля
// карточки всё равно надо чем-то заполнить.
const PARAM_DEFAULTS = {
  speed: 1, cfg_strength: 2, nfe_step: 32,
  target_rms: 0.1, gain_db: 0, pitch_semitones: 0, pause_override_ms: null,
};

// Ручки, которые правятся у одной реплики поверх карточки спикера. Границы взяты
// из config.py: это те же ползунки, что и в панели спикеров, только на реплику.
const PARAM_RANGES = {
  speed: { min: 0.5, max: 2, step: 0.05, label: 'Скорость речи' },
  cfg_strength: { min: 1, max: 4, step: 0.1, label: 'CFG strength — стабильность ↔ выразительность' },
  target_rms: { min: 0.02, max: 0.3, step: 0.01, label: 'Целевая громкость куска (RMS)' },
  gain_db: { min: -20, max: 20, step: 0.5, label: 'Громкость' },
  pitch_semitones: { min: -12, max: 12, step: 0.5, label: 'Питч-шифт' },
};

const PROJECT_KEY = 'voiceStudio.projectId';

// --- источники значений -------------------------------------------------------
// Иерархия «движок → пресет голоса → участник → реплика» считается на бэкенде
// (backend/settings_resolution.py), и каждая ручка приходит вместе с источником.
// Своего порядка слоёв у клиента нет намеренно: вторая точка слияния однажды уже
// разошлась с первой, и подпись «наследуется от голоса» показывала бы не то,
// чем реплика читается на самом деле.
const SOURCE_ENGINE = 'engine';
const SOURCE_VOICE = 'voice';
const SOURCE_SPEAKER = 'speaker';
const SOURCE_REPLICA = 'replica';

// Ручка слоя: значение, источник и (у перекрытой) значение нижнего слоя — то,
// куда вернёт сброс. Пустой `settings` означает «голос не выбран»: слои разрешить
// не из чего, поэтому показываем дефолты движка, чтобы поля не были пустыми.
function settingItem(scope, field) {
  const item = ((scope && scope.settings) || {})[field];
  if (item && item.value !== undefined) return item;
  return { value: PARAM_DEFAULTS[field], source: SOURCE_ENGINE };
}

// Значение, которым ручка читалась бы без правки этого слоя. По нему видно,
// выбрал ли слот что-то своё: равенство нижнему слою — это не выбор, а наследование.
function inheritedValue(scope, field) {
  const item = settingItem(scope, field);
  return item.source === SOURCE_SPEAKER || item.source === SOURCE_REPLICA
    ? item.inherited
    : item.value;
}

// Подпись о происхождении значения: видно, что унаследовано, а что выбрано здесь.
// Про дефолт движка молчим — это исходное состояние, а не решение пользователя.
// `own` — слой, который правит эта панель: у карточки реплики это реплика, у панели
// слотов — участник. Одно и то же значение читается по-разному: правка этого слоя
// («правка этой реплики») или чужая, унаследованная («наследуется от участника»).
function sourceNoteHtml(item, voiceLabel = '', own = SOURCE_REPLICA) {
  let note = '';
  if (item.source === SOURCE_VOICE) note = `наследуется от голоса${voiceLabel ? ` «${voiceLabel}»` : ''}`;
  else if (item.source === SOURCE_SPEAKER) note = item.source === own ? 'правка участника' : 'наследуется от участника';
  else if (item.source === SOURCE_REPLICA) note = 'правка этой реплики';
  return note ? `<span class="muted param-source">${esc(note)}</span>` : '';
}

// Кнопка сброса — только у правки этого слоя: унаследованное сбрасывать некуда.
// `own` — слой панели, как и у подписи: в карточке реплики кнопка снимает правку
// реплики, а значение, пришедшее от участника, снимать отсюда нечем — такой сброс
// ушёл бы в никуда. В подсказке — значение, которым ручка будет читаться после сброса.
function resetButtonHtml(item, dataField, param = '', backField = dataField, own = SOURCE_REPLICA) {
  if (item.source !== own) return '';
  const back = item.inherited === undefined ? '' : ` → ${fmtParam(backField, item.inherited)}`;
  return `<button class="tiny ghost param-reset" data-role="reset" data-field="${esc(dataField)}"${param ? ` data-param="${esc(param)}"` : ''}
                title="вернуть наследуемое значение${esc(back)}">сброс</button>`;
}

function fmtParam(field, value) {
  if (field === 'speed') return `${Number(value).toFixed(2)}x`;
  if (field === 'cfg_strength') return Number(value).toFixed(1);
  if (field === 'nfe_step') return String(Math.round(value));
  if (field === 'gain_db') return fmtGain(Number(value));
  if (field === 'pitch_semitones') return fmtPitch(Number(value));
  if (field === 'target_rms') return fmtRms(Number(value));
  if (field === 'pause_override_ms') return value === null ? 'общая для диалога' : `${value} мс`;
  return String(value);
}

function speakerByKey(key) {
  const speakers = (state.project && state.project.speakers) || [];
  return speakers.find((speaker) => speaker.key === key) || null;
}

// Подпись спикера: имя из «ИВАН:» или «Слот N» для маркера «(N)».
function speakerLabel(key) {
  const speaker = speakerByKey(key);
  return (speaker && speaker.label) || (key.startsWith('#') ? `Слот ${key.slice(1)}` : key);
}

function voiceName(voiceId) {
  const voice = voiceById(voiceId);
  return voice ? voice.name : (voiceId || 'не выбран');
}

// Два представления одного диалога: карточки реплик и исходный текст с маркерами.
function switchDialogueView(view) {
  document.querySelectorAll('.view-switch button').forEach((button) => {
    button.classList.toggle('active', button.dataset.view === view);
  });
  $('visual-view').hidden = view !== 'visual';
  $('source-view').hidden = view !== 'source';
  if (view === 'source') setSourceState();
}

// Разбор идёт только по команде: правка текста меняет состав реплик, и молча
// переписывать их (вместе с вариантами и правками карточек) на каждом символе
// нельзя — об этом сообщает подпись рядом с кнопкой.
function markSourceDirty() {
  if (state.sourceDirty) return;
  state.sourceDirty = true;
  setSourceState();
  renderParseSummary();
}

function setSourceState() {
  const element = $('source-state');
  const replicas = (state.project && state.project.replicas) || [];
  if (state.sourceDirty) {
    element.textContent = 'текст изменён — разберите заново';
    element.classList.add('warn');
    return;
  }
  element.classList.remove('warn');
  element.textContent = state.project
    ? `разобрано: ${replicas.length} ${plural(replicas.length, 'реплика', 'реплики', 'реплик')}`
    : 'разбор ещё не выполнялся';
}

// --- проект: загрузка, разбор, применение ответа API --------------------------
async function ensureProject() {
  if (state.project) return state.project;
  const source = $('dialogue').value;
  const project = await api('/api/projects', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name: projectName(source), source_text: source }),
  });
  state.project = project;
  rememberProject(project.id);
  return project;
}

// Имя нужно только для базы: в интерфейсе его не видно. Берём начало первой
// реплики без маркеров и имени-метки — по нему проект и узнаётся в списке.
function projectName(source) {
  const first = String(source).split('\n').map((line) => line.trim()).find(Boolean) || '';
  const clean = first.replace(/\(\s*\d{1,3}[^()]*\)/g, ' ').replace(/^[^:]{0,24}:\s*/, '').trim();
  return (clean || 'Диалог').slice(0, 48);
}

// Идентификатор открытого проекта запоминаем: без этого перезагрузка страницы
// начинала бы новый проект, и прежний вместе с вариантами оставался бы мусором.
function rememberProject(projectId) {
  try {
    if (projectId) localStorage.setItem(PROJECT_KEY, projectId);
    else localStorage.removeItem(PROJECT_KEY);
  } catch (_) { /* приватный режим — просто не запоминаем */ }
}

async function loadSavedProject() {
  let projectId = null;
  try { projectId = localStorage.getItem(PROJECT_KEY); } catch (_) { /* приватный режим */ }
  if (!projectId) return;
  try {
    const project = await api(`/api/projects/${projectId}`);
    if (!project.source_text) return;
    $('dialogue').value = project.source_text;
    applyProject(project);
  } catch (_) {
    // Проект могли удалить: начинаем с чистого листа, а не с ошибки на загрузке.
    rememberProject(null);
  }
  setSourceState();
}

function applyProject(project, { speakers = true } = {}) {
  state.project = project;
  // Реплики заменены целиком: показанный ранее preview относился к прежнему
  // тексту или голосу и после разбора был бы уже неправдой. Таймлайн тоже
  // устарел: его длительности приходят отдельным запросом и считаются по take'ам.
  state.replicaPreview = {};
  state.timeline = null;
  state.timelineError = null;
  if (speakers) renderVoiceConfigs();
  renderReplicaCards();
  renderParseSummary();
  setSourceState();
  updateGenerateButton();
  renderTimeline();
  refreshTimeline();
}

async function refreshProject() {
  if (!state.project) return;
  applyProject(await api(`/api/projects/${state.project.id}`), { speakers: false });
}

// Таймлайн отдаёт сегменты в форме карточек реплик (`_replica_payload` на
// бэкенде), поэтому список реплик берётся прямо из него: карточки и инспектор
// сегмента читают одни и те же поля, и второй разбор ответа не нужен.
function applyTimeline(data) {
  if (!state.project || !data) return;
  const segments = data.replicas || [];
  if (!segments.length && (state.project.replicas || []).length) return;
  state.project.replicas = segments;
  renderReplicaCards();
  renderParseSummary();
}

// «Применить и разобрать»: исходный текст уходит в проект, проект разбирается в
// реплики. Правки, сделанные в карточках, при повторном разборе перекрываются
// параметрами из маркеров — текст и карточки правят одно и то же.
async function applySource() {
  showAlert($('parse-error'), '');
  try {
    const project = await ensureProject();
    await api(`/api/projects/${project.id}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ source_text: $('dialogue').value }),
    });
    const parsed = await api(`/api/projects/${project.id}/parse`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ chunk_strategy: $('chunk-strategy').value }),
    });
    state.sourceDirty = false;
    applyProject(parsed);
    if (!parsed.replicas.length) {
      showAlert($('parse-error'), 'Текст пуст — вставьте диалог или хотя бы одну фразу.');
    }
    return true;
  } catch (error) {
    showAlert($('parse-error'), error.message);
    setSourceState();
    return false;
  }
}

// --- голоса спикеров (слоты из маркеров и имена спикеров) ---------------------
// Панель правит спикера целиком — голос и набор параметров, от которого
// наследуют его реплики. Отдельной реплике можно назначить своё в её карточке.
// В карточке стоит разрешённое значение (движок → пресет голоса → слот), а не
// `overrides`: иначе поля показывали бы дефолты движка и при сохранении молча
// перекрывали бы настройки, подобранные для голоса.
function speakerConfig(speaker) {
  const cfg = defaultConfig();
  cfg.voice_id = speaker.voice_id || '';
  Object.keys(PARAM_DEFAULTS).forEach((field) => {
    cfg[field] = settingItem(speaker, field).value;
  });
  return cfg;
}

const SPEAKER_CONTROLS = {
  speed: 'speed', cfg_strength: 'cfg', nfe_step: 'nfe',
  gain_db: 'gain', pitch_semitones: 'pitch', target_rms: 'rms',
};

// Голова ручки: название слева, значение и «сброс» — справа. Кнопка лежит внутри
// общего блока со значением, чтобы её появление не сдвигало значение к центру.
function sliderHeadHtml(title, label, reset = '') {
  return `<span class="slider-head">${esc(title)}<span class="row">${label}${reset}</span></span>`;
}

const valueHtml = (role, text) => `<b data-role="${role}-label">${text}</b>`;

// Возврат контрола карточки к наследуемому значению — так работает «сброс»:
// сохранение сравнивает поля с нижним слоем, и совпадение само снимает правку.
function setSpeakerControl(card, field, value) {
  if (field === 'pause_override_ms') {
    const inherit = value === null;
    card.querySelector('[data-role="pause-inherit"]').checked = inherit;
    const range = card.querySelector('[data-role="pause-override"]');
    range.disabled = inherit;
    if (!inherit) range.value = value;
    card.querySelector('[data-role="pause-label"]').textContent = fmtPause(value);
    return;
  }
  const role = SPEAKER_CONTROLS[field];
  const control = role ? card.querySelector(`[data-role="${role}"]`) : null;
  if (!control) return;
  control.value = value;
  const label = card.querySelector(`[data-role="${role}-label"]`);
  if (label) label.textContent = fmtParam(field, value);
}

// Ручки движка в карточке слота: значения приходят разрешёнными, поэтому ползунок
// стоит на том, чем слот читается сейчас, а подпись говорит, откуда это взялось.
function speakerEngineParamsHtml(speaker, engineId, voiceLabel) {
  const info = engineInfo(engineId);
  if (!info || !info.params.length) return '';
  return `<div class="grid-2">${info.params.map((param) => {
    const item = settingItem(speaker, param.name);
    const value = typeof item.value === 'number' ? item.value : param.default;
    return `
    <div class="field">
      ${sliderHeadHtml(param.label, `<b data-param-label="${esc(param.name)}">${fmtByStep(value, param.step)}</b>`, resetButtonHtml(item, 'engine_params', param.name, param.name, SOURCE_SPEAKER))}
      <input type="range" data-engine-param="${esc(param.name)}"
             min="${param.min}" max="${param.max}" step="${param.step}" value="${value}" />
      ${param.hint ? `<span class="muted">${esc(param.hint)}</span>` : ''}
      ${sourceNoteHtml(item, voiceLabel, SOURCE_SPEAKER)}
    </div>`;
  }).join('')}</div>`;
}

function renderVoiceConfigs() {
  const box = $('voice-configs');
  const speakers = (state.project && state.project.speakers) || [];
  if (!speakers.length) {
    box.innerHTML = '<span class="muted">Вставьте диалог и нажмите «Применить и разобрать».</span>';
    return;
  }
  box.innerHTML = speakers.map((speaker) => {
    const cfg = speakerConfig(speaker);
    const key = speaker.key;
    const label = speaker.label || key;
    const slot = key.startsWith('#') ? key.slice(1) : null;
    const options = state.voices
      .map((v) => `<option value="${esc(v.id)}"${v.id === cfg.voice_id ? ' selected' : ''}>${esc(v.name)}</option>`)
      .join('');
    const markerTag = slot === null ? '' : `<span class="tag">маркер (${slot})</span>`;
    // Набор настроек диктует движок выбранного голоса: у XTTS вместо CFG и NFE
    // температура и штраф за повторы. Движок можно смешивать в одном диалоге —
    // каждый слот считает свой набор.
    const voice = voiceById(cfg.voice_id);
    const voiceLabel = voice ? voice.name : '';
    const item = (field) => settingItem(speaker, field);
    const reset = (field) => resetButtonHtml(item(field), field, '', field, SOURCE_SPEAKER);
    const note = (field) => sourceNoteHtml(item(field), voiceLabel, SOURCE_SPEAKER);
    const noteText = voice
      ? engineNote(voice.engine)
      : 'выберите голос — набор настроек появится под движок этого голоса';
    return `
      <div class="card" data-voice="${esc(key)}">
        <div class="voice-top">
          <span class="avatar${voice && voice.engine !== ENGINE_F5 ? ' violet' : ''}">${esc(initials(label))}</span>
          <div class="voice-name">
            <h3>${esc(label)}</h3>
            <p data-role="engine-note">${esc(noteText)}</p>
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

        <div class="field">
          ${sliderHeadHtml('Скорость речи', valueHtml('speed', `${cfg.speed.toFixed(2)}x`), reset('speed'))}
          <input type="range" data-role="speed" min="0.5" max="2" step="0.05" value="${cfg.speed}" />
          ${note('speed')}
        </div>
        <div data-role="engine-params">${voice ? speakerEngineParamsHtml(speaker, voice.engine, voiceLabel) : ''}</div>
        <div data-role="f5-params" ${voice && voice.engine === ENGINE_F5 ? '' : 'hidden'}>
          <div class="field">
            ${sliderHeadHtml('CFG strength — стабильность ↔ выразительность', valueHtml('cfg', cfg.cfg_strength.toFixed(1)), reset('cfg_strength'))}
            <input type="range" data-role="cfg" min="1" max="4" step="0.1" value="${cfg.cfg_strength}" />
            ${note('cfg_strength')}
          </div>
          <div class="field" style="margin-bottom:0">
            ${sliderHeadHtml('NFE steps — быстрее ↔ качественнее', '', reset('nfe_step'))}
            <select data-role="nfe">
              ${nfeOptions(cfg.nfe_step)}
            </select>
            ${note('nfe_step')}
          </div>
        </div>

        <details class="advanced"${state.speakerDetails[key] ? ' open' : ''}>
          <summary>Ещё настройки голоса</summary>
          <div class="field">
            ${sliderHeadHtml('Громкость', valueHtml('gain', fmtGain(cfg.gain_db)), reset('gain_db'))}
            <input type="range" data-role="gain" min="-20" max="20" step="0.5" value="${cfg.gain_db}" />
            ${note('gain_db')}
          </div>
          <div class="field">
            ${sliderHeadHtml('Питч-шифт', valueHtml('pitch', fmtPitch(cfg.pitch_semitones)), reset('pitch_semitones'))}
            <input type="range" data-role="pitch" min="-12" max="12" step="0.5" value="${cfg.pitch_semitones}" />
            <span class="muted">экспериментально: меняет высоту тона, а не пол голоса</span>
            ${note('pitch_semitones')}
          </div>
          <div class="field">
            ${sliderHeadHtml('Целевая громкость куска (RMS)', valueHtml('rms', fmtRms(cfg.target_rms)), reset('target_rms'))}
            <input type="range" data-role="rms" min="0.02" max="0.3" step="0.01" value="${cfg.target_rms}" />
            ${note('target_rms')}
          </div>
          <div class="field" style="margin-bottom:0">
            ${sliderHeadHtml('Пауза перед репликами', valueHtml('pause', fmtPause(cfg.pause_override_ms)), reset('pause_override_ms'))}
            <input type="range" data-role="pause-override" min="0" max="5000" step="50"
                   value="${cfg.pause_override_ms ?? 400}" ${cfg.pause_override_ms === null ? 'disabled' : ''} />
            <label class="toggle">
              <input type="checkbox" data-role="pause-inherit" ${cfg.pause_override_ms === null ? 'checked' : ''} />
              как в общих настройках
            </label>
            ${note('pause_override_ms')}
          </div>
        </details>
      </div>`;
  }).join('');
}

// Русская форма слова по числу: 1 реплика, 2 реплики, 5 реплик.
function plural(count, one, few, many) {
  const mod100 = Math.abs(count) % 100;
  const mod10 = mod100 % 10;
  if (mod100 >= 11 && mod100 <= 14) return many;
  if (mod10 === 1) return one;
  if (mod10 >= 2 && mod10 <= 4) return few;
  return many;
}

// Сводка над карточками реплик. Про неназначенный голос сообщаем здесь, а не
// только подсказкой отключённой кнопки: иначе непонятно, почему генерация
// недоступна. Считаем участников и реплики с правками — по ним видно, что
// диалог уже настраивали, а что осталось на значениях спикера.
function renderParseSummary() {
  const summary = $('parse-summary');
  const replicas = (state.project && state.project.replicas) || [];
  const speakers = (state.project && state.project.speakers) || [];
  if (!state.project) {
    summary.textContent = 'Реплик пока нет: вставьте текст в Source / Advanced и нажмите «Применить и разобрать».';
    summary.classList.remove('warn');
    return;
  }
  const unassigned = speakers.filter((speaker) => !speaker.voice_id).length;
  const edited = replicas.filter((replica) => Object.keys(replica.overrides || {}).length).length;
  const parts = [
    `${replicas.length} ${plural(replicas.length, 'реплика', 'реплики', 'реплик')}`,
    `${speakers.length} ${plural(speakers.length, 'участник', 'участника', 'участников')}`,
  ];
  if (unassigned) parts.push(`без голоса: ${unassigned}`);
  if (edited) parts.push(`с правками: ${edited}`);
  if (state.sourceDirty) parts.push('текст изменён — нужен повторный разбор');
  summary.textContent = parts.join(' · ');
  summary.classList.toggle('warn', unassigned > 0 || state.sourceDirty);
}

// --- работа с карточками спикеров ---------------------------------------------
// Читаем карточку перед каждой записью: панель и база должны совпадать, иначе
// незаписанная правка молча не попадёт ни в рендер, ни в пересинтез реплики.
// В слот уходит только то, что отличается от наследуемого: равенство настройкам
// голоса — это не выбор слота, а наследование. Иначе слот всегда перекрывал бы
// пресет голоса всеми ручками сразу, и «настроил голос один раз» не работало бы
// ни в одном новом диалоге.
function readSpeakerCard(card) {
  const speaker = speakerByKey(card.dataset.voice) || {};
  const overrides = {};
  const values = {
    speed: parseFloat(card.querySelector('[data-role="speed"]').value),
    cfg_strength: parseFloat(card.querySelector('[data-role="cfg"]').value),
    nfe_step: parseInt(card.querySelector('[data-role="nfe"]').value, 10),
    gain_db: parseFloat(card.querySelector('[data-role="gain"]').value),
    pitch_semitones: parseFloat(card.querySelector('[data-role="pitch"]').value),
    target_rms: parseFloat(card.querySelector('[data-role="rms"]').value),
    pause_override_ms: card.querySelector('[data-role="pause-inherit"]').checked
      ? null
      : parseInt(card.querySelector('[data-role="pause-override"]').value, 10),
  };
  Object.entries(values).forEach(([field, value]) => {
    if (value !== inheritedValue(speaker, field)) overrides[field] = value;
  });
  // Ручки движка (temperature у XTTS и т.п.) проверяются так же: в слот идёт
  // только то, что слот выбрал сам, а F5-специфичные cfg/nfe остаются полями выше.
  const params = {};
  Object.entries(readEngineParams(card)).forEach(([name, value]) => {
    if (value !== inheritedValue(speaker, name)) params[name] = value;
  });
  if (Object.keys(params).length) overrides.engine_params = params;
  return { voice_id: card.querySelector('[data-role="voice"]').value, overrides };
}

// Спикеры сохраняются целиком: карточка правит голос и набор параметров, от
// которого наследуют все его реплики. Отдельные значения можно прислать готовыми
// (`overrides`) — так смена голоса сбрасывает ручки прошлого движка.
async function saveSpeakers(overrides = {}, errorElement = null) {
  if (!state.project) return false;
  const speakers = {};
  document.querySelectorAll('#voice-configs .card').forEach((card) => {
    const key = card.dataset.voice;
    speakers[key] = overrides[key] || readSpeakerCard(card);
  });
  try {
    const project = await api(`/api/projects/${state.project.id}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ speakers }),
    });
    applyProject(project, { speakers: false });
    return true;
  } catch (error) {
    showAlert(errorElement || $('replica-error'), error.message);
    return false;
  }
}

// Сохранение с пересборкой панели: правка меняет и источники значений, и набор
// кнопок «сброс», а после неудачной записи перерисовка возвращает карточку к
// тому, что действительно лежит в базе.
async function saveAndRenderSpeakers() {
  await saveSpeakers();
  renderVoiceConfigs();
}

function updateGenerateButton() {
  const speakers = (state.project && state.project.speakers) || [];
  const assigned = speakers.length > 0 && speakers.every((speaker) => speaker.voice_id);
  const button = $('btn-generate');
  button.disabled = !(state.modelReady && assigned)
    || Boolean(state.jobId) || state.sourceDirty;
  // Кнопка отмены живёт ровно столько, сколько живёт задача.
  $('btn-cancel-job').hidden = !state.jobId;
  if (!state.modelReady) button.title = 'Модель ещё загружается';
  else if (!speakers.length) button.title = 'Сначала разберите текст на реплики';
  else if (state.sourceDirty) button.title = 'Текст изменён — нажмите «Применить и разобрать»';
  else if (!assigned) button.title = 'Выберите голос для каждого участника диалога';
  else button.title = '';
  // Кнопка экспорта живёт по другому условию — проекта с репликами, а не модели:
  // выгрузка ничего не синтезирует и от готовности движка не зависит.
  updateExportButtons();
}

// --- таймлайн проекта ----------------------------------------------------------
// Таймлайн не считает время сам: start/end/длительность приходят из `/timeline`,
// где они посчитаны по реальным длительностям активных take'ов. Своя арифметика
// на клиенте (по символам или по числу реплик) разошлась бы с файлом на первом же
// рендере — а обещать пользователю место звука в файле нужно точно.
const TIMELINE_MIN_PPS = 6;    // пикселей на секунду, когда трек длинный
const TIMELINE_MAX_PPS = 240;  // …и когда он совсем короткий
const TIMELINE_MIN_BLOCK = 6;  // минимум для сегмента: по нему всё равно надо кликнуть

function fmtClock(seconds) {
  const total = Math.max(Math.floor(Number(seconds) || 0), 0);
  const minutes = Math.floor(total / 60);
  return `${String(minutes).padStart(2, '0')}:${String(total % 60).padStart(2, '0')}`;
}

// Дорожки таймлайна: по строке на спикера, сегменты — подряд по start_sec.
// Реплика без готового звука остаётся пустым штрихованным блоком: её надо видеть,
// чтобы понять, что именно ещё не сгенерировано.
function timelineLanes(data) {
  const lanes = new Map();
  const order = [];
  (data.speakers || []).forEach((speaker) => {
    lanes.set(speaker.key, { key: speaker.key, label: speaker.label || speaker.key, voice: speaker.voice_name || '', segments: [] });
    order.push(speaker.key);
  });
  (data.replicas || []).forEach((segment) => {
    if (!lanes.has(segment.speaker)) {
      lanes.set(segment.speaker, { key: segment.speaker, label: segment.speaker_label || segment.speaker, voice: '', segments: [] });
      order.push(segment.speaker);
    }
    const takes = segment.takes || [];
    const note = segment.has_audio
      ? ''
      : takes.length
        ? 'файл take потерян'
        : 'нет take';
    lanes.get(segment.speaker).segments.push({
      index: segment.index,
      start: Number(segment.start_sec) || 0,
      duration: Number(segment.duration_sec) || 0,
      hasAudio: Boolean(segment.has_audio),
      note,
      takes: takes.length,
      activeTake: segment.take_id,
      title: timelineSegmentTitle(segment, note),
    });
  });
  return order.map((key) => lanes.get(key));
}

function timelineSegmentTitle(segment, note) {
  const lines = [
    `Replica ${segment.index + 1} · ${segment.speaker_label || segment.speaker}`,
    `${fmtClock(segment.start_sec)} — ${fmtClock(segment.end_sec)} (${Number(segment.duration_sec).toFixed(2)} с)`,
    `пауза ${segment.pause_ms} мс`,
    `take: ${segment.take_id === null || segment.take_id === undefined
      ? 'нет' : `${segment.take_label || 'вариант'} (#${segment.take_id})`}`,
    note ? `статус: ${note}` : 'статус: готово',
  ];
  return lines.join('\n');
}

// Подписи оси: шаг выбирается так, чтобы на дорожке было 6–12 отметок. Своих
// значений не выдумываем — берём круглые секунды/минуты.
const TIMELINE_STEPS = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800];

function timelineTicks(duration) {
  if (!(duration > 0)) return [];
  const step = TIMELINE_STEPS.find((value) => duration / value <= 12)
    || Math.ceil(duration / 10);
  const ticks = [];
  for (let at = 0; at <= duration + 1e-6; at += step) ticks.push(at);
  return ticks;
}

function renderTimeline() {
  const box = $('timeline-body');
  const label = $('timeline-duration');
  const refresh = $('timeline-refresh');
  if (!box) return;
  refresh.disabled = state.timelineBusy;
  showAlert($('timeline-error'), state.timelineError || '');

  const data = state.timeline;
  if (state.timelineBusy && !data) {
    label.textContent = '…';
    box.innerHTML = '<span class="muted">читаю длительности…</span>';
    return;
  }
  if (!data) {
    label.textContent = '—';
    box.innerHTML = '<span class="muted">Разберите текст — таймлайн покажет, где какая реплика звучит.</span>';
    return;
  }
  const duration = Number(data.duration_sec) || 0;
  const lanes = timelineLanes(data);
  const hasSegments = (data.replicas || []).length > 0;
  label.textContent = duration > 0
    ? `${fmtClock(duration)} · ${duration.toFixed(1)} с`
    : (hasSegments ? 'звука ещё нет · 00:00' : '—');

  if (!lanes.length || !hasSegments) {
    box.innerHTML = '<span class="muted">Реплик нет: вставьте диалог в Source / Advanced и нажмите «Применить и разобрать».</span>';
    return;
  }

  // Масштаб общий для всех дорожек: сегменты соседних спикеров должны стоять
  // друг под другом. Короткий трек растягивается на читаемую ширину, длинный
  // сжимается — дорожка прокручивается по горизонтали.
  const pxPerSec = Math.min(
    Math.max(duration > 0 ? 900 / duration : TIMELINE_MAX_PPS, TIMELINE_MIN_PPS),
    TIMELINE_MAX_PPS,
  );
  const width = Math.max(Math.ceil(duration * pxPerSec), 240);
  const ticks = timelineTicks(duration);

  const axis = `<div class="timeline-axis" style="width:${width}px">${ticks
    .map((at) => `<span class="timeline-tick" style="left:${(at * pxPerSec).toFixed(1)}px">${fmtClock(at)}</span>`)
    .join('')}</div>`;

  const rows = lanes.map((lane, laneIndex) => {
    const blocks = lane.segments.map((segment) => {
      const left = Math.max(segment.start * pxPerSec, 0);
      const blockWidth = Math.max(segment.duration * pxPerSec, TIMELINE_MIN_BLOCK);
      const classes = `timeline-block${segment.hasAudio ? '' : ' no-audio'}${laneIndex % 2 ? ' alt' : ''}`;
      const inner = blockWidth >= 34
        ? `<b>${segment.index + 1}</b>${segment.note ? `<i>${esc(segment.note)}</i>` : ''}`
        : `<b>${segment.index + 1}</b>`;
      return `<button class="${classes}" data-index="${segment.index}" data-audio="${segment.hasAudio ? '1' : '0'}"
        style="left:${left.toFixed(1)}px;width:${blockWidth.toFixed(1)}px"
        title="${esc(segment.title)}">${inner}</button>`;
    }).join('');
    const empty = lane.segments.length
      ? ''
      : '<span class="muted timeline-lane-empty">реплик нет</span>';
    return `<div class="timeline-lane" data-speaker="${esc(lane.key)}">
        <span class="timeline-name" title="${esc(`${lane.label}${lane.voice ? ` · ${lane.voice}` : ''}`)}">${esc(lane.label)}</span>
        <div class="timeline-track" style="width:${width}px">${blocks}${empty}</div>
      </div>`;
  }).join('');

  box.innerHTML = `${axis}${rows}`;
}

// Открывает инспектор сегмента, не создавая второй редактор: прокручиваем к
// карточке этой реплики, подсвечиваем её и ставим фокус на главное действие —
// прослушать готовое звучание или сгенерировать его, если звука ещё нет.
function openReplicaInspector(index) {
  const card = document.querySelector(`#replica-cards .replica-card[data-index="${index}"]`);
  if (!card) {
    showAlert($('timeline-error'), 'Карточка этой реплики не найдена — обновите проект.');
    return;
  }
  showAlert($('timeline-error'), '');
  card.classList.add('inspected');
  if (typeof card.scrollIntoView === 'function') {
    card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }
  const target = card.querySelector('[data-role="play"]')
    || card.querySelector('[data-role="regen"]');
  if (target && typeof target.focus === 'function') target.focus({ preventScroll: true });
}

async function refreshTimeline() {
  if (!state.project) {
    state.timeline = null;
    state.timelineError = null;
    renderTimeline();
    return;
  }
  state.timelineBusy = true;
  state.timelineError = null;
  renderTimeline();
  const requested = state.project.id;
  try {
    const data = await api(`/api/projects/${requested}/timeline`);
    // Проект могли сменить, пока запрос был в полёте: чужой таймлайн показывать
    // нельзя — сегменты вели бы к карточкам другого диалога.
    if (!state.project || state.project.id !== requested) return;
    state.timeline = data;
    applyTimeline(data);
  } catch (error) {
    state.timelineError = error.message;
  } finally {
    state.timelineBusy = false;
    renderTimeline();
  }
}

function bindTimelineEvents() {
  const body = $('timeline-body');
  if (body) {
    body.addEventListener('click', (event) => {
      const block = event.target.closest('.timeline-block');
      if (block) openReplicaInspector(parseInt(block.dataset.index, 10));
    });
  }
  const refresh = $('timeline-refresh');
  if (refresh) refresh.addEventListener('click', refreshTimeline);
}

// --- карточки реплик (Visual editor) ------------------------------------------
// Карточка — основное место работы с диалогом: у каждой реплики видно, каким
// голосом и движком она читается, что в ней можно поправить и какие звучания у
// неё уже есть. Текст с маркерами остаётся advanced-режимом и правит то же
// самое — реплики проекта, поэтому оба представления смотрят на одни данные.

const replicaByIndex = (index) =>
  ((state.project && state.project.replicas) || []).find((item) => item.index === index) || null;

// Ползунок одной ручки реплики: подпись со значением, источник и кнопка «сброс»
// у правки этой реплики. Кнопка стоит вне <label>: внутри него клик по ней заодно
// двигал бы ползунок.
function replicaSliderHtml(replica, field, title, range, trackValue) {
  const item = settingItem(replica, field);
  const track = trackValue === undefined ? item.value : trackValue;
  return `
    <div class="field">
      <span class="param-head">${esc(title)}
        <b data-role="${field}-label">${esc(fmtParam(field, item.value))}</b>
        ${resetButtonHtml(item, field)}</span>
      <input type="range" data-role="param" data-field="${field}"
             min="${range.min}" max="${range.max}" step="${range.step}" value="${track}" />
      ${sourceNoteHtml(item, voiceName(replica.voice_id))}
    </div>`;
}

function replicaSelectHtml(replica, field, title, options) {
  const item = settingItem(replica, field);
  return `
    <div class="field">
      <span class="param-head">${esc(title)}
        ${resetButtonHtml(item, field)}</span>
      <select data-role="param" data-field="${field}">${options}</select>
      ${sourceNoteHtml(item, voiceName(replica.voice_id))}
    </div>`;
}

// Ручки движка диктует сам движок: у XTTS это температура и штраф за повторы,
// у F5 их нет. Поэтому блок собирается по паспорту движка, а не по списку полей.
function replicaEngineParamsHtml(replica, engine) {
  const info = engineInfo(engine);
  if (!info || !info.params.length) return '';
  return info.params.map((param) => {
    const item = settingItem(replica, param.name);
    const value = typeof item.value === 'number' ? item.value : param.default;
    return `
    <div class="field">
      <span class="param-head">${esc(param.label)}
        <b data-param-label="${esc(param.name)}">${fmtByStep(value, param.step)}</b>
        ${resetButtonHtml(item, 'engine_params', param.name, param.name)}</span>
      <input type="range" data-engine-param="${esc(param.name)}"
             min="${param.min}" max="${param.max}" step="${param.step}" value="${value}" />
      ${param.hint ? `<span class="muted">${esc(param.hint)}</span>` : ''}
      ${sourceNoteHtml(item, voiceName(replica.voice_id))}
    </div>`;
  }).join('');
}

function renderReplicaCards() {
  const box = $('replica-cards');
  const replicas = (state.project && state.project.replicas) || [];
  // Разметка пересобирается целиком, поэтому прослушивание, начатое до
  // обновления, надо остановить: иначе кнопка осталась бы в состоянии «стоп»
  // у строки, которой уже нет.
  stopTake();
  if (!replicas.length) {
    box.innerHTML = '<span class="muted">Реплик нет: вставьте диалог в Source / Advanced и нажмите «Применить и разобрать».</span>';
    return;
  }
  box.innerHTML = replicas.map(replicaCardHtml).join('');
}

// История звучаний реплики: вариант — это готовое аудио, поэтому переключение
// между ними мгновенное и без повторного синтеза.
function takesHtml(replica) {
  const takes = replica.takes || [];
  if (!takes.length) return '';
  const rows = takes.map((take) => {
    const playing = state.takeKey === `${replica.index}:${take.id}`;
    return `
      <div class="take-block">
        <div class="variant-row" data-take="${take.id}">
          <button class="tiny" data-role="take-play">${playing ? 'стоп' : 'слушать'}</button>
          <span class="variant-label">${esc(take.label || 'вариант')}${take.active ? ' · активно' : ''}</span>
          <span class="muted">${Number(take.duration_sec || 0).toFixed(1)} с · ${esc(seedText(take.seed))}</span>
          ${takeWarnings(take.quality)}
          ${qaNote(take.qa)}
          ${take.active ? '' : '<button class="tiny" data-role="take-pick">поставить</button>'}
        </div>
        ${takeDiagnostics(take.quality)}
      </div>`;
  }).join('');
  return `<div class="variant-list">${rows}</div>`;
}

function replicaCardHtml(replica) {
  const index = replica.index;
  const label = replica.label || speakerLabel(replica.speaker);
  const voice = voiceById(replica.voice_id);
  const engine = replica.engine || (voice ? voice.engine : '');
  const takes = replica.takes || [];
  const active = takes.find((take) => take.active) || null;
  const speaker = speakerByKey(replica.speaker);
  const pause = settingItem(replica, 'pause_override_ms').value;
  const options = state.voices
    .map((item) => `<option value="${esc(item.id)}"${item.id === (replica.voice_override || '') ? ' selected' : ''}>${esc(item.name)}</option>`)
    .join('');
  const takeLine = !takes.length
    ? 'Take: ещё не синтезировалась'
    : active
      ? `Take: ${takes.indexOf(active) + 1} из ${takes.length} · ${esc(active.label || '')}`
      : `Take: ${takes.length} (активный не выбран)`;
  return `
    <div class="card replica-card${state.replicaBusy === index ? ' busy' : ''}" data-index="${index}">
      <div class="voice-top">
        <div class="voice-name">
          <h3>${esc(label)} · Replica ${index + 1}</h3>
          <p>Voice: ${esc(voiceName(replica.voice_id))} · Engine: ${esc(engine ? engineLabel(engine) : '—')}</p>
        </div>
        ${engine ? `<span class="engine-tag${engine === ENGINE_F5 ? '' : ' violet'}" title="${esc(engineNote(engine))}">${esc(engine.toUpperCase())}</span>` : ''}
      </div>
      <p class="ref-text" title="${esc(replica.text)}">${esc(replica.text)}</p>

      <div class="row between" style="margin-bottom: 10px">
        <span class="row">
          <button class="tiny" data-role="play">Play</button>
          <button class="tiny" data-role="regen"${state.replicaBusy === index ? ' disabled' : ''}>Regenerate</button>
          ${state.replicaBusy === index
            ? '<button class="tiny ghost danger" data-role="regen-cancel">Отменить</button>'
            : ''}
          <button class="tiny" data-role="preview">Что услышит модель</button>
        </span>
        <span class="muted">${takeLine}</span>
      </div>

      <div class="field">
        <span>Голос реплики</span>
        <select data-role="voice">
          <option value="">— как у спикера: ${esc(voiceName(replica.inherited_voice_id))} —</option>
          ${options}
        </select>
        <span class="muted">Свой голос этой реплики; остальные читает голос спикера.</span>
      </div>

      ${replicaSliderHtml(replica, 'speed', 'Speed', PARAM_RANGES.speed)}
      ${replicaSliderHtml(replica, 'pause_override_ms', 'Pause before', { min: 0, max: 5000, step: 50 }, pause === null ? 400 : pause)}

      <details class="advanced"${state.openDetails[index] ? ' open' : ''}>
        <summary>Ещё настройки реплики</summary>
        ${engine === ENGINE_F5 ? `
          ${replicaSliderHtml(replica, 'cfg_strength', 'CFG strength', PARAM_RANGES.cfg_strength)}
          ${replicaSelectHtml(replica, 'nfe_step', 'NFE steps', nfeOptions(settingItem(replica, 'nfe_step').value))}` : ''}
        ${replicaEngineParamsHtml(replica, engine)}
        ${replicaSliderHtml(replica, 'gain_db', 'Громкость', PARAM_RANGES.gain_db)}
        ${replicaSliderHtml(replica, 'pitch_semitones', 'Питч-шифт', PARAM_RANGES.pitch_semitones)}
        ${replicaSliderHtml(replica, 'target_rms', 'Целевая громкость (RMS)', PARAM_RANGES.target_rms)}
      </details>

      ${takesHtml(replica)}

      ${replicaPreviewHtml(replica)}
    </div>`;
}

// --- прослушивание вариантов реплик -------------------------------------------
// Активный вариант — то, что попало в файл; «Play» в шапке карточки играет
// именно его, а список ниже даёт сравнить с прежними звучаниями.
function playActiveTake(index) {
  const replica = replicaByIndex(index);
  const active = replica && (replica.takes || []).find((take) => take.active);
  if (!active) {
    showAlert($('replica-error'), 'У реплики ещё нет звучания: нажмите Regenerate.');
    return;
  }
  toggleTake(index, active.id);
}

function toggleTake(index, takeId) {
  const player = $('take-player');
  const key = `${index}:${takeId}`;
  if (state.takeKey === key && !player.paused) {
    player.pause();
    return;
  }
  state.takeKey = key;
  player.src = `/api/projects/${state.project.id}/replicas/${index}/takes/${takeId}/audio`;
  markTakePlaying();
  // play() отказывается промисом, а не исключением: молча оставить это нельзя —
  // кнопка показывала бы «стоп» у варианта, который не звучит.
  player.play().catch((error) => {
    state.takeKey = null;
    markTakePlaying();
    showAlert($('replica-error'), `Не удалось воспроизвести вариант: ${error.message}`);
  });
}

function markTakePlaying() {
  document.querySelectorAll('#replica-cards [data-role="take-play"]').forEach((button) => {
    const row = button.closest('.variant-row');
    const card = button.closest('.replica-card');
    button.textContent =
      state.takeKey === `${card.dataset.index}:${row.dataset.take}` ? 'стоп' : 'слушать';
  });
  document.querySelectorAll('#replica-cards [data-role="play"]').forEach((button) => {
    const card = button.closest('.replica-card');
    const replica = replicaByIndex(parseInt(card.dataset.index, 10));
    const active = replica && (replica.takes || []).find((take) => take.active);
    const playing = Boolean(active) && state.takeKey === `${card.dataset.index}:${active.id}`;
    button.textContent = playing ? 'Stop' : 'Play';
  });
}

function stopTake() {
  const player = $('take-player');
  player.pause();
  player.removeAttribute('src');
  state.takeKey = null;
}

// --- правки реплики -----------------------------------------------------------
// Правка точечная: одна ручка не затирает остальные правки карточки. `null` в
// значении означает «вернуть наследуемое у спикера» — так работает «сброс».
async function patchReplica(index, payload) {
  if (!state.project) return;
  showAlert($('replica-error'), '');
  try {
    const data = await api(`/api/projects/${state.project.id}/replicas/${index}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    setReplica(data.replica);
  } catch (error) {
    showAlert($('replica-error'), error.message);
  }
}

// Ответ API содержит реплику целиком — правки, голос, движок и варианты, поэтому
// перезапрашивать проект ради одной карточки не нужно.
function setReplica(replica) {
  if (!state.project) return;
  const position = state.project.replicas.findIndex((item) => item.index === replica.index);
  if (position < 0) return;
  state.project.replicas[position] = replica;
  // Голос или параметры реплики могли измениться — прежний preview устарел.
  delete state.replicaPreview[replica.index];
  renderReplicaCards();
  renderParseSummary();
}

async function regenerateReplica(index) {
  if (!state.project || state.replicaBusy !== null) return;
  const replica = replicaByIndex(index);
  if (!replica || !replica.voice_id) {
    showAlert($('replica-error'), 'Сначала выберите голос для этой реплики.');
    return;
  }
  showAlert($('replica-error'), '');
  state.replicaBusy = index;
  state.replicaJobId = null;
  renderReplicaCards();
  try {
    const job = await api(`/api/projects/${state.project.id}/replicas/${index}/regenerate`, {
      method: 'POST',
    });
    state.replicaJobId = job.job_id;
    startTakePoll(job.job_id);
  } catch (error) {
    state.replicaBusy = null;
    renderReplicaCards();
    showAlert($('replica-error'), error.message);
  }
}

// Отмена пересинтеза: реплика остаётся с прежним звучанием — вариант в проект
// сохраняется только после успешного синтеза, так что терять нечего.
async function cancelReplicaRegen() {
  const jobId = state.replicaJobId;
  if (!jobId) return;
  showAlert($('replica-error'), '');
  try {
    await cancelJob(jobId);
    showAlert($('replica-error'), 'Отменяю пересинтез', 'info');
  } catch (error) {
    showAlert($('replica-error'), error.message);
  }
}

// Синтез идёт через общий serial worker, поэтому завершения ждём опросом задачи:
// пока она считается, карточка помечена занятой, а список вариантов не трогаем —
// иначе он разошёлся бы с тем, что лежит на диске.
function startTakePoll(jobId) {
  clearInterval(state.replicaTimer);
  state.replicaTimer = setInterval(() => pollTakeJob(jobId), 1500);
}

async function pollTakeJob(jobId) {
  try {
    const job = await api(`/api/jobs/${jobId}`);
    if (job.status === 'queued' || job.status === 'processing') return;
    clearInterval(state.replicaTimer);
    state.replicaTimer = null;
    state.replicaBusy = null;
    state.replicaJobId = null;
    if (job.status === 'cancelled') {
      // Опрос прекращён: задача отменена, и прогресс по ней больше не идёт.
      renderReplicaCards();
      showAlert($('replica-error'), 'Пересинтез отменён', 'info');
      return;
    }
    if (job.status === 'error') {
      renderReplicaCards();
      showAlert($('replica-error'), job.error || 'Реплика не пересинтезировалась');
      return;
    }
    // Новое звучание сохранено вариантом реплики — забираем проект целиком,
    // чтобы карточка показала его активным, а прежнее осталось в списке.
    await refreshProject();
  } catch (error) {
    clearInterval(state.replicaTimer);
    state.replicaTimer = null;
    state.replicaBusy = null;
    state.replicaJobId = null;
    renderReplicaCards();
    showAlert($('replica-error'), error.message);
  }
}

// Выбор варианта синтеза не запускает: он уже готов, и сравнение двух версий
// на слух не должно превращаться в ожидание модели.
async function selectTake(index, takeId) {
  if (!state.project || state.replicaBusy !== null) return;
  showAlert($('replica-error'), '');
  try {
    const data = await api(
      `/api/projects/${state.project.id}/replicas/${index}/takes/${takeId}`,
      { method: 'POST' },
    );
    setReplica(data.replica);
    // Активный take сменился — его длительность могла быть другой, и хвост
    // таймлайна обязан пересчитаться: границы приходят из `/timeline`, а не из
    // прежнего ответа.
    await refreshTimeline();
  } catch (error) {
    showAlert($('replica-error'), error.message);
  }
}

// --- генерация диалога --------------------------------------------------------
// Сборка идёт по проекту: реплики, голоса и правки уже лежат в базе, поэтому
// перед запуском сохраняем карточки спикеров (в них могли печатать только что)
// и ставим задачу рендера. Отдельной задачи «по тексту» больше нет — текст
// попадает в проект разбором.
async function generate() {
  if (!state.project) return;
  showAlert($('job-error'), '');
  if (!(await saveSpeakers($('job-error')))) return;

  const payload = {
    pause_ms: parseInt($('pause').value, 10),
    cross_fade_duration: parseFloat($('crossfade').value),
    auto_accent: $('auto-accent').checked,
    qa: $('qa').value,
    output_format: $('output-format').value,
    chunk_strategy: $('chunk-strategy').value,
  };

  $('btn-generate').disabled = true;
  $('player').hidden = true;
  $('download-link').hidden = true;
  // Готовый файл относится к прошлому прогону — до готовности нового он неактуален.
  stopTake();
  setProgress(0);

  try {
    const job = await api(`/api/projects/${state.project.id}/render`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    state.jobId = job.job_id;
    $('job-status').textContent = `задача ${job.job_id}: в очереди`;
    $('btn-cancel-job').hidden = false;
    state.pollTimer = setInterval(pollJob, 1500);
    watchEngines();
  } catch (error) {
    showAlert($('job-error'), error.message);
    state.jobId = null;
    updateGenerateButton();
  }
}

// Хвост «≈ 2 мин 40 сек» к строке статуса. Строку собирает бэкенд (поле
// `eta_text` в ответе GET /api/jobs/{id}): русские формы числительных — не дело
// UI. Если ответ пришёл без неё (старый бэкенд), число секунд показывается как было.
function etaLabel(job) {
  if (job.eta_text) return `, ≈ ${job.eta_text}`;
  if (job.eta_sec) return `, осталось ~${Math.round(job.eta_sec)} с`;
  return '';
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
      // ETA приходит с бэкенда уже строкой («2 мин 40 сек»): русские формы
      // числительных считает backend/eta.py, а не UI. `eta_sec` оставлен
      // запасным вариантом для старых ответов без `eta_text`.
      const eta = etaLabel(job);
      $('job-status').textContent = job.cancel_requested
        ? 'останавливаю на безопасной точке…'
        : `${job.message}${eta}`;
      return;
    }
    clearInterval(state.pollTimer);
    state.pollTimer = null;
    state.jobId = null;
    $('btn-cancel-job').hidden = true;
    if (job.status === 'cancelled') {
      // Опрос прекращён: задача больше не живая, и прогресс по ней не идёт.
      $('job-status').textContent = 'отменено';
      setProgress(0);
      updateGenerateButton();
      return;
    }
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
      // Куски рендера сохранены вариантами реплик проекта — забираем проект,
      // чтобы карточки показали новое звучание активным, а прежнее осталось
      // в их истории.
      await refreshProject();
    }
    updateGenerateButton();
  } catch (error) {
    clearInterval(state.pollTimer);
    state.pollTimer = null;
    state.jobId = null;
    $('btn-cancel-job').hidden = true;
    showAlert($('job-error'), error.message);
    updateGenerateButton();
  }
}

// Отмена идущего рендера. Кнопка сразу показывает, что запрос ушёл, а состояние
// `cancelled` придёт ответом на ближайший опрос — отдельного запроса статуса не нужно.
async function cancelRender() {
  const jobId = state.jobId;
  if (!jobId) return;
  const button = $('btn-cancel-job');
  button.disabled = true;
  button.textContent = 'Отменяю…';
  try {
    await cancelJob(jobId);
    $('job-status').textContent = 'останавливаю на безопасной точке…';
  } catch (error) {
    showAlert($('job-error'), error.message);
  } finally {
    button.disabled = false;
    button.textContent = 'Отменить';
  }
}

// --- экспорт и импорт проекта ---------------------------------------------------
// Экспорт — обычное скачивание файла, а не задача очереди: ни один из этих
// вариантов не синтезирует, файл собирается из уже готовых take'ов и уходит
// ответом. Поэтому здесь нет опроса статуса — только загрузка и ошибка.
const EXPORT_URLS = {
  archive: (id) => `/api/projects/${id}/export`,
  wav: (id) => `/api/projects/${id}/export/audio?format=wav`,
  mp3: (id) => `/api/projects/${id}/export/audio?format=mp3`,
  replicas: (id) => `/api/projects/${id}/export/replicas`,
  stems: (id) => `/api/projects/${id}/export/stems`,
  transcript: (id) => `/api/projects/${id}/export/transcript`,
  srt: (id) => `/api/projects/${id}/export/subtitles?format=srt`,
  vtt: (id) => `/api/projects/${id}/export/subtitles?format=vtt`,
};

const EXPORT_FALLBACK_NAMES = {
  archive: 'project.ttsproject',
  wav: 'dialogue.wav',
  mp3: 'dialogue.mp3',
  replicas: 'replicas.zip',
  stems: 'stems.zip',
  transcript: 'transcript.json',
  srt: 'subtitles.srt',
  vtt: 'subtitles.vtt',
};

function updateExportButtons() {
  const ready = Boolean(state.project && (state.project.replicas || []).length);
  const button = $('btn-export-project');
  if (!button) return;
  button.disabled = !ready || state.exportBusy === true;
  button.title = ready ? '' : 'Сначала разберите текст на реплики';
}

// Имя файла берём из Content-Disposition: бэкенд уже сделал его безопасным и
// осмысленным (имя проекта), и второй способ его собирать разошёлся бы с первым.
function downloadName(response, fallback) {
  const header = response.headers.get('Content-Disposition') || '';
  const utf = header.match(/filename\*=UTF-8''([^;]+)/i);
  if (utf) {
    try { return decodeURIComponent(utf[1]); } catch (_) { /* оставим запасное имя */ }
  }
  const plain = header.match(/filename="?([^";]+)"?/i);
  return plain ? plain[1] : fallback;
}

async function downloadExport(url, fallbackName) {
  const response = await fetch(url);
  if (!response.ok) {
    let detail = `Ошибка ${response.status}`;
    try {
      const body = await response.json();
      if (body.detail) detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail);
    } catch (_) { /* тело не JSON — оставляем текст статуса */ }
    throw new Error(detail);
  }
  const blob = await response.blob();
  const href = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = href;
  link.download = downloadName(response, fallbackName);
  document.body.appendChild(link);
  link.click();
  link.remove();
  // Ссылку освобождаем после клика: до этого браузер ещё читает blob.
  setTimeout(() => URL.revokeObjectURL(href), 10000);
}

async function exportProject() {
  if (!state.project) {
    showAlert($('export-error'), 'Проект ещё не создан — вставьте текст и разберите его.');
    return;
  }
  const kind = $('export-kind').value;
  const build = EXPORT_URLS[kind];
  if (!build) return;
  showAlert($('export-error'), '');
  state.exportBusy = true;
  updateExportButtons();
  $('export-status').textContent = 'готовлю файл…';
  try {
    await downloadExport(build(state.project.id), EXPORT_FALLBACK_NAMES[kind]);
    $('export-status').textContent = 'файл скачан';
  } catch (error) {
    $('export-status').textContent = '—';
    showAlert($('export-error'), error.message);
  } finally {
    state.exportBusy = false;
    updateExportButtons();
  }
}

async function importProject(file) {
  if (!file) return;
  showAlert($('export-error'), '');
  state.exportBusy = true;
  updateExportButtons();
  $('export-status').textContent = `загружаю ${file.name}…`;
  try {
    const form = new FormData();
    form.append('file', file);
    const project = await api('/api/projects/import', { method: 'POST', body: form });
    // Импорт мог создать голоса — список перечитываем, иначе карточки реплик
    // покажут «не выбран» у голоса, который уже есть на этой машине.
    await loadVoices();
    rememberProject(project.id);
    $('dialogue').value = project.source_text || '';
    state.sourceDirty = false;
    applyProject(project);
    $('export-status').textContent = `открыт проект «${project.name}»`;
  } catch (error) {
    $('export-status').textContent = '—';
    showAlert($('export-error'), `Импорт не выполнен: ${error.message}`);
  } finally {
    state.exportBusy = false;
    updateExportButtons();
  }
}

function bindExportEvents() {
  const exportButton = $('btn-export-project');
  if (exportButton) exportButton.addEventListener('click', exportProject);
  const importButton = $('btn-import-project');
  const importInput = $('import-file');
  if (importButton && importInput) {
    importButton.addEventListener('click', () => importInput.click());
    importInput.addEventListener('change', async () => {
      const file = importInput.files && importInput.files[0];
      // Поле очищаем до загрузки: повторный выбор того же файла иначе не поднял бы
      // событие change, и импорт «не сработал бы» второй раз.
      importInput.value = '';
      await importProject(file);
    });
  }
  updateExportButtons();
}

// --- отметки и подписи звучаний ------------------------------------------------
// Отметка проверки качества. Показывается только у тех кусков, что не прошли её
// полностью: «проверено и прошло» — это норма, и подпись у каждой реплики была бы
// шумом, а принятое без полной проверки должно быть видно — вместе со степенью
// расхождения, чтобы решать «оставить или перегенерировать» осознанно.
// Причины дешёвого отбора уходят в подсказку: по ним видно, за что кусок вообще
// попал в расшифровку.
const SCREEN_REASONS = {
  empty: 'пустое аудио',
  too_short: 'слишком короткий кусок',
  duration_short: 'короче текста',
  duration_long: 'длиннее текста',
  silence: 'много тишины',
  clipping: 'перегруз',
  level: 'аномальный уровень',
  repeat: 'повтор участка',
};

function qaNote(qa) {
  if (!qa || qa.status === 'passed') return '';
  const reasons = {
    budget_exhausted: 'время проверки вышло',
    attempts_exhausted: 'попытки исчерпаны',
    unavailable: 'расшифровка не удалась',
  };
  const wer = qa.wer === null || qa.wer === undefined ? '' : `WER ${qa.wer.toFixed(2)} · `;
  const reason = reasons[qa.status] || qa.status;
  const screened = ((qa.screening && qa.screening.reasons) || [])
    .map((code) => SCREEN_REASONS[code] || code);
  const title = `Попыток: ${qa.attempts}` + (screened.length ? `. Отбор: ${screened.join(', ')}` : '');
  return `<span class="replica-qa muted warn" title="${esc(title)}">` +
    `⚠ ${esc(wer + reason)}</span>`;
}

const QA_MODE_LABELS = { off: 'выключена', smart: 'smart', strict: 'strict' };

// --- диагностика take'а --------------------------------------------------------
// Обычному пользователю показываются только предупреждения: «клиппинг», «много
// тишины», «необычно длинная реплика». Числа (WER, LUFS, peak, RMS) уходят в
// раскрывающуюся «Диагностику» — по ним видно техническую причину, но решение
// «оставить или перегенерировать» принимается на слух. Автоматического балла
// естественности у take'а нет намеренно: из этих измерений он не выводится.
function takeWarnings(quality) {
  const warnings = (quality && quality.warnings) || [];
  return warnings
    .map((item) => `<span class="take-warn muted warn" title="${esc(item.text)}">⚠ ${esc(item.text)}</span>`)
    .join('');
}

function takeDiagnostics(quality) {
  if (!quality) return '';
  const number = (value, digits, suffix = '') =>
    typeof value === 'number' && Number.isFinite(value) ? `${value.toFixed(digits)}${suffix}` : null;
  const percent = (value, digits) =>
    typeof value === 'number' && Number.isFinite(value) ? `${(value * 100).toFixed(digits)} %` : null;
  const count = (value) => (Number.isInteger(value) ? String(value) : null);
  const rows = [
    ['WER', number(quality.wer, 2)],
    ['Знаков', count(quality.chars)],
    ['Длительность', number(quality.duration_sec, 2, ' с')],
    ['Секунд на знак', number(quality.duration_per_char, 3)],
    ['Peak', number(quality.peak_dbfs, 1, ' dBFS')],
    ['RMS', number(quality.rms_dbfs, 1, ' dBFS')],
    ['LUFS', number(quality.lufs, 1)],
    ['Доля тишины', percent(quality.silence_ratio, 1)],
    ['Клиппинг', typeof quality.clipping === 'boolean' ? (quality.clipping ? 'есть' : 'нет') : null],
    ['Доля клиппинга', percent(quality.clipping_ratio, 2)],
    ['Попыток QA', count(quality.qa_attempts)],
    ['Режим QA', quality.qa_mode ? QA_MODE_LABELS[quality.qa_mode] || quality.qa_mode : null],
    ['Причины отбора', (quality.screening_reasons || [])
      .map((code) => SCREEN_REASONS[code] || code).join(', ') || null],
  ].filter((row) => row[1] !== null && row[1] !== undefined && row[1] !== '');
  if (!rows.length) return '';
  const items = rows
    .map(([label, value]) => `<div><dt>${esc(label)}</dt><dd>${esc(value)}</dd></div>`)
    .join('');
  return `<details class="take-diagnostics"><summary>Диагностика</summary>` +
    `<dl class="diag-list">${items}</dl></details>`;
}

function seedText(seed) {
  return seed === null || seed === undefined ? 'без сида' : `сид ${seed}`;
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
  $('btn-cancel-text-job').hidden = !state.textJobId;
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
    qa: $('text-qa').value,
    output_format: $('text-output-format').value,
    chunk_strategy: $('text-chunk-strategy').value,
  };

  $('btn-render-text').disabled = true;
  $('text-player').hidden = true;
  $('text-download-link').hidden = true;
  // Отметка относится к прошлому файлу — до готовности нового она неактуальна.
  $('text-job-note').hidden = true;
  setTextProgress(0);

  try {
    const job = await api('/api/render-text', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    state.textJobId = job.job_id;
    $('text-job-status').textContent = `задача ${job.job_id}: кусков ${job.total_replicas}`;
    $('btn-cancel-text-job').hidden = false;
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
      const eta = etaLabel(job);
      $('text-job-status').textContent = job.cancel_requested
        ? 'останавливаю на безопасной точке…'
        : `кусок ${job.current_replica + 1} из ${job.total_replicas}${eta}`;
      return;
    }
    clearInterval(state.textPollTimer);
    state.textPollTimer = null;
    state.textJobId = null;
    $('btn-cancel-text-job').hidden = true;
    if (job.status === 'cancelled') {
      $('text-job-status').textContent = 'отменено';
      setTextProgress(0);
    } else if (job.status === 'error') {
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
      showTextQaNote(job.replicas);
    }
    updateTextButton();
  } catch (error) {
    clearInterval(state.textPollTimer);
    state.textPollTimer = null;
    state.textJobId = null;
    $('btn-cancel-text-job').hidden = true;
    showAlert($('text-job-error'), error.message);
    updateTextButton();
  }
}

// Отмена идущей озвучки сплошного текста — тем же эндпоинтом, что и у задач рендера.
async function cancelTextRender() {
  const jobId = state.textJobId;
  if (!jobId) return;
  const button = $('btn-cancel-text-job');
  button.disabled = true;
  button.textContent = 'Отменяю…';
  try {
    await cancelJob(jobId);
    $('text-job-status').textContent = 'останавливаю на безопасной точке…';
  } catch (error) {
    showAlert($('text-job-error'), error.message);
  } finally {
    button.disabled = false;
    button.textContent = 'Отменить';
  }
}

// У сплошного текста нет списка реплик, поэтому неполное прохождение проверки
// показываем одной строкой: сколько кусков принято без неё и насколько
// близко подошёл лучший из них.
function showTextQaNote(replicas) {
  const note = $('text-job-note');
  const all = replicas || [];
  const failed = all.filter((replica) => replica.qa && replica.qa.status !== 'passed');
  if (!failed.length) {
    note.hidden = true;
    note.classList.remove('warn');
    return;
  }
  const wers = failed
    .map((replica) => replica.qa.wer)
    .filter((wer) => wer !== null && wer !== undefined);
  const best = wers.length ? `, лучший WER ${Math.min(...wers).toFixed(2)}` : '';
  note.textContent = `Проверка качества: ${failed.length} из ${all.length} кусков приняты ` +
    `без полного прохождения${best}`;
  note.classList.add('warn');
  note.hidden = false;
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
// События обеих частей редактора реплик: карточек и исходного текста.
function bindReplicaCards() {
  const box = $('replica-cards');
  const indexOf = (element) => parseInt(element.closest('.replica-card').dataset.index, 10);

  // На движении ползунка меняется только подпись: запись в базу — на отпускании,
  // иначе каждое движение слало бы запрос на правку реплики.
  box.addEventListener('input', (event) => {
    const card = event.target.closest('.replica-card');
    if (!card) return;
    if (event.target.dataset.engineParam) {
      updateEngineParamLabel(card, event.target);
      return;
    }
    const field = event.target.dataset.field;
    if (field) {
      const label = card.querySelector(`[data-role="${field}-label"]`);
      if (label) label.textContent = fmtParam(field, parseFloat(event.target.value));
    }
  });

  box.addEventListener('change', (event) => {
    const card = event.target.closest('.replica-card');
    if (!card) return;
    const index = indexOf(event.target);
    // Голос реплики — это её собственный голос поверх голоса спикера.
    if (event.target.dataset.role === 'voice') {
      patchReplica(index, { voice_id: event.target.value });
      return;
    }
    if (event.target.dataset.engineParam) {
      patchReplica(index, { overrides: { engine_params: readEngineParams(card) } });
      return;
    }
    const field = event.target.dataset.field;
    if (!field) return;
    const value = event.target.tagName === 'SELECT'
      ? parseInt(event.target.value, 10)
      : parseFloat(event.target.value);
    patchReplica(index, { overrides: { [field]: value } });
  });

  box.addEventListener('click', (event) => {
    const card = event.target.closest('.replica-card');
    if (!card) return;
    const index = parseInt(card.dataset.index, 10);
    const reset = event.target.closest('[data-role="reset"]');
    if (reset) {
      // `null` возвращает параметр к наследуемому: у ручек движка — к значению
      // голоса, у остальных — к значению спикера.
      const field = reset.dataset.field;
      patchReplica(index, {
        overrides: field === 'engine_params'
          ? { engine_params: { [reset.dataset.param]: null } }
          : { [field]: null },
      });
      return;
    }
    if (event.target.closest('[data-role="play"]')) { playActiveTake(index); return; }
    if (event.target.closest('[data-role="regen"]')) { regenerateReplica(index); return; }
    if (event.target.closest('[data-role="regen-cancel"]')) { cancelReplicaRegen(); return; }
    if (event.target.closest('[data-role="preview"]')) { previewReplica(index); return; }
    const row = event.target.closest('.variant-row');
    if (!row) return;
    const takeId = parseInt(row.dataset.take, 10);
    if (event.target.closest('[data-role="take-play"]')) { toggleTake(index, takeId); return; }
    if (event.target.closest('[data-role="take-pick"]')) selectTake(index, takeId);
  });

  // Список карточек пересобирается после каждой правки, поэтому раскрытые
  // «ещё настройки» запоминаем: иначе блок закрывался бы на каждом ползунке.
  box.addEventListener('toggle', (event) => {
    const details = event.target.closest('.replica-card details.advanced');
    if (!details) return;
    state.openDetails[details.closest('.replica-card').dataset.index] = details.open;
  }, true);

  const takePlayer = $('take-player');
  takePlayer.addEventListener('play', markTakePlaying);
  takePlayer.addEventListener('pause', markTakePlaying);
  takePlayer.addEventListener('ended', markTakePlaying);
  // Собранный диалог и отдельный вариант реплики одновременно звучать не должны.
  $('player').addEventListener('play', stopTake);
}

function bindEvents() {
  document.querySelectorAll('.tab').forEach((tab) => {
    tab.addEventListener('click', () => switchTab(tab.dataset.tab));
  });

  // Два представления одного диалога: карточки реплик и исходный текст с маркерами.
  document.querySelectorAll('.view-switch button').forEach((button) => {
    button.addEventListener('click', () => switchDialogueView(button.dataset.view));
  });
  // Разбор — только по команде: правка текста меняет состав реплик, и молча
  // переписывать их (вместе с правками карточек и историей вариантов) нельзя.
  $('dialogue').addEventListener('input', markSourceDirty);
  $('btn-apply-source').addEventListener('click', applySource);
  bindReplicaCards();
  // Панели «что услышит модель» монтируются здесь, а списки голосов и движков
  // заполняются, когда они придут из API (renderPreviewPanels).
  renderPreviewPanels();
  $('auto-accent').addEventListener('change', () => syncPreviewAccent('dialogue-preview', 'auto-accent'));
  $('text-auto-accent').addEventListener('change', () => syncPreviewAccent('text-preview', 'text-auto-accent'));
  $('btn-generate').addEventListener('click', generate);
  $('btn-cancel-job').addEventListener('click', cancelRender);
  $('btn-reload-voices').addEventListener('click', () => loadVoices().catch((e) => alert(e.message)));
  $('btn-save-voice').addEventListener('click', createVoice);
  $('btn-recognize-voice').addEventListener('click', recognizeRefText);
  bindModelsEvents();

  $('pause').addEventListener('input', (e) => { $('pause-value').textContent = `${e.target.value} мс`; });
  // Смена стратегии меняет состав реплик: без повторного разбора карточки
  // показывали бы старые куски, поэтому текст помечается как «нужен разбор».
  $('chunk-strategy').addEventListener('change', markSourceDirty);
  $('crossfade').addEventListener('input', (e) => {
    $('crossfade-value').textContent = `${parseFloat(e.target.value).toFixed(2)} с`;
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
  $('btn-cancel-text-job').addEventListener('click', cancelTextRender);
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

  // --- предложения для словаря (вкладка «Сплошной текст») ---
  $('btn-find-suggestions').addEventListener('click', findSuggestions);
  $('btn-add-all-suggestions').addEventListener('click', addAllSuggestions);
  $('suggestions-list').addEventListener('click', (event) => {
    const card = event.target.closest('.suggestion');
    const button = event.target.closest('button');
    if (!card || !button) return;
    const index = Number(card.dataset.index);
    if (button.dataset.role === 'add') confirmSuggestion(index);
    if (button.dataset.role === 'reject') rejectSuggestion(index);
  });

  // --- вкладка «Словарь» ---
  $('btn-reload-dictionary').addEventListener('click', () => {
    loadDictionary().catch((error) => showAlert($('dictionary-error'), error.message));
  });
  $('btn-save-dict-entry').addEventListener('click', saveDictionaryEntry);
  $('btn-reset-dict-form').addEventListener('click', resetDictionaryForm);
  $('btn-dict-preview').addEventListener('click', previewDictionary);

  const dictionaryList = $('dictionary-list');
  dictionaryList.addEventListener('change', (event) => {
    const card = event.target.closest('.dict-rule');
    if (!card) return;
    const id = Number(card.dataset.entryId);
    if (event.target.dataset.role === 'enabled') updateDictionaryEntry(id, { enabled: event.target.checked });
    if (event.target.dataset.role === 'case') updateDictionaryEntry(id, { case_sensitive: event.target.checked });
  });
  dictionaryList.addEventListener('click', (event) => {
    const card = event.target.closest('.dict-rule');
    const button = event.target.closest('button');
    if (!card || !button) return;
    const id = Number(card.dataset.entryId);
    if (button.dataset.role === 'edit') editDictionaryEntry(id);
    if (button.dataset.role === 'delete') deleteDictionaryEntry(id);
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
    const role = event.target.dataset.role;
    if (role === 'benchmark-text') {
      benchmarkState(card.dataset.voiceId).text = event.target.value;
      return;
    }
    const cfg = state.preview[card.dataset.voiceId];
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
    if (role === 'benchmark-engine') {
      const engine = event.target.dataset.engine;
      const cfg = benchmarkState(voiceId);
      cfg.engines = event.target.checked
        ? cfg.engines.concat(engine)
        : cfg.engines.filter((item) => item !== engine);
      return;
    }
    if (role === 'benchmark-qa') {
      benchmarkState(voiceId).qa = event.target.value;
      return;
    }
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
    const voiceId = card.dataset.voiceId;
    if (button.dataset.role === 'preview') previewVoice(card);
    if (button.dataset.role === 'preview-cancel') cancelPreview(card);
    if (button.dataset.role === 'save-preset') saveVoicePreset(voiceId);
    if (button.dataset.role === 'compare') toggleBenchmark(voiceId);
    if (button.dataset.role === 'benchmark-run') runBenchmark(voiceId);
    if (button.dataset.role === 'benchmark-play') playBenchmarkTake(voiceId, button.dataset.url);
    if (button.dataset.role === 'benchmark-select') {
      selectBenchmarkEngine(voiceId, button.dataset.engine);
    }
    if (button.dataset.role === 'delete') {
      if (!confirm('Удалить голос? Файл референса будет стёрт.')) return;
      try {
        await api(`/api/voices/${voiceId}`, { method: 'DELETE' });
        delete state.preview[voiceId];
        stopBenchmarkPoll(voiceId);
        delete state.benchmark[voiceId];
        await loadVoices();
      } catch (error) {
        showAlert($('voice-error'), error.message);
      }
    }
  });

  // --- карточки голосов диалога ---
  // Спикер — это голос и набор параметров, от которого наследуют его реплики.
  // Значения сохраняются на отпускании ручки: карточки реплик берут из них
  // наследуемые значения, и незаписанная правка разошлась бы с базой.
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
    updateGenerateButton();
  });

  voiceConfigs.addEventListener('change', async (event) => {
    const role = event.target.dataset.role;
    if (role === 'pause-inherit') {
      const card = event.target.closest('.card');
      const range = card.querySelector('[data-role="pause-override"]');
      range.disabled = event.target.checked;
      card.querySelector('[data-role="pause-label"]').textContent = event.target.checked
        ? fmtPause(null)
        : fmtPause(parseInt(range.value, 10));
      await saveAndRenderSpeakers();
      return;
    }
    if (role === 'voice') {
      // Набор ручек диктует движок голоса: значения прошлого движка к новому не
      // относятся, поэтому у слота они сбрасываются, а синтез берёт сохранённые
      // у нового голоса (см. audio_pipeline.voice_layer). Панель после этого
      // пересобирается: набор полей тоже зависит от движка.
      const card = event.target.closest('.card');
      const speaker = readSpeakerCard(card);
      delete speaker.overrides.engine_params;
      await saveSpeakers({ [card.dataset.voice]: speaker });
      renderVoiceConfigs();
      return;
    }
    await saveAndRenderSpeakers();
  });

  voiceConfigs.addEventListener('click', async (event) => {
    const card = event.target.closest('.card');
    const reset = event.target.closest('[data-role="reset"]');
    if (!card || !reset) return;
    const speaker = speakerByKey(card.dataset.voice) || {};
    // Сброс правки слота: контрол возвращается к наследуемому значению, и
    // сохранение само видит, что поле снова совпадает с нижним слоем.
    if (reset.dataset.param === undefined) {
      setSpeakerControl(card, reset.dataset.field, inheritedValue(speaker, reset.dataset.field));
    } else {
      const input = card.querySelector(`[data-engine-param="${reset.dataset.param}"]`);
      if (input) {
        input.value = inheritedValue(speaker, reset.dataset.param);
        updateEngineParamLabel(card, input);
      }
    }
    await saveAndRenderSpeakers();
  });

  // Панель пересобирается после каждой правки, поэтому раскрытые «ещё настройки»
  // запоминаем: иначе блок закрывался бы на каждой ручке.
  voiceConfigs.addEventListener('toggle', (event) => {
    const details = event.target.closest('.card details.advanced');
    if (!details) return;
    state.speakerDetails[details.closest('.card').dataset.voice] = details.open;
  }, true);
}

// --- старт --------------------------------------------------------------------
async function init() {
  renderRecordPhrases();
  bindEvents();
  bindTimelineEvents();
  bindExportEvents();
  renderTimeline();
  // Таймер создаём до первого опроса: если всё уже готово, refreshStatus его снимет.
  state.statusTimer = setInterval(refreshStatus, 4000);
  await refreshStatus();
  try {
    // Паспорта движков — до голосов: и карточки, и форма рисуют по ним набор настроек.
    await loadEngines();
    await loadVoices();
    // Прошлый проект открывается сам: реплики, голоса и варианты живут в базе,
    // а не в разовой задаче, поэтому перезагрузка страницы их не теряет.
    await loadSavedProject();
    renderParseSummary();
    setSourceState();
  } catch (error) {
    showAlert($('voice-error'), error.message);
  }
  updateGenerateButton();
}

init();
