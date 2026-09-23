'use strict';

// Точка входа фронтенда: `index.html` подключает этот файл как `type="module"`.
// Он держит общий фундамент (состояние, запросы, хелперы движков) и грузит
// вкладку «Голоса» из voices.js. Связь двусторонняя: карточки голосов зовут
// перерисовку соседних вкладок через `loadVoices`, а события карточек — отсюда.
import {
  cancelJob,
  setPreviewStatus,
  renderVoiceCards,
  addVoiceReference,
  updateVoiceReference,
  deleteVoiceReference,
  benchmarkState,
  changeVoiceEngine,
  saveVoiceEngineParams,
  previewVoice,
  cancelPreview,
  saveVoicePreset,
  toggleBenchmark,
  runBenchmark,
  playBenchmarkTake,
  selectBenchmarkEngine,
  stopBenchmarkPoll,
  createVoice,
  recognizeRefText,
  updateNewVoiceEngineNote,
} from './voices.js';
import { renderTimeline, refreshTimeline, bindTimelineEvents } from './timeline.js';
import { updateExportButtons, downloadExport, bindExportEvents } from './project-io.js';

const $ = (id) => document.getElementById(id);

const state = {
  // Политика коротких реплик из /api/status — для подписи «Авто» в панели.
  shortPolicy: null,
  voices: [],
  engines: {},         // id -> паспорт движка из /api/engines
  engineTouched: false, // пользователь сам выбрал движок в форме нового голоса
  project: null,       // проект диалога: исходный текст, спикеры, реплики и варианты
  sourceDirty: false,  // исходный текст правили после последнего разбора
  openDetails: {},     // индекс реплики -> раскрыт ли блок «ещё настройки»
  preparedDetails: {}, // индекс реплики -> раскрыт ли блок «подготовленный текст»
  speakerDetails: {},  // ключ спикера -> раскрыт ли блок «ещё настройки голоса»
  replicaBusy: null,   // индекс реплики, которая синтезируется прямо сейчас
  replicaJobId: null,  // задача этого пересинтеза — по ней работает отмена
  replicaTimer: null,
  takeKey: null,       // "индекс:take_id" варианта, играющего в плеере реплик
  analysis: null,      // последний ответ /analysis: состояние подготовки диалога
  llm: null,           // последний ответ /linguistic-analysis: разбор локальной LLM
  analyzeBusy: false,  // анализ в полёте — кнопка заблокирована
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
  cache: null,         // последний ответ /api/cache: размеры по трём категориям
  cacheChecked: {},    // target -> отмечен ли чекбокс (сохраняется при перерисовке)
  cacheBusy: false,    // очистка в полёте — кнопка заблокирована
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
  // Вкладка «Сам себе звукорежиссер». Источник истины — backend: `project`
  // всегда свежий ответ project-эндпоинта. Локально живут только то, что
  // относится к сеансу: открытая реплика, выбранный микрофон, несохранённые
  // ползунки обработки и незагруженный на сервер дубль.
  recording: {
    project: null,
    projects: [],       // список проектов для select
    activeIndex: 0,     // индекс открытой реплики (индекс из разбора диалога)
    dialogueDirty: false, // текст правили после последнего разбора
    nameDirty: false,   // имя проекта правили и ещё не отправили
    dirtyProfiles: {},  // profile_id -> {speed, pitch, denoise} до сохранения
    deviceId: '',       // выбранный входной микрофон; '' — устройство по умолчанию
    loaded: false,      // вкладка уже открывалась: не перечитывать проект зря
    render: null,       // последний ответ /render/status (живой прогресс)
    renderTimer: null,
    recorder: null,
    stream: null,
    chunks: [],
    suffix: 'webm',
    blob: null,         // записанный, но ещё не сохранённый дубль
    url: null,
    analyser: null,
    ctx: null,
    raf: null,
    countdown: null,    // {left, timer} — отсчёт перед записью
    tick: null,
    startedAt: 0,
    peak: 0,
    busy: false,        // идёт отсчёт или запись
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
// клонирует слишком узкий кусок голосового диапазона. Кроме регистров (вопрос,
// восклицание, просьба, приказ, ирония, радость, огорчение, перечисление) список
// гарантирует «ё» в разных позициях: модель воспроизводит то, что было в
// референсе, поэтому слова с «ё» должны в нём реально звучать.
//
// Правило начитки: одна эмоция и ровный темп на всю фразу, разнообразие — между
// фразами. Внутри одной фразы интонацию не меняем: рваный темп и скачки эмоции
// портят референс сильнее, чем узкий набор звуков.
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
  // --- новые: интонация + буква «ё» в разных словах ---
  {
    label: 'Радость, восторг',
    text: 'Ты представляешь, у нас всё получилось! Ребёнок сам застегнул все пуговицы, а потом ещё и спел мне целую песню про самолёт.',
  },
  {
    label: 'Огорчение, сочувствие',
    text: 'Мне так жаль, что всё так обернулось. Она ждала этот день, а теперь идёт домой одна и не знает, что делать дальше.',
  },
  {
    label: 'Лёгкая ирония',
    text: 'Ну конечно, именно сегодня лифт снова сломан. Как будто он специально ждёт, пока актёр из соседней квартиры опять устроит репетицию.',
  },
  {
    label: 'Строгий тон, короткий приказ',
    text: 'Немедленно отдайте мне ключи от квартиры. Я всё сказал предельно ясно, и вы прекрасно понимаете, о чём идёт речь.',
  },
  {
    label: 'Перечисление, ровный ритм',
    text: 'На столе лежали ключи, кошелёк, блокнот, ручка и ещё какие-то бумаги, которые я так и не успел разобрать.',
  },
  {
    label: 'Быстрая, взволнованная речь',
    text: 'Скорее, мы опаздываем! Автобус уже подъезжает, а нам ещё нужно забрать вещи, запереть дверь и найти, куда делись ключи.',
  },
];

// Mapping «подпись фразы → ключ профиля» (UPDATE 3 §16). Одиннадцать фраз дают
// одиннадцать **разных** интонаций и записываются в **один** голос: каждая фраза
// становится отдельным `ReferenceProfile` этого голоса, а не новым голосом.
// Ключом карты стоит подпись, потому что её же пользователь видит в списке фраз:
// две параллельные таблицы «по индексу» разошлись бы на первой вставке.
// Порядок значений — порядок документа, он же порядок записи в мастере.
const RECORD_PHRASE_PROFILES = {
  'Вопрос и утверждение': 'NEUTRAL_QUESTION',
  'Восклицания, много шипящих': 'EXCLAMATION',
  'Спокойная просьба (короче)': 'CALM',
  'Только вопросы': 'QUESTION',
  'Сложные сочетания согласных': 'NEUTRAL',
  'Радость, восторг': 'DELIGHT',
  'Огорчение, сочувствие': 'SAD_SYMPATHETIC',
  'Лёгкая ирония': 'IRONIC',
  'Строгий тон, короткий приказ': 'STRICT',
  'Перечисление, ровный ритм': 'ENUMERATION',
  'Быстрая, взволнованная речь': 'EXCITED',
};

// Позиция фразы в `RECORD_PHRASES` — это её устойчивый идентификатор
// (`source_record_phrase_id`, §16). Он стабилен, пока стабилен порядок списка, а
// порядок закреплён тестом: по нему бэкенд понимает, какой профиль заменяет
// повторная запись, — «одна фраза = один профиль».
const recordPhraseId = (index) => String(index);

// Тестовые данные, а не фраза для записи. Слова подобраны так, чтобы «ё» стояла
// и в начале, и после согласной, — на них видно работу шага восстановления «ё» в
// панели «Что услышит модель». В RECORD_PHRASES её намеренно нет: искусственная
// плотность одного звука ломает ровный темп начитки, а он и определяет качество
// клонирования (см. правило выше).
const YO_STRESS_TEST_PHRASE = 'Ёж и ёлка стоят у забора, а рядом идёт актёр в жёлтом плаще: он несёт мёд и лёд, поёт песню и совсем не мёрзнет.';

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
// Режим записи «новый голос» в выборе цели. Пустая строка — потому что это не
// идентификатор голоса, и путать режим с голосом нельзя даже по имени переменной.
const RECORD_NEW_VOICE = '';
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
  } catch (error) {
    // Отмена запроса вызывающим — не сбой связи: пробрасываем её как есть, иначе
    // «переключил вкладку» выглядело бы как «сервер упал».
    if (isAbort(error)) throw error;
    // fetch отклоняется только на сетевом сбое: сервер не запущен или упал.
    // Без этой ветки пользователь видел невнятное «Failed to fetch».
    throw new Error('Бэкенд недоступен: сервер не отвечает. Запустите ./run.sh и обновите страницу.');
  }
  if (!response.ok) {
    let detail = `Ошибка ${response.status}`;
    let body = null;
    try {
      body = await response.json();
      if (body && body.detail) detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail);
    } catch (_) { /* тело не JSON — оставляем текст статуса */ }
    const error = new Error(detail);
    // Структурированная причина отказа (409 монтажа: {error, missing, message})
    // нужна вызывающему как данные, а не как строка: иначе список незаписанных
    // реплик пришлось бы разбирать обратно из текста ошибки.
    error.status = response.status;
    error.detail = body && body.detail !== undefined ? body.detail : null;
    throw error;
  }
  return response.status === 204 ? null : response.json();
}

// --- отмена устаревших запросов (F-W1) ----------------------------------------
// У каждого вида запроса «в полёте» остаётся только последний. Без этого быстрое
// переключение вкладок давало «догоняющие» ответы: медленный ответ по прежнему
// проекту приходил после быстрого по новому и перетирал уже показанное состояние.
// Замена контроллера в словаре сама чистит предыдущий — держать больше одного на
// вид и не нужно.
const inflight = new Map();

function staleSignal(kind) {
  const previous = inflight.get(kind);
  if (previous) previous.abort();
  const controller = new AbortController();
  inflight.set(kind, controller);
  return controller.signal;
}

function isAbort(error) {
  return Boolean(error) && error.name === 'AbortError';
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

// Читает ли движок референс голоса. Спрашивается у паспорта, а не у списка id:
// новый движок без клонирования объявляет это одним полем (см. supports_cloning
// в backend/engines/base.py), и интерфейсу не нужно про него знать.
function usesReference(engineId) {
  const info = engineInfo(engineId);
  return !info || info.supports_cloning;
}

// Предупреждение о движке без клонирования: он синтезирует встроенным голосом
// модели, и записанный референс в синтез не попадает вовсе — пайплайн даже не
// зовёт резолвер профилей. Молчать об этом нельзя: референс выглядел бы
// используемым, а его настройки — действующими.
function cloningNote(engineId) {
  if (usesReference(engineId)) return '';
  return `${engineLabel(engineId)} синтезирует встроенным голосом модели — записанный референс не используется`;
}

const fmtByStep = (value, step) =>
  (step >= 1 ? String(Math.round(value)) : value.toFixed(step < 0.1 ? 2 : 1));

// Режимы работы движка (черновик/качество/эксперимент) объявлены паспортом: у F5 и
// XTTS их нет, и переключателя им не рисуется — обещать выбор, который ничего не
// меняет, нельзя. Режим — такая же ручка, как температура: он живёт в том же
// словаре `engine_params` под ключом `mode` и уходит и в запрос, и в голос.
function engineModeFor(engineId, values) {
  const info = engineInfo(engineId);
  if (!info || !info.modes.length) return '';
  const raw = values ? values.mode : undefined;
  if (typeof raw === 'string' && info.modes.some((mode) => mode.id === raw)) return raw;
  return info.default_mode || info.modes[0].id;
}

// Пресет выбранного режима: значения объявленных ручек, которые режим ставит сам.
function modePreset(engineId, values) {
  const info = engineInfo(engineId);
  const mode = info && info.modes.find((item) => item.id === engineModeFor(engineId, values));
  return (mode && mode.overrides) || {};
}

function engineModesHtml(engineId, values) {
  const info = engineInfo(engineId);
  if (!info || !info.modes.length) return '';
  const current = engineModeFor(engineId, values);
  const mode = info.modes.find((item) => item.id === current) || {};
  return `
    <label class="field">
      <span>Режим работы</span>
      <select data-engine-mode>${info.modes
        .map((item) => `<option value="${esc(item.id)}"${item.id === current ? ' selected' : ''}>${esc(item.label)}</option>`)
        .join('')}</select>
      <span class="muted">${esc(mode.hint || '')}</span>
    </label>`;
}

// Значения ручек движка: то, что уже сохранено у голоса или набрано в карточке,
// иначе — пресет выбранного режима, а за ним дефолт из паспорта. Ручки чужого
// движка отбрасываются: у XTTS нет nfe_step, а у F5 нет temperature, и тащить их
// через запрос незачем.
function engineParamsFor(engineId, values) {
  const info = engineInfo(engineId);
  const result = {};
  if (!info) return result;
  const preset = modePreset(engineId, values);
  info.params.forEach((param) => {
    const raw = values ? values[param.name] : undefined;
    if (typeof raw === 'number') {
      result[param.name] = raw;
      return;
    }
    result[param.name] = typeof preset[param.name] === 'number' ? preset[param.name] : param.default;
  });
  if (info.modes.length) result.mode = engineModeFor(engineId, values);
  return result;
}

const sameNumber = (a, b) => Math.abs(a - b) <= 1e-9 * Math.max(1, Math.abs(b));

// Выбор пользователя: ручки, отличающиеся от базовой линии — пресета режима или
// дефолта движка. Совпавшее с базовой линией не отправляем и не храним: записать
// его — значит явно перекрыть пресет выбранного режима, и переключатель режимов
// не менял бы ничего.
function chosenEngineParams(engineId, values) {
  const info = engineInfo(engineId);
  if (!info) return {};
  const preset = modePreset(engineId, values);
  const current = engineParamsFor(engineId, values);
  const result = {};
  info.params.forEach((param) => {
    const baseline = typeof preset[param.name] === 'number' ? preset[param.name] : param.default;
    if (!sameNumber(current[param.name], baseline)) result[param.name] = current[param.name];
  });
  if (info.modes.length) result.mode = engineModeFor(engineId, values);
  return result;
}

// Смена режима: ручки, которых пользователь не выбирал, пересчитываются по пресету
// нового режима. Иначе ползунки показывали бы прошлый режим, а модель считала бы
// новый. Подобранное руками остаётся и по-прежнему важнее пресета — тот же порядок,
// что и на бэкенде (дефолты → пресет режима → явные ручки).
function applyEngineMode(container, engineId, values, modeId) {
  values.mode = modeId;
  container.innerHTML = engineParamsHtml(engineId, values);
  return chosenEngineParams(engineId, readEngineParams(container));
}

function engineParamsHtml(engineId, values) {
  const info = engineInfo(engineId);
  const modes = engineModesHtml(engineId, values);
  if (!info || !info.params.length) return modes;
  const current = engineParamsFor(engineId, values);
  return `${modes}<div class="grid-2">${info.params.map((param) => `
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
  const mode = container.querySelector('[data-engine-mode]');
  if (mode) result.mode = mode.value;
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
  $('tab-recording').hidden = name !== 'recording';
  $('tab-dictionary').hidden = name !== 'dictionary';
  $('tab-models').hidden = name !== 'models';
  // Вкладка записи грузится при первом открытии: список проектов и сам проект
  // нужны только тому, кто пришёл записывать, а микрофон — тем более.
  if (name === 'recording') {
    loadRecordingTab().catch((error) => showAlert($('rec-dialogue-error'), error.message));
  } else {
    // Уход с вкладки закрывает микрофон и снимает опрос монтажа: незавершённая
    // запись не должна держать устройство занятым, а прогресс — долбить сервер
    // из фоновой вкладки. При возврате опрос возобновится сам.
    stopRecordingCapture();
    stopRecordingRenderPoll();
  }
  // Словарь грузится при первом открытии вкладки: он не нужен для озвучки, и
  // запрашивать его на каждом старте дашборда незачем.
  if (name === 'dictionary') {
    loadDictionary().catch((error) => showAlert($('dictionary-error'), error.message));
  }
  // Лингвистический анализ — общая ось обеих вкладок озвучки: обновляем состояние
  // при входе, чтобы блок не показывал устаревшее «выкл» или чужой проект.
  if (name === 'text' || name === 'dialogue') {
    refreshLlmAnalysis();
  }
  // Модели — так же: список весов нужен только тому, кто пришёл его смотреть,
  // а обход каталогов с гигабайтными файлами на старте дашборда незачем.
  if (name === 'models') {
    loadModels().catch((error) => showAlert($('models-error'), error.message));
    loadCache().catch((error) => showAlert($('cache-error'), error.message));
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

// --- очистка кеша приложения --------------------------------------------------
// Секция рядом с менеджером моделей, но о своём: три категории транзитных файлов
// самого дашборда. Веса моделей и кеш HF здесь намеренно отсутствуют — их удаление
// остаётся отдельным осознанным потоком в списке выше. Пути категорий жёстко заданы
// на бэкенде, поэтому форма не может попросить больше, чем описано в подписях.
const CACHE_CATEGORIES = [
  {
    id: 'output',
    label: 'Готовые файлы задач',
    hint: 'output/ — то же, что чистит TTL, но сейчас',
  },
  {
    id: 'benchmarks',
    label: 'Сравнение движков',
    hint: 'output/benchmarks/ — под TTL не попадает вовсе',
  },
  {
    id: 'temp_files',
    label: 'Временные файлы',
    hint: 'промежуточные .tmp.wav и остатки экспорта/импорта',
  },
];

const CACHE_LABELS = Object.fromEntries(CACHE_CATEGORIES.map((item) => [item.id, item.label]));

function cacheSize(category) {
  const data = state.cache;
  if (!data) return null;
  if (category === 'temp_files') return { mb: data.temp_files_mb, files: data.temp_files_count };
  return { mb: data[`${category}_mb`], files: null };
}

function pluralFiles(count) {
  const mod10 = count % 10;
  const mod100 = count % 100;
  if (mod10 === 1 && mod100 !== 11) return 'файл';
  if (mod10 >= 2 && mod10 <= 4 && (mod100 < 12 || mod100 > 14)) return 'файла';
  return 'файлов';
}

function renderCache() {
  const container = $('cache-list');
  if (!container) return;
  const data = state.cache;
  if (!data) {
    container.innerHTML = '<p class="muted">Считаю занятое место…</p>';
    return;
  }
  let totalFiles = 0;
  let totalMb = 0;
  const rows = CACHE_CATEGORIES.map((category) => {
    const size = cacheSize(category.id);
    const checked = state.cacheChecked[category.id] ? ' checked' : '';
    const suffix = size.files === null
      ? ''
      : ` · ${size.files} ${pluralFiles(size.files)}`;
    if (size.mb > 0) {
      totalMb += size.mb;
      if (size.files !== null) totalFiles += size.files;
    }
    return `<label class="toggle cache-row">
      <input type="checkbox" data-cache-target="${category.id}"${checked}${state.cacheBusy ? ' disabled' : ''} />
      <span>${esc(category.label)}<br /><small class="muted">${esc(category.hint)}</small></span>
      <b class="muted">${size.mb > 0 ? `${size.mb.toFixed(1)} МБ${suffix}` : 'пусто'}</b>
    </label>`;
  }).join('');
  const empty = totalMb <= 0
    ? '<p class="muted">Очищать нечего: все три категории пусты.</p>'
    : '';
  container.innerHTML = rows + empty;
  updateCacheHint(totalFiles);
}

function updateCacheHint(totalFiles) {
  const hint = $('cache-hint');
  const button = $('btn-clear-cache');
  if (!hint || !button) return;
  const count = CACHE_CATEGORIES.filter((item) => state.cacheChecked[item.id]).length;
  button.disabled = state.cacheBusy || count === 0;
  if (state.cacheBusy) {
    hint.textContent = 'Освобождаю выбранное…';
    return;
  }
  if (count === 0) {
    hint.textContent = 'Ничего не отмечено: выберите хотя бы одну категорию — '
      + 'очистка работает только по явному выбору.';
    return;
  }
  const words = totalFiles === 1 ? 'файл' : 'файлов';
  hint.textContent = `Отмечено категорий: ${count}`
    + (totalFiles > 0 ? ` · всего в них ${totalFiles} ${words}` : '');
}

async function loadCache() {
  state.cache = await api('/api/cache');
  showAlert($('cache-error'), '');
  renderCache();
}

async function clearCache() {
  const targets = CACHE_CATEGORIES
    .map((item) => item.id)
    .filter((id) => state.cacheChecked[id]);
  if (!targets.length) {
    showAlert($('cache-error'), 'Ничего не выбрано: отметьте хотя бы одну категорию очистки.');
    renderCache();
    return;
  }
  state.cacheBusy = true;
  renderCache();
  try {
    const result = await api('/api/cache/clear', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ targets }),
    });
    const parts = targets
      .map((target) => `${CACHE_LABELS[target]}: ${(result.freed_mb[target] || 0).toFixed(2)} МБ, `
        + `${result.freed_files[target] || 0} ${pluralFiles(result.freed_files[target] || 0)}`)
      .join(' · ');
    showAlert($('cache-note'), `Освобождено ${result.total_mb.toFixed(2)} МБ (${parts}).`, 'info');
    showAlert($('cache-error'), '');
  } catch (error) {
    showAlert($('cache-error'), error.message);
  } finally {
    state.cacheBusy = false;
    // Размеры перечитываются после любой попытки: интерфейс не должен показывать
    // «занято», если файлы уже удалены.
    await loadCache().catch((error) => showAlert($('cache-error'), error.message));
  }
}

function bindCacheEvents() {
  const list = $('cache-list');
  if (list) {
    list.addEventListener('change', (event) => {
      const box = event.target.closest('[data-cache-target]');
      if (!box) return;
      state.cacheChecked[box.dataset.cacheTarget] = box.checked;
      updateCacheHint();
    });
  }
  const clear = $('btn-clear-cache');
  if (clear) clear.addEventListener('click', () => {
    clearCache().catch((error) => showAlert($('cache-error'), error.message));
  });
  const reload = $('btn-reload-cache');
  if (reload) reload.addEventListener('click', () => {
    loadCache().catch((error) => showAlert($('cache-error'), error.message));
  });
}

// --- сброс вкладки -------------------------------------------------------------
// Одна кнопка на вкладку, ступень выбирается рядом с ней. Ступени разные по цене
// ошибки — «только кеш» освобождает транзитные файлы, «полный сброс» уносит ещё и
// то, с чем работала вкладка, — поэтому кнопки «стереть всё» без выбора здесь нет:
// случайное нажатие одной кнопки не должно стоить проекта. Что именно удаляется,
// решает бэкенд (`backend/app_reset.py`); клиент не может попросить больше, чем
// разрешено списком ступеней, и в теле запроса перечисляет только выбор.
const RESET_TABS = {
  dialogue: {
    scopeId: 'reset-scope',
    buttonId: 'btn-reset',
    errorId: 'reset-error',
    noteId: 'reset-note',
    hintId: 'reset-hint',
    question: 'Полный сброс удалит открытый проект вместе с репликами, вариантами '
      + 'озвучки и архивами диагностики. Остальные проекты не тронет. Продолжить?',
    hints: {
      cache: 'Освободит готовые файлы задач, результаты сравнения движков и временные '
        + 'остатки. Диалог и настройки вкладки не тронет.',
      full: 'Освободит то же, что «только кеш», и удалит открытый проект вместе с '
        + 'репликами, вариантами озвучки и архивами диагностики. Остальные проекты, '
        + 'голоса и словарь произношения остаются. Настройки анализатора вернутся к '
        + 'значениям окружения.',
    },
  },
  text: {
    scopeId: 'text-reset-scope',
    buttonId: 'btn-reset-text',
    errorId: 'text-reset-error',
    noteId: 'text-reset-note',
    hintId: 'text-reset-hint',
    question: 'Полный сброс очистит текст и настройки вкладки «Сплошной текст». '
      + 'Голоса и словарь произношения останутся. Продолжить?',
    hints: {
      cache: 'Освободит готовые файлы задач, результаты сравнения движков и временные '
        + 'остатки. Текст и настройки вкладки не тронет.',
      full: 'Освободит то же, что «только кеш», и очистит текст, выбранный голос и '
        + 'настройки вкладки. Голоса, словарь произношения и открытый диалог остаются. '
        + 'Настройки анализатора вернутся к значениям окружения.',
    },
  },
};

// Настройки, которые «полный сброс» возвращает к умолчанию. Значения берутся из
// разметки один раз при старте, пока пользователь их не тронул: держать вторую
// копию умолчаний в JS — значит однажды разойтись с формой.
const RESET_DIALOGUE_CONTROLS = [
  'pause', 'crossfade', 'auto-accent', 'output-format', 'output-name', 'qa',
  'short-enabled', 'short-strategy', 'short-very-short', 'short-short', 'chunk-strategy',
];
const RESET_TEXT_CONTROLS = [
  'text-voice', 'text-pause', 'text-crossfade', 'text-auto-accent', 'text-output-format',
  'text-output-name', 'text-qa', 'text-short-enabled', 'text-short-strategy',
  'text-short-very-short', 'text-short-short', 'text-chunk-strategy', 'text-body',
  'text-speed', 'text-cfg', 'text-nfe', 'text-gain', 'text-pitch', 'text-rms',
];
// Не поля формы, а подписи, которые меняются по ходу работы: у сплошного текста
// дропзона показывает имя загруженного файла, а после сброса должна снова звать
// перетащить файл.
const RESET_HTML_BLOCKS = ['text-drop'];
const RESET_DEFAULTS = {};

function captureResetDefaults() {
  [...RESET_DIALOGUE_CONTROLS, ...RESET_TEXT_CONTROLS, ...RESET_HTML_BLOCKS].forEach((id) => {
    const element = $(id);
    if (!element) return;
    const control = ['INPUT', 'SELECT', 'TEXTAREA'].includes(element.tagName);
    RESET_DEFAULTS[id] = control
      ? { control: true, value: element.type === 'checkbox' ? element.checked : element.value }
      : { control: false, value: element.innerHTML };
  });
}

function restoreDefaults(ids) {
  ids.forEach((id) => {
    const saved = RESET_DEFAULTS[id];
    const element = $(id);
    if (!saved || !element) return;
    if (!saved.control) {
      element.innerHTML = saved.value;
      return;
    }
    if (element.type === 'checkbox') element.checked = saved.value;
    else element.value = saved.value;
    // Подписи ползунков обновляют их же обработчики, а программная запись события
    // не порождает — стреляем тем, что они слушают, вместо копии форматирования.
    if (element.type === 'range') element.dispatchEvent(new Event('input', { bubbles: true }));
  });
}

function resetScope(tab) {
  const checked = document.querySelector(`#${RESET_TABS[tab].scopeId} input:checked`);
  return checked ? checked.value : 'cache';
}

function updateResetHint(tab) {
  const hint = $(RESET_TABS[tab].hintId);
  if (hint) hint.textContent = RESET_TABS[tab].hints[resetScope(tab)];
}

function setResetBusy(tab, busy) {
  const button = $(RESET_TABS[tab].buttonId);
  if (!button) return;
  button.disabled = busy;
  button.textContent = busy ? 'Сбрасываю…' : 'Сбросить';
}

// Отчёт читается как список сделанного, а не как намерение: «проект удалён» здесь
// появляется только тогда, когда удаление действительно произошло.
function resetReportText(report) {
  const cache = report.cache || {};
  const parts = [`освобождено ${Number(cache.total_mb || 0).toFixed(2)} МБ кеша`];
  if (report.project_deleted) {
    const archives = report.diagnostics_deleted
      ? `, архивов диагностики удалено: ${report.diagnostics_deleted}`
      : '';
    parts.push(`открытый проект удалён${archives}`);
  }
  if (report.llm_settings_reset) {
    parts.push('настройки анализатора возвращены к значениям окружения');
  }
  return `Сброс выполнен: ${parts.join(' · ')}.`;
}

// Панель «что услышит модель» держит последний показанный разбор: после сброса он
// относился бы к тексту, которого больше нет.
function clearPreviewMount(mountId) {
  const mount = $(mountId);
  if (!mount || !mount.dataset.ready) return;
  previewEl(mount, 'text').value = '';
  previewEl(mount, 'voice').value = '';
  previewEl(mount, 'engine').value = '';
  previewEl(mount, 'status').textContent = '';
  showAlert(previewEl(mount, 'error'), '');
  const result = previewEl(mount, 'result');
  result.hidden = true;
  result.innerHTML = '';
  previewEl(mount, 'empty').hidden = false;
}

function clearDialogueTab() {
  stopTake();
  if (state.pollTimer) {
    clearInterval(state.pollTimer);
    state.pollTimer = null;
  }
  state.jobId = null;
  state.replicaJobId = null;
  state.project = null;
  state.sourceDirty = false;
  state.analysis = null;
  state.llm = null;
  state.timeline = null;
  state.timelineError = null;
  state.preview = {};
  state.replicaPreview = {};
  state.openDetails = {};
  state.preparedDetails = {};
  state.speakerDetails = {};
  rememberProject(null);
  $('dialogue').value = '';
  $('dialogue-file-note').hidden = true;
  restoreDefaults(RESET_DIALOGUE_CONTROLS);
  showAlert($('parse-error'), '');
  showAlert($('analyze-error'), '');
  showAlert($('job-error'), '');
  showAlert($('timeline-error'), '');
  $('job-status').textContent = '—';
  setProgress(0);
  $('player').pause();
  $('player').hidden = true;
  $('player').removeAttribute('src');
  $('download-link').hidden = true;
  clearPreviewMount('dialogue-preview');
  // Проекта больше нет, а список архивов строится по нему: без явной очистки в нём
  // остались бы ссылки на удалённые файлы.
  renderDiagnosticsList([]);
  applyProject(null, { speakers: false, analysis: false });
  refreshLlmAnalysis();
}

function clearTextTab() {
  if (state.textPollTimer) {
    clearInterval(state.textPollTimer);
    state.textPollTimer = null;
  }
  state.textJobId = null;
  state.textEngineParams = {};
  state.textEngineVoice = '';
  restoreDefaults(RESET_TEXT_CONTROLS);
  restoreDefaults(RESET_HTML_BLOCKS);
  showAlert($('text-error'), '');
  showAlert($('text-job-error'), '');
  $('text-job-status').textContent = '—';
  $('text-job-note').hidden = true;
  setTextProgress(0);
  $('text-player').pause();
  $('text-player').hidden = true;
  $('text-player').removeAttribute('src');
  $('text-download-link').hidden = true;
  resetSuggestions();
  $('suggestions-status').textContent = '';
  $('suggestions-empty').hidden = false;
  // Голос сброшен в «не выбран», поэтому набор ручек движка тоже пуст — иначе он
  // относился бы к голосу, которого в форме уже нет.
  renderTextEngineParams();
  updateTextSummary();
  updateTextButton();
  clearPreviewMount('text-preview');
  refreshLlmAnalysis();
}

async function resetTab(tab) {
  const config = RESET_TABS[tab];
  const scope = resetScope(tab);
  if (scope === 'full' && !window.confirm(config.question)) return;
  const projectId = tab === 'dialogue' && state.project ? state.project.id : null;
  showAlert($(config.errorId), '');
  setResetBusy(tab, true);
  try {
    const report = await api('/api/reset', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      // Проект уходит только у полного сброса диалога: сервер не знает, что открыто
      // в браузере, и угадывать это по «последнему» проекту здесь нельзя.
      body: JSON.stringify({
        scope,
        tab,
        project_id: scope === 'full' ? projectId : null,
      }),
    });
    // Файл настроек удалён, а живой статус помнит прежние значения: перечитываем
    // его, иначе панель анализа показывала бы настройки, которых уже нет.
    if (report.llm_settings_reset) {
      state.llmStatus = null;
      state.llmStatusSettings = null;
    }
    if (scope === 'full') {
      if (tab === 'dialogue') clearDialogueTab();
      else clearTextTab();
    }
    showAlert($(config.noteId), resetReportText(report), 'info');
  } catch (error) {
    showAlert($(config.errorId), error.message);
  } finally {
    setResetBusy(tab, false);
  }
}

function bindResetEvents() {
  Object.keys(RESET_TABS).forEach((tab) => {
    const group = $(RESET_TABS[tab].scopeId);
    if (group) {
      group.addEventListener('change', () => {
        updateResetHint(tab);
        // Отчёт прошлой ступени к новой не относится: «кеш очищен» рядом с
        // «полным сбросом» читалось бы как выполненный полный сброс.
        showAlert($(RESET_TABS[tab].noteId), '');
      });
    }
    const button = $(RESET_TABS[tab].buttonId);
    if (button) {
      button.addEventListener('click', () => {
        resetTab(tab).catch((error) => showAlert($(RESET_TABS[tab].errorId), error.message));
      });
    }
    updateResetHint(tab);
  });
}

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
    <div class="card dict-rule${entry.enabled ? '' : ' off'}" data-entry-id="${esc(entry.id)}">
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
      <button class="tiny ghost" data-role="yo-example"
        title="Тестовые данные для проверки шага «восстановление ё», а не фраза для записи референса: плотность «ё» здесь противоестественная и ломает ровный темп начитки.">пример с «ё» — только для проверки</button>
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
        // Подстановка тестовой фразы с плотной «ё»: она показывает стадии
        // «нормализация» и «восстановление ё» на материале, где шаг срабатывает
        // много раз. Это не референс и не попадает в список фраз для записи.
        if (event.target.closest('[data-role="yo-example"]')) {
          previewEl(mount, 'text').value = YO_STRESS_TEST_PHRASE;
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
  // Выбор цели записи — из того же списка голосов: без него мастер интонационных
  // профилей не знал бы, куда записывать, а план записи не показывал бы прогресс.
  renderRecordTargets();
  updateGenerateButton();
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
  // В подписи видно и фразу, и интонацию, которую она записывает: пользователь
  // начитывает одну и ту же фразу в конкретном регистре, и знать его до записи
  // важнее, чем после (§16).
  $('record-phrase').innerHTML = RECORD_PHRASES
    .map((phrase, index) => {
      const profile = RECORD_PHRASE_PROFILES[phrase.label];
      const suffix = profile ? ` → ${emotionLabel(profile)}` : '';
      return `<option value="${index}">${esc(phrase.label + suffix)}</option>`;
    })
    .join('');
  showRecordPhrase();
}

// --- мастер интонационных профилей --------------------------------------------
// «Куда записывать» — это режим, а не голос: пустая строка означает обычный путь
// через форму нового голоса. Значение живёт в самом `select`, а не в `state`:
// перерисовка списка голосов не должна терять выбор пользователя, и хранить его
// в двух местах значило бы однажды их разойтись.
function recordTargetId() {
  const select = $('record-target');
  return select ? select.value || RECORD_NEW_VOICE : RECORD_NEW_VOICE;
}

function recordTargetVoice() {
  const voiceId = recordTargetId();
  return voiceId ? voiceById(voiceId) : null;
}

// Какие фразы у голоса уже записаны. Считаем по `source_record_phrase_id`, а не по
// интонации: именно по этому полю бэкенд заменяет профиль при повторной записи, и
// список должен показывать ровно то, что считает он.
function recordedPhraseIds(voice) {
  const ids = new Set();
  ((voice && voice.reference_profiles) || []).forEach((profile) => {
    if (profile.source_record_phrase_id) ids.add(String(profile.source_record_phrase_id));
  });
  return ids;
}

function renderRecordTargets() {
  const select = $('record-target');
  if (!select) return;
  const current = select.value;
  select.innerHTML = [
    `<option value="${RECORD_NEW_VOICE}">Новый голос (форма ниже)</option>`,
    ...state.voices.map(
      (voice) => `<option value="${esc(voice.id)}">${esc(voice.name)} — интонационные профили</option>`,
    ),
  ].join('');
  // Цель сохраняем между перерисовками: список голосов перечитывается после каждой
  // записи, а сбрасывать выбор посреди мастера нельзя.
  select.value = state.voices.some((voice) => voice.id === current) ? current : RECORD_NEW_VOICE;
  renderRecordPlan();
}

function renderRecordPlan() {
  const box = $('record-plan');
  if (!box) return;
  const voice = recordTargetVoice();
  if (!voice) {
    box.innerHTML = '';
    $('record-target-note').textContent = '';
    return;
  }
  const done = recordedPhraseIds(voice);
  const recorded = RECORD_PHRASES.filter((_, index) => done.has(recordPhraseId(index))).length;
  const missing = RECORD_PHRASES.length - recorded;
  $('record-target-note').textContent = missing
    ? `Запись станет профилем голоса «${voice.name}». Осталось фраз: ${missing} из ${RECORD_PHRASES.length}.`
    : `У голоса «${voice.name}» записаны все ${RECORD_PHRASES.length} фраз — можно перезаписать любую.`;
  box.innerHTML = RECORD_PHRASES.map((phrase, index) => {
    const profile = RECORD_PHRASE_PROFILES[phrase.label];
    const mark = done.has(recordPhraseId(index))
      ? '<span class="tag ok">записано</span>'
      : '<span class="muted">нет записи</span>';
    return `
      <div class="row between" style="margin-top:4px">
        <span class="muted">${esc(phrase.label)} → ${esc(emotionLabel(profile))}</span>
        ${mark}
      </div>`;
  }).join('');
}

// Мастер ведёт по списку сам: после записи выбирается первая фраза без профиля,
// иначе одиннадцать записей требовали бы одиннадцати переключений вручную.
function advanceRecordPhrase() {
  const voice = recordTargetVoice();
  if (!voice) return;
  const done = recordedPhraseIds(voice);
  const next = RECORD_PHRASES.findIndex((_, index) => !done.has(recordPhraseId(index)));
  if (next >= 0) $('record-phrase').value = String(next);
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

  // Мастер (§16): запись идёт прямо в профиль выбранного голоса. Форму нового
  // голоса не трогаем — иначе та же запись осела бы в двух местах сразу, и
  // «Сохранить голос» создал бы из неё лишний голос.
  if (recordTargetVoice()) {
    saveRecordedReference(file);
    return;
  }

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

// Мастер интонационных профилей (§16): одна фраза — один профиль одного голоса.
// Расшифровка берётся из самой фразы (её читают с экрана), поэтому сверку через
// Whisper выключаем: текст известен точно, а десятки секунд ожидания здесь лишние.
// Проверка «референс ↔ расшифровка» на бэкенде всё равно выполняется и вернёт
// профиль с `quality_status = warning`, если запись разошлась с фразой.
async function saveRecordedReference(file) {
  const voice = recordTargetVoice();
  if (!voice) return;
  const index = parseInt($('record-phrase').value || '0', 10);
  const phrase = RECORD_PHRASES[index] || RECORD_PHRASES[0];
  const profile = RECORD_PHRASE_PROFILES[phrase.label];
  if (!profile) {
    setRecordStatus('запись готова, но профиль не сохранён');
    showAlert($('record-error'), `Для фразы «${phrase.label}» не задан профиль — запись отменена.`);
    return;
  }
  setRecordStatus('сохраняю профиль…');
  const form = new FormData();
  form.append('file', file);
  form.append('emotion', profile);
  form.append('ref_text', phrase.text);
  form.append('label', phrase.label);
  form.append('verify_ref_text', 'false');
  // Идентификатор фразы: по нему бэкенд заменяет прежний профиль этой же фразы,
  // а не заводит второй (§16).
  form.append('source_record_phrase_id', recordPhraseId(index));
  try {
    const result = await api(`/api/voices/${voice.id}/references`, { method: 'POST', body: form });
    setRecordStatus('профиль сохранён');
    const saved = result.profile || {};
    const warning = saved.quality_status === 'warning'
      ? `\n\nВнимание: ${saved.quality_note || 'расшифровка не совпала с записью'} — профиль помечен «проверить» и в синтезе уступает нейтральному.`
      : '';
    showAlert(
      $('record-notes'),
      `«${phrase.label}» → ${emotionLabel(profile)}: профиль голоса «${voice.name}» сохранён.${warning}`,
      'info',
    );
    // Список голосов перечитываем до перехода к следующей фразе: план записи и
    // счётчик берут данные из него, а не из ответа на одну запись.
    await loadVoices();
    advanceRecordPhrase();
  } catch (error) {
    setRecordStatus('запись готова, но профиль не сохранён');
    showAlert($('record-error'), error.message);
  }
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

// --- обязательная подготовка диалога ------------------------------------------
// Синтез возможен только по проанализированному тексту, и запрет держит сервер
// (409), а не погашенная кнопка. Здесь показывается состояние, причина и то, что
// делать дальше: «кнопка не работает» без объяснения — худший вид интерфейса.
const ANALYSIS_LABELS = {
  raw: 'не подготовлен',
  analyzing: 'анализируется',
  needs_review: 'нужно подтверждение',
  ready: 'готов',
  error: 'ошибка анализа',
};

function analysisStatus() {
  return (state.analysis && state.analysis.status) || 'raw';
}

function updateAnalysisBar() {
  const analysis = state.analysis;
  const badge = $('analysis-badge');
  const text = $('analysis-state');
  const button = $('btn-analyze');
  const summary = $('prepare-summary');

  if (!state.project || !analysis) {
    badge.textContent = '—';
    badge.className = 'badge';
    text.textContent = 'Разберите текст — анализ подготовит реплики к синтезу';
    button.disabled = !state.project || state.analyzeBusy;
    summary.hidden = true;
    renderReviewPanel();
    return;
  }

  const status = analysisStatus();
  badge.textContent = ANALYSIS_LABELS[status] || status;
  badge.className = `badge analysis-${status}`;
  const done = analysis.replicas_done || 0;
  const total = analysis.replicas_total || 0;
  const parts = [`реплик подготовлено: ${done} из ${total}`];
  if (analysis.replicas_pending) parts.push(`ждут анализа: ${analysis.replicas_pending}`);
  if (analysis.replicas_error) parts.push(`с ошибкой: ${analysis.replicas_error}`);
  if (analysis.candidates_total) parts.push(`слов на проверку: ${analysis.candidates_total}`);
  text.textContent = parts.join(' · ');

  button.disabled = state.analyzeBusy || !total;
  button.textContent = status === 'ready' ? 'Пересчитать анализ' : 'Анализировать диалог';

  // Подтверждение перед синтезом (§7 шаг D): что именно подготовлено.
  const voices = (state.project.speakers || []).filter((item) => item.voice_id).length;
  const rules = (state.project.replicas || []).reduce(
    (sum, replica) => sum + ((replica.dictionary_matches || []).length), 0
  );
  const engines = [
    ...new Set((state.project.replicas || []).map((replica) => replica.engine).filter(Boolean)),
  ];
  const accents = engines.length
    ? engines.map((engine) => `${engineLabel(engine)}: ${engineInfo(engine).supports_accents ? 'включены' : 'не поддерживает'}`).join(', ')
    : '—';
  summary.hidden = false;
  summary.innerHTML = [
    `<span>Подготовлено реплик: <b>${done}</b></span>`,
    `<span>Голоса: <b>${voices}</b></span>`,
    `<span>Словарь: <b>${rules}</b> применённых правил</span>`,
    `<span>Ударения: <b>${esc(accents)}</b></span>`,
  ].join('');
  renderReviewPanel();
}

async function refreshAnalysis() {
  if (!state.project) {
    state.analysis = null;
    updateAnalysisBar();
    return null;
  }
  const signal = staleSignal('analysis');
  try {
    state.analysis = await api(`/api/projects/${state.project.id}/analysis`, { signal });
  } catch (error) {
    // Отменённый запрос уступил место новому: состояние и панель обновит он.
    if (isAbort(error)) return null;
    // Состояние — вспомогательная информация: не смогли прочитать, покажем
    // «не подготовлен» и не будем мешать работать с карточками.
    state.analysis = null;
    showAlert($('analyze-error'), error.message);
  }
  updateAnalysisBar();
  updateGenerateButton();
  await refreshLlmAnalysis();
  return state.analysis;
}

// Подписи состояний лингвистического анализа (§12 Task 2). Отдельный словарь, а не
// переиспользование ANALYSIS_LABELS: это другая ось — «LLM смотрела или нет», а не
// «текст подготовлен или нет».
const LLM_LABELS = {
  DISABLED: 'выкл',
  PENDING: 'ожидает',
  RUNNING: 'анализ',
  READY: 'готов',
  NEEDS_REVIEW: 'требует проверки',
  FAILED: 'ошибка',
  STALE: 'устарел',
};

async function refreshLlmAnalysis() {
  const panel = $('llm-panel');
  if (!panel) return null;
  // Один сигнал на весь вызов: оба запроса ниже — часть одного обновления панели,
  // и отменять их друг другом нельзя.
  const signal = staleSignal('llm');
  if (!state.llmStatus) {
    try {
      state.llmStatus = await api('/api/llm/status', { signal });
      state.llmStatusSettings = state.llmStatus.settings || null;
      renderLlmPanel(state.llmStatus);
    } catch (error) {
      if (isAbort(error)) return null;
      state.llmStatus = null;
    }
  }
  if (!state.project) {
    state.llm = null;
    renderLlmPanel(state.llmStatus);
    return null;
  }
  try {
    state.llm = await api(`/api/projects/${state.project.id}/linguistic-analysis`, { signal });
  } catch (error) {
    // Отменённый запрос уступил место новому — он и обновит панель.
    if (isAbort(error)) return null;
    // Состояние анализа — вспомогательная информация: не смогли прочитать, панель
    // просто не показываем, работа с репликами не блокируется.
    state.llm = null;
  }
  renderLlmPanel();
  return state.llm;
}

function renderLlmPanel(statusData = null) {
  const panel = $('llm-panel');
  if (!panel) return;
  if (statusData) {
    // Статус приходит из /api/llm/status и /api/llm/settings: он описывает
    // анализатор, а не проект, поэтому хранится отдельно от state.llm.
    state.llmStatus = statusData;
    state.llmStatusSettings = statusData.settings || state.llmStatusSettings;
    syncLlmControls(statusData);
  }
  const data = state.llm;
  const textPanel = $('text-llm-panel');
  // Во вкладке сплошного текста проекта нет: там показываем только состояние
  // анализатора из /api/llm/status, а разбор появляется в ответе рендера.
  if (textPanel && !state.project) {
    textPanel.hidden = false;
    renderLlmStatusLine($('text-llm-badge'), $('text-llm-state'), null);
  } else if (textPanel) {
    textPanel.hidden = false;
    renderLlmStatusLine($('text-llm-badge'), $('text-llm-state'), data);
  }
  syncLlmControls(statusData);
  if (!data) {
    // Проекта ещё нет: показываем состояние анализатора, панель не прячем — иначе
    // включить анализ было бы негде.
    panel.hidden = false;
    renderLlmStatusLine(
      $('llm-badge'),
      $('llm-state'),
      statusData || state.llmStatus || null
    );
    $('llm-list').innerHTML = '';
    return;
  }
  panel.hidden = false;
  renderLlmStatusLine($('llm-badge'), $('llm-state'), data, true);
  const list = $('llm-list');
  const candidates = (data.candidates || []).filter((item) => item.needs_review);
  if (!candidates.length) {
    list.innerHTML = '<div class="muted">Спорных слов нет — модель ничего не требует подтверждать.</div>';
    return;
  }
  list.innerHTML = candidates
    .map(
      (item, index) => `
      <div class="card review-row" data-llm-index="${index}">
        <div class="row between">
          <span><b>${esc(item.word)}</b>${item.target ? ` → <span class="muted">${esc(item.target)}</span>` : ''}</span>
          <span class="muted">реплика ${(item.replica_index || 0) + 1}</span>
        </div>
        <div class="muted review-reason">${esc(item.reason || '')}${
          item.agreement === 'conflict' ? ' · <b>конфликт со словарём</b>' : ''
        }</div>
        <div class="row">
          <button class="tiny primary" data-llm-review="project">Только в этот проект</button>
          <button class="tiny ghost" data-llm-review="global">Во все проекты</button>
          <button class="tiny ghost" data-llm-review="skip">Пропустить</button>
        </div>
      </div>`
    )
    .join('');
  // Действия — через тот же review-путь, что и у детерминированных кандидатов:
  // второго способа писать в словарь быть не должно.
  list.querySelectorAll('[data-llm-review]').forEach((button) => {
    button.addEventListener('click', () => {
      const card = button.closest('[data-llm-index]');
      reviewLlmCandidate(Number(card.dataset.llmIndex), button.dataset.llmReview);
    });
  });
}

// Переключатель анализа: без него «выключен» нельзя было исправить из приложения,
// и пользователь видел только сообщение про LLM_ANALYZER_ENABLED=0.
async function loadLlmModels() {
  const select = $('llm-model');
  if (!select) return;
  try {
    const payload = await api('/api/llm/models');
    const models = payload.models || [];
    select.innerHTML = models
      .map((item) => `<option value="${esc(item.tag)}">${esc(item.tag)}${
        item.role ? ` (${item.role === 'primary' ? 'основная' : 'запасная'})` : ''
      }</option>`)
      .join('');
    if (payload.selected) select.value = payload.selected;
    state.llmModels = models;
  } catch (error) {
    select.innerHTML = '';
  }
}

function syncLlmControls(data) {
  const enabled = $('llm-enabled');
  const required = $('llm-required');
  const note = $('llm-settings-note');
  if (!enabled) return;
  const settings = (data && data.settings) || state.llmStatusSettings || {};
  enabled.checked = Boolean((data && data.enabled) ?? settings.enabled);
  if (required) required.checked = Boolean(data ? data.required_for_render : settings.required_for_render);
  if (note) {
    const ollama = (data && data.ollama) || {};
    if (!enabled.checked) {
      note.textContent = 'Анализ выключен — включите его здесь, чтобы модель разбирала текст';
    } else if (data && !data.model_available) {
      note.textContent = `Модель «${data.model || ''}» не скачана: ollama pull ${data.model || ''}`;
    } else if (!ollama.available) {
      note.textContent = `Ollama недоступна: ${ollama.error || 'запустите ollama serve'}`;
    } else {
      note.textContent = 'Модель остаётся локальной: текст никуда не отправляется';
    }
  }
}

async function saveLlmSettings(patch) {
  const note = $('llm-settings-note');
  try {
    const payload = await api('/api/llm/settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(patch),
    });
    state.llmStatusSettings = (payload.status && payload.status.settings) || null;
    renderLlmPanel(payload.status);
    if (state.project) await refreshLlmAnalysis();
  } catch (error) {
    if (note) note.textContent = error.message;
  }
}

function renderLlmStatusLine(badge, text, data, detailed = false) {
  if (!badge || !text) return;
  if (!data) {
    badge.textContent = '—';
    badge.className = 'badge';
    text.textContent = 'Состояние анализатора читается при открытии проекта';
    return;
  }
  const status = data.status || 'DISABLED';
  badge.textContent = LLM_LABELS[status] || status;
  badge.className = `badge analysis-${status === 'READY' ? 'ready' : status === 'NEEDS_REVIEW' ? 'needs_review' : status === 'FAILED' ? 'error' : 'raw'}`;
  const parts = [];
  if (data.model) parts.push(`модель: ${data.model}`);
  if (detailed) {
    parts.push(`предложений: ${data.candidates_total || 0}`);
    parts.push(`на проверку: ${data.needs_review_total || 0}`);
    if (data.conflicts_total) parts.push(`конфликтов со словарём: ${data.conflicts_total}`);
  }
  if (data.error) parts.push(`причина: ${data.error}`);
  if (!data.enabled) parts.push('анализатор выключен (LLM_ANALYZER_ENABLED=0)');
  text.textContent = parts.join(' · ');
}

async function reviewLlmCandidate(index, action) {
  const candidates = (state.llm && state.llm.candidates || []).filter((item) => item.needs_review);
  const item = candidates[index];
  if (!item || !state.project) return;
  const accepted = action !== 'skip' && Boolean((item.target || '').trim());
  try {
    await api(`/api/projects/${state.project.id}/pronunciation/review`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        source: item.word,
        target: item.target || item.word,
        scope: action === 'global' ? 'global' : 'project',
        enabled: accepted,
        replica_index: item.replica_index,
        note: 'решение по предложению локальной LLM',
      }),
    });
    await applyProject(await api(`/api/projects/${state.project.id}`), { analysis: false });
    await refreshAnalysis();
    if (typeof loadDictionary === 'function') loadDictionary().catch(() => {});
  } catch (error) {
    showAlert($('analyze-error'), error.message);
  }
}

async function analyzeDialogue() {
  if (!state.project || state.analyzeBusy) return;
  showAlert($('analyze-error'), '');
  state.analyzeBusy = true;
  updateAnalysisBar();
  try {
    if (state.sourceDirty) {
      // Анализ разбирает текст сам: держать несохранённую правку в стороне
      // значило бы анализировать не то, что видит пользователь.
      const parsed = await api(`/api/projects/${state.project.id}/parse`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ chunk_strategy: $('chunk-strategy').value }),
      });
      state.sourceDirty = false;
      applyProject(parsed);
    }
    const report = await api(`/api/projects/${state.project.id}/analyze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ auto_accent: $('auto-accent').checked }),
    });
    await applyProject(await api(`/api/projects/${state.project.id}`), { analysis: false });
    await refreshAnalysis();
    if (report.status === 'error') {
      showAlert($('analyze-error'), report.errors.map((item) => item.text).join('; '));
    }
  } catch (error) {
    showAlert($('analyze-error'), error.message);
  } finally {
    state.analyzeBusy = false;
    updateAnalysisBar();
    updateGenerateButton();
  }
}

function renderReviewPanel() {
  const panel = $('review-panel');
  const list = $('review-list');
  const candidates = (state.analysis && state.analysis.candidates) || [];
  panel.hidden = !candidates.length;
  if (!candidates.length) {
    list.innerHTML = '';
    return;
  }
  $('review-title').textContent =
    `Слова для проверки: ${candidates.length}. Пока они не решены, синтез недоступен.`;
  $('review-note').textContent =
    '«Только в этот проект» — правило подействует здесь и не изменит другие диалоги: омограф '
    + 'почти всегда зависит от контекста. «Во все проекты» — для терминов и брендов, которые '
    + 'читаются одинаково везде. Пропуск оставляет слово как есть и больше его не предлагает.';
  list.innerHTML = candidates
    .map(
      (item, index) => `
      <div class="card review-row" data-index="${index}">
        <div class="row between">
          <span><b>${esc(item.word)}</b> → <span class="muted">${esc(item.target || 'нужна замена')}</span></span>
          <span class="muted">реплика ${item.replica_index + 1}</span>
        </div>
        <div class="muted review-reason">${esc(item.reason || '')}</div>
        <div class="row">
          <button class="tiny primary" data-review="project" data-index="${index}">Только в этот проект</button>
          <button class="tiny ghost" data-review="global" data-index="${index}">Во все проекты</button>
          <button class="tiny ghost" data-review="skip" data-index="${index}">Пропустить</button>
        </div>
      </div>`
    )
    .join('');
}

async function reviewCandidate(index, action) {
  const item = (state.analysis.candidates || [])[index];
  if (!item) return;
  const accepted = action !== 'skip' && Boolean((item.target || '').trim());
  try {
    // Решение уходит одним запросом: правило нужной области создаётся, затронутые
    // реплики помечаются устаревшими, а пересчитываются только они — слово
    // встретилось именно в этой реплике.
    await api(`/api/projects/${state.project.id}/pronunciation/review`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        source: item.word,
        target: item.target || item.word,
        scope: action === 'global' ? 'global' : 'project',
        enabled: accepted,
        replica_index: item.replica_index,
      }),
    });
    await applyProject(await api(`/api/projects/${state.project.id}`), { analysis: false });
    await refreshAnalysis();
    if (typeof loadDictionary === 'function') loadDictionary().catch(() => {});
  } catch (error) {
    showAlert($('analyze-error'), error.message);
  }
}

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
  const signal = staleSignal('project');
  try {
    const project = await api(`/api/projects/${projectId}`, { signal });
    if (!project.source_text) return;
    $('dialogue').value = project.source_text;
    applyProject(project);
  } catch (error) {
    // Отменённый запрос уступил место новому — проект запомнен, гасить его нельзя.
    if (isAbort(error)) return;
    // Проект могли удалить: начинаем с чистого листа, а не с ошибки на загрузке.
    rememberProject(null);
  }
  setSourceState();
}

function applyProject(project, { speakers = true, analysis = true } = {}) {
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
  updateAnalysisBar();
  updateGenerateButton();
  renderTimeline();
  refreshTimeline();
  // Состояние подготовки живёт отдельным запросом: оно меняется и без правок
  // проекта (словарь, инвалидация), поэтому обновляем его вместе с карточками.
  // Вызывающий, который уже получил свежий ответ, просит не повторять запрос.
  if (analysis) refreshAnalysis().catch(() => {});
  // Список архивов диагностики — принадлежность проекта: при переключении проекта
  // в списке не должно остаться файлов соседнего.
  loadDiagnostics();
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
      ? [engineNote(voice.engine), cloningNote(voice.engine)].filter(Boolean).join(' · ')
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
  // Анализ — не подсказка, а условие: сервер откажет в рендере без него (409),
  // поэтому кнопка гаснет по тому же признаку и объясняет причину.
  const status = analysisStatus();
  const prepared = Boolean(state.analysis) && status === 'ready';
  button.disabled = !(state.modelReady && assigned && prepared)
    || Boolean(state.jobId) || state.sourceDirty;
  // Кнопка отмены живёт ровно столько, сколько живёт задача.
  $('btn-cancel-job').hidden = !state.jobId;
  const reason = (state.analysis && state.analysis.analysis_error) || '';
  if (!state.modelReady) button.title = 'Модель ещё загружается';
  else if (!speakers.length) button.title = 'Сначала разберите текст на реплики';
  else if (state.sourceDirty) button.title = 'Текст изменён — нажмите «Применить и разобрать»';
  else if (!assigned) button.title = 'Выберите голос для каждого участника диалога';
  else if (!state.analysis) button.title = 'Состояние подготовки неизвестно — откройте вкладку заново';
  else if (status === 'needs_review') {
    button.title = 'Есть слова для проверки — подтвердите или пропустите их в панели «Слова для проверки»';
  } else if (status === 'error') {
    button.title = `Анализ не удался${reason ? `: ${reason}` : ''}`;
  } else if (status !== 'ready') {
    button.title = `Сначала выполните анализ диалога${reason ? ` (${reason})` : ''}`;
  } else button.title = '';
  // Кнопка экспорта живёт по другому условию — проекта с репликами, а не модели:
  // выгрузка ничего не синтезирует и от готовности движка не зависит.
  updateExportButtons();
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
        <div class="variant-row" data-take="${esc(take.id)}">
          <button class="tiny" data-role="take-play">${playing ? 'стоп' : 'слушать'}</button>
          <span class="variant-label">${esc(take.label || 'вариант')}${take.active ? ' · активно' : ''}</span>
          <span class="muted">${Number(take.duration_sec || 0).toFixed(1)} с · ${esc(seedText(take.seed))}</span>
          ${takeWarnings(take.quality)}
          ${shortNote(take)}
          ${qaNote(take.qa)}
          ${take.active ? '' : '<button class="tiny" data-role="take-pick">поставить</button>'}
        </div>
        ${takeDiagnostics(take.quality)}
      </div>`;
  }).join('');
  return `<div class="variant-list">${rows}</div>`;
}

// Что сделал слой коротких реплик с этим звучанием (§24): класс, стратегия и
// источник контекста лежат в параметрах куска, поэтому видны задним числом.
function shortNote(take) {
  const params = take.parameters || {};
  const strategy = params.short_utterance_strategy;
  const fallback = params.short_utterance_fallback || '';
  if (!strategy) return '';
  // Откат показываем явно: раньше он выглядел как «слой ничего не сделал», и
  // пользователь не мог отличить «выключено» от «не нашлась граница».
  if (strategy === 'direct' && !fallback) return '';
  const source = params.short_utterance_context_source || '';
  const klass = params.short_utterance_class || '';
  const parts = [];
  if (fallback) {
    parts.push(`короткая: откат — ${fallback}`);
  } else {
    parts.push(SHORT_STRATEGY_LABELS[strategy] || strategy);
    if (source) parts.push(SHORT_SOURCE_LABELS[source] || source);
    // Граница могла найдена запасным методом: это тоже часть решения.
    const method = params.short_utterance_boundary_method;
    if (method) parts.push(`граница: ${method}`);
  }
  const title = `Класс: ${klass}. Отпечаток синтез-текста: ${params.synthesis_text_hash || '—'}`;
  return `<span class="replica-qa muted" title="${esc(title)}">${esc(parts.join(' · '))}</span>`;
}

const SHORT_STRATEGY_LABELS = {
  punctuation: 'короткая: пунктуация',
  same_speaker_context: 'короткая: контекст спикера',
  synthetic_context: 'короткая: нейтральный контекст',
  batch_and_crop: 'короткая: группа',
};

const SHORT_SOURCE_LABELS = {
  same_speaker_previous: 'предыдущая реплика',
  same_speaker_next: 'следующая реплика',
  synthetic: 'носитель',
  group: 'группа',
};

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

      ${replicaProsodyHtml(replica)}

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

      ${replicaPreparedHtml(replica)}

      ${takesHtml(replica)}

      ${replicaPreviewHtml(replica)}
    </div>`;
}

// Интонация реплики: селектор профиля, фактический результат «Авто» и состояние
// референса. Интонация — метаданные, а не текст: она выбирает референс голоса
// (UPDATE 2 §11, UPDATE 3 §23–§28) и никогда не попадает в произносимую строку.
// Поэтому здесь только селектор и подписи — правка текста живёт в другом месте карточки.
const EMOTION_SOURCE_LABELS = {
  llm: 'модель',
  heuristic: 'по тексту',
  none: 'нет',
};

// Подписи значений (UPDATE 3 §38). Полный набор из 14, а не пятёрка UPDATE 2:
// без новых значений профиль голоса и детали реплики показывались бы кодом
// «SAD_SYMPATHETIC», а не словами.
const EMOTION_LABELS = {
  AUTO: 'Авто',
  NEUTRAL: 'Нейтрально',
  CALM: 'Спокойно',
  QUESTION: 'Вопрос',
  NEUTRAL_QUESTION: 'Вопрос + нейтрально',
  EXCLAMATION: 'Восклицание',
  DELIGHT: 'Радость / восторг',
  SAD_SYMPATHETIC: 'Огорчение / сочувствие',
  IRONIC: 'Ирония',
  STRICT: 'Строго',
  ENUMERATION: 'Перечисление',
  EXCITED: 'Взволнованно',
  SURPRISE: 'Удивление',
  FEAR: 'Испуг',
};

// Типы высказывания (controlled vocabulary LLM, 16 значений). В интерфейсе нужен
// русский: «QUESTION» рядом с «Интонация: Вопрос» читалось бы как второй код той
// же вещи, хотя это разные оси — что человек делает и как это звучит (§57).
const DIALOGUE_ACT_LABELS = {
  STATEMENT: 'утверждение',
  QUESTION: 'вопрос',
  ANSWER: 'ответ',
  REQUEST: 'просьба',
  COMMAND: 'приказ',
  REACTION: 'реакция',
  COMPLIMENT: 'комплимент',
  REFUSAL: 'отказ',
  AGREEMENT: 'согласие',
  DISAGREEMENT: 'несогласие',
  GREETING: 'приветствие',
  FAREWELL: 'прощание',
  WARNING: 'предупреждение',
  EXCLAMATION: 'восклицание',
  ENUMERATION: 'перечисление',
  OTHER: 'другое',
};

const PROSODY_PACE_LABELS = {
  SLOW: 'медленный',
  NORMAL: 'обычный',
  FAST: 'быстрый',
};

const CONTEXT_DEPENDENCY_LABELS = {
  LOW: 'низкая',
  MEDIUM: 'средняя',
  HIGH: 'высокая',
};

// Ниже этого порога авто-интонация показывается словами «низкая уверенность».
// Число само по себе не помогает: 0.41 и 0.58 на глаз неразличимы, а решение
// «верить автоматике или выбрать руками» пользователь принимает именно по этому
// порогу (§39). Сам процент остаётся в деталях — там, где его читают осознанно.
const LOW_CONFIDENCE = 0.6;

function emotionLabel(value) {
  return EMOTION_LABELS[value] || value || '—';
}

function confidenceText(value) {
  if (typeof value !== 'number' || !isFinite(value) || value <= 0) return '';
  if (value < LOW_CONFIDENCE) return 'низкая уверенность';
  return `${Math.round(value * 100)}%`;
}

function replicaProsodyHtml(replica) {
  const prosody = replica.prosody || {};
  const reference = replica.reference || {};
  // Каталог интонаций считается на голос, а не на реплику: у одного голоса набор
  // один и тот же. Из него берутся только записываемые профили (§10) плюс «Авто»;
  // семантические SURPRISE/FEAR остаются в списке лишь тогда, когда именно они
  // выбраны руками раньше — иначе селект показывал бы «Авто» вместо правды (§17).
  const speaker = speakerByKey(replica.speaker);
  const catalog = (speaker && speaker.emotions && speaker.emotions.length)
    ? speaker.emotions
    : [{ value: 'AUTO', title: 'Авто' }];
  const selected = prosody.override || 'AUTO';
  const offered = catalog.filter((item) => (
    item.value === 'AUTO' || item.profile || item.value === selected
  ));
  const options = offered
    .map((item) => `<option value="${esc(item.value)}"${item.value === selected ? ' selected' : ''}>${esc(item.title || emotionLabel(item.value))}</option>`)
    .join('');

  const effective = prosody.effective_title || emotionLabel(prosody.effective);
  const source = prosody.source || 'none';
  const confidence = confidenceText(prosody.confidence);
  const act = prosody.dialogue_act
    ? DIALOGUE_ACT_LABELS[prosody.dialogue_act] || prosody.dialogue_act
    : '';
  // «Авто → Ирония · низкая уверенность (по тексту)» — результат автоматики и то,
  // откуда он взялся. Ручной выбор показывается без стрелки: это решение
  // пользователя, а не догадка.
  const result = selected === 'AUTO'
    ? `Авто → ${esc(effective)}`
      + (confidence ? ` · <b>${esc(confidence)}</b>` : '')
      + ` <span class="muted">(${esc(EMOTION_SOURCE_LABELS[source] || source)})</span>`
    : `выбрано: ${esc(effective)}`;

  const referenceTitle = reference.emotion
    ? emotionLabel(reference.emotion)
    : (reference.profile_id ? reference.profile_id : 'ещё не синтезировалось');
  const fallback = reference.fallback_used
    ? `есть — ${reference.fallback_reason || 'профиль интонации недоступен'}`
    : 'нет';

  const details = [
    ['Распознано моделью', prosody.profile ? emotionLabel(prosody.profile) : '—'],
    ['Уверенность', typeof prosody.confidence === 'number' ? `${Math.round(prosody.confidence * 100)}%` : '—'],
    ['Тип высказывания', act || '—'],
    ['Связь с контекстом', CONTEXT_DEPENDENCY_LABELS[prosody.context_dependency] || prosody.context_dependency || '—'],
    ['Насыщенность', typeof prosody.intensity === 'number' ? prosody.intensity.toFixed(2) : '—'],
    ['Темп', PROSODY_PACE_LABELS[prosody.pace] || prosody.pace || '—'],
    ['Reference', referenceTitle],
    ['Fallback', fallback],
    ['Профиль референса', reference.profile_key ? emotionLabel(reference.profile_key) : '—'],
  ];
  const detailRows = details
    .map(([name, value]) => `<div class="row between"><span class="muted">${esc(name)}</span><span>${esc(value)}</span></div>`)
    .join('');

  return `
      <div class="field" style="margin-top: 4px">
        <span>Интонация</span>
        <div class="grid-2">
          <select data-role="prosody">${options}</select>
          <span class="muted">${result}${act ? ` · ${esc(act)}` : ''}</span>
        </div>
        <details class="advanced prosody-details"${state.openDetails[`prosody-${replica.index}`] ? ' open' : ''}>
          <summary>Детали интонации и референса</summary>
          ${detailRows}
          <span class="muted">
            Интонация выбирает референс этого же голоса: своя запись для интонации
            звучит ею, а если записи нет — нейтральной записью того же голоса, и об
            этом скажет строка «Fallback». Ручной выбор важнее автоматического.
          </span>
        </details>
      </div>`;
}

// Подготовленный текст реплики: то, что реально уйдёт в модель, и предупреждения.
// Отдельно от панели «Что услышит модель»: та считает стадии заново (полезно при
// подборе параметров), а здесь — **сохранённый** результат анализа, по которому и
// пойдёт рендер. Разойтись они не могут, но показывают разное: preview отвечает
// «что будет», этот блок — «что уже зафиксировано».
const REPLICA_ANALYSIS_LABELS = {
  pending: 'ждёт анализа',
  done: 'подготовлено',
  error: 'ошибка подготовки',
};

function replicaPreparedHtml(replica) {
  const status = replica.analysis_status || 'pending';
  const label = REPLICA_ANALYSIS_LABELS[status] || status;
  const candidates = replica.pronunciation_candidates || [];
  const matches = replica.dictionary_matches || [];
  const warnings = [];
  if (status === 'error') {
    warnings.push(replica.analysis_error || 'реплику не удалось подготовить');
  }
  if (status === 'done' && replica.auto_accent && replica.supports_accents === false) {
    warnings.push(
      `движок «${engineLabel(replica.engine)}» не поддерживает ударения — текст уйдёт без разметки`
    );
  }
  if (candidates.length) {
    warnings.push(
      `слов на проверку: ${candidates.length} (${candidates.map((item) => item.word).join(', ')})`
    );
  }
  const stage = (title, body) => `
    <p class="eyebrow" style="margin:10px 0 4px">${title}</p>
    <blockquote class="ref-phrase">${esc(body || '—')}</blockquote>`;
  const final = replica.final_text || '';
  const body = status === 'pending'
    ? '<p class="hint muted" style="margin-top:6px">Реплика ещё не проанализирована: нажмите «Анализировать диалог». Рендер её не возьмёт.</p>'
    : status === 'error'
    ? '<p class="hint muted" style="margin-top:6px">Подготовка не удалась — текста для модели нет. '
      + 'Устраните причину и запустите анализ ещё раз.</p>'
    : stage('Исходный текст', replica.source_text || replica.text)
      + stage('Нормализация', replica.normalized_text)
      + stage('Восстановление ё', replica.yo_text)
      + stage('Словарь произношения', replica.dictionary_text)
      + stage(
          replica.supports_accents ? 'Ударения' : 'Ударения (движок не поддерживает)',
          replica.accentized_text
        )
      + stage('Уйдёт в модель', final)
      + (matches.length
        ? `<div class="tag-row" style="margin-top:6px">${matches.map((match) => (
            `<span class="tag">${esc(match.source)} → ${esc(match.target)} ×${match.count}</span>`
          )).join('')}</div>`
        : '<p class="hint muted" style="margin-top:6px">Словарь эту реплику не менял.</p>');
  return `
    <details class="advanced prepared"${state.preparedDetails[replica.index] ? ' open' : ''}>
      <summary>Подготовленный текст: ${esc(label)}</summary>
      ${warnings.length
        ? `<div class="alert info">${warnings.map((item) => esc(item)).join('<br />')}</div>`
        : ''}
      ${body}
    </details>`;
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
// Настройки коротких реплик из панели (§23). Префикс различает панели диалога и
// сплошного текста: у них свои наборы полей, а серверу уходит одно и то же.
function shortUtterancePayload(prefix = '') {
  const enabledEl = $(`${prefix}short-enabled`);
  if (!enabledEl || !enabledEl.checked) return { enabled: false };
  const veryShort = parseInt($(`${prefix}short-very-short`).value, 10);
  const short = parseInt($(`${prefix}short-short`).value, 10);
  return {
    enabled: true,
    strategy: $(`${prefix}short-strategy`).value,
    thresholds: {
      very_short_words: Number.isFinite(veryShort) ? veryShort : undefined,
      short_words: Number.isFinite(short) ? short : undefined,
    },
  };
}

// Политика приложения из /api/status: подставляем измеренные значения, чтобы
// «Авто» в панели означало именно то, что произойдёт на сервере.
function applyShortPolicy(status) {
  const policy = status && status.short_utterance;
  if (!policy) return;
  const thresholds = policy.thresholds || {};
  for (const prefix of ['', 'text-']) {
    const enabled = $(`${prefix}short-enabled`);
    if (!enabled) continue;
    enabled.checked = Boolean(policy.enabled);
    if (thresholds.very_short_words) {
      $(`${prefix}short-very-short`).value = thresholds.very_short_words;
    }
    if (thresholds.short_words) {
      $(`${prefix}short-short`).value = thresholds.short_words;
    }
  }
  state.shortPolicy = policy;
}

// Прогрев коротких реплик: галочка показывает политику приложения из /api/status,
// а не собственную догадку интерфейса — как и «Авто» у короткого слоя.
function applyWarmupPolicy(status) {
  const box = $('warmup-enabled');
  if (!box) return;
  const policy = (status && status.warmup) || {};
  box.checked = Boolean(policy.enabled);
  box.title = policy.enabled
    ? `Прогрев включён для движков: ${(policy.engines || []).join(', ') || '—'}`
    : 'Прогрев выключен в настройках приложения (TTS_WARMUP_ENABLED)';
}

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
    short_utterance: shortUtterancePayload(),
    // Прогрев коротких реплик: скрытый контекст перед короткой фразой. Технический
    // текст — в готовое аудио не попадает и текст реплики не меняет.
    warmup_short_replicas: $('warmup-enabled').checked,
    // Имя готового файла: свойство запуска, а не проекта. Пустое поле означает
    // прежнее поведение — имя из номера задачи.
    output_name: $('output-name').value.trim(),
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
      // Имя файла с бэкенда: если пользователь задал своё, оно уже там;
      // `output_format` остаётся запасным вариантом для старых ответов.
      link.download = job.file_name || `dialogue.${job.output_format}`;
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

// --- диагностика качества озвучки ----------------------------------------------
// Архив диагностики собирается на сервере, а не в браузере: только там есть
// настройки рендера, план коротких реплик и готовое аудио. Браузер просит собрать
// (POST) и забирает готовый файл (GET), потому что сборка — это работа, а не
// скачивание: тело ответа с параметрами и списком архивов интерфейсу тоже нужно.
function diagnosticsJobId() {
  // Сначала задача этой сессии: именно её звучание слушал пользователь. Затем
  // последний рендер проекта — он переживает перезапуск приложения.
  return state.jobId || (state.project && state.project.job_id) || null;
}

function renderDiagnosticsList(archives = []) {
  const mount = $('diagnostics-list');
  if (!mount) return;
  mount.innerHTML = '';
  if (!archives.length) return;
  const title = document.createElement('span');
  title.className = 'muted';
  title.textContent = 'Собранные архивы:';
  mount.appendChild(title);
  const list = document.createElement('ul');
  list.className = 'muted';
  archives.forEach((item) => {
    const row = document.createElement('li');
    const link = document.createElement('a');
    link.href = `/api/projects/${state.project.id}/diagnostics/${encodeURIComponent(item.name)}`;
    link.textContent = `${item.name} (${item.size_mb} МБ)`;
    const remove = document.createElement('button');
    remove.type = 'button';
    remove.className = 'link-button';
    remove.textContent = 'удалить';
    remove.addEventListener('click', () => deleteDiagnostics(item.name));
    row.append(link, document.createTextNode(' '), remove);
    list.appendChild(row);
  });
  mount.appendChild(list);
}

async function exportDiagnostics() {
  if (!state.project) {
    showAlert($('export-error'), 'Проект ещё не создан — диагностику собирать не из чего.');
    return;
  }
  const button = $('btn-diagnostics');
  showAlert($('export-error'), '');
  button.disabled = true;
  $('diagnostics-status').textContent = 'собираю архив…';
  try {
    const payload = {
      job_id: diagnosticsJobId(),
      include_references: $('diagnostics-references').checked,
    };
    const result = await api(`/api/projects/${state.project.id}/diagnostics`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    await downloadExport(result.download_url, 'diagnostics.zip');
    const warnings = (result.bundle && result.bundle.warnings) || [];
    $('diagnostics-status').textContent =
      `архив ${result.bundle.name} · ${result.bundle.size_mb} МБ · ${result.path}`
      + (warnings.length ? ` · предупреждений: ${warnings.length}` : '');
    renderDiagnosticsList(result.archives || []);
  } catch (error) {
    $('diagnostics-status').textContent = '—';
    showAlert($('export-error'), error.message);
  } finally {
    button.disabled = false;
  }
}

async function deleteDiagnostics(name) {
  showAlert($('export-error'), '');
  try {
    await api(
      `/api/projects/${state.project.id}/diagnostics/${encodeURIComponent(name)}`,
      { method: 'DELETE' },
    );
    $('diagnostics-status').textContent = `архив ${name} удалён`;
    await loadDiagnostics();
  } catch (error) {
    showAlert($('export-error'), error.message);
  }
}

async function loadDiagnostics() {
  if (!state.project || !$('diagnostics-list')) return;
  const signal = staleSignal('diagnostics');
  try {
    const data = await api(`/api/projects/${state.project.id}/diagnostics`, { signal });
    renderDiagnosticsList(data.archives || []);
  } catch (error) {
    // Отменённый запрос уступил место новому: список покажет он.
    if (isAbort(error)) return;
    // Список архивов вторичен: без него кнопка «Собрать» всё равно работает.
  }
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
  state.textEngineParams = chosenEngineParams(
    voice.engine,
    sameVoice ? readEngineParams(box) : voice.engine_params,
  );
  box.innerHTML = engineParamsHtml(voice.engine, state.textEngineParams);
  $('text-engine-note').textContent = [engineNote(voice.engine), cloningNote(voice.engine)]
    .filter(Boolean)
    .join(' · ');
  $('text-f5-params').hidden = voice.engine !== ENGINE_F5;
}

// Ручки сплошного текста: выбор пользователя, без значений, которые и так следуют
// из режима и дефолтов движка (см. chosenEngineParams). Отправить их явно — значит
// перекрыть пресет режима, и «черновик» перестал бы отличаться от «качества».
function readTextEngineParams() {
  const voice = voiceById($('text-voice').value);
  return voice ? chosenEngineParams(voice.engine, readEngineParams($('text-engine-params'))) : {};
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

const DIALOGUE_FILE_SUFFIXES = ['.txt', '.md'];
// Ограничение то же, что у сплошного текста: сервер откажет в тексте длиннее
// MAX_TEXT_CHARS, и лучше сказать об этом до чтения файла, а не после запроса.
const DIALOGUE_FILE_MAX_BYTES = 2 * 1024 * 1024;

function loadDialogueFile(file) {
  if (!file) return;
  const note = $('dialogue-file-note');
  const name = String(file.name || '').toLowerCase();
  const suffix = DIALOGUE_FILE_SUFFIXES.find((item) => name.endsWith(item));
  if (!suffix) {
    note.hidden = false;
    showAlert($('parse-error'), `Файл ${file.name}: поддерживаются .txt и .md`);
    return;
  }
  if (file.size === 0) {
    note.hidden = false;
    showAlert($('parse-error'), `Файл ${file.name} пуст — вставьте диалог вручную`);
    return;
  }
  if (file.size > DIALOGUE_FILE_MAX_BYTES) {
    note.hidden = false;
    showAlert(
      $('parse-error'),
      `Файл ${file.name} больше ${Math.round(DIALOGUE_FILE_MAX_BYTES / 1024 / 1024)} МБ — разбейте его на части`
    );
    return;
  }
  const reader = new FileReader();
  reader.onload = () => {
    const text = String(reader.result || '');
    if (!text.trim()) {
      showAlert($('parse-error'), `В файле ${file.name} нет текста`);
      return;
    }
    if (text.includes('\uFFFD')) {
      // Браузер заменяет нечитаемые байты на «�», а не падает: молча
      // отправить такой текст в модель значит получить мусор в аудио.
      showAlert(
        $('parse-error'),
        `Файл ${file.name} не в UTF-8 — пересохраните его в UTF-8 и попробуйте снова`
      );
      return;
    }
    showAlert($('parse-error'), '');
    $('dialogue').value = text;
    // Текст считается изменённым: разбор и анализ запускает пользователь, и это
    // тот же путь, что для вставленного текста.
    markSourceDirty();
    note.hidden = false;
    note.textContent = `Загружен файл ${file.name}: ${text.trim().length} символов. Нажмите «Применить и разобрать», затем «Анализировать диалог».`;
    setSourceState();
  };
  reader.onerror = () => showAlert($('parse-error'), `Не удалось прочитать файл ${file.name}`);
  // Испорченная кодировка не должна превращаться в мусор в модели: decode с
  // фатальной ошибкой лучше тихой подмены символов.
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
    engine_params: readTextEngineParams(),
    gain_db: parseFloat($('text-gain').value),
    pitch_semitones: parseFloat($('text-pitch').value),
    target_rms: parseFloat($('text-rms').value),
    pause_ms: parseInt($('text-pause').value, 10),
    cross_fade_duration: parseFloat($('text-crossfade').value),
    auto_accent: $('text-auto-accent').checked,
    qa: $('text-qa').value,
    short_utterance: shortUtterancePayload('text-'),
    output_format: $('text-output-format').value,
    chunk_strategy: $('text-chunk-strategy').value,
    output_name: $('text-output-name').value.trim(),
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
      link.download = job.file_name || `text.${job.output_format}`;
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
    applyShortPolicy(status);
    applyWarmupPolicy(status);
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
    // Интонация — метаданные реплики, а не текст: смена только выбирает референс и
    // не требует повторного анализа (§11). Ручной выбор один и задаёт и эмоцию, и
    // интонацию — это одно поле `emotion_override`, а не два (§40). `AUTO` снимает
    // ручной выбор (null) и возвращает реплику автоматике.
    if (event.target.dataset.role === 'prosody') {
      const value = event.target.value || 'AUTO';
      patchReplica(index, { emotion_override: value === 'AUTO' ? null : value });
      return;
    }
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
    const index = details.closest('.replica-card').dataset.index;
    // У каждого раскрывающегося блока своё состояние: общий флаг открывал бы вместе
    // с ним и остальные — три разных вопроса с одним ответом.
    if (details.classList.contains('prepared')) state.preparedDetails[index] = details.open;
    else if (details.classList.contains('prosody-details')) {
      state.openDetails[`prosody-${index}`] = details.open;
    } else state.openDetails[index] = details.open;
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
  $('btn-analyze').addEventListener('click', analyzeDialogue);
  // Переключатель анализатора: без него «выключен» нельзя было исправить из
  // приложения, и в панели оставалось только сообщение про LLM_ANALYZER_ENABLED=0.
  const llmEnabled = $('llm-enabled');
  if (llmEnabled) {
    llmEnabled.addEventListener('change', () => saveLlmSettings({ enabled: llmEnabled.checked }));
    $('llm-required').addEventListener('change', () =>
      saveLlmSettings({ required_for_render: $('llm-required').checked })
    );
    $('llm-model').addEventListener('change', () =>
      saveLlmSettings({ primary_model: $('llm-model').value })
    );
    loadLlmModels();
    refreshLlmAnalysis();
  }
  const dialogueFile = $('dialogue-file');
  $('dialogue-drop').addEventListener('click', () => dialogueFile.click());
  dialogueFile.addEventListener('change', () => {
    loadDialogueFile(dialogueFile.files[0]);
    dialogueFile.value = ''; // повторный выбор того же файла должен срабатывать
  });
  $('source-view').addEventListener('dragover', (event) => {
    event.preventDefault();
  });
  $('source-view').addEventListener('drop', (event) => {
    event.preventDefault();
    loadDialogueFile(event.dataTransfer.files[0]);
  });
  bindReplicaCards();
  $('review-list').addEventListener('click', (event) => {
    const button = event.target.closest('[data-review]');
    if (button) reviewCandidate(Number(button.dataset.index), button.dataset.review);
  });
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
  bindCacheEvents();

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
    if (event.target.dataset.engineParam) {
      updateEngineParamLabel($('text-engine-params'), event.target);
      state.textEngineParams = readTextEngineParams();
    }
  });
  // Смена режима перерисовывает блок: пресет нового режима мог поменять ручки,
  // и ползунки обязаны показывать то, чем движок читает текст.
  $('text-engine-params').addEventListener('change', (event) => {
    if (event.target.dataset.engineMode === undefined) return;
    const engine = (voiceById($('text-voice').value) || {}).engine;
    state.textEngineParams = applyEngineMode(
      $('text-engine-params'), engine, state.textEngineParams, event.target.value,
    );
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
  // Смена цели меняет и план записи, и текущую фразу: у нового голоса свой набор
  // уже записанного, и вести мастера надо с его первой незакрытой фразы.
  $('record-target').addEventListener('change', () => {
    renderRecordPlan();
    advanceRecordPhrase();
  });
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
      cfg.engine_params = chosenEngineParams(
        (voiceById(card.dataset.voiceId) || {}).engine,
        readEngineParams(card),
      );
    }
  });

  voiceCards.addEventListener('change', async (event) => {
    const card = event.target.closest('.card');
    if (!card) return;
    const voiceId = card.dataset.voiceId;
    const role = event.target.dataset.role;
    if (role === 'reference-file') {
      // Загрузка эмоционального референса: расшифровку не проверяем (`false`) —
      // запись сделана по показанной фразе, и распознавание только удлинило бы шаг.
      const file = event.target.files && event.target.files[0];
      event.target.value = '';
      const emotion = event.target.dataset.emotion || 'NEUTRAL';
      if (!file) return;
      setPreviewStatus(card, 'загружаю референс…');
      try {
        await addVoiceReference(voiceId, emotion, file);
        await loadVoices();
        renderVoiceCards();
      } catch (error) {
        setPreviewStatus(card, error.message, true);
      }
      return;
    }
    if (role === 'reference-auto') {
      // Подтверждение профиля (§35) — это доверие, а не настройка: сохраняем сразу.
      // Карточки не пересобираем: `loadVoices` перерисовывает их и закрыл бы
      // раскрытый список референсов прямо под рукой. Вместо перерисовки берём
      // новый список профилей из ответа сервера — он и есть источник галочки.
      try {
        const answer = await updateVoiceReference(voiceId, event.target.dataset.profile, {
          enabled_for_auto: event.target.checked,
        });
        const voice = voiceById(voiceId);
        if (voice) voice.reference_profiles = answer.reference_profiles;
      } catch (error) {
        // Флаг не сохранён — галочка обязана вернуться к настоящему состоянию.
        event.target.checked = !event.target.checked;
        setPreviewStatus(card, error.message, true);
      }
      return;
    }
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
    // Смена режима перерисовывает блок ручек: пресет нового режима мог поменять
    // их значения, и ползунки обязаны показывать то, чем движок читает текст.
    if (event.target.dataset.engineMode !== undefined) {
      const engine = (voiceById(voiceId) || {}).engine;
      const params = applyEngineMode(card.querySelector('[data-role="engine-params"]'), engine, state.preview[voiceId].engine_params, event.target.value);
      state.preview[voiceId].engine_params = params;
      saveVoiceEngineParams(voiceId, params);
      return;
    }
    if (event.target.dataset.engineParam) {
      // Значения ручек сохраняются у голоса — это его настройки по умолчанию,
      // карточка слота в диалоге может переопределить их на одну генерацию.
      const params = chosenEngineParams((voiceById(voiceId) || {}).engine, readEngineParams(card));
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
    if (button.dataset.role === 'reference-add') {
      // Файл выбирает пользователь: запись эмоции — это отдельный референс, и
      // подставлять вместо неё что-то другое нельзя (§38).
      const card = button.closest('.card');
      const input = card.querySelector('[data-role="reference-file"]');
      input.dataset.emotion = card.querySelector('[data-role="reference-emotion"]').value;
      input.click();
      return;
    }
    if (button.dataset.role === 'reference-delete') {
      try {
        await deleteVoiceReference(voiceId, button.dataset.profile);
        await loadVoices();
        renderVoiceCards();
      } catch (error) {
        setPreviewStatus(button.closest('.card'), error.message, true);
      }
      return;
    }
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

// --- вкладка «Сам себе звукорежиссер» -----------------------------------------
// Источник истины — backend. Каждый project-эндпоинт возвращает полный payload,
// и он целиком заменяет `state.recording.project`: только так готовность,
// активные дубли и роли остаются согласованными после удаления дубля или
// повторного разбора. Локально хранится лишь то, что не является состоянием
// проекта: открытая реплика, выбранный микрофон, несохранённые ползунки и
// записанный, но ещё не отправленный дубль.
const RECORDING_BASE = '/api/recording-projects';
const RECORDING_KEY = 'voiceStudio.recordingProjectId';
const RECORDING_POLL_MS = 1000;
// Границы ручек обработки. В backend/config.py диапазоны шире (0.5–2.0 и ±12),
// но здесь задача фазы сужает их: на человеческой записи более резкая правка
// скорости и высоты слышна как артефакт, а не как режиссура.
const RECORDING_SPEED = { min: 0.75, max: 1.25, step: 0.05 };
const RECORDING_PITCH = { min: -6, max: 6, step: 1 };
const RECORDING_DEFAULT_PAUSE = 400;

const fmtRecTime = (seconds) => {
  const value = Math.max(0, Number(seconds) || 0);
  const minutes = Math.floor(value / 60);
  const secs = Math.floor(value % 60);
  const tenths = Math.floor((value * 10) % 10);
  return `${String(minutes).padStart(2, '0')}:${String(secs).padStart(2, '0')}.${tenths}`;
};

const fmtRecDuration = (seconds) => {
  const value = Number(seconds) || 0;
  return value >= 60
    ? `${Math.floor(value / 60)} мин ${(value % 60).toFixed(1)} с`
    : `${value.toFixed(1)} с`;
};

// «0 пт», «+3 пт», «−2 пт»: знак нужен, потому что ноль и плюс читаются
// по-разному, а ползунок ходит в обе стороны.
const fmtRecPitch = (value) => {
  const number = Number(value) || 0;
  const text = Number.isInteger(number) ? String(number) : number.toFixed(1);
  return `${number > 0 ? '+' : ''}${text} пт`;
};

function recPlural(count, forms) {
  const value = Math.abs(Number(count) || 0) % 100;
  const last = value % 10;
  if (value > 10 && value < 20) return forms[2];
  if (last === 1) return forms[0];
  if (last >= 2 && last <= 4) return forms[1];
  return forms[2];
}

const recordingId = () => (state.recording.project ? state.recording.project.id : null);

const recordingActiveTake = (index) => ((state.recording.project && state.recording.project.takes) || [])
  .find((take) => take.replica_index === index && take.active) || null;

const recordingTakesFor = (index) => ((state.recording.project && state.recording.project.takes) || [])
  .filter((take) => take.replica_index === index)
  .sort((a, b) => String(a.created_at).localeCompare(String(b.created_at)));

function recordingReplicaAt(index) {
  const project = state.recording.project;
  if (!project) return null;
  return (project.replicas || []).find((item) => item.index === index) || null;
}

function recordingProfileValues(profileId) {
  const project = state.recording.project;
  const profile = ((project && project.voice_profiles) || []).find((item) => item.id === profileId);
  return profile
    ? { speed: profile.speed, pitch: profile.pitch_semitones, denoise: profile.denoise }
    : { speed: 1, pitch: 0, denoise: false };
}

function recordingJson(method, payload) {
  return {
    method,
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  };
}

// Порядок обхода при записи. «По диалогу» — как реплики стоят в тексте,
// «по ролям» — роли группами, внутри группы по порядку диалога. На монтаж это
// не влияет: он всегда идёт по `index` реплики (см. recording_pipeline.assemble).
function recordingNavOrder() {
  const project = state.recording.project;
  if (!project) return [];
  const replicas = [...(project.replicas || [])].sort((a, b) => a.index - b.index);
  if (project.recording_order !== 'roles') return replicas.map((replica) => replica.index);
  const order = [];
  (project.roles || []).forEach((role) => {
    order.push(...[...(role.replicas || [])].sort((a, b) => a - b));
  });
  return order;
}

// Ближайшая незаписанная реплика после текущей — по кругу: с последней реплики
// логично вернуться к пропущенной в начале, а не останавливаться.
function recordingNextMissingIndex(from) {
  const order = recordingNavOrder();
  if (!order.length) return null;
  const start = order.indexOf(from);
  const sequence = start >= 0
    ? order.slice(start + 1).concat(order.slice(0, start + 1))
    : order;
  const found = sequence.find((index) => !recordingActiveTake(index));
  return found === undefined ? null : found;
}

function rememberRecording(projectId) {
  try {
    if (projectId) localStorage.setItem(RECORDING_KEY, projectId);
    else localStorage.removeItem(RECORDING_KEY);
  } catch (_) { /* приватный режим — просто не запоминаем */ }
}

// --- загрузка проекта ---------------------------------------------------------
async function loadRecordingProjects() {
  const data = await api(RECORDING_BASE);
  state.recording.projects = data.projects || [];
  renderRecordingProjectSelect();
}

async function loadRecordingTab() {
  try {
    await loadRecordingProjects();
  } catch (error) {
    showAlert($('rec-dialogue-error'), error.message);
    return;
  }
  if (!state.recording.loaded) {
    state.recording.loaded = true;
    let saved = null;
    try { saved = localStorage.getItem(RECORDING_KEY); } catch (_) { /* приватный режим */ }
    if (saved && state.recording.projects.some((item) => item.id === saved)) {
      try {
        applyRecordingProject(await api(`${RECORDING_BASE}/${saved}`));
      } catch (_) {
        // Проект могли удалить: начинаем с чистого листа, а не с ошибки.
        rememberRecording(null);
        renderRecordingAll();
      }
    } else {
      renderRecordingAll();
    }
  } else if (recordingId()) {
    try {
      applyRecordingProject(await api(`${RECORDING_BASE}/${recordingId()}`));
    } catch (_) { /* проект удалён — ниже останется прежняя картинка */ }
  } else {
    renderRecordingAll();
  }
  // Монтаж мог идти, пока вкладка была закрыта: возобновляем опрос, а не
  // показываем замерший прогресс.
  const render = state.recording.project && state.recording.project.render;
  if (render && render.status === 'running') startRecordingRenderPoll();
  refreshRecordingDevices().catch(() => {});
}

function applyRecordingProject(project) {
  state.recording.project = project;
  // Живой прогресс из /render/status больше не нужен: тот же статус пришёл
  // в payload и уже перезаписал прошлое значение.
  state.recording.render = null;
  const order = recordingNavOrder();
  if (order.length && !order.includes(state.recording.activeIndex)) {
    state.recording.activeIndex = order[0];
  }
  if (!order.length) state.recording.activeIndex = 0;
  renderRecordingAll();
}

function renderRecordingAll() {
  renderRecordingDialogue();
  renderRecordingReadiness();
  renderRecordingRoles();
  renderRecordingReplica();
  renderRecordingTakes();
  renderRecordingMontage();
  renderRecordingSettings();
  renderRecordingOutput();
}

// --- панель 1: диалог ---------------------------------------------------------
function renderRecordingProjectSelect() {
  const select = $('rec-project-select');
  if (!select) return;
  const current = recordingId() || '';
  select.innerHTML = [
    '<option value="">— новый проект —</option>',
    ...state.recording.projects.map((item) => {
      const ready = item.replicas ? ` · ${item.recorded}/${item.replicas}` : '';
      return `<option value="${esc(item.id)}">${esc(item.name)}${esc(ready)}</option>`;
    }),
  ].join('');
  select.value = state.recording.projects.some((item) => item.id === current) ? current : '';
}

function renderRecordingDialogue() {
  const project = state.recording.project;
  // Имя и текст перечитываются из проекта только когда пользователь их не
  // правил: иначе любая перерисовка (например, после сохранения дубля) или
  // смена проекта затирала бы уже набранное название нового проекта.
  if (!state.recording.nameDirty && document.activeElement !== $('rec-name')) {
    $('rec-name').value = project ? project.name : '';
  }
  if (!state.recording.dialogueDirty) {
    $('rec-dialogue').value = project ? project.dialogue_text : '';
  }
  $('rec-btn-apply').textContent = project ? 'Разобрать диалог' : 'Создать проект';
  $('rec-btn-reparse').hidden = !project;
  $('rec-btn-delete').hidden = !project;
  const total = project ? (project.replicas || []).length : 0;
  const roles = project ? (project.speakers || []).length : 0;
  $('rec-summary').textContent = project
    ? `${total} ${recPlural(total, ['реплика', 'реплики', 'реплик'])} · `
      + `${roles} ${recPlural(roles, ['роль', 'роли', 'ролей'])}`
    : 'Проект не открыт';
  $('rec-source-state').textContent = state.recording.dialogueDirty
    ? 'текст изменён — нужен разбор'
    : '';
  renderRecordingProjectSelect();
}

function renderRecordingReadiness() {
  const project = state.recording.project;
  const readiness = project && project.readiness;
  $('rec-readiness').textContent = readiness
    ? `Записано ${readiness.replicas_recorded} / ${readiness.replicas_total} · `
      + `ролей готово ${readiness.roles_ready} / ${readiness.roles_total}`
    : '—';
}

// Создание проекта и повторный разбор — одно действие для пользователя:
// backend при PATCH с `dialogue_text` разбирает диалог тем же парсером.
async function applyRecordingDialogue() {
  showAlert($('rec-dialogue-error'), '');
  const text = $('rec-dialogue').value;
  if (!text.trim()) {
    showAlert($('rec-dialogue-error'), 'Вставьте текст диалога — разбирать нечего.');
    return;
  }
  const name = $('rec-name').value.trim() || 'Запись';
  const id = recordingId();
  try {
    const project = id
      ? await api(`${RECORDING_BASE}/${id}`, recordingJson('PATCH', { name, dialogue_text: text }))
      : await api(RECORDING_BASE, recordingJson('POST', { name, dialogue_text: text }));
    state.recording.dialogueDirty = false;
    state.recording.nameDirty = false;
    rememberRecording(project.id);
    applyRecordingProject(project);
    await loadRecordingProjects();
    renderRecordingProjectSelect();
    if (!(project.replicas || []).length) {
      showAlert($('rec-dialogue-error'), 'В тексте не нашлось реплик — проверьте формат.', 'info');
    }
  } catch (error) {
    showAlert($('rec-dialogue-error'), error.message);
  }
}

async function reparseRecordingProject() {
  const id = recordingId();
  if (!id) return;
  showAlert($('rec-dialogue-error'), '');
  try {
    const project = await api(`${RECORDING_BASE}/${id}/parse`, { method: 'POST' });
    state.recording.dialogueDirty = false;
    state.recording.nameDirty = false;
    applyRecordingProject(project);
    await loadRecordingProjects();
  } catch (error) {
    showAlert($('rec-dialogue-error'), error.message);
  }
}

async function openRecordingProject(projectId) {
  showAlert($('rec-dialogue-error'), '');
  // Смена проекта сбрасывает локальные правки: иначе в поле остался бы текст
  // прежнего проекта, а открытым считался бы новый.
  state.recording.dialogueDirty = false;
  state.recording.nameDirty = false;
  state.recording.dirtyProfiles = {};
  if (!projectId) {
    stopRecordingRenderPoll();
    stopRecordingCapture();
    discardRecordingTake();
    state.recording.project = null;
    state.recording.render = null;
    state.recording.activeIndex = 0;
    rememberRecording(null);
    renderRecordingAll();
    return;
  }
  applyRecordingProject(await api(`${RECORDING_BASE}/${projectId}`));
  rememberRecording(projectId);
  const render = state.recording.project.render;
  if (render && render.status === 'running') startRecordingRenderPoll();
}

async function deleteRecordingProject() {
  const id = recordingId();
  if (!id) return;
  if (!window.confirm('Удалить проект записи вместе со всеми дублями?')) return;
  showAlert($('rec-dialogue-error'), '');
  try {
    await api(`${RECORDING_BASE}/${id}`, { method: 'DELETE' });
    stopRecordingRenderPoll();
    stopRecordingCapture();
    discardRecordingTake();
    state.recording.project = null;
    state.recording.render = null;
    state.recording.activeIndex = 0;
    state.recording.dialogueDirty = false;
    state.recording.nameDirty = false;
    state.recording.dirtyProfiles = {};
    rememberRecording(null);
    await loadRecordingProjects();
    renderRecordingAll();
  } catch (error) {
    showAlert($('rec-dialogue-error'), error.message);
  }
}

async function renameRecordingProject() {
  const id = recordingId();
  if (!id) return;
  const name = $('rec-name').value.trim();
  if (!name || name === state.recording.project.name) {
    state.recording.nameDirty = false;
    renderRecordingDialogue();
    return;
  }
  try {
    state.recording.nameDirty = false;
    applyRecordingProject(await api(`${RECORDING_BASE}/${id}`, recordingJson('PATCH', { name })));
    await loadRecordingProjects();
  } catch (error) {
    // Имя не сохранилось — возвращаем в поле настоящее, а не оставляем
    // «грязным»: иначе следующая перерисовка снова показала бы несохранённое.
    state.recording.nameDirty = false;
    showAlert($('rec-dialogue-error'), error.message);
    renderRecordingDialogue();
  }
}

// --- панель 2: роли и голоса --------------------------------------------------
function renderRecordingRoles() {
  const box = $('rec-roles');
  const project = state.recording.project;
  if (!project || !(project.roles || []).length) {
    box.innerHTML = '<span class="muted">Разберите диалог — появятся роли.</span>';
    return;
  }
  box.innerHTML = (project.roles || []).map((role) => recordingRoleCard(role, project)).join('');
}

function recordingRoleCard(role, project) {
  const profiles = project.voice_profiles || [];
  const assigned = role.voice_profile_id || '';
  const profile = profiles.find((item) => item.id === assigned) || null;
  const pending = state.recording.dirtyProfiles[assigned];
  const values = pending || {
    speed: profile ? profile.speed : 1,
    pitch: profile ? profile.pitch_semitones : 0,
    denoise: profile ? profile.denoise : false,
  };
  const options = [
    '<option value="">— не выбран —</option>',
    ...profiles.map((item) => `<option value="${esc(item.id)}"${
      item.id === assigned ? ' selected' : ''}>${esc(item.name)}</option>`),
  ].join('');
  // Без назначенного голоса ручки показывать нечего: они принадлежат профилю,
  // а не роли. Дубли при этом не пропадают — backend обработает их значениями
  // по умолчанию, поэтому честно об этом говорим.
  const controls = profile ? `
    <label class="field">
      <span class="slider-head">Скорость <b data-role="speed-label">${values.speed.toFixed(2)}×</b></span>
      <input type="range" data-role="speed" min="${RECORDING_SPEED.min}" max="${RECORDING_SPEED.max}"
        step="${RECORDING_SPEED.step}" value="${values.speed}" />
    </label>
    <label class="field">
      <span class="slider-head">Высота голоса <b data-role="pitch-label">${fmtRecPitch(values.pitch)}</b></span>
      <input type="range" data-role="pitch" min="${RECORDING_PITCH.min}" max="${RECORDING_PITCH.max}"
        step="${RECORDING_PITCH.step}" value="${values.pitch}" />
    </label>
    <label class="toggle">
      <input type="checkbox" data-role="denoise"${values.denoise ? ' checked' : ''} />
      Очистка записи
    </label>
    <p class="hint muted">
      Уменьшает постоянный фоновый шум и комнатный гул. Речь других людей и громкие
      отдельные звуки может не удалить.
    </p>
    <div class="row" style="margin-top: 8px">
      <button class="tiny" data-role="profile-save"${pending ? '' : ' disabled'}>Сохранить настройки</button>
      <button class="tiny ghost" data-role="profile-preview">Прослушать обработку</button>
      <span class="profile-hint"${pending ? '' : ' hidden'}>есть несохранённые изменения</span>
    </div>` : `
    <p class="muted">
      Голос не выбран: дубли обработаются значениями по умолчанию (скорость 1.00×,
      высота 0 пт, без очистки).
    </p>`;
  return `
    <div class="card" data-speaker="${esc(role.speaker)}" data-profile="${esc(assigned)}">
      <div class="voice-top">
        <span class="avatar violet">${esc(initials(role.speaker))}</span>
        <span class="voice-name">
          <h3>${esc(role.speaker)}</h3>
          <p>${role.total} ${recPlural(role.total, ['реплика', 'реплики', 'реплик'])}
            · Записано: ${role.recorded} / ${role.total}</p>
        </span>
      </div>
      <label class="field">
        <span>Записываемый голос</span>
        <select data-role="role-voice">${options}</select>
      </label>
      <div class="row" style="margin-bottom: 10px">
        <button class="tiny" data-role="role-new-voice">+ Новый голос</button>
      </div>
      ${controls}
    </div>`;
}

function markRecordingProfileDirty(card) {
  const profileId = card.dataset.profile;
  if (!profileId) return null;
  const speed = parseFloat(card.querySelector('[data-role="speed"]').value);
  const pitch = parseFloat(card.querySelector('[data-role="pitch"]').value);
  const denoise = card.querySelector('[data-role="denoise"]').checked;
  state.recording.dirtyProfiles[profileId] = { speed, pitch, denoise };
  card.querySelector('[data-role="speed-label"]').textContent = `${speed.toFixed(2)}×`;
  card.querySelector('[data-role="pitch-label"]').textContent = fmtRecPitch(pitch);
  card.querySelector('[data-role="profile-save"]').disabled = false;
  card.querySelector('.profile-hint').hidden = false;
  return state.recording.dirtyProfiles[profileId];
}

async function assignRecordingRole(speaker, profileId) {
  const project = await api(
    `${RECORDING_BASE}/${recordingId()}/roles/${encodeURIComponent(speaker)}`,
    recordingJson('PUT', { profile_id: profileId || null }),
  );
  applyRecordingProject(project);
}

async function createRecordingProfile(speaker) {
  const id = recordingId();
  if (!id) return;
  const answer = window.prompt('Имя записываемого голоса', speaker);
  if (answer === null) return;
  showAlert($('rec-roles-error'), '');
  const known = new Set(((state.recording.project.voice_profiles) || []).map((item) => item.id));
  try {
    let project = await api(`${RECORDING_BASE}/${id}/voice-profiles`, recordingJson('POST', {
      name: answer.trim() || speaker,
      speed: 1,
      pitch_semitones: 0,
      denoise: false,
    }));
    // Профиль создаётся и сразу назначается роли: без назначения он повис бы
    // отдельной строкой, а роль осталась бы без голоса.
    const created = (project.voice_profiles || []).find((item) => !known.has(item.id));
    if (created) {
      project = await api(
        `${RECORDING_BASE}/${id}/roles/${encodeURIComponent(speaker)}`,
        recordingJson('PUT', { profile_id: created.id }),
      );
    }
    applyRecordingProject(project);
  } catch (error) {
    showAlert($('rec-roles-error'), error.message);
  }
}

async function saveRecordingProfile(card) {
  const profileId = card.dataset.profile;
  const values = state.recording.dirtyProfiles[profileId];
  if (!profileId || !values) return;
  showAlert($('rec-roles-error'), '');
  const project = await api(
    `${RECORDING_BASE}/${recordingId()}/voice-profiles/${profileId}`,
    recordingJson('PATCH', {
      speed: values.speed,
      pitch_semitones: values.pitch,
      denoise: values.denoise,
    }),
  );
  delete state.recording.dirtyProfiles[profileId];
  applyRecordingProject(project);
}

// Какой дубль слушать в превью: активный дубль открытой реплики, если он
// принадлежит этой роли, иначе первый записанный дубль роли. Превью всегда
// строится от сырого файла, поэтому «какой именно дубль» — это только выбор
// примера звучания, а не правка проекта.
function recordingPreviewTakeId(speaker) {
  const project = state.recording.project;
  if (!project) return null;
  const replica = recordingReplicaAt(state.recording.activeIndex);
  const take = recordingActiveTake(state.recording.activeIndex);
  if (take && replica && replica.speaker === speaker) return take.id;
  const role = (project.roles || []).find((item) => item.speaker === speaker);
  if (!role) return null;
  const indexes = [...(role.replicas || [])].sort((a, b) => a - b);
  for (const index of indexes) {
    const found = recordingActiveTake(index);
    if (found) return found.id;
  }
  return null;
}

async function previewRecordingProfile(card) {
  const id = recordingId();
  const profileId = card.dataset.profile;
  const speaker = card.dataset.speaker;
  if (!id || !profileId) return;
  const takeId = recordingPreviewTakeId(speaker);
  if (!takeId) {
    showAlert(
      $('rec-roles-error'),
      `У роли «${speaker}» ещё нет дублей: обработку можно послушать только на записанной реплике.`,
      'info',
    );
    return;
  }
  // Превью показывает в том числе несохранённые значения: пользователь слушает
  // то, что собирается сохранить, а не прошлое состояние профиля.
  const values = state.recording.dirtyProfiles[profileId] || recordingProfileValues(profileId);
  showAlert($('rec-roles-error'), '');
  try {
    const data = await api(
      `${RECORDING_BASE}/${id}/takes/${takeId}/preview`,
      recordingJson('POST', {
        speed: values.speed,
        pitch_semitones: values.pitch,
        denoise: values.denoise,
      }),
    );
    const player = $('rec-preview-player');
    player.src = data.url;
    player.hidden = false;
    player.play().catch(() => { /* автозапуск может быть запрещён политикой браузера */ });
    const warnings = data.warnings || [];
    showAlert(
      $('rec-roles-error'),
      warnings.length ? `Превью собрано, но: ${warnings.join('; ')}` : '',
      'info',
    );
  } catch (error) {
    showAlert($('rec-roles-error'), error.message);
  }
}

// --- панель 3: запись ---------------------------------------------------------
function renderRecordingOrderSwitch() {
  const project = state.recording.project;
  const order = (project && project.recording_order) || 'dialogue';
  document.querySelectorAll('#rec-order button').forEach((button) => {
    button.classList.toggle('active', button.dataset.order === order);
  });
}

function renderRecordingReplica() {
  const project = state.recording.project;
  const order = recordingNavOrder();
  const index = state.recording.activeIndex;
  const replica = recordingReplicaAt(index);
  const position = order.indexOf(index);
  $('rec-replica-label').textContent = replica
    ? `${replica.speaker} · Реплика ${position >= 0 ? position + 1 : '?'} из ${order.length}`
    : '—';
  $('rec-replica-text').textContent = replica ? replica.text : 'Диалог не разобран.';
  $('rec-replica-progress').textContent = project && project.readiness
    ? `Записано ${project.readiness.replicas_recorded} / ${project.readiness.replicas_total}`
    : '';
  const empty = !order.length || !replica;
  const busy = state.recording.busy;
  $('rec-btn-prev').disabled = empty || busy || position <= 0;
  $('rec-btn-next').disabled = empty || busy || position < 0 || position >= order.length - 1;
  $('rec-btn-next-missing').disabled = empty || busy || recordingNextMissingIndex(index) === null;
  $('rec-btn-record').disabled = empty;
  renderRecordingOrderSwitch();
  // Пустой индикатор уровня: без кадра канвас остаётся прозрачным, и непонятно,
  // что это вообще индикатор.
  if (!state.recording.analyser) drawMeterBar($('rec-level'), 0);
}

function renderRecordingTakeActions() {
  const record = state.recording;
  const hasBlob = Boolean(record.blob) && !record.busy;
  $('rec-btn-listen').hidden = !hasBlob;
  $('rec-btn-rerecord').hidden = !hasBlob;
  $('rec-btn-save-take').hidden = !hasBlob;
}

function setRecordingStatus(message) {
  $('rec-record-status').textContent = message;
}

function setRecordingBusy(busy) {
  state.recording.busy = busy;
  $('rec-btn-record').textContent = busy
    ? (state.recording.countdown ? '■ Отменить' : '■ Остановить')
    : '● Записать';
  $('rec-btn-record').classList.toggle('danger', busy);
  renderRecordingTakeActions();
  renderRecordingReplica();
}

function setRecordingReplica(index) {
  const order = recordingNavOrder();
  if (!order.includes(index)) return;
  // Смена реплики уносит несохранённый дубль с собой, поэтому спрашиваем —
  // но только когда он реально есть.
  if (state.recording.blob
    && !window.confirm('Несохранённый дубль будет потерян. Перейти к другой реплике?')) {
    return;
  }
  discardRecordingTake();
  state.recording.activeIndex = index;
  setRecordingStatus('—');
  $('rec-timer').textContent = fmtRecTime(0);
  renderRecordingReplica();
  renderRecordingTakes();
  renderRecordingRoles();
}

function stepRecordingReplica(delta) {
  const order = recordingNavOrder();
  const position = order.indexOf(state.recording.activeIndex);
  const next = position + delta;
  if (position < 0 || next < 0 || next >= order.length) return;
  setRecordingReplica(order[next]);
}

async function saveRecordingOrder(order) {
  const id = recordingId();
  if (!id) return;
  applyRecordingProject(await api(`${RECORDING_BASE}/${id}`, recordingJson('PATCH', {
    recording_order: order,
  })));
}

// Длительность реплики для списка дублей и монтажа. Считается по активному
// дублю, а не по всей записи: монтаж берёт ровно его.
function renderRecordingTakes() {
  const box = $('rec-takes');
  const project = state.recording.project;
  if (!project) {
    box.innerHTML = '<span class="muted">—</span>';
    showAlert($('rec-take-warning'), '');
    return;
  }
  const index = state.recording.activeIndex;
  const takes = recordingTakesFor(index);
  const active = recordingActiveTake(index);
  // Предупреждение из level_warning — это подсказка, а не запрет: дубль уже
  // сохранён и его можно выбрать.
  showAlert($('rec-take-warning'), active && active.level_warning ? active.level_warning : '');
  if (!takes.length) {
    box.innerHTML = '<span class="muted">Дублей пока нет — запишите реплику.</span>';
    return;
  }
  box.innerHTML = takes.map((take) => `
    <div class="take-row${take.active ? ' active' : ''}">
      <span class="take-id">${esc(String(take.id).slice(0, 6))}</span>
      <span class="take-dur">${fmtRecDuration(take.duration_sec)}</span>
      <span class="take-note">${esc(take.active ? 'активный' : (take.level_warning || ''))}</span>
      <button class="tiny ghost" data-role="take-play" data-take="${esc(take.id)}">▶</button>
      <button class="tiny ghost" data-role="take-select" data-take="${esc(take.id)}"${
        take.active ? ' disabled' : ''}>Сделать активным</button>
      <button class="tiny ghost danger" data-role="take-delete" data-take="${esc(take.id)}">удалить</button>
    </div>`).join('');
}

// --- панель 4: монтаж ---------------------------------------------------------
const recordingReplicaName = (index) => {
  const replica = recordingReplicaAt(index);
  return `№${index + 1}${replica && replica.speaker ? ` (${replica.speaker})` : ''}`;
};

function renderRecordingMontage() {
  const box = $('rec-montage');
  const project = state.recording.project;
  if (!project || !(project.replicas || []).length) {
    box.innerHTML = '<span class="muted">Разберите диалог — появится список реплик.</span>';
    updateRecordingRenderButton();
    return;
  }
  box.innerHTML = [...project.replicas].sort((a, b) => a.index - b.index).map((replica, position) => {
    const take = recordingActiveTake(replica.index);
    const mark = take
      ? '<span class="montage-mark ok">✓</span>'
      : '<span class="montage-mark miss">!</span>';
    return `
      <div class="montage-row" data-index="${replica.index}" title="Открыть реплику в панели записи">
        ${mark}
        <span class="take-id">${String(position + 1).padStart(2, '0')}</span>
        <span class="montage-name">${esc(replica.speaker)}</span>
        <span class="montage-text">${esc(replica.text)}</span>
        <span class="montage-dur">${esc(take ? fmtRecDuration(take.duration_sec) : 'не записано')}</span>
      </div>`;
  }).join('');
  updateRecordingRenderButton();
}

// Кнопка сборки — единственное место, где видно блокирующее условие монтажа,
// поэтому она пересчитывается и от готовности, и от статуса уже идущего рендера.
function updateRecordingRenderButton() {
  const project = state.recording.project;
  const readiness = project && project.readiness;
  const render = state.recording.render || (project && project.render) || null;
  const running = Boolean(render && render.status === 'running');
  const ready = Boolean(readiness && readiness.ready_to_render);
  $('rec-btn-render').disabled = !ready || running;
  const missing = readiness && readiness.missing ? readiness.missing.length : 0;
  $('rec-missing-note').textContent = project
    ? (missing ? `Не записано: ${missing}` : 'Все реплики записаны')
    : '';
  $('rec-btn-goto-missing').hidden = !missing || running;
}

function renderRecordingSettings() {
  const project = state.recording.project;
  const settings = (project && project.render_settings) || {};
  const pause = Number(settings.pause_ms);
  const value = Number.isFinite(pause) ? pause : RECORDING_DEFAULT_PAUSE;
  $('rec-pause').value = String(value);
  $('rec-pause-value').textContent = `${value} мс`;
}

async function saveRecordingPause(value) {
  const id = recordingId();
  if (!id) return;
  showAlert($('rec-render-error'), '');
  try {
    applyRecordingProject(await api(`${RECORDING_BASE}/${id}`, recordingJson('PATCH', {
      render_settings: { pause_ms: value },
    })));
  } catch (error) {
    showAlert($('rec-render-error'), error.message);
  }
}

async function startRecordingRender() {
  const id = recordingId();
  if (!id) return;
  showAlert($('rec-render-error'), '');
  const pause = parseInt($('rec-pause').value, 10);
  try {
    await api(`${RECORDING_BASE}/${id}/render`, recordingJson('POST', { pause_ms: pause }));
    startRecordingRenderPoll();
  } catch (error) {
    // 409 приходит структурой {error, missing, message}: список незаписанных
    // реплик показываем именами, а не JSON-строкой.
    const detail = error.detail;
    if (detail && typeof detail === 'object' && Array.isArray(detail.missing)) {
      const names = detail.missing.map((index) => recordingReplicaName(index)).join(', ');
      showAlert(
        $('rec-render-error'),
        `${detail.message || detail.error || 'Монтаж невозможен'}. Не записано: ${names}.`,
      );
    } else {
      showAlert($('rec-render-error'), error.message);
    }
  }
}

function startRecordingRenderPoll() {
  stopRecordingRenderPoll();
  applyRecordingRenderState({ status: 'running', progress: 0, message: 'монтаж запущен' });
  state.recording.renderTimer = setInterval(pollRecordingRender, RECORDING_POLL_MS);
}

function stopRecordingRenderPoll() {
  if (state.recording.renderTimer) clearInterval(state.recording.renderTimer);
  state.recording.renderTimer = null;
}

async function pollRecordingRender() {
  const id = recordingId();
  if (!id) { stopRecordingRenderPoll(); return; }
  try {
    const render = await api(`${RECORDING_BASE}/${id}/render/status`);
    applyRecordingRenderState(render);
    if (render.status === 'done' || render.status === 'error') {
      stopRecordingRenderPoll();
      // Готовый файл и статус проекта живут на бэкенде: перечитываем payload
      // целиком, а не достраиваем состояние из ответа опроса.
      applyRecordingProject(await api(`${RECORDING_BASE}/${id}`));
    }
  } catch (error) {
    stopRecordingRenderPoll();
    showAlert($('rec-output-error'), error.message);
  }
}

function applyRecordingRenderState(render) {
  state.recording.render = render;
  renderRecordingOutput();
  updateRecordingRenderButton();
}

// --- панель 5: готовый звук ---------------------------------------------------
function renderRecordingOutput() {
  const project = state.recording.project;
  const id = recordingId();
  const render = state.recording.render || (project && project.render) || null;
  const status = render ? render.status : 'idle';
  const progress = render && Number.isFinite(render.progress) ? render.progress : 0;
  $('rec-render-bar').style.width = `${Math.round(Math.min(Math.max(progress, 0), 1) * 100)}%`;
  const labels = {
    idle: 'Диалог ещё не собран',
    running: 'идёт монтаж',
    done: 'готово',
    error: 'монтаж не удался',
  };
  const message = render && render.message ? render.message : '';
  $('rec-render-status').textContent = status === 'error'
    ? `монтаж не удался: ${(render && render.error) || message || 'неизвестная причина'}`
    : `${labels[status] || status}${message && status !== 'idle' ? ` · ${message}` : ''}`;

  const hasOutput = Boolean(project && project.has_output)
    || Boolean(status === 'done' && render && render.url);
  const player = $('rec-player');
  const link = $('rec-download');
  player.hidden = !hasOutput;
  link.hidden = !hasOutput;
  if (hasOutput && id) {
    const src = `${RECORDING_BASE}/${id}/audio`;
    // Пересборка даёт новый файл по тому же адресу: без метки версии браузер
    // проигрывал бы закешированную прошлую сборку.
    const stamp = (render && (render.finished_at || render.duration_sec)) || '';
    const versioned = `${src}?v=${encodeURIComponent(String(stamp))}`;
    if (player.getAttribute('src') !== versioned) {
      player.src = versioned;
      player.load();
    }
    link.href = `${src}?download=true`;
  }
  const duration = render && render.duration_sec ? render.duration_sec : 0;
  $('rec-output-meta').textContent = hasOutput && duration
    ? `длительность ${fmtRecDuration(duration)}`
    : '';
  const warnings = (render && render.warnings) || [];
  showAlert($('rec-output-warnings'), warnings.length ? warnings.join('\n') : '', 'info');
  showAlert(
    $('rec-output-error'),
    status === 'error' ? ((render && render.error) || 'монтаж не удался') : '',
  );
}

// --- микрофон -----------------------------------------------------------------
async function refreshRecordingDevices() {
  const select = $('rec-device');
  const field = $('rec-device-field');
  if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
    field.hidden = true;
    return;
  }
  let devices = [];
  try {
    devices = await navigator.mediaDevices.enumerateDevices();
  } catch (_) {
    field.hidden = true;
    return;
  }
  const inputs = devices.filter((device) => device.kind === 'audioinput');
  if (inputs.length <= 1) {
    // Один микрофон — выбирать нечего; но запоминаем его id, чтобы запись шла
    // именно с него, а не с «устройства по умолчанию», которое могли сменить
    // в системе между записями.
    state.recording.deviceId = inputs.length ? inputs[0].deviceId : '';
    field.hidden = true;
    return;
  }
  field.hidden = false;
  const current = state.recording.deviceId;
  select.innerHTML = inputs.map((device, position) =>
    `<option value="${esc(device.deviceId)}">${
      esc(device.label || `Микрофон ${position + 1}`)}</option>`).join('');
  select.value = inputs.some((device) => device.deviceId === current) ? current : inputs[0].deviceId;
  state.recording.deviceId = select.value;
}

function recordingConstraints() {
  const deviceId = state.recording.deviceId;
  return deviceId ? { audio: { deviceId: { exact: deviceId } } } : { audio: true };
}

// Общие тексты `recordErrorMessage` написаны для вкладки «Голоса», где рядом есть
// загрузка файла. Здесь файл загрузить нельзя, поэтому про отсутствующий
// микрофон говорим иначе — но остальные случаи берём из общего помощника, чтобы
// формулировки не разъезжались.
function recordingMicrophoneError(error) {
  if (error && (error.name === 'NotFoundError' || error.name === 'OverconstrainedError')) {
    return 'Микрофон не найден. Подключите его и нажмите «Записать» снова — список устройств перечитается.';
  }
  return recordErrorMessage(error);
}

function drawRecordingMeter() {
  const record = state.recording;
  const canvas = $('rec-level');
  if (!record.analyser) return;
  const data = new Uint8Array(record.analyser.fftSize);
  record.analyser.getByteTimeDomainData(data);
  let peak = 0;
  for (const sample of data) peak = Math.max(peak, Math.abs(sample - 128) / 128);
  // Пик держим с затуханием: мгновенное значение мелькает быстрее, чем глаз
  // успевает заметить перегрузку.
  record.peak = Math.max(peak, record.peak * 0.9);
  drawMeterBar(canvas, record.peak);
  record.raf = requestAnimationFrame(drawRecordingMeter);
}

// Закрытие захвата без сохранения: уход с вкладки или отмена отсчёта. Дорожку и
// аудиоконтекст закрываем всегда — иначе браузер продолжает считать, что идёт
// захват микрофона, и не гасит свой индикатор.
function closeRecordingStream() {
  const record = state.recording;
  if (record.stream) record.stream.getTracks().forEach((track) => track.stop());
  record.stream = null;
  if (record.raf) cancelAnimationFrame(record.raf);
  record.raf = null;
  if (record.ctx) record.ctx.close();
  record.ctx = null;
  record.analyser = null;
  record.peak = 0;
  drawMeterBar($('rec-level'), 0);
}

function stopRecordingCapture() {
  const record = state.recording;
  if (record.countdown) {
    clearInterval(record.countdown.timer);
    record.countdown = null;
  }
  if (record.tick) {
    clearInterval(record.tick);
    record.tick = null;
  }
  if (record.recorder && record.recorder.state !== 'inactive') {
    // Обработчик `stop` снимет поток и покажет дубль; здесь только команда.
    record.recorder.stop();
    return;
  }
  record.recorder = null;
  if (record.busy) {
    setRecordingBusy(false);
    setRecordingStatus('запись прервана');
  }
  closeRecordingStream();
}

async function startRecordingTake() {
  if (!recordingReplicaAt(state.recording.activeIndex)) return;
  showAlert($('rec-record-error'), '');
  showAlert($('rec-take-warning'), '');
  const format = recordFormat();
  if (!format) {
    showAlert($('rec-record-error'), 'Браузер не умеет записывать звук. Загрузите готовый файл во вкладке «Голоса».');
    return;
  }
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    // getUserMedia работает только в защищённом контексте: localhost тоже подходит.
    showAlert($('rec-record-error'), 'Запись с микрофона доступна только на localhost или по https.');
    return;
  }
  // Прошлый несохранённый дубль затирается новой записью — это и есть «Перезаписать».
  discardRecordingTake();
  const record = state.recording;
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia(recordingConstraints());
  } catch (error) {
    showAlert($('rec-record-error'), recordingMicrophoneError(error));
    return;
  }
  record.stream = stream;
  record.chunks = [];
  record.suffix = format[1];

  // Индикатор уровня вспомогательный: если аудиоконтекст не поднялся (нет
  // устройства вывода, политика браузера), запись всё равно должна состояться,
  // а не падать из-за индикатора.
  record.ctx = null;
  record.analyser = null;
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 2048;
    ctx.createMediaStreamSource(stream).connect(analyser);
    record.ctx = ctx;
    record.analyser = analyser;
    drawRecordingMeter();
  } catch (_) {
    drawMeterBar($('rec-level'), 0);
  }

  // Подписи устройств появляются только после выдачи разрешения: до этого
  // enumerateDevices отдаёт пустые label, и выбрать «правильный» микрофон нельзя.
  refreshRecordingDevices().catch(() => {});

  const seconds = parseInt($('rec-countdown').value || '1', 10);
  record.startedAt = Date.now();
  // Отсчёт заводится до setRecordingBusy: подпись кнопки зависит от того, идёт
  // отсчёт или уже запись, и «Отменить» должно появиться сразу, а не после
  // первого тика таймера.
  if (seconds > 0) record.countdown = { left: seconds, timer: null };
  setRecordingBusy(true);
  $('rec-timer').textContent = fmtRecTime(0);
  if (seconds > 0) {
    setRecordingStatus(`отсчёт: ${seconds}`);
    record.countdown.timer = setInterval(() => {
      if (!record.countdown) return;
      record.countdown.left -= 1;
      if (record.countdown.left > 0) {
        setRecordingStatus(`отсчёт: ${record.countdown.left}`);
        return;
      }
      clearInterval(record.countdown.timer);
      record.countdown = null;
      startRecorderEngine(format[0]);
      setRecordingBusy(true);
    }, 1000);
  } else {
    startRecorderEngine(format[0]);
  }
}

function startRecorderEngine(mimeType) {
  const record = state.recording;
  if (!record.stream) return;
  const recorder = new MediaRecorder(record.stream, { mimeType });
  record.recorder = recorder;
  record.startedAt = Date.now();
  recorder.addEventListener('dataavailable', (event) => {
    if (event.data.size) record.chunks.push(event.data);
  });
  recorder.addEventListener('stop', finishRecordingTake);
  recorder.start();
  setRecordingStatus('идёт запись');
  record.tick = setInterval(() => {
    $('rec-timer').textContent = fmtRecTime((Date.now() - record.startedAt) / 1000);
  }, 100);
}

function stopRecordingTake() {
  const record = state.recording;
  if (record.countdown) {
    // Отсчёт прерывается до старта: захват закрываем, дубль не создаём.
    clearInterval(record.countdown.timer);
    record.countdown = null;
    closeRecordingStream();
    record.recorder = null;
    setRecordingBusy(false);
    setRecordingStatus('запись отменена');
    $('rec-timer').textContent = fmtRecTime(0);
    return;
  }
  if (!record.recorder) return;
  if (record.recorder.state !== 'inactive') record.recorder.stop();
}

function finishRecordingTake() {
  const record = state.recording;
  if (record.tick) clearInterval(record.tick);
  record.tick = null;
  record.recorder = null;
  closeRecordingStream();
  const elapsed = (Date.now() - record.startedAt) / 1000;
  $('rec-timer').textContent = fmtRecTime(elapsed);
  setRecordingBusy(false);
  const blob = new Blob(record.chunks, { type: record.chunks.length ? record.chunks[0].type : '' });
  record.chunks = [];
  if (!blob.size) {
    showAlert($('rec-record-error'), 'Запись получилась пустой — проверьте, что выбран верный микрофон.');
    setRecordingStatus('—');
    return;
  }
  record.blob = blob;
  if (record.url) URL.revokeObjectURL(record.url);
  record.url = URL.createObjectURL(blob);
  const player = $('rec-take-player');
  player.src = record.url;
  player.hidden = false;
  setRecordingStatus(`дубль готов: ${elapsed.toFixed(1)} с — сохраните или перезапишите`);
  renderRecordingTakeActions();
}

function discardRecordingTake() {
  const record = state.recording;
  if (record.url) URL.revokeObjectURL(record.url);
  record.url = null;
  record.blob = null;
  const player = $('rec-take-player');
  if (!player) return;
  player.pause();
  player.hidden = true;
  player.removeAttribute('src');
  renderRecordingTakeActions();
}

function listenRecordingTake() {
  const player = $('rec-take-player');
  if (!state.recording.url) return;
  if (player.paused) player.play().catch(() => { /* политика автозапуска */ });
  else player.pause();
}

async function uploadRecordingTake() {
  const record = state.recording;
  const id = recordingId();
  if (!record.blob || !id) return;
  const index = state.recording.activeIndex;
  const suffix = record.suffix || 'webm';
  const file = new File([record.blob], `take.${suffix}`, {
    type: record.blob.type || 'audio/webm',
  });
  const form = new FormData();
  form.append('file', file);
  $('rec-btn-save-take').disabled = true;
  setRecordingStatus('сохраняю дубль…');
  try {
    const data = await api(`${RECORDING_BASE}/${id}/replicas/${index}/takes`, {
      method: 'POST',
      body: form,
    });
    // Дубль сохранён — локальная запись больше не нужна: дальше всё берётся
    // из payload, где новый дубль уже активный.
    discardRecordingTake();
    applyRecordingProject(data.project);
    setRecordingStatus('дубль сохранён');
    if ($('rec-autonext').checked) {
      const next = recordingNextMissingIndex(index);
      if (next !== null) setRecordingReplica(next);
    }
    // Замечание к только что записанному дублю показываем после перехода: иначе
    // автопереход сразу затёр бы его состоянием следующей реплики, и пользователь
    // записал бы весь диалог, не увидев предупреждения ни разу.
    const warning = data.take && data.take.level_warning;
    if (warning) {
      showAlert($('rec-take-warning'), `Дубль сохранён, но: ${warning}`);
      setRecordingStatus('дубль сохранён — есть замечание к записи');
    }
  } catch (error) {
    showAlert($('rec-record-error'), error.message);
    setRecordingStatus('дубль не сохранён');
  } finally {
    $('rec-btn-save-take').disabled = false;
  }
}

function playRecordingTake(takeId) {
  const project = state.recording.project;
  const take = ((project && project.takes) || []).find((item) => item.id === takeId);
  if (!take) return;
  const player = $('rec-takes-player');
  player.src = take.audio_url;
  player.play().catch(() => { /* политика автозапуска */ });
}

const recordingTakeNotFound = 'Дубль не найден — возможно, его уже удалили. Обновите список.';

async function selectRecordingTake(index, takeId) {
  showAlert($('rec-record-error'), '');
  try {
    const project = await api(
      `${RECORDING_BASE}/${recordingId()}/replicas/${index}/takes/${takeId}/select`,
      { method: 'POST' },
    );
    applyRecordingProject(project);
  } catch (error) {
    showAlert($('rec-record-error'), error.status === 404 ? recordingTakeNotFound : error.message);
  }
}

async function deleteRecordingTake(index, takeId) {
  showAlert($('rec-record-error'), '');
  try {
    const project = await api(
      `${RECORDING_BASE}/${recordingId()}/replicas/${index}/takes/${takeId}`,
      { method: 'DELETE' },
    );
    applyRecordingProject(project);
  } catch (error) {
    showAlert($('rec-record-error'), error.status === 404 ? recordingTakeNotFound : error.message);
  }
}

// --- события вкладки записи ---------------------------------------------------
function bindRecordingEvents() {
  // Панель 1
  $('rec-dialogue').addEventListener('input', () => {
    state.recording.dialogueDirty = true;
    renderRecordingDialogue();
  });
  $('rec-btn-apply').addEventListener('click', applyRecordingDialogue);
  $('rec-btn-reparse').addEventListener('click', reparseRecordingProject);
  $('rec-btn-delete').addEventListener('click', deleteRecordingProject);
  $('rec-btn-reload-projects').addEventListener('click', () => {
    loadRecordingProjects().catch((error) => showAlert($('rec-dialogue-error'), error.message));
  });
  $('rec-project-select').addEventListener('change', (event) => {
    openRecordingProject(event.target.value)
      .catch((error) => showAlert($('rec-dialogue-error'), error.message));
  });
  $('rec-name').addEventListener('input', () => {
    state.recording.nameDirty = true;
  });
  // Имя сохраняется по уходу из поля: на каждый символ это был бы PATCH с полной
  // перерисовкой вкладки.
  $('rec-name').addEventListener('change', renameRecordingProject);

  // Панель 2. Ползунки только помечают профиль «грязным»: PATCH идёт по кнопке,
  // иначе каждое движение ползунка пересобирало бы карточки и сбивало захват.
  const roles = $('rec-roles');
  roles.addEventListener('input', (event) => {
    const card = event.target.closest('.card');
    if (!card) return;
    const role = event.target.dataset.role;
    if (role === 'speed' || role === 'pitch') markRecordingProfileDirty(card);
  });
  roles.addEventListener('change', (event) => {
    const card = event.target.closest('.card');
    if (!card) return;
    if (event.target.dataset.role === 'denoise') markRecordingProfileDirty(card);
    if (event.target.dataset.role === 'role-voice') {
      assignRecordingRole(card.dataset.speaker, event.target.value).catch((error) => {
        showAlert($('rec-roles-error'), error.message);
        // Назначение не сохранилось — возвращаем select к настоящему состоянию.
        applyRecordingProject(state.recording.project);
      });
    }
  });
  roles.addEventListener('click', (event) => {
    const card = event.target.closest('.card');
    const button = event.target.closest('button');
    if (!card || !button) return;
    if (button.dataset.role === 'role-new-voice') createRecordingProfile(card.dataset.speaker);
    if (button.dataset.role === 'profile-save') {
      saveRecordingProfile(card).catch((error) => showAlert($('rec-roles-error'), error.message));
    }
    if (button.dataset.role === 'profile-preview') previewRecordingProfile(card);
  });

  // Панель 3
  $('rec-order').addEventListener('click', (event) => {
    const button = event.target.closest('button[data-order]');
    if (!button || !state.recording.project) return;
    if (button.dataset.order === state.recording.project.recording_order) return;
    saveRecordingOrder(button.dataset.order)
      .catch((error) => showAlert($('rec-dialogue-error'), error.message));
  });
  $('rec-btn-prev').addEventListener('click', () => stepRecordingReplica(-1));
  $('rec-btn-next').addEventListener('click', () => stepRecordingReplica(1));
  $('rec-btn-next-missing').addEventListener('click', () => {
    const next = recordingNextMissingIndex(state.recording.activeIndex);
    if (next !== null) setRecordingReplica(next);
  });
  // Одна кнопка на весь цикл: отсчёт → запись → остановка.
  $('rec-btn-record').addEventListener('click', () => {
    if (state.recording.busy) stopRecordingTake();
    else startRecordingTake();
  });
  $('rec-btn-listen').addEventListener('click', listenRecordingTake);
  $('rec-btn-rerecord').addEventListener('click', () => {
    discardRecordingTake();
    setRecordingStatus('—');
    startRecordingTake();
  });
  $('rec-btn-save-take').addEventListener('click', uploadRecordingTake);
  $('rec-device').addEventListener('change', (event) => {
    state.recording.deviceId = event.target.value;
    // Устройство выбирают до записи: если захват уже открыт (идёт отсчёт),
    // смена микрофона к нему не применится — честно говорим об этом.
    if (state.recording.busy) {
      setRecordingStatus('смена микрофона применится к следующей записи');
    }
  });
  $('rec-takes').addEventListener('click', (event) => {
    const button = event.target.closest('button[data-take]');
    if (!button) return;
    const takeId = button.dataset.take;
    const index = state.recording.activeIndex;
    if (button.dataset.role === 'take-play') playRecordingTake(takeId);
    if (button.dataset.role === 'take-select') selectRecordingTake(index, takeId);
    if (button.dataset.role === 'take-delete') {
      if (window.confirm('Удалить дубль? Сырая запись будет стёрта.')) {
        deleteRecordingTake(index, takeId);
      }
    }
  });

  // Панель 4
  $('rec-montage').addEventListener('click', (event) => {
    const row = event.target.closest('.montage-row');
    if (row) setRecordingReplica(parseInt(row.dataset.index, 10));
  });
  $('rec-btn-goto-missing').addEventListener('click', () => {
    const readiness = state.recording.project && state.recording.project.readiness;
    const missing = (readiness && readiness.missing) || [];
    if (missing.length) setRecordingReplica(missing[0]);
  });
  $('rec-pause').addEventListener('input', (event) => {
    $('rec-pause-value').textContent = `${event.target.value} мс`;
  });
  // Пауза сохраняется на отпускании ползунка: на каждом шаге это был бы десяток
  // PATCH-запросов с полной перерисовкой панели.
  $('rec-pause').addEventListener('change', (event) => {
    saveRecordingPause(parseInt(event.target.value, 10));
  });
  $('rec-btn-render').addEventListener('click', startRecordingRender);
}

// --- старт --------------------------------------------------------------------
async function init() {
  // Умолчания настроек запоминаются до первого запроса к API: загрузка голосов
  // подставляет голос в форму сплошного текста, и снятое позже «умолчание» было бы
  // уже не умолчанием, а случайным первым голосом.
  captureResetDefaults();
  renderRecordPhrases();
  // Вкладка записи грузится лениво (switchTab), но её обработчики и пустое
  // состояние нужны сразу: иначе первый же клик по вкладке ждал бы разметки.
  bindRecordingEvents();
  renderRecordingAll();
  bindEvents();
  bindResetEvents();
  bindTimelineEvents();
  bindExportEvents();
  renderTimeline();
  // Таймер создаём до первого опроса: если всё уже готово, refreshStatus его снимет.
  state.statusTimer = setInterval(refreshStatus, 4000);
  await refreshStatus();
  // Каждый шаг — в своём try (F-W2): раньше один общий catch гасил все три сразу,
  // и падение загрузки движков оставляло пользователя без голосов и без проекта,
  // хотя ни то, ни другое от движков не зависит.
  try {
    // Паспорта движков — до голосов: и карточки, и форма рисуют по ним набор настроек.
    await loadEngines();
  } catch (error) {
    showAlert($('voice-error'), error.message);
  }
  try {
    await loadVoices();
  } catch (error) {
    showAlert($('voice-error'), error.message);
  }
  try {
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

// Общий фундамент, который берут вынесенные модули: вкладка «Голоса» (voices.js),
// таймлайн (timeline.js) и экспорт/импорт проекта (project-io.js). Списки совпадают
// с их `import`: модули связаны двусторонне, но значения читаются только внутри
// функций — на этапе оценки модуля цикл безопасен.
export {
  state,
  $,
  esc,
  api,
  showAlert,
  voiceById,
  PARAM_DEFAULTS,
  PREVIEW_TEXT,
  BENCHMARK_TEXT,
  DROP_HINT,
  ENGINE_F5,
  EMOTION_LABELS,
  usesReference,
  cloningNote,
  engineNote,
  engineLabel,
  engineOptionsHtml,
  nfeOptions,
  engineParamsHtml,
  initials,
  suggestedEngine,
  engineInfo,
  loadVoices,
  watchEngines,
  resetRecording,
  applyTimeline,
  rememberProject,
  applyProject,
  exportDiagnostics,
};

init();
