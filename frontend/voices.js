// Вкладка «Голоса»: карточки голосов, прослушивание, референс-профили эмоций,
// сравнение движков и форма создания голоса.
//
// Модуль вынесен из монолита app.js без изменения поведения (F-W3).
// Направление зависимостей здесь двустороннее: карточки голосов зовут
// перерисовку соседних вкладок (через `loadVoices` в app.js), а app.js зовёт
// обработчики карточек. У `loadVoices` намеренно оставлена роль «оркестратора»:
// она перерисовывает вкладки диалога, текста, записи и превью, то есть знает про
// весь интерфейс, а не только про голоса.
import {
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
} from './app.js';
// Сброс записи голоса живёт в модуле записи: карточка нового голоса очищает
// панель «Запись с микрофона» после сохранения, а та — его поле.
import { resetRecording } from './voice-record.js';

function voiceTags(voice) {
  const label = voice.gender === 'male' ? 'муж.' : voice.gender === 'female' ? 'жен.' : '—';
  const tags = [`<span class="tag">${label}</span>`];
  // Голос пресетного движка записи не имеет: он и есть встроенный голос модели.
  // Отдельная метка нужна, чтобы отсутствие референса читалось как норма, а не
  // как «файл потерян».
  if (!usesReference(voice.engine)) {
    tags.push(`<span class="tag ok" title="Встроенный голос модели: ${esc(engineLabel(voice.engine))} выбирает его по полу карточки, запись не нужна">встроенный</span>`);
  }
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

function updateNewVoiceEngineNote() {
  const info = engineInfo($('new-voice-engine').value);
  $('new-voice-engine-note').textContent = info
    ? `${info.description}${info.note ? ` ${info.note}` : ''}`
    : '';
  updateNewVoiceRefBlocks();
}

// Поля записи в форме нового голоса нужны не всякому движку: у пресетного
// (см. `usesReference`) референс не используется вовсе, и голос у него — это имя
// плюс пол. Скрыть их здесь важнее, чем на карточке: форма требовала файл даже
// для движка, который его игнорирует.
function updateNewVoiceRefBlocks() {
  const needsReference = usesReference($('new-voice-engine').value);
  [
    'new-voice-drop',        // выбор или перетаскивание записи
    'record-block',          // запись с микрофона (референс и профили эмоций)
    'new-voice-ref-text-block',  // расшифровка записи
    'new-voice-verify-block',    // сверка расшифровки с записью
    'new-voice-denoise-block',   // очистка записи
    'new-voice-ref-hint',        // пояснения про референс F5/XTTS
  ].forEach((id) => {
    const element = $(id);
    if (element) element.hidden = !needsReference;
  });
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
      + (voice.ref_text || !usesReference(voice.engine)
        ? ''
        : '<span class="tag warn" title="Без референс-текста синтез невозможен">нет текста</span>');
    const cloning = cloningNote(voice.engine);
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
          ${cloning ? `<span class="muted" data-role="cloning-note">${esc(cloning)}</span>` : ''}
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
        ${voiceReferencesHtml(voice)}
        ${usesReference(voice.engine)
          ? '<button class="tiny ghost compare-button" data-role="compare">⇄ Сравнить движки</button>'
          : ''}
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

// Референс-профили голоса (UPDATE 3 §16–§19). Один голос — один `voice_id`, но
// несколько референсов: нейтральный (загруженный при создании) и интонационные.
// Показываем интонацию, подпись и качество; добавление и удаление — здесь же,
// потому что «где записать восторг» и «где выбрать голос» — одно место.
//
// Список — ровно записываемые профили `PROFILE_KEYS` (§10). Семантических SURPRISE
// и FEAR здесь нет: под них нет фразы записи, и бэкенд отвергает такой профиль —
// подставить их в этот список значило бы предлагать выбор, который кончается
// ошибкой 400. Ручной выбор SURPRISE у реплики остаётся возможным и честно
// показывается откатом на нейтральный референс (§17).
const REFERENCE_EMOTIONS = [
  'NEUTRAL', 'CALM', 'QUESTION', 'NEUTRAL_QUESTION', 'EXCLAMATION', 'DELIGHT',
  'SAD_SYMPATHETIC', 'IRONIC', 'STRICT', 'ENUMERATION', 'EXCITED',
];

function voiceReferencesHtml(voice) {
  // Пресетному движку референсы не адресованы вовсе: он не зовёт резолвер
  // профилей, и любая загруженная запись не доехала бы до синтеза. Панель
  // предлагала бы действие без результата.
  if (!usesReference(voice.engine)) return '';
  const profiles = voice.reference_profiles || [];
  const rows = profiles.map((profile) => {
    const quality = profile.quality_status === 'warning'
      ? `<span class="tag warn" title="${esc(profile.quality_note || 'расшифровка не совпала с записью')}">проверить</span>`
      : '';
    const emotion = EMOTION_LABELS[profile.emotion] || profile.emotion;
    const isBase = profile.id === `${voice.id}-neutral`;
    const removable = isBase
      ? '<span class="muted">основной</span>'
      : `<button class="tiny ghost danger" data-role="reference-delete" data-profile="${esc(profile.id)}">удалить</button>`;
    // §35: подтверждение открывает профилю автоматический выбор по интонации.
    // У нейтрального референса флага нет: он и есть сам голос, автоматика берёт
    // его всегда — подтверждать там нечего. Пока флаг снят, профиль остаётся
    // рабочим при ручном выборе: сначала benchmark, потом доверие.
    const auto = profile.emotion === 'NEUTRAL'
      ? ''
      : `<label class="muted" title="Разрешить автоматический подбор этого профиля по интонации реплики. Ставьте после прослушивания; без флага профиль доступен только вручную.">
           <input type="checkbox" data-role="reference-auto" data-profile="${esc(profile.id)}" ${profile.enabled_for_auto ? 'checked' : ''} /> авто
         </label>`;
    return `
      <div class="row between" style="margin-top:6px">
        <span class="muted">${esc(emotion)} · ${esc(profile.label || '')} ${quality}</span>
        <span class="row">${auto}${removable}</span>
      </div>`;
  }).join('');
  const options = REFERENCE_EMOTIONS
    .map((value) => `<option value="${value}">${esc(EMOTION_LABELS[value] || value)}</option>`)
    .join('');
  return `
    <details class="advanced">
      <summary>Референсы эмоций (${profiles.length})</summary>
      ${rows || '<div class="muted" style="margin-top:6px">Отдельных эмоциональных записей пока нет.</div>'}
      <div class="row" style="margin-top:8px">
        <select data-role="reference-emotion">${options}</select>
        <button class="tiny" data-role="reference-add">Загрузить запись</button>
        <input type="file" accept="audio/*" hidden data-role="reference-file" />
      </div>
      <span class="muted">
        Своя запись для эмоции даёт модели нужную интонацию: вопрос, восторг,
        удивление, испуг. Без неё эмоция озвучивается нейтральным референсом того же
        голоса — синтез не блокируется, но интонация будет нейтральной. Референс
        другого голоса не используется никогда. Галочка «авто» разрешает брать
        профиль автоматически по интонации реплики — ставьте её после того, как
        послушали запись: без неё профиль выбирается только вручную.
      </span>
    </details>`;
}

async function addVoiceReference(voiceId, emotion, file) {
  if (!file) return;
  const form = new FormData();
  form.append('file', file);
  form.append('emotion', emotion);
  form.append('verify_ref_text', 'false');
  const result = await api(`/api/voices/${voiceId}/references`, { method: 'POST', body: form });
  return result;
}

async function deleteVoiceReference(voiceId, profileId) {
  return api(`/api/voices/${voiceId}/references/${profileId}`, { method: 'DELETE' });
}

async function updateVoiceReference(voiceId, profileId, changes) {
  return api(`/api/voices/${voiceId}/references/${profileId}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(changes),
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
  // Пресетному движку запись не нужна: голос у него — имя плюс пол, и файл в
  // запрос не уходит даже если его успели выбрать до смены движка.
  const needsReference = usesReference($('new-voice-engine').value);
  showAlert($('voice-error'), '');
  if (needsReference && !file) { status.textContent = 'выберите аудиофайл или запишите голос'; return; }

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
  if (needsReference && file) form.append('file', file);

  const cleaning = denoise ? 'чищу запись от шума, ' : '';
  if (!needsReference) status.textContent = 'сохраняю голос…';
  else if (!verify) status.textContent = `${cleaning}сохраняю голос…`;
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

export {
  voiceTags,
  updateNewVoiceEngineNote,
  previewState,
  presetText,
  saveVoicePreset,
  previewCard,
  renderVoiceCards,
  REFERENCE_EMOTIONS,
  voiceReferencesHtml,
  addVoiceReference,
  deleteVoiceReference,
  updateVoiceReference,
  setPreviewStatus,
  cancelJob,
  setCancelButton,
  previewVoice,
  cancelPreview,
  pollPreview,
  changeVoiceEngine,
  saveVoiceEngineParams,
  benchmarkState,
  stopBenchmarkPoll,
  benchmarkSeconds,
  benchmarkWer,
  benchmarkRowHtml,
  benchmarkResultsHtml,
  benchmarkPanelHtml,
  renderBenchmarkPanel,
  syncBenchmarkPanel,
  toggleBenchmark,
  runBenchmark,
  pollBenchmark,
  playBenchmarkTake,
  selectBenchmarkEngine,
  recognizeRefText,
  createVoice,
  resetDropZone,
};
