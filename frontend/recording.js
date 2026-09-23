// Вкладка «Сам себе звукорежиссер»: запись реплик диалога с микрофона.
//
// Модуль вынесен из монолита app.js без изменения поведения (F-W3).
// Источник истины — backend: каждый project-эндпоинт возвращает полный payload и
// целиком заменяет `state.recording.project`. Локально живёт только то, что не
// является состоянием проекта: открытая реплика, выбранный микрофон,
// несохранённые ползунки и записанный, но ещё не отправленный дубль.
//
// Общее с мастером записи голоса (voice-record.js) — только выбор формата
// контейнера, индикатор уровня и тексты ошибок доступа к микрофону; они
// импортируются оттуда.
import {
  $,
  api,
  esc,
  initials,
  showAlert,
  state,
} from './app.js';
import { drawMeterBar, recordErrorMessage, recordFormat } from './voice-record.js';

// --- константы вкладки --------------------------------------------------------
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

export {
  bindRecordingEvents,
  loadRecordingTab,
  renderRecordingAll,
  stopRecordingCapture,
  stopRecordingRenderPoll,
};
