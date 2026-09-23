// Панель «Что услышит модель»: показ стадий подготовки текста к синтезу.
//
// Модуль вынесен из монолита app.js без изменения поведения (F-W3).
// Один компонент на две вкладки: панель — это разметка плюс функции, а не два
// набора кода. Проверку делает бэкенд ровно тем же путём, каким готовит текст к
// синтезу (/api/text/preview), поэтому фронт ничего не пересчитывает и не может
// показать одно, а отправить в модель другое. Синтез здесь не запускается и ни
// одна модель TTS не поднимается — только читается текст.
//
// Тумблер ударений начинается со значения панели настроек той же вкладки: preview
// обязан показывать то, что произойдёт при текущих настройках, а не собственный
// дефолт, который разошёлся бы с рендером.
//
// Здесь же preview конкретной реплики проекта: тот же эндпоинт и те же стадии,
// но текст и голос бэкенд берёт по project_id и индексу. Карточку реплики рисует
// app.js — модуль только возвращает ей разбор и просит перерисовать.
import {
  state, $, esc, api, showAlert, engineLabel, renderReplicaCards,
} from './app.js';

const PREVIEW_MOUNTS = [
  { id: 'dialogue-preview', accent: 'auto-accent', fromTextarea: null },
  { id: 'text-preview', accent: 'text-auto-accent', fromTextarea: 'text-body' },
];

// Тестовые данные, а не фраза для записи. Слова подобраны так, чтобы «ё» стояла
// и в начале, и после согласной, — на них видно работу шага восстановления «ё» в
// панели «Что услышит модель». В RECORD_PHRASES её намеренно нет: искусственная
// плотность одного звука ломает ровный темп начитки, а он и определяет качество
// клонирования (см. правило выше).
const YO_STRESS_TEST_PHRASE = 'Ёж и ёлка стоят у забора, а рядом идёт актёр в жёлтом плаще: он несёт мёд и лёд, поёт песню и совсем не мёрзнет.';

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

export {
  renderPreviewPanels,
  syncPreviewAccent,
  clearPreviewMount,
  replicaPreviewHtml,
  previewReplica,
};
