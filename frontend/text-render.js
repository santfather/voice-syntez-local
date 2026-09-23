// Сплошной текст: один голос на весь файл — панель, запуск задачи и её прогресс.
//
// Модуль вынесен из монолита app.js без изменения поведения (F-W3).
// Это не вкладка «Диалог»: там текст разбирается на реплики, роли и take'и, а здесь
// весь текст читается одним голосом и склеивается в один файл без пауз по спикерам.
// Общего у них только панель аудио-настроек, поэтому и разбор диалога (`loadDialogueFile`
// с проверкой кодировки и размера файла) остался в точке входа — он относится к диалогу.
//
// Ручки движка живут в общем `state` (`textEngineParams`/`textEngineVoice`), а не в
// модуле: сброс вкладки и переключение режима движка в app.js трогают их напрямую.
import {
  state, $, esc, api, showAlert, voiceById, ENGINE_F5,
  engineNote, cloningNote, engineParamsHtml,
  chosenEngineParams, readEngineParams, shortUtterancePayload, etaLabel,
  watchEngines,
} from './app.js';
// Отмена задачи — общий путь с очередью озвучки голосов, живёт в модуле голосов.
import { cancelJob } from './voices.js';

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

export {
  renderTextVoiceOptions,
  renderTextEngineParams,
  readTextEngineParams,
  updateTextSummary,
  updateTextButton,
  loadTextFile,
  setTextProgress,
  renderText,
  cancelTextRender,
};
