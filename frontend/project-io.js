// Экспорт и импорт проекта: скачивание архива/аудио/субтитров и открытие чужого
// `.ttsproject`.
//
// Модуль вынесен из монолита app.js без изменения поведения (F-W3).
// Экспорт — обычное скачивание файла, а не задача очереди: ни один из этих
// вариантов не синтезирует, файл собирается из уже готовых take'ов и уходит
// ответом. Поэтому здесь нет опроса статуса — только загрузка и ошибка.
import {
  state, $, showAlert, api, loadVoices, rememberProject, applyProject, exportDiagnostics,
} from './app.js';

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
  if (button) {
    button.disabled = !ready || state.exportBusy === true;
    button.title = ready ? '' : 'Сначала разберите текст на реплики';
  }
  // Диагностика возможна и на проекте без take'ов: тогда в архиве не будет аудио,
  // но текст, настройки и журнал останутся — а именно их чаще всего и не хватает.
  const diagnosticsButton = $('btn-diagnostics');
  if (diagnosticsButton) {
    diagnosticsButton.disabled = !state.project;
    diagnosticsButton.title = state.project ? '' : 'Сначала создайте проект';
  }
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
  const diagnosticsButton = $('btn-diagnostics');
  if (diagnosticsButton) diagnosticsButton.addEventListener('click', exportDiagnostics);
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

export { updateExportButtons, downloadExport, bindExportEvents };
