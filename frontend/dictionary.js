// Словарь произношения и предложения для него.
//
// Модуль вынесен из монолита app.js без изменения поведения (F-W3).
// Словарь глобальный: правило, добавленное здесь, применяется во всех проектах.
// Проверка текста идёт тем же эндпоинтом, что и стадии синтеза, поэтому фронт не
// повторяет порядок шагов и не может показать одно, а отправить в модель другое.
//
// Предложения лежат рядом со словарём, а не отдельно: своего хранилища у них нет,
// подтверждение и отклонение — обычные записи словаря через тот же CRUD. Кандидатов
// считает бэкенд тем же путём, что и preview («/api/pronunciation/suggestions»),
// поэтому фронт ничего не пересчитывает. Отклонение создаёт выключенное правило с
// пометкой — оно остаётся на вкладке 04 и работает памятью об отказе, поэтому слово
// больше не предлагается (фильтр на бэкенде смотрит и на выключенные правила).
import {
  state, $, esc, api, showAlert, voiceById,
} from './app.js';

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

export {
  renderDictionaryEngineOptions,
  loadDictionary,
  editDictionaryEntry,
  resetDictionaryForm,
  saveDictionaryEntry,
  updateDictionaryEntry,
  deleteDictionaryEntry,
  previewDictionary,
  resetSuggestions,
  findSuggestions,
  confirmSuggestion,
  rejectSuggestion,
  addAllSuggestions,
};
