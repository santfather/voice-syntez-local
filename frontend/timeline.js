// Таймлайн проекта: дорожки спикеров, ось времени и переход к карточке реплики.
//
// Модуль вынесен из монолита app.js без изменения поведения (F-W3).
// Таймлайн не считает время сам: start/end/длительность приходят из `/timeline`,
// где они посчитаны по реальным длительностям активных take'ов. Своя арифметика
// на клиенте (по символам или по числу реплик) разошлась бы с файлом на первом же
// рендере — а обещать пользователю место звука в файле нужно точно.
import { state, $, esc, api, showAlert, applyTimeline } from './app.js';

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

export { renderTimeline, refreshTimeline, bindTimelineEvents };
