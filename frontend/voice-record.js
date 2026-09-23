// Запись голоса с микрофона: фразы начитки, мастер интонационных профилей,
// индикатор уровня и отправка готовой записи в голос или в форму нового голоса.
//
// Модуль вынесен из монолита app.js без изменения поведения (F-W3).
// Запись идёт прямо в профиль выбранного голоса, а не в новый голос: одна фраза —
// один профиль одной интонации (§16). Форму нового голоса ветка мастера не трогает,
// иначе та же запись осела бы в двух местах сразу.
//
// Вкладка «Сам себе звукорежиссер» (`state.recording` в app.js) — другая подсистема:
// она про запись реплик диалога. Общие с ней места здесь — только выбор формата
// контейнера, индикатор уровня и тексты ошибок доступа к микрофону; они экспортируются.
import { state, $, esc, api, showAlert, voiceById, loadVoices, emotionLabel } from './app.js';

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

// Наружу — то, что зовут app.js (разметка и события вкладки «Голоса») и voices.js
// (сброс записи после смены голоса). `drawMeterBar`, `recordFormat` и
// `recordErrorMessage` общие с вкладкой «Запись»: там рисуют тот же индикатор.
export {
  renderRecordTargets,
  renderRecordPhrases,
  showRecordPhrase,
  renderRecordPlan,
  advanceRecordPhrase,
  startRecording,
  stopRecording,
  analyzeRecording,
  resetRecording,
  recordFormat,
  recordErrorMessage,
  drawMeterBar,
};
