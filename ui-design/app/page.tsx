'use client'

import { useState } from 'react'

const voices = [
  { name: 'АРТЕМ // PRIME', type: 'F5-TTS', color: 'lime', meta: 'RU · Низкий тембр', accent: '98%', bars: [18, 31, 22, 44, 28, 50, 36, 25, 46, 31, 54, 40, 29, 48, 34, 42] },
  { name: 'LENA // NEURAL', type: 'XTTS v2', color: 'violet', meta: 'RU · Мягкий тембр', accent: '94%', bars: [28, 42, 35, 52, 24, 44, 58, 35, 48, 30, 54, 40, 28, 46, 38, 56] },
  { name: 'ИГОРЬ // CORE', type: 'F5-TTS', color: 'lime', meta: 'RU · Средний тембр', accent: '91%', bars: [14, 24, 40, 30, 48, 27, 41, 53, 32, 45, 24, 37, 51, 29, 43, 34] },
]

function Waveform({ bars, color = 'lime' }: { bars: number[]; color?: string }) {
  return <div className={`waveform ${color}`} aria-label="Audio waveform">{bars.map((height, i) => <i key={i} style={{ height }} />)}</div>
}

function Slider({ label, value, setValue }: { label: string; value: number; setValue: (value: number) => void }) {
  return <label className="slider-row"><span>{label}</span><input type="range" min="0" max="100" value={value} onChange={(e) => setValue(Number(e.target.value))} /><b>{value}%</b></label>
}

export default function Page() {
  const [tab, setTab] = useState('Голоса')
  const [accent, setAccent] = useState(true)
  const [playing, setPlaying] = useState<string | null>(null)
  const [cfg, setCfg] = useState(72)
  const [temp, setTemp] = useState(48)

  return (
    <main className="studio-shell">
      <header className="topbar">
        <div className="brand"><div className="brand-mark"><span /><span /><span /><span /><span /></div><div><strong>VOICE<span>_</span> SYNTEZ</strong><small>LOCAL AI VOICE SYNTHESIS STUDIO</small></div><em>v0.9.7</em></div>
        <div className="status-cluster"><div className="status"><span className="online-dot" /> BACKEND <b>ONLINE</b></div><div className="status"><small>DEVICE</small><b>APPLE SILICON MPS</b></div><div className="status"><small>ENGINE</small><b>F5-TTS <span className="purple">●</span></b></div><div className="memory"><small>MEMORY</small><div><span style={{ width: '42%' }} /></div><b>6.8 / 16 GB</b></div><div className="status accent-status"><span className="online-dot" /> ACCENTIZER <b>READY</b></div></div><button className="icon-button" aria-label="Settings">⚙</button>
      </header>
      <nav className="tabs" aria-label="Studio sections">{['Голоса', 'Озвучка диалога', 'Сплошной текст'].map((item, i) => <button key={item} className={tab === item ? 'active' : ''} onClick={() => setTab(item)}><span>0{i + 1}</span>{item}{tab === item && <i />}</button>)}</nav>
      <section className="workspace">
        {tab === 'Голоса' && <><div className="section-heading"><div><p className="eyebrow">VOICE LIBRARY / 03 ACTIVE</p><h1>Голоса <span>//</span> My Voices</h1></div><button className="outline-button">↻ Синхронизировать</button></div><div className="voice-layout"><article className="panel creator-panel"><div className="panel-title"><span className="number">01</span><div><h2>NEW VOICE CREATOR</h2><p>Создайте голос из референсного аудио</p></div><span className="signal">● REC READY</span></div><div className="dropzone"><div className="upload-icon">↥</div><strong>Перетащите аудиофайл сюда</strong><span>или нажмите для выбора · WAV, MP3, FLAC</span><div className="mini-wave"><Waveform bars={[12, 20, 15, 28, 35, 18, 30, 43, 26, 18, 36, 22, 29, 15, 24, 34]} color="violet" /></div><small>Рекомендуем: 10–30 секунд чистой речи</small></div><div className="form-grid"><label>ИМЯ ГОЛОСА<input placeholder="например: АРТЕМ // V2" /></label><label>ПОЛ<select defaultValue=""><option value="" disabled>Выберите тип</option><option>Мужской</option><option>Женский</option></select></label></div><div className="notice warning"><span>△</span><div><b>Ожидается файл для анализа</b><small>После загрузки проверим F0 и качество транскрипции</small></div></div><button className="primary-button">✦ АВТО-ТРАНСКРИБИРОВАТЬ <small>⏎</small></button></article><div className="voices-column"><div className="subheading"><span>MY VOICES <b>03</b></span><button className="filter-button">FILTER: ALL⌄</button></div>{voices.map((voice) => <article className="panel voice-card" key={voice.name}><div className="voice-top"><div className={`avatar ${voice.color}`}>{voice.name.slice(0, 2)}</div><div className="voice-name"><h3>{voice.name}</h3><p>{voice.meta}</p></div><span className={`engine-tag ${voice.color}`}>{voice.type}</span><button className="more" aria-label={`More actions for ${voice.name}`}>•••</button></div><div className="card-wave"><Waveform bars={voice.bars} color={voice.color} /><span>{voice.accent} ACCENT</span></div><div className="sliders"><Slider label="CFG STRENGTH" value={voice.name === voices[0].name ? cfg : 64} setValue={setCfg} /><Slider label="TEMPERATURE" value={voice.name === voices[0].name ? temp : 52} setValue={setTemp} /></div><div className="test-row"><input defaultValue="Привет, это тестовый голос." aria-label="Quick test text" /><button className={playing === voice.name ? 'playing' : ''} onClick={() => setPlaying(playing === voice.name ? null : voice.name)}>{playing === voice.name ? '■' : '▶'} ПРОСЛУШАТЬ</button></div></article>)}</div></div></>}
        {tab === 'Озвучка диалога' && <DialoguePanel accent={accent} setAccent={setAccent} />}
        {tab === 'Сплошной текст' && <RawPanel />}
      </section><footer><span>VOICE_SYNTEZ // LOCAL INSTANCE</span><span>GPU TEMP <b>48°C</b> · LATENCY <b>12ms</b> · SESSION <b>00:42:18</b></span></footer>
    </main>
  )
}

function DialoguePanel({ accent, setAccent }: { accent: boolean; setAccent: (v: boolean) => void }) { return <><div className="section-heading"><div><p className="eyebrow">SYNTHESIS PIPELINE / DIALOGUE</p><h1>Озвучка диалога <span>//</span> Script Engine</h1></div><button className="outline-button">＋ Новый проект</button></div><div className="dialogue-grid"><article className="panel script-panel"><div className="panel-title"><span className="number">01</span><div><h2>SCRIPT EDITOR</h2><p>Разметьте реплики и параметры синтеза</p></div><button className="parse-button">✦ ОПРЕДЕЛИТЬ ГОЛОСА</button></div><div className="code-editor">{['АРТЕМ(1): Привет. Система активна.', 'ЛЕНА(2): Отлично. Запускаем протокол?', 'АРТЕМ(1): (speed=1.1) Да, начинаем.', '', '# Поддерживаются: (speed=1.2)', '# Пауза между репликами: (pause=0.5)'].map((line, i) => <div key={i}><span>{String(i + 1).padStart(2, '0')}</span><code className={line.startsWith('#') ? 'comment' : i < 3 ? 'code-line' : ''}>{line || ' '}</code></div>)}</div></article><article className="panel speaker-panel"><div className="panel-title"><span className="number">02</span><div><h2>SPEAKER SLOTS <b>02</b></h2><p>Голоса определены автоматически</p></div></div>{['АРТЕМ (1)', 'ЛЕНА (2)'].map((speaker, i) => <div className="speaker-slot" key={speaker}><span className={`slot-avatar ${i ? 'violet' : ''}`}>{i ? 'Л' : 'А'}</span><div><b>{speaker}</b><select defaultValue={i ? 'LENA // NEURAL' : 'АРТЕМ // PRIME'}><option>АРТЕМ // PRIME</option><option>LENA // NEURAL</option></select></div><label>× {i ? '1.0' : '1.1'} SPEED</label></div>)}<div className="parameter-box"><p>GLOBAL PARAMETERS</p><label>ФОРМАТ<select defaultValue="WAV"><option>WAV</option><option>MP3</option><option>FLAC</option></select></label><label className="switch-row">AUTO-ACCENTIZER <button className={`switch ${accent ? 'on' : ''}`} onClick={() => setAccent(!accent)}><i /></button></label></div><button className="generate-button">▶ СГЕНЕРИРОВАТЬ ДИАЛОГ</button></article></div></> }
function RawPanel() { return <><div className="section-heading"><div><p className="eyebrow">SYNTHESIS PIPELINE / RAW TEXT</p><h1>Сплошной текст <span>//</span> Text Renderer</h1></div></div><div className="raw-grid"><article className="panel raw-drop"><div className="upload-icon">↥</div><h2>Загрузите .TXT или .MD</h2><p>или вставьте текст прямо в редактор</p><button className="outline-button">Выбрать файл</button></article><article className="panel raw-editor"><div className="panel-title"><div><h2>RAW TEXT INPUT</h2><p>Подготовьте текст для рендера</p></div><span className="char-count">0 / 50 000</span></div><textarea placeholder="Вставьте текст, который нужно озвучить..." /><div className="raw-controls"><label>VOICE<select defaultValue="АРТЕМ // PRIME"><option>АРТЕМ // PRIME</option><option>LENA // NEURAL</option></select></label><label>CHUNKING<select defaultValue="По абзацам"><option>По абзацам</option><option>По предложениям</option></select></label><button className="generate-button">▶ РЕНДЕРИТЬ ТЕКСТ</button></div></article></div></> }
