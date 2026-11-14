// The talk (docs/design.md section 13): one black stage, moved on by key, with the calendar inside it.
// → / Space / PageDown: next. ← / PageUp: back. In the calendar's sentence the arrows move the caret once
// there is text and Space types; Esc leaves the sentence. #n in the address is the beat, and can be typed.
// Nothing fades or moves between beats; each one is drawn as it is.
import { type CSSProperties, useEffect, useRef, useState } from 'react'
import { createRoot } from 'react-dom/client'
import { App, REF } from '../src/App.tsx'
import { call, onFrame } from '../src/client.ts'
import { iso, parseISO } from '../src/core/calendar.ts'
import { NO_TIME, endOf } from '../src/core/time.ts'
import type { Inspection, Learnable, Slot, Solved, Summary } from '../src/protocol.ts'
import '../src/styles.css'
import './talk.css'
import prices from './nvda.json'

const meme = new URL('./meme.jpg', import.meta.url).href
const video = new URL('./demo.mp4', import.meta.url).href
const report = new URL('./report.png', import.meta.url).href

// 'gene' and 'rmv' (Sheet and Ids below) are out of the talk for now; put them back in this list to show them.
const BEATS = ['blank', 'rise', 'trend', 'drop1', 'why1', 'drop2', 'name', 'meme', 'question', 'ran', 'quote', 'report', 'semi', 'demo', 'video', 'read', 'learn'] as const
type Beat = (typeof BEATS)[number]
const at = (b: Beat) => BEATS.indexOf(b)
const DEMO = at('demo')
const clamp = (n: number) => Math.max(0, Math.min(BEATS.length - 1, n))
const fromHash = () => clamp(+location.hash.slice(1) || 0)

function Talk() {
  const [beat, setBeat] = useState(fromHash)
  useEffect(() => { history.replaceState(null, '', `#${beat}`) }, [beat])
  useEffect(() => {
    const go = () => setBeat(fromHash())
    const key = (e: KeyboardEvent) => {
      if (e.metaKey || e.ctrlKey || e.altKey || e.repeat) return // browser shortcuts, and a held key, move nothing
      const field = e.target instanceof HTMLInputElement ? e.target : undefined
      if (field && e.key === 'Escape') return field.blur()
      // in the sentence an arrow moves the caret once there is text, and Space is typed
      const free = !field || (field.value === '' && e.key !== ' ')
      const step = e.key === 'PageDown' ? 1 : e.key === 'PageUp' ? -1 : !free ? 0 : e.key === 'ArrowRight' || e.key === ' ' ? 1 : e.key === 'ArrowLeft' ? -1 : 0
      if (!step) return
      e.preventDefault()
      setBeat(b => clamp(b + step))
    }
    addEventListener('keydown', key)
    addEventListener('hashchange', go)
    return () => { removeEventListener('keydown', key); removeEventListener('hashchange', go) }
  }, [])
  // entering the calendar puts the caret in its sentence; leaving it takes the caret out
  useEffect(() => {
    if (beat === DEMO) document.querySelector<HTMLInputElement>('.composer input')?.focus()
    else if (document.activeElement instanceof HTMLElement) document.activeElement.blur()
  }, [beat])

  const lesson = useLesson(beat)
  const name = BEATS[beat]
  return (
    <>
      <div className="talk" data-beat={name}>
        <Chart beat={beat} />
        <p className="question" data-on={name === 'question' || undefined}>為什麼他出軌，<span>綠</span>的是我們？</p>
        <Program name={name} />
        <blockquote className="quote" data-on={name === 'quote' || undefined}>
          <p>“It is invariably the case that a program, once in operation, will be felt to be only part of the answer to the problems at hand.”</p>
          <cite>Peter Naur, 1985</cite>
        </blockquote>
        <figure className="report" data-on={name === 'report' || undefined}>
          <img src={report} alt="" />
          <figcaption>紐約時報中文網，2026/10/30（英文版 10/29）</figcaption>
        </figure>
        <Video on={name === 'video'} />
        <Architecture beat={beat} lesson={lesson} />
      </div>
      <div className="demo" data-on={beat === DEMO || undefined} inert={beat !== DEMO}>
        <App onModel={onModel} />
      </div>
    </>
  )
}

/* ---------- The stock: real closes (Yahoo Finance), one chart from mid-July to 11/13 ---------- */

const day = (d: string) => prices.findIndex(([q]) => q === d)
const START = day('2026-07-16') // the window opens before the first of the rising lows
const CUT = day('2026-10-06') // the highest close
const D1 = day('2026-11-06') // the last close before Huang's email of 11/8
const LAST = prices.length - 1
const close = (i: number) => prices[i][1] as number
const pct = (a: number, b: number) => Math.round((close(b) / close(a) - 1) * 100)
const signed = (n: number) => `${n > 0 ? '+' : n < 0 ? '−' : ''}${Math.abs(n)}%`
const date = (i: number) => { const d = prices[i][0] as string; return `${+d.slice(5, 7)}/${+d.slice(8)}` }

// the days to the right of 10/6 are empty until they happen; the lower third holds the labels and the name
const X = (i: number) => 150 + ((i - START) / (LAST - START)) * 1300
const Y = (p: number) => 600 - ((p - 155) / 90) * 520
const line = (a: number, b: number) => prices.slice(a, b + 1).map(([, p], k) => `${k ? 'L' : 'M'}${X(a + k).toFixed(1)} ${Y(p as number).toFixed(1)}`).join('')

// What a chart reader would point at on 10/6. All from the closes in nvda.json:
const LOWS = ['2026-07-29', '2026-08-24', '2026-09-14'].map(day) // each dip ended higher: 190.01, 208.48, 210.96
const support = (i: number) => close(LOWS[0]) + ((close(LOWS[2]) - close(LOWS[0])) / (LOWS[2] - LOWS[0])) * (i - LOWS[0])
const HIGH = day('2026-09-04') // 230.36, the last high before; closed above it from 10/1

// The largest falls before 11/8, with what the news tied them to: the whole chip sector, not Nvidia alone.
// 10/22 AI spending and borrowing costs; 10/30 Intel's Q4 guidance (Intel −29%); 11/5 chip stocks (Nasdaq −5.5%, AMD and Micron −17%).
const FALLS: [number, string][] = [[day('2026-10-22'), 'AI 泡沫疑慮'], [day('2026-10-30'), 'Intel 財測差'], [day('2026-11-05'), '晶片股崩跌']]

function Chart({ beat }: { beat: number }) {
  const name = BEATS[beat], on = (b: Beat) => beat >= at(b)
  const shown = on('rise') && !on('question')
  return (
    <svg className="chart" viewBox="0 0 1600 900" data-on={shown || undefined}>
      {name === 'trend' && <g className="ann">
        <path d={`M${X(LOWS[0] - 4)} ${Y(support(LOWS[0] - 4))}L${X(CUT)} ${Y(support(CUT))}`} />
        {LOWS.map(i => <circle key={i} cx={X(i)} cy={Y(close(i))} r={8} />)}
        <text x={X(LOWS[1])} y={Y(support(LOWS[1])) + 78} textAnchor="middle">低點越來越高</text>
        <path d={`M${X(HIGH)} ${Y(close(HIGH))}H${X(CUT)}`} />
        <circle cx={X(HIGH)} cy={Y(close(HIGH))} r={8} />
        <text x={X(CUT) - 30} y={Y(close(HIGH)) - 24} textAnchor="end">突破前高</text>
      </g>}
      <path className="rise" d={line(START, CUT)} />
      {!on('drop1') && <circle className="up" cx={X(CUT)} cy={Y(close(CUT))} r={9} />}
      {/* the two falls draw themselves; everything else is there at once */}
      {shown && on('drop1') && <>
        <path className="fall" d={line(CUT, D1)} pathLength={1} />
        <text className="when" x={X(CUT)} y={Y(close(CUT)) - 28} textAnchor="middle">{date(CUT)}</text>
      </>}
      {shown && on('drop1') && !on('drop2') && <circle className="down late" cx={X(D1)} cy={Y(close(D1))} r={9} />}
      {on('why1') && <g className="ann">
        {FALLS.map(([i, why], k) => {
          const lx = X(FALLS[2][0]) - (2 - k) * 190 // a row of labels ending under the last fall
          return <g key={i}>
            <path d={`M${lx} 616L${X(i)} ${Y(close(i)) + 16}`} />
            <circle cx={X(i)} cy={Y(close(i))} r={8} />
            <text x={lx} y={652} textAnchor="middle" className="when">{date(i)} {signed(pct(i - 1, i))}</text>
            <text x={lx} y={692} textAnchor="middle">{why}</text>
          </g>
        })}
        <text x={X(D1) + 20} y={Y(close(D1)) - 20} className="total">{date(D1)} <tspan>{signed(pct(CUT, D1))}</tspan></text>
      </g>}
      {shown && on('drop2') && <>
        <path className="fall" d={line(D1, LAST)} pathLength={1} />
        <circle className="down late" cx={X(LAST)} cy={Y(close(LAST))} r={9} />
        <text x={X(LAST) + 20} y={Y(close(LAST)) + 50} textAnchor="middle" className="total late">{date(LAST)} <tspan>{signed(pct(D1, LAST))}</tspan></text>
      </>}
      {on('name') && <text className="name" x={150} y={800}>NVIDIA</text>}
      {/* the reason for the second fall, over the chart, its right edge just left of where that fall begins */}
      {name === 'meme' && <g className="reason">
        <image href={meme} x={X(D1) - 40 - 597} y={110} width={597} height={740} preserveAspectRatio="xMaxYMin meet" />
        <text x={X(D1) - 40} y={876} textAnchor="end">圖：HADISY AI ACADEMY</text>
      </g>}
    </svg>
  )
}

/* ---------- The rule as a program, then with one function whose body is a description ---------- */

function Program({ name }: { name: Beat }) {
  const ran = name === 'ran', semi = name === 'semi'
  return (
    <div className="prog" data-on={ran || semi || undefined} data-semi={semi || undefined}>
      {semi && <>
        <div className="ln"><i>def</i><span>沒出大事(公司):</span></div>
        <div className="ln"><i /><span className="in">新聞 = scrape_news(公司)</span></div>
        <div className="ln"><i /><span className="in said">「新聞裡，沒有會讓股價大跌的事」</span></div>
        <div className="ln gap" />
      </>}
      {/* each mark sits right after its line */}
      <div className="ln"><i>if</i><span>收盤價 &gt; 前高{ran && <b>✓</b>}</span></div>
      <div className="ln"><i>and</i><span>低點 &gt; 前一個低點{ran && <b>✓</b>}</span></div>
      <div className="ln"><i>and</i><span>收盤價 &gt; 均線(50){ran && <b>✓</b>}</span></div>
      {semi && <div className="ln"><i>and</i><span>沒出大事("NVIDIA")</span></div>}
      <div className="ln"><i /><span>買進(){ran && <em>{signed(pct(CUT, LAST))}</em>}</span></div>
    </div>
  )
}

/* ---------- Two readings decided before use, out of the talk for now: gene names in a spreadsheet, a name at the RMV ---------- */

// Excel sees a month and a day in these gene symbols and stores dates (the HGNC renamed the genes in 2020)
const GENES = [['MARCH', '1', '1-Mar'], ['SEPT', '1', '1-Sep'], ['SEPT', '2', '2-Sep']]

/** The symbols as typed, the month and day the spreadsheet finds in them, then the dates it keeps. */
function Sheet({ on }: { on: boolean }) {
  return (
    <table className="sheet" data-on={on || undefined}>
      <thead><tr><th /><th>A</th></tr></thead>
      <tbody>
        <tr><th>1</th><td><span>gene</span></td></tr>
        {GENES.map(([month, d, date], i) => (
          <tr key={i} style={{ '--i': i } as CSSProperties}>
            <th>{i + 2}</th>
            <td><span className="typed"><b>{month}</b><b>{d}</b></span><span className="date">{date}</span></td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

function Ids({ on }: { on: boolean }) {
  return (
    <div className="ids" data-on={on || undefined}>
      <div className="id"><small>護照</small><i /><div><span>Given name</span><b>HEN-CHEN</b></div></div>
      <span className="ne">≠</span>
      <div className="id"><small>RMV</small><i /><div><span>First name</span><b>HEN<s>-CHEN</s></b></div></div>
    </div>
  )
}

/** The recorded demo, for when the live one fails: plays muted from the start on its beat; a click pauses. */
function Video({ on }: { on: boolean }) {
  const ref = useRef<HTMLVideoElement>(null)
  useEffect(() => {
    const v = ref.current!
    if (on) { v.currentTime = 0; v.play().catch(() => {}) } else v.pause()
  }, [on])
  return <video ref={ref} className="video" src={video} muted playsInline preload="auto" data-on={on || undefined}
    onClick={e => { const v = e.currentTarget; if (v.paused) v.play().catch(() => {}); else v.pause() }} />
}

/* ---------- After the demo: the sentence, the weights, the program they give, and where it puts the event ---------- */

type Reading = { how: { date: string; start: string; end: string }; slot: Slot }
type Lesson = { text: string; title: string; before: Reading; after: Reading; changed: number; parameters: number }

/** The most probable program of each part, and the place they give together. */
function reading(i: Inspection): Reading {
  const d = i.date[0], s = i.start[0], e = i.end[0]
  const start = !s || s.value === NO_TIME ? null : s.value
  const end = start === null ? null : e ? endOf(e.value, start) : Math.min(start + 60, 1440)
  return { how: { date: d.how, start: s?.how ?? '', end: e?.how ?? '' }, slot: { date: d.label, start, end } }
}
const changed = (l: Lesson) => (['date', 'start', 'end'] as const).some(k => l.before.how[k] !== l.after.how[k])

// The demo's corrections: the frame a solve posts before its first step holds the sentence as it was
// read; when the solve is done, the sentence is read again.
let before: Inspection | undefined
onFrame(f => { if (!f.step) before = f.inspection })
let publish: (l: Lesson) => void = () => {}
let latest: Lesson | undefined
async function onModel(summary: Summary, solved?: Solved & { text: string; learnable?: Learnable }) {
  const was = before
  before = undefined
  const c = solved?.learnable && summary.corrections.find(x => x.text === solved.text)
  if (!was || !c) return
  const [now, parse] = await Promise.all([call('inspect', c.text, c.ref), call('parse', c.text, c.ref)])
  const l: Lesson = { text: c.text, title: parse.title, before: reading(was), after: reading(now), changed: summary.changed, parameters: summary.parameters }
  if (changed(l)) publish(latest = l)
}

const FALLBACK = 'movie night tmrw 4p'
/** The demo's last correction that changed a program. With none, the figure teaches one itself, so it can be shown on its own. */
function useLesson(beat: number): Lesson | undefined {
  const [lesson, setLesson] = useState(latest)
  const taught = useRef(false)
  useEffect(() => { publish = setLesson }, [])
  useEffect(() => {
    if (beat < at('read') || lesson || taught.current) return
    taught.current = true
    void (async () => {
      const c = { text: FALLBACK, ref: REF, date: iso(parseISO(REF)! + 1), start: 960, end: 1020 }
      // the frame posted before the first step is the sentence as it was read (onFrame above)
      // the weights may still be loading; a correction from the demo meanwhile makes this one unneeded
      for (let tries = 0; tries < 60 && !latest; tries++) {
        const s = await call('teach', c).catch(() => undefined)
        if (s) return onModel(s.summary, { ...s, text: FALLBACK })
        await new Promise(r => setTimeout(r, 1000))
      }
    })()
  }, [beat, lesson])
  return lesson
}

/** A program as the page writes it, with the parts a lay reader does not need left out. */
const plain = (how: string) => how.replace(/minute: :00, /, '').replace(/, named$/, '')

function Architecture({ beat, lesson }: { beat: number; lesson?: Lesson }) {
  const shown = beat === at('read') || beat === at('learn'), learn = beat === at('learn')
  if (!lesson) return <div className="arch" data-on={shown || undefined} />
  const { before, after } = lesson
  const parts = [['day', 'date'], ['start', 'start'], ['end', 'end']] as const
  return (
    <div className="arch" data-on={shown || undefined} data-learn={learn || undefined}>
      <div className="a-sentence">{lesson.text}</div>
      <i className="a-arrow" />
      <div className="a-weights">
        <span>event.bin</span>
        <small>{lesson.parameters.toLocaleString('en-US')} 個數字</small>
        <small className="a-moved">{lesson.changed.toLocaleString('en-US')} 個改變</small>
      </div>
      <i className="a-arrow" />
      <div className="a-program">
        {parts.map(([label, k]) => {
          const was = plain(before.how[k]), now = plain(after.how[k])
          if (!was && !now) return null
          return (
            <div key={k} className="a-line">
              <span>{label}</span><i>=</i>
              {was === now ? <code>{now}</code> : <code><del>{was}</del><ins>{now}</ins></code>}
            </div>
          )
        })}
      </div>
      <i className="a-arrow" />
      <Week title={lesson.title || 'New event'} from={before.slot} to={after.slot} learn={learn} />
      {/* back from the place to the program that gives it, and from that program to the weights; x in tenths of a percent of the row (talk.css columns) */}
      <svg className="a-back" viewBox="0 0 920 100">
        <defs><marker id="head" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" /></marker></defs>
        <path d="M791 4 C791 70, 560 76, 520 14" markerEnd="url(#head)" />
        <path d="M440 14 C410 76, 300 70, 271 6" markerEnd="url(#head)" />
      </svg>
    </div>
  )
}

const DAY = (z: number) => { const d = new Date(`${iso(z)}T12:00`); return `${d.toLocaleDateString('en-US', { weekday: 'short' })} ${d.getDate()}` }
const hm = (m: number) => `${m / 60 % 12 | 0 || 12}${m % 60 ? `:${String(m % 60).padStart(2, '0')}` : ''} ${m < 720 ? 'AM' : 'PM'}`

/** A few days around the two places, and the event moving from the first to the second. */
function Week({ title, from, to, learn }: { title: string; from: Slot; to: Slot; learn: boolean }) {
  const a = parseISO(from.date)!, b = parseISO(to.date)!, lo = Math.min(a, b), hi = Math.max(a, b)
  // ponytail: two places more than three days apart get one column each, with the days between left out
  const days = hi - lo <= 3 ? Array.from({ length: Math.max(3, hi - lo + 1) }, (_, i) => lo + i) : [lo, hi]
  // an all-day reading has no hour; it sits in the first row shown
  const timed = [from, to].filter(s => s.start !== null), starts = timed.map(s => s.start!), ends = timed.map(s => s.end!)
  if (!timed.length) { starts.push(480); ends.push(540) }
  const h0 = Math.max(0, Math.floor(Math.min(...starts) / 60) - 1), h1 = Math.min(24, Math.max(h0 + 4, Math.ceil(Math.max(...ends) / 60) + 1))
  const place = (s: Slot): CSSProperties => {
    const col = days.indexOf(parseISO(s.date)!), start = s.start ?? h0 * 60, end = s.end ?? start + 60
    return { left: `${(col / days.length) * 100}%`, width: `${100 / days.length}%`, top: `${((start - h0 * 60) / ((h1 - h0) * 60)) * 100}%`, height: `${((end - start) / ((h1 - h0) * 60)) * 100}%` }
  }
  const slot = learn ? to : from
  return (
    <div className="a-week" style={{ '--days': days.length, '--hours': h1 - h0 } as CSSProperties}>
      <div className="a-days">{days.map(z => <span key={z} data-today={iso(z) === REF || undefined}>{DAY(z)}</span>)}</div>
      <div className="a-grid">
        <div className="a-block" style={place(slot)}><b>{title}</b><span>{slot.start === null ? 'all day' : hm(slot.start)}</span></div>
      </div>
    </div>
  )
}

createRoot(document.getElementById('root')!).render(<Talk />)
