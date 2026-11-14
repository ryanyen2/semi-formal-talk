// The talk (docs/design.md section 13): one white stage, moved on by key, with the calendar inside it.
// → / Space / PageDown: next. ← / PageUp: back. In the calendar's sentence the arrows move the caret once
// there is text and Space types; Esc leaves the sentence. #n in the address is the beat, and can be typed.
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

const BEATS = ['blank', 'rise', 'trend', 'rule', 'drop', 'name', 'meme', 'question', 'quote', 'gene', 'renamed', 'rmv', 'demo', 'read', 'learn', 'end'] as const
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
        <Program beat={beat} />
        <img className="meme" src={meme} alt="" data-on={name === 'meme' || undefined} />
        <small className="credit" data-on={name === 'meme' || undefined}>圖：HADISY AI ACADEMY</small>
        <p className="question" data-on={name === 'question' || undefined}>為什麼他出軌，<span>綠</span>的是我們？</p>
        <blockquote className="quote" data-on={name === 'quote' || undefined}>
          <p>“It is invariably the case that a program, once in operation, will be felt to be only part of the answer to the problems at hand.”</p>
          <cite>Peter Naur, 1985</cite>
        </blockquote>
        <Sheet beat={beat} />
        <Ids on={name === 'rmv'} />
        <Architecture beat={beat} lesson={lesson} />
      </div>
      <div className="demo" data-on={beat === DEMO || undefined} inert={beat !== DEMO}>
        <App onModel={onModel} />
      </div>
    </>
  )
}

/* ---------- The stock: real closes, 2026-03-27 to 2026-11-13 (Yahoo Finance), cut where it peaked ---------- */

const day = (d: string) => prices.findIndex(([q]) => q === d)
const CUT = day('2026-10-06')
const LAST = prices.length - 1
const close = (i: number) => prices[i][1] as number
// the straight line through the last 25 closes before the cut, carried on: what the rule expects
const slope = (() => {
  const from = CUT - 24, mx = from + 12, my = prices.slice(from, CUT + 1).reduce((s, [, p]) => s + (p as number), 0) / 25
  let num = 0, den = 0
  for (let i = from; i <= CUT; i++) { num += (i - mx) * (close(i) - my); den += (i - mx) ** 2 }
  return num / den
})()
const EXPECT = close(CUT) + slope * (LAST - CUT)
const LO = 148, HI = Math.max(262, EXPECT + 6)
const X = (i: number) => 150 + (i / LAST) * 1300
const Y = (p: number) => 600 - ((p - LO) / (HI - LO)) * 520 // the lower third is left for the rule
const line = (from: number, to: number) => prices.slice(from, to + 1).map(([, p], k) => `${k ? 'L' : 'M'}${X(from + k).toFixed(1)} ${Y(p as number).toFixed(1)}`).join('')
const md = (d: string) => `${+d.slice(5, 7)}/${+d.slice(8)}`

// What a chart reader would point at on 10/6. All from the closes in nvda.json:
const GAIN = Math.round((close(CUT) / close(0) - 1) * 100) // +43% since 3/27
const LOWS = ['2026-07-29', '2026-08-24', '2026-09-14'].map(day) // each dip ended higher: 190.01, 208.48, 210.96
const support = (i: number) => close(LOWS[0]) + ((close(LOWS[2]) - close(LOWS[0])) / (LOWS[2] - LOWS[0])) * (i - LOWS[0])
const PEAK = day('2026-05-14') // 235.74, the highest close before October; closed above on 10/5 and 10/6

function Chart({ beat }: { beat: number }) {
  const on = (b: Beat) => beat >= at(b)
  return (
    <svg className="chart" viewBox="0 0 1600 900" data-on={(on('rise') && !on('question')) || undefined}>
      <line className="now" x1={X(CUT)} x2={X(CUT)} y1={60} y2={640} data-on={on('rule') || undefined} />
      <g className="ann" data-on={(on('trend') && !on('drop')) || undefined}>
        <g>
          <path className="span" d={`M${X(0)} 106V92H${X(CUT)}V106`} />
          <text x={(X(0) + X(CUT)) / 2} y={72} textAnchor="middle">半年 +{GAIN}%</text>
        </g>
        <g>
          <path d={`M${X(LOWS[0] - 4)} ${Y(support(LOWS[0] - 4))}L${X(CUT)} ${Y(support(CUT))}`} />
          {LOWS.map(i => <circle key={i} cx={X(i)} cy={Y(close(i))} r={8} />)}
          <text x={X(LOWS[1])} y={Y(support(LOWS[1])) + 78} textAnchor="middle">低點越來越高</text>
        </g>
        <g>
          <path d={`M${X(PEAK)} ${Y(close(PEAK))}H${X(CUT)}`} />
          <circle cx={X(PEAK)} cy={Y(close(PEAK))} r={8} />
          <text x={X(CUT) - 30} y={Y(close(PEAK)) - 24} textAnchor="end">突破前高</text>
        </g>
      </g>
      <path className="rise" d={line(0, CUT)} pathLength={1} data-on={on('rise') || undefined} />
      <path className="guess" d={`M${X(CUT)} ${Y(close(CUT))}L${X(LAST)} ${Y(EXPECT)}`} data-on={on('rule') || undefined} data-past={on('drop') || undefined} />
      <path className="fall" d={line(CUT, LAST)} pathLength={1} data-on={on('drop') || undefined} />
      <circle className="dot up" cx={X(CUT)} cy={Y(close(CUT))} r={9} data-on={(on('rise') && !on('drop')) || undefined} />
      <circle className="dot down" cx={X(LAST)} cy={Y(close(LAST))} r={9} data-on={on('drop') || undefined} />
      <g className="dates" data-on={on('name') || undefined}>
        <text x={X(0)} y={Y(close(0)) + 52} textAnchor="middle">2026/{md(prices[0][0] as string)}</text>
        <text x={X(CUT)} y={Y(close(CUT)) - 30} textAnchor="middle">{md(prices[CUT][0] as string)}</text>
        <text x={X(LAST)} y={Y(close(LAST)) + 52} textAnchor="middle">{md(prices[LAST][0] as string)}</text>
      </g>
      <text className="name" x={150} y={800} data-on={on('name') || undefined}>NVIDIA</text>
    </svg>
  )
}

/* ---------- The rule as a program: under the chart, and at the end with one line left informal ---------- */

function Program({ beat }: { beat: number }) {
  const name = BEATS[beat], end = name === 'end'
  return (
    <div className="prog" data-at={beat > at('drop') ? 'center' : 'chart'} data-on={(beat >= at('rule') && beat <= at('drop')) || end || undefined} data-faint={name === 'drop' || undefined}>
      <div className="ln"><i>if</i><span>財報好</span><b>✓</b></div>
      <div className="ln"><i>and</i><span>產品強</span><b>✓</b></div>
      <div className="ln"><i>and</i><span>市場熱</span><b>✓</b></div>
      <div className="ln extra" data-on={end || undefined}><i>and</i><span className="said">「公司沒出大事」</span><b /></div>
      <div className="ln act"><i /><span>買進()</span></div>
    </div>
  )
}

/* ---------- Two readings decided before use: gene names in a spreadsheet, a name at the RMV ---------- */

// Excel reads these symbols as dates; in 2020 the HGNC renamed the genes instead
const GENES = [['MARCH1', '1-Mar', 'MARCHF1'], ['SEPT1', '1-Sep', 'SEPTIN1'], ['SEPT2', '2-Sep', 'SEPTIN2']]

function Sheet({ beat }: { beat: number }) {
  const name = BEATS[beat]
  return (
    <table className="sheet" data-on={name === 'gene' || name === 'renamed' ? name : undefined}>
      <thead><tr><th /><th>A</th></tr></thead>
      <tbody>
        <tr><th>1</th><td><span>gene</span></td></tr>
        {GENES.map(([typed, date, now], i) => (
          <tr key={typed} style={{ '--i': i } as CSSProperties}>
            <th>{i + 2}</th>
            <td><span className="typed">{typed}</span><span className="date">{date}</span><span className="now"><del>{typed}</del><ins>{now}</ins></span></td>
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
        <path d="M791 4 C791 70, 560 76, 520 14" pathLength={1} markerEnd="url(#head)" />
        <path d="M440 14 C410 76, 300 70, 271 6" pathLength={1} markerEnd="url(#head)" />
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
