// The page: a sentence, the week it is read onto, and the events kept. Moving a reading before
// saving it, or an event after, is a correction; the worker solves for weights that read it so.
import { useEffect, useRef, useState } from 'react'
import { type Block, Calendar } from './Calendar.tsx'
import { Composer } from './Composer.tsx'
import { Inspector, type Run } from './Inspector.tsx'
import { call, onFrame, onStep } from './client.ts'
import { fromDay, iso, parseISO, toDay, weekStart } from './core/calendar.ts'
import type { Guess, Learnable, Parse, Probe, Slot, Solved, Summary } from './protocol.ts'
import './calendar.css'

const date = new Date()
const TODAY = toDay(date.getFullYear(), date.getMonth() + 1, date.getDate()), THIS_WEEK = weekStart(TODAY)
export const REF = iso(TODAY)
const MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
// Read as on the Sunday before, so that each weekday named is the one in the week shown.
const SEEDS = ['standup mon 9:30am for 15 min', 'design review tue 2pm', 'lunch with dana wed 12-1', 'gym fri 6-7:30pm', 'dinner with mom sat 7pm']
const SEED_REF = iso(THIS_WEEK - 1)

type Entry = {
  id: number
  text: string // its sentence, and the day it was read on: together they name its correction
  ref: string
  title: string
  slot: Slot
  taught: boolean    // a correction stands for it
  learning: boolean  // that correction is queued or being solved
  unlearned?: string
}

let ids = 0, runs = 0
const entry = (p: Parse, ref: string, slot: Slot): Entry => ({ id: ++ids, text: p.text, ref, title: p.title || 'New event', slot, taught: false, learning: false })
/** A reading as a place on the calendar; a start with no end read is given an hour. */
const slotOf = (g: Guess): Slot => ({ date: g.date, start: g.start, end: g.start === null ? null : g.end ?? Math.min(g.start + 60, 1440) })
const sameSlot = (a: Slot, b: Slot) => a.date === b.date && a.start === b.start && a.end === b.end
const missing = (l: Learnable) => (['date', 'start', 'end'] as const).filter(k => !l[k]).join(', ')

/** For a page around the app: the sentence being read (typed, or the selected event's), and the model after it loads and after each solve. */
type Props = { onSentence?(text: string, ref: string): void; onModel?(summary: Summary, solved?: Solved & { text: string; learnable?: Learnable }): void }

export function App({ onSentence, onModel }: Props = {}) {
  const [summary, setSummary] = useState<Summary>()
  const [text, setText] = useState('')
  const [parse, setParse] = useState<Parse>()
  const [pick, setPick] = useState(0)     // which reading is chosen
  const [moved, setMoved] = useState<Slot>() // where the chosen reading was dragged to
  const [events, setEvents] = useState<Entry[]>([])
  const [selected, setSelected] = useState<number>()
  const [week, setWeek] = useState(THIS_WEEK)
  const [run, setRun] = useState<Run>()
  const [version, setVersion] = useState(0)
  const [open, setOpen] = useState(false)
  const [learned, setLearned] = useState('') // what the last solve took new words to mean
  const said = useRef(new Set<string>())
  const kept = useRef(events)
  kept.current = events
  const latest = useRef('')                // the sentence as typed, for responses that arrive late
  const solves = useRef(Promise.resolve()) // one solve at a time, so each step belongs to the run shown

  useEffect(() => {
    let live = true
    void (async () => {
      // the model files may still be on their way
      for (;;) {
        try { const s = await call('init'); if (live) { setSummary(s); onModel?.(s) } break } catch { await new Promise(r => setTimeout(r, 2000)) }
        if (!live) return
      }
      const parses = await Promise.all(SEEDS.map(t => call('parse', t, SEED_REF)))
      if (live) setEvents(parses.map(p => entry(p, SEED_REF, slotOf(p.guesses[0]))))
    })()
    return () => { live = false }
  }, [])

  function read(t: string) {
    latest.current = t
    if (!t.trim()) return setParse(undefined)
    call('parse', t, REF).then(p => { if (latest.current === t) setParse(p) }, () => {})
  }
  function type(t: string) {
    setText(t)
    setPick(0)
    setMoved(undefined)
    if (t) setSelected(undefined)
    read(t)
  }

  const guesses = parse?.guesses ?? []
  const chosen = moved ?? (guesses[pick] && slotOf(guesses[pick]))
  // The week follows the chosen reading.
  useEffect(() => { if (chosen) setWeek(weekStart(parseISO(chosen.date)!)) }, [chosen?.date])

  /** The events other than e, to be read beside its sentence while a solve runs. */
  const others = (e?: Entry): Probe[] => kept.current.filter(x => x.id !== e?.id).map(x => ({ text: x.text, ref: x.ref }))
  /** Run one solve: its steps and frames as they come, then what it left. */
  function solve(id: number, text: string, ref: string, work: () => Promise<Solved & { learnable?: Learnable }>) {
    solves.current = solves.current.then(async () => {
      setRun({ id: ++runs, text, ref, steps: [], frames: [], done: false })
      const offStep = onStep(step => setRun(r => ({ ...r!, steps: [...r!.steps, step] })))
      const offFrame = onFrame(frame => setRun(r => ({ ...r!, frames: [...r!.frames, frame] })))
      const off = () => { offStep(); offFrame() }
      try {
        const s = await work()
        setRun(r => ({ ...r!, steps: s.trace, done: true, ms: s.ms }))
        setSummary(s.summary)
        setVersion(v => v + 1)
        onModel?.(s.summary, { ...s, text })
        // each meaning is said once, though every later solve finds it again
        const news = s.aliases.map(a => `${a.word} = ${a.means}`).filter(a => !said.current.has(a))
        news.forEach(a => said.current.add(a))
        setLearned(news.join('   '))
        // a word learned as a date or a time is no longer part of a title
        const titles = new Map(await Promise.all(kept.current.map(async e => [e.id, (await call('parse', e.text, e.ref)).title] as const)))
        setEvents(es => es.map(e => ({ ...e, title: titles.get(e.id) || e.title, ...(e.id === id && { learning: false, unlearned: s.learnable && missing(s.learnable) }) })))
        read(latest.current)
      } finally {
        off()
      }
    })
  }
  function correct(e: Entry, slot: Slot) {
    setEvents(es => es.map(x => (x.id === e.id ? { ...x, slot, taught: true, learning: true } : x)))
    solve(e.id, e.text, e.ref, () => call('teach', { text: e.text, ref: e.ref, ...slot }, others(e)))
  }
  async function commit() {
    if (!summary || !text.trim()) return
    const fresh = parse?.text === text
    const p = fresh ? parse : await call('parse', text, REF)
    const top = slotOf(p.guesses[0]), slot = fresh && chosen ? chosen : top
    const e = entry(p, REF, slot)
    setEvents(es => [...es, e])
    setSelected(e.id)
    setText('')
    setParse(undefined)
    setPick(0)
    setMoved(undefined)
    latest.current = ''
    if (!sameSlot(slot, top)) correct(e, slot)
  }
  function remove(e: Entry) {
    setEvents(es => es.filter(x => x.id !== e.id))
    if (e.taught) solve(e.id, e.text, e.ref, () => call('forget', { text: e.text, ref: e.ref, ...e.slot }, others(e)))
  }

  const steps = run && !run.done ? run.steps.length : 0
  const title = parse?.title || 'New event'
  // a correction the solve could not make the sentence read as: the event stays, marked like a part that cannot be learned
  const lost = (e: Entry) => !e.learning && summary?.corrections.some(c => c.text === e.text && c.ref === e.ref && !c.held)
  const blocks: Block[] = [
    ...events.map(e => ({ id: `e${e.id}`, title: e.title, slot: e.slot, active: e.id === selected, learning: e.learning ? steps : undefined, unlearned: e.unlearned || (lost(e) ? 'not learned' : undefined) })),
    // a reading the user chose over the first, or moved, is theirs: it shows at full strength
    ...guesses.map((g, i) => ({ id: `g${i}`, title, slot: i === pick && moved ? moved : slotOf(g), ghost: i === pick && (moved || pick > 0) ? 1 : g.p, active: i === pick })),
  ]
  const ghost = (id: string) => id[0] === 'g'
  const eventOf = (id: string) => events.find(e => e.id === +id.slice(1))!
  const shown = fromDay(week + 3), about = text ? undefined : events.find(e => e.id === selected)
  const reading = about?.text ?? text, readOn = about?.ref ?? REF
  useEffect(() => { if (reading.trim()) onSentence?.(reading, readOn) }, [reading, readOn])

  return (
    <div className="app" data-ready={summary ? '' : undefined}>
      <main className="stage">
        <header className="bar" onMouseDown={e => { if (!(e.target instanceof HTMLInputElement)) e.preventDefault() /* the caret stays in the sentence */ }}>
          <div className="bar-when">
            <h1>{MONTHS[shown.m - 1]} <span>{shown.y}</span></h1>
            <button aria-label="Previous week" onClick={() => setWeek(w => w - 7)}><svg viewBox="0 0 8 12" aria-hidden="true"><path d="M6 1.5 1.5 6 6 10.5" /></svg></button>
            <button onClick={() => setWeek(THIS_WEEK)}>Today</button>
            <button aria-label="Next week" onClick={() => setWeek(w => w + 7)}><svg viewBox="0 0 8 12" aria-hidden="true"><path d="M2 1.5 6.5 6 2 10.5" /></svg></button>
          </div>
          <div className="bar-sentence">
            <Composer value={text} parse={parse} ready={!!summary} onChange={type} onCommit={commit} onEscape={() => { setPick(0); setMoved(undefined) }} />
            {learned && <output key={version} className="learned">{learned}</output>}
          </div>
          <button className="code" aria-expanded={open} onClick={() => setOpen(o => !o)}>
            read <i>=</i> <b>weights</b><i>(</i><span>"event.bin"</span><i>)</i>
          </button>
        </header>
        <Calendar today={TODAY} week={week} blocks={blocks}
          onPress={id => {
            const i = +id.slice(1)
            if (!ghost(id)) return setSelected(i)
            // from a marker at the edge, a reading brings its week with it
            setWeek(weekStart(parseISO((i === pick && moved ? moved : slotOf(guesses[i])).date)!))
            if (i !== pick) { setPick(i); setMoved(undefined) }
          }}
          onAdjust={(id, slot) => (ghost(id) ? setMoved(slot) : setEvents(es => es.map(e => (e.id === +id.slice(1) ? { ...e, slot } : e))))}
          onRelease={(id, slot) => { if (slot && !ghost(id)) correct(eventOf(id), slot) }}
          onAction={id => (ghost(id) ? commit() : remove(eventOf(id)))} />
      </main>
      <aside className="panel" data-open={open || undefined} inert={!open}>
        <div className="panel-body">
          <Inspector text={reading} today={readOn} summary={summary} run={run} version={version} others={events.map(e => ({ text: e.text, ref: e.ref }))} />
        </div>
      </aside>
    </div>
  )
}
