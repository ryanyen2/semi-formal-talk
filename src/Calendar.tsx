// The week: day columns, an all-day row and a time grid. Everything on it is a block; the page
// says which blocks are readings (ghosts) and which are events, and hears how they were moved.
import { type CSSProperties, type PointerEvent as ReactPointerEvent, useEffect, useRef, useState } from 'react'
import { type Day, fromDay, iso, parseISO, weekStart } from './core/calendar.ts'
import type { Slot } from './protocol.ts'

export type Block = {
  id: string
  title: string
  slot: Slot
  ghost?: number     // a reading: its probability
  active: boolean    // the chosen reading, or the selected event
  learning?: number  // optimiser steps so far, while its correction is being solved
  unlearned?: string // the parts of its correction that could not be learned
}
type Props = {
  today: Day
  week: Day // the Monday shown
  blocks: Block[]
  onPress(id: string): void
  onAdjust(id: string, slot: Slot): void
  onRelease(id: string, slot?: Slot): void // slot: where it was left, when that is not where it was
  onAction(id: string): void               // the corner control: save a reading, delete an event
}

const HOUR = 56, STEP = 15, LONGEST = 240
const WEEKDAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v))
/** Snap with friction: the value holds until the pointer is well past halfway to the next step. */
const stick = (raw: number, held: number, step: number) => (Math.abs(raw - held) < step * 0.65 ? held : Math.round(raw / step) * step)
const clock = (m: number) => `${Math.floor(m / 60) % 12 || 12}${m % 60 ? ':' + String(m % 60).padStart(2, '0') : ''}`
const half = (m: number) => (m % 1440 < 720 ? 'AM' : 'PM')
const times = ({ start, end }: Slot) =>
  start === null || end === null ? '' : `${clock(start)}${half(start) === half(end) ? '' : ' ' + half(start)} – ${clock(end)} ${half(end)}`
/** A reading's opacity: the root keeps an unsure first reading legible, the floor keeps one the weights give almost nothing findable. */
const faint = (p: number) => Math.max(0.4, Math.sqrt(p))
const nowMinutes = () => new Date().getHours() * 60 + new Date().getMinutes()

export function Calendar({ today, week, blocks, onPress, onAdjust, onRelease, onAction }: Props) {
  const scroll = useRef<HTMLDivElement>(null), view = useRef<HTMLDivElement>(null)
  const [dragging, setDragging] = useState<string>()
  const [now, setNow] = useState(nowMinutes)
  // Blocks and day headers sit on a strip measured in columns from this Monday; the strip slides to the week shown.
  const origin = weekStart(today)
  // While the strip slides, the weeks it passes over need their headers too.
  const [from, setFrom] = useState(week)
  useEffect(() => { const t = setTimeout(() => setFrom(week), 600); return () => clearTimeout(t) }, [week])
  const first = Math.min(from, week), days = Array.from({ length: Math.abs(from - week) + 7 }, (_, i) => first + i)

  useEffect(() => { const t = setInterval(() => setNow(nowMinutes()), 30_000); return () => clearInterval(t) }, [])
  useEffect(() => { scroll.current!.scrollTop = 7.25 * HOUR }, [])

  // Bring the chosen reading into view when it is read at an hour that is scrolled away.
  const chosen = blocks.find(b => b.ghost !== undefined && b.active)?.slot
  useEffect(() => {
    const el = scroll.current!
    if (dragging || chosen?.start == null || chosen.end === null) return
    const top = (chosen.start / 60) * HOUR, bottom = (chosen.end / 60) * HOUR
    // once the reading has rested: half-typed, "4pm" reads as four in the morning for a moment
    const t = setTimeout(() => { if (top < el.scrollTop || bottom > el.scrollTop + el.clientHeight) el.scrollTo({ top: top - el.clientHeight / 3 }) }, 350)
    return () => clearTimeout(t)
  }, [chosen?.start, chosen?.end, chosen?.date])

  function grab(e: ReactPointerEvent, b: Block, mode: 'move' | 'start' | 'end') {
    if (e.button !== 0) return
    e.stopPropagation()
    onPress(b.id)
    setDragging(b.id)
    const pane = scroll.current!
    /** The pointer in columns and minutes; the all-day row is everything above the time grid. */
    const at = (p: { clientX: number; clientY: number }) => {
      const r = view.current!.getBoundingClientRect()
      return { col: (p.clientX - r.left) / (r.width / 7), min: ((p.clientY - r.top) / HOUR) * 60, allDay: p.clientY < pane.getBoundingClientRect().top }
    }
    let slot = b.slot, moved = false, col = parseISO(slot.date)! - week
    const length = slot.start === null ? 60 : slot.end! - slot.start
    // where along the block it was grabbed, so it does not jump under the pointer
    const hold = slot.start === null ? length / 2 : at(e).min - slot.start

    const move = (ev: PointerEvent) => {
      // ponytail: the pane scrolls only while the pointer moves near its edge; a timer loop if long drags matter
      const edge = pane.getBoundingClientRect()
      if (ev.clientY > edge.bottom - 24) pane.scrollTop += 8
      else if (ev.clientY < edge.top + 24 && ev.clientY > edge.top) pane.scrollTop -= 8
      const p = at(ev)
      let next = slot
      if (mode === 'move') {
        col = clamp(stick(p.col - 0.5, col, 1), 0, 6)
        const start = clamp(stick(p.min - hold, slot.start ?? NaN, STEP), 0, 1440 - length)
        next = p.allDay ? { date: iso(week + col), start: null, end: null } : { date: iso(week + col), start, end: start + length }
      } else if (mode === 'start') {
        next = { ...slot, start: clamp(stick(p.min, slot.start!, STEP), Math.max(0, slot.end! - LONGEST), slot.end! - STEP) }
      } else {
        next = { ...slot, end: clamp(stick(p.min, slot.end!, STEP), slot.start! + STEP, Math.min(1440, slot.start! + LONGEST)) }
      }
      if (next.date === slot.date && next.start === slot.start && next.end === slot.end) return
      slot = next
      moved = true
      onAdjust(b.id, next)
    }
    const up = () => {
      removeEventListener('pointermove', move)
      removeEventListener('pointerup', up)
      removeEventListener('pointercancel', up)
      setDragging(undefined)
      onRelease(b.id, moved ? slot : undefined)
    }
    // on the window, not captured by the block: the block is a new element when it crosses into or out of the all-day row
    addEventListener('pointermove', move)
    addEventListener('pointerup', up)
    addEventListener('pointercancel', up)
  }

  const shown = (b: Block) => { const col = parseISO(b.slot.date)! - week; return col >= 0 && col < 7 }
  const item = (b: Block) => {
    const { start, end, date } = b.slot, timed = start !== null && end !== null
    const ghost = b.ghost !== undefined
    const style = {
      transform: `translate(${(parseISO(date)! - origin) * 100}%, ${timed ? (start / 60) * HOUR : 0}px)`,
      height: timed ? ((end - start) / 60) * HOUR : undefined,
      '--p': ghost ? faint(b.ghost!) : 1,
      '--steps': b.learning ?? 0,
    } as CSSProperties
    return (
      <div key={b.id} className="block" style={style} tabIndex={ghost ? undefined : 0} title={b.unlearned}
        data-kind={ghost ? 'ghost' : 'event'} data-active={b.active || undefined} data-dragging={dragging === b.id || undefined}
        data-learning={b.learning !== undefined || undefined} data-unlearned={b.unlearned ? '' : undefined}
        data-size={!timed ? 'day' : end - start < 30 ? 'xs' : end - start < 60 ? 's' : undefined}
        onPointerDown={e => grab(e, b, 'move')}
        onKeyDown={e => { if (!ghost && (e.key === 'Backspace' || e.key === 'Delete')) onAction(b.id) }}>
        <div className="block-face">
          <span className="block-title">{b.title}</span>
          {timed && <span className="block-time">{times(b.slot)}</span>}
          {b.active && (
            <button className="block-action" tabIndex={-1} aria-label={ghost ? 'Save' : 'Delete'}
              onPointerDown={e => { e.stopPropagation(); e.preventDefault() }} onClick={() => onAction(b.id)}>
              <svg viewBox="0 0 12 12" aria-hidden="true">{ghost ? <path d="M9.5 2.5v3a2 2 0 0 1-2 2h-5m2-2.2-2.2 2.2 2.2 2.2" /> : <path d="m3 3 6 6m0-6-6 6" />}</svg>
            </button>
          )}
        </div>
        {timed && b.active && <i className="block-grip" data-edge="start" onPointerDown={e => grab(e, b, 'start')} />}
        {timed && b.active && <i className="block-grip" data-edge="end" onPointerDown={e => grab(e, b, 'end')} />}
      </div>
    )
  }
  // A reading in another week: a mark on the side it lies on, at its hour.
  const marker = (b: Block) => {
    const before = parseISO(b.slot.date)! < week, { y, m, d } = fromDay(parseISO(b.slot.date)!)
    return (
      <button key={b.id} className="cal-away" data-side={before ? 'before' : 'after'} aria-label={`${MONTHS[m - 1]} ${d}, ${y}`}
        style={{ top: ((b.slot.start ?? 0) / 60) * HOUR, '--p': Math.max(0.7, faint(b.ghost!)) } as CSSProperties}
        onClick={() => onPress(b.id)}>
        <svg viewBox="0 0 8 12" aria-hidden="true"><path d={before ? 'M6 1.5 1.5 6 6 10.5' : 'M2 1.5 6.5 6 2 10.5'} /></svg>
        {MONTHS[m - 1]} {d}{y === fromDay(week + 3).y ? '' : ` ${y}`}
      </button>
    )
  }
  const allDay = blocks.filter(b => b.slot.start === null), timed = blocks.filter(b => b.slot.start !== null)
  const away = (bs: Block[]) => bs.filter(b => b.ghost !== undefined && !shown(b)).map(marker)

  return (
    <div className="cal" data-dragging={dragging ? '' : undefined}
      // only an event takes the focus: a reading is adjusted with the caret still in the sentence
      onMouseDown={e => { if (!(e.target as Element).closest('[data-kind=event]')) e.preventDefault() }}
      style={{ '--week': (week - origin) / 7, '--hour': `${HOUR}px` } as CSSProperties}>
      <div className="cal-row cal-head">
        <div className="cal-gutter" />
        <div className="cal-view">
          <div className="cal-strip">
            {days.map(z => (
              <div key={z} className="cal-day" data-today={z === today || undefined} style={{ transform: `translateX(${(z - origin) * 100}%)` }}>
                {WEEKDAYS[(z - first) % 7]} <b>{fromDay(z).d}</b>
              </div>
            ))}
          </div>
        </div>
      </div>
      <div className="cal-row cal-allday">
        <div className="cal-gutter">all-day</div>
        <div className="cal-view cal-columns">
          <div className="cal-strip">{allDay.map(item)}</div>
          {away(allDay)}
        </div>
      </div>
      <div className="cal-scroll" ref={scroll}>
        <div className="cal-row cal-grid">
          <div className="cal-gutter">
            {Array.from({ length: 23 }, (_, i) => <span key={i} style={{ top: (i + 1) * HOUR }}>{(i + 1) % 12 || 12} {i + 1 < 12 ? 'AM' : 'PM'}</span>)}
          </div>
          <div className="cal-view cal-columns cal-hours" ref={view}>
            <div className="cal-strip">
              <div className="cal-today" style={{ transform: `translateX(${(today - origin) * 100}%)` }}><i style={{ top: (now / 60) * HOUR }} /></div>
              {timed.map(item)}
            </div>
            {away(timed)}
          </div>
        </div>
      </div>
    </div>
  )
}
