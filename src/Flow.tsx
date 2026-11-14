// How one sentence reads, drawn so that a solve can be watched step by step. Tokens on the left,
// the values each part can take on the right, and between them what each token gives each value:
// pink where a program takes the token's number, blue-gray where the value is less probable
// without the word. Below, the probes: the sentence with one thing changed, and the other events,
// each placed on a day's track. Every number drawn is one the worker returned.
import { type PointerEvent, useEffect, useMemo, useRef, useState } from 'react'
import type { Frame, Inspection, Option, Placed, Slot } from './protocol.ts'

/**
 * Eases each keyed vector toward its target. The last number of a vector is its opacity: a key that
 * arrives fades in where it lands, and one that leaves fades out where it is, then is dropped.
 */
export function useEased(target: Map<string, number[]>, ms = 280): Map<string, number[]> {
  const now = useRef(new Map<string, number[]>())
  const [, redraw] = useState(0)
  useEffect(() => {
    const c = now.current
    for (const [k, v] of target) if (!c.has(k)) c.set(k, [...v.slice(0, -1), 0])
    if (matchMedia('(prefers-reduced-motion: reduce)').matches) {
      c.clear()
      for (const [k, v] of target) c.set(k, [...v])
      return redraw(n => n + 1)
    }
    let last = performance.now(), raf = 0
    const tick = () => {
      // the clock, not the frame's timestamp, which can be earlier than the time this effect began
      const t = performance.now(), a = 1 - Math.exp(-Math.max(0, t - last) / (ms / 4))
      last = t
      let moving = false
      for (const [k, v] of c) {
        const to = target.get(k) ?? [...v.slice(0, -1), 0]
        for (let i = 0; i < v.length; i++) {
          v[i] += (to[i] - v[i]) * a
          if (Math.abs(to[i] - v[i]) > 1e-3) moving = true
          else v[i] = to[i]
        }
        if (!target.has(k) && v[v.length - 1] < 0.01) c.delete(k)
      }
      redraw(n => n + 1)
      if (moving) raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [target, ms])
  return now.current
}

const PARTS = [['date', 'day'], ['start', 'start'], ['end', 'end']] as const
type Part = (typeof PARTS)[number][0]
const W = 348, TOK = 22, HEAD = 18, ROW = 20, GAP = 8
const XT = 92     // right edge of the tokens
const XC = 196    // left edge of the values
const THICK = 14  // a ribbon's width at probability 1
const LEAST = 0.02 // thinner than this is not drawn
const pct = (p: number) => (p >= 0.995 ? '100%' : p < 0.01 ? '<1%' : `${Math.round(p * 100)}%`)
const DOW = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'], MON = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
/** "Sat 14", with the month when it is not today's. */
export function dayLabel(iso: string, today: string) {
  const d = new Date(`${iso}T12:00`)
  return `${DOW[d.getDay()]} ${d.getDate()}${iso.slice(0, 7) === today.slice(0, 7) ? '' : ` ${MON[d.getMonth()]}`}`
}
/** The values worth a row: the most probable, and up to two more above 3%. */
const shown = (os: Option[]) => os.filter((o, i) => i === 0 || o.p >= 0.03).slice(0, 3)

type Value = { part: Part; o: Option; top: boolean }
type Ribbon = { value: string; token: number; kind: 'read' | 'sway' }
/** Where everything goes for one inspection: values [y, p, p before or -1, opacity], part names [y, opacity], ribbons [width, opacity]. */
function arrange(r: Inspection, before?: Inspection) {
  const to = new Map<string, number[]>(), values = new Map<string, Value>(), ribbons = new Map<string, Ribbon>()
  let y = 0
  for (const [part] of PARTS) {
    const os = shown(r[part])
    if (!os.length) continue
    to.set(`h:${part}`, [y + HEAD / 2, 1])
    y += HEAD
    os.forEach((o, i) => {
      const key = `c:${part}:${o.value}`
      to.set(key, [y + ROW / 2, o.p, before?.[part].find(b => b.value === o.value)?.p ?? -1, 1])
      values.set(key, { part, o, top: i === 0 })
      for (const kind of ['sway', 'read'] as const)
        o[kind].forEach((v, token) => {
          if (v < LEAST) return
          const id = `r:${key}:${token}:${kind}`
          to.set(id, [v, 1])
          ribbons.set(id, { value: key, token, kind })
        })
      y += ROW
    })
    y += GAP
  }
  return { to, values, ribbons, height: Math.max(y - GAP, r.tokens.length * TOK) }
}

const band = (x0: number, y0: number, x1: number, y1: number, w: number) => {
  const m = (x0 + x1) / 2, h = w / 2
  return `M${x0},${y0 - h}C${m},${y0 - h} ${m},${y1 - h} ${x1},${y1 - h}L${x1},${y1 + h}C${m},${y1 + h} ${m},${y0 + h} ${x0},${y0 + h}Z`
}

export type Focus = { rows: number[]; how?: string }

/** What a token gives the values shown, in words, from the numbers the worker returned. */
function tokenSays(r: Inspection, k: number, today: string): string {
  const t = r.tokens[k], out: string[] = []
  for (const [part, name] of PARTS)
    for (const o of shown(r[part])) {
      const label = part === 'date' ? dayLabel(o.label, today) : o.label
      if (o.read[k] >= LEAST) out.push(`${name} ${label} takes its number (${pct(o.read[k])})`)
      if (o.sway[k] >= LEAST) out.push(`${name} ${label}: ${pct(o.p)} with it, ${pct(o.p - o.sway[k])} without`)
    }
  return out.length ? out.slice(0, 2).join(' · ') : `no value shown depends on "${t.text}"`
}

export function Flow({ r, before, today, onFocus }: { r: Inspection; before?: Inspection; today: string; onFocus(f?: Focus): void }) {
  const { to, values, ribbons, height } = useMemo(() => arrange(r, before), [r, before])
  // what a key was last, so that one leaving can still be drawn while it fades
  const seenValues = useRef(new Map<string, Value>()), seenRibbons = useRef(new Map<string, Ribbon>())
  for (const [k, v] of values) seenValues.current.set(k, v)
  for (const [k, v] of ribbons) seenRibbons.current.set(k, v)
  const now = useEased(to)
  const [hover, setHover] = useState<{ token?: number; value?: string }>({})
  const lit = (b: Ribbon) => (hover.token === undefined && hover.value === undefined ? undefined : hover.token === b.token || hover.value === b.value)

  const enter = (h: typeof hover, f: Focus) => { setHover(h); onFocus(f) }
  const leave = () => { setHover({}); onFocus(undefined) }
  return (
    <div className="flow" style={{ height }} onPointerLeave={leave}>
      <svg className="flow-ribbons" width={W} height={height} aria-hidden="true">
        {[...now].filter(([k]) => k[0] === 'r').map(([k, [w, o]]) => {
          const b = seenRibbons.current.get(k)!, y1 = now.get(b.value)?.[0]
          if (y1 === undefined || b.token >= r.tokens.length) return null
          const on = lit(b)
          return <path key={k} d={band(XT + 4, b.token * TOK + TOK / 2, XC - 6, y1, Math.max(1, w * THICK))} data-kind={b.kind} data-lit={on} style={{ opacity: o }} />
        })}
      </svg>
      {r.tokens.map((t, k) => (
        <span key={k} className="flow-token insp-token" data-role={t.role} style={{ top: k * TOK, right: W - XT }} tabIndex={0}
          onPointerEnter={() => enter({ token: k }, { rows: t.rows, how: tokenSays(r, k, today) })} onFocus={() => enter({ token: k }, { rows: t.rows, how: tokenSays(r, k, today) })} onBlur={leave}>
          {t.text}
        </span>
      ))}
      {[...now].map(([k, v]) => {
        if (k[0] === 'h') {
          const part = k.slice(2) as Part
          return <span key={k} className="flow-part" style={{ top: v[0] - HEAD / 2, left: XC, opacity: v[1] }}>{PARTS.find(p => p[0] === part)![1]}</span>
        }
        if (k[0] !== 'c') return null
        const { part, o, top } = seenValues.current.get(k)!, [y, p, was, op] = v
        const label = part === 'date' ? dayLabel(o.label, today) : o.label
        return (
          <div key={k} className="flow-value" data-top={top || undefined} style={{ top: y - ROW / 2, left: XC, opacity: op }} tabIndex={0}
            onPointerEnter={() => enter({ value: k }, { rows: [], how: `${o.how} · ${pct(o.p)}` })} onFocus={() => enter({ value: k }, { rows: [], how: `${o.how} · ${pct(o.p)}` })} onBlur={leave}>
            <span className="flow-label">{label}</span>
            <span className="flow-bar"><i style={{ transform: `scaleX(${p})` }} />{was >= 0 && <b style={{ left: `${was * 100}%` }} />}</span>
            <span className="flow-p">{pct(p)}</span>
          </div>
        )
      })}
    </div>
  )
}

const DAY = 1440
const span = (s: Slot) => (s.start === null ? { left: '0%', width: '100%' } : { left: `${(s.start / DAY) * 100}%`, width: `${(Math.max(s.end ?? s.start + 60, s.start + 5) - s.start) / DAY * 100}%` })
const clock = (m: number) => `${Math.floor(m / 60)}:${String(m % 60).padStart(2, '0')}`
const hours = (s: Slot) => (s.start === null ? 'all day' : `${clock(s.start)}–${clock(s.end ?? s.start + 60)}`)
const same = (a: Placed, b: Placed) => a.date === b.date && a.start === b.start && a.end === b.end

/** The probes, each where it reads now; during a solve, also where it read before, outlined. */
export function Probes({ r, before, today }: { r: Inspection; before?: Inspection; today: string }) {
  return (
    <div className="probes">
      {r.probes.map((q, i) => {
        const now = r.placed[i], b = before?.probes.findIndex(x => x.text === q.text && x.ref === q.ref) ?? -1
        const was = b >= 0 ? before!.placed[b] : undefined, moved = was && !same(was, now)
        return (
          <div key={q.text + q.ref} className="probe" data-moved={moved || undefined} data-variant={q.mark ? '' : undefined}>
            <span className="probe-text">{q.mark ? <>{q.text.slice(0, q.mark[0])}<b>{q.text.slice(...q.mark)}</b>{q.text.slice(q.mark[1])}</> : q.text}</span>
            <span className="probe-day">{was && was.date !== now.date && <s>{dayLabel(was.date, today)}</s>}{dayLabel(now.date, today)}<span>{hours(now)}</span></span>
            <span className="probe-track">
              {moved && <i className="probe-was" style={span(was)} />}
              <i className="probe-now" style={{ ...span(now), opacity: Math.max(0.35, now.p) }} />
            </span>
          </div>
        )
      })}
    </div>
  )
}

/** The steps of a solve as a track: the frame before it, then one tick per step, pink once every correction holds. */
export function Scrub({ frames, at, onSeek, onPlay, done, ms }: { frames: Frame[]; at: number; onSeek(i: number): void; onPlay(): void; done: boolean; ms?: number }) {
  const n = frames.length, f = frames[at], x = (i: number) => `${n > 1 ? (i / (n - 1)) * 100 : 0}%`
  const seek = (e: PointerEvent<HTMLDivElement>) => {
    const box = e.currentTarget.getBoundingClientRect()
    onSeek(Math.round(Math.min(1, Math.max(0, (e.clientX - box.left) / box.width)) * (n - 1)))
  }
  return (
    <div className="scrub">
      <div className="scrub-row">
        <button className="scrub-play" onClick={onPlay} aria-label="Play the solve again"><svg viewBox="0 0 10 10" aria-hidden="true"><path d="M2.5 1.5v7l6-3.5z" /></svg></button>
        <div className="scrub-track" role="slider" aria-label="Step of the solve" aria-valuemin={0} aria-valuemax={n - 1} aria-valuenow={at} tabIndex={0}
          onPointerDown={e => { e.currentTarget.setPointerCapture(e.pointerId); seek(e) }}
          onPointerMove={e => { if (e.currentTarget.hasPointerCapture(e.pointerId)) seek(e) }}
          onKeyDown={e => { if (e.key === 'ArrowLeft') onSeek(Math.max(0, at - 1)); if (e.key === 'ArrowRight') onSeek(Math.min(n - 1, at + 1)) }}>
          <span className="scrub-fill" style={{ width: x(at) }} />
          {frames.map((g, i) => <i key={i} style={{ left: x(i) }} data-held={(g.step && g.step.held === g.of) || undefined} data-before={!g.step || undefined} />)}
          <b style={{ left: x(at) }} />
        </div>
      </div>
      <div className="insp-label">
        <span>{f.step ? `step ${at} of ${n - 1}` : 'before'}</span>
        <span>{f.step ? `${f.step.held} of ${f.of} hold` : ''}{done && at === n - 1 && ms !== undefined ? ` · ${ms.toLocaleString('en-US')} ms` : ''}</span>
      </div>
    </div>
  )
}
