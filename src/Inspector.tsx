// The weight file and how the sentence reads under it: what each token gives each value (Flow),
// the same sentence with one thing changed and the other events (Probes), the latest solve as
// steps to play back (Scrub), and every number the worker holds (Field). Everything here is what
// the worker returned.
import { useEffect, useMemo, useRef, useState } from 'react'
import { call } from './client.ts'
import type { Step } from './core/teach.ts'
import { type Focus, Flow, Probes, Scrub } from './Flow.tsx'
import type { Frame, Inspection, Line, Probe, Summary, Weights } from './protocol.ts'
import './inspector.css'

/** A solve as the page saw it: the sentence it was about, its steps and frames so far, and whether it has finished. */
export type Run = { id: number; text: string; ref: string; steps: Step[]; frames: Frame[]; done: boolean; ms?: number }
export type InspectorProps = {
  text: string      // the sentence in the input box, possibly empty
  today: string     // ISO date
  summary?: Summary // undefined until the model has loaded
  run?: Run         // the latest solve
  version: number   // increases whenever the weights change
  others: Probe[]   // the events on the calendar, read beside the sentence
}

const num = (n: number) => n.toLocaleString('en-US')
const GAP = 22 // CSS px between the first tensor and the rest: room for a label

/** Where each weight sits, in device pixels: one pixel per weight, lines of whole embedding rows, a gap after the first tensor. */
type Layout = { cols: number; split: number; gap: number; width: number; height: number }

function layout(w: Weights, cssWidth: number, dpr: number): Layout {
  const head = w.manifest[0], split = head.r * head.c
  const cols = Math.max(1, Math.floor((cssWidth * dpr) / head.c)) * head.c
  const lines = Math.ceil(split / cols) + Math.ceil((w.now.length - split) / cols)
  const gap = Math.round(GAP * dpr)
  return { cols, split, gap, width: cols, height: lines + gap }
}

const xOf = (L: Layout, i: number) => (i < L.split ? i : i - L.split) % L.cols
const yOf = (L: Layout, i: number) =>
  Math.min(L.height - 1, i < L.split ? Math.floor(i / L.cols) : Math.ceil(L.split / L.cols) + Math.floor((i - L.split) / L.cols) + L.gap)

/** One pixel per weight; `colour` gives a packed ABGR pixel. */
function paint(canvas: HTMLCanvasElement, L: Layout, n: number, colour: (i: number) => number) {
  canvas.width = L.width
  canvas.height = L.height
  const ctx = canvas.getContext('2d')!, img = ctx.createImageData(L.width, L.height), px = new Uint32Array(img.data.buffer)
  for (let i = 0; i < n; i++) px[yOf(L, i) * L.width + xOf(L, i)] = colour(i)
  ctx.putImageData(img, 0, 0)
}

/** A custom property's colour as [r, g, b]. The browser resolves it, so any CSS colour syntax works. */
function token(el: Element, name: string): number[] {
  const ctx = document.createElement('canvas').getContext('2d')!
  ctx.fillStyle = getComputedStyle(el).getPropertyValue(name).trim()
  ctx.fillRect(0, 0, 1, 1)
  return [...ctx.getImageData(0, 0, 1, 1).data.slice(0, 3)]
}
const mix = (a: number[], b: number[], t: number) =>
  (0xff000000 | ((a[2] + (b[2] - a[2]) * t) << 16) | ((a[1] + (b[1] - a[1]) * t) << 8) | (a[0] + (b[0] - a[0]) * t)) >>> 0
const rgba = (a: number[], b: number[], t: number, alpha: number) => `rgba(${a.map((v, k) => Math.round(v + (b[k] - v) * t)).join(',')},${alpha})`

/**
 * A weight counts as moved when it is this far from the shipped value (the worker uses the same).
 * A gradient step touches tens of thousands of weights, almost all by less than 0.001; drawn at
 * all, they hide the few hundred that carry the change.
 */
export const MOVED = 0.01

/** The spread of the shipped values, for the blue-gray scale. */
function spread({ shipped }: Weights) {
  let squares = 0
  for (let i = 0; i < shipped.length; i++) squares += shipped[i] ** 2
  return 3 * Math.sqrt(squares / shipped.length)
}

// The ledger under the file: one line per row, column or tensor that moved, largest first.
const SHOWN = 8, LINE = 12, LABEL = 58, LEDGER_GAP = 12 // lines shown, CSS px per line, label width, space above
const FLY = 520, SHIFT = 300 // ms: from the file to the ledger; a line changing place, or a weight going home
/** Where a moved weight is drawn and where it is going: [x, y, size, opacity] in CSS px. */
type Mark = { key: string; d: number; from: number[]; to: number[]; t0: number; dur: number; arc: number; leaving?: boolean }
const ease = (p: number) => (p < 0.5 ? 4 * p * p * p : 1 - (-2 * p + 2) ** 3 / 2)
function at(m: Mark, t: number): number[] {
  const p = ease(Math.min(1, Math.max(0, (t - m.t0) / m.dur)))
  const v = m.from.map((f, k) => f + (m.to[k] - f) * p)
  v[0] += m.arc * Math.sin(Math.PI * p) // a slight bow, so a flight reads as one weight travelling
  return v
}

/**
 * Every weight as one pixel, blue-gray by size; under it, the moved ones lifted out into lines, each
 * labelled with what it is. When the lines change (a solve step, a scrub, a new solve) a weight that
 * starts moving flies from its place in the file to its line, and one that stops flies back. Its place
 * in the file stays marked pink.
 */
export function Field({ weights, rows, lines = weights?.lines, playing = false, reserve = false }: { weights?: Weights; rows: number[]; lines?: Line[]; playing?: boolean; reserve?: boolean }) {
  const box = useRef<HTMLDivElement>(null), base = useRef<HTMLCanvasElement>(null)
  const lift = useRef<HTMLCanvasElement>(null), mark = useRef<HTMLCanvasElement>(null)
  const marks = useRef(new Map<number, Mark>()), labels = useRef(new Map<string, Mark & { text: string }>())
  const drawn = useRef(false) // whether anything was laid out yet: the first layout does not fly
  const [width, setWidth] = useState(0)
  useEffect(() => {
    const seen = new ResizeObserver(([e]) => setWidth(Math.floor(e.contentRect.width)))
    seen.observe(box.current!)
    return () => seen.disconnect()
  }, [])

  const dpr = window.devicePixelRatio
  const L = useMemo(() => (weights && width ? layout(weights, width, dpr) : undefined), [weights?.manifest, width, dpr]) // eslint-disable-line react-hooks/exhaustive-deps
  const head = weights?.manifest[0]
  const all = lines ?? [], count = all.reduce((n, l) => n + l.cells.length, 0)
  const shown = all.slice(0, SHOWN), more = all.length - shown.length
  const fileH = L ? L.height / dpr : 0, cssW = L ? L.width / dpr : 0
  const ledgerH = (reserve ? SHOWN : shown.length) * LINE + (more > 0 ? LINE : 0)
  const height = fileH + (ledgerH ? LEDGER_GAP + ledgerH : 0)

  useEffect(() => {
    if (!weights || !L) return
    const el = box.current!, wash = token(el, '--slate-wash'), slate = token(el, '--slate'), s = spread(weights)
    paint(base.current!, L, weights.now.length, i => mix(wash, slate, Math.min(1, Math.abs(weights.now[i]) / s)))
  }, [weights, L])

  // Where each moved weight should be now; flights to get there.
  useEffect(() => {
    if (!L) return
    const now = performance.now(), still = matchMedia('(prefers-reduced-motion: reduce)').matches || !drawn.current
    drawn.current = true
    const cw = (cssW - LABEL) / 48, n = weights!.now.length
    const home = (i: number) => [xOf(L, i < n ? i : n - 8 + (i - n)) / dpr, yOf(L, i < n ? i : n - 1) / dpr, 1 / dpr, 1]
    const go = (m: Mark | undefined, to: number[], dur: number, delay = 0, from?: number[]) => {
      if (still) return { from: to, to, t0: now, dur: 1, arc: 0 }
      return { from: from ?? (m ? at(m, now) : to), to, t0: now + delay, dur, arc: m ? 0 : 10 }
    }
    const want = new Map<number, { key: string; d: number; to: number[] }>()
    shown.forEach((l, k) => {
      const y = fileH + LEDGER_GAP + k * LINE + (LINE - cw) / 2
      for (const c of l.cells) if (c.slot < 48) want.set(c.i, { key: l.key, d: c.d, to: [LABEL + c.slot * cw, y, cw - 1, 1] })
    })
    const fresh = [...want.keys()].filter(i => !marks.current.has(i) || marks.current.get(i)!.leaving)
    const spreadOver = Math.min(250, fresh.length * 1.5)
    for (const [i, w] of want) {
      const m = marks.current.get(i)
      const k = fresh.indexOf(i)
      if (k >= 0) marks.current.set(i, { key: w.key, d: w.d, ...go(undefined, w.to, FLY, fresh.length > 1 ? (spreadOver * k) / (fresh.length - 1) : 0, m ? at(m, now) : home(i)) })
      else if (m!.to.some((v, j) => v !== w.to[j]) || m!.d !== w.d) marks.current.set(i, { key: w.key, d: w.d, ...go(m, w.to, SHIFT) })
    }
    for (const [i, m] of marks.current) if (!want.has(i) && !m.leaving) marks.current.set(i, { ...m, ...go(m, [...home(i).slice(0, 3), 0], SHIFT), leaving: true })
    const seen = new Set<string>()
    shown.forEach((l, k) => {
      seen.add(l.key)
      const to = [0, fileH + LEDGER_GAP + k * LINE + LINE / 2, 0, 1], m = labels.current.get(l.key)
      labels.current.set(l.key, { key: l.key, d: 0, text: l.label, ...go(m && !m.leaving ? m : undefined, to, SHIFT, 0, m && !m.leaving ? undefined : [0, to[1], 0, 0]), arc: 0 })
    })
    for (const [k, m] of labels.current) if (!seen.has(k) && !m.leaving) labels.current.set(k, { ...m, ...go(m, [0, m.to[1], 0, 0], SHIFT), leaving: true })
  }, [lines, L]) // eslint-disable-line react-hooks/exhaustive-deps

  // Drawn every frame while anything moves.
  useEffect(() => {
    if (!L) return
    const c = lift.current!, el = box.current!
    const soft = token(el, '--pink-2'), pink = token(el, '--pink'), line = token(el, '--line'), ink = token(el, '--ink-3')
    const font = getComputedStyle(el).fontFamily
    const largest = all.reduce((x, l) => Math.max(x, l.size), MOVED)
    const lit = new Set(rows.map(r => `ngram ${r}`)), dim = (key: string) => (lit.size && !lit.has(key) ? 0.25 : 1)
    const n = weights!.now.length, cw = (cssW - LABEL) / 48
    let raf = 0
    const frame = () => {
      const t = performance.now()
      c.width = Math.round(cssW * dpr)
      c.height = Math.round(height * dpr)
      const ctx = c.getContext('2d')!
      ctx.scale(dpr, dpr)
      // the places in the file that moved
      ctx.fillStyle = rgba(soft, pink, 1, 1)
      for (const l of all) for (const x of l.cells) if (x.i < n) ctx.fillRect(xOf(L, x.i) / dpr, yOf(L, x.i) / dpr, 1 / dpr, 1 / dpr)
      // the lines: label, and a faint slot for each of the line's numbers
      ctx.font = `10px ${font}`
      ctx.textBaseline = 'middle'
      let busy = false
      for (const [k, m] of labels.current) {
        const [, y, , a] = at(m, t)
        if (t < m.t0 + m.dur) busy = true
        else if (m.leaving) { labels.current.delete(k); continue }
        const l = all.find(x => x.key === k), slots = !l ? 0 : l.key.startsWith('ngram') || l.key.startsWith('length') ? 48 : l.key === 'u' ? 8 : Math.min(48, l.cells.length)
        ctx.globalAlpha = a * dim(k)
        ctx.fillStyle = rgba(ink, ink, 0, 1)
        ctx.fillText(m.text.length > 9 ? m.text.slice(0, 8) + '…' : m.text, 0, y)
        ctx.fillStyle = rgba(line, line, 0, 1)
        for (let s = 0; s < slots; s++) ctx.fillRect(LABEL + s * cw, y - (cw - 1) / 2, cw - 1, cw - 1)
      }
      if (more > 0) { ctx.globalAlpha = 1; ctx.fillStyle = rgba(ink, ink, 0, 1); ctx.fillText(`+ ${more} more`, 0, fileH + LEDGER_GAP + shown.length * LINE + LINE / 2) }
      // the moved weights, wherever they are on their way
      for (const [i, m] of marks.current) {
        const [x, y, s, a] = at(m, t)
        if (t < m.t0 + m.dur) busy = true
        else if (m.leaving) { marks.current.delete(i); continue }
        ctx.globalAlpha = a * dim(m.key)
        ctx.fillStyle = rgba(soft, pink, Math.min(1, Math.abs(m.d) / largest), 1)
        ctx.fillRect(x, y, s, s)
      }
      if (busy) raf = requestAnimationFrame(frame)
    }
    frame()
    return () => cancelAnimationFrame(raf)
  }, [lines, L, rows, height]) // eslint-disable-line react-hooks/exhaustive-deps

  // The embedding rows a token reads: copied from the field at full strength and outlined, while the rest dims.
  const linked = head?.name === 'ngram' && rows.length > 0
  useEffect(() => {
    if (!L || !head || !linked) return // the last drawing stays while it fades out
    const c = mark.current!, pad = 2 * dpr
    c.width = L.width
    c.height = L.height
    const ctx = c.getContext('2d')!
    ctx.strokeStyle = getComputedStyle(c).color
    ctx.lineWidth = dpr
    for (const r of rows) {
      const x = xOf(L, r * head.c), y = yOf(L, r * head.c), w = head.c
      ctx.drawImage(base.current!, x, y, w, 1, x, y, w, 1)
      const left = Math.max(dpr / 2, x - pad), right = Math.min(L.width - dpr / 2, x + w + pad) // the outline stays on the canvas
      ctx.strokeRect(left, y - pad, right - left, 1 + 2 * pad)
    }
  }, [rows, L, head, linked, dpr, weights])

  const rest = L && L.height - (Math.ceil(L.split / L.cols) + L.gap)
  const largest = all.reduce((x, l) => Math.max(x, l.size), 0)
  return (
    <div className="insp-field" ref={box} data-linked={linked} data-dim={playing ? 'deep' : count ? '' : undefined}>
      {L && head && weights && <>
        <div className="insp-label"><span>{head.name}</span><span>{num(head.r)} × {head.c}</span></div>
        <div className="insp-stage" style={{ width: cssW, height }}>
          <canvas ref={base} className="insp-base" style={{ height: fileH }} />
          <canvas ref={mark} className="insp-mark" style={{ height: fileH }} />
          <canvas ref={lift} className="insp-lift" />
          <div className="insp-label insp-label-rest" style={{ top: fileH - rest! / dpr }}>
            <span>the rest of the network</span><span>{num(weights.now.length - L.split)}</span>
          </div>
        </div>
        <div className="insp-count">
          <span><b data-any={count > 0}>{num(count)}</b> of {num(weights.now.length)} moved</span>
          {count > 0 && <span className="insp-scale" title="how far a moved weight moved">{MOVED}<i />{largest.toPrecision(2)}</span>}
        </div>
      </>}
    </div>
  )
}

const NONE: Focus = { rows: [] }
const PACE = 300 // ms per step when a solve plays back

export function Inspector({ text, today, summary, run, version, others }: InspectorProps) {
  const ready = summary !== undefined
  const [weights, setWeights] = useState<Weights>()
  const [reading, setReading] = useState<Inspection>()
  const [bytes, setBytes] = useState<number>()
  const [focus, setFocus] = useState(NONE)
  // The latest solve plays back while its sentence is the one inspected.
  const replay = run && run.text === text && run.ref === today && run.frames.length ? run : undefined
  const [at, setAt] = useState(0)
  const [playing, setPlaying] = useState(false)
  useEffect(() => { setAt(0); setPlaying(true) }, [run?.id])
  const last = replay ? replay.frames.length - 1 : 0, more = at < last
  useEffect(() => {
    if (!playing) return
    if (!more) { if (replay?.done) setPlaying(false); return }
    const t = setTimeout(() => setAt(a => a + 1), PACE)
    return () => clearTimeout(t)
  }, [playing, at, more, replay?.done])
  const frame = replay?.frames[Math.min(at, last)]
  const shown = frame?.inspection ?? reading, before = replay && frame?.step ? replay.frames[0].inspection : undefined

  const probes = others.map(o => `${o.text}\n${o.ref}`).join('\n')
  useEffect(() => {
    if (!ready) return
    let live = true
    call('weights').then(w => live && setWeights(w), () => {})
    return () => { live = false }
  }, [ready, version])
  useEffect(() => {
    if (!ready || !text.trim() || replay) return setReading(undefined)
    let live = true
    call('inspect', text, today, others).then(r => live && setReading(r), () => {})
    return () => { live = false }
  }, [ready, text, today, version, probes, !!replay]) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (ready) fetch(import.meta.env.BASE_URL + 'event.bin', { method: 'HEAD' }).then(r => setBytes(Number(r.headers.get('content-length')) || undefined), () => {})
  }, [ready])

  return (
    <section className="inspector" data-ready={ready}>
      <pre className="insp-reveal">
        read = weights(<b>"event.bin"</b>){'\n'}
        <span>{ready ? `// ${bytes ? `${num(Math.round(bytes / 1000))} kB · ` : ''}${num(summary.parameters)} numbers` : '// loading'}</span>
      </pre>
      {ready && <>
        {replay && <Scrub frames={replay.frames} at={Math.min(at, last)} done={replay.done} ms={replay.ms}
          onSeek={i => { setPlaying(false); setAt(i) }} onPlay={() => { setAt(0); setPlaying(true) }} />}
        {shown && <>
          <Flow r={shown} before={before} today={today} onFocus={f => setFocus(f ?? NONE)} />
          <div className="insp-how">{focus.how ?? `${num(shown.programs)} candidate programs`}</div>
          <Probes r={shown} before={before} today={today} />
        </>}
        <Field weights={weights} rows={focus.rows} lines={replay ? frame?.lines : weights?.lines} playing={!!replay && (at < last || !replay.done)} reserve={!!replay} />
      </>}
    </section>
  )
}
