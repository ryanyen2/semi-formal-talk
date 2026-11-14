// The page's side of the worker that holds the weights. Everything shown is what it returned.
import type { Step } from './core/teach.ts'
import type { Api, Event, Frame } from './protocol.ts'

const worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' })
let seq = 0
const pending = new Map<number, { ok: (v: unknown) => void; no: (e: Error) => void }>()
const listeners = new Set<(s: Step) => void>(), watchers = new Set<(f: Frame) => void>()
worker.onmessage = (e: MessageEvent<Event | { id: number; result?: unknown; error?: string }>) => {
  if ('event' in e.data) {
    const ev = e.data
    return ev.event === 'step' ? listeners.forEach(l => l(ev.step)) : watchers.forEach(w => w(ev.frame))
  }
  const p = pending.get(e.data.id)
  pending.delete(e.data.id)
  if (e.data.error) p?.no(new Error(e.data.error))
  else p?.ok(e.data.result)
}

export function call<M extends keyof Api>(method: M, ...args: Parameters<Api[M]>): Promise<ReturnType<Api[M]>> {
  return new Promise((ok, no) => {
    pending.set(++seq, { ok: ok as (v: unknown) => void, no })
    worker.postMessage({ id: seq, method, args })
  })
}

/** Every optimiser step of a solve, as it happens. Returns the unsubscribe. */
export function onStep(listener: (s: Step) => void): () => void {
  listeners.add(listener)
  return () => listeners.delete(listener)
}

/** The frames of the sentence a teach or forget is about, one per step. Returns the unsubscribe. */
export function onFrame(watcher: (f: Frame) => void): () => void {
  watchers.add(watcher)
  return () => watchers.delete(watcher)
}
