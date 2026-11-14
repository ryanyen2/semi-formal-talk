// A checkpoint is a manifest (names and shapes, in order) plus one Float32Array.
import { T } from './tensor.ts'

export type Manifest = { name: string; r: number; c: number }[]

export function pack(ts: Record<string, T>): { manifest: Manifest; data: Float32Array } {
  const manifest = Object.entries(ts).map(([name, t]) => ({ name, r: t.r, c: t.c }))
  const data = new Float32Array(manifest.reduce((s, m) => s + m.r * m.c, 0))
  let off = 0
  for (const t of Object.values(ts)) { data.set(t.d, off); off += t.d.length }
  return { manifest, data }
}

export function unpack(manifest: Manifest, data: Float32Array): Record<string, T> {
  const out: Record<string, T> = {}
  let off = 0
  for (const m of manifest) {
    const t = new T(m.r, m.c, data.slice(off, off + m.r * m.c))
    t.name = m.name
    out[m.name] = t
    off += m.r * m.c
  }
  return out
}
