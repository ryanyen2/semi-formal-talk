// What the page and the model worker say to each other.
import type { Manifest } from './core/checkpoint.ts'
import type { Alias } from './core/lexicon.ts'
import type { Step } from './core/teach.ts'

/** Where an event sits: a day, and minutes of that day. start null: all day. */
export type Slot = { date: string; start: number | null; end: number | null }
/** One reading of a sentence, with its probability and the programs that compute it. */
export type Guess = Slot & { p: number; how: { date: string; start: string; end: string | null } }
/** A run of the sentence and what the reading uses it for; none: nothing depends on it. */
export type Chunk = { text: string; role: 'title' | 'date' | 'time' | 'none' }
export type Parse = { text: string; title: string; chunks: Chunk[]; guesses: Guess[] } // guesses: most probable first
export type Correction = Slot & { text: string; ref: string }
export type Standing = Correction & { held: boolean } // held: the sentence reads as corrected under the weights in force
export type Summary = {
  corrections: Standing[]
  parameters: number
  changed: number // weights that differ from the shipped ones
  table: number   // the seed convention nearest to u: index into ALL_CONVENTIONS
}
/** Which parts of a correction some program computes. The others cannot be learned from it. */
export type Learnable = { date: boolean; start: boolean; end: boolean }
/** aliases: unknown words this solve took to mean known ones (core/lexicon.ts). */
export type Solved = { summary: Summary; trace: Step[]; ms: number; aliases: Alias[] }
/**
 * The weights 0.01 or more from the shipped ones, grouped by what they are: a row of ngram (labelled
 * with the word pieces of the corrected sentences that read it), a column of time.words (a length),
 * another tensor, or u. Per cell: i, its index in the file (u's eight after the last tensor); slot, its
 * place on the line (a row's or a column's own index; packed for another tensor); d, how far it moved.
 * size: the line's largest |d|. Lines come largest first.
 */
export type Line = { key: string; label: string; cells: { i: number; slot: number; d: number }[]; size: number }
/** Every weight, in manifest order, now and as shipped; u: the convention vector, now and as shipped; lines: the moved ones. */
export type Weights = { manifest: Manifest; now: Float32Array; shipped: Float32Array; u: Float32Array; u0: Float32Array; lines: Line[] }
/**
 * A word as the network reads it: the mean of its pieces' ngram rows, now and as shipped.
 * known: the training sentences wrote it. near: the other training word whose numbers are most alike now (cosine).
 */
export type Vector = { word: string; known: boolean; pieces: string[]; rows: number[]; now: Float32Array; shipped: Float32Array; near: { word: string; sim: number; now: Float32Array } }
/**
 * One value a part can read as. Per token: read, the probability of the programs that compute this
 * value and take the token's number; sway, how much this value's probability falls when the word is
 * left out of the sentence (words only; 0 when it rises).
 */
export type Option = { value: number; label: string; p: number; how: string; read: number[]; sway: number[] }
/** A sentence read beside the inspected one: a variant of it (mark: the characters changed), or another event. */
export type Probe = { text: string; ref: string; mark?: [number, number] }
/** Where a probe reads now, and the probability of that reading. */
export type Placed = Slot & { p: number }
export type Inspection = {
  tokens: { text: string; role: Chunk['role']; rows: number[] }[] // rows: the embedding rows the token reads (none for a number)
  date: Option[]
  start: Option[]
  end: Option[]
  programs: number
  probes: Probe[]
  placed: Placed[]
}
/** The inspected sentence and its probes at one step of a solve; no step: the weights before it. of: the corrections taught. */
export type Frame = { step?: Step; of: number; inspection: Inspection; lines: Line[] }

export type Api = {
  init(): Summary
  parse(text: string, ref: string): Parse
  teach(c: Correction, others?: Probe[]): Solved & { learnable: Learnable }
  forget(c: Correction, others?: Probe[]): Solved
  weights(): Weights
  inspect(text: string, ref: string, others?: Probe[]): Inspection
  vectors(words: string[]): Vector[]
}
/** While a teach or forget runs: every step, and the frames of the sentence it is about. */
export type Event = { event: 'step'; step: Step } | { event: 'frame'; frame: Frame }
