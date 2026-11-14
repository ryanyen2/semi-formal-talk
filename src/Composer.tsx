// The sentence. Its words are coloured as the model reads them, on a layer under a transparent
// input with the same metrics, so marking never moves the text or the caret.
import { useEffect, useRef } from 'react'
import type { Chunk, Parse } from './protocol.ts'

type Props = { value: string; parse?: Parse; ready: boolean; onChange(text: string): void; onCommit(): void; onEscape(): void }

/**
 * The sentence in words, each with the role of the chunk it starts in. A word keeps its element
 * while the sentence grows, so its colour changes only when the model reads it differently.
 * The parse may be one keystroke behind; words past what it covers are unmarked.
 */
function words(text: string, parse?: Parse) {
  // ponytail: offsets are into the NFKC form the worker reads; text that normalisation lengthens or shortens is marked a few characters off
  const roles: Chunk['role'][] = []
  for (const c of parse?.chunks ?? []) for (let i = 0; i < c.text.length; i++) roles.push(c.role)
  let at = 0
  return text.split(/(\s+)/).map(word => { const role = roles[at]; at += word.length; return { word, role } })
}

export function Composer({ value, parse, ready, onChange, onCommit, onEscape }: Props) {
  const input = useRef<HTMLInputElement>(null), marks = useRef<HTMLDivElement>(null)

  // Typing anywhere goes to the sentence.
  useEffect(() => {
    const key = (e: KeyboardEvent) => {
      if (e.key.length === 1 && !e.metaKey && !e.ctrlKey && !e.altKey && !(e.target instanceof HTMLInputElement)) input.current!.focus()
    }
    addEventListener('keydown', key)
    return () => removeEventListener('keydown', key)
  }, [])
  useEffect(() => { if (ready) input.current!.focus() }, [ready])

  return (
    <div className="composer" data-ready={ready || undefined}>
      <div className="composer-marks" ref={marks} aria-hidden="true">
        {words(value, parse).map((w, i) => <span key={i} data-role={/\S/.test(w.word) ? w.role : undefined}>{w.word}</span>)}
      </div>
      <input ref={input} value={value} disabled={!ready} placeholder="movie night tomorrow 4pm" aria-label="New event"
        spellCheck={false} autoComplete="off" autoCapitalize="off"
        onChange={e => onChange(e.target.value)}
        onScroll={e => { marks.current!.scrollLeft = e.currentTarget.scrollLeft }}
        onKeyDown={e => { if (e.key === 'Enter' && !e.nativeEvent.isComposing) onCommit(); else if (e.key === 'Escape') onEscape() }} />
    </div>
  )
}
