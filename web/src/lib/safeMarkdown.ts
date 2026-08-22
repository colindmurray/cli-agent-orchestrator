// Safe rendering policy for captured communication content (design §9).
//
// Captured bodies and attachments are UNTRUSTED TEXT: authored by agents,
// stored and hashed faithfully, and never executed or interpreted. This module
// is the pure-policy half of the one genuine security boundary in the feature
// (the React half lives in components/SafeContentView.tsx):
//
//   * raw HTML is never re-hydrated (no rehype-raw, no
//     dangerouslySetInnerHTML anywhere in the feature);
//   * links are allow-listed to absolute http(s) — javascript:, data:, file:,
//     vbscript:, custom schemes, protocol-relative //host, and relative
//     filesystem navigation are all refused;
//   * images are replaced with a non-fetching placeholder;
//   * byte and node budgets make an over-budget document FAIL VISIBLY rather
//     than silently truncate. They bound the SIZE of the content admitted and
//     the NODE COUNT of the tree handed to the renderer. The budget CHECK is
//     itself bounded work (cond-0502): the counting pass parses the document
//     in small capped chunks and checks the node rail after each chunk, so a
//     marker-dense document trips the rail after milliseconds-to-tens-of-ms
//     instead of hanging the UI inside one monolithic parse of the whole
//     document (measured on this build: 512 KiB of dense emphasis ≈ 59 s
//     monolithic; the same shape trips the rail in well under 100 ms when
//     chunked). The chunked count is an ESTIMATE of the monolithic tree,
//     biased conservative: every blank-line-delimited block parses as its
//     own document and adds one root node, so many tiny blocks inflate the
//     count by up to one node per block (4,999 one-word paragraphs count
//     14,997 chunked vs 9,999 monolithic — a document near the rail can be
//     refused that one monolithic parse would admit), and fragments cut
//     out of a table block re-attach the block's header and delimiter rows
//     so detached rows keep counting their cells. A mid-line slice through
//     any other construct shifts the count only by that construct's own
//     nodes. The byte ceiling is an owner decision (design §13.3),
//     revisited from measured usage rather than loosened here.

import { unified } from 'unified'
import remarkParse from 'remark-parse'
import remarkGfm from 'remark-gfm'

/**
 * Design §13's Markdown-render budget. Documents larger than this are still
 * viewable raw and downloadable; they are simply never parsed as Markdown.
 */
export const MAX_MARKDOWN_RENDER_BYTES = 512 * 1024

/**
 * A node-count rail for pathological input under the byte budget (deeply
 * nested or marker-dense documents that reconcile far more elements than
 * their byte count suggests). A rail, not a tuning knob: exceeding it fails
 * visibly, exactly like the byte budget.
 */
export const MAX_MARKDOWN_NODES = 10_000

// ASCII whitespace plus every C0 control below it, so a scheme cannot be
// smuggled past the prefix check with an embedded tab or newline.
const WHITESPACE_AND_CONTROLS = /[\u0000-\u0020]+/g

// C0/C1 controls and DEL — never legal in a filename.
const FILENAME_CONTROLS = /[\u0000-\u001f\u007f-\u009f]+/g

/**
 * The href allow-list, applied twice at render time (react-markdown's
 * `urlTransform`, then the anchor component again).
 *
 * The CHECK runs on a compacted copy with ASCII whitespace/control characters
 * removed; the VALUE returned is the original, trimmed — a value that passed
 * the compacted prefix test still begins with http(s) after a browser's own
 * control-character stripping. Anything else — including protocol-relative
 * `//host` and relative paths — returns null, which the renderer draws as
 * inert text.
 */
export function safeLinkHref(raw: string | null | undefined): string | null {
  if (!raw) return null
  const trimmed = raw.trim()
  const compact = trimmed.replace(WHITESPACE_AND_CONTROLS, '')
  const lower = compact.toLowerCase()
  if (lower.startsWith('https://') || lower.startsWith('http://')) return trimmed
  return null
}

/** react-markdown's `urlTransform` hook: the allow-list, and nothing else. */
export function safeUrlTransform(url: string): string {
  return safeLinkHref(url) ?? ''
}

/**
 * Rendered-vs-raw is decided by the document's DECLARED media type, never by
 * sniffing the body: a plain-text report full of Markdown-looking characters
 * must render literally.
 */
export function isMarkdownMediaType(mediaType: string | null | undefined): boolean {
  if (!mediaType) return false
  return mediaType.split(';', 1)[0].trim().toLowerCase() === 'text/markdown'
}

/** Which budget refused the render, or null when rendering is within budget. */
export type MarkdownBudgetBreach = 'bytes' | 'nodes'

// One frozen processor for the counting pass. `.parse` applies remark-gfm's
// micromark/mdast extensions, so the counted tree is the tree react-markdown
// would render; the async `run` transforms are not needed to count nodes.
const countingProcessor = unified().use(remarkParse).use(remarkGfm)

// Chunk cap for the counting parse. Parse cost on marker-dense input grows
// superlinearly in document size (measured on this build: dense emphasis
// costs ~12 ms at 4 KiB, ~62 ms at 16 KiB, ~941 ms at 64 KiB), so small
// chunks are what bound the check's worst case; the rail is checked after
// every chunk and the remaining bytes are never parsed once it trips.
const MAX_COUNTING_CHUNK_BYTES = 4096

/**
 * The pieces the counting pass parses one at a time: blank-line-delimited
 * blocks, with any block over the cap cut again at line ends — and a single
 * line over the cap hard-sliced mid-line.
 *
 * A block whose second line is a GFM delimiter row is a table: its rows
 * count as table structure ONLY while a header + delimiter pair sits above
 * them, so every fragment cut out of such a block re-attaches the block's
 * own first two lines. Without this, a detached row parses as a ~3-node
 * paragraph instead of its row+cell subtree, and a wide-table document
 * counts 40 nodes where the monolithic parse counts 25,014 (measured) —
 * bypassing the rail and recreating inside the renderer the freeze the
 * bounded check exists to prevent. With it, each fragment keeps counting
 * its cells and pays only the small fixed cost of the re-attached pair,
 * so table drift is conservative. Other mid-line slices can split an
 * inline construct, which shifts the count by that construct's own nodes;
 * both drifts are bounded by split frequency, never by document size.
 */
function countingChunks(content: string): string[] {
  const chunks: string[] = []
  for (const block of content.split(/\n[ \t]*\n/)) {
    if (!block) continue
    if (block.length <= MAX_COUNTING_CHUNK_BYTES) {
      chunks.push(block)
      continue
    }
    const lines = block.split('\n')
    // Delimiter-row shape: spaces, pipes, colons, and dashes only, with at
    // least one dash (the lookahead). Loose on purpose — mistaking a
    // thematic break for a delimiter row merely adds a harmless two-line
    // prefix to fragments.
    const contextHead =
      lines.length >= 2 && /^ *(?=[|: -]*-)[|: -]*\r?$/.test(lines[1])
        ? `${lines[0]}\n${lines[1]}`
        : ''
    const tableContext = contextHead ? `${contextHead}\n` : ''
    const pushFragment = (piece: string) => {
      // The block's own first fragment already carries its header +
      // delimiter (it may end exactly at the delimiter's newline); every
      // later fragment starts at a row boundary or mid-cell and needs the
      // pair re-attached.
      chunks.push(!tableContext || piece.startsWith(contextHead) ? piece : tableContext + piece)
    }
    let rest = block
    while (rest.length > MAX_COUNTING_CHUNK_BYTES) {
      const newline = rest.lastIndexOf('\n', MAX_COUNTING_CHUNK_BYTES)
      const cut = newline > 0 ? newline : MAX_COUNTING_CHUNK_BYTES
      pushFragment(rest.slice(0, cut))
      rest = rest.slice(cut).replace(/^\n/, '')
    }
    if (rest) pushFragment(rest)
  }
  return chunks
}

/**
 * The budget check, run BEFORE react-markdown sees the content.
 *
 * Bounded work, not just bounded trees: each chunk is parsed and counted on
 * its own and the rail is checked between chunks, so the verdict for a
 * pathological document arrives after only the first few kilobytes have been
 * parsed. A parse failure here is reported as a breach: the alternative is
 * handing the same input to the renderer and hoping its failure mode is
 * prettier. Failing visibly is the design's whole point (§10).
 */
export function markdownBudgetBreach(content: string): MarkdownBudgetBreach | null {
  if (new TextEncoder().encode(content).length > MAX_MARKDOWN_RENDER_BYTES) return 'bytes'
  try {
    let nodes = 0
    const walk = (node: { children?: unknown[] }) => {
      nodes += 1
      if (nodes > MAX_MARKDOWN_NODES) return
      if (Array.isArray(node.children)) {
        for (const child of node.children) walk(child as { children?: unknown[] })
      }
    }
    for (const chunk of countingChunks(content)) {
      walk(countingProcessor.parse(chunk) as unknown as { children?: unknown[] })
      if (nodes > MAX_MARKDOWN_NODES) return 'nodes'
    }
    return null
  } catch {
    return 'nodes'
  }
}

/**
 * The filename a download may carry. `display_name` is conductor-authored
 * metadata, never a path: separators, control characters, and dotfile forms
 * are stripped, and an empty or over-long result falls back to a generated
 * name whose extension matches the declared media type.
 */
export function safeDownloadName(
  displayName: string | null | undefined,
  fallbackBase: string,
  mediaType: string | null | undefined,
): string {
  const stripped = (displayName ?? '')
    .split(/[\\/]/)
    .pop()!
    .replace(FILENAME_CONTROLS, '')
    .trim()
    .replace(/^\.+/, '')
  const ext = isMarkdownMediaType(mediaType) ? '.md' : '.txt'
  if (!stripped) return `${fallbackBase}${ext}`
  return stripped.length > 64 ? `${fallbackBase}${ext}` : stripped
}
