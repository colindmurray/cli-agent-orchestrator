// Safe Markdown/plain-text rendering (design §9, §10).
//
// Every test in this file renders the REAL component with an attack payload
// and asserts on the produced DOM or on spies watching the network surfaces —
// never on the sanitiser helper alone, because a helper returning a safe
// string proves nothing about what the component wired it into.
//
// FIXTURE DISCLOSURE — cond-0477: The communications catalog fixtures in
// sibling test files that carry a bound task_occurrence_id model a state no
// shipped conductor writer currently produces — all current writers record
// task_occurrence_id = NULL (cond-0477). The fork's contract is the published
// index format and a bound occurrence is a legal value of it. The API reports
// `coverage:"complete"`, `total:0` with no reason code for the unbound case,
// so the reader cannot distinguish "unbound" from "genuinely empty" — a known
// limitation that resolves when cond-0477 lands.

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, cleanup, screen, fireEvent, waitFor, act } from '@testing-library/react'
import { SafeContentView, SafeMarkdown } from '../components/SafeContentView'
import * as safeMarkdownLib from '../lib/safeMarkdown'
import {
  isMarkdownMediaType,
  markdownBudgetBreach,
  safeDownloadName,
  safeLinkHref,
  MAX_MARKDOWN_RENDER_BYTES,
  MAX_MARKDOWN_NODES,
} from '../lib/safeMarkdown'

function renderMd(content: string) {
  return render(<SafeMarkdown content={content} />)
}

/** Every element attribute in the container, flattened. */
function allAttrs(container: HTMLElement): string[] {
  const out: string[] = []
  for (const el of Array.from(container.querySelectorAll('*'))) {
    for (const attr of Array.from(el.attributes)) out.push(`${attr.name}=${attr.value}`)
  }
  return out
}

beforeEach(() => {
  vi.spyOn(globalThis, 'fetch')
})

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

describe('GFM still works', () => {
  it('renders a table, a task list, and strikethrough', () => {
    const { container } = renderMd(
      ['| a | b |', '| - | - |', '| 1 | 2 |', '', '- [x] done', '- [ ] todo', '', '~~gone~~'].join('\n'),
    )
    expect(container.querySelector('table')).not.toBeNull()
    expect(container.querySelector('td')!.textContent).toBe('1')
    const boxes = container.querySelectorAll('input[type="checkbox"]')
    expect(boxes).toHaveLength(2)
    expect((boxes[0] as HTMLInputElement).checked).toBe(true)
    expect((boxes[0] as HTMLInputElement).disabled).toBe(true)
    expect(container.querySelector('del')!.textContent).toBe('gone')
  })

  it('renders http and https links with safe window semantics', () => {
    const { container } = renderMd('[a](https://example.com/x) and [b](http://example.com/y)')
    const [a, b] = Array.from(container.querySelectorAll('a'))
    expect(a.getAttribute('href')).toBe('https://example.com/x')
    expect(a.getAttribute('target')).toBe('_blank')
    expect(a.getAttribute('rel')).toBe('noopener noreferrer')
    expect(b.getAttribute('href')).toBe('http://example.com/y')
  })

  it('autolinks a bare https URL and keeps www autolinks on http', () => {
    const { container } = renderMd('<https://example.com> and www.example.com')
    const hrefs = Array.from(container.querySelectorAll('a')).map(a => a.getAttribute('href'))
    expect(hrefs).toContain('https://example.com')
    expect(hrefs).toContain('http://www.example.com')
  })
})

describe('raw HTML executes nothing', () => {
  it('<script> produces no script element; the text is inert or omitted', () => {
    const { container } = renderMd('before\n\n<script>alert(1)</script>\n\nafter')
    expect(container.querySelector('script')).toBeNull()
    expect(allAttrs(container).join(' ')).not.toContain('alert')
    // The surrounding Markdown still rendered.
    expect(container.textContent).toContain('before')
    expect(container.textContent).toContain('after')
  })

  it('<img src=x onerror=alert(1)> produces no img and no event handler', () => {
    const { container } = renderMd('<img src=x onerror=alert(1)>')
    expect(container.querySelector('img')).toBeNull()
    expect(allAttrs(container).some(a => a.startsWith('onerror='))).toBe(false)
  })

  it('an HTML block with an onclick handler produces no element carrying it', () => {
    const { container } = renderMd('<div onclick="alert(1)">hello</div>')
    expect(container.querySelector('div[onclick]')).toBeNull()
    expect(allAttrs(container).some(a => /^on/i.test(a))).toBe(false)
  })

  it('no rendered document ever carries an on* attribute', () => {
    const payloads = [
      '<svg onload=alert(1)>',
      '<a href="https://x" onclick="alert(1)">y</a>',
      '<form action="https://evil.example"><button>go</button></form>',
      '<iframe src="https://evil.example"></iframe>',
    ]
    for (const p of payloads) {
      const { container, unmount } = renderMd(p)
      expect(allAttrs(container).some(a => /^on/i.test(a))).toBe(false)
      expect(container.querySelector('iframe')).toBeNull()
      expect(container.querySelector('form')).toBeNull()
      unmount()
    }
  })
})

describe('URL policy', () => {
  it.each([
    ['javascript:alert(1)'],
    ['JaVaScRiPt:alert(1)'],
    ['java&#115;cript:alert(1)'],
    ['data:text/html,<script>alert(1)</script>'],
    ['vbscript:msgbox(1)'],
    ['file:///etc/passwd'],
    ['//evil.example/protocol-relative'],
    ['../../etc/passwd'],
    ['./sibling.md'],
    ['custom-scheme:do-thing'],
  ])('blocks %s at the DOM, not just in a helper', (url) => {
    const { container } = renderMd(`[click](${url})`)
    expect(container.querySelector('a')).toBeNull()
    expect(allAttrs(container).join(' ')).not.toContain(url.replace(/&#115;/g, 's'))
    // The link text survives as inert, readable, labelled text.
    expect(screen.getByText('click')).toBeInTheDocument()
    expect(container.querySelector('[data-blocked-link="true"]')).not.toBeNull()
  })

  it('blocks a javascript: autolink', () => {
    const { container } = renderMd('<javascript:alert(1)>')
    expect(container.querySelector('a')).toBeNull()
  })

  it('blocks a javascript: reference-style link', () => {
    const { container } = renderMd('[click][x]\n\n[x]: javascript:alert(1)')
    expect(container.querySelector('a')).toBeNull()
    expect(screen.getByText('click')).toBeInTheDocument()
  })
})

describe('images fetch nothing', () => {
  it('a remote image becomes a non-fetching placeholder with its alt text', () => {
    const imageCtor = vi.fn()
    vi.stubGlobal('Image', class {
      constructor() {
        imageCtor()
      }
    })
    const { container } = renderMd('![tracking pixel](https://evil.example/pixel.png)')
    expect(container.querySelector('img')).toBeNull()
    expect(container.querySelector('[src]')).toBeNull()
    expect(allAttrs(container).join(' ')).not.toContain('evil.example')
    const placeholder = screen.getByTestId('md-image-placeholder')
    expect(placeholder).toHaveTextContent('tracking pixel')
    expect(imageCtor).not.toHaveBeenCalled()
    expect(fetch).not.toHaveBeenCalled()
  })

  it('even a data:-URL image is not rendered as an element', () => {
    const { container } = renderMd('![x](data:image/png;base64,AAAA)')
    expect(container.querySelector('img')).toBeNull()
    expect(allAttrs(container).join(' ')).not.toContain('data:image')
  })
})

describe('budgets fail visibly', () => {
  it('a document over the byte budget shows the named state, not an excerpt', () => {
    const content = `word `.repeat(MAX_MARKDOWN_RENDER_BYTES / 5 + 10)
    render(<SafeContentView content={content} mediaType="text/markdown" downloadBase="doc" />)
    const notice = screen.getByTestId('markdown-render-budget')
    expect(notice).toHaveTextContent('Too large to render')
    expect(notice).toHaveTextContent('Raw')
    expect(screen.queryByTestId('md-rendered')).toBeNull()
  })

  it('a marker-dense document trips the node budget under the byte budget', () => {
    // MAX_MARKDOWN_NODES = 10_000. Each `*a* ` contributes ~3 nodes (emphasis
    // + text + paragraph wrapper), so 3*N+1 nodes. Require a margin well over
    // the budget so a future change in the node-per-marker ratio cannot
    // silently flip the test back to a pass. Derived from the constant, not a
    // magic number, and measured locally at ~140 ms vs ~3644 ms for the old
    // 20_000-repeat fixture — well under 500 ms and an order of magnitude of
    // headroom on a 3×-slower runner.
    const repeats = Math.ceil((MAX_MARKDOWN_NODES + 1000) / 3)
    const content = '*a* '.repeat(repeats) // ~14.7 KiB, ~11 002 nodes
    expect(new TextEncoder().encode(content).length).toBeLessThan(MAX_MARKDOWN_RENDER_BYTES)
    expect(markdownBudgetBreach(content)).toBe('nodes')
    render(<SafeContentView content={content} mediaType="text/markdown" downloadBase="doc" />)
    const notice = screen.getByTestId('markdown-render-budget')
    expect(notice).toHaveTextContent('Too complex to render')
    expect(screen.queryByTestId('md-rendered')).toBeNull()
  })

  it('the raw view remains available for an over-budget document, complete', () => {
    const content = `# heading\n\n${'line of text\n'.repeat(50_000)}`
    render(<SafeContentView content={content} mediaType="text/markdown" downloadBase="doc" />)
    expect(screen.getByTestId('markdown-render-budget')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Raw' }))
    const raw = screen.getByTestId('content-raw')
    expect(raw.textContent).toBe(content) // complete, not an excerpt
  })

  it('the breach is computed once per (content, mode), not once per render', async () => {
    vi.useFakeTimers()
    try {
      const spy = vi.spyOn(safeMarkdownLib, 'markdownBudgetBreach')
      const writeText = vi.fn().mockResolvedValue(undefined)
      Object.defineProperty(navigator, 'clipboard', { value: { writeText }, configurable: true })
      const content = '# memo\n\nsome content'
      render(<SafeContentView content={content} mediaType="text/markdown" downloadBase="doc" />)
      expect(spy).toHaveBeenCalledTimes(1)
      // A Copy click re-renders twice — setCopied(true), then the 2s reset —
      // and neither re-render may re-pay the counting parse.
      await act(async () => {
        fireEvent.click(screen.getByTestId('content-copy'))
      })
      expect(screen.getByTestId('content-copy')).toHaveTextContent('Copied')
      act(() => {
        vi.advanceTimersByTime(2100)
      })
      expect(screen.getByTestId('content-copy')).toHaveTextContent('Copy')
      expect(spy).toHaveBeenCalledTimes(1)
    } finally {
      vi.useRealTimers()
    }
  })
})

describe('the counting pass is bounded work', () => {
  // cond-0502: the budget check used to parse the whole document in one call,
  // so a marker-dense document under the byte budget spent ~59 s (512 KiB) —
  // or ~0.9 s for the 64 KiB fixture below — inside the counting parse before
  // the node rail tripped: a silent hang, not the visible failure §10
  // requires. The check now parses capped chunks and checks the rail between
  // chunks, so the remaining bytes are never parsed once the rail trips.
  const denseEmphasis = '*a* '.repeat((64 * 1024) / 4)

  it('a marker-dense document reaches its visible verdict in bounded time', () => {
    expect(new TextEncoder().encode(denseEmphasis).length).toBeLessThan(MAX_MARKDOWN_RENDER_BYTES)
    const t0 = performance.now()
    expect(markdownBudgetBreach(denseEmphasis)).toBe('nodes')
    const elapsed = performance.now() - t0
    // The property under test is BOUNDED VISIBLE FAILURE against the ~59 s
    // monolithic hang, not a latency SLA: measured ~88 ms locally and
    // 338.9 ms on the shared CI runner, so a 5 s bound keeps ~70× of proof
    // that the monolithic parse is gone while leaving an order of magnitude
    // of runner-noise headroom — reverting still goes red by two minutes'
    // worth of timeout long before 5 s matters.
    expect(elapsed).toBeLessThan(5000)
  })

  it('that verdict is visible through the component, not a silent hang', () => {
    render(<SafeContentView content={denseEmphasis} mediaType="text/markdown" downloadBase="doc" />)
    const notice = screen.getByTestId('markdown-render-budget')
    expect(notice).toHaveTextContent('Too complex to render')
    expect(screen.queryByTestId('md-rendered')).toBeNull()
  })

  it('legitimate documents keep their admissions', () => {
    // Table density is what makes these heavy: ~250 nodes/KiB means a table
    // document near 48 KiB trips the NODE RAIL itself (both before and after
    // this change), so the admitted table class sits at 24 KiB. Prose and
    // fenced code are pinned at their design-discussion sizes.
    const table = ['| alpha | beta | gamma |', '| --- | --- | --- |',
      ...Array.from({ length: 40 }, (_, i) => `| r${i}a | r${i}b | r${i}c |`)].join('\n')
    const tables = (table + '\n\n').repeat(Math.ceil((24 * 1024) / (table.length + 2))).slice(0, 24 * 1024)
    const words = 'lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor'.split(' ')
    const sentence = Array.from({ length: 100 }, (_, i) => words[i % words.length]).join(' ')
    const prose = Array.from({ length: 700 }, () => sentence).join('\n\n').slice(0, 400 * 1024)
    const fence = ['```ts', 'function f(x: number): number {', '  return x * 2 + 1;', '}', '```'].join('\n')
    const fences = (fence + '\n\n').repeat(Math.ceil((200 * 1024) / (fence.length + 2))).slice(0, 200 * 1024)
    for (const [name, doc] of [['tables', tables], ['prose', prose], ['fences', fences]] as const) {
      expect(new TextEncoder().encode(doc).length, name).toBeLessThanOrEqual(MAX_MARKDOWN_RENDER_BYTES)
      expect(markdownBudgetBreach(doc), name).toBeNull()
    }
  })

  it('a wide table cut into fragments keeps counting its cells', () => {
    // Reviewer repro (cond-0502 repair cycle 1): one header + delimiter
    // above five ~5,000-character rows of small cells counts 25,014 nodes
    // monolithic. Fragments cut out of that block once lost their
    // header/delimiter context, so each detached row parsed as a ~3-node
    // paragraph — about 40 nodes counted, the rail bypassed, and the render
    // freeze this check exists to prevent recreated inside the renderer.
    // The cutter now re-attaches the block's own header + delimiter to
    // every fragment, so the cells keep counting and the rail trips.
    const header = '| col one | col two |'
    const delimiter = '| --- | --- |'
    const wideRow = '| x'.repeat(1600) // ≈4.8 KB, ~3,200 cells' worth of structure
    const doc = [header, delimiter, ...Array.from({ length: 5 }, () => wideRow)].join('\n')
    expect(new TextEncoder().encode(doc).length).toBeLessThan(MAX_MARKDOWN_RENDER_BYTES)
    expect(markdownBudgetBreach(doc)).toBe('nodes')
  })

  it('a many-row table of normal-length rows stays admitted through fragment cuts', () => {
    // The same re-attachment must not flip legitimate tables: ~300 normal
    // rows form one ~19 KiB block — well over the module's chunk cap, so it
    // IS fragmented — yet every fragment keeps its rows counting as table
    // structure and the total sits far under the rail, exactly as the one
    // monolithic parse counted it.
    const doc = [
      '| a | b | c |',
      '| --- | --- | --- |',
      ...Array.from({ length: 300 }, (_, i) => `| r${i} | value ${i} | note ${i} |`),
    ].join('\n')
    expect(new TextEncoder().encode(doc).length).toBeGreaterThan(2 * 4096)
    expect(markdownBudgetBreach(doc)).toBeNull()
  })

  it('many tiny blocks stay admitted far from the rail', () => {
    // Every blank-line-delimited block parses as its own document and adds
    // one root node: 2,000 one-word paragraphs count 6,000 chunked vs
    // 4,001 monolithic — an inflation of one node per block with the same
    // verdict here, far from the rail. Near the rail that bias can refuse
    // a document one monolithic parse would admit; the module header
    // discloses that arithmetic deliberately rather than pinning it here,
    // where any exact boundary would break on unrelated count changes.
    const doc = Array.from({ length: 2000 }, (_, i) => `w${i}`).join('\n\n')
    expect(markdownBudgetBreach(doc)).toBeNull()
  })

  it('constructs spanning chunk boundaries still count as one document', () => {
    // A loose list spans blank lines: chunked counting sees several short
    // lists where the renderer sees one, and the estimate must stay
    // rail-honest — admitted here, where the monolithic count is also far
    // under the rail.
    const looseList = Array.from({ length: 400 }, (_, i) => `- item ${i} text`).join('\n\n')
    expect(markdownBudgetBreach(looseList)).toBeNull()
    // An unclosed fence swallows every blank line below it into one code
    // block; split per blank line it becomes several small fences, which is
    // the disclosed wrapper-node estimate, never a rail-scale shift.
    const unclosedFence = '```ts\n' + Array.from({ length: 400 }, (_, i) => `let x${i} = ${i}`).join('\n\n')
    expect(markdownBudgetBreach(unclosedFence)).toBeNull()
    // A single line over the chunk cap is hard-sliced mid-line. An emphasis
    // run cut at the slice point degrades to literal text in both pieces, so
    // the summed count stays within a couple of nodes of the monolithic one;
    // the dense variant of the same shape trips the rail above.
    const longLine = 'word **bold** '.repeat(Math.ceil((16 * 1024) / 14)).slice(0, 16 * 1024)
    expect(markdownBudgetBreach(longLine)).toBeNull()
  })
})

describe('plain text stays literal', () => {
  it('Markdown-looking characters in text/plain render as text', () => {
    const content = '# not a heading\n\n**not bold** [not a link](https://example.com)\n\n- not\n- a\n- list'
    render(<SafeContentView content={content} mediaType="text/plain" downloadBase="doc" />)
    const raw = screen.getByTestId('content-raw')
    expect(raw.textContent).toBe(content)
    expect(raw.tagName).toBe('PRE')
    expect(screen.queryByTestId('md-rendered')).toBeNull()
    expect(screen.queryByRole('button', { name: 'Rendered' })).toBeNull()
  })

  it('an unknown media type takes the plain-text path', () => {
    expect(isMarkdownMediaType('application/x-whatever')).toBe(false)
    render(<SafeContentView content={'# x'} mediaType="application/x-whatever" downloadBase="doc" />)
    expect(screen.getByTestId('content-raw')).toBeInTheDocument()
  })

  it('media-type parameters do not confuse the markdown decision', () => {
    expect(isMarkdownMediaType('text/markdown; charset=utf-8')).toBe(true)
    expect(isMarkdownMediaType('Text/Markdown')).toBe(true)
  })
})

describe('copy and download', () => {
  it('copies the exact content', async () => {
    const writeText = vi.fn().mockResolvedValue(undefined)
    Object.defineProperty(navigator, 'clipboard', { value: { writeText }, configurable: true })
    const content = 'exact bytes\nwith  spacing'
    render(<SafeContentView content={content} mediaType="text/plain" downloadBase="doc" />)
    fireEvent.click(screen.getByTestId('content-copy'))
    await waitFor(() => expect(writeText).toHaveBeenCalledWith(content))
  })

  it('downloads a Blob of the exact content under the sanitized display name', async () => {
    const blobs: Blob[] = []
    const clicks: HTMLAnchorElement[] = []
    vi.stubGlobal('URL', {
      ...URL,
      createObjectURL: (blob: Blob) => {
        blobs.push(blob)
        return `blob:${blobs.length}`
      },
      revokeObjectURL: () => {},
    })
    const realClick = HTMLAnchorElement.prototype.click
    HTMLAnchorElement.prototype.click = function () {
      clicks.push(this)
    }
    const content = '# report\n'
    render(
      <SafeContentView content={content} mediaType="text/markdown" downloadBase="communication-c1" displayName="final-report.md" />,
    )
    fireEvent.click(screen.getByTestId('content-download'))
    expect(clicks).toHaveLength(1)
    expect(clicks[0].download).toBe('final-report.md')
    expect(blobs).toHaveLength(1)
    // jsdom Blob predates .text(); FileReader reads the same bytes.
    const text = await new Promise<string>((resolve, reject) => {
      const reader = new FileReader()
      reader.onload = () => resolve(String(reader.result))
      reader.onerror = () => reject(reader.error)
      reader.readAsText(blobs[0])
    })
    expect(text).toBe(content)
    HTMLAnchorElement.prototype.click = realClick
  })
})

describe('helper contracts the DOM tests lean on', () => {
  it('safeLinkHref admits only absolute http(s)', () => {
    expect(safeLinkHref('https://example.com')).toBe('https://example.com')
    expect(safeLinkHref('  https://example.com/x  ')).toBe('https://example.com/x')
    expect(safeLinkHref('javascript:alert(1)')).toBeNull()
    expect(safeLinkHref('java\tscript:alert(1)')).toBeNull()
    expect(safeLinkHref('//evil.example')).toBeNull()
    expect(safeLinkHref('/local/path')).toBeNull()
    expect(safeLinkHref('')).toBeNull()
  })

  it('safeDownloadName strips paths, dotfiles, and controls; falls back by media type', () => {
    expect(safeDownloadName('report.md', 'x', 'text/markdown')).toBe('report.md')
    expect(safeDownloadName('../../etc/passwd', 'x', 'text/plain')).toBe('passwd')
    expect(safeDownloadName('C:\\tmp\\evil.exe', 'x', 'text/plain')).toBe('evil.exe')
    expect(safeDownloadName('..', 'comm-1', 'text/markdown')).toBe('comm-1.md')
    expect(safeDownloadName('', 'comm-1', 'text/plain')).toBe('comm-1.txt')
    expect(safeDownloadName(null, 'att-2', undefined)).toBe('att-2.txt')
    expect(safeDownloadName('a'.repeat(100), 'comm-3', 'text/markdown')).toBe('comm-3.md')
  })
})
