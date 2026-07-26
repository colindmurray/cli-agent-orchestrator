import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, cleanup, fireEvent, screen, waitFor } from '@testing-library/react'
import { TerminalView } from '../components/TerminalView'

// xterm renders nothing meaningful under jsdom, so replace it with a minimal
// fake whose `element` is a real DOM node. TerminalView dispatches synthetic
// `wheel` events onto that element, which is exactly the boundary we assert on.
// Defined via vi.hoisted so the (hoisted) vi.mock factory below can reference it.
const { wheelEvents, termRegistry, FakeTerminal } = vi.hoisted(() => {
  const wheelEvents: WheelEvent[] = []
  const termRegistry: { current: FakeTerminal | null } = { current: null }

  class FakeTerminal {
    rows = 24
    cols = 80
    element: HTMLDivElement
    dataHandler: ((data: string) => void) | null = null

    constructor(_opts: unknown) {
      this.element = document.createElement('div')
      this.element.addEventListener('wheel', (e) => {
        wheelEvents.push(e as WheelEvent)
        this.dataHandler?.('\x1b[<64;1;1M')
      })
      termRegistry.current = this
    }

    loadAddon() {}
    open(parent: HTMLElement) {
      parent.appendChild(this.element)
    }
    onData(handler: (data: string) => void) {
      this.dataHandler = handler
    }
    emitData(data: string) {
      this.dataHandler?.(data)
    }
    onSelectionChange() {}
    attachCustomKeyEventHandler() {}
    getSelection() {
      return ''
    }
    focus() {}
    write() {}
    dispose() {}
  }

  return { wheelEvents, termRegistry, FakeTerminal }
})

vi.mock('@xterm/xterm/css/xterm.css', () => ({}))
vi.mock('@xterm/xterm', () => ({ Terminal: FakeTerminal }))
vi.mock('@xterm/addon-fit', () => ({
  FitAddon: class {
    fit() {}
  },
}))

// TerminalView opens a WebSocket and observes resizes; jsdom has neither.
class FakeWebSocket {
  static OPEN = 1
  static instances: FakeWebSocket[] = []
  readyState = FakeWebSocket.OPEN
  binaryType = ''
  sent: string[] = []
  onopen: (() => void) | null = null
  onmessage: (() => void) | null = null
  onclose: (() => void) | null = null
  constructor(public url: string) {
    FakeWebSocket.instances.push(this)
  }
  send(data: string) {
    this.sent.push(data)
  }
  close() {}
}

class FakeResizeObserver {
  observe() {}
  disconnect() {}
}

// jsdom lacks TouchEvent/Touch; craft a plain event carrying the `touches`
// shape the handlers read.
function touch(type: string, clientYs: number[]): Event {
  const ev = new Event(type, { bubbles: true, cancelable: true })
  Object.defineProperty(ev, 'touches', {
    value: clientYs.map((clientY) => ({ clientY })),
    configurable: true,
  })
  return ev
}

function terminalContainer(): HTMLElement {
  // FakeTerminal.open() appends its element to the ref'd container div.
  const el = termRegistry.current?.element.parentElement
  if (!el) throw new Error('terminal container not mounted')
  return el
}

// Under jsdom there is no layout, so el.clientHeight is 0 and rowHeight() falls
// back to DEFAULT_LINE_HEIGHT = round(TERMINAL_FONT_SIZE * 1.2) = round(14 * 1.2)
// = 17px. Every notch below therefore corresponds to 17px of finger travel.
const LINE_H = 17

describe('TerminalView touch scrolling', () => {
  beforeEach(() => {
    wheelEvents.length = 0
    termRegistry.current = null
    FakeWebSocket.instances.length = 0
    ;(globalThis as unknown as { WebSocket: unknown }).WebSocket = FakeWebSocket
    ;(globalThis as unknown as { ResizeObserver: unknown }).ResizeObserver = FakeResizeObserver
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({ managed: false }),
      }),
    )
  })

  afterEach(() => {
    cleanup()
    vi.unstubAllGlobals()
  })

  it('emits exactly one line-mode notch per row of travel', () => {
    render(<TerminalView terminalId="t-1" onClose={() => {}} />)
    const el = terminalContainer()

    // Swipe up by exactly 3 rows of travel.
    el.dispatchEvent(touch('touchstart', [200]))
    el.dispatchEvent(touch('touchmove', [200 - 3 * LINE_H]))

    expect(wheelEvents).toHaveLength(3)
    // Line mode (bypasses xterm 6's trackpad damping), one line per notch.
    expect(wheelEvents.every((e) => e.deltaMode === WheelEvent.DOM_DELTA_LINE)).toBe(true)
    expect(wheelEvents.every((e) => Math.abs(e.deltaY) === 1)).toBe(true)
    // Finger up scrolls toward newer output → positive deltaY.
    expect(wheelEvents.every((e) => e.deltaY > 0)).toBe(true)
  })

  it('scrolls the opposite direction when the finger moves down', () => {
    render(<TerminalView terminalId="t-1" onClose={() => {}} />)
    const el = terminalContainer()

    el.dispatchEvent(touch('touchstart', [100]))
    el.dispatchEvent(touch('touchmove', [100 + 3 * LINE_H])) // finger down 3 rows

    expect(wheelEvents).toHaveLength(3)
    expect(wheelEvents.every((e) => e.deltaMode === WheelEvent.DOM_DELTA_LINE)).toBe(true)
    expect(wheelEvents.every((e) => e.deltaY === -1)).toBe(true)
  })

  it('accumulates sub-row moves and emits one notch when they cross a row', () => {
    render(<TerminalView terminalId="t-1" onClose={() => {}} />)
    const el = terminalContainer()

    // Two moves of ~half a row each: neither alone crosses a row, together they do.
    const half = Math.ceil(LINE_H / 2) // 9px
    el.dispatchEvent(touch('touchstart', [200]))

    el.dispatchEvent(touch('touchmove', [200 - half])) // 9px < 17px
    expect(wheelEvents).toHaveLength(0)

    el.dispatchEvent(touch('touchmove', [200 - 2 * half])) // 18px total ≥ 17px
    expect(wheelEvents).toHaveLength(1)
    expect(wheelEvents[0].deltaMode).toBe(WheelEvent.DOM_DELTA_LINE)
    expect(wheelEvents[0].deltaY).toBe(1)
  })

  it('preventsDefault on touchmove so the page does not pan/rubber-band', () => {
    render(<TerminalView terminalId="t-1" onClose={() => {}} />)
    const el = terminalContainer()

    el.dispatchEvent(touch('touchstart', [200]))
    const move = touch('touchmove', [100])
    el.dispatchEvent(move)

    expect(move.defaultPrevented).toBe(true)
  })

  it('ignores multi-finger gestures', () => {
    render(<TerminalView terminalId="t-1" onClose={() => {}} />)
    const el = terminalContainer()

    el.dispatchEvent(touch('touchstart', [200, 220]))
    el.dispatchEvent(touch('touchmove', [100, 120]))

    expect(wheelEvents.length).toBe(0)
  })

  it('resets on touchcancel so a stray move after it does not scroll', () => {
    render(<TerminalView terminalId="t-1" onClose={() => {}} />)
    const el = terminalContainer()

    el.dispatchEvent(touch('touchstart', [200]))
    el.dispatchEvent(touch('touchmove', [200 - 3 * LINE_H]))
    expect(wheelEvents.length).toBe(3)

    // Gesture cancelled (e.g. palm rejection): state must reset.
    el.dispatchEvent(touch('touchcancel', []))
    wheelEvents.length = 0

    // A move with no fresh touchstart is ignored (touchY is null again).
    el.dispatchEvent(touch('touchmove', [100]))
    expect(wheelEvents.length).toBe(0)
  })

  it('stops dispatching after unmount (listeners cleaned up)', () => {
    const { unmount } = render(<TerminalView terminalId="t-1" onClose={() => {}} />)
    const el = terminalContainer()

    unmount()
    wheelEvents.length = 0
    el.dispatchEvent(touch('touchstart', [200]))
    el.dispatchEvent(touch('touchmove', [100]))

    expect(wheelEvents.length).toBe(0)
  })

  it('forwards only wheel-originated mouse reports for a managed TUI', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({
          managed: true,
          generation: 'generation-1',
          execution_mode: 'native_tui',
        }),
      }),
    )
    render(<TerminalView terminalId="t-native" onClose={() => {}} />)

    await screen.findByText('Managed native TUI · identity-bound controls')
    const socket = FakeWebSocket.instances[0]
    const terminal = termRegistry.current
    if (!socket || !terminal) throw new Error('terminal did not mount')

    terminal.emitData('ordinary keyboard input')
    expect(socket.sent).toHaveLength(0)

    terminal.element.dispatchEvent(
      new WheelEvent('wheel', {
        deltaY: 1,
        deltaMode: WheelEvent.DOM_DELTA_LINE,
        bubbles: true,
        cancelable: true,
      }),
    )
    expect(socket.sent).toEqual([
      JSON.stringify({ type: 'input', data: '\x1b[<64;1;1M' }),
    ])

    await Promise.resolve()
    terminal.emitData('\x1b[<64;1;1M')
    expect(socket.sent).toHaveLength(1)
  })

  it('routes native Send and Compact through identity-bound control input', async () => {
    const requests: Array<{ url: string; body?: Record<string, unknown> }> = []
    let controlNumber = 0
    vi.stubGlobal('crypto', {
      randomUUID: () => `control-${++controlNumber}`,
    })
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: string | URL | Request, init?: RequestInit) => {
        const url = String(input)
        const body = typeof init?.body === 'string' ? JSON.parse(init.body) : undefined
        requests.push({ url, body })
        let response: Record<string, unknown>
        if (url.endsWith('/managed-control')) {
          response = {
            managed: true,
            generation: 'generation-1',
            execution_mode: 'native_tui',
          }
        } else if (url.endsWith('/control-identity')) {
          response = {
            terminal_id: 't-native',
            terminal_incarnation: 'incarnation-1',
            terminal_generation: 'generation-1',
            pane_birth_id: '%7',
            provider_process_id: '42@start',
            provider: 'kimi_cli',
            native_session_id: 'session-1',
            execution_mode: 'native_tui',
            session_name: 'cao-test',
            pane: { pane_id: '%7' },
          }
        } else {
          response = {
            control_id: body?.control_id,
            outcome: 'success',
          }
        }
        return {
          ok: true,
          json: async () => response,
        } as Response
      }),
    )

    render(<TerminalView terminalId="t-native" onClose={() => {}} />)

    expect(
      await screen.findByText('Managed native TUI · identity-bound controls'),
    ).toBeTruthy()
    expect(screen.queryByRole('button', { name: 'Cancel turn' })).toBeNull()

    fireEvent.change(
      screen.getByPlaceholderText('Send literal text to the native composer…'),
      { target: { value: 'continue the review' } },
    )
    fireEvent.click(screen.getByRole('button', { name: 'Send' }))

    await waitFor(() => {
      const control = requests.find(request => request.url.endsWith('/control-input'))
      expect(control?.body?.text).toBe('continue the review')
      expect(control?.body?.enter).toBe(true)
      expect(control?.body?.expected_identity).toEqual({
        terminal_id: 't-native',
        terminal_incarnation: 'incarnation-1',
        terminal_generation: 'generation-1',
        pane_birth_id: '%7',
        provider_process_id: '42@start',
        provider: 'kimi_cli',
        native_session_id: 'session-1',
        execution_mode: 'native_tui',
        session_name: 'cao-test',
      })
    })

    fireEvent.click(screen.getByRole('button', { name: 'Compact' }))
    await waitFor(() => {
      const controls = requests.filter(request => request.url.endsWith('/control-input'))
      expect(controls).toHaveLength(2)
      expect(controls[1].body?.text).toBe('/compact')
      expect(controls[1].body?.enter).toBe(true)
    })
  })
})
