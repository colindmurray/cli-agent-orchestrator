import { useEffect, useId, useRef, useState } from 'react'
import { Check, Loader2, Plus, Search, X } from 'lucide-react'

export interface SearchableOption {
  value: string
  label?: string
  description?: string
}

interface SharedProps {
  loadOptions: (query: string) => Promise<SearchableOption[]>
  placeholder: string
  ariaLabel: string
  allowCreate?: boolean
  disabled?: boolean
  emptyMessage?: string
  className?: string
}

function useOptions(open: boolean, query: string, loadOptions: SharedProps['loadOptions']) {
  const [options, setOptions] = useState<SearchableOption[]>([])
  const [loading, setLoading] = useState(false)
  const request = useRef(0)

  useEffect(() => {
    if (!open) return
    const requestId = ++request.current
    const timer = window.setTimeout(async () => {
      setLoading(true)
      try {
        const rows = await loadOptions(query)
        if (request.current === requestId) setOptions(rows)
      } catch {
        if (request.current === requestId) setOptions([])
      } finally {
        if (request.current === requestId) setLoading(false)
      }
    }, 120)
    return () => window.clearTimeout(timer)
  }, [open, query, loadOptions])

  return { options, loading }
}

export function SearchableSelect({
  value,
  onChange,
  loadOptions,
  placeholder,
  ariaLabel,
  allowCreate = false,
  disabled = false,
  emptyMessage = 'No matching options',
  className = '',
}: SharedProps & { value: string; onChange: (value: string) => void }) {
  const [open, setOpen] = useState(false)
  const [query, setQuery] = useState(value)
  const [active, setActive] = useState(0)
  const root = useRef<HTMLDivElement>(null)
  const listId = useId()
  const { options, loading } = useOptions(open, query, loadOptions)
  const candidate = query.trim()
  const canCreate = allowCreate && candidate.length > 0 && !options.some(
    option => option.value.toLocaleLowerCase() === candidate.toLocaleLowerCase(),
  )
  const choices = options.length + (canCreate ? 1 : 0)

  useEffect(() => {
    if (!open) setQuery(value)
  }, [value, open])

  useEffect(() => {
    const close = (event: MouseEvent) => {
      if (root.current && !root.current.contains(event.target as Node)) {
        setOpen(false)
        setQuery(value)
      }
    }
    document.addEventListener('mousedown', close)
    return () => document.removeEventListener('mousedown', close)
  }, [value])

  const choose = (next: string) => {
    onChange(next)
    setQuery(next)
    setOpen(false)
  }

  return (
    <div ref={root} className={`relative ${className}`}>
      <div className="relative">
        <Search size={13} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-gray-600 pointer-events-none" />
        <input
          role="combobox"
          aria-label={ariaLabel}
          aria-expanded={open}
          aria-controls={listId}
          aria-autocomplete="list"
          disabled={disabled}
          value={query}
          onFocus={() => { setOpen(true); setActive(0) }}
          onChange={event => {
            setQuery(event.target.value)
            if (value) onChange('')
            setOpen(true)
            setActive(0)
          }}
          onKeyDown={event => {
            if (event.key === 'Escape') {
              setOpen(false)
              setQuery(value)
            } else if (event.key === 'ArrowDown' && choices) {
              event.preventDefault()
              setActive(index => (index + 1) % choices)
            } else if (event.key === 'ArrowUp' && choices) {
              event.preventDefault()
              setActive(index => (index - 1 + choices) % choices)
            } else if (event.key === 'Enter' && open) {
              event.preventDefault()
              if (active < options.length) choose(options[active].value)
              else if (canCreate) choose(candidate)
            }
          }}
          placeholder={placeholder}
          className="w-full pl-8 pr-8 py-2 rounded bg-gray-950 border border-gray-800 text-sm text-gray-200 placeholder-gray-600 focus:outline-none focus:border-emerald-600/60 disabled:opacity-50"
        />
        {value && (
          <button
            type="button"
            aria-label={`Clear ${ariaLabel}`}
            onClick={() => { onChange(''); setQuery(''); setOpen(true) }}
            className="absolute right-2 top-1/2 -translate-y-1/2 p-1 text-gray-600 hover:text-gray-300"
          >
            <X size={12} />
          </button>
        )}
      </div>
      {open && (
        <div id={listId} role="listbox" className="absolute z-50 mt-1 w-full max-h-64 overflow-y-auto rounded-lg border border-gray-700 bg-gray-950 shadow-xl shadow-black/40">
          {loading && (
            <div className="flex items-center gap-2 px-3 py-3 text-xs text-gray-500">
              <Loader2 size={12} className="animate-spin" /> Searching…
            </div>
          )}
          {!loading && options.map((option, index) => (
            <button
              key={option.value}
              type="button"
              role="option"
              aria-selected={option.value === value}
              onMouseEnter={() => setActive(index)}
              onClick={() => choose(option.value)}
              className={`w-full px-3 py-2 text-left flex items-center gap-2 ${active === index ? 'bg-gray-800' : 'hover:bg-gray-900'}`}
            >
              <span className="min-w-0 flex-1">
                <span className="block truncate text-xs text-gray-200">{option.label ?? option.value}</span>
                {option.description && <span className="block truncate text-[11px] text-gray-500">{option.description}</span>}
              </span>
              {option.value === value && <Check size={13} className="shrink-0 text-emerald-400" />}
            </button>
          ))}
          {!loading && canCreate && (
            <button
              type="button"
              role="option"
              aria-selected={false}
              onMouseEnter={() => setActive(options.length)}
              onClick={() => choose(candidate)}
              className={`w-full px-3 py-2 text-left flex items-center gap-2 text-xs text-emerald-300 ${active === options.length ? 'bg-gray-800' : 'hover:bg-gray-900'}`}
            >
              <Plus size={13} /> Create “{candidate}”
            </button>
          )}
          {!loading && options.length === 0 && !canCreate && (
            <div className="px-3 py-3 text-xs text-gray-500">{emptyMessage}</div>
          )}
        </div>
      )}
    </div>
  )
}

export function SearchableMultiSelect({
  values,
  onChange,
  loadOptions,
  placeholder,
  ariaLabel,
  allowCreate = false,
  disabled = false,
  emptyMessage = 'No matching options',
  className = '',
}: SharedProps & { values: string[]; onChange: (values: string[]) => void }) {
  const [open, setOpen] = useState(false)
  const [query, setQuery] = useState('')
  const [active, setActive] = useState(0)
  const root = useRef<HTMLDivElement>(null)
  const listId = useId()
  const { options: loaded, loading } = useOptions(open, query, loadOptions)
  const options = loaded.filter(option => !values.includes(option.value))
  const candidate = query.trim()
  const canCreate = allowCreate && candidate.length > 0
    && !values.some(value => value.toLocaleLowerCase() === candidate.toLocaleLowerCase())
    && !loaded.some(option => option.value.toLocaleLowerCase() === candidate.toLocaleLowerCase())
  const choices = options.length + (canCreate ? 1 : 0)

  useEffect(() => {
    const close = (event: MouseEvent) => {
      if (root.current && !root.current.contains(event.target as Node)) {
        setOpen(false)
        setQuery('')
      }
    }
    document.addEventListener('mousedown', close)
    return () => document.removeEventListener('mousedown', close)
  }, [])

  const add = (value: string) => {
    if (!values.includes(value)) onChange([...values, value])
    setQuery('')
    setActive(0)
    setOpen(true)
  }

  return (
    <div ref={root} className={`relative ${className}`}>
      <div className="min-h-10 flex flex-wrap items-center gap-1.5 rounded bg-gray-950 border border-gray-800 px-2 py-1.5 focus-within:border-emerald-600/60">
        {values.map(value => (
          <span key={value} className="inline-flex items-center gap-1 rounded bg-emerald-500/10 border border-emerald-500/25 px-2 py-1 text-[11px] text-emerald-300">
            {value}
            <button
              type="button"
              aria-label={`Remove ${value}`}
              onClick={() => onChange(values.filter(item => item !== value))}
              className="text-emerald-500 hover:text-emerald-200"
            >
              <X size={10} />
            </button>
          </span>
        ))}
        <div className="relative min-w-[160px] flex-1">
          <Search size={12} className="absolute left-1 top-1/2 -translate-y-1/2 text-gray-600 pointer-events-none" />
          <input
            role="combobox"
            aria-label={ariaLabel}
            aria-expanded={open}
            aria-controls={listId}
            aria-autocomplete="list"
            disabled={disabled}
            value={query}
            onFocus={() => { setOpen(true); setActive(0) }}
            onChange={event => { setQuery(event.target.value); setOpen(true); setActive(0) }}
            onKeyDown={event => {
              if (event.key === 'Backspace' && !query && values.length) {
                onChange(values.slice(0, -1))
              } else if (event.key === 'Escape') {
                setOpen(false)
                setQuery('')
              } else if (event.key === 'ArrowDown' && choices) {
                event.preventDefault()
                setActive(index => (index + 1) % choices)
              } else if (event.key === 'ArrowUp' && choices) {
                event.preventDefault()
                setActive(index => (index - 1 + choices) % choices)
              } else if (event.key === 'Enter' && open) {
                event.preventDefault()
                if (active < options.length) add(options[active].value)
                else if (canCreate) add(candidate)
              }
            }}
            placeholder={values.length ? 'Add another…' : placeholder}
            className="w-full bg-transparent pl-5 py-1 text-xs text-gray-200 placeholder-gray-600 focus:outline-none disabled:opacity-50"
          />
        </div>
      </div>
      {open && (
        <div id={listId} role="listbox" aria-multiselectable="true" className="absolute z-50 mt-1 w-full max-h-64 overflow-y-auto rounded-lg border border-gray-700 bg-gray-950 shadow-xl shadow-black/40">
          {loading && (
            <div className="flex items-center gap-2 px-3 py-3 text-xs text-gray-500">
              <Loader2 size={12} className="animate-spin" /> Searching…
            </div>
          )}
          {!loading && options.map((option, index) => (
            <button
              key={option.value}
              type="button"
              role="option"
              aria-selected={false}
              onMouseEnter={() => setActive(index)}
              onClick={() => add(option.value)}
              className={`w-full px-3 py-2 text-left ${active === index ? 'bg-gray-800' : 'hover:bg-gray-900'}`}
            >
              <span className="block truncate text-xs text-gray-200">{option.label ?? option.value}</span>
              {option.description && <span className="block truncate text-[11px] text-gray-500">{option.description}</span>}
            </button>
          ))}
          {!loading && canCreate && (
            <button
              type="button"
              role="option"
              aria-selected={false}
              onMouseEnter={() => setActive(options.length)}
              onClick={() => add(candidate)}
              className={`w-full px-3 py-2 text-left flex items-center gap-2 text-xs text-emerald-300 ${active === options.length ? 'bg-gray-800' : 'hover:bg-gray-900'}`}
            >
              <Plus size={13} /> Create “{candidate}”
            </button>
          )}
          {!loading && options.length === 0 && !canCreate && (
            <div className="px-3 py-3 text-xs text-gray-500">{emptyMessage}</div>
          )}
        </div>
      )}
    </div>
  )
}
