import { useRef, useState } from 'react'
import './App.css'

const API_URL = import.meta.env.VITE_API_URL ?? 'http://127.0.0.1:8000'
const MAX_QUERY = 1000
const EXAMPLES = [
  'I want a psychological thriller with a plot twist',
  'something like Inception',
  'a feel-good romantic comedy for a rainy evening',
]

function clampInt(text, min, max, fallback) {
  const value = Number.parseInt(text, 10)
  if (Number.isNaN(value)) return fallback
  return Math.min(max, Math.max(min, value))
}

function formatDetail(detail) {
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail)) {
    return detail.map((d) => `${(d.loc ?? []).slice(1).join('.')}: ${d.msg}`).join('; ')
  }
  return ''
}

async function requestRecommendations({ query, k, n }, signal) {
  let response
  try {
    response = await fetch(`${API_URL}/recommend`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, k, n, show_neighbors: 10 }),
      signal,
    })
  } catch (err) {
    if (err.name === 'AbortError') throw err
    throw new Error(
      `Tidak dapat terhubung ke backend (${API_URL}). Pastikan server FastAPI sedang berjalan.`,
    )
  }
  if (!response.ok) {
    let detail = ''
    try {
      detail = formatDetail((await response.json()).detail)
    } catch {
      // isi balasan bukan JSON; abaikan
    }
    throw new Error(
      `Server menolak permintaan (HTTP ${response.status})${detail ? `: ${detail}` : ''}`,
    )
  }
  return response.json()
}

function Results({ result }) {
  const { recommendations, neighbors, k, top_similarity: topSimilarity } = result
  if (recommendations.length === 0) {
    return <p className="status">Tidak ada rekomendasi yang dapat dihitung untuk query ini.</p>
  }
  const maxVotes = Math.max(...recommendations.map((r) => r.votes))

  return (
    <div className="results">
      <section aria-label="Rekomendasi film">
        <h2>Top-{recommendations.length} rekomendasi</h2>
        <p className="meta">
          Query: “{result.query}”. Dihitung dari {k} percakapan terdekat.
          {topSimilarity != null && (
            <>
              {' '}
              Kemiripan teks tetangga terdekat: {topSimilarity.toFixed(3)} (hanya kemiripan teks,
              bukan ukuran keyakinan sistem).
            </>
          )}
        </p>
        <p className="meta">Panjang batang relatif terhadap film dengan suara terbanyak.</p>
        <ol className="cards">
          {recommendations.map((r) => (
            <li key={r.movie_id} className="card">
              <div className="rank">{r.rank}</div>
              <div className="card-body">
                <h3>{r.title}</h3>
                <div className="bar" aria-hidden="true">
                  <span style={{ width: `${(r.votes / maxVotes) * 100}%` }} />
                </div>
                <p className="explanation">{r.explanation}</p>
                <details>
                  <summary>Percakapan pendukung (maks. 5 teratas)</summary>
                  <ul>
                    {r.supporters.map((s) => (
                      <li key={s.conversation_id}>
                        percakapan {s.conversation_id} · similarity {s.similarity.toFixed(3)}
                      </li>
                    ))}
                  </ul>
                </details>
              </div>
            </li>
          ))}
        </ol>
      </section>

      <aside className="neighbors" aria-label="Percakapan tetangga">
        <h2>Percakapan tetangga</h2>
        <p className="meta">
          {neighbors.length} teratas dari {k} percakapan yang dipakai untuk voting.
        </p>
        <ol className="neighbor-list">
          {neighbors.map((nb) => (
            <li key={nb.conversation_id}>
              <div className="nb-head">
                #{nb.rank} · similarity {nb.similarity.toFixed(3)} · percakapan {nb.conversation_id}
              </div>
              <p className="nb-context">“{nb.context}”</p>
              <p className="nb-items">Film yang disarankan: {nb.items.join(', ')}</p>
            </li>
          ))}
        </ol>
      </aside>
    </div>
  )
}

export default function App() {
  const [query, setQuery] = useState('')
  const [kText, setKText] = useState('90')
  const [nText, setNText] = useState('10')
  const [phase, setPhase] = useState('idle') // idle | loading | done | error
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const controllerRef = useRef(null)

  async function submit(text) {
    controllerRef.current?.abort()
    const controller = new AbortController()
    controllerRef.current = controller
    setPhase('loading')
    setError('')
    try {
      const data = await requestRecommendations(
        {
          query: text,
          k: clampInt(kText, 1, 200, 90),
          n: clampInt(nText, 1, 50, 10),
        },
        controller.signal,
      )
      setResult(data)
      setPhase('done')
    } catch (err) {
      if (err.name === 'AbortError') return
      setError(err.message)
      setPhase('error')
    }
  }

  function handleSubmit(event) {
    event.preventDefault()
    submit(query)
  }

  function useExample(text) {
    setQuery(text)
    submit(text)
  }

  const loading = phase === 'loading'

  return (
    <div className="app">
      <header className="app-header">
        <h1>CineMatch</h1>
        <p>
          Rekomendasi film berbasis percakapan. Tulis preferensi filmmu dalam bahasa Inggris. Sistem
          mencari percakapan serupa, lalu menghitung film yang paling sering disarankan di sana.
        </p>
      </header>

      <form className="panel" onSubmit={handleSubmit}>
        <label htmlFor="query">Preferensi film (bahasa Inggris)</label>
        <textarea
          id="query"
          rows={3}
          maxLength={MAX_QUERY}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="e.g. I'm looking for a thriller with a plot twist"
        />
        <div className="form-row">
          <button type="submit" className="primary" disabled={loading}>
            {loading ? 'Mencari…' : 'Rekomendasikan'}
          </button>
          <span className="counter">
            {query.length}/{MAX_QUERY}
          </span>
        </div>
        <div className="examples">
          <span>Coba contoh:</span>
          {EXAMPLES.map((text) => (
            <button key={text} type="button" className="chip" onClick={() => useExample(text)}>
              {text}
            </button>
          ))}
        </div>
        <details className="advanced">
          <summary>Pengaturan lanjutan</summary>
          <div className="advanced-fields">
            <label>
              k: jumlah percakapan tetangga (1–200)
              <input
                type="number"
                min="1"
                max="200"
                value={kText}
                onChange={(e) => setKText(e.target.value)}
              />
            </label>
            <label>
              N: jumlah film ditampilkan (1–50)
              <input
                type="number"
                min="1"
                max="50"
                value={nText}
                onChange={(e) => setNText(e.target.value)}
              />
            </label>
          </div>
        </details>
      </form>

      <div aria-live="polite">
        {phase === 'idle' && (
          <p className="hint">Belum ada pencarian. Tulis preferensi atau pilih salah satu contoh.</p>
        )}
        {loading && (
          <p className="status" role="status">
            <span className="spinner" aria-hidden="true" />
            Mencari percakapan serupa…
          </p>
        )}
        {phase === 'error' && (
          <div className="alert" role="alert">
            {error}
          </div>
        )}
        {phase === 'done' && result.status === 'empty_query' && (
          <p className="status">{result.message}</p>
        )}
        {phase === 'done' && result.status === 'ok' && <Results result={result} />}
      </div>

      <footer className="app-footer">
        <p>
          Prototype implementasi konsep NBCRS (bukan replikasi penuh paper). Data: ReDial (Li dkk.,
          NeurIPS 2018), lisensi CC BY 4.0.
        </p>
        <p>
          Keterbatasan: hanya query berbahasa Inggris. Sistem mencocokkan kemiripan teks lalu
          menghitung suara, jadi tidak memahami syarat seperti negasi (misalnya “tanpa kekerasan”).
          Hasil adalah film yang sering disarankan pada percakapan serupa, bukan jaminan kecocokan.
        </p>
      </footer>
    </div>
  )
}