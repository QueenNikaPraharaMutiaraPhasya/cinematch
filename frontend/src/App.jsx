import { useRef, useState } from 'react'
import MovieCard from './components/MovieCard'
import NeighborPanel from './components/NeighborPanel'
import './App.css'

const API_URL = import.meta.env.VITE_API_URL ?? 'http://127.0.0.1:8000'
const MAX_QUERY = 1000
const SHOWN_NEIGHBORS_LIMIT = 10
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
      body: JSON.stringify({ query, k, n, show_neighbors: SHOWN_NEIGHBORS_LIMIT }),
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
  const maxVotes = Math.max(0, ...result.recommendations.map((movie) => movie.votes))
  // Jika daftar memenuhi batas tampilan, total retrieval bisa lebih besar.
  const foundCount = result.k <= SHOWN_NEIGHBORS_LIMIT || result.neighbors.length < SHOWN_NEIGHBORS_LIMIT
    ? result.neighbors.length
    : null
  return (
    <section className="results-section" aria-labelledby="results-title">
      <div className="section-heading"><div><p className="eyebrow">PILIHAN BERDASARKAN PERCAKAPAN</p><h2 id="results-title">Untuk selera filmmu<span className="accent-dot">.</span></h2></div><span className="count-label">{result.recommendations.length} film ditemukan</span></div>
      <p className="query-caption">Hasil untuk <q>{result.query}</q></p>
      <div className="result-facts"><span><strong>{result.k}</strong> tetangga diminta (k)</span><span><strong>{foundCount ?? `Minimal ${result.neighbors.length}`}</strong> tetangga ditemukan</span><span><strong>{result.neighbors.length}</strong> tetangga ditampilkan</span>{result.top_similarity != null && <span><strong className="mono">{result.top_similarity.toFixed(3)}</strong> similarity tertinggi</span>}</div>
      <p className="detail-note result-note">Skor utama adalah jumlah suara tetangga. Panjang batang relatif terhadap suara terbanyak dalam hasil ini. Similarity mengukur kemiripan teks; keduanya bukan probabilitas kecocokan.</p>
      <div className="results-layout">
        {result.recommendations.length ? <ol className="movie-grid">{result.recommendations.map((movie) => <MovieCard key={movie.movie_id} movie={movie} maxVotes={maxVotes} />)}</ol> : <div className="feedback"><span className="feedback-symbol" aria-hidden="true">⌕</span><h3>Belum menemukan rekomendasi</h3><p>Coba jelaskan genre, suasana, atau film yang kamu sukai dalam bahasa Inggris.</p></div>}
        <NeighborPanel neighbors={result.neighbors} k={result.k} foundCount={foundCount} />
      </div>
    </section>
  )
}

export default function App() {
  const [query, setQuery] = useState('')
  const [kText, setKText] = useState('90')
  const [nText, setNText] = useState('10')
  const [phase, setPhase] = useState('idle')
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
      const data = await requestRecommendations({ query: text, k: clampInt(kText, 1, 200, 90), n: clampInt(nText, 1, 50, 10) }, controller.signal)
      if (controller.signal.aborted) return
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

  function selectExample(text) {
    setQuery(text)
    submit(text)
  }

  const loading = phase === 'loading'
  return (
    <div className="app">
      <a className="skip-link" href="#search">Langsung ke pencarian</a>
      <header className="site-header"><a className="brand" href="#" aria-label="CineMatch, kembali ke atas"><span className="brand-mark" aria-hidden="true">C<span /></span>Cine<span>Match</span></a><span className="research-label">PROTOTIPE AKADEMIK <span>NBCRS</span></span></header>
      <main>
        <section className="hero" aria-labelledby="hero-title">
          <div className="hero-copy"><p className="eyebrow"><span className="small-rule" /> REKOMENDASI DARI PERCAKAPAN</p><h1 id="hero-title">Cerita berikutnya,<br /><span>sesuai seleramu.</span></h1><p className="hero-description">Dari preferensi sederhana ke pilihan film berikutnya. CineMatch mencari percakapan serupa, lalu menemukan film yang paling sering disarankan di sana.</p></div>
          <div className="hero-aside" aria-hidden="true"><span className="orbit orbit-one" /><span className="orbit orbit-two" /><span className="orbit-center">C<span className="orbit-dot" /></span><span className="orbit-caption">PREFERENSI → PERCAKAPAN → FILM</span></div>
        </section>
        <section id="search" className="search-panel" aria-labelledby="search-title">
          <div className="search-heading"><div><p className="eyebrow">MULAI DARI SELERAMU</p><h2 id="search-title">Sedang ingin menonton apa?</h2></div><span className="language-tag">Query dalam bahasa Inggris</span></div>
          <form onSubmit={handleSubmit}>
            <label className="sr-only" htmlFor="query">Preferensi film dalam bahasa Inggris</label>
            <div className="query-field"><textarea id="query" rows={3} maxLength={MAX_QUERY} value={query} onChange={(event) => setQuery(event.target.value)} placeholder="I'm looking for a psychological thriller with a plot twist..." aria-describedby="query-help query-count" /><div className="query-actions"><span id="query-count" className="counter">{query.length} / {MAX_QUERY}</span><button type="submit" className="primary" disabled={loading}>{loading ? 'Mencari film…' : 'Temukan film'}<span aria-hidden="true">↗</span></button></div></div>
            <p id="query-help" className="detail-note">Ceritakan genre, suasana, atau film yang kamu sukai. Tekan tombol untuk mencari; Enter membuat baris baru.</p>
            <div className="examples"><span>Coba inspirasi</span>{EXAMPLES.map((text) => <button key={text} type="button" className="chip" onClick={() => selectExample(text)}>{text}<span aria-hidden="true">↗</span></button>)}</div>
            <details className="advanced"><summary>Pengaturan rekomendasi <span aria-hidden="true">+</span></summary><div className="advanced-fields"><label><span>Jumlah tetangga <b>k</b></span><input type="number" min="1" max="200" value={kText} onChange={(event) => setKText(event.target.value)} /><small>1–200 percakapan · bawaan 90</small></label><label><span>Jumlah film <b>N</b></span><input type="number" min="1" max="50" value={nText} onChange={(event) => setNText(event.target.value)} /><small>1–50 film · bawaan 10</small></label></div></details>
          </form>
        </section>
        <div className="result-area" aria-live="polite" aria-busy={loading}>
          {phase === 'idle' && <div className="idle-state"><span className="idle-icon" aria-hidden="true">⌕</span><div><h2>Penemuan film dimulai di sini.</h2><p>Tulis preferensimu atau pilih contoh. Setiap rekomendasi dilengkapi bukti percakapan pendukung.</p></div></div>}
          {loading && <div className="loading-state"><p role="status"><span className="spinner" aria-hidden="true" />Mencari percakapan serupa dan menghitung suara film…</p><div className="skeleton-grid" aria-hidden="true">{[1, 2, 3].map((item) => <div className="skeleton-card" key={item}><span /><span /><span /></div>)}</div></div>}
          {phase === 'error' && <div className="feedback error-state" role="alert"><span className="eyebrow">PENCARIAN BELUM BERHASIL</span><h2>Ada kendala saat mencari film.</h2><p>{error}</p><button className="secondary" type="button" onClick={() => submit(query)}>Coba lagi <span aria-hidden="true">↗</span></button></div>}
          {phase === 'done' && result.status === 'empty_query' && <div className="feedback"><span className="eyebrow">PREFERENSIMU MASIH KOSONG</span><h2>Ceritakan film yang ingin kamu tonton.</h2><p>{result.message}</p><a className="text-link" href="#query">Isi preferensi film ↗</a></div>}
          {phase === 'done' && result.status === 'ok' && <Results result={result} />}
        </div>
        <section className="method-note" aria-label="Cara kerja CineMatch"><span className="eyebrow">CARA KERJANYA</span><p><span>01</span> Preferensi teks <i aria-hidden="true">→</i> <span>02</span> Percakapan serupa <i aria-hidden="true">→</i> <span>03</span> Voting film</p></section>
      </main>
      <footer className="site-footer"><div><span className="footer-brand">CineMatch<span> / Research prototype</span></span><p>Implementasi konsep NBCRS, bukan replikasi penuh paper. Dataset ReDial (Li dkk., NeurIPS 2018), CC BY 4.0.</p></div><p>Hanya query berbahasa Inggris. Sistem mencocokkan kemiripan teks dan menghitung suara; belum memahami syarat seperti “tanpa kekerasan”. Hasil bukan jaminan kecocokan.</p></footer>
    </div>
  )
}
