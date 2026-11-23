import { Fragment, useEffect, useRef, useState } from 'react'
import './App.css'

const API_URL = import.meta.env.VITE_API_URL ?? 'http://127.0.0.1:8000'
const MAX_QUERY = 1000
const DEFAULT_K = 90
const DEFAULT_N = 10
const CONTEXT_PREVIEW = 160
const SHORT_QUERY_WORDS = 3
const EXAMPLES = [
  'I want a psychological thriller with a plot twist',
  'something like Inception',
  'a feel-good romantic comedy for a rainy evening',
]

function validateInt(text, min, max) {
  if (!/^\d+$/.test(text.trim())) return 'Isi dengan bilangan bulat.'
  const value = Number.parseInt(text, 10)
  if (value < min || value > max) return `Harus antara ${min} dan ${max}.`
  return ''
}

function formatDetail(detail) {
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail)) return detail.map((d) => `${(d.loc ?? []).slice(1).join('.')}: ${d.msg}`).join('; ')
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
    throw new Error(`Tidak dapat terhubung ke backend (${API_URL}). Pastikan server FastAPI sedang berjalan.`)
  }
  if (!response.ok) {
    let detail = ''
    try { detail = formatDetail((await response.json()).detail) } catch { /* non-JSON response */ }
    throw new Error(`Server menolak permintaan (HTTP ${response.status})${detail ? `: ${detail}` : ''}`)
  }
  return response.json()
}

function HowItWorks() {
  return (
    <section className="howto section-block" id="how-it-works" aria-label="Cara kerja">
      <div className="section-heading">
        <span className="eyebrow">// SYSTEM PROCESS</span>
        <h2>How CineMatch works<span className="accent">.</span></h2>
        <p>Three steps to discover your next favorite movie.</p>
      </div>
      <div className="steps-grid">
        <article className="step-card"><span className="step-number">01</span><div className="step-icon">⌨</div><h3>DESCRIBE YOUR MOOD</h3><p>Tulis preferensi film dalam bahasa Inggris. Ceritakan genre, suasana, atau film yang kamu sukai.</p></article>
        <article className="step-card"><span className="step-number">02</span><div className="step-icon">⌕</div><h3>FIND SIMILAR CHATS</h3><p>Sistem mencari percakapan terdekat dari dataset ReDial menggunakan kemiripan teks.</p></article>
        <article className="step-card"><span className="step-number">03</span><div className="step-icon">▤</div><h3>GET YOUR MATCH</h3><p>Film yang paling sering direkomendasikan pada percakapan tetangga akan masuk daftar hasil.</p></article>
      </div>
      <p className="hint"><span className="status-dot" /> SYSTEM READY — Belum ada pencarian. Masukkan preferensi untuk memulai.</p>
    </section>
  )
}

function NeighborContext({ text }) {
  const [expanded, setExpanded] = useState(false)
  const long = text.length > CONTEXT_PREVIEW
  const shown = long && !expanded ? `${text.slice(0, CONTEXT_PREVIEW).trimEnd()}…` : text
  return <p className="nb-context">“{shown}”{long && <> <button type="button" className="link-button" onClick={() => setExpanded(!expanded)}>{expanded ? 'ringkas' : 'selengkapnya'}</button></>}</p>
}

function Results({ result }) {
  const { recommendations, neighbors, k, top_similarity: topSimilarity } = result
  if (recommendations.length === 0) return <p className="status">Tidak ada rekomendasi yang dapat dihitung untuk query ini.</p>
  const maxVotes = Math.max(...recommendations.map((r) => r.votes))
  const topTitles = new Set(recommendations.map((r) => r.title))
  return (
    <div className="results" id="results">
      <section className="recommendation-section" aria-label="Rekomendasi film">
        <div className="section-heading compact"><span className="eyebrow">// OUTPUT GENERATED</span><h2>Your movie matches<span className="accent">.</span></h2><p>Top-{recommendations.length} rekomendasi berdasarkan percakapan serupa.</p></div>
        <p className="meta">Query: “{result.query}”. Dihitung dari {k} percakapan terdekat.{topSimilarity != null && <> Kemiripan teks tetangga terdekat: <strong>{topSimilarity.toFixed(3)}</strong> (bukan ukuran keyakinan sistem).</>}</p>
        <ol className="cards">
          {recommendations.map((r) => (
            <li key={r.movie_id} className="movie-card">
              <div className="rank">{String(r.rank).padStart(2, '0')}</div>
              <div className="card-body">
                <div className="card-head"><h3>{r.title}</h3><span className="votes-badge">{r.votes} VOTES</span></div>
                <div className="bar" aria-hidden="true"><span style={{ width: `${(r.votes / maxVotes) * 100}%` }} /></div>
                <p className="explanation">{r.explanation}</p>
                <details><summary>VIEW SUPPORTING CONVERSATIONS (MAX. 5)</summary><ul>{r.supporters.map((s) => <li key={s.conversation_id}>Percakapan {s.conversation_id} · similarity {s.similarity.toFixed(3)}</li>)}</ul></details>
              </div>
            </li>
          ))}
        </ol>
      </section>
      <details className="neighbors" open>
        <summary><span className="eyebrow">// NEIGHBORHOOD DATA</span><h2>Similar conversations</h2></summary>
        <p className="meta">{neighbors.length} percakapan teratas dari {k} tetangga yang dipakai untuk voting. Judul tebal masuk Top-{recommendations.length}.</p>
        <ol className="neighbor-list">{neighbors.map((nb) => <li key={nb.conversation_id}><div className="nb-head">#{nb.rank} · similarity {nb.similarity.toFixed(3)} · conversation {nb.conversation_id}</div><NeighborContext text={nb.context} /><p className="nb-items">Film yang disarankan: {nb.items.map((title, i) => <Fragment key={`${title}-${i}`}>{i > 0 && ', '}{topTitles.has(title) ? <strong className="nb-hit">{title}</strong> : title}</Fragment>)}</p></li>)}</ol>
      </details>
    </div>
  )
}

export default function App() {
  const [query, setQuery] = useState('')
  const [kText, setKText] = useState(String(DEFAULT_K))
  const [nText, setNText] = useState(String(DEFAULT_N))
  const [phase, setPhase] = useState('idle')
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const controllerRef = useRef(null)
  const kError = validateInt(kText, 1, 200)
  const nError = validateInt(nText, 1, 50)
  const hasSettingsError = Boolean(kError || nError)
  const isDefaultSettings = kText === String(DEFAULT_K) && nText === String(DEFAULT_N)
  const wordCount = query.trim() === '' ? 0 : query.trim().split(/\s+/).length
  const isShortQuery = wordCount > 0 && wordCount < SHORT_QUERY_WORDS
  const loading = phase === 'loading'

  async function submit(text) {
    if (!text.trim() || hasSettingsError) return
    controllerRef.current?.abort()
    const controller = new AbortController()
    controllerRef.current = controller
    setPhase('loading'); setError('')
    try {
      const data = await requestRecommendations({ query: text, k: Number.parseInt(kText, 10), n: Number.parseInt(nText, 10) }, controller.signal)
      setResult(data); setPhase('done')
      window.setTimeout(() => document.getElementById('results')?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 80)
    } catch (err) {
      if (err.name === 'AbortError') return
      setError(err.message); setPhase('error')
    }
  }
  function handleSubmit(event) { event.preventDefault(); submit(query) }
  function handleExample(text) { setQuery(text); submit(text) }
  function resetSettings() { setKText(String(DEFAULT_K)); setNText(String(DEFAULT_N)) }

  return (
    <div className="site-shell" id="home">
      <div className="scanlines" aria-hidden="true" />
      <header className="topbar">
        <a className="brand" href="#home"><span className="brand-mark">C</span><span>CINE<span className="accent">MATCH</span><small>MOVIE DISCOVERY SYSTEM</small></span></a>
        <nav className="main-nav" aria-label="Navigasi utama"><a href="#home">HOME</a><a href="#how-it-works">HOW IT WORKS</a><a href="#discover">DISCOVER</a></nav>
        <span className="online-label"><i /> SYSTEM ONLINE</span>
      </header>

      <main>
        <section className="hero">
          <div className="hero-copy"><span className="eyebrow">[ CONVERSATIONAL MOVIE RECOMMENDER ]</span><h1>FIND YOUR NEXT<br /><span className="hero-highlight">MOVIE MATCH</span><span className="cursor">_</span></h1><p>Not sure what to watch? Describe the movie experience you want. Our neighborhood-based recommendation system will find your match.</p><a className="hero-cta" href="#discover">START DISCOVERING <span>↘</span></a></div>
          <div className="hero-art" aria-label="Ilustrasi layar bioskop retro"><div className="orbit orbit-one"/><div className="orbit orbit-two"/><div className="film-frame"><div className="film-screen"><span className="screen-star">✦</span><span className="screen-play">▶</span><span className="screen-caption">YOUR NEXT<br />FAVORITE FILM</span></div><div className="film-holes"><i/><i/><i/><i/><i/><i/></div></div><span className="art-tag tag-top">FIG. 01 / CINEMATCH</span><span className="art-tag tag-bottom">RECOMMENDATION ENGINE</span></div>
          <div className="hero-index">01 <span>/ 03</span></div>
        </section>

        <section className="stats-strip" aria-label="Informasi sistem"><div><strong>9,566</strong><span>TRAIN CONVERSATIONS</span></div><div><strong>90</strong><span>DEFAULT NEIGHBORS (K)</span></div><div><strong>10</strong><span>DEFAULT MOVIE RESULTS (N)</span></div><div><strong>NB<span className="accent">CRS</span></strong><span>RECOMMENDATION METHOD</span></div></section>

        <section className="discover-section section-block" id="discover">
          <div className="section-heading"><span className="eyebrow">// INTERACTIVE TERMINAL — 001</span><h2>Let's find your <span className="accent">match.</span></h2><p>Describe your ideal movie in English. Be as specific as you like.</p></div>
          <form className="terminal-panel" onSubmit={handleSubmit}>
            <div className="terminal-top"><div className="terminal-lights"><i/><i/><i/></div><span>CINEMATCH_TERMINAL.EXE</span><span className="terminal-version">V 1.0.0</span></div>
            <div className="terminal-content">
              <label htmlFor="query" className="field-label"><span className="prompt-symbol">&gt;_</span> MOVIE PREFERENCE INPUT <span className="lang-badge">ENGLISH ONLY</span></label>
              <textarea id="query" rows={4} maxLength={MAX_QUERY} value={query} onChange={(e) => setQuery(e.target.value)} placeholder="e.g. I'm looking for a thriller with a plot twist..." aria-describedby={isShortQuery ? 'short-query-hint' : undefined} />
              {isShortQuery && <p id="short-query-hint" className="field-hint">Query terlalu pendek (kurang dari {SHORT_QUERY_WORDS} kata). Tambahkan genre, suasana, atau film acuan.</p>}
              <div className="input-bottom"><span className="counter">{String(query.length).padStart(4, '0')} / {MAX_QUERY} CHARACTERS</span><button type="submit" className="primary" disabled={loading || hasSettingsError || !query.trim()}>{loading ? 'SEARCHING...' : 'RUN RECOMMENDATION'} <span>→</span></button></div>
              <div className="examples"><span className="examples-label">QUICK PROMPTS //</span>{EXAMPLES.map((text) => <button key={text} type="button" className="chip" disabled={loading || hasSettingsError} onClick={() => handleExample(text)}>{text}</button>)}</div>
              <details className="advanced"><summary>ADVANCED CONFIGURATION <span>＋</span></summary><div className="advanced-fields"><label>k: NEIGHBOR COUNT (1–200)<input type="number" min="1" max="200" value={kText} onChange={(e) => setKText(e.target.value)} aria-invalid={kError ? 'true' : undefined}/>{kError && <span className="field-error">{kError}</span>}</label><label>N: MOVIE RESULTS (1–50)<input type="number" min="1" max="50" value={nText} onChange={(e) => setNText(e.target.value)} aria-invalid={nError ? 'true' : undefined}/>{nError && <span className="field-error">{nError}</span>}</label><button type="button" className="link-button reset" onClick={resetSettings} disabled={isDefaultSettings}>RESET DEFAULTS (K=90, N=10)</button></div></details>
            </div>
            <div className="terminal-bottom"><span><i/> READY TO PROCESS</span><span>POWERED BY NBCRS</span></div>
          </form>
          <div aria-live="polite" className="live-output">
            {phase === 'idle' && <HowItWorks/>}
            {loading && <p className="status"><span className="spinner"/> SCANNING NEIGHBORHOOD DATABASE...</p>}
            {phase === 'error' && <div className="alert" role="alert"><strong>! SYSTEM ERROR</strong><br/>{error}</div>}
            {phase === 'done' && result.status === 'empty_query' && <p className="status">{result.message}</p>}
            {phase === 'done' && result.status === 'ok' && <Results result={result}/>}
          </div>
        </section>

        <section className="method-section section-block"><div className="section-heading"><span className="eyebrow">// UNDER THE HOOD</span><h2>Powered by <span className="accent">NBCRS.</span></h2><p>A lightweight conversational recommendation prototype.</p></div><div className="method-grid"><article><span>01 / RETRIEVAL</span><h3>Semantic similarity</h3><p>Query direpresentasikan sebagai sentence embedding, lalu dibandingkan dengan konteks percakapan ReDial.</p></article><article><span>02 / NEIGHBORHOOD</span><h3>Top-K neighbors</h3><p>Sistem mengambil sejumlah percakapan dengan kemiripan tertinggi sebagai tetangga terdekat.</p></article><article><span>03 / VOTING</span><h3>Community picks</h3><p>Film diberi skor berdasarkan jumlah kemunculannya dalam percakapan tetangga, lalu diurutkan menjadi Top-N.</p></article></div></section>
      </main>

      <footer className="app-footer"><a className="footer-brand" href="#home">CINE<span className="accent">MATCH</span></a><p>CONVERSATIONAL MOVIE RECOMMENDATION SYSTEM · NBCRS PROTOTYPE</p><details><summary>ABOUT & LIMITATIONS</summary><p>Prototype ini hanya menerima query bahasa Inggris. Sistem mencocokkan kemiripan teks lalu menghitung suara, sehingga belum memahami konteks negasi secara penuh. Hasil bukan jaminan kecocokan film.</p></details><span className="copyright">© 2026 CINEMATCH / GROUP 8</span></footer>
    </div>
  )
}
