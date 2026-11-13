export default function MovieCard({ movie, maxVotes }) {
  return (
    <li className="movie-card">
      <div className="movie-card-top"><span className="eyebrow">REKOMENDASI</span><span className="movie-rank">#{String(movie.rank).padStart(2, '0')}</span></div>
      <h3>{movie.title}</h3>
      <div className="vote-summary"><strong>{movie.votes}</strong><span>suara tetangga</span></div>
      <div className="vote-bar" aria-hidden="true"><span style={{ width: `${maxVotes ? movie.votes / maxVotes * 100 : 0}%` }} /></div>
      <p className="movie-explanation">Film ini didukung oleh {movie.votes} percakapan tetangga. Setiap percakapan pendukung memberi satu suara.</p>
      <details className="supporters">
        <summary>Lihat bukti pendukung <span aria-hidden="true">↗</span></summary>
        <p className="detail-note">Total similarity pendukung: {movie.sim_sum.toFixed(2)}. Maksimal 5 percakapan ditampilkan.</p>
        <ul>{movie.supporters.map((supporter) => <li key={supporter.conversation_id}><span>Percakapan {supporter.conversation_id}</span><span className="mono" aria-label={`Similarity ${supporter.similarity.toFixed(3)}`}>{supporter.similarity.toFixed(3)}</span></li>)}</ul>
      </details>
    </li>
  )
}
