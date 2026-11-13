export default function NeighborPanel({ neighbors, k, foundCount }) {
  return (
    <aside className="evidence-panel" aria-label="Bukti percakapan tetangga">
      <details open>
        <summary className="evidence-heading"><span><span className="eyebrow">DI BALIK REKOMENDASI</span><span className="panel-title">Percakapan tetangga</span></span><span className="disclosure" aria-hidden="true">⌄</span></summary>
        <div className="evidence-body">
          <p className="detail-note">Panel ini memperlihatkan dasar rekomendasi NBCRS: percakapan serupa dan film yang disarankan di dalamnya. Film diurutkan berdasarkan suara dari percakapan tetangga.</p>
          <p className="detail-note neighbor-counts">Diminta: {k}. {foundCount == null ? `Ditemukan: minimal ${neighbors.length}; total tidak dilaporkan karena daftar bukti dibatasi.` : `Ditemukan: ${foundCount}.`} Ditampilkan: {neighbors.length} percakapan.</p>
          {neighbors.length === 0 ? <p className="detail-note">Tidak ada percakapan yang ditampilkan pada respons ini.</p> : <ol className="neighbor-list">{neighbors.map((neighbor) => (
            <li key={neighbor.conversation_id}>
              <div className="neighbor-meta"><span>#{String(neighbor.rank).padStart(2, '0')} · Percakapan {neighbor.conversation_id}</span><span className="similarity mono" aria-label={`Similarity ${neighbor.similarity.toFixed(3)}`}>{neighbor.similarity.toFixed(3)}</span></div>
              <p className="neighbor-context">“{neighbor.context}”</p>
              <p className="neighbor-items"><span>Film saran</span>{neighbor.items.join(' · ')}</p>
            </li>
          ))}</ol>}
          <p className="evidence-footnote">Similarity mengukur kemiripan teks, bukan keyakinan sistem.</p>
        </div>
      </details>
    </aside>
  )
}
