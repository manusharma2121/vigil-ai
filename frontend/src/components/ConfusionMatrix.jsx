/**
 * ConfusionMatrix Component
 * Renders a 2×2 confusion matrix with colour-coded cells.
 *
 * Props:
 *   matrix: [[TN, FP], [FN, TP]]  — standard scikit-learn format
 */
export default function ConfusionMatrix({ matrix }) {
  if (!matrix || matrix.length < 2) {
    return (
      <div className="glass-card p-6 text-center text-slate-500 text-sm">
        No confusion matrix data available yet.
      </div>
    )
  }

  const [[TN, FP], [FN, TP]] = matrix
  const total = TN + FP + FN + TP || 1

  const cells = [
    { label: 'True Negative',  abbr: 'TN', value: TN,
      desc: 'Correctly predicted Non-Violent',
      style: { background: 'rgba(34,197,94,0.15)', border: '1px solid rgba(34,197,94,0.3)', color: '#4ade80' } },
    { label: 'False Positive', abbr: 'FP', value: FP,
      desc: 'Non-Violent predicted as Violent',
      style: { background: 'rgba(239,68,68,0.15)', border: '1px solid rgba(239,68,68,0.3)', color: '#f87171' } },
    { label: 'False Negative', abbr: 'FN', value: FN,
      desc: 'Violent predicted as Non-Violent',
      style: { background: 'rgba(249,115,22,0.15)', border: '1px solid rgba(249,115,22,0.3)', color: '#fb923c' } },
    { label: 'True Positive',  abbr: 'TP', value: TP,
      desc: 'Correctly predicted Violent',
      style: { background: 'rgba(59,130,246,0.15)', border: '1px solid rgba(59,130,246,0.3)', color: '#60a5fa' } },
  ]

  return (
    <div className="glass-card p-6">
      <h3 className="text-sm font-semibold text-slate-300 mb-4 uppercase tracking-widest">Confusion Matrix</h3>

      {/* Axis labels */}
      <div className="flex gap-2">
        {/* Predicted label (column axis) */}
        <div className="flex flex-col justify-center w-6 mr-1">
          <span className="text-[10px] text-slate-500 -rotate-90 whitespace-nowrap origin-center translate-y-6">
            Actual
          </span>
        </div>

        <div className="flex-1">
          {/* Column headers */}
          <div className="grid grid-cols-2 gap-2 mb-1">
            <div className="text-center text-[10px] text-slate-500 uppercase tracking-wider">Pred. Non-Violent</div>
            <div className="text-center text-[10px] text-slate-500 uppercase tracking-wider">Pred. Violent</div>
          </div>

          {/* Row header + cells */}
          {[0, 1].map(row => (
            <div key={row} className="grid grid-cols-[auto_1fr_1fr] gap-2 mb-2">
              <div className="flex items-center justify-center text-[10px] text-slate-500 uppercase tracking-wider w-16">
                {row === 0 ? 'Non-Viol.' : 'Violent'}
              </div>
              {[0, 1].map(col => {
                const cell = cells[row * 2 + col]
                return (
                  <div key={col} className="rounded-xl p-4 text-center transition-all duration-200 hover:scale-105"
                       style={cell.style}>
                    <div className="text-2xl font-bold font-mono" style={{ color: cell.style.color }}>
                      {cell.value}
                    </div>
                    <div className="text-xs font-semibold mt-1" style={{ color: cell.style.color }}>
                      {cell.abbr}
                    </div>
                    <div className="text-[10px] text-slate-400 mt-0.5">
                      {((cell.value / total) * 100).toFixed(1)}%
                    </div>
                  </div>
                )
              })}
            </div>
          ))}
        </div>
      </div>

      {/* Legend */}
      <div className="grid grid-cols-2 gap-2 mt-4">
        {cells.map(c => (
          <div key={c.abbr} className="text-xs text-slate-500">
            <span className="font-semibold" style={{ color: c.style.color }}>{c.abbr}</span>
            {' '}– {c.desc}
          </div>
        ))}
      </div>
    </div>
  )
}
