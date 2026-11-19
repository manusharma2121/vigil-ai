import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import {
  History, Trash2, Download, RefreshCw, ChevronDown,
  AlertTriangle, CheckCircle, Search, Filter
} from 'lucide-react'
import { getHistory, deleteHistory, exportHistory } from '../api/client'

function PredictionBadge({ prediction }) {
  return prediction === 'Violent'
    ? <span className="badge-violent">{prediction}</span>
    : <span className="badge-nonviolent">{prediction}</span>
}

export default function PredictionHistory() {
  const [records, setRecords] = useState([])
  const [loading, setLoading] = useState(true)
  const [search, setSearch]   = useState('')
  const [filter, setFilter]   = useState('all')
  const [deleting, setDeleting] = useState(null)
  const [expanded, setExpanded] = useState(null)

  const fetchHistory = async () => {
    try {
      setLoading(true)
      const r = await getHistory()
      setRecords(r.data)
    } catch {
      setRecords([])
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { fetchHistory() }, [])

  const handleDelete = async (id) => {
    setDeleting(id)
    try {
      await deleteHistory(id)
      setRecords(r => r.filter(x => x.id !== id))
    } catch {
      alert('Failed to delete record.')
    } finally {
      setDeleting(null)
    }
  }

  const handleExport = async () => {
    try {
      const r = await exportHistory()
      const url = window.URL.createObjectURL(new Blob([r.data]))
      const a = document.createElement('a')
      a.href = url
      a.download = `vigil_history_${Date.now()}.csv`
      a.click()
    } catch {
      alert('Export failed.')
    }
  }

  const filtered = records
    .filter(r => filter === 'all' || r.prediction === filter)
    .filter(r => r.filename.toLowerCase().includes(search.toLowerCase()))

  return (
    <div className="flex-1 p-6 lg:p-8 overflow-y-auto">
      <div className="max-w-5xl mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-3xl font-bold text-white mb-1">Prediction History</h1>
            <p className="text-slate-400 text-sm">{records.length} total predictions stored</p>
          </div>
          <div className="flex gap-3">
            <button onClick={handleExport} id="export-history-btn" className="btn-secondary text-sm">
              <Download size={15} /> Export CSV
            </button>
            <button onClick={fetchHistory} id="refresh-history-btn" className="btn-secondary text-sm">
              <RefreshCw size={15} /> Refresh
            </button>
          </div>
        </div>

        {/* Filters */}
        <div className="flex flex-col sm:flex-row gap-3 mb-6">
          <div className="relative flex-1">
            <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
            <input
              type="text"
              placeholder="Search by filename…"
              value={search}
              onChange={e => setSearch(e.target.value)}
              className="input-dark pl-9"
            />
          </div>
          <select value={filter} onChange={e => setFilter(e.target.value)} className="input-dark w-auto">
            <option value="all">All Predictions</option>
            <option value="Violent">Violent Only</option>
            <option value="Non-Violent">Non-Violent Only</option>
          </select>
        </div>

        {/* Table */}
        {loading ? (
          <div className="flex items-center justify-center py-20">
            <div className="w-10 h-10 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
          </div>
        ) : filtered.length === 0 ? (
          <div className="glass-card p-12 text-center">
            <History size={40} className="mx-auto mb-4 text-slate-600" />
            <h3 className="text-white font-bold mb-2">No Records Found</h3>
            <p className="text-slate-400 text-sm">
              {records.length === 0
                ? 'Upload and analyse a video to create your first prediction record.'
                : 'No records match the current filter.'}
            </p>
          </div>
        ) : (
          <div className="space-y-2">
            {filtered.map((rec, i) => (
              <motion.div key={rec.id}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.03 }}
                className="glass-card overflow-hidden">
                {/* Row */}
                <div className="flex items-center gap-4 p-4 cursor-pointer"
                     onClick={() => setExpanded(expanded === rec.id ? null : rec.id)}>
                  <div className="flex-1 min-w-0">
                    <div className="font-mono text-sm text-slate-200 truncate">{rec.filename}</div>
                    <div className="text-xs text-slate-500 mt-0.5">
                      {new Date(rec.created_at).toLocaleString()}
                      {rec.is_demo && <span className="badge-demo ml-2">DEMO</span>}
                    </div>
                  </div>

                  <PredictionBadge prediction={rec.prediction} />

                  <div className="text-right hidden sm:block">
                    <div className="text-sm font-mono text-blue-400">
                      {(rec.confidence * 100).toFixed(1)}%
                    </div>
                    <div className="text-xs text-slate-500">confidence</div>
                  </div>

                  <div className="flex items-center gap-2">
                    <button onClick={e => { e.stopPropagation(); handleDelete(rec.id) }}
                      disabled={deleting === rec.id}
                      className="p-2 rounded-lg text-slate-500 hover:text-red-400 hover:bg-red-500/10 transition-all">
                      {deleting === rec.id
                        ? <div className="w-4 h-4 border-2 border-red-400/30 border-t-red-400 rounded-full animate-spin" />
                        : <Trash2 size={15} />}
                    </button>
                    <ChevronDown size={16}
                      className={`text-slate-500 transition-transform ${expanded === rec.id ? 'rotate-180' : ''}`} />
                  </div>
                </div>

                {/* Expanded details */}
                {expanded === rec.id && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: 'auto', opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    className="border-t border-blue-900/20 px-4 pb-4 pt-3"
                  >
                    <div className="grid sm:grid-cols-3 gap-4 text-sm">
                      {[
                        ['Violence Probability', `${(rec.violence_prob * 100).toFixed(2)}%`, '#f87171'],
                        ['Non-Violence Probability', `${(rec.nonviolence_prob * 100).toFixed(2)}%`, '#4ade80'],
                        ['Confidence', `${(rec.confidence * 100).toFixed(2)}%`, '#60a5fa'],
                        ['Processing Time', `${rec.processing_time.toFixed(3)}s`, '#c084fc'],
                        ['Model Used', rec.model_used, '#94a3b8'],
                        ['Ground Truth', rec.ground_truth || 'Not labelled', rec.ground_truth ? '#4ade80' : '#64748b'],
                        ['Category', rec.activity_category || 'Not annotated', '#94a3b8'],
                        ['Demo Mode', rec.is_demo ? 'Yes' : 'No', rec.is_demo ? '#fbbf24' : '#4ade80'],
                      ].map(([k, v, color]) => (
                        <div key={k}>
                          <div className="text-xs text-slate-500 mb-0.5">{k}</div>
                          <div className="font-mono text-sm truncate" style={{ color }}>{v}</div>
                        </div>
                      ))}
                    </div>
                  </motion.div>
                )}
              </motion.div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
