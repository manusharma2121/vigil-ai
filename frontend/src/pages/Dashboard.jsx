import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import {
  LayoutDashboard, Video, AlertTriangle, CheckCircle,
  TrendingUp, Clock, RefreshCw, Download
} from 'lucide-react'
import {
  PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis,
  CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from 'recharts'
import StatCard from '../components/StatCard'
import { getDashboardStats, exportHistory } from '../api/client'

const COLORS = { Violent: '#ef4444', 'Non-Violent': '#22c55e' }

function PredictionBadge({ prediction }) {
  return prediction === 'Violent'
    ? <span className="badge-violent">{prediction}</span>
    : <span className="badge-nonviolent">{prediction}</span>
}

export default function Dashboard() {
  const [stats, setStats] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [lastRefresh, setLastRefresh] = useState(new Date())

  const fetchStats = async () => {
    try {
      setLoading(true)
      const res = await getDashboardStats()
      setStats(res.data)
      setLastRefresh(new Date())
      setError(null)
    } catch (e) {
      setError('Could not connect to backend. Make sure the FastAPI server is running.')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { fetchStats() }, [])

  if (loading) return (
    <div className="flex-1 flex items-center justify-center">
      <div className="text-center">
        <div className="w-12 h-12 border-2 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
        <p className="text-slate-400">Loading dashboard...</p>
      </div>
    </div>
  )

  if (error) return (
    <div className="flex-1 flex items-center justify-center p-8">
      <div className="glass-card p-8 text-center max-w-md">
        <AlertTriangle size={40} className="text-red-400 mx-auto mb-4" />
        <h3 className="text-white font-bold mb-2">Backend Unavailable</h3>
        <p className="text-slate-400 text-sm mb-6">{error}</p>
        <button onClick={fetchStats} className="btn-primary mx-auto" id="retry-dashboard">
          <RefreshCw size={16} /> Retry
        </button>
      </div>
    </div>
  )

  const { total_videos, violent_count, nonviolent_count, false_positive_count,
          avg_confidence, recent_predictions, confidence_distribution,
          prediction_distribution } = stats

  return (
    <div className="flex-1 p-6 lg:p-8 overflow-y-auto">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold text-white mb-1">Dashboard</h1>
          <p className="text-slate-400 text-sm">
            Last updated: {lastRefresh.toLocaleTimeString()}
          </p>
        </div>
        <div className="flex gap-3">
          <button onClick={exportHistory} id="export-history-btn" className="btn-secondary text-sm">
            <Download size={15} /> Export CSV
          </button>
          <button onClick={fetchStats} id="refresh-dashboard-btn" className="btn-secondary text-sm">
            <RefreshCw size={15} /> Refresh
          </button>
        </div>
      </div>

      {/* Stat Cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        <StatCard icon={Video}         label="Total Analysed"    value={total_videos}         color="blue"   />
        <StatCard icon={AlertTriangle} label="Violent Videos"    value={violent_count}        color="red"    />
        <StatCard icon={CheckCircle}   label="Non-Violent"       value={nonviolent_count}     color="green"  />
        <StatCard icon={TrendingUp}    label="False Positives"   value={false_positive_count} color="orange"
          subtitle={false_positive_count > 0 ? 'Requires ground-truth labels' : 'No labelled data yet'} />
      </div>

      {/* Charts row */}
      <div className="grid lg:grid-cols-2 gap-6 mb-8">
        {/* Distribution pie */}
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="glass-card p-6">
          <h3 className="text-sm font-semibold text-slate-300 uppercase tracking-widest mb-6">
            Prediction Distribution
          </h3>
          {total_videos === 0 ? (
            <div className="h-48 flex items-center justify-center text-slate-500 text-sm">
              No predictions yet. Upload a video to begin.
            </div>
          ) : (
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={prediction_distribution} cx="50%" cy="50%"
                     outerRadius={80} innerRadius={45}
                     dataKey="value" nameKey="name"
                     label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                     labelLine={false}>
                  {prediction_distribution.map((entry) => (
                    <Cell key={entry.name} fill={COLORS[entry.name] || '#6366f1'} />
                  ))}
                </Pie>
                <Tooltip />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          )}
        </motion.div>

        {/* Confidence distribution bar */}
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.1 }} className="glass-card p-6">
          <h3 className="text-sm font-semibold text-slate-300 uppercase tracking-widest mb-6">
            Confidence Distribution
          </h3>
          {total_videos === 0 ? (
            <div className="h-48 flex items-center justify-center text-slate-500 text-sm">
              No data yet.
            </div>
          ) : (
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={confidence_distribution}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="range" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="count" name="Videos" fill="#3b82f6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          )}
        </motion.div>
      </div>

      {/* Recent predictions table */}
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.2 }} className="glass-card p-6">
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-sm font-semibold text-slate-300 uppercase tracking-widest">
            Recent Predictions
          </h3>
          <span className="text-xs text-slate-500">Last 10 results</span>
        </div>

        {recent_predictions.length === 0 ? (
          <div className="text-center py-12 text-slate-500">
            <Video size={32} className="mx-auto mb-3 opacity-30" />
            <p>No predictions yet. Upload a video clip on the Detection page.</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-blue-900/30">
                  {['File', 'Prediction', 'Confidence', 'Violence %', 'Time', 'Date'].map(h => (
                    <th key={h} className="text-left text-xs text-slate-500 uppercase tracking-wider pb-3 pr-4">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {recent_predictions.map((p, i) => (
                  <tr key={p.id}
                      className="border-b border-blue-900/10 hover:bg-blue-500/5 transition-colors">
                    <td className="py-3 pr-4">
                      <div className="font-mono text-xs text-slate-300 max-w-[160px] truncate">{p.filename}</div>
                      {p.is_demo && <span className="badge-demo mt-1 inline-block">DEMO</span>}
                    </td>
                    <td className="py-3 pr-4"><PredictionBadge prediction={p.prediction} /></td>
                    <td className="py-3 pr-4">
                      <div className="progress-bar w-24">
                        <div className="progress-fill" style={{ width: `${(p.confidence * 100).toFixed(0)}%` }} />
                      </div>
                      <span className="text-xs text-slate-400">{(p.confidence * 100).toFixed(1)}%</span>
                    </td>
                    <td className="py-3 pr-4 font-mono text-red-400">{(p.violence_prob * 100).toFixed(1)}%</td>
                    <td className="py-3 pr-4 text-slate-400 font-mono text-xs">{p.processing_time.toFixed(2)}s</td>
                    <td className="py-3 text-slate-500 text-xs">
                      {new Date(p.created_at).toLocaleString()}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </motion.div>

      {/* Avg confidence */}
      {total_videos > 0 && (
        <div className="mt-4 glass-card p-4 flex items-center gap-4">
          <Clock size={16} className="text-slate-400" />
          <span className="text-sm text-slate-400">
            Average confidence across all predictions:
            <span className="text-blue-400 font-mono font-bold ml-2">
              {(avg_confidence * 100).toFixed(1)}%
            </span>
          </span>
        </div>
      )}
    </div>
  )
}
