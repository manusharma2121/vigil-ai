import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import {
  AlertTriangle, PieChart as PieIcon, BarChart2,
  RefreshCw, Download, Info, Filter
} from 'lucide-react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, PieChart, Pie, Cell, Legend
} from 'recharts'
import ConfusionMatrix from '../components/ConfusionMatrix'
import { getFPAnalysis, exportFPReport } from '../api/client'

const COLORS = ['#3b82f6', '#8b5cf6', '#06b6d4', '#f97316', '#22c55e', '#ef4444']

export default function FalsePositiveAnalysis() {
  const [data, setData]       = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError]     = useState(null)

  const fetchData = async () => {
    try {
      setLoading(true)
      const res = await getFPAnalysis()
      setData(res.data)
      setError(null)
    } catch {
      setError('Could not load false positive analysis. Is the backend running?')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { fetchData() }, [])

  if (loading) return (
    <div className="flex-1 flex items-center justify-center">
      <div className="text-center">
        <div className="w-12 h-12 border-2 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
        <p className="text-slate-400">Loading analysis...</p>
      </div>
    </div>
  )

  if (error) return (
    <div className="flex-1 flex items-center justify-center p-8">
      <div className="glass-card p-8 text-center max-w-md">
        <AlertTriangle size={40} className="text-red-400 mx-auto mb-4" />
        <p className="text-slate-400 mb-4">{error}</p>
        <button onClick={fetchData} id="retry-fp" className="btn-primary mx-auto">
          <RefreshCw size={16} /> Retry
        </button>
      </div>
    </div>
  )

  const { total_false_positives, total_evaluated, overall_fp_rate,
          most_misclassified, categories, has_annotations, message } = data

  return (
    <div className="flex-1 p-6 lg:p-8 overflow-y-auto">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-3xl font-bold text-white mb-1">False Positive Analysis</h1>
            <p className="text-slate-400 text-sm">
              Identifying non-violent activities misclassified as violent
            </p>
          </div>
          <div className="flex gap-3">
            <button onClick={exportFPReport} id="export-fp-btn" className="btn-secondary text-sm">
              <Download size={15} /> Export Report
            </button>
            <button onClick={fetchData} id="refresh-fp-btn" className="btn-secondary text-sm">
              <RefreshCw size={15} /> Refresh
            </button>
          </div>
        </div>

        {/* Info message */}
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
          className="glass-card p-4 mb-6 flex gap-3"
          style={{ borderColor: has_annotations ? 'rgba(34,197,94,0.3)' : 'rgba(234,179,8,0.3)' }}>
          <Info size={18} className={has_annotations ? 'text-green-400' : 'text-yellow-400'} />
          <p className="text-sm text-slate-300">{message}</p>
        </motion.div>

        {/* Stat cards */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
          {[
            { label: 'Total Evaluated', value: total_evaluated,                 color: 'blue' },
            { label: 'False Positives', value: total_false_positives,            color: 'red' },
            { label: 'FP Rate',         value: `${(overall_fp_rate*100).toFixed(1)}%`, color: 'orange' },
            { label: 'Top Category',    value: most_misclassified ? '⚠' : '—',  color: 'violet',
              subtitle: most_misclassified || 'N/A' },
          ].map(s => (
            <motion.div key={s.label} whileHover={{ y: -2 }}
              className="glass-card p-5 relative overflow-hidden">
              <div className="text-2xl font-bold font-mono text-white mb-1">{s.value}</div>
              <div className="text-sm text-slate-300">{s.label}</div>
              {s.subtitle && <div className="text-xs text-slate-500 mt-1 truncate">{s.subtitle}</div>}
            </motion.div>
          ))}
        </div>

        {/* Why FP matters */}
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
          className="glass-card p-6 mb-8"
          style={{ borderColor: 'rgba(139,92,246,0.2)' }}>
          <h3 className="text-sm font-semibold text-violet-400 uppercase tracking-widest mb-3">
            Why False Positives Matter
          </h3>
          <div className="grid md:grid-cols-3 gap-4 text-sm text-slate-400">
            <div>
              <strong className="text-slate-200 block mb-1">Safety Systems</strong>
              In real-world surveillance, false positives cause unnecessary alerts, waste responder
              resources, and erode trust in automated systems.
            </div>
            <div>
              <strong className="text-slate-200 block mb-1">Sports & Events</strong>
              Contact sports such as hockey, rugby, and wrestling contain aggressive physical motion
              that visually resembles violence, making them common FP sources.
            </div>
            <div>
              <strong className="text-slate-200 block mb-1">Social Interactions</strong>
              Hugging, dancing, and crowd movement also trigger false alarms because they share motion
              speed and intensity patterns with violent acts.
            </div>
          </div>
        </motion.div>

        {/* Charts */}
        {has_annotations && categories.length > 0 ? (
          <div className="grid lg:grid-cols-2 gap-6 mb-8">
            {/* Bar chart */}
            <motion.div initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }}
              className="glass-card p-6">
              <h3 className="text-sm font-semibold text-slate-300 uppercase tracking-widest mb-6">
                False Positives by Category
              </h3>
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={categories} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" tick={{ fontSize: 11 }} />
                  <YAxis type="category" dataKey="category"
                         width={150} tick={{ fontSize: 10 }} />
                  <Tooltip
                    formatter={(v, n) => [v, n === 'false_positives' ? 'False Positives' : n]}
                  />
                  <Bar dataKey="false_positives" name="False Positives" radius={[0, 4, 4, 0]}>
                    {categories.map((_, i) => (
                      <Cell key={i} fill={COLORS[i % COLORS.length]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </motion.div>

            {/* Pie chart */}
            <motion.div initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }}
              className="glass-card p-6">
              <h3 className="text-sm font-semibold text-slate-300 uppercase tracking-widest mb-6">
                Contribution to Total FPs
              </h3>
              <ResponsiveContainer width="100%" height={260}>
                <PieChart>
                  <Pie data={categories} dataKey="contribution_pct" nameKey="category"
                       cx="50%" cy="50%" outerRadius={90} innerRadius={50}
                       label={({ contribution_pct }) => `${contribution_pct.toFixed(0)}%`}
                       labelLine={false}>
                    {categories.map((_, i) => (
                      <Cell key={i} fill={COLORS[i % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip formatter={(v) => [`${v}%`, 'Contribution']} />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            </motion.div>
          </div>
        ) : (
          <div className="glass-card p-8 text-center mb-8">
            <BarChart2 size={40} className="mx-auto mb-4 text-slate-600" />
            <h3 className="text-white font-bold mb-2">Category Charts Not Available</h3>
            <p className="text-slate-400 text-sm max-w-md mx-auto">
              {total_evaluated === 0
                ? 'No labelled predictions yet. Upload videos and set ground-truth labels via the Detection page.'
                : 'Activity category annotations are required for category charts. See the Dataset Explorer for annotation instructions.'}
            </p>
          </div>
        )}

        {/* Category table */}
        {categories.length > 0 && (
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
            className="glass-card p-6">
            <h3 className="text-sm font-semibold text-slate-300 uppercase tracking-widest mb-6">
              Detailed Category Breakdown
            </h3>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-blue-900/30">
                    {['Category', 'Total Clips', 'False Positives', 'FP Rate', 'Contribution'].map(h => (
                      <th key={h} className="text-left text-xs text-slate-500 uppercase tracking-wider pb-3 pr-4">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {categories.sort((a, b) => b.false_positives - a.false_positives).map((cat, i) => (
                    <tr key={cat.category} className="border-b border-blue-900/10">
                      <td className="py-3 pr-4">
                        <div className="flex items-center gap-2">
                          <div className="w-3 h-3 rounded-full flex-shrink-0"
                               style={{ background: COLORS[i % COLORS.length] }} />
                          <span className="text-slate-200">{cat.category}</span>
                        </div>
                      </td>
                      <td className="py-3 pr-4 font-mono text-slate-300">{cat.total_clips}</td>
                      <td className="py-3 pr-4 font-mono text-red-400">{cat.false_positives}</td>
                      <td className="py-3 pr-4">
                        <div className="flex items-center gap-2">
                          <div className="progress-bar w-20">
                            <div className="progress-fill"
                                 style={{ width: `${cat.fp_rate * 100}%`,
                                          background: COLORS[i % COLORS.length] }} />
                          </div>
                          <span className="font-mono text-xs text-slate-400">
                            {(cat.fp_rate * 100).toFixed(1)}%
                          </span>
                        </div>
                      </td>
                      <td className="py-3 font-mono text-slate-300">{cat.contribution_pct.toFixed(1)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </motion.div>
        )}
      </div>
    </div>
  )
}
