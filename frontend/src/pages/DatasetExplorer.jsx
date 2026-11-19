import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import { Database, RefreshCw, CheckCircle, XCircle, AlertTriangle, Download } from 'lucide-react'
import { getDatasetInfo, loadDataset } from '../api/client'

export default function DatasetExplorer() {
  const [datasets, setDatasets] = useState([])
  const [loading, setLoading]   = useState(true)
  const [scanning, setScanning] = useState(false)

  const fetchInfo = async () => {
    try {
      setLoading(true)
      const res = await getDatasetInfo()
      setDatasets(res.data)
    } catch {
      setDatasets([])
    } finally {
      setLoading(false)
    }
  }

  const handleScan = async (name) => {
    setScanning(true)
    try {
      const dataset = name === 'RWF-2000' ? 'rwf2000' : 'hockey'
      await loadDataset(dataset)
      await fetchInfo()
    } finally {
      setScanning(false)
    }
  }

  useEffect(() => { fetchInfo() }, [])

  return (
    <div className="flex-1 p-6 lg:p-8 overflow-y-auto">
      <div className="max-w-5xl mx-auto">
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-3xl font-bold text-white mb-1">Dataset Explorer</h1>
            <p className="text-slate-400 text-sm">Manage and inspect training datasets</p>
          </div>
          <button onClick={fetchInfo} id="refresh-datasets-btn" className="btn-secondary text-sm">
            <RefreshCw size={15} /> Refresh
          </button>
        </div>

        {loading ? (
          <div className="flex items-center justify-center py-20">
            <div className="w-10 h-10 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
          </div>
        ) : (
          <>
            <div className="grid md:grid-cols-2 gap-6 mb-8">
              {datasets.map((ds, i) => (
                <motion.div key={ds.name}
                  initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: i * 0.1 }}
                  className="glass-card p-6"
                  style={{ borderColor: ds.available ? 'rgba(34,197,94,0.3)' : 'rgba(100,116,139,0.2)' }}>
                  <div className="flex items-start justify-between mb-4">
                    <div className="flex items-center gap-3">
                      <Database size={20} className={ds.available ? 'text-green-400' : 'text-slate-500'} />
                      <h3 className="font-bold text-white">{ds.name}</h3>
                    </div>
                    {ds.available
                      ? <span className="badge-nonviolent"><CheckCircle size={12} className="inline mr-1" />Available</span>
                      : <span className="badge-demo"><XCircle size={12} className="inline mr-1" />Not Found</span>
                    }
                  </div>

                  <div className="text-xs font-mono text-slate-500 mb-4 truncate">{ds.path}</div>

                  {ds.available ? (
                    <>
                      <div className="grid grid-cols-2 gap-3 mb-4">
                        {[
                          { label: 'Total Videos', value: ds.total_videos },
                          { label: 'Violent', value: ds.violent_count, color: '#f87171' },
                          { label: 'Non-Violent', value: ds.nonviolent_count, color: '#4ade80' },
                          { label: 'Annotations', value: ds.has_annotations ? 'Yes' : 'No',
                            color: ds.has_annotations ? '#4ade80' : '#94a3b8' },
                        ].map(({ label, value, color }) => (
                          <div key={label} className="rounded-lg p-3"
                               style={{ background: 'rgba(59,130,246,0.05)', border: '1px solid rgba(59,130,246,0.1)' }}>
                            <div className="text-lg font-bold font-mono" style={{ color: color || '#60a5fa' }}>
                              {value}
                            </div>
                            <div className="text-xs text-slate-500">{label}</div>
                          </div>
                        ))}
                      </div>

                      {/* Train/val/test split bars */}
                      <div className="mb-4">
                        <div className="text-xs text-slate-500 mb-2">Data Split (80 / 10 / 10)</div>
                        <div className="flex h-3 rounded-full overflow-hidden gap-0.5">
                          <div style={{ flex: 8, background: '#3b82f6' }} />
                          <div style={{ flex: 1, background: '#8b5cf6' }} />
                          <div style={{ flex: 1, background: '#06b6d4' }} />
                        </div>
                        <div className="flex justify-between text-[10px] text-slate-500 mt-1">
                          <span>Train: {ds.train_count}</span>
                          <span>Val: {ds.val_count}</span>
                          <span>Test: {ds.test_count}</span>
                        </div>
                      </div>
                    </>
                  ) : (
                    <div className="glass-card p-4 mb-4 text-sm text-slate-400">
                      <AlertTriangle size={16} className="text-yellow-400 inline mr-2" />
                      Dataset not found. See README for download and setup instructions.
                    </div>
                  )}

                  <button onClick={() => handleScan(ds.name)} id={`scan-${ds.name.replace(/\s/g, '-')}`}
                    disabled={scanning} className="btn-secondary text-sm w-full justify-center">
                    {scanning ? <div className="w-4 h-4 border-2 border-blue-400/30 border-t-blue-400 rounded-full animate-spin" /> : <RefreshCw size={14} />}
                    Re-scan
                  </button>
                </motion.div>
              ))}
            </div>

            {/* Annotation guide */}
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }} className="glass-card p-6">
              <h3 className="text-sm font-semibold text-slate-300 uppercase tracking-widest mb-4">
                Category Annotation Guide
              </h3>
              <p className="text-sm text-slate-400 mb-4 leading-relaxed">
                To enable category-level false positive analysis, create a CSV file at:
                <code className="ml-2 font-mono text-blue-400 bg-blue-500/10 px-2 py-0.5 rounded">
                  datasets/annotations/category_annotations.csv
                </code>
              </p>
              <div className="rounded-xl overflow-hidden mb-4"
                   style={{ background: 'rgba(5,13,26,0.8)', border: '1px solid rgba(59,130,246,0.15)' }}>
                <div className="px-4 py-2 border-b border-blue-900/30 text-xs text-slate-500 font-mono">
                  CSV Format
                </div>
                <pre className="p-4 text-xs font-mono text-blue-300 overflow-x-auto">{`filename,activity_category,ground_truth,notes
video01.avi,Sports Activities,Non-Violent,Hockey tackle
video02.avi,Hugging and Friendly Physical Contact,Non-Violent,Airport reunion
video03.avi,Crowded Scenes,Non-Violent,Concert crowd`}</pre>
              </div>
              <p className="text-xs text-slate-500">
                Valid categories: <strong className="text-slate-400">Sports Activities</strong>,{' '}
                <strong className="text-slate-400">Hugging and Friendly Physical Contact</strong>,{' '}
                <strong className="text-slate-400">Crowded Scenes</strong>,{' '}
                <strong className="text-slate-400">Dancing and Group Movement</strong>,{' '}
                <strong className="text-slate-400">Normal Physical Interactions</strong>,{' '}
                <strong className="text-slate-400">Other Non-Violent Activities</strong>
              </p>
            </motion.div>
          </>
        )}
      </div>
    </div>
  )
}
