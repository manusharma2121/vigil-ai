import { useEffect, useState, useRef } from 'react'
import { motion } from 'framer-motion'
import {
  Play, Square, Save, RefreshCw, Cpu, AlertTriangle,
  CheckCircle, TrendingUp, Clock, Info
} from 'lucide-react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Legend
} from 'recharts'
import { startTraining, getTrainingStatus, getModelInfo } from '../api/client'

const ARCHITECTURES = [
  { id: 'cnn_lstm',  label: 'CNN + LSTM',  desc: 'MobileNetV2 feature extractor + LSTM temporal head' },
  { id: '3dcnn',    label: '3D-CNN',       desc: 'Lightweight 3D convolutional network' },
]

export default function ModelTraining() {
  const [config, setConfig]       = useState({ architecture: 'cnn_lstm', epochs: 20, batch_size: 8, lr: 0.0001 })
  const [status, setStatus]       = useState(null)
  const [modelInfo, setModelInfo] = useState(null)
  const [history, setHistory]     = useState([])
  const [polling, setPolling]     = useState(false)
  const [error, setError]         = useState(null)
  const pollRef = useRef(null)

  const fetchModel = async () => {
    try {
      const r = await getModelInfo()
      setModelInfo(r.data)
    } catch {}
  }

  const fetchStatus = async () => {
    try {
      const r = await getTrainingStatus()
      const s = r.data
      setStatus(s)
      if (s.history && s.history.length > 0) setHistory(s.history)
      if (s.status === 'done' || s.status === 'error' || s.status === 'idle') {
        clearInterval(pollRef.current)
        setPolling(false)
        fetchModel()
      }
    } catch {}
  }

  useEffect(() => {
    fetchModel()
    fetchStatus()
    return () => clearInterval(pollRef.current)
  }, [])

  const handleStart = async () => {
    setError(null)
    try {
      const res = await startTraining({
        model_arch:    config.architecture,
        epochs:        config.epochs,
        batch_size:    config.batch_size,
        learning_rate: config.lr,
      })
      setPolling(true)
      pollRef.current = setInterval(fetchStatus, 2000)
    } catch (e) {
      setError(e.response?.data?.detail || 'Failed to start training.')
    }
  }

  const handleStop = () => {
    clearInterval(pollRef.current)
    setPolling(false)
  }

  const isRunning = status?.state === 'running'

  return (
    <div className="flex-1 p-6 lg:p-8 overflow-y-auto">
      <div className="max-w-5xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-white mb-1">Model Training</h1>
          <p className="text-slate-400 text-sm">Configure and launch deep learning training runs</p>
        </div>

        <div className="grid lg:grid-cols-3 gap-6 mb-8">
          {/* Config panel */}
          <div className="lg:col-span-1 space-y-4">
            <div className="glass-card p-6">
              <h3 className="text-sm font-semibold text-slate-300 uppercase tracking-widest mb-5">
                Training Configuration
              </h3>

              {/* Architecture */}
              <div className="mb-4">
                <label className="block text-xs text-slate-400 mb-2">Architecture</label>
                {ARCHITECTURES.map(a => (
                  <label key={a.id} className={`flex items-start gap-3 p-3 rounded-xl mb-2 cursor-pointer transition-all ${
                    config.architecture === a.id
                      ? 'border border-blue-500/50 bg-blue-500/10'
                      : 'border border-blue-900/20 hover:border-blue-500/30'
                  }`}>
                    <input type="radio" name="arch" value={a.id}
                      checked={config.architecture === a.id}
                      onChange={e => setConfig(c => ({ ...c, architecture: e.target.value }))}
                      className="mt-0.5" />
                    <div>
                      <div className="text-sm font-semibold text-slate-200">{a.label}</div>
                      <div className="text-xs text-slate-500 mt-0.5">{a.desc}</div>
                    </div>
                  </label>
                ))}
              </div>

              {/* Epochs */}
              <div className="mb-4">
                <label className="block text-xs text-slate-400 mb-1">
                  Epochs <span className="text-blue-400 font-mono">{config.epochs}</span>
                </label>
                <input type="range" min={1} max={100} value={config.epochs}
                  onChange={e => setConfig(c => ({ ...c, epochs: +e.target.value }))}
                  className="w-full accent-blue-500" />
              </div>

              {/* Batch size */}
              <div className="mb-4">
                <label className="block text-xs text-slate-400 mb-1">Batch Size</label>
                <select value={config.batch_size}
                  onChange={e => setConfig(c => ({ ...c, batch_size: +e.target.value }))}
                  className="input-dark">
                  {[4, 8, 16, 32].map(v => <option key={v} value={v}>{v}</option>)}
                </select>
              </div>

              {/* Learning rate */}
              <div className="mb-6">
                <label className="block text-xs text-slate-400 mb-1">Learning Rate</label>
                <select value={config.lr}
                  onChange={e => setConfig(c => ({ ...c, lr: +e.target.value }))}
                  className="input-dark">
                  {[0.001, 0.0001, 0.00001].map(v => (
                    <option key={v} value={v}>{v}</option>
                  ))}
                </select>
              </div>

              {/* Error */}
              {error && (
                <div className="glass-card p-3 mb-4 flex gap-2 items-start"
                     style={{ borderColor: 'rgba(239,68,68,0.3)' }}>
                  <AlertTriangle size={14} className="text-red-400 mt-0.5" />
                  <p className="text-xs text-red-300">{error}</p>
                </div>
              )}

              {/* Controls */}
              <div className="flex gap-2">
                <button onClick={handleStart} id="start-training-btn"
                  disabled={isRunning}
                  className="btn-primary flex-1 justify-center py-3">
                  <Play size={16} /> Start
                </button>
                {isRunning && (
                  <button onClick={handleStop} id="stop-training-btn"
                    className="btn-secondary px-4">
                    <Square size={16} />
                  </button>
                )}
              </div>
            </div>

            {/* Model info */}
            {modelInfo && (
              <div className="glass-card p-5">
                <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-4">
                  Loaded Model
                </h3>
                <div className="space-y-2 text-sm">
                  {[
                    ['Architecture', modelInfo.architecture],
                    ['Parameters',   modelInfo.parameters ? modelInfo.parameters.toLocaleString() : '—'],
                    ['Device',       modelInfo.device],
                    ['Weights',      modelInfo.weights_loaded ? '✅ Loaded' : '⚠ Untrained (demo)'],
                  ].map(([k, v]) => (
                    <div key={k} className="flex justify-between">
                      <span className="text-slate-500">{k}</span>
                      <span className="text-slate-200 font-mono text-xs text-right max-w-[140px] truncate">{v}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Training progress */}
          <div className="lg:col-span-2 space-y-4">
            {/* Status card */}
            <div className="glass-card p-5">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-sm font-semibold text-slate-300 uppercase tracking-widest">
                  Training Status
                </h3>
                <button onClick={fetchStatus} className="btn-secondary text-xs py-1.5 px-3">
                  <RefreshCw size={12} /> Refresh
                </button>
              </div>

              {!status || status.status === 'idle' ? (
                <div className="text-center py-8 text-slate-500">
                  <Cpu size={32} className="mx-auto mb-3 opacity-30" />
                  <p className="text-sm">No training job running. Configure and start above.</p>
                  <p className="text-xs mt-1">
                    Note: Training requires dataset files to be present. See Dataset Explorer.
                  </p>
                </div>
              ) : (
                <div>
                  <div className="flex items-center gap-3 mb-4">
                    {status.status === 'running' ? (
                      <div className="w-5 h-5 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
                    ) : status.status === 'done' ? (
                      <CheckCircle size={20} className="text-green-400" />
                    ) : (
                      <AlertTriangle size={20} className="text-red-400" />
                    )}
                    <span className="text-slate-200 capitalize font-semibold">{status.status}</span>
                    {status.message && (
                      <span className="text-slate-400 text-sm">— {status.message}</span>
                    )}
                  </div>

                  {status.status === 'running' && (
                    <div className="mb-4">
                      <div className="flex justify-between text-xs text-slate-400 mb-1">
                        <span>Epoch {status.current_epoch} / {status.total_epochs}</span>
                        <span>{Math.round((status.current_epoch / status.total_epochs) * 100)}%</span>
                      </div>
                      <div className="progress-bar">
                        <div className="progress-fill"
                          style={{ width: `${(status.current_epoch / status.total_epochs) * 100}%` }} />
                      </div>
                    </div>
                  )}

                  {/* Latest metrics */}
                  {status.latest_metrics && (
                    <div className="grid grid-cols-4 gap-3 mb-4">
                      {[
                        ['Train Loss',  status.latest_metrics.train_loss?.toFixed(4)],
                        ['Val Loss',    status.latest_metrics.val_loss?.toFixed(4)],
                        ['Train Acc',   status.latest_metrics.train_acc ? `${(status.latest_metrics.train_acc*100).toFixed(1)}%` : '—'],
                        ['Val Acc',     status.latest_metrics.val_acc   ? `${(status.latest_metrics.val_acc*100).toFixed(1)}%`   : '—'],
                      ].map(([k, v]) => (
                        <div key={k} className="rounded-lg p-3 text-center"
                             style={{ background: 'rgba(59,130,246,0.05)', border: '1px solid rgba(59,130,246,0.1)' }}>
                          <div className="text-blue-400 font-mono font-bold text-sm">{v ?? '—'}</div>
                          <div className="text-xs text-slate-500 mt-0.5">{k}</div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Loss / Accuracy charts */}
            {history.length > 1 && (
              <>
                <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
                  className="glass-card p-5">
                  <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-4">
                    Loss Curves
                  </h3>
                  <ResponsiveContainer width="100%" height={200}>
                    <LineChart data={history}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="epoch" tick={{ fontSize: 11 }} label={{ value: 'Epoch', position: 'insideBottom', offset: -2, fontSize: 11 }} />
                      <YAxis tick={{ fontSize: 11 }} />
                      <Tooltip />
                      <Legend />
                      <Line type="monotone" dataKey="train_loss" name="Train Loss" stroke="#3b82f6" dot={false} strokeWidth={2} />
                      <Line type="monotone" dataKey="val_loss"   name="Val Loss"   stroke="#8b5cf6" dot={false} strokeWidth={2} />
                    </LineChart>
                  </ResponsiveContainer>
                </motion.div>

                <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.1 }} className="glass-card p-5">
                  <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-4">
                    Accuracy Curves
                  </h3>
                  <ResponsiveContainer width="100%" height={200}>
                    <LineChart data={history}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="epoch" tick={{ fontSize: 11 }} />
                      <YAxis tick={{ fontSize: 11 }} domain={[0, 1]} tickFormatter={v => `${(v*100).toFixed(0)}%`} />
                      <Tooltip formatter={(v) => `${(v*100).toFixed(2)}%`} />
                      <Legend />
                      <Line type="monotone" dataKey="train_acc" name="Train Acc" stroke="#22c55e" dot={false} strokeWidth={2} />
                      <Line type="monotone" dataKey="val_acc"   name="Val Acc"   stroke="#f97316" dot={false} strokeWidth={2} />
                    </LineChart>
                  </ResponsiveContainer>
                </motion.div>
              </>
            )}

            {/* Dataset warning */}
            <div className="glass-card p-4 flex gap-3"
                 style={{ borderColor: 'rgba(234,179,8,0.2)' }}>
              <Info size={16} className="text-yellow-400 flex-shrink-0 mt-0.5" />
              <p className="text-xs text-slate-400 leading-relaxed">
                Training requires the RWF-2000 or Hockey dataset files to be present in the
                <code className="mx-1 text-blue-400">datasets/</code> directory.
                If no dataset is found, training will fail with a clear error.
                Use the Dataset Explorer to verify dataset availability.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
