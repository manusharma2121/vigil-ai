import { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Upload, Video, Shield, AlertTriangle, CheckCircle,
  Clock, Cpu, Zap, Camera, Info, X
} from 'lucide-react'
import { predictVideo, annotateVideo } from '../api/client'

const CATEGORIES = [
  'Sports Activities',
  'Hugging and Friendly Physical Contact',
  'Crowded Scenes',
  'Dancing and Group Movement',
  'Normal Physical Interactions',
  'Other Non-Violent Activities',
]

function ConfidenceBar({ label, value, color }) {
  return (
    <div className="mb-3">
      <div className="flex justify-between text-sm mb-1">
        <span className="text-slate-300">{label}</span>
        <span className="font-mono" style={{ color }}>{(value * 100).toFixed(2)}%</span>
      </div>
      <div className="h-3 rounded-full overflow-hidden" style={{ background: 'rgba(255,255,255,0.05)' }}>
        <motion.div initial={{ width: 0 }} animate={{ width: `${value * 100}%` }}
          transition={{ duration: 0.8, ease: 'easeOut' }}
          className="h-full rounded-full" style={{ background: color }} />
      </div>
    </div>
  )
}

export default function VideoDetection() {
  const [file, setFile]           = useState(null)
  const [preview, setPreview]     = useState(null)
  const [result, setResult]       = useState(null)
  const [loading, setLoading]     = useState(false)
  const [uploadPct, setUploadPct] = useState(0)
  const [error, setError]         = useState(null)
  const [annotating, setAnnotating] = useState(false)
  const [annotation, setAnnotation] = useState({ gt: '', cat: '' })
  const [annotated, setAnnotated] = useState(false)

  const onDrop = useCallback(acceptedFiles => {
    if (acceptedFiles.length === 0) return
    const f = acceptedFiles[0]
    setFile(f)
    setPreview(URL.createObjectURL(f))
    setResult(null)
    setError(null)
    setAnnotated(false)
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'video/*': ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv'] },
    maxFiles: 1,
    multiple: false,
  })

  const handleAnalyse = async () => {
    if (!file) return
    setLoading(true)
    setError(null)
    setUploadPct(0)
    const fd = new FormData()
    fd.append('file', file)
    try {
      const res = await predictVideo(fd, e => {
        if (e.total) setUploadPct(Math.round((e.loaded / e.total) * 100))
      })
      setResult(res.data)
    } catch (e) {
      setError(e.response?.data?.detail || 'An error occurred during analysis. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  const handleAnnotate = async () => {
    if (!result || !annotation.gt) return
    try {
      await annotateVideo({
        prediction_id: result.id,
        activity_category: annotation.cat || 'Other Non-Violent Activities',
        ground_truth: annotation.gt,
      })
      setAnnotated(true)
      setAnnotating(false)
    } catch {
      alert('Failed to save annotation. Please try again.')
    }
  }

  const isViolent = result?.prediction === 'Violent'

  return (
    <div className="flex-1 p-6 lg:p-8 overflow-y-auto">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-white mb-1">Video Detection</h1>
          <p className="text-slate-400">Upload a short video clip for violence classification.</p>
        </div>

        <div className="grid lg:grid-cols-2 gap-6">
          {/* Upload panel */}
          <div>
            {/* Dropzone */}
            <div {...getRootProps()} id="video-dropzone"
              className={`glass-card p-8 text-center cursor-pointer transition-all duration-200 mb-4 ${
                isDragActive ? 'border-blue-500 shadow-glow-blue' : 'hover:border-blue-500/40'
              }`}>
              <input {...getInputProps()} id="video-file-input" />
              <Upload size={40} className={`mx-auto mb-4 ${
                isDragActive ? 'text-blue-400' : 'text-slate-500'
              }`} />
              {isDragActive ? (
                <p className="text-blue-400 font-semibold">Drop the video here...</p>
              ) : (
                <>
                  <p className="text-slate-300 font-medium mb-2">Drag & drop a video file</p>
                  <p className="text-slate-500 text-sm mb-4">or click to browse</p>
                  <span className="text-xs text-slate-600 font-mono">
                    MP4, AVI, MOV, MKV, WEBM · Max 100MB
                  </span>
                </>
              )}
            </div>

            {/* File info */}
            {file && (
              <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
                className="glass-card p-4 mb-4 flex items-center gap-3">
                <Video size={18} className="text-blue-400 flex-shrink-0" />
                <div className="flex-1 min-w-0">
                  <div className="font-mono text-sm text-slate-200 truncate">{file.name}</div>
                  <div className="text-xs text-slate-500">
                    {(file.size / 1024 / 1024).toFixed(2)} MB
                  </div>
                </div>
                <button onClick={() => { setFile(null); setPreview(null); setResult(null) }}
                  className="text-slate-500 hover:text-red-400 transition-colors">
                  <X size={16} />
                </button>
              </motion.div>
            )}

            {/* Upload progress */}
            {loading && uploadPct > 0 && uploadPct < 100 && (
              <div className="mb-4">
                <div className="flex justify-between text-xs text-slate-400 mb-1">
                  <span>Uploading...</span><span>{uploadPct}%</span>
                </div>
                <div className="progress-bar">
                  <div className="progress-fill" style={{ width: `${uploadPct}%` }} />
                </div>
              </div>
            )}
            {loading && uploadPct >= 100 && (
              <div className="glass-card p-4 mb-4 flex items-center gap-3">
                <div className="w-5 h-5 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
                <span className="text-slate-300 text-sm">Analysing video with AI model...</span>
              </div>
            )}

            {/* Error */}
            {error && (
              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                className="glass-card p-4 mb-4 flex items-center gap-3"
                style={{ borderColor: 'rgba(239,68,68,0.3)' }}>
                <AlertTriangle size={18} className="text-red-400" />
                <p className="text-sm text-red-300">{error}</p>
              </motion.div>
            )}

            {/* Analyse button */}
            <button onClick={handleAnalyse} id="analyse-video-btn"
              disabled={!file || loading}
              className="btn-primary w-full justify-center py-4 text-base">
              {loading
                ? <><div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" /> Processing...</>
                : <><Zap size={18} /> Analyse Video</>
              }
            </button>

            {/* Explainability note */}
            <div className="mt-4 glass-card p-4 flex gap-3">
              <Info size={16} className="text-blue-400 flex-shrink-0 mt-0.5" />
              <p className="text-xs text-slate-500 leading-relaxed">
                16 frames are uniformly sampled from the video and processed by the CNN+LSTM model.
                If running in demo mode (no trained weights), predictions are from an untrained network
                and are labelled accordingly.
              </p>
            </div>
          </div>

          {/* Preview + Result panel */}
          <div>
            {preview && (
              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="mb-4">
                <video src={preview} controls className="w-full rounded-2xl" id="video-preview"
                       style={{ maxHeight: 280, background: '#000' }} />
              </motion.div>
            )}

            {!preview && (
              <div className="glass-card mb-4 flex items-center justify-center rounded-2xl"
                   style={{ height: 200 }}>
                <div className="text-center text-slate-600">
                  <Camera size={40} className="mx-auto mb-2 opacity-30" />
                  <p className="text-sm">Video preview will appear here</p>
                </div>
              </div>
            )}

            {/* Result */}
            <AnimatePresence>
              {result && (
                <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0 }}
                            className="glass-card p-6"
                            style={{ borderColor: isViolent ? 'rgba(239,68,68,0.4)' : 'rgba(34,197,94,0.4)' }}>
                  {/* Result header */}
                  <div className="flex items-center gap-3 mb-6">
                    {isViolent
                      ? <AlertTriangle size={28} className="text-red-400" />
                      : <CheckCircle size={28} className="text-green-400" />}
                    <div>
                      <div className="text-2xl font-bold" style={{ color: isViolent ? '#f87171' : '#4ade80' }}>
                        {result.prediction}
                      </div>
                      {result.is_demo && (
                        <span className="badge-demo text-xs">DEMO MODE – untrained model</span>
                      )}
                    </div>
                  </div>

                  {/* Confidence bars */}
                  <ConfidenceBar label="Violence Probability"     value={result.violence_prob}     color="#ef4444" />
                  <ConfidenceBar label="Non-Violence Probability" value={result.nonviolence_prob}  color="#22c55e" />
                  <ConfidenceBar label="Confidence Score"         value={result.confidence}        color="#3b82f6" />

                  {/* Meta */}
                  <div className="mt-4 pt-4 border-t border-blue-900/30 space-y-2 text-xs text-slate-400">
                    <div className="flex justify-between">
                      <span className="flex items-center gap-1"><Clock size={12} /> Processing Time</span>
                      <span className="font-mono">{result.processing_time.toFixed(3)}s</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="flex items-center gap-1"><Cpu size={12} /> Model Used</span>
                      <span className="font-mono text-blue-400 text-right max-w-[180px] truncate">{result.model_used}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Frames Analysed</span>
                      <span className="font-mono">{result.frames_sampled ?? 16}</span>
                    </div>
                  </div>

                  {/* Annotate */}
                  {!annotated && !annotating && (
                    <button onClick={() => setAnnotating(true)} id="annotate-btn"
                      className="btn-secondary w-full justify-center mt-4 text-sm">
                      Add Ground-Truth Label
                    </button>
                  )}
                  {annotating && (
                    <div className="mt-4 space-y-3">
                      <select value={annotation.gt}
                        onChange={e => setAnnotation(a => ({ ...a, gt: e.target.value }))}
                        className="input-dark">
                        <option value="">Select ground-truth label…</option>
                        <option value="Violent">Violent</option>
                        <option value="Non-Violent">Non-Violent</option>
                      </select>
                      <select value={annotation.cat}
                        onChange={e => setAnnotation(a => ({ ...a, cat: e.target.value }))}
                        className="input-dark">
                        <option value="">Select activity category (optional)…</option>
                        {CATEGORIES.map(c => <option key={c} value={c}>{c}</option>)}
                      </select>
                      <div className="flex gap-2">
                        <button onClick={handleAnnotate} id="save-annotation-btn"
                          disabled={!annotation.gt}
                          className="btn-primary text-sm py-2 flex-1 justify-center">Save Annotation</button>
                        <button onClick={() => setAnnotating(false)} className="btn-secondary text-sm py-2 flex-1 justify-center">Cancel</button>
                      </div>
                    </div>
                  )}
                  {annotated && (
                    <div className="mt-4 glass-card p-3 text-center text-green-400 text-sm">
                      <CheckCircle size={14} className="inline mr-1" /> Annotation saved!
                    </div>
                  )}
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </div>
    </div>
  )
}
