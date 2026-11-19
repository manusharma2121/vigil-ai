import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  timeout: 120000,
})

// ── Predictions ──────────────────────────────────────────────
export const predictVideo = (formData, onUploadProgress) =>
  api.post('/predict', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress,
  })

export const annotateVideo = (data) =>
  api.post('/predict/annotate', data)

// ── History ──────────────────────────────────────────────────
export const getHistory = (params) =>
  api.get('/history', { params })

export const deleteHistory = (id) =>
  api.delete(`/history/${id}`)

export const exportHistory = () =>
  api.get('/history/export', { responseType: 'blob' })

// ── Dashboard ────────────────────────────────────────────────
export const getDashboardStats = () =>
  api.get('/dashboard/stats')

// ── Analysis ─────────────────────────────────────────────────
export const getFPAnalysis = () =>
  api.get('/analysis/false-positives')

export const exportFPReport = () =>
  api.get('/analysis/false-positives/export', { responseType: 'blob' }).then(r => {
    const url = window.URL.createObjectURL(new Blob([r.data]))
    const a = document.createElement('a')
    a.href = url
    a.download = `vigil_fp_report_${Date.now()}.csv`
    a.click()
  })

// ── Dataset ──────────────────────────────────────────────────
export const getDatasetInfo = () =>
  api.get('/dataset/info')

export const loadDataset = (name) =>
  api.post('/dataset/load', { dataset_name: name })

// ── Training ─────────────────────────────────────────────────
export const startTraining = (config) =>
  api.post('/train', config)

export const getTrainingStatus = () =>
  api.get('/train/status')

// ── Model ────────────────────────────────────────────────────
export const getModelInfo = () =>
  api.get('/model/info')

export default api
