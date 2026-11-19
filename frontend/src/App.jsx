import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import Sidebar from './components/Sidebar'
import Landing from './pages/Landing'
import Dashboard from './pages/Dashboard'
import VideoDetection from './pages/VideoDetection'
import FalsePositiveAnalysis from './pages/FalsePositiveAnalysis'
import DatasetExplorer from './pages/DatasetExplorer'
import ModelTraining from './pages/ModelTraining'
import PredictionHistory from './pages/PredictionHistory'
import About from './pages/About'

function AppShell() {
  return (
    <div className="flex h-screen overflow-hidden" style={{ background: '#050d1a' }}>
      <Sidebar />
      <main className="flex-1 flex flex-col overflow-hidden">
        {/* Mobile top bar — prevents hamburger icon overlapping page headings */}
        <div className="lg:hidden h-14 flex-shrink-0 flex items-center px-16 border-b border-blue-900/20"
             style={{ background: 'rgba(5,13,26,0.95)' }}>
          <span className="text-sm font-semibold text-slate-400 tracking-wider uppercase">VIGIL AI</span>
        </div>
        <div className="flex-1 overflow-y-auto flex flex-col">
          <Routes>
            <Route index element={<Navigate to="dashboard" replace />} />
            <Route path="dashboard" element={<Dashboard />} />
            <Route path="detect"    element={<VideoDetection />} />
            <Route path="analysis"  element={<FalsePositiveAnalysis />} />
            <Route path="dataset"   element={<DatasetExplorer />} />
            <Route path="train"     element={<ModelTraining />} />
            <Route path="history"   element={<PredictionHistory />} />
            <Route path="about"     element={<About />} />
          </Routes>
        </div>
      </main>
    </div>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/"      element={<Landing />} />
        <Route path="/app/*" element={<AppShell />} />
        <Route path="*"      element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  )
}
