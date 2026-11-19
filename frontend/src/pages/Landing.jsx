import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import {
  Shield, Eye, Zap, Brain, ChevronRight, Database,
  Cpu, AlertTriangle, CheckCircle, ArrowRight, Activity,
  ExternalLink, Code
} from 'lucide-react'

const features = [
  {
    icon: Brain,
    title: 'Deep Learning',
    desc: 'CNN+LSTM architecture with MobileNetV2 backbone for spatial and temporal feature extraction.',
    color: 'blue',
  },
  {
    icon: AlertTriangle,
    title: 'False Positive Analysis',
    desc: 'Identifies and categorises misclassified non-violent activities including sports, hugging, and crowd scenes.',
    color: 'orange',
  },
  {
    icon: Activity,
    title: 'Real-Time Inference',
    desc: 'Process video clips on-device and receive violence probability scores in seconds.',
    color: 'violet',
  },
  {
    icon: Database,
    title: 'Multi-Dataset Support',
    desc: 'Integrates with RWF-2000 and Hockey Fight Detection datasets for robust training.',
    color: 'cyan',
  },
  {
    icon: Eye,
    title: 'Explainable AI',
    desc: 'Sample frames and frame-level analysis to understand model decision-making.',
    color: 'green',
  },
  {
    icon: CheckCircle,
    title: 'Evaluation Metrics',
    desc: 'Accuracy, Precision, Recall, F1-Score, ROC-AUC and Confusion Matrix reporting.',
    color: 'blue',
  },
]

const steps = [
  { n: '01', title: 'Upload Video', desc: 'Drag and drop a short video clip in any common format.' },
  { n: '02', title: 'Frame Extraction', desc: '16 frames are uniformly sampled and preprocessed with OpenCV.' },
  { n: '03', title: 'Deep Learning Inference', desc: 'CNN+LSTM model extracts spatial and temporal features.' },
  { n: '04', title: 'Results & Analysis', desc: 'View violence probability, confidence score, and explanations.' },
]

const techStack = [
  { name: 'PyTorch', category: 'Deep Learning', color: '#ef4444' },
  { name: 'FastAPI',  category: 'Backend',       color: '#22c55e' },
  { name: 'React',    category: 'Frontend',      color: '#06b6d4' },
  { name: 'OpenCV',   category: 'Vision',        color: '#f97316' },
  { name: 'SQLite',   category: 'Database',      color: '#8b5cf6' },
  { name: 'Vite',     category: 'Build',         color: '#facc15' },
]

function FloatingOrb({ size, x, y, color, duration }) {
  return (
    <motion.div
      className="absolute rounded-full pointer-events-none"
      style={{
        width: size, height: size,
        left: x, top: y,
        background: `radial-gradient(circle, ${color}22, transparent 70%)`,
        filter: 'blur(40px)',
      }}
      animate={{ y: [0, -30, 0], opacity: [0.4, 0.7, 0.4] }}
      transition={{ duration, repeat: Infinity, ease: 'easeInOut' }}
    />
  )
}

export default function Landing() {
  return (
    <div className="min-h-screen overflow-x-hidden" style={{ background: '#050d1a' }}>
      {/* ── Navbar ───────────────────────────────────────────────────────── */}
      <nav className="fixed top-0 left-0 right-0 z-50 flex items-center justify-between px-6 py-4"
           style={{ background: 'rgba(5, 13, 26, 0.85)', backdropFilter: 'blur(20px)',
                    borderBottom: '1px solid rgba(59,130,246,0.1)' }}>
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl flex items-center justify-center"
               style={{ background: 'linear-gradient(135deg, #3b82f6, #8b5cf6)' }}>
            <Shield size={18} className="text-white" />
          </div>
          <span className="font-bold text-white text-lg tracking-tight">VIGIL AI</span>
        </div>
        <div className="hidden md:flex items-center gap-6 text-sm text-slate-400">
          <a href="#features" className="hover:text-white transition-colors">Features</a>
          <a href="#how-it-works" className="hover:text-white transition-colors">How It Works</a>
          <a href="#datasets" className="hover:text-white transition-colors">Datasets</a>
          <a href="#tech" className="hover:text-white transition-colors">Tech Stack</a>
        </div>
        <Link to="/app/dashboard" id="launch-dashboard-nav"
          className="btn-primary text-sm py-2 px-4">
          Launch Dashboard <ArrowRight size={14} />
        </Link>
      </nav>

      {/* ── Hero ─────────────────────────────────────────────────────────── */}
      <section className="relative min-h-screen flex items-center justify-center px-6 overflow-hidden">
        <FloatingOrb size={600} x="10%" y="10%"  color="#3b82f6" duration={8} />
        <FloatingOrb size={400} x="60%" y="50%"  color="#8b5cf6" duration={10} />
        <FloatingOrb size={300} x="30%" y="60%"  color="#06b6d4" duration={7} />

        <div className="relative z-10 max-w-5xl mx-auto text-center pt-24">
          {/* Badge */}
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-full text-xs font-semibold mb-8"
            style={{ background: 'rgba(59,130,246,0.1)', border: '1px solid rgba(59,130,246,0.2)', color: '#60a5fa' }}>
            <Zap size={12} className="text-blue-400" />
            AI-Powered Video Violence Detection Research System
          </motion.div>

          {/* Headline */}
          <motion.h1 initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}
            className="text-5xl md:text-7xl font-black leading-none mb-6 text-white">
            See Beyond the Frame.
            <br />
            <span className="gradient-text">Detect Violence</span>
            <br />
            Intelligently.
          </motion.h1>

          <motion.p initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.35 }}
            className="text-lg md:text-xl text-slate-400 max-w-3xl mx-auto mb-12 leading-relaxed">
            VIGIL AI is a production-quality deep learning system that classifies violent and non-violent
            activities in short video clips. It analyses false positive patterns to understand why AI models
            sometimes confuse harmless activities with actual violence.
          </motion.p>

          {/* CTA buttons */}
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.5 }}
            className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link to="/app/dashboard" id="launch-dashboard-hero"
              className="btn-primary text-base py-4 px-8 justify-center">
              <Shield size={18} /> Launch AI Dashboard
            </Link>
            <Link to="/app/detect" id="try-detection-hero"
              className="btn-secondary text-base py-4 px-8 justify-center">
              Try Video Detection <ChevronRight size={18} />
            </Link>
          </motion.div>

          {/* Animated orb/scanner */}
          <motion.div
            initial={{ opacity: 0, scale: 0.8 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: 0.7 }}
            className="relative mt-20 max-w-2xl mx-auto"
          >
            <div className="glass-card p-8 relative overflow-hidden"
                 style={{ borderColor: 'rgba(59,130,246,0.25)' }}>
              {/* Scanner line */}
              <div className="absolute left-0 right-0 h-0.5 animate-scan opacity-60"
                   style={{ background: 'linear-gradient(90deg, transparent, #3b82f6, transparent)', top: '20%' }} />

              <div className="grid grid-cols-3 gap-4">
                {[
                  { label: 'Model Architecture', value: 'CNN + LSTM' },
                  { label: 'Input Frames', value: '16 Frames' },
                  { label: 'Frame Resolution', value: '112 × 112' },
                  { label: 'Primary Dataset', value: 'RWF-2000' },
                  { label: 'Training Mode', value: 'PyTorch' },
                  { label: 'Inference', value: 'CPU / GPU' },
                ].map(({ label, value }) => (
                  <div key={label} className="text-center py-3 rounded-lg"
                       style={{ background: 'rgba(59,130,246,0.05)', border: '1px solid rgba(59,130,246,0.1)' }}>
                    <div className="text-blue-400 font-bold font-mono text-sm">{value}</div>
                    <div className="text-slate-500 text-xs mt-1">{label}</div>
                  </div>
                ))}
              </div>
            </div>
          </motion.div>
        </div>
      </section>

      {/* ── Features ─────────────────────────────────────────────────────── */}
      <section id="features" className="py-24 px-6">
        <div className="max-w-6xl mx-auto">
          <motion.div initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }}
            className="text-center mb-16">
            <div className="text-blue-400 text-sm font-semibold uppercase tracking-widest mb-4">Capabilities</div>
            <h2 className="text-4xl font-bold text-white mb-4">Key Features</h2>
            <p className="text-slate-400 max-w-2xl mx-auto">
              A complete research platform combining deep learning inference, explainability, and false positive analytics.
            </p>
          </motion.div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {features.map((f, i) => (
              <motion.div key={f.title}
                initial={{ opacity: 0, y: 30 }} whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }} transition={{ delay: i * 0.08 }}
                className="glass-card-hover p-6">
                <div className={`w-10 h-10 rounded-xl flex items-center justify-center mb-4`}
                     style={{ background: `rgba(59,130,246,0.1)`, border: '1px solid rgba(59,130,246,0.2)' }}>
                  <f.icon size={18} className="text-blue-400" />
                </div>
                <h3 className="text-lg font-bold text-white mb-2">{f.title}</h3>
                <p className="text-slate-400 text-sm leading-relaxed">{f.desc}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* ── How It Works ─────────────────────────────────────────────────── */}
      <section id="how-it-works" className="py-24 px-6"
               style={{ background: 'rgba(59,130,246,0.02)', borderTop: '1px solid rgba(59,130,246,0.08)' }}>
        <div className="max-w-5xl mx-auto">
          <motion.div initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }}
            className="text-center mb-16">
            <div className="text-violet-400 text-sm font-semibold uppercase tracking-widest mb-4">Process</div>
            <h2 className="text-4xl font-bold text-white mb-4">How It Works</h2>
          </motion.div>

          <div className="relative">
            {/* Connecting line */}
            <div className="hidden md:block absolute top-10 left-0 right-0 h-0.5 mx-16"
                 style={{ background: 'linear-gradient(90deg, #3b82f6, #8b5cf6)' }} />

            <div className="grid md:grid-cols-4 gap-6">
              {steps.map((step, i) => (
                <motion.div key={step.n}
                  initial={{ opacity: 0, y: 30 }} whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }} transition={{ delay: i * 0.1 }}
                  className="relative text-center">
                  <div className="relative z-10 w-20 h-20 mx-auto mb-4 rounded-2xl flex flex-col items-center justify-center"
                       style={{ background: 'linear-gradient(135deg, #1a3468, #1a0533)',
                                border: '1px solid rgba(59,130,246,0.3)', boxShadow: '0 0 20px rgba(59,130,246,0.2)' }}>
                    <span className="text-blue-400 font-bold font-mono text-sm">{step.n}</span>
                  </div>
                  <h3 className="text-white font-bold mb-2">{step.title}</h3>
                  <p className="text-slate-400 text-sm">{step.desc}</p>
                </motion.div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* ── Datasets ─────────────────────────────────────────────────────── */}
      <section id="datasets" className="py-24 px-6">
        <div className="max-w-5xl mx-auto">
          <motion.div initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }}
            className="text-center mb-16">
            <div className="text-cyan-400 text-sm font-semibold uppercase tracking-widest mb-4">Data</div>
            <h2 className="text-4xl font-bold text-white mb-4">Dataset Integration</h2>
          </motion.div>

          <div className="grid md:grid-cols-2 gap-8">
            {[
              {
                name: 'RWF-2000',
                desc: '2000 video clips (1000 fight, 1000 non-fight) collected from real-world surveillance camera footage. High quality with diverse scenarios.',
                size: '2,000 clips',
                split: '80/10/10 train/val/test',
                color: '#3b82f6',
              },
              {
                name: 'Hockey Fight Detection',
                desc: 'Hockey game videos labelled as fight or no-fight. Contains sports activities that are frequently misclassified as violence – ideal for false positive research.',
                size: '1,000 clips',
                split: '80/10/10 train/val/test',
                color: '#8b5cf6',
              },
            ].map((ds, i) => (
              <motion.div key={ds.name}
                initial={{ opacity: 0, x: i === 0 ? -30 : 30 }} whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                className="glass-card p-8"
                style={{ borderColor: `${ds.color}33` }}>
                <div className="flex items-center gap-3 mb-4">
                  <Database size={20} style={{ color: ds.color }} />
                  <h3 className="text-xl font-bold text-white">{ds.name}</h3>
                </div>
                <p className="text-slate-400 text-sm mb-6 leading-relaxed">{ds.desc}</p>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-slate-500">Dataset Size</span>
                    <span className="font-mono" style={{ color: ds.color }}>{ds.size}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500">Data Split</span>
                    <span className="font-mono" style={{ color: ds.color }}>{ds.split}</span>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Tech Stack ───────────────────────────────────────────────────── */}
      <section id="tech" className="py-24 px-6"
               style={{ borderTop: '1px solid rgba(59,130,246,0.08)' }}>
        <div className="max-w-4xl mx-auto text-center">
          <motion.div initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }}
            className="mb-12">
            <div className="text-green-400 text-sm font-semibold uppercase tracking-widest mb-4">Stack</div>
            <h2 className="text-4xl font-bold text-white mb-4">Technology Stack</h2>
          </motion.div>
          <div className="flex flex-wrap gap-4 justify-center">
            {techStack.map((t, i) => (
              <motion.div key={t.name}
                initial={{ opacity: 0, scale: 0.8 }} whileInView={{ opacity: 1, scale: 1 }}
                viewport={{ once: true }} transition={{ delay: i * 0.05 }}
                className="px-6 py-4 rounded-xl"
                style={{ background: 'rgba(10,22,40,0.8)', border: `1px solid ${t.color}33` }}>
                <div className="font-bold text-white">{t.name}</div>
                <div className="text-xs mt-0.5" style={{ color: t.color }}>{t.category}</div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* ── CTA ──────────────────────────────────────────────────────────── */}
      <section className="py-24 px-6">
        <motion.div initial={{ opacity: 0, y: 30 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }}
          className="max-w-3xl mx-auto text-center glass-card p-16"
          style={{ background: 'linear-gradient(135deg, rgba(59,130,246,0.1), rgba(139,92,246,0.1))' }}>
          <Shield size={40} className="text-blue-400 mx-auto mb-6" />
          <h2 className="text-4xl font-bold text-white mb-4">Ready to Analyse?</h2>
          <p className="text-slate-400 mb-8">
            Launch the full AI dashboard to upload videos, train models, and explore false positive analytics.
          </p>
          <Link to="/app/dashboard" id="launch-dashboard-cta" className="btn-primary text-lg py-4 px-10 mx-auto">
            <Shield size={20} /> Launch AI Dashboard <ArrowRight size={18} />
          </Link>
        </motion.div>
      </section>

      {/* ── Footer ───────────────────────────────────────────────────────── */}
      <footer className="py-8 px-6 text-center text-slate-600 text-sm"
              style={{ borderTop: '1px solid rgba(59,130,246,0.08)' }}>
        <div className="flex items-center justify-center gap-2 mb-2">
          <Shield size={14} className="text-blue-500" />
          <span className="font-semibold text-slate-400">VIGIL AI</span>
        </div>
        <p>AI-Powered Video Violence Detection &amp; Analysis System · Built with PyTorch, FastAPI &amp; React</p>
      </footer>
    </div>
  )
}
