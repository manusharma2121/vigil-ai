import { motion } from 'framer-motion'
import {
  BookOpen, Target, Database, Cpu, FlaskConical,
  AlertTriangle, Rocket, Users, ChevronRight
} from 'lucide-react'

const Section = ({ icon: Icon, title, color, children }) => (
  <motion.div initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }}
    viewport={{ once: true }} className="glass-card p-6 mb-6">
    <div className="flex items-center gap-3 mb-4">
      <div className="p-2 rounded-xl" style={{ background: `${color}20` }}>
        <Icon size={20} style={{ color }} />
      </div>
      <h2 className="text-lg font-bold text-white">{title}</h2>
    </div>
    <div className="text-slate-400 text-sm leading-relaxed space-y-2">{children}</div>
  </motion.div>
)

export default function About() {
  return (
    <div className="flex-1 p-6 lg:p-8 overflow-y-auto">
      <div className="max-w-3xl mx-auto">
        {/* Hero */}
        <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }}
          className="text-center mb-12">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl mb-6"
               style={{ background: 'linear-gradient(135deg,#1e40af,#7c3aed)' }}>
            <FlaskConical size={28} className="text-white" />
          </div>
          <h1 className="text-4xl font-black text-white mb-3">About VIGIL AI</h1>
          <p className="text-slate-400 text-lg max-w-xl mx-auto">
            An academic research system for intelligent violence detection and false positive analysis
            in video surveillance scenarios.
          </p>
        </motion.div>

        <Section icon={Target} title="Problem Statement" color="#3b82f6">
          <p>
            Automated video-based violence detection is critical for public safety in crowded venues,
            transport hubs, and urban surveillance networks. However, current deep learning models
            frequently produce <strong className="text-slate-200">false positive</strong> predictions —
            classifying non-violent activities as violent — particularly when the motion patterns
            resemble aggressive behaviour.
          </p>
          <p>
            Activities such as contact sports, crowd movement, dancing, and physical greetings share
            temporal motion features with actual violence, causing the model to misfire.
            Understanding and quantifying these error categories is the central research objective of
            this project.
          </p>
        </Section>

        <Section icon={BookOpen} title="Research Objectives" color="#8b5cf6">
          <ul className="space-y-2">
            {[
              'Train a deep learning classifier (CNN+LSTM or 3D-CNN) on real-world violence datasets.',
              'Generate calibrated violence confidence scores for every uploaded video clip.',
              'Identify, categorise, and quantify false positive predictions.',
              'Determine which non-violent activity categories are most frequently misclassified.',
              'Provide visual analytics and exportable reports to support academic research.',
              'Enable annotation-based ground-truth labelling for rigorous evaluation.',
            ].map(o => (
              <li key={o} className="flex items-start gap-2">
                <ChevronRight size={14} className="text-violet-400 flex-shrink-0 mt-0.5" />
                {o}
              </li>
            ))}
          </ul>
        </Section>

        <Section icon={Database} title="Datasets" color="#06b6d4">
          <div className="grid sm:grid-cols-2 gap-4">
            {[
              {
                name: 'RWF-2000',
                desc: '2,000 video clips collected from YouTube covering real-world fight and non-fight scenarios in diverse settings including streets, bars, and public spaces.',
                url: 'https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection',
              },
              {
                name: 'Hockey Fight',
                desc: '1,000 clips from professional hockey games, balanced between fight and non-fight clips. Commonly used for benchmarking violence detection models.',
                url: 'http://viplab.cs.nthu.edu.tw/~lain/data/',
              },
            ].map(ds => (
              <div key={ds.name} className="rounded-xl p-4"
                   style={{ background: 'rgba(6,182,212,0.05)', border: '1px solid rgba(6,182,212,0.15)' }}>
                <h4 className="text-cyan-400 font-semibold mb-2">{ds.name}</h4>
                <p className="text-xs">{ds.desc}</p>
              </div>
            ))}
          </div>
          <p className="mt-3 text-xs">
            If neither dataset is installed, the system operates in <strong className="text-yellow-400">demo mode</strong>:
            predictions are computed with an untrained model and are clearly labelled as such.
            No accuracy values are fabricated.
          </p>
        </Section>

        <Section icon={Cpu} title="Model Architecture" color="#f97316">
          <div className="space-y-4">
            <div className="rounded-xl p-4"
                 style={{ background: 'rgba(249,115,22,0.05)', border: '1px solid rgba(249,115,22,0.15)' }}>
              <h4 className="text-orange-400 font-semibold mb-2">Primary: CNN + LSTM</h4>
              <p>
                MobileNetV2 is used as a spatial feature extractor for each sampled frame. The
                per-frame feature vectors (1280-dim) are fed into a 2-layer bidirectional LSTM that
                captures temporal motion dynamics across 16 uniformly sampled frames. A fully
                connected head outputs a binary classification with Sigmoid activation.
              </p>
            </div>
            <div className="rounded-xl p-4"
                 style={{ background: 'rgba(249,115,22,0.05)', border: '1px solid rgba(249,115,22,0.15)' }}>
              <h4 className="text-orange-400 font-semibold mb-2">Alternative: 3D-CNN</h4>
              <p>
                A lightweight 3D convolutional network with three Conv3D blocks, batch normalisation,
                and max pooling extracts joint spatial-temporal features directly from the video
                volume (16×112×112). A global average pooling layer feeds a linear classifier.
              </p>
            </div>
          </div>
        </Section>

        <Section icon={FlaskConical} title="Methodology" color="#22c55e">
          <ol className="space-y-2 list-decimal list-inside">
            {[
              'Video frames are uniformly sampled (16 frames per clip) using OpenCV.',
              'Frames are resized to 112×112 and normalised using ImageNet mean/std.',
              'The model processes the frame sequence and outputs violence probability.',
              'Predictions are stored in SQLite with metadata for historical analysis.',
              'Ground-truth labels (when provided) enable computation of evaluation metrics.',
              'Activity category annotations drive the false positive breakdown analysis.',
            ].map(s => (
              <li key={s}>{s}</li>
            ))}
          </ol>
        </Section>

        <Section icon={AlertTriangle} title="Limitations" color="#ef4444">
          <ul className="space-y-2">
            {[
              'Model accuracy is only meaningful when trained on the target dataset — demo mode results are not predictive.',
              'False positive analysis requires manually annotated category labels.',
              'The system does not detect violence in real-time streams — only uploaded short clips.',
              'Grad-CAM saliency maps are approximations and do not prove why a prediction was made.',
              'Dataset imbalance, lighting, and camera angle affect generalisation.',
            ].map(l => (
              <li key={l} className="flex gap-2">
                <AlertTriangle size={12} className="text-red-400 flex-shrink-0 mt-1" />
                {l}
              </li>
            ))}
          </ul>
        </Section>

        <Section icon={Rocket} title="Future Scope" color="#a855f7">
          <ul className="space-y-2">
            {[
              'Real-time video stream analysis with WebSocket API.',
              'Automated active learning to annotate ambiguous clips.',
              'Optical flow feature fusion for improved temporal modelling.',
              'Multi-label classification for simultaneous violence type detection.',
              'Federated learning for privacy-preserving distributed training.',
              'Mobile deployment via ONNX / TFLite model export.',
            ].map(f => (
              <li key={f} className="flex gap-2">
                <ChevronRight size={14} className="text-purple-400 flex-shrink-0 mt-0.5" />
                {f}
              </li>
            ))}
          </ul>
        </Section>

        {/* Tech stack */}
        <motion.div initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }} className="glass-card p-6">
          <h2 className="text-lg font-bold text-white mb-5">Technology Stack</h2>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
            {[
              ['React + Vite', '#61DAFB'], ['Tailwind CSS', '#38BDF8'],
              ['Framer Motion', '#BB22FF'], ['Recharts', '#22C55E'],
              ['FastAPI', '#009688'], ['PyTorch', '#EE4C2C'],
              ['OpenCV', '#5C3EE8'], ['Scikit-learn', '#F7931E'],
              ['SQLite', '#003B57'], ['NumPy', '#4DABCF'],
            ].map(([name, color]) => (
              <div key={name} className="flex items-center gap-2 rounded-xl p-3"
                   style={{ background: `${color}10`, border: `1px solid ${color}30` }}>
                <div className="w-2 h-2 rounded-full flex-shrink-0" style={{ background: color }} />
                <span className="text-sm text-slate-200">{name}</span>
              </div>
            ))}
          </div>
        </motion.div>
      </div>
    </div>
  )
}
