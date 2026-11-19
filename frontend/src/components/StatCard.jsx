import { motion } from 'framer-motion'

const colorMap = {
  blue:   { border: 'rgba(59,130,246,0.3)',  glow: 'rgba(59,130,246,0.1)',  text: '#60a5fa', accent: '#3b82f6' },
  violet: { border: 'rgba(139,92,246,0.3)', glow: 'rgba(139,92,246,0.1)', text: '#a78bfa', accent: '#8b5cf6' },
  red:    { border: 'rgba(239,68,68,0.3)',   glow: 'rgba(239,68,68,0.1)',   text: '#f87171', accent: '#ef4444' },
  green:  { border: 'rgba(34,197,94,0.3)',   glow: 'rgba(34,197,94,0.1)',   text: '#4ade80', accent: '#22c55e' },
  cyan:   { border: 'rgba(6,182,212,0.3)',   glow: 'rgba(6,182,212,0.1)',   text: '#22d3ee', accent: '#06b6d4' },
  orange: { border: 'rgba(249,115,22,0.3)',  glow: 'rgba(249,115,22,0.1)',  text: '#fb923c', accent: '#f97316' },
}

export default function StatCard({ icon: Icon, label, value, subtitle, color = 'blue', className = '' }) {
  const c = colorMap[color] || colorMap.blue

  return (
    <motion.div
      whileHover={{ y: -4, scale: 1.01 }}
      transition={{ duration: 0.2 }}
      className={`glass-card p-6 relative overflow-hidden ${className}`}
      style={{ borderColor: c.border }}
    >
      {/* Glow background */}
      <div className="absolute inset-0 opacity-30 pointer-events-none" 
           style={{ background: `radial-gradient(circle at top right, ${c.glow}, transparent 70%)` }} />

      <div className="relative z-10">
        <div className="flex items-start justify-between mb-4">
          <div className="w-11 h-11 rounded-xl flex items-center justify-center"
               style={{ background: c.glow, border: `1px solid ${c.border}` }}>
            <Icon size={20} style={{ color: c.text }} />
          </div>
        </div>
        <div className="text-3xl font-bold text-white mb-1 font-mono">
          {value ?? '—'}
        </div>
        <div className="text-sm font-medium text-slate-300 mb-1">{label}</div>
        {subtitle && <div className="text-xs text-slate-500">{subtitle}</div>}
      </div>
    </motion.div>
  )
}
