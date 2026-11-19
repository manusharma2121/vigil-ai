import { useState, useEffect } from 'react'
import { Link, useLocation } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Shield, LayoutDashboard, Video, AlertTriangle,
  Database, Cpu, History, Info, Menu, X, Activity
} from 'lucide-react'

const navItems = [
  { path: '/app/dashboard', icon: LayoutDashboard, label: 'Dashboard' },
  { path: '/app/detect',    icon: Video,           label: 'Video Detection' },
  { path: '/app/analysis',  icon: AlertTriangle,   label: 'False Positive Analysis' },
  { path: '/app/dataset',   icon: Database,        label: 'Dataset Explorer' },
  { path: '/app/train',     icon: Cpu,             label: 'Model Training' },
  { path: '/app/history',   icon: History,         label: 'Prediction History' },
  { path: '/app/about',     icon: Info,            label: 'About Project' },
]

export default function Sidebar() {
  const location = useLocation()
  const [collapsed, setCollapsed] = useState(false)
  const [mobileOpen, setMobileOpen] = useState(false)

  // Close mobile nav on route change
  useEffect(() => { setMobileOpen(false) }, [location.pathname])

  const SidebarContent = () => (
    <div className="flex flex-col h-full">
      {/* Logo */}
      <div className="flex items-center gap-3 px-4 py-6 border-b border-blue-900/30">
        <div className="relative">
          <div className="w-10 h-10 rounded-xl flex items-center justify-center"
               style={{ background: 'linear-gradient(135deg, #3b82f6, #8b5cf6)' }}>
            <Shield size={20} className="text-white" />
          </div>
          <div className="absolute -top-1 -right-1 w-3 h-3 rounded-full bg-green-400 animate-pulse" />
        </div>
        {!collapsed && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
            <div className="font-bold text-white text-lg leading-none">VIGIL AI</div>
            <div className="text-xs text-slate-500 mt-0.5">Violence Detection</div>
          </motion.div>
        )}
      </div>

      {/* Nav items */}
      <nav className="flex-1 px-3 py-4 space-y-1 overflow-y-auto">
        {navItems.map(({ path, icon: Icon, label }) => {
          const active = location.pathname === path || location.pathname.startsWith(path)
          return (
            <Link key={path} to={path}
              className={`nav-link ${active ? 'active' : ''} ${collapsed ? 'justify-center px-3' : ''}`}
              title={collapsed ? label : undefined}
            >
              <Icon size={18} className={active ? 'text-blue-400' : ''} />
              {!collapsed && <span>{label}</span>}
              {active && !collapsed && (
                <motion.div layoutId="active-pill"
                  className="ml-auto w-1.5 h-1.5 rounded-full bg-blue-400" />
              )}
            </Link>
          )
        })}
      </nav>

      {/* Footer */}
      <div className="px-4 py-4 border-t border-blue-900/30">
        {!collapsed && (
          <div className="text-xs text-slate-600 font-mono">v1.0.0 · PyTorch</div>
        )}
      </div>
    </div>
  )

  return (
    <>
      {/* Desktop sidebar */}
      <motion.aside
        animate={{ width: collapsed ? 72 : 240 }}
        transition={{ duration: 0.2 }}
        className="hidden lg:flex flex-col flex-shrink-0 relative"
        style={{
          background: 'rgba(5, 13, 26, 0.95)',
          borderRight: '1px solid rgba(59, 130, 246, 0.1)',
        }}
      >
        {/* Collapse toggle */}
        <button
          onClick={() => setCollapsed(!collapsed)}
          className="absolute -right-3 top-7 z-10 w-6 h-6 rounded-full flex items-center justify-center transition-colors"
          style={{ background: 'rgba(10, 22, 40, 0.9)', border: '1px solid rgba(59, 130, 246, 0.3)' }}
        >
          <Activity size={12} className="text-blue-400" />
        </button>
        <SidebarContent />
      </motion.aside>

      {/* Mobile hamburger */}
      <div className="lg:hidden fixed top-4 left-4 z-50">
        <button onClick={() => setMobileOpen(true)}
          className="w-10 h-10 rounded-xl flex items-center justify-center glass-card">
          <Menu size={20} className="text-slate-300" />
        </button>
      </div>

      {/* Mobile drawer */}
      <AnimatePresence>
        {mobileOpen && (
          <>
            <motion.div
              initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
              className="fixed inset-0 bg-black/70 z-40 lg:hidden"
              onClick={() => setMobileOpen(false)}
            />
            <motion.aside
              initial={{ x: -260 }} animate={{ x: 0 }} exit={{ x: -260 }}
              transition={{ type: 'spring', damping: 25 }}
              className="fixed left-0 top-0 bottom-0 w-64 z-50 lg:hidden"
              style={{
                background: 'rgba(5, 13, 26, 0.98)',
                borderRight: '1px solid rgba(59, 130, 246, 0.2)',
              }}
            >
              <button onClick={() => setMobileOpen(false)}
                className="absolute right-4 top-4 p-2 rounded-lg text-slate-400 hover:text-white">
                <X size={18} />
              </button>
              <SidebarContent />
            </motion.aside>
          </>
        )}
      </AnimatePresence>
    </>
  )
}
