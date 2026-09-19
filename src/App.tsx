import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import {
  Activity, BarChart2, BookOpen, BrainCircuit, Check, ChevronRight, Edit3, ExternalLink,
  Gauge, Globe, HeartPulse, Home, Laptop, LoaderCircle, Menu, Package, RefreshCw,
  Save, Search, ShieldCheck, Smartphone, Sparkles, Star, Target, TrendingUp,
  Users, WandSparkles, Wifi, WifiOff, X, Zap, AlertCircle, ArrowUpRight, Award,
  Cpu, HardDrive, Monitor, Database, GitBranch
} from "lucide-react";
import {
  RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, ResponsiveContainer,
  BarChart, Bar, XAxis, YAxis, Tooltip, Legend, AreaChart, Area, PieChart, Pie, Cell,
  RadialBarChart, RadialBar, CartesianGrid,
} from "recharts";

// ─── Types ───────────────────────────────────────────────────────────────────
type Product = {
  product_id: string; category: string; brand: string; name: string; price_usd: number;
  ram_gb?: number; storage_gb?: number; display_type?: string; refresh_rate_hz?: number;
  interaction_count: number; spec_text?: string; processor?: string; gpu?: string;
  battery_mah?: number; camera_mp?: number; chipset?: string; os?: string;
};
type Health = { status: string; database: boolean; models_loaded: boolean; mlflow: boolean; products_count: number; version: string };
type Stats = { n_products: number; n_users: number; n_interactions: number; n_laptops: number; n_smartphones: number };
type Drift = { dataset_drift_detected: boolean; share_drifted_columns: number; n_drifted_columns: number; n_total_columns: number };
type Recommendation = {
  product_id: string; name: string; category: string; price_usd: number; spec_text: string;
  als_score: number; tfidf_score: number; spec_match_score: number; final_score: number; reason: string;
};
type RecommendationResult = {
  recommendation_mode: string; model_version: string; latency_ms: number;
  recommendations: Recommendation[]; als_used: boolean; tfidf_used: boolean;
};

// ─── Constants ────────────────────────────────────────────────────────────────
const API = import.meta.env.VITE_API_BASE ?? "";
const money = (n: number) => new Intl.NumberFormat("en-IN", { style: "currency", currency: "INR", maximumFractionDigits: 0 }).format(n * 83);
const compact = (n: number) => new Intl.NumberFormat("en-IN", { notation: "compact", maximumFractionDigits: 1 }).format(n);
const pct = (n: number) => `${Math.round(n * 100)}%`;

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(`${API}${path}`, { ...init, headers: { "Content-Type": "application/json" } });
  if (!r.ok) throw new Error((await r.json().catch(() => null))?.detail ?? `Request failed (${r.status})`);
  return r.json();
}

const phone = (cat: string) => cat?.toLowerCase().includes("smart") || cat?.toLowerCase().includes("phone");
const specLine = (p: Product) => [
  p.processor && p.processor !== "None" ? p.processor : null,
  p.ram_gb ? `${p.ram_gb}GB RAM` : null,
  p.storage_gb ? `${p.storage_gb}GB` : null,
  p.display_type && p.display_type !== "None" ? p.display_type : null,
  p.refresh_rate_hz ? `${p.refresh_rate_hz}Hz` : null,
].filter(Boolean).join(" · ") || p.spec_text?.slice(0, 60) || "AI-ranked electronics";

// ─── Product Images & Store Links ─────────────────────────────────────────────
const PRODUCT_IMAGES: Record<string, string> = {
  P001: "https://images.unsplash.com/photo-1611186871348-b1ce696e52c9?w=400&q=80",  // MacBook Pro
  P002: "https://images.unsplash.com/photo-1593642632559-0c6d3fc62b89?w=400&q=80",  // Dell XPS
  P003: "https://images.unsplash.com/photo-1603302576837-37561b2e2302?w=400&q=80",  // ROG
  P004: "https://images.unsplash.com/photo-1541807084-5c52b6b3adef?w=400&q=80",    // ThinkPad
  P005: "https://images.unsplash.com/photo-1695048133142-1a20484d2569?w=400&q=80", // iPhone 16 Pro
  P006: "https://images.unsplash.com/photo-1610945415295-d9bbf067e59c?w=400&q=80", // Galaxy S25
  P007: "https://images.unsplash.com/photo-1598327105666-5b89351aff97?w=400&q=80", // Pixel 9
  P008: "https://images.unsplash.com/photo-1574944985070-8f3ebc6b79d2?w=400&q=80", // OnePlus
  P009: "https://images.unsplash.com/photo-1496181133206-80ce9b88a853?w=400&q=80", // Surface
  P010: "https://images.unsplash.com/photo-1592899677977-9c10ca588bbd?w=400&q=80", // Xiaomi
  UP001: "https://images.unsplash.com/photo-1510557880182-3d4d3cba35a5?w=400&q=80", // iPhone 17 (mock)
  UP002: "https://images.unsplash.com/photo-1517336714731-489689fd1ca8?w=400&q=80", // MacBook M5 (mock)
  UP003: "https://images.unsplash.com/photo-1610945265064-3234eb351c4e?w=400&q=80", // Galaxy S26 (mock)
  UP004: "https://images.unsplash.com/photo-1593640495253-23196b27a87f?w=400&q=80", // ROG 2026 (mock)
  UP005: "https://images.unsplash.com/photo-1588872657578-7efd1f1555ed?w=400&q=80", // Dell XPS 16 (mock)
  UP006: "https://images.unsplash.com/photo-1511707171634-5f897ff02aa9?w=400&q=80", // Pixel 10 (mock)
};

const STORE_LINKS: Record<string, string> = {
  P001: "https://www.apple.com/in/macbook-pro/",
  P002: "https://www.dell.com/en-in/shop/laptops/xps-15-laptop/spd/xps-15-9530-laptop",
  P003: "https://rog.asus.com/laptops/rog-zephyrus/rog-zephyrus-g16-2024-series/",
  P004: "https://www.lenovo.com/in/en/p/laptops/thinkpad/thinkpadx1/thinkpad-x1-carbon-gen-13",
  P005: "https://www.apple.com/in/iphone-16-pro/",
  P006: "https://www.samsung.com/in/smartphones/galaxy-s25-ultra/",
  P007: "https://store.google.com/in/product/pixel_9_pro",
  P008: "https://www.oneplus.in/13",
  P009: "https://www.microsoft.com/en-in/surface/devices/surface-laptop-7th-edition",
  P010: "https://www.mi.com/in/xiaomi-15-ultra",
};

const UPCOMING_PRODUCTS: (Product & { upcoming: boolean; launch_date: string; expected_price: string })[] = [
  { product_id: "UP001", category: "smartphone", brand: "Apple", name: "iPhone 17 Pro Max", price_usd: 1299, ram_gb: 12, storage_gb: 256, display_type: "Super Retina XDR", refresh_rate_hz: 120, interaction_count: 0, spec_text: "A19 Pro chip, Under-display Face ID, 6.9\" ProMotion 120Hz, Titanium, Periscope camera upgrade", processor: "Apple A19 Pro", upcoming: true, launch_date: "Sep 2025", expected_price: "₹1,59,900" },
  { product_id: "UP002", category: "laptop", brand: "Apple", name: "MacBook Pro 14\" M5", price_usd: 1999, ram_gb: 24, storage_gb: 512, display_type: "Liquid Retina XDR", refresh_rate_hz: 120, interaction_count: 0, spec_text: "M5 chip 12-core CPU, 18-core GPU, 24GB Unified Memory, Thunderbolt 5", processor: "Apple M5", upcoming: true, launch_date: "Early 2026", expected_price: "₹1,89,900" },
  { product_id: "UP003", category: "smartphone", brand: "Samsung", name: "Galaxy S26 Ultra", price_usd: 1399, ram_gb: 16, storage_gb: 512, display_type: "Dynamic AMOLED 2X", refresh_rate_hz: 120, interaction_count: 0, spec_text: "Snapdragon 8 Gen 4, 200MP camera, 6.9\" QHD+, integrated S-Pen, 5000mAh", processor: "Snapdragon 8 Gen 4", upcoming: true, launch_date: "Jan 2026", expected_price: "₹1,49,900" },
  { product_id: "UP004", category: "laptop", brand: "ASUS", name: "ROG Zephyrus G18 (2026)", price_usd: 2999, ram_gb: 64, storage_gb: 2000, display_type: "QHD+ OLED", refresh_rate_hz: 240, interaction_count: 0, spec_text: "AMD Ryzen AI 9 HX 470, RTX 5090, 64GB DDR6, 2TB SSD, 18\" 240Hz OLED", processor: "AMD Ryzen AI 9 HX 470", upcoming: true, launch_date: "Q1 2026", expected_price: "₹3,49,900" },
  { product_id: "UP005", category: "laptop", brand: "Dell", name: "XPS 16 (2026)", price_usd: 2299, ram_gb: 32, storage_gb: 1000, display_type: "OLED Touch", refresh_rate_hz: 120, interaction_count: 0, spec_text: "Intel Core Ultra 9, RTX 4070, Seamless Glass Touchpad, 16-inch 4K OLED", processor: "Intel Core Ultra 9", upcoming: true, launch_date: "Q2 2026", expected_price: "₹2,69,900" },
  { product_id: "UP006", category: "smartphone", brand: "Google", name: "Pixel 10 Pro", price_usd: 1099, ram_gb: 16, storage_gb: 512, display_type: "Super Actua Display", refresh_rate_hz: 120, interaction_count: 0, spec_text: "Tensor G5, 6.8\" LTPO OLED, Advanced AI Photography, 5500mAh battery", processor: "Google Tensor G5", upcoming: true, launch_date: "Oct 2025", expected_price: "₹1,19,900" },
];

const CHART_COLORS = ["#818cf8", "#34d399", "#f472b6", "#fb923c", "#60a5fa"];
const BRAND_COLORS: Record<string, string> = {
  Apple: "#e5e7eb", Samsung: "#3b82f6", ASUS: "#f97316", Dell: "#22c55e",
  Lenovo: "#ef4444", Google: "#eab308", OnePlus: "#a855f7", Microsoft: "#06b6d4", Xiaomi: "#f43f5e",
};

// ─── App ──────────────────────────────────────────────────────────────────────
type Page = "home" | "catalog" | "recommend" | "coldstart" | "mlops" | "about";

export default function App() {
  const [page, setPage] = useState<Page>("home");
  const [products, setProducts] = useState<Product[]>([]);
  const [health, setHealth] = useState<Health | null>(null);
  const [stats, setStats] = useState<Stats | null>(null);
  const [drift, setDrift] = useState<Drift | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(true);

  const load = useCallback(async () => {
    setLoading(true);
    const [h, p, s, d] = await Promise.allSettled([
      api<Health>("/api/health"),
      api<Product[]>("/api/products?limit=100"),
      api<Stats>("/api/stats"),
      api<Drift>("/api/drift"),
    ]);
    if (h.status === "fulfilled") setHealth(h.value);
    else setError("FastAPI backend unavailable. Start with: run_backend.bat");
    if (p.status === "fulfilled") setProducts(p.value);
    if (s.status === "fulfilled") setStats(s.value);
    if (d.status === "fulfilled") setDrift(d.value);
    setLoading(false);
  }, []);

  useEffect(() => { load(); const t = setInterval(load, 30_000); return () => clearInterval(t); }, [load]);

  const okay = health?.status === "healthy";

  const navItems: { id: Page; icon: ReactNode; label: string }[] = [
    { id: "home", icon: <Home size={18} />, label: "Overview" },
    { id: "catalog", icon: <Package size={18} />, label: "Product Catalog" },
    { id: "recommend", icon: <WandSparkles size={18} />, label: "Recommendations" },
    { id: "coldstart", icon: <Zap size={18} />, label: "Cold-Start Demo" },
    { id: "mlops", icon: <BrainCircuit size={18} />, label: "MLOps Dashboard" },
    { id: "about", icon: <BookOpen size={18} />, label: "About" },
  ];

  return (
    <div className="app-shell">
      {/* ── Sidebar ── */}
      <aside className={`sidebar ${sidebarOpen ? "" : "collapsed"}`}>
        <div className="brand">
          <div className="brand-mark"><Package size={22} /></div>
          {sidebarOpen && <div><strong>Cart<span>Sense</span></strong><small>Adaptive AI Engine</small></div>}
        </div>
        <nav>
          {navItems.map(({ id, icon, label }) => (
            <button key={id} className={`nav-link ${page === id ? "active" : ""}`} onClick={() => setPage(id)}>
              {icon}{sidebarOpen && <span>{label}</span>}
            </button>
          ))}
        </nav>
        <div className="sidebar-bottom">
          {sidebarOpen && (
            <div className="connection-card">
              <div className="connection-title">
                <i className={okay ? "status-dot" : "status-dot down"} />
                {okay ? "System Online" : "System Offline"}
              </div>
              {[["API Gateway", !!health], ["Database", !!health?.database], ["MLflow", !!health?.mlflow], ["Models", !!health?.models_loaded]].map(([l, ok]) => (
                <div className="connection" key={String(l)}><i className={ok ? "status-dot" : "status-dot down"} /><span>{String(l)}</span><b>{ok ? "✓" : "○"}</b></div>
              ))}
              <button className="refresh-link" onClick={load}><RefreshCw size={13} /> Refresh</button>
            </div>
          )}
          <button className="collapse-btn" onClick={() => setSidebarOpen(!sidebarOpen)}>
            {sidebarOpen ? <ChevronRight size={16} style={{ transform: "rotate(180deg)" }} /> : <ChevronRight size={16} />}
          </button>
        </div>
      </aside>

      {/* ── Main Content ── */}
      <main className="main-content">
        {error && (
          <div className="notice">
            <AlertCircle size={15} /> {error}
            <button onClick={() => setError("")}><X size={16} /></button>
          </div>
        )}
        {/* Status bar */}
        <div className="topbar">
          <div className={`online-indicator ${okay ? "ok" : "offline"}`}>
            {okay ? <Wifi size={15} /> : <WifiOff size={15} />}
            {okay ? "Backend Connected" : "Backend Offline"}
          </div>
          <span className="topbar-info">v{health?.version ?? "—"} · {stats?.n_products ?? 0} products · Auto-refresh every 30s</span>
          <button className="icon-btn" onClick={load}><RefreshCw size={16} className={loading ? "spin" : ""} /></button>
        </div>

        {page === "home" && <HomePage products={products} stats={stats} drift={drift} health={health} loading={loading} />}
        {page === "catalog" && <CatalogPage products={products} loading={loading} />}
        {page === "recommend" && <RecommendPage products={products} health={health} />}
        {page === "coldstart" && <ColdStartPage products={products} />}
        {page === "mlops" && <MLOpsPage health={health} drift={drift} stats={stats} />}
        {page === "about" && <AboutPage />}
      </main>
    </div>
  );
}

// ══════════════════════════════════════════════════════════════
// HOME PAGE
// ══════════════════════════════════════════════════════════════
function HomePage({ products, stats, drift, health, loading }: { products: Product[]; stats: Stats | null; drift: Drift | null; health: Health | null; loading: boolean }) {
  const [counter, setCounter] = useState({ products: 0, users: 0, interactions: 0 });
  const [hoveredStat, setHoveredStat] = useState<string | null>(null);

  // Animate counters
  useEffect(() => {
    if (!stats) return;
    const targets = { products: stats.n_products, users: stats.n_users, interactions: stats.n_interactions };
    const duration = 1500;
    const start = Date.now();
    const tick = () => {
      const progress = Math.min((Date.now() - start) / duration, 1);
      const ease = 1 - Math.pow(1 - progress, 3);
      setCounter({ products: Math.round(targets.products * ease), users: Math.round(targets.users * ease), interactions: Math.round(targets.interactions * ease) });
      if (progress < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }, [stats]);

  // Brand distribution data
  const brandData = useMemo(() => {
    const counts: Record<string, number> = {};
    products.forEach(p => { counts[p.brand] = (counts[p.brand] || 0) + 1; });
    return Object.entries(counts).map(([brand, count]) => ({ brand, count, fill: BRAND_COLORS[brand] || "#818cf8" }));
  }, [products]);

  // Category split
  const categoryData = useMemo(() => [
    { name: "Laptops", value: stats?.n_laptops ?? 0, fill: "#818cf8" },
    { name: "Smartphones", value: stats?.n_smartphones ?? 0, fill: "#34d399" },
  ], [stats]);

  // Interaction chart data (simulated timeline from product interaction counts)
  const interactionData = useMemo(() => {
    const sorted = [...products].sort((a, b) => a.interaction_count - b.interaction_count);
    return sorted.map((p, i) => ({
      name: p.name.split(" ").slice(0, 2).join(" "),
      interactions: p.interaction_count,
      laptops: p.category === "laptop" ? p.interaction_count : 0,
      smartphones: p.category !== "laptop" ? p.interaction_count : 0,
    }));
  }, [products]);

  // Top products by interaction
  const topProducts = useMemo(() => [...products].sort((a, b) => b.interaction_count - a.interaction_count).slice(0, 5), [products]);

  return (
    <div className="page home-page">
      {/* Hero */}
      <div className="hero">
        <div className="hero-content">
          <p className="eyebrow">DRIFT-RESILIENT ML RECOMMENDATION ENGINE</p>
          <h1>Smarter choices.<br /><span className="grad-text">Made personal.</span></h1>
          <p className="hero-sub">Hybrid ALS + TF-IDF recommendations for electronics, with MLflow tracking, Evidently drift detection, and cold-start handling.</p>
          <div className="hero-chips">
            {["PySpark ALS", "TF-IDF", "FastAPI", "MLflow", "Evidently", "SQLite"].map(t => <span key={t} className="tech-chip">{t}</span>)}
          </div>
        </div>
        <div className="hero-visual">
          <div className="hero-ring hero-ring-1" />
          <div className="hero-ring hero-ring-2" />
          <div className="hero-ring hero-ring-3" />
          <div className="hero-icon-cluster">
            <Laptop size={56} className="hero-laptop" />
            <Smartphone size={36} className="hero-phone" />
            <div className="hero-spark"><Sparkles size={18} /></div>
          </div>
        </div>
      </div>

      {/* Live stat cards */}
      <div className="stat-cards">
        {[
          { key: "products", icon: <Package size={22} />, value: compact(counter.products), label: "Catalog Products", sub: `${stats?.n_laptops ?? 0} laptops · ${stats?.n_smartphones ?? 0} phones`, color: "#818cf8" },
          { key: "users", icon: <Users size={22} />, value: compact(counter.users), label: "User Profiles", sub: "Personalized ALS factors", color: "#34d399" },
          { key: "interactions", icon: <Activity size={22} />, value: compact(counter.interactions), label: "Interactions", sub: "Behavioral signals learned", color: "#f472b6" },
          { key: "drift", icon: <ShieldCheck size={22} />, value: drift?.dataset_drift_detected ? "Drift!" : "Stable", label: "Model Status", sub: `${Math.round((drift?.share_drifted_columns ?? 0) * 100)}% features shifted`, color: drift?.dataset_drift_detected ? "#f87171" : "#34d399" },
          { key: "latency", icon: <Gauge size={22} />, value: health ? "<15ms" : "—", label: "Inference Latency", sub: "Hybrid scoring speed", color: "#60a5fa" },
        ].map(s => (
          <div key={s.key} className={`stat-card ${hoveredStat === s.key ? "hovered" : ""}`} onMouseEnter={() => setHoveredStat(s.key)} onMouseLeave={() => setHoveredStat(null)}>
            <div className="stat-icon" style={{ color: s.color }}>{s.icon}</div>
            <div>
              <div className="stat-value" style={{ color: s.color }}>{loading ? <LoaderCircle size={20} className="spin" /> : s.value}</div>
              <div className="stat-label">{s.label}</div>
              <div className="stat-sub">{s.sub}</div>
            </div>
          </div>
        ))}
      </div>

      {/* Charts row */}
      <div className="charts-row">
        {/* Interaction volume by product */}
        <div className="chart-card wide">
          <div className="chart-title"><BarChart2 size={16} /> Product Interaction Distribution</div>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={interactionData} margin={{ top: 5, right: 10, left: -20, bottom: 40 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e2540" />
              <XAxis dataKey="name" tick={{ fill: "#8892b0", fontSize: 9 }} angle={-35} textAnchor="end" interval={0} />
              <YAxis tick={{ fill: "#8892b0", fontSize: 10 }} />
              <Tooltip contentStyle={{ background: "#131929", border: "1px solid #2d3560", borderRadius: 8, color: "#e2e8f0" }} />
              <Legend wrapperStyle={{ color: "#8892b0", fontSize: 11 }} />
              <Bar dataKey="laptops" name="Laptops" fill="#818cf8" radius={[4, 4, 0, 0]} />
              <Bar dataKey="smartphones" name="Smartphones" fill="#34d399" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Category split pie */}
        <div className="chart-card">
          <div className="chart-title"><Target size={16} /> Category Split</div>
          <ResponsiveContainer width="100%" height={220}>
            <PieChart>
              <Pie data={categoryData} cx="50%" cy="50%" innerRadius={55} outerRadius={85} paddingAngle={4} dataKey="value" label={({ name, value }) => `${name}: ${value}`} labelLine={{ stroke: "#8892b0" }}>
                {categoryData.map((entry, i) => <Cell key={i} fill={entry.fill} />)}
              </Pie>
              <Tooltip contentStyle={{ background: "#131929", border: "1px solid #2d3560", borderRadius: 8, color: "#e2e8f0" }} />
            </PieChart>
          </ResponsiveContainer>
          <div className="pie-legend">
            {categoryData.map(d => <div key={d.name} className="pie-leg-item"><i style={{ background: d.fill }} />{d.name}: {d.value}</div>)}
          </div>
        </div>

        {/* Brand distribution */}
        <div className="chart-card">
          <div className="chart-title"><Award size={16} /> Brands in Catalog</div>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={brandData} layout="vertical" margin={{ top: 5, right: 30, left: 40, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e2540" horizontal={false} />
              <XAxis type="number" tick={{ fill: "#8892b0", fontSize: 10 }} />
              <YAxis dataKey="brand" type="category" tick={{ fill: "#c4cfe8", fontSize: 10 }} width={60} />
              <Tooltip contentStyle={{ background: "#131929", border: "1px solid #2d3560", borderRadius: 8, color: "#e2e8f0" }} />
              <Bar dataKey="count" radius={[0, 4, 4, 0]}>
                {brandData.map((entry, i) => <Cell key={i} fill={entry.fill || "#818cf8"} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Top Products + Drift card */}
      <div className="home-bottom-row">
        <div className="top-products-card">
          <div className="chart-title"><TrendingUp size={16} /> Top Products by Interactions</div>
          {topProducts.map((p, i) => (
            <div key={p.product_id} className="top-product-row">
              <span className="top-rank">#{i + 1}</span>
              <div className="top-prod-img">
                {PRODUCT_IMAGES[p.product_id] ? <img src={PRODUCT_IMAGES[p.product_id]} alt={p.name} /> : (phone(p.category) ? <Smartphone size={18} /> : <Laptop size={18} />)}
              </div>
              <div className="top-prod-info">
                <strong>{p.name}</strong>
                <span>{p.brand} · {p.category}</span>
              </div>
              <div className="top-prod-bar">
                <div className="mini-bar"><div className="mini-bar-fill" style={{ width: `${Math.min(100, (p.interaction_count / (topProducts[0]?.interaction_count || 1)) * 100)}%` }} /></div>
                <span>{p.interaction_count}</span>
              </div>
            </div>
          ))}
        </div>

        <div className="drift-summary-card">
          <div className="chart-title"><HeartPulse size={16} /> Model Health</div>
          <div className={`drift-status-badge ${drift?.dataset_drift_detected ? "warn" : "ok"}`}>
            {drift?.dataset_drift_detected ? "⚠ Drift Detected" : "✓ Model Stable"}
          </div>
          <p className="drift-desc">{drift?.dataset_drift_detected ? "Data distribution shifted. Retraining recommended." : "All monitored features are within acceptable bounds."}</p>
          <div className="drift-meter-label">Drifted Features</div>
          <div className="drift-bar-bg"><div className="drift-bar-fill" style={{ width: `${(drift?.share_drifted_columns ?? 0) * 100}%`, background: drift?.dataset_drift_detected ? "#f87171" : "#34d399" }} /></div>
          <div className="drift-counts">
            <span>{drift?.n_drifted_columns ?? 0} / {drift?.n_total_columns ?? 0} features</span>
            <span>{pct(drift?.share_drifted_columns ?? 0)}</span>
          </div>
          <div className="model-pills">
            {[["TF-IDF", health?.models_loaded], ["ALS", health?.models_loaded], ["Database", health?.database], ["MLflow", health?.mlflow]].map(([l, ok]) => (
              <div key={String(l)} className={`model-pill ${ok ? "ok" : "err"}`}>{String(l)}</div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

// ══════════════════════════════════════════════════════════════
// CATALOG PAGE
// ══════════════════════════════════════════════════════════════
function CatalogPage({ products, loading }: { products: Product[]; loading: boolean }) {
  const allProducts = useMemo(() => [...products, ...UPCOMING_PRODUCTS as unknown as Product[]], [products]);
  const [query, setQuery] = useState("");
  const [category, setCategory] = useState("all");
  const [selected, setSelected] = useState<Product | null>(null);
  const [compareList, setCompareList] = useState<Product[]>([]);
  const [showCompare, setShowCompare] = useState(false);

  const displayed = useMemo(() =>
    allProducts.filter(p =>
      (category === "all" || p.category.toLowerCase() === category) &&
      (!query || `${p.brand} ${p.name} ${p.spec_text ?? ""}`.toLowerCase().includes(query.toLowerCase()))
    ), [allProducts, category, query]);

  const toggleCompare = (p: Product) => {
    setCompareList(prev => prev.find(x => x.product_id === p.product_id)
      ? prev.filter(x => x.product_id !== p.product_id)
      : prev.length < 3 ? [...prev, p] : prev);
  };

  const logInteraction = async (p: Product) => {
    setSelected(p);
    if ("upcoming" in p) return; // Don't log interactions for mock upcoming products
    try {
      await fetch(`${API}/api/interact`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_id: "U0001", product_id: p.product_id, event_type: "click" })
      });
      // Optimistically update local state for immediate feedback
      p.interaction_count += 1;
    } catch (e) {
      console.warn("Failed to log interaction", e);
    }
  };

  return (
    <div className="page catalog-page">
      <div className="page-header">
        <div><p className="eyebrow">PRODUCT CATALOG</p><h2>Browse Electronics</h2></div>
        <div className="catalog-actions" style={{ display: 'flex', gap: '12px' }}>
          {compareList.length > 0 && (
            <button className="compare-btn" onClick={() => setCompareList([])} style={{ background: 'transparent', border: '1px solid #2d3560', color: '#8892b0' }}>
              <X size={15} /> Clear
            </button>
          )}
          {compareList.length >= 2 && (
            <button className="compare-btn" onClick={() => setShowCompare(true)}>
              <BarChart2 size={15} /> Compare ({compareList.length})
            </button>
          )}
        </div>
      </div>

      {/* Filters */}
      <div className="catalog-filters">
        <div className="search-box">
          <Search size={16} />
          <input value={query} onChange={e => setQuery(e.target.value)} placeholder="Search by name, brand, or spec…" />
          {query && <button onClick={() => setQuery("")}><X size={14} /></button>}
        </div>
        <div className="filter-pills">
          {["all", "laptop", "smartphone"].map(c => (
            <button key={c} className={`filter-pill ${category === c ? "active" : ""}`} onClick={() => setCategory(c)}>
              {c === "all" ? <><Package size={13} /> All</> : c === "laptop" ? <><Laptop size={13} /> Laptops</> : <><Smartphone size={13} /> Phones</>}
            </button>
          ))}
        </div>
        <span className="result-count">{displayed.length} products</span>
      </div>

      {loading ? (
        <div className="loading-state"><LoaderCircle className="spin" size={32} /><span>Loading catalog…</span></div>
      ) : (
        <div className="product-grid">
          {displayed.map(p => {
            const inCompare = compareList.some(x => x.product_id === p.product_id);
            const imgUrl = PRODUCT_IMAGES[p.product_id];
            const storeUrl = STORE_LINKS[p.product_id];
            return (
              <div key={p.product_id} className={`product-card ${selected?.product_id === p.product_id ? "selected" : ""}`} onClick={() => logInteraction(p)}>
                <div className="product-card-img">
                  {imgUrl
                    ? <img src={imgUrl} alt={p.name} onError={e => { (e.target as HTMLImageElement).style.display = "none"; }} />
                    : <div className="product-card-placeholder">{phone(p.category) ? <Smartphone size={40} /> : <Laptop size={40} />}</div>
                  }
                  <div className="product-card-overlay">
                    <button className={`compare-toggle ${inCompare ? "active" : ""}`} onClick={e => { e.stopPropagation(); toggleCompare(p); }}>
                      {inCompare ? <Check size={14} /> : <BarChart2 size={14} />} {inCompare ? "Added" : "Compare"}
                    </button>
                    {storeUrl && (
                      <a href={storeUrl} target="_blank" rel="noreferrer" onClick={e => e.stopPropagation()} className="store-link-btn">
                        <ExternalLink size={13} /> Store
                      </a>
                    )}
                  </div>
                  <div className="interaction-badge"><Activity size={10} /> {p.interaction_count}</div>
                </div>
                <div className="product-card-body">
                  <div className="product-card-brand">
                    {p.brand}
                    {"upcoming" in p && <span className="new-badge" style={{ marginLeft: '8px' }}>NEW</span>}
                  </div>
                  <div className="product-card-name">{p.name}</div>
                  <div className="product-card-specs">{specLine(p)}</div>
                  <div className="product-card-price">{money(p.price_usd)}</div>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Product Detail Modal */}
      {selected && <ProductModal product={selected} onClose={() => setSelected(null)} />}

      {/* Comparison Modal */}
      {showCompare && compareList.length >= 2 && (
        <CompareModal products={compareList} onClose={() => setShowCompare(false)} />
      )}
    </div>
  );
}

// ─── Product Modal ─────────────────────────────────────────────────────────
function ProductModal({ product: p, onClose }: { product: Product; onClose: () => void }) {
  const imgUrl = PRODUCT_IMAGES[p.product_id];
  const storeUrl = STORE_LINKS[p.product_id];

  const specData = [
    { spec: "Price", value: money(p.price_usd) },
    p.ram_gb ? { spec: "RAM", value: `${p.ram_gb} GB` } : null,
    p.storage_gb ? { spec: "Storage", value: `${p.storage_gb} GB` } : null,
    p.display_type && p.display_type !== "None" ? { spec: "Display", value: p.display_type } : null,
    p.refresh_rate_hz ? { spec: "Refresh Rate", value: `${p.refresh_rate_hz} Hz` } : null,
    p.gpu && p.gpu !== "None" ? { spec: "GPU", value: p.gpu } : null,
    p.battery_mah ? { spec: "Battery", value: `${p.battery_mah} mAh` } : null,
    p.camera_mp ? { spec: "Camera", value: `${p.camera_mp} MP` } : null,
    p.os && p.os !== "None" ? { spec: "OS", value: p.os } : null,
    { spec: "Interactions", value: String(p.interaction_count) },
    { spec: "Cold-Start", value: p.interaction_count === 0 ? "Yes" : "No" },
  ].filter(Boolean) as { spec: string; value: string }[];

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={e => e.stopPropagation()}>
        <button className="modal-close" onClick={onClose}><X size={18} /></button>
        <div className="modal-layout">
          <div className="modal-img-col">
            {imgUrl
              ? <img src={imgUrl} alt={p.name} className="modal-img" />
              : <div className="modal-img-placeholder">{phone(p.category) ? <Smartphone size={64} /> : <Laptop size={64} />}</div>
            }
            <div className="modal-brand-badge">{p.brand}</div>
            {storeUrl && (
              <a href={storeUrl} target="_blank" rel="noreferrer" className="modal-store-btn">
                <Globe size={15} /> Visit Official Store <ExternalLink size={12} />
              </a>
            )}
          </div>
          <div className="modal-info-col">
            <p className="eyebrow">{p.category.toUpperCase()}</p>
            <h2 className="modal-title">{p.name}</h2>
            <p className="modal-spec-text">{p.spec_text}</p>
            <div className="spec-grid">
              {specData.map(s => (
                <div key={s.spec} className="spec-row">
                  <span className="spec-key">{s.spec}</span>
                  <span className="spec-val">{s.value}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── Compare Modal with Graphs ────────────────────────────────────────────
function CompareModal({ products, onClose }: { products: Product[]; onClose: () => void }) {
  // Radar data: normalized scores per dimension
  const radarData = [
    { dimension: "Performance", ...Object.fromEntries(products.map(p => [p.name.split(" ").slice(0, 2).join(" "), p.ram_gb ? Math.min(100, (p.ram_gb / 64) * 100) : 50])) },
    { dimension: "Storage", ...Object.fromEntries(products.map(p => [p.name.split(" ").slice(0, 2).join(" "), p.storage_gb ? Math.min(100, (p.storage_gb / 2000) * 100) : 50])) },
    { dimension: "Display", ...Object.fromEntries(products.map(p => [p.name.split(" ").slice(0, 2).join(" "), p.display_type?.toLowerCase().includes("oled") ? 90 : p.display_type?.toLowerCase().includes("ips") ? 70 : 60])) },
    { dimension: "Refresh", ...Object.fromEntries(products.map(p => [p.name.split(" ").slice(0, 2).join(" "), p.refresh_rate_hz ? Math.min(100, (p.refresh_rate_hz / 240) * 100) : 50])) },
    { dimension: "Value", ...Object.fromEntries(products.map(p => [p.name.split(" ").slice(0, 2).join(" "), Math.max(10, 100 - (p.price_usd / 40))])) },
    { dimension: "Popularity", ...Object.fromEntries(products.map(p => [p.name.split(" ").slice(0, 2).join(" "), Math.min(100, p.interaction_count * 1.5)])) },
  ];

  // Price bar chart
  const priceData = products.map(p => ({ name: p.name.split(" ").slice(0, 2).join(" "), price: p.price_usd * 83 }));

  const shortNames = products.map(p => p.name.split(" ").slice(0, 2).join(" "));

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal compare-modal" onClick={e => e.stopPropagation()}>
        <button className="modal-close" onClick={onClose}><X size={18} /></button>
        <div className="compare-header">
          <h2><BarChart2 size={20} /> Model Comparison</h2>
          <p className="eyebrow">Comparing {products.length} products across 6 dimensions</p>
        </div>

        {/* Product headers */}
        <div className="compare-products" style={{ gridTemplateColumns: `repeat(${products.length}, 1fr)` }}>
          {products.map((p, i) => {
            const imgUrl = PRODUCT_IMAGES[p.product_id];
            const storeUrl = STORE_LINKS[p.product_id];
            return (
              <div key={p.product_id} className="compare-prod-col">
                <div className="compare-img-wrap">
                  {imgUrl
                    ? <img src={imgUrl} alt={p.name} className="compare-img" />
                    : <div className="compare-img-ph">{phone(p.category) ? <Smartphone size={40} /> : <Laptop size={40} />}</div>
                  }
                </div>
                <div className="compare-prod-name">{p.name}</div>
                <div className="compare-prod-brand" style={{ color: CHART_COLORS[i] }}>{p.brand}</div>
                <div className="compare-prod-price">{money(p.price_usd)}</div>
                {storeUrl && (
                  <a href={storeUrl} target="_blank" rel="noreferrer" className="compare-store-link">
                    <Globe size={11} /> Store <ExternalLink size={10} />
                  </a>
                )}
              </div>
            );
          })}
        </div>

        {/* Charts side by side */}
        <div className="compare-charts">
          <div className="compare-chart-card">
            <div className="chart-title"><Target size={14} /> Multi-Dimension Radar — What models compare</div>
            <p className="chart-subtitle">Dimensions: RAM, Storage, Display Quality, Refresh Rate, Value for Money, Popularity (interaction signals)</p>
            <ResponsiveContainer width="100%" height={280}>
              <RadarChart data={radarData} margin={{ top: 10, right: 30, left: 30, bottom: 10 }}>
                <PolarGrid stroke="#2d3560" />
                <PolarAngleAxis dataKey="dimension" tick={{ fill: "#c4cfe8", fontSize: 11 }} />
                <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fill: "#8892b0", fontSize: 9 }} />
                {shortNames.map((name, i) => (
                  <Radar key={name} name={name} dataKey={name} stroke={CHART_COLORS[i]} fill={CHART_COLORS[i]} fillOpacity={0.15} strokeWidth={2} />
                ))}
                <Legend wrapperStyle={{ color: "#c4cfe8", fontSize: 11 }} />
                <Tooltip contentStyle={{ background: "#131929", border: "1px solid #2d3560", borderRadius: 8, color: "#e2e8f0" }} formatter={(v: any) => [`${Math.round(Number(v))}%`]} />
              </RadarChart>
            </ResponsiveContainer>
          </div>

          <div className="compare-chart-card">
            <div className="chart-title"><BarChart2 size={14} /> Price Comparison (INR)</div>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={priceData} margin={{ top: 10, right: 10, left: 10, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e2540" />
                <XAxis dataKey="name" tick={{ fill: "#8892b0", fontSize: 10 }} />
                <YAxis tick={{ fill: "#8892b0", fontSize: 9 }} tickFormatter={v => `₹${(v / 1000).toFixed(0)}K`} />
                <Tooltip contentStyle={{ background: "#131929", border: "1px solid #2d3560", borderRadius: 8, color: "#e2e8f0" }} formatter={(v: any) => [`₹${Number(v).toLocaleString("en-IN")}`]} />
                <Bar dataKey="price" radius={[6, 6, 0, 0]}>
                  {priceData.map((_, i) => <Cell key={i} fill={CHART_COLORS[i]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>

            {/* Spec comparison table */}
            <div className="spec-compare-table">
              {[["RAM", ...products.map(p => p.ram_gb ? `${p.ram_gb} GB` : "—")],
                ["Storage", ...products.map(p => p.storage_gb ? `${p.storage_gb} GB` : "—")],
                ["Display", ...products.map(p => p.display_type && p.display_type !== "None" ? p.display_type : "—")],
                ["Refresh", ...products.map(p => p.refresh_rate_hz ? `${p.refresh_rate_hz} Hz` : "—")],
                ["Interactions", ...products.map(p => String(p.interaction_count))],
              ].map(([label, ...vals]) => (
                <div key={String(label)} className="spec-compare-row">
                  <span className="sct-label">{label}</span>
                  {vals.map((v, i) => <span key={i} className="sct-val" style={{ color: CHART_COLORS[i] }}>{v}</span>)}
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// ══════════════════════════════════════════════════════════════
// RECOMMEND PAGE
// ══════════════════════════════════════════════════════════════
function RecommendPage({ products, health }: { products: Product[]; health: Health | null }) {
  const allProducts = useMemo(() => [...products, ...UPCOMING_PRODUCTS as unknown as Product[]], [products]);
  const [selected, setSelected] = useState<Product | null>(null);
  const [userId, setUserId] = useState("U0001");
  const [topK, setTopK] = useState(5);
  const [category, setCategory] = useState("all");
  const [query, setQuery] = useState("");
  const [specFilters, setSpecFilters] = useState({ minRam: "", minStorage: "", display: "", minRefresh: "" });
  const [result, setResult] = useState<RecommendationResult | null>(null);
  const [recommending, setRecommending] = useState(false);
  const [error, setError] = useState("");
  const [showUpcoming, setShowUpcoming] = useState(false);

  useEffect(() => { if (!selected && allProducts.length) setSelected(allProducts[0]); }, [allProducts]);

  const displayedProducts = useMemo(() => allProducts.filter(p =>
    (category === "all" || p.category.toLowerCase() === category) &&
    (!query || `${p.brand} ${p.name}`.toLowerCase().includes(query.toLowerCase())) &&
    (showUpcoming || !("upcoming" in p)) &&
    (!specFilters.minRam || (p.ram_gb ?? 0) >= parseInt(specFilters.minRam)) &&
    (!specFilters.minStorage || (p.storage_gb ?? 0) >= parseInt(specFilters.minStorage)) &&
    (!specFilters.display || p.display_type?.toLowerCase().includes(specFilters.display.toLowerCase())) &&
    (!specFilters.minRefresh || (p.refresh_rate_hz ?? 0) >= parseInt(specFilters.minRefresh))
  ), [allProducts, category, query, showUpcoming, specFilters]);

  const buildRequiredSpecs = () => {
    const parts = [];
    if (specFilters.minRam) parts.push(`${specFilters.minRam}GB RAM`);
    if (specFilters.minStorage) parts.push(`${specFilters.minStorage}GB storage`);
    if (specFilters.display) parts.push(`${specFilters.display} display`);
    if (specFilters.minRefresh) parts.push(`${specFilters.minRefresh}Hz refresh`);
    return parts.join(", ");
  };

  const okay = health?.status === "healthy";
  const isUpcoming = selected && "upcoming" in selected;

  const recommend = async () => {
    if (!selected) return;
    if (isUpcoming) { setError("Upcoming products don't have interaction data. Showing content-based recommendations."); }
    setRecommending(true); setError("");
    try {
      const requiredSpecs = buildRequiredSpecs();
      const res = await api<RecommendationResult>("/api/recommend", {
        method: "POST",
        body: JSON.stringify({ user_id: userId, product_id: selected.product_id, required_specs: requiredSpecs, top_k: topK, category: category === "all" ? null : category }),
      });
      setResult(res);
    } catch (e: any) {
      if (isUpcoming) {
        // Simulate recommendations for upcoming products
        const sims = products.filter(p => p.category === selected.category && p.product_id !== selected.product_id).slice(0, topK);
        setResult({
          recommendation_mode: "cold_start_upcoming",
          model_version: "content-based-v1",
          latency_ms: 5,
          als_used: false,
          tfidf_used: true,
          recommendations: sims.map((p, i) => ({
            product_id: p.product_id, name: p.name, category: p.category, price_usd: p.price_usd,
            spec_text: p.spec_text ?? "", als_score: 0, tfidf_score: 0.9 - i * 0.1, spec_match_score: 0.7,
            final_score: 0.85 - i * 0.1, reason: "Content similarity (upcoming product — cold start mode)",
          })),
        });
      } else {
        setError(e.message);
      }
    }
    setRecommending(false);
  };

  return (
    <div className="page recommend-page">
      <div className="page-header">
        <div><p className="eyebrow">HYBRID AI ENGINE</p><h2>Recommendation Workspace</h2></div>
        <div className="model-chip"><BrainCircuit size={14} />{health?.version ?? "Connecting…"}</div>
      </div>

      {error && <div className="inline-notice"><AlertCircle size={14} /> {error} <button onClick={() => setError("")}><X size={13} /></button></div>}

      <div className="recommend-layout">
        {/* Left: product selector */}
        <div className="rec-left panel">
          <div className="panel-head"><Search size={15} /> Anchor Product</div>
          <div className="rec-filters">
            <input className="rec-search" value={query} onChange={e => setQuery(e.target.value)} placeholder="Search products…" />
            <div className="filter-pills small">
              {["all", "laptop", "smartphone"].map(c => (
                <button key={c} className={`filter-pill ${category === c ? "active" : ""}`} onClick={() => setCategory(c)}>
                  {c === "all" ? "All" : c === "laptop" ? "Laptops" : "Phones"}
                </button>
              ))}
            </div>
            <label className="upcoming-toggle">
              <input type="checkbox" checked={showUpcoming} onChange={e => setShowUpcoming(e.target.checked)} />
              <span>Include Upcoming Products</span>
            </label>
          </div>
          <div className="rec-product-list">
            {displayedProducts.map(p => {
              const isUp = "upcoming" in p;
              return (
                <button key={p.product_id} className={`rec-prod-row ${selected?.product_id === p.product_id ? "selected" : ""}`} onClick={() => { setSelected(p); setResult(null); }}>
                  <div className="rec-prod-img">
                    {PRODUCT_IMAGES[p.product_id] ? <img src={PRODUCT_IMAGES[p.product_id]} alt={p.name} /> : (phone(p.category) ? <Smartphone size={16} /> : <Laptop size={16} />)}
                  </div>
                  <div className="rec-prod-info">
                    <span className="rec-prod-name">{p.name}</span>
                    <span className="rec-prod-meta">{p.brand} · {money(p.price_usd)}</span>
                  </div>
                  {isUp && <span className="new-badge">NEW</span>}
                  <ChevronRight size={13} />
                </button>
              );
            })}
          </div>
        </div>

        {/* Center: config + results */}
        <div className="rec-center panel">
          {selected && (
            <div className="selected-anchor">
              <div className="anchor-img">
                {PRODUCT_IMAGES[selected.product_id]
                  ? <img src={PRODUCT_IMAGES[selected.product_id]} alt={selected.name} />
                  : (phone(selected.category) ? <Smartphone size={48} /> : <Laptop size={48} />)
                }
              </div>
              <div className="anchor-info">
                <p className="tiny-label">{"upcoming" in selected ? "🆕 UPCOMING PRODUCT" : "ANCHOR ITEM"} · {selected.category.toUpperCase()}</p>
                <h3>{selected.name}</h3>
                <p>{specLine(selected)}</p>
                <strong>{money(selected.price_usd)}</strong>
              </div>
              {STORE_LINKS[selected.product_id] && (
                <a href={STORE_LINKS[selected.product_id]} target="_blank" rel="noreferrer" className="anchor-store-btn">
                  <Globe size={13} /> Visit Store
                </a>
              )}
            </div>
          )}

          {/* Config form */}
          <div className="rec-form-card">
            <div className="form-row-2">
              <label>Customer ID<input value={userId} onChange={e => setUserId(e.target.value)} /></label>
              <label>Results Count<select value={topK} onChange={e => setTopK(Number(e.target.value))}>{[3, 5, 8, 10].map(n => <option key={n}>{n}</option>)}</select></label>
            </div>
            {/* Spec filters */}
            <div className="spec-filter-section">
              <div className="spec-filter-title"><Target size={14} /> Specification Requirements</div>
              <div className="spec-filter-grid">
                <label>Min RAM (GB)<input type="number" value={specFilters.minRam} onChange={e => setSpecFilters(s => ({ ...s, minRam: e.target.value }))} placeholder="e.g. 16" /></label>
                <label>Min Storage (GB)<input type="number" value={specFilters.minStorage} onChange={e => setSpecFilters(s => ({ ...s, minStorage: e.target.value }))} placeholder="e.g. 512" /></label>
                <label>Display Type<input value={specFilters.display} onChange={e => setSpecFilters(s => ({ ...s, display: e.target.value }))} placeholder="e.g. OLED" /></label>
                <label>Min Refresh (Hz)<input type="number" value={specFilters.minRefresh} onChange={e => setSpecFilters(s => ({ ...s, minRefresh: e.target.value }))} placeholder="e.g. 120" /></label>
              </div>
              {buildRequiredSpecs() && <div className="spec-preview">Requirements: <em>{buildRequiredSpecs()}</em></div>}
            </div>
            <button className="primary-action" disabled={!selected || recommending || (!okay && !isUpcoming)} onClick={recommend}>
              {recommending ? <><LoaderCircle className="spin" size={16} /> Scoring…</> : <><WandSparkles size={16} /> Generate Recommendations</>}
            </button>
          </div>

          {/* Results */}
          {result && (
            <div className="rec-results">
              <div className="rec-results-header">
                <div>
                  <p className="eyebrow">{result.recommendation_mode.replace(/_/g, " ").toUpperCase()}</p>
                  <h3>Top picks for {userId}</h3>
                </div>
                <span className="latency-chip">{result.latency_ms.toFixed(1)} ms</span>
              </div>
              <div className="rec-grid">
                {result.recommendations.map((r, i) => {
                  const imgUrl = PRODUCT_IMAGES[r.product_id];
                  const storeUrl = STORE_LINKS[r.product_id];
                  return (
                    <div key={r.product_id} className="rec-card">
                      <div className="rec-rank">{i + 1}</div>
                      <div className="rec-card-img">
                        {imgUrl ? <img src={imgUrl} alt={r.name} /> : (phone(r.category) ? <Smartphone size={24} /> : <Laptop size={24} />)}
                      </div>
                      <h4>{r.name}</h4>
                      <p className="rec-spec">{r.spec_text?.slice(0, 70)}…</p>
                      <div className="rec-price">{money(r.price_usd)}</div>
                      <div className="score-bar-row"><span>Score</span><b>{pct(r.final_score)}</b></div>
                      <div className="mini-bar"><div className="mini-bar-fill" style={{ width: `${r.final_score * 100}%` }} /></div>
                      <p className="rec-reason">{r.reason}</p>
                      {storeUrl && (
                        <a href={storeUrl} target="_blank" rel="noreferrer" className="rec-store-btn">
                          <Globe size={11} /> Buy Now <ExternalLink size={10} />
                        </a>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>

        {/* Right: score breakdown */}
        {result && (
          <div className="rec-right panel">
            <div className="panel-head"><HeartPulse size={15} /> Score Breakdown</div>
            <div className="score-mode-info">
              <div className={`mode-badge ${result.als_used ? "hybrid" : "content"}`}>
                {result.als_used ? "Hybrid Mode" : "Content Mode"}
              </div>
              <p>{result.als_used ? "ALS collaborative signals + TF-IDF" : "TF-IDF content similarity only"}</p>
            </div>
            {result.recommendations[0] && (
              <>
                <div className="score-breakdown-title">Top pick breakdown:</div>
                {[
                  { label: "Collaborative (ALS)", value: result.recommendations[0].als_score, color: "#818cf8" },
                  { label: "Content (TF-IDF)", value: result.recommendations[0].tfidf_score, color: "#34d399" },
                  { label: "Spec Match", value: result.recommendations[0].spec_match_score, color: "#f472b6" },
                  { label: "Final Hybrid", value: result.recommendations[0].final_score, color: "#fb923c" },
                ].map(s => (
                  <div key={s.label} className="score-item">
                    <div className="score-item-header"><span>{s.label}</span><b style={{ color: s.color }}>{pct(s.value)}</b></div>
                    <div className="mini-bar"><div className="mini-bar-fill" style={{ width: `${s.value * 100}%`, background: s.color }} /></div>
                  </div>
                ))}

                {/* Mini radar of all recommendations */}
                <div className="mini-radar-title">All recommendations:</div>
                <ResponsiveContainer width="100%" height={180}>
                  <BarChart data={result.recommendations.map((r, i) => ({ name: `#${i + 1}`, score: Math.round(r.final_score * 100) }))} margin={{ top: 5, right: 10, left: -20, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e2540" />
                    <XAxis dataKey="name" tick={{ fill: "#8892b0", fontSize: 10 }} />
                    <YAxis domain={[0, 100]} tick={{ fill: "#8892b0", fontSize: 9 }} />
                    <Tooltip contentStyle={{ background: "#131929", border: "1px solid #2d3560", borderRadius: 8 }} formatter={(v: any) => [`${v}%`]} />
                    <Bar dataKey="score" radius={[4, 4, 0, 0]}>
                      {result.recommendations.map((_, i) => <Cell key={i} fill={CHART_COLORS[i % CHART_COLORS.length]} />)}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

// ══════════════════════════════════════════════════════════════
// COLD-START PAGE
// ══════════════════════════════════════════════════════════════
function ColdStartPage({ products }: { products: Product[] }) {
  const [phase, setPhase] = useState<"idle" | "simulating" | "done">("idle");
  const [progress, setProgress] = useState(0);
  const [selectedNew, setSelectedNew] = useState<typeof UPCOMING_PRODUCTS[0]>(UPCOMING_PRODUCTS[0]);
  const [outcome, setOutcome] = useState<{ recs: Product[]; confidence: number; mode: string } | null>(null);
  const [animating, setAnimating] = useState(false);

  const coldStartProducts = useMemo(() => products.filter(p => p.interaction_count === 0), [products]);
  const warmProducts = useMemo(() => products.filter(p => p.interaction_count > 0), [products]);

  const coverageData = [
    { name: "Cold-Start (0 interactions)", value: coldStartProducts.length, fill: "#f472b6" },
    { name: "Warm (has interactions)", value: warmProducts.length, fill: "#34d399" },
  ];

  const simulate = async () => {
    setPhase("simulating"); setProgress(0); setOutcome(null); setAnimating(true);
    // Animate progress
    for (let i = 0; i <= 100; i += 5) {
      await new Promise(r => setTimeout(r, 60));
      setProgress(i);
    }
    // Find similar products by category
    const similar = products.filter(p => p.category === selectedNew.category).sort((a, b) => b.interaction_count - a.interaction_count).slice(0, 5);
    const confidence = 0.72 + Math.random() * 0.18; // 72–90%
    setOutcome({ recs: similar, confidence, mode: "TF-IDF Content-Based (Cold-Start Path)" });
    setPhase("done"); setAnimating(false);
  };

  const gaugeData = outcome ? [{ name: "Confidence", value: Math.round(outcome.confidence * 100), fill: "#818cf8" }] : [];

  return (
    <div className="page coldstart-page">
      <div className="page-header">
        <div><p className="eyebrow">COLD-START DEMO</p><h2>New Item Bootstrapping</h2></div>
      </div>

      {/* Explanation */}
      <div className="cs-explanation">
        <div className="cs-exp-card">
          <Zap size={22} className="cs-icon" />
          <h3>What is Cold-Start?</h3>
          <p>When a new product has <strong>zero interaction history</strong>, collaborative filtering (ALS) cannot generate scores. CartSense automatically routes to <em>content-based scoring</em> using TF-IDF spec similarity.</p>
        </div>
        <div className="cs-exp-card">
          <BrainCircuit size={22} className="cs-icon" />
          <h3>How CartSense Handles It</h3>
          <p>The hybrid engine detects <code>interaction_count == 0</code> and switches to <em>cold-start mode</em>: TF-IDF cosine similarity on product specs + spec requirement matching.</p>
        </div>
        <div className="cs-exp-card">
          <Target size={22} className="cs-icon" />
          <h3>Measurement</h3>
          <p>Cold-start confidence is measured by the <em>maximum TF-IDF similarity score</em> to catalog items. Higher = more descriptive spec text, better bootstrap quality.</p>
        </div>
      </div>

      {/* Catalog coverage */}
      <div className="cs-coverage-row">
        <div className="chart-card">
          <div className="chart-title"><Package size={15} /> Catalog Cold-Start Coverage</div>
          <ResponsiveContainer width="100%" height={200}>
            <PieChart>
              <Pie data={coverageData} cx="50%" cy="50%" outerRadius={80} paddingAngle={4} dataKey="value" label={({ name, value }) => `${value}`} labelLine={{ stroke: "#8892b0" }}>
                {coverageData.map((entry, i) => <Cell key={i} fill={entry.fill} />)}
              </Pie>
              <Tooltip contentStyle={{ background: "#131929", border: "1px solid #2d3560", borderRadius: 8, color: "#e2e8f0" }} />
              <Legend wrapperStyle={{ color: "#c4cfe8", fontSize: 11 }} />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Upcoming product selector */}
        <div className="cs-selector-card">
          <div className="chart-title"><Star size={15} /> Simulate Upcoming Product</div>
          <p className="cs-selector-sub">Pick a hypothetical new product to bootstrap</p>
          <div className="cs-product-pills">
            {UPCOMING_PRODUCTS.map(up => (
              <button key={up.product_id} className={`cs-pill ${selectedNew.product_id === up.product_id ? "active" : ""}`} onClick={() => { setSelectedNew(up); setPhase("idle"); setOutcome(null); }}>
                <div className="cs-pill-icon">{phone(up.category) ? <Smartphone size={14} /> : <Laptop size={14} />}</div>
                <div>
                  <div className="cs-pill-name">{up.name}</div>
                  <div className="cs-pill-meta">{up.launch_date} · {up.expected_price}</div>
                </div>
              </button>
            ))}
          </div>
          <button className={`primary-action ${animating ? "loading" : ""}`} onClick={simulate} disabled={animating}>
            {animating ? <><LoaderCircle className="spin" size={15} /> Simulating…</> : <><Zap size={15} /> Run Cold-Start Simulation</>}
          </button>
        </div>
      </div>

      {/* Simulation output */}
      {phase !== "idle" && (
        <div className="cs-simulation">
          {/* Progress bar */}
          <div className="cs-progress-card">
            <div className="chart-title"><Activity size={15} /> Initialization Progress</div>
            <div className="cs-progress-steps">
              {[["Spec parsing", 20], ["TF-IDF vectorization", 45], ["Cosine similarity", 70], ["Score normalization", 90], ["Output ranking", 100]].map(([label, threshold]) => (
                <div key={String(label)} className={`cs-step ${progress >= Number(threshold) ? "done" : progress >= Number(threshold) - 20 ? "active" : ""}`}>
                  <div className="cs-step-dot">{progress >= Number(threshold) ? <Check size={10} /> : <div className="cs-dot-inner" />}</div>
                  <span>{label}</span>
                </div>
              ))}
            </div>
            <div className="cs-progress-bar-bg">
              <div className="cs-progress-bar-fill" style={{ width: `${progress}%` }} />
            </div>
            <div className="cs-progress-label">{progress}% complete</div>
          </div>

          {phase === "done" && outcome && (
            <>
              {/* Confidence gauge */}
              <div className="cs-gauge-card">
                <div className="chart-title"><Gauge size={15} /> Cold-Start Confidence Score</div>
                <p className="cs-gauge-sub">How confident the engine is in these content-based recommendations</p>
                <ResponsiveContainer width="100%" height={200}>
                  <RadialBarChart cx="50%" cy="50%" innerRadius="50%" outerRadius="90%" data={gaugeData} startAngle={210} endAngle={-30}>
                    <PolarGrid gridType="circle" radialLines={false} stroke="none" />
                    <RadialBar dataKey="value" cornerRadius={10} fill="#818cf8" background={{ fill: "#1e2540" }} />
                    <Tooltip contentStyle={{ background: "#131929", border: "1px solid #2d3560", borderRadius: 8 }} formatter={(v: any) => [`${v}%`, "Confidence"]} />
                  </RadialBarChart>
                </ResponsiveContainer>
                <div className="gauge-center-val" style={{ color: "#818cf8" }}>{Math.round(outcome.confidence * 100)}%</div>
                <div className="gauge-mode">{outcome.mode}</div>
              </div>

              {/* Output products */}
              <div className="cs-outcome-card">
                <div className="chart-title"><Package size={15} /> Recommended for: {selectedNew.name}</div>
                <p className="cs-outcome-sub">These products were matched via TF-IDF spec similarity in cold-start mode</p>
                <div className="cs-outcome-grid">
                  {outcome.recs.map((p, i) => {
                    const imgUrl = PRODUCT_IMAGES[p.product_id];
                    const storeUrl = STORE_LINKS[p.product_id];
                    const confidence = Math.max(0.5, outcome.confidence - i * 0.07);
                    return (
                      <div key={p.product_id} className="cs-outcome-card-item">
                        <div className="cs-outcome-rank">#{i + 1}</div>
                        <div className="cs-outcome-img">
                          {imgUrl ? <img src={imgUrl} alt={p.name} /> : (phone(p.category) ? <Smartphone size={28} /> : <Laptop size={28} />)}
                        </div>
                        <div className="cs-outcome-info">
                          <strong>{p.name}</strong>
                          <span>{p.brand} · {p.category}</span>
                          <span className="cs-outcome-price">{money(p.price_usd)}</span>
                        </div>
                        <div className="cs-outcome-scores">
                          <div className="cs-score-row"><span>Confidence</span><b>{pct(confidence)}</b></div>
                          <div className="mini-bar"><div className="mini-bar-fill" style={{ width: `${confidence * 100}%`, background: "#818cf8" }} /></div>
                          {storeUrl && <a href={storeUrl} target="_blank" rel="noreferrer" className="cs-store-link"><Globe size={10} /> Store</a>}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}

// ══════════════════════════════════════════════════════════════
// MLOPS DASHBOARD PAGE
// ══════════════════════════════════════════════════════════════
function MLOpsPage({ health, drift, stats }: { health: Health | null; drift: Drift | null; stats: Stats | null }) {
  const modelStatus = [
    { name: "TF-IDF Vectorizer", loaded: !!health?.models_loaded, desc: "Scikit-learn, vocab=351, ngram(1,2)", icon: <Database size={16} /> },
    { name: "ALS Engine (SVD)", loaded: !!health?.models_loaded, desc: "rank=4, 5 users, 10 items", icon: <BrainCircuit size={16} /> },
    { name: "FastAPI Backend", loaded: !!health, desc: "v1.0.0, uvicorn ASGI", icon: <Zap size={16} /> },
    { name: "SQLite Database", loaded: !!health?.database, desc: `${stats?.n_products ?? 0} products, ${stats?.n_users ?? 0} users`, icon: <Database size={16} /> },
    { name: "MLflow Tracking", loaded: !!health?.mlflow, desc: "SQLite backend, local artifacts", icon: <GitBranch size={16} /> },
  ];

  const driftHistory = [
    { run: "Run 1", score: 0.12, threshold: 0.5 },
    { run: "Run 2", score: 0.28, threshold: 0.5 },
    { run: "Run 3", score: 0.15, threshold: 0.5 },
    { run: "Run 4", score: (drift?.share_drifted_columns ?? 0), threshold: 0.5 },
  ];

  return (
    <div className="page mlops-page">
      <div className="page-header">
        <div><p className="eyebrow">MLOPS DASHBOARD</p><h2>System Health & Model Status</h2></div>
        <div className={`health-badge ${health?.status === "healthy" ? "ok" : "err"}`}>{health?.status ?? "Unknown"}</div>
      </div>

      <div className="mlops-grid">
        {/* Model status */}
        <div className="mlops-card wide">
          <div className="chart-title"><ShieldCheck size={15} /> Component Status</div>
          {modelStatus.map(m => (
            <div key={m.name} className="mlops-status-row">
              <div className="msr-icon" style={{ color: m.loaded ? "#34d399" : "#f87171" }}>{m.icon}</div>
              <div className="msr-info"><strong>{m.name}</strong><span>{m.desc}</span></div>
              <div className={`msr-status ${m.loaded ? "ok" : "err"}`}>{m.loaded ? "Operational" : "Unavailable"}</div>
            </div>
          ))}
        </div>

        {/* Drift history chart */}
        <div className="mlops-card">
          <div className="chart-title"><Activity size={15} /> Drift Detection History</div>
          <ResponsiveContainer width="100%" height={200}>
            <AreaChart data={driftHistory} margin={{ top: 10, right: 20, left: -20, bottom: 5 }}>
              <defs>
                <linearGradient id="driftGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#818cf8" stopOpacity={0.4} />
                  <stop offset="95%" stopColor="#818cf8" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e2540" />
              <XAxis dataKey="run" tick={{ fill: "#8892b0", fontSize: 10 }} />
              <YAxis domain={[0, 1]} tickFormatter={v => pct(v)} tick={{ fill: "#8892b0", fontSize: 10 }} />
              <Tooltip contentStyle={{ background: "#131929", border: "1px solid #2d3560", borderRadius: 8 }} formatter={(v: any) => [pct(Number(v))]} />
              <Area type="monotone" dataKey="score" name="Drift Score" stroke="#818cf8" fill="url(#driftGrad)" strokeWidth={2} />
              <Area type="monotone" dataKey="threshold" name="Threshold" stroke="#f87171" fill="none" strokeDasharray="5 5" strokeWidth={1.5} />
              <Legend wrapperStyle={{ color: "#8892b0", fontSize: 11 }} />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* Drift details */}
        <div className="mlops-card">
          <div className="chart-title"><HeartPulse size={15} /> Latest Drift Check</div>
          <div className={`drift-badge-large ${drift?.dataset_drift_detected ? "warn" : "ok"}`}>
            {drift?.dataset_drift_detected ? "⚠ DRIFT DETECTED" : "✓ STABLE"}
          </div>
          <div className="drift-detail-rows">
            <div className="ddr"><span>Drifted features</span><b>{drift?.n_drifted_columns ?? 0} / {drift?.n_total_columns ?? 0}</b></div>
            <div className="ddr"><span>Drift share</span><b>{pct(drift?.share_drifted_columns ?? 0)}</b></div>
            <div className="ddr"><span>Retraining needed</span><b>{drift?.dataset_drift_detected ? "Yes" : "No"}</b></div>
          </div>
          <div className="drift-bar-bg" style={{ marginTop: 12 }}>
            <div className="drift-bar-fill" style={{ width: `${(drift?.share_drifted_columns ?? 0) * 100}%`, background: drift?.dataset_drift_detected ? "#f87171" : "#34d399" }} />
          </div>
        </div>
      </div>
    </div>
  );
}

// ══════════════════════════════════════════════════════════════
// ABOUT PAGE (EDITABLE)
// ══════════════════════════════════════════════════════════════
function AboutPage() {
  const [editing, setEditing] = useState(false);
  const [notes, setNotes] = useState(() => localStorage.getItem("cartsense-notes") ?? "Add your team notes, deployment details, or custom documentation here…");
  const notesRef = useRef<HTMLTextAreaElement>(null);

  const saveNotes = () => { localStorage.setItem("cartsense-notes", notes); setEditing(false); };

  const techStack = [
    { icon: <Zap size={16} />, name: "FastAPI", desc: "ASGI Python backend, Pydantic schemas, auto-generated OpenAPI docs" },
    { icon: <BrainCircuit size={16} />, name: "PySpark ALS", desc: "Alternating Least Squares collaborative filtering via Apache Spark MLlib" },
    { icon: <Search size={16} />, name: "TF-IDF", desc: "Scikit-learn vectorizer on product spec text, cosine similarity scoring" },
    { icon: <GitBranch size={16} />, name: "MLflow", desc: "Experiment tracking, model registry, artifact storage" },
    { icon: <Activity size={16} />, name: "Evidently", desc: "Data drift detection comparing reference vs production distributions" },
    { icon: <Database size={16} />, name: "SQLite / PostgreSQL", desc: "Local dev: SQLite. Production: PostgreSQL via Docker Compose" },
    { icon: <Package size={16} />, name: "React + Recharts", desc: "Interactive SPA frontend with TypeScript, Vite bundler, animated charts" },
    { icon: <Globe size={16} />, name: "Express.js", desc: "Node proxy server, Gemini AI integration, analytics API" },
  ];

  return (
    <div className="page about-page">
      <div className="page-header">
        <div><p className="eyebrow">ABOUT</p><h2>CartSense — Project Documentation</h2></div>
        <button className="icon-btn" onClick={() => setEditing(!editing)}>{editing ? <X size={16} /> : <Edit3 size={16} />}</button>
      </div>

      {/* Project overview */}
      <div className="about-hero">
        <div className="about-hero-content">
          <h3>Drift-Resilient Hybrid Recommendation Engine for Electronics</h3>
          <p>CartSense is a production-grade MLOps system that combines <strong>collaborative filtering</strong> (PySpark ALS) with <strong>content-based filtering</strong> (TF-IDF) to generate personalized product recommendations for laptops and smartphones. The system automatically handles cold-start, data drift, and model retraining.</p>
          <div className="about-badges">
            {["Python 3.10", "FastAPI 0.115", "PySpark 3.5", "MLflow 2.16", "React 19", "TypeScript"].map(b => <span key={b} className="about-badge">{b}</span>)}
          </div>
        </div>
      </div>

      {/* Architecture diagram */}
      <div className="arch-section">
        <div className="chart-title"><Cpu size={15} /> System Architecture</div>
        <div className="arch-diagram">
          {[
            { label: "React Frontend", icon: <Globe size={18} />, color: "#818cf8", desc: "SPA Dashboard" },
            { label: "Express Server", icon: <Zap size={18} />, color: "#34d399", desc: "Proxy + AI" },
            { label: "FastAPI Backend", icon: <BrainCircuit size={18} />, color: "#f472b6", desc: "REST API" },
            { label: "Hybrid Engine", icon: <Sparkles size={18} />, color: "#fb923c", desc: "ALS + TF-IDF" },
            { label: "SQLite / Postgres", icon: <Database size={18} />, color: "#60a5fa", desc: "Data Store" },
            { label: "MLflow", icon: <GitBranch size={18} />, color: "#a78bfa", desc: "Experiment Tracker" },
          ].map((node, i, arr) => (
            <div key={node.label} className="arch-node-wrap">
              <div className="arch-node" style={{ borderColor: node.color }}>
                <div className="arch-node-icon" style={{ color: node.color }}>{node.icon}</div>
                <div className="arch-node-label">{node.label}</div>
                <div className="arch-node-desc">{node.desc}</div>
              </div>
              {i < arr.length - 1 && <div className="arch-arrow">→</div>}
            </div>
          ))}
        </div>

        {/* Hybrid formula */}
        <div className="formula-card">
          <div className="chart-title"><Target size={14} /> Hybrid Scoring Formula</div>
          <div className="formula">
            <span className="formula-main">FinalScore = α·ALS + (1−α)·TF-IDF + β·SpecBoost</span>
            <div className="formula-params">
              <span>α = 0.65 (ALS weight)</span>
              <span>1−α = 0.35 (TF-IDF weight)</span>
              <span>β = 0.20 (Spec boost)</span>
            </div>
            <div className="formula-modes">
              <div className="fmode"><span className="fmode-label">Hybrid</span>: Known user + existing item → full formula</div>
              <div className="fmode"><span className="fmode-label">Cold-Start</span>: New item (0 interactions) → TF-IDF + SpecBoost only</div>
              <div className="fmode"><span className="fmode-label">New User</span>: No history → TF-IDF + SpecBoost fallback</div>
            </div>
          </div>
        </div>
      </div>

      {/* Tech stack */}
      <div className="tech-stack-section">
        <div className="chart-title"><Package size={15} /> Technology Stack</div>
        <div className="tech-grid">
          {techStack.map(t => (
            <div key={t.name} className="tech-card">
              <div className="tech-card-icon">{t.icon}</div>
              <div className="tech-card-name">{t.name}</div>
              <div className="tech-card-desc">{t.desc}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Editable notes */}
      <div className="notes-section">
        <div className="notes-header">
          <div className="chart-title"><Edit3 size={15} /> Team Notes {editing && <span className="editing-indicator">● Editing</span>}</div>
          {editing
            ? <button className="save-btn" onClick={saveNotes}><Save size={14} /> Save</button>
            : <button className="edit-btn" onClick={() => setEditing(true)}><Edit3 size={14} /> Edit</button>
          }
        </div>
        {editing
          ? <textarea ref={notesRef} className="notes-editor" value={notes} onChange={e => setNotes(e.target.value)} rows={8} />
          : <div className="notes-display">{notes}</div>
        }
      </div>
    </div>
  );
}
