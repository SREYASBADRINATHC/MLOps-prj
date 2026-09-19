import express from "express";
import path from "path";
import { createServer as createViteServer } from "vite";
import { GoogleGenAI } from "@google/genai";
import dotenv from "dotenv";

dotenv.config();

// ==========================================================================
// SEARCH ANALYTICS — Customer search pattern tracker
// ==========================================================================
interface SearchEvent {
  query: string;
  resolvedName: string;
  category: string;
  timestamp: number;
  brand: string;
  persona?: string;
}

const searchAnalytics: SearchEvent[] = [];

function extractBrand(name: string): string {
  const brands = [
    "Apple", "Samsung", "Google", "OnePlus", "Xiaomi", "Realme", "Vivo", "Oppo",
    "Motorola", "Nokia", "Sony", "LG", "Lenovo", "Dell", "HP", "ASUS", "Acer",
    "Microsoft", "Huawei", "Nothing", "iQOO", "Poco", "Redmi", "Honor",
    "Infinix", "Tecno", "Lava", "Micromax", "Razer", "MSI"
  ];
  const nameLower = name.toLowerCase();
  return brands.find(b => nameLower.includes(b.toLowerCase())) || "Other";
}

function recordSearch(query: string, resolvedName: string, category: string) {
  const brand = extractBrand(resolvedName);
  searchAnalytics.push({
    query: query.trim(),
    resolvedName,
    category,
    timestamp: Date.now(),
    brand
  });
  // Keep last 200 events
  if (searchAnalytics.length > 200) searchAnalytics.shift();
  console.log(`[Analytics] Recorded search: "${query}" → "${resolvedName}" (${brand}, ${category})`);
}

function getSearchInsights() {
  const now = Date.now();
  const recent = searchAnalytics.filter(e => now - e.timestamp < 24 * 60 * 60 * 1000); // last 24h
  
  // Brand frequency
  const brandCount: Record<string, number> = {};
  const categoryCount: Record<string, number> = {};
  const queryCount: Record<string, number> = {};
  
  recent.forEach(e => {
    brandCount[e.brand] = (brandCount[e.brand] || 0) + 1;
    categoryCount[e.category] = (categoryCount[e.category] || 0) + 1;
    queryCount[e.query.toLowerCase()] = (queryCount[e.query.toLowerCase()] || 0) + 1;
  });

  const topBrands = Object.entries(brandCount).sort((a, b) => b[1] - a[1]).slice(0, 5);
  const topCategories = Object.entries(categoryCount).sort((a, b) => b[1] - a[1]).slice(0, 5);
  const trendingQueries = Object.entries(queryCount).sort((a, b) => b[1] - a[1]).slice(0, 10);

  return {
    totalSearches: recent.length,
    topBrands,
    topCategories,
    trendingQueries,
    recentSearches: recent.slice(-10).reverse()
  };
}

// ==========================================================================
// LAPTOP & PHONE PRODUCT KNOWLEDGE BASE
// Used for local smart matching without any API call
// ==========================================================================
interface KnownProduct {
  name: string;
  category: "Laptop" | "Mobile Phone";
  brand: string;
  price: string;
  specs: string;
  link: string;
  keywords: string[];
}

const PRODUCT_KB: KnownProduct[] = [
  // ─── LAPTOPS ──────────────────────────────────────────────────────────────
  {
    name: "Apple MacBook Pro 16\" M4 Max", category: "Laptop", brand: "Apple",
    price: "₹3,49,900.00",
    specs: "M4 Max (16-core CPU, 40-core GPU), 48GB Unified Memory, 1TB SSD, Liquid Retina XDR 16.2\".",
    link: "https://www.apple.com/in/macbook-pro/",
    keywords: ["macbook pro 16", "macbook pro m4 max", "apple laptop max", "mac pro 16"]
  },
  {
    name: "Apple MacBook Pro 14\" M4 Pro", category: "Laptop", brand: "Apple",
    price: "₹1,99,900.00",
    specs: "M4 Pro (12-core CPU, 16-core GPU), 24GB Unified Memory, 512GB SSD, Liquid Retina XDR 14.2\".",
    link: "https://www.apple.com/in/macbook-pro/",
    keywords: ["macbook pro 14", "macbook pro m4", "macbook m4 pro", "mac pro 14"]
  },
  {
    name: "Apple MacBook Air 13\" M3", category: "Laptop", brand: "Apple",
    price: "₹1,14,900.00",
    specs: "M3 (8-core CPU, 10-core GPU), 16GB Unified Memory, 512GB SSD, 13.6\" Liquid Retina, fanless.",
    link: "https://www.apple.com/in/macbook-air/",
    keywords: ["macbook air", "macbook air m3", "macbook air 13", "mac air"]
  },
  {
    name: "Lenovo ThinkPad X1 Carbon Gen 13 AI", category: "Laptop", brand: "Lenovo",
    price: "₹1,94,900.00",
    specs: "Intel Core Ultra 9 185H with NPU, 32GB LPDDR5X, 1TB SSD, 14\" 2.8K OLED 120Hz, 1.09kg.",
    link: "https://www.lenovo.com/in/en/p/laptops/thinkpad/thinkpadx1/",
    keywords: ["thinkpad", "thinkpad x1", "lenovo thinkpad", "x1 carbon", "lenovo laptop"]
  },
  {
    name: "Dell XPS 15 (2025)", category: "Laptop", brand: "Dell",
    price: "₹1,79,900.00",
    specs: "Intel Core Ultra 9 285H, NVIDIA RTX 4070 (8GB), 32GB DDR5, 1TB SSD, 15.6\" OLED 4K 120Hz.",
    link: "https://www.dell.com/en-in/shop/laptops/xps-15-laptop/spd/xps-15-9530-laptop",
    keywords: ["xps 15", "dell xps 15", "dell laptop", "xps15"]
  },
  {
    name: "Dell XPS 13 Copilot+ Snapdragon", category: "Laptop", brand: "Dell",
    price: "₹1,39,900.00",
    specs: "Snapdragon X Elite (12-core), 16GB LPDDR5X, 512GB SSD, 13.4\" FHD+ InfinityEdge, 27h battery.",
    link: "https://www.dell.com/en-in/shop/laptops/xps-13-laptop/spd/xps-13-9340-laptop",
    keywords: ["xps 13", "dell xps 13", "xps13", "dell xps"]
  },
  {
    name: "ASUS ROG Zephyrus G16 (2025)", category: "Laptop", brand: "ASUS",
    price: "₹2,39,900.00",
    specs: "AMD Ryzen AI 9 HX 370, NVIDIA RTX 4090 (16GB), 32GB DDR5, 2TB SSD, 16\" QHD+ OLED 240Hz.",
    link: "https://rog.asus.com/laptops/rog-zephyrus/rog-zephyrus-g16-2024-series/",
    keywords: ["rog zephyrus", "asus rog", "zephyrus g16", "asus gaming laptop", "rog laptop"]
  },
  {
    name: "HP Omen 16 (2025)", category: "Laptop", brand: "HP",
    price: "₹1,59,900.00",
    specs: "Intel Core i9-14900HX, NVIDIA RTX 4080 (12GB), 32GB DDR5, 1TB SSD, 16\" QHD 165Hz.",
    link: "https://www.hp.com/in-en/shop/pdp/omen-16-gaming-laptop",
    keywords: ["hp omen", "omen 16", "omen laptop", "hp gaming laptop", "hp omen 16"]
  },
  {
    name: "HP Spectre x360 14\" (2025)", category: "Laptop", brand: "HP",
    price: "₹1,49,900.00",
    specs: "Intel Core Ultra 7 155H, 32GB LPDDR5, 1TB SSD, 14\" 2.8K OLED Touch 120Hz, 2-in-1 convertible.",
    link: "https://www.hp.com/in-en/shop/pdp/hp-spectre-x360-14-eu0095tu",
    keywords: ["hp spectre", "spectre x360", "hp spectre 14", "hp 2-in-1"]
  },
  {
    name: "Microsoft Surface Laptop 7 (Snapdragon)", category: "Laptop", brand: "Microsoft",
    price: "₹1,44,900.00",
    specs: "Snapdragon X Elite X1E-80-100, 32GB LPDDR5X, 1TB SSD, 15\" PixelSense 2496×1664 120Hz.",
    link: "https://www.microsoft.com/en-in/surface/devices/surface-laptop-7th-edition",
    keywords: ["surface laptop", "microsoft surface", "surface laptop 7", "surface laptop snapdragon"]
  },
  {
    name: "Razer Blade 16 (2025)", category: "Laptop", brand: "Razer",
    price: "₹3,19,900.00",
    specs: "Intel Core i9-14900HX, NVIDIA RTX 4090 (16GB), 32GB DDR5, 2TB SSD, 16\" QHD+ OLED 240Hz.",
    link: "https://www.razer.com/gaming-laptops/razer-blade-16",
    keywords: ["razer blade", "razer laptop", "razer blade 16", "razer gaming"]
  },
  {
    name: "MSI Titan GT77 HX (2025)", category: "Laptop", brand: "MSI",
    price: "₹3,49,900.00",
    specs: "Intel Core i9-14900HX, NVIDIA RTX 4090 (16GB), 64GB DDR5, 4TB SSD, 17.3\" UHD 144Hz.",
    link: "https://www.msi.com/Laptop/Titan-GT77-HX-13V",
    keywords: ["msi titan", "msi laptop", "msi gaming", "msi gt77"]
  },
  {
    name: "Samsung Galaxy Book4 Ultra", category: "Laptop", brand: "Samsung",
    price: "₹2,29,900.00",
    specs: "Intel Core Ultra 9 185H, NVIDIA RTX 4070 (8GB), 32GB LPDDR5X, 1TB SSD, 16\" 3K 120Hz AMOLED.",
    link: "https://www.samsung.com/in/computers/galaxy-book/galaxy-book4-ultra-16-np960xfh-xa1in/",
    keywords: ["galaxy book", "samsung laptop", "galaxy book4 ultra", "samsung galaxy book"]
  },
  {
    name: "ASUS Vivobook Pro 16 OLED", category: "Laptop", brand: "ASUS",
    price: "₹1,24,900.00",
    specs: "AMD Ryzen 9 7945HX, NVIDIA RTX 4060 (8GB), 16GB DDR5, 1TB SSD, 16\" 3.2K OLED 120Hz.",
    link: "https://www.asus.com/in/laptops/for-creators/vivobook/asus-vivobook-pro-16-oled-k6604/",
    keywords: ["vivobook pro", "asus vivobook", "vivobook oled", "asus oled laptop"]
  },
  {
    name: "Acer Predator Helios 18 (2025)", category: "Laptop", brand: "Acer",
    price: "₹2,49,900.00",
    specs: "Intel Core i9-14900HX, NVIDIA RTX 4090 (16GB), 32GB DDR5, 2TB SSD, 18\" QHD+ 250Hz IPS.",
    link: "https://www.acer.com/in-en/laptops/predator/predator-helios-18",
    keywords: ["predator helios", "acer predator", "helios 18", "acer gaming laptop"]
  },

  // ─── MOBILE PHONES ────────────────────────────────────────────────────────
  {
    name: "Apple iPhone 17 Pro Max", category: "Mobile Phone", brand: "Apple",
    price: "₹1,59,900.00",
    specs: "Apple A19 Pro, 6.9\" ProMotion 120Hz Super Retina XDR, Under-Display Face ID, Titanium build.",
    link: "https://www.apple.com/in/iphone/",
    keywords: ["iphone 17 pro max", "iphone 17 pro", "iphone 17", "apple iphone 17", "iphone pro max"]
  },
  {
    name: "Apple iPhone 16 Pro Max", category: "Mobile Phone", brand: "Apple",
    price: "₹1,44,900.00",
    specs: "Apple A18 Pro, 6.9\" Super Retina XDR ProMotion 120Hz, Camera Control, Titanium frame.",
    link: "https://www.apple.com/in/iphone-16-pro/",
    keywords: ["iphone 16 pro max", "iphone 16 pro", "iphone 16", "apple iphone 16", "iphone pro"]
  },
  {
    name: "Apple iPhone 16 Plus", category: "Mobile Phone", brand: "Apple",
    price: "₹89,900.00",
    specs: "Apple A18, 6.7\" Super Retina XDR 60Hz, aluminum build, 27h video playback.",
    link: "https://www.apple.com/in/iphone-16/",
    keywords: ["iphone 16 plus", "iphone plus", "iphone 16plus"]
  },
  {
    name: "Samsung Galaxy S25 Ultra", category: "Mobile Phone", brand: "Samsung",
    price: "₹1,30,900.00",
    specs: "Snapdragon 8 Elite, 12GB RAM, 256GB, 6.9\" QHD+ Dynamic AMOLED 120Hz, embedded S-Pen, 200MP.",
    link: "https://www.samsung.com/in/smartphones/galaxy-s25-ultra/",
    keywords: ["galaxy s25 ultra", "s25 ultra", "samsung s25 ultra", "samsung galaxy s25"]
  },
  {
    name: "Samsung Galaxy S25+", category: "Mobile Phone", brand: "Samsung",
    price: "₹99,900.00",
    specs: "Snapdragon 8 Elite, 12GB RAM, 256GB, 6.7\" FHD+ Dynamic AMOLED 2X 120Hz, 50MP camera.",
    link: "https://www.samsung.com/in/smartphones/galaxy-s25/",
    keywords: ["galaxy s25 plus", "s25+", "samsung s25 plus", "galaxy s25"]
  },
  {
    name: "Google Pixel 9 Pro XL", category: "Mobile Phone", brand: "Google",
    price: "₹1,09,900.00",
    specs: "Google Tensor G4, 16GB RAM, 256GB, 6.8\" LTPO OLED 120Hz, 50MP triple camera, Gemini AI built-in.",
    link: "https://store.google.com/in/product/pixel_9_pro",
    keywords: ["pixel 9 pro", "google pixel 9", "pixel 9 pro xl", "google pixel"]
  },
  {
    name: "OnePlus 13", category: "Mobile Phone", brand: "OnePlus",
    price: "₹69,999.00",
    specs: "Snapdragon 8 Elite, 12GB RAM, 256GB, 6.82\" 2K AMOLED 120Hz ProXDR, Hasselblad triple 50MP.",
    link: "https://www.oneplus.in/13",
    keywords: ["oneplus 13", "one plus 13", "oneplus flagship"]
  },
  {
    name: "OnePlus 12", category: "Mobile Phone", brand: "OnePlus",
    price: "₹59,999.00",
    specs: "Snapdragon 8 Gen 3, 12GB RAM, 256GB, 6.82\" 2K AMOLED 120Hz, 50MP Hasselblad cameras.",
    link: "https://www.oneplus.in/12",
    keywords: ["oneplus 12", "one plus 12"]
  },
  {
    name: "Xiaomi 15 Ultra", category: "Mobile Phone", brand: "Xiaomi",
    price: "₹1,09,999.00",
    specs: "Snapdragon 8 Elite, 16GB RAM, 512GB, 6.73\" WQHD+ AMOLED 120Hz, Leica quad camera 200MP.",
    link: "https://www.mi.com/in/xiaomi-15-ultra",
    keywords: ["xiaomi 15 ultra", "mi 15 ultra", "xiaomi ultra", "xiaomi flagship"]
  },
  {
    name: "iQOO 13", category: "Mobile Phone", brand: "iQOO",
    price: "₹54,999.00",
    specs: "Snapdragon 8 Elite, 12GB RAM, 256GB, 6.82\" 2K AMOLED 144Hz, 50MP triple camera.",
    link: "https://www.iqoo.com/in/product/iqoo13.html",
    keywords: ["iqoo 13", "vivo iqoo 13", "iqoo flagship"]
  },
  {
    name: "Nothing Phone (3) Pro", category: "Mobile Phone", brand: "Nothing",
    price: "₹64,999.00",
    specs: "Snapdragon 8 Elite, 12GB RAM, 256GB, 6.67\" AMOLED 120Hz, Glyph Interface 2.0, triple camera.",
    link: "https://in.nothing.tech/products/phone-3-pro",
    keywords: ["nothing phone 3", "nothing phone 3 pro", "nothing phone"]
  },
  {
    name: "Realme GT 7 Pro", category: "Mobile Phone", brand: "Realme",
    price: "₹41,999.00",
    specs: "Snapdragon 8 Elite, 12GB RAM, 256GB, 6.78\" AMOLED 144Hz ProDimming, 50MP triple camera.",
    link: "https://www.realme.com/in/realme-gt7-pro",
    keywords: ["realme gt 7 pro", "realme gt7 pro", "realme flagship"]
  },
  {
    name: "Motorola Edge 50 Ultra", category: "Mobile Phone", brand: "Motorola",
    price: "₹59,999.00",
    specs: "Snapdragon 8s Gen 3, 12GB RAM, 512GB, 6.67\" pOLED 165Hz, 50MP triple Pantone camera.",
    link: "https://www.motorola.in/smartphones-motorola-edge-50-ultra/p",
    keywords: ["motorola edge 50 ultra", "moto edge ultra", "motorola flagship"]
  },
  {
    name: "Vivo X200 Pro", category: "Mobile Phone", brand: "Vivo",
    price: "₹94,999.00",
    specs: "Dimensity 9400, 16GB RAM, 512GB, 6.78\" 2K AMOLED 120Hz, ZEISS 200MP triple camera.",
    link: "https://www.vivo.com/in/products/x200pro",
    keywords: ["vivo x200 pro", "vivo x200", "vivo flagship", "vivo zeiss"]
  },
  {
    name: "Sony Xperia 1 VI", category: "Mobile Phone", brand: "Sony",
    price: "₹1,19,990.00",
    specs: "Snapdragon 8 Gen 3, 12GB RAM, 256GB, 6.5\" 4K OLED 120Hz, triple Zeiss camera, 3.5mm jack.",
    link: "https://www.sony.co.in/en/articles/xperia-1-vi",
    keywords: ["sony xperia 1 vi", "xperia 1 vi", "sony xperia", "sony flagship"]
  },
  {
    name: "ASUS ROG Phone 9 Pro", category: "Mobile Phone", brand: "ASUS",
    price: "₹99,999.00",
    specs: "Snapdragon 8 Elite, 24GB RAM, 1TB, 6.78\" AMOLED 185Hz, AirTrigger 7, GameCool 9 cooling.",
    link: "https://rog.asus.com/phones/rog-phone-9-pro/",
    keywords: ["rog phone 9", "asus rog phone", "rog phone pro", "gaming phone"]
  }
];

// Smart local search — finds the best KB match for a query
function findBestKBMatch(query: string): KnownProduct | null {
  const q = query.toLowerCase().trim();
  
  // Score each product
  let best: KnownProduct | null = null;
  let bestScore = 0;

  for (const product of PRODUCT_KB) {
    let score = 0;
    // Keyword match (exact)
    for (const kw of product.keywords) {
      if (q.includes(kw)) score += 10;
      if (kw.includes(q)) score += 6;
    }
    // Brand match
    if (q.includes(product.brand.toLowerCase())) score += 4;
    // Name match (partial)
    const nameWords = product.name.toLowerCase().split(/\s+/);
    nameWords.forEach(w => { if (q.includes(w) && w.length > 2) score += 2; });

    if (score > bestScore) {
      bestScore = score;
      best = product;
    }
  }

  return bestScore >= 4 ? best : null;
}

// Generate context-aware recommendations for a product from the KB
function generateKBRecommendations(product: KnownProduct) {
  const isLaptop = product.category === "Laptop";
  const isMobile = product.category === "Mobile Phone";

  // Pick relevant candidates (same category, different products)
  const sameCatPool = PRODUCT_KB.filter(p => p.category === product.category && p.name !== product.name);
  const otherCatPool = PRODUCT_KB.filter(p => p.category !== product.category);

  const pickRandom = (pool: KnownProduct[], n: number): KnownProduct[] => {
    const shuffled = [...pool].sort(() => Math.random() - 0.5);
    return shuffled.slice(0, n);
  };

  const toReco = (p: KnownProduct, als: number, tfidf: number) => ({
    id: p.name.toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_|_$/g, ""),
    name: p.name,
    specs: p.specs.substring(0, 100),
    baseALS: als,
    baseTFIDF: tfidf,
    icon: p.category === "Laptop" ? "laptop" : "smartphone",
    category: p.category,
    whyChips: [
      p.brand === product.brand ? "Same brand" : "Top alternative",
      p.category === product.category ? "Same category" : "Cross-category",
      Math.random() > 0.5 ? "Similar price tier" : "Spec match"
    ]
  });

  // Persona-tailored picks
  const gamerLaptops = PRODUCT_KB.filter(p => p.category === "Laptop" && ["ASUS", "Razer", "MSI", "Acer", "HP"].includes(p.brand) && p.name !== product.name).slice(0, 3);
  const officeLaptops = PRODUCT_KB.filter(p => p.category === "Laptop" && ["Apple", "Lenovo", "Dell", "Microsoft", "HP"].includes(p.brand) && p.name !== product.name).slice(0, 3);
  const creativeLaptops = PRODUCT_KB.filter(p => p.category === "Laptop" && ["Apple", "ASUS", "Samsung", "Dell"].includes(p.brand) && p.name !== product.name).slice(0, 3);
  const topPhones = PRODUCT_KB.filter(p => p.category === "Mobile Phone").sort(() => Math.random() - 0.5).slice(0, 3);
  const topLaptops = PRODUCT_KB.filter(p => p.category === "Laptop").sort(() => Math.random() - 0.5).slice(0, 3);
  
  const fallbackSame = pickRandom(sameCatPool, 3);
  const fallbackOther = pickRandom(otherCatPool, 3);

  const fillReco = (arr: KnownProduct[], fallback: KnownProduct[]) => {
    const filled = [...arr];
    while (filled.length < 3) filled.push(fallback[filled.length % fallback.length]);
    return filled.slice(0, 3);
  };

  const gamer = fillReco(isLaptop ? gamerLaptops : topLaptops, fallbackSame);
  const office = fillReco(isLaptop ? officeLaptops : topPhones, fallbackSame);
  const creative = fillReco(isLaptop ? creativeLaptops : topPhones, fallbackSame);
  const enthusiast = fillReco([...sameCatPool].slice(0, 3), fallbackOther);
  const coldStart = fillReco([...sameCatPool].slice(0, 3), fallbackSame);

  return {
    gamer:      gamer.map((p, i)     => toReco(p, 0.85 - i*0.05, 0.78 - i*0.05)),
    office:     office.map((p, i)    => toReco(p, 0.90 - i*0.05, 0.82 - i*0.04)),
    creative:   creative.map((p, i)  => toReco(p, 0.88 - i*0.04, 0.80 - i*0.04)),
    enthusiast: enthusiast.map((p, i)=> toReco(p, 0.92 - i*0.04, 0.86 - i*0.04)),
    coldStart:  coldStart.map((p, i) => toReco(p, 0.00, 0.90 - i*0.05))
  };
}

async function startServer() {
  const app = express();
  const PORT = process.env.PORT ? parseInt(process.env.PORT) : 3000;
  const FASTAPI_URL = process.env.FASTAPI_URL || process.env.VITE_PROXY_TARGET || "http://localhost:8000";

  app.use(express.json());

  // ==========================================================================
  // FASTAPI ML BACKEND PROXY — forwards ML routes to FastAPI on port 8000
  // ==========================================================================
  const http = await import("http");

  function proxyToFastAPI(req: any, res: any, method: string, path: string, body?: any) {
    const url = new URL(path, FASTAPI_URL);
    const postData = body ? JSON.stringify(body) : undefined;
    const options = {
      hostname: url.hostname,
      port: Number(url.port) || (url.protocol === "https:" ? 443 : 80),
      path: url.pathname + (url.search || ""),
      method,
      headers: {
        "Content-Type": "application/json",
        ...(postData ? { "Content-Length": Buffer.byteLength(postData) } : {}),
      },
    };

    const proxyReq = http.request(options, (proxyRes) => {
      res.writeHead(proxyRes.statusCode || 200, proxyRes.headers);
      proxyRes.pipe(res, { end: true });
    });

    proxyReq.on("error", (err) => {
      console.error(`[Proxy] FastAPI error on ${path}:`, err.message);
      res.status(502).json({
        error: "FastAPI backend unavailable",
        detail: "Start FastAPI with: .venv\\Scripts\\uvicorn backend.main:app --port 8000",
        fastapi_url: FASTAPI_URL,
      });
    });

    if (postData) proxyReq.write(postData);
    proxyReq.end();
  }

  // ML product catalog (from SQLite via FastAPI)
  app.get("/api/ml/products", (req, res) => proxyToFastAPI(req, res, "GET", "/api/products"));
  app.get("/api/products", (req, res) => proxyToFastAPI(req, res, "GET", req.originalUrl || "/api/products"));

  // ML hybrid recommendations (TF-IDF + ALS)
  app.post("/api/ml/recommend", (req, res) => proxyToFastAPI(req, res, "POST", "/api/recommend", req.body));
  app.post("/api/recommend", (req, res) => proxyToFastAPI(req, res, "POST", "/api/recommend", req.body));

  // Evidently drift metrics
  app.get("/api/ml/drift", (req, res) => proxyToFastAPI(req, res, "GET", "/api/drift"));
  app.get("/api/drift", (req, res) => proxyToFastAPI(req, res, "GET", "/api/drift"));

  // FastAPI health check
  app.get("/api/ml/health", (req, res) => proxyToFastAPI(req, res, "GET", "/api/health"));
  app.get("/api/health", (req, res) => proxyToFastAPI(req, res, "GET", "/api/health"));

  // Additional FastAPI endpoints
  app.get("/api/stats", (req, res) => proxyToFastAPI(req, res, "GET", "/api/stats"));
  app.get("/api/models/status", (req, res) => proxyToFastAPI(req, res, "GET", "/api/model/status"));
  app.post("/api/retrain", (req, res) => proxyToFastAPI(req, res, "POST", "/api/retrain", req.body));

  // Initialize Gemini
  const apiKey = process.env.GEMINI_API_KEY;
  let ai: GoogleGenAI | null = null;
  if (apiKey) {
    ai = new GoogleGenAI({
      apiKey,
      httpOptions: {
        headers: { 'User-Agent': 'aistudio-build' }
      }
    });
    console.log("[Gemini] API key found — AI grounding enabled.");
  } else {
    console.log("[Gemini] No API key — using local product knowledge base.");
  }

  // ==========================================================================
  // SANITIZE — laptops & phones ONLY, no accessories
  // ==========================================================================
  const ALLOWED_CATEGORIES = ["laptop", "mobile phone", "smartphone", "phone"];
  const BANNED_KEYWORDS = ["keyboard", "headset", "headphone", "earbud", "mouse", "accessory",
    "controller", "joystick", "webcam", "microphone", "speaker", "gpu", "cpu",
    "monitor", "display", "tablet", "ipad", "console", "tv"];

  function sanitizeRecommendations(recos: any) {
    if (!recos) recos = {};

    const laptopFallbacks = [
      { id: "macbook_pro_14_m4", name: "Apple MacBook Pro 14\" M4 Pro", specs: "M4 Pro 12-core, 24GB RAM, Liquid Retina XDR 14\".", baseALS: 0.90, baseTFIDF: 0.88, icon: "laptop", category: "Laptop", whyChips: ["Premium alternative", "Top benchmark"] },
      { id: "thinkpad_x1_gen13", name: "Lenovo ThinkPad X1 Carbon Gen 13", specs: "Intel Ultra 9, 32GB RAM, 1TB SSD, OLED 120Hz.", baseALS: 0.87, baseTFIDF: 0.82, icon: "laptop", category: "Laptop", whyChips: ["Business flagship", "Spec match"] },
      { id: "dell_xps_15_2025", name: "Dell XPS 15 (2025)", specs: "Intel Ultra 9, RTX 4070, 32GB, 4K OLED 15.6\".", baseALS: 0.84, baseTFIDF: 0.79, icon: "laptop", category: "Laptop", whyChips: ["Creator-grade", "OLED display"] }
    ];
    const phoneFallbacks = [
      { id: "iphone_17_pro_max", name: "Apple iPhone 17 Pro Max", specs: "A19 Pro, 6.9\" 120Hz ProMotion, Titanium, Camera Control.", baseALS: 0.91, baseTFIDF: 0.87, icon: "smartphone", category: "Mobile Phone", whyChips: ["Top flagship", "Premium tier"] },
      { id: "galaxy_s25_ultra", name: "Samsung Galaxy S25 Ultra", specs: "Snapdragon 8 Elite, 200MP, 6.9\" QHD+ 120Hz, S-Pen.", baseALS: 0.89, baseTFIDF: 0.85, icon: "smartphone", category: "Mobile Phone", whyChips: ["Android flagship", "S-Pen"] },
      { id: "pixel_9_pro_xl", name: "Google Pixel 9 Pro XL", specs: "Tensor G4, 16GB, 6.8\" OLED, Gemini AI native.", baseALS: 0.85, baseTFIDF: 0.80, icon: "smartphone", category: "Mobile Phone", whyChips: ["Pure AI", "Google camera"] }
    ];

    const personas = ["gamer", "office", "creative", "enthusiast", "coldStart"] as const;

    for (const persona of personas) {
      if (!recos[persona] || !Array.isArray(recos[persona])) {
        recos[persona] = persona === "coldStart" ? [...phoneFallbacks] : [...laptopFallbacks];
        continue;
      }
      recos[persona] = recos[persona].map((item: any, i: number) => {
        if (!item) return { ...(i % 2 === 0 ? laptopFallbacks[i % 3] : phoneFallbacks[i % 3]) };
        const name = (item.name || "").toLowerCase();
        const cat  = (item.category || "").toLowerCase();
        const spec = (item.specs || "").toLowerCase();
        const isBanned = BANNED_KEYWORDS.some(kw => name.includes(kw) || cat.includes(kw) || spec.includes(kw));
        const isAllowed = ALLOWED_CATEGORIES.some(c => cat.includes(c));
        if (isBanned || (!isAllowed && cat !== "")) {
          const fb = i % 2 === 0 ? laptopFallbacks : phoneFallbacks;
          return { ...fb[i % 3] };
        }
        return item;
      });
      while (recos[persona].length < 3) {
        recos[persona].push({ ...laptopFallbacks[recos[persona].length % 3] });
      }
      recos[persona] = recos[persona].slice(0, 3);
    }
    return recos;
  }

  // ==========================================================================
  // GEMINI PROMPT — Laptops & Phones ONLY
  // ==========================================================================
  const SYSTEM_INSTRUCTION = `You are a cutting-edge product intelligence engine specializing in LAPTOPS and MOBILE PHONES only.

Your task: Given a search query, identify the exact latest 2025/2026 flagship model of the laptop or phone.

STRICT RULES:
1. ONLY return Laptop or Mobile Phone products. NEVER recommend tablets, iPads, GPUs, CPUs, monitors, consoles, headphones, keyboards, mice, or any other accessories.
2. For the main product: use Google Search to find the EXACT model name, current INR price, real official URL, and concise specs (under 20 words).
3. For each persona's 3 recommendations, ONLY suggest other flagship laptops or flagship mobile phones.
4. Recommendations must be direct competitors or complementary devices in the laptop/phone space.

Persona recommendation guidelines (ALL must be laptops or phones ONLY):
- gamer: High-performance gaming laptops or gaming-tuned smartphones
- office: Business laptops or productivity-focused smartphones
- creative: Creator laptops with OLED/display or camera-flagship smartphones
- enthusiast: Premium flagship laptops or premium flagship phones
- coldStart: 3 direct competitor laptops/phones to the searched product

Each recommendation item MUST have:
- id: lowercase with underscores, no spaces
- name: full product name
- specs: concise spec highlights (under 15 words)
- baseALS: float 0.60–0.98
- baseTFIDF: float 0.55–0.95
- icon: ONLY "laptop" or "smartphone"
- category: ONLY "Laptop" or "Mobile Phone"
- whyChips: array of 2–3 short reason strings (e.g. ["Same brand", "Spec upgrade", "OLED display"])

Return ONLY raw JSON (no markdown):
{
  "name": "Exact 2025/2026 model name",
  "specs": "Key specs in under 20 words",
  "price": "₹X,XX,XXX.00",
  "category": "Laptop" or "Mobile Phone",
  "link": "https://official-page-or-store.com",
  "recos": {
    "gamer": [ { "id":"...", "name":"...", "specs":"...", "baseALS":0.85, "baseTFIDF":0.78, "icon":"laptop", "category":"Laptop", "whyChips":["..."] }, ... ],
    "office": [...],
    "creative": [...],
    "enthusiast": [...],
    "coldStart": [...]
  }
}`;

  // ==========================================================================
  // API: POST /api/search
  // ==========================================================================
  app.post("/api/search", async (req, res) => {
    const { query } = req.body;
    if (!query || !query.trim()) {
      return res.status(400).json({ success: false, error: "Search query is required." });
    }

    const q = query.trim();

    // --- Step 1: Try local KB match first (instant, zero API calls) ---
    const kbMatch = findBestKBMatch(q);
    
    // --- Step 2: Try Gemini with Google Search grounding ---
    if (ai) {
      try {
        console.log(`[Gemini] Searching: "${q}"`);
        const response = await ai.models.generateContent({
          model: "gemini-2.5-flash",
          contents: `Find the latest flagship laptop or mobile phone for query: "${q}". Return the exact 2025/2026 model.`,
          config: {
            systemInstruction: SYSTEM_INSTRUCTION,
            tools: [{ googleSearch: {} }],
          },
        });

        const rawText = response.text || "";
        let parsed: any;

        try {
          parsed = JSON.parse(rawText.trim());
        } catch {
          const cleaned = rawText.replace(/```json/g, "").replace(/```/g, "").trim();
          // Extract JSON from response
          const jsonMatch = cleaned.match(/\{[\s\S]*\}/);
          if (jsonMatch) {
            parsed = JSON.parse(jsonMatch[0]);
          } else {
            throw new Error("No valid JSON in Gemini response");
          }
        }

        // Enforce laptop/phone only
        if (parsed?.recos) {
          parsed.recos = sanitizeRecommendations(parsed.recos);
        }

        // Add whyChips to recos if missing (backward compat)
        if (parsed?.recos) {
          const personas = ["gamer", "office", "creative", "enthusiast", "coldStart"];
          for (const persona of personas) {
            if (Array.isArray(parsed.recos[persona])) {
              parsed.recos[persona] = parsed.recos[persona].map((r: any) => ({
                ...r,
                whyChips: r.whyChips || ["AI Recommended", "Top match"]
              }));
            }
          }
        }

        // Fix buy link from grounding if missing
        const chunks = response.candidates?.[0]?.groundingMetadata?.groundingChunks;
        const references = (chunks || []).map((c: any) => ({ title: c.web?.title, uri: c.web?.uri })).filter((c: any) => c.uri);
        if (!parsed.link || parsed.link.includes("example.com")) {
          if (references.length > 0) parsed.link = references[0].uri;
        }

        // Record analytics
        recordSearch(q, parsed.name || q, parsed.category || "Unknown");

        console.log(`[Gemini] OK: "${parsed.name}" (${parsed.category})`);
        return res.json({ success: true, product: parsed, references, source: "gemini-grounded" });

      } catch (geminiErr: any) {
        console.log(`[Gemini] Error: ${geminiErr.message}. Falling back to KB/local.`);
      }
    }

    // --- Step 3: Use local KB match ---
    if (kbMatch) {
      console.log(`[KB] Matched: "${kbMatch.name}"`);
      const recos = generateKBRecommendations(kbMatch);
      recordSearch(q, kbMatch.name, kbMatch.category);
      return res.json({
        success: true,
        product: {
          name: kbMatch.name,
          category: kbMatch.category,
          specs: kbMatch.specs,
          price: kbMatch.price,
          link: kbMatch.link,
          recos: sanitizeRecommendations(recos)
        },
        references: [],
        source: "local-kb"
      });
    }

    // --- Step 4: No verified result ---
    return res.status(404).json({
      success: false,
      error: "No verified product match found for the query.",
      source: "no-verified-result"
    });
  });

  // ==========================================================================
  // API: GET /api/analytics — Customer search analysis
  // ==========================================================================
  app.get("/api/analytics", (req, res) => {
    const insights = getSearchInsights();
    res.json({ success: true, insights });
  });

  // ==========================================================================
  // API: GET /api/trending — Top searched products
  // ==========================================================================
  app.get("/api/trending", (req, res) => {
    const insights = getSearchInsights();
    const trending = insights.trendingQueries.map(([query, count]) => ({ query, count }));
    const brands = insights.topBrands.map(([brand, count]) => ({ brand, count }));
    res.json({ success: true, trending, brands, totalSearches: insights.totalSearches });
  });

  // ==========================================================================
  // API: GET /api/catalog — Full laptop + phone catalog from KB
  // ==========================================================================
  app.get("/api/catalog", (req, res) => {
    const laptops = PRODUCT_KB.filter(p => p.category === "Laptop");
    const phones = PRODUCT_KB.filter(p => p.category === "Mobile Phone");
    res.json({ success: true, laptops, phones, total: PRODUCT_KB.length });
  });

  // ==========================================================================
  // STATIC / HTML ROUTES
  // (Removed manual / route so Vite can inject HMR and handle index.html)
  // ==========================================================================

  // STATIC / VITE DEV SERVER
  // ==========================================================================
  if (process.env.NODE_ENV !== "production") {
    try {
      const vite = await createViteServer({
        server: { middlewareMode: true },
        appType: "spa",
      });
      app.use(vite.middlewares);
      app.use("*", async (req, res, next) => {
        try {
          const fs = await import("fs/promises");
          let template = await fs.readFile(path.resolve(process.cwd(), "index.html"), "utf-8");
          template = await vite.transformIndexHtml(req.originalUrl, template);
          res.status(200).set({ "Content-Type": "text/html" }).end(template);
        } catch (e: any) {
          vite.ssrFixStacktrace?.(e);
          next(e);
        }
      });
    } catch (e) {
      // Vite not available — serve static files directly
      app.use(express.static(process.cwd()));
      app.get("*", (req, res) => {
        res.sendFile(path.join(process.cwd(), "index.html"));
      });
    }
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`\n🚀 CartSense Server running on http://localhost:${PORT}`);
    console.log(`📦 Product KB: ${PRODUCT_KB.length} products (${PRODUCT_KB.filter(p=>p.category==="Laptop").length} laptops, ${PRODUCT_KB.filter(p=>p.category==="Mobile Phone").length} phones)`);
    console.log(`🔑 Gemini API: ${apiKey ? "Connected (Google Search Grounding)" : "Not configured (Local KB mode)"}`);
    console.log(`📊 Analytics: http://localhost:${PORT}/api/analytics`);
    console.log(`📈 Trending:  http://localhost:${PORT}/api/trending`);
  });
}

startServer();
