import express from "express";
import path from "path";
import { createServer as createViteServer } from "vite";
import { GoogleGenAI } from "@google/genai";
import dotenv from "dotenv";

dotenv.config();

async function startServer() {
  const app = express();
  const PORT = 3000;

  app.use(express.json());

  // Initialize Gemini
  const apiKey = process.env.GEMINI_API_KEY;
  let ai: GoogleGenAI | null = null;
  if (apiKey) {
    ai = new GoogleGenAI({
      apiKey,
      httpOptions: {
        headers: {
          'User-Agent': 'aistudio-build',
        }
      }
    });
  }

  // Helper to sanitize recommendations and filter out banned items (keyboards, headsets, headphones, mice)
  function sanitizeRecommendations(recos: any) {
    if (!recos) recos = {};
    const bannedKeywords = ["keyboard", "headset", "headphone", "earbud", "mouse", "accessory"];
    
    const replacements = {
      gamer: [
        { id: "rog_ally_x", name: "ASUS ROG Ally X Handheld", specs: "AMD Ryzen Z1 Extreme, 24GB LPDDR5X, 1TB SSD, 120Hz gaming handheld.", baseALS: 0.92, baseTFIDF: 0.68, icon: "gamepad", category: "Console" },
        { id: "rog_swift_pg27", name: "ASUS ROG Swift PG27AQDM OLED Monitor", specs: "27\" 1440p OLED 240Hz gaming monitor.", baseALS: 0.94, baseTFIDF: 0.72, icon: "tv", category: "Display" },
        { id: "rtx_4080s_gpu", name: "NVIDIA GeForce RTX 4080 Super GPU", specs: "16GB GDDR6X, ultra ray tracing, DLSS 3.0 graphic processing.", baseALS: 0.88, baseTFIDF: 0.65, icon: "cpu", category: "GPU" }
      ],
      office: [
        { id: "galaxy_tab_s9", name: "Samsung Galaxy Tab S9 Ultra", specs: "14.6\" Dynamic AMOLED 2X screen, Snapdragon 8 Gen 2, perfect companion.", baseALS: 0.91, baseTFIDF: 0.74, icon: "tablet", category: "Tablet" },
        { id: "dell_ultrasharp_32", name: "Dell UltraSharp 32 4K USB-C Hub Monitor", specs: "IPS Black, 90W power delivery, Daisy Chain productivity screen.", baseALS: 0.95, baseTFIDF: 0.82, icon: "tv", category: "Display" },
        { id: "ipad_pro_13", name: "Apple iPad Pro 13\" M4", specs: "Ultra Retina XDR Tandem OLED, M4 performance chip, super thin.", baseALS: 0.89, baseTFIDF: 0.78, icon: "tablet", category: "Tablet" }
      ],
      creative: [
        { id: "wacom_intuos_pro", name: "Wacom Intuos Pro Medium Graphic Tablet", specs: "Professional pen tablet with 8192 pressure levels for precision.", baseALS: 0.87, baseTFIDF: 0.61, icon: "tablet", category: "Tablet" },
        { id: "apple_studio_display", name: "Apple Studio Display 27\"", specs: "5K Retina, 600 nits, built-in 12MP ultra-wide camera and speakers.", baseALS: 0.93, baseTFIDF: 0.79, icon: "tv", category: "Display" },
        { id: "rtx_4090_gpu", name: "NVIDIA GeForce RTX 4090 Flagship", specs: "24GB GDDR6X, Ada Lovelace, maximum creative rendering power.", baseALS: 0.91, baseTFIDF: 0.84, icon: "cpu", category: "GPU" }
      ],
      enthusiast: [
        { id: "apple_vision_pro", name: "Apple Vision Pro Spatial Computer", specs: "Dual-micro-OLED 4K displays, M2+R1 chips for ultimate spatial workflows.", baseALS: 0.96, baseTFIDF: 0.84, icon: "tv", category: "Display" },
        { id: "ryzen_9950x_cpu", name: "AMD Ryzen 9 9950X CPU Processor", specs: "16 cores, 32 threads, Zen 5 architecture, peak enthusiast performance.", baseALS: 0.92, baseTFIDF: 0.78, icon: "cpu", category: "CPU" },
        { id: "lg_oled_flex", name: "LG OLED Flex 42\" Bendable Smart TV", specs: "42\" bendable 4K OLED screen with 120Hz and smart hub.", baseALS: 0.89, baseTFIDF: 0.69, icon: "tv", category: "Display" }
      ],
      coldStart: [
        { id: "macbook_m4_pro", name: "Apple MacBook Pro 14\" M4 Pro", specs: "Direct competing premium notebook with elite computing capability.", baseALS: 0.75, baseTFIDF: 0.94, icon: "laptop", category: "Laptop" },
        { id: "xps13_snapdragon", name: "Dell XPS 13 Copilot+ Snapdragon", specs: "Ultraportable flagship Windows laptop offering elite battery life.", baseALS: 0.72, baseTFIDF: 0.88, icon: "laptop", category: "Laptop" },
        { id: "pixel_9_pro", name: "Google Pixel 9 Pro XL AI", specs: "Direct premium AI phone competitor with pure Google Gemini integration.", baseALS: 0.68, baseTFIDF: 0.85, icon: "smartphone", category: "Smartphone" }
      ]
    };

    const personas = ["gamer", "office", "creative", "enthusiast", "coldStart"] as const;
    for (const persona of personas) {
      if (!recos[persona] || !Array.isArray(recos[persona])) {
        recos[persona] = [...replacements[persona]];
        continue;
      }

      // Filter out banned items
      recos[persona] = recos[persona].map((item: any, index: number) => {
        if (!item) return { ...replacements[persona][index % 3] };
        
        const name = (item.name || "").toLowerCase();
        const id = (item.id || "").toLowerCase();
        const category = (item.category || "").toLowerCase();
        const specs = (item.specs || "").toLowerCase();

        const isBanned = bannedKeywords.some(kw => 
          name.includes(kw) || id.includes(kw) || category.includes(kw) || specs.includes(kw)
        );

        if (isBanned) {
          // Replace with the approved fallback from replacements list at corresponding index
          return { ...replacements[persona][index % 3] };
        }
        return item;
      });

      // Ensure we have exactly 3 items
      while (recos[persona].length < 3) {
        recos[persona].push({ ...replacements[persona][recos[persona].length % 3] });
      }
      if (recos[persona].length > 3) {
        recos[persona] = recos[persona].slice(0, 3);
      }
    }
    return recos;
  }

  // High-fidelity fallback product generator for offline mode or quota limits
  function getFallbackProduct(query: string) {
    const q = query.toLowerCase();
    let name = "";
    let category = "";
    let price = "";
    let specs = "";
    let link = "";
    
    if (q.includes("keyboard") || q.includes("mouse") || q.includes("headset") || q.includes("headphones")) {
      // Respect the strict constraint of not suggesting keyboards or headsets by showing premium core tablets
      name = "Apple iPad Pro 13\" (M4)";
      category = "Tablet";
      price = "₹1,29,900.00";
      specs = "Ultra Retina XDR Tandem OLED, M4 chip, 512GB, ProMotion, super thin design.";
      link = "https://www.apple.com/in/ipad-pro/";
    } else if (q.includes("iphone") || q.includes("apple phone")) {
      name = "Apple iPhone 17 Pro Max";
      category = "Smartphone";
      price = "₹1,59,900.00";
      specs = "A19 Pro chip, 6.9-inch 120Hz ProMotion screen, Under-Display FaceID, Titanium design.";
      link = "https://www.apple.com/in/iphone/";
    } else if (q.includes("ipad") || q.includes("tablet")) {
      name = "Apple iPad Pro 13\" (M4)";
      category = "Tablet";
      price = "₹1,29,900.00";
      specs = "Ultra Retina XDR, M4 chip, 512GB, ProMotion, super thin design.";
      link = "https://www.apple.com/in/ipad-pro/";
    } else if (q.includes("macbook") || q.includes("apple laptop") || q.includes("mac")) {
      name = "Apple MacBook Pro 16\" (M5 Max)";
      category = "Laptop";
      price = "₹3,49,900.00";
      specs = "M5 Max 16-core CPU, 40-core GPU, 48GB unified memory, 1TB high-speed SSD.";
      link = "https://www.apple.com/in/macbook-pro/";
    } else if (q.includes("sony") || q.includes("headphone") || q.includes("earphone") || q.includes("audio") || q.includes("sound") || q.includes("speaker")) {
      name = "Sony HT-A7000 7.1.2ch Dolby Atmos Soundbar";
      category = "Audio";
      price = "₹1,39,900.00";
      specs = "Flagship soundbar with 360 Spatial Sound Mapping, built-in dual subwoofers.";
      link = "https://www.sony.co.in/electronics/sound-bars/ht-a7000";
    } else if (q.includes("rtx") || q.includes("gpu") || q.includes("nvidia") || q.includes("graphics")) {
      name = "NVIDIA GeForce RTX 5090 Founders Edition";
      category = "GPU";
      price = "₹1,99,900.00";
      specs = "32GB GDDR7, Blackwell architecture, DLSS 4.0 frame synthesis, peak desktop gaming.";
      link = "https://www.nvidia.com/en-in/geforce/graphics-cards/";
    } else if (q.includes("thinkpad") || q.includes("lenovo")) {
      name = "Lenovo ThinkPad X1 Carbon Gen 13 AI";
      category = "Laptop";
      price = "₹1,94,900.00";
      specs = "Intel Core Ultra 9 Gen 2 with integrated NPU, 32GB RAM, 1TB SSD, 120Hz OLED.";
      link = "https://www.lenovo.com/in/en/p/laptops/thinkpad/thinkpadx1/";
    } else if (q.includes("playstation") || q.includes("ps5") || q.includes("xbox") || q.includes("console")) {
      name = "PlayStation 5 Pro";
      category = "Gaming Console";
      price = "₹69,900.00";
      specs = "Upgraded GPU, advanced ray tracing, PlayStation Spectral Super Resolution.";
      link = "https://www.playstation.com/en-in/ps5/";
    } else {
      // General fallback - ensure it returns a modern flagship
      const capitalizedQuery = query.split(' ').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
      name = `${capitalizedQuery} Flagship Pro Max (2026 Edition)`;
      category = "Tech Gear";
      price = "₹99,900.00";
      specs = "Latest high-performance edition with smart features and advanced 2026 specs.";
      link = `https://www.google.com/search?q=buy+${encodeURIComponent(query)}`;
    }

    // Recommendations for the 5 personas
    const recos = sanitizeRecommendations({});

    return {
      name,
      specs,
      price,
      category,
      link,
      recos
    };
  }

  // API Route for Product Search Grounding
  app.post("/api/search", async (req, res) => {
    const { query } = req.body;
    if (!query) {
      return res.status(400).json({ error: "Query is required" });
    }

    // Direct fallback if Gemini is not configured
    if (!ai) {
      console.log("No Gemini API key configured. Executing high-fidelity fallback logic.");
      const fallbackProd = getFallbackProduct(query);
      return res.json({ success: true, product: fallbackProd, references: [], fallback: true });
    }

    try {
      // System instruction for Gemini 3.6-flash with Google Search Grounding to find the latest model of the product
      const systemInstruction = `You are an expert product and retail database crawler. 
Your goal is to identify the latest, most up-to-date flagship model of the product searched by the user (e.g. if the user searches "iphone", return "iPhone 17 Pro Max" or the latest flagship available in 2026).
You must use Google Search to get the exact latest model, specs, price in INR, a standard category, and a real web link to the official product page or major e-commerce store.

CRITICAL CONSTRAINT: You are STRICTLY FORBIDDEN from recommending or suggesting any keyboards, headsets, headphones, audio earbuds, mice, game controllers, or basic input/output accessories under any persona (neither in 'gamer', 'office', 'creative', 'enthusiast', nor 'coldStart'). 
Instead, recommend actual full-fledged devices or core components such as Laptops, Smartphones, Tablets, Monitors, Smartwatches, Graphics Cards (GPUs), CPU Processors, Motherboards, or Smart Speaker Systems.
For example:
- For 'gamer', recommend handheld gaming consoles (e.g. ASUS ROG Ally X), high-end gaming monitors, or elite GPUs (e.g. RTX 4090).
- For 'office', recommend tablet computers (e.g. Samsung Galaxy Tab S9 Ultra), high-efficiency monitors, or premium ultraportable laptops.
- For 'creative', recommend drawing tablets (e.g. Wacom Intuos Pro), pro creator displays (e.g. Apple Studio Display), or workstation graphics processors.
- For 'enthusiast', recommend spatial computing devices (e.g. Apple Vision Pro), flagship high-core CPU processors, or curved high-res displays.
- For 'coldStart', recommend direct flagship competitor phones or laptops.

Each persona must have exactly 3 recommendation items.
Each recommendation item MUST have:
1. 'id': a unique clean lowercase string with no spaces (e.g., 'rtx_4090', 'galaxy_tab_s9')
2. 'name': full product name (e.g., "Samsung Galaxy Tab S9 Ultra")
3. 'specs': pairing justification or specs (e.g., "14.6 inch OLED screen with Snapdragon processor for creative multitasking")
4. 'baseALS': a float between 0.60 and 0.98
5. 'baseTFIDF': a float between 0.30 and 0.85
6. 'icon': a standard Lucide icon name matching the item's category. Only use one of: 'laptop', 'cpu', 'tv', 'tablet', 'smartphone', 'gamepad', 'speaker', 'activity', 'eye', 'zap'
7. 'category': category name (e.g., "Display", "Laptop", "GPU", "Tablet", "Smartphone", "Console", "CPU")

You MUST return strictly a single valid JSON object. No other text, no markdown block wrappers (do not wrap in \`\`\`json ... \`\`\`), just the raw JSON:
{
  "name": "The latest model name",
  "specs": "Specs details (under 15 words)",
  "price": "Price in INR format, e.g. ₹79,900.00",
  "category": "The standard category (e.g. Smartphone, Laptop, Tablet, etc.)",
  "link": "A real official webpage or purchase link (using HTTPS)",
  "recos": {
    "gamer": [
      { "id": "simple_id", "name": "...", "specs": "...", "baseALS": 0.85, "baseTFIDF": 0.55, "icon": "cpu", "category": "..." },
      ...
    ],
    "office": [...],
    "creative": [...],
    "enthusiast": [...],
    "coldStart": [...]
  }
}`;

      const response = await ai.models.generateContent({
        model: "gemini-3.6-flash",
        contents: `Search query: "${query}"`,
        config: {
          systemInstruction,
          tools: [{ googleSearch: {} }],
          responseMimeType: "application/json",
        },
      });

      const responseText = response.text || "";
      let parsedData;
      try {
        parsedData = JSON.parse(responseText.trim());
      } catch (e) {
        // Fallback parsing or cleaning markdown
        let cleaned = responseText.replace(/```json/g, "").replace(/```/g, "").trim();
        parsedData = JSON.parse(cleaned);
      }

      // Enforce the strict no-accessories constraint by sanitizing parsed recommendations
      if (parsedData && parsedData.recos) {
        parsedData.recos = sanitizeRecommendations(parsedData.recos);
      }

      // Extract search grounding chunks for additional links if needed
      const chunks = response.candidates?.[0]?.groundingMetadata?.groundingChunks;
      const references = chunks ? chunks.map((c: any) => ({
        title: c.web?.title,
        uri: c.web?.uri
      })).filter((c: any) => c.uri) : [];

      // If the model didn't return a link or it's a dummy, use the first valid grounding link
      if ((!parsedData.link || parsedData.link.includes("example.com")) && references.length > 0) {
        parsedData.link = references[0].uri;
      }

      res.json({ success: true, product: parsedData, references });
    } catch (error: any) {
      console.log("Info: Activated local high-fidelity fallback generator due to API rate limit/status.");
      try {
        const fallbackProd = getFallbackProduct(query);
        return res.json({ success: true, product: fallbackProd, references: [], fallback: true });
      } catch (fallbackError: any) {
        res.status(500).json({ success: false, error: "Failed to search product or generate fallback." });
      }
    }
  });

  // Serve static files or Vite middleware
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*', (req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running on http://0.0.0.0:${PORT}`);
  });
}

startServer();
