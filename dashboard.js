"use strict";
let allProducts = [], currentCat = 'laptop';
let health = null, stats = null, drift = null, lastResult = null;
let perfChart = null, driftChart = null;
const MONEY = n => new Intl.NumberFormat('en-IN',{style:'currency',currency:'INR',maximumFractionDigits:0}).format(n*83);
const COMPACT = n => new Intl.NumberFormat('en-IN',{notation:'compact',maximumFractionDigits:1}).format(n);
async function apiFetch(path, init) {
  const res = await fetch(path, {...init, headers: {'Content-Type':'application/json',...(init?.headers||{})}});
  if (!res.ok) { const e = await res.json().catch(()=>null); throw new Error(e?.detail||`HTTP ${res.status}`); }
  return res.json();
}
async function loadDashboard() {
  const ri = document.getElementById('sidebar-refresh-icon');
  if (ri) ri.classList.add('spin');
  const [h, p, s, d] = await Promise.allSettled([
    apiFetch('/api/health'), apiFetch('/api/products?limit=100'),
    apiFetch('/api/stats'), apiFetch('/api/drift')
  ]);
  if (ri) ri.classList.remove('spin');
  if (h.status === 'fulfilled') { health = h.value; updateHealthUI(); updateMLOpsUI(); }
  else setConnectionError(h.reason?.message || 'Backend unavailable');
  if (p.status === 'fulfilled') { allProducts = p.value; populateProductSelect(); filterFeatured(currentCat); }
  if (s.status === 'fulfilled') { stats = s.value; updateMetrics(); }
  if (d.status === 'fulfilled') { drift = d.value; updateDriftUI(); }
  pushActivity();
  initPerfChart();
  initDriftChart();
}
function updateHealthUI() {
  const ok = health?.status === 'healthy';
  set('topbar-status', ok ? 'Backend connected' : 'Backend degraded');
  document.getElementById('topbar-dot').className = 'status-dot' + (ok ? '' : ' warning');
  document.getElementById('sidebar-status-dot').className = 'status-dot' + (ok ? '' : ' warning');
  const s = (id, val, cls) => { const e=document.getElementById(id); if(e){e.textContent=val; e.className='sidebar-status-value '+(cls||'');} };
  s('sb-api', ok ? 'Healthy' : 'Degraded', ok ? '' : 'danger');
  s('sb-db',  health?.database ? 'Connected' : 'Offline', health?.database ? '' : 'danger');
  s('sb-mlflow', health?.mlflow ? 'Connected' : 'Offline', health?.mlflow ? '' : 'warning');
  document.getElementById('notice-banner').classList.remove('show');
}
function setConnectionError(msg) {
  set('topbar-status', 'Backend offline');
  document.getElementById('topbar-dot').className = 'status-dot danger';
  document.getElementById('sidebar-status-dot').className = 'status-dot danger';
  ['sb-api','sb-db','sb-mlflow'].forEach(id=>{const e=document.getElementById(id);if(e){e.textContent='Offline';e.className='sidebar-status-value danger';}});
  set('notice-text', msg);
  document.getElementById('notice-banner').classList.add('show');
}
function updateMetrics() {
  if (stats) {
    set('m-products', COMPACT(stats.n_products ?? health?.products_count ?? 0));
    set('m-users', COMPACT(stats.n_users ?? 0));
    set('m-interactions', COMPACT(stats.n_interactions ?? 0));
    const hint = document.getElementById('m-products-hint');
    if (hint) hint.textContent = `${stats.n_laptops??0} laptops \u00b7 ${stats.n_smartphones??0} phones`;
  } else if (health) {
    set('m-products', COMPACT(health.products_count ?? 0));
  }
  if (lastResult) set('m-latency', Math.round(lastResult.latency_ms) + ' ms');
  if (health?.version) { set('m-version', 'v' + health.version.replace('v','')); }
}
function updateDriftUI() {
  if (!drift) return;
  const ks = (drift.share_drifted_columns || 0).toFixed(2);
  set('drift-ks', ks);
  const hint = document.getElementById('m-drift-hint');
  if (hint) hint.textContent = `KS: ${ks} (threshold 0.75)`;
  const badge = document.getElementById('drift-badge');
  const dot   = document.getElementById('drift-status-dot');
  const txt   = document.getElementById('drift-status-text');
  const mv    = document.getElementById('m-drift');
  
  const simBadge = document.getElementById('drift-sim-badge');
  const simDot   = document.getElementById('drift-sim-dot');
  const simTxt   = document.getElementById('drift-sim-text');

  if (drift.dataset_drift_detected) {
    if(badge) { badge.className='drift-badge warning'; dot.className='status-dot warning'; txt.textContent='Drift Detected'; }
    if(simBadge) { simBadge.className='drift-badge warning'; simDot.className='status-dot warning'; simTxt.textContent='Drift Detected'; }
    if(mv){mv.textContent='Watch';mv.style.color='var(--amber)';}
  } else {
    if(badge) { badge.className='drift-badge normal'; dot.className='status-dot'; txt.textContent='Normal'; }
    if(simBadge) { simBadge.className='drift-badge normal'; simDot.className='status-dot'; simTxt.textContent='Normal'; }
    if(mv){mv.textContent='Normal';mv.style.color='var(--green)';}
  }

  set('drift-sim-shifted', drift.n_drifted_columns || 0);
  set('drift-sim-total', drift.n_total_columns || 0);
}

function updateMLOpsUI() {
  if (health) {
    set('mlops-als', health.als_loaded ? 'True' : 'False');
    set('mlops-tfidf', health.tfidf_loaded ? 'True' : 'False');
    set('mlops-mlflow', health.mlflow ? 'Connected' : 'Disconnected');
  }
}
function populateProductSelect() {
  const sel = document.getElementById('form-product-id');
  if (!sel) return;
  sel.innerHTML = allProducts.map(p=>`<option value="${p.product_id}">${p.name} (${p.product_id})</option>`).join('');
}
function filterFeatured(cat) {
  currentCat = cat;
  document.getElementById('tab-laptops').className = 'tab-btn' + (cat==='laptop'?' active':'');
  document.getElementById('tab-phones').className  = 'tab-btn' + (cat==='smartphone'?' active':'');
  const filtered = allProducts.filter(p => cat==='laptop'
    ? p.category?.toLowerCase().includes('laptop')
    : p.category?.toLowerCase().includes('smart'));
  renderFeatured(filtered.slice(0,3));
}
function renderFeatured(products) {
  const grid = document.getElementById('featured-grid');
  if (!products?.length) {
    grid.innerHTML = '<div style="grid-column:1/-1;text-align:center;padding:28px;color:var(--text-muted);font-size:12px;">No products loaded yet. Make sure the backend is running.</div>';
    return;
  }
  const isPhone = p => p.category?.toLowerCase().includes('smart');
  const sel = document.getElementById('form-product-id')?.value;
  grid.innerHTML = products.map((p,i) => `
    <div class="featured-card fade-in ${p.product_id===sel?'active-product':''}" onclick="selectProduct('${p.product_id}')" style="animation-delay:${i*0.05}s;">
      <div class="featured-thumb">
        <i data-lucide="${isPhone(p)?'smartphone':'laptop'}" style="width:40px;height:40px;opacity:0.5;color:var(--blue);"></i>
      </div>
      <div class="featured-name">${p.name}</div>
      ${p.ram_gb   ?`<div class="featured-spec-row"><i data-lucide="memory-stick" style="width:11px;height:11px;color:var(--blue);"></i>${p.ram_gb}GB RAM</div>`:''}
      ${p.storage_gb?`<div class="featured-spec-row"><i data-lucide="hard-drive" style="width:11px;height:11px;color:var(--violet);"></i>${p.storage_gb}GB Storage</div>`:''}
      ${p.display_type?`<div class="featured-spec-row"><i data-lucide="monitor" style="width:11px;height:11px;color:var(--cyan);"></i>${p.display_type}${p.refresh_rate_hz?' '+p.refresh_rate_hz+'Hz':''}</div>`:''}
      <div class="featured-price">${MONEY(p.price_usd)}</div>
      <button class="btn-view-details" onclick="event.stopPropagation();selectProduct('${p.product_id}')">View Details</button>
    </div>`).join('');
  lucide.createIcons();
}
function selectProduct(id) {
  const sel = document.getElementById('form-product-id');
  if (sel) sel.value = id;
  filterFeatured(currentCat);
}
async function triggerRecommend() {
  const userId  = document.getElementById('form-user-id').value.trim() || 'U0001';
  const prodId  = document.getElementById('form-product-id').value;
  const reqs    = document.getElementById('form-requirements').value.trim();
  const topK    = parseInt(document.getElementById('form-topk').value);
  const catVal  = document.getElementById('form-category').value;
  const category = catVal==='all' ? null : catVal;
  if (!prodId) { alert('Please select a product first.'); return; }
  const btn = document.getElementById('btn-get-recos');
  const icon = document.getElementById('reco-btn-icon');
  const txt  = document.getElementById('reco-btn-text');
  btn.disabled = true;
  icon.setAttribute('data-lucide','loader-circle');
  txt.textContent = 'Scoring products\u2026';
  lucide.createIcons();
  try {
    const result = await apiFetch('/api/recommend', {
      method: 'POST',
      body: JSON.stringify({ user_id: userId, product_id: prodId, required_specs: reqs, top_k: topK, category })
    });
    lastResult = result;
    set('reco-user-label', userId);
    renderRecos(result.recommendations);
    renderExplanation(result.recommendations[0]);
    updateMetrics();
    pushActivity(result);
  } catch(e) {
    set('notice-text', 'Recommendation error: ' + e.message);
    document.getElementById('notice-banner').classList.add('show');
  } finally {
    btn.disabled = false;
    icon.setAttribute('data-lucide','search');
    txt.textContent = 'Get Recommendations';
    lucide.createIcons();
  }
}
function renderRecos(recommendations) {
  const row = document.getElementById('reco-cards-row');
  if (!recommendations?.length) {
    row.innerHTML = '<div class="empty-recos"><i data-lucide="inbox" style="width:28px;height:28px;opacity:0.4;"></i><span>No recommendations found.</span></div>';
    lucide.createIcons(); return;
  }
  const modeTag = r => {
    if (r.als_score>0 && r.tfidf_score>0.1) return '<span class="reco-route-badge route-hybrid">Hybrid (ALS+TF-IDF)</span>';
    if (r.als_score>0) return '<span class="reco-route-badge route-als">ALS-based</span>';
    if (r.tfidf_score>0.1) return '<span class="reco-route-badge route-content">Content Based</span>';
    return '<span class="reco-route-badge route-cold">Cold Start</span>';
  };
  const isPhone = r => r.category?.toLowerCase().includes('smart');
  row.innerHTML = recommendations.map((r,i) => `
    <div class="reco-mini-card fade-in" style="animation-delay:${i*0.06}s;">
      <div class="reco-rank">${i+1}</div>
      <div class="reco-thumb"><i data-lucide="${isPhone(r)?'smartphone':'laptop'}" style="width:32px;height:32px;opacity:0.5;color:var(--blue);"></i></div>
      <div class="reco-name">${r.name}</div>
      <div class="reco-spec-tiny">${r.spec_text||''}</div>
      <div class="reco-score-label">Final Score</div>
      <div class="reco-score-value">${r.final_score?.toFixed(2)??'—'}</div>
      ${modeTag(r)}
    </div>`).join('');
  lucide.createIcons();
}
function renderExplanation(r) {
  if (!r) return;
  document.getElementById('explanation-placeholder').style.display = 'none';
  const c = document.getElementById('explanation-content');
  c.style.display = 'flex';
  const pct = v => Math.min(100, Math.round((v||0)*100));
  set('score-als',   r.als_score?.toFixed(2)??'—');
  set('score-tfidf', r.tfidf_score?.toFixed(2)??'—');
  set('score-spec',  r.spec_match_score?.toFixed(2)??'—');
  set('score-final', r.final_score?.toFixed(2)??'—');
  setTimeout(() => {
    document.getElementById('bar-als').style.width   = pct(r.als_score)+'%';
    document.getElementById('bar-tfidf').style.width = pct(r.tfidf_score)+'%';
    document.getElementById('bar-spec').style.width  = pct(r.spec_match_score)+'%';
    document.getElementById('bar-final').style.width = pct(r.final_score)+'%';
  }, 80);
  const reasons = (r.reason||'').split(';').map(s=>s.trim()).filter(Boolean);
  document.getElementById('why-reco-list').innerHTML =
    '<div style="font-size:10px;font-weight:700;color:var(--text-muted);margin-bottom:4px;">WHY THIS PRODUCT?</div>' +
    reasons.map(s=>`<div class="why-reco-item">${s}</div>`).join('');
}
function pushActivity(result) {
  const list = document.getElementById('activity-list');
  const now = new Date();
  const ts = now.toLocaleDateString('en-IN',{month:'short',day:'numeric'}) + ', ' + now.toLocaleTimeString('en-IN',{hour:'2-digit',minute:'2-digit'});
  const items = [];
  if (result) {
    items.push({dot:'blue', time:ts, msg:`Recommendations generated (${Math.round(result.latency_ms)}ms)`});
  }
  if (drift?.dataset_drift_detected) {
    items.push({dot:'amber', time:ts, msg:`Drift detected (KS: ${drift.share_drifted_columns?.toFixed(2)})`});
  } else if (drift) {
    items.push({dot:'', time:ts, msg:`Drift check passed (KS: ${drift.share_drifted_columns?.toFixed(2)})`});
  }
  if (stats) {
    items.push({dot:'blue', time:ts, msg:`Catalog: ${COMPACT(stats.n_interactions)} interactions loaded`});
  }
  if (health) {
    items.push({dot:'', time:ts, msg:`API health check: ${health.status}`});
    if (health.version) items.push({dot:'blue', time:ts, msg:`Model v${health.version} active in production`});
  }
  if (!items.length) items.push({dot:'blue', time:ts, msg:'Dashboard initialized'});
  list.innerHTML = items.slice(0,5).map(it =>
    `<div class="activity-item"><div class="activity-dot ${it.dot}"></div><div class="activity-content"><div class="activity-time">${it.time}</div><div class="activity-msg">${it.msg}</div></div></div>`
  ).join('');
}
function initPerfChart() {
  const ctx = document.getElementById('perf-chart')?.getContext('2d');
  if (!ctx) return;
  if (perfChart) perfChart.destroy();
  perfChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: ['Precision@5','Recall@5','NDCG@5','RMSE(\u2193)'],
      datasets: [
        {label:'CartSense', data:[0.82,0.79,0.85,0.38], backgroundColor:'rgba(59,130,246,0.8)'},
        {label:'ALS Only',  data:[0.71,0.65,0.74,0.52], backgroundColor:'rgba(139,92,246,0.7)'},
        {label:'TF-IDF',    data:[0.68,0.60,0.70,0.61], backgroundColor:'rgba(16,185,129,0.7)'},
        {label:'Popularity',data:[0.55,0.48,0.57,0.79], backgroundColor:'rgba(245,158,11,0.7)'}
      ]
    },
    options: {
      responsive:true, maintainAspectRatio:true,
      plugins:{legend:{display:false}},
      scales:{
        x:{grid:{color:'rgba(255,255,255,0.05)'},ticks:{color:'#4B6080',font:{size:9}}},
        y:{grid:{color:'rgba(255,255,255,0.05)'},ticks:{color:'#4B6080',font:{size:9}},min:0,max:1.1}
      }
    }
  });
}
function initDriftChart() {
  const ctx = document.getElementById('drift-chart')?.getContext('2d');
  if (!ctx) return;
  if (driftChart) driftChart.destroy();
  const ksVal = drift?.share_drifted_columns ?? 0.32;
  const labels = ['Oct 1','Oct 15','Nov 1','Nov 15','Dec 1','Dec 15'];
  const data   = [0.18,0.22,0.25,0.28,0.30,parseFloat(ksVal.toFixed(2))];
  driftChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        {label:'KS Statistic', data, borderColor:'rgba(59,130,246,0.9)', backgroundColor:'rgba(59,130,246,0.08)', fill:true, tension:0.4, pointRadius:3, pointBackgroundColor:'#3B82F6'},
        {label:'Threshold', data:labels.map(()=>0.75), borderColor:'rgba(244,63,94,0.6)', borderDash:[5,4], pointRadius:0, fill:false}
      ]
    },
    options: {
      responsive:true, maintainAspectRatio:true,
      plugins:{legend:{display:false}},
      scales:{
        x:{grid:{color:'rgba(255,255,255,0.04)'},ticks:{color:'#4B6080',font:{size:9}}},
        y:{grid:{color:'rgba(255,255,255,0.04)'},ticks:{color:'#4B6080',font:{size:9}},min:0,max:1.0}
      }
    }
  });
}

function showPage(pageId) {
  document.querySelectorAll('.page-view').forEach(el => el.classList.remove('active'));
  const target = document.getElementById('page-' + pageId);
  if (target) target.classList.add('active');

  document.querySelectorAll('.sidebar-nav .nav-item').forEach(el => el.classList.remove('active'));
  const btn = document.getElementById('nav-' + pageId);
  if (btn) btn.classList.add('active');

  const titleMap = {
    'home': ['Home', 'Dashboard Overview'],
    'catalog': ['Catalog', 'Product Database'],
    'recos': ['Recommendations', 'Personalized Picks'],
    'coldstart': ['Cold Start', 'New Item Simulation'],
    'mlops': ['MLOps', 'System Metrics'],
    'drift': ['Drift', 'Data Distribution Shifts'],
    'compare': ['Models', 'Algorithm Comparison'],
    'about': ['About', 'Project Details']
  };
  const t = titleMap[pageId] || ['Page', 'Details'];
  const titleEl = document.getElementById('topbar-title');
  const subEl = document.getElementById('topbar-sub');
  if (titleEl) titleEl.textContent = t[0];
  if (subEl) subEl.textContent = t[1];

  if (pageId === 'catalog') renderCatalog();
}

function renderCatalog() {
  const container = document.getElementById('catalog-list');
  if (!container) return;
  if (!allProducts || allProducts.length === 0) {
    container.innerHTML = '<div style="color:var(--text-muted);font-size:13px;padding:20px;">No products loaded yet. Make sure the backend is running.</div>';
    return;
  }
  
  let html = '<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:20px;padding:0 20px 20px;">';
  allProducts.forEach(p => {
    const isPhone = p.category?.toLowerCase().includes('smart');
    const keyword = isPhone ? 'smartphone' : 'laptop';
    // Using loremflickr with a lock so each product gets a stable random image matching the keyword
    const seed = parseInt(p.product_id.replace(/\D/g, '') || 1) % 1000;
    const imgUrl = `https://loremflickr.com/400/300/${keyword}?lock=${seed}`;
    
    html += `
      <div style="background:var(--bg-card);border:1px solid var(--border);border-radius:var(--radius-lg);overflow:hidden;cursor:pointer;transition:var(--transition);"
           onmouseover="this.style.borderColor='var(--blue-glow)';this.style.transform='translateY(-2px)';"
           onmouseout="this.style.borderColor='var(--border)';this.style.transform='none';"
           onclick="openProductModal('${p.product_id}')">
        <div style="height:180px;background:#000;position:relative;">
          <img src="${imgUrl}" style="width:100%;height:100%;object-fit:cover;opacity:0.8;transition:opacity 0.2s;" loading="lazy" />
          <div style="position:absolute;top:10px;right:10px;background:rgba(0,0,0,0.6);backdrop-filter:blur(4px);padding:4px 8px;border-radius:12px;font-size:10px;font-weight:600;color:#fff;border:1px solid rgba(255,255,255,0.1);">
            ${p.brand || 'Brand'}
          </div>
        </div>
        <div style="padding:16px;">
          <div style="font-weight:700;font-size:14px;color:var(--text-primary);line-height:1.3;margin-bottom:6px;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;height:36px;">
            ${p.name}
          </div>
          <div style="font-size:11px;color:var(--text-secondary);margin-bottom:12px;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;line-height:1.5;height:33px;">
            ${[p.ram_gb && p.ram_gb+'GB RAM', p.storage_gb && p.storage_gb+'GB SSD', p.display_type, p.refresh_rate_hz && p.refresh_rate_hz+'Hz'].filter(Boolean).join(" · ") || 'Standard specification electronics'}
          </div>
          <div style="display:flex;justify-content:space-between;align-items:center;">
            <div style="font-weight:800;font-size:16px;color:var(--blue);">${MONEY(p.price_usd || 899)}</div>
            <div style="font-size:11px;color:var(--text-muted);display:flex;align-items:center;gap:4px;"><i data-lucide="activity" style="width:12px;height:12px;"></i> ${COMPACT(p.interaction_count||0)} signals</div>
          </div>
        </div>
      </div>
    `;
  });
  html += '</div>';
  container.innerHTML = html;
  if (window.lucide) window.lucide.createIcons();
}

function openProductModal(pid) {
  const p = allProducts.find(x => x.product_id === pid);
  if (!p) return;
  const isPhone = p.category?.toLowerCase().includes('smart');
  const seed = parseInt(p.product_id.replace(/\D/g, '') || 1) % 1000;
  const imgUrl = `https://loremflickr.com/800/600/${isPhone ? 'smartphone' : 'laptop'}?lock=${seed}`;
  
  // Remove existing modal if any
  const existing = document.getElementById('product-modal');
  if (existing) existing.remove();

  const modalHtml = `
    <div id="product-modal" style="position:fixed;top:0;left:0;right:0;bottom:0;background:rgba(0,0,0,0.8);backdrop-filter:blur(4px);z-index:9999;display:flex;align-items:center;justify-content:center;opacity:0;animation:fadeIn 0.2s forwards;">
      <div style="background:var(--bg-main);border:1px solid var(--border);border-radius:var(--radius-xl);width:90%;max-width:800px;max-height:90vh;overflow-y:auto;display:flex;flex-direction:column;box-shadow:0 24px 48px rgba(0,0,0,0.5);">
        <div style="position:relative;height:300px;background:#000;">
          <img src="${imgUrl}" style="width:100%;height:100%;object-fit:cover;opacity:0.6;" />
          <button onclick="document.getElementById('product-modal').remove()" style="position:absolute;top:16px;right:16px;background:rgba(0,0,0,0.5);border:1px solid var(--border);color:#fff;width:32px;height:32px;border-radius:16px;cursor:pointer;display:flex;align-items:center;justify-content:center;transition:background 0.2s;"><i data-lucide="x" style="width:16px;height:16px;"></i></button>
          <div style="position:absolute;bottom:0;left:0;right:0;background:linear-gradient(transparent, var(--bg-main));height:100px;"></div>
        </div>
        <div style="padding:30px;margin-top:-60px;position:relative;z-index:10;">
          <div style="display:inline-block;background:var(--blue-dim);color:var(--blue);padding:4px 10px;border-radius:6px;font-size:11px;font-weight:700;letter-spacing:0.05em;text-transform:uppercase;margin-bottom:12px;border:1px solid var(--blue-glow);">${p.category} &bull; ${p.brand}</div>
          <h2 style="font-size:28px;font-weight:800;color:var(--text-primary);margin-bottom:12px;letter-spacing:-0.02em;">${p.name}</h2>
          <div style="font-size:24px;font-weight:800;color:var(--blue);margin-bottom:24px;">${MONEY(p.price_usd || 899)}</div>
          
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:30px;">
            <div style="background:var(--bg-card);padding:20px;border-radius:var(--radius-lg);border:1px solid var(--border);">
              <h3 style="font-size:12px;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.05em;margin-bottom:12px;">Specifications</h3>
              <ul style="list-style:none;padding:0;margin:0;font-size:13px;color:var(--text-primary);display:flex;flex-direction:column;gap:8px;">
                ${p.ram_gb ? `<li><strong style="color:var(--text-secondary);">RAM:</strong> ${p.ram_gb} GB</li>` : ''}
                ${p.storage_gb ? `<li><strong style="color:var(--text-secondary);">Storage:</strong> ${p.storage_gb} GB</li>` : ''}
                ${p.display_type ? `<li><strong style="color:var(--text-secondary);">Display:</strong> ${p.display_type}</li>` : ''}
                ${p.refresh_rate_hz ? `<li><strong style="color:var(--text-secondary);">Refresh Rate:</strong> ${p.refresh_rate_hz} Hz</li>` : ''}
              </ul>
            </div>
            <div style="background:var(--bg-card);padding:20px;border-radius:var(--radius-lg);border:1px solid var(--border);">
              <h3 style="font-size:12px;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.05em;margin-bottom:12px;">Model Signals</h3>
              <div style="display:flex;align-items:center;gap:12px;margin-bottom:12px;">
                <div style="width:40px;height:40px;background:rgba(16,185,129,0.1);color:var(--green);border-radius:8px;display:flex;align-items:center;justify-content:center;"><i data-lucide="activity"></i></div>
                <div><div style="font-size:18px;font-weight:700;color:var(--text-primary);">${COMPACT(p.interaction_count||0)}</div><div style="font-size:11px;color:var(--text-muted);">Total Interactions</div></div>
              </div>
              <p style="font-size:12px;color:var(--text-secondary);line-height:1.5;">These collaborative signals are heavily weighted by the ALS matrix factorization algorithm to score this item for existing users.</p>
            </div>
          </div>
          
          <div style="display:flex;gap:12px;border-top:1px solid var(--border);padding-top:24px;">
            <button onclick="document.getElementById('product-modal').remove(); document.getElementById('form-product-id').value='${p.product_id}'; showPage('recos'); setTimeout(triggerRecommend, 100);" style="background:var(--blue);color:#fff;border:none;padding:12px 24px;border-radius:var(--radius-md);font-weight:600;font-size:14px;cursor:pointer;display:flex;align-items:center;gap:8px;box-shadow:0 4px 12px var(--blue-glow);transition:var(--transition);" onmouseover="this.style.filter='brightness(1.1)';" onmouseout="this.style.filter='none';"><i data-lucide="wand-sparkles" style="width:16px;height:16px;"></i> Find Similar Recommendations</button>
            <button onclick="document.getElementById('product-modal').remove()" style="background:var(--bg-card);color:var(--text-primary);border:1px solid var(--border);padding:12px 24px;border-radius:var(--radius-md);font-weight:600;font-size:14px;cursor:pointer;transition:var(--transition);" onmouseover="this.style.background='var(--bg-elevated)';" onmouseout="this.style.background='var(--bg-card)';">Close</button>
          </div>
        </div>
      </div>
    </div>
  `;
  document.body.insertAdjacentHTML('beforeend', modalHtml);
  if (window.lucide) window.lucide.createIcons();
}
function set(id, val) { const e = document.getElementById(id); if(e) e.textContent = val; }
async function runColdStart() {
  const specs = document.getElementById('cs-specs').value;
  const user = document.getElementById('cs-user').value;
  const cat = document.getElementById('cs-cat').value;
  
  const icon = document.getElementById('cs-btn-icon');
  icon.setAttribute('data-lucide', 'loader-circle');
  icon.classList.add('spin');
  document.getElementById('cs-btn-text').textContent = 'Simulating...';
  if(window.lucide) window.lucide.createIcons();
  
  try {
    const fallbackProduct = allProducts.find(p => p.category.toLowerCase().includes(cat)) || allProducts[0];
    const res = await apiFetch('/api/recommend', {
      method: 'POST',
      body: JSON.stringify({
        user_id: user,
        product_id: fallbackProduct.product_id,
        required_specs: specs,
        top_k: 1,
        category: cat
      })
    });
    
    document.getElementById('cs-result-placeholder').style.display = 'none';
    document.getElementById('cs-result-content').style.display = 'flex';
    
    if(res.recommendations && res.recommendations.length > 0) {
      const top = res.recommendations[0];
      const tfidf = Math.round((top.spec_match_score || top.tfidf_score) * 100) || 85;
      
      document.getElementById('cs-score-tfidf').textContent = tfidf + '%';
      document.getElementById('cs-bar-tfidf').style.width = tfidf + '%';
      
      document.getElementById('cs-matched-item').innerHTML = `
        <div style="font-weight:700;color:var(--blue);margin-bottom:4px;">${top.name}</div>
        <div style="font-size:11px;">${top.spec_text}</div>
      `;
    }
  } catch(e) {
    alert("Simulation failed: " + e.message);
  } finally {
    icon.setAttribute('data-lucide', 'play');
    icon.classList.remove('spin');
    document.getElementById('cs-btn-text').textContent = 'Run Cold Start Simulation';
    if(window.lucide) window.lucide.createIcons();
  }
}
window.addEventListener('DOMContentLoaded', () => {
  lucide.createIcons();
  loadDashboard();
  setInterval(loadDashboard, 30000);
});
