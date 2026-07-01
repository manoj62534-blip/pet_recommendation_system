"""
app.py  ─  PET Recommendation System
──────────────────────────────────────────────────────────────────────────────
Full-stack single file: Flask backend + embedded HTML/CSS/JS frontend.

Run:
    pip install flask scikit-learn pandas
    python train_model.py        # first time only — generates .pkl files
    python app.py                # starts server at http://127.0.0.1:5000

Routes:
    GET  /          → serves the full frontend UI
    POST /api/nlp   → NLP free-text recommendation  { "text": "..." }
    POST /api/form  → Form-based ML recommendation  { "Age": 25, ... }
"""

from flask import Flask, request, jsonify
import pandas as pd
import pickle

# Import NLP logic from nlp_module.py
from nlp_module import extract_personality

app = Flask(__name__)

# ── Load trained models (run train_model.py first) ────────────────────────────
pet_model     = pickle.load(open("pet_model.pkl",     "rb"))
breed_model   = pickle.load(open("breed_model.pkl",   "rb"))
encoders      = pickle.load(open("encoders.pkl",      "rb"))
pet_encoder   = pickle.load(open("pet_encoder.pkl",   "rb"))
breed_encoder = pickle.load(open("breed_encoder.pkl", "rb"))

CATEGORICAL_COLS = [
    "House_Type", "Budget", "Personality",
    "Activity_Level", "Noise_Tolerance", "Experience",
]


def predict_from_form(data: dict) -> dict:
    """Encode form inputs and run the Random Forest models."""
    row = {
        "Age":             int(data["Age"]),
        "House_Type":      data["House_Type"],
        "Budget":          data["Budget"],
        "Free_Time_Hours": int(data["Free_Time_Hours"]),
        "Personality":     data["Personality"],
        "Activity_Level":  data["Activity_Level"],
        "Noise_Tolerance": data["Noise_Tolerance"],
        "Experience":      data["Experience"],
    }
    df_in = pd.DataFrame([row])
    for col in CATEGORICAL_COLS:
        df_in[col] = encoders[col].transform(df_in[col])

    pet_pred   = pet_model.predict(df_in)[0]
    breed_pred = breed_model.predict(df_in)[0]

    return {
        "recommended_pet":   str(pet_encoder.inverse_transform([pet_pred])[0]),
        "recommended_breed": str(breed_encoder.inverse_transform([breed_pred])[0]),
    }


# ── Embedded Frontend (HTML + CSS + JS) ──────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>PetMatchmaker AI — Find Your Perfect Companion</title>
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet"/>
<style>
  :root{--cream:#f5f0e8;--brown:#2b1d0e;--amber:#c8873a;--rust:#a04a2a;--sand:#e8d9c0;--white:#fffdf8;}
  *,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}
  body{background:var(--cream);color:var(--brown);font-family:'DM Sans',sans-serif;min-height:100vh;overflow-x:hidden;}
  body::before{content:'';position:fixed;inset:0;background-image:url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.04'/%3E%3C/svg%3E");pointer-events:none;z-index:0;}

  /* Header */
  header{position:relative;z-index:10;padding:2.5rem 4rem 2rem;display:flex;justify-content:space-between;align-items:flex-end;border-bottom:1px solid var(--sand);}
  .logo{font-family:'Playfair Display',serif;font-size:2.4rem;font-weight:700;letter-spacing:-0.02em;line-height:1;}
  .logo span{color:var(--amber);font-style:italic;}
  .tagline{font-size:0.78rem;font-weight:300;letter-spacing:0.18em;text-transform:uppercase;color:var(--rust);opacity:0.7;}

  /* Hero */
  .hero{position:relative;z-index:5;text-align:center;padding:4rem 2rem 3rem;}
  .hero h1{font-family:'Playfair Display',serif;font-size:clamp(2.4rem,5vw,4.2rem);line-height:1.15;font-weight:400;max-width:700px;margin:0 auto 1.2rem;}
  .hero h1 em{color:var(--amber);font-style:italic;}
  .hero p{font-size:1rem;font-weight:300;max-width:480px;margin:0 auto;line-height:1.7;opacity:0.65;}

  /* Tabs */
  .tabs{position:relative;z-index:5;display:flex;justify-content:center;margin:2.5rem auto 0;max-width:420px;background:var(--sand);border-radius:999px;padding:5px;}
  .tab-btn{flex:1;padding:0.65rem 1.6rem;border:none;background:transparent;cursor:pointer;font-family:'DM Sans',sans-serif;font-size:0.85rem;font-weight:500;color:var(--brown);opacity:0.5;border-radius:999px;transition:all .3s ease;letter-spacing:0.04em;}
  .tab-btn.active{background:var(--brown);color:var(--cream);opacity:1;}

  /* Card */
  .main-card{position:relative;z-index:5;max-width:680px;margin:2.5rem auto 4rem;background:var(--white);border-radius:24px;box-shadow:0 8px 48px rgba(43,29,14,.10),0 2px 8px rgba(43,29,14,.06);overflow:hidden;animation:cardIn .6s ease both;}
  @keyframes cardIn{from{opacity:0;transform:translateY(28px)}to{opacity:1;transform:translateY(0)}}

  /* Panels */
  .panel{display:none;padding:2.8rem 3rem;}
  .panel.active{display:block;}
  .panel-label{font-size:0.72rem;letter-spacing:0.2em;text-transform:uppercase;color:var(--amber);font-weight:500;margin-bottom:1.8rem;}

  /* NLP */
  .chat-area{background:var(--cream);border-radius:14px;padding:1.4rem 1.6rem;min-height:90px;max-height:200px;overflow-y:auto;margin-bottom:1rem;font-size:0.88rem;line-height:1.75;opacity:0.6;font-style:italic;}
  .chat-row{display:flex;gap:10px;align-items:flex-end;}
  textarea{flex:1;resize:none;border:2px solid var(--sand);border-radius:14px;padding:.9rem 1.2rem;font-family:'DM Sans',sans-serif;font-size:.92rem;background:var(--white);color:var(--brown);transition:border-color .2s;outline:none;line-height:1.6;min-height:56px;}
  textarea:focus{border-color:var(--amber);}
  textarea::placeholder{color:var(--brown);opacity:0.3;}

  /* Form */
  .form-grid{display:grid;grid-template-columns:1fr 1fr;gap:1.2rem 1.6rem;}
  .field{display:flex;flex-direction:column;gap:.45rem;}
  label{font-size:.75rem;letter-spacing:.1em;text-transform:uppercase;font-weight:500;opacity:0.5;}
  select,input[type="number"]{border:2px solid var(--sand);border-radius:10px;padding:.65rem 1rem;font-family:'DM Sans',sans-serif;font-size:.9rem;background:var(--white);color:var(--brown);outline:none;transition:border-color .2s;-webkit-appearance:none;}
  select:focus,input[type="number"]:focus{border-color:var(--amber);}

  /* Buttons */
  .btn-primary{display:flex;align-items:center;justify-content:center;gap:8px;width:100%;margin-top:1.6rem;padding:1rem 2rem;border:none;border-radius:14px;background:var(--brown);color:var(--cream);font-family:'DM Sans',sans-serif;font-size:.95rem;font-weight:500;letter-spacing:.04em;cursor:pointer;transition:background .2s,transform .15s;}
  .btn-primary:hover{background:var(--rust);}
  .btn-primary:active{transform:scale(.98);}
  .btn-primary.loading{opacity:.6;pointer-events:none;}
  .send-btn{width:48px;height:48px;border-radius:12px;border:none;background:var(--amber);color:var(--white);cursor:pointer;display:flex;align-items:center;justify-content:center;font-size:1.2rem;flex-shrink:0;transition:background .2s,transform .15s;}
  .send-btn:hover{background:var(--rust);}
  .send-btn:active{transform:scale(.95);}

  /* Result */
  .result-wrap{max-width:680px;margin:0 auto 4rem;position:relative;z-index:5;animation:cardIn .5s ease both;display:none;}
  .result-wrap.show{display:block;}
  .result-card{background:var(--brown);color:var(--cream);border-radius:24px;padding:2.8rem 3rem;position:relative;overflow:hidden;}
  .result-card::before{content:'';position:absolute;top:-60px;right:-60px;width:240px;height:240px;border-radius:50%;background:var(--amber);opacity:0.12;}
  .result-title{font-size:.72rem;letter-spacing:.2em;text-transform:uppercase;color:var(--amber);margin-bottom:1.4rem;font-weight:500;}
  .result-pet{font-family:'Playfair Display',serif;font-size:2.4rem;font-weight:700;line-height:1.15;margin-bottom:.3rem;}
  .result-breed{font-size:1.05rem;opacity:.55;margin-bottom:2rem;font-style:italic;font-family:'Playfair Display',serif;}
  .result-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:.9rem;}
  .result-stat{background:rgba(255,255,255,.06);border-radius:12px;padding:.9rem 1rem;}
  .stat-label{font-size:.65rem;text-transform:uppercase;letter-spacing:.15em;opacity:.45;margin-bottom:.35rem;}
  .stat-value{font-size:.95rem;font-weight:500;}
  .stat-bar{margin-top:.4rem;height:3px;background:rgba(255,255,255,.12);border-radius:2px;overflow:hidden;}
  .stat-bar-fill{height:100%;background:var(--amber);border-radius:2px;transition:width 1s ease;}

  /* Error */
  .error-msg{color:var(--rust);font-size:.85rem;margin-top:.8rem;display:none;background:rgba(160,74,42,.08);border-radius:8px;padding:.6rem .9rem;}
  .error-msg.show{display:block;}

  footer{position:relative;z-index:5;text-align:center;padding:2rem;font-size:.75rem;opacity:.35;letter-spacing:.08em;}

  @media(max-width:600px){
    header{padding:1.5rem 1.5rem 1.2rem;}
    .panel{padding:2rem 1.6rem;}
    .result-card{padding:2rem 1.6rem;}
    .form-grid{grid-template-columns:1fr;}
    .result-grid{grid-template-columns:1fr 1fr;}
  }
</style>
</head>
<body>

<header>
  <div class="logo">Pet<span>Maker</span>AI</div>
  <div class="tagline">Personality · Environment · Type</div>
</header>

<section class="hero">
  <h1>Discover your <em>perfect</em><br>animal companion</h1>
  <p>Tell us about yourself — we'll match you with a pet that fits your personality, lifestyle &amp; home.</p>
</section>

<div class="tabs">
  <button class="tab-btn active" data-tab="nlp">✦ Tell Your Story</button>
  <button class="tab-btn"        data-tab="form">◈ Quick Form</button>
</div>

<div class="main-card">

  <!-- ── NLP Panel ── -->
  <div class="panel active" id="panel-nlp">
    <div class="panel-label">✦ Describe yourself in your own words</div>
    <div class="chat-area" id="chat-log">
      e.g. "I'm an introvert who loves quiet evenings at home. I need a calm, gentle companion…"
    </div>
    <div class="chat-row">
      <textarea id="nlp-input" rows="2"
        placeholder="Write freely about your personality, daily life, or what you're looking for in a companion…"></textarea>
      <button class="send-btn" id="nlp-submit" title="Find my match">➤</button>
    </div>
    <div class="error-msg" id="nlp-error"></div>
  </div>

  <!-- ── Form Panel ── -->
  <!-- Fields match CSV exactly: Age, House_Type, Budget, Free_Time_Hours,
       Personality, Activity_Level, Noise_Tolerance, Experience -->
  <div class="panel" id="panel-form">
    <div class="panel-label">◈ Fill in your lifestyle details</div>
    <div class="form-grid">

      <div class="field">
        <label>Age</label>
        <input type="number" id="f-age" min="1" max="100" placeholder="e.g. 25"/>
      </div>

      <div class="field">
        <label>House Type</label>
        <select id="f-house">
          <option value="">Select…</option>
          <option>Apartment</option>
          <option>Farmhouse</option>
          <option>House</option>
        </select>
      </div>

      <div class="field">
        <label>Budget</label>
        <select id="f-budget">
          <option value="">Select…</option>
          <option>Low</option>
          <option>Medium</option>
          <option>High</option>
        </select>
      </div>

      <div class="field">
        <label>Free Time Hours / Day</label>
        <input type="number" id="f-freetime" min="0" max="24" placeholder="e.g. 4"/>
      </div>

      <div class="field">
        <label>Personality</label>
        <select id="f-personality">
          <option value="">Select…</option>
          <option>Introvert</option>
          <option>Extrovert</option>
          <option>Ambivert</option>
        </select>
      </div>

      <div class="field">
        <label>Activity Level</label>
        <select id="f-activity">
          <option value="">Select…</option>
          <option>Low</option>
          <option>Medium</option>
          <option>High</option>
        </select>
      </div>

      <div class="field">
        <label>Noise Tolerance</label>
        <select id="f-noise">
          <option value="">Select…</option>
          <option>Low</option>
          <option>Medium</option>
          <option>High</option>
        </select>
      </div>

      <div class="field">
        <label>Experience with Pets</label>
        <select id="f-exp">
          <option value="">Select…</option>
          <option>Beginner</option>
          <option>Intermediate</option>
          <option>Expert</option>
        </select>
      </div>

    </div>
    <button class="btn-primary" id="form-submit"><span>Find My Match</span></button>
    <div class="error-msg" id="form-error"></div>
  </div>

</div>

<!-- ── Result Card ── -->
<div class="result-wrap" id="result-wrap">
  <div class="result-card">
    <div class="result-title">✦ Your Perfect Companion</div>
    <div class="result-pet"  id="res-pet">—</div>
    <div class="result-breed" id="res-breed">—</div>
    <div class="result-grid" id="res-stats"></div>
  </div>
</div>

<footer>PET Recommendation System &nbsp;·&nbsp; Flask + scikit-learn</footer>

<script>
  // ── Tab switcher ──────────────────────────────────────────────────────────
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      document.getElementById('panel-' + btn.dataset.tab).classList.add('active');
      document.getElementById('result-wrap').classList.remove('show');
    });
  });

  // ── Render result card ────────────────────────────────────────────────────
  function showResult(data) {
    document.getElementById('res-pet').textContent   = data.recommended_pet   || '—';
    document.getElementById('res-breed').textContent = data.recommended_breed
      ? '· ' + data.recommended_breed : '';

    const statsEl = document.getElementById('res-stats');
    statsEl.innerHTML = '';

    const numStats = [
      { label:'Introversion Score', value:data.introversion_score, max:10 },
      { label:'Emotional Need',     value:data.emotional_need,     max:10 },
      { label:'Overthinking',       value:data.overthinking,       max:10 },
      { label:'Activity Level',     value:data.activity_level,     max:10 },
      { label:'Social Need',        value:data.social_need,        max:10 },
      { label:'Companionship Need', value:data.companionship_need, max:10 },
    ].filter(s => s.value !== undefined && s.value !== null);

    const textStats = [
      { label:'Personality', value:data.personality },
    ].filter(s => s.value && s.value !== 'Unknown');

    [...textStats, ...numStats].forEach(s => {
      const pct = s.max !== undefined
        ? Math.round((s.value / s.max) * 100) : null;
      const bar = pct !== null
        ? `<div class="stat-bar"><div class="stat-bar-fill" style="width:0%" data-pct="${pct}%"></div></div>` : '';
      statsEl.innerHTML +=
        `<div class="result-stat">
           <div class="stat-label">${s.label}</div>
           <div class="stat-value">${s.value}</div>${bar}
         </div>`;
    });

    const wrap = document.getElementById('result-wrap');
    wrap.classList.add('show');
    wrap.scrollIntoView({ behavior:'smooth', block:'nearest' });
    requestAnimationFrame(() => {
      document.querySelectorAll('.stat-bar-fill')
        .forEach(el => { el.style.width = el.dataset.pct; });
    });
  }

  function showError(id, msg) {
    const el = document.getElementById(id);
    el.textContent = '⚠ ' + msg;
    el.classList.add('show');
  }
  function clearError(id) {
    document.getElementById(id).classList.remove('show');
  }

  // ── NLP submit ────────────────────────────────────────────────────────────
  document.getElementById('nlp-submit').addEventListener('click', async () => {
    clearError('nlp-error');
    const text = document.getElementById('nlp-input').value.trim();
    if (!text) { showError('nlp-error', 'Please describe yourself first.'); return; }

    const btn = document.getElementById('nlp-submit');
    btn.textContent = '…'; btn.disabled = true;

    const log = document.getElementById('chat-log');
    log.textContent = text;
    log.style.cssText = 'font-style:normal;opacity:0.9;';

    try {
      const res  = await fetch('/api/nlp', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text })
      });
      const data = await res.json();
      if (!res.ok || data.error) throw new Error(data.error || 'Server error');
      showResult(data);
    } catch(e) {
      showError('nlp-error', e.message);
    } finally {
      btn.textContent = '➤'; btn.disabled = false;
    }
  });

  // Shift+Enter = newline, Enter alone = submit
  document.getElementById('nlp-input').addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      document.getElementById('nlp-submit').click();
    }
  });

  // ── Form submit ───────────────────────────────────────────────────────────
  document.getElementById('form-submit').addEventListener('click', async () => {
    clearError('form-error');

    const age      = document.getElementById('f-age').value.trim();
    const house    = document.getElementById('f-house').value;
    const budget   = document.getElementById('f-budget').value;
    const freetime = document.getElementById('f-freetime').value.trim();
    const pers     = document.getElementById('f-personality').value;
    const activity = document.getElementById('f-activity').value;
    const noise    = document.getElementById('f-noise').value;
    const exp      = document.getElementById('f-exp').value;

    if (!age || !house || !budget || !freetime || !pers || !activity || !noise || !exp) {
      showError('form-error', 'Please fill in all fields.'); return;
    }

    const btn = document.getElementById('form-submit');
    btn.classList.add('loading');
    btn.querySelector('span').textContent = 'Analysing…';

    try {
      const res  = await fetch('/api/form', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          Age:             parseInt(age),
          House_Type:      house,
          Budget:          budget,
          Free_Time_Hours: parseInt(freetime),
          Personality:     pers,
          Activity_Level:  activity,
          Noise_Tolerance: noise,
          Experience:      exp
        })
      });
      const data = await res.json();
      if (!res.ok || data.error) throw new Error(data.error || 'Server error');
      showResult(data);
    } catch(e) {
      showError('form-error', e.message);
    } finally {
      btn.classList.remove('loading');
      btn.querySelector('span').textContent = 'Find My Match';
    }
  });
</script>
</body>
</html>"""


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    """Serve the embedded frontend."""
    return HTML, 200, {"Content-Type": "text/html; charset=utf-8"}


@app.route("/api/nlp", methods=["POST"])
def nlp_route():
    """NLP free-text → personality + pet recommendation."""
    body      = request.get_json(force=True)
    user_text = body.get("text", "").strip()
    if not user_text:
        return jsonify({"error": "No text provided"}), 400
    try:
        return jsonify(extract_personality(user_text))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/form", methods=["POST"])
def form_route():
    """Form fields → Random Forest pet + breed prediction."""
    body     = request.get_json(force=True)
    required = [
        "Age", "House_Type", "Budget", "Free_Time_Hours",
        "Personality", "Activity_Level", "Noise_Tolerance", "Experience"
    ]
    missing = [k for k in required if k not in body]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400
    try:
        return jsonify(predict_from_form(body))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
