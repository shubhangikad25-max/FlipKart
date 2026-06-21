"""Patches inject_css() in app.py with a safe version that doesn't break Plotly charts."""
import os

NEW_CSS = '''def inject_css():
    st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
[data-testid="stAppViewContainer"]{background:#0B1220!important}
[data-testid="stMain"]{background:#0B1220!important}
body{font-family:'Inter',sans-serif!important;background:#0B1220!important}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0D1933 0%,#0B1220 100%)!important;border-right:1px solid rgba(59,130,246,0.18)!important}
section[data-testid="stSidebar"] > div:first-child{padding-top:1rem!important}
#MainMenu,footer,[data-testid="stToolbar"],[data-testid="stDecoration"]{display:none!important}
[data-testid="stHeader"]{background:rgba(11,18,32,0.9)!important}
[data-testid="stSidebarCollapseButton"]{opacity:1!important;visibility:visible!important}
[data-testid="stSidebarCollapseButton"] svg{fill:#94A3B8!important}
[data-testid="stSidebarCollapseButton"]:hover svg{fill:#3B82F6!important}
[data-testid="collapsedControl"]{background:#0D1933!important;opacity:1!important;visibility:visible!important}
[data-testid="collapsedControl"] svg{fill:#94A3B8!important}
::-webkit-scrollbar{width:5px}
::-webkit-scrollbar-track{background:#0B1220}
::-webkit-scrollbar-thumb{background:#1E3A5F;border-radius:3px}
[data-testid="stTabs"] button{color:#475569!important;font-size:13px!important;font-weight:500!important}
[data-testid="stTabs"] button[aria-selected="true"]{color:#3B82F6!important;border-bottom:2px solid #3B82F6!important}
div.stButton>button{background:linear-gradient(90deg,#3B82F6,#8B5CF6)!important;color:#fff!important;font-weight:600!important;border:none!important;border-radius:8px!important;padding:10px 24px!important}
div.stButton>button:hover{box-shadow:0 8px 25px rgba(59,130,246,.4)!important}
[data-testid="stMetricValue"]{color:#F8FAFC!important;font-size:24px!important;font-weight:700!important}
[data-testid="stMetricLabel"]{color:#64748B!important;font-size:11px!important;text-transform:uppercase!important}
div[data-testid="stForm"]{background:rgba(255,255,255,0.03)!important;border:1px solid rgba(255,255,255,0.07)!important;border-radius:12px!important;padding:20px!important}
.stPlotlyChart{overflow:visible!important}
.kpi-card{background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);border-radius:14px;padding:22px 20px;transition:all .3s}
.kpi-card:hover{border-color:rgba(59,130,246,.4);transform:translateY(-2px);box-shadow:0 12px 32px rgba(59,130,246,.12)}
.kpi-val{font-size:30px;font-weight:800;letter-spacing:-1px;margin:6px 0 2px}
.kpi-lbl{font-size:11px;text-transform:uppercase;letter-spacing:.6px;color:#64748B;font-weight:500}
.kpi-delta{font-size:11px;font-weight:600;margin-top:6px}
.section-title{font-size:20px;font-weight:700;color:#60A5FA;margin-bottom:4px}
.section-sub{font-size:13px;color:#64748B;margin-bottom:16px}
.info-card{background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.07);border-radius:12px;padding:20px;margin-bottom:14px}
.tag{display:inline-block;padding:3px 10px;border-radius:20px;font-size:11px;font-weight:600;margin:2px}
.tag-blue{background:rgba(59,130,246,.15);color:#60A5FA}
.tag-purple{background:rgba(139,92,246,.15);color:#A78BFA}
.tag-green{background:rgba(16,185,129,.15);color:#34D399}
.tag-red{background:rgba(239,68,68,.15);color:#F87171}
.tag-yellow{background:rgba(245,158,11,.15);color:#FCD34D}
.hero-badge{display:inline-flex;align-items:center;gap:6px;background:rgba(16,185,129,.12);border:1px solid rgba(16,185,129,.3);color:#34D399;border-radius:20px;padding:5px 14px;font-size:12px;font-weight:600}
.pred-result{background:linear-gradient(135deg,rgba(59,130,246,.1),rgba(139,92,246,.07));border:1px solid rgba(59,130,246,.3);border-radius:16px;padding:28px;margin-top:16px}
.model-row{background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);border-radius:10px;padding:14px 18px;margin-bottom:8px;transition:all .2s}
.model-row:hover{border-color:rgba(59,130,246,.3);background:rgba(59,130,246,.04)}
.status-dot{height:8px;width:8px;border-radius:50%;display:inline-block;margin-right:6px}
</style>""", unsafe_allow_html=True)
'''

path = 'dashboard/app.py'
content = open(path, encoding='utf-8').read()
start = content.find('def inject_css():')
end   = content.find('\n@st.cache_data', start)
new_content = content[:start] + NEW_CSS + '\n' + content[end:]
open(path, 'w', encoding='utf-8').write(new_content)
print('Patched successfully. File size:', os.path.getsize(path), 'bytes')
