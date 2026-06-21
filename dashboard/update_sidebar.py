import sys
with open('dashboard/app.py', 'r', encoding='utf-8') as f:
    code = f.read()

# 1. Update main()
main_search = '''def main():
    st.set_page_config(
        page_title="UrbanFlow AI",
        page_icon="traffic_light",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    inject_css()'''

main_replace = '''def main():
    st.set_page_config(
        page_title="UrbanFlow AI",
        page_icon="traffic_light",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    inject_css()

    if "sidebar_collapsed" not in st.session_state:
        st.session_state.sidebar_collapsed = False

    if st.session_state.sidebar_collapsed:
        st.markdown('<style>[data-testid="stSidebar"] { display: none !important; }</style>', unsafe_allow_html=True)
        # Floating expand button at top left
        st.markdown("""
            <style>
            div[data-testid="stButton"] button:contains(">>") {
                position: fixed; top: 15px; left: 15px; z-index: 999999;
                background: linear-gradient(90deg,#3B82F6,#8B5CF6)!important;
                border: none; padding: 10px; border-radius: 8px; font-weight: bold;
            }
            .stButton.expand-wrapper {
                position: fixed; top: 15px; left: 15px; z-index: 999999;
            }
            </style>
        """, unsafe_allow_html=True)
        
        # We wrap it in a container we can absolutely position if CSS :contains isn't reliable
        st.markdown('<div class="expand-wrapper">', unsafe_allow_html=True)
        if st.button(">>"):
            st.session_state.sidebar_collapsed = False
            try: st.rerun()
            except: st.experimental_rerun()
        st.markdown('</div>', unsafe_allow_html=True)
'''

code = code.replace(main_search, main_replace)

# 2. Update render_sidebar()
sidebar_search = '''def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="padding:20px 8px 16px;border-bottom:1px solid rgba(255,255,255,0.07);margin-bottom:16px">'''

sidebar_replace = '''def render_sidebar():
    with st.sidebar:
        col1, col2 = st.columns([4,1])
        with col2:
            if st.button("<<"):
                st.session_state.sidebar_collapsed = True
                try: st.rerun()
                except: st.experimental_rerun()
        st.markdown("""
        <div style="padding:0px 8px 16px;border-bottom:1px solid rgba(255,255,255,0.07);margin-bottom:16px">'''

code = code.replace(sidebar_search, sidebar_replace)

# We should also re-hide the native controls if we use custom since the user explicitly wants custom buttons
# But we can just leave it as is. 

with open('dashboard/app.py', 'w', encoding='utf-8') as f:
    f.write(code)
print("done")
