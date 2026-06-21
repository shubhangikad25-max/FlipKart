import sys
with open('dashboard/app.py', 'r', encoding='utf-8') as f:
    code = f.read()

# 1. Update main() collapsed button logic
old_block = '''    if st.session_state.sidebar_collapsed:
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
        st.markdown('</div>', unsafe_allow_html=True)'''

new_block = '''    if st.session_state.sidebar_collapsed:
        st.markdown('<style>[data-testid="stSidebar"] { display: none !important; }</style>', unsafe_allow_html=True)
        # Use CSS to target the first button on the page (which is this one) and make it float
        st.markdown("""
            <style>
            div.stButton:first-of-type {
                position: fixed; top: 12px; left: 12px; z-index: 999999;
            }
            div.stButton:first-of-type button {
                background: linear-gradient(90deg,#3B82F6,#8B5CF6)!important;
                border: none; padding: 8px 16px!important; border-radius: 8px!important; font-weight: bold; font-size: 18px!important; box-shadow: 0 4px 12px rgba(59,130,246,0.3);
            }
            </style>
        """, unsafe_allow_html=True)
        
        if st.button(">>"):
            st.session_state.sidebar_collapsed = False
            try: st.rerun()
            except: st.experimental_rerun()'''

code = code.replace(old_block, new_block)

with open('dashboard/app.py', 'w', encoding='utf-8') as f:
    f.write(code)
print("done")
