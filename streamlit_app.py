import streamlit as st
import threading
import queue
import time
import pandas as pd
import difflib
from mirror_mind import DebateManager, MirrorMindConfig

st.set_page_config(page_title="MirrorMind: Self-Refining AI Dialogue Engine", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    /* Dark Mode Fixes and General Styling */
    .stApp { margin: 0; padding: 0; }
    
    /* Dialogue Container */
    .dialogue-container { 
        padding: 10px; 
        border-radius: 8px; 
        margin-bottom: 10px; 
        font-size: 0.95em;
    }

    /* Proposer/Reviser Bubble (Green/Success) */
    .agent-proposer { 
        background-color: #e6ffed; 
        border-left: 5px solid #00c49a;
        color: #1a1a1a; /* Ensure readable text */
    }
    /* Critic/Plan Bubble (Yellow/Warning/Neutral) */
    .agent-critic { 
        background-color: #f0f8ff; /* Light blue/grey for neutral/plan */
        border-left: 5px solid #ffc107;
        color: #1a1a1a;
    }
    /* Metacritic Plan Bubble */
    .agent-metacritic {
        background-color: #f7f7f7;
        border-left: 5px solid #2196f3;
        color: #1a1a1a;
    }

    /* Diff Viewer Styling */
    .diff-del { background-color: rgba(255, 0, 0, 0.2); text-decoration: line-through; }
    .diff-add { background-color: rgba(0, 128, 0, 0.2); font-weight: bold; }

    /* Dark Mode specific overrides */
    [data-theme="dark"] .agent-proposer { background-color: #004d40; color: #e0f2f1; border-left: 5px solid #388e3c; }
    [data-theme="dark"] .agent-critic { background-color: #263238; color: #b0bec5; border-left: 5px solid #ff9800; }
    [data-theme="dark"] .agent-metacritic { background-color: #1a237e; color: #e3f2fd; border-left: 5px solid #4fc3f7; }
</style>
""", unsafe_allow_html=True)

def show_diff(text1, text2):
    """Generates simple HTML word-level diff."""
    seqm = difflib.SequenceMatcher(None, text1, text2)
    output = []
    for opcode, a0, a1, b0, b1 in seqm.get_opcodes():
        if opcode == 'equal':
            output.append(text1[a0:a1])
        elif opcode == 'insert':
            output.append(f'<span class="diff-add">{text2[b0:b1]}</span>')
        elif opcode == 'delete':
            output.append(f'<span class="diff-del">{text1[a0:a1]}</span>')
        elif opcode == 'replace':
            output.append(f'<span class="diff-del">{text1[a0:a1]}</span><span class="diff-add">{text2[b0:b1]}</span>')
            
    return "".join(output).replace("\n\n", "<p>").replace("\n", "<br>").replace("<p>", "<br><br>")

if "queue" not in st.session_state: st.session_state.queue = queue.Queue()
if "running" not in st.session_state: st.session_state.running = False
if "history" not in st.session_state: st.session_state.history = []

def thread_worker(cfg, prompt, manual_critique, q):
    try:
        mgr = DebateManager(cfg)
        mgr.run(prompt, user_critique=manual_critique, callback=lambda d: q.put(("UPDATE", d)) or True)
        q.put(("DONE", None))
    except Exception as e:
        q.put(("ERROR", str(e)))
    
st.title("🪞 MirrorMind: Self-Refining AI Dialogue Engine")
with st.sidebar:
    st.header("⚙️ Configuration")
    model_opt = st.selectbox("Model (Proposer/Reviser)", ["google/flan-t5-large", "google/flan-t5-base"])
    candidates = st.slider("Tournament Candidates (Best-of-N)", 1, 5, 3)
    max_iters = st.slider("Max Refinements", 1, 10, 4)
    st.divider()
    st.markdown("### 🧑‍💻 Human Intervention")
    manual_input = st.text_area("Inject Critique (Optional)", placeholder="E.g., 'You forgot to mention gravity!'", help="This will be added to the critics in Round 1.")

query = st.text_input("Enter your complex query:", "Explain how a Neural Network learns via Backpropagation.")
start_btn = st.button("🧠 Start AI Dialogue", disabled=st.session_state.running, type="primary")

if start_btn:
    st.session_state.running = True
    st.session_state.history = []
    
    cfg = MirrorMindConfig(model_name=model_opt, num_candidates=candidates, max_iters=max_iters)
    
    t = threading.Thread(target=thread_worker, args=(cfg, query, manual_input if manual_input else None, st.session_state.queue), daemon=True)
    t.start()
    st.rerun()

container = st.container()
with container:
    for i, r in enumerate(st.session_state.history):
        st.markdown(f"## 🔄 Round {r['iteration']}")
        st.markdown("#### 🗣️ Proposer: Initial Draft")
        st.markdown(f'<div class="dialogue-container agent-proposer">{r["draft"]}</div>', unsafe_allow_html=True)
        
        col_metrics, col_critiques = st.columns([1, 2])
        
        with col_metrics:
            st.markdown("#### 📊 Metrics")
            st.metric("Avg Quality Score", f"{r['avg_score']:.1f}/10")
            st.metric("Novelty (Change)", f"{r['novelty']:.1%}")

            scores = {c['Persona']: c['Score'] for c in r['critiques']}
            st.bar_chart(pd.Series(scores))

        with col_critiques:
            st.markdown("#### 🕵️ Critic Ensemble: Feedback")
            for c in r['critiques']:
                icon = "👤" if c['Persona'] == "Human User" else "🤖"
                st.markdown(
                    f'<div class="dialogue-container agent-critic" style="margin-bottom: 5px;">'
                    f'**{icon} {c["Persona"]}** (Score: {c["Score"]}/10): {c["Critique"]}'
                    f'</div>', 
                    unsafe_allow_html=True
                )
            
            st.markdown("#### 💡 Metacritic: Strategy")
            st.markdown(
                f'<div class="dialogue-container agent-metacritic">'
                f'{r["plan"]}'
                f'</div>', 
                unsafe_allow_html=True
            )

        st.markdown("#### ✨ Reviser: Final Result of Round")
        st.markdown(
            f'<div class="dialogue-container agent-proposer">'
            f'{r["revised"]}'
            f'</div>', 
            unsafe_allow_html=True
        )

        with st.expander("👀 View Changes (Diff)", expanded=False):
            st.markdown("##### Differences (Draft → Revised)")
            diff_html = show_diff(r['draft'], r['revised'])
            st.markdown(f'<div style="background-color: var(--background-color, #f9f9f9); padding:10px; border-radius:5px; font-family:monospace; white-space: pre-wrap;">{diff_html}</div>', unsafe_allow_html=True)

if st.session_state.running:
    try:
        while True:
            msg, data = st.session_state.queue.get_nowait()
            if msg == "UPDATE":
                st.session_state.history.append(data)
                st.rerun()
            elif msg == "DONE":
                st.session_state.running = False
                st.balloons()
                st.rerun()
            elif msg == "ERROR":
                st.error(data)
                st.session_state.running = False
                st.rerun()
    except queue.Empty:
        time.sleep(0.5)
        st.rerun()
if not st.session_state.running and st.session_state.history:
    st.markdown("---")
    st.markdown("## ✅ Final answer")
    final_answer = st.session_state.history[-1]['revised']
    st.success(final_answer)
