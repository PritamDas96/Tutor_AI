"""
Shared, lightweight UI helpers for all GenAI-Tutor editions.

Keeps the look consistent and professional across the four apps without
duplicating CSS. Import and call:

    from ui import inject_css, hero, status_bar, footer
    inject_css()
    hero("Version 2 : RAG Edition")
    status_bar([("Model: <b>Llama-3.1-8B</b>", ""), ("Mode: RAG", "mode")])
"""
import streamlit as st

_CSS = """
<style>
:root { --gt-primary:#4F46E5; --gt-primary-2:#7C3AED; --gt-ink:#0F172A; }

/* layout */
.block-container { padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1160px; }

/* hero banner */
.gt-hero {
  background: linear-gradient(135deg, var(--gt-primary), var(--gt-primary-2));
  border-radius: 14px; padding: 22px 26px; margin-bottom: 14px;
  color: #ffffff; box-shadow: 0 6px 20px rgba(79,70,229,.22);
}
.gt-hero-title { font-size: 1.75rem; font-weight: 800; letter-spacing: -.5px; margin: 0; }
.gt-hero-sub { font-size: .98rem; opacity: .95; margin-top: 3px; }
.gt-hero-tag {
  display: inline-block; margin-top: 11px; background: rgba(255,255,255,.18);
  border: 1px solid rgba(255,255,255,.38); padding: 3px 13px; border-radius: 999px;
  font-size: .8rem; font-weight: 600;
}

/* status chips */
.gt-status { display: flex; flex-wrap: wrap; gap: 8px; margin: 2px 0 12px; }
.gt-chip {
  display: inline-flex; align-items: center; gap: 6px; background: #F1F5F9; color: #334155;
  border: 1px solid #E2E8F0; padding: 4px 12px; border-radius: 999px; font-size: .82rem; font-weight: 600;
}
.gt-chip b { color: #0F172A; font-weight: 700; }
.gt-chip-ok   { background: #ECFDF5; color: #047857; border-color: #A7F3D0; }
.gt-chip-off  { background: #FEF2F2; color: #B91C1C; border-color: #FECACA; }
.gt-chip-mode { background: #EEF2FF; color: #4338CA; border-color: #C7D2FE; }

/* buttons */
.stButton > button {
  border-radius: 9px; font-weight: 600; border: 1px solid #E2E8F0; transition: all .15s ease;
}
.stButton > button:hover { border-color: var(--gt-primary); color: var(--gt-primary); }
.stButton > button[kind="primary"] { background: var(--gt-primary); border-color: var(--gt-primary); color: #fff; }
.stButton > button[kind="primary"]:hover { background: #4338CA; color: #fff; }

/* expanders + chat */
[data-testid="stExpander"] { border: 1px solid #E9EDF3; border-radius: 12px; }
[data-testid="stChatMessage"] { border-radius: 12px; }

/* sidebar */
[data-testid="stSidebar"] { background: #FBFCFE; border-right: 1px solid #EEF2F7; }

/* footer */
.gt-foot { color: #94A3B8; font-size: .82rem; border-top: 1px solid #EEF2F7; padding-top: 10px; margin-top: 18px; }
</style>
"""


def inject_css():
    """Inject the shared stylesheet. Call once, right after set_page_config."""
    st.markdown(_CSS, unsafe_allow_html=True)


def hero(tag, subtitle="Intelligent Conversational Learning Assistant", title="GenAI-Tutor"):
    """Render the gradient header banner with a version tag."""
    st.markdown(
        f'<div class="gt-hero">'
        f'<div class="gt-hero-title">{title}</div>'
        f'<div class="gt-hero-sub">{subtitle}</div>'
        f'<span class="gt-hero-tag">{tag}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


def status_bar(chips):
    """chips: list of (html_label, kind) where kind is '', 'ok', 'off', or 'mode'."""
    html = '<div class="gt-status">'
    for label, kind in chips:
        cls = "gt-chip" + (f" gt-chip-{kind}" if kind else "")
        html += f'<span class="{cls}">{label}</span>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def footer(text):
    """Render a subtle footer note."""
    st.markdown(f'<div class="gt-foot">{text}</div>', unsafe_allow_html=True)
