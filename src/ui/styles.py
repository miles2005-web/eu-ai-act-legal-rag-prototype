"""Centralized visual tokens and layout styling for Streamlit."""

from __future__ import annotations

import streamlit as st


ENTERPRISE_STYLES = """
<style>
:root {
    --ui-bg: #f5f5f7;
    --ui-surface: #ffffff;
    --ui-surface-subtle: #fafafa;
    --ui-text: #1d1d1f;
    --ui-text-secondary: #6e6e73;
    --ui-border: #e5e5e7;
    --ui-accent: #1769e0;
    --ui-success: #207a4b;
    --ui-warning: #9a6700;
    --ui-danger: #b42318;
    --ui-radius: 14px;
}

.stApp {
    background: var(--ui-bg);
    color: var(--ui-text);
}

[data-testid="stHeader"] {
    background: rgba(245, 245, 247, 0.88);
    backdrop-filter: blur(18px);
}

[data-testid="stSidebar"] {
    background: var(--ui-surface-subtle);
    border-right: 1px solid var(--ui-border);
}

.block-container {
    max-width: 1180px;
    padding-top: 3.25rem;
    padding-bottom: 4rem;
}

h1, h2, h3, h4 {
    color: var(--ui-text);
    letter-spacing: -0.025em;
}

h1 {
    font-weight: 680;
}

p, label, [data-testid="stCaptionContainer"] {
    line-height: 1.55;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--ui-surface);
    border-color: var(--ui-border) !important;
    border-radius: var(--ui-radius);
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.025);
}

[data-testid="stMetric"] {
    background: var(--ui-surface);
    border: 1px solid var(--ui-border);
    border-radius: 12px;
    padding: 1rem 1.1rem;
}

.stButton > button,
.stFormSubmitButton > button {
    border-radius: 10px;
    font-weight: 600;
    min-height: 2.65rem;
}

.stButton > button[kind="primary"],
.stFormSubmitButton > button[kind="primary"] {
    background: var(--ui-accent);
    border-color: var(--ui-accent);
}

.ui-section-header {
    margin: 2.6rem 0 1.1rem;
}

.ui-section-eyebrow {
    color: var(--ui-accent);
    font-size: 0.73rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    margin-bottom: 0.35rem;
    text-transform: uppercase;
}

.ui-section-title {
    color: var(--ui-text);
    font-size: 1.42rem;
    font-weight: 650;
    letter-spacing: -0.02em;
    line-height: 1.25;
    margin: 0;
}

.ui-section-description {
    color: var(--ui-text-secondary);
    font-size: 0.94rem;
    margin: 0.45rem 0 0;
    max-width: 760px;
}

.ui-badge {
    align-items: center;
    border: 1px solid transparent;
    border-radius: 999px;
    display: inline-flex;
    font-size: 0.75rem;
    font-weight: 650;
    line-height: 1;
    padding: 0.42rem 0.64rem;
    white-space: nowrap;
}

.ui-badge--neutral {
    background: #f0f0f2;
    border-color: #e3e3e7;
    color: #4b4b50;
}

.ui-badge--accent {
    background: #edf4ff;
    border-color: #d6e6ff;
    color: #1457b8;
}

.ui-badge--success {
    background: #edf8f2;
    border-color: #d7eddf;
    color: var(--ui-success);
}

.ui-badge--warning {
    background: #fff7e6;
    border-color: #f5dfad;
    color: var(--ui-warning);
}

.ui-badge--danger {
    background: #fff0ee;
    border-color: #f5d3cf;
    color: var(--ui-danger);
}

.ui-framework-heading {
    align-items: center;
    display: flex;
    gap: 0.65rem;
    justify-content: space-between;
    margin-bottom: 0.7rem;
}

.ui-framework-name {
    color: var(--ui-text);
    font-size: 1rem;
    font-weight: 650;
}

.ui-muted {
    color: var(--ui-text-secondary);
}

.ui-mono {
    background: #f3f3f5;
    border-radius: 6px;
    color: #3a3a3c;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    font-size: 0.76rem;
    overflow-wrap: anywhere;
    padding: 0.18rem 0.35rem;
}
</style>
"""


def apply_enterprise_styles() -> None:
    """Apply presentation-only styling once per Streamlit rerun."""

    st.markdown(ENTERPRISE_STYLES, unsafe_allow_html=True)
