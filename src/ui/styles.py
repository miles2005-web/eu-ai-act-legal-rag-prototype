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
    padding-top: 2.25rem;
    padding-bottom: 3rem;
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
    margin: 1.9rem 0 0.85rem;
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

.ui-product-mark {
    border-bottom: 1px solid var(--ui-border);
    margin-bottom: 1.1rem;
    padding-bottom: 1rem;
}

.ui-product-mark__eyebrow {
    color: var(--ui-accent);
    font-size: 0.68rem;
    font-weight: 750;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

.ui-product-mark__name {
    color: var(--ui-text);
    font-size: 1.05rem;
    font-weight: 680;
    letter-spacing: -0.02em;
    line-height: 1.25;
    margin-top: 0.32rem;
}

.ui-product-mark__meta {
    color: var(--ui-text-secondary);
    font-size: 0.76rem;
    margin-top: 0.28rem;
}

.ui-hero {
    padding: 1.2rem 0 1rem;
}

.ui-hero__eyebrow {
    color: var(--ui-accent);
    font-size: 0.75rem;
    font-weight: 750;
    letter-spacing: 0.1em;
    margin-bottom: 0.7rem;
    text-transform: uppercase;
}

.ui-hero__title {
    color: var(--ui-text);
    font-size: clamp(2.3rem, 5vw, 4.15rem);
    font-weight: 690;
    letter-spacing: -0.055em;
    line-height: 0.98;
    margin: 0;
    max-width: 950px;
}

.ui-hero__summary {
    color: #424245;
    font-size: 1.12rem;
    line-height: 1.55;
    margin: 1.15rem 0 0;
    max-width: 790px;
}

.ui-framework-strip {
    align-items: center;
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-top: 1.1rem;
}

.ui-capability-number {
    color: var(--ui-accent);
    font-size: 0.72rem;
    font-weight: 750;
    letter-spacing: 0.08em;
}

.ui-capability-title {
    color: var(--ui-text);
    font-size: 0.98rem;
    font-weight: 680;
    margin: 0.5rem 0 0.3rem;
}

.ui-capability-copy {
    color: var(--ui-text-secondary);
    font-size: 0.84rem;
    line-height: 1.45;
    margin: 0;
}

.ui-demo-label {
    color: var(--ui-accent);
    font-size: 0.7rem;
    font-weight: 750;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.ui-demo-title {
    color: var(--ui-text);
    font-size: 1.35rem;
    font-weight: 680;
    letter-spacing: -0.025em;
    margin: 0.55rem 0 0.55rem;
}

.ui-demo-copy {
    color: #515154;
    font-size: 0.91rem;
    line-height: 1.5;
    min-height: 4.1rem;
}

.ui-demo-meta {
    border-top: 1px solid var(--ui-border);
    color: var(--ui-text-secondary);
    font-size: 0.78rem;
    margin-top: 0.9rem;
    padding-top: 0.75rem;
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
