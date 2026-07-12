"""Single-theme visual system for the Streamlit assessment interface."""

from __future__ import annotations

import streamlit as st


ENTERPRISE_STYLES = """
<style>
:root {
    --app-bg: #F5F5F7;
    --app-surface: #FFFFFF;
    --app-surface-subtle: #FAFAFC;
    --app-text-primary: #1D1D1F;
    --app-text-secondary: #6E6E73;
    --app-text-tertiary: #86868B;
    --app-border: rgba(0, 0, 0, 0.10);
    --app-border-strong: rgba(0, 0, 0, 0.16);
    --app-accent: #0071E3;
    --app-accent-hover: #0077ED;
    --app-accent-text: #FFFFFF;
    --app-warning-bg: #FFF8E6;
    --app-warning-text: #6E5200;
    --ui-radius: 18px;
}

.block-container {
    max-width: 1180px;
    padding-top: 2.7rem;
    padding-bottom: 4rem;
}

h1, h2, h3, h4 {
    color: var(--app-text-primary);
    letter-spacing: -0.035em;
}

h1 {
    font-weight: 690;
}

p, label, [data-testid="stCaptionContainer"] {
    line-height: 1.58;
}

[data-testid="stMarkdownContainer"],
[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] p,
[data-testid="stCaptionContainer"] {
    color: var(--app-text-primary);
}

[data-testid="stCaptionContainer"] {
    color: var(--app-text-secondary);
    opacity: 1;
}

[data-testid="stSidebar"] {
    border-right: 1px solid var(--app-border);
}

[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: var(--app-surface-subtle);
    border-color: var(--app-border) !important;
    border-radius: var(--ui-radius);
    box-shadow: none;
}

[data-testid="stMetric"] {
    background: transparent;
    border: 0;
    border-top: 1px solid var(--app-border-strong);
    border-radius: 0;
    padding: 0.9rem 0;
}

[data-testid="stExpander"] details {
    background: transparent;
    border-color: var(--app-border-strong) !important;
    color: var(--app-text-primary);
}

[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary p {
    color: var(--app-text-primary) !important;
}

[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea,
[data-baseweb="select"] > div {
    color: var(--app-text-primary);
}

[data-testid="stTextInput"] input::placeholder,
[data-testid="stTextArea"] textarea::placeholder {
    color: var(--app-text-tertiary);
    opacity: 1;
}

.stButton > button,
.stFormSubmitButton > button {
    border-radius: 11px;
    font-weight: 620;
    min-height: 2.65rem;
    color: var(--app-text-primary);
}

.stButton > button[kind="primary"],
.stFormSubmitButton > button[kind="primary"] {
    color: var(--app-accent-text);
}

.stButton > button[kind="primary"] [data-testid="stMarkdownContainer"],
.stButton > button[kind="primary"] [data-testid="stMarkdownContainer"] p,
.stFormSubmitButton > button[kind="primary"] [data-testid="stMarkdownContainer"],
.stFormSubmitButton > button[kind="primary"] [data-testid="stMarkdownContainer"] p {
    color: var(--app-accent-text) !important;
}

.stButton > button:disabled,
.stFormSubmitButton > button:disabled {
    color: var(--app-text-tertiary);
    opacity: 0.68;
}

.stButton > button:disabled [data-testid="stMarkdownContainer"],
.stButton > button:disabled [data-testid="stMarkdownContainer"] p,
.stFormSubmitButton > button:disabled [data-testid="stMarkdownContainer"],
.stFormSubmitButton > button:disabled [data-testid="stMarkdownContainer"] p {
    color: var(--app-text-tertiary) !important;
}

[data-testid="stSidebar"] .stButton > button:not(:disabled) {
    color: var(--app-text-primary);
}

[data-testid="stSidebar"] .stButton > button[kind="secondary"]
    [data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] .stButton > button[kind="secondary"]
    [data-testid="stMarkdownContainer"] p {
    color: var(--app-text-primary) !important;
}

.ui-section-header {
    margin: 2.8rem 0 1.05rem;
}

.ui-section-eyebrow,
.ui-hero__eyebrow,
.ui-product-mark__eyebrow,
.ui-demo-label,
.ui-finding-kicker,
.ui-capability-number {
    color: var(--app-text-secondary);
    font-size: 0.72rem;
    font-weight: 750;
    letter-spacing: 0.095em;
    text-transform: uppercase;
}

.ui-section-eyebrow {
    margin-bottom: 0.4rem;
}

.ui-section-title {
    color: var(--app-text-primary);
    font-size: 1.72rem;
    font-weight: 680;
    letter-spacing: -0.04em;
    line-height: 1.2;
    margin: 0;
}

.ui-section-description {
    color: var(--app-text-secondary);
    font-size: 0.95rem;
    margin: 0.48rem 0 0;
    max-width: 760px;
}

.ui-badge {
    align-items: center;
    background-color: var(--app-surface-subtle);
    border: 1px solid var(--app-border);
    border-radius: 999px;
    color: var(--app-text-primary);
    display: inline-flex;
    font-size: 0.75rem;
    font-weight: 650;
    line-height: 1;
    padding: 0.42rem 0.64rem;
    white-space: nowrap;
}

.ui-badge--neutral,
.ui-badge--success,
.ui-badge--danger {
    background-color: var(--app-surface-subtle);
    border-color: var(--app-border);
    color: var(--app-text-primary);
}

.ui-badge--accent {
    background-color: var(--app-accent);
    border-color: var(--app-accent);
    color: var(--app-accent-text);
}

.ui-badge--warning {
    background-color: var(--app-warning-bg);
    border-color: var(--app-border);
    color: var(--app-warning-text);
}

.ui-framework-heading {
    align-items: center;
    display: flex;
    gap: 0.65rem;
    justify-content: space-between;
    margin-bottom: 0.7rem;
}

.ui-framework-name,
.ui-product-mark__name,
.ui-capability-title,
.ui-demo-title,
.ui-finding-title,
.ui-legal-basis-citation,
.ui-evidence-summary-citation,
.ui-trace-stage-title,
.ui-trace-fact-row {
    color: var(--app-text-primary);
}

.ui-muted,
.ui-product-mark__meta,
.ui-hero__summary,
.ui-capability-copy,
.ui-demo-copy,
.ui-demo-meta,
.ui-finding-evidence-count,
.ui-finding-summary,
.ui-legal-basis-instrument,
.ui-evidence-summary-source,
.ui-evidence-summary-version,
.ui-trace-stage-description,
.ui-trace-connector,
.ui-audit-authority,
.ui-report-context,
.ui-evidence-trace-meta {
    color: var(--app-text-secondary);
}

.ui-product-mark {
    border-bottom: 1px solid var(--app-border);
    margin-bottom: 1.2rem;
    padding-bottom: 1rem;
}

.ui-product-mark__name {
    font-size: 1.05rem;
    font-weight: 680;
    letter-spacing: -0.025em;
    line-height: 1.25;
    margin-top: 0.32rem;
}

.ui-product-mark__meta {
    font-size: 0.76rem;
    margin-top: 0.28rem;
}

.ui-hero {
    padding: 1.35rem 0 1.15rem;
}

.ui-hero__eyebrow {
    margin-bottom: 0.75rem;
}

.ui-hero__title {
    color: var(--app-text-primary);
    font-size: clamp(2.4rem, 5vw, 4.3rem);
    font-weight: 700;
    letter-spacing: -0.06em;
    line-height: 0.98;
    margin: 0;
    max-width: 950px;
}

.ui-hero__summary {
    font-size: 1.12rem;
    line-height: 1.58;
    margin: 1.2rem 0 0;
    max-width: 790px;
}

.ui-framework-strip {
    align-items: center;
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-top: 1.15rem;
}

.ui-capability-block {
    border-left: 1px solid var(--app-border);
    min-height: 7.3rem;
    padding: 0.25rem 1.1rem 0.25rem 0.95rem;
}

.ui-capability-title {
    font-size: 0.98rem;
    font-weight: 680;
    margin: 0.5rem 0 0.3rem;
}

.ui-capability-copy {
    font-size: 0.84rem;
    line-height: 1.47;
    margin: 0;
}

.ui-demo-title {
    font-size: 1.35rem;
    font-weight: 690;
    letter-spacing: -0.03em;
    margin: 0.55rem 0;
}

.ui-demo-copy {
    font-size: 0.91rem;
    line-height: 1.52;
    min-height: 4.1rem;
}

.ui-demo-meta {
    border-top: 1px solid var(--app-border);
    font-size: 0.78rem;
    margin-top: 0.9rem;
    padding-top: 0.75rem;
}

.ui-finding-evidence-count {
    font-size: 0.78rem;
    font-weight: 600;
    padding-top: 0.35rem;
    text-align: right;
}

.ui-finding-kicker {
    margin-top: 1.15rem;
}

.ui-finding-title {
    font-size: 1.65rem;
    font-weight: 690;
    letter-spacing: -0.04em;
    line-height: 1.18;
    margin: 0.38rem 0 1rem;
}

.ui-finding-title--hero {
    font-size: clamp(2rem, 4vw, 3.3rem);
    font-weight: 700;
    letter-spacing: -0.055em;
    line-height: 1.03;
    max-width: 900px;
}

.ui-finding-summary {
    font-size: 1.08rem;
    line-height: 1.64;
    margin: 0 0 1.65rem;
    max-width: 850px;
}

.ui-legal-basis-row,
.ui-evidence-summary-row {
    align-items: center;
    background: transparent;
    border: 0;
    border-bottom: 1px solid var(--app-border-strong);
    border-radius: 0;
    display: flex;
    justify-content: space-between;
    margin-bottom: 0.55rem;
    padding: 0.82rem 0;
}

.ui-legal-basis-citation,
.ui-evidence-summary-citation {
    font-size: 0.9rem;
    font-weight: 680;
}

.ui-legal-basis-instrument,
.ui-evidence-summary-source,
.ui-evidence-summary-version {
    font-size: 0.76rem;
}

.ui-evidence-summary-version {
    padding-left: 1rem;
    text-align: right;
}

.ui-trace-stage-header {
    align-items: center;
    display: flex;
    gap: 0.9rem;
    margin: 0.15rem auto 0.7rem;
    max-width: 860px;
}

.ui-trace-stage-number {
    align-items: center;
    background: transparent;
    border: 1px solid var(--app-border-strong);
    border-radius: 50%;
    color: var(--app-text-primary);
    display: flex;
    flex: 0 0 2.25rem;
    font-size: 0.68rem;
    font-weight: 750;
    height: 2.25rem;
    justify-content: center;
    letter-spacing: 0.04em;
}

.ui-trace-stage-title {
    font-size: 1.35rem;
    font-weight: 690;
    letter-spacing: -0.03em;
}

.ui-trace-stage-description {
    font-size: 0.82rem;
    margin-top: 0.12rem;
}

.ui-trace-connector {
    font-size: 1.3rem;
    font-weight: 500;
    height: 2.25rem;
    line-height: 2.25rem;
    margin-left: auto;
    margin-right: auto;
    max-width: 860px;
    padding-left: 0.52rem;
    text-align: left;
}

.ui-trace-fact-row {
    align-items: center;
    display: flex;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    font-size: 0.82rem;
    gap: 0.65rem;
    margin-left: auto;
    margin-right: auto;
    max-width: 860px;
    padding: 0.42rem 0;
}

.ui-trace-fact-dot {
    background: var(--app-text-secondary);
    border-radius: 50%;
    display: inline-block;
    flex: 0 0 0.4rem;
    height: 0.4rem;
    opacity: 0.55;
}

.ui-audit-authority {
    background: var(--app-surface-subtle);
    border: 1px solid var(--app-border);
    border-radius: 999px;
    float: right;
    font-size: 0.74rem;
    font-weight: 650;
    margin-top: 0.25rem;
    padding: 0.38rem 0.62rem;
}

.ui-report-context,
.ui-evidence-trace-meta {
    align-items: center;
    border-bottom: 1px solid var(--app-border-strong);
    border-top: 1px solid var(--app-border-strong);
    display: flex;
    flex-wrap: wrap;
    font-size: 0.8rem;
    gap: 0.65rem 1.25rem;
    padding: 0.85rem 0;
}

.ui-evidence-trace-meta {
    border-top: 0;
    margin-bottom: 0.8rem;
    padding-top: 0;
}

.ui-mono {
    background: var(--app-surface-subtle);
    border-radius: 6px;
    color: var(--app-text-primary);
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    font-size: 0.76rem;
    overflow-wrap: anywhere;
    padding: 0.18rem 0.35rem;
}
</style>
"""


def apply_enterprise_styles() -> None:
    """Apply the product's single light-theme component styles."""

    st.markdown(ENTERPRISE_STYLES, unsafe_allow_html=True)
