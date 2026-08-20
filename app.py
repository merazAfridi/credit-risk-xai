# needs the joblib files from credit-risk.ipynb (section 9) before this will run

import os

import gradio as gr
import joblib

MODEL_PATH = "fair_loan_model.joblib"
COLUMNS_PATH = "model_columns.joblib"
TEMPLATE_PATH = "template_row.joblib"

for path in (MODEL_PATH, COLUMNS_PATH, TEMPLATE_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing '{path}'. Run credit-risk.ipynb end-to-end (through "
            "section 9, Export Artifacts) to generate it first."
        )

mitigator = joblib.load(MODEL_PATH)
model_columns = joblib.load(COLUMNS_PATH)
template_row = joblib.load(TEMPLATE_PATH)

# need whatever sensitive col the model was trained with, same trick as the notebook
sensitive_col = next(c for c in model_columns if "Gender" in c or "Race" in c)

GENDER_COLUMNS = {
    "Female": None,  # dropped as baseline during one-hot encoding
    "Male": "Gender_Male",
    "Non-binary": "Gender_Non-binary",
}

NUMERIC_LABELS = {
    "Age": "Age",
    "Income": "Annual Income",
    "Credit_Score": "Credit Score",
    "Loan_Amount": "Requested Loan Amount",
}

# The model is trained on 29 columns but the form only exposes 5; template_row
# supplies fixed defaults for the rest (Race, Employment_Type, Zip_Code_Group...).
# Those must not appear in "Why this decision" for two reasons:
#   1. the user never entered them, so citing them is just confusing, and
#   2. they are protected/proxy attributes - an adverse action notice that cites
#      race, gender, disability or citizenship is precisely the direct bias the
#      notebook's SHAP + Fairlearn audit exists to rule out (ECOA).
# So the explanation is limited to the legitimate credit criteria below.
EXPLAINABLE_FEATURES = ("Age", "Income", "Credit_Score", "Loan_Amount")


def humanize_feature(col, row):
    val = row[col]
    if col in ("Income", "Loan_Amount"):
        return f"{NUMERIC_LABELS[col]} (${val:,.0f})"
    return f"{NUMERIC_LABELS[col]} ({val:g})"


def _join_labels(labels):
    if len(labels) == 1:
        return labels[0]
    return ", ".join(labels[:-1]) + " and " + labels[-1]


def build_summary_sentence(final_denied, top, fairness_flipped_it):
    against = [humanize_feature(col, row) for col, contrib, row in top if contrib > 0]
    supporting = [humanize_feature(col, row) for col, contrib, row in top if contrib <= 0]

    verb = "denied" if final_denied else "approved"
    sentence = f"This application was <strong>{verb}</strong>"

    if final_denied and against:
        sentence += f" mainly because of {_join_labels(against)}"
        if supporting:
            sentence += f", which outweighed the positive effect of {_join_labels(supporting)}"
    elif not final_denied and supporting:
        sentence += f" mainly because of {_join_labels(supporting)}"
        if against:
            sentence += f", which outweighed the negative effect of {_join_labels(against)}"
    sentence += "."

    if fairness_flipped_it:
        sentence += (
            " Note: the underlying model's raw confidence actually favored the "
            f"opposite outcome — this result comes from the fairness-adjusted "
            "decision threshold applied across demographic groups, not the raw score."
        )

    return sentence


def build_reasons_html(applicant_row, contributions, final_denied, raw_denial_prob):
    applicable = [
        (col, contrib)
        for col, contrib in contributions
        if col in EXPLAINABLE_FEATURES
    ]
    applicable.sort(key=lambda pair: abs(pair[1]), reverse=True)
    # three keeps the whole app on one screen without an internal scroll
    top_cols = applicable[:3]

    if not top_cols:
        return ""

    top = [(col, contrib, applicant_row) for col, contrib in top_cols]
    fairness_flipped_it = (raw_denial_prob >= 0.5) != final_denied
    summary = build_summary_sentence(final_denied, top, fairness_flipped_it)

    max_abs = max(abs(c) for _, c in top_cols) or 1.0
    items = []
    for col, contrib in top_cols:
        label = humanize_feature(col, applicant_row)
        toward_denial = contrib > 0
        css_class = "toward-denial" if toward_denial else "toward-approval"
        icon = "▲" if toward_denial else "▼"
        note = "increases denial risk" if toward_denial else "supports approval"
        bar_pct = round(abs(contrib) / max_abs * 100, 1)
        items.append(f"""
            <li class="reason-item {css_class}">
                <span class="reason-icon">{icon}</span>
                <div class="reason-body">
                    <div class="reason-label">{label}</div>
                    <div class="reason-note">{note}</div>
                    <div class="reason-bar-track">
                        <div class="reason-bar-fill" style="width:{bar_pct}%;"></div>
                    </div>
                </div>
            </li>
        """)

    fairness_note = (
        '<div class="reasons-fairness-note">⚖️ Fairness-adjusted decision — see note above.</div>'
        if fairness_flipped_it else ""
    )

    return f"""
    <div class="reasons-wrap">
        <div class="reasons-title">Why this decision</div>
        <p class="reasons-summary">{summary}</p>
        {fairness_note}
        <ul class="reasons-list">{''.join(items)}</ul>
        <div class="reasons-note">
            From the model's own feature attributions, in order of influence.
            Protected attributes (race, gender, disability, etc.) are excluded
            from this explanation even though the fairness pass accounts for them.
        </div>
    </div>
    """


def predict_live(age, income, credit_score, loan_amount, gender):
    applicant = template_row.copy()
    applicant["Age"] = age
    applicant["Income"] = income
    applicant["Credit_Score"] = credit_score
    applicant["Loan_Amount"] = loan_amount

    for col in GENDER_COLUMNS.values():
        if col is not None:
            applicant[col] = False
    gender_col = GENDER_COLUMNS.get(gender)
    if gender_col is not None:
        applicant[gender_col] = True

    applicant = applicant[model_columns]

    # raw model confidence, before the fairness threshold gets applied
    raw_denial_prob = float(mitigator.estimator_.predict_proba(applicant)[:, 1][0])
    approve_pct = round((1 - raw_denial_prob) * 100, 1)
    deny_pct = round(raw_denial_prob * 100, 1)

    sensitive_value = applicant[sensitive_col]
    prediction = mitigator.predict(applicant, sensitive_features=sensitive_value)[0]

    # LightGBM's native pred_contrib gives per-feature SHAP-equivalent contributions
    # (in margin/log-odds space toward the "Denied" class) without needing the shap package
    contrib_row = mitigator.estimator_.predict(applicant, pred_contrib=True)[0]
    contributions = list(zip(model_columns, contrib_row[:-1]))
    reasons = build_reasons_html(
        applicant.iloc[0], contributions, final_denied=(prediction == 1), raw_denial_prob=raw_denial_prob
    )

    if prediction == 1:
        banner = f"""
        <div class="decision-banner denied">
            <span class="decision-icon">✕</span>
            <div>
                <div class="decision-title">Loan Denied</div>
                <div class="decision-sub">Applicant doesn't clear the risk bar</div>
            </div>
        </div>
        """
    else:
        banner = f"""
        <div class="decision-banner approved">
            <span class="decision-icon">✓</span>
            <div>
                <div class="decision-title">Loan Approved</div>
                <div class="decision-sub">Applicant clears the risk bar</div>
            </div>
        </div>
        """

    gauge = f"""
    <div class="gauge-wrap">
        <div class="gauge-group">
            <div class="gauge-row">
                <span>Approve</span><span class="gauge-pct">{approve_pct}%</span>
            </div>
            <div class="gauge-track">
                <div class="gauge-fill approve-fill" style="width:{approve_pct}%;"></div>
            </div>
        </div>
        <div class="gauge-group">
            <div class="gauge-row">
                <span>Deny</span><span class="gauge-pct">{deny_pct}%</span>
            </div>
            <div class="gauge-track">
                <div class="gauge-fill deny-fill" style="width:{deny_pct}%;"></div>
            </div>
        </div>
        <div class="gauge-note">Raw model confidence, before the fairness pass.</div>
    </div>
    """

    return banner, gauge, reasons

# ---------------------------------------------------------------------------
# Design system
#
# Everything below is presentation only. The palette, near-black backdrop and
# floating accent orbs are the app's existing identity - this just expresses
# them as tokens so spacing, radii, type and states stay consistent.
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.gradio-container {
    /* surfaces */
    --surface-1: rgba(255, 255, 255, 0.04);
    --surface-2: rgba(255, 255, 255, 0.06);
    --surface-3: rgba(255, 255, 255, 0.10);
    --border-1: rgba(255, 255, 255, 0.10);
    --border-2: rgba(255, 255, 255, 0.18);

    /* text */
    --text-1: #f5f5f5;
    --text-2: rgba(245, 245, 245, 0.72);
    --text-3: rgba(245, 245, 245, 0.50);

    /* accents (existing identity) */
    --accent: #ff4f04;
    --accent-2: #ffa600;
    --accent-3: #ff0099;
    --ok: #2ecc71;
    --ok-strong: #119726;

    /* radii */
    --r-sm: 10px;
    --r-md: 14px;
    --r-lg: 20px;
    --r-pill: 999px;

    /* spacing scale */
    --s-1: 4px;
    --s-2: 8px;
    --s-3: 12px;
    --s-4: 16px;
    --s-5: 24px;
    --s-6: 32px;

    --ring: 0 0 0 3px rgba(255, 79, 4, 0.28);
    --shadow-card: 0 8px 32px rgba(0, 0, 0, 0.45);
}

* { font-family: 'Inter', system-ui, -apple-system, sans-serif; }

/* ---------------------------------------------------------------- backdrop */

.gradio-container {
    position: relative;
    background: linear-gradient(160deg, #0a0a0a, #0d0d0d 45%, #1c1c1c);
    color: var(--text-1);
    /* full-bleed: the backdrop must reach the browser edges on wide screens,
       not stop at the content's reading width - otherwise everything past
       that width shows the page's plain background instead, as a hard seam */
    max-width: none !important;
    width: 100%;
    padding: var(--s-3) var(--s-4) var(--s-2) !important;
    /* the whole app is meant to sit on one screen: fill the viewport, then let
       the footer take up the slack rather than pushing the page into a scroll */
    min-height: 100dvh;
    display: flex;
    flex-direction: column;
    /* flex/grid items refuse to shrink below their content's natural
       (min-content) width by default. Two panels full of prose text can
       easily want to be wider than the 1340px cap below, and without this
       they don't wrap or shrink to fit - they silently push the whole row
       past its container's right edge instead, which is what caused the
       lopsided "empty gap on the left, content cut off on the right" bug:
       centering math used the *declared* 1340px width while the *actual*
       rendered width was wider and bled off-screen uncentered. */
    overflow-x: hidden;
}
.gradio-container,
.gradio-container * {
    min-width: 0;
}
/* Gradio's own root wrapper sits between the container and our content */
.gradio-container > .main,
.gradio-container > .main > .wrap,
.gradio-container > .main > .wrap > .contain {
    display: flex;
    flex-direction: column;
    flex: 1 1 auto;
    gap: var(--s-3);
    width: 100%;
}
/* ...and this innermost one is what actually gets capped to a comfortable
   reading width and centered, while .gradio-container (and its background)
   stays full-bleed to the browser edges.
   !important is required here: Gradio sets an inline `margin-right: 0px` on
   this exact element (its own scrollbar-gutter compensation), which silently
   cancels only the right half of a plain `margin: 0 auto` - that's what was
   pushing all the centering offset onto the left side only. */
.gradio-container > .main > .wrap > .contain {
    max-width: 1340px !important;
    margin-left: auto !important;
    margin-right: auto !important;
}

/* floating blurred orbs, tinted with the accent palette */
.gradio-container::before,
.gradio-container::after {
    content: "";
    position: fixed;
    width: clamp(260px, 34vw, 440px);
    aspect-ratio: 1;
    border-radius: 50%;
    filter: blur(110px);
    opacity: 0.26;
    z-index: 0;
    pointer-events: none;
    animation: float 16s ease-in-out infinite;
}
.gradio-container::before { top: -130px; left: -110px; background: #0056ff; }
.gradio-container::after  { bottom: -150px; right: -110px; background: var(--accent); animation-delay: -8s; }

@keyframes float {
    0%, 100% { transform: translate(0, 0) scale(1); }
    50%      { transform: translate(40px, 30px) scale(1.1); }
}

@media (prefers-reduced-motion: reduce) {
    .gradio-container::before,
    .gradio-container::after { animation: none; }
    * { transition-duration: 0.01ms !important; }
}

/* ------------------------------------------------------------------ header */

.app-header {
    position: relative;
    z-index: 1;
    text-align: center;
    margin: 0 0 var(--s-2);
}
.app-header h1 {
    font-size: clamp(1.2rem, 0.85rem + 1.5vw, 1.85rem);
    font-weight: 800;
    line-height: 1.25;
    letter-spacing: -0.02em;
    margin: 0 0 var(--s-2);
    /* background-image (longhand), not the `background` shorthand: Gradio's CSS
       scoper reserializes each rule, and pairing the shorthand with a later
       background-clip: text in the same rule comes out the other side with an
       empty gradient - this is the one combination that survives intact. */
    background-image: linear-gradient(90deg, var(--accent), var(--accent-2), var(--accent-3));
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    color: transparent;
    /* gradient text clips descenders without a little breathing room */
    padding-bottom: 2px;
    text-wrap: balance;
}
.app-header p {
    font-size: clamp(0.85rem, 0.8rem + 0.2vw, 0.95rem);
    line-height: 1.55;
    color: var(--text-2);
    max-width: 68ch;
    margin: 0 auto;
    text-wrap: balance;
}

/* ------------------------------------------------------------------- cards */

.app-row {
    width: 100%;
    flex-wrap: wrap;
}

.app-panel {
    position: relative;
    z-index: 1;
    min-width: 0;
    display: flex;
    flex-direction: column;
    gap: var(--s-2);
    border-radius: var(--r-lg) !important;
    padding: var(--s-3) var(--s-5) !important;
    background: var(--surface-1) !important;
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid var(--border-1) !important;
    box-shadow: var(--shadow-card);
}

/* Gradio nests each component in its own wrapper that carries a themed
   background. Neutralise those so the card reads as one surface. */
.app-panel .form,
.app-panel .block,
.app-panel .wrap,
.app-panel .head {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}
.app-panel .form { display: flex; flex-direction: column; gap: var(--s-2); }

/* section heading */
.panel-title h3 {
    font-size: 0.78rem !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    color: var(--text-2) !important;
    margin: 0 0 var(--s-1) !important;
}
.panel-hint p {
    font-size: 0.84rem !important;
    line-height: 1.5;
    color: var(--text-3) !important;
    margin: 0 !important;
}
.panel-head { display: flex; flex-direction: column; gap: var(--s-1); }

/* ------------------------------------------------------------------ fields */

/* Kill the themed "pill" behind every field label - it rendered as a light
   chip, which is what made the light label text unreadable. */
.app-panel label > span,
.app-panel span[data-testid="block-info"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0 0 var(--s-2) !important;
    display: block;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    line-height: 1.4;
    color: var(--text-1) !important;
    opacity: 1 !important;
}

/* text inputs -------------------------------------------------------------- */

.app-panel input[type="number"],
.app-panel input[type="text"] {
    width: 100%;
    color: var(--text-1) !important;
    background: var(--surface-2) !important;
    border: 1px solid var(--border-1) !important;
    border-radius: var(--r-sm) !important;
    font-size: 1rem !important;
    font-weight: 500;
    line-height: 1.4;
    transition: border-color 0.18s ease, box-shadow 0.18s ease, background 0.18s ease;
}
.app-panel input::placeholder { color: var(--text-3) !important; }

/* The browser's native number spin buttons render in the OS/browser's own
   light theme regardless of our dark background, and their tiny hit-area
   means a hovering cursor flickers rapidly in and out of them - that's the
   "blinking" against a dark input. Gradio already gives sliders their own
   custom +/- control, so the native ones are pure visual noise here. */
.app-panel input[type="number"]::-webkit-outer-spin-button,
.app-panel input[type="number"]::-webkit-inner-spin-button {
    -webkit-appearance: none;
    margin: 0;
}
.app-panel input[type="number"] {
    -moz-appearance: textfield;
}

.app-panel input[type="number"]:hover,
.app-panel input[type="text"]:hover { background: var(--surface-3) !important; }

.app-panel input[type="number"]:focus,
.app-panel input[type="text"]:focus,
.app-panel input[type="number"]:focus-visible {
    border-color: var(--accent) !important;
    box-shadow: var(--ring) !important;
    outline: none !important;
}

/* Standalone number fields get the comfortable target size. Scoped by
   elem_classes so the slider's small companion input keeps its own metrics
   and never clips its digits. */
.field-number input[type="number"] {
    min-height: 40px;
    padding: var(--s-2) var(--s-4) !important;
}

/* sliders ------------------------------------------------------------------ */

/* A slider's header is [label ......... (number|reset)]. Gradio already draws
   the border on .tab-like-container, so the inner input must stay borderless -
   otherwise it renders as a box inside a box. */
.field-slider .head {
    display: flex;
    align-items: center;
    gap: var(--s-3);
    margin-bottom: var(--s-2) !important;
}
.field-slider .head > label > span { margin-bottom: 0 !important; }

.field-slider .tab-like-container {
    display: flex;
    align-items: stretch;
    height: 34px;
    background: var(--surface-2) !important;
    border: 1px solid var(--border-1) !important;
    border-radius: var(--r-sm) !important;
    overflow: hidden;
    margin-bottom: 0 !important;
    transition: border-color 0.18s ease, box-shadow 0.18s ease;
}
.field-slider .tab-like-container:focus-within {
    border-color: var(--accent) !important;
    box-shadow: var(--ring);
}
.field-slider .tab-like-container input[type="number"] {
    height: 100%;
    min-width: 58px;
    padding: 0 var(--s-2) !important;
    border: none !important;
    border-radius: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
    font-size: 0.88rem !important;
    font-variant-numeric: tabular-nums;
    text-align: center;
}
.field-slider .reset-button {
    border: none !important;
    border-left: 1px solid var(--border-1) !important;
    background: transparent !important;
    color: var(--text-3) !important;
    min-width: 30px;
    transition: background 0.15s ease, color 0.15s ease;
}
.field-slider .reset-button:hover:not(:disabled) {
    background: var(--surface-3) !important;
    color: var(--text-1) !important;
}

.app-panel input[type="range"] {
    width: 100%;
    accent-color: var(--accent);
    background: transparent;
    cursor: pointer;
}
.app-panel input[type="range"]::-webkit-slider-runnable-track {
    height: 6px;
    border-radius: var(--r-pill);
    background: var(--surface-3);
}
/* A transform-based hover (e.g. scale()) here looks like jitter: the thumb's
   hit-area is a few px tall, so the slightest mouse movement flips hover on
   and off, re-triggering the transition every time and reading as a visible
   "vibrate." A stable design changes only the glow, never the geometry. */
.app-panel input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none;
    appearance: none;
    width: 18px;
    height: 18px;
    margin-top: -6px;
    border-radius: 50%;
    background: linear-gradient(135deg, var(--accent), var(--accent-2));
    box-shadow: 0 0 0 4px rgba(255, 79, 4, 0.18), 0 2px 6px rgba(0, 0, 0, 0.4);
    transition: box-shadow 0.15s ease;
}
.app-panel input[type="range"]::-moz-range-track {
    height: 6px;
    border-radius: var(--r-pill);
    background: var(--surface-3);
}
.app-panel input[type="range"]::-moz-range-thumb {
    width: 18px;
    height: 18px;
    border: none;
    border-radius: 50%;
    background: linear-gradient(135deg, var(--accent), var(--accent-2));
    box-shadow: 0 0 0 4px rgba(255, 79, 4, 0.18), 0 2px 6px rgba(0, 0, 0, 0.4);
    transition: box-shadow 0.15s ease;
}
.app-panel input[type="range"]:hover::-webkit-slider-thumb,
.app-panel input[type="range"]:focus-visible::-webkit-slider-thumb {
    box-shadow: 0 0 0 6px rgba(255, 79, 4, 0.28), 0 2px 8px rgba(0, 0, 0, 0.5);
}
.app-panel input[type="range"]:hover::-moz-range-thumb,
.app-panel input[type="range"]:focus-visible::-moz-range-thumb {
    box-shadow: 0 0 0 6px rgba(255, 79, 4, 0.28), 0 2px 8px rgba(0, 0, 0, 0.5);
}

/* slider min / max end labels */
.app-panel .min_value,
.app-panel .max_value {
    font-size: 0.75rem !important;
    font-weight: 500;
    color: var(--text-3) !important;
    opacity: 1 !important;
}

/* radio pills -------------------------------------------------------------- */

.field-radio .wrap {
    display: flex !important;
    flex-wrap: wrap;
    gap: var(--s-2);
}
.field-radio label {
    display: inline-flex !important;
    align-items: center;
    gap: var(--s-2);
    min-height: 42px;
    padding: var(--s-2) var(--s-4) !important;
    border-radius: var(--r-pill) !important;
    background: var(--surface-2) !important;
    border: 1px solid var(--border-1) !important;
    color: var(--text-2) !important;
    font-size: 0.92rem !important;
    font-weight: 600 !important;
    box-shadow: none !important;
    cursor: pointer;
    transition: background 0.18s ease, border-color 0.18s ease, color 0.18s ease, transform 0.15s ease;
}
.field-radio label:hover {
    background: var(--surface-3) !important;
    border-color: var(--border-2) !important;
    color: var(--text-1) !important;
}
.field-radio label.selected {
    background: linear-gradient(135deg, var(--accent), var(--accent-3)) !important;
    border-color: transparent !important;
    color: #fff !important;
    box-shadow: 0 4px 14px rgba(255, 79, 4, 0.3) !important;
}
.field-radio label:focus-within {
    box-shadow: var(--ring) !important;
}
/* the radio dot inherits themed colours that clash on the gradient pill */
.field-radio input[type="radio"] { accent-color: #fff; }

/* --------------------------------------------------------------- decision */

.decision-banner {
    display: flex;
    align-items: center;
    gap: var(--s-3);
    padding: var(--s-2) var(--s-3);
    border-radius: var(--r-md);
    transition: background 0.25s ease, border-color 0.25s ease;
}
.decision-banner.approved {
    background: rgba(17, 151, 38, 0.16);
    border: 1px solid rgba(17, 151, 38, 0.45);
}
.decision-banner.denied {
    background: rgba(255, 79, 4, 0.16);
    border: 1px solid rgba(255, 79, 4, 0.45);
}
.decision-icon {
    flex: 0 0 auto;
    width: 34px;
    height: 34px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 50%;
    font-size: 1.05rem;
    font-weight: 800;
    color: #fff;
}
.approved .decision-icon { background: var(--ok-strong); }
.denied   .decision-icon { background: var(--accent); }

.decision-title {
    font-size: 1.05rem;
    font-weight: 700;
    line-height: 1.3;
    color: var(--text-1);
}
.decision-sub {
    font-size: 0.86rem;
    line-height: 1.45;
    color: var(--text-2);
    margin-top: 1px;
}

/* confidence gauges */
.gauge-wrap { display: flex; flex-direction: column; gap: var(--s-2); }
.gauge-group { display: flex; flex-direction: column; gap: var(--s-2); }
.gauge-row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: var(--s-3);
    font-size: 0.74rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-2);
}
.gauge-row .gauge-pct {
    font-size: 0.95rem;
    font-weight: 700;
    letter-spacing: 0;
    color: var(--text-1);
    font-variant-numeric: tabular-nums;
}
.gauge-track {
    width: 100%;
    height: 8px;
    border-radius: var(--r-pill);
    background: var(--surface-3);
    overflow: hidden;
}
.gauge-fill {
    height: 100%;
    border-radius: var(--r-pill);
    transition: width 0.45s cubic-bezier(0.4, 0, 0.2, 1);
}
.approve-fill { background: linear-gradient(90deg, var(--ok-strong), var(--ok)); }
.deny-fill    { background: linear-gradient(90deg, var(--accent), var(--accent-2)); }

.gauge-note,
.reasons-note {
    font-size: 0.78rem;
    line-height: 1.45;
    color: var(--text-3);
    margin-top: var(--s-2);
}

/* ------------------------------------------------------------- reasons list */

.reasons-wrap {
    padding-top: var(--s-2);
    border-top: 1px solid var(--border-1);
}
.reasons-title {
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    color: var(--text-2);
    margin-bottom: var(--s-2);
}
.reasons-summary {
    font-size: 0.88rem;
    line-height: 1.5;
    color: var(--text-1);
    margin: 0 0 var(--s-2);
}
.reasons-summary strong { color: #fff; }
.reasons-fairness-note {
    font-size: 0.76rem;
    line-height: 1.4;
    color: var(--accent-2);
    background: rgba(255, 166, 0, 0.1);
    border: 1px solid rgba(255, 166, 0, 0.25);
    border-radius: var(--r-sm);
    padding: var(--s-1) var(--s-2);
    margin-bottom: var(--s-2);
}
.reasons-list {
    list-style: none;
    margin: 0;
    padding: 0;
    display: flex;
    flex-direction: column;
    gap: var(--s-1);
}
.reason-item {
    display: flex;
    align-items: flex-start;
    gap: var(--s-2);
    padding: var(--s-1) var(--s-3);
    border-radius: var(--r-md);
    background: var(--surface-1);
    border: 1px solid var(--border-1);
}
.reason-icon {
    flex: 0 0 auto;
    width: 24px;
    height: 24px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 50%;
    font-size: 0.68rem;
    font-weight: 800;
    margin-top: 1px;
}
.toward-denial   .reason-icon { background: rgba(255, 79, 4, 0.18); color: #ff8a4c; }
.toward-approval .reason-icon { background: rgba(17, 151, 38, 0.18); color: #4ade80; }

.reason-body { flex: 1 1 auto; min-width: 0; }
.reason-label {
    font-size: 0.92rem;
    font-weight: 600;
    line-height: 1.4;
    color: var(--text-1);
    /* long humanised feature names must wrap, never clip */
    overflow-wrap: anywhere;
}
.reason-note {
    font-size: 0.78rem;
    line-height: 1.4;
    color: var(--text-3);
    margin-top: 1px;
}
.reason-bar-track {
    width: 100%;
    height: 4px;
    border-radius: var(--r-pill);
    background: var(--surface-3);
    overflow: hidden;
    margin-top: var(--s-2);
}
.reason-bar-fill {
    height: 100%;
    border-radius: var(--r-pill);
    transition: width 0.45s cubic-bezier(0.4, 0, 0.2, 1);
}
.toward-denial   .reason-bar-fill { background: linear-gradient(90deg, var(--accent), var(--accent-2)); }
.toward-approval .reason-bar-fill { background: linear-gradient(90deg, var(--ok-strong), var(--ok)); }

/* ------------------------------------------------------------------ footer */

.app-footer {
    position: relative;
    z-index: 1;
    text-align: center;
    /* absorbs any leftover viewport height so the page ends exactly at the fold */
    margin-top: auto;
    padding-top: var(--s-3);
    border-top: 1px solid var(--border-1);
    font-size: 0.78rem;
    line-height: 1.5;
    color: var(--text-3);
}
.app-footer-wrap { margin-top: auto !important; }
/* one row on desktop, wraps to two on narrow screens */
.app-footer {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: center;
    gap: var(--s-1) var(--s-4);
}
.app-footer .footer-name { font-weight: 700; color: var(--text-2); }
.app-footer .footer-contact {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: var(--s-1) var(--s-4);
}

/* -------------------------------------------------------------- responsive */

@media (max-width: 768px) {
    /* one screen is a desktop goal; phones scroll normally */
    .gradio-container { min-height: 0; }

    .gradio-container { padding: var(--s-4) var(--s-3) var(--s-3) !important; }
    .app-panel { padding: var(--s-4) !important; gap: var(--s-3); }
    .app-header { margin-bottom: var(--s-4); }
    .decision-icon { width: 34px; height: 34px; font-size: 1.05rem; }
    .decision-title { font-size: 1.05rem; }
    /* full-width pills are easier to hit on a phone */
    .field-radio label { flex: 1 1 auto; justify-content: center; }
}
"""

# Theme tokens. These are set for the light *and* dark variants so the surface
# colours resolve identically regardless of which one Gradio picks - the app is
# designed against the near-black backdrop either way.
_TRANSPARENT = "rgba(0,0,0,0)"
_SURFACE = "rgba(255,255,255,0.04)"
_INPUT = "rgba(255,255,255,0.06)"
_BORDER = "rgba(255,255,255,0.10)"
_TEXT = "#f5f5f5"

theme = gr.themes.Soft(
    primary_hue="orange",
    secondary_hue="blue",
    neutral_hue="gray",
    radius_size="lg",
    spacing_size="md",
    text_size="md",
).set(
    body_background_fill="#0a0a0a",
    body_background_fill_dark="#0a0a0a",
    body_text_color=_TEXT,
    body_text_color_dark=_TEXT,
    block_background_fill=_SURFACE,
    block_background_fill_dark=_SURFACE,
    block_border_width="0px",
    border_color_primary=_BORDER,
    border_color_primary_dark=_BORDER,
    panel_background_fill=_TRANSPARENT,
    panel_background_fill_dark=_TRANSPARENT,
    # the themed label chip is what made field labels unreadable - drop it
    block_label_background_fill=_TRANSPARENT,
    block_label_background_fill_dark=_TRANSPARENT,
    block_label_text_color=_TEXT,
    block_label_text_color_dark=_TEXT,
    block_label_border_width="0px",
    block_label_shadow="none",
    # inputs
    input_background_fill=_INPUT,
    input_background_fill_dark=_INPUT,
    input_border_color=_BORDER,
    input_border_color_dark=_BORDER,
    input_border_width="1px",
    input_shadow="none",
    input_shadow_focus="none",
    input_placeholder_color="rgba(245,245,245,0.50)",
    input_placeholder_color_dark="rgba(245,245,245,0.50)",
    slider_color="#ff4f04",
    slider_color_dark="#ff4f04",
)

# The palette is built for the near-black backdrop, so pin the app to dark mode.
FORCE_DARK_JS = """
function forceDark() {
    const url = new URL(window.location);
    if (url.searchParams.get('__theme') !== 'dark') {
        url.searchParams.set('__theme', 'dark');
        window.location.href = url.href;
    }
}
"""

# Toasts render outside the scoped app CSS, so they need unscoped <head> styles.
TOAST_FIX_HEAD = """
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
.toast-body {
    background: #1c1c1c !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 14px !important;
}
.toast-title, .toast-message-text, .toast-close {
    color: #f5f5f5 !important;
    opacity: 1 !important;
}
</style>
"""

with gr.Blocks(
    title="Credit Risk Analysis and Decision Making Using Explainable AI",
    # Gradio sets an inline `min-height` on .gradio-container that silently beats
    # any CSS rule without !important - fill_height is the supported way to make
    # it actually fill (and be capped to) the viewport instead of fighting that.
    fill_height=True,
    # without this, Gradio's own `.fillable:not(.fill_width)` rule imposes its
    # own competing max-width breakpoints on .main, fighting our custom one
    fill_width=True,
) as demo:
    # plain HTML rather than Markdown: Gradio's `.prose h1` rule is more specific
    # than any .app-header selector and forces its own colour, which kills the
    # gradient-clipped title.
    gr.HTML(
        """
        <header class="app-header">
            <h1>🔍 Credit Risk Analysis &amp; Decision Making Using Explainable AI</h1>
            <p>
                Enter the applicant's financial details to analyze credit risk. The model
                predicts approval or denial, and shows the factors behind it.
            </p>
        </header>
        """
    )

    with gr.Row(equal_height=False, elem_classes="app-row"):
        # min_width lets Gradio stack the two panels on narrow viewports
        with gr.Column(scale=1, min_width=340, elem_classes="app-panel"):
            with gr.Column(elem_classes="panel-head"):
                gr.Markdown("### Applicant Profile", elem_classes="panel-title")
                gr.Markdown(
                    "Type a value or drag a slider — the decision updates on release.",
                    elem_classes="panel-hint",
                )

            age = gr.Slider(
                18, 100, value=35, step=1, label="Age", elem_classes="field-slider"
            )
            income = gr.Number(
                value=55000, label="Annual Income ($)", elem_classes="field-number"
            )
            credit_score = gr.Slider(
                300, 850, value=650, step=1, label="Credit Score", elem_classes="field-slider"
            )
            loan_amount = gr.Number(
                value=20000, label="Requested Loan Amount ($)", elem_classes="field-number"
            )
            gender = gr.Radio(
                list(GENDER_COLUMNS), value="Female", label="Gender", elem_classes="field-radio"
            )

        with gr.Column(scale=1, min_width=340, elem_classes="app-panel"):
            with gr.Column(elem_classes="panel-head"):
                gr.Markdown("### Decision", elem_classes="panel-title")
            banner = gr.HTML()
            gauge = gr.HTML()
            reasons = gr.HTML()

    inputs = [age, income, credit_score, loan_amount, gender]
    outputs = [banner, gauge, reasons]

    # sliders fire on drag-release / enter / blur, not every keystroke, so typing
    # a number doesn't trigger min/max errors on incomplete values (e.g. "3" of "350")
    age.release(fn=predict_live, inputs=inputs, outputs=outputs)
    credit_score.release(fn=predict_live, inputs=inputs, outputs=outputs)
    income.submit(fn=predict_live, inputs=inputs, outputs=outputs)
    income.blur(fn=predict_live, inputs=inputs, outputs=outputs)
    loan_amount.submit(fn=predict_live, inputs=inputs, outputs=outputs)
    loan_amount.blur(fn=predict_live, inputs=inputs, outputs=outputs)
    gender.change(fn=predict_live, inputs=inputs, outputs=outputs)

    demo.load(fn=predict_live, inputs=inputs, outputs=outputs)

    gr.HTML(
        """
        <div class="app-footer">
            <span class="footer-name">Developed by Gazi Meraz Mehedi</span>
            <span class="footer-contact">
                <span>📞 +8801859426070</span>
                <span>✉️ meraz.afridi@gmail.com</span>
            </span>
        </div>
        """,
        elem_classes="app-footer-wrap",
    )

if __name__ == "__main__":
    # Gradio 6 takes theme/css/js/head on launch() rather than the Blocks constructor
    demo.launch(
        theme=theme,
        css=CUSTOM_CSS,
        js=FORCE_DARK_JS,
        head=TOAST_FIX_HEAD,
        footer_links=[],
    )
