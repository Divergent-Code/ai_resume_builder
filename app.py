"""Résumé Tailor — a Streamlit UI over the AI-201 resume-tailoring pipeline.

Wraps the notebook pipeline (pipeline.py) in a form-driven web app: paste a job
description, upload a résumé, pick a provider, and get a grounded, source-tagged
tailored résumé plus a coverage report and fabrication check.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

API keys: set them as Streamlit secrets (.streamlit/secrets.toml) or environment
variables (ANTHROPIC_API_KEY / GOOGLE_API_KEY / DEEPSEEK_API_KEY / GROQ_API_KEY),
or paste one into the sidebar (session-only — never commit a key).
"""
from __future__ import annotations

import os
import re

import streamlit as st

import pipeline as pl

# ------------------------------------------------------------------ config ---

st.set_page_config(page_title="Résumé Tailor", page_icon="✎", layout="centered")

# Neo-Kinpaku product styling (adapted from the impeccable design system):
# one sans, hairline borders, flat surfaces, gold accent used only for the
# primary action; patina = "improved/covered", vermilion = gaps/warnings.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Albert+Sans:wght@300;400;500;600&display=swap');
:root{
  --gold:#F4B93C; --gold-pale:#F6CE7D; --patina:#43C1B8; --vermilion:#D85D47;
  --input:#22201C; --hairline:rgba(255,255,255,0.10); --champagne:#E8E8E8; --muted:#A8A8A8;
}
html, body, .stApp, [data-testid="stMarkdownContainer"]{
  font-family:'Albert Sans', system-ui, -apple-system, 'Segoe UI', sans-serif;
}
.block-container{ padding-top:2.4rem; padding-bottom:4rem; max-width:820px; }
h1,h2,h3{ font-family:'Albert Sans',system-ui,sans-serif; color:var(--champagne); letter-spacing:-0.01em; }
h1{ font-weight:600; font-size:2.05rem; margin-bottom:.15rem; }
h3{ font-weight:600; font-size:1.12rem; }
p, li{ line-height:1.7; }
hr{ border-color:var(--hairline); margin:1.2rem 0; }
.eyebrow{ font-family:ui-monospace,SFMono-Regular,'Roboto Mono',Consolas,monospace;
  font-size:.7rem; letter-spacing:.22em; text-transform:uppercase; color:var(--muted); }
/* inputs — hairline, small radius, gold focus (no glow) */
.stTextInput input, .stTextArea textarea{
  background:var(--input) !important; border:1px solid var(--hairline) !important; border-radius:4px !important; }
.stTextInput input:focus, .stTextArea textarea:focus{ border-color:var(--gold) !important; box-shadow:none !important; }
/* buttons — small radius, no bounce */
.stButton>button, .stDownloadButton>button{ border-radius:2px !important; font-weight:600 !important; }
/* bordered containers → flat hairline card, not a thick side-tab */
[data-testid="stVerticalBlockBorderWrapper"]{ border:1px solid var(--hairline) !important; border-radius:4px; }
/* source-tag + status pills */
.src{ display:inline-block; font-family:ui-monospace,monospace; font-size:.6rem; letter-spacing:.08em;
  padding:1px 6px; border-radius:3px; border:1px solid var(--hairline); margin-left:6px; vertical-align:middle; }
.src-resume{ color:var(--muted); }
.src-github{ color:var(--gold-pale); border-color:rgba(244,185,60,.35); }
.src-reframed{ color:var(--patina); border-color:rgba(67,193,184,.35); }
.pill{ display:inline-block; font-family:ui-monospace,monospace; font-size:.66rem; letter-spacing:.05em;
  padding:2px 9px; border-radius:999px; border:1px solid var(--hairline); }
.pill-cov{ color:var(--patina); border-color:rgba(67,193,184,.4); }
.pill-gap{ color:var(--vermilion); border-color:rgba(216,93,71,.45); }
.cov-track{ height:8px; background:var(--input); border:1px solid var(--hairline); border-radius:999px; overflow:hidden; margin:.5rem 0 .2rem; }
.cov-fill{ height:100%; background:var(--patina); }
.callout{ border:1px solid var(--hairline); border-radius:4px; padding:12px 14px; margin:.4rem 0; }
.callout-warn{ border-color:rgba(216,93,71,.5); }
.callout-good{ border-color:rgba(67,193,184,.45); }
.dim{ color:var(--muted); }
[data-baseweb="tab-list"]{ gap:2px; }
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------- helpers ---

def get_api_key(provider: str) -> str:
    """Resolve a key: Streamlit secrets → env var → sidebar input (session only)."""
    secret_name = pl.PROVIDER_CONFIG[provider]["secret"]
    try:
        if secret_name in st.secrets:
            return st.secrets[secret_name]
    except Exception:
        pass  # no secrets.toml present
    return os.environ.get(secret_name, "") or st.session_state.get(f"key_{provider}", "")


def resolve_resume(uploaded, pasted: str) -> str:
    """Turn an uploaded .md/.txt/.pdf file or pasted text into résumé text.

    PDFs (issue #8) are extracted with pypdf; a scanned/image-only PDF has no text
    layer, so extraction comes back near-empty — we raise so the caller can tell the
    user to paste instead of silently feeding the pipeline a blank résumé.
    """
    if uploaded is not None:
        data = uploaded.getvalue()
        if (uploaded.name or "").lower().endswith(".pdf"):
            text = pl.extract_pdf_text(data)
            if len(text.strip()) < 30:
                raise ValueError(
                    "Couldn't read text from that PDF — it may be scanned or image-only. "
                    "Export a text-based PDF, or paste your résumé below instead.")
            return text
        return data.decode("utf-8", errors="replace")
    return (pasted or "").strip()


def source_badges(md: str) -> str:
    """Replace [SOURCE: X] tags with styled inline badges."""
    def repl(m):
        kind = m.group(1).strip().upper()
        cls = {"RESUME": "src-resume", "GITHUB": "src-github", "REFRAMED": "src-reframed"}.get(kind, "src-resume")
        return f'<span class="src {cls}">{kind}</span>'
    return re.sub(r"\[SOURCE:\s*([^\]]+)\]", repl, md)


def split_output(text: str):
    """Split the model output into (tailored résumé, what-changed) sections."""
    m = re.search(r"\n[#>*\s\d.]*?what changed.*?\n", text, re.IGNORECASE)
    if m:
        return text[:m.start()].strip(), text[m.start():].strip()
    return text.strip(), ""


# ----------------------------------------------------------------- sidebar ---

with st.sidebar:
    st.markdown('<div class="eyebrow">AI 201 · Production RAG</div>', unsafe_allow_html=True)
    st.markdown("### Résumé Tailor")
    st.caption("Grounded résumé tailoring — retrieval keeps it honest, so it reframes what you have instead of inventing new experience.")
    st.divider()

    provider = st.selectbox(
        "LLM provider",
        options=list(pl.PROVIDER_CONFIG.keys()),
        format_func=lambda p: pl.PROVIDER_CONFIG[p]["label"],
    )
    default_model = pl.PROVIDER_CONFIG[provider]["model"]
    model = st.text_input("Model", value=default_model,
                          help="Defaults per provider; override if the ID has changed.")

    secret_name = pl.PROVIDER_CONFIG[provider]["secret"]
    resolved_key = get_api_key(provider)
    if resolved_key:
        st.markdown(f'<span class="dim">✓ key detected via <code>{secret_name}</code></span>', unsafe_allow_html=True)
    else:
        st.session_state[f"key_{provider}"] = st.text_input(
            f"{pl.PROVIDER_CONFIG[provider]['label']} API key", type="password",
            help="Session-only — not stored or committed. Prefer Streamlit secrets or an env var for deploys.")
        resolved_key = st.session_state.get(f"key_{provider}", "")

    st.divider()
    st.caption("Embeddings run locally (all-MiniLM-L6-v2), so retrieval is identical across providers.")


# ------------------------------------------------------------------- header --

st.markdown('<div class="eyebrow">Tailor · Ground · Verify</div>', unsafe_allow_html=True)
st.title("Tailor your résumé to a job")
st.markdown('<p class="dim">Paste a job description and upload your résumé. Retrieval matches your real '
            'experience to each requirement, so the result is grounded — with source tags, a coverage '
            'report, and an independent fabrication check.</p>', unsafe_allow_html=True)


# -------------------------------------------------------------------- inputs -

with st.form("inputs", border=False):
    jd_raw = st.text_area("Job description (paste text, or a URL)", height=160,
                          placeholder="Paste the job posting text, or a link to it…")
    col_a, col_b = st.columns([3, 2])
    with col_a:
        uploaded = st.file_uploader("Résumé (.pdf, .md, or .txt)", type=["pdf", "md", "txt"])
    with col_b:
        github_username = st.text_input("GitHub username", placeholder="optional")
    pasted_resume = st.text_area("…or paste your résumé", height=120,
                                 placeholder="Paste résumé text if you'd rather not upload a file")
    submitted = st.form_submit_button("Tailor my résumé", type="primary", use_container_width=True)

if submitted:
    st.session_state.pop("results", None)
    st.session_state.pop("override_injection", None)
    try:
        resume_text = resolve_resume(uploaded, pasted_resume)
    except ValueError as e:
        st.error(str(e))
        st.stop()
    st.session_state["pending"] = {
        "provider": provider, "model": model.strip() or default_model, "key": resolved_key,
        "jd_raw": jd_raw, "resume": resume_text,
        "github": github_username.strip(),
    }


# ------------------------------------------------------------- run pipeline --

def run_pipeline(req):
    client = pl.LLMClient(req["provider"], req["key"], model=req["model"])

    jd_text, jd_note = pl.scrape_jd(req["jd_raw"])
    if jd_note:
        st.caption(jd_note)
    if len(jd_text.strip()) < 40:
        st.error("The job description looks empty or too short. Paste the full text and try again.")
        return None
    if len(req["resume"].strip()) < 40:
        st.error("The résumé looks empty. Upload a .pdf/.md/.txt file or paste the text, then try again.")
        return None

    # Injection screen (substring + LLM classifier) before anything reaches a prompt.
    flagged, reason = pl.screen_for_injection(jd_text, client=client)
    if flagged and not st.session_state.get("override_injection"):
        st.markdown(f'<div class="callout callout-warn"><strong>Possible prompt injection in the job '
                    f'description</strong><br><span class="dim">Detected: {reason}. The JD is untrusted '
                    f'input — review it before tailoring.</span></div>', unsafe_allow_html=True)
        st.checkbox("I've reviewed the job description and want to proceed anyway", key="override_injection")
        return None

    results = {"provider": req["provider"], "model": req["model"], "jd_text": jd_text}
    with st.status("Tailoring your résumé…", expanded=True) as status:
        st.write("Fetching GitHub repositories…")
        try:
            repos = pl.fetch_github_repos(req["github"]) if req["github"] else []
        except Exception as e:
            st.write(f"GitHub fetch failed ({e}) — continuing without repo evidence.")
            repos = []
        results["repos"] = repos

        st.write("Chunking & indexing your résumé… (first run downloads the embedding model)")
        chunks, chunk_warning = pl.chunk_source_material(req["resume"], repos)
        collection = pl.index_chunks(chunks)
        results["chunk_count"] = len(chunks)
        results["chunk_warning"] = chunk_warning

        st.write("Extracting the job's requirements…")
        requirements = pl.extract_jd_requirements(client, jd_text)

        st.write("Retrieving matching evidence…")
        evidence = pl.retrieve_relevant_experience(requirements, collection)
        results["requirements"] = requirements
        results["evidence"] = evidence

        st.write("Writing the tailored résumé…")
        tailored = pl.generate_tailored_resume(client, requirements, evidence, req["resume"])
        results["tailored"] = tailored

        st.write("Fact-checking against your sources…")
        results["fabrication"] = pl.fabrication_check(client, req["resume"], repos, tailored)

        st.write("Scoring requirement coverage…")
        results["coverage"] = pl.evaluate_rag_coverage(requirements, evidence)
        results["diff"] = pl.section_diff(req["resume"], tailored)
        results["call_log"] = list(client.call_log)
        status.update(label="Done", state="complete", expanded=False)
    return results


pending = st.session_state.get("pending")
if pending and "results" not in st.session_state:
    try:
        out = run_pipeline(pending)
    except Exception as e:
        st.error(f"Something went wrong talking to {pl.PROVIDER_CONFIG[pending['provider']]['label']}: {e}")
        st.session_state.pop("pending", None)
        out = None
    if out is not None:
        st.session_state["results"] = out
        st.session_state.pop("pending", None)
        st.rerun()


# -------------------------------------------------------------- empty state --

if "results" not in st.session_state and not pending:
    st.divider()
    st.markdown('<div class="eyebrow">How it works</div>', unsafe_allow_html=True)
    st.markdown(
        "1. **Extract** the job's key requirements.\n"
        "2. **Retrieve** your matching experience by semantic similarity (a relevance threshold means "
        "a requirement with no real match shows up as a gap, not a fake match).\n"
        "3. **Generate** a tailored résumé from the retrieved evidence only — every bullet carries a "
        "`[SOURCE]` tag, and a second pass fact-checks it against your originals.")
    st.caption("Nothing is invented: gaps are reported honestly rather than bridged with fabricated experience.")


# ---------------------------------------------------------------- results ----

if "results" in st.session_state:
    r = st.session_state["results"]
    cov = r["coverage"]

    st.divider()
    resume_md, changed_md = split_output(r["tailored"])

    # Coverage summary strip
    st.markdown('<div class="eyebrow">Requirement coverage</div>', unsafe_allow_html=True)
    st.markdown(f"**{cov['covered']} / {cov['total']}** requirements matched real experience "
                f"&nbsp;·&nbsp; <span class='dim'>{cov['coverage_pct']}%</span>", unsafe_allow_html=True)
    st.markdown(f'<div class="cov-track"><div class="cov-fill" style="width:{cov["coverage_pct"]}%"></div></div>',
                unsafe_allow_html=True)
    if cov["gaps"]:
        st.markdown('<span class="dim">Honest gaps (no strong match — not bridged):</span>', unsafe_allow_html=True)
        st.markdown(" ".join(f'<span class="pill pill-gap">{g}</span>' for g in cov["gaps"]), unsafe_allow_html=True)
    if r.get("chunk_warning"):
        st.markdown(f'<div class="callout callout-warn dim">{r["chunk_warning"]}</div>', unsafe_allow_html=True)

    tabs = st.tabs(["Tailored résumé", "What changed", "Coverage detail", "Fact-check", "Diff", "Run log"])

    with tabs[0]:
        st.markdown(source_badges(resume_md), unsafe_allow_html=True)
        st.download_button("Download as Markdown", data=resume_md,
                           file_name="tailored_resume.md", mime="text/markdown")

    with tabs[1]:
        st.markdown(changed_md or "_The model didn't return a separate 'what changed' section._")

    with tabs[2]:
        for req_text, ev in r["evidence"].items():
            covered = bool(ev)
            pill = '<span class="pill pill-cov">covered</span>' if covered else '<span class="pill pill-gap">gap</span>'
            st.markdown(f"{pill} &nbsp; **{req_text}**", unsafe_allow_html=True)
            for e in ev:
                st.markdown(f'<span class="dim">· [{e["source"]} · d={e["distance"]}] {e["text"]}</span>',
                            unsafe_allow_html=True)

    with tabs[3]:
        fab = r["fabrication"]
        good = "no fabrication" in fab.lower() or "no fabrications" in fab.lower()
        st.markdown(f'<div class="callout {"callout-good" if good else "callout-warn"}">'
                    f'{"Independent check found nothing unsupported." if good else "Review the flagged items below."}'
                    f'</div>', unsafe_allow_html=True)
        st.markdown(fab)

    with tabs[4]:
        st.caption("Deterministic line-level diff (original résumé → tailored), independent of the model's self-report.")
        st.code(r["diff"] or "(no differences)", language="diff")

    with tabs[5]:
        st.caption(f"Provider: {pl.PROVIDER_CONFIG[r['provider']]['label']} · model: {r['model']} · "
                   f"{r['chunk_count']} chunks indexed · {len(r['call_log'])} model calls.")
        for c in r["call_log"]:
            st.markdown(f'<span class="dim">· {c["label"]} (prompt_len={c["prompt_len"]})</span>',
                        unsafe_allow_html=True)

    if st.button("Start over"):
        st.session_state.pop("results", None)
        st.rerun()
