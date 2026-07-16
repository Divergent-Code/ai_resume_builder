# AI Résumé Builder

Tailors a résumé to a specific job using **retrieval-grounded generation**: it matches your real
experience to each job requirement, so it reframes what you actually have instead of inventing new
experience. Built for the AI-201 workshop, now with a web UI.

Works with **Anthropic, Gemini, DeepSeek, or Groq** — pick one; embeddings run locally, so retrieval
is identical whichever you choose.

## Web app (Streamlit)

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then set an API key one of three ways (in order of preference):

1. **Streamlit secrets** — create `.streamlit/secrets.toml`:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-…"   # or GOOGLE_API_KEY / DEEPSEEK_API_KEY / GROQ_API_KEY
   ```
2. **Environment variable** — `export ANTHROPIC_API_KEY=…`
3. **Sidebar field** — paste it in the running app (session-only; never committed).

Deploys as-is to **Streamlit Community Cloud** or **Hugging Face Spaces** (add the key as a secret there).

### What you get
- A grounded, **source-tagged** tailored résumé (`[SOURCE: RESUME | GITHUB | REFRAMED]`).
- A **coverage report** — which requirements matched real experience, and which are honest gaps.
- An independent **fabrication check** and a deterministic **diff** against your original.

### Architecture
- **`pipeline.py`** — provider-agnostic pipeline (LLM client, scraping, chunking, retrieval, generation,
  eval, fabrication check). No Streamlit or Colab dependency; importable anywhere.
- **`app.py`** — the Streamlit UI over that pipeline.

## Notebooks (workshop)
- `AI_Resume_Builder_v1.ipynb` — the basics: a single tailoring prompt.
- `AI_Resume_Builder_v2_Phase2.ipynb` — production patterns: RAG, a multi-tool agent, and safety guardrails.

Open them in Colab and add your provider's API key to the Secrets tab (🔑).

## License
GNU GPL v3 — see [LICENSE](LICENSE).
