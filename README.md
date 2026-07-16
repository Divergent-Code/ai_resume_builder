# AI Resume Builder

A hands-on workshop that builds an **AI resume-tailoring tool** in stages — from a single-prompt script to a production-style pipeline with retrieval, a multi-tool agent, and safety guardrails. Each notebook is self-contained and runs in Google Colab.

Given a job description, a resume, and (optionally) a GitHub username, the tool rewrites the resume to fit the role and explains *what changed and why* — while staying grounded in what the source material actually says.

## Notebooks

| Notebook | What it teaches | Open |
| --- | --- | --- |
| **V1 — Learning the Basics** | One Gemini prompt: paste a JD + resume, get a tailored resume back. Introduces the API and the "what changed and why" pattern. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/camunity/ai_resume_builder/blob/main/AI_Resume_Builder_v1.ipynb) |
| **V2 — Production AI Features** | Rebuilds the tool with production patterns: RAG (chunking, embeddings, retrieval, evaluation), a multi-tool agent with graceful failure, and AI-safety guardrails (prompt-injection screening, rate limiting, audit logging). | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/camunity/ai_resume_builder/blob/main/AI_Resume_Builder_v2_Phase2.ipynb) |

> **V1.1** is an interim step (URL-based JD fetch, resume file upload, graceful GitHub failures) summarized inside the V1 notebook rather than shipped as a separate file.

## Prerequisites

- A **Google Gemini API key** — create one at [Google AI Studio](https://aistudio.google.com/app/apikey).
- A Google account for Colab (recommended), or a local Jupyter environment.

## Running in Colab (recommended)

1. Open a notebook with a badge above.
2. Add your API key to Colab **Secrets**: click the 🔑 icon in the left sidebar, add a secret named `GOOGLE_API_KEY`, and enable notebook access.
3. Run the cells top to bottom. The first cell installs dependencies.

## Running locally

```bash
git clone https://github.com/camunity/ai_resume_builder.git
cd ai_resume_builder
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

Locally, the notebooks read the key from Colab Secrets, so replace the
`from google.colab import userdata` / `userdata.get('GOOGLE_API_KEY')` lines with
`os.environ["GOOGLE_API_KEY"]` and export the variable before launching Jupyter:

```bash
export GOOGLE_API_KEY="your-key-here"        # Windows: setx GOOGLE_API_KEY "your-key-here"
```

## A note on data / PII

Your resume (name, email, phone) and any scraped job description are sent to the Gemini API, and the optional GitHub step calls the public GitHub API. Don't paste anything you wouldn't share with a third-party service — prefer a redacted resume in a live workshop.

## License

Licensed under the **GNU GPL v3** — see [LICENSE](LICENSE).
