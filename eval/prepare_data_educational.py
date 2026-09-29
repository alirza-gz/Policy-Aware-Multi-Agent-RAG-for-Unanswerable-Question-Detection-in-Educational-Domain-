"""Build the dedicated Educational RAG evaluation dataset (thesis advisor spec).  [v2 - fixed]

5 courses x ~500 questions = ~2500 questions, across 5 categories:
    answerable, unanswerable, out_of_domain, false_premise, partially_answerable
"""

import argparse
import asyncio
import hashlib
import json
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Optional

import httpx

COURSE_TOPICS = {
    "data_science": [
        "Data science", "Data cleansing", "Exploratory data analysis",
        "Data visualization", "Big data", "Data wrangling", "ETL",
        "Data pipeline", "Feature engineering", "A/B testing",
        "Data governance", "Business intelligence", "Data lake",
        "Data warehouse", "Dashboard (business)",
        "Data collection", "Data quality", "Data integration", "Data modeling",
        "Database normalization", "Metadata", "Data analysis", "Data management",
        "Master data management", "Predictive analytics", "Data engineering",
        "Data curation", "Information visualization", "Data mart",
        "Data architecture",
    ],
    "statistics": [
        "Statistics", "Standard deviation", "Variance", "Hypothesis testing",
        "p-value", "Confidence interval", "Regression analysis",
        "Correlation", "Normal distribution", "Central limit theorem",
        "Sampling (statistics)", "Analysis of variance", "Bayesian statistics",
        "Descriptive statistics", "Statistical significance",
        "Maximum likelihood estimation", "Student's t-test", "Chi-squared test",
        "Linear regression", "Statistical inference", "Type I and type II errors",
        "Effect size", "Sampling distribution", "Statistical power",
    ],
    "machine_learning": [
        "Machine learning", "Supervised learning", "Unsupervised learning",
        "Neural network", "Overfitting", "Gradient descent",
        "Decision tree learning", "Support vector machine", "Random forest",
        "K-means clustering", "Reinforcement learning", "Cross-validation (statistics)",
        "Feature selection", "Ensemble learning", "Convolutional neural network",
        "Logistic regression", "Backpropagation", "Regularization (mathematics)",
        "Bias–variance tradeoff", "Deep learning", "Recurrent neural network",
        "Transfer learning", "Hyperparameter optimization", "Semi-supervised learning",
    ],
    "data_mining": [
        "Data mining", "Association rule learning", "Cluster analysis",
        "Anomaly detection", "Decision tree", "Market basket analysis",
        "Text mining", "Web mining", "Data pre-processing",
        "Dimensionality reduction", "Outlier", "Classification (machine learning)",
        "Apriori algorithm", "Knowledge extraction", "Pattern recognition",
        "DBSCAN", "Hierarchical clustering", "K-nearest neighbors algorithm",
        "Naive Bayes classifier", "Process mining", "Data stream mining",
        "Local outlier factor", "Isolation forest", "Principal component analysis",
        "Information retrieval", "Sentiment analysis", "Recommender system",
        "Sequential pattern mining", "Collaborative filtering", "Data reduction",
    ],
    "probability": [
        "Probability theory", "Random variable", "Probability distribution",
        "Bayes' theorem", "Expected value", "Conditional probability",
        "Binomial distribution", "Poisson distribution", "Law of large numbers",
        "Markov chain", "Independence (probability theory)", "Combinatorics",
        "Joint probability distribution", "Probability density function", "Variance",
        "Stochastic process", "Monte Carlo method", "Exponential distribution",
        "Covariance", "Cumulative distribution function", "Law of total probability",
        "Random walk", "Geometric distribution", "Uniform distribution (continuous)",
    ],
}

CATEGORY_SPLIT = {
    "answerable": 200,
    "unanswerable": 100,
    "out_of_domain": 50,
    "false_premise": 75,
    "partially_answerable": 75,
}

CATEGORY_TO_FIELDS = {
    "answerable":           dict(gold_answerable=True,  expected_action="ANSWER"),
    "unanswerable":         dict(gold_answerable=False, expected_action="ABSTAIN"),
    "out_of_domain":        dict(gold_answerable=False, expected_action="ABSTAIN"),
    "false_premise":        dict(gold_answerable=False, expected_action="ABSTAIN"),
    "partially_answerable": dict(gold_answerable=False, expected_action="CLARIFY"),
}

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:3b-instruct"

BASE_SEED = 42


def _seed(attempt: int, retry: int) -> int:
    return BASE_SEED + attempt * 101 + retry * 13


WIKI_HEADERS = {
    "User-Agent": ("RA3G-EducationalDatasetBuilder/2.0 (thesis research project; contact: "
                   + os.environ.get("WIKI_CONTACT", "set-your-email-here") + ")")
}

_INVISIBLE = re.compile(r"[\u2060\u200b\u200c\u200d\ufeff]")


def _good_paragraph(p: str) -> bool:
    if not p:
        return False
    if "displaystyle" in p or "\\" in p or "{" in p:
        return False
    if not (p[0].isupper() or p[0].isdigit()):
        return False
    prose = sum(c.isalpha() or c.isspace() for c in p)
    return prose / len(p) >= 0.85


async def fetch_wikipedia_passages(client: httpx.AsyncClient, title: str) -> List[str]:
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query", "format": "json", "prop": "extracts",
        "explaintext": 1, "exsectionformat": "plain",
        "titles": title, "redirects": 1,
    }
    text = ""
    for attempt in range(4):
        await asyncio.sleep(0.6)
        try:
            resp = await client.get(url, params=params, timeout=20.0, headers=WIKI_HEADERS)
            if resp.status_code == 429:
                wait = float(resp.headers.get("Retry-After", 5 * (attempt + 1)))
                print(f"  [warn] 429 for '{title}', retrying in {wait:.0f}s (attempt {attempt + 1}/4)")
                await asyncio.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            for page in data.get("query", {}).get("pages", {}).values():
                text = page.get("extract", "") or ""
                break
            break
        except Exception as e:
            print(f"  [warn] failed to fetch '{title}' (attempt {attempt + 1}/4): {e}")
            await asyncio.sleep(2 * (attempt + 1))
    if not text:
        print(f"  [warn] giving up on '{title}' after retries")
        return []
    cleaned = []
    for p in text.split("\n"):
        p = re.sub(r"\s+", " ", _INVISIBLE.sub("", p)).strip()
        if 200 <= len(p) <= 1200 and _good_paragraph(p):
            cleaned.append(p)
    return cleaned


async def build_course_corpus(client: httpx.AsyncClient, course: str) -> List[Dict]:
    passages = []
    for title in COURSE_TOPICS[course]:
        paras = await fetch_wikipedia_passages(client, title)
        for p in paras:
            passages.append({"text": p, "title": title, "course": course})
        print(f"  [{course}] {title}: {len(paras)} passages")
    return passages


# ---------------------------------------------------------------------------
# Step 2: LLM calls  [PATCHED: retry-with-backoff + fallback off format=json]
# ---------------------------------------------------------------------------
OLLAMA_FAIL_STREAK = {"n": 0}  # tracks consecutive total failures for diagnostics


async def call_ollama_json(client: httpx.AsyncClient, prompt: str, seed: int = 42) -> Optional[Dict]:
    """POST to Ollama with retry-with-backoff on 5xx/timeout, and a fallback
    that drops "format": "json" on the last attempt (grammar-constrained JSON
    decoding is heavier and has been observed to trigger 500s on long runs;
    plain generation + manual JSON extraction is the lighter-weight path that
    worked reliably before format=json was added)."""
    for attempt in range(4):
        use_format_json = attempt < 3  # last attempt: plain generation, no grammar constraint
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.4, "seed": seed},
        }
        if use_format_json:
            payload["format"] = "json"
        try:
            resp = await client.post(OLLAMA_URL, json=payload, timeout=90.0)
            if resp.status_code >= 500:
                wait = 3 * (attempt + 1)
                print(f"  [warn] Ollama {resp.status_code} (attempt {attempt + 1}/4), "
                      f"retrying in {wait}s{' without format=json' if not use_format_json else ''}")
                await asyncio.sleep(wait)
                continue
            resp.raise_for_status()
            raw = (resp.json().get("response", "") or "").strip()
            if raw.startswith("```"):
                raw = re.sub(r"^```(json)?|```$", "", raw, flags=re.MULTILINE).strip()
            # LaTeX-style backslashes (\pi, \mu) are not valid JSON escapes.
            raw = re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", raw)
            try:
                out = json.loads(raw)
            except json.JSONDecodeError:
                m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
                out = json.loads(m.group(0)) if m else None
            if out is not None:
                OLLAMA_FAIL_STREAK["n"] = 0
                return out
        except Exception as e:
            print(f"  [warn] LLM call/parse failed (attempt {attempt + 1}/4): {e}")
            await asyncio.sleep(2 * (attempt + 1))

    OLLAMA_FAIL_STREAK["n"] += 1
    if OLLAMA_FAIL_STREAK["n"] in (5, 15, 30):
        print(f"  [!!] {OLLAMA_FAIL_STREAK['n']} consecutive Ollama failures -- the server itself "
              f"may be hung/crashed, not just overloaded. Consider checking from another terminal:\n"
              f"       curl http://localhost:11434/api/tags   (should respond instantly)\n"
              f"       nvidia-smi                             (check for OOM / stuck process)\n"
              f"       pkill -9 ollama && ollama serve &       (restart if the above hangs)")
    return None


ANSWERABLE_PROMPT = """You are building an educational quiz question strictly from the passage below.
Article title: {title}
Passage:
{passage}

Write ONE clear, natural, self-contained question a student could ask, whose
answer is explicitly stated in this passage, plus the short correct answer (a
few words, copied from the passage).
Rules:
- The question must name the specific subject (e.g. "In k-means clustering, ...").
  Never use vague references such as "this term", "the field", "it", "the model".
- The question must be understandable on its own, without seeing the passage,
  contain exactly one question mark, and start with a capital letter.
- Do NOT write negative questions ("What is not ...", "Which does not ...").
- Do NOT use ellipses ("..."), angle brackets, or LaTeX.
- The answer must be specific (a term, number, or short phrase) and must appear
  in the passage. Never answer with vague words like "a location" or "a method".
- Do NOT say "the passage", "the text", "this article", or any other meta-reference.
Return ONLY JSON:
{{"question": "...", "answer": "..."}}
"""

FALSE_PREMISE_STYLES = [
    "a 'Why' question that takes the wrong fact for granted",
    "a 'How' question that takes the wrong fact for granted",
    "a 'What' or 'Which' question that takes the wrong fact for granted",
    "a yes/no question that asserts the wrong fact",
    "a question of the form '<wrong claim>, right?'",
]

FALSE_PREMISE_PROMPT = """You are building a "false premise" test question strictly from the passage below.
Article title: {title}
Passage:
{passage}

Write ONE question that embeds a PLAUSIBLE BUT FACTUALLY WRONG assumption about
this passage's topic. The wrong assumption must be the OPPOSITE or a clear
distortion of something the passage actually states. Style: {style}.
Rules:
- Refer to the subject by name; never say "the passage", "the text" or any
  other meta-reference.
- Exactly one question mark; start with a capital letter; no ellipses, no angle brackets.
- The premise MUST really be false according to the passage. If you cannot
  build a question whose premise is false, set "premise_is_false" to false.
Also state briefly which part of the premise is false and why, citing the passage.
Return ONLY JSON:
{{"question": "...", "premise_is_false": true, "false_premise_explanation": "..."}}
"""

PARTIAL_PROMPT = """You are building material for a "partially answerable" test question.
Passage A (about topic {topic_a}):
{passage_a}

Passage B (about topic {topic_b}; it is NOT in the student's reference material):
{passage_b}

Return TWO independent sub-questions and their answers. Do NOT combine them.
- first_part_question: a short, self-contained question about {topic_a},
  answerable from Passage A alone. It must not mention {topic_b}.
- first_part_answer: short answer copied from Passage A.
- second_part_question: a short, self-contained question about {topic_b},
  answerable ONLY from Passage B, that names {topic_b}.
- second_part_answer: short answer copied from Passage B.
Rules for every question: exactly one question mark, start with a capital
letter, no negative form ("What is not ..."), no ellipses, no angle brackets,
and never mention "Passage A", "Passage B", "the passage" or "the text".
Return ONLY JSON:
{{"first_part_question": "...", "first_part_answer": "...",
  "second_part_question": "...", "second_part_answer": "..."}}
"""


META_LEAK_TERMS = [
    "the passage", "the text", "this article", "passage a", "passage b",
    "topic x", "topic y", "mentioned in the", "the full compound",
]
VAGUE_STARTS = ("this ", "these ", "it ", "the field ", "the term ")
VAGUE_ANSWERS = {"a location", "a method", "a few words", "a term", "a number", "a type"}
NEG_RE = re.compile(r"\b(is|are|was|were|does|do|did|has|have|can|could|should)\s+not\b|n't\b", re.I)

_STOP = set("""the and for are that this with what which how does did from have has can used use when why who
into than then their they its not any all one two type term called name known was were been being also
such each other more most some many much only over under about between during after before while where
whose whom would could should will shall may might must you your our""".split())


def _has_meta_leak(text: Optional[str]) -> bool:
    low = (text or "").lower()
    return any(t in low for t in META_LEAK_TERMS)


def _norm(text: Optional[str]) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", (text or "").lower())).strip()


def _tokens(text: Optional[str], min_len: int = 3) -> set:
    return {t for t in re.findall(r"[a-z0-9]+", (text or "").lower())
            if len(t) >= min_len and t not in _STOP}


def _anchor_ok(question: str, title: str) -> bool:
    title_toks = _tokens(title)
    if not title_toks:
        return True
    q_toks = _tokens(question, min_len=2)
    return any(any(qt.startswith(tt[:5]) for qt in q_toks) for tt in title_toks)


def _question_ok(q: Optional[str], title: str = "", allow_negation: bool = False) -> bool:
    q = (q or "").strip()
    if not q.endswith("?") or len(q.split()) < 5 or q.count("?") != 1:
        return False
    if any(ch in q for ch in "<>\n\\") or "..." in q or "…" in q:
        return False
    if not q[0].isupper():
        return False
    if _has_meta_leak(q) or q.lower().startswith(VAGUE_STARTS):
        return False
    if not allow_negation and NEG_RE.search(q):
        return False
    if title and not _anchor_ok(q, title):
        return False
    return True


def _answer_supported(answer: str, passage: str, thr: float = 0.8) -> bool:
    a = _norm(answer)
    if not a:
        return False
    if a in _norm(passage):
        return True
    a_toks = _tokens(answer, min_len=1)
    if not a_toks:
        return False
    return len(a_toks & _tokens(passage, min_len=1)) / len(a_toks) >= thr


def _q_grounded(question: str, passage: str, thr: float = 0.4) -> bool:
    q_toks = _tokens(question)
    if not q_toks:
        return False
    return len(q_toks & _tokens(passage, min_len=2)) / len(q_toks) >= thr


def _answer_ok(answer: str, question: str) -> bool:
    a = (answer or "").strip()
    if len(a) < 2 or a.lower() in VAGUE_ANSWERS or _has_meta_leak(a):
        return False
    if re.search(r"\b(not|no)\b.*\b(specified|mentioned|provided|stated)\b|\bcannot\b", a.lower()):
        return False
    return bool(_tokens(a, min_len=1) - _tokens(question, min_len=1))


def _answer_in_corpus(question: str, answer: str, corpus_sets: List[frozenset],
                      corpus_text: str, q_overlap: float = 0.5) -> bool:
    a_norm = _norm(answer)
    if a_norm and a_norm in corpus_text:
        return True
    a_toks = _tokens(answer, min_len=2)
    if not a_toks:
        return True
    q_toks = _tokens(question)
    if not q_toks:
        return True
    return any(a_toks <= ps and len(q_toks & ps) / len(q_toks) >= q_overlap for ps in corpus_sets)


def _is_dup(question: str, seen: set, seen_sets: List[frozenset], jac: float = 0.8) -> bool:
    key = _norm(question)
    if key in seen:
        return True
    ts = frozenset(_tokens(question, min_len=2))
    if ts:
        for s in seen_sets:
            if s and len(ts & s) / len(ts | s) >= jac:
                return True
    seen.add(key)
    seen_sets.append(ts)
    return False


def _compose(q1: str, q2: str) -> str:
    q1, q2 = q1.strip(), q2.strip()
    if len(q2) > 1 and q2[0].isupper() and q2[1].islower():
        q2 = q2[0].lower() + q2[1:]
    return f"{q1} And {q2}"


async def generate_answerable(client, passage: Dict, qid: str, attempt: int = 0) -> Optional[Dict]:
    for retry in range(2):
        out = await call_ollama_json(
            client, ANSWERABLE_PROMPT.format(title=passage["title"], passage=passage["text"]),
            seed=_seed(attempt, retry))
        if not out:
            continue
        q, ans = (out.get("question") or "").strip(), (out.get("answer") or "").strip()
        if not _question_ok(q, passage["title"]) or not _answer_ok(ans, q):
            continue
        if not _q_grounded(q, passage["text"]) or not _answer_supported(ans, passage["text"]):
            continue
        return {
            "id": qid, "course": passage["course"], "category": "answerable",
            "question": q, "gold_answers": [ans],
            "source_type": "wikipedia", "source_document": passage["title"],
            "evidence_passage": passage["text"],
            **CATEGORY_TO_FIELDS["answerable"],
        }
    return None


def _style_for(qid: str, attempt: int, retry: int) -> str:
    h = int(hashlib.md5(f"{qid}-{attempt}-{retry}".encode()).hexdigest(), 16)
    return FALSE_PREMISE_STYLES[h % len(FALSE_PREMISE_STYLES)]


async def generate_false_premise(client, passage: Dict, qid: str, attempt: int = 0) -> Optional[Dict]:
    for retry in range(2):
        prompt = FALSE_PREMISE_PROMPT.format(
            title=passage["title"], passage=passage["text"], style=_style_for(qid, attempt, retry))
        out = await call_ollama_json(client, prompt, seed=_seed(attempt, retry))
        if not out:
            continue
        if str(out.get("premise_is_false")).strip().lower() != "true":
            continue
        q = (out.get("question") or "").strip()
        expl = (out.get("false_premise_explanation") or "").strip()
        if not _question_ok(q, passage["title"], allow_negation=True) or not expl:
            continue
        if not _q_grounded(q, passage["text"], thr=0.3):
            continue
        return {
            "id": qid, "course": passage["course"], "category": "false_premise",
            "question": q, "gold_answers": [],
            "source_type": "wikipedia", "source_document": passage["title"],
            "evidence_passage": passage["text"],
            "note": expl,
            **CATEGORY_TO_FIELDS["false_premise"],
        }
    return None


async def generate_partial(client, passage_a: Dict, passage_b: Dict, qid: str, attempt: int,
                           corpus_sets: List[frozenset], corpus_text: str) -> Optional[Dict]:
    ta, tb = passage_a["title"], passage_b["title"]
    for retry in range(2):
        out = await call_ollama_json(
            client,
            PARTIAL_PROMPT.format(passage_a=passage_a["text"], passage_b=passage_b["text"],
                                  topic_a=ta, topic_b=tb),
            seed=_seed(attempt, retry))
        if not out:
            continue
        q1 = (out.get("first_part_question") or "").strip()
        a1 = (out.get("first_part_answer") or "").strip()
        q2 = (out.get("second_part_question") or "").strip()
        a2 = (out.get("second_part_answer") or "").strip()
        if not (q1 and a1 and q2 and a2):
            continue
        if not _question_ok(q1, ta) or not _answer_ok(a1, q1):
            continue
        if not _q_grounded(q1, passage_a["text"]) or not _answer_supported(a1, passage_a["text"]):
            continue
        if tb.lower() in q1.lower() or tb.lower() in a1.lower():
            continue
        if not _question_ok(q2, tb) or not _answer_ok(a2, q2):
            continue
        if not _q_grounded(q2, passage_b["text"]) or not _answer_supported(a2, passage_b["text"]):
            continue
        if _answer_in_corpus(q2, a2, corpus_sets, corpus_text, q_overlap=0.4):
            continue
        return {
            "id": qid, "course": passage_a["course"], "category": "partially_answerable",
            "question": _compose(q1, q2),
            "gold_answers": [a1],
            "answerable_part_question": q1,
            "out_of_scope_part_question": q2,
            "source_type": "wikipedia", "source_document": ta,
            "evidence_passage": passage_a["text"],
            "note": (f"second part ({q2}) is out of scope; its answer '{a2}' "
                     f"exists only in a held-out passage of '{tb}'"),
            **CATEGORY_TO_FIELDS["partially_answerable"],
        }
    return None


async def generate_heldout_unanswerable(client, passage: Dict, qid: str, attempt: int,
                                        corpus_sets: List[frozenset], corpus_text: str) -> Optional[Dict]:
    for retry in range(2):
        out = await call_ollama_json(
            client, ANSWERABLE_PROMPT.format(title=passage["title"], passage=passage["text"]),
            seed=_seed(attempt, retry))
        if not out:
            continue
        q, ans = (out.get("question") or "").strip(), (out.get("answer") or "").strip()
        if not _question_ok(q, passage["title"]) or not _answer_ok(ans, q):
            continue
        if not _q_grounded(q, passage["text"]) or not _answer_supported(ans, passage["text"]):
            continue
        if _answer_in_corpus(q, ans, corpus_sets, corpus_text):
            continue
        return {
            "id": qid, "course": passage["course"], "category": "unanswerable",
            "question": q, "gold_answers": [],
            "source_type": "wikipedia_heldout", "source_document": passage["title"],
            "note": f"answer exists only in a held-out passage (not in corpus): {ans}",
            "held_out_passage": passage["text"],
            **CATEGORY_TO_FIELDS["unanswerable"],
        }
    return None


OOD_MMLU_SUBJECTS = [
    "world_religions", "philosophy", "sociology", "astronomy", "public_relations",
    "nutrition", "marketing", "prehistory", "high_school_geography", "global_facts",
    "virology", "human_aging", "management", "us_foreign_policy", "security_studies",
    "clinical_knowledge", "anatomy", "high_school_us_history",
    "high_school_world_history", "business_ethics", "miscellaneous",
]
OOD_NEEDS_OPTIONS = re.compile(
    r"of the following|which statement|most correct|which one of|according to (the )?(passage|text)"
    r"|\bexcept\b|the (table|figure|graph|chart|map) (above|below|shown)", re.I)
OOD_DOMAIN_TERMS = re.compile(
    r"statist|probabilit|algorithm|regression|variance|standard deviation|machine learning|"
    r"neural|cluster|correlat|distribution|dataset|data set|database|big data|average", re.I)


def load_out_of_domain_questions(n_per_course: int, courses: List[str], rng: random.Random) -> List[Dict]:
    from datasets import load_dataset

    pool = []
    for subject in OOD_MMLU_SUBJECTS:
        try:
            pool.extend(load_dataset("cais/mmlu", subject, split="test"))
        except Exception as e:
            print(f"  [warn] could not load MMLU subject '{subject}': {e}")

    cleaned, seen = [], set()
    for r in pool:
        q = re.sub(r"\s+", " ", r["question"]).strip()
        if not q.endswith("?") or "____" in q or len(q.split()) < 6:
            continue
        if OOD_NEEDS_OPTIONS.search(q) or OOD_DOMAIN_TERMS.search(q):
            continue
        k = _norm(q)
        if k in seen:
            continue
        seen.add(k)
        cleaned.append({"question": q, "subject": r.get("subject", "")})
    if not cleaned:
        raise RuntimeError("No out-of-domain questions could be loaded from any MMLU subject.")

    needed = n_per_course * len(courses)
    if len(cleaned) < needed:
        print(f"  [shortfall] out_of_domain: only {len(cleaned)} usable MMLU questions for {needed} "
              f"needed; questions will repeat across courses. Add more OOD_MMLU_SUBJECTS.")
    rng.shuffle(cleaned)

    out, i = [], 0
    for course in courses:
        for k in range(n_per_course):
            row = cleaned[i % len(cleaned)]
            i += 1
            out.append({
                "id": f"{course}-ood-{k:04d}", "course": course, "category": "out_of_domain",
                "question": row["question"], "gold_answers": [],
                "source_type": "mmlu", "source_document": row["subject"],
                **CATEGORY_TO_FIELDS["out_of_domain"],
            })
    return out


async def fill_quota(gen_fn, sources: List, quota: int, prefix: str, seen: set,
                     seen_sets: List[frozenset], label: str, max_passes: int = 3) -> List[Dict]:
    rows: List[Dict] = []
    for attempt in range(max_passes):
        for src in sources:
            if len(rows) >= quota:
                break
            r = await gen_fn(src, f"{prefix}-{len(rows):04d}", attempt)
            if not r or _is_dup(r["question"], seen, seen_sets):
                continue
            rows.append(r)
        if len(rows) >= quota:
            break
    if len(rows) < quota:
        print(f"  [shortfall] {label}: got {len(rows)}/{quota} after {max_passes} passes "
              f"(add more Wikipedia titles or raise --max-passes)")
    return rows


def write_corpus(course: str, passages: List[Dict], corpus_dir: Path, clean: bool):
    corpus_dir.mkdir(parents=True, exist_ok=True)
    out_file = corpus_dir / f"edu2_{course}.txt"
    if clean and out_file.exists():
        out_file.unlink()
    seen, unique = set(), []
    for p in passages:
        if p["text"] not in seen:
            seen.add(p["text"])
            unique.append(p["text"])
    out_file.write_text("\n\n".join(unique), encoding="utf-8")
    print(f"  wrote {len(unique)} passages to {out_file}")


def final_validate(rows: List[Dict]) -> List[Dict]:
    kept, ids, qs = [], set(), set()
    dropped = 0
    for r in rows:
        q = r["question"].strip()
        r["question"] = q
        blob = (q + " " + " ".join(r.get("gold_answers", []))).lower()
        bad = (any(ch in q for ch in "<>\n") or "..." in q or "…" in q
               or re.search(r"passage [ab]\b|the passage|the text\b", blob)
               or r["id"] in ids or _norm(q) in qs
               or not q.endswith("?"))
        if bad:
            dropped += 1
            continue
        ids.add(r["id"])
        qs.add(_norm(q))
        kept.append(r)
    if dropped:
        print(f"  [validate] dropped {dropped} rows in final validation")
    return kept


async def main():
    global BASE_SEED
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-course-limit", type=int, default=500,
                        help="Target questions per course (scales all categories proportionally).")
    parser.add_argument("--corpus-dir", default="data/corpus")
    parser.add_argument("--out-suffix", default="")
    parser.add_argument("--clean-corpus", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-passes", type=int, default=3,
                        help="Max passes over the passages per category (different LLM seed each pass).")
    args = parser.parse_args()

    BASE_SEED = args.seed
    rng = random.Random(args.seed)
    courses = list(COURSE_TOPICS.keys())
    scale = args.per_course_limit / 500.0
    quota = {k: max(1, round(v * scale)) for k, v in CATEGORY_SPLIT.items()}

    corpus_by_course: Dict[str, List[Dict]] = {}
    heldout_by_course: Dict[str, List[Dict]] = {}

    async with httpx.AsyncClient() as client:
        for course in courses:
            print(f"\n=== {course} ===")
            passages = await build_course_corpus(client, course)
            if not passages:
                print(f"  [error] no passages fetched for {course}, skipping")
                corpus_by_course[course], heldout_by_course[course] = [], []
                continue
            rng.shuffle(passages)
            n_hold = min(len(passages) // 3, (quota["unanswerable"] + quota["partially_answerable"]) * 2)
            heldout_by_course[course] = passages[:n_hold]
            corpus_by_course[course] = passages[n_hold:]
            write_corpus(course, corpus_by_course[course], Path(args.corpus_dir), args.clean_corpus)
            print(f"  corpus: {len(corpus_by_course[course])} passages | held-out: {n_hold}")

        corpus_text = _norm(" ".join(p["text"] for c in courses for p in corpus_by_course[c]))
        corpus_sets = [frozenset(_tokens(p["text"], min_len=2))
                       for c in courses for p in corpus_by_course[c]]
        all_heldout = [p for c in courses for p in heldout_by_course[c]]

        all_rows: List[Dict] = []
        seen: set = set()
        seen_sets: List[frozenset] = []
        for course in courses:
            corpus = corpus_by_course[course]
            held = heldout_by_course[course]
            if not corpus:
                continue
            print(f"\n--- generating questions for {course} ---")
            src_a = list(corpus); rng.shuffle(src_a)
            src_f = list(corpus); rng.shuffle(src_f)
            src_p = list(corpus); rng.shuffle(src_p)
            src_h = list(held); rng.shuffle(src_h)

            async def gen_ans(p, qid, attempt):
                return await generate_answerable(client, p, qid, attempt)

            async def gen_fp(p, qid, attempt):
                return await generate_false_premise(client, p, qid, attempt)

            async def gen_part(p, qid, attempt):
                same = [h for h in held if h["title"] != p["title"]]
                pool_b = same or [h for h in all_heldout if h["title"] != p["title"]]
                if not pool_b:
                    return None
                return await generate_partial(client, p, rng.choice(pool_b), qid, attempt,
                                              corpus_sets, corpus_text)

            async def gen_unans(p, qid, attempt):
                return await generate_heldout_unanswerable(client, p, qid, attempt,
                                                           corpus_sets, corpus_text)

            mp = args.max_passes
            print(f"  answerable ({quota['answerable']})...")
            all_rows += await fill_quota(gen_ans, src_a, quota["answerable"], f"{course}-ans",
                                         seen, seen_sets, f"{course}/answerable", mp)
            print(f"  false_premise ({quota['false_premise']})...")
            all_rows += await fill_quota(gen_fp, src_f, quota["false_premise"], f"{course}-fp",
                                         seen, seen_sets, f"{course}/false_premise", mp)
            print(f"  partially_answerable ({quota['partially_answerable']})...")
            all_rows += await fill_quota(gen_part, src_p, quota["partially_answerable"], f"{course}-part",
                                         seen, seen_sets, f"{course}/partially_answerable", mp)
            print(f"  unanswerable ({quota['unanswerable']})...")
            all_rows += await fill_quota(gen_unans, src_h, quota["unanswerable"], f"{course}-unans",
                                         seen, seen_sets, f"{course}/unanswerable", mp)

    print("\nloading out-of-domain questions (MMLU)...")
    all_rows.extend(load_out_of_domain_questions(quota["out_of_domain"], courses, rng))

    all_rows = final_validate(all_rows)
    rng.shuffle(all_rows)
    out_path = Path(f"data/eval/educational_v2_questions{args.out_suffix}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    by_cat: Dict[str, int] = {}
    for r in all_rows:
        by_cat[r["category"]] = by_cat.get(r["category"], 0) + 1
    print(f"\n[prepare_data_educational] wrote {len(all_rows)} questions to {out_path}")
    print("category counts:", by_cat)
    print("targets         :", {k: quota[k] * len(courses) for k in CATEGORY_SPLIT})


if __name__ == "__main__":
    asyncio.run(main())