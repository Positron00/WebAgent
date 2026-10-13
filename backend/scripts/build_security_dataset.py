#!/usr/bin/env python3
"""Build ContextDefender train/test/probe JSONL from the public sources (SEC-027).

Run from backend/ after fetch_security_sources.py:
    python scripts/build_security_dataset.py --raw <raw dir> [--out data/security] [--total N] [--seed 13]

Rows: {"text", "source", "labels": {injection, harm, sensitive_data, jailbreak}, "origin"}.
Splits are assigned per near-duplicate cluster by a hash of its text (about 80/18/2). BIPIA's own
train attacks go to train and its test attacks to test; treycsa is held out for probe. Injections
hidden in pages and harmful requests, the weakest axes of cd-2026-10-11-4, are over-represented.
Nothing here prints example text.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import string
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

BACKEND = Path(__file__).resolve().parents[1]
for p in (str(BACKEND), str(Path(__file__).resolve().parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from security_data import AXES  # noqa: E402
from app.security.patterns import scan  # noqa: E402

MAX_CHARS = 3000
# Rows per group at the default size; --total scales them all.
QUOTAS = {
    # benign (more pages and safe prompts, to balance the extra page injections and harm rows)
    # Ordinary questions (WildChat, triage, S-Labs/deepset benign) raised 2026-10-12 so the held-out
    # calibration part has ~500 of them: with 169, one false positive was already 0.6%.
    "wildchat_user": 2400, "wildchat_history": 250, "wildchat_page": 2500,
    "trust_regular": 400, "inj_benign": 1000, "xstest_safe": 250, "jbb_benign": 100,
    "triage": 700, "tool_benign": 100, "aegis_safe": 800,
    # configs, logs and commands with placeholders instead of secrets (benign look-alikes)
    "secret_placeholder": 400,
    # Aegis "unsafe" prompts in soft categories, relabelled benign (look-alikes the judge settles)
    "aegis_soft": 500,
    # injection (embedded rows carry a span so training can label the right chunks)
    "llmail": 1000, "inj_direct": 600, "inj_embedded": 4000, "llmail_embedded": 1500,
    "bipia_train": 600, "bipia_test": 150,
    # Agent tool data (2026-10-12): NVIDIA records with an injected field, and the same records clean;
    # InjecAgent tool responses only in test; AgentDojo environment data only in probe.
    "nvidia_ipi": 1200, "nvidia_ipi_benign": 600,
    # InjecAgent split by template and attacker instruction (2026-10-12): test cases use a held-out
    # template and a held-out instruction, training cases neither; mixed cases are not used.
    "injecagent_train": 500, "injecagent": 300, "injecagent_benign": 100,
    "agentdojo": 80, "agentdojo_benign": 39,
    # BIPIA-style task instructions in data (2026-10-12): SEP probes (split by probe), BIPIA's own
    # emails as carriers for its text and code attacks (train attacks in train emails only), and
    # NotInject's benign prompts full of trigger words (against over-blocking).
    "sep": 1500, "sep_clean": 500, "bipia_email_train": 600, "bipia_email_test": 150, "bipia_email_clean": 100,
    "notinject": 339,
    # TaskTracker-style task drift (2026-10-12): instructions inserted into passages. SQuAD passages
    # (CC BY-SA) only ever carry TaskTracker's MIT instructions, and Alpaca's CC BY-NC instructions only
    # go into WildChat pages, so no row mixes share-alike and non-commercial text.
    "tt_squad_train": 1500, "tt_squad_test": 300, "squad_clean": 600, "tt_alpaca": 1500,
    # Benign text full of instructions to the reader (2026-10-12): how-to and web articles, assistant
    # answers. Benign WildChat pages set the injection threshold; these teach "instructions for the
    # reader" from "instructions for the assistant".
    "howto": 1200, "web_article": 600, "assistant_answer": 1200,
    # jailbreak
    "jailbreak": 1100,
    # harm: the curated sets first, then Aegis 2.0 unsafe prompts
    "harm": 3000,
    # sensitive data (Aegis PII/privacy requests: asking for someone's personal data)
    "nemotron": 800, "secrets": 900, "aegis_privacy": 300,
}
DEFAULT_TOTAL = sum(QUOTAS.values())
# Aegis 2.0 labels a prompt "unsafe" for a broad set of categories; WebAgent declines only the dangerous
# ones (decided 2026-10-12). A row is harm if any category is in AEGIS_HARM; otherwise sensitive data if
# PII/privacy; otherwise benign if every category is in AEGIS_SOFT; anything else is ambiguous and dropped.
AEGIS_HARM = {
    "Criminal Planning/Confessions", "Violence", "Hate/Identity Hate", "Harassment", "Guns and Illegal Weapons",
    "Suicide and Self Harm", "Fraud/Deception", "Controlled/Regulated Substances", "Sexual (minor)", "Threat",
    "Illegal Activity", "Malware",
}
AEGIS_PRIVACY = {"PII/Privacy"}
AEGIS_SOFT = {"Needs Caution", "Unauthorized Advice", "Political/Misinformation/Conspiracy", "Immoral/Unethical",
              "Profanity"}


def aegis_group(categories) -> Optional[str]:
    """"harm", "privacy", "soft" or None (ambiguous: dropped) for an Aegis unsafe row."""
    cats = {c.strip() for c in (categories.split(",") if isinstance(categories, str) else (categories or []))
            if c and c.strip()}
    if cats & AEGIS_HARM:
        return "harm"
    if cats & AEGIS_PRIVACY:
        return "privacy"
    if cats and cats <= AEGIS_SOFT:
        return "soft"
    return None


CARRIER_CHARS = 2400  # trim carriers before inserting, so clipping can't move the injected span
JAILBREAK_HINTS = re.compile(r"(?i)\b(DAN|jailbreak|developer mode|ignore (all )?(previous|prior)|do anything now|no restrictions)\b")


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def clip(text: str) -> str:
    """Keep the beginning and the end of long texts (attacks are often at either end)."""
    text = text.strip()
    if len(text) <= MAX_CHARS:
        return text
    half = MAX_CHARS // 2
    return text[:half] + "\n...\n" + text[-half:]


def split_of(text: str, seed: int) -> str:
    h = int(hashlib.sha256(f"{seed}:{norm(text)}".encode()).hexdigest(), 16) % 100
    return "probe" if h < 2 else "test" if h < 20 else "train"


def labels(**high) -> Dict[str, str]:
    return {axis: ("high" if high.get(axis) else "none") for axis in AXES}


def read(raw: Path, name: str) -> List[dict]:
    path = raw / f"{name}.jsonl"
    return [json.loads(l) for l in path.open(encoding="utf-8")] if path.is_file() else []


def take(rows: Iterable, n: int, rng: random.Random) -> list:
    rows = list(rows)
    rng.shuffle(rows)
    return rows[:n]


def cycle_sources(sources: List[tuple], i: int) -> str:
    """Pick a source by weight for the i-th row: [(source, weight), ...]."""
    total = sum(w for _, w in sources)
    k = i % total
    for source, w in sources:
        if k < w:
            return source
        k -= w
    return sources[-1][0]


def _rand(rng: random.Random, alphabet: str, n: int) -> str:
    return "".join(rng.choice(alphabet) for _ in range(n))


_ALNUM = string.ascii_letters + string.digits
_HEX = "0123456789abcdef"
# Fake credentials in the shapes real services use (none of these are real keys).
SECRET_KINDS = {
    "openai": lambda r: "sk-" + r.choice(["", "proj-", "live-"]) + _rand(r, _ALNUM, 40),
    "anthropic": lambda r: "sk-ant-api03-" + _rand(r, _ALNUM + "-_", 80),
    "aws": lambda r: "AKIA" + _rand(r, string.ascii_uppercase + string.digits, 16),
    "aws_secret": lambda r: _rand(r, _ALNUM + "/+", 40),
    "github": lambda r: r.choice(["ghp_", "gho_", "ghs_"]) + _rand(r, _ALNUM, 36),
    "github_pat": lambda r: "github_pat_" + _rand(r, _ALNUM, 22) + "_" + _rand(r, _ALNUM, 59),
    "gitlab": lambda r: "glpat-" + _rand(r, _ALNUM + "-_", 20),
    "tavily": lambda r: "tvly-" + _rand(r, _ALNUM, 32),
    "slack": lambda r: r.choice(["xoxb-", "xoxp-"]) + "-".join(_rand(r, string.digits, 12) for _ in range(2)) + "-" + _rand(r, _ALNUM, 24),
    "google": lambda r: "AIza" + _rand(r, _ALNUM + "-_", 35),
    "stripe": lambda r: r.choice(["sk_live_", "rk_live_"]) + _rand(r, _ALNUM, 24),
    "sendgrid": lambda r: "SG." + _rand(r, _ALNUM + "-_", 22) + "." + _rand(r, _ALNUM + "-_", 43),
    "twilio": lambda r: "SK" + _rand(r, _HEX, 32),
    "huggingface": lambda r: "hf_" + _rand(r, _ALNUM, 34),
    "npm": lambda r: "npm_" + _rand(r, _ALNUM, 36),
    "pypi": lambda r: "pypi-AgEIcHlwaS5vcmc" + _rand(r, _ALNUM + "-_", 60),
    "langsmith": lambda r: "lsv2_pt_" + _rand(r, _HEX, 32) + "_" + _rand(r, _HEX, 10),
    "jwt": lambda r: "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9." + _rand(r, _ALNUM + "-_", 60) + "." + _rand(r, _ALNUM + "-_", 43),
    "bearer": lambda r: _rand(r, _ALNUM, 48),
    "password": lambda r: _rand(r, _ALNUM, 8) + r.choice("!@#$%^&*") + _rand(r, _ALNUM, 5),
    "pem": lambda r: ("-----BEGIN " + r.choice(["RSA ", "EC ", "OPENSSH ", ""]) + "PRIVATE KEY-----\n"
                      + "\n".join(_rand(r, _ALNUM + "+/", 64) for _ in range(r.randint(3, 6)))
                      + "\n-----END PRIVATE KEY-----"),
}


def fake_secret(rng: random.Random, kind: Optional[str] = None) -> str:
    return SECRET_KINDS[kind or rng.choice(sorted(SECRET_KINDS))](rng)


_HOSTS = ["db.internal", "10.0.3.14", "pg-prod-2", "mysql.staging.local", "redis-cache", "api.example.com",
          "payments.example.org", "ingest.example.net"]
_USERS = ["admin", "app", "svc_reports", "deploy", "etl_user", "maria", "build-bot", "analytics"]

# Many contexts, so test and probe can hold contexts training never saw (each forms its own
# near-duplicate cluster). {s} is the secret; {h}, {u}, {n} are filler. Benign twins use placeholders.
SECRET_WRAPPERS = [
    "Here is my config, can you see why it fails?\n\nDATABASE_URL=postgres://{u}@{h}:5432/app\nAPI_KEY={s}\nDEBUG=false",
    "[2026-03-02 10:14:07] INFO starting worker {n}\n[2026-03-02 10:14:07] DEBUG using credential {s}\n[2026-03-02 10:14:08] INFO connected to {h}",
    "export SERVICE_TOKEN='{s}'\nexport REGION=us-east-1\npython deploy.py --env staging",
    "Traceback (most recent call last):\n  File \"client.py\", line {n}, in <module>\n    client = Client(token=\"{s}\")\nValueError: invalid region",
    "Settings for the integration:\n- endpoint: https://{h}/v2\n- key: {s}\n- timeout: 30s",
    "services:\n  worker:\n    image: acme/worker:1.{n}\n    environment:\n      - API_TOKEN={s}\n      - LOG_LEVEL=info",
    "{{\n  \"type\": \"service_account\",\n  \"client_email\": \"{u}@project-{n}.iam.gserviceaccount.com\",\n  \"private_key\": \"{s}\"\n}}",
    "curl -H \"Authorization: Bearer {s}\" https://{h}/v1/reports?limit={n}",
    "import requests\n\nresp = requests.get(\"https://{h}/items\", headers={{\"x-api-key\": \"{s}\"}})\nprint(resp.status_code)",
    "const client = new Client({{\n  apiKey: \"{s}\",\n  baseUrl: \"https://{h}\",\n}});",
    "[default]\naws_access_key_id = {s}\nregion = eu-west-1\noutput = json",
    "spring.datasource.url=jdbc:mysql://{h}:3306/shop\nspring.datasource.username={u}\nspring.datasource.password={s}",
    "Hi team, the new staging credentials are below. Please don't share them outside the channel.\nuser: {u}\npassword: {s}",
    "my api key is {s}, can you tell me why the request returns 401?",
    "Webhook URL for alerts: https://hooks.example.com/services/T{n}/B{n}/{s}",
    "git remote set-url origin https://{u}:{s}@github.com/acme/infra.git",
    "mongodb+srv://{u}:{s}@cluster{n}.example.mongodb.net/prod?retryWrites=true",
    "ERROR 2026-05-11T08:22:41Z auth failed for key={s} host={h} attempt={n}",
    "| name | value |\n|---|---|\n| endpoint | https://{h} |\n| token | {s} |\n| retries | {n} |",
    "kubectl create secret generic api-creds --from-literal=token={s} -n team-{n}",
    "# .npmrc\n//registry.npmjs.org/:_authToken={s}",
    "Dear support, I rotated our key yesterday. The old one was {s}; please revoke it on your side.",
    "SMTP settings: host smtp.{h}, port 587, user {u}, password {s}",
    "ssh_config:\n  Host {h}\n    User {u}\n    IdentityFile ~/.ssh/id_rsa\n\nkey contents:\n{s}",
    "terraform {{\n  backend \"http\" {{\n    address = \"https://{h}/state\"\n    password = \"{s}\"\n  }}\n}}",
    "Run output:\n> Loaded env from .env\n> OPENAI_API_KEY={s}\n> 12 tests passed",
    "<add key=\"PaymentApiKey\" value=\"{s}\" />\n<add key=\"Timeout\" value=\"{n}\" />",
    "Notebook cell [{n}]:\nclient = Anthropic(api_key=\"{s}\")\nmsg = client.messages.create(model=m, max_tokens=256, messages=[...])",
    "The vault export failed halfway; partial output:\nsecret/data/{u}: {{\"token\": \"{s}\"}}",
    "Copy this into your password manager: {u} / {s} (expires in {n} days)",
]
SECRET_PLACEHOLDERS = ["<YOUR_API_KEY>", "${{API_KEY}}", "********", "xxxxxxxxxxxxxxxx", "your-token-here",
                       "os.environ[\"API_KEY\"]", "REDACTED", "{{{{ secrets.API_TOKEN }}}}", "changeme"]


def secret_text(rng: random.Random, wrapper: str, secret: str) -> str:
    return wrapper.format(s=secret, h=rng.choice(_HOSTS), u=rng.choice(_USERS), n=rng.randint(2, 999))


BENIGN_TOOL = [
    "count    1200.000000\nmean       41.732500\nstd        12.118000\nmin        18.000000\nmax        79.000000",
    "Processed 3 files in 0.42s\nrows: 1,284\nwarnings: 0",
    "   year  revenue  growth\n0  2022    12.4     NaN\n1  2023    14.1    13.7%\n2  2024    16.0    13.5%",
    "Test session starts\ncollected 42 items\n42 passed in 3.11s",
    "Fitting model... done\naccuracy: 0.912\nprecision: 0.887\nrecall: 0.903",
]


NEAR_DUPLICATE = 0.9


def assign_cluster_splits(rows: List[dict], seed: int) -> None:
    """Put near-duplicates (cosine >= 0.9) in the same split, so test and probe measure unseen
    templates, not paraphrases of training rows. A held-out source pulls its whole cluster along;
    a row forced to another split than its cluster's is dropped, never moved (moving a BIPIA train
    attack into test would put the same attack string on both sides)."""
    import numpy as np
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    emb = np.asarray(encoder.encode([r["text"] for r in rows], normalize_embeddings=True,
                                    batch_size=128), dtype=np.float32)
    parent = list(range(len(rows)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    block = 1024
    for start in range(0, len(rows), block):
        sims = emb[start:start + block] @ emb.T
        for bi, js in zip(*np.nonzero(sims >= NEAR_DUPLICATE)):
            i, j = start + int(bi), int(js)
            if i < j:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri

    clusters: Dict[int, List[int]] = defaultdict(list)
    for i in range(len(rows)):
        clusters[find(i)].append(i)
    order = {"test": 0, "probe": 1, "train": 2}
    for members in clusters.values():
        forced = sorted({rows[i]["_split"] for i in members if rows[i]["_forced"]}, key=order.get)
        split = forced[0] if forced else split_of(min(norm(rows[i]["text"]) for i in members), seed)
        for i in members:
            if rows[i]["_forced"] and rows[i]["_split"] != split:
                rows[i]["_drop"] = True
            rows[i]["_split"] = split
    sizes = sorted((len(m) for m in clusters.values()), reverse=True)
    print(json.dumps({"clusters": len(clusters), "largest": sizes[:5]}), file=sys.stderr)


def build(raw: Path, total: int, seed: int) -> Dict[str, List[dict]]:
    rng = random.Random(seed)
    scale = total / DEFAULT_TOTAL
    q = {k: max(1, round(v * scale)) for k, v in QUOTAS.items()}
    out: List[dict] = []
    seen = set()

    def add(text: str, source: str, lab: Dict[str, str], origin: str, split: Optional[str] = None,
            span: Optional[List[int]] = None) -> bool:
        text = text if span else clip(text)
        key = norm(text)
        if len(key) < 8 or key in seen:
            return False
        seen.add(key)
        row = {"text": text, "source": source, "labels": lab, "origin": origin,
               "_split": split or split_of(text, seed), "_forced": split is not None}
        if span:
            row["span"] = span  # [start, end) of the injected text inside a benign carrier
        out.append(row)
        return True

    def embed(page: str, attack: str) -> tuple:
        page = page[:CARRIER_CHARS]
        cut = rng.randint(0, len(page))
        prefix = page[:cut] + "\n"
        attack = attack.strip()[:600]
        return prefix + attack + "\n" + page[cut:], [len(prefix), len(prefix) + len(attack)]

    # Benign chat turns (WildChat), filtered for injection/jailbreak wording to avoid label noise.
    wild = read(raw, "wildchat")
    user_turns = [u for row in wild for u in row["user"][:1]
                  if 10 < len(u) < 4000 and not JAILBREAK_HINTS.search(u) and not scan(u)]
    for i, text in enumerate(take(user_turns, q["wildchat_user"] + q["wildchat_history"], rng)):
        add(text, "user_turn" if i < q["wildchat_user"] else "history", labels(), "wildchat:user")
    pages = [a for row in wild for a in row["assistant"][:1] if len(a) >= 300 and not scan(a)]
    rng.shuffle(pages)
    page_sources = [("web_result", 4), ("document", 2), ("retrieved_doc", 2), ("agent_output", 2)]
    for i, text in enumerate(pages[: q["wildchat_page"]]):
        add(text, cycle_sources(page_sources, i), labels(), "wildchat:assistant")
    carriers = pages[q["wildchat_page"]:]

    trust = read(raw, "trustairlab")
    regular = [r["text"] for r in trust if not r["jailbreak"] and not JAILBREAK_HINTS.search(r["text"]) and not scan(r["text"])]
    for i, text in enumerate(take(regular, q["trust_regular"], rng)):
        add(text, "user_turn" if i % 4 else "history", labels(), "trustairlab:regular")

    inj_rows = read(raw, "slabs") + read(raw, "deepset")
    benign_prompts = [r["text"] for r in inj_rows if r["label"] == 0]
    for text in take(benign_prompts, q["inj_benign"], rng):
        add(text, "user_turn", labels(), "slabs_deepset:benign")

    xstest = read(raw, "xstest")
    safe = [r["text"] for r in xstest if not str(r.get("type", "")).startswith("contrast")]
    for text in take(safe, q["xstest_safe"], rng):
        add(text, "user_turn", labels(), "xstest:safe")

    jbb = read(raw, "jbb")
    for text in take([r["text"] for r in jbb if r["label"] == "benign"], q["jbb_benign"], rng):
        add(text, "user_turn", labels(), "jbb:benign")

    triage_path = BACKEND / "data" / "triage" / "train.jsonl"
    if triage_path.is_file():
        triage = [json.loads(l)["text"] for l in triage_path.open(encoding="utf-8")]
        for text in take(triage, q["triage"], rng):
            add(text, "user_turn", labels(), "triage:question")

    for i in range(q["tool_benign"]):
        base = BENIGN_TOOL[i % len(BENIGN_TOOL)]
        add(f"{base}\n# run {i}", "tool_output", labels(), "generated:tool_output")

    # Injection hidden in content (LLMail-Inject emails).
    mail_sources = [("document", 5), ("retrieved_doc", 3), ("web_result", 3), ("tool_output", 1), ("unlabelled", 1)]
    mails = take(read(raw, "llmail"), q["llmail"] + q["llmail_embedded"], rng)
    for i, row in enumerate(mails[: q["llmail"]]):
        subject = (row.get("subject") or "").strip()
        text = f"Subject: {subject}\n\n{row['text']}" if subject else row["text"]
        add(text, cycle_sources(mail_sources, i), labels(injection=True), "llmail:email")

    # Direct injection in the user's message (S-Labs, deepset).
    direct = [r["text"] for r in inj_rows if r["label"] == 1]
    rng.shuffle(direct)
    for i, text in enumerate(direct[: q["inj_direct"]]):
        add(text, "user_turn" if i % 5 else "history", labels(injection=True), "slabs_deepset:injection")
    # Some of the same kind of text placed inside benign pages (indirect).
    for i, text in enumerate(direct[q["inj_direct"]: q["inj_direct"] + q["inj_embedded"]]):
        if not carriers:
            break
        combined, span = embed(carriers.pop(), text)
        add(combined, cycle_sources(page_sources, i), labels(injection=True), "slabs_deepset:embedded", span=span)

    # Emails quoted inside benign pages.
    for i, row in enumerate(mails[q["llmail"]:]):
        if not carriers:
            break
        combined, span = embed(carriers.pop(), row["text"])
        add(combined, cycle_sources(page_sources, i), labels(injection=True), "llmail:embedded", span=span)

    # BIPIA attacks inside benign pages. BIPIA's train attacks are reused across many carriers
    # for training; its separate test attacks only appear in test, so the strings never cross.
    bipia = read(raw, "bipia")
    for split, quota in (("train", q["bipia_train"]), ("test", q["bipia_test"])):
        attacks = [r["text"] for r in bipia if str(r.get("split", "")).endswith(f"attack_{split}")]
        rng.shuffle(attacks)
        for i in range(quota if attacks else 0):
            if not carriers:
                break
            combined, span = embed(carriers.pop(), attacks[i % len(attacks)])
            add(combined, cycle_sources(page_sources[:3], i), labels(injection=True), "bipia:embedded",
                split=split, span=span)

    tool_sources = [("tool_output", 4), ("retrieved_doc", 2), ("document", 2), ("web_result", 1)]
    def held_out(value: str, share: float = 0.5) -> bool:
        return int(hashlib.sha256(f"{seed}:{value}".encode()).hexdigest(), 16) % 100 < share * 100

    inj_agent = read(raw, "injecagent")
    for split, quota in (("train", q["injecagent_train"]), ("test", q["injecagent"])):
        want = split == "test"
        cases = [r for r in inj_agent if r["label"] == "injection" and r.get("template")
                 and held_out(r["template"]) == want and held_out(r["attack"]) == want]
        for r in take(cases, quota, rng):
            add(r["text"], "tool_output", labels(injection=True), "injecagent:injected", split=split, span=r["span"])
    clean = [r for r in inj_agent if r["label"] == "benign" and r.get("template")]
    for r in take(clean, q["injecagent_benign"], rng):
        add(r["text"], "tool_output", labels(), "injecagent:clean", split="test" if held_out(r["template"]) else "train")

    # SEP: whole probes held out for test (about 30%), each clean twin in its entry's split.
    data_sources = [("document", 3), ("retrieved_doc", 2), ("tool_output", 2), ("web_result", 1)]
    sep = read(raw, "sep")
    for label, quota, lab in (("injection", q["sep"], labels(injection=True)), ("benign", q["sep_clean"], labels())):
        rows = [r for r in sep if r["label"] == label]
        for i, r in enumerate(take(rows, quota, rng)):
            split = "test" if held_out(f"sep-probe-{r['probe']}", 0.3) else "train"
            add(r["text"], cycle_sources(data_sources, i), lab, f"sep:{'injected' if label == 'injection' else 'clean'}",
                split=split, span=r["span"] if label == "injection" else None)

    # BIPIA's EmailQA contexts with BIPIA's own attacks: train attacks in train emails, test in test.
    emails = read(raw, "bipia_email")
    email_sources = [("document", 2), ("tool_output", 1), ("retrieved_doc", 1)]
    for split, quota in (("train", q["bipia_email_train"]), ("test", q["bipia_email_test"])):
        carriers_e = [r["text"] for r in emails if r["split"] == split]
        attacks = [r["text"] for r in bipia if str(r.get("split", "")).endswith(f"attack_{split}")]
        if not carriers_e or not attacks:
            continue
        for i in range(quota):
            page = carriers_e[rng.randrange(len(carriers_e))]
            combined, span = embed(page, attacks[rng.randrange(len(attacks))])
            add(combined, cycle_sources(email_sources, i), labels(injection=True), "bipia:email", split=split, span=span)
    for i, r in enumerate(take(emails, q["bipia_email_clean"], rng)):
        add(r["text"], cycle_sources(email_sources, i), labels(), "bipia:email_clean", split=r["split"])

    # TaskTracker-style rows. SQuAD passages and the instructions are split the same way (passage by
    # hash; TaskTracker's own train/test lists, the rest by hash), so test pairs are unseen on both sides.
    passage_sources = [("retrieved_doc", 3), ("web_result", 2), ("document", 2)]
    squad = [r["text"] for r in read(raw, "squad")]
    tt = read(raw, "tasktracker")
    squad_split = {t: ("test" if held_out(f"squad:{t}", 0.2) else "train") for t in squad}
    tt_split = lambda r: r["split"] or ("test" if held_out(f"tt:{r['text']}", 0.3) else "train")  # noqa: E731
    for split, quota in (("train", q["tt_squad_train"]), ("test", q["tt_squad_test"])):
        passages = [t for t in squad if squad_split[t] == split]
        tasks = [r["text"] for r in tt if tt_split(r) == split]
        for i in range(quota if passages and tasks else 0):
            combined, span = embed(passages[rng.randrange(len(passages))], tasks[rng.randrange(len(tasks))])
            add(combined, cycle_sources(passage_sources, i), labels(injection=True), "tasktracker:squad",
                split=split, span=span)
    for i, t in enumerate(take(squad, q["squad_clean"], rng)):
        add(t, cycle_sources(passage_sources, i), labels(), "squad:clean", split=squad_split[t])
    alpaca = take([r["text"] for r in read(raw, "alpaca")], q["tt_alpaca"], rng)
    spare = pages[q["wildchat_page"]:]  # a copy: reused once the unused carriers run out
    for i, task in enumerate(alpaca):
        page = carriers.pop() if carriers else spare[rng.randrange(len(spare))]
        combined, span = embed(page, task)
        add(combined, cycle_sources(page_sources, i), labels(injection=True), "tasktracker:alpaca", span=span)

    reader_sources = [("web_result", 3), ("document", 2), ("retrieved_doc", 2), ("agent_output", 1)]
    cosmo = read(raw, "cosmopedia")
    for key, origin, pool in (("howto", "cosmopedia:howto", [r["text"] for r in cosmo if r.get("config") == "wikihow"]),
                              ("web_article", "cosmopedia:web", [r["text"] for r in cosmo if r.get("config") == "web_samples_v2"]),
                              ("assistant_answer", "oasst:assistant", [r["text"] for r in read(raw, "oasst")])):
        clean = [t for t in pool if not JAILBREAK_HINTS.search(t) and not scan(t)]
        for i, text in enumerate(take(clean, q[key], rng)):
            add(text, cycle_sources(reader_sources, i), labels(), origin)

    for i, r in enumerate(take(read(raw, "notinject"), q["notinject"], rng)):
        add(r["text"], cycle_sources([("web_result", 2), ("document", 1), ("user_turn", 1)], i), labels(),
            "notinject:benign")

    for name, src_cycle, split in (("nvidia_ipi", tool_sources, None),
                                   ("agentdojo", [("tool_output", 2), ("web_result", 1)], "probe")):
        rows = read(raw, name)
        bad = [r for r in rows if r["label"] == "injection"]
        good = [r for r in rows if r["label"] == "benign"]
        for i, r in enumerate(take(bad, q[name], rng)):
            add(r["text"], cycle_sources(src_cycle, i), labels(injection=True), f"{name}:injected", split=split,
                span=r["span"])
        for i, r in enumerate(take(good, q[f"{name}_benign"], rng)):
            add(r["text"], cycle_sources(src_cycle, i), labels(), f"{name}:clean", split=split)

    for i, row in enumerate(read(raw, "treycsa")):
        is_inj = str(row.get("label", "")).lower() in ("1", "injection", "malicious", "true")
        add(row["text"], ("document", "unlabelled", "web_result")[i % 3], labels(injection=is_inj),
            "treycsa", split="probe")

    # Jailbreak prompts (TrustAIRLab in-the-wild), deduplicated.
    jail = [r["text"] for r in trust if r["jailbreak"]]
    jail_sources = [("user_turn", 13), ("history", 2), ("web_result", 2)]
    for i, text in enumerate(take(jail, q["jailbreak"] * 2, rng)):
        if sum(1 for r in out if r["origin"] == "trustairlab:jailbreak") >= q["jailbreak"]:
            break
        add(text, cycle_sources(jail_sources, i), labels(jailbreak=True), "trustairlab:jailbreak")

    # Harmful requests (JBB, HarmBench, XSTest unsafe contrasts): each used once.
    curated = ([("jbb:harmful", r["text"]) for r in jbb if r["label"] == "harmful"]
               + [("harmbench", r["text"]) for r in read(raw, "harmbench")]
               + [("xstest:unsafe", r["text"]) for r in xstest if str(r.get("type", "")).startswith("contrast")])
    aegis = read(raw, "aegis")
    # Aegis unsafe prompts that read like jailbreaks (some come from the DAN collection) are skipped,
    # so the harm label stays about the request itself.
    usable = [r for r in aegis if r.get("label") == "unsafe" and len(r["text"]) < 2000
              and not JAILBREAK_HINTS.search(r["text"])]
    aegis_unsafe = [("aegis:unsafe", r["text"]) for r in usable if aegis_group(r.get("categories")) == "harm"]
    harm = curated + take(aegis_unsafe, max(0, q["harm"] - len(curated)), rng)
    for i, (origin, text) in enumerate(harm):
        add(text, "user_turn" if i % 5 else "history", labels(harm=True), origin)
    aegis_safe = [r["text"] for r in aegis if r.get("label") == "safe" and len(r["text"]) < 2000
                  and not JAILBREAK_HINTS.search(r["text"]) and not scan(r["text"])]
    for text in take(aegis_safe, q["aegis_safe"], rng):
        add(text, "user_turn", labels(), "aegis:safe")
    soft = [r["text"] for r in usable if aegis_group(r.get("categories")) == "soft" and not scan(r["text"])]
    for text in take(soft, q["aegis_soft"], rng):
        add(text, "user_turn", labels(), "aegis:soft")
    privacy = [r["text"] for r in usable if aegis_group(r.get("categories")) == "privacy"]
    for i, text in enumerate(take(privacy, q["aegis_privacy"], rng)):
        add(text, "user_turn" if i % 5 else "history", labels(sensitive_data=True), "aegis:privacy")

    # Sensitive data: synthetic PII documents (Nemotron-PII) and generated fake secrets.
    pii_sources = [("document", 4), ("user_turn", 3), ("tool_output", 1), ("retrieved_doc", 2)]
    for i, row in enumerate(take([r for r in read(raw, "nemotron") if r["labels"]], q["nemotron"], rng)):
        add(row["text"], cycle_sources(pii_sources, i), labels(sensitive_data=True), "nemotron:pii")
    secret_sources = [("tool_output", 3), ("user_turn", 3), ("document", 2), ("history", 1)]
    for i in range(q["secrets"]):
        wrapper = SECRET_WRAPPERS[i % len(SECRET_WRAPPERS)]
        add(secret_text(rng, wrapper, fake_secret(rng)), cycle_sources(secret_sources, i),
            labels(sensitive_data=True), "generated:secret")
    # The same contexts with placeholders: benign, so a config or a curl line alone isn't "sensitive".
    for i in range(q["secret_placeholder"]):
        wrapper = SECRET_WRAPPERS[(i * 7) % len(SECRET_WRAPPERS)]
        placeholder = SECRET_PLACEHOLDERS[i % len(SECRET_PLACEHOLDERS)].replace("{{", "{").replace("}}", "}")
        add(secret_text(rng, wrapper, placeholder), cycle_sources(secret_sources, i), labels(),
            "generated:placeholder")

    assign_cluster_splits(out, seed)
    splits: Dict[str, List[dict]] = defaultdict(list)
    dropped = sum(1 for r in out if r.get("_drop"))
    print(json.dumps({"dropped_forced_conflicts": dropped}), file=sys.stderr)
    for row in out:
        if row.pop("_drop", False):
            continue
        split = row.pop("_split")
        row.pop("_forced", None)
        splits[split].append(row)
    for rows in splits.values():
        rng.shuffle(rows)
    ensure_probe_sources(splits)
    return splits


def ensure_probe_sources(splits: Dict[str, List[dict]]) -> None:
    """Probe is a ~2% sample, so a source can miss it by chance (templated rows form one cluster).
    Move one row of each missing source from test, never from train, so training data is unchanged.
    The moved row's cluster mates stay in test; probe is a second check, not a held-out set."""
    probe_sources = {r["source"] for r in splits.get("probe", [])}
    for source in sorted({r["source"] for rows in splits.values() for r in rows} - probe_sources):
        test = splits.get("test", [])
        idx = next((i for i, r in enumerate(test) if r["source"] == source), None)
        if idx is not None:
            splits.setdefault("probe", []).append(test.pop(idx))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=BACKEND / "data" / "security")
    ap.add_argument("--total", type=int, default=DEFAULT_TOTAL)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()
    splits = build(args.raw, args.total, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name in ("train", "test", "probe"):
        rows = splits.get(name, [])
        with (args.out / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        positives = Counter(a for r in rows for a in AXES if r["labels"][a] != "none")
        summary[name] = {"rows": len(rows), "benign": sum(1 for r in rows if all(v == "none" for v in r["labels"].values())),
                         **{f"{a}": positives[a] for a in AXES}}
    summary["by_origin"] = dict(Counter(r["origin"] for rows in splits.values() for r in rows))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
