from __future__ import annotations

import argparse
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

WORD_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)


def normalize_tokens(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text or "").casefold().replace("’", "'")
    return [m.group(0) for m in WORD_RE.finditer(text)]


@dataclass
class EditStats:
    hits: int
    substitutions: int
    deletions: int
    insertions: int


def align_counts(ref: list[str], hyp: list[str]) -> EditStats:
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

    hits = subs = dels = ins = 0
    i, j = n, m
    while i or j:
        if i and j:
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            if dp[i][j] == dp[i - 1][j - 1] + cost:
                if cost == 0:
                    hits += 1
                else:
                    subs += 1
                i -= 1
                j -= 1
                continue
        if i and dp[i][j] == dp[i - 1][j] + 1:
            dels += 1
            i -= 1
            continue
        ins += 1
        j -= 1
    return EditStats(hits, subs, dels, ins)


def text_metrics(reference: str, hypothesis: str) -> dict:
    ref = normalize_tokens(reference)
    hyp = normalize_tokens(hypothesis)
    edits = align_counts(ref, hyp)
    denom = len(ref)
    wer = (edits.substitutions + edits.deletions + edits.insertions) / denom if denom else (0.0 if not hyp else 1.0)

    ref_chars = list("".join(normalize_tokens(reference)))
    hyp_chars = list("".join(normalize_tokens(hypothesis)))
    cedit = align_counts(ref_chars, hyp_chars)
    cdenom = len(ref_chars)
    cer = (cedit.substitutions + cedit.deletions + cedit.insertions) / cdenom if cdenom else (0.0 if not hyp_chars else 1.0)

    return {
        "reference_words": len(ref),
        "hypothesis_words": len(hyp),
        "hits": edits.hits,
        "substitutions": edits.substitutions,
        "deletions": edits.deletions,
        "insertions": edits.insertions,
        "wer": wer,
        "cer": cer,
    }


def phrase_count(tokens: list[str], phrase: list[str]) -> int:
    if not phrase or len(phrase) > len(tokens):
        return 0
    return sum(1 for i in range(len(tokens) - len(phrase) + 1) if tokens[i:i + len(phrase)] == phrase)


def keyterm_metrics(reference: str, hypothesis: str, keyterms: Iterable[str]) -> dict:
    rt = normalize_tokens(reference)
    ht = normalize_tokens(hypothesis)
    details = []
    ref_total = hits = 0
    for raw in keyterms:
        pt = normalize_tokens(raw)
        if not pt:
            continue
        rc = phrase_count(rt, pt)
        hc = phrase_count(ht, pt)
        matched = min(rc, hc)
        ref_total += rc
        hits += matched
        details.append({"term": raw, "reference_count": rc, "hypothesis_count": hc, "hits": matched})
    return {
        "reference_occurrences": ref_total,
        "hits": hits,
        "recall": (hits / ref_total) if ref_total else None,
        "details": details,
    }


def read_keyterms(path: Path | None) -> list[str]:
    if not path or not path.exists():
        return []
    return [x.strip() for x in path.read_text(encoding="utf-8", errors="replace").splitlines() if x.strip() and not x.lstrip().startswith("#")]


def compare_files(reference_path: Path, hypothesis_path: Path, keyterms_path: Path | None = None) -> dict:
    reference = reference_path.read_text(encoding="utf-8", errors="replace")
    hypothesis = hypothesis_path.read_text(encoding="utf-8", errors="replace")
    terms = read_keyterms(keyterms_path)
    return {
        "reference": str(reference_path),
        "hypothesis": str(hypothesis_path),
        "text": text_metrics(reference, hypothesis),
        "keyterms": keyterm_metrics(reference, hypothesis, terms),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Compara una transcripción con una referencia humana.")
    p.add_argument("reference", type=Path)
    p.add_argument("hypothesis", type=Path)
    p.add_argument("--keyterms", type=Path, default=None)
    p.add_argument("--json", dest="json_path", type=Path, default=None)
    args = p.parse_args()
    report = compare_files(args.reference, args.hypothesis, args.keyterms)
    text = report["text"]
    print(f"WER: {text['wer']:.4f} | CER: {text['cer']:.4f}")
    print(f"Sustituciones: {text['substitutions']} | Borrados: {text['deletions']} | Inserciones: {text['insertions']}")
    if report["keyterms"]["recall"] is not None:
        print(f"Recall términos clave: {report['keyterms']['recall']:.4f}")
    if args.json_path:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Informe JSON: {args.json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
