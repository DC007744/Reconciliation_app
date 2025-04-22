# Transaction Reconciliation App

## Overview
A Flask app that ingests bank and ledger CSVs, fuzzy‐matches transactions (±₹1, ±3 days, description similarity), enforces 1:1 pairing via a graph‐matching algorithm, and surfaces matched/unmatched results in a tabbed UI with an AI‑powered chat helper.

## Features
- Flexible column name detection (fuzzy match “date”, “amount”, “description”)
- Composite scoring (amount 45%, date 25%, description 30%) via RapidFuzz
- Global 1:1 matching using NetworkX’s max‐weight matching (Hungarian)
- Interactive, user-friendly interface (ouput formatted into multiple tab views)
- Floating, modern chat widget backed by a LangChain (OpenAI implementation)

## Installation

```bash
git clone https://github.com/DC007744/Reconciliation_app.git
cd Reconciliation_app
pip install -r requirements.txt
python app.py
```

## Matching Logic

- Load & Clean: normalise date → midnight, amount → float, description → lowercase alphanumeric.
- Candidate Generation: cross‑join, then filter ±₹1 and ±3 days.
- Scoring:
      amount score = max(0, 1 – Δamt/₹1) × 100
      date score = max(0, 1 – Δdays/3) × 100
      desc score = RapidFuzz’s token_set_ratio
      composite = 0.45⋅amt + 0.25⋅date + 0.30⋅desc
- Max‑Weight Matching: treat bank & ledger rows as bipartite graph nodes, edge = score, pick the one‑to‑one set maximizing total score, then threshold ≥ 70.
- Render & Export: Flask writes three CSVs and populates the UI tables.
