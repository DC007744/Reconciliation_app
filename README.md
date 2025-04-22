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
git clone https://github.com/yourusername/reconciliation-app.git
cd reconciliation-app
pip install -r requirements.txt
