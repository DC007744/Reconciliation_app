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
![app_dfd](https://github.com/user-attachments/assets/16b457d6-3b6d-49b7-92c1-ffad80ab02b5)


1. **Load & Clean CSVs**  
   - Read the bank & ledger files  
   - Normalize dates (to midnight), amounts (to floats), and descriptions (lowercase + strip symbols)

2. **Generate & Score Pairs**  
   - Pair every bank row with every ledger row within ±₹1 and ±3 days  
   - Compute three sub‑scores (0–100):  
     - **Amount**: how close the amounts are  
     - **Date**: how close the dates are  
     - **Description**: fuzzy text similarity  
   - Combine into one score:  
     ```
     total = 0.45*amount + 0.25*date + 0.30*description
     ```

3. **Pick the Best 1:1 Matches**  
   - Build a graph where nodes are rows and edge weights are scores  
   - Use a max‑weight matching algorithm to choose the highest‑scoring, one‑to‑one pairs  
   - Discard any matches below the threshold (default 70)

4. **Format & Present Results**  
   - Merge matched pairs back into full records  
   - List unmatched rows with a “No match” reason  
   - Display as three CSVs or in a tabbed web UI  

