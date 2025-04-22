# This is a simple reconciliation app using Flask
# It takes two CSV files as input: bank transactions and ledger entries
# It matches the transactions based on amount, date, and description
# The app uses RapidFuzz for fuzzy matching and NetworkX for graph-based matching

# -----------------------------------------------------------
# Section-1: Importing the required libraries
# -----------------------------------------------------------
import warnings
from flask import Flask, jsonify, render_template, request, send_file, redirect, url_for, flash, session
import pandas as pd
import numpy as np
from rapidfuzz import fuzz
import networkx as nx
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# -----------------------------------------------------------
# Section-2: Setting up Flask app & initializing parameters
# -----------------------------------------------------------

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "random_secret_123412341234112341234"  # random secret key
app.config['JSON_SORT_KEYS'] = False

# Initialize OpenAI chat model with LangChain
llm = ChatOpenAI(
    model="gpt-4.1-2025-04-14",
    api_key="sk-...",  # enter api key here
)

# Chatbot memory & chain (message history)
memory = ConversationBufferMemory(
    memory_key="history",
    return_messages=True
)

chatbot = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=False
)

# Reconciliation parameters & thresholds
AMOUNT_TOLERANCE  = 1.00   # ±₹1
DATE_TOLERANCE   = 3      # ±3 days
WEIGHTS          = {"amount": 0.45, "date": 0.25, "desc": 0.30}
MATCH_THRESHOLD  = 70     # minimum score

# -----------------------------------------------------------
# Section-3: Defining helper functions
# -----------------------------------------------------------
def load_csv_from_file(file_input) -> pd.DataFrame:
    """
    Description:
        Loads a CSV into a Pandas DataFrame, flexibly detects Date/Amount/Description
        columns via fuzzy‐matching, then cleans those fields for reconciliation.

    Args:
        file_input (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame with normalized columns.
    """
    
    # Read CSV file
    df = pd.read_csv(file_input)

    # normalize column names
    orig_cols = list(df.columns)
    normalized = [c.lower().strip() for c in orig_cols]
    df.columns = normalized

    # helper to pick the best‐matching column
    def pick(col_name):
        best_col, best_score = None, 0
        COL_MATCH_THRESHOLD = 75
        for c in df.columns:
            score = fuzz.token_set_ratio(c, col_name)
            if score > best_score:
                best_col, best_score = c, score
        return best_col if best_score >= COL_MATCH_THRESHOLD else None

    date_col = pick("date")
    amt_col  = pick("amount")
    desc_col = pick("description")

    # parse date
    if date_col:
        df["date"] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    # numeric amount
    if amt_col:
        df["amount"] = pd.to_numeric(df[amt_col], errors="coerce")
    # clean description
    if desc_col:
        df["description_clean"] = (
            df[desc_col].astype(str)
                .str.lower()
                .str.replace(r"[^a-z0-9 ]", "", regex=True)
                .str.strip()
        )
    return df


def _score_row(r) -> float:
    """
    Description:
        Computes a score for a row based on the amount, date, and description.

    Args:
        r: Row of the DataFrame.

    Returns:
        float: Score for the row.
    """
    amt_score = max(0, 1 - abs(r.amount_bank - r.amount_ledger) / AMOUNT_TOLERANCE) * 100
    date_delta = abs((r.date_bank - r.date_ledger).days)
    date_score = max(0, 1 - date_delta / DATE_TOLERANCE) * 100
    desc_score = fuzz.token_set_ratio(r.description_clean_bank, r.description_clean_ledger)
    total = WEIGHTS["amount"]*amt_score + WEIGHTS["date"]*date_score + WEIGHTS["desc"]*desc_score
    return round(total,2)


def find_potential_matches(bank, ledger):
    """
    Description:
        Compares each bank and ledger entry based on amount, date, and description
        and returns a DataFrame of potential matches with scores.

    Args:
        bank (pd.DataFrame): Bank transactions DataFrame.
        ledger (pd.DataFrame): Ledger entries DataFrame.

    Returns:
        pd.DataFrame: DataFrame of candidate matches with scores.
    """
    b = bank.reset_index().rename(columns={"index":"bank_idx"})
    l = ledger.reset_index().rename(columns={"index":"ledger_idx"})
    b["amt_low"], b["amt_high"] = b.amount - AMOUNT_TOLERANCE, b.amount + AMOUNT_TOLERANCE

    # cross-join
    cand = b.merge(l, how="cross", suffixes=("_bank","_ledger"))
    cand = cand.query("amt_low <= amount_ledger <= amt_high")
    cand = cand[ (cand.date_bank.notna()) & (cand.date_ledger.notna()) 
        & (abs((cand.date_bank-cand.date_ledger).dt.days) <= DATE_TOLERANCE) ]
    cand["score"] = cand.apply(_score_row, axis=1)
    return cand


def choose_best_matches(cand):
    """
    Description:
        Chooses the best matches from the candidates using a graph-based approach.
        After the find_potential_matches is called, it creates a bipartite graph
        where one set of nodes is bank transactions and the other is ledger entries.
        The edges are weighted by the score of the potential match. It then finds
        the maximum weight matching in the graph and filters out matches below
        the MATCH_THRESHOLD. The function returns a DataFrame of the best matches
        with their scores.
    
    Args:
        cand (pd.DataFrame): DataFrame of candidate matches with scores.
    
    Returns:
        pd.DataFrame: DataFrame of best matches with scores.
    """
    G = nx.Graph()
    for _,r in cand.iterrows():
        b, l = f"b{r.bank_idx}", f"l{r.ledger_idx}"
        G.add_edge(b,l, weight=r.score)
    matches = nx.max_weight_matching(G)
    pairs=[]
    for u,v in matches:
        if u.startswith('l'): u,v=v,u
        w=G.edges[u,v]['weight']
        if w>=MATCH_THRESHOLD:
            pairs.append({'bank_idx':int(u[1:]),'ledger_idx':int(v[1:]),'score':w})
    return pd.DataFrame(pairs)

# -----------------------------------------------------------
# Section-4: Defining Flask routes
# -----------------------------------------------------------

#NOTE: This is the main Flask route (for the main page) where users upload their files and get the reconciliation results.
@app.route('/', methods=['GET','POST'])
def upload_and_reconcile():
    if request.method == 'POST':
        # Ensure both files are uploaded
        if 'bank_file' not in request.files or 'ledger_file' not in request.files:
            flash('Please upload both files')
            return redirect(request.url)

        # Load and clean each CSV
        bank_f   = request.files['bank_file']
        ledger_f = request.files['ledger_file']
        bank     = load_csv_from_file(bank_f)
        ledger   = load_csv_from_file(ledger_f)

        # Find candidates and pick best matches
        cand = find_potential_matches(bank, ledger)
        best = choose_best_matches(cand)

        # Build matched/unmatched DataFrames
        matched = (best
                   .merge(bank,  left_on='bank_idx',   right_index=True)
                   .merge(ledger, left_on='ledger_idx', right_index=True))
        
        # Drop internal matching indices before rendering
        matched = matched.drop(columns=['bank_idx', 'ledger_idx','description_clean_x', 'description_clean_y'], errors='ignore')

        unmatched_bank   = bank.loc[~bank.index.isin(best.bank_idx)].copy()
        unmatched_ledger = ledger.loc[~ledger.index.isin(best.ledger_idx)].copy()

        # Drop internal matching indices before rendering
        unmatched_bank = unmatched_bank.drop(columns=['description_clean'], errors='ignore')
        unmatched_ledger = unmatched_ledger.drop(columns=['description_clean'], errors='ignore')

        unmatched_bank['reason']   = 'No match ≥ threshold'
        unmatched_ledger['reason'] = 'No match ≥ threshold'

        # store in session for chat context
        session['recon'] = {
            'matched': matched.to_dict('records'),
            'unmatched_bank': unmatched_bank.to_dict('records'),
            'unmatched_ledger': unmatched_ledger.to_dict('records')
        }

        # 5) Rename columns for human‑friendly headers
        matched = matched.rename(columns={
            'score':                 'Match Score',
            'transaction id_x':      'Bank Transaction ID',
            'date_x':                'Bank Date',
            'amount_x':              'Bank Amount',
            'description_x':         'Bank Description',
            'transaction id_y':      'Ledger Transaction ID',
            'date_y':                'Ledger Date',
            'amount_y':              'Ledger Amount',
            'description_y':         'Ledger Description',
        })
        
        unmatched_bank = unmatched_bank.rename(columns={
            'transaction id':        'Bank Transaction ID',
            'date':                  'Bank Date',
            'amount':                'Bank Amount',
            'description':           'Bank Description',
            'reason':                'Reason Unmatched',
        })

        unmatched_ledger = unmatched_ledger.rename(columns={
            'transaction id':        'Ledger Transaction ID',
            'date':                  'Ledger Date',
            'amount':                'Ledger Amount',
            'description':           'Ledger Description',
            'reason':                'Reason Unmatched',
        })

        # 6) Render results
        return render_template(
            'reconcile.html',
            matched_records    = matched.to_dict('records'),
            unmatched_bank     = unmatched_bank.to_dict('records'),
            unmatched_ledger   = unmatched_ledger.to_dict('records'),
            show_results       = True
        )

    # GET: initial upload form
    return render_template('reconcile.html', show_results=False)

#NOTE: This route is for handling the chatbot interaction.
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json(force=True, silent=True) or {}
    user_msg = data.get('message','').strip()
    if not user_msg:
        return jsonify({'error':'No message'}),400

    # on first chat turn, prime with reconciliation tables
    history = memory.load_memory_variables({}).get('history', [])
    if not history:
        recon = session.get('recon', {})
        ctx = "Reconciliation Data:\n"
        ctx += "Matched Transactions:\n" + str(recon.get('matched',[])) + "\n"
        ctx += "Unmatched Bank:\n"     + str(recon.get('unmatched_bank',[])) + "\n"
        ctx += "Unmatched Ledger:\n"   + str(recon.get('unmatched_ledger',[]))
        chatbot.predict(input=ctx)

    # suppress langchain warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        reply = chatbot.predict(input=user_msg)

    # return both reply and full history
    hist = memory.load_memory_variables({}).get('history', [])
    hist_list = [m.content for m in hist] if isinstance(hist,list) else hist.split("\n")
    return jsonify({'reply':reply,'history':hist_list})

# ------------------------------------------------------------
# Section-5: Running the Flask app
# ------------------------------------------------------------
if __name__=='__main__':
    app.run(debug=True)