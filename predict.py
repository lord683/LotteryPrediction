#!/usr/bin/env python3
import os
import sys
import csv
import math
from datetime import datetime, timezone
import pandas as pd
import numpy as np

# ---------- CONFIG ----------
DATA_FILE = os.path.join("data", "uk49s_history.csv")   # must exist
PRED_OUT  = "predictions.txt"
BANKROLL_LOG = os.path.join("logs", "bankroll.csv")

# Bankroll & staking
DEFAULT_START_BANKROLL = float(os.environ.get("BANKROLL_START", "1000"))  # starting bank, if no log exists
STRATEGY = os.environ.get("STAKE_STRATEGY", "fixed_pct")  # fixed_pct or simple_conf
FIXED_PCT = float(os.environ.get("FIXED_PCT", "0.02"))    # 2% default
MAX_STAKE_PCT = float(os.environ.get("MAX_STAKE_PCT", "0.05"))  # cap at 5%

# Scoring windows
FREQ_WINDOW = int(os.environ.get("FREQ_WINDOW", "120"))   # lookback draws for "hot" frequency
RECENCY_DECAY = float(os.environ.get("RECENCY_DECAY", "0.97"))  # exponential decay for recent appearances (closer = stronger)
MOMENTUM_WINDOW = int(os.environ.get("MOMENTUM_WINDOW", "30"))  # short window boost

# Mode:
#   PREDICT (default) -> makes a prediction, logs stake
#   SETTLE -> settle last bet with RETURN_AMOUNT env var
MODE = os.environ.get("MODE", "PREDICT").upper()
RETURN_AMOUNT = os.environ.get("RETURN_AMOUNT")  # used only in SETTLE

# ---------- UTIL ----------
def utc_now_str():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def ensure_dirs():
    os.makedirs("data", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

def load_history(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"History not found at {path}")
    df = pd.read_csv(path)
    # Expected columns: date,num1,num2,num3,num4,num5,num6,bonus  (bonus optional)
    # Sanitize numeric:
    num_cols = [c for c in df.columns if c != "date"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # Drop rows that don't have at least 6 valid numbers
    df = df.dropna(subset=num_cols[:6])
    for c in num_cols:
        df[c] = df[c].astype(int)
    return df

def get_current_bankroll():
    """Read last bankroll from BANKROLL_LOG; if none, start default."""
    if not os.path.exists(BANKROLL_LOG):
        return DEFAULT_START_BANKROLL, 0.0  # bankroll, total_profit
    try:
        df = pd.read_csv(BANKROLL_LOG)
        if df.empty:
            return DEFAULT_START_BANKROLL, 0.0
        last = df.iloc[-1]
        return float(last["bankroll_after"]), float(df["profit_cum"].iloc[-1])
    except Exception:
        return DEFAULT_START_BANKROLL, 0.0

def write_bankroll_entry(ts, prediction, stake, bankroll_before, bankroll_after, ret_amount, pnl, pnl_cum):
    exists = os.path.exists(BANKROLL_LOG)
    with open(BANKROLL_LOG, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["timestamp","prediction","stake","return","pnl","profit_cum","bankroll_before","bankroll_after"])
        w.writerow([ts, " ".join(map(str, prediction)), f"{stake:.2f}",
                    "" if ret_amount is None else f"{ret_amount:.2f}",
                    "" if pnl is None else f"{pnl:.2f}",
                    "" if pnl_cum is None else f"{pnl_cum:.2f}",
                    f"{bankroll_before:.2f}", f"{bankroll_after:.2f}"])

def append_prediction(ts, prediction, confidence, stake):
    with open(PRED_OUT, "a") as f:
        f.write(f"{ts} Prediction: {prediction} | Confidence: {confidence:.3f} | Stake: {stake:.2f}\n")

# ---------- SCORING ----------
def build_flat_list(df):
    """Flatten numbers per draw into list of lists (excluding bonus)."""
    nums = df[["num1","num2","num3","num4","num5","num6"]].values.tolist()
    return nums

def frequency_score(nums_list, window=120):
    """Hot numbers in recent window."""
    start = max(0, len(nums_list) - window)
    recent = nums_list[start:]
    counts = np.zeros(49, dtype=float)
    for draw in recent:
        for n in draw:
            if 1 <= n <= 49:
                counts[n-1] += 1.0
    if counts.max() > 0:
        counts = counts / counts.max()
    return counts  # 0..1

def recency_score(nums_list, decay=0.97):
    """Boost numbers that appeared recently, with exponential decay."""
    last_seen = {i: None for i in range(1,50)}
    for idx, draw in enumerate(nums_list):
        for n in draw:
            last_seen[n] = idx
    # Now convert to recency with decay (more recent -> closer to 1)
    scores = np.zeros(49, dtype=float)
    if not nums_list:
        return scores
    last_index = len(nums_list)-1
    for n in range(1,50):
        if last_seen[n] is None:
            scores[n-1] = 0.0
        else:
            gap = last_index - last_seen[n]
            scores[n-1] = decay**gap
    # normalize
    if scores.max() > 0:
        scores = scores / scores.max()
    return scores

def momentum_score(nums_list, window=30):
    """Short-window frequency (momentum)."""
    return frequency_score(nums_list, window=window)

def hybrid_rank_scores(df):
    nums_list = build_flat_list(df)
    freq = frequency_score(nums_list, window=FREQ_WINDOW)
    rec  = recency_score(nums_list, decay=RECENCY_DECAY)
    mom  = momentum_score(nums_list, window=MOMENTUM_WINDOW)
    # Weighted blend
    # tweakable weights
    w_freq, w_rec, w_mom = 0.5, 0.3, 0.2
    score = w_freq*freq + w_rec*rec + w_mom*mom
    # Normalize 0..1
    if score.max() > 0:
        score = score / score.max()
    return score

def pick_six(score_vec, df):
    """Ensure 6 unique numbers; lightly de-correlate by avoiding pairs that often appear together."""
    # Base ranking
    candidates = np.argsort(score_vec)[::-1] + 1  # numbers 1..49
    pick = []
    # Simple pair penalty using last 120 draws
    recent = build_flat_list(df)[-120:] if len(df) > 120 else build_flat_list(df)
    pair_counts = {}
    for draw in recent:
        for i in range(len(draw)):
            for j in range(i+1, len(draw)):
                a, b = sorted((draw[i], draw[j]))
                pair_counts[(a,b)] = pair_counts.get((a,b), 0) + 1

    def pair_penalty(p, q):
        a, b = sorted((p,q))
        return pair_counts.get((a,b), 0)

    for n in candidates:
        # try to avoid building very frequent historical pairs
        bad = any(pair_penalty(n, x) > 8 for x in pick)  # threshold can be tuned
        if not bad:
            pick.append(n)
        if len(pick) == 6:
            break
    # Fallback if not enough picked (very rare)
    if len(pick) < 6:
        for n in candidates:
            if n not in pick:
                pick.append(n)
                if len(pick) == 6:
                    break
    pick.sort()
    return pick

def confidence_from_scores(score_vec, chosen):
    # Average of chosen scores
    vals = [score_vec[n-1] for n in chosen]
    return float(np.mean(vals)) if vals else 0.0

# ---------- STAKING ----------
def compute_stake(bankroll, confidence):
    if STRATEGY == "fixed_pct":
        stake = bankroll * FIXED_PCT
    else:
        # simple_conf: scale between 2% and up to MAX_STAKE_PCT based on confidence (0..1)
        pct = min(FIXED_PCT + confidence * (MAX_STAKE_PCT - FIXED_PCT), MAX_STAKE_PCT)
        stake = bankroll * pct
    # Safety floor/ceil
    stake = max(5.0, min(stake, bankroll * MAX_STAKE_PCT))
    stake = min(stake, bankroll)  # never exceed bankroll
    return round(stake, 2)

# ---------- MAIN FLOWS ----------
def do_predict():
    ensure_dirs()
    df = load_history(DATA_FILE)
    bankroll_before, profit_cum = get_current_bankroll()

    scores = hybrid_rank_scores(df)
    pick = pick_six(scores, df)
    conf = confidence_from_scores(scores, pick)
    stake = compute_stake(bankroll_before, conf)
    ts = utc_now_str()

    # Log prediction & provisional bankroll (stake reserved)
    bankroll_after = bankroll_before - stake
    append_prediction(ts, pick, conf, stake)
    write_bankroll_entry(ts, pick, stake, bankroll_before, bankroll_after, ret_amount=None, pnl=None, pnl_cum=None)

    print(f"[OK] {ts}")
    print(f"Pick (6): {pick}")
    print(f"Confidence: {conf:.3f}")
    print(f"Stake: {stake:.2f} | Bankroll: {bankroll_before:.2f} -> {bankroll_after:.2f}")
    print(f"Saved to {PRED_OUT} and {BANKROLL_LOG}")

def do_settle():
    """Settle the last bet by providing RETURN_AMOUNT env var (the amount bookie paid back, 0 if lost)."""
    if RETURN_AMOUNT is None:
        print("ERROR: SETTLE mode requires env RETURN_AMOUNT, e.g. RETURN_AMOUNT=0 or RETURN_AMOUNT=2400")
        sys.exit(1)
    ensure_dirs()
    if not os.path.exists(BANKROLL_LOG):
        print("No bankroll log to settle.")
        sys.exit(1)

    df = pd.read_csv(BANKROLL_LOG)
    if df.empty:
        print("No entries to settle.")
        sys.exit(1)

    # Find last unsettled row = last row with empty return/pnl/bankroll_after already set at reserve stage
    idx = len(df) - 1
    # Read reserved stake & bankroll_before/after
    stake = float(df.loc[idx, "stake"])
    bankroll_before = float(df.loc[idx, "bankroll_before"])
    reserved_after = float(df.loc[idx, "bankroll_after"])  # at prediction time
    ret = float(RETURN_AMOUNT)

    pnl = ret - stake
    # Profit cumulative:
    prev_cum = 0.0
    if "profit_cum" in df.columns and df["profit_cum"].notna().any():
        prev_cum = float(df["profit_cum"].dropna().iloc[-1])

    pnl_cum = prev_cum + pnl
    bankroll_final = reserved_after + ret

    # Update row values
    df.loc[idx, "return"] = f"{ret:.2f}"
    df.loc[idx, "pnl"] = f"{pnl:.2f}"
    df.loc[idx, "profit_cum"] = f"{pnl_cum:.2f}"
    df.loc[idx, "bankroll_after"] = f"{bankroll_final:.2f}"

    df.to_csv(BANKROLL_LOG, index=False)

    print(f"[SETTLED] Return: {ret:.2f} | PnL: {pnl:.2f} | Profit Cum: {pnl_cum:.2f} | New Bankroll: {bankroll_final:.2f}")

if __name__ == "__main__":
    if MODE == "SETTLE":
        do_settle()
    else:
        do_predict()
