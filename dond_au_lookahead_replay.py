#!/usr/bin/env python3
"""
Deal or No Deal Australia — Utility + Loss Aversion + Chosen-Case-Preserved Rollout
=================================================================================

AU classic rules:
- 22 cases, contestant keeps chosen case (never opened during actual play)
- Open schedule: 6, 5, 4, 3, 1 (banker offer after each round)
- No swapping

Strategies:
1) utility_rollout:
   - Uses CARA utility U(x)=1-exp(-x/R)
   - Adds loss aversion vs a reference point (e.g., current offer / last offer)
   - Computes "Deal vs Continue" by estimating E[Utility(continuation)] via Monte Carlo rollouts
   - Rollouts PRESERVE chosen-case identity:
       * sample a hypothetical chosen value at start of each rollout
       * never remove it during case openings
       * banker offers computed on all remaining (including chosen)
2) ev_baseline:
   - Deal if offer >= EV(remaining) (risk-neutral baseline)

Outputs:
- <out>_raw.csv
- <out>_summary.csv
- <out>_win_distribution.png
- <out>_deal_timing.png
- <out>_best_replay.json
- <out>_best_replay_timeline.png

Install:
  pip install numpy pandas matplotlib tqdm

Run:
  python dond_au_lookahead_replay.py --trials 3000 --lookahead_sims 140 --out dond_run

Key knobs (contestant psychology):
  --risk_tolerance 30000     (R; 10k cautious, 30k moderate, 80k gambler)
  --loss_aversion 2.0        (lambda; 0 disables)
  --ref_mode current_offer   (current_offer | last_offer | max_offer)
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


# =========================
# GAME CONFIG (AU common)
# =========================

PRIZES = [
    0.5, 1, 5, 10, 25, 50, 75, 100, 250, 500, 750,
    1000, 2500, 5000, 7500, 10000, 20000, 30000, 40000, 50000, 75000, 100000
]

OPEN_SCHEDULE = [6, 5, 4, 3, 1]  # offers after each; end with 2 cases (chosen + 1)


# =========================
# BANKER MODEL
# =========================

@dataclass
class BankerParams:
    ev_mult_by_round: Tuple[float, float, float, float, float] = (0.72, 0.80, 0.88, 0.95, 1.02)
    vol_penalty: float = 0.15   # 0.0 disables
    noise_std: float = 0.03     # 0.0 disables


def _mean_std(vals: List[float]) -> Tuple[float, float]:
    m = sum(vals) / len(vals)
    v = sum((x - m) ** 2 for x in vals) / len(vals)
    return m, math.sqrt(v)


def banker_offer(remaining: List[float], round_idx: int, rng: random.Random, p: BankerParams) -> float:
    ev, std = _mean_std(remaining)
    cv = (std / ev) if ev > 0 else 0.0
    mult = p.ev_mult_by_round[round_idx]
    offer = ev * mult * (1.0 - p.vol_penalty * cv)

    if p.noise_std > 0:
        offer *= math.exp(rng.gauss(0.0, p.noise_std))

    offer = round(offer / 50.0) * 50.0
    return max(0.0, offer)


# =========================
# UTILITY + LOSS AVERSION
# =========================

def cara_utility(x: float, R: float) -> float:
    # U(x)=1-exp(-x/R). R>0
    R = max(1.0, float(R))
    return 1.0 - math.exp(-max(0.0, float(x)) / R)


def utility_with_loss_aversion(x: float, R: float, ref: float, loss_aversion: float) -> float:
    """
    Base: CARA utility.
    Loss aversion: outcomes below reference are penalized more heavily.

    We implement this in utility-space:
      u = U(x)
      if x < ref:
          u -= lambda * (U(ref) - U(x))   # extra penalty for falling below ref
    where lambda >= 0.
    """
    u = cara_utility(x, R)
    if loss_aversion > 0 and x < ref:
        u_ref = cara_utility(ref, R)
        u -= float(loss_aversion) * (u_ref - u)
    return u


def choose_reference(ref_mode: str, current_offer: float, last_offer: float, max_offer_so_far: float) -> float:
    if ref_mode == "current_offer":
        return current_offer
    if ref_mode == "last_offer":
        return last_offer
    if ref_mode == "max_offer":
        return max_offer_so_far
    raise ValueError(f"Unknown ref_mode: {ref_mode}")


def expected_utility_over_distribution(values: List[float], R: float, ref: float, loss_aversion: float) -> float:
    # From contestant POV (not knowing their case), case is uniform over remaining values
    return sum(utility_with_loss_aversion(v, R, ref, loss_aversion) for v in values) / len(values)


# =========================
# FAST POP HELPERS
# =========================

def _swap_pop(vals: List[float], idx: int) -> float:
    vals[idx], vals[-1] = vals[-1], vals[idx]
    return vals.pop()


def _pop_random(vals: List[float], rng: random.Random) -> float:
    return _swap_pop(vals, rng.randrange(len(vals)))


def _pop_random_nonchosen(remaining: List[float], rng: random.Random) -> float:
    # chosen is at index 0; pop among indices 1..end-1
    if len(remaining) <= 2:
        raise RuntimeError("No non-chosen to pop (<=2 remaining).")
    j = rng.randrange(1, len(remaining))
    return _swap_pop(remaining, j)


# =========================
# ROLLOUT (PRESERVE CHOSEN IDENTITY) + UTILITY DECISIONS
# =========================

def rollout_continuation_expected_utility(
    remaining_now_including_chosen: List[float],
    round_idx: int,
    rng: random.Random,
    banker: BankerParams,
    rollout_sims: int,
    R: float,
    loss_aversion: float,
    ref_mode: str,
    last_offer: float,
    max_offer_so_far: float,
) -> float:
    """
    Estimate E[Utility(continuation)] if contestant says NO DEAL now (at round_idx).

    Key realism fixes:
    - Chosen case identity is preserved inside each rollout:
        * Sample chosen_hat from remaining values (uniform)
        * Keep chosen_hat fixed, never opened
        * Open only from the "other cases" pool
        * Banker offers computed on full remaining = [chosen_hat] + others_remaining
    - Utility decision in future rollout steps uses:
        take deal if Utility(offer) >= ExpectedUtility(case distribution of remaining)
      with loss aversion applied vs the chosen reference point (ref_mode).

    Returns expected utility (not dollars).
    """
    sims = max(30, int(rollout_sims))
    total_u = 0.0

    for _ in range(sims):
        # Sample a hypothetical chosen value and preserve it
        all_remaining = remaining_now_including_chosen[:]  # includes unknown chosen among them
        chosen_hat = all_remaining[rng.randrange(len(all_remaining))]

        # Other cases = all remaining minus one instance of chosen_hat
        others = all_remaining[:]
        # remove one instance of chosen_hat (swap-pop via index)
        idx = others.index(chosen_hat)
        _swap_pop(others, idx)

        # rollout state for reference tracking
        last = float(last_offer)
        max_offer = float(max_offer_so_far)

        dealt = False
        for r in range(round_idx, len(OPEN_SCHEDULE)):
            # open cases (from others only)
            to_open = OPEN_SCHEDULE[r]
            for _k in range(to_open):
                if len(others) <= 1:
                    break
                _pop_random(others, rng)

            remaining_full = [chosen_hat] + others
            offer = banker_offer(remaining_full, r, rng, banker)

            max_offer = max(max_offer, offer)
            ref = choose_reference(ref_mode, current_offer=offer, last_offer=last, max_offer_so_far=max_offer)

            u_offer = utility_with_loss_aversion(offer, R, ref, loss_aversion)
            eu_case = expected_utility_over_distribution(remaining_full, R, ref, loss_aversion)

            # Myopic but psychologically realistic: "Is this offer better than the utility of taking my chances
            # on the remaining distribution (given I don't know my case)?"
            if u_offer >= eu_case:
                total_u += u_offer
                dealt = True
                last = offer
                break

            last = offer  # they saw the offer, even if they said no

        if not dealt:
            # End: no deal. Payout is chosen_hat.
            # Reference point: use latest offer info (last/max). No "current offer" exists now.
            # We'll treat ref as last offer for end-utility, because psychologically that's what was on the table.
            ref_end = choose_reference(
                "last_offer" if ref_mode == "current_offer" else ref_mode,
                current_offer=last,
                last_offer=last,
                max_offer_so_far=max_offer,
            )
            total_u += utility_with_loss_aversion(chosen_hat, R, ref_end, loss_aversion)

    return total_u / sims


# =========================
# GAME SIMULATION WITH FULL EVENT LOGGING
# =========================

@dataclass
class GameLog:
    strategy: str
    seed: int
    chosen_value: float
    opened_by_round: List[List[float]]
    remaining_by_round: List[List[float]]
    ev_by_round: List[float]
    offer_by_round: List[float]
    ref_by_round: List[float]
    deal_decision_by_round: List[bool]
    deal_round: Optional[int]   # 1..5 or None
    winnings: float


def play_game_with_log(
    strategy: str,
    rng: random.Random,
    banker: BankerParams,
    lookahead_sims: int,
    game_seed: int,
    R: float,
    loss_aversion: float,
    ref_mode: str,
) -> GameLog:
    # Assign values to 22 cases; contestant keeps index 0 as their case
    cases = PRIZES[:]
    rng.shuffle(cases)
    chosen_value = cases[0]

    # Remaining includes chosen at index 0; open only indices 1..end in real play
    remaining = cases[:]

    opened_by_round: List[List[float]] = []
    remaining_by_round: List[List[float]] = []
    ev_by_round: List[float] = []
    offer_by_round: List[float] = []
    ref_by_round: List[float] = []
    deal_decision_by_round: List[bool] = []

    deal_round: Optional[int] = None
    winnings: Optional[float] = None

    last_offer = 0.0
    max_offer = 0.0

    for r, open_n in enumerate(OPEN_SCHEDULE):
        opened_this_round: List[float] = []
        for _ in range(open_n):
            if len(remaining) <= 2:
                break
            opened_this_round.append(_pop_random_nonchosen(remaining, rng))

        opened_by_round.append(sorted(opened_this_round))
        remaining_by_round.append(sorted(remaining))

        ev = sum(remaining) / len(remaining)
        ev_by_round.append(ev)

        offer = banker_offer(remaining, r, rng, banker)
        offer_by_round.append(offer)

        max_offer = max(max_offer, offer)
        ref = choose_reference(ref_mode, current_offer=offer, last_offer=last_offer, max_offer_so_far=max_offer)
        ref_by_round.append(ref)

        if strategy == "ev_baseline":
            deal = offer >= ev

        elif strategy == "utility_rollout":
            # Compare utility(offer) vs expected utility of continuing (rollouts)
            u_offer = utility_with_loss_aversion(offer, R, ref, loss_aversion)
            eu_cont = rollout_continuation_expected_utility(
                remaining_now_including_chosen=remaining,
                round_idx=r,
                rng=rng,
                banker=banker,
                rollout_sims=lookahead_sims,
                R=R,
                loss_aversion=loss_aversion,
                ref_mode=ref_mode,
                last_offer=last_offer,
                max_offer_so_far=max_offer,
            )
            deal = u_offer >= eu_cont

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        deal_decision_by_round.append(deal)

        if deal:
            deal_round = r + 1
            winnings = offer
            last_offer = offer
            break

        last_offer = offer

    if winnings is None:
        winnings = chosen_value  # no deal, reveal chosen

    return GameLog(
        strategy=strategy,
        seed=game_seed,
        chosen_value=chosen_value,
        opened_by_round=opened_by_round,
        remaining_by_round=remaining_by_round,
        ev_by_round=ev_by_round,
        offer_by_round=offer_by_round,
        ref_by_round=ref_by_round,
        deal_decision_by_round=deal_decision_by_round,
        deal_round=deal_round,
        winnings=winnings,
    )


# =========================
# RUN MANY + PICK BEST + OUTPUTS (WITH PROGRESS)
# =========================

def run_trials(
    trials: int,
    strategy: str,
    banker: BankerParams,
    base_seed: int,
    lookahead_sims: int,
    R: float,
    loss_aversion: float,
    ref_mode: str,
    show_progress: bool = True,
) -> Tuple[pd.DataFrame, GameLog]:
    rows = []
    best_log: Optional[GameLog] = None

    start_time = time.time()
    it = range(trials)

    if show_progress:
        it = tqdm(it, total=trials, desc=strategy, unit="game", dynamic_ncols=True)

    for t in it:
        game_seed = base_seed + t * 1009 + (0 if strategy == "utility_rollout" else 17)
        rng = random.Random(game_seed)
        glog = play_game_with_log(strategy, rng, banker, lookahead_sims, game_seed, R, loss_aversion, ref_mode)

        rows.append({
            "strategy": strategy,
            "seed": glog.seed,
            "chosen_value": glog.chosen_value,
            "winnings": glog.winnings,
            "deal_round": glog.deal_round if glog.deal_round is not None else 6,  # 6 = no deal
        })

        if best_log is None or glog.winnings > best_log.winnings:
            best_log = glog

        if show_progress and t > 0 and (t % 25 == 0):
            elapsed = time.time() - start_time
            rate = t / elapsed if elapsed > 0 else 0.0
            eta = (trials - t) / rate if rate > 0 else 0.0
            try:
                it.set_postfix(rate=f"{rate:.2f}/s", eta=f"{eta/60:.1f}m")
            except Exception:
                pass

    df = pd.DataFrame(rows)
    assert best_log is not None
    return df, best_log


# =========================
# PLOTTING
# =========================

def save_distribution_chart(df: pd.DataFrame, out_prefix: str) -> None:
    plt.figure(figsize=(10, 5))
    for strat in df["strategy"].unique():
        subset = df[df["strategy"] == strat]["winnings"].values
        plt.hist(subset, bins=60, alpha=0.55, label=strat)
    plt.xlabel("Winnings ($)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Winnings")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_win_distribution.png", dpi=160)
    plt.close()


def save_deal_timing_chart(df: pd.DataFrame, out_prefix: str) -> None:
    timing = pd.crosstab(df["strategy"], df["deal_round"], normalize="index").sort_index(axis=1)
    ax = timing.plot(kind="bar", stacked=True, figsize=(10, 4))
    ax.set_title("Deal Timing Distribution (6 = No Deal / Final Reveal)")
    ax.set_xlabel("Strategy")
    ax.set_ylabel("Probability")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_deal_timing.png", dpi=160)
    plt.close()


def save_replay_timeline(best: GameLog, out_prefix: str) -> None:
    rounds = list(range(1, len(best.offer_by_round) + 1))
    evs = best.ev_by_round[: len(rounds)]
    offers = best.offer_by_round[: len(rounds)]
    deals = best.deal_decision_by_round[: len(rounds)]

    plt.figure(figsize=(10, 5))
    plt.plot(rounds, evs, marker="o", label="EV (remaining)")
    plt.plot(rounds, offers, marker="o", label="Banker offer")

    for r, off, d in zip(rounds, offers, deals):
        plt.text(r, off, "DEAL" if d else "NO", fontsize=8, ha="center", va="bottom")

    plt.xticks(rounds)
    plt.xlabel("Round (AU schedule: 6,5,4,3,1 opens)")
    plt.ylabel("Dollars ($)")
    plt.title(f"Best Game Replay Timeline — {best.strategy} — winnings ${best.winnings:,.2f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_best_replay_timeline.png", dpi=160)
    plt.close()


def print_replay(best: GameLog) -> None:
    print("\n" + "=" * 80)
    print(f"BEST GAME REPLAY — strategy={best.strategy} seed={best.seed}")
    print(f"Chosen case value (hidden during play): ${best.chosen_value:,.2f}")
    print("=" * 80)

    for i in range(len(best.offer_by_round)):
        opened = best.opened_by_round[i]
        remaining = best.remaining_by_round[i]
        ev = best.ev_by_round[i]
        offer = best.offer_by_round[i]
        ref = best.ref_by_round[i]
        decision = "DEAL ✅" if best.deal_decision_by_round[i] else "NO DEAL"
        print(f"\nRound {i+1} (opened {OPEN_SCHEDULE[i]}):")
        print(f"  Opened: {', '.join(f'${x:,.2f}' for x in opened)}")
        print(f"  Remaining count: {len(remaining)} | EV: ${ev:,.2f}")
        print(f"  Banker offer: ${offer:,.2f} | Reference: ${ref:,.2f} -> {decision}")
        if best.deal_decision_by_round[i]:
            break

    if best.deal_round is None:
        print(f"\nFinal: NO DEAL taken. Revealed chosen case = ${best.chosen_value:,.2f}")
    else:
        print(f"\nResult: DEAL in round {best.deal_round} for ${best.winnings:,.2f}")


# =========================
# MAIN
# =========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=2000)
    ap.add_argument("--out", type=str, default="results")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--lookahead_sims", type=int, default=140, help="Rollout sims per decision (more=better, slower).")
    ap.add_argument("--no_noise", action="store_true", help="Disable banker noise for determinism.")
    ap.add_argument("--no_progress", action="store_true", help="Disable progress bars.")

    # Contestant psychology
    ap.add_argument("--risk_tolerance", type=float, default=30000.0, help="CARA R: 10k cautious, 30k moderate, 80k gambler.")
    ap.add_argument("--loss_aversion", type=float, default=2.0, help="lambda >=0; 0 disables loss aversion penalty below reference.")
    ap.add_argument("--ref_mode", choices=["current_offer", "last_offer", "max_offer"], default="current_offer")

    args = ap.parse_args()

    banker = BankerParams()
    if args.no_noise:
        banker.noise_std = 0.0

    show_progress = not args.no_progress

    # Run strategies
    df_u, best_u = run_trials(
        args.trials, "utility_rollout", banker, args.seed, args.lookahead_sims,
        args.risk_tolerance, args.loss_aversion, args.ref_mode, show_progress
    )
    df_ev, best_ev = run_trials(
        args.trials, "ev_baseline", banker, args.seed, args.lookahead_sims,
        args.risk_tolerance, args.loss_aversion, args.ref_mode, show_progress
    )

    df = pd.concat([df_u, df_ev], ignore_index=True)

    # Save raw + summary
    df.to_csv(f"{args.out}_raw.csv", index=False)
    summary = df.groupby("strategy")["winnings"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
    summary.to_csv(f"{args.out}_summary.csv")

    # Charts
    save_distribution_chart(df, args.out)
    save_deal_timing_chart(df, args.out)

    # Best game across both strategies (by winnings)
    best = best_u if best_u.winnings >= best_ev.winnings else best_ev

    # Save replay JSON
    replay_dict: Dict[str, object] = {
        "strategy": best.strategy,
        "seed": best.seed,
        "chosen_value": best.chosen_value,
        "opened_by_round": best.opened_by_round,
        "remaining_by_round": best.remaining_by_round,
        "ev_by_round": best.ev_by_round,
        "offer_by_round": best.offer_by_round,
        "ref_by_round": best.ref_by_round,
        "deal_decision_by_round": best.deal_decision_by_round,
        "deal_round": best.deal_round,
        "winnings": best.winnings,
        "open_schedule": OPEN_SCHEDULE,
        "prizes": PRIZES,
        "banker_params": {
            "ev_mult_by_round": banker.ev_mult_by_round,
            "vol_penalty": banker.vol_penalty,
            "noise_std": banker.noise_std,
        },
        "contestant_model": {
            "risk_tolerance_R": args.risk_tolerance,
            "loss_aversion_lambda": args.loss_aversion,
            "ref_mode": args.ref_mode,
            "utility": "CARA U(x)=1-exp(-x/R) with loss aversion penalty below reference",
        },
        "lookahead_sims": args.lookahead_sims,
        "notes": "utility_rollout uses chosen-case-preserved rollouts; ev_baseline is risk-neutral dollars.",
    }
    with open(f"{args.out}_best_replay.json", "w", encoding="utf-8") as f:
        json.dump(replay_dict, f, indent=2)

    # Replay chart + print transcript
    save_replay_timeline(best, args.out)
    print(summary)
    print_replay(best)

    print("\nSaved:")
    print(f"  {args.out}_raw.csv")
    print(f"  {args.out}_summary.csv")
    print(f"  {args.out}_win_distribution.png")
    print(f"  {args.out}_deal_timing.png")
    print(f"  {args.out}_best_replay.json")
    print(f"  {args.out}_best_replay_timeline.png")


if __name__ == "__main__":
    main()
