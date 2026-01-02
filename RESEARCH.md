When Is a Deal a Good Deal?
A Quantitative Framework for Optimal Stopping Under Risk and Regret in Deal or No Deal
Abstract

Deal or No Deal presents contestants with a repeated optimal stopping problem under extreme uncertainty, emotional pressure, and asymmetric information. Classical expected-value (EV) reasoning predicts that rational contestants should rarely accept banker offers, yet empirical behaviour shows frequent early deals. This paper reconciles this discrepancy by introducing a Deal Fairness Index (DFI), a composite metric that integrates expected value, risk-averse utility, and downside regret. Using Monte Carlo simulation with fixed-identity case rollouts, we demonstrate that banker offers become objectively favourable only within narrow conditions late in the game. We show that risk-neutral strategies maximise expected payout but exhibit extreme variance, while utility-based strategies align closely with observed human behaviour. The DFI provides a principled, model-agnostic criterion for when accepting a deal is objectively correct.

1. Introduction

Deal or No Deal is a canonical example of a real-world decision problem where optimal play under expected value diverges sharply from human behaviour. Contestants face a sequence of offers from a banker while eliminating unknown prize values, with the option to stop at any time.

From a purely mathematical standpoint, the game is fair: the expected value of continuing equals the expected value of the remaining prizes. However, banker offers are systematically biased below expected value, especially in early rounds. Despite this, contestants frequently accept early deals, a phenomenon often attributed to fear, regret, or irrationality.

This paper argues that such behaviour is not irrational once risk aversion and loss aversion are accounted for. We formalise this intuition by constructing a quantitative framework that defines when a deal is objectively good, independent of hindsight.

2. Game Structure and Assumptions

We model the Australian format of Deal or No Deal:

22 sealed cases with fixed prize values ranging from $0.50 to $100,000.

The contestant selects one case at the start and retains it throughout.

Cases are opened according to the schedule: 6, 5, 4, 3, 1.

After each round, the banker offers a cash amount in exchange for stopping the game.

If no deal is taken, the contestant receives the value of their chosen case.

The banker does not know the contestant’s chosen case but observes the remaining prize set. Offers are modelled as a function of expected value, volatility, and stochastic noise.

3. Baseline Strategy: Expected Value Optimality

The classical stopping rule is:

Take Deal if Offer
≥
𝐸
[
Remaining Prizes
]
Take Deal if Offer≥E[Remaining Prizes]

We refer to this as the EV Baseline Strategy.

Properties:

Maximises long-run expected winnings.

Exhibits extreme variance.

Produces frequent very low outcomes.

Rarely accepts banker offers before the final rounds.

Monte Carlo simulations (3,000 games) show that while this strategy achieves the highest mean payout, it is psychologically implausible for real contestants.

4. Utility-Based Decision Model
4.1 Risk Aversion (CARA Utility)

We introduce a Constant Absolute Risk Aversion (CARA) utility function:

𝑈
(
𝑥
)
=
1
−
𝑒
−
𝑥
/
𝑅
U(x)=1−e
−x/R

where 
𝑅
R represents risk tolerance:

𝑅
≈
$
10,000
R≈$10,000: cautious

𝑅
≈
$
30,000
R≈$30,000: moderate

𝑅
≈
$
80,000
R≈$80,000: risk-seeking

4.2 Loss Aversion and Reference Dependence

To capture regret, we incorporate a reference point 
𝑟
r and loss aversion coefficient 
𝜆
λ:

𝑣
(
𝑥
)
=
{
𝑥
−
𝑟
	
if 
𝑥
≥
𝑟


−
𝜆
(
𝑟
−
𝑥
)
	
if 
𝑥
<
𝑟
v(x)={
x−r
−λ(r−x)
	​

if x≥r
if x<r
	​


Utility is then applied to this adjusted value. We show that using the previous banker offer as the reference point produces the most realistic behaviour.

5. Fixed-Identity Monte Carlo Lookahead

A key methodological contribution is preserving the identity of the contestant’s chosen case during lookahead simulation. In each rollout:

A hypothetical chosen case value is sampled once.

Future case openings are simulated only from the non-chosen pool.

Decisions are evaluated using the same utility rule.

This avoids optimism bias present in naive rollouts and produces accurate continuation valuations.

6. The Deal Fairness Index (DFI)

We define the Deal Fairness Index (DFI) as a composite measure:

DFI
=
0.4
𝐹
𝐸
𝑉
+
0.4
𝐹
𝑈
+
0.2
𝐹
𝑅
𝑖
𝑠
𝑘
DFI=0.4F
EV
	​

+0.4F
U
	​

+0.2F
Risk
	​


Where:

6.1 EV Fairness
𝐹
𝐸
𝑉
=
Offer
−
EV
EV
F
EV
	​

=
EV
Offer−EV
	​

6.2 Utility Fairness
𝐹
𝑈
=
𝑈
(
Offer
)
−
𝐸
[
𝑈
(
Continuation
)
]
F
U
	​

=U(Offer)−E[U(Continuation)]
6.3 Downside Risk
𝐹
𝑅
𝑖
𝑠
𝑘
=
𝑃
(
Continuation
<
Offer
)
−
0.5
F
Risk
	​

=P(Continuation<Offer)−0.5

Each component is clamped to 
[
−
1
,
+
1
]
[−1,+1].

7. Interpretation of DFI
DFI Range	Classification	Interpretation
< –0.3	Bad Deal	Banker underpaying; upside remains
–0.3 to +0.3	Borderline	Trade-off zone
> +0.3	Good Deal	Fair, utility-positive
> +0.6	Snap Deal	Continuation irrational
8. Monetary Interpretation (Mid-Game Example)

Assuming EV ≈ $30,000:

Banker Offer	% of EV	DFI Zone	Recommendation
$10k–$15k	35–50%	Bad	No Deal
$20k–$23k	65–75%	Borderline	Depends
$24k–$27k	80–90%	Fair	Lean Deal
$28k–$32k	95–105%	Good	Deal
$33k+	110%+	Snap	Immediate Deal
9. Results

Simulation results show:

EV baseline achieves the highest mean payout but extreme downside risk.

Utility-based strategies produce tighter outcome distributions and earlier deals.

DFI > 0.3 strongly predicts deals that outperform continuation in both money and utility.

Many deals criticised by audiences are, in fact, objectively correct under DFI.

10. Discussion

The apparent conflict between “mathematical correctness” and “human intuition” in Deal or No Deal arises from conflating expected value with decision optimality. By explicitly modelling utility and regret, we show that early deals are often rational responses to asymmetric risk.

The DFI formalises this insight and separates bad luck from bad decisions.

11. Conclusion

We provide a unified framework for evaluating banker offers in Deal or No Deal. A deal is objectively good not when it maximises expected payout, but when it dominates continuation across expected value, utility, and downside risk.

This framework generalises beyond game shows to any high-stakes stopping problem involving uncertainty, regret, and asymmetric offers.
