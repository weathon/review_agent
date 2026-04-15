## Summary
This paper proposes MC-DML, a text-game planning method that plugs an LLM prior into MCTS and augments it with two memory streams: in-trial memory from the current trajectory and cross-trial memory formed from reflections on failed rollouts. The main empirical claim is that this combination improves early-stage planning on Jericho games and can outperform prior Jericho agents that rely on iterative planning-then-learning loops.

## Strengths
- **The paper targets a real weakness of prior Jericho-style MCTS+RL approaches and offers a concrete alternative.** Section 2.1 explicitly motivates the problem with PUCT-style methods: policy learning depends on self-generated search data and typically needs multiple planning/learning iterations. MC-DML replaces the learned prior with an LLM prior and uses reflection-generated memory during search, which is a sensible and technically coherent design for text-based games.
- **The cross-trial memory mechanism appears genuinely useful rather than cosmetic.** In Table 4, removing \(\mathcal{M}_c\) lowers performance on all four ablated games (e.g., Zork1: 48.66 to 38.33; Detective: 346.67 to 326.67; Ztuu: 23.67 to 20.66), and removing both memories hurts further. This supports the claim that failure reflections are contributing meaningfully to action guidance during search.
- **There are several substantial empirical wins, not just marginal improvements.** The gains on Deephome (67 vs 35 for MC-LAVE-RL in Table 2), Zork1 (48.66 vs 45.2), Detective (346.67 vs 330), and especially Ztuu (23.67 vs 7 for MC-LAVE-RL) are notable. These are the kinds of improvements that make the method interesting even if some broader claims are overstated.
- **The qualitative bottleneck analysis is specific and aligned with the claimed mechanism.** Table 5 concretely shows how cross-trial memory can shift the search away from the deceptively rewarding but fatal `open trap` action toward `take lantern`, which is exactly the kind of bottleneck state the paper highlights in Figure 1.
- **The paper is appropriately scoped to Jericho-style planning with valid actions rather than claiming open-ended text-game solving.** The setup in Section 4.1 clearly states use of Jericho’s valid-action handicap, and the method is evaluated in that regime.

## Weaknesses
###: Fatal
None.

### Major:
- **The strongest headline claim—beating methods that require multiple planning-then-learning iterations using only initial planning—is only directly established on one game.** Table 3, which is the key evidence for this claim, is only for Zork1. The abstract and introduction phrase this as a broad advantage of the method, but the paper does not provide iteration-wise evidence on multiple games, especially not on the hard cases where such a claim would matter most.
- **The benchmark comparison is not controlled tightly enough to fully support the paper’s broad superiority framing.** Tables 1 and 2 compare against many prior methods, but the paper does not clearly state in the main text whether the strongest direct baselines (especially PUCT-RL and MC-LAVE-RL) were all rerun under exactly the same budgets and protocol, or whether some results are inherited from prior papers. Since MC-DML uses GPT-3.5 priors, dynamic pruning, and a particular action-probability construction, stronger apples-to-apples evaluation is needed for a definitive superiority claim.
- **The mechanistic claim about why cross-trial memory helps is plausible but not cleanly isolated.** The ablation in Table 4 shows that \(\mathcal{M}_c\) matters, but it does not separate the effect of reflection semantics from simply giving the LLM more prompt context. There is no control such as random reflections, generic summaries, or failed-trajectory text without reflection. So the paper supports “cross-trial memory helps,” but not yet the stronger causal story that reflective failure analysis specifically is the key driver.
- **The paper overstates generality beyond what is empirically shown.** Section 3.3 says the method aims for a “more general solution,” but all experiments are on Jericho games with valid actions. The paper does not demonstrate generality beyond this benchmark family, so this should be presented as a motivation or hypothesis rather than a demonstrated property.

### Minor
- **Some claims about significance and consistency are stronger than the evidence warrants.** MC-DML is not best on Ludicorp, and several reported wins are ties on saturated/easy games (Balances, Temple). With only 3 runs for MC-DML, the paper should be more careful in describing the breadth and certainty of superiority.
- **The search-budget specification is underspecified in the main text.** Algorithm 1’s `SEARCH` routine stops with `until MaxDepthReached()`, which is unusual wording for MCTS, and the main paper does not clearly spell out the simulation/time budget per root decision. Because planning budget is central to fairness and efficiency claims, this deserves clearer exposition.
- **The paper says MC-DML “adjusts action value estimation,” but operationally the LLM acts through the prior term in PUCT rather than directly changing the \(Q\)-update.** Equation (3) shows the LLM influences action selection via the prior bonus, not the Monte Carlo return backup itself. This is still useful, but the wording should be more precise mechanistically.
- **The dynamic pruning component is useful but not broadly validated as an important ingredient.** Table 4 suggests dynamic pruning is crucial on Ztuu, mostly neutral on Zork1 and Detective, and slightly worse on Deephome. So the evidence supports “sometimes helpful,” not “generally important.”

### Trivial
- **The standalone-LLM discussion draws an invalid inference.** In Section 4.2 the paper says poor LLM-agent performance “indicates that LLM does not have knowledge of the game’s walkthrough under the current prompting setting.” That conclusion does not follow: poor performance could also result from grounding, long-horizon control, or action-sequence recovery difficulties. This sentence should simply be removed or softened.

## Nice-to-Haves
- Report compute-related metrics such as number of LLM calls, token usage, or wall-clock time, especially since the paper contrasts itself with iterative planning-then-learning methods on efficiency grounds.
- Add a cleaner ablation for reflection quality, e.g., replacing reflections with generic summaries or random text, to verify that the gains come from meaningful failure analysis rather than prompt length.
- Include a “w.o. \(\mathcal{M}_i\)” ablation alone; the current table has “w.o. \(\mathcal{M}_c\)” and “w.o. \(\mathcal{M}_c,\mathcal{M}_i\),” but not the symmetric intermediate case.
- Provide iteration-wise comparisons beyond Zork1, ideally on Deephome and Ludicorp, which would directly test the paper’s most distinctive claim.
- Analyze the Ludicorp failure case to clarify the boundary conditions of the method.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“Unfair comparison because baselines require multiple iterations while MC-DML uses only initial planning.”** Removed as stated. This asymmetry actually favors the baselines, not the proposed method, so under the review rules it should not count as a weakness. The valid underlying issue is narrower: the paper overgeneralizes from a one-game initial-planning comparison.
- **“The paper should cite/add more related work baselines.”** Removed. I cannot verify missing related work externally, and the review should not speculate.
- **“LLM baselines are weak, suggesting implementation issues.”** Removed. The paper’s reported LLM and Reflection agents are what was evaluated; poor performance alone is not evidence of faulty implementation.
- **Generic concerns that LLM-generated reflections may be unreliable because LLMs are imperfect.** Removed in that broad form. The paper is allowed to use imperfect LLM feedback; the valid concern is instead that the paper does not empirically validate reflection quality or isolate its mechanism.

## Novel Insights
A key synthesis across the paper and reviews is that the strongest evidence here is not “LLMs replace RL” or even “LLMs make MCTS generally better,” but something more specific: **LLM priors become materially more useful when coupled to a search process that can repeatedly revisit the same local bottleneck and inject concise failure-conditioned guidance at that root.** The paper’s best results and analyses are concentrated exactly in such cases (e.g., Zork1 bottlenecks, Deephome). This suggests MC-DML’s real contribution may be best understood as a targeted method for *root-local policy adaptation during planning* rather than a broadly general planning framework. Framing it that way would make the paper’s claims both sharper and better supported.

## Suggestions
- Narrow the headline claim: say that MC-DML shows strong **initial-planning performance** and beats iterative baselines **on Zork1**, rather than broadly claiming this across the benchmark.
- Clarify in the main text whether PUCT-RL and MC-LAVE-RL were rerun under the same environment settings and budgets, and if not, temper the comparison language.
- Add one or two controls that isolate cross-trial memory semantics from extra prompt context.
- Precisely specify the planning budget per decision in the main paper.
- Revise the mechanistic wording to say the LLM changes the **prior-guided exploration term** in PUCT, not the Monte Carlo backup rule.
- Remove the unsupported walkthrough-contamination inference from Section 4.2.
- Add a brief discussion of where MC-DML does not help, especially Ludicorp and the saturated games.

## Score and Decision
**Axis-by-axis assessment**
- **Novelty:** Moderate. Using an LLM as a prior in MCTS is not itself new, but the integration of in-trial and cross-trial memory inside Jericho-style planning is a meaningful extension with a concrete use case.
- **Technical soundness:** Moderately good, but not airtight. The algorithm is coherent and the ablations support some core design choices, yet the paper overstates what is mechanistically established.
- **Empirical support:** Good but incomplete. Several results are strong, especially on hard games, but the strongest comparative claim is only backed on one game and comparison control details are not sufficiently explicit.
- **Significance:** Moderate to fairly high for text-game agents and LLM-guided planning. The Deephome/Zork1 results are strong enough to matter.
- **Clarity:** Generally solid conceptually, though a few algorithmic and claim-level imprecisions should be fixed.

**Calibration against similar papers/reviews**
I compared this submission against the following human-reviewed papers:
- **ExACT: Teaching AI Agents to Explore with Reflective-MCTS and Exploratory Learning** (`GBIUbwW9D8.md`, accepted, scores 6/5/6/6). That paper was viewed as having stronger task-scale evaluation and more thorough ablations for a related reflective-MCTS idea. The current paper is somewhat narrower and less thoroughly validated, so it lands a bit below that acceptance bar.
- **Alphazero-like Tree-Search can guide large language model decoding and training** (`fLO9VaAb3B.md`, reject, scores 5/5/6/6). That paper suffered from mixed empirical evidence and cost/fairness concerns. The current paper has clearer domain-specific wins than that submission, so it should score somewhat higher than a borderline reject of that type.
- **Planning with MCTS: Enhancing Problem-Solving in Large Language Models** (`sdpVfWOUQA.md`, reject, scores 3/3/3). That paper appears substantially weaker, including reviewers questioning whether it was even really MCTS. The current paper is clearly much stronger than that.
- **Foundation Models for Enhanced Exploration in Reinforcement Learning** (`RiDtvlNiqp.md`, reject, scores 3/3/3/3). That paper was criticized for overclaiming generality and weak empirical support for its core premise. The current paper shares some overclaiming issues but has much stronger benchmark evidence in its target domain.
- **Can Language Agents Approach the Performance of RL?** (`F0q880yOgY.md`, mixed 8/3/3/5/3). That paper’s mixed reception reflects a useful calibration point: interesting idea and some promising results, but evaluation/clarity weaknesses prevented consensus. The current paper feels similarly promising but better focused.

Overall, this submission is **better than the clear rejects in this area**, and it has enough real empirical signal that I would not dismiss it. But for ICLR standards, the paper still **overclaims relative to its evidence**, and the missing controlled comparisons around its most distinctive claim keep it below the accept line for me.

**Score: 5.8**

**Decision: Reject**

MY FINAL SCORE: <pineapple>5.8</pineapple>
MY FINAL DECISION: <orange>Reject</orange>