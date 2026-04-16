## Summary

The paper proposes EQO, a tabular finite-horizon RL algorithm that uses a very simple exploration bonus \(b^k(s,a)=c_k/N^k(s,a)\) instead of empirical-variance-based bonuses. Through a new analytical notion of “quasi-optimism” and Freedman-style concentration, the authors derive minimax-optimal regret bounds with improved logarithmic factors and matching PAC/BPI sample-complexity guarantees, and present RiverSwim experiments suggesting better regret and runtime than several strong baselines.

## Strengths

- **Conceptual novelty in analysis (quasi-optimism):**  
  The paper departs from the standard “full optimism” paradigm. Lemma 2 shows that estimates satisfy
  \[
  V_h^k(s) + \tfrac{3}{2}\lambda_k H \ge V_h^*(s),
  \]
  i.e., they may underestimate but only by a controlled amount. The way this quasi-optimism is proved—via a Freedman-type bound (Lemma 1) that separates variance and \(1/n\), and a difference-type variance inequality (Lemma 27)—is technically original and may be useful beyond this specific algorithm.

- **Algorithmic simplicity and clarity:**  
  Algorithm 1 is a standard model-based OFU planner except for its exploration term \(c_k/N^k(s,a)\) and a simple handling of unvisited pairs (set to \(H\)). It avoids empirical variances entirely at the algorithmic level, uses a single scalar parameter \(c_k\), and is clearly specified. This is an appealing contrast to previous minimax algorithms with more complex, variance-based bonuses.

- **Strong and broad theoretical guarantees:**  
  - The regret bounds in Theorems 1 and 2 have leading term \(\tilde O(H\sqrt{SAK})\) and non-leading term \(\tilde O(HS^2A)\), matching known lower bounds in order and matching the best-known non-leading term in the time-homogeneous setting.  
  - The paper also establishes mistake-style PAC bounds (Theorem 3) and best-policy identification guarantees (Theorem 4) that meet known lower bounds up to logs. This breadth (regret + PAC + BPI) for a single simple algorithm is a significant theoretical package.

- **Weaker modeling assumptions (within the usual bounded-reward framework):**  
  Assumption 2 allows the reward distribution at \((s_h^k,a_h^k)\) to adapt to the full past history, requiring only the conditional mean \(\mathbb{E}[R_h^k\mid\mathcal{F}_h^k]=r(s_h^k,a_h^k)\), and not that rewards be i.i.d. given the state-action. This martingale-style assumption is more general than the standard i.i.d. reward model and is nontrivial for the analysis.

- **Clear high-level exposition of the proof ideas:**  
  Section 4.4 gives an informative sketch: how Freedman’s inequality in the Lemma 1 form is used, how the variance-sum is bounded without assuming bounded returns, and how the auxiliary sequence \(\lambda_k\) trades off estimation and exploration. This makes the technically involved proofs more accessible.

## Weaknesses

### Fatal

None. The core algorithm and analysis are coherent, and I see no error that invalidates the main regret/PAC claims based on the text provided.

### Major

- **Overstated “weakest boundedness assumption” claim**  
  The paper’s positioning heavily emphasizes that it works under the “mildest” or “weakest” boundedness assumptions (Abstract, bullets, Table 1, Section 4.1). However, Assumption 1 states:
  > “\(0 \le V_h^*(s) \le H\) holds for all \(s,h\), and \(0 \le R_h^k \le H\) holds for all \(h,k\).”

  The text then argues:
  > “Since the value function is the expected return, *our bounded value condition is weaker than the bounded return assumption* (and hence, also weaker than the widely used uniform boundedness of rewards) used in the previous literature.”

  This conflates two aspects:
  - A bound on the **expected** return or value \(V_h^*(s)\); and  
  - An almost-sure bound on each reward, \(0 \le R_h^k \le H\), which they also impose.

  Prior “bounded return” work allows the total return per episode to be bounded by \(H\) while per-step rewards may vary (possibly more flexibly than \(R_h \in [0,H]\) individually) but is still within the standard bounded-reward framework. Here, the analysis repeatedly uses Freedman-style inequalities (Lemma 1) with a hard bound \(X \le C\), which in the RL instantiations is ultimately guaranteed via the reward bound \(R_h^k \le H\). There is no demonstration that the analysis would continue to hold under a genuinely weaker condition such as only \(V_h^*(s)\in[0,H]\) without an almost-sure bound on \(R_h^k\).

  Within the usual tabular RL regime where rewards are assumed bounded by a constant, their “bounded value” narrative is somewhat misleading: they still need reward boundedness and then add a further bound on \(V_h^*\). The hierarchy “reward-bounded > return-bounded > value-bounded” as presented is not convincingly justified under the precise form of Assumption 1. Since this “weakest assumption” story is repeatedly used to claim superiority over prior minimax methods (Table 1, conclusion), it should be toned down or clarified (e.g., explicitly: “we match the standard bounded-reward assumption and additionally require only that \(V^*\) be bounded, not the entire return pathwise,” with a careful comparison).

- **“Sharpest known regret bound” claim not fully supported by comparison**  
  The paper repeatedly states that its regret bound is “the sharpest known” and has “even tighter logarithmic factors than Zhang et al. (2021a)” (Abstract, contributions, Table 1, Section 4.2). While theorems do give leading term
  \[
  38H\sqrt{SAK\ell_1\ell_{2,K}},
  \]
  with \(\ell_1\) and \(\ell_{2,K}\) log terms, the manuscript does not explicitly state the exact logarithmic dependence from Zhang et al. (2021a) nor provide a direct symbolic comparison (e.g., a lemma showing that for all \(S,A,H,K,\delta\) their product \(\sqrt{\ell_1\ell_{2,K}}\) is asymptotically smaller than the polylogs in Zhang et al.). The text simply asserts this improvement.

  Given how strong the claim “sharpest known” is, a reader is left to take on faith that the logs are strictly better in all regimes. This is an over-claim relative to what is explicitly shown in the text; the safe, well-justified statement is that EQO matches the known minimax rate and matches or slightly improves previous logarithmic factors, rather than definitively dominating all prior bounds.

- **Empirical evaluation is too narrow for the claimed practical superiority**  
  The abstract and conclusion present EQO as “consistently” and broadly empirically superior (“empirically outperforms existing algorithms… providing the best of both theoretical soundness and practical effectiveness”). However, Section 5 reports results on a single environment family (RiverSwim) with two \((S,H)\) configurations. While RiverSwim is indeed a hard-exploration chain and a standard benchmark, this is far from sufficient to support general claims of practical superiority over a diverse set of tabular MDPs.

  Moreover:
  - There are no confidence intervals or variability measures; the stability of the advantage is unclear.  
  - There is no sensitivity analysis for the key parameter \(c_k\) (chosen values vs. theory, robustness to mis-tuning).  
  - Only brief mention is made of runtime comparisons (Table 4 in the appendix), with no quantitative summary in the main text.

  As a result, the current experiments should be interpreted as a promising *case study*, not as evidence that EQO is broadly more practical than prior minimax methods. The empirical narrative in the abstract and conclusion is overstated given the data presented.

### Minor

- **Rhetoric around “no empirical variance” somewhat overplays conceptual distance**  
  Algorithmically, EQO indeed avoids empirical-variance-based bonuses; its bonus is \(c_k/N^k(s,a)\) and easy to compute. At the same time, the analysis is deeply variance-centric: Lemma 1, Lemma 2, and the central recursion in (2) all involve \(\mathrm{Var}(V_{h+1}^*)\) and sums of variances along trajectories. The paper’s statement that this “demonstrates that empirical variance … is not necessary for achieving efficient exploration” is correct in the narrow sense that the algorithm does not *compute* empirical variances, but the mechanism and analysis are still grounded in variance considerations. This should be presented a bit more cautiously to avoid giving the impression that variance structure has become irrelevant.

- **“First to use \(c/N\) bonus with regret guarantees in RL” is plausible but not rigorously argued**  
  The related work section claims:
  > “our algorithm is the first to use an exploration bonus of the form \(c/N\) for the reinforcement learning setting and attain regret guarantees,”

  and contrasts with Simchi-Levi et al. in bandits. This may well be true, but given the extensive literature on count/pseudo-count bonuses and UCB-like model-based methods, it is a strong historical statement. The paper does not systematically discuss closely related count-based or pseudo-count approaches to rule out effectively \(O(1/N)\)-like schemes. This is not a correctness problem, but the “first” language should be softened (e.g., “to our knowledge, no prior tabular RL regret analysis has used a purely \(c/N\)-type bonus without variance or \(\sqrt{1/N}\) terms”).

- **Known-\(K\) vs anytime setting and practical tuning are under-discussed**  
  Theorem 1 requires setting a constant \(c\) using the total horizon \(K\). Theorem 2 handles the anytime case via a doubling-style schedule for \(c_k\). This is a reasonable solution theoretically, but the paper does not discuss:
  - How \(c_k\) is set in experiments (do they use the anytime schedule, a hand-tuned constant, or the theorem’s prescription?).  
  - How sensitive EQO is to the choice of \(c_k\) in practice.  

  Given that “single parameter” tunability is advertised as a practical virtue, a brief empirical or conceptual discussion of this trade-off would add credibility.

- **Magnitude of quasi-optimism gap not interpreted**  
  Lemma 2 allows underestimation by \(\frac{3}{2}\lambda_k H\), and in the regimes used in the theorems, \(\lambda_k\) can be as large as 1. So in principle the estimate could be up to \(1.5H\) below optimal. The authors do not explain whether this is just an analysis artifact (with much smaller gaps in practice) or whether significant underestimation is expected in some regimes. This is a minor clarity/exposition issue.

- **Empirical section lacks some standard reporting details**  
  The description of experimental setup (number of runs, random seeds, horizon \(K\), and exact hyperparameters for all baselines) is deferred to the appendix. For a paper making nontrivial empirical claims, adding at least a concise summary in the main text would help readers gauge how robust the reported curves are.

### Trivial

- Slight sign inconsistency in the definition of “regret” in Section 2.1 (they write \(V_1^\pi - V_1^*\), but later treat regret as something positive to be minimized) may confuse some readers; the intent is clear but could be cleaned up.

- Minor wording issues (e.g., “fastest convergence to optimality” when improvements are only in logs) add hype without changing substance.

## Nice-to-Haves

- Additional environments beyond RiverSwim, such as randomly generated MDPs with varying structure (multi-branch chains, multiple goals, different reward sparsity patterns), to show how EQO behaves across a broader class of problems.

- Sensitivity plots for the parameter \(c_k\) (for both constant and anytime schedules), showing how robust performance is to mis-calibration relative to the theoretically suggested values.

- A brief intuitive explanation (possibly with a toy example) of why the \(c/N\) bonus, combined with the quasi-optimism analysis, does not “overshoot” exploration compared to \(\sqrt{1/N}\)-type bonuses despite discarding variance information from the algorithm itself.

- A small example or diagram illustrating the quasi-optimism mechanism: e.g., tracking \(V_h^k(s)\) and \(V_h^*(s)\) on a simple chain MDP to show how the underestimation is bounded and how it interacts with the \(c/N\) bonus.

## Removed Points

These points are flagged to be removed, treat them with caution.

- **Claim that Assumption 1 is strictly stronger than bounded-return in the sense of allowing rewards larger than \(H\)**  
  The harsh review suggested that bounded-return might allow “individual rewards larger than \(H\) at an intermediate step,” but the paper itself frames all comparisons in terms of scaling to \([0,H]\) and standard bounded-reward assumptions. Given the usual normalization in RL theory (rewards typically assumed in \([0,1]\) or \([0,H]\)), arguing over whether one could allow a single larger reward is not central and not grounded in the paper’s own setting. The substantive (and retained) criticism is that the “weakest” claim is overstated, not that the assumption is literally stronger in some contrived sense.

- **Speculation that the analysis could or should work under unbounded rewards with only bounded values**  
  The prior critique implied that the authors ought to handle the case of only \(V^*\) bounded and no almost-sure bound on \(R_h^k\). This goes beyond the scope of standard tabular RL theory, where bounded rewards are nearly universal. Demanding removal of this assumption would be scope creep; the more reasonable point (kept above) is that the paper should not oversell its assumption as substantially weaker than others.

## Novel Insights

The genuinely novel conceptual contribution here is the formalization and exploitation of “quasi-optimism”: instead of ensuring that value estimates are optimistic in the usual upper-confidence sense, the analysis tolerates a uniform, controlled degree of underestimation and shows that this is sufficient for minimax-optimal regret and optimal PAC/BPI rates. Combined with a Freedman-style inequality that separates variance and visit-count dependencies, this offers a fresh analytic template suggesting that sophisticated empirical-variance bonuses may be more an artifact of proof technique than a necessity for optimal exploration in tabular RL. While the empirical section is limited, this analytic perspective is likely to influence future work, including model-free and function-approximation settings.

## Suggestions

- **Tone down or clarify global claims in the abstract/introduction/conclusion.**  
  Replace language like “sharpest known regret bound” and “weakest boundedness assumption” with more precise formulations such as “minimax-optimal regret with improved logarithmic factors over prior work” and “retains standard bounded-reward assumptions while requiring only bounded optimal values rather than bounded returns.” Similarly, rephrase “empirically outperforms existing algorithms” to “performs strongly on RiverSwim experiments” unless/until a broader empirical suite is provided.

- **Add an explicit comparison to Zhang et al. (2021a)’s bound.**  
  Include the full regret expression from that work (with log factors) and a short lemma or paragraph that algebraically compares the two logarithmic dependencies to justify the “sharper logs” claim. This will make the theoretical positioning much more transparent.

- **Clarify the role and necessity of the reward bound in Assumption 1.**  
  Explicitly state that the analysis uses bounded rewards (via Lemma 1’s \(X \le C\)) and that, in that sense, the boundedness conditions are comparable to prior work, with the new aspect being that only \(V^*\) (not full return) is bounded. Remove or carefully qualify any statements implying a strict hierarchy without caveats.

- **Expand and document the experimental section.**  
  - Add at least one or two additional classes of tabular MDPs.  
  - Report confidence intervals over multiple runs.  
  - Specify how \(c_k\) and \(\delta\) were chosen in practice and whether theoretical or tuned values were used.  
  - Summarize runtime comparisons in a small table in the main text (not only the appendix).

- **Add a brief discussion on parameter sensitivity and known-\(K\) vs anytime usage.**  
  Explain to practitioners which variant they should use when \(K\) is unknown, how sensitive the algorithm is to mis-specified \(c_k\), and whether simple heuristics (e.g., constant \(c_k\) scaling with \(H\)) work well in practice.

- **Clarify the regret sign convention.**  
  Adjust or annotate the definition so it aligns with the standard “optimal minus realized” convention or at least clearly mark that the paper uses the opposite sign but always upper-bounds its magnitude.

## Score and Decision

To calibrate, I compared against several human-reviewed RL theory papers:

- **/home/wg25r/review_agent/human_reviews/6tyPSkshtF.md** (“Gap-Dependent Bounds for Q-Learning using Reference-Advantage Decomposition”) — accepted (spotlight), scores 6/8/8/8. That paper presents substantial new gap-dependent analyses in a very active area, with stronger and more clearly novel results and, typically, modest or no experiments.  
- **/home/wg25r/review_agent/human_reviews/en3NwykrHW.md** (“Minimax Optimal Regret Bound for RL with Trajectory Feedback”) — rejected, scores 6/3/5/5/8/6; reviewers appreciated the technical work but criticized limited novelty relative to existing paradigms and some over-claims.  
- **/home/wg25r/review_agent/human_reviews/qybJSeG2VH.md** (“Achieving Minimax Optimal Sample Complexity of Offline RL”) — generally seen as technically solid but somewhat incremental; scores in the 3–5 range.  
- **/home/wg25r/review_agent/human_reviews/SdBApv9iT4.md** (“Horizon-Free Regret for Linear MDPs”) — accepted (poster), scores 6/5/6/8, with strong theory but limited empirical validation.

Relative to these anchors, EQO’s paper:

- Has a **solid and somewhat original analytic idea** (quasi-optimism + Freedman decomposition), arguably more conceptually fresh than some incremental re-analyses, so it sits above clearly incremental works like some re-analyses in qybJSeG2VH.  
- Has **less breadth or boldness than a spotlight-level contribution** like 6tyPSkshtF, which delivers new gap-dependent theory in a more competitive area.  
- Shares with en3NwykrHW and SdBApv9iT4 the pattern of strong theory and modest or narrow experiments, but with somewhat clearer conceptual novelty than en3NwykrHW and arguably on par with SdBApv9iT4 in terms of overall technical interest.

Given this, I judge it as a *borderline but positive* theory paper: it makes a meaningful conceptual and technical contribution, but the framing/claims and experiments need tempering and expansion.

**My overall score: 6.0 (weak accept).**  
The paper should be accepted if the venue is willing to take strong-theory, limited-experiment tabular RL work, provided that the claims about assumptions, “sharpest” bounds, and empirical generality are toned down. It is above the threshold of papers like en3NwykrHW that were ultimately rejected and below standout theoretical contributions like 6tyPSkshtF.

MY FINAL SCORE: <pineapple>6.0</pineapple>  
MY FINAL DECISION: <orange>Accept</orange>