## Summary
This paper proposes EQO, a tabular episodic RL algorithm that replaces the empirical-variance bonuses used in prior minimax-optimal methods with a simpler \(c/N(s,a)\) bonus. Its central technical idea is **quasi-optimism**, which relaxes standard full optimism and enables a minimax-optimal \(\tilde O(H\sqrt{SAK})\) regret guarantee with \(\tilde O(HS^2A)\) lower-order term, along with PAC/BPI guarantees, while avoiding empirical variance estimation.

## Strengths
- **Conceptually clean algorithmic contribution.** The paper’s main algorithmic change is simple and meaningful: Algorithm 1 uses \(b^k(s,a)=c_k/N^k(s,a)\) rather than empirical-variance bonuses. This is a genuine departure from the cited minimax-optimal tabular RL methods in Table 1.
- **Nontrivial theoretical novelty via quasi-optimism.** Section 4.4.2 introduces a real analytical idea, not just a parameter retuning of prior UCBVI-style proofs. Lemma 2 establishes
  \[
  V_h^k(s)+\tfrac{3}{2}\lambda_k H \ge V_h^*(s),
  \]
  which is weaker than standard optimism but appears sufficient for the regret analysis. The proof sketch explains how controlled underestimation substitutes for fully variance-calibrated optimism.
- **Strong theorem package if correct.** The paper gives regret bounds (Theorems 1–2), mistake-style PAC (Theorem 3), and best-policy identification (Theorem 4). The stated regret matches the minimax leading term up to logs and preserves the best-known \(\tilde O(HS^2A)\) lower-order dependence.
- **Useful generalization in the reward model.** Assumption 2 allows adaptive random rewards with only conditional mean matching, which is broader than the standard fixed reward-distribution setup. This is a legitimate theoretical plus.
- **Generally clear high-level exposition.** The motivation and proof sketch are readable for a theory paper, especially the decomposition around \(I_1\) and \(I_2\) in Section 4.4.2 and the use of Freedman-style decoupling in Lemma 1.

## Weaknesses

###: Fatal

None.

### Major:
- **The empirical evidence is too narrow to support the paper’s broad practical claims.** The abstract and introduction repeatedly claim that EQO is “practical,” “consistently outperforms existing provably efficient algorithms,” and resolves the tradeoff between optimal theory and practical performance. In the visible main paper, however, Section 5 reports only RiverSwim, with two displayed settings in Figure 1. That is far too limited to substantiate broad superiority claims across tabular RL. The baseline set is reasonable, but one benchmark family is not enough for the rhetoric used.
- **The “mildest assumptions / weakest boundedness” messaging is overstated and not presented carefully enough.** Section 4.1 does not assume *only* bounded value. Assumption 1 explicitly requires both \(0\le V_h^*(s)\le H\) and \(0\le R_h^k \le H\) for all \(h,k\). So the paper’s contribution is not simply “replace bounded return with bounded value”; rather, it changes the package of assumptions. The claim that this is the “mildest assumptions” or “broadest setting” is stronger than what is cleanly established in the paper itself. The comparison in Table 1 compresses this into one “Boundedness” column and obscures that the paper still imposes a per-step reward bound.
- **The strongest practicality story is weakened by the parameterization of \(c_k\).** The paper emphasizes a “single-parameter” simple algorithm, but Theorem 1 requires
  \[
  c=\max\{7H\ell_1,\;1.4H\sqrt{K\ell_1/(SA\ell_{2,K})}\},
  \]
  which depends on the known episode budget \(K\). The anytime version in Theorem 2 removes that dependence but uses a more elaborate schedule and worse constants. This does not invalidate the theoretical result, but it weakens the paper’s practical simplicity narrative and should be discussed more candidly.

### Minor
- **The practical computational-efficiency claim is under-supported in the main paper.** It is plausible that removing empirical variance computation improves constants, but Algorithm 1 remains a model-based dynamic-programming method that recomputes values over all \((s,a)\). Section 5 says execution time is better, but the evidence is deferred to Appendix G and no formal complexity comparison is given in the main text. For a paper marketing simplicity and efficiency, this deserves stronger support.
- **The “sharpest known regret bound” claim should be framed more carefully.** From the text, the main improvement over prior work is in logarithmic factors while preserving the same leading and lower-order structural terms as Zhang et al. (2021a). That can still be a valid state-of-the-art claim, but the surrounding prose sometimes reads as if the improvement were more substantial than “same order with somewhat sharper logs/constants.”
- **The non-leading term can dominate in substantial regimes, but the practical consequences are not discussed.** The paper itself notes matching the lower bound for \(K\ge S^3A\). This matters, since for moderate \(K\) the \(\tilde O(HS^2A)\) term may dominate the asymptotics emphasized in the abstract and introduction. This is not a flaw in the theorem, but the paper should better characterize when the headline improvement is meaningful.
- **Experiments lack variance reporting and parameter sensitivity analysis.** Since the paper strongly emphasizes practical tunability and single-parameter control, it would be important to show sensitivity to \(c_k\) and some measure of variability across runs. Without that, the practical tuning argument remains mostly asserted.
- **A few notation/presentation issues reduce confidence in polish.** The paper defines regret with the wrong sign in Section 2.1:
  \[
  V_1^\pi(s_1)-V_1^*(s_1),
  \]
  and similarly for cumulative regret, even though later theorems upper-bound regret by positive quantities. This is clearly a notation mistake rather than a conceptual flaw, but it should have been caught. Likewise, Proposition 1 writes \(\sum_{k=1}^K \mathrm{Regret}(K)\), which is inconsistent as written.

### Trivial
- **Pseudocode underspecifies the \(N^k(s,a)=0\) case for \(\hat r^k,\hat P^k\).** Line 10 handles zero counts for \(Q\)-values, so this is probably harmless in implementation, but the pseudocode could be cleaner.
- **The conclusion’s speculation about broader applicability is not yet supported.** The suggestion that quasi-optimism may transfer to model-free or function-approximation settings is reasonable future-work language, but currently remains speculative.

## Nice-to-Haves
- Add experiments on several qualitatively different tabular benchmarks beyond RiverSwim.
- Include a sensitivity study for \(c_k\), including misspecified \(K\) or a fixed heuristic \(c\).
- Move the runtime comparison into the main paper and provide a simple complexity table versus UCBVI-BF / EULER / ORLC / MVP.
- Better characterize the regime in which the leading \(\tilde O(H\sqrt{SAK})\) term dominates the lower-order term.
- Add an illustrative plot or case study showing quasi-optimism empirically, e.g., \(V_h^k\) vs. \(V_h^*\) or bonus magnitude comparisons against variance-based methods.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **Requests for additional comparisons to time-inhomogeneous algorithms.** The paper is explicitly about the time-homogeneous setting, so demanding comparisons to time-inhomogeneous methods is scope creep.
- **Pure formatting/style complaints and generic typo lists.** There are minor notation issues worth noting, but exhaustive typo criticism is not substantive enough for the final review.
- **Claims that the algorithm is “too similar” to prior UCBVI-style methods as a major weakness.** The high-level planning structure is indeed related to UCBVI, but the paper’s actual novelty is the \(1/N\) bonus plus the quasi-optimism analysis. That is a meaningful theoretical contribution, so “incremental because it still does DP over tabular MDPs” would be unfair if elevated too strongly.
- **Criticism about missing external related work.** Per instruction, I do not include absent-related-work complaints.
- **Reproducibility complaints rooted in appendix placement or omitted implementation minutiae.** The main issue is weak experimental breadth, not missing low-level details.

## Novel Insights
The real contribution of the paper is narrower, and stronger, than its broad framing suggests: this is best viewed as a **theory-forward paper showing that full optimism and empirical-variance bonuses are not indispensable for minimax-optimal tabular RL**. The most interesting insight is not merely that a \(1/N\) bonus works, but that by weakening the proof target from optimism to quasi-optimism, the analysis can absorb controlled underestimation and thereby decouple exploration design from explicit variance estimation. That conceptual shift is the part most likely to matter beyond this specific algorithm. Conversely, the paper is less convincing as evidence that “practical and provably optimal” tabular RL has now been broadly solved; the experiments do not establish that stronger systems-level claim.

## Suggestions
- Reframe the paper more honestly as a **strong theoretical contribution with promising but preliminary empirical evidence**, rather than as a definitive resolution of the theory/practice tradeoff.
- Clarify the assumption comparison in Section 4.1 and Table 1: explicitly separate bounded value, per-step reward boundedness, and bounded return, rather than presenting a single dominance ordering too aggressively.
- Add at least 2–3 additional tabular benchmarks and report variability across runs.
- Include a sensitivity analysis for \(c_k\), especially since Theorem 1 depends on known \(K\).
- Provide a compact runtime/complexity comparison in the main text.
- Fix the regret sign convention and the Proposition 1 notation inconsistency.

## Score and Decision
**Assessment on key axes:**  
- **Originality:** good. The quasi-optimism proof idea is genuinely novel, and the \(1/N\) bonus result is not a routine tweak.  
- **Importance:** good. Minimax-optimal tabular RL remains a canonical theory problem, and showing variance bonuses are unnecessary is meaningful.  
- **Claims supported:** mixed. The theoretical claims appear substantial from the paper text, but the assumptions framing and practical-superiority rhetoric are overstated relative to the evidence shown.  
- **Experimental soundness:** below average for the scope of the claims; only one benchmark family in the main paper.  
- **Clarity:** generally solid for a theory paper, though marred by a few notable notation mistakes.  
- **Value to the community:** meaningful, especially for RL theory researchers interested in exploration analysis.

**Calibration against human-reviewed anchors:**  
- Compared to **“Minimax Optimal Regret Bound for Reinforcement Learning with Trajectory Feedback”** (`en3NwykrHW`, scores 6/3/5/5/8/6, final reject), this paper is in a somewhat similar regime: strong theory with overemphasized asymptotics and limited empirical support. I find the present paper somewhat stronger because the core analytical idea (quasi-optimism) is cleaner and the paper is better positioned, but the same caution about headline claims vs. lower-order terms applies.  
- Compared to **“Horizon-Free Regret for Linear MDPs”** (`SdBApv9iT4`, scores 6/5/6/8, accept poster), this paper is comparably theory-centric and arguably cleaner algorithmically, but weaker empirically and a bit more overstated in its practical framing.  
- Compared to **“Gap-Dependent Bounds for Q-Learning using Reference-Advantage Decomposition”** (`6tyPSkshtF`, scores 6/8/8/8, accept spotlight), the present paper is less complete: that anchor had strong technical novelty and did not seem to overclaim practice to the same extent.  
- Compared to **“The Critic as an Explorer”** (`Z7FLmWFUFo`, scores 6/3/3/3, reject), this submission is clearly stronger theoretically and less vulnerable to novelty concerns, so it should score materially higher.

Overall, this paper looks like a **borderline-to-positive theory acceptance** if the proofs check out, but not as a full “practical and provably optimal” empirical paper. I would place it **above the weak rejects driven by lack of novelty or poor soundness, but below clearly strong accepts with both airtight framing and broader validation**.

**Final score: 6.5 / 10**  
**Decision: Accept**

MY FINAL SCORE: <pineapple>6.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>