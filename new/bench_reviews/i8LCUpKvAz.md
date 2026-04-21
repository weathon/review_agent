Now I have a thorough understanding of the paper and calibration anchors. Let me write the final review.

## Summary

The paper introduces EQO (Exploration via Quasi-Optimism), a tabular RL algorithm that achieves the sharpest known minimax regret bound using a simple c/N(s,a) bonus instead of empirical-variance-based bonuses, and does so under the weakest known boundedness assumption (bounded values rather than bounded returns or rewards). The central technical innovation is the concept of quasi-optimism—allowing estimated values to underestimate the optimal value by a bounded amount rather than requiring full optimism—which, combined with a decoupled form of Freedman's inequality and a novel variance summation bound, enables the simpler bonus structure to suffice.

## Strengths

- **Quasi-optimism is a genuine and potentially influential conceptual contribution** (Section 4.4.2, Lemma 2). Relaxing full optimism to bounded underestimation, and showing this suffices via the variance-sum bound under value boundedness, is a clean theoretical insight that could influence future algorithm design beyond tabular RL. The backward-induction argument (Eq. 2) showing that underestimation of I₁ can be controlled by bounding I₂ is technically elegant.

- **Achieves the sharpest known regret bound under the weakest boundedness assumption** (Theorems 1–2, Table 1). The improvement over Zhang et al. (2021a) is in logarithmic factors, and the non-leading term matches Õ(HS²A). Crucially, this is the first minimax optimal result under bounded values rather than bounded returns (Assumption 1), enabled by the novel variance summation bound (Lemma 27) that does not require bounded returns.

- **The decoupled Freedman's inequality (Lemma 1)** isolates the variance term from the 1/n term, avoiding the analytical difficulty of alternating between expected and sampled trajectories that plagues prior Bernstein-based analyses. This is a useful technical tool likely applicable beyond this paper.

- **Algorithm simplicity is real** (Algorithm 1). A c/N bonus with no empirical variance computation is genuinely simpler to implement and faster per update than variance-based methods (confirmed in Table 4, Appendix G).

- **Tight PAC bounds** (Theorems 3–4) for mistake-style PAC and best-policy identification match known lower bounds for ε < H/S with the tightest non-leading term, extending the algorithm's theoretical coverage.

## Weaknesses

### Fatal
None.

### Major

- **Overclaimed practical superiority based on extremely limited experiments.** The abstract states EQO "consistently outperforms existing algorithms in both regret performance and computational efficiency," and the introduction claims "many minimax optimal algorithms often underperform compared to algorithms with sub-optimal theoretical guarantees." Yet experiments consist of only the RiverSwim environment with two configurations (S=30/H=120 and S=40/H=160), no error bars visible, and no sensitivity analysis on c. RiverSwim is specifically designed for hard exploration, where aggressive uniform exploration (which a c/N bonus provides) naturally excels. No environments testing variance-adaptation scenarios (e.g., random MDPs, sparse-reward MDPs, or MDPs with heterogeneous transition variance) are included. The claim of "consistent" superiority across algorithms and problems is not established by a single environment (Section 5, Figure 1). If practical performance were presented as a secondary observation, this would be minor; but the paper frames it as a co-equal pillar of the contribution, making the gap between claims and evidence a significant issue.

- **The introduction's claim that prior minimax optimal algorithms "often underperform" is stated without evidence** (Section 1, second paragraph: "many minimax optimal algorithms often underperform compared to algorithms with sub-optimal theoretical guarantees, such as UCRL2"). This is a strong empirical claim that requires empirical support—or at least a citation—but receives neither. It serves as a primary motivation for the paper, yet remains unsubstantiated.

### Minor

- **The improvement over Zhang et al. (2021a) is in logarithmic factors only.** Both share the same leading term Õ(H√(SAK)) and non-leading term Õ(HS²A). The abstract's "sharpest known regret bound" is accurate but may lead readers to expect a more substantial gap. The paper could be more upfront about the magnitude of the improvement.

- **The single-parameter "tuning advantage" claim is overstated.** Section 3 claims that consolidating parameters into {c_k} makes tuning "much more straightforward," but the theoretically optimal c in Theorem 1 depends on K, S, A, H, and δ. While Theorem 1 does show c can be a k-independent constant when K is known, no sensitivity analysis or practical tuning guidance is provided. The claim of practical tuning convenience is asserted but not demonstrated.

- **Tiapkin et al. (2022) also achieves minimax optimality without empirical variance computation** (Section 1.1, line 83), via posterior sampling. The paper briefly mentions this but could more clearly articulate how EQO's contribution—a UCB/OFU-style algorithm eliminating empirical variance—differs from a posterior-sampling approach that also avoids it, and when each might be preferable.

### Trivial
None.

## Nice-to-Haves

- Experiments on environments where variance-adaptation should matter (e.g., random MDPs, MDPs with heterogeneous transition variance) would strengthen or moderate the practical superiority claims.
- Sensitivity analysis for the constant c (e.g., performance with various fixed c values) would substantiate or revise the "single easy parameter" claim.
- A brief discussion of what is lost by not using empirical variance—e.g., problem-dependent bounds that depend on variance—would strengthen the paper's credibility through honest self-assessment.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"The no-empirical-variance framing misrepresents what has changed"** (Harsh Critic Critical Issue 2): The paper's claim that EQO "does not rely on empirical variances" is about the *algorithm's bonus computation*—it literally does not compute or use empirical variances. Section 4.4.1 explicitly states "Although our algorithm does not use empirical variances, all the concentration results in the analysis are based on Freedman's inequality," making clear the analysis still involves variance. The distinction between "not computing empirical variance at runtime" and "variance-free analysis" may be subtle, but the paper does not conflate them—it states both accurately. This is at most a presentation concern, not a misrepresentation.

- **Formatting/typos concerns**: Removed per hard rules about parser artifacts.

- **Missing related works**: Removed per hard rule about not confirming existence of uncited works.

- **Reproducibility concerns about hyperparameters or training logs**: Removed per hard rules on trivial reproducibility nitpicks.

- **Demand for larger-scale experiments or more environment families as a Fatal/Major issue**: Downgraded—the paper is fundamentally a theory contribution; broader experiments would strengthen the empirical claims but are not required for the core theoretical contribution.

## Novel Insights

The quasi-optimism concept introduces an asymmetric approach to controlling estimation error: instead of forcing estimates above the true value (full optimism), one tolerates bounded underestimation and controls the resulting error propagation through the variance-sum structure. This reframing separates the roles of the bonus and the analysis more cleanly than prior work—the bonus handles only the 1/N concentration error, while the variance terms are controlled analytically through the Law of Total Variance chain. This decomposition of labor between the algorithm (simple c/N bonus) and the analysis (quasi-optimism + variance summation) may be a useful design principle for future algorithms where computational simplicity is desired without sacrificing optimality.

## Suggestions

- Moderate the language in the abstract and introduction: replace "consistently outperforms" with "outperforms on RiverSwim" or add the qualifier "on hard-exploration environments." Alternatively, add 1–2 more environments to substantiate the broader claim.
- Add a brief, honest discussion of settings where variance-adaptive bonuses may outperform a uniform c/N bonus (e.g., MDPs with heterogeneous variance, where c/N over-explores well-understood state-action pairs).
- Reference Tiapkin et al. (2022) more explicitly in the algorithm section to clarify how EQO's OFU-style approach to avoiding empirical variance differs from posterior sampling.

## Evaluation

**Originality**: High. The quasi-optimism concept, the decoupled Freedman's inequality, and the variance summation bound under bounded values are all novel technical contributions. The algorithm design (simple c/N bonus achieving minimax optimality) is also original for the OFU framework.

**Importance of research question**: High. Whether minimax optimal regret can be achieved without empirical variance computation is a well-posed and meaningful question that the RL theory community cares about.

**Claims support**: The theoretical claims are well-supported by rigorous analysis (Theorems 1–4). The practical superiority claims are not well-supported by the limited experimental evidence.

**Soundness of experiments**: Limited. Only RiverSwim with two parameter configurations, no error bars, no sensitivity analysis, no diverse environments.

**Clarity of writing**: Good. The paper is well-structured, with clear algorithm presentation and an accessible proof sketch. The quasi-optimism argument is communicated effectively.

**Value to research community**: High theoretical value—the quasi-optimism insight and simplified algorithm could influence future algorithm design. Moderate practical value until broader empirical validation is provided.

## Calibration Anchors

| Paper Path | Avg Score | Comparison |
|---|---|---|
| /home/wg25r/review_agent/human_reviews/6tyPSkshtF.md | 7.5 | Similar tabular RL regret analysis; spotlight. That paper had incremental improvements and some novelty questions, but solid theory. Our paper has arguably more conceptual novelty (quasi-optimism) but weaker experiments. |
| /home/wg25r/review_agent/human_reviews/6yv8UHVJn4.md | 7.5 | Strong theory in adversarial linear MDPs, rate-optimal; spotlight. Comparable theory-first profile with limited practical applicability of first algorithm. |
| /home/wg25r/review_agent/human_reviews/hyfe5q5TD0.md | 8.0 | Oral; novel efficient algorithm for linear Bellman complete. Much stronger novelty profile. Our paper's theoretical contribution is notable but narrower in scope. |
| /home/wg25r/review_agent/human_reviews/U0c2IaQhHk.md | 5.0 | RKHS-RL with sublinear regret; rejected. Had real theory issues (incomplete proofs, unclear analysis). Our paper's theory is solid—clearly above this. |
| /home/wg25r/review_agent/human_reviews/WpQbM1kBuy.md | 4.25 | Overclaimed practical advantage, no real new theory; rejected. Our paper has genuinely strong theory, unlike this anchor. |
| /home/wg25r/review_agent/human_reviews/f0cGihOlgH.md | 4.0 | EXP-based RL with questionable theory; rejected. Our paper's theory is sound—well above this. |
| /home/wg25r/review_agent/human_reviews/VyWv7GSh5i.md | 2.75 | Incorrect derivations; rejected. Our paper has no such fundamental flaws. |

The paper sits above the medium-band anchors (4–6) because its theoretical contributions are genuinely novel and rigorous—quasi-optimism, the decoupled Freedman inequality, and the variance summation bound under bounded values are all real advances. It sits below the high-band anchors (7.5–8) primarily because: (1) the overclaimed practical superiority based on minimal experiments is a meaningful presentation issue that undermines some claims, and (2) the theoretical improvement is in logarithmic factors and the non-leading term rather than a fundamentally new order. The paper is a solid theory paper with a significant conceptual contribution but somewhat oversold practical framing.

MY FINAL SCORE: <pineapple>6.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>