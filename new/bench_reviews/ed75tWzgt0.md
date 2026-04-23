Now I have all the information needed. Let me write the final consolidated review.

## Summary

The paper proposes a self-play RLHF framework that balances exploration and exploitation for LLM alignment. It introduces TANPO, a two-agent algorithm derived from the framework where the max-player optimizes a DPO-style loss and the min-player adds an exploration bonus, and SADPO, a single-agent approximation using rejection sampling. The paper proves that the theoretical framework achieves sublinear regret under finite reward class and low TGEC conditions (Theorem 1), and empirically demonstrates improvements over baselines on AlpacaEval 2.0, MT-Bench, and academic benchmarks.

## Strengths

- **Clean derivation showing DPO as a special case of the self-play framework**: The derivation in Section 4.1 showing that the max-player objective reduces to the DPO loss (Eq. 11) under the minimax interchange is a useful structural insight — it reveals that DPO emerges naturally from the exploration-exploitation balanced framework, and that the source of improvement in TANPO is specifically data diversity rather than a different optimization objective for the primary policy.

- **Well-designed ablation isolating data diversity effects**: The comparison between TANPO (max-player) and Online DPO — which share the same DPO objective but differ only in training data — cleanly isolates the effect of data diversity, with TANPO (max-player) achieving 51.3% PairRM win rate against Online DPO (Section 6.2, Figure 3). This is a principled experimental design.

- **Overfitting mitigation evidence**: Figure 4 shows TANPO continuing to improve across 6 iterations (2 epochs) on the same dataset, with the min-player LC win rate increasing from ~19.5% to ~30.5% as judged by GPT-4-Turbo. This supports the claim that the two-agent competitive setup with active exploration prevents overfitting on repeated data.

- **Theoretical framework provides conceptual clarity**: The formulation in Section 3.2, where the max-player optimizes Nash value plus negative loss and the min-player optimizes best-response value plus loss, is a principled way to incorporate exploration into the game-theoretic formulation, even if the direct applicability to neural networks is conditional.

## Weaknesses

### Fatal

None.

### Major

- **Theory-practice gap undermines the central "provably efficient AND practical" framing**: The paper's title and abstract promise an algorithm that is simultaneously provably efficient and practical. However, the sublinear regret guarantee (Theorem 1) applies to the theoretical framework in Section 3.2, which requires a finite reward function class R (Assumption 1: "|R| < +∞") and exact optimization over R at each step (Eqs. 4–7). TANPO (Algorithm 1) uses neural network policies optimized by gradient descent with no explicit finite reward class — the DPO-style reparameterization absorbs the reward into the policy. The claimed equivalence in Section 4.1 depends on Assumption 4 (relegated to Appendix C), which includes conditions for the minimax theorem. The paper acknowledges this in Section 5: "the theoretical analysis in this section also applies to TANPO (Algorithm 1), provided the reward function class R meets Assumption 4," but the abstract claims "provable sample efficiency" and "theoretical guarantees" without qualifying that these are conditional and apply to a different algorithmic object than what is implemented. The title "Provably Efficient and Practical" structurally overclaims — the provable guarantees and the practical implementation do not cover the same algorithm under the same assumptions.

- **SADPO's exploration mechanism is qualitatively different from TANPO's, and the approximation is unjustified**: The min-player in TANPO (Eq. 15) includes the exploration bonus α·E_{x∼d_0, a∼π^{t+1}(·|x)}[log μ(a|x)], which encourages the min-player to cover the *current max-player's evolving* action distribution — an adaptive adversarial coverage mechanism. In SADPO (Eq. 16), this becomes α·E_{x∼d_0, a∼π_ref(·|x)}[log π(a|x)], which encourages the policy to diverge from the *fixed* reference policy regardless of training progress. These incentivize fundamentally different behaviors: adaptive exploration vs. static divergence. The paper presents SADPO as an "approximation" of TANPO but provides no theoretical analysis of how well SADPO approximates TANPO, nor any ablation comparing the TANPO-style bonus vs. the SADPO-style bonus. Without this, the claimed "approximation" relationship is merely asserted.

### Minor

- **PairRM is used for both training feedback and evaluation in Figure 3**: PairRM provides AI feedback during online alignment training (Section 6.1) and serves as the judge for the pairwise win rate in Figure 3. A model optimized for PairRM preferences and then evaluated by PairRM will naturally appear strong — a form of evaluation contamination. However, the main results in Table 1 use AlpacaEval 2.0 (GPT-4-Turbo judge) and Figure 4 uses GPT-4-Turbo, so the contamination is limited to one supplementary figure. The AlpacaEval 2.0 improvements are modest (1–4% LC win rate over the next-best baseline), and no error bars or significance tests are provided.

- **No analysis of sensitivity to K in SADPO's rejection sampling**: SADPO selects highest/lowest π_ref probability responses from K=4 samples. With K=4, this is a coarse selection — it's unclear how performance changes with different K values, and whether the diversity enforced by selecting extremes produces informative training pairs (rather than pairs where the low-probability response is simply low-quality gibberish).

- **Diversity metric measures probability difference, not semantic diversity**: Figure 1 uses |log π_ref(a^1) - log π_ref(a^2)|, which captures that two responses have different probabilities under π_ref, not that they are semantically diverse or that the diversity is useful for learning. A min-player generating near-gibberish would score high on this metric.

### Trivial

- The paper reports TANPO results based on the min-player in the main text (Table 1) without explaining why the min-player is the primary reporting target. This is a presentation choice that could be clearer, but both players' results are available in the appendix.

## Nice-to-Haves

- A "K-sample DPO" baseline (sample K responses, select best/worst by reward model score, train with standard DPO) would isolate whether the improvement comes from the data diversity strategy alone or from the specific TANPO/SADPO algorithmic structure.
- An ablation comparing SADPO with and without the exploration bonus (α·E_{a∼π_ref}[log π(a|x)]) would quantify its contribution.
- Qualitative examples of min-player responses would reveal whether the "exploration" produces meaningfully different but coherent outputs, or simply degraded text.
- Error bars or significance tests for the AlpacaEval 2.0 results would strengthen confidence in the modest improvements.

## Removed Points

These points are flagged to be removed, treat them with caution.

- **"Missing baselines (SPIN, DNO)"**: Removed per hard rule — do not flag missing related works or baselines.
- **"Missing appendix, Assumption 4 in appendix"**: Removed per hard rule — the parser strips appendix sections; they exist in the original submission.
- **"Computational complexity of enumerating finite reward class R"**: Removed as this is a standard feature of theoretical RLHF frameworks, not a novel criticism, and the paper's contribution is precisely to show the framework reduces to practical objectives.
- **"Overfitting mitigation claim based only on AlpacaEval 2.0 with length bias"**: Weakened — the paper uses length-controlled win rate (specifically designed to address length bias) and the evaluation is by GPT-4-Turbo, so length bias concerns are partially addressed.
- **"The pseudo-code / implementation details are missing"**: Removed per hard rule on reproducibility nitpicks.
- **"Request for theoretical proofs for neural network settings"**: Moved to nice-to-have; demanding proofs for neural network parameterizations is not standard in this community.
- **Strength claim "Provable sublinear regret for self-play RLHF"**: This strength is partially filtered — the provable guarantee applies to the theoretical framework, not directly to the implemented algorithms. The framework's regret guarantee is a valid contribution, but claiming it as a direct strength of the practical system overstates the case.

## Novel Insights

The most insightful observation across the reviews is that TANPO's value proposition is fundamentally about *data diversity*, not algorithmic novelty for the primary policy. The max-player in TANPO optimizes the exact same DPO objective as Online DPO — the only difference is the training data distribution. This means the entire theoretical machinery of the self-play framework, while intellectually interesting, ultimately serves the practical function of generating more diverse training pairs. This raises the question of whether much simpler data augmentation strategies (e.g., temperature scaling, K-sample selection) could achieve similar gains without the two-agent overhead, which the paper does not investigate.

## Suggestions

- Qualify the title and abstract: replace "Provably Efficient and Practical" with a framing that honestly separates the theoretical guarantees (for the finite-reward-class framework) from the practical algorithms (TANPO/SADPO), e.g., "Theory-Motivated Self-Play for Better LLM Alignment."
- Add an ablation comparing SADPO with the TANPO-style exploration bonus (α·E_{a∼π^t}[log π(a|x)]) vs. the current SADPO bonus (α·E_{a∼π_ref}[log π(a|x)]) to justify the substitution.
- Add a simple "K-sample DPO" baseline that samples K responses, selects best/worst by reward model, and trains standard DPO — this would clarify whether the algorithmic structure or just the data diversity drives improvements.

## Calibration Anchors

| Paper | Path | Avg Score | Comparison |
|-------|------|-----------|------------|
| Iterative Nash Policy Optimization (INPO) | /home/wg25r/review_agent/human_reviews/Pujt3ADZgI.md | 6.0 | Very similar: game-theoretic RLHF with no-regret learning. INPO was accepted as Oral with clean algorithm and strong experiments but also limited novelty clarity. This paper has a larger theory-practice gap than INPO. |
| Value-Incentivized Preference Optimization (VPO) | /home/wg25r/review_agent/human_reviews/SQnitDuow6.md | 5.5 | Similar: theory-motivated RLHF with theory-practice gap. VPO was accepted as Poster. This paper has comparable theory but weaker empirical gains. |
| Crucial Role of Samplers in Online DPO | /home/wg25r/review_agent/human_reviews/F6z3utfcYw.md | 6.0 | Related: analyzes sampling in online DPO. Accepted as Poster. This paper's diversity argument is complementary but less rigorously analyzed. |
| Provable Reward-Agnostic PbRL | /home/wg25r/review_agent/human_reviews/yTBXeXdbMf.md | 7.5 | Stronger theory paper with clean PAC guarantees. This paper's theoretical contributions are less novel and less tight. |
| CVX-DPO | /home/wg25r/review_agent/human_reviews/EVZnnhtMNX.md | 3.0 | Claims global optimality for DPO via convex reformulation, but poorly justified. This paper is clearly better — the theory is more standard but sound. |
| RL Stability Guarantee | /home/wg25r/review_agent/human_reviews/vBNTeQ7dPP.md | 2.5 | Theory relies on unguaranteed assumption. This paper's theory-practice gap is less severe — the conditional nature is acknowledged in Section 5. |

**Score reasoning**: This paper sits between VPO (5.5) and the low-scoring CVX-DPO/RL Stability papers (2.5-3.0). The theory-practice gap and SADPO approximation concerns are real but not as severe as the low-scoring papers, where theoretical claims were fundamentally undermined. The paper is weaker than INPO (6.0) because INPO doesn't claim provable efficiency for its practical algorithm — it's motivated by game theory but doesn't overclaim. This paper's overclaiming in the title/abstract is a meaningful drawback that lowers it below the 6.0 tier. The DPO insight and the data diversity mechanism are genuine contributions, but the modest empirical gains and the unjustified SADPO approximation keep it from scoring higher.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>