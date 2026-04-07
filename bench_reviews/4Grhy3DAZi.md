## Summary

This paper proposes Introspective Adversarial Learning (IAL), a framework for LLM alignment that uses a Player-Advisor mechanism to generate synthetic preference data without additional human annotation. The Player generates initial responses, the Advisor provides refinement suggestions, and a reward model (PairRM) ranks the original versus refined responses to create preference pairs for training. The method combines this self-improvement loop with SPACP, a modified preference optimization loss that adds a "no-regression" penalty term.

## Strengths

- **Empirical improvements over baselines:** IAL achieves meaningful gains on the HuggingFace Open LLM Leaderboard (63.38% vs. SPIN's 62.72% and DPO's 60.92%) and MT-Bench (6.96 vs. SFT baseline's 5.98), demonstrating the method produces aligned models that maintain general task performance.

- **Diagnosis of BT modeling limitation:** Appendix B.3 provides empirical evidence that standard SPAC optimization can decrease the log-probability of target responses over training steps, justifying the SPACP penalty term. The paper shows SPACP maintains increasing real reward while SPAC oscillates and degrades (Figure 6), which is a valuable diagnostic finding.

- **Ablation on hyperparameter sensitivity:** The γ ablation (Figure 5) clearly demonstrates that extreme values (γ=50, 500) cause overfitting while moderate values (γ=5) yield optimal performance, showing the authors understand the method's stability requirements.

- **Reproducibility:** The paper provides detailed hyperparameters, dataset information, and links to anonymous code, meeting community standards for reproducibility.

## Weaknesses

- **Key ablation undermines central narrative:** Appendix B.4 reveals that when using greedy decoding, 70% of regenerated responses are identical to initial responses, and even with temperature 0.7, 3-5% remain identical. Critically, Figure 7 shows the method *still improves* when forced to use identical response pairs throughout training. This suggests the SPACP loss—not the Advisor-generated suggestions—drives the performance gains, contradicting the paper's framing. The finding deserves prominence in the main text with proper analysis, not relegation to the appendix.

- **Missing comparison to highly relevant prior work:** Constitutional AI (Bai et al., 2022), SELF-REFINE (Madaan et al., 2023), and Self-Rewarding Language Models (Yuan et al., 2024) all employ model-generated feedback and iterative self-improvement—the core mechanism of IAL's Player-Advisor loop. These are not cited as baselines despite being the closest conceptual predecessors. This significantly weakens novelty claims.

- **"No human supervision" claim is misleading:** The abstract and introduction state IAL operates "without requiring additional human supervision," yet PairRM—the sole oracle for ranking responses—is trained on human preference data. Human supervision is shifted to the reward model training phase, not eliminated. This should be stated as "reduced human annotation" or "no new human preference labels required."

- **Confound in baseline comparisons:** DPO and SPA train on UltraFeedback (preference-labeled data), while IAL and SPIN train on Ultrachat200k (SFT-format data). Performance differences cannot be cleanly attributed to the method versus the training data. A fair comparison would include DPO trained on PairRM-ranked data from the same source.

- **Limited novelty of SPACP loss:** The penalty term P_{π,πt}(x,y^t) = max{0, log π(y^t|x)/πt(y^t|x)} is directly borrowed from DPO-Positive (DPOP, Pal et al. 2024). The paper acknowledges "in the limiting case... our method reverts to DPOP" (Appendix B.4). SPACP is a straightforward combination of SPAC with DPOP's penalty—no new theoretical analysis of convergence or stability is provided.

- **Computational overhead not disclosed in main text:** Appendix B.5 shows IAL requires ~57 hours (3 iterations × 18.9h) versus DPO's ~13.8 hours—approximately 4× more compute. The main text emphasizes "efficiency" and "scalable pathway" without mentioning this significant overhead.

- **Statistical significance not reported:** All results report single-run metrics without variance across seeds. The margin over SPIN (63.38% vs. 62.72%) is 0.66 percentage points—whether this is meaningful without error bars is unclear.

- **PPO comparison contradicts claims of superiority:** Table 4 shows PPO achieves higher MT-Bench (7.08) than IAL (6.96), while IAL slightly edges PPO on the LLM Leaderboard. The paper claims IAL "slightly surpasses PPO" for the leaderboard but does not acknowledge MT-Bench shows the opposite pattern.

## Nice-to-Haves

- Human preference validation of final outputs to verify alignment with human values (beyond automated benchmarks)
- Experiments varying the reward model quality to assess robustness when PairRM is unavailable or degraded
- Analysis of Advisor feedback quality across iterations (partially addressed in Table 6, but limited to 3 samples)

## Removed Points

These points are flagged to be removed, treat them with caution:

- *Harsh critic complaint about "adversarial" terminology:* The Stackelberg game formulation (leader-follower) in Section 3.3 is a valid adversarial framework. While the Advisor provides constructive feedback, the critic/adversary role in optimization literature often serves to improve the policy—this terminology is acceptable.

- *Harsh critic complaint about equation ordering:* While the methodology section is dense, Algorithm 1 in the appendix provides clear pseudocode. The ordering complaint is a minor presentation issue.

- *Spark finder demand for human evaluation as required:* While human evaluation would strengthen an alignment paper, it is not a strict requirement at ICLR for methods proposing new optimization frameworks. The paper uses standard automated benchmarks—this is acceptable for initial validation.

- *Harsh critic complaint about MT-Bench trajectory:* The paper does show per-iteration results for LLM Leaderboard (Table 2), and MT-Bench results are presented as final-iteration comparisons, which is consistent with how baselines are reported.

- *Neutral reviewer concern about "bias propagation risk":* While theoretically valid, this is speculative without evidence of actual degradation. The empirical results do not show collapse over iterations.

## Novel Insights

The most significant insight from the reviews is the tension between the paper's narrative (Advisor-driven improvement) and empirical reality (SPACP loss may be the primary driver). The ablation showing identical response pairs still yield gains fundamentally challenges whether the Player-Advisor mechanism contributes meaningfully beyond providing any preference pairs for the optimization objective. This finding transforms the contribution from "a novel Player-Advisor framework" to potentially "a modified DPO variant that works well on self-generated preference data." Future work should investigate whether the Advisor mechanism provides *any* marginal benefit over randomly constructed pairs—a question the current experiments do not answer.

## Suggestions

- Move the identical-pairs ablation (Figure 7) and the 70% identity rate finding to the main paper. Explicitly analyze what drives the improvements and reframe the contribution accordingly.

- Add Constitutional AI and SELF-REFINE to related work, and ideally include them as experimental baselines to clarify what IAL adds beyond existing self-improvement methods.

- Revise claims of "no human supervision" to "no new human preference labels" or similar, and explicitly acknowledge PairRM's dependence on human-trained data.

- Report standard deviations across at least 3 random seeds for key metrics to establish statistical significance of improvements.

- Disclose computational cost in the main text when claiming efficiency benefits.