Now I have thoroughly verified all claims. Let me write the final review.

## Summary

The paper proposes a weak-to-strong control mechanism for scaling Bayesian Theory-of-Mind (ToM) reasoning to large language models (up to 405B parameters). Instead of directly post-training a large LM—a costly and difficult-to-tune process—the method transfers ToM-specific behavioral shifts from a post-trained small LM to guide a large LM's likelihood estimation at test time via the probability ratio π^E/π^N (Equation 6). This enables Bayesian inverse planning to leverage the world knowledge of large LMs without incurring post-training costs at scale.

## Strengths

- **Genuine improvement over prior SOTA**: The method achieves 81.3% overall accuracy on the multimodal MMToM-QA benchmark, a ~4.6% improvement over the best prior baseline BIPALM w/ Llama-2-7B (76.7%). This improvement is consistent across both belief inference (87.0% vs. 80.3%) and goal inference (77.0% vs. 73.0%), overcoming the "seesaw effect" where prior methods traded off between task types (Table 1, Section 3.2).

- **Systematic scaling analysis**: Table 2 provides a thorough evaluation across three Llama families (2, 3, 3.1) and model sizes from 7B to 405B. The positive correlation between model size and weak-to-strong performance, and the continued improvement from 70B to 405B (79.38% → 81.29%), demonstrates that scaling benefits are real and not yet saturated.

- **Strong transferability to unseen environments**: Table 4 shows the method generalizes across five diverse unseen scenarios (Andersen tales, ancient Egyptian, outer space, wild west, medieval castle). The 8B↔405B configuration maintains 79.7–80.5% accuracy across all unseen domains, demonstrating robust out-of-distribution generalization.

- **Practical downsizing analysis**: Table 3 demonstrates that 4B-parameter controllers (both width- and depth-reduced) paired with 70B models achieve 78.38–78.52%, close to the full 8B↔70B result of 79.38%. This is practically valuable for deployment where controller size matters.

- **Qualitative understanding of the mechanism**: Section 3.6 (Figure 2) shows progressive likelihood redirection across Bayesian steps, and Section 3.7 (Figure 3) reveals that post-training shifts the small LM's focus toward fine-grained item-level concepts, explaining why the behavioral transfer benefits ToM predictions.

## Weaknesses

### Fatal
None.

### Major

- **Insufficient engagement with the contrastive decoding literature**: The core mechanism (Equation 6: π̄(a) ∝ π^L(a) · π^E(a)/π^N(a)) is mathematically equivalent to contrastive decoding / probability-ratio output adjustment, where the ratio between a fine-tuned and base model's distributions is used to redirect a target model. While the paper cites general "reweighting techniques" in related work (Section 4, lines 417–420), it does not explicitly discuss or compare with the contrastive decoding literature (e.g., Li et al., 2023; others). This is significant because it bears on the novelty of the methodological contribution: the technique itself is known, and the paper's contribution is primarily the application to Bayesian ToM inference. Acknowledging this connection and comparing against standard contrastive decoding baselines would strengthen the paper and clarify its contribution boundary. This matters because without this comparison, readers cannot assess whether the "ToM behavior transfer" framing provides value beyond what contrastive decoding already offers.

- **Questionable 70B post-training baselines for Llama-3 and Llama-3.1**: In Table 2, the 70B-post-trained models for Llama-3 (73.05%) and Llama-3.1 (71.86%) underperform their respective 8B-post-trained models (73.33% and 75.71%). This inversion—where a properly fine-tuned larger model underperforms a smaller one—is highly unusual and strongly suggests the 70B LoRA hyperparameters (rank 8, alpha 16, described as "adjusted" from the 8B settings) were not adequately optimized. For Llama-2, where the 70B baseline appears well-tuned (76.43% > 72.71%), the weak-to-strong method still wins (78.05% vs. 76.43%), providing a fair comparison that supports the paper's claims. However, the paper draws conclusions about the Llama-3 and Llama-3.1 70B baselines (Section 3.3: "direct post-training on the 70B model...does not produce results as robust as our method") based on what appear to be suboptimal baselines. The authors attribute this to "the difficulty of finding optimal hyperparameters for larger models," which is a reasonable practical observation but does not make the comparison scientifically fair. A properly tuned 70B baseline for at least one of these model families is needed to establish the method's advantage under fair conditions.

### Minor

- **Evaluation limited to a single benchmark for main results**: The primary comparison in Table 1 uses only MMToM-QA, a benchmark introduced by the prior work (BIPALM). While the transfer experiments (Table 4) extend to unseen scenarios, they share the same evaluation protocol and question types. Evaluating on additional ToM benchmarks (e.g., ToMi, BigToM) would strengthen generalizability claims.

- **No failure case analysis**: The paper does not analyze when weak-to-strong control *hurts* performance or produces incorrect redirections. The ratio π^E/π^N could be counterproductive when the large model's pre-training already captures relevant knowledge in a different distributional form. Identifying failure modes would strengthen the paper's practical guidance.

### Trivial
None.

## Nice-to-Haves

- Comparison with standard contrastive decoding applied to the same Bayesian framework (e.g., using an untrained small model as the "amateur" and a post-trained small model as the "expert" to modify the large model's logits directly), which would isolate the contribution of the Bayesian ToM integration versus the ratio mechanism itself.

- Statistical significance tests or confidence intervals for the improvement margins, particularly on the transfer experiments where sample sizes per scenario may be small.

- Per-example qualitative analysis showing specific cases where weak-to-strong redirection changes the inferred mental state correctly vs. incorrectly, complementing the averaged analysis in Figure 2.

## Removed Points

*These points were flagged for removal and should be treated with caution:*

- **Claim that the 4.6% improvement is overstated and the real improvement is only 1.3%**: The harsh critic confused the 2.4 subtask score (80.0%) for BIPALM w/ Llama-2-7B with the overall "all" score. The actual "all" score is approximately 76.7% (weighted by the 600 questions: 300 belief questions equally split across 3 subtasks, 300 goal questions equally split across 4 subtasks). The 4.6% improvement claim (81.3% − 76.7% = 4.6%) is correct.

- **Criticism about missing appendix/proofs**: The parser strips appendix sections; Theorem 1 and its proof exist in the original submission.

- **Formatting and presentation nitpicks**: These are parser artifacts, not author errors.

- **Reproducibility concerns about undisclosed hyperparameters**: The paper specifies LoRA configurations, learning rates, and epochs (Section 3.1). Implementation details for large-scale experiments are typically impractical to include in full.

- **Demand for user studies**: Not standard for an algorithmic/neurosymbolic methods paper in this venue.

- **Criticism of comparisons with GPT-4V and other non-adapted models**: The paper compares against these models to provide context about general multimodal ToM capabilities, not to claim superiority over adapted methods. This is standard practice for benchmark papers.

## Novel Insights

The paper reveals an interesting asymmetry in how post-training behavioral shifts transfer across model scales: the ratio π^E/π^N from a small model captures *directional* shifts (e.g., toward item-level concepts) that are productively applied to large models, even though the large models have very different base distributions. This is consistent with the contrastive decoding insight that probability ratios capture "what fine-tuning adds" in a way that is somewhat independent of the base model's distribution. However, the paper misses the opportunity to explicitly connect this finding to the broader contrastive decoding literature, which has studied exactly this phenomenon in other contexts.

## Suggestions

- Run a contrastive decoding baseline: apply standard contrastive decoding (logit adjustment using the π^E/π^N ratio) to the same Bayesian framework, without the "weak-to-strong" framing. This would clarify whether the Bayesian integration adds value beyond the known technique.

- Conduct a proper hyperparameter search for at least one 70B model family (e.g., Llama-3.1 70B) to establish a fair baseline, or explicitly reframe the contribution as a cost-efficiency method that avoids expensive tuning of large models.

- Add evaluation on at least one additional ToM benchmark to strengthen generalizability claims beyond the MMToM-QA benchmark.

## Evaluation

**Originality**: The application of probability-ratio transfer to Bayesian ToM inference is novel in its specific context, but the core mechanism (Equation 6) is equivalent to contrastive decoding, which is not acknowledged. The neurosymbolic integration is a genuine contribution, but the methodological novelty is narrower than claimed.

**Importance of research question**: The scalability of Bayesian ToM methods is a real and important problem. The paper addresses a meaningful gap between small models' task-specific adaptation and large models' world knowledge.

**Claims support**: The main claim (4.6% improvement over SOTA) is well-supported by the data. The secondary claim (weak-to-strong outperforms direct 70B post-training) is partially supported—valid for Llama-2 but questionable for Llama-3/3.1 due to undertuned baselines.

**Experimental soundness**: Generally sound, with systematic scaling analysis and transfer experiments. The single-benchmark limitation and lack of contrastive decoding baselines are gaps.

**Clarity**: The paper is well-structured with clear formal development (Equations 1–6). The table formatting makes the "all" scores difficult to verify directly, which contributed to the reviewer confusion about the 4.6% figure.

**Community value**: The practical insight that post-training behavioral shifts from small models can effectively guide large models at test time is valuable for the neurosymbolic reasoning community.

## Calibration

I compared this paper against the following anchors:

1. **BIRD (avg 8.0, Oral)** — Bayesian inference framework aligning Bayesian networks with LLM abductions, 30% improvement over baselines. Our paper is weaker: smaller improvement margin, less engagement with related decoding techniques, single benchmark.

2. **Weak-to-Strong Preference Optimization (avg 7.25, Spotlight)** — Similar weak-to-strong concept using distribution ratios for alignment. Our paper has comparable mechanism novelty but more extensive evaluation (scaling analysis, transfer experiments). WSPO had larger practical impact on benchmarks.

3. **Weak-to-Strong Generalization Framework (avg 6.5, Poster)** — Theoretical framework with missing baselines. Our paper has a similar profile: novel application with some experimental gaps.

4. **Weak-to-Strong Jailbreaking (avg 5.4, Reject)** — Uses the same probability ratio mechanism for adversarial purposes. Our paper is stronger: more principled motivation (Bayesian framework), more extensive evaluation, positive rather than adversarial application.

5. **Prior-Aware Decoding (avg 5.5, Reject)** — Contrastive decoding method with overclaimed improvements on synthetic tasks. Our paper is stronger: legitimate improvement, real-world benchmark, better experimental design.

6. **Low-scoring anchors (avg 1-2)** — Fundamentally flawed papers. Our paper is far above these.

The paper sits between the medium-scoring decoding papers (5.4-5.5) and the medium-high Bayesian/neurosymbolic papers (6.5-7.25). Its genuine 4.6% improvement, transfer experiments, and systematic scaling analysis push it above the decoding papers, but the lack of contrastive decoding engagement and the questionable 70B baselines prevent it from reaching the 7+ range. The Llama-2 fair comparison partially salvages the baseline concern.

**Score: 6.0** — The paper makes a real contribution to scaling Bayesian ToM inference, with a legitimate 4.6% improvement and valuable practical insights. However, the incomplete engagement with the contrastive decoding literature and the questionable 70B post-training baselines for two of three model families are significant concerns that a rebuttal should address.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>