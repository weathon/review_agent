## Summary

This paper argues that autoregressive (AR) language models are structurally disadvantaged on complex reasoning and planning tasks due to "subgoal imbalance"—certain sequential dependencies are inherently harder to learn with left-to-right conditioning. The authors propose Multi-Granularity Diffusion Modeling (MGDM), which extends discrete diffusion with token-level focal-loss-style reweighting to prioritize difficult subgoals during training. On three constraint-satisfaction benchmarks (Countdown, Sudoku, 3-SAT), MGDM achieves striking improvements over autoregressive counterparts (e.g., 91.5% vs. 45.8% on Countdown-4, 100% vs. 20.7% on Sudoku) using far fewer parameters.

---

## Strengths

- **Empirically compelling parameter-efficiency gap**: A 6M MGDM outperforms fine-tuned 13B LLaMA on Sudoku (100% vs. ~35%), and an 85M MGDM outperforms all AR baselines on Countdown-4 by a margin that does not close even with AR scaling to 13B. These are not small differences and are difficult to dismiss.

- **The "Regretful Compromise" error analysis is a genuine and specific contribution**: By decomposing Countdown-4 errors into planning errors (wrong number choices) and calculation errors (arithmetic mistakes) at each reasoning step, the paper produces a mechanistic explanation of *how* AR fails—early planning errors cascade into systematically incorrect final equations. This is a concrete, task-specific diagnostic, not a generic claim.

- **Clean synthetic ablation isolating the planning distance variable (Figure 2)**: The synthetic graph planning task is carefully controlled, with symmetry enforced to prevent reverse-AR shortcuts. The exponential data scaling requirement for AR as planning distance increases, contrasted with diffusion's consistent performance, provides a principled empirical grounding for the subgoal imbalance hypothesis.

- **Decoding speed flexibility is practically meaningful**: The observation that single-step MGDM achieves 75% accuracy on CD4 (vs. 45.8% for AR) at 10× the throughput is a genuine practical advantage for constraint-satisfaction settings where latency matters.

---

## Weaknesses

- **Critical missing baseline: bidirectional masked language models (MLM/BERT-style).** The paper's core theoretical claim is that diffusion's multi-view training helps with hard subgoals. However, a BERT-style masked language model is also bidirectional and conditions on full context at training time. Without comparing against such a baseline (e.g., CMLM, iterative masked decoding), it is impossible to determine whether MGDM's advantage stems from (a) bidirectionality in general, (b) the iterative multi-step denoising process, or (c) the specific token-level reweighting. This is not a minor gap—it is central to the paper's theoretical contribution and is especially important for Sudoku, where the bidirectional constraint structure is inherently suited to any non-AR model.

- **Unexplained and undiscussed scaling anomaly**: The 303M MGDM (88.3% on CD4, 39.0% on CD5) substantially *underperforms* the 85M MGDM (91.5%, 46.6%). This regression is present in Table 1 and is never addressed in the text. If MGDM does not scale monotonically with parameters, this undermines the claim that diffusion is a better paradigm at scale and raises questions about training stability or optimization sensitivity.

- **Hyperparameter sensitivity in MGDM**: Table 3 shows that the performance of MGDM varies substantially with reweighting parameters: linear reweighting with $\alpha=0.25, \beta=2$ achieves 91.5%, while $\alpha=0.25, \beta=1$ gives only 88.0%—barely above the no-reweighting baseline (87.0% with TopK). The paper does not provide principled guidance for selecting $\alpha$ and $\beta$, and the reported best configuration appears to have been selected by grid search. This reduces practical usability and transparency.

- **Proposition 1 is too informal for its presentation**: The paper labels a qualitative statement about subgoal difficulty as "Proposition 1," but provides no proof, no quantitative bound, and no non-trivial condition on the data distribution. It is a motivating intuition rather than a formal result. Presenting it as a proposition overstates the theoretical contribution. The subsequent multi-view learning analogy (Xu et al., 2013) is also asserted rather than formally established: diffusion timesteps form a Markov chain, not the independent views required by formal multi-view learning theory.

- **SAT instances are extremely small**: With only $n \in \{5, 7, 9\}$ variables, the near-threshold 3-SAT instances have a maximum search space of $2^9 = 512$ assignments. The asymptotic hardness characterization ($m \approx 4.258n$) is derived for $n \to \infty$ and may not accurately characterize difficulty at such small $n$. The performance gap, while real, is modest at these scales (both models exceed 85% at $n=9$).

- **The "OOD" evaluation for Countdown is weak**: The 10% held-out test targets are randomly sampled from the same distribution (10–100). This does not constitute meaningful out-of-distribution evaluation; it is standard held-out evaluation. The paper should either not use the term "OOD" or test genuinely out-of-distribution targets (e.g., targets > 100).

- **Figure 3 shows training loss, not generalization loss**: The key evidence that diffusion "effectively learns hard subgoals" (Figure 3) is measured on training loss for a specific example. Whether this loss reduction reflects genuine learning or bidirectional context availability at training time is not disentangled. Testing this on held-out examples would strengthen the claim.

---

## Nice-to-Haves

- **Compute-normalized training comparison**: Diffusion requires multiple forward passes per training sample. The paper compares parameter counts but not total training FLOPs. Including a FLOP-normalized comparison would make the efficiency claims more rigorous.

- **AR + Search baselines on Sudoku and SAT**: The paper includes Tree-of-Thought comparison only for Game of 24. Adding AR + ToT or constraint propagation baselines for Sudoku would clarify whether the gap persists against search-augmented AR.

- **Larger SAT instances**: Extending to $n \geq 15$ would provide a more convincing demonstration that MGDM's advantage holds at meaningful problem scales where the hardness characterization is more reliable.

- **Denoising trajectory visualization for Sudoku**: Showing intermediate diffusion states would empirically validate whether global constraints are satisfied progressively, directly supporting the "planning" narrative.

- **Quantification of subgoal difficulty in real tasks**: The paper validates the subgoal imbalance concept on a synthetic graph. Providing analogous difficulty measurements for tokens in Countdown/Sudoku solutions would strengthen the link between the theoretical framing and the empirical results.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Stream-of-Search comparison being "unfair"**: The harsh critic argues that SoS uses augmented (richer) training data while MGDM uses direct solutions. However, this asymmetry *favors the baseline* (SoS)—if MGDM still wins despite training on less supervision, this makes a *stronger* point for MGDM, not a weaker one. This is an intentionally asymmetric comparison in the paper's favor that demonstrates a stronger claim.

- **GPT-4 vs 85M MGDM comparison being methodologically questionable**: The paper uses this comparison to illustrate token-cost efficiency, not to claim that MGDM is a "better model" than GPT-4. The framing is clearly about computational cost, and the comparison is explicitly contextualized. This is not a methodological flaw.

- **Figure 2 figure description appearing to show AR reaching 100% accuracy**: This appears to be an OCR/parsing artifact in the submitted document. The text unambiguously states diffusion achieves perfect accuracy across all PDs; the figure description caption is likely mis-parsed.

- **Sudoku Figure 4 showing ~40% for 6M MGDM vs. 100% claimed in text**: This discrepancy is also likely an OCR/image-parsing artifact from the PDF-to-text conversion. The main text and abstract consistently state 100%.

- **Criticism that cited references or benchmarks do not exist**: Per review policy, if the paper cites a reference, it is assumed to exist.

---

## Novel Insights

The most genuinely novel analytical contribution of this paper is the "Regretful Compromise" phenomenon: AR models do not simply fail to find solutions—they commit to wrong intermediate numbers early, then make *intentionally incorrect arithmetic* in the final step in order to reach the target, producing solutions with high calculation error rates in the last equation (48.9%) but low planning error in earlier steps. This provides a concrete, task-level mechanism for why AR planning fails that goes beyond the generic claim of "limited long-range dependencies." The multi-view learning reinterpretation of the diffusion ELBO—showing that each noisy variant $\mathbf{x}_t$ provides a more tractable learning signal for hard subgoals—is also insightful as an explanatory lens, even if not formalized rigorously.

---

## Suggestions

1. **Add a BERT/CMLM baseline**: Train a masked language model on the same data with the same architecture and compare directly on all three benchmarks. This is the single most important experiment to support the paper's theoretical claims about *why* diffusion helps.

2. **Discuss and investigate the 303M scaling anomaly**: Either provide an explanation (e.g., optimization instability at scale, capacity overfitting) or frame it explicitly as a known limitation requiring further investigation.

3. **Provide principled guidance for $\alpha, \beta$ selection**: Report a held-out validation procedure and make the hyperparameter selection process transparent. A sensitivity analysis plot would strengthen confidence in the method.

4. **Rename or downgrade Proposition 1**: Present it as a "motivation" or "observation" rather than a formal proposition, to avoid overstating the theoretical contribution.

5. **Replace the term "OOD evaluation" with "held-out evaluation"** unless genuinely out-of-distribution targets are tested.

---

**Evaluation Summary:**

- *Novelty*: Moderate-to-high. The subgoal imbalance framing and Regretful Compromise analysis are original contributions; MGDM itself (focal-loss reweighting + borrowed TopK decoding) is incremental over existing discrete diffusion.
- *Technical soundness*: Mixed. The empirical methodology is generally sound; the theoretical framing is informal and the missing bidirectional baseline is a significant gap.
- *Empirical support*: Strong for the specific tasks studied. The numerical gaps are large and consistent across multiple baselines.
- *Significance*: Meaningful for the community studying non-autoregressive generation and planning, though limited to constraint-satisfaction domains.
- *Clarity*: Generally clear, with some ambiguity in Figure 4 (OCR issues) and insufficient discussion of the scaling anomaly.

MY FINAL SCORE: <pineapple>6.2</pineapple>