Now I have a thorough understanding of the paper and calibration anchors. Let me compose the final review.

## Summary

DeCoRe proposes a training-free decoding strategy that mitigates LLM hallucinations by contrasting the output distributions of a base model and a variant with masked retrieval heads (attention heads identified as responsible for information retrieval via the Needle-in-a-Haystack protocol). The contrastive weight α is dynamically set to the conditional entropy of the base model's next-token distribution, amplifying correction when the model is uncertain. Experiments across 8 datasets and 2 model sizes show consistent improvements on faithfulness tasks and some factuality tasks.

## Strengths

- **Novel application of mechanistic interpretability to decoding**: DeCoRe uses retrieval head masking (identified via Wu et al., 2024's NiTH-based approach) to create an "amateur" model for contrastive decoding. This is a principled bridge between mechanistic interpretability insights and practical hallucination mitigation. Figure 2 provides a concrete example where masking 10 retrieval heads in Llama3-8b-Instruct causes it to predict "Addie Horton" instead of the correct "Ozzie Smith" on NQ-Swap, validating that masking can induce hallucinations (Section 2.1, Figure 2).

- **Dynamic entropy-guided contrastive decoding**: Setting α = H(x_t) (Section 2.3, Eq. 8) provides a principled, automatic way to modulate contrastive strength. Figure 4 demonstrates that DeCoRe_entropy produces significantly lower length-normalized conditional entropy than Greedy and ITI across XSum and MuSiQue tasks, confirming the mechanism reduces uncertainty in long-generation settings as intended.

- **Consistent improvements on faithfulness tasks**: Table 1 shows DeCoRe_entropy improves over base Llama3-8b-Instruct on MemoTrap Micro Accuracy by +10.47% (64.40% → 74.87%), on NQ-Swap EM by +5.46% (60.62% → 66.08%), and on NQ-Open EM by +0.98% (69.68% → 70.66%). Table 5 shows DeCoRe_entropy achieves the highest overall aggregate score for both model sizes (8B: 48.72 vs. 46.29 baseline; 70B: 54.98 vs. 54.80 baseline).

- **Broader applicability than competing methods**: Unlike ITI (requires labeled training data) and CAD (only applicable to tasks with additional context), DeCoRe is training-free and works across both context-dependent and context-free tasks. Table 2 shows DeCoRe consistently improves EM across all positions of the gold document in the Lost-in-the-Middle setup, while ITI drops to 11.45% EM and CAD to 29.30% when gold is 9th for Llama3-8b-Instruct.

- **Comprehensive evaluation scope**: 8 datasets, 2 model sizes, 6 baselines, 3 DeCoRe variants, plus ablations on retrieval heads and LitM evaluation. The MuSiQue experiments with CoT (Table 4) test multi-hop reasoning, going beyond standard benchmarks.

## Weaknesses

### Fatal
None.

### Major

- **Core conceptual contribution not validated in the main text**: The paper's central claim is that masking *retrieval heads specifically* induces hallucinations and that contrasting against this hallucinating variant reduces hallucination. However, the critical ablation—comparing retrieval head masking against random head masking—appears only in Appendix G, referenced in a single sentence: "These findings can be combined with the results of masking random attention heads (Appendix G) further supporting our hypothesis" (line 311). Without demonstrating in the main text that masking retrieval heads is meaningfully different from masking any equally-sized random subset of attention heads, the paper cannot establish that its "retrieval head" framing is load-bearing rather than cosmetic. If random masking produces comparable contrastive effects, the method still works but the conceptual contribution collapses—DeCoRe becomes "contrastive decoding against a degraded model" rather than "decoding by contrasting retrieval heads." This ablation is essential because the paper's entire narrative and title depend on retrieval heads being special.

- **Dramatic variation with N raises robustness concerns**: Figure 3 shows that performance with DeCoRe_entropy varies dramatically with the number of masked retrieval heads (N), with Pearson correlations ranging from +0.98 (XSum) to −0.98 (TriviaQA). For faithfulness tasks like XSum and MemoTrap, more masked heads helps; for factual recall tasks like TriviaQA and PopQA, more masked heads hurts. This means the optimal N is task-dependent, and it is unclear from the main text whether a single N works reasonably across all tasks or whether the reported results reflect per-task tuning. While the implementation details (including N) are likely in Appendix K, the main text should explicitly address this: does a universal N exist, or must practitioners tune N per task?

### Minor

- **DeCoRe_entropy-lite conflates two distinct methods**: Section 3.3 describes DeCoRe_entropy-lite as using "a smaller LLM with the same vocabulary space as the masked LLM" (LLaMA3-70B-Instruct as base, LLaMA3-8B-Instruct as "masked" model). No retrieval heads are masked—this is Contrastive Decoding (Li et al., 2023) with entropy-weighted α, not a DeCoRe variant. While the description is technically accurate, labeling it as a DeCoRe variant inflates the apparent scope of the method. A clearer framing would distinguish it as a hybrid baseline (CD + dynamic α).

- **CAD outperforms DeCoRe on key faithfulness benchmarks where applicable**: For the 70B model (Table 1), CAD achieves MemoTrap Micro Accuracy of 83.89% vs. DeCoRe_entropy's 73.65% (a ~10pp gap), and XSum factKB of 66.64% vs. 65.49%. The paper acknowledges CAD's limited applicability and DeCoRe's broader scope, which is a fair argument, but the magnitude of CAD's advantage on tasks it was designed for is notable and somewhat undersold.

- **Negative correlations for factuality tasks complicate the narrative**: Figure 3 shows strong negative correlations between the number of masked retrieval heads and performance on TriviaQA (r=−0.98) and PopQA (r=−0.94). While the paper offers an "information transfer" hypothesis (Section 4), this remains speculative and untested. The fact that masking more retrieval heads actively *harms* factual recall—while the paper claims DeCoRe improves both faithfulness and factuality—suggests the mechanism operates differently across task types, and the unified "retrieval head" story may not fully account for this.

- **Some improvements are marginal**: Several improvements are within 1-2 percentage points (e.g., MuSiQue closed-book CoT: 20.15→20.60 EM for 70B; NQ-Open: 69.68→70.66 EM for 8B), which may be within evaluation noise without confidence intervals.

### Trivial
None.

## Nice-to-Haves

- Analysis of the distribution of α = H(x_t) values during typical generation runs, to clarify whether the entropy mechanism is doing substantial work or whether α is typically near zero.
- Side-by-side generation examples across multiple tasks showing specifically what tokens DeCoRe changes relative to greedy decoding, and whether those changes correspond to faithfulness improvements vs. generic confidence shifts.
- Explicit comparison with DoLa on why retrieval-head masking should be preferred over layer contrasting, given that Tables 1–4 often show the two methods performing similarly.

## Removed Points

These points are flagged to be removed; treat them with caution.

- **"Essentially temperature sharpening" claim (Harsh Critic #1)**: The characterization that DeCoRe is "essentially temperature sharpening with an adaptive mechanism" is incorrect. Equation 7 computes p(x_t) ∝ p_base · (p_base/p_masked)^α, which specifically amplifies tokens where p_base > p_masked and suppresses tokens where p_base < p_masked. This is fundamentally different from temperature sharpening p(x_t) ∝ p(x_t)^(1/T), which uniformly sharpens the entire distribution. The contrastive mechanism can redirect probability mass between specific tokens, not just sharpen.

- **N not specified for main experiments (Harsh Critic #2, partially)**: The specific value of N is likely specified in Appendix K (referenced in Section 3.2: "All implementation details are available in Appendix K"). Per the rules, missing appendix content should not be penalized since the parser strips those sections. However, the broader concern about N's task-dependent variation (shown in Figure 3) is kept as a Major weakness above.

- **Missing error bars (Harsh Critic, Missing Experiments #2)**: The paper references pairwise statistical significance analyses in Appendices H.1–H.3. While error bars in main tables would be preferable, demanding them when significance tests exist in appendices is a standard-reproducibility nitpick.

- **Abstract percentage claims misleading (Harsh Critic, Abstract)**: The claimed improvements (e.g., "XSum by 18.6%") correspond to absolute percentage-point differences on specific sub-metrics (factKB for XSum), which is standard reporting in this field. While cherry-picking the best sub-metric is mildly misleading, it doesn't misrepresent the direction or existence of improvements.

- **Demand for per-token analysis and generation examples (Harsh Critic, Deeper Analysis)**: These would strengthen the paper but go beyond what is standard for an empirical methods paper at this venue.

- **Sensitivity of head identification to Wu et al. (Harsh Critic, Section 2.1)**: While valid, this is a dependency on an established method, not a flaw in the current paper. The quality of head identification is an area for future work, not a reason to reject.

## Novel Insights

The most insightful observation emerging from the review is the tension between DeCoRe's faithfulness and factuality results. The positive correlations between N and performance on copy/induction tasks (XSum r=+0.92, MemoTrap r=+0.98) versus the negative correlations on factual recall tasks (TriviaQA r=−0.98, PopQA r=−0.94) suggest that retrieval heads serve dual functions—context extraction and information transfer—that are differentially affected by masking. This implies that a one-size-fits-all N is suboptimal, and that the ideal DeCoRe configuration may require task-adaptive head selection, not just task-adaptive N. This observation, if validated, could inform a more nuanced approach to mechanistic-interpretability-guided decoding.

## Suggestions

- Move the random head masking ablation from Appendix G to the main text (even as a condensed table or figure). This is the single most important change: without it, the paper cannot establish that "retrieval heads" in the title are doing real conceptual work.
- Explicitly state the value of N used for each main experiment, and report results with at least one universal N across all tasks to demonstrate robustness.
- Rename DeCoRe_entropy-lite to something that reflects its hybrid nature (e.g., "CD-entropy") to avoid conflating distinct methods.

## Score and Decision

**Calibration comparison:**

- **DoLa** (avg 7.25, Accept poster): Closest comparison—contrastive decoding for hallucination using internal model structure. DoLa had similar concerns about its mechanistic hypothesis not being convincingly supported (one reviewer gave a 5 for this reason). DeCoRe has comparable experimental scope but weaker mechanistic validation (the critical ablation is in the appendix, not main text). DeCoRe is below DoLa.

- **Instructive Decoding** (avg 7.50, Accept spotlight): Contrastive decoding with noisy instructions. Simpler method with incremental gains (1-2pp), but the mechanism is more transparent. DeCoRe has more comprehensive experiments. Roughly comparable, with DeCoRe slightly weaker due to the unvalidated mechanism.

- **Adaptive Contrastive Learning** (avg 6.00, Accept poster): Contrastive learning for hallucination with knowledge quadrant framing. Accepted despite limited experiment coverage. DeCoRe has stronger experiments but a similar-level conceptual contribution.

- **GACD** (avg 4.75, Reject): Gradient-based contrastive decoding for multimodal hallucinations. Weaker validation and more questionable claims. DeCoRe is clearly above this.

- **FS-GEN** (avg 5.25, Reject): Collaborative decoding analysis with poor presentation. DeCoRe is above this in quality.

- **EDU-RAG** (avg 2.33, Reject): Weak benchmark paper. DeCoRe is far above this.

DeCoRe sits between the medium and high anchors. It has stronger experiments and more consistent results than the rejected papers, but its core mechanism is not well-validated in the main text (the random masking ablation being appendix-only is a significant gap for a paper whose title and narrative depend on retrieval heads being special). This places it in the borderline range, somewhat below DoLa (7.25) where a similar weakness was noted but the ablation was at least partially in the main text.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>