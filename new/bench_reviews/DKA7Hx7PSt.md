## Summary

The paper proposes Learning Embedding Linear Projections (LELP), a knowledge distillation method for few-class and binary classification. LELP extracts pseudo-subclasses from a frozen teacher's final-layer embeddings via per-class PCA (augmented with null-space projection and random rotation), then trains the student to predict these pseudo-subclasses using a single KL-divergence loss. The method aims to match the efficacy of Subclass Distillation without the costly requirement of retraining the teacher, and evaluates on both synthetic (binarized CIFAR) and real-world NLP benchmarks.

## Strengths

- **Practical advantage over Subclass Distillation.** The paper correctly identifies that Subclass Distillation requires iterative teacher retraining with varying hyperparameters, which is computationally prohibitive for large models (Section 2). LELP eliminates this bottleneck while remaining competitive, a genuine contribution for deployment scenarios.
- **Strong NLP benchmark results.** Table 2 shows that LELP frequently outperforms Vanilla KD and other embedding-matching baselines on few-class language tasks (e.g., Amazon Reviews 5-class: 78.06% vs. Vanilla KD 75.13%; Sentiment-60 Bin: 87.60% vs. Vanilla KD 85.95%). Notably, the ALBERT-Base student sometimes exceeds the much larger ALBERT-XXL teacher.
- **Systematic comparison of clustering strategies.** Table 1 and Figure 3 convincingly demonstrate that LELP's linear projection approach outperforms agglomerative clustering, K-means, and t-SNE+K-means on binarized CIFAR tasks, establishing that pseudo-subclass quality is highly dependent on the extraction method.
- **Implementation simplicity.** LELP requires only a single classification loss and a modest expansion of the student's final output layer, avoiding the multi-term loss balancing required by methods like FitNet and Embedding Distillation.

## Weaknesses

### Fatal
None.

### Major

- **Critical numerical inconsistencies and missing cross-architecture results.** Figure 1 and the abstract highlight specific results on Amazon Reviews (~78.5%, +1.85% over best baseline) and Sentiment140 (~82.5%, +0.88%). These values do not match Table 2, which reports 78.06% for Amazon 5-class and 87.60% for "Sentiment-60 Bin" (a subset whose relationship to Sentiment140 is unexplained), with summary rows claiming inexplicable margins of +0.05. Furthermore, Section 4.1 describes cross-architecture distillation to a lightweight MLP over a frozen Sentence-T5 encoder as a key versatility test, yet these results are entirely absent from Table 2 and the main text. Because the paper's headline claims and architecture-flexibility assertions cannot be verified from the material provided, the empirical foundation is seriously undermined.

- **Confounded comparison with Subclass Distillation.** In every setting in Table 2, the "Subclass Distillation Teacher" achieves different accuracy than the standard "Teacher" (e.g., 78.02/78.45 vs. 77.58 on Amazon 5-class). The authors themselves note that this makes direct comparison "not entirely fair" (Section 4.1, Baselines paragraph). Yet the paper repeatedly claims LELP surpasses Subclass Distillation in absolute accuracy. While avoiding teacher retraining is a clear practical win, the claim of empirical superiority is confounded by teacher-quality differences that the authors do not control for or disentangle.

### Minor

- **Core design choices lack main-text ablations.** The paper introduces null-space projection and random rotation as critical steps (Section 3.1) but defers all ablations to Appendix C. Without even a condensed main-text ablation, the reader cannot assess how much of LELP's gain derives from its pseudo-subclass formulation versus these heuristic preprocessing steps.
- **Student-teacher reversal is presented but not analyzed.** The student outperforms the 20× larger teacher on Amazon Reviews (Table 2). This is unusual and could signal important phenomena (e.g., beneficial regularization), but the paper offers no analysis or discussion.
- **Headline CIFAR experiments are synthetic proof-of-concept.** Binarizing CIFAR-10/100 guarantees subclass structure (the original fine-grained labels). The paper correctly positions these as motivation, but they do not establish that LELP discovers meaningful pseudo-subclass structure in real-world few-class tasks where such structure is unknown or absent. The NLP experiments partially address this, but the synthetic setting still dominates the methodological development.

### Trivial

- Table 2 contains ambiguous duplicate column labels (e.g., two "Am. Reviews Bin" and two "Am. Reviews" columns with differing results) without clear headers distinguishing them, making the table hard to parse.

## Nice-to-Haves

- Include the MLP-over-Sentence-T5 cross-architecture results in the main text, as they are a key claim in the abstract.
- Move a condensed version of the Appendix C ablations into the main paper.
- Statistical significance tests or additional seeds for the NLP experiments to strengthen confidence in the reported margins.
- Brief discussion or analysis of why the student exceeds the teacher on Amazon Reviews.

## Removed Points

These points are flagged to be removed, treat them with caution.
- **Dismissal of DKD without citation/proof:** The paper states DKD is functionally identical to Vanilla KD in binary classification and demonstrates this empirically (identical values in Table 2). A formal proof would be nice but is not required in an empirical systems paper.
- **Setting α = 0 disadvantages baselines:** The paper explicitly justifies this choice as necessary to isolate the distillation loss and relevant to the semi-supervised setting (Section 4.1). This is a deliberate and fair experimental design.
- **Request for random-projection baseline:** The paper already compares PCA vs. Random Projections in Appendix C ablations.
- **Lack of qualitative inspection of projections:** While interesting, nearest-neighbor analysis of projected embeddings is beyond the scope of the core empirical validation.
- **Statistical tests with only 3 trials:** Three seeded runs are standard practice for large-scale NLP experiments; overlapping standard deviations are noted but do not invalidate the core findings.
- **Request to evaluate on standard few-class vision benchmarks without synthetic subclass structure:** The paper explicitly scopes its contribution to few-class and binary settings and includes real-world NLP experiments (Section 4.3.2).
- **Claims about DKD, CRD, and vision-specific methods:** The paper's dismissal of vision-specific augmentation-dependent methods for NLP is well-reasoned and supported.

## Novel Insights

The paper's most compelling insight is practical rather than theoretical: by combining per-class PCA with null-space projection and random rotation, LELP converts the expensive subclass-discovery problem into a lightweight, closed-form preprocessing step that requires no teacher retraining. This reframes the debate from "how do we retrain the teacher to discover subclasses?" to "how do we read out existing structure from the frozen teacher?" The Oracle Clustering experiments (Section 4.2) provide a useful upper-bound motivation, suggesting that even idealized pseudo-subclass knowledge can substantially outperform vanilla distillation—and that linear projections are an unexpectedly effective proxy for this ideal.

## Suggestions

1. **Reconcile or clearly separate the results in Figure 1 and Table 2.** If Figure 1 shows the MLP-over-T5 student, label it explicitly; if it shows a different Sentiment140 subset, explain the discrepancy. Ensure that numerical claims in the abstract match the data in the main table.
2. **Control for the Subclass Distillation teacher difference.** Either report a version of Subclass Distillation that uses the same frozen teacher (even if suboptimal for that method), or reframe the claim to emphasize practical parity without retraining rather than absolute superiority.
3. **Include at least one main-text ablation** showing the contribution of null-space projection and random rotation, so readers can judge whether gains come from the pseudo-subclass formulation or the preprocessing heuristics.

<context>
- **Original reviewer signal:** The Harsh Critic argued the paper is structurally flawed due to confounded comparisons, missing/inconsistent headline results, and absent cross-architecture data, recommending rejection. The Strength Finder viewed the paper as a solid contribution with strong NLP results and practical advantages over Subclass Distillation.

- **What was dropped and why:** 
  - DKD/CRD dismissal criticism: removed because the paper provides empirical demonstration and brief justification.
  - "α = 0 disadvantages baselines": removed because the paper explicitly and reasonably justifies this choice.
  - "Random rotation makes PCA meaningless" and related methodological nitpicks: weakened to minor/nice-to-have because the paper cites Appendix C ablations.
  - Strength Finder claim #5 (cross-architecture results in Table 2): dropped because the claim is factually wrong—Table 2 contains no cross-architecture results. The raw text says "Our results are shown in Table 2" but the MLP-over-T5 results are missing from the table and main text entirely.
  - Various formatting, grammar, and appendix-deferred proof complaints: removed as parser artifacts or standard practice.

- **Cross-checks performed:** 
  - Verified Table 2 teacher values differ between "Teacher" and "Subclass Distillation Teacher" across all columns (e.g., 90.19 vs 90.05, 77.58 vs 78.02/78.45), confirming the confound.
  - Checked that Figure 1 reports ~78.5% for Amazon and ~82.5% for Sentiment140, while Table 2 reports 78.06% for Amazon 5-class and 87.60% for "Sentiment-60 Bin"—the Sentiment values are severely inconsistent.
  - Checked Table 2's summary row ("Avg. gain over the best baseline"): the reported +0.05 values do not match any arithmetic difference with the best baseline in the table above them, and directly contradict the abstract's claim of +1.85%/+0.88%.
  - Confirmed Section 4.1 describes MLP-over-Sentence-T5 experiments but Table 2 only lists ALBERT-Base students, and no other main-text table/figure contains these results.
  - Verified that Appendix C ablations are referenced in Section 3.1 for null-space projection and random rotation.

- **Severity read:** The surviving weaknesses are serious but not fatal. The numerical inconsistencies between the abstract/figure and Table 2 constitute a major evidential gap that threatens the paper's core empirical claims. The confounded Subclass Distillation comparison is mitigated by the paper's own admission of unfairness, but its repeated superiority claims are still overreached. These are major issues that require reconciliation in a rebuttal, but the underlying method is sound and the problem is well-motivated.

- **Anything else load-bearing:** The paper explicitly scopes itself to few-class settings and honestly notes LELP is not designed for many-class problems (Section 5), which is appropriate. Three-trial evaluation with standard deviations is standard for large-language-model distillation experiments in NLP. The parser has introduced some table-formatting artifacts, but the raw table values are readable enough to confirm the numerical discrepancies.
</context>