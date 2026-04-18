Now let me write the consolidated final review, carefully checking each claimed weakness against the paper itself.

## Summary

This paper reformulates the Reward Augmented Decoding (RAD) approach to controlled language generation as a reward matrix completion problem. The authors observe that RAD's learned reward matrices are empirically low-rank despite RAD having the capacity to represent high-rank structures, which means RAD's expensive per-token forward passes are unnecessary overhead. Motivated by this, they propose ARM (Autoregressive Reward Model), a low-rank parametrization that decomposes reward scores into a prefix baseline plus a bilinear interaction between context representations and token embeddings. ARM requires only a single forward pass per decoding step (versus k for RAD with k candidate tokens), achieving competitive control–fluency trade-offs on detoxification and sentiment tasks while being substantially faster.

## Strengths

- **Insightful conceptual reframing.** Recasting RAD's training objective as minimizing $\|P_\Omega(R) - P_\Omega(\hat{R})\|_F^2$ (Section 3.1.1) and empirically measuring the rank of RAD's learned reward matrix (Figure 1) is a genuine and non-obvious contribution. The observation that RAD produces low-rank reward matrices despite having high-rank capacity provides a principled motivation for ARM that goes beyond a simple engineering trick. The connection to dueling networks and Q-function style parametrization (Section 4) is also well-identified.

- **Significant practical efficiency improvement.** The reduction from O(k) forward passes per decoding step to a single forward pass is clearly substantiated. Table 1 cleanly summarizes the comparison with GeDi, DExperts, and RAD. Figure 6 provides concrete wall-clock measurements showing ARM is ~5× faster than RAD at k=40 and ~10× faster at k=80, with near-constant time for ARM.

- **Competitive empirical quality.** ARM matches RAD on detoxification (Figure 3) and slightly outperforms RAD on sentiment control (Figure 4) in the distillation setting. The ablation in Figure 5 demonstrating that regularization reduces rank and improves fluency provides useful internal validation of the matrix factorization perspective.

- **Multiple training regimes evaluated.** Both distillation from a RAD teacher and direct training on responses are tested, with honest reporting that distillation works slightly better and a plausible explanation: "the teacher already performs this compression and provides a single deterministic target" (Section 5.4).

## Weaknesses

### Fatal
None.

### Major

- **Missing rank ablation directly testing the core claim.** The paper's central argument is that RAD's extra rank flexibility is unnecessary and can be traded for efficiency. However, no experiment explicitly varies the rank of ARM (e.g., using W = UV^T with rank r ≪ d) to establish where quality degrades. The paper shows rank of RAD is ~10² (Figure 1) and the full ARM rank is d=768, but never tests whether an even lower-rank ARM would also suffice, nor where the quality–rank trade-off actually bites. This is a surprising gap for a paper whose thesis centers on "low-rank is enough" — a rank sweep experiment would directly verify or falsify this claim and substantially strengthen the contribution.

- **Limited task generality for a claim about architectural expressivity.** Both detoxification and sentiment control are single-scalar, coarse-grained attributes known to correlate with shallow lexical features. The paper's framing — "RAD's rank expressiveness can be traded for efficiency without quality drop" — generalizes beyond these tasks, but the evidence does not. Multi-attribute or fine-grained control tasks (e.g., style + topic + safety jointly, or attribute-specific token interactions) would stress-test whether higher-rank reward matrices are truly unnecessary. The Limitations section acknowledges this ("further qualitative research is needed to investigate whether certain toxicity patterns require high rank"), but the main text's conclusions reach further than the experiments support.

### Minor

- **Distillation setting confounds architectural comparison with training signal quality.** The best ARM results come from distillation (eq. 10) where a trained RAD teacher provides compressed, deterministic targets. When ARM is trained directly on raw responses ("ARM resp. only"), it lags slightly behind. This means the most favorable ARM results partially benefit from RAD's label compression rather than purely from ARM's architectural efficiency. The "resp. only" variant provides a fairer architectural comparison, but the paper foregrounds the distillation results. The honest Section 5.4 discussion partially addresses this, noting the teacher "already performs this compression."

- **Rank analysis relies on incomplete data with no sensitivity analysis.** The rank estimates in Figure 1 use SVD with a standard singular value cutoff on sampled rows (Appendix C.4), but there is no sensitivity analysis to the cutoff threshold, no reporting of singular value spectra, and no extrapolation beyond 4k contexts. The gap between "the learned matrix is empirically low-rank" and "low-rank is sufficient for generalization to unseen contexts" is acknowledged (Section 3.1.3) but not empirically closed. The rank-1 existence argument (Appendix B.1) for singly-observed prefixes, while correct, is almost vacuous for generalization claims.

- **No variance or statistical significance reported.** The trade-off plots in Figures 3 and 4 show point estimates without error bars or confidence intervals across random seeds. It is impossible to assess whether the slight ARM advantage on sentiment or the slight ARM disadvantage on detoxification (resp. only) are meaningful.

### Trivial

- **Inconsistency in model dimension.** The main text states d=768 while Figure 1's caption states d=764. This is likely a typo (GPT-2 Small has d=768).

## Nice-to-Haves

- Experiments on multi-attribute or compositional control tasks to test the generalizability of the low-rank finding.
- Full end-to-end latency and total training cost analysis (including the cost of training the RAD teacher for distillation).
- Comparison with ARM trained on responses with matched hyperparameters to better isolate architectural from training-signal effects.
- A direct rank sweep (rank 1, 2, 10, 50, d) to find the minimal sufficient rank and connect the theoretical analysis to empirical performance.
- Human or LLM-based evaluation of generation quality beyond perplexity and MAUVE.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Distillation requires an expensive RAD teacher, undermining the efficiency narrative" (Neutral Reviewer, Spark):** The paper explicitly evaluates ARM trained on responses without distillation (ARM resp. only), which requires no RAD teacher. The distillation setting is an additional option, not a requirement. The efficiency claim is about inference, not training.

- **"No comparison against a smaller/faster RAD baseline" (Spark):** This is a category error. ARM uses the same backbone as RAD with a different head. Reducing RAD's backbone size would change the model capacity, not just the per-step efficiency. The contribution is about the architectural parametrization, not about model scaling.

- **"No qualitative examples of generated text" (Spark):** While qualitative examples would be helpful, the paper uses standard metrics (Perspective API, DistilBERT, MAUVE, perplexity) that are well-established in this literature. This is a nice-to-have, not a substantive weakness.

- **"Outdated backbone models" (Human Finder):** The paper does experiment with LLaMa-2 7B/13B in addition to GPT-2 (in appendix). The GPT-2 experiments are standard for this line of work and facilitate direct comparison with prior baselines (GeDi, DExperts, RAD).

- **"β parameter lacks principled selection" (Human Finder):** This is standard in the controlled generation literature; sweeping β is the norm for trade-off plots. Every method in this space (GeDi, DExperts, RAD) uses the same approach.

- **"Evaluation relies on single classifiers with known idiosyncrasies" (Harsh Critic):** The paper uses Perspective API and DistilBERT, which are the standard evaluation tools for detoxification and sentiment control in this literature. An alternative open-weight toxicity classifier is included in Appendix F.3.1 with consistent results. This is the community norm for this benchmark suite.

## Novel Insights

The matrix completion reformulation of RAD's training objective is genuinely insightful and connects reward model design to a well-studied mathematical framework. The empirical finding that RAD's learned reward matrices have rank ~10² (far below both the vocabulary size and model dimension) directly motivates the ARM parametrization and provides a concrete answer to the question "is RAD's per-candidate expressivity actually needed?" The regularization analysis (Figure 5) connecting explicit regularization to both reduced rank and improved fluency further enriches this story, suggesting that pushing toward low-rank structure is not just compatible with good performance but may actively benefit it.

## Suggestions

1. **Run explicit rank ablation experiments** — Add ARM variants with inner dimensions r ∈ {1, 2, 4, 16, 64, 256} (i.e., W = UV^T with rank r) to directly determine the minimal sufficient rank and test whether even lower-rank models succeed. This is the single most impactful experiment for strengthening the core claim.

2. **Tone down the generality claims** — In the abstract and conclusion, replace language like "RAD's rank expressiveness can be traded for efficiency without quality drop" with "for detoxification and sentiment control tasks, low-rank parametrization suffices" or similar, making the scope limitation clear in the claims themselves, not just the Limitations section.

3. **Report variance across runs** and, if possible, statistical significance for the trade-off curve comparisons between ARM and RAD.

## Score and Decision

**Calibration:** I compared against several controlled generation and efficient decoding papers:
- Language Model Arithmetic (ICLR spotlight, human scores ~7-8): broader scope, similar tasks, stronger generality
- SASA Self-Detoxification (ICLR poster, human scores ~6): similar task scope, comparable novelty
- SF-GEN (withdrawn, scores ~6): similar idea (efficient controlled decoding), similar evaluation breadth
- Collab/Mixture-of-Agents (ICLR poster, scores 5-8): similar efficiency-focus, broader evaluation
- DAB Discrete Autoregressive Biasing (ICLR poster, scores 3-8): similar tasks, similar methodological profile

This paper has a stronger conceptual contribution (matrix completion + rank analysis) than many comparable papers, and clear practical gains. Its main weakness is limited task breadth and the missing rank ablation. It falls in the range of solid poster-level contributions: above average novelty for the venue, clear empirical support but with some untested generalizability. Compared to the 6-7 range for similar papers, this lands at a 6.5 — the conceptual insight and efficiency contribution are real, but the missing rank ablation is a meaningful gap in a paper whose thesis centers on "low-rank suffices."

MY FINAL SCORE: <pineapple>6.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>