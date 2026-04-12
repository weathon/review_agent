## Summary
This paper presents **U-Bench**, a large-scale benchmark of **100 U-shaped medical image segmentation models** across **28 datasets and 10 modalities**, with analyses along in-domain accuracy, zero-shot transfer, efficiency, and architectural/data characteristics. Beyond collecting results, it introduces an efficiency-aware metric (**U-Score**) and a ranking-based advisor agent intended to help practitioners choose architectures under dataset and resource constraints.

## Strengths
- **The paper contributes a genuinely broad and unusually structured benchmark for U-shaped segmentation models.** It spans 100 variants across CNN, Transformer, Mamba, RWKV, and hybrid families, and evaluates them on 28 datasets / 10 modalities with both in-domain and zero-shot settings. This breadth is not a generic “many experiments” claim; the benchmark is explicitly organized to compare architectural paradigms under common preprocessing, training, efficiency measurement, and zero-shot transfer protocols.
- **The paper’s main empirical insight is concrete and important: many apparent improvements over vanilla U-Net are small and often not statistically convincing in-domain, while zero-shot gains are more substantial.** This is a strong, benchmark-driven claim that goes beyond leaderboard reporting and is central to the paper’s value. The analysis in Section 3.1, supported by per-dataset rankings and significance summaries in the appendix, gives the community a more sober view of progress than typical single-dataset papers.
- **U-Score is a useful benchmark artifact even if not yet definitive as a universal metric.** The paper clearly defines the metric from IoU, parameters, FLOPs, and FPS using percentile normalization and harmonic means, and importantly includes a substantial sensitivity analysis (Appendix E, Tables 10–13). That analysis shows the ranking is reasonably stable to weighting and quantile choices, which is a stronger validation than many newly proposed benchmark metrics receive.
- **The benchmark includes a meaningful zero-shot protocol rather than treating generalization as an afterthought.** The paper evaluates transfer to unseen datasets within the same modality/task (e.g., Kvasir→CVC300/CVC-ClinicDB, ISIC2018→PH2, Montgomery→NIH-test), and this is one of the more practically relevant aspects of the work.
- **The paper attempts to turn benchmark observations into actionable guidance through dataset-characteristic analysis and an advisor agent.** The feature characterization of foreground scale, shape complexity, and boundary sharpness, together with family-level comparisons, is a useful step toward task-aware model selection rather than one-size-fits-all ranking.

## Weaknesses

### Major:
- **The training protocol is internally inconsistent, and this directly affects the fairness of the benchmark conclusions.**  
  The paper states in the main text that it follows “official implementations … adopting their predefined settings, pretrained weights, and deep supervision strategies when available” (Section 2.2 / Introduction), but Appendix F.2.2 and Table 15 also state that training is unified across models with **SGD, lr=0.01, 300 epochs, batch size 8**, and a common BCE+Dice loss. These are not minor implementation details; they are competing descriptions of the actual protocol. For a benchmark whose central claims depend on fairness across very different architectures, the reader needs a precise answer to: which parts are inherited from official code, which are standardized, and whether pretrained initialization is actually used during benchmark training. As written, this ambiguity weakens confidence in architecture-level conclusions, especially when the paper interprets weaker performance of some families (e.g., Mamba in Section 3.2) as architectural rather than protocol-induced.

- **The statistical-significance framing is weaker than the paper claims, because the t-test setup is not sufficiently justified and appears to rely on single-seed training.**  
  The paper repeatedly emphasizes “statistical rigor,” but Table 15 lists a single random seed (**41**), and the paper does not clearly describe repeated runs per dataset/model. If significance is computed from one trained model per method, then the test is necessarily based on per-case predictions within a dataset rather than training-run variability; if so, the paper should state that explicitly and justify why that is the right inferential target. As written, the presentation blurs “statistical significance of a model difference on a test set” with “robustness of an architecture’s improvement,” which are not the same. This does not make the comparisons useless, but it does mean the benchmark overstates the rigor of its inferential claims.

- **The multiple-testing issue is not addressed.**  
  The paper performs large numbers of pairwise significance tests (each variant against U-Net across many datasets/modalities), but there is no discussion of multiple-hypothesis correction. Given how central Fig. 1E / Fig. 5 and the “few significant gains” narrative are to the paper, some treatment of family-wise error or false discovery rate is warranted. Without it, the precise counts of significant/non-significant wins should be interpreted cautiously.

- **U-Score is cohort-dependent, which limits its claim as a stable long-term metric of progress.**  
  By construction, U-Score normalizes each component using the **10th/90th quantiles over the current model zoo**, so a model’s score depends on which other models are included in the benchmark. The appendix does show sensitivity to quantile choices and weightings, which is good, but that is not the same as solving the core issue: U-Score is best understood as a ranking device *within this benchmark cohort*, not yet as an absolute metric that can track progress over time without recalibration. The paper occasionally describes it in broader terms than the methodology supports.

- **The model advisor agent is interesting but not yet a strong standalone contribution.**  
  Its evaluation is relatively narrow: training on 18 in-domain datasets, validating on 2 held-out datasets in the main setup, with appendix LOMO results showing that a simple heuristic can outperform it on IoU-only rankings in several settings. The paper’s own appendix notes that the heuristic is “extremely competitive” for IoU-only ranking. This makes the advisor more of a promising benchmark utility than a convincingly validated recommendation system. The main paper currently gives it more prominence than the evidence justifies.

### Minor
- **The paper’s “2D benchmark” positioning is somewhat imprecise because some included datasets are volumetric and are evaluated slice-wise.**  
  This is not a fatal issue—the paper does state in Appendix F.2.2 that 3D datasets like Synapse and ACDC are processed by axial slicing—but the title/abstract framing could more clearly say this is a **2D / slice-based benchmark**, not a benchmark of volumetric segmentation architectures.
- **The motivating literature audit (e.g., “84% papers neglect zero-shot evaluation,” “73% papers lack statistical significance testing”) is not documented with sufficient methodological detail in the main paper.**  
  Since these percentages are used prominently in Fig. 1 and the introduction, the sampling criteria for the 100 reviewed papers should be stated more transparently.
- **Some of the architecture-level interpretations are more speculative than established.**  
  For example, statements like RWKV showing “structural superiority” or Mamba underperforming due to difficulty with fine-grained detail are plausible but not strongly isolated from confounds such as training recipe, model size, or source-domain bias. The descriptive benchmark results are useful; the causal architectural explanations are less secure.
- **The preprocessing protocol may introduce cross-modality simplifications that are acceptable for standardization but should be discussed more carefully.**  
  The paper resizes datasets to 256×256 (or keeps 224×224 for some fixed-input models) and uses common augmentations across modalities. That is a practical benchmark choice, but for medical imaging it can distort scale/aspect information and may interact differently with certain architectures. This is a limitation of scope rather than a methodological error, but it deserves explicit acknowledgment in the main text.

### Trivial
- **The paper would benefit from a cleaner separation between benchmark contribution and auxiliary tools.**  
  U-Bench itself is substantial; the advisor agent and some interpretive claims would read more convincingly if presented as secondary extensions rather than co-equal headline contributions.

## Nice-to-Haves
- Provide a **clear protocol table** in the main paper that disambiguates: official code used, pretrained initialization used or not, optimizer/loss/scheduler standardized or not, and which model-specific training components are retained.
- Add a **multiple-testing correction analysis** for the significance results, or at minimum report how the conclusions change under an FDR procedure.
- Compare U-Score to **Pareto-front reporting** or simpler efficiency-accuracy summaries to clarify what additional decision value it provides beyond standard multi-objective views.
- Strengthen the advisor section with more **failure analysis** and clearer positioning as a benchmark-derived helper rather than a mature recommender.
- Make the title/abstract wording explicitly say **slice-based 2D evaluation** for volumetric datasets.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“Limited scope to 2D models / missing 3D evaluation makes the benchmark unacceptable.”**  
  The paper explicitly scopes itself as a **2D benchmark** from the abstract onward (“the first large-scale, statistically rigorous 2D benchmark”), and later explains that volumetric datasets are handled slice-wise. This can be noted as a scope clarification issue, but not as a substantive flaw for failing to do something outside its stated scope.
- **Claims questioning release status / availability / legal actionability of datasets, weights, or tools.**  
  Per instruction, these should not be treated as valid weaknesses.
- **Reproducibility complaints about missing hyperparameters or training logs.**  
  The paper actually provides substantial implementation detail in Appendix F and Tables 15–16.
- **Pure demands for more architectural ablations of internal modules as if this were an architecture paper.**  
  More granular ablations would be useful, but the paper is a benchmark, not a new model paper; this is better framed as a nice-to-have than a core weakness.
- **“Unfair comparison with other methods because the authors standardized preprocessing/training.”**  
  Standardization is the point of a benchmark. The real issue is not asymmetry itself, but that the paper is ambiguous about how much it standardizes versus preserves from official settings.

## Novel Insights
The most important synthesis across the reviews is that the paper’s **value is real but narrower than its rhetoric**: it is strongest as a large, useful empirical resource showing that in-domain gains over U-Net are often modest and that zero-shot differences are more informative than conventional single-dataset leaderboards. However, its attempt to elevate this into a claim of *statistically rigorous and architecturally diagnostic benchmarking* is currently undermined by protocol ambiguity (official settings vs unified training), insufficiently justified significance testing, and a cohort-relative U-Score. In other words, the benchmark appears practically valuable, but the paper overstates how cleanly its methodology supports architecture-level and significance-level conclusions.

## Suggestions
- **Resolve the protocol ambiguity first.** Add a concise, explicit statement of the exact training/evaluation policy and revise all contradictory wording about “official predefined settings.”
- **Clarify the inferential target of the t-tests.** State exactly what is being paired, what randomness is being modeled, and what “statistical significance” should and should not be interpreted as in this benchmark.
- **Add multiple-testing correction** and revise the significance claims if needed.
- **Reframe U-Score more modestly** as a benchmark-relative deployment metric unless an absolute normalization scheme is introduced.
- **Condense and demote the advisor agent** unless stronger validation is added; the benchmark itself is already the main contribution.
- **Tighten the title/abstract language** around slice-based evaluation for volumetric datasets.