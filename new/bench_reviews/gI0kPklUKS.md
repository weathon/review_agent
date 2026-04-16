## Summary
This paper proposes using bilinear MLPs—GLUs without elementwise nonlinearities—as an architecture that admits exact weight-based analysis via symmetric interaction matrices and eigendecomposition. The core technical contribution is clean and real: for any chosen output direction, the layer can be rewritten as a quadratic form whose eigenvectors expose low-rank input directions, and the paper demonstrates this on toy vision tasks, a ground-truth reverse-engineering task, and small bilinear language models using SAE feature bases.

## Strengths
- **Clear and technically sound core formulation.** The paper’s rewrite of bilinear layers into interaction matrices/tensors is elegant and exact. In particular, the symmetrization argument in Sec. 2 is correct for the scalar quadratic form \(x^TQx\), and Sec. 3.2 gives an exact eigendecomposition-based analysis for any chosen output direction.
- **The best result is genuinely compelling:** Sec. 4.3 provides a rare ground-truth-style validation where the method recovers the target-similarity computation from weights alone on a mechanistic interpretability challenge task. This goes beyond mere visualization and directly supports the usefulness of the decomposition.
- **The paper includes more than anecdotal qualitative evidence.** Sec. 4.2 studies cross-run consistency and truncation, showing top eigenvectors are stable across runs and that retaining only a few dominant eigenvectors preserves performance well. This strengthens the claim that the extracted structure is not purely accidental.
- **Useful practical demonstrations.** The adversarial-mask construction from weight-derived directions and the overfitting diagnosis via noisy eigenvectors are interesting demonstrations that the decomposition is not just descriptive but can guide interventions and diagnostics.
- **Good organization and clarity.** The three analysis regimes in Sec. 3—using input/output features, output features only, or no features—make the method easy to understand. The paper is also relatively candid in its limitations section, especially about dependence on meaningful output directions and possible limits of orthogonal eigenvectors.

## Weaknesses

###: Fatal
- None.

### Major:
- **The paper overstates its empirical scope relative to its headline claims.** The abstract and conclusion claim that bilinear MLPs are an “interpretable drop-in replacement” and that “weight-based interpretability is viable for understanding deep-learning models.” The actual evidence is much narrower: shallow MNIST/Fashion-MNIST classifiers, one specially structured ground-truth toy task, and small bilinear transformers where the language analysis is preliminary. That supports a more modest claim—namely that bilinear layers *admit exact and sometimes useful weight-based decompositions*—but not yet the broader practical conclusion suggested in the title/abstract/discussion.
- **The strongest language-model results are not “from the weights alone.”** In Sec. 5, the main analysis depends on SAE-derived input/output feature bases learned from activations and data. The paper explicitly states in Sec. 3.1 and the limitations that deeper-model analysis “rel[ies] on features derived from sparse autoencoders that are dependent on an input dataset.” This does not negate the contribution, but it materially narrows the headline framing. What is demonstrated is: given a meaningful feature basis, bilinear weights can be analyzed cleanly. That is weaker than weight-only mechanistic interpretability for realistic language models.
- **The language evidence for mechanism recovery is suggestive rather than fully convincing.** The sentiment-negation circuit is explicitly “cherry-pick[ed]” in Sec. 5.1, and the two-eigenvector approximation reaches correlation 0.66 on active cases (0.76 at large activations), which is interesting but not a clean recovery of the computation. Likewise, Sec. 5.2’s low-rank approximation results show substantial structure, but correlation-on-active-examples is only a partial proxy for mechanistic faithfulness. The paper needs more systematic evidence about how often interpretable circuits are recovered and how causally decisive the recovered directions are.
- **The vision interpretability claims rely heavily on visual plausibility, with limited quantitative validation beyond one toy ground-truth task.** Sec. 4.1’s main conclusion is essentially that top eigenvectors “appear interpretable,” and the regularization story in Fig. 4 is also largely qualitative (“more digit-like eigenvectors”). The ground-truth task in Sec. 4.3 is a meaningful validation, but it is also unusually well aligned with a quadratic-form analysis. Overall, the paper shows that the decomposition yields useful and often intuitive directions, but not yet that these directions generally constitute robust mechanistic explanations.

### Minor
- **The paper does not fully disentangle how much interpretability comes from the bilinear weights versus the semantics of the chosen output direction.** In Sec. 3.2/4.1 the decomposition is conditioned on a selected output direction \(u\), often from a classifier head/unembedding or SAE feature. This is mathematically appropriate, but it leaves open whether the interpretability is intrinsic to the bilinear weights or partly imported by choosing a human-meaningful \(u\). Controls with random/rotated output directions or alternative decompositions would have clarified this.
- **The “no-features” route is underdeveloped relative to the paper’s framing.** Sec. 3.3 presents HOSVD/SVD as the pathway most aligned with “no prior features,” yet the main empirical narrative relies much more on chosen output directions or SAE bases. This weakens the strongest framing around input-free discovery.
- **Performance/utility tradeoffs are not prominent enough in the main text.** The paper says bilinear layers are competitive and refers to Appendix I for corroboration, but for a paper advocating an architectural replacement, the practical performance comparison versus standard choices should be surfaced more centrally.
- **Some analyses would benefit from stronger baselines.** For example, the adversarial-mask experiment mainly compares against random masks/permutations; stronger baselines would help establish that the proposed eigenstructure is specifically useful rather than merely yielding class-correlated perturbation directions.

### Trivial
- **Orthogonality may limit semantic cleanliness.** The paper itself acknowledges in Sec. 6 that eigenvectors need not be monosemantic and that orthogonality may hurt interpretability in higher-rank settings. This is a real limitation, though not a flaw in the current derivation.
- **Quadratic/XOR-like activations make interpretation less intuitive than linear features.** Figure 2A makes clear that both positive and negative regions can contribute through squaring, which adds cognitive overhead even if the decomposition is exact.

## Nice-to-Haves
- Add a more systematic census of language-model circuits/features: what fraction of SAE output features are well-approximated by top-\(k\) eigenvectors, and what fraction admit semantically coherent interpretations.
- Include a causal intervention on the language circuit, e.g., ablating/amplifying the identified eigenvector directions and measuring the effect on logits or feature activations.
- Add controls with random or rotated output directions to better isolate interpretability arising from bilinear weights themselves.
- Move the bilinear-vs-standard-MLP performance comparison from the appendix into the main text.
- Show some failure cases where eigenvectors are not interpretable or low-rank approximations fail.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“The paper should compare against standard MLP/SwiGLU interpretability methods on the same tasks, otherwise comparisons are unfair.”** This is not a fair core criticism here. The paper’s main contribution is to analyze bilinear MLPs as a deliberately more interpretable architecture, not to claim superiority over every existing interpretability method on standard MLPs. Such a comparison would strengthen the paper, but its absence is not a fatal methodological flaw.
- **“The symmetry discussion is wrong because symmetrization changes the original factorization \(w_av_a^T\).”** Removed as a weakness. The paper does not claim the factorization itself is preserved; it correctly claims only that the antisymmetric part does not affect the scalar quadratic form \(x^TBx\). That is mathematically sufficient for the eigendecomposition analysis.
- **“The work is not really weight-based because it ever uses data-dependent components.”** Overstated and removed in that form. The paper genuinely includes weight-only analyses in the shallow and ground-truth settings. The valid criticism is narrower: the *language-model* results depend on SAE features, so the strongest weight-only framing should be softened.
- **Pure scalability demands such as requiring frontier-scale 70B+ validation.** This is beyond reasonable scope for a paper introducing a new interpretability architecture and method. The valid concern is simply that current evidence is limited to small models, not that the paper must already demonstrate frontier-scale deployment.

## Novel Insights
The most interesting synthesis is that the paper’s real contribution is less “mechanistic interpretability from weights alone” in the absolute sense and more a design principle for interpretability-aware architectures: by replacing difficult nonlinearities with a bilinear form, one obtains an exact quadratic object whose semantics can be interrogated with standard linear algebra. This positions the work as a promising architectural path for interpretability, but the current experiments support that claim mainly in controlled settings and only partially in deeper language models.

## Suggestions
- Narrow the headline claims to match the evidence, especially around “drop-in replacement” and “from the weights alone.”
- Strengthen Sec. 5 with systematic coverage metrics over many SAE features, not just one cherry-picked circuit.
- Add a causal intervention experiment for the language-model circuit.
- Bring the bilinear-vs-standard performance comparison into the main paper.
- Include controls using random/rotated output directions and stronger baselines for adversarial masks.
- Explicitly distinguish three regimes in the narrative: true weight-only analysis in shallow settings, weight-conditioned-on-output-directions, and weight analysis in SAE feature coordinates for deeper models.

## Score and Decision
**Assessment by axis:**  
- **Originality:** good. The eigendecomposition view of bilinear MLPs for mechanistic interpretability is novel and technically meaningful.  
- **Importance of question:** high. Understanding how MLPs compute from weights is an important interpretability problem.  
- **Support for claims:** moderate. The core mathematical claims are sound, but the broad empirical framing is stronger than the evidence warrants.  
- **Experimental soundness:** moderate to good. There are several useful experiments, including one strong ground-truth case, but the language evidence is still preliminary and somewhat selective.  
- **Clarity:** good. The paper is unusually clear in its derivations and presentation.  
- **Value to the community:** meaningful, especially as an architectural direction for interpretability, though immediate impact is limited by the restricted empirical scope.

**Calibration:** I compared this paper against retrieved human-reviewed papers with similar profiles:
- **yBhSORdXqq (“Modular addition without black-boxes”)** — rejected with mixed scores (6/3/8/5): similar in being a mechanistic interpretability paper with a real technical idea but limited scope/toy validation. The current paper is somewhat stronger because it spans more settings and has a stronger exact architectural story.
- **PJjHILiQHC (“Approaching Deep Learning through the Spectral Dynamics of Weights”)** — rejected with 8/6/6/5: similar in using spectral/weight analysis with broad framing and limited support for strongest claims. The current paper feels comparable in ambition and somewhat better grounded technically, but still not fully supported at the scale implied by the framing.
- **5IWJBStfU7 (“Is Mechanistic Interpretability Identifiable?”)** — accepted with 6/6/8/8: a stronger paper in terms of conceptual depth and support. The current submission falls below this level because its empirical case for broad viability is not yet as convincing.
- **hOMVq57Ce0 (“Piecewise Linear Parametrization...”)** — accepted poster with 6/8/6/3: similar in proposing an interpretable architecture rather than a post hoc method. The current paper is in that neighborhood, but the mismatch between headline claims and evidence keeps it slightly below a comfortable accept.

Overall, this is a **good, promising paper with a real core contribution**, but the strongest claims need to be narrowed and the language-model evidence is not yet strong enough to fully justify the title/abstract-level conclusions. I land slightly below the acceptance threshold.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>