Now let me look at some of the comparison papers to calibrate the score.Now I have enough information to write the consolidated review. Let me produce it.

---

## Summary

This paper proposes using bilinear MLPs—a GLU variant without element-wise nonlinearity—as interpretable drop-in replacements for standard MLPs in transformers. The core insight is that bilinear layer computations can be expressed as a third-order tensor, enabling exact weight-space analysis via eigendecomposition of output-conditioned interaction matrices. The authors demonstrate this approach on toy tasks, MNIST/Fashion-MNIST image classification, and small language models, showing interpretable low-rank structure, ground-truth circuit recovery, adversarial mask construction from weights, and a sentiment negation circuit in a 6-layer bilinear transformer.

---

## Claims and Support

**Claim 1: Bilinear MLPs admit an exact weight-space decomposition.**
**✅ Well-supported.** The math is clean: the bilinear layer output is $x^T Q x$ where $Q = u^T B$ is the output-conditioned interaction matrix. The symmetry argument (antisymmetric components contribute zero) and eigendecomposition yielding $\sum_i \lambda_i (v_i^T x)^2$ are mathematically sound. This is the paper's strongest contribution.

**Claim 2: Bilinear MLPs are a competitive "interpretable drop-in replacement."**
**⚠️ Partially supported.** The paper appeals to prior work (Shazeer 2020) and Appendix I for performance claims; the abstract fairly states "competitive performance." The paper correctly identifies that bilinear layers beat ReLU and nearly match SwiGLU. The "interpretable drop-in replacement" conclusion is reasonable given the evidence in the paper, though the interpretability advantage is demonstrated mainly qualitatively and on small models.

**Claim 3: Eigendecomposition reveals interpretable low-rank structure.**
**⚠️ Partially supported.** For images, the evidence is convincing qualitatively and backed by consistency experiments (cosine similarities 0.8–0.9 across runs) and truncation experiments. For language, the average correlation for 1 eigenvector is ~0.65, rising with more eigenvectors; most features exceed 0.75 with 2 eigenvectors. The "interpretable" part remains largely visual/qualitative.

**Claim 4: Extracted eigenvectors are causally meaningful (adversarial masks and overfitting diagnosis).**
**✅ Supported for adversarial construction; ⚠️ Partially for overfitting diagnosis.** The pseudoinverse-based adversarial masks cause significantly larger accuracy drops than random baselines—a genuine causal demonstration. The overfitting diagnosis (noisier eigenvectors = overfit model) is suggestive but remains qualitative.

**Claim 5: The method identifies a language-model circuit "directly from the weights alone."**
**⚠️ Partially supported and somewhat overstated.** The paper itself acknowledges in its Limitations section that "in deeper models, we rely on features derived from sparse autoencoders that are dependent on an input dataset." The abstract phrase "directly from the weights alone" is therefore in tension with the actual method for language models. The sentiment negation circuit is cherry-picked, and the evidence is correlational (0.66 overall, 0.76 on active-only features)—no causal intervention is performed.

**Claim 6: Weight-based interpretability is viable for understanding LLMs.**
**⚠️ Partially supported.** The paper shows viability for small bilinear transformers (6–16 layer). The claim is reasonable as a proof-of-concept but overstated as a general conclusion in the Discussion ("even for large language models").

---

## Strengths

- **Clean theoretical framework.** The exact quadratic-form representation of bilinear layers and its eigendecomposition is mathematically principled. The symmetry argument is a compact, non-obvious simplification that underpins the whole approach.
- **Ground-truth validation (Section 4.3).** The mechanistic interpretability challenge task provides a compelling ground-truth test: the decomposition recovers the known cosine-similarity labeling function from weights alone, without data or hints. This is one of the most convincing causal demonstrations in the paper.
- **Cross-run and truncation consistency (Section 4.2).** Cosine similarity of 0.8–0.9 for top eigenvectors across independent training runs, and near-identical accuracy after top-few-eigenvector truncation, provides real quantitative evidence of stable low-rank structure rather than isolated anecdotes.
- **Adversarial masks from weights (Section 4.4).** Constructing adversarial examples via pseudoinverses of top eigenvectors—without any forward passes—is a strong practical causal demonstration that the extracted directions correspond to real predictive structure.
- **Breadth of evaluation.** The paper covers toy tasks, two image datasets, and three language model scales, strengthening the claim that the method is general.
- **Honest limitations section.** The paper explicitly acknowledges SAE dependence in deeper models, orthogonality limitations, and lack of monosemanticity guarantees—which is commendably self-aware.

---

## Weaknesses

### Fatal
*None.*

### Major

- **"Weights alone" framing for language is inconsistent with the method.** The abstract states "identify small language model circuits directly from the weights alone," but the language-model analysis requires SAE input/output features trained on activation datasets. The Limitations section admits this, but the abstract and Discussion do not reflect it. This is not a trivial gap—it reframes what the paper has actually shown. The language section demonstrates weight-based analysis *in an SAE feature basis*, not pure weight-based analysis. The authors should correct this framing throughout, particularly in the abstract and Discussion.

- **No causal validation of the language circuit.** The sentiment negation circuit is supported by feature co-occurrence patterns, eigenspectrum geometry, and a moderate correlation (0.66 overall) between the low-rank approximation and SAE feature activation. There are no activation patching experiments, ablation tests, or controlled next-token probability measurements showing the circuit is necessary or sufficient for the claimed negation behavior. Without causal validation, the "circuit discovery" framing is too strong for what is essentially a correlational, cherry-picked observation.

- **Single cherry-picked language circuit.** The authors explicitly acknowledge this. As a standalone existence proof, a cherry-picked example is acceptable, but it cannot support the broad claim that the method enables language-model circuit discovery. Without knowing what fraction of features yield similarly interpretable low-rank structures, the method's reliability in language settings remains unclear.

### Minor

- **Approximation quality is moderate.** The average correlation of ~0.65 for a single eigenvector, rising with more eigenvectors, means substantial variance in output feature activations is not captured by top eigenvectors. This is not fatal but limits the claim that interaction matrices are "surprisingly low-rank."

- **The overfitting diagnostic is subjective.** Equating visual cleanliness of eigenvectors with generalization/overfitting relies on human judgment. The paper shows that Gaussian noise regularization produces more "digit-like" eigenvectors and reports similar test accuracy across regularization levels (97.2%–98.1%), but provides no quantitative link between eigenvector pathology and train-test gap.

### Trivial

- Section 3.2 describes $d$ eigenvectors as "the rank of $W, V$," which is slightly imprecise (rank of $W$ or $V$ separately vs. rank of the interaction matrix). This does not affect the substance.

---

## Nice-to-Haves

- **Systematic interpretability quality evaluation across all features.** A histogram or automated metric (e.g., probing accuracy, concept alignment) showing what fraction of all eigenvectors are interpretable—not just top-picked examples—would significantly strengthen the paper's reliability claims.

- **Causal ablation for the language circuit.** Even a small-scale activation patching experiment on controlled prompts ("not good" vs. "good") would substantially upgrade the circuit claim from correlational to causal.

- **Scalability discussion.** Computing eigendecompositions of $d \times d$ matrices for each output direction at realistic LLM hidden dimensions (thousands) could be expensive. A brief analysis of computational cost and approximations (randomized SVD, top-k power iteration) would be valuable for practitioners.

- **Failure case visualization.** Showing examples where top eigenvectors are incoherent or uninterpretable would honestly characterize the method's boundary conditions.

- **Exploring SDL directly on the bilinear tensor** (mentioned in Limitations as future work) to relax the orthogonality constraint—even preliminary results would strengthen the paper's vision.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"No direct controlled comparison of bilinear vs. standard MLPs at matched budgets" (Harsh Reviewer, Claim 2).** The paper explicitly references Appendix I for performance comparisons and cites Shazeer (2020) for the competitive-performance claim. Criticizing the paper for not providing this comparison in the main body, when it is deferred to the appendix and prior work, is a reproducibility nitpick rather than a genuine evidential gap—especially since the paper's main contribution is the *analysis method*, not a performance benchmark. Removed per the reproducibility rule.

- **"Weak baselines for adversarial masks" (Harsh Reviewer).** The harsh reviewer asks for comparison against "non-top eigenvectors or top singular vectors without class conditioning." The paper compares against random mask permutations, which is the natural baseline for evaluating whether the specific interpretive decomposition matters. Demanding stronger non-standard baselines is scope creep.

- **"Scalability to ≥1B models needed to establish drop-in replacement claim" (Spark Reviewer).** The paper's main contribution is the analysis methodology and its demonstration at several scales; it does not need to validate at frontier model sizes to be a valid research contribution. Removed as it demands methodological practice not standard for an interpretability methods paper.

- **"Incomplete comparison to activation-based methods" (Neutral Reviewer / Human Finder).** This would demand a different paper with different baselines trained on the same tasks. The paper's claim is that bilinear layers are *more amenable* to analysis, not that the extracted circuits are provably better than what SAE-based methods find on equivalent architectures. Removed as scope creep.

---

## Novel Insights

The paper's most genuinely novel observation is that the exact algebraic tractability of bilinear MLPs—achieved by removing element-wise nonlinearities while retaining competitive performance—creates a new regime where weight-space analysis is not an approximation but an exact reformulation. The symmetry argument (antisymmetric components vanish under quadratic evaluation) is a clean, useful simplification that enables principled eigendecomposition. The adversarial mask result (Section 4.4)—demonstrating that pseudoinverse-based masks constructed from eigenvectors cause significantly greater accuracy drops than random baselines without any forward passes—is a particularly striking causal proof-of-concept that the extracted directions correspond to real model structure rather than post-hoc rationalization. The observation (Section 5.2) that SAE feature activations improve dramatically with SAE training time while standard SAE metrics barely change, suggesting a "hidden" convergence transition, is an intriguing secondary finding that warrants independent follow-up.

---

## Suggestions

1. **Fix the abstract's "from the weights alone" language** to acknowledge SAE dependence for the language section. Something like "using weight-based analysis in an SAE feature basis" is more accurate and still highlights the contribution.

2. **Add a small causal validation experiment** for the sentiment negation circuit: patch or ablate the two identified eigenvector directions on controlled prompts ("not good," "not bad," "very good") and measure next-token distribution shifts.

3. **Provide a distribution histogram of per-feature approximation quality** (Figure 9B style) across all layers and features, so readers can assess what fraction of the model's computations the method covers—not just at 2/3 depth.

4. **Separate the Introduction's contribution list more clearly** between (a) pure weight-based analysis (vision, toy tasks) and (b) SAE-assisted weight-based analysis (language). This accurately scopes each claim.

5. **For the overfitting diagnostic**, add at least one quantitative metric (e.g., train-test gap correlation with eigenvalue spread) alongside the qualitative figures to move from illustrative to evidential.

---

## Score and Decision

**Calibration:**

- **5IWJBStfU7** ("Is MI Identifiable?", scores 6-8, accepted poster): Theoretical MI paper with toy experiments, accepted. This paper has comparable breadth and stronger empirical grounding (multiple domains, causal adversarial result) but slightly weaker rigor on its strongest claims.
- **1Njl73JKjB** ("Towards Principled SAE Evaluations", scores 6-8, accepted poster): Strong empirical SAE study, accepted. The present paper is somewhat less rigorous than this but covers a more novel angle (architectural choice enabling weight-based analysis).
- **yBhSORdXqq** ("Modular addition without black-boxes", scores 3-8, rejected): Single-circuit MLP interpretability paper with scope limitations; rejected. This paper is broader and has more genuine novelty.
- **OeHSkJ58TG** ("Incidental Polysemanticity", scores 5-6, rejected): Polysemanticity study on toy models only. The present paper covers more ground.

**Assessment by axis:**
- *Originality*: Good. Identifying bilinear layers as the "right" architecture for weight-based analysis and developing the eigendecomposition method is genuinely novel.
- *Importance*: Moderate-to-high. Weight-based mechanistic interpretability is an important open problem; this is a concrete step forward.
- *Claims vs. support*: Partially adequate. Vision claims are well-supported; language claims are overstated relative to evidence.
- *Soundness of experiments*: Good for vision; weak for language (no causal validation).
- *Clarity*: Generally clear; framing inconsistency in abstract/Discussion is the main writing flaw.
- *Value to community*: Real value as a new methodology, even with current limitations.

Relative to calibration anchors: this paper is comparable to the accepted poster range (6–8) rather than rejected papers (3–6). The SAE-dependence framing inconsistency and lack of causal language validation are real weaknesses but do not undermine the core algebraic contribution and the vision-side evidence. I place it at **6.0**—above the borderline rejected papers, consistent with the lower end of the accepted poster cluster, pending revision of the language-section framing and causal evidence.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>