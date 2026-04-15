## Summary

This paper applies Bayesian non-negative matrix factorization (NMF) to decompose fMRI responses from the ventral, lateral, and dorsal visual streams in four NSD subjects, identifying distinct component selectivities across streams (faces/scenes/bodies in ventral; social interactions/implied motion/hand actions in lateral; scenes/motion in dorsal). To explain why standard alignment metrics (linear encoding, RSA) fail to reflect these functional differences, the authors introduce Sparse Component Alignment (SCA), a novel method that measures representational alignment based on whether stimulus pairs activate the same dominant sparse component. Using SCA, they report markedly higher alignment between standard image-trained DNNs and the ventral stream compared to dorsal/lateral streams — a pattern obscured by rotation-invariant methods.

---

## Claims and Support

**Claim 1: Distinct dominant components exist in the three visual streams.**
*Partially supported.* Ventral components are plausible and corroborated by behavioral saliency ratings and prior work. Lateral components (group interactions r=0.454, implied motion r=0.660, hand actions r=0.448) are novel and reasonably validated. Dorsal is thin — only 2 consistent components emerge (scenes r=0.393, implied motion r=0.428) vs. 5 in ventral and lateral. The paper itself concedes dorsal components are "less interpretable" in the abstract. Importantly, the claim that this is "free of a priori hypotheses regarding spatial layout" is overstated: the decomposition is run *within* predefined anatomical stream masks, so only the component identities — not stream boundaries — are hypothesis-free.

**Claim 2: Bayesian NMF is preferable over PCA/standard NMF.**
*Partially supported.* Simulation results in Fig. 2 show better latent recovery and sparser factors vs. alternatives. However, simulations are constructed from sparse latent components, inherently favoring sparse models. The explained variance gap between bnmf and PCA in real brain data is small (~3–5% per Fig. 3 extracted values), not "notable." Biological motivation for non-negativity is reasonable for ventral cortex specifically; claiming it applies uniformly across all preprocessed fMRI streams is an overreach but minor.

**Claim 3: Standard alignment methods fail to reflect stream differences due to rotational invariance.**
*Partially supported.* RSA does show a non-trivial ventral advantage (ventral r=0.347 vs. dorsal r=0.199, lateral r=0.222), contradicting the framing that standard metrics are completely blind to stream differences. Linear encoding shows dorsal (r=0.232) ≥ ventral (r=0.180), consistent with the paper's point that linear encoding is less discriminative. The mechanistic explanation — rotational invariance is "why" RSA partially fails — is not isolated from other factors SCA changes simultaneously (sparsification, binary discretization). The causal claim is therefore too strong.

**Claim 4: SCA is a "better" measure of brain-model alignment.**
*Unsupported as stated.* The paper demonstrates SCA is *different* and axis-sensitive, not that it is *more correct*. No external validation criterion (split-half reliability, prediction of held-out behavior, noise robustness) is established. The abstract's phrase "better captures latent neural tuning" exceeds what the evidence shows.

**Claim 5: Standard DNNs are substantially more aligned with ventral than dorsal/lateral.**
*Supported under SCA, with caveats.* The SCA pattern (ventral r=0.187 vs. lateral r=0.047, dorsal r=0.058) is consistent across all seven tested models. But this conclusion rests entirely on an unvalidated metric, and no confidence intervals, subject-level variance, or statistical tests are reported for these key numbers.

**Claim 6: ICMs capture behaviorally relevant information similarly to RSA.**
*Partially supported.* Fig. 6 shows ICMs preserve the ventral > dorsal/lateral trend and trained > untrained trend. However, ICMs suppress the intermediate lateral/dorsal alignment visible in RDMs, indicating they capture different (not "similar") structure. The claim should be narrowed.

---

## Strengths

- **Addresses a genuine conceptual contradiction.** The paper cleanly frames a real paradox — identical DNNs align well with all three visual streams despite their distinct functions — and provides a tractable causal hypothesis (rotational invariance of standard metrics). This is well-motivated.
- **Novel lateral stream characterization.** Data-driven identification of sub-components selective for group interactions, hand actions, and reachspaces in the lateral stream is a genuine contribution beyond the prior ventral-focused literature (e.g., Khosla et al., 2022). Behavioral saliency correlations provide quantitative grounding.
- **SCA simulation validation is clear and informative.** Figure 2 convincingly demonstrates that SCA is sensitive to axis rotations while RSA is not — a genuine proof-of-concept that motivates the method.
- **Comprehensive multi-method comparison.** Four alignment metrics across seven model architectures (including self-supervised variants) with a behavioral validation anchor (Meadows dataset) make the empirical contribution multifaceted.
- **Well-written framing.** The introduction sets up the research questions sharply, and the discussion honestly acknowledges that static images are insufficient for fully probing dorsal/lateral streams.

---

## Weaknesses

### Fatal
*None — the paper's contributions are real, though several major gaps require revision before the main claims can be fully accepted.*

### Major

- **SCA is not validated as "better," only as "different."** The paper's abstract and discussion claim SCA "better captures latent neural tuning" and reveals DNN–ventral alignment "with greater resolution," but no criterion validity is established: no split-half reliability analysis, no test of whether SCA predicts held-out behavior better than RSA, and no robustness to hyperparameters (C, N, hard vs. soft assignment). The observed ventral-favoring pattern under SCA could reflect the metric's inductive bias toward sparse/category-like axes rather than a more faithful measurement. This is the central evidential gap; without it, the headline conclusion that DNNs truly align specifically with the ventral stream (and not just that SCA is tuned to amplify categorical structure) is not established.

- **Algorithm 1 contains a genuine indexing inconsistency.** The connectivity matrix C^n is declared as dimension S×S (line 3, S = number of stimuli), but the inner loop (line 5) iterates `i, j ← 1:C` over component indices and `C^n_{i,j}` is assigned at component indices. This is inconsistent with the mathematical definition in Equation 2, which defines c_{ij} over stimulus pairs. The pseudocode does not cleanly specify the algorithm's behavior and would prevent independent implementation. The mathematical equation itself is clearer, but the gap should be resolved.

- **Binary hard-assignment in SCA is unjustified and unablated.** SCA collapses continuous component responses to a winner-take-all binary assignment (same dominant component or not). This discards magnitude and secondary-component information and likely amplifies sparse/categorical structure. No ablation compares soft assignments, top-k assignments, or continuous correlation on component response vectors. The alignment difference between streams could be driven by discretization alone, not by axis sensitivity.

- **No statistical testing for the headline result.** The key comparison (ventral SCA r=0.187 vs. dorsal r=0.058, lateral r=0.047) is reported without confidence intervals, subject-level variance, or significance tests. With N=4 subjects, the between-stream differences must be demonstrated to not be driven by a single subject outlier or decomposition seed.

- **C=20 fixed across all streams without per-stream justification.** The choice is borrowed from Khosla et al. (2022), who selected it specifically for the ventral stream via BIC. Applying it to dorsal and lateral streams with no separate optimization is unwarranted, especially given that only 2 dorsal components pass the consistency threshold compared to 5 in ventral/lateral. The claim that "similar results arise from 10–30 components" is stated without supporting data.

### Minor

- **Dorsal stream evidence is thin and asymmetric.** Only 2 consistent dorsal components are found vs. 5 each in ventral and lateral. This asymmetry could mean (a) the static-image stimulus set poorly samples dorsal computations, (b) dorsal representations are genuinely less categorical/sparse, or (c) the decomposition is noisier in dorsal. The paper does not distinguish these explanations. The phrase "clear difference in component response profiles across the three visual streams" in the abstract overstates the dorsal evidence; the dorsal findings are illustrative rather than robust.

- **Explained variance gap between bnmf and PCA is small.** Figure 3 shows the difference is ~3–5 percentage points, not the "notable" gap implied by some reviewers. This is actually a minor strength: bnmf retains comparable variance while producing much sparser factors.

- **No video-trained or action-trained models tested.** The paper concludes that "video-trained networks would better fit neural responses sensitive to motion" and points toward this as future work — but actually testing even one video model (e.g., VideoMAE) would directly test the paper's mechanistic story about training objectives.

- **The "free of a priori hypotheses" framing is partially overstated.** Stream masks are anatomically predefined, so only component identities (not stream boundaries) are hypothesis-free. The paper should clarify this.

### Trivial

- The biological argument against negative values in R applies specifically to the ventral pathway (the paper does say "in the ventral visual pathway") and is reasonable there; the harsh critic's challenge that preprocessing/centering undermines this argument is valid in general but does not damage the paper since NMF's non-negativity is also justified on interpretability grounds.

---

## Nice-to-Haves

- A soft/graded version of SCA (e.g., correlation of continuous component-loading profiles) as an ablation would clarify whether the ventral advantage is specifically due to hard discretization.
- Extending to at least one video-trained model (e.g., VideoMAE) would directly test the paper's central speculative claim in the discussion.
- A sensitivity analysis showing SCA alignment scores across C = 5–40 for each stream would make the C=20 choice more defensible.
- Component-level decomposition of what drives SCA alignment (e.g., which ventral component pairs are responsible for the ventral–DNN alignment) would substantially improve interpretability.
- Per-subject SCA alignment scores with error bars would address the statistical concern given N=4.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"Linear encoding also shows a ventral advantage"** (Spark reviewer): Factually incorrect. The paper reports linear encoding alignment as dorsal r=0.232 > ventral r=0.180 ≈ lateral r=0.179 — dorsal is highest, not ventral. The paper itself states this is "similarly well" across streams. This criticism was based on a misread of the quantitative results.

- **"Explained variance in brain data is notably below PCA"** (Harsh Critic): Fig. 3 data shows bnmf is within ~3–5% of PCA for all streams. The word "notably" is not supported; the gap is small and the paper correctly frames it as an acceptable tradeoff for sparsity.

- **Lack of comparison to CKA/Procrustes/other alignment metrics** (Human Finder reviewer): While additional comparisons would be informative, the paper explicitly studies the rotational invariance of RSA and linear encoding and does not claim CKA/Procrustes are also invariant. Demanding a comprehensive benchmark of all similarity metrics is outside the paper's stated scope.

- **Missing video/multimodal models as a weakness** (multiple reviewers, partial): This is noted as a nice-to-have but not a fatal weakness; the paper explicitly scopes to image-trained DNNs and acknowledges the gap in the discussion. Treated as a nice-to-have above.

---

## Novel Insights

The paper's most genuinely novel insight is that rotation-invariant alignment metrics may produce a misleading picture of model–brain alignment by conflating two distinct questions: "does the model span the same subspace?" (which RSA/encoding measure) and "does the model use the same native axes of tuning?" (which SCA attempts to measure). The distinction between geometric subspace overlap and axis-specific tuning correspondence is underappreciated in the brain-model alignment literature, and SCA — despite needing more rigorous validation — offers a concrete (if imperfect) operationalization of the latter. The lateral stream component characterization is also novel: decomposing it into sub-components for group interactions, implied motion, hand actions, scenes, and reachspaces provides the most granular data-driven functional taxonomy of this stream to date.

---

## Suggestions

1. **Reframe the abstract claim:** Replace "better captures" with "provides complementary and axis-sensitive measurement of" latent neural tuning. This accurately reflects what the evidence shows.
2. **Fix Algorithm 1:** Rewrite the pseudocode to correctly iterate over stimulus pairs (i,j ← 1:S) and use distinct notation to avoid overloading C for both the matrix and the component count.
3. **Add split-half reliability for SCA:** Randomly divide the 50 MCMC iterations into two halves and report the correlation between the resulting ICMs. This is the minimum needed to establish the metric's reliability.
4. **Ablate hard vs. soft SCA:** Report SCA scores using the cosine similarity of component-loading profiles (continuous) instead of binary co-assignment, and compare across streams and models.
5. **Report per-subject SCA alignment values** with SEM in a supplementary table to demonstrate the ventral advantage is consistent across all 4 subjects.
6. **Provide a C sensitivity analysis** (C = 5, 10, 20, 30, 40) showing the number of consistent components and SCA alignment per stream — this is claimed but never shown.
7. **Qualify the dorsal claims more carefully:** The dorsal stream section should explicitly state that only 2 consistent components emerged (vs. 5 in other streams) and that this likely reflects stimulus limitations rather than a complete functional characterization.

---

## Score and Decision

**Originality:** High — SCA is a genuinely novel idea addressing an underappreciated methodological limitation; lateral stream characterization is new.

**Importance of research question:** High — the DNN-brain alignment paradox is central to computational neuroscience.

**Claim support:** Moderate — ventral decomposition findings are well-supported; SCA's "better" claim and the dorsal characterization are not.

**Soundness of experiments:** Moderate — no statistical testing, small N, hard-assignment unablated, pseudocode error.

**Clarity of writing:** Good — well-structured with an honest limitations discussion, but the abstract overclaims.

**Value to the research community:** Moderate-to-high — opens a new methodological direction even if the current execution is incomplete.

**Overall:** An interesting paper with a creative methodological idea and solid neuroscience contributions, but the main "better" claim is not established, the pseudocode has a genuine error, the critical design choice (hard assignment) is unablated, and statistical validation is absent for the headline result. The paper is close to acceptable but needs targeted revisions before the core claims can be trusted.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>