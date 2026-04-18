Now I have a good calibration base. Let me synthesize my final review.

Let me re-examine the specific claims in the paper more carefully to verify reviewer concerns:

1. **Source data dependency**: The paper states "We use the full training set of CIFAR-10, CIFAR-100 to compute the class means for Voronoi sites and 10% of ImageNet for similar calculation." This means TTVD requires accessing labeled training data to precompute Voronoi sites. Class means are essentially prototypes stored with the model. In standard TTA, methods use only the pre-trained model parameters. However, CIVD requires computing not just class means but also rotation-cluster centers, which requires running the feature extractor on augmented training data. This is a genuine concern about the TTA setting definition, but many TTA methods (including T3A) use stored prototypes. The question is whether the Voronoi sites are just simple prototypes that could be derived from the classifier weights (via Lemma 3.1), or whether they require actual access to training data.

Actually, looking more carefully: T3A also uses source prototypes computed from source data. And Lemma 3.1 explicitly links PD to classifier weights. The paper states the sites are "commonly...set using the class mean of the training data" (Section 3.1). Table 4 shows they tested with just 1% of ImageNet data and got similar results. So this concern exists but is partially addressed.

2. **Influence function being heuristic**: This is a valid concern. The exponent γ in Equation 4 and the sign(γ) mechanism are indeed not well-justified. The paper doesn't provide theoretical grounding for why this specific form of influence function is appropriate.

3. **PD-based noise filtering being underspecified**: This is a valid concern. The paper says "subtracting the PD from the VD" but doesn't give a precise algorithmic definition in high dimensions.

4. **Parameters updated**: The paper says "commonly, only the channel-wise affine parameters in normalization layers are updated during TTA" (Section 3). Algorithm 1 updates σ (feature extractor), which is ambiguous.

5. **Unfair comparisons**: Multiple baselines (T3A, TAST) are neighbor-based methods that operate differently from TTVD. The improvement may come from the combination of self-supervision + entropy minimization + prototype guidance, rather than purely from the geometric framework.

6. **Ablation only on CIFAR-10-C**: This is a valid concern — Table 2 only shows ablations on one dataset.

7. **No error bars**: Valid concern for modest improvements.

8. **Missing recent TTA methods**: Valid but somewhat standard concern.

9. **Computational overhead**: Valid concern.

10. **Architecture diversity**: Only ResNet tested.

Now let me carefully evaluate which weaknesses are truly substantive vs. which are overstated or invalid.

**On the "source data dependency" concern (Harsh Critic #1):**
This is partially but not fully valid. The paper clearly states that Voronoi sites are precomputed from source data (Section 4.1). However, T3A also computes class prototypes from source data. Table 4 shows robustness to reduced source data. Moreover, via Lemma 3.1, the PD weights can be derived from the classifier. The concern about "accessing 10% of ImageNet" is real, but Table 4 shows 1% also works, and class means could theoretically be derived from the classifier weights. This is a valid concern about setting fairness but the paper addresses it to some degree. I'll rate this as a **major** weakness.

**On CIVD/CIPD influence functions being heuristic (Harsh Critic #2):**
This is valid. The specific form of the influence function (Eq. 4 with the exponent γ and sign function) is not theoretically motivated. The claim that CIVD "unifies" self-supervision and entropy minimization and "avoids negative transfer" is asserted without formal justification. This is a meaningful gap. **Major** weakness.

**On PD-based noise filtering being underspecified (Harsh Critic #3):**
This is valid. The "diagram subtraction" concept is only illustrated on a 2D toy and lacks precise algorithmic specification for high-dimensional spaces. The ablation in Table 2 doesn't isolate the filtering contribution. **Major** weakness.

**On unfair comparisons vs neighbor-based methods (Harsh Critic #4):**
This is partially valid. TTVD combines self-supervision, entropy minimization, and prototype guidance — it's doing strictly more than T3A or TAST alone. However, the paper does compare against TENT and SAR which also optimize the feature extractor, and still shows TTVD outperforming them. The improvement over combined methods like SHOT (which also uses self-supervision + entropy) is more modest (1.4% on CIFAR-10-C). So the comparison concern is valid but partially addressed. **Minor** weakness.

**On ambiguity about what parameters are updated (Harsh Critic #5):**
This is a valid clarity concern. The main paper says "commonly only channel-wise affine parameters" are updated, but Algorithm 1 uses σ (feature extractor). The final algorithm is deferred to Appendix H. However, this is standard in TTA papers — they typically update only BN parameters and the notation is just convention. **Minor** weakness.

**On missing TTA methods (Spark):**
Valid but fairly standard. Several recent methods (EATA, CoTTA, EcoTTA) are absent. **Minor** weakness.

**On only synthetic corruption benchmarks (Human Finder):**
Valid but these are the standard TTA benchmarks. **Trivial** weakness.

**On classification-only (Human Finder):**
Valid scope limitation but typical for this area. **Trivial**.

**On novelty concerns about adapting existing geometric structures (Human Finder):**
This is a fair point. CIVD and PD are classical structures; the paper applies them to TTA. The novel contribution is the framing and the specific application, not the geometric structures themselves. But this is how applied research works — the insight of connecting neighbor-based TTA to Voronoi diagrams and extending to CIVD/PD is the contribution. **Minor** novelty concern.

**On no error bars (Human Finder):**
Valid for modest improvements. **Minor**.

**On computational overhead (Human Finder & Harsh Critic):**
Valid. Not discussed in the paper. **Minor**.

**On architecture diversity (Human Finder):**
Only ResNet-26 and ResNet-50 tested. This limits generalizability claims. **Minor**.

**On ablation only on CIFAR-10-C (Spark):**
Valid — Table 2 only shows the VD→CIVD→CIPD progression on one dataset. **Minor**.

Let me now assess whether there are any **fatal** issues. The paper is a real contribution — it proposes a geometric framework, shows improvements, and provides ablations. The concerns about heuristic influence functions and underspecified noise filtering are significant but don't invalidate the empirical results. No fatal issues.

The paper has genuine strengths: novel geometric perspective, consistent improvements across benchmarks, structured ablation showing VD→CIVD→CIPD improvements, strong ECE gains, and useful robustness analysis.

The main issues are: (1) the theoretical justification for the specific CIVD/CIPD formulation is weak, (2) the PD-based noise filtering is underspecified in the main text and not properly ablated, and (3) the source data requirement for Voronoi site computation limits the TTA setting purity somewhat, though this is partially addressed by Table 4.

For calibration:
- ViDA (CTTA, accepted): scores 6,6,6,8, reasonable method with empirical improvements
- PRO (TTA, rejected): scores 3,3,3,6, just combining existing methods without clear novelty
- PIF (TTA prototype, rejected): scores 5,5,3,5,5, method with approximation concerns
- ATTA (TTA with theory, accepted poster): scores 5,6,8,8, novel setting + theory + good experiments

TTVD has: novel geometric perspective (genuine conceptual contribution), good empirical results, but significant gaps in theoretical grounding of key claims and underspecification of the noise filtering mechanism. It's better than PRO or PIF in terms of novelty and empirical strength, but has more theoretical gaps than ATTA. The empirical improvements are real but modest.

I'd place this paper at around 5-6, leaning toward 5.5 due to the conceptual novelty being undercut by weak theoretical justification.</think>## Summary

TTVD proposes a geometric framework for test-time adaptation (TTA) that formulates neighbor-based TTA methods as Voronoi Diagrams, then extends this foundation through two structures: Cluster-Induced Voronoi Diagram (CIVD), which uses multi-site influence to unify self-supervision and entropy minimization, and Power Diagram (PD), which weights cells to better identify noisy samples near decision boundaries. Experiments on CIFAR-10-C, CIFAR-100-C, ImageNet-C, and ImageNet-R using the TTAB benchmark show consistent improvements over prior TTA methods in both classification error and expected calibration error (ECE).

## Strengths

- **Novel geometric perspective**: The observation that neighbor-based TTA methods correspond to Voronoi Diagrams (Definition 3.1, Eq. 2) is a genuine conceptual insight that provides a principled foundation. The visualization in Figure 1 effectively illustrates the connection between VD partitions and adaptation performance, and the progressive extension from VD → CIVD → CIPD is logically structured.

- **Strong and consistent empirical results**: Table 1 shows TTVD achieves the best results across all four benchmarks on both error and ECE, with particularly notable ECE improvements (3.4%, 1.8%, 4.1%, 4.3% reductions). The improvement over strong recent baselines like SAR is meaningful (1.6% on ImageNet-C error). The ablation in Table 2 demonstrates that each geometric component contributes: VD (28.4%) → CIVD (22.7%) → CIPD (20.5%) on CIFAR-10-C.

- **Standardized evaluation**: The use of TTAB (a peer-reviewed, open-source toolkit) for fair comparison and rigid grid-search increases confidence in the reported results over ad-hoc evaluation pipelines.

- **Practical robustness analysis**: Table 4 shows that TTVD is robust to class mean precision, with virtually no degradation when using only 1% of ImageNet data. Appendix experiments on batch size and label shift effects further support practical applicability.

## Weaknesses

### Fatal
None.

### Major

- **Core influence function design is heuristic and under-justified**: The CIVD influence function (Eq. 4, $F(z, C_k) = -\text{sign}(\gamma) \sum_\alpha (d(\mu_k^{(\alpha)}, z))^\gamma$) and the CIPD variant (Eq. 6) are central to the paper's claimed contributions. However, the choice of the exponent $\gamma$ and the $\text{sign}(\gamma)$ mechanism is asserted without mathematical or empirical justification. The paper claims CIVD "unifies" self-supervision and entropy minimization and "avoids negative transfer since the objective is now unified" (Section 3.2), but this unification claim is not formally established — it replaces two separate losses with a single entropy on an aggregated soft label derived from the influence function, without showing this corresponds to any principled multi-task objective. Without theoretical grounding, the method's central conceptual contribution (the geometric reformulation) is not convincingly connected to the stated goals.

- **PD-based noisy sample filtering is underspecified and insufficiently validated**: Section 3.3 proposes "subtracting the PD from the VD" to identify noisy samples near partition boundaries, but this operation is not concretely defined for high-dimensional spaces. It is unclear whether this means computing cell membership differences (symmetric difference), erosion, or something else. No pseudocode or computational procedure is provided in the main text. Figure 2 illustrates this only on 2D MNIST. Critically, the ablation (Table 2) conflates the PD weighting mechanism and the filtering strategy — there is no experiment isolating PD-based filtering from the CIPD weighting, so the claim that "PD-based boundary shifting improves noisy sample filtering" lacks direct empirical support.

### Minor

- **Incomplete specification of which parameters are updated**: The paper states that "commonly, only the channel-wise affine parameters in normalization layers are updated" (Section 3), but Algorithm 1 updates the full feature extractor $\sigma$. The final algorithm (CIPD) is deferred to Appendix H, and the main text does not clarify whether TTVD updates only BN parameters or a broader set, which matters for fair comparison with baselines constrained to BN-only updates.

- **Ablation study limited to CIFAR-10-C**: Table 2 shows the VD → CIVD → CIPD progression only on CIFAR-10-C. Without ablations on CIFAR-100-C, ImageNet-C, and ImageNet-R, it is unclear whether the 5.7% gain from VD→CIVD and 2.2% from CIVD→CIPD generalize beyond CIFAR-10-C.

- **Fairness concern in comparisons with neighbor-based methods**: TTVD combines self-supervision, entropy minimization, and prototype guidance — it enjoys strictly more adaptation mechanisms than T3A or TAST alone. While TTVD also outperforms Tent and SAR (which do update feature extractors), the claimed superiority over "neighbor-based methods due to their inherent simplicity" could be confounded by capacity differences rather than the geometric framework per se.

- **No error bars or statistical significance tests**: The reported improvements are modest in some cases (0.7% error reduction on CIFAR-100-C and ImageNet-R), and no error bars over multiple runs are provided, making it difficult to assess statistical significance.

- **Architecture diversity**: Only ResNet-26 and ResNet-50 are evaluated. Modern architectures like Vision Transformers (which lack batch normalization) are not tested, and it is unclear whether TTVD generalizes beyond CNNs.

### Trivial
- The abstract claims "remarkable improvements" — the actual improvements (0.7–1.6% in error) are better described as consistent but modest.

## Nice-to-Haves

- Ablation on larger-scale datasets (CIFAR-100-C, ImageNet-C) to verify that VD→CIVD→CIPD improvements generalize beyond CIFAR-10-C.
- More recent TTA methods (EATA, CoTTA, EcoTTA) in the comparison.
- Computational overhead analysis (FLOPs or wall-clock time per batch) comparing TTVD to baselines.
- A separate ablation isolating PD-based filtering from CIPD weighting to validate the noise filtering claim.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"TTVD requires access to source data, undermining TTA setting"**: The Harsh Critic raised this as a major structural concern, but Table 4 explicitly shows TTVD is robust down to 1% of source data, and many TTA methods (including T3A) also use stored prototypes from source data. Furthermore, Lemma 3.1 links PD weights to the classifier weights already stored in the model. The concern is partially valid but overstated as a fatal or even major issue — it is a setting clarification issue, not a fundamental flaw.

- **"Missing related works (EATA, CoTTA, EcoTTA, ViDA)"**: While valid as a nice-to-have, demanding specific recent works is standard scope creep for a paper that already compares against 9 methods using a standardized benchmark.

- **"Only synthetic corruption benchmarks"**: CIFAR-C and ImageNet-C/R are the standard TTA benchmarks. Testing on real-world shifts would strengthen the paper but is not a core flaw.

- **"TTVD is limited to classification"**: This is a scope limitation, not a weakness. The paper explicitly targets classification, and this is standard for TTA research.

- **"Novelty concerns about adapting existing geometric structures"**: CIVD and PD are classical structures, but the insight of connecting them to TTA through the Voronoi Diagram perspective and designing the specific influence functions for TTA constitutes a genuine applied contribution. Dismissing applied novelty because the base structures are classical would undervalue meaningful engineering-conceptual work.

- **"Unclear hyperparameters (γ, τ) not analyzed"**: While hyperparameter sensitivity analysis would strengthen the paper, the experiments follow TTAB's grid-search protocol, and this is a standard critique applicable to most empirical papers.

- **"Reproducibility concerns about missing implementation details in main text"**: The CIPD algorithm is in Appendix H, which is standard for supplementary material. TTAB provides a standardized framework.

## Novel Insights

The most novel observation in this paper is the formal connection between neighbor-based TTA methods and Voronoi Diagrams — specifically that T3A-style prototype classification is exactly equivalent to a nearest-site Voronoi partition. This reframing reveals that the "simplicity" of neighbor-based methods is geometrically equivalent to a single-point-per-class space partition, and that generalizing to multi-site (CIVD) and weighted (PD) partitions can naturally incorporate self-supervision and handle boundary noise. However, the promise of this geometric insight is undercut by the heuristic nature of the influence function design and the underspecification of the PD filtering mechanism.

## Suggestions

- Provide a concrete, algorithmic specification of the PD-based noise filtering in the main text (not just in figures/appendix), including how "diagram subtraction" is computed in high dimensions.
- Add an ablation that directly isolates the filtering mechanism (e.g., CIVD with entropy-based filtering vs. CIVD with PD-based filtering vs. full CIPD) to validate the filtering claim independently of the weighting mechanism.
- Clarify which parameters are updated during TTA (BN affine only, or full feature extractor) and ensure baselines are compared under matching adaptation capacity.

## Score and Decision

**Calibration anchors:**
- ViDA (CTTA, Accept poster, scores 6/6/6/8): Novel method + good empirical results, minor theoretical gaps
- PIF (TTA prototype, Reject, scores 5/5/3/5/5): Prototype-based TTA with approximation concerns, similar domain
- PRO (TTA, Reject, scores 3/3/3/6): Weak novelty, combining existing methods
- ATTA (TTA, Accept poster, scores 6/5/8/8): Novel setting + theory + good experiments

TTVD has genuine conceptual novelty (geometric reframe of TTA) and solid empirical results, but the theoretical grounding of the core influence functions is weak, the PD filtering is underspecified and not properly ablated, and some comparison fairness issues exist. It is stronger than PRO and PIF (clearer novelty, better experiments, structured framework) but weaker than ATTA (which has theoretical guarantees) and weaker than ViDA (which has cleaner presentation and more controlled experiments). The paper's strengths are real but the major weaknesses in justification of the core mechanism prevent it from being a clean contribution.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>