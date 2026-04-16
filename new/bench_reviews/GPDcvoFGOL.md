Now I have a good calibration. Let me synthesize my review.

Key calibration anchors:
- **TextSpan (predecessor paper)**: 8, 8, 8, 8 — oral acceptance. Strong paper with similar approach (interpreting CLIP via text decomposition), but applied to attention heads' direct effects. Clean methodology, good ablations, strong applications.
- **PatchSAE**: 6, 6, 6, 8 — poster. SAE approach for CLIP interpretability. Weaker evaluation, some ad-hoc choices.
- **NeurFlow**: 6, 6, 6, 8 — poster. Neuron grouping for interpretability. Moderate novelty, some issues with evaluation scope.
- **Sparse Feature Circuits**: 8, 8, 8, 8 — oral. Strong causal interpretation method for language models with clean methodology and concrete applications.
- **What do ViTs learn?**: 5, 6, 3, 5 — reject. Empirical analysis with limited novelty and significance.
- **Principled Evaluation Framework**: 5, 6, 6, 3, 5, 5 — reject. Unifying framework without sufficient novelty.

Now let me carefully go through the paper and each reviewer's points to determine what's valid.

The harsh critic raises 5 main issues. Let me evaluate each against the actual paper:

**Issue 1: "What exactly is being interpreted?" — the rank-1 PC vs. actual neuron function.**
The paper clearly states that they approximate φ_n^l(I) with x_n^l(I) * r_n^l + b_n^l, where r_n^l is the first PC. The paper claims this is a good approximation based on Figure 3 ("reconstruction from PC #1"). The harsh critic says this conflates a summary statistic with the neuron's function. 

Looking at the paper, the claim is that "each effect can be approximated by a single direction in the text-image space of CLIP" (Abstract). The paper does show this empirically (Fig 3). However, the critic has a point: the paper doesn't systematically analyze where the approximation fails, or what fraction of neuron behavior is captured. The paper only shows downstream accuracy after replacement—not per-neuron variance explained or failure cases. This is a real gap but not fatal—the approximation works well enough for the applications demonstrated.

However, the claim that "we interpret neurons" when what's actually being interpreted is "the first principal component of the neuron's second-order effect projected through attention value paths" is a meaningful distinction but the paper's framing is common in interpretability work. The paper is transparent about the approximation. I'll include this as a **major** weakness but not fatal.

**Issue 2: "Second-order effects are not truly causal"**
The paper defines φ_n^l(I) as the contribution flowing through the value pathway only, fixing attention patterns. The critic correctly notes this is a partial, linearized decomposition. However, the paper is upfront about this limitation—"We investigated how the neurons flow through individual consecutive attention values, and ignored the effect of neurons on consecutive queries and keys in the attention mechanism" (Section 6). The comparison with indirect effects (Table 1) shows mean-ablation of second-order effects causes a larger accuracy drop (29.6 vs. 52.3 accuracy retained), which the paper interprets as second-order effects being more informative. 

The critic argues this doesn't show second-order is more faithful—just that it's more brittle/low-rank. This is a fair point. The paper's claim that second-order effects "uncover neuron functionality that is obfuscated by self-repair" is somewhat overstated—the comparison shows a linearized decomposition concentrates more of the variance, not that it's more faithful to true causal effect. But the method does still produce useful results for applications. This is a **major** weakness about overclaiming, but not a fatal methodological flaw.

**Issue 3: "Evidence for polysemanticity is anecdotal"**
This is a fair criticism. The paper shows a handful of qualitative examples (Table 2, Fig 5) that demonstrate neurons respond to multiple concepts, but doesn't provide systematic quantification. However, polysemanticity is already established in the literature (Elhage et al., 2022), and the paper's contribution is showing it manifests in a specific way through the second-order lens. The qualitative evidence is suggestive but not rigorous. This is a **minor** weakness.

**Issue 4: "Adversarial evaluation: heavy human filtering and small scale"**
Let me check the paper. The paper states: "We repeat the experiment 3 times and manually remove images that include c_2 objects or do not include c_1 objects." The success rates in Table 3 are measured on the 100 remaining images. The critic notes this biases success upward and the absolute rates are low (5-23%).

This is a legitimate concern. The manual filtering removes failures of the generative pipeline rather than failures of the interpretability method. The paper should report results both with and without filtering. However, the filtering is removing images where the generative model didn't follow the prompt (e.g., didn't include the target class), which is a limitation of the generative model rather than the interpretability method. Still, the claim of "mass-production of semantic adversarial examples" is overstated given these low success rates and manual filtering. This is a **major** weakness.

**Issue 5: "Zero-shot segmentation: ad-hoc and thin validation"**
The paper presents segmentation results in Table 4 with modest improvements over TextSpan (78.1 vs 76.5, 59.0 vs 58.1, 84.9 vs 84.1). The critic notes: no ablations on hyperparameters, and no comparison to actual zero-shot segmentation methods (rather than attribution methods). The improvements are indeed modest and without variance. However, comparing against attribution/explainability methods (as listed in the table) is fair since the paper explicitly frames this as "zero-shot segmentation" built from interpretability methods, not as competing with task-specific segmentation methods. The lack of ablations is a minor weakness, and the "outperforms recent work" claim is technically correct for the methods compared but would be more honest if qualified. This is a **minor** weakness.

Now, the "Spark" reviewer's suggestions:
- Compare against actual segmentation methods: This is outside scope—the paper is comparing against attribution-based methods, which is the natural comparison.
- Scale adversarial evaluation beyond 5 tasks: Fair, but this is a proof-of-concept application.
- Compare neuron descriptions against alternative interpretability methods: This would strengthen the paper but is a nice-to-have.
- Ablate attention heads: Good suggestion.
- Quantify rank-1 approximation quality: Fair criticism, already covered above.
- Analyze sensitivity of PCA directions to dataset: Fair but not essential.

The human finder highlights:
1. Limited evaluation of faithfulness of neuron descriptions — valid, major
2. Ad-hoc design choices without ablation — valid, minor
3. Limited scope of evaluation — valid, minor
4. Limited diversity of text pool — valid but partially addressed with multiple pool sizes, minor
5. Adversarial pipeline failure modes — valid, major
6. Ignoring neuron-neuron interactions — acknowledged by authors, minor

Let me now compose the final review.</think>Now I have enough context to write the final review.

## Summary
The paper introduces the "second-order lens" for interpreting individual neurons in CLIP-ViT, analyzing their effect as it flows through subsequent attention heads (value pathways only) to the output. The authors find these second-order effects are concentrated in late layers, significant for only ~2% of images, and approximately rank-1 in the joint text-image space. They decompose these rank-1 directions into sparse text descriptions to reveal polysemantic neuron behavior, then apply this understanding to two downstream tasks: semantic adversarial example generation and zero-shot segmentation.

## Strengths
- **Well-motivated problem and principled formulation.** The paper correctly identifies that neither direct effects (logit lens) nor indirect effects (mean ablation) faithfully capture neuron contributions in CLIP—direct effects are near-constant and indirect effects are obscured by self-repair. The second-order lens provides a concrete, derivable alternative (Eq. 5) that is tractable and produces meaningful text descriptions.
- **Empirical characterization of second-order effects is informative.** The findings that these effects concentrate in late layers (Fig. 3), are sparse across images (~2%), and are approximately rank-1 (recoverable via a single PC with negligible accuracy loss) are useful observations about how CLIP neurons contribute to the output. The comparison to indirect effects in Table 1 (48.2% vs. 11.0% variance explained by PC1) quantitatively demonstrates that second-order effects are more compressible.
- **Creative downstream applications.** The semantic adversarial pipeline (Section 5.1)—using neuron decomposition to find spurious concept overlaps, then generating adversarial images via LLM + text-to-image—is conceptually novel and demonstrates that mechanistic understanding can be operationalized. The zero-shot segmentation method (Section 5.2) is an elegant reuse of the same neuron-attribution machinery, achieving improvement over prior attribution methods (Table 4).
- **Good use of the joint text-image space.** Leveraging CLIP's own text encoder to decompose neuron output directions into sparse text descriptions is a natural and scalable choice for this model class, providing human-readable labels without manual annotation.

## Weaknesses

### Major:
- **Second-order effects are a partial, linearized decomposition, but are presented as revealing "total contribution" and neuron "function."** The definition in Eq. (5) traces neuron contributions only through the value pathway (OVP matrices) of subsequent attention layers, holding attention patterns fixed. This ignores how modifying a neuron's activation would alter queries and keys, and therefore attention weights, in later layers. The paper acknowledges this limitation (Section 6) but elsewhere describes φ_n^l as "its total contribution to the output, flowing via all the consecutive attention heads" (Abstract, Section 3.2). The comparison in Table 1 against indirect effects is then misleading: the larger accuracy drop from mean-ablating second-order effects (29.6 vs. 52.3 retained accuracy) shows these effects are more brittle and low-rank, not necessarily that they better represent true causal influence. A partial linearization can concentrate variance while missing essential nonlinear pathways. This overclaim does not invalidate the method, but the framing should be adjusted—the paper presents a useful approximation, not a complete causal account.

- **Adversarial evaluation: small scale, manual filtering, and overclaimed "mass production."** Only 5 binary CIFAR-10 tasks are evaluated, with 100 images per task after manual removal of failure cases (where the generative model omitted the target class or included the wrong class). Success rates are low (5.3–22.7%), and no results are reported before filtering, making it impossible to assess end-to-end success. The abstract's claim of "mass-producing semantic adversarial examples" is significantly overstated relative to what the experiments show—at best, the paper demonstrates a proof-of-concept with modest success. Additionally, the baselines (random neurons, indirect effects, similar words) are relatively weak, and it is unclear whether the success comes specifically from second-order neuron analysis rather than from finding correlated concepts in CLIP space by any means.

- **No systematic faithfulness evaluation of neuron descriptions.** The sparse text decompositions are evaluated only through proxy tasks (ImageNet accuracy after reconstruction in Fig. 4, qualitative examples in Table 2/Fig. 5). There is no direct test of whether the assigned text accurately predicts which images activate a neuron, no human evaluation of description quality, and no comparison to alternative interpretability methods (e.g., network dissection, sparse autoencoders, or LLM-based labeling as in Bills et al. 2023). The reconstruction test shows the text directions preserve task-relevant information, but this is insufficient to validate that the descriptions faithfully capture neuron function—directions correlated with a neuron's effect could be semantically unrelated yet still preserve accuracy by proximity in the shared embedding space.

### Minor:

- **Polysemanticity claims are supported only by qualitative examples.** The paper states neurons are polysemantic based on Table 2 and Figure 5, but no quantitative analysis measures how many neurons exhibit this behavior, how many distinct concepts each neuron represents, or how this varies across layers. Since polysemanticity is already established in the literature, the paper's contribution should be showing how it manifests specifically through the second-order lens, which requires more systematic evidence.

- **Zero-shot segmentation improvements are modest and lack ablations.** Gains over TextSpan are 1.6/0.9/0.8 points on pixel accuracy/mIoU/mAP (Table 4), with no variance reported or statistical significance testing. The hyperparameters (top-200 neurons, layers 8-10, threshold 0.5) are not ablated. The comparison is against other attribution methods rather than task-specific zero-shot segmentation methods, which is a reasonable comparison group but should be explicitly scoped.

- **Rank-1 approximation is validated only at the level of downstream classification accuracy.** The paper shows that replacing φ_n^l with its first PC leaves ImageNet accuracy nearly unchanged (Fig. 3), but does not report per-neuron variance explained, distribution of approximation quality across neurons, or examples where the approximation breaks down. This matters because the entire sparse text decomposition pipeline depends on the rank-1 assumption being reasonable for most neurons, not just on average.

### Trivial:
- None.

## Nice-to-Haves
- Systematic comparison of the second-order lens neuron descriptions against alternative interpretability methods (e.g., sparse autoencoders, network dissection).
- Results for adversarial generation *before* manual filtering, to quantify the full end-to-end pipeline failure rate.
- Ablation on segmentation hyperparameters (number of neurons, layer range, threshold).
- Evaluation on additional CLIP architectures beyond ViT-B-32 (ViT-L-14 results appear only in the appendix and are not deeply discussed).
- Per-neuron analysis of where the rank-1 approximation fails.

## Removed Points
These points are flagged to be removed, treat them with caution:

- **"The paper ignores LayerNorm":** The paper explicitly states "Throughout the paper, we ignore layer-normalization terms to simplify derivations. We address layer-normalization in detail in Appendix A.6." While a full discussion in the main text would be preferable, this is not a hidden omission. Softened to a nice-to-have.

- **"The segmentation comparison is against methods not designed for segmentation":** The paper compares to attribution/explainability methods (GradCAM, LRP, TextSpan) that have been used for segmentation in prior work. This is a natural comparison group for a paper that frames its contribution as an interpretability method. This would be scope creep.

- **"The text pool may be insufficient for fine-grained concepts":** The paper tests multiple pool sizes and types (10k words, 30k words, 28k descriptions) and shows convergence at m=128 (Fig. 4). The concern is generic and the paper partially addresses it.

- **"Not evaluated on diverse/fine-grained datasets":** This is a generic scope-broadening request. The paper evaluates on ImageNet (classification), ImageNet-R, CIFAR-10 (adversarial), and ImageNet-Segmentation, which is reasonable for a methods paper of this scope.

- **"The second-order lens only works for late layers":** The paper explicitly shows and discusses this (Fig. 3, Section 3.3). This is a property of the method, not a flaw—the paper explains why (late-layer MLP neurons have significant second-order effects while early ones don't). The criticism that this "fails to provide interpretation for the majority of the network" mischaracterizes the contribution.

## Novel Insights
The key insight—that CLIP neurons' downstream influence flows primarily through attention value pathways rather than directly through the residual stream, and that this influence is both sparse (affecting <2% of images) and approximately one-dimensional in the shared embedding space—is genuinely novel and operationally useful. The observation that first-order effects are near-constant while second-order effects concentrate in late layers (right before the direct-effect-heavy attention layers) provides a complementary picture to prior work on CLIP's direct effects. The adversarial generation application, while modest in results, reveals an interesting connection: when polysemantic neurons write to multiple concepts, the spurious correlation can be exploited for targeted semantic attacks without optimization—this is a novel attack vector enabled by mechanistic understanding.

## Suggestions
- Reframe claims about "total contribution" and "function" of neurons to accurately reflect that φ_n^l is a partial linearized decomposition through value pathways only. This strengthens rather than weakens the paper.
- Report adversarial success rates both with and without manual filtering, and scale to more classification tasks (at least the full CIFAR-10 pair set or a subset of ImageNet classes).
- Add a per-neuron variance-explained histogram for the rank-1 approximation to give a more complete picture of how well the approximation holds across neurons, not just on average.
- Include at least one comparison to an alternative interpretability method (e.g., sparse autoencoder features mapped to text) to validate that the second-order lens provides descriptions that are distinctively useful.

## Score and Decision
**Calibration anchors:**
- TextSpan (predecessor, very similar topic, clean methodology): 8/8/8/8, oral
- PatchSAE (CLIP interpretability, weaker evaluation): 6/6/6/8, poster
- Sparse Feature Circuits (interpretability with strong causal methodology and applications): 8/8/8/8, oral
- NeurFlow (neuron interpretability, moderate novelty, some evaluation gaps): 6/6/6/8, poster
- "What do ViTs learn?" (empirical ViT analysis, limited novelty): 3/5/5/6, reject

This paper is stronger than NeurFlow and PatchSAE in its theoretical formulation and downstream applications, but weaker than TextSpan and Sparse Feature Circuits in evaluation rigor. The core methodological contribution (second-order lens) is novel and well-motivated, and the empirical characterization is genuinely insightful. However, the overclaiming about "total contribution" (while in fact computing a partial linearization), the weak adversarial evaluation with manual filtering, and the absence of faithfulness validation for neuron descriptions are substantive gaps. The paper would need more rigorous evaluation to match the TextSpin-level quality. I place this between NeurFlow/PatchSAE (6) and TextSpan (8), leaning toward the lower end due to the overclaiming and evaluation gaps.

MY FINAL SCORE: <pineapple>6</pineapple>
MY FINAL DECISION: <orange>Accept</orange>