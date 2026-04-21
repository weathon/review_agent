Now I have a thorough understanding of the paper. Let me synthesize the final review.

## Summary

The paper introduces the "second-order lens" for interpreting individual MLP neurons in CLIP-ViT: instead of analyzing direct effects (negligible for MLP neurons) or indirect effects (obscured by self-repair), the authors trace each neuron's contribution through subsequent attention heads to the output. They show these effects are sparse, concentrate in late layers, and are approximately rank-1, enabling a sparse text decomposition via orthogonal matching pursuit. The decompositions reveal polysemantic neurons and are applied to two downstream tasks: semantic adversarial example generation and zero-shot segmentation.

## Strengths

- **The second-order effect formulation is a genuine conceptual contribution.** The derivation in Section 3.2 (Equation 5) cleanly identifies how neuron contributions flow through subsequent attention heads, and Table 1 provides quantitative evidence that second-order effects capture functionality missed by indirect effects (mean-ablation drops accuracy to 29.6% vs. 52.3%, first PC explains 48.2% of variance vs. 11.0%).

- **The characterization of second-order effects is thorough and well-supported.** The three key empirical properties—late-layer concentration, sparsity across images (<2%), and approximate rank-1 structure—are convincingly demonstrated through the mean-ablation experiments in Figure 3. The rank-1 finding is especially consequential as it directly enables the subsequent text decomposition.

- **The adversarial generation framing is a creative way to validate interpretability.** The idea that neuron decompositions should produce testable predictions about model behavior (spurious concept overlaps enabling semantic attacks) is the right kind of validation for interpretability work, even if the current execution has limitations.

- **The qualitative correspondence between text decompositions and top-activating images is compelling.** Figure 5 and Table 2 show convincing alignment—e.g., neuron #4's decomposition includes "snowy," "frost" and its top images contain snow scenes; neuron #2914 encodes both "yacht" and "cabriolet" with matching images of boats and cars.

## Weaknesses

### Fatal

None.

### Major

- **No quantitative evaluation of interpretability quality.** The paper's central claim is that it provides automated interpretation of CLIP's neurons via text descriptions. However, Section 4 evaluates the sparse decomposition only for *functional fidelity*—whether replacing $\phi_n^l$ with the text-approximated version preserves classification accuracy (Figure 4). This tests whether the approximation is functionally adequate; it does not test whether the *interpretation* is meaningful. The paper promises in the Introduction (line 39) that "these concepts correctly track which inputs activate a given neuron," but Section 4 delivers only qualitative visual correspondence and functional reconstruction, not a quantitative test (e.g., measuring correlation between predicted activation from decomposed concepts and actual second-order effect norm). Prior work on automated interpretability (Bills et al., 2023; Oikarinen & Weng, 2023) uses explicit prediction-evaluation protocols; the absence of any such test is a significant gap for a paper whose core contribution is an interpretability method. While the adversarial attack in Section 5.1 provides indirect evidence that the interpretations capture something real, it cannot substitute for direct evaluation of description quality, as the attack's success is also influenced by LLM and text-to-image model capabilities.

- **Adversarial evaluation is confounded by manual filtering and limited scale.** The paper manually removes images containing wrong-class objects or missing correct-class objects (Section 5.1, line 219: "manually remove images that include $c_2$ objects or do not include $c_1$ objects"), which inflates success rates. The paper does not state whether identical filtering is applied with identical stringency to baselines. Additionally, absolute success rates are modest (5–23%, Table 3) and only 5 binary classification pairs from CIFAR-10 are tested. The baselines are weak variants (random neurons, indirect effect, similar words) with no comparison to existing semantic/prompt-based attack methods. For a claim of "mass production" of adversarial examples, this is insufficient evidence.

### Minor

- **The "frozen attention" approximation is acknowledged but untested.** Equation 5 computes the neuron's contribution using actual attention weights $a_i^{l',h}(I)$, which include the neuron's own contribution to attention patterns. If neurons substantially shift attention patterns at later layers, this formulation misses those effects. The paper acknowledges this in Section 6 ("ignored the effect of neurons on consecutive queries and keys") but frames it as future work. An ablation comparing frozen-attention effects against true causal intervention would establish whether this approximation is empirically valid, but its absence is not fatal because the downstream results (adversarial attacks, segmentation) suggest the approximation works reasonably well in practice.

- **Segmentation improvements over TextSpan are marginal and lack variance.** Table 4 shows improvements of +1.6 PixAcc, +0.9 mIoU, +0.8 mAP over TextSpan. No variance or significance is reported, and more recent CLIP-based segmentation methods are not compared. These gains do not independently validate the interpretability method but serve as a secondary application.

- **The first-order-effects-are-constant premise is inherited without independent verification.** The claim that "first-order effects of MLP layers are close to constants" (Section 3.2, line 107) is attributed to Gandelsman et al. (2024) without independent verification, though this is a key premise for motivating the second-order lens. This is minor because the premise is from a closely related prior work by the same group.

### Trivial

None.

## Nice-to-Haves

- Quantitative interpretability evaluation via activation prediction (test whether text decompositions predict neuron activations on held-out images) — this would directly validate the core claim.
- Ablation with oracle text descriptions in the adversarial pipeline to disentangle interpretation quality from generative model quality.
- Analysis of decomposition stability across different text pools, random seeds, or calibration datasets.
- More adversarial task pairs and analysis of what makes the attack succeed on some pairs but not others.
- Failure case analyses showing where the interpretation does not match top-activating images.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Harsh Critic: "polysemanticity is already well-established; the paper confirms it through a new lens but doesn't reveal it."** — The paper never claims to *discover* polysemanticity; it explicitly cites Elhage et al. (2022) and uses "reveal" in the context of revealing it *in CLIP's neurons specifically* via the second-order lens. This is a misread of the paper's framing.

- **Harsh Critic: "Table 1's comparison conflates importance with interpretability."** — The paper correctly uses mean-ablation to demonstrate that second-order effects are more *functionally significant* than indirect effects (a different claim from interpretability). This is explicitly about demonstrating the lens captures something meaningful, not about interpretability per se. The critic is reading ambiguity that isn't there.

- **Harsh Critic: "does not discuss how to choose $m$ (the sparsity level)."** — The paper tests $m \in \{4, 8, 16, 32, 64, 128\}$ systematically in Figure 4. While there's no single recommended value, the scaling behavior is clearly characterized. This is a presentation choice, not a methodological gap.

- **Harsh Critic: "segmentation comparison excludes more recent methods (MaskCLIP, CLIPSurgery, SAM-based approaches)."** — The paper compares against the direct prior work (TextSpan/Gandelsman et al., 2024) and a standard set of explainability methods. Demanding comparison with a broader segmentation literature is scope creep for a paper whose segmentation contribution is secondary.

- **Strength Finder: "The sparse text decomposition is validated quantitatively at scale: Figure 4 shows a clear scaling curve."** — This conflates functional fidelity with interpretability, which is the same conflation the paper itself makes. The scaling curve shows functional reconstruction quality, not that the text descriptions are meaningful interpretations. Moved because this "strength" conflicts with the verified major weakness about lacking interpretability evaluation.

## Novel Insights

The second-order lens represents a genuinely useful structural observation about CLIP-ViT: that MLP neurons' contributions to the output are primarily routed through subsequent attention heads' value projections, and that this routing is approximately rank-1 and image-sparse. This has implications beyond the paper's specific applications—it suggests that CLIP-ViT's late-layer MLP neurons function as "write instructions" to attention heads rather than directly shaping the output, a structural finding that could inform future work on model editing, pruning, and sparse autoencoder design for vision transformers.

## Suggestions

- The single most impactful addition would be a quantitative evaluation of interpretability quality: for each neuron, measure the correlation between (predicted activation from decomposed concepts present in the image) and (actual second-order effect norm), on held-out images. This directly validates the paper's core claim and differentiates functional fidelity from interpretability.
- Report whether manual filtering in the adversarial evaluation is applied identically to all baselines, or better yet, report results both with and without manual filtering.
- Add failure case examples: neurons whose text decompositions don't match top-activating images, or adversarial generations that fail despite apparently correct spurious cues.

---

<context>
**Original reviewer signal**: Harsh Critic considered the paper promising but ultimately unsupported because the core interpretability claim lacks quantitative evaluation of description quality; Strength Finder emphasized the strong empirical characterization (Table 1, Figure 3) and practical applications as validating evidence.

**What was dropped and why**:
- "Polysemanticity is already established / paper doesn't reveal it" — misread; paper cites Elhage et al. explicitly and uses "reveal" in the specific context of CLIP neurons via the second-order lens.
- "Table 1 conflates importance with interpretability" — paper uses Table 1 to show functional significance, not interpretability; no conflation in the text.
- "$m$ selection not discussed" — Figure 4 systematically varies $m$; the scaling behavior is characterized.
- "Missing more recent segmentation baselines" — scope creep; segmentation is a secondary application.
- Strength Finder's "validated quantitatively at scale via Figure 4" — dropped because Figure 4 shows functional fidelity, not interpretability quality; conflicts with the verified major weakness.

**Cross-checks performed**:
- Verified the Introduction's claim about "concepts correctly track which inputs activate a given neuron" (line 39) against Section 4's actual content — Section 4 provides only qualitative correspondence and functional reconstruction, no quantitative activation prediction test. The Introduction's promise is not fulfilled.
- Verified manual filtering language (line 219) — it applies to "the experiment" without specifying whether baselines receive identical treatment.
- Verified that Section 6 acknowledges the frozen-attention limitation but does not test it empirically.

**Severity read**: The surviving major weaknesses are genuine but not fatal. The lack of quantitative interpretability evaluation is the most serious issue — it means the paper's core claim (automated interpretation of neurons) rests on functional fidelity + qualitative examples, without the standard prediction-evaluation protocol used in prior interpretability work. However, the adversarial attack provides *some* indirect evidence that the interpretations capture real structure. The manual filtering concern in the adversarial evaluation is a real confound but partially mitigated by the paper's transparency about it. No single weakness invalidates the core contribution, but the interpretability evaluation gap substantially weakens the paper's central claim.

**Anything else load-bearing**: The paper is by the same group as Gandelsman et al. (2024), which is the direct predecessor (TextSpan). The second-order lens is a natural but non-trivial extension, and the structural findings (rank-1, sparsity, late-layer concentration) are independently valuable regardless of the interpretability evaluation issue. The community norm in automated interpretability (Bills et al., Oikarinen & Weng) does include activation prediction evaluation, so the gap is real against field standards.
</context>