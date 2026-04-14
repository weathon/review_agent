## Summary

This paper introduces a "second-order lens" for interpreting individual MLP neurons in CLIP-ViT, analyzing the flow of information from each neuron through subsequent attention heads to the final output representation—a pathway distinct from both the negligible direct (first-order) effects and the self-repair-confounded indirect (ablation) effects. The authors demonstrate empirically that these second-order effects concentrate in late layers (8–10), are highly sparse (<2% of images per neuron), and are approximately rank-1 in the joint text-image embedding space. Building on this, they decompose each neuron's principal direction into a sparse set of text representations (via OMP), revealing polysemantic behavior. Two applications are demonstrated: generating semantic adversarial examples that exploit spurious concept co-activations, and zero-shot image segmentation that outperforms prior methods on ImageNet-Segmentation.

---

## Strengths

- **Well-motivated interpretability object.** The three-way argument—(1) direct effects are negligible in CLIP, (2) indirect effects are masked by self-repair, (3) second-order effects avoid both pitfalls—is logically tight and backed by concrete quantitative evidence (Table 1: ablating second-order effects drops accuracy to 29.6% vs. 52.3% for indirect effects; first PC explains 48.2% vs. 11.0% variance). This is a specific, non-generic justification for why this particular lens is needed.

- **Clean factorization enabling interpretability.** Equation (5) cleanly separates the second-order effect into (a) input-dependent attention-weighted activations (scalar terms per head/token) and (b) fixed input-independent direction vectors $PW_{VO}^{l',h}w^{l,n}$. This factorization is the key structural insight that makes the rank-1 approximation plausible and the text decomposition tractable, and the paper articulates it clearly.

- **Rank-1 approximability rigorously tested.** The claim that $\phi_n^l(I) \approx x_n^l(I)r_n^l + b_n^l$ is validated through a functional downstream test (Figure 3, "rec. from PC #1"): replacing all second-order effects with their rank-1 approximations causes negligible accuracy drop across all layers and both ViT-B-32 and ViT-L-14. This is a meaningful, not circular, validation.

- **Polysemanticity via mechanistic route.** Prior work (Elhage et al., 2022; Bricken et al., 2023) studies polysemanticity through activation patterns. This paper identifies polysemanticity through the direction a neuron writes to in output space (Table 2, Figure 5), which is a more functionally grounded characterization: the neuron's multi-concept behavior is observed in what it *outputs*, not just what activates it.

- **Applications connect interpretation to action.** Unlike most interpretability work that stops at description, this paper uses the neuron decompositions to (a) find spurious concept couplings for adversarial generation and (b) select class-relevant neurons for segmentation—demonstrating that the interpretability method produces actionable understanding.

---

## Weaknesses

### Fatal
None.

### Major

- **The rank-1 approximation lacks mechanistic explanation.** The entire downstream pipeline—text decomposition, polysemanticity analysis, adversarial mining, segmentation—depends on $\phi_n^l(I)$ being approximately rank-1. The paper demonstrates this empirically via downstream accuracy (Figure 3), but provides no explanation for *why* it holds. From Eq. (5), the image-to-image variation in $\phi_n^l(I)$ comes from the product of scalar activations $p_i^{l,n}(I)$ and scalar attention weights $a_i^{l',h}(I)$, weighted by fixed direction vectors. Whether this product over $i$, $l'$, $h$ remains rank-1 is non-trivial and depends on correlated structure in the $W_{VO}$ matrices—structure that is not analyzed. The absence of this analysis means we cannot tell whether rank-1 behavior is a principled structural property of CLIP or a coincidence of scale. An intuitive or partial explanation would substantially strengthen the contribution.

- **Adversarial success rates are absolutely low and selection criterion is unstated.** Table 3 reports 5.3–22.7 images out of 100 fooling the classifier (5 class pairs), well below what "mass production" implies for a practical attack. Only 5 out of 45 possible CIFAR-10 pairs are evaluated, with no criterion given for selection. The binary classification setting (50% random baseline) also substantially simplifies the task vs. full 10-class or ImageNet-scale evaluation. Taken together, these design choices make it difficult to assess whether the adversarial utility generalizes.

- **Manual filtering in adversarial experiments is not quantified.** The paper states: "we manually remove images that include $c_2$ objects or do not include $c_1$ objects." The pre-filtering success rate is never reported. If the text-to-image model frequently generates $c_2$ objects (the target class that should be absent), removing them inflates the reported success rate by discarding easy-to-identify failures. The number of images removed per task and run should be reported to properly interpret Table 3.

### Minor

- **Segmentation improvements are modest and lack significance testing.** The gains over TextSpan are +1.6pp pixel accuracy, +0.9pp mIoU, +0.8pp mAP (Table 4). Given that both methods use the same CLIP backbone and share design choices, these small absolute differences could plausibly fall within run-to-run or evaluation variance. No confidence intervals or significance testing are provided.

- **Linearization approximation (ignoring neuron effects on QK circuits) is acknowledged but not bounded.** The paper correctly notes in Section 6 that it ignores how neuron outputs modify subsequent query/key computations. While this limitation is acknowledged, no estimate is given of how large this error could be. The aggregate accuracy metric in Figure 3 could mask substantial per-neuron errors that cancel in the mean. A small-scale intervention study on a subset of neurons would help characterize this.

- **The "top 200 neurons from layers 8-10" hyperparameter for segmentation is not ablated.** There is no sensitivity analysis varying the number of neurons (e.g., 100 or 300) or the layer range. Given the marginal improvement over TextSpan, it is unclear whether this choice is principled or tuned to the evaluation set.

- **Table 1 comparison (second-order vs. indirect effects) is only for layer 9 in ViT-B-32.** This is the core empirical argument for why second-order effects are preferable. The appendix shows ViT-L-14 results for ablation magnitude (Figure 10) but the variance-explained comparison in Table 1 is reported for a single layer of a single model, which weakens the generality claim.

### Tiny

- **"Second-order" terminology.** The term "second-order" in the mechanistic interpretability literature (Elhage et al., 2021) often denotes 2-hop attention-to-attention composition (virtual weights). Here it means neuron → downstream attention values → output, which is a different hop structure. The paper defines its usage, but a one-sentence clarification of the terminological choice vs. prior usage would avoid reader confusion.

- **Qualitative neuron examples are illustrative, not systematic.** Table 2 and Figure 5 showcase 4 neurons. These appear to be selected for clean illustrative polysemanticity. While Figure 13 in the appendix shows top-50 activating images for these same neurons, there is no quantification of what fraction of all neurons exhibit the degree of polysemanticity shown.

---

## Nice-to-Haves

- **SAE/sparse autoencoder baseline for text decomposition.** Sparse autoencoders (Bricken et al., 2023) are now a standard approach for disentangling polysemous neurons. A comparison of OMP sparse text decomposition against SAE-derived features would help situate the method in the current landscape. The paper acknowledges SAEs in related work but does not evaluate against them.

- **Causal validation of the text direction.** The identified text direction $r_n^l$ is shown to have high reconstruction accuracy, but ablating $r_n^l$ directly in the output space to verify it causally removes the neuron's conceptual contribution would strengthen the mechanistic claim.

- **Prompt engineering baseline for adversarial attacks.** A simpler baseline—directly prompting the text-to-image model with "a $c_1$ that looks like $c_2$" or "a $c_1$ with $c_2$-like features"—would clarify whether the neuron-derived spurious concepts provide uplift beyond what a human creative prompter could achieve without any mechanistic analysis.

- **Polysemanticity statistics across the neuron population.** Report what fraction of all neurons (not just the 4 illustrated) exhibit multi-concept decompositions, and the distribution of concept-cluster counts. This would allow assessment of how widespread polysemanticity is in CLIP's MLP layers.

- **Segmentation sensitivity analysis.** A curve of mIoU vs. number of neurons selected would clarify whether the +0.9pp gain is robust across a range of hyperparameter choices.

- **Attention-head-level attribution.** Identifying *which* downstream attention heads carry the majority of a given neuron's second-order signal would complete the mechanistic picture and might reveal interpretable head-neuron relationships.

---

## Removed Points

*These points are flagged for removal; treat them with caution.*

- **"In-distribution advantage" of ImageNet class descriptions.** The critic argues that using ChatGPT-3.5-generated ImageNet descriptions biases the evaluation toward ImageNet. However, the paper explicitly evaluates all three pools (10k words, 30k words, IN descriptions) and shows the gap vanishes at large $m$. The ImageNet descriptions are simply a convenient pool that converges faster; using them as one condition among several is not misleading.

- **Rank-1 finding may be a trivial artifact of the projection bottleneck.** The Spark Finder raises the concern that rank-1 behavior might be a trivial consequence of the final projection matrix $P$. However, $P$ is full-rank (maps $d'$-dimensional ViT output to $d$-dimensional CLIP space). The rank-1 behavior is observed in the $d$-dimensional output space, and the paper's approximation operates in that space directly. The bottleneck argument does not trivially explain the finding.

- **"Second-order terminology" as a major problem.** The paper defines its use of "second-order" explicitly. The difference from Elhage et al.'s "virtual weights" is a matter of terminological convention, not error. This is a tiny note at most.

- **Requesting confidence intervals for Table 4.** Single-run evaluation is the norm for zero-shot segmentation benchmarks on ImageNet-Segmentation with deterministic methods. However, given the marginal gains, at least reporting whether results are from a single run or average is worth noting—retained as the minor weakness on significance testing.

- **Claiming the "similar words" baseline undermines the method.** In Table 3, "similar words" performs at 3.3–5.0% for "dog→deer" and "bird→frog," while the second-order method achieves 22.7% and 8.0% respectively. The second-order method dominates consistently. No criticism warranted.

---

## Novel Insights

The most conceptually novel contribution beyond the paper's own stated claims is the structural consequence of factoring neuron contributions through OV circuits: because the directional part of the second-order effect ($PW_{VO}^{l',h}w^{l,n}$) is input-independent, the entire image-dependent variation collapses to a scalar weighting of fixed directions across all attention heads and tokens. This is what makes the rank-1 approximation non-trivially plausible, and it implies that *the semantic identity of a neuron is encoded jointly in its output weight vector $w^{l,n}$ and the downstream OV circuits*, not in its activation pattern per se. This framing suggests that neuron "polysemanticity" in CLIP may be structurally intertwined with the OV matrix geometry—a direction worth pursuing. Additionally, the finding that adversarial vulnerability arises directly from neuron polysemanticity (concepts spuriously co-encoded in the same neuron's direction) provides one of the clearest mechanistic accounts to date of why semantic adversarial examples exist in contrastive vision-language models.

---

## Suggestions

1. **Quantify the linearization error.** Select a random sample of ~20–50 neurons from layer 9 and directly measure how much the true second-order effect (computed with full, non-linearized forward passes) deviates from the linearized Eq. (5). Report the mean and distribution of this error. Even a coarse bound would substantially increase confidence in the derivation.

2. **Provide pre-filtering adversarial numbers.** In Table 3, report the success rate *before* manual filtering alongside the post-filtering numbers. Also report, for each task, the fraction of generated images that were removed for containing $c_2$ objects. This makes the adversarial pipeline fully reproducible and interpretable.

3. **Expand adversarial evaluation.** Evaluate on at least 15 of 45 CIFAR-10 pairs using a stated selection criterion (e.g., all pairs with differing super-categories, or all pairs alphabetically). Report aggregate statistics across pairs to avoid cherry-picking concerns.

4. **Add significance testing for segmentation.** Use bootstrap resampling over the ImageNet-Segmentation test images to produce confidence intervals for the mIoU and pixel accuracy gains. Given the small absolute delta, this is necessary to establish whether the improvement is reliable.

5. **Offer an intuition for rank-1 behavior.** Add a brief analysis—even informal—of why the sum $\sum_{l',h,i} (p_i^{l,n} a_i^{l',h}) \cdot (PW_{VO}^{l',h}w^{l,n})$ concentrates in one direction. For instance: do the fixed vectors $PW_{VO}^{l',h}w^{l,n}$ across different $(l',h)$ pairs tend to align? If so, show a small correlation analysis. This single addition would give the empirical rank-1 finding a principled foundation.

---

**Novelty:** High within mechanistic interpretability for multimodal models. The second-order lens is a distinct and non-obvious interpretability object for CLIP neurons.

**Technical soundness:** Moderate-to-good. The derivation is correct under stated approximations, and the core empirical claims are backed by functional validation. The unquantified linearization error and the unexplained rank-1 structure are genuine gaps.

**Empirical support:** Adequate for the characterization claims (sparsity, rank-1, polysemanticity) but weak for the adversarial application (low absolute rates, unquantified filtering, limited task coverage) and marginal for segmentation (small gains, no significance test).

**Significance:** Meaningful for the mechanistic interpretability community and potentially impactful for understanding CLIP's failure modes. The connection between neuron polysemanticity and semantic adversarial vulnerability is a genuinely useful insight.

**Clarity:** Good overall. The paper is well-organized, and the central ideas are explained clearly. The limitations section is honest and appropriately scoped.

MY FINAL SCORE: <pineapple>6.2</pineapple>