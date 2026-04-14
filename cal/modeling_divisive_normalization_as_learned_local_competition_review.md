=== CALIBRATION EXAMPLE 29 ===

# Final Consolidated Review
## Summary
This paper proposes CH DivNorm, an efficient channel-wise divisive normalization (DN) layer for CNNs inspired by the Carandini & Heeger (2012) model of primary visual cortex. By using non-overlapping channel neighborhoods, CH DivNorm reduces computational complexity from O(N²) (Miller et al., 2022) to O(N), and is shown to improve categorization accuracy and robustness to image corruptions on AlexNet and VGG-16. Additionally, the work demonstrates that in shallow networks, DN can act as a non-linear activation function replacing ReLU, and that training with DN causes color-opponent and orientation-selective filters to self-organize within neighborhoods.

---

## Strengths

- **Concrete and consistent efficiency advantage**: Table 2 shows CH DivNorm halves runtime (~1250 vs. ~2800 sec/epoch) and GPU memory (~1.9 vs. 2.9 GB) compared to Miller et al. (2022) on AlexNet, while matching or exceeding its top-1 accuracy (62.3% vs. 61.3%). This is specific and reproducible.

- **ReLU-free performance in shallow networks is a surprising and clean finding**: The result in Table 1 (rows 4 and 9) — that removing ReLU entirely and relying on DN alone outperforms all ReLU-based variants on both CIFAR-10 (73.07% vs. 69.50% baseline) and CIFAR-100 AlexNet (62.7% vs. 57.9% baseline) — is non-obvious and adds a mechanistic insight: squaring preserves both inhibitory and excitatory signal magnitudes, whereas ReLU discards inhibitory signals.

- **Color-opponent neighborhood emergence is a genuinely novel observation**: While prior work has shown that DN encourages grouping of orientation-selective filters, the visual evidence in Figure 2 that color-opponent cells co-segregate into neighborhoods under DN appears to be new and is biologically motivated. No prior DN work in CNNs, to the reviewers' knowledge, has shown this.

- **Robustness claims are statistically tested**: Unlike many robustness papers that rely on point estimates, Figure 4 applies per-noise-level t-tests across multiple runs, providing an appropriately cautious characterization (acknowledging which noise levels show NS differences).

---

## Weaknesses

### Fatal
None.

### Major

- **Missing Group Normalization baseline**: CH DivNorm partitions channels into non-overlapping groups and normalizes within each — which is structurally very similar to Group Normalization (GroupNorm; Wu & He, 2018). The paper acknowledges GroupNorm in the related work but provides no direct empirical comparison. Without a "GroupNorm with matched P" baseline, it is impossible to determine whether performance gains come from the DN competitive formula (squared numerator, gamma/sigma parameters) or simply from the grouping structure. This is the single most important missing experiment.

- **Limited architectural scope**: All experiments are on AlexNet, VGG-16, and a 2-layer CNN — architectures from 2012–2015. There is no evaluation on ResNet, DenseNet, EfficientNet, or any architecture with skip connections or modern training recipes. The paper argues these architectures are more "anatomically consistent" (Section 5), but this does not make them representative of the settings where DN would be deployed. Without at least one experiment on a modern architecture, the generalizability of the efficiency and robustness claims is untested.

- **Color-opponent filter competition claim is purely qualitative**: The claim in Section 4.3 that "competition between color-opponent cell types" emerges is supported only by visual inspection of Figure 2. The authors themselves acknowledge they "lacked a robust metric to quantify this similarity" (Section 5). For a claim positioned as a key novel contribution, no quantitative measure (e.g., intra-neighborhood vs. inter-neighborhood pairwise cosine similarity, silhouette scores on filter clusters) is provided. The finding is visually suggestive but remains anecdotal.

### Minor

- **Equation 2 description is internally inconsistent with no-ReLU experiments**: Equation 2 explicitly states that $y_i^l(x)$ is "the activity of each neuron at the $i$-th channel *after ReLU*." Yet Sections 4.1 and 4.2 present "CH DivNorm" variants *without* any preceding ReLU (Table 1 rows 4, 9; Table 2 last row). It is never clarified that the no-ReLU variant feeds pre-activation values into Eq. 2. The paper should present the two variants of the equation explicitly to avoid confusion.

- **No sensitivity analysis for neighborhood hyperparameter P**: The paper mentions P in Table 2's caption but never ablates its effect on accuracy, robustness, or the qualitative filter grouping in Figure 2. Figure 2 compares neighborhood sizes 8 and 32 only visually. How sensitive are the performance gains to the choice of P? How was P selected per architecture and dataset?

- **Robustness analysis restricted to AlexNet**: Figure 4 evaluates corruption robustness only on AlexNet. It is unstated whether VGG-16 + CH DivNorm shows similar robustness improvements, which weakens the claim that DN robustness generalizes across model families.

- **Activation map analysis uses very few stimuli**: Figure 3b covers only 5 images, and the claim that these are "representative of those on other stimuli from the dataset" is asserted but not verified statistically. The sharpening effect should be quantified (e.g., edge energy or gradient magnitude across a held-out sample) rather than illustrated on a handful of examples.

- **Training protocol for ImageNet experiments is not disclosed**: The AlexNet baseline achieves only 58.5% top-1 on ImageNet, which is below typically reported values with modern training recipes (~63%). Since all variants are trained identically, the relative improvements remain interpretable, but the absolute numbers raise questions about training hyperparameters and make it harder to compare against numbers reported in other papers. Learning rate schedule, batch size, and augmentation strategy should be reported.

- **Custom perturbations (log, hog) in Figure 4 are undefined in the main text**: Two non-standard corruption types appear in the robustness figure without definition or reference.

- **Non-overlapping vs. overlapping design is asserted, not ablated**: The paper critiques Miller et al.'s overlapping neighborhoods as expensive but does not provide a direct experiment comparing non-overlapping vs. overlapping DN at matched cost or matched performance.

### Tiny

- **±0.00 standard deviations in Table 1** (Row 2) and Table 2 (Miller et al.) are likely rounding artifacts at 2 decimal places but should be reported with higher precision (e.g., ±0.003) to avoid appearing as deterministic results across 10 / 5 runs.

---

## Nice-to-Haves

- **Brain-Score evaluation**: Given the paper's positioning around biological plausibility and V1-like representations, submitting to the Brain-Score benchmark would provide a quantitative test of whether DN improves V1/V2 prediction, which is currently absent. The authors acknowledge this limitation.

- **Gradient flow analysis for ReLU-free failure in VGG-16**: The paper notes (Section 4.2) that CH DivNorm without ReLU fails to converge in VGG-16 but attributes this speculatively to depth. An analysis of gradient norms by depth during training would clarify the mechanism.

- **FLOPs count alongside wall-clock time**: Table 2 reports wall-clock runtime, which may conflate implementation efficiency with algorithmic complexity. Reporting FLOPs would make the computational claim implementation-independent.

- **t-SNE or PCA visualization of first-layer filters colored by neighborhood**: This would make the color-opponent competition claim substantially more convincing and could be computed with existing filter weights.

- **Testing on a residual architecture**: Even ResNet-18 would meaningfully extend the architectural coverage and address whether DN interacts well with skip connections.

---

## Removed Points

*These points were flagged for removal; treat them with caution.*

- **"Abstract overstates the ReLU claim"** (Harsh Critic): The abstract explicitly qualifies the claim: *"in smaller networks, divisive normalization as a non-linear operation eliminates the need for a non-linear activation function like ReLU."* The qualifier is present; the criticism is factually incorrect.

- **"O(N²) claim is misleading"** (Harsh Critic): The paper correctly frames this as a worst-case asymptotic analysis, which is standard practice. The claim is labeled "in the worst case" explicitly in Section 3.3. This is not misleading.

- **Section 3.5 placement criticism** (Harsh Critic): This is a pure organizational style nitpick with no bearing on the paper's correctness or contribution.

- **"Multivariate Gaussian initialization section is disconnected"** (Harsh Critic): The paper is transparent that Kaiming is used for best performance and Gaussian for understanding competition. The section is explicitly scoped as exploratory; criticizing it for not improving performance misreads its stated role.

- **"Miller et al.'s missing VGG-16 results constitute an unfair comparison"**: The absence of Miller et al.'s results on VGG-16 (Table 3) is because their method is computationally intractable and fails to converge — a result *unfavorable to the authors' method's competitor*, not favorable. This absence actually demonstrates a strength of CH DivNorm and is not an unfair comparison.

---

## Novel Insights

The review process highlights one insight not articulated in the paper itself: the absence of a GroupNorm baseline is not merely a methodological gap but a theoretical one. CH DivNorm and GroupNorm are structurally homomorphic in their channel partitioning; the functional difference — z-scoring vs. competitive division — can be tested in isolation. If GroupNorm with the same P matches CH DivNorm, the gain is attributable to grouping alone and the DN formula is incidental. If CH DivNorm consistently outperforms GroupNorm, it provides the first direct evidence that the competitive normalization formula (as opposed to channel grouping per se) is the active ingredient. This comparison would substantially elevate the theoretical contribution of the paper.

---

## Suggestions

1. **Add a GroupNorm baseline with matched P**: This is the most impactful single experiment. Run GroupNorm with the same neighborhood sizes as CH DivNorm on AlexNet ImageNet and report in Table 2.

2. **Clarify Equation 2 for the no-ReLU case**: Either add a second equation explicitly defining the no-ReLU variant, or add a sentence noting that $y_i^l(x)$ represents raw pre-activation responses in the no-ReLU experiments.

3. **Provide a quantitative metric for filter grouping**: Report average intra-neighborhood vs. inter-neighborhood pairwise cosine similarity for first-layer filters to validate the color-opponent competition claim.

4. **Ablate neighborhood size P**: A table or plot showing accuracy as a function of P would establish whether the method is robust to this hyperparameter or requires careful tuning.

5. **Report full training protocol**: Batch size, learning rate schedule, augmentation, and optimizer details for all ImageNet experiments should be provided to enable reproducibility and contextualize the baseline accuracy.

6. **Quantify activation sharpening**: Replace the anecdotal Figure 3 with a quantitative comparison of edge energy or gradient magnitude across a representative sample (e.g., 1,000 validation images).

---

**Evaluation Summary:**

- *Novelty*: Moderate. The DN formulation itself is incremental over Miller et al. and related to GroupNorm. The color-opponent emergence finding is genuinely new if validated, and the ReLU-replacement result is a clean and interesting observation.
- *Technical soundness*: Fair. The formulation is well-defined and the complexity analysis is correct, but a key competing baseline (GroupNorm) is absent, leaving the attribution of gains ambiguous.
- *Empirical support*: Weak-to-moderate. Results are consistent across reported architectures, but the architectures are dated, the filter competition claim is unquantified, and the robustness evaluation covers only AlexNet.
- *Significance*: Limited at ICLR 2025 standards. The bio-inspired ML and computational neuroscience communities would find value in this work, but the lack of modern architecture validation and the GroupNorm gap reduce immediate applicability to mainstream deep learning.
- *Clarity*: Adequate, with one notable inconsistency in the ReLU/no-ReLU formulation description that needs correction.

# Actual Human Scores
Individual reviewer scores: [1.0, 6.0, 3.0, 3.0]
Average score: 3.2
Binary outcome: Reject
