## Summary
SPARK is a physics-guided augmentation framework for dynamical system modeling that addresses data scarcity and distribution shift. It builds a vector-quantized discrete memory bank conditioned on boundary information and physical parameters, augments training samples by mixing latent representations with retrieved codebook entries, and employs a Fourier-enhanced graph ODE for long-horizon prediction. Experiments span five benchmarks (Prometheus, ERA5, Navier-Stokes, Spherical-SWE, 3D Reaction-Diffusion) and include OOD and cross-domain transfer evaluations.

---

## Strengths

- **SPARK genuinely functions as a plugin, supported by concrete evidence.** Figure 1 shows backbone+SPARK outperforming backbone across ViT, CNO, U-Net, and NMO on ERA5. Table 3 demonstrates SPARK applied to three distinct backbones (SimVP, PredRNN, Earthfarseer) with consistent improvements on SEVIR under varying data fractions. This is specific, multi-backbone validation that most augmentation papers do not provide.

- **Energy spectrum evaluation goes beyond standard MSE.** Figure 6 compares SPARK, Swin-T, and FNO on power spectra for Navier-Stokes, Spherical-SWE, and 3D Reaction-Diffusion. This is a physically meaningful diagnostic—important for the SciML community—and SPARK visibly recovers high-frequency structure that FNO and Swin-T miss.

- **Cross-domain transfer experiment with controlled data fractions is a strong evaluation design.** Table 3 systematically varies target-domain data from 20% to 100% while transferring from ERA5 to SEVIR, and shows that SPARK+backbone consistently outperforms backbone alone (whereas vanilla backbone transfer can actually degrade performance at high data fractions). This directly tests the stated data-scarcity motivation.

- **Benchmark diversity is above average.** Five datasets spanning synthetic PDEs (Navier-Stokes, Spherical-SWE, 3D Reaction-Diffusion) and real meteorological data (ERA5, SEVIR), with both OOD and non-OOD splits, is a thorough empirical scope for a single paper.

---

## Weaknesses

### Fatal
*None identified, but the combination of Major weaknesses below would significantly undermine confidence in the results without revisions.*

### Major

- **Complete absence of ablation study.** The method has four distinct components: boundary/parameter injection (§3.2), the VQ memory bank (Eq. 5–6), the memory-bank augmentation (Eq. 7), and the Fourier-enhanced graph ODE (§3.3). Table 1 only reports "Ours + SPARK" as a monolithic system; there is no table removing one component at a time. It is therefore impossible to determine whether the gains come from the augmentation mechanism (the paper's core claim), the bespoke downstream predictor, or some combination. This is the single most critical omission for an ICLR submission making component-level claims.

- **Unexplained ERA5 baseline discrepancy undermines comparison fairness.** In Table 1, FNO, UNO, and CNO report MSE of 0.7233, 0.6652, and 0.5243 on ERA5 (w/o OOD), while ViT achieves 0.0762 and NMO achieves 0.0432. Neural operators are specifically designed for PDE problems and typically outperform generic vision backbones on such tasks; a 20× gap versus vision models is anomalous and unexplained. Possible causes (unsuitable preprocessing, incompatible resolution, poor tuning) are never discussed. Until clarified, the ERA5 results cannot be trusted, which affects the headline comparison.

- **Augmented sample label preservation is conceptually unjustified.** Eq. (7) produces augmented inputs by interpolating a sample's latent representation with Top-K codebook entries from potentially different physical environments. The corresponding output label $\mathcal{Y}_i$ is the original future state, implicitly assumed to be preserved under this mixing. However, if the retrieved codes represent different boundary conditions or parameter regimes, the future dynamics associated with the mixed input could differ substantially from the original label. The paper does not address this, and it is a fundamental question for the validity of the augmentation scheme.

- **OOD split definitions are absent for all datasets.** Table 1 reports w/o OOD and w/ OOD numbers, and Table 4 repeats this, but the paper never specifies what constitutes the OOD condition for each benchmark—whether it is an unseen parameter range, unseen boundary condition, unseen temporal period, or some combination. The near-zero OOD degradation for SPARK on ERA5 (0.0322 → 0.0321) raises the question of whether the OOD split is sufficiently challenging. Without defining the shift, the OOD claim cannot be evaluated.

- **No data scarcity experiments on primary benchmarks.** The abstract and introduction prominently motivate data scarcity, yet Table 1 uses full training data for all five benchmarks. The only scarcity experiment is Table 3, which conflates limited data with cross-domain transfer. A direct evaluation varying training data fraction (e.g., 10%, 25%, 50%) on Prometheus or Navier-Stokes is needed to validate the stated motivation.

### Minor

- **Notation conflicts impede reproducibility.** The symbol $\delta$ denotes physical parameters in Eq. (2) but appears as an activation/transform in Eq. (8) without redefinition. Similarly, Eq. (5) assigns $z_i$ via argmin (making it an index), yet it is described in the text as "the nearest neighbor code embedding." These inconsistencies are not merely cosmetic; they make it difficult to implement the method correctly.

- **No VQ codebook diagnostics.** VQ-VAE methods are known to be prone to codebook collapse where only a small fraction of codes are ever used. The paper employs the standard straight-through estimator (sg[·]) but reports no codebook utilization, perplexity, or dead-code percentage. If the memory bank is degenerate, the augmentation mechanism would not function as claimed.

- **Figure 5 (sea ice RQ2) is self-referential.** The quantitative evidence for RQ2 consists of SPARK's own training loss, SSIM, and PSNR curves over 80 epochs. These curves demonstrate that the model trains successfully but provide no comparison against baselines on the sea ice task. The qualitative Figure 4 compares against FNO and U-Net, but not against NMO (the strongest baseline in Table 1).

- **Physical parameters are ill-defined for ERA5.** §3.2 conditions channel attention on "physical parameters $\delta$" (scalars like viscosity). For ERA5, the paper says it uses u, v, and humidity as "forcing terms"—these are dynamic spatiotemporal fields, not scalar parameters in the same sense as viscosity or diffusion coefficient. How these are projected into a single parameter vector for Eq. (2) is not explained.

- **"Quantitative" vs. "quantized" title inconsistency.** The paper title says "quantitative augmentation" while the abstract, methodology, and conclusion consistently use "quantized augmentation" (referring to VQ-VAE). These are different concepts. The title should match the method.

### Tiny

- The theoretical section (§3.4) presents standard information-theoretic and PAC-Bayes generalization bounds with physical prior $\mathcal{P}$ substituted in. The conclusion that "physical priors reduce $I(\theta; \mathcal{D} | \mathcal{P})$" is stated as an implication of the theorem, but it is actually an assumption. The theorems do not analyze VQ discretization, the augmentation rule, or the Fourier graph ODE, and are not connected to any measurable quantity in the experiments. The theory as written does not add analytical insight specific to SPARK.

---

## Nice-to-Haves

- **Compare augmentation against simple baselines (MixUp, noise injection).** The physics-guided memory bank mixing is the central novelty over standard augmentation; a direct comparison would strengthen the claim.

- **Sensitivity analysis for key hyperparameters** ($\lambda$, $K$, memory bank size $M$). These are central to the augmentation behavior and their robustness across datasets is unknown.

- **Training and inference time alongside Table 2.** Table 2 explores model size but omits wall-clock training/inference time and memory overhead. Since the method adds VQ pretraining, GNN encoding, and ODE solving on top of any backbone, cost transparency would help practitioners.

- **Codebook visualization (t-SNE/UMAP colored by physical parameter).** Visualizing whether the learned codes organize by physically meaningful axes would provide qualitative evidence that the memory bank captures physics rather than memorizing training samples.

- **Controlled OOD severity analysis.** Stratifying OOD splits by shift magnitude (e.g., small vs. large viscosity extrapolation) would make the robustness claims more precise and informative.

---

## Removed Points
*These points are flagged to be removed; treat them with caution.*

- **"Plugin claim is unvalidated"** (Harsh Critic, strong form): Removed. Figure 1 explicitly shows backbone+SPARK vs. backbone for multiple architectures on ERA5, and Table 3 applies SPARK to SimVP, PredRNN, and Earthfarseer. The plugin is validated, though the main Table 1 does not isolate it — which is a fair ablation concern but different from the plugin claim itself.

- **Missing error bars / no statistical tests** (Harsh Critic): Removed. Single-run evaluation is the norm for large-scale benchmarks in this community. Multiple-run statistics are not standard practice for neural operator and dynamical systems benchmarks at ICLR scale.

- **Ethics statement is perfunctory** (Harsh Critic): Removed. This is a style/formatting concern and ICLR does not mandate extensive societal impact statements for applied ML papers.

- **Requests for missing related work citations** (Harsh Critic): Removed per instructions; external references cannot be verified.

- **Title does not signal technical novelty precisely enough** (Harsh Critic): Removed as a pure style nitpick. The title issue kept is the substantive "quantitative" vs. "quantized" semantic confusion.

- **Strength: "comprehensive empirical evaluation" / "well-written"** (Reviewer 2): Removed as generic. The specific strength retained is the breadth and design of the transfer experiment (Table 3), not generically "extensive experiments."

---

## Novel Insights

The most underappreciated aspect of the paper is the interaction between the physics-aware discrete memory bank and transfer learning. Table 3 reveals a striking asymmetry: vanilla backbone transfer from ERA5 to SEVIR **hurts** performance at higher data fractions (e.g., SimVP degrades 15.79% at 100% SEVIR, PredRNN degrades 8.70% at 100%), while SPARK+backbone transfer consistently helps. This suggests that SPARK's physics-conditioned quantization may act as a domain-invariant regularizer—filtering out ERA5-specific distributional artifacts that would otherwise cause negative transfer—rather than simply providing more training signal. This mechanism is not analyzed in the paper but is worth investigating: if true, it would explain why SPARK's benefit is largest in low-data regimes and implies a specific use case beyond generic augmentation.

---

## Suggestions

1. **Add an ablation table** removing each of the four components (boundary encoding, parameter channel attention, VQ memory bank, Fourier graph ODE) on at least Prometheus and Navier-Stokes. This is the highest-priority revision.

2. **Explain or fix the ERA5 FNO/UNO/CNO results.** If these operators genuinely perform poorly on this ERA5 formulation (e.g., because the task is on irregular grids), explain why and note that the comparison is one-sided in SPARK's favor for those methods — this actually makes SPARK's advantage over NMO more meaningful.

3. **Define OOD splits precisely** in the experimental section or appendix: for each dataset, state what physical parameter ranges, boundary conditions, or temporal windows are held out, and quantify the shift magnitude.

4. **Report codebook diagnostics** (% active codes, assignment entropy, average nearest-neighbor distance in codebook space) to validate that the VQ memory bank is functioning as intended rather than collapsing.

5. **Add at least one direct data scarcity curve** (training fraction vs. MSE) on a primary benchmark (e.g., Prometheus) to directly substantiate the data-scarcity claim from the abstract.

6. **Justify label preservation** in the augmentation (Eq. 7): either argue theoretically that interpolation in the learned physics-conditioned latent space preserves output labels, or test empirically that augmented samples have lower prediction error than purely random latent interpolations.

7. **Fix the $\delta$ notation conflict** between §3.2 (physical parameters) and Eq. (8) (activation function), and reconcile the $z_i$ index vs. embedding ambiguity between Eq. (5) and the surrounding text.

---

## Evaluation

| Axis | Assessment |
|---|---|
| **Originality** | Moderate. The combination of VQ-VAE memory bank with physics-conditioned attention and graph ODE for augmentation in dynamical systems is novel in its integration, though each individual component (VQ, GNN, Neural ODE, Fourier operator) is well-established. The augmentation-for-OOD framing in physical systems is a fresh angle. |
| **Importance of research question** | High. Robustness to distribution shift and data scarcity in physical system modeling are genuine bottlenecks in scientific ML with real deployment consequences. |
| **Claims well supported** | Weak to moderate. The headline performance gains (Table 1) are plausible but tainted by unexplained baseline discrepancies (ERA5 operators) and complete absence of ablations. The plugin claim is supported. The data scarcity claim is not directly demonstrated on primary benchmarks. |
| **Soundness of experiments** | Moderate concern. The five-benchmark comparison is broad, and the transfer experiment is well-designed. However, missing ablations, undefined OOD splits, and the ERA5 operator anomaly prevent confident interpretation. |
| **Clarity of writing** | Adequate but with specific notation conflicts that impede reproduction (δ overloading, z_i as index vs. embedding, unclear spectral convolution definition on irregular graphs). |
| **Value to the research community** | Moderate to high if ablations confirm that the VQ augmentation is the driver of gains. The transfer learning finding (SPARK mitigates negative transfer) is potentially the most valuable and least discussed result. |
| **Contextualized relative to prior work** | Adequate at a high level; the paper covers the right related work categories. More discriminative positioning against retrieval-augmented and memory-based scientific ML would strengthen the novelty argument. |