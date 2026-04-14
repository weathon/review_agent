## Summary
SPARK proposes a physics-guided vector-quantized memory bank for augmenting dynamical system training data, combined with a Fourier-enhanced graph ODE for long-horizon forecasting. The central idea is to pre-train a discrete codebook enriched with boundary conditions and physical parameters, then use nearest-neighbor retrieval in the codebook to create physics-consistent augmented samples, thereby improving robustness to data scarcity and distribution shift. The method is evaluated across five PDE/weather benchmarks and shows strong improvement over neural-operator and vision-backbone baselines.

---

## Strengths

- **Physics-guided discrete augmentation is a genuinely novel framing.** While VQ-VAEs and physics-informed neural networks both exist, fusing boundary positional encodings and physical-parameter channel attention into a shared discrete codebook, then using that codebook specifically for latent-space data augmentation, is a distinct contribution not present in prior operator-learning or scientific ML literature. The motivation—that discrete prototypes over physics conditions provide a structured interpolation space for OOD generalization—is concrete and falsifiable.

- **Demonstrated plugin utility across multiple backbones.** Figure 1 (ERA5 radar chart across ViT, CNO, U-Net, SwinT, NMO) and Table 3 (SimVP, PredRNN, Earthfarseer + SPARK on SEVIR transfer) provide direct evidence that SPARK improves diverse backbone architectures without architectural modification. This partially substantiates the plugin claim and sets SPARK apart from papers that only propose end-to-end models.

- **Breadth and difficulty of benchmark coverage.** Five heterogeneous datasets (Prometheus CFD, ERA5 atmospheric, Navier-Stokes, Spherical-SWE, 3D Reaction-Diffusion) plus a challenging sea ice transfer task represent a genuinely demanding evaluation regime for dynamical system modeling. Including both synthetic PDE and real-world meteorological data strengthens the generality claim.

- **Near-zero OOD degradation on ERA5.** SPARK's degradation from in-distribution to OOD on ERA5 (0.0398 → 0.0401, Table 4) is remarkably small compared to competing OOD-specific methods (LEADS: 0.2367 → 0.4233; CODA: 0.1233 → 0.2367). If the OOD protocol is appropriately challenging, this result is a compelling empirical signal.

---

## Weaknesses

### Fatal
*None identified.*

### Major

- **No ablation studies.** The method combines at least four distinct components: (1) boundary positional encoding, (2) physical-parameter channel attention, (3) VQ memory bank augmentation, and (4) Fourier-enhanced graph ODE. No ablation table appears in the paper. This means the improvement observed in Table 1 cannot be attributed to any specific component. It is equally consistent with the result that the Fourier-enhanced graph ODE alone explains nearly all gains and the augmentation plugin contributes little. This is the single most important missing experiment for an ICLR submission about a plugin augmentation method.

- **OOD evaluation protocol is undefined.** The paper reports "w/ OOD" and "w/o OOD" across all tables but never specifies what is shifted: which physical parameters are out-of-range, by how much, whether shifts are interpolative or extrapolative, and whether OOD test environments contain unseen boundary geometries, unseen parameter values, or unseen time windows. Since OOD robustness is the central empirical claim, this omission makes the results difficult to interpret or reproduce, and prevents assessing whether the challenges are actually challenging.

- **Numerical inconsistency across tables undermines reproducibility.** SPARK's Prometheus MSE is reported as 0.0294/0.0308 (w/o OOD / w/ OOD) in Table 1 but as 0.0323/0.0328 in Table 4. Similarly, SPARK's ERA5 numbers differ between Table 1 (0.0322/0.0321) and Table 4 (0.0398/0.0401). These are not within rounding error and are unexplained. Either different model variants, data splits, or hyperparameter settings are used—none of which are disclosed. This raises questions about selective reporting and significantly weakens confidence in both tables.

- **The augmentation target semantics are undefined.** Equation 7 defines augmented latent inputs v_i by mixing node embeddings with codebook entries. The paper states these augmented samples are added to the training set, but never specifies the corresponding prediction target Y_i. If the original target is reused unchanged, this requires a justification that the augmented latent is label-preserving under physics-guided interpolation. Without this, the theoretical rationale for why augmented samples improve generalization rather than corrupt it is missing.

- **Suspicious ERA5 baseline performance.** Table 1 shows FNO (0.7233/0.9821), UNO (0.6652/0.7621), and CNO (0.5243/0.7821) having dramatically higher MSE on ERA5 than NMO (0.0432/0.0563) or the authors' method (0.0322/0.0321). These order-of-magnitude gaps for well-known methods on a standard atmospheric dataset—without any explanation of hyperparameter tuning, input normalization, or task adaptation—suggest potential misconfiguration of baselines. If these methods were not properly adapted to the ERA5 task (e.g., different input resolution, normalization, or rollout horizon), the reported improvement is inflated.

### Minor

- **Equation 7 normalization is inconsistent with K.** The augmented representation is $v_i = \lambda h_i + (1-\lambda)\sum_{n=1}^K e_n$. As written, the sum of K codebook entries is not normalized by K, so the scale of the second term grows with K. This is likely a typo (should be $\frac{1}{K}\sum$), but as stated it makes the interpolation invalid for K > 1.

- **Equation 8 notation is under-specified.** The symbol δ in $q_i = \frac{1}{T_0}\sum_t \delta(\alpha_i^t \cdot v_t^i)$ is never defined (activation function? identity?), and the attention scores $\alpha_i^t$ lack normalization, making it unclear how the weighted average is computed. In Equation 9, $H^l$ appears in the ODE derivative but is not defined in this section; it is unclear whether it refers to layer-wise hidden states or historical observation embeddings.

- **Sea ice section lacks comparative quantitative evaluation.** Section 4.3 shows SPARK's training convergence curves (Figure 5) but does not report a comparison table of SPARK vs FNO vs U-Net on the sea ice task in terms of MSE, SSIM, or PSNR. The qualitative Figure 4 is suggestive but insufficient for a quantitative claim of superiority on this challenging task.

- **Theoretical analysis is generic.** Theorems 1 and 2 are a standard mutual-information generalization bound and a PAC-Bayesian bound. Neither theorem depends on vector quantization, the specific augmentation formula (Eq. 7), the Fourier-enhanced ODE, or any property of the proposed architecture. The key step—showing that SPARK specifically reduces $I(\theta; \mathcal{D} | \mathcal{P})$ or $\text{KL}(Q\|P)$—is asserted rather than proved. As written, the theory justifies "any physics-informed prior helps" rather than "SPARK's design helps."

### Tiny

- The paper title says "Quantitative Augmentation" while the abstract and body consistently use "Quantized Augmentation." The former is an incorrect description of the method.
- The scalability claim (Table 2) shows ERA5 MSE rising from 0.0302 to 0.0391 when model size drops from 24.56MB to 2.18MB—a 29% increase. Describing this as "stable" performance is overstated; the degradation is monotonic and non-negligible.

---

## Nice-to-Haves

- **Sensitivity analysis on λ, K, and memory bank size M.** These three hyperparameters directly control the augmentation behavior, yet no analysis of sensitivity is provided. Even a small grid search over λ ∈ {0.1, 0.3, 0.5, 0.7} and K ∈ {1, 5, 10} would indicate whether the method is robust or tightly tuned.
- **Memory bank interpretability.** A t-SNE/UMAP of learned codebook entries colored by physical parameters (e.g., viscosity bins, boundary condition type) would directly test whether the discrete codes correspond to physically meaningful regimes—this would strengthen the "physics-guided" framing.
- **Controlled data scarcity experiments on primary benchmarks.** The transfer experiment (Table 3) tests data scarcity in a cross-domain setting. Systematically training with 5%, 10%, 20%, 50% of Prometheus or Navier-Stokes data and comparing with and without SPARK would more directly validate the data-scarcity claim.
- **Computational cost comparison.** The method stacks a VQ-VAE pretraining stage on top of a Fourier graph ODE. A table comparing training/inference time and GPU memory against FNO/NMO baselines would clarify whether the improved accuracy justifies the added cost.
- **Augmentation visualization.** Showing a decoded augmented sample (decoded from $v_i$) alongside the original and the nearest codebook neighbor would make the augmentation mechanism interpretable and would help readers assess physical plausibility of the generated samples.
- **Physical constraint metrics.** Reporting conservation error (e.g., mass or energy conservation residuals) alongside MSE would strengthen the "physical consistency" claim beyond the visual energy-spectrum comparison.

---

## Removed Points
*These points are flagged for removal—treat them with caution.*

- **"First to propose" claim (Harsh Critic).** While the claim is poorly supported, removing it from the paper is a style fix, not a substantive weakness. The actual novelty of the combination stands on its own without this phrase. Not a reviewable weakness.
- **KNN graph vs. physical topology (Harsh Critic).** The paper follows prior work on graph-based spatial modeling (Fan et al., 2019) in using KNN. This is standard practice in the GNN-for-PDEs literature. Criticizing KNN without evidence that mesh adjacency would perform better is scope creep.
- **Physical parameter conditioning may be too weak (Harsh Critic §3.2).** The channel attention in Eq. (3) is a standard and well-motivated design (following Takamoto et al., 2023). Claiming it cannot capture higher-order interactions is speculative.
- **Potential train-test information leakage (Harsh Critic).** The concern about physical parameters/boundary conditions at test time is reasonable in principle, but without evidence that the datasets actually hide this information at test time, this is conjecture rather than an identified flaw.
- **Unfair comparison due to SPARK using more side information (Harsh Critic, Reviewer 2).** The baselines do not use boundary/parameter conditioning in Table 1; however, the asymmetry disadvantages SPARK's competitors, not SPARK—this makes SPARK's advantage conservative rather than inflated. Per the rules, this should be removed as a weakness.
- **Confidence intervals on single-run results (Harsh Critic §4.2).** For large-scale PDE and weather benchmarks (ERA5, Prometheus), single-run evaluation is standard. This is a nice-to-have at best.
- **Demand for theoretical proof of ODE stability under quantization (Harsh Critic, Reviewer 2).** Requesting a theoretical bound on how VQ error propagates through ODE integration is beyond standard expectations for an empirical systems paper at ICLR.
- **Broader impact discussion (Harsh Critic).** This is a formatting/completeness issue rather than a scientific weakness.
- **Why is transfer fine-tuning sometimes worse for baselines (Harsh Critic §4.5)?** The observation that SimVP without SPARK slightly degrades at higher SEVIR data fractions (Table 3) is a known phenomenon (negative transfer) and does not imply baseline misconfiguration.

---

## Novel Insights

The most genuinely novel conceptual observation—under-discussed in the sub-reviews—is that the VQ discretization step serves a dual purpose: it compresses physics-rich representations for efficiency, and it implicitly defines a manifold of physically plausible states as the codebook. Augmentation then amounts to interpolating on this physics-constrained manifold rather than in unconstrained input space, which is a principled mechanism for generating physically plausible synthetic samples. This is a more interesting idea than standard latent-space mixup because the codebook geometry is shaped by physical priors. However, the paper does not articulate or test this interpretation explicitly—whether the VQ codebook actually organizes by physics modes (rather than arbitrary clusters) remains unvalidated. Demonstrating this (via codebook visualization colored by physical parameters) would substantially elevate the conceptual contribution.

---

## Suggestions

1. **Add a full ablation table** as the highest priority. At minimum: (a) full SPARK, (b) SPARK without augmentation (just the Fourier ODE), (c) SPARK without Fourier ODE (just augmentation + standard predictor), (d) SPARK without boundary encoding, (e) SPARK without VQ (continuous latent). This directly answers what drives the improvement.

2. **Define the OOD protocol precisely** for each dataset—what parameter/condition is shifted, what range constitutes in-distribution vs. out-of-distribution, and whether test-time physical parameters are available. Add this as a table in the appendix.

3. **Reconcile and explain the number discrepancies** between Table 1 and Table 4 (Prometheus and ERA5). If the model configurations differ, state which variant is used in each table and why.

4. **Specify the augmentation target:** explicitly state that Y_i is held fixed during augmentation of X_i, and provide a justification (e.g., augmented and original samples share the same physics regime, so the same future trajectory is a valid target under the VQ interpolation).

5. **Fix Equation 7** to normalize by K: $v_i = \lambda h_i + \frac{1-\lambda}{K}\sum_{n=1}^K e_n$, or explicitly justify the un-normalized form.

6. **Audit ERA5 baselines.** Reproduce FNO/UNO/CNO on ERA5 with standard hyperparameters from their respective papers and report the configuration used. The order-of-magnitude error gap vs. NMO is suspicious and needs explanation.

7. **Define δ in Eq. 8** and clarify the attention mechanism (is $\alpha_i^t$ normalized? softmax over t?). Define H^l in Eq. 9 and clarify what dimension the DFT operates over for graph node features.

---

**Evaluation axes:**

- **Novelty:** Moderate-to-good. The specific combination of physics-informed VQ augmentation for dynamical system OOD robustness is a concrete and original contribution. Individual components (VQ-VAE, boundary GNN, Fourier ODE) are established, but the integration and motivation are non-trivial.
- **Technical soundness:** Weak. Multiple equations are under-specified, no ablations exist, and the theoretical analysis is disconnected from the proposed design. The method as written is not fully reproducible.
- **Empirical support:** Mixed. The breadth of benchmarks and the plugin validation across backbones are genuine strengths. However, undefined OOD protocols, unexplained number inconsistencies across tables, and suspicious baseline results on ERA5 significantly weaken the evidential strength.
- **Significance:** Moderate. The problem (OOD-robust PDE modeling under data scarcity) is important and practically relevant. If the ablations confirm that the augmentation plugin independently drives improvement, the significance is high; currently this is unestablished.
- **Clarity:** Below acceptable ICLR standard. Key symbolic definitions are missing or inconsistent (δ, H^l, z_i as index vs. embedding), the augmentation procedure is not fully specified, and the absence of ablations leaves the contribution ambiguous.