## Summary

RetinexGDP is a training-free, zero-shot framework for personalized low-light image enhancement that combines Retinex decomposition with a pre-trained diffusion model guided by CLIP-based text instructions. The method's two core technical novelties are: (1) a Gaussian Total Variation (TV) layer that provides deterministic, single-image illumination estimation without any training, and (2) patch-wise DDIM inversion with reflectance conditioning and directional CLIP loss to steer enhancement toward user-specified styles. The paper evaluates on nine datasets against ten baselines and presents qualitative personalization results.

---

## Strengths

- **Deterministic illumination estimation via Gaussian TV layer**: Replacing the random convolutional kernel with a fixed Gaussian kernel yields provably consistent illumination maps (Fig. 3), eliminating the instability inherent in DIP-based Retinex methods like RetinexDIP and DRP. This is a concrete, practically-motivated design insight specific to the training-free setting.
- **Patch-wise DDIM inversion enabling arbitrary-resolution processing**: The patch aggregation scheme with a binary overlap counter G allows the method to handle images of any size — a genuine limitation of standard patch-based diffusion models. The ablation in Fig. 9 provides visual evidence that this strategy prevents structure distortion and dark-region artifacts.
- **Meaningful gains over training-free peers on paired benchmarks**: On LOL and VELOL (Table 2), RetinexGDP achieves 15.66/0.66 PSNR/SSIM vs. RetinexDIP's 8.59/0.30 and NeuralBR's 11.36/0.44, demonstrating that conditioning on the Retinex reflectance substantially improves fidelity over unconditioned or alternative zero-shot baselines.
- **Text-guided personalization in an underexplored direction**: The qualitative results in Fig. 6 show that text prompts like "Summer sunset," "Winter morning," and "Cool tones of a winter twilight" produce visibly distinct, semantically coherent style variations while preserving scene structure. This capability is genuinely absent from all compared baselines.

---

## Weaknesses

### Fatal
None that are strictly fatal in isolation, but the combination of (a) an unquantified core contribution and (b) ablation evidence that the contribution actively degrades image quality represents a serious unresolved tension.

### Major

- **No quantitative evaluation of the title-level contribution (personalization).** The paper's headline claim is text-based *personalized* enhancement, yet Section 4.1 provides only qualitative figures. There is no user study, no CLIP-image alignment score, and no text-image similarity metric. For an ICLR submission, this is a critical omission: the method's distinguishing feature is evaluated only subjectively.

- **Text guidance demonstrably degrades standard quality metrics (Table 3).** Adding text instruction to L_recon worsens NIQE from 5.44 to 6.47, NIQMC from 5.03 to 4.81, and CPCQI from 1.05 to 0.69. The paper acknowledges this as a "slight drop" but does not analyze why, nor does it explain what the user gains that compensates for this degradation. This is not a nitpick — if the text-guided variant produces quantitatively worse images than the non-personalized variant, the paper needs to justify the trade-off with either perceptual or user-preference evidence. Without this, the core contribution is undermined by the paper's own data.

- **Poor NIQE performance in Table 1 (no-reference, unpaired evaluation).** RetinexGDP ranks among the worst on NIQE for 5 of 7 datasets (e.g., 4.80 on ExDark, 5.22 on Fusion, 5.54 on LIME) while methods like DiffusionLL score 3.27, 3.30, 3.58 respectively. The abstract's claim of "performance comparable to state-of-the-art models" is not supported by Table 1 for the primary no-reference quality metric. The paper's framing needs to be substantially more candid.

### Minor

- **Modified CLIP loss (Eq. 9) lacks principled justification.** Removing the source text prompt reduces the loss to: cosine similarity between (output embedding − input reflectance embedding) and the target text embedding. Without an anchoring source prompt, any image change in the target direction minimizes the loss — including content drift unrelated to the intended style. The paper attributes the removal to "misalignment between natural language descriptions and the reflectance component" but provides no ablation or analysis showing whether content preservation is actually maintained. The risk of semantic hallucination or unintended drift is not assessed.

- **Inference time is not reported.** The method uses patch-wise DDIM inversion + T=50 reverse steps with CLIP and VGG19 gradient backpropagation at each step. Given that Zero_DCE processes images in milliseconds, the inference cost is almost certainly orders of magnitude higher. The limitations section mentions "real-time limitation" but gives no numbers. Absolute runtime per image on the TITAN X GPU is essential information for evaluating practical utility.

- **Single-iteration TV solver unjustified.** Section 4 states the TV optimization runs for "a single iteration," but convergence of the TV proximity operator typically requires many iterations. No justification is given for why one iteration produces a usable illumination map, and no ablation tests 5, 10, or more iterations.

- **Hyperparameter sensitivity analysis absent.** The method involves σ=0.5, λ=30, kernel size=7, γ=0.5, loss scales 5000/100/7000 — none of which have sensitivity analysis or justification. This is particularly important for a "training-free" system intended to be generally applicable without per-dataset tuning.

- **Variance aggregation in patch-wise DDIM inversion is mathematically imprecise.** The paper computes φ_t = φ_t ⊘ G (variance divided by overlap count), which is a simple arithmetic mean of variances over overlapping regions. Statistically correct aggregation of independent Gaussian noise estimates requires inverse-variance weighting, not simple averaging. Whether this approximation affects reconstruction quality is not analyzed.

### Tiny

- **Percentage comparisons on PSNR (dB) in Section 4.2 are misleading.** PSNR is a logarithmic metric; stating "26.39% higher PSNR than CLIP-LIT" is not meaningful. The correct framing is "+3.27 dB over CLIP-LIT" and "+7.07 dB over RetinexDIP." The absolute differences are genuinely good — the misleading framing is unnecessary.
- **Patch stride p is not specified in the main text** (only patch size = 256 is given in Section 4). Reproducibility is limited without this detail.
- **Notation inconsistency**: illumination appears as I, I', I_c, and I^γ at different points without a unified reference. A notation summary in the appendix would help.

---

## Nice-to-Haves

- **Ablation on the Retinex decomposition stage for personalization**: An experiment replacing the Retinex-conditioned DDIM with direct CLIP-guided DDIM on the raw low-light image would confirm whether the Retinex stage actually benefits the personalization goal (as opposed to pure enhancement quality).
- **Qualitative failure case analysis**: Cases where CLIP guidance causes color hallucinations, unintended semantic changes, or instability with ambiguous/contradictory prompts would add important understanding of the method's limits.
- **Perceptual metrics (LPIPS) for personalization evaluation**: Standard no-reference metrics penalize stylistic changes introduced by personalization. LPIPS or FID could better capture whether the method produces high-quality personalized outputs even when NIQE degrades.
- **Adaptive or schedule-aware Gaussian kernel**: The fixed σ=0.5 may be suboptimal for images with very different illumination scale. An input-adaptive or learned σ could improve generalization.
- **Fewer sampling steps / distillation**: Investigating DDIM acceleration or consistency distillation to mitigate the real-time limitation is a natural extension.

---

## Removed Points

*These points were flagged for removal; treat them with caution.*

- **η value inconsistency (Harsh Critic):** The paper explicitly states η = √(1−ᾱ_t) in Section 3.2. Figure 10 is an ablation comparing both schedules — not an ambiguity about which was used. This criticism is factually incorrect.
- **CLIP guidance on noisy intermediate states (Harsh Critic):** Applying CLIP (trained on clean images) at noisy diffusion timesteps is a well-known approximation that is standard practice in the classifier guidance literature since Dhariwal & Nichol (2021) and widely adopted. Criticizing it as a unique flaw of this paper is unfair.
- **Gaussian and TV being "in opposition" (Harsh Critic):** The Gaussian convolution pre-smooths the input to provide an illumination estimate, and the TV proximity operator subsequently enforces piecewise smoothness on this estimate. These are sequential operations serving complementary roles, not contradictory ones.
- **"Gamma correction not being true Retinex reflectance" (Harsh Critic):** The paper explicitly cites Guo et al. (2017) and Zhao et al. (2024) to justify using gamma-corrected illumination for reflectance computation as a recognized initial enhancement technique. The conceptual choice is grounded in prior work and within scope.
- **Comparison fairness with training-based methods having more capacity (Harsh Critic):** Table 1 explicitly marks training-free methods in gray, and the paper's comparisons are presented with this distinction. The asymmetry is transparent and standard in the field.
- **Comprehensive evaluation on 9 datasets as a strength (Positive Reviewer):** This is generic and would apply to any paper with a thorough evaluation section.

---

## Novel Insights

The paper surfaces an underappreciated observation: **standard no-reference image quality metrics actively penalize the stylistic modifications that text-based personalization produces** (Table 3). This is not a flaw unique to RetinexGDP — it is likely a general problem for any image enhancement method that allows user-controlled style deviation from "natural" appearance distributions. The community may need specialized perceptual metrics for evaluating personalized enhancement that distinguish intentional stylization from quality degradation. This points to a broader measurement gap: NIQE and NIQMC were designed for naturalness, not for evaluating adherence to user preference, and using them as the primary metric for a personalization paper conflates two different objectives.

---

## Suggestions

1. **Add a user preference study** for the personalization results — even a small-scale study (20–30 participants rating which enhancement best matches the text prompt) would directly evaluate the core contribution. Include both "does it match the prompt?" and "is it visually acceptable?" as dimensions.
2. **Report inference time per image** on the TITAN X GPU, and compare it directly against Zero_DCE and RetinexDIP to give readers a concrete sense of the trade-off.
3. **Add an ablation confirming which η is used in all main experiments** and unify this in the text (e.g., a single clear statement: "all results in Tables 1–2 use η = √(1−ᾱ_t)").
4. **Replace percentage PSNR comparisons with absolute dB differences** in Section 4.2.
5. **Report Table 3 ablation on at least two additional datasets** — single-dataset ablations are difficult to generalize from.
6. **Investigate and report the effect of 1 vs. 5 vs. 10 TV iterations** to justify the single-iteration choice.
7. **Add CLIP cosine similarity between generated images and target text prompts** as a quantitative personalization metric in Table 1 or a dedicated personalization table.

---

**Overall assessment**: RetinexGDP addresses a genuine and underexplored problem, and its Gaussian TV layer is a clean, well-motivated technical contribution. However, the paper's evaluation is not yet ready for ICLR: the core personalization capability has no quantitative support; the method's own ablation (Table 3) shows text guidance hurts image quality without any compensating evidence; and general enhancement performance (Table 1) is below several baselines on the primary metric. The technical novelty is moderate and the empirical support is weak for the paper's stated claims. Significant revisions — particularly a user study, inference time reporting, and honest reframing of Table 1 results — are needed before the work can be assessed fairly.