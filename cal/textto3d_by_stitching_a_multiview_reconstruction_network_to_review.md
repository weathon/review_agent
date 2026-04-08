=== CALIBRATION EXAMPLE 44 ===

# Harsh Critic Review
## Section-by-Section Critical Review

---

### Title & Abstract

The title "Text-to-3D by Stitching a Multi-View Reconstruction Network to a Video Generator" accurately reflects the method and is appropriately descriptive. The acronym VIST3A (Video VAE Stitching and 3D Alignment) feels somewhat forced, but this is a minor concern.

The abstract's claim that "all tested pairings markedly improve over prior text-to-3D models that output Gaussian splats" is bold and largely supported by Tables 1–2, though the evaluation relies heavily on no-reference and CLIP-based metrics that can conflate image quality with 3D-specific properties. The abstract also omits any mention of a key practical compromise revealed in Section C.2: that the MVDUSt3R-based variant requires 100 steps of per-scene Gaussian refinement at inference time, which somewhat contradicts the paper's framing of VIST3A as a clean feedforward approach.

---

### Introduction & Motivation

The motivation is compelling: (1) existing LDM-based text-to-3D methods train 3D decoders from scratch, wasting powerful pretrained 3D models; (2) training with only a generative objective provides no direct guarantee that generated latents will be decodable into good 3D outputs. Both points are well-supported.

One minor issue: the second limitation is attributed partly to "separate training" causing latents to be "out of domain from the perspective of the decoder," yet the proposed solution also involves separate training phases (stitching fine-tuning first, alignment second). The distinction between this and prior work's alignment problems deserves clearer articulation.

---

### Method (Sections 3.1–3.2)

**Stitching layer: theory-implementation gap.** This is the most significant methodological concern. Equation (2) formulates the stitching layer as a globally optimal linear map solved in closed form (ordinary least squares). However, Appendix B.1 reveals that **S** is actually implemented as a 3D convolution with specific kernel sizes, strides, and paddings (e.g., kernel 5×7×7 for MVDUSt3R). A 3D convolution is a locally-weighted linear operator, not a global linear map. The closed-form expression in Eq. (2) is no longer directly applicable: B^⊤B^{-1}B^⊤A_k is not the solution to a convolutional least-squares problem. The paper should either (a) acknowledge that the closed form only initializes a subsequent optimization, or (b) clarify what "closed form" means in the convolutional setting, perhaps via the Wiener–Hopf equations or by noting that the global solution approximates the convolutional one under certain stationarity assumptions. The Lipschitz bound (Eq. 4) applies to the full stitching error, and remains valid in principle, but the discrepancy between theory and implementation should be explicit.

**Layer search criterion.** The MSE-based selection (minimize ‖BS_k* − A_k‖²_F) is elegant and computationally cheap. The empirical validation in Figure 5 is convincing. The paper also correctly notes that absolute MSE values do not predict cross-architecture performance (Section E), which is an honest and important caveat. However, it is not clear whether the same number of training samples (200–3200 scenes) used for the search are sufficient to make the least-squares problem well-conditioned; the rank of B is never discussed.

**Direct reward finetuning.** The reward function design is reasonable: CLIP+HPSv2 for 2D image quality and 3D rendered image quality, plus ℓ₁+LPIPS for 3D consistency. However, CLIP score and HPSv2 are also used as evaluation metrics in Tables 1 and 2. This creates an optimization-evaluation entanglement: the model is explicitly trained to maximize CLIP/HPS scores and then evaluated by those same scores. The gains in CLIP score and Imaging Quality (which is MUSIQ, likely correlated with overall image sharpness) may therefore partly reflect reward exploitation rather than genuine 3D quality improvements. Reference-based metrics on held-out views, or metrics computed with models unused during training, would strengthen the claims.

**Inconsistency in the 3D consistency reward (Table 6).** Adding the 3D consistency reward alone (*Multi-view + Consistency*) causes a notable imaging quality drop from 54.56 → 38.67, which the paper attributes to blurring. While the combination of both rewards recovers performance, the mechanism is not explained: why does adding the quality reward undo the blurring from the consistency reward? This raises questions about the reward weighting (1/16 each for quality rewards vs. 0.05 for consistency), which seems empirically tuned but is not ablated.

**Gaussian refinement step.** Section C.2 reveals that VIST3A with MVDUSt3R requires 100 optimization steps of Gaussian primitive refinement at inference "to correct scale estimation errors." This is glossed over in the main paper and significantly undermines the claim of a fully feedforward pipeline. The computational cost and the fact that this represents a fundamental limitation of the MVDUSt3R pairing (it "does not generalize well across diverse domains") should appear in the main text, not buried in an appendix.

---

### Experiments & Results

**Table 2 (DPG-Bench) is incomplete.** The table as presented shows only baselines (Matrix3D-omni, Director3D, Prometheus3D, SplatFlow, VideoRFSplat) with no VIST3A rows, yet the main text claims "our models greatly outperform the baselines, mostly scoring > 75 (often even ≈ 85)." This is either a PDF parsing artifact or a genuine omission; either way, the reviewers and readers cannot verify the numbers. The results referenced in the text should match the table.

**Text-to-pointmap: qualitative evaluation only.** The paper presents text-to-pointmap generation as a novel capability and includes it in the abstract, yet evaluates it only qualitatively (Section 4.2: "evaluated qualitatively, as no established benchmarks or baselines exist"). This is insufficient for ICLR. A reasonable proxy evaluation would compare rendered novel-view synthesis from the generated pointmaps on a benchmark split, or use existing pointmap/depth evaluation protocols (which the authors already apply in Table 5 for reconstruction, not generation).

**Stitching evaluation (Table 5) measures reconstruction, not generation.** The pointmap results in Table 5 use real images as input to the stitched VAE encoder, demonstrating that stitching preserves the original 3D model's reconstruction quality. This is a meaningful result. However, it does not directly assess whether generated latents from the diffusion model produce equally good pointmaps. A full end-to-end evaluation of text → latent → pointmap quality is not provided quantitatively.

**User study.** 28 participants rating 14 samples is a small study. Statistical uncertainty (confidence intervals or standard errors on the average ranks) is not reported. It is possible that the large margin in "visual quality" (VIST3A ranked #1 in >87% of comparisons) is partially driven by the overall sharpness boost from reward tuning on HPSv2, rather than genuine 3D-structural superiority.

**Sequential baseline is missing.** The paper argues that stitching (integrated latent approach) is better than a sequential decode-then-reconstruct pipeline. Section D.2 investigates this specifically as a noise-robustness analysis (Figure 8), which is a useful ablation. However, there is no direct performance comparison on T3Bench or SceneBench between VIST3A and a comparable sequential pipeline (video model → RGB frames → feedforward 3D model), which would be the most natural baseline for readers to evaluate the stitching contribution.

**Benchmark evaluation mismatch.** The paper evaluates T3Bench on all 300 prompts (including complex multi-object prompts) rather than the original 100-prompt single-object subset. While this is arguably more comprehensive, it makes direct numerical comparison with prior works that used 100 prompts potentially unfair, even if the authors follow the same extended protocol for all methods. This should be clearly flagged.

---

### Writing & Clarity

The technical content is generally clear. The only substantive clarity issue is that the relationship between the theoretical linear stitching (Eq. 2) and the convolutional implementation (Appendix B.1) is never reconciled in the text. Section 3.1 says "fitting a single, linear stitching layer (in closed form)" and then Appendix B.1 describes it as a Conv3D. A reader who takes the theory seriously will be confused.

Also, Sections 4.4 and the "Additional results" paragraph at the end of 4.4 (lines 780-788) are nearly identical repetitions, suggesting an editing oversight.

---

### Limitations & Broader Impact

The limitations section (Appendix F) is superficial and limited to one point: the video encoder's preference for ordered, smooth-transition inputs. Several substantive limitations are unacknowledged:

1. **Scale estimation failure in MVDUSt3R** is a significant known issue (addressed by heuristic per-scene refinement) that is not discussed as a fundamental limitation.
2. **Camera trajectory bias**: The generated multi-view sequences inherit the trajectory distribution of the video generator, which may produce limited viewpoint coverage (e.g., predominantly forward-facing, drone-shot, or lateral panning). This limits the 360° scene understanding of the resulting 3D representation.
3. **Training data bias**: Training on DL3DV-10K (outdoor) and ScanNet (indoor) may skew performance toward these scene types. The strong T3Bench results on object-centric prompts may be driven by the powerful video backbone's prior rather than the 3D decoder.
4. **Reward hacking potential**: The use of CLIP/HPSv2 as both training signal and evaluation metric is not acknowledged as a limitation.
5. **Computational cost**: Combining Wan 2.1 Large with AnySplat/MVDUSt3R creates a very large model. Inference cost is not discussed.

---

### Overall Assessment

VIST3A is a genuinely interesting and timely contribution. The core idea — repurposing pretrained 3D reconstruction models as decoders for video latent diffusion models via model stitching — is creative and practically motivated: it avoids the expensive and data-hungry reconstruction-from-scratch training of prior methods. The empirical results are strong across multiple benchmarks and multiple VAE/3D-model pairings, and the reward-based alignment is a principled way to close the latent-space distribution gap. The main concerns that must be addressed before acceptance are: (1) the unresolved gap between the theoretical linear-map formulation and the 3D-convolutional implementation, including whether the "closed-form" claim is accurate for the actual implementation; (2) the optimization-evaluation entanglement from using CLIP/HPSv2 as both training rewards and evaluation metrics; (3) the missing VIST3A rows in Table 2; (4) the lack of quantitative evaluation for the text-to-pointmap capability; and (5) the Gaussian primitive refinement step for MVDUSt3R that should be clearly disclosed in the main text. Taken together, these are significant but fixable issues; the underlying contribution is sound and the paper warrants acceptance conditional on adequate revision.

# Neutral Reviewer
## Balanced Review

### Summary
VIST3A introduces a modular framework for text-to-3D generation by linearly stitching the latent space of a pretrained video diffusion model to an intermediate layer of a feedforward 3D reconstruction network, creating a 3D-aware decoder without training from scratch. To ensure the generative backbone produces latents compatible with this decoder, the authors employ a direct reward finetuning scheme that jointly optimizes for multi-view visual quality, rendered fidelity, and 3D consistency. The resulting pipeline achieves strong quantitative and qualitative results across multiple text-to-3D benchmarks and demonstrates flexible text-to-pointmap generation.

### Strengths
1. **Practical and effective architectural paradigm:** The paper directly addresses a recognized bottleneck in current text-to-3D LDMs: the need to train custom 3D decoders from scratch. By reusing pretrained 3D foundation models via stitching, VIST3A bypasses costly dataset collection and training while preserving state-of-the-art geometric reasoning. Evidence: Consistent performance gains over strong baselines (SplatFlow, Director3D, Prometheus3D) across T3Bench, SceneBench, and DPG-Bench (Table 1 & 2).
2. **Rigorous empirical validation and broad generalization:** The framework is systematically evaluated across four video generative backbones (Wan 2.1, CogVideoX, SVD, HunyuanVideo) and three 3D reconstruction models (AnySplat, MVDUSt3R, VGGT), demonstrating robustness to backbone choice. Evidence: Comprehensive ablation studies on stitching layer selection (MSE correlates with lower pointmap error, Fig. 5) and reward component impact (Table 6), alongside successful extension to text-to-pointmap generation.
3. **Methodological clarity and training stability:** The integration of theoretical insights (stitching risk bound in Eq. 4 justifies MSE-based layer selection) with practical training tricks (DRTune-style gradient detachment, randomized timestep sampling) ensures stable and efficient alignment. Evidence: Section 3.2 and Appendix B.2 provide clear algorithmic details, and the framework successfully mitigates the typical instability of reward-based diffusion tuning.

### Weaknesses
1. **Asymmetric evaluation pipeline for MVDUSt3R:** The inclusion of a lightweight 100-step 3DGS optimization pass specifically for the MVDUSt3R variant introduces a methodological asymmetry that may confound fair comparison with end-to-end baselines. Evidence: Appendix C.2 states, "we refined the Gaussian primitives using the source view for 100 optimization steps... This lightweight refinement effectively corrected the scale estimation errors," yet this step is not applied to competing methods.
2. **Over-reliance on perceptual and LLM-based metrics:** While the authors rightly critique no-reference metrics like NIQE, the primary quantitative claims rest entirely on CLIP, HPSv2, and the UnifiedReward LLM, which are known proxies that do not strictly measure geometric accuracy. The human evaluation, though present, is quite small. Evidence: Section 4.2 notes a user study with only 28 participants ranking 14 samples, which limits statistical confidence in the subjective rankings (Table 4).
3. **Missing computational and efficiency analysis:** Direct reward finetuning with in-the-loop 3D rendering and perceptual scoring is computationally intensive, but the paper lacks a thorough discussion of training time, GPU memory overhead, or wall-clock convergence compared to standard multi-view generative fine-tuning. Evidence: Section 3.2 mentions computational optimizations (fewer timesteps, selective gradient steps) but provides no FLOPs, iteration count, or hardware/runtime breakdown to contextualize the trade-off.

### Novelty & Significance
The conceptual novelty is moderate, as model stitching and direct reward finetuning are established techniques; however, their systematic combination to solve the specific misalignment problem between video latents and 3D decoders represents a clever and well-executed engineering advance. The significance is high for the 3D vision and generative modeling communities at ICLR. It demonstrates a scalable, modular pathway to unlock strong foundation models for 3D generation without monolithic retraining, aligning with current trends in representation alignment and efficient model adaptation. The methodological clarity, extensive cross-backbone validation, and open project page strongly support reproducibility.

### Suggestions for Improvement
1. **Address the MVDUSt3R refinement asymmetry:** Either report metrics for the Wan+MVDUSt3R variant with and without the 100-step refinement, or apply a comparable lightweight optimization step to all competing 3DGS baselines to ensure a strictly fair comparison.
2. **Provide computational efficiency metrics:** Include a dedicated subsection or table detailing training compute (GPU hours, batch size effects, time per reward step) and compare inference throughput/latency against the primary baselines. This will help readers assess the practical cost-benefit of the reward tuning scheme.
3. **Strengthen geometric evaluation:** Supplement the perceptual/LLM metrics with geometry-focused evaluations where possible, such as Chamfer distance or normal consistency on synthetic/semi-synthetic datasets, or expand the human study with standardized rating rubrics and more participants to increase statistical power.
4. **Analyze temporal-to-static latent conversion:** Since video VAEs are trained on temporally coherent sequences, explicitly analyze how the framework suppresses unwanted motion/blur when generating static 3D scenes. Adding a quantitative measure of temporal variance (e.g., optical flow magnitude between decoded frames) or a discussion on how the reward implicitly penalizes motion would strengthen the technical narrative.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Add comparison to SDS-based methods with equivalent compute budgets** — The paper claims feedforward superiority but doesn't show whether slower optimization-based methods (DreamFusion, Magic3D) produce better geometric quality when given similar inference time, undermining the efficiency claim.

2. **Add ablation isolating the stitching benefit vs. training a decoder from scratch** — Without comparing to a freshly-trained decoder on the same data, you cannot claim stitching preserves pretrained knowledge rather than just providing a better initialization.

3. **Add quantitative 3D geometric consistency metrics** — Current evaluation relies on rendered 2D images; ICLR expects actual 3D metrics (chamfer distance, normal consistency on reconstructed geometry) to validate the "3D-consistent" claim.

4. **Test on more diverse video backbones systematically** — Only Wan 2.1 is thoroughly evaluated; other video models (CogVideoX, SVD, Hunyuan) appear in Table 3/5 but lack full text-to-3D benchmark results, weakening the generality claim.

5. **Add zero-shot transfer experiments** — The method should be tested on 3D models and video generators not seen during stitching search to prove the approach generalizes beyond the specific combinations tested.

### Deeper Analysis Needed (top 3-5 only)
1. **Explain why 3D-consistency reward harms performance in Table 6** — The ablation shows "Multi-view + 3D Consistency" drops Imaging quality from 54.56 to 38.67; this contradicts the core motivation and requires mechanistic explanation, not just observation.

2. **Analyze failure modes systematically** — No discussion of when stitching fails (e.g., which prompt types, scene complexities, or geometry types break); ICLR reviewers expect honest limitation analysis with supporting evidence.

3. **Quantify computational cost of reward finetuning** — The reward computation involves rendering, CLIP, HPSv2, and LPIPS per training step; without FLOPs/time comparison to standard finetuning, the practical viability is unclear.

4. **Validate the MSE stitching criterion theoretically** — Theorem 1 from Insulla et al. (2025) is cited but not verified on your models; show the Lipschitz constant bound actually holds for your specific VAE-3D combinations.

5. **Analyze latent space distribution shift before/after alignment** — Without showing how the generative model's latent distribution changes to match the decoder's expected input, the alignment mechanism remains a black box.

### Visualizations & Case Studies
1. **Show 360° turntable renders of generated 3D assets** — Single-view renders cannot verify true 3D consistency; provide multi-angle videos exposing geometric artifacts that single views hide.

2. **Visualize stitching layer activations before/after finetuning** — Show whether the linear stitching actually achieves representation alignment or if the decoder simply adapts to mismatched latents during finetuning.

3. **Display failure cases with geometric inconsistencies** — Show examples where the method produces impossible geometry (floating objects, inconsistent depth) to demonstrate you understand the method's boundaries.

4. **Compare point cloud depth distributions** — For text-to-pointmap results, show depth histograms comparing generated vs. ground-truth distributions to verify metric accuracy beyond qualitative visuals.

### Obvious Next Steps
1. **Include direct comparison to Chen et al. (2026)** — This concurrent work on VAE component interchange is mentioned in Related Work but not experimentally compared; ICLR expects engagement with concurrent methods.

2. **Add training curve analysis showing convergence behavior** — Show how reward finetuning affects training stability and whether the method converges reliably across different model combinations.

3. **Test on out-of-distribution prompt categories** — Evaluate on prompts significantly different from training data (e.g., abstract concepts, non-photorealistic styles) to assess true generalization beyond DL3DV domain.

# Final Consolidated Review
## Summary
VIST3A proposes a framework for text-to-3D generation that stitches pretrained video VAE encoders to pretrained feedforward 3D reconstruction models (AnySplat, MVDUSt3R, VGGT) via a learned stitching layer, then aligns the generative model to the stitched decoder using direct reward finetuning. This avoids training 3D decoders from scratch and leverages state-of-the-art 3D foundation models.

## Strengths
- **Principled reuse of pretrained models**: The stitching approach cleverly repurposes powerful 3D reconstruction models as decoders for video latent spaces. The MSE-based layer selection criterion is theoretically justified (Eq. 4 cites Lipschitz bounds from Insulla et al. 2025) and empirically validated across multiple VAE–3D model combinations (Fig. 5).
- **Strong empirical performance across diverse backbones**: VIST3A demonstrates consistent improvements over strong baselines (SplatFlow, Director3D, VideoRFSplat) on T3Bench and SceneBench (Table 1). The framework generalizes across four video generators (Wan, CogVideoX, SVD, Hunyuan) and three 3D models, showing the method's modularity.
- **Comprehensive ablation studies**: The paper ablates the stitching layer criterion (Fig. 5, 6), reward components (Table 6), and compares integrated vs. sequential pipelines (Fig. 8). The finding that early layers have lower MSE and correlate with better 3D reconstruction is well-supported.

## Weaknesses
- **Gap between theoretical formulation and implementation**: Section 3.1 states the stitching layer is found "in closed form" via Eq. 2 (ordinary least squares), but Appendix B.1 reveals it is implemented as a 3D convolution with specific kernels. A Conv3D is a locally-weighted operator, not a global linear map; the closed-form solution in Eq. 2 does not directly apply. The paper should clarify whether the OLS solution initializes an optimization problem or approximate the Conv3D under certain assumptions.
- **Optimization–evaluation metric entanglement**: The reward function uses CLIP and HPSv2 scores (Eq. 5), and these same metrics appear in Tables 1–2. While CLIP score is standard for text-image alignment, training to maximize HPSv2 and then reporting HPSv2 gains risks overfitting to the metric rather than genuine quality improvement. Reporting additional metrics not used in training (e.g., geometry-focused measures) would strengthen validity.
- **Incomplete DPG-Bench results**: Table 2 shows only baseline methods but the main text claims "our models greatly outperform the baselines, mostly scoring > 75 (often even ≈ 85)." Without the VIST3A rows, readers cannot verify these claims against the benchmark.
- **Qualitative-only evaluation for text-to-pointmap generation**: The paper prominently features text-to-pointmap in the abstract and figures, yet Section 4.2 states "text-to-pointmap models are evaluated qualitatively, as no established benchmarks or baselines exist." A quantitative proxy (e.g., novel-view synthesis from generated pointmaps) would validate this capability.
- **MVDUSt3R refinement step not disclosed in main text**: Appendix C.2 reveals that Wan+MVDUSt3R requires 100-step per-scene Gaussian primitive refinement at inference "to correct scale estimation errors." This compromises the feedforward claim for this variant and should appear in the main text with its computational cost.
- **3D-consistency reward degrades imaging quality (Table 6)**: Adding the 3D-consistency reward alone drops imaging quality from 54.56 to 38.67. The paper attributes this to "optimizing for geometric correctness at the expense of detail, resulting in overly blurred images," but does not explain why the quality reward subsequently recovers performance or how the reward weights were chosen.

## Nice-to-Haves
- **Computational cost analysis**: Training time, GPU memory, and inference latency comparisons against baselines would help readers assess practical viability. The reward computation involves rendering and multiple perceptual models per training step.
- **User study statistical rigor**: The user study has 28 participants ranking 14 samples. Reporting confidence intervals or standard errors on average ranks would strengthen the subjective evaluation.
- **Sequential pipeline baseline on main benchmarks**: While Section D.2 compares stitched vs. sequential approaches under noise injection, a direct comparison on T3Bench/SceneBench would isolate the stitching contribution more clearly.

## Removed Points
*These points are flagged to be removed, treat them with caution*

- **Acronym feel**: The claim that "VIST3A" is a forced acronym is a minor stylistic nitpick with no bearing on technical contribution.

- **Separate training phases concern**: The criticism that the paper's solution also involves "separate training phases" conflates training stages (stitching finetuning then alignment) with the paper's critique of "separate training" causing decoder-generator misalignment. These are fundamentally different: VIST3A aligns end-to-end through reward signals, whereas prior work trains decoder and generator independently with no joint objective.

- **Benchmark evaluation mismatch**: The paper evaluates on all 300 T3Bench prompts while some baselines used 100; however, the author's protocol applies consistently across all methods, making the comparison fair. This is a methodology difference, not a flaw.

- **Camera trajectory bias speculation**: The claim that generated views inherit limited trajectory distributions is speculative without empirical evidence showing insufficient viewpoint coverage.

- **Reward hacking claim**: The criticism of "reward hacking potential" is speculative; the paper shows improved human evaluation results (Table 4) alongside automated metrics.

- **SDS comparison request**: Comparing to slow per-scene optimization methods (DreamFusion, Magic3D) with "equivalent compute budgets" is outside the paper's scope, which focuses on feedforward approaches.

- **Chen et al. (2026) comparison**: Requesting experimental comparison to concurrent work is unreasonable when the concurrent work's code/models may not be available; the paper appropriately discusses it in Related Work.

- **Zero-shot transfer experiments**: Testing on models not seen during stitching search is a nice extension but not a core requirement for demonstrating the method's efficacy on tested combinations.

## Novel Insights
The MSE-based stitching layer criterion offers a practical diagnostic for model interoperability: layers with lower reconstruction MSE after linear alignment consistently yield better final 3D generation quality (Fig. 5). This aligns with theoretical bounds from representation alignment literature but importantly, CKA (a common similarity metric) fails to predict the optimal layer (Fig. 6), suggesting that linear transferability is a more precise measure than representational similarity alone. This insight—that what matters is not whether representations look similar, but whether one can be linearly transformed into the other—has implications beyond 3D generation for any model composition task.

## Suggestions
- Add a footnote in Section 3.1 clarifying the relationship between the closed-form linear map formulation and the Conv3D implementation.
- Move the MVDUSt3R refinement requirement from Appendix C.2 to the main text (Section 4.2), with approximate inference time overhead.
- Include VIST3A results in Table 2 or clearly note if results are deferred to supplementary.
- Add at least one geometry-focused metric (e.g., depth error or multi-view consistency score on rendered views) not used in training to strengthen evaluation validity.
- Explain the reward weighting strategy (why 1/16 for quality vs. 0.05 for consistency) and why quality reward mitigates the blurring from consistency reward alone.

# Actual Human Scores
Individual reviewer scores: [8.0, 8.0, 8.0, 8.0]
Average score: 8.0
Binary outcome: Accept
