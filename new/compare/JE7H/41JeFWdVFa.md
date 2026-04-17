---
job_id: 400970c6-f89a-4b84-b715-22c7bcb2b19b
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: 41JeFWdVFa.pdf
paper: LDP: A Lightweight Denoising Plugin Enhancing Generalization in Single-Image Super-Resolution
main_score_norm: 0.4
desk_reject: false
---
# Desk Rejection Assessment:

## Paper Length
Pass ✅.  
The submission includes abstract, introduction, related work, method, experiments, ablations, limitations & conclusion, references, plus appendices.

## Topic Compatibility
Pass ✅.  
The work is on single-image super-resolution, degradation modeling, and denoising autoencoders/diffusion-style corruption, which fits ICLR topics in representation learning and generative models for vision.

## Minimum Quality
Pass ✅.  
All required scientific sections are present. The paper is written in English, technically coherent, with substantial experiments and ablations. While I will be critical about novelty, positioning, and certain design choices, there is no fatal flaw that warrants desk rejection.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I do not see any hidden prompts, instructions to reviewers, or manipulation attempts in the provided content.

---

# Expected Review Outcome:

## Summary

The paper proposes LDP, a lightweight denoising-autoencoder-style degradation model that can be used as a plug-in for single-image super-resolution (SISR). LDP takes an HR image (ground-truth or SR output) plus a high-frequency component of the LR input as condition, adds patch-wise Gaussian noise, denoises with a small CNN conditioned via prompts, then downsamples to predict an LR image, which is used to enforce LR cyclic consistency during SR training or to guide diffusion posterior sampling at inference. Experiments on synthetic multi-degradation benchmarks and several real-world datasets show that integrating LDP into a variety of SR architectures generally improves robustness to unseen degradations and reduces artifacts.

## Strengths

1. **Clear overall idea and useful plugin formulation.**  
   The central idea, using an explicit conditional degradation model to enforce LR cyclic consistency for arbitrary SR backbones, is well-motivated and clearly articulated (Sec. 3.1–3.3). Unlike prior dual-branch approaches like DRN and DualSR, LDP is intentionally lightweight (642k params for s=4, Page 6) and is framed as a reusable plug-in loss or inference module, which is practically appealing.

2. **Thoughtful architecture design with interpretable components.**  
   The LDP framework in **Figure 2(a–d)** is well-structured: high-frequency LR residual \(y_{hf}\) (Eq. 4) is processed by the Degradation Prediction Module to produce a degradation map \(C'\) (Eqs. 5–6), patch-wise noise is added via diffusion-style corruption (Eq. 7), then a conditioned denoiser with AdaLN (Eqs. 8–11) estimates features that are downsampled to yield LR predictions (Eq. 12). This modular decomposition helps connect each design choice to the degradation modeling goal.

3. **Consistent improvements across many SR backbones on synthetic benchmarks.**  
   **Table 3** shows that fine-tuning FeMaSR, StableSR, SwinIR, and MambaIR with LDP improves PSNR/SSIM and reduces LPIPS on all five degradation types. The gains are particularly large for StableSR (e.g., +2.16 dB PSNR on Hybrid, +0.1541 SSIM), indicating that LR-consistency guided by LDP is especially helpful for strong generative SR models. **Figure 4** qualitatively supports the claim that LDP reduces artifacts and improves structural fidelity for diverse degradations.

4. **Non-trivial results on real-world datasets and cross-architecture robustness.**  
   **Table 4** demonstrates that LDP improves multiple no-reference quality metrics (MANIQA, MUSIQ, QAlign) across RealSR, DPED, and RealSRSet for most architectures. The fact that a single degradation plug-in improves GAN-based, diffusion-based, Transformer-based, and SSM-based SR models is a meaningful empirical result, even if not universally positive. **Figure 5** and extended qualitative **Figures 8–9** show visible suppression of ringing and GAN-type artifacts while preserving textures reasonably well.

5. **Posterior sampling integration for diffusion models.**  
   The extension of LDP to posterior sampling (Eq. 17) is conceptually clean: the gradient of \(\mathcal{L}_{sym}^{FT}\) is used in a DPS-like update to encourage LR-consistent samples. **Table 5** shows mainly positive shifts in non-reference metrics for StableSR, ResShift, and UPSR; **Figure 6** visually illustrates that diffusion outputs become less artifact-heavy with LDP guidance.

6. **Extensive ablation and analysis.**  
   The ablations in **Tables 6–10, 11–13** cover: loss components, \(\tau\) scaling, patch size in the Noise Addition Module, DWT subband choice, scale \(s'\) for LR residuals, severe blur robustness, and LDP’s computational overhead. For example, **Table 6** systematically explores which combinations of \(\mathcal{L}^{Sym}\), \(\mathcal{L}_{LPIPS}\), \(\mathcal{L}^{Sym}_{1o}\), and SR frequency loss produce the best trade-off, and **Table 8** shows that the patch-wise noise scheme (P=16) consistently outperforms baseline SR without LDP. This level of analysis is above average for SR papers.

7. **Reasonably detailed training and implementation description.**  
   Hyper-parameters and training procedures for both LDP and downstream SR models are largely specified (Sec. 4.1, Appendix D–F), including the choice of timesteps \(t_i\in[500,1000]\), patch size \(P=16\), prompt dimension \(N_p\), and how losses are incorporated for each backbone. This should make replication feasible.

## Weaknesses

1. **Conceptual novelty is moderate; LDP is mostly a specific instantiation of known ideas.**  
   The paper situates itself in degradation modeling and diffusion-based consistency constraints, but the core components are largely combinations or refinements of prior ingredients:
   - Bidirectional degradation–SR cycles (DRN, DualSR, SCL-SASR, Lway) already use LR reconstruction consistency as a constraint.
   - Diffusion-based inverse problem methods (ILVR, DR2, MCG, DPS) already exploit noise-injected LR–HR alignment and gradient-based data-fidelity guidance.
   - High-frequency conditioning from LR (e.g., Lway) and frequency-domain losses (Eq. 14–15) have appeared before.  
   LDP’s specific recipe (patch-wise timesteps, LR high-frequency residual as condition, prompt-like degradation map) is carefully engineered, but the paper sometimes oversells its conceptual contribution relative to this prior art. For example, the “reinterpreting denoising as controllable degradation applied to HR images” in Sec. 3.1 is closely aligned with existing diffusion inversion / cold diffusion style ideas, but this connection is not acknowledged fully.

2. **Limited theoretical grounding of the “diffusion alignment” argument.**  
   The method relies heavily on the statement that “after noise is added, HR and LR features become aligned, so denoising noisy HR features is equivalent to denoising noisy LR features” (Intro, Sec. 3.1, citing Wang et al. 2023b). However:
   - There is no derivation clarifying under what degradation operators \(D\) and noise schedules \(\hat{\alpha}_t\) this approximate equivalence holds.
   - Eq. (7) defines noise addition on HR patches \(x_i\), but there is no explicit connection showing that the resulting distribution of noisy \(x_t\) is sufficiently close to that of noisy LR images \(y_t\) under realistic degradations from Eq. (1).
   - The patch-dependent timesteps \(t_i\) further complicate the diffusion interpretation; there is no consistency argument that a single denoiser conditioned on a merged \(x_t\) is equivalent to independent per-patch denoisers.  
   This makes the “diffusion alignment” story more heuristic than the text suggests. It does not necessarily invalidate the empirical success, but it weakens the claimed theoretical motivation.

3. **Ambiguities and inconsistencies in loss formulations and notation.**  
   Several mathematical definitions involving the symmetry and frequency losses are unclear:
   - Eq. (13) defines \(\mathcal{L}_{sym}^T\) with a weight map \(M\) computed from the DWT of \(y'\), but the construction of \(M\) is not fully specified: “summed and normalized to form a weight map” leaves open whether normalization is per-pixel, per-channel, or global, and whether gradients flow through M.
   - Eq. (16) then introduces \(\mathcal{L}_{sym}^{FT}\) with an additional \(\mathcal{L}_{fre}(M'\otimes y', M'\otimes y)\). However, \(\mathcal{L}_{fre}\) in Eq. (14–15) is defined over Fourier transforms of HR images \(x', x\) indexed in frequency domain \((u,v)\); applying the same notation to LR-level patches weighted by M’ is conceptually odd, and no details are given about padding / transform resolution.
   - In Table 6, the loss labels \(\mathcal{L}_1^{Sym}\), \(\mathcal{L}_{1o}^{Sym}\), and \(\mathcal{L}_{1o}^{SR}\) do not map clearly back to Eqs. (13–16); the notation seems inconsistent and makes it hard to verify what exactly was ablated.
   - Eq. (17) uses \(\mathcal{L}_{sym}^{FT}(LDP(\hat{x_0}, y_{hf}), y)\) with gradient \(\nabla_{x_t}\); however, the chain rule through decoding, LDP forward, and the symmetric losses is not discussed, and it is unclear how gradients are stabilized.  
   These issues collectively make the mathematical presentation less precise than it should be for a denoising-based degradation model.

4. **Evaluation of the degradation model itself is incomplete and somewhat contradictory.**  
   The degradation modeling experiments in Sec. 4.2 and **Tables 1–2** raise several questions:
   - **Table 1** shows that DRN achieves higher PSNR/SSIM and lower LPIPS than LDP on “Down” and “JPEG”, while LDP outperforms on “Noise”, “Blur”, “Hybrid” in LPIPS and SSIM, but with often lower PSNR than DRN. Yet the text concludes that “LDP performs consistently well across all degradation types” and implies it is superior. A more nuanced discussion is needed, especially since DRN is much closer to simple downsampling (Table 2) but still matches or beats LDP in some metrics.
   - **Table 2** reports similarity between the predicted LR and downsampled SR, interpreted as a proxy for trivial downsampling. However, LDP’s PSNR to downsampled SR is not extremely low; the gap between Table 1 and Table 2 metrics is not quantified in a clear way, and no curve is shown to justify the chosen threshold for “not degenerate”.
   - Qualitative **Figure 3** shows that LDP degrades high-frequency structures more aggressively than DRN and DualSR, but there is no perceptual evaluation (e.g., human study) to reconcile the numerical and visual differences.  
   Overall, while LDP clearly is not just bicubic downsampling, the current analysis does not fully clarify what sort of degradation distribution it actually learns.

5. **Dependence on synthetic BSRGAN-based degradation in both LDP training and SR fine-tuning.**  
   LDP is trained solely on LSDIR with BSRGAN-style synthetic degradations (Sec. 4.1), and fine-tuning of SR models also uses DF2K with BSRGAN (Appendix D). This is a strong assumption: the degradation distribution for LDP and for SR fine-tuning are the same or very similar. Consequently:
   - Improvements on synthetic DIV2K-bsrgan_plus benchmarks (Table 3, Figure 4, Figure 7) partially reflect better matching to that synthetic pipeline, rather than clearly improved robustness to “unknown” degradations.
   - Real-world performance gains in **Table 4** are mixed and sometimes negative, especially for FeMaSR on DPED and RealSRSet (e.g., CLIPIQA and MUSIQ drop substantially). This suggests that the BSRGAN-trained LDP may over-regularize away certain textures when the real-world degradations differ from the training distribution.  
   The paper does not explore training LDP on more diverse or real-based degradations, nor ablate the mismatch between LDP’s training degradations and test-time degradations.

6. **Real-world results are not consistently positive and are under-analyzed.**  
   In **Table 4**, many metrics get worse in certain configurations:
   - FeMaSR+LDP on DPED: MANIQA, CLIPIQA, MUSIQ all drop notably, and NIQE increases; similar degradations occur on RealSRSet for FeMaSR.  
   The text partially acknowledges this for FeMaSR, claiming that “no-reference metrics may favor visually striking but structurally inaccurate results”. However, this explanation is hand-wavy:
   - **Figure 5** shows that LDP often smooths out GAN-induced artifacts, which is good; but it also sometimes reduces fine texture contrast, making images somewhat less sharp.
   - No user study or more interpretable-metric analysis is provided to argue that the metrics are in fact misaligned with human perception in those cases.  
   Given that the main selling point is better generalization to “unseen real-world degradations”, these non-trivial regressions should be discussed more systematically, with perhaps more architectures and training settings where LDP is not beneficial.

7. **Posterior sampling experiments show marginal or mixed gains relative to added complexity.**  
   While the conceptual integration with DPS is sound, **Table 5** shows fairly modest and sometimes negative changes:
   - For LDM on RealSR, MANIQA, CLIPIQA, MUSIQ, and QAlign all slightly worsen with LDP.
   - For UPSR on DPED, CLIPIQA and QAlign slightly degrade.  
   **Table 13** further indicates that the strongest PSNR/LPIPS gains come from applying LDP at every diffusion step (LDPtV1), which increases inference time from 19 s to 178 s per image, whereas the cheaper LDPtV3 configuration yields only very slight gains. This makes the posterior sampling use case look less compelling, yet the paper’s narrative in Sec. 4.4 is quite positive and underplays these trade-offs.

8. **Some architectural design choices appear ad hoc and are under-justified.**  
   Examples:
   - Patch-wise timesteps \(t_i\in[500,1000]\) (Page 6) are chosen “to align the noisy HR and LR features” without ablation on the time range, or explanation of why late diffusion timesteps are always preferable. Early timesteps might also be informative.
   - The scale factor \(s'\) for constructing \(y_{hf}\) is fixed at 2 in main experiments; **Table 10** shows some trend, but there is no intuition about why \(s'=2\) is optimal relative to, say, using a learned low-pass filter for the residual.
   - The prompt dimension \(N_p\) and conditioning scheme in DPM (Eqs. 5–6, Figure 2(b)) draw on PromptIR, but there is no ablation of non-prompt baselines (e.g., pure CNN from \(y_{hf}\) to C’) to justify the extra complexity.  
   These issues make the method feel more like an empirically tuned system than a principled design.

9. **Related work on diffusion-based SR and denoising-based degradation modeling is incomplete.**  
   The paper cites DR2, ILVR, DPS, and some diffusion SR works (e.g., Wang et al. 2024; Zhang et al. 2025), but misses or barely discusses several closely related recent methods (see next section). This contributes to the overstatement of novelty and the somewhat shallow comparison with diffusion-based SR methods that also rely on LR-consistency or robust degradation priors.

10. **Presentation: a number of typos, inconsistent variable names, and table formatting errors.**  
   There are multiple minor issues:
   - In Sec. 4.4, **Table 5**’s column headers contain typos such as “MANXQA”, “CLIPQA”, “QAlqut”, which is distracting.
   - In references, some citations are malformed (e.g., “Hyungjin Chung et al. [2023] ... Learning a deep convolutional network for image super-resolution” clearly mismatched).
   - Some equations use inconsistent capitalization (\(LR_{hf}\) vs \(y_{hf}\)), and occasional missing subscripts (e.g., Eq. 17’s \(p_t(x_t|y)\) vs previously defined distributions).  
   These do not fundamentally undermine correctness but do reduce clarity.

Given the number and severity of these issues, particularly the limited theoretical justification and the somewhat mixed real-world and diffusion sampling results, I view the current version as falling short of clear ICLR acceptance despite its substantial empirical and engineering effort.

## Potentially Missing Related Work

Below are directly relevant works that should be discussed and compared:

1. **Li, H., Yang, Y., Chang, M., “SRDiff: Single Image Super-Resolution with Diffusion Probabilistic Models”, 2021.**  
   - Relevance: One of the earliest diffusion-based SISR models, explicitly modeling SR as a diffusion process. SRDiff also emphasizes robustness and explores conditioning on LR images.  
   - Where to cite: Section 2.1 & 2.2 when discussing diffusion-based SR, and in Sec. 4.4 when positioning LDP’s posterior sampling relative to SR-specific diffusion priors.

2. **Niu, A., Zhang, K., Pham, T. X., “CDPMSR: Conditional Diffusion Probabilistic Models for Single Image Super-Resolution”, 2023.**  
   - Relevance: Conditional diffusion model tailored for SR, focusing on efficient and flexible conditioning schemes. Closely related to the idea of treating SR as a conditional denoising process from LR to HR.  
   - Where to cite: Section 2.1 and 3.1–3.3, especially since LDP leverages a similar notion of conditional denoising with high-frequency guidance.

3. **Niu, A., Zhang, K., Pham, T. X., “ACDMSR: Accelerated Conditional Diffusion Models for Single Image Super-Resolution”, 2023.**  
   - Relevance: Explores accelerated diffusion-based SR with fewer sampling steps; relevant given LDP’s use in posterior sampling and the inference time discussion (Table 13).  
   - Where to cite: Section 2.1 and 4.4, as a reference point for efficient diffusion-SR sampling.

4. **Sahak, H., Watson, D., Saharia, C., “Denoising Diffusion Probabilistic Models for Robust Image Super-Resolution in the Wild” (e.g., SR3+), 2023.**  
   - Relevance: Focuses explicitly on robustness to real-world degradations using diffusion SR, directly aligned with this paper’s goal of generalization to unknown degradations.  
   - Where to cite: Section 1 and 2.1, and contrasted in Sec. 4.3/4.4 to highlight how LDP’s degradation plug-in differs from full diffusion SR models.

5. **Xiao, H., Wang, X., Wang, J., “Single Image Super-Resolution with Denoising Diffusion GANs”, 2024.**  
   - Relevance: Combines diffusion and GANs for SR, tackling robustness and sampling speed. LDP’s use of diffusion-style corruption and GAN-based backbones (FeMaSR) would benefit from contrast with this line of work.  
   - Where to cite: Section 2.1, and when interpreting FeMaSR+LDP behavior in Sec. 4.3.

6. **Wang, X., Yan, J.-K., Cai, J.-Y., “Super-Resolution Reconstruction of Single Image for Latent Features”, 2024.**  
   - Relevance: Addresses SR in latent feature space with emphasis on fast sampling and high-quality reconstructions; relates to LDP’s use with latent diffusion models (StableSR, LDM, UPSR).  
   - Where to cite: Section 2.2 and 4.4, as part of the broader latent-space SR discussion.

Incorporating and explicitly contrasting with these works will clarify where LDP truly advances the state of the art versus where it is a pragmatic engineering refinement.

## Questions

1. **Clarification of the “HR–LR alignment” claim and choice of timesteps.**  
   - Could the authors formalize or empirically verify the statement that “after noise is added, HR and LR features become aligned” (Sec. 3.1, citing Wang et al. 2023b) in the context of their specific degradation model (Eq. 1)? For instance, can you show distributions of \(\|x_t - y_t^\uparrow\|\) as a function of t to justify picking \(t_i\in[500,1000]\)?  
   - Would results change substantially if timesteps for the Noise Addition Module were sampled from a wider range or focused on mid-range t?

2. **More precise definition of the symmetry and frequency losses.**  
   - For Eq. (13), please specify exactly how the DWT subbands are combined and normalized into M (per-channel, per-pixel normalization, etc.), and whether gradients flow through M.  
   - For Eq. (16), how is \(\mathcal{L}_{fre}(M'\otimes y', M'\otimes y)\) implemented, given that Eq. (14–15) originally defined it for HR images x’, x? Is the Fourier transform taken over the full LR image (after applying M’) or patch-wise? Clarifying these implementation details would increase confidence in the loss design.

3. **Trade-offs on real-world benchmarks and no-reference metrics.**  
   - For FeMaSR + LDP on DPED and RealSRSet, several metrics degrade (Table 4). Could the authors provide qualitative zoom-ins for these failure cases and discuss whether human observers indeed prefer +LDP outputs?  
   - Are there training hyper-parameters (e.g., smaller \(\tau\), lower loss weights) that alleviate these regressions without sacrificing synthetic performance?

4. **Comparison against more recent diffusion-based SR and degradation modeling baselines.**  
   - Can the authors either run or at least discuss comparisons against diffusion-based SR methods like SRDiff, CDPMSR, ACDMSR, or SR3+ on at least one synthetic and one real-world benchmark? Since these explicitly address robustness to real degradations, it would be important to see where a plug-in like LDP stands relative to directly training a diffusion SR model.

5. **Ablation on the Degradation Prediction Module and prompt mechanism.**  
   - How much does the use of the degradation prompt \(P_D\) (Eq. 5–6, Figure 2(b)) contribute compared to a simpler CNN that maps \(y_{hf}\) to a feature map C’ directly? An ablation removing prompt-style conditioning would help isolate the value of this design choice.  
   - Relatedly, is the LR residual-based condition \(y_{hf}\) really necessary, or would using full LR y (or a heavily blurred version thereof) as condition lead to degenerate shortcuts in practice? An empirical comparison would be useful.

6. **Practical guidance on when/how to use posterior sampling with LDP.**  
   - Given the substantial inference overhead in **Table 13**, could the authors provide practitioner guidelines on when LDP posterior sampling is worth the cost? For example, is there a setting where LDP is crucial to removing StableSR’s repeat-spot artifacts and no other cheaper technique suffices?

Clarifying these points and, where possible, adding small-scale experiments in a rebuttal would substantially increase my confidence in the method and its interpretation.

## Flag For Ethics Review

No ethics review needed.

## Details Of Ethics Concerns

N/A.

## Soundness Rating

2 fair.  
The method is empirically validated across multiple backbones and benchmarks with extensive ablations, but the theoretical underpinnings (particularly HR–LR diffusion alignment and some loss formulations) are under-specified or heuristic, and some real-world and diffusion results are mixed.

## Presentation Rating

3 good.  
The core idea and architecture are clearly visualized (Figure 2) and most experimental details are provided, but there are non-trivial notation ambiguities, minor table/typo issues, and missing clarity around key losses and assumptions.

## Contribution Rating

2 fair.  
The work provides a useful and generally effective degradation plug-in with broad applicability, but the conceptual novelty over existing degradation modeling and diffusion-based SR frameworks is moderate, and robustness/generalization claims are not uniformly supported.

## Overall Rating

4 Marginally below the acceptance threshold. But would not mind if paper is accepted.  
The paper presents a well-engineered and fairly lightweight degradation plug-in that improves multiple SR backbones on synthetic multi-degradation benchmarks and often on real-world datasets, with thorough ablation. However, its conceptual novelty is modest, the theoretical justification is thin, several design choices feel ad hoc, and real-world & diffusion posterior sampling results are mixed and under-analyzed. With strengthened theoretical grounding, clearer loss definitions, more balanced discussion of failures, and better positioning against recent diffusion-based SR literature, this could reach ICLR standard.

## Reviewer Confidence

4 confident.  
I am familiar with SR, degradation modeling, and diffusion-based inverse problems, and have carefully checked the equations, experiments, and comparisons, though I did not attempt to re-implement the method.