=== CALIBRATION EXAMPLE 29 ===

# Final Consolidated Review
## Summary
This paper proposes PIN, an INR that replaces standard activations with Prolate Spheroidal Wave Functions (PSWFs), motivated by PSWFs’ classical optimal space-frequency concentration properties. The paper combines this activation design with a simple learnable affine modulation of the activation and evaluates it on image regression, 3D occupancy representation, inpainting, and a small NeRF setup, with generally promising empirical results but also substantial overclaiming and some important inconsistencies.

## Strengths
- **The activation choice is genuinely novel and technically well-motivated.** Using PSWFs as INR activations is a distinctive idea rooted in classical signal processing, and the paper clearly ties this choice to the space/frequency localization issues that matter in coordinate-based networks. This is more principled than many ad hoc activation proposals.
- **The paper identifies a specific INR failure mode and targets it directly.** The discussion around preserving fine detail without injecting artifacts into smooth regions is concrete and reflected in the image regression examples (Figures 2–3), rather than merely claiming generic performance gains.
- **Kodak image representation results appear meaningfully strong.** Among the experiments shown in the main paper, the Kodak image regression section is the most convincing: it evaluates over a dataset rather than a single exemplar, and the reported trend is that PIN consistently outperforms SIREN, WIRE, GAUSS, and ReLU+PE.
- **The paper attempts to connect activation structure to expressivity rather than presenting a purely empirical tweak.** Theorem 1 is not sufficient to support all of the paper’s downstream claims, but the effort to analyze PIN as a polynomial in first-layer PSWF atoms is still a substantive contribution.
- **The learnable activation modulation is practically useful.** Section 6’s use of \(\tilde{\psi}(x)=T\psi(wx)+b\) gives PIN some task adaptivity without requiring manual grid search over handcrafted activation parameters, which addresses a real nuisance in prior INR activations.

## Weaknesses

### Fatal
- **The paper’s central theoretical explanation is not technically supported as written.**  
  After Theorem 1, the paper claims: “Since \(\psi\) is band-limited, and the convolution of band-limited functions is band-limited, then \(\Phi_\theta(\mathbf{r})\) is also band-limited.” This is not a valid general justification for the claimed localization mechanism. Even granting Theorem 1’s polynomial representation, the subsequent argument does not establish the strong conclusions the paper repeatedly draws about preserved band-limitation and rapid spatial decay of the full network output. Because the paper’s main novelty is not just empirical performance but the claim that PSWF optimal concentration *explains* superior expressivity/generalization, this unsupported theoretical leap materially weakens the core contribution.

### Major:
- **There is a direct inconsistency between the inpainting claims and the numbers shown in Figure 5.**  
  The text says PIN is “the only architecture that maintains the highest PSNR value in both instances” and “the top image inpainting performer,” but the visible table under Figure 5 shows WIRE at **25.56 dB** versus PIN at **23.18 dB**. This is not a minor presentation issue: it undermines trust in one of the headline application claims. If the figure only reflects one of two settings, the paper needs to state that clearly; as written, the main-text claim is contradicted by the displayed results.
- **The empirical evidence is too limited for the breadth of the paper’s claims.**  
  The abstract and conclusion claim strong gains across image representation, 3D shapes, inpainting, novel view synthesis, denoising, and edge detection. But in the main paper, several of these are supported only weakly: occupancy uses only two shapes, NeRF uses only a single scene (“drums”), and denoising/edge detection are not shown in the main body at all. For a paper making broad superiority claims across INR tasks, the evidence shown is not commensurate with the scope of the claims.
- **The NeRF evaluation is especially underpowered.**  
  Section 7.5 evaluates on one scene with a vanilla NeRF and reports a small gain over GAUSS (25.70 vs. 25.21 PSNR). That is not enough to substantiate the paper’s strong claims about novel view synthesis, especially when no multi-scene averages or robustness evidence are provided.
- **The paper does not isolate what actually drives the gains: PSWF itself or the added learnable activation modulation.**  
  Section 6 introduces \(\tilde{\psi}(x)=T\psi(wx)+b\), but the experimental section does not cleanly disentangle the contribution of the PSWF shape from the contribution of this adaptive scaling/shifting. Since the paper explicitly criticizes prior methods for hyperparameter sensitivity, this matters: without controlled ablations, it remains unclear whether the gains are due to PSWF’s intrinsic properties or simply to giving the activation additional learnable flexibility.
- **The practical cost of PSWF activations is not analyzed.**  
  The method relies on numerical approximation of PSWFs rather than a simple closed-form activation such as sine or Gaussian. The paper provides no training-time, inference-time, or memory analysis. Since the claimed gains are modest in some settings and PSWF evaluation is plausibly more expensive, this omission limits the practical significance of the contribution.

### Minor
- **Theoretical claims are overstated relative to what Theorem 1 actually proves.**  
  Theorem 1 shows a polynomial-in-PSWF form under a polynomial approximation assumption for the activation. It does not by itself prove the stronger downstream claims repeatedly made in Sections 4–5 about superior localization/generalization behavior. The paper should state the theorem’s scope more carefully.
- **The approximation assumptions in Theorem 1 are underspecified.**  
  The theorem assumes PSWF can be approximated by a polynomial of degree \(K\), but the paper does not discuss the approximation regime, truncation error, or how this affects the claimed expressivity/localization conclusions.
- **The PSWF-specific design space is not adequately ablated.**  
  The paper uses PSWF of order 0 and discusses bandwidth-related properties, but the main paper does not analyze sensitivity to PSWF order, the underlying PSWF bandwidth parameter, or initialization of the learnable modulation parameters. The existing ablation mostly studies generic network hyperparameters instead.
- **Some conclusions from qualitative examples are stronger than the evidence supports.**  
  For instance, Section 7.2 claims PIN “can resolve this wide frequency spectrum challenge” based essentially on one example image. This is suggestive, but not sufficient to establish a general capability.
- **Some experiments would benefit from more objective task-specific metrics.**  
  In occupancy representation, the paper relies heavily on visual comparison when GAUSS has similar reported SSIM to PIN. If visual artifact reduction is a key point, more discriminative shape-quality metrics would strengthen the claim.

### Trivial
- **A few methods appearing in Figure 5 are not introduced in the main text.**  
  C-INR and Susper appear in the inpainting figure but are not contextualized in the main narrative, which makes that comparison harder to interpret.

## Nice-to-Haves
- Add spectral analyses of reconstructed outputs, not just activation plots, to verify the claimed balance of low- and high-frequency preservation.
- Include per-pixel error maps or region-wise analyses to support the claim that PIN improves both smooth areas and detailed areas simultaneously.
- Discuss failure cases or regimes where PSWF activations are less advantageous.
- Expand the robustness analysis across seeds if feasible, especially in settings where gains are small.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **Complaints about missing unrelated or external baselines (e.g., Gaussian Splatting, video diffusion methods, unspecified newer methods).**  
  These are outside the paper’s stated scope as an INR-activation paper and/or rely on external knowledge not verifiable from the submission.
- **Generic reproducibility complaints about omitted training details or hyperparameters in the main text.**  
  For this kind of paper, such details may reasonably live in the appendix/supplement and are not themselves a core scientific flaw.
- **Claims that the paper is invalid because some cited tools/methods are unavailable or unverifiable.**  
  Per instruction, cited entities are assumed to exist.
- **Purely generic praise such as “the paper is well-written” or “the topic is important.”**  
  These were omitted because they do not identify specific strengths of this paper.

## Novel Insights
The most interesting underlying tension in this submission is that the empirical story and the theoretical story are not equally strong. The paper may indeed have found a practically useful activation family for INRs—especially for image regression-like settings where balancing detail and smoothness matters—but the current manuscript over-attributes these gains to a propagation of PSWF optimal localization properties through the full deep network. A stronger version of this paper would likely succeed by narrowing and sharpening its claim: PSWF activations appear promising empirically and are plausibly motivated by localization theory, but the current proof machinery does not yet justify the stronger mechanistic narrative the paper adopts.

## Suggestions
- **Correct the Figure 5 inconsistency immediately.** Either the figure/table is wrong, the text is wrong, or the setup contains two cases that are not clearly separated. This must be reconciled.
- **Substantially tone down the theory claims unless they are repaired.** In particular, revise the post-Theorem 1 argument about band-limitation/localization, and clearly distinguish what is proved from what is intuition.
- **Scale the empirical claims to match the evidence, or add more evidence.** At minimum, multi-scene NeRF evaluation and clearer aggregation for inpainting/3D tasks would be needed to support the breadth of the abstract and conclusion.
- **Add PSWF-specific ablations.** Analyze the effect of PSWF order, the underlying bandwidth parameter, and the learnable modulation \(T,w,b\).
- **Isolate the source of improvement.** Compare fixed PSWF vs. learnably modulated PSWF, and if possible include equally modulated baselines to test whether the gains are due to PSWF specifically.
- **Report computational overhead.** Even a simple wall-clock and memory comparison against SIREN/GAUSS/WIRE would materially improve the practical credibility of the method.
- **Tighten the paper’s claims around significance and generality.** As written, novelty is solid, but technical soundness and empirical support do not yet justify the strongest statements in the abstract and conclusion.

# Actual Human Scores
Individual reviewer scores: [6.0, 6.0, 6.0, 6.0]
Average score: 6.0
Binary outcome: Accept
