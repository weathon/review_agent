=== CALIBRATION EXAMPLE 9 ===

# Final Consolidated Review
## Summary
This paper proposes **SecureGS**, a 3D Gaussian Splatting steganography framework built on Scaffold-GS-style anchors. The core idea is to decouple public/original-scene Gaussian generation from hidden-content generation by storing hidden offsets and attributes in private MLPs, and to add a region-aware density optimization (RDO) scheme that densifies public anchors around hidden-object regions to reduce visible geometric leakage in the published point cloud. Empirically, the method improves rendering fidelity and speed over GS-Hider and qualitatively reduces hidden-geometry leakage in point-cloud visualization.

## Strengths
- **It identifies and directly targets a concrete weakness of prior 3DGS steganography: visible hidden geometry in the public point cloud.** The motivation is specific and well-evidenced in the paper via Fig. 1(b), Fig. 4, and Fig. 7, where hidden-object structure is visible for GS-Hider and for SecureGS without RDO, but much less visible with the full method.
- **The representation design is technically well matched to the stated engineering goal.** In Sec. 3.3, the paper explicitly stores original-scene offsets while moving hidden offsets/attributes into private MLPs (`F_o^\dagger`, `F_c^\dagger`, `F_\alpha^\dagger`, etc.), which is a sensible way to avoid exposing hidden geometry directly in public explicit Gaussian parameters.
- **The empirical gains over the main directly relevant baseline (GS-Hider) are meaningful on fidelity/efficiency.** In Table 1 and Table 5, SecureGS improves both original-scene and hidden-content PSNR while substantially increasing FPS and reducing storage relative to GS-Hider. This is the strongest part of the submission and supports the claim that the new architecture is practically better than the prior coupled-feature design.
- **The paper demonstrates modality flexibility within one framework.** Beyond hiding 3D objects, the same framework is extended to hidden images and bits (Sec. 4.6, Tables 3 and 5), suggesting the anchor-based design is not narrowly overfit to a single payload type.
- **The decoupled hidden-anchor visualization in Fig. 6 is a useful diagnostic result.** It shows that the hidden object is still represented coherently in the private branch even while the public anchor cloud is visually less revealing, which helps explain why the method can improve concealment without collapsing hidden-content reconstruction.

## Weaknesses
### Fatal
- None.

### Major:
- **The central “security” claim is under-supported by the evaluation and framing.** The paper defines security primarily as (i) file-format consistency with Scaffold-GS and (ii) lack of visually obvious hidden geometry in point-cloud visualization (Sec. 1, Sec. 3.2, Sec. 4.3). That is a legitimate engineering notion of concealment, but it is much narrower than the broader “secure and reliable” language used in the abstract, introduction, and conclusions. There is no explicit threat model, no steganalysis-style detectability evaluation, no attack study for an informed adversary with access to the public anchors, and no quantitative metric for hidden-content detectability. As a result, the paper convincingly shows **reduced visual/geometric leakage**, but it does **not** establish security in a stronger adversarial sense.
- **Security ablations are missing even though security is the main novelty claim.** Table 4 studies HDGER and RDO only through PSNR/SSIM and size. But Sec. 4.5 itself states these modules are “essential components for ensuring the security of SecureGS.” If so, the paper should quantify the security benefit of adding them, rather than relying only on qualitative figures. Right now the reader can see the fidelity cost of RDO, but cannot judge the fidelity–security tradeoff numerically.
- **The experimental evidence supports superiority over GS-Hider on the tested settings, but not the broader wording used throughout.** The strongest like-for-like comparison is against GS-Hider. For object hiding, the paper explicitly notes that prior methods struggle on isolated-object (“object-level”) hiding and therefore compares them in “scene-level” settings instead, while SecureGS is additionally shown in object-level mode with much higher hidden PSNR (Sec. 4.2, Table 1, Fig. 5). Those object-level numbers are interesting, but they are not directly comparable to prior methods. Likewise, the bit-hiding experiments compare against NeRF watermarking methods rather than a directly matched 3DGS bit-hiding baseline. So the evidence supports a narrower claim—strong improvement over GS-Hider plus promising flexibility across tasks—more than the paper’s repeated broad claim of surpassing existing methods across steganography settings.
- **The paper does not define a clear attacker model for “authorized” versus “unauthorized” users.** Sec. 3.3 says the hidden offsets and attributes are retrievable only by authorized users through private MLPs, but the operational security story is left vague: what exactly is assumed public, what is secret, and what adversarial access is being defended against? This matters because the method’s security framing rests on private decoders, yet the paper never concretely states the access assumptions under which hidden content is considered protected.

### Minor
- **The robustness analysis is too limited for a steganography/security paper.** Sec. 4.4 reports only random pruning of anchors. That is useful, but it is a weak corruption model relative to likely manipulations of 3D assets; more realistic edits or post-processing would better characterize robustness.
- **There is a concrete inconsistency in the robustness section.** The text says “Even at a larger pruning rate of 25%,” but Table 2 reports results only up to **20%** pruning. This should be corrected.
- **Some methodological choices are under-analyzed despite being central to the method.** The RDO procedure depends on `τ_fix`, `r_down`, DBSCAN clustering, and asynchronous gradient accumulation frequencies, but only one setting is provided in Sec. 4.1 and no sensitivity study is shown. Since RDO is presented as the key mechanism balancing concealment and efficiency, this leaves the method somewhat under-characterized.
- **The view-dependent hidden-offset parameterization is not fully explained.** Eq. (4) predicts hidden offsets from blended anchor features plus camera distance/direction. The paper demonstrates that this works empirically, but it does not clearly explain why this view-conditioned parameterization still yields stable hidden 3D geometry across views, which is an important modeling choice.
- **The current scope is narrower than some of the framing suggests.** Sec. 3.2 explicitly states that the method only hides a **3D object** within a 3D scene, because hiding a large-scale 3D scene makes point-cloud confidentiality difficult. That is a reasonable scope decision, but the broader framing around protecting arbitrary 3D assets could be read as more general than what is actually demonstrated.

### Trivial
- **There are a few notation / presentation inconsistencies in the extracted text.** Examples include the mismatch between “GS-Header” and “GS-Hider,” inconsistent offset notation (`O_{v@j}^{hid}` vs. `O_{v\oplus j}^{hid}`), and the table abbreviation mismatch for kitchen (`KI` vs `KL`). These do not affect the main technical content but should be cleaned up.

## Nice-to-Haves
- Add a **quantitative security metric** for geometric leakage, e.g., hidden-object detectability from public anchors, or a geometric similarity metric between public visible structure and hidden content.
- Include a simple **steganalysis experiment**, such as training a classifier to distinguish clean Scaffold-GS anchors from SecureGS anchors using anchor features/densities/statistics.
- Provide a **threat model** that explicitly states what is public, what is private, and what adversarial capabilities are assumed.
- Expand the **RDO tradeoff analysis**, varying `r_down`, `τ_fix`, and possibly the gradient accumulation schedule to show a fidelity–storage–concealment Pareto frontier.
- Add a small discussion or experiment on **capacity/scaling**, e.g., larger hidden objects or multiple hidden objects.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“Missing related-work comparisons to GaussianStego / 3D-GSW”** — removed under the instruction not to criticize missing related works, since external completeness cannot be verified here. It is still fair to note that the strongest empirical evidence is mainly versus GS-Hider, but not to penalize the paper for omitting specific cited methods.
- **Reproducibility criticism about undisclosed implementation details / exact asynchronous frequencies / full parameterization details** — weakened and largely removed as a core weakness. The paper does provide key hyperparameters in Sec. 4.1 (`λ`, `α`, `β`, `τ_fix`, `r_down`, `k`, MLP depth/width). A sensitivity study would improve the work, but this is not a decisive reproducibility flaw.
- **Claims that the paper’s private models or cited tools/benchmarks might be unavailable or unverifiable** — removed by rule.
- **Generic transferred criticisms from unrelated steganography settings (e.g., prompt sensitivity, paraphrasing attacks, exact model-version dependence in text LLM steganography)** — removed as inapplicable to this paper.

## Novel Insights
The paper’s real contribution is stronger when framed as **representation-level concealment of hidden 3D content in explicit 3DGS systems**, rather than as a general security breakthrough. The most convincing insight is that moving hidden Gaussian generation into private neural decoders is not sufficient by itself: even then, the **public anchor geometry can still leak hidden structure**, and the paper’s RDO mechanism is specifically about masking that leakage by reshaping the density profile of the public scaffold. This suggests an important distinction for future work between **parameter secrecy** and **structural secrecy** in explicit neural scene representations—a sharper framing than the paper currently gives itself.

## Suggestions
- Reframe the main claim more precisely around **geometric concealment / leakage reduction** unless stronger adversarial security evidence is added.
- Add a dedicated **threat model section** specifying what data, weights, architectures, and rendering access an attacker has.
- Introduce at least one **quantitative security evaluation**, such as public-anchor-based hidden-object detection or a geometric leakage score.
- Extend the ablation table to include **security metrics**, not only fidelity and size.
- Clarify the rationale for the **view-dependent hidden-offset predictor** in Eq. (4), and explain how cross-view consistency is maintained.
- Fix the **25% vs 20% pruning inconsistency** in Sec. 4.4 / Table 2.
- Add a small **hyperparameter sensitivity study** for RDO to substantiate the claimed security–efficiency tradeoff.
- Tighten the paper’s claims so that they match what is actually shown: the method appears **novel and practically strong as an engineering improvement over GS-Hider**, but the present evidence is not yet sufficient for broad security claims in a fully adversarial sense.

# Actual Human Scores
Individual reviewer scores: [5.0, 6.0, 8.0, 8.0, 3.0]
Average score: 6.0
Binary outcome: Accept
