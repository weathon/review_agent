=== CALIBRATION EXAMPLE 37 ===

# Final Consolidated Review
Now I have enough context to produce the consolidated review. Let me synthesize everything carefully.

---

## Summary

AdvI2I proposes an adversarial framework for Image-to-Image (I2I) diffusion models that injects perturbations into conditioning images — rather than text prompts — to induce NSFW content generation. The paper first motivates the image-attack direction by showing that simple text filters substantially reduce the effectiveness of existing adversarial prompt attacks. It then trains a VAE-based adversarial generator by aligning latent features of the adversarial-conditioned run with those guided by a NSFW-concept-shifted embedding. AdvI2I-Adaptive extends the base framework with an additional safety-checker evasion loss and Gaussian noise augmentation during training.

---

## Strengths

- **Empirically grounded motivation for the image-attack modality.** Table 2 is a concrete and novel contribution: it systematically shows that four independent text filters (perplexity, keyword, LLM, embedding) reduce ASR of existing adversarial prompt attacks by 58–100%, thereby establishing a clear capability gap that image-based attacks fill. Most prior work simply assumes text filters are impractical without actually measuring it.

- **Adaptive attack closes the Safety Checker gap convincingly.** The non-adaptive AdvI2I collapses under Safety Checker (18% ASR), but AdvI2I-Adaptive recovers to 70.5–72.0% across both models and both NSFW concepts, surpassing MMA's SC-resistant performance (64.5% on InstructPix2Pix nudity). This shows that the safety-checker evasion loss is a meaningful technical addition, not just a label change.

- **Generalization analysis across unseen images and prompts.** Table 5 demonstrates that the trained generator achieves ASR >63.5% on held-out images and >68.5% on held-out prompts, providing some evidence that the vulnerability is systemic rather than a per-sample overfit.

- **Cross-model evaluation of the attack.** Results are reported on both InstructPix2Pix and SDv1.5-Inpainting, and Appendix A.1 includes a transferability analysis to other SD inpainting models, showing the attack is not narrowly tuned to one architecture.

---

## Weaknesses

### Fatal
None identified.

### Major

- **Dataset is small and domain-biased.** The 400 training/evaluation images come exclusively from the "sexy" category of an NSFW scraper, with NSFW images filtered out. The remaining pool consists of near-NSFW, human-body-centric images. All "unseen" images in Table 5 are drawn from the same 400-image pool. This means the attack generator is trained and tested entirely on a near-NSFW distribution; claimed ASRs are not evidence that the attack works against ordinary benign images (e.g., portraits, street scenes, COCO images). A realistic deployment of such an attack would target generic images, not near-NSFW ones. Without testing on genuinely benign, diverse inputs, the real-world threat level is substantially overstated.

- **No perceptual image quality metrics despite an imperceptibility claim.** The optimization constraint $\|g_\psi(x) - x\|_p \leq \epsilon$ is the paper's only evidence of visual indistinguishability. Table 6 evaluates $\epsilon \in \{32/255, 64/255, 128/255\}$; $\epsilon = 128/255$ is large enough to produce plainly visible corruption. Yet the paper reports no PSNR, SSIM, LPIPS, or human study at any $\epsilon$ value. Without these, the core "stealthy" characteristic of the attack — central to the claim that image attacks are harder to detect than text attacks — is entirely unverified.

- **Notation inconsistency in $\mathcal{L}_{sc}$.** Equation (3) writes $\cos\!\bigl(\mathcal{D}(f_\theta^t(g_\psi(x)),\, \tau_\theta(p)),\, C_i\bigr)$, making $\tau_\theta(p)$ (the text embedding) a second argument to the VAE decoder $\mathcal{D}$, which is architecturally incoherent — a VAE decoder does not consume text features. Algorithm 1 (line 11) corrects this to $\cos\!\bigl(\mathcal{V}(\mathcal{D}(f_\theta^t(g_\psi(x)),\, \tau_\theta(p))),\, C_i\bigr)$, applying the Safety Checker's vision encoder $\mathcal{V}$ to the decoded image before cosine similarity, which is the correct formulation. The text description also matches Algorithm 1. Nevertheless, Eq. (3) as printed is technically wrong, creating genuine confusion about how the safety-checker loss is actually computed.

### Minor

- **No ablation on the $t=1$ timestep choice.** The paper selects only the final denoising step's latent feature as the optimization target, with the justification that "the latent feature at the final timestep directly influences the content of the generated image" (footnote 1). While plausible, diffusion-model outputs depend on the full denoising trajectory. There is no ablation comparing $t = 1$ to intermediate timesteps or averaging across timesteps. This design choice is foundational and deserves empirical justification.

- **Generator architecture and training details are under-specified.** The paper states that $g_\psi$ is "a pre-trained VAE" but does not specify which components (encoder, decoder, or both) have trainable parameters, what the learning rate schedule is, or how many training iterations are run. Given that the VAE being fine-tuned is likely the SD VAE, omitting what is frozen vs. updated is a reproducibility gap.

- **No formal threat model.** The paper assumes white-box access to model weights, the safety checker, and gradient flow through the generator — but never states this explicitly. Without a threat model, it is unclear which aspects of the attack require access to internals and whether any component is transferable to a gray- or black-box setting.

- **Evaluation relies solely on automated NSFW classifiers (NudeNet, Q16), with no human validation.** Because the adversarial generator is optimized to push the diffusion model's latent features toward NSFW concept embeddings, there is a non-trivial risk that generated images fool NudeNet/Q16 via classifier-specific artifacts rather than genuinely depicting NSFW content. A human evaluation on even a small random sample would substantially increase the credibility of the reported ASRs, particularly for the safety-critical claim this paper makes.

- **Model coverage is limited to the SD v1.5 family.** Both InstructPix2Pix and SDv1.5-Inpainting are fine-tuned from SDv1.5. The broad claim that I2I models face "urgent security concerns" is weakened by the absence of results on architecturally distinct models (e.g., SDXL, SD3, or IP-Adapter-style conditioning). If the attack exploits SD v1.5-specific architecture or latent space properties, the threat may not generalize.

- **Abstract slightly misrepresents the Safety Checker result for the non-adaptive version.** The abstract states that AdvI2I "circumvents existing defense mechanisms, such as Safe Latent Diffusion," but under Safety Checker, the non-adaptive AdvI2I drops to 18% — well below MMA's 64.5% under the same defense. While this is resolved by AdvI2I-Adaptive, the abstract's framing implies that the base method handles all tested defenses, which it does not. This should be clarified.

- **Minimal ethical discussion.** The paper provides a single-sentence caution about NSFW content. Given that it presents a complete, working recipe for bypassing safety mechanisms in widely-deployed models, a more substantive ethics section covering limitations on reproduction, responsible disclosure, and proposed countermeasures is warranted at an ICLR-level venue.

### Tiny

- **Notation inconsistency in Algorithm 1 Step 1:** The NSFW concept vector extraction uses $\psi_\theta(\cdot)$ in Algorithm 1 but $\tau_\theta(\cdot)$ everywhere in the text and equations. This is a straightforward copy-paste error.

- **"Unseen" evaluation is in-distribution.** Table 5 reports results on "unseen images and prompts," but both sets are subsets of the same 400-image, 30-prompt pool used for training. True generalization to out-of-distribution inputs (e.g., CelebA, COCO) is not tested.

---

## Nice-to-Haves

- **Propose or sketch a defense.** Papers that expose vulnerabilities are more impactful when they also suggest a credible direction for mitigation (e.g., adversarial training on the image encoder, latent-space anomaly detection on the conditioning image). The paper currently ends with a call for "stronger defenses" without specifying what form they might take.

- **Expand to additional NSFW categories (hate symbols, graphic injury).** Testing is limited to nudity and violence; demonstrating breadth across more harmful concept categories would strengthen the threat assessment.

- **Discuss the connection to artist-protection attacks (Glaze, PhotoGuard, Anti-DreamBooth).** AdvI2I is, in a formal sense, the inverse direction of these protection attacks: protection attacks steer diffusion models *away* from a concept; AdvI2I steers them *toward* one. The paper cites Glaze as a baseline ("Attack VAE") but does not articulate this optimization duality. An explicit discussion would situate the contribution better.

- **Report computational cost.** Training time and per-image inference overhead compared to the "W/o Generator" per-image baseline would help practitioners understand the practical trade-off of the generator-based approach.

- **Statistical reliability.** While single-run evaluation is accepted practice in adversarial ML papers, reporting variance over multiple generation seeds — at least for the key headline numbers — would strengthen the comparisons in Table 3, where differences between methods are sometimes only a few percentage points (e.g., AdvI2I-Adaptive 70.5% vs. MMA 65.5% under SC).

---

## Removed Points

*These points were flagged for removal or significant weakening; treat with caution.*

- **[REMOVED] "Adversarial text prompt evaluation as a non-contribution."** The harsh critic argued that Table 2 is "merely a baseline analysis." However, systematically benchmarking four text-filter types against five prompt-attack methods is a concrete empirical contribution that motivates the paper's core direction and would be new information to many readers.

- **[REMOVED] "MMA outperforms AdvI2I under SC, undermining the main claim."** The harsh critic presents this as fatal, but ignores that (a) AdvI2I-Adaptive achieves *higher* ASR under SC than MMA on InstructPix2Pix nudity (70.5% vs. 64.5%), and (b) without defense, AdvI2I substantially outperforms MMA (81.5% vs. 68.5%). The comparison is not a failure of the method; it is addressed by the adaptive variant.

- **[REMOVED] Title typo "ADV12I" vs. "AdvI2I".** Pure OCR/formatting artifact from text extraction; not a substantive issue.

- **[REMOVED – WEAKENED ABOVE] "MMA's SC robustness weakens the motivation for AdvI2I-Adaptive."** The harsh critic argued that if MMA already handles SC, AdvI2I-Adaptive is unmotivated. This ignores that AdvI2I-Adaptive outperforms MMA under SC (70.5% > 64.5%) while maintaining better baseline ASR. The comparison actually validates the adaptive extension.

- **[REMOVED] Demand for confidence intervals / multiple-run statistics as a core weakness.** Single-run evaluation is the community norm for adversarial attack papers. Not a genuine weakness per the evaluation standards of this field.

- **[REMOVED] Claim that text-filter false-positive rates must be reported for the paper to be valid.** The text filter analysis is motivational context; the paper is not proposing a defense. The absence of false-positive rates does not invalidate the core claim that ASR drops dramatically under these filters.

---

## Novel Insights

The most genuinely novel observation — surfaced by the spark finder but not fully articulated in the paper itself — is that AdvI2I is the *dual* of artist-protection attacks: where Glaze/PhotoGuard optimize perturbations to steer a diffusion model *away* from a target artist's style, AdvI2I steers it *toward* a harmful concept. This optimization duality means that any progress in making protection attacks more effective will simultaneously provide blueprints for making NSFW-induction attacks more effective, and vice versa. This creates a structural tension in the "protective perturbation" line of work that the paper could highlight but does not. A second insight — visible in Table 3 — is that the Safety Checker defense differentially defeats image-only attacks (AdvI2I: 18%) while leaving image-and-text co-optimization attacks (MMA: 64.5%) largely intact; this suggests that current SC architectures are implicitly tuned to detect feature patterns in purely image-conditioned generation rather than the combined conditioning regime.

---

## Suggestions

1. **Re-run the evaluation on genuinely benign, diverse images** (e.g., CelebA faces, COCO person images) to replace or supplement the near-NSFW "sexy" source images. This is the single most important fix for credibility.
2. **Add LPIPS or SSIM** at each $\epsilon$ level in Table 6 to quantify imperceptibility alongside ASR, even if in the appendix.
3. **Fix Equation (3):** Remove $\tau_\theta(p)$ from inside $\mathcal{D}(\cdot)$ and align Eq. (3) with Algorithm 1 line 11, using $\mathcal{V}(\mathcal{D}(\cdot))$ notation.
4. **Add a sentence-level threat model paragraph** in Section 3.2 stating the attacker's assumptions (access to model weights, safety checker architecture, gradient access).
5. **Include at least one test on an architecturally distinct model** (e.g., SDXL with inpainting) in the main results or appendix.
6. **Provide ablation on timestep selection** ($t = 1$ vs. $t \in \{10, 50, T\}$) to justify the design choice.
7. **Perform a small-scale human study** (e.g., crowdsourced on 100 randomly sampled outputs) to validate NudeNet/Q16 accuracy on adversarially generated images.

---

**Overall evaluation:** The paper identifies a real and underexplored attack surface, and the core empirical results are non-trivial. However, it is currently held back by a biased and small evaluation dataset, the absence of imperceptibility metrics, a notation error in the central adaptive-loss equation, and limited model coverage. These are addressable through revision but represent substantive gaps rather than cosmetic issues. The technical novelty is moderate — the framework competently combines existing components (VAE-based generator, NSFW concept steering, adversarial safety-checker evasion) in a new setting, but does not introduce a fundamentally new algorithmic idea. The significance is genuine given the societal risk. In its current form the work requires major revision to meet ICLR standards; with the suggested fixes, particularly the dataset and metrics issues, it could become a solid contribution.

# Actual Human Scores
Individual reviewer scores: [5.0, 6.0, 6.0, 5.0]
Average score: 5.5
Binary outcome: Reject
