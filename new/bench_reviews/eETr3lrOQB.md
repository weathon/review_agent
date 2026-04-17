Now I have sufficient calibration data. Let me synthesize the review.

**Calibration summary:**
- FSQ (VQ-VAE made simple), scores 6,6,8,6 → Accept (poster). Novel quantization method, simple but effective, strong experiments including generation.
- Wasserstein VQ (Vector Quantization by Distribution Matching), scores 5,3,5,6 → Reject. Incremental, problematic Gaussian assumption, limited novelty.
- TaCA (Hot-Plugging Upgrades), scores 6,5,5,5 → Reject. Plug-and-play adaptation for foundation models, limited generality, incremental.
- BSQ (Binary Spherical Quantization), scores 6,6,6,6 → Accept (poster). Novel quantization, strong results, comprehensive.
- LlamaGEN/LFQ tokenizer, scores 8,8,8 → Accept. Very strong tokenizer paper with generation results.
- SimVQ (Addressing Representation Collapse), scores 3,5,5,3 → Withdrawn/Reject. Simple solution, missing generation results.

Now let me carefully evaluate the VQ-Transplant paper against these anchors.

The paper's key claims:
1. VQ-Transplant framework: plug-and-play VQ module replacement + lightweight decoder adaptation
2. MMD-VQ: nonparametric distribution-matching VQ method
3. 95%/21.8x cost reduction
4. Near state-of-the-art reconstruction fidelity

Key issues I need to verify:
1. Is the 95% cost claim misleading?
2. Is the framework truly general or VAR-specific?
3. Is MMD-VQ clearly superior to Wasserstein VQ?
4. Is the method fundamentally novel or obvious?
5. Missing downstream generation evaluation

Let me verify the paper's claims about cost in Table 1:
- VAR: OpenImages, 16×A100, 60h → 960 A100-hours
- VQ-Transplant: ImageNet-1k, 2×A100, 22h → 44 A100-hours
- The 21.8x speedup is computed as 960/44 ≈ 21.8

This is comparing different datasets (OpenImages vs ImageNet) AND different hardware (16 vs 2 GPUs). The from-scratch baselines in Table 6 run 25-35 hours on 2×A100 on ImageNet but are deliberately undertrained.

Now let me write the consolidated review.</think>Now I have sufficient calibration. Let me synthesize the final review.

## Summary

The paper proposes **VQ-Transplant**, a two-stage framework for efficiently integrating new Vector Quantization (VQ) modules into frozen, pre-trained visual tokenizers: (1) substitute the VQ module while freezing the encoder-decoder, and (2) perform lightweight decoder adaptation (5 epochs on ImageNet-1k) to resolve distributional mismatch. The paper also introduces **MMD-VQ**, which uses Maximum Mean Discrepancy for nonparametric distributional alignment between feature and codebook vectors. Evaluated primarily on the VAR tokenizer, VQ-Transplant with MMD-VAR reportedly achieves 0.81 r-FID on ImageNet-1k (vs. 0.92 for original VAR) at ~21.8× lower training cost.

## Strengths

- **Clear and practically motivated problem**: Training state-of-the-art VQ tokenizers with adversarial training is genuinely expensive. The idea of decoupling VQ development from full end-to-end retraining directly addresses a real barrier to VQ research.

- **Comprehensive empirical comparison of VQ methods under the framework**: The systematic evaluation of five VQ algorithms (Vanilla, EMA, Online, Wasserstein, MMD) across multi-scale and fixed-scale configurations, multiple codebook sizes (4096–65536), and four datasets (ImageNet-1k, FFHQ, CelebA-HQ, LSUN-Churches) provides a valuable empirical baseline for the community.

- **Strong reconstruction results under the framework**: MMD-VAR with K=8192 achieves 0.81 r-FID on ImageNet-1k, improving over original VAR's 0.92, and cross-dataset results on FFHQ (r-FID 1.21) surpass listed baselines. The systematic demonstration that decoder adaptation is necessary (lower quantization error after Stage I does not guarantee better reconstruction) is a genuinely insightful empirical finding.

- **Cross-dataset generalization**: Demonstrating that a VAR backbone trained on OpenImages, adapted with a new VQ module and 5-epoch decoder tuning, transfers well to FFHQ, CelebA-HQ, and LSUN-Churches is a meaningful result.

## Weaknesses

### Major:

- **The 95%/21.8× training cost reduction claim is not apples-to-apples and is misleading as presented.** Table 1 compares VAR trained on OpenImages (16×A100, 60h) against VQ-Transplant on ImageNet-1k (2×A100, 22h). These are different datasets, and VQ-Transplant presupposes access to a pretrained VAR model whose full training cost is excluded from the accounting. The "from-scratch" baselines in Table 6 train for only 5–7 epochs (which the authors acknowledge is far from convergence—"typically require hundreds of epochs"), making them strawmen rather than fair comparisons. A proper efficiency claim requires either (a) the total cost of pretraining the backbone plus adaptation, or (b) a fully converged from-scratch MMD-VAR baseline on the same dataset. As stated, "reducing the training cost by 95%" conflates conditional savings (given an existing pretrained model) with unconditional savings, which overstates the practical impact.

- **No evaluation of downstream generation quality.** The paper evaluates only reconstruction metrics (r-FID, r-IS, PSNR, SSIM, LPIPS) but never assesses whether transplanted tokenizers produce effective discrete representations for downstream generative modeling (e.g., autoregressive image generation). Visual tokenizers are ultimately used *as components* in generative pipelines, and reconstruction quality alone does not guarantee generation quality. This is a critical gap for a framework that aims to "democratize quantization research" for generative models.

- **The framework's generality is undersupported — experiments are overwhelmingly VAR-centric.** The only non-VAR experiment (LDM-16 in Appendix D) is acknowledged to show "lower adaptability compared to VAR-based models," but this limitation is not discussed in the main text. The headline claims of a "general" "plug-and-play" framework are based on one specific, adversarially-trained, multi-scale tokenizer. Whether VQ-Transplant works for other architectures (VQGAN, RQVAE, LlamaGEN) remains unclear. The paper would be substantially stronger with at least one additional tokenizer demonstrating comparable success.

- **MMD-VQ's contribution over Wasserstein-VQ is marginal, and its theoretical advantage is not empirically validated.** Across experiments, MMD-VQ and Wasserstein-VQ are essentially tied: on multi-scale adaptation K=8192, r-FID is 0.81 vs. 0.83; on FFHQ adaptation K=32768, Wasserstein-VQ achieves 1.21 r-FID vs. MMD-VQ's 1.37. The theoretical motivation claims MMD-VQ avoids Gaussian assumptions, but no experiment demonstrates a scenario where Wasserstein-VQ fails due to non-Gaussianity while MMD-VQ succeeds. Without error bars, differences of 0.02 r-FID cannot be considered meaningful. The claim that MMD-VQ demonstrates "superior reconstruction fidelity compared to the vanilla VAR approach" (Abstract, contribution 2) is true only because the entire VAR decoder stack is reused and adapted — not because MMD-VQ itself is substantially better than Wasserstein-VQ.

### Minor:

- **The two-stage framework is conceptually straightforward.** Stage I (train VQ module on frozen features) and Stage II (fine-tune decoder with adversarial+perceptual losses) are natural choices for anyone wanting to swap a quantizer into a pretrained autoencoder. Appendix C even shows that joint optimization (encoder+decoder+VQ together) often performs *better*, which somewhat undermines the centrality of the staged design. The paper does not ablate simpler alternatives (e.g., jointly fine-tuning VQ+decoder from the start, or L2/LPIPS-only adaptation without adversarial training).

- **Decoder adaptation still requires adversarial training.** The paper motivates the work partly by the instability of adversarial training, yet Stage II uses the full adversarial stack (DINO-S discriminator, DiffAug, LeCAM, consistency regularization). While shorter (5 epochs), the framework does not fully escape the instability it critiques, and no analysis of sensitivity to this adversarial setup is provided.

- **The 5-epoch adaptation choice is not justified, and Table 5 shows continued improvement to 20 epochs.** Table 5 shows r-FID dropping from 0.81→0.74 (K=8192) by epoch 20, suggesting the "5 epochs" framing understates what is possible. If 5 epochs already beats the original VAR (0.92), the additional improvement is notable, but the trade-off between adaptation cost and performance is not analyzed.

### Trivial:

- The multi-Gaussian kernel bandwidth parameters σ_i for MMD are not discussed in the main text; their sensitivity and selection are important practical details that are absent.

## Nice-to-Haves

- Train or fine-tune a downstream autoregressive generation model on the transplanted tokenizer's tokens and report generation FID/IS. This would validate the framework's utility for its intended use case.
- Demonstrate VQ-Transplant on at least one additional tokenizer architecture (e.g., VQGAN or LlamaGEN) to support the "plug-and-play" generality claim.
- Provide a fully converged from-scratch baseline (or at least a cost estimate thereof) so that the efficiency claim can be properly contextualized.
- Include error bars across multiple runs for key metrics, especially when r-FID differences are as small as 0.02.
- Ablate the decoder adaptation loss (L2 only, L2+LPIPS, full adversarial) to show which components are essential.

## Removed Points

These points are flagged to be removed; treat them with caution.

- **Missing VQ baselines (FSQ, SimVQ, rotation trick)**: Suggested by the human finder, but the paper focuses on VQ module substitution within existing multi-scale tokenizer architectures (primarily VAR). Methods like FSQ, SimVQ, and the rotation trick are different quantization paradigms that may not be directly compatible with the multi-scale VAR backbone; their absence does not constitute an fairness gap in the comparison as presented. Moreover, per the hard rules, we should not flag missing related works.

- **Strawman "from-scratch baselines are unfairly undertrained"**: While this is partially valid (the from-scratch baselines in Table 6 are deliberately short), the paper itself acknowledges tokenizers "typically require hundreds of epochs." This weakness is already captured above as a criticism of the efficiency claim, which conflates conditional vs. unconditional savings. Removing the misleading claim would resolve this; the from-scratch comparison as a demonstration of the *difficulty* of training from scratch is actually informative.

- **Reproducibility concerns about discriminator setup**: The paper uses the same DINO-S discriminator and training recipes as VAR (citing Tian et al., 2024). Per the rules, this is not a reproducibility weakness.

- **Formatting/style nitpicks**: The typo "repid" in the contributions and equation numbering artifacts are trivial.

## Novel Insights

The most interesting empirical finding is the **decoupling between quantization error and reconstruction quality**: after Stage I, MMD-VAR achieves lower quantization error (0.255) than the original VAR (0.283) yet worse r-FID (1.52 vs. 0.92). This demonstrates that simply having a better codebook is insufficient when the decoder's priors are misaligned with the new quantization space — the decoder adaptation stage is not merely cosmetic but structurally necessary. This insight about encoder–decoder–quantizer couplings in pretrained tokenizers deserves more discussion than it receives.

## Suggestions

1. **Reframe the efficiency claim honestly**: Report total cost as "cost of pretraining original VAR + cost of VQ-Transplant adaptation." Condition the 21.8× claim explicitly on reusing an existing pretrained backbone. Avoid the ambiguous "95% reduction" phrasing.

2. **Add at least one downstream generation experiment**: Even a simple autoregressive model trained on the transplanted tokenizer's outputs with generation FID would dramatically strengthen the paper.

3. **Add at least one more tokenizer backbone**: Show VQ-Transplant works on VQGAN or another architecture to support the "plug-and-play framework" framing.

4. **Be precise about MMD-VQ's contribution**: Describe it as a competitive alternative to Wasserstein-VQ with theoretical robustness advantages, not as a clearly superior method. The current "superior reconstruction fidelity" language is not supported by the data.

## Evaluation

**Originality**: Moderate. The VQ-Transplant framework is straightforward (substitute VQ module → fine-tune decoder with adversarial losses). MMD-VQ applies a well-known nonparametric divergence to the distributional matching framework of concurrent Wasserstein-VQ work. Both contributions are incremental in isolation, though their combination for practical VQ research is useful.

**Importance of research question**: High. Decoupling VQ algorithm development from expensive adversarial tokenizer training is genuinely important for resource-constrained researchers.

**Claims support**: Partially. The reconstruction quality results are strong, but the efficiency claims are overstated, the generality claims are under-supported, and MMD-VQ's superiority over Wasserstein-VQ is marginal at best.

**Experimental soundness**: Good for reconstruction evaluation; missing downstream generation evaluation is a significant gap. Cross-dataset generalization is a genuine strength.

**Clarity**: Generally well-written, though key limitations are relegated to appendices.

**Community value**: High potential utility if the framework generalizes, but the current evidence base is narrow.

**Calibration**: FSQ (accepted, 6,6,8,6) proposed a novel quantization method with generation results. Wasserstein-VQ (rejected, 5,3,5,6) was similarly incremental and had problematic assumptions. TaCA (rejected, 6,5,5,5) was a plug-and-play adaptation framework with limited generality. VQ-Transplant sits in a similar tier to TaCA — useful engineering practice with experimental value, but with overstated claims and limited novelty. The absence of downstream generation experiments is a particularly notable gap relative to accepted tokenizer papers like FSQ and BSQ, which all demonstrate generation quality.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>