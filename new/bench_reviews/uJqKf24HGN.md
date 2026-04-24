## Summary

This paper proposes UniCon, a control adapter for diffusion models that replaces the standard bidirectional feature-injection paradigm (e.g., ControlNet) with a unidirectional flow: a frozen base model extracts intermediate features, and a trainable adapter directly outputs the denoised result. By cutting off gradient backpropagation through the base model, UniCon roughly halves training VRAM and achieves a ~2.3× training speedup for matched adapter sizes, while improving controllability and generation quality across U-Net and DiT backbones on tasks ranging from Canny-edge guidance to super-resolution.

## Strengths

- **Novel and well-motivated architectural paradigm.** The unidirectional design is a clean departure from existing bidirectional adapters, and it naturally addresses the scaling challenges of transformer-based diffusion models where encoder/decoder separation is ambiguous (Section 3).  
- **Strong training-efficiency evidence on DiT.** Figure 6 shows that, for matched full-network adapter sizes, UniCon eliminates the diffusion model’s backward-pass memory and time overhead, cutting peak training VRAM by nearly half and speeding up training by ~2.3× relative to a full-network ControlNet on DiT (Section 4.2).  
- **Controlled DiT ablation isolating the design choice.** Table 1c holds the adapter architecture constant (Decoder-copy or Full-copy) and varies only whether the adapter outputs directly (unidirectional) or injects residuals (bidirectional). On DiT SR, the unidirectional Full adapter outperforms its bidirectional counterpart (PSNR 37.34 vs. 36.53; FID 20.34 vs. 23.04), suggesting the gains are not merely a capacity effect (Section 4.2).  
- **Consistent cross-task and cross-architecture improvements.** The method is tested on five diverse tasks—Canny, depth, pose, 4× SR, and deblur-downsampling—and the proposed ZeroFT connector is shown to outperform ZeroMLP and ShareAttention variants (Table 1b).  
- **Empirical validation that preserving the full frozen base model matters.** Figure 4 shows that discarding the base model and fine-tuning only the adapter causes a sharp drop in generation quality, supporting the design choice of using the frozen network as a feature extractor (Section 4.2).

## Weaknesses

### Fatal
None.

### Major
- **SD U-Net comparisons in Table 2 confound architectural direction with adapter capacity.** On SD U-Net, standard ControlNet copies only the encoder (~half the U-Net parameters). Table 2 compares this against full UniCon (full U-Net copy) for Canny and Depth without a same-parameter UniCon-Half baseline. Although UniCon-Half is reported for SR (PSNR 34.38 vs. ControlNet 31.66) and outperforms ControlNet, its absence for the other SD tasks means the reported gaps cannot be uniquely attributed to the unidirectional mechanism. This is particularly problematic because the paper’s own Table 1a shows that, on DiT, a Full ControlNet (36.53 PSNR) performs substantially better than the Encoder variant (34.82 PSNR) used as the Table 2 baseline, yet Table 2 still uses the weaker Encoder baseline for DiT as well, exaggerating the margin of victory.  
- **Inference cost is entirely omitted.** Because UniCon requires a full forward pass of the frozen base model plus a full forward pass of the adapter, inference compute and activation memory are higher than standard ControlNet (base + encoder copy). For a method positioned as a solution for “next generation” large-scale models, doubling inference cost is a severe practical drawback, and the paper never reports per-step latency, sampling FLOPs, or peak inference VRAM (Sections 4–5).  
- **No validation on the large-scale models that motivate the work.** The introduction emphasizes scaling to 8B-parameter transformers such as SD3, but the experiments are limited to Stable Diffusion 2.1 (0.86B U-Net) and PixelArt-α DiT. The SUPIR integration mentions SD3 but offers only qualitative results with no metrics (Section 4.3, Figure 8).

### Minor
- **Mischaracterization of standard adapter training.** The introduction states that existing methods “calculate and store the gradients of the diffusion model” (Section 1). In standard ControlNet/T2I-Adapter training, the base model is frozen and its parameter gradients are not stored; the memory overhead arises from retaining activations for backpropagation through the frozen network. While the practical savings of UniCon are real, this description is technically imprecise.  
- **U-Net architectural details are under-specified.** Figure 2(b) depicts a U-Net adapter with encoder, mid, and decoder blocks, but the text never clarifies whether the adapter mirrors U-Net skip connections, how multi-resolution features are spatially aligned, or whether the adapter consumes the noisy latent $z_t$ directly (Section 3).

### Trivial
None.

## Nice-to-Haves
- Quantitative metrics for the SUPIR-UniCon integration.  
- Analysis of which intermediate layers contribute most to adapter performance (e.g., deep vs. shallow features).  
- Compatibility study with parameter-efficient tuning (e.g., LoRA) inside the UniCon adapter.

## Removed Points
These points are flagged to be removed; treat them with caution.

- **“Table 1c is conceptually garbled / mixes metrics.”** The parsed submission text garbles Table 1c (SSIM values appearing under SR/PSNR rows), but this is a PDF-parsing artifact, not an author error. The actual table in the submission is coherent.  
- **“ControlNet-Full does not exist in the literature.”** For U-Net, this is true but beside the point the authors are making on DiT; for DiT there is no standard ControlNet because transformers lack a clear encoder/decoder split, so full-network and skip-layer variants are reasonable ablation baselines. The criticism that the standard SD U-Net baseline is omitted from Figure 6 is retained above as a Major weakness.  
- **“The ablation confounds gradient isolation with replacing the diffusion decoder.”** The paper compares two design paradigms: bidirectional residual injection vs. unidirectional direct output. Holding adapter capacity constant (Table 1c), this is a fair comparison of the paradigms as defined by the authors. Requesting a baseline where the adapter replaces the decoder yet gradients flow bidirectionally through a frozen base model is a pedantic separation of factors that are coupled by the very design being proposed.

## Novel Insights
None beyond the paper's own contributions.

## Suggestions
- For the next revision, run UniCon-Half on *all* SD U-Net tasks and report results alongside ControlNet-Encoder to disentangle capacity from architectural direction.  
- Include inference latency and FLOPs for both SD and DiT backbones, comparing against standard ControlNet at identical batch sizes and resolutions.  
- Correct the description of memory overhead to accurately reflect that standard frozen-base training stores activations for backward traversal, not parameter gradients.

## Score and Decision

**Calibration anchors used:**
- *3DTrajMaster* (avg 6.75): Strong novel task, extensive results, minor dataset diversity limitations — used as a high anchor. UniCon is below this because its core comparison table has baseline-fairness issues and it omits inference analysis entirely.  
- *CTRL* (avg 6.50): Interesting RL-based idea, limited baselines, missing recent comparisons — used as a medium-high anchor. UniCon has broader experiments but shares the “limited fair baselines” weakness; its efficiency claims are better quantified, but its SD U-Net quality comparisons are less controlled.  
- *3D-Adapter* (avg 5.60, Reject): Missing comparisons, confusing notation, scope too broad, inference overhead under-discussed — used as a direct peer anchor. UniCon is conceptually cleaner and more focused, with better internal ablations, but similarly omits key baselines and inference costs.  
- *MoveAnything* (avg 4.50, Withdrawn): Original idea but limited to a single synthetic dataset, weak quantitative support — used as a low-medium anchor. UniCon has far broader task coverage and stronger metrics.  
- *DynamicsDiffusion* (avg 3.00, Reject): Fundamental scalability and novelty concerns — used as a low anchor. UniCon is clearly above this; its core idea is novel and its DiT results are sound.

**Reasoning:** The paper’s central training-efficiency claim is well supported on DiT, and the unidirectional ablation on DiT (Table 1c) is reasonably controlled. However, the main comparison table (Table 2) systematically uses a weaker, smaller ControlNet baseline (Encoder-only) against a larger UniCon variant on both SD U-Net and DiT, and the paper completely ignores inference cost despite targeting deployment on large models. These are not fatal flaws—the core idea and DiT evidence are real—but they are serious enough that the paper should not be accepted without revision. The quality sits between the 3D-Adapter (5.6, Reject) and MoveAnything (4.5) anchors: better experimental breadth than the latter, but with baseline-fairness problems comparable to the former.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>