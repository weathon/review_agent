---
job_id: bef0c810-e250-4771-acca-7acf40e04a2a
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: Iq1fNZus2W.pdf
paper: Patch-Wise and Keyword-Aware: Efficient Multi-Condition Control of Diffusion Transformers via Position-Aligned and Keyword-Scope Attention
main_score_norm: 0.4
desk_reject: false
---
# Desk Rejection Assessment:

## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper is about efficient attention mechanisms and conditioning schemes for Diffusion Transformers, clearly within generative models and representation learning, which are core ICLR topics.

## Minimum Quality
Pass ✅.  
All required sections (Abstract, Introduction, Related Work, Method, Experiments, Results, Conclusion) are present and written in English. The method is reasonably well specified, experiments are nontrivial with baselines and metrics, and there are no obvious fatal methodological or theoretical errors that would justify immediate desk rejection.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I do not see any instructions targeting automated reviewers or hidden prompt-like content in the manuscript.

---

# Expected Review Outcome:

## Summary

The paper addresses the computational and memory overhead of multi-condition control in Diffusion Transformers, where current approaches concatenate all condition and image tokens and apply full attention. The authors empirically analyze attention maps for spatial-aligned (e.g., canny/depth) and subject-driven conditions and argue that most cross-condition attention is redundant. They propose Patch-wise and Keyword-Aware Attention (PKA), consisting of Position-Aligned Attention (PAA) for spatial conditions and Keyword-Scoped Attention (KSA) for subject conditions, combined with KV caching for condition tokens and an early-timestep sampling strategy for training. Experiments on a subset of Subject200K with multi-conditional tasks (Subject-Canny, Subject-Depth, Canny-Depth) show large speed and VRAM savings compared to UniCombine and OminiControl2, while largely maintaining or improving generation quality.

## Strengths

1. **Clear motivation from attention visualization and structural priors**  
   The paper starts from empirical inspection of attention maps in existing DiT-based multi-condition models. **Figure 2** (Page 2) clearly shows that attention between noisy image tokens and spatial condition tokens is strongly concentrated near the diagonal, supporting the claim that long-range cross-condition interactions are rarely used for spatial-aligned signals. **Figure 3** similarly visualizes that only keyword-relevant regions activate for subject-driven conditions. These figures ground the core design choices of PAA and KSA in observed model behavior, rather than purely ad-hoc sparsification.

2. **Simple, architecture-conscious decomposition of attention with caching**  
   The PKA design in **Figure 4(b)** (Page 5) cleanly separates self-attention within each condition, full attention between noisy image and text tokens, and specialized PAA/KSA connections to spatial/subject conditions. This inherently enables the “Condition Cache” in **Figure 4(a)**, where K/V for conditions are computed once at the first denoising step and reused. This is a practical and implementation-friendly idea that fits well with DiT-style multi-step inference and directly addresses a real bottleneck in multi-condition DiTs.

3. **Computational complexity reduction is principled and easy to analyze**  
   The PAA formula in **Equation (2)** (Page 4) shows a strict one-to-one alignment between the query from image token \(X_i\) and the K/V of spatial token \(SP_i\). This reduces complexity from \(\mathcal{O}(N^2)\) to \(\mathcal{O}(N)\) for the spatial branch while still using dot-product attention at each position. The KSA formulation in **Equations (3–4)** (Page 5–6) gives a clear mechanism: compute an image–keyword relevance map, threshold it to a mask, and then restrict later image–subject attention to masked regions. The math is simple but explicit.

4. **Strong efficiency gains demonstrated with scaling in number of conditions**  
   **Figure 7** (Page 7) shows inference time as a function of the number of conditions. The curve for the proposed method grows much more slowly than UniCombine, and is consistently faster than OminiControl2, yielding claimed \(3.9\times\)–\(10\times\) speedups. **Figure 8** shows VRAM consumption of the attention module dropping by factors of \(2.46\times\)–\(5.12\times\). These plots directly support the central efficiency claims and are exactly the kind of scaling experiment that matters for multi-condition control.

5. **Reasonable quality/controllability metrics relative to strong baselines**  
   **Table 1** (Page 8) provides a comprehensive comparison vs. OminiControl2 and UniCombine across generative quality (FID/SSIM), controllability (F1 for edges, MSE for depth), subject consistency (CLIP-I, DINOv2), and text fidelity (CLIP-T) for three multi-condition tasks. The proposed method dominates in FID and SSIM across all tasks and substantially improves CLIP-I / DINOv2, with competitive controllability metrics (especially on depth and dual Canny+Depth). This indicates that the sparsified attention does not degrade and can even enhance quality and consistency.

6. **Useful ablations that connect design choices to behavior**  
   - **Figure 9** isolates the PAA module vs. full attention and sliding-window attention, with latency and VRAM numbers, supporting the claim that a strict positional alignment is an efficient yet adequate alternative to windowed attention for spatial conditions.  
   - **Figure 10** studies the KSA threshold \(\epsilon\), qualitatively showing the tradeoff between pruning aggressiveness and fine subject details, with latency/VRAM indicated.  
   - **Figure 11** visualizes generations under different logit-normal shifts \(\mu\) and \(\delta\), supporting the idea that early-timestep-focused sampling yields better adherence to visual conditions.

7. **Early-timestep sampling is empirically motivated and could generalize**  
   The perturbation analysis summarized in **Figure 5** (Page 6) and illustrated more in **Figure 12** (Appendix, Page 13) shows that removing conditions early (“high-to-low”) quickly harms SSIM/structure, while removing them late has milder effects. This is a sensible and important observation for flow-matching DiTs. The proposed modified logit-normal sampling \(t \sim \text{Logit-}\mathcal{N}(\mu,\delta)\) for \(\mu>0\) is straightforward to implement and may be beneficial beyond this specific setting.

## Weaknesses

1. **Methodological novelty is moderate; connection to broader sparse/structured attention is underdeveloped**  
   PAA essentially enforces a fixed one-to-one mapping between image and spatial tokens, and KSA is a thresholded attention mask guided by text keywords. While these are reasonable specializations, the paper largely presents them as if they were the only or primary way to exploit the observed sparsity. There is minimal engagement with the broader literature on structured or sparse attention for images (e.g., local windows, block-sparse patterns, deformable attention) beyond a comparison to sliding-window attention for PAA in Figure 9. This makes the conceptual contribution feel more like a targeted engineering simplification than a more general framework for multi-condition attention. The paper could be clearer on where PKA stands relative to other structured-attention techniques adapted to multi-condition DiTs.

2. **Some key components are under-specified or glossed over mathematically**  
   - **Keyword mask construction (Equation 3)**: \(\mathrm{Norm}(\cdot)\) is left vague. Is it softmax along the spatial dimension, layer norm, or per-token normalization? This choice has a material impact on which regions exceed the threshold \(\epsilon\) and on the gradient distribution. The paper only says “Norm”, and it is not clear whether training backpropagates through this normalization and the threshold, or whether the mask is treated as a non-differentiable stop-grad operator.  
   - The binary mask \(M^t = \mathrm{Norm}(\cdot) \ge \epsilon\) makes the attention region a hard, discontinuous function of the activations. The authors do not discuss how this interacts with optimization or whether they use straight-through estimation, detach, or rely on the fact that the mask is used only across timesteps. This is a nontrivial design choice and matters for both training stability and reproducibility.  
   - The text says the keyword set \(\mathbb{K}\) “typically contains just 1 to 2 tokens”, but there is no detailed description of how these tokens are chosen from the caption in practice; e.g., through parsing noun phrases, manual annotation in Subject200K, or simply using CLIP’s first token.

3. **Limited benchmark scope and dataset scale**  
   Experiments are restricted to a curated subset of Subject200K with captions that contain a descriptive keyword, trained with LoRA on FLUX.1 for only 20K iterations (Page 6). This is a narrow experimental setting:  
   - There are no results on standard large-scale benchmarks or public conditioning datasets (e.g., COCO with layout, ADE20K/LAION multi-condition variants) that would help evaluate generality.  
   - Multi-conditional combinations are limited to Canny and/or depth plus a single subject, and always with the same base DiT model. We do not see how PKA behaves on more diverse or noisy conditions (e.g., segmentation maps, pose, scribbles) or with different DiT backbones.  
   - All quality metrics in **Table 1** are computed against ground-truth images from this subset, which may not be representative of broader generative use cases. This limits the external validity of the empirical claims.

4. **Baselines for efficiency and control are relatively narrow**  
   The main baselines are UniCombine and OminiControl2, which are indeed very relevant, but this excludes several other proposed efficiency or control mechanisms for DiTs. For example, the paper compares PAA only with sliding-window attention (Pan et al., 2023) in Figure 9, but not with more modern efficient DiT conditioning or attention-specialization approaches that also reduce computation. There is no quantitative comparison with methods that exploit KV caching or token pruning tailored to multi-condition generation beyond the two chosen baselines. As a result, the “state-of-the-art efficiency” claim in the abstract feels slightly overstated.

5. **Analysis of quality–efficiency tradeoffs is mostly qualitative and lacks quantitative ablations**  
   - For KSA, Figure 10 qualitatively suggests that raising \(\epsilon\) to 0.4 preserves subject fidelity while reducing latency/VRAM, but there are no numerical metrics (FID, CLIP-I, SSIM, etc.) plotted as a function of \(\epsilon\). Without a more quantitative view, it is hard to assess how robust KSA is to threshold choice and how steep the tradeoff curve is.  
   - Similarly, for PAA, Figure 9 gives latency and VRAM but does not report quantitative quality/controllability metrics for PAA vs full attention vs SWA. The text states “both methods produce high-fidelity images that adhere to spatial conditions”, but this is evaluated visually; a simple F1/MSE summary would help substantiate that PAA is not giving up controllability.  
   - For early-timestep sampling, the main paper provides qualitative examples in **Figure 11** and SSIM curves only in Appendix **Figure 13**, but no aggregate quantitative table comparing final FID/CLIP-I/etc. under different (\(\mu,\delta\)) choices. That makes it unclear how much of the final performance gains in Table 1 actually stem from this sampling scheme versus PKA itself.

6. **Early-timestep sampling design is heuristic and under-explained**  
   The paper proposes sampling timesteps from \(\text{Logit-}\mathcal{N}(\mu,\delta)\) with \(\mu>0,\delta>1\) (Page 6), but the main text does not specify the exact values used in the reported experiments; these only appear in figures (e.g., \(\mu=0.5, \delta=1.5\) in **Figure 11** and **Figure 13**). There is no motivation for how these parameters were chosen beyond empirical trial, nor a sensitivity analysis.  
   Furthermore, concentrating training on high-\(t\) steps may degrade denoising accuracy at low noise levels, which could affect final image sharpness or text alignment, but this risk is not discussed or measured explicitly. Without a more rigorous analysis, this component feels like a plausible but somewhat ad-hoc tweak.

7. **Some experimental design choices limit interpretability of results in Table 1**  
   - It is not clear whether all methods (OminiControl2, UniCombine, Ours) are fine-tuned under exactly the same LoRA configuration, training iterations, and timestep sampling; the text suggests that the proposed early-timestep sampling is used only for the authors’ method. If baselines are trained under the standard sampling while the proposed method is trained under the skewed sampling, some of the gains in **Table 1** might be attributable to better training rather than the attention structure itself.  
   - Metrics like CLIP-I, CLIP-T, and DINOv2 can be sensitive to batch size, augmentation choices, and prompt truncation. The paper does not clearly state whether all methods are evaluated with identical generation hyperparameters (e.g., number of diffusion steps, guidance scale) and random seeds. Given the fairly small numeric margins on text fidelity (CLIP-T differs by ~0.002–0.004 between methods), small differences in setup could matter.

8. **Limited discussion of failure modes and qualitative diversity**  
   The qualitative figures (**Figures 1, 6, 14–17**) mostly show successful cases where PKA matches all conditions nicely. There is no systematic analysis of typical failure modes, such as when the keyword mask incorrectly localizes the subject, or when spatial conditions conflict with subject placement. For example, if the keyword-scoped mask omits some regions due to ambiguous phrasing, does the subject vanish, or does the model hallucinate? Understanding such behavior is important for a control mechanism that intentionally prunes attention.

9. **Missing and under-discussed related work directly on attention control in DiTs and multi-conditional generation**  
   Beyond the works already cited (OminiControl, OminiControl2, UniCombine, PixelPonder), several directly-relevant recent papers focusing on attention specialization, multi-instance synthesis, and multi-conditional control are not referenced or compared, which weakens the positioning of the contribution; details are in the next section.

10. **Minor clarity and notation issues**  
   - The description of the KV cache in **Figure 4(a)** is somewhat terse: it states that K/V for “all condition tokens” are cached after the first step, but given that KSA uses time-dependent keyword keys \(K_i^t\) and subject keys \(K_{SJ}^{t+1}\), it is not entirely clear what exactly is cached for which timesteps. Are subject K/V also frozen over time, or only spatial/text conditions?  
   - In Equation (4), the notation \(K_{SJ}^{t+1}{}^{\top}\) and \(V_{SJ}^{t+1}\) suggests that subject K/V are recomputed at each step despite the earlier KV cache. The relationship between KSA and caching should be clarified to avoid confusion.

## Potentially Missing Related Work

1. **Li & Ye, “Dual-Channel Attention Guidance for Training-Free Image Editing Control in Diffusion Transformers”, 2026**  
   This work manipulates key and value channels in DiTs to guide image editing control and is highly relevant to the idea of modifying attention structure for controllability and efficiency. It should be discussed in Section 2.2 and compared conceptually as another approach to structured attention for control, even though it targets training-free editing rather than fine-tuned multi-conditional generation.

2. **Zhang, Sun & Zhang, “Hierarchical and Step-Layer-Wise Tuning of Attention Specialty for Multi-Instance Synthesis in Diffusion Transformers”, 2025**  
   This paper focuses on tuning attention specialization across layers and steps to handle complex multi-instance synthesis in DiTs, which is closely related to the multi-condition control problem addressed here. It should be cited in Section 2.1 or 2.2 and in the discussion around PKA, as it also uses attention structure to improve controllability and efficiency in complex scenes.

3. **Chen, Ma & Jia, “Context-Aware Autoregressive Models for Multi-Conditional Image Generation”, 2025**  
   Although based on autoregressive models rather than diffusion, this work also deals explicitly with multi-conditional image generation by embedding multiple conditions in the token sequence. It is relevant for situating this paper within the broader landscape of multi-conditional generators and should be mentioned in Section 2.1, including a short discussion that contrasts DiT-based multi-condition control with autoregressive alternatives.

## Questions

1. **Details of mask construction and differentiability in KSA**  
   - What exactly is the normalization operator \(\mathrm{Norm}\) in Equation (3)? Is it a softmax over spatial positions, an L2 normalization, or something else?  
   - Do you backpropagate through the thresholding \(M^t = (\mathrm{Norm}(\cdot) \ge \epsilon)\), and if so, how? If gradients are blocked at this operation, can you clarify how the upstream supervision signals affect the earlier layers that produce the relevance scores?

2. **Selection of keyword tokens \(\mathbb{K}\)**  
   How are the keyword tokens chosen from the text prompt in practice? Is there a deterministic rule (e.g., last noun, manually annotated keyword from Subject200K) or any heuristic based on CLIP/attention saliency? It would be helpful if you could specify this precisely and discuss how robust KSA is when the chosen tokens are ambiguous or not clearly localizable.

3. **Interplay between PKA and early-timestep sampling in the reported gains**  
   In Table 1 and Figures 7–8, are the baselines (OminiControl2 and UniCombine) fine-tuned using the same early-timestep sampling strategy, or do they use the standard \(\text{Logit-}\mathcal{N}(0,1)\) schedule? Could you provide an ablation that isolates: (a) PKA with standard sampling, and (b) full attention with early-timestep sampling, to clarify how much each component contributes to the quality and efficiency improvements?

4. **Scope of KV caching relative to subject tokens**  
   In Figure 4(a), you state that K/V of “all condition tokens” are cached after the first step. However, Equation (4) uses \(K_{SJ}^{t+1}\) and \(V_{SJ}^{t+1}\). Are subject K/V recomputed at each step (which would reduce the cache benefit) or are they actually constant and shared across timesteps? Please clarify the exact caching protocol and, if possible, quantify how much of the overall speedup comes from caching vs. reduced attention sizes.

5. **Failure cases of KSA and impact of mis-localization**  
   Have you observed cases where the keyword-scoped mask \(M^t\) fails to cover the full extent of the subject, or focuses on background regions due to noisy/ambiguous attention? If so, what typical visual artifacts arise, and do you have any mitigation strategies (e.g., dilation of the mask, multi-keyword aggregation, or fallback to full attention)?

6. **Quantitative ablations on thresholds and sampling parameters**  
   Can you provide additional quantitative results (e.g., on the validation set) showing FID/CLIP-I/controllability metrics as a function of \(\epsilon\) and (\(\mu,\delta\))? This would make the trade-off surfaces in Figures 10–11 and 13 more precise and increase confidence that your chosen hyperparameters are not overly tuned to the specific experiments.

## Flag For Ethics Review

No ethics review needed.

## Details Of Ethics Concerns

N/A.

## Soundness Rating

2: fair.  
The core ideas (PAA, KSA, and early-timestep sampling) are plausible and mostly sound, and there is supporting empirical evidence. However, some important implementation details (mask normalization and differentiability, exact keyword selection, interaction of caching with KSA) are under-specified, and the attribution of gains across components is not fully disentangled.

## Presentation Rating

2: fair.  
The paper is generally readable and the figures (especially Figures 2–4, 6–11) are helpful, but several crucial methodological details are missing or glossed over, and the discussion of limitations and related structured-attention work is shallow. Some notation around KSA and the KV cache is confusing.

## Contribution Rating

2: fair.  
The work tackles a relevant and practical problem and demonstrates substantial efficiency gains, which is valuable. However, the conceptual novelty is moderate, the experimental scope is limited to a single curated dataset and backbone, and positioning relative to closely related attention-control methods in DiTs is incomplete.

## Overall Rating

4: Marginally below the acceptance threshold. But would not mind if paper is accepted.  

The paper presents a well-motivated and practically useful simplification of multi-condition attention in DiTs, with clear efficiency improvements and largely preserved or improved quality on the tested tasks. At the same time, the work feels somewhat narrow and engineering-focused, with under-specified components, limited benchmarks, and incomplete comparison to related attention-control methods. With stronger experimental breadth, clearer mathematical specification of KSA and sampling, and better positioning within the structured-attention literature, this could be a solid ICLR contribution; in its current form, it falls slightly short of that bar.

## Reviewer Confidence

4: confident.  
I am familiar with diffusion transformers, multi-condition control, and efficient attention mechanisms, and I carefully examined the equations, figures, and experimental setup. Some implementation details are missing, so there is room for clarification in rebuttal, but my overall assessment is unlikely to change dramatically.