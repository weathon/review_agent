---
job_id: 419ca5e7-af92-4dcf-909c-58c04fe38e6a
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: ppXAVexrAM.pdf
paper: ARSS: Taming Decoder-only Autoregressive Visual Generation for View Synthesis From Single View
main_score_norm: 0.6
desk_reject: false
---
# Desk Rejection Assessment:
## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper is clearly about generative modeling and representation learning for vision (autoregressive token-based models for novel view synthesis), which is squarely within ICLR scope.

## Minimum Quality
Pass ✅.  
All required sections (Abstract, Introduction, Related Work, Method, Experiments, Results, Discussion) are present. The work is reasonably novel and technically nontrivial, with quantitative and qualitative experiments. There are no obvious fatal methodological flaws, although there are some clarity and positioning issues discussed below.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I did not find any hidden prompts, manipulative instructions to LLM reviewers, or other integrity violations in the paper text.

---

# Expected Review Outcome:

## Summary

The paper proposes ARSS, a framework for single-image novel view synthesis using a decoder-only autoregressive transformer instead of diffusion models. Multi-view sequences are tokenized using a video tokenizer (VidTok), while camera trajectories are encoded into per-position “camera tokens” via a Plücker-ray-based camera autoencoder, and both token streams are interleaved and fed to a GPT-style model trained with next-token prediction under a spatially shuffled but temporally ordered token permutation. Experiments on RealEstate10K, ACID, and zero-shot DL3DV indicate that ARSS is competitive with, and in many metrics better than, several diffusion-based and transformer baselines, with ablations on tokenization and token-ordering strategies.

## Strengths

1. **Clear problem focus and timely direction**  
   The paper targets an important and actively studied problem: controllable novel view synthesis from a single image along a specified camera trajectory, framed within the “world models” narrative. Moving this task from diffusion-based, jointly denoised latents to a strictly causal autoregressive setup is conceptually well motivated and relevant for sequential world modeling.

2. **Reasonable architectural design combining three components**  
   The overall system design in **Figure 2** (video tokenizer + camera autoencoder + decoder-only transformer) is coherent. Using a *video* tokenizer (VidTok) to produce temporally consistent visual tokens, combined with per-position camera tokens derived from Plücker raymaps via a dedicated camera autoencoder, is a sensible way to give an AR transformer both temporal context and explicit 3D guidance. The camera autoencoder loss in **Equation (5)**, which includes ray direction normalization and orthogonality constraints between direction and Plücker momentum, shows some care in preserving physical consistency of camera geometry.

3. **Token-order permutation strategy is well motivated and empirically supported**  
   The hybrid ordering scheme that shuffles spatial tokens within each frame while preserving temporal order is intuitively appealing given the bidirectional nature of spatial context and the causal nature of the transformer. The ablation in **Table 2** and the visual comparison in **Figure 7** provide concrete evidence: relative to “raster” ordering, the proposed strategy improves PSNR (16.29 → 19.22), SSIM (0.488 → 0.565), LPIPS (0.402 → 0.294), and FID (71.17 → 60.11), with corresponding qualitative improvements in later frames where raster shows noticeable distortions.

4. **Video tokenizer vs VQ image tokenizer comparison is meaningful**  
   The comparison in **Table 3** between a standard VQ image tokenizer and the adopted video tokenizer is compelling: the video tokenizer yields markedly better PSNR (19.22 vs 15.69), SSIM (0.565 vs 0.437), LPIPS (0.294 vs 0.498), and, most importantly, FVD (52.56 vs 137.68), strongly supporting the claim that video-level quantization is important for temporal consistency in autoregressive NVS. This is further visible in **Figure 3** and **Figure 10**, where sequences appear more stable across time.

5. **Competitive quantitative results with multiple baselines**  
   **Table 1** shows that ARSS generally performs competitively across RealEstate10K, ACID, and DL3DV. For example, on RealEstate10K, ARSS achieves best PSNR (19.02) and best LPIPS (0.269), and on ACID it achieves best PSNR (21.93) and LPIPS (0.265), outperforming both diffusion baselines (SEVA, MotionCtrl, ViewCrafter) and transformer-based LVSM on several metrics. Zero-shot results on DL3DV (PSNR 16.70, SSIM 0.449, LPIPS 0.347) also compare favorably to LVSM and MotionCtrl.

6. **Analysis of long-horizon behavior**  
   The error accumulation analysis in **Figure 6** is a nice touch. The per-frame PSNR/SSIM/LPIPS curves show that ARSS degrades more slowly with frame index compared to baselines, reinforcing the claim that a causal, sequence-aware model can better maintain quality over long camera sweeps.

7. **Diverse qualitative demonstrations**  
   Qualitative results in **Figure 3**, **Figure 4**, **Figure 5**, **Figure 9–11** convincingly show that the method produces visually sharp and geometrically plausible sequences across indoor, outdoor, aerial, and even stylized/AI-generated inputs. The camera-path visualizations in **Figure 1** and **Figure 5** help clarify that the model respects user-specified trajectories.

## Weaknesses

1. **Positioning vs very closely related autoregressive NVS work is incomplete**  
   The paper claims “to the best of our knowledge, ARSS is the first that applies the GPT-style causal autoregressive model in novel view generation with camera control” (end of Section 3.2.3, and again in the Discussion). However, several highly relevant recent works are omitted and potentially contradict this uniqueness claim:
   - Autoregressive *diffusion* models for NVS that explicitly aim for causal multi-view generation and flexible input-output configurations (e.g., CausNVS-like approaches).
   - Autoregressive next-scale models that can do single-image object view synthesis in a zero-shot manner.  
   While the details differ (diffusion vs discrete AR, object-centric vs full-scene, etc.), these works directly address autoregressive or causal novel view generation and should be discussed carefully in **Section 2** and contrasted in **Section 5**. As it stands, readers get an overstated impression of novelty.

2. **Mathematical notation and objective formulation are inconsistent or incomplete**  
   Several core equations have issues that undermine clarity:
   - **Equation (6)**, which is supposed to define the permuted sequence, uses `\pi_{11}^{P_1(1)}` throughout and seems to duplicate tokens (e.g., `\pi_{11}` repeated) and mix up notation between `x` and `π`. It is not actually clear whether `\mathcal{S}` contains both camera and visual tokens, and the form written is almost certainly not what is implemented. This matters because the token order drives the causal mask and training.
   - **Equation (7)** is syntactically malformed: `CE(f_{\theta}([\mathcal{S},[x_{21}^{P_{2}(1)},...,x_{ln}^{P_{l}(n)}]),` is missing closing brackets and does not clearly specify the target sequence. This is the central training objective for the AR transformer, so it should be unambiguous.
   - **Equation (8)** attempts to express the autoregressive factorization with permuted indices, but the notation `\pi_{\leq i,\leq j}^{P_{\leq i}(\leq j)}` and `x_{<i,<j}^{P_{<i}(<j)}` is very hard to parse and does not match the description just above. It is also unclear how this factorization relates to the actual causal mask in **Figure 2** (right).  
   - In **Equation (5)**, the variable naming is confusing: “where $\bm{d}$ is the normalized camera ray direction, $\bm{d}$ is the momentum term formulated as $\bm{m}=\bm{o}\times\bm{d}$” is clearly a typo, and $\bm{o}$ is never defined.  
   These are not mere cosmetic issues: the reader cannot reconstruct the exact training and sampling procedures from the equations, which is problematic for a method paper. The authors should rewrite **Equations (6)–(8)** with consistent notation (clearly distinguishing visual vs camera tokens), correct the loss in **Equation (5)**, and explicitly state input/target sequences and masks.

3. **Limited analysis of computational cost and scalability of AR vs diffusion**  
   One of the implicit motivations for AR is better causality and reusability along trajectories, but autoregressive token-by-token generation can be slow and memory-intensive, especially with long sequences. For the described setup, the video tokenizer maps a 17×256×256 sequence to 5×32×32 tokens (Page 7); with camera tokens interleaved, the sequence length is on the order of several thousand tokens. Yet the paper does not:
   - Provide any inference-time comparison (e.g., seconds per frame or per sequence) to diffusion baselines such as SEVA or ViewCrafter.
   - Analyze memory requirements or maximum trajectory length that can be handled on typical hardware.
   - Quantify the benefit of the claimed parallel decoding ability enabled by shuffling (Section 3.2.3, last paragraph), beyond an intuitive statement.  
   For a method that emphasizes causality and potential world-model applications, omitting a runtime/efficiency study significantly weakens the practical impact and makes it hard to assess the trade-off vs diffusion models.

4. **Empirical advantages over strongest baselines are mixed and under-analyzed**  
   While **Table 1** shows several metrics where ARSS is best, the story is not uniformly positive:
   - On RealEstate10K, SEVA attains better SSIM (0.670 vs 0.624) and slightly better FID (46.98 vs 47.60) while ARSS has higher PSNR and lower LPIPS. The text acknowledges “minor geometric inconsistencies” but does not dig into *why* the model lags in SSIM/FID (e.g., oversharpening, local distortions) or show failure cases.
   - On ACID, ARSS has best PSNR and LPIPS but SEVA has notably better SSIM (0.664 vs 0.623) and much better FID (33.16 vs 47.76), and LVSM also beats ARSS on FID. Again there is no analysis of this discrepancy.
   - For DL3DV zero-shot, ARSS does beat LVSM and MotionCtrl in most metrics, which is good, but the absolute numbers (PSNR 16.70, SSIM 0.449) are still modest.  
   Without more detailed qualitative side-by-side comparisons focused on these failure regimes or an explanation tied to the model design (e.g., tokenizer limitations, camera encoding inaccuracies), the claim that ARSS is “overall comparable to state-of-the-art diffusion models” feels a bit too strong. **Figure 3** and **Figure 4** are helpful, but a more balanced discussion of where ARSS fails compared to SEVA or other strong baselines would improve credibility.

5. **Insufficient ablations on camera token design and conditioning mechanism**  
   The camera autoencoder and per-position camera tokens are a central contribution, yet the ablation coverage is thin:
   - There is no comparison between Plücker-ray-based encoding vs simpler encodings (e.g., 6D pose repeated spatially, or ray directions without momentum) to justify the additional complexity of the Plücker representation and geometry losses in **Equation (5)**.
   - There is no ablation showing “no camera tokens” or “single global camera token per frame”, which would help quantify how much of the performance in **Table 1** comes from detailed 3D guidance versus just the video tokenizer + AR model.
   - The camera autoencoder architecture is relegated to **Appendix A.1** and **Figure 8**, but even there there are no training details (dataset size, reconstruction error, whether it is trained jointly with the AR model, etc.).  
   Given that the authors emphasize camera control and 3D consistency, more rigorous experimentation on the conditioning mechanism is essential.

6. **Ambiguity around causal video tokenizer configuration and training**  
   Section 3.1 (“Causal Video Tokenization”) briefly describes a “causal scenario” where the first frame is not compressed temporally and is used as conditional tokens. However, key details are missing:
   - Is VidTok pretrained and frozen, or finetuned on the NVS datasets? Section 4.1 (“Implementation Details”) implies it is simply “applied,” but this should be stated precisely; tokenizer quality directly constrains ARSS quality, as acknowledged in Section 5.
   - How exactly are the `(L+1)` frames mapped to `(l+1)` latent time steps in practice given the chosen patch sizes? It would help to make **Equation (4)** more concrete for the causal case, perhaps with an explicit mapping illustration extending **Figure 2** (left).
   - Does the tokenizer itself operate with a causal temporal receptive field, or is “causal” only at the AR transformer level? The wording in Section 3.1 suggests some frames are treated differently but does not fully specify the tokenizer’s temporal mask.  
   This ambiguity affects reproducibility and interpretation of the “causal” claim.

7. **Clarity and writing issues in several places**  
   Beyond the equations, there are several instances where the text is confusing or contains typos:
   - In Section 3.2.3, the permuted sequence in **Equation (6)** uses `π` to denote what earlier was `x` (visual tokens), and the sentences right before and after use inconsistent variable names (`x`, `π`, sometimes `n` vs `N`).  
   - The description of the training loss immediately below **Equation (7)** has mismatched brackets and is hard to follow even with **Figure 2** as reference.
   - In Section 4.3, the explanation of token-ordering strategies references “full perm.” and “raster” but the text there partially duplicates **Appendix A.4** and has slightly contradictory phrasings (e.g., “permutes target tokens only respect to spatial dimension” vs “permutes the target tokens only respect to spatial dimension while keeping the original temporal order”, which is presumably the same thing).  
   These issues do not make the paper unreadable, but they do lower presentation quality and make it harder to fully trust the exact formalization.

8. **Lack of discussion on limitations specific to AR for NVS**  
   The Discussion section mentions tokenizer limitations but not AR-specific ones. For instance:
   - Exposure to limited token orders: only random spatial permutations with fixed temporal order are used; no analysis of whether certain orders are harder, or learned ordering strategies (as explored in other AR works) would help.
   - Error compounding in autoregression: **Figure 6** suggests ARSS degrades more slowly than baselines, but the figure also shows non-negligible degradation; it would be useful to explicitly discuss in what scenarios the AR approach breaks down (e.g., very long trajectories, large parallax).  
   A more candid and concrete limitations section would strengthen the paper.

## Potentially Missing Related Work

1. **Kong, X., Watson, D., Strümpler, Y., “CausNVS: Autoregressive Multi-view Diffusion for Flexible 3D Novel View Synthesis” (2025)**  
   - Relevance: Directly tackles causal multi-view synthesis with autoregressive diffusion, addressing flexibility in input-output configurations and temporal causality, which is very close in spirit to ARSS (though with a different generative backbone).  
   - Where to cite: Should be discussed in **Section 2 (Novel View Generation with Diffusion Models)** as a key causal NVS method, and compared conceptually in **Section 5**, especially regarding how discrete AR vs AR-diffusion trade off quality, efficiency, and control.

2. **Yuan, S., Zhao, H., “Next-Scale Autoregressive Models are Zero-Shot Single-Image Object View Synthesizers” (2025)**  
   - Relevance: Uses autoregressive models for zero-shot single-image novel view synthesis of objects. While not directly scene-scale, it is conceptually relevant because it demonstrates AR-based view synthesis from a single image, with claims of fast inference and high accuracy.  
   - Where to cite: Should be added to **Section 2 (Autoregressive Visual Generation)** and contrasted in the Introduction and Discussion regarding AR for view synthesis vs AR for generic image generation.

3. **Wiles, O., Gkioxari, G., Szeliski, R., “SynSin: End-to-end View Synthesis from a Single Image” (ICCV 2019)**  
   - Relevance: Classic single-image NVS method that explicitly predicts 3D point clouds from a single image and renders new views, often used as a baseline in subsequent NVS literature. It focuses on 3D-aware generation from a single view and is directly relevant to the problem ARSS solves.  
   - Where to cite: Should be discussed in **Section 2 (Novel View Generation with Diffusion Models)** or a separate subsection on single-image NVS, and possibly mentioned in **Section 4.1** as an older but conceptually related baseline to contextualize advances beyond early non-generative methods.

## Questions

1. **Clarification of token permutation and causal masking**  
   Could the authors provide a precise, corrected definition of the input and target sequences used to train the transformer, including both visual and camera tokens, and the exact causal mask? Specifically:
   - What is the order of tokens in `\mathcal{S}` (camera vs visual) for each frame?
   - What is the exact next-token prediction target for each input position?
   - Is the causal attention mask purely 1D along the permuted order, or is any frame-structure preserved?  
   A small table or diagram extending **Figure 2 (right)** with one concrete example would be very helpful.

2. **Camera autoencoder ablations**  
   How much performance is lost if:
   - (a) camera tokens are removed entirely and only the input frame tokens are used as condition, or  
   - (b) a simpler camera encoding is used (e.g., repeating 6D pose per spatial position, or using only ray directions without the Plücker momentum term) instead of the full Plücker-based autoencoder?  
   Even a partial ablation with PSNR / LPIPS for such variants on RealEstate10K would clarify the contribution of the geometry modeling in **Equation (5)**.

3. **Runtime and scalability comparisons**  
   Can the authors report:
   - Average inference time per 17-frame trajectory on RealEstate10K for ARSS vs SEVA and LVSM on the same hardware.
   - Maximum sequence length (number of target frames) they could generate without running out of memory on a single GPU.  
   This would substantiate the claims about suitability for long trajectories and world modeling.

4. **Tokenizer training and causal configuration**  
   Is VidTok used as-is from its original pretraining, or is it finetuned on RealEstate10K/ACID? If finetuned, what objective and training schedule are used? Furthermore, in the “causal scenario” described in Section 3.1, does VidTok itself use a causal temporal receptive field, or is it trained non-causally but only applied causally via AR at generation time?

5. **Failure cases and comparison to SEVA**  
   Can the authors show some representative failure visualizations where SEVA does better than ARSS, particularly on ACID where SEVA has much better FID, and analyze whether issues stem from the tokenizer, camera encoding, or AR modeling? This would help clarify when diffusion models remain preferable.

## Flag For Ethics Review

- No ethics review needed.  

## Details Of Ethics Concerns

N/A.

## Soundness Rating

3: good.  
The core idea and experimental methodology are generally sound, and the ablations on token permutation and tokenizer choice support key design decisions. However, some mathematical definitions (Equations 5–8) are inconsistent or incomplete, and there is missing analysis of runtime/scalability and camera-encoder ablations that would strengthen the empirical support.

## Presentation Rating

2: fair.  
The high-level story and figures are understandable, but several equations are malformed or confusing, some notation is inconsistent, and important details (token order, causal masks, camera autoencoder training) are underspecified. This hurts clarity and reproducibility.

## Contribution Rating

3: good.  
The paper pushes autoregressive modeling into a challenging NVS setting, with a coherent combination of video tokenization, geometrically informed camera tokens, and spatial permutation. While not conceptually radical, it is a meaningful extension with competitive results and insightful ablations.

## Overall Rating

6: Marginally above the acceptance threshold. But would not mind if paper is rejected.  
The paper presents a timely and technically interesting AR-based approach to novel view synthesis, with solid quantitative performance and thoughtful analyses of tokenization and token ordering. At the same time, it has notable issues in mathematical clarity, limited exploration of the camera conditioning design, incomplete positioning versus closely related AR NVS work, and no runtime/scalability analysis. With clarifications and additional experiments, it could be a strong contribution, but in its current form I view it as a decent, slightly-above-threshold paper rather than a clear accept.

## Reviewer Confidence

4: confident.  
I am familiar with autoregressive visual generation, diffusion-based NVS, and tokenization methods, and I carefully checked the equations and experiments, though I did not attempt to reimplement the method.