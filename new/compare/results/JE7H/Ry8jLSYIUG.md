---
job_id: a34dc2e4-c55d-4ff8-8038-fa449762a9cd
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: Ry8jLSYIUG.pdf
paper: We Can Hide More Bits: The Unused Watermarking Capacity in Theory and in Practice
main_score_norm: 0.6
desk_reject: false
---
# Desk Rejection Assessment:

## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper studies theoretical and empirical limits of deep-learning-based image watermarking, including capacity bounds, robustness under transformations, and a new large-scale watermarking model, all of which fall well within ICLR’s scope of representation learning, learning theory, and applications to computer vision.

## Minimum Quality
Pass ✅.  
The paper is in English and has all major components: Abstract, Introduction, Related Work (Section 2.1 and Appendix A), clear theoretical methodology (Section 2 and appendices), experiments and results (Sections 3–4, Tables 1–5, Figures 1, 3–6, 12, 13–14), and Discussion/Conclusions (Section 5). The work is technically nontrivial and not obviously flawed; experiments are reasonably substantial and reproducible from the description. No fatal methodological or statistical issues are apparent.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I see no hidden prompts or attempts to manipulate automated reviewing systems in the main paper content.

---

# Expected Review Outcome:

## Summary

The paper analyzes the fundamental capacity of digital images for robust watermarking under PSNR constraints and linear robustness constraints. Using a geometric view of the image space, it derives multiple upper bounds on the number of embeddable bits for various PSNR regimes, cover positions, and linear transformations (including a linearized JPEG). The authors then empirically compare these bounds with existing watermarking models, showing large capacity gaps, and introduce Chunky Seal, a heavily scaled-up version of Video Seal that reaches 1024-bit payloads with competitive robustness and quality, thereby demonstrating that current architectures significantly underutilize the available watermarking capacity.

## Strengths

1. **Clear, well-structured theoretical framework for capacity under PSNR constraints.**  
   The progression from the absolute capacity (Bound 1), through PSNR-only constraints (Bounds 2–6), to arbitrary covers (Bounds 7–9), is conceptually clean and mathematically traceable. Equation (1) on Page 3 correctly connects PSNR to an $\ell_2$ ball with radius $\epsilon(\tau) = \rho\sqrt{cwh}10^{-\tau/20}$, grounding later geometric arguments. The use of both volume approximations (Equation (3)) and exact lattice counts via Mitchell’s algorithm (Algorithm 2) is thoughtful and the transition between them is quantified in Figure 7.

2. **Convincing visualization of the theoretical–practical gap.**  
   Figure 1 nicely juxtaposes theoretical capacity curves (PSNR-only and with robustness) against recent methods’ performance on a log-scale inset. Combined with Figure 3, which shows that even for a $16\times16\times3$ image and $\text{PSNR}=45\,$dB the bounds predict roughly 2000 bits, the paper makes a strong visual case that current practice (≈0.001 bpp) is far from saturating capacity. Figure 4 further shows that crop, rotation, and LinJPEG transformations, while reducing capacity, still leave substantial bpp, strengthening the main claim that robustness alone does not explain the gap.

3. **Detailed and nontrivial analysis of robustness via linear operators.**  
   Modeling augmentations as linear maps followed by quantization (Section 2.5 and Appendices G & H) is technically solid. The heuristic capacity scaling factor $\xi_M$ in Equation (6) and the linear-transformation bounds (Bounds 10–12) are well-motivated from singular values. The paper is refreshingly honest about their heuristic nature, and demonstrates both over- and under-estimation (Figures 8–9). The very conservative but formal lower bound (Bound 13 and Theorem 2) is mathematically intricate, using zonotope over-approximations and box–ball intersection volumes via Theorem 1, and provides a nontrivial “worst-case” sanity check.

4. **Careful experimental probing of architectural limitations using simple setups.**  
   The experiments on a single gray image under PSNR-only constraints (Section 3) are minimalistic yet very informative. Figure 5 and Table 1 clearly show that Video Seal fails to reliably embed 1024 bits (especially at $256\times256$ and $32\times32$), while a simple linear encoder–decoder can embed 2048 bits at $\approx40$–44 dB with 100% bit accuracy. This is a strong empirical argument that the gap is not due to realism of constraints but due to model structure and training, directly supporting hypothesis E in Section 3.

5. **Nice use of model “sanity checks” and tiling trick.**  
   The observation that Video Seal’s effective capacity at $256\times256$ is similar to that at $32\times32$ (Figure 5 center and Table 1) is a sharp diagnostic. The tiling experiment, which yields 32,768 bits at ≈41.7 dB by applying a $32\times32$ 512-bit model across patches, is a clever and simple demonstration that capacities closer to the theoretical bounds are accessible even with off-the-shelf architectures once trivial geometric decompositions are exploited. Figure 6 summarizes these empirical baselines against the theory nicely: the handcrafted scheme from Equation (2) nearly matches the bound curve, the tiled model sits substantially higher than the original, and Video Seal is clearly underperforming.

6. **Chunky Seal provides concrete, positive evidence that capacity can be scaled.**  
   Chunky Seal, although mostly a scaled-up Video Seal, is an important empirical data point: Table 3 shows 1024 bits at 0.0052 bpp with PSNR 45.3 dB and overall bit accuracy ≈99.15%, broadly matching Video Seal’s robustness across a wide set of augmentations (flip, rotation ≤10°, cropping, brightness, contrast, JPEG, blur). Extended Tables 4–5 over SA-1B and COCO reinforce that this 4× capacity increase does not catastrophically hurt robustness or visual quality (SSIM/MS-SSIM remain very high).

7. **Good coverage of classical watermarking capacity theory and motivation for a new angle.**  
   Section 2.1 and Appendix A present a fair and reasonably comprehensive overview of Gel’fand-Pinsker based analyses, Gaussian “dirty paper” results, and hiding capacity formulations. The critique that these rely on idealized continuous, probabilistic models and do not handle geometric distortions or discrete quantization is well articulated, making a convincing case for the proposed discrete geometric approach.

8. **Clarity and transparency.**  
   The writing is clear overall, with most definitions and notation introduced explicitly. The authors are transparent about the limitations of their approximations (e.g., volume vs lattice counts, numerical integration issues in Bound 5, and the non-rigorous status of Bounds 10–12), which increases confidence in the integrity of the work.

## Weaknesses

1. **Theory is limited to PSNR and linear robustness, which are still far from real watermarking objectives.**  
   While the paper explicitly adopts PSNR (Equation (1)) and linear operators plus quantization as a tractable model, in practice modern watermarking methods are constrained by much richer perceptual metrics and highly nonlinear postprocessing (e.g., adaptive resampling, nonlinear filters, generative edits). The authors briefly acknowledge this in Section 3 but then use success on gray images with PSNR-only constraints to argue that models are fundamentally underperforming. This is a strong leap: the derived capacities at, say, 40–45 dB may be substantially overoptimistic when more realistic, learned perceptual metrics (LPIPS; GAN-perceptual distances) and non-linear, content-dependent attacks are considered. As a result, the conclusion that “current models have not yet saturated watermarking capacity” is convincing for the PSNR+linear setting, but less clearly transferable to the full real-world problem.

2. **Heuristic robustness bounds lack tightness and are only partially validated.**  
   Bounds 10–12 hinge on the heuristic factor $\xi_M$ in Equation (6), which is only guaranteed to be accurate for axis-aligned transformations. Figures 8 and 9 explicitly show that for simple 2D examples with rotation, the heuristic is neither an upper nor a lower bound: capacity can be 20% higher (Figure 8) or as low as 0.837 of the original (Figure 9) despite $\xi_M=0.5$ or 1. The paper then proceeds in Section 2.5 (and Figure 4) to heavily rely on these heuristics for crop & rescale, rotation, and LinJPEG, all of which involve non-axis-aligned mixing and interpolation. Although Bound 13 is given as a conservative lower bound, it essentially decouples from the heuristic regime and is extremely loose due to the zonotope -> axis-aligned box over-approximation (Theorem 2, Equations (8,9)). This raises real concerns about how informative the mid-range robustness curves in Figure 4 actually are: they could be off by a large factor in either direction. A more systematic empirical validation (e.g., Monte Carlo capacity estimates for low-dimensional problems under each transform) would help calibrate these bounds.

3. **Some key mathematical constructs are opaque or underexplained for the general ML audience.**  
   Several theoretical pieces are heavy and hard to parse. For example, Bound 5 (Section 2.3.3 and Appendix E) uses Theorem 1 with Fresnel integrals and an infinite sum to compute cube–ball intersections (Equation (4)), but the practical implementation details and truncation error are not discussed quantitatively. Similarly, Theorem 2 and Bound 13 rely on constructing a zonotope over-approximation of the preimage of a hypercube under $M$; while Appendix I provides a step-by-step derivation, the core idea that the over-approximation can be arbitrarily loose is only qualitatively mentioned, not quantified. A reader not already comfortable with convex-geometry techniques may struggle to assess the reliability of the theoretical conclusions; more intuition, bounds on approximation errors, or simple 2D illustrations beyond Figures 8–9 would help.

4. **Empirical evidence primarily targets Video Seal; other robust methods are mostly evaluated only at low capacity.**  
   The central empirical criticism is that Video Seal is structurally limited, based on gray-image experiments (Figure 5, Table 1) and scaling behavior (Section 3.2). However, the argument that “modern deep watermarking architectures are flawed” would be much stronger if repeated with other architectures like HiDDeN, MBRS, TrustMark, or WAM in the same simplified PSNR-only regime. Instead, those methods are only evaluated at their default low capacities under standard augmentations (Tables 4–5 and Figures 13–14) and not in the crucial gray-image capacity-stretching experiment. This omission weakens the generality of the main empirical claim: it may be that Video Seal (with its U-Net + ConvNeXt design) is particularly suboptimal for capacity, while other architectures are less so.

5. **Handcrafted and linear baselines are not robustness-aware, which limits their interpretability.**  
   The linear encoder–decoder and the handcrafted mapping given in Equation (2) achieve strong capacity–PSNR trade-offs (Figure 6 and Table 1), nearly matching the PSNR-only bounds in the gray-image setting. However, these constructions are not shown to be robust to *any* realistic augmentations, including even mild cropping, rotation, or JPEG. In contrast, Chunky Seal and baselines like MBRS, WAM, etc., are explicitly trained with such attacks (Tables 3–5). The paper uses these baselines to argue that “our bounds are empirically reachable,” but the comparison is apples-to-oranges from a robustness standpoint. A more balanced perspective would either: (i) explicitly test robustness of the handcrafted and linear schemes, even in a simplified transform regime; or (ii) clearly state that their success only validates the *PSNR-only* bounds and does not speak to the joint capacity–robustness frontier.

6. **Chunky Seal’s contribution is mostly scale, with limited architectural or algorithmic insight.**  
   While Chunky Seal convincingly increases capacity 4× (Table 3), it is essentially a brute-force scaling of Video Seal: channel multipliers [4,8,16,32], embedding dimension 2048, all RGB channels, ConvNeXt-base extractor with more depth, plus gradient clipping. The embedder has 1B parameters and the extractor 774M (Table 3), which is arguably impractical for real-world deployment, and the training details beyond clipping are fairly standard. There is no ablation on which architectural changes matter for capacity (e.g., role of multi-channel watermarking, depth vs width, removing luma-only restriction, etc.), and no exploration of more structured designs motivated by the geometric theory. Consequently, while Chunky Seal is an informative feasibility study, its scientific contribution beyond “scale it up” is limited.

7. **Capacity claims in the presence of data distributions could be more carefully defended.**  
   Section 2.6 argues that the data distribution reduces capacity by at most ≈0.05 bpp, using a VQ latent representation with a 1024-codebook and a $32\times32$ grid (Muckley et al., 2023). The argument assumes that all possible VQ code combinations are valid perceptual images and could lie within a single PSNR ball around a given cover, which is extremely conservative, but also assumes that collisions among *watermarked* images for different covers are handled only by simple partitioning into $N$ regions, leading to a $\log_2 N$ penalty. This is plausible, but the text glosses over the more subtle question of how decoding ambiguity manifests in blind watermarking when the decoder is a learned neural net rather than a combinatorial map. Some empirical support (e.g., measuring collisions across a dataset when pushing capacity) would make the dataset-level conclusions more robust.

8. **Some experimental design choices could be better motivated or expanded.**  
   - In Figure 5, the grid search over learning rates and MSE weights $\lambda_i$ is relatively coarse; for the critical 1024-bit case, it is plausible that more extensive tuning (or curriculum training from lower capacities) could push Video Seal closer to 100% accuracy, weakening the argument of “fundamental structural limitations.”  
   - Chunky Seal’s training description is fairly brief: batch sizes, optimizer hyperparameters, augmentation schedules, and any regularization beyond gradient clipping are not fully detailed in the main text.  
   - The evaluation uses bit accuracy as the main robustness metric. For capacities as high as 1024 bits, it would be informative to also report message accuracy (all bits correct) and per-bit error rates under heavy attacks to quantify how errors scale with length.

9. **Minor mathematical and expository issues.**  
   - In Equation (2), the definition of $q = 2\lfloor 2^k 10^{-\tau/20}\rfloor + 1$ and the derivation of $d = \rho 10^{-\tau/20}$ from Equation (1) rely on the cover being exactly at the center and the largest inscribed cube; this is fine in the gray-image case, but it would be helpful to explicitly state these assumptions near the equation rather than referring back to prior sections.  
   - Bound 7 in Appendix F underestimates capacity due to ignoring lattice points on cube faces. This is acknowledged, but the discontinuity between Bounds 7 and 8 in Figure 3 right could be highlighted more clearly, as it visually suggests an artificial “kink” that is not present in the true capacity curve.  
   - Algorithm 4 and BallCubeIntersection (Page 22) are quite intricate; a small worked 2D example, analogous to Figure 2 but with numeric values, would significantly improve readability.

## Potentially Missing Related Work

1. **Bistroń et al., “Deep Learning for Image Watermarking: A Comprehensive Review and Analysis of Techniques, Challenges, and Applications,” 2026.**  
   This is a recent, comprehensive survey of deep learning-based watermarking methods, discussing capacity, imperceptibility, and robustness trade-offs. It is directly relevant to Sections 1 and 3, where the paper positions existing methods as plateauing around 100–200 bits. It should be cited in the Introduction and Appendix A when summarizing the state of the field, and might provide a more systematic baseline list for Table 4–5 comparisons.

2. **Chahar et al., “Deep Learning-Empowered Image Steganography: Architectural Innovations and Performance Benchmarking,” 2025.**  
   While focusing on steganography rather than watermarking per se, this paper analyzes architectural choices and performance trade-offs similar to those discussed here (capacity vs imperceptibility vs robustness). It is especially relevant to the discussion in Sections 3 and 5 about architectural limitations and scaling behavior. It should be referenced in Appendix A’s “Deep learning based-watermarking” subsection and considered in the discussion of model design space.

## Questions

1. **How sensitive are the gray-image Video Seal capacity failures (Figure 5, Table 1) to training protocol details?**  
   Would more extensive hyperparameter sweeps, curriculum training from lower capacities, or architectural tweaks (e.g., removing the identity-path bias or modifying the bottleneck) allow 1024-bit or higher capacities at comparable PSNR? Some controlled ablation (even small-scale) could quantify whether the issue is truly architectural or mostly training-related.

2. **Can you empirically validate the heuristic robustness bounds (Bounds 10–12) in moderate dimensions?**  
   For example, for a $16\times16$ gray image under specific linear transforms (e.g., Crop&Rescale 50% and Rotation 30° with bilinear interpolation), could you estimate capacity via random coding (Monte Carlo sampling and decoding) and compare it to the predictions from Figure 4 and Table 2? This would provide crucial evidence that the heuristics are not wildly optimistic.

3. **What happens if you push Chunky Seal beyond 1024 bits under the same training settings?**  
   Even if training becomes unstable, it would be informative to see up to what capacity (e.g., 2048 bits, 4096 bits) the model can converge with acceptable PSNR and how robustness degrades. This could better characterize the practical limits of “scale it up” before architectural changes become mandatory.

4. **How robust are the handcrafted and linear schemes to simple attacks?**  
   While you focus on PSNR-only constraints for them, a small experiment showing their behavior under, say, mild JPEG compression (quality 80), small rotations (≤10°), or 75–95% crops would help position these constructions relative to neural methods in realistic settings.

5. **Could you clarify the error behavior of your numerical integration-based bounds?**  
   For Bound 5 and Theorem 1, how many terms in the infinite sum are used in practice, and do you have empirical or theoretical guarantees on the relative error in the resulting capacity bound? A small table of radius vs. truncation level vs. relative error in 2D or 3D would help readers trust these computations.

6. **Do your results extend to color images in theory, not just empirically?**  
   Most bounds are parameterized by $(c,w,h,k)$, but many examples focus on gray covers. Are there any subtle issues when dealing with 3 channels and color spaces (e.g., YCbCr vs RGB) in the capacity analysis, especially under LinJPEG, beyond the straightforward factor $c$?

## Flag For Ethics Review

- No ethics review needed.

## Details Of Ethics Concerns

N/A.

## Soundness Rating

3 good – The theoretical framework is largely sound and carefully developed, with clear assumptions; heuristic robustness bounds are a concern but are transparently labeled, and empirical methodology is reasonable though focused on a subset of architectures.

## Presentation Rating

3 good – The paper is well written and logically structured; some mathematical parts (Theorem 1, Bound 13, Algorithm 4) are dense for a general ML audience and could use more intuition, but overall exposition and figures (especially Figures 1, 3–6, 12–14) effectively communicate the main insights.

## Contribution Rating

3 good – The combination of geometric capacity analysis, empirical diagnosis of architecture limitations, and a 4×-capacity scaled model is a meaningful and timely contribution to the watermarking community, though Chunky Seal itself is mostly a scaling study and some claims about underutilization under robustness remain partially speculative.

## Overall Rating

6 Marginally above the acceptance threshold. But would not mind if paper is rejected. – The paper provides a well-argued theoretical and empirical case that contemporary deep watermarking models, at least in PSNR+linear robustness regimes, operate far below available capacity. The analysis is careful and the experiments around gray images and Chunky Seal are insightful, despite limitations in the robustness theory and the narrow focus on Video Seal-style architectures. On balance, the strengths and potential to spur better-capacity watermarking research justify a positive recommendation.

## Reviewer Confidence

4 confident – I am familiar with watermarking and representation learning literature and have read the math and experiments carefully; while some of the convex-geometry technicalities (Theorem 1, Theorem 2) are intricate, they do not undermine the core conclusions as presented.