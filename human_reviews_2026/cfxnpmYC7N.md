# Mark3DGS: Protecting the Intellectual Property of 3D Gaussian Splatting with Robust Watermarking

- Avg Score: 4.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 4, 6

## Abstract
3D Gaussian Splatting (3DGS) has become a leading technique in computer vision and graphics, offering photorealistic scene representation and real-time rendering. However, due to high computational demands and the sensitivity of training data, 3DGS models face significant intellectual property theft risks, yet current protection mechanisms are insufficient. In this paper, we introduce Mark3DGS, a novel watermarking framework designed to protect 3DGS models. The framework includes perception-aware pruning for efficient Gaussian reduction, uncertainty-frequency-guided HVQ for resilient watermark embedding, tile-based rasterization with early termination and caching for optimized splatting, and adaptive extraction strategies for reliable watermark recovery. Additionally, we present MarkGS-Sim, a platform to evaluate watermark robustness across various 3DGS variants and conditions. Experimental results show that Mark3DGS outperforms state-of-the-art methods in watermark capacity, imperceptibility, and computational efficiency, achieving 206 FPS rendering, minimal storage ($\textless$ 200MB), compatibility with multiple 3DGS variants, and strong robustness to various watermark attacks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes Mark3DGS, a watermarking framework for protecting the intellectual property of 3D Gaussian Splatting (3DGS) models. The method includes perception-aware pruning, uncertainty-frequency-guided hierarchical vector quantization (HVQ) for watermark embedding, tile-based rasterization for efficient rendering, and adaptive extraction strategies. The authors also introduce MarkGS-Sim, a simulator for evaluating watermark robustness. Experiments demonstrate improvements in capacity, imperceptibility, and computational efficiency compared to existing methods.

### Strengths
## Strengths

1. MarkGS-Sim is a valuable contribution for the research community, enabling systematic evaluation of watermarking methods under various conditions and across 3DGS variants.

2. The method achieves significant improvements in computational efficiency (1.5× faster rendering than GS-hider, <50% storage) while maintaining high quality, making it practical for deployment.

3. Table 4 demonstrates successful adaptation to multiple 3DGS variants (2DGS, 4DGS, CompactGS, DreamGaussian) with >92% bit accuracy across all variants.

### Weaknesses
## Weaknesses

1. Mark3DGS is a combination of the existing methods of 3DGSW, GaussianMarker and NeRFSignature. Although it shows better performance, the contribution of this work is incremental.

2. The paper builds heavily on existing techniques (HVQ, SVD, DWT), and the perception-aware pruning using impact scores is similar to existing compression methods [1][2]. The technical novelty lies more in the integrations.

3. What is the theoretical maximum bit capacity? How does it scale with scene complexity and number of primitives? The capacity is basically similar to the existing methods [1,2,3], but there is no breakthrough in the capacity.

4. The extractor needs to be trained for each scene, and it lacks generalizability compared with the HiDDeN-based message extractor, which is generalizable to different scenes.

[1] 3D-GSW: 3D Gaussian Splatting for Robust Watermarking

[2] WateRF: Robust Watermarks in Radiance Fields for Protection of Copyrights

[3] GaussianMarker: Uncertainty-Aware Copyright Protection of 3D Gaussian Splatting

### Questions
## Questions 

1.  Why can the proposed method cooperate with 4DGS?  The uncertainty, frequency, and codebook methods are validated on the static scene method. There is a lack of proof that these methods can be extended to dynamic scenes.

2. How do you validate that the simulator accurately reflects real-world attack scenarios? Are physical simulations realistic for copyright protection use cases?

3. The method mentioned can be against the 3DGS compression attack while only including the Compact3DGS for experiments. How does the Mark3DGS perform against other quantization-based methods, such as HAC[1], or the pruning method PUP 3D-GS [2]? How does the Mark3DGS perform differently from the CompMarkGS [3] for compression?

[1] Hac: Hash-grid assisted context for 3d Gaussian splatting compression

[2] PUP 3D-GS: Principled Uncertainty Pruning for 3D Gaussian Splatting

[3] COMPMARKGS: ROBUST WATERMARKING FOR COM-PRESSED 3D GAUSSIAN SPLATTING

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
Mark3DGS proposes a 3DGS IP-protection framework combining perception-aware pruning, uncertainty- and frequency-guided HVQ, SVD-based codebook embedding with differentiable distortion layers, tile-based rasterization with early termination, and adaptive extraction, plus a simulator (MarkGS-Sim). It reports higher bit accuracy, imperceptibility, and efficiency than baselines (e.g., 206 FPS, <200 MB) across datasets and variants (Tables 1–4).

### Strengths
- Addresses the important problem of intellectual property protection for 3DGS models, which is increasingly relevant as 3DGS gains adoption.

- Impact-score pruning, adaptive thresholding, SVD codebook perturbation with fidelity regularization, frequency-weighted residuals, tile early-termination criterion, and extraction rule form a clear, implementable flow.

- Mark3DGS tops bit-accuracy vs. payload and robustness to image/model attacks (Table 2), while achieving 206 FPS and 193 MB storage.

### Weaknesses
- The core contributions are primarily engineering combinations of existing techniques (HVQ, SVD, DWT) rather than fundamental algorithmic innovations.

- The impact score formulation (Eq. 4) and adaptive threshold mechanism (Eq. 5) lack principled theoretical foundation or analysis. 

- Robustness covers standard distortions (Table 2) and limited fine-tuning; performance drops markedly when attackers have clean images + pose key (to ~53.71% at 500 epochs), yet this scenario is downplayed as “difficult to achieve” (Table 5). A principled active remover is not evaluated.

- Many critical thresholds/weights (e.g., $ \gamma $, $ \tau $, $ \tau_{\text{unc}} $, $ \lambda_{\text{freq}} $, $ \lambda_{\text{msg}} / \lambda_{\text{rec}} / \lambda_{\text{wavelet}} $) are tuned empirically (Sec. 3.1–3.4) with no error-bounds or optimality analysis.

### Questions
- Could the authors include an “active remover” benchmark, such as a parameter-domain denoiser within MarkGS-Sim, and report extraction AUC vs. PSNR under this adversary?

- Please provide sensitivity curves linking early-termination threshold $ \tau $ and pruning $ \gamma $ to $\{ \mathrm{FPS}, \mathrm{storage}, \mathrm{BitAcc}, \mathrm{PSNR} \}$ to expose trade-off knees, beyond the point results/Tables.

- Could you provide a theoretical analysis or empirical justification for the specific impact score formulation in Eq. 4? Why this particular combination of terms?

- How does the adaptive threshold mechanism (Eq. 5) generalize across different scene types and scales? Does it require per-scene tuning?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces Mark3DGS, a watermarking framework designed to protect 3D Gaussian Splatting (3DGS) models from intellectual property theft. The authors address existing protection inadequacies through a comprehensive approach that includes perception-aware pruning, uncertainty-frequency-guided HVQ for watermark embedding, optimized tile-based rasterization, and adaptive extraction strategies. The work also presents MarkGS-Sim, a platform for evaluating watermark robustness under various conditions. Experimental results demonstrate that Mark3DGS achieves superior performance in watermark capacity, visual imperceptibility, and computational efficiency (206 FPS rendering, <200MB storage) while maintaining compatibility across multiple 3DGS variants and resistance to various watermark attacks.

### Strengths
1. The method achieves state-of-the-art (SOTA) results in watermarking 3D Gaussian Splatting (3DGS) models, demonstrating both high visual fidelity and robustness.
2. The authors propose a unified simulation-rendering platform specifically designed for evaluating watermarked 3DGS models, which facilitates standardized benchmarking and future research.
3. The paper is well-written, with a logical flow and clear explanations that make it easy to follow and understand.

### Weaknesses
1. The wavelet transform computation in the Gaussian primitives pruning section should properly cite any referenced work, as building on existing methods requires appropriate attribution.
2. The second contribution, efficient watermark embedding, appears overly tricky, involving numerous parameters, which may limit its generalization in real-world scenarios.
3. The technical contribution of efficient watermark embedding is somwhat limited as it is based on HVQ. Meanwhile, ablation studies should be provided for identifying the advantage of introducing uncertainties into HVQ.
4. The proposed tile-based rasterization, seems to be a general techinique and irrelevant to watermark embedding.
5. No ablation studies to identify the contribution of each design.

### Questions
1. Why is clustering applied after adaptive pruning in this approach? Authors should provide more detailed clarifications.
2. How are the parameters τ_base and γ in equation 4 determined, and what is their sensitivity in general application scenarios? I think this is important because this affects whether the method can be directly applied to arbitrary GS scenarios.
3. Is tile-based rasterization relevant to the watermark embedding? 
4. What is the main design for the performance improvement?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes Mark3DGS, a 3DGS watermarking framework that includes perception-aware pruning, uncertainty-/frequency-guided HVQ, SVD-based embedding, and a tile-based rasterizer; it also releases a simulator to stress-test robustness. Experiments report strong bit accuracy at 16–48 bits, high visual fidelity, and notable efficiency, plus resilience to both image-level and model-level attacks.

### Strengths
1. Breadth of robustness evaluation. The results presented cover classic image attacks and model manipulations with consistently high bit accuracy on 32-bit payloads, which is rare in 3DGS watermarking papers. 

2. This paper provides clear ablations that aid reproducibility. The paper systematically studies loss design, hyperparameters, pruning ratio, and embedding scale, making the design choices transparent. 

3. Variant coverage. Demonstrations on 2DGS and other variants suggest the method isn’t tightly coupled to a single 3DGS stack.

### Weaknesses
1. Statistics of the results: Authors say the results are averaged over 10 simulations, while the tables don’t report variance or confidence interval, making it hard to judge the stability across runs. 

2. Limited scene diversity. Core experiments average across just four scenes plus a custom set with only ~180 images for the large-scale case; this narrows external validity.

### Questions
1. What is the storage breakdown of base Gaussians, SH codebooks and watermark metadata? 

2. Can the author explain the extension of this method to NeRF-based methods?

### Soundness
3

### Presentation
3

### Contribution
3
