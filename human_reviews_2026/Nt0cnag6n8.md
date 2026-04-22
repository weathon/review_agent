# Zero-Shot Video Restoration and Enhancement with Assistance of Video Diffusion Models

- Avg Score: 5.50
- Decision: Reject
- Scores: 8, 4, 4, 6

## Abstract
Although diffusion-based zero-shot image restoration and enhancement methods have achieved great success, applying them to video restoration or enhancement will lead to severe temporal flickering. In this paper, we propose the first framework that utilizes the rapidly-developed video diffusion model to assist the image-based method in maintaining more temporal consistency for zero-shot video restoration and enhancement. We propose homologous latents fusion, heterogenous latents fusion, and a COT-based fusion ratio strategy to utilize both homologous and heterogenous text-to-video diffusion models to complement the image method. Moreover, we propose temporal-strengthening post-processing to utilize the image-to-video diffusion model to further improve temporal consistency. Our method is training-free and can be applied to any diffusion-based image restoration and enhancement methods. Experimental results demonstrate the superiority of the proposed method.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes ZVRV, a training-free framework that leverages video diffusion models to improve temporal consistency in zero-shot video restoration. Key contributions include homologous and heterogeneous latent fusion methods (enabling the use of any T2V model regardless of VAE compatibility), a COT-based strategy for adaptive fusion-weight selection, and temporal post-processing with I2V models. Experiments show 73-84% reduction in temporal flickering across super-resolution and low-light enhancement tasks, achieving state-of-the-art results.

### Strengths
- The paper identifies and addresses a real limitation: temporal flickering when applying image restoration methods to videos by leveraging temporal priors from pre-trained video diffusion models.
- The key technical insight enabling cross-VAE latent fusion through encode-decode operations is elegant and practical. This allows the framework to utilize state-of-the-art T2V models (CogVideoX, HunyuanVideo) regardless of VAE compatibility.
- Demonstrates 73-84% reduction in temporal flickering (Warping Error) while maintaining/improving visual quality metrics. The method outperforms both supervised and zero-shot baselines across three different tasks.
- The framework requires no training and works with any LDM-based restoration method, making it immediately practical and deployable.
- Applying chain-of-thought reasoning with test-time verification to select fusion ratios is an interesting adaptation of LLM techniques to diffusion models.

### Weaknesses
- The method runs multiple models simultaneously (image restoration + homologous T2V + heterogeneous T2V + I2V post-processing), and the COT strategy requires M+1 forward passes at each timestep for three different fusion ratios (\lambda^F1, \lambda^F2, \lambda^F). This represents massive computational overhead, yet the paper provides no runtime analysis, memory requirements, or comparison with baselines.
- The paper's core contributions have limited novelty. Homologous fusion is acknowledged as similar to FVDM (Lu et al. 2024), adapted from video editing to restoration, and heterogeneous fusion via VAE encode-decode is relatively straightforward. The COT application to fusion ratio selection, while interesting, feels more like an engineering trick than a fundamental contribution. The central insight is essentially "use video diffusion models to reduce temporal flickering," which is intuitive but not deeply novel.
- The evaluation uses very small test sets (18 videos for super-resolution, 10 for enhancement), raising concerns about generalization, and resolution is severely restricted to 576x320 due to computational constraints, which doesn't reflect real-world high-resolution video restoration needs. The paper lacks failure case analysis or discussion of when the method doesn't work, and critical ablations are missing for key hyperparameters such as M and r in the COT strategy, and the choice of fusion schedules.

### Questions
- Can you provide detailed runtime comparisons (wall-clock time, memory usage, FLOPs) against baseline methods? How many total forward passes does your method require per video with typical M values? What is the practical maximum resolution your method can handle on standard GPUs, and how does this scale?
- How sensitive is performance to the choice of M (sample number) and r (range)? Can you provide ablation studies showing the performance vs. computational cost trade-offs for different M values? Why use COT sampling at every timestep rather than learning or predicting good fusion ratios?
- The test sets are very small (18 and 10 videos). Have you evaluated on larger benchmarks? Can you provide results on standard video restoration datasets at full resolution? How does the method perform on longer videos (>50 frames)?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a zero-shot video restoration framework that leverages diffusion generative priors for multiple degradation tasks without training. The method performs diffusion inversion on each frame and introduces a temporal alignment module for consistency across frames. The approach is applied to denoising, deblurring, and super-resolution tasks, showing competitive zero-shot performance.

### Strengths
- The proposed zero-shot setting is novel and practically motivated.  
- Integrating diffusion priors with temporal alignment is conceptually elegant.  
- The method demonstrates versatility across multiple restoration tasks.

### Weaknesses
- Lacks comparison with recent diffusion-based video restoration models such as **Upscale-A-Video (CVPR 2024)** and **SeedVR (CVPR 2025)**, making it hard to gauge true competitiveness.  
- No runtime, peak memory, or parameter analysis is provided, which limits understanding of efficiency and scalability.  
• Temporal consistency evaluation is weak, reporting only **Warping Error (WE)** without metrics like **DOVER** or **tLPIPS**, which better reflect human-perceived temporal coherence and detail stability.

### Questions
1. How does the method perform compared with diffusion-based video restoration baselines such as Upscale-A-Video or SeedVR  
2. Can the authors report runtime, peak memory, and parameter counts for a fair efficiency analysis  
3. Please include more temporal consistency metrics (e.g., DOVER, tLPIPS) to strengthen the evaluation

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
I think the paper grafts pre-trained T2V/I2V diffusion models onto image-based zero-shot IR to reduce flicker:  (1) homologous latent fusion when the 2D VAE matches.  (2) heterogeneous latent fusion via 2D↔3D VAE round-trips. (3) a “CoT-based” per-timestep fusion-ratio search with CLIP-IQA + warping-error as the verifier.  (4) A Stable Video Diffusion post-processing pass for extra temporal smoothing. On REDS/Vid4/UDM10/DAVIS and a small low-light set, the method beats PSLD and some zero-shot/editing baselines on PSNR/SSIM/LPIPS/CLIP-IQA/WE/FVD.

### Strengths
(1) I think the goal is practical: using actual video priors to fix temporal instability in zero-shot diffusion IR.

(2) The heterogeneous latent bridge (2D↔3D VAE encode/decode to align latents) is a straightforward engineering workaround that makes modern T2V usable.

(3) The pipeline is training-free w.r.t. new networks and slots into several zero-shot IR backbones.

(4) Results show lower WE/FVD and perceptual gains over PSLD-only variants; ablations indicate each block helps.

### Weaknesses
(1) “First framework” is oversold. I think the novelty is thinner than claimed. Homologous fusion is basically FVDM-style latent mixing applied to restoration rather than editing; heterogeneous fusion is a vanilla encode–decode bridge between VAEs; and the “CoT-based” strategy is just a best-of-N hyperparameter search per timestep with two off-the-shelf metrics. Slapping “CoT” on a verifier-guided grid search doesn’t make it reasoning-based. The pitch feels buzzwordy rather than conceptually new.    

(2) The “training-free” recipe quietly burns a lot of test-time compute. I think sampling multiple fusion ratios at every diffusion step for every clip is a huge multiplier on runtime. Then you post-process with SVD (inversion + EDM sampling) on top. There’s no honest wall-clock accounting or energy/latency comparison vs. stronger baselines at their own optimal step counts. This makes the method look practical on paper but painful in reality.   

(3) Metric gaming and small-scale evaluation. The main verifier metric at test-time includes CLIP-IQA, which your own method optimizes against during ratio selection; unsurprisingly, you win it. Datasets are tiny (e.g., 18 videos for SR; 10 for low-light). Frames are even down-cropped to 576×320 “due to slow sampling, which conveniently hides high-res temporal artifacts. I think this undercuts the generality of the claims.  

(4) Fairness & clarity of baselines. I think comparisons are muddy: you mix supervised VSR models (trained for the task) with zero-shot methods, and for diffusion baselines, you lock them to specific step counts instead of reporting speed–quality curves. The DavIS blind SR table shows large gains, but I don’t see solid controls that exclude metric bias from the verifier loop.

### Questions
- What’s the end-to-end wall-clock (and GPU budget) per 1-second 1080p clip for your full pipeline (fusion-ratio search + SVD)? Give a step-wise breakdown.   

- Verifier leakage: Since CLIP-IQA is used to select fusion ratios, do you also report metrics not used in selection (e.g., t-LPIPS variants, VMAF-temporal) where you didn’t tune?  

- Please provide speed–quality trade-off curves for PSLD/other diffusion baselines (10/25/50 steps) and your method with/without SVD, at matched wall-clock.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper proposes a new video restoration method that leverages an existing image restoration model, along with homologous and heterogeneous T2V models. The key idea is to fuse the latents from different models to perform per-frame restoration while maintaining temporal consistency. A post-processing step is then applied using an I2V model to further enforce consistency. The method generally shows improved results when combined with an existing image restoration (IR) method.

### Strengths
1. The method is unsupervised and training-free, leveraging existing pre-trained models, which makes it practical.

2. The method generally shows improvements compared to the baseline or other compared methods.

### Weaknesses
1. I think the method section could be much better presented and many details should be introduced. I struggled to fully understand the method.  How do you get the noise $z_T$ in line 201? Do you invert the degraded video? How do you use the T2V model to generate a video similar to the input one?  Which text prompt are you using? In line 182, you mentioned that your input is only a video. 

2. Based on my understanding, I think the performance depends heavily on the hyperparameters used for latent fusion, which makes the method seem somehow ad hoc.

3. The method applies multiple models, which makes it computationally expensive. A computational comparison is needed.

### Questions
See weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
