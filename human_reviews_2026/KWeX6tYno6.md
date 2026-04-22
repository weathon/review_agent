# WorldSplat: Gaussian-Centric Feed-Forward 4D Scene Generation for Autonomous Driving

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6

## Abstract
Recent advances in driving-scene generation and reconstruction have demonstrated significant potential for enhancing autonomous driving systems by producing scalable and controllable training data. Existing generation methods primarily focus on synthesizing diverse and high-fidelity driving videos; however, due to limited 3D consistency and sparse viewpoint coverage, they struggle to support convenient and high-quality novel-view synthesis (NVS). Conversely, recent 3D/4D reconstruction approaches have significantly improved NVS for real-world driving scenes, yet inherently lack generative capabilities. To overcome this dilemma between scene generation and reconstruction, we propose \textbf{WorldSplat}, a novel feed-forward framework for 4D driving-scene generation. Our approach effectively generates consistent multi-track videos through two key steps: ((i)) We introduce a 4D-aware latent diffusion model integrating multi-modal information to produce pixel-aligned 4D Gaussians in a feed-forward manner. ((ii)) Subsequently, we refine the novel view videos rendered from these Gaussians using a enhanced video diffusion model. Extensive experiments conducted on benchmark datasets demonstrate that \textbf{WorldSplat} effectively generates high-fidelity, temporally and spatially consistent multi-track novel view driving videos.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses novel-view synthesis in driving-scene generation and introduces WorldSplat, a feed-forward 4D generation framework bridging video synthesis and 3D/4D reconstruction. Leveraging a 4D-aware latent diffusion model to produce pixel-aligned Gaussians, followed by refined video diffusion, it generates high-fidelity, temporally and spatially consistent multi-track novel-view videos. Experiments on benchmark datasets show its effectiveness in producing realistic and controllable driving scenes.

### Strengths
1. The paper is well written, and the figures are clear and informative.
2. The method demonstrates strong performance in generation metrics.
3. The motivation is reasonable — leveraging Gaussian splatting to ensure geometric consistency, while incorporating diffusion models to enhance quality on novel regions.

### Weaknesses
1. The main issue with the paper lies in the inconsistency between the motivation and the experiments metrics. The method is only evaluated using FID and FVD, which primarily measure generation quality. These metrics fail to capture geometric consistency, as well as multi-view and temporal coherence, all of which are missing from the validation. When comparing with NVS methods, it is usually also report reconstruction metrics (psnr, ssim) on the original viewpoints to ensure consistency. FID and FVD alone only reflect perceptual generation quality, while many details may differ significantly — as can be clearly observed in the appendix figures, where the generated backgrounds differ noticeably from the original real images.
2. I have some concerns about the proposed method. It seems that the success of novel view rendering may largely stem from the strong conditioning provided. If the projected boxes and trajectories are given, the overall object layout and scene distribution are already well constrained. Previous generative methods could also leverage this conditioning to perform novel view generation. The real challenge lies in maintaining background consistency, such as the mismatch of elements like utility poles observed in the appendix visualizations. Therefore, it remains unclear how much geometric consistency is truly contributed by the Gaussian-centric module, and how much is simply due to the strong conditioning.
3. Since the method is based on 4D Gaussian Splatting, it would be important to include experiments demonstrating temporal consistency. Qualitative visualizations could also help illustrate this aspect. Currently, the appendix only provides static rendering results, which are insufficient to evaluate the model’s performance on temporal coherence.
4. Table 4(a) lacks a comparison with the BEVFormer baseline.

### Questions
1. With the rapid development of generative models, stronger models naturally lead to better generation performance. Compared with previous methods, is the performance improvement mainly due to the proposed approach or the use of a more powerful diffusion model?
2. Are Diff1, the GS decoder, and Diff2 trained jointly, or are they optimized in separate stages? It’s not entirely clear from the Training Details section.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
In this paper, the authors propose a novel feed-forward framework for 4D driving-scene generation, which can integrate multi-modal information via a 4D-aware latent diffusion model to produce pixel-aligned 4D Gaussians in a feed-forward manner. What's more, the authors use an enhanced video diffusion model to refine the novel view videos rendered from these Gaussians. The experiments show that WorldSplat can generate high-fidelity novel view driving videos.

### Strengths
1. The authors focus on the limitations of existing methods—2D video generation lacks 3D consistency, while 3D/4D reconstruction lacks generative capabilities—and select 4D Gaussians as the core representation to improve 3D consistency, bridging the gap between generation and reconstruction.
2. The framework can generate a dynamic 4D Gaussian representation via a latent Gaussian decoder and then render novel-view videos through these Gaussians. This design effectively enhances both spatial and temporal geometric consistency, addressing the 3D inconsistency issue of existing 2D video generation methods.
3. Experiments on the nuScenes dataset show that the rendered novel-view videos have high fidelity(Tab.2) and the framework exhibits strong generative capabilities(Tab.1)

### Weaknesses
1. It seems that the training of the Gaussian Decoder relies on the seg loss, which requires additional time consumption due to the external use of SegFormer. Could the authors discuss the total training time of the framework, particularly the time overhead introduced by SegFormer?

### Questions
1. I am confused about Fig. 2. It seems that only the 4D-Aware Diffusion Model needs to be trained, but in the paper, the Latent Gaussian Decoder also needs to be trained.
2. It is hoped that the authors can provide more descriptions regarding the training time of the entire framework and all its training stages.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a Gaussian-centric, feed-forward 4D scene generation method for autonomous driving, called WorldSplat. The work explores how to combine generative models with accurate 4D reconstruction. WorldSplat introduces three key modules: the 4D-aware latent diffusion module, which transforms user-defined control conditions into temporally coherent latents; the latent Gaussian decoder, which converts these latents into explicit 4D Gaussian scene geometry; and the enhanced diffusion refinement model, which further improves the quality of rendered frames. Compared to other methods, WorldSplat achieves state-of-the-art performance.

### Strengths
- WorldSplat elegantly combines the generative ability of novel scene generation with the custom scene reconstruction capability of Gaussian scene (GS) models.
- WorldSplat achieves state-of-the-art performance under various conditions, including both custom and novel-view synthesis, outperforming other leading methods.

### Weaknesses
- The ablation study in this work is somewhat simplistic, and the authors tend to overstate their claims.
- The novelty of the module design is limited, and the methods section delves into well-known concepts, such as the diffusion process, without offering significant new insights.

### Questions
- In L226, it says "our decoder supports over 48 simultaneous input views." Could you clarify how the model supports 48 views? Why can't prior feed-forward reconstruction models achieve this capacity?
- Both the introduction and the ablation study mention the "Mixed Aug" technology, but I don't believe this technology is a key contribution of the current work. Could you clarify why it is mentioned so frequently?
- In WorldSplat, I would like to know how the performance is affected without the reprojection conditions in the Enhanced Diffusion Model. It seems that the restricted views sampled from the Aggregated 4D Gaussians are somewhat aligned with the conditions. Could this lead to performance degradation in novel-view synthesis?

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
4

### Summary
Current Driving scene generation methods lack 3D consistency for novel-view synthesis, while reconstruction methods cannot generate unseen scenarios. This paper introduces WorldSplat, a feed-forward framework that bridges this gap using explicit 4D Gaussian representations. The approach employs: (1) a 4D-aware latent diffusion model to generate multi-modal latents (RGB, depth, semantic) from control conditions, (2) a Gaussian decoder that produces pixel-aligned 4D Gaussians with static-dynamic decomposition for efficient novel-view rendering, and (3) an enhanced diffusion model for refinement. Experiments on nuScenes demonstrate impressive performance in video generation and novel-view synthesis, with improvements on downstream perception tasks.

### Strengths
- The submission contributes a novel feed-forward Gaussian decoder which effectively uses the latents from the 4D latent diffusion model to create accurate static/dynamic decomposed 4D Gaussians.
- The method effectively addresses a key limitation in the field by bridging the gap between video generation methods (which may contain artifacts and inconsistencies) and reconstruction methods (which lack generative capability). The use of explicit 4D Gaussian representations as an intermediate representation is clever and well-justified.
- The paper demonstrates significant improvements across multiple metrics, including FVD/FID scores for video generation (Table 1), novel-view synthesis quality (Table 2), and  downstream driving tasks (Table 4)
- The paper is well structured and lays out the rationale for the architecture clearly. The combination of a generative pipeline to create 4D Gaussians paired with a final diffusion model for inpainting unseen regions and overall refinement is well motivated by related work.

### Weaknesses
W1. While the improvements in FID and FVD compared to baselines even at varying degrees of camera displacements is solid, the work focuses on this improvement without examining or ablating how much of this improvement can be attributed to the backbone model (OpenSora v1.2) versus their own models.The downstream task comparisons also show meaningful improvements in mIoU and mAP for BEVFormer. Of note is the fact that the closest model in terms of performance to WorldSplat is DiVE, which also employs an OpenSora backbone in contrast to the MagicDrive works. The paper is missing a comparison of FID and FVD against DiVE despite it being a comparable model in terms of the end video diffusion model used to generate RGB frames.

W2. While the paper shows successful examples, it lacks in-depth discussion of failure cases and ablations. Table 3 demonstrates the improvements when using all of WorldSplat's individual components, but doesn't provide any explanation or intuition behind the results. For example, Version D shows that without enhanced diffusion, FVD degrades from 47.41 to 107.58 (>2x worse), suggesting the end diffusion model is doing a lot of the heavy lifting to uphold the FID and FVD scores rather than the generated Gaussians (despite the consistently better performance with displacement compared to other methods). There's also little discussion of failure cases at all.

### Questions
Additional questions:

Q1. In reference to W1, can the authors provide an comparison between methods using modern video backbones like OpenSora for FID and FVD?

Q2. What amount of generated data was used in the mixed "Real + Ours" result in Table 4b?

Q3. How do you prevent redundant Gaussians from multiple views of the same static regions?

### Soundness
3

### Presentation
4

### Contribution
3
