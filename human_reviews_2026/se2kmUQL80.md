# Skyfall-GS: Synthesizing Immersive 3D Urban Scenes from Satellite Imagery

- Avg Score: 6.00
- Decision: Reject
- Scores: 4, 6, 6, 8

## Abstract
Synthesizing large-scale, explorable, and geometrically accurate 3D urban scenes is a challenging yet valuable task in providing immersive and embodied applications. The challenges lie in the lack of large-scale and high-quality real-world 3D scans for training generalizable generative models. In this paper, we take an alternative route to create large-scale 3D scenes by synergizing the readily available satellite imagery that supplies realistic coarse geometry and the open-domain diffusion model for creating high-quality close-up appearances. We propose **Skyfall-GS**, a novel hybrid framework that synthesizes immersive city-block scale 3D urban scenes by combining satellite reconstruction with diffusion refinement, eliminating the need for costly 3D annotations, also featuring real-time, immersive 3D exploration. We tailor a curriculum-driven iterative refinement strategy to progressively enhance geometric completeness and photorealistic textures. Extensive experiments demonstrate that Skyfall-GS provides improved cross-view consistent geometry and more realistic textures compared to state-of-the-art approaches.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This work proposes a city-block scale scene synthesis framework, which aims to synthesis 3D urban scenes solely from satellite imagery, without relying on 3D or street-view data.
The framework first initializes coarse geometry from satellite imagery, then leverages off-the-shelf diffusion model to iteratively refine close-up appearances in a "sky fall" way.
This approach shows geometrically consistent, visually realistic, and detailed textures.

### Strengths
1. Clear motivation. This work presents an interesting approach to avoid using 3D data by reconstructing urban scenes directly from satellite imagery and refining them with 2D diffusion priors.
2. Significant problem setting. Immersive 3D City is essential for various applications in gaming, animations and virtual reality content. 
3. Solid results. Quantitative metrics and user studies indicate that the proposed method outperforms existing state-of-the-art approaches. The ablation study is sufficient.

### Weaknesses
1. Ambiguous statement. From the current workflow, it is more like a reconstruction framework rather than a generative one. While the diffusion model improves visual quality for lower viewing angles, it's unclear how much diversity it can provide, given that the initial geometry is strongly determined in the reconstruction stage. The paper's claim "the first city-block scale 3D scene creation framework without costly 3D annotations" is therefore somewhat misleading. If this is a generative framework, diversity evaluation should be provided (although the visual results suggest potential overfitting).
2. Limited novelty. While the proposed framework is well-constructed, many of its technical components build upon existing methods. The reconstruction stage adopts techniques from SatelliteSfM[a], WildGaussians[b], and MoGe[c]. The iterative diffusion-based refinement for 3DGS is also explored in prior 3D scene generation methods (e.g., LucidDreamer[d], RealmDreamer[e], WonderWorld[f]).
3. Visual Quality. 
   - The synthesized scenes contain noticeable blur, artifacts, and hollow areas, even in central regions (e.g., Supplementary/video_results/NYC_010/stage2_aligned.mp4, 00:01–00:03, red building in the middle). The proposed IDU module aims to gradually reveal occluded regions and refine them. However, it is unclear whether these artifacts and hollow arise from occlusion that is not revealed by the IDU technique, or whether the revealed regions cannot be effectively refined. In addiction, it would be helpful to include a metric such as the percentage change of revealed area (revealed / [revealed + occluded]).
   - The scene quality degrades significantly near the boundaries of synthesized region, which limits scalability to larger urban areas. Since the paper claims city-block-scale synthesis, it would be helpful to clarify how much of the synthesized region is actually usable or explorable. Is this degradation caused by the center-oriented curriculum refinement strategy?
4. Potentially biased demonstration. It is unclear whether the shown visual results overlap with the refinement iterations from synthesis stage, which may produces overly favorable visual results. Showing more dynamic camera viewpoints and path would provide better validation of the proposed method.

[a] Kai Zhang, Jin Sun, and Noah Snavely. Leveraging vision reconstruction pipelines for satellite imagery. ICCV Workshops, 2019.

[b] Jonas Kulhanek, Songyou Peng, Zuzana Kukelova, Marc Pollefeys, and Torsten Sattler. WildGaussians: 3D gaussian splatting in the wild. NeurIPS, 2024.

[c] Ruicheng Wang, Sicheng Xu, Cassie Dai, Jianfeng Xiang, Yu Deng, Xin Tong, and Jiaolong Yang. Moge: Unlocking accurate monocular geometry estimation for open-domain images with optimal training supervision. CVPR, 2025.

[d] Jaeyoung Chung, Suyoung Lee, Hyeongjin Nam, Jaerin Lee, Kyoung Mu Lee. LucidDreamer: Domain-free Generation of 3D Gaussian Splatting Scenes. arXiv 2311.13384, 2023.

[e] Jaidev Shriram, Alex Trevithick, Lingjie Liu, Ravi Ramamoorthi. RealmDreamer: Text-Driven 3D Scene Generation with Inpainting and Depth Diffusion. 3DV, 2025.

[f] Hong-Xing Yu and Haoyi Duan and Charles Herrmann and William T. Freeman and Jiajun Wu. WonderWorld: Interactive 3D Scene Generation from a Single Image. CVPR, 2025.

### Questions
Some more minor questions/discussion are in below:
1. How diverse are the synthesized results when the reconstruction fails to provide accurate geometry, especially when buildings are occluded in the satellite view or only top-down imagery is available (e.g., [Longitude 37.626, Latitude 55.752])?
2. How does the method handle complex or irregular geometries (e.g., bridges, castles[Longitude 10.749, Latitude 47.557], or Gothic buildings[Longitude -2.644, Latitude 51.21])? It would be impressive if the proposed method can successfully handle these challenging cases.
3. Have you experimented with concatenated satellite imagery for larger-area synthesis, and if so, how does performance scale?

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
This paper introduces Skyfall-GS, a novel method for generating city-scale 3D scenes from satellite imagery. The key contribution lies in a two-stage reconstruction pipeline: first, an initial coarse 3D scene is reconstructed using 3D Gaussian Splatting (3DGS); second, the result is refined with a pre-trained text-to-image diffusion model. The authors further design a curriculum-based iterative dataset update strategy to progressively enhance reconstruction quality. Experimental results demonstrate that the proposed approach achieves significant visual improvements over existing methods.

### Strengths
1. The manuscript is well-written, clear, and easy to follow.
2. The use of text-to-image diffusion models for refinement is conceptually sound.
3. The proposed method achieves strong visual and quantitative performance.

### Weaknesses
1. The paper employs pre-trained text-to-image diffusion models to iteratively refine reconstructed results. However, diffusion models are inherently stochastic, which means outputs can vary across different (or even identical) viewpoints. This randomness may lead to inconsistencies across views. Although the authors propose generating multiple samples to mitigate this issue, such an approach may cause over-smoothing, as the results could converge toward the mean of the distribution. A deeper discussion on this trade-off would strengthen the paper.
2. The authors use a prompt-to-prompt editing technique to guide image refinement by emphasizing geometric distortion and structural clarity between source and target images. It would be beneficial to further investigate the prompt design and analyze how different prompts influence refinement quality. Additionally, the authors could consider comparing this approach with noising-denoising image refinement pipelines explored in prior works [1].
3. To more clearly illustrate the effectiveness of the proposed refinement process, it would be helpful to include qualitative samples from multiple refinement stages, showing progressive improvements.
4. The paper discusses rendering efficiency but does not address training efficiency. Given that the proposed method involves a complex, multi-stage refinement pipeline, it would be valuable to provide an analysis or discussion of its training efficiency.

[1] Tang, Jiaxiang, et al. Dreamgaussian: Generative gaussian splatting for efficient 3d content creation. ICLR 2024.

### Questions
Please refer to weaknesses.

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
3

### Summary
The paper introduces Skyfall-GS, a framework for synthesizing immersive, block-scale 3D urban scenes from multi-view satellite imagery. It operates in two stages: Initial reconstruction using 3D Gaussian Splatting with appearance modeling, pseudo-camera depth supervision, and opacity regularization to build coarse geometry from satellites; Synthesis via curriculum-driven Iterative Dataset Update, leveraging pre-trained text-to-image diffusion models (e.g., FLUX.1) to iteratively refine renders from high to low elevation angles, filling occlusions (e.g., facades) and enhancing geometric sharpness and texture realism. No additional 3D annotations or street-level data are required.

### Strengths
1. Skyfall-GS demonstrates originality through a creative recombination of established tools in a novel domain.
2. The curriculum-driven Iterative Dataset Update (IDU) is a fresh problem formulation: treating 3DGS novel-view renders as noisy intermediates in a denoising diffusion process.
3. The reconstruction stage introduces targeted regularizers: entropy-based opacity pruning and Pearson-correlation depth supervision, which are directly address satellite-specific artifacts (floaters, limited parallax) with minimal overhead.

### Weaknesses
1. Some typo errors like 266L: this approach significant improves the visual quality ->this approach significantly improves the visual quality
2. Initial camera parameter approximation (RPC to perspective) may introduce errors in complex terrains, which needs some clarification.
3. Lack of handling of dynamic elements (e.g., vehicles, pedestrians); multi-date appearance modeling may leave transient artifacts.

### Questions
1. What is the exact spatial extent (m²) of evaluated scenes? Have you tested Skyfall-GS on contiguous multi-block areas (e.g., 1km×1km)?
2. In multi-date satellite sets, how are moving cars/trees handled? Can you show how to assess temporal stability
3. Is the elevation sequence {Ei} fixed or could it be conditioned on per-scene metrics

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper presents Skyfall-GS, a framework for synthesizing explorable 3D urban scenes directly from multi-view satellite imagery, without relying on ground-level data or 3D supervision. The pipeline has two stages: (1) initial 3D Gaussian Splatting (3DGS) reconstruction with appearance modeling, opacity regularization, and pseudo-depth supervision to address limited satellite parallax; and (2) a curriculum-based iterative dataset update (IDU) stage that refines renders using prompt-to-prompt diffusion editing. The model gradually transitions camera viewpoints from aerial to ground-level to recover occluded facades. Experiments on DFC2019 and GoogleEarth datasets show improvements over Sat-NeRF, CityDreamer, and GaussianCity in perceptual metrics (FID-CLIP, CMMD) and in user studies, with real-time rendering capability.

### Strengths
- The combination of 3DGS reconstruction and diffusion-based refinement is carefully engineered. The inclusion of appearance embeddings and pseudo-depth supervision for satellite data addresses unique challenges like illumination change and weak parallax.
- The elevation-progressive IDU strategy is original and intuitively sound, improving geometric coherence as camera angles descend from satellite to near-ground viewpoints.
- Quantitative results on two datasets, perceptual and pixel-based metrics, and two large user studies provide convincing validation. Ablations on all main components (opacity regularization, depth supervision, multi-sample diffusion, curriculum schedule) are thorough.

### Weaknesses
- While technically solid, the framework mainly integrates existing elements (3DGS + diffusion + curriculum learning). The novelty lies in applying these to satellite imagery rather than in introducing a fundamentally new algorithmic idea.
- Dependence on heavy computation. The iterative refinement process and multiple diffusion samples per view make the approach resource-intensive, somewhat at odds with the claimed scalability.

### Questions
- Could the authors quantify how the pseudo-depth supervision improves metric depth alignment (e.g., MAE vs. LiDAR) beyond perceptual scores?
- How sensitive is the refinement to the chosen text prompts? Would generic prompts (e.g., “clear buildings, realistic textures”) yield similar results?
- What are the memory and time costs of one complete IDU cycle relative to baseline 3DGS training?

### Soundness
3

### Presentation
3

### Contribution
3
