# ARGOS: Hierarchical Autoregressive Generation of Unbounded 3D Outdoor Scenes with High Fidelity and Spatial Control

- Decision: Reject
- Scores: 2, 6, 6, 4

## Abstract
We present ARGOS, a hierarchical autoregressive framework for generating unbounded 3D outdoor scenes with high fidelity and spatial control. Existing methods for large-scale 3D scene generation are limited by a fundamental trade-off between global consistency and fine geometric detail. While prior diffusion-based approaches struggle with long-range coherence, our framework resolves this tension by decoupling the challenge into two stages. First, a causal autoregressive model establishes a globally coherent layout by processing scene chunks in sequence. Second, a masked autoregressive model generates detailed local geometry conditioned on this global layout and its neighbors. These geometric latents are then decoded by our enhanced VAE to ensure high-fidelity reconstruction. To enable user control, we introduce an automated pipeline that extracts complex spatial relationships from scenes, producing a structured dataset that allows ARGOS to follow precise text-based commands. Comprehensive experiments demonstrate that ARGOS significantly outperforms existing methods in unconditional generation, achieving superior FPD and KPD metrics across multiple scales. For text-conditioned synthesis, our approach excels at generating coherent, large-scale scenes that precisely adhere to complex spatial instructions.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a hierarchical autoregressive model for 3D outdoor synthesis. Specifically, a causal autoregressive model is designed for globally coherent layout generation and a masked autoregressive model for conditional detailed local geometry generation. Experimental results have shown improved performance on FPD and KPD compared to existing methods.

### Strengths
+ The qualitative results and quantitative results shown are better than existing models for outdoor scene generation. 
+ Summary of related works are sufficient.

### Weaknesses
The novelty of the paper overall is not significant. The paper seems to be a direct replacement of diffusion model in NuiScene with autoregressive model. In details: 
1. The proposed VAE compared to the VAE in NuiScene is simply an additional dense surface sampling for high fidelity geometry. 
2. Unclear motivation for the design of $\textbf{L}_{height}$. 
3. The motivation of using an autoregressive model rather than diffusion model is unclear. 
4. The motivation of the hierarchical AR model rather than a direct next token prediction of 3D grid. 
5. Missing ablation study on visual results. 
6. Unclear explanation of 13 scene data for training. What advantage does 13 scenes brings?
7. Missing ablation study on $\lambda_{coh}$ and $\lambda_{dpo}$. 
8. There is only one example (Figure 5) that shows the performance compared to existing method. More results should be provided to validate the design of the model. Additionally, the visual results does not show improved results than NuiScene. 
9. Diversity was not shown.

### Questions
See Weakness

### Soundness
2

### Presentation
1

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
ARGOS is introduced as a hierarchical autoregressive generation framework that excels at large-scale outdoor scene generation. It also supports text-based conditional generation. Based on previous framework from NuiScene, it introduces a two-stage generation process: a global layout generation stage that leverages causal autoregressive modeling to create coherent layouts, followed by a second autoregressive modeling that fills in fine details at each local chunk. ARGOS also leverages a new VAE design for better local detail reconstruction compared to NuiScene. The authors claim ARGOS achieves superior performance in unconditional generation and shows promising results in text-conditional generation.

### Strengths
1. Introducing a hierarchical autoregressive framework for large-scale outdoor scene generation is intuitive and well-motivated. It brings solid improvements over large scene coherence compared to prior work with diffusion outpainting.
2. New VAE design with mesh voxelization and salient geometry guidance brings better resolution and detail reconstruction.
3. Solid empirical comparison with NuiScene, retraining NuiScene with larger receptive field adds credibility to the comparison.

### Weaknesses
- **Major:**
1. The 13-scene dataset processed by the authors is sufficiently large for training and in-context evaluation. However, compared to the full NuiScene43 dataset and the original Objaverse data, the lack of diversity in only 13 scenes may limit the generalizability to other outdoor scenes. I did not see any evaluation of ARGOS on data outside of the 13-scene dataset.
2. The evaluation is primarily constrained to authors's own data pipeline, this does not tell the full story if ARGOS can still work significantly better than NuiScene using arbitrary reconstruction that wasn’t meshed and chunked this way.
3. The discussion of Related Work is limited. The authors claim there's limited work for structurally controllable outdoor scene generation while many works have explored this area, such as [1, 2, 3, 4, 5]
4. Following that, text-based signal is fundamentally unsuitable for complex scene control. Linguistic prompts are inherently ambiguous and struggle to provide precise spatial signal needed for complex scene generation. Example prompts in the manuscript such as “highest region in the top-right…” works because the authors derives a data-specific structure from their data pipeline. This strategy may not work for all scenarios. The authors should discuss this limitation more thoroughly.

- **Minor:**
1. I would like to see more qualitative comparisons beyond the scene style showed in the current manuscript. It would also be nice if the authors include their full dataset visualizations in the appendix for better understanding of the data distribution.
2. Label text color in Fig 2's global and neighbors latent blocks are hard to read against the background. Please change to a darker color.
3. Add a short reproducibility statement about code, model and dataset release plan.
4: The authors claim ARGOS has superior performance in unconditional generation, while it only significantly out performs NuiScene in larger scales, please change the wording to "ARGOS has superior performance over previous methods at large scale scene generation".

[1] UrbanWorld: An Urban World Model for 3D City Generation 

[2] Text2LiDAR: Text-guided LiDAR Point Cloud Generation via Equirectangular Transformer 

[3] Controllable 3D Outdoor Scene Generation via Scene Graphs 

[4] InfiniCube: Unbounded and Controllable Dynamic 3D Driving Scene Generation with World-Guided Video Models 

[5] SSEditor: Controllable Mask-to-Scene Generation with Diffusion Model

### Questions
Refer to Weaknesses.

### Soundness
2

### Presentation
4

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
The paper proposes a hierarchical autoregressive pipeline for large-scale scene synthesis. A global causal model generates chunk-level layout in Z-order, and a local masked model with a diffusion head fills fine geometry per chunk. For controllability, the authors develop an automated text description pipeline and utilize CLIP text embeddings for conditioning.

### Strengths
1. The two-stage design, containing global causal over chunks and local masked autoregressive generation, directly targets coherence across large grids. DPO loss is introduced in the paper, which is nice to see, and helps to stabilize large-scale synthesis and diversity. 

2. The paper is clearly written and easy to follow.

### Weaknesses
1.  3D chunk VAE is not technically more novel or has more differences than NuiScene.

2. Most comparisons are against NuiScene. Given the claim of outperforming “existing methods,” the paper should also compare with other recent large-scene or chunked methods.

3. The dataset is a bit narrow, as it only contains 13 scenes. This has an overfit risk, especially for the text-to-scene problem.

4. The paper details training setups, but does not systematically report inference latency, memory, or throughput for large grids.

5. L134-135 names two techniques, FRC and SGG, and refers to Figure 1. Yet, they are not drawn to it. The current way of designing the pipeline figure is no different from NuiScene.

6. Figure 2 needs a more obvious Z-order direction to be clearer.

7. Some work on controllable scene generation is missing in L93-94, e.g., CommonScenes, EchoScene, etc.

### Questions
Please see the weakness above.

Why did the authors just choose the adjacent top chunk and the left chunk for generating local details? How to define this window size? For a given chunk, conditioning can extend beyond the top and left neighbors to include diagonal context, such as the top-left and top-right chunks, for instance.

Speaking to this, why did the authors choose to use the z-order rather than the masked strategy for global chunk generation (see MaskGIT, for example)?

It would be better if the authors could demonstrate some texture baking for their examples.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors tackles the challenge of large scale 3d scene generation for outdoor. It developes on top of one prior work, NuiScene, with improvements on (1) input preprocessing procedure in terms of how to densely sample points on chunk mesh, (2) hierarchical generation in terms of global chunk embedding first followed with NuiScene pipeline), and (3) additional text related spatial control. Experiment-wise, they showcase their coarse-to-fine generation pipeline is helpful in the 3Dvector2set chunk features, and they showcase they can do spatial control over text prompt with qualitative results. Finally, they ablate over different generation  paradigm to showcase their procedure can bring the most fidelity 3d scene results.

### Strengths
1. One of the key message of this paper is hirerachical generation (coarse-to-fine) is helpful to bring global consistency to large-scale 3d scene generation. It verifies on top of NuiScene overall framework. It demonstrates sufficiently with quantitaive mesaurements (KPD, FPD) in Table 3. The larger, the more significant the difference is. I think this part is clear

2. The proposed contributions are mostly backuped up by qualitative and quantitative evidences, e.g., Coherency-Aware Regularization and Direct Preference Optimization. 

3. It is clear that their VAE reconstruction with denser and focused sampling are effective from both qualitative results (Fig. 3 and Table 2) 

4. Experiments protocol are detailed discussed with results detailed benchmarked.

### Weaknesses
1. Visuals and method descriptions are not clear in a few number of places. 
-- Between Line 146 and 148, "To address the limitation, we propose a novel pipeline that directly samples the chunk mesh from the
scene mesh." It brings up some novel pipeline, but it does not describe what exactly it is, and which part is the one the authors refer to as novel pipeline? 

-- In Figure 6, what does the highlighted region suppose to indicate? Is it bad or good? It lacks of connection between visuals and captions. 

-- In Table 5 and Line 418, I believe it should be FRC instead of RFC. 

-- What does Table 1 convey? Similarly, what is take-away from Table 4? The caption seems to be not helpful on analyzing its contents. 


2. For the global chunk embeddings generation, it still follows an iterative autoregressive process, i.e., progressive generation paradigm. For z_i^global, in practice, it does not sound feasible to incorporate all previous generated tokens. This part seems not trivial but details are missing. 

3. Coarse-to-fine generation has been verified to be helpful in existing 3d scene generation work, e.g., XCube. The novelty might be some concerns for acceptance. 


4. It is unclear the performance gap between NuiScene is mainly due to better VAE reconstruction or latter coarse-to-fine generation. If we compare the visual results between Nuiscene (16x160 and theirs, the qualitative results are hard to tell which is better. If the gap mostly lies in data sampling when preparing the features, it is a bit unclear how necessary we need to do the global chunk embedding first, and then detailed chunk + diffusion later. 

5. The qualitative results are not very convincing. All qualitative results are only given 1 example, which makes the conclusion questionable for generalization. For example, the control of contents are not very clear with only one text prompt example. 

6. The comparisons with key 3d scene gen baselines are missing. For example, Earthcraft (https://arxiv.org/abs/2507.16535). XCube (https://arxiv.org/pdf/2312.03806). LT3SD (https://openaccess.thecvf.com/content/CVPR2025/papers/Meng_LT3SD_Latent_Trees_for_3D_Scene_Diffusion_CVPR_2025_paper.pdf).

### Questions
1. It would be nice to have additional comparison on the same feature derived from the same process of input mesh. And then feed those to Nuiscene and the authors proposed generation pipeline. I think it could be helpful to reveal how much the hierarhical process really help compared with their key baseline. 

2. It would also be great to have additional comparison over other SOTA 3d scene baselines.

### Soundness
3

### Presentation
3

### Contribution
3
