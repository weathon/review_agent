# H$^3$DP: Triply‑Hierarchical Diffusion Policy for Visuomotor Learning

- Avg Score: 6.67
- Decision: Accept (Poster)
- Scores: 8, 6, 6, 6, 6, 8

## Abstract
Visuomotor policy learning has witnessed substantial progress in robotic manipulation, with recent approaches predominantly relying on generative models to model the action distribution. However, these methods often overlook the critical coupling between visual perception and action prediction. In this work, we introduce Triply-Hierarchical Diffusion Policy (H$^3$DP), a novel visuomotor learning framework that explicitly incorporates hierarchical structures to strengthen the integration between visual features and action generation. H$^3$DP contains $\mathbf{3}$ levels of hierarchy: (1) depth-aware input layering that organizes RGB-D observations based on depth information; (2) multi-scale visual representations that encode semantic features at varying levels of granularity; and (3) a hierarchically conditioned diffusion process that aligns the generation of coarse-to-fine actions with corresponding visual features. Extensive experiments demonstrate that H$^3$DP yields a $+ \mathbf{27.5}$% average relative improvement over baselines across $\mathbf{44}$ simulation tasks and achieves superior performance in $\mathbf{4}$ challenging bimanual real-world manipulation tasks. Project Page: https://h3-dp.github.io/.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This work proposes a visual Imitation Learning (IL) method built on Diffusion Policy (DP) which incorporates additional structure in the perception modules and aligns this structure with the action diffusion process, leveraging depth input in addition to RGB. The method incorporates 3 forms of structure: (1) Scene decomposition based on depth. (2) Coarse-to-fine-grained visual features are encoded separately using multi-scale feature maps. (3) Coarse-to-fine-grained visual features are aligned with coarse-to-fine-grained generated action details by conditioning the action diffusion process with increasingly fine feature maps with the denoising timestep progression. The method is thoroughly evaluated on various simulated and real-world robotic manipulation tasks, comparing with both standard and 3D point-cloud conditioned DPs, clearly demonstrating the efficacy of the approach.

### Strengths
**Overview**
- Well written, clear and easy to follow.
- The method seems novel.
- Interesting use of depth for scene decomposition.
- Interesting alignment of visual feature granularity with action generation detail.
- Evaluated on diverse simulated environments.
- Evaluated generalization scenarios.
- Real-world experiments.
- Thorough method analysis and ablation study.
- Website with demonstration videos.
- Detailed appendix.

### Weaknesses
**Overview**
- Some aspects and design choices of the method are not thoroughly discussed or motivated.
- No limitations section.

**Codebook**

Why did the authors choose to have a separate codebook for each image layer?

Why are the codebooks shared across the depth layers and not the visual frequency? My intuition is that features in the same scale have more in common than features in the same depth image segment.

I believe the reader would benefit from a discussion about the motivation for this design choice and/or an ablation study. If this is present in the appendix, please refer to it from the main text.

**Diffusion Policy Conditioning**

Lines 230-232 state that conditioning on f_K during training implicitly optimizes all f_k for k=1,...,K. Why is this the case? Can the reader understand this without reading the appendix?

Why does this design choice promote consistency of representations at each scale? Can’t it just disregard certain scales?

There is a clear discrepancy between the diffusion condition during training and inference. Why does this work? Have you attempted training with the hierarchical conditioning mechanism used for action generation?

**Missing Architectural Details**

There are some architectural details that I find missing in the paper:
- What architecture is used for the diffusion policy?
- How exactly is the diffusion model conditioned on the different feature maps?
- Can the authors give some basic details about the image encoder architecture other than citing VQGAN so the paper is more self-contained?

**Limitations**

What are the limitations and underlying assumptions of your method? These are not discussed in the paper. In my opinion, acknowledging limitations and assumptions are very important for the reader to acquire a deeper understanding of the method and how various aspects of it can be applied to other methods in future work.

**Misc**
- Although it makes for a nice name, the depth-aware layering does not exactly constitute a hierarchy, I would say it is more a decomposition. It seems that it is treated as a set (i.e., there is no ordering) in the later stages of the pipeline.
- Standard deviations across the evaluated seeds should be reported alongside the mean.

### Questions
**Questions**

- Line 196 refers to the “original features f_m”, what are those features? I believe I have an understanding of what the authors are referring to after reading the appendix but it should be clear from the main text.
- What is the purpose of line 10 in Algorithm 1?
- Section 4.1.3. The authors claim that the coarse-to-fine frequency progression in action generation is inherent to the diffusion process. It is therefore not a trait of your specific method, am I correct? If so, what is the purpose of this section? If your method somehow facilitates this, why is there not a comparison to the spectral decomposition in a standard DP? 
- Lines 473-475: do you apply the noise after training? How about during training? Is the noise not meant to simulate real depth input which is also noisy during training?
- Can you provide more visualizations of the depth segmentation on the different environments? Would you say the resulting representation roughly results in an agent-object-bg segmentation in the evaluated envs? Is it an underlying assumption of your method that depth corresponds to agent-object-background decomposition?
- How are the real world demonstrations collected? Are there any special techniques in the data collection that facilitate the generalization behavior observed in the website videos?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper, $\text{H}^3\text{DP}$, proposes several modifications to diffusion policy. The authors propose binning RGBD pixels by its depth value into 3-4 bins. Then, for each layer, they train a separate VQGAN-like encoder for multi-scale visual representation. These representations are fed into a diffusion policy head which sequentially uses coarse to fine feature maps for action denoising. Experiments on real and sim show that the proposed method works better than prior state of the art, and ablations show the necessity of each component.

### Strengths
- The ideas of binning depth, hierarchical feature maps, and coarse-to-fine denoising are well-motivated.
- Various experiments on sim and real environments show that the proposed method outperforms prior state of the art.
- Extensive ablations in the main text and appendix validate the necessity of each component.
- Minimal inference speed overhead compared to vanilla formulation.

### Weaknesses
- Complicated setup. While the reviewer is impressed by the performance improvement, the proposed modifications are quite involved. Specifically, the reviewer would like to see an ablation where the downstream policy only denoises on the finest feature map.
- Lack of baselines. As the paper largely deals with improved representation and policy learning, it would be nice to see comparisons with recent 3D-aware representation and policy learning works. Right now the paper only compares with DP and DP3.

### Questions
- Clarification: is temporal ensembling and p-masking on proprio also done for the baselines for the real robot experiments?
- Baselines: it would be nice to see comparisons with other works on visual representation like 3D Diffuser Actor.
- Ablation: how well does the method perform if you only train the downstream policy with the finest feature map?

### Soundness
4

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
This paper introduces H3DP, a triply-hierarchical diffusion policy that seamlessly couples visual perception with action generation in visuomotor learning. The framework systematically incorporates three key innovations: depth-aware input layering for foreground/background disentanglement, multi-scale visual representations for granular spatial semantics, and a coarse-to-fine diffusion process conditioned on these hierarchies. Extensive evaluations on 44 simulated tasks and 4 challenging real-world bimanual scenarios demonstrate that H3DP consistently outperforms strong baselines (e.g., DP and DP3), achieving performance gains of +27.5% and +72.4% respectively, while maintaining efficient inference.

### Strengths
H3DP offers a genuinely original reframing of visuomotor diffusion policies by hierarchizing the entire pipeline—from depth-layered inputs to coarse-to-fine denoising—a conceptual contribution not previously articulated. This idea is validated through massive, well-documented experiments (44 simulated and 4 real-robot tasks), which demonstrate clear and substantial improvements over strong baselines without requiring extra sensors or manual preprocessing. With the code set to be open-sourced, the community can immediately build upon this work. Although the writing is at times dense with numerical details and the notation could be streamlined, the core contribution remains easily accessible through a single overview figure and is poised to influence not only robotic manipulation but also broader domains employing diffusion-based decision-making.

### Weaknesses
The performance gain attributed to depth-layer processing appears confounded by increased network capacity. While Table 3 reports an average improvement of ≈ +14% with depth layering, the baseline model ("w/o depth layering") uses a smaller, single-stream encoder. In computer vision, such a design would typically prompt the question: is the observed improvement truly due to the proposed method, or simply a result of additional parameters?

Furthermore, the paper claims "superior generalization," yet all images were captured under identical lab conditions—consistent lighting and a fixed tripod-mounted camera. A rigorous evaluation would require testing under at least one domain shift, such as variation in illumination, camera viewpoint, or background texture.

Lastly, all results are presented as "mean ± std over 3 seeds." Recent methodological standards in machine learning have raised concerns that fewer than 5 seeds, without supporting confidence intervals or p-values, may provide insufficient statistical evidence.

### Questions
Q1. Capacity-controlled ablation
In Table 3, the reported ≈14% performance gain from depth layering is based on a comparison with a smaller, single-stream encoder. To more rigorously isolate the effect of the proposed architecture, could you conduct a capacity-matched ablation by widening the RGB-only baseline to match the FLOPs or parameter count of your model (e.g., by increasing channel width by 1.5×)? If the performance lift then diminishes to, say, below 7%, how would you substantiate the claim that the improvement stems from the depth-layered design itself, rather than merely increased model capacity?

Q2. Out-of-distribution robustness
The claim of "superior generalization" would be strengthened by evaluating under modest domain shifts. Could you provide a simple robustness curve—for instance, by taking an existing H3DP checkpoint and testing it under altered conditions, such as with 50% dimmed lighting or a handheld iPhone RGB-D stream (with ±10 cm vertical motion)? A simple bar plot showing the performance drop would be sufficient. If accuracy declines by more than 20%, openly reporting and explaining this result would be more informative than an unqualified generalization claim.

Q3. Statistical significance evaluation
All results are currently reported as “mean ± std over 3 seeds.” For your two key real-world tasks (PJ and ST), could you supplement this with either (a) p-values from a bootstrap test across at least 5 seeds, or (b) 95% confidence intervals? In recent CV meta-reviews, results based on fewer than 5 seeds without confidence intervals have been noted as providing insufficient statistical evidence. Providing such information would help preempt concerns regarding result stability.

### Soundness
2

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
5

### Summary
This paper proposes Triply-Hierarchical Diffusion Policy (H³DP), a variant of Diffusion Policy designed to improve the coupling between perception and action in robotic manipulation. H³DP introduces three hierarchical components: depth-aware input layering that organizes RGB-D observations by depth, multi-scale visual representations that encode semantic features at varying granularities, and a hierarchically conditioned diffusion process that generates actions in a coarse-to-fine manner aligned with corresponding visual cues. Experiments show that H³DP achieves substantial performance gains over baselines across multiple simulated manipulation tasks and demonstrates strong results in challenging bimanual real-world settings.

### Strengths
1. The paper presents extensive and well-designed experiments, including both simulation and real-world evaluations. The tasks cover a diverse range of object types, manipulation actions, and embodiments, such as pick-and-place, articulated/deformable object manipulation—using both parallel grippers and dexterous hands. The ablation studies thoroughly analyze the contribution of each hierarchical component, examine robustness to noisy depth inputs, and compare different vision encoders, resulting in a comprehensive experimental analysis.

2. The hierarchical design that jointly structures visual encoding and action generation is clear, well-motivated, and conceptually sound, and the experimental results demonstrate consistent and significant improvements over strong baselines.

3. The figures and visualizations are clear, informative, and effectively convey the hierarchical framework and experimental results.

### Weaknesses
1. The paper lacks a comparison of segmentation-based layering as an alternative to the proposed depth-aware layering. While the authors highlight the importance of point cloud segmentation for DP3, they do not explore or justify why semantic segmentation was not considered. Specifically, we can use a tool such as Grounded-SAM to segment objects and backgrounds based on semantic grounding. In some scenarios, semantically similar target objects may appear at different depths but play equivalent roles in the task, suggesting that semantic segmentation layering could potentially yield more meaningful representations than purely depth-based ones.

2. Hierarchical actions remain limited. The current hierarchy in the diffusion process focuses on coarse-to-fine action refinement within a short temporal window, but it does not extend to high-level task decomposition or subtask sequencing. It would be interesting to explore whether the framework could be extended to generate hierarchical action chunks. For example, first predicting subtask-level actions and then refining them into low-level motor commands.

3. Minor writing issues: Some abbreviations (e.g., MW, MS, RT in Tables 3 and 4) are not defined when first introduced. Clarifying these at the start of the experimental section would improve readability and presentation quality.

### Questions
Could the authors provide additional analysis or experiments to address the points raised in the weakness section? In particular, it would be helpful to (1) compare depth-aware layering with a segmentation-based alternative and (2) explore extensions of the current hierarchical diffusion process toward higher-level action decomposition across longer horizons.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
**Motivation**

Previous methods have focused on refining inputs, perception, and action separately. In contrast, humans process information hierarchically at all stages.

**Proposal**

The authors present H3DP – Triply‑Hierarchical Diffusion Policy. An agent with hierarchical processing at three stages of action prediction: inputs, representation and the action prediction.

The agent uses RGB-D inputs, where the images are segmented into multiple layers using the depth maps.

For representation, multiple scaled input images are used to provide information from all levels of granularity.

For action, the diffusion is conditioned progressively from coarse to fine features.

**Results**

The agent is evaluated in sim benchmarks (44 tasks in 5 sims) and real world tasks (4 bimanual manipulation).

The agent performs well in the simulation tasks (75.6%), outperforming the next performing baseline (DP3) by 27.5% (relative) in sim tasks.

The agent also performs well in real world tasks (57.75% overall, +72.4% relative to next baseline).

Additionally, the method performs better than baselines with 20% data.

### Strengths
- Good integration of multiple complex architectures: multi-layered depth separated visual inputs, multi-scale image features, and scale-specific feature conditioned diffusion policy.
- Method works in the real world under cluttered conditions (move the bottle in the fridge, pour water and a scoop of powder, sweep trash into the dustbin using a broom and dustpan, and move the bottle to the designated spot).
- Extensive evaluation and ablation study.

### Weaknesses
- The evaluation uses only 3 seeds per task. It should ideally be at least 5.
- In some environments, the top 5 success rates are used, and only the top success rate in some (with only 20 episodes in each). This makes it difficult to judge the variance in performance. The number of episodes should be increased (at least 50), and maybe a plot with evaluation scores averaged across all seeds against training step should be reported for a few tasks.
- While the method is tested in general/standard settings, if possible, validation on tasks where the contributions must be required to complete would be great. For example, testing on an occlusion-heavy task, where the depth-wise layering is needed for success.
- The paper does not explicitly mention failure modes or scalability issues. Given the number of hyperparameters (different $\beta$s, learning‑rate schedules, codebook sizes per simulator), a brief discussion on sensitivity and potential deployment challenges would be valuable.

### Questions
- How susceptible is the performance to the hyperparameter choice?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 6

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes H3DP (Triply-Hierarchical Diffusion Policy), a hierarchical generative framework for visuomotor policy learning in robotic manipulation. The method integrates three levels of hierarchy: depth-aware input layering that partitions RGB-D observations, multi-scale visual representations that preserve global-to-local semantics, and a coarse-to-fine action generation diffusion process. Extensive experiments are conducted across 44 simulated tasks (5 benchmarks) and 4 challenging real-world bimanual manipulation tasks, demonstrating significant performance gains over major baselines and ablation variants.

### Strengths
1.  **Ablation and analysis depth:**  The authors ablate each hierarchical design aspect (see Tables 13–15), show model behaviors under low-data and noisy depth regimes (Tables 7, 22), and compare to closely-related algorithmic variants (DP3, CARP). The connection between model robustness and architectural choices (quantization, depth-layering) is examined through concrete experiments and Figure 7.
2.  **Efficiency:**  H3DP achieves results rivaling or surpassing point-cloud and multi-view methods but with only single-view RGB-D as input, reducing computational and engineering demands. Model size and inference speed are competitive (Tables 20–21).
3. **Empirical evaluation:**  The paper covers a broad suite of tasks: 44 simulated and 4 real-world challenges, while using established benchmarks (MetaWorld, ManiSkill, Adroit, etc.) and challenging bimanual/long-horizon settings. The use of both instance and spatial generalization metrics is appreciated. Table 1 and Figure 1 provide clear, accessible summaries of the breadth and magnitude of gains.

### Weaknesses
1.  **Incomplete baselines:** I think the main weakness lies in the incomplete set of baselines used for comparison. The model is primarily evaluated against DDPM-based approaches, making it unclear how well it performs relative to flow-based [1, 2, 3] or equivariance-based designs [4, 5, 6].

[1] Zhang et al., "FlowPolicy: Enabling Fast and Robust 3D Flow-based Policy via Consistency Flow Matching for Robot Manipulation". 2024. https://arxiv.org/abs/2412.04987

[2] Chisari et al., "Learning Robotic Manipulation Policies from Point Clouds with Conditional Flow Matching". 2024. https://arxiv.org/abs/2409.07343

[3] Ding et al., "Fast and Robust Visuomotor Riemannian Flow Matching Policy". 2025. https://doi.org/10.1109/TRO.2025.3601293

[4] Wang et al., "Equivariant Diffusion Policy". 2024. https://openreview.net/forum?id=wD2kUVLT1g

[5] Yang et al., "EquiBot: SIM(3)-Equivariant Diffusion Policy for Generalizable and Data Efficient Learning". 2024. https://arxiv.org/abs/2407.01479

[6] Tie et al., "ET-SEED: Efficient Trajectory-Level SE(3) Equivariant Diffusion Policy". 2024. https://arxiv.org/abs/2411.03990

### Questions
1. See weaknesses.
2. How were the depth maps obtained? It would be interesting to evaluate how well the method performs when the depth maps are generated by a neural model from RGB images.

### Soundness
3

### Presentation
3

### Contribution
3
