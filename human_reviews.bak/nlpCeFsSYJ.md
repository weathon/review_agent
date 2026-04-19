# DOME: Taming Diffusion Model into High-Fidelity Controllable Occupancy World Model

- Decision: Reject
- Scores: 5, 5, 6, 6

## Abstract
We propose DOME, a diffusion-based world model that predicts future occupancy frames based on past occupancy observations. The ability of this world model to capture the evolution of the environment is crucial for planning in autonomous driving. Compared to 2D video-based world models, the occupancy world model utilizes a native 3D representation, which features easily obtainable annotations and is modality-agnostic. This flexibility has the potential to facilitate the development of more advanced world models. Existing occupancy world models either suffer from detail loss due to discrete tokenization or rely on simplistic diffusion architectures, leading to inefficiencies and difficulties in predicting future occupancy with controllability. Our DOME exhibits two key features: (1) High-Fidelity and Long-Duration Generation. We adopt a spatial-temporal diffusion transformer to predict future occupancy frames based on historical context. This architecture efficiently captures spatial-temporal information, enabling high-fidelity details and the ability to generate predictions over long durations. (2)Fine-grained Controllability. We address the challenge of controllability in predictions by introducing a trajectory resampling method, which significantly enhances the model’s ability to generate controlled predictions. Extensive experiments on the widely used nuScenes dataset demonstrate that our method surpasses existing baselines in both qualitative and quantitative evaluations, establishing a new state-of-the-art performance on nuScenes. Specifically, our approach surpasses the baseline by 10.5% in mIoU and 21.2% in IoU for occupancy reconstruction, and by 36.0% in mIoU and 24.6% in IoU for 4D occupancy forecasting.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper introduces DOME, a novel diffusion-based world model for autonomous driving that predicts future occupancy frames from past observations. DOME features high-fidelity, long-duration generation, and fine-grained controllability through a spatial-temporal diffusion transformer and a trajectory resampling method. Experiments on the nuScenes dataset demonstrate DOME's state-of-the-art performance, surpassing baselines by significant margins in both occupancy reconstruction and forecasting. The model's ability to generate long-duration sequences and respond to trajectory conditions could enhance end-to-end planning in autonomous driving.

### Strengths
1. DOME utilizes a spatial-temporal diffusion transformer that efficiently captures both spatial and temporal information, enabling the generation of detailed, long-lasting occupancy predictions.
2. DOME enhances the model's ability to generate controlled predictions, which is an important property for world models. 
3. Extensive experiments on the nuScenes dataset show that DOME surpasses existing baselines by a considerable margin in both qualitative and quantitative evaluations for occupancy reconstruction and forecasting.

### Weaknesses
1. DOME's sophisticated architecture, while effective, might lead to increased complexity in implementation and maintenance. This complexity could make it harder for other researchers to reproduce the results or integrate the model into their own systems without a deep understanding of the underlying mechanisms.
2. The control of directions is still not accurate. From the visualizations, the ego car just slightly turns left or right, given the high-level command. Also, the generated scenes (e.g., the buildings and trees) are different for the same input scene with different trajectory conditions. I think a good world model should be able to generate consistent results for different trajectory conditions. 
3. The authors do not compare their methods with OccSora (visualizations), which can also achieve long-term and trajectory-conditioned generation.
4. Despite the trajectory resampling method aimed at enhancing diversity, the paper does not extensively discuss the potential for overfitting, especially given the model's complexity. More analysis on how the model performs on unseen data would strengthen the paper's claims.

### Questions
Can you provide a visualization comparison with OccSora about trajectory conditions?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper proposes a diffusion-based world model designed to predict future occupancy frames based on past occupancy observations. The authors introduce a spatial-temporal diffusion transformer, which leverages historical context to forecast future occupancy frames, effectively capturing spatiotemporal information for high-fidelity detail and enabling long-duration predictions. Additionally, the authors present a trajectory resampling method to address controllability challenges in prediction, significantly enhancing the model's ability to generate controlled predictions. Through experiments on the nuScenes dataset, the authors demonstrate that their approach outperforms existing baselines in both qualitative and quantitative evaluations, establishing new state-of-the-art performance on the nuScenes dataset. Specifically, their method surpasses the baseline by 10.5% in mIoU and 21.2% in IoU for occupancy reconstruction, and by 36.0% in mIoU and 24.6% in IoU for 4D occupancy prediction.

### Strengths
1. The authors propose a spatial-temporal interactive VQ encoder that facilitates spatiotemporal information exchange, contributing to higher-accuracy predictive generation.
2. They introduce a trajectory resampling method that reconstructs trajectories using the A* algorithm and designs possible scenarios based on a BEV map, improving the relationship between control and scene.
3. Experiments on the nuScenes dataset show that their approach achieves superior performance in both occupancy reconstruction mIoU and 4D occupancy prediction compared to existing baselines.

### Weaknesses
1. The spatial-temporal diffusion transformer lacks strong innovation; leveraging spatial-temporal information fusion and cross-attention is common in existing works, making this a less distinctive contribution.
2. The proposed trajectory resampling method, which reconstructs possible scenes based on a BEV map, may not adequately address dataset imbalance and limited diversity. The dataset used is an autonomous driving dataset with ego-vehicle movement, not a static city-scale occupancy dataset, which limits the effectiveness of trajectory resampling in addressing these issues.
3. The experiments for occupancy reconstruction and 4D occupancy prediction lack fairness in comparison. The reconstruction phase includes multi-frame inputs and spatio-temporal interaction, and the 4D occupancy prediction also leverages training parameters from the reconstruction stage. The authors should provide a fair comparison with other multi-frame input methods in the experimental section.

### Questions
1. In Figure 4(a), the authors highlight the limited trajectory diversity in the original dataset, while Figure 2 in OccLLaMA [1] demonstrates a more diverse range of trajectories, so whether the dataset is indeed as sparse as described in the paper. 

2. The authors claim to achieve occupancy predictions under different trajectory controls;   however, the visualized results appear somewhat limited, as the vehicle's movement along the trajectory is mini  Could the authors present a 32-second controlled trajectory prediction and provide a clearer visualization of the relationship between trajectory control and the scene?

3. How does the author solve the problem of dynamic objects in the reconstruction of possible scenes by using BEV map in trajectory resamping? The nuScenes data set will move with the self-driving car, and the surrounding vehicles will also move. It seems difficult to make a reasonable dynamic scene by changing the track of a self-driving vehicle (i.e. maintaining consistency between multiple different vehicle resampling tracks and their respective Occupancy dynamic scenes).

[1] Wei, Julong, et al. "OccLLaMA: An Occupancy-Language-Action Generative World Model for Autonomous Driving." arXiv preprint arXiv:2409.03272 (2024).

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces DOME, a diffusion-based world model for predicting future occupancy frames from past observations. It demonstrates high-fidelity generation, accurately predicting future changes in occupancy space, and can generate long-duration occupancy sequences that are twice as extensive as those produced by previous methods.

### Strengths
The model offers two notable strengths: 1) High-Fidelity and Long-Duration Generation: DOME utilizes a spatial-temporal diffusion transformer to effectively leverage historical occupancy data. This architecture captures detailed spatial-temporal dependencies, enabling the generation of high-fidelity predictions across extended timeframes. 2) Fine-Grained Controllability: To tackle the challenge of controlling predictive outputs, DOME incorporates a trajectory resampling method. This enhancement significantly improves the model’s ability to produce controlled and precise predictions, marking a key advancement in diffusion-based modeling. Together, these features establish DOME as a valuable contribution to predictive world modeling, especially for applications that demand both high-detail and controllable long-term forecasts.

### Weaknesses
1. In Table 1, could the authors clarify why DOME performs less effectively in the "Others" category? This clarification is important, as it relates to the zero-shot capability of the approach and may highlight limitations in handling diverse or unseen scenarios. 
2. Additionally, could the authors provide a comparison of model parameters and computation costs relative to other approaches? This information would help assess DOME's efficiency and scalability in practical applications.

### Questions
It would be helpful if the authors provided clear demonstrations with corresponding annotations for each approach. This would enhance the figure’s readability, making it easier for readers to compare and understand the differences between methods.

====================================

Thank you to the authors for the rebuttal. The results look acceptable, so I will keep my score unchanged.

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
The authors proposed a pipeline to predict future occupancy frames conditioned on historical occupancy observations and trajectory control. The approach entails Occ-VAE and a spatial-temperal diffusion with trajectory encoder and a novel resampling technique.

### Strengths
1. The paper is well organized and cleanly presented.
2. Thorough ablations and good experiment results are demonstrated.
3. The trajectory resampling technique for data augmentation is considered novel, allowing for good performance.

### Weaknesses
Please see questions.

### Questions
1. The authors mentioned around line 123 in the related work section that "BEV methods struggle to convey detailed 3D information due to their top-down projection." However, the authors' approach is still operating in "transform[ed] ... Bird’s Eye View (BEV) style tensor" (line 199). How would the BEV style feature be better than other approaches in maintaining "detailed 3D information" during VAE latent space compression?
2. Section 3.1 mentions the "class embedding" that needs to be learned for OCC-VAE, without much detailed discussion on its purpose and downstream impacts. Would this be, for example, a potential bottolneck for making the model more generalizable?
3. It seems like that, in Tab. 3, the trajectory resampling technique allows for a rather reasonable improvement on the mIoU and IoU metric. Do the authors think of the technique as the central piece of this paper's novelty that allows for SOTA performance? If similar data augmentation is applied to OccWorld or OccLLaMA, would they also perform on par with DOME? (sorry I may not be super familiar with this specific line of work, and I am willing to improve my rating as needed)
4. Missing spaces in line 467 "further demonstrating that DOMEhas strong generalizability" and line 494 ": 4D occupancy forecasting performance.Avg. denotes..." I would suggest further proof reading.
5. Tab. 4 suggests that DOME allows for 32s rollout for long duration generation without explicitly discussing the potential accumulated error. What's preventing the model from generating a even longer rollout if error range is not considered (particularly given the autoregressive approach)?

### Soundness
3

### Presentation
3

### Contribution
3
