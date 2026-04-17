# CoMo: Compositional Motion Customization  for Text-to-Video Generation

- Decision: Reject
- Scores: 2, 4, 6, 4

## Abstract
While recent text-to-video models excel at generating diverse scenes, they struggle with precise motion control, particularly for complex, multi-subject motions. Although methods for single-motion customization have been developed to address this gap, they fail in compositional scenarios due to two primary challenges:  motion-appearance entanglement and ineffective multi-motion blending. This paper introduces CoMo, a novel framework for $\textbf{compositional motion customization}$ in text-to-video generation, enabling the synthesis of multiple, distinct motions within a single video. CoMo addresses these issues through a two-phase approach. First, in the single-motion learning phase, a static-dynamic decoupled tuning paradigm disentangles motion from appearance to learn a motion-specific module. Second, in the multi-motion composition phase, a plug-and-play divide-and-merge strategy composes these learned motions without additional training by spatially isolating their influence during the denoising process. To facilitate research in this new domain, we also introduce a new benchmark and a novel evaluation metric designed to assess multi-motion fidelity and blending. Extensive experiments demonstrate that CoMo achieves state-of-the-art performance, significantly advancing the capabilities of controllable video generation.  Our project page is at \url{https://como6.github.io/}.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper focuses on motion customization from multiple reference videos. The authors propose a two-stage LoRA training methodology to decouple motion and appearance information from the reference frames. Subsequently, they employ a latent composition technique to generate a smooth latent representation under given motion conditions. To quantitatively evaluate multi-motion customization, a new benchmark and a corresponding evaluation metric are also introduced.

### Strengths
1. **Clarity of Presentation:** The description of the proposed method is clear and easy to follow.
    
2. **Quantitative Performance:** The paper demonstrates strong quantitative results in comparison to the baseline methods.

### Weaknesses
1. **Missing Key Information on a Core Contribution:** The paper claims the proposal of a new benchmark as one of its main contributions. However, crucial details regarding this benchmark are absent from both the main paper and the appendix, making it difficult to assess its validity and scope.
    
2. **Insufficient Experimental Comparisons:** In the evaluation of compositional motion customization, the paper only compares its method against VACE, which is a zero-shot approach. The experiments would be more comprehensive if they included comparisons with single-motion customization methods that use a simple linear merging of latent codes.

### Questions
1. **Object Position Discrepancy:** In Figure 2, there is a noticeable difference in the positions of the woman and the monkey between the results of "ours" and the other methods. Could you please elaborate on why linear merging and joint training appears to cause errors in object positioning?
    
2. **Impact of Latent Merging on Denoising:** In Section 3.2, the paper merges latent representations via a weighted sum. This operation can alter the variance of the resulting latent code compared to the original distribution. Given that diffusion models are highly sensitive to the latent distribution, could you provide an analysis of how the proposed merging technique affects the original denoising process?
    
3. **Generalization to Overlapping Bounding Boxes:** The proposed latent merging method relies on the bounding boxes of characters in the target video. How does this method perform in scenarios where bounding boxes overlap, for instance, when two characters walk past each other, crossing from one side of the frame to the other?
    
4. **Clarification of Motion Fidelity Metric:** Could author(s) please provide a detailed description of how "motion fidelity" is calculated? While Section 3.3 mentions multiple references for this metric, its specific implementation and formula are not detailed in the paper, which is essential for reproducibility.

5. **Construction of the Benchmark**: Author(s) should provide the detailed information about how the benchmark is constructed.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces CoMo, a compositional motion customization method for integrating multiple motions into video generation. The approach consists of two phases: a single-motion learning phase where each motion is customized separately, followed by a multi-motion composition phase during inference. The visual results demonstrate the method's capability to combine two or three motions within a single video, though the overall visual quality of the generated videos could be improved. Additionally, the paper proposes a new metric for evaluating multi-motion customization performance. The overall framework is functional, but the visual quality is not satisfactory for a customization method. Overall, I believe this is a borderline paper, and I am willing to see the authors' response.

### Strengths
1. The proposed two-phase framework successfully synthesizes multiple motions into a single video. With the carefully designed merging strategy, the resulting videos demonstrate good harmonization, particularly in the boundary regions between different motions.

2. The proposed C&C metrics provide a reasonable approach for evaluating multi-motion customization.

### Weaknesses
1. The visual quality of the multi-motion customization is not entirely satisfactory given the significant training process might already overfit on a single input video. Additionally, there is insufficient evaluation of the visual quality of the generated videos.

2. The evaluation lacks comparison with other motion-conditional generative models that use motion representations such as human skeletons. I am also curious whether providing those models with merged skeleton sequences would enable them to perform multi-motion transfer effectively.

### Questions
See weaknesses.

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
4

### Summary
This paper introduces CoMo, a novel framework that enables the learning and composition of multiple distinct motions within a single video. The method addresses two major challenges in motion customization: motion-appearance entanglement and multi-motion blending, through a two-phase design. First, a static-dynamic decoupled tuning approach disentangles motion from appearance to learn motion-specific LoRA modules. Then, a plug-and-play divide-and-merge strategy composes these motions spatially during denoising, allowing different subjects to perform distinct actions simultaneously. The authors also propose a new benchmark and evaluation metric (C&C score) for assessing multi-motion fidelity. Experiments demonstrate that CoMo achieves state-of-the-art performance in both single- and multi-motion customization, offering a flexible and training-efficient solution for controllable video generation.

### Strengths
1. This paper is well-written and structured, making it accessible to readers with varying levels of expertise in the field.
2. The motivation and the design of method are both reasonable and innovative.
3. Qualitative and quantitative results show clear improvements over baselines.
4. The paper provides thorough experimental evaluations, and all results supports the claim

### Weaknesses
1. It is recommended to enhance the diversity of motion customization scenarios, not only translation, but also rotation and scaling. 
2. From my point of view, the authors should discuss (or compare with, if possible) more existing methods that achieve similar motion customization (both U-Net-based ones and DiT-based ones), including but not limited to:
    1. MOFT: Video Diffusion Models are Training-free Motion Interpreter and Controller
    2. MotionClone: Training-Free Motion Cloning for Controllable Video Generation
    3. VD3D: Taming Large Video Diffusion Transformers for 3D Camera Control
3. This paper does not specify or thoroughly discuss the evaluation dataset, which may not provide a comprehensive view of CoMo’s effectiveness. Releasing more details of the evaluation dataset or incorporating videos from publicly available benchmarks would greatly enhance the credibility of the paper.
4. The comparison regarding computational efficiency should be provided.

### Questions
Please see the weakness.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes CoMo, a framework for compositional motion customization in text-to-video generation. The key novelty lies in enabling the synthesis of multiple distinct motions within a single video. The method features two stages: (1) Single-motion learning, which disentangles motion and appearance through a static–dynamic decoupled LoRA tuning scheme. (2) Multi-motion composition, achieved by a plug-and-play divide-and-merge strategy to compose multiple motion patterns during denoising.
 The authors also introduce a new benchmark and a Crop-and-Compare metric to evaluate multi-motion fidelity and blending. Extensive experiments show that CoMo achieves state-of-the-art results over baselines such as MotionDirector, DeT, and DreamBooth.

### Strengths
1. The paper proposes a two-phase pipeline that effectively separates motion and appearance through sequential LoRA tuning, and demonstrates clear improvements in motion disentanglement compared with joint training baselines. Both quantitative metrics and visual comparisons indicate clear gains in motion fidelity and compositional accuracy. 
2. The introduction of benchmark for compositional motion and the Crop-and-Compare (C&C) metric provides systematic toolset for assessing multi-motion fidelity and blending, which likely have lasting impact for subsequent research in controllable video generation.
3. This paper is well-structured with intuitive figures. And implementation details are transparently provided in the appendix.

### Weaknesses
1. **Limited analysis of scalability complexity.**
   The paper mainly demonstrates two- to four-motion composition in spatially separated regions. It remains unclear how the method performs when motions overlap or extend to longer temporal durations.
2. **Benchmark validation is somewhat limited.**
   While the introduced dataset and C&C metric are valuable, their correlation with human perceptual judgment is not quantified. More discussion or cross-validation with existing metrics would improve confidence in the evaluation.
3. **Insufficient description and discussion of training and benchmark data.**
   The paper does not quantize the data composition and diversity for the proposed benchmark, nor specify the scale and source of training videos used in training.  Without clearer dataset statistics and examples, the evaluation's representativeness and training reproducibility remain uncertain.

### Questions
1. **Data composition and usage.** 

   a) **Training data**: What datasets are used for training the single-motion LoRA modules? Please specify the data sources, scale, as well as whether the videos were curated or filtered in any way.  

   b) **Evaluation data**: Providing dataset statistics (such as the total scale, motion diversity and distribution, ..., in quantitative form) and representative examples would make the benchmark’s coverage and difficulty clearer.

2. **Handling of overlapping or interacting motions.**
   How does the divide-and-merge mechanism behave when spatial regions partially overlap or when two subjects physically interact?  Is there any mechanism to ensure temporal coherence across motion boundaries? 

3. **Analysis about robustness.** 

   How robust is CoMo when reference videos differ in viewpoint or temporal length? Section4.1 claims that evaluation data includes "camera motion", but it seems that this part was not involved in the subsequent analysis and visualization.

4. **Analysis about the region partitioning**. 

   The region partitioning process (dividing global video into several rectangular regions) in the Divide-and-Merge stage seems predefined. Could the authors provide more information about the partition strategy and discuss whether an adaptive or learned partition could improve compositional quality or reduce artifacts at motion boundaries?

5. **Evaluation reliability.**
   Has the C&C metric been validated through human studies or correlation with existing metrics?

6. **Generalization across base models.**

   Can the learned motion LoRA modules trained on one base model (e.g., Wan) be transferred to another DiT-based backbone?   This would be important to evaluate the modularity claim.

### Soundness
2

### Presentation
4

### Contribution
3
