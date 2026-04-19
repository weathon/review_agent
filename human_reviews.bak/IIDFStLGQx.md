# MAVIN: Multi-Action Video Generation with Diffusion Models via Transition Video Infilling

- Decision: Reject
- Scores: 5, 5, 6, 5, 6

## Abstract
Diffusion-based video generation has achieved significant progress, yet generating multiple actions that occur sequentially remains a formidable task. Directly generating a video with sequential actions can be extremely challenging due to the scarcity of fine-grained action annotations and the difficulty in establishing temporal semantic correspondences and maintaining long-term consistency. To tackle this, we propose an intuitive and straightforward solution: splicing multiple single-action video segments sequentially. The core challenge lies in generating smooth and natural transitions between these segments given the inherent complexity and variability of action transitions. We introduce MAVIN (Multi-Action Video INfilling model), designed to generate transition videos that seamlessly connect two given videos, forming a cohesive integrated sequence. MAVIN incorporates several innovative techniques to address challenges in the transition video infilling task. Firstly, a consecutive noising strategy coupled with variable-length sampling is employed to handle large infilling gaps and varied generation lengths. Secondly, boundary frame guidance (BFG) is proposed to address the lack of semantic guidance during transition generation. Lastly, a Gaussian filter mixer (GFM) dynamically manages noise initialization during inference, mitigating train-test discrepancy while preserving generation flexibility. Additionally, we introduce a new metric, CLIP-RS (CLIP Relative Smoothness), to evaluate temporal coherence and smoothness, complementing traditional quality-based metrics. Experimental results on horse and tiger scenarios demonstrate MAVIN's superior performance in generating smooth and coherent video transitions compared to existing methods.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper proposes a video prediction method that outputs intermediate transition frames given varying numbers of preceding and succeeding frames. Additionally, a CLIP-based evaluation method is introduced to verify inter-frame coherence.

### Strengths
1. The MAVIN model is proposed to generate transition videos between two independent video segments, and it is capable of performing this task.

2. BFG is introduced to guide the Diffusion process in generating transition videos, and GFM is proposed to smooth noise for better quality generation.

3. A new metric, CLIP-RS, is introduced to evaluate whether the smoothness of generated videos aligns with the original videos.

4. The paper presents clear illustrations, with Figure 1 effectively conveying the entire model's flow.

5. The authors provide code to enhance reproducibility.

6. The method has also been applied in extended contexts.

### Weaknesses
1. This task has some overlap with video prediction or frame interpolation, making it not particularly novel, and there are numerous models that have completed this work.

2. I believe the effectiveness of the proposed method lies in the use of Dynamic Boundary Frame guidance, which is fundamentally about injecting image features into cross-attention using CLIP—this technique has been widely applied in Image to Video generation.

3. Regarding GFM, many papers have discussed the influence of initialization noise on generation [1].

4. The testing data in this paper is somewhat limited; for example, it only uses two self-created datasets and does not test on publicly available datasets.

5. For the ICLR conference, this paper lacks explanatory depth.

[1] Wu T, Si C, Jiang Y, et al. Freeinit: Bridging initialization gap in video diffusion models[J]. arXiv preprint arXiv:2312.07537, 2023.

### Questions
1. What does "PDF" mean in line 215?

2. In line 427, the roles of BFG and GFM are discussed, but the data in Table 3 does not convincingly support this viewpoint. For instance, while BFG provides contextual guidance, removing BFG does not result in significant changes to the Clipscore; only when both are removed do substantial changes occur. Does this prove the combined effect of BFG and GFM?

3. There are many ways to transition between the two input videos; is it reasonable to use SSIM to evaluate the structural similarity of generated videos and GT frame by frame? For example, in Figure 2, it is reasonable for the tiger to rise in any frame, but using SSIM requires the tiger to rise at a specific frame.

### Soundness
3

### Presentation
3

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
The paper addresses the task of generating smooth and natural transitions between two video segments. A consecutive denoising strategy with variable-length sampling is adopted for infilling training. Boundary frame guidance is proposed to ensure semantically coherent frames, and the Gaussian filter mixer method is applied to balance the noise discrepancy between training and testing. A novel CLIP-relative smoothness metric is introduced to evaluate temporal coherence and smoothness. Experimental results on horse and tiger videos show that the proposed method achieves superior performance compared to SEINE and DynamiCrafter.

### Strengths
- Among the proposed components, the GFM module is an especially interesting idea.
- The authors specifically introduced the CLIP-RS metric to quantify relative smoothness, and Table 2 provides valuable insights on this metric.
- The videos in the supplementary materials and results in Table 1 demonstrate that MAVIN achieves superior results compared to prior work using the proposed benchmark.
- The ablation study effectively shows the impact of the proposed BFG and GFM components.

### Weaknesses
- The dataset used in the experiments is limited to tiger and horse videos, restricted domains with only 45 minutes of video available for training. Furthermore, the method requires fine-tuning an existing T2V model for 40k steps with a batch size of 1. This setup is limited in both scope and scale, making it insufficient to fully demonstrate the proposed method’s effectiveness. Larger-scale experiments across multiple domains would strengthen the evaluation.
- The compared models, SEINE and DynamiCrafter, are trained on open-domain data, enabling them to handle a broader range of scenarios. However, they are not specifically trained on animals, which makes the comparison to MAVIN (focused on animal videos) less direct. Training SEINE and DynamiCrafter on the proposed benchmark could enable a fairer comparison.
- The proposed components rely on standard approaches in video generation. For example: 1) the infilling objective in L201–L208, 2) variable-length sampling with a length embedding added to the timestep encoding, and 3) BFG uses CLIP embedding rather than a low-level embedding. These contributions, while effective, are not technically new.

### Questions
The authors argue that the BERT-like training in Chen et al. is less effective due to the alternating appearance of target and reference frames. To address this, they propose applying noise to a consecutive subsequence of training data. However, this approach is relatively standard, as seen in methods like MAGVIT (Yu et al.). Providing additional background and discussion would help contextualize this choice and highlight its novelty.

### Soundness
3

### Presentation
3

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
This video proposes a method for filling in between two videos to build multi-action videos, while most video generation models are focused on generating videos containing one atomic action. For this goal, a training paradigm has been proposed to ensure the smoothness and preserving context in the transition state of a video. Authors also proposed a new evaluation metric.

### Strengths
The proposed method has enough novelty.
The authors touched on the problem of evaluation of the multi-action video generation.

### Weaknesses
The main weakness of this manuscript is the small scope of its train/test data. Also, the few examples make it very hard to draw a conclusion.

### Questions
1- MAVIN performs relatively better under the `test-auto` compared to the baselines. What could be the reason?
2- I don't have a clear understanding of how Eq.5 & 6 outputs are being used. Can you define each term and also explain more of where each is being used? Also, it would be beneficial if their usage be visualized in Fig.1.
3- Why not use a simple (like linear) interpolation between (z^s and z^e)? How much performance advantage is gained by the FFT and IFFT operations and doing the low pass filtering in the frequency domain?
4- Motion (and its smoothness) is very important in temporal coherence. How CLIP-RS address that? If the CLIP-RS can just measure the contextual coherence and not the motion, which other metric do you observe for that?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
2

### Summary
This paper proposes a diffusion-model based method to generate transition video frames for two given videos. It first designs a consecutive noising strategy coupled with variable-length sampling for handling large infilling gaps and varied generation lengths. To address the lack of semantic guidance during transition generation, it utilizes boundary frames as guidance. During inference, Gaussian filter mixer (GFM) dynamically manages noise initialization, mitigating train-test discrepancy while preserving generation flexibility. Finally, it introduces a new metric, CLIP-RS (CLIP Relative Smoothness), to evaluate temporal coherence and smoothness. Experiments are conducted on two animal datasets and demonstrate good results.

### Strengths
1. The motivation of proposed methods is clear introduced.
2. The paper writing is good and easy to follow.

### Weaknesses
1. It needs a clear definition for multiple actions in the Abstract and Introduction. And how existing method processes this scenario. 
2. In the related work, please state how different between the proposed method and existing works, which helps reader to decide the novelty.
3. Are the two datasets are multiple action videos?
4. More quality results are needed and it is better to show some failure cases.

### Questions
See the above weaknesses.

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
4

### Summary
This paper introduces MAVIN, a framework for the transition video infilling task, which aims to generate an intermediate video clip between two adjoining clips, ensuring a fluid and seamless transition. To enhance the performance of the video model, the authors propose several methods. First, they introduce a self-supervised training pipeline that allows transition video infilling without the need for text prompt guidance. To address large infilling gaps and accommodate varied generation lengths, they employ a consecutive noising strategy combined with variable-length sampling. Second, boundary frame guidance is utilized to mitigate issues caused by the lack of semantic guidance. Third, a Gaussian Filter Mixer is proposed to balance initialization discrepancies and generation flexibility. Additionally, they introduce a metric to evaluate temporal coherence and smoothness. This approach is significant for its potential in fields like digital art making, where smooth and natural transitions are crucial. Extensive qualitative experiments demonstrate the effectiveness of the proposed method. However, I have concerns regarding the experimental setup (see weaknesses section).

### Strengths
This paper focuses on the transition video infilling with diffusion models, which is an important research topic that is relatively under-explored compared to the Text-to-Video or Video-to-Video setting. It is significant due to its potential applications in fields like digital art making. Generating a transition video with large motion gaps while maintaining motion consistency  is a challenge problem, for which this paper contributes interesting training pipeline and system design for offering guided information and stabilizing the inference. In particular, the use of GFM to balance initialization discrepancy and generation flexibility is technically sound and interesting. The generated results seem to be more smooth and natural than other methods by leveraging the proposed methods.

### Weaknesses
Lack of discussion and comparison with most relevant works on the  transition video infilling setting (e.g., [1], [2]).
Lack of ablation studies.
The experiment results on horses and tigers data set is not convinced, since it is too simple. I recommend the author to examine the proposed method on open-domain test set. I also expect the video result on digital art or cartoon scenarios.

[1] Generative Inbetweening: Adapting Image-to-Video Models for Keyframe Interpolation, 2024.

[2]  Explorative inbetweening of time and space, 2024.

Overall, while I appreciate the important problem that this paper proposes to solve. However, the unsatisfactory performance could come from pre-mature experiments. I believe this paper can benefit from more thorough experiments and in-depth study.

### Questions
I expect the experimental results on other scenarios.

### Soundness
2

### Presentation
4

### Contribution
2
