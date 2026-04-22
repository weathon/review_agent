# High-Fidelity Human Motion Generation with Motion Quality Feedbacks

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 2, 6

## Abstract
Text-to-motion generation aims to synthesize realistic human motions from natural language descriptions. Prevailing approaches typically condition generative models on embeddings from the pre-trained CLIP text encoder. However, a fundamental discrepancy exists: CLIP's embeddings are optimized for static visual semantics, failing to capture the dynamic nuances essential for motion, consequently leading to suboptimal generation quality. To bridge this semantic gap, we propose AdaQF, a novel diffusion-based framework that enables the autonomous and efficient adaptation of the CLIP text encoder through feedback-driven co-optimization. AdaQF introduces a quality feedback loop, where semantic consistency constraints, between the generated motion, the conditioning text, and the ground truth motion, guide the fine-tuning of the CLIP encoder via low-rank adaptation. This process yields AdaCLIP, a motion-specialized text encoder that produces semantically rich and  dynamic-aware embeddings. Our framework delivers advantages from three perspectives: it achieves state-of-the-art performance on standard benchmarks, achieving state-of-the-art results with an FID of 0.039 and an R-Precision of 0.888 on the HumanML3D database; it facilitates dramatically faster convergence (up to 8x); moreover, the resulting AdaCLIP module demonstrates remarkable transferability, serving as a versatile drop-in replacement that elevates the performance of various motion generation models including the VQ-VAE-based and latent diffusion-based ones, thus presenting a general and efficient solution for high-fidelity text-to-motion synthesis. The code of this paper will be released.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper presents AdaQF to improve the performance of text2motion generation. One common issue with CLIP encoder in the T2M task is: CLIP is trained for static visual semantics and is bad at temporal information. This paper proposes to fine-tune CLIP text encoder with LoRA rather than freeze it when training the text2motion diffusion models. It introduces two additional loss terms in the semantics space via pretrained text and motion contrastive encoders. Experiments show significantly improved performance on common benchmarks.

### Strengths
1. The proposed method is easy to follow and makes a lot of sense to me. The performance improvement is significant. A simple yet effective approach should always be preferable.
2. The proposed method also shows faster convergence and transferability, which might potentially have a larger impact on the broader research community by using similar techniques or pretrained text encoder.
3. The qualitative results look very good compared to previous SOTA methods.

### Weaknesses
1. For the terminology, I cannot agree that the proposed method should be described as "feedback". For "feedback", I would rather expect an iterative loop that improves the model from the old version. The proposed method adds additional loss terms to the common supervision. Though it's simple, the experiment shows its effectiveness. I don't think it's proper to wrap it with a fancy term that people think of RLHF, which has nothing to do with the proposed method. I neither agree that it's about "quality" rather than semantics.
2. The paper should discuss more related work that uses additional loss terms computed in latent semantics space, such as LaMP[1]. The experiment should also include more SOTA methods or explain the reason why not reported those methods.

[1] Li, Zhe, et al. "LaMP: Language-Motion Pretraining for Motion Generation, Retrieval, and Captioning." The Thirteenth International Conference on Learning Representations.

### Questions
1. I wonder how the proposed method is different from previous work, i.e. LaMP [1], which also trains the text encoder along with the generator, as well as semantic latent space supervision via some contrastive encoder. What's the advantage of proposed method compared to the closely related work?
2. What's the necessity of LoRA finetuning instead of full parameter fine-tuning? Will full parameter fine-tuning lead to even better performance?
3. From Table 4, it seems that tuning the CLIP without the proposed two loss terms will degrade the performance. I wonder what's the possible reason.
4. If the additional two loss term is the actual effective design, I'm curious how the vanilla MDM could benefit from the proposed loss terms. So, I would expect to see the result of vanilla MDM trained with the proposed "consistency constraints".

[1] Li, Zhe, et al. "LaMP: Language-Motion Pretraining for Motion Generation, Retrieval, and Captioning." The Thirteenth International Conference on Learning Representations.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes AdaQF, which accomplishes the motion generation task by fine-tuning the pre-trained CLIP in the form of LoRA and introducing a semantic consistency regularization loss. Experiments demonstrate that this method effectively improves the quality of generated motions and enhances the semantic information of the motions.

### Strengths
1.This paper identifies the inadequacy of CLIP in motion generation capabilities and improves the generation performance by fine-tuning CLIP.

2.It proposes MoCLIP to replace the pre-trained CLIP, and introduces a semantic consistency loss by training a motion extractor and a text extractor, thereby enhancing both the generation quality and the semantic information of the generated motions.

3.The experiments are relatively comprehensive, and the structure of the paper is clear.

4.Some video demos are provided, making the result presentation more clear.

### Weaknesses
1.My primary concern is that the novelty of this contribution is very limited. The inadequacy of CLIP's representational capability in motion generation has already been addressed in similar works, which have also proposed corresponding solutions (e.g., LaMP). However, this paper does not discuss this work, nor does it include it in the baselines.

2.The authors pre-train a Motion Extractor and a Text Extractor, use these two modules to generate corresponding embeddings, and calculate the similarity between these embeddings as the loss. In my opinion, this is an approach that treats a metric as a loss. A pre-trained evaluator should achieve the same effect, and the similarity is also closely related to the R-Precision metric.

3.The related works included in the baselines are insufficient; comparisons with more existing works are required.

4.Some visual demo images can be further added in the main text.

### Questions
1.Discuss the main differences from LaMP, compare with its experimental results, and elaborate on the advantages over it.

2.Although CLIP is a foundation model, its parameter count is not large. Why is fine-tuning performed in the form of LoRA, and would full-parameter tuning yield better results?

3.The authors claim that this optimization method retains CLIP's text understanding capability. Can this be proven?

4.In the ablation study, why is the FID score of the "without both" setting better than that of the "without consistency alone" setting?

5.In the appendix, I find it strange why the results of AdaQF trained without using KIT-ML are better than those trained on KIT-ML. Does this indirectly indicate that the method is only effective for the R-Precision metric? Because I believe it is an approach that optimizes the loss by adopting the metric calculation method.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces an innovative framework for human motion generative model, which mainly focuses on text embedding. The proposed method achieves noticeable improvement while significantly reduces the time cost of covergence.

### Strengths
1. The paper demonstrates impressive quantitative and qualitative results, establishing a new state-of-the-art with a substantial margin.

2. The proposed framework is novel. The associated conclusions and experiments make significant contributions to the research community.

3. The paper is well-written, ensuring that its content is easily understandable for readers.

4. The authors show the improvement brought by the proposed methods on serveral different backbones, which strongly shows the generalizability and effectiveness of the proposed method.

### Weaknesses
1. In the supplementary material, please include a user study as a complement to the qualitative comparison to further demonstrate the effectiveness of the improvements.

2. I have a concern that, when training the contrastive-learning model, the authors used the same architecture as the standard evaluator. This risks 'hacking' the evaluation protocol rather than genuinely improving performance. It would be better to train several evaluators with different architectures and report the corresponding results to show the gains come from the proposed training paradigm rather than a metric-specific hack.

3. It would be better if the authors report the number of parameters and inference speed of the proposed method.

### Questions
Please kindly refer to the weaknesses mentioned above.

### Soundness
3

### Presentation
3

### Contribution
3
