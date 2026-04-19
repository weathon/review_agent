# VideoAgent: Self-Improving Video Generation

- Decision: Reject
- Scores: 3, 5, 5, 6

## Abstract
Video generation has been used to generate visual plans for controlling robotic systems. Given an image observation and a language instruction, previous work has generated video plans which are then converted to robot controls to be executed. However, a major bottleneck in leveraging video generation for control lies in the quality of the generated videos, which often suffer from hallucinatory content and unrealistic physics, resulting in low task success when control actions are extracted from the generated videos. While scaling up dataset and model size provides a partial solution, integrating external feedback is both natural and essential for grounding video generation in the real world. With this observation, we propose VideoAgent for self-improving generated video plans based on external feedback. Instead of directly executing the generated video plan, VideoAgent first refines the generated video plans using a novel procedure which we call self-conditioning consistency, utilizing feedback from a pretrained vision-language model (VLM). As the refined video plan is being executed, VideoAgent collects additional data from the environment to further improve video plan generation. Experiments in simulated robotic manipulation from MetaWorld and iTHOR show that VideoAgent drastically reduces hallucination, thereby boosting success rate of downstream manipulation tasks. We further illustrate that VideoAgent can effectively refine real-robot videos, providing an early indicator that robotics can be an effective tool in grounding video generation in the physical world.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper proposes VideoAgent for self-improving generated video plans based on external feedback. It first refines the generated video plans using self-conditioning consistency and utilizes feedback from a pretrained vision-language model (VLM), then it collects additional data from the environment to further improve video plan generation. From both simulation and real-world video generation experiments, it shows the effectiveness of the proposed method.

### Strengths
1. Clear equation writing, which shows the strong technique capabilities of the authors.

2. Figure 1 is clear, which makes me understand the main idea of this paper quickly.

3. Experiments for the VLM feedback is interesting.

### Weaknesses
1. I am unsure whether the *self-conditioning consistency* is an original contribution of this paper because of the poor writing clarity of Section 3.1. From the author's writing, it seems like some other papers also endow similar ideas. If the authors want to claim this as part of their contribution, the author should improve the writing of this part to clearly show which idea or equation is original.

2. I am confused about how to use the feedback from VLM for Equation 10. The authors did not provide any methodological description in Section 3.1 or Appendix, e.g., what is the *feedback*, or how to put the *feedback* into Equation 9.

3. Also, I am unsure whether using VLM feedback to improve video generation models is original since the author didn't discuss any related works for this part. If this is novel (or at least part of this is novel), the author should clearly point out which part is novel. Currently, people may assume that this section is totally based on some existing methods, because the author seems to assume that readers know how Equation 10 works, and therefore does not provide any method explanation.

4. From my point of view, the main contribution (if the above 3 questions can be solved) are: 1) self-consistency conditioning; 2) VLM-feedback; 3) real-world interaction feedback. However, from Table 1, we can see these three modules didn't bring much improvement (VideoAgent v.s. AVDC, only 2.7% increase). Instead, the main improvement comes from Online and Replan (23.5% increase for AVDC and 27.7% for VideoAgent). What's more, Online-and-Replan works across two different video generation methods (AVDC and VideoAgent). Thus, this greatly reduces the value of the three main contributions of this article.

### Questions
See the weakness part.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper introduces VideoAgent which self-improves generated video plans based on external feedback.
Given an initial frame and a language instruction, VideoAgent first generates a video plan and then iteratively refines it based on feedback from a pre-trained VLM.
The refined video plan can be converted to low-level robot actions for execution.
Successful trajectories are added to the dataset for further fine-tuning the video generation and refinement models.
Experiments on video plan generation were performed in both simulation and real-robot videos.
Results show that the proposed method can reduce hallucination in video plan generation and thus boost performance in downstream manipulation tasks.

### Strengths
The paper presents a novel method which uses feedback from VLMs to refine generated video plans.
It introduces a self-improving consistency model which predicts the clean video from a generated video and feedback from the VLM.
The proposed method can be continuously improved through online fine-tuning.
Experiment results indicate that the proposed method effectively enhances video generation and improves performance on downstream manipulation tasks.
The paper also includes ablation studies to examine the impact of different types of feedback from the VLM, and the effect of refinement and online iterations.
Additionally, it offers an in-depth analysis of the VLM's performance in providing feedback for video generation.

### Weaknesses
1. The paper leverages a self-conditioning consistency loss (Eqn. 7) for video refinement. The first term in Eqn. 7 is a diffusion loss and the second term is for consistency. The reason why including the second term in Eqn. 7 is not very clear. It seems that it encourages generating consistent $x^{(0)}$ from different $\hat x_{i}$. Is it possible to provide more explanation on why the consistency loss is necessary to be included? It would be great to include an ablation study which compares the refinement performance with and without this consistency loss.

2. Figure 2 is not very clear. If I understood correctly, $x^{(t + \Delta)}, x^{(t)}, x^{(t - \Delta)}$ are noisy latents at different denoising timesteps in a diffusion. But what is the target of this diffusion process? Is it $\hat x_{(i+1)}$? It would be great to provide more descriptions in the caption for further clarification.

3. The paper lacks real-robot experiments to validate the effectiveness of the proposed method in real-world policy learning. The paper would benefit strongly from incorporating a real-robot policy learning experiment. And comparing the proposed method with baseline methods without self-improving consistency (e.g. AVDC) would help understand how the proposed video refinement process helps downstream policy learning in the real world.

### Questions
1. VLMs are prone to hallucinations in some cases. Were any hallucinations from VLM feedback observed in the experiments? If such hallucinations were observed, would they potentially mislead the refinement process?

2. How does the proposed method perform in generalization settings? Does introducing feedback from VLM help the proposed method on handling novel text instructions? It would be great to incorporate a generalization experiment on video plan generation on real-robot data (e.g. BridgeData V2).

3. In Algorithm 2, if I understood correctly, the $\pi_{\theta}$ in line 284 should be $\hat f_{\theta}$?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper proposes VideoAgent, a model designed to improve video generation quality based on feedback from a VLM. Specifically, VideoAgent trains both a video generation model and a video refinement model, which share parameters and are based on diffusion processes. The video generation model functions as a conventional text-to-video model, while the video refinement model takes as input the ground-truth video, the predicted video, and feedback from the VLM to produce a refined video. The final output video can be mapped to an action trajectory. When the environment is interactive and can confirm task success, VideoAgent performs rollouts within the environment to gather additional data for training (but very slow now). This method is evaluated at the policy level in the Meta-World and iTHOR benchmarks, and for video quality on the Bridge dataset.

### Strengths
1. The idea of VideoAgent is novel; I haven’t seen an iterative approach to improving video generation quality before.
2. Obtaining feedback on videos from VLM is feasible.

### Weaknesses
1. The writing section can be strengthened. The distinction between the consistency model and DDPM is vague in this paper, yet it is crucial. Given that I believe DDPM + DDIM can achieve the same results (referring to the implementation of video generation models and video refinement models), it’s even more necessary to explain the necessity and motivation for using the consistency model. Specific issues can be referenced in the questions section.

2. The experimental section of the paper is relatively weak. The baseline is only AVDC, a typical video generation model without a corresponding video refinement model. Possible baselines could include VLP[1] and a naive text-to-video diffusion model (under similar training and inference FLOP). Additionally, results from traditional baselines based on BC and RL should also be included for reference.

3. More realistic robotic manipulation benchmarks should be considered, such as Simper[2].

4. It would be better to conduct experiments on a real robot; currently, the focus is only on testing quality using real-robot videos. Success rate evaluations are missing, which may necessitate the use of a real robot. If a real robot is not available, providing results from Simper would be acceptable.

5. Even after refinement, the video generation quality of VideoAgent on the Bridge dataset is not high compared to modern classic diffusion models, and the resolution is very low.

6. Fundamentally, generating a video and then generating actions is not widely regarded as a correct approach; the improvements made in this paper are still limited by the fundamental flaws of video planning. The video generation is too slow to be suitable as a policy. If the paper could demonstrate the importance of video generation for policy, it would greatly strengthen its contributions.

7. The paper seems to avoid the discussion of video generation speed. In reality, how many actions can be generated per second with this method? Or what is the average time required to generate a single action? The resolution should also be reported.

[1] Video Language Planning.
[2] Evaluating Real-World Robot Manipulation Policies in Simulation

### Questions
When I read this paper, I encountered some areas of confusion, and I hope my feedback can help the authors improve the clarity of the writing. If there are any inaccuracies in my understanding, I welcome any corrections.

1. The motivation for introducing the consistency model should be clearly articulated. I didn’t see a compelling reason to use the consistency model, as it seems possible to achieve similar results by using the training objective of DDPM and employing DDIM for inference.

2. Lines 119-120 should describe the loss for the consistency model rather than for DDPM. In DDPM, the loss should be calculated from ($x^{t+1}$ to $x^t$), which also makes the "vanilla objective for video diffusion" mentioned in line 197 confusing.

3. In line 224, it seems that video generation should be a single step if using a consistency model. Why are multiple steps required, and if they are, what distinguishes this approach from DDIM?

4. The motivation in Section 3.1 could be more clearly defined. The overall objective of the proposed method is to (1) generate video and (2) refine the video based on feedback until it is accepted by the VLM, using the same parameters. However, this goal is not clearly explained at the beginning of Section 3.1, which requires readers to infer the motivation after understanding the method.

5. Line 164 states, "the model can learn to preserve the realistic part of the video while refining the hallucinatory part," which could be misleading. Feedback is applied to the entire video, not frame-level feedback. This statement may somewhat overstate the model’s capability.

6. It is unclear how the parameters are shared between the video generation model and the video refinement model.

7. It is not shown in the method how the video maps to action. Indicating where this is discussed in the paper would be helpful.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces VideoAgent, an approach that refines video plans generated by video diffusion models through self-conditioning consistency and feedback from pre-trained vision-language models. When online interaction is available, VideoAgent closes the self-improvement loop by alternating between collecting successful online data and finetuning video models. Evaluations are performed on three benchmarks, across both simulated and real-world domains, and from the perspective of both task success rate and video generation quality, showing the effectiveness of the proposed method.

### Strengths
1. This paper proposes several novel techniques, such as self-conditioning consistency and incorporating VLM feedback, that improve the quality of the generated video plans through iterative refinement
2. The proposed method adopts a self-improving loop by finetuning the video models with additional successful trajectories collected online
3. The authors provide extensive evaluation results as well as experiment details, which back up the efficacy of the proposed method
4. This paper is well-motivated and easy to follow, it thoroughly discusses the limitations of the proposed work

### Weaknesses
1. The loop of collecting successful data through environment interaction and finetuning video models might incur large computational overhead, and the improvement seems to become marginal after two online iterations in Figure 4.
2. [nitpick] I believe it is essential to reduce hallucinations and improve the quality of video plans for better decision-making performance and I appreciate the overall contribution. However, in the Metaworld example (Table 1) provided in this work, simply replanning during inference still seems to be a comparatively cost-effective approach that can achieve higher overall success rates with a less performant video plan generator. I understand refinement and replanning can be combined for better improvement, but I wonder if the authors have stronger examples to highlight the necessity of refinement and self-improvement for task success.

### Questions
1. There seems to be a discrepancy between the self-conditioning-consistency loss In Algorithm 1 and Equation 7, is the second term in Eq. 7 missing in Algorithm 1?
2. During evaluation, is every video plan executed in an open-loop manner? What is the planning horizon for each setup, and how does it compare to the task horizon?
3. In Figure 4, how many additional trajectories are collected per iteration to achieve the improvement?
4. In the last row of Table 5, how is the "task success" defined specifically in the Bridge human evaluation? Is it defined as whether the generated videos respect the prompt from human perception? or as the authors mentioned in line 467, it depends on whether a generated video looks realistic.
5. As the video models are language-conditioned, it would be clearer to include the conditioning variable in the equations or algorithms somewhere.
6. Typo in line 342: brining

### Soundness
3

### Presentation
3

### Contribution
3
