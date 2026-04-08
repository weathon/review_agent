## Human Reviewer 1

### Summary
The paper addresses the problem of fine-tuning the pretrained diffusion models on a target reward while aiming for 1) preserving diversity of generated images and 2) fast convergence. It proposes Nabla-GFlowNet to do so, inspired by generative Flow Nets (GFlow-Nets) that sample with unnormalized density of the reward function. Experiments on different benchmarks show that the proposed method generally achieves the best diversity vs reward trade-off compared to baselines.

### Strengths
- The proposed idea is based on the generative flow nets, which makes it intuitive and straightforward. 
- The Nabla-GFlowNet can leverage the first order information of the reward function (gradient) while the baselines only use the zero-order information. 
- The experimental results show that the proposed method can generally achieve the best diversity vs. reward trade-off frontiers.

### Weaknesses
- I think the "predicted reward" estimation in Eq. 15 can be severely unreliable, especially for the high-noise time-steps of the diffusion model. The predicted clean image will be noisy, and if the reward function is calculated by a model that has been trained on not noisy images, the predicted reward will be inaccurate.

- The parameter \lambda and the output regularization described in Page 7 seems to be crucial to the model's performance, but they are not the paper's contribution.

- The qualitative samples that are compared with the baselines are only for the aesthetic score. I think a qualitative comparison on the HPSv2 can be more valuable and insightful about the model's performance.

### Questions
- I suggest that the authors include the qualitative comparative results for the HPS-v2 reward in the paper.

If the authors address my concerns, I am willing to increase my score.

### Soundness
3

### Presentation
3

### Contribution
2

### Rating
6

### Confidence
4

---

## Human Reviewer 2

### Summary
This paper introduces an interesting method to address the challenges of fine-tuning multistep sampling in diffusion models. It employs GFlowNets to incorporate a middle term, $F(x_t)$ or $g_\phi(x_t)$, which allows the reward score to effectively influence different timesteps.  This method successfully eliminates the need to train a reward model that accepts noisy input. This paper implements their idea in both theoretical and practical contexts. Section 3.1 covers the theoretical aspect, while sections 3.2 and 3.3 address the practical application. Experiments also show that this method can enhance reward tuning in diffusion models.

### Strengths
1. This paper presents a new method for addressing the challenges of fine-tuning multistep sampling in diffusion models using GFlowNets. This method effectively eliminates the need to train a reward model that processes noisy input.
2. This paper implements their idea in both theoretical and practical contexts. Section 3.1 covers the theoretical aspect, while sections 3.2 and 3.3 address the practical application.

### Weaknesses
The main weakness is in the experiment part.
1. The function $g_\phi(x_t)$ is an interesting and reasonable choice for achieving the fitness task; however, it results in approximately zero vectors, with a terminal constraint of $g_\phi(x_T) = 0$. It remains unclear whether Unet is a suitable option for this purpose.
2. The regularization term appears significant, with $\lambda=1000$ in the Aesthetic Score experiments and $\lambda=100$ in the HPSv2 experiments. However, Section 3.2 states that it "may eventually over-optimize the reward and thus neglect the pretrained prior." Is Section 3.2 more helpful for preventing over-optimization than the regularization term?
3. The method is interesting and useful, but the paper's claim of "Diversity-Preserving" in the title creates a gap. Does this imply that the method theoretically ensures better diversity, or is it based solely on observational results? From my perspective, this method prioritizes improving information backpropagation for fine-tuning multistep sampling rather than enhancing diversity.

If the authors can resolve my issue, I will contemplate raising my score.

### Questions
See the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
4

### Rating
6

### Confidence
3

---

## Human Reviewer 3

### Summary
The authors propose Nabla-GFlowNet (∇-GFlowNet) to efficiently finetune pretrained diffusion models. This approach addresses issues of limited sample diversity and slow convergence in existing methods by leveraging reward gradients through ∇-DB and its variant, residual ∇-DB. Empirical results show that residual ∇-DB enables fast, diversity-preserving finetuning of models like StableDiffusion on various realistic reward functions.

### Strengths
- The paper offers a comprehensive theoretical deduction of the proposed method, thoroughly explaining how the objectives nabla-DB and residual nabla-DB are derived. 
- By introducing residual ∇-DB, the authors extend the applicability of their work to pretrained large-scale models, which is crucial.
- The paper enhances the quantitative evaluation of diversity in generated samples. By employing a broader range of metrics and more extensive comparisons.

### Weaknesses
- The current experimental setting appears somewhat outdated. To enhance the study's relevance, please consider using more recent schedulers and pre-trained models instead of DDPM or Stable Diffusion 1.5.
- The qualitative results shown in Figure 2 are confusing. Additional explanation is needed to clearly demonstrate the superiority of ∇-DB, as DDPO and DAG-DB also exhibit strong performance.
- A user study would be helpful for evaluating diversity.

### Questions
Please see weaknesses.

### Soundness
2

### Presentation
2

### Contribution
3

### Rating
6

### Confidence
2

---

## Human Reviewer 4

### Summary
The authors propose a method, called $\nabla$-GFlowNet which which is modification of GFLowNets. They define a new objective $\nabla$-DB, which is a gradient informed version the Detailed Balance objective. The paper then goes on to propose a residual version of this loss which at optimality samples proportionally to the argumented distribution $r(x_T)p^\sharp (x_T)$, which maintains diversity of generations. The paper then presents experiments which shows a good pareto frontier of reward vs diversity on two tasks.

### Strengths
- This paper offers a good way at maintaining diversity in generations while aligning generations to a reward model.
- The paper is theoretically founded and shows that their new training objective maintains the validity of GFlowNets while taking into account gradient information.
- The work has a number of good ablations and experiments to show diversity and reward tradeoff.

### Weaknesses
- The problem of bias is well known in optimizations for diffusion models (elaborated in question section). Some treatment of this problem problem would be desirable since it seems that this method may be optimizing for the biased distribution

### Questions
- It is known that optimizing the KL constrained optimization problem, with the closed form solution of the augmented distribution, can lead to a biased result [0, 1]. Does this method have this bias problem? If not, how do you get around it?

[0] https://arxiv.org/abs/2409.08861
[1] https://arxiv.org/abs/2402.15194

### Soundness
3

### Presentation
4

### Contribution
3

### Rating
6

### Confidence
3

---

## Human Reviewer 5

### Summary
This paper introduces Nabla-GFlowNet with a new objective, $ \nabla$-DB, and its variant, residual $ \nabla$-DB, for finetuning pretrained diffusion models. The method aims to enhance sample diversity and finetuning efficiency by utilizing reward gradients. Experiments on two reward functions, Aesthetic Score and Human Preference Score (HPSv2), demonstrate improved performance.

### Strengths
- The paper has a clear motivation, addressing the challenge of preserving diversity and improving efficiency in finetuning diffusion models, which is crucial for real-world applications.
- The paper introduces a unique application of GFlowNet principles to diffusion model finetuning, specifically focusing on the reward gradient to preserve sample diversity.

### Weaknesses
- In the quantitative evaluation, the proposed method performs well on the smaller dataset of Aesthetic Score, nearly doubling the baseline in the DreamSim metric. However, on the larger dataset of HPSv2, its performance is similar to baselines, showing no clear advantage. The effectiveness of the method remains inconclusive due to the differences in dataset size.
- While the authors mention “fast and efficient finetuning” as a contribution, only Figure 4 shows comparable convergence speed to the baseline. It would be helpful to include details on training resource consumption, such as GPU usage and computational cost, to substantiate this claim.
- Figure 3 only compares the results of the pretrained model and the proposed method, lacking visual comparisons with other baselines.
- Figure 5 lacks sufficient information in the title and annotations to clarify what each point represents (e.g., training iteration). To improve clarity, consider adding an explanation in the figure caption or in Section 4.4 to specify the meaning of each point.
- The conclusion section is missing, which may limit the clarity of the paper’s overall findings and contributions.

### Questions
Please see weaknesses.

### Soundness
3

### Presentation
2

### Contribution
2

### Rating
6

### Confidence
2