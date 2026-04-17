# Info-GRPO: Training Reasoning Models via Correlation-Aware Exploration

- Decision: Reject
- Scores: 6, 4, 4

## Abstract
Recent studies have revealed policy collapse in advanced reasoning models trained with Group Relative Policy Optimization (GRPO), and entropy regularization has stood out as an elegant approach to promote exploration. Yet, within the vast token space of language models, entropy gradients often exhibit severe singularities, creating a direct conflict with the natural entropy decay required for convergence and thereby disturbing optimization dynamics. 
To resolve this tension, we present Info-GRPO, an information-theoretic framework that reconciles the opposing entropic forces of exploration and convergence by cultivating correlation between the policy and a latent prior. Info-GRPO leverages a contrastive regularization that maximizes the mutual information between latent variables and the policy. Intuitively, by augmenting prompts with latent variables, the model explores a more diverse set of policies that remain correlated with the latent prior, guiding conditional entropy toward convergence. Through this correlation-aware design, Info-GRPO respects the natural entropy reduction during training while enabling more effective exploration.
Extensive experiments demonstrate that Info-GRPO significantly outperforms vanilla GRPO and entropy-regularized GRPO across diverse reasoning benchmarks.
For instance, it achieves improvements of 3.75\%, 1.66\%, and 4.16\% in Avg@8 compared to GRPO based on Qwen2.5-Math-7B, Qwen2.5-7B, and DeepSeek-R1-Distill-Qwen-7B, respectively, under the AIME24 benchmark.
Furthermore, analysis reveals that Info-GRPO induces distinct and interpretable reasoning patterns conditioned on the latent variable, showcasing a more systematic and effective exploration strategy.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper present Info-GRPO, an information-theoretic framework that reconciles the opposing entropic forces of exploration and convergence by cultivating correlation between the policy and a latent prior. Info-GRPO leverages a contrastive regularization that maximizes the mutual information between latent variables and the policy. Intuitively, by augmenting prompts with latent variables, the model explores a more diverse set of policies that remain correlated with the latent prior, guiding conditional entropy toward convergence. Through this correlation-aware design, Info-GRPO respects the natural entropy reduction during training while enabling more effective exploration.

### Strengths
1.  The paper proposes a new perspective by leveraging a contrastive regularization that maximizes the mutual information between latent variables and the policy. By augmenting prompts with latent variables, the model explores a more diverse set of policies that remain correlated with the latent prior, guiding conditional entropy toward convergence. The paper also provides theoretical analysis to support this approach.
2.  The analysis correctly points out that the failure of traditional entropy regularization in LLMs is due to gradient singularities induced by the massive number of tokens in the tail of the distribution.

### Weaknesses
1.  Missing baselines: There have been several recent works rethinking entropy regularization, such as KL-Cov and Clip-Cov [1], training on 20% high-entropy tokens [2], and entropy-based advantage shaping [3]. I suggest the authors compare with these baselines for a more up-to-date evaluation.
    
2.  The evaluation is currently limited to the math domain on Qwen-series models. Extending the experiments to other model series and domains would help assess the robustness of the proposed method.

[1] The entropy mechanism of reinforcement learning for reasoning language models.  
[2] Beyond the 80/20 rule: High-entropy minority tokens drive effective reinforcement learning for LLM reasoning.  
[3] Reasoning with exploration: An entropy perspective.

### Questions
See weakness.

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
Reinforcement Learning with Verifiable Rewards (RLVR) is a primary training paradigm used to post-train modern-day Large Language Models (LLMs). Despite its popularity, RLVR is still plagued with several limitations, one of them being its limited ability to encourage exploration. While entropy regularization is commonly used in RL to offset the policy collapse caused by the lack of policy diversity, its optimization objective, which drives the policy towards a higher entropy region, inherently goes against the "natural optimization trajectory." Therefore, this work proposes Info-GRPO, a novel training framework that attains a balance between exploration and convergence without utilizing opposing objectives. Info-GRPO augments prompts with latent variables and maximizes the mutual information between them, such that a diverse set of reasoning strategies can be sought throughout the training process. The experimental results highlight the limitations of entropy regularization-based exploration in RLVR for LLMs and showcase the effectiveness of Info-GRPO across diverse benchmarks and models.

### Strengths
- The proposed method is well-motivated and technically sound. The authors show that entropy regularization is ineffective in RL-based LLM post-training due to the sheer vocabulary size, and then propose a novel framework that avoids policy collapse without relying on entropy regularization.

- The proposed method is simple and intuitive (I meant this as a compliment). Its implementation is easy to follow and can be reproduced at ease. 

- Although the performance improvement is marginal in some cases, it is consistently observed across diverse experimental scenarios (models, datasets).

- The authors present thorough ablation studies that analyze the effectiveness of each technical component.

### Weaknesses
- I believe it is standard practice nowadays to verify the results of LLM RL on two different domains - coding and mathematics. I kindly ask the authors to additionally share the results in the coding domain to show that Info-GRPO can be used for various problem-solving tasks.

- Although the authors present some case analysis in section A.3, I would appreciate a more structured (or quantitative, if possible) analysis, which clearly evidences that Info-GRPO obtains improved performance by seeking more diverse reasoning trajectories. I believe such an analysis would further set this work apart from other countless GRPO variants. The authors' explanation behind why Info-GRPO does not obtain a high Pass@1 score touches on this (lines 400-403), but this is also a mere guess based on a piece of experiment. 

- Also, I found it quite intriguing that randomizing the initial seeds alone could induce sufficient diversity in reasoning trajectories (especially as training progresses). Thus, explicitly analyzing how reasoning trajectories across different seeds evolve over time would aid my understanding (similar to the above point). Also, what happens if a greater/smaller number of latent priors are used to condition the policy?

- I acknowledge that this work does not have to be compared against all variants of GRPO because not all of them share the same goal. Nonetheless, it would be meaningful to add a few more noteworthy baselines to the results (e.g., Dr. GRPO, GSPO).

### Questions
Please refer to the Weaknesses section.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper examines the tradeoff in RL (GRPO) conflicting objectives with respect to diversity / entropy (that favour exploration) versus the collapse in singular high-entropy configurations.  The paper's analysis conclude that this an inherent modelling challenge due to the vast action space of almost uniform low probabilities, due to the huge vocabulary (token) space.  

The authors then construct a regularisation framework to apply to GRPO that conditions on a set of latent variables to add additional constraints to the cost function

### Strengths
* I really enjoyed the cause analysis of the failure of high entropy states as attributed to the long vocabulary $|V|$ tail.
* Plots in Fig 4 are very nice to motivate the output of Info-GRPO (but it takes a lot of room, there should be a better way to illustrate the same but with less space).

### Weaknesses
* Using latent codes to condition on as a regularisation technique is not necessarily well-motivated by your analysis; there are other forms of embodiment that I can think of.  Help your audience understand **why** latent codes directly follow from your causal analysis.  That is, I found the storytelling of the solutioning not compelling and disconnected from the motivating analysis.  
* I find some of the claims of underperformance of Info-GPRO too shallow.  Some of the analysis contributes too little to explanation, without any further empirical investigation to back up its hypotheses. unlike the motivation for the method (which I felt was strong).  More substantial analysis would help to strengthen their and make the paper more uniformly justified. 
* I found the choice of evaluation metrics Why choose K as 8 for Pass@K metric?  It's not clear whether this is adopting prior evaluation standards set by prior work.  Help defend that these values aren't cherry picked (Tables 2-4 round percentage values are particularly suspect, given the supposed large sample sizes).  I might just have favoured the reporting of Pass@1 and Avg@8 (or Majority@{K=8}) as others in this line report.
  * Related, why isn't some form of a MRR metric chosen?  Having some favouring of an earlier rank carrying better value makes sense in my opinion.
* It seems you are using $\LaTeX$ for typesetting.  Please fix your use of quotation marks accordingly: ``x'' versus ''x''
*While I agree that the related work is comprehensive (but a bit unfocused), factoring in some parts of this to the main paper is essential: your use of the Appendix A.2 I feel is cheating space requirements (I have to mark you down for this; I feel this is an unfair usage of space)

### Questions
* Why / how did you choose a value of 16 for the number of samples per prompt?  
* Are the $\epsilon$ values for clipping in DAPO the standard ones?

### Soundness
3

### Presentation
2

### Contribution
3
