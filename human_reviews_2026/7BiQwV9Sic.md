# Advancing Multi-agent Traffic Simulation via R1-Style Reinforcement Fine-Tuning

- Decision: Accept (Poster)
- Scores: 4, 6, 2, 8

## Abstract
Scalable and realistic simulation of multi-agent traffic behavior is critical for advancing autonomous driving technologies. Although existing data-driven simulators have made significant strides in this domain, they predominantly rely on supervised learning to align simulated distributions with real-world driving scenarios. A persistent challenge, however, lies in the distributional shift that arises between training and testing, which often undermines model generalization in unseen environments. To address this limitation, we propose SMART-R1, a novel R1-style reinforcement fine-tuning paradigm tailored for next-token prediction models to better align agent behavior with human preferences and evaluation metrics. Our approach introduces a metric-oriented policy optimization algorithm to improve distribution alignment and an iterative "SFT-RFT-SFT" training strategy that alternates between Supervised Fine-Tuning (SFT) and Reinforcement Fine-Tuning (RFT) to maximize performance gains. Extensive experiments on the large-scale Waymo Open Motion Dataset (WOMD) validate the effectiveness of this simple yet powerful R1-style training framework in enhancing foundation models. The results on the Waymo Open Sim Agents Challenge (WOSAC) showcase that SMART-R1 achieves state-of-the-art performance with an overall realism meta score of 0.7858, ranking first on the leaderboard at the time of submission.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper apply the R1 training style to the traffic model learning task to solve the distribution shift problem. It directly uses the waymo evaluation metrics to assign the reward. The method achives good result in the Waymo Sim Agent benchmark.

### Strengths
1. The performance of the method is good on the Waymo Sim Agent benchmark.
2. The paper is good written and easy to follow.
3. The ablation experiments are extensive.

### Weaknesses
1. The contribution of the paper is limited. It looks like an engineering work to apply the existing R1 training style to finetune an existing model based on CATK. 
2. The metric-oriented policy optimization method directly uses the evaluation metric as the reward, which makes its strong performance less meaningful. It may simply overfit to the metric itself rather than improving genuine behavioral realism across other metrics.
3. Taking the waymo sim agent score as reward is time consuming because the computation is slow which requires 32 rollouts.

### Questions
1. How do you assign the rewad among the 32 rollout? Do all 32 rollouts use the same reward?
2. What is the performance of using the original KL penalty?
3. What is the performance of using more iterations of SFT and RFT or making them optimize together?

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
This paper proposes to use reinforcement learning to post-train the Next-token-prediction-based motion planning model. It uses a three-stage post-training method, SFT-RLFT-SFT, and finally achieves SOTA on Waymo Open Sim Agent Challenge (WOSAC).

### Strengths
1. Agent simulation is an important problem for self-driving, especially building agents with realistic behaviors.
2. The paper proposes to use RL post-training to explore better behaviors to increase the realism and achieve SOTA on the commonly used benchmark WOSAC. Comprehensive experiment results, including ablations, demonstrate the effectiveness of the proposed method.
3. The paper writing is good and easy to understand.

### Weaknesses
1. The method is mainly designed for an NTP-based model, while new adaptations are needed for continuous prediction models like diffusion models, which may limit its future application.
2. It would be better to evaluate the method on more end-to-end planning benchmarks like nuPlan to show its general capabilities, which can further increase its impact.

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
- This paper tries to address the problem of distribution shift in the learned (supervised learning) simulators affecting their generalization capabilities.
- The authors propose SMART-R1 to align agent behavior with human preferences and relevant evaluation metrics for agent traffic simulation tasks.
- The authors introduce an iterative (interleaving) SFT-RFT-SFT post-training pipeline to maximize the performance and minimize the skills learned in previous training runs (catastrophic forgetting).
  - They introduce Metric-oriented Policy Optimization (MPO) to exploit the knowledge from priors and can try to optimize for collision, off-route cases, etc. that might not be there in the SFT data. 
- With the above recipe the proposed method was able achieve best performance on Waymo Open Sim Agents Challenge (WOSAC)

Given the comments related to weaknesses and limitations, I can have the rating as 2. Flexible to move this during or after rebuttal.

### Strengths
- The authors presented the text, figures and tables nicely.
- Converting the trajectories into motion tokens makes it easier for the sequence to be fed to the transformer and formulate it as a next-token-prediction (NTP) task.
- The closed-loop SFT seems to reduce the covariate shift in NTP models leading to improved performance.
- RFT stage aligns the model with metric preferences and enhances realism. 
- The paper does a good job in performing experiments and ablations to study the effect KL regularization, effect of different post-fine-tuning methods, effects of using different RL algorithms for RFT.

### Weaknesses
The approach proposed has off-late been a pretty common fine-tuning strategy for many robotics tasks. It is appreciated that the authors have done research in incorporating the multi-step post-training pipeline serving a dual purpose of maximizing performance and minimizing forgetting. However, there has been some work done on dealing with forgetting as well in LLMs ([Towards a Unified View of Large Language Model Post-Training](https://arxiv.org/pdf/2509.04419)) .

### Questions
1. Was keeping a KL term in the SFT tried?

2. Was there any KL term in the RL loss to minimize the divergence from the base policy while maximizing returns?

3. Was any other regularizer tried to prevent forgetting (for example, Elastic Weight Consolidation)?

4. Was there any KL term in the second SFT training (refer to [RL's Razor: Why Online Reinforcement Learning Forgets Less](https://arxiv.org/abs/2509.04259))?

5. How many agents are considered in the scenario and how are selected or filtered, if any?

6. Line 397: “Notably, applying two consecutive SFT phases” - isn't this just one big SFT stage with 2x epochs?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper introduces a trajectory forecasting system that is based on SMART and adds DeepSeek-R1-style training, which ultimately leads to SotA performance on a difficult dataset.

### Strengths
- **S.1:** The idea is simple and well-justified. It's not methodologically revolutionary but the results are very impressive.
- **S.2:** The writing is clear, the illustrations are simple but good.
- **S.3:** I really like your ablations, especially table 3-6. I think this is very good scientific practice.

### Weaknesses
- **W.1:** Reproducibility. I really dislike when papers don't submit any code with their submission for a system that's quite complex. Especially since your method is built on top of an existing codebase, I think it would've been great if you could've included an anonymous github link or a zip file with the submission.
- **W.2:** Performance/Train time? Along the same lines of the previous comment: how long was this trained, on which hardware? How fast is inference?
- **W.3:** The writing is moooostly clear but there are some sharp edges that I think need to be improved. In the abstract, you already mentioned "R1-style..." without mentioning what this is referring to (e.g."the SFT-RL-SFT training regimen of DeepSeek-R1") and you often refer to CAT-K rollouts, which I'm not familiar with and since it seems important to your method, it'd be nice to have a quick explanation in the main body of your paper.

### Questions
n/a

### Soundness
4

### Presentation
3

### Contribution
3
