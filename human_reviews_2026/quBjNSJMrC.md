# Squeeze the Soaked Sponge: Efficient Off-policy RFT for Large Language Model

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
Reinforcement Learning (RL) has demonstrated its potential to improve the reasoning ability of Large Language Models (LLMs), yet most existing Reinforcement Finetuning (RFT) methods are inherently *on-policy* RL, failing to reuse historical data and thus preventing efficient scaling. In this work, we explore the potential of *off-policy* RL to leverage historical data for rollout-efficient RFT. Specifically, we propose **Re**incarnating **Mix**-policy Proximal Policy Gradient (**ReMix**), which enables on-policy RFT methods to leverage off-policy data. ReMix consists of three major components: (1) Mix-policy proximal policy gradient with an increased Update-To-Data (UTD) ratio that utilizes the data from both current and past policies for efficient training; (2) KL-Convex policy constraint that combines the KL constraints on the base and precedent model to balance stability and flexibility; (3) Policy reincarnation that replaces the base model with the mix-policy RFT model in the mid way of training and restarts on-policy training, to achieve a seamless transition from early efficiency to steady convergence. In our experiments, we train a series of ReMix models based on PPO, GRPO from 1.5B, 7B base models. On five math reasoning benchmarks (i.e., AIME'24, AMC'23, Minerva, OlympiadBench, and MATH500), ReMix achieves an average Pass@1 accuracy of **52.10%** (with **0.079M rollouts**) and **64.39%** (with **0.011M rollouts**) on 1.5B and 7B models, respectively. Compared with 15 recent advanced models, ReMix shows SOTA-level performance with an over **30x to 450x reduction in training cost in terms of rollout data volume**, demonstrating superior training efficiency. Additionally, our multifaceted analysis reveals insightful findings, including the implicit preference for shorter responses of off-policy RFT, the collapse mode of self-reflection under severe off-policyness, etc. The code and the trained models are available at https://anitaleungxx.github.io/ReMix/ .

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a novel method designed to improve the sample efficiency of Reinforcement Finetuning for LLMs. The core problem addressed is the inefficiency of on-policy RL algorithms like PPO, which are standard in RFT but discard historical data after each update. ReMix combines three main components to enable the use of off-policy data including mix-policy proximal policy gradient objective, KL-Convex policy constraint and policy reincarnation step. This paper demonstrates that ReMix achieves better performances on several math reasoning benchmarks than baseline approaches and models with a great reduction in the required volume of rollout data.

### Strengths
1. The paper tackles a critical and expensive problem in the development of advanced LLMs: the prohibitive computational cost of reinforcement finetuning. The promise of achieving comparable or better performance with orders of magnitude less data is a highly significant contribution that could democratize the development of powerful reasoning models and enable further scaling.

2. The empirical evaluation is extensive. The method is tested on two different model scales across several math reasoning benchmarks in comparison with a lot of recent models. Moreover, this paper includes a lot of abalation study such as UTD ratios, KL-convex and policy reincarnation, and experiments in different scenarios such as training with constrained max response length and with differnet prompts.

### Weaknesses
1. Insufficient Comparison with SOTA Off-Policy Methods: The paper's primary weakness is its failure to adequately benchmark against other contemporary off-policy RFT methods. As a lot of papers discussed, GRPO is a biased estimation of KL penalty.[1, 2] Therefore, a lot of recent baselines should be included, such as DAPO [3]. Moreover, in Appendix F.3, the paper discusses the reason why it does not include other baseline algorithms including RePO, SRPO, SPO, AGRO and AsymRE with excuses like the lack of public checkpoints, different scales, or restrictive generation lengths.. However, such reasons are not sufficient to support the reason why the paper chooses only 2 baselines methods to re-implement. In addition, this paper chooses several baseline models which are trained with different datasets and settings. Therefore, it is unclear whether the quality of datasets and training hyper-parameters contributes to the improvement of the proposed method over these models.

2. Limited Scope of Evaluation: All experiments are conducted exclusively on math reasoning tasks. While this is a challenging and important domain, the paper makes general claims about improving RFT for LLMs. It is unclear whether the method's effectiveness, particularly its implicit bias toward shorter responses, would be validated in other domains such as code generation. The findings may be domain-specific, a limitation that is not acknowledged.

References: 

[1] Liu, Z., Chen, C., Li, W., Qi, P., Pang, T., Du, C., ... & Lin, M. (2025). Understanding r1-zero-like training: A critical perspective. arXiv preprint arXiv:2503.20783.

[2] Zhang, Y., Liu, Y., Yuan, H., Yuan, Y., Gu, Q., & Yao, A. C. C. (2025). On the design of kl-regularized policy gradient algorithms for llm reasoning. arXiv preprint arXiv:2505.17508.

[3] Yu, Q., Zhang, Z., Zhu, R., Yuan, Y., Zuo, X., Yue, Y., ... & Wang, M. (2025). Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476.

### Questions
How did you determine the optimal values for policy reincarnation step T in {50, 100}? Is there a risk that a poorly chosen T could erase the early-stage efficiency gains or lead to premature convergence before the model has fully benefited from off-policy exploration?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a new RL workflow for reasoning models, composed of three key designs: 1) mixture of online and offline data 2) mixture of current policy and base policy in KL regularization 3) mixture of training algorithms. The experiments are conducted on 5 different benchmarks, showing advantages over vanilla PPO and GRPO, in performance and training cost.

### Strengths
- clear figures
- extensive experiments on various benchmarks
- prominent cost reduction compared with vanilla PPO and GRPO
- the performance is generally comparable with online versions, even better sometimes

### Weaknesses
- only trained on Qwen model, which is known to be unreliable for reasoning training [1].
- no theoretical explanation for the design of KLC loss and ReMix algorithm
- no explanation for the choice of $\lambda$ and $T$
- figure 4 is too noisy for solid analysis
- proof for eq.6 is missed

[1] Shao et al. Spurious Rewards: Rethinking Training Signals in RLVR. https://arxiv.org/abs/2506.10947

### Questions
- why does ReMix-GRPO make the model even worse then base model for AIME in table 1?
- what is the exact formulation of distribution $i\sim \nu$ used in practice, and is there any theoretical explanation?
- is pass@1 enough? as [2] indicates that, the performance on pass@k might not be improved. 
- vanilla GRPO has a length normalization term, while some variants like Dr. GRPO [3] removed it. Did you apply length normalization in your experiments?

[2] Yue et al. Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model? https://arxiv.org/abs/2504.13837

[3] Liu et al. Understanding R1-Zero-Like Training: A Critical Perspective. https://arxiv.org/abs/2503.20783

### Soundness
3

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
5

### Summary
This paper presents **ReMix** (*Reincarnating Mix-policy Proximal Policy Optimization*), a framework designed to make reinforcement fine-tuning (RFT) of large language models more sample-efficient by effectively leveraging off-policy data.  

ReMix extends standard on-policy approaches such as **PPO** and **GRPO** through three main components:  

- **Mix-policy proximal gradient:** Combines on- and off-policy trajectories to achieve higher update-to-data (UTD) ratios and more efficient data reuse.  
- **KL-convex constraint:** Interpolates between the base and current policies to ensure updates remain both stable and adaptive.  
- **Policy reincarnation:** Periodically resets the base model to the current policy, enabling a smooth transition from rapid early learning to stable convergence.  

Across five math-reasoning benchmarks—**AIME24**, **AMC23**, **Minerva**, **OlympiadBench**, and **MATH500**—ReMix achieves state-of-the-art Pass@1 accuracy while using **30×–450× fewer rollouts** than leading baselines such as **DeepScaleR** and **AceReason**.  
The authors also analyze the behavioral effects of off-policy learning, observing patterns like shorter responses and reduced self-reflection when off-policyness becomes excessive.

### Strengths
**Clear Motivation & Contribution** – Addresses the well-known issue of *sample inefficiency* in RFT with a principled and well-structured solution.  
**Novel Integration of Off-Policy Concepts** – The combination of mix-policy updates, convex KL constraint, and policy reincarnation is original and intuitively justified.  
**Strong Empirical Results** – Achieves SOTA reasoning accuracy while reducing training cost by 30×–450× across multiple benchmarks.  
**Comprehensive Experiments** – Includes ablation studies, efficiency–accuracy trade-off curves, and behavioral analyses (e.g., response length, self-reflection rate).  
**Readable and Well-Presented** – The paper is clearly written and easy to follow.

### Weaknesses
Overall, there are no major weaknesses in this paper. A minor concern is that the authors did not compare **ReMix** with other *off-policy* algorithms, relying only on *on-policy* baselines. Including comparisons with off-policy methods in future work would provide a more comprehensive evaluation and strengthen the empirical claims.

### Questions
1. In Eq. (3), why do the authors use \(\frac{\pi_K}{\pi_{K-i}}\) to control the trust region? What is the motivation or benefit compared to using a fixed constant region? It would be helpful to see empirical evidence or ablation results demonstrating that this adaptive trust-region design improves performance.  
2. In Fig. 3, ReMix-PPO appears to achieve roughly \(6\times\) faster improvement than vanilla PPO in the early training stage. Does this imply that early trajectories are reused multiple times for updates? If so, could this repeated use of the same trajectories lead to training instability or bias accumulation?

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
2

### Summary
This paper proposes ReMix, a method for efficient off-policy Reinforcement Finetuning (RFT) of Large Language Models. ReMix integrates three key components: a Mix-Policy Proximal Policy Gradient to utilize historical data with an increased Update-to-Data ratio, a KL-Convex policy constraint that dynamically blends penalties from the base model and the previous policy, and a Policy Reincarnation mechanism that switches the base model and reverts to on-policy training mid-way. The authors demonstrate strong empirical results on mathematical reasoning benchmarks, showing that ReMix-based models can achieve state-of-the-art or competitive performance with a dramatic reduction (30x to 450x) in the amount of rollout data required compared to several strong baselines.

### Strengths
- Well-Designed and Synergistic Method: The three-component architecture is intuitive and addresses key challenges in off-policy RFT. The components work synergistically, as shown by the ablations, with Policy Reincarnation being a particularly clever and novel idea for bridging early-stage efficiency and late-stage stability.

- Rigorous Ablation Studies: The ablation study in Table 3 is comprehensive and effectively validates the contribution of each proposed component, clearly showing that the full combination is necessary for optimal performance.

### Weaknesses
- Overly Optimistic and Potentially Misleading Framing: The paper's central claim of "30x to 450x" reduction in rollout data is framed in a way that overstates the advantage. This is achieved by comparing ReMix's early-stage performance against baselines' final performance (e.g., ReMix-PPO at 100 steps vs. PPO at 900 steps). A comparison of final performances shows a much more modest efficiency gain, undermining the dramatic narrative.
- Limited Scope of Evaluation: The empirical validation is constrained in two key dimensions: model scale (≤7B parameters) and task domain (mathematical reasoning only). This leaves the generality of ReMix an open question. Its effectiveness on larger models (e.g, 70B+) or on fundamentally different tasks (e.g., code generation) is not established. Furthermore, the absence of ReMix-GRPO results for the 7B model is a notable omission, as it prevents a consistent assessment of the method's performance across different underlying RL algorithms at a more capable scale.

### Questions
- The KL-Convex constraint uses a dynamically decaying coefficient $\lambda(t)$ to balance between the base and previous policy. The design of this decay schedule $\lambda(t) = \max\left(1 − 0.1 \cdot \lceil \max(t − 50, 0) / 10 \rceil, 0.5\right)$ appears to be heuristic. Could you provide an analysis that justifies this specific form?
- The Policy Reincarnation mechanism is a critical component, yet its triggering is based on a predetermined step $T$. How was this specific hyperparameter chosen, and what is the sensitivity of the final performance to this choice?
- The analysis reveals a strong "preference for shorter responses." Do you think this bias could be harmful in tasks where comprehensive reasoning is crucial, and if so, how could ReMix be adapted to control for this effect?

### Soundness
2

### Presentation
3

### Contribution
2
