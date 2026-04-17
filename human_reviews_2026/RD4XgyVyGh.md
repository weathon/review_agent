# ActiveDPO: Active Direct Preference Optimization for Sample-Efficient Alignment

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6

## Abstract
The recent success in using human preferences to align large language models (LLMs) has significantly improved their performance in various downstream tasks, such as question answering, mathematical reasoning, and code generation. However, achieving effective LLM alignment depends on high-quality datasets of human preferences. Collecting these datasets requires human preference annotation, which is costly and resource-intensive, necessitating efficient active data selection methods. Existing methods either lack a strong theoretical foundation or depend on restrictive assumptions about the reward function, such as linear latent reward functions. To this end, we propose an algorithm, ActiveDPO, that uses a theoretically grounded data selection criterion for non-linear reward functions while directly leveraging the LLM itself to parameterize the reward model used for active data selection. As a result, ActiveDPO explicitly accounts for the LLM's influence on data selection, unlike methods that select data without considering the LLM that is being aligned, thereby leading to more effective and efficient data collection. Our extensive experiments demonstrate that ActiveDPO outperforms existing methods across various models and real-world preference datasets.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper aims to develop an active preference data selection algorithm.
The authors use the gradient information to estimate the difference between the current model's reward and the ground-truth, i.e., the uncertainty of the model.
Based on this, they propose to select data that maximizes the uncertainty for annotation.
The authors further introduce batch selection and random projection with LoRA gradients to reduce computational cost.
Empirical results on several models and datasets demonstrate that the propsed method outperforms baselines in terms of trained models' reward.

### Strengths
1. **Theoretical Grounding**: The proposed method is well-grounded in theory, with clear derivations from the estimation error in reward modeling to the data selection criterion.
2. **Practical Considerations**: The authors address the computational challenges of the proposed method. These practical techniques are useful and insightful for applying gradient-based data selection in real-world scenarios. The authors also provide in-depth ablation studies to validate the effectiveness of these techniques.

### Weaknesses
1. **Training and Evaluation Settings**: The preference dataset is annotated by small reward models (OpenAssistant/reward-model-deberta-v3-large), and the evaluation metric is also this reward signal.
However, a more convincing and up-to-date training and evaluation protocol is to use more powerful RMs (e.g., ArmoRM) for annotation and alignment benchmarks (such as AlpacaEval, Arena-hard, etc.) for evaluation aside from only the RM score [1].
2. **Computational Cost**: Although the authors propose several techniques to reduce the computational cost, the method still requires significant resources due to gradient computations. 
For example, the gradient computation may outweigh inference cost of an external RM for data selection.
The authors should at least provide a more detailed analysis of the computational overhead compared to baseline methods and annotation time, since the goal of active learning is to reduce the training cost. 

[1] Meng, Yu, Mengzhou Xia, and Danqi Chen. "Simpo: Simple preference optimization with a reference-free reward." Advances in Neural Information Processing Systems 37 (2024): 124198-124235.

### Questions
1. $V_{t-1}$ is an important term in the proposed method, but there is little explanation about its meaning. Can you provide more details about it? 
2. How does the proposed method compare with full data annotation (i.e., no selection) in terms of computational cost and performance?

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
This paper develops an active learning approach for training models with direct preference optimization (DPO). Grounding their choice in the theory around dueling bandits, the paper develops a gradient-based uncertainty criterion which is used to select prompts for human labelers to consume. The method has significant computational and memory requirements: several mitigations are developed in order for the method to be deployed tractably. The method is evaluated on several different data sets and shown to help perform similarly to active learning approaches for DPO.

### Strengths
+ The motivation of the Active DPO method is theoretically grounded and principled.
+ The implementation details which allow the method to become tractable are quite clever and dramatically improve the tractability of the method.
+ The ablations are quite in-depth and demonstrate effectively which parts of the algorithm are important; and show that their design is robust to various different models and datasets.
+ The Active DPO method appears to outperform all other Active DPO approaches in the experiments and on some experiments significantly outperform the random baseline.

### Weaknesses
+ The description of the algorithm is slightly unclear at times. Particularly around the description of the matrix V_t. This is presumably an outer product of the gradients, but the authors don't comment on the fact that this is obviously intractable to store for modern LLMs, let alone invert. I appreciate that the matrix is tractable when projected to 8192 dimensions with the approximations made later on, but the authors should highlight this difficulty earlier on.
+ The computational requirements of this approach are pretty large, even with the author's mitigations. As far as I understand from the paper, ActiveDPO requires computing the gradient on every example T times, and so (assuming that the fwd/bwd pass is the bulk of the computation, and that the alternative is optimization with T steps, computing the gradient on each example only once), ActiveDPO will take many more FLOPs and wall-time than random selection. The authors don't address this point very much, and it would be very interesting for them to provide the results from figure 1, but normalized by FLOPs instead of iteration. With this amount of extra computation, a practitioner could even potentially even use a method like GRPO, training a separate reward model on random samples, and potentially even get better results than using ActiveDPO. 
+ The computational complexity section in the appendix is overly simplified and even a bit misleading. For instance, the authors say that the projection of the gradients to d dimensions is going to dominate the model computation of the gradients, but this is not likely to be very true given the small size of the LoRA gradients vs. the large number of FLOPS required to compute the full gradient with a full forward or backward pass.

### Questions
Empirically, what is the amount of flops needed to run the various different selection strategies and how do they compare with the flops required to compute the gradient and do the forward/backward passes?

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
4

### Summary
The paper proposed a new objective to conduct active preference learning in the LLM alignment task. The proposed method used something similar to the influence function, which is a intuitive choice.

### Strengths
The active preference learning research topic is very important. I like the method which is similar to the influence function to conduct active learning. The writing is clear and the scale of the experiment is OK.

### Weaknesses
**1. Discussion on Comparison with Active Preference Learning for Large Language Models**

A detailed comparison with Active Preference Learning for Large Language Models (arXiv:2402.08114
) would strengthen the paper. In particular, while the APL paper focuses on reward difference—which only reflects the immediate step before updates—the current work’s use of gradient difference captures the potential improvement after updates. This distinction highlights a more forward-looking and theoretically meaningful criterion. Including a discussion that formally articulates this advantage would enhance the theoretical depth of the paper.

**2. Evaluation of Win-Rate Over More Iterations**

It would be valuable to present win-rate comparisons over a larger number of iterations to examine how performance evolves as active learning progresses. Since the proposed approach operates in an online learning setting, it is important to investigate whether performance degradation occurs due to catastrophic forgetting. Prior work suggests that active learning strategies may sometimes exacerbate forgetting compared to random sampling, so empirical results here would be insightful.

**3. Impact Study on Batch Diversity Encouragement Term ($V$)**

The inclusion of the batch diversity encouragement term, $V$, is an elegant design choice that facilitates batch updates while promoting sample diversity. However, an ablation or impact study isolating the contribution of $V$ would help clarify its empirical importance and guide future design choices.

### Questions
See the 2/3 points above.

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
This paper introduces ActiveDPO, which integrates active learning into DPO for reinforcement learning from human feedback (RLHF). The key innovation is the linearization of the DPO objective at the policy’s last layer, enabling the application of D-optimal design to select the most informative feedback—either online (ADPO) or offline (ADPO+). The authors prove that the logit error decays as $O(d/\sqrt{n})$ and validate their algorithms on both synthetic log-linear models and large language models (Llama-3.2, Phi-3) using the Nectar dataset.

### Strengths
- First theoretical and algorithm formulation of active learning for DPO.
- Provides algorithm for both online and offline settings, enabling flexible application.
- Validate the theoretical foundation with reasonable empirical results.

### Weaknesses
- Reliance on a log-linear policy approximation may lead to shallow alignment, easy reward-hacking and neglected task complexity.
    - The paper assumes (Assumption 1) that *all policies* are log-linear in the last layer features. This assumptions is the key assumption that enables the D-optimal design analysis, but concurrently restricts the model's expressive capacity.
    - In realistic settings, such as aligning LLMs, the relation between prompt/response pairs and human judgements is likely nonlinear.
    - Therefore, ADPO may select prompts that maximize separation in the linear space, but those may correspond to shallow differences (e.g., length of the reponse).
    - This concern could be releaved by providing analysis of how *linear* each tasks are.
- Evaluation scale
    - The LLM experiments are relatively small; and conclusions about the practicality of ADPO in largs-scale PO remains unclear.
- The comparison of the Active DPO baselines in the perspective of computational overhead is needed.
I am willing to further raise the score if my concerns are addressed properly.

### Questions
- Refer to the weakness section.

### Soundness
4

### Presentation
3

### Contribution
3
