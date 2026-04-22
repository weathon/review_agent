# Value-aligned World Model Regularization for Model-based Reinforcement Learning

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
Model-based reinforcement learning (MBRL) aims to construct world models for imagined interactions to enable efficient sampling. Based on training strategy, current mainstream algorithms can be categorized into two types: maximum likelihood and value-aware world models. The former adopts structured Recurrent/Transformer State-Space Models (RSSM/TSSM) to capture environmental dynamics but may overlook task-relevant features. The latter focuses on decision-critical states by minimizing one-step value evaluations, but it often obtains sub-optimal performance and is difficult to scale. Recent work has attempted to integrate these approaches by leveraging the strong priors of pre-trained large models, though at the cost of increased computational complexity. In this work, we focus on combining these two approaches with minimal modifications. We empirically demonstrate that the key to their integration lies in: RSSM/TSSM ensuring the lower bound of the world model, while value awareness enhances the upper bound. To this end, we introduce a value-alignment regularization term into the maximum likelihood world model learning, promoting task-aware feature reconstruction while modeling the stochastic dynamics. To stabilize training, we propose a warm-up phase and an adaptive weight mechanism for value-representation balance. Extensive experiments across 46 environments from the Atari 100k and DeepMind Control Suite benchmarks, covering both continuous and discrete action control tasks with visual and proprioceptive vector inputs, show that our algorithm consistently boosts existing MBRL methods performance and convergence speed with minimal additional code and computational complexity.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper explores incorporating a value regularization loss into world model learning. 
Value regularization encourages the learned model to align predicted states with their corresponding value estimates, helping it focus on task-relevant features.
Albeit simple, it demonstrates substantial improvement on the Atari 100k benchmark (and slight improvement on DMC) when applied to DreamerV3 and STORM.

### Strengths
**S1. Simplicity of the method**
* The proposed modification is conceptually simple and easy to implement; it adds a KL-based regularization term on the value during model learning (after a warm-up phase). 
* The method can be seamlessly integrated into existing model-based RL frameworks (such as DreamerV3 and STORM).

**S2. Substantial improvement and interpretability**
* Despite its simplicity, the proposed method yields significant improvements in the Atari 100k benchmark (e.g., HNS mean: 1.1 -> 1.34 for DreamerV3 and 1.14 -> 1.36 for STORM). 
* Visualization results show that the world model focuses more on task-relevant objects, supporting the claim that value regularization enhances the quality of the representation.

### Weaknesses
**W1. Experimental results for the baselines are different (worse) from the original paper**
* In the paper, DreamerV3 and STORM scores 1.10 and 1.14, while the score in the original paper is 1.25 and 1.27 (HNS mean in Atari 100k); DreamerV3 scores 817, 827, while the score in the original paper is 871, 861 (Task mean).
* While authors mentioned that they used torch version of DreamerV3, but I wonder the implementation correctly reproduces DreamerV3, as there is substantial gap in the performance (I tried to take a look on the code, but I couldn’t find the DreamerV3 code in the supplementary materials).
* Moreover, while I assume that STORM is ran with the official implementation (as the paper builds on top of official codebase of STORM), but still shows a large discrepancy.

### Questions
**Q1. Clarification on the results**
* Why the numbers are different from the original DreamerV3 and STORM paper? 
Could the authors clarify whether differences in architecture, hyperparameters, or experimental setup could explain this discrepancy?
* It would also help if the DreamerV3 code used for experiments were made available in the supplementary materials.

**Q2. Experiments with distracting dimensions**
* For DMC, all state dimensions are task-relevant, so value regularization might not strongly affect performance. 
* How about adding distracting or irrelevant dimensions (as in [1]) to test whether the proposed method helps the model focus on the necessary states? 
* This could provide stronger evidence for the intended mechanism.

**Q3. Plot of reconstruction loss with value regularization**
* I think it would be interesting to analyze how value regularization influences the reconstruction loss. 
* Specifically, does the improvement come from 1) higher overall reconstruction loss but better value-relevant fidelity, or 2) lower overall reconstruction loss due to improved learning signals?

* Although not critical, such an analysis would clarify how value regularization benefits model learning.

PLEASE READ: I would like to emphasize that I believe this paper is strong and has substantial potential. 
However, **Q1 (W1) is a critical concern** for me. I currently lean toward a weak reject, but if the authors can provide the clarification, I'm inclined to significantly raise my recommendation.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a method to integrate value-alignment into world model training for model-based RL. The key idea is adding a value-alignment regularization term (Var) to the world model loss, using a KL divergence between the value function's distribution on real and imagined states. The authors argue this combines the stability of maximum likelihood methods (lower bound) with the decision-awareness of value-aware methods (upper bound), and introduce a warmup phase and an adaptive weighting mechanism to stabilize the training. The idea is evaluated on Atari100k and DMC, showing consistent improvements when added to DreamerV3 and STORM.

### Strengths
+ the integration of value-alignment into world model training as a KL divergence term on the value network's distributional output is a simple, elegant, and easy-to-implement idea
+ the claim and validation of "RSSM/TSSM ensures the lower bound... while value awareness enhances the upper bound" offers new insights and understanding
+ validation across visual and proprioceptive as well as discrete and continuous control tasks is strong

### Weaknesses
- The exact definition of lower bound and upper bound is missing and makes the claim vague.
- the proposed method should have been compared against modern value-aware methods such as VaGraM or CVAML across all tasks
- the failures of the proposed method on e.g., PrivateEye for DreamerV3+Var, Frostbite are not analyzed

### Questions
1) How does the method conceptually and empirically differ from prior value gradient weighting methods like VaGraM, beyond the obvious architectural differences (RSSM/TSSM vs RNN)? 
2) The lower/upper bounds concept is central to your motivation. Can you provide a more formal definition or a toy example that illustrates this concept?
3) Have you tried a direct, large-scale comparison? The proposed method regularizes the world model to produce states yielding a similar distribution to real states. Will it lead to delusional states that are easy for the value function to predict but are world model dynamically inconsistent?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes Value-aligned World Model Regularization (VAR), which augments a maximum-likelihood world model loss with a KL term between the critic’s intermediate value distribution at real vs. predicted latent states. The method includes a warm-up schedule and an adaptive weighting scheme. Experiments on Atari-100k and DeepMind Control suggest consistent gains over DreamerV3 and STORM, with minimal implementation overhead.

### Strengths
- The experiments are extensive and well presented, covering both discrete and continuous control, along with useful ablations on weighting and computational overhead.
- The motivation for “value-aligned world models” is clear, and the concept is explained effectively.
- The paper is generally well written and clearly structured.

### Weaknesses
- The paper is heavily centered around previous work (Dreamer-v3) which is totally fine. However to my knowledge, the authors build upon a false assumption that questions a lot of the paper content. By the author's own description of "value aware world models", Dreamer-v3 is indeed a value aware/aligned world model and not just a "classic maximum likelihood" WM. In the original Dreamer-v3 paper (https://arxiv.org/pdf/2301.04104, page 6, first Paragraph) the value alignment is described and can also be found in the original implementation (https://github.com/danijar/dreamerv3, see Agent.py line 219-235 (replay value loss).

- The contribution "we propose Value-aligned World Model, a novel and effective MBRL algorithm" is overstated. Based on my understanding of DreamerV3, the contribution seems to be "yet another way to do value alignment".

- The authors's experiment results for Dreamer-v3 (Table 1) differ significantly from the original paper (https://arxiv.org/pdf/2301.04104, Table 9) in which the models are indeed value-aligned. This seems **not** like a "fair comparison" (see line 371).

- In Appendix, policy entropy is higher than the original, yet claimed to be "consistent" (see Appendix) and "identical" (line 372).

- Figure 1. b) is not discussed or suited w.r.t experiments.

- Equation 4: It is unclear what $\tilde s_t$ is. There is no definition provided.

- The paper repeatedly claims RSSM/TSSM ensures a lower bound while value awareness lifts the upper bound (of what? performance?), but provides no formal definition or guarantee. This reads as metaphor rather than theorem.

- the paper reiterates the Dreamer-v3 theory. It also adds small modifications for, I suspect, "novelty" (e.g. $s_t = (h_t, z_t)$) that seem unnecessary.

### Questions
It would be great if the author could address the major weakness I outlined above.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses a common limitation in model-based reinforcement learning, where world models trained via maximum likelihood on reconstruction objectives often expend model capacity on pixel-level details that are irrelevant to the downstream task. This can lead to state representations full of redundant information.

The authors propose a value-alignment regularization term that is added to the standard MLE objective. This new loss term is designed to make the world model task-aware by aligning its learned representations with the agent’s value function. Specifically, the regularization loss minimizes the KL divergence between the value predicted from the model's imagined latent state and the value predicted from the latent state inferred from observation. By integrating this value-based loss directly into the world model's training, the model is explicitly encouraged to prioritize learning and representing state features that are useful for predicting future returns, rather than focusing solely on reconstructing all sensory details. This additional loss term can be added to any supervised model learning problem within an RL setting.

### Strengths
- interesting idea with easy integration into different maximum likelihood MBRL approaches, possibly contributing to a better understanding of what aspects of environment interactions / agent experience carry meaningful information for good world models
- helps with task-specific world models by forcing the world model to align with the actual problem (potentially filtering out unnecessary information to solve the task)
- experiments show improved performance on most tasks

### Weaknesses
- the approach seems very similar to the representational learning approach in MuDreamer. to judge the originality of the present approach it would be important to know how the authors position themselves wrt this prior work.
- not well motivated: why is the choice of regularization, namely matching imagined values vs. grounded values, reasonable? The analogy to “perceptual losses” in computer vision is mentioned but not explained.
- warm-up seems a bit arbitrary, adaptive weight mechanisms to prevent instabilities during early training introduces additional hyper-parameters
- missing analysis of the actual learnt representations, mostly performance tests. Figure 3 tries to achieve this but leaves a lot of room for interpretation. Focus was on performance on the benchmark task, not on understanding the representational implications of the approach.
- Figure 2 compares performance for a) different model sizes and b) different observation sizes but only for 100k steps. The results may be interpreted as the proposed approach being more sample efficient but none of the models are trained to full convergence, making statements about performance (model captures dynamics better / worse) inconclusive.
- Figure 1 does not help much for understanding

### Questions
- performance improvements seem much more pronounced in the case of ATARI, but a lot weaker for DMC. Why is that? This raises questions about the general applicability of the approach
- How does $\beta_{var}$  behave over time? Do the two paradigms align or are they counteracting each other, resulting in alternating optimization between maximum likelihood optimization and value alignment?
- Representation power? do the learned representations actually align with the task? something clearer than Fig 3
 How does this approach compare to MuDreamer, which seems to follow a similar idea?

### Soundness
2

### Presentation
2

### Contribution
2
