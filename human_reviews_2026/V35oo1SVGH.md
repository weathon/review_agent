# VITA: Zero-Shot Value Functions via Test-Time Adaptation of Vision–Language Models

- Decision: Accept (Poster)
- Scores: 6, 6, 4

## Abstract
Vision–Language Models (VLMs) show promise as zero-shot goal-conditioned value functions, but their frozen pre-trained representations limit generalization and temporal reasoning. We introduce VITA, a zero-shot value function learning method that enhances both capabilities via test-time adaptation. At inference, a lightweight adaptation module is updated via a gradient step on a meta-learned self-supervised loss, such that each test-time update improves value estimation. By updating sequentially over a trajectory, VITA encodes history into its parameters, addressing the temporal reasoning limitations. To mitigate shortcut learning, we propose a dissimilarity-based sampling strategy that selects semantically diverse segments of the trajectory during training. In real-world robotic manipulation tasks, VITA generalizes from a single training environment to diverse out-of-distribution tasks, environments, and embodiments, outperforming the state-of-the-art zero-shot method using autoregressive VLMs. Furthermore, we demonstrate that VITA’s zero-shot value estimates can be utilized for reward shaping in offline reinforcement learning, resulting in multi-task policies on the Meta-World benchmark that exceed the performance of those trained with the simulation’s fuzzy-logic dense rewards. Project website: https://chziakas.github.io/vita/.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work proposes using Test-time adaptation modules to improve language-goal conditioned value estimation. Experiments demonstrate that it performs better than GVL (a prompting method) and other VLM based CLIP reward functions.

### Strengths
* the paper does a good job explaining TTT
* the idea is rather intuitive -- by gaining more context about the current setting we can produce a better value estimate.
* The paper evaluates both VOC and also running offline RL using the value estimates as reward.
* the data weighting scheme for diverse sampling makes sense.
* the paper performs well on the bridge data setting in comparison to GVL.

### Weaknesses
* The paper could benefit substantially from a more detailed figure showing the overall architecture (concatenating clip latents), then how each of the projection matrices are used for TTT, and then how the final observation in the window is used for estimating the value function. I think this would help clarify the total overall flow.
* The paper does not consider other value learning baselines like LIV (ma et al.) which consider langauge and goal representations are also based on CLIP. Generally the paper seems to lack a number of baselines which actually train on the target data like the proposed TTT method. In this regard, LIV seems like a rather natural baseline.
* MetaWorld experiments where the scripted expert demonstrations are used are generally weak as most tasks in the benchmark can be solved by BC on ~10-20 scripted demos! So I'm not sure how significant the policy learning results are (it might be good to include a BC baseline, since IQL is basically weighting the data by the value fn).

### Questions
* Just to clarify, TTT is run on each latent observation (concat of visual and lang embed from clip) within a window, and the value function is predicted after the final observation post adaptation.

### Soundness
3

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
3

### Summary
This paper proposes VITA, a novel method for zero-shot goal-conditioned value function learning through test-time adaptation of vision-language models. The key contributions include: (1) A test-time adaptation framework that enhances both generalization and temporal reasoning capabilities of contrastive VLMs; (2) A dissimilarity-based sampling strategy to mitigate shortcut learning; (3) Comprehensive evaluation showing state-of-the-art performance on real-world robotic manipulation tasks under various distribution shifts, and successful application to offline RL reward shaping in simulated environments.

### Strengths
1. The integration of test-time adaptation for temporal reasoning in zero-shot value functions is novel and creative;
2. Extensive experiments across real-world and simulated environments demonstrate robust performance;
3. Addresses important limitations of current VLM-based value functions and enables better generalization without task-specific fine-tuning.

### Weaknesses
1. The computational overhead of per-timestep adaptation may limit real-time applicability in some robotic systems;
2. Lack of comparison with model baselines possessing explicit memory capabilities.

### Questions
The authors claim that their method has the advantage of "implicit memory through parameter updates." Could we propose the most direct comparative baseline, which would be using an explicitly memory-equipped network (e.g., RNN, Transformers) to explicitly form parameterized memory?

### Soundness
3

### Presentation
2

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
This paper introduces VITA, a zero-shot goal-conditioned value function learning framework designed to enhance generalization and temporal reasoning through test-time adaptation (TTT).
The core idea is to integrate a lightweight adaptation module that is updated online via a meta-learned self-supervised loss. Each adaptation step refines value estimation, enabling the model to implicitly capture temporal dependencies within its parameters. To avoid shortcut learning, the authors propose a dissimilarity-based sampling strategy that encourages semantic diversity during training.

Empirically, VITA achieves state-of-the-art performance on real-world robotic manipulation tasks, outperforming both autoregressive zero-shot VLMs (e.g., GVL) and CLIP-based baselines. Moreover, on the Meta-World MT10 benchmark, VITA demonstrates effective zero-shot reward shaping, even surpassing the simulator’s fuzzy-logic dense rewards.

### Strengths
1. The paper proposes an implicit memory formulation that effectively incorporates temporal modeling into value estimation. In addition, the proposed dissimilarity-based sampling method mitigates overfitting to chronological shortcuts—a nontrivial improvement over prior work.

2. Experimental results demonstrate that ViTA consistently and substantially outperforms existing baselines across diverse tasks and domains.

### Weaknesses
1. While the proposed dissimilarity-based sampling method is conceptually interesting, the paper lacks a formal theoretical explanation or empirical justification for why this approach effectively mitigates shortcut learning. Providing analytical or visual evidence would strengthen the technical soundness of the claim.

2. Key hyperparameters (e.g., learning rate, adaptation steps) are only mentioned as “selected from a sweep,” leaving uncertainty about model sensitivity.

3. The paper does not clarify why variants such as TTT-TR (same structure, fewer updates) perform worse than baselines.

4. Some formatting inconsistencies remain (e.g., inconsistent capitalization of Lself).

### Questions
1. TTT Variants Explanation: In Table 5, VITA outperforms both TTT-TR and TTT-RS, despite all models sharing the same network architecture. This observation raises several questions:
1.1 Could the authors clarify why the single-trajectory update variant (TTT-TR) performs worse than the baseline?
1.2 Were all training hyperparameters (e.g., learning rate, adaptation steps, λ₍self₎) kept consistent across these variants?
1.3 Have the authors explored different adaptation frequencies (e.g., updating every n steps) to evaluate how temporal update intervals influence performance?

2. Online RL Integration: How would VITA perform in an online reinforcement learning setting, where the agent interacts with the environment continuously and receives real-time feedback? Discussing or experimenting with this setup could help demonstrate the method’s robustness and adaptability beyond offline contexts.

### Soundness
3

### Presentation
2

### Contribution
2
