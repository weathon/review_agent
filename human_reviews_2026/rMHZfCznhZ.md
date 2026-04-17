# RLAP-CLIP: Continual Multimodal Learning with Prototype Adaptation and Difficulty-Aware Routing

- Decision: Accept (Poster)
- Scores: 8, 6, 4, 6

## Abstract
Vision-language models, such as CLIP, achieve strong zero-shot performance through contrastive pre-training but face significant challenges in class-incremental image classification scenarios. When learning new tasks sequentially, current methods suffer from degradation in prototype quality due to passive averaging and underutilize their visual adaptation capabilities. We propose RLAP-CLIP, which addresses these limitations through three components. First, Reinforcement Learning-based Prototype Optimization (RLPO) formulates prototype construction as a reinforcement learning problem to actively optimize class separability rather than relying on simple averaging. Second, difficulty-aware cross-modal fusion uses a mixture-of-experts to route samples through specialized processing pathways based on complexity. Third, dual-modal prompting balances visual and textual adaptation. Experiments on eight image classification benchmarks demonstrate consistent improvements, with RLAP-CLIP achieving average accuracy gains of 3.72-4.46 points and final accuracy improvements of 0.49-4.48 points over other methods, validating that RLAP-CLIP achieves state-of-the-art performance.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes a reinforcement learning framework for continual image classification. The method, RLAP-CLIP, uses Reinforcement Learning–based Prototype Optimization (RLPO) to actively refine class representations and mitigate prototype (representation of each class or a embedding of class) degradation. It incorporates difficulty-aware cross-modal fusion to route samples by complexity and enhanced dual-modal prompting to balance visual and textual adaptation. Experiments across multiple image classification benchmarks show RLAP-CLIP outperforms prior methods.

### Strengths
- The reinforcement learning–based prototype update method empirically outperforms prior work. Enhanced dual-modal prompting and difficulty-aware sample handling further contribute to the gains, as demonstrated in experiments.

- The paper provides empirical and theoretical evidence of training stability—covering hyperparameter sensitivity and convergence—which is critical for reinforcement learning frameworks.

### Weaknesses
- The method improves continual image classification by combining established components—prototype updates, reinforcement learning, learnable prompts, and routing. While the gains on benchmarks are clear, the pipeline is heuristic and tailored only for classification, making its generalization to other continual recognition tasks (e.g., retrieval, detection...) uncertain.

- Key terminology should be clearly defined (e.g., “prototype,” “center-based exemplar selection”) to improve readability and understanding. Only objectives and architectures are only explained in the paper. Training and inference process are not explained. Please more clarify what are frozen or learnable weights in the framework.

### Questions
- What is center-based example selection in the Figure 1. This part is not explained in the paper.

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
2

### Summary
This paper proposes RLAP-CLIP, a framework for continual multimodal learning with vision-language models (e.g., CLIP). aiming to alleviate prototype quality degradation and asymmetric adaptation in sequential task learning. The method introduces (1) RLPO, a reinforcement learning-based approach to actively optimize prototype construction for better class separability; (2) a difficulty-aware, mixture-of-experts mechanism for dynamic sample routing; and (3) dual-modal prompting to balance textual and visual adaptation. Experiments on eight datasets across diverse domains and tasks demonstrate RLAP-CLIP outperforms strong baselines, especially in fine-grained and out-of-distribution settings.

### Strengths
1. Strong Motivation and Illustration: Section 2 gives a nice layout of the prototype quality degradation issue and comparison of different prompting strategies in class separation, effectively highlighting the failure cases of existing continual learning methods for VLMs. 

2. Clear Presentation and Novel Designs: Section 3 gives a thorough introduction of different components of the RLAP-CLIP framework. Particularly, the RLPO module transforms prototype averaging into a reinforcement learning problem, with well-articulated mathematical objectives and proof. 

3. Comprehensive Experiments and Solid Results: RLAP-CLIP is benchmarked on a wide range of datasets, showing consistent improvement over state-of-the-art baselines. Additionally, a stepwise ablation is provided, attributing improvements to each module Hyperparameter sensitivity analysis is also conducted to verify the robustness of the framework.

### Weaknesses
1. Lack of Details on Prototype Policy: The paper provides some equations but is somewhat vague regarding the policy network architecture itself, such as policy hyperparameters and normalization details.

2. Limited Analysis of Scalability: While RLPO's theoretical soundness is established, discussion of potential computational bottlenecks (especially for large-scale, real-world continual learning) is lacking. For example, how does RLPO's policy network scale for hundreds/thousands of classes and when data distributions heavily shift?

3. Limited Task Scenarios: The experimental setup follows class-incremental protocols with a fixed number of exemplars per class, but there is minimal exploration of memory constraints or more severe task shift scenarios (e.g., open-world settings). Additionally, the paper could benefit from presenting qualitative failure cases or edge conditions.

### Questions
1. Can the authors clarify policy network architecture details for RLPO (exact layer sizes, normalization)? How does the policy adapt when exemplar set sizes are very small, or as class counts grow?

2. Are there task types (e.g., language-driven tasks) where visual or dual-modal prompting hurts? Figure 2 and Table 2 suggest continuous improvement, but are there more fine-grained trade-offs?

3. Could you contextualize the paper against more recent, related works that were not discussed, such as [1] which introduces a mixture-of-experts network for improved sample efficiency in visual RL, and [2], which proposes strategies for continual learning also using RL?

References:

[1] Huang, S., Zhang, Z., Liang, T., "MENTOR: Mixture-of-Experts Network with Task-Oriented Perturbation for Visual Reinforcement Learning" (2025)

[2] Liu, Z., Fu, G., Du, C., "Continual Reinforcement Learning by Planning with Online World Models" (2025).

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
3

### Summary
The paper proposes a novel reinforcement learning based framework, RLAP-CLIP, for improving vision-language models on continual image classification tasks. They introduce three different components 1) RLPO to mitigate prototype degradation in continual classification tasks, difficulty-aware cross-modal fusion for better cross-modality integration for training, and enhanced dual-modal prompting to resolve modality imbalance. The authors present results on a variety of image classification tasks to support their claims and also provide theoretical guarantees to strengthen them.

### Strengths
1. The paper is well-written and easy to follow, with each design choice for RLAP-CLIP clearly explained.
2. The authors show a comparison with a variety of past approaches that strengthen their work.
3. I also like that they clearly introduced the problem first by showing how forgetting increases in vision-language models as tasks increase and conventional averaging-based approaches are not ideal to resolve this.

### Weaknesses
1. The effect of dual modal prompting in forming better compact clusters for each class is difficult to see. Can you provide some quantitative measure of how effective the correct cluster formation is?
2. The idea of normalized advantages and comparing between intra-class and inter-class seems quite similar to GRPO [1] reward optimization. What is the novelty in RLPO, and how does it compare with this RL fine-tuning approach?
3. How do other dual prompting compare with other parameter-efficient finetuning approaches like LoRA, Prefix tuning?
4. Also, the proposed framework is limited to classification tasks, whereas other methods like C-CLIP work on a variety of different tasks, like retrieval and captioning. This limits the generalizability of RLAP-CLIP.
5. Please add some qualitative examples from the datasets used for benchmarking that help understand the advantage of RLAP-CLIP better.

[1] DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models

### Questions
Please refer to the questions raised above in the weaknesses.

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
4

### Summary
The paper introduces RLAP-CLIP, a continual multimodal learning framework that (i) replaces passive prototype averaging with reinforcement-learning-based prototype optimization (RLPO), (ii) incorporates dual-modal prompting for visual and textual inputs, and (iii) employs difficulty-aware mixture-of-experts (MoE) routing. Experiments on eight datasets show consistent improvements in both average and final accuracy over strong CLIP-based baselines, with ablation studies confirming the contribution of each component.

### Strengths
- The prototype drift argument is convincing, and the analysis figures clearly show how simple averaging causes performance degradation.

- Framing prototype construction as a reinforcement learning weighting problem, where rewards promote intra-class cohesion and inter-class separation, offers an interesting new perspective. The use of KL regularization toward a reference policy provides reasonable stability.

- The study demonstrates that visual prompts are important in continual learning and that dual-modal prompting outperforms text-only and visual-only approaches.

### Weaknesses
- RLPO introduces a policy network and reward normalization, while MoE adds routing and a deeper hard path. However, the paper does not provide a clear comparison of training time, inference latency, or FLOPs and parameter counts against the baselines under the same hardware and memory conditions, which is crucial for evaluating methods in continual learning settings.
- The paper focuses on class-incremental classification, However, it remains unclear how RLPO and MoE would perform in task-agnostic or open-world scenarios involving unknown classes, or multimodal image–text continual learning.
- The results depend on exemplar buffers of 20 samples per class. Please evaluate performance under smaller memory budgets or exemplar-free settings (e.g., with synthetic replay) to demonstrate the robustness of RLPO.

### Questions
- How do results change with 10/5/0 exemplars per class? Could RLPO operate with pseudo-exemplars (e.g., feature replay) instead of images?

### Soundness
3

### Presentation
3

### Contribution
3
