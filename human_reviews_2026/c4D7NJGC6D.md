# Contextual Latent World Models for Offline Meta Reinforcement Learning

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 4, 6

## Abstract
Offline meta-reinforcement learning seeks to overcome the challenges of poor generalization and expensive data collection by leveraging datasets for related tasks. Context encoding is a prevalent approach, where an encoder maps transition histories to a task representation. In parallel, latent world models -- which map observations into temporally consistent latent spaces -- advanced self-supervised representation learning for planning and policy optimization. In this work, we unify these directions by introducing contextual latent world models: world models conditioned on the task representation and trained jointly with the context encoder. Coupling task inference with predictive modeling yields task representations that capture variation factors across tasks and empirically improves generalization to out-of-distribution tasks in diverse benchmarks, including MuJoCo, Contextual-DeepMind Control suite, and Meta-World.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces C-DCWM, an offline model-based meta-reinforcement learning algorithm. The method utilizes a context-based approach, training a task encoder with an InfoNCE loss. It incorporates the learned task representations into the Discrete Codebook World Model (DCWM), extending it to the multi-task setting. The paper demonstrates that C-DCWM achieves superior performance compared to existing offline meta-rl baselines on the MuJoCo, Contextual-DMC, and Meta-World benchmarks.

### Strengths
1. The study of world models capable of generalizing across multiple tasks is of significant practical importance, as it promises to substantially advance the real-world application of reinforcement learning.
2. The authors provide extensive experiments across multiple benchmarks (MuJoCo, Contextual-DMC, and Meta-World) , demonstrating the method's strong empirical performance against several baselines.

### Weaknesses
1. The proposed method appears to be a relatively straightforward extension of the DCWM frameworkto the multi-task setting by conditioning it on a task representation. This potentially limits the technical novelty of this work.
2. The evaluation of generalization is limited. The out-of-distribution experiments primarily focus on parametric variations within the same task families. A more challenging and practically relevant OOD evaluation, such as generalization to unseen tasks in Meta-World, is notably absent.
3. The paper lacks a comparative analysis of the computational complexity or overhead (e.g., parameter count, training time) relative to the baseline methods.

### Questions
1. The current experiments (e.g., in MuJoC) primarily demonstrate OOD generalization to **new parameters of the same task family**. Does the model also exhibit **cross-task** generalization? For example, if the model is trained on a diverse set of tasks from Meta-World (e.g., 'door-open', 'window-close' et al. without ''window-open'), could it generalize (few-shot or zero-shot) to a completely **unseen** task, such as 'window-open'?
2. Model-based methods typically have a larger parameter count compared to model-free method. Can authors provide analysis (e.g., an ablation study or parameter comparison) to confirm that the superior performance of C-DCWM stems from the effectiveness of the contextual world model itself, and not merely from an increased number of trainable parameters compared to the baselines?  
- I will raise my score if my concerns above are addressed.

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper presents Contextual Discrete Codebook World Models (C-DCWM) for offline meta-reinforcement learning. It jointly learns (i) a context encoder that infers a task representation z from short histories, and (ii) a latent world model conditioned on z, built with a finite scalar quantization (FSQ) module and trained via cross-entropy-based temporal consistency and contrastive InfoNCE objectives. The latent representations are then used to train policies and critics (IQL) in the quantized latent space. Experiments on MuJoCo, Contextual-DMC, and Meta-World benchmarks show improved in-distribution and out-of-distribution performance compared to several existing meta-RL baselines.

### Strengths
1. The integration of context-conditioned latent world models is clearly implemented and experimentally well-controlled.  

2. Empirical performance across multiple offline meta-RL benchmarks is consistent, showing that classification-based temporal consistency can outperform standard regression losses.

### Weaknesses
1. Limited novelty. The approach mainly combines existing ingredients, e.g., context encoders from meta-RL, discrete latent world models from Dreamer/TD-MPC, and contrastive representation learning, into a joint training scheme. While the integration is clean, the conceptual advance over prior work such as CSRO, UNICORN, and discrete-latent Dreamer variants is marginal. 

2. Overreliance on IQL head. The offline RL component is fixed to IQL; no evidence is provided that the representation benefits generalize across different offline learners (e.g., CQL, TD3+BC). 

3. Missing comparison to contemporary discrete-latent models. Direct comparison to recent planning-based or discrete-latent world models (e.g., Dreamer-V3-Discrete, TD-MPC2 variants) is absent.

### Questions
None.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a novel offline meta-RL method called Contextual Discrete Codebook World Models (C-DCWM). C-DCWM encodes offline datasets from different tasks into task related representations. The world model is then conditioned on different learned representations. Experiments demonstrate that jointly training the world model and the context encoder leads to improved generalization performance.

### Strengths
1.The paper is clearly written and easy to follow.
2.The idea of extending DCWM to a conditioned version for generalizing across different tasks, and jointly training the model to obtain better task representations, is novel and well-motivated.
3.The authors have conducted comprehensive experiments and ablation studies to verify and analyze the effectiveness of C-DCWM.

### Weaknesses
1. While the current experimental validation is primarily conducted on continuous control tasks, these environments typically feature consistent dynamics within each task, which likely simplifies the learning of a discrete codebook. To further substantiate the generalizability of the proposed method, it would be compelling to evaluate it on more heterogeneous domains, such as the Atari benchmark. In these environments, a single task (or game) often comprises distinct levels requiring diverse policies and decision-making skills, presenting a more significant challenge for codebook learning. Extending the analysis to such tasks would greatly strengthen the empirical evidence for the method's robustness and effectiveness.
2. some figures and tables are placed far from where they are referenced. For example, Figure 1 is distant from Section 3, which makes it inconvenient to cross-reference the figure with the corresponding formulas and explanations.

### Questions
1.The placement of the figures and tables could be reorganized for better readability. For instance, Figure 1 might be placed after current Figure 2.
2. As I understand it, to generalize the policy to an out-of-distribution task, C-DCWM requires a dataset from that task to compute the task representation. However, this assumption may be unrealistic, as we may not always have access to a dataset for an unknown task. Would it be possible instead to compute $z$ in an online manner—for example, starting from an initial $z$ and updating the task representation after each interaction step?

### Soundness
3

### Presentation
3

### Contribution
3
