# CRPA: Curriculum-driven Reinforcement Pre-Alignment for Domain-Adaptive Vision-Language Models

- Decision: Reject
- Scores: 6, 4, 6, 4

## Abstract
Vision-Language Models (VLMs) demonstrate remarkable general-purpose capabilities but often fall short in specialized domains such as medical imaging or geometric problem-solving. Supervised Fine-Tuning (SFT) can enhance performance within a target domain, but it typically causes catastrophic forgetting, limiting its utility as a general AI agent. The central challenge, therefore, is to adapt VLMs to new domains while preserving their general-purpose capabilities.
Continual pretraining is effective for expanding knowledge in Large Language Models (LLMs), but it is less feasible for VLMs due to prohibitive computational costs and the unavailability of pretraining data for most open-source models. This necessitates efficient post-training adaptation methods. Reinforcement learning (RL)–based approaches such as Group Relative Policy Optimization (GRPO) have shown promise in preserving general abilities, yet they often fail in domain adaptation scenarios where the model initially lacks sufficient domain knowledge, leading to optimization collapse.
To bridge this gap, we propose Curriculum-driven Reinforcement Pre-Alignment (CRPA), a novel post-training paradigm that introduces a curriculum-aware progressive modulation mechanism. In the early phase, CRPA applies partial output constraints to safely expose the model to new domain concepts. As the model’s domain familiarity increases, training gradually transitions to full generation optimization, refining responses and aligning them with domain-specific preferences. This staged adaptation balances domain knowledge acquisition with the preservation of general multimodal capabilities.
Extensive experiments across specialized domains (e.g., OpenI for medical imaging and Geo170K for geometry) and general benchmarks (e.g., COCO Captions) validate the effectiveness of CRPA. Results show that CRPA achieves domain-specific performance competitive with SFT while significantly outperforming SFT in retaining general multimodal understanding, establishing a practical pathway toward building high-performing, domain-adaptive VLMs.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes Curriculum-Driven Reinforcement Pre-Alignment (CRPA), a post-training framework for domain adaptation of Vision-Language Models (VLMs). CRPA seeks to balance domain knowledge acquisition and general multimodal capability retention by progressively transitioning from constrained imitation learning to full reinforcement alignment. The key modules are: (1) Curriculum Progress Perception (CPP), which gradually adjusts reward thresholds and injects partial answer prefixes to bootstrap stable learning signals, and (2) Curriculum Difficulty Perception (CDP), which reweights training samples based on difficulty to enhance efficiency and avoid overfitting.

### Strengths
**Good Motivation:** The paper clearly identifies a relevant challenge in VLM domain adaptation—catastrophic forgetting vs. domain transfer—and provides a systematic approach to address it.

**Comprehensive Evaluation:** Evaluations across multiple domains (captioning, geometry, medical imaging) and both domain-specific and general benchmarks support the main claims.

### Weaknesses
**Incremental Technical Novelty:** While CRPA integrates curriculum learning with GRPO effectively, the novelty is mainly procedural—combining well-known techniques (progressive constraints, dynamic thresholds, sample reweighting). There is no fundamentally new algorithmic insight or theoretical development.

**Limited Theoretical Analysis:** The paper lacks a rigorous explanation of why the proposed curriculum mechanisms (CPP and CDP) systematically improve optimization stability or generalization.


**Overreliance on GRPO Backbone:** Most of CRPA’s improvements may stem from careful reward engineering or training schedules within GRPO, rather than an inherently new framework. It is unclear whether these benefits generalize beyond this specific RL setup.

**Missing Ablation Study:** There is no quantitative breakdown of computation cost or convergence time. Since CRPA adds multiple stages and reward evaluations, its training efficiency and scalability remain unclear.

### Questions
Please refer to the weakness section.

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
3

### Summary
This work propose Curriculum-driven Reinforcement Pre-Alignment (CRPA), a post-training paradigm that intro- duces a curriculum-aware progressive modulation mechanism, which gradually shift the training from the specific target data to preserving general abilities, such that  the catastrophic forgetting can be addressed.

### Strengths
- the idea is simple and easy to follow
- the task is important as re-training of foundation models is generally impractical

### Weaknesses
__Major Concerns:__
- the algorithmic design is based on pure heuristic (e.g., Eq.(3, 4, 5), $w$) and highly dependent on the accuracy of the similarity measurement.
- $q$ in Eq.(6) does not appear in Algorithm 1; what is it exactly?
- too much hyper-parameters, including $S, \delta_{min}, \delta_{max}, \sigma, \alpha, \beta, \epsilon, G, \gamma$, offset,  which makes it hard to deploy in real world problems without enough validation data; besides, the robustness against hyper-parameters is not provided.
- how is General-Purpose Ability measured ?
- the task is limited to VQA, where the actual performance is not straightforward by just displaying bunch of scores; experiments on some DA benchmarks on object recognition are highly encouraged to make the results more convincing.
- FFT is too naive to demonstrate the effectiveness of the proposal, since in the literature of continual learning, there are many techniques to relieve catastrophic forgetting; for instance, [1,2,3] use prompt learning, normalization layer update and source knowledge calibration to regularize the training on target data, some even tackle a more general problem where the target domain continuously shifts. 

__Minor Concerns:__
- the table is hard to read; it would be better to highlight the best and second.
- the code cannot be accessed from the link
- no details in appendix

***
[1] Test-time prompt adaptation for vision language models, NeurIPS 2024

[2] Towards Stable Test-Time Adaptation in Dynamic Wild World, ICLR 2023

[3] A Versatile Framework for Continual Test-Time Domain Adaptation: Balancing Discriminability and Generalizability, CVPR 2024

### Questions
see above

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors want to solve the collapse of GRPO, when the model lacks knowledge of a specific domain. So they propose CRPA, a staged alg that progressively use a curriculum learning method to solve that and backed by experiment results.

### Strengths
1. Experiments breadth and ablations are sufficient for the method to be convincing
2. They are trying to solve a long-standing problem that people talk about, and the method is promising

### Weaknesses
1. It’s not always clear whether all baselines had equivalent compute steps, parameter budgets, or similar hyperparameter tuning.
2. It would help to know where CRPA fails.

### Questions
1. Due to its 2-phase nature, how does the computational efficiency look like compared with other methods in the table?
2. What are the limitations of this method, despite promising results?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents CRPA, a framework to adapt Vision-Language Models (VLMs) to new domains. It addresses the problem that standard fine-tuning can cause models to forget general skills, while reinforcement learning can fail if the model has no starting knowledge of the new domain.
CRPA works in two stages:
Pre-Alignment: Initially, the model is given parts of the correct answer (prefixes) to guide its learning and make the task easier.
Reinforcement Alignment: As the model learns, the prefixes are gradually removed, and training transitions to a standard reinforcement learning method where the model generates full answers based on reward signals.

### Strengths
This paper is well-written and easy to follow.  In addition, the motivation is very clear.

This paper proposed an RL training recipe to mitigate the forgetting issue (which is usually severe in SFT) while enhance model performance on new (target) tasks.  The idea of this method is promising.

### Weaknesses
"While SFT effectively enables the injection of domain-specific knowledge via imitation learning, it relies exclusively on supervised labels. As a result, it is prone to catastrophic forgetting: previously acquired capabilities are overwritten by specialized
information, thereby undermining the generalization of VLMs.". It's somehow common sense that SFT is prone to cause forgetting problems, but this cause is not: SFT requires supervised labels. If SFT data is sampled from the pretrain distribution, it will not easily cause forgetting. An important component to stabilize VLM's original performance is the KL divergence loss, so it could also test SFT+KL's performance. 

I have a concern about the experiment setting on GRPO and DAPO baselines. Do these methods directly perform RL training from the same checkpoint with the proposed method? An common practice is that SFT (cold start) first then RL. Directly RL is easy to train a "bad" model. 

In addition, the proposed method use answers for prefix, if DAPO and GRPO is RL without cold start, I think the comparison is unfair.

### Questions
-

### Soundness
3

### Presentation
3

### Contribution
3
