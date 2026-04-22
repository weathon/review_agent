# STAR: Similarity-guided Teacher-Assisted Refinement for Super-Tiny Function Calling Models

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 2, 6, 8, 6

## Abstract
The proliferation of Large Language Models (LLMs) in function calling is pivotal for creating advanced AI agents, yet their large scale hinders widespread adoption, necessitating transferring their capabilities into smaller ones. However, existing paradigms are often plagued by overfitting, training instability, ineffective binary rewards for multi-solution tasks, and the difficulty of synergizing techniques. We introduce STAR: Similarity-guided Teacher-Assisted Refinement, a novel holistic framework that effectively transfers LLMs' capabilities to super-tiny models. STAR consists of two core technical innovations: (1) Constrained Knowledge Distillation (CKD), a training objective that augments top-k forward KL divergence to suppress confidently incorrect predictions, ensuring training stability while preserving exploration capacity for downstream RL. STAR holistically synergizes these strategies within a cohesive training curriculum, enabling super-tiny models to achieve exceptional performance on complex function calling tasks; (2) Similarity-guided RL (Sim-RL), a RL mechanism that introduces a fine-grained, similarity-based reward. This provides a robust, continuous, and rich signal for better policy optimization by evaluating the similarity between generated outputs and the ground truth. Extensive experiments on challenging and renowned benchmarks demonstrate the effectiveness of our method. Our STAR models establish SOTA in their size classes, significantly outperforming baselines. Remarkably, our 0.6B STAR model achieves the best performance among all open models under 1B, surpassing even several well-known open models at a larger scale. STAR demonstrates a training framework that distills capabilities of LLMs into super-tiny models, paving the way for powerful, accessible, and efficient AI agents.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper explores training extremely small language models for function calling. The authors introduce the STAR framework, which extends the traditional Knowledge Distillation (KD) + Reinforcement Learning (RL) pipeline with two innovations. First, Constrained Knowledge Distillation (CKD) mitigates key shortcomings of existing KD methods. Second, during the RL phase, SimRL employs a similarity-based reward function aligned with ground-truth outputs. Through evaluations on standard benchmarks, the authors demonstrate that STAR models surpass state-of-the-art KD and RL approaches within the super-tiny model regime.

### Strengths
Strengths:

1. The paper is well written and easy to follow, the experimental results are also well presented.

2. The CKD loss is well motivated, and the gradient analysis is particularly insightful.

3. The final performance of the STAR models are impressive.

4. The paper shows that current SOTA  RL / distillation methods struggle in the super tiny models regime, which is interesting.

### Weaknesses
The two main contributions of this paper are 1) the CKD loss function and 2) the sim-rl reward function, and the impact of both has not been well studied.

1. In Table 4, CKD’s performance does not appear significantly stronger than other loss functions. For instance, CKD outperforms AKL by only a small margin (~0.1–0.5) on the BFCLv3 benchmark but performs worse than AKL on ACEBench without RL. Were the improvements of CKD tested for statistical significance?

2. The proposed reward function has not been evaluated against prior approaches—for instance, the similarity-based reward function in ToolRL or other variants based on ASTs and PRMs. As a result, its effectiveness remains unvalidated.

3. There is no ablation on the effect of distilling from a teacher; what if STAR is applied to D directly?

4. (Line 418) “standard metrics are unreliable..” – what are the standard metrics? I also didn’t follow how sim-rl is better suited to handling the stochasticity of the teacher?

5. (Minor) Since CKD is applied first in the pipeline, the paper reads better if it’s introduced first.

6. (Minor) A lot of citations are missing a space after the text, for ex. Line 38: “function calling(Patil et al., 2024; Jin et al., 2025)”

### Questions
1. Is the improvement of CWD on baselines (such as AKL) statistically significant?

2. How does the Sim-RL reward function compare to other reward functions in the literature (ex: ToolRL)?

3. What is the impact of using the teacher’s generations D^T v/s D?

4. In KD, the asymmetry of the divergence seems to be leading to poor performance. What if you replace it with a symmetric divergence? Ex: Jensen Shannon

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces a training pipeline to adapt tiny-scale language models for function calling. Specifically, the introduced pipeline is composed of a RL method, Sim-RL, with rewards crafted based on the similarity between generated function calls against the ground truth and a knowledge distillation method, Constrained Knowledge Distillation (CKD), that uses forward KL-divergence as the divergence metrics with an additional component to penalize the confident-but-wrong tokens. The combination of these two approaches, STAR, shows the best performance among models under 1B and closes the performance gap with larger models on two benchmarks.

### Strengths
- The paper is relatively well-written and easy to follow.
- The performance gains on the 0.6B model scale is consistent over most metrics in the two evaluation benchmarks, and the performance gap with larger is significantly reduced.
- The paper includes discussion on the comparison between KD+RL and SFT+RL besides the empirical results that might be insightful for future work.
- The paper includes analysis on the comparison among different KD methods.

### Weaknesses
- Sim-RL looks highly dependent on the generated function calls’ format that the author defines based on the Qwen tool calling template. It might be important to show the generalizability of this method for alternative formats. 
- STAR requires RL training on the teacher model, which introduces additional non-trivial compute cost compared to SFT+RL.
- More potential analysis studies could improve the persuasiveness of the paper in showing the advantages of CKD over SFT. For example, a comparison between these methods in a larger model (e.g. 1.7B), an ablation on the teacher model’s size, etc.

### Questions
- It seems that the methodology of CKD is not specific to the task of generating function calls. Has the author considered applying this method to other tasks? If not, what constrains CKD to this specific task?
- The paper lacks some explanations on the categories of the benchmarks shown in Table 1 and 2.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposed a framework to effectively imbue tiny-LMs with tool calling capabilities of their larger models. The proposed method consists of two stages: (1) Constrained Knowledge Distillation to prevent highly confident incorrect predictions by the student model, and (2) a similarity-based RL refinement on top of the trained student that computes a fine-grained similarity-based reward between the ground-truth functional calls and the model prediction (more suitable for multi solutions problems). The authors verify their approach by comparing it to several baselines evaluated on BFCL and Acebench (for testing generalization). Results showed improved performance across all benchmarks with the more profound boost on Acebench showing better generalization to unseen function call formats.

### Strengths
- The paper is generally well-written and well motivated
- Experimental results are promising and show strong generalization compared to baselines
- The method seems to simple and effective at mitigating overfitting problem especially when compared to conventional approach of SFT+RL
- Results on closing the performance gap with much stronger models in Table 3 is pretty interesting
- Also appreciate additional theoretical explanation for their Top-K truncation with FK

### Weaknesses
- There’s no direct comparison with existing RL-based methods. It’s not clear how the proposed reward is different from those proposed in related prior works for example one in [1]. In general more comparison with existing RL rewards would be nice. (see more in questions)

[1] Anna Goldie, Azalia Mirhoseini, Hao Zhou, Irene Cai, and Christopher D Manning. Synthetic
data generation and multi-step reinforcement learning for reasoning and tool use.

### Questions
1- Line 057: typo: the -> the

2- Minor suggestion for structuring Sec 3: chronologically, it would have made sense to start with distillation and then talk about refinement (sim-RL)

3- [line 106] Add citation for RLVR: Lambert, Nathan, et al. 2025. “Tulu 3: Pushing Frontiers in Open Language Model Post-Training.” In Proceedings of the Second Conference on Language Modeling

4- [Line 270]: If I understand correctly, you have two iterations of Sim-RL? you refine both the teacher and the student using Sim-RL.  It would make sense to try to clarify this both in text and figure to avoid confusion.

5- Baslines: which baseline is representing a simple binary RL reward? This is specially important and relevant to you analysis section explaining your reward design

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
This paper proposes a two-stage training framework—a knowledge distillation phase followed by a reinforcement learning (RL) phase—to enable a compact 0.6B Qwen model to achieve strong tool-calling capabilities. For each phase, the authors introduce targeted improvements. In the distillation phase, they use a forward KL divergence variant that augments the standard top-k forward KL to suppress confidently incorrect predictions. In the RL phase, they design a heuristic reward function that evaluates the similarity between rollouts and ground truth with more fine-grained criteria, providing richer reward feedback compared to conventional binary rewards. By combining these techniques, the authors demonstrate that their method effectively trains a small model with strong tool-calling performance.

### Strengths
* The paper provides targeted and practical improvements for enhancing tool-calling capability, offering insights that could be useful for others working in this area. The performance gains are solid and well-demonstrated through experiments

### Weaknesses
* The framework is sound and the empirical results are solid; however, the methodological contribution is not particularly significant.

### Questions
NA

### Soundness
3

### Presentation
3

### Contribution
2
