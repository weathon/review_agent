# Decomposing Policy Optimization into Proxy Objective and Knowledge Distillation for Long-context Reasoning

- Decision: Reject
- Scores: 6, 4, 6, 4

## Abstract
Reinforcement learning (RL) paradigms have demonstrated remarkable success in enhancing the complex reasoning capabilities of Large Language Models (LLMs). However, models trained exclusively on short-context tasks often struggle to generalize their capabilities to longer documents. Extending RL to long-context settings remains a fundamental challenge, primarily due to the high computational cost and deteriorating quality of the sampling reasoning trajectories. To bridge this gap, we propose Decomposed Policy Optimization (DePO), a novel framework that decouples policy optimization for long-context reasoning into two parallel parts: (1) leveraging short-context environments as a tractable proxy objective to acquire high-quality and efficient reasoning patterns via RL; and (2) transferring these capabilities to long-context settings through knowledge distillation, enabling the model to replicate its refined short-context reasoning strategies over extended contexts. Empirical evaluations across multiple long-context document question-answering benchmarks show that DePO consistently outperforms all baseline approaches. Notably, compared to naive long-context reinforcement learning, DePO achieves 2.8\% improvement in performance while also reducing training time overhead by 49.5\%. Furthermore, DePO demonstrates strong generalization capabilities, compatibility with various RL algorithms, and consistent improvements across multiple base models, offering a scalable and efficient solution for advancing long-context reasoning.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper explores the problem of long in-context reasoning with reinforcement learning. Existing RL for long-context settings is challenging, given the high computational cost and deteriorating quality of the sampling reasoning trajectories. In this paper, the authors propose a new method called DePO, which decouples policy optimization for long-context reasoning into two parts: (1) first optimize the short-context reasoning patterns via RL; and (2) transfer the short-context reasoning to long-context reasoning through knowledge distillation. Empirical evaluations across multiple benchmarks show the effectiveness of the proposed DePO method.

### Strengths
1. The paper is nicely written and very easy to follow.
2. The method proposed in this paper is sound and well-motivated.
3. The authors conduct extensive experiments to demonstrate the effectiveness of the proposed method and obtain in-depth insights.

### Weaknesses
1. It is not straightforward for me to directly distill the desired rollout for short context to the long context ones. There are two reasons:
- The answer to the short context is off-policy to the long-context ones. 
- There can be an information mismatch. For example, in the long context, there will be passages which does not appear in the short context. Directly training on the short-context answer will lead to insufficient reasoning about the passages which does not appear in the short context, but in the long context.

2. Some experimental settings are not very clear. For example, how to develop the labeled data for the supervised finetuning baselines and preference optimization baselines?

### Questions
1. How to solve the insufficient answer problem if we do distillation from the short context answer to the long context answer?
2. How to develop the labeled data for the supervised finetuning baselines and preference optimization baselines? Is it a fair comparison?

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
The paper introduces **DePO**, a novel reinforcement learning framework that addresses the inefficiency of training large language models (LLMs) on long-context tasks. DePO **decouples policy optimization** into two parallel components: (1) reinforcement learning on short-context reasoning as a **proxy objective** to efficiently learn high-quality reasoning behaviors, and (2) **knowledge distillation** to transfer these learned reasoning patterns to long-context settings.

This design mitigates the high computational cost and trajectory quality degradation commonly encountered in long-context RL, resulting in reduced training time and improved accuracy compared to direct long-context RL methods. DePO generalizes across different RL algorithms (e.g., GRPO, DAPO, GSPO) and foundation models (Qwen, Llama), maintaining strong short-context reasoning performance while enhancing long-context reasoning.

### Strengths
1. **Strong Experiment Results** Experiments across multiple long-document QA benchmarks show that DePO consistently outperforms supervised fine-tuning, preference optimization, and direct long-context RL methods, achieving 2.8% higher accuracy and 49.5% less training time.

2. **Compatibility**: DePO proves compatible with multiple RL algorithms (GRPO, DAPO, GSPO) and different foundation models (Qwen and Llama series), maintaining consistent performance gains and scalability across architectures and training paradigms

### Weaknesses
1. **Methodological Clarity** I think the methodology section lacks clarity to some extent. Regarding the knowledge distillation, I am under the impression that the teacher model and student model are being updated simultaneously. Is this the correct understanding of the proposed approach? The teacher is defined as the policy conditioned on the short context, and the student model refers to the same policy conditioned on the long context (line 251). Equation 5 in line 298 indicates that the short context policy model (teacher) and knowledge distillation training (student) are being trained jointly. If this is the case, then there is a concern for model collapse. 

2. **Reward Sparsity Concern** Using exact match as a reward signal may lead to reward sparsity issue, something not adequately addressed by the paper.

### Questions
1. Are the teacher model and student model being updated simultaneously?

2. Since the overall performance gain is very small, I am wondering if there is any fluctuation of performance across multiple runs? Did you perform any significance test?

### Soundness
2

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
This paper proposes a novel approach to enhancing the long-context capabilities of large language models (LLMs) through reinforcement learning. The authors design a clever setup that constructs both short-context and long-context versions of documents for a given query. The policy is first trained on short-context data and then used as a teacher to guide training on long-context documents. The idea is interesting, the writing is clear, and the overall presentation is well-organized and visually appealing.

The main concern lies in the generalizability of the proposed method for long-context training. The approach assumes that for every long-context input, a corresponding short-context version can be found, which may hold true for retrieval-augmented generation (RAG) tasks but not necessarily for other long-context scenarios. I would be interested in hearing the authors’ thoughts on how their algorithm could be extended to handle more complex, non-RAG settings.

Overall, the idea is novel and promising, and I would recommend the paper for acceptance.

### Strengths
- The paper proposes a novel and well-motivated approach to enhance the long-context capabilities of LLMs using reinforcement learning.
- The idea of constructing paired short-context and long-context documents for the same query is clever and intuitively appealing. And the two-stage training strategy based on the constructed data — first training on short-context data and then transferring to long-context settings — is conceptually sound and easy to follow.
- The writing and presentation quality are excellent; the paper is clearly structured, visually appealing, and easy to read.
- The experimental setup and motivation are coherent, making the contributions easy to understand and evaluate.

### Weaknesses
- The generalizability of the proposed approach is limited. The method assumes that for every long-context input, there exists a corresponding short-context version, which may hold for RAG-style tasks but not for other long-context scenarios such as reasoning or summarization.

- The paper lacks a discussion or analysis of how the proposed algorithm could be extended or adapted to non-RAG settings, where short-context counterparts are not naturally available.

### Questions
- How can this method generalize to non-RAG settings where high-quality short-context inputs cannot be easily obtained for each long-context instance?  

- Some clarification is needed regarding the different variants of DePO (GROP, DAPO, GSPO). In Table 2, is the reported result based on DAPO, while Table 3 combines DAPO and GSPO? And are all later tables and figures based on GROP?  

- In Figure 5, how is the training time comparison between DePO and GROP conducted? Did you fix the total number of training steps, set a performance threshold and measure the steps needed to reach it, or use another criterion?  

- The performance visualization in Figure 5 seems slightly misleading — the reported improvement is only about $5.3\%$, yet the blue and orange bars appear to differ by nearly a factor of two. This is not a major issue, but clarification or a note on scaling would be appreciated.  

- The description of the DePO-IW algorithm in the appendix needs further clarification. It appears that this algorithm computes a knowledge-distillation loss rather than a KL divergence between student and teacher policies, and that it applies the policy gradient twice — with the second application (line 16 in Algorithm 2) being somewhat off-policy with reweighting. I am not sure whether I understand correctly.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes Decomposed Policy Optimization (DePO), which decomposes long-context policy optimization for reinforcement learning (RL) into two parts: learning a policy on short-context environments, and then using knowledge distillation to transfer it to long-context environments. The RL model can thus learn robust local reasoning on the short-context datasets. The paper further introduces a variant of DePO, called DePO-IW, that formulates the distillation process as an off-policy reinforcement learning one. The paper presents extensive numerical comparisons to various types of baseline algorithms for long contexts, on ten different datasets, showing that integrating DePO or DePO-IW into existing RL methods generally increases their accuracy and yields higher accuracy than supervised fine-tuning or preference optimization methods.

### Strengths
+ The experimental results seem quite thorough, covering ten datasets and multiple supervised fine-tuning, preference optimization, and reinforcement learning baselines. In most of these comparisons, DePO and its variant DePO-IW improve the performance of these baselines.

+ The paper addresses an important problem of handling long contexts, and it can be integrated into existing reinforcement learning methods for long-context queries.

### Weaknesses
--The intuition behind why we would expect any reasoning learned on short contexts to be useful for longer contexts is explained only very briefly, with the idea that local reasoning can be learned on short contexts without being confounded by long contexts. Given that this is the entire premise of DePO, I’d expect a more thorough discussion motivating the method and explaining why it would be expected to work.

--The technical challenges of DePO are not explained well. Beyond the insight that we can transfer reasoning learned on short contexts to long contexts, the paper does not seem to have any technical novelty. The reinforcement learning formulations in Equations (3) to (7) are taken from past work, and that in Equation (7) is almost the same as the formulation in Equation (3), yielding little additional insight.

-- Many of the details of knowledge distillation (KD) are missing from the paper. For example, how often is KD performed? Only once, after the reinforcement learning model is trained on short contexts? Can any KD algorithm be used, or does the paper need to adapt existing methods to this setting?

### Questions
1) How long does the KD algorithm take to run in practice? Does this represent significant computational overhead compared to training the reinforcement learning methods?

2) In Table 3, neither DePO nor DePO-IW consistently outperform each other. Under which circumstances would a user want to use DePO-IW instead of DePO? Are there particular problem characteristics that might indicate whether one or the other would perform better?

3) Lemma 3.1 does not seem to use any specifics of the long- or short-context settings, and appears to apply to any scenario in which we apply a policy to two different distributions. Does it apply only to the present long vs. short context setting, or can it be applied to these more general settings as well?

Please see also the weaknesses above.

### Soundness
3

### Presentation
2

### Contribution
3
