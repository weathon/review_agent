# MaskSearch: Towards Scalable Agentic Pre-Training for Search-Enhanced Reasoning

- Avg Score: 5.33
- Decision: Reject
- Scores: 4, 8, 4

## Abstract
Retrieval-Augmented Large Language Models (LLMs) excel at knowledge-intensive tasks but struggle in complex scenarios due to passive retrieval. 
    While search agents empower LLMs to actively use tools for reasoning, existing training-based methods remain constrained by task-specific data.
    Therefore, we propose MaskSearch, a two-stage training framework that bridges foundation models with search agents through a novel Retrieval-Augmented Mask Prediction (RAMP) task. Models first learn to recover masked spans via multi-step search and reasoning as a pre-training stage to endow foundational agentic capabilities that can be further improved by post-training on downstream tasks.
    We apply Supervised Fine-tuning (SFT) or Reinforcement Learning (RL) for training. 
    For SFT, we combine multi-agent trajectory synthesis with iterative self-evolution distillation to construct data.
    For RL, we employ DAPO with a hybrid reward system consisting of answer and format rewards. 
    Additionally, we introduce a curriculum learning strategy based on the number of masked spans. 
    We evaluate the effectiveness of our framework in the scenario of open-domain multi-hop question answering. Extensive experiments demonstrate that MaskSearch effectively equips LLMs with transferable agentic abilities, advancing the development of search agents.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes MaskSearch, a two-stage training framework designed to enhance the search and reasoning capabilities of Large Language Models (LLMs) through scalable agentic pre-training. Existing Retrieval-Augmented Language Models (RALMs) often use passive retrieval, while training-based search agents are constrained by task-specific data. To address this, the authors introduce a novel pre-training task called RAMP.  

Experimental results demonstrate that models pre-trained with RAMP show significant performance improvements on several open-domain multi-hop QA benchmarks compared to baselines trained only on the downstream task.

### Strengths
- This work reframes the training of agentic search and reasoning abilities as a self-supervised-like mask prediction task. RAMP can leverage massive unlabeled text corpora (like Wikipedia) to generate nearly infinite training data. This scalability is key to training more robust and general search agents.

- The experimental results demonstrate that adding the RAMP pre-training stage consistently and significantly outperforms baselines trained only with downstream task fine-tuning, across various model sizes (Qwen and LLaMA).

- The framework is not limited to a single training methodology. It supports both SFT and RL (DAPO) and provides an experimental comparison between them.

### Weaknesses
1. The "alignment" of the pre-training task with downstream tasks and its potential harm to generalization. The proposed RAMP task (Sec 2.2) is fundamentally a *Convergent* task, i.e., locating and reasoning about specific, masked facts. However, many real-world search tasks (e.g., "write a research report on...") are *Divergent*, requiring exploration, synthesis, and generation of new knowledge. Over-optimizing the RAMP task on search agents may lead to a "narrow-sighted" agent, harming its ability to handle divergent tasks. 

2. The paper observes in Sec 4.2 (Fig 3) that although RAMP pre-training is effective, the performance improvement for the 7B model flattens as the data scale increases. As noted in the draft, this is likely due to an upper bound on the "self-evolved" data. This suggests that a quality bottleneck in SFT data limits the ceiling of the pre-training method, especially for bigger models.

3. The ablation study on masking strategies in Sec 5.1 shows that the harder PPL strategy does not always yield better results. This indicates that the core question of "what kind of RAMP task best promotes agentic learning" remains unanswered, and the current design may not be optimal.

### Questions
1. Have the authors tested the RAMP pre-trained model on "Wide Search" or "deep research" type benchmarks (such as those in arxiv:2506.11763 or 2508.07999)? If RAMP indeed harms performance on such tasks, this would be a limitation that should be discussed.

2.The diversity bottleneck of SFT data appears to limit the 7B model's performance. Have the authors tried using a stronger external model (instead of "self-evolution") to generate SFT trajectories, to test if the performance ceiling of the 7B model can be further unlocked by higher-quality data?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper introduces a pre-training task for LLMs specifically designed for search-oriented agents. The core idea is to define a mask-based prediction task that incorporates information retrieved through a search operation. In practice, a sentence is sampled from a dataset, and search-relevant information is masked. A solution trajectory is then generated via interaction with a search agent, and this trajectory is used to train the model through either supervised fine-tuning or RL-based methods. Finally, a post-training stage using multi-hop QA data is applied to enable question answering. The effectiveness of the approach is evaluated on two model families (LLaMA 3 and Qwen2.5) of various sizes and across multiple datasets.

### Strengths
1. The approach is novel. The idea of introducing a masked prediction pre-training task before applying the Search-R1-style post-training is smart and empirically shown to be effective.    
2. The evaluation demonstrates consistent improvements across different architectures and model scales. The performance gains are significant and clearly support the usefulness of the method.
3. The manuscript is well written, with a clear problem setup and a well-presented methodological contribution.

### Weaknesses
While the authors have tested their method on multiple model architectures, it is unclear if such a pre-training strategy would scale effectively to larger models. This concern is particularly relevant given the observation, as shown in Figure 3, that performance improvements tend to saturate as model size increases. Although this is not a major issue and does not undermine the validity of the contribution, it would be helpful if the authors could comment on scalability toward larger models.

As a minor point, the paper could benefit from an analysis of the computational overhead introduced by the additional pre-training task. It would be useful to explicitly report how the computational cost of the search-based pre-training compares to that of the original pre-training and the QA-oriented post-training stages.

### Questions
1. Is there any intuition of the behavior on larger models?
2. What is the computational overhead?

### Soundness
4

### Presentation
4

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
The paper MASKSEARCH: Towards Scalable Agentic Pre-training for Search-Enhanced Reasoning introduces a unified two-stage training framework for large language models (LLMs) that integrates retrieval-based reasoning and agentic search abilities. The key innovation, Retrieval-Augmented Mask Prediction (RAMP), serves as a scalable pre-training task where the model learns to fill masked spans through multi-step search and reasoning, thereby simulating agentic behavior. The framework employs both Supervised Fine-tuning (SFT) and Reinforcement Learning (RL) in pre- and post-training stages, with a hybrid reward system and curriculum learning strategy. Extensive experiments across multiple open-domain multi-hop QA benchmarks demonstrate that MASKSEARCH substantially improves both in-domain and out-of-domain reasoning performance, effectively bridging foundational pre-training and search-augmented agentic learning.

### Strengths
1. Proposes the first scalable Retrieval-Augmented Mask Prediction (RAMP) task, connecting masked language modeling with agentic search reasoning.
2.  Elegantly combines pre-training and post-training using both SFT and RL, enhancing flexibility and generalizability.
3. Demonstrates consistent and transferable improvements across diverse multi-hop QA datasets and model sizes.
4. Integrates multi-agent data synthesis, self-evolution distillation, and curriculum learning to ensure scalability and stability.

### Weaknesses
1. The paper lacks a theoretical analysis of the convergence and generalization properties of the RAMP task, remaining primarily at the empirical level.
2. The framework shows strong structural similarities to prior works such as Search-R1 and DeepSeek-R1, making the boundary of innovation insufficiently clear.
3. Although multiple datasets are evaluated, the paper lacks fine-grained discussions on failure cases, cross-task transferability, and robustness to noise.
4. While the DAPO algorithm and reward design are described, the paper does not provide comparative analyses of training stability or convergence behavior.

### Questions
See the Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
