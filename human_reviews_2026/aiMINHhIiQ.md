# Flexible Feature Distillation for Large Language Models

- Decision: Reject
- Scores: 6, 0, 4, 2

## Abstract
Knowledge distillation (KD) has become a cornerstone for compressing large language models (LLMs). However, existing LLM-KD methods have primarily focused on logit-based approaches, which achieve good performance but overlook the rich internal representations of LLMs. Feature-level KD could leverage this structure to provide complementary benefits, yet it remains underexplored because current feature-KD approaches typically assume identical teacher–student hidden sizes, a restrictive and unrealistic assumption. A common workaround is to train a linear projector to align their feature spaces; however, this introduces additional parameters, distorts teacher embeddings, and often degrades downstream performance, especially in generative tasks. We propose Flex-KD, a parameter-free framework for task-driven feature distillation for LLMs. Instead of projecting the entire teacher representation, Flex-KD uses gradient-based scores to identify the most task-relevant dimensions of the teacher’s hidden states and distills only this subspace into the student. This ensures that the student’s limited capacity is allocated to informative components, while avoiding projector-induced distortion and extra parameters. Flex-KD integrates seamlessly with existing KD pipelines and supports differing teacher–student hidden sizes.
Extensive experiments across both classification and generative tasks, i.e., instruction-following and summarization, show that Flex-KD **consistently** boosts the student performance, achieving up to a 3.75\% performance gain over the linear projection baseline.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes **Flex-KD**, a new method for distilling LLMs based on features instead of logits. Traditional KD methods mostly rely on output logits, but this paper argues that **feature-level distillation** is often more effective. Flex-KD is a **parameter-free** approach that dynamically selects and distills important subspaces of the hidden representations, guided by gradient-based importance metrics. It avoids costly projection modules and maintains flexibility across layers and token positions. Experiments on various teacher–student setups show that Flex-KD achieves superior performance compared to logit-based and feature-based KD baselines like MiniLLM, Projector, and CKA.

### Strengths
- Good problem statement; the paper clearly identifies a compelling limitation of traditional logit-based KD, its lack of generalizability to feature representations. The motivation to shift toward a more flexible, representation-level distillation is both well-articulated and timely. Flex-KD framework is a simple yet effective approach that provides a parameter-free alternative to other feature-based KD methods.
- The idea of identifying the most important feature dimensions and distilling that subspace is novel and effective to circumvent the representation projection approach. It seems computationally efficient and architecture-agnostic.

### Weaknesses
- **Lack of comparisons with recent logit-based KD baselines**: While the paper compares against Projector, CKA, and MiniLLM, it misses some more recent and state-of-the-art distillation methods such as GKD (Agarwal et al., 2024), DistiLLM (Ko et al., 2024), DistiLLM-2 (Ko et al., 2025), and Speculative KD (Xu et al., 2024). These should be comprehensively included in the comparison, since Flex-KD is meant to compete with these in practical settings. Also, the authors should discuss more about why feature-based KD methods are better than logit-based KD methods.
- The authors mention that *Flex-KD can be combined with logit-based KD*, but **no experiments are provided to support this hybrid strategy**. It must be demonstrated how well this integration performs, and how the state-of-the-art KD methods could be combined with Flex-KD framework.
- All experiments are conducted with **same-family teacher–student pairs** (e.g., LLaMA-7B → LLaMA-1.3B, GPT2-1.5B → GPT2-120M). It remains unclear whether cross-family distillation is feasible under Flex-KD, even when the teacher and student have different embedding subspaces. For example, does it still work when distilling from a LLaMA teacher to a GPT2 student?

### Questions
See weaknesses

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
This paper introduces Flex-KD, a framework for feature-based knowledge distillation designed to compress LLMs into smaller student models with different hidden state dimensions.

### Strengths
The idea of selecting a feature subspace based on gradient-based importance is a clever, parameter-free way to direct the student's limited capacity towards what matters most for a specific task.

### Weaknesses
1. The primary weakness lies in the potentially simplistic definition of "task-relevance" and the static nature of the unit selection process. The method relies on a first-order gradient magnitude, which captures local sensitivity but may fail to account for more complex, non-linear, or cooperative interactions between hidden units. 
2. The paper only attempts to distill a smaller language model from several very old, small-parameter models. Small models under the latest training paradigms have become more powerful, and the effectiveness of the proposed method under these new models is questionable. Furthermore, the effectiveness of the proposed Flex-KD method using a very large-parameter model as the teacher is questionable.
3. The methods compared in this paper are too old. In the past year, a lot of work has considered feature alignment rather than just logits alignment [1]. Also, because the tasks in this paper on are too simple, the advantage over such simple baselines is very minor.

[1] DDK: Distilling Domain Knowledge for Efficient Large Language Models

### Questions
I would suggest the author to conduct experiments based on the most cutting-edge LLM in the next submission.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces Flex-KD, a parameter-free, task-driven feature distillation method for compressing LLMs. Unlike traditional feature-level KD techniques that require matching hidden dimension or rely on learnable, potentially distorting projectors, Flex-KD selects and transfers only the most task-relevant teacher features to the student. It achieves this by computing gradient-based importance scores to identify key dimensions in the teacher's hidden states and distills only the top-scoring subset, enabling effective transfer even when teacher and student have mismatched hidden sizes. Flex-KD integrates seamlessly into existing KD pipelines. Experiments across classification, instruction-following, and summarization on multiple models and datasets demonstrate consistent improvements over prior state-of-the-art baselines, particularly in low-data regimes.

### Strengths
1. This paper tackles a well-motivated and underexplored limitation of feature-based KD for LLMs, namely, the rigid requirement for teacher and student hidden size alignment or the pitfalls of linear projection. This directly addresses real-world deployment needs for LLM compression.

2. Flex-KD's selective, gradient-based feature matching is a creative and simple idea, which avoids introducing new parameters and sidesteps projector-induced feature distortion. It encourages the student to focus on the most informative subspace rather than all features, which is both intuitive and empirically validated.

### Weaknesses
1. While Flex-KD's selective feature distillation is effective, the manuscript's novelty over CKA and activation/importance-based selection is somewhat incremental. CKA already enables flexible hidden size matching, and the added selectivity via gradient-based ranking, while empirically justified, is a modest step forward.

2. Although Table 9 reports computational overheads, the paper does not compare against logit-based KD, which is the most lightweight baselines. Without this reference, the actual extra cost of Flex-KD's feature distillation / gradient-based selection remains unclear.

3. Flex-KD always selects the top-$d_S$ features globally, which may be suboptimal for tasks with input-dependent importance. The approach does not explore dynamic or per-example selection strategies or consider distributing features in a more nuanced way. This weakens the case that such a hard selection will always work.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces Flex-KD, a parameter-free, task-driven feature distillation framework for large language models (LLMs).Unlike conventional feature-based knowledge distillation (KD) methods that assume identical hidden sizes between teacher and student models—or rely on learnable linear projectors to bridge them—Flex-KD identifies the most task-relevant subspace of the teacher’s hidden representations using gradient-based importance scores, and distills only that subset into the student model.The method allows flexible hidden-size matching and avoids projector-induced distortion or extra parameters. The authors integrate Flex-KD into existing KD pipelines and evaluate it extensively across classification, instruction-following, and summarization tasks (13 datasets, 8 model pairs).Flex-KD consistently improves student performance over baselines (up to 3.75% ROUGE-L gain) while remaining stable and computationally efficient

### Strengths
1.Novel and elegant idea – The parameter-free subspace selection via gradient attribution is both conceptually simple and technically effective. It directly addresses a longstanding limitation in feature-based KD for LLMs.
2.Strong empirical validation – The experiments span a wide range of architectures (GPT2, BERT, OPT, LLaMA, BART) and tasks. The results are consistent and show meaningful improvements over strong baselines such as Projector-based and CKA-based feature KD.
3.Efficiency and practicality – Flex-KD adds no trainable parameters, integrates seamlessly into existing KD pipelines, and performs robustly even under limited training data (5% of Dolly).

### Weaknesses
Limited theoretical justification – The use of gradient magnitude as a proxy for task relevance is intuitive but lacks a rigorous theoretical foundation or ablation on its correlation with mutual information or task-specific utility.
2.Computation cost concern – Gradient-based importance estimation still requires additional backward passes over the dataset, which may become expensive for extremely large teacher models. Efficiency comparisons are missing.
3.Scope of applicability – Experiments are limited to Transformer-based architectures; it remains unclear whether Flex-KD can generalize to non-Transformer or multi-modal settings (e.g., Mamba, vision-language models).

### Questions
1.How often are the gradient-based importance scores recomputed during training? Is the subspace selection static (computed once) or dynamic (updated per epoch)?
2.What is the computational overhead of Flex-KD compared to projector-based distillation?
3.Could the proposed method be combined with structured pruning or neuron masking for further compression?
4.Would gradient saturation in very deep networks affect the stability of importance estimation?
5.How does Flex-KD behave when the student’s hidden size is larger than the teacher’s (e.g., upscaling scenarios)?

### Soundness
3

### Presentation
3

### Contribution
3
