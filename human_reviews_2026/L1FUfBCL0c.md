# CoT Vectors: Transferring and Probing the Reasoning Mechanisms of LLMs

- Decision: Accept (Poster)
- Scores: 6, 6, 6

## Abstract
Chain-of-Thought (CoT) prompting has emerged as a powerful approach to enhancing the reasoning capabilities of Large Language Models (LLMs). However, existing implementations, such as in-context learning and fine-tuning, remain costly and inefficient. To improve CoT reasoning at a lower cost, and inspired by the task vector paradigm, we introduce CoT Vectors, compact representations that encode task-general, multi-step reasoning knowledge. Through experiments with Extracted CoT Vectors, we observe pronounced layer-wise instability, manifesting as a U-shaped performance curve that reflects a systematic three-stage reasoning process in LLMs. To address this limitation, we propose Learnable CoT Vectors, optimized under a teacher–student framework to provide more stable and robust guidance. Extensive evaluations across diverse benchmarks and models demonstrate that CoT Vectors not only outperform existing baselines but also achieve performance comparable to parameter-efficient fine-tuning methods, while requiring fewer trainable parameters. Moreover, by treating CoT Vectors as a probe, we uncover how their effectiveness varies due to latent space structure, information density, acquisition mechanisms, and pre-training differences, offering new insights into the functional organization of multi-step reasoning in LLMs. The source code will be released.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents CoT Vectors, a novel extension of task vectors for modeling multi-step reasoning in LLMs. The approach is well-motivated and provides meaningful insights into reasoning dynamics, with Learnable CoT Vectors showing stronger and more stable gains than extraction-based methods. However, the variability across intermediate layers suggests limitations in capturing intra-task diversity. Further exploration of adaptive or hierarchical vectorization could enhance robustness and generalization. Overall, this is a valuable contribution to understanding structured reasoning in LLMs.

### Strengths
The paper’s strengths lie in its novel approach to vectorizing Chain-of-Thought (CoT) reasoning, offering a compact and efficient representation of multi-step reasoning processes. The introduction of a teacher–student framework for learning CoT vectors is noteworthy, as it effectively captures the influence and transfer of reasoning knowledge. In addition, the paper is supported by comprehensive experimental evaluations, which clearly demonstrate the advantages and robustness of the proposed method across different settings.

### Weaknesses
Here are some concerns and/or issues about this work:

The proposed method is intriguing, but its systematic assessment of effectiveness is lacking. Specifically, it lacks a measurable way to categorize different types of CoTs and demonstrate when the proposed method works well and when it doesn’t.

The authors claim that the method saves compute cost because fine-tuning is not required. However, the inference process involves invoking both the frozen LLM and the student model. Given that models are typically used for inference far more often than training, the overall efficiency of the proposed method in real-world scenarios remains uncertain.

If I understand the authors correctly, there may be a hidden issue in Eq. 4. In Eq. 4, the extracted CoT vector takes the average. This means that if the answer is long and there are only a few mismatched tokens, the model may accept the result. However, if the problem is similar to the one depicted in Figure 1, a few mismatched tokens can invalidate the entire answer.

The experimental study lacks evaluation of the end-to-end efficiency of the proposed method. The authors propose comparing it to fine-tuning, so they should include some experimental results to support the significance of the proposed method compared to fine-tuning.

### Questions
See the weakness section.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces CoT Vectors, a compact representation that encodes multi-step reasoning knowledge, enabling LLMs to enhance reasoning without modifying model weights.
Two approaches are proposed to obtain these vectors: extracted CoT vectors and learnable CoT vectors optimized via a teacher-student distillation framework, with analysis of their layer-wise effects.
Experiments show that the proposed methods can improve performance on the studied tasks.

### Strengths
1. The method proposed in the paper is overall clear in its conceptual approach and is presented in a way that is relatively easy to understand.
2. Layer-wise analysis offers a novel perspective on how reasoning in LLMs is internally organized.

### Weaknesses
1. The main experiments in the paper focus on mathematical reasoning and some domain-specific tasks, and the performance of the proposed method on more natural language logical reasoning tasks remains to be further validated.

2. The training of task vectors may depend on factors such as the number and quality of sampled support instances, as well as the balancing factor $\lambda$ between the two losses. Providing additional experimental results on these aspects could better demonstrate the robustness of the proposed method.

3. Although the authors provide some formula-based intuitive insights in Section 3.1, there remains a certain gap between the theoretical explanation and practical implementation. For example, the $\mu$ and shift vector in Equation 1 should ideally be functions that vary with the input, whereas the task-general CoT vectors seem to assume a single shared reasoning chain prompt (i.e., the same $K_C$ and $V_C$) for all instances of the same task. Even under this assumption, $\mu$ would still be expected to vary with the input.

4. While the proposed method introduces relatively few trainable parameters, in practice an additional teacher model is required to perform CoT reasoning and align the output distributions with the student model. Reporting only the number of trainable parameters may therefore provide an incomplete picture of the overall overhead.

5. The comparison in the paper is limited to LoRA as a parameter-efficient fine-tuning baseline, lacking evaluation against other fine-tuning or prompt optimization methods, such as Prefix-Tuning or Adapters.

### Questions
see Weakness.

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
3

### Summary
The paper introduces CoT vectors, an extension of the task vectors to encode and transfer multi-step reasoning in LLMs. The authors propose and study two initiation of the CoT vectors: extracted CoT vectors and learnable CoT vectors. The extract CoT vectors are computed as the activation difference between reasoning and non-reasoning traces at a given layer; the learnable CoT vectors are parametrized as a learnable shift added to the hidden state of a layer and optimized in a teacher-student framework to distill reasoning representations. Experiments on three benchmarks (GSM8K, MATH, and MMLU-Pro) across two model families (Qwen2.5 Math and Llama3.1 8B) show consistent improvement over zero-shot CoT and LoRA baselines using only 3-4K parameters. Furthermore, the authors use CoT vectors to uncover a three-stage reasoning process in LLM and provide new insights that contrast with prior task vector findings. The observation is supported by additional analyses on layer-wise performance and latent space visualization.

### Strengths
- The paper has clear question formulation and motivation, is well-structured, and easy to follow. 
- By probing with CoT vectors, the authors discover a U-shaped layer-wise performance curve and provide a three-stage reasoning process interpretation, offering novel insights that are different from prior task vector research. Supporting analyses provide a valuable contribution to understanding model internals. 
- The proposed learnable CoT vectors address limitations of extraction-based methods (layer-wise instability), and effectiveness is supported through experiments and visualizations
- Evaluations span across multiple models, datasets, and baselines. The results demonstrate consistent improvement, validating the effectiveness of the proposed method.
- The method is parameter-efficient and requires negligible inference overhead, avoiding prompt lengthening and full-finetuning, which aligns with practical needs in efficient LLM deployment.

### Weaknesses
- The scope of "task-general" vectors is not fully supported by experiments. The CoT vectors are obtained per dataset, and the paper doesn't perform cross-task transferability tests to back the "task-general reasoning" claim. For example, the authors should consider applying vectors learned on GSM8K on MATH and vice versa.
- The improvements on Llama are small, and MMLU-pro only uses 70 annotated questions. It is difficult to assess the reliability of small gains without variance across seeds and significance tests.
- LoRA is the only parameter-efficient baseline, whereas other activation intervention methods are not compared or mentioned, including [1], [2], [3]. 
- The paper reports the best injection results. However, in practice, a comprehensive layer search could limit applicability.

[1] Azizi, Seyedarmin, Erfan Baghaei Potraghloo, and Massoud Pedram. "Activation Steering for Chain-of-Thought"
[2] Zhang, Jason, and Scott W. Viteri. "Uncovering Latent Chain of Thought Vectors in Large Language Models."
[3] Tang, Xinyu, et al. "Unlocking General Long Chain-of-Thought Reasoning Capabilities of Large Language Models via Representation Engineering."

### Questions
- In equation 7, why is the intervention applied to the hidden state rather than directly to the attention output? How do the author explain the difference between formulation and practical use?

### Soundness
3

### Presentation
3

### Contribution
3
