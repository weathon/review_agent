# Disentangled Safety Adapters Enable Efficient Guardrails and Flexible Inference-Time Alignment

- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
Existing paradigms for ensuring AI safety, such as guardrail models and alignment training, often compromise either inference efficiency or development flexibility. We introduce Disentangled Safety Adapters (DSA), a novel framework addressing these challenges by decoupling safety-specific computations from a task-optimized base model. DSA utilizes lightweight adapters that leverage the base model's internal representations, enabling diverse and flexible safety functionalities with minimal impact on inference cost. Empirically, DSA-based safety guardrails substantially outperform comparably sized standalone models across hate speech classification, detecting unsafe model inputs and responses and hallucination detection with relative improvements of up to 53\% in AUC.  Furthermore, DSA-based safety alignment allows dynamic, inference-time adjustment of alignment strength and a fine-grained trade-off between instruction following performance and model safety. Importantly, combining the DSA safety guardrail with DSA safety alignment facilitates context-dependent alignment strength, boosting safety on StrongReject by 93\% while maintaining 98\% performance on MTBench—a total reduction in alignment tax of 8 percentage points compared to standard safety alignment fine-tuning. Overall, DSA presents a promising path towards more modular, efficient, and adaptable AI safety and alignment.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces Disentangled Safety Adapters (DSA), a modular approach to improving LLM safety without the trade-offs of guardrails or full alignment fine-tuning. DSA uses lightweight adapters that tap into the base model’s internal representations, enabling efficient and flexible safety control at inference time. Experiments on models like Qwen, Mistral, Gemma, and Zephyr show strong results.

### Strengths
1. The paper is clearly structured and highly readable. I think this paper effectively motivates the problem of AI safety trade-offs between guardrails and alignment fine-tuning, and communicates the key ideas in an intuitive way. 

1. The concept of Disentangled Safety Adapters (DSA) offers a compelling alternative to existing paradigms; that is, separate guardrail models and alignment fine-tuning. By decoupling safety-specific computations while leveraging base model representations, DSA provides a modular and efficient framework for safety interventions at inference time. This represents a potentially important step forward in designing adaptive and resource-efficient safety systems for large language models.

1. The proposed method is technically solid and easy to follow. The authors provide a clear mathematical formulation for both DSA-based guardrails and alignment  and a thorough comparison to related architectures such as LoRA, Res-Tuning, and Ladder Side-Tuning. The description of targeted alignment effectively shows how context-dependent safety control can be achieved without retraining the base model.

1. The empirical results are impressive and demonstrate clear superiority over baseline methods across multiple datasets and model families.

### Weaknesses
1. The paper provides insufficient detail for reproduction. Although the appendices briefly list hyperparameters and dataset splits, the source code and trained models are not provided, and several important training settings (e.g., exact adapter dimensions, implementation details for cross-attention, initialization strategies, and data preprocessing) are only briefly summarized. Given the complexity of multi-module training (guardrail + alignment + targeted interpolation), reproducing the experiments from the text alone would be extremely difficult. The absence of public code limits the practical verification of the claimed efficiency and scalability benefits.

1. The paper indirectly evaluates helpfulness via MTBench scores, showing that DSA maintains most of the model’s helpfulness while improving safety. However, there is no direct human evaluation of response quality or real-world usefulness.

### Questions
1. Did the authors examine how the actual generated outputs change after applying DSA? If so, what qualitative characteristics or behavioral differences were observed in the model’s responses?

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
4

### Summary
The paper introduces Disentangled Safety Adapters (DSA), a novel framework that decouples safety-specific computations from the base model by attaching lightweight adapters that reuse internal representations. DSAs are applied to both safety guardrails and safety alignment. The authors show empirical results on various benchmarks, demonstrating that DSA guardrails outperform standalone classifiers with similar cost and that targeted alignment reduces the alignment tax compared to standard LoRA fine-tuning.

### Strengths
•⁠  ⁠Clear and Novel Concept: The DSA framework is an intuitive design that reuses the base model's computations for safety tasks, enabling low-cost, disentangled safety modules.
•⁠  ⁠Guardrail Performance: DSA guardrails match the performance of full-sized, high-cost models while adding minimal compute overhead (<1% FLOPs). They outperform comparable low-cost models, with up to a 53% relative AUC improvement on hallucination detection.
•⁠  ⁠Targeted Alignment Results: The method achieves safety levels comparable to standard LoRA fine-tuning while preserving ~98% of the base model's general task performance, effectively mitigating the alignment tax.
•⁠  ⁠Architectural Analysis: The paper provides clear justification for its design choices by comparing adapter architectures and iterating on alignment adapters.

### Weaknesses
•⁠  ⁠Novelty: While presented as a novel framework, DSA is largely an incremental adaptation of existing adapter and side-tuning ideas. The contribution lies mainly in applying these methods to safety tasks, which is valuable but not conceptually deep.
•⁠  ⁠Weak Baselines: Comparisons rely on low-cost guardrails rather than competitive guardrails (e.g., Llama-Guard 2). DSAs also receive richer initialization and distillation, compromising fairness. Stronger baselines and matched compute would be needed to validate the claimed efficiency advantage.
•⁠  ⁠Limited interpretability and diagnostic analysis: The paper offers no qualitative analyses, error breakdowns, or visualization of adapter activations. Without understanding what DSAs actually learn or when they fail, it is hard to assess their safety reliability or transparency.

### Questions
See weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper focuses on two current technical approaches for LLM safety: guardrail models and safety alignment, with an emphasis on the trade-off between development flexibility and inference efficiency. It proposes a Disentangled Safety Adapter (DSA), which enhances safety by training independent guardrail and alignment adapter modules, without modifying the base model's parameters or altering its core inference path for general tasks.

### Strengths
1. The paper identifies the crucial trade-off between development flexibility and inference efficiency in current guardrail models and safety alignment methods, particularly in LLM deployment.
2. The DSA framework proposed in this paper unifies the training of guardrail models and safety alignment, which is an innovative approach.

### Weaknesses
1. Lack of Generalization Verification for the DSA Method:

 1.1. The experiments were conducted on only one base model, which itself has poor safety performance. It would be beneficial to see experiments on at least three different models, not just the base model, as models fine-tuned with instructions may be more prone to performance degradation after safety alignment.
 
1.2. The paper uses only one training dataset, but the introduction mentions one of the advantages of DSA as "updating safety policies without retraining or redeploying the base model." This claim requires validation with more diverse training data.

2. Missing Key Baselines:

The DSA framework proposed in this paper essentially adds additional modular parameters to the model, improving safety through optimization. There is a similar prior work, “MoGU: A Framework for Enhancing Safety of Open-Sourced LLMs While Preserving Their Usability,” but no comparison is made in the experiments.

3. Lack of Hyperparameter Analysis:

From Figure 2, it appears that the DSA method is quite sensitive to hyperparameters. Whether this impacts the method's practicality needs further analysis and clarification.

4. The necessary ablation analysis is lacking：

The proposed DSA Targeted Alignment framework includes two trainable modules: a guardrail module and an alignment module; however, there are no experiments directly comparing which of the two modules is more important.

### Questions
1. In the introduction, one of the advantages of DSA is mentioned as "updating safety policies without retraining or redeploying the base model," but the experiments lack additional validation for this point. I carefully reviewed the paper, and the authors only trained one model on one dataset.
2. Table 3 primarily compares DSA with LoRA-based DPO, but in fact, DSA increases the number of parameters compared to the base model, whereas LoRA-based methods do not add parameters during inference. This seems to contradict the motivation of the paper. 
3. Additionally, LoRA's specific settings, such as hyperparameters like rank, can impact performance. It's unclear whether the authors considered this in their experiments. Why not consider full model fine-tuning?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces Disentangled Safety Adapters (DSA), a framework that decouples safety-specific computations from task-optimized base models using lightweight adapters. The authors propose DSA for two applications: (1) safety guardrails that classify unsafe inputs/outputs by reusing base model representations, and (2) safety alignment that steers generation by interpolating between base model and adapter logits. The key innovation is "targeted alignment" that dynamically adjusts alignment strength based on input safety classification. Experiments show DSA guardrails outperform comparably-sized standalone models (up to 53% AUC improvement on hallucination detection) while adding <1% computational overhead, and DSA targeted alignment reduces alignment tax by 8 percentage points compared to standard fine-tuning.

### Strengths
1. Well-motivated problem: The paper clearly articulates the trade-off between separate guardrails (flexible but expensive) and alignment training (efficient but inflexible), positioning DSA as an elegant solution combining advantages of both approaches.
2. Good experimental evaluation:
    - Multiple base models (Qwen, Mistral, Gemma) and scales (7B-14B)
    - Diverse safety tasks (hate speech, unsafe prompts/responses, hallucination detection)
    - Systematic architecture comparison (PLR → RTB → LST)
    - Thorough ablations and multiple benchmarks
3. Good empirical results: DSA:LST matches full-sized guardrails at <1% overhead. Consistent improvements across model families and tasks

### Weaknesses
1. Limited architectural novelty: The paper primarily applies existing adapter architectures (Side-Tuning, LST, Res-Tuning) to safety tasks. While the application is novel, the core technical contribution is incremental. The DSA:LST+ and DSA:LLD variants introduced for alignment are relatively minor modifications.

2. Binary switching mechanism: Using a step function ($\lambda \in \{0,1\}$) is simplistic. The paper mentions smooth interpolation as future work, but this seems essential for robustness.

3. DSA:LST and DSA:LST+ significantly underperform LoRA baseline (Figure 2). And I don't agree the statement that
```
This method (Fig. 2, "LORA") incurs no additional inference cost but entangles safety with
base model computations, precluding flexible alignment strength adjustment without running both
aligned and unaligned models
```
Actually, LoRA can be only used in the last layer and most computaions can be used. 

4. Limited baseline comparisons: No comparison with other efficient guardrail methods. Is there any other methods for similar tasks? For example, a good and simple baseline could be just prompt engineering: Prompting LLM itself to output whether the input/output is safe or not, and most computations could be reused due to prompt KV caching mechanism.

### Questions
1. What is the actual latency overhead in your experiments and in production settings with batching?
2. Can adapters trained for one safety policy be efficiently adapted to new policies?

### Soundness
2

### Presentation
2

### Contribution
2
