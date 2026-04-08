## Human Reviewer 1

### Summary
This paper introduces RECAST, a data-synthesis framework that mines verifiable constraints from existing instruction-response pairs and rewrites the original instructions to include many more constraints. The authors use this pipeline to build a training dataset RECAST-30K (~ 30k examples; 19 constraint types), with both rule-based and model-based validators (LLM-as-judge) attached to each constraint. They also propose RLVC (Reinforcement Learning with Verifiable Constraints) which turns per-constraint satisfaction into a fine-grained reward and optimizes policies via GRPO. They also introduce a new benchmark, RECAST-Test, with 4 difficult levels. The experiments show sizable gains in hard satisfaction rate on RECAST-Test benchmark, as well as improvements on IFEVAL and FollowBench, while largely preserving general knowledge performance (evaluated with GPQA and MUSR). Human evaluations support the quality of constraint filtering and selection.

### Strengths
- **Clear motivation and problem:** The paper targets a real gap, LLMs’ performance drops as the number of explicit constraints increases, and existing SFT datasets usually don’t cover a high number of constraints. 

- **The pipeline is complete and reproducible:** The paper details each stage (constraint extraction, instruction enhancement, response synthesis, validation), gives prompts/templates, and reports human agreement metrics. RLVC is specified with GRPO objective and training setup. This level of detail supports reimplementation. The authors promised to release code, data, and trained models upon acceptance. 

- **Evaluation setup is strong:** The primary metric, HSR, is well defined (satisfies all constraints simultaneously) with subtype metrics RSR/MSR for rule/model-based subsets. The proposed benchmark RECAST-Test (cover 4 difficulty levels) and it is associated with two external sets (IFEVAL, FollowBench) for the task. Also, the work included a general ability evaluation with GPQA/MUSR, which is important.

- **Empirical gains & informative analyses:** RECAST-30K improves complex instruction-following against multiple baselines, with RLVC further improving especially at higher difficulty levels. The paper includes informative ablations (constraint type only, quantity caps, component removals) and training dynamics for RLVC that illustrate different learning curves for rule- vs. model-based constraints. It also contains Human evaluation to back up the automated choices.

### Weaknesses
- **Lack of theoretical positioning against RLVR methods:** The paper does not situate its RL formulation within the growing line of Reinforcement Learning from Verifiable Rewards (RLVR) research, despite clear conceptual overlap. Despite the original paper of RLVR being cited [1], it is cited only in the context of the dataset. Other foundational RLVR works [2,3] and subsequent works applying the verifiable reward mechanisms to multi-constraint or format-constrained instruction-following [4,5] are not cited or compared against. As a result, the contribution of RLVC relative to existing verifiable-reward frameworks remains under-theorized: it is unclear whether the proposed per-constraint reward structure offers advantages over scalar verifiable rewards, or whether similar behavior would emerge from established RLVR baselines. Strengthening this connection and including appropriate baselines would significantly clarify the novelty and necessity of RLVC.

\* It is acceptable to not cite works that appeared less than 2 months from the submission deadline, but in that case still recommendable for the final version. 



- **Generalization:** Parts of the training data and evaluation share the same constraint types and verification machinery. Although external benchmarks are included, they also contain very similar constraint types. I propose the authors to present stronger isolation tests (e.g., unseen constraint types or schemas) would help mitigate concerns about over-specialization to the authors’ verification prompts and demonstrate generalization. To do that, maybe it is not needed to run new experiments, but rather present the results of tables 2 (and 1 if possible) including only constraint types not included in the trained data. 


- **Model-based validators error rate and impact:** Rule-based validators are clear, but false positives/negatives and model-based misjudgments are not deeply quantified beyond aggregate human-agreement numbers. It would be nice to have a better quantification of how those affect the performance, or if the method is robust enough to some mislabeled examples. 

References:

[1] Lambert, N. et al. Tulu 3: Pushing Frontiers in Open Language Model Post-Training. COLM 2025 (appeared in Nov, 2024). 

[2] Su, Yi, et al. "Crossing the Reward Bridge: Expanding RL with Verifiable Rewards Across Diverse Domains." arXiv preprint arXiv:2503.23829 (2025).

[3] Wang, Y., et al. "Reinforcement learning for reasoning in large language models with one training example." arXiv preprint arXiv:2504.20571 (2025).

[4] Peng, H., et al. VerIF: Verification Engineering for Reinforcement Learning in Instruction Following. arXiv preprint arXiv:2506.09942 (2025).

[5] Pyatkin, V., et al. "Generalizing Verifiable Instruction Following." arXiv preprint arXiv:2507.02833 (2025).

### Questions
1) Can you clarify what exactly is the novelty of RLVC in comparison with RLVR methods? 

2) Can RECAST-trained models handle novel constraint types not present in RECAST-30K? Please add an experiment with held-out constraint categories (Weakness 2)

### Soundness
2

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
4

---

## Human Reviewer 2

### Summary
This work proposes a framework to construct lagre-scale datasets to test Large Language Models's instruction-following ability when facing complex tasks. Different from previous works, the constructed dataset encompasses various types of constraints, including both subjective and objective ones, and features a larger number of constraints than existing datasets. Furthermore, this study adopts both supervised fine-tuning and reinforcement learning to enhance the Large Language Model's capability to follow instructions.

### Strengths
This submission has the following strengths:
- The paper demonstrates clear writing and a well-structured organization.
- The proposed dataset RECAST-30K is large in scale and contains an adequate number of constraints.
- Experimental results have shown that training with RECAST can effectively enhance large language model's ability in instruction following.

### Weaknesses
This submission has the following weaknesses:
- For model-based constraints, the quality depends on used large language models.
- The count of Rule-based constraints are much less than model-based constraints.

### Questions
I have the following questions / suggestions:
- Would it be possible to extend the constraints numbers of rule-based constraints?
- Why HSR metric is not used in Table 1?
- Why the performance of RECAST-30K-RLVC on Qwen-2.5-7B become worse than RECAST-30K-SFT from time to time? (Bottom of Table 1)
- It would be better to also test some instruction-following enhancement works. For example,
    - Branch-Solve-Merge Improves Large Language Model Evaluation and Generation. In NAACL. 2024.
    - Divide-Verify-Refine: Can LLMs Self-align with Complex Instructions?. In Findings of ACL. 2025.

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
4

---

## Human Reviewer 3

### Summary
This paper introduces RECAST, an automated framework for generating instructions comprising far more constraints than those in current benchmarks and corresponding responses. Using this framework, the authors construct RECAST-30K, which contains 30K instances with diverse constraint types. The dataset is then used for training LLMs via RLVC, an optimization method that leverages the verifiable nature of constraints to provide fine-grained reward signals. Experiments demonstrate the effectiveness of RECAST compared with other complex instruction fine-tuning datasets.

### Strengths
1. The framework incorporates an extended number of constraints in a single instruction, which facilitates the evaluation and improvement of LLMs' capability of following more complex instructions as the tasks for LLMs are becoming increasingly complicated.
2. The framework is automated and scalable, providing an efficient approach of synthesizing datasets with multiple constraints in instructions.
3. LLMs trained on data from this framework exhibit improved performance and good generalization on instruction following tasks, indicating the effectiveness of this framework.

### Weaknesses
1. Some model-based constraints may not suitable for a binary evaluation. For example, for the "Helpfulness" constraint, it is common for LLMs to generate two helpful responses but with discrepant levels. If both of them are judged as satisfying the constraint, the gap between the two responses will be eliminated, which is not conducive to fine-grained model performance optimization.
2. The effectiveness of RLVC on reasoning models are not validated. Both Qwen-2.5-7B and Llama-3.1-8B are non-reasoning base models. Considering the widespread application of the reasoning model, it is necessary to validate the effectiveness of RLVC on base models with reasoning capabilities.
3. In Table 1, 2 and 3, it seems that the percent sign of the result values is missing. And the presentation form of results is also inconsistent across different contexts (e.g. "achieves a 31.25% average satisfaction rate" around line 349 and "achieves average scores of 34.64" around line 373).

### Questions
See Weaknesses.

### Soundness
3

### Presentation
4

### Contribution
3

### Rating
8

### Confidence
4