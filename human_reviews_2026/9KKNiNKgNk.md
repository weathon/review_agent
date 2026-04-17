# Train for Truth, Keep the Skills: Binary Retrieval-Augmented Reward Mitigates Hallucinations

- Decision: Reject
- Scores: 2, 6, 2, 4

## Abstract
Language models frequently generate factually incorrect information with high confidence, a phenomenon known as extrinsic hallucination. Existing approaches for improving factuality often come at the cost of diminished performance on other downstream tasks, limiting their practical deployment.
We propose a novel on-policy reinforcement learning (RL) approach that uses binary retrieval-augmented rewards (RAR) to address this challenge. Our binary reward scheme assigns a reward of zero whenever any factual error is detected and one otherwise. 
We evaluate our method through continual RL from Qwen3 models across multiple tasks. For open-ended generation, binary RAR achieves a 39.3\% reduction in hallucination rates, significantly outperforming supervised training or online RL with dense reward. 
In short-form settings, models learn calibrated abstention, answering ``I don’t know'' when parametric knowledge is insufficient, leading to 44.4\% and 21.7\% fewer incorrect answers on PopQA and GPQA, respectively. 
Crucially, these factuality gains come without performance degradation on instruction following, math, or code, whereas dense-reward RL, despite improving factuality, induces quality regressions.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces Retrieval-Augmented Reward, an online RL method that 0/1 rewards to mitigate hallucinations by verifying outputs against retrieved evidence. The approach achieves reduction in hallucination rates while preserving general capabilities. They claim that binary rewards are more robust than continuous alternatives, avoiding reward hacking and utility degradation.

### Strengths
1. Binary RAR demonstrates both computational efficiency advantages over dense reward methods and superior hallucination reduction performance.
2. The experimental design systematically evaluates both hallucination reduction and capability preservation, demonstrating that Binary RAR achieves 39.3% factuality improvement while maintaining utility.

### Weaknesses
The main concerns for this paper that it provided limited explanation that binary reward design should be better.
1. The paper's claim that 0/1 rewards are more effective at mitigating hallucinations lacks adequate explanation of the underlying mechanism. Based on Figures 2 and 3, the binary reward's effectiveness appears to stem from making the model significantly more conservative—refusing to answer most questions, providing less information, and generating shorter responses. However, this conservative behavior may not always be desirable. The authors should provide more discussion about the impact on content recall rather than focusing primarily on precision.
2.  The paper fails to sufficiently explain why 0/1 rewards better preserve general capabilities. A more detailed analysis of the major performance changes in Table 2 benchmarks (such as IFEval and GSM8K) is needed to better understand the advantages of binary rewards. The paper should investigate why binary rewards avoid the utility degradation observed with dense rewards.
3. The experiments are conducted exclusively on Qwen3 models (4B and 8B), lacking validation across diverse model families.

### Questions
1. What is the underlying mechanism that makes binary rewards outperform dense rewards for hallucination mitigation?
2. Could the authors provide analysis of training dynamics, including what percentage of prompt-response pairs receive 0 vs 1 rewards during training?

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes a Binary Retrieval-Augmented Reward (RAR) framework for reinforcement learning to reduce hallucinations in large language models. The method verifies model outputs against retrieved evidence using an LLM judge and optimizes a binary reward through the GRPO algorithm. This design replaces complex continuous factuality scores with a simple verifiable signal focused on factual correctness. Experiments on Qwen3-8B and Qwen3-4B show clear gains in factual precision and reduced QA errors, while maintaining instruction, math, and coding abilities. The model also learns calibrated abstention when evidence is lacking, offering a scalable path to improve factual reliability.

### Strengths
1. The paper replaces traditional continuous factuality scores with a binary retrieval-augmented reward, explicitly optimizing the model toward *avoiding factual errors*. This design leads to a simple yet effective mechanism for factual correction and hallucination reduction.
2. The experiments are well-structured and comprehensive, including ablation studies, multi-task and multi-model evaluations, and assessments of general capabilities. The results appear credible and consistent.
3. The paper is clearly written, well-organized, and supported by intuitive figures and tables that effectively illustrate both the method and findings.
4. In terms of significance, the work demonstrates a substantial reduction in hallucinations without sacrificing general capabilities, offering a scalable paradigm for building more reliable and safer LLMs. It makes a meaningful contribution to the application of reinforcement learning for model alignment.

### Weaknesses
1. The optimization objective fully depends on retrieved evidence — the binary reward is determined by whether the retrieved documents contradict the model output. Since retrieval systems inherently contain bias (e.g., source preference, recency errors, and incomplete coverage), the model is effectively optimized to align with retrieval consensus rather than the ground truth. In essence, the model learns retrieval alignment, not truth alignment.
2. This RLKF paper (https://arxiv.org/abs/2403.18349) has already introduced binary knowledge-feedback rewards ****within a PPO framework to train models to respond or abstain based on verifiable knowledge. The current work mainly differs by (i) replacing PPO with GRPO, and (ii) defining the binary signal through retrieval-augmented evidence checking. These are implementation-level modifications rather than conceptual innovations. Without deeper analysis of learning dynamics or generalization behavior, the contribution appears incremental.
3. Reward design may encourage conservative or evasive behavior. With a “no contradiction = 1, otherwise 0” reward, the safest strategy under incomplete evidence is to produce shorter or abstaining outputs. The paper’s own results show high abstention rates (55.2% on PopQA and 27.5% on GPQA), alongside decreased factual accuracy (PopQA 20.4 → 18.0; GPQA 48.9 → 44.2). This suggests that the reported hallucination reduction might partly result from answering less, not answering more accurately. The authors should report recall and response-part accuracy ****to clarify whether improvements stem from better correctness or more frequent refusal.

### Questions
1. Why did you choose GRPO instead of trying other RL algorithms such as Reinforce++ or DAPO?
2. Why not directly train with ground truth instead of constructing retrieval candidates? The dataset already contains ground-truth responses. Instead of directly using them to compute factual consistency rewards, the paper constructs a retrieval pool seeded by the ground-truth answer and then measures agreement between model outputs and retrieved evidence. Is this two-step design necessary? Could this intermediate retrieval layer introduce additional noise or bias?
3. If the retrieved documents contain contradictions or errors, how does the method decide which one is more trustworthy? Is there any consistency filtering, aggregation, or multi-judge mechanism? Moreover, how sensitive is the binary reward to the choice of retriever (e.g., BM25 vs. dense retrieval) and the judge model?
4. It would be informative to include recall and response-part accuracy to differentiate between “answering less” and “answering better.”In addition, could the authors test on simpler QA datasets? If the dataset is easier and more closed-domain, would the model become even more conservative (i.e., show a stronger accuracy drop), revealing a deeper behavioral bias?
5. Have the authors evaluated whether the trained model itself demonstrates generalization beyond factual QA — for example, on mathematical reasoning, code generation, or multi-step logical tasks?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a Binary Retrieval-Augmented Reward (Binary RAR) reinforcement learning approach that uses a 0/1 reward signal to train large language models to reduce hallucinations, while claiming not to degrade their general capabilities. The method determines whether the model’s responses contradict retrieved documents and assigns a reward of 1 when no contradiction is found. Experiments on Qwen3-8B/4B demonstrate that the approach significantly improves factuality and induces calibrated abstention behavior, where the model appropriately responds with “I don’t know.”

### Strengths
1. The method is simple and effective, easy to implement from an engineering perspective, and demonstrates greater robustness compared to dense reward models.
2. The experimental design simultaneously addresses both hallucination mitigation and preservation of general capabilities.
3. Experimental results show that the proposed approach improves the model’s factual accuracy, achieving strong overall performance.

### Weaknesses
1. The authors claim that full-text contradiction detection avoids the “error accumulation” of claim-wise verification, but this statement lacks justification. Claim-level verification is an independent process without cumulative error, whereas full-text inputs may introduce contextual interference and order bias. A controlled comparison between the two verification granularities is recommended to support this claim.
2. The “I don’t know” samples are not human-labeled but automatically detected through string matching, without considering semantically equivalent expressions or manual validation. The appendix prompt explicitly instructs judgments to rely only on retrieved text, ignoring the model’s internal (parametric) knowledge—meaning the model is trained to answer “I don’t know” even when it actually knows the answer. The prompt design and labeling mechanism should be refined to better distinguish retrieval gaps from genuine uncertainty.
3. The paper repeatedly emphasizes that “general capabilities do not degrade,” but this is largely enforced by the training configuration (early stopping and KL regularization) rather than emerging naturally. The comparison is therefore not entirely fair. Experiments using dense reward methods under the same settings should be added to verify whether Binary RAR truly achieves better factuality–utility balance.

### Questions
1. Has there been a controlled comparison between VeriScore and Binary RAR under the same KL coefficient? Without such an experiment, it is difficult to determine whether the observed balance between factuality and utility arises from the binary reward mechanism itself or simply from the degree of KL regularization applied.

2.How does the evaluation of “I don’t know” responses account for the influence of the model’s internal parametric knowledge?
Even if the assessment is restricted to external (retrieval-based) knowledge, how consistent are the “I don’t know” labels between model-generated judgments and human evaluation? Clarifying this alignment is essential to verify that the model’s abstention truly reflects epistemic uncertainty rather than retrieval insufficiency.

### Soundness
2

### Presentation
2

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
This paper applies reinforcement learning methods to mitigate hallucinations in large language models. Specifically, after the model generates a response, relevant evidence is retrieved, and a binary reward signal is constructed based on whether factual conflicts exist between the model’s response and the retrieved evidence. Experimental results demonstrate that this approach is effective in reducing hallucinations.

### Strengths
* The writing is clear, and the proposed method is straightforward.
* The experiments are thorough, and the work is relatively comprehensive.

### Weaknesses
* The main difference between this work and the most relevant baseline, VeriScore, lies in the design of the reward signal. While VeriScore uses a soft reward based on the proportion of correct claims, this paper adopts a hard binary reward that simply judges whether any factual inconsistency exists. Apart from this modification, there are no substantial differences between the two approaches. Overall, this incremental improvement makes the work resemble more of a technical report than a full research contribution.
* In the experimental results, apart from Factual Precision, the task-level performance metrics (e.g., accuracy) are not reported in detail. In addition, Table 1 uses Factual Precision for the long-form setting but Hallucination Rate for the short-form setting. It would be helpful to clarify the rationale behind using different evaluation metrics for these two settings.
* Does the choice of model used for reward computation have a significant impact on the results?
* It is unclear whether the conclusions would remain valid if alternative reinforcement learning algorithms, such as PPO, were employed. A discussion or comparison on this aspect would strengthen the paper.

### Questions
None

### Soundness
1

### Presentation
3

### Contribution
2
