# ConfRAG: Confidence-Guided Retrieval-Augmenting Generation

- Avg Score: 3.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 2, 6

## Abstract
Can Large Language Models (LLMs) be trained to avoid hallucinating factual statements, and can Retrieval-Augmented Generation (RAG) be triggered only when necessary to reduce retrieval and computation costs? In this work, we address both challenges simultaneously. We introduce ConfQA, a fine-tuning strategy that reduces hallucination rates from 20–40% to below 5% across multiple factuality benchmarks. The approach is simple: when the model answers correctly, it is trained to output the answer; otherwise, it is trained to respond with “I am unsure.” Two design choices make this training effective: (1) a dampening prompt (“answer only if you are confident”) that explicitly discourages overconfident hallucinations, and (2) training data drawn from atomic factual statements (e.g., knowledge graph attribute values), which calibrates model confidence and yields robust generalization across domains and question types. Building on ConfQA, we propose ConfRAG, a triggering strategy that invokes RAG only when the model responses with unsure. This framework achieves accuracy above 95% in ideal case while reducing unnecessary external retrievals by over 30%.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces a ConfRAG, a RAG strategy that only triggers retrieval when the answer has low confidence. To implement this strategy, the authors introduce a fine-tuning method, ConfQA, which teaches the LLM to state “I am unsure” when the answer is incorrect, via a dampening prompt “Answer only if you are confident” and a training dataset composed of atomic factual statements.

### Strengths
The proposed finetuning method can teach LLM to refrain from generating inconfident outputs while simultaneously improving retrieval efficiency.

### Weaknesses
Clarity
- The paper's central premise is to use model uncertainty as a trigger for retrieval. However, there appears to be a fundamental mismatch between this trigger and the fine-tuning objective, which is based on correctness. The paper does not sufficiently address the gap between model confidence and answer correctness. For instance, a model can be uncertain about a correct answer or highly confident in an incorrect one. This discrepancy seems to undermine the core mechanism, and it is unclear how the proposed method accounts for it.

Novelty
- The paper's claimed contributions for RQ1 (confidence calibration) and RQ2 (encouraging abstention) appear to substantially overlap with existing literature. The overconfidence of LLMs [1-2] and the use of "I don't know" for selective abstention [3-5] are both well-studied topics. The authors should more clearly articulate the specific novelty of their approach in light of this extensive prior work. As written, the incremental contribution is not clear.

Experiments
- The empirical evidence is not compelling. ConQA seems to significantly harm correctness on short-form benchmarks, while the performance improvements on long-form generation tasks are marginal. For instance, in Table 5, the proposed method is consistently outperformed by a standard RAG baseline with contriever.

- The paper fails to compare against standard methods for hallucination detection (e.g., P(True) [6], semantic uncertainty [7]) or abstention (e.g., [8,9]). Adaptive RAG baselines, such as self-RAG and DRAGIN, were mentioned in the related work section but not compared experimentally.

[1] Xiong, Miao, et al. "Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs." The Twelfth International Conference on Learning Representations.
[2] Tian, Katherine, et al. "Just ask for calibration: Strategies for eliciting calibrated confidence scores from language models fine-tuned with human feedback." arXiv preprint arXiv:2305.14975 (2023).
[3] Cheng, Qinyuan, et al. "Can AI assistants know what they don't know?." arXiv preprint arXiv:2401.13275 (2024).
[4] Chen, Lida, et al. "Teaching large language models to express knowledge boundary from their own signals." arXiv preprint arXiv:2406.10881 (2024).
[5] Li, Jiaqi, Yixuan Tang, and Yi Yang. "Know the unknown: An uncertainty-sensitive method for llm instruction tuning." arXiv preprint arXiv:2406.10099 (2024).
[6] Kadavath, Saurav, et al. "Language models (mostly) know what they know." arXiv preprint arXiv:2207.05221 (2022).
[7] Kuhn, Lorenz, et al. "Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation." The Eleventh International Conference on Learning Representations.
[8] Feng, Shangbin, et al. "Don't hallucinate, abstain: Identifying LLM knowledge gaps via multi-LLM collaboration." arXiv preprint arXiv:2402.00367 (2024).
[9] Yadkori, Yasin Abbasi, et al. "Mitigating llm hallucinations via conformal abstention." arXiv preprint arXiv:2405.01563 (2024).

### Questions
Please refer to the weaknesses above.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper focuses on an important question: when should an LLM trigger retrieval. It first measures the knowledge boundaries of the LLM, then uses correctness annotations as supervision to train the model to express uncertainty when it does not know the answer, and performs retrieval only when the model expresses uncertainty.

### Strengths
1.  The paper focuses on an important question: teaching the model to recognize its own knowledge boundaries and to trigger retrieval only when it does not know the answer.
2. The paper is well written and logically coherent.
3. The paper uses the principle of “answer only if you are confident” to suppress overconfidence, and it trains on atomic facts, which leads to relatively high accuracy.

### Weaknesses
1. The paper lacks novelty — the idea of triggering retrieval only when the model does not know the answer is not new.
2. There have been many works between 2023 and 2024 that use SFT (Supervised Fine-Tuning) to enable models to express uncertainty, and this paper is not fundamentally different from those approaches.
3. The paper includes too few baselines and lacks citations to several foundational works in the areas of adaptive RAG and LLM knowledge boundary perception.


[1] SAC3: Reliable Hallucination Detection in Black-Box Language Models via Semantic-aware Cross-check Consistency. EMNLP 2023

[2] When Do LLMs Need Retrieval Augmentation? Mitigating LLMs' Overconfidence Helps Retrieval Augmentation. ACL 2024

[3] Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection. ICLR 2024

[4] Alignment for Honesty. NeurIPS 2024

[5] Teaching Models to Express Their Uncertainty in Words. TMLR 2022

### Questions
See Weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This work focuses two key challenges in factual question answering with Large Language Models (LLMs): reducing hallucinated answers and minimizing unnecessary retrieval operations in Retrieval-Augmented Generation (RAG) systems.
The authors introduce ConfQA, a fine-tuning method where the model is trained to answer only when confident and otherwise respond with “I am unsure.”
This is achieved via a dampening prompt and training on atomic factual statements.
Building upon this, they propose ConfRAG, a system that triggers RAG only if the base model expresses uncertainty, resulting in reduced hallucination rates and fewer external retrievals, while maintaining high accuracy.

### Strengths
1. The motivation is clear. This work focuses important issues of LLM hallucinations and computational efficiency in RAG.
2. The use of confidence signaling (“I am unsure”) and SFT objectives makes the framework conceptually straightforward.
3. Experimental results demonstrate improvements over baselines across multiple benchmarks, notably lowering hallucination rates.

### Weaknesses
1. The primary limitation of this work is its lack of novelty.
Training with the 'unknown' token is a commonly employed technique in many existing RAG systems (e.g., [1]).
This study does not offer substantial new insights beyond some empirical observations.

2. Several design choices are not clearly explained. For example, the rationale behind the design of the dampener prompt and the "unsure" answer remains unclear. Is model performance highly sensitive to the choice prompt and answer? Additionally, how to adjust the proportion of "unknown" answers in the training data, and is this ratio critical to overall model performance? Regarding training with negative signals, what is the justification for using SFT instead of DPO or PPO? Providing further details and analysis on these points would strengthen the contribution of this work.

[1] Enhancing Retrieval-Augmented Generation with Dehallucinating Parallel Context Extension.

### Questions
See weakness.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces ConfRAG, a framework for triggering Retrieval-Augmented Generation (RAG) based on model confidence. The authors propose ConfQA, a fine-tuning strategy that teaches LLMs to respond with "I am unsure" when they lack confidence in factual answers. The key contributions include: (1) a RAG triggering mechanism based on explicit confidence assessment, (2) a fine-tuning method using atomic factual statements from DBPedia with a "dampening prompt," and (3) comprehensive evaluation across 7 benchmarks showing hallucination reduction to <5% while maintaining accuracy comparable to always-on RAG with reduced latency.

### Strengths
- The paper methodically addresses the RAG triggering problem with clear motivation (Figure 3 shows LLMs are overconfident)
- Strong empirical results:

1. Consistent hallucination reduction across diverse benchmarks
2. Maintains or improves factuality scores
3. Practical latency improvements demonstrated


- The dampening prompt and DBPedia focus are well-motivated through ablations
- 7 benchmarks covering different question types and domains show generalization
- Detailed prompts, implementation details, and clear methodology facilitate reproduction
- The framework is lightweight (no hidden-state inspection) and compatible with existing systems

### Weaknesses
- The core contribution beyond R-Tuning and IDK appears to be (1) the dampening prompt and (2) using DBPedia instead of MMLU. While effective, this feels incremental.
- Training data limitations:

1. Only 3K samples seems small for teaching general confidence behavior
2. DBPedia focus on entity attributes may not transfer well to other factual question types
3. No systematic study of data diversity vs. quality trade-offs


- Fine-tuning results are primarily on Llama-3.1-70B. Claims about generalization would be stronger with results across model families and sizes.
- RAG baseline concerns:

1. The "ideal RAG" assumption (Section 3.1) is unrealistic
2. Real RAG results (Table 3) show modest improvements, and on CRAG, RAG-everywhere still underperforms LLM-only
3. More sophisticated RAG methods should be compared


- Evaluation gaps:

1. No human evaluation to validate LLM-as-a-judge decisions
2. The "ceiling" metrics assume perfect RAG, which may be misleading
3. Missing analysis of failure modes (when does ConfQA incorrectly say "unsure"?)



- Limited theoretical insight: Why does the dampening prompt work so well? Why does DBPedia generalize better than MMLU? The paper is primarily empirical without deeper understanding.

### Questions
- Can ConfQA fine-tuned on Llama-3.1-70B be used to trigger RAG for other models (e.g., GPT-4o)? Or does each model need separate fine-tuning?
- How sensitive is the approach to the exact wording of the dampening prompt? Have you experimented with variations?
- What is the optimal mix of entity popularities (head/torso/tail) in training data? Current 1K/1K/1K split seems arbitrary.
- In Table 2, ConfQA still has 5.2% incorrect on DBPedia (in-domain). What characterizes these failures? Are they specific entity types or question patterns?
- How do you determine if a question requires dynamic information? This seems crucial for the architecture in Figure 2.
- How does this compare to calibration techniques or uncertainty estimation methods that don't require fine-tuning?
- What is the cost of fine-tuning vs. the savings from reduced RAG calls? Is this approach cost-effective at scale?
- he long-form results (Table 5) show modest improvements. Why doesn't the approach transfer as well to multi-claim scenarios?
- Is binary "unsure" vs. answer optimal? Have you explored soft confidence scores that could enable more nuanced triggering?
Real-world deployment: Have you deployed this in production? What practical challenges emerged beyond the experimental setting?

### Soundness
3

### Presentation
3

### Contribution
3
