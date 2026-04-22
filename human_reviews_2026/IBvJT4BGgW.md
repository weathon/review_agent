# Efficient Multimodal Planning Agent for Visual Question-Answering

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 4, 4, 8

## Abstract
Visual Question-Answering (VQA) is a challenging multimodal task that requires integrating visual and textual information to generate accurate responses. While multimodal Retrieval-Augmented Generation (mRAG) has shown promise in enhancing VQA systems by providing more evidence on both image and text sides, the default procedure that addresses VQA queries, especially the knowledge-intensive ones, often relies on multi-stage pipelines of mRAG with inherent dependencies. To mitigate the inefficiency limitations while maintaining VQA task performance, this paper proposes a method that trains a multimodal planning agent, dynamically decomposing the mRAG pipeline to solve the VQA task. Our method optimizes the trade-off between efficiency and effectiveness by training the agent to intelligently determine the necessity of each mRAG step. In our experiments, the agent can help reduce redundant computations, cutting search time by over 60\% compared to existing methods and decreasing costly image retrieval calls. Meanwhile, experiments demonstrate that our method outperforms all baselines, including a carefully designed prompt-based method, on average over six various datasets. Code will be released at https://github.com

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper addresses the inefficiency of static multi-stage pipelines in multimodal Retrieval-Augmented Generation (mRAG) for Visual Question-Answering (VQA)—a task requiring integration of visual and textual information. To solve this, the paper proposes a multimodal planning agent trained to dynamically decompose the mRAG pipeline: it intelligently omits unnecessary steps (e.g., skipping retrieval for simple queries) and selects only essential components (e.g., image/text retrieval for knowledge-intensive queries) based on the input VQA query.

### Strengths
1. Proposes a dynamic agent that optimizes mRAG pipelines for VQA, moving beyond static pipelines and prompt-based methods by integrating actionable decision-making (not just knowledge boundary detection).
2. Reduces search time by over 60% and minimizes expensive image retrieval calls compared to baselines (e.g., OmniSearch, full mRAG), addressing scalability challenges in real-world VQA.
3. Outperforms all baselines (including prompt-based methods and full mRAG) on average across six diverse VQA datasets, demonstrating that efficiency does not come at the cost of accuracy.
4.  The fine-tuned 7B-scale agent enhances performance across other MLLMs (e.g., GPT-4o, Qwen-VL-Max) without additional fine-tuning, highlighting its generalizability.

### Weaknesses
1. While the agent is designed to handle diverse queries, the paper does not deeply analyze how it adapts to specific query categories (e.g., knowledge-intensive vs. simple, visual vs. text-heavy). For example, it shows the agent skips mRAG for 60% of NoCaps queries but does not explain why these queries are solvable without retrieval (e.g., are they simple object-recognition questions?).
2. The paper only reports search time (time for retrieval steps) but not end-to-end inference latency (including MLLM generation time). In real-world VQA, generation time often dominates latency—especially if the agent reduces input sequence length (by skipping retrieval, thus shortening context). Without this measurement, the paper’s efficiency claims are incomplete.
3. The training data relies on three datasets: InfoSeek, VQAv2.0 (general VQA), and WanWu (Chinese VQA). There is no evaluation on domain-specific VQA (e.g., medical VQA, satellite image VQA), where mRAG is often critical (e.g., retrieving medical knowledge for image-based diagnosis). This limits the paper’s claim of the agent’s "adaptability in open-domain VQA."
4. The innovation of the paper is limited, as many engineering practices have adopted similar approaches. The method suffers from a significant scaffolding issue. Additionally, compared to Vision-Language Deep Research Agents, what are the differences in end-to-end performance and efficiency for this router-based method?

### Questions
See the weaknesses.

### Soundness
3

### Presentation
4

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
The paper’s core idea, learning to pick among four retrieval/actions to avoid overusing multimodal RAG, is timely and practically useful, and the empirical comparison to a strong prompt-based controller with ~60% search-time reduction is attractive. The evaluation is also commendably broad across six datasets, and the fact that the learned planner transfers to other MLLMs suggests the approach is not overfit to one model. However, a few high-impact weaknesses remain: (i) the whole pipeline leans on unquantified LLM-generated labels, with no noise/quality reporting (W1); (ii) the main effectiveness numbers use LLM-based scoring instead of standard VQA metrics or human evaluation, so the gains are hard to situate in the VQA literature; (iii) efficiency is reported via estimated tool-time, not end-to-end latency, so the headline 60% saving is not fully substantiated; and (iv) there is no comparison to other learned/routing agents, so it is unclear how much is due to the specific proposed planner vs. “any learned router”, combined with rebalanced training that may not reflect real-world distributions, and a lack of significance reporting.

### Strengths
S1: Targets a concrete, high-impact pain point in multimodal RAG for VQA, unnecessary image/text retrieval and inflated context length, by turning the pipeline into an adaptive one.

S2: Automated LLM-based data construction (visual query decomposition + correctness checking) enables building large supervision without heavy manual labeling.

S3: Broad evaluation on six heterogeneous VQA(-like) datasets, demonstrating generality across dynamic, knowledge-intensive, and easier visual tasks.

S4: Strong empirical gain over the prompt-based OmniSearch controller: similar or better answer quality with about 60%–66% lower search time.

S5: Shows cross-model transfer to other MLLMs (GPT-4o, Qwen-VL-Max, DeepSeek-VL), suggesting the learned policy is not tied to a single backbone.

### Weaknesses
W1: Despite the automated LLM-based data pipeline being a key enabler, the approach still relies heavily on LLM-generated supervision for both decomposition and correctness checking, but the paper does not quantify annotation noise or its impact.

W2: Although the evaluation covers six datasets and looks broad, it leans on LLM-based scoring rather than standard VQA metrics or human judgment, which weakens comparability to prior VQA work.

W3: While the method reports a 60%+ search-time reduction over a strong prompt-based controller, this efficiency is derived from estimated tool/API times rather than end-to-end wall-clock measurements on a fixed setup.

W4: The four routing categories are well motivated and interpretable, but the training set is rebalanced across them, so the learned policy is not evaluated under the real, potentially skewed, test-time distribution.

W5: The system still depends on an inference-time query-rewriting MLLM; if that component changes or is weaker in deployment, some of the claimed savings/accuracy may erode.

W6: There is no analysis of misrouted or failed cases (e.g., choosing no mRAG when external knowledge was needed), so failure modes remain opaque.

W7: The pipeline uses multiple Qwen variants/APIs for different substeps, which could hurt replicability when those endpoints change.

### Questions
Q1: Since you rely on LLM-based scoring, add at least one standard VQA-style metric (e.g., VQAv2 accuracy / EM) on the English datasets and report variance or significance for the LLM-eval scores, so we can judge robustness.

Q2: Please quantify the quality of the LLM-generated supervision: report error/noise rates for visual query decomposition and correctness checking on a manually annotated subset, and add an ablation showing planner accuracy with/without cleaned labels.

### Soundness
3

### Presentation
3

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
This paper addresses Visual Question Answering (VQA) by proposing a multimodal planning agent to optimize multimodal Retrieval-Augmented Generation (mRAG) workflows. The key idea is to dynamically decide whether to perform image retrieval, text retrieval, both, or neither, rather than executing all steps in a fixed pipeline. The authors train this agent through automatic data annotation and fine-tuning, and evaluate it across multiple VQA datasets.

### Strengths
- The paper identifies a real problem: existing mRAG pipelines suffer from multi-stage dependencies and redundant computations. This is a well-motivated research direction.
- The method is essentially a classifier that predicts which retrieval path to take for each VQA input. This makes it easy to implement and integrate into existing systems. The automatic data generation process also avoids expensive manual annotation.
- The experiments results show significant reduction in retrieval operations and search time while maintaining comparable task performance across multiple datasets.

### Weaknesses
- My main concern is the significance of the contribution. The proposed method is essentially training an LLM-based classifier for a four-way classification task, rather than a true multi-step planning agent. While effective in practice, the methodological innovation is relatively limited.
- The evaluation scope is somewhat narrow: 1) The main results rely solely on LLM Eval scores (0-100) without standard VQA metrics like accuracy, BLEU, or ROUGE. 2) Only Qwen-Max is used for evaluation, and the evaluation protocol (prompts, rubrics) is not fully transparent.
- The paper does not compare against specialized mRAG optimization methods or recent VQA SOTA systems (such as state-of-the-art agent-based VQA approaches). The comparisons feel more like ablations rather than positioning against the broader literature.
- Training labels are entirely generated by automated models without reported quality checks or human verification rates.
- All scores are single values without error bars, confidence intervals, or significance tests.

### Questions
The paper describes the main contribution as a "planning agent," but the system only makes a single four-way classification decision rather than multi-step planning. Could you explain why you call it a "planning agent" instead of a "strategy classifier"? Do you plan to extend it to multi-step, context-aware decision-making in future work?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces a multimodal planning agent designed to optimize the efficiency of Retrieval-Augmented Generation (mRAG) for Visual Question-Answering (VQA) tasks. Existing mRAG approaches often utilize rigid, multi-stage pipelines (e.g., always performing image retrieval, then query rewriting, then text retrieval), which can be computationally expensive, slow, and sometimes redundant for simpler queries. The authors propose a dynamic approach where a fine-tuned 7B agent (Qwen2.5-VL-7B-Inst) analyzes the input VQA query and determines the necessary mRAG steps: no RAG, image RAG only, text RAG only, or both.

The agent trains by using an automated data construction pipeline that employs a larger, stronger MLLM (Qwen2.5-VL-72B) in order to generate "gold queries" and determine which ground truth labels for which retrieval steps are actually required to correctly answer a given training question.

Empirical results across six diverse VQA datasets demonstrate that this agent-based approach reduces search time by over 60% compared to prompt-based baselines (OmniSearch) and significantly cuts down on expensive image retrieval calls while maintaining or slightly improving overall VQA accuracy. The paper also shows that this planning agent can effectively guide other downstream MLLMs (e.g., GPT-4o, DeepSeek-VL) without further fine-tuning.

### Strengths
Practical Significance: Addressing the inefficiency of rigid RAG pipelines is a highly relevant problem for real-world deployment of MLLMs. The reported >60% reduction in search time is a substantial practical improvement.

Strong Empirical Results: The approach strikes a balance of efficiency with strong performances. It manages to match or outperform the computationally expensive "$+k_{i,t}$" full RAG baseline, while using a fraction of the retrieval calls.

Transferability: A strength of the agent is that it can improve the performance of other, closed-source MLLMs (such as GPT-4o) when used as a plug-and-play planner. An example is shown in Table 5. This increases the general utility of the proposed method.

Comprehensive Evaluation: The use of six diverse datasets, ranging from the standard benchmarks of Visual7W to the more complex and dynamic DynVQA, provides a robust assessment of the agent's adaptability.

### Weaknesses
Dependency on Oracle for Training Data: Automated data annotation has a strong dependency on establishing if a model can answer a question with/without RAG. This may be inherently noisy if the base model used for annotation itself has no clearly demarcated performance boundaries. The authors generate the gold query with a strong model, Qwen-72B, but perhaps biases in that model's knowledge boundary propagate to the agent.

Comparison to Baseline for "Agents" is Limited: The main comparison to dynamically generated approaches is to OmniSearch, a prompt-based approach. Comparisons to other recent adaptive RAG approaches (albeit unimodal, adapted for the task) would better support the claim of outperforming existing planning approaches.

### Questions
1. Latency Overhead: While the search time is reduced, what is the computational overhead of introducing the extra agent step (inference of the 7B model) before the main solver? Is this negligible compared to the retrieval latencies listed in Section 5.1?

2.  Sensitivity to Annotation Model: How sensitive is the final agent's performance to the choice of the MLLM used for the automated data annotation phase - currently Qwen2.5-VL-72B? Would using a weaker model for annotation significantly degrade the planner's judgment?

3.  Understanding LoRA rank: In Section 5.3, you mention that LoRA (r=8) failed while (r=32) succeeded. Do you have any guesses as to why rank 8 was not sufficient for this task, since often in fine-tuning of NLP tasks, r=8 is sufficient? Does this suggest that the routing task requires capturing more complex relationships than typical instruction-following?

4. Training Data Balance: Table 1 is a representative balance of the final training data categories, such as 30k for No mRAG, Query mRAG, Both mRAG, but only 8.8k for Image mRAG, and so on until the very last. Was this imbalance intentional, based on natural distribution? And did this affect the agent's willingness to select specifically "Image mRAG"?

### Soundness
3

### Presentation
3

### Contribution
3
