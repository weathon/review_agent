# Route Before Retrieve: Activating Latent Routing Abilities of LLMs for RAG vs. Long Context Selection

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 4, 6, 6

## Abstract
Recent advances in large language models (LLMs) have expanded the context window to beyond 128K tokens, enabling long-document understanding and multi-source reasoning. A key challenge, however, lies in choosing between **retrieval-augmented generation (RAG)** and **long-context (LC)** strategies: RAG is efficient but constrained by retrieval quality, while LC supports global reasoning at higher cost and with position sensitivity. Existing methods such as *Self-Route* adopt failure-driven fallback from RAG to LC, but remain passive, inefficient, and hard to interpret. We propose **Pre-Route**, a proactive routing framework that performs structured reasoning *before* answering. Using lightweight metadata (e.g., document type, length, initial snippet), Pre-Route enables task analysis, coverage estimation, and information-need prediction, producing explainable and cost-efficient routing decisions. Our study shows three key findings: (i) LLMs possess latent routing ability that can be reliably activated with guidelines, allowing single-sample performance to approach that of multi-sample (Best-of-N) results; (ii) linear probes reveal that structured prompts sharpen the separability of the "optimal routing dimension" in representation space; and (iii) distillation transfers this reasoning structure to smaller models for lightweight deployment. Experiments on LaRA (in-domain) and LongBench-v2 (OOD) confirm that Pre-Route outperforms Always-RAG, Always-LC, and Self-Route baselines, achieving superior overall cost-effectiveness.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper investigates how to choose an optimal strategy over **Long Context** and **RAG** for long context LLM situations. The methods use LLMs to decide the strategy based on the meta information of the long context. The experiments over LARK and LongBenchv2 showed that the proposed method achieves lower cost and better performance.

### Strengths
1. The studied problem is important, as it is useful for long context LLM inference.
2. The method is novel, where LLMs utilize meta information to judge the strategy. The results on two classic Benchmark is validated as effective.

### Weaknesses
1. The presentation is hard to understand, especially for the methods. Do you explain what the meaning of (m_i, T_i, y_i) is in equation 4?
2. The accuracy of methods over the benchmarks is very low. I know it's not caused by methods. However, it's weird to see the analysis of such poor performance.  
3. The metadata is not available for every long context situation. Do you provide  the solution for such problems?

### Questions
See weakness.

### Soundness
2

### Presentation
2

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
This paper proposes Pre-Route, a framework for choosing between RAG and LC processing before answering queries . The core approach involves using lightweight metadata (document type, length, title, initial snippet) to perform structured reasoning. The authors claim LLMs possess "latent routing ability" that can be activated through structured prompts, validated via Best-of-N sampling and linear probing experiments . They distill this capability from large models (Qwen3-235B, DeepSeek-R1) into smaller ones (Qwen3-1.7B) using rejection sampling on an "ideal label" defined by QA performance with RAG preference in ties. Experiments on LaRA (in-domain) and LongBench-v2 (out-of-domain) show Pre-Route achieves better accuracy-cost trade-offs than Self-Route and fixed baselines.

### Strengths
**1. Comprehensive Experimental Validation**
The paper presents extensive experiments across multiple dimensions: behavioral analysis via Best-of-N sampling showing accuracy improvement from 0.53 (N=1) to 0.87 (N=8) under direct prompting; representation analysis using linear probes demonstrating that structured prompts improve ideal-label accuracy from 0.396 to 0.625 on Qwen3-1.7B; evaluations on 6+ answer models ranging from 1.7B to 235B parameters; and robustness tests under different retrieval configurations (rerank size 5/7/10)

**2. Interpretable Decision Process**
The structured six-step reasoning chain provides transparency in routing decisions, addressing a key limitation of Self-Route's black-box nature. The ablation study confirms each step contributes to the final performance, with removing decision rules causing LC rate to spike from 20.7% to 45.3

### Weaknesses
**1. Lack of Justification for Design Choices**
The design lacks theoretical grounding and empirical support for why these specific meta-information features and guidelines were chosen. The paper does not explain the principled basis for selecting document type, length, title, and initial snippet as the metadata inputs, nor does it justify why the six reasoning steps are structured in this particular order and formulation. No ablation study explores alternative metadata configurations or reasoning structures, making the design appear arbitrary rather than well-motivated.

**2. Highly Engineered Approach with Limited Innovation**
The method is a highly customized heuristic system—essentially a hand-crafted prompt engineering solution. The six reasoning steps (task & document characterization, distribution pattern judgment, context-window feasibility, retrieval feasibility, model capability consideration, efficiency trade-off) are manually designed heuristics that require pre-specification rather than being automatically learned or generated. This pre-designed nature significantly limits innovation, as the approach lacks unique architectural contributions beyond careful prompt construction. The core claim of "activating latent routing ability" essentially reframes better prompting as a discovery of hidden capabilities, which overstates the conceptual novelty

**3. High Inference Cost and Missing Comparison with Modern Agentic RAG Paradigms**
The framework incurs substantial additional computational overhead by requiring deployment of a separate routing LLM just to decide between RAG and LC. This stands in contrast to mainstream Agentic RAG [1][2] approaches, which internalize the routing capability within the answering model itself through reinforcement learning, enabling the model to dynamically switch between RAG and LC during generation without external routing overhead. The paper completely lacks discussion of such RL-based integrated approaches, which further diminishes its claimed innovation.

**4. Limited Generalization and Marginal Performance Gains**
The effectiveness is only substantial on in-domain data. The LongBench-v2 out-of-domain results show Pre-Route's advantage narrows considerably, with Self-Route becoming "more competitive in QA scores", suggesting the approach may not generalize beyond LaRA's 4-point scoring regime. This raises questions about whether the added complexity of distillation and separate router deployment justifies the marginal gains.

[1] Singh A, Ehtesham A, Kumar S, et al. Agentic retrieval-augmented generation: A survey on agentic rag[J]. arXiv preprint arXiv:2501.09136, 2025.

[2] Liang J, Su G, Lin H, et al. Reasoning RAG via System 1 or System 2: A Survey on Reasoning Agentic Retrieval-Augmented Generation for Industry Challenges[J]. arXiv preprint arXiv:2506.10408, 2025.

### Questions
**1. Missing Detailed Linear Probing Experimental Setup**
What is the exact training data and label generation procedure for the linear probes? The paper states that probes are trained on "frozen penultimate last token embeddings" , but does not specify: (a) the size of the training set used for probe fitting; (b) whether the same train/val/test split from LaRA is used or a separate split; (c) the optimization procedure (optimizer, learning rate, number of epochs); (d) how the "ideal," "route," "doc_type," and "task_type" labels are generated for the probing dataset. Without these details, the linear probing results in Table 1 cannot be reproduced or properly interpreted.

**2. Inadequate Explanation of Table 2 (Cost Analysis)**
Table 2 lacks detailed explanation of the experimental setup. Specifically: (a) What is the exact experimental configuration—are these costs averaged over the entire LaRA test set or based on representative samples? (b) Why do the two models with vastly different parameter counts (Qwen3-235B vs. Qwen3-1.7B) have identical input and output token counts (1205 input, 648 output) for Pre-Route? This seems implausible unless the prompts and reasoning chains are exactly the same length regardless of model size. (c) Why does Self-Route have 2600 input tokens compared to Pre-Route's 1205—where does this additional input come from? Is Self-Route including retrieved chunks in the routing decision, while Pre-Route uses only metadata? This critical distinction is not clarified.

**3. Potential Annotation Error in Main Results**
In Table 3 (LaRA main results), under the Qwen3-235B [T] answer model configuration, Always-LC (Baseline) achieves a QA score of 3.51, which appears to be the highest score in that column.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces Pre-Route, a proactive framework to address the challenge of choosing between efficient Retrieval-Augmented Generation and costly Long-Context methods. Unlike reactive models, Pre-Route uses lightweight metadata for structured reasoning before answering, enabling it to make explainable and cost-efficient routing decisions. The study shows this approach activates LLMs' latent routing abilities and can be distilled into smaller models.

### Strengths
1. This paper introduces "Pre-Route," a novel framework that shifts from a reactive to a proactive "plan-then-execute" paradigm. By performing structured reasoning before generating an answer, it makes more intelligent, cost-efficient decisions and outperforms existing baselines.
2. The research empirically proves that LLMs have a latent, inherent ability for routing. 
3. The complex routing logic can be successfully distilled into smaller, lightweight models. This transforms the sophisticated reasoning into a practical, plug-and-play module, enabling efficient and low-cost deployment in real-world applications where large models are not feasible.

### Weaknesses
1. The experimental setup for Figure 2 is highly unclear, as key details such as the task type and context length are not specified. Furthermore, it is not explained how the "answer directly" and "unconstrained CoT" methods were implemented. The authors also fail to explain why Pre-route's performance is even inferior to the other two approaches when the Best-of-N sample size is 4 or greater.
2. The main experiments lack a direct comparison against the "answer directly" and "unconstrained CoT" settings.
3. The scope of Pre-Route is presented as overly constrained, limited to the binary choice between RAG and LC. Nevertheless, considering the described methodology and experimental analysis, this framework shows potential for application in a broader spectrum of model routing scenarios.
4. The tables on pages 8 and 9 are difficult to read because the text is too small and the layout is disorganized. The authors should select and present only the most critical information in the main text and move the rest to the appendix.

### Questions
please see Weaknesses

### Soundness
2

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
This paper proposes the Pre-Route framework: an proactive and efficient routing framework that implements a "plan-first, execute-later" model. It uses structured reasoning to select between RAG (Retrieval-Augmented Generation) and LC (Local Computation) before generating responses.  Through experiments such as Best-of-N sampling and linear probe analysis, it proves that LLMs inherently possess the ability to decide whether to use RAG or LC. This capability, however, remains in a "dormant" state. The structured guidance of Pre-Route can effectively "activate" and stabilize this capability.  It successfully distills this complex routing planning capability from expensive large models to smaller models. This enables smaller models to make high-quality routing decisions, significantly reducing deployment costs.  Whether on in-domain (LaRA) or out-of-domain (LongBench-v2) datasets, the Pre-Route framework outperforms existing baseline methods (e.g., Always-RAG, Always-LC, Self-Route), achieving a better balance between performance and cost.

### Strengths
The paper breaks away from the passive "switch-after-failure" paradigm (e.g., Self-Route) and proposes a new paradigm of "Proactive Routing" (Pre-Route). This "plan-first, execute-later" approach is logically more superior and efficient.
The most prominent highlight of Pre-Route is that it only uses "low-cost metadata" (such as document type, length, and opening snippets) for planning. This means the computational cost of the "planning" step is nearly zero. It does not require reading the entire document in advance or running an expensive RAG process first just to make a decision. This is highly appealing in engineering practice.
Additionally, the paper successfully distills this complex "planning capability" into a small model with only 1.7B parameters, while still maintaining excellent performance. This enables the Pre-Route framework to serve as a "lightweight, low-cost, and plug-and-play" module.

### Weaknesses
1. The Definition of "Ideal Label" May Be Overly Idealized
The paper defines "Ideal Label" as follows: between RAG and LC, the one with a higher QA score is selected (RAG is chosen in case of a tie). This definition is based on an "after-the-fact" omniscient perspective. In practical applications, however, it is impossible to know in advance which method will yield a higher score.
The Pre-Route model learns to fit this "Ideal Label", but the label itself is generated in a controlled environment where "all results are known". Can this be generalized to new, unseen documents and questions? In other words, has the model learned the "statistical patterns" specific to the LaRA dataset, or truly universal "routing reasoning capabilities"? The paper dedicates significant space in Section 2 to arguing this point (e.g., using linear probes), but "correlation" does not equate to "causality".
2. Dependence on Metadata Is a "Weakness"
Pre-Route relies heavily on low-cost "metadata" (titles, lengths, document types, opening snippets).
What if the quality of the metadata is poor?
For web pages, chat records, or code repositories without a clear "document type" (doc_type), the initial reasoning step of Pre-Route may fail.
If the opening snippet (doc_head) of a document is misleading (e.g., the opening resembles a factual retrieval task, but the core lies in complex reasoning at the end of the document), Pre-Route may be misled into selecting RAG.
The paper's experiments seem to focus on relatively well-structured documents (novels, papers, financial reports). Its robustness on "messier", unstructured real-world data has not been fully verified.

### Questions
see weakness

### Soundness
2

### Presentation
3

### Contribution
2
