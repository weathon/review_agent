# Inference Scaling for Long-Context Retrieval Augmented Generation

- Avg Score: 8.00
- Decision: Accept (Oral)
- Scores: 8, 8, 8, 8

## Abstract
The scaling of inference computation has unlocked the potential of long-context large language models (LLMs) across diverse settings. For knowledge-intensive tasks, the increased compute is often allocated to incorporate more external knowledge.  However, without effectively utilizing such knowledge, solely expanding context does not always enhance performance. In this work, we investigate inference scaling for retrieval augmented generation (RAG), exploring the combination of multiple strategies beyond simply increasing the quantity of knowledge, including in-context learning and iterative prompting. These strategies provide additional flexibility to scale test-time computation (e.g., by increasing retrieved documents or generation steps), thereby enhancing LLMs’ ability to effectively acquire and utilize contextual information. We address two key questions: (1) How does RAG performance benefit from the scaling of inference computation when optimally configured? (2) Can we predict the optimal test-time compute allocation for a given budget by modeling the relationship between RAG performance and inference parameters? Our observations reveal that increasing inference computation leads to nearly linear gains in RAG performance when optimally allocated, a relationship we describe as the inference scaling laws for RAG. Building on this,  we further develop the computation allocation model to estimate RAG performance across different inference configurations.  The model predicts optimal inference parameters under various computation constraints, which align closely with the experimental results. By applying these optimal configurations, we demonstrate that scaling inference compute on long-context LLMs achieves up to 58.9% gains on benchmark datasets compared to standard RAG.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper systematically investigates the performance of RAG systems as inference computation resources scale up, demonstrating an almost linear improvement in RAG performance with optimal test-time compute allocation. Furthermore, the authors derive an empirical scaling law that can predict the optimal inference parameter configuration for RAG systems under various computational budgets.

### Strengths
1. This paper studies a significant issue in the LLM community.
2. The authors conducts extensive experiments across various datasets and evaluation metrics, yielding convincing results.
3. This paper is well structured, clearly written, and easy to understand.

### Weaknesses
1. Novelty of the proposed methods is somewhat limited. Both DRAG and IterDRAG are conventional approaches in academia and industry, similar to some classic methods like ITER-RETGEN, IRCoT, and Active-RAG.

2. The experiments were only conducted with Gemini-1.5-Flash and Gecko retrieval. Gemini-1.5-Flash is a relatively small LLM, and different conclusions might be drawn on larger, more powerful LLMs. Moreover, if the scaling laws derived could be generalized to other LLMs and retrieval methods, it would add greater value to this work.

3. Some experimental phenomena lack more in-depth discussion. For example, on line 371, the impact of few-shot examples on IterDRAG should be assessable, at least through ablation studies to determine whether it is the in-context query decomposition or the knowledge extraction capability that is more influential.

4. While increasing the context size can improve RAG performance, it also leads to greater inference time and token consumption, especially when using iterative retrieval. The authors did not discuss this trade-off.

### Questions
See Weaknesses.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The paper explores inference scaling for retrieval augmented generation (RAG) in long-context large language models (LLMs), focusing on strategies beyond simply expanding the knowledge base. The authors investigate how in-context learning and iterative prompting can optimize the use of additional compute resources during inference. They address two primary questions: the benefit of inference computation scaling for RAG and the prediction of optimal compute allocation within a given budget. Their findings reveal that optimal allocation of inference computation results in nearly linear performance gains for RAG, a phenomenon described as inference scaling laws. The authors also develop a computation allocation model that accurately predicts the optimal inference parameters, with experimental results showing up to 58.9% performance improvements on benchmark datasets compared to standard RAG configurations.

### Strengths
The research question is quite interesting, as there is not much work on inference time scaling for RAG; this study systematically explores this area and may draw some attention.

### Weaknesses
I am concerned that this work is more suitable as a technical report rather than a research-oriented study. There is considerable related work combining long-context LLMs and RAG, and the main contribution of this work is mainly the proposed RAG inference scaling law. However, this conclusion is method-specific and may not apply to other methods.

### Questions
1. Can you provide a straightforward explanation of your findings and how they guide the use of long-context LLMs for RAG?
2. If other prompts or methods are used, is your computation allocation model still applicable?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper explores inference scaling in long-context RAG; it analyses downstream QA results in different configurations, showing that RAG performance improves ~linearly with increasing test-time compute under optimal inference parameters. Furthermore, based on these observations, authors derive a set of inference scaling laws for RAG and a computation allocation model.

My only concern is to what extent the current model generalises -- from Fig. 4b, it seems like the considered models might suffer from "lost in the middle" (https://arxiv.org/abs/2307.03172) problems, while more recent long-context models seem to suffer from this significantly less (e.g., https://arxiv.org/abs/2404.16811)

### Strengths
Inference-time scaling laws for RAG systems -- extremely interesting, and the community really needs an analysis like this one.

### Weaknesses
It is not clear whether the current analysis may generalise to future SFT/RLHF regimens.

### Questions
More recent models may suffer less from "lost in the middle" issues -- does the current analysis still hold?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper studies the inference scaling behaviors of two retrieval augmented generation (RAG) methods, demonstration-based RAG (DRAG) and iterative demonstration-based RAG (IterDRAG). The inference computation can be scaled in multiple ways, including increasing the number of retrieved documents, in-context examples, or introducing additional generation steps in IterDRAG. Experimental results show DRAG and IterDRAG achieve scaling properties with the proposed configurations, and demonstrate the performance of DRAG and IterDRAG can scale almost linearly with an increasing computation budget. Besides, the paper also learns a computational allocation model that could provides configuration guidance for DRAG and IterDRAG.

### Strengths
The paper studies two interesting research questions including the scaling behavior and the prediction of test-time computation allocation long-context RAG methods. The paper conducts systematical experiments on inference scaling of long-context RAG models, and reveals the scaling properties of DRAG and IterDRAG, i.e., the performance improves almost linearly with optimal configuration. Besides, the computational allocation model generalizes well across domains and context lengths, which potentially helps the community to better configure RAGs.

### Weaknesses
I have a question on the application of the computational allocation model. When pretraining LLMs, computational allocation models are crucial since pretraining is extremely resource-intensive. However, inference is typically much less costly by comparison. So, why not determine the best configuration by simply searching it?

### Questions
Please see the question above.

### Soundness
3

### Presentation
4

### Contribution
3
