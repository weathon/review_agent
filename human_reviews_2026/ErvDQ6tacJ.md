# PureCover: Bridging the Gap in Re-ranking for Retrieval-Augmented Generation via Balancing Coverage and Noise

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 4

## Abstract
Re-ranking, originating from  Information Retrieval (IR), has become a critical technique for filtering retrieved documents in Retrieval-Augmented Generation (RAG). Current RAG systems often directly apply re-rankers from traditional IR, which were originally designed to provide relevant and diverse documents to human users. However, this adoption overlooks a fundamental gap: unlike humans can use selective attention to filter noise and focus on key evidence, LLMs lack this ability. This gap causes traditional re-rankers to fail in covering essential evidence and minimizing noise for LLMs, significantly hurting RAG performance, especially in complex question-answering tasks. To address this, we argue that RAG re-rankers should serve a distinct objective: not only ensuring the coverage of key information but also minimizing noise in the selected document set. To achieve this objective, we propose PureCover, a document selection framework tailored for RAG. Instead of relying on traditional Top-K re-ranking, we reformulate the document selection process as a multi-objective optimization problem and solve it by exploiting LLM attention patterns during goal-oriented reasoning. To improve efficiency, we distill the selection capability into an LLM selector via a set-wise strategy. Experiments on four multi-hop QA benchmarks demonstrate that PureCover consistently outperforms state-of-the-art baselines, achieving a better balance between coverage and noise for RAG.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
PureCover reframes RAG re-ranking as selecting a set that jointly maximizes evidence coverage while minimizing noise, rather than ranking by per-doc relevance or diversity; the system estimates per-query information needs from CoT reasoning attention, calibrates away position bias, solves a submodular 0–1 objective with a greedy procedure, then distills the selection into a lightweight LLM selector that filters by first-token logits at inference.

### Strengths
1. LLMs lack human selective attention, so RAG needs a list-level objective balancing coverage and noise rather than pure relevance. The paper formalizes this and ties it to CoT-based estimation.

2. PureCover edges strong cross-encoder and LLM rankers on all four datasets.

### Weaknesses
1. end-to-end tables fix K=5 for all datasets, but several baselines are K-sensitive; the paper only shows K-sweep on HotpotQA. Best-K comparisons and Pareto fronts (EM/F1 vs #docs) across datasets would be more convincing. 


2. Information Coverage/Purity partly rely on “contains answer string” heuristics; multi-hop bridging evidence may be under-counted. Please validate these proxies with human-labeled supports on a sample.

3. experiments use one retriever (e5-base-v2) and one generator setup; results may depend on the retrieval pool and generator size. Cross-retriever/generator tests would strengthen claims.

### Questions
see weakness

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper introduces PureCover, a novel document re-ranking framework for Retrieval-Augmented Generation (RAG) that reframes document selection as a multi-objective optimization problem to balance information coverage and noise minimization. Unlike traditional re-rankers designed for human users, PureCover leverages LLMs’ internal attention patterns during goal-oriented reasoning to identify key evidence and employs a greedy algorithm with set-wise distillation for efficient inference. Experiments on multiple multi-hop QA benchmarks show that PureCover consistently outperforms state-of-the-art baselines by better balancing coverage and noise, leading to significant improvements in RAG performance.

### Strengths
1. Clear Problem Formulation: It identifies a specific, often-overlooked flaw in applying traditional re-rankers to RAG systems.

2. Strong Empirical Results: It demonstrates consistent and significant performance improvements over many state-of-the-art baselines.

### Weaknesses
Limited Justification for Core Assumptions: The entire method hinges on using LLM attention during Chain-of-Thought (CoT) reasoning as a proxy for "information requirements" and document utility. While the results are positive, the paper provides no validation that these attention patterns are a reliable and generalizable ground truth for what constitutes essential evidence versus noise. This is a significant assumption. A stronger justification, such as a human annotation study correlating high-attention tokens with key facts, or an ablation showing that the content of the CoT steps (rather than just the attention signal) is crucial, would make the foundational premise more convincing.

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
3

### Summary
The paper argues IR re-rankers assume human selective attention while LLMs don’t, so RAG needs re-ranking that balances evidence coverage with noise suppression. 
It defines coverage and noise for complex QA, then reframes document selection as a multi-objective problem solved with calibrated reasoning-attention signals.

### Strengths
The goal is RAG-native and practical: cover what’s needed and keep junk out of the context.

The position-bias calibration acknowledges “lost-in-the-middle” issues and corrects attention before using it.

The set-wise distillation and first-token scoring make deployment feel realistic instead of research-only.

### Weaknesses
The method leans hard on CoT prompting and attention introspection, which can drift across models and prompts. 

The noise term uses a max over requirements, which might miss partial usefulness or cross-requirement interactions. 

Submodularity and monotonicity hinge on attention-based estimates that could be noisy in practice. 

The calibration relies on dummy CoTs/docs and may not transfer cleanly to other layout or packing strategies. 

The distilled selector introduces thresholds and budgets that still need tuning in new domains.

### Questions
How stable are P(e|q) and P(d|e) across different CoT prompts, temperatures, and model sizes.

What’s a practical recipe to set lambda and the selection threshold without dataset-specific sweeps.

Can the framework capture “two-docs-together” interactions rather than “at least one doc” coverage only.

### Soundness
3

### Presentation
3

### Contribution
2
