# Causal Cartographer: From Mapping to Reasoning Over Counterfactual Worlds

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Causal world models are systems that can answer counterfactual questions about an environment of interest, i.e., predict how it would have evolved if an arbitrary subset of events had been realized differently. The ability to answer such questions is crucial for models to reliably understand the world. However, this task currently eludes large language models (LLMs), which do not have demonstrated causal reasoning capabilities beyond the memorization of existing causal relationships. Furthermore, evaluating counterfactuals in real-world applications is challenging since only the factual world is observed, limiting evaluation to synthetic datasets. We address these problems by proposing the Causal Cartographer, a twofold system composed of two agents: the first extracts causal relationships from data and builds a vast repository of causal knowledge, while the second uses them as constraints to perform reliable step-by-step causal inference. We evaluate our approach on real-world counterfactuals obtained by matching data from diverse news sources. We show that our approach can extract accurate causal knowledge and enhance the robustness of LLMs for causal reasoning tasks. In particular, the proposed causal conditioning mitigates the impact of spurious correlations and greatly reduces inference costs (by up to 70\%) compared to chain-of-thought reasoning.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
It is difficult to both perform causal reasoning with large language models and evaluate them, due to ladder of causality wich states that interventional and counterfactual quantities can generally not be inferred from observed data unless strong assumptions. The submission proposes to extract known causal relationships from real-world articles, yielding a causal world model, then use that model to perform causal reasoning with LLMs as well as evaluating them.

### Strengths
- Original approach, which I had not seen before while seeming natural in the context of LLMs.

- The paper is generally clear and well-written.

- Experiments support the method outperforming past alternatives in terms of performance and efficiency.

### Weaknesses
- The definition of SCMs used by authors ignores noise in individual structural equations. Notably, Definition 1 assumes deterministic relationships between the causal blanket and the target variable, while in general, noise variables can be present. This makes it unclear to assess whether Definition 1 and Theorem 1 is only possible in the absence of noise, which is a generally restrictive scenario.

- "We also excluded outliers (∼4% of the answers were nonsensical numbers)." (l.413-415) : this seems a bit quick to me... It would helpful to know the fraction of outliers for each evaluated model, how they change results, and how to evaluate performance in a way that is robust to them if they dominate averages.

### Questions
- How do Definition 1 and Theorem 1 generalize in the presence of noise variables?

- What if you include outliers, and check the things indicated above?

### Soundness
2

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
The authors present Causal Cartographer which extracts causal relationships from data and then uses them as constraints to perform step-by-step causal inference. The performance is evaluated on real-world counterfactuals obtained from news sources.

### Strengths
1. Understanding LLM performance on counterfactual reasoning tasks is crucial is furthering reasearch on LLMs ability to do causal tasks
2. Using real-world data instead of synthetic is encouraging
3. The proposed method is more interpretable which is good for future research

### Weaknesses
1. It is not clear how this method can scale to production LLM systems.
2. Causal graph building would require a very careful control so as to not introduce bias

### Questions
1. Does this framework have the risk of running "stale". In a constantly evolving world, what if the causal relationships from the first stage change? How would one go about keeping them up to date? Would this update process eat into the inference cost savings?
2. How does the system defend itself against adversarial attacks where noisy/false claims are injected into the causal knowledge repository?
3. Sorry if I missed this, but how is it ensured that the extracted relationships are causal and not noise?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes Causal Cartographer, a two-agent framework for causal reasoning over natural-language sources. The framework consists of (1) CTG-Extract, which performs graph-RAG–assisted causal extraction from news articles to build a large causal knowledge base (CausalWorld), and (2) CTG-Reason, which performs step-by-step, causally constrained inference (including counterfactuals) by conditioning only on parents/children along the graph. The authors also introduce “causal blankets” which, along with a K-matching procedure, enable approximating real-world counterfactuals by matching “worlds” across documents. Empirically, on a 400-query dataset (CausalWorld-CR) derived from news in 2020, CTG-Reason attains accuracy on par with, or better than, a CoT baseline while reducing context and output tokens (up to 70% fewer), with especially large efficiency gains on small models.

### Strengths
- The paper argues well for why explicit causal constraints can mitigate spurious correlations and reduce inference cost.
- The two-agent split via decomposition of the task as extraction and reasoning enables each agent to focus specifically on its own task.
- The causal-blanket definition and K-Matching Equivalence theorem formalize when matched worlds yield valid counterfactual targets—useful for this emerging evaluation paradigm.
- The reported token/input reductions and output length shrinkage are substantial while maintaining accuracy.

### Weaknesses
- The text corpus utilized is 2020 news with focus on economics. What factors led to this choice? How well does this approach perform in other domains?
- The method leans on SCM framing (DAGs), yet the constructed CausalWorld allows cycles/feedback loops (Fig. 6).
- Causal blankets are defined as fully determining the target (deterministic f). Real news variables are often noisy. Can the theorem and agent be generalized to stochastic blankets?

### Questions
1) A small set of bridge nodes routes information across communities. Did you measure how removing a top-k bridge node affects the fraction of nodes still usable for counterfactuals and the success rate of K-matching?

2) You remove queries with ≥50 causal paths and rebalance degree skew. How sensitive are results to the “≥50” threshold, and what happens if you keep hard queries?

3) What max recursion depth or search budget do you set for anticausal inference when parents/children are missing, and how often do queries exceed it?

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
This paper, “Causal Cartographer: From Mapping to Reasoning over Counterfactual Worlds,” proposes a novel two-agent framework for enabling large language models (LLMs) to extract, organize, and reason with causal knowledge from real-world text. The system, called Causal Cartographer, consists of (1) CTG-Extract, a causal extraction agent based on Graph Retrieval-Augmented Generation (Graph-RAG) that builds a large-scale causal graph (“CausalWorld”) from unstructured text (e.g., 500 economic news articles), and (2) CTG-Reason, a counterfactual reasoning agent that performs stepwise inference under causal constraints. The authors introduce theoretical contributions, notably the concept of causal blankets (a generalization of Markov blankets) and a K-matching algorithm for identifying counterfactual pairs of worlds in text data. Experiments compare CTG-Reason with the chain-of-thought-based CausalCoT method on a new dataset, CausalWorld-CR, derived from real-world causal extractions. Results show comparable or better accuracy and reduced computational cost (up to 70% reduction in inference cost), especially for smaller models like o3-mini and LLaMA-3.1-8B.

### Strengths
The paper addresses an important gap between abstract causal reasoning and real-world data extraction. Its proposed combination of causal extraction and counterfactual reasoning within an LLM framework is both ambitious and well-motivated. The introduction of CausalWorld, a large-scale, structured repository of 975 nodes and 1337 causal relations, is an impressive resource that could stimulate further research. The integration of Graph-RAG retrieval ensures grounding in prior causal context during extraction, improving coherence and scalability.

### Weaknesses
Despite its strengths, the paper has several limitations that hinder its maturity for a top-tier conference. The evaluation is limited in scope and realism: the CausalWorld-CR dataset is constructed via synthetic matching across news articles rather than ground-truth counterfactual data. This raises concerns about the validity of “real-world” claims and the soundness of the evaluation metric.

### Questions
The concept of causal blankets (Section 5.1) should be more carefully distinguished from Pearl’s Markov blankets beyond lineage claims.

### Soundness
2

### Presentation
2

### Contribution
2
