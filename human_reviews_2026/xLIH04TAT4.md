# AOT*: Efficient Synthesis Planning via LLM-Empowered AND-OR Tree Search

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 4, 4

## Abstract
Retrosynthesis planning enables the discovery of viable synthetic routes for target molecules, playing a crucial role in domains like drug discovery and materials design.
Multi-step retrosynthetic planning remains computationally challenging due to exponential search spaces and inference costs. 
While Large Language Models (LLMs) demonstrate chemical reasoning capabilities, their application to synthesis planning faces constraints on efficiency and cost.
To address these challenges, we introduce AOT*, a framework that transforms retrosynthetic planning by integrating LLM-generated chemical synthesis pathways with systematic AND-OR tree search.
To this end, AOT* atomically maps the generated complete synthesis routes onto AND-OR tree components, with a mathematically sound design of reward assignment strategy and retrieval-based context engineering, thus enabling LLMs to efficiently navigate in the chemical space.
Experimental evaluation on multiple synthesis benchmarks demonstrates that AOT* achieves SOTA performance with significantly improved search efficiency. 
AOT* exhibits competitive solve rates using 3-5× fewer iterations than existing LLM-based approaches, with the performance advantage becoming more pronounced on complex molecular targets.
Our code is available at https://anonymous.4open.science/r/AOTstar-31FD/.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
# Overview

This paper introduces AOT*, a framework that integrates large language models (LLMs) with an AND–OR tree search for multi-step retrosynthetic planning. The reported results appear promising—AOT* achieves strong performance and efficiency improvements across several benchmarks.

However, the manuscript is difficult to follow, even for readers familiar with the retrosynthesis planning literature. The paper reads more like a results report or implementation summary than a coherent research contribution. For example, the abstract claim that AOT* “transforms retrosynthetic planning by integrating LLM-generated synthesis pathways with systematic AND–OR tree search,” but this statement is not well explained in the paper. After reading both the Introduction and Methods sections, it remains unclear what is actually new, what specific problem the method solves (now just retrosynthesis planning).

Overall, I find it difficult to fairly evaluate the contribution of AOT* due to its poor presentation.

### Strengths
* AOT* achieves top or near-top solve rates across multiple benchmarks and search budgets, outperforming both classical search (MCTS-based) and recent LLM-based baselines.
* The method shows stable efficiency and cost advantages across diverse LLMs (e.g., GPT-4o, DeepSeek-V3, Claude-Sonnet, Gemini-2.5), demonstrating robustness and practical applicability.

### Weaknesses
**1. Ambiguous statement of contributions.**
The paper’s listed “threefold contributions” in introduction are ambiguous. Only the first point, which introduces AOT* as a framework that integrates LLM-generated synthesis pathways with an AND–OR tree search, corresponds to a methodological contribution, but even this part is described at a high level without clearly defining what is technically new about the integration.
The other two points mainly emphasize performance gains and robustness across models, which are valuable but do not substitute for a clear methodological innovation. As a result, it is difficult for readers to precisely identify what aspect of the proposed framework is novel or how it differs in substance from existing LLM-guided retrosynthesis or tree-search methods.

**2. The introduction is overly abstract and lacks concrete grounding.**
The introduction presents the motivation and claimed innovation in highly abstract terms, making it difficult for readers to understand the actual problem being solved or the mechanism behind the proposed approach. For instance, the authors write that
```
However, extending these successes to practical multi-step synthesis planning remains challenging
due to the computational expense of LLM inference, limited search efficiency with constrained
iteration budgets, and the difficulty of incorporating chemical knowledge into the search process
effectively... The key innovation of AOT* lies in its systematic
integration of pathway-level LLM generation with AND-OR tree search, where complete synthesis
routes are atomically mapped to tree structures, enabling efficient exploration through intermediate
reuse and structural memory that reduces search complexity while preserving the strategic coherence
of generated pathways.
```
While this description sounds ambitious, it remains conceptually vague. It is not explained how integrating pathway-level generation with an AND–OR tree concretely alleviates the high inference cost of LLMs, improves search efficiency under budget constraints, or incorporates chemical knowledge more effectively. As a result, the introduction fails to establish a coherent causal link between the stated limitations and the proposed framework, leaving readers uncertain about the true novelty or mechanism of AOT*.

**3. Missing ablation on core methodological components.**
Although the paper includes ablation analyses on hyperparameter sensitivity and RAG sample–token trade-offs, these studies focus mainly on parameter tuning rather than validating the contribution of the proposed algorithmic components. The reader cannot tell which parts of AOT* are responsible for the reported performance gains.
There is no ablation or comparison that isolates the impact of the key design elements that are claimed to be new — for example, the pathway-to-tree mapping, the UCB-based exploration scheme, or the composite reward formulation. Without such experiments, it is difficult to assess whether these components introduce genuine novelty or if the improvements stem primarily from stronger LLMs and prompt engineering.

### Questions
1. Can you define what is the "LLM-generated chemical synthesis pathways" mentioned in the abstract?
2. What is the key challenge?
3.  Which parts of the method are new in the paper? How these parts deal with the challenge?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces AOT*, a framework designed to enhance the efficiency of multi-step retrosynthesis planning. The core contribution is the integration of Large Language Models (LLMs) with a systematic AND-OR tree search. The method utilizes an LLM to generate complete, multi-step synthesis routes for a given molecule. These entire pathways are then atomically mapped onto the AND-OR tree structure, which enables systematic exploration and the crucial reuse of common intermediates across different proposed routes. Experimental results demonstrate that AOT* achieves state-of-the-art solve rates with a 3-5x improvement in search efficiency compared to contemporary LLM-based approaches that rely on evolutionary search.

### Strengths
- The paper's main strength lies in its clever combination of the holistic planning capabilities of LLMs with the rigorous, memory-efficient search of a classical AND-OR tree. This hybrid approach effectively leverages the strengths of both components.
- The 3-5x reduction in search iterations required to find a solution is a substantial and practical contribution. The framework's inherent ability to reuse intermediates directly addresses a key source of inefficiency in unstructured search methods.
- The experiments are exceptionally thorough. The validation of the framework's effectiveness across 11 different LLMs convincingly demonstrates that the algorithmic gains are model-agnostic. The detailed ablation studies and difficulty-stratified analyses further solidify the paper's claims.

### Weaknesses
- The framework's performance may be fundamentally tied to the quality of the initial LLM-generated pathways. The paper could benefit from a more detailed analysis of the search algorithm's behavior and recovery mechanisms when the LLM produces a high rate of invalid or low-quality routes.
- The reliance on a template-based reaction validator ensures chemical correctness but also constrains the search to known chemical transformations. This limits the potential for discovering entirely novel synthetic pathways not covered by the template database.
- The "complete pathway generation" strategy is effective on the tested benchmarks, but its scalability to targets requiring exceptionally long synthetic sequences (e.g., >15-20 steps) is not fully explored. The quality of a single, coherent generation from an LLM might degrade for such long planning horizons.

### Questions
- Have you considered an alternative expansion strategy where the LLM generates shorter, fixed-length sub-pathways (e.g., 3-5 steps) instead of a "complete" route? Could this offer a better balance between LLM's strategic guidance and the search algorithm's flexibility for more complex targets?
- Could you provide statistics on the raw output from the LLM? Specifically, what is the typical success rate and step number of pathways passing the template-based validation? This would offer a clearer insight into the generation quality and the workload handled by the validation and search components.
- Regarding the Retrieval-Augmented Generation (RAG) component, the current approach retrieves the most similar synthesis examples. Have you investigated the impact of retrieving a more diverse set of examples (e.g., using different reaction types, even if Tanimoto similarity is slightly lower) to potentially expose the LLM to a broader range of strategic disconnections?
- The paper compellingly demonstrates superior iteration efficiency. However, an "iteration" in AOT* (involving a computationally expensive LLM API call) and an iteration in the baseline methods (e.g., a single forward pass of a smaller model) likely represent vastly different amounts of wall-clock time. To provide a more direct comparison of overall computational efficiency:
  - Could the authors report the average time per iteration for AOT* and the key baselines, along with the specifications of the computational platform (CPU/GPU) used?
  - Furthermore, to make a fairer comparison, have you considered running an experiment under a fixed computation budget? For instance, allowing the baseline methods to run for a significantly higher number of iterations such that their total computation cost matches that of AOT* (e.g., at N=100), and then comparing the final solve rates. This would offer a crucial perspective on which approach is more effective given the same amount of computation cost. For the computation budget, it could be measured in monetary terms, assuming all methods are run on standardized machines rented from a cloud provider (e.g., AWS, Azure, GCP), with either CPU or GPU instances as required. The total cost would be the sum of the machine rental fee (runtime multiplied by the instance price) and, for the LLM-based methods, the total cost of the API calls.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces AOT*, a novel framework designed to enhance the efficiency of multi-step retrosynthesis planning by leveraging the chemical reasoning capabilities of Large Language Models (LLMs). AOT integrates LLM-generated complete synthesis pathways with a systematic AND-OR tree search.

### Strengths
According to the authors, the advantages are: SOTA on multiple synthesis benchmarks Performance; Computational efficiency (lower complexity) comparing to existing LLM-based approaches; Capability for complex molecular targets; Can be combined to any general LLM.

The paper is well written and I really like the figures. The innovative formulation of the problem is clearly delivered: it atomically maps the LLM-generated complete synthesis routes onto AND-OR tree components.

### Weaknesses
I am so sorry, I am not an expert in this area.

### Questions
I understand that this task is intrinsically hard. May I ask if this method generates complicated chemicals that inspire research?

My understanding is that the LLM is not trained on chemical data but only on language data. What is the intuition that LLM transfers its knowledge to such a different area, without finetuning?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Authors propose a framework to use LLMS for synthesis planning. The key contribution from the paper is the mapping from a synthetic plan into a structured AND-OR representation. In this framework, LLMS are used only at the generation stage for applying single step transformations, this is then mapped into the AND-OR structured and expansions are selected using UCB criteria. 
They use several techniques like RAG to improve performance and tun thorough evaluations on multiple public benchmarks.

### Strengths
Good and thorough evaluation on multiple benchmarks showing that this method can overperform other baselines in the literature. A lot of ablations that show the individual contibution of each component.

### Weaknesses
As I understand, the method relies on the LLMs to generate the molecule smiles and thus the chemical tranformations. This makes it a template free method and brings the question of what is the chemical validity of the proposed reactions. 
The authors should discuss some specific examples and explore if the proposed reactions are chemically sensible. 
There's also severals ways of doing this in large benchmark-scale as well:
Use atom-mapping tools or reaction analysis tools to discover if the proposed reactions are valid or not, or analize what reaction types are proposed?

- The use of RAG always bring the question of data leakage. This is a sensible topic that was not discussed on the paper. Given how much a performance gain RAG brings, even with only one sample, it makes sense to further study the reasons behind such a performance boost. Two possible ideas are:
1. Randomize the reactions given to the LLM, and measure performance. By doing this, authors could prove that the data being given as part of the RAG is indeed useful, and the boost is not due to some artifact of prompting. 
2. A more controlled variation of this would be to fix the retrieved samples to the same set. This converts RAG into in-context learning and it works also as a good ablation.

-Analyze the overlap between the datasets being used for retrival, and the molecules and reactions in the benchmark. 
- Use out of distribution examples and analyze then manually, showing actual molecular depictions so that it becomes more transparent for readers if the chemistry actually makes sense, beyond simple solve rates.

### Questions
See the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2
