# Structured Reasoning for LLMs: A Unified Framework for Efficiency and Explainability

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 6

## Abstract
Recent Large Language Models (LLMs) have made remarkable progress, but they still struggle with complex reasoning tasks such as logical deduction and planning. This is partly because they rely primarily on token-level probability relationships, which limits their ability to reason effectively. 
In this paper, inspired by cognitive science and neurosymbolic AI, we introduce **Structured Reasoning**, which aimes at enhancing the reasoning capabilities of LLMs from the step level. 
To this end, we first collect high‑frequency, domain‑agnostic reasoning step tags and  construct a structured reasoning dataset with those tags. 
Then, we treat a reasoning process as a *directed acyclic graph*, where the vertices represent steps and the edges indicate the direction of reasoning. 
In this context, an efficient reasoning process corresponds to, or can be characterized by, a sparse reasoning graph.  
To construct reasoning graphs, we introduce *structured tags* for reliable step extraction from LLM outputs. For single-graph optimization, we propose the *MaxFlow reward*, which rewards graphs with balanced node contributions and fewer redundant steps. The quality of a sparse reasoning graph can be reflected by the total flow from all steps to the final answer. For multi-graph comparison, we propose the *LCS reward*, which selects reliable reasoning paths by identifying optimal common subsequences (consecutive steps) shared across multiple generated responses (sequences). 
Experiments with DeepSeek-R1-Distill-Qwen-1.5B and 7B models show that our method  consistently outperforms GRPO and other carefully tuned baselines across various context lengths (0.5k–8k).
Structured Reasoning shows particular strength in efficiency (better performance with fewer steps) and stability (consistently generating high-quality outputs across a temperature range of 0.1 to 1.0).  
Methods and examples is currently available on our website: https://cnsdqd-dyb.github.io/structured-reasoning.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper presents a new reasoning approach, aimed at structuring the reasoning traces of LLMs.  They convert unstructured reasoning traces by prompting a teacher LLM to output a linear sequence of abstract steps (labels). After some filtering, the teacher is prompted to generate reasoning for i) a given q and ii) a set of sampled labels from the label collection. 

The resulting collection is the dataset used for training the model. The latter predicts answer / reasoning from question and a "structured reasoning prompt". A graph across reasoning steps is built, where they compute the "step-to-step" attention scores to measure the  strength of an edge, 

They leverage the max flow of a given node / step to assess its performance (the bigger it is, the more important to reaching the final solution). The flow analogy is used to measure the "robustness" of the solution; a higher score if the spread of importance is high, i.e. no single or few crucial nodes, but rather many pathways to the solution. 

The authors evaluate on math and reasoning benchmarks. The authors show gains overall, especially when capping the response length. The authors also show that performance of the overall system is more stable to temperature variations than the baselines. Finally, the authors also demonstrate that their approach for measuring the importance of a reasoning step is the better approach when filtering out reasoning steps w.r.t. performance

### Strengths
1. Overall an interesting approach for structuring reasoning traces. 
2. The authors' experiment protocol is quite rigorous, from the base set of results to the explainability tasks. 
3. Overall good performance on the benchmarks.

### Weaknesses
0. The paper is very dense, from the abstract to figure 2, and even section 3.1. The overall flow of the paper could be improved, where more explanations / motivations for the equations would be appreciated, and fewer results presented. For example, what is F in equation 1 ? 

1. The approach is quite cumbersome; from structuring / labelling reasoning traces to the 3 pronged loss function. 

2. The paper is missing several ablations to correctly access the importance of the proposed components. Likewise, simply training on the filtered data without any additional loss terms (i.e. standard GRPO) would be very informative. 

3. A comparison to Dr. GRPO [1] would be a nice addition to the paper, as they propose a simple, proper fix to the length issue in GRPO. 


[1] Understanding R1-Zero-Like Training: A Critical Perspective, Liu et al., COLM 2025.

### Questions
1. Is a step reward given specifically to the step ? or do all step receive the same sequence level reward ?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors propose an approach for reasoning using large language models (LLMs), introducing a framework called structured reasoning to improve efficiency through the pre-computation of the extracted common reasoning patterns. They apply this method to enhance the GRPO. The authors conduct experimental study to verify the proposed solution for insights.

### Strengths
This paper presents a Structured Reasoning approach that improves the efficiency, stability, and interpretability of reasoning, experimentally verified using models such as DeepSeek-R1-Distill-Qwen. The authors demonstrate that their method enhances performance even at shorter input lengths and maintains consistent quality across varied temperature settings. 

A key contribution from the work is the automatic extraction of common reasoning patterns, allowing unstructured reasoning to be restructured into coherent chains via a specially designed dataset. Furthermore, the introduction of an attention-based layer-wise analysis provides detailed step-to-step attention maps, underscoring the importance of middle layers in effectively integrating reasoning context. 

The authors implement two algorithms, MAX-Flow and LCS. The former creates sparse reasoning graphs to evaluate the impact of each reasoning step, and the latter identifies optimal subsequences from multiple outputs to improve reasoning quality. 

Overall, the proposed methods offer promising advancements in structured reasoning frameworks.

### Weaknesses
The proposed three-stage pipeline for enhancing LLMs with Structured Reasoning presents an interesting and potentially valuable design that integrates multiple design choices and algorithmic components. However, this layered structure makes it challenging to clearly identify and understand the contribution and impact of each individual stage. The paper does not sufficiently analyze or disentangle these effects. Furthermore, the performance and behavior of each stage appear to depend on a number of implementation details that are not fully specified, which makes the overall approach somewhat difficult to evaluate and replicate. For example, the granularity of the labels to consider in the first step, and the edge weight definition in Step 3. 

Here are a few more detailed feedback:

In Equation 1, the notation F() in the condition is not explained in the text, unless I’ve overlooked something. It seems to be a filter, but it needs to be clearly defined.

It’s great to have the complexity estimate in Section 3.3. The authors should provide some insights into whether the scale of the complexity is reasonable or not

In Equation 5, the score design heavily relies on the correct completion, which is somewhat vague in the description. As a key concept or operation, the authors should elaborate on it.

In Table 1, the performance of LCS appears stronger than MaxFlow in most cases. If this aligns with expectations, the authors should explain the rationale.

### Questions
see the questions in the weaknesses section.

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
This work targets the limitations of LLM on complex reasoning tasks such as logical deduction and planning, with a focus on efficiency and explaninability.
It aims to enhance the reasoning capabilities of LLM at the step-level by introducing the structured reasoning approach, viewing the reasoning process as a directed graph.
The proposed framework first extracts the unstructured data into a structured format by incorporating the explicit reasoning steps, which enable adaptive fine-tuning of LLM afterward.
The authors also propose a layer-wise dependency tracing procedure using step-to-step attention matrices for in-depth analysis.
The fine-tuning part, they also extend GRPO with two structure-aware algorithms- Max-Flow and Longest Common Subsequence.

The empirical experiments showcase the improved efficacy, efficiency as well as the stability.

### Strengths
1. This paper is well-motivated, and the proposed techniques are rational.
2. The paper is well-written and easy to follow.
3. The experiments are comprehensive, covering efficacy, efficiency, and stability.

### Weaknesses
1. As extensions of GRPO, the two proposed variants do not always demonstrate clear advantages over GRPO. For example, in some cases, MaxFlow outperforms GRPO while LCS achieves similar performance; in other cases, LCS performs better, whereas MaxFlow's results are comparable to GRPO.
However, there is no discussion explaining these observations or directly comparing LCS and MaxFlow.

2. The stability and efficiency study only considers the MaxFlow variant while the LCS variant is ignored.

### Questions
1. Can smaller LLMs also benefit from the training-free structured reasoning?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper studies structured reasoning for LLMs by (1) explicitly labeling reasoning steps with semantic tags, (2) extracting step-wise dependencies from attention weights, and (3) introducing two reinforcement learning rewards: a Max-Flow reasoning reward that encourages balanced step utility and a longest-common-subsequence (LCS)-based consistency reward. The method aims to inject structure into chain-of-thought processes and improve reasoning efficiency. Experiments on math and reasoning benchmarks show improvements over supervised and GRPO baselines, alongside qualitative analyses of attention structure.

### Strengths
1. The paper addresses an important problem: encouraging more structured and efficient reasoning in LLMs rather than unconstrained chain-of-thought generation.
2. The proposed Max-Flow reward is a creative attempt to use internal model signals (attention structures) to shape reasoning behavior, moving beyond pure outcome-based rewards.
3. Experiments are conducted on multiple reasoning benchmarks and across different model sizes, showing consistent improvements under constrained output budgets.

### Weaknesses
1. The approach heavily relies on attention maps reflecting causal reasoning order. This link is assumed rather than validated. No experiments demonstrate whether attention graphs truly correspond to semantic dependency structures, raising validity concerns.
2. Key components (structured tags, Max-Flow reward, LCS reward) are not fully isolated. It remains unclear which part contributes most. A more refined ablation (e.g., Max-Flow only vs LCS only vs tags only) would strengthen the causal claims.

### Questions
1. Table 1 max-length execution. How is the “max length” constraint applied in Table 1? hard decoding cap? truncation after generation?
2. Does attention truly correspond to reasoning structure?
3. The Max-Flow reward implicitly assumes that “better reasoning” corresponds to more balanced (not overly concentrated) attention flow across intermediate steps. Can the authors clarify the motivation for this assumption?

### Soundness
3

### Presentation
3

### Contribution
3
