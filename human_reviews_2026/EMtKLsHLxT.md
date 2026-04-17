# LLM Program Optimization via Retrieval Augmented Search

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 2, 4

## Abstract
With the advent of large language models (LLMs), there has been great interest in applying them to solve difficult programming tasks. Recent work has demonstrated their potential at program optimization, a key challenge in programming languages research.  We propose a blackbox adaptation method called Retrieval Augmented Search (RAS) that performs beam search over candidate optimizations; at each step, it retrieves in-context examples from a given training dataset of slow-fast program pairs to guide the LLM. Critically, we find that performing contextual retrieval based on an LLM-generated natural language description significantly outperforms retrieval based on the source code. In addition, we propose a method called AEGIS for improving interpretability by decomposing training examples into ''atomic edits'' that are significantly more incremental in nature. We show that RAS performs up to 2.04$\times$ better than prior state-of-the-art blackbox adaptation strategies on optimizing C++ programs, and that AEGIS performs 1.37$\times$ better while performing significantly smaller edits. We also show that using RAS improves the mean runtime percentile of Python programs by 10.27 as compared to other strategies.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes Retrieval-Augmented Search (RAS) and an atomic-edit pipeline (AEGIS) for black-box program optimization with LLMs. RAS iteratively retrieves examples via natural-language program descriptions and performs beam-style search; AEGIS decomposes slow-fast pairs into atomic edits to improve interpretability. Reported results on PIE and Mercury show that the proposed RAS method can improve the efficiency of the generated code.

### Strengths
+ Clear problem framing for black-box adaptation to performance optimization. 
+ The proposed method is rational, combining contextual retrieval and iterative search to improve the efficiency of the generated code. 
+ Experiments show that AEGIS improves edit granularity and interpretability.

### Weaknesses
- Comparisons center on closely related prompt-engineering variants (dynamic retrieval, no-contextual retrieval, instruct-only). Missing are stronger and more diverse baselines: e.g., white-box adaptation (e.g., fine-tuning), stronger compiler optimization pipelines, or more recent LLM-based code optimization methods. As a result, the superiority of RAS may mostly reflect an advantage within a narrow family of RAG methods rather than against the broader state of the art.
- The paper does not report wall-clock optimization running time in any of the baseline or proposed method.
- The paper lacks a discussion of correctness after optimization. The paper lacks analysis of the correctness after LLM-based optimization is applied. The reviewer is not sure that if the proposed RAS generates faster but incorrect code solutions?

### Questions
- What is the post-optimization correctness rate? Do you observe bugs introduced by optimization, and with what frequency and types?

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
3

### Summary
This paper presents two black-box adaptation methods for Large Language Models (LLMs) to optimize programs: Retrieval Augmented Search (RAS) and Atomic Edit Guided Search (AEGIS). RAS iteratively optimizes the program through a retrieve-optimize-evaluate loop. At each step, it retrieves the "slow-fast" program pairs from a training set to guide LLM generation using the similarity of the natural language summarization instead of using code retrieval directly. AEGIS aims to improve the interpretability of program optimization by using a sequence of "atomic edits. On PIE and Mercury, RAS significantly outperforms the baselines, and AEGIS can reduce the average edit distance.

### Strengths
1. The proposed RAS is simple yet effective.

2. The proposed RAS shows significant improvements on the benchmark, even surpassing human baselines.

### Weaknesses
1. Although the proposed method RAS improves the optimization performance, it requires too many LLM calls ($m \times k$), which is costly in practice.

2. The authors prove that contextual retrieval can outperform code retrieval. However, it seems counterintuitive because the raw code should naturally reflect more details than code descriptions. I wonder if it is just because the embedding model used in the paper cannot handle the code input well.

3. RAS in fact is different from beam search as it does not keep top-k but only the best candidates at each step.

4. The value of AEGIS is uncertain. First, it does not reach the same level of optimization as RAS, although it involves more LLM calls to process the dataset. Second, although the paper claims that it can improve the interpretability of the generated samples by showing the decrease of the edit distance, this metric is not straightforward to represent the interpretability as sometimes the optimization requires changing high-level algorithms and data structures (also mentioned in Line 82)

5. The descriptions of Algorithm 1 and Algorithm 2 need to be improved. Some notions are not used in the workflow (e.g. $F_{context}$)

### Questions
1. Could you provide more insights on why contextual retrieval can outperform code retrieval?

2. In AGEIS, why do we need to generalize from $s_i$ to $e_i$ (Line 297)? How does it generalize exactly?

### Soundness
2

### Presentation
2

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
This paper proposes Retrieval Augmented Search (RAS) and Atomic Edit Guided Search (AEGIS) for program optimization using large language models. The key idea is to perform iterative beam search where, at each step, training examples are retrieved based on LLM-generated natural language descriptions of the current program rather than direct code similarity. AEGIS additionally decomposes training pairs into sequences of "atomic edits" to enable more incremental modifications. Experiments on the PIE (C++) and Mercury (Python) benchmarks show substantial improvements over prior methods.

### Strengths
- The core idea of combining iterative search with contextual retrieval is intuitive and well-motivated.
- The paper demonstrates substantial improvements over baseline methods, achieving 2× better speedup compared to dynamic retrieval on the PIE benchmark, with consistent gains on the Mercury benchmark as well.
- AEGIS's attempt to improve interpretability through atomic edits is an interesting method, even though the performance trade-off is not ideal.

### Weaknesses
**1. Writing and presentation issues**

The paper's organization and clarity need improvement. For example:

- In the "Problem formulation" section (line 152), “Problem formulation. In the program optimization problem, the goal is to take a program $p \in P$ as input, and output an optimized program $p′ \in  P$ that is semantically equivalent to p.” spends an entire paragraph discussing semantic equivalence rather than defining the "optimization" (runtime? memory? how is it measured?).
- Line 86 mentions "retrieval and search are not integrated" in SBLLM, which is confusing since retrieval and search are conceptually similar. The same issue applies to the proposed "retrieval augmented search"—this needs clearer explanation in the introduction.
- There are many vague references throughout: "This existing approach" (line 42), "They find..." (line 33), "they propose..." (line 33) without clear antecedents or citations.

**2. Limited baseline comparisons**

The paper only compares against one established baseline (SBLLM). The related work section is brief and lacks a systematic overview of recent progress in LLM-based code optimization. If this domain hasn't seen much recent work, it would be valuable to discuss why—what are the key challenges? Additionally, the paper could strengthen its evaluation by including generic but effective methods like CoT or ReAct, or some generic agent workflow as baselines.

The choice of PIE benchmark is not well justified. I’ve seen some other code optimization benchmarks in recent year [1,2]. A discussion of benchmark selection would be helpful.

[1] SWE-Perf: Can Language Models Optimize Code Performance on Real-World Repositories?

[2] CodeScope: An Execution-based Multilingual Multitask Multidimensional Benchmark for Evaluating LLMs on Code Understanding and Generation

**3. Unfair computational comparison**

The comparison appears unfair: RAS uses k=8 retrievals times m=4 beam search steps (32 total retrievals), while the baseline uses k=4 retrievals × 1 step (4 total). To ensure fair comparison, the total retrieval volume across all methods should be the same. 

Additionally, the paper claims to "normalize computation" by counting LLM calls, but this ignores the cost of generating natural language descriptions for train set, and the preprocessing cost for AEGIS. How the LLM calls are controlled is not explicitly or formally clarified either. From my understanding, sampling h candidates from baseline method only need 1 LLM call. While the multistep RAS need m LLM call per sample.

**4. Unclear source of improvement for RAS**

While RAS shows improvements over the baseline, it's unclear whether the gains come from the method itself or simply from using more computation (4 steps). The baseline does one-pass generation with only more sampled candidates, while RAS does multi-step agent-like retrieval and generation. A fairer comparison would control for total computational budget (such as token) or provide ablations showing diminishing returns.

**5. Limited novelty of contextual retrieval**

The core contribution of RAS—retrieving based on natural language descriptions rather than direct code embeddings—has been widely and extensively studied and applied in prior work. The paper should better articulate what is novel beyond applying an existing technique to this setting.

**6. AEGIS shows degraded performance**

Table 2 shows AEGIS (6.08× speedup) actually performs worse than RAS alone (8.01× in Table 1), despite being the paper's second major contribution. The smaller edit distance is interesting but doesn't offset the performance loss. Why should we prefer AEGIS over RAS?

### Questions
1. Can you provide more details on how computational costs are controlled across baselines?

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
This paper addresses the limitation that large language models (LLMs) struggle with "out-of-the-box" program optimization, which proposes two novel retrieval-based adaptation methods: Retrieval Augmented Search (RAS) and Atomic Edit Guided Search (AEGIS).  RAS introduces a beam search mechanism combined with a crucial insight: performing contextual retrieval based on LLM-generated natural language descriptions of programs significantly outperforms code-based retrieval. AEGIS extends this by decomposing training examples into "atomic edits," enabling more incremental and interpretable optimizations. This work makes some contributions by advancing the efficacy and interpretability of LLM-driven program optimization through intelligent retrieval and structured search, offering a compelling framework for future research in code generation tasks.

### Strengths
- **Clear Baselines and Ablations**: Comparisons against state-of-the-art dynamic retrieval, "Instruct Only," and "No Contextual" ablations (using source code retrieval instead of contextual retrieval) effectively isolate the impact of core innovations. For example, RAS achieves an 8.61× speedup on PIE—2.04× better than dynamic retrieval—while the "No Contextual" variant's 3.63× speedup confirms that contextual retrieval is essential.
- **Cross-Language Generalizability**: Evaluations on PIE (C++ optimization with deterministic gem5 simulation) and Mercury (Python optimization with runtime percentiles) demonstrate strong cross-language generalizability, addressing a key limitation of prior work—for example, SBLLM only reported C++ results with 1.55× speedup.

### Weaknesses
- **Effectiveness of Atomic Operation Decomposition in AEGIS:** The paper proposes AEGIS, which decomposes atomic operations in the training dataset. However, experimental results show that AEGIS underperforms RAS. In the "No Contextual" setting, AEGIS's optimization effect falls below Dynamic Retrieval. This raises questions about whether atomic operation decomposition effectively improves optimization performance. While atomic operations enhance interpretability, the paper lacks statistical data from developers on the practical interpretability gains AEGIS provides. Without user-centered evaluation, it remains unclear whether the increased interpretability truly benefits real-world programming scenarios.
- **Inadequate Exploration of Model Generalizability:** The experiments primarily use GPT-4o and Qwen3-Coder on function-level optimization benchmarks. Given the strong capabilities of models like Claude-4, it might achieve high-level program optimization on simple functions without relying on extensive external knowledge. This absence limits the generalizability analysis of the proposed methods across different LLM architectures.
- **Limited Practical Application Potential:** The benchmarks (PIE and Mercury) focus on competitive programming problems with enumerable optimization patterns (e.g., loop-to-DP conversions), making retrieval-based optimization straightforward. However, repository-level tasks (e.g., swe-pref) require understanding broader code context and architectural constraints—challenges not addressed here. It remains unclear whether RAS and AEGIS generalize to large-scale, real-world software optimization.
- **Lack of Exploration on Long-Thinking Models:** For models with strong reasoning capabilities, such as OpenAI o3 or DeepSeek-R1, which outperform at reasoning tasks. Given that program optimization is a complex reasoning-intensive task, these long-thinking models might offer unique advantages. This gap restricts the comprehensiveness of the study in leveraging the full potential of different LLM types for program optimization.

### Questions
1. AEGIS performs worse than RAS, and without contextual retrieval it even falls behind Dynamic Retrieval. Does atomic decomposition itself not improve optimization, or are there other issues (like limited atomic edit diversity)? Also, have you tested with actual developers to see if AEGIS truly improves interpretability, rather than just measuring edit distance?
2. Have the authors tried Claude-4, which is strong at code tasks? If not, why? 
3. On advanced reasoning models: Have you tested models with strong reasoning capabilities like OpenAI's o3 or DeepSeek-R1? Do you think their reasoning abilities would further improve performance on Instruct-Only?
4. For repository-level optimization tasks such as SWE-perf, how do you envision obtaining the training corpus (slow-fast pairs) needed for your retrieval-based methods?

### Soundness
2

### Presentation
3

### Contribution
2
