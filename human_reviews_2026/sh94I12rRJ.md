# Searching Meta Reasoning Skeleton to Guide LLM Reasoning

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 4, 6, 4

## Abstract
Meta reasoning behaviors work as a skeleton to guide large language model (LLM) reasoning, thus help to improve reasoning performance.
However, prior researches implement meta reasoning skeleton with manually designed structure, limiting ability to adapt to query-specific requirement and capture intricate logical dependency among reasoning steps.
To deal with the challenges, we represent meta reasoning skeleton with directed acyclic graph (DAG) to unify skeletons proposed in prior works and model intricate logical dependency.
Then we propose AutoMR, a framework that searches for query-aware meta reasoning skeleton automatically inspired by automated machine learning (AutoML).
Specifically, we construct search space based on DAG representation of skeleton and then formulate the search problem.
We design a dynamic skeleton sampling algorithm by expanding meta reasoning skeleton along with reasoning context at inference time.
This algorithm can derive any meta reasoning skeleton in search space efficiently and adapt skeleton to evolving base reasoning context, thus enable efficient query-aware skeleton search.
We conduct experiments on extensive benchmark datasets. Experimental results show that AutoMR achieves better reasoning performance than previous works broadly.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
AutoMR reframes “meta reasoning” for LLMs as searching a query-aware **reasoning skeleton** represented by a single-source, edge-heterogeneous DAG that can encode sequential, parallel, tree, and more intricate dependencies among steps. It defines a strategy set (e.g., Next, Reflect, Explore, Decompose, Summarize, Recall, Answer) and introduces a **dynamic skeleton sampling** algorithm that interleaves with inference: for each step it selects incoming strategy-typed edges conditioned on the evolving reasoning context, generating content only when needed. A lightweight MLP guides edge selection, and a policy over skeletons is optimized with REINFORCE; this adds minimal overhead compared to vanilla reasoning while adapting structure per query. Across math QA (GSM8K, MATH-500, AMC, Olympiad) and general multiple-choice (MMLU-Pro) with LLaMA-3B and Qwen-3B backbones under the same token budgets, AutoMR consistently outperforms CoT, MRP, Meta-Reasoner, rStar, and an agent-workflow NAS baseline, and scales compute more efficiently, highlighting the advantage of DAG-based, instance-specific meta-reasoning.

### Strengths
1. The authors’ motivation is well founded, where for different problems, the meta-reasoning framework should be dynamic rather than static.
2. The proposed method is highly effective, achieving substantial performance gains across settings compared with prior approaches.
3. While improving accuracy, the method also demonstrates better efficiency than previous work.

### Weaknesses
1. Experiments are limited to short-CoT and small-scale models, which constrains the validation of the approach. Evaluating on long-CoT models (e.g., DeepSeek-R1-Distilled, Qwen3) and larger models (e.g., 8B, 14B) would more convincingly substantiate the method’s effectiveness.
2. Although many prior works also rely on training for meta reasoning, in many practical scenarios it is difficult to obtain sufficient training data and compute. Even when such resources are available, one could instead train a LoRA to boost performance, which makes the niche for this work somewhat awkward.
3. The method bears similarities to Graph-of-Thought [1]. Although the granularity differs, both enhance performance by structuring the reasoning process as a graph.
4. The writing is somewhat disorganized: for example, Figure 1 is not placed at the top, and Figure 5 appears to be a Table.

[1] Graph of Thoughts: Solving Elaborate Problems with Large Language Models. Besta et al., AAAI 2024.

### Questions
1. How did you determine the set of strategies used in Table 2? Why not include more or fewer strategies?
2. What are the MLP hyperparameters in your experiments?
3. Why does Figure 4 report results only on MATH-500 rather than an average across all datasets?
4. It is surprising that on MMLU-Pro such significant gains are achieved with only 70 training examples. Did you attempt to train the MLP on MATH-500 with fewer than 100 training samples as well?

### Soundness
3

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
3

### Summary
This paper proposes AutoMR, a framework designed to search for meta-reasoning skeletons more effectively and efficiently. Experiments conducted across a wide range of tasks demonstrate the framework’s strong performance and provide a comprehensive analysis of its effectiveness.

### Strengths
- The presentation and writing are clear and generally easy to follow.

- The core idea and motivation are interesting and relevant.

- The proposed method appears highly effective.

- The experiments and analyses are comprehensive and well-executed.

### Weaknesses
- Design Motivation: The paper would benefit from a deeper explanation of the design motivation behind AutoMR. Many parts of the explanation are highly technical and provide limited intuition for understanding why the framework works. While the inclusion of formulas is appreciated, the methodology—especially Section 3.2.1—would be easier to follow if accompanied by more intuitive explanations or illustrative examples.

- Baselines: The paper should include comparisons with more recent and relevant reasoning models to strengthen its empirical claims.

- Analysis Depth: Additional analysis could enhance the interpretability and transparency of the framework, helping readers gain a deeper understanding of its internal mechanisms and behavior. (e.g., error analysis to understand the limitations)

I would be willing to raise my score if the authors can adequately address these weaknesses.

### Questions
The meta-reasoning behaviors summarized in the paper are derived from existing studies on LLM reasoning. Are these behaviors comprehensive enough? Have the authors considered incorporating other types of reasoning behaviors, perhaps inspired by cognitive or psychological perspectives?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper purposes AutoMR, a meta-reasoning framework that treats the reasoning skeleton as a single-source, edge-heterogeneous DAG. The search space uses a strategy set plus a zero edge, enabling rich inter-step dependencies that prior fixed skeletons miss. Trained via policy-gradient over the sampler, AutoMR consistently outperforms CoT, Meta-Reasoner, rStar, etc. across math QA and MMLU-Pro subsets and exhibits superior token-scaling efficiency.

### Strengths
1. The paper reframes meta-reasoning as a creative, NAS-style control layer interleaved with the LLM’s ongoing reasoning rather than fixed upfront.
2. The unified DAG view plus inference-time search offers a general recipe for meta-control that can envelop many existing meta-reasoning templates.
3. The paper cleanly motivates the limitations of fixed skeletons, then walks through the search space and dynamic sampler with a concrete algorithmic presentation, which makes it easy to follow.

### Weaknesses
Results are on 3B instruction models (Qwen2.5-3B-Inst, LLaMA-3.2-3B-Inst) with a 1024-token budget for all methods. It remains unclear if AutoMR’s gains persist with stronger models or longer budgets typical in practice.

### Questions
1. When a node has multiple incoming edges with possibly different strategies, it is unclear how instructions are composed into one prompt for generating $c_i$​. (line 249-250)
2. Why not use Graph of thoughts as a baseline?
3. Some typos e.g. “dose” (line 161) “AutoTTS” (Table 1)

### Soundness
3

### Presentation
3

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
This paper introduces AutoMR, a framework that improves large language model (LLM) reasoning performance by automatically searching for query-aware meta-reasoning skeletons. It represents these skeletons as directed acyclic graphs (DAGs) and employs a dynamic sampling algorithm. Experiments demonstrate that AutoMR improves reasoning performance and efficiency compared to existing methods that rely on manually designed skeletons.

### Strengths
1. This paper introduces a new method that eliminate the need for traditional manual design. 
2. The dynamic skeleton sampling algorithm enables the generated meta-reasoning skeletons to possess strong query-awareness and adaptability.

### Weaknesses
I have the following concerns. *If the authors could address my concerns during the rebuttal stage, I will consider raising my score.*
1. Despite the authors stating that 3B models were used for fair comparison, I am still curious about the potential performance gains when applying AutoMR to larger LLMs, especially given the method's reported low training cost and inference efficiency.
2. The performance improvement on knowledge-intensive tasks is notably more limited compared to thinking-intensive tasks, and the method's performance on knowledge-intensive tasks seems insufficient to demonstrate the method's advantages fully.
3. As the paper only evaluates the method on math Q&A and general multiple-choice tasks across different disciplines and difficulties, I am worried about the method's scalability and broad effectiveness across a wider range of diverse tasks. I look forward to the authors providing performance results on complex reasoning datasets, including, but not limited to, Game of 24 [1], BIG-Bench Hard (BBH) [2], and Python Programming Puzzles [3].
4. Buffer of Thoughts (BoT) [4] is a great thought-augmented reasoning approach, which utilizes a meta-buffer to store and retrieve high-level thought-templates distilled from various problem-solving processes. I'm very curious about the performance difference between this work and BoT.
5. There are several spelling errors in the paper, for instance, "an" in the Table 1 caption, "subet" on page 7, and "to to enhance" on page 8.

[1] Tree of thoughts: Deliberate problem solving with large language models. NeurIPS 2024.

[2] Challenging big-bench tasks and whether chain-of-thought can solve them. ACL 2023 Findings.

[3] Programming puzzles, in Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 
2021.

[4] Buffer of Thoughts: Thought-Augmented Reasoning with Large Language Models. NeurIPS 2024.

### Questions
Please see Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
