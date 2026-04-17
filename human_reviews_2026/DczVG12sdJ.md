# Think in Graphs: Infrastructure and Benchmark for Large Language Model Reasoning Frameworks

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 4, 2

## Abstract
Enhancing the reasoning ability of Large Language Models (LLMs) has become a central focus of current research. While approaches based on prompt engineering have significantly improved LLM performance, the increasing complexity of reasoning frameworks has led to higher development costs. Moreover, these frameworks often require extensive redesigns to actually work on different tasks, with their performance heavily dependent on these specific designs. This creates challenges in establishing clear and consistent evaluation benchmarks. To address these issues, we propose a unified infrastructure that represents reasoning processes as graphs, thereby standardizing and structuring the reasoning workflow. This approach enables more consistent and efficient implementation of diverse reasoning frameworks, facilitates objective comparisons, and supports deeper analysis through graph algorithms. Building on this infrastructure, we develop an LLM reasoning benchmark and demonstrate its effectiveness through multiple experiments, enabling more comprehensive evaluation and analysis.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes TiG (Think in Graphs), a unified infrastructure and benchmark that represents diverse LLM reasoning methods (CoT/ToT/GoT/AoT/EGoT) as a directed acyclic graph (DAG) with user-specified rules for branching, linking, stopping, and backtracking. Rather than introducing a new paradigm, TiG serves as a graph-based intermediate representation and execution engine that standardizes how reasoning trajectories are built and run. It further contributes process-level metrics—including node/thought redundancy, optimal path cost, and a graph kernel (RGWL) for structural similarity—shifting evaluation beyond final accuracy to the structure and efficiency of reasoning. The authors also release a 3,529-item benchmark and re-implement several frameworks within TiG to enable apples-to-apples comparisons of both outcomes and reasoning behavior.

### Strengths
1. Unifying many prompt-based reasoning paradigms under a single DAG abstraction with user-specified decision and reasoning rules is a useful conceptual consolidation that can lower implementation friction and enable standardized analysis. The RGWL kernel and the newly proposed redundancy and path metrics are thoughtful add-ons for comparing process rather than only final answers.  
2. The infrastructure is specified with a concrete graph-evolution algorithm and a selection function s(v,G,\Phi) for choosing rules, plus a running example that clarifies how TiG encodes backtracking without cycles. The RGWL kernel comes with a PSD guarantee. 
3. The paper clearly separates construction / reasoning / analysis phases and provides a readable pseudo-code (Algorithm 1) for the reasoning loop. Figures and tables illustrate both the infrastructure and benchmark composition.  
4. A reasonably broad benchmark (3,529 items) plus implemented baselines (CoT/ToT/GoT/AoT/EGoT) make TiG a potentially valuable testbed for studying how structural choices affect efficiency and redundancy, beyond accuracy alone.

### Weaknesses
1. Limited novelty in core idea — While the unification under a graph abstraction is conceptually clean, it can be seen more as a standardization layer rather than a fundamentally new reasoning mechanism. Existing works on ToT / GoT and meta-reasoning supervision (e.g., ReAct-structured traces, GraphRAG introspection) already hint at “reasoning as graph”; TiG mainly formalizes this into an IR rather than introducing a new reasoning paradigm.
2. DAG constraint may oversimplify richer reasoning dynamics — Some reasoning processes naturally contain cycles, loop-style reflection, or state revisiting, which TiG forces into a DAG by construction. The paper does not quantify whether such projection into DAG introduces behavioral loss compared to native cyclic frameworks.
3. Configuration dependency & sensitivity not analyzed — The framework heavily relies on user-defined rule sets (branch, link, stop, reliance). The sensitivity of results to different configurations or misconfigured rules is not studied, leaving uncertainty about robustness and reproducibility across users.
4. RGWL kernel interpretability is under-validated — Although mathematically well-defined and PSD-guaranteed, there is no user study or empirical correlation showing that high graph-kernel similarity actually aligns with semantic/process-level similarity from a human reasoning perspective.
5. Benchmark curation lacks methodological transparency — The dataset spans legal, mathematical, and logical domains, but annotation standards, verification protocols, and inter-rater reliability are insufficiently documented, which weakens the benchmark’s credibility as a standard testbed.
6. Evaluation mostly descriptive, lacking controlled causal insight — The paper reports redundancy/exploration trends (e.g., “more branches → better accuracy”) but does not include controlled ablations on rule complexity, branch budgets, or graph evolution depth, making it hard to isolate why certain frameworks perform better under TiG.

### Questions
See weaknesses.

### Soundness
2

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
2

### Summary
This paper introduces Think in Graphs (TiG), a unified infrastructure for implementing and evaluating LLM reasoning frameworks by representing reasoning processes as DAGs. 
The authors propose that all reasoning architectures can be expressed as graph specializations. 
TiG allows users to define reasoning frameworks via configuration files specifying decision rules and reasoning rules, eliminating the need for extensive re-implementation. The paper also introduces the RGWL kernel for measuring similarity between reasoning processes and constructs a benchmark across four categories. 
Experiments compare five reasoning frameworks (CoT, ToT, GoT, AoT, EGoT) using metrics including accuracy, node redundancy, and invalid branch counts.

### Strengths
1. It provides a unified graph-based abstraction for reasoning frameworks. 
2. The included graph-based analysis metrics and RGWL are helpful in determining the effectiveness of each reasoning method.
3. It includes a good benchmark with questions from 4 categories.

### Weaknesses
1. **Limited utility and unclear guidance of proposed metrics beyond accuracy.** The paper introduces node redundancy, thought redundancy, and invalid branch counts, but fails to demonstrate how these metrics provide actionable insights beyond accuracy. For instance, CoT consistently shows 0% redundancy across all datasets but achieves the lowest accuracy, while methods with high redundancy achieve higher accuracy.  \
The paper does not clarify what high redundancy + high accuracy signifies beyond high cost, which is already captured by the "Time Cost" metric in the same tables. Without correlation analysis, ablation studies, or cost-benefit quantification linking these metrics to reasoning quality or efficiency, their value remains unclear compared to the standard accuracy/cost trade-off.

2. **Insufficient analysis of LLM backbone effects undermines benchmark generalizability**
The benchmark experiments (Tables 2-5) do not specify which LLM backbone was used, and provide no comparison across different base models. This omission is severe because different prompting frameworks may perform differently on various LLMs.

3. **RGWL kernel validation lacks depth**
RGWL kernel results only provide straightforward observations. Correct answers have higher similarities, and reasoning on legal is more stable.
It would be more exciting if we could see how to stabilize future algorithm designs or how to improve methods based on the analyses based on the RGWL framework.

4. **Unclear novelty relative to existing benchmarks and limited positioning**
Section 2.2 mentions existing reasoning benchmarks (MMLU, BIG-bench, BBH, GSM8K, MATH, ReClor, MME-CoT, REVEAL) but provides insufficient differentiation of TiG Benchmark's unique contributions. The paper states it approaches "from a different perspective...based on graphs", but does not quantitatively compare task characteristics, difficulty levels, or coverage gaps that TiG addresses.

5. (Minor) In Table 5, CoT has the lowest acc, but it is bold. The color of the \cite seems different from standard templates.

### Questions
1. What is the difference between TiG and previous benchmarks? Are the conclusions of TiG consistent with the previous ones? 

2. Can reasoning methods like latent reasoning be analyzed using this framework?

3. How generalizable is TiG to the reasoning method family?

### Soundness
2

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
This paper presents a unified infrastructure that illustrates the reasoning processes of LLMs using graphs. This infrastructure standardizes and structures the reasoning workflow, thus enabling a thorough analysis of the reasoning process. The approach is consistently and effectively implemented across a variety of reasoning frameworks.

### Strengths
1. This work is well-driven, and the proposed infrastructure is a valuable tool for understanding the reasoning of LLMs.
2. The analysis results displayed using the proposed framework are comprehensive, and they provide intriguing insights into various reasoning techniques.
3. The writing of the paper is clear and easy to follow. The figures for demonstration are highly effective.

### Weaknesses
1. Several formulas are problematic and/or unclear.
- $\text{Pa}(u)$ is used in Eq. (4) but only $\cup_{u \in \text{Ch}(v)} \text{Pa}(u)$  is defined in Eq. (3).
- The definition of Eq. (3) is also problematic: It is not appropriate to have a function with $u$ as input, but then there is $\exists u$ in the condition.
- In Eq. (4), $\\{ (u,w) \in \mathcal{E}^{(t+1)} \mid u \in \mathrm{Ch}(v),\, w \in \mathrm{Pa}(u) \\}
\cap \\{ (v,u) \in \mathcal{E}^{(t+1)} \mid u \in \mathrm{Ch}(v) \\}$ is ill-defined that $w$ shall be $v$ even in the undirected-graph case. Shall it be $\cup$ instead of $\cap$?


2. The proposed framework allows for analyzing the reasoning processes as graphs, including new metrics such as node redundancy, thought redundancy, etc. However, the analytical experiments still leave it unclear how these metrics relate to the performance of the reasoning methods.


3. The adoption of the WL-kernel for the analysis of reasoning graphs is meaningful to provide further understanding on correct and incorrect reasoning paths. However, there are flaws in it:
- The WL-kernel is proposed for undirected graphs, while the reasoning graphs are obviously directed graphs. However, the extension of the WL-kernel to directed graphs is not thoroughly discussed. For example, 
    - What is the value from the RGWL-kernel for two isomorphic directed graphs?
    - (type-ll error) Whether the isomorphic directed graph-pairs might have lower RGWL-kernel values than non-isomorphic directed graph-pairs?
    - (distinguishability) Whether the RGWL-kernel properly distinguishes non-isomorphic directed graphs? As seen in Fig.6, several C-I graph pairs have high RGWL values. Is it the limitation from the RGWL?

### Questions
See Weakness.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes Think in Graphs (TiG), a framework unifying tree or graph based multi-step LLM reasoning frameworks. It can be used as a foundation to support various such reasoning algorithms. It also proposes a Reasoning Graph Weisfeiler-Lehman (RGWL) kernel as a evaluatin metrics of similarities of multiple reasoning trajectories. It builds a multi-domain reasoning benchmark based on existing dataset, and conducts experiments and evaluation of existing LLM reasoning frameworks with the proposed TiG and RGWL.

### Strengths
+ The TiG framework is intuitive and explains in details. It covers most aspects of the previous tree and graph based multi-step LLM reasoning framework.
+ RGWL is a novel metric to evaluate the similarities of reasoning trajectories.
+ Extensive experiments are conducted on different reasoning domains and reasoning frameworks, including detailed analysis such as accuracy, time cost, redundancy, path length, and trajectory similarities.

### Weaknesses
+ TiG algorithm lacks a "reward" mechanism to prioritize some of the nodes in the rollout. For example, ToT and RAP both has such mechanism to choose the best (several) nodes for rollout before others. Besides, the algorithm is actually a BFS, lacking "DFS" variant of ToT, and thus cannot cover the original reasoning framework.
+ TiG is only a implementation foundation for other methods. The paper only evaluates existing methods (CoT, ToT, AoT, GoT, EGoT), but does not propose any new algorithms. It is not necessary that further research on multi-step reasoning are still graph-based prompting methods (e.g. the recently popular RL reasoning), which limits the use case of the proposed framework. Thus, I think TiG lacks novelty and interest to the community.
+ The paper does not mention which LLM is used (or I miss it), and does not use multiple LLMs (different open-source or API-based ones) to confirm generalizability of the evaluation results.
+ The paper does not compare the graph-based methods with other reasoning methods (e.g. enable "thinking" mode of recently popular "think" LLMs). It is not clear whether this line of research is still promising. Besides, it does not compare with the results of the original implementation of the covered methods. It is not clear whether the unified framework will possibly hurt the performance because of some missing functionality or details.
+ The proposed consists only of recently popular existing datasets, which thus cannot be considered a major contribution of this work.

### Questions
+ Which LLM is used for evaluation?
+ How is the total token usage and node count limit decided?
+ How to control the threshold of "reasoning based on $v$ should be stopped"? If the threshold is too tight, the correct trajectory may be dropped early. If the threshold is too loose, there will be exponentially increased number of nodes, so how to ensure the complexity of the graph before the above limitation is reached (so that we can at least output some answer)?

### Soundness
2

### Presentation
2

### Contribution
1
