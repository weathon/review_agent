# CEA: Context Engineering Agent for Enhanced Reliability and Sustainability in Deep Research Systems

- Avg Score: 3.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 2, 4

## Abstract
The increasing capacity of frontier models to process long contexts has fueled enthousiasm for deep research agents. However, longer contexts alone do not guarantee better responses. In fact, context overloading can lead to unexpected agent failures. 
To tackle this challenge, we introduce an autonomous context control framework built around a Context Engineering Agent (CEA). The CEA maintains structured context by efficiently managing historical interactions, tracking ongoing progress, and identifying critical clues, hence achieving an optimal trade-off between token efficiency and memory integrity.
In conjunction with this framework, we introduce CERL, an end-to-end multi-turn reinforcement learning method designed for CEA. We enhance training by filtering out trajectories with non CEA-attributable errors before gradient updates, thereby enhancing the stability of training.
Our \textbf{CEA} approach has demonstrated substantial efficacy in enhancing performance on complex information-seeking tasks, as evidenced by increased interaction sustainability and notable performance improvements across various benchmarks.
Despite its sophisticated context-processing mechanisms, CEA is a plug-and-play solution that seamlessly integrates into existing systems, enhancing agents' context management with minimal code modifications. This combination of internal sophistication and external simplicity makes CEA both powerful and practical for real-world deployment.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work aims to tackle the problem of excessive context lengths required by agentic frameworks by introducing Context Engineering Agent (CEA). CEA is an isolated module and instantiated as an LLM that compresses a task query, a dynamic plan, an episodic history, and semantic facts into an actionable context for a reasoning module represented by another LLM. Furthermore, CEA can be trained via RL to maximize task completion including a certain filtering that isolates contributions of CEA. The method is evaluated on several benchmarks and the results show consistent improvements over not using CEA and other agentic frameworks.

### Strengths
- Well-structured experiment design
- CEA provides consistent performance improvements 
- CEA is agnostic to the reasoner and can be incorporated into any agentic pipeline

### Weaknesses
**Novelty**

The main contribution of this work revolves around compressing and abstracting context into smaller digestible chunks that improve performance and enables longer rollouts. One concern of mine is that prior works like [1,2] already explored similar ideas and it is not clear to me whether the contributions of this work are novel. This is the main reason for my current rating leaning towards rejection, however I am not particularly familiar with the literature on agentic frameworks, so if the authors can clarify the novelty of their contributions I am willing to increase my score.

[1] Reflexion: Language Agents with Verbal Reinforcement Learning, Shinn et al., NeurIPS 2023

[2] Generative Agents: Interactive Simulacra of Human Behavior, Park et al., UIST 2023

**Baselines**

The comparisons to baselines appears pretty sparse, namely there are only comparisons to ReAct and Search-o1. Are these agentic frameworks state-of-the-art on those benchmarks? Are there any other baselines that could be compared to, like [1,2]? A more thorough comparison would help determine the significance of results. In addition I recommend to also report error bars for the collected results.

[1] Reflexion: Language Agents with Verbal Reinforcement Learning, Shinn et al., NeurIPS 2023

[2] Generative Agents: Interactive Simulacra of Human Behavior, Park et al., UIST 2023

**Presentation**

While the paper is generally easy to follow it lacks important details, especially when it comes to the methodology. For example, to me it is unclear what the input to the CEA module is, Figure 3 shows only the query as input, while equation (1) and Section 3.1.2 list more information. LLM-as-a-judge is mentioned for assigning rewards without elaborating on how it is implemented. Reward distribution is mentioned without any specifics. Figure 3 also shows a reference model which is not explained at all in the methodology. It is unclear how a rollout looks like and how it is scored. Moreover, Equation (2) contains several symbols that are undefined, such as $\theta$, $S$, $\mathcal{T}_\text{all}$. To clarify those points, I recommend to add an algorithm that clearly defines all introduced symbols and formally describes how the rollout and the update of CEA works in detail. 

Another point I was struggling with is to understand the parallel execution of CEA. How is it possible to execute CEA in parallel during inference when performing a rollout? Or is the parallel execution strictly about the training phase? If so, this should be clarified in the text, ideally in the methodology section.

**Minor remarks:**

- Table 1 is confusing as the second column in the first block is different from the same column in the lower block without explicit mention about it.

### Questions
Line 239: How is the reward distributed across the entire trajectory?
- Is CERL just a fine-tuned Qwen3-8B model? It is not really explicitly defined anywhere, or maybe I missed it.
- An important missing information is some characteristics about the used benchmarks, i.e. how many turns do they feature, what context length would be required if you provide everything in context, etc. This is valuable to determine how much CEA compresses the context.
- What is the concept of compelled response generation?
- It would be good to provide a definition of the term sustainability, it was not always entirely clear what it means to me.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors have developed a CEA (Context Engineering Agent), which is a modular architecture which claims to help to manage context in long-horizon LLM-based general research agents . It aims to fix context rot through a two-component design that splits the processing of reasoning from context management. CEA is built from a four-part context (Task Query, Dynamic Plan, Historical Memory, Semantic Facts) and is trained via a novel algorithm which is labeled Context Engineering RL, which filters out trajectories to focus on context-attributable errors. Experiments on three web-browsing benchmarks (two of which are the same, but one in Chinese and one in English) show 30–100% improvements over ReAct (2023) and Search-o1 (2025) baselines.

### Strengths
1) The problem that is being tackled (context rot) is a well-known one, which is clearly a bottleneck in many long context-window systems, so the aims of the work are clear and important.
2) The architectural design itself is clean and makes intuitive sense. Having the Reasoner/CEA split is sensible, and the four-part context representation is well-structured and also aligns with the cognitive needs of multi-step reasoning.
3) The empirical results are clearly very strong and show very large improvement over the baselines.
4) There are some pseudo-ablations implemented (invocation frequency comparison, serial versus parallel execution, compelled response versus standard reasoning)

### Weaknesses
There are some very important weaknesses to the work, which I believe are, at the moment critical, but could potentially all be addressed.
1) The main methodological innovation in CERL seems to be that of filtering trajectories to find CEA-attributable errors. However, the details of this are entirely missing. The paper refers to sections of the supplementary material (appendix 11 and 12) for algorithmic details, but these don't seem to exist and contain neither pseudocode nor ablation results which verify the mechanism. The explanation is completely absent from the main text where the contribution is claimed. The main missing details are:
a) The algorithm or indeed any heuristic for error attribution.
b) The labelling procedure (human vs. automated) and error taxonomy
c) The filtering statistics (acceptance rate, error-type distribution)
d) Any validation method for attribution accuracy
e) Any true ablations: no-filter, random-filter, threshold sweep
2) There is only a very narrow task scope. The paper claims to solve a problem in a lot more generality than is actually shown. All of the benchmarks involve web-information retrieval. There are no benchmarks that include pure reasoning without external search (like mathematical reasoning, code generation, or long-context Q&A). This is fine, but then the claims have to be appropriately tempered. The empirical results only indicate that the approach works as a specialized web-retrieval agent
3) The baselines are very limited. One baseline is from 2023 and only one is from 2025, but in the related material, many other approaches are discussed (like HiRA, BrowseMaster, WebRL, and DeepResearcher). It's claimed that "These approaches have significantly advanced agent capabilities but have not specifically focused on optimizing context management as a distinct learning objective" but they still seem like important comparators. It's not clear why there is such a narrow comparison when it would seem that there are other systems out there that may be different but still comparable in terms of empirical results.
4) The authors have included prompt templates and an overview of the architecture but they haven't included some really important information, like the CERL hyper parameters, the compute and data budgets, the parallel execution protocol or more in-depth wall-clock profiling (there is, but it's very limited). Where there are prompt templates, there's no clarity in the details.
5) As indicated already, there are major issues with the supplementary material:
a) There are no numbered or labeled sections despite references to them (appendix 11 and 12 are mentioned in the main text, but don't exist).
b) There is placeholder text such as “add caption” left in tables.
c) There is also misplaced content, including a block labeled turn 31 appearing mid-section and Example 2 which is presented out of sequence.

Some more minor typos and issues:
Typo: “enthousiasm” → “enthusiasm”
Figure 2: What does "CEA pad zoom details." mean?
Eq. 2: sampling process undefined anywhere

### Questions
The questions are all related to the weaknesses and so there is a lot which is unclear from the text which needs to be addressed.

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
2

### Summary
The work attempts to address an alleged inference failure called “context rot,” arguing that as the context window grows, large language models’ accuracy degrades and on binary-success tasks could fail entirely. To counter this, the authors propose a Context Engineering Agent (CEA) that leverages trajectories of intermediate reasoning and responses to track task progress and surface the indispensable words within each step; they also explore boosting reasoning with GRPO and CEA as CERL. Empirically, CEA outperforms ReAct and Search-o1 across reported settings. Results for the CERL variant are more mixed: on XBench-DS it ties GPT-OSS-120B and surpasses Doubao-Think-1.6, while larger models remain stronger on other benchmarks. The reported improvement depends on applying CEA at every turn of the reasoning process. The authors then investigated compelled response generation.

### Strengths
* Demonstrates consistent performance gains across numerous experiments, even when limited to small language models. This is useful for realistic settings with tight memory and compute budgets.
* Presents a model-agnostic approach: CEA can be layered onto existing LLMs by adjusting reasoning chains and context handling, making it straightforward to adopt without retraining full models.
* Provides clear prompt templates in the appendix, which improves transparency and reproducibility for future comparisons and deployments even if the code is absent.

### Weaknesses
* The abstract’s claim that “longer contexts alone do not guarantee better responses” is uncited and conflicts with parts of the scaling literature (see https://aclanthology.org/2024.naacl-long.260/, https://arxiv.org/abs/2403.05530, https://aclanthology.org/2024.acl-long.776/, https://arxiv.org/abs/2402.13753 ); it needs empirical support or authoritative references. 
* Marketing-style metrics (“optimal trade-off between token efficiency and memory integrity”) are undefined; there’s no formal metric, measurement procedure, or units reported.
* The “context rot” problem relies on a single technical report (Hong et al., 2025) from a startup; the authors are not findable on Google Scholar, and there’s no peer-reviewed corroboration. Further, the propagation of errors affecting performance is well known in lots of sequential modelling tasks for machine learning. I don’t see how creating a theory de nouveau “context rot” offers anything substantial when long-term credit assignment (see https://ieeexplore.ieee.org/document/4066245, https://dl.acm.org/doi/10.5555/2969239.2969370,) /vanishing gradients (see https://ieeexplore.ieee.org/document/279181, https://ieeexplore.ieee.org/abstract/document/6795963) /redundant information (see https://arxiv.org/abs/physics/0004057)  has existed in machine learning for a while. The authors should engage with more foundational work. 
* Typo: “yan yan” on line 064.
* Long-horizon claims (e.g., “compounding errors accumulate rapidly…”) are uncited and should be backed by prior work or new experiments.
* I want to acknowledge related work is mostly from 2025 preprints. This may be common with how fast LLM development happens but engaging with past related work is essential and offers better context on the novelty of the presented work; established literature on long-horizon reasoning, memory, retrieval/summarization, and hierarchical planning is under-cited.
* Reward design is underexplored as a 0/1 output; stronger alternatives (eg. KL-regularized rewards, learned judges,) are not compared.
* GRPO is re-derived without a clear difference from the standard method; the only change appears to be using CEA trajectories.
* Performance claims are overstated: CEA performed better than search o-1 and ReAct but CERL does not perform well over normal CEA for the majority of models and ties/loses on some benchmarks; qualify wins, ties, and losses precisely. Why use CERL at all?
* Benchmark scope is limited by search-API quotas. Remaining benchmarks need to be included.
* “Affect” should be “effect” in “greatly affect the inference performance.”
* Baseline/model choices are questionable (use of an un-optimized Qwen3-8B rather than an optimized Qwen).
* The section “HOW OFTEN SHOULD WE USE CEA TO MANAGE CONTEXT?” lacks motivation given the finding that it “always improves”; it reads like an appendix-level ablation.
* The statement that prior “deep research” compelled response generation for higher scores lacks citations.
* The references are ill formatted in the middle of the appendix. This is ironic in that the authors acknowledged LLM assistance for writing a paper about improvement LLM’s retrieval and response abilities

### Questions
This work generally does not feel finished and the claims seem too extended from what was presented. Particularly, the references in the middle of the appendix and not evaluating on all benchmarks  (since they are different from each other) for the last half of the paper allude to unfinished work.



* Why not replicate the experiments from the ‘Context Rot’ technical report to improve the validity of this allegedly unique phenomena? 

* The paper mentioned CEA and CERL decouples context engineering from the primary reasoning tasks. Why not test performances on Graphwalks (https://huggingface.co/datasets/openai/graphwalks), which was tested in the ‘Context Rot’, and the matheval dataset (https://huggingface.co/datasets/RyanYr/MathEval) to see how reasoning ability is maintained or not?

* GRPO is a popular baseline but why not use Dr. GRPO (https://arxiv.org/abs/2503.20783) or RLOO with a normalized baseline instead due to the resource constraints? 

* None of the accuracy metrics in the tables include an interval of uncertainty like standard error. Why not decrease the temperature to .80 or more and use the mean accuracy to gather a standard error or deviation? If omitting the uncertainty intervals is standard in this research area, it would be novel to include them.

* Does CERL possibly scale with larger models like QWEN 32B?

### Soundness
2

### Presentation
1

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
The paper studies deep research agents, focusing on the challenge of long-context reasoning. Longer contexts do not necessarily lead to better responses.

To address this issue, the paper introduces an autonomous context control framework built around a Context Engineering Agent (CEA). The proposed method maintains a structured context by efficiently managing historical interactions, tracking ongoing progress, and identifying critical clues.

In addition, the paper presents CERL, an end-to-end multi-turn reinforcement learning method designed specifically for training the CEA. The overall framework leverages hierarchical agent architectures and reinforcement learning for LLMs to achieve effective context management.

### Strengths
The problem of ensuring the reliability and robustness of deep research agents is interesting and important.

CEA provides an effective approach to optimising context management.

The paper’s use of hierarchical agent architectures and reinforcement learning for LLMs in managing context is also noteworthy.

### Weaknesses
What is the relationship between the CEA module and the LLM-powered memory operations?

It would be helpful to provide more details about Figure 2. Specifically, where are the four components located in the figure, and what does the Plan Checklist represent?
Does the History Summary include only the content generated by the LLM, or are other sources involved?
How is the Clues Memo constructed?

CEA appears to follow a hierarchical agent architecture. In Figure 3, is the CEA rollout performed by a single agent? It would be better to include an example of a trajectory for clarity.
What is the reference model used in the experiments?

The training details of the CEA model are missing and should be described more thoroughly.

In Table 1, the proposed method does not outperform Qwen3-32B, which requires further explanation.

### Questions
Could the paper provide more details to enable the reproduction of the results?

### Soundness
4

### Presentation
2

### Contribution
3
