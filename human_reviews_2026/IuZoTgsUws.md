# AgentFold: Long-Horizon Web Agents with Proactive Context Folding

- Decision: Accept (Poster)
- Scores: 4, 4, 6, 8

## Abstract
LLM-based web agents show immense promise for information seeking, yet their effectiveness on long-horizon tasks is hindered by a fundamental trade-off in context management. Prevailing ReAct-based agents suffer from context saturation as they accumulate noisy, raw histories, while methods that fixedly summarize the full history at each step risk the irreversible loss of critical details. Addressing these, we introduce AgentFold, a novel agent paradigm inspired by the human cognitive process of retrospective consolidation. AgentFold treats its context as a dynamic cognitive workspace to be actively sculpted, rather than a passive log to be filled. At each step, it learns to execute a folding operation, which manages its historical trajectory at multiple scales: it can perform granular condensations to preserve vital, fine-grained details, or deep consolidations to abstract away entire multi-step sub-tasks. The results on prominent benchmarks are striking: our AgentFold-30B-A3B agent achieves 36.2% on BrowseComp and 47.3% on BrowseComp-ZH. Notably, this performance not only surpasses or matches open-source models of a dramatically larger scale, such as the GLM-4.5-355B-A32B and the DeepSeek-V3.1-671B-A37B, but also surpasses leading proprietary agents like OpenAI's o4-mini.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses the important long-history handling issue among agents, by proposing AgentFold, which consolidates the latest interactions into fine- and coarse-grained memory. Experiments across different benchmarks demonstrate its empirical advantages among open-source LMs.

### Strengths
**1. Importance of Problem and Novel Approach.**
> This paper tackles an important and prevailing issue among agents – their inability to process long contexts as tasks grow more complex. This paper proposes a new angle to actively manage, instead of passively log, past interaction history.

**2. Substantial Experimental Improvements.**
> Results in Table 1 show the empirical effectiveness of the proposed AgentFold method, among open-source models. Further analysis of the number of turns provides additional insights.

### Weaknesses
**1. Lack of Clarity in Method Description.**
> While Section 3 describes the two condensation methods at a high level, it is still unclear what counts as a “fine-grained” summary (in granular condensation) and what counts as a “coarse-grained” summary (dense condensation). It would be helpful if Figure 2 (or an additional figure) could show a few concrete examples of both.

**2. Lack of Hyper-Parameter Choices.**
> Several values could be critical to the empirical performance of the proposed method, but are not reported, justified, or hard to find. These values include: (i) number of steps to trigger consolidation, (ii) number of fine- and coarse-grained memory entries to maintain, (iii) number of steps of the “latest interaction”. Further, if these values need to be adjusted when applying the methods to different benchmarks, providing information on how these values are determined, either empirically or automatically, is important.

### Questions
1. From how humans manage memory, it seems there should be some transition between/from fine-grained memory to coarse-grained memory, yet that is not presented in the method. Why? Both intuitively and empirically.

2. How to ensure the quality of the synthesized training data for AgentFold. While the method is proposed to emulate the human memory consolidation process, it is unclear if the synthesized data ensures this critical property.

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
This paper proposes AgentFold, an agent for information seeking tasks which proactively manages its context, enabling more efficient processing and efficacy for long-horizon tasks. The context management is achieved via context folding approach which leverages two types of operations, 1) Granular Condensation that preserves fine-grained details of single steps, 2) Deep Consolidation that merges multiple steps into coarse summaries. To train the model to effectively leverage folding, a dataset is generated using an LLM, which the final model is tuned on with SFT. The experimental results on information-seeking benchmarks performance show matching or exceeding much larger models, while maintaining efficient context usage.

### Strengths
- The proposed approach demonstrates strong empirical results, consistent across several benchmarks and compared against several baseline approaches.
- The problem of context explosion in ReAct agents is well-motivated, and the design of folding mechanisms are intuitive.
- The proposed approach shows significantly enhanced context efficiency, and shows better long-horizon scaling properties.

### Weaknesses
- Missing details of Fold Generator: It is not specified which LLM is used, which prompts were used, and how many trajectories were generated and leveraged for the training data generation. Without these details, it's difficult to assess whether the improvements come from the proposed folding method or superior training data.
- Lack of ablations: Similarly, the authors do not provide any ablation studies of folding components, and it is difficult to understand how the proposed components contribute to performance.

### Questions
- Have the authors tried adding the folding mechanism to baseline and proprietary LLMs? 
- What are the failure modes of the proposed approach? Are there cases where folding can impact performance due to information loss?
- Is the approach generalizable to other long-horizon tasks, or applicable to other web automation tasks such as WebArena?

### Soundness
2

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
4

### Summary
This paper introduces AgentFold, a web agent architecture designed to handle long-horizon tasks by proactively managing its interaction history. Instead of either storing the entire trajectory (ReAct-style) or repeatedly summarizing it at every step (which risks losing crucial details), AgentFold performs learned context folding. It decides when to keep high-resolution details, when to condense recent actions into smaller summaries, and when to merge longer sub-trajectories into coarse abstractions. This yields a multi-scale memory representation that scales more gracefully with task length. Experiments are conducted on BrowseComp and BrowseComp-ZH, where the approach shows state-of-the-art performance among open-source agents of comparable and even substantially larger parameter sizes. The approach is technically sound and frames its memory representation rigorously. This paper presents an idea is both intuitive and practically impactful, and the experimental results substantiate its benefits. Some further analysis, like training pipeline cost and folding behavior interpretability, would strengthen the contribution.

### Strengths
1. The idea of proactively managing its interaction history and dynamic consolidation of interactions is intuitive.
2. The results on BrowseComp and BrowseComp-zh are promising.

### Weaknesses
1. There is no discussion of the cost of the folding procedure and its associated overhead. 
2. The evaluation on BrowseComp* is rather limited. How about adding this folding mechanism to web agents in general and evaluating on Online-Mind2Web?

### Questions
Have the authors considered adding this folding mechanism to web agents in general and evaluating it on Online-Mind2Web?

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
This paper introduces AgentFold, a new paradigm for long-horizon web agents that addresses a persistent trade-off in context management: ReAct-style agents accumulate noisy, ever-growing logs, while fully summarizing agents lose key details prematurely. AgentFold proposes a proactive “context folding” mechanism inspired by human cognitive consolidation—learning when to condense or merge past interactions into structured multi-scale summaries. The system dynamically curates its own context via two operations—granular condensation (folding a single step) and deep consolidation (folding multiple steps). It is trained through a specialized Fold-Generator pipeline producing structured reasoning trajectories, and fine-tuned on Qwen3-30B. Across benchmarks like BrowseComp, BrowseComp-ZH, and WideSearch, AgentFold achieves strong performance (36.2%, 47.3%, and 62.1% respectively), outperforming much larger open-source and even proprietary agents such as OpenAI’s o4-mini, while maintaining sublinear context growth over hundreds of steps

### Strengths
The conceptual clarity and human-analogy framing are exceptional. The authors identify context saturation as a bottleneck and present a clean, modular solution that integrates memory curation into the agent’s reasoning loop. The paper demonstrates strong experimental rigor: detailed comparisons with major baselines (DeepSeek-V3.1-671B, GLM-4.5-355B, WebSailor, etc.), concrete scaling analyses, and insightful case studies that show the agent’s ability to self-reflect and prune unproductive trajectories. The writing is polished, the methodology well-motivated, and the results both impressive and interpretable. The sub-linear token and block growth analyses convincingly validate the practical efficiency of the folding mechanism.

### Weaknesses
Overall solid paper. No significant weakness.

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
3
