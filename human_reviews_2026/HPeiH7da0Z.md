# Sculptor: Empowering LLMs with Cognitive Agency via Active Context Management

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 4, 4, 6, 2

## Abstract
Large Language Models (LLMs) suffer from significant performance degradation when processing long contexts due to proactive interference, where irrelevant information in earlier parts of the context disrupts reasoning and memory recall. While most research focuses on external memory systems to augment LLMs' capabilities, we propose a complementary approach: empowering LLMs with Active Context Management (ACM) tools to actively sculpt their internal working memory. We introduce Sculptor, a framework that equips LLMs with three categories of tools: (1) context fragmentation, (2) summary, hide, and restore, and (3) precise search. Our approach enables LLMs to proactively manage their attention and working memory, analogous to how humans selectively focus on relevant information while filtering out distractions. Experimental evaluation on diverse long-context benchmarks demonstrates that Sculptor significantly improves performance even without specific training, leveraging LLMs' inherent tool-calling and instruction-following capabilities. To further optimize these strategies, we introduce a novel dynamic context-aware reinforcement learning (RL) approach, advancing the training of an agent that actively modifies its own conversational history. By enabling Active Context Management, Sculptor not only mitigates proactive interference but also provides a cognitive foundation for more reliable reasoning across diverse long-context tasks—highlighting that explicit context-control strategies, rather than merely larger token windows, are key to robustness at scale.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper claims that long contexts harm LLM reasoning because earlier, irrelevant content interferes with later, relevant content (proactive interference). It proposes Active Context Management (ACM): give the model a small, deterministic toolset that lets it restructure its own conversation state. The concrete system, Sculptor, adds six tools across three buckets: context fragmentation; summary, fold, and restore; and exact-match search with optional detail expansion. Tools are reversible and preserve message order to make credit assignment stable. The authors first show that frontier closed models can use these tools zero-shot but with inefficient patterns, then improve usage via task-specific prompts.

### Strengths
The paper isolates a real failure mode—proactive interference—and targets it with simple, reversible operators rather than more infrastructure. The tool design is deliberately deterministic, self-contained, and order-preserving, which is helpful for stable RL and reproducibility claims. The modified GSPO training with conditional trajectory collection is a clear algorithmic idea for non-monotonic context states.

### Weaknesses
Several comparisons are set up against relatively weak or constrained baselines. For example, RAG is limited to BM25 without dense retrieval or hybrid retrieval, which undercuts the conclusion that ACM is stronger than retrieval-style systems. The MemAgent baseline may not reflect the best configured memory agents in current literature. The initial performance lift on frontier models depends on benchmark-specific prompt scaffolding; that makes external validity less clear and raises the question of how much of the gain comes from policy hints rather than the tool suite itself. The reinforcement learning section reports very high PI-LLM accuracy but much smaller or mixed effects on LongBenchV2 and FRAMES, which are closer to varied, open-ended reasoning. This suggests the method is strongest on structured interference tasks and less general for broad comprehension. The attention and cost analyses are compelling but would be stronger with direct wall-clock and energy plots, ablations on tool budgets, and stress tests where summaries introduce errors.

### Questions
1. How sensitive are the results to the exact tool schemas and constraints (for example, preserving message order)? If the toolset allows reordering or hierarchical grouping, does training remain stable and do results improve or regress

2. Can you provide end-to-end latency and cost curves, including KV-cache reuse effects and cache breaks caused by folding or restoring?

3. How robust is Sculptor to summary errors? If summarize_fragment drops a key detail and the agent fails to restore, what is the observed failure rate and how often does the RL policy learn to safeguard against this with targeted restore calls?

### Soundness
3

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
This paper proposed a set of active context management tools for LLMs to conduct long context editting on the fly. The 6 tools make reversable changes to the context inputs of an LLM, and especially, when triggered at the proper locations with good orders, they can filter out the irrelevent noised from the context and make the LLM's attention focus more on the critical information. The ACM tools work most effectively when specific task-based prompt engineering. The work also fine-tune a base LLM via GSPO based RL algorithm with conditional trajectory collection and incremental loss assignment. Dramatical performance boosts are achieve in 3 out of 5 benchmark datasets. The designs have novelty to a certain extent, but since the proposed techniques are only tested on 1 LLM, M3 (pretrained by the authors), the generality of the methodology can not be fully verified.

### Strengths
1. The paper provides another direction of solving the performance drop problem of LLMs due to long context without external memory augmentation.
2. The proposed tools are effective with simple prompt engineering, and their effacy is maximized by the RL based training process with a novel incremental loss assignment mechanism. 
3. The context length can be dramatically shortened so that the inference efficiency can be largely improved, while obtaining even better performances. 
4. The impact on attention to the key tokens are well analyzed.

### Weaknesses
1. The biggest concern is on whether the proposed methodology is generally effective for other open-source LLMs, such as the Qwen and Llama series. Table 3 in the appendix shows the base model comparison, which contains 2 version of Qwen3 models. Why can't the authors test their method on these two models? I notice that M3 is especially good on tool use task, while not so powerful on other tasks comparing with Qwen3. Does that mean the proposed method can only be applied by LLMs specialized in tool use, or only by M3? 
2. The RAG baseline is not representative enough. In table 2, the RAG related variants have much lower performances comparing to the base model. The authors explained that is due to BM25-only matching is used. But the problem is: embedding based RAG is widely adopted in various use cases, and if the proposed method is about to show its true impact in neither industry or academia, one main competitor is embedding based RAG. The drastic performance drops caused by the BM25-only matching clearly show it is not a proper RAG baseline.
3.  Reasoning capability. Modest gains are observed on LongBench v2 and Frames. Why the method is not that effective on sophisiticated reasoning tasks? In line 481-482, the authors mention "folding or suppressing erroneous early steps may reset the trajectory
and improve robustness", which means the proposed method has strong potential in solving reasoning problems. I understand mathematical reasoning is not exactly the same with the long context reasoning required for LongBench v2 and Frames, but the key to all of them is logical thinking. So Is this statement contradictory to the results on LongBench v2 and Frames?

I'll consider changing my score if the above concerns are properly addressed.

### Questions
See weaknesses.

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
2

### Summary
This paper introduces Sculptor, a framework designed to augment large language models (LLMs) with "Active Context Management" (ACM) capabilities. The framework equips LLMs with a suite of tools (context fragmentation, summary/hide/restore, and precise search) that enable the model to proactively manage its working memory, mirroring human selective attention and memory curation. The authors argue that mere expansion of context windows or attaching external memories is insufficient to fully address the problem of proactive interference in long-context LLMs, and they advocate for explicit, reversible, and cognitively aligned context control by the model itself. They present both zero-shot tool use and a reinforcement learning (RL) based training scheme using Group Sequence Policy Optimization (GSPO), reporting substantial gains on a range of long-context benchmarks. Comprehensive empirical analysis, including attention visualization, performance, and computational cost, supports their claims.

### Strengths
- The paper articulates a clear and compelling motivation, identifying proactive interference as an underexplored yet fundamental bottleneck in long-context LLMs, and proposes active context management as a complementary direction to architectural scaling or external-memory augmentation.
- The Sculptor framework is carefully designed and well-documented, with six context management tools whose functionalities are grounded in cognitive principles (e.g., selective attention, reversible memory) and implemented with strong considerations for determinism, reversibility, and deployment practicality (Sections 2.1–2.2).
- The experimental results are impressive.

### Weaknesses
- The reinforcement learning formulation remains under-analyzed: outcome-based rewards for long-horizon, agentic tasks such as active context management are known to suffer from credit assignment and instability issues, yet the paper provides no empirical or diagnostic analysis of training stability.
- There seems to be some missing comparisons with some well-known related works, such as MemGPT and MemoryLLM.
- Despite advocating for autonomous context management, the framework still relies heavily on manual prompt engineering to elicit effective tool usage, suggesting that its performance ceiling may depend on human-crafted strategies rather than fully learned behaviors.

### Questions
- What is the motivation behind choosing GSPO as the primary RL objective? Has the authors tried GRPO/PPO/XPO-variants?
- The paper introduces a fixed set of design principles and tool protocols for Active Context Management. Are these principles intended to remain static, or do the authors envision an evolving or self-adaptive design where the tools and their operational rules can be learned or refined jointly with the model?

### Soundness
3

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
The authors propose Sculptor, an Active Context Management (ACM) framework that gives LLMs internal tools to structure their own working memory. This approach mitigates the issue where irrelevant long-context data degrades reasoning. It improves benchmark performance, both zero-shot and when optimized with a novel reinforcement learning strategy.

### Strengths
The paper considers the setting of actively managing working memory with agents tool calls as opposed to adding everything into context (e.g. RAG). The proposed RL technique improves the model's usage of the memory sculpting tools.

### Weaknesses
The primary weakness of this paper is the severe lack of contextualization within the broader field of agent memory and context management. The authors' attempt to frame their contribution as "internal working memory" as distinct from "external memory systems" feels artificial and allows them to avoid engaging with a vast and highly relevant body of work. Moving the important related works into the Appendix also seems like a malicious attempt to evade the comparison with the many existing memory management methods.

### Questions
* Related works significantly lacks discussion on recent memory management papers (see https://github.com/FoundationAgents/awesome-foundation-agents/blob/main/assets/2-2-memory.png for some examples of relevant works).
* Baselines only consider weak retrieval baselines (with BM25) and their own ablations. Comparison with other SOTA active memory management methods (e.g. Dynamic Cheatsheet) and others agents with dedicated RAG retrievers is necessary.

### Soundness
2

### Presentation
3

### Contribution
1
