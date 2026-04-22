# Scaling Long-Horizon Agent via Context Folding

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 4, 2, 8

## Abstract
Large language model (LLM) agents are fundamentally constrained by context length on long-horizon tasks. 
Existing agent frameworks usually rely on manually defined context engineering pipelines, such as multi-agent or post-hoc summary.
We introduce Context Folding, a framework that empowers agents to actively manage their working context. 
An agent can procedurally branch into a sub-trajectory to handle a subtask and then fold it upon completion, collapsing the intermediate steps while retaining a concise summary of the outcome. 
To make this behavior learnable, we propose FoldPO, an end-to-end reinforcement learning framework with specific process rewards to encourage effective task decomposition and context management. 
On complex long-horizon tasks, our agent matches the performance of baselines while using an active context up to 10$\times$ smaller, and significantly outperforms models constrained to the same context size.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces "Context Folding," a mechanism for LLM agents to manage their context by "branching" into sub-trajectories for specific subtasks and "returning" a summary, thereby "folding" the intermediate context. To train this behavior, the authors propose "FoldPO," an RL framework using process-level rewards (an "Unfolded Token Penalty" and an "Out-of-Scope Penalty") to guide the agent. The authors claim that on BrowseComp-Plus and SWE-Bench, their method (using a 32K active context) outperforms a 327K long-context ReAct baseline and a context summarization baseline.

### Strengths
1. The paper tackles a well-recognized and critical bottleneck in agent scalability. Managing context limitations is a key challenge for enabling more complex, long-horizon tasks.

2. The authors correctly identify that sparse task-level rewards (pass/fail) are insufficient to learn such a complex, hierarchical behavior. The design of the dense process rewards is clever and directly targets the two most likely failure modes: (1) failing to branch when needed (solved by the Unfolded Token Penalty) and (2) losing focus within a branch (solved by the Out-of-Scope Penalty). This process-oriented reward shaping is crucial for the method's success.

3. the empirical validation is comprehensive and convincing. The authors include the most critical baselines: (a) an agent with the same active context limit (ReAct 32K), (b) an agent with the same total token budget (ReAct 327K), and (c) a strong alternative approach (Context Summarization). The fact that the proposed method (with a 32K active context) outperforms the 327K long-context baseline is a significant result.

### Weaknesses
1. The "Out-of-Scope Penalty" relies entirely on a separate "GPT-5-nano" model to function. This may lead to reward hacking. Moreover, this may introduce bias originating from GPT-5-nano.

2. The context summarization baseline introduced in the paper is relatively simple, triggering only when the context is fully populated. I recommend introducing some other learning-based context summarization methods as baselines for comparison.

### Questions
see weakness

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper tackles long-horizon LLM context management by actively controlling context rather than accumulating full histories. It introduces context folding: the agent uses tool tokens to branch into sub-trajectories with filtered, task-specific context, then returns a concise summary that is folded back into the main thread. To learn this behaviour, the authors propose FoldPO, an RL method that augments GRPO with token-level process rewards to encourage effective branching, scope adherence, and compact summarization. On deep-research and coding benchmarks, the approach matches systems with ~10× larger active context and outperforms models with similar context windows.

### Strengths
- The method is principled and applicable to a wide range of tasks.
- The method is general and principled  and appears applicable across tasks.

### Weaknesses
- Missing comparisons to closely related branching paradigms (e.g., Tree of Thoughts, Graph of Thoughts, “Everything of Thought”) to position the contribution.
- Some baselines are not directly comparable: fine-tuned models are compared against pre-trained long-context models. Where feasible, fine-tune the long-context baselines (e.g., GRPO) on the same benchmarks for a fair comparison.
- The method mainly **matches** the performance of models with much larger context windows rather than surpassing them. Since the introduction argues that long windows (i) can degrade performance and (ii) incur compute overhead, the paper should quantify compute/latency savings to justify the advantage of folding over long windows if (i) is not improved.
- Efficiency is under-reported. Please include metrics such as tokens read/generated and/or the average cost per query. It is unclear whether spawning additional branches increases total token usage and cost.

### Questions
- How does the plan-execution framework in 2.2 operate during RL training? What specific failure modes does it address?
- In FoldPO, how is the frequency of branch-token calls controlled or regularized? What prevents reward hacking via excessive branching, and conversely, collapse to zero branching if gains are marginal? Did you observe either behaviour in practice?

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors propose context folding, an agentic mechanism for managing long-horizon trajectories by selectively folding ephemeral sub-trajectories while preserving only essential decision-relevant information.
Additionally, they introduce FoldPO (a GRPO-style RL variant) with token-level process rewards (e.g., unfolded-token and out-of-scope penalties) to teach when to branch/return and what to retain.
On BrowseComp-Plus and SWE-Bench Verified, they report competitive pass@1 with a 32K active context (up to 10 branches), outperforming summarization baselines and approaching long-context (327K) ReAct agents.

### Strengths
* Ablations of the algorithms are informative.
* Clear problem framing (linear context growth), clean mechanism (branch/return), and simple plan–execute instantiation.
* Strong empirical performance at fixed active context; shows the benefit of learned context management vs. naive summarization.
* RL helps materially (FoldPO > vanilla GRPO); behavior analyses (more tool calls, more thorough exploration) are useful.
* Presentation is clear; diagrams/examples make the folding workflow easy to follow.

### Weaknesses
* Empirical validation lacks error bars; would encourage authors to always run experiments for multiple random seeds then report error bars.
* Missing direct comparisons to the most relevant recent systems: ReSum (periodic learned summarization), MemAgent/MEM1 (RL-trained memory/constant-size state). Without these, novelty/positioning is underspecified.
* Overclaims: “SOTA/comparable to 100B+” is too strong given GPT-5 baselines still lead; please tone down and specify precisely the scope (at fixed active context, same 36B base model, etc.).
* Summarization baseline may be underpowered vs. recent learned-memory methods; please ensure parity (tools, RL, prompts) or include those methods as baselines.
* No per-component reward ablation (e.g., remove unfolded-token penalty, remove out-of-scope penalty) to show each term’s necessity.
* No evaluation of summary fidelity (do fold summaries ever drop critical info?); even a small audit study or automatic factuality check would help.
* Minor inconsistency in reported numbers across text/table; ensure a single definitive pass@1 is reported and clarify variance across runs.
* Heavy reliance on an LLM judger (BrowseComp-Plus): discuss robustness/sensitivity; consider human spot-checking or multiple-judge aggregation.

### Questions
* Can you provide head-to-head results vs. ReSum and MemAgent/MEM1 under identical settings? If code/results are unavailable, a careful reimplementation of their summarization/memory policies would strengthen claims.
* Reward design ablation: what happens when each process reward is removed independently? Any degenerate behaviors (e.g., never returning, excessive branching)?
* Summary fidelity: do you have any quantitative/qualitative analysis showing critical facts survive folding? What failure modes did you observe?
* Please clarify the small discrepancy in BrowseComp-Plus pass@1 between text and table; also report mean±95% CI over multiple seeds.
* Compute/latency: how does rolling back KV-cache and creating branches affect wall-clock time vs. long-context and summarization baselines?
* Generality: outside web/coding, how does folding behave in dialogue or embodied tasks? Any constraints on nested tasks or multi-level folding?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper introduces Context Folding, a framework for long-horizon LLM operation that temporarily “branches” to perform token-intensive subtasks before folding the resulting information back into the main context via a compact summary, keeping the main thread manageable in scale. The core claim is that learning when to branch and how to summarize improves both quality and efficiency compared to ad hoc summarization prompting or simply scaling to longer contexts. This is paired with FoldPO, a variant of GRPO that uses token-level process rewards (a penalty for long main threads and a judge-based penalty for out-of-scope actions within branches). Empirically, on BrowseComp-Plus and SWE-Bench Verified, the method improves pass@1 on long-horizon benchmarks (e.g., browsing and code-fix tasks) while maintaining a much smaller active context than a long-context ReAct baseline and a context summarization method.

### Strengths
* **Originality.** Turning context management into an explicitly learned skill is a valuable idea at the intersection of hierarchical planning and summarization. Additionally, FoldPO's process-reward shaping directly targets the desired properties.  
* **Quality.** Experiments cover multiple tasks and ablations, indicating that the different design choices matter.  
* **Clarity.** The paper is well written and easy to follow; the figures effectively illustrate how branches are created/closed and how folded summaries condition subsequent actions.   
* **Significance.** The approach can reduce reliance on very long contexts and enable more efficient agents, which constitutes a timely contribution.

### Weaknesses
* **Heuristic reward shaping.** The out-of-scope reward is provided by an auxiliary LLM judger. This risks bias/reward hacking. The unfolded-token penalty uses a fixed fraction of the working context budget, which seems very ad hoc and leaves room for designs that might lead to better learning.  
* **Definition clarity.**   
  * Equation (1) currently represents a probability over full traces at the action level, with the use of T that is dubious. Since the method in FoldPO is autoregressive at the token (or, if you prefer, action) level, I would encourage writing Eq. (1) in terms of the next-action distribution rather than the full-trace distribution, since I have doubts it is currently formally correct.   
  * Q appears in the objective (L202) before it is defined. I encourage introducing Q immediately where first used.  
* **Efficiency accounting.** Results emphasize *active* context compression. To highlight this, it would be relevant to include wall-clock latency and the total number of tokens processed (main thread \+ branches) per instance.  
* **Formatting issues.** In Table 2, the value 78.10 should be (\<1) (likely 0.781).  
* **Related work**. See the works below, which, although they do not subtract from the contributions of this work, which lies at the agency of context management, should be properly discussed in the related literature.

\[1\] Xu, B., Peng, Z., Lei, B., Mukherjee, S., Liu, Y., & Xu, D. (2023). ReWOO: Decoupling Reasoning from Observations for Efficient Augmented Language Models. *ArXiv*. [https://arxiv.org/abs/2305.18323](https://arxiv.org/abs/2305.18323)  
\[2\] Holt, S., & Luyten, M. R. (2023). L2MAC: Large Language Model Automatic Computer for Extensive Code Generation. *ArXiv*. [https://arxiv.org/abs/2310.02003](https://arxiv.org/abs/2310.02003)  
\[3\] Ning, X., Lin, Z., Zhou, Z., Wang, Z., Yang, H., & Wang, Y. (2023). Skeleton-of-Thought: Prompting LLMs for Efficient Parallel Generation. *ArXiv*. [https://arxiv.org/abs/2307.15337](https://arxiv.org/abs/2307.15337)  
\[4\] Besta, M., Blach, N., Kubicek, A., Gerstenberger, R., Podstawski, M., Gianinazzi, L., Gajda, J., Lehmann, T., Niewiadomski, H., Nyczyk, P., & Hoefler, T. (2023). Graph of Thoughts: Solving Elaborate Problems with Large Language Models. *ArXiv*. [https://doi.org/10.1609/aaai.v38i16.29720](https://doi.org/10.1609/aaai.v38i16.29720)

### Questions
1. In eq 1, how is T formally defined? Would you be willing to refactor Eq. (1) into an autoregressive one?  
2. You disabled nested branches for simplicity. Did you attempt it and observe failure modes when enabling shallow nesting? Although this is left as future work, preliminary results would, in my opinion, greatly increase the contribution.

### Soundness
3

### Presentation
3

### Contribution
3
