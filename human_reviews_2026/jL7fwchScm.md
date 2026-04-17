# ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory

- Decision: Accept (Poster)
- Scores: 6, 8, 6, 6

## Abstract
With the growing adoption of large language model agents in persistent real-world roles, they naturally encounter continuous streams of tasks. A key limitation, however, is their failure to learn from the accumulated interaction history, forcing them to discard valuable insights and repeat past errors. We propose ReasoningBank, a novel memory framework that distills generalizable reasoning strategies from an agent's self-judged successful and failed experiences. At test time, an agent retrieves relevant memories from ReasoningBank to inform its interaction and then integrates new learnings back, enabling it to become more capable over time. Building on this powerful experience learner, we further introduce memory-aware test-time scaling (MaTTS), which accelerates and diversifies this learning process by scaling up the agent's interaction experience. By allocating more compute to each task, the agent generates abundant, diverse experiences that provide rich contrastive signals for synthesizing higher-quality memory. The better memory in turn guides more effective scaling, establishing a powerful synergy between memory and test-time scaling. Across web browsing and software engineering benchmarks, ReasoningBank consistently outperforms existing memory mechanisms that store raw trajectories or only successful task routines, improving both effectiveness and efficiency; MaTTS further amplifies these gains. These findings establish memory-driven experience scaling as a new scaling dimension, enabling agents to self-evolve with emergent behaviors naturally arise. Our code can be found at https://github.com/google-research/reasoning-bank.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This work addresses a key limitation of large language model agents: their inability to learn from accumulated experience in ongoing tasks like web browsing and software engineering. The authors propose two innovations. REASONINGBANK is a structured memory that extracts generalizable reasoning strategies from both successes and failures, operating in a loop of retrieval, construction, and integration. MATTS combines REASONINGBANK with test-time expansion: memory guides exploration while diverse expansions enrich memory and improve performance. The main contributions are a memory framework that leverages failures for generalizable reasoning, a synergy with test-time expansion, and empirical validation of memory-driven agent self-improvement.

### Strengths
1. The problem definition is novel, focusing on extracting reasoning from experience rather than mere experience storage, and highlighting the value of failed experiences.
2. The experiments are extensive, with diverse benchmarks and thorough ablations, providing strong evidence for the effectiveness of REASONINGBANK.

### Weaknesses
1. The paper evaluates generalization only within subsets of WebArena and Mind2Web. The claimed generalization gains may simply reflect similarities between tasks within the same benchmark, as human or LLM patterns are often consistent internally. Stronger evidence would require cross-benchmark tests, such as using WebArena Shopping as an experience pool to evaluate completely unrelated Mind2Web tasks, or testing SWE-Bench with WebArena experiences.
2. Success and failure of trajectories are judged by the LLM, but the accuracy of these judgments is not analyzed. Using the LLM as a critic may propagate errors.
3. Experiments focus only on historical task-level memory, without considering temporal or user-level memory, which are more valuable in practice. Task-level memory yields only modest metric gains that could also be achieved with additional SFT data or RL, but at the cost of efficiency, storage, and increased framework complexity. Temporal or user-level memory cannot be solved as easily and would provide more significant practical value, yet these aspects are not studied.
4. The analysis of emergent behavior is limited to changes in the memory bank. It does not examine whether prior failed tasks can be corrected through reflection on successes and failures, which would demonstrate true emergent capability. Currently, the memory module mainly helps stabilize model outputs via complex multi-step processes, increasing engineering complexity, lowering scalability, and reducing efficiency. SFT or RL may achieve similar gains more efficiently. Showing experiments that capture emergent improvement would significantly increase the work’s impact.
5. Only strong closed-source models are evaluated. Open-source models and models of different sizes are not tested.
6. Experiments are limited to web navigation and software engineering agents, with insufficient evaluation on more general tool-use agents such as Taubench or BFCL. Moreover, the experience extraction prompt appears tailored to web navigation, implying that each new scenario may require a custom prompt and tuning, reducing usability and scalability.
7. While the paper claims efficiency gains in terms of fewer steps, actual runtime may be higher. The memory module adds significant inference overhead, particularly on the TTFT, which may degrade user experience and hinder practical deployment in real-world agent scenarios.

### Questions
1. Has any subset of trajectories been manually labeled to verify the consistency between LLM Critic judgments and human labels? If there is a discrepancy (e.g., LLM misclassifies 15% of failed trajectories as successful), how would this affect the quality of REASONINGBANK memories and downstream agent performance?
2. How does REASONINGBANK perform when the memory bank grows large (e.g., over 500 entries)? Could pruning or merging strategies, such as removing low-similarity memories relative to the current query, maintain or improve performance? Which metrics are used to evaluate the effect of such optimizations?
3. In sequential expansion, how is the termination condition determined (e.g., fixed k=5 steps versus stopping when no new insights emerge)? Do intermediate optimization notes (e.g., “recheck filtering labels”) provide unique value to memory, or are they largely redundant compared with final trajectory insights?
4. Can memory entries from web browsing tasks transfer to software engineering tasks (e.g., using a “web form data consistency check” strategy for code debugging)? If not, how can the framework be adapted to support cross-task memory transfer?
5. Are failures categorized by type (e.g., “operational errors” versus “reasoning errors,” such as clicking the wrong button versus misinterpreting a user query), and are separate insights extracted for each type? Would such granularity improve the practical usefulness of the memory?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper presents a memory framework for agents called REASONINGBANK and a complementary memory-aware test-time scaling (MATTS) technique. Unlike previous agent memory frameworks, REASONINGBANK (a) extracts generalized learnings (so-called memory items) from previous reasoning trajectories (i.e., attempts at solving a problem) instead of simply memorizing the entire raw trajectories and (b) learns by comparing successful and unsuccessful trajectories instead of only learning from the successful ones. MATTS complements REASONINGBANK by generating multiple reasoning trajectories, either in sequence or in parallel and extracting memory items from across all trajectories. The authors empirically demonstrate that REASONINGBANK substantially outperforms other agent memory frameworks and a baseline without memory in both task-specific success rate as well as efficiency (measured by the number of reasoning steps). They further show that REASONINGBANK and MATTS work particularly well in combination.

### Strengths
Originality: While the key ideas (e.g., generalizing learnings, incorporating learnings into prompts, multiple reasoning trajectories, etc.) behind REASONINGBANK and MATTS are not entirely novel and have also been used in similar papers in the last 2 years, the exact way how these techniques are put together and implemented is sufficiently original and this apparently leads to a substantial outperformance of existing approaches.

Quality: The implementation details (e.g., prompts) presented in the main part and the appendix appear very well-crafted. The chosen evaluation experiments are rigorous and suitable to test various aspects of the presented methods.

Clarity: The vast majority of the paper is very clearly presented in both text and visuals. The paper offers easily accessible explanations to all parts of the framework and makes suitable references to the appendix which includes an even more comprehensive documentation. Key ideas (e.g., learning from successful and failed experiences alike) are being repeated in several places, because they are so important.

Significance: The paper presents a well-performing solution to a timely and relevant problem. It further makes extensive references to recent related work and similar techniques. The advantage of the presented methods over their baselines is so substantial and consistent that it can hardly be ignored.

### Weaknesses
In some areas the paper could have been more precise or elaborate with some explanations, e.g.,
- The REASONINGBANK framework includes a memory consolidation step. While the name suggests that memories are actually being consolidated (i.e., transforming a set of memories into a somewhat more information-dense, better structured, or simply smaller set of memories), the used implementation does nothing of that but simply appends new memory items to the set of existing ones. This appears like a waste of potential. However, the authors offer a justification for their choice in the appendix (lines 927ff.) arguing that advanced consolidation mechanisms could have led to unwanted confounding factors, and leaving such mechanisms for future work. For their paper, the authors may want to rename “memory consolidation” into something that doesn’t use the word consolidation (which suggests that something more complex is going on) and/or making clearer why they decided to keep their implementation simple in the main part, not just the appendix.
- Line 260 mentions the two main performance metrics “success rate” and “average steps”. While the readers may already have a good intuition about these metrics, it could have been even more clear if the authors had given a more precise definition of how the success rate is calculated and what constitutes a single step.
- The concept of “rollouts” is mentioned in line 81 but not further explained. It is not immediately clear what rollouts are and how they relate to other parts of the method.

Some aspects of the framework remain somewhat unclear to me, for instance:
- Lines 428f. mention that memory items can gradually evolve over time. It is not clear to me how this is possible when no actual consolidation of memory items is happening but when new memory items are simply added. Perhaps the authors mean that memory items evolve into new and somewhat more advanced memory items? The sentence, however, would suggest that REASONINGBANK includes a mechanism that alters the same existing memory items. If so, this should be better explained.
- Line 341 presents “MATTS w/o aggregation” as one of the methods tested in one of the experiments. At this point, it was not entirely clear to me, what the aggregation is and how it works. Perhaps the authors could make this more explicit in some of the previous explanations.

The paper makes some claims that are not entirely credible and could benefit from additional evidence:
- Lines 294ff. make the claim that REASONINGBANK employs a memory extraction strategy that is superior to those of other frameworks. Unlike most other claims in the paper, this claim seems not entirely backed by evidence, as there are no more detailed experiments capable of exactly attributing REASONINGBANK’s outperformance to the memory extraction component. While this appears likely, it could have been other factors such as the memory retrieval, the memory consolidation, or the way that memory items are structured as well.
- The authors present several experiments exploring various aspects of REASONINGBANK and MATTS. The numbers generally look very favourable for their methods, yet some of the experiments are only performed on subsets (e.g., subset of the tested LLMs, subset of the datasets). Not performing all experiments on the full set of all possible combinations is understandable due to budget limitations, however, it is unclear if these restrictions somehow affect the reliability and generalizability of the results. Perhaps, the authors could add at least some error bars or confidence markers to their results to provide an indication of the degree of noise and fluctuation of their results and hence their reliability.

There are also minor issues with the presentation and cosmetics:
- Section 3.1 presents a mathematical problem formalization. I see that the authors likely added this, because some reviewers expect a mathematical formalization in every paper. While the formalization is clear and useful here, it is not used anywhere else in the paper and thus appears as it was simply added pro forma.
- The paper includes a small number of typos, e.g., lines 233 and 238 appear to have been changed and now include incorrect phrases “we generates” and “we refines”.

### Questions
Based on the mentioned weaknesses, the following questions can be raised to the authors:
1. How can memory items gradually evolve? What is the underlying mechanism?
2. How does “MATTS w/o aggregation” work? A short explanation may be useful for readers unfamiliar with vanilla TTS.
3. How large are the confidence intervals of the results, especially those results that are only presented for a subset of the LLMs and datasets and how does that affect their reliability and generalizability?

### Soundness
3

### Presentation
4

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
The authors introduce two key enhancements to existing memory systems. First, they design a memory framework that performs knowledge distillation not only from correct trajectories but also from potentially incorrect ones. Second, during test-time scaling, they propose distilling knowledge from multiple parallel or refinement trajectories (denoted as k trajectories). Their approach is empirically validated through extensive experiments on web navigation and software engineering (SWE) problem-solving benchmarks, demonstrating superior performance over baseline methods. Additionally, ablation studies highlight the contribution of learning from "incorrect" traces.

### Strengths
1. The paper presents a memory system that leverages all available logs (both correct and incorrect) and thoughtfully explores its integration with test-time scaling methods.
2. The authors conduct a comprehensive set of experiments and analyses to substantiate their claims.
3. The paper is clearly written and easy to follow.

### Weaknesses
1. The proposed memory system may incur high costs since it relies on multiple LLM calls per task trajectory to derive the memory item. Reporting the cost of each method, including memory updates, would help justify the method’s benefits.
2. Because memory updates depend on several LLM calls, the system risks compounding errors—for instance, if the LLM judge incorrectly classifies trajectories or if knowledge distilled from different trajectories fails to generalize.
3. The experiments are limited to the Gemini-2.5 family of models; testing on smaller, publicly available models could strengthen the generalizability of the findings.

### Questions
Are there known cases where the LLM judge misclassifies correct versus incorrect traces? If so, what strategies or mechanisms could mitigate such misclassifications?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes ReasoningBank, a structured and interpretable memory for continual test-time learning in LLM agents that distills reusable strategies from both successful and failed trajectories. It further introduces memory-aware test-time scaling (MATS), combining parallel and sequential exploration to generate richer, more generalizable memory entries. On WebArena, Mind2Web, and SWE-Bench-Verified, the approach delivers consistent performance and efficiency gains over memory and non-memory baselines, with analyses on memory evolution, the value of failures, and compute efficiency.

### Strengths
- **Clear Problem Motivation and Context.** The work identifies that existing memory modules limited learn from failures and disproportionately weight successes, and it clearly argues why this imbalance undermines long-term agent adaptability.

- **Experimental Breadth and Depth.** Evaluation covers diverse agentic tasks (WebArena, Mind2Web, SWE-Bench), several LLM backbones, and robust memory baselines, assessed via effectiveness, step efficiency, and transfer/generalization. Performance comparisons are clear in Tables 1 and 3; Table 4 provides a focused breakdown of success vs. failure step efficiency.

- **Concrete Technical Design.** This method organizes self-verified trajectory analyses into a structured memory of compact, modular strategies, improving both interpretability and reusability.

### Weaknesses
- **Ambiguity in Self-Judgment/Verification.** Because trajectory labels rely on the model’s own self-assessment, robustness is uncertain: calibration, error propagation, and blind-spot risks are not quantitatively evaluated. Targeted ablations (e.g., noisy/adversarial self-judgments, third-party or rule-based verifiers) would clarify how sensitive the curation process is. 


- **Scalability and Long-Term Lifelong Learning**: Although the paper claims to address "lifelong agents," the experiments are limited to several hundred tasks, and do not simulate truly open-ended deployment or demonstrate the maintenance/pruning of memory over extended horizons. No experiments address issues like memory interference, long-tail drift, or resource constraints on memory expansion.

- **Missing Failure/Robustness/Noise Analysis.** Although Figure 7 illustrates handling of failure trajectories, the work lacks a deeper study of negative cases, when failure-derived memories degrade behavior, and offers limited analysis of noise, redundancy, and their accumulation over time.

### Questions
- **Self-Judgment Robustness.** How consistent are the LLM’s success/failure self-labels used for induction? Please analyze misjudged episodes and quantify how such errors propagate to memory quality and downstream performance.  

- **Scalability**: Could you provide more detail or share experiments on how ReasoningBank scales as the number of memory items grows—e.g., in lifelong deployment exceeding thousands or millions of tasks?

- **Failure Memory Impact.** Have you observed cases where failure-derived entries bias or degrade behavior (e.g., overfitting to pathological patterns)? What safeguards (filters, calibration, de-duplication) ensure only constructive failures persist?

### Soundness
3

### Presentation
3

### Contribution
2
