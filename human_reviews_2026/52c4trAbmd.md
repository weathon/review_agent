# AutoTool: Dynamic Tool Selection and Integration for Agentic Reasoning

- Decision: Reject
- Scores: 4, 4, 6

## Abstract
Agentic reinforcement learning has advanced large language models (LLMs) to reason through long chain-of-thought trajectories while interleaving external tool use. Existing approaches assume a fixed inventory of tools, limiting LLM agents' adaptability to new or evolving toolsets. 
We present AutoTool, a framework that equips LLM agents with dynamic tool-selection capabilities throughout their reasoning trajectories.
We first construct a 200k dataset with explicit tool-selection rationales across 1,000+ tools and 100+ tasks spanning mathematics, science, code generation, and multimodal reasoning. Building on this data foundation, AutoTool employs a dual-phase optimization pipeline: (i) supervised and RL-based trajectory stabilization for coherent reasoning, and (ii) KL-regularized Plackett–Luce ranking to refine consistent multi-step tool selection.
Across ten diverse benchmarks, we train two base models, Qwen3-8B and Qwen2.5-VL-7B, with AutoTool. With significantly fewer parameters, AutoTool consistently outperforms advanced LLM agents and tool-integration methods, yielding average gains of 6.4\% in math \& science reasoning, 4.5\% in search-based QA, 7.7\% in code generation, and 6.9\% in multimodal understanding. In addition, AutoTool exhibits stronger generalization by dynamically leveraging unseen tools from evolving toolsets during inference.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces AutoTool, a method designed to evoke dynamic tool selection and integration in LLM reasoning, aiming for development of robust LLM agents under evolving toolsets. AutoTool constructs a 200k agentic reasoning dataset that contains a wide collection of tools and tasks, with tool-selection rationales. Based on that, it develops a dual-phase training scheme for LLM agents, including a SFT + RL phase to stabilize the learning of tool-integrated reasoning trajectories, followed by a KL-regularized Plackett–Luce (PL) ranking phase to refine the tool-selection part of the reasoning. AutoTool achieves performance gains on two LLMs (Qwen3-8B and Qwen2.5-VL-7B) in a diverse set of reasoning tasks, outperforming other advanced
LLM agents and baseline tool-integration methods.

### Strengths
- AutoTool innovatively integrates embedding-anchored tool selection and KL-regularized PL ranking into the learning of LLM agents, which contributes to decent originality.
- The presentation of AutoTool dual-phase learning scheme is theoretically well-motivated and mathematically well-grounded.
- AutoTool’s proposed challenge of dynamic tool selection under evolving tool environments is crucial for robust and scalable LLM agentic framework development.

### Weaknesses
- The experimental analysis of this paper falls short of justifying AutoTool’s effectiveness on improving dynamic tool selection under evolving tool environments, i.e., whether AutoTool performs better tool selection when generalizing to unseen toolsets, which is however the most significant challenge raised by the paper. Evaluation on a new or heldout set of tools and tasks that are unseen at training phase would help further justify this important point.
- It is unclear how the evolving toolset T with dynamic size is constructed for AutoTool learning, i.e., for each training sample or question, which candidate tools are chosen to form the evolving toolset and how to control a decent tool-selection difficulty regarding to the number or proportion of useful and irrelevant tools in the toolset. The design of evolving toolset T at training phase is crucial for learning the dynamic tool selection.
- There is no quantitative analysis to verify the positive correlation between the tool selection accuracy and the final answer accuracy. It would be better to more directly justify that, compared to baseline methods, AutoTool has a better hit rate of selecting the correct oracle tool, and this contributes to its better final performances.

### Questions
- Any additional experimental results to resolve the above weaknesses?
- Is there an ablation study to measure how many performance gains are due to the incorporation of additional tool-selection rationales introduced in AutoTool?
- Are there any qualitative or quantitative comparisons with regard to the scope of toolsets studied in AutoTool and in other related work of tool integration, such as ToolLLM, RestGPT and HuggingGPT?

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
4

### Summary
This paper introduces a dynamic selection method for tools in agentic large language models. Typical agentic models assume a fixed set of tools to use in their reasoning process. This paper introduces a dynamic selection process where the model can utilize a large set of tools through retrieval. The proposed approach can also handle new tools which can be added to the tool repository for the models to use. Empirical results show the dynamic tool selection method outperforms existing tool-integration methods and can generalize well to unseen tools during inference.

### Strengths
- Comprehensive empirical results, spanning a diverse set of evaluation datasets
- Results compared against relevant baselines such as stronger reasoning models, existing tool integration methods and traditional fine-tuning
- Strong results, the proposed AutoTool framework achieves consistent gains on the diverse datasets compared to multiple approaches.

### Weaknesses
- I couldn't find the results on the generalization performance on unseen tools during inference. The key proposal for the embedding-anchored selection method is that it should be able to dynamically adapt to new tools provided during inference, but none of the experimental results seem to highlight it.
- Not sure I follow why the analysis of autotool is needed with an oracle tool assignment agent. Ideally, the oracle numbers should be present in Table 1 to directly compare other methods on how close they too are with the oracle assignment, if its necessary.

While overall the paper and contribution is good, its missing this key ingredient (generalization) - I'm ready to raise my scores if it is presented and analyzed comprehensively.

### Questions
Same as my weakness - where is the generalization result? I think that should be the key result to highlight, along with analysis where the generalization works and fails.

### Soundness
3

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
3

### Summary
This paper presents AutoTool, a framework that equips LLM agents with dynamic tool-selection capabilities throughout their reasoning trajectories. The authors construct a 200k dataset with explicit tool-selection rationales across 1,000+ tools and 100+ tasks, then employ a dual-phase optimization pipeline: (i) trajectory stabilization via SFT and RL, and (ii) KL-regularized Plackett-Luce ranking for tool selection refinement. Experiments show consistent improvements across math, science, code generation, and multimodal benchmarks.

### Strengths
The paper addresses a genuine limitation in existing work—most approaches assume fixed toolsets, whereas real-world scenarios require dynamic tool selection from evolving inventories.

The dual-phase optimization pipeline is well-designed, with Phase I establishing stable reasoning patterns and Phase II specifically targeting tool-selection refinement through PL ranking.

### Weaknesses
While the combination is effective, the individual components (SFT, GRPO, Plackett-Luce ranking) are well-established techniques. The main contribution appears to be applying PL ranking to tool selection, which is somewhat incremental. The paper would benefit from discussing recent work on tool retrieval and generation. 
Also there are notation inconsistencies: The paper switches between τ and T for trajectories/trajectory sets.

### Questions
I do have some scalability concerns. How does the approach scale beyond 1,000 tools? The embedding-based selection (Eq. 4) requires computing distances to all tools at each selection step.

### Soundness
3

### Presentation
2

### Contribution
3
