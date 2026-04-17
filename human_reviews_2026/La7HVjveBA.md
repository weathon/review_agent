# DeepHA: Scaling Action Chains Elicits Deep Hierarchical Agents

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 2, 4, 2

## Abstract
Prevailing autonomous agents are often constrained by a single, predefined action space, which limits their generalization capabilities across diverse tasks and can introduce compounding errors through decoupled policy execution. To address these limitations, we introduce the Deep Hierarchical Agent (DeepHA), a unified architecture that operates across a mixture of heterogeneous action spaces, flexibly generating actions ranging from high-level semantic skills to low-level motor controls. We further propose a Chain-of-Action (CoA) reasoning framework, which enables the agent to use higher-level abstract actions as structured `thoughts' to guide the generation of more granular, subsequent actions. To manage the computational demands of this deep reasoning in long-horizon tasks, we develop a memory-efficient mechanism that dynamically compresses historical context and leverages Key-Value (KV) caching, reducing context length by approximately 75\% without sacrificing performance. We conduct extensive evaluations on a new, large-scale benchmark of over 800 diverse Minecraft tasks. Results show that DHA significantly outperforms prior methods, establishing a new state-of-the-art and demonstrating superior generalization, particularly in complex, multi-step planning tasks. Our work presents a novel, unified framework for building more capable and efficient autonomous agents.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors propose DeepHA, a hierarchical agent architecture enabling dynamic action generation across multiple abstraction levels (skills, grounding, motion, raw actions) via Chain-of-Action reasoning. They introduce a memory-efficient mechanism that reduces context length by 75% through dynamic history compression while preserving high-level semantic goals for long-horizon tasks.

### Strengths
Originality
- The paper presents a novel and insightful approach. The memory compression of past intermediate steps, which offers a clever mechanism for designing scalable and efficient Vision-Language-Action (VLA) architectures.

Quality & Clarity
- The formulation is comprehensive and well-articulated, with clear definitions of key concepts such as action levels, inference modes, and policy mixtures.
- The experiments effectively validate the proposed method and concepts.
- The illustrations and the experiment details (including case studies and configurations in the Appendix) clearly demonstrate the concepts and outputs across action levels.

Significance
- The results are promising. The proposed approach outperforms other SOTA approaches, including instruction-conditioned policies and hierarchical agents. They also introduce a new metric ASR to show the competence of the approach.

### Weaknesses
- While the paper includes comprehensive ablation studies, it lacks an analysis of failure modes. I encourage the authors to include case studies and detailed examinations of failures across inference modes, action levels, and termination mechanisms. Such analyses would offer deeper insights into the rationale behind the chosen designs and clarify whether these components are complementary and essential to the overall approach.
- The paper lacks implementation details, such as the actual prompts for action generation and the implementation code. The authors should provide this information for reproduction.

### Questions
- I wonder if the memory efficiency could also be supported by theoretical analysis. Beyond empirical results, deriving a theoretical lower bound on memory usage would provide stronger insights into the scalability and potential applications of the proposed approach.
- Have you applied the bottom-up approach in your framework? You mention the approach in Section 2.2, while there is no other discussion in this paper. How would it be used?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes the Deep Hierarchical Agent (DeepHA), an agent architecture for complex, open-world environments like Minecraft. The authors identify two main limitations in prior work: reliance on a single, predefined action space and errors from decoupled high-level and low-level policies. To address this, DeepHA introduces a manually-defined, multi-level action hierarchy (e.g., Skill, Grounding, Motion, Raw) and a Mixture-of-Policies (MoP) framework where a central VLM generates an action at a chosen level of abstraction, which is then routed to a specialized low-level policy. The paper also proposes a "Chain-of-Action" (CoA) reasoning framework, where the VLM autoregressively generates a sequence of actions from high-to-low abstraction (e.g., $Skill \rightarrow Grounding \rightarrow Motion$), using the higher-level actions as "thoughts" to guide the lower-level ones. Finally, to handle long context lengths, the authors describe a "memory-efficient mechanism" that prunes historical tokens and uses KV caching. The agent is evaluated on a large, proprietary benchmark of over 800 Minecraft tasks, where it is shown to outperform previous methods.

### Strengths
1.  **Strong Empirical Results:** The paper's primary strength is its extensive dataset curation and empirical validation. The authors have tested a large-scale benchmark and performed a detailed evaluation, demonstrating state-of-the-art performance within their chosen domain.
2.  **Clear Ablation Studies:** The ablation study provides clear, quantitative evidence that, within their hand-crafted framework, deeper hierarchical reasoning leads to better performance than shallow reasoning. This validates their central design choice.
3.  **Detailed System Documentation:** The paper and its appendix are very transparent about the complex, multi-stage training pipeline and the extensive data curation process.

### Weaknesses
1.  **Critical Flaw in Memory Contribution:** The "memory-efficient mechanism" is the most significant weakness. The paper claims to "manage the computational demands... in long-horizon tasks", but the proposed method is explicitly described as an "**inference-time process**". It offers **no solution** for the memory bottleneck during **training**, which is the main pain point for long-sequence models. The method is a simple application of token pruning + standard KV caching for generation. This is not a novel contribution and does not solve the problem it claims to.
2.  **Lack of Algorithmic Novelty (Action Space):** The paper's core, the action hierarchy, is entirely hand-crafted. While the *concept* of hierarchy is general, the *implementation* is a fixed, domain-specific engineering choice. The work offers no generalizable method for *learning* this hierarchy, which severely limits its scientific contribution beyond the specific domain of Minecraft.
3.  **Conflation of Data-Scaling with Algorithmic Novelty:** The SOTA results are impressive but appear to be the product of a massive, multi-stage data engineering effort and finetuning on data from powerful proprietary models. This is a great engineering result, but it's unclear how much of the gain comes from the *method* versus this extensive, domain-specific data advantage.
4.  **Reproducibility:** The reliance on proprietary programs / pipelines for generating the foundational dataset makes a key component of the work impossible to reproduce.

### Questions
My questions are stated as above. Overall I do acknowledge the empirical result of this work. But the paper is a simple composition of many established / common knowledge techniques and the novelty of this work is limited. I think this paper's empirical success worth to share on data mining conferences but not ICLR.

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
*Motivation*

Previous methods are confined to a predefined action space, limiting their applicability to a diverse range of tasks. For e.g., "an action space that excels at navigation may be ill-suited for precise object manipulation”.

*Proposal*

The authors address this by proposing a mixture of action spaces, where each action space defines actions of a specific hierarchical level (semantic or temporal).

The authors fix the level abstractions as: “high-level language skills, coordinate-based grounding actions, mid-level motion commands, and low-level raw action sequences”.

To allow the action prediction to follow the pattern as, high to lower level actions, the authors propose “long action-chain reasoning through action pyramid”.

Secondly, as the memory context can become intractable easily, the authors also propose an efficient memory management scheme that trims the past low-level actions from memory.

The authors test the agent in an expanded version of the minecraft dataset (800 tasks) from OpenHA. The method performs significantly better than the previous SOTA methods.

### Strengths
- Good integration of multiple modules to achieve a high performance.

### Weaknesses
- Not completely clear on the full details of the construction of abstract actions. While section 2.2 discusses possible methods for automatically extracting abstract actions from datasets, the method does not use them. The only section that provides usable insights into the action abstractions used is B.2. The paper does not mention how the dataset was created, manual attribution, or automatic generation. Other small details, like the total number of instances in each abstraction, are also missing.
- While the proposed abstractions improve performance (at the MineBlock task), it has not been discussed why this specific hierarchy was chosen. As the abstraction hierarchy is listed as a core contribution, it should be addressed.
- A lot of data has been curated for each training phase: world knowledge QA, VQA, scene captioning, Visual grounding, Reasoning capabilities enhancement, abstracted action data, and chain of action structured data. This makes it difficult to judge the impact and transferability of the method to other tasks (such as robotics).
- Not really end-to-end as there are multiple training phases, each with its own objectives and target weights subset. Only the finetuning phases are end-to-end.
- While a lot of effort has been put into getting the architecture working better than SOTA, including data curation and training. The only novel contributions I see are the constructed action abstractions, but they may not be valid across domains. The other contribution of memory pruning only provides modest improvements over full memory (+1.1% ASR and -1.2% FT).
- Ablation 4.3.3 shows that the greedy mode, which unrolls the full hierarchy, performs worse than the eager mode, which can break the hierarchy, which seems to invalidate the hypothesis that action hierarchies are crucial to improving performance. An analysis of how a greedy strategy can fail, or of how eager mode chooses an exit, should be provided.
- The paper is slightly confusing to read, and I have to hunt for information. There are sections in the main paper that are not very relevant, while those in the appendix are required for a core understanding of the method. The appendix is currently almost mandatory to understand the method clearly. I believe a rewrite can significantly enhance readability.
- Small typo in line 725

### Questions
- How are the action abstraction policies trained?
- Does ablation 4.3.1 use greedy mode?
- What kind of memory does ablation 4.3.3 use? How is the memory context of direct mode disproportionately highest while it produces the fewest tokens?
- I understood eager mode as model choosing the exit, so what does Eager-Motion/Grounding mean in ablations 4.3.3?
- In section A.2, the inspirations for the abstractions are mentioned. For grounding-based policies, it is noted that they are “adept at interpreting coordinate-based instructions or visual targets and translating them into navigational actions”. However, the grounding policy in the paper’s case translates high-level goals into raw actions. Similarly, don’t all policies output in the raw action space (Fig. 3). Can the author’s clarify this?
- How does the model learn to output the eager-stop token? Is it present in the data? If so, how is the early exit decided when creating the data?
- The way eager mode allows for an exit is fundamentally different from how hierarchical agents act. In the proposed architecture, it seems all abstractions exist side-by-side, allowing the router to choose from them rather than stacking them vertically.

### Soundness
2

### Presentation
2

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
The paper proposes Deep Hierarchical Agent (DeepHA), a single, unified architecture for operating in heteregenous actions spaces. The authors propose the Chain-of-Action (CoA) framework that enabels the agent to generate higher-level actions as thoughts to guide the generation of finer-grained actions. The primary motivation is the observation that for domains such as minecraft, complex tasks naturally decompose into a hierarchy of actions. 

To deal with hetereogenous action spaces, the action use a single high-level VLM along with a  low=level mixture of policies. The VLM is fine-tuned to generate actions from a specific action space which a router routes to the appropriate policy. The VLM can be etrained in different ways
In direct mode, the vlm generates an abstract action from a signle space. The router simply routes it.
In Greedy mode, the vlm sequentially generates actions from the action hierarchy ultimately resulting in a low-level action. Each action in the sequence serves as a thought to improve the generation of the ext action. 

In eager mode, execution is halted once an executable action is produced. This is either by detecting a sepcial tag for manual experiments or allowing the vlm to autonomously learn it by fine-tuning to produce the tag as its generation. 

The authors also propose a memory-efficient chain-of-action by compressing parts of the execution history. The experiments are conducted on Minecraft with several baselines and 800 total tasks.

### Strengths
1. The overall idea seems intuitive.

2. The compression feature is interesting.

3. Results look promising although there are some concerns here.

### Weaknesses
1. I dont think this paper is written well. For the most part it is okay but there are a couple of major issues that might cause confusion and as a result misunderstanding of your work

1a. Where is this Chain-of-Action architecture defined? It is proposed as a new contribution but Im not able to find it detailed anywhere? As a result, I cannot even imagine how to apply this to other tasks.

1b. There is no contrast with related work. Lines 104-107 and 131-133 seem like a citation dump with no contrast on how your approach is different. This makes it very hard to understand what value your work provides since these are hierarchical agents as well several of which you use as baselines.

2. The approach seems very hand-coded. For example, the authors state a complex task is naturally decomposed to simpler tasks. Their CoA framework in greedy mode generates a very specific sequence of actions that help minecraft (A^s -> A^g -> A^m -> a). Who determines this order? Does this also work for other tasks that are not minecraft? How to scale your approach to other domains automatically.

I think currently this needs clarificaiton and seems like a major limitation on the generalizability of this framework. Please clarify if the training data needs to be formulated as such for training. If it is indeed the case, then i think you owuld need to show more than just Minecraft to demonstrate that this approach works for different types of hieracchies. You mention that such action pyramids can be learned in lines 161-185 but this is not clarified or explained in detail. 

3.  For Table 1, DeepHA, what mode was used? was it eager mode? if it was, then was it using manually terminated tags or a vlm fine-tuned to generate the tag that allows routing to an executable action using a pretrained policy?

4. Lines 377-378 say that the baselines are trained on the same expert datasets. Could you elaborate on how the dataset was processed for your appraoch since the action pyramid owuld need to be learned I think? As a result, you would need to annotate the dataset in a different way?

5. Why are the standard deviations so high in some tasks? eg. Mine Blocks and Kill entities. Is this just one run of 800 tasks with the avg and sd computed across the 800. Or is it the sd of the ASR across X runs?

### Questions
Ive asked questions in weaknesses itsefl. Overall interesting work but needs more clarifications before i can accept it. Happy to engage in discussions and increase my score.

### Soundness
3

### Presentation
2

### Contribution
2
