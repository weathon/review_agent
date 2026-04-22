# FaSTA*: Fast-Slow Toolpath Agent with Subroutine Mining for Efficient Multi-turn Image Editing

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 8, 6, 4

## Abstract
We develop a cost-efficient neurosymbolic agent to address challenging multi-turn image editing tasks such as "Detect the bench in the image while recoloring it to pink. Also, remove the cat for a clearer view and recolor the wall to yellow." It combines the fast, high-level subtask planning by large language models (LLMs) with the slow, accurate, tool-use, and local A* search per subtask to find a cost-efficient toolpath---a sequence of calls to AI tools. To save the cost of A* on similar subtasks, we perform inductive reasoning on previously successful toolpaths via LLMs to continuously extract/refine frequently used subroutines and reuse them as new tools for future tasks in an adaptive fast-slow planning, where the higher-level subroutines are explored first, and only when they fail, the low-level A* search is activated. The reusable symbolic subroutines considerably save exploration cost on the same types of subtasks applied to similar images, yielding a human-like fast-slow toolpath agent ``FaSTA*'': fast subtask planning followed by rule-based subroutine selection per subtask is attempted by LLMs at first, which is expected to cover most tasks, while slow A* search is only triggered for novel and challenging subtasks. By comparing with recent image editing approaches, we demonstrate FaSTA* is significantly more computationally efficient while remaining competitive with the state-of-the-art baseline in terms of success rate.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces an AI agent called FaSTA* designed to handle multi-step image editing tasks based on a previous work "CoSTA*". The agent works by combining an LLM, which quickly plans the high-level subtasks, with a more detailed A* search that finds the best sequence of AI tools to accomplish each individual step. The key idea is that the system learns from its successful past actions. It uses an LLM to identify common sequences of tools that work well and saves them as reusable subroutines. This creates a "fast-slow" system where the agent first tries to solve a new task quickly by using these pre-saved subroutines. If these shortcuts don't work for a novel or difficult task, the agent then switches to the slower, more methodical A* search to find a custom solution. The authors show that this approach makes FaSTA* significantly more computationally efficient while maintaining a success rate that is competitive with CoSTA*.

### Strengths
1. Despite intuitive, the proposed "fast-slow" strategy is a successful attempt to save inference time.
2. The writing and figures ease the understanding of the motivation and method.
3. The editing results are impressive, and the coverage of editing types is comprehensive.

### Weaknesses
1. Limited Technical Contribution. The paper's technical contributions to both the neurosymbolic and agent planning fields appear limited. From the neurosymbolic aspect, the mechanism for learning symbolic "rules" is a straightforward process of summarizing and retrieving successful execution paths. While functional, this approach offers limited insights for rule induction and program synthesis explored in the broader neurosymbolic literature. From the agent planning perspective, the core innovation can be viewed as augmenting the A* search with a retrieval-augmented generation (RAG) step to recall past solutions. While this is a practical engineering choice, it offers limited new insight into the fundamental challenges of agentic planning. I also suggest the authors conduct a more comprehensive review of agents' learning capability and memory, which is very relevant to this work but was unfortunately missed.

2. Potentially Inequitable Experimental Comparison. The experimental comparisons may not provide a fair assessment. The baseline methods were not specifically optimized for the custom multi-turn editing benchmark introduced in CoSTA∗, potentially placing them at a significant disadvantage. A more informative comparison would involve a well-engineered ReAct agent equipped with the exact same toolset and effective prompt engineering. Such a baseline would help isolate whether the observed performance gains stem truly from the proposed fast-slow planning mechanism or from factors common to most modern agentic frameworks. The core logic of CoSTA*, FaSTA*, and a ReAct agent is fundamentally similar.

3. Questionable Problem Framing and Motivation. The paper frames its primary contribution around optimizing for "cost" in multi-turn image editing. However, it is questionable whether inference time is the most critical bottleneck that needs addressing in this domain. Furthermore, as an incremental work building upon the CoSTA* framework, the added contribution, i.e., the learning and reuse of subroutines, feels like a modest step forward rather than a significant leap. The novelty seems insufficient for a standalone publication at ICLR.

4. Minor Point on Formatting. As a minor note, the excessive use of vspace throughout the paper negatively impacts readability. Not sure if this should be desk rejected.

### Questions
See weaknesses.

### Soundness
2

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
3

### Summary
This paper introduces FaSTA, a neurosymbolic agent designed to significantly improve the computational efficiency of complex, multi-turn image editing. The work addresses the key bottleneck of its predecessors like e.g CoSTA, which rely on computationally expensive A* search for every task without learning from past experiences.

FaSTA*'s core innovation is a novel planning framework that automatically analyzes successful execution traces from previous tasks, using LLMs for inductive reasoning to extract frequently used sequences of tool calls as reusable "subroutines." These subroutines are generalized into symbolic rules with activation conditions (e.g., object size, background complexity).

During execution, FaSTA* first generates a cost-effective "Fast Plan" by having an LLM select appropriate learned subroutines for each subtask. When a subroutine is unavailable or fails a quality check, it triggers a localized A* search for that specific subtask. This adaptive strategy mimics human-like reasoning, prioritizing efficient, learned shortcuts while retaining robust search for novel challenges.

The results demonstrate that FaSTA* achieves a paradigm shift in cost-efficiency. It reduces average execution cost compared to CoSTA* at the price of a slightly reduced output quality. By transitioning from a test-time-only planner to a continuously learning agent, FaSTA* offers a more scalable and practical solution for complex, tool-based image editing, effectively combining the speed of LLM planning with the guaranteed performance of heuristic search.

### Strengths
The main strengths of the paper lie in:
-  The proposed approach exhibits a good efficiency at the price of a small reduction of the quality of the output.

- The "fast-slow" planning paradigm is intuitive and powerful, effectively mimicking human problem-solving.

-  Moves beyond one-off planning to a continuously learning system, turning past costly searches into future time savings.

- Comprehensive experiments and ablations validation of the approach, clearly demonstrating the contribution of each component.

- Provides a clear path towards more affordable and scalable AI tool-use agents for complex tasks.

### Weaknesses
The main weaknesses of this work are the following:
- The performances are suboptimal initially, requiring a "warm-up" period of task execution before the subroutine library becomes effective.

- The entire framework hinges on capable (and often expensive) LLMs for planning, subroutine selection, and induction, creating a dependency and potential cost center.

- Builds upon the already complex CoSTA* framework, making the overall system sophisticated and potentially challenging to reproduce, although the code is avaiable at least for reviewing purposes.

- The success of the "Fast Plan" relies on the generality of the learned symbolic rules, which may not capture all edge cases or nuanced visual contexts, and if the learning is bad, the results are properly of bad quality.

- The paper is too short and not self contained. Indeed, without the additional material it is very hard to understand and assess the details. From reading the paper it is also difficult to reproduce the results. For reviewing purposes the code is available, and we recommend to make t availabe if the paper will be accepted.

### Questions
1. The paper states performance improves with more tasks, but could you provide a concrete curve or table showing the relationship between the number of "warm-up" tasks and the resulting cost savings/success rate? How many tasks are required to reach, for example, 80% of the maximum efficiency gain?

2. Have you explored any strategies to mitigate the cold-start problem, such as pre-populating the subroutine library with a set of manually defined or "common-sense" rules from a small seed dataset to bootstrap the learning process?

3. While execution cost (time) is reported, could you provide an analysis of the total inference cost, including the API calls to the LLMs (GPT-4o, GPT-o1) used for planning, subroutine selection, and induction? Does the 49% savings in execution time still hold when these substantial LLM costs are factored in?

4. The system relies heavily on powerful, expensive LLMs (GPT-o1 for induction). Were any ablations performed using smaller, open-source LLMs for any of the components (e.g., subroutine selection) to assess the trade-off between performance and cost/dependency?

5. The paper builds upon the complex CoSTA* framework. Beyond code availability, what are the minimum system requirements (e.g., GPU memory) and estimated engineering effort to set up and run the full FaSTA* pipeline? Is there a containerized environment or a more detailed setup guide to facilitate replication?

6. How reliant is FaSTA* on the specific, pre-defined structures of CoSTA* (the Tool Dependency Graph, Model Description Table, Benchmark Table)? How much manual effort is required to adapt FaSTA* to a new set of AI tools not in the original CoSTA* setup?

7. The paper mentions a verification step for new subroutines, but what happens when a "bad" rule is nonetheless learned and passes verification? Could you provide an example of such a case and its impact on subsequent task performance? How often does the inductive reasoning propose rules that are later found to be flawed?

8. The learned rules seem effective for tasks similar to those in the training distribution. How does FaSTA* perform on a task that requires a genuinely novel combination of tools or operates on image types (e.g., medical imagery, technical diagrams) far outside its experience? Does the fallback to A* search reliably recover quality in these OoD scenarios?

9. Key details for understanding the methodology—such as the full structure of a trace, the exact prompts for inductive reasoning, and the complete Subroutine Rule Table—are in the appendices. Would you consider moving the most critical elements (e.g., the rule table structure, a trace example) to the main paper to make it self-contained?

10. The reviewers note the code is available for review. We strongly recommend you explicitly state in the paper the commitment to make the code and model weights (if applicable) publicly available upon acceptance, to ensure the community can build upon this work.

### Soundness
3

### Presentation
2

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
This paper proposes FaSTA* (Fast-Slow Toolpath Agent), an efficient neuro-symbolic agent designed for multi-turn image editing tasks. FaSTA* combines the fast, high-level subtask planning capabilities of large language models (LLMs) with slow, precise tool usage and local A* search.

### Strengths
The agent effectively integrates the symbolic reasoning capabilities of large language models (for subroutine mining and high-level planning) with the precision of localized A* search (for low-level tool path optimization), resulting in a robust and efficient system.

### Weaknesses
The extraction of subroutines relies on the inductive reasoning capabilities of LLMs, and is represented in the form of symbolic rules (for example, conditions based on object area and mask ratio). The generalization ability and robustness of these rules may be challenged when faced with changes in the toolset or with complex and ambiguous editing tasks.

### Questions
Please explain in detail how an LLM performs inductive reasoning to extract symbolic rules, particularly how it determines the quantification conditions in the rules (such as object_area ≤ θ). How are these threshold values determined?

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
The paper proposes FaSTA*, a Fast-Slow Toolpath Agent that builds upon CoSTA* (Cost-Sensitive Toolpath Agent) to enhance efficiency in multi-turn image editing tasks. The core idea is to mine frequently used subroutines (i.e., reusable sequences of tool calls) from prior editing experiences using large language model (LLM)-based inductive reasoning. These symbolic subroutines are stored and reused for similar future tasks, allowing the system to perform a “fast plan” by recalling suitable subroutines, with fallback to a slower A*-based search (“slow plan”) if no appropriate rule exists or the quality check fails. Experiments show that FaSTA* reduces computational cost by roughly 49% compared to CoSTA* while maintaining comparable quality.

### Strengths
- The idea of leveraging reusable subroutines for planning efficiency is a logical extension of CoSTA*. It introduces an experience-based learning mechanism that mimics human procedural reuse and contributes to more scalable agentic workflows.
- The fast-slow execution strategy and subroutine mining pipeline are well described, supported by detailed diagrams and ablations. The authors conduct a careful comparison with CoSTA* and show meaningful runtime reductions.
- Representing toolpath knowledge as symbolic rules rather than black-box model weights improves transparency and offers a potential bridge between LLM-based reasoning and symbolic planning.

### Weaknesses
- Despite its engineering refinement, FaSTA* is largely an incremental enhancement of CoSTA*. The “subroutine mining” mechanism mainly involves clustering or pattern extraction of successful tool sequences using an LLM prompt, without introducing new algorithmic insights or theoretical guarantees. The resulting “fast-slow” hybrid is conceptually intuitive but not methodologically groundbreaking.
- The reported efficiency gains (≈49%) are based on the same dataset and evaluation setting as CoSTA, limiting generalizability.  The comparison to baselines (MagicBrush, GenArtist, CLOVA) is inherited from CoSTA*, with no new external benchmark validation. Human evaluation of image quality lacks standardized metrics or reproducibility details, and visual examples (Fig. 4–8) are mostly qualitative.
- The “subroutine mining” uses CoSTA’s outputs for initialization, meaning FaSTA* effectively benefits from pre-computed search traces, inflating its efficiency advantage. The cold-start performance (on unseen domains or without prior CoSTA* traces) is underexplored and acknowledged as a limitation but not quantified.
- The inductive reasoning step is described as “in-context reinforcement learning,” but in practice it is a rule-extraction procedure with periodic prompt-based updates—no reward modeling or policy improvement is performed. This weakens the conceptual link to ICRL and exaggerates the claimed neurosymbolic contribution.

### Questions
NA

### Soundness
2

### Presentation
3

### Contribution
2
