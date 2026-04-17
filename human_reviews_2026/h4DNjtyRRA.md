# ReCode: Unify Plan and Action for Universal Granularity Control

- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
Real-world tasks require decisions at varying granularities, and humans excel at this by leveraging a unified cognitive representation where planning is fundamentally understood as a high-level form of action.
However, current Large Language Model (LLM)-based agents lack this crucial capability to operate fluidly across decision granularities.
This limitation stems from existing paradigms that enforce a rigid separation between high-level planning and low-level action, which impairs dynamic adaptability and limits generalization.
We propose **ReCode** (**Re**cursive **Code** Generation), a novel paradigm that addresses this limitation by unifying planning and action within a single code representation.
In this representation, ReCode treats high-level plans as abstract placeholder functions, which the agent then recursively decomposes into finer-grained sub-functions until reaching primitive actions.
This recursive approach dissolves the rigid boundary between plan and action, enabling the agent to dynamically control its decision granularity.
Furthermore, the recursive structure inherently generates rich, multi-granularity training data, enabling models to learn hierarchical decision-making processes.
Extensive experiments show ReCode significantly surpasses advanced baselines in inference performance and demonstrates exceptional data efficiency in training, validating our core insight that unifying planning and action through recursive code generation is a powerful and effective approach to achieving universal granularity control.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes ReCode, a recursive code generation paradigm for LLM-based agents that unifies planning and action within a single executable code framework. The key idea is to treat planning as high-level action, **recursively** decomposing abstract placeholder functions into executable API calls until primitive actions are reached. This approach aims to enable dynamic granularity control, eliminate the need for predefined action spaces, and generate rich hierarchical training data. The authors evaluate ReCode across three text-based environments (ALFWorld, WebShop, ScienceWorld) and demonstrate improvements in inference performance, generalization, and training efficiency compared to strong baselines like ReAct and CodeAct.

### Strengths
1. This approach of initially employing placeholders for complex actions and iteratively refining them into atomic actions or more detailed plans is insightful.
2. Both plans and actions are represented as Python function calls (placeholders, atomic), which are generated and executed by a single policy $\pi$, thereby eliminating the conventional separation between planning and execution. This is conceptually elegant and practically implementable.
3. Figure 2 offers an intuitive illustration. Once you see it, you pretty much get how ReCode works.
4. The Cost Analysis effectively demonstrates the cost-efficient nature of the ReCode framework.

### Weaknesses
1. About the writing, it would be better to just clearly list out the specific contributions in the Introduction.
2. Algorithm 1 mentions using a HeuristicConvert function to turn task instruction into a root. I feel like this part isn't really explained—it's not clear how it actually works.
3. All three are text-world benchmarks. This seems a bit limited. I'm not totally convinced it would work in more complex settings. I'd like to see how it performs on more diverse and harder benchmarks, like EXP-Bench [1], or RExBench [2].

[1] Kon, P. T. J., Liu, J., Zhu, X., Ding, Q., Peng, J., Xing, J., ... & Chen, A. (2025). EXP-Bench: Can AI Conduct AI Research Experiments?. arXiv preprint arXiv:2505.24785.

[2] Edwards, N., Lee, Y., Mao, Y. A., Qin, Y., Schuster, S., & Kim, N. (2025). RExBench: Can coding agents autonomously implement AI research extensions?. arXiv preprint arXiv:2506.22598.

### Questions
1. When a placeholder expansion fails (invalid code, violated state assumptions), is there any recovering or replanning mechanism? Can you quantify the recovery rate and its extra cost?
2. About the HeuristicConvert part, could you share some insights into how it was designed? It would be great to see an ablation study testing if different heuristics would change the results.
3. Can you conduct experiments on EXP-bench, or RExBench? 
4. For Table 2, could you also add the results for AdaPlanner and ADaPT using models like Gemini 2.5 Flash and DeepSeek-V3.1 on these three environments?

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
4

### Summary
This paper introduces ReCode, a novel paradigm for LLM-based agents that unifies planning and action through recursive code generation. The key insight is that planning and action are not fundamentally different but represent decisions at different levels of abstraction. ReCode represents both plans and actions as executable code, with high-level plans as placeholder functions that are recursively refined into primitive executable actions. The authors demonstrate that this approach achieves superior performance across three benchmark environments (ALFWorld, ScienceWorld, WebShop) with over 20.9% improvement in inference and remarkable training efficiency.

### Strengths
- The paper is well-motivated and easy to read
- ReCode achieves remarkable cost efficiency and training efficience. This is a significant practical advantage.

### Weaknesses
- While the three environments are diverse, they are all text-based simulation environments. The approach needs validation on more complex, real-world tasks or environments with continuous action spaces. According to the paper's content, it seems possible to have primitive actions at much lower levels of API (for example, continuous actions like moving forward 3.5m). Experiments on this aspect are needed.
- The paper doesn't discuss how the system handles errors in code generation or execution failures. If errors occur in some parts of the code, it's questionable whether the task can be performed normally. If not, can this methodology effectively respond when LLM performance decreases?

### Questions
- In Section 4.2, we can see that AdaPlanner has the second-best performance. Can we see the results of testing AdaPlanner based on Gemini-Flash or DeepSeek?
- What is the maximum recursion depth observed in practice, and how does performance degrade with increasing task complexity?
- Can you provide analysis on failure modes? When does the recursive expansion fail to converge to executable actions?

### Soundness
3

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
This work proposes ReCode, a single-policy, recursive “code as plan and action” framework that refines unimplemented placeholders into executable steps, enabling flexible control over reasoning granularity. Across multiple simulated environments, it outperforms code-reasoning baselines and its hierarchical trajectories further strengthen supervised fine-tuning; however, the novelty over prior recursive code generation works (e.g., REPL-Plan) remains unclear without head-to-head comparisons.

### Strengths
*  **Simple, unified mechanism.** Both plans  and actions are written as code, and the system only recurses when something isn’t executable, easy to reason about and implement. 
* **Gains at lower cost.** Across benchmarks, the method outperforms baselines at lower cost.

### Weaknesses
* The novelty over prior similar recursive code work is unclear; it could be better to add a brief comparison to previous works(e.g., REPL-Plan, Code-as-Policies) to make the contribution explicit.
*  No granulity measurement and analysis.

### Questions
* Could you share distributions of recursion depth and the number of expanded placeholders per episode? 

* Could you add a brief, explicit paragraph contrasting ReCode with other recursive code generation papers, i.e., REPL-Plan and Code-as-Policies? A small comparison table would make the distinctions especially clear.

### Soundness
3

### Presentation
3

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
This paper presents ReCode, a framework that represents an policy as a recursive code generator, unifying planning and action within a single paradigm. High-level tasks are expressed as abstract placeholder functions that the LLM recursively expands into finer-grained subfunctions and primitive actions, enabling dynamic control over decision granularity.

### Strengths
ReCode unifies high-level planning and low-level action within a single code-based framework, allowing LLM agents to dynamically adjust decision granularity.

### Weaknesses
1. The proposed approach is closely related to recent advances in code-as-policies paradigms that leverage code-generation LLMs (e.g., [1–5]). However, the paper does not sufficiently analyze or position ReCode against these methods in the related work section.

[1] Code as Policies: Language Model Programs for Embodied Control. ICRA 2023.

[2] RoboCodex: Multimodal Code Generation for Robotic Behavior Synthesis. ICML 2024.

[3] PoAct: Policy and Action Dual-Control Agent for Generalized Applications. arXiv  2025.

[4] Executable Code Actions Elicit Better LLM Agents. ICML 2024

[5] Towards Reliable Code-as-Policies: A Neuro-Symbolic Framework for Embodied Task Planning, arxiv 2025

2. The recursive expansion and execution of functions or subtasks in ReCode resemble hierarchical code-as-policies frameworks previously explored in works [6, 7]. Since these studies also pursue recursive, code-centric planning structures, the paper needs to articulate more clearly what distinguishes ReCode’s recursive granularity control from prior hierarchical code-generation methods. Moreover, given the conceptual similarity, it would be valuable to include empirical comparisons with particularly [7], one unifying reasoning and acting through executable code with adaptive feedback during execution. Technical novelties should be clearly specified. 

[6] Demo2Code: From Summarizing Demonstrations to Synthesizing Code via Extended Chain-of-Thought. NeurIPS 2023.

[7] Interactive and Expressive Code-Augmented Planning with Large Language Models.  arXiv 2024.

3. While the paper claims that flexible control of decision granularity leads to superior adaptability and efficiency, the current experiment results do not isolate granularity as a directly measurable factor. To substantiate this claim, ablation experiments varying the recursive depth could demonstrate how granularity impacts reward, cost, and performance–efficiency trade-offs.

4. Beyond qualitative case studies, the paper would present quantitative analyses that reveal how granularity dynamically changes during execution. Statistics such as the average depth of generated decision trees, branching factors, and the ratio of placeholder functions to primitive actions across different tasks would provide stronger empirical support for the proposed mechanism.

5. While ReCode emphasizes the unification of hierarchical decision-making within a single code-based framework, the chosen benchmarks and baselines primarily focus on high-level planning tasks. These environments do not inherently require dynamic adjustment of decision granularity, as most decisions occur at task-planning level. For instance, the ALFWorld case study in Appendix Figure 3 represents a classic high-level planning scenario where expressing actions as executable code offers limited additional benefit over traditional reasoning.

6. To substantiate the claimed benefits of unified decision granularity, ReCode would be evaluated in environments where both high-level planning and low-level control are essential such as robotic manipulation or embodied control tasks. In such settings, policies must integrate perception, planning, and motor control, providing a more realistic test of ReCode’s ability to adjust decision granularity. Moreover, conducting quantitative comparisons with recent frameworks like Code-as-Policies or Vision-Language-Action (VLA) models would more convincingly demonstrate the distinct contribution and practical value of ReCode’s recursive, code-centric decision mechanism.

### Questions
How is the mechanism for learning and controlling decision granularity concretely implemented and trained in ReCode?

### Soundness
2

### Presentation
3

### Contribution
2
