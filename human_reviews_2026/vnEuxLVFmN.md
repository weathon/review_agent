# WebOperator: Action-Aware Tree Search for Autonomous Agents in Web Environment

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 4, 2, 4

## Abstract
LLM-based agents often operate in a greedy, step-by-step manner, selecting actions solely based on the current observation without considering long-term consequences or alternative paths. 
This lack of foresight is particularly problematic in web environments, which are only partially observable—limited to browser-visible content such as the current page’s DOM and UI elements—where a single misstep often requires complex and brittle navigation to undo.
Without an explicit backtracking mechanism, agents struggle to correct errors or systematically explore alternative paths. 
Tree-search methods provide a principled framework for such structured exploration, but existing approaches lack mechanisms for safe backtracking, making them prone to unintended side effects. They also assume that all actions are reversible, ignoring the presence of irreversible actions—limitations that reduce their effectiveness in realistic web tasks.
To address these challenges, we introduce WebOperator, a tree-search framework that enables reliable backtracking and strategic exploration. Our method incorporates a best-first search strategy that ranks actions by both reward estimates and safety considerations, along with a robust backtracking mechanism that verifies the feasibility of previously visited paths before replaying them, preventing unintended side effects. To further guide exploration, WebOperator generates action candidates from multiple, varied reasoning contexts to ensure diverse and robust exploration, and subsequently curates a high-quality action set by filtering out invalid actions pre-execution and merging semantically equivalent ones.
Experimental results on WebArena and WebVoyager demonstrate the effectiveness of WebOperator. Notably, on WebArena, WebOperator achieves state-of-the-art performance with gpt-4o, achieving a 54.56% success rate, underscoring the critical advantage of integrating strategic foresight with safe execution.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes WebOperator, a holistic framework for web navigation that treats browser interaction as a controlled search process rather than simple next-action prediction. At each step, the system generates multiple candidate actions, scores them with a process reward model, filters out invalid or unsafe actions, prioritizes safe and reversible actions over potentially destructive ones, and uses simulation-backed backtracking to recover before committing irreversible changes. The authors report that this holistic framework outperforms prior web agents on WebArena and WebVoyager, even when using an open-weight backbone.

### Strengths
- The paper directly targets a well-known, real weakness of current web agents: irreversible, high-impact actions in partially observable web environments.
- The method includes a safety-aware control layer (deferring destructive actions, simulating rollback, backtracking only after verification) rather than relying purely on greedy next-step policies.
- The authors report higher success rates than prior systems on WebArena and WebVoyager, in some cases even against agents using proprietary LLMs.

### Weaknesses
- **Limited technical novelty**: The paper seems much more like a holistic engineering stack than a focused research contribution. The claimed contribution is presented as a large bundle (“holistic exploration strategy”) with many different parts (retrieval, instruction rewriting, candidate action generation, WebShepherd-style scoring, validation, safety tagging, destructive-action deferral, simulation-based backtracking, etc.), but the paper does not cleanly isolates one core idea or the concept of the proposed framework. Furthermore, each of the individual components proposed in this paper(e.g., retrieval of similar trajectories, action-space pruning, checklist-based action scoring, simulation backtracking) has already been widely explored (e.g., [1], [2], [3], [4], [5], [6]) not only in general agent domain but also in web navigation tasks. As a result, it’s hard to tell what is fundamentally new, which specific mechanism is essential, or where the performance actually comes from.

- **Heavy dependence on WebShepherd-style process reward**: The framework appears to lean critically on an off-the-shelf checklist/reward model (WebShepherd) for action scoring. However, it is unclear (i) whether other baselines would also gain large improvements if given the same reward model, and (ii) how much of WebOperator’s gains remain if that reward model is removed or weakened. In other words, is WebOperator itself responsible for the improvement, or is it mainly “strong process reward + safety heuristics”?

- **Missing cost/latency analysis despite a clearly more expensive pipeline**: Because WebOperator stitches together many stages (multiple candidate expansions per step, action validation, backtracking, prioritized search, etc.), one would expect significantly higher inference cost and latency compared to simpler baselines. However, the paper does not report any experiments or analysis related to this. Furthermore, if the proposed framework requires substantially more computation than existing baselines, it remains unclear whether the performance gain is large enough to justify such additional cost and latency, or whether the method is actually more efficient in practice. Additional experiments or clarification from the authors regarding cost and efficiency would strengthen the submission.

- **Lack of specified explanation of the baselines and fairness of the comparison**: The paper claims to outperform proprietary-model baselines on WebArena using only an open-source backbone, but some of those baselines are not using the same reasoning backbone (i.e, GPT-OSS-20B vs non-reasoning proprietary models), which makes the comparison hard to interpret. It is not clear whether baselines were re-run with equivalent prompting, retrieval, instruction rewriting, WebShepherd-scoring, or action validation — or whether only WebOperator benefits from those enhancements. Without this, it’s difficult to attribute gains to the proposed framework itself rather than stronger scaffolding around it.

- **Over-claiming robustness / generalizability**: The paper sometimes suggests robustness “in the wild,” but all evaluation is still within WebArena/WebVoyager-style environments. It’s not convincingly shown that the full framework generalizes to truly open, uncontrolled web settings. Given how heuristic and component-heavy the method is, it’s not obvious that it would seamlessly transfer. (e.g., to claim generalizability in the wild, shouldn’t WebOperator demonstrate that it performs well even in environments that are more in the wild than existing WebArena-like settings such as [7], [8], and [9]? The current benchmarks it compares against are not truly in-the-wild—they are benchmarks on which most other web navigation methods are already shown to work, and thus cannot fully support the claimed generalization.)

[1] HiAgent: Hierarchical Working Memory Management for Solving Long-Horizon Agent Tasks with Large Language Model, ACL’25

[2] AgentOccam: A Simple Yet Strong Baseline for LLM-Based Web Agents, arXiv’25

[3] Web-Shepherd: Advancing PRMs for Reinforcing Web Agents, NeurIPS’25

[4] WebRollback: Enhancing Web Agents with Explicit Rollback Mechanisms, arXiv’25

[5] Is Your LLM Secretly a World Model of the Internet? Model-Based Planning for Web Agents, arXiv’25

[6] Web Agents with World Models: Learning and Leveraging Environment Dynamics in Web Navigation, ICLR’25

[7] ASSISTANTBENCH: Can Web Agents Solve Realistic and Time-Consuming Tasks?, EMNLP’24

[8] Online Mind2web: An Illusion of Progress? Assessing the Current State of Web Agents, arXiv’25

[9] WorkArena++: Towards Compositional Planning and Reasoning-based Common Knowledge Work Tasks, NeurIPS’24

### Questions
1. Key contribution
- What is the conceptual novelty that the author wants to mainly claim, beyond “a holistic stack that works well”?
2. Claims about Process reward / WebShepherd in line 46
- The intro argues that off-the-shelf step-level usefulness models (like WebShepherd-style process rewards) are inherently short-sighted and imperfect, yet WebOperator still uses exactly such a model to score and prioritize actions. I’m a little confused in this sentence, since the weboperator itself adopts the webshephered. Can you further clarify this statements or provide concrete evidence that your framework actually mitigates that myopia (e.g., cases where WebShepherd gave a misleading score but the search / destructive-action deferral / backtracking logic recovered)? Or quantitative results showing meaningful gains even when that reward model is ablated?
3. Cost and scalability.
- Can you please provide per-task end-to-end time of web operator and the compared baselines, or any kind of latency/computation analysis? Is WebOperator actually practical at scale, or are we trading significantly higher inference cost for higher success rate?

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
4

### Summary
The paper introduces WebOperator, a tree-search based web-agent framework that selects actions using an action aware, best-first search strategy. It shows that WebOperator with open source LLMs can outperform other agent approaches using closed-source LLMs on WebArena and WebVoyager. 

WebOperator combines various techniques, some of which build upon prior works, which when put together attains SOTA results on WebArena benchmark for open models. It,
1. Refines the task instruction to make it less ambiguous 
2. Performs observation space pruning: Extend AgentOccam [1] to filter redundant observations and prune infeasible actions.
3. Performs action consolidation: Merge redundant or semantically similar actions into a unified representation for reward computation
4. Reduces computational overhead in backtracking: Extend Tree-Search by Koh-et.al [2] and adopt a best-first search strategy. They make backtracking efficient by using URL-based jumps to the closest ancestor and replaying only the minimum required actions. They also handle destructive actions by heuristically detecting destructive actions and marking them as a point of no return in the search process.     
5. Reuses experience: Use RAG for candidate action generation and validation
6. Uses the existing WebShepherd reward model to generate a list of sub-goals for an instruction. This checklist guides the reward signal generation which controls the tree search. 

[1] AgentOccam: A Simple Yet Strong Baseline for LLM-Based Web Agents (https://arxiv.org/abs/2410.13825)

[2] Tree Search for Language Model Agents (https://arxiv.org/pdf/2407.01476)

### Strengths
1. WebOperator combines many techniques which when put together achieve state-of-the-art results on WebArena (55.68%) with an open-source backbone LM, outperforming baselines using closed-source LLMs like AgentSymbiotic, ScribeAgent etc.(leaderboard can be found here: https://docs.google.com/spreadsheets/d/1M801lEpBbKSNwP-vDBkC_pF7LdyGU1f_ufZb_NWNBZQ/edit?gid=0#gid=0)
2. On WebVoyager, WebOperator achieves a higher success rate (63.57%)  when compared to AgentOccam (48.84%).
3. The paper implements novel techniques to tackle typical challenges of tree-search algorithms for web-agents, like improving backtracking via action simulation and detecting destructive actions.


Overall, the technical contribution is not entirely novel but they combine multiple ideas well to demonstrate strong results on web navigation benchmarks.

### Weaknesses
1. The approach is not original in any one of the many techniques the paper uses. Tree-search for web-agents has been explored in prior works as mentioned in the paper, and it is unclear what percentage gains are brought about by WebOperator’s novelties on top of existing works like AgentOccam [1] and Tree-search [2]. A baseline which could have been used is Tree-search algorithm by Koh et. al. with AgentOccam’s observation space improvements or Tree-search algorithm with the WebShepherd reward model. 
2. It would help to have a detailed ablation study of all heuristics/techniques used in WebOperator, like instruction reframing, hyperparameters used in the tree search (like branching factor, search depth etc.), handling JavaScript alerts in the axtree, and action merging. Right now it is not clear which of these heuristics brings the most improvement.
3. The experiments in Table 2 do not help ablate gains from models v/s agent designs. 
4. The results WebOperator attains on WebVoyager are not close to state-of-the-art. Since AgentOccam is the only baseline used for WebVoyager, it is not proven that WebOperator’s gains generalize to benchmarks apart from WebArena.
5. No quantitative results are provided to back the claims made in Error Analysis (Section 3.3). 
6. The paper attempts to solve issues with backtracking to enable tree-search, but the techniques used are still heuristics to optimize the search instead of a clear solution for backtracking based exploration. Thus, the claim that the paper addresses key limitations of existing web agents (L480-481) is incorrect. 
7. Humans navigate the web by using browser-based navigation instead of relying on isolated simulations on the environment. The practicality, feasibility or challenges of using WebOperator on real websites is not addressed in the paper.


[1] AgentOccam: A Simple Yet Strong Baseline for LLM-Based Web Agents (https://arxiv.org/abs/2410.13825)

[2] Tree Search for Language Model Agents (https://arxiv.org/pdf/2407.01476)

### Questions
Suggestions:
1. In table 4, Cambridge Dict max result is bolded incorrectly. The bold font for ties misrepresents the ablation study column.
2. In Figure 5, what is Tool?
3. The claim that benchmark attains state-of-the-art performance on WebVoyager (eg, L484) is incorrect. (ref: https://github.com/sagekit/webvoyager)
4. Appendix Section C (L897) is empty
Questions:
1. How are past trajectories selected for the RAG examples? (L212)
2. Can the authors determine the performance of other baseline agent frameworks on WebVoyager? AgentOccam is not previous state-of-the-art on WebVoyager to warrant a fair comparison.
3. What is the cost/latency of running this agent? How does it compare to other baseline agents?
4. How are the instructions rephrased (L197-198)? Is this a multi-turn interaction or single turn?
Can we clarify what we mean by Dynamic Environment in Table 1?

### Soundness
3

### Presentation
3

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
WebOperator is an action-aware tree-search framework for reliable web agents. It ranks actions by both reward and safety, integrates simulation-verified backtracking to avoid irreversible errors, and refines exploration through instruction reformulation, adaptive observation, and retrieval-augmented examples. These design choices enable efficient and robust task completion in partially observable web environments. Experiments on WebArena and WebVoyager show that even with open-source LLMs, WebOperator surpasses state-of-the-art proprietary agents, achieving strong generalization and practical efficiency.

### Strengths
S1. Engineering Maturity and Real-World Robustness
- The system demonstrates a high level of engineering completeness and stability, functioning reliably in real browser environments (e.g., simulation tabs, URL-based backtracking).
- It effectively handles realistic constraints such as partial observability, irreversible actions, and search efficiency, showcasing strong robustness and precision under real-world conditions.

S2. Competitive and Generalized Performance
- The model achieves consistently strong results across multiple realistic benchmarks, including WebArena and WebVoyager.
- It outperforms or matches powerful proprietary baselines, demonstrating generalization beyond a single dataset and adaptability across diverse web environments.

S3. High Quantitative Performance
- Quantitatively, the model shows solid empirical results across benchmarks, supporting the effectiveness of the proposed approach in web-based reasoning and navigation tasks.

S4. Action Safety and Classification
- The paper acknowledges the risk of destructive actions and classifies action types to mitigate potential harm.
- This shows awareness of safety considerations in web-agent design, contributing to practical reliability.

### Weaknesses
W1. Lack of Readability and Coherence
- The paper’s writing style and organization are inconsistent, making it difficult to follow the main narrative.
- The introduction fails to clearly convey the motivation, problem statement, and core contributions.
- Key components—such as Rephrase Instruction and Optimized Observation—are insufficiently described, weakening conceptual clarity.

W2. Limited Research Novelty
- The method primarily combines existing heuristics and engineering improvements rather than introducing a fundamentally new algorithmic concept.
- Theoretical depth is limited; for instance, the URL-based backtracking closely resembles prior work such as WebRollback (https://arxiv.org/abs/2504.11788).
- The paper lacks clear articulation of what distinguishes its conceptual contributions from existing frameworks.

W3. Insufficient Quantitative Analysis of Core Features
- Although backtracking and destructive-action handling are presented as key innovations, their actual impact is not empirically validated.
- No dedicated experiments, subsets, or metrics (e.g., efficiency, success rate) directly measure these components’ contributions.

W4. Missing Ablation and Comparative Studies
- The absence of ablation analysis prevents understanding of how much each module contributes to overall performance.
- Comparisons against strong baselines (e.g., GPT-4, tree-search vs. non–tree-search methods) are limited, leaving the empirical strength of claims unsubstantiated.

W5. Efficiency and Inference Cost Concerns
- Tree-search–based reasoning likely increases inference latency and computational cost, raising concerns about practical usability.
- The paper lacks measurements or discussions of efficiency (e.g., runtime, search cost, latency).
- Without fair comparisons under equal model capacity, claims of performance gains may not justify the added computational overhead.

W6. Shallow Qualitative and Case Analyses
- The paper provides little qualitative insight into model behavior—no example traces, success/failure case studies, or interpretive discussions.
- This omission limits readers’ understanding of why the system succeeds or fails under specific conditions.

### Questions
- Comparison with World Model–based planning: The motivation emphasizes the necessity of tree search–based exploration, yet no direct comparison is provided against World Model–based planning approaches such as WebDreamer (https://arxiv.org/abs/2411.06559). The trade-offs between these paradigms should be clearly presented.
- Necessity of backtracking: It remains unclear whether backtracking is always needed. For instance, in a movie-booking scenario where the user mistakenly selects The Lord of the Rings instead of Harry Potter, it would be more efficient to correct the action within the current page rather than returning to the initial state. Such realistic use cases should be discussed to clarify when backtracking provides real benefit.
- Table 5 issue: The performance peak when Rephrased is disabled and Retrieved examples = 0 is unexplained. This suggests unclear roles and interactions among the system’s components and warrants further clarification.
- World Model:  Even if world model has more inference time, heuristic has lots of risk in the partially observable web environment. It is well done to classify the type of an action, isn't it better to predict the following state with world model for action classification? Why doesn’t the paper use a world model for action validation or destructiveness detection, as in prior work (https://arxiv.org/abs/2410.13232)? A learned world model could offer more general and adaptive validation than heuristic rules, so clarifying this choice would strengthen the paper’s justification.

### Soundness
3

### Presentation
2

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
The paper extends a best-first search approach for solving web tasks by so-called action-awareness, where a type of action-safety reasoning, action merging inside the search tree and backtracking using URL-based state jumping are proposed. The authors suggest heuristics for guiding the agent towards safer actions which are not destructive for the task at hand.

The approach is evaluated on WebArena and WebVoyager via BrowserGym. Baselines tree search methods such as LM-TS and non-tree search methods such as AgentOccam.

The results show superior performance for numerous baselines on WebArena, and better performance compared to AgentOccam on the lite variant and WebVoyager.

### Strengths
- The methodological improvements over LM-TS are well-motivated
- The evaluation is sufficiently thorough, focussing on WebArena.
- The evaluation results are promising achieving significant improvement with an open-source model compared to competitors on larger, commercial models.

### Weaknesses
- Improvements are iterative compared to LM-TS. 
- Also provided methods for dealing with destructive actions are heuristics to sort-of shape rewards towards safe actions, so more generalized solutions are still needed. 
- Safety is a central concept in the paper, but the related works as well as empirical evaluation do not sufficiently focus on this aspect. I think this needs to be improved before acceptance.

### Questions
- Are there cases where the heuristics could make the agent too conservative?
- Are there additional related works focussing on LLM agent safety in other ways, e.g., post-hoc to inference or via alignment? Could we use these too and could these perform better?

### Soundness
3

### Presentation
3

### Contribution
2
