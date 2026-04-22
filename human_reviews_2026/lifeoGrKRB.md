# AGENT*: Optimizing Test-Time Compute for Multi-Agent Systems with Modularized Collaboration

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 6, 4, 4

## Abstract
Scaling test-time computation has emerged as a powerful and increasingly popular approach for improving the performance of large language models without additional training. Recent work demonstrates that techniques such as repeated sampling, self-verification, and self-reflection can significantly enhance task success by allocating more inference-time compute. However, applying these techniques directly to multi-agent systems is challenging, as they provide no principled way to encourage collaboration or manage compute allocation across multiple agents under budget constraints. To address this, we propose AGENT*, a general framework for enabling effective multi-agent collaboration while operating within strict compute budgets. AGENT* introduces the notion of \emph{modularized collaboration}, formalized as callable functions that encapsulate reusable multi-agent workflows, automatically constructed via self-play reflection by abstracting recurring interaction patterns from past trajectories. Building on these collaboration modules, AGENT* proposes \emph{a dual-level planning architecture} that optimizes compute allocation by reasoning over the current task state while also \emph{speculating} on future steps. Experiments on complex agent benchmarks demonstrate that AGENT* consistently outperforms baselines across diverse budget settings, validating its effectiveness for multi-agent collaboration in inference-time optimization.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces AGENT*, a framework that optimizes test-time computation in multi-agent systems under fixed compute budgets. It proposes collaboration modules that capture reusable coordination strategies and a dual-level planning architecture combining short-term decision making with long-term budget estimation. Experiments on GAIA and BrowseComp-Plus benchmarks using Claude and Qwen3 models show improved success rates and more efficient compute utilization compared to orchestration and budget-aware baselines.

### Strengths
1. Clearly defines an important and emerging problem on managing inference-time compute for multi-agent systems.
2. The dual-level planning structure provides an intuitive way to coordinate agents while maintaining compute constraints.
3. The experiments are systematic and demonstrate consistent performance gains across several models and budget settings.

### Weaknesses
1. Limited empirical diversity Experiments are conducted only on GAIA and BrowseComp-Plus, both text-based web-agent benchmarks. The method’s generality for other multi-agent settings (e.g., embodied collaboration, code generation, or reasoning-only domains) remains unverified, making it unclear whether AGENT*’s benefits generalize beyond these specific tasks.
2. Evaluation metrics and significance. The reported gains (roughly 2–4% absolute accuracy improvement) are modest and lack statistical validation such as confidence intervals or significance testing. It is difficult to judge whether these differences are meaningful given the stochasticity of LLM outputs and sampling variability.
3. Unclear cost realism and reproducibility. The “budget” is defined in monetary units ($0.2–0.5) without clear mapping to real compute (tokens, latency, or FLOPs). This abstraction complicates interpreting efficiency claims and reproducing the budgeted setting on other models or infrastructures.

### Questions
1. How exactly is the “budget” quantified during experiments? Is it based on token usage, latency, or monetary cost estimates from API calls? A clearer and reproducible definition would help assess whether the efficiency improvements are practical and transferable to other model families.
2. Could the authors provide an ablation isolating the effects of the short-term and long-term planning components? It would help determine whether both levels are essential, or if most of the gains come from one part of the dual-level design.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes Agent*, a framework for budget-aware multi-agent collaboration. The framework introduces reusable collaboration modules that the orchestrator can invoke directly, and selects actions using a dual-level planning architecture that balances short-term task completion and long-term speculation under budget constraints. Experiments on GAIA and BrowseComp-Plus show that Agent* achieves higher performance than baselines and utilizes the budget more effectively.

### Strengths
1.	The paper is clearly written and addresses an important and practical issue: multi-agent reasoning under constrained budgets.
2.	The design of Agent* is intuitive and well-structured. It incorporates commonly used components such as reusable skills and self-consistency sampling, organizing them in a coherent and elegant manner.
3.	Experimental results demonstrate that both the collaboration modules and budget-aware planning contribute effectively, and that Agent* outperforms standard test-time scaling methods with comparable costs.

### Weaknesses
1. The related work section should include papers that explore budget constraints in agentic reasoning, e.g., [1] [2].

[1] Zheng, Yuanhang, et al. "Budget-Constrained Tool Learning with Planning." Findings of the Association for Computational Linguistics ACL 2024. 2024.

[2] Qiu, Rennai, et al. "Co-Saving: Resource Aware Multi-Agent Collaboration for Software Development." arXiv preprint arXiv:2505.21898 (2025).

2. The current baselines function more as ablation studies within the proposed framework. Additional comparisons with other multi-agent collaboration and orchestration frameworks would strengthen the validation of Agent*’s effectiveness.

### Questions
1.	Why is normalization applied to the budget feasibility score? Does this favor trajectories with lower costs and cause the average cost to remain consistently below the budget?
2.	Could you provide an analysis of how Agent* allocates resources? Does it explicitly perform test-time scaling for particularly difficult tasks?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces AGENT*, a framework for optimizing test-time compute allocation in multi-agent systems (MAS) under fixed inference-time budgets. Experiments on GAIA and BrowseComp-Plus benchmarks comparing AGENT* with several baselines (no modules, with modules, budget-aware prompting, and test-time scaling methods like best-of-N and iterative verification). The paper claims that AGENT* consistently outperforms baselines in both accuracy and budget utilization, demonstrating more efficient test-time scaling for collaborative agents.

### Strengths
1. This paper tries to address an important and timely problem, the efficient inference-time scaling for multi-agent systems.

### Weaknesses
1. The numerical improvements are within standard deviation of baselines. Accuracy improvements (≈ +2–3 %) are modest and not statistically validated
2. Only GAIA and BrowseComp-Plus are used. It would be better to evaluate on more benchmarks to test the efficacy of the method on diverse domains.
3. The paper lacks experiments isolating each component (dual-level planner, self-play reflection, cost estimation). Without these, it’s impossible to know what truly drives performance.
4. The related-work section misses recent agentic workflow optimization and inference-time orchestration papers
5. The “budget” concept is abstracted into token cost, but real inference latency and system cost are not reported.
6. Figures and captions are poorly formatted. Many fonts are too small. quantitative data tables are cluttered;

### Questions
1. How sensitive are results to the number of self-play rounds or to the number of collaboration modules derived?
2. Would AGENT* still outperform when all systems are trained with the same cost-budgeted objective (e.g., Muennighoff et al., 2025 “S1”)?
3. How often are collaboration modules reused across tasks, and do they transfer?

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
5

### Summary
This paper investigates the problem of optimizing **test-time scaling (TTS)** strategies within **multi-agent systems (MAS)**. The authors employ **self-play** on a validation set to automatically discover potential **collaboration modules** and their corresponding computational costs, thereby expanding the search space beyond that of a single agent. During inference, the system leverages this expanded space to solve complex tasks for each query in a **budget-aware**, stepwise manner, following an **orchestrator–worker** collaboration paradigm to adaptively select appropriate agents or modules.

### Strengths
1. **Novel research problem.** This paper extends the optimization of test-time scaling to multi-agent systems under the orchestrator–worker paradigm, exploring how to maximize overall system performance and effectively utilize individual agent capabilities under a fixed computational budget. The problem setting demonstrates strong research value.

2. Experimental results show that the proposed collaboration modules and dual-level planning mechanisms effectively enhance the performance utilization of multi-agent systems under budget constraints, validating the practical potential of the proposed approach.

### Weaknesses
1. **Insufficient baseline design.** The paper lacks well-designed comparative baselines, making it difficult to demonstrate the state-of-the-art effectiveness of the proposed method convincingly.
   (1) In terms of automated agent collaboration optimization, the paper does not compare against recent works that explicitly optimize multi-agent coordination—such as GPTSwarm [1], G-Desiger [2], AFlow [3], MAS-GPT, and TTS-related agent scaling [4], [5], [6]—whose underlying philosophy is closely related to the proposed budget-aware orchestration mechanism.
   (2) Regarding the TTS setting, the authors only compare one way of utilization of two TTS strategies (best-of-N and iterative verification) which include the TTS-related modules in the agent space. Still, they omit other simple combination strategies of TTS and agents, e.g., applying TTS directly to the original single-agent in each subtask (for instance, dynamically assigning budget to each agent under TTS strategies such as *Best-of-N*).

[1] Zhuge, Mingchen, et al. *“GPTSwarm: Language Agents as Optimizable Graphs.”* *Forty-first International Conference on Machine Learning (ICML)*, 2024.
[2] Zhang, Guibin, et al. "G-designer: Architecting multi-agent communication topologies via graph neural networks." arXiv preprint arXiv:2410.11782 (2024).
[3] Zhang, Jiayi, et al. "Aflow: Automating agentic workflow generation." arXiv preprint arXiv:2410.10762 (2024).
[4] Ye, Rui, et al. "MAS-GPT: Training LLMs to build LLM-based multi-agent systems." arXiv preprint arXiv:2503.03686 (2025).
[5] Jin, Can, et al. "Two heads are better than one: Test-time scaling of multi-agent collaborative reasoning." arXiv preprint arXiv:2504.09772 (2025).
[6] Qian, Chen, et al. "Scaling large language model-based multi-agent collaboration." arXiv preprint arXiv:2406.07155 (2024).


2. **Misalignment between research goals and methodology.** Although the paper claims to aim for “performance maximization,” the proposed method lacks an explicit mechanism for performance optimization. For example, the short-term planning component relies solely on self-consistency scoring and uses a single LLM to generate execution strategies. It remains unclear whether this LLM can adaptively select the optimal collaboration pattern for performance across diverse tasks.

3. **Insufficient methodological details.**
   (1) Although the authors mention that their policy generation follows the *ReAct* framework, they do not clarify how candidate actions are concretely generated under the proposed setting (e.g., prompt templates, sampling strategies).
   (2) The description of the *self-play* process omits key implementation details, such as which TTS techniques are used, how experience is store, updated, and summarized, and how the number of sampled modules is determined.

4. **Lack of discussion on model scaling effects.** The paper does not address performance differences between small and large models under equivalent computational budgets—an essential topic in TTS research. Prior work [7] has shown that smaller models can sometimes outperform large models given equivalent compute resources.

[7] Snell, Charlie, et al. *“Scaling LLM Test-Time Compute Optimally Can Be More Effective than Scaling Model Parameters.”* *arXiv preprint arXiv:2408.03314*, 2024.

5. **Questionable use of a validation set.** The paper aims to address query-level inference scenarios, yet it relies on a validation set to discover collaboration modules. This assumption is unrealistic, as no such validation set would be available in real-world inference settings to support experience-based module selection.

6. **Spelling and grammatical issues.** Several minor language problems reduce readability. For instance, in Section 2, “each initizlied with a large language model (LLM)” should be corrected to “initialized.”

7. **Incomplete citations.** Some key references are missing. For example, line 189 (“A* algorithm”) and line 46 (“orchestrator–worker”) should be accompanied by appropriate citations.

8. **Ambiguity in terminology.** The term *“collaboration module”* is potentially misleading, as it suggests a method component rather than an *agent group*. The authors are advised to adopt a clearer term to better reflect its intended meaning.

### Questions
1. The paper states that “we conduct five rounds of self-play reflection for each benchmark, with each round producing one candidate module.” Would increasing the number of reflection rounds further improve overall performance?

2. Which specific **test-time scaling** techniques are employed in the paper? Does the approach incorporate strategies such as *Best-of-N*, *Self-Refinement*, or other advanced TTS strategies?

3. How does the proposed algorithm ensure performance maximization in its design? Are there explicit optimization objectives or adaptive mechanisms?

4. What is the specific role of the symbol $\kappa$ in the formula at line 171?

5. Is the task step number $H$ fixed or dynamically determined? If it is dynamic, how does this affect the overall budget constraint?

6. Can the proposed method be extended to other agent collaboration ways beyond orchestrator–worker multi-agent system configurations, like debate-based collaboration?

### Soundness
2

### Presentation
2

### Contribution
3
