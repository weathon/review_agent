# TopoWeaver-R1: Reinforcing Difficulty-Aware Topology Evolution in Multi-Agent Competition-Level Code Generation

- Decision: Reject
- Scores: 4, 4, 6, 6

## Abstract
Recent studies have shown that large language model (LLM)-driven multi-agent systems (MAS) are promising for addressing complex problems, with competition-level code generation as a representative domain. By emulating the collaboration among human programmers,  these systems leverage predefined interaction topologies to achieve notable gains. However, such fixed structures introduce interaction redundancy and excessive token costs as task difficulty drops. While graph pruning and generation methods can produce sparser topologies, they remain static during inference, unable to adapt to execution feedback, and often converge to limited density ranges. To overcome these issues, we propose TopoWeaver-R1, a reinforcement learning–optimized MAS centered on an LLM orchestrator agent, which supports end-to-end evolutionary dynamic interaction topology generation. For each query, it infers agent roles and task difficulty, then constructs a task-adapted, density-aware layered directed acyclic graph (DAG) topology. The topology evolves via execution feedback and history, thereby improving the task-solving performance of the generated code. On three competition-level and two basic code datasets, TopoWeaver-R1 achieves state-of-the-art accuracy, with up to 14.6\% higher accuracy, 13\% lower density and 68\% lower token cost than the strongest baseline. Our approach transitions multi-agent topologies from static designs to dynamic, feedback-driven evolutionary designs with fine-grained, difficulty-aware density control.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces TopoWeaver-R1, a framework designed to optimize MAS for competition-level code generation. It addresses the inefficiency of fixed communication topologies, which lead to redundancy and excessive token costs, by introducing a difficulty-aware mechanism. The framework utilizes RL to dynamically generate efficient interaction graphs for agent communications.

### Strengths
The method explores addressing the problem of dynamic topology optimization in LLM-based MAS, moving beyond static, handcrafted structures to improve efficiency and reduce token costs. The optimized MAS achieves a better balance of performance and computational expense.

### Weaknesses
1. The paper lacks discussion on the design principles, heuristic choices, and specific numerical weighting applied to the reward functions. This critical omission prevents the rigorous assessment of the sensitivity of the final performance to these crucial hyperparameters.

2. The proposed method is actually a general paradigm, but the experimental validation is limited to the specific domain of "competition-level code generation". The general applicability to more representative MAS problems, such as multi-step arithmetic reasoning, complex planning, or scientific document summarization, where communication requirements, agent roles, and reward structures differ significantly, remains unproven. 

3. The quantitative results are presented without standard deviation across multiple experimental runs, which should be added for more solid validation.

4. The paper lacks crucial details regarding the SFT data, including the data source, filtering process, and quality assurance protocols. This missing information makes it hard to assess the potential bias or limitations of the dataset.

### Questions
1. The current document's font, layout appear inconsistent with the official ICLR submission template？

2. Why does the method only optimize the topology of MAS without considering other MAS factors such as agent role, prompts or functions?

### Soundness
3

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
5

### Summary
This paper addresses the inefficiency and rigidity of fixed-topology multi-agent systems (MAS) in complex domains like competition-level code generation. The authors propose TopoWeaver-R1, an MAS framework centered on an LLM "orchestrator agent" trained with reinforcement learning. This orchestrator dynamically generates a layered, directed acyclic graph (DAG) topology, represented in YAML, based on the task's inferred difficulty. The topology evolves across multiple turns in response to execution feedback. The orchestrator is trained using a multi-objective reward function that balances code accuracy, structural correctness, and topology density, with density targets explicitly conditioned on task difficulty. Experimental results on five code benchmarks show that TopoWeaver-R1 achieves state-of-the-art pass@1 accuracy while simultaneously reducing token costs and utilizing sparser, more efficient topologies compared to baselines.

### Strengths
1. The paper is well-organized and uses language that is easy to understand.

2. The proposed TopoWeaver-R1 framework is a complete and sensible solution, employing a three-stage pipeline (SFT, RL) to train a dedicated orchestrator agent.

3. The design choice of using YAML to represent the layered DAG topology is strong, as it is both human-readable and easily generated by an LLM.

### Weaknesses
1. Limited Novelty of Dynamic Topologies: The paper's central claim of addressing static agent structures appears overstated.  The core idea of dynamic or optimized interaction patterns is not entirely novel;  similar motivations to move beyond fixed topologies have been previously explored in works such as MaAS[1] and FlowReasoner[2].

2. Inaccurate Characterization of FlowReasoner: The paper's distinction from FlowReasoner seems to be based on an inaccurate premise.  The authors classify FlowReasoner as a method that "focuses on optimizing sequential workflows."  However, to my understanding, FlowReasoner operates on a search space defined by ADAS[3], which is not limited to sequential structures and likely encompasses the parallel, graph-based topologies defined in this work.

3. Limited Empirical Scope: The paper's evaluation is confined solely to the domain of code generation.  This narrow focus makes it difficult to assess the generalizability of the TopoWeaver-R1 framework.  This contrasts with other significant works in agent architecture optimization (e.g., ADAS[3], Aflow[4], AgentSquare[5], MaAS[1]), which have demonstrated the generality of their approaches by evaluating on a more diverse set of benchmarks across different domains.

4. The experimental setup appears problematic. Limiting the multi-agent interaction to 2 turns suggests the model may be receiving and using feedback from the test set to correct its initial solution. This protocol, which allows for test-time adaptation, differs from Aflow's setup (search on validation, single pass on test) and thus seems to constitute an unfair comparison.

Reference

[1]Zhang G, Niu L, Fang J, et al. Multi-agent Architecture Search via Agentic Supernet[C]//Forty-second International Conference on Machine Learning.

[2]Gao, Hongcheng, et al. "Flowreasoner: Reinforcing query-level meta-agents." arXiv preprint arXiv:2504.15257 (2025).

[3]Hu S, Lu C, Clune J. Automated Design of Agentic Systems[C]//The Thirteenth International Conference on Learning Representations.

[4]Zhang J, Xiang J, Yu Z, et al. AFlow: Automating Agentic Workflow Generation[C]//The Thirteenth International Conference on Learning Representations.

[5]Shang Y, Li Y, Zhao K, et al. AgentSquare: Automatic LLM Agent Search in Modular Design Space[C]//The Thirteenth International Conference on Learning Representations.

### Questions
1. To further substantiate the claims within the code domain, could the authors conduct experiments on other significant benchmarks, such as SWE-bench[6]?

2. To demonstrate the generalizability of the framework, could the authors provide experimental results on benchmarks from more diverse domains, for example, GAIA[7] or HLE[8]?

3. Could the authors please clarify the aforementioned issue regarding the experimental setup?

Reference

[6]Jimenez C E, Yang J, Wettig A, et al. SWE-bench: Can Language Models Resolve Real-world Github Issues?[C]//The Twelfth International Conference on Learning Representations.

[7]Mialon G, Fourrier C, Wolf T, et al. Gaia: a benchmark for general ai assistants[C]//The Twelfth International Conference on Learning Representations. 2023.

[8]Phan L, Gatti A, Han Z, et al. Humanity's last exam[J]. arXiv preprint arXiv:2501.14249, 2025.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes TopoWeaver-R1, a multi-agent system (MAS) for competition-level code generation. It tackles the inefficiency of fixed-topology systems by using a reinforcement learning (RL) optimized orchestrator agent. This orchestrator dynamically generates an interaction topology (a layered DAG) in YAML format, adapting the graph's density based on the problem's inferred difficulty and execution feedback. The system is trained via Supervised Fine-Tuning (SFT) followed by RL (GRPO) using a multi-objective reward function.

### Strengths
- The core idea of an RL-trained orchestrator that generates *difficulty-aware* and *evolutionary* (feedback-driven) topologies is highly original.
- The method achieves SOTA accuracy while being significantly more cost-effective than all baselines, a rare and important result.
- The multi-objective reward function, especially the $\mathcal{S}_{complex}$ density metric (Eq. 7) that is tied to task difficulty (Eq. 13), is a clever and effective design.

### Weaknesses
- The SFT stage, which is shown to be crucial, relies on data generated by a powerful proprietary model (GPT-40).
- The agent roles (planner, coder, etc.) are predefined. The system optimizes the *interaction graph* but not the *composition* of the team itself.
- It remains unclear how well the orchestrator transfers to unseen problem types or agent role definitions, or whether it overfits to the YAML schema used in training.

### Questions
1. How sensitive is the final performance to the weights ($\lambda_1, \lambda_2, \lambda_3$) chosen for the $\mathcal{S}_{complex}$ reward in Eq. 7? Were these tuned per dataset or fixed globally?

2. How does the system perform if the initial difficulty-level inference (which sets $N_{max}(l)$ in Eq. 13) is incorrect? Can the RL policy recover through execution feedback, or does it remain constrained by the wrong density cap?

3. Have the authors analyzed whether the RL-trained orchestrator produces diverse graph patterns across problem types, or does it converge to a small family of template structures?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes TopoWeaver-R1, a reinforcement-learning–based framework for multi-agent code generation that introduces dynamic, difficulty-aware topology evolution. An LLM-based orchestrator outputs layered YAML-DAG structures describing agent collaboration workflows, which adapt across multiple turns according to execution feedback and inferred task difficulty. A composite reward integrating code execution results, topology complexity, and YAML validity guides this evolution through Group Relative Policy Optimization (GRPO). Experiments on APPS, LiveCodeBench, CodeContests, HumanEval, and MBPP demonstrate consistent gains compared to strong multi-agent baselines.

### Strengths
1.	Novel conceptual framing: Recasting multi-agent orchestration as a dynamic topology evolution task is original and impactful.
2.	Structured and interpretable representation: The YAML-based layered DAG allows both fine-grained control and interpretability.
3.	Comprehensive experimentation: Solid benchmarking and ablations convincingly show the benefit of dynamic evolution.
4.	Reward engineering: The integration of task difficulty, code execution, and structural complexity into a single reward is elegant and domain-specific.
5.	Systematic pipeline: The SFT + RL training workflow is clearly delineated and reproducible.

### Weaknesses
1.	Algorithmic innovation remains limited.
The use of SFT and GRPO is largely standard; while well-executed, the RL stage mainly adds domain-specific reward shaping rather than introducing new optimization techniques.
2.	Empirical boundary conditions not fully explored.
The dynamic process is capped at two evolution turns, yet the paper provides no analysis of whether additional turns would further improve or destabilize performance. Similarly, there is no study on how sensitive the model is to reward weightings or the choice of density evaluation function.
3.	Difficulty-awareness partially hand-designed.
The mechanism for controlling topology density by preset thresholds (4/7/10) is heuristic. It would be stronger if this control were learned automatically rather than manually fixed.
4.	Data quality concerns in the SFT stage.
The SFT dataset used to pretrain the orchestrator is synthetically generated, but the paper does not clarify how the ground truth YAML topologies are validated or filtered. Given that topology correctness directly affects downstream RL stability, more evidence of data verification or human curation would strengthen confidence in the results.

### Questions
How is the quality of the SFT training data ensured? If YAML topologies are generated automatically, what validation mechanisms or filtering thresholds are used to ensure they align with valid ground-truth agent workflows?

How sensitive is performance to the task difficulty classification? Have you tested robustness against misclassified or noisy difficulty labels?

The dynamic process is limited to two turns — is there an empirical reason or trade-off behind this bound?

How does the orchestrator handle inconsistent or cyclic YAML outputs? Are such cases frequent, and how are they penalized in training?

Can the reward weighting parameters (α,β,γ) be tuned automatically, or do they require manual adjustment per dataset?

### Soundness
3

### Presentation
4

### Contribution
3
