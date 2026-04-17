# Weaving in the Clouds: Achieving Synergistic Collaboration among LLM Agents via Federated Learning

- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Multi-Agent Systems (MAS) powered by Large Language Models (LLMs) have shown immense potential in solving complex, sequential tasks by simulating expert collaboration. However, their reliance on centralized data clashes with real-world privacy constraints and data silos. Conversely, existing privacy-preserving paradigms like Federated Learning (FL) typically ignore the inherent sequential dependencies present in collaborative workflows, leading to suboptimal performance. To bridge this critical gap, we introduce FedWave, a novel framework for federated multi-agent collaboration. FedWave empowers LLM-based agents to collaboratively solve complex sequential tasks under strict privacy constraints by employing three core mechanisms: (1) a collaborative Value Chain Layer to model sequential dependencies, enabling efficient local fine-tuning through Federated Learning with LoRA adapters; (2) an intelligent Mixture of Experts (MoE) router at the server level for dynamic, task-aware aggregation of expert knowledge, moving beyond simple averaging; and (3) a final Direct Preference Optimization (DPO) stage to align the model's collaborative outputs with human preferences. Extensive experiments demonstrate that FedWave significantly outperforms both traditional federated learning and centralized multi-agent baselines, effectively achieving synergistic collaboration without compromising data privacy. The codes are available at https://anonymous.4open.science/r/FedWave-111A.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a novel framework, FedWave, which integrates federated learning with large language model (LLM)-based multi-agent collaboration. The framework captures sequential dependencies through a Value Chain Layer (VCLayer), achieves cross-task knowledge sharing via Mixture-of-Experts (MoE) dynamic aggregation, and aligns outputs with human preferences using Direct Preference Optimization (DPO). The paper is well-structured, and the experiments are comprehensive. The work is innovative, technically solid, and represents a cutting-edge exploration in federated learning and LLM collaboration. However, the theoretical analyses are slightly lacking, and the interpretability needs to be strengthened, so the paper has a strong potential for acceptance if it can be further improved.

### Strengths
1. Originality and Novelty: This is the first work to introduce sequential dependency modeling into a federated learning framework, formalizing multi-agent collaboration as a distributed optimization problem.

2. Completeness of System Design: The proposed three-stage architecture, structure modeling, knowledge aggregation, and human alignment, is well-conceived. The approach is reproducible and integrates LoRA-based lightweight updates, effectively reducing communication costs and mitigating privacy risks.

### Weaknesses
1. The motivation of the paper presents logical inconsistencies. The authors claim that the motivation for proposing the FedWave framework lies in the “conflict between centralized data dependency and real-world privacy constraints as well as data silos.” However, this statement lacks sufficient justification. In multi-agent collaboration tasks, centralized training is generally not limited by data silo or privacy issues. As shown in Table 2, the authors use the same dataset for both centralized and federated settings. Therefore, using this as the main motivation to introduce a federated learning framework appears unconvincing. The authors are encouraged to clarify: Why existing centralized multi-agent cooperation mechanisms are not applicable in this scenario, and what exactly the so-called “data silo” problem refers to in the context of their experiments.

2. Figure 1 contains excessive information and appears visually cluttered. In particular, notations such as W_i^t  and W_g^t  are not clearly explained in the main text, which makes it difficult for readers to follow the overall workflow. The authors are encouraged to improve the figure’s clarity and provide explicit definitions for all symbols used.

3. In terms of writing, Section 4.4 is densely formatted, with different experimental analyses (e.g., ablation and robustness tests) mixed in the same subsection. The paragraphing and organization make it difficult for readers to understand the experimental conclusions. The authors should consider restructuring this section to separate different types of analyses for better readability and logical flow.

4. In the methodology section, three loss functions are designed for the VC Layer, and two additional loss terms are introduced in the MoE Router. However, the specific roles, relationships, and impacts of these losses on the overall training objective are insufficiently discussed. The authors should clarify the intuition and purpose of each loss function and, preferably, include convergence analysis or theoretical discussions—particularly under non-IID and sequentially dependent multi-client scenarios—to justify whether FedWave’s local updates and global aggregation can converge stably.

5.In Section 3.2, the proposed MoE Router is claimed as one of the core innovations of FedWave. However, as a global module updated via backpropagation from multiple clients, its behavior remains unclear. The authors should explain how the router’s weight distribution evolves with different client inputs or tasks, whether the dynamic routing mechanism could cause gradient oscillations or mode collapse, and how stable global updates and routing policy learning are achieved across clients.

6. In Section 3.3, the DPO optimization process is executed in a centralized manner, where preference data pairs are constructed from the responses of individual experts. This design may introduce privacy leakage risks since such preference data could implicitly reveal client information. The authors should provide reasonable justification or mitigation strategies for this issue.

7. From Table 3, removing the DPO layer leads to the most significant performance drop, while removing other components has only minor effects. This suggests that the DPO stage plays the dominant role in the model’s performance. The authors are encouraged to further analyze the interaction between different modules or report results when only one component (e.g., DPO-only or MoE-only) is retained, to validate the independent contribution of each module.

### Questions
Please see the weaknesses above.

### Soundness
2

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
3

### Summary
The paper targets privacy-preserving multi-agent LLM collaboration across data silos. It proposes FedWave, combining: (i) a Value Chain Layer (VCLayer) to encode sequential/causal dependencies among expert roles via three auxiliary losses (positional, continuity, and
consistency), (ii) a server-side trainable MoE router for task-aware aggregation instead of static FedAvg, and (iii) a post-federated DPO stage that uses router-induced “win/lose” expert outputs to auto-construct preference pairs for alignment. Experiments on MSCoRe workflow datasets across several 7B/8B backbones demonstrate consistent gains over FL baselines and competitive performance compared to centralized multi-agent systems.

### Strengths
1. The paper tackles a highly relevant and novel research problem at the intersection of MAS, FL, and LLMs. Reconciling the collaborative power of agents with the privacy constraints of real-world data is a significant challenge, and the paper's focus on "sequential dependencies" in FL is a key insight.
 
2. The proposed three-component architecture (VCLayer, MoE Router, and DPO) is well-designed and coherent. Each component addresses a distinct aspect of the problem logically.

### Weaknesses
1. The VCLayer and its associated losses appear to be designed for a fixed, linear sequence of N experts. The experiments use N=4. It is unclear how this approach would scale to a much larger number of experts or to more complex, non-linear workflows.

2. There is a slight ambiguity about aggregation. Section 1 states the router "discards traditional FL aggregation", but Section 3.2 implies that the router's own parameters are aggregated using a FedAvg-like method. Furthermore, FedWave's default aggregation is FedAvg, which can be replaced by other FL optimizers.

3. There are only 4 expert clients with full participation and 20 rounds, which is far from realistic FL.

4. MoE routing, VCLayer, and DPO add nontrivial local compute; router parameters add communication. However, the computation and communication overhead are not analyzed and reported.

### Questions
1. How does FedWave perform with more clients and partial participation?

2. How would VCLayer approach scale?

3. Is our understanding correct that during training, the parameters are aggregated via standard FL, while at inference, the MoE router's function is what replaces the averaging of expert outputs?

4. Could you provide the analysis and report of the computation and communication overhead?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces **FedWave**, a federated learning framework designed for synergistic collaboration among LLM-based agents tackling complex, sequential business workflow tasks, all while respecting privacy constraints. FedWave comprises three core components: (1) a Value Chain Layer (VCLayer) that models inter-agent dependencies within workflows; (2) a server-side Mixture-of-Experts (MoE) Router that enables dynamic, task-aware expert aggregation; and (3) a Direct Preference Optimization (DPO) stage to align the global model’s outputs with human preferences. Extensive experiments demonstrate that FedWave consistently outperforms both federated learning and centralized multi-agent baselines across multiple real-world datasets. Qualitative analyses further corroborate these findings, showcasing the framework’s ability to generate coherent, role-appropriate outputs even under stringent privacy constraints.

### Strengths
1. The paper addresses a tangible limitation in current LLM multi-agent and federated learning methods: their inability to jointly support sequential dependency modeling (essential to real-world workflows) and stringent privacy constraints.
2. Across three business-oriented workflow datasets (Automotive, E-commerce, Pharmaceutical) and multiple LLM architectures (Qwen2-7B, Llama2-7B, Llama3-8B), FedWave yields consistent and substantial performance improvements. Table 1 demonstrates significant gains on key metrics, particularly on semantic metrics like Meteor and BS-F.

### Weaknesses
1. **Dynamic Knowledge Aggregation**: When the number of expert clients is large, organizing them into an MoE system can incur significant resource overhead. The experiments only consider a scenario with four clients and do not explore cases with a larger number of clients, under which FedWave may become difficult to execute. Furthermore, there is no detailed breakdown of training, validation, and test splits, nor is it explicitly stated whether best-validation checkpoint selection is performed or if test-overfitting might occur. Additionally, the algorithm’s pseudocode is not provided, which further complicates reproducibility.
2. **Theoretical Analysis**: This work lacks theoretical analysis, particularly regarding the convergence and stability of the MoE router, and there is a lack of formal justification. For example, does the load balancing loss suffice, and are there regret bounds expected from this dynamic expert-selection? 
3. **Computational Overhead and Scalability**: While communication efficiency is claimed (thanks to LoRA and expert routing), there is little quantitative analysis of actual resource usage, per-round communication cost, or training/inference latency.

### Questions
See weakness.

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
This paper introduces FedWave, a novel federated learning framework that allows LLM agents to solve sequential tasks like business workflows while preserving data privacy. Its core contributions are three mechanisms: a Value Chain Layer (VCLayer) to model the sequential dependencies between agents, an intelligent MoE router for dynamic, task-aware aggregation of agent knowledge, and a DPO stage to align the final collaborative output with human preferences.

### Strengths
- The paper's primary strength is its novel FedWave framework for solving the problem of collaborative, sequential agent tasks within a privacy-preserving federated learning setting. 

- The research is of high quality. The design is validated by extensive experiments showing FedWave not only outperforms federated baselines but also competes with or even surpasses centralized, non-private systems.

### Weaknesses
- The final DPO phase is presented as aligning the model to human preferences, but its mechanism is entirely self-referential. The preference dataset is generated by anointing the MoE router's top-ranked expert as the "winning" response and a lower-ranked expert as "losing". This creates a high risk of a bias amplification loop. If the MoE router learned any inaccuracies or biases during the SFT phase, this DPO stage would not correct them but would instead amplify them, training the final model to be even more confident in the router's potentially flawed "preference."
- The fixed, linear, and known-in-advance workflow (e.g., A $\rightarrow$ B $\rightarrow$ C). Real-world business processes are often more complex, involving branches (A $\rightarrow$ B and A $\rightarrow$ C), conditions (if X, then A $\rightarrow$ B; else A $\rightarrow$ D), or dynamic routing. 
- The framework introduces a large number of new, sensitive hyperparameters to tune, including three for the VCLayer ($\lambda_{pos}$, $\lambda_{cont}$, $\lambda_{cons}$), one to balance SFT and VC losses ($\gamma$), two for the MoE router's auxiliary losses ($\delta_1$, $\delta_2$), and one for DPO ($\beta$). While the paper provides a sensitivity analysis (Figures 2 and 3), it only analyzes one parameter at a time. This is insufficient for such a complex loss function, as these terms likely have strong, non-obvious interactions.

### Questions
See weaknesses

### Soundness
3

### Presentation
3

### Contribution
2
