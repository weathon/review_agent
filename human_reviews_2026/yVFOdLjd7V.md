# Deft Scheduling of Dynamic Cloud Workflows with Varying Deadlines via Mixture-of-Experts

- Decision: Accept (Poster)
- Scores: 4, 4, 6, 6

## Abstract
Workflow scheduling in cloud computing demands the intelligent allocation of dynamically arriving, graph-structured workflows with varying deadlines onto ever-changing virtual machine resources. However, existing deep reinforcement learning (DRL) schedulers remain limited by rigid, single-path inference architectures that struggle to handle diverse scheduling scenarios. We introduce $\textbf{DEFT}$ ($\textbf{D}$eadline-p$\textbf{E}$rceptive Mixture-o$\textbf{F}$-Exper$\textbf{t}$s), an innovative DRL policy architecture that leverages a specialized mixture of experts, each trained to manage different levels of deadline tightness. To our knowledge, DEFT is the first to introduce and validate a Mixture-of-Experts architecture for dynamic cloud workflow scheduling. By adaptively routing decisions through the most appropriate experts, DEFT is capable of meeting a broad spectrum of deadline requirements that no single expert can achieve. Central to DEFT is a $\textbf{graph-adaptive}$ gating mechanism that encodes workflow DAGs, task states, and VM conditions, using cross-attention to guide expert activation in a fine-grained, deadline-sensitive manner. Experiments on dynamic cloud workflow benchmarks demonstrate that DEFT significantly reduces execution cost and deadline violations, outperforming multiple state-of-the-art DRL baselines.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses the problem of Cost-Aware Dynamic Workflow Scheduling (CADWS) in cloud computing. The goal is to intelligently assign dynamically arriving graph-structured workflows (DAGs) with varying deadlines to virtual machines (VMs) to minimize total cost (including VM rental fees and deadline violation penalties).

The authors point out that existing deep reinforcement learning (DRL) schedulers often use rigid, single-path policy networks (i.e., a single Priority Mapping Module, PMM), making it difficult for them to adapt to variable deadline tightness.

To solve this problem, the paper proposes DEFT (Deadline-pErceptive Mixture-oF-Experts), an innovative DRL policy architecture. To the best of the authors' knowledge, DEFT is the first work to apply a Mixture-of-Experts (MoE) architecture to dynamic cloud workflow scheduling.

Its core contributions include:

1. An MoE policy network where each "expert" is specialized to handle a specific level of deadline tightness (controlled by a $\gamma$ coefficient).
2. A novel "graph-adaptive gating mechanism." It uses a Graph Attention Network (GAT) to encode the workflow's DAG structure and combines VM state and deadline information to dynamically select the most appropriate expert via a cross-attention mechanism.
3. A two-stage training strategy: first, pre-training each expert independently on specific $\gamma$ values, and then jointly training the gating network and fine-tuning all experts in an environment with mixed $\gamma$ values.

Experimental results show that DEFT significantly outperforms existing SOTA DRL baselines (such as GATES, SPN-CWS) on several CADWS benchmarks. The advantage of DEFT in reducing total cost is particularly evident on medium-to-large scale (M/L) workflows.

### Strengths
1. **Originality:** This paper is the first to introduce a Mixture-of-Experts (MoE) architecture to the field of dynamic cloud workflow scheduling (CADWS). The idea of dedicating different experts to handle scheduling strategies for different levels of deadline tightness directly addresses the "one-size-fits-all" pain point of existing DRL schedulers.

2. **Quality:** The core component, the "graph-adaptive gating mechanism," is very well-designed. It not only uses GAT to encode the DAG topology of the workflow but also skillfully integrates VM status, task features, and the critical deadline information, performing expert routing via cross-attention. This design is highly aligned with the characteristics of the CADWS problem domain.

3. **Clarity:** The paper is clearly written. The problem definition, related work, and methodology are well-articulated. The appendix is detailed and aids in reproducibility.

4. **Significance:** The CADWS problem solved in this paper has significant practical importance and challenges in cloud computing resource management. The method demonstrates strong performance, especially on medium (M) and large (L) scale workflows, indicating good scalability and practical value.

### Weaknesses
1. Insufficient Baseline Comparison and Ablation: A major weakness of the paper is the lack of a rigorous ablation study on the contribution of the MoE architecture itself. DEFT is built upon the SEM from GATES, with the main modification being the replacement of GATES's PMM with an MoE-PMM. A crucial baseline is missing: one that uses the same SEM as GATES, but where the PMM is a single, deeper, or wider FFN (or the original GATES PMM), trained end-to-end on the mixed-deadline dataset (i.e., the stage-2 dataset). It is currently unclear if the GATES baseline was trained on this mixed dataset. If not, DEFT's advantage might partially stem from a richer data distribution or longer training (two-stage), rather than solely from the MoE architecture.

2. Furthermore, DEFT uses a two-stage training process: stage 1 pre-trains experts on fixed $\gamma$ values, and stage 2 trains the gate and fine-tunes experts on mixed $\gamma$ values. However, the paper does not explicitly state whether its SOTA baselines (especially GATES) were also trained or fine-tuned on the same mixed $\gamma$ value dataset (the stage-2 dataset). If GATES was trained on a single $\gamma$ value or a simpler dataset, then DEFT's performance advantage (especially on M/L scales) might simply come from exposure to more diverse training data, rather than from its MoE architecture.

3. **Expert Generalization:** The expert pre-training (stage 1) uses a fixed set of $\gamma$ values ({1.25, 1.75, 2.25, 5.0}), but during the gating network training (stage 2) and testing, $\gamma$ is sampled from a different, denser set ({1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 3.0}). The authors do not adequately explain how the gating network handles $\gamma$ values not seen during pre-training (e.g., 1.0, 1.5, 2.0, 3.0). Does the gating network interpolate between these points? Or does it rely primarily on the stage-2 fine-tuning? An analysis of how the gate makes decisions when encountering these unseen $\gamma$ values would be valuable.

4. **Simplification of Top-k Routing:** The paper states in Appendix F that the Top-k setting is $k=1$ in experiments. This means that at each decision step, the gating network selects only one expert, rather than mixing the outputs of multiple experts. This appears to be an oversimplification of the MoE paradigm, making it more of a "hard switch" than a "mixture." Why not use $k=2$? Using $k=1$ might lead to increased variance in routing decisions and forfeits the advantage of MoE's ability to combine experts.

5. **Lack of Hyperparameter Sensitivity Analysis:** The MoE architecture introduces new hyperparameters, particularly the number of experts and the choice of $\gamma$ values for pre-training. The paper uses 4 experts but does not discuss why these 4 specific $\gamma$ values were chosen. How would performance change if the number of experts was increased or decreased, or if their specialized $\gamma$ values were different? This is crucial for the method's robustness and practicality.

6. The core contribution of this paper is applying the MoE architecture to a DRL scheduler. However, MoE itself is a mature technique and has already been explored in the DRL field for solving combinatorial optimization and VRP problems. This paper's problem is analogous to a special case of these.

7. The paper emphasizes "Mixture-of-Experts" in its title and abstract, but admits in Appendix F that its experimental setup uses $k=1$. This implies that DEFT in practice is not a "mixture" model but a "Switch-of-Experts" model—selecting only one expert at each decision point. This somewhat undermines the core advantage of MoE (i.e., achieving smoother, more robust decisions by combining outputs from multiple experts). The authors do not provide experimental results for $k>1$ (e.g., $k=2$), and if they did not, an explanation for choosing a single expert per decision point is warranted.

### Questions
1. **Regarding the GATES baseline:** Was the GATES baseline (used for comparison in Tables 1 and 2) trained on the same mixed deadline distribution as DEFT's stage 2 (i.e., with $\gamma$ sampled from {1.0, ..., 3.0})? If not, could you please provide the performance of a GATES or DEFT variant with a single PMM (instead of MoE) trained on this same mixed dataset, to fairly evaluate the true gains from the MoE architecture?
2. **Regarding expert generalization:** The stage-1 pre-trained experts target specific $\gamma$ values. When new $\gamma$ values (like 1.0, 1.5, 2.0) are encountered in stage 2 and during testing, which expert(s) does the gating network tend to select? For example, when $\gamma=1.5$, does it choose the $\gamma=1.25$ expert or the $\gamma=1.75$ expert? Does this rely entirely on the stage-2 fine-tuning?
3. **Regarding Top-k = $k=1$:** Why was $k=1$ chosen instead of $k=2$ or higher? Does using $k=1$ mean that DEFT forgoes the ability to "mix" experts and instead just "switches" between them?
4. **Regarding the number and selection of experts:** How was the number of experts (4) and the choice of $\gamma$ values ({1.25, 1.75, 2.25, 5.0}) determined? Could you provide a sensitivity analysis on the number of experts (e.g., 2 or 6 experts) or their $\gamma$ value selection? Or add a section explaining the reasoning for these hyperparameter choices.

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
### **Review Summary**

This paper introduces DEFT, a Deep Reinforcement Learning (DRL) agent for Cost-Aware Dynamic Workflow Scheduling (CADWS). The central hypothesis is that monolithic DRL policies are inflexible and perform poorly when faced with a wide spectrum of workflow deadline requirements. The proposed solution, DEFT, replaces the final policy-mapping module with a Mixture-of-Experts (MoE) architecture. The model includes multiple "expert" sub-networks, each pre-trained to handle a specific deadline tightness ($\gamma$), and a "graph-adaptive gating network" that routes tasks to the most suitable expert based on the workflow's DAG structure, task state, and deadline. The authors present a two-phase training strategy and show that their model outperforms prior SOTA baselines, including GATES, on benchmark cloud scheduling tasks.

While the core idea of applying MoE to this problem is interesting, the paper's contributions are undermined by a lack of critical ablation studies, an incremental design, and a complete omission of inference cost analysis. The added complexity of the MoE and the graph-adaptive gate is not sufficiently justified, making the performance gains appear marginal in some cases and the overall contribution feel incomplete.

### Strengths
### **Strengths**

1.  **Problem Motivation:** The paper correctly identifies a practical and significant limitation of existing DRL schedulers: a single, fixed policy struggles to be optimal for the diverse range of deadline tightness found in real-world scenarios (e.g., a cost-saving policy will miss tight deadlines, while an aggressive policy will overspend on lenient ones) [cite: 77, 143-145].
2.  **Novelty of MoE Application:** Applying a Mixture-of-Experts (MoE) architecture to DRL for this *specific* scheduling domain appears to be a novel contribution[cite: 6]. The idea of specialising experts for different deadline regimes is intuitive and logical[cite: 5, 147].
3.  **Strong Scalability Results:** The empirical results in Table 1 are a clear strength. DEFT outperforms the SOTA GATES baseline, and this performance gap widens significantly as the problem scale increases (from a 0.9% improvement at small scale to a 29.6% improvement at large scale), demonstrating the scalability and robustness of the MoE approach [cite: 771-772].

### Weaknesses
### **Weaknesses and Questions**

1.  **Incremental Contribution:** The paper builds directly on the GATES architecture, explicitly stating that it "directly inherits its GNN-based policy network as its SEM module"[cite: 756]. The core contribution is replacing the final Priority Mapping Module (PMM) with the MoE-gate assembly[cite: 226]. This makes the work feel like a highly incremental improvement to a 2025 baseline (GATES) [cite: 755] rather than a fundamentally new framework.
2.  **Missing Critical Ablation Studies:** The paper fails to provide the necessary ablations to justify its core design choices. This is its most significant weakness.
    * **Gating Network Complexity:** The "graph-adaptive gating network" is a major new component, using a GAT and cross-attention [cite: 8, 156, 485, 595-598] that must be executed *at every scheduling step*. [cite_start]The paper claims simpler gating networks are "inadequate" [cite: 594] but provides no experimental comparison. How does this complex gate compare to a simple MLP gate that only looks at the workflow's deadline $\gamma$ as input? Without this baseline, the value of the complex GAT-based router is unproven.
    * **MoE Hyperparameters:** The model uses four experts [cite: 480-481, 761]. This number is a critical hyperparameter, yet it is presented without any justification or sensitivity analysis. How does performance change with two experts, or eight? [cite_start]Furthermore, Appendix F reveals that $k=1$ (hard routing) was used for expert selection[cite: 1455]. This is a major simplification of the MoE paradigm and is not justified. Did the authors experiment with soft mixtures or $k>1$, and if so, how did it impact performance and (presumably) inference cost?
3.  **No Inference Cost Analysis:** For a paper on real-time dynamic scheduling, the complete omission of any inference latency analysis is a critical flaw. The proposed DEFT model adds a GAT and a cross-attention mechanism *on top of* the existing GATES GNN, all of which runs at *every decision step* [cite: 595-598]. This added complexity must have a non-trivial impact on the decision-making time. Without a direct runtime comparison to the GATES baseline, the practical value of the model is impossible to assess.
4.  **Marginal Gains at Trained Scale:** The model is pre-trained and fine-tuned on Small (S) scale instances[cite: 761, 763]. However, on the S-scale test set, the performance gain over the GATES baseline is marginal (52.95 vs. 52.46, a 0.9% improvement) [cite: 771-772]. This suggests that all this added complexity (MoE, GAT-gate, cross-attention) provides almost no benefit for the data distribution it was actually trained on, questioning the cost/benefit ratio of the proposed architecture.

### Questions
No questions

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
3

### Summary
This paper presents DEFT, a novel DRL-based scheduling framework for dynamic cloud workflows with varying deadlines. By leveraging a MoE architecture, each expert specializes in a specific level of deadline urgency. A graph-adaptive gating network with cross-attention dynamically selects the most suitable expert based on workflow structure, task states, VM conditions, and deadline tightness. Experiments demonstrate that DEFT significantly reduces execution cost and deadline violations, outperforming existing DRL and heuristic baselines, especially on large-scale workflows.

### Strengths
The proposed MoE architecture tailored for dynamic cloud workflow scheduling, where each expert is specialized for a specific level of deadline urgency. This design enhances adaptability over traditional monolithic DRL approaches. Additionally, the proposed graph-adaptive gating mechanism effectively integrates workflow DAG structures and real-time system context via cross-attention, enabling fine-grained, deadline-aware expert selection and improving scheduling precision.

### Weaknesses
1. Degraded MoE with K=1 Routing: The use of Top-1 routing (K=1 in Part F. Additional Training And Testing Details) turns the MoE into a hard selection of a single expert, preventing any combination of expert behaviors. This limits the model’s flexibility and may reduce its ability to generalize to intermediate deadline scenarios.


2. Lack of Justification for Gating Complexity: The paper does not compare the complex gating network with simpler baselines. Without such comparisons, it is unclear whether the added computational and implementation cost is justified by actual performance improvements.

### Questions
The paper does not report inference latency or computational overhead. Given the complexity of the MoE + GNN + attention-based gating architecture, it is unclear whether DEFT is suitable for real-time scheduling scenarios. What is the inference latency of DEFT compared to simpler DRL baselines?

### Soundness
3

### Presentation
3

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
This paper introduces DEFT, an innovative DRL policy architecture that leverages a specialized mixture of experts for dynamic cloud workflow scheduling. By adaptively routing decisions through the most appropriate experts, DEFT is capable of meeting a broad spectrum of deadline requirements that no single expert can achieve. Central to DEFT is a graph-adaptive gating mechanism that encodes workflow DAGs, task states, and VM conditions, using cross-attention to guide expert activation in a fine-grained, deadline-sensitive manner. Experiments on dynamic cloud workflow benchmarks demonstrate that DEFT significantly reduces execution cost and deadline violations, outperforming multiple state-of-the-art DRL baselines.

### Strengths
The paper introduces Mixture-of-Experts (MoE) into DRL-based workflow scheduling for different deadline tightness, and explicitly considers workflow-level characteristics.

### Weaknesses
1.	In real-world cloud environments, scheduling decisions often involve multiple concurrent workflows and cross-workflow resource contention. It would be helpful to clarify whether the proposed approach can handle inter-workflow dependencies or coordination.

2.	The paper assigns each expert to a specific deadline tightness regime. However, it is unclear why deadline tightness was chosen as the sole dimension for expert specialization. Could other factors—such as workflow size, task heterogeneity, or resource type—also guide expert assignment?

3.	The scalability of the proposed MoE framework remains uncertain. As the number of workflows or task dimensions increases, how does the method scale in terms of computational cost and expert routing efficiency?

### Questions
1.	Can the proposed approach handle inter-workflow dependencies or coordination?

2.	Could other factors—such as workflow size, task heterogeneity, or resource type—also guide expert assignment?

3.	As the number of workflows or task dimensions increases, how does the method scale in terms of computational cost and expert routing efficiency?

### Soundness
3

### Presentation
3

### Contribution
3
