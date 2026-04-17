# Separable Policy Learning for Emergency Vehicle Prioritized Traffic Signal Control

- Decision: Reject
- Scores: 2, 2, 8, 4

## Abstract
Traffic Signal Control plays a vital role in optimizing urban traffic flow and reducing accidents by regulating signal phases at intersections. While traditional fixed-time control methods are simple and infrastructure-efficient, they fail to adapt to complex and dynamic traffic patterns, particularly during peak periods or in the presence of emergency vehicles. In this paper, we address the emergency-vehicle-aware traffic signal control problem by proposing a decoupled policy fusion framework that separately optimizes control strategies for regular vehicles and emergency vehicles. The two policies are later combined into a global strategy with automatically learned weights, mitigating the negative impact of $Q$-function approximation errors. We further introduce SplitEMV, a novel multi-agent model that enhances inter-agent communication and decision efficiency. Experiments demonstrate that our method significantly improves emergency vehicle response times while preserving efficiency of regular vehicles. The learned emergency vehicle prioritized policy also integrates seamlessly with existing traffic signal control methods in a zero-shot manner, supporting practical deployment.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes Separable Policy Learning (SplitEMV) for emergency-vehicle-prioritized traffic signal control. It introduces a decoupled reinforcement learning framework that independently learns policies for regular and emergency vehicles, then fuses them through Adaptive Strategy Merging (ASM) to ensure robust, weight-free integration. The proposed SplitEMV model enhances inter-agent communication and achieves significant reductions in emergency vehicle travel time while maintaining normal traffic efficiency.

### Strengths
S1. Separately optimizes policies for regular and emergency vehicles, avoiding conflicts and improving training stability.

S2. Dynamically fuses policies without manual tuning, ensuring robust performance under varying traffic densities.

S3. Enables seamless zero-shot integration of the learned emergency-vehicle policy with existing traffic control methods.

### Weaknesses
W1. The assumption that the control policies for regular and emergency vehicles can be fully decoupled may oversimplify their intrinsic interaction within shared traffic dynamics.

W2. The experimental validation relies on only two simulated datasets and a limited number of baseline methods, which restricts the generality of the findings.

W3. Much of the reported performance gain may result from richer state representation and a stronger attention-based model rather than the proposed decoupled learning mechanism itself.

### Questions
Q1. The proposed method assumes that the objectives of regular vehicles (RVs) and emergency vehicles (EMVs) can be fully separated into two independent Q-functions that are later linearly combined. However, in real traffic systems, the behaviors of RVs and EMVs are strongly coupled, priority signals for EMVs inevitably affect surrounding traffic flows. This independence assumption may therefore be unrealistic, and the linear combination $Q=Q_N=\beta Q_E$ lacks theoretical justification under such interdependence. As a result, the optimality and stability of the decoupled strategy remain uncertain.

Q2. Although the paper presents promising results on the Hangzhou and Jinan datasets, the experimental scope is relatively narrow. The evaluation lacks diversity in traffic conditions and network scales. Furthermore, the baselines are mostly classical MaxPressure-based reinforcement learning models, without comparisons to SOTA methods. This limited set of benchmarks and environments constrains the robustness and external validity of the reported improvements.

Q3. While the paper attributes performance improvement to separable policy learning and adaptive strategy merging, the state representation and model architecture differ significantly from the baselines. SplitEMV uses fine-grained lane-level inputs with multiple features, including lane direction, signal phase, and emergency-vehicle indicators, and employs multi-head attention for inter-lane communication. In contrast, the baselines such as CoLight, PressLight, and MPLight rely primarily on aggregated vehicle counts. Without controlling for state complexity or citing related representation-learning studies, it is difficult to isolate whether the improvements stem from the proposed learning framework or from the enhanced feature design and network capacity.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper tackles emergency-vehicle-prioritized traffic signal control by proposing SplitEMV, a reinforcement learning framework that separately learns policies for regular and emergency vehicles, then merges them via an Adaptive Strategy Merging (ASM) mechanism. The model aims to achieve efficient coordination among intersections, minimize emergency response times, and preserve normal traffic efficiency.

### Strengths
S1. The paper introduces a clear decoupling of regular and emergency vehicle control policies, thereby mitigating objective interference and improving interpretability.
S2. The adaptive merging strategy elegantly eliminates the need for manual tuning of reward weights, resulting in more stable integration across varying traffic densities.
S3. Extensive experimental results, including ablation studies and zero-shot transfer to other TSC models, demonstrate the robustness and generality of the proposed method.

### Weaknesses
W1. The assumption of full separability between regular and emergency vehicle objectives may not hold in practice, where traffic interactions are inherently coupled.
W2. The empirical evaluation is limited to two public datasets and a relatively small set of baselines; comparisons with more recent multi-agent or GNN-based approaches would strengthen the claims.
W3. Some of the reported advantages might stem from architectural improvements (e.g., richer state inputs and attention-based communication) rather than the decoupled learning formulation itself.

### Questions
Q1. The paper assumes that regular and emergency vehicle policies can be trained independently and later linearly combined. Could the authors justify the theoretical validity of this decomposition under coupled system dynamics?
Q2. How sensitive is ASM to errors in estimating the normalization statistics (e.g., variance of the Q-values)? Are there cases where this normalization could destabilize learning?
Q3. The ablation studies suggest strong performance even without certain communication modules. Could the authors clarify whether the gains primarily come from ASM or the improved network architecture?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper addresses a practically significant problem in emergency-vehicle–prioritized urban traffic signal control. It proposes SplitEMV, a decoupled multi-objective reinforcement learning framework that separates the learning of regular-vehicle and emergency-vehicle policies and fuses them through an Adaptive Strategy Merging (ASM) mechanism. This design aims to balance global traffic efficiency and emergency-vehicle priority under varying traffic conditions. Experiments on real-world city networks show consistent improvements over both traditional and RL-based baselines. However, while the method is technically sound and mathematically well-formulated, the paper would benefit from clearer motivation for ASM, richer evaluation metrics, and broader comparisons with recent MARL frameworks to strengthen its overall contribution.

### Strengths
1.Addresses a practically important and underexplored problem of EMV-prioritized multi-intersection control.

2.The proposed Decoupled Learning + ASM framework is conceptually sound and demonstrates consistent performance gains across all tested baselines.

3.The mathematical formulation is well-developed, with clear propositions and proofs supporting the proposed adaptive merging process.

4.Experimental setup is reasonable, and the comparisons with both traditional and RL-based methods provide a solid foundation.

### Weaknesses
1. The figure clarity needs improvement.

2. The presentation of experimental results is rather monotonous.

3. The transition between problem background, limitations of existing methods, and proposed solutions is abrupt. The introduction would benefit from an explicit “problem → limitation → solution → contributions” structure.

4. Limited evaluation metrics.

5. Limited baseline algorithm.

### Questions
1. Figure 2 employs excessively small fonts and visual elements that are difficult to read, thereby reducing the clarity of the proposed architecture.

2. The experimental section presents all results in tables only. Additional visualizations would improve readability and interpretability.

3. The introduction does not clearly explain why adaptive merging is needed or what specific problem it solves.

4. Current results rely primarily on Average Travel Time (ATT). To more comprehensively evaluate performance, additional indicators such as Weighted Waiting Time (WWT), queue length, or fairness variance are recommended.

5. The paper should include more recent baselines, such as hierarchical or value-decomposition MARL frameworks, to contextualize the proposed method better.

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
The paper introduces a two-phase learning scheme, Decoupled Learning and Adaptive Strategy Merging (ASM), that separates policy learning for regular vehicles (RVs) and emergency vehicles (EMVs).

### Strengths
This design mitigates Q-function approximation errors and minimizes interference between conflicting optimization objectives. It provides a generalizable framework that could extend beyond traffic signal control to other multi-objective reinforcement learning (RL) problems.  Moreover, the proposed EMV model can integrate with existing traffic signal control (TSC) methods without retraining (“zero-shot EMV generalization”), demonstrating strong adaptability and deployment potential.

### Weaknesses
1. All experiments are conducted in simulated SUMO environments. There is no field deployment or real-world validation to demonstrate robustness against sensor noise, communication delays, or unexpected vehicle behaviors.  Suggestion: It would be informative to evaluate performance in CityFlow or other simulation platforms with more realistic and heterogeneous traffic conditions.
2. While SplitEMV outperforms EMVLight and several DRL baselines, it does not include comparisons with recent multi-agent or graph-based TSC systems, such as RobustLight (ICML 2025), DMBP, or DiffLight. The absence of these baselines makes the “state-of-the-art” claim somewhat overstated.
3. Although the paper reports comparable runtime performance, the training process appears more complex due to: two independent Q-functions (QN, QE), multi-stage learning phases, and adaptive normalization (ASM) for β.  Moreover, no quantitative analysis of training time, convergence rate, or scalability is provided.
4. The method assumes accurate and delay-free EMV state communication to all agents. In real-world urban networks, localization errors, sensor failures, or communication latency could cause incorrect priority assignments. These limitations are not discussed.
5. The connection between ASM normalization and policy optimality remains largely qualitative. During training, the RV model (Q_N) and EMV model (Q_E) are trained independently, yet later considered equivalent in the joint stage. Similarly, rewards $r^n$ and $r^e$ are treated as identical, without justification. These components should instead be adjustable to environmental conditions.

### Questions
1. The approach requires extensive manual tuning of β and other hyperparameters, making it labor-intensive and potentially difficult to replicate.
2. Several recent and relevant baselines (e.g., RobustLight, DMBP, DiffLight) were not included, which limits the comprehensiveness of the comparative study.
3. Although the paper criticizes fixed β, it still relies on empirical normalization constants. Despite dedicating significant space to discussing β’s effects, it fails to explain how ASM adapts dynamically to unseen traffic patterns. The automatic adaptive normalization mechanism for β remains unclear and potentially unstable.
4. The training pipeline, particularly the adaptive strategy merging process shown in Figure 1, is insufficiently explained. A detailed algorithmic description or pseudocode should be provided to clarify the procedure.
5. The claimed zero-shot generalization is effectively a linear policy fusion, not true zero-shot transfer to unseen tasks. Since the EMV policy requires offline pretraining on similar distributions, this claim could mislead readers.
6. The method seems overfitting to training cities due to heavy reliance on tuned β values and validation-based hyperparameter adjustment. This risks hidden overfitting when the validation and target environments overlap.

### Soundness
2

### Presentation
2

### Contribution
2
