# Personalized Collaborative Learning with Affinity-Based Variance Reduction

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 4

## Abstract
Multi-agent learning faces a fundamental tension: leveraging distributed collaboration without sacrificing the personalization needed for diverse agents. This tension intensifies when aiming for full personalization while adapting to unknown heterogeneity levels—gaining collaborative speedup when agents are similar, without performance degradation when they are different. Embracing the challenge, we propose personalized collaborative learning (PCL), a novel framework for heterogeneous agents to collaboratively learn personalized solutions with seamless adaptivity. Through carefully designed bias correction and importance correction mechanisms, our method AffPCL robustly handles both environment and objective heterogeneity. We prove that AffPCL reduces sample complexity over independent learning by a factor of $\max\\{n^{-1}, \delta\\}$, where $n$ is the number of agents and $\delta\in[0,1]$ measures their heterogeneity. This *affinity-based* acceleration automatically interpolates between the linear speedup of federated learning in homogeneous settings and the baseline of independent learning, without requiring prior knowledge of the system. Our analysis further reveals that an agent may obtain linear speedup even by collaborating with arbitrarily dissimilar agents, unveiling new insights into personalization and collaboration in the high heterogeneity regime.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes an affinity-based variance reduction approach that adaptively interpolates between federated learning and independent learning according to the level of objective-function heterogeneity (or data-distribution heterogeneity) among agents. Theoretically, when the data distributions across agents are homogeneous (i.e., all agents share the same objective function and the local optimal solution coincides with the global optimal solution), the proposed approach achieves the same convergence rate as federated learning. In contrast, when the data distributions are heterogeneous, the proposed approach converges to each agent's local optimal solution with a convergence rate similar to that of independent learning (where agents train in isolation). Numerical simulations validate the effectiveness of the proposed approach.

My main concerns lie in: i) overly idealized assumptions (see Weakness 1 and 2); ii) extremely heavy communication burden (see Weakness 3); iii) lack of empirical validation on benchmark datasets (see Weakness 4); iv) the reliance on a DRE oracle contradicts the "adaptivity without prior knowledge" claim (see Problem 1).
While the theoretical analysis is sound under the given assumptions, the absence of real-world evaluations and restricted applicability of the proposed approach limit the contribution of the paper.

### Strengths
The paper is clearly presented, and the theoretical results are well explained. The personalized collaborative learning problem studied in this work is also an active research topic.

### Weaknesses
1. **Limited applicability:** The authors assume that all agents share a common and fixed feature extractor (i.e., $\bar{A}$; see the first formula in Section 3). This assumption is unrealistic in many real-world applications. For example, in end-to-end deep learning, both the backbone and task-specific heads are typically trained jointly. Therefore, the proposed approach is only applicable to "frozen feature + personalized head" settings, which limits its practical applicability.

2. **Overly simplified problem setting:** The linear system in Eq. (2) essentially corresponds to the optimization problem $\min F(x)=\frac{1}{n}\sum_{i=1}^{n}f_{i}(x)$, where each local objective function $f_{i}(x)=||x^{\top}\bar{A}x-(\bar{b}^{i})^{\top}x||^2$ is strongly convex and even quadratic. This formulation is considerably simpler than those used in most existing studies on federated learning or personalized collaborative learning, which typically consider convex or nonconvex objective functions. 

3. **Heavy communication overhead:** According to Algorithm 1 in the Appendix (which provides a summarized version of the proposed approach presented in the main text), each agent is required to send $s_{t}^{i}$, $g_{t}^{i}(x_{t}^{c})$, $g_{t}^{i,b}(\theta_{t}^{c})$, $g_{t}^{c\rightarrow i}(x_{t}^{c})$ to the centralized server, while the server requires to broadcast $g_{t}^{0,c}(x_{t}^{c})$, $g_{t}^{0,b}(\theta_{t}^{c})$, $g_{t}^{c\rightrightarrows i}(x_{t}^{c})$ to all agents. All these shared variables are high-dimensional, as they depend on the dimensionality of the underlying data in machine learning applications.
Moreover, this bidirectional communication must occur at every iteration, which leads to an extremely heavy communication overhead. This contradicts the core motivation of FL, where communication is typically much more expensive than local computation.

4. **Weakness on empirical evaluation:** The experiments are performed only on synthetic linear systems, without any evaluation on real-world datasets (e.g., FeMNIST, Shakespeare). Moreover, there lacks comparison with existing personalized FL approaches such as pFedMe, Ditto, or Clustered FL.

### Questions
See the Weaknesses section above. In addition, I have a few additional questions:

1. The paper repeatedly states that the proposed approach "does not require prior knowledge" and "automatically adapts to unknown heterogeneity." However, the algorithm relies on a density ratio term. Although the authors mention using a density-ratio estimator when this ratio is unknown, Theorem 2 shows that estimating it is nontrivial and inevitably introduces error. Therefore, the claim of "no prior knowledge required" is overstated unless the authors either (i) prove that the estimation is unbiased or (ii) analyze convergence in the presence of this estimated error.

2. The definition of personalization in this paper differs from that in mainstream studies, as it emphasizes only local optimality while ignoring global generalization. In fact, personalized models must balance adaptation to local data with maintaining global generalization. The paper does not discuss this point, which is an omission. Please clarify this point.

3. Some notation is unclear (e.g., in Equation 1, "t" denotes the iteration, not sample size).

### Soundness
2

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a personalized collaborative learning (PCL) framework tailored for heterogeneous agents, addressing both environmental and objective heterogeneity. The proposed method, AffPCL, is theoretically shown to reduce sample complexity by a factor of max(n^{-1},\sigma) and analysis shows agent may obtain linear speeedup with arbitraily disimilar agents. The simulation results shows improvement over compared methods.

### Strengths
- The topic of personalized collaborative learning for heterogeneous agents is relevant and important. The formulation of PCL is a meaningful contribution, 
- The proposed framework and algorithm are clearly described and well-motivated. The formulation is broad enough to cover multiple domains such as supervised learning, reinforcement learning, and statistical decision-making.
- Simulation results convincingly demonstrate the advantages of AffPCL, particularly under high heterogeneity settings.

### Weaknesses
- While Theorem 2 provides a lower bound, the paper would be stronger if it integrated and analyzed a practical DRE method within the AffPCL framework.
- The quantification of objective heterogeneity is not clearly defined, which may hinder its application in real-world scenarios.
- The current experimental evaluation is limited to synthetic linear systems. Including experiments on real-world benchmarks would significantly strengthen the empirical validity of the method.
- Although the theoretical framework is applicable to reinforcement learning, the settings are not covered in the experiments.
- The framework considers asynchronous importance estimation, but this aspect is not evaluated or discussed in the simulation results. 
- The proposed update rule incurs higher communication overhead compared to standard federated averaging, which may limit its scalability in certain practical settings.

### Questions
Please see the weakness part.

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
This paper introduces Personalized Collaborative Learning (PCL), a unified framework for learning across heterogeneous agents that strikes a balance between personalization and inter-agent collaboration. The core algorithm, Affinity-based PCL (AffPCL), dynamically interpolates between independent and collaborative updates based on inter-agent affinity, leveraging both bias correction and importance correction mechanisms. The authors provide thorough convergence guarantees under this framework, showing that collaborative training never hurts and yields linear speedup under high affinity. Application scope spans supervised learning, reinforcement learning, and general decision-making tasks, positioning PCL as a general approach for multi-agent systems.

### Strengths
The paper presents a principled and flexible learning framework that smoothly interpolates between personalized and collaborative training, depending on the agent's similarity. By using affinity-adjusted optimization, AffPCL intelligently adapts collaboration strength instead of relying on static weighting schemes. The theoretical results are rigorous, incorporating valid assumptions and proofs that explain the model’s behavior across a range of heterogeneous settings. The clarity of exposition and strong conceptual coherence contribute to the paper’s accessibility and impact.

### Weaknesses
The limited scope of empirical evaluation undermines the practical validity of these claims. The experiments are confined to small-scale synthetic setups and do not compare AffPCL with more diverse or state-of-the-art baselines (e.g., pFedMe, Ditto, SCAFFOLD), despite some of these being mentioned in the related work. The computational and communication overhead of AffPCL, particularly in large-scale federated deployments that require density ratio estimation and per-sample corrections, remains unclear. The method’s reliance on accurately estimating density ratios or heterogeneity measures may also restrict applicability.

### Questions
How does the computational and communication complexity of AffPCL scale when deployed in realistic settings with hundreds or thousands of agents? An analysis of per-iteration runtime or messaging overhead relative to methods like FedAvg would help clarify scalability.

Can the authors discuss or evaluate AffPCL on larger or more realistic datasets, beyond small-scale synthetic setups? This would help confirm whether the benefits observed in controlled environments persist under real-world data heterogeneity.

The method relies on estimating density ratios or affinity between agents. How might practitioners estimate these measures in practical federated learning scenarios where distributions are unknown or evolving?

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
3

### Summary
This paper introduces Personalized Collaborative Learning (PCL) for federated settings with heterogeneous agents, where each agent has unique data and goals but collaborates via a central server. The proposed algorithm, AffPCL, uses affinity-based variance reduction to balance personalization and collaboration so that agents benefit from similar peers while avoiding degradation from dissimilar ones. It achieves finite-sample convergence with a mean-squared error of order $O(t^{-1} \\max\\{n^{-1}, \\delta\\})$, adapting to agent similarity without needing prior clustering. Experiments show AffPCL outperforms independent training and matches or exceeds FedAvg across varying heterogeneity levels. Key contributions include the PCL formulation, AffPCL algorithm, convergence guarantees, and empirical validation.

### Strengths
1. This paper is clearly written, guiding the reader from the simplest FL algorithm to fully personalized FL with variance reduction method. The authors clearly motivate why this is needed (e.g., common global models can be suboptimal under heterogeneity).

2. The proposed algorithm *AffPCL* is conceptually simple yet effective. Each agent’s update is adjusted via principled bias correction and importance weighting to account for differences in objectives and data distributions.

### Weaknesses
1. The finite-sample bounds presented in this paper appear quite loose. For instance, Prop. 1 provides an error bound for the vanilla heterogeneous FL algorithm of $\\tilde{O}(\\kappa^2 t^{-1}n^{-1})$, while Prop. 2 states the error bound for AffPCL as $\\tilde{O}(\\kappa^2 t^{-1}\\max\\{n^{-1}, \\delta_{obj}\\})$, with $\\delta_{obj} \in [0,1]$ representing the effect of objective heterogeneity. Since $\\max\\{n^{-1}, \\delta_{obj}\\} \geq n^{-1}$, the bound in Prop. 2 is no tighter than Prop. 1. The same argument applies to the bound in Thm 1.

2. The experiments compare AffPCL *only* against FedAvg (which provides a single global model) and independent learning (no sharing). These are the very foundational algorithms without considering personalized collaborative learning, and the paper does not include any comparisons to existing personalized federated learning methods. There is a rich literature on approaches like per-user fine-tuning, model interpolation, or clustered federated learning.

### Questions
1. In the AffPCL algorithm, each agent must perform bias-corrected local updates and compute importance weights for others’ data contributions. How does the computational or communication overhead from this AffPCL algorithm affect its complexity?

2. I find the authors’ argument in Lines 240–249 unclear. Could you clarify how the affinity-based variance reduction specifically benefits the proposed AffPCL algorithm?

### Soundness
2

### Presentation
3

### Contribution
3
