# DIPT: Deep Identification of Propagation Trees in Graph Diffusion

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 2, 6

## Abstract
Understanding how information or influence propagates through a network—such as during an epidemic outbreak or the spread of misinformation—is a fundamental yet challenging problem. Existing methods have largely focused on source localization, overlooking the reconstruction of propagation trees, i.e., the underlying "who-infected-whom" paths that are essential for interpreting diffusion dynamics. We introduce DIPT (Deep Identification of Propagation Trees), a probabilistic framework that infers propagation trees from observed node diffusion states. DIPT models local influence strengths between nodes and leverages an alternating optimization strategy to jointly learn the diffusion mechanism and reconstruct the propagation structure. Empirical results across five real-world datasets demonstrate that DIPT consistently outperforms existing approaches in accurately reconstructing propagation trees.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes a probabilistic framework Deep Identification of Propagation Trees,  to infer propagation trees from final observed node diffusion states, without knowledge of the diffusion mechanism. Empirical results across eight real-world datasets demonstrate that it  outperforms existing approaches in reconstructing propagation trees.

### Strengths
This paper recover actual ”who-infected-whom” propagation tree for a specific diffusion instance, which is limited in prior woks.

### Weaknesses
It is not clear why the path indentification works can't resolve this problem.

### Questions
How to sample the node set s from P(s)? What is the complexity?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper claims that existing study on information propagation does not recover the actual propagation tree and then proposes a probabilistic framework DIPT to infer the propagation trees.

### Strengths
1.	This paper studies an important and heavily investigated research problem, which is critical in real-world applications.

2.	The illustrations are well-designed and helpful in content comprehension.

### Weaknesses
1.	The authors claim existing methods do not recover the “who-infected-whom” paths. However, this is a typical problem in network inference, which has been widely studied, for example, NetInf [1], NetRate [2], NMF [3], and FIM [4]. As far as I know, NMF does not assume any diffusion model. Please further justify this claim.

[1] Inferring networks of diffusion and influence. KDD, 2010

[2] Uncovering the Temporal Dynamics of Diffusion Networks. ICML, 2011

[3] Network Diffusions via Neural Mean-Field Dynamics. NeurIPS, 2018

[4] Scalable Continuous-time Diffusion Framework for Network Inference and Influence Estimation. WWW, 2024

2.	The idea of maximizing a posterior has been utilized early, e.g., NetInf and NetRate. Please justify the fundamental distinction either in the theoretical perspective or the technique perspective for better understanding.

3.	Important related work is not discussed in the related work, and related baselines are not compared in the experiment. Please refer to W1 for examples.

### Questions
Please refer to my weakness comments.

### Soundness
2

### Presentation
2

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
This paper introduces DIPT (Deep Identification of Propagation Trees), a probabilistic framework aiming to infer “who-infected-whom” propagation trees in graph diffusion processes from a single final snapshot of infected nodes, without requiring knowledge of the diffusion mechanism. The method jointly learns a diffusion model and reconstructs propagation structures via an alternating discrete–continuous optimization scheme. A variational prior is used to model source node distributions, and an iterative propagation process learns edge-wise influence strengths.

### Strengths
The problem formulation of propagation tree identification is novel and relevant, bridging the gap between source localization and full diffusion reconstruction.

The probabilistic treatment and alternating optimization are theoretically justified and well-detailed.

DIPT achieves strong results on multiple benchmarks, particularly in sparse and near-tree diffusion settings.

### Weaknesses
As acknowledged in the results, DIPT struggles on networks with dense connectivity or multiple overlapping cascades, where “who-infected-whom” relations are highly ambiguous. The model’s design (monotonic tree updates) inherently favors sparse, acyclic diffusion patterns.

The learned source prior and node features play a critical role; when these correlations weaken (e.g., random or noisy features), accuracy drops substantially (Table 9).

The datasets where DIPT excels often have clear structural constraints or ground-truth trees closely aligned with static network topology, which might inflate apparent improvements.

Although time complexity is analyzed, GPU runtime for very large graphs (e.g., Twitter-scale) may still be prohibitive.

The model outputs a single best tree, without quantifying uncertainty or providing multiple plausible propagation hypotheses.

### Questions
How sensitive is DIPT’s accuracy to the assumed number of propagation steps K? Would over- or under-estimating K lead to systematic errors?

Could the authors provide results or discussion on very dense diffusion graphs (e.g., >0.1 edge density), where the acyclicity assumption may be unrealistic?

Other questions please refer to Weaknesses above.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces DIPT, a novel probabilistic framework for tackling the challenging inverse problem of propagation tree identification. Given only a single, final snapshot of infected nodes in a graph, along with node features, DIPT aims to reconstruct the entire propagation tree. The core of the method is a discrete-continuous alternating optimization strategy that jointly learns an unknown, feature-based diffusion mechanism and infers the latent tree structure without requiring any direct supervision on propagation paths. The method's effectiveness is demonstrated through extensive experiments on eight diverse datasets.

### Strengths
The paper formally defines and addresses "propagation tree identification from a single snapshot," a problem that is substantially more challenging and informative than the well-studied task of source localization. The proposed DIPT framework is technically sound and novel, creatively integrating a learnable influence model, a variational prior for sources, and an alternating optimization scheme into a coherent system. The paper is written with good clarity, making the complex methodology easy to follow. Finally, the empirical evaluation is of good quality, demonstrating state-of-the-art performance across multiple real-world and synthetic datasets against strong baselines.

### Weaknesses
- The model's strong and unexamined dependency on the availability and quality of node features. In the setting of some other works, the feature of the nodes is usually unknown. However, the core influence function, `f_ψ(F_u, F_v)`, is entirely feature-driven. This potentially limits the method's applicability in domains where informative node features are scarce or unavailable (e.g., anonymous networks). The experiments, while extensive, do not sufficiently probe this limitation. For instance, an evaluation and comparison on featureless graphs is missing, making it difficult to disentangle the contribution of the novel optimization framework from the predictive power of the input features themselves.

### Questions
1.  The methodology critically assumes the input is a "final" snapshot of the diffusion process. In real-world applications, how can one reliably determine if an observed state is final? Could you comment on the robustness of DIPT and the validity of the inferred tree if the input is actually an **intermediate snapshot** where diffusion is still ongoing?
2.  Given the model's reliance on node features, what would be its performance on graphs where only structural information and infection indicators are available? A discussion on this would help clarify the method's practical application boundaries.

### Soundness
4

### Presentation
3

### Contribution
3
