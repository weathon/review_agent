# Joint Structure Search for Tensor Network Operators Inspired by Symmetry Breaking

- Avg Score: 5.20
- Decision: Reject
- Scores: 4, 4, 8, 4, 6

## Abstract
Tensor networks (TNs) offer a compact representation for high-dimensional operators in physics and machine learning. While TN structure search (TN-SS) has advanced model selection, prior work is limited to a single operator. Yet real systems, such as transformers and quantum circuits, would contain multiple coupled operators, where treating them independently or enforcing a single shared structure is fundamentally limiting. We introduce joint TN-SS, the first framework for multi-operator structure search. Our physics-inspired algorithm runs in two phases: a symmetry phase, where standard TN-SS finds a shared structure capturing common inductive bias; and a symmetry-breaking phase, where operator-specific diversity emerges through greedy core masking, guided by task-explainable loss tolerances. Across tensor decomposition, parameter-efficient fine-tuning of LLMs, and quantum circuit optimization, joint TN-SS delivers more compact representations with equal or better accuracy than state-of-the-arts, with affordable search cost. These results demonstrate that symmetry-driven diversification offers a simple, general, and scalable solution to TN structure selection in multi-operator systems.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper proposes a new tensor network operator system structure search approach. Inspired by symmetry breaking in physics, they propose a two phase optimization procedure: First, they run normal tensor networ structure search, then they add a regularizer in the structure search optimization problem, which encourages asymmetric task-specific tensor network structures. The regularizer takes the form of a simple core tensor masking. They show that this formulation yields significantly more compact tensor representations in three distinct tensor network settings: Tensor network decomposition, parameter-efficient fine-tuning, and quantum circuit optimization.

### Strengths
Tensor networks seem to be a general enough formulation that this might have a lot of use cases, although I am a bit unsure about it, see the weaknesses section

The results of the proposed algorithm look convincing, consistently yielding good performance.

### Weaknesses
I have two concerns with the current paper:

1.I find the paper quite inaccessible in its current form for readers not already familiar with tensor networks. A better motivation of tensor networks and the structure search setup, with an example formulation for an ML application, would be helpful at the beginning or at least in the appendix of the paper. Something of the form: A tensor network is an expression of the form einsum(A_ij,B_jk,C_kl), with A,B,C being called core tensors and could stand for ... in < application>. 

The metaphor with the Higgs potential also seems unhelpful to me; I don't see how the Higgs potential maps to tensor networks or the structure search problem. In my opinion, as someone not very familiar with this topic, it did not aid my understanding, and the space would be better used for more intuitive motivation and problem setup of tensor network structure search in general, and how it can be useful for an ML practitioner. 


2.The method is specifically designed for high-order tensor networks. While I have no doubt that they are common in computational physics, I am unsure how common these forms of tensor networks are in ML specifically. Could you give some more examples where these methods could be useful in ML? 

PEFT for LLMs is given as an ML example, but also prefaced that it is not intended as a new practical PEFT method. Could you expand on what's missing for a practical algorithm?

### Questions
Instead of the proposed regularizer, could one just directly add the number of parameters into the optimization problem to incentivize more efficient solutions?  

Overall, I am currently unsure about the paper, in particular the relevance for an ML conference, but if the questions are addressed satisfactorily, I am willing to increase my score.

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper introduces a framework for joint tensor network structure search (joint TN SS) in systems that involve multiple linear operators, such as neural network layers or quantum circuit blocks. The authors propose a two-stage algorithm inspired by the concept of symmetry breaking in physics:
- Phase I (symmetry stage): All operators are constrained to share a common tensor network structure. A standard TN SS method is then applied to identify an expressive shared graph G. 
- Phase II (symmetry-breaking stage): Operator-specific variations are introduced through a greedy core-masking process, which removes selected cores as long as the overall loss remains within a user-defined tolerance η_2. The transition between phases is controlled by two tolerance parameters η_1and η_2, reflecting a mechanism similar to the Mexican-hat potential in physics.

The approach is evaluated on three tasks: synthetic tensor decomposition, parameter-efficient fine-tuning of large language models (using the QuanTA method on Llama2 7B), and quantum circuit optimization (including QFT). Across these domains, the method achieves comparable or better accuracy while reducing the number of cores or parameters and lowering search costs compared to existing TN SS baselines such as TNGA, TNLS, TnALE, and Greedy.

### Strengths
- Well-defined problem and strong motivation: The paper clearly addresses the challenge of handling heterogeneous operator structures in multi-operator systems and formalizes this through TNO systems and joint structure search.
- Straightforward and general approach: The proposed two-phase method—starting with shared structure and then introducing diversity—is conceptually simple, easy to implement, and builds on existing TN-SS techniques. The use of core masking in Phase II offers an efficient way to add variability at low cost.
- Interpretability of design choices: Linking the phase transition to task-specific tolerances (η_1,η_2) makes the method more transparent compared to approaches that rely on opaque regularization parameters.
- Comprehensive experimental coverage: The evaluation spans synthetic tensor decomposition, parameter-efficient fine-tuning of LLMs (Llama2-7B across multiple reasoning benchmarks), and quantum circuit optimization, demonstrating the method’s versatility.
- Theoretical contribution: Proposition 3.1 provides a useful criterion for pruning based on operator–Schmidt rank, which is relevant for understanding structural efficiency in tensor networks.

### Weaknesses
- Dependence on a shared graph: The method assumes that a sufficiently expressive common graph Gexists, drawing on the universal approximation property of circuit-like TNs. In practice, selecting Gand its order Lis challenging, and the paper acknowledges failure cases when Lis mis-specified without offering a clear solution. A systematic way to choose Lwould make the approach more robust.
- Phase II is heuristic: The diversification step relies on greedy core masking, which may be sensitive to the choice of cores and can get stuck in suboptimal configurations. Other diversification strategies—such as core splitting, rank adjustment, or permutation changes—are not explored.
- Limited scope in PEFT experiments: While the PEFT section is presented as illustrative, the comparisons are narrow. Stronger baselines like rank-adaptive methods (e.g., CoMERA, GaLore/WeLore) and adapter mixtures (e.g., MixLoRA, DoRA variants) are missing, which limits the strength of the claims.
- Small-scale quantum results: The quantum circuit experiments are restricted to modest sizes (synthetic cases up to Q=12, QFT up to Q=6) and use basic DMRG setups on limited hardware. This leaves open questions about scalability and performance in realistic scenarios.
- Theory lacks practical guidance: Proposition 3.1 depends on constants such as the environment norm C and Lipschitz parameter L, but these are not estimated or validated empirically. The bound may be loose, and the paper does not provide guidance on how to compute or approximate these values in practice.
- Reproducibility concerns: Code for quantum circuit optimization is not yet available and is promised only after publication. Given the importance of that section, early release would improve transparency and allow verification.

### Questions
N/A

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper introduces joint tensor network structure search (joint TN-SS) for systems with multiple linear operators (e.g., multi-layer Transformers, quantum circuits). Inspired by symmetry breaking in physics, the method balances a shared structure with operator-specific diversity to obtain more compact and accurate tensor network operator (TNO) representations. The algorithm proceeds in two phases: (i) symmetry phase, where a standard TN-SS finds a common structure expressive enough for all operators; (ii) symmetry-breaking phase, where greedy core masking introduces operator-specific specialization, controlled by task-explainable loss tolerances (η1, η2). The paper formalizes circuit-like TNOs, proposes a vertex-indexed incidence (VI) matrix to encode structures, and recasts joint TN-SS with an objective whose “negative regularization” plays the role of a mass parameter to trigger diversification. Experiments across synthetic tensor decomposition, parameter-efficient fine-tuning (PEFT) for LLMs, and quantum circuit optimization show more compact structures with equal or better accuracy than baselines at comparable or lower search cost. A perturbation bound further shows low operator–Schmidt rank cores can be masked with no or bounded loss increase.

### Strengths
(1) Novelty in extending TN-SS from a single operator to multi-operator systems with a principled “symmetry to symmetry breaking” transition, yielding a simple, general, and scalable search paradigm.

(2) Clear method design: a formal treatment of circuit-like TNOs, VI-matrix encoding, and a unified objective; core masking as a low-cost structural perturbation; and tolerance thresholds replacing opaque regularization weights to improve explainability and usability.

(3) Broad empirical coverage: in synthetic tensor decomposition, the method substantially reduces core counts while maintaining RSE less than 1e−6; in LLM PEFT, it outperforms QuanTA variants and LoRA/DoRA with the same or fewer parameters; in quantum circuits, it achieves more compact representations than brick-wall baselines at high fidelity and matches classic QFT designs while avoiding SWAP gates.

(4) Efficiency: the two-phase search decomposes an exponential space into a shared-then-specialize process with evaluation cost growing roughly linearly in the number of operators in Phase II; it is markedly more efficient than naïvely composing all structures into one large graph.

(5) Theoretical support: a perturbation bound tied to operator–Schmidt rank offers a sufficient condition for safe masking and clarifies when accuracy is preserved, along with discussion on non-necessity.

### Weaknesses
(1) Quantum-circuit experiments are small-scale and primarily proof-of-concept; scalability to larger Q, richer gate sets, and noisy hardware constraints remains to be demonstrated.

(2) Phase II uses greedy masking: fast and straightforward but potentially suboptimal. There is no systematic comparison with stronger global or metaheuristic searches (e.g., Bayesian optimization), nor an exploration of hybrid strategies.

(3) Tolerance choices ($\eta 1$/$\eta 2$) affect outcomes. While sensitivity analyses are given, there is no adaptive or learned policy for setting them across tasks, which may hinder plug-and-play deployment.

(4) Assumptions in TNO/VI encoding impose specific constraints on dimensions and graph structure; robustness and expressivity under heterogeneous dimensions and tightly coupled cross-layer patterns in real models are not fully assessed.

### Questions
(1) Can Phase II’s greedy masking be combined with stronger global heuristics (e.g., Bayesian optimization or gradient/sensitivity-informed prioritization) to improve optimality under tight budgets? How does accuracy–cost trade off?

(2) Can you propose and evaluate adaptive strategies for $\eta 1$/$\eta 2$ (e.g., validation-curve-based adjustment, early-stopping signals, or a learned threshold predictor) to reduce manual tuning and stabilize performance across tasks?

(3) Do you have generalized perturbation results for simultaneous multi-core masking and inter-operator coupling? Could spectral properties of the environment tensors yield computable criteria for safe pruning?

(4) On larger quantum circuits (bigger Q, constrained connectivity, noise models), and with hardware constraints (depth, two-qubit count, SWAP cost), how do benefits compare to heuristic synthesis/optimization tools?

(5) For LLM PEFT, what happens if search is more tightly coupled to training (e.g., few-step updates after each masking decision) to better reflect downstream generalization? Can compute be controlled via low-fidelity proxies?

### Soundness
4

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
The authors present joint Tensor Network Structure Search, a new framework for discovering optimal tensor network structures when dealing with multiple coupled operators a setting common in large models like transformers or quantum circuits. The method operates in two phases: a symmetry phase, where a shared TN structure captures common inductive biases, and a symmetry-breaking phase, where operator-specific variations emerge through a greedy masking process guided by task-specific loss tolerances. This approach allows for both shared efficiency and individualized expressivity. Experiments across tensor decomposition, LLM fine-tuning, and quantum circuit optimization show that joint TN-SS achieves more compact models with equal or better accuracy than existing methods, all at a reasonable search cost.

### Strengths
S1. The paper introduces a novel extension of TN structure search to multi-operator systems, providing a fresh symmetry-based perspective that effectively balances shared structure and task-specific flexibility. The symmetry breaking concept is very interesting. 

S2. The experimental validation spans diverse domains, from classical tensor tasks to quantum circuits, showing both versatility and efficiency gains.

S3. Proper limitations acknowledgments and related works section.

S4. The authors evaluate the proposed algorithm on diverse domains: joint tensor decomposition, parameter efficient fine-tuning of LLMs, and quantum circuit optimization with improvement results.

### Weaknesses
W1. The experiments on quantum circuits remain small-scale, so it’s unclear how the method performs under more realistic, large-system conditions.

W2. The optimization stability issues mentioned (e.g. with gradient or SVD-based methods) highlight that the current approach may still face robustness challenges in practical deployments.

W3. The significance to the broader ICLR community is not so clear as it the method is specific to Tensor Network Operators. 

W4. The novelty is limited as it borrows on a elegant but relatively simple concept of symmetry breaking.

### Questions
Q1. How sensitive is the performance of joint TN-SS to the balance between the shared symmetry phase and the symmetry-breaking phase, does tuning this trade-off require manual effort?

Q2. Efficiency is mentioned 21 times but can you give some quantitative estimate on the efficiency gains expected within the overall framework proposed here?

Q3. Have you explored alternative optimization schemes to improve robustness, is there an ablation study?

Q4. How transferable are the discovered TN structures across different but related tasks, can a structure found for one operator family generalize to another without retraining?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors provide a Heuristic method for connected tensor operators structure search algorithm, from the intuition of the symmetry breaking. The algorithm provide a efficient way to optimize the tensor structure will considering multiple operators together, and have good performance considering the simplicity.

### Strengths
1. The first paper which have a practiclly useful algorithm for tensor structure search for multi-operator system
2. demonstrate application in LLM finetuning and operator optimization in quantum circuits.

### Weaknesses
1. The narrative need improvement.For example, the author demonstrated the second phases's optimization strategy, while the frist phase, hwo to proposed new G, and the graph optimization procedure is lacking.

### Questions
1. How scalable is this algorithm? What is the complexity in each part of the optimization?

2. For circuit optimization, is that applicable to all kind of quantum operators?

### Soundness
3

### Presentation
3

### Contribution
3
