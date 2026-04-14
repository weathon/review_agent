# Hybrid MILP to efficiently and accuratly  solve hard DNN verification instances

- Decision: Reject
- Scores: 6, 5, 5

## Abstract
Deep neural networks have demonstrated remarkable capabilities, achieving human-like or even superior performance across a wide range of tasks. However, their robustness is often compromised by their susceptibility to input perturbations. This vulnerability has catalyzed the verification community to develop various methodologies, each presenting a unique balance between completeness and computational efficiency. $\alpha,\beta$-CROWN has won the last 4 VNNcomp(etitions), as the DNN verifier with the best 
trade-off between accuracy vs computational time. VNNcomp however is focusing on relatively easy verification instances (network, inputs (images)), with few {\em unstable nodes}. In this paper, we consider harder verification instances. On such instances, $\alpha,\beta$-CROWN displays a large gap ($20-58$%) between instances that can be verified, and instances with an explicit attack. Enabling much larger time-outs for $\alpha,\beta$-CROWN only improves verification rate by few percents, leaving a large gap of undecided instances while already taking a considerable amount of time. Resorting to other techniques, such as complete verifiers, does not fare better even with very large time-outs: They would theoretically be able to close the gap, but with an untractable runtime on all but small {\em hard} instances.

In this paper, we propose a novel Utility function that selects few neurons to be encoded with accurate but costly integer variables in a {\em partial MILP} problem. The novelty resides in the use of 
the solution of {\em one} (efficient LP) solver to accurately compute a selection $\varepsilon$-optimal for a given input. 
Compared with previous attempts, we can reduce the number of integer variables by around 4 times while maintaining the same level of accuracy. Implemented in {\em Hybrid MILP}, calling first $\alpha,\beta$-Crown with a short time-out to solve easier instances, and then partial MILP for those for which $\alpha,\beta$-Crown fails, produces a very accurate yet efficient verifier, reducing tremendously the number of undecided instances ($8-15\%$), while keeping a reasonable runtime ($46s-417s$ on average per instance).

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This work proposed Hybrid MILP, a neural network verifier which first uses the optimisation based ab-CROWN to solve easy instances before invoking a MILP solver to solve the remaining hard instances. They key to Hybrid MILP is to only encode a subset of unstable neurons using binary variables (i.e., to 'open' them). This subset is chosen based on an upper bound on the individual neurons utility in such an encoding. The resulting method demonstrates strong performance on hard verification instances on small DNNs.

### Strengths
* Novel utility function based on the primal pre-activation values of neurons instead of just their bounds.
* Empirically high effectiveness across a range of neural networks
* Extensive background and discussion of (some) related work.

### Weaknesses
* Lack of ablation studies confirming the importance of the proposed utility function and partial encoding (see below).
* Key parameters of the experimental setup for the main experiment in Table 4 are not discussed, e.g., how was K chosen, how were intermediate bounds computed, how was 'z' chosen for the utility function computation, were different sets used for robustness against different alternative classes
* Lack of theoretical and empirical, comparison (or even mention) of the closely related work on branching heuristics (e.g. Bunel et al, De Palma et al., Ferrari et al. and Henriksen et al.) , similarly trying to estimate the importance of encoding neurons exactly in BaB.
* Lack of comparison to the optimizer's relaxation strategy at equal runtime, i.e., the importance of partial encodings in the first place.
* Applicability only to very small DNNs, which previous work (Ferrari et al.) found to be more easily solvable by LP/MILP based verifiers like ERAN, which were not compared to.
* Overclaims regarding the novelty of Proposition 1, which was discussed in similar form in Singh et al Equation 2 and Salman et al. Theorem 4.2.
* Poor copywriting and large number of typos, including in and the abstract formulas (e.g. Line 174 (half sentence missing), Line 292 (UB - UB -> UB - LB), first expression in Line 359 (three closing but only one opening bracket))

**Minor Comments**
* BaB-based methods such as ab-CROWN and MN-BaB are complete, but the opposite is implied in several places.
* Completeness is binary and can not be traded of with efficiency, but precision (at a given timeout) can.

**References**  
* Ferrari et al. "Complete verification via multi-neuron relaxation guided branch-and-bound."
* Salman, et al. "A convex relaxation barrier to tight robustness verification of neural networks."
* Singh et al. "An abstract domain for certifying neural networks."
* Bunel et al. "Branch and bound for piecewise linear neural network verification."
* De Palma et al. "Improved branch and bound for neural network verification via lagrangian decomposition."
* Henriksen et al. "DEEPSPLIT: An Efficient Splitting Method for Neural Network Verification via Indirect Effect Analysis."

### Questions
1) Can you include experiments using a full-MILP encoding and different partial MILP encodings using other utility functions after ab-CROWN to better assess the contribution of the novel utility function vs. the combination of ab-CROWN with (any) MILP based strategy?
2) How does your partial MILP encoding compare to a full MILP encoding (of unstable ReLUs) in terms of the achieved upper bounds over time? This is particularly interesting, as a popular approach to solving MILP problems is exactly to relax binary variables (automatically and based on the solvers strategies).
3) How is the solve time in Table 3 affected by the choice of neurons? And how do the results change if no exact MILP bounds but only LP bounds are available for Layer 2?
4) Can you discuss the experimental details as per the weaknesses?
5) Can you include a formulation of the inductive definition of the utility function?
6) Can you include experiments on larger networks to investigate the limitations of the proposed method?

**Conclusion**  
The work is a promising new direction for solving hard verification instances on small Networks. However, comparison to key related work is missing and experimental validation is too granular to assess the effectiveness of the novel components proposed in this work (see weaknesses). In addition, the limitations of this work with regards to applicability to larger and perhaps more relevant networks are not investigated. Overall, I believe this work does thus not meet the bar for acceptance at ICLR but I am more than happy to reconsider this assessment should my concerns be addressed.

### Soundness
2

### Presentation
1

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper introduces Hybrid MILP, a method designed to efficiently solve complex verification problems in deep neural networks (DNNs), particularly those involving ReLU-based architectures. The paper addresses the limitations of the current state-of-the-art verifier, α, β-CROWN, which performs well on relatively simple instances but struggles with more challenging ones. The authors propose a hybrid approach that initially applies α, β-CROWN with a short time-out to quickly address easier instances. For instances that remain undecided, Hybrid MILP selectively applies a partial MILP (Mixed Integer Linear Programming) that combines integer variables with linear relaxations for a subset of neurons, reducing the number of integer variables by approximately four times compared to prior methods. This approach effectively narrows the undecided instance rate and is experimentally validated to be both accurate and efficient. Results on benchmark datasets, including MNIST and CIFAR, demonstrate Hybrid MILP’s effectiveness in reducing undecided cases by up to 43%, with a manageable runtime.

### Strengths
-  Hybrid MILP innovatively combines MILP and linear relaxation techniques to handle challenging verification tasks effectively.
- Experimental results highlight substantial improvements in both verification accuracy and runtime efficiency.

### Weaknesses
- No complexity analysis of the proposed method.

- Limited to ReLU activation functions.

### Questions
- Could the proposed method extend to MaxPool nonlinear layer? If can, how is the performance of Hybrid MILP compared to $\alpha,\beta$-CROWN?
- MILP is complete but time-consuming. I am curious about the complexity of the hybrid MILP and why it is more efficient than other baseline?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The authors propose a new algorithm for verifying neural networks against local robustness properties. This method takes a hybrid approach, combining an existing “Branch and Bound” verifier ($\alpha,\beta$-CROWN) with MILP-based verifiers. The authors show that this hybrid approach effectively reduces the number of undecided verification instances for non-robustly trained networks, where verification is known to be hard.

### Strengths
- Apart from a few typos, the paper is clearly written. 
- The proposed approach effectively reduces undecided verification instances for non-robustly trained networks within a reasonable time limit.

### Weaknesses
**Motivation of the work**

Q1. This work focuses on networks that are challenging to verify because they are not robustly trained. Typically, standard-trained networks have low verified accuracy and are vulnerable to attacks. Given that these networks lack robustness, why should we even attempt to verify such difficult cases? For example, in recent papers on certified training [1], networks trained on MNIST and CIFAR-10 show verified accuracies higher than the upper bound in Table 1. Therefore, no sound verifier can achieve verified accuracy beyond what is reported in [1] on these networks. Additionally, even for ReLU networks robustness verification is known to be NP-hard, meaning that in the worst case, complete verification will require exponential time with respect to the number of unsettled ReLU nodes (assuming $ P \neq NP $). So in theory there will always be these hard instances and this was the reason behind certifiable training that makes verification easier.

[1] “Certified Training: Small Boxes are All You Need”, ICLR, 2023.

**Missing related works**

(Lines 296 - 298) The use of two lower bounds (commonly known as the triangle relaxation) within an LP formulation is not new and has been applied in prior works [1] and more recently in [2].


**Technical Contributions:**

The main technical contributions of this work are not entirely clear to me. 

- Q1: Proposition 1 appears to be a well-known result and is used in existing works [1, 2]. The authors should cite the relevant papers.

- Q2: The authors do not provide sufficient detail on the hybrid-MILP algorithm. Authors should include a pseudo-code outlining the key steps of the proposed algorithm.

-  Q3: The high-level idea behind the utility function described in Section 4 closely resembles branching heuristics, such as BaBSR (cited by the authors), in terms of evaluating the importance of an unsettled neuron with respect to a specific verification property. The authors should clarify how their approach differs from existing branching heuristics.


[1] “Formal Verification of Piece-Wise Linear Feed-Forward Neural Networks”, ATVA, 2017.\
[2] “Input-Relational Verification of Deep Neural Networks”, PLDI, 2024.

**Missing Experiments with current SOTA verifiers**
 
$\alpha,\beta$-CROWN is no longer the SOTA verifier for local robustness. The authors should consider providing comparisons with GCP-CROWN and if possible with MN-BaB (on all networks table 4).

**Missing details in the experimental setup**

Q1. The authors should provide the config files (example - https://github.com/Verified-Intelligence/alpha-beta-CROWN/blob/main/complete_verifier/exp_configs/beta_crown/cifar_resnet_2b.yaml) they are using for comparing with $\alpha,\beta$-CROWN including the following details 
- Branching heuristic 
- Batch size 
- Cuts are applied or not (as done in GCP-CROWN)

**UAP/Hyperproperties/Relational property verification:**

Is the proposed approach applicable to verifying robustness against UAP or more general hyperproperties? It appears that prior works [1,2,3] take a similar approach, combining abstract interpretation or bounding techniques with a MILP formulation that is easy to optimize. 

[1] “Towards Robustness Certification Against Universal Perturbations”, ICLR, 2023. \
[2] “Input-Relational Verification of Deep Neural Networks”, PLDI, 2024. \
[3] “Relational DNN Verification With Cross Executional Bound Refinement”, ICML, 2024.



**Typos and Minor comments**

- Line 77 - Missing citation
- Line 174 - incomplete sentence 
- Line 233 - “Box abstraction” - Interval or Box domain is a well-known abstract domain used in traditional program analysis and not introduced by the cited paper. The cited paper introduced DeepPoly a type of symbolic interval domain or restricted polyhedra domain. 
- Line 310 - Typo `nappears`

### Questions
Refer to the Weaknesses section

### Soundness
2

### Presentation
2

### Contribution
1
