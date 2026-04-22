# MaskCO: Masked Generation Drives Effective Representation Learning and Exploiting for Combinatorial Optimization

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 4, 4, 8

## Abstract
Neural Combinatorial Optimization (NCO) has long been anchored in paradigms such as solution construction or improvement that treat the solution as a monolithic reference, squandering the rich local decision patterns embedded in high-quality solutions. Inspired by the scalability of self-supervised pretraining in language and vision, we propose a shift in perspective: Can combinatorial optimization adopt a fundamental training paradigm to enable scalable representation learning? We introduce MaskCO, a masked generation approach that reframes learning to optimize as self-supervised learning on given reference solutions. By strategically masking portions of optimal solutions and training models to recover the missing content,  MaskCO turns a single instance-solution pair into a multitude of local learning signals, forcing the model to internalize fine-grained structural dependencies. At inference time, we employ a mask-and-reconstruct procedure, i.e., a refinement loop that iteratively masks variables and regenerates them to progressively improve solution quality. Our findings show that these learned representations are highly transferable, facilitating effective fine-tuning and boosting the performance of alternative inference approaches. Experimental results demonstrate that MaskCO achieves remarkable performance improvements over previous state-of-the-art neural solvers, reducing the optimality gap by more than 99% and achieving a 10x speedup on problems such as the Travelling Salesman Problem (TSP).

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents MaskCO, a masked generation paradigm that formulates the learning process of neural combinatorial optimization (NCO) as a solution-level self-supervised learning framework. Specifically, it masks part of an optimal solution and reconstructs it to learn fine-grained, localized decision patterns. During inference, the model constructs a complete solution through a mask-and-reconstruct procedure, resembling a local-search-like refinement: in each iteration, certain variables are masked and regenerated, progressively improving the current solution. Extensive experiments on TSP, CVRP, and MIS demonstrate its superiority over previous baselines.

### Strengths
* The topic of this paper is exciting and challenging, exploring a foundational training paradigm that enables effective and scalable representation learning for CO.
* Self-supervised training is an appealing and promising paradigm for NCO.
* The reported training overhead appears lightweight, as shown in Table 23.
* The empirical results are strong.

### Weaknesses
* The authors claim that the proposed masked generation serves as a `foundational paradigm` for NCO. Does “foundational” here imply the goal of developing a foundation model for CO? If so, why not consider the multi-task training setting? Moreover, the paper does not discuss recent efforts toward multi-task or foundation models for CO, such as [1–4].
* The generality of the proposed approach is unclear. This work addresses TSP, CVRP, and MIS, which do not involve complex constraints. Could the proposed method handle more complex constrained VRPs [5] or other CO problems as studied in [1]?
* The training process still requires high-quality solutions as labels. Have the authors explored self-improvement learning as studied in [6]?
* The paper emphasizes representation learning. Could the authors provide a deeper analysis of the learned representations?
* The writing of this paper could be improved:
  * Parts of the introduction appear overly generated (possibly by LLMs). I would expect more direct and concrete opinions from the authors, rather than an overly abstract and ambitious presentation.
  * The descriptions in Sections 3 and 4 are somewhat verbose and make the approach seem more complicated than it is. A clear figure illustrating the overall process would improve readability.
  * It would be helpful to fully elaborate the model architecture in mathematical form.
  * Visualizing how the (partial) solution evolves through the decoding process would make the approach more intuitive.
* Minor comments:
  * Line 79: “instancesolution” → missing space.
  * Line 80: “scalabilityparticularly” → missing space.
  * Line 84: Clarify “BETR and ?”.

[1] GOAL: A Generalist Combinatorial Optimization Agent Learner. ICLR 2025.  
[2] MVMoE: Multi-Task Vehicle Routing Solver with Mixture-of-Experts. ICML 2024.  
[3] RouteFinder: Towards Foundation Models for Vehicle Routing Problems. TMLR 2025.  
[4] UniCO: On Unified Combinatorial Optimization via Problem Reduction to Matrix-Encoded General TSP. ICLR 2025.  
[5] Learning to Handle Complex Constraints for Vehicle Routing Problems. NeurIPS 2024.  
[6] Boosting neural combinatorial optimization for large-scale vehicle routing problems. ICLR 2025.

----

Overall, the studied topic of this paper is exciting and challenging. I believe it makes a valuable contribution to the NCO community, and therefore I recommend acceptance.

### Questions
* Can the proposed method ensure 100% solution feasibility? If so, please explain how. If not, the feasibility rate should be reported in the main experimental table.
* For the TSP case in Section 4.2 (MultiStepDecoding), is the $|U(G)|=m^2$? The proposed approach seems to generate multiple dynamic heatmaps rather than a single static one as in previous methods. If so, why is the inference time of MaskCO significantly lower than that of DIFUSCO and Fast T2T? Moreover, is the decoding process conceptually similar to that used in diffusion-based LLMs?

### Soundness
3

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
The paper proposes MaskCO, a method that masks parts of optimal solutions and trains policies to reconstruct them. Through experiments on the TSP, CVRP, and MIS problems, the authors show the efficacy of their approach.

### Strengths
- The paper is well-written and easy to read.
- The proposed approach, MaskCO, tackles neural CO with a novel approach: mask portions of the solutions to generate more data.

### Weaknesses
- My primary concern with the paper is that it requires expert solutions. In general, the community is moving away from supervised learning based methods towards RL.
- The paper does not consider several SOTA baselines [1, 2].

[1] Grinsztajn et al. Winner Takes It All: Training Performant RL Populations for Combinatorial Optimization, NeurIPS 2023.

[2] Hottung et al. PolyNet: Learning Diverse Solution Strategies for Neural Combinatorial Optimization, ICLR 2025.

### Questions
See weaknesses.

### Soundness
2

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
5

### Summary
This paper introduces MaskCO, a new method that uses a masked generation approach, similar to techniques in NLP and computer vision. Instead of generating a whole solution at once, the model is trained to fill in missing parts of known good solutions, which helps it learn the important local patterns within those solutions. The experiments show that MaskCO works well on three different COP tasks:  the Traveling Salesman Problem (TSP), Capacitated Vehicle Routing Problem (CVRP), and Maximum Independent Set (MIS). During inference, it employs the 2-opt local search to further enhance the performance.

### Strengths
1. The self-supervised paradigm of masked generation in CO is interesting, although it was first noted in [1].
2. The performance, easpecially on large-scale CVRP, is impressive.


[1] Solving Diverse Combinatorial Optimization Problems with a Unified Model.

### Weaknesses
1. The idea of masked generation for CO is similar to that in [1].

2. Some equations and definitions in Sections 3 and 4 could be simplified; currently, they are unnecessarily complicated, which reduces the paper’s readability.

3. The implementation details for the 2-opt heuristic are unclear. How does it, with the use of penalty terms, enforce constraint satisfaction?

4. The effect of adding 2-opt appears minor for TSP but significantly different for CVRP, as shown in Table 11. Most importantly, on CVRP-500 and CVRP-1000, without 2-opt, MaskCO fails to generate feasible solutions. This raises concerns about its adaptability to more complex problems.

5. The hyperparameters (e.g., $K$ and $p$) vary across problem types and sizes, as shown in Tables 12–15. It would be helpful to provide the rationale behind these choices. The results in Figures 2–9 demonstrate that model performance is highly sensitive to these hyperparameters.

### Questions
How is MaskCO scalable? Any designs and empirical results to support this claim?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces MaskCO, a novel and compelling paradigm for Neural Combinatorial Optimization (NCO) inspired by the success of self-supervised masked auto-encoding in natural language processing and computer vision. The paper identifies a key limitation in
existing NCO methods: they typically treat solutions as monolithic objects during construction or improvement, which is data-inefficient and fails to utilise the local substructures embedded within high quality solutions.

To address this, MaskCO reframes the learning problem as solution-level self-supervised
learning. The core of the training methodology involves strategically masking portions of
known optimal or near-optimal solutions and training a model to reconstruct the missing
components. This approach improves data efficiency by transforming a single (instance,
solution) pair into a vast number of (instance, partial solution) training examples.This process compels the model to internalize fine-grained, localized decision patterns.

For inference, the paper proposes a "mask-and-reconstruct" iterative refinement procedure.
This algorithm begins with an initial solution and progressively improves it by repeatedly
masking a random subset of decision variables and using the trained model to regenerate
them in a single forward pass. This process effectively mimics a highly efficient, parallelized
local search.


The authors validate their approach through extensive experiments on three CO problems: the
Traveling Salesman Problem (TSP), the Capacitated Vehicle Routing Problem (CVRP), and the
Maximum Independent Set (MIS). The results demonstrate that MaskCO achieves new
state-of-the-art performance, significantly outperforming prior neural solvers. The paper also shows that the learned representations are highly versatile, transferring effectively to alternative decoding methods and enabling a powerful self-training paradigm that works even without access to optimal solutions.

### Strengths
Novel & Data-Efficient Method: It introduces a new learning paradigm by reframing optimization as a "masked reconstruction" task. This is highly data-efficient, as one optimal solution can be used to create a large number of training examples, forcing the model to learn robust local patterns.

State-of-the-Art Performance & Speed: The model achieves high quality results on TSP, CVRP, and MIS. For example, on TSP-1000, it shows its 9x faster than the previous best neural solver, making its "mask-and-reconstruct" inference a highly efficient, parallelized local search.

High-Quality, Versatile Representations: The learned representations are strong that the model can outperform other methods even when using their decoders. Furthermore, it enables a powerful "optimal-solution-free" mode where the model can teach itself, bootstrapping from weak solutions to high performance.

Significant quality improvements are shown on benchmark datasets such as TSPLIB,

### Weaknesses
Check questions

### Questions
1. What were the parameters used for the baselines? Were they default parameters from the existing papers or tuned for the target task, Eg:- for BQ-NCO Drakulic et al.  Request the authors to clarify this to ensure fairness of the setup. Were the number of training samples and the training samples used same for baselines and proposed method. A discussion on this would clarify this.

### Soundness
3

### Presentation
4

### Contribution
3
