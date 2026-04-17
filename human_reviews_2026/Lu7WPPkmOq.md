# Reversible GNS for Dissipative Fluids with Consistent Bidirectional Dynamics

- Decision: Reject
- Scores: 2, 4, 8, 4

## Abstract
Simulating physically plausible trajectories toward user-defined goals is a fundamental yet challenging task in fluid dynamics. While particle-based simulators can efficiently reproduce forward dynamics, inverse inference remains difficult, especially in dissipative systems where dynamics are irreversible and optimization-based solvers are slow, unstable, and often fail to converge. In this work, we introduce the Reversible Graph Network Simulator (R-GNS), a unified framework that enforces bidirectional consistency within a single graph architecture. Unlike prior neural simulators that approximate inverse dynamics by fitting backward data, R-GNS does not attempt to reverse the underlying physics. Instead, we propose a mathematically invertible design based on residual reversible message passing with shared parameters, coupling forward dynamics with inverse inference to deliver accurate predictions and efficient recovery of plausible initial states. Experiments on three dissipative benchmarks (Water-3D, WaterRamps, and WaterDrop) show that R-GNS achieves higher accuracy and consistency with only one quarter of the parameters, and performs inverse inference more than 100× faster than optimization-based baselines. For forward simulation, R-GNS matches the speed of strong GNS baselines, while in goal-conditioned tasks it eliminates iterative optimization and achieves orders-of-magnitude speedups. On goal-conditioned tasks, R-GNS further demonstrates its ability to complex target shapes (e.g., characters “L” and “N”) through vivid, physically consistent trajectories. To our knowledge, this is the first reversible framework that unifies forward and inverse simulation for dissipative fluid systems.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents "reversible-GNS", a graph architecture based on Graph Network Simulator (GNS) able to predict both forward and backwards dynamics. The methods relies on the composition of several invertible mappings: (1) linear encoding and pseudo-inverse decoder, and (2) reversible message passing based on RevNet, resulting in an end-to-end invertibile process. This enables efficient solutions to inverse problems using a standard forward pass, avoiding costly iterative optimization. They provide several examples of dissipative fluid simulations in forward and inverse inference, and goal-conditioned tasks.

### Strengths
* The adaptation of RevNet to message passing GNNs is novel and interesting. 
* The writting is clear and easy to read.
* Comparisons over other state-of-the-art methods are provided, including EGNN, NeuralSPH and GNS baselines.

### Weaknesses
* The examples are not convenient for the method. Learning reversible mappings for irreversible processes is conceptually impossible (more on questions sections). 
* The source code is not ready for review.
* There is a lack of details about the baselines hyperparameters and the dataset physical properties and size.

### Questions
* What is the advantage of this approach compared to using a fully expressive forward model combined with a separate model to learn the inverse mapping? This question is motivated by the design of autoencoders, whose encoder–decoder architecture is typically asymmetric, even when trained to approximate the identity mapping. Such a setup is often simpler to implement and can be more expressive overall.

* Lines 79-80: This is more like a general comment rather than a question. I think that branding the method as "reversible-GNS" is a poor choice. I understand that the authors refer to forward/inverse flexibility, but in the context of physics simulations solving clearly irreversible dynamics with a "reversible" network is very counterintuitive. I think that the correct term mathematically speaking would be "invertible-GNS".

* Section 4.1: The paper provides no dataset details. The original GNS datasets were generated with MPM/SPH simulators, yet the fluid regime is unspecified: is it low-viscosity (Euler-like) or high-viscosity (Navier–Stokes)? This distinction is crucial, since for turbulent or highly viscous flows, reversibility breaks down and the method is unlikely to hold.

* Related to the previous question, I have many concerns about the uniqueness and validity of the proposed approach. As already mentioned in the paper, irreversible dynamics such as diffusion inherently destroys information, making inversion ill-posed. The paper claims applicability to any dissipative fluid, but this is simply impossible. I suspect that the examples provided have low dissipation and are quasi-reversible. I've found the architecture very interesting but the results are presented very naively from the physics perspective. I would have preferred to have a couple of truly reversible dynamics.

* Section 4.2.3: How far in the time horizon can the network inference a state from a given target? Does the error accumulation on integration affect the results? Even though the overall process in reversible, the integration noise is not modelled and might break the reversibility ssumption.

### Soundness
1

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper builds upon the work of Sanchez-Gonzalez et al. (2020) [Learning to Simulate Complex Physics with Graph Networks], which introduced the use of Graph Neural Networks (GNNs) for learning physics simulations from trajectory data. The original goal was to train a model capable of predicting future simulation states (rollouts) from an initial condition, without relying on the original simulator (which is only used for obtaining the training trajectories).

The authors extend this framework by introducing a reversible message-passing mechanism to GNNs. Specifically, they partition node features and design a forward update rule that admits an analytically derived inverse (up to numerical precision). This leads to their proposed model, R-GNS, which they evaluate across a range of tasks.

Interestingly, R-GNS not only supports inversion but also improves forward simulation accuracy. The authors, that for "dissipative fluids, where inversion is inherently ill-posed and optimization-based solvers break down", R-GNS enables "accurate and efficient recovery of initial states" and try to show this by experiments.

### Strengths
The core idea seems conceptually sound, and the most compelling result (in my opinion) is the improved forward simulation performance. Another promising aspect seems to be the semi-symmetric input-output design, which appears to leverage multiple past states via masking. However, this mechanism is underexplained and would benefit from clearer exposition.

### Weaknesses
- The authors emphasize the applicability of R-GNS to dissipative fluids. While their experiments suggest improved invertibility in such systems, it is unclear why this should be possible in principle. This raises concerns about the physical plausibility of their results.
- Several parts of the manuscript are difficult to follow, particularly in Section 4.2.3. It is unclear whether Sep-GNS uses a separately trained reverse model and whether R-GNS uses its inverse mode to predict initial states, followed by forward inference using the respective forward modules.
- The paper does not provide variance estimates for all reported metrics and describe how they are computed.
- The manuscript lacks a thorough discussion of prior reversible GNN architectures, e.g., Li et al. (2021) [Training Graph Neural Networks with 1000 Layers]. A comprehensive literature review would help contextualize the novelty of R-GNS.

### Questions
## Major Remarks

### Claims Regarding Dissipative Fluids

The authors emphasize the applicability of R-GNS to dissipative fluids. While their experiments suggest improved invertibility in such systems, it is unclear why this should be possible in principle. This raises concerns about the physical plausibility of their results.

In particular, the inversion shown in Figure 4b appears to contradict fundamental thermodynamic principles. If such inversion were truly possible, it would imply either:
(a) the system is not genuinely dissipative, or
(b) the authors would have found a method to circumvent the second law of thermodynamics.

A more plausible explanation is that the training, validation, and test sets are biased toward a narrow class of initial conditions, e.g., rectangular "water blocks." It would be valuable to explore greater diversity in the "water block" shapes used in the experiments. For instance, one could consider using circular shapes in the test set while keeping the training set restricted to rectangles, or vice versa. My expectation is that, in the absence of circular shapes during training, R-GNS would likely default to predicting rectangular water blocks when presented with circles in the test set. If this is not observed, I would be very interested in the authors' explanation of how R-GNS manages to generalize effectively in such inversion scenarios.

Additionally, applying R-GNS to non-dissipative systems could help clarify whether its performance is specifically only superior in the dissipative regime or also helps under other conditions.

### Clarity of Experimental Setup

Several parts of the manuscript are difficult to follow, particularly in Section 4.2.3. It is unclear whether Sep-GNS uses a separately trained reverse model and whether R-GNS uses its inverse mode to predict initial states, followed by forward inference using the respective forward modules.

If this interpretation is correct, the following ablations would be valuable:
- Use the **ground-truth simulator** for forward inference from predicted initial states.
- also check for DiffTaichi+SPH whether the inverted "N" and "L" shapes can be recovered using the ground-truth simulator.
- cross-initial state inference: apply R-GNS forward on Sep-GNS inversion outputs, and vice versa.

In Section 4.2.2, the source of the initial state for inference is also not clear. Is it derived from the ground-truth trajectory or from a model-generated forward rollout? If the latter, an ablation using the ground-truth forward state would be informative.

Regarding MSE:
- Is the reported MSE averaged over all test trajectories?
- How comparable are trajectories in terms of particle count? / How comparable are trajectories at all for aggregation?
- How meaningful is MSE as a metric compared to alternatives like optimal transport based ones? Table 4 in the appendix touches on this, but it's unclear whether those results pertain to forward or inverse predictions.

### Variance and Robustness

Please provide variance estimates for all reported metrics and describe how they are computed. How sensitive are results to different training re-runs?

### Experimental results reported

Did the authors run all the experiments in the tables from the different methods themselves or did they copy the results from other papers?

The manuscript lacks clarity on how results compare to Sanchez-Gonzalez et al. (2020). Discrepancies in reported metrics (e.g., Table 1 vs. Appendix C.4 in Sanchez-Gonzalez et al.) should be explained.

### Relation to Reversible GNN Literature

The manuscript lacks a thorough discussion of prior reversible GNN architectures, e.g., Li et al. (2021) [Training Graph Neural Networks with 1000 Layers]. How does R-GNS relate to this and other reversible GNNs? A comprehensive literature review would help contextualize the novelty of R-GNS.

### Bidirectional Training

The concept of bidirectional training is introduced but not well explained. Key questions include:
- How does it differ exactly from standard training regimes? Pseudocode could help here.
- Is bidirectional training applied within the same batch?
- Is there a random use of forward mode or inverse mode training on the provided training trajectories?

Clarifying the training protocol is essential for understanding the method's mechanics and reproducibility.

## Minor Remarks
There furthermore seem to be several typos:
- I guess the initial state in line 138/139 should be $\tilde{\chi}^{0}$ instad of $\chi^{0}$?
- The indexing in Figure 2 is strange: I guess "t:t-k+1" means that states from "t-k+1" up to "t" are taken? Please explain in the manuscript if you introduce such non-standard notation.
- Appendix B and Appendix C are "unattached" to the main text.
- A.0.1 and A.0.2 have strange chapter names. There is no proof at all. 
- The "*" in R-GNS* in Table 4 is unexplained.
- The notation {e} in A.0.2 is strange.
- "Theoretical guarantees of reversibility are formally stated in the Appendix A." ==> I don't see where theoretical guarantees would be shown. A.0.1 seems to be an empirical "guarantee". But even for this further details would be needed.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes a reversible Lagrangian model for learning the inverse dynamics of asymmetric Lagrangian systems. This is a particularly challenging problems and existing works either focus on forward dynamics or rely on computationally expensive optimization based models. In contrast this paper proposes a reversible GNS model to learn the inverse formulation. To handle assymetry the authors propose a semi-symmetric input-output design where the node dynamic quantity is predicted while the other features are masked. The inverse operation is defined as the psuedo inverse of a linear transformation.

### Strengths
1. This is a novel approach to model inverse problems in Lagrangian systems. 
2. The paper is well motivated, with clear explanation of key concepts (reversibility, inverse problem formulation, psuedo-inverses, goal-conditioned generation)
3. The figures are well designed and intuitive to understand.

### Weaknesses
1. The paper does not provide sufficient training details and sufficient ablations on hyperparameters. 
2. Many experiments are presented in a tabular manner, but it's not clear if all the models were trained sufficiently. If R-GNS outperforms GNS with fewer training steps, it should be mentioned. GNS requires 1 million steps to stabilize and 20 million steps to faithfully generalize. It would be good to have a discussion on training stability of R-GNS.

### Questions
1. Figure 3 is a little confusing to me. Unidirectional R-GNS is identical to GNS is it not? In which case, I don't understand why the behavior deviates from GNS. 
2. GNS uses random walk noise to stabilize long-horizon rollouts. In the inverse setting, is it still necessary to add random walk noise? How does noise affect the performance? GNS is extremely sensitive so it would be interesting to see whether the reversible layers affect the noise dependency. 
3. It is not clear to me under what scenarios R-GNS strictly outperforms GNS. Due to high sensitivity to noise, training steps, hyper-parameters etc., it would be good to have a discussion specifying at what point R-GNS starts to outperform GNS, rather than only providing tabular numbers.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces an unified differentiable framework for forward and inverse simulation of fluids. Unlike traditional optimization-based inverse solvers or feed-forward neural simulators that approximate backward dynamics, the proposed approach (R-GNS) enforces bidirectional consistency through a reversible design. The approach combines three main components: a semi-symmetric input-output structure, an invertible linear projection (ILP) encoder-decoder, and a residual reversible message-passing (RRMP) network. The model achieves comparable speed to standard GNS for forward simulation while enabling inverse inference more than 100x faster than optimization-based methods. A few experiments demonstrate improved accuracy, forward-inverse consistency, and stable goal-conditioned control tasks, such as shaping fluids into target geometries. The authors claim this is the first reversible framework to unify forward and inverse simulation for dissipative systems.

### Strengths
- R-GNS achieves lower rollout errors than competing neural simulators (GNS, DMCF, EGNN) across the tested datasets, indicating that enforcing bidirectionality can also improve forward dynamics.
- The paper proposes a coherent framework integrating forward and inverse simulation with a single reversible GNN architecture, reducing parameter count and ensuring consistent physical behavior.
- The idea of enforcing mathematical reversibility (without assuming physical reversibility) is conceptually interesting and well-justified for tackling ill-posed inverse problems in fluids.

### Weaknesses
- The paper lacks detail on how the model was trained relative to the evaluation benchmarks. It is not explicitly stated whether the R-GNS is trained separately for each dataset (WaterDrop, WaterRamp, Water_3D) or jointly across them. This ambiguity makes it difficult to assess generalization capabilities.
- The evaluation focuses on a narrow set of examples: two for forward/inverse simulation and a single goal-conditioned task. While quantitative metrics are provided, the diversity of test cases and ablation analysis is limited, raising concerns about the robustness and scalability of the method beyond these specific datasets.
- The paper does not present applications that would motivate the need for reversible inference in real-world settings. Typical use cases for inverse solvers (e.g., shape optimization, control, or parameter estimation) are absent, making it hard to gauge the broader utility of the approach.
- The paper does not clearly describe the training setup, including dataset composition and train/validation/test splits. Important aspects such as training duration, convergence behavior, and the use (or absence) of data augmentation are missing, making it difficult to assess the reproducibility and robustness of the reported results.

### Questions
- I don't understand why edge features are fixed across time-steps. I assume that they encode the relative positions between particles, and these could be changing (albeit less than compared with the absolute positions) over the simulation. Is this correct or is there something I'm missing?
- Could you better elucidate how the data was prepared for training? How long did the model took to converge compared to the other approaches?

### Soundness
2

### Presentation
2

### Contribution
3
