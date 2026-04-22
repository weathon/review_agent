# Multiphysics Bench: Benchmarking and Investigating Scientific Machine Learning for Multiphysics PDEs

- Avg Score: 5.50
- Decision: Reject
- Scores: 6, 8, 6, 2

## Abstract
Solving partial differential equations (PDEs) with machine learning has attracted great attention, as PDEs are fundamental tools for modeling real-world systems that range from fundamental physical science to advanced engineering disciplines. Most real-world physical systems across various disciplines are involved in multiple coupled physical fields rather than a single field. For example, in 3D integrated circuits (ICs), electrical current injection or electromagnetic wave propagation can induce localized heating, which in turn alters the electromagnetic properties of the embedded components. However, previous machine learning studies mainly focused on solving single-field problems, but overlooked the importance and characteristics of multiphysics problems in real world. Multiphysics PDEs typically entail multiple strongly coupled field quantities, thereby introducing additional complexity and challenges, such as inter-field coupling. Nevertheless, benchmark testing for the application of machine learning in solving multiphysics problems remains largely unexamined. To identify and address the emerging challenges in multiphysics problems, we make three main contributions in this work. First, we collect the first general multiphysics dataset, the Multiphysics Bench, which focuses on multiphysics PDE solving with machine learning. Multiphysics Bench is also the most comprehensive multiphysics PDE dataset to date, featuring the broadest range of coupling types, the greatest diversity of multiphysics PDE formulations, and the largest scale of coupled physics data. Second, we conduct the first systematic investigation on multiple representative learning-based PDE solvers, such as Physics-Informed Neural Networks (PINNs), Fourier Neural Operators (FNO), Deep Operator Networks (DeepONet), and DiffusionPDE solvers, on multiphysics problems. Unfortunately, naively applying these existing solvers usually shows very poor performance for solving multiphysics. Third, through extensive experiments and discussions, we report multiple insights and a bag of useful tricks for solving multiphysics with machine learning, motivating future directions in the study and simulation of complex, coupled physical systems. Notably, our multiphysics data enables PDE solvers to incorporate more comprehensive physical laws, leading to more accurate solutions to real-world problems.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces the first general multiphysics dataset, the Multiphysics Bench, for PDE solving. It includes six canonical coupled systems (e.g., electro-thermal, thermo-fluid, magneto-hydrodynamic, etc.) generated via finite element simulations. The authors evaluate four representative neural PDE solvers, PINNs, FNO, DeepONet, and DiffusionPDE, on the Multiphysics Bench.

### Strengths
1. The paper is well-written, clearly motivated, and easy to follow.
2. Multiphysics problems are largely underexplored in SciML benchmarks. While many benchmarks exist for single-physics problems, multiphysics systems are far more representative of real-world challenges but have lacked a standardized dataset for evaluation.
3. The experiments are systematic and provide valuable insights. The failure of general-purpose PDE solvers like FNO stresses the importance of investigating into new PDE solvers for multi-physics PDEs.

### Weaknesses
1. Since the Multiphysics Bench's focus is on coupled PDEs, the evaluation would be significantly stronger if it included at least one baseline specifically designed for such coupled systems. One possible solver might be [1].
2. The paper's primary contribution appears to be the introduction of new datasets. The work would be substantially strengthened by the inclusion of a novel PDE solver tailored to the specific challenges these datasets present.
3. The citation style is incorrect for ICLR guidelines (\citep should be used for parenthetical citations). 

[1] Xiao, X., Cao, D., Yang, R., Gupta, G., Liu, G., Yin, C., ... & Bogdan, P. (2023). Coupled multiwavelet neural operator learning for coupled partial differential equations. arXiv preprint arXiv:2303.02304.

### Questions
1. What do you think is the difference between coupled PDE in a single-physics domain compared to multi-physics one?
2. The authors conduct experiments on the 'Complete vs. Incomplete Physical Priors' section to prove the significance of understanding coupling PDEs instead of focusing on the sliced PDEs. However, the distinction between complete and incomplete prior isn't that obvious. Could authors provide stronger proof about the importance of physcial priors here? And also, could the authors provide an explanation on why DeepONet works better with incompete physical prior.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper introduces a multi-physics benchmark dataset. 

The paper describes $6$ paired tasks and evaluates the performance with $4$ models. 

The paper also demonstrates that the current models' performance may suffer when naively applying the method to this dataset and proposes two strategies to alleviate the problem. 

The paper provides data and the script to generate the data, and the baseline code (at least planned).

### Strengths
A new problem that has practical implications for modeling multi-physics systems.

### Weaknesses
It may not be of general interest.

### Questions
A fundamental question is if this class of problems can be reduced to a single PDE and the conditions under which it is possible to do this. 

Another interesting question is that if we had a unique PDE, what would make a problem more difficult to solve?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces a novel benchmark (Multiphysics Bench) for evaluating Scientific Machine Learning methods applied to coupled multiphysics Partial Differential Equations (PDEs), consisting of six canonical coupling problems that cover diverse scenarios like bidirectional/unidirectional coupling and both steady-state/frequency-domain and transient dynamics. A robust baseline is established by rigorously testing four machine learning approaches: Physics-Informed Neural Networks (PINNs), Fourier Neural Operators (FNO), DeepONet, and DiffusionPDE.

### Strengths
1. This paper systematically focuses on the critical and complex domain of coupled multiphysics PDEs. The inclusion of diverse and challenging coupling types (bidirectional/unidirectional, equation/parameter-level) demonstrates an original and comprehensive problem formulation that accurately reflects real-world engineering and science.
2. The authors use four leading learning-based PDE solvers (PINN, FNO, DeepONet, and DiffusionPDE) on all six problems
3. The problems are deliberately chosen to expose specific technical weaknesses of current models, such as gradient inconsistencies in PINNs and mode collapse in FNOs when applied to coupled systems.
4. The generation of the dataset, particularly its size ($10^4$ training samples) and the complexity of the output fields (e.g., 12 output channels for Acoustic-Structure coupling), represents a significant benchmark for training modern learning-based PDE solvers.

### Weaknesses
1. The paper needs to explain why the relative $L_2$ error performance of baseline models remains unchanged (or nearly unchanged) despite a significant increase in the data scale (number of training samples) in Table 4.
2. The paper correctly identifies various failure modes (e.g., FNO mode collapse, PINN gradient inconsistency), but the justification is often descriptive rather than quantitatively analytical.

### Questions
See the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents Multiphysics Bench, a dataset and benchmark suite for evaluating SciML PDE solvers on multiphysics problems. It consists FEM-generated data for a collection of six multiphysics problem (electro-thermal, thermo-fluid, electro-fluid, magneto-hydrodynamic, acoustic–structure, mass-transport–fluid). The paper also evaluates four classes of SciML solvers (PINNs, FNO, DeepONet, DiffusionPDE), reports empirical findings, and suggests a couple of "tricks" to help improve performance.

While the paper presents a solid engineering effort, it offers limited scientific insights or conceptual contribution. It finds that SciML tools behave the same way on larger, coupled PDEs, which is interesting to note, but it neither advances our understanding of the unique challenges presented by multiphysics over single-physics problems for SciML methods, nor offer any deeper insights into the generalization of SciML methods.

### Strengths
- Multi physics problems are widely present in the real-world. 

highly Creating a standardized, multi-scenario multiphysics benchmark is valuable for the SciML community; the chosen scenarios are relevant to real applications (electronics, fluid/thermal systems, acoustics, porous-media transport).
- Many different PDEs are included in the benchmark with different initial and boundary conditions, although there are a few omissions.
- Evaluation of four major solver families (PINNs, DeepONet, FNO, DiffusionPDE), although there are a few omissions. The paper also reports many metrics (RMSE, relative L2, MaxError, etc.)
- The paper alludes to practical issues such as imbalance of residual magnitudes and degradation in transient tasks and suggests remedies like quantile normalization and auto-balanced weighting.

### Weaknesses
Major Weakness: The scientific contribution of the paper is unclear.  

- The multiphysics benchmark dataset does not focus on what sets multiphysics problems apart from single-physics problems (cross-field interactions, different spatio/temporal scales across fields, etc.). Simply measuring global reconstruction error does not inform us why SciML methods might fail on multiphysics problems or how to fix them? 

- The empirical takeaway is essentially that existing SciML models for PDEs generalize to multiphysics problems about as well as they do for single physics problems. This is neither surprising not theoretically illuminating. In fact, it suggests that the multiphysics coupling is weak or moderate. So, the benchmark does not isolate multiphysics difficulty.

- The benchmark currently emphasizes steady-state (frequency-domain) systems, and has only one transient case. So, the temporal diversity is limited and excludes a class of time-dependent or dynamically unstable multiphysics systems that are highly relevant in practice. So, the benchmark does not fully capture the challenges of transient coupling, time-scale stiffness, or nonlinear feedback dynamics, which are common in many real-world multiphysics problems.

Other Weaknesses:

- SciML methods are known to be very brittle and sensitive to hyperparameters (see [1]). The empirical results in the paper may not be reliable. There are no confidence intervals in the majority of the results (except Figure 9).

- The abstract and the introduction point to insights and "bag of tricks". These insights do not appear to be different from single physics scenarios, and the "bag of tricks" are not substantial enough to be included in the main paper.

- The benchmarks do not capture real-world operational challenges, where measurements are noisy, PDE parameters are not known exactly, PDEs are approximations of the underlying physical phenomenon. The utility of evaluating on a benchmark that does not incorporate the such real-world challenges is unclear.

[1] McGreivy, Nick, and Ammar Hakim. "Weak baselines and reporting biases lead to overoptimism in machine learning for fluid-related partial differential equations." Nature Machine Intelligence 6, no. 10 (2024): 1256-1269.

### Questions
- How sensitive are the SciML methods to hyperparameter tuning? Would deeper FNOs, or PINNs with different loss weights change the conclusions?

- Do any of the SciML methods violate conservation laws in multiphysics problems? If yes, can the violation be quantified?

- Do PINN residuals overfit to one physics and underfit to another?

- How would foundation models for PDEs like Poseidon [2] perform?

- There is a claim in the introduction: "By aligning computational solvers more closely with the governing physical laws, MultiphysicsBench provides a robust foundation for developing models that are more accurate, resilient, and generalizable—paving the way for future advances in simulating complex, coupled physical systems." This is not apparent to the reviewer. Can you elaborate how the contributions of the paper aligns with this claim?

[2] Poseidon: Efficient Foundation Models for PDEs, NeurIPS 2024

### Soundness
2

### Presentation
3

### Contribution
2
