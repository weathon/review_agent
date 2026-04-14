# On the Benefits of Memory for Modeling Time-Dependent PDEs

- Decision: Accept (Oral)
- Scores: 8, 6, 8, 8

## Abstract
Data-driven techniques  have emerged as a promising alternative to traditional numerical methods for solving PDEs. For time-dependent PDEs, many approaches are Markovian---the evolution of the trained system only depends on the current state, and not the past states. In this work, we investigate the benefits of using memory for modeling time-dependent PDEs: that is, when past states are explicitly used to predict the future. Motivated by the Mori-Zwanzig theory of model reduction, we theoretically exhibit examples of simple (even linear) PDEs, in which a solution that uses memory is arbitrarily better than a Markovian solution. Additionally, we introduce Memory Neural Operator (MemNO), a neural operator architecture that combines recent state space models (specifically, S4) and Fourier Neural Operators (FNOs) to effectively model memory.  We empirically demonstrate that when the PDEs are supplied in low resolution or contain observation noise at train and test time, MemNO significantly outperforms the baselines without memory---with up to $6 \times$ reduction in test error. Furthermore, we show that this benefit is particularly pronounced when the PDE solutions have significant high-frequency Fourier modes (e.g., low-viscosity fluid dynamics) and we construct a challenging benchmark dataset consisting of such PDEs.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This work applies a state space model to model the memory term of a PDE that (under assumption) decomposed by the Mori-Zwanzig theory. The method is mostly well-motivated and demonstrates improvement in solving PDEs in low resolution. Overall the paper addresses an important issue in the field of neural operators.

### Strengths
1. The paper is theoretically well motivated. 
2. The proposed method for memory operators can be implemented easily.
3. The method demonstrates improvement in performance for solving PDEs.

### Weaknesses
1. The motivation for using S4 as the model of choice is not convincing.
2. Lack of numerical comparison against benchmarks in 2D cases.

### Questions
1. The main question I have is regarding the flexibility of the proposed method. The author showcased that the memory term modeled by S4 achieves improved performance, but in Appendix G, the memory term modeled by LSTM can be almost just as good. I hope the authors can also highlight this in the main text as I think this shows that adding the memory term is important, no matter the architecture. 
2. Related to the above, can similar memory terms be added to the other benchmarks like U-Net neural operator? If so this can also be highlighted as an ablation study such that the importance of the memory term can be further recognized. 
3. When the high-frequency mode matters, would it simply be more effective to increase the complexity of FFNO rather than applying the memory term? Can the author briefly discuss how this trade-off affects the choice of using a memory term (or not) in this case?
4. How well can the model perform for zero-shot super-resolution once the model is trained?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper analyzes the advantages of incorporating memory into AI models for solving time-dependent partial differential equations (PDEs). Traditional AI-driven approaches often rely on a Markovian assumption, where future states are predicted solely from the current state. However, the authors argue that including memory (i.e., information from past states) improves model accuracy, particularly when solutions are low-resolution or contain noise.

The authors further present the Memory Neural Operator (MemNO), a novel architecture that combines state space models (namely S4) and Fourier Neural Operators (FNOs) to capture both spatial and temporal dynamics effectively. Theoretical foundations based on the Mori-Zwanzig theory suggest that memory-inclusive models can outperform Markovian models in specific scenarios, particularly with high-frequency Fourier components. Empirical results show MemNO achieves up to 6x lower test error compared to memoryless models.

### Strengths
* This paper seeks to formalize and theoretically analyze the role of memory in learning time-dependent PDEs, distinguishing itself from prior studies that have approached memory in this context primarily from an empirical standpoint.
* The experiments and ablations are thorough and well-executed. Additionally, the empirical results are presented using relative errors, which is exemplary and should be the standard in this field. Unfortunately, this point deserves emphasis, as many other studies overlook this.
* the paper is well written and clearly structured
* the paper introduces recent state-of-the-art state-space models to the field of learning PDE solutions

### Weaknesses
* The paper claims to improve learning the solution of time-dependent PDEs theoretically grounded in the 
    the Mori-Zwanzig formalism. However, equation 6 of the paper clearly states that the 'memory term' (i.e., second summand on the right hand side of the equation) involves the differential operator of the PDE, i.e., a local spatial operator. But the proposed method does not take into account spatial interactions and applies the memory cell (S4 layer) to each element of the spatial dimension independently. This is clearly not covered by the formalism and should be stated very clearly. That being said, it would be interesting to incorporate spatial interactions via Graph Neural Networks acting on the nearest neighbors for instance.
* The experiments are interesting and extensive. However, there are some issues that need to be addressed. It's not clear up to which extend the performance increase is due to the memory or simply due to the very expressive S4 model. One good way to analyze this empirically would be to plot the performance of the proposed method on the y-axis, while on the x-axis the size of the look-back window is varied, i.e., how many past instances are fed into the memory cell. Currently, all available past instances are fed to the memory cell. If the performance does not change significantly by varying that, this would indicate that memory is not as important and the performance difference can be explained by the different architectures. Figure 7 in the appendix already suggests that the architecture choice influences the performance.
* A general weakness: Why do we need an advanced AI model with 5M parameters only to solve a simple 1D Burgers equation that can be solved in milliseconds with traditional methods? I acknowledge that the aim of the paper is to improve AI models applied to PDEs. However, it would still be important to draw some connections to traditional methods, which in particular for the experiments considered in this paper are vastly superior. A good starting point to look into this is the referenced paper by McGreivy and Hakim (2024).

### Questions
See weaknesses

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors propose a variant of a factorized Fourier neural operator with memory and show how it can improve over the standard Markovian neural operator for modeling PDEs with small grid sizes or noise added to the observations. Additionally, some theory motivated by the Mori-Zwanzig theory of model reduction is given to motivate and explain the success of the addition of memory (Eq. 6). This success in certain regimes is observed in experiments.

### Strengths
1) The problem of whether to use memory to model time-dependent PDEs is well-defined and well-motivated.
2) The theoretical motivation (Eq. 6) and example (Section 4) are clear, concise, and show the possible situations where adding memory in Neural Operators could improve performance.
3) The two experiments varying resolution and additive noise nicely demonstrate regimes where memory can be helpful. Figure 6 also provides good context in this regard.

### Weaknesses
1) There should be a brief discussion about the algorithmic cost of adding memory to the FNO. The authors use the diagonal matrix parameterization of S4, so this should be a reasonable increase in cost. Empirical timing would also be helpful to make this point.
2) In the added noise experiments (Section 6.2) it is expected that memory improves performance because the memory unit can approximately filter the input data. The discussion surrounding this experiment is less fleshed out than that of the resolution experiments in Section 6.1 and does not have an obvious connection to the theory and example presented in sections 3 and 4. What practical situation is this intended to model?
3) lines 1119-1121: this is a strange training setup, but maybe necessary due to the GPU memory limitations.

### Questions
See Weakness 2.

### Soundness
3

### Presentation
3

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
The paper proposes a new form of neural operators (NOs) that are inspired by the Mori–Zwanzig (MZ) formalism. This operator is to model spatio-temporal dynamics that arise in time-dependent (nonlinear) partial differential equations (PDEs). The paper points out that the previous roll-out-of-NOs which is caused by memorylessness and, thus, proposes to implement the memory term similarly in the MZ formalism. The specific example architecture has been investigated: Fourier NO + S4, which are empirically shown to be more effective in capturing the dynamics. Moreover, the paper points that the previous benchmark PDEs are easy (including relatively only frequency) and, thus, proposes a new set of benchmark datasets that gives challenge to previous NOs.

### Strengths
The paper is well-motivated and proposes solution methods as well as a new set of datasets.

The paper is well-written, providing necessary details without going into too much detail and explaining why including the memory term can be beneficial in modeling complex dynamics. 

The paper introduces a new challenging dataset (although not extensive), the dataset could be used as a next important benchmark for future works.

The empirical results seem to show that the proposed method is more effective in modeling the dynamics. Interestingly, the results are somewhat aligned with the effect of the MZ formalism in heterogeneous systems; that is, on a coarse mesh (on a coarse-grained or homogenized system in MZ), the Markovian surrogate model can be aided by the memory term to capture the influence the state defined on a finer mesh (the eliminated variables in MZ).

### Weaknesses
It is not easy to see exactly how the architecture is designed. Including diagrams showing the forward computational path will be very helpful. 

Although taking a cue from the MZ formalism is good, some discussions on the related topic such as closure modeling for surrogate models seems to be missing.

FNOs are known to select low-frequency modes and are expected to struggle with high-frequency solutions. Are transformer-based neural operators expected to behave similarly? Could the authors provide why GKT performs poorly? 

Also, could the authors provide experimental results with some advanced transformers-based neural operators? such as OFormer (Li, et al, TMLR, 2023) and their variants (for example, FactFormer, Li et al, NeurIPS 2023, 2023)

### Questions
Please see the questions in the weaknesses above.

### Soundness
3

### Presentation
3

### Contribution
3
