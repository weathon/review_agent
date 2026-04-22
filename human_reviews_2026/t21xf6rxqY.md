# STDDN: A Physics-Guided Deep Learning Framework for Crowd Simulation

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 4, 6, 6, 8

## Abstract
Accurate crowd simulation is crucial for public safety management, emergency evacuation planning, and intelligent transportation systems. However, existing methods, which typically model crowds as a collection of independent individual trajectories, are limited in their ability to capture macroscopic physical laws. This microscopic approach often leads to error accumulation and compromises simulation stability. Furthermore, deep learning-driven methods tend to suffer from low inference efficiency and high computational overhead, making them impractical for large-scale, efficient simulations. To address these challenges, we propose the Spatio-Temporal Decoupled Differential Equation Network (STDDN), a novel framework that guides microscopic trajectory prediction with macroscopic physics. We innovatively introduce the continuity equation from fluid dynamics as a strong physical constraint. A Neural Ordinary Differential Equation (Neural ODE) is employed to model the macroscopic density evolution driven by individual movements, thereby physically regularizing the microscopic trajectory prediction model. We design a density-velocity coupled dynamic graph learning module to formulate the derivative of the density field within the Neural ODE, effectively mitigating error accumulation. We also propose a differentiable density mapping module to eliminate discontinuous gradients caused by discretization and introduce a cross-grid detection module to accurately model the impact of individual cross-grid movements on local density changes. The proposed STDDN method has demonstrated significantly superior simulation performance compared to state-of-the-art methods on long-term tasks across four real-world datasets, as well as a major reduction in inference latency.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This work proposes a physics-guided deep learning framework for crowd simulation (STDNN). The authors introduce the continuity equation from fluid dynamics as a strong physical constraint and design a density-velocity coupled dynamic graph learning module. They show that STDNN is significantly superior to simulation performance compared to SOTA methods.

### Strengths
1.	The authors propose a network of time-space decoupled differential equations combined with the continuity equation, which is helpful for predicting the physical laws of trajectories in the macroscopic world.
2.	In experiments, Tables 1 and 2 clearly illustrate the trajectories and verifies main results from the paper.

### Weaknesses
1.	The proposed method uses Neural ODEs to solve $\rho$, but there are many similar ideas, and the use of Neural ODEs in trajectory prediction is also a very common approach.
2.	The proposed method utilizes constraints based on continuity equations, but the specific implementation of this constraint in the Neural ODE framework requires more detailed explanation.
3.	The authors conducted many experiments, but it seems that it is necessary to split each subset of the dataset and compare with newer baselines and methods. The current baseline only reaches the year 2024. Based on the trajectory dataset used by the authors, it seems that there are a large number of sota in pedestrian trajectory prediction that have not been compared.

### Questions
1.	How is the continuity equation incorporated into the Neural ODE solution process? It requires a more detailed explanation.
2.	The detailed parameters used when solving the Neural ODE in torchdiffeq are not disclosed.
3.	Figure 1 contains many typo errors.
⦁	For example, $Gin$($Gout$) should actually be $G_{in}$($G_{out}$).
⦁	The input to Microscopic seems to be $\pho^0$.
⦁	The DDM and CGD in the figure are also too simple.
4.	Should the use of the loss function in Eq 10 be more explicit? Eq 8 does not seem to be included in it.
5.	Are there more granular comparative tests, such as what the results were for ETH/HOTEL/ZARA1/ZARA2/UNIV, respectively?
6.	Should ADE and FDE also be reported for general trajectories?

### Soundness
2

### Presentation
2

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
This paper proposes STDDN (Spatio-Temporal Decoupled Differential Equation Network), a novel physics-guided deep learning framework for crowd simulation. Unlike prior microscopic or purely data-driven approaches, STDDN introduces a Neural ODE formulation guided by the continuity equation from fluid dynamics, thereby coupling macroscopic density evolution with microscopic trajectory prediction. The model integrates three modules — Differentiable Density Mapping (DDM), Continuous Cross-Grid Detection (CGD), and Node Embedding (NE) — to ensure differentiability and physical consistency. Experiments on four real-world datasets (GC, UCY, ETH, HOTEL) show that STDDN significantly improves both simulation accuracy and inference speed compared with state-of-the-art baselines such as SPDiff and PCS.

### Strengths
1、Novel Integration of Physics and Deep Learning:
The paper introduces a principled way to integrate the continuity equation into deep models for crowd simulation. This macro–micro coupling via Neural ODE is both original and physically interpretable.
2、Methodological Sophistication:
The DVCG module cleverly connects density and velocity fields through a graph structure, while the DDM and CGD modules effectively address gradient discontinuity and cross-grid flux detection. These designs are mathematically sound and technically detailed.
3、Interpretability and Physical Consistency:
The approach offers clear interpretability grounded in physics, addressing a key limitation of previous purely data-driven models that violate conservation laws

### Weaknesses
1、The proposed model enforces strict mass conservation through the continuity equation, implying that the total population density within the target spatial domain remains constant over time. However, in realistic datasets and surveillance scenarios, the number of pedestrians in view is not fixed — new individuals may enter the scene, and others may leave. Such open-world dynamics inherently violate the closed-system assumption of the continuity equation. Without explicit treatment of source or sink terms (i.e., inflow/outflow of mass) or adaptive boundary conditions, the model may experience cumulative density drift or numerical instability, particularly when crowd density fluctuates significantly. The authors are encouraged to clarify whether boundary inflows are modeled, or to discuss potential modifications to better handle non-conserved population scenarios.

2、The ablation study provides useful insights, particularly regarding the contributions of the ODE solver and the mass constraint loss. Both components appear meaningful; however, the current experimental setup only uses discrete outputs in the loss computation. As a result, the experiments do not adequately demonstrate the benefit of continuous-time modeling enabled by the ODE formulation. To strengthen this section, I suggest decomposing the “w/o ODE” setting into two variants:
(1)Purely autoregressive training, as mentioned in the paper (“trained using purely autoregressive methods”).
(2)Discrete neural network replacement for ODE, where the ODE solver is replaced with a discrete neural module that still leverages the combined loss function including the mass constraint term.
Such a refinement would better isolate the contribution of the continuous-time ODE formulation from the general modeling capacity and loss design, making the ablation analysis more convincing.

### Questions
Can the authors explain why the fluid physics improves results in low-density datasets like ETH and HOTEL, where the fluid assumption barely holds?

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
4

### Summary
This paper proposes STDDN, a novel framework for crowd simulation that addresses the common issues of error accumulation and physical inconsistency in long-term predictions. Its core contribution is the unique integration of a macroscopic physical law—the continuity equation from fluid dynamics—with a microscopic deep learning model for trajectory prediction. By using a Neural ODE to model crowd density evolution, STDDN enforces a strong physical constraint during training. Experiments show that STDDN not only achieves state-of-the-art accuracy but also significantly reduces inference latency compared to leading methods.

### Strengths
The paper's primary strength is its originality in creating a macro-micro coupled framework. Using the continuity equation to regularize trajectory prediction is a conceptually novel and powerful idea for this field. The quality of the work is good, supported by rigorous and comprehensive experiments that convincingly demonstrate superior performance in both accuracy and efficiency over strong baselines. The paper is also written with exceptional clarity.

### Weaknesses
- **Lack of Direct Physical Metrics**: The paper claims to improve physical realism by avoiding issues like congestion and collisions, but it fails to provide direct quantitative evidence. The evaluation relies on general error metrics (MAE/OT), which are insufficient proxies. The work would be much stronger if it included systematic measurements and comparisons of collision rates, obstacle penetration rates or density extremum analysis to directly support its core claims.
- **Training Cost**: While the paper rightly emphasizes its fast inference speed, it completely neglects to discuss the training cost. The use of a Neural ODE likely makes the training process computationally expensive and slow. An additional analysis should be included in the paper.
- **A minor issue**: the table in page 8 has a wrong caption: "**Figure** 4".

### Questions
- The fluid dynamics assumption is a strong prior. Could you clarify the intended scope of your method? In which crowd scenarios (e.g., panic, counterflow) might this assumption become a limitation?
- Given the model's sensitivity to grid size, can you offer any practical guidelines or a more principled approach for selecting this crucial hyperparameter for new scenes?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes STDDN (Spatio-Temporal Decoupled Differential Equation Network), a novel physics-guided deep learning framework for crowd simulation.
STDDN explicitly combines microscopic trajectory prediction with macroscopic density evolution by embedding the continuity equation from fluid dynamics into a Neural ODE structure.
The model separates local trajectory dynamics from global density fields, enabling physical consistency and stable long-term simulations.
Experiments on four real-world crowd datasets (GC, UCY, ETH, HOTEL) show that STDDN outperforms prior physics-guided baselines such as SPDiff and PCS in both accuracy and inference speed.

### Strengths
1.	Good motivation on coupling of micro- and macro-level dynamics.

The paper’s main contribution is conceptually sound. By using the continuity equation as a bridge between trajectory prediction and density evolution, STDDN unifies local motion modeling with global flow consistency.

2.	Physically meaningful ODE formulation.

The introduction of a Neural ODE to simulate density evolution is well justified. It provides continuous-time reasoning while enforcing conservation principles, addressing a key limitation of purely data-driven models that tend to accumulate errors over time.

3.	Strong empirical performance.

Across four datasets, STDDN shows consistent gains over all baselines, including both physics-based and deep learning methods. The improvements in both accuracy and latency demonstrate that the proposed framework is practically beneficial.

4.	Interpretability and efficiency.

The method retains interpretability through its physically grounded formulation while remaining computationally tractable, which is uncommon in physics-guided models.

### Weaknesses
1.	Limited experimental diversity.

Although the method is tested on multiple datasets, all belong to similar crowd domains. It would strengthen the generality claim to include different physical systems, such as vehicle or swarm simulation.

2.	Ablation breadth.

The ablation study is informative but it would be useful to show how performance changes under different ODE solvers or with alternative coupling strengths between density and trajectory modules.

3.	Minor missing citations for ODE-based trajectory forecasting.

The paper would benefit from acknowledging prior studies that have already explored ODE formulations for trajectory or crowd prediction, such as Social ODE: Multi-agent Trajectory Forecasting with Neural Ordinary Differential Equations (ECCV 2022) and Improving Transferability for Cross-Domain Trajectory Prediction via Neural Stochastic Differential Equation (AAAI 2024).
These works share conceptual overlap in embedding physical dynamics into continuous differential frameworks.

### Questions
Please see the weakness section

### Soundness
4

### Presentation
3

### Contribution
4
