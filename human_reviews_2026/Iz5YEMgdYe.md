# AstRL: Analog and Mixed-Signal Circuit Synthesis with Deep Reinforcement Learning

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 6, 2

## Abstract
Analog and mixed-signal (AMS) integrated circuits (ICs) lie at the core of modern computing and communications systems. However, despite the continued rise in design complexity, advances in AMS automation remain limited. This reflects the central challenge in developing a generalized optimization method applicable across diverse circuit design spaces, many of which are distinct, constrained, and non-differentiable. To address this, our work casts circuit design as a graph generation problem and introduces a novel method of $\underline{\textbf{A}}$MS $\underline{\textbf{S}}$yn$\underline{\textbf{T}}$hesis driven by deep $\underline{\textbf{R}}$einforcement $\underline{\textbf{L}}$earning ($\textbf{AstRL}$). Based on a policy-gradient approach, AstRL generates circuits directly optimized for user-specified targets within a simulator-embedded environment that provides ground-truth feedback during training. Through behavioral-cloning and discriminator-based similarity rewards, our method demonstrates, for the first time, an expert-aligned paradigm for generalized circuit generation validated in simulation. Importantly, the proposed approach operates at the level of individual transistors, enabling highly expressive, fine-grained topology generation. Strong inductive biases encoded in the action space and environment further drive structurally consistent and valid generation.  Experimental results for three realistic design tasks illustrate substantial improvements in conventional design metrics over state-of-the-art baselines, with 100\% of generated designs being structurally correct and over 90\% demonstrating required functionality.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes AstRL, a deep reinforcement learning approach for automated analog and mixed-signal (AMS) circuit design. The method formulates circuit synthesis as a sequential graph generation problem using PPO with behavioral cloning. Key features include: (1) symmetry-aware action space with structural constraints, (2) simulator-embedded training environment providing ground-truth feedback, and (3) discriminator-based similarity rewards for expert alignment. The approach is evaluated on three circuit design tasks (ring oscillator, comparator, OTA) and achieves 100% netlist validity and >90% simulation validity, outperforming LLM-based baselines.

### Strengths
1. Well-structured paper with clear motivation
2.  Action masking and symmetry-aware modifiers ensure 100% valid circuits by construction
3. Combines multiple techniques (graph generation, PPO, BC, discriminator rewards) effectively

### Weaknesses
1. Core techniques (graph-based circuit generation, PPO+BC, discriminator rewards) are not new. Main contribution is engineering integration rather than algorithmic innovation.
2. No training time, wall-clock time, or simulation budget reported
3. No justification or ablation on GINE architecture

### Questions
1. How is the efficiency of the algorithm with the SPICE simulator in the loop?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes AstRL, a deep reinforcement learning (RL)-driven synthesis method for analog and mixed-signal (AMS) integrated circuits (ICs), which corelies in formulating AMS circuit design as a graph generation problem. Built on a policy-gradient framework (Proximal Policy Optimization, PPO), the method leverages Graph Neural Networks (Graph Isomorphism Network with Edge Features, GINE) to enable fine-grained transistor-level topology generation of circuits. It integrates behavioral cloning (for initial expert alignment) and discriminator-based similarity rewards (for trajectory stability maintenance), while incorporating ground-truth feedback from a simulator-embedded environment to optimize circuit performance. Experiments were conducted on three representative design tasks—ring oscillator (RO), comparator, and operational transconductance amplifier (OTA)—using the Skywater 130nm process. Results demonstrate that 100% of generated circuits are structurally valid, and over 90% meet functional requirements, outperforming state-of-the-art baselines such as AnalogCoder (LLM-based) and AnalogGenie (RLHF-based) in terms of simulation validity and specification fulfillment.

### Strengths
1.	Notable Originality: 
By formulating AMS circuits as a graph generation problem and enabling transistor-level fine-grained generation via RL, the work overcomes limitations of existing LLM-based methods (which lose structural information) and RLHF-based methods (which lack real simulator feedback), establishing a dual-drive mechanism of "expert alignment + exploratory optimization".
2.	Solid Methodological Quality: 
Each module—from graph representation (GINE network) and action space (symmetric modifiers + masking) to reward design (three-stage categorized rewards)—is supported by theoretical foundations and validated through ablation studies, without "black-box" design. Experiments use real-world processes and standard tasks, ensuring high result reproducibility.
3.	Strong Domain Significance: 
It addresses "structural validity" and "functional compliance" issues of industrial concern. The released code and datasets can drive subsequent research in the field (e.g., multi-module integration of complex AMS circuits and adaptation to larger-scale processes).
4.	High Expressive Clarity: 
Framework diagrams, formulas, and experimental tables (multi-task comparison in Table 1) complement each other. The comprehensive overview of related work not only demonstrates an understanding of the field’s history but also clarifies the paper’s positioning, reducing the comprehension burden for reviewers and readers.

### Weaknesses
1.	Limited Task Generalization: 
Experiments only cover three basic AMS circuits (RO, comparator, OTA) and do not validate more complex industrial-grade circuits (e.g., high-speed SERDES, phase-locked loops (PLLs)). This makes it impossible to determine the method’s adaptability to scenarios with "multi-module coupling and high nonlinearity", restricting the generalization of conclusions.
2.	Lack of Mass Production Metrics: 
Existing results focus on "functional correctness" (e.g., frequency, delay, gain) but do not evaluate industrial-critical mass production metrics such as area, power consumption, and yield. The absence of actual tape-out test results for generated circuits makes it difficult to assess the method’s practicality in mass production.
3.	Insufficient Analysis of Expert Data Impact:
Behavioral cloning relies on a dataset of 1,172 expert circuits, but the impact of "expert data scale/diversity" on model performance is not analyzed (e.g., whether the initialization effect of behavioral cloning degrades significantly when data volume is reduced by 50%). This hinders the method’s application in small-data scenarios.
4.	Unmentioned Computational Efficiency: 
No indicators of computational efficiency—such as model training time and average time for generating a single circuit—are reported. AMS design requires rapid iteration; if generating one circuit takes several hours, the method may fail to meet industrial rapid-iteration needs, necessitating supplementary efficiency analysis.
5.	Insufficient related work: 
More research for circuit synthesis such as [1] [2] should be discussed and compared in related work or experiments.

[1]. Bai Y, Wang J, Chen L, et al. A Graph Enhanced Symbolic Discovery Framework For Efficient Logic Optimization. The Thirteenth International Conference on Learning Representations.

[2]. Wang Z, Wang J, Yang Q, et al. Towards next-generation logic synthesis: A scalable neural circuit generation framework. Advances in Neural Information Processing Systems, 2024, 37: 99202-99231.

### Questions
1.	Has your team attempted to apply AstRL to more complex industrial-grade AMS circuits (e.g., PLLs, SERDES)? If not, could you analyze potential challenges (such as increased trajectory depth and sparse reward signals) that the method may face in scenarios with "multi-module coupling and high nonlinearity"?
2.	The paper does not evaluate mass production metrics (area, power consumption, yield) of generated circuits. Do you plan to supplement tape-out tests or related simulations in the future? These metrics are critical for industrial deployment—could you explain the potential of the existing method in optimizing mass production metrics?
3.	For the dataset of 1,172 expert circuits used in behavioral cloning, how do the scale and diversity of the data impact model performance? If the volume of expert data is reduced (e.g., to only 100 circuits), will the initialization effect and final performance of the model degrade significantly?
4.	When comparing with AnalogGenie, your team supplemented "netlist conversion + Bayesian optimization" for simulator validation. Were the parameter settings of these two steps (e.g., number of iterations and hyperparameters for Bayesian optimization) consistent with the original AnalogGenie method? If there are differences, could they affect the fairness of the comparison results?
5.	What is the computational efficiency of the model? For example, how long does it take to train the model for the three tasks on a single GPU? What is the average time required to generate an OTA circuit that meets specifications? Could you compare its efficiency with existing methods (e.g., AnalogCoder)?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes an RL-based method to generate optimized analog circuits that meet given targets. Three downstream tasks are evaluated to show the performance of the proposed RL method.

### Strengths
Leveraging RL to search optimal-sized circuit topologies is exciting.

The circuit-level graph modeling by considering rich domain knowledge is well presented.

The RL framework is also introduced well, with detailed reward design, which is the key to this framework.

### Weaknesses
The scalability of the method is limited, especially when the number of devices in analog circuits increases.

The training of the method is very time-consuming. There are two searching loops in the framework. First, the RL will search topologies, and meanwhile, the reward generation needs to search for optimal device parameters. The latter could be even time-consuming. This makes the framework unlikely to be practical.

Evaluations and comparisons are pretty unfair. The three circuits for evaluations are quite small compared to the circuits in Analogcoder and AnalogGenie. If you focus solely on these three circuits and conclude that your RL method achieves the state-of-the-art (SOTA), it is not convincing.

### Questions
How is each circuit designed? Is a separate agent required to be trained to achieve this purpose?

What do the different normal distribution curves represent in Figure 4d, e, f?

What is the training cost? 

What is the initial starting point of each episode in your RL agent, and how do you determine the end of each episode?

If you just use this RL to achieve the given design target, why not pick up an existing topology?

### Soundness
2

### Presentation
2

### Contribution
2
