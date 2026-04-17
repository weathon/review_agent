# A$^2$SG: Adaptive and Asymmetric Surrogate Gradients for Training Deep Spiking Neural Network

- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
Training deep spiking neural networks (SNNs) remains challenging due to sharp loss landscapes and temporal inconsistency caused by surrogate gradients.
To address these challenges, we propose a unified framework: adaptive and asymmetric surrogate gradients ($\textit{A$^2$SG}$).
The adaptive gradients adjust an effective window for spatio-temporal adaptation, reducing spatial gradient variation and maintaining directional consistency of gradients over time.
The asymmetric gradients reflect neuronal dynamics by assigning larger gradients to neurons with higher membrane potentials, and we prove that they yield lower variation than symmetric surrogates.
Our analysis further establishes a direct connection between local gradient variation and the curvature of the loss landscape, providing a principled explanation for how $\textit{A$^2$SG}$ promotes convergence to flatter minima and improves generalization.
We conduct extensive experiments on diverse models, including CNN-based and Transformer-based SNNs, across various tasks such as image classification using both static and neuromorphic datasets, as well as segmentation.
The results demonstrate that $\textit{A$^2$SG}$ consistently improves accuracy and energy efficiency, establishing it as a general and reliable solution for training deep SNNs.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces A$^2$SG, a unified framework designed to address the challenges of sharp loss landscapes and temporal inconsistency in deep SNN training, which arise from the use of surrogate gradients. The method employs a novel spatio-temporal adaptive strategy (ST-ASG) and a neuron-dynamics-aware asymmetric surrogate gradient (ASY) to guide the network towards flatter minima, thereby enhancing performance and generalization.

### Strengths
The work tackles a critical bottleneck in the SNN field. Its core contribution lies in establishing a theoretical link between the micro-level design of surrogate gradients (shape, window) and the macro-level geometry of the loss landscape (flatness). The effectiveness of the proposed method is substantiated by extensive and thorough experiments across a wide range of architectures and tasks, with convincing results.

### Weaknesses
1.A major flaw of this paper is the complete omission of the additional computational overhead introduced by A$^2$SG. The cost of running a Bayesian optimization search (Alg. A1) for every layer at each update interval appears to be substantial. 

2.The theoretical foundation of the paper rests on strong assumptions that are not empirically validated. For instance, the analysis in Section 4.1 assumes a small local gradient variation $CV(\delta)$, and the proof of Thm. A2 relies on a linear approximation of a Gaussian distribution. The absence of discussion or experimental validation of these assumptions' validity during practical training weakens the persuasiveness of the theoretical contributions.

3.The authors apply strong data augmentation strategies (e.g., CutMix, RandAugment) for their method while comparing against SOTA results that appear to be cited directly from original papers. These baseline methods may not have been trained with the same level of augmentation, constituting an unfair comparison that obscures the true performance gain attributable to A$^2$SG itself.

4.The ablation study in Table 6 is conducted on CIFAR-10. Under strong data augmentation, the baseline performance is already very high, which diminishes the significance of A$^2$SG's marginal improvements. An ablation study on a more challenging dataset where the gains are larger (e.g., CIFAR-100) would be more convincing.

5.In Table 3, a special network architecture from E-SpikeFormer with T=1, D=4 is used. This is effectively an integer-activation network, not a conventional spiking network. The paper fails to explain how the surrogate gradient, which depends on membrane potential and a firing threshold, is applied and how β is searched in this non-binary, single-timestep setting. This leaves the reader confused about its application to this important model.

6.The adaptive process of A$^2$SG introduces several new, critical hyperparameters (e.g., search frequency $i_update$, number of observation points $n_obs$), yet the paper lacks a sensitivity analysis for them. This poses a significant challenge for reproducing the results.

### Questions
1.I strongly recommend that the authors provide a comparison of the wall-clock training time between A$^2$SG and a baseline method in the appendix to quantify the computational overhead.

2.To more robustly demonstrate the contribution of each component, I suggest adding an ablation study on the CIFAR-100 dataset

3.Please provide a detailed explanation of how A$^2$SG is adapted to the integer-activation model with T=1, D=4 in Table 3. What is the precise mechanism for applying the surrogate gradient and performing the β search in this scenario?

4.There are several undefined or unclear symbols in the paper, such as "CV" after Eq. 5 and the unit "in k" for spike counts in Table 6. Please proofread the entire manuscript carefully and correct these details.

### Soundness
2

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
4

### Summary
This work proposes an adaptive and asymmetric calculation scheme for surrogate gradients of SNNs, which is based on the theoretical analysis of local gradient variation and the curvature of the loss landscape.

### Strengths
1. The authors discuss their adaptive control strategies for the surrogate gradient interval from the perspectives of spatial gradient variation (SGV) and temporal gradient consistency (TGC). The overall argumentation process is logical.

### Weaknesses
1. Since this work considers issues such as temporal gradient consistency (TGC) when calculating surrogate gradients, performance validation on time-series datasets is crucial. However, in Tab. 4, the performance improvement of $A^2SG$ compared to STBP-tdBN (baseline) on CIFAR10-DVS is not significant (81.68% v.s. 81.30%, < 1%).

2. For large-scale datasets (e.g. ImageNet-1k), the performance improvement achieved by this work in Tab. 3 is also less than 1%. In addition, using E-SpikeFormer as the baseline model may not be suitable because it considers a multi-threshold spike firing mechanism and combines other optimization techniques, which cannot fully reflect the impact of surrogate gradient calculation on performance. On the contrary, in some relatively simple network structures, the influence of surrogate gradients may be more prominent.

3. In recent years, researchers have attempted to further enhance the performance of SNNs from various perspectives, such as proposing advanced spiking models, designing spiking self-attention calculation schemes, introducing loss functions and BN modules based on temporal information, etc. The improvement of surrogate gradient calculation is merely one of these optimization routes. Therefore, I tend to think that the overall contribution of this work to the SNN community is relatively limited.

### Questions
See Weaknesses Section.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper propose a method of surrogate gradients to improve SNN backpropagation training. This method dynamically adjusts the surrogate gradient window width for spatio-temporal adaptation and introduces an asymmetric surrogate that assigns larger gradients to neurons with higher membrane potentials. The authors provide theoretical analysis linking gradient variation to loss landscape curvature and demonstrate consistent improvements across diverse architectures and tasks.

### Strengths
- Theoretical analysis between relationship of surrogate gradient design to loss landscape sharpness, asymmetric surrogates to CV 
- New design of surrogate gradient where larger gradients were assigned to neurons with higher membrane potentials
- Comprehensive experimental validation on image classification and segmentation datasets and models

### Weaknesses
- Bayesian optimization for β search runs every epoch. This computational overhead was not analyzed. No wall-clock training time comparisons provided. 
- Performance improvement is limited. This may imply that this question is not a hard core problem for snn training. 
- No study on $\beta$ update frequency (every epoch vs every N epochs). Sensitivity to Bayesian optimization hyperparameters (nobs, neval, $\delta$)?
- How about other neuron models (PLIF, ALIF, etc.)? Please verify this model apply to them too.

### Questions
- Line 884: "β fixed to 0.5" but then say "optimal β searched every epoch"?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper addresses sharp loss landscapes and inconsistent gradients across time steps in training spiking neural networks by proposing adaptive and asymmetric surrogate gradients. At the final time step, the method minimizes spatial gradient variability (SGV) to stabilize updates. At earlier time steps, it maximizes temporal gradient consistency (TGC) to align update directions. The asymmetric surrogate assigns larger gradients to membrane potentials closer to the firing threshold. Together, these choices maintain low SGV and high TGC throughout training and are associated with a smaller maximum eigenvalue of the Fisher information matrix, steering optimization toward flatter minima and better generalization. Theoretical analysis supports the design, and experiments across multiple architectures and datasets show consistent improvements over baselines.

### Strengths
1.	The method targets core challenges in SNN training and is supported by clear, rigorous analysis and proofs, yielding coherent and well-substantiated conclusions.

2.	Extensive experiments and ablations across diverse datasets and architectures consistently confirm the method’s effectiveness over baselines.

### Weaknesses
1.	Theorem 1 assumes Gaussian membrane potentials with mean below $a$. In practice, layer and time-dependent distributions may violate this and drift during training.

2.	The paper does not report the overhead introduced by Bayesian search for $\beta$


3.	The gains over STBP on neuromorphic datasets are not obvious. This may suggest only moderate performance on neuromorphic data. More evaluations across additional models and neuromorphic benchmarks are needed to substantiate the method’s effectiveness in this setting.

4.	TGC is only meaningful when T>1. If configurations such as E-SpikeFormer with $A^2SG$ use T=1, it is difficult to establish the effectiveness of the proposed method.

### Questions
1.	Why SGV be measured at the final time step rather than at other time steps?

2.	How large is the practical gain from Bayesian search? How much improvement does it deliver over using a fixed $\beta$, a simple grid search, or other methods?

I would raise my score if the authors can address these weaknesses and questions.

### Soundness
2

### Presentation
3

### Contribution
3
