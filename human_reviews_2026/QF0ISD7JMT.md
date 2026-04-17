# Anatomy-DT: A Cross-Diffusion Digital Twin for Anatomical Evolution

- Decision: Reject
- Scores: 2, 4, 6, 2

## Abstract
Accurately modeling the spatiotemporal evolution of tumor morphology from baseline imaging is a pre-requisite for developing digital twin frameworks that can simulate disease progression and treatment response. Most existing approaches primarily characterize tumor growth while neglecting the concomitant alterations in adjacent anatomical structures. In reality, tumor evolution is highly non-linear and heterogeneous, shaped not only by therapeutic interventions but also by its spatial context and interaction with neighboring tissues. Therefore, it is critical to model tumor progression in conjunction with surrounding anatomy to obtain a comprehensive and clinically relevant understanding of disease dynamics. We introduce a mathematically grounded framework that unites mechanistic partial differential equations (PDEs) with differentiable deep learning. Anatomy is represented as a multi-class probability field on the simplex and evolved by a cross-diffusion reaction--diffusion system that enforces inter-class competition and exclusivity. A differentiable implicit--explicit (IMEX) scheme treats stiff diffusion implicitly while handling nonlinear reaction and event terms explicitly, followed by projection back to the simplex. To further enhance global plausibility, we introduce a topology regularizer that simultaneously enforces centerline preservation and penalizes region overlaps. The approach is validated on synthetic datasets (Voronoi, Vessel) and a clinical dataset (UCSF-ALPTDG brain glioma). On synthetic benchmarks, our method achieves state-of-the-art accuracy (e.g., Voronoi-DSC: $95.70\pm0.30$ and Vessel-DSC: $71.14\pm0.25$) while preserving topology, and also demonstrates superior performance on the clinical dataset (UCSF-DSC: $65.37\pm0.35$). By integrating PDE dynamics, topology-aware regularization, and differentiable solvers, this work establishes a principled path toward anatomy-to-anatomy generation for digital twins that are visually realistic, anatomically exclusive, and topologically consistent. Code will be made available upon acceptance.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents a mathematically grounded framework for modeling the spatiotemporal evolution of tumor morphology and surrounding anatomy to support digital twin simulations of disease progression and treatment response. Unlike previous methods that focus only on tumor growth, the proposed approach jointly models tumor and anatomical changes using a cross-diffusion reaction–diffusion system combined with differentiable deep learning. It employs an implicit–explicit (IMEX) scheme for stable numerical integration and introduces a topology regularizer to preserve anatomical structures and prevent overlaps. Validated on synthetic (Voronoi, Vessel) and clinical (UCSF-ALPTDG glioma) datasets, the method achieves state-of-the-art accuracy and topological consistency, offering a principled foundation for anatomically realistic and clinically meaningful digital twin models.

### Strengths
N/A

### Weaknesses
I noticed that Fisher-KPP (Fisher, 1937; Kolmogorov, 1937) and Cross-Diffusion (Vanag & Epstein, 2009) are used as comparison methods in this paper. How can such outdated methods be considered appropriate for comparison?
Additionally, UNet and ConvLSTM are also quite old methods.

### Questions
N/A

### Soundness
2

### Presentation
2

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
This paper introduces Anatomy-DT, a mathematically grounded framework that unites mechanistic partial
differential equations (PDEs) with differentiable deep learning to model tumor progression in conjunction with surrounding anatomy.
Anatomy is resented as a multi-class probability field on the simplex evolved by a cross-diffusion reaction–diffusion system, employing a differentiable implicit–explicit (IMEX) scheme and integrating a topology regularizer. The approach is validated on synthetic datasets (Voronoi, Vessel) and a clinical dataset (UCSF-ALPTDG brain glioma).

### Strengths
1. Clinical relevance. This work represents a novel method for joint tumor-anatomy modeling. Past research has focused mainly in modeling tumor growth while in this paper, the authors propose a coupled evolution of tumors and adjacent anatomical structures, which sounds more relevant for clinical applications. Anatomy-DT uses cross-diffusion PDEs on the probability simplex is clearly defined. 

2. Contributions. There are two interesting contributions in this work. First, the differentiable IMEX solver which treats stiff diffusion implicitly while handling nonlinear terms can be generalised to other deep learning pipelines. The Topology-aware regularization is effective at reinforcing structural information unlike generative models that do not necessarily preserve the structure.

3. Evaluation: The model is evaluated on both synthetic (Voronoi, Vessel) and clinical (UCSF-ALPTDG) datasets. The performance improvements over baselines are promising.

### Weaknesses
1. Concerns on generalisation. Only one clinical dataset is used, the UCSF-ALPTDG brain glioma dataset. Generalization to other cancer types and other imaging modalities is not shown.

2. Methodology. There is no sufficient information that explains how the Growth CNN operates. It is unclear how much of the performance can be attributed to CNN vs PDE. 

3. Parameter choices. There is no reasoning behind the selection of the cross-diffusion coefficients, growth rates, and carrying capacities.

4. Baseline comparison missing. The current method is not compared to any diffusion models conditioned on anatomy.

5. Dataset Limitations. The synthetic datasets (Voronoi, Vessel) may not fully represent the amount of anatomical evolution.

6. Reproducibility. Some implementation details are missing for the choice of hyperparameters, training setup, and architectures.

### Questions
1) An ablation showing PDE performance vs. CNN or combined is missing. Could you show how relevant the PDE really is ?  

2) Are the parameters in the PDE learnt or set in advance? How can these learnt parameters be guaranteed to reflect the biological principles behind the tumour and its anatomy? 

3) Could you provide more detailed information about the number of patients, the train/validation/test splits ?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces Anatomy-DT, a hybrid framework combining cross-diffusion PDEs, differentiable implicit–explicit (IMEX) integration, and topology-preserving regularization for simulating tumor and anatomy evolution from baseline imaging. The authors define the anatomy representation as a probability field on the simplex, enforcing tissue exclusivity, while a small Growth CNN learns residual corrections to the PDE dynamics. The approach is tested on synthetic datasets (Voronoi, Vessel) and a clinical UCSF glioma dataset, outperforming U-Net, ConvLSTM, NeuralODE, and classical PDE baselines in Dice and HD95 metrics. According to the results shown in the paper, Anatomy-DT achieves the best performance.

### Strengths
1. The paper addresses a very valuable problem. It bridges mechanistic tumor PDEs and deep learning for a digital twin application, utilizing the neighboring anatomy jointly. This is a very valuable design (clinically). 
2. The paper provides a very reasonable mathematical structure. Cross-diffusion on the simplex elegantly captures tissue competition and exclusivity; the IMEX solver offers a stable and differentiable temporal evolution layer. Incorporating clDice and overlap penalties meaningfully constrains geometric plausibility, addressing a known issue in medical image generation. 
3. The presentation is very clear overall.

### Weaknesses
1. The baselines (U-Net, ConvLSTM, NeuralODE) are not designed for deformation or flow-based modeling. **Comparisons should include ImageFlowNet (https://arxiv.org/abs/2406.14794) or other neural image-registration / diffeomorphic flow methods that explicitly predict voxel-wise motion fields.** Without these, the evaluation does not fully test the advantage of Anatomy-DT over strong, domain-relevant baselines.
2. It is unclear whether the CNN predicts residuals for the reaction term, diffusion coefficients, or updates directly on the state variable.
3. The “projection to the simplex” step ensures feasibility, but its implementation (exact projection vs softmax normalization) is unspecified; differentiability through this step is questionable.
4. The phrase “diffusion-based PDE” may be misinterpreted as diffusion models used in generative AI. The paper should clarify that the diffusion here refers to physical heat diffusion, not stochastic diffusion modeling.
5. All experiments involve only two timepoints (baseline to follow-up). Handling irregular or continuous time series is unclear; the PDE formalism could, in principle, allow this, but it is never demonstrated or validated.

### Questions
1. Classical PDE models use a single scalar tumor field. How does Anatomy-DT initialize and evolve multi-class anatomies—do all tissues obey the same PDE or class-specific diffusion matrices?
2. Which topological invariants (e.g., Betti numbers, Euler characteristic) are actually preserved by the clDice + overlap loss? Is any persistent-homology-based measure computed, or is topology regularization purely heuristic?
3. How does the model manage patients with variable or irregular imaging intervals?
4. What exact operator is used for projection, and how are gradients propagated through it?
5. Have you compared against ImageFlowNet, which also predicts anatomy-to-anatomy transformations (seems highly relevant to your topic )?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper presents Anatomy-DT that integrates cross-diffusion PDEs, differentiable IMEX solvers, and topology-aware regularization to model the spatiotemporal change of anatomical structures and tumors.

### Strengths
1. The paper focuses on a well-motivated problem to model the anatomy-toantomy evaluation for longitudinal medical imaging analysis.
2. The paper includes both synthetic and clinical datasets with both quantitative and qualitative results.
3. The paper includes the ablation study to verify the effectiveness of each key designs including spatial modeling, topology constraints and growth CNN.

### Weaknesses
1. Although the paper is well motivated, it only evaluates on one real dataset and the performance improvement seems to be trivial (e.g., less than 1% compared to runner-up). Therefore, the generalizability in the real-world application remains a concern.
2. Some of the experiment details are missing, for example, what is the testbed of ablation study. Also, it is quite confusing how the baselines are implemented. Were the topology and regularization terms also applied to the baseline method? More details should be described in the paper.
3. How’s the ablation performance on the other two datasets?
4. The related work is placed in an uncommon position.

### Questions
N/A

### Soundness
2

### Presentation
2

### Contribution
2
