# Neural Latent Arbitrary Lagrangian-Eulerian Grids for Fluid-Solid Interaction

- Decision: Accept (Poster)
- Scores: 8, 6, 2, 4

## Abstract
Fluid-solid interaction (FSI) problems are fundamental in many scientific and engineering applications, yet effectively capturing the highly nonlinear two-way interactions remains a significant challenge. Most existing deep learning methods are limited to simplified one-way FSI scenarios, often assuming rigid and static solid to reduce complexity. Even in two-way setups, prevailing approaches struggle to capture dynamic, heterogeneous interactions due to the lack of cross-domain awareness. In this paper, we introduce **Fisale**, a data-driven framework for handling complex two-way **FSI** problems. It is inspired by classical numerical methods, namely the Arbitrary Lagrangian–Eulerian (**ALE**) method and the partitioned coupling algorithm. Fisale explicitly models the coupling interface as a distinct component and leverages multiscale latent ALE grids to provide unified, geometry-aware embeddings across domains. A partitioned coupling module (PCM) further decomposes the problem into structured substeps, enabling progressive modeling of nonlinear interdependencies. Compared to existing models, Fisale introduces a more flexible framework that iteratively handles complex dynamics of solid, fluid and their coupling interface on a unified representation, and enables scalable learning of complex two-way FSI behaviors. Experimentally, Fisale excels in three reality-related challenging FSI scenarios, covering 2D, 3D and various tasks. The code is available at https://github.com/therontau0054/Fisale.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces Fisale, a data-driven framework for solving complex two-way FSI problems. Inspired by classical numerical methods such as the Arbitrary Lagrangian-Eulerian (ALE) approach and partitioned coupling algorithms, Fisale excels in three reality-related challenging FSI scenarios, covering 2D, 3D and various tasks by leveraging multiscale latent ALE grids and partitioned coupling module.

### Strengths
1. Clear structure and good readability.
2. Innovation in introducing the ALE approach and partitioned coupling module.
3. Abundant experiment and superior experimental results compared with current methods.
4. Rich practical application scenarios.

### Weaknesses
1. Lack of consistency in the experimental results: For example, as shown in Figures 8 and 9, the fluid fitting effect of LNO is significantly worse than MGN, but the results in Table 1 are the opposite.
2. Inadequate experiment on complex scenarios: Each experiment in the paper involves relatively low complexity of surfaces and does not prove whether the model has a good fitting ability for more complex surfaces, such as Shape-Net Car (mentioned in AMG).
3. Insufficient proof of generalization ability: The experiment cannot fully demonstrate that the model has learned the physical laws in the fluid dynamics scenario rather than the fitting data in a few scenarios.

### Questions
Explain the weaknesses please.

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper proposes to model fluid-solid interaction (FSI) problems by explicitly modeling fluid/solid/coupling interface as three separate components. The approach boils down to:

1) Projecting state data (positions, features) from all three components onto a shared latent grid
2) Processing data on the grid through a partitioned coupling module (PCM) that sequentially updates solid → grid → fluid → interface
3) Decoding back to the original points

The approach achieves superior performance compared to other methods (GNN and Transformer) that treat the FSI problem as one single unified system.

### Strengths
- The paper is well and clearly written, the introduction is comprehensive, and the problem is demonstrated in an intuitive way with Figure 1.
- Architecture design is well-motivated:
  - It suggests a novel approach to handling a gap in prior works, and the problem tackled by the method is practically relevant for many engineering fields.
  - The approach is inspired by established numerical techniques (ALE and partitioned coupling).
  - Each component has a physical justification and is grounded in domain knowledge.
- The method is experimentally validated against multiple baselines and has demonstrated superior performance on multiple tasks. Benchmarks include 2D and 3D problems with different complexity and fidelity.
- The chosen baselines are state-of-the-art methods in physical modelling. The proposed method consistently outperforms all of them.
- Ablation studies are thorough and target specific components of the method (e.g. multiscale ablation and having explicit interface).

### Weaknesses
- The paper adopts linear attention which is empirically weaker than standard attention despite the scale of the problems being manageable for flash attention.
- The method does not scale well with increasing the number of latent points, which, in my opinion, is a limitation, see Questions.
- Ablation studies indicate that some of the components might not even be necessary:
  - For example, Table 15 demonstrates that the ordering within PCM doesn't change the performance. Since that is the case, couldn't you simplify the model to:
    1) update solid via cross attention;
    2) update fluid via cross attention;
    3) update interface with self attention;
    4) update grid?

    Note that 1) and 2) can be potentially done in parallel if solid and fluid concatenated, which would make it even faster. Am I missing anything here? Perhaps PCM converges faster?
   - In table 16, substituting the PCM module with attention only slightly drops the performance. However, it is not clear to me what attention is used: linear or self-attention. If not latter, then I would expect much better performance from using standard attention which would, in my opinion, question the value of having the PCM module. It would be great if authors could do the ablation measuring runtime, memory and performance. Having a single kernel might also make it faster, hence I am curious about the runtime.
- The paper does not do an interpretability analysis of the latent ALE grids. While it is not strictly necessary, I do find Transolver slice assignment useful, perhaps the paper would benefit from a similar plot. Besides, studying attention patterns would be a neat addition, similar to how it is done in the EAGLE paper [1].

[1] https://eagle-dataset.github.io/

### Questions
I am overall learning towards accept, but answering those question might significantly improve my rating and I also generally improve the paper.
- Did you try using standard attention instead of linear one? The scale appears manageable to me.
- Regarding Table 17, performance vs mesh resolution. It appears to me that the performance does not depend on the mesh resolution. That is counter-intuitive to me as I would expect improved performance with increased number of grid points. Do you have any explanation why it does not happen?
- Which attention is used in Table 16? If linear, would it be possible to update the table with standard attention (flash attention) and compare memory, runtime on top of performance?
- For the set of problems, how does the method compare against a numerical solver?
- Authors report Relative L2 instead of MSE for the CoDA-NO dataset. Is it possible for you to provide MSE just for a direct comparison to the CoDA-NO's paper?
- You claim to be calable to large-scale problems. Do you plan further experiments on large-scale CFD simulations? Would you method be able to handle million point grids?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This work introduces Fisale, a purely data-driven surrogate model for two-way fluid structure interaction between incompressible fluids and deformable solids. The combined solid-fluid + interface state is represented using multiscale latent ALE grids and the dynamic interface is evolved in time using self and cross attention mechanisms in a partitioned interative fashion. Multiscale modeling is handled using parallel processing of latent ALE grids, interspersed with aggregation after every layer, to capture phenomena at various scales. The authors test the effectiveness of their method by predicting future states in three different scenarios: Structure Oscillation, Venous Valve and Flexible Wing, and compare the obtained frames with the numerically simulated ground truth as well as other neural models.

### Strengths
1.	Multiscale modeling has proved to be an effective way in solving PDEs which involve dynamic boundaries and multiple domains, such as in fluid-structure interaction. Using more samples in interface regions for accuracy while using less samples is domain interiors for efficiency is a well-studied approach. The authors seem to leverage this well in the construction of their architecture.
	2.	In the experiments provided in the paper, Fisale seems to outperform other baselines, which is a good indication.
	3.	The experimentation in the main paper, together with the Appendix, is quite extensive, with a great number of relevant baselines.

### Weaknesses
1.	Although the authors claim to leverage the classical numerical formulations like ALE and Partitioned Coupling, it does not seem to be the defining factor here. This is clear from the ablations performed in Appendix G. The increased accuracy in the experiments in the main paper can simply arise from the high representational capacity of the architecture itself, since there are a lot of learnable components. Also, the choice of the dimension of the latent ALE grid (i.e. D) does not seem to be explicitly defined anywhere in the paper. A large value of D can also cause overfitting.  
	2.	The intuition behind using attention for coupling is not very clear. In my understanding, attention mechanisms are used to model long-range dependencies which arise in text based domains, for example. The interface dynamics in fluid-structure interaction are necessarily short range, as far as both fluid and solid domains are concerned. So, using attention would be wasteful, and might even be numerically unstable. 
	3.	The proposed model does not seem to be generalizable across tasks, which involve different geometry (static and dynamic). There is separate training and dataset for each of the three tasks. In that case the gain in inference time is not so much to offset the training time required for each of the tasks, when compared with a classical numerical simulation. 
	4.	There is no mention of how the initial values of physical quantities q is fed to the model. The only point where it seems to be involved is in defining the initial latent state x^{0,h} = Linear(u). Even that uses a trainable linear layer. Without information about the initial state, no model can accurately predict future states. This supports my earlier observation that the proposed model may just be overfitting the data and not actually learning the dynamics. 
	5.	 The limitations should be mentioned in the main paper, not in the Appendix.

### Questions
1.	The authors should consider expanding some of the captions for exposition purposes, especially for Figure 2 and Table 3
	2.	In this work, I believe that even the grid sample positions are inferred from the model, along with the sample values, which is good. But how are the sample positions defined for the initial state. Specifically, how is g defined for the initial state?

### Soundness
1

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors propose a new neural operator framework for handling fluid-solid interaction problems with cross-domain awareness. The proposed architecture Fisale leverages multi-scale latent grids to provide geometry aware embeddings. They further enable problem decomposition to handle nonlinear dependencies.

### Strengths
1. The paper provides good motivation to model ALE systems for fluid-solid interactions. 
2. The authors conduct thorough experiments, with a very detailed appendix section. 
3. The authors have considered SOTA baselines and SOTA problem settings.

### Weaknesses
1. The writing is quite dense in section 3 and it is not clear to me exactly how this proposed architecture compares to "A Neural Material Point Method for Particle-based Emulation, O Sharabi"
2. It seems like grid update is similar to message passing. It seems like section 3.3 is describing typical neural network operations and as such can be moved to the appendix for better readability. 
3. The overall writing can be significantly improved, with many parts being unclear with substantial focus on describing techniques that are well established in ML community (such as update fluid state and update interface influence) which describe the sequence of operations performed within the network, without any intuition for the said operations. Similarly, "update grid coordinate" is overly detailed.

### Questions
1. It was shown in GNS that the model is able to handle multi-material systems (water-sand, water-jelly) by simply creating material-type embedding and leveraging data-driven training to learn the interaction dynamics. How does that compare against the proposed approach?
2. Lagrangian systems typically require GNN based processing and models designed to handle Eulerian grid inputs fail to handle inter-particle interactions. This was shown in UPT, Transolver and GIOROM. The problem setting describes a system that is Eulerian or Lagrangian but the results don't seem to include any interaction dynamics datasets despite referencing GNS several times in the paper. 
3. The rationale behind KNN based grid is not fully clear to me. Typically neural operators leverage radius based grids with mean aggregation to enable discretization invariance (GINO, multipole graph kernel network). This is because when the input discretization changes, radius aggregation ensures an entire region is captured. With KNN based grids, the aggregation region changes based on discretization -- how does this enforce discretization convergence? 
4. The approach inherently handles multi-scale inputs. It would be interesting to see if it also handles AMR, as part of future work.

### Soundness
2

### Presentation
1

### Contribution
2
