# CryoNet.Refine: A One-step Diffusion Model for Rapid Refinement of Structural Models with Cryo-EM Density Map Restraints

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6

## Abstract
High-resolution structure determination by cryo-electron microscopy (cryo-EM) requires the accurate fitting of an atomic model into an experimental density map. Traditional refinement pipelines like Phenix.real_space_refine and Rosetta are computationally expensive, demand extensive manual tuning, and present a significant bottleneck for researchers. We present CryoNet.Refine, an end-to-end, deep learning framework that automates and accelerates molecular structure refinement. Our approach utilizes a one-step diffusion model that integrates a density-aware loss function with robust stereochemical restraints, enabling it to rapidly optimize a structure against the experimental data. CryoNet.Refine stands as a unified and versatile solution capable of refining not only protein complexes but also nucleic acids (DNA/RNA) and their assemblies. In benchmarks against Phenix.real_space_refine, CryoNet.Refine consistently yields substantial improvements in both model–map correlation and overall model geometric quality. By offering a scalable, automated, and powerful alternative, CryoNet.Refine is poised to become an essential tool for next-generation cryo-EM structure refinement. Web server: https://cryonet.ai/refine; Source code: https://github.com/kuixu/cryonet.refine.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes CryoNet.Refine, a neural refinement framework that integrates a one-step diffusion model with experimental cryo-EM density map restraints.
Unlike traditional refinement programs such as Phenix.real_space_refine, CryoNet.Refine introduces:
1. A learnable density model that maps atomic coordinates to voxelized densities, enabling end-to-end differentiable correlation with experimental maps, and

2. A geometry loss suite that includes differentiable Ramachandran, rotamer, and bond-length/angle penalties.

The model refines initial structures (often AlphaFold3 predictions) toward experimental densities, showing improvements in map correlation (CCmask, CCmain-chain) and stereochemical quality across 63 cryo-EM complexes.

### Strengths
- The combination of a density-aware differentiable loss and stereochemical constraints is technically sound.
- The paper is well written
- The paper demonstrates consistent gains over Phenix in both correlation coefficients and geometric metrics (notably, Ramachandran favored and rotamer outliers).

### Weaknesses
- The proposed “one-step diffusion” formulation is described as an innovation, but its necessity is not clearly demonstrated. The ablation shows marginal benefit, and it is unclear why a diffusion step—traditionally for generative sampling—improves deterministic refinement. A comparison to standard coordinate optimization (e.g., gradient descent using the same losses) would help establish that the diffusion mechanism contributes beyond architecture novelty.
- The “density generator” is mentioned but not architecturally described (layers, parameters, training regime, loss balance weights γ). It is also unclear how the network differs from a Gaussian scattering baseline (molmap from ChimeraX).
- The ablation in Table 4 shows individual loss terms but does not isolate the impact of the density model vs. geometry components.
The diffusion vs. non-diffusion refinement comparison (one-step vs. multi-step) is qualitative; quantitative runtime and accuracy tradeoffs are missing.

### Questions
1. What concrete advantage does the diffusion step provide compared to standard coordinate optimization using the same losses?
2. The authors state that they developed a novel, parameter-free, and differentiable density generator in the Introduction, but Section 3.2.1 indicates that they simply follow the ‘molmap’ implementation in ChimeraX. What, specifically, is novel in this component?

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
3

### Summary
CryoNet.Refine is an end-to-end framework that automates and accelerates molecular structure refinement. It utilizes a one-step diffusion model that integrates a density-aware loss function with robust stereochemical restraints, enabling it to rapidly optimize a structure against the experimental data.

### Strengths
- This study bridges the gap between structure prediction models and cryoEM densities in a modern approach. With some carefully designed loss functions, CryoNet.Refine brings the power of folding models to cryo-EM model building.
- The ablation study is comprehensive, and the figures are well made.

### Weaknesses
- Why is the model named “one-step diffusion” but takes several recycling numbers?
- The method part lacks several technical details, and is hard to follow. 
	- What is the training set of CryoNet.Refine? Is it “trained” for each protein (like ReLION/CryoDRGN), or trained over a set of proteins and evaluated on some test set without tuning the model parameters?
    - In Section 3.1, line 202, the authors wrote that the model is initialized from Boltz-2’s parameters.
		- Does CryoNet.Refine has exactly the same model architecture as Boltz-2?
		- If I am understanding correctly, CryoNet.Refine can be viewed as something like AF3 with classifier guidance. Is that true? 
	- In line 208, the authors said that AF3 requires hundreds of sampling steps. In line 212, the authors wrote that one-step diffusion poses a key advantage that guidance can be performed directly on the predictions. I admit these two statements, but what confused me is that: AF3 is also a diffusion model which predicts the final sample (instead of predicting the velocity or noise), what is the difference between CryoNet.Refine and AF3?
		- Can I view AF3 as a one-step diffusion model, although it takes hundreds of steps in the sampling process?

### Questions
- In Section 3.1, line 193, the initial atomic structure is fed into CryoNet.Refine. You said that the model derived pair representation $z$ from the atomic structures. What is the shape of the pair representation? Is the side length of $z$ the number of residues, or the number of atoms?
- I think AF3 can also benefit from the loss you proposed. What if adopting the loss you proposed to the predicted sample of AF3, like diffusion posterior sampling? (ref: (1) Diffusion posterior sampling for general noisy inverse problems, (2) CryoFM: a flow-based foundation model for cryoEM densities)
- Since the model is initialzied from Boltz-2, why do you need some geometry loss to constrain the structures? I think a folding model has already contained such knowledge.

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
3

### Summary
This paper proposed a model for the refinement of atomic models based on CryoEM density map. Specifically, this work finetunes Boltz2 for each pair of density map and atomic model with the density loss and the geometry loss. The structure is output by a specially designed one-step diffusion module which enables direct back-propagation from the complex loss functions.

### Strengths
- The density loss is conceptually elegant and novel. The one-step diffusion module is well motivated as typical diffusion models can only be trained using specific loss function. The one-step diffusion allows flexible loss defitions.
- Structure-specific post-training of the Boltz2 model ensures generalization. Previous works directly learn a mapping from density maps to atomic models, which might fail on structures that are significantly different from training data.
- Better performance than previous methods.

### Weaknesses
- As it requires fine-tuning for each structure at inference time, the efficiency is still limited, and the efficiency improvement is not very significant compared to previous methods.

### Questions
- What would happen if we directly optimize the coordinates of the input structure instead of network parameters using the density loss and the geometry loss?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper focuses on the process of fitting atomic models into experimental density maps for structure determination by cryo-electron microscopy (cryo-EM). Traditional refinement pipelines are computationally expensive and require extensive manual tuning. To address these challenges, the authors present CryoNet.Refine, an end-to-end deep learning framework that automates and accelerates molecular structure refinement. Specifically, CryoNet.Refine employs a one-step diffusion model that integrates a density-aware loss function with robust stereochemical restraints to rapidly optimize structures against experimental data. The framework is capable of refining not only protein complexes but also DNA/RNA–protein assemblies. Experimental results demonstrate that CryoNet.Refine achieves clear improvements over traditional approaches in both model–map correlation and overall geometric accuracy.

### Strengths
1. This paper connects two important things in structural biology, computational modeling (e.g., AlphaFold3) and cryo-EM experimental density maps.

2. The architecture design is well-motivated, improving efficiency while maintaining refinement quality.

3. Presented experimental results demonstrate strong refinement performance, reduced manual effort, and clear efficiency gains over traditional methods.

### Weaknesses
1. It is confusing what exactly happens during a training step versus inference. I assume Fig. 2 provides an overview of the training process. During inference, there seems to be no computation of density or geometry loss, and the model likely performs only a single pass through the Atom Encoder, Sequence Embedder, and Diffusion Module. Clarifying this distinction would help readers better understand the workflow and computational efficiency.

2. Several simple yet informative baselines are missing — for example, numerical optimization starting from the initial structure. Including such comparisons would better contextualize the claimed improvements.

3. There is no ablation study comparing the classical diffusion model with the one-step diffusion approach. 

4. The density generator component is not evaluated. Details about its data patterns or visualizations are absent, and there is no discussion on how well the generated densities align with experimental cryo-EM maps, which can vary significantly depending on the instrument and imaging conditions.

5. The number of reported test cases is too small. A larger-scale evaluation is necessary to establish robustness and generalizability across diverse molecular systems.

### Questions
1. If my assumption in Weakness 1 is correct, that no density or geometry loss is computed during inference, would it be beneficial to incorporate these losses as a form of test-time refinement? 

2. Figure 2 includes a Pairformer module, but it is never discussed in the text. Is this a Transformer-based component where cross-attention occurs between atom and sequence embeddings?

### Soundness
3

### Presentation
2

### Contribution
3
