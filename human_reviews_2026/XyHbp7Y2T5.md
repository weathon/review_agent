# PhyCo: Physics-Consistent Learning of Implicit Constitutive Laws via Monocular Observations of 3D Gaussians

- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
We present **PhyCo**, a framework for learning implicit constitutive laws from \textbf{monocular dynamic observations} of Gaussian splatting. Existing implicit methods often suffer from local minima under noisy supervision and lack physical interpretability, while explicit approaches rely on predefined constitutive equations, limiting generalizability. To address these issues, our framework, **PhyCo**, introduces two key innovations. First, **initializing from a static multi-view scan, we propose *Edge-Aware Depth Consensus Anchors* to establish robust geometric constraints from subsequent monocular dynamic observations**, circumventing unreliable pixel-level supervision. Second, a *Multi-Hypothesis Physics Verifier* integrates classical constitutive models as differentiable hypotheses, providing strong physical priors to regularize the optimization while preserving the flexibility of implicit modeling. This unified approach ensures physical plausibility without sacrificing generality. Extensive experiments on synthetic, real-to-sim, and real-world datasets demonstrate that **PhyCo** significantly outperforms existing methods, achieving state-of-the-art performance in learning accurate and generalizable physical dynamics from monocular videos.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a method to learn the implicit constitutive laws of deformable objects via monocular video observation. The key innovation is how the authors introduce regularizations through depth-based loss with the help of a monocular depth network and through a library of explicit physics rules.

### Strengths
I like that this approach learns a neural implicit dynamics model with the help of existing explicit physics laws as guidance. The final learning outcome can be correlated back to the explicit models while not constrained by them. The way the explicit physics laws are used is in a spirit similar to expectation maximization. 

The experimental result shows superior performance of the proposed model.

### Weaknesses
I'm not sure how realistic the problem and experiment setup is. The deformation of the objects is very significant and uncommon in the real world. The dropping motion is doable but also limiting and not very common in real-world experience. I wonder what the practical application of this task are, given that it takes more than 1 hour to reason about one video of one object. 

There is a lack of ablation study of the methodology. Only comparisons shown are without either of the two loss functions. This is a very coarse comparison. I think more detailed experiments are helpful especially given that there are very specific designs of the loss functions (e.g., global vs anchor-level supervision, rank-based loss formulation, procedure of the Multi-Hypothesis Physics Verifier, etc).

### Questions
Why is the color-space supervision not used? Does it hurt the result? 

What is the rationale behind the depth loss based on the rank correlation instead of metric-based losses?

### Soundness
3

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
This paper proposes PHYCO, a unified framework for learning implicit constitutive laws from 3D Gaussian Splatting observations. The authors aim to find the trade-off between explicit physics-based models which are physically interpretable but less generalizable, and implicit constitutive laws which are flexible but less interpretable.

PHYCO introduces two novel components. On the one hand, Edge-Aware Depth Consensus Anchors are established to align geometry based on depth instead of unreliable colors. On the other hand, a Multi-Hypothesis Physics Verifier is constructed to integrate classical constitutive laws into implicit constitutive model during optimization process.

Experiments on synthetic, real-to-sim, and real-world datasets show significant improvements.

### Strengths
1. Instead of purely relying on implicit constitutive models, the author introduces multi-hypothesis physics verifier module to inject explicit constitutive model priors. Meanwhile, this priors are adapted by the parameter estimation consistency, no human annotation for the exact material is needed. 
2. The authors introduce depth supervision to enable the model applicable on monocular videos. 
3. Promising performance gains on various datasets.

### Weaknesses
1. One of the core parts, implicit constitutive law estimation models rely on existing NCLaw model, weakening the novelty of this paper. 
2. The authors argues to use LoRA for better efficiency. But no analysis about efficiency gains is shown and what the performance gap is induced. LoRA does reduce the memory cost, but the authors should show the comparable performance compared to full-size finetuning. 
3. The setting of Gaussian initialization is not clear. Is the model only using the first frame to initialize the Gaussian kernels? If so, how the method deal with the emerging parts in the later frame which is occluded in the first frame. If not, I’ll challenge the monocular setting. 
4. Although PAC-NeRF and GIC is designed for multi-view supervision. But it’s easy to include a depth supervision and adapt to monocular estimation (although the performance is not guaranteed). The comparison is missed here. 
5. Ablation missing: how the spliting number N affects the verifier performance? This hyperparameter really influences the varaince accuracy. 
6. Ablation should be included in the main context. 

Although this paper proposes an interesting pipeline, due to the above weaknesses, I cannot give an acceptance recommendation till now. However, I’m very open to increase my score based on the authors’ responses and other reviewers’ opinions.

### Questions
1. As asked in weakness 3, how gaussian kernels are initialized?
2. How NCLaw is initially trained or warmed-up?
3. What is the computational cost of the proposed method? Especially for the time burdern. 
4. How sensitive is PHYCO’s performance to the chosen set of classical constitutive hypotheses in the Multi-Hypothesis Physics Verifier?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a novel framework called PHYCO, which aims to learn implicit physical laws from GS’s monocular observation data. The method effectively addresses two major challenges existing in current implicit learning approaches: unstable geometric learning and lack of physical interpretability. Specifically, PHYCO introduces Edge-Aware Depth Consensus Anchors (EADCA) to stabilize geometric reconstruction and designs a Physics-Consistent Loss that integrates physical laws into the training of implicit functions. This enables robust, interpretable, and highly generalizable learning of complex physical processes.

### Strengths
The authors demonstrate significant innovation and advantages in applying implicit learning to physical modeling:
1.	The paper introduces EADCA to effectively tackle the issue of inaccurate or locally optimal geometric representations in implicit methods under noisy monocular supervision. This mechanism ensures high-quality geometric reconstruction, providing a solid foundation for subsequent physical parameter inversion.
2.	One of the core contributions of this paper is the design of MHPV, which embeds known physical conservation laws (such as momentum conservation) as hard constraints into the training of implicit constitutive laws. This ensures that the learned constitutive functions are physically reasonable and reliable, greatly enhancing the model’s interpretability — something that purely data-driven methods can hardly achieve.
3.	The PHYCO framework successfully combines the efficient rendering capability of GS with the expressive power of implicit functions, enabling the direct learning of complex, nonlinear, and non-elastic constitutive laws from monocular videos. This avoids dependence on traditional predefined explicit constitutive equations and significantly broadens the model’s generalization and modeling capability for various complex materials and physical phenomena.

### Weaknesses
However, I do have several concerns about this work:
1.	The core components of the framework rely on several pre-trained modules. Although the authors partially address lighting variations by changing illumination conditions in the dataset, if these modules perform poorly under certain conditions (for example, large-scale deformations), the overall performance of the framework could be greatly affected.
2.	Due to the high structural complexity and diverse optimization objectives, the framework may suffer from high debugging and training costs, leading to potential instability during optimization.
3.	In MHPV, the authors select a set of classical constitutive models to validate material physical behaviors. However, if these selected material models do not adequately approximate or represent real material physics, the “physical rationality” constraints imposed by MHPV might become counterproductive rather than optimizing PHYCO. Essentially, it strongly assume that the chosen constitutive equations are trustworthy but lacks rigorous proof of their validity.

### Questions
see above

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces PHYCO, a new method for learning how objects behave physically from monocular videos, which are videos taken from a single camera view. The key ideas are to use Edge-Aware Depth Consensus Anchors to get reliable geometric information from limited data, and a Multi-Hypothesis Physics Verifier to incorporate well-known physical laws as guiding hypotheses. The approach effectively handles noisy and sparse supervision and works well even with complex materials and real-world scenes. Experiments on synthetic and real data show that PHYCO outperforms existing methods, producing realistic, physically consistent results while maintaining generalization to new scenarios.

### Strengths
1. Uses Edge-Aware Depth Consensus Anchors to extract reliable geometric information from limited and noisy data, improving the accuracy of 3D shape and motion understanding.

2. Incorporates classical physical laws as differentiable hypotheses through the Multi-Hypothesis Physics Verifier, ensuring learned models are physically consistent.

3. Performs well even with monocular videos that have limited detail and are affected by noise.

4. Outperforms some existing state-of-the-art methods on both synthetic and real-world datasets, producing better physical simulations and renderings.

### Weaknesses
1. While the method handles single-object dynamics well, its scalability to multi-object interactions or highly complex scenes remains underexplored.

2. Although not explicitly discussed, the integration of multiple components such as the verifiers and anchors potentially increases training complexity and time, which could hinder practical adoption.

3. The method assumes reasonably accurate geometric initializations; cases with severe geometric ambiguities might challenge the approach.

4. The choice and diversity of the classical models used in the physics verifier may limit applicability to certain material classes or behaviors not represented by the hypotheses.

5. The real-world experiments are limited in scale and variety (e.g., only a few objects like dragon, wolf, pudding, etc.). Although comparisons with methods like NeuMA and NCLaw are conducted, the scope of evaluation could be broader by including more recent or diverse approaches, or ablation studies that more directly isolate the contributions of individual components.

### Questions
1. How do the balance factors λm​ and λg​ influence the training stability and convergence?

2. Could you conduct ablation experiments on the components you proposed to demonstrate the effectiveness of each part?

### Soundness
3

### Presentation
2

### Contribution
2
