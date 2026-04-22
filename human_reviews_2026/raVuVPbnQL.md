# Learning Flexible Forward Trajectories for Masked Molecular Diffusion

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 2, 6, 4, 6

## Abstract
Masked diffusion models (MDMs) have achieved notable progress in modeling discrete data, while their potential in molecular generation remains underexplored. In this work, we explore their potential and introduce the surprising result that naively applying standards MDMs to molecules leads to severe performance degradation. We trace this critical issue to a *state-clashing problem*-where the forward diffusion trajectories of distinct molecules collapse into a common state, resulting in a mixture of reconstruction targets that cannot be learned with a typical reverse diffusion with unimodal predictions. To mitigate this, we propose **M**asked **E**lement-wise **L**earnable **D**iffusion (**MELD**) that orchestrates per-element corruption trajectories to avoid collisions between different molecular graphs. This is realized through a parameterized noise scheduling network that learns distinct corruption rates for individual graph elements, *i.e.*, atoms and bonds. Across extensive experiments, **MELD** achieves 100\% chemical validity in unconditional generation on QM9 and ZINC250K datasets, while markedly improving distributional and property alignment over standard MDMs on both conditional and unconditioned generation.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The authors identify state-calshing as a key limitation of Masked Diffusion Models (MDMs), which refers to the way graph components with different labels end up in the same intermediate states when using a unified forward process. This issue
makes denoising difficult, especially in the highly symmetrical space of molecular structures. To mitigate this issue,
the authors propose state-dependent learned forward processes to collect structural information about the molecule and maitain the partial identity of graph elements during denoising. The paper compares to relevant baselines in diffusion-based molecular generation and shows empirical improvements in some metrics.

### Strengths
- The paper identifies a specific limitation in MDMs and addresses it appropriately.
- The algorithm is clearly described. 
- The main baselines in molecular generation are included in the empirical analysis.

### Weaknesses
**TL;DR** the weakest points for me are the positioning w.r.t the symmetry breaking literature, and the comparison to other noise methods used in diffusion (e.g. uniform, marginal, etc). I am willing to change my score if the author clarify these points and address some presentation issues. 

**Connection to the symmetry breaking literature**: I understand that MELD seeks to prevent the state-clashing problem from happening in the first place, but how does it compare to methods using an equivariant forward process and denoiser, and breaking the symmetry of the 'highly similar' intermediate states during denoising? In other words, what is the connection between your work and the broader topic of symmetry breaking as tackled by [1,2,3,4]? I am asking in particular because MELD seems to sacrifice equivariance entirely, I am curious to see how it compares to models using the inductive bias of equivariance (e.g. to model the symmetries inherent to molecules and other graphs) while breaking the symmetry introduced by the forward process (highly similar masked intermediates).

**Claims regarding 'substitution-based corruption methods'**: The authors explain that other noising approaches (I am assuing methods like uniform or marginal noise) suffer from state clashing less than MDMs. While noising with random labels *looks* less similar than noising with the same mask token, uniformally generated intermediates should have the same amount of information as masked intermediates (i.e. in the o-Phenylenediamine and m-Phenylenediamine molecules, a uniform noise schedule should struggle as much as MDMs with recovering the right isomer). I find the claim that such methods 'preserve structural similarity and retain partial identity' unsubstantiated. Can you elaborate more on this? 

**Presentation**: 
- Since the state clashing problem is the main motivation of the paper, it would be more natural to present it first in the methods section and to emphasize it more in earlier sections (introductions, related work).
- Some claims in the paper are exagerated. For example, saying that applying diffusion to molecules is "underexplored" (abstract and introduction line 51), or that MELD achieves notable empirical improvements over the baselines.

### Questions
- How does MELD perform compared to equivariant models using symmetry breaking?
- Why do you think other corruption methods maintain partial identity through the forward process?

**Typos & nitpicks**
- Line 39: scalability
- Line 90: scalability and generalizability
- Line 255: '... as less probability of state-clashing,...' => incomplete phrase
- Line 457: '... a higher count of represents' => missing word

## References 
[1] "Improved Equivariant Networks with Probabilistic Symmetry Breaking", Lawrence et al., 2025.

[2] "Equivariant Denoisers Cannot Copy Graphs: Aligns Your Graph Diffusion Models", Laabid et al., 2025.

[3] "Discovering symmetry breaking in physical systems with relaxed group convolution", Wang et al., 2024.

[4] "Equivariant networks for crystal structures, Kaba et al., 2022.

### Soundness
3

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
This paper focuses on the adaptation problem of Masked Diffusion Models (MDMs) for discrete data in molecular graph generation. It points out that "fixed, element-independent" forward masking scheduling leads to different molecules collapsing to the same intermediate state in the forward trajectory, making reverse denoising, typically unimodal and predicting independently by node or edge, difficult to learn the correct reconstruction target. To address this, the paper proposes MELD: which learns the forward masking rate at the element level (node/edge) and assigns an independent erosion trajectory to each graph element through a parameterized noise scheduling network; it is jointly optimized with the reverse denoising network during training. The authors claim that MELD achieves high efficiency in unconditional generation of QM9 and ZINC250K graphs and outperforms standard MDM and several diffusion baselines in distribution alignment and property alignment.

### Strengths
1.An intuitive explanation and formal analysis of the "state-clashing" phenomenon are given, pointing out that fixed, element-independent forward occlusion makes it easy for different graphs to fall into intermediate states with poor distinguishability, resulting in a highly multimodal posterior and a model approximating a "unimodal, decompositional" distribution, which in turn produces solutions with high entropy and distribution shift. Formulas (3) and (4) are relatively clear with textual explanations.

2.Both unconditional (QM9, ZINC250K, Guacamol) and conditional generation are evaluated; it also includes ablation (fixed vs. learned scheduling, node/edge/node+edge) and the "number of intermediate states" metric to characterize state-clashing.

3.The training objective employs CE weighting by node and edge, and the gradient of the discrete sampling is discussed in the pass-through estimation; these are consistent with recent MDM literature.

4.The occlusion rate is learned at the element level to avoid large-scale collisions in the middle time step, and a differentiable ST-Gumbel training path is given.

### Weaknesses
1.The element-level kernel renders the forward process non-equivariant, meaning the intermediate state distribution is affected by vertex permutations. For molecular graphs, this contradicts the fundamental principle that isomorphism should not alter the generative distribution. Current methods merely introduce a learnable embedding H for each graph element and "randomly permutate columns" to "distinguish graph states with the same number of nodes/edges," but this does not restore the guarantee of permutation equivariance. It needs to be proven that this forward process, which breaks equivariance, does not induce dependencies on node labels and generalization issues, especially whether relabeling input nodes during testing maintains a consistent sampling distribution.

2.The abstract and main text claim that MELD is "the first diffusion model to achieve 100% chemigenicity in unconditional generation on QM9 and ZINC250K," but several MDM baselines in Table 1 also show 100%. The wording needs to be corrected.

3.The paper does not provide an explicit collision risk function or upper and lower bound analysis; the loss in Equation (3) does not directly minimize the "collision probability". It is suggested to provide a computable proxy metric and its relationship with the gradient direction, or to supplement the appendix with a simplified derivation of the "collision probability as a function of {𝑤_{𝑖}}".

4.The manuscript states that "unless otherwise specified, standard MDM and MELD use the same DiT backbone," but were the other discrete/continuous diffusion baselines in Table 1 also retrained and had their backbones and training budgets aligned? If comparisons are only made within the MDM family without aligning the backbones/hyperparameters of external distributed models, the conclusions may overestimate the advantages of MELD. Please provide the number of training epochs, GPU configuration, total duration, and FLOPs in the appendix, as well as the retraining/reproduction practices for each baseline.

5.The use of V.U.N.↑ in Tables 3 and 6 lacks a clear explanation of its meaning and calculation in the text (it seems to be a composite score for Validity/Uniqueness/Novelty?). Please define it explicitly at its first appearance in the text.

### Questions
1.The statement "first 100% validity" conflicts with Table 1. It is recommended to change it to "significantly reduced FCD/NSPDK while maintaining 100% validity." Could you please report the confidence intervals for inefficiency (multiple sampling)?

2.Can a more systematic comparison be made between the key differences and complexity of existing "adaptive/category-level" scheduling (such as DiffusionBERT, GenMD4, TabDiff) and the "element-level" scheduling in this paper? Currently, only a rough comparison is made in Table 3, lacking a theoretical analysis of the differences in expressive power.

3.Please list the number of training epochs, learning rate, scheduler, backbone, number of GPUs, and training time for all baselines; and specify which baselines were retrained by the authors and which were reproduced from the original paper.

4.Please add "Node relabeling robustness test" (variance of distribution index/property MAE under multiple labels of the same molecule).

5.Table 5 only performs isomorphism counting on 12 nodes/131 samples, which is costly but has a small sample size; it is recommended to provide estimation methods for larger scales (such as approximate GI or fingerprint hash upper/lower bounds) and statistical confidence intervals.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper investigates the application of Masked Diffusion Models (MDMs) to molecular generation. The authors identify a key limitation in standard MDMs, which they term the "state-clashing problem," where fixed, element-agnostic noise schedules cause distinct molecular graphs to collapse into identical corrupted states during the forward process. To address this, the paper proposes Masked Element-wise Learnable Diffusion (MELD), a framework that learns a flexible, per-element (atom and bond) noise schedule. Through extensive experiments on unconditional and property-conditioned molecular generation tasks, the authors demonstrate that MELD achieves 100% chemical validity on QM9 and ZINC250K and outperforms standard MDMs and other diffusion-based baselines in distributional and property alignment.

### Strengths
1. Originality and Significance. The paper makes a significant and original contribution by identifying the "state-clashing problem" as an obstacle to applying standard MDMs to structured data like molecular graphs. The core idea of learning an element-wise forward process to orchestrate distinct corruption trajectories is an elegant and insightful solution. 
2. Quality. The technical quality of the work is high. The hypothesis about state-clashing is well-motivated and convincingly demonstrated through both theoretical formalization and empirical analysis. The proposed MELD framework is a technically sound and well-designed solution. The experimental evaluation is comprehensive, covering multiple datasets (QM9, ZINC250K, Polymers, Guacamol) and tasks (unconditional, conditional), and benchmarking against a wide range of strong baselines.
3. Clarity. The paper is well-written and easy to follow. The state-clashing problem is introduced with intuition and supported by clear illustrations. The experiments are well-structured, and the results are clearly communicated through tables and figures.

### Weaknesses
1. The paper's central claim of superiority is undermined by an incomplete set of baseline comparisons. While MELD is shown to be effective against standard MDMs and some diffusion models, it omits a direct comparison to some relevant works. Methods presented in "Conditional Diffusion Based on Discrete Graph Structures for Molecular Graph Generation" and "Learning Joint 2-D and 3-D Graph Diffusion Models for Complete Molecule Generation" have demonstrated exceptional performance on ZINC250K benchmarks. These models achieve their results using strong denoising architectures but with simple, fixed noise schedules. 
This raises a critical question: Is the added complexity of learning the forward process truly necessary if a stronger denoising architecture with a fixed schedule can achieve similar results? The paper needs to more explicitly articulate the unique advantages of its approach beyond incremental performance gains. For instance, does MELD offer better parameter efficiency or faster training? Without a clear, compelling advantage, the rationale for introducing a learnable noise schedule, which slows down training, is weakened.

2. Ineffective Visualization in Figure 3. The visualization in Figure 3, which aims to demonstrate MELD's faster recovery during the reverse process, is not very effective. The overlaid text ("[MASK]") and molecular structures are small, cluttered, and difficult to parse. This makes it challenging to visually verify the claim that MELD reconstructs meaningful fragments earlier than the element-agnostic schedule. A clearer visualization would be much more impactful.

3. Missing Discussion of Training Overhead in Main Text. The paper introduces a learnable noise scheduling network and a joint optimization procedure, which inherently adds computational cost and complexity during training compared to models with a fixed forward process. However, the main text lacks any discussion of this overhead.  A summary of these findings should be included in the main paper to provide readers with a complete picture.

### Questions
1. To better situate your work, could you discuss and ideally provide an experimental comparison against more baselines with a fixed schedule?  This is essential to demonstrate the advantages of a learnable schedule over a powerful denoiser with a fixed schedule.
2. Could you revise Figure 3 for better clarity? The current visualization is difficult to interpret. A clearer format, such as using 2D renderings and highlighting unmasked elements, would more effectively demonstrate the claimed faster recovery of your method.
3. Could you add a brief discussion of the training overhead (e.g., increased training time) introduced by the learnable scheduling network to the main paper? This would help readers better understand the practical trade-offs of your approach.

### Soundness
3

### Presentation
3

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
This paper proposes a state-clashing problem, meaning that when masked diffusion models are applied to molecular generation, different molecules collapse into the same state, which increases the difficulty of learning. The authors propose an element-wise learnable method that alleviates this issue by learning different corruption rates.

### Strengths
1.	This paper explores the performance of the currently popular MDM in the field of molecular generation, which is a research topic worth pursuing.
2.	The authors make improvements based on the MDM by introducing element-wise embedding to adapt it to molecular generation tasks.
3.	The authors also validate the effectiveness of the method on large-scale datasets such as Guacamol.

### Weaknesses
1.	Some overclaims in the paper need clarification, such as the statement that in previous work the transition probabilities between elements in the forward process are all uniformly distributed.
2.	When proposing the state-clashing problem, the authors lack demonstrations on large-scale datasets. This makes it difficult to convince readers whether such a problem truly exists.
3.	The cases in Figure 2 are not easy to understand and require clearer explanation.
4.	Some parts of the method that are based on existing methods could be moved to the appendix.

### Questions
1.	The authors propose an element-wise learnable method that learns different corruption rates. Is there any suitable case study that has been analyzed to show the relationship between the learned different rates and the various types of elements?

### Soundness
2

### Presentation
3

### Contribution
2
