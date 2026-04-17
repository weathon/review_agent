# Let Physics Guide Your Protein Flows: Topology-aware Unfolding and Generation

- Decision: Reject
- Scores: 2, 2, 2, 2

## Abstract
Protein structure prediction and folding are fundamental to understanding biology, with recent deep learning advances reshaping the field. Diffusion-based generative models have revolutionized protein design, enabling the creation of novel proteins. However, these methods often neglect the intrinsic physical realism of proteins, driven by noising dynamics that lack grounding in physical principles. To address this, we first introduce a physically motivated non-linear noising process, grounded in classical physics, that unfolds proteins into secondary structures (e.g., $\alpha$-helices, linear $\beta$-sheets) while preserving structural integrity—maintaining bonds and preventing collisions. We then integrate this process with the flow-matching paradigm on $\mathrm{SE(3)}$ to model the invariant distribution of protein backbones with high fidelity, incorporating sequence information to enable sequence-conditioned folding and expand the generative capabilities of our model. Experimental results demonstrate state-of-the-art performance in unconditional protein generation, producing more designable and novel protein structures while accurately folding monomer sequences into precise protein conformations.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces PhysFlow, a protein–backbone generative model. Its key idea is a physics-inspired, nonlinear “noising” process that mimics protein unfolding. This process is formulated as a second-order decay–Hamiltonian system designed to prevent steric clashes during generation. The authors couple this physical process with flow matching on SE(3) generation. Experiments show that PhysFlow achieves excellent performance on both unconditional protein generation and sequence-conditioned monomer folding.

### Strengths
1. The work successfully adapts second-order dynamics to the complex geometry of protein structures, demonstrating a promising application of physical priors to protein deep learning domain. It can effectively reduce the clashes and atomic collisions.

2. The paper demonstrates compelling results across key benchmarks, especially in unconditional generation.

### Weaknesses
1. The most confusing aspect is the coordinate story in Section 4.2. Proposition 1 argues invariance in an angular representation and the forward ODE is defined in the latent $z$ (angle) space, yet the loss is pushed—via the chain rule—into Cartesian $x$-space, and the network architecture (e.g., IPA), loss function, part of energy also operate on $x$. What, then, is the precise role of the angular space? It seems that angular space is just used to introduce the Proposition 1, which is somehow meaningless (or can be substituted by $x$).
2. Angles and Cartesian coordinates are not globally bijective. They are only almost-everywhere invertible under restrictive conditions. In high-dimensional, complex systems it is easy to encounter angle configurations that do not map back uniquely to coordinates, errors accumulate through the transformation, and angles cannot encode bond lengths,  etc. How does the training ensure unbiased/accurate conversion between the two transformation?
3. The manuscript lacks an ablation to justify the necessity of second-order dynamics. While a second-order decay–Hamiltonian formulation is intuitively reasonable, the paper does not demonstrate that it is required. For example, would a first-order ODE with explicit collision-avoidance terms perform worse? Does the second-order just aim for novelty?
4. Typically, joint training of conditional and unconditional objectives slightly degrades folding performance—especially in the conditional setting. Could you report results for the conditional folding task on CAMEO2022 (e.g., TM-score, RMSD, GDT-TS, IDDT), and include stability/geometry metrics that may highlight PhysFlow’s advantages (e.g., Cα clash rate, peptide-bond breaks)?
5. Sections 1 and 2 are weakly written. The introduction lacks a broad discussion of the protein folding task and its relevance to drug design. A second-order Hamiltonian/Langevin formulation has been widely explored in diffusion and flow frameworks (underdamped/kinetic samplers, second-order flow matching, acceleration with momentum, etc.). The authors miss discussion about this content. By contrast, much of Section 2 and Section 4.1 feels peripheral to the main contribution.

### Questions
The same as Weaknesses. If the authors can address these concerns, I would be inclined to increase my score.

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper introduces PhysFlow, a generative model for protein backbone structures. The authors' core motivation is that standard diffusion and flow models use a "noising" process that is not physically realistic, ignoring principles like topological integrity and steric clashes. To address this, they propose a physics-inspired, non-linear "unfolding" process grounded in Hamiltonian dynamics. This forward process is designed to unfold a protein into a secondary structure (like a $\beta$-sheet) while explicitly preserving bonds and using a Coulomb-like repulsion term to prevent residue collisions.This physics-driven process is integrated into the flow-matching paradigm on $SE(3)$ to learn the distribution of protein backbones. The model also incorporates sequence information, enabling it to perform both unconditional backbone generation and sequence-conditioned folding. The authors claim that PhysFlow achieves state-of-the-art performance in unconditional generation, producing more designable and novel structures, and accurately folds monomer sequences.

### Strengths
The paper's motivation to incorporate more physics into the generative forward process is a novel and interesting research direction. The idea of replacing a generic noise process with a more structured unfolding trajectory that respects physical constraints like collision avoidance is intuitive.

### Weaknesses
1. The paper proposes a complex, non-linear, second-order Hamiltonian system for the forward unfolding process but completely fails to discuss or justify the reverse generative process. It is a major theoretical gap to assume that a standard, first-order flow-matching objective is sufficient to learn the true time-reversal of such a complex non-linear dynamic system, which is required to satisfy the Continuity Equation or the Fokker-Planck equation. The original reverse processes for diffusion models or flow matching generally cannot be directly applied here without theoretical modification.
2. The model is explicitly designed to unfold to a target distribution $p_0$ defined as a linear $\beta$-sheet, yet the empirically generated samples are biased toward the $\alpha$-helix structure (82.2%). This contradiction fundamentally weakens the paper's core claim, suggesting that the physics-guided forward process is either not being reversed correctly or has little impact on the final generated distribution.
3. The paper's motivation, that a "physically plausible" forward trajectory is necessary, is questionable. The power of modern generative models originates from the strong generalization ability of the denoising network, which requires the forward process to be stochastic and highly noisy to enable learning recovery from arbitrary, highly corrupted states. Restricting the model to a single, well-defined forward path may hurt its ability to generalize to novel states not encountered on this narrow trajectory.
4. The proposed algorithm requires an expensive pre-processing step to simulate and store the forward trajectory for every protein. This prevents on-the-fly training and introduces a critical scalability bottleneck, especially since the forward simulation takes over one second per protein (as shown in Figure 4), severely limiting its application to larger datasets like AlphaFold DB.
5. The evaluation is undermined by non-standard and weak comparisons. (1) The authors compare against some models trained on shorter sequences (e.g., up to 256 aa) while generating samples up to 300 aa, leading to an unfair comparison. (2) They exclude stronger baselines (like Proteina and Genie2) with the excuse of using different datasets, even though this paper uses a self-curated dataset as well. (3) The comparison for the folding task (Table 3) is made against co-design models (MultiFlow, FoldFlow-2) rather than specialized, state-of-the-art folding models, making the claim of strong folding performance unconvincing.
6. The reported "Diversity Cluster Ratio" is affected by the designability value. For a clear assessment of structural variety, the authors should report the absolute number of designable clusters instead, which is a more robust metric for diversity.
7. The paper lacks definitions for key terms, specifically "linear diffusion" in the first paragraph of Section 3 and "unfolding flow" (Line 187).
8. The choice to set the target state $p_0$ from a beta-sheet distribution is arbitrary and unexplained. Ideally, the model should be able to generate any secondary structure, so the selection of a single predefined distribution for the "unfolded" state needs strong justification.
9. The paper is lacking crucial hyperparameter configurations for the forward simulation, including $\gamma$ (drag coefficient), $\sigma_\beta^2$, $\mu_\beta$ (for the target distribution), and $\sigma_v^2$, hindering the reproducibility of the work.
10.  In Table 1, the categorization of existing models into DDPM, CFM, OT, or DSM is an oversimplification, as these algorithms are well-known to be deeply related and can be unified under the Stochastic Interpolant framework.
11. Equation (3) is wrong, as in flow matching, there is only one ODE. The reverse process simply simulates the same ODE in the reverse time direction.

### Questions
See above.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors propose a new non-linear noising process for protein structures that is inspired by Hamiltonian dynamics that is supposed to reduce clashes and steric violations and train a structure generation model similar to FoldFlow with it. They compare it to other models on unconditional backbone design and sequence conditioned monomer folding.

### Strengths
[S1] Novel approach: the authors depart from the common linear noising schedule and propose an interesting physics-inspired noising process, and introduce modifications to their model training such as the Look-Ahead loss that help their model to learn this denoising objective (an ablation of this would add to the paper).

[S2] Comparing to models trained on the same dataset for structure generation, they have favorable performance in unconditional backbone design.

### Weaknesses
[W1] Theu authors claim that linear noising processes make “models prone to steric clashes and violations”. However, in previous work there has been extensive validation that models with linear noising processes and without any auxiliary losses have very strong biophysical plausibility with clashes and violations on the level of experimental structures (Geffner, Tomas, et al. "La-proteina: Atomistic protein generation via partially latent flow matching." arXiv preprint arXiv:2507.09466 (2025).). The authors do not substantiate their claim with evidence, I would therefore recommend to remove this claim which most of their motivation relies on; there is no evaluation of biophysical plausibility, steric clashes or violations in the paper. Therefore the motivation for the topology-aware unfolding process seems weak and its advantage not demonstrated. In addition, the authors themselves use auxiliary losses to supervise bond distances etc and avoid clashes; if the proposition of the authors would be correct their method should not need to rely on these auxiliary losses since other methods achieve high performance without them.

[W2] the authors claim that “PhysFlow achieves state-of-the-art performance in unconditional protein backbone generation and sequence-conditioned monomer folding”. Both of these statements are not true: in Table 2 PhysFlow is clearly worse than a number of methods, including RFDiffusion, Chroma and Proteina. The fact that these models are trained on different datasets is no excuse: the authors could themselves have trained their model on these openly available datasets. So claiming SOTA performance while ignoring all strong baselines is not justified. Similarly for folding performance: they show that their method achieves 11-15A folding performance, making it useless from a practical perspective. It is way worse than ESMFold, and again just because ESMFold was trained on a different dataset that does not justify claiming SOTA performance by ignoring that method.

### Questions
[Q1] In practice researcher use these models for conditional tasks such as binder design or moti scaffolding. Do you think your current physical noising process is suitable for these tasks or would you do additional adjustments to it to account for the different tasks?

[Q2] Many modern protein structure generation generate not only the backbone structure but all-atom structures including sequences. Can your framework be used in that respect, and if yes how?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
PhysFlow introduces a physics-inspired non-linear noising process that targets the process of unfolding through secondary structures to avoid clashes. The central idea is to mimic the folding process during de novo design. PhysFlow leverages SE(3) flow matching along with sequence conditioning for both uncodnitional backbone design and forward sequence conditioned protein folding.

### Strengths
- Novel flow matching procedure. Clearly motivated and defined. The change from gaussian prior to leveraging Riemanian flow matching to go from linear sheet to de novo protein is very interesting.
- This framework allows for both structure generation and sequence-conditioned folding using SE3-equivariant architectures and a unified backbone/sequence conditioning, which enables accurate folding performance compared to some baseline models.

### Weaknesses
While interesting it seems to be an overly complex framework with little empirical benefits. PhysFlow seems to suffer from the  structural bias of early AlphaFold2-architecture-based backbone design methods as well. The motivation seems 

- No analysis of the claim of linear flow models prone to steric clashes. Further the method cited RNA-Frameflow does not make this claim either.
- State-of the-art claims "PhysFlow generates the most designable sample" yet table 2 shows prior work with better designability, diversity, and novelty. Furthermore FoldFlow (OT) achieves 97.2% designability from lengths 50-250 as that matches its training length distribution from Geffner et al. Similarly Proteina is < 99% in their reporting. Overall it does not seem like a fair comparison as methods are compared outside of their training data length.
- Like FoldFlow, PhysFlow has significant alpha helix biased which is known to inflate designability. 
- Folding benchmark not the same evaluation set as MultiFlow and FoldFlow2? Is there a reason a different evaluation was used? Could very well be a fair comparison but it is odd that FoldFlow-2 goes from ~3.2 to > 13 RMSD.
- No motif scaffolding when refereced as part of the motivation compared to prior backbone generative models, which likely may benefit from having a more physics grounded denoising process to place the motif correctly.

### Questions
- How long was the model trained for (time/steps)? 
- What is the value of preserving physics during the generative process if only the final samples are evaluated and its backbone only compared to recent all-atom models?
- How does PhysFlow perform when trained on AFDB like recent state-of-the-art methods, Genie2 and Proteina? Also how long would it take to process AFDB to train PhysFlow to compared to those AFDB-using methods?
- How do the model size and inference speed compare to prior methods?
- Is there any reason why the physics-informed flow matching could not be used on cartesian CA-based models? Or is the SO(3) and angular factorizations a requirement?

### Soundness
2

### Presentation
2

### Contribution
2
