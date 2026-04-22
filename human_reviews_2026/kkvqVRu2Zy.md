# Constrained Diffusion for Protein Design with Hard Structural Constraints

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 8, 2, 6, 6

## Abstract
Diffusion models offer a powerful means of capturing the manifold of realistic protein structures, enabling rapid design for protein engineering tasks. However, existing approaches observe critical failure modes when precise constraints are necessary for functional design. To this end, we present a constrained diffusion framework for structure-guided protein design, ensuring strict adherence to functional requirements while maintaining precise stereochemical and geometric feasibility. The approach integrates proximal feasibility updates with ADMM decomposition into the generative process, scaling effectively to the complex constraint sets of this domain. We evaluate on challenging protein design tasks, including motif scaffolding and vacancy-constrained pocket design, while introducing a novel curated benchmark dataset for motif scaffolding in the PDZ domain. Our approach achieves state-of-the-art, providing perfect satisfaction of bonding and geometric constraints with no degradation in structural diversity.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This author proposed a method to solve a gap in protein backbone generation: enforcing hard functional and stereochemical constraints during diffusion-based design. The authors rproposed emthod use reverse diffusion with a predict–prox–renoise procedure: (i) predict a clean structure, (ii) apply a proximal feasibility update (with a Moreau–envelope penalty that tightens over time), and (iii) re-noise to stay on the data manifold. They further decouple local stereochemistry from global topology via a consensus ADMM split, enabling separate proximal updates that are warm-started across steps. Theoretical results bound constraint violations and motivate schedules for the penalty and trust parameters. Empirically, the method is evaluated on (1) a new PDZ motif-scaffolding benchmark and (2) vacancy-constrained pocket design with box/void geometry. Results claim perfect constraint satisfaction and strong usable rates vs. RFdiffusion baselines and constraint-guided SMC, while maintaining structural diversity and reasonable compactness.

### Strengths
The predict–prox–renoise design is simple, modular, and theoretically grounded; the scheduling guidance (λ↑ as σ↓; η tied to diffusion variance) is clear and actionable. ADMM splitting of local vs. global constraints is well-motivated for proteins (stereochemistry vs. long-range couplings) and yields a practical proximal scheme with consensus guarantees under mild conditions. Theorems quantify contraction of constraint violations and give scheduling rules that drive terminal feasibility, aligning with the method’s design. On the PDZ benchmark, baselines produce no usable designs whereas the proposed method yields 21% usable (up to 83% for well-posed ligands) with perfect constraint satisfaction and better diversity/compactness; similarly strong wins appear in the vacancy-constraint task.

### Weaknesses
Feasibility guarantees rely on prox-regularity; highly nonconvex sets common in protein design may violate this, limiting formal guarantees. Proximal/ADMM steps add overhead and may require careful tuning of λ/η/ρ schedules; guidance on sensitivity and robustness across architectures and tasks could be expanded. Results are strong on PDZ and a geometric vacancy constraint, but broader functional constraints (e.g., catalytic geometry, interface polarity, multi-chain assemblies) and end-to-end sequence design with wet-lab validation remain future work.

### Questions
1. Beyond PDZ scaffolding and geometric vacancy control, which biologically grounded constraints (e.g., catalytic triad geometry, specific H-bond networks) most motivate hard-constraint enforcement, and how will the framework extend to them?
2. Can you quantify how often unconstrained RFdiffusion fails due to global vs. local constraint violations in real pipelines, to better motivate where hard constraints matter most?
3. How does enforcing exact feasibility impact downstream functional success (e.g., binding affinity predictions) compared with soft guidance—any surrogate metrics supporting the motivation? 
4. The Moreau-envelope penalty transitions toward hard constraints as λ→∞. How sensitive are outcomes to the λt and ηt schedules, and do you provide heuristics or automatic tuning beyond the variance-linked choice?
5. In the ADMM split, how do you select which residues belong to the “local” vs. “global” blocks in more complex topologies, and what is the computational cost per sweep as structure size grows?
6. On the PDZ benchmark, your method attains 21% usable overall (up to 83% for favorable cases). What characteristics define “well-posed” ligands, and how general is that 83% across targets?
7. In the vacancy-constraint experiment, could you provide additional quality metrics (e.g., secondary-structure preservation per segment, clash scores) to complement constraint satisfaction and radius of gyration?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper presents a constrained diffusion framework for protein design, aiming to enforce hard structural and functional constraints.1 The core method is a "predict-prox-renoise" stochastic proximal method, which applies feasibility corrections to the predicted clean state $\hat{x}_{0}^{t}$ rather than the noisy state $x_t$.1 The authors also propose an ADMM decomposition to decouple local and global constraints.1 The method is evaluated on two tasks: motif scaffolding (using a newly introduced PDZ benchmark) and vacancy-constrained pocket design, claiming 100% constraint satisfaction rates where baselines achieve 0%.

### Strengths
Novel and well-motivated "predict-prox-renoise" method for hard constraints.   

(Superficially) perfect 100% constraint satisfaction on complex tasks.   

Contributes a new, curated benchmark dataset for PDZ motif scaffolding.

### Weaknesses
Fails to cite or compare against any true SOTA competitors in motif scaffolding (e.g., Genie 2 , OriginFlow ), invalidating its SOTA claim.   

The "Ours" method in Exp 2 is a combination of the new method and the baselines , making results impossible to attribute.   

The ADMM method  is not general , and the theory  does not apply to the problem.   

 0% success for all baselines is not credible.

### Questions
see weakness

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents a constrained diffusion framework for precision engineering of detailed molecular features in proteins. Authors validate their approach in silico on a contributed tasks and novel curated benchmark datasets.

### Strengths
- This is definitely an interesting and strong contribution focusing on a slightly neglected aspect that is however of very high practical & theoretical relevance in precision engineering of protein architectures & binding sites.
- The proposed solution is benchmarked against convincing baselines and authors report significant improvements.
- Code & benchmark data is released.

### Weaknesses
- I feel a benchmark is scarce, it’s a relevant problem but I would welcome an effort to introduce more interesting cases. There are many problems requiring precision engineering that could show that the method is robust to different cases. E.g. one widely recognised problem used to study precision protein engineering for long time is parametric protein design with coiled coil bundles.
- While the technical / implementation aspects of the method are properly described I think the formulation and description of the proposed benchmark dataset is less understandable (I suppose even for struct/bio readearship, not mentioning purely ML audience). I suggest to rewrite this section so it reads more clear (also minor thing - the PBM is never defined in the paper).
- Finally, I believe being able to come up with backbones that satisfy the complex geometric criteria is an important contribution I have one follow up q to the authors - are these backbones designable? I miss the experiment that would prove (in silico would naturally be enough given the venue) that indeed we can design compatible sequences that will fold into structures that are predicted by the authors e.g. with the ProteinMPNN / AF2 pipeline that authors already set up.

### Questions
See weaknesses for potential discussion points.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a framework for incorporating hard structural constraints into diffusion-based protein design. The authors identify that existing guidance methods often fail to satisfy precise, non-convex constraints required for functional design. They propose viewing the reverse diffusion process through the lens of proximal optimization. Their empirical results on PDZ scaffolding and molecule encapsulation show state-of-the-art constraint satisfaction (near 100% vs. about 0% for the baselines).

### Strengths
The overall idea of the proposed method is creative and explores a new direction for diffusion models. I see the main strengths of the paper as follows:
- strong results: the 100% satisfaction on PDZ vs 0% for strong baselines appears meaningful
- method: framing the reverse step as proximal optimization is a neat idea, and applying the correction to the estimated $\hat{x}_0$ instead of the noisy $x_t$ is well-motivated
- benchmark: the PDZ benchmark designed for the evaluation seems like a meaningful contribution in its own right

### Weaknesses
I believe there are a few minor weaknesses:
- baselines: while RFDiffusion is a good baseline, the authors should compare to the method in Christopher et al. (2024) [1] given the similarity between the proposed methods.
- the paper does not present any sensitivity studies for hyperparameters. How sensitive is the proposed approach to the schedule $\lambda$, the number of ADMM iterations, etc.?

[1]  Christopher et al., Constrained Synthesis with Projected Diffusion Models, 2024.

### Questions
- How sensitive is the proposed approach to the schedule $\lambda$, the number of iterations, etc.?
- How does the proposed method perform compared to the method in Christopher et al. (2024)?
- Did you try a baseline projecting $x_t$ directly? Did it fail as expected?

### Soundness
3

### Presentation
4

### Contribution
3
