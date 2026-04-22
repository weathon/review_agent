# Unified Biomolecular Trajectory Generation via Pretrained Variational Bridge

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 8, 4, 8, 4

## Abstract
Molecular Dynamics (MD) simulations provide a fundamental tool for characterizing molecular behavior at full atomic resolution, but their applicability is severely constrained by the computational cost. To address this, a surge of deep generative models has recently emerged to learn dynamics at coarsened timesteps for efficient trajectory generation, yet they either generalize poorly across systems or, due to limited molecular diversity of trajectory data, fail to fully exploit structural information to improve generative fidelity. Here, we present the Pretrained Variational Bridge (PVB) in an encoder-decoder fashion, which maps the initial structure into a noised latent space and transports it toward stage-specific targets through augmented bridge matching. This unifies training on both single-structure and paired trajectory data, enabling consistent use of cross-domain structural knowledge across training stages. Moreover, for protein-ligand complexes, we further introduce a reinforcement learning-based optimization via adjoint matching that speeds progression toward the holo state, which supports efficient post-optimization of docking poses. Experiments on proteins and protein-ligand complexes demonstrate that PVB faithfully reproduces thermodynamic and kinetic observables from MD while delivering stable and efficient generative dynamics.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces the Pretrained Variational Bridge (PVB), a unified generative framework for biomolecular trajectory generation that leverages pretraining on static 3D structures and finetuning on coarse-grained MD trajectory data.
The central idea is to bridge the gap between single-structure pretraining and trajectory-conditioned finetuning with a unified objective:  $\mu(X_1 \mid X_0)$. 

During pretraining, the model learns from diverse molecular structures to capture cross-domain structural knowledge.
During finetuning, the model is finetuned on paired transition data $(X_t, X_{t + \Delta t})$ from dynamic datasets (e.g., ATLAS and MISATO).

An additional innovation is the RL-based adjoint finetuning using stochastic optimal control, enabling direct optimization for holo state generation in protein–ligand systems.

Empirical results show that PVB reproduces thermodynamic and kinetic observables (Rg, torsion, TIC projections, MSM occupancy) with stability comparable to MD and substantial improvement over baselines (ITO, MDGen, UniSim) in validity (VAL-CA = 0.97) and decorrelation metrics.
In protein–ligand docking, PVB with RL finetuning outperforms AutoDock Vina and non-RL variants.

### Strengths
* PVB elegantly integrates structural pretraining and trajectory learning through a shared encoder–decoder bridge, aligning objectives across domains. Pretraining on heterogeneous biomolecular structures allows transfer to proteins, small molecules, and protein–ligand complexes without retraining.

* The adjoint-based stochastic control formulation enables memory-efficient fine-tuning toward functional states (e.g., holo forms) without additional networks.

* PVB consistently achieves better or comparable results to classical MD and generative baselines in reproducing both kinetic and thermodynamic observables.

* The RL variant shows meaningful progress toward real drug-design applications, improving ligand placement beyond traditional docking and static generative methods, which can serve as an alternative method for docking.

### Weaknesses
* While conceptually elegant, the experimental scope is somewhat narrow and the demonstrated benefits are modest under realistic scales. The datasets (ATLAS, MISATO) are relatively small, and the observed improvements, though consistent, are incremental, especially given that baseline MDGen and UniSim already yield physically valid trajectories. Including results on the recently released MDCATH dataset will be make the manuscript stronger.

* As mentioned by the authors, the generation remains sequential, limiting scalability to long timescales or high-throughput ensemble generation. The paper does not analyze the runtime of the PVB for trajectory generation, which is also a concerning fact. While coarse timesteps improve efficiency conceptually, inference speed, wall-clock cost, and scaling to larger systems (e.g., >10⁴ atoms) remain unreported.

* The claimed cross-domain transferability is supported only by protein and protein–ligand tasks; other molecular domains (RNA, materials, polymers) are underexplored.

### Questions
1. I am particularly interested in the experimental results on the large-scale MDCath dataset, as well as the runtime analysis of the proposed method. Could the authors provide more details or quantitative comparisons to illustrate the computational efficiency and scalability of PVB?

2. Is it possible to extend the proposed framework for parallel trajectory generation, rather than sequential sampling? This could further improve scalability, especially for long biomolecular simulations.

3. Lastly, could the authors elaborate on the role of the latent variable X0? Specifically, in the statement “The latent variable X0 is introduced to avoid the collapse of the conditional probability µ(X1|X0) from degenerating into a Dirac delta measure," it would be helpful to clarify why this latent variable is necessary and what would happen if one directly generated X1 conditioned on X0 without it.

### Soundness
3

### Presentation
3

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
The paper proposes a method for coarse-grained molecular dynamics simulation, using Brownian bridges. More specifically, they first go to a latent state, and then to the target state. This allows generalization over both single-structure and paired trajectory data, especially in terms of pretraining. Method is evaluated on relevant datasets.

### Strengths
- The paper works on a relevant problem.
- Optimal control methods are leveraged for more efficient training.
- The method is evaluated on relevant benchmarks.

### Weaknesses
- The explanation of the theory and the notation is quite confusing. I sympathize that this is not trivial, especially with having in a mind a relatively broad target audience from diverse research backgrounds. But considerable effort should be made to improve the writing. I try to make some concrete suggestions below. I'm certainly willing to raise my score if readability is improved!

### Questions
- Fig. 1. This can be a very informative figure, but the elements are quite small (the arrows and black dot). Consider indicating on the figure what the meaning is of the three small modes on the left and the big one on the right.
- Why do you use the rmsd, and not the (Gaussian) log likelihood?
- At the start of 3. Method, Z and C are defined, but are not used in the remaining sections. Do you also model these? If so, how exactly?
- Why exactly do you use \mu vs. p? They both indicate probability measures I assume?
- Please define more clearly what \mu, and X are on line 41. In general, please take some time to rethink where you define the different mathematical concepts and objects. Now it's a bit all over the place and does not seem to follow a structured explanation, or logical build up in your story.
- eq. 11, why is there a gradient stop on u when sampling Y_0:1?
- footnote on line 225. Please don't put this in a footnote, it is very confusing! The bridge is from \tilde X to X_1, correct? Why not call it Y from the start, please take some time to think about this, I'm sure there's ways to make this paper much more readable if you decide on certain notations from the start and don't start changing/adding things in the middle of your explanation.
- Prop. 3.2: You are solving an optimal control problem analytically, correct? Is solving the ODE for \tilde a related to solving the HJB equation? Can you please compare your approach to [1], where this is done for the same kind of ELBO, for linear SDEs?

[1] https://arxiv.org/abs/2505.17150


Minor suggestions:
- line 92. RL is not defined.
- line 102: 'applying'
- I would not use (so much) abbreviations in the abstract, it does not improve readability, and RL is not defined. The abstract is quite wordy which makes it also harder to read (e.g. 'inefficiency' -> cost, 'nevertheless', 'remarkable', ..)
- please use larger brackets and ||, e.g. \left[ \right], see eqs. 5, 7, ..
- line 57: 'domain' (singular)
- line 234: 'prove', this sentence is also not clear?
- line 278: Y=Y, very confusing. I suppose one is a stochastic variable and the other one is a realization. You could e.g. use small letters for the realizations.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces pretrained variational bridge (PVB). The core contribution is a unified framework that first pretrains an encoder-decoder model on a large and diverse dataset of single, static molecular structures to learn generalizable structural features. This pretrained model is then finetuned on paired molecular dynamics (MD) trajectory data to learn system-specific dynamics. Furthermore, the paper presents RL finetuning procedure using adjoint matching, which efficiently optimizes the model to guide trajectories toward specific target states, such as the holo conformation in protein-ligand docking.

### Strengths
- The paper proposes a novel methodology to include pretraining on datasets with static structures but diverse chemical space, and then finetuning on dynamical data. This enables the model to achieve better chemical transferability despite the limited chemical space coverage of the dynamical data.
- The integration of RL with adjoint matching for pose-optimization in docking is a novel application. And the authors have shown the improvement in the ligand pose after the finetuning.
- The model's performance is thoroughly benchmarked across multiple demanding tasks, including protein dynamics, protein-ligand complex dynamics, and holo state exploration. The comparison against several relevant baselines on different datasets demonstrates the effectiveness of the method
- PVB shows outperformance over baselines across most metrics.
- Ablation studies have been performed to show the benefit of pretraining and finetuning procedure

### Weaknesses
- While the paper evaluates against other trajectory-based models, it assesses performance on free energy landscapes. While most of the metrics compared in the paper are actually thermodynamic properties, they can be evaluated with i.i.d. (time-agnostic) sampling models. It will be helpful to benchmark against those methods as well.
- In the meanwhile, although the time-dependent model describes dynamics, it's not obvious from the benchmarks and applications shown in the paper why the time-dependence is needed, what is its advantage over i.i.d. sampling model. It will help to justify the motivation if the authors can clarify that (time-dependence makes the model to describe thermodynamics better than i.i.d. sampling model) or show some cases when kinetics/dynamics are of practical interests in applications.
- The rationale behind using two separately finetuned models for the protein (ATLAS) and protein-ligand (MISATO) tasks is not explained. One might expect that a single model finetuned on both could offer better transferability, especially for the protein component of the dynamics.

### Questions
- Have the authors checked the physicality of the ligands (and proteins as well)? Not only the bond break or clashes, but also stereochemical errors. Does that get better or worse with RL finetuning?
- The paper claims cross-domain generalization, but it is not specified whether the train/test splits of datasets were performed based on sequence similarity or other metrics to prevent data leakage and rigorously test generalization to unseen protein folds.

### Soundness
4

### Presentation
4

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
The paper proposes a strategy for developing next-step MD emulators by first pretraining on a bridge that maps between distributions of static structures. This allows models trained on vast amounts of rich static data to be easily tuned on dynamics prediction. The authors show that models pretrained on the PDB and PDBBind can be fine-tuned on ATLAS (protein simulations) and MISATO (protein-ligand simulations) to replicate observables. The authors also develop a RL training strategy for the bridge to steer rollouts towards the holo state of a protein-ligand complex.

### Strengths
The idea of pretraining a bridge to recapitulate the initial state, and then fine-tuning it to produce the evolved state, is quite interesting. The work also touches upon simulation of protein-ligand simulations, which have been somewhat neglected in the ML for MD literature, despite their significant practical importance.

### Weaknesses
**Method**
* The RL formulation of the holo complex finetuning task seems gratuitous. In particular, if the reward is the RMSD to the holo state, why can't the holo state be used in a supervised fine-tuning fashion? If would seem that if the reward is simply the similarity to an explicit, known state, that is the setting of supervised learning, not reinforcement learning.

**Experiments**
* There are missing controls that make the value of the pretraining bridge hard to interpret. What if we pretrain without a bridge, such as AlphaFlow (with templates)? What if we don't pretrain at all, but use the same architecture? (I assume the retrained ITO baseline is using the ITO architecture).
* Although I am willing to judge these as not the focus of the paper, the protein-ligand docking evaluations are extremely sparse - the single baseline is AutoDock Vina, despite vast amounts of recent literature.

### Questions
What is the state matrix in Figure 3, right? The caption says "Probability differences between PVB
and MD across the 10 metastable states estimated by MSM." --- this shouldn't be a matrix, then.

### Soundness
3

### Presentation
2

### Contribution
3
