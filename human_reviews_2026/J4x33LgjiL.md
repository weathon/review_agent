# AbFlowNet: Optimizing Antibody-Antigen Binding Energy via Diffusion-GFlowNet Fusion

- Decision: Reject
- Scores: 4, 6, 6, 4

## Abstract
Complementarity Determining Regions (CDRs) are critical segments of an antibody that facilitate binding to specific antigens. Current computational methods for CDR design utilize reconstruction losses and do not jointly optimize binding energy, a crucial metric for antibody efficacy. Rather, binding energy optimization is done through computationally expensive Online Reinforcement Learning (RL) pipelines rely heavily on unreliable binding energy estimators. In this paper, we propose AbFlowNet, a novel generative framework that integrates GFlowNet with Diffusion models. By framing each diffusion step as a state in the GFlowNet framework, AbFlowNet jointly optimizes standard diffusion losses and binding energy by directly incorporating energy signals into the training process, thereby unifying diffusion and reward optimization in a single procedure. Experimental results show that AbFlowNet outperforms the base diffusion model by 3.06% in amino acid recovery, 20.40% in geometric reconstruction (RMSD), and 3.60% in binding energy improvement ratio. AbFlowNet also decreases Top-1 total energy and binding energy errors by 24.8% and 38.1% without pseudo-labeling the test dataset or using computationally expensive online RL regimes.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces AbFlowNet, a framework that unifies diffusion models and GFlowNets for antibody–antigen design.The core idea is to reinterpret each denoising step in a conditional diffusion process as a trajectory transition within a GFlowNet, where the terminal state corresponds to a fully denoised CDR sequence + backbone, and its reward is proportional to binding energy (computed offline via Rosetta InterfaceAnalyzer). Training uses a Trajectory Balance (TB) objective jointly with standard diffusion reconstruction losses, allowing reward information to propagate along the entire denoising trajectory without requiring online energy evaluation.

### Strengths
This work is the first to treat diffusion denoising trajectories as GFlowNet paths, which is an elegant conceptual unification, and seems to achieve good results.

### Weaknesses
1. In my experience, the Rosetta energy evaluation is quite cheap and i do not think online RL is an issue. Especially considering the protein design area, the real cost is never the simulation efforts but the wet-lab evaluation. So i think online RL makes more sense

2. The Rosetta evaluation is not that reliable. Does the author have any wet-lab exps?

3. Eq.(18) introduces an extra hyperparameter w. It can be quite tricky to tune this w for different datasets.

4. In fact, there has been work using RL to fine-tune diffusion models by treating denoising trajectories as paths. https://arxiv.org/abs/2305.13301 but i did not see the authors compare it or mention it. RL is closely relateed to GFlowNet so this denoising traj as path contribution is not that strong considering this paper. 

5. The authors seem to retrain diffab but there is already diffab checkpoint. Why re-train it? 

6. Also, the authors seem to emphasize the same gradient update step for baselines. I do not think that is the real metric we should care about or i think we should not care that metric at all. First, the real metric should be sth like FLOPS, as the gradient step for different model/algo can be different. Second, as i mention before, we should not care about much about computational cost in the protein domain, and it does not matter as much as we did in general ML area (the comp is not large in bio domain or that important).

### Questions
See Weakness

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
This paper proposed AbFlow, a combination of diffusion models and GFlowNet for antibody design. AbFlow interprets each diffusion step as a GFlowNet state, and use the trajectory balance loss to propagate Rosetta binding energy rewards. This model improves upon AAR, RMSD, and top-1 Rosetta binding energy.

### Strengths
- This model unifies data and physical prior through the combination of diffusion and GFlowNet.
- It is more efficient than previous preference optimization-based methods which require collection of massive training pairs.
- It demonstrates performance improvement compared to previous antibody design models (GNN-based and diffusion-based).

### Weaknesses
- Evaluation metrics are outdated. Only Rosetta scores are computed which is however noisy and sensitive to minor perturbation. A more reliable evaluation method that has  been used more recently is using AlphaFold to predict the complex structure and compare the RMSD between generation and AlphaFold prediction.

### Questions
- Since Rosetta is very noise-sensitive, is it possible to use AlphaFold quality predictor such as PLDDT, which serves as a good proxy for binding in previous studies (e.g. BindCraft, EvoBind), as the reward?

### Soundness
3

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
4

### Summary
The authors reframe the generative design of antibodies by integrating (current) diffusion models with the GFlowNet framework, such that the diffusion step is a state in the GFlowNet framework. This allows them to jointly optimize standard diffusion loss and binding energy in the training procedure, unifying diffusion and reward optimisation in a single procedure, as a competitive approach  to reinforcement learning in which binding energy is based on unreliable estimators. As binding energy can only be computed after complete denoising, they use the trajectory balance objective (TB) to propagate rewards back through the diffusion process. As a consequence, the related rewards for transitions in GFlowNet can be precomputed for each CDR in the training dataset. The resulting method is competitive with a current online reinforcement strategy to post-train a diffusion model (DiffAb) to optimize CDR binding energy.

### Strengths
- nice idea to integrate GFlowNet framework with a diffusion model to enforce binding energy constraints in training of generative antibody designs
- clever use of the trajectory balance objective

### Weaknesses
- parts are repetitive (intro/related work) wheras other (critical) parts where too condensed (trajectory balance)
- solution relies on trajectory balance objective, which has been proposed in earlier work
- somewhat limited set of metrics used in main text (RMSD,AAR...); comparison on these other metrics is also not trivial/conclusive
- could have expanded the results more top gain more insight in performance, eg not restricting to top-1 metrics but also include distributions of generated designs
- how would you include multiple objectives into this framework?
- Fig 3, explain shading of structure (be more clear on visual differences, instead of only adressing DG)

### Questions
Please address points of identified weakness (most textual, but 4th point also with additional figures/tables).

### Soundness
3

### Presentation
2

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
This paper introduces AbFlowNet, a generative framework for structure-based antibody redesign with additional emphasis and consideration of the force-field estimate binding energy.

### Strengths
- Authors introduce a method allowing to jointly optimise for standard generative / diffusion objectives and binding energy estimated through force field methods.
- Benchmarking on standard datasets / splits reveals slight improvements across investigated standard metrics (amino acid recovery, RMSD to wild-type conformation).

### Weaknesses
- I find the benchmarking to be not convincing, which is reflected in my score. Qualitative example shown in Fig 3 is not clear at all, even for a person with trained structural eye and energy differences are miniscule.

### Questions
- Following on the highlighted weakness point - I’d suggest authors to consider showing some properties of generated sequences and how they overlap with some reference distribution. There are multiple datasets with several binders and non-binders around wild-type sequence with corresponding structure information. Furthermore, highlighting generative abilities of the model on this data (e.g. are affinity improving mutations suggested by the model more frequently compared to other baselines? or, if possible within the framework, is the estimated likelihood of improving variant compared to wild-type better?) would allow to assess the improvements postulated by the authors without direct experimental validation.

### Soundness
3

### Presentation
3

### Contribution
2
