# DoMiNO: Down-scaling Molecular Dynamics with Neural Graph Ordinary Differential Equations

- Avg Score: 3.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 6, 4

## Abstract
DoMiNO: Down-scaling Molecular Dynamics with Neural Graph Ordinary Differential Equations

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes a hierarchical neural graph ode to simulate molecular trajectories on K different time scales, resulting in a reduced time complexity scaling, from O(T) down to O(KT^1/K) with K=3 in practice. The model uses K distinct graph neural networks that are integrated in time, and the latent features of the previous coarsest time scales feed into the features of the finer timescale. The authors show reduced mean-squared errors in forward time step prediction compared to a handful of time series prediction models on some small-scale molecular benchmarks of single-molecule trajectories.

### Strengths
The authors tackle an important problem with a creative and, in principle, intuitive idea. The reduced scaling from O(T) down to O(KT^1/K) is significant and the empirical results are encouraging

### Weaknesses
While the idea has potential and initial results are encouraging, i am currently not convinced that the results generalize to practical settings:

Only small-scale single-molecule trajectories are used, and the model only evaluates MSE. This leaves several open questions:
1. How does the model transfer between molecules?
2. How does the model generalize to larger more interesting systems?
3. Can the model successfully predict transitions between different conformations not seen during training? Can the model, for example, simulate the folding of proteins or reactions of molecules?
4. Currently, there is no temperature or initial velocity input. The velocity/temperature is implicitly inherited from the training data. This is a very limited setting, in particular for non-biological tasks where we dont want to retrain models for each new temperature setting.
5. As there is no notion of an energy function, the learned ode is not enforced to be conservative. Therefore, there are no guardrails against energy drifts, and the model sampling energetically inaccessible states.
6. No real-time comparisons or memory measurements are given. This makes it unclear how significant the time savings really are
7. No thermodynamical observables are calculated. For example, the radius of gyration and time correlation functions would be easy out-of-the-box observables that could be reported
8. The models are trained with 10% of MD17 frames, which amounts to tens to almost a hundred thousand training examples if the pytorch geometric datasets are being used. The dataset authors specify that not more than 1000 samples should be used to avoid data leakage: https://archive.materialscloud.org/records/pfffs-fff86
9. The only baseline model that is explicitly designed for the prediction of large time steps in molecular systems is ITO. However, ITO is a distribution-level model; it tries to predict distributions at a time in the future, not individual samples. Comparing MSE of individual samples is therefore not very meaningful and inflates ITO's error. A fair comparison would need to sample some initial distribution, push the ensemble of states forward in time with DoMiNO and the distribution with ITO, and then compare the models on statistical divergence metrics, not MSE.

### Questions
See above

### Soundness
1

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors introduce a framework for learning molecular dynamics trajectories. The core idea is to use multiple Neural ODEs that target different time step sizes. The initial state (atom positions) is encoded and solved by the coarsest Neural ODE (largest timesteps). The resulting latent states are then passed on to the Neural ODEs at finer levels (smaller timesteps). The latents at each level are combined using attention to output the final predictions. Each Neural ODE consists of EGNN layers. The authors evaluate on MD17 and MD22 molecules.

### Strengths
The authors report a significant reduction in MSE (although it is unclear to me of what, see weaknesses below)
Composing neural graph ODEs that specialize on different time step sizes is interesting and novel

### Weaknesses
1. Clarity of the experiments: 
- In table 1, what is the MSE of? Average in the difference in atom positions over a rollout of a certain length?
- The main motivation of the paper is that MD with MLIPs are slow and accumulate error. This claim is not properly verified or compared to their method. Inference time is not compared, and the error drift experiments lack a proper MLIP and do not mention important parameters like time step size
2. I am unsure how technically sound the idea of multiple Neural ODEs at different time step sizes is. Shouldn’t Neural ODEs be implicit representations of the continuous dynamics?
3. The approach requires training data from trajectories, while MLIPs can be trained with unordered samples from e.g. MC?

### Questions
1. Can you show experiments comparing the and drift error over the steps of you method vs MD with a SOTA MLIP like Mace/Equiformer/LeftNet? EGNN is a long outdated architecture from over four years ago
2. How does the inference time of your method compare to the baselines (or just an MLIP+MD)?
3. What time step size did you use for MD? This should strongly affect the drift error
4. How does the training time of the methods compare? Training the neural ODE should be significantly more expensive
5. Do the hierarchy levels improve results even if a single hierarchy model is matched in terms of training compute and memory (i.e. increasing the parameters)?
6. There is a broken citation of MD22 in line 713

### Soundness
1

### Presentation
1

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
This paper proposes DoMiNO, a hierarchical neural ODE framework for molecular dynamics (MD). The key idea is to decompose molecular motion into multiple temporal scales, each modeled by an E(n)-equivariant Graph ODE that captures dynamics from slow global motions down to fast local vibrations. These levels are fused via an attention mechanism to reconstruct molecular trajectories. The authors evaluate DoMiNO on standard MD datasets like MD17 and alanine dipeptide, showing large gains in both prediction accuracy and long-term stability compared to baselines like EGNN, EGNO, and ITO. Ablations highlight the importance of hierarchical decomposition and local time normalization.

### Strengths
-The paper clearly identifies and tackles a central challenge in MD — the multi-scale nature of atomic dynamics — and provides a coherent neural ODE-based solution.
- The hierarchical ODE formulation is conceptually elegant, with each level operating in its own local time scale. This is a nice balance between coarse- and fine-grained temporal modeling.
- The model is physically grounded, maintaining SE(3) symmetry through equivariant GNNs and decoding steps.
-The empirical results are strong and comprehensive, covering both small molecules and larger systems. The performance improvements (especially on benzene/toluene) are quite impressive.
-The authors provide detailed ablations and implementation details, which improves credibility and reproducibility.

### Weaknesses
-While the decomposition idea is solid, the connection to physical timescales (e.g., mapping levels to specific frequencies or normal modes) remains largely heuristic. There’s no explicit analysis linking learned scales to real physical processes.
- The paper is heavy on architectural details but light on intuition for why attention fusion is the right way to combine scales. It could use some visualization of learned weights or interpretability results beyond benzene/toluene.
- The evaluation, while broad, focuses mainly on predictive accuracy. There are no experiments showing practical utility for sampling, free-energy estimation, or integration with existing MD workflows.
- Computational cost is mentioned, but no direct runtime comparison versus baselines is shown.
- It’s unclear how well DoMiNO generalizes across molecular systems, or whether it needs retraining for every molecule type.

### Questions
1: How sensitive is the model to the choice of number of hierarchical levels or the relative time intervals between them?
2: Can the learned timescales be interpreted physically — e.g., do they correspond to specific vibrational or conformational modes?
3: Could the authors compare computational efficiency (e.g., wall-clock time) with neural ODE baselines or generative MD methods like ITO?
4: Since only a single initial frame is used, how robust is the model when initial configurations are noisy or sampled from different conditions?
5: Could this method be combined with energy conservation constraints or learned force fields to improve physical fidelity?

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
3

### Summary
The paper proposes a hierarchical multi-scale Neural Graph ODE framework for simulating molecular dynamics: an equivariant encoder projects atomic states to latents,  evolve dynamics in the coarse latent space via neural ODE and evolve at a local time-scale for finer level space, and then an attention fuser merges all levels to reconstruct coordinates. The design aims to resolve the small-timestep vs large-timestep dilemma by letting each level specialize in a characteristic timescale. Benchmark shows competitive MSE and slower error growth on datasets like MD17, alaine dipeptide.

### Strengths
* The paper is clearly written and easy to follow.

* The paper proposes to use an SE(3)-equivariant encoder/decoder with different spatio-temporal levels of graph neural ode to capture different scale of physics and achieve good computational effciency.

* On MD17/ALA2 and larger systems, DoMiNO shows slower error growth than baselines across extended trajectories, indicating better stability beyond short-term fit.

### Weaknesses
As MD dynamics are chaotic, matching a single deterministic trajectory quickly becomes ill-posed; long-horizon MSE on coordinates is therefore not a meaningful objective beyond short transients. What typically matters are ensemble/statistical properties (e.g., RDFs, energy drift, diffusion constants, autocorrelation times, free-energy landscapes) and long-term stability. An example of more practical evaluation is Fu et al. [1]—running sustained simulations and assessing thermodynamic/kinetic statistics with appropriate confidence intervals—rather than emphasizing single-trajectory reconstruction.

[1] Fu, Xiang, et al. "Forces are not enough: Benchmark and critical evaluation for machine learning force fields with molecular simulations."

### Questions
Can the method retain its accuracy and long-horizon stability on substantially larger, multi-molecule systems (e.g., solvated biomolecules with periodic boundaries)?

### Soundness
2

### Presentation
3

### Contribution
2
