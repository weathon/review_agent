# A Two-Phase Deep Learning Framework for Adaptive Time-Stepping in High-Speed Flow Modeling

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 4, 8, 2

## Abstract
We consider the problem of modeling high-speed flows using machine learning methods. While most prior studies focus on low-speed fluid flows in which uniform time-stepping is practical, flows approaching and exceeding the speed of sound exhibit sudden changes such as shock waves. In such cases, it is essential to use adaptive time-stepping methods to allow a temporal resolution sufficient to resolve these phenomena while simultaneously balancing computational costs. Here, we propose a two-phase machine learning method, known as ShockCast, to model high-speed flows with adaptive time-stepping. In the first phase, we propose to employ a machine learning model to predict the timestep size. In the second phase, the predicted timestep is used as an input along with the current fluid fields to advance the system state by the predicted timestep. We explore several physically-motivated components for timestep prediction and introduce timestep conditioning strategies inspired by neural ODE and Mixture of Experts. We evaluate our methods by generating three supersonic flow datasets, available at https://huggingface.co/divelab. Our code is publicly available as part of the AIRS library (https://github.com/divelab/AIRS).

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes using an ML model to predict time steps for fluid simulations that use adaptive time steps.  It then uses those timesteps in an autoregressive fashion to solve compressible flow problems with a neural solver.

### Strengths
- The paper makes an interesting observation that when using a neural PDE solver, you can't necessarily rely on the CFL number computed on the fine computational mesh used for the simulation, because the neural model will effectively downsample the mesh inside the network, and so the adaptive time step chosen should be on that coarser scale.
- The numerical experiments seem convincing to me of the utility of the proposed method.

### Weaknesses
- I think "Our work represents the first steps towards developing machine learning models for high-speed flows" is a bit of an inflated statement that could be toned down.
- Although it's a fair point that a neural solver will have a different internal grid resolution than what is used for the PDE outside the network, presumably one knows what is the architecture they're dealing with and what that coarse grid resolution is.  Why not just compute an adaptive time step - classically - using that coarser scale (which should be known from looking at the network architecture you choose) and eschewing the proposed neural CFL predictor?
- If the answer to the above is "you could do either, but neural CFL predictor is faster while still just as accurate," can we see a study on that?

### Questions
- It's probably worth making the connection in the into that adaptive time stepping is not some CFD-specific thing - it is core to ML as well, e.g. "adaptive learning rates" with SGD/Adam/etc.
- Probably also worth noting in the intro that the problem with adaptive time stepping is more extreme than what you describe - if you don't have good timesteps, not only will you miss fine-scale flow phenomena, but your simulation will just crash due to predicting negative pressures etc. - before the solution ever has a chance to diverge - see e.g. Patkar et al., "Towards positivity preservation for monolithic two-way solid–fluid coupling" (in particular the issue of positivity preservation that is highlighted - even for relatively low-Mach flows)
- In 2.3, since you are interested in CFD/engineering, it may be worth noting that methods like PINNs are not convergent (they are, in their vanilla versions at least, one-shot predictors of the solution of a PDE, which can't be convergent).  Recent hybrid solver approaches like that of Kaneda et al., "A deep conjugate direction method for iteratively solving linear systems" use a neural network for preconditioning but use a classical solver loop to ensure convergence (that paper happens to be for incompressible flow, but same ideas for compressible flow systems).

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper proposes ShockCast, a two-phase deep learning framework for modeling high-speed flows with adaptive time-stepping.

- Phase one uses a “Neural CFL” model to predict the timestep ∆t;

- Phase two employs a timestep-conditioned neural solver to evolve the flow field.

The authors evaluate their framework on two internally generated supersonic datasets (circular blast and coal dust explosion), comparing multiple backbone architectures and conditioning strategies.

### Strengths
- Clear motivation: aiming to addresses PDEs under varying time intervals.

- Clear architecture modularity: Two phases modeling.

- Inclusion of physically inspired features (∇u, wave speed, CFL terms) is a positive step toward physics-awareness.

- Comprehensive validations across several backbones and conditioning methods.

- Figures are clear and the manuscript is overall readable.

### Weaknesses
- Conceptual shallowness: The so-called “Neural CFL” model merely regresses ∆t from local features; it does not derive from or guarantee compliance with the true CFL stability condition. There is no theoretical guarantee that predicted timesteps are stable or physically valid.

- Lack of physical consistency: The paper never checks conservation of mass, momentum, or energy, which is essential for high-Mach or compressible flows.

- Experimental validation is weak – Both datasets are self-generated 2D toy problems. Under the same physical conditions, which configuration for each benchmark is the most effective?

- No efficiency or stability analysis – The core motivation of adaptive stepping (e.g., computational savings or others) is never quantified; no wall-clock or rollout stability plots of flow evolution.

- Limited generalization – Only Mach < 3 cases are tested; unclear if the method scales to realistic hypersonic or turbulent regimes.

- Over-claimed novelty – Prior works (e.g., continuous-time neural solvers, time-conditioned FNOs, and physics-aware operator networks) already include similar temporal adaptivity ideas; Also many works for supersonic/hypersonic flow modeling (not the first machine learning framework).

- Method complexity vs. benefit – The two-phase design and multiple conditioning mechanisms add heavy machinery without showing clear improvement over a simpler time-conditioned baseline. Reporting a clear and full coparison (vs. graph-based methods, transformer-based methods, FNO-based methods, and etc. ) on one new table.

### Questions
- If the ∆t distribution is known in training data, why we need the two-phases modeling?

- If the two-phases modeling is usefull, the predicted ∆t from Phase one is inherently impossible for it to perfectly match the true ∆t  (based on a continuous value space) . In this case, what is the significance of designing the Phase one?

- Can you demonstrate that ShockCast preserves key physical invariants or avoids divergence during long rollouts?

- How much computational speedup or advantage (vs. one-phase modeling with time variable inputs) is achieved in practice?

- Have you tested the model on higher Mach (>5) or 3D flow cases to assess scalability?

- Have you tested the model on irregular domain cases to assess scalability?

Should you be able to satisfactorily address the points I've raised above, I will accordingly provide a positive rating.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper, ShockCast, proposes a novel two-phase deep learning framework for adaptive time-stepping in high-speed flow modeling (e.g., supersonic and hypersonic regimes). High-speed flows exhibit transient sharp gradients (shocks) requiring dynamic adjustment of the timestep size (Δt) using the Courant-Friedrichs-Lewy (CFL) condition, which is computationally expensive for classical solvers.

ShockCast addresses this by decomposing the task:

Neural CFL Phase ($\psi$): A ConvNeXt backbone predicts the optimal, large timestep $\Delta t$ based on the current flow state. This module is trained to emulate the $\Delta t$ choices from the classical solver used for data generation, circumventing issues caused by coarse computational meshes.

Neural Solver Phase ($\phi$): The flow state is evolved by the predicted $\hat{\Delta}t$. The authors introduce three novel timestep conditioning strategies for various neural solver backbones (F-FNO, U-Net, CNO, Transolver): Euler Residuals, Mixture of Experts (MoE), and Affine/Spatial-Spectral Conditioning.

The framework is evaluated on two new supersonic flow datasets—Coal Dust Explosion (multiphase) and Circular Blast (single-phase)—and achieves strong performance in accurately predicting both instantaneous fields and integrated physical quantities (TKE and Mean Flow).

### Strengths
Improved Training Objective for Transient Dynamics: The core motivation is strong: adaptive time-stepping naturally balances the training objective by inversely scaling $\Delta t$ according to the rate of change. This more evenly distributes the learning difficulty across states with smooth and sharp gradients (i.e., shocks), a highly pertinent consideration for high-speed flows.


Physically-Informed Neural CFL Model: The Neural CFL phase successfully emulates the true adaptive time mesh, as shown by the close match between predicted and true $\Delta t$ during autoregressive rollout. Furthermore, incorporating physically-motivated inputs like spatial gradients ($\nabla u$) and CFL features substantially improves the $\Delta t$ prediction accuracy for the complex multiphase Coal Dust Explosion scenario


Novel and Effective Conditioning Strategies: The introduction of Euler Residuals and Mixture of Experts (MoE) as timestep conditioning strategies is technically insightful. The results demonstrate that these methods are highly competitive, achieving the lowest TKE error for the Circular Blast (F-FNO backbone with MoE/Euler) and best Mean Flow/TKE performance for the Coal Dust Explosion (U-Net backbone with MoE/Euler)

### Weaknesses
1. Missing Quantification of Speedup: The paper's primary motivation is the immense computational cost of classical high-speed flow solvers. However, the results section fails to quantify the final speedup achieved by the full ShockCast pipeline (inference runtime) relative to the original classical solver. Without this figure, the practical utility of the entire framework remains unproven. (Only the classical solver runtime is given in Table 5, min: $\sim 15$K seconds, mean: $\sim 67$K seconds)

2. Complexity of MoE Implementation: The MoE conditioning significantly increases peak training memory (e.g., F-FNO: $18.9$ GiB (Affine) vs. $37.2$ GiB (MoE); Transolver: $41.8$ GiB (Affine) vs. $62.4$ GiB (MoE))9. While the complexity is offset by reducing the latent dimension for some models, the substantial jump in memory requirement suggests that the MoE approach is challenging to implement and scale, requiring clarification on the trade-off.

### Questions
Missing Speedup Quantification: The core justification is computational efficiency, yet the paper fails to state the final speedup factor (e.g., $1000\times$) of ShockCast relative to the original classical solver runtime (which can take tens of thousands of seconds). This critical number must be explicitly provided


rade-Off for MoE Complexity: The Mixture of Experts (MoE) component, while showing excellent results, dramatically increases memory consumption (e.g., up to $\sim 62$ GiB)14. The efficiency/performance trade-off needs deeper analysis, as the simpler Euler Residuals or Affine methods often perform comparably

### Soundness
4

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
3

### Summary
The paper proposes ShockCast, a two-phase deep learning framework for adaptive time-stepping in high-speed flow modeling. The first phase employs a Neural CFL model to predict the time-step size ($\Delta t$) based on the current flow state, while the second phase uses a time-conditioned neural solver to evolve the flow field by the predicted $\Delta t$. The authors generate two new datasets (spherical blast and coal dust explosion) and demonstrate that ShockCast effectively handles the sharp gradients and varying time scales in supersonic/hypersonic flows.

### Strengths
First work to address adaptive time-stepping in neural solvers for high-speed flows, filling a critical gap in ML-based CFD. Potential to accelerate simulations in aerodynamics, aerospace, and explosion modeling.

### Weaknesses
1.Evaluated only on two synthetic datasets (spherical blast & coal dust explosion). Training neural CFL + solver may still be expensive compared to classical adaptive methods.

2.Be overly dependent on data.

3.The proposed component is not sufficiently validated.

### Questions
1.How were the initial conditions (e.g., Mach numbers, pressure ratios) for the datasets chosen? Are they representative of real-world scenarios?

2.Can you visualize/analyze which features (e.g., velocity gradients, sound speed) most influence the predicted $\Delta t$?

3.Small errors in $\Delta t$ prediction may compound during autoregressive rollout. How does ShockCast handle stability over long simulations?

4.Does ShockCast generalize to unstructured meshes or three-dimensional (3D) flows?

5.The paper lacks an analysis of error accumulation beyond 100 autoregressive steps.

6.The requirement for high-fidelity solver data for training may be prohibitive for certain users.

7.Please specify the computational costs of the proposed methods and the actual speedup achieved compared to numerical simulations.

### Soundness
2

### Presentation
3

### Contribution
2
