# Beyond Ensembles: Simulating All-Atom Protein Dynamics in a Learned Latent Space

- Decision: Accept (Poster)
- Scores: 4, 4, 8, 2

## Abstract
Simulating the long-timescale dynamics of biomolecules is a central challenge in computational science. While enhanced sampling methods can accelerate these simulations, they rely on pre-defined collective variables that are often difficult to identify, restricting their ability to model complex switching mechanisms between metastable states. A recent generative model, LD-FPG, demonstrated that this problem could be bypassed by learning to sample the static equilibrium ensemble as all-atom deformations from a reference structure, establishing a powerful method for all-atom ensemble generation. However, while this approach successfully captures a system's probable conformations, it does not model the temporal evolution between them. We introduce the Graph Latent Dynamics Propagator (GLDP), a modular component for simulating dynamics within the learned latent space of LD-FPG. We then compare three classes of propagators: (i) score-guided Langevin dynamics, (ii) Koopman-based linear operators, and (iii) autoregressive neural networks. Within a unified encoder–propagator–decoder framework, we evaluate long-horizon stability, backbone and side-chain ensemble fidelity, and temporal kinetics via TICA. Benchmarks on systems ranging from small peptides to mixed-topology proteins and large GPCRs reveal that autoregressive neural networks deliver the most robust long rollouts and coherent physical timescales; score-guided Langevin best recovers side-chain thermodynamics when the score is well learned; and Koopman provides an interpretable, lightweight baseline that tends to damp fluctuations. These results clarify the trade-offs among propagators and offer practical guidance for latent-space simulators of all-atom protein dynamics.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper introduces Graph Latent Dynamics Propagator (GLDP), a framework for simulating protein dynamics in the learned latent space of LD-FPG. The approach uses a frozen encoder-decoder from LD-FPG and compares three propagator classes: (i) score-guided Langevin dynamics, (ii) Koopman-based linear operators, and (iii) autoregressive neural networks. The work evaluates these propagators on systems of increasing complexity (alanine dipeptide, 7JFL, A1AR, A2AR)

### Strengths
- The paper is well written, the concept and results are well presented
- The paper proposes novel method which uses LD-FPG for encoding and decoding, and performs simulation in the latent space
- Different methods of propagator strategies are compared
- The authors have performed benchmarks on different scales of systems

### Weaknesses
Generalization:
- The model is trained only on one time interval (frame stride), there's no evidence of generalizing to different time intervals
- All the models are trained and tested on the same system
- From Table 4,6,9, it seems that different systems use different hyperparameters. It's then questionable how to select proper hyperparameters when having a new system.

Unclear Advantage Over Equilibrium Models:
- All evaluations (RMSF, dihedrals, FES) measure equilibrium properties, not dynamics.
- No transition timescales, autocorrelation functions, or mean first passage times.
- The benchmarks and applications make the reviewer think that time-independent samplers such as AlphaFlow, BioEmu, might achieve same results without modeling dynamics.

### Questions
- Even training and testing on the same system, the model is trained on long simulation trajectory. But that does not make this model useful if it always needs a long simulation over a certain time scale to be able to forecast well. The authors should investigate how the model performance depends on the training trajectory length
- Regarding Langevin dynamics, as the authors mention that the score at t near 0 is used. Notice that the denominator is close to 0, then numerically there could be instability, which might explain the dynamics instability

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents Graph Latent Dynamics Propagator (GLDP), a modular framework for simulating all-atom protein dynamics from learned space from LP-FPG. It compares three latent-space propagators: (i) score-guided Langevin dynamics, (ii) linear Koopman operators, and (iii) a nonlinear autoregressive neural network. Latent propagators are evaluated across several systems, from alanine dipeptide to A2AR, showing the trade-offs of propagators.

### Strengths
1. Systematic comparison of latent propagator (quality)

The latent propagators are compared under a shared encoder, decoder, and latent space. 

2. Clearly written and organized (clarity)

### Weaknesses
1. Detailed comparison with baselines

The authors only report the JSD of dihedral and coordinates to the ground truth. As in Figure 2 of MDGen, it would be more convincing to also show each dihedral distribution against the baselines.

2. Validness of decoded molecules

I could not find any content on the validity of the decoded full atom resolution. For the 3th and 7th molecule in Figure 5 (when going from left to right and up to down), the structure seems a bit odd. Simply plotting the energy distribution would make the paper more convincing. Additionally, in minor, plotting more qualitative plots in Figure 5 for each latent propagator and molecule, would also be a good case to see whether the latent dynamics succeed.

Minor

- Line 316 - Full results on Pearson correlation is missing, a wrap table for it would be good
- Figure 3 - the ordering or propagators is different for the left one
- Figure 4 - four trajectories with the background downgrades visibility a bit, plotting four plots separately seems also good. Also, the visibility of  inactive and active site are hindered.

### Questions
1. Residual prediction with an autoregressive neural network

Just a suggestion, perhaps learning the $f_\theta(z_t)$ to approximate $z_{t+1} - z_{t}$ would improve the performance even more.

2. Long horizon stability (section 4.2)

I am a little confused about the conclusion for section 4.2. I understand the task, but since molecular dynamics trajectories contain randomness from the Brownian motion, maybe a propagator that did understand the molecular dynamics could result in sending the structure totally different from the ground truth data? Does 1DDT threshold include distinct local energy minima?

### Soundness
1

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
3

### Summary
The paper proposes the Graph Latent Dynamic Propagator (GLDP), an approach for modeling molecular dynamics in the latent space of an encoder–decoder framework (LD-FPG). The encoder and decoder are kept fixed, while latent-space propagators are trained on the temporal sequences of latent representations. Three types of propagators are proposed: score-based Langevin dynamics, Koopman-based linear operators, and autoregressive neural networks.

To evaluate the flexibility and distributional fidelity of the dynamics generated by GLDP, several metrics are computed—such as the Jensen–Shannon Divergence (JSD) with respect to the ground-truth ensemble and the average RMSF across the sequence—and compared with baseline approaches including LSS, GeoTDM, and MD-Gen. The results indicate superior performance in recovering the ground-truth distribution and achieving flexibility values closer to the reference.

In the long-horizon modeling scenario, the three propagators are compared, with the autoregressive neural network demonstrating the greatest stability. Free-energy surfaces (FES) are computed in the space of two variables to measure fidelity to the equilibrium ensemble. Finally, GLDP is shown to successfully reproduce the inactive-to-active transition of $\mathrm{A}_{2A}$R, where the score-guided Langevin dynamics covers most of the FES valley as well as the corridor connecting the inactive and active regions.

### Strengths
1. The paper conducts a thorough evaluation across multiple dimensions, including quantitative metrics for stability, flexibility, and distributional fidelity. It also verifies that GLDP recovers the two metastable states of $\mathrm{A}_{2A}$R (active and inactive), demonstrating consistency with real biological processes.
2. The proposed method is encoder/decoder agnostic, as the encoder and decoder remain frozen. This design makes the framework easily adaptable to different latent spaces.
3. The paper is clearly written and well presented.

### Weaknesses
Overall, this is a solid paper with well-designed experiments and sound conclusions. However, it could be further improved in the following aspects:
1. The evaluation is conducted on only three proteins. Although these systems cover increasing complexity from ADP to A1AR GPCR, experiments on additional systems would strengthen the conclusions regarding the relative performance of the propagators.
2. It would be interesting to examine whether other baselines can also recover the active–inactive transition of $\mathrm{A}_{2A}$R.
3. In addition to performance metrics, it would be valuable to include efficiency comparisons between different propagators, which are particularly important for long-horizon molecular dynamics.
4. The necessity or advantage of modeling dynamics in latent space, rather than Cartesian space, is not clearly articulated in the paper.

### Questions
1. There are space formatting issues in lines 260, 264, and 265.
2. There are also space formatting issues in lines 60 and 61.
3. In Figure 4, the corridor and the regions representing inactive and active states are not clearly visible.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper investigates generating protein trajectories in the latent space of pretrained encoder-decoder models, rather than directly in Cartesian coordinates. The authors present GLDP, a plug-in module for the graph-based conformation generation model LD-FPG, enabling trajectory propagation in latent conformational space, followed by decoding to atomic coordinates. 
Within this framework, the paper systematically compares three latent-space propagation strategies: Score-based Langevin dynamics (similar to Two-for-One, https://arxiv.org/abs/2302.00600); Koopman operator-based linear propagation, and Neural network-based nonlinear propagation, for autoregressive generation.
These approaches are evaluated on three protein systems of varying sizes. The neural autoregressive propagator is found to be the most stable and best at capturing ensemble-level statistics, while Langevin dynamics can perform better in recovering in side-chain torsional distributions.
Overall, the work offers an interesting perspective on biomolecular dynamics by exploring trajectory generation in latent space and systematically comparing reasonable propagation strategies.

### Strengths
Exploring protein dynamics in latent space as a potential way to accelerate MD simulations is an interesting direction, and this work provides a controlled comparison of three propagation strategies.
This study evaluates methods across protein systems of different sizes, offering some insight into applicability across system sizes

### Weaknesses
1. The idea of modeling dynamics in latent space is interesting, but the overall architecture (e.g., LD-FPG encoder-decoder) feels dated, and the evaluation is limited to three systems in non-transferable settings.

2. Some evaluation choices are not fully convincing, and it is unclear whether certain results are statistically significant or lead to conclusive insights on this latent dynamic problem.

3. The paper would benefit from stronger organization, clearer presentation of results, and inclusion of key experimental details that are currently missing. 

See Questions for details.

### Questions
[Model]

1. What data are used to train the LD-FPG encoder–decoder? Is single encoder/decoder modules shared model used for different proteins, or separately for each protein system?
2. Why choose LD-FPG (ChebNet + MLP) instead of more modern transformer-based architectures (e.g., as in AlphaFold3, https://www.nature.com/articles/s41586-024-07487-w)? This design also requires frame alignment and offset prediction, which limits transferability.
3. The use of "pooling’" and "decoder" commonly appear together (e.g., in Table 3). Is my understanding right that pooling happens after encoding and before propagation, and is not part of the decoder?
4. For score-based Langevin dynamics, score estimation near $t\approx 0$ is known to be unstable due to very low noise level in the denominator. Is this a problem in practice?
5. For baseline models (e.g., MD Gen), were their pretrained weights used, or were all models retrained for each system?

[Data]

6. How are the trajectory splits defined for training, validation, and testing? Has any time-based or conformation-based split applied? Are models always trained and evaluated on the same protein system?
7. 7JFL_C is a small and fully helical protein (47 residues). Have larger or systems with other secondary structures (e.g., from ATLAS) been tested?

[Results]

8. Is lDDT alone sufficient to assess long-horizon physical stability, given it does not capture energetics or steric quality? The choice of failure threshold (lDDT < 0.65) also appears arbitrary - how was it determined?
9. Figure 2a shows stable rollouts beyond 10,000 steps for the autoregressive model, but line 357 states failure at 3,176 steps. 
10. The claim that Langevin fails earlier on alanine due to step sizes tuned for GPCRs seems inconsistent with the earlier statement that step sizes are tuned per system. Can you clarify?
11. In Table 2 (A1AR side-chain torsions), results are shown for only one protein and from single test, and differences between AR and Langevin are small. Are these statistically significant?
12. For the A2AR case study, how does this system differ from A1AR, and what are the main takeaways?
13. How does the runtime of GLDP compare to classical MD simulations? It seems that this method requires system-specific MD trajectories exist for training before it can generate new trajectories - would it become a problem in practical use?

[Other comments]

The related work section is somewhat long and loosely organized. GPCR datasets appear alongside general method discussions, while other relevant datasets are not covered. This section could be better organized by theme and shortened.

### Soundness
2

### Presentation
2

### Contribution
2
