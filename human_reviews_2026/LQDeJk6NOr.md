# BioMD: All-atom Generative Model for Biomolecular Dynamics Simulation

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 6, 4, 6

## Abstract
Molecular dynamics (MD) simulations are essential tools in computational chemistry and drug discovery, offering crucial insights into dynamic molecular behavior. However, their utility is significantly limited by substantial computational costs, which severely restrict accessible timescales for many biologically relevant processes. Despite the encouraging performance of existing machine learning (ML) methods, they struggle to generate extended biomolecular system trajectories, primarily due to the lack of MD datasets and the large computational demands of modeling long historical trajectories. Here, we introduce BioMD, the first all-atom generative model to simulate long-timescale protein-ligand dynamics using a hierarchical framework of forecasting and interpolation. We demonstrate the effectiveness and versatility of BioMD on the DD-13M (ligand unbinding) and MISATO datasets. For both datasets, BioMD generates highly realistic conformations, showing high physical plausibility and low reconstruction errors. Besides, BioMD successfully generates ligand unbinding paths for 97.1% of the protein-ligand systems within ten attempts, demonstrating its ability to explore critical unbinding pathways. Collectively, these results establish BioMD as a tool for simulating complex biomolecular processes, offering broad applicability for computational chemistry and drug discovery.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper develops a method BioMD, which generates time evolution of protein and ligand structures using a hierarchical architecture and flow matching. BioMD combined coarse grained time interval for structure ensemble generation conditioned on an initial frame, and fine grained interpolation conditioned on the starting and ending frames. It has demonstrated performance on MD datasets MISATO (unbiased protein-ligand simulations) and DD-13M (enhanced sampling MD simulations for ligand unbinding events), and shown advantage in the physicality, flexibility and unbinding success.

### Strengths
- The model architecture is novel, proposing a hierarchical framework in combination with flow matching
- The authors have tested multiple different ways of generation, including predicting relative conformational change v.s. absolute conformations, and predicting all frames v.s. auto-regressively, providing a thorough examination
- The BioMD method has shown advantages in the MISATO and DD-13M benchmarks, in the physicality, flexibility and unbinding success.

### Weaknesses
- In the introduction, the paper states: "The core bottleneck lies in the intensive calculation of non-bonded forces, particularly van der Waals and electrostatic interactions, which scale quadratically with the number of atoms." This is misleading, because with methods like Particle Mesh Ewald (PME) the scaling is approximately O(N log N), not O(N²). 
- The model is trained with a fixed time interval between frames (k=10 steps in coarse-graining). Can the model generalize to different time intervals, or is it restricted to this specific temporal resolution? If the latter, this should be explicitly stated as a limitation. 
- The DD-13M dataset is generated using enhanced sampling simulation, instead of unbiased MD. Therefore, BioMD is learning to reproduce biased trajectories, not true thermodynamic/kinetic unbinding behavior. The generated trajectories may not reflect realistic dissociation kinetics, and the model may be learning to mimic the "push" behavior by the biased potential in enhanced sampling. Currently, the paper does not clearly distinguish this subtlety.
- The success criterion (Section A.3.4) only requires that "at least one predicted ligand centroid position lies outside this convex hull." The success rate is thus measuring "can find a way out" rather than "finds the correct way out." 
- The reported MAE and MSE metrics average over all frames and trajectories. This can mask catastrophic failures in individual frames. The authors should consider providing more comprehensive failure analysis, including per-frame error distributions, percentage of frames with physical violations etc.
- The paper proposes the hierarchical architecture, but lacks the justification and ablation on the necessity of the hierarchy. Specifically, why the fine-grained generation is needed and how it will influence the performance should be demonstrated and discussed.

### Questions
- Section A.3.1 states: "we calculate the deviations of bond lengths and bond angles with respect to the initial frame of the reference trajectories." However, in Section 5, the "Static" baseline is defined as "the initial conformation of the system is held constant throughout the entire trajectory." If errors are computed relative to the initial frame, the Static model should have exactly zero error, yet Table 1 and Table 2 report non-zero errors for Static. The authors need to clarify: (1) what exactly is the reference for computing these errors, and (2) whether errors are computed for ligand only, protein only, or both.
- The phrase "sampling every k = 10 steps" (Section 4.2.1) is imprecise without specifying the timestep used in the underlying simulations. The actual time interval (in ps or fs) should be stated explicitly.

### Soundness
3

### Presentation
4

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
The paper proposes BioMD, a model for all-atom molecular dynamics prediction focusing on protein–ligand complexes. BioMD is based on flow matching conditioned on the protein sequence, ligand atom types, and the initial complex structure. By combining coarse-grained prediction for long time intervals with fine-grained interpolation for the short intervals in between, BioMD can model long molecular dynamics sequences. In the model architecture, the decoupling of intra-frame spatial attention and inter-frame temporal dependencies enables the simultaneous prediction of velocities across multiple time frames at all-atom resolution. To enhance physical plausibility, auxiliary losses are introduced. Experimental results demonstrate superior physical stability, conformational flexibility, and ligand unbinding success rates with BioMD.

### Strengths
1. The paper addresses an important problem: long-timescale molecular modeling of protein–ligand binding complexes at all-atom resolution, which has significant applications in drug discovery.
2. The efficient training and inference pipeline, along with the model architecture design, enables the prediction of multiple time frames simultaneously—crucial for modeling long MD trajectories. Experimental results show that the forecasting–interpolation workflow effectively reduces error accumulation.
3. The evaluation of the unbinding success rate demonstrates the model’s capability to capture real biomolecular processes and the authenticity of the predicted MD trajectories.
4. The paper is clearly written and well-presented.

### Weaknesses
1. Some implementation details are missing from both the main text and the appendix. For example, the architecture of the FlowTrajectoryTransformer is not fully described. In Algorithm 6, TemporalAttention and AttentionPairBias are used, but their definitions are missing, and only limited information about these modules is provided in the main text.
2. The evaluation focuses primarily on low-dimensional metrics, including bond/angle geometry and RMSF. The unbinding process is described using the centroid coordinates of the ligand. These metrics can reflect trajectory quality to some extent, but higher-dimensional evaluations or demonstrations would be more informative. For instance, analyzing distributional similarity between generated and ground-truth trajectories projected onto two TICA components, or examining the transition processes between clusters of conformers in the projected space, would provide deeper insight.

### Questions
1. In the formulation of the velocity model in Equation (2) (line 195), $\mathbf{Z}$ is said to contain the “amino acid sequence” and “ligand atom types.” Since the model is all-atom, how and when is atom-wise information from the residues incorporated into the model? From Figure 3, it seems that the atom-wise information of residues is only used in the velocity module after the $SE(3)$ Transformer backbone.
2. The paper focuses on protein–ligand complexes. However, the proposed techniques could, in principle, be applied to protein monomers and multimers as well. What is the motivation for focusing solely on protein–ligand complexes in this work?
3. How is the time interval in the coarse-grained forecasting stage chosen? How does the scale of this interval influence the quality and efficiency of trajectory generation?

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
2

### Summary
This paper introduces BioMD, a novel all-atom generative model designed to simulate long-timescale protein-ligand systems. The primary motivation is to overcome the significant computational cost of traditional Molecular Dynamics simulations, which severely limits the accessible timescales for many biologically relevant processes. The core contribution is a hierarchical framework that decomposes the complex task of long-trajectory generation into coarse-grained forecasting and fine-grained interpolation. The authors validate BioMD on two datasets. BioMD generates trajectories with high physical plausibility and successfully generates complete ligand unbinding paths.

### Strengths
1. The hierarchical  framework decomposes the long-sequence problem into "capturing the backbone first, then filling the details,"  effectively reduces the sequence length and the complexity of long-range dependencies in the generation process.
2. Using a single flow matching model to achieve two different tasks (forecasting and interpolation) simply by changing the masking schedules.
3. Supportive experimental results, especially the ligand unbinding experiment.
4. The paper is clearly written and presented.

### Weaknesses
1. **Definition of "Long-Timescale"**: The paper claims to simulate "long-timescale" dynamics. However, the MISATO dataset's simulation time (8 ns) is relatively short, insufficient to observe major protein conformational transitions (e.g., open-to-closed states). While DD-13M is a dissociation task, it was generated by metadynamics, an enhanced sampling method, not a spontaneous process in equilibrium. Therefore, the model's ability to simulate equilibrium rare events at the microsecond (µs) to millisecond (ms) level has not been demonstrated
2. **Insufficient Evaluation of Protein Conformational Changes**: The evaluation of protein flexibility relies mainly on RMSF, which is an aggregate metric. In drug discovery, we are often more concerned with specific conformational changes near the binding pocket, such as the flipping of key side-chains or the movement of "gating" residues. The paper lacks an in-depth case study on these critical biological events.
3. The model's performance on DD-13M is highly dependent on the quality and coverage of the training data. The DD-13M dataset was generated using metadynamics, which means BioMD may have learned a biased distribution of "how to dissociate quickly" rather than the dynamics of the true Boltzmann distribution. Could the author provide some study on the OOD cases using BioMD?
4. Lack of ablation studies, such as how the definition of "large-step" influences the final results and what is the effect of different sampling steps.

### Questions
- Was any data cleaning performed on the MISATO dataset? For instance, were unstable simulation results (according to the RMSD result) filtered out before training?

 - Did you evaluate the difference between the generated protein conformations and the reference MD simulations, particularly regarding key conformational changes in the binding pocket? 

- Have you attempted to extend the model to even longer timescales (beyond those in DD-13M)? Given the acknowledged error accumulation, do you observe simulation collapse or instability over extended periods? Furthermore, the MISATO simulations are relatively short (8 ns), making it difficult to observe significant events like major conformational transitions. Could you provide representative case studies to demonstrate the model's ability (or limitations) in capturing such rare but significant events?

- Regarding Figure 6, how consistent are the generated unbinding pathways with the reference metadynamics results?

- I recommend including related works that do not directly recover the dynamics trajectories but recover the key binding pose and also use flow matching or diffusion models. There are some related works recommended. For example, DynamicBind [1] learns to transform unbound to bound states of protein-ligand complexes; DynamicFlow [2] uses a flow model to transform protein apo state and a noisy ligand to a holo state and a binding ligand to consider protein dynamics in structure-based drug design.


**Note: I might raise my score according to the situation of rebuttal discussion and paper revision.**

References:

[1] DynamicBind: predicting ligand-specific protein-ligand complex structure with a deep equivariant generative model, Nature Machine Intelligence 2024

[2] Integrating Protein Dynamics into Structure-Based Drug Design via Full-Atom Stochastic Flows. ICLR 2025.

### Soundness
2

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
The paper introduces BioMD, an all-atom generative model for biomolecular dynamics simulation. It combines flow matching with a hierarchical forecasting–interpolation framework to generate long-timescale protein–ligand trajectories efficiently. The model is evaluated on MISATO and DD-13M, achieving high physical plausibility and impressive ligand-unbinding success rates.

### Strengths
* The work addresses a challenging and underexplored problem: generating full, time-resolved protein–ligand trajectories at all-atom resolution. The motivation is clearly articulated and relevant to molecular simulation and drug discovery.

* The proposed hierarchical framework is conceptually elegant. Separating long-range forecasting from local interpolation aligns well with physical intuition and effectively mitigates error accumulation over long trajectories.

* The unified conditional flow-matching formulation (“noising-as-masking”) is technically interesting and flexible, enabling both forecasting and interpolation within one architecture.

* The experiments are comprehensive within the chosen scope.

* The auxiliary geometric losses (bond, collision, and center penalties) are physically meaningful and help ensure chemically plausible outputs.

### Weaknesses
* The paper mainly reports geometric errors and RMSF correlations, but omits more direct physical checks (e.g., energy or temperature conservation, relaxation consistency). Without such validation, “physical plausibility” remains largely geometric.

* The experimental baselines are narrow. Although the authors mention Str2Str (Lu et al., 2024) and BioEmu (Lewis et al., 2025) in the related work, these strong generative baselines are not included in the experiments. The justification that they are “time-agnostic” is somewhat artificial, since BioMD also measures static geometric stability and RMSF correlations—metrics directly comparable to those ensemble models. Moreover, both Str2Str and BioEmu are flow- or diffusion-based architectures closely related to BioMD’s conditional flow-matching design. Their absence weakens the empirical comparison and makes it unclear whether BioMD truly surpasses existing conformational generators.

*  The architecture figure is dense but lacks quantitative configuration (model size, training schedule, runtime benchmarking). 

* It is unclear whether the hierarchical masking approach can generalize to unseen complexes or trajectories longer than those used in training. A discussion of scalability and failure cases would strengthen the work.

### Questions
See above

### Soundness
3

### Presentation
3

### Contribution
3
