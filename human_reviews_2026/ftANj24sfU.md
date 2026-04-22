# Large-Scale Molecular Dynamics Simulation: Direct Interatomic Modeling with Dilated Message Passing

- Avg Score: 5.00
- Decision: Reject
- Scores: 0, 8, 6, 6

## Abstract
Large-scale molecular dynamics simulation is essential in understanding chemical and biological processes, necessitating the accurate and efficient modeling of interatomic interactions. Existing learning-based methods generally are based on message passing mechanisms; they either are not scalable or are too coarse to offer accurate modeling. We propose a new message passing framework that can effectively and efficiently model interatomic interactions for simulating large-scale molecular dynamics at full atomic resolution. Specifically, our framework is stacked with a sequence of message passing neural network layers, each realizing the message passing over a distinct and dilated star-structured path. These star-structured paths are constructed progressively along dilated regions to capture the distance-dependent interactions. The crux of our framework is that it resolves the problem of dense interatomic interactions of large-scale atomic systems with sparser and region-based message passing graphs. We evaluate the framework on four benchmarks: the MD22 (molecules with 42–370 atoms), the Chignolin (a 166-atom protein featuring diverse conformations), the AdK dataset (a protein trajectory with up to 3,000 atoms), and the MISATO dataset (over 10,000 heterogeneous protein-ligand complexes, including systems with up to 40,000 atoms). Comprehensive evaluations demonstrate that our approach delivers state-of-the-art performance overall across various benchmarks. In particular, it is the first learning-based method to achieve atomic-level accuracy in protein-ligand dynamics simulation while preserving computational efficiency.

## Human Reviews

## Human Reviewer 1

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
The study shows a dilated operation in message passing and benchmarked it on several datasets for energy and force predictions. Overall speaking, the design lacks novelty; the experiments cannot fully support the conclusion; some results are suspicious; many professional terms are mistanken used; and the paper is poor written. It's far from the standard of ILCR.

### Strengths
1. reasonable overiview and introduction on the background and latest progress in this field
2. detailed hyperparametes are shown in supplementary materials.

### Weaknesses
1. Lack of novelty
The dilated operation presented in this study is too simple and lacks novelty. Almost operations and equations in the model design are from ViSNet paper. The dilation on radius is just a tiny trick insteaf of a novel design as the dilation employed in CNN or other models has been demonstrated more than 10 years before.

2. Suspicous results
Table 1. There is no evaluation on MISATO dataset in the paper (Wang et al., 2024a). Where the results in "AIMD" and "CHARMM27" rows come from? 
Table 2. The authors claimed all the other models suffered from OOM issue. As PaiNN and EGNN are smaller than DKMP, why they all failed?
Table 1 and Table 3. Why the compared models are inconsistent?
Table 1 and Table 3. Why neglect the SoTA models, e.g., MACE-OFF, SO3Krate, Equiformer v2 for comparison?
Table 3 and Figure 4. Why choose Equiformer rather than Equiformer v2 for comparison? Even for v2, it has been published for several years.
Figure 5. It seems that all the other models except DKMP show abnormal energy flunctuations in NVE simulations. First, it does NOT show DKMP's supriority. Instead, it means the MD settings for other models are probably WRONG! Second, even DKMP's simulation show stable energy values. It has nothing to do with the statement "throughout long-term MD simulations, more accurately capturing
the physics of real MD simulations for large molecular systems"! 100ps simulation is too short. Where's the "accurate" come from? where the "physics" come from? What does "real MD" mean? Chignolin has only 166 atoms. It has nothing to do with "large molecular system"!
The author lack basic domain knowledge on MD simulations and the experimental results are highly suspicous!

3. Cofunsed terminology
Painn, MACE, Allegro, etc...  are NOT MD simulation methods. Instead, they are machine learning force fields. 
The unit of force is kcal/ (mol*Angstrom) instead of kcal/mol/Angstrom. 
As far as I know, Chignolin dataset has 2 million samples, insteaf of 10 thousand. https://figshare.com/articles/dataset/_strong_AIMD-Chig_exploring_the_conformational_space_of_166-atom_protein_strong_em_strong_Chignolin_strong_em_strong_with_strong_em_strong_ab_initio_strong_em_strong_molecular_dynamics_strong_/22786730
The confused and misused terminology show the authors lack adequate knowledge in this field.

4. Poor written
The demonstration on model design is not clear. Almost operations in the model are directly from ViSNet. Too many confused terminology. And more grammer errors..

### Questions
The authors should seriously address the concerns shown in Weakness point by point to improve the quality of the paper.

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes Dilated K-Star Message Passing (DKMP*) for large-scale MD at full-atom resolution. The key idea is to partition edges by distance into mutually exclusive, increasingly “dilated” K-star neighborhoods and stack one-layer MPNNs over these partitions. Two concrete variants are presented: (1) DKMPC with dilated radius-cutoff intervals for single-scale systems (MD22, Chignolin); and (2) DKMPR with dilated distance-ranking + a light graph-attention core (dropping strict equivariance) for various-scale protein–ligand systems (MISATO, AdK). Results claim SOTA accuracy/efficiency, including atomic-level next-step precision on protein–ligand systems up to ~40k atoms, and improved NVE stability on Chignolin.

### Strengths
1. Broad evaluation: S2EF on Chignolin/MD22 and S2S on MISATO/AdK; useful mix of small and very large systems
2. Scalability: MISATO results include cases with 11k–40k atoms; baselines reportedly OOM while DKMPR runs in ≤~18 s/snapshot for the largest complex

### Weaknesses
1. Proposition 3.1 claims that DKMP* interactions are “immune to over-squashing.” While rewiring can mitigate message-passing bottlenecks, describing the model as “immune” is overstated unless bounded influence distortion is formally demonstrated—e.g., via curvature or flow-based analyses rather than adjacency-power arguments. I recommend tempering this claim or expanding the proof to include a modern formalism such as influence decay bounds, effective resistance, or discrete Ricci curvature, to substantiate robustness beyond intuitive edge-connectivity reasoning.
2. Additional ablations that could strengthen the empirical analysis include: (1) Edge-budget control: compare one-shot KNN with K = M versus L dilated K-stars summing to M under identical total edge counts; (2) Partition strategy: random versus distance-sorted partitioning to demonstrate the impact of ordering; (3) Hyperparameter sensitivity: vary K and L to map accuracy–latency trade-offs and reveal potential non-trivial optima; (4) Mutual-exclusion analysis: with versus without edge reuse across layers to verify its necessity; and (5) Equivariance ablation: equivariant versus non-equivariant variants on MISATO with efficiency–accuracy curve
3. The paper occasionally uses inconsistent notation (e.g., $N_{K}^{l}(i)$ vs. $N_{C}^{l}(i)$), which can obscure the hierarchy and semantics of neighborhood definitions. A unified notation scheme, accompanied by a concise schematic diagram (with notation and equations) illustrating the structure and information flow of DKMP layers, would substantially improve clarity and readability.
4. While energy conservation is qualitatively demonstrated (Fig. 5), the paper lacks quantitative physical validations such as energy drift rates or RMSD stability analyses. Including velocity autocorrelation function (VACF) plots would also provide valuable insight into the system’s dynamical fidelity.

### Questions
1. What is the time complexity as a function of (N, K, L, M) for both variants and contrast to dense-cutoff MPNN?
2. How does your framework handle periodic boundary conditions (PBC)? If applicable, could you demonstrate its validity on at least one standard PBC benchmark?  
3. A common and informative analysis is to examine potential decay—could you plot the learned potential to verify that it exhibits the correct decaying behavior with increasing interatomic distance?
4. How does this sampling strategy ensure smooth transitions of energy and forces between consecutive MD frames? In other words, how does it avoid discontinuities caused by differences where one neighbor jumps from one edge set to a different edge set?
5. Could you clarify how the model scales to extremely large systems (e.g., tens of thousands of atoms)? The main text currently lacks sufficient technical details on the specific design or computational strategies—such as memory partitioning, neighbor sampling, or distributed message-passing—that enable efficient handling of such large systems.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces DKMP$^\*$, a novel message passing framework for large-scale molecular dynamics (MD) simulations at full atomic resolution. It addresses the failure of current methods, which either suffer from over-squashing when stacked deep or prohibitive computational cost when the interaction radius is increased. DKMP$^\*$ resolves this by stacking layers that each pass messages over distinct, sparse, "dilated" graphs, progressively capturing interactions at different distances.

The paper presents two implementations:

* DKMP$^C$ (Dilating Radius Cutoff Interval): For single-scale systems.

* DKMP$^R$ (Dilating Distance Ranking): For various-scale systems.

This approach is the first to achieve atomic-level accuracy on large-scale benchmarks like MISATO, successfully simulating protein-ligand systems with up to 40,000 atoms where all baselines failed .

### Strengths
**Significance & Originality**: This work is a significant breakthrough for large-scale MD. The core idea of using dilated, sparse message passing graphs (instead of one dense one) is a novel and effective solution to the scaling and over-squashing problems .

**Performance**: The method achieves state-of-the-art results on four benchmarks. Most impressively, it is the only ML-based method shown to successfully run and maintain atomic-level accuracy on the largest, most complex systems in the MISATO dataset (up to 40,000 atoms), where all baselines failed due to out-of-memory (OOM) errors .

**Quality & Clarity**: The paper is well-written, clearly motivating the problem (Figure 1) and solution. The experimental validation is strong, particularly the parameter analysis (Figure 6) which empirically proves its hypothesis: DKMP$^\*$ benefits from deeper layers while baselines suffer from over-squashing .

### Weaknesses
I don't perceive any major weaknesses in this paper, though I feel it lacks an experimental validation. While the author finds the DKMP$^C$ (radius cutoff) implementation challenging to learn, I believe empirical evidence is still needed to substantiate this conjecture. Since EGNN inherently handles node/edge variations, it should be capable of learning.

### Questions
**Impact of Omitting Equivariance**: Could you quantify the impact of omitting equivariance in the DKMP$^R$ model? How does an equivariant version perform on MISATO? Does it also fail with OOM errors like the baselines?

**Long-Term Stability**: Why does the model's error (F-MSE) grow so large over long trajectories, even when its single-step prediction (N-MSE) is excellent? Does your spatial dilation approach have any blind spots for long-term temporal stability?

**Implementation Crossover**: Have you experimentally confirmed that the DKMP$^C$ (radius cutoff) implementation is computationally inefficient on the various-scale MISATO dataset, as you hypothesize?

### Soundness
3

### Presentation
3

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
This paper introduces a new framework, Dilated K-star Message-Passing (DKMP*), for simulating large-scale molecular dynamics (MD). The core problem addressed is that standard message passing neural networks (MPNNs) struggle to scale to large atomic systems due to computational cost (with large cutoffs) or information propagation issues like over-squashing (with many layers). The proposed solution is to stack a sequence of shallow MPNNs for each subgraphs
 . These graphs are constructed by partitioning the set of all atomic pairs based on their Euclidean distance, effectively creating a "dilated" receptive field that grows with each layer. This allows the model to directly capture interactions at various distances without incurring the computational cost of a dense graph or the propagation issues of deep GNNs.

The authors propose two concrete implementations:

DKMPC (Dilating Radius Cutoff Interval): For single-scale systems, where edges are partitioned into concentric spherical shells.
DKMPR (Dilating Distance Ranking): For various-scale systems, where a fixed number of neighbors are selected from progressively distant rank-ordered sets. This version also uses a non-equivariant attention mechanism for efficiency.
The framework is evaluated on several benchmarks, including MD22, Chignolin, AdK, and the large-scale MISATO dataset (up to 40,000 atoms). The results demonstrate state-of-the-art performance in both structure-to-energy-and-forces (S2EF) and structure-to-structure (S2S) tasks, significantly outperforming baselines on large systems where many other methods fail due to memory constraints.

### Strengths
1. Significance: The paper tackles a highly significant and challenging problem in computational science: creating accurate and efficient machine learning potentials for large-scale molecular systems. The ability to simulate protein-ligand complexes with tens of thousands of atoms at full atomic resolution, as demonstrated on the MISATO dataset, is a major step forward and has substantial implications for fields like drug discovery.

2. Quality: The experimental evaluation is extensive and convincing. The method is benchmarked on a diverse set of four datasets, covering different system sizes, tasks (S2EF, S2S), and molecular types (small molecules, proteins, protein-ligand complexes).
The results are impressive, showing state-of-the-art accuracy while maintaining or improving computational efficiency. 

3. Clarity: The paper is well-written, and the core idea is presented clearly. Figure 1 provides a straight-forward visual intuition for the problem and the proposed solution. The formalization of the dilation mechanism via the four constraints in Eq. 2 is a clear and effective way to define the framework.

### Weaknesses
Novelty in a Broader Context: While the application and specific formulation are novel, the underlying idea can be viewed as an architectural variation of existing principles. The method is essentially a sequence of MPNN blocks, each operating on a pre-determined, rewired graph. This connects it closely to the graph rewiring literature (e.g., Gutteridge et al., 2023), which also seeks to improve long-range information flow. The contribution could be framed more as a highly effective, structured rewiring strategy tailored for molecular physics, rather than an entirely new paradigm. The novelty is more in the successful engineering and application than in a fundamental algorithmic breakthrough.

Lack of Equivariance in DKMPR: The paper states that for the DKMPR model—the one used for the largest and most challenging systems—equivariance constraints are omitted for efficiency, inspired by AlphaFold3. This is a significant design choice that merits a more thorough justification and analysis. Equivariance is a fundamental inductive bias for physics simulation, ensuring that predictions transform correctly under rotations and translations. Dropping it risks the model's physical consistency and generalization. While the empirical results are strong, it is unclear if the model is learning approximate equivariance from the large dataset or if its success is limited to the data distribution seen during training. An ablation study quantifying the impact of this choice would greatly strengthen the paper.

Handling of Long-Range Interactions: The dilation mechanism is a heuristic for capturing interactions at increasing distances. It is well-motivated for interactions that decay with distance, like van der Waals forces. However, it is less clear how this approach compares to principled methods for handling long-range electrostatic interactions, which decay slowly (1/r) and are critical in many biomolecular systems. The paper mentions Ewald-based methods (Kosmala et al., 2023) as orthogonal but does not discuss the limitations of its own approach in this context. A deeper discussion or an experiment on a system dominated by electrostatics would be insightful.

Missing baselines and related works:
1. SE(3) Equivariant Graph Neural Networks with Complete Local Frames;  ICML 2022;
2. AlphaNet: Scaling Up Local Frame-based Atomistic Foundation Model, Npj Comput. Mater. (2025)

### Questions
Novelty and Graph Rewiring: Could you further elaborate on the relationship between DKMP* and graph rewiring methods like DRew? Both seem to address over-squashing by modifying graph connectivity to facilitate long-range information flow. Is it fair to characterize DKMP* as a deterministic, multi-stage rewiring strategy where the graph is rewired at each stage according to a distance-based partitioning?

Justification for Dropping Equivariance: Regarding the non-equivariant DKMPR model: could you provide an ablation study or further analysis on the effect of removing the equivariance constraint? For example, how does a non-equivariant model perform if the test set molecules are rotated randomly compared to their training orientation? Does the model implicitly learn this symmetry from the data, and if so, what is the data requirement for this to occur?

Choice of Hyperparameters L and C: In your parameter analysis, DKMPC's performance improves with the number of layers L. How should one choose the optimal L and maximum cutoff C? Is there a trade-off where increasing L too much creates overly sparse graphs in each message-passing step, potentially harming the learning of collective interactions within each distance shell?

Long-Range Electrostatics: Your method captures interactions at longer distances through dilation. How do you see this approach performing on systems where long-range electrostatics are known to be dominant for the system's dynamics? Would the model be able to learn the (1/r) decay, or would it need to be integrated with a method like Neural P3M, as you suggest in your future work?

### Soundness
3

### Presentation
2

### Contribution
2
