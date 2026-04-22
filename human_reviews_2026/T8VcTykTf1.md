# Geometric Graph Neural Diffusion for Stable Molecular Dynamics Simulations

- Avg Score: 5.60
- Decision: Accept (Poster)
- Scores: 2, 8, 8, 6, 4

## Abstract
Geometric graph neural networks (Geo-GNNs) have revolutionized molecular dynamics (MD) simulations by providing accurate and fast energy and force predictions. However, minor prediction errors could still destabilize MD trajectories in real MD simulations due to the limited coverage of molecular conformations in training datasets. Existing methods that focus on in-distribution predictions often fail to address extrapolation to unseen conformations, undermining the simulation stability. To tackle this, we propose Geometric Graph Neural Diffusion (GGND), a novel framework that can capture geometrically invariant topological features, thereby alleviating error accumulation and ensuring stable MD simulations. The core of our framework is that it iteratively refines atomic representations, enabling instantaneous information flow between arbitrary atomic pairs while maintaining equivariance. Our proposed GGND is a plug-and-play module that can seamlessly integrate with existing local equivariant message-passing frameworks, enhancing their predictive performance and simulation stability. We conducted sets of experiments on the 3BPA and SAMD23 benchmark datasets, which encompass diverse molecular conformations across varied temperatures. We also ran real MD simulations to evaluate the stability. GGND outperforms baseline models in both accuracy and stability under significant topological shifts, advancing stable molecular modeling for real-world applications.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The study aim to address a critical limitation of existing Geometric Graph Neural Networks (Geo-GNNs) in molecular dynamics (MD) simulations: minor force prediction errors can destabilize long-temporal trajectories, especially when extrapolating to unseen molecular conformations (e.g., different temperatures) due to limited training data coverage. To solve this, the authors propose GGND, a plug-and-play framework that captures geometrically invariant topological features via an equivariant diffusion process on fully connected graphs, mitigating error accumulation and ensuring stable MD simulations. Experiments were conducted on the 3BPA and SAMD23 datasetsshow that GGND outperforms baselines in both accuracy and stability.

### Strengths
1. The paper provides solid theoretical support: it formalizes the causal mechanism of geometric topology formation (linking environment, cutoff radius, and latent graphons) and derives a regret bound for extrapolation error under topological shifts. Importantly, it proves GGND’s SE(3) equivariance via detailed analysis of gradient/diffusivity operators
2.  It compares with 4 representative baselines (NequIP, MACE, SEGNO, VisNet) and shows its usefulness in different models.

### Weaknesses
1. Scalability Limitation for Large Biomolecular Systems​
GGND’s fully connected graph diffusion introduces quadratic computational complexity (O(N²) for N atoms), making it impractical for large systems like proteins (millions of atoms) or complex biomolecules. The paper only tests systems up to 510 atoms (SAMD23 HfO), and its memory/time costs will grow exponentially with larger N. This restricts its real-world applicability to small/medium molecules, as large-scale MD simulations (e.g., protein folding) are common in biology and drug discovery.
2. Incomplete stability analysis: While bond length and RDF are used to measure stability, the paper does not evaluate other critical MD metrics (e.g., energy conservation, pressure fluctuations, or thermodynamic observables like melting points), which are essential for validating real-world utility.
3. Unclear relationship between diffusion operation and geometric topology. It lacks illustration how the diffusion operations map diffusion structural topology.
4. Lack strcuture analysis. Simulating a biomolecule at 1200K is WEIRD and meanningless. You can get nothing from the unrealistic experiment. The author should carefully check the structure changes during the simulation or perform another one under a normal environment.
5. lack interpretability: while it mentions attention weights reflect interaction strength, no visualizations are provided to illustrate how it captures unobserved topological features.
6. Time consumption
Again, the diffusion module seems ultra time-consuming. The authors should seriously analyze and demonstrate the extra huge computational cost during model training compared to the baseline model, e.g., ViSNet. Will it be applied to large molecular systems? 
7. Misleading concept of Generalization
The defination in this article seems a little misleading. When taking about generalization, the generalizable ability among different kinds of molecules, i.e., model trained in a molecules and tested on other kinds of molecules is much more important and convincing than that for different conformations of the same kind of molecule. Unfortunately, through the whole article, the authors neglected the important point for some reason.
8. Unphysical structures
Again, it's so werid and totally meaningless to sample protein structures at such high temerature (1200K). First, the author must show the structes they sampled at that temperature. I guess most of structures are totally collapsed. Second, the authors should perform classical MD simulations at such meaningless temperature to compare the RMSD/system energy or other observables with the models shown in the article.

### Questions
The authors should seriously address the concerns shown in Weakness point by point to improve the quality of the paper.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper addresses a critical instability issue in Geometric Graph Neural Networks (Geo-GNNs) used for molecular dynamics (MD) simulations. It identifies the cause as "geometric topological shifts," where models trained on certain molecular conformations (e.g., at 300 K) fail to extrapolate to unseen conformations (e.g., at 1200 K).

The authors propose Geometric Graph Neural Diffusion (GGND), a plug-and-play module that runs an equivariant diffusion process on a fully-connected graph. This allows the model to capture global, all-pair atomic information, making it robust to the local topological changes that cause standard models to fail. The paper provides a theoretical regret bound to justify this improved robustness and shows strong empirical results, with GGND massively improving simulation stability on the 3BPA and SAMD23 datasets.

### Strengths
**Significance**: It tackles a major practical bottleneck for ML force fields: long-term simulation stability. Solving this OOD extrapolation problem is critical for reliable MD simulations.

**Originality**: The solution is novel. Augmenting local message passing with a parallel, global equivariant diffusion process is a creative and technically sound approach to capturing physics robustly.

**Quality**: The paper is high-quality. The theoretical analysis provides a formal justification for the method's robustness. The experiments are excellent, using temperature variations (300 K vs 1200 K) to create a perfect test for the hypothesis, and the results are dramatic and convincing.

**Clarity**: The paper is well-written, clearly motivating the problem as a "geometric topological shift" and explaining the GGND solution .

### Weaknesses
**Scalability**: This is the most significant weakness. The method's reliance on a fully-connected graph introduces $O(N^2)$ computational and memory complexity, which the authors admit is "impractical" for the large-scale systems (e.g., proteins) where this stability is most needed.

### Questions
**Scalability**: Given the $O(N^2)$ bottleneck, have you explored approximations? For example, what is the performance trade-off if you use a large-radius cutoff (e.g., 20-30 Å) instead of a fully-connected graph?

**Computational Cost**: What is the practical wall-clock time overhead for a simulation step when adding the GGND module to a baseline like VisNet?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper addresses a critical challenge for Geometric Graph Neural Networks (Geo-GNNs) in molecular dynamics (MD) simulations: minor prediction errors can accumulate and destabilize long-term MD trajectories. The authors attribute this problem to the model's poor ability to extrapolate to unseen conformations, particularly those arising from environmental changes like different temperatures, a phenomenon they formalize as "Geometric Topological Shifts".

To tackle this, the paper proposes a novel framework called Geometric Graph Neural Diffusion (GGND). The core idea of GGND is to serve as a "plug-and-play" module that integrates with existing local equivariant message-passing frameworks. It achieves this by performing an equivariant diffusion process on a fully-connected graph, enabling instantaneous information flow between arbitrary atomic pairs. This process utilizes novel "equivariant gradient" and "equivariant diffusivity" operators, designed to capture geometrically invariant topological features, thereby mitigating error accumulation and ensuring stable MD simulations.

### Strengths
### Novel and Effective Method: 
The GGND framework is a novel contribution. Its core design—applying equivariant diffusion on a fully-connected graph —is intuitively sound. It overcomes the limitations of local message passing during topological changes by capturing "all-pair information flows". The ablation study  effectively demonstrates the superiority of "fully-connected diffusion" over "local diffusion" and "fully-connected message passing," validating the design's efficacy.

### Sufficient Experimental Validation:
- The experiment on the 3BPA dataset is well-designed, using different temperatures (300K, 600K, 1200K) to simulate conformational shifts. The results are impressive; for example, at 1200K, VisNet's stability is boosted from 0.004 ps to 11.209 ps, and MACE's stability also sees a 15-fold improvement.

- The comparison against multiple SOTA models on the SAMD23 dataset shows that GGND not only enhances stability but is also highly competitive in force and energy prediction accuracy on OOD splits.

- Solid Theoretical Support: The authors provide a theoretical analysis for GGND, including a proof of SE(3) equivariance and a regret bound under geometric topological shifts , which adds to the work's rigor.

### Strong Practicality: 
GGND is designed as a plug-and-play module that can be seamlessly integrated into various existing equivariant GNN architectures. This significantly increases the method's potential applicability and impact.

### Weaknesses
### Scalability: 
This is the paper's most significant limitation. The method's reliance on diffusion over a fully-connected graph incurs $O(N^2)$ computational and memory complexity (where N is the number of atoms). The authors acknowledge this makes the method "impractical" for large biomolecular systems. While the SAMD23 dataset reaches up to 510 atoms, this is still orders of magnitude smaller than many common systems in MD simulations

### Justification for the Diffusion Framework: 
The ablation study shows that fully-connected diffusion (GGND) outperforms a fully-connected message passing baseline (GGND‡). The paper attributes the baseline's poorer performance to "training challenges". This explanation is somewhat brief and does not fully explore why the PDE-based diffusion process is inherently superior to a standard (but also fully-connected) message-passing layer for this task.

### Questions
1. Scalability of GGND:

While the proposed method shows significant improvements in smaller molecular systems, can GGND scale efficiently to larger molecules or systems with a higher number of atoms? What are the computational bottlenecks, and how might they be addressed?

2. Comparison to Other Approaches:

How does GGND compare with other domain adaptation methods, such as active learning or test-time adaptation, in terms of improving stability and extrapolation? Have these methods been considered as part of your baseline comparisons?

3. Model Limitations:

The theoretical analysis shows GGND's ability to manage geometric topological shifts. Are there any specific types of conformational shifts or molecular behaviors that GGND might still struggle with, even with its diffusion mechanism?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper proposes Geometric Graph Neural Diffusion (GGND), a module augments local SE(3)-equivariant GNNs with a global graph diffusion process to propagate information across all atom pairs while preserving equivariance.  GGND introduces equivariant gradient and equivariant diffusivity (attention) operators, aiming to learn features that are invariant to geometric–topological shifts to improve long-horizon MD stability beyond in-distribution accuracy. Experiments on 3BPA and SAMD23 report gains in energy/force MAE and higher stability in OOD scenarios.

### Strengths
*  GGND is a plug-in-play diffusion module that can integrate with common equivariant GNN backbones, preserves equivariance, and facilitates global information mixing.

* On 3BPA across 600–1200 K, GGND increases stable rollout length by orders of magnitude versus baselines while simultaneously reducing energy/force MAE. On SAMD23 (SiN/HfO), indicating robustness well beyond in-distribution regimes.

### Weaknesses
Disclaimer: The theory analysis of the paper is beyond my expertise, so I’m open to the authors’ clarifications and to other reviewers’ perspectives.

The theorem 3.1 justifies how geometric graph neural diffusion is able to reduce the representation variation of the graph neural network, but I'm generally curious why (and when) graph neural diffusion is superior over  equivariant attention models (e.g., Equiformer, TorchMD-Net) once those baselines are given sufficient effective depth/receptive field. A discussion or an ablation study would help clarify the conditions under which GGND is actually superior.

### Questions
What is the compute and memory increase when adding GGND to GNN backbone?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The manuscript proposes Geometric Graph Neural Diffusion (GGND), a plug-in SE(3)-equivariant module that performs fully connected, diffusion-style updates on atom features and is then fused with a local equivariant GNN backbone. The central claim is that this global diffusion makes node representations insensitive to "geometric topological shifts" caused by temperature-dependent conformations and cutoff-dependent neighbor graphs, thereby improving out-of-distribution (OOD) stability in molecular dynamics. Formally, the model evolves features by a PDE-like update, $\partial_t Z(t)=(S(Z, X, t)-I) Z(t)$, and the paper presents a regret-style bound to argue reduced OOD error under adjacency shifts. Experiments on 3BPA (train at 300K, test at 600/1200K and dihedral slices) and on the SAMD23 SiN/HfO benchmark report notable gains in both force/energy MAE and a stability metric based on long-rollout NVE simulations up to 100 ps . The manuscript positions GGND as a general "plug-and-play" layer to retrofit existing local EGNNs.

### Strengths
From the angle of originality, the paper makes a concrete architectural move: it injects an equivariant, all‑pairs diffusion operator into otherwise local, cutoff‑dependent message passing, with spherical harmonics and Clebsch–Gordan structure to preserve symmetry. This is not only technically consistent with modern E(3)/SE(3) practice but is also aligned with a well‑motivated failure mode: changing neighbor graphs across temperatures or cutoffs degrade MLFFs even when in‑distribution errors look small. The formalization of “geometric topological shift” and the decomposition of the OOD gap give the work a clearer problem statement than usual in this area. The empirical section goes beyond pointwise MAE to include trajectory stability under NVE, which, in the opinion of the reviewer, is the right stress test to separate superficially accurate models from those that do not explode in practice.

### Weaknesses
The theoretical core reads fragile. In §3.1 the diffusivity $S(Z, X, t)$ is described as dependent on the adjacency $A$, which matches the paper's narrative about topology shift; however, the proof of Theorem 3.1 later asserts the model "assumes full connectivity" and that $S$ depends only on positions/features and thus is independent of $A$, hence $Z(T)$ does not depend on $A$ at all. Under that assumption, the main bound collapses to a trivial statement. This contradiction must be resolved unambiguously, because the claimed robustness hinges on the exact role of $A$. 

The reporting has unit inconsistencies that undermine trust. Table 1 explicitly states energy MAE in eV, yet the surrounding text speaks of meV for the same numbers and conclusions (e.g., improvements claimed “below chemical accuracy”). This is a three‑orders‑of‑magnitude discrepancy that changes the physical interpretation and must be corrected before one can judge the effect size.

The experimental design does not yet establish that the new diffusion mechanism is the decisive factor. The ablation includes a “fully‑connected message passing” variant, but there is no head‑to‑head against strong global equivariant baselines that already propagate information across all pairs, such as Allegro (strictly local but designed to capture many‑body correlations without message passing) and modern equivariant Transformers like EquiformerV2. Without those comparisons, one cannot attribute the OOD gains specifically to the PDE‑style update rather than to the mere presence of global interactions. A fair suite here would at least include NequIP and MACE (already present) plus Allegro and EquiformerV2, all tuned for the same cutoff and time step.

The evaluation scope feels narrow for the strength of the claims. Stability is reported up to 100 ps in NVE with Velocity‑Verlet and a 1 fs step; that is useful but short. The paper should probe longer horizons, and also include NVT (Nosé‑Hoover and Langevin) to show the method is not sensitive to ensemble or integrator/thermostat coupling. The “Forces are not Enough” paper makes exactly this point about trajectory‑level validation; aligning with that protocol would improve credibility.

Related work is incomplete in a way that directly affects positioning. GG‑ODE (KDD’23) explicitly addresses cross‑environment dynamics by learning a shared graph‑neural ODE with environment‑specific latent exogenous factors (e.g., temperature), a complementary strategy to the paper’s “invariance to topology change.” It is not discussed or compared, and the reader cannot see whether conditioning on environment would reduce or even remove the need for diffusion on a fully connected graph.

Finally, scalability is acknowledged as $O\left(N^2\right)$ but not quantified. If the core is all-pairs diffusion, memory and wall-clock scaling should be reported on realistic MD supercells, not only 27-atom 3BPA or 96-atom HfO cells. A convincing story would include sparsification or hierarchical diffusion and a clear regime where GGND remains practical.

### Questions
I would like to see precise clarifications to the following questions. First, is $S$ ever computed from a sparse $A$ in training or inference, or is the operator strictly complete-graph at all times? Please fix the contradiction between §3.1 and the proof of Theorem 3.1, and state the exact dependence on $A$ in the method section. Second, please correct units in every table and paragraph and re-evaluate claims like "below chemical accuracy"; the present mix of $\mathrm{eV} / \mathrm{meV}$ is not acceptable. Third, what happens beyond 100 ps and under thermostats? Showing stability curves up to $\geq 1 \mathrm{~ns}$ and including NVT/NPT would reduce concerns about integrator-specific artefacts. Fourth, why is there no comparison to Allegro and EquiformerV2 as strong global baselines? A single paragraph in Related Work is not sufficient; the comparison needs to be empirical on the same splits, with matched cutoffs, time steps, and neighbor policies. Fifth, can the authors release scripts for the stability metric (bond- and RDF-based) and the exact MD settings (neighbor list rebuilds, cutoffs, barostat/thermostat where applicable), since small choices there can create large apparent gaps? Finally, is there a way to incorporate environment conditioning, something like GG-ODE, on top of GGND to see whether invariance and conditioning are complementary rather than alternatives?

### Soundness
2

### Presentation
3

### Contribution
2
