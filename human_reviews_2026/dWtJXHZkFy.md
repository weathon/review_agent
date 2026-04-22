# Mesh Field Theory: Port–Hamiltonian Formulation of Mesh-Based Physics

- Avg Score: 5.50
- Decision: Reject
- Scores: 4, 6, 8, 4

## Abstract
We present Mesh Field Theory (MeshFT) and its neural realization, MeshFT-Net: a structure-preserving framework for mesh-based continuum physics that cleanly separates the physics’ topological structure from its metric structure. Imposing minimal physical principles (locality, permutation equivariance, orientation covariance, and energy balance/dissipation inequality), we prove a reduction theorem for mesh-based physics. Under these conditions, the physical dynamics admit a local factorization into a port–Hamiltonian form: the conservative interconnection is fixed uniquely by mesh topology, whereas metric effects enter only through constitutive relations and dissipation. This reduction clarifies what must be fixed and what should be learned, directly informing MeshFT-Net’s design.
Across evaluations on analytic and realistic datasets, physics-consistency tests, and out-of-distribution validation, MeshFT-Net achieves near-zero energy drift and strong physical fidelity—correct dispersion and momentum conservation—along with robust extrapolation and high data efficiency. By eliminating non-physical degrees of freedom and learning only metric-dependent structure, MeshFT provides a principled inductive bias for stable, faithful, and data-efficient physical simulation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The authors present a discrete exterior calculus framework for fitting port-Hamiltonian dynamics to co-chain data on meshes. The work is sound and an important direction for the overall scientific machine learning community. These types of formalisms are important to approach meaningful fields. This paper is acceptable for publication assuming the authors address a few major concerns.

My primary feedback is that the authors seem to be unaware of an important body of work where others have already pursued a similar strategy. There are distinct contributions in this paper that merit publication, but the authors must properly contextualize their contribution. There are two primary technical components here that others have also considered: (1) separating the (learnable) metric from the topological exterior derivatives and (2) posing dynamics via a dissipative bracket (port-Hamiltonian in their case) to ensure learned dynamics are stable despite the fact that model form is a priori unknown. The bulk of the literature review is out of date and focuses on the early HNN and LagrangianNN work that looked at low dimensional dynamical systems, and there have been some significant contributions since then which I lay out in the weaknesses section.

From my perspective, the identification of a class of a trainable port-Hamiltonian systems with topological connections is new. To my knowledge this has been done in other dissipative systems but not a port-Hamiltonian.  The authors stress 4 contributions (locally bounded receptive area, permutation symmetry, universality under orientation, and non-increasing energy). The first 3 are good but not particularly novel as they come "for free" working with message passing on graphs. The fourth is a primary contribution.

The benchmarks are of standard quality for ICLR or related AI focused conferences, but weak from the perspective of a serious numerical PDE community. They work through the acoustic scattering problem on the well, but this is simply the linear wave equation and should not be referred to in the title of 5.4 as "real physics field data". It would be much stronger if they attempted a more serious problem with nontrivial physics. At the core of this, there is a very strong assumption in lines 218-221 that J depend only on exterior derivatives independent of state. Very few systems admit J of this form (the wave equation and Maxwell are exceptions, which are coincidentally the cases the authors presented). Therefore it would be much more interesting for the authors to consider one of the significantly more nontrivial physics examples in the well. For example, shear flow could be swapped in with minimal effort and gives a good example of a simple dissipative system.

In the benchmarks, errors are presented in either one-step MSE or energy drift. The is a necessary metric for the method to work, but doesnt capture the primary claimed contribution (stable long time dynamics). Results should be presented for the accuracy of the entire time series.

### Strengths
Already covered in my summary. Technically sound and a strong direction of research that many are looking at right now.

The proofs are correct. They could be moved into an appendix as they are more or less a direct consequence of skew/spd symmetries and are standard for port-hamiltonian systems - this is just a suggestion if the authors wanted more space in the manuscript for additional benchmarks.

### Weaknesses
The following is a list of modern references from a few groups (Trask, Karniadakis, Cueto, Bronstein) who have worked in this area. Taking a look at these and the papers that cite them will help to contextualize this work.

- The DEC idea was already introduced: https://www.sciencedirect.com/science/article/pii/S0021999122000316
- Using DEC to identify dissipative brackets (although not port-Hamiltonian) https://proceedings.neurips.cc/paper_files/paper/2023/file/7903af0a1cffb43dbb2f8160d110a5f3-Paper-Conference.pdf
----- the authors are likely to find some useful techniques in the appendices that can be used to relax the assumption that J not be state dependent
- Metriplectic brackets (of which port hamiltonians are a special case) are being pursued by several groups:
-- https://arxiv.org/abs/2106.12619
-- https://royalsocietypublishing.org/doi/10.1098/rsta.2021.0207
-- https://arxiv.org/abs/2004.04653
-- https://arxiv.org/abs/2508.12569
- Bronsteins group have presented a number of papers posing GNNs in exterior calculus language: https://arxiv.org/abs/2106.10934

Meshgraphnets and HNNs are not close to state of the art for the well, although they do provide important pedagogical comparisons about whether combining a GNN and hamiltonian dynamics give you something more than the sum of the parts. A comment clarifying this would be useful (autoregressive vision transformers give much more accurate rollouts).

If this work is properly contextualized and the weakness of the benchmarks are improved this is suitable for ICLR - I would bump my score up significantly.

### Questions
I had a few technical points that weren't clear to me:

- Why use a splitting scheme to solve this, and not just directly solve the dynamics using a quasi-newton method? These are small 2D systems being considered, so it isn't clear to me that the computational savings justify introducing a splitting error in the dynamics.

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
2

### Summary
This paper focuses on Mesh Field Theory (MeshFT) and its neurbl version, MeshFT-Net, which represents an advance in mesh-based continuum physics. Ideas, such as the definition and separation of topological structure from metric and dissipative components, are novel. The authors define and formalize four minimal, yet critical, physical requirements: locality, permutation equivariance, orientation covariance, and energy balance/passivity. They demonstrate that the dynamics of mesh-based physics within these parameters allow local factorization into a port-Hamiltonian structure. The interconnection is determined exclusively by the mesh, leaving only the metric and dissipation components to be learned. Building on this insight, MeshFT-Net hardwires the signed incidence matrix as the fixed interconnection and learns positive-definite metrics and positive-semidefinite dissipative elements. MeshFT-Net is demonstrated to have near-zero energy drift and significantly improved physical fidelity. For the demonstration of this, authors use analytic plane-wave benchmarks, physics-consistency tests, and a real acoustic scattering dataset, "The Well,. Authors compare their approach to  MeshGraphNet (MGN), MGN with a Hamiltonian penalty, and Hamiltonian neural networks.

### Strengths
1. Demonstrated that mesh based physics, under simple physical assumptions, reduces to a port-Hamiltonian form where the interconnection is determined solely by mesh topology. 
2. MeshFT-Net hard-wires this interconnection and trains exclusively the metric and dissipation terms so that the updates are energy consistent. 
3. For 2D wave and acoustic tests, authors demonstrated near zero energy drift and more accurate wave speed and momentum conservation compared to MeshGraphNet and Hamiltonian baseline.. 
4. Similar or better accuracy is achieved with approximately five times less training data, and generalization occurs for different setups.

### Weaknesses
1. Other baselines compared remain classical (MGN, MGN-HP, HNN). Some recent work on graph simulators focus on oversmoothing or long-range dependencies. Additionally, there are some recent variants of MeshGraphNets, such as PI-MGNs (physics-informed MeshGraphNets). These baselines are not included and, as such, makes it harder to assess competitiveness.
2. Authors talk about “mesh-based physics” broadly, but results are mainly on linear wave/acoustics.
3. Probably, the "5x data-efficiency" claim needs additional verification and strict formulation (not rough).

### Questions
1. I think authors should narrow the "mesh-based physics" claims or add at least one non-linear / advection example to show the idea generalizes.
2. Make comparison with GraphCON or other recent work focusing to graph simulators with corrected long-range information flow (qualitatively or quantitively). Also, there are some recent variants of MeshGraphNets, such as PI-MGNs (physics-informed MeshGraphNets), it is interesting to compare with them because paper speaks about data efficiency. 
3. It is interesting to see some list of counterexamples where main theorem about local reduction doesn't work.
4. As I understand, Figures 2 and 4 provide data-size trends only for the analytic plane-wave task. I have not found it for example for “The Well” dataset. So, please either limit the “5× data efficiency” claim or add train-size analysis for additional datasets.

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
The paper shows that for mesh-based physics, imposing four minimal physical principles (locality, permutation equivariance, orientation covariance, and energy balance/dissipation inequality) ensures that system's physical wiring is fixed by the mesh itself. Building on this, the authors propose MeshFT-Net, which hard-codes the topology-driven wiring and uses stable time-stepping scheme. On 2D wave tests, out-of-distribution shifts (frequency/resolution/parameters), and a real acoustic dataset MeshFT-Net keeps energy/momentum stable and achieves competitive or better accuracy with less data.

### Strengths
- solid theory and result in Theorem 1 shows that mesh-based dynamics of mesh graph nets satisfying the built-in biases (locality and permutation equivariance) together with the physical principles introduced admit a local reduction to port–Hamiltonian representation.

- MeshFT-Net implements the theory in a principled manner

- compelling empirical evidence.

- the method is robust in OOD settings.

- the method is validated on the real physics-field data.

- ablations support the theorem's ingredients.

### Weaknesses
- local nature of the theorem: the reduction is jacobian level/local. global guarantees are not analyzed.
- the scope of PDEs used in the main paper are limited to 2D waves and acoustic dataset.
- including neural operator and DEC baselines would strengthen the claims.
- including runtime/wall-clock timings and memory scaling would strengthen the efficiency claims.

### Questions
see weaknesses

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
4

### Summary
This paper proposes a port-Hamiltonian formulation for mesh-based physics learning. It claims a "local reduction theorem" showing that the mesh dynamics on MeshGraphNet [1] satisfy orientation covariance and energy passivity, and can be factorized into the fixed topological interconnection $J$ (from incidence matrices $D$) and learnable metric/dissipation maps $G$ and $R$. Introducing these physical assumptions in MeshGraphNet, it designs MeshFT-Net, which hardwires the incidence-based skew operator $J$ and learns only symmetric positive (semi-)definite metric maps $(G_\theta, R_\theta)$.

*[1] Pfaff, T., Fortunato, M., Sanchez-Gonzalez, A., & Battaglia, P. (2020, October). Learning mesh-based simulation with graph networks. In International conference on learning representations.*

### Strengths
- Theoretical clarity: The separation between topology ($J$) and metric/dissipation ($G$, $R$) is clearly discussed and physically interpretable. In the corresponding model, enforcing skew-symmetry and energy passivity introduces strong structural bias that improve stability but potentially limit flexibility for systems outside the port–Hamiltonian class.

- Readable presentation: Proofs and algorithms are self-contained and systematically laid out.

### Weaknesses
See questions.

### Questions
- **More experiments are encouraged (e.g., parabolic, elliptic, or nonlinear systems). Or, alternatively, the authors could clarify the constraints under which the model may fail (non-Hamiltonian systems?).** As all experimental data presented here are wave-type systems, it would be helpful to know whether this approach provides benefits on other types of benchmarks.

- **Clarify overall complexity**: Although the authors analyze the complexity of individual modules, an analysis of the full model compared to baselines would give a clearer view of the model’s advantages. This would also help interpret the timing results reported in Table 9 and assess the model’s scalability.

- **Behavior near discontinuities**: The proof relies on local smoothness. How does the model behave near material discontinuities? Is there any demonstration of its ability to handle physical problems with discontinuities? Including benchmarks from the original MeshGraphNet or the Well dataset could be informative.

- **Necessity and minimality of assumptions (L, P, O, E):**

   - Are the assumptions (L, P, O, E) truly minimal for the reduction, or could a weaker or alternative set also yield the factorization? The proofs show sufficiency; can the authors provide evidence of necessity, or counterexamples when any single assumption is removed? If some physical systems slightly violate one of them, the reduction may fail.

- **Uniqueness of the $J$ / $G$ /$R$**:

   * The theorem is stated at the Jacobian level. Is this decomposition unique for nonlinear $F$ globally, or is it only a local Jacobian factorization? (Uniqueness is essential to claim that learning $G$ and $R$ is the only "freedom" left.)

- **Generalization of learned metric/dissipation terms**:
   - Are the learned $(G_\theta, R_\theta)$ mesh-invariant? That is, do they generalize across different embeddings, rescaled edge lengths, or even different mesh topologies? Suggested test: Train on one mesh and test on geometrically deformed or topologically different meshes without retraining.

Overall, I acknowledge that the model is effectively built on a rigorous topological analysis. However, I have reservations about its generalizability across different physical scenarios (*MeshGraphNet currently appears to perform better in this regard*) as well as some theoretical “strengths” and limitations. If the authors can address my concerns without introducing new major issues, I would consider raising my recommendation to marginally accept (readers may interpret my initial overall score as 5).

### Soundness
2

### Presentation
2

### Contribution
3
