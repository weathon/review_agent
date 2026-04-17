# Sigmadock: Untwisting Molecular Docking With Fragment-Based Se(3) Diffusion

Alvaro Prat∗
, Leo Zhang, Charlotte M. Deane, Yee Whye Teh, Garrett M. Morris Department of Statistics, University of Oxford

## Abstract

Determining the binding pose of a ligand to a protein, known as molecular docking, is a fundamental task in drug discovery. Generative approaches promise faster, improved, and more diverse pose sampling than physics-based methods, but are often hindered by chemically implausible outputs, poor generalisability, and high computational cost. To address these challenges, we introduce a novel fragmentation scheme, leveraging inductive biases from structural chemistry, to decompose ligands into rigid-body fragments. Building on this decomposition, we present SIGMADOCK, an SE(3) Riemannian diffusion model that generates poses by learning to reassemble these rigid bodies within the binding pocket. By operating at the level of fragments in SE(3), SIGMADOCK exploits well-established geometric priors while avoiding overly complex diffusion processes and unstable training dynamics. Experimentally, we show SIGMADOCK achieves stateof-the-art performance, reaching Top-1 success rates (RMSD < 2 & PB-valid) above 79.9% on the PoseBusters set, compared to 12.7-32.8% reported by recent deep learning approaches, whilst demonstrating consistent generalisation to unseen proteins. SIGMADOCK is the *first deep learning approach* to surpass classical physics-based docking under the PB train-test split, marking a significant leap forward in the reliability and feasibility of deep learning for molecular modelling.

## 1 Introduction

The biological function of a protein is determined primarily by its 3D structure and the interactions it mediates. Thus, a central goal of drug discovery is to design small-molecule ligands that bind to a target protein and modulate its function to achieve a therapeutic effect. Since a change in protein function is highly correlated with the bound pose of the binding ligand, a consequence of energetically favourable protein-ligand interactions, the ability predict these structural conformations, which is the primary aim of molecular docking, is essential for reliable and accelerated drug discovery. Deep learning approaches for molecular docking, in particular, diffusion-based methods, have been recently touted as providing superior accuracy over traditional, industry-standard physics-based methods (Halgren et al., 2004; Morris & Lim-Wilby, 2008; Trott & Olson, 2010). However, the need for such claims to be further qualified has been highlighted by Harris et al. (2023); Buttenschoen et al. (2024), who demonstrated that solely focusing on metrics, such as Root Mean Squared Deviation (RMSD) between the bound and predicted ligand poses, can obfuscate the actual predictive ability of deep learning-based docking tools. For instance, Buttenschoen et al. (2024) showed that when controlling for the chemical plausibility of generated samples, deep learning approaches performed far worse than traditional docking methods. While notable progress on this front has been made with the advent of co-folding models (Abramson et al., 2024; Boitreaud et al., 2024; Wohlwend et al., 2024) such as AlphaFold3 (AF3), which need not assume the protein to be a rigid structure, these models still have key limitations. Firstly, they require massive quantities of data and compute to train, hindering the ability of other researchers to contribute and improve such models. Secondly, as co-folding models jointly model the structure of proteins and ligands, they suffer from slow inference, which makes their practical applicability to drug discovery computationally prohibitive; especially for high-throughput virtual screening (HTVS), where it is often required to query millions of protein-ligand pairs.

∗Correspondence: alvaro.prat@stats.ox.ac.uk 1 To address these issues, we revisit the commonly-used *torsional model* approach (Corso et al., 2022; Huang et al., 2024; Cao et al., 2025) for diffusion-based molecular docking. This approach defines a diffusion process over a ligand's global roto-translations and its torsional angles. Fundamentally, structural (geometric) chemical constraints imply that this low-dimensional manifold forms the principal degrees of freedom underlying any chemically feasible pose. By operating over a space with significantly reduced dimensionality, torsional models promise improved data efficiency, generalisation and faster inference over *all-atom* approaches (favoured by co-folding models). However, this has not been borne out empirically, with relatively disappointing results reported in the literature (Skrinjar et al. ˇ , 2025).

In this work, we seek to resolve the discrepancy between the poor performance of torsional models and the desired benefits of exploiting inductive biases from structural chemistry. We suspect torsional models underperform because the score model must implicitly account for the mapping from 3D coordinates to torsional updates, a non-local, highly nonlinear, and sometimes ambiguous inverse problem. To bypass this burden, we propose a *fragment model*. We decompose a ligand into *molecular fragments* by breaking rotatable bonds; due to structural chemical constraints, we can treat each fragment's internal geometry as essentially *fixed*. The generative task thereby reduces to predicting an SE(3) rigid transformation for every fragment, from which any chemically feasible pose can be recovered by composing these transformations, obviating explicit modelling of torsional angles. Building on this, we introduce SIGMADOCK, an SE(3) Riemannian diffusion model that defines a diffusion process over the translation and orientation of rigid-body fragments. During sampling, SIGMADOCK iteratively reassembles the ligand's constituent fragments into a predicted bound pose. To reduce the additional degrees of freedom introduced from fragmentation
(compared to the torsional model), we make the following novel contributions: (i) a fragmentation scheme that reduces the number of fragments required to represent a ligand; (ii) soft triangulation constraints to provide further inductive biases on the preserved bond lengths and angles across fragments; (iii) an SO(3)-equivariant architecture tailored for reasoning over fragment geometry and protein-ligand interactions. We adopt the standard re-docking protocol in which the receptor is fixed (*holo-conformation*) and the binding pocket is known. This choice is deliberate for two reasons: first, it is the long-standing setting for benchmarking docking methods, and permits fair 'apples-to-apples' comparison with classical and recent deep learning baselines; second, it reflects industrial HTVS/lead-optimisation practice, where rigid-receptor docking is computationally tractable at scale. Prior generative methods have struggled in precisely this regime, and our aim is to close that gap first.

Empirically, we demonstrate SIGMADOCK surpasses prior deep learning approaches1and traditional physics-based docking, achieving *state-of-the-art* performance on the challenging Pose- Busters set (Buttenschoen et al., 2024), and generalising to unseen proteins (Figure 4). We highlight that, with a fraction of the training data, training/sampling time, and lower test-train leakage, we reach AF3-level performance and substantially outperform previous generative methods on the redocking task. With this, we wish to state our main contribution as the careful and rigorous design of a well-characterised diffusion process, and a detailed construction of structural inductive biases which help learn simpler functions for the task of molecular docking. We highlight that our fragment SE(3)m formulation naturally extends to flexible docking by treating selected side chains as additional fragments, and is also adaptable to co-folding, both of which we leave as future work.

## 2 Method 2.1 Notation

Molecular notation. The 3D graph of a molecular structure (e.g. ligand, protein or molecular fragment) is defined by the collection G = {x, v, b} where |G| is the number of atoms in the structure, x ∈ R
|G|×3are atomic coordinates, v ∈ R
|G|×da represents the features (e.g. atom type, atomic charge etc.) associated with each atom, and b ∈ R
|G|×|G|×dbrepresents the features
(e.g. bond type, bond conjugation etc.) associated with each bond. Furthermore, we define the corresponding 2D graph of a molecular structure by G
2D = {v, b}. For further details on our choice of molecular featurisation see Appendix G.1.

1Under fair comparison with models trained on the PoseBusters train-test split.

![2_image_0.png](2_image_0.png)

SE(3) **notation.** Let x = [x1*, . . . , x*N ] ∈ R
N×3 be a collection of points xi ∈ R
3in a rigid-body system where [·] denotes row-wise concatenation. The pose (*i.e.* position and orientation) of x can be parametrised by elements (p, R) from the Lie group SE(3) via the standard group action:
(*p, R*) · x = [p + R · x1*, . . . , p* + R · xN ],
where p ∈ T(3) ∼= R
3is an element of the translation group and R ∈ SO(3) is a special orthogonal matrix representing a rotation in R
3. For further details about SE(3), see Appendix B.

Fragment notation. In this work, we develop a novel fragmentation scheme which allows us to represent the 2D graph of a ligand G
2D
ligand = {v, b} in terms of *rigid-body fragments* {GFi}
m i=1 where GF = {x˜F , vF , bF } for each F ∈ {Fi}
m i=1. In particular, we take x˜F ∈ R
|GF |×3to represent the *local coordinates* of the fragment centered at the origin2, *i.e.* 1 |GF | 1 T x˜F = 0 where 1 is a vector of unit entries; at a high level, this can be viewed as the predetermined coordinates of a local, rigid substructure within the ligand due to structural chemical constraints from G
2D
ligand. Hence, the 3D coordinates x of the ligand can only be constructed from some arrangement of the rigid-body translations and rotations of {x˜Fi}
m i=1. Formally, we identify z = (p, R) ∈ SE(3)m with the *global* coordinates {xFi}
m i=1 of the fragments through the usual group action: xFi = (pFi, RFi) · x˜Fi where p = (pF1*, . . . , p*Fm) ∈ T(3)m and R = (RF1*, . . . , R*Fm) ∈ SO(3)m. This allows us to parametrise the pose of the ligand x in terms of SE(3)m by x = [xF1*, . . . ,* xFN ] and we denote this mapping3 by φ : SE(3)m → R
|Gligand|×3.

## 2.2 Structurally-Aware Fragmentation For Se(3) Diffusion

The task of molecular docking can be summarised as predicting a ligand's bound pose x ∈ R 
|Gligand|×3for some query protein, given the 2D graph of the ligand G
2D
ligand = {v, b} and the 3D
graph of the protein Gprotein = {y, vy, by}. The guiding idea behind SIGMADOCK is to exploit the inherent structure of a ligand's topology to simplify and condition a smooth and well-defined rigid-body diffusion process in SE(3). In particular, we rely on a novel fragmentation strategy that preserves common stereochemical symmetries, reduces degrees of freedom, and creates a set of geometric priors which help SIGMADOCK learn more general physicochemical correlations. A visual overview of the SE(3) diffusion process described in this section is outlined in Figure 1.

## 2.2.1 The Conformational Manifold

SIGMADOCK makes use of well-established thermodynamic priors in structural chemistry. Namely, it leverages the fact that the conformational space xc defining the local geometry of a ligand (in a vacuum) with known topological structure G
2D
ligand follows a Boltzmann distribution with probability mass concentrated about a manifold Mc with holonomic constraints4 of the form:
Mc = {xc ∈ R
|Gligand|×3: g(xc) ≈ 0},
2We defer the issue of choosing the orientation of the local coordinates to Section 2.4.

3We abuse notation by leaving the dependence on {x˜Fi }
m i=1 from the fragmentation {GFi }
m i=1 implicit.

4Holonomic constraints (see (Ryckaert et al., 1977)) restrict the configuration space (position) of a body.

![3_image_0.png](3_image_0.png)

where g : R
|Gligand|×3 → R
m
+ maps from Cartesian coordinates to an m-vector of scalar holonomic constraints, encoding m independent (soft) boundary conditions. The latter represent locally conserved geometrical priors such as bond lengths (dAB = d0) and bond angles (τABC = τ0), and exclude dihedral angles/torsions (ϕ*ABCD* = ϕ); although the distribution ϕ*ABCD* is anisotropic, bodies adjacent to a torsional bond defined over atoms *B, C* are free to rotate (Figure 2a). Importantly, we can safely assume the holonomic constraints implicit in Mc can be derived from G2D
ligand and thermodynamic equilibria. A more formal definition of the form of g(·) is detailed in Appendix D.1. Under this construct, the probability measure supported on Mc is the constrained Boltzmann measure πMc which we may sample from: xc ∼ πMc(G
2D
ligand). As shown in Figure 2c, although this distribution is complex and multimodal, there are abundant preserved symmetries: with RDKit's ETKDGv3 (Landrum (2025)) as our proxy for πMc, we observe that, excluding global rigid motions, the 3D structural differences in xc are effectively dominated by changes in dihedrals (torsion).

To faithfully sample rigid body fragments, we first need to justify that samples in SE(3)m drawn from the conformational manifold Mc can be consistently aligned to the bound manifold (Mb),
whose distribution πMbrepresents the target (data) distribution of bound states. We verify this by aligning ligand conformations from Mc to Mb via joint roto–translational and torsional registration.

Concretely, let xc ∼ πMc
(·) denote a conformer on Mc and let ϕc ∈ T
krepresent the corresponding dihedral angles which define the k torsional bonds in xc. Let ψ(ϕc) : T
k → R
|Gligand|×3 denote the invertible map, where T
k = (S
1)
k ∼= SO(2)kis the hypertorous defining the space of dihedrals.

With (*p, R*) ∈ SE(3) representing global rigid motion, we use the total map Ψ(*p, R,* ϕ) = (*p, R*) · ψ(ϕ) ∈ R
|Gligand|×3, and perform Kabsch alignment jointly with torsional adjustment by solving min p∈R3, R∈SO(3), ϕ∈Tk RMSD( xb, Ψ(*p, R,* ϕ) ), xb ∼ πMb
(G
2D
ligand)
Empirically, we find that the RMSDs between experimentally bound poses xb and their corresponding aligned conformers x
′b = Ψ(p
⋆, R⋆, ϕ
⋆) are substantially below both (i) standard error rates reported by docking baselines (Buttenschoen et al., 2024; Harris et al., 2023), and (ii) the commonly used success threshold of 2A. This provides sufficient support to claim that the variability in ˚ bond lengths and bond angles subsumed in Mc can be generally ignored in the task protein-ligand docking. Crucially, this allows us to treat bound states as being approximately contained in the set of structures reachable by torsions and SE(3) transforms on conformers drawn from πMc: for any xc ∈ Mc and xb ∈ Mb, we can align xc to xb with negligible error such that RMSD(xb, x
′b
) ≪ 2A. ˚
See Figure 2b for an aligned example and Appendix D.3 for additional results and empirical analysis. This empirical inclusion is paramount to our approach as it justifies assembling our stationary distribution from fragments sampled from Mc, without falling out of distribution. Throughout the remainder of the paper, we will absorb the alignment into the notation by writing x ← xb ← x
′b
,
where we use the aligned conformation x
′b at the start of forward noising process.

## 2.2.2 Challenges Of Torsional Models & Motivation For Our Method

The idea behind directly modelling dihedral angles has been adopted as the standard approach to modelling small molecules (Corso et al., 2022; Jing et al., 2023) and amino acid side-chains in proteins (Jumper et al., 2021). However, formulations that directly model time-dependent dihedrals ϕABCD(xt, t) via torsional updates suffer from fundamental caveats which we aim to resolve in our approach. In Theorem 1 we show that the induced torsional density in Cartesian space is generally not a product distribution, leading to highly entangled implicit dynamics. For further details and a proof of Theorem 1, see Appendix C.2. Theorem 1. *For standard molecular topologies, torsional models define nonlinear mappings from* torsion angles to Cartesian coordinates, producing highly entangled, non-product induced measures. In contrast, disjoint rigid fragments yield a factorised product of Haar measures on SE(3)m. Consequently, we argue that diffusing a molecule in fragment space SE(3)m offers a simpler learning task than diffusing in torsion space T
k × SE(3). Intuitively, local changes in torsional angles often produce non-local Cartesian displacements, creating strong geometric coupling along torsional chains: a change in a single torsion can substantially displace remote atoms. Therefore, independent torsional perturbations become correlated once mapped to Cartesian coordinates (where the model observes the data) under the induced measure, breaking the product structure. This leads to an ill-conditioned learning problem and stiff sampling dynamics. In contrast, our forward kernel factorises over disjoint SE(3)m fragments (product); inter-fragment correlations enter only via the learnt score, rather than being induced by the noise, yielding simpler, better-conditioned mappings, and more stable reverse-time integration.

Furthermore, mapping a torsional increment ∆ϕito a Cartesian displacement ∆x is intrinsically ambiguous: one must chose an extrinsic gauge (which side of the torsional bond is rotated, or which combination). Implementations often apply heuristics such as RMSD alignment to remove the net rigid motion caused by torsional updates, which would otherwise break the product-space structure. However, this does not mitigate the ambiguity of the intrinsic to extrinsic mapping, especially when torsional steps are large. Practical solutions often commit an extrinsic realisation (rotate left, rotate right, or a combination), and the model must learn a score consistent with that convention; this choice cannot guarantee consistency during sampling as the selected torsional realisation may not align with the true score direction. Moreover, as k (and thus molecular size and flexibility) increases, torsions produce amplified nonlocal Cartesian displacement (lever effect), coupling distant degrees of freedom. The combinatorial growth of possible extrinsic realisations exacerbates this geometric entanglement, making this framework unscalable. We hypothesise that, in general settings, these issues make torsional frameworks can become poorly conditioned and unnecessarily complex to model. These shortcomings motivate our approach of representing molecules via independent rigid fragments, allowing us to operate over a well defined and geometrically independent product space.

## 2.2.3 Irreducible Fragmentation & Soft Geometric Constraints On Se(3)

The naive choice to define our fragments is to break the molecular graph obtained from πMcat the torsional bonds, producing a set {GFi }
mˆ
i=1 of torsion-free rigid-body fragments with global coordinates parametrised by SE(3)mˆ. This approach yields a set of mˆ = (k + 1) fragments with a total of 6 ˆm DoFs5. In contrast, we note torsional models have (k + 6) DoFs (k torsional bonds in S
1and 6 for rigid body SE(3)). Thus, the natural question arises: How can we reduce the DoFs of the system and in turn abstract the problem in a general form? In SIGMADOCK we tackle this problem by creating a simple yet effective molecular fragmentation reduction (FR3D) that recursively merges adjacent fragments from mˆ = (k + 1) down to m (Figure 3). Instead of biasing the fragmentation order, FR3D performs a stochastic search, starting from the torsion-free mˆ fragments and branching through candidate neighbour proposing merge actions until reaching an irreducible set of size m. Hence, FR3D not only reduces the learnable DoFs by reducing 5Assuming |GF | > 1, each fragment is defined by its T(3) ∼= R
3and SO(3) parametrisation.

the number of fragments, but also provides a promising stream for data augmentation. Merging is possible in molecular graphs where two or more consecutive torsional bonds are linked, so there are topologies that are irreducible. Hence, we are upper-bounded in the number of fragments: 1 ≤ m ≤ k + 1. For a fragment hyper-graph with m fragments (and no loop closures), the effective DoFs concentrate between k + 6 (triangulation-induced lower bound)6and 6m (unconstrained). Under na¨ıve fragmentation (mˆ =k+1), this becomes k+6 ≤ DoF ≤ 6 ˆm. As FR3D reduces m (empirically we find m ≈
2 3mˆ ), the upper bound shrink in practice. We refer the reader to Appendix D.4 to a more extensive analysis and empirical results. During fragmentation, we retain torsional bond length and angle information by introducing dummy atoms at either side of the bond. Hence, |Gligand| ≤ Pm i=1 |GFi |. Importantly, we only retain *free* dummy atoms in GFiand otherwise prune dummies which are *over-constrained*. Here, we label a dummy as over-constrained whenever FR3D merges the torsional bond it belongs with a neighbouring fragment, as it naturally over-defines a dihedral angle (Figure 3b). Removing over-constrained atoms is fundamental since an immutable dihedral sampled from πMc would violate the free torsional requirements outlined in Section 2.2.1, forcing our generator πMcto yield structures that do not strictly overlap with the bound manifold under optimal alignment: Ψ · πMc
(·) ̸= πMb
(·).

We refer the reader to Algorithm 1 in Appendix D.4 for an overview of FR3D and further analysis. Figure 3: Illustrative example of how FR3D reduces the number of fragments (colour coded) re-

![5_image_0.png](5_image_0.png) quired to represent rigid bodies on ligand TNK into irreducible form. A: Defining fragments by snapping all torsional bonds (ribbons); B: FR3D recursively attempts to reduce the k torsional bonds and removes over-constrained dummies in the process (denoted by the coloured rings), which otherwise define a dihedral across the merged fragment; C; Over-constrained dummies removed and triangulation edges displayed under a different stochastic reduction (equiprobable to solution b).

Soft geometric constraints. A core ingredient of our method is the inclusion of geometric priors as a mechanism to provide soft (implicit) boundary conditions. Specifically, although FR3D produces irreducible fragments, we define a triangulation distance conditioning scheme which enables pseudo-reductions to the observable DoFs. Concretely, for any torsional bond BC connecting adjacent fragments A and D, we define triangles (*A, B, C*) and (*B, C, D*) using neighbouring atoms A ∈ A and D ∈ D on either side of the set of dihedrals ϕ*ABCD* across BC. Through Lemma 1, we show that by defining cross-fragment distances ||A − C|| and ||B − D|| on top of the rigid fragment template, the corresponding bond angles ∠(*A, B, C*) and ∠(*B, C, D*) become uniquely determined.

See Figure 3c for an illustration, and for a proof of Lemma 1, see Appendix D.2. Lemma 1. ∀(A, B) ∈ (A, D) bond lengths ||A − B||, ||B − C||, ||C − D|| *and bond angles*
∠(A, B, C), ∠(B, C, D) are fully determined with triangulation conditioning, without restricting changes in the dihedral angles ∆ϕ*ABCD*.

## 2.3 Se(3) Diffusion

From our identification of ligand poses x with z = (p, R) ∈ SE(3)m via the fragmentation {GFi }
m i=1, we adopt the SE(3) diffusion model framework introduced in Yim et al. (2023)
to construct a generative model pθ(z|Gdock) for sampling the docked pose of some ligand, 6The resulting DoFs are lower-bounded because the triangulation scheme imparts soft boundary constraints; it provides a strong signal for SIGMADOCK to reduce ∆dA,C to 0 as t → 0.

given its 2D graph G
2D
ligand, its fragmentation {GFi}
m i=1 and some query protein Gprotein; we use Gdock = (G
2D
ligand, {GFi }
m i=1, Gprotein) to denote this conditioning information. We provide an overview of this framework below, and for further details, see Appendix C.

Forward process. For each protein-ligand pair (Gligand, Gprotein) in our dataset pdata, with an associated fragmentation {GFi
}
m i=1, we define the forward process (Z
(t))t∈[0,T] = ((p
(t), R(t)))t∈[0,T]
with Z
(t) ∼ pt(z|Gdock) via the SDE:

$$\mathrm{d}{\bf Z}^{(t)}=\left[-\frac{1}{2}{\bf p}^{(t)},0\right]\mathrm{d}t+\left[\mathrm{d}{\bf B}_{\mathbb{R}^{m\times3}}^{(t)},\mathrm{d}{\bf B}_{\mathrm{SO}(3)}^{(t)}\right],\tag{1}$$

where B
(t)
Rm×3 , B
(t)
SO(3)m denotes Brownian motion on R
m×3and SO(3)m respectively, with the initial condition Z
(0) = (p
(0), R(0)) = φ
−1(x) where Gligand = {x, v, b} contains the ground-truth docked pose x. We note that the SDE is designed for the forward kernel pt|0(z
(t)|z
(0)) to be tractably sampled from, and we take T > 0 large enough for pT to be close to the stationary distribution q(z) = N (p; 0, I) ⊗ USO(3)m(R), where USO(3)m denotes the uniform distribution on SO(3)m.

Backward process. The associated backward process (
←
Z
(t))t∈[0,T]is then given by the SDE:

d
$$\hat{\mathbf{Z}}^{(t)}=\left[\frac{1}{2}\hat{\mathbf{D}}^{(t)}+\nabla_{p}\log\text{pr}_{T-t}(\hat{\mathbf{Z}}^{(t)}|\mathcal{G}_{\text{dock}}),\nabla_{R}\log\text{pr}_{T-t}(\hat{\mathbf{Z}}^{(t)}|\mathcal{G}_{\text{dock}})\right]\text{d}t+\left[\text{d}\mathbf{B}^{(t)}_{\text{R}^{m\times3}},\text{d}\mathbf{B}^{(t)}_{\text{SO}(3)^{m}}\right],\tag{2}$$
where ∇z log pt(z|Gdock) = [∇p log pt(z|Gdock), ∇R log pt(z|Gdock)] denotes the score function of the induced probability path pt; we note that this should be understood as a Riemannian gradient which lives in the tangent space Tanz SE(3)m.

Training and sampling. We see that we can generate bound poses under Gdock from simulating the backward SDE in Equation 2, however, the true score function ∇z log pt is intractable. Therefore, we train a neural network approximation sθ(z*, t,* Gdock) via the score matching objective:

$$\mathcal{L}(\theta)=\mathbb{E}_{p(t),p_{\text{data}}(\mathcal{G}_{\text{post}},\mathcal{G}_{\text{prompt}}),p_{1|0}(\mathbf{Z}^{(t)}|\mathbf{Z}^{(0)})}\left[\left\|s_{\theta}(\mathbf{Z}^{(t)},t,\mathcal{G}_{\text{fork}})-\nabla_{z}\log p_{1|0}(\mathbf{Z}^{(t)}|\mathbf{Z}^{(0)})\right\|_{\text{SE}(3)=\pi}^{2}\right].\tag{3}$$
$$(1)$$

Hence, we denote pθ(z|Gdock) as the distribution of generated samples, from first sampling
←
Z
(0) ∼ q and then simulating the backward SDE with our learnt approximation sθ, which approximates the true distribution p0(z|Gdock). The corresponding 3D coordinates xˆ of samples zˆ ∼ pθ(z|Gdock) can then be recovered by the mapping xˆ = φ(zˆ).

## 2.4 Architecture

A significant contribution of SIGMADOCK is the design of our architecture sθ(z*, t,* Gdock) which parametrises the score function. In particular, we augment EquiformerV2 (Liao et al., 2023) to handle protein-ligand (and other molecular) diffusion; we use this as the backbone for our model to ensure SO(3)-equivariance7. Our main innovations are: (i) we augment the input graph with virtual nodes and edges on top of the original chemical graph Gdock, creating a hierarchical topology.

This reduces risk of over-squashing by reducing the average node degree, whilst promoting global information flow and mitigating over-smoothing: less layers needed to pass global information; (ii) we tailor our featurisations of nodes and edges according to their structural role; (iii) we ensure messages and gradients along the edges, which represent local interactions (present on proximity), smoothly decay to zero as the distance between the neighbouring nodes approaches some cutoff; this prevents instabilities from sudden changes in the input graph's topology as we perturb z. Moreover, we note that a critical issue for the design of our architecture is that the parametrisation of the global coordinates xF in terms of (*p, R*) ∈ SE(3) is *not uniquely* defined. This is due to the fact that we do not have a canonical choice for the orientation of the local coordinates x˜F . For instance, we have the equally valid choices x˜F , x˜
′Ffor the local coordinates of GF if x˜
′F = R0 · x˜F where R0 ∈ SO(3). Hence, we can have two different representations of global coordinates xF from xF =
7EquiformerV2 is also translation invariant but we do not require this property since our problem setting has a canonical centre of mass given by the binding pocket.

(*p, R*)·x˜F = (*p, RR*−1 0)·x˜
′F
depending on the initial choice of orientation. To resolve this issue, we adapt the SO(3)-equivariant prediction head introduced in Jin et al. (2023), based on the Newton- Euler equations from rigid-body mechanics, to which we pass the outputs of our backbone model into. Particularly, we predict pseudo-forces for all atoms pertaining to the m fragments and use these as a basis to define our scores in the tangent space of SE(3) (more details in Appendix G.4). With this choice, Theorem 2 shows that SIGMADOCK is invariant to the choice of local coordinate axes. Theorem 2. *Our training objective and sampling procedure are invariant with respect to the choice* of orientations for local coordinates. Moreover, our score model is SO(3)*-equivariant which ensures* pθ(z|Gdock) *is a stochastically* SO(3)*-equivariant kernel.*
Conditioning. We define the triangulation conditioning by feeding the relative distance mismatch as an edge feature (compact notation): ∆dA,C (xt, t|Gdock) = ||A(t) − C(t)*|| −* d ref A,C , (with d ref A,C
defined from the initial RDKit conformer), such that limt→0 ∆dA,C (xt, t|Gdock) = 0 across all cross-fragment triangulation edges. Dummy atoms which define triangular geometry are discarded after sampling; torsional bonds are reconstructed from anchors so there is no discrepancy between conditioning inputs and the final conformation. With this conditioning, only dihedral angles and rigid body roto-translations remain free as t→0. For further architectural details, see Appendix G, and for a proof of Theorem 2, see Appendix H.1.

## 2.5 Training And Inference

We outline our training setup for SIGMADOCK in Appendix E. In particular, we discuss how we preprocess our data for fragmentation and training, as well as computational tricks for increasing training throughput. We detail the definition of the binding pocket in Appendix E.1. Briefly, the pocket includes all residues with any atom within a stochastic cutoff dr of any ligand atom. Formally, dr:= d0 + N (0, σr), where d0 and σr default to 5A and 1 ˚ A respectively during training ˚ . Our sampling procedure is outlined in Appendix F, where we discuss the fact that, due to the reliability of SIGMADOCK in generating chemically plausible samples, SIGMADOCK *does not* require the use of a separately trained confidence model to filter out poor generations. Instead, we propose using the simple and cheap heuristic of evaluating both the (pseudo) binding energy of the generated proteinligand system, as well as a set of physicochemical checks (such as, bond angles, bond lengths, internal energy) to rank our Nseeds samples for evaluation.

## 3 Experiments 3.1 Data And Metrics

Datasets. We use PDBBind(v2020) (Wang et al., 2005), a curated set of 19,443 protein-ligand complexes obtained through crystallography, as our training set. Crucially, we deliberately restrict ourselves to this dataset for fair comparison8, isolating any increase in performance obtained in this study to our proposed framework. For validation, we use the well established PoseBusters (Buttenschoen et al., 2024) and Astex (Hartshorn et al., 2007) datasets. PoseBusters(v2) (PB) acts as our temporal-split validation set containing 308 protein-ligand complexes with unseen protein sequences realised from 2021 onwards. The Astex (AX) dataset consists of an additional 85 diverse and highly curated protein-ligand complexes originally designed to faithfully evaluate the quality of protein-ligand docking algorithms.

Metrics. We evaluate a generated pose xˆ by measuring the symmetry-corrected RMSD (Meli & Biggin, 2020) between the crystallographic (bound) pose x and the generated pose xˆ obtained by the mapping xˆ = φ(zˆ) where zˆ ∼ pθ(z|Gdock). To account for sampling variability, either from different conformers (inducing differences in fragment local coordinates), or by resampling z
(1) ∼ q(z), we report the Top-k success rate, i.e., the fraction of complexes where at least one of the top k poses
(from Nseeds samples) has RMSD < 2A. We also use PoseBuster to assess PB-validity, indicating ˚
whether generated structures also satisfy standard physicochemical tolerances.

## 3.2 Results

Using the sampling algorithm described in Appendix F, we benchmark the base performance of SIGMADOCK and present our main results in Figure 4. To the best of our knowledge, SIGMADOCK is the first deep learning-based method to surpass classical physics-based approaches in the PB
and AX sets using the intended train-test split on the re-docking task9. Not only does SIGMADOCK
achieve a 6.3× higher PB-validity than DiffDock, the best open-source alternative tested on the same split, but it also excels on proteins with low sequence similarity, overcoming the common critique that deep learning models memorise rather than learn physics. Notably, Corso et al. (2024) train DiffDock-L on a significantly larger corpus (PDBBind(v2020) ∪ BindingMOAD) and report 50% Top-1 (RMSD only) on PB, whereas SIGMADOCK attains a Top-1 (RMSD & PB validity) of 79.9%.

Overall, these results support our main contribution: a theoretically-grounded SE(3)m Riemannian diffusion framework with strong generalisation. Conversely, torsional-space baselines and pointcloud docking models evaluated under identical conditions do not attain comparable results. We also highlight SIGMADOCK does not require minimisation to achieve high PB validity, a common yet computationally expensive hack used to artificially improve deep learning methods. Notably, we achieve AF3-level performance (Top-1 of 84%: see Extended Data Fig. 4e in Abramson et al. (2024)) with just 19k training data-points, *significantly lower train-test leakage* (see App. J), and 50× faster sampling. Together with an outstanding performance in the AX set, reaching nearperfect Top-1 (above 90%), we believe these results mark a *major leap forward* in the feasibility and reliability of deep learning for molecular modelling.

![8_image_0.png](8_image_0.png) 

Ablations. To better characterise and highlight the contribution of some key components in our method, we perform an ablation study covering a set of training-time and test-time variables (see Table 1). Namely, we report the influence of our fragmentation merging strategy and triangulation conditioning, as well as the effect of including protein-ligand interactions as part of the computational graph, and give empirical evidence of their relevance (4-12% relative improvement). In addition, we show how sampling fragments from Mc vs. Mb leads to a small but expected decrease in sample quality. By excluding PB-checks in the heuristic, SIGMADOCK maintains a high PB-valid Top-1. Finally, we show how increasing Nseeds improves performance at the expense of more computational overhead, and highlight the importance of our simple yet effective heuristic for ranking our samples and picking our best candidate(s). On top of sequence similarity, we stratify the PB set into distinct chemical environments determined by the nature of ligand interactions with additional co-factors (ions, crystallisation aids, natural ligands, or other co-factors). We hypothesise that, since SIGMADOCK is deliberately designed (for simplicity) to exclude co-factors, higher failure rates should be observed when the true bound pose is realised in conjunction to additional artefacts (co-binding event), as the setup is partially observable.

After isolating the protein-ligand pairs for which SIGMADOCK fails to generate accurate poses10, we find this hypothesis to hold true, as per Table 2. This result provides additional confidence that SIGMADOCK does not blindly memorise and hallucinate protein-ligand poses.

| Conf.   | Description             | RMSD < 2   | PB Val.   |
|---------|-------------------------|------------|-----------|
| A       | (−) Tri. Cond.          | 71.9       | 67.1      |
| B       | (−) PL Interactions     | 79.2       | 76.3      |
| C       | (−) Frag. Merging       | 74.4       | 73.7      |
| G       | Sampling from Mb        | 86.4       | 85.4      |
| D       | (−) Energy Scoring      | 67.2       | 66.1      |
| E       | (−) PB Scoring          | 82.1       | 70.8      |
| H       | SIGMADOCK (Nseeds = 10) | 74.7       | 72.2      |
| ∗       | SIGMADOCK (Nseeds = 40) | 80.5       | 79.9      |
| I       |                         |            |           |

Table 2: Performance analysis (Top-1 accuracy (%)) across PB subsets according to the presence of various co-factors. The subset size is shown next to the co-factor species key. The failure rate represents the sample failure rate, averaged across 40 seeds, for all complexes in the subset.

| Co-factor Presence        | RMSD < 2   | PB Val.   | Fail. Rate   |
|---------------------------|------------|-----------|--------------|
| Natural Ligands (17)      | 58.8       | 58.8      | 41.2         |
| Ions (57)                 | 75.4       | 75.4      | 23.6         |
| Other (60)                | 76.7       | 76.7      | 28.1         |
| Crystallisation Aids (37) | 81.1       | 81.1      | 35.0         |
| None (165)                | 84.2       | 83.0      | 16.2         |

Docking tools are typically provided with a search region (e.g. bounding box or centre-radius) defining the pocket search space. To assess robustness to larger pockets, corresponding greater uncertainty of the binding region, we sweep the deterministic cutoff d0, whilst reducing the jitter σr → 0. As presented in Table 3, SIGMADOCK remains robust to larger pockets, with a moderate drop when operating outside the training support (e.g. ∅pocket = 7A is 2 ˚ σ relative to the training mean d0 = 5, σr = 1). Notably, reducing the pocket size (--autobox + 5A) does not improve ˚
Vina's Top-1 (57.2% vs. 56.0%), indicating that gains over classical methods are not attributable to smaller pocket definitions. Although we cannot directly compare SIGMADOCK to co-folding methods, we show competitive performance relative to AF3 with a fraction of the training data and lower test-train leakage (Table 4). We leave a more detailed comparison in Appendix J.2. Table 3: Sensitivity to pocket definition (PB
set). Pocket diameter ∅pocket is the maximum pairwise distance between Cα's across the selected Nres residues; we report dataset means.

Table 4: Per-sequence-similarity comparison between SIGMADOCK (left) and AF3 (right) on the PB set. Values for AF3 are extracted from Extended Data 4c (Abramson et al., 2024).

| Metric || d0(A) ˚   | 4         | 5           | 6    | 7    |               |       |         |
|---------------------|-----------|-------------|------|------|---------------|-------|---------|
| ∅pocket (A)         | 20.9      | 22.4        | 24.2 | 26.2 |               |       |         |
| ˚                   |           |             |      |      |               |       |         |
| Nres                | 15.6      | 20.3        | 26.5 | 37.8 |               |       |         |
| RMSD < 2            | 80.5      | 81.5        | 78.3 | 69.8 |               |       |         |
| PB-Val.             | 80.2      | 80.5        | 77.3 | 68.2 | Seq. Sim. (%) | Count | PB-Val. |
| [0, 30)             | 109 | 38  | 72 | 87     |      |      |               |       |         |
| [30, 95)            | 76 | 83   | 79 | 82     |      |      |               |       |         |
| [95, 100]           | 123 | 187 | 87 | 78     |      |      |               |       |         |
| Total / Avg.        | 308 | 308 | 79.9 | 80.2 |      |      |               |       |         |

We refer the reader to Appendix I for extended results, and Appendix J for a detailed discussion on the current limitations of our method and future work.

## 4 Conclusion

We believe SIGMADOCK represents a major step forward in the reliability and feasibility of deep learning as a promising tool for accelerating drug discovery. Moving away from torsional parametrisation, our proposed SE(3)m fragment-space Riemannian diffusion model is, to our knowledge, the first generative method trained on the intended PB split to surpass classical dockers on the redocking task. We extensively lay out the key components of our framework in the Appendices and open-source our codebase to proliferate reproducibility and further development, and we view extensions to flexible-receptor docking and co-folding as natural next steps to make SIGMADOCK a more general and practical tool. We demonstrate the critical role of principled inductive biases in enabling superior generalisation and data efficiency, and hope our work encourages rethinking progress on geometry and conditioning, as opposed to relying on scale alone (Abramson et al., 2024).

10Here we define a failure if the majority of samples generated across N seeds have an RMSD above 2A. ˚

## Author Contributions

As the main contributor, A.P. instigated and led the formulation, development, and analysis of SIG- MADOCK. A.P and L.Z. both developed the mathematical framework, proofs, and the key components of SIGMADOCK. A.P and L.Z. organised and wrote the manuscript. G.M.M, C.M.D and Y.W.T supervised the project, partaking in relevant scientific discussions and proof-reading the manuscript.

## Acknowledgments

AP is funded by EPSRC and AstraZeneca via an iCASE award for a DPhil in Machine Learning. LZ is supported by the EPSRC CDT in Modern Statistics and Statistical Machine Learning (EP/S023151/1). AP thanks Kathryn Giblin for insightful scientific discussions, as well as Jochem Nelen and other PhD students in the OPIG and OxCSML groups for valuable exchanges. AP also thanks Yasmin Baba for her unwavering support and for compelling him to take a rare break from research, during which the core idea of this work emerged. LZ thanks Jessica Harrison for helpful witticisms in informing (subliminally) our work's moniker. The authors declare no competing interests.

## Reproducibility Statement & Code Availability

All training and evaluation data used in this study are publicly available. The full codebase for training, sampling, and evaluation used to generate the reported results will be released open source upon publication at github.com/alvaroprat97/sigmadock.

## References

Josh Abramson, Jonas Adler, Jack Dunger, Richard Evans, Tim Green, Alexander Pritzel, Olaf Ronneberger, Lindsay Willmore, Andrew J Ballard, Joshua Bambrick, et al. Accurate structure prediction of biomolecular interactions with alphafold 3. *Nature*, 630(8016):493–500, 2024.

Eric Alcaide, Ziyao Li, Hang Zheng, Zhifeng Gao, and Guolin Ke. Umd-fit: Generating realistic ligand conformations for distance-based deep docking models. In NeurIPS 2023 Generative AI
and Biology (GenBio) Workshop, 2023.

Eric Alcaide, Zhifeng Gao, Guolin Ke, Yaqi Li, Linfeng Zhang, Hang Zheng, and Gengmo Zhou.

Uni-mol docking v2: Towards realistic and accurate binding pose prediction. In International Conference on Artificial Neural Networks, pp. 34–41. Springer, 2025.

Benjamin Bloem-Reddy and Yee Whye Teh. Probabilistic symmetries and invariant neural networks.

Journal of Machine Learning Research, 21(90):1–61, 2020.

Jacques Boitreaud, Jack Dent, Matthew McPartlon, Joshua Meier, Vinicius Reis, Alex Rogozhnikov, and Kevin Wu. Chai-1: Decoding the molecular interactions of life. *BioRxiv*, 2024.

Simon Boothroyd, Pavan Kumar Behara, Owen C Madin, David F Hahn, Hyesu Jang, Vytautas Gapsys, Jeffrey R Wagner, Joshua T Horton, David L Dotson, Matthew W Thompson, et al. Development and benchmarking of open force field 2.0. 0: the sage small molecule force field. Journal of chemical theory and computation, 19(11):3251–3275, 2023.

Martin Buttenschoen, Garrett M Morris, and Charlotte M Deane. Posebusters: Ai-based docking methods fail to generate physically valid poses or generalise to novel sequences. Chemical Science, 15(9):3130–3139, 2024.

Duanhua Cao, Mingan Chen, Runze Zhang, Zhaokun Wang, Manlin Huang, Jie Yu, Xinyu Jiang, Zhehuan Fan, Wei Zhang, Hao Zhou, et al. Surfdock is a surface-informed diffusion generative model for reliable and accurate protein–ligand complex prediction. *Nature Methods*, 22(2):310– 322, 2025.

Peter A. Cock, Tiago Antao, Jeffrey T. Chang, Brad A. Chapman, Cymon J. Cox, Andrew Dalke, Iddo Friedberg, Thomas Hamelryck, Fred Kauff, Bartek Wilczynski, and Michiel J. L. de Hoon. Biopython: freely available python tools for computational molecular biology and bioinformatics. Bioinformatics, 25(11):1422–1423, 2009. doi: 10.1093/bioinformatics/btp163.

Rob Cornish. Stochastic neural network symmetrisation in markov categories. arXiv preprint arXiv:2406.11814, 2024.

Gabriele Corso, Hannes Stark, Bowen Jing, Regina Barzilay, and Tommi Jaakkola. Diffdock: Dif- ¨
fusion steps, twists, and turns for molecular docking. *arXiv preprint arXiv:2210.01776*, 2022.

Gabriele Corso, Arthur Deng, Benjamin Fry, Nicholas Polizzi, Regina Barzilay, and Tommi Jaakkola. Deep confident steps to new pockets: Strategies for docking generalization. *ArXiv*, pp. arXiv–2402, 2024.

Valentin De Bortoli, Emile Mathieu, Michael Hutchinson, James Thornton, Yee Whye Teh, and Arnaud Doucet. Riemannian score-based generative modelling. Advances in neural information processing systems, 35:2406–2422, 2022.

Peter Eastman, Jason Swails, John D Chodera, Robert T McGibbon, Yutong Zhao, Kyle A
Beauchamp, Lee-Ping Wang, Andrew C Simmonett, Matthew P Harrigan, Chaya D Stern, et al.

Openmm 7: Rapid development of high performance algorithms for molecular dynamics. PLoS computational biology, 13(7):e1005659, 2017.

Wenhao Gao, Shitong Luo, and Connor W Coley. Generative artificial intelligence for navigating synthesizable chemical space. *arXiv preprint arXiv:2410.03494*, 2024.

Jiaqi Guan, Xingang Peng, PeiQi Jiang, Yunan Luo, Jian Peng, and Jianzhu Ma. Linkernet: Fragment poses and linker co-design with 3d equivariant diffusion. *Advances in Neural Information* Processing Systems, 36:77503–77519, 2023.

Thomas A Halgren, Robert B Murphy, Richard A Friesner, Hege S Beard, Leah L Frye, W Thomas Pollard, and Jay L Banks. Glide: a new approach for rapid, accurate docking and scoring. 2. enrichment factors in database screening. *Journal of medicinal chemistry*, 47(7):1750–1759, 2004.

Charles Harris, Kieran Didi, Arian R. Jamasb, Chaitanya K. Joshi, Simon V. Mathis, Pietro Lio, and Tom Blundell. Benchmarking generated poses: How rational is structure-based drug design with generative models? *ArXiV*, 2023.

Michael J. Hartshorn, Marcel L. Verdonk, Gianni Chessari, Suzanne C. Brewerton, Wijnand T. M.

Mooij, Paul N. Mortenson, and Christopher W. Murray. Diverse, high-quality test set for the validation of protein−ligand docking performance. *Journal of Medicinal Chemistry*, 50(4):726– 741, 2007. doi: 10.1021/jm061277y. PMID: 17300160.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

Emiel Hoogeboom, Vıctor Garcia Satorras, Clement Vignac, and Max Welling. Equivariant diffu- ´
sion for molecule generation in 3d. In *International conference on machine learning*, pp. 8867– 8887. PMLR, 2022.

Chin-Wei Huang, Milad Aghajohari, Joey Bose, Prakash Panangaden, and Aaron C Courville. Riemannian diffusion models. *Advances in Neural Information Processing Systems*, 35:2750–2761, 2022.

Yufei Huang, Odin Zhang, Lirong Wu, Cheng Tan, Haitao Lin, Zhangyang Gao, Siyuan Li, Stan Li, et al. Re-dock: towards flexible and realistic molecular docking with diffusion bridge. *arXiv* preprint arXiv:2402.11459, 2024.

Yize Jiang, Xinze Li, Yuanyuan Zhang, Jin Han, Youjun Xu, Ayush Pandit, Zaixi Zhang, Mengdi Wang, Mengyang Wang, Chong Liu, Guang Yang, Yejin Choi, Wu-Jun Li, Tianfan Fu, Fang Wu, and Junhong Liu. Posex: Ai defeats physics approaches on protein-ligand cross docking, 2025. URL https://arxiv.org/abs/2505.01700.