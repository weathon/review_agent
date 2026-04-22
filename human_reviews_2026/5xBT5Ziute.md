# MatRIS: Toward Reliable and Efficient Pretrained Machine Learning Interatomic Potentials

- Avg Score: 4.67
- Decision: Accept (Poster)
- Scores: 6, 4, 4

## Abstract
Foundation MLIPs demonstrate broad applicability across diverse material systems and have emerged as a powerful and transformative paradigm in chemical and computational materials science. Equivariant MLIPs achieve state-of-the-art accuracy in a wide range of benchmarks by incorporating equivariant inductive bias. However, the reliance on tensor products and high-degree representations makes them computationally costly. This raises a fundamental question: as quantum mechanical-based datasets continue to expand, can we develop a more compact model to thoroughly exploit high-dimensional atomic interactions? 
In this work, we present MatRIS (\textbf{Mat}erials \textbf{R}epresentation and \textbf{I}nteraction \textbf{S}imulation), an invariant MLIP that introduces attention-based modeling of three-body interactions. MatRIS leverages a novel separable attention mechanism with linear complexity $O(N)$, enabling both scalability and expressiveness. MatRIS delivers accuracy comparable to that of leading equivariant models on a wide range of popular benchmarks (Matbench-Discovery, MatPES, MDR phonon, Molecular dataset, etc). Taking Matbench-Discovery as an example, MatRIS achieves an F1 score of up to 0.847 and attains comparable accuracy at a lower training cost. The work indicates that our carefully designed invariant models can match or exceed the accuracy of equivariant models at a fraction of the cost, shedding light on the development of accurate and efficient MLIPs.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work proposes MatRIS, a novel E(3)-invariant model architecture for machine learning interatomic potentials. They adopt dim-wise attention and separable attention for line graphs to encode three-body features. The model outperforms previous SOTAs on Matbench-Discovery, DPA2, and several molecular datasets in the zero-shot setting, performs on par or slightly better than others on MatCalc.

### Strengths
1. Using self-attention in the line graph to encode three-body information is novel?, sound, and simple.
2. The separable attention is also with reasonable motivation.
3. The method performs well in several benchmarks.
4. The experiments are sufficient.

### Weaknesses
1. Although I assume the methodology is novel, some evidences need to be given with respect to the claim in L94-95: ``$\cdots$, our model is the first to leverage attention mechanisms for modeling three-body interaction''. Specifically, the authors should justify its differences to VisNet [1], FreeCG [2], and MGT [3], which all adopted self-attention mechanisms and captured up to four-body interactions. Is that the case that your model uses self-attention to encode the many-body interactions while they encode many-body interactions without using self-attention in this specific step, but using self-attention to refine the features that have already been encoded? A detailed methodological comparison can be added to the appendix.

2. It appears to be a very rushed paper. Many typographical and presentational errors are spotted in it. I show here some of them:

a) Missing space before almost half of the left parenthesis;

b) Misusing \citep and \cite in many sentences, e.g., ``GemNet Gasteiger et al. (2024) incorporated dihedral angles, and SphereNet Liu et al. (2022) further integrated torsional information. '' in L118-119.

c) Table X or Tab. X. No Table.X, e.g., ``More details on hyper-parameters are given in Table.7 in appendix. Moreover, inspired by (Zaidi et al., 2022; Liao et al., 2024a),'' in L268-269.

d) Omitting math mode, e.g., ``In equivarant models, the predominant strategy for enforcing higher-order equivariance relies on computationally intensive tensor products of rotation order L(Batzner et al., 2022; Batatia et al., 2023; Liao & Smidt, 2023; Liao et al., 2024b).'' in L62-64;

e) The value vector $v$ should be explained in the main text;

f) In Eq. 1, the $j$ is for feature dimension and $i$ for atom index. But in the adjacent context, $i$ and $j$ both represent atom index.

The typographical and presentational errors are too massive to list them all out here. I sincerely recommend the authors go through a complete writing check. I am not that kind of person who rejects a paper solely based on presentations, but the authors could pay more attention to such detail, or it would make a good work fail for such things instead of the methodology itself. It is not worth it for you.

[1] Wang Y, Wang T, Li S, et al. Enhancing geometric representations for molecules with equivariant vector-scalar interactive message passing[J]. Nature Communications, 2024, 15(1): 313.

[2] Shao S, Geng H, Wang Z, et al. FreeCG: Free the Design Space of Clebsch-Gordan Transform for Machine Learning Force Fields[J]. arXiv preprint arXiv:2407.02263, 2024.

[3] Anselmi M, Slabaugh G, Crespo-Otero R, et al. Molecular graph transformer: stepping beyond ALIGNN into long-range interactions[J]. Digital Discovery, 2024, 3(5): 1048-1057.

### Questions
1. How can framework be applied to equivariant models? The equivariance, and even the whole invariance will be broken if we manipulate the channel of a geometric tensor individually whose weight higher than 0. Does this framework have the potential to be applied in such scenario?

2. From Eq. 1, the denominator does not contain $x_{ij}$ itself, as it is not in the neighboring list of itself. Is it a typo?

### Soundness
3

### Presentation
1

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
The paper proposes MatRIS, an invariant methodology to construct machine learning interatomic potentials that use three-body attention between edges with a common node. 

MatRIS is benchmarked on a broad range of benchmark datasets, including the Matbench Discovery benchmark (for "compliant" tracking fitting only on the MPTrj datasets), MatCalc, and others. 

Additionally, the paper reports the computational efficiency of the model.

### Strengths
The model, MatRIS, achieves state-of-the-art efficiency of the compliant track of Matbench Discovery with the F1 score of 0.844, and performs equally well on other benchmarks - MatCalc, Molecular Zero Shot Benchmark, and DPA2. 

The paper provides several ablations on the model design and fitting procedure. 

The computational efficiency of the MatRIS model is competitive in both training and inference.

### Weaknesses
The position of the available MLIPs architectures into invariant and equivariant groups is not correct. The paper groups all the models into only two groups, invariant and equivariant, missing the third big group of models - rotationally unconstrained architectures, and assigns many of the rotationally unconstrained architectures into the invariant groups. Because of this, some claims are not entirely true, for instance, "Recent studies (Neumann et al., 2024; Qu & Krishnapriyan, 2024; Rhodes et al., 2025) show that invariant models allow more flexibility in exploring the potential energy surface while being computationally more efficient." The cited papers are unconstrained, not invariant, and thus the claim that ' invariant models allow more flexibility' is hardly correct. 

Invariant models are those where all the intermediate representations are invariant with respect to rotations - they are not changed at all if a system is rotated. Example - SchNet. 

Equivariant models are those where intermediate representations are invariant or covariant, which means that they transform in a specific manner with respect to the rotation of the input system, such as vectors, Cartesian tensors of higher rank, or spherical tensors. 

Unconstrained models are those where there is no restriction on intermediate representations, and they can be learned to transform arbitrarily with respect to rotations. 

In general, invariant models are the least expressive and fastest; equivariant models are more expressive and more computationally demanding because of the associated tensor products, and unconstrained models are most expressive and simultaneously fast, roughly as invariant ones, but produce not rotationally invariant predictions, which can sometimes be a problem in molecular dynamics simulations. 

The Edge-based design, while claimed to be new, has already been proposed in the community. Examples are Atomistic Line Graph Neural Network, Point Edge Transformer, and many others. 

Similarly, there are multiple attention-based architectures with O(N) scaling that apply attention to local neighborhoods.

### Questions
Could you clarify the discussion in light of arranging models into invariant, covariant, and unconstrained ones?

Could you provide more insights into the reasons why MatRIS is so computationally efficient?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper introduces MatRIS, a new invariant universal machine learning interatomic potential that achieves high accuracy at a lower computational cost than existing equivariant models. Instead of relying on tensor operations to enforce symmetry, MatRIS uses an attention-based architecture that explicitly models three-body interactions through a line graph and a separable linear attention mechanism with O(N) complexity. MatRIS matches or outperforms state-of-the-art models on benchmarks like Matbench-Discovery and MatPES. Ablation studies confirm that components like dimwise softmax, separable attention, and learnable envelopes significantly improve performance.

### Strengths
1. The architecture is new. The authors introduce a new architectural approach by explicitly modeling three-body interactions using a line graph attention mechanism. Each edge in the atom graph becomes a node in the line graph, enabling the model to attend over bond angles. This is claimed to be the first use of attention for three-body interactions in ML potentials, extending beyond traditional message-passing that usually considers only pairwise interactions.
2. They employ a separable attention mechanism that splits the message-passing into two parallel branches: one capturing how each neighbor affects a central atom, and another for how the central atom influences its neighbors. Also, this "attention" is implemented with linear complexity in the number of atoms.
3. The proposed model demonstrates competitive or superior results on a few benchmarks.

### Weaknesses
1. The evaluation on large-scale datasets is limited. A notable gap in the experiments is the absence of results on the latest massive and diverse datasets, specifically Open Materials 2024 (OMat 24) and Open Molecules 2025 (OMol 25). If the model is claimed to be scalable, then it should perform well on these datasets. At least a benchmark on OMol 4M would be nice to have.
2. Despite the terminology, MatRIS’s mechanism differs from standard scaled dot-product self-attention with Q/K/V projections and global token mixing. Weights are computed per-dimension over local neighbors (a “dim-wise softmax”) and derived by applying a linear map to edge features, and followed by separate target/source softmax normalizations. In other words, the “attention” acts more like edge-conditioned, per-channel neighbor weighting than full QKV attention, and it is local (achieving linear O(N) complexity by avoiding all-pairs interactions). This design brings efficiency but may limit the capacity for global, content-based mixing compared to traditional Transformer attention.
3. MatRIS is built on local, cutoff-based neighborhoods (pairwise and three-body/angle graphs), which means long-range electrostatics and other delocalized effects are not explicitly modeled or evaluated. To demonstrate robustness in regimes where long-range interactions matter (e.g., biomolecules, electrolytes), it would be great to test on OMol 25 and report results on biomolecule / electrolyte validation splits.
4. The paper frames MatRIS as a universal MLIP, but the core pretraining and evaluations are materials-centric: Matbench trained on MPTrj, MatCalc using MatPES/MPTrj, and results on DPA2 materials test sets. While the abstract highlights “universal” scope and mentions a “Molecular dataset”, the molecular side is limited to small zero-shot style checks rather than large-scale molecular pretraining/evaluation. And there's no mention on other types of chemical systems, such as surface, organic crystals, polymers, MOFs, etc.

### Questions
1. What unique signals does the source/target split capture that a single asymmetric aggregator cannot? A controlled ablation that matches parameters and FLOPs would be nice (or does your ablation matched parameters and FLOPs?)
2. In low-data regimes, how does MatRIS compare to strong equivariant baselines?
3. Since forces/stress are energy-conservative, how does MatRIS behave in long NVE MD runs (energy drift, thermostatted stability)?

### Soundness
3

### Presentation
3

### Contribution
3
