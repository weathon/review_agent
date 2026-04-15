# Incorporating gauge-invariance in equivariant networks

- Decision: Reject
- Scores: 3, 6, 3, 5

## Abstract
Gauge theories, which describe fundamental forces in nature, arise from the principle of locality in physical interactions. These theories are characterized by their invariance under local symmetry transformations and the presence of a gauge field that mediates interactions. While recent works have introduced gauge equivariant neural networks, these models often focus on specific cases like tangent bundles or quotient spaces, limiting their applicability to the diverse gauge theories in physics. We propose a novel architecture for learning general gauge invariant quantities by explicitly modeling the gauge field in the context of graph neural networks. Our framework fills a critical gap in the existing literature by providing a general recipe for gauge invariance without restrictions on the fiber spaces. This approach allows for the modeling of more complex gauge theories, such as those with $SU(N)$ gauge groups, which are prevalent in particle physics. We evaluate our method on classical physical systems, including the XY model on various curved geometries, demonstrating its ability to capture gauge invariant properties in settings where existing equivariant architectures fall short. Our work takes a significant step towards bridging the gap between gauge theories in physics and equivariant neural network architectures, opening new avenues for applying machine learning to fundamental physical problems.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
After introducing foundational concepts such as fiber bundles, connections, gauge equivariance, and the XY model, the presented proposes GaugeNet, a graph neural network architecture specifically designed to manage gauge-invariant tasks by addressing local symmetries, which conventional equivariant models handle only at a global scale. Additionally, the authors propose a synthetic dataset designed for energy regression and vortex counting tasks on the XY model across different topologies. Finally, GaugeNet is evaluated on the proposed dataset using energy regression for toroidal topology and vortex counting for spherical topology.

### Strengths
* Highlights an important limitation of current equivariant methods that restrict their ability to model tasks with local symmetries.

### Weaknesses
- **Reproducibility Concerns:** The training procedures are insufficiently described, with no details provided on the dataset split for training, validation, and test sets, nor on the loss function, learning rate, or optimizer. Furthermore, I believe it is necessary to include an explanation of how the baselines were implemented for the given tasks or how they process the XY model configurations.
-  **Lack of Empirical Support for Comparison Claims:** The third contribution (L104) suggests that the authors apply their approach to classical physical systems to showcase the ability to capture gauge invariance, addressing properties that previous models do not account for (Chen et al., 2022; Bogatskiy et al., 2020). However, I believe it is necessary to validate this claim with empirical evidence by presenting an experimental comparison with these models.
- **Generalization Claim Not Supported:** The first contribution (L98) claims that the proposed model handles arbitrary fiber bundles and generalizes previous models restricted to principal bundles. However, all examples and the computational model presented are limited to vector bundles, which are a specific case of principal bundles, contradicting the claim. As an example, at L328 the gauge field $A$ is defined as a function mapping to linear endomorphisms of fibers. This supposes that the fiber is a vector space, hence the $\mathcal{E}$ is a vector bundle. To corroborate this claim and validate its effectiveness, I believe it is necessary to provide examples and experimental evidence on GaugeNet models defined on non-principal bundles.
- **Unclear and unpolished writing (No Impact on Recommendation):**

  I suggest that the authors carefully revise the text, as it contains numerous typos or inaccuracies. For example:

    - L394: "which is..."
    - L343: ".A"
    - L437: The text ends abruptly.
    - Fig 2a: "MLP" should be replaced by "linear layer" or similar.
    - L212: Group $G$ does not necessarily need to act freely on homogeneous spaces.

  In certain instances, the tone and language seem unsuited for academic purposes. For example:

    - L274, L295: "let us" should be used instead of the colloquial "let's".
    - L103: The sentence "Application of our approach to classical physical systems, including the XY model" is misleading since only the XY model is considered.

  The paper structure could be optimized by streamlining certain paragraphs and clarifying cross-references. For example:

    - L294-313: The discussion on curvature can be minimized, as it’s only briefly mentioned in L365-367 without further use.
    - L264: Unclear reference to "original text."

### Questions
1. What are the boundary conditions introduced at L393? Additionally, what is meant by "isomorphic" in the next line?
2. In L345-360, the gauge-equivariant message-passing scheme involves learnable functions $φ$ and $ψ$ (L349, L359). How are these functions constructed, and how do they respect gauge equivariance, in particular for bivector features?
3. Why was energy regression tested only on grid topology and vorticity classification only on spherical topology?
4. Why was ResNet18 chosen as a baseline, given that it is a model designed for image processing—a class of tasks quite different from the one it is trained on in this paper?
5. At L240, what does the term "embedding" in the paragraph title refer to? Is it related to local trivialization?

### Soundness
1

### Presentation
1

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
This paper introduces a novel approach to incorporating gauge invariance in equivariant neural networks for physical systems. The authors develop a general architecture that explicitly models gauge fields within graph neural networks, allowing for the handling of arbitrary fiber bundles rather than being restricted to specific cases like tangent bundles. The method is evaluated on the XY model across different geometries (flat, torus, sphere), demonstrating superior performance in energy estimation and vortex classification tasks compared to baseline models.

### Strengths
1. The idea is novel and the authors provide a clear explanation of connections to existing works.
2. The paper provides a comprehensive theoretical foundation, carefully connecting differential geometry concepts to practical neural network implementation.
3. The model achieves loss ~10^-4 with only ~10^3 parameters, while ResNet requires ~10^7 parameters for loss ~10^-2 (Fig 2b). This represents a 100x improvement in accuracy with 10000x fewer parameters.

### Weaknesses
1. In its current form, the paper can be hard for non-physicists to decipher. It would be preferable to focus more on the intuitions in the main text and move some technical discussion into the appendix.
2. Although impressive, the model is only evaluated on the classical XY model. It would be interesting to explore other classical and quantum models.

### Questions
1. How does the choice of graph discretization affect the gauge field representation, particularly for curved manifolds like spheres?
2. How can the model be applied to quantum systems?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper presents a generalized formulation of gauge equivariant networks that explicitly represents the gauge field, and works for arbitrary fiber bundles. The paper covers all the necessary background on gauge theory and fiber bundles, and then defines a GNN-based gauge equivariant architecture. The basic idea is to include an explicit gauge field as a map from edges to linear transformations of the fiber, and to use this to compute gauge in/equivariant features such as invariant inned products or bivector features. The paper then studies two application, energy estimation and vorticity estimation, and shows significant gains over non-gauge-equivariant methods.

### Strengths
+ The paper carefully explains background material in an accessible manner.
+ Well motivated
+ Somewhat novel formulation / architecture

### Weaknesses
- The paper lacks references to key papers, such as: "Gauge covariant neural network for quarks and gluons, Yuki Nagai & Akio Tomiya", "Equivariant Flow-Based Sampling for Lattice Gauge Theory, Kanwar et al.", "Learning lattice quantum field theories with equivariant continuous flows, Gerdes et al."

- The paper claims that the gauge field absent in earlier work. but this is not quite true. The works that use the tangent bundle generally make implicit use of the Levi-Civita connection. Other gauge equivariant networks exist that also use the connection before comparing features in different fibers (see e.g. the references above).

- Generally, even if some other works are specialized to particular kinds of fiber bundles, the paper does not present a truly novel idea for generalizing to arbitrary bundles. The paper simply takes the general idea of transporting features between fibers using a connection, and then says that one can do this in general (how to do this is standard stuff in the theory of fiber bundles).

- For most of the paper, it is not clear where the gauge field comes from. In eq. 20 it just says that A is a map from edges to fiber maps. Given that this is not an infinitesimal transformation, should it be computed by integrating or exponentiating? More generally, do I understand correctly that the gauge field should be derived from known theory? (I had hoped that the gauge field could be learned or computed as a feature)

- The paper does not do any ablations on the architecture, and compares to weak baselines without gauge equivariance. With the current evidence we can't tell if the particular features proposed in this work are really very effective or not.

### Questions
n/a

### Soundness
4

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
2

### Summary
The paper presents GaugeNet, an architecture designed to learn general gauge-invariant quantities by explicitly modeling the gauge field within a graph neural network framework. GaugeNet addresses the limitation of existing gauge-equivariant neural networks that are often constrained to specific cases, such as tangent bundles or quotient spaces, limiting broader applications to diverse gauge theories in physics. The authors validate GaugeNet on classical physical systems, such as the XY model on various curved geometries, demonstrating its effectiveness in capturing gauge-invariant properties.

### Strengths
* The approach is well-motivated, as developing a gauge-equivariant neural network adaptable to various fiber spaces is essential for capturing the full scope of gauge theories in physics. By explicitly modeling the gauge field, GaugeNet represents a meaningful step in this direction.
* GaugeNet shows excellent performance in handling gauge invariance in the XY model compared to conventional non-equivariant and equivariant neural networks, achieving this in a parameter-efficient manner.

### Weaknesses
* The methodology/theory section is challenging to follow, especially for a general machine learning audience. Several methodological aspects could benefit from additional clarification:
     * The connection between energy forms in discrete and continuous physical spaces is somewhat unclear. Specifically, why does the coupling matrix $J$ in Equations 1 and 2 disappear in Equation 4? Could the authors provide a step-by-step derivation of how Equation 4 is obtained from Equations 1 and 2?
    * The logic from lines 227 to 234 is hard to follow and could benefit from more elaboration.  
    * The purpose of the “Curvature from Parallel Transport” section in establishing the gauge-equivariant framework is unclear. Could the authors include a brief explanation of how the "Curvature from Parallel Transport" section relates to the proposed method
    * Are there any variations in the physical forms of the field value  $\mathbf{s}_i$  across different topologies? If so, does GaugeNet need architectural adjustments to accommodate these cases? Additionally, why are all field values represented as vector features instead of higher-order features?
    * What distinguishes $\mathbf{s}_i$  from  $\mathbf{x}_i$ in Equation 23? If $\mathbf{x}_i$ represents vector features of the field, does the term  $\mathbf{x}_i - \mathbf{x}_j$ maintain equivariance to local gauge transformations?
* The comparison of GaugeNet is limited to two global equivariant neural networks, EMLP and EGNN, rather than models that address local equivariant transformations, as mentioned in the literature review (e.g., [1], [2]). This hinders readers' justification for the advantage of GaugeNet over previous methods in this area.

[1] Pim de Haan, Maurice Weiler, Taco Cohen, and Max Welling. Gauge equivariant mesh cnns: Anisotropic convolutions on geometric graphs.
[2] Di Luo, Zhuo Chen, Kaiwen Hu, Zhizhen Zhao, Vera Mikyoung, and Bryan Clark. Gauge-invariant and anyonic-symmetric autoregressive neural network for quantum lattice models

### Questions
* Could the authors provide insight into why EGNN struggled to converge to a reasonable solution in the vortex numbers classification task?
* Possible typos:
    * Line 431: Does the first “equivariant” refer to “non-equivariant”?
    * Line 437: The sentence appears incomplete.

### Soundness
2

### Presentation
1

### Contribution
2
