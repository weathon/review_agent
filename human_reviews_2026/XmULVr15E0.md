# MAVEN: A Mesh-Aware Volumetric Encoding Network for Simulating 3D Flexible Deformation

- Decision: Accept (Poster)
- Scores: 6, 6, 2, 6

## Abstract
Deep learning-based approaches, particularly graph neural networks (GNNs), have gained prominence in simulating flexible deformations and contacts of solids, due to their ability to handle unstructured physical fields and nonlinear regression on graph structures. However, existing GNNs commonly represent meshes with graphs built solely from vertices and edges. These approaches tend to overlook higher-dimensional spatial features, e.g., 2D facets and 3D cells, from the original geometry. As a result, it is challenging to accurately capture boundary representations and volumetric characteristics, though this information is critically important for modeling contact interactions and internal physical quantity propagation, particularly under sparse mesh discretization. In this paper, we introduce MAVEN, a \textbf{m}esh-\textbf{a}ware \textbf{v}olumetric \textbf{e}ncoding \textbf{n}etwork for simulating 3D flexible deformation, which explicitly models geometric mesh elements of higher dimension to achieve a more accurate and natural physical simulation. MAVEN establishes learnable mappings among 3D cells, 2D facets, and vertices, enabling flexible mutual transformations. Explicit geometric features are incorporated into the model to alleviate the burden of implicitly learning geometric patterns. Experimental results show that MAVEN consistently achieves state-of-the-art performance across established datasets and a novel metal stretch-bending task featuring large deformations and prolonged contacts.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
MAVEN explicitly models high-dimension geometric mesh elements for physical simulation.

**I am not an expert in this field, so an accurate evaluation is difficult for me. I will defer to the assessment of other reviewers.**

### Strengths
- It is important to utilize more diverse 3D information when performing mesh-based simulation.

- The work demonstrates both accuracy and computational efficiency; despite using more information, the computational efficiency did not significantly degrade.

### Weaknesses
- As the authors themselves mentioned in the Related Works section, existing studies have also used high-dimensional information such as Cell.

### Questions
See Weakness section

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The proposed paper, MAVEN, presents a graph neural network (GNN) model for simulating physics, particularly flexible solid deformations, including contacts. The key contribution of the work is the incorporation of additional explicit geometric features (cells and faces) into mesh graph network line of simulators, which previously considered only vertices and edges. This design reduces the burden of implicitly learning geometric patterns, enabling the network to capture 3D spatial structure and behaviour better.  

The authors describe how to integrate these higher-order geometric entities using geometric aggregators and disaggregators, along with additional message passing on cell–facet graphs. MAVEN is compared against MeshGraphNets and several follow-up models in this category. It consistently achieves state-of-the-art performance across a few established datasets and on a new metal stretch–bending benchmark that features large deformations and sparse meshes.

### Strengths
1.	The core idea that incorporating higher-dimensional elements, such as 2D facets and 3D cells, enables better geometric representation of volumetric solids is a strong conceptual insight. 

2.	The paper carefully explains how to integrate these geometric features, introducing geometric aggregators/disaggregators and modified message-passing schemes. The authors analyse design choices such as averaging versus learning coefficients in a local coordinate system and demonstrate where this new model is most beneficial (for coarse mesh discretisation and contact scenarios). Overall, it extends the expressive capacity of GNNs for physical simulation without much computational cost.

3.	The draft is clearly written. It systematically and thoroughly investigates different aspects of the idea. Considering that GNN-based simulation has received qwuite a bit of traction, this work provides nice insights, valuable to this community.

### Weaknesses
1.	It is unclear whether MAVEN can handle beyond volumetric solids. MeshGraphNets can model a wider range of systems, such as cloth (thin shell) or fluid dynamics, whereas MAVEN seems focused only on deformable solids. It would be useful to clarify whether MAVEN can handle surface-based or thin-shell geometries (such as cloth) and whether a version using only faces, without volumetric cells, is feasible. This would also make the comparison with surface-based GNNs such as HOOD fairer. 

2.	The discussion focuses exclusively on Lagrangian (mesh-based) systems. The authors should comment on how MAVEN could extend to or differ from Eulerian formulations. Eulerian systems are explored in the original MeshGraphNet and its variants, but MAVEN's volumetric encoding will prevent GNN simulation in such scenarios. 

3.	The paper would benefit from positioning the approach relative to Physics-Informed Neural Networks (PINNs), see [Karniadakis et al.]

[Karniadakis et al.] - Karniadakis, George Em, et al. "Physics-informed machine learning." Nature Reviews Physics 3.6 (2021): 422-440.

Minor comments and suggestions

1.	The term *flexible deformations* (used in the title and abstract) is somewhat vague. Consider using elastic or soft-body deformations, which more accurately describe 3D deformable solids.  

2.	The overview figure’s caption could be expanded to explain the method flow better.

3.	There are a few typos:  
a.	L292 likely refers to a geometric aggregator rather than an encoder.  b.	In Figure 4, the label GT in the top-right should probably read FIG.

### Questions
1.	How can MAVEN handle surface meshes, such as thin shells or 2D manifolds, without volume or Eulerian systems? Could the model be extended to such cases?  

2.	Around L285/L318, the paper states: “To ensure translational and permutation invariance, we sort the vertices of each facet by their distances to the facet centroid, thereby enforcing a unique representation.” It is not clear how this sorting leads to invariance. Please provide a detailed justification. 

3.	What are the optimisable parameters in the method? In Section 3.5, a list of additional optimisable parameters or MLPs compared to the baseline method (MGN) could be added.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a model that integrates volumetric geometric information into a GNN architecture to handle dynamic deformation simulation of meshes. Instead of using only vertices, as in conventional methods, the model integrates geometric features obtained from facets and cells and proposes a method for processing these elements. By incorporating this geometric information, the model demonstrated the advantage of achieving good performance in sparse meshes compared to other baselines.

### Strengths
I agree that the existing baselines failed to adequately capture the volumetric geometric information of the mesh. It is a strength that they proposed an architecture capable of containing all information that can be extracted from all possible constituent elements of the mesh: vertex, facet, and cell.

The architecture, designed to be applicable regardless of whether the mesh is tetrahedral or hexahedral, is a good foundation for generalization.

### Weaknesses
1. The biggest concern is that the computational cost for calculating the surface area, volume, and perimeter of cells and facet at every step seems to be vast for a fine mesh. Please provide the computational cost of the process of calculating geometric information based on the number of nodes and edges.

2. It was stated that facets within the search radius r are found using BVH when searching for contact surfaces. If this is the case, there is a need to specifically explain how the problem illustrated in Figure 1(d) is solved even in a sparse mesh. This seems to be the biggest differentiator from node-based methods, but I do not clearly understand how it is resolved.

3. If we assume that all facets in a sparse mesh within the radius r are in contact, a different problem from node-based methods might arise: since the model recognizes contact even when the facets overlap only slightly, wouldn't this lead to the issue of perceiving a wider contact area than the actual contact surface? A case study on how MAVEN resolves the issue shown in Figure 1(d) seems necessary.

4. The ablation study is conducted at too coarse a level, making it difficult to figure out what is effective for the model's performance. That is, it is difficult to clearly confirm whether the proposed model performs better because it actually incorporates more geometric information. Is the complex aggregator utilized superior to a simpler method of merely averaging and including the surrounding cell and facet information in the node features?

5. Long-range interaction is one of the important problems for GNN-based simulators. However, this methodology does not seem to include a discussion on that aspect.

### Questions
See Weaknesses.

### Soundness
3

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
4

### Summary
This paper proposes a mesh-aware volumetric encoding network to predict physically meaningful 3D deformation. The paper shows better performance compared with alternative baseline models.

### Strengths
The overall idea of encoding meshes at different levels is plausible.

The performance improvements compared with existing methods.

### Weaknesses
The deformation evaluated is somewhat limited. For more diverse materials and object shapes, it would be useful to show the generalizability. 

The proposed method essentially relies on volumetric information, but it is unclear whether the compared methods are surface or volume based. This leads to questions regarding whether the evaluation is fair.

### Questions
As all the examples shown have limited variations for the geometry, does the method generalize to more flexible shapes?

### Soundness
3

### Presentation
3

### Contribution
2
