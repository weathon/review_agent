# AdS-GNN - a Conformally Equivariant Graph Neural Network

- Decision: Accept (Poster)
- Scores: 8, 6, 4

## Abstract
Conformal symmetries, i.e.\ coordinate transformations that preserve angles, play a key role in many fields, including physics, mathematics, computer vision and (geometric) machine learning. Here we build a neural network that is equivariant under general conformal transformations. To achieve this, we lift data from flat Euclidean space to Anti de Sitter (AdS) space. This allows us to exploit a known correspondence between conformal transformations of flat space and isometric transformations on the Anti de Sitter space. We then build upon the fact that such isometric transformations have been extensively studied on general geometries in the geometric deep learning literature. In particular, we employ message-passing layers conditioned on the proper distance, yielding a computationally efficient framework. We validate our model on tasks from computer vision and statistical physics, demonstrating strong performance, improved generalization capacities, and the ability to extract conformal data such as scaling dimensions from the trained network.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces a new architecture, AdS-GNN, which is equivariant under conformal symmetries (preserving angles). 
Conformal symmetries include scaling and a type of inversion, called the special conformal transformation, in addition to rotation and translations. 
The key idea is to first map the data onto an Anti-de Sitter space, a uniform negatively curved space. The reason for this mapping is that at its boundary, AdS has conformal symmetries. The mapping to AdS far from its boundary breaks invariance under the special conformal transformations, but the rest remain intact. This mapping is motivated by the so-called AdS/CFT correspondence. 
The approximate conformal inviariance on AdS corresponds to a certain norm being invariant and that norm is used to define equivariant and invariant features for the GNN. 

They apply AdS-GNN to a few problems and show strong results in physics examples, especially those enjoying some scale invariance.

### Strengths
1. AdS-GNN outperforms other baseline on the physics tasks, such as Ising and N-body electrostatic simulation
2. It performs well even in small data regimes.
3. The idea of first mapping to AdS before applying the model is interesting and inspired by known fact from AdS/CFT. 
4. This is a nice model for equivariant model scale-invariant systems. 
5. The mapping to AdS is fairly easy, though it involves some preprocessing. Other than that, the model seems easy to implement.

### Weaknesses
1. Given the fact that special conformal is broken by the AdS bulk embedding, what is left is scaling + rotation + translation. So, really the model is adding scale equivariance to EGNN. This should be discussed or emphasized. 
2. The choice of experiments isn't explained much. As above, I think AdS-GNN works best for scale-invariant problems. In physics, these are the systems which we expect to have conformal invariance, as discussed in the intro. This may be why on Superpixel MNIST they don't outperform EGNN. 
3. A general discussion of limitations and realm of applicability is needed. 
4. All systems and datasets used in experiments seem very small. It is fine as a proof of concept, but scalability needs to be discussed.

### Questions
1. How does the run time, including AdS embedding, scale with size of the systems? 
2. Do you only use scalars and vectors as features in AdS-GNN? This should work fine for Ising which has i=only quadratic interactions. But what if you had a spin-glass with higher order interactions? Do you think higher tensors would be needed to learn correlations there? Is it known which of those systems, or in what regime they would be scale-invariant? 
3. Any good criteria for deciding when to use AdS-GNN, other than obvious scale-invariance?  
4. Some of the scale-invariant systems here seem fully conformal invariant. How much does the breaking of special conformal cause a problem? How do you measure the effect of this breaking?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper presents a conformally equivariant graph neural network through showing how to lift data from Euclidean space to Anti de Sitter (AdS) space. Distance metrics in AdS space can then be used in conjunction with invariant message passing to create neural networks that are equivariant to the conformal group. Experimental results on computer vision tasks (SuperPixel MNIST and shape segmentation) and physics tasks (2d/3d Ising model and N-body simulation) are presented. In particular, for the physics tasks, AdS-GNN seems to require less data and generalizes better. Additionally, it recovers the correct conformal dimension, illustrating interpretability.

### Strengths
The paper presents an interesting new framework for conformal group equivariance, using ideas from theoretical physics (and building open pre-existing work for constructing group equivariant convolutional layers). It is well-organized with experiments in multiple sub-domains, and provides a self contained introduction to the conformal group. I particularly like the figure at the top of pg. 3 showing possible transformations under the conformal group. Experimental results show that AdS-GNN maintains scale invariance (e.g. SuperPixel MNIST). The physics tasks are good illustrations of the utility of AdS-GNN, showing that AdS-GNN outperforms other models (EGNN and MPNN) for predicting N-point correlation functions for the 2D Ising model and requires less training data. I found it particularly interesting as well that the learned values of the conformal dimensions match the ground truth, and it would be interesting to explore this point further in more complex physics datasets.

### Weaknesses
I am not sure of the usefulness of the image experiments. I wouldn’t expect image datasets to require conformal equivariance/to me the orientation of the image would probably matter the most. However, the AdS-GNN model is invariant, so this orientation is not taken into account. It does not seem to be persuasive that one would use AdS-GNN in a computer vision setting.

I found the part about embedding points in AdS somewhat confusing, and the choice of regular $z_0$ is unclear. I am concerned that this would impact the dataset/break certain symmetries (see questions).  I think further clarification is needed in this section, perhaps an additional figure showing another embedded shape (or embeddings of circles, triangles as in the shape analysis data) would be helpful.

AdS is unable to handle orientation and relies on invariant descriptors. This seems to be a significant limitation, for both image datasets and physics datasets, in light of other work on equivariant neural networks with message passing of higher-order tensorial features (e.g. [1]). 

The shape segmentation task is quite toy, the authors could consider exploring a more realistic dataset such as ModelNet.

Overall, most of the experiments seem to test scale invariance, but there are pre-existing scale invariant models (from my understanding). It may be good to benchmark against these pre-existing scale invariant models and see what conformal invariance actually gives us.

### Questions
What is the computational cost of these models compared to other invariant models (e.g. EGNN)? Does the lifting procedure incur significant computational costs?

Another scientific domain of interest could be fluid dynamics. At sufficiently high Reynolds number, the statistics of turbulent motions in the so-called “inertial range” become universal and exhibit scale invariance properties [2]. I would be interested to see the performance of these models on turbulence modeling tasks.

In what settings would conformal equivariance be useful for images? It seems like it would be more useful in dynamical systems/more-physics motivated problems as shown in the physics task experiments. Is there further motivation for why one would want robustness to conformal transformations in computer vision tasks?

Would it be possible to have non-scalar features in message passing? Were there any experiments done extending the model to include non-scalar features?

How is the regulator $z_0$ chosen, and how does this impact the resulting lifting procedure? Are there other ways that one could lift data into AdS/why was this way in particular chosen? If the lifting procedure breaks certain conformal transformations, which transformations does it break? Figure 3 shows one broken transformation, but what do these “special” transformations correspond to physically, and why would this not be a problem that these are broken? Is it accurate to call the model conformally equivariant if this is the case (maybe approximately conformally equivariant)? Correct me if I'm wrong, but currently it seems conformally invariant rather than equivariant?

Were any comparisons done to pre-existing scale invariant models? This would be interesting to include with comparison to the rotationally/translationally invariant models.

[1] Batatia et al. MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields (2022).

[2] Pope, Stephen. Turbulent Flows. (Cambridge University Press, 2000).

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
3

### Summary
This paper proposes "AdS-GNN," a new graph neural network that is equivariant to conformal transformations (translations, rotations, scale transformations, and special conformal transformations). To achieve this equivariance, the authors, inspired by insights from the AdS/CFT correspondence in physics, introduce a method to lift input data from flat Euclidean space $\mathbb{R}^d$ to Anti-de Sitter (AdS) space $AdS_{d+1}$, which has one additional dimension. On AdS space, the conformal transformations of the original space manifest as isometric transformations (distance-preserving transformations). Therefore, the paper efficiently achieves conformal equivariance by constructing a message-passing GNN that utilizes the proper distance of AdS space as an invariant. The proposed method was evaluated on tasks from computer vision (e.g., SuperPixel MNIST) and statistical physics (e.g., 2D/3D Ising models). Particularly in the physics tasks, it was confirmed to show high generalization performance even under extrapolation (OOD) scenarios, such as scale transformations or changes in system size (number of points). Furthermore, the interpretability of the model was also demonstrated, as it was able to extract a physical universal quantity—the scaling dimension of the Ising model—from the trained network with high precision.

### Strengths
**Novelty of the Idea**: The application of the profound idea of AdS/CFT correspondence from physics to the context of geometric deep learning, thereby constructing a GNN architecture with conformal equivariance, is highly original and commendable.

**High Affinity with Physics Tasks**: Due to its design background, the proposed method is an excellent fit for tasks in statistical physics where conformal symmetry plays a dominant role (e.g., the Ising model near its critical point).

**Excellent Generalization and Interpretability**: In experiments on physics tasks, AdS-GNN demonstrated performance superior to existing equivariant GNNs (like EGNN). Its robustness to extrapolation tasks, such as changing the system size (number of points), is particularly noteworthy. Furthermore, the fact that the model can automatically learn and extract a physically meaningful universal quantity—the scaling dimension—from data and recover its true value with high precision is a testament to the model's high interpretability and a significant contribution.

### Weaknesses
**Lack of Generality and Performance**: The main contribution of this method appears to be limited to physics tasks. As shown by the experimental results (Table 1), in standard image (point cloud) classification tasks like SuperPixel MNIST, the performance does not reach that of existing SOTA methods (e.g., PONITA), and the method's superiority in general-purpose benchmarks has not been demonstrated.

**Limited Applicability**: This method requires prior knowledge that the target data or task possesses "conformal symmetry." Its application to many general machine learning tasks where such strong symmetry does not exist or is unknown is difficult, and the method's utility is inherently restricted.

**Insufficient Appeal to the ICLR Community**: As a result of the above two points, the paper's contribution feels strongly directed primarily at the physics community. There is a lack of discussion or evidence regarding what new possibilities this conformal equivariance approach could bring to the broader ICLR audience (including computer vision, reinforcement learning, natural language processing, etc.).

### Questions
**Regarding generality:** While I understand the method's primary contribution lies in physics tasks, what advantages do the authors believe this conformal equivariance approach offers over existing equivariant GNNs (e.g., those with rotational or translational equivariance) in domains outside of physics, such as general computer vision or robotics? I would also like to ask for the authors' insights into why the method failed to achieve SOTA performance on SuperPixel MNIST. Are there specific reasons to consider, such as the approximation error from the lifting procedure, the nature of the point cloud data, or a potential lack of expressive power in the architecture?

### Soundness
2

### Presentation
3

### Contribution
2
