# Cartan Networks: Group theoretical Hyperbolic Deep Learning

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 2, 6

## Abstract
Hyperbolic deep learning leverages the metric properties of hyperbolic spaces to develop efficient and informative embeddings of hierarchical data. Here, we focus on the solvable group structure of hyperbolic spaces, which follows naturally from their construction as symmetric spaces. This dual nature of Lie groups and Riemannian manifolds allows us to propose a new class of hyperbolic deep learning algorithms where group homomorphisms are interleaved with metric-preserving diffeomorphisms. The resulting algorithms, which we call Cartan networks, show promising results on various benchmark datasets and open the way for a novel class of hyperbolic deep learning architectures, both feedforward and convolutional.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper proposes Cartan Networks, a hyperbolic deep learning architecture that treats hyperbolic space as a solvable Lie group (via its symmetric-space structure) and builds layers by alternating group homomorphisms and isometries. This yields closed-form “linear” layers (mixing Cartan/fiber coordinates) and a hyperbolic logistic/softmax based on geodesic hyperplanes.

### Strengths
1. Clear geometric formulation of isometries and homomorphisms with explicit coordinate formulas
2. The architecture is easy to implement with standard Riemannian optimizers (GeoOpt) and the code is promised.

### Weaknesses
1. Most of what is framed as new (stacking “linear” layers built from isometries + homomorphisms; softmax via hyperbolic hyperplanes) is already implicit or explicit in prior gyrovector/Poincaré constructions and Lie-group views of $H^n$. The paper itself notes that prior hyperbolic layers parametrize all isometries (via exp/log + parallel transport) and that its layer differs mainly by coordinate choice and homomorphism tying. This reads as a repackaging rather than a fundamentally new capability.
2.  Benchmarks are small, with parity on CelebA/CIFAR-10 and a single strong gain on CIFAR-100; there is no task where Euclidean/Poincaré fails but Cartan succeeds, nor graphs/hierarchies where hyperbolic geometry is a priori motivated. This makes the practical utility unclear.
3. Hyperbolic hyperplanes and distance-monotone logits are standard (e.g., Poincaré SVM/softmax). The paper acknowledges they are the same subspaces as prior work; the novelty is just the solvable-coord formula.

### Questions
1. What is truly new? Prior work already realizes isometries via exp/log pipelines, but the paper should prove that this yields strictly larger (or different) function classes than existing hyperbolic “linears,” beyond coordinate convenience.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
1

### Summary
This paper introduces a new type of deep learning architecture named Cartan networks, where hidden features lie on solvable Lie groups and where each layer consists of a homomorphism from the source to the target group followed by an isometry within the target group. This type of network is motivated by a connection between non-compact symmetric spaces (such as hyperbolic space) and solvable Lie groups. The new type of layer is tested in small networks on small datasets.

### Strengths
- The text of the paper seems well written.
- If the new formulation is correct, then it could be an interesting new perspective on hyperbolic learning.
- Again, if the formulation is correct, then automatically being able to generalize this kind of learning to any non-compact symmetric space could be very interesting as well.

### Weaknesses
I want to start by saying that I do not have a sufficiently strong background in Lie group theory for understanding the theoretical part of this paper, which spans most of it. As a result, I am incapable of providing any meaningful review of the theoretical part of the text. 

I do have significant experience with hyperbolic representation learning and computer vision, so I believe that I can judge the experimental part of this paper. There are, in my opinion, a few weaknesses to point out in the experiments:
1. The clearest weakness is the choice of datasets. Currently, the experiments involve a toy dataset, MNIST (and variants), CIFAR-10/100 and CelebA. All of these datasets are considered far too small and simple for drawing any conclusion regarding large scale networks. Therefore, I think some larger experiments (on ImageNet for example) should be included to support the authors' findings.
2. The experiments use small fully connected networks and an AlexNet-like network architecture. In order to see whether this type of layer actually scales to deep neural networks, the experiments should at least include something like a ResNet architecture in my opinion.
3. The layer is compared to the HNN layer in the Poincaré ball model by (Ganea et al., 2018) and to the standard Euclidean layer. However, as mentioned by the authors in the previous literature section, (Shimizu et al., 2021) have proposed an updated variant of the Poincaré layer and (Chen et al., 2022) have also proposed a (geometrically different) hyperboloid layer. A proper comparison should include at least both of these networks. ResNet-like architectures involving these layers already exist (Bdeir et al., 2024, [1]) and could easily be compared against as well.
4. The authors mention an increase in computational complexity, but the experiments section does not seem to contain a clear comparison in terms of actual computation time. Such an experiment would be very insightful. 
5. The results in the currently included experiments appear mixed to me, making it unclear to me what the benefit of this new formulation is.

Based on these weaknesses I am leaning towards rejection. However, given my lack of understanding of the theoretical parts, my rating should be taken with a grain of salt. 

If the authors can add the experiments that I mentioned above and find strong results, then I would be willing to increase my rating.

[1] Max van Spengler, Erwin Berkhout, and Pascal Mettes. *Poincaré resnet.* Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023.

### Questions
1. You mention in the discussion that this new architecture could be more interpretable than existing architectures. Can you explain this claim a bit further?
2. What does metric equivalence mean within the context of your paper? The only common definition that I know is when two metrics on the same set induce some equivalent structure such as the same topology. However, I'm not sure I understand how this would work when there are two different underlying spaces.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes “Cartan networks,” a hyperbolic deep learning architecture that treats each layer as a solvable Lie group and maps between layers as a composition of a group homomorphism and an isometry. They proposed linear layers as a composition of homomorphisms between solvable groups with associated softmax and regression layers. They showed that adding pointwise activations recovers universal approximation and contains Euclidean networks as a special case. Their benchmark on synthetic and image classification tasks show in general comparable performance to Euclidean and tangent-space-based Poincare models.

### Strengths
1. The proposed hyperbolic linear and regression layers based on group isomorphisms/homorphisms and series of solvable groups is interesting
2. The proposed layers are defined by operations that are rather intrinsic, which is a potential useful way to improve hyperbolic neural network operations
3. Experiments show incorporating activation in the model leads to improvement performance, which is consistent with the theory presented in the paper

### Weaknesses
1. The experiments show more or less comparable performance against the Euclidean and Poincare model, with the few cases of noticeable improvements being synthetic datasets. This shows relatively weak motivation for the need for the model and whether the proposed method is actually effective
2. The baseline Poincare model is rather naive and simple. Since Ganea et al., 2018, many works have proposed hyperbolic linear and activation layers that lead to significant improvements (e.g. Chen et al. 2022, Shimizu et al. 2021, Yang et al. 2024). Missing comparison with these methods weakens the claims about the effectiveness of the proposed method as well. Coincidently, Chen et al. 2022 is also falsely cited to use nonlinearity on the tangent space on lines 358-359, when in reality they incorporate the nonlinearity into a fully hyperbolic layer. Yang et al. 2024 also does not use tangent space for nonlinearity.

Yang et al. 2024: https://arxiv.org/abs/2407.01290

3. The proposed method incurs computational overhead and rigidity into the model, even against some of the newer hyperbolic methods, but no analysis or discussion is given for this aspect.

### Questions
1. Since the proposed method does not show convincing improvements over the baselines, especially against the Euclidean model on image tasks, why strong are the motivation for this method? Can the authors find an application or dataset where preserving the group structure of hyperbolic manifolds is actually beneficial?
2. Also see weakness

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
This paper proposes Cartan networks which is a new class of hyperbolic deep learning algorithms where group homomorphisms are interleaved with metric-preserving diffeomorphisms. In particular, the authors highlight the metric equivalence of the hyperbolic space with a solvable Lie group to exploit the group structure as a tool in architectural design. Extensive experiments show the proposed architectures achieve competitive or better performance w.r.t. Euclidean and standard hyperbolic neural networks on real and synthetic datasets.

### Strengths
1. The proposed Cartan networks posses intrinsic Hyperbolic architecture which avoids the need for tangent-space approximations (as in Poincaré networks), making computations geometrically consistent and potentially more expressive.

2. Cartan networks allow stacking of homomorphisms and isometries, giving architectural flexibility and they are compatible with standard deep learning techniques (convolutions, batch normalization, dropout).

3. Cartan networks outperform Euclidean and standard hyperbolic networks on several benchmarks, including regression tasks, MNIST variants, CIFAR-100, and convolutional model.

### Weaknesses
1. The proposed architectures seem complex and require careful handling of solvable coordinates, fiber rotations, and Lie group operations, which may be non-trivial to implement. Also, it is not clear if the the proposed architecture can be applied to large datasets and extended to larger architectures. 

2. Although parameter-efficient compared to some Euclidean layers, the matrix multiplications and exponential/fiber rotations could introduce extra computational cost.

3. The use of solvable group coordinates, fiber rotations, and Cartan/pain coordinates makes the learned representations even less interpretable compared to standard Euclidean embeddings.

### Questions
1. How do Cartan networks perform on very large-scale datasets or in high-dimensional hyperbolic spaces? Are there practical limits due to the complexity of solvable group operations and fiber rotations?

2. Can the representations learned by Cartan networks be transferred across tasks or datasets, similar to pre-trained Euclidean or graph neural networks? How well do they generalize beyond the training domain?

### Soundness
3

### Presentation
3

### Contribution
3
