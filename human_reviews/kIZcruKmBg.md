# Physics-informed neural networks for transformed geometries and manifolds

- Avg Score: 3.25
- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 6, 3, 1

## Abstract
Physics-informed neural networks (PINNs) effectively embed physical principles into machine learning, but often struggle with complex or alternating geometries.
We propose a novel method for integrating geometric transformations within PINNs to robustly accommodate geometric variations. Our method incorporates a diffeomorphism as a mapping of a reference domain and adapts the derivative computation of the physics-informed loss function. This generalizes the applicability of PINNs not only to smoothly deformed domains, but also to lower-dimensional manifolds and allows for direct shape optimization while training the network.
We demonstrate the effectivity of our approach on several problems: (i) Eikonal equation on Archimedean spiral, (ii) Poisson problem on surface manifold, (iii) Incompressible Stokes flow in deformed tube, and (iv) Shape optimization with Laplace operator.
Through these examples, we demonstrate the enhanced flexibility over traditional PINNs, especially under geometric variations. The proposed framework presents an outlook for training deep neural operators over parametrized geometries, paving the way for advanced modeling with PDEs on complex geometries in science and engineering.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper intends to improve the performance of PINN on domains of complex geometries. The method is to  use smooth transformations to transform a complex geometry to less complex one which is a called reference domain. If the transformations are differentiable, the training of modified PINNs is the same as training vanilla PINNs.

### Strengths
**Originality:** The paper implements diffeomorphisms to the problem of PINN on complex geometries.

**Quality:** The paper explores the proposed methods on some typical examples to demonstrate the effectiveness of the method.

**Clarity:** The idea is conveyed directly and straightforward.

**Significance:** Combining diffeomorphism with training of neural network is somewhat interesting and natural, due to the differentiability of transformations.

### Weaknesses
One of the major weakness is that the paper does not include experiments of comparison between modified PINN and vanilla PINN. In order to show the effectiveness of the proposed method, the author should also test the performance of PINN on all the problems in section 4.

### Questions
If the original problems $L(u)=f$ in $\Omega$ is transformed to $L_x(u \circ \phi) = f$ on reference domain $\Omega_{ref}$, then $L_x$ should not equal $L$. The calculation of $L_x$ should use chain rule. In your paper, this part is hardly touched. How did you actually implement your method in experiments?

### Soundness
2 fair

### Presentation
1 poor

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
In this paper, it is argued that the existing approaches to physics-informed neural networks are not apt for complex and transforming geometries. To this end, the paper presents an approach to introduce geometric transformation within the physics-informed neural network design. Concretely, it enforces the Dirichlet boundary condition using distance function to account for complex geometries. Experimental results on four different examples are shown to demonstrate the suitability of the method.

### Strengths
* It is a well-written paper. 
* The use of Dirichlet boundary conditions is promising.
* An initial approach to explore a new direction for more promising neural network design.

### Weaknesses
* Some of the technical notations are not fully exposed and detailed.
* Experiments are limited on toy-example and missing on the manifolds which are widely used in science and engineering application.
* The paper misses to highlight the limitations of the proposed approach.


Kindly refer to the Questions section for more comments.

### Questions
## Domain and Transformation

It’s better to include the dimension of the variables on the side of the Eq(1). 

## 2.2.1 Manifold: $m < n$

$\mathcal{L}_x$ and $\mathcal{L}_y$ need more explanation. The subscripts have not been explained. Diagram conveys that one is in the reference domain and other is in the computational domain yet it's better to write near the equation (4)-(5) and following equation.

## 3.1 Exact boundary condition with output transform

Kindly help me understand the approximation of $\hat{u}$, given that the inverse must hold and the proposed approximation is not linear.

## 4.4 Shape Optimization with Laplace Operator
I am not entirely convinced with the imposed boundary condition. What could be considered a weak boundary condition is not fully exposed in the paper. Furthermore, I request the authors to perform some experiments and analysis of the proposed theory on negative curvature surfaces with the introduced local approach. Also, the use of Laplace-Beltrami operator for shapes.

In addition to the above, experiment on Low-Dimensional manifolds is simple and not convincing to me for real application. I request the authors to provide some analysis and results on popular manifolds such as low-dimensional SPD, Grassmannian manifolds, etc.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper employs physics-informed neural networks (PINNs) for addressing intricate or changing geometrical configurations. The primary technical innovation lies in the incorporation of a geometric transformation (diffeomorphism) of a reference domain to describe the computational domain.

### Strengths
The problem is well defined and the author proposes a clear formulation in solving the problem.

### Weaknesses
Unfortunately,  it appears that the problem tackled in the paper is somewhat incremental, and the proposed solution lacks a surprising or profound aspect. In the context of an ICLR paper, I'm seeking a novel problem that has not previously been successfully addressed, made attainable through this approach, or a novel method to solve a well-established problem that has been extensively explored. Unfortunately, neither of these elements seems to be present in the paper.

Furthermore, the examples provided mainly consist of small-scale 2D toy examples. To comprehensively assess the efficacy of this approach, it would be necessary for the authors to set up larger-scale problems that are well-documented in CFD/JCP/CMAME papers.

### Questions
How does this work compare with Bonev+ ICML 2023? These authors propose a neural PDE approach using spherical coordinate. Your paper seems to be more general. Can you reproduce some of the examples in their paper so we can have an apple to apple comparison?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
1: strong reject

### Rating Number
1

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The work proposes a method to enhance Physics-informed Neural Networks (PINNs) by integrating geometric transformations, to address challenges posed by complex or non-euclidean geometries. 
The method utilizes a diffeomorphism $\phi$ that maps a reference domain $\Omega_{ref}$ to the observation domain $\Omega$, adapting the derivative computation in the physics-informed loss function. The approach was demonstrated through various problems: Eikonal equation on Archimedean spiral, Poisson problem on surface manifold, Incompressible Stokes flow in deformed tube. Finally, they show that their method can be applied to perform shape optimization according to a Laplace PDE loss.

### Strengths
The paper is easy to read and the geometric transformation seems reasonable to solve this kind of problem. The first three different examples each test a different geometric setting. The figures are pretty.

### Weaknesses
The method relies on the output transformation trick to enforce boundary conditions (BC), which is well suited for Dirichlet BC only. It would not be applicable as is for different kinds of BC, but the authors have a much more general claim.

Except for the last example, which we will discuss next, the diffeomorphism $\phi$ is known a priori. Therefore the method in such case simply looks like a change in variable with a known function. How can you apply this method on a domain which is not equipped with such a transformation ?

The last example is very mysterious to me. I actually do not understand what the method is supposed to achieve by learning simultaneously to impose the PDE constraint and the geometric transformation. Do we know what target geometry the network should converge to ? Besides, the network that learns the transformation is not a diffeomorphism, so there is no guarantee that the optimization problem finds a correct solution. 

The authors do not compare their method with any existing work. There is no literature review. As a result, we do not really understand why these problems cannot be tackled with existing methods. Why do they fail ?

The authors do not provide any numerical results for their methods, and even the qualitative results do not include the ground truth solutions. It is therefore impossible to judge the effectiveness of the method.

### Questions
What is the difference between $\mathcal{L}$, $\mathcal{L}_x$ and $\mathcal{L}_y$ concretely for each example ?

 What does the following sentence mean ? "transformed PINN finds the exact length with an error of = 0.1 \%" .

### Soundness
2 fair

### Presentation
2 fair

### Contribution
1 poor
