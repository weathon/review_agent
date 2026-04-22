# CORDS - Continuous Representations of Discrete Structures

- Avg Score: 5.20
- Decision: Accept (Poster)
- Scores: 2, 6, 8, 6, 4

## Abstract
Many learning problems require predicting sets of objects when the number of objects is not known beforehand. Examples include object detection, molecular modeling, and scientific inference tasks such as astrophysical source detection. Existing methods often rely on padded representations or must explicitly infer the set size, which often poses challenges. We present a novel strategy for addressing this challenge by casting prediction of variable-sized sets as a continuous inference problem. Our approach, CORDS (Continuous Representations of Discrete Structures), provides an invertible mapping that transforms a set of spatial objects into continuous fields: a density field that encodes object locations and count, and a feature field that carries their attributes over the same support. Because the mapping is invertible, models operate entirely in field space while remaining exactly decodable to discrete sets. We evaluate CORDS across molecular generation and regression, object detection, simulation-based inference, and a mathematical task involving recovery of local maxima, demonstrating robust handling of unknown set sizes with competitive accuracy.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper claims to develop CORDS (continuous representation of discrete structures), which provides a "bijective" representation that maps sets of spatial objects with features to continuous "density" and "feature fields".

### Strengths
None.

### Weaknesses
The major problem with this paper is that it uses some jargon and terms that are not mathematically defined, nor are they standard in machine learning research. This hinders the ability of a reviewer to fully understand and judge this paper. For example, starting from the first sentence of the Abstract, what does "number" mean when the paper says "their number in advance"? The Abstract talks about existing methods in the Abstract, but no problem has been laid out. The Abstract then talks about a challenge that is not substantiated.  What does "variable cadinality" mean? What does "bijective representing" mean? What is "Feature field"? These are all non-standard terminologies that I have not seen before.

### Questions
What is the mathematical/ML problem that the paper is solving?

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper tackles the problem of predicting the cardinality of object sets and proposes a method that utilize kernel based continuous field that can map to the finite set of spatial objects. In this case, the total density mass is the number of objects. The proposed CORDs provides an invertible mapping that enable the exact decoding from the field space to the discrete space. The authors demonstrate the effectiveness of proposed representation in various applications, including molecular generation, object detection, simulation-based inference and synthetic functions.

### Strengths
- The proposed method demonstrates a clean and clear idea that enable the encoding to continuous fields and exact decoding to discrete sets, which also makes the prediction of object cardinality feasible
- This problem it tackles itself is a rather fundamental issue that can be further applied to many downstream tasks as shown in the paper without being restricted to specific use case
- The bijective construction make the transition between discrete and continuous space smooth without the need of additional module

### Weaknesses
- The computational cost including the encoding and decoding process might increase when dealing with larger samples
- The experimental performance of CORDS may fall behind some baselines, for instance, the performance in the molecular tasks, so what could be the benefit of this method compared to those baselines in this case.

### Questions
- Now, the current representation is rather unified for all downstream tasks, do you think it might be valuable to calibrate current method to become some task-specific representation?
- How the kernel choice and kernel parameter affect the construction?

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
The paper proposes to represent discrete sets of arbitrary cardinality as continuous density and feature fields. This enables models to predict/generate sets of objects without fixing cardinality in advance. The authors formally show this bijection. The density field is simply a sum of Gaussians centered at each object, whereas the feature field is the same except each Gaussian is weighted by the corresponding feature vector. Recovering a discrete set requires drawing samples to integrate the density field to find the cardinality $N$. Then, positions are recovered by fitting a Gaussian mixture model to the samples and densities and refining object positions with LBFGS. Feature vectors are recovered by solving a linear system with coefficients and constraints defined by the recovered Gaussians and feature field. The framework of CORDS generalizes Gaussians to generic kernels. CORDS is validated with experiments across 3D molecules, images, astrophysical signals, and local maxima of functions.

### Strengths
Motivation for modelling cardinality is clear in the introduction.

Contextualization with related works is extensive without being wordy.

The paper is beautifully written with both intuitive descriptions as well as formalism. Figures are clear and visually appealing.

The proposed method is validated with experiments across multiple domains, showcasing the generality of CORDS. In 3D molecules, CORDS achieves state-of-the art generation quality out of field/voxel-based methods.

### Weaknesses
The paper does not discuss its limitations enough. Representing 3D molecules as continuous fields still requires handling discrete samples as inputs to the neural network, so this blows up $N\approx 20$ atoms into $M\approx 1000$ points, and incurs extra cost in integrating densities, gradient-based recovery of positions, and a $O(N^3)$ linear solve to recover feature vectors. It is unclear how much overhead is added by each of these steps, as wall-clock timing, number of training steps, and compute resources are not reported for molecule generation.

### Questions
1. In 3D molecule generation, how much is the computational cost of integration-based decoding, both in absolute wall-clock timing and relative to the neural network part? How long does training take, in wall-clock time and number of training steps, and what compute resources are used?
2. Is it possible to make the kernel learnable?
3. Why is it that (line 1129) "feature solve typically dominates cost only for very small $\hat{N}$"?
4. The bijection between discrete sets and continuous fields seems more promising in the other direction: mapping high-dimensional signals down to sparse discrete sets, like tokenization. Alternatively, one could coarse-grain large, discrete sets into continuous fields with fewer points. Does the CORDS framework enable/motivate any approaches in this line of thinking? In physics and chemistry, there are many continuous fields that if represented as discrete but sparse objects could reap benefits: electron densities, wavefunctions, scalar and vector fields in quantum field theory and statistical field theory (e.g. $\phi^4$ theory or solutions to a Landau-Ginzburg Hamiltonian).
5. Future work in decoding spectra (NMR, IR) with overlapping signals could be interesting.

Some potentially related works are Gaussian splatting for rendering images [1] and representing electron density as sums of Gaussians [2].

[1] Kerbl, B., Kopanas, G., Leimkühler, T., & Drettakis, G. (2023). 3D Gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4), 139-1.

[2] Elsborg, J., Thiede, L., Aspuru-Guzik, A., Vegge, T., & Bhowmik, A. (2025). ELECTRA: A Cartesian Network for 3D Charge Density Prediction with Floating Orbitals. arXiv preprint arXiv:2503.08305.


nit-picking:
1. line 269: "using Eq. equation 1" has an extra word
2. Several citations do not have parentheses around them.
3. line 886: "we will define Field of Graph is a quadruple"
4. line 1062: BIC is not defined
5. line 1102: there is no mention of Laplacian/Epanechnikov kernels elsewhere in the paper
6. line 1253: "diffusion machinery"
7. When introducing kernels, it may be clearer to suggest that the reader imagine Gaussians as a typical example.

### Soundness
4

### Presentation
4

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
This paper introduces CORDS, a framework that maps variable-sized discrete sets (e.g., points with features) into continuous representations through density and feature fields. Each element contributes a kernel function centered at its position, whose integrated density encodes cardinality while the feature field carries attribute information. The author conducts experiments on diverse domains—molecular generation, object detection (MultiMNIST), scientific simulation inference, and detection of local maxima. Experiments show competitive performance and robustness to variable object counts.

### Strengths
CORDS provides an elegant formulation that continuously encodes cardinality, positions, and attributes in a single differentiable field, with provable invertibility. This offers a strong conceptual link between discrete structures and continuous implicit fields.

The framework is tested across four distinct domains, showing broad potential beyond a single application type.

### Weaknesses
The approach relies on dense sampling (∼10³ points per structure) for high-fidelity reconstruction, which can become prohibitively expensive for large-scale data (e.g., complex molecules or real images). The paper lacks quantitative analysis of runtime, memory footprint, or scaling trends with respect to the number of objects or spatial resolution. Without this, the practical feasibility on larger datasets remains unclear.

While CORDS achieves performance comparable to existing models on molecular generation task, it does not clearly surpass them in generation quality. Moreover, as the number of atoms increases, the computational overhead of the continuous-field representation (especially the sampling and kernel-matching stages) grows rapidly, potentially making the approach less scalable than discrete or equivariant diffusion models. Given these results, it remains unclear what practical advantage CORDS offers for molecular generation compared to current SOTA methods—beyond the theoretical elegance of its continuous formulation.

How are weights for LMSE and count penalty (λ) tuned in detection, and how do they affect OOD-count robustness?

### Questions
See Weaknesses.

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
3

### Summary
The paper proposes a representation for discrete data based on neural fields, named CORDS. The central idea is to define an invertible mapping between features and continuous fields. The authors show experiments with CORDS on different tasks/domains (molecular generation, object detection, simulation-based inference in astronomy).

### Strengths
* The unification of continuous fields and discrete encodings is interesting.
* The presentation is generally clear, and the theoretical derivations are also clear.
* The idea is applied to different tasks/domains, which means that the method shows some generalization.

### Weaknesses
* In molecular generation (Table 1) the performance is considerably lower than other models. In molecular regression (Table 4) the performance is also significantly worse than other methods (e.g. EGNN, PaiNN, SE-GNN). In particular, the dipole estimation has the worst error (0.086) of all methods, several times higher than other methods (e.g. PaiNN: 0.012, SE-GNN: 0.023). Results for object detection are limited and have a similar issue. This does not match the "competitive performance" described in the abstract. Results for OOD are relatively better, but generally worse than other methods.

### Questions
* The idea seems interesting, but the method is not showing particularly good results for the chosen tasks. Moreover, the results do not support the initial claims.

* For molecular benchmarks, please clarify if baseline numbers were re-run or taken from earlier papers (if any).

### Soundness
2

### Presentation
3

### Contribution
2
