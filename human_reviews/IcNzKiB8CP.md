# TELEPORTATION WITH NULL SPACE GRADIENT PROJECTION FOR OPTIMIZATION ACCELERATION

- Decision: Reject
- Scores: 3, 3, 3, 6

## Abstract
Optimization techniques have become increasingly critical due to the ever-growing model complexity and data scale. In particular, teleportation has emerged as a promising approach, which accelerates convergence of gradient descent-based methods by navigating within the loss invariant level set to identify parameters with advantageous geometric properties. Existing teleportation algorithms have primarily demonstrated their effectiveness in optimizing Multi-Layer Perceptrons (MLPs), but their extension to more advanced architectures, such as Convolutional Neural Networks (CNNs) and Transformers, remains challenging. Moreover, they often impose significant computational demands, limiting their applicability to complex architectures. To this end, we introduce an algorithm that projects the gradient of the teleportation objective function onto the input null space, effectively preserving the teleportation within the loss invariant level set and reducing computational cost. Our approach is readily generalizable from MLPs to CNNs, transformers, and potentially other advanced architectures. We validate the effectiveness of our algorithm across various benchmark datasets and optimizers, demonstrating its broad applicability.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This paper proposes an improved algorithm for teleportation-based optimization, in which at certain times during training the network weights are moved to a different location with roughly the same loss but higher gradient norm. Their approach uses an approximate projection onto the loss level set while optimizing the gradient norm during the teleportation step. The algorithm is validated on experiments with MLPs, CNNs, and Transformers.

### Strengths
1. The authors propose a faster algorithm for the teleportation step that can also be applied to architectures beyond MLPs.
2. The authors validate their approach by using it to modify a variety of standard optimizers on multiple problems, showing consistent improvements in training loss.
3. Code is attached to reproduce the results.

### Weaknesses
1. The approximation scheme is not very clearly described. It would be helpful to have pseudocode for the actual core contribution of the work, rather than just for the teleportation approach more broadly. The goal of the approximation (“to ensure that the gradient update in Equation 5 preserves the correlation between the weights and the space of significant representation as much as possible”) is only defined in words, and the error we should expect is not analyzed.
2. A major claim is the improved speed over past teleportation approaches, such as the one using symmetry or the one using linear approximation. However, no comparison is done for either case. For example, it would be useful to see whether the reduced per-iteration complexity is enough to overcome the approximation error and to do better than the symmetry-based approach. Similarly, a wallclock (rather than epoch) comparison between this approach and vanilla optimizers would also be useful.
3. The observed improvement is mainly in training performance, with minimal improvement in generalization. The authors suggest that there may be ways to overcome that, but do not evaluate this. Without a clear path towards test-time performance improvement the utility of teleportation-based approaches is unclear.

### Questions
1. What does “enhance the gradient norm” mean?
2. In what sense is symmetry teleportation “a state-of-the-art algorithm”?
3. It is strange to cite a 2018 applications paper for MLPs (they have been around for a long time), a 2018 paper proposing ReLUs for classification for ReLUs (which have also been around for a long time), and a 2022 applying Transformers to time series for multi-head attention when they were introduced in 2017 by Vaswani et al.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper presents a teleportation technique for accelerating the convergence of gradient descent. The idea of teleportation is to move the parameters on the level set of the loss function to get a better (e.g., steeper) point before taking a gradient descent step, and this is done every few iterations. The authors aim to address limitations in previous teleportation methods, which were primarily restricted to multi-layer perceptrons (MLPs), by generalizing it to a broader range of architectures, including convolutional neural networks (CNNs) and transformers. Their method aims to reduce computational overhead by eliminating dependency on specific group actions and using an efficient gradient projection technique.

### Strengths
- The paper concerns an interesting problem, specifically exploring optimization acceleration through teleportation techniques.
- The introduction effectively provides background context and situates the study within related work, giving readers a clear overview of the existing literature and the motivation for this approach.
- The idea of doing gradient ascent on the gradient norm, while projecting it onto the level set of the original loss is interesting.

### Weaknesses
- The authors mix layer-wise and global operations without a clear explanation or justification. While Equations (4) and (5) and Algorithm 1 (correctly) outline parameter updates at a global level, Sections 3.1, 3.2, and 3.3 abruptly pivot to layer-wise operations and projections without justification. The authors should clarify why the projection of the entire parameter vector 𝜋 is equivalent to projecting subsets of the elements (i.e., layers) separately.
- Relatedly, from Equation (5), it appears that 𝜋 is defined to take inputs of the same dimensionality as the complete set of weights. If so, then reusing 𝜋 in Equations (11-13) is incorrect, as these equations seem to imply different dimensionality.
- Again, on a related note, the statement that "the gradient of the teleportation objective function resides within the space spanned by the input data" is incorrect and misleading. The paper actually seems to refer to the input spaces at each layer, not the space spanned by the input data. This claim should be corrected and replaced with a precise statement about the layers if there exists one.

### Questions
- Please refer to the weaknesses above.
- In the unnumbered equations between (7) and (8) for MLP, CNN, and Self-Attention: I see that the order of multiplication in which the input to the layer appears differs between architectures (x*\delta in CNN and Self-Attention, as opposed to \delta*x in MLP). Can the authors explain whether this order affects the theoretical claims or the projected outcome, and if so, in what ways?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This paper presents a novel algorithm, "teleportation with null space gradient projection," designed to accelerate optimization in deep learning models. The proposed method improves upon existing teleportation techniques by reducing runtime and mitigating error accumulation. Additionally, it extends teleportation optimization to a broader range of models. The authors validate their approach across diverse model architectures, including Multi-Layer Perceptrons (MLPs), Convolutional Neural Networks (CNNs), and Transformers, employing various benchmark datasets and optimization methods to demonstrate its effectiveness.

### Strengths
- The paper is well-written and easy to follow.
- The proposed method is architecturally flexible, as it does not rely on group action from the underlying architecture. This allows the approach to be applicable across a broader range of architectures.

### Weaknesses
- The explanation for why projecting onto the Residual Gradient Space (RGS) keeps the parameters within the loss-invariant level set of the original loss is unclear, since the gradient being projected is the gradient of the teleport loss projected to RGS.
- The paper lacks a comparison with other state-of-the-art methods that employ teleportation, such as Zhao et al. (2022) and Mishkin et al. (2024).
- A runtime comparison with non-teleportation optimizers counterparts (SGD, Momentum, Adagrad, Adam) is absent.

### Questions
- How does projecting the gradient of the teleportation loss onto RGS ensure the parameters remain on the level set of the original loss?
- Could you provide a performance comparison of the proposed method with other teleportation techniques, specifically comparing with Zhao et al. (2022) on MLP models and with Mishkin et al. (2024) on applicable models?
- What are the runtime of the baseline optimizers before and after introducing teleportation with null-space gradient projection?

### Soundness
2

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
This work considers an alternative strategy for teleportation in optimization landscapes, i.e., going to locations within the level set of the loss but which are better suited for further optimization, like having higher gradient norm. Past methods consider using the group action of continuous symmetries, but this makes them limited to MLPs. This work considers maximizing the gradient norm to teleport while ensuring the layerwise updates of parameters falls in the null space of the layer input. Experiments showcase that this method leads to faster convergence in train and test performance while performing at par with baseline optimization methods.

### Strengths
- The idea to utlize (layer) input-space null projection is built atop the work of GPM (Saha et al'21) in continual learning and its use here for the purposes of teleportation is neat.

- This extends the applicability of the approach and might draw further interest in these methods. The resulting method is also more efficient than symmetry teleport based methods.

- The experiments cover a variety of scenarios, wherein the method results in faster convergence while being as good if not slightly better than the baselines.

### Weaknesses
- **Wall-clock comparison of convergence:**
 It is unclear how much excess time is being used during the process of teleportation, and if the gains in faster convergence are worth the extra effort. 

- **Difference in arrived solutions:**
I would like to see if the solutions reached with teleportation differ qualitatively to those reached without. Can the authors make the LMC curves (Frankle et al, 2019) to see if there are barriers between the reached solutions?

- **Comparison to group action based method:**
 I think it would be interesting to compare the results of your method to group action based ones in a simple setting with MLPs to see in what ways the methods differ. 


- **Stability of the hyperparameters**:
It is not clear to me how much do the hyperparameters of teleportation, as well as SVD thresholds, have to be tuned. Can you present a study showing the robustness (or lack of)?

- **Poor referencing of related work:**
A lot of the citations are plain wrong. E.g., I don't think the citations are representative, and some are just plain wrong: 

  - Hessian matrix (Sun et al., 2019).
  - Adam (Kashyap 2022)
  - ReLU (Agarap, 2018)
  - MLP (Taud & Mas, 2018).
  - CNN (Li et al., 2021).
  - multi-head self-attention layers (Wen et al., 2022).

### Questions
See weaknesses section. 

Regarding presentation, I would also make a few suggestions. Instead of input-space, quality it as input space of layers. Otherwise it is confusing. Also, please fix the image titles to TinyImageNet (which currently say ImageNet).

### Soundness
2

### Presentation
2

### Contribution
3
