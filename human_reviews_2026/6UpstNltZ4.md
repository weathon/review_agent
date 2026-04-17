# A Recovery Guarantee for Sparse Neural Networks

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 8, 6

## Abstract
We prove the first guarantees of sparse recovery for ReLU neural networks, where the sparse network weights constitute the signal to be recovered. Specifically, we study structural properties of the sparse network weights for two-layer, scalar-output networks under which a simple iterative hard thresholding algorithm recovers these weights exactly, using memory that grows linearly in the number of nonzero weights. We validate this theoretical result with simple experiments on recovery of sparse planted MLPs, MNIST classification, and implicit neural representations. Experimentally, we find performance that is competitive with, and often exceeds, a high-performing but memory-inefficient baseline based on iterative magnitude pruning. Code is available at https://github.com/voilalab/MLP-IHT.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper studies sparse training through a recovery-guarantee lens for two-layer ReLU MLPs with scalar output. Using a convex reformulation that enumerates activation patterns and fuses layer weights, the training objective becomes a structured linear sensing problem. Under Gaussian inputs and two separability conditions on activation patterns, the sensing matrix satisfies restricted strong convexity and smoothness, which makes the planted sparse weights uniquely identifiable. Experiments on planted MLP recovery, MNIST classification, and implicit neural representations show the proposed method is competitive with or better than iterative magnitude pruning while using far less memory, even beyond the exact theory regime.

### Strengths
1.The paper gives identifiability and efficient recovery for planted sparse weights in two-layer scalar-output MLPs under Gaussian data.
2. The activation-pattern convexification connects ReLU training to linear sensing in a way that enables analysis with classical tools.
3. Across planted recovery, MNIST, and image INRs, IHT is typically stronger than IMP while using memory that scales with s rather than with dense parameter counts.

### Weaknesses
1. Scope limited to shallow scalar-output MLPs in theory.
2. The main theorem relies on i.i.d. Gaussian covariates and full enumeration of activation patterns, which can be unrealistic or computationally heavy.

### Questions
1. Extension beyond Gaussian inputs? Can the proof be adapted using sub-Gaussian or whitened real data, and what additional conditions would be needed on X?

2. The experiments use randomly sampled patterns and sequential convex updates. Is there a provable guarantee when A is updated during training, perhaps via a stability or tracking argument?

3. The appendix describes vector-output handling and layer-wise training. Could the convex deep-net formulations be combined with your analysis to yield recovery guarantees for deeper or multi-output networks?

### Soundness
3

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
2

### Summary
This paper presents the theoretical recovery guarantees for sparse ReLU neural network weights, focusing on two-layer scalar-output MLP networks. The authors prove that under certain structural conditions on sparse network weights and random Gaussian training data, an Iterative Hard Thresholding algorithm can exactly recover these weights with memory linear in the number of nonzero weights. The theoretical analysis uses convex reformulations of MLPs and establishes restricted strong convexity and restricted smoothness properties. Experimental validation on planted MLPs, MNIST classification, and implicit neural representations demonstrates competitive or superior performance compared to iterative magnitude pruning.

### Strengths
- This work provides formal recovery guarantees for sparse neural network weights, filling an important gap between compressed sensing theory and neural network optimization.
- The paper clearly articulates assumptions and provides detailed proofs showing how these conditions arise with high probability under Gaussian data, making the theoretical results verifiable and interpretable.
- The experimental studies, which extend beyond the theoretical analysis, demonstrate the broader applicability of the guarantees, including vector outputs, deeper networks, and various tasks.

### Weaknesses
- The theoretical guarantees, although interesting and important, are restrictive. It is limited to two-layer, scalar-output networks with Gaussian random data. However, real-world applications may involve deeper networks, structured data distributions, and multi-dimensional outputs.
- Assumption 1 requires concrete weight structures, i.e., binary hidden weights or binary output weights, which may not reflect realistic sparse networks.
- The paper only compares against IMP. More recent sparse training methods, such as dynamic sparse training, pruning at initialization variants, and other memory-efficient approaches, can be compared. Another limitation is that the runtime analysis, with IHT sometimes being slower than IMP for larger problems, which may be a limitation in practice.

### Questions
- How can the proposed analysis be extended to non-Gaussian data distributions?
- More analysis on how performance degrades when assumptions are violated would strengthen the work.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces and studies a method for training sparse shallow ReLU neural networks. The method reformulates the neural network optimization problem as a convex optimization problem and uses iterative hard-thresholding to obtain a sparse solution. Theoretical results guarantee the recovery under reasonable assumptions in the case of shallow networks. Empirical experiments verify that the method also works well for deep networks.

### Strengths
The method for training sparse networks is innovative. The theoretical guarantees are interesting and relevant. The practical performance of the methods is good. The presentation of the paper is good, it is easy to read and clear.

### Weaknesses
The theory only applies to shallow networks. It would be interesting to try to extend it to deeper networks. Nonetheless, this is a valuable theoretical contribution.

### Questions
Can the theory be extended beyond Gaussian training data?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper provides the first recovery guarantee for sparse ReLU neural networks by reformulating sparse MLP training as a structured compressed-sensing problem. Under Gaussian data and enumerated activation patterns, the sensing matrix satisfies restricted strong convexity and smoothness, allowing Iterative Hard Thresholding (IHT) to exactly recover sparse weights with high probability. The method requires memory linear in the number of nonzeros. Experiments on planted MLPs, MNIST, and implicit neural representations validate the theory and show that IHT performs comparably or better than Iterative Magnitude Pruning (IMP) while being significantly more memory-efficient.

### Strengths
A rigorous and elegant theoretical framework connecting sparse neural network recovery with compressed sensing, supported by solid experimental validation.
1. First work proving exact recovery (Theorem 1) for sparse MLP weights with explicit high-probability bounds and convergence rates; recovery builds on Assumption 2 and the RSC/RSS properties (Lemma 1).
2. Convex reparameterization (Equation 3) via weight fusion turns the nonconvex problem into a linear sensing framework; Assumptions 1–2 are well-motivated and verified through constructive examples in Appendix D.
3. Memory scales linearly with sparsity (s); sample complexity grows with active weights rather than total parameters. IHT achieves competitive or superior accuracy to IMP while using less memory.
4. Well-structured from motivation through theory to experiments, with a comprehensive appendix providing implementation and proofs.

### Weaknesses
The main limitation lies in the unquantified theory–practice gap and missing diagnostics connecting empirical behavior to theoretical parameters.
1. Theorem 1 assumes a fixed, enumerated sensing matrix A, but experiments adopt sequential convex updates—updating A after each IHT iteration (following an initial fixed epoch). The paper acknowledges this deviation (Appendix A) but provides no convergence analysis or ablation on update frequency.
2. Empirical values of the RSC and RSS constants and their ratio are not reported, leaving unclear how tight the theoretical conditions are in practice.
3. Results are averages over three runs without error bars, standard deviations, or variance. Adding uncertainty plots (e.g., box plots or success-probability curves) would strengthen performance claims.

### Questions
1.Can you analyze how A evolves during sequential convex updates—for example, by measuring similarity between consecutive A matrices—and test how update frequency affects convergence?

2.Can you report empirical ($\alpha,\beta,\beta/\alpha$) for at least one representative setting to show whether the RSC/RSS conditions approximately hold in practice.

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper provides theoretical guarantees for recovering the sparse weights of a ReLU network using an efficient, memory-linear algorithm. The proposed method, validated on tasks on MNIST, matches or exceeds the performance of a memory-inefficient pruning baseline.

### Strengths
1. This work establishes the sparse recovery result for ReLU MLPs. 

2. Using shallow networks on Gaussian data, the experiments demonstrate that Iterative Hard Thresholding (IHT) is more effective and memory-efficient than a strong IMP baseline, recovering superior sparse networks.

### Weaknesses
Using the eq. (2), training the 2-layer MLP with MSE loss is a nonconvex optimization problem.  From the lines 162-175, the authors consider fixed generator vectors $h_i\in R_d$ and fusing the weights via $w_i=u_iv_i$ to build the eq. (3). This transformation allows the results to be readily obtained using sparse recovery theory. However, several issues arise:

1. The theory requires the matrix $A$ to satisfy the conditions of Lemma 1, which is attributed to the sparsity of $h_i$ or $u_i$. Typically, 
 $X$ is assumed to be a Gaussian data matrix. How is Lemma 1 satisfied when the input data is non-Gaussian, such as the binary-valued pixels in the MNIST dataset

2. The theoretical results are promising, but how does the method perform on more complex, real-world datasets like CIFAR-10 or ImageNet-200?

3. This work focuses on recovering the sparsity pattern of the weights $u_i$. What can be said about the sparsity or recovery guarantees for the corresponding weights $v_i$?

### Questions
please see weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
