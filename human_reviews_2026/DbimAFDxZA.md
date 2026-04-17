# Universality and kernel-adaptive training for classically trained, quantum-deployed generative models

- Decision: Reject
- Scores: 6, 4, 6, 6

## Abstract
The instantaneous quantum polynomial (IQP) quantum circuit Born machine (QCBM) has been proposed as a promising quantum generative model over bitstrings. Recent works have shown that the training of IQP-QCBM is classically tractable w.r.t. the so-called Gaussian kernel maximum mean discrepancy (MMD) loss function, while maintaining the potential of a quantum advantage for sampling itself. Nonetheless, the model has a number of aspects where improvements would be important for more general utility: (1) the basic model is known to be not universal - i.e. capable of representing arbitrary distributions, and it was not known whether it is possible to achieve universality by adding hidden (ancillary) qubits; (2) A fixed Gaussian kernel used in the MMD loss can cause training issues, e.g., vanishing gradients, as we demonstrate in this paper. For the former, we prove that for an $n$-qubit IQP generator, adding $n + 1$ hidden qubits makes the model universal. For the latter, we propose a kernel-adaptive training method, where the kernel is adversarially trained. We formally prove that such adaptive kernels have strictly greater discriminative power, and also show that in the kernel-adaptive method, the convergence of the MMD value implies convergence in distribution of the generator. We also analyze the limitations of the MMD-based training method. Finally, we verify the performance benefits of our contributions on a synthetic, parity-check dataset. The results show that the kernel-adaptive training method outperforms the Gaussian kernel w.r.t. the total variation distance between the generator and the data, and the advantage of the adaptive method becomes larger as the qubit number increases. These modifications and analyses shed light on the limits and potential of these new quantum generative methods, which could offer the first truly scalable insights in the comparative capacities of classical versus quantum models, even without access to scalable quantum computers.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work presents a well-structured and technically solid QML study on enhancing the Instantaneous Quantum Polynomial (IQP) Quantum Circuit Born Machine (QCBM). The authors identify two major limitations of existing IQP-QCBMs including non-universality and training instability under a fixed Gaussian kernel MMD loss. Then, they theoretically prove that adding $n+1$ ancillary qubits enables universality for an n-qubit generator, and introduce kernel-adaptive MMD loss to improves training stability under a min-max optimization framework.

### Strengths
1. it is well-written and structured, making it easy for readers to follow the main arguments and technical developments.
2. the authors provides rigorous proofs of addressing two main questions that are the universality of IQP-QCBM when extended with ancillary qubits and the theoretical advantage of kernel-adaptive MMD training in terms of discriminative power and convergence guarantees.

### Weaknesses
the paper would benefit from more comprehensive numerical experiments to substantiate the claims and demonstrate the empirical behavior of the proposed approach under different settings.

### Questions
1. Theorem 1 indicates that $m = n + 1$ ancilla qubits are theoretically required to represent an arbitrary probability distribution over ${0,1}^n$ by tracing out the ancilla subsystem. However, in the empirical example shown in Fig. 1, the model achieves reasonable performance with only a single ancilla qubit. This observation raises several important questions. Does the representation of most practical probability distributions indeed require fewer than $m = n + 1$ ancilla qubits? Is the required number of ancilla qubits dependent on the structure or complexity of the target distribution? Furthermore, how can one systematically determine the minimal number of ancilla qubits necessary to faithfully approximate a given probability distribution?

2. Given that the enhanced IQP-QCBM serves as a quantum generative model, it would be valuable to further assess its performance on more realistic and complex datasets beyond the synthetic parity-check case. How does the proposed model perform on more practical benchmarks such as structured classical datasets or quantum states?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors explore improvements to the Instantaneous Quantum Polynomial Quantum Circuit Born Machine, a type of quantum generative model that produces bit-string samples. The authors tackle two main issues: first, the model’s lack of universality, its inability to represent all possible distributions and second, training difficulties arising from using a fixed Gaussian kernel in the MMD loss. They prove that adding n+1 hidden qubits to an n-qubit IQP generator can make it universal, at least theoretically. They also propose a kernel-adaptive training approach, where the kernel is trained adversarially to better guide optimization. Their experiments on a parity-check dataset show improved training stability and better performance, especially as system size increases.

### Strengths
S1. The theoretical contribution around proving universality with hidden qubits is a strong and novel result, it clarifies fundamental expressivity limits in a clean, rigorous way. 

S2.the adaptive kernel method is a practical and well-motivated innovation, bridging a clear gap in MMD-based quantum model training and showing meaningful empirical gains.

S3. The authors present a direct discussion in of current limitations and their proposed solutions, both in the introduction but also in their method section, e.g. showing that the that MMD with the Gaussian kernel might be ineffective in distinguishing probability
distribution.

S4. For the numerical results the Adaptive kernel shows a very clear, while realistic, improvement over the Gaussian Kernel.

### Weaknesses
W1. The universality proof is largely theoretical, with limited insight into whether such hidden-qubit setups are realistic or scalable in current hardware.

W2. The evaluation remains narrow, relying mostly on synthetic data and not demonstrating performance on more complex or real-world distributions, which makes the broader significance of the results a bit speculative.

W3. The theoretical work is on a specific class of quantum generative models, limiting the significance of this work.

W4. The numerical experiments are very limited, limiting again the generalizability and real-world confidence of the theoretical results

W5. The work is missing some good related work comparison, either in the introduction or separate section.

### Questions
Q1. Can the method be compared to in a more generalizable way, specifically more existing methods as well as more synthetic datasets? -if not can you please describe thoroughly why only this comparison is reasonable and comment on the possible lack on benchmark tests.

Q2.  Can you comment on the significance of this work in the more general context of generative models e.t.c. e.g. what is this work aiming to achieve/influence in the future. 

Q3. Is it possible to make the proof of theorem 1 constructive, explicitly map a target distribution $p$ to specific phase parameters $\theta_{k,y}$?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper focuses on advancing the IQP QCBM, a quantum generative model for bitstring generation that is classically trainable and quantum-deployable, by addressing two key limitations of existing versions. First, the base n-qubit IQP-QCBM is non-universal, meaning it cannot represent arbitrary probability distributions. The authors prove that adding n+1 hidden qubits enables exact universality, and even a small number of hidden qubits significantly improves approximation accuracy. Second, training with a fixed Gaussian kernel in MMD often causes vanishing gradients or fails to distinguish distinct distributions. The authors propose a kernel-adaptive adversarial training method: a critic network dynamically tunes the kernel’s spectral measure to focus on frequencies where the model and data distributions differ most. Theoretically, they show this adaptive kernel has stronger discriminative power, and MMD convergence under this method guarantees convergence in distribution. Experiments on synthetic parity-check datasets (12 to 16 bits) confirm that kernel-adaptive training outperforms fixed Gaussian kernels in minimizing TV distance between generated and true distributions, with the performance gap widening as qubit count increases. The paper also notes a limitation: MMD-based training (even with adaptive kernels) fails for certain "worst-case" distributions where TVD is maximal but MMD remains exponentially small.

### Strengths
Resolves two limitations of IQP-QCBM
Advances "classical training, quantum deployment", critical for scarce quantum hardware
Reusable frameworks (hidden-qubit universality, spectral kernel adaptation) extend to other quantum models
The theory seems rigorous

### Weaknesses
Exclusively uses synthetic parity-check datasets
Proves n+1 hidden qubits suffice for universality but no analysis of minimal qubit count
Adaptive kernel’s FVSBN has O(n^2) parameters. no runtime/memory comparisons vs. fixed Gaussian kernels for large n

### Questions
how do hardware noise/cross-talk affect universality? Have you tested small-scale on real quantum hardware?
Did you sweep ε or FVSBN initializations? 
Experiments show test TVD > training TVD. Did you try regularization? 
Can your framework adapt to IQP-QCBM for continuous quantum states (e.g., chemistry)? If yes, what spectral measure modifications are needed?

### Soundness
3

### Presentation
2

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
This paper addresses two key limitations of IQP-QCBM (Instantaneous Quantum Polynomial Quantum Circuit Born Machines): (1) lack of universality and (2) training difficulties with fixed Gaussian kernels in MMD-based objectives. The authors prove that adding n+1 hidden qubits makes the n-qubit IQP-QCBM universal, and propose an adversarial kernel-adaptive training method in which the kernel's spectral measure is learned. Experiments on synthetic parity-check datasets demonstrate improved performance with adaptive kernels.

### Strengths
The paper makes substantial theoretical contributions to understanding IQP-QCBM expressivity and trainability. The universality proofs are rigorous and constructive, providing both an asymptotic result and an elegant exact construction requiring only n+1 hidden qubits for an n-qubit generator. The problem is well-motivated through concrete examples demonstrating fundamental limitations of fixed Gaussian kernels, and the generalized MMD framework via spectral representation maintains the crucial property of classical tractability while enabling adaptive kernel learning. The authors demonstrate intellectual honesty by acknowledging fundamental limitations through worst-case analysis, and the overall presentation is clear and well-structured with intuitive explanations accompanying formal results.

### Weaknesses
The experimental validation is insufficient to support the practical claims, with evaluation limited to small synthetic datasets specifically constructed to challenge Gaussian kernels, which may artificially inflate the perceived advantage of the adaptive method. The practical utility of the hidden qubit construction remains largely theoretical and unexplored, with no empirical validation of the n+1 hidden qubit universality result and limited analysis of how trainability scales with hidden qubits. The kernel parameterization uses a relatively simple FVSBN architecture without exploration of alternatives or ablation studies, and the initialization strategy appears dataset-specific. Critical experimental details are missing, including computational costs, sensitivity analyses, and proper investigation of the significant overfitting observed in Figure 4. The theoretical analysis, while strong in some areas, lacks finite-sample convergence rates, optimization landscape analysis for the min-max game, and investigation of whether the n+1 hidden qubit bound is tight.

### Questions
Lemma 1 and Remark 1: The efficiency claims rely on classical simulation algorithms from prior work (Nest, 2009). More detail on the polynomial dependence on n, m, and ε would strengthen these claims, particularly the hidden constants.

Equation (14): The log-derivative trick for gradient estimation is mentioned but the variance of this estimator isn't analyzed. High variance could make the adversarial training unstable.

Section 5: The worst-case analysis (Lemma 6) constructs pathological distributions with exponentially decaying characteristic functions, but it's unclear how frequently such distributions arise in practice or what structural properties make distributions amenable to MMD-based training.

Overfitting in Figure 4: The appendix shows adaptive kernels achieve much lower training TVD but similar or worse test TVD compared to Gaussian kernels, indicating serious overfitting. This fundamentally undermines the practical utility claims but receives only brief mention.

### Soundness
3

### Presentation
3

### Contribution
3
