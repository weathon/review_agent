# An Efficient Quantum Classifier Based on Hamiltonian Representations

- Avg Score: 3.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 3, 3, 3

## Abstract
Quantum computing shows great potential for expanding the range of efficiently solvable problems. This promise arises from the advantageous resource and runtime scaling of certain quantum algorithms over classical ones. Quantum machine learning (QML) seeks to extend these advantages to data-driven methods. Initial evidence suggests quantum-based models can outperform classical ones in terms of scaling, runtime and generalization capabilities. However, critics have pointed out that many works rely on extensive feature reduction or use toy datasets to draw their conclusions, raising concerns about their applicability to larger problems. Scaling up these results is challenging due to hardware limitations and the high costs generally associated with encoding dense vector representations on quantum devices. To address these challenges, we propose an efficient approach called Hamiltonian classifier inspired by ground-state energy optimization in quantum chemistry. This method circumvents the costs associated with data encoding by mapping inputs to a finite set of Pauli strings and computing predictions as their expectation values. In addition, we introduce two variants with different scaling in terms of parameters and sample complexity. We evaluate our approach on text and image classification tasks, comparing it to well-established classical and quantum models. Our results show the Hamiltonian classifier delivers performance comparable to or better than these methods. Notably, our method achieves logarithmic complexity in both qubits and quantum gates, making it well-suited for large-scale, real-world applications.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
This paper argues that previous NISQ quantum machine learning algorithms generally require a large number of computational resources, which may not be suitable for NISQ devices. To address the data-encoding problem, the authors encode classical data into a physical Hamiltonian, referred to as the Hamiltonian classifier. To demonstrate the power of the Hamiltonian classifier, they numerically test their model on several popular classical datasets, including text and image classification tasks.

### Strengths
From a high-level perspective, this paper is well-written, clearly introducing the research background and presenting their results. Meanwhile, the authors have put significant effort into the numerical simulation section, where they compare various related quantum-classical methods across several datasets.

### Weaknesses
(1) One of the main concerns is that the "Hamiltonian Classifier" has been proposed and studied in several papers, such as [S. Jerbi et al., Shadows of Quantum Machine Learning, Nat. Comm., 2024] and [Y. Song et al., A Quantum Federated Learning Framework for Classical Clients, Sci. China Phys. Mech. Astron. (2024)]. However, this paper does not cite these highly relevant works. As a result, the authors' contributions are significantly diminished, especially the claim in Sec. 3, page 4, "To the best of our knowledge..."

(2) As a high-level conference (ICLR) in the field of machine learning, we expect to see more surprising results in quantum machine learning. The authors still utilize the standard classical-quantum workflow, despite encoding the data into a physical Hamiltonian. While many papers adopt this framework and numerically benchmark their methods on some datasets, such research contributes very little to quantum computation theory and the quantum machine learning community, particularly at this stage. This is supported by recent findings: it has been shown that many classical-quantum QML methods (including the authors') are classically simulable when the model does not suffer from the barren plateaus phenomenon [A. Angrisani et al., Classically Estimating Observables of Noiseless Quantum Circuits, arXiv:2409.01706]. Furthermore, when the QML algorithm is limited to a 2D architecture, all constant-depth (or constant evolution time) quantum-classical approaches can be classically simulated [S. Bravyi et al., Classical Algorithms for Quantum Mean Values, Nat. Phys., 2021]. From this perspective, it appears that the Hamiltonian classifier method may not provide a clear quantum advantage.

### Questions
Here are some minor questions:
(1) On Page 6, authors claimed that they can randomly define $p$ Pauli strings, and decompose the data matrix $H_{\phi}(\tilde{x})$ onto the sampled Pauli basis. Then, is there any creteria on selecting these Pauli strings? What is the scaling of the parameter $p$, and what is the relationship between $p$ and the power of Hamiltonain classifier model (such as generalization error upper bound or the effective dimension)? 
(2) In equation 10, $\alpha_i$ has an exponentially small factor $2^{-n}$. Does this factor cause the measurement accuracy to become exponentially small, leading to a large amount of measurement overhead?
(3) In Table 1, Cong et al. (2019) do not utilize the QCNN to solve a classical task; instead, they predict the quantum phase transition problem. Is it fair to compare Cong et al. (2019) to the authors' work on a classical task?
(4) In Tables 2 and 3, it is observed that the proposed methods (HAM, PEFF, and SIM) do not outperform all the listed methods. Given these facts, what is the advantage of the proposed Hamiltonian classifier method?

### Soundness
3

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
Authors provide a new method to implement variational quantum circuits (VQCs) that can be used for machine learning tasks using quantum hardware. This method achieves a logarithmic scaling in both qubits and quantum gate counts, while having worse sample complexity (as presented in Table 1). They then numerically simulate their quantum algorithm on text and image classification, and achieve results that are mostly better than other forms of VQCs and on-par with classical neural networks (such as MLPs and CNNs).

### Strengths
This paper is clearly written and scientifically sound. The improvement in gate complexity is substantial, since no other methods achieve logarithmic scaling for these quantities.
I believe the paper provides a good advance in the field of variational quantum circuits.

### Weaknesses
The main critique of this paper is that the method achieves mostly the same performance than very simple classical models, such as logistic regression, MLPs and CNNs, while being based on having access to ideal quantum hardware which is not readily available (and will only be made available in the long-term). Therefore, the main value of the paper is an algorithmic advance for VQCs (and in how data is encoded into the quantum computer), but not for the global field of machine learning itself.
Simulations that include noise could be useful, which hopefully would show that even on noisy quantum hardware the results are not significantly altered.
The benchmarks are somewhat sparse, and some of them include results that have 100% test accuracy for many methods, so we cannot know which is better. Considering more difficult datasets would be more interesting.
The maximum number of qubits considered is $n=10$ if I am not mistaken, which is significantly smaller than what can be simulated with even small computational resources ($n=20$ is feasible in the noiseless setting).

### Questions
In the complexity analysis, section 3.6, you mention that your method incurs an additional $1/\epsilon^2$ cost. Is this the same for other methods presented in Table 1?

### Soundness
3

### Presentation
3

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
This paper presents the Hamiltonian classifier, a quantum machine learning approach that enhances the data encoding by mapping inputs to Pauli strings, and provides related proof-of-principle experiments to demonstrate the advantages.

### Strengths
it is well-written and clearly presents the concept and idea.

### Weaknesses
1. the similar idea was proposed in Post-variational quantum neural networks (https://arxiv.org/pdf/2307.10560)
2. missing some theoretical analysis of the performance of the proposed model.

### Questions
1. How to choose the {${P_j}$}$_{j=1}^p$ in Eq.12?
2. is there any theoretical analysis that the generalization error is bounded with respect to the number of finite terms $P$?
3. instead of using parametrized quantum circuit, whether is it enough to only optimize the the parameters in measurement (parametrized hamiltonian? like https://arxiv.org/pdf/2307.10560)?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
Summary: In this paper, the authors put forward a new quantum machine learning (QML) classifier. Their new approach leverages the variational quantum eigensolver algorithm to find a parametrized quantum circuit that classifies inputs based on their expectation value. Underlying their new approach is their idea to embed input states within a Hamiltonian.

### Strengths
Strengths: The paper is exceptionally well-written. The exposition is clear, the protocol is well explained, and the background well-done. The idea to embed states in a Hamiltonian and then leverage VQE is interesting as well.

### Weaknesses
Weaknesses: The article has three main weaknesses.
   - No theoretical performance guarantees: There are no theoretical results guaranteeing any a super-polynomial improvements over classical methods. 
   - Weak results on simulated data: The new method is routinely outperformed by other methods including logistic regression on both text datasets. Sure, some of the competitors require more parameters, but none of the other models are honestly that big. These results would be far more compelling if there was a clear super-polynomial advantage with the new method.
   - No discussion of robustness to noise: This is a big one. There’s no discussion or evaluation of the methods robustness to noise. Unless you’re proposing a fault-tolerant algorithm, you need to discuss noise and compare your method’s performance in the presence of NISQ-era levels of noise to classical competitors.

### Questions
What happens to your method’s performance when it is run on noisy quantum hardware? 

How resource-intensive is it to make your method fault-tolerant?

Why did you not try HAM and PEFF on the Fashion-MNIST dataset? Apologies if this was already covered, but I struggled to find it.

### Soundness
2

### Presentation
3

### Contribution
2
