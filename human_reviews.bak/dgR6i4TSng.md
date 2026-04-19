# Quantum-PEFT: Ultra parameter-efficient fine-tuning

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
This paper introduces Quantum-PEFT that leverages quantum computations for parameter-efficient fine-tuning (PEFT). Unlike other additive PEFT methods, such as low-rank adaptation (LoRA), Quantum-PEFT exploits an underlying full-rank yet surprisingly parameter efficient _quantum unitary parameterization_. With the use of Pauli parameterization, the number of trainable parameters grows only logarithmically with the ambient dimension, as opposed to linearly as in LoRA-based PEFT methods. Quantum-PEFT achieves vanishingly smaller number of trainable parameters than the lowest-rank LoRA as dimensions grow, enhancing parameter efficiency while maintaining a competitive performance. We apply Quantum-PEFT to several transfer learning benchmarks in language and vision, demonstrating significant advantages in parameter efficiency.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a new formulation for low-rank based PEFT, based on quantum mechanical notation. 
The authors demonstrate that their ansatz using Pauli Parametrization of low-rank adapters provides a superior paramters-accuracy ratio in numerious contemprary benchmarks

### Strengths
I find the authors approach of unification of various low-rank adapter notations based upon quantum-mechanical notations interesting. 
The paper has strong numerical evaluations.

### Weaknesses
While I enjoyed reading most of the paper, a few things are missing in the explanation of the method, causing some neccessary revisions. 

- The paper describe the compute ansatz in eq. (2), although the notations is quite heavy for readers that are not familiar with the field. The ansatz governs the compute instructions of the forward pass of the neural network. However, the backward-pass, especially computing the gradient update of the parameters is not covered. Especially with the dense notation, an algorithmic description is nessecary to explain how the gradient update is performed and a discussion of its computational feasibility should be added. 

- Updating a tensor-factorization as Tensor-Trains or Tucker Decomposition naively may induces unexpected perils for the low-rank optimization. In particular (in combination with the lacking update scheme), it needs to be discussed if a gradient descend on the Factors of the proposed parametrization indeed decreases the overall loss function of the method.

### Questions
- How does your factorized update scheme relates to robust Riemannian optimization schemes on Stiefel Manifolds, e.g. as described in 
[1] for Tucker Tensors and in [2,3] for Matrix Factorizations of the Type USV (as in Adalora)?

- Is there an option to make the method rank-adaptive, as e.g. in [1,2,3] or AdaLora?

[1] Emanuele Zangrando, Steffen Schotthöfer, Gianluca Ceruti, Jonas Kusch, and Francesco Tudisco.
Rank-adaptive spectral pruning of convolutional layers during training. In Advances in Neural
Information Processing Systems, 2024.

[2] Steffen Schotthöfer, Emanuele Zangrando, Jonas Kusch, Gianluca Ceruti, and Francesco
Tudisco. Low-rank lottery tickets: finding efficient low-rank neural networks via matrix differential equations. In Advances in Neural Information Processing Systems,
2022. URL https://proceedings.neurips.cc/paper_files/paper/2022/
file/7e98b00eeafcdaeb0c5661fb9355be3a-Paper-Conference.pdf.

[3] Steffen Schotthöfer and M. Paul Laiu. Federated dynamical low-rank training with global loss
convergence guarantees, 2024. URL https://arxiv.org/abs/2406.17887.

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
3

### Summary
This paper proposes Quantum-PEFT, a novel parameter-efficient fine-tuning method inspired by quantum computing concepts. The core contribution lies in achieving logarithmic parameter scaling through Pauli parameterization while maintaining orthogonality via Stiefel manifold mapping.

### Strengths
- Novel theoretical framework combining quantum-inspired parameterization with PEFT
- Significant reduction in parameter count compared to existing methods
- Comprehensive experiments across multiple tasks and architectures

### Weaknesses
1. The paper lacks clear delineation of learnable parameters in the mathematical formulations. While extensive comparisons with LoRA are provided, the fundamental step of identifying and justifying which parameters are learnable is overlooked. This impedes understanding of the method's core mechanism.

2. While the parameter efficiency is well-demonstrated, the computational complexity analysis is insufficient:
- The introduction of Stiefel manifold mapping introduces additional computational operations beyond standard matrix arithmetic
- Limited discussion of practical computational bottlenecks
- Experimental results show no significant advantage in fine-tuning time efficiency
- Absence of detailed analysis on computational overhead

### Questions
* In Figure 5, which provides Intuitive illustrations of idea of Q-PEFT, is relegated to the appendix. Given its importance for understanding the method's foundations, especially for researchers from non-quantum backgrounds, innovations within this figure should be moved to the main text.

* The mathematical notation requires better organization and clearer presentation. A dedicated notation section would significantly improve readability and accessibility.

* What are the practical limitations and deployment considerations for real-world applications?

### Soundness
3

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
3

### Summary
This paper introduces Quantum-PEFT, a novel parameter-efficient fine-tuning method that uses quantum-inspired unitary parameterizations. The key innovation is using Pauli parameterization to achieve logarithmic scaling of trainable parameters with matrix dimensions, compared to linear scaling in traditional LoRA methods. The method is evaluated on language and vision tasks, demonstrating comparable performance to LoRA while using significantly fewer parameters.

### Strengths
1. The paper introduces a parameterization based on quantum circuits that require only (2L+1)log₂(N)-2L parameters, a significant improvement over LoRA's 2NK parameters.
2. The reduction in parameters is experimentally verified.
3. The authors proposed generalized RY and CZ quantum gates for arbitrary dimensions beyond power-of-2.
4. The proposed method seems robust under quantization.

### Weaknesses
1. Analysis of how L layers affect entanglement capacity is limited. There is no evaluation of entanglement entropy between layers.

2. The paper only focuses on benchmarking against LoRA/adapter variants on small models (such as GPT2). Benchmarks on larger (and newer) models such as LLaMA seem to be quite standard and more related to practical use cases. 

1. Some of the recently proposed PEFT methods are not adequately addressed/discussed (at least mentioned somewhere in the paper), e.g.

[1] https://arxiv.org/abs/2403.17919 (NeurIPS)

[2] https://arxiv.org/abs/2404.03592 (NeurIPS)

[3] https://arxiv.org/abs/2405.12130 

[4] https://arxiv.org/abs/2406.00132 (NeurIPS)

Both [3] and [4] seem to be able to model high-rank matrices, and [4] appears to be also based on quantum circuit.

### Questions
1. How does Eq. 4 work?
2. What is the optimal number of alternating entanglement layers L for different model scales?
3. For the quantum Shannon decomposition approach to handle non-power-of-two dimensions, how does the decomposition choice (N₁, N₂) affect model performance?
4. Does the proposed method have a large computational overhead from the entanglement layers?
5. How does the proposed method compare with other methods mentioned above? My understanding of the main difference between this method and [4] is the use of unitary and diagonal matrices. Is this correct?
6. The proposed method focuses on reduction in parameters. However, [2] and [4] claim both reduction in parameter and performance improvement. Does the proposed method use even fewer parameters?
6. How is the proposed method related to quantum machine learning?

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
3

### Summary
The paper introduces Quantum-PEFT, a quantum-inspired parameter-efficient fine-tuning (PEFT) framework for large language and vision models. Unlike traditional methods such as LoRA, which require low-rank adaptations of model weights, Quantum-PEFT uses quantum unitary parameterizations, resulting in a logarithmic scaling of parameters. Key contributions include the development of quantum-inspired modules using Pauli parametrization, significantly reducing the number of trainable parameters, while retaining competitive performance in various transfer learning benchmarks.

### Strengths
1. Quantum-PEFT’s application of quantum unitary parameterizations to PEFT is novel, differentiating it from conventional LoRA models.
2. It drastically reduces trainable parameters without performance loss and it is very important for the field, especially for resource-constrained training.

### Weaknesses
1. I'm not familiar with quantum ML, so it's a bit hard for me to understand the core concept. I think it's also hard to digest for someone who is not familiar with quantum ML. But this is fine because I don't think the intended audience of this paper includes someone not familiar with quantum ML.

### Questions
See above. It would be great if authors could add some self-contained introduction to the necessary quantum concepts used in the paper.

### Soundness
3

### Presentation
3

### Contribution
3
