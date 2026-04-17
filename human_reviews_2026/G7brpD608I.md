# Simulating Mixed State Dynamics to Enable Differentiable Quantum Architecture Search

- Decision: Reject
- Scores: 4, 2, 4, 4

## Abstract
Variational Quantum Algorithms (VQAs) are a promising approach to leverage Noisy Intermediate-Scale Quantum (NISQ) computers. However, choosing optimal quantum circuits that efficiently solve a given VQA problem is a non-trivial task. Quantum Architecture Search (QAS) algorithms enable automatic generation of quantum circuits tailored to the provided problem. Existing QAS approaches typically adapt classical neural architecture search techniques, training machine learning models to sample relevant circuits, but often overlook the inherent quantum nature of the circuits they produce. By reformulating QAS from a quantum perspective, we propose a sampling-free differentiable QAS algorithm that models the search process as the evolution of a quantum mixed state, which emerges from the search space of quantum circuits. The mixed state formulation also enables our method to incorporate generic noise models, for example the depolarizing channel, which cannot be modeled by state vector simulation. We validate our method by finding circuits for state initialization and Hamiltonian optimization tasks, namely the variational quantum eigensolver and the unweighted max-cut problems. We show our approach to be comparable to, if not outperform, existing QAS techniques while requiring significantly fewer quantum simulations during training, and also show improved robustness levels to noise.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces $\rho$DARTS, a differentiable quantum architecture search (QAS) algorithm based on the density matrix formalism. The authors' motivation stems from a theoretical observation: the original DARTS "relaxation" (a probabilistic sum of operations) is physically invalid for quantum state vectors. While prior work (e.g., qDARTS) addressed this by using Gumbel-Softmax to enable differentiable sampling, this paper proposes an alternative. It returns to the original DARTS "summation" idea by using density matrices, where a probabilistic sum (convex combination) is physically valid. This allows $\rho$DARTS to create a single mixed state representing the entire search ensemble, enabling a "sampling-free" optimization. A key benefit of this formalism is the natural ability to incorporate arbitrary noise models, such as the depolarizing channel. The authors present experiments on state initialization, VQE, and Max-Cut to demonstrate their method's efficacy.

### Strengths
1. Theoretically Sound Formulation: The core premise of the paper is well-founded. It correctly identifies that the density matrix formalism is the quantum-mechanically proper way to execute the original DARTS-style "relaxation" (i.e., a weighted summation over all possible operations), which is not possible with state vectors. This "sampling-free" approach is a clean and logical alternative to the sampling-based path taken by methods like qDARTS.

2. Native Noise-Aware Search Capability: A significant and practical advantage of the density matrix approach is the ability to naturally incorporate arbitrary quantum noise channels (as defined in Eq. 1). This is a clear benefit over state-vector-based methods, which cannot easily model incoherent, non-unitary noise like the depolarizing channel. The noise-robustness experiments (Fig. 5) are a strong point of the paper.

### Weaknesses
1. The Discretization Gap: The method optimizes the parameters $(\alpha, \theta)$ of a mixed state $\rho'$, which represents a probabilistic ensemble of all circuits. However, to obtain the final architecture, it applies a "hard" argmax operation (Algorithm 1, line 14) after training. This creates a "discretization gap" common to DARTS-like methods. There is no guarantee that the performance of the final, discrete circuit $\mathcal{A}^*$ will match the optimized performance of the mixed state $\rho'$.
2. Highly Incremental Contribution: The paper's methodological novelty is limited. It heavily borrows its entire experimental framework, including the "macro search" and "micro search" concepts, directly from prior work (qDARTS, Wu et al., 2023). The only substantial change is the replacement of the Gumbel-Softmax sampling component with the density matrix summation component. While this change is logical, it makes the paper feel highly derivative and more like an incremental follow-up rather than a novel contribution.
3. Failure to Address the Core QAS Challenge: The quantum computing field has evolved significantly since 2023. The core, unresolved challenge for all simulation-based QAS is the $O(2^n)$ exponential scaling wall, which restricts them to classically-trivial problems. The QAS field now needs fundamental breakthroughs that solve this scaling problem. This paper, while theoretically neat, fails to provide this. In fact, it regresses on the most critical axis by introducing an $O(4^n)$ computational cost, making it less scalable than its predecessors. This work does not fundamentally change the game for QAS, making it less attractive for the general ML community.

### Questions
No questions.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The author extended the qDARTS which is doing differentiable architecture search over noiseless unitary circuits, to search over density matrix. The author perform different experiments to validate their methods.

### Strengths
1. Th $\rho$-based search might be useful for noise-aware design
2. The author show extensive experiments
3. The paper is well written, and in good structure

### Weaknesses
1. The proposed method does not provide clear advantages. For the experiments in the paper, it is generally equivalent to searching with state vectors. Circuits are unitary and objectives are linear in the state. The gradients are equivalent: $\partial_\theta f=\partial_\theta\langle\psi| O|\psi\rangle=2 \operatorname{Re}\left\langle\partial_\theta \psi\right| O|\psi\rangle=\operatorname{Tr}\left[O \partial_\theta \rho\right]$.
2. A state vector stores $2^n$ complex amplitudes, whereas a density matrix stores a $2^n\times 2^n$ complex array. Memory and time therefore scale like $O(2^n)$ for state vectors and $O(4^n)$ for density matrices, an extra factor of $2^n$. 
3. Training the architecture is a classical differentiable program that does not require a quantum device. It is unclear whether this approach can find circuits that are classically hard to simulate.
4. The experimental results do not show definitive advantages over other methods, making it even harder to justify the extra factor of $2^n$.

### Questions
1. Can the authors comment on the scalability of their method?
2. Can the authors clearly state the advantages of their method over others?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a novel quantum architecture search algorithm based on the mixed-state formalism, where the search process is interpreted as the evolution of mixed quantum states. The mixture arises from randomness in the distribution over potential circuit elements. The authors benchmark their method and demonstrate that it outperforms the baseline, qDART, both with and without noise.

### Strengths
- The paper introduces a new differentiable quantum architecture search algorithm that is conceptually simple and elegant.


- Numerical results show strong performance improvements over qDART, and the algorithm exhibits notable robustness to noise, particularly in the noise probability range of 0.01–0.1.

### Weaknesses
- The experiments are conducted only in simulation, not on real quantum hardware. As a result, it remains unclear how ρDART would perform under more realistic noise conditions. Since the algorithm requires exponential time when simulated classically, its practical utility relies on implementation on real quantum hardware. Demonstrating strong performance under current NISQ hardware would therefore be a crucial validation.

- Moreover, when executed on real hardware, the method would likely lose its “sampling-free” advantage, as multiple experimental shots would be required to estimate the loss. It is also unclear how gradients would be computed in this setting.

- While the approach is technically sound, the conceptual advance over qDART appears limited, the key difference being the substitution of Gumbel-softmax with softmax. The contribution would be more compelling if supported by stronger and more rigorous experimental results.

### Questions
Can the proposed algorithm be extended to run efficiently on real quantum hardware, thereby avoiding the exponential simulation cost?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents a novel differentiable quantum architecture search (QAS) algorithm by modeling the search process as the evolution of a mixed quantum state. Compared to previous QAS algorithms, this new method has two prominent features: (1) it is "sample-free", in the sense that no quantum circuits need to be sampled in the learning process; (2) the mixed state formulation naturally enables the method to incorporate generic noise models, such as depolarizing channels, which is not supported in previous state-vector-based simulation. Numerical experiments on VQE and QAOA demonstrate that this new method requires significantly fewer quantum simulations, with an improved level of noise resilience.

### Strengths
- The formulation based on mixed states is novel in the literature of differentiable QAS, and it facilitates a more efficient exploration of the search space by eliminating the need to sample quantum circuits. 
- This framework allows the modeling of hardware noise, which is crucial for the implementation of VQAs on near-term quantum devices. 
- Numerical experiments show good performance for a range of standard tasks, including (entangled) state preparation, quantum chemistry (VQE), and Max-Cut problems.

### Weaknesses
- While the numerical experiments show promising performance in terms of energy errors or state fidelity, the runtime of this new $\rho$DARTS algorithm is not reported. In particular, an end-to-end efficiency measure (e.g., wall-clock time) does not appear to be taken into account. Therefore, it is hard to judge the practical value of this new method. 
- The scalability of this proposed method has not been fully discussed either. If this type of differentiable QAS can only be simulated using classical devices, it can not be scaled up to a few tens of qubits. However, even moderate chemistry/optimization problems may involve up to hundreds of qubits. At this scale, is this method still applicable?

### Questions
- How is the runtime of $\rho$DARTS compared to existing methods? Is it going to be significantly more time-consuming (to achieve comparable results)?
- Is it possible for this new differentiable QAS algorithm to generate better circuits for error mitigation (i.e., enabling an active search for noise-resilient variational circuits, instead of passive noise robustness)?
- In principle, can we implement a similar differentiable QAS using a quantum computer (for better scalability)? If not, what are the potential barriers?

### Soundness
2

### Presentation
3

### Contribution
3
