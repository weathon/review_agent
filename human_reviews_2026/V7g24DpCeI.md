# Quantum Algorithm for Deep Neural Networks with Efficient I/O

- Decision: Reject
- Scores: 4, 6, 8, 2

## Abstract
A primary aim of research in quantum computing is the realization of quantum advantage within deep neural networks. However, it is hindered by known challenges in constructing deep architectures and the prohibitive overhead of quantum data I/O. We introduce a framework to overcome these barriers, designed to achieve an asymptotic speedup over the large input dimension of modern DNNs. This framework is based on the belief that a deep learning model can achieve similar performance when "rough" copies of the data are allowed, which is called the good-enough principle in this paper. Our framework enables the design of multi-layer Quantum ResNet and Transformer models by strategically breaking down the task into subroutines and assigning them to be executed by quantum linear algebra (QLA) or quantum arithmetic modules (QAM). This modularity is enabled by a novel data transfer protocol, Discrete Chebyshev Decomposition (DCD). Numerical validation reveals a pivotal insight: the measurement cost required to maintain a target accuracy scales sublinearly with the input dimension, verifying the good-enough principle. This sublinear scaling is key to preserving the quantum advantage, ensuring that I/O overhead does not nullify the computational gains. A rigorous resource analysis further corroborates the superiority of our models in both efficiency and flexibility. Our research provides strong evidence that quantum neural networks can be more scalable than classical counterparts on a fault-tolerant quantum computer.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a hybrid quantum-classical framework for deep learning, which combines Quantum Linear Algebra (QLA) modules and Quantum Arithmetic Modules (QAM). The authors demonstrate how these modules can compose larger architectures such as Quantum ResNet and Quantum Transformer (qResNet / qTransformer). Besides, a Discrete Chebyshev Decomposition (DCD) protocol is introduced to mitigate inter-layer I/O bottlenecks. The paper claims that, under this modular design, large-scale matrix operations can be executed with asymptotic complexity $O(\mathrm{polylog}N)$, offering a potential quantum advantage in both forward and backward passes.

### Strengths
1. The split between QLA for handling high-dimensional linear algebra and QAM for handling low-dimensional arithmetic and nonlinearities is elegant and well motivated.

2. The DCD protocol is a technically meaningful idea. By approximating inter-layer quantum states with a truncated Chebyshev expansion, the DCD reduces measurement and reconstruction cost from $O(d^2)$ (state tomography) to $\tilde O(r/\delta)$, where $r \ll d$. This approach provides a solution to one of the most practical barriers in quantum deep learning, i.e. the I/O bottleneck, which is also numerically validated in Fig. 5.

3. The fidelity–accuracy study in Fig. 6 demonstrates that model accuracy remains stable even when quantum state fidelity drops from 0.99 to 0.9, which is an empirical validation of the good-enough principle.

### Weaknesses
1. The first problem is about the oracle construction complexity (without QRAM). The central assumption that the block-encoding unitary $U_A$ can be constructed and called in $O(\mathrm{polylog} N)$ time is not generally valid. This holds only if $A$ has a structured form (e.g., sparse or low-rank), or there exists a QRAM-like quantum oracle access to matrix entries. However, for dense and dynamical matrices, such as the attention matrices in Transformers, the paper does not provide a polylogarithmic-efficient block-encoding construction. The claimed polylogarithmic gate complexity implicitly assumes an oracle that is itself expensive to build from classical parameters. Without an explicit construction of $U_A$ independent of QRAM, the claimed quantum advantage remains hypothetical.

2. During training, the QLA operates on dynamical matrices such as attention matrices, which are parameterised. Therefore, its block-encoding directly depends on these parameters. So every update in the training stage implies a change in parameters and the the need to rebuild or recompile the corresponding oracle circuit. This reconstruction cost can be potentially expensive to negates any claimed quantum advantage once full training dynamics are considered, since the oracle must be regenerated each iteration.

### Questions
1. How is $U_A$ constructed from classical weights without assuming QRAM or sparsity? What is the actual gate complexity for a dense attention matrix?

2. When $W_Q, W_K, W_V$ in a quantum transformer are updated, how is $U_A(W)$ adjusted? Is there any incremental update mechanism, or must the oracle be recompiled entirely?

3. Given that gradients are computed and stored classically, do you consider the claimed quantum advantage to apply only to inference? If so, could this distinction be made explicit?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a promising path toward scalable quantum deep networks. The proposed Discrete Chebyshev Decomposition protocol could be a potential enabler of deep architectures on future fault-tolerant quantum hardware

The proposed framework is based on the assumption that similar performance can be achieved by a deep learning method when "rough" copies of the data are given.

A novel data transfer protocol is proposed that allow breaking down the task into subroutines.

The major finding is that the cost required to maintain a target accuracy scales sublinearly with the input dimension.

### Strengths
- the finding that the cost required to maintain a target accuracy scales sublinearly with the input dimension
- clear motivation and relevance
- a novel data transfer protocol
- theoretical foundation

### Weaknesses
- the good-enough assumption (approximate inter-layer information) is plausible but under-theorized
- only numerical simulations are reporte in the experiments

### Questions
Could you discuss the sensitivity of you r proposal to the choice of truncation rank r?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper introduces a hybrid quantum-classical framework for designing deep quantum neural networks (QDNNs), addressing two of the field's most pressing challenges: the construction of deep architectures and the quantum data I/O bottleneck. The authors propose a novel data transfer mechanism, the Discrete Chebyshev Decomposition (DCD) protocol, shown to reduce overhead compared to standard tomography methods. The feasibility of these proposals is supported through concrete quantum implementations of ResNet and the Transformer. The paper is supported by a comprehensive suite of numerical experiments and resource analyses that validate its claims of favorable scaling and a tangible quantum advantage in specific, well-defined computational regimes.

### Strengths
The work is guided by the "good-enough" principle—arguing that perfect, high-fidelity state reconstruction between layers is unnecessary—provides a powerful justification for mitigating the I/O bottleneck. The modular architecture is an innovative solution to managing quantum overhead.

The proposed DCD protocol addresses the critical data transfer bottleneck. The numerical evidence for its sublinear scaling of measurement cost with input dimension is a significant result.

The paper support the conclusions thorough sufficient empirical validation. The authors provide not only a comparison of DCD against standard tomography but also a detailed resource analysis comparing their hybrid model to a fully QLA-based baseline.

### Weaknesses
The entire framework's efficacy, and particularly the success of the DCD protocol, is predicated on the core assumption that intermediate quantum states are highly compressible. While the numerical results strongly support this hypothesis for the tested models, its universality is an open question. It would significantly strengthen the paper if the authors discussed the potential failure modes of this assumption. 

The proposed algorithms are designed for a fault-tolerant quantum computer, placing them in a long-term research context. While a detailed resource compilation is beyond the scope of this work, the paper would benefit from a brief, order-of-magnitude discussion on the required hardware scale.

While the high-level architectural descriptions  are clear, the paper could be improved by providing more fine-grained implementation details for some of the key quantum modules.

### Questions
The entire framework's efficacy, and particularly the success of the DCD protocol, is predicated on the core assumption that intermediate quantum states are highly compressible. While the numerical results strongly support this hypothesis for the tested models, its universality is an open question. It would significantly strengthen the paper if the authors discussed the potential failure modes of this assumption. 

The proposed algorithms are designed for a fault-tolerant quantum computer, placing them in a long-term research context. While a detailed resource compilation is beyond the scope of this work, the paper would benefit from a brief, order-of-magnitude discussion on the required hardware scale.

While the high-level architectural descriptions  are clear, the paper could be improved by providing more fine-grained implementation details for some of the key quantum modules.

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes a hybrid quantum-classical framework for constructing deep quantum neural networks (QDNNs), such as Quantum ResNets and Quantum Transformers. The core idea is to separate computations into Quantum Linear Algebra (QLA) modules for large-dimension ($N$) operations and Quantum Arithmetic Modules (QAMs) for small-dimension ($d$) operations. To address the I/O bottleneck between layers, the paper introduces the "good-enough" principle, postulating that intermediate states can be imperfectly reconstructed. This is implemented via a novel "Discrete Chebyshev Decomposition" (DCD) protocol, a lossy compression scheme designed to be more efficient than full tomography.

### Strengths
This work presents a framework to overcome two major obstacles in quantum deep neural networks: (1) constructing deep architectures (e.g., multi-layer ResNets and Transformers) in the quantum domain, and (2) the prohibitive quantum data I/O or state-preparation overhead. The authors introduce the “good-enough” principle: that a deep learning model can still perform well even when given “rough” copies of the input data. They propose a modular architecture that splits computations into quantum linear algebra modules (QLA) and quantum arithmetic modules (QAM), enabled by a novel data-transfer protocol called Discrete Chebyshev Decomposition (DCD). Their numerical validation suggests that the measurement cost to maintain target accuracy scales sublinearly with input dimension (thus preserving quantum advantage by mitigating I/O bottlenecks). They also provide a resource-analysis arguing that, on a fault-tolerant quantum computer, such quantum neural networks could be more scalable than classical counterparts.

### Weaknesses
The "Good-enough" Principle is an Unproven Hypothesis: The paper's central claim is that a deep network can "achieve similar performance when 'rough' copies of the data are allowed" (Abstract). This is implemented by the DCD protocol, which truncates the Chebyshev spectrum, effectively keeping only $r$ low-frequency components ($r \ll d$).

Lack of Justification: This principle is "believed" (Abstract) or "posited" (Introduction) rather than proven. There is no theoretical guarantee that the high-frequency information discarded by DCD is irrelevant. For many tasks, this high-frequency data (e.g., fine textures, specific tokens) is precisely what distinguishes different classes.

Weak Analogy: The comparison to classical pruning/quantization is misleading. Those methods are applied with care, often require retraining to compensate for information loss, and do not typically involve discarding the entire high-frequency spectrum of a layer's output.

Flawed Complexity Analysis and Hidden Scaling: The paper claims its QLA-QAM hybrid (Fig 1d) avoids the multiplicative complexity of fully QLA-based networks (Fig 1c). However, the proposed solution (DCD) requires a measurement-and-re-preparation cycle at every layer.

Re-introduction of Multiplicative Cost: This very cycle is a known bottleneck that decoheres the state and introduces a multiplicative sampling cost that scales with depth. The overall cost to execute $L$ layers is not additive.

Unrealistic Assumption of Fault-Tolerance: The paper explicitly targets fault-tolerant quantum computers (FTQC) (Abstract, Sec 1).

This assumption sidesteps all of the real challenges in quantum computing (noise, coherence, qubit count). The proposals are purely theoretical and have no path to near-term implementation.

The QAMs, in particular, which rely on complex quantum arithmetic (adders, multipliers) to implement non-linearities (like QReLU), would require an enormous number of gates and an exceptionally high degree of fault tolerance, making them practically infeasible even on projected early FTQCs. The paper glosses over this immense resource cost.

### Questions
While the paper is ambitious and addresses a meaningful barrier in quantum deep learning (the I/O bottleneck), several concerns remain:

1. Empirical evidence mostly simulation-based and limited scope

The results are reported from simulations rather than real quantum hardware; thus actual overheads (error rates, decoherence, state-preparation overhead, readout latency) are not demonstrated.

The claim of sublinear measurement scaling is intriguing but might depend heavily on idealised conditions (noise-free, fault-tolerant assumptions). It is unclear how robust this behaviour is under realistic quantum hardware constraints.

It is not fully clear how “input dimension” is defined in the experiments (e.g., image size, sequence length, parameter count) or whether the datasets used reflect real large-scale deep learning problems (e.g., full-sized Transformers, very large inputs).

2. “Good-enough” principle needs stronger justification and clarity

The idea that approximate or “rough” data copies suffice is appealing but not deeply analysed: under what error bounds or fidelity reductions does performance degrade? What kind of “roughness” is acceptable?

Without a more quantitative sensitivity or ablation study on the fidelity/approximation error vs accuracy trade-off, the principle remains somewhat heuristic.

3. Architectural and algorithmic details are underdeveloped

The mapping of classical deep network layers (ResNet, Transformer) into quantum modules is described at a high level but lacks full transparency: How exactly are QLA and QAM modules defined for, say, attention in Transformers? What is the exact workflow or quantum circuit structure?

State preparation and encoding (which is often the main bottleneck) are only addressed by the DCD protocol; it would help to compare DCD explicitly against existing state-preparation or classical-quantum hybrid I/O protocols.

The resource analysis is valuable, but details such as qubit counts, gate depths, measurement sampling cost, ancillary qubits, and error mitigation strategies are not fully fleshed out.

4. Comparison against classical deep learning and classical I/O bottleneck solutions is weak

Since the paper claims a quantum advantage over classical counterparts, there should be clearer benchmarking or discussion of comparable classical I/O mitigation strategies (e.g., sketching, approximation, sparsification) and how the quantum approach compares or improves on them.

It is not shown how the quantum architecture performs (or would perform) in practice compared to highly optimised classical deep learning pipelines under I/O constraints. Without that, the practical significance is harder to assess.

5. Scalability to near-term hardware is unclear

The paper posits a fault-tolerant quantum computer scenario, but near-term quantum devices (NISQ) have major limitations: gate fidelity, coherence times, qubit connectivity, readout noise. The gap between the idealised framework and actual hardware readiness is not sufficiently addressed.

The work would benefit from a discussion of how one might implement DCD and the modular architecture on upcoming quantum hardware (or hybrid classical-quantum systems) and what the minimum requirements are.

**Relevant References for Inclusion**

Here are key references the authors should cite to situate their work properly and acknowledge related work in quantum deep learning, quantum neural networks, and quantum I/O/data-preparation concerns:

1. Quantum deep learning / quantum neural network foundational works

- Beer, K. et al. (2020). \textit{Training deep quantum neural networks.} Nature Communications, 11, 808.
- Zhao, R. \& Wang, Z. (2021). \textit{A review of quantum neural networks: methods, models, and applications.} arXiv:2109.01840.
- Valdez, F. \& Melin, P. (2022). \textit{A review on quantum computing and deep learning algorithms and their applications.} Applied Intelligence (via PMC).

2. Quantum I/O, data encoding and quantum machine learning scalability

- Peral-García, D. et al. (2024). \textit{Systematic literature review: Quantum machine learning.} Journal of Computer Languages, 79, 102–?.
- Kwak, Y. (2023). \textit{Quantum distributed deep learning architectures: Models and possibilities.} Journal of Data and Information Science (special issue).
- Stein, S. A. et al. (2021). \textit{QuClassi: A Hybrid Deep Neural Network Architecture based on Quantum State Fidelity.} arXiv:2103.11307.

3. Deep network/architecture adaptation to quantum settings

- Levine, Y., Sharir, O., Cohen, N. \& Shashua, A. (2018). \textit{Quantum Entanglement in Deep Learning Architectures.} arXiv:1803.09780.
- Pan, X. et al. (2022). \textit{Deep quantum neural networks equipped with backpropagation on a superconducting processor.} arXiv:2212.02521.

4. Data preparation / state-preparation and quantum linear algebra subroutines

- Childs, A. M., Kothari, R. \& Somma, R. (2017). \textit{Quantum algorithm for systems of linear equations with exponentially improved dependence on precision.}
- Kerenidis, I. \& Prakash, A. (2016). \textit{Quantum recommendation systems.}

### Soundness
2

### Presentation
2

### Contribution
2
