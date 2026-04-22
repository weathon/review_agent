# Hybrid Quantum-Classical Recurrent Neural Networks

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 4, 2, 4

## Abstract
We present a new hybrid quantum-classical recurrent neural network (QRNN) architecture in which the recurrent core is realized as a parametrized quantum circuit (PQC) controlled by a classical feedforward network. The hidden state is the quantum state of an $n$-qubit PQC in an exponentially large Hilbert space $\mathbb{C}^{2^n}$, which serves as a coherent recurrent quantum memory. The PQC is unitary by construction, making the hidden-state evolution norm-preserving without external constraints. At each timestep, mid-circuit Pauli expectation-value readouts are combined with the input embedding and processed by the feedforward network, which provides explicit classical nonlinearity. The outputs parametrize the PQC, which updates the hidden state via unitary dynamics. The QRNN is compact and physically consistent, and it unifies (i) unitary recurrence as a high-capacity memory, (ii) partial observation via mid-circuit readouts, and (iii) nonlinear classical control for input-conditioned parametrization. We evaluate the model in simulation with up to 14 qubits on sentiment analysis, MNIST, permuted MNIST, copying memory, and language modeling. For sequence-to-sequence learning, we further devise a soft attention mechanism over the mid-circuit readouts and show its effectiveness for machine translation. To our knowledge, this is the first model (RNN or otherwise) grounded in quantum operations to achieve competitive performance against strong classical baselines across a broad class of sequence-learning tasks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper presents a hybrid-classical quantum recurrent neural network. In the proposed architecture, the hidden state is the quantum state of a parametric quantum circuit (PQC). The parameters of the PQC, in turn, are controlled by a classical neural network that processes both the prior hidden state (or rather, a measurement taken from it) and the new input.
The paper provides an empirical study with extensive experiments on various datasets.

### Strengths
- Designing algorithms and architectures that allow to integrate quantum computing into machine learning is an important task. Thereby, the paper contributes to a relevant research area. 
- The experiments are carried out and described (mostly) very thoroughly, providing important experimental insights on the performance of QRNN architectures in comparison to classical architectures. 
- In the experimental study, the proposed architecture appears to perform well in comparison to classical architectures.  
- Overall, the paper is well-written and clearly presented.

### Weaknesses
- Important experimental details on baselines are missing: the precise choices for number of hidden layers, activation functions, etc. for the classical baseline models (RNNs, LSTMs) are not reported. While Dropout is applied for the QRNNs, it is not stated whether it is also used for the classical baseline models. Moreover, it remains unclear how exactly the network sizes / total number of parameters were chosen for the different experiments. This makes it impossible to reproduce experimental results and undermines the main claims of the paper. 

- Claimed outperformance not supported by experiments:  the paper states that QRNNs outperform standard LSTMs/RNNs on four of six tasks. However, this is not fully supported by the experiments: only in two of the experiments QRNNs appear to really obtain a significant improvement (Table 2). For the remaining experiments, this does not seem to be the case. For the first task (reported in Table 1), the difference between the RNN performance and the best QRNN is very minor, while the RNN uses fewer parameters (only 5K instead of 5.2K). For the task in Section 4.3, no table is provided, but the text states that LSTMs and QRNNs perform equally well. For the remaining two tasks LSTMs outperform QRNNs. This would mean that only in 2/6 tasks QRNNs outperform classical methods. 
Moreover, to allow for a full comparison of the considered architectures, it would be crucial to also report runtimes. 

- Only classical baselines used: several other recurrent quantum neural networks architectures have been proposed (Ubale et al. 2025, Yu et al. 2024a, etc. as discussed in Section 2), but the paper does not compare performance to any of them. This would be important for justifying why yet another QRNN architecture should be considered. In particular, this is important because in comparison to these works the novelty of the architecture is incremental (adding non-linearity to the circuits from Li et al. 2023 and Siemaszko et al. 2023)

- No code is available with the paper, which limits reproducibility of the experimental results. 

- Missing details on the methodology: the model architecture is never spellt out in full detail, which makes it not possible to reproduce the results. Figure 1(a) is stated as an example, but the paper does not state how exactly the unitary operator U looks like in general and which gates are used exactly. As for measurement, lines 260-262 state that "measurement outcomes are combined to form $z_t$, but does not state how exactly these measurements are combined. 

- Impact of measurement noise: Related to this point, it appears that the measurement operation in (3) would be prone to noise. This needs to be thoroughly addressed in the paper, as it may have a major impact on the performance. How is this addressed in the current implementation? If it is currently neglected, it would be highly important to assess its impact on the performance. 

- The discussion on norm preservation is misleading: only the norms of the quantum states are preserved, but this does not have any impact on or relation to the norms of the inputs.

### Questions
- Can you provide full experimental details regarding the baselines (use of dropout, activation functions, etc., see first point above)? 
- How sensitive are the baselines with respect to choices of activation functions? 
- According to which procedure did you select network sizes / hyperparameters / activation functions for the classical baselines?
- Can you add a table with results also for Section 4.3? 
- What happens in Table 1 if you increase the number of RNN parameters to 5.2K? 
- Can you provide insights on the use of alternative, previously proposed QRNN architectures?
- Can you provide code for your implementation? 
- Can you write out in detail the model architecture: how exactly is U defined and how does the measurement operator look like. 
- Can you assess the impact of measurement noise on the results? 
- The paper states "To preserve coherence across timesteps, we simulate mid-circuit measurements, allowing recurrent structure without collapsing the full quantum state, retaining the quantum memory throughout the sequence." It appears that this would create problems on real quantum hardware  - can you comment on this?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a hybrid quantum-classical RNN architecture in which the recurrent core is implemented as a PQC. At each timestep, mid-circuit measurements from the previous quantum state are combined with classical embeddings of the current input through a feedforward network, which outputs the next set of PQC parameters. The quantum state acts as the hidden state, evolving unitarily in a high-dimensional Hilbert space. The authors argue that this design offers a principled way to integrate unitary evolutions into recurrent architectures while preserving state norm and enabling per-timestep readouts.
The model is evaluated in simulation (up to 14 qubits) on standard benchmarks including sentiment analysis, MNIST, permuted MNIST, copying memory, and machine translation. The authors report modest accuracy improvements over classical RNN baselines.

### Strengths
Novel architectural concept: The idea of embedding quantum circuits within RNN recurrence steps is original and conceptually interesting, especially for sequence modeling.
Comprehensive benchmarking: The authors evaluate the model across diverse tasks, including language modeling and translation, showing its general applicability.
Mathematical consistency: The recurrent evolution is unitary by construction, automatically ensuring norm preservation—an elegant contrast to the ad-hoc regularization required in classical RNNs.
Practical transparency: The paper reports training times, qubit counts, and parameter sizes for both classical and hybrid models, which is commendable.

The work presents an interesting blueprint for integrating parametrized quantum circuits into recurrent models, but the evidence for impact is limited. The modest performance improvements could stem from added representational capacity rather than genuine quantum effects. The unclear separation between encoding and learning blocks, combined with incomplete training details, makes replication and interpretation difficult.
Given that all experiments rely on classical simulation and incur high computational costs, the results do not substantiate the paper’s implicit suggestion that such architectures are practical or advantageous in the near term.

### Weaknesses
Ambiguity in circuit design: The paper does not clearly distinguish between data-encoding and trainable parts of the PQC. From Figure 1A, the first layer of RY gates seems to correspond to data encoding, but this is not explicitly stated. Without this separation, it is difficult to evaluate what portion of the model’s expressivity truly arises from quantum effects. 

Modest empirical gains: Across tasks, improvements are small (e.g., +2\% accuracy in classification, BLEU 29.2 --> 31.9 for German–English translation). These gains do not convincingly justify the significant simulation cost or architectural complexity, especially when comparable improvements could be achieved by scaling classical models.

Limited discussion of simulation cost: Although the authors mention that training takes between 4–60 minutes per epoch (depending on task and qubit count), no wall-clock comparisons are provided against classical baselines. Given that all results are obtained on simulators rather than real quantum hardware, the practical feasibility and scalability of the method remain unclear.

Measurement and training details missing: The paper reports “24 measurements for 8 qubits” but does not specify whether this is per timestep or per sample, nor how parameter gradients were obtained. If the parameter-shift rule was used, the number of measurements would change, significantly impacting the total computational cost. Clarification is needed to assess efficiency and scalability.

Unclear goal between physics and simulation: It remains ambiguous whether the main objective is to demonstrate a computationally advantageous quantum model or to motivate unitary evolution as a conceptual analogue for classical nonlinearities. Without a clear experimental focus, the narrative risks oscillating between hardware-motivated and purely simulated reasoning.

Related work gap: Although several hybrid sequence models are cited, the omission of Quantum Deep Equilibrium Models is notable, as it directly addresses the encoding overhead and representational collapse issues that also affect this architecture.

While conceptually creative, the paper does not provide compelling empirical or theoretical evidence that hybrid quantum-classical RNNs outperform, generalize better, or offer unique interpretability compared to classical counterparts. The design lacks clarity regarding the role of data encoding and omits key implementation details needed to evaluate its feasibility on hardware. The work’s contribution lies more in architectural exploration than in demonstrated quantum advantage.

### Questions
In this type of architecture, reading and feeding back to the QCP could have a great cost; is it the case for this QML model too? 

Are canonical initialization schemes (e.g., amplitude or angle encoding) used for classical input embeddings?

Is the reported “24 measurements for 8 qubits” the number of observables per timestep or per sample? For training, how was gradient estimation done (e.g., via parameter-shift rule)?

How do the runtime costs per epoch compare quantitatively against the classical RNN baseline (e.g., same hardware and dataset)?

Will replacing the PQC with an orthogonal matrix RNN test whether unitarity alone explains observed gains?

### Soundness
3

### Presentation
3

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
In this work, the authors propose a hybrid RNN in which the recurrent core is a PQC and the classical part comprises a small feedforward network. The quantum state itself is treated as the RNN hidden state and evolved unitarily, which the authors claim naturally yields norm-preserving recurrence and, hence, better gradient propagation. To obtain per-timestep output but keep quantum memory, they simulate mid-circuit measurements/projective readouts and feed the measured values back to the classical controller for the next step. Their empirical results on six sequence tasks show that QRNN variants are competitive with RNNs, LSTMs, and scoRNNs of roughly similar parameter count.

### Strengths
1. The proposed quantum model is run across several/realistic sequence tasks instead of just MNIST or toy memory tasks. 

2. The authors give hyperparameters, optimizer, qubit counts, measurement sets, and even report variability across 50-100 runs in the appendix.

### Weaknesses
1. All results are obtained in TorchQuantum on GPUs, and there is no real hardware, no noisy simulator, no demonstration that the mid-circuit readout trick they rely on can actually be executed at the depth/width they need. The paper itself admits that it models mid-circuit measurement “as a limiting case” and that present toolchains are “less optimized” for hybrid recurrence.

2. The paper leans heavily on: “the PQC is unitary ⇒ norm-preserving ⇒ better gradients ⇒ better long-sequence learning.” But: (a) We already have unitary/orthogonal RNNs (Arjovsky et al. 2016; Jing et al. 2019) that give this without simulating a quantum circuit; (b) Their own best results require adding classical nonlinearities (ReLU, GELU, GLU) in the controller, so that the actual performance bump seems to come from the classical part, not the quantum recurrence. They even show QRNNLinear is clearly worse. So the “quantum as recurrent memory” story is blurred; (c) They do not show that the PQC is doing something strictly more complex than an (efficient) unitary RNN. 

3. Although the authors cite Bausch 2020, QRNN-like PQC recurrences, and QLSTM variants, the novelty is not tight compared with prior quantum-RNN / quantum-RL / hybrid quantum-classical NN.

### Questions
1. Can the authors show one nontrivial sequence task (longer than 400 steps, or multi-turn translation) where all classical baselines degrade but your QRNN still trains?

2. Can the authors show a side-by-side with a classical unitary/orthogonal RNN that has the same number of parameters as your PQC (counting gates) and the same measurement dimension?

3. Can the authors cost and prototype the mid-circuit measurement pipeline on any real backend (even for 4–6 qubits) to demonstrate that the feedback loop is not purely a simulator artifact?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents a novel Hybrid Quantum-Classical Recurrent Neural Network (QRNN). The work is one of the first demonstrations of a quantum-grounded model achieving competitive or superior performance against classical baselines across a broad and realistic (but toy) suite of sequence-learning tasks. The paper is well-written, but I do not have enough expertise to properly evaluate this paper and I made an educational guess.

### Strengths
- The model is evaluated on six diverse tasks (sentiment analysis, MNIST, pMNIST, copying memory, language modeling, machine translation) and is shown to be competitive with or outperform classical RNNs, LSTMs, and specifically designed orthogonal RNNs (scoRNN).

- The paper is generally well-structured and clearly written. I enjoyed the reading flow.

- The paper thoughtfully discusses the path to hardware implementation, acknowledging current simulation limits and proposing a realistic ancilla-mediated measurement scheme for future work. The choice of a simple, hardware-native PQC ansatz strengthens the practical relevance of the work.

### Weaknesses
- What is the advantage  of Hybrid Quantum-Classical Recurrent Neural Network? comparing to Classical RNN or  other Hybrid quantum NN (Like Hybrid quantum CNN if one can implement)? 
- It would be nice if the authors could explain something related to GPU consumption or efficency.
- It would be nice if the authors could visualize something related to the intermediate hidden states, like the state change in QRNN.
- To which perspective does the design of QCNN could  benefit the majority of  ICLR audiences. 
- The reviewer would appreciate it if the authors could provide some discussions on existing work like Li et.al. https://arxiv.org/pdf/2302.13812 and many other quantum-classifical hybrid models.

### Questions
see the weakness.

### Soundness
3

### Presentation
4

### Contribution
3
