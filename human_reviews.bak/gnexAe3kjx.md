# Quantum Neural Fields

- Decision: Reject
- Scores: 6, 1, 8, 5

## Abstract
This paper introduces a new type of neural field for visual computing with components compatible with gate-based quantum hardware or simulators thereof. Our Quantum Neural Field Network (QNF-Net) expects as input a query coordinate and, optionally, a latent variable value, and outputs the corresponding field value. QNF-Net includes a new feature map for classical data encoding and a parametrised quantum circuit. The proposed neuro-deterministic data encoding converts, into qubit amplitudes, an energy spectrum of the Gibbs-Boltzmann distribution corresponding to the learned problem energy manifold. We provide a theoretical analysis of the model and its components and perform experiments on a simulator of a gate-based quantum computer with 2D images and 3D shapes (and their collections as learnt priors) and compare results with several classical baselines. QNF-Net consistently outperforms the classical baselines with a comparable number of parameters and achieves faster convergence speed, therefore showing its potential quantum advantages, even for relatively large-scale problems compared to what has been demonstrated in quantum machine learning so far. We will release the source code to facilitate method reproducibility.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces QNF-Net, a quantum neural field network for efficiently encoding and representing 2D images and 3D shapes. Using a unique neuro-deterministic encoding method, QNF-Net maps classical data into quantum states, achieving faster convergence and higher accuracy than classical models. It performs well on visual tasks like image rendering and 3D shape reconstruction, highlighting potential quantum advantages in efficiency and scalability for large-scale visual data.

### Strengths
1. The paper presents a novel alternative to the classical neural field with the help of quantum computing. 
2. The paper empirically demonstrates that the method is superior to the classical counterpart, making it a promising direction to the field.

### Weaknesses
1. Consider the novelty of this method. Are there any considerations when applying this method to the current NISQ devices? No related experiments are shown in this work.
2. This paper does not show time—and space-related complexity, which further concerns the practicality of this algorithm on classical simulators and real quantum devices.

### Questions
1. What is the computational complexity of this method?
2. What are the computational resources required for this method?
3. What are the impacts of this method on the effect of noise in the quantum devices?
4. I haven't seen any experiments on real quantum devices; are there any relevant experiments?

### Soundness
3

### Presentation
1

### Contribution
3

---

## Human Reviewer 2

### Rating
1

### Rating Number
1

### Confidence
5

### Summary
This work presents a quantum version of the neural field named "QNF-Net". QNF-Net is claimed to have potential quantum advantages and can be applied to many scenarios, including robotics and 3D reconstruction. They introduce a method to translate classical data into quantum states, followed by a parameterized quantum circuit to train the neural field network.

### Strengths
Neural field network is a good model in CG/CV with growing interest. So it will be interesting to implement a quantum-enhanced neural field network.

### Weaknesses
1. Confusing logic.

To be honest, I do not think the logic of this paper is clear. For example, from line 083 to line 086, the discussion and remarks on the previous works, the challenges for quantum machine learning, and the methods of this paper ALL exist in THE SAME sentence, making it hard to figure out what are the authors' contributions.

2. Insufficient literature review.

In section 2 "Related work", the authors mention many other quantum machine learning algorithms for CV/CG. However, despite feeble criticism, I am afraid that I can not get any insight into other works. In particular, I can not find the difference between this paper and previous works. I think the authors should spend more time to explain also other works and the contribution of this paper. 

3. Unclear and undefined algorithm procedures.

There are many black-box algorithm subroutines that are unclear or undefined: For example, the so-called "inferred energy spectrum E" is claimed to play a central role in data encoding. However, the MLP to infer such energy is unclear: what is the MLP structure, what is the loss function, and what is the training procedure?

4. Unfriendly paper writing.

(1) Inconsistent terminologies and undefined notations. 

There are many inconsistent and informal notations so it is sometimes confusing for the reader to follow. For example, there are 73 "neural" and 3 "neuro", as well as 41 "circuit" and 4 "circuitry". I do not think that there is an explanation for the reader to distinguish between those confusing terminologies. Further, undefined notations can be found everywhere in this paper.

(2) Poor grammar and long sentences with no comma. 

There are many grammar and punctuation errors, making this paper hard to follow, e.g., "training neural fields can be computationally and resource-demanding" in line 045,  "All these applications became possible in recent years, as there has been a notable shift from hand-crafted priors, primarily based on heuristics, to learning priors in the form of neural fields directly from data Xie et al. (2022b), with multi-layer perceptron (MLP) with ReLU activation being one popular building block for such a neural field, in the early days." from line 040 to line 043, "as most other work do" in line 083.

### Questions
I do not have any further questions.

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper provides a novel approach to implicit neural representations by encoding scenes using parameterised quantum circuits. The method involves a coordinate and latent code encoding through a lightweight MLP (classical component) to generate a energy manifold, which is used to construct a quantum circuit (quantum component). Outputs are generated by sampling from the circuit. Experiments are run on a quantum simulator on image regression and occupancy field experiments, showing higher accuracy and improved convergence relative to baseline MLP models. The authors provide detailed theoretical descriptions of the quantum circuit and methodology, and perform a number of ablations to support their findings.

### Strengths
The main strength of the paper is in its novel approach for combining QML with implicit neural representations. Aside from one contemporaneous paper (QIREN Zhao et al. 2024 https://arxiv.org/abs/2406.03873, which seems to use a very different approach; and a preprint Quantum Radiance Fields, Yuan-Fu and Sun 2023 https://arxiv.org/abs/2211.03418), this appears to be one of the only papers on this topic. This is therefore significant as it extends a QML methodology to the important and rapidly expanding area of implicit neural representation research. 

The paper is clearly written and provides sufficient background details for both INR and QML literature to act as a great bridging paper for researchers in each area. Theoretical details are clearly described, and presentation in general is at a very high level both for written descriptions and figures (e.g. Figures 1, 2, and 4). Experiments are conducted on a variety of implicit neural representations (images and occupancy fields; image in-painting and shape completion), with appropriate metrics showing improvements relative to baselines. Ablations (qubits, block repetitions, encoder layers, and periodic activations) are appropriate and show a thorough evaluation.

### Weaknesses
While the paper presents a novel, well-described, and interesting approach to implicit neural representation training, there are a few weaknesses of the paper relating to experimental conclusions (e.g. parameter improvements compared to traditional INR approaches). 

Image experiments are conducted on CIFAR10 images. Very small MLPs can be used to fit these and larger images to high quality (see: Dupont et al. 2021 https://arxiv.org/abs/2103.03123, who fit larger Kodak images [768, 512, 3] with smaller networks of 2,000 - 15,000 parameters). This indicates that the classical energy encoding component of the network (15,000 parameters) may already be over-specified with respect to the target signal and doing the heavy lifting relative to the quantum parameters. Figure 8 shows relatively little difference with the number of qubits used (although it is highly sensitive to the number of classical layers, and partially sensitive to the number of block repetitions). This again calls into question whether the classical or quantum component doing more of the processing of the signal. It would be useful if the authors could discuss this and check two baselines: 1) A more restricted (in terms of parameters) classical energy encoder and MLP baseline; 2) To evaluate the quantum parameters (e.g. distribution of values at initialisation vs following training, to check whether these exhibit much change).

### Questions
The authors restrict their circuits to use only real-valued components to simplify the optimization problem. Could the authors provide any details about the impact of loosening this constraint? In addition, it would be useful to discuss in more detail the limitations of practically applying this method on a quantum computer rather than a simulator (e.g. backpropagation through the quantum circuit as described in Appendix A). 

The authors note that they encounter memory depletion issues due to storage of intermediate results (L375). Could the authors possibly describe this issue (e.g. what are the memory requirements of the method for the image / occupancy experiments, time required for convergence, etc)?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper introduces QNF-Net, a hybrid quantum-classical learning framework designed for visual computing tasks. QNF-Net combines an innovative neuro-deterministic encoding module, which maps classical data into quantum states via an energy inference process, with a parameterized quantum circuit (PQC) for efficient visual data modeling. Through experimental results on 2D images and 3D shapes, the authors show that QNF-Net outperforms classical MLP baselines in both accuracy and convergence speed, indicating potential quantum advantages in neural field representation.

### Strengths
The authors propose an original framework of applying quantum machine learning for neural field representations, with a novel approach for input data encoding. The experiments highlight the potential of QNF-Net for faster convergence and higher parameter efficiency relative to traditional MLPs, suggesting promising applications for quantum-enhanced neural fields. The results provide a compelling indication that quantum computing could benefit data-intensive fields, such as visual computing, where efficiency and scalability are critical.

### Weaknesses
The experimental section could benefit from further depth, particularly in baseline comparisons and scalability analysis. The chosen MLP baseline, while helpful for initial comparisons, lacks contextualization with more recent advancements in neural field representations. 

This limitation also weakens the argument for quantum advantage, especially given that QNF-Net is only tested with up to 6 qubits, a relatively small-scale quantum setup, and it is not clear why more qubits leads to worse performance. Additionally, the paper would be strengthened by discussing potential quantum hardware issues, such as noise and decoherence, as these are critical for real-world applicability. 

Finally, Sec. 4.4 on alternative energy inference designs feels disconnected from the core narrative and could be integrated more cohesively into the broader experimental results.

### Questions
1.	The proposed encoding method shares some similarities with neural network quantum states (e.g., Carleo and Troyer, Science 355, 602 (2017)), where a similar energy function is used to parameterize quantum states. Could the authors clarify the conceptual similarities and differences between these approaches?
2.	In Eq. (3), the normalization appears to only account for spatial dimensions but not the quantum state’s spin degrees of freedom. For a system with n qubits, summing over 2^n basis states would become infeasible as n grows. How is the normalization handled in practice for large n?
3.	Could the authors elaborate on how the proposed data encoding would be implemented on a physical quantum computer? Details on how the Gibbs-Boltzmann energy distribution could be prepared as qubit amplitudes would enhance understanding.
4.	What causes the spikes in the loss curves in Fig. 8? 
5.	Fig. 1(c) feels out of place and is not referenced until Section 4, which impacts flow and clarity. Additionally, Figs. 1, 2, and 3 contain overlapping information. A reorganization could help improve clarity and readability.

### Soundness
2

### Presentation
2

### Contribution
3
