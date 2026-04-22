# Amplitude-based Input Attribution in Quantum Learning via Integrated Gradients

- Avg Score: 3.00
- Decision: Reject
- Scores: 4, 4, 4, 0

## Abstract
Quantum machine learning (QML) algorithms have demonstrated early promise across hardware platforms, but remain difficult to interpret due to the inherent opacity of quantum state evolution. Widely-used classical interpretability methods, such as integrated gradients and surrogate-based sensitivity analysis, are not directly compatible with quantum circuits due to measurement collapse and the exponential complexityof simulating state evolution. In this work, we introduce HATTRIQ, a general-purpose framework to compute amplitude-based input attribution scores in circuit-based QML models. HATTRIQ supports the widely-used input amplitude embedding feature encoding scheme and uses a Hadamard test–based construction to compute input gradients directly on quantum hardware to generate provably faithful attributions. We validate HATTRIQ on classification tasks across several datasets (Bars and Stripes, MNIST, and FashionMNIST).

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces HattriQ, a novel framework for computing input attribution scores in QML models. HattriQ adapts the classical Integrated Gradients method to the quantum setting, specifically for models using amplitude embedding, an encoding scheme that allows for an exponential number of input features relative to the number of qubits. The core of the method is a quantum-native circuit construction based on the Hadamard test to compute the exact gradients of the model output with respect to the input features directly on quantum hardware, without requiring access to or simulation of the internal quantum state. The method is evaluated across datasets such as bars and stripes, MNIST and FashionMNIST.

### Strengths
1. This work proposes to use quantum-native Hadamard tests to evaluate gradients of the cost function w.r.t. input features. It primarily targets amplitude encoding, but also generalizes angle encoding strategies.

2. To enhance scalability, the authors propose a parallelization strategy utilizing multiple ancilla qubits, which mitigates the linear scaling of circuit evaluations by enabling the concurrent computation of numerous gradient components

3. Comprehensive experiments are performed on diverse benchmark datasets.

### Weaknesses
A significant weakness of the proposed method lies in its fundamental scalability, which stem directly from its targeting of amplitude encoding. While this encoding allows for an exponential data capacity in principle, the attribution process requires computing a gradient component for each of the exponentially many amplitude features. Although the multi-ancilla parallelization strategy offers a theoretical mitigation, it necessitates a number of ancilla qubits that scales linearly with the number of features to be computed concurrently, presenting a substantial resource demand.

### Questions
No.

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
4

### Summary
The paper proposes HATTRIQ, a general-purpose framework for computing amplitude-based input attribution scores in circuit-based QML models. The approach adapts the integrated gradients (IG) method, used in classical deep learning, to quantum models by leveraging Hadamard test constructions and parameter-shift rules to estimate gradients directly on quantum hardware.

HATTRIQ supports amplitude embedding, a popular encoding strategy in QML, and the authors demonstrate its use in binary classification tasks on benchmark datasets such as Bars & Stripes, MNIST, and Fashion-MNIST. They also introduce a gradient-parallelization strategy to accelerate computation and report visualizations of pixel-level attributions for the tested datasets.

### Strengths
Relevance and motivation: The paper addresses a timely problem, explainability in quantum ML models, which is increasingly important as QML gains traction across platforms.

Methodological soundness: The derivations using Hadamard tests and parameter-shift rules are well-founded and clearly described.

Implementation details and scope: The authors validate their approach on multiple datasets and provide extensive visual examples, showing an effort to bridge theory and practice.

Parallelization component: The inclusion of a gradient parallelization scheme makes the approach more computationally feasible on current simulators or small hardware setups.

The authors provide mathematically valid constructions for gradient estimation using Hadamard tests, and the implementation is commendable. However, interpretability in quantum systems requires more than visual gradient maps—it demands explanations that connect attribution scores to physical quantities (e.g., qubit sensitivity, entanglement contribution, or measurement basis relevance). These links are missing. As presented, HATTRIQ shows that integrated gradients can be computed, but not that they are useful.

### Weaknesses
Unclear interpretability value: The main issue lies in the semantic meaning of the attribution results. While the paper technically computes integrated gradients on quantum amplitudes, it is never made clear what positive or negative attributions signify in the quantum context. In the classical setting, the sign and magnitude of IG scores correspond to each input’s contribution toward or against a class prediction. However, for amplitude-based encodings, the physical interpretation of these signed contributions is not explained or demonstrated convincingly.
Lack of interpretive consistency across examples: In Figures 1–3, there is no clear correspondence between the sign and the spatial pattern of the attributions and visually salient regions in the input data. For example, in MNIST digits, one could expect attributions to align with stroke regions or empty background, but this pattern is inconsistent. Similarly, for Fashion-MNIST, attributions do not correspond to intuitive parts of the object (e.g., bag body, handles, or trouser legs). As a result, it remains unclear what a practitioner can learn from these attributions.
Currently, it is impossible to judge whether the quantum attributions are faithful or simply artifacts of circuit sensitivity.

Conceptual disconnect: The paper demonstrates how to compute IGs for quantum models, but not why this is useful or what new insight it offers. Without a clearer discussion of the interpretability objective—e.g., identifying relevant qubits, detecting encoding bias, or assessing circuit sensitivity—the proposed method risks being a purely formal adaptation of a classical technique. 

Missing discussion of computational cost and noise sensitivity:
The paper does not analyze the measurement cost required to compute integrated gradients for different circuit sizes. Since IG involves multiple interpolated evaluations per input dimension, the total number of circuit executions could scale rapidly. Moreover, there is no discussion on how hardware noise or shot fluctuations affect the attribution stability. Given that the authors emphasize NISQ feasibility, a quantitative noise analysis or measurement budget estimation is essential to assess practical viability.

While the paper is technically correct and well written, it falls short in demonstrating the practical or conceptual value of amplitude-based integrated gradients for interpretability. The adaptation of a classical method to the quantum domain is interesting but insufficiently motivated and only superficially evaluated. Without a clear explanation of what the computed attributions mean or how they enhance understanding of QML models, the contribution remains limited in impact.

### Questions
A clearer explanation of what is being attributed—probability amplitude, measurement outcome probability, or circuit parameter sensitivity—would greatly help the reader.

Quantitative comparisons with classical IG attributions for hybrid models (e.g., classical-quantum classifiers) could strengthen the validation.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors' core contribution is an adaptation of the classical Integrated Gradients (IG) method to the quantum setting. The method is built on two key technical pillars:

A formal derivation (Lemma 3.1) for the gradient of a QML model's expectation value with respect to the individual (real or complex) amplitudes of the input state. A hardware-native circuit construction (Theorem 4.2) that uses the Hadamard test to compute these exact gradients on quantum hardware , thereby avoiding measurement collapse and the exponential cost of classical simulation.

The paper also proposes a multi-ancilla parallelization technique to accelerate the computation of gradient components. The HATTRIQ framework is validated on several image classification datasets (Bars and Stripes, NIST, MNIST, FashionMNIST), demonstrating high-fidelity attribution maps that highlight salient input features.

### Strengths
The demonstration that the method is robust to low shot noise (Fig. 5)  is a major practical strength. It suggests that HATTRIQ could be viable even on near-term devices where measurement budgets are constrained.

### Weaknesses
- The paper's primary evaluation relies on "ideal error-corrected hardware" and only analyzes the impact of shot noise (statistical sampling error). The proposed Hadamard test circuit (Sec 4.2) requires implementing deep, controlled unitaries, including the entire model $C-(U^\dagger O U)$. This type of circuit is known to be extremely sensitive to hardware/gate noise and decoherence, a much bigger challenge on NISQ-era devices. 

- The paper acknowledges that "generic controlled operations can incur additional circuit overhead" but does not analyze this critical bottleneck. The gate complexity of compiling $C-V(x)$ and particularly $C-(U^\dagger O U)$ will be the true scaling limit of this method. A discussion of the compilation overhead (e.g., in terms of CNOT count) is necessary for a complete picture of the method's "scalability."

- The attribution results are contingent on the choice of a "blank image (0 for all pixel values)" as the baseline $x'$. In classical IG, the baseline choice is a known, sensitive parameter. A zero-vector is an "off-manifold" point that may not be a meaningful reference. The paper would be strengthened by a discussion of this choice or a brief exploration of alternative baselines (e.g., an average image from the dataset)

### Questions
1. Could the authors provide a more concrete analysis of the gate complexity overhead required to implement the controlled unitaries, specifically $C-V(x)$ and $C-(U^\dagger O U)$? This seems to be the primary scalability bottleneck.

2. Can the authors comment on the expected robustness of HATTRIQ to hardware/gate noise? The proposed Hadamard test circuit seems particularly vulnerable to decoherence, which could severely degrade the fidelity of the inner product estimation.

3. The choice of a zero-vector baseline  is simple, but often problematic in classical IG. Have the authors considered other baselines (e.g., averaged images, blurred images) and how this might affect the attribution maps?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
The paper introduces a new interpretability technique for a subclass of quantum machine learning (QML) methods. The interpretability technique, called HattriQ, is designed for QML models that use parametrized quantum circuits. HattriQ computes attribution scores for input features by extending the populare integrated gradient framework from classical ML. The paper's contributions are as follows:
1. HattriQ, the aforementioned interpretability technique.
2. A simple-to-implement technique for calculating the gradients used by HattriQ on quantum hardware.
3. A proposal for computing attribution scores in parallel on large quantum processors.
4. Several demonstrations on simulated noise-free quantum processors.

### Strengths
1. Interesting Technique: The paper presents a novel method (HattriQ) for computing input attribution scores for quantum circuits, which is an intriguing approach in the field of quantum machine learning (QML).
2. Theoretically Sound Method: The method is theoretically sound and extends a popular classical technique, which adds value to the existing body of knowledge.
3. Demonstrations on Classic ML Datasets: The paper includes several demonstrations on well-known classical machine learning datasets (e.g., Bars and Starts, MNIST, FashionMNIST), showcasing the applicability of the method, at least on error-free quantum processors.
4. Growing Field: The focus on interpretable QML aligns with the increasing interest in making AI models more understandable, which is crucial for broader adoption.
5. Acknowledgment of Challenges: The paper correctly identifies the lack of interpretability methods as a barrier to AI adoption, which is an important consideration for the future of QML.

### Weaknesses
Since the paper performed demonstrations on simulated noise-free hardware, I will focus my analysis on HattriQ's relevance to fault-tolerant quantum computing (FTQC).

**Major weaknesses**
1. Scalability Challenges: The technique requires as many ancilla qubits as there are features, negating the space savings typically associated with QML.
2. Implementation Challenges: The method necessitates complicated controlled gates, with no clear guidance on how to implement these gates on FTQC systems or reckoning with the overhead.
3. Limited Long-Term Utility: The paper fails to reckon with QML's uncertain future in FTQC. At the moment, QML is not a serious application for FTQC without significant technical advancements (due to overhead, difficulties with loading classical data, lack of realizable advantages, and barren plateaus), which raises concerns about its practical relevance.

**Minor weaknesses**
1. Inadequate Support for Some Claims: The references in the introduction do not adequately support the claims made, particularly regarding the potential speedups of QML algorithms over classical methods.
2. Insufficient Background Information: The quantum states and gates background provided may not adequately prepare readers unfamiliar with QML, lacking explanations of key concepts like observables and parameterization of gates.
3. Unclear Accuracy Metrics: It is unclear what the "accuracy scores" in the paper refer to. Are they the accuracy scores of the QML models or of HattriQ?

### Questions
How robust is HattriQ to device noise?

Were the trained QML classifiers any good? 

Is there a way around using complicated controlled gates to compute the input feature gradients?

### Soundness
3

### Presentation
3

### Contribution
2
