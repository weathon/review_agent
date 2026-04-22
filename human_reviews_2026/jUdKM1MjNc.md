# QMill: Quantum Data Generation for Effective and Efficient Quantum Machine Learning

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 4, 2, 6

## Abstract
Quantum machine learning (QML) has the potential to transform various fields, especially the ones that utilize quantum datasets, as QML tasks with quantum datasets have provable speedups. Yet, QML’s progress is limited by a lack of suitable quantum datasets for training and evaluation. While methods have been proposed to generate synthetic quantum datasets, these methods fail to accurately capture the entanglement properties necessary for effective generation of QML datasets. This lack of diverse and entanglement-rich data hampers the development and benchmarking of QML models. To address this, we present QMILL, a versatile quantum data generation framework that
emulates diverse classical and quantum data distributions with low circuit depth, producing entangled, high-quality dataset samples to support QML advancement.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper introduces QMILL, a low-depth algorithm that produces synthetic quantum data with an arbitrary target distribution of considerable entanglement (CE) across the generated quantum states. The algorithm starts from Haar-random product states, applies one of four shallow trainable ansatz circuits, and optimizes circuit parameters to minimize the total variation distance (TVD) between the generated CE histogram and the target one. The CE measurements are obtained through measurement surrogates and the optimization is done through dual annealing. QMILL also implements a SWAP-test -based diversity check to assure that the generated states differ from each other. 
The authors test the algorithm with target CE distribution obtained from classical (MNIST, Fashion-MNIST, CIFAR-10) and quantum (chemistry, soil moisture, dark matter) sources and artificial ones. The authors also demo a 3-qubit QNN trained on QMILL-generated CE features performing comparably under ideal vs. noisy settings.

### Strengths
- The problem is of interest and, unlike previous work, it bridges the lack of entanglement-rich datasets for QML by targeting full CE distributions rather than a single CE value.

- Adding a diversity check strengthens the approach

- The whole pipeline is simple and the presence of experiments gives useful evidence, even if they are not satisfactory (see weaknesses).

### Weaknesses
Although the above strength, I do not think the paper currently achieves the high standards required by ICLR. My main concern is the scalability of the algorithm with the number of qubits, under different aspects:

- First, I am not sure the surrogate measure for CE is indeed efficient for a large number of qubits. To compute NZP it is required to estimate the probability of outputting the bitstring 0^n, which becomes infeasible as n grows, as it amounts to rare-event estimation/post-selection.


- Furthermore, no claims are made on the performance of the algorithm (shot cost, optimization time, depth required) depending on the number of qubits.


- In addition to all of this, experiments are only provided for a small number of qubits (3/4), which casts doubts on the performance of the algorithm for larger numbers of qubits.

- Separately, the algorithm does not reproduce the right-skewed (right-wing) distribution, indicating a clear limitation.

### Questions
- Could the authors clarify theoretically how the cost of performing NZP scales with n, as well as the scaling of other quantities like shot cost and time complexity?


- Could the authors run experiments for a larger number of qubits (around 10) to assess the performance and the scaling with time of the algorithm?

### Soundness
2

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
The manuscript proposes a low-depth quantum data generation framework designed to mitigate the lack of suitable training data in quantum machine learning. The authors argue that existing synthetic data approaches fail to capture essential entanglement characteristics, limiting their relevance for QML tasks. By generating entangled, high-quality samples that emulate both classical and quantum distributions, QMILL aims to facilitate more representative benchmarking and improve the development and evaluation of QML models.

### Strengths
the work provide the extensive numerical evaluations on the proposed dataset.

### Weaknesses
1. The proposed method might suffer scalability issue for generating dataset for QML. for instance:
* as it select TVD as objective function, the computational complexity might grow exponentially as the system increase. 
* as it utilize the SWAP test for sample diversity validation, it would double the system size which is intractable for large system.
2. many typos such as in line 187, double cite; $\phi$->phi in line 310.

### Questions
1. In 295, the standardized data are reduced via PCA from $2^n-1$ feature to $n$ qubits and then encode them into amplitude. why does it not directly encodes the feature into amplitude, it also only need $n$-qubit quantum states to store the feature vector without information lose.
2. The paper mentions the use of amplitude encoding. However, efficiently implementing such encoding circuits typically incurs significant resource overhead for large systems. Could the authors elaborate on how scalability is maintained?
3. In line 304, what's the quantum-sensed workloads used for? are they the other two dataset mentioned in 299?
4. The manuscript claims that the proposed dataset offers advantages over the previous dataset with fixed CE. However, it is unclear where the supporting numerical evidence for this claim is presented. Could the authors provide or highlight the corresponding experimental results?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The authors present QMILL, a quantum data generation framework aimed at addressing the lack of suitable quantum datasets for quantum machine learning. Since most QML research currently relies on classical data, QMILL provides a way to generate synthetic quantum datasets that capture essential entanglement properties, especially concentratable entanglement. The framework uses low-depth, customizable quantum circuits with SWAP tests to efficiently produce high-quality, entangled samples across a range of concentratable entanglement values. The authors validate QMILL on multiple classical and quantum distributions, showing that it generates more realistic and diverse quantum data than existing approaches, thereby enabling better training and evaluation of QML models.

### Strengths
S1. The paper clearly lays out the data scarcity problem in QML and shows how QMILL tackles it by generating diverse, entangled datasets.

S2. The explanation of the low-depth circuits and SWAP-test validation is straightforward and shows why QMILL works well on current hardware.

S3. The dataset analysis is clear and convincing, showing good variability and robustness to noise.

### Weaknesses
W1. The overall impact is a bit unclear since the usefulness of synthetic quantum datasets is still limited and not fully established.

W2. The evaluation feels more like a proof of concept, with few benchmarks comparing models trained on QMILL data to other datasets.

W3. It’s not clear how well QMILL generalizes across different hardware setups or noise levels.

W4. The experiments only use a small three-qubit QNN, so broader testing would be needed to show real scalability.

W5. There’s no solid quantitative comparison against other synthetic dataset methods.

### Questions
Q1. How were the four ansatz designs chosen for comparison, what criteria guided the picks, and are there other families (e.g. hardware-efficient) you considered?

Q2. Why limit the experiments to a three-qubit QNN—hardware or training constraints? Could you scale to more qubits/deeper models and report results?

Q3. Beyond experiments, can you offer any theoretical guarantees on when QMILL-generated datasets help (e.g. generalization bounds) and where they might fail?

### Soundness
2

### Presentation
2

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
This paper addresses the problem of the scarcity of high-quality, representative quantum training data . The authors argue that QML's promise of speedups is contingent on operating on quantum data (i.e., data with entanglement), but the field is stuck using classical datasets due to this data gap. The paper identifies a key flaw in existing synthetic data generation methods: they typically target a single, fixed value of Concentratable Entanglement (CE), which is not representative of real quantum datasets that exhibit a distribution of CE values. To solve this, the authors propose QMILL, a low-depth quantum data generation framework. The core contribution is its ability to generate quantum data samples whose CE values, in aggregate, match a user-specified target distribution.

### Strengths
- Novelty of the Goal: The conceptual shift from targeting a single CE value (as in prior work ) to a full CE distribution is the key insight. This is a far more realistic and representative target for emulating real-world quantum data.

- Empirical Validation: The authors have been exceptionally thorough. They test their framework against 12 different distributions (4 synthetic, 8 real) . They design and compare 4 different low-depth ansatzes (Fig. 2, Fig. 9) . They explicitly check for mode collapse using SWAP tests (Fig. 8a) . They also run on both ideal simulators, noisy simulators, and real hardware (IBM Sherbrooke). This is a comprehensive evaluation.

### Weaknesses
- On the motivation of CE. The entire framework is predicated on the assumption that Concentratable Entanglement (CE) is the primary, sufficient proxy for "quantum-ness". While it's an important metric, it's an open question whether matching the CE distribution is sufficient to capture all the relevant correlations and structures of a quantum dataset that a QML model might exploit. The paper demonstrates this is a necessary step forward, but not definitively that it is the final step.

- Unusual Noisy Result: In Table 1, the QNN trained on "Noisy" data (84.8% accuracy) inexplicably outperforms the one trained on "Ideal" data (81.8%). The paper frames this as simple robustness , but it's a counter-intuitive result that is left unexplained.

- Ansatz-Dependent Performance: The results in Section 6.5 (Fig. 9) show that the choice of ansatz matters significantly, with A3 being the best all-rounder. This implies that a user must still engage in a trial-and-error process to find the right ansatz for their target distribution, rather than the framework being a single, universal generator.

### Questions
See weakness.

### Soundness
3

### Presentation
3

### Contribution
3
