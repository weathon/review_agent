# Quantum Generator Kernels

- Avg Score: 4.67
- Decision: Reject
- Scores: 6, 4, 4

## Abstract
Quantum kernel methods offer significant theoretical benefits by rendering classically inseparable features separable in quantum space. Yet, the practical application of *Quantum Machine Learning* (QML), currently constrained by the limitations of *Noisy Intermediate-Scale Quantum* (NISQ) hardware, necessitates effective strategies to compress and embed large-scale real-world data like images into the constrained capacities of existing quantum devices or simulators.  To this end, we propose *Quantum Generator Kernels* (QGKs), a generator-based approach to quantum kernels, comprising a set of *Variational Generator Groups* (VGGs) that merge universal generators into a parameterizable operator, ensuring scalable coverage of the available quantum space. Thereby, we address shortcomings of current leading strategies employing hybrid architectures, which might prevent exploiting quantum computing's full potential due to fixed intermediate embedding processes. To optimize the kernel alignment to the target domain, we train a weight vector to parameterize the projection of the VGGs in the current data context. Our empirical results demonstrate superior projection and classification capabilities of the QGK compared to state-of-the-art quantum and classical kernel approaches and show its potential to serve as a versatile framework for various QML applications.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces the Quantum Generator Kernel (QGK), a novel quantum kernel method designed to overcome the high-dimensional data-embedding bottleneck in quantum machine learning. Instead of using fixed angle encoding, they propose constructing a complete set of Hermitian generators spanning the SU(2) lie algebra. These generators are then partitined into groups, with each forming a Hermitian operator. Data dependent features are then generate the corresponding unitaries for each group. The QGK is built on top of this input embedding.

### Strengths
1. This work introduces a novel data encoding paradigm that moves beyond state-based methods by constructing parameterized unitary operators derived from the Lie algebra su(2^n), offering exponential increase in representational capacity; variational generator group (VGG) is proposed by structuring these unitaries through systematic combinations of algebraic generators, 

2. Quantum generator kernel is defined on top of VGG with groups parametrized by a trainable linear projection; it supports standard kernel based methods like support vector machines (SVM). The parameters of the linear projection are optimized via Kernel-Target Alignment (KTA). 

3. The method is benchmarked across a range of datasets with simuated noise. The results suggest that it is robust and a promising approach.

### Weaknesses
1. While the paper presents the Quantum Generator Kernel (QGK) as its central contribution, the more fundamental innovation appears to be the Variational Generator Group (VGG) framework for data embedding. The kernel method itself builds upon well-established concepts in quantum machine learning.

2. The VGG approach is positioned as a departure from traditional angle encoding, but it can be more precisely characterized as a generalized, structured multi-qubit angle encoding that systematically covers the su(2^n) space. This clarification would provide better context within the existing literature on quantum data embeddings and more accurately represent the methodological advancement.

3. The grouping of generators into VGGs represents a critical hyperparameter that significantly impacts the method's expressivity and efficiency. The paper would benefit from a more rigorous theoretical analysis of the grouping strategy with empirical results.

4. Another mainstream approach for encoding high-dimensional data is to use a non-linear neural net for compression followed by simpler encodings; this seems to be a more suitable approach with existing hardware in the near future. It would be good if the authors could provide theoretical analysis of the specific advanatges of the proposed encoding compared to the existing appraoch.

### Questions
See the weakness. I will increase my score if my concern is addressed.

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
This paper propose a novel quantum kernel architecture. Specifically, firstly, the authors construct a set of quantum kernel generators and merge them into Variational Generator Groups (VGGs). Secondly, they introduce a linear feature extractor that is pre-trained to project high-dimensional input into a compressed generator space to improve kernel alignment. Finally, they employs Hamiltonian-driven unitaries with learnable generator weights, enabling expressive and scalable data encoding. They also provided relevant theoretical analysis.

### Strengths
1. The writing of this paper is good and easy to follow.

2. This paper has a certain degree of innovation, introducing a novel method for constructing quantum kernels.

3. The authors provide theoretical analysis and experimental verification of their own algorithm.

### Weaknesses
1. The authors are advised to conduct experiments in larger quantum systems (e.g. 20 qubits).

2. The results in Table 1 do not reflect the superiority of the quantum generator kernels (QGK). Because the dimension of moons, circles and bank are too small, it does not conform to the actual application scenario as we are in big data era. While for cifar10, the accuracy achieved by the QGK is clearly unacceptable. The reviewer suggested that the authors include additional experiments to illustrate the superiority of QGK over classical kernels.  

3. The time complexity of QGK is exponentially large, and it lacks good scalability. It is difficult to run the method on quantum devices with larger qubits, which greatly reduces the practicality and impact of this method.

4. What does VQC mean on page 6, line 317? 

5. For the experiment in Figure 2, the authors recommend increasing the number of circuit layers to 16 to reflect the advantages of the QGK. Because the [1] shows that the expressibility of Projected Quantum Kernel (PQK) [2] does not decrease significantly when the number of circuit layers is less than 8.

6. The reviewer is pleased to see the authors include PQK as a comparison algorithm in the entire experiment.

[1] Exponential concentration and untrainability in quantum kernel methods.

[2] Power of data in quantum machine learning

### Questions
Please see the weaknesses.

### Soundness
2

### Presentation
2

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
This paper proposes Quantum Generator Kernels (QGK), a new approach to quantum kernel methods aimed at efficiently embedding high-dimensional data into quantum feature spaces. The key idea is to use Variational Generator Groups (VGGs), which are sets of Lie algebra generators combined into parameterized Hermitian operators, to build expressive quantum feature maps. The authors use a classical linear feature extractor to compress the input into generator coefficients, which then weight the quantum generators to produce a quantum state. The kernel value is computed via state fidelity. This approach is highly configurable and aims to span the full Hilbert space expressivity. The authors provide theoretical analysis and empirical validation on synthetic and real-world (MNIST/CIFAR-10) datasets. QGK consistently outperforms classical and other quantum kernels, notably even under simulated hardware noise.

### Strengths
QGK directly tackles the practical and important issue of embedding high-dimensional data into quantum states, which is a major bottleneck for QML.
The use of Variational Generator Groups (VGGs) is a fresh and powerful idea, allowing for a learnable feature map rather than a fixed one. This is a commendable cross-disciplinary innovation.
The fact that QGK’s advantage persists under noise simulations is a major strength. This demonstrates robustness and suggests viability for NISQ devices, which is a significant step beyond many "ideal simulation" papers.

### Weaknesses
The method relies on a classical linear compression stage before the quantum kernel. This blurs the line of quantum advantage. It's unclear how much of the performance gain comes from the trained classical pre-processing versus the quantum kernel itself. The paper needs a stronger ablation study to disentangle these two contributions.
Experiments on MNIST/CIFAR-10 use small subsets (n=1000). This is insufficient to demonstrate scalability. Kernel methods are notoriously difficult to scale with the number of training samples (N), often requiring O(N^2) or O(N^3) complexity, and this critical weakness is not addressed at all.
Sensitivity to design choices (number of generator groups $g$, compression ratio $\gamma$) is not deeply analyzed. This makes it hard for a practitioner to know how to tune this model.
The comparison is mostly versus kernel baselines, not small neural nets.

### Questions
I suggest the authors clarify, in a sentence or two, how the classical compression is obtained and whether it is fixed. Furthermore, a more detailed ablation study is needed to disentangle the effects of this compression stage from the QGK's contribution.
Please comment briefly on whether you expect QGK to scale to larger training sets (e.g., full MNIST). The computational complexity with respect to $N$ (number of samples) seems to be the elephant in the room.
A short remark on how performance changes when you vary the number of generator groups would help readers judge tuning effort.
It would help to summarize (even approximately) the qubit count and circuit depth for your best MNIST/CIFAR setting.
A one-line comparison to a tiny MLP/CNN on the same 1k-sample split would help contextualize absolute accuracy.

### Soundness
2

### Presentation
2

### Contribution
3
