# Fully Quanvolutional Networks for Time Series Classification

- Decision: Reject
- Scores: 5, 3, 3, 6, 6

## Abstract
Quanvolutional neural networks have shown promise in areas such as computer vision and time series analysis. However, their applicability to multi-dimensional and diverse data types remains underexplored. Existing quanvolutional networks heavily rely on classical layers, with minimal quantum involvement, due to inherent limitations in current quanvolution algorithms. In this study, we introduce a new quanvolution algorithm that addresses previous shortcomings related to performance, scalability, and data encoding inefficiencies. Specifically targeting time series data, we propose the Quanv1D layer, which is trainable, capable of handling variable kernel sizes, and can generate a customizable number of feature maps. Unlike previous implementations, Quanv1D can seamlessly integrate at any position within a neural network, effectively processing time series of arbitrary dimensions. Our chosen ansatz and the overall design of Quanv1D contribute to its significant parameter efficiency and inherent regularization properties. In addition to this new layer, we present a new architecture called Fully Quanvolutional Networks (FQN), composed entirely of Quanv1D layers. We tested this lightweight model on 20 UEA and UCR time series classification datasets and compared it against both quantum and classical models, including the current state-of-the-art, ModernTCN. On most datasets, FQN achieved accuracy comparable to the baseline models and even outperformed them on some, all while using a fraction of the parameters.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The paper introduces a novel Quanvolutional Networks (FQN) for time series classification. The key innovation is the Quanv1D layer, which uses quantum circuits instead of classical convolutional filters to process time series data. The model employs amplitude embedding to encode classical data into quantum states efficiently, requiring fewer qubits than previous approaches. The authors tested FQN on 12 UEA time series datasets and is vaildated against other baseline models.

### Strengths
1. The Quanv1D layer design is novel and can handle arbitrary input dimensions, and the paper proves the potential of quantum operations on time series problems 

2. The newly design architecture is parameter efficient (2-7x fewer parameters than classical counterparts as the author claimed)

3. Comparable performance against models with much more parameters

### Weaknesses
1.	My major concern is experiments. The tasks used in the paper contains 12 datasets out of UEA classification archive which consists of 30 tasks. As it is commonly known that the performance gap among time series models can greatly vary among different datasets, there could be potential dataset set selection bias in this paper. And no explicit explanation of why these specific 12 datasets were chosen. 

Besides, the only ‘strong’ baseline is from ModernTCN, which happens to use 10 datasets from UEA 30 (check Appendix A.3 in https://openreview.net/pdf?id=vpJMJerXHU). I’m not commenting on that paper, but unfortunately there’s only 1 dataset (SelfRegulationSCP2) overlapped between the two papers, the other datasets are different.

In this case, as a fair comparison, I suggest the authors to introduce more SOTA time series classification models (Omni-scale cnn, TimesNet), and show the complete experiment results over all 30 UEA datasets or provide a clear explanation of why specific datasets were included/excluded.

2.	The motivation and introduction of quantum operations can be further clarified. 

Firstly, for general audience without quantum computing background, it’s better to give some introduction about the basics in the appendix. (i.e. computational basis state in equation 2, definition of shots) along with the benefit of using potential quantum computing. 

Secondly, can you clarify the motivation of the proposed method? Do you try to deploy the whole network to quantum computer sometime in the long future or the work is inspired by quantum operations to improve parameter efficiency on classical computer? If the former, I’d like to understand if the other components(activation/matrix operation)of the NN are also quantum operator compatible, and how much it can speed up the algorithm. If the latter, I’d like know by saving the parameter number, how much numerical accuracy does it lose/computational efficiency can it gain?

### Questions
1.	By simulating the quantum operation on classical computers, what is the computational overhead of simulating these quantum circuits compared to classical convolutions? And do you have quantitative comparison of computational time to classical time series models?

2.	Can you give a comparison on the numerical performance between the quantum operation vs classical operations in the ideal scenario? For instance, regarding the amplitude embedding, does n qubits lose information or will it behave probabilistically?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
In this paper, a new method built upon the quanvolutional operator, which is derived by **quantum circuits in place of standard convolutional filters**, is proposed. To overcome the exclusive reliance of quanvolutional neural networks on classical layers, as well as their limited application to multi-dimensional data with potential scalability issues, authors propose a novel **Quanv1D layer** for time series data, which acts as the main building block in a lightweight architecture, the so-called Fully Quanvolutional Network (FQN). For increased efficiency, they also propose the incorporation of *amplitude embedding for data encoding* that enables minimal qubit usage. The proposed Quanv1D layers are stacked with *increased dilation rates* to enable larger receptive fields, followed by a final projection layer. Authors evaluate FQN architecture in time series classification, achieving competitive performance with standard CNN-based architectures and other quanvolutional-based architectures while remaining significantly more efficient in terms of memory.

### Strengths
Important strong aspects of the proposed method and study are the following:
1. Authors tackle the very promising and *relatively underexplored* methodological field of quantum machine learning.
2. They propose a building block that combines ideas from *1D convolutional layers and quantum circuits* (Quanv1D layer) and design it specifically for **multi-dimensional time series** data.
3. Importantly, the proposed Quanv1D layer adopts the *standard hyperparameters of 1D convolution*, such as kernel size, stride, dilation, and padding, enabling their straightforward application.
4. Their proposed architecture, built upon an efficient amplitude embedding layer and stacked Quanv1D layers, remains *light in terms of parameters while achieving competitive performance* in few time series classification datasets. Additionally, while relying solely on quantum layers, it showcases their promising application, which is not affected by other standard neural networks.

### Weaknesses
Significant weaknesses of the presented work are summarized as follows:
1. **Presentation of Related Works:** Related works in quantum-based architectures are not thoroughly presented as preliminaries for the proposed method. It is not clear if the main design of the proposed quanvolutional layer significantly expands previous designs beyond the incorporation of user-friendly hyperparameters. This raises doubts about the novelty of the proposed method, making it mostly limited to the selection of the embedding layer.
2. **Experimental Design:** The experimental setup differs from common baselines in the time series classification field. Specifically, the 12 datasets from UEA used by the authors are totally different from the ones used in recent studies [1,2] and commonly used benchmarks [3]. Additionally, the comparisons are limited to two CNN-based and one quant-based network, excluding several common baselines in the field. These choices raise questions about the generalization performance of the proposed method beyond the selected datasets/baselines.
3. **Applicability to Real Hardware and Impact:** The performance achieved by the proposed FQN model in time series classification is, in several cases, inferior to CNN-based baselines. Yet, the number of parameters used by FQN is significantly lower, which makes it computationally attractive and promising for large-scale applications. On the other hand, as mentioned by the authors, the proposed method remains a theoretical framework tested on conventional computers, which questions the potential impact of this contribution if quantum computations are enabled. This is further confirmed by the presented simulation of FQN on finite shots, where the performance was not matched for most datasets to the one achieved with analytical expectation values.

[1] Luo, D., & Wang, X. (2024). Moderntcn: A modern pure convolution structure for general time series analysis. In The Twelfth International Conference on Learning Representations.

[2] Wu, H., Hu, T., Liu, Y., Zhou, H., Wang, J., & Long, M. (2022). Timesnet: Temporal 2d-variation modeling for general time series analysis. arXiv preprint arXiv:2210.02186.

[3] https://github.com/thuml/Time-Series-Library

### Questions
- **Q1 - Datasets:** Based on weakness (2), could the authors explain the selection of this subset of UEA, which differs from the standard 10 preferred datasets in most studies? Have you conducted experiments on the whole data repository? Similarly, most studies in time series classification first evaluate methods on the univariate UCR archive [1]. 
- **Q2 - Experimental Evaluation:** Incorporating additional baselines or tasks could further support performance comparisons in favor of your proposed architecture. Please justify your choices if this is not possible.
- **Q3 - Related Works as Preliminaries:** To better position your contribution in the field of quantum machine learning, it would be essential to clarify better where the technical design of Quanv1D differs from previous quanvolutional layers (weakness (1)). For instance, you can use relevant notations from the literature as a “preliminaries” section in the method.

### Soundness
2

### Presentation
2

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
This paper proposes a novel layer based on quantum computation, and defines a model for time series classification that, on average, surpass previous models. Overall, the idea may sound but the paper requires a refactoring in terms of writing, style, depth of investigation and discussion, and motivation.

### Strengths
1) The proposed method achieves interesting results also with respect to ModernTCN.
2) The motivation and the advantages with respect to QuanvNet are clear.

### Weaknesses
1) Writing needs huge improvements. The authors begin both the abstract and the introduction talking about quanvolutional stuff without explaining what they are or the reason to use them. Without a proper introduction, it is arduous to understand the novelty of this paper. Overall, the paper seems a coding report.
2) Equation (1) does not have any non-quantum comparison so it is difficult to understand the differences.
3) Experiments focus on time series classification, while the method needs a broader discussion on the implications and possible applications.
4) I guess more references exist together with QuanvNet, and a deeper investigation is required.

Minor comments:

The Saxon genitive should be avoided in scientific writing, although I know that both ChatGPT and Grammarly suggest using it. However, I suggest the authors remove all the Saxon genitives from the paper.

### Questions
1) As the authors correctly point out in 3.5, the proposed method avoids overfitting while the counterpart FCN overfits. Together with the demonstration that FQN naturally acts as a regularization technique, some experiments should be performed: it would important to prove it by defining the FCN model with the same number of paramaters of FQN to show that the overfitting is not due to the higher number of params.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
## Summary of the Paper's Contribution
The paper introduces a novel quantum convolutional layer, Quanv1D, and a fully quantum convolutional network (Fully Quanvolutional Networks, FQN), for time series classification tasks. By using amplitude embedding, the method reduces qubit requirements and demonstrates competitive experimental results on 12 time series classification datasets. This work highlights the potential of quantum models in time series classification.

### Strengths
## Strengths:
1.	**Innovation**: The Quanv1D layer and FQN architecture broaden the applicability of quantum convolutional networks, offering a fresh perspective for combining quantum computing and time series analysis.
2.	**Comprehensive Experimental Design**: The experiments cover multiple time series classification datasets, demonstrating FQN’s comparable or superior performance to current state-of-the-art models on various datasets.
3.	**Transparent Limitations Analysis**: The paper thoroughly discusses the gap between the model’s theoretical framework and hardware implementation, especially addressing the quantum hardware limitations in the NISQ era.

### Weaknesses
## Weaknesses:
1.	**Limited Real Hardware Applicability**: As the experiments were all conducted in simulation, there is currently a lack of data on FQN’s performance on real quantum hardware, which limits its feasibility for practical applications.
2.	**Limited Task Scope**: FQN has been validated only on classification tasks, with no exploration of applications like time series forecasting or imputation.
3.	**Resource Demands Remain High**: Although amplitude embedding reduces qubit requirements, circuit depth still increases rapidly with data dimensions, potentially impacting scalability.

### Questions
## Questions for the Authors
1.	In future real-hardware tests, is there a plan to optimize amplitude embedding to mitigate rapid circuit depth growth?
2.	Has there been any validation of FQN’s applicability to other time series tasks (e.g., forecasting or data imputation)? If so, would these tasks require architecture adjustments?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes Fully Quanvolutional Networks (FQN) for time series classification, aiming to address limitations of previous quantum-classical hybrid models by developing an architecture composed entirely of quantum- inspired layers. The contributions of this paper include introducing the Quanv1D layer and utilizing Amplitude Embedding to reduce the number of required qubits for processing multi-dimensional data efficiently. Experimental results on 12 UEA time series datasets demonstrate that FQN achieves comparable or superior performance to existing models, including ModernTCN, with significantly fewer parameters.

### Strengths
- The manuscript effectively demonstrates the necessity of quantum operations and amplitude embedding, particularly through a self-regularization perspective, by analyzing gradient values.
- The manuscript thoroughly evaluates the proposed method against a comprehensive set of baseline models, including both quantum and classical approaches, and notably the current state-of-the-art, ModernTCN.

### Weaknesses
- The novelty of the proposed method appears somewhat incremental, as it primarily focuses on adapting amplitude embedding to the time series domain for improved scalability.
- While the authors provide a self-regularization perspective to explain the generalizability of the fully convolutional structure, additional analysis or insights into why this approach outperforms other quantum models are limited.

### Questions
- Could the authors provide insights into why FQN performs better than ModernTCN or QuanvNet?
- In some cases, FCN outperforms FQN across the 12 datasets. Beyond the similarity between the training and test sets, are there specific characteristic(s) of the datasets where FQN consistently excels?

### Soundness
3

### Presentation
2

### Contribution
3
