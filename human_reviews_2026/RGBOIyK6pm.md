# Quantum Kolmogorov–Arnold Networks

- Avg Score: 4.00
- Decision: Reject
- Scores: 0, 2, 8, 6

## Abstract
The pursuit of quantum advantage in machine learning drives the exploration of quantum analogues of powerful classical architectures. The recent introduction of Kolmogorov-Arnold Networks (KANs) provides a mathematically grounded framework for enhanced expressivity and accuracy in function approximation. However, KANs and their variants fundamentally rely on predefined basis functions. In this work, we introduce Quantum Kolmogorov-Arnold Networks (QKANs), which leverage parameterized quantum circuits to implement learnable activation functions without predefined bases. We establish the theoretical foundations of QKANs and demonstrate their effectiveness through numerical experiments, showing superior performance on function approximation tasks. QKANs accurately model complex nonlinear relationships and establish a new benchmark for expressive power in quantum machine learning. This work bridges KANs with quantum computation, providing a new paradigm for expressive quantum machine learning.

## Human Reviews

## Human Reviewer 1

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
The authors propose a quantum implementation of Komolgorov-Arnold networks. They describe the methodology and provide experiments to showcase their architecture.

The authors state that quantum generalizations of KANs are missing from the literature, however, there is a well-known paper in the community, which proposed the architecture more than a year ago (arXiv:2410.04435). There are even follow-up works published (https://www.nature.com/articles/s41598-025-22705-9), thus, the authors failed to provide a review of state-of-the-art literature.

Beyond that, the paper has significant structural issues, in particular, lacking a proper background section introducing KANs. The authors have plenty of space left for the page limit, which should have been used. Section 3 is as a result very hard to follow for a reader not fully familiar with the framework (in particular, the quantum walk operators, QGSP correspondences) are just mentioned as a side note, despite seemingly playing an important role).

I would recommend adding details to the experimental setup, as it is not clear what has been optimized. As far as I can tell, the authors benchmarked one specific architecture/instantiation, which is not enough to provide evidence on how the method performs generally. Further, also the comparison to the classical baseline is missing details on what was optimized. The authors state the five independent runs were conducted to ensure statistical robustness, however, five runs do not suffice for that. When claiming statistical robustness, statistical methods should be employed accordingly to ensure they hold. Further, the very significant standard deviation should be analyzed and discussed, which is also missing from the paper.

The authors failed to provide a comprehensive literature review and since the QKAN architecture was already proposed more than a year, it is not clear what the novelty of this paper is. Besides that, it has significant structural issues, in particular, missing background and definitions, and an unclear experimental setup. Thus, I do not think that the work is neither sufficiently well written, nor has sufficient scientific merit to be published at ICLR at this stage.

### Strengths
The approach of exploring KANs for quantum computing indeed is interesting, in particular, given the inherent relation.

### Weaknesses
- QKANs have been proposed already - there is no mention of the work nor discussion on differences
 - No related work section, in particular, the paper proposing this very work is missing
 - Structural issues; no background and state of the art, missing definitions for the methods, missing information for experimental setup - the paper is hard to follow
 - Experiments; missing statistical methods for claims on robustness, limited experiments - only one architecture, classical baseline is weak (optimization)?; no discussion on the results generally except for one table - I would expect at least a discussion on the incredibly high standard deviation that was observed
 - Eq4: I think the vector should be made up of $m$ sums, I think the dots are missing.

### Questions
- What is meant with "target outputs are appropriately scaled to improve training performance"?
 - What is meant with "input data are scaled to lie within the valid range for the quantum circuits"? Why can this not explicitly be stated again?
 - Could you provide more intuition about the significant differences in standard deviation on classical vs quantum KANs? Can you provide visuals for the results?

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper introduces a quantum version of Kolmogorov Arnold Networks (KANs). KANs is a type of neural network architecture in which the activation functions can be trained. The quantum KANs that the authors introduce in this paper are built by combining (i) projected unitary encodings for scalar inputs with polynomial transformations via Generalized Quantum Signal Processing (GQSP, Motlagh & Wiebe 2024). These univariate results are then combined into a single result using normalized "Linear Combination of Unitaries" (LCUs). Finally, a Hadamard test is applied to extract scalar values from the output layer. 

The paper introduces this architecture in detail, then provides two experiments for regression and classification on synthetic data.

### Strengths
- The paper is generally well written and clearly presented. The outline follows a clear line of thought and introduces relevant concepts / provides sources at the appropriate places. 
- The paper contains a discussion on required circuit width, depth and gate count. Moreover, it also discusses potential advantages of KANs and function approximation properties.

### Weaknesses
There are several central issues concerning novelty (QKANs have already been proposed previously), experimental rigour and overall relevance of the suggested architecture. Based on these, unfortunately, I can not recommend the paper for acceptance.
- Novelty: the title / abstract / introduction of the paper are suggesting that a quantum version of KANs has not been proposed in the literature. However, the concept of Quantum KANs has been introduced already previously in the literature in "QKAN: Quantum Kolmogorov-Arnold Networks" by Petr Ivashkov, Po-Wei Huang, Kelvin Koor, Lirandë Pira, Patrick Rebentrost, 
https://arxiv.org/abs/2410.04435. The paper does not cite or compare their architecture with these previously proposed quantum KANs. 
- Missing experimental details: the paper does not provide sufficient details on training and implementation of the Quantum KANs. How did you train the Quantum KANs? Which optimizer did you use for the classical KANs? Supporting code is not provided, hence the experiments can not be reproduced in any way. 
- Concepts not defined: In experiments, the Quantum KANs are said to be of hidden dimension 5. But the hidden dimension of the Quantum KAN has not been defined previously. For a proper comparision it is important to understand the number of trainable parameters: will these be the same for a KAN and QKAN?
- Insufficient comparisons and baselines: The Quantum KAN and classical KAN are only compared for a fixed depth (2) and hidden size (5). Sound conclusions would require a more thorough comparisons for other network sizes, also including other classical neural network architectures. Moreover, what are the runtimes for each of the networks? Overall runtimes are also a crucial factor for comparing different models. 
- Model architecture concerns: the discussion in Section 5.2 suggests that, as a map from inputs to outputs, the proposed QKAN ultimately just applies a multivariate polynomial. (see lines 299/300). This would mean that the approximation and expressivity properties of this model are the same as for a multivariate polynomial. Why could you not directly apply a polynomial then? Also, if these are just polynomial transformations, this does not justify calling the architectures "neural networks".

### Questions
- How does your proposed architecture compare to the quantum versions of KANs as previously proposed in the literature? 
- Which training algorithm did you use for optimizing the KAN and QKAN parameters? What learning rates did you choose? 
- What were the training times for training KANs and QKANs in the experiments? 
- How do KANs and QKANs compare for different choices of network size / depth?
- How do QKANs compare to classical fully trainable neural networks? How do they compare to other QNN architectures? 
- What is the hidden dimension of a QKAN? Are the number of trainable parameters the same for the KAN and QKANs compared in the experiments?
- What is the advantage of using QKANs in comparison to just applying multivariate polynomials? 
- Is there a way of introducing a non-polynomial non-linearity in the model to ensure that the term "neural network" is indeed justified?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
In this paper, a quantum implementation that leverages parameterized quantum circuits to realize the learnable activation functions of Kolmogorov-Arnold Networks is proposed.

The proposal is conceptually novel and technically sound. It bridges KANs with quantum computation.

Experimental validation is limited, and scalability (w.r.t. number of layers) is questionable.

I think the following paper should be cited (even if is not propose a quantum implementation of KAN):
Kanqas: Kolmogorov-arnold network for quantum architecture search
A Kundu, A Sarkar, A Sadhu
EPJ Quantum Technology, 2024

The very recent paper "QuKAN: A Quantum Circuit Born Machine Approach to Quantum Kolmogorov Arnold Networks" should be discussed.

### Strengths
- the novelty of the proposal
- technical depth
- reported simulation results are encouraging
- quality of presentation

### Weaknesses
- Evaluation is limited to the simulation of small-scale functions and toy datasets
- Being a quantum implementation of KAN, the success and relevance of the proposed approach are strictly related to those of KANs

### Questions
Could you discuss the scalability of QKANs w.r.t. number of layers?

How does the recent paper "QuKAN: A Quantum Circuit Born Machine Approach to Quantum Kolmogorov Arnold Networks" affect the discussion of this paper?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents Quantum Kolmogorov–Arnold Networks (QKANs), extending the recently developed classical Kolmogorov–Arnold Networks (KANs) into the quantum domain. KANs are a powerful function approximation architecture grounded in the Kolmogorov–Arnold representation theorem. The authors design a quantum analog by using parameterized quantum circuits to realize the learnable univariate function modules central to KANs. The paper provides a detailed theoretical formulation of QKANs, including quantum circuit constructions. Experimentally, QKANs are evaluated on function approximation tasks (regression and classification) and show better performance than classical KANs of similar size in simulation. The authors also analyze the quantum resource requirements.

### Strengths
The introduction of QKANs is a novel and timely idea, bridging the new KAN architecture with quantum computation.
The paper provides a rigorous mathematical underpinning for QKANs, building upon quantum signal processing.
The simulation results are promising, showing a potential expressivity advantage for QKANs over classical KANs of the same size on the tested tasks.

### Weaknesses
All experiments are on noiseless simulation; robustness under noise or hardware is not evaluated. This is a critical omission. Given that parameterized quantum circuits are notoriously sensitive to noise, the claimed performance gains over classical KANs may not hold in any realistic (NISQ) setting.
Circuit depth can grow quickly. The paper's own analysis shows potentially exponential depth scaling depending on the construction, but this major practical barrier is dismissed too abstractly. This scaling issue seems to make the approach impractical for any problem of meaningful size, undermining the paper's central claim of utility.
The comparison is mainly to classical KANs, not to other small classical baselines (e.g., an MLP). While KANs are a good target, the lack of comparison to a standard, well-tuned MLP makes it hard to gauge if the complexity of QKAN is truly justified.

### Questions
1.I suggest the authors add a short note on how they expect noise to affect QKAN performance. A full noise simulation is necessary to make any claims about practical advantage.
2.A small example resource estimate (qubits / depth for one representative QKAN) would make feasibility clearer. The abstract scaling laws are not enough; a concrete example would highlight the severity of the depth problem.
3.A brief comment on how a simple MLP or small CNN might compare would help position the gains.

### Soundness
3

### Presentation
3

### Contribution
3
