# Node2ket: Efficient High-Dimensional Network Embedding in Quantum Hilbert Space

- Avg Score: 6.00
- Decision: Accept (poster)
- Scores: 6, 6, 6

## Abstract
Network embedding (NE) is a prominent technique for network analysis where the nodes are represented as vectorized embeddings in a continuous space. Existing works tend to resort to the low-dimensional embedding space for efficiency and less risk of over-fitting. In this paper, we explore a new NE paradigm whose embedding dimension goes exponentially high w.r.t. the number of parameters, yet being very efficient and effective. Specifically, the node embeddings are represented as product states that lie in a super high-dimensional (e.g. $2^{32}$-dim) quantum Hilbert space, with a carefully designed optimization approach to guarantee the robustness to work in different scenarios. In the experiments, we show diverse virtues of our methods, including but not limited to: the overwhelming performance on downstream tasks against conventional low-dimensional NE baselines with the similar amount of computing resources, the super high efficiency for a fixed low embedding dimension (e.g. 512) with less than 1/200 memory usage, the robustness when equipped with different objectives and sampling strategies as a fundamental tool for future NE research. As a relatively unexplored topic in literature, the high-dimensional NE paradigm is demonstrated effective both experimentally and theoretically.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This work proposes two high-dimensional network embedding methods called node2ket and node2ket+ that outperform standard methods such as word2ket. As evidence of these claims, they perform experiments studying several tasks, obtaining better compressive ratio than other approaches. Additionally, they provide an implementation in a library called LIBN2K.

### Strengths
- This paper improves upon existing network embedding techniques by proposing a high-dimensional embedding with better efficiency over previous quantum-inspired methods such as word2ket.
- The theoretical analysis is clear, showing that node2ket gets a high-rank approximation of the information matrix
- The experiments appear to be quite thorough and shows advantages over existing methods, as promised.
- It is nice that the code is made available.

### Weaknesses
- This is a quantum-inspired algorithm for classical machine learning, which suffers from the lack of a clear connection to quantum computation (see Questions).
- There are gaps for actually making this algorithm "quantum-friendly," as it is generally not easy to load classical information into a quantum device.

### Questions
- What is the significance of the embeddings being designed for "quantum Hilbert space"? To me, the relationship to quantum computation is not clear and seems more like an afterthought. Quantum computers are known to be good at problems with certain structure, and it's not clear to me what structure is being leveraged here (and what benefits are obtained as a result).
- The fact that pure quantum states are normalized leads to the constraint $\\|\mathbf{x}_i\\|=1$. Is there any consideration for eliminating a global phase, which I assume would affect the embedding? Also, I wonder what embedding might be developed for more general quantum states, such as mixed states.
- Forgive my ignorance, but what is the definition of positive node pairs and negative nodes? Does it just mean the inner product is positive or negative?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper introduces a groundbreaking paradigm for network embedding (NE), departing from traditional low-dimensional embeddings and exploring high-dimensional quantum state representations. The authors propose two NE methods, node2ket and node2ket+, and implement them in a flexible, efficient C++ library (LIBN2K). Experimental results showcase the good performance of their proposal, boasting advantages in parameter efficiency, running speed, and memory usage.

### Strengths
The paper is well organized. The insights offered in the paper have the potential to inspire the development of other quantum-inspired methods and contribute to the broader application of quantum computing in the field of network embedding.

### Weaknesses
The primary concern in this submission pertains to the technical contribution. First, the extension of the word2ket concept to product states appears relatively straightforward. Second, the utilization of product states might limit the embedding's expressivity since these states occupy a smaller portion of the Hilbert space and result in a low-dimensional representation.

### Questions
No questions at the moment.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
For the standing and important task of network embedding in data mining and machine learning, the paper proposes explores the exponentially high embedding space for network embedding, which largely differs from existing works dewelling on the low-dimensional embedding. This is achieved by product quantum states in a super high-dimensional quantum Hilbert space. The experiments show surprisingly strong performance of the approach, in terms of both high memory and running efficiency with strong robustness across different tasks of network reconstruction, link prediction, and node classification. The authors also provide the source code to ensure the soundness of the experiments.

### Strengths
1) This paper innovatively resorts to the high-dimensional embedding space for network embedding, which quite departures from existing literature.
2) The paper is well presented and the overview plot in Fig. 1 is very informative and useful to readers. The paper is well organized with strong content in appendix that signifcantly enriches the paper.
3) The experiments are impressive. Provided with the source code, I am convinced by the strong performance.
4) The authors give strong theoretical understanding of the essence of their approaches, which I really appreciate.

### Weaknesses
As the authors emphasized, the presented techniques are mainly suited for the structure networks, without considering the attributes. I understand this setting and think it is reasonable in practice. It also fits with many previous works in literature that have also been compred in this paper.

### Questions
1) Comaperd with Fig. 1, can the authors provide a more succinct plot to convey the main idea of the paper? Fig. 1 is still a bit busy which is useful yet a more direct illustration in the begining of the paper is welcomed. Something like Fig. 2 is better.
2) Can the approach be useful for solving combinatorial problems especially for large-scale ones? As there is little attributes need to be considered in these cases thus it seems suited to the proposed methods?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
