# Extending Power of Nature from Binary to Real-Valued Graph Learning in Real World

- Decision: Accept (poster)
- Scores: 6, 6, 5

## Abstract
Nature performs complex computations constantly at clearly lower cost and higher performance than digital computers. It is crucial to understand how to harness the unique computational power of nature in Machine Learning (ML). In the past decade, besides the development of Neural Networks (NNs), the community has also relentlessly explored nature-powered ML paradigms. Although most of them are still predominantly theoretical, a new practical paradigm enabled by the recent advent of CMOS-compatible room-temperature nature-based computers has emerged. By harnessing a dynamical system's intrinsic behavior of chasing the lowest energy state, this paradigm can solve some simple binary problems delivering considerable speedup and energy savings compared with NNs, while maintaining comparable accuracy. Regrettably, its values to the real world are highly constrained by its binary nature. A clear pathway to its extension to real-valued problems remains elusive. This paper aims to unleash this pathway by proposing a novel end-to-end Nature-Powered Graph Learning (NP-GL) framework. Specifically, through a three-dimensional co-design, NP-GL can leverage the spontaneous energy decrease in nature to efficiently solve real-valued graph learning problems. Experimental results across 4 real-world applications with 6 datasets demonstrate that NP-GL delivers, on average, $6.97\times 10^3$ speedup and $10^5$ energy consumption reduction with comparable or even higher accuracy than Graph Neural Networks (GNNs).

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper tackles the issue of the binary nature of existing Ising graph learning by proposing a new Hamiltonian for real-value graph learning problems.

### Strengths
* The paper is well-written and easy to follow, more like a report.
* Solve an important problem in current Ising graph learning using nature-law-based machines, i.e., only supporting binary values.
*  Efficient training methods with three optimizations that ensure training speed and quality.
* Achieves the best accuracy compared to three baselines

### Weaknesses
* Over-exaggerated descriptions in the main context. The paper seems to overclaim some parts, like mentioning, “Regrettably, despite their perceived potential, most
nature-powered ML methods are still predominantly theoretical, outperforming NNs only in toy problems under highly idealized conditions” while the Ising machine is also not quite partial and only applicable for some problems.
* Out-dated baselines. In comparison, the authors chose three “SOTA” spatial-temporal GNNs, while the earliest was published in 2020. The author should compare with more recent advances.

### Questions
*  Could the authors provide more comparison with recent SOTA GNN spatial-temporal GNNs? I found one recent paper,
    * Jiang, Renhe, et al. "Spatio-temporal meta-graph learning for traffic forecasting." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 37. No. 7. 2023.
* What’s the physical meaning of asymmetric weight decomposition? As it is a natural process (Hamitonian), is this symmetric weight decomposition meaningful for the physical system besides the efficiency consideration?

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
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
**Objective:** The paper aims to extend the capabilities of nature-powered computations from binary problems to real-valued graph learning problems. The authors propose a novel end-to-end Nature-Powered Graph Learning (NP-GL) framework, which leverages the natural power of entropy increase to efficiently solve real-valued graph learning problems.

**Methodology:** The NP-GL framework is designed through a three-dimensional co-design, incorporating a new Hamiltonian that is hardware-friendly, maintains distinct stable states with real values, and ensures high expressivity. The training algorithms of NP-GL adopt an improved conditional likelihood method with optimizations for complexity reduction, convergence expediation, and better learning from temporal information. Additionally, a new nature-based computer is developed to support the NP-GL Hamiltonian, enabling the solution of real-valued graph learning problems.

**Results:** Experimental results across four real-world applications and six datasets demonstrate that NP-GL delivers, on average, a 6.97 × 10^3 speedup and 10^5× energy consumption reduction, with comparable or even higher accuracy than Graph Neural Networks (GNNs).

The contribution lies in

**Extending Nature-Powered ML to Real-Valued Problems:** The paper introduces NP-GL, an end-to-end nature-powered graph learning method that breaks the binary limitation of existing nature-powered ML methods, extending their applicability to real-valued problems.

**New Hardware-Friendly Hamiltonian:** A new Hamiltonian is designed for real-valued support, coupled with an efficient training method that ensures high training speed and quality.

**Development of a New Nature-Based Computer:** A new nature-based computer is developed for the NP-GL Hamiltonian, using the Ising machine as a backbone, enabling the solution of real-valued graph learning problems using nature’s power.

**Significant Speedup and Energy Savings:** NP-GL demonstrates a substantial speedup (6.97 × 10^3) and energy savings (10^5×) compared to GNNs, with even higher accuracy across various real-world applications and datasets.

### Strengths
1. Originality:
- Innovative Approach: The paper introduces a novel end-to-end Nature-Powered Graph Learning (NP-GL) framework, extending the capabilities of nature-powered computations from binary problems to real-valued graph learning problems. This is a significant departure from existing nature-powered ML methods, showcasing a high level of originality.
- Unique Integration: The three-dimensional co-design integrating a new Hamiltonian, training algorithms, and a nature-based computer is a unique approach that has not been explored extensively in previous works.
2. Quality:
- Robust Methodology: The paper employs a robust methodology, incorporating a hardware-friendly Hamiltonian, efficient training methods with optimizations, and the development of a new nature-based computer.
- Comprehensive Evaluation: The experimental results across four real-world applications and six datasets provide a comprehensive evaluation of the NP-GL framework, demonstrating its effectiveness in delivering significant speedup, energy savings, and high accuracy compared to GNNs.
3. Clarity:
- Well-Structured: The paper is well-structured, with a clear introduction, background, methodology, results, and conclusion sections. This structure aids in the reader’s understanding of the content.
- Detailed Explanations: The authors provide detailed explanations of the NP-GL framework, the new Hamiltonian, the training algorithms, and the nature-based computer, ensuring that readers can grasp the complexities of the work.
4. Significance:
- Addressing Real-World Problems: By extending the applicability of nature-powered ML methods to real-valued problems, the paper addresses a significant gap in the field, making it highly relevant to real-world applications.
- Potential for Impact: The demonstrated speedup, energy savings, and accuracy improvements have the potential to make a substantial impact in the field of graph learning, showcasing the significance of the work.

### Weaknesses
- Insufficient Discussion on Challenges: While the paper provides a comprehensive overview of the NP-GL framework and its benefits, there could be a more in-depth discussion on the potential challenges and limitations of the proposed approach. Providing such insights would offer a balanced view and help guide future research. 

Minor: Add one-liner for the future insights summary to main text from Appendix.

### Questions
- Could you elaborate on the adaptation for NP-GL on top of SOTA Ising machine? The hardware implementation, challenges, and potential optimizations could provide valuable insights for readers interested in the practical aspects of the work.
- To confirm I assume the the engergy and latency are based on the simulation in CAD software?
- Would this machine be able to generalize beyond GNN?

- How it performs as the size of the graph increases and any potential strategies for handling large-scale graph learning problems?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper propose a novel graphic model architecture by improving the existing ising model. The authors further propose the efficient training and inference algorithm to search for the local minimum of the proposed solution. The paper also shows the potential hardware architecture to implement this novel graphical model. The results demonstrate that the proposed solution can achieve comparable performance as GNN over multiple tasks.

### Strengths
+ A novel graphical model architecture
+ Inference and training methods to achieve the local min
+ Efficient hardware implementation

### Weaknesses
- To me, the most significant problem of this work is insufficient review for the prior work. Section 2.1 and 2.2 is a good background introduction, but not too many prior works on the variation of Ising model are discussed, making the contribution of this work hard to justify.
- A section is missing to introduce the prior work on hardware implementation for Ising model.
- Technically, this paper simply propose a modified version of Ising model by using a pure quadratic term to replace the linear term in Ising Hamiltonian. 
- Graphic model may not be well-suited for ICLR, which mostly focus on deep learning.

### Questions
Please see the weakness section and solve the problem accordingly.

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good
