# Discovering Group Structures via Unitary Representation Learning

- Decision: Accept (Poster)
- Scores: 6, 6, 5

## Abstract
Discovering group structures within data poses a fundamental challenge across diverse scientific domains. A key obstacle is the non-differentiable nature of group axioms, hindering their integration into deep learning frameworks. To address this, we introduce a novel differentiable approach leveraging the representation theory of finite groups. Our method features a unique network architecture that models interactions between group elements via matrix multiplication of their representations, along with a regularizer promoting the unitarity of these representations. The interplay between the network architecture and the unitarity condition implicitly encourages the emergence of valid group structures. Evaluations demonstrate our method's ability to accurately recover group operations and their unitary representations from partial observations, achieving significant improvements in sample efficiency and a $\times 1000$ speedup over the state of the art. This work lays the foundation for a promising new paradigm in automated algebraic structure discovery, with potential applications across various domains, including automatic symmetry discovery for geometric deep learning.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces a novel differentiable approach for discovering group structures within data by leveraging group representation theory. Traditionally, the non-differentiable nature of group axioms has posed challenges for their integration into deep learning frameworks. The proposed method addresses this limitation by employing a tensor-factorization model with matrix embeddings for each group element, along with a regularizer that encourages the learning of unitary matrix representations. This setup provides a strong inductive bias toward capturing group structures in data. The model is evaluated using Symbolic Operation Completion (SOC) tasks, where it successfully learns group operations from limited data and accurately discovers unitary representations. Furthermore, the model introduces an implicit complexity metric, facilitating the discovery of group structures in a variety of datasets, with potential applications in areas like symmetry discovery.

The method builds on well-established principles from group representation theory but extends them into a differentiable framework, making it applicable in modern deep learning contexts. By transforming SOC tasks into a tensor completion problem, the authors present a linearized framework that captures binary operations through bilinear maps, with a novel architecture—Hypercube—used to factorize the tensors. The accompanying regularizer plays a critical role in promoting unitary representations, ensuring that the model adheres to group properties.

### Strengths
- Differentiable Framework for Group Discovery: A notable strength of the paper is its introduction of a differentiable method for discovering group structures, addressing a long-standing challenge in integrating group theory with deep learning. By employing tensor-factorization and matrix embeddings, along with a regularizer promoting unitary matrices, the approach allows for the learning of group representations within a differentiable framework. This innovation enhances the applicability of group theory in machine learning, providing a more seamless way to incorporate group structures into data-driven models, particularly in contexts requiring automatic discovery of algebraic properties.
- Solid Theoretical Foundation with Practical Relevance: Another strength lies in the paper's solid grounding in group representation theory, combined with its practical application to SOC tasks. The method’s reliance on well-established theoretical principles, such as the use of a complexity metric and regularization to guide learning, is complemented by its practical effectiveness in recovering group operations and unitary representations from limited data. This balance between theory and practice makes the approach both rigorous and relevant, showcasing its potential for broader applications in tasks involving the discovery of group structures within data.

### Weaknesses
- Narrow Focus on SOC Tasks: One of the paper's weaknesses is its narrow focus on SOC tasks for evaluation. While SOC tasks provide a controlled environment to test group structure discovery, they may not fully capture the model’s performance or versatility in real-world applications, where data and tasks are often more varied and complex. By limiting the scope to SOC tasks, the paper does not explore how well the proposed method generalizes to other tasks like graph classification, natural language processing, or scientific data analysis, which could broaden the understanding of its practical impact.
- Limited Baseline Comparisons: The comparison of Hypercube to other models, especially the Transformer from Power et al. (2022), is somewhat limited. While the paper shows that Hypercube outperforms the Transformer in terms of learning speed, the focus on just one baseline model limits the robustness of the evaluation. There is no detailed comparison against other relevant methods that also target group discovery or structure learning. Including a wider range of baseline comparisons would provide a more comprehensive assessment of Hypercube's relative strengths and weaknesses.

### Questions
The paper does not provide much detail on how sensitive the performance of HyperCube is to various hyperparameters, such as the regularization strength, learning rate, or factor initialization. Understanding the impact of these hyperparameters, especially across different tasks (group vs. non-group operations), would be useful for replicating and extending the work to new datasets and applications. Can the authors provide more clarity on the tuning process and how these parameters affect the model's performance?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper studies a method to predict the complete group elements based on partial observations of the action of a finite group using tensor decomposition. The HyperCube decomposition and the regularization term presented in equation (6) are newly proposed. This regularization term theoretically promotes the "Imbalance" and "Unitarity" of the decomposed factor parameters. Experiments are conducted on data involving simple operations like integer addition and subtraction, comparing the performance with a Transformer.

### Strengths
- The idea of connecting group theory to tensor decomposition to solve this problem is unique and interesting.
- The paper is written in a clear and accessible way, starting from fundamental concepts.

### Weaknesses
- The necessity of the inductive bias introduced in this work is not adequately explained, particularly why this specific bias was chosen over others.
- The proposed method does not scale well. If the number of group elements is n, the computational complexity is O(n^3).
- The experimental results mentioned in the text (Figures 10 and 11) are pushed to the appendix, potentially violating the page limit.

### Questions
- The paper solves the problem of predicting group actions, but how does this lead to symmetry discovery? Could the authors provide concrete ideas?
- How could this method be adapted to continuous groups, such as Lie groups?
- Could the authors explain in what criterion the inductive bias introduced by (5) was chosen? For example, the tensor T[i, j, :] is strongly constrained to become a one-hot vector for all i, j. Could another approach, like a classification model where $y_{ijk} = \langle A_i, C_k, B_j \rangle$ (with parameters $A_i, B_j \in \mathbb{R}^k $ and $ C_k \in \mathbb{R}^{k \times k}$ where k is an embedding dimension) serve as a better model? What advantages does the proposed inductive bias offer compared to this alternative?
- The authors claim that the sample complexity is better than that of a Transformer, but what about the computational time (wallclock) in comparison?
- Could the authors provide a more intuitive explanation of the "Imbalanceness" in Lemma 5.1?
- How sensitive is the prediction accuracy to $ \epsilon $?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The paper presents a novel differentiable approach for discovering group structures in data using group representation theory and tensor-factorization models. It effectively learns group operations and unitary representations from limited data while defining a complexity metric that enhances group structure discovery, impacting various scientific domains and automatic symmetry discovery applications.

### Strengths
1. This paper proposes a differentiable approach to the automatic discovery of finite group structures through the learning of their underlying representations.

2. This paper introduces a novel regularization technique that encourages the matrices to maintain unitarity, thereby improving the preservation of group structures.

3. This paper offers a theoretical analysis of the inductive bias associated with HyperCube.

### Weaknesses
1. The clarity of the paper's writing is lacking, which makes it somewhat challenging to read. For example, the rationale for introducing HyperCube and its advantages are not adequately explained. Additionally, the underlying intuition behind HyperCube regularization is insufficiently articulated.

2. The paper does not include a comprehensive set of experiments. L2 regularization is relatively weak compared to other regularization techniques. To strengthen the paper's contributions, it is essential to compare HyperCube regularization with more robust regularization methods.

3. Given that HyperCube regularization involves matrix multiplication, it is important to present both the computation and runtime in the experimental results. This information would provide valuable insights into the efficiency of the proposed method.

### Questions
1.  In Line 185, can a and b mapped to vector embeddings or diagonal matrix embeddings?

2. Can the tensor T be parameterized using other tensor decomposition models?

### Soundness
3

### Presentation
2

### Contribution
3
