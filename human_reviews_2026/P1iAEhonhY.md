# From Neural Networks to Logical Theories: The Correspondence between Fibring Modal Logics and Fibring Neural Networks

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 8

## Abstract
Fibring of modal logics is a well-established formalism for combining countable families of modal logics into a single fibred language with common semantics, characterized by fibred models. Inspired by this formalism, fibring of neural networks was introduced as a neurosymbolic framework for combining learning and reasoning in neural networks. Fibring of neural networks uses the (pre-)activations of a trained network to evaluate a fibring function computing the weights of another network whose outputs are injected back into the original network. However, the exact correspondence between fibring of neural networks and fibring of modal logics was never formally established. In this paper, we close this gap by formalizing the idea of fibred models compatible with fibred neural networks. Using this correspondence, we then derive non-uniform logical expressiveness results for Graph Neural Networks (GNNs), Graph Attention Networks (GATs) and Transformer encoders. Longer-term, the goal of this paper is to open the way for the use of fibring as a formalism for interpreting the logical theories learnt by neural networks with the tools of computational logic.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
Fibring of neural networks was introduced as a neurosymbolic framework to integrate learning and reasoning within neural networks. This paper bridges the gap between the fibring of neural networks and the fibring of modal logics by formalizing fibred models that are compatible with fibred neural networks. Building on this foundation, the authors derive results on the non-uniform logical expressiveness of Graph Neural Networks (GNNs), Graph Attention Networks (GATs), and Transformer encoders.

### Strengths
The paper establishes the connections between the fibring of neural networks and the fibring of modal logic. It provides a rigorous formalization linking fibred logic to fibred neural networks and demonstrates how this theory can be applied to widely used neural architectures, including GNNs, GATs, and Transformers, to enhance interpretability.

### Weaknesses
Although the paper is solid, some details are not explained clearly, making the overall presentation rather obscure. For instance, an illustrative figure for the fibred network could help clarify the concept. Please see the detailed questions below.

### Questions
1. In line 191, what is meant by the “accessibility relation”? Could the authors provide a formal definition or illustrative examples to clarify this concept?
2. The authors claim that non-uniform logical expressiveness results can be derived from several modern neural architectures. I wonder whether these models need to be pre-trained on specific datasets. 
3. Furthermore, what specific logical expressions can be derived from GNNs, GATs, and Transformers? Could the authors provide more illustrative examples or figures to demonstrate this derivation process?

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
3

### Summary
This paper formally connects fibring of modal logics and fibring of neural networks, addressing a gap in the literature. It introduces fibred models compatible with neural networks and derives non-uniform logical expressiveness results for GNNs, GATs, and Transformers. The long-term goal, according to the authors, is to enable fibring as a tool for interpreting neural networks using computational logic.

### Strengths
- The paper presents a novel and rigorous mathematical proof connecting fibring modal logic and fibring neural networks, filling a gap in the literature.
- It extends this framework to GNNs, GATs, and Transformers, which are widely used architectures in modern AI.
- The theoretical foundation is solid, with clear definitions and formalizations of fibred models for neural networks.

### Weaknesses
- The practical relevance of connecting fibring neural networks to fibring modal logic is unclear. What are the advantages of this connection for real-world applications or further research?
- Fibring neural networks are a niche approach in neuro-symbolic integration and are rarely used in practice. This raises concerns about the applicability of the paper’s findings.
- The goal of connecting fibring neural networks and GNNs is not well-articulated. The connection feels fuzzy, and a clearer presentation is needed to justify its importance.
- Theorems 5.1 and 5.2 prove the existence of a fibred neural network equivalent to a given GNN but provide no practical insights on how to identify or construct such a network. Additionally, the advantages of using the equivalent fibred neural network over the original GNN are unclear.
- The paper claims its findings apply to Transformer architectures, but the discussion focuses on GNNs and GATs. While GATs and Transformers share connections, this is insufficient to generalize the results to Transformers.
- The paper is purely theoretical, and the connection between fibring logic, fibred neural networks, and GNNs remains abstract. How can these results be applied in practice? Experiments or case studies would strengthen the paper’s impact.

### Questions
- What are the practical advantages of establishing a connection between fibring neural networks and fibring modal logic? Could the authors provide examples or use cases where this connection is beneficial?
- The connection between fibring neural networks and GNNs is described as fuzzy. Could the authors provide a clearer explanation or additional examples to strengthen this link?
- Theorems 5.1 and 5.2 discuss the existence of equivalent fibred neural networks. Could the authors provide guidance or methods for identifying or constructing these networks in practice?
- How could the theoretical results be applied in real-world scenarios? Could the authors include experiments or simulations to demonstrate the practical implications of their findings?
- The paper suggests that the results apply to Transformers, but the discussion focuses on GNNs and GATs. Could the authors clarify or expand on how their findings extend to Transformer architectures?
- The paper derives non-uniform logical expressiveness results for GNNs, GATs, and Transformers. Could the authors clarify what non-uniform expressiveness entails and how it differs from uniform expressiveness in this context?
- The authors mention a long-term goal of using fibring to interpret neural networks. Could they provide a roadmap or examples of how this interpretation might work in practice?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper makes two key theoretical contributions. First, it establishes the connection between fibred neural networks and fibring modal logics. Specifically, the formulation relies on showing that evaluating the fibred network on an input is equivalent to checking the truth value of a formula in the fibring logic. Next, they apply this equivalence to derive logical expressiveness of well-known deep models such as GNNs, GATs and transformers. In particular, they focus on non-uniform logical expressiveness, i.e., is there a logical formula that coincides with with the output of the deep model depending on the input. The proof relies on showing that there exists a fibred network that produces the same input as the Graph Neural Network. Thus, using  the equivalence established between fibred neural networks and firing modal logic, the non-uniform expressiveness can be established.

### Strengths
Strengths
+ Seems to be the first work to make the prove non-uniform expressiveness results for well known Deep models including transformers
+ discovering the connection between fibring logics to prove logical expressiveness results could have implications on future theoretical results for more general logical expressiveness results which could be significant for interpretable AI.
The paper seems to make novel theoretical contributions that can prove useful in understanding why and how models like transformers can perform logical reasoning.

### Weaknesses
Weakness
- While this is a theoretical paper, it would have been nice to provide some application benefits that could potentially result from these theoretical advances. 
- In terms of the presentation, it will be helpful to have maybe a summary of results that are known in logical expressiveness and what remains open problems, and how the proposed results addresses some of these. Section 6 describes this but it would be easier to understand the significance with a concise summary of the contributions.

### Questions
What are the practical implications of showing uniform vs non-uniform expressiveness of models? How would they potentially be useful in practice?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper presents a formal correspondence between fibring modal logics and fibring neural networks and makes the following key contributions:
1.  It defines the concept of compatible fibred models and proves that fibred neural networks can be interpreted as logical formulas in this framework
2. It establishes non-uniform expressiveness results for Graph Neural Networks, Graph Attention Networks, and Transformer encoders.
3. It suggests using fibring as a potential unifying formalism for studying logical expressiveness of different neural architectures.

### Strengths
1. **Relevance**
The paper addresses a critical challenge: formally characterize the logical expressivity of various neural network architectures 

2. **Significance and Novelty**
To the best of my knowledge, the proposed approach is novel in its problem formulation, the formal correspondence  between fibring modal logics and fibring neural networks, and the important theoretical results established.  Fibring could potentially be very significant and serve as a unifying formalism for studying logical expressiveness of different neural architectures.

3. **Soundness and Presentation**
Overall, the paper is well organized and well written. The key theoretical results are formally proven.

### Weaknesses
I could not find any major weaknesses. 

One minor error in the formal definition of the recursive computation of a fibred neural network: in line 169, "N^{l_{i−1}:l_i} (**y_{i−1}**)" should be replaced with   "N^{l_{i−1}:l_i} (**h_{i−1}**)". In order words,  h_{i-1} should be used instead of y_{i-1}. y_{i-1} is the input of the instance network of u_{i-1}. It is the output of that instance network combined with x_{i-1} (i.e., h_{i-1} ) that should directly be used to compute x_i

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
3
