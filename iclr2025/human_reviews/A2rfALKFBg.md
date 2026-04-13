## Human Reviewer 1

### Summary
This work presents an approach for analyzing communication between attention heads in transformer models using SVD. It is shown that attention scores are often sparsely encoded in singular vectors of attention head matrices, enabling efficient isolation of features used for inter-head communication. The method is validated on GPT-2 small using the Indirect Object Identification (IOI) task, showing the redundant pathways and communication mechanisms between attention heads.

### Strengths
The main contribution of this paper lies in introducing a more scalable approach for interpreting the information flow within Transformers. Specifically,
- The use of SVD in circuit tracing seems simple and effective. 
- The paper identifies new functionally important components (e.g., attention head 2,8) and provides a detailed analysis of redundant pathways in the model.
- Overall, the paper is well-structured and written.

### Weaknesses
The paper can be improved in several major aspects. 
- The technical novelty seems limited. The idea of using dimensionality reduction (SVD in particular) to interpret and visualize models is not new. 
- The study focuses on the attention layers. Do the MLP layers and layer normalization contribute to the change of causality relationships from layer to layer?
- The analysis is limited to a specific model GPT-2 small and a specific task (IOI). How do the findings generalize to other settings?
- Most of the findings are empirical. It's suggested to also explore why sparse decomposition occurs. 
- Some of the study designs require further justification. For instance, the 70% threshold for filtering contributions seems arbitrary. Also, more justification is needed for the signal/noise separation approach.
- More discussion of failure cases where sparse decomposition might not hold would be valuable.
- The intervention focuses mainly on single-edge and simple multi-edge cases. What about more complex cases that involve multiple edges?
- There is limited comparison with other circuit analysis methods.

### Questions
See the detailed comments above.

### Soundness
2

### Presentation
3

### Contribution
2

### Rating
3

### Confidence
4

---

## Human Reviewer 2

### Summary
This paper introduces a method based on sparse attention decomposition for analyzing the communication and coordination among attention heads in Transformer models. By constructing attention scores sparsely in a new basis through Singular Value Decomposition (SVD), we identify key communication paths between attention heads within the model. Experiments demonstrate that the communication paths identified through sparse decomposition have a practical causal effect on model functionality, enhancing the model's interpretability and offering new insights for understanding and improving Transformer models.

### Strengths
1, Novelty: This paper addresses the previously challenging issue of identifying and interpreting the complex interactions between attention heads in Transformer models by proposing a novel SVD-based sparse decomposition method.

2, Interpretability: The paper uncovers the communication pathways between attention heads in Transformer models, enhancing researchers' understanding of the model's internal workings.

### Weaknesses
1, Possible computational complexity: The computation of SVD and sparse decomposition is usually very complex and requires a lot of computing resources. What is the computational complexity and computing resources consumed in this paper? Have the factors related to computational complexity and required computing resources been considered?

2, Lack of open source code: The author should provide source code to facilitate others to reproduce and verify.

### Questions
The computation of SVD and sparse decomposition is usually very complex and requires a lot of computing resources. What is the computational complexity and computing resources consumed in this paper? Have the factors related to computational complexity and required computing resources been considered?

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
3

---

## Human Reviewer 3

### Summary
The authors explored some of the technical details of GPT-2 through SPARSE ATTENTION DECOMPOSITION. Their tracing study reveals considerable detail not present in previous studies, shedding light on the nature of redundant paths present in GPT-2. Their traces go beyond previous work by identifying features used to communicate between attention heads when performing IOI.

### Strengths
(1) The theoretical proofs in this paper are remarkable.
(2) The figures and tables in the paper are visually appealing, which enhances readability to a certain paper.
(3) The related work and literature survey are adequate and well organized.

### Weaknesses
(1) My big concern is that the paper may be technically obsolete. More mainstream experiments are now being conducted in GPT 4o and GPT o1-based settings. I don't understand why the authors are still conducting experiments on GPT 2. The gap between GPT 2 and GPT 4o and GPT o1-based methods is huge, so I think the experiments and the motivation are very limited, and the techniques in the paper may not be valid for the GPT 4 and GPT o1-based settings.
(2) The writing of the article is obscure. Maybe this article is hard to understand and follow. Reading through the entire paper, I'm not sure what the focus of the article FOCUSED on.
(3) The topics “SPARSE ATTENTION DECOMPOSITION” and “ Circuit Tracing ” did not attract widespread interest, and the importance of this area was not emphasized.
(4) In sum, our contributions are twofold. First, we draw attention to the fact that attention scores are typically sparsely decomposable given the right basis. This has significant implications for the interpretability of model activations. Why? The authors' experiments did not prove their interpretability.
(5) The paper was not compared to multiple state-of-the-art BASELINE methods, so there is insufficient validation of its effectiveness. For example, no quantitative comparison results can be seen in Figures 1, 2, and 3.
(6)The interpretability of the paper is assessed by the "Contribution" to attn. score"?

### Questions
See the  Weaknesses.

### Soundness
3

### Presentation
2

### Contribution
2

### Rating
5

### Confidence
4