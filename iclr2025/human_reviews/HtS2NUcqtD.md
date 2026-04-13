## Human Reviewer 1

### Summary
The paper proposes a closed-form solution to compute gradients for Tensor Attention 
in higher-order transformer models efficiently. By utilizing polynomial approximation
methods and tensor algebraic techniques, the algorithm achieves nearly linear time
n1+O(1), the same complexity for both forward and backward computation under 
certain assumptions, which are proven to be necessary and tight in the paper. The 
theoretical results establish the feasibility of efficient higher-order transformer 
training and may promote practical applications of tensor attention architectures.

### Strengths
1. The paper proposes the algorithm which can quickly compute the backward gradient 
of tensor attention training in almost linear time by utilizing polynomial 
approximation methods and tensor computation techniques, based on the closed-
form solution of the gradient computation of tensor attention provided in the 
paper.

2. Under hardness analysis, the paper proves that the assumption should be necessary
and tight, meaning that there is no algorithm which can achieve almost linear time in solving 
the tensor attention gradient computation, which is very important for future 
related research.

### Weaknesses
1. The paper's assumptions are relatively strict and may be difficult to meet in real-
world situations. The paper lacks discussion on the practical applicability and 
limitations of the method.

2. The work should talk more about the benefits of using a tensor-based attention mechanism, because otherwise the motivation is quite weak, although I understand its theoretical novelty.

3. The authors should talk more about the tradeoffs of Algorithm 1. Namely what it sacrifices for faster computation, and how does it compares with matrix attention.

### Questions
1. The paper mentions that the "bounded entries assumption" needs to be met. Are
there any studies or experimental results that support that this assumption is correct
in actual transformer models? If this assumption is not met in some actual scenarios,
can your algorithm still remain efficient, or what adjustments need to be made?

### Soundness
3

### Presentation
4

### Contribution
3

### Rating
8

### Confidence
4

---

## Human Reviewer 2

### Summary
Authors consider the tensor attention within the framework of transformer models and propose an algorithm that allows calculating gradients for the tensor attention with linear complexity. While standard attention captures pairwise interactions between tokens, the tensor attention (third-order tensor attention variant) was suggested by [Sanford et al; NeurIPS-2023] in the context of capturing triplet interactions. In the basic implementation, such a mechanism has cubic complexity on the forward pass, and in a subsequent work [Alman et al; ICLR-2024] a new algorithm based on polynomial expansions was proposed that allows (theoretically) to have linear complexity for forward passes. The authors, continuing this line of work, constructed an algorithm for calculating derivatives with linear complexity (theoretically), which potentially allows the method to be practically implemented and overcome the theoretical cubic time complexity barrier both in inference and training.

### Strengths
1. The work is neatly structured and the presentation is consistent.
2. The formulations seem correct and the evidence rigorous.
3. If successfully implemented in practice, tensor attention could be in demand in ML community as it has the potential to account for more complex correlations in data.

### Weaknesses
1. If in [Alman et al; ICLR-2024] work the lack of experimental confirmation of the proposed approach could be attributed to the lack of an effective method for calculating gradients, then in this work, as stated, the last ingredient for the successful implementation of tensor attention has been prepared. In this regard, it is confusing that no practical implementation or numerical experiments are provided.

2. In the context of the previous point, the degree of scientific novelty raises certain doubts and it seems that it is only a certain development of the previous theoretical works [Sanford et al; NeurIPS-2023], [Alman et al; ICLR-2024].

3. The list of references takes up 9 full pages. The reasons for such extensive citations are not entirely clear, and the usefulness of these references for the reader is questionable. For example, after the phrase "Moreover, tensors are crucial in numerous machine learning applications..." there are about 9 references to various works of little relevance to the subject of these manuscript. Instead, the authors could have referred here to a 1-2 review works.

### Questions
-

### Soundness
3

### Presentation
2

### Contribution
2

### Rating
5

### Confidence
3

---

## Human Reviewer 3

### Summary
This paper studies an approximation algorithm for computing gradients of high-order tensor attention. While the exact computation has cubic complexity, the authors prove that the proposed approximation algorithm has almost linear complexity. Also, the authors prove that the assumptions cannot be further weakened to achieve sub-cubic complexity.

### Strengths
1. It is important to study the gradient approximation of tensor attention, which could make this structure practically usable.

2. For proving the results, the authors establish some results of tensor computation, which may be useful for related fields.

### Weaknesses
1. Since this paper  aims to propose a fast training algorithm for tensor attention, the lack of experiments could be a weakness. I am wondering about the practicality of the proposed algorithm. And are there still large gaps to implement or have experiments on these algorithms?

2. I am wondering whether the approximation error would have a major issue when training large models in practice. For example, for large models with deep architectures, will these errors accumulate and cause training difficulties?

### Questions
See above

### Soundness
3

### Presentation
3

### Contribution
2

### Rating
6

### Confidence
2

---

## Human Reviewer 4

### Summary
This paper derives an efficient algorithm to approximate the gradients of high-order transformers. Such models extend the classical self attention mechanism, where only pairwise interactions are modelled, to higher order interactions where e.g. each token can attend to every possible pairs of tokens in the sentence. This model can also be used to model interactions between different modalities or views of a same object. 

The attention model itself was already introduced previously in 2023 and 2024, along with an efficient algorithm to approximate the forward pass computation. The contributions of the submission is to design an analogous efficient algorithm for fast approximation of the backward pass computation. 

The main theoretical result shows that the proposed approximation algorithm can achieve epsilon = 1 / poly(n) approximation guarantee in almost linear time to compute gradients over a sequence of length n. The authors also provide a hardness analysis showing that their assumption are tight, in some sense. 

The contribution is only theoretical, no experiments are provided.

### Strengths
- The problem of efficient computation of higher-order attention mechanism is relevant

- A hardness analysis is provided to strengthen the result.

### Weaknesses
- The paper is poorly written and hard to follow. In my opinion, it is not ready for publication.

- No experiments are provided to demonstrate the effectiveness (and correctness) of the proposed analysis / algorithm.

- Some key aspects of the results are (at least to me) very difficult to get from the paper. E.g., what are the "assumptions" referred to in the statement "_we proved the necessity and tightness of our assumption_"? What were the key novel techniques needed to extend the previous approach to the gradient computation? 

- The proof (in appendix) is very difficult to follow and the results hard to interpret because they are presented in a very convoluted manner. E.g., Lemma 4.1 (which is central to the derivation and understanding of the main theorem) involves a quantity F(x) which is only defined in the Appendix. Furthermore, the definition given in the appendix itself relies on other quantities defined previously in the appendix (S and W), which are themselves  functions of other quantities previously defined (K,V,L...).... After processing backward through this series of definitions, one realize that F(x) is actually a function of many key variables of the problem (A1,A2,A3,Y,...). In the end it is thus very difficult to understand what is Lemma 4.1 suppose to highlight, since F(x) hides many dependencies on central variables of the attention computation. Note that not much context is given before the lemma to help/guide the reader. This is unfortunately not an isolated example.

- This is minor, but there are many grammatical imprecisions / errors that need to be fixed.

### Questions
- Line 260-261 are not clear at all: Z has not been defined, it is impossible to guess what A1 to A5 are. A posteriori, after reading the next page, the reader can retrospectively guess what was meant by this statement but it is, in my opinion, not reasonable to expect the reader to not stumble on these lines and be confused (as I was). 

- Similarly, when reading Definition 3.8 it is not clear at all why the minimization is only w.r.t. X. Only after the definition is explained that minimization w.r.t. the other variables is ignored because the minimization w.r.t. X is the computational bottleneck. In my opinion, the authors should explained this *before* the definition. 

- Line 3.9: unless I am mistaken the log(n) bit model has never been introduced.

- Introducing the column-wise Kronecker product as "one kind of tensor operation" (line 59 and in other places) can be confusing. I would suggest directly calling it by its name along with a forward reference to the definition. 

- The column-wise Kronecker product is known as the Kathri-Rao product. The row-wise Kronecker product is sometimes referred to as the face-splitting product (https://en.wikipedia.org/wiki/Khatri%E2%80%93Rao_product), this should be mentioned. 

- Most of the proofs of the basic facts in appendix can be found in the seminal survey (Kolda and Bader, 2009), e.g. C.7, C.11, C.12. Some are trivial, e.g. part 1 of C.10. While I understand the desire to be self-contained, this long list of basic facts intertwined with less obvious identities  sometimes hinders the readability and understanding of the overall proof. 

**Typos / imprecisions**

line 170: w.r.t -> w.r.t.

line 173: "to denote a length nd vector" is confusing. Directly gives the definition: "to denote the length nd vector obtained by stacking the rows of A into a column vector". 

line 176: "denote an identity matrix" -> "denote the identity matrix" (there is only one). Idem for the following one when defining *the* identity tensor (not "an" identity tensor).

line 181: "And we have X = mat(X)" is not a proper way to introduce a mathematical statement, in my opinion. 

line 180: tesnorization

def 3.2 and 3.3. : "define matrix" -> "define the matrix"

[...]
_I stopped keeping track of such small typos / grammatical errors at this point. The paper needs a thorough proof read._

### Soundness
2

### Presentation
1

### Contribution
2

### Rating
3

### Confidence
4