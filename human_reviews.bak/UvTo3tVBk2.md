# Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues

- Decision: Accept (Oral)
- Scores: 8, 8, 8

## Abstract
Linear Recurrent Neural Networks (LRNNs) such as Mamba, RWKV, GLA, mLSTM, and DeltaNet have emerged as efficient alternatives to Transformers for long sequences. However, both Transformers and LRNNs struggle to perform state-tracking, which may impair performance in tasks such as code evaluation. In one forward pass, current architectures are unable to solve even parity, the simplest state-tracking task, which non-linear RNNs can handle effectively. Recently, Sarrof et al. (2024) demonstrated that the failure of LRNNs like Mamba to solve parity stems from restricting the value range of their diagonal state-transition matrices to $[0, 1]$ and that incorporating negative values can resolve this issue. We extend this result to non-diagonal LRNNs such as DeltaNet. We prove that finite precision LRNNs with state-transition matrices having only positive eigenvalues cannot solve parity, while non-triangular matrices are needed to count modulo $3$. Notably, we also prove that LRNNs can learn any regular language when their state-transition matrices are products of identity minus vector outer product matrices, each with eigenvalues in the range $[-1, 1]$. Our experiments confirm that extending the eigenvalue range of Mamba and DeltaNet to include negative values not only enables them to solve parity but consistently improves their performance on state-tracking tasks. We also show that state-tracking enabled LRNNs can be pretrained  stably and efficiently at scale (1.3B parameters), achieving competitive performance on language modeling and showing promise on code and math tasks.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper investigates the expressivity of linear recurrent neural networks (LRNNs), which is a term synonymous with recent state space models (SSMs) and linear attention variants such as GLA. The paper proves a number of theoretical results on the representation power of different LRNNs with various structures on the transition matrices. First it shows several "necessary" theoretical results, such that simple language recognition problems such as parity and modulus counting can only be solved by LRNNs with more expressive state transitions, in particular those that extend the eigenvalue range from $[0,1]$ to $[-1, 1]$ or complex numbers. A number of "sufficient" theoretical results are also proven for the expressivity of products of generalized Householder matrices (i.e. those found in the recent DeltaNet model), showing its ability to recognize any regular language.

### Strengths
- The theoretical results are very comprehensive and illuminating.
- The paper's writing is very clear.
- There are empirical results on both these synthetic tasks (tracking the theory) as well as more end-to-end language modeling evaluations.

### Weaknesses
- No major weaknesses. Some minor additional inclusions could be helpful (see below)
- The paper's impact may be somewhat limited by the focus on synthetic tasks - it's not clear whether these actually matter in practice for real world tasks, restricting its applicability to large scale models on end-to-end modeling. E.g. these flavor of negative expressivity results for Transformers clearly haven't limited their applicability. I don't think this is a weakness of this particular paper, but a property of this broader research area, which prevents me from increasing my score.

### Questions
1. Do the models with this larger eigenvalue range run into more stability issues, especially at scale (e.g. the largest LM experiments)?
2. For Table 3, it may be nice to include a simple Transformer baseline.
3. Differences between models are sometimes conflated with differences in the neural network architecture (sometimes called the "block"). Meanwhile your results are about the inner recurrent layers and does not account for the block architecture. Does controlling for these two orthogonal factors affect results at all?
4. For Figure 4, does extending the eigenvalue range of Mamba also help on coding/math tasks?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
### Summary of contributions

This paper studies the expressive capabilities of Linear Recurrent Neural Networks (LRNNs) on tasks requiring state-tracking. In particular, they look at the impact of the state-transition matrix structure which is usually a diagonal or a Generalized Householder (GH) matrix with positive values. They provide four theoretical findings promoting the use of non diagonal state-transition matrices with negative eigenvalues.

1) Building on the result from Sarrof et al. 2024 which states that LRNNs constrained to diagonal positive state-transition matrices cannot solve parity, they show the same is true for non-diagonal LRNNs.  
2) Furthermore, to count modulo $m$ ($m\>2$), a LRNN (with finite precision and number of layers) requires at least one eigenvalue of the state-transition matrix to have modulus greater than or equal to one and with non-zero imaginary part.  
3) One layer LRNNs\* can implement finite state-automata whose transition monoid are groups.  
4) Multiple layer LRNNs\* can implement any FSA and thus can recognize regular languages.

\*(with transition matrix structured as product of sufficiently many GH matrices with eigenvalues ranging from \-1 to 1\)

These findings are supported by experiments comparing Mamba and DeltaNet, with and without negative eigenvalues. The results show that the architecture modifications suggested do not interfere with training and stability as well as confirming the applicability of these theoretical insights.

### Strengths
### Strong and weak points of the paper (list):

- The paper is well written and structured which enhanced clarity of the results and overall fluidity. In particular, the few lines introducing each section give a great summary of the key take-aways.  
- The study of LRNNs expressivity, more precisely their capacity to achieve state-tracking, is well motivated through its connection to more complex and higher level problems such as reasoning and handling nested structures. The authors do contextualize their results by comparing and contrasting them with similar studies in the literature.  
- The mathematical arguments presented in the paper are sound and rigorous which suggests the proofs (in appendix) are reliable. However, in some instances, more clarity should be provided (see questions below).  
- Experiments supports not only the main claims of the theorems (e.g. Parity can be solved with negative eigenvalues, but not without), but also the more detailed insights (e.g. different levels of complexity explored in 5.2 State-tracking) and higher level conjectures (e.g. improved performance on language modeling tasks relying on state-tracking like coding). These experimental results make a compelling practical case for the modifications suggested by the theory in particular since the non-diagonal case is conclusive even with only one GH state-transition matrix (DeltaNet) while some theoretical results relied on a *product of $k$* GH matrices.

### Weaknesses
(see above)

### Questions
### Recommendation:

Accept
  
This work brings important clarification on the capabilities of LRNNs when given greater flexibility in their parameterization than the current widely used one. In particular, the result demonstrating the ability of LRNNs (with finite precision, depth and width) to recognize any regular languages independently of the sequence length is a strong one. Moreover, the experiments confirm that these findings do translate in practice without impacting training or stability.

### Questions (to clarify and increase confidence in the assessment):

- Some issues with the references:  
  - 113:  Why referring to Liu et al 2023 for a tutorial on language theory and circuit complexity?  Did you mean to reference *Saturated Transformers are Constant-Depth Threshold Circuits* from Merrill et al 2022 (for circuit complexity)?  
  - 116: Why citing Deletang et al. 2023 for Turing completeness of Transformers with arbitrary precision? Perez et al. 2021 seems to be the appropriate reference for arbitrary precision. *Theoretical Limitations of Self-Attention in Neural Sequence Models* by Hahn 2020 might be relevant for infinite depth.  
- Could you explain how you obtained the equations for $\\mathbf{A(x\_t)}$ and $\\mathbf{B(x\_t)}$ for Mamba in Table 1? I am not able to come to this result from Algorithm 2 and the discretization (equation 4\) of Gu and Dao 2024\. Despite the mention that the matrices are applied to each row of the matrix $\\mathbf{H\_t}$, there needs to be more clarifications since the dimensions of the equations in Table 1 are in contradiction with the ones in equation (1) ($\\mathbf{A(x\_t)}$ is dxd vs nxn). This does not affect the theoretical results of the paper, but the experiments appear to rely on this definition given the chosen expression of $\\mathbf{s(x\_t)}$.  
As suggested by the ICLR AI reviewer assistant, can you

>    1) Provide a step-by-step derivation of $\mathbf{A(x_t)}$ and $\mathbf{B(x_t)}$ for Mamba in Table 1.
>    2) Clarify how these equations relate to Algorithm 2 and equation 4 from Gu and Dao 2024.
>    3) Address the dimension mismatch between Table 1 and equation (1) for $\mathbf{A(x_t)}$.
>    4) Explain how this affects (or doesn't affect) the experimental results, particularly in relation to the expression of $\mathbf{s(x_t)}$.

- How does the comment on shortcut solutions (lines 220 to 222\) relate to the argument on finite precision? (The explanation as to why this example of recurrence can not be implemented in a finite precision LRNN is clear without this sentence.)  
- 229: the statement *“so are unable to learn parity.”* in this context is too strong, the example simply proves that current diagonal LRNNs cannot implement *this* specific FSA for parity. To conclude they cannot solve parity would require a reference to Sarrof et al.  
- 313: Why does the inclusion over the domain of eigenvalues imply inclusion over the sets of matrices (despite $k\neq k’$)? Is it implied that  $k=k’$?  
- 314: Why mention that *“Repeated products of diagonal matrices… remain diagonal…”* given that GN matrices are not in general diagonal?

### Feedback to help improve the paper (not influencing assessment)

- 108 : Given the sentence *“Several studies have explored the expressive power of transformers”* a more comprehensive list would be appropriate. Some additional related references:  
  - Separations in the Representational Capabilities of Transformers and Recurrent Architectures by Bhattamishra et al. 2024,  
  - What Formal Languages Can Transformers Express? A Survey by Strobl et al. 2024  
  - Simulating Weighted Automata over Sequences and Trees with Transformers by Rizvi-Martel et al. 2024  
- Also, any reason why limiting to transformers and not mentioning the line of work on the expressive power of RNNs?  
- 248 : Adding parenthesis in the counting modulo $m$ equation would be preferable to be consistent with line 212 and to prevent any ambiguity on operation priority.  
- Figure 3 : Interesting that DeltaNet with 1 layer outperforms DeltaNet with 5 layers on S\_5 only swaps. Any intuition other than simple over parameterization?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper explores the connection between the eigenvalue range of the state transition matrix of a linear RNN and its ability to perform certain fundamental tasks, including parity and count modulo 3. The results point to changing the allowed eigenvalue range from [0,1] to [-1, 1] both for diagonal state transition matrices and ones constructed via identity plus a rank one component like DeltaNet. The authors show how to practically extend the eigenvalue range of both Mamba and Delta without harming training stability. Finally empirical results are shown on a 340M parameter DeltaNet and 370M parameter Mamba model with and without this eigenvalue range extension.

### Strengths
The paper provides a number of expressivity results which motivate considering state transition matrices with extended eigenvalue range. The first is an extension of previous work that for non-diagonal state transition matrices of a finite-precision LRNN, it still holds that solving parity for arbitrary input lengths requires at least one eigenvalue that is not real and positive (Theorem 1). Second, for modular counting with modulus greater than 2, at least one eigenvalue with a nonzero imaginary part is required to perform the task for arbitrary input lengths (Theorem 2). The paper then turns to practically how to extend the eigenvalue range for diagonal and generalized Householder state transition matrices (identity plus rank 1 component). Finally Theorem 3 and 4 provide two more expressivity results culminating in the  fact that "LRNNs" with state transition matrices that are repeated products of GH matrices each with eigenvalues in the range [-1,1] can recognize any regular language. 

The work also provide a number of empirical results comparing Mamba and DeltaNet with eigenvalues in the range [0,1] and in the range [-1,1] as proposed by the paper. This includes looking at length generalization on a number of these more fundamental tasks discussed in the expressivity results, and the eigenvalue range is typically shown to be helpful. A 340M parameter DeltaNet and 370M parameter Mamba are then trained on 32B tokens of FineWeb with and without the eigenvalue range extension. I will discuss these experiments more in the next section.

Overall the paper is well-written, the results easy to follow, and the limitations clearly discussed. My confidence score reflects that I am not as familiar with related work in this area.

### Weaknesses
The main weakness of the results is that the performance of the eigenvalue range extension is somewhat mixed for the larger models discussed in Section 5.3. In particular, for the LM-harness benchmarks in Table 4, the accuracy is mostly at or below the original model. This means that the work may not have immediate impact on the SSM model architectures used at scale.

It would be very helpful in Table 4 to have the better accuracy/perplexity bolded for each model group. Figure 3 does not have the y-axis labelled.

### Questions
Do you any hypotheses for why for Mamba the [-1,1] model had worse perplexity on the datasets (Fig. 8) while for DeltaNet the [-1,1] model typically has better perplexity (Fig. 4)? More generally, when using this [-1,1] extension, would you have any recommendation for whether to use DeltaNet or Mamba?

### Soundness
4

### Presentation
4

### Contribution
3
