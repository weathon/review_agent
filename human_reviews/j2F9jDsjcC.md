# Train Short, Test Long In Combinatorial Optimization

- Decision: Withdrawn (Treated as Reject)
- Scores: 1, 3, 5, 5

## Abstract
The inherent characteristics of the Transformer enable us to train on shorter datasets and extrapolate to testing on longer ones directly. Numerous researchers in the realm of natural language processing have proposed a variety of methods for length extrapolation, the majority of which involve position embeddings. Nonetheless, in combinatorial optimization problems, Transformers are devoid of position embeddings. We aspire to achieve successful length extrapolation in combinatorial optimization problems as well. As such, We propose an entropy invariant extrapolation method (EIE), which obviates the need for positional embeddings and employs varying scale factors according to different lengths. Our approach eliminates the need for retraining, setting it apart from prior work. Results on multiple combinatorial optimization datasets demonstrate that our method surpasses existing ones.

## Human Reviews

## Human Reviewer 1

### Rating
1: strong reject

### Rating Number
1

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper tackles combinatorial optimization (CO) tasks using a transformer. The transformer, that according to theorem 1 is permutation equivariant, is used inside a simple algorithm (see Alg. 2) for tackling CO. The intuition is that the model can be learnt using sequences of a certain length, but then be tested using larger sequences and it should still work. The paper provides an intuition based on the entropy on how this can happen and formulates the objective following this intuition. The paper conducts experiments in 3d bin packing, traveling salesman problem (TSP), and capacitated vehicle routing problem.

### Strengths
+ CO tasks are important in theoretical CS and have diverse applications. 

+ The generalization in CO tasks beyond the instances the models are trained on is a significant challenge.

### Weaknesses
The paper presents a large number of weaknesses, which largely outweigh the benefits. 

- There is **no related work on CO**, while there are multiple papers on this topic the last few years. A more proper search is required to include and discuss the related work. To provide concrete references, the popular “Pointer networks” (Vinyals et al) and the papers citing it, or the thorough surveys of a) Machine Learning for Combinatorial Optimization: a Methodological Tour d'Horizon (Bengio et al), b) Combinatorial Optimization and Reasoning with Graph Neural Networks (Cappart et al), c) End-to-end constrained optimization learning: A survey (Kotary et al) provide a good starting point. The lack of related work makes it unclear where the paper is placed with respect to the literature. 

- Theorem 1 is incomplete. For instance, neither any definition, the theorem or the proof refer to the exact formulation of the transformer that is being “proved”. In addition, the proof does not offer an explanation of why each component (e.g. softmax, layer norm) are permutation equivariant. 

- The paper requires thorough proofreading, as it currently is not a format ready to be accepted by top-venues, such as ICLR: a) “Xavier initialization” (versus “Xaiver”), b) “are independent of n, Thus”, c) “As such, We propose”, etc. 

- The experimental validation is unclear and non-standard, which does not make it easy for the reader to understand. The vehicle routing and the TSP are some problems that are commonly met, but a reference to the literature would be helpful for the readers that want to find out more. However, in all tasks, there are no comparisons with other methods (see the lack of literature mentioned previously). There are decades of literature in those tasks, with heuristics and/or deep learning methods developed. **The lack of comparisons makes it impossible to evaluate this method**.

### Questions
- What is a “box” mentioned in sec. 4? 

- Is there any proof of the claim “a well-trained transformer is accustomed to a certain range of entropy”? Could the authors elaborate on that? 

- The conclusion mentions that the model has only a few hundred parameters, however there is no table referring to the parameters or the runtime comparison of the proposed models. Could the authors elaborate on that? 

- I am wondering why those specific tasks are selected. Does this approach extend to other tasks? 

- What are the exact contributions with respect to the Press et al, ICLR’21 paper?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
1 poor

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper studies "length" extrapolation in the context of Transformer-based combinatorial optimization solvers. The authors propose an entropy invariant extrapolation method called EIE to improve the extrapolation ability of Transformers. Specifically, EIE learns the derivative of the temperature in attention modules. The authors show that EIE leads to some performance gains experimentally.

### Strengths
- It's important to study extrapolation for neural-network-based combinatorial optimization solvers. 

- The proposed method shows some performance improvement in the experiments (although it's unclear how much computational overhead it brings).

### Weaknesses
**Related works are significantly under-discussed and under-cited.** 
- In Sec. 1, relevant papers should have been cited to support the claims (e.g., the significance of combinatorial optimization problems, failures of traditional solvers, the emergence and success of neural-network-based solvers, etc).
- Sec. 2 misses many related papers. To name a few:
   - [1] Location Attention for Extrapolation to Longer Sequences.
   - [2] Monotonic Location Attention for Length Generalization.
   - [3] Induced Natural Language Rationales and Interleaved Markup Tokens Enable Extrapolation in Large Language Models.
   - [4] From Local Structures to Size Generalization in Graph Neural Networks.
   - [5] Conditional Positional Encodings for Vision Transformers.
   - [6] The Impact of Positional Encoding on Length Generalization in Transformers.

**Factual errors.**
- Def. 1 is called _permutation equivariant_ in literature.
- Thm. 1 has to specify "Transformer _encoders_", since Transformer decoders are not permutation equivariant.


**Problematic assumptions.**
- In Sec. 4.1, each element in $\mathbf{X}$ is assumed to be Gaussian without enough justification. 
- Furthermore, in Equation 6, it's claimed that $\\mathbf{Q}\_{i,:}$ and $\\mathbf{K}\_{j,:}$ are Gaussian, but their joint distribution is not specified. I check the code and it seems that the authors implicitly assume $\\mathbf{Q}\_{i,:}$ and $\\mathbf{K}_{j,:}$ are independent because of these two lines:
```
    q=torch.randn(batch_size,1,dim)
    y=torch.randn(batch_size,dim,N)
```
$\qquad$This is problematic to me because $\mathbf{Q}=\mathbf{XW}_Q$ and $\mathbf{K}=\mathbf{XW}_K$ are generally not independent by definition (unless $\mathbf{W}_Q$ and $\mathbf{W}_K$ satisfy some strong assumptions).

**Weak evaluation.**
- The authors should have also discussed the efficiency of the proposed method, as it introduces an additional neural network.

**Presentation issues.**
- Sec. 4.1, "the entropy of the i-th row": i should be italic.
- Redundant "," in Equations 1-3.
- In Figure 2, legend and numbers are too small for readers. The caption is very unclear.

### Questions
See **weaknesses**.

### Soundness
2 fair

### Presentation
1 poor

### Contribution
1 poor

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The authors propose a way to adapt pretrained permutation invariant transformers trained to solve combinatorial problems to longer sequences. The method does not require any re-training of the underlying transformer model.

### Strengths
- Combinatorial problems are an interesting application of permutation invariant transformers
- The results are impressive and show an increase on longer sequence length problems without requiring retraining.

### Weaknesses
- Theorem 1 has been proposed in previous works. I believe the first work was [1]. This should be cited as such in the text.

- The assumption that the inputs after layernorm are Gaussian may be too strong of an assumption. For instance, if the input experiences a feature collapse, then all the values could collapse to a constant even after layernorm. While this is unlikely, it would be more reassuring to present some empirical result of the distributions of the features after layernorm on the models which are already trained.

- I do not understand the statement below equation 8. You state that you need to ensure the derivative is positive and it would be difficult to ensure that equation 8 holds if you were to use a neural network to directly approximate $\lambda$. In order to enforce the positivity condition, you apply a non-negative activation function to the derivative approximation network. Couldn't this same non-negative function be used on a network to directly approximate $\lambda$? If you were to do this and check every $n$ in the integration, you would find that they are all positive, which means that the positivity condition has been met.

- I do not understand the justification for such a complex opertaion to integrate and train the model (algorithms which are presented in Algorithm 1 and 2). Building on the point above, if $\lambda$ could be predicted directly, the process would become much simpler. If indeed $\lambda$ cannot be predicted directly, then it would be good to include an ablation study of what happens when it is attempted to be directly predicted, the outcome of this study would validate the hypothesis presented about predicting the derivative.

- Figure 2 has an almost meaningless caption which doesn't describe the figure well.

- The conclusion states that the model contains only a few hundred parameters, and that training time can be done on a CPU. Doesn't the entropy calculation on line 5 of Algorithm 2 require the inputs in order to train? If so, then the underlying transformer model must be run in order to calculate equation 6. Even if it does not require tracking gradients for the underlying model, it seems like this should be mentioned, because it could potentially be a huge transformer.

### Questions
- At the end of section 2, it is stated that "In the aforementioned work, the logarithm log is presumed to have the natural number e as its base." Why not just use $\ln n$ in the expression if this is the case?  

- Why is it that other experiments which utilize permutation invariant transformers do not suffer from degrading performance when extrapolating to longer sequences during testing? For instance, in [2] (Table 1), the performance of a permutation invariant transformer sees an increase in performance when drastically increasing the set size during testing. Do you have any intuition why this might be the case even though it no doubt exhibits the same increase in entropy in the attention matrix?

### References
 - [1] Set Transformer - https://arxiv.org/pdf/1810.00825.pdf

 - [2] Slot Set Encoder - https://arxiv.org/pdf/2103.01615.pdf

---

Overall, I think the results presented are interesting, but I do not see a clear justification of the proposed method (see points above in "Weaknesses"). If the authors can present a concrete reason why the integration is needed (with an ablation study if possible), I would consider raising my score.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
In this paper the authors propose an adaptation strategy for making transformer models to combinatorial optimization problems. Essentially, the key element of novelty resides in the "entropy invariant extrapolation" method (EIE). This builds on some adaptations for making transformer models work on these problems: 

- permutation invariance should be guaranteed; 
- the entropy of the output for multi-head attention should be as low as possible (given that it passes through some softmax, such quantity is bound);
- an auxiliary neural network is trained to tune the inverse temperature in the softmax, which tunes as well the entropy.

The network architecture design relies on ReQUr and ReQU activations. The experiments are conducted on some problems of combinatorial optimization, and the comparison is performed against other fixed inverse temperature scalings.

### Strengths
- the employment of transformers for combinatorial optimization is interesting, more specifically the employment of a transformer-like architecture seems intriguing
- the motivation behind permutation invariance is very clear and further motivated in the appendix
- the experiments are averaged on 5 different seeds

### Weaknesses
- the experimental section is very limited
- the text is in many points difficult to follow (eg. in (4) it is unclear what $(O_1, O_2,..., O_h)$ is - is it a concatenation? Or rather, is the "box" definition in Sec.4.1 is never provided)
- Sec. 2 is essentially a "list" of works for IPO with not much connection, and the discussion around NPO is very limited
-  there is no ablation study regarding the designed approach
- in most of the experiments, the proposed approach is within 1std other fixed $\lambda$ approaches (which do not involve the optimization of $\lambda$ and for this reason are computationally less expensive). For this, the proposed approach does not seem very effective
- the quality of the figures is not great - for example, Fig.2b requires massive zoom, and the captions are in general not well-explanatory

### Questions
- Can you compare EIE to different annealing policies for $\lambda$ (exponential, linear)?
- Can you provide a study as a function of the depth of the employed transformer model?
- Can you elaborate more on why in Fig.2b the red curve behaves better than the blue, and yet is outperformed?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair
