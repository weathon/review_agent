# Differentiable Top-k: From One-Hot to k-Hot

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 4, 4, 2

## Abstract
The one-hot representation, argmax operator, and its differentiable relaxation, softmax, are ubiquitous in machine learning. These building blocks lie at the heart of everything from the cross-entropy loss and attention mechanism to differentiable sampling. Their $k$-hot counterparts, however, are not as universal. In this paper, we consolidate the literature on differentiable top-$k$, showing how the $k$-capped simplex connects relaxed top-$k$ operators and $\pi$ps sampling to form an intuitive generalization of one-hot sampling. In addition, we propose sigmoid top-$k$, a scalable relaxation of the top-$k$ operator that is fully differentiable and defined for continuous $k$. We validate our approach empirically and demonstrate its computational efficiency.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper introduces a new differentiable relaxation of the top-k operator, called sigmoid top-k. The authors derive properties of sigmoid top-k, such as differentiability w.r.t. to $k$ as well as the input $x$ and show that sigmoid top-k leads to an entropy-regularized optimization problem. The paper then discusses different top-k sampling strategies. Finally, the authors show that their proposed method has better scalability than previous approaches and slightly improved performance (w.r.t. test loss).

### Strengths
* S1: The proposed method seems to noticeably improve in terms of scalability compared to previous methods.

* S2: The proposed method allows for continuous values of $k$ that can even be optimized for. When optimizing for $k$, the method seems to robustly converge to an optimal value of $k$ (Fig. 4).

* S3: The authors show that sigmoid top-k is equivalent to solving a binary-entropy-regularized optimization problem (Proposition 2).

### Weaknesses
* W1: The paper is not well presented, e.g., it lacks motivation. Why is it interesting to study relaxations of top-k? Why do we even need to generalize from one-hot to k-hot? For example, discussing related work earlier (e.g., in introduction or as section 2) could motivate this from a scalability perspective.

* W2: Given the very simple datasets used (i.e., MNIST and Fashion-MNIST), the experimental setup does not seem convincing to me. While the demonstration of scalability (Figure 5) appears valid, the performance improvements in Tables 2 and 3 seem negligible and sometimes even inconsistent. The results and their potential significance are also not discussed or contextualized in the text.

* W3: It'd be meaningful to add experiments comparing the shifted sigmoid variants due to the similarity to the proposed approach.

## Comments

* C1: Figures 4 and 5 are quite misplaced from their text (e.g., Figure 4 is on page 6 and only referenced on page 9)

* C2: The font size of table and figure captions does not match the text font size. The authors should refrain from changing font sizes from the official style file.

### Questions
* Q1: I assume that the VAEs from Table 3 were trained on image reconstruction. If so, can the authors provide qualitative comparisons in addition to the test losses?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper develops a framework (sigmoid top-k) for differentiable top-k sampling operations, generalizing the one-hot case to k-hot cases. The sigmoid top-k framework is easy to compute and show efficiency advantage over prior work (SOFT). Moreover, the authors generalize $\pi ps$ sampling from one-hot to k-hot, and discuss how to implicitly learn the k. Empirically, this paper validates the proposed methods on feature selection and sparse representation learning, showing lower loss and faster runtime than prior differentiable top-k relaxations on MNIST and Fashion MNIST.

### Strengths
**Unified theoretical framework** This paper connects differentiable relaxations (softmax to sigmoid top-k) and differentiable sampling (categorical to $\pi ps$) under one consistent view, which provides conceptual clarity that was missing in prior work. Moreover, the authors shows that the output of sigmoid top-k can be understood as a regularized optimization problem (see Proposition 2) which provides mathematical explanation of the algorithms.

**Novel Top-k algorithm** The authors propose an efficient scalable differentiable top-k algorithm, and demonstrate its efficiency advantage over other sampling algorithms (see Figure 5). Moreover, the authors propose using phantom unit mechanism to handle the emergent non-integer k during the learning process. Finally, the algorithm replies a simple root solver which is easy to implement and memory-efficient, enabling applications in high-dimensional problems.

**Clear presentation** The paper is well-written, with comprehensive properties and illustrative examples of the top-k sampling.

### Weaknesses
**Limited theoretical novelty** Though this paper offers an elegant and unified framework connecting the k-hot sampling with differentiable relaxations, much of its contribution builds upon prior works. The proposed sigmoid top-k framework can be viewed as a natural extension of softmax when constraining $\sum p_i=k$ instead of $\sum p_i=1$. Moreover, as is pointed out in the paper, entropy-regularized optimization problem and k-capped simplex have been studied in prior work. Therefore, I think the novelty of this paper lies in the combination of these existing ideas rather than discovery of them.

**Lack of deeper analysis of optimization behavior** This paper lacks discussion of how the error of root solver affects gradient accuracy, convergence stability, or numerical conditioning in practical optimization. Similarly, while the authors prove that the straight-through estimator yields a first-order approximation of the true gradient for top-k sampling (see Proposition 3), there is no analysis of bias, variance,  and comparison with existing estimators.

**Limited empirical evaluation** The experimental section demonstrates clear improvements on two relatively simple tasks: feature selection and sparse representation learning on MNIST and Fashion MNIST. Though promising, the tasks seem too simple to convince me that this method does work in more general tasks. 

**Missing interpretation in the context of probabilistic modeling** While the paper discusses the $\pi ps$ sampling design, it lacks the discussion of how the proposed algorithm will affect the probabilistic modeling. For example, does the sigmoid top-k lead to a normalized probability model over subsets?

### Questions
**Question 1** Does the algorithm applies to beam search? I'm thinking of the application of sampling in LLMs where one needs to sample a sequence of tokens. If the algorithm can demonstrate superior performance when applied to beam search, this can be a huge plus.

**Question 2** How does the algorithm work in the multi-label classification problems?

**Question 3** What is the trade-offs between the entropy regularization strength and approximation accuracy of the true gradient? Can you rigorously characterize it?

### Soundness
2

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The authors propose a novel differentiable relaxation to the top-K operator as the sigmoid-k function. They define the operator for continuous k, handling implicit values of k in the process. They show that this operator can be combined with stochastic relaxation techniques similar to gumbel-softmax and straight through estimator. Empirically the method is evaluated on MNIST and Fashion-MNIST.

### Strengths
1. The method is technically sound and seems simple to implement. 
2. The algorithm models the inclusion probabilities exactly.

### Weaknesses
1. Exposition of the method and prior work can be improved.  Section 2, I believe is poorly organized for a novice reader with limited background. $\pi$ps sampling (which I believe is an R package and not an algorithm?) is a bit hard to understand what it exactly is, even after multiple reads of the paper. 
2. The limited evaluation setup might not convince the reader of the utility of the method proposed. I believe prior works have shown the superiority of their categorical modeling on more challenging and recent tasks like stochastic beam search. I am not sure if something like that would be possible. 
3. A clear (empirical) comparison with the baseline methods considered (and prior work) would help the appreciate the novelty and technical challenges better. There are a lot of baselines which are considered in the experiment section like SIMPLE, Soft Top-k (among others). A baseline subsection clearly elaborating and explaining the baselines would help the paper.

### Questions
1. Should the conditions in Eqn 15 be switched? (Just clarifying, I might have every well mis-understood the equation or made error in my calculations)

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
Nice work in sense of technology development and extending stochastic/relaxed discrete sampling. However, the main problem are 1)not enough large scale exerperiment 2) not good writting 

The writting and experiments looks to me potentially purely generated by Chatgpt /claude code, which I could be wrong. I request to see the code.

### Strengths
discrete representation optimization is an important problem in machine learning , such as sparse latent, quantization etc. Traditionally, there are in general 3 ways to walk around the non-differnetiability problem 1) fake it (eg. STE) 2) relax it (gumbel softmax) 3) shortcut it (eg. REINFORCE, each with its own problem 

Most research in the field focus on one hot representation, and relatively little has been done on top-k. This makes contribution by this manuscript unique.To me personally, two major contribution by this manuscript are 1) the sigmoid top k which serves as a relaxation ( in a similar style as Hinton sigmoid straight through) and 2) the implicit "top k" ( which I think the name is misleading)

### Weaknesses
The few weakness:

1) the introduction part is not well written and difficult to follow, looks very likey a LLM generated piece of text 

2) value of implicit top k and application is unclear. 

3) The author mentioned mixture of expert, ,however most recent works on MOE are no longer using this style of discrete choice now. Correct me if I am wrong. 

4) most importantly: the experiments are too simple and to small scale. A large scale experiment on modern architecture are definitely needed

### Questions
I am curious about formulation in equation 9.  how about the bias and variance your relaxation? Also, shared by all relaxation approach, how is the sampling distribution( during inference) distribution from model trained using your method different from model trained using other estimation method  

If the author can 1) show that the manuscript and code are not LLM generated 2) explain behavior of their relaxation method 3) provide at least one solid large scale experiment , I am happy to consider raise the score to >= 6

### Soundness
2

### Presentation
2

### Contribution
3
