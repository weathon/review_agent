# Koopman-based generalization bound: New aspect for full-rank weights

- Decision: Accept (poster)
- Scores: 6, 8, 5

## Abstract
We propose a new bound for generalization of neural networks using Koopman operators. Whereas most of existing works focus on low-rank weight matrices, we focus on full-rank weight matrices. Our bound is tighter than existing norm-based bounds when the condition numbers of weight matrices are small. Especially, it is completely independent of the width of the network if the weight matrices are orthogonal. Our bound does not contradict to the existing bounds but is a complement to the existing bounds. As supported by several existing empirical results, low-rankness is not the only reason for generalization. Furthermore, our bound can be combined with the existing bounds to obtain a tighter bound. Our result sheds new light on understanding generalization of neural networks with full-rank weight matrices, and it provides a connection between operator-theoretic analysis and generalization of neural networks.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proves generalization bounds for deep neural networks by establishing that they belong to the RHKS of sobolev spaces of a given order: in the most basic result (theorem 1 and proposition 5), a very concrete bound is shown which under the assumption that the transformations associated to the weights in each layer are invertible. The bound scales as the product of the $s$th power of the operator norms of the weights divided by the square root of their determinant. There is also a factor  $\|K_{\sigma}\|_H$ which depends on the activation function. Furthermore, follow-up theorems extend the results to the situation where the weight matrices are not invertible but only injective: in this case, a similar bound holds with the determinant of the weight replaced by the square root of the determinant of $W^{adj} W$, but at the cost of a product of factors $G_j$, which depend on the isotropy of the networks' function. Finally, further generalizations are provided which completely circumvent the need for injectivity by considering an augmented version of the network where each layer outputs a copy of its input (in addition to outputting the usual output). It is also shown that the bounds can be combined with existing bounds by employing one method for some of the layers and another for the remaining layers. Experiments demonstrate that imposing regularization inspired from the bounds provided improves performance.

### Strengths
This is an **extremely interesting direction** which, to the best of my knowledge, is underexplored. The results presented in this paper have the potential to be of great interest to the community for further research, since it approaches the problem from an entirely different perspective: instead of controlling the function class capacity through norms of the weights or number of parameters, properties of the learned functions in terms of their smoothness over the inputs are used instead. This may really be a key to more a more satisfying approach to generalization bounds in the overparametrized setting.


Given the great potential this work has, I am quite disappointed that the treatment offered doesn't more thoroughly study in a reader-friendly way the concrete implications in terms of lack of architectural dependencies for concrete networks. Fortunately, ICLR allows authors to upload a new pdf with very substantial revisions, so I am looking forward to that and may increase my score to 8 if my doubts are all thoroughly resolved.

### Weaknesses
The writing is quite crisp and abstract, sometimes to the detriment of precision. I think the results make sense in that bounds for the norms of the neural network are indeed given, but the interpretation in terms of asymptotics and the lack of dependence on the number of parameters don't fully make sense due to the presence of obscure quantities with unclear asymptotic behavior. 

 To be honest, I am **not completely convinced of the correctness** of the final conclusions from a mathematical standpoint. In particular, the notations used by the authors rely a lot on Sobolev norms which are then absorbed into the O notations as if they were constants, and this process is not performed nearly carefully enough to ensure that no additional dependency on the number of parameters appears.  Here are a few examples where there are imprecisions which must be very thoroughly cleared up during the rebuttal for me to keep my score or increase it: 

1 (may be fixable): Proposition 5 includes factors of $\|K_{\sigma}\|$. According to the authors, such factors can be controlled by Proposition 1, which seems to be the justification for absorbing those factors into the $O$ notation in equation (1). I can sort of believe that the final conclusion, but note that the proof given is at best sloppy: each term in the sum over multi-indices with components living in a set of cardinality equal to the layer width is bounded individually by a term $C_{\beta,\gamma,\delta}$, thus, this analysis cannot be used directly to obtain a result truly of the order claimed in equation (1), since the use of this proposition would introduce **an additional dependency on the width of the network**. Now, I agree that this is probably avoidable by using a **component-wise** activation function, which would make most of the terms cancel in the sum,  but this assumption is not even clearly stated, nor is it used anywhere. There are many missing details for the results to truly qualify as applicable to concrete neural networks.


2. (more serious) I am not convinced by the applicability of the result in the case of injective but non bijective maps: the bound contains the factors $G_j$, which depend on the "the isotropy of $f_j$", with $f_j$ only definable via the composition of the network's functions. It is absolutely unclear to what extent these quantities can be considered as constants. Even in the extremely hypothetical case where $G_j\leq G$ for some absolute constant $G$, there is still an exponential dependence in the depth of the network which is not explicitly written in the $O$ notation. In addition, it doesn't seem to be the case that the $G_j$ can be controlled properly either. In page 6, below lemma 8, the authors attempt to reassure the readers by giving an example where $G_j$ can be bounded by $(4/\pi)^{dim(R(W_j)^\top)/4}$. Note that even in that case, **this term introduces intractable, exponential dependence** in the architectural parameters $L$ (the depth) and $dim(R(W_j)^\top)$ (which is closely related to the width). 


3. Similarly, the argument in appendix B is a bit vague and certainly appears to introduce dimensional dependence. 







==============Minor comments (maths)=======


In general, it would be nice to make the paper more reader-friendly by adding more lines in the calculations for readers not familiar with the techniques used. For instance, it would be nice to remind the readers in a separate theorem of the Faa di Bruno formula mentioned on page 13 at the beginning of the proof of Proposition 1. 


For instance, in the proof of Theorem 2 on page 14, the fact that the $s_i$s are Rademacher variables is not even explicitly stated. 

The implication of Assumption 2 should be explained in terms of concrete assumptions about the network and the inputs to the network. I know that later theorems rely on a concrete formula for $p(\omega)$ which appears to make this assumption hold trivially, but this absolutely should be explained explicitly. It also seems like not choosing the $p(\cdot)$ earlier on 

The spaces $R(W_j)$, which refer to the column spaces of $W_j$, should also be defined somewhere. The same applies to the non-standard notation $W^{-\star}$, which refers to the inverse of the adjoint of $W^{*}$. It also seems a little strange to use conjugate transpose notation (without introduction or justification) when all the weights are presumably real). 

In addition, some of the basic notation for Fourier analysis and the relevant inner products absolutely should be included. Note that various authors use different constants in the definition of the Fourier transform, and it is not clearly stated in this paper which convention is used. I deduced from the first line of the proof of Lemma 3 on page 13 the convention used is the one where there is no constant factor in the definition of the Fourier transform. Similarly, the second equality on the same line takes some time to digest without additional details. 

Similarly, the fact that the Sobolev norms are both equal to sums over all multi-indices of the relevant derivatives, and to the RKHS norm defined at the bottom of page 3 should be explained in much more detail. 

Also, the main paper and the appendix are too heavily reliant on each other, it is better practice to make the appendix fully mathematically self-contained, which would imply at a minimum the following reminders to the readers: the definition of $J\sigma^{-1}$ on page 13, a reminder of the definitions of $F_{inj}$ on page 14 in the proof of Theorem 6, a better separation go Proposition 11 from proposition 10 on page 16 (since it actually refers to a completely different setting), and the definition of $G_j$ at the beginning of page 15. 





============minor typos/ language=======


Fourth line of the introduction: "a large number of parameters make the complexity" should be "a large number of parameters makes the complexity" 

at the beginning of remark 1, it would be better to write " Let $g$ be a smooth function which doesn't decay at infinity (e.g., a sigmoid), ..." 

In the beginning of Section 4.3, "as a result $h\circ W_j$ does not contained in..." should be "as a result $h\circ W_j$ is not contained in..."

Section 4.3.2 "We only need...., not whole $W_j$" should be "We only need...., not the whole of  $W_j$"


Just before the second equation in Section 4.4 " set of all functions which has" ==> "set of all functions which have" 

Page 9 "we constructed a network and learned it" ===> "trained" 


Top of Page 18 in Appendix B, there shouldn't be a capital letter at  "then By"

### Questions
What is the norm of the bump functions $\phi_j$ assumed to satisfy in Proposition 10? Are they constant, or equal to 1? How can this be achieved without additional dependence on the ambient dimension at all? Could you provide a concrete example? 

Could you address the points mentioned in the weaknesses, especially the control over the quantity $G_j$ (from lemma 8)and avoiding dimensional dependence when summing over multi-indices in Sobolev norms? 

Could you write a complete and detailed version of your result for a fully concrete example where the loss function is either the cross entropy loss or the square loss and the activation is component-wise and fully explicit (e.g. the smooth version of leaky relu), without using any undefined quantity such as $G_j$, $f_j$ or even $\phi_j$ (you can choose a concrete $\phi_j$ if necessary, and show that the corresponding factors in the bound are bounded by the absolute constants)?  How can you concretely control the quantity $G_j$ in the case where we have a two-layer neural network with a very wide hidden layer?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper establishes a new generalization bound for neural networks based on the Koopman operator. To be specific, the authors first represent the network by the product of Koopman operators. Then, new upper bounds of Rademacher complexity are derived for invertible, injective, and non-injective weight matrices, respectively. Furthermore, the Koopman-based bound is combined with other generalization bounds such that both high and low layers can be sharply bounded. Finally, numerical results validate the effectiveness of the regularization induced by the Koopman-based bound, and the different behaviors of singular values of the weight matrix for each layer are also observed.

### Strengths
The proposed generalization bound is sharp and fills the theoretical gap. Specifically, benefiting from the denominator induced by the Koopman operator, the generalization bound can be sharp when the condition number of the weight matrix is small. What’s more, if the weight matrices are orthogonal, the bound reduces to 1 and is independent of the width of the network. This result explains the generalization ability of neural networks when the weight matrices are full-rank.  By contrast, existing results either depend on the $(p, q)$ matrix norm, which scales by the order of $d^{1/p}$ for a $d \times d$ matrix, or become loose when faced with high-rank weight matrices.
- The authors validate the proposed bound with the help of numerical results. On one hand, experimental results on both regression and classification validate the proposed generalization bound. On the other hand, the different behaviors of singular values of the weight matrix for each layer are also observed for AlexNet on the CIFAR dataset.
- This paper is well organized, which makes it easy to understand.

### Weaknesses
- The authors mainly consider the neural networks with dense layers. I wonder whether these theoretical results can generalize well to neural networks with other structures such as convolution. A simple explanation is recommended.
- The experimental results on MNIST validate the effectiveness of the induced regularization term. Can it boost model performance on datasets with larger scales such as CIFAR?
- Besides, there are some typos. For example, 
	- In the introduction part, "depth of the network.Another approach" should be "depth of the network. Another approach". 
	- In the introduction part, the third paragraph has an extra indent.
	- In Table 1, a larger line spacing is recommended.

### Questions
Please refer to Weakness

### Soundness
3 good

### Presentation
4 excellent

### Contribution
4 excellent

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper provide an operator-theoretic approach to analyzing networks. They proposed a novel bound for generalization of neural networks using Koopman operators and Rademacher complexity, which reveals a new aspect of neural networks.

### Strengths
1.	This paper proposed a new complexity bound that involves both the norm and determinant of the weight matrices. This bound is particularly useful when the condition numbers of the weight matrices are small. 
2.	It provides a new perspective on why networks with high-rank weights generalize well. By combining our bound with existing bounds, we can obtain a more comprehensive description of the role of each layer in the network. 
3.	This paper presented an operator-theoretic approach to analyzing networks, using Koopman operators to derive the determinant term in our bound. This approach offers a new way to analyze the generalization performance of neural networks.

### Weaknesses
This paper gives the generalization error bound of neural networks from a novel perspective which sounds very interesting and introduces new tools to generalization analysis. But since I'm not familiar with dynamic-based Koopman operators, I have some concerns that I'd like to see answered by the author.

1. As the author said, Efficient learning algorithms have been proposed by describing the learning dynamics of the parameters of neural networks by Koopman operators. It seems that the author represents the composition structure of neural networks using Koopman operators, and then uses the complexity method to give an upper bound. My question is, dynamics sound algorithm-related, while complexity is algorithm-independent, so what's the point of using Koopman operators here.

2. Looking at the entire proof, the conclusion of this paper seems to be highly related to the hypothesis of RKHS. My intuitive feeling is that the conclusion of this paper mainly comes from the RKHS assumption, and whether the neural network is abstracted into Koopman operators has little relevance. I hope the author can explain the relationship between Koopman operators, RKHS and the final conclusion and give an idea of what kind of these techniques/assumptions play a role in the proofs.

### Questions
See above

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
