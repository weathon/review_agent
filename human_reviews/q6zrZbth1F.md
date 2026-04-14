# How many samples are needed to train a deep neural network?

- Decision: Accept (Poster)
- Scores: 8, 6, 8, 6

## Abstract
Even though neural networks have become standard tools in many areas, many important statistical questions remain open. This paper studies the question of how much data are needed to train a ReLU feed-forward neural network. Our theoretical and empirical results suggest that the generalization error of ReLU feed-forward neural networks scales at the rate $1/\sqrt{n}$ in the sample size $n$-rather than the "parametric rate"  $1/n$, which might be suggested by traditional statistical theories. Thus, broadly speaking, our results underpin the common belief that neural networks need "many" training samples. Along the way, we also establish new technical insights, such as the first lower bounds of the entropy of ReLU feed-forward networks.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors establish a lower bound on the packing number for deep ReLU networks with L1 constraints on the weights. Then they use the packing number bound and Fano’s inequality to provide a minimax risk lower bound that scale as sqrt(log(d)/n). 

The rate of 1/sqrt(n) has been established in a few other settings for neural networks (shallow feedforward networks, linear networks), but not for deep neural networks with non-linear activations.

One of the main implications is the separation between this bound and the known minimax rates for linear methods which scale as 1/n. 

 The authors provide some empirical support to their analysis, by experimenting with small datasets and networks, fitting O(1/n) and O(1/sqrt(n)) curves and showins that the latter achieves smaller residuals.

The question remains open on whether with specific data-dependent and algorithm-dependent assumptions 1/n is feasible in practical applications of neural networks.

### Strengths
Importantly the bound depends on the logarithm of the input dimension, d, and scales as 1/sqrt(n) matching comparable upper bounds. While this type of bound exists before, it had not been established for deep, non-linear networks. 

Great clarity, organization and writing.

### Weaknesses
1. Some of the writing is missing clarity and could be rewritten in a more straightforward fashion. 
- The sentences in lines 38-43
- The sentences in lines 120-123.

2. [Minor] In your proof sketch of Section 4, please properly introduce random variable J (it is only introduce in the appendix).

### Questions
1. Can you come up with some conditions under which the lower bound O(1/sqrt(n)) is actually not valid? For example, what is your intuition about the optimal rate for different architectures? What do you expect for the case when \sigma=0?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper focuses on analyzing the sample complexity of learning deep ReLU feed-forward networks that have a bounded elementwise $\ell_1$ norm. The authors show that the minimax rate scales as $1/\sqrt{n}$, where $n$ is the sample size, and back up their theoretical claims with experimental results.

### Strengths
-	The problem of understanding the sample complexity of learning neural networks is an Important research topic.
-	The research presents a minimax lower bound for deep neural networks, which seems to be new.

### Weaknesses
-	It is not very clear to me why one should compare the results with $1/n$ rate. Based on my understanding, to achieve $1/n$ rate one often needs more structural assumptions on the target function, such as the ground-truth is spare in spare linear regression setting. Given that there is no such structural assumption in the paper, I’m not sure why one would expect such $1/n$ rate to happen.
-	The considered function space is elementwise $\ell_1$ norm bounded, which is not a very commonly considered setting.
-	The lower bound has an exponential dependency on $L$, which is not ideal though it is understandable that there might be technical difficulties to remove this.
-	It is not very clear to me the experiment setup is the right way to verify the correctness of the bound. These bounds are asymptotic bounds, meaning that the rate are only meaningful when $n$ is large enough. 
First, in the current scale of $n$, the difference between $\sqrt{n}$ and $n$ are not very significant. 
Second, the way of fitting the curve seems to consider all the data points, including those relatively small $n$. I believe such curve could not correctly represent the correct decaying rate.
Moreover, I believe these bounds in fact also depend on the norm of weight matrices.  It is possible the norm increases when $n$ increases, which could also affect the rate.
-	The experiment results many also affect by the optimization-related factors, for example converging to a local minima. These factors related to optimization are not considered in the theory, so it seems hard to draw conclusions from experiments.

### Questions
See weakness section above.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper establishes minimax lower bounds for deep ReLU feedforward neural networks in the context of regression. Of note, to establish such lower bounds, the paper establishes a lower bound on the packing number of a ReLU network function class.

### Strengths
The minimax lower bound for deep ReLU networks and the lower bound for the packing number of deep ReLU neural networks is novel and interesting.

### Weaknesses
*

### Questions
* Is it possible to get similar minimax rates if the input $x_i$ are subgaussian random vectors and/or noise are subgaussian?

* In view of lemma 5 of [1] (an upper bound on the generalization error) and theorem 2.2 in the paper, could a similar minimax lower bound be established for a function class consisting of more general architecture (e.g. other lipschitz activations, containing convolutions, etc) where the norm of the layers are bounded and the entire model is lipschitz?

[1] Mahsa Taheri, Fang Xie, & Johannes Lederer. (2020). Statistical Guarantees for Regularized Neural Networks.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes to bound by below the minimax risk of a sub-class of ReLU Multi-Layer Perceptrons (MLP). This sub-class, denoted by $\mathcal{B}\_{\mathrm{L}}$, is built as the $\mathcal{L}^1$-ball of a given radius $v\_{\mathrm{s}}$ of the full space of parameters. The lower bound is obtained by using the packing number of $\mathcal{B}\_{\mathrm{L}}$.

### Strengths
# Originality

This paper attempts to find a lower bound for the minimax risk, which is not common in the literature.

# Clarity

The paper is easy to ready.

# Quality

The proof seems to be sound (I did not check everything).

### Weaknesses
# Originality

The paper uses almost the same tools as in [1], including the $\mathcal{L}^1$ regularization (which becomes the $\mathcal{L}^1$ ball in the space of parameters). The covering number is replaced with the packing number.

# Clarity

The motivation of finding such a lower bound is unclear: why would it be useful to get a lower bound on the minimax risk? What kind of information does it provide on the dataset, the function to approximate and the NN architecture?

Except it fits the framework of [1], what is the motivation for studying a subspace of MLPs defined by the $\mathcal{L}^1$ ball in the space of parameters? Why not choosing another metric?

# Significance

Unclear motivation (see Clarity).

# Quality

## Sharpness of the bound

According to the authors (see paragraph starting line 216), the lower bound is sharp, since it contains the same factors as the upper bound in [1]. Please note that the similarity between the two bounds is easily explained by the similarity of tools used.

But, very importantly, the bound in [1] is obtained with a loss with a strong $\mathcal{L}^1$ regularization term. Why would it be relevant to compare the two bounds?

## Interpretation of $V\_{\mathcal{F}}$

In paragraph starting line 216, the authors interpret $V\_{\mathcal{F}} = (v\_{\mathrm{s}} / L)^L$ as "a product over the $\mathcal{L}^1$-norm bounds of different layers in $\mathcal{F}\_{\mathcal{B}\_{\mathrm{L}}}$". This bound is obtained by performing an AM-GM inequality, which is known to be tighter when all the weight matrices have a similar $\mathcal{L}^1$ norm. However, it may be very loose otherwise, and the proposed interpretation tends to become inaccurate.

## Experiments

The authors have chosen to focus on experiments related to real-world data. This is undeniably useful, but the conditions of Theorem 2.2 are not fulfilled with such datasets. It would have been very useful to have results with a toy dataset, whose data are distributed in accordance to the assumptions (Gaussian inputs, outputs generated by a MLP, Gaussian noise...).


# References

[1] *Statistical guarantees for regularized neural networks*, Taheri et al., 2021.

### Questions
There is little discussion about the number $n$ of samples that are needed to train a NN (only at lines 233-238). What are typical values for $n$?

See above:
 * experiments on toy datasets matching the assumptions?
 * motivation for finding a lower bound?
 * relevance of the comparison with [Taheri et al., 2021]?

### Soundness
3

### Presentation
3

### Contribution
2
