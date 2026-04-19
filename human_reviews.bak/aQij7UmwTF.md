# Generalization Bounds for Magnitude-Based Pruning via Sparse Matrix Sketching

- Decision: Reject
- Scores: 6, 5, 6, 1

## Abstract
Magnitude-based pruning is a popular technique for improving the efficiency of Machine Learning, but also surprisingly maintains strong generalization behavior. Explaining this generalization is difficult, and existing analyses connecting sparsity to generalization rely on more structured compression than simple magnitude-based weight dropping. However, we circumvent the need for structured compression by using recent random matrix theory and sparse matrix sketching results to more tightly tie the connection between pruning-based sparsity and generalization and provide bounds on how Magnitude-Based Pruning and Iterative Magnitude Pruning affects generalization. We empirically verify that our bounds capture the connection between pruning-based sparsity and generalization more than existing bounds.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper studies the generalization bounds for neural networks with magnitude-based pruning. The approach is as follows. First, the paper proposes a magnitude-based pruning algorithm, which randomly sets the non-diagonal weights of neural networks to zero according to a Bernoulli random variable. This is followed by a discretization approach and a matrix sketching approach to improve the dependency of the bounds on the number of trainable parameters. The main result is a generalization bound based on a compression approach. Experimental analysis is also included to verify the assumptions used in the analysis, and to verify the theoretical results.

### Strengths
The paper proposed generalization bounds based on magnitude-based pruning, which does not require structured compression as considered in the existing studies.

The paper considers sparse matrix sketching to decrease the number of trainable parameters, which improves the generalization bounds.

### Weaknesses
Theorem 5.2 requires an assumption $\gamma$ to be no smaller than a number. It seems that this number would be very large since it has exponential dependency on $L$, and $\Gamma_l$ is also large according to the definition. If $\gamma$ is large, then $\hat{R}_\gamma$ would be very large.

To make the high-probability bound in Theorem 5.2 meaningful, one needs to choose very large $\epsilon_l,\lambda_l$ and $p_l$. In this case, the generalization bound would also be large. Furthermore, Lemma B.5 requires $\epsilon_l$ to be sufficiently small. This seems to lead to a contradiction.

In Theorem 5.2, the dependency of the generalization bound on $d$ seems to be $d^{3/4}$. If I understand correctly, if we do not use the matrix sketching, the dependency would be $d$. Therefore, the improvement is $d^{1/4}$, which seems not to be quite significant. As stated in the conclusion, the analysis still leads to a vacuous generalization bound.

In Lemma 5.1, the statement holds with probability $1-1/\lambda_1-d^{-1/3}$. This dependency on $d$ is not appealing since it would require a very large $d$ to get a bound holding with large probability. For example, if we want the bound to hold with probability $1-0.01$, then $d$ should be as large as $10^6$.

### Questions
Please see the comments above.

Minor comments:

Section 3.1: it seems that the definition of $\hat{R}_\gamma$ is not correct, i.e., $\geq \gamma$ should be $\leq\gamma$?

Theorem 3.1 only shows the existence of a $A\in\mathcal{A}$. How to find such a $A$?

Lemma 4.1: $\Gamma_l$ is used without giving its meaning

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
In this paper, the authors present a generalization bound derived from magnitude-based pruning (MBP) sparsity. 
The central idea is that if (a) the pruned model exhibits good generalization, and (b) the original model's performance closely matches the pruned model, 
then the original model also generalizes well. 
The validity of point (a) has been established by previous work (Arora et al.), so the primary focus of this paper is to substantiate point (b).
By assuming that magnitudes follow a Gaussian distribution, the authors demonstrate that pruned and discretized parameters closely approximate the original parameters. Furthermore, the authors introduce the concept of matrix sketching to reduce parameter size, leading to a more favorable bound. Empirical experiments support the authors' claims, showing that their bounds exceed test loss and outperform prior bounds.

Despite the paper's overall significance and the intuitiveness of the generalization bound based on MBP sparsity, there are still notable shortcomings. The most substantial concern lies in the limited ability to validate the proposed bound's practical effectiveness. The assumptions made, particularly regarding the Gaussian distribution of weights, lack robust justification in the experimental context. Additionally, the graphical verification of the bound may not be entirely convincing, as the predicted error bound substantially exceeds the actual error. This raises questions about the bound's validity and practical utility. Rather than focusing solely on the bound, it would be more insightful to verify the underlying assumptions.

Therefore, unfortunately, I cannot recommend an acceptance at the time being.

### Strengths
1. The authors introduce the intuitive concept of utilizing magnitude-based pruning sparsity to address generalization concerns.
2. Unlike prior approaches, this paper directly analyzes the generalization performance of the original model, rather than focusing solely on the pruned model.
3. The authors illustrate the adaptability of their technique to various contexts.

### Weaknesses
Overall, the proposed bound's practical effectiveness is challenging to confirm, representing a primary concern.

1. The assumptions, especially those concerning Gaussian weight distribution, lack adequate empirical validation.
2. The graphical verification of the bound might be misleading due to a significant predicted error bound compared to the actual error. A more rigorous approach is necessary to establish the validity of the assumptions.
3. Considering the assumed Gaussian distribution of weights, it is worth exploring whether better discretization methods exist.
4. The empirical verification of bounds in Fig~3 displays bound values that are excessively large, impairing their practical utility. It would be advisable to offer theoretical comparisons rather than relying solely on empirical evidence.

Minor Concerns:
This paper could significantly benefit from improved writing clarity. The current version may be confusing for readers. Some specific points to consider include:
In the Abstract:
1. Emphasize the necessity of Magnitude-Based Pruning and its advantages over structured pruning for clarity.
2. The use of "However" in the abstract seems to have unclear logic.
In the Introduction:
1. In the first paragraph, provide evidence supporting the claim that MBP significantly reduces memory requirements and inference time.
2. Consider rephrasing the use of "However" in the second paragraph for improved clarity.

### Questions
See limitations.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper aims to provide generalization bounds for pruned models using tools from sparse matrix sketching. The authors focus on developing a generalization bound for Magnitude based pruning methods but also show their proof methodology for other sparse subnetworks.

The proposed method first bounds the difference in layerwise activation norms between the dense and sparse model after pruning and discretization. The generalization bounds are then given by translating the sparse network into a small dense one using sparse matrix sketching and then applying bounds from Arora et al [1], which results in much tighter bounds.

### Strengths
1. The authors propose a novel idea of using sparse matrix sketching to develop generalization bounds for sparse networks.

2. Their establishes a connection between sparse networks obtained via magnitude pruning and generalization.

### Weaknesses
1. The proof follows a structure of bounding the layerwise error and then accounting for discretization of the parameters in order to conform with the setting of Theorem 3.1 based on the proof of Arora et al. [1]. However, the need for discretization has not been motivated clearly in the paper and can benefit from additional explanation regarding the same, which will make the proof easier to follow.

2. The magnitude pruning algorithm assumed by the authors uses a Bernoulli based construction of the sparse mask, however, in practice magnitude pruning is done based on sorting the parameters in each layer (or the entire network). Does the $(j_r, j_c)$ structure in each layer hold then? For pruning methods like Iterative Magnitude Pruning, larger layers are known to have dead neurons (corresponding to all zeros in a row). This problem is especially amplified in a CNN where IMP sets most channels to zero while having very few dense channels. The current proof seems to be unable to handle these situations and how will the assumption change for CNNs.

3. How will the generalization bounds change for different pruning methods, for eg: random pruning or other gradient based pruning criterions. Would random be strictly worse or similar? Insights on different pruning methods will help understand the relevance of the proposed bounds for different pruning criteria.

The key idea of the paper is novel and for the first time establishes generalization bounds for magnitude based pruning methods.

[1] Arora, Sanjeev, et al. "Stronger generalization bounds for deep nets via a compression approach." International Conference on Machine Learning.

### Questions
See above section for questions.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
1: strong reject

### Rating Number
1

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proves new generalization bounds for deep neural networks via a compression based approach. They assume that the weights of the neural network are normally distributed and from this argue that a simple weight pruning approach well approximates the original network. They they argue that after pruning, the network can be represented with few parameters. This allows them to apply a generalization bound of Arora et al. 2018 which depends on the compression error and parameters of the compressed network.

The paper uses two key tools: (1) for their approximation step, they apply random matrix theory results to bound the error introduced by pruning (which due to their assumption of random weights is essentially an iid random perturbation of the weight matrix). (2) To bound the parameters of the pruned network, they argue that its weight matrices are sparse and thus can be compressed with a ‘sparse matrix sketching approach’ which basically allows recovering a sparse matrix X from a sketch AXB^T where A and B have many fewer rows than X.

Unfortunately, I found the results in the paper difficult to grasp — little intuition is given for the bounds and most of the results have undefined quantities which make them impossible to interpret. See my questions below. Further, it is not clear what role the two key technical tools actually have in establishing these bounds. Again, see my detailed questions below but: (1) it is not clear what the improvement is from applying random matrix theory bounds instead of simple Frobenius norm bounds to bound the spectral norm of th erandom perturbation (2) sparse matrix sketching does not represent a sparse matrix X with any fewer parameters than a naive sparse matrix format would. It is is clear that without additional assumptions, doing so would be impossible. Thus, it is unclear what role this tool is playing and why it is being used to bound the parameters of the compressed model.

### Strengths
See full review.

### Weaknesses
See full review.

### Questions
I have included comments/questions below. I starred points that I think are more important and would be helpful to have addressed during the author response period.

Questions/Comments:
- The abstract is confusing as it seems to focus on ML in general. Never mentions the context — e.g. neural networks.
- Overall, paper has lots of typos and grammatical errors. It would need significant proofing before final publication. The intro in particular is very vague and difficult to read. I could not understand from it what the paper was actually doing. E.g. what do you mean by sparse matrix sketching? From what I can tell, this is not a standard term and no citations are given in the intro when the term is introduced. Are you talking about random projection with sparse random matrices? Something else? How are you using this sparse matrix sketching? Is it an alternative to magnitude based pruning? A way of analyzing magnitude based pruning? Something else? Some of these questions become clearer later, but IMO the intro needs to be significantly reworked to be more concrete.
- In 3.1, what is the different between R(M) and R_0(M). Both notations are used and both seem to mean the case when gamma = 0? I.e. no margin? 
-**  I didn’t understand the role of the ‘fixed string’ s in Def 3.1. It is not used in either the definition of G_{A,s} nor in the definition of compressibility. It seems that dropping it would not change the definition nor Theorem 3.1 at all. So what is its role? Relatedly, G_A,s is defined differently in Def 3.1 and Thm 3.1, which I presume is a typo.
- In Assumption 3.1, are the weights *independent* of each other? Or just marginally Gaussian but potentially correlated? (Later I see that independence is needed but this was not clear in the assumption)
- In Remark 4.1, what is meant by ‘diagonal elements’ given that the weight matrices are not square.
-** Given the explanation of the proof in 4.1, I’m not clear on why an assumption of Gaussianity was needed. And I’m not clear why randomized pruning was needed. As long as the original distribution was mean 0 and symmetric, if I just pruned all values below some threshold t, then Delta^l would have i.i.d. mean 0 entries with bounded moments right? And then random matrix theory results could be applied.
- In Lemma 4.1, what is Gamma_l? I couldn’t find where this was defined.
- ** I also couldn’t make intuitive sense of Lemma 4.1. I also don’t see how the bound could possibly not depend on some norm bound for the original input points. Say I just have 1 layer and the non-linearity is actually just the linear function, then if I multiply my input by some arbitrarily large constant C, then x^L and hat x^L will also be multiplied by C and thus the error would be scaled by C. So in this case, the error bound must depend on the scaling of the input points.
-** How does the error bound of Lemma 4.1 compared to the maximum size that ||x^L||_2 could be? Shouldn’t this maximum size be roughly proportional to Product L_l ||A^l||_2. Thus, wouldn’t this make the error bound very weak? As the error itself is large compared to ||x^L||_2?
- Fig 1 — is Lemma B.5 meant to refer to Lemma 4.1?
-** Where does the random matrix theory actually come in to Lemma 4.1? In particular, if I have a matrix with random mean 0 entries, then I can bound the spectral norm of that matrix trivially by the Frobenius norm, which by Markov’s inequality is at most something like n*E[Var(random variable)] which good probability. Random matrix theory results improve this bound to something scaling more like \sqrt{n}. If you instead used the trivial Frobenius norm bound, how would Lemma 4.1 change?
- In Section 5 it was unclear of the Sparse Matrix Sketching idea was novel to this paper, or something from the literature? No citations are given.
- **Say I have a p x p matrix with j*p nonzeros. Then to represent this matrix in sparse matrix format, I need j*p values for by non-zero entries, along with j*p indices indicating their positions. These indices take ~ log p bits each. Thus, I need roughly j*p*log p parameters to represent the whole matrix. I don’t see how sparse matrix sketching is improving on this simple argument. In fact, since it compressed to a \sqrt{jp}logp x \sqrt{jp} log p sized dense matrix, it seems to lose a log factor since (\sqrt{jp} log p)^2 = jp log^2 p parameters.
- ** Related to the above question, an alternative bound based on a simple counting argument is given in Lemma D.1. The claim is that this bound is ‘combinatorial’. But it is not, given that {d_1 d_2 choose alpha} is in a log and thus bounded by roughly alpha*log(d_1 d_2).  I could not make sense of the bound of Theorem 5.2 enough to compare them but I couldn’t understand why Lemma D.1 was obviously a worse bound. We should be able to bound the number of non-zero entries alpha directly using Lemma 5.1 by something like max(d_1,d_2)*max(j_r,j_c). Doing so looks like it would in fact give a stronger bound that Theorem 5.2
- In Lemma 5.1, what is lambda_l? Without this being defined, it is impossible to interpret the theorem. E.g. we could have lambda_l = max(d_1,d_2). Also what roughly is Chi? 
- In Theorem 5.2, it says that d must be chosen to make some expression hold. However, that expression does not depend on d. So I don’t understand what this is saying.

### Soundness
1 poor

### Presentation
1 poor

### Contribution
2 fair
