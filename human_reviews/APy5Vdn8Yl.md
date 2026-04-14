# Differentially Private Network Training under Hidden State Assumption

- Decision: Reject
- Scores: 5, 6, 6, 6

## Abstract
We present a novel approach called differentially private stochastic block coordinate descent (DP-SBCD) for training neural networks with provable guarantees of differential privacy under the hidden state assumption. Our methodology regards neural networks as optimization problems and decomposes the training process of the neural network into sub-problems, each corresponding to the training of a specific layer. By doing so, we extend the analysis of differential privacy under the hidden state assumption to encompass non-convex problems and algorithms employing proximal gradient descent. Furthermore, in contrast to existing methods, we adopt a novel approach by utilizing calibrated noise sampled from adaptive distributions, yielding improved empirical trade-offs between utility and privacy.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The paper proposes a method for training multi-layer neural networks with differential privacy (DP) guarantees in the hidden-state model of DP. This is obtained using a convex dual formulation of the network [(see Zeng et al., 2019, for non-DP literature)](http://proceedings.mlr.press/v97/zeng19a/zeng19a.pdf) and by using the privacy amplification by iteration analysis (Feldman et al., 2018). I see that the contribution is two-fold: a) introduce the convex reformulation for training multi-layer networks using the privacy amplification by iteraiton analysis and b) tailoring and improving the RDP analysis by Ye and Shokri (2022). Experimental results given in the appendix illustrate the effectiveness of the approach.

### Strengths
- I think the idea is nice. Although this convex reformulation (Eq. 3 in the paper) has not become very popular in the non-DP literature and it is unclear to me how well it compares to SGD training of non-convex formulations (and whether it can escape local minima etc.), I think it is interesting to study whether it could be beneficial in DP training. Essentially, all the DP updates are GD steps to convex subproblems so the privacy amplification by iteration analysis seems to fit well.

### Weaknesses
I am not sure about the soundness of the technical part of the paper, i.e., the RDP analysis for the privacy amplification by iteration. Let me explain in more detail:

- The analysis is essentially following the steps of [Ye and Shokri, 2022](https://proceedings.neurips.cc/paper_files/paper/2022/file/04b42392f9a3a16aea012395359b8148-Paper-Conference.pdf). First, the 2-page proof of Theorem A.2 given in the Appendix is a step-by-step copy of the proof of [Lemma 3.1 , Appendix of Ye and Shokri, 2022](https://proceedings.neurips.cc/paper_files/paper/2022/file/04b42392f9a3a16aea012395359b8148-Supplemental-Conference.pdf). While it is ok to repeat the steps, I think it should be mentioned that this is essentially the same proof.

- Second, the Corollary 4.4 also seems to be very similar to  [Lemma 3.2, Ye and Shokri, 2022](https://proceedings.neurips.cc/paper_files/paper/2022/file/04b42392f9a3a16aea012395359b8148-Paper-Conference.pdf). Here is the first point where the analysis deviates: instead of -1 there is $-1/L_T^2$ in that exponent. However, I cannot find the proof for Corollary 4.4 anywhere in the appendix. 

- The rest of the analysis seems to also follow the steps of Ye and Shokri. I see that the only difference is that instead of having the strong convexity parameter $\lambda$ used in the analysis by Ye and Shokri (L2-regularization) you have the Lipschitz constant $L_T$ of the proximal operator. I think there is an inverse relation between $\lambda$ and $L_T$. It remains unclear to me, whether this analysis is really making improvement to the analysis by Ye and Shokri, since here $\lambda$ is hidden in $L_T$ and otherwise similar steps are carried out in the analysis. I think there should be some comparison (e.g., numerical) between these two analyses, it would be easy to carry out. There are some numerical results illustrating the RDP bounds in Figure 2, but the RDP order is not reported nor the Lipschitz constant $L_T$.  I think it should be indicated clearly where you make improvements compared to the analysis by Ye and Shokri.

- I think the experimental comparisons fall short. Those reported RDP parameters of Table 2 would correspond to approx. $(0.1,10^{-5})$-DP, and I find it hard to believe that DP-SGD for MNIST dataset would then give 10\% test accuracy. You say that DP-SGD and privacy amplification by iteration analyses are incomparable, though I think the comparison is ok since DP-SGD guarantees are valid also in the hidden state model. Also, using the analysis by Ye and Shokri for training a logistic regression, it is very difficult to beat the privacy-utility trade-off of DP-SGD, so looking at these positive results, it seems to me that the RDP bounds have to be much lower than those of DP-SGD. And this again raises the question where do the improvement in the analysis really come from (see above). 

Small remark: 

- Also the proof of Lemma 4.1 in the appendix can already be found in the appendix of [Feldman et al., 2018 (Proposition 18)](https://arxiv.org/pdf/1808.06651). 


Due to the above reasons (e.g., a missing proof of a central lemma in the analysis), I think the paper is not yet mature for publication in its current form, though I think the idea is interesting and worth studying. I am ready to reconsider my score after the rebuttals.

### Questions
- Where does the exponent $-1/L_T^2$ in Corollary 4.4 come from ? How to prove Corollary 4.4 ?

- How would the RDP bounds compare to the RDP bounds by Ye and Shokri if you just consider a simple strongy convex model (e.g., logistic regression with L2-regularization) ? Where do the improvements then really come from? What is the relation of $\lambda$ and $L_T$ ?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes a privacy mechanism for non-convex models by transforming it into a sub-convex one, offering privacy guarantees through this relaxation.

### Strengths
The approach is well-organized and is presented in a clear manner.

### Weaknesses
I have some concerns regarding the motivation and advantages of this method compared to the classical DP-SGD algorithm. 

DP-SGD is applicable to non-convex problems and does provide privacy guarantees at stable points. How does the proposed method theoretically and empirically compare to DP-SGD in terms of privacy and utility? The relaxation from a non-convex to a convex problem in the proposed method could impact utility. How does the utility of the proposed method compare to DP-SGD, especially in terms of balancing privacy and performance?  Whether the relaxation to a sub-convex problem offers distinct privacy advantages or if it primarily facilitates privacy analysis? Empirical evidence highlighting this trade-off could help demonstrate the practical benefits or limitations of this approach.

I may adjust my evaluation score depending on these responses.

### Questions
1. There are several typos. E.g., ", which" on line 174.
2. Line 193: use text descriptions instead of math symbols directly in the text wherever applicable. E.g., "for all d"

### Soundness
3

### Presentation
3

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
This paper proposes DP-SBCD, which train neural networks with theoretical guarantees of DP under hidden state assumption

### Strengths
The paper theoretically proved the connection between DP-SBCD and its DP property.

### Weaknesses
Over-sale for the contribution. DP-SGD is feasible in many cases. For example, the number of learning steps is not extremely large.

### Questions
What is the complexity of the proposed algorithm?

### Soundness
4

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces the differentially private stochastic block coordinate descent (DP-SBCD) algorithm for training neural networks with differential privacy guarantees under the hidden state assumption. This method decomposes the training process into sub-problems for each network layer, allowing for a more precise privacy analysis. By utilizing adaptive noise distributions, DP-SBCD achieves a better trade-off between model utility and privacy. The theoretical analysis demonstrates that privacy loss converges under the hidden state assumption, making it suitable for non-convex and non-smooth loss functions in neural networks. Empirical results show that DP-SBCD has lower privacy loss and better utility compared to traditional differentially private stochastic gradient descent (DP-SGD) methods.

### Strengths
- The authors propose differentially private stochastic block coordinate descent (DP-SBCD), which solves non-convex problems with differential privacy guarantees under the hidden state assumption.
- The authors provide insights to understand the convergence of the privacy loss in Theorem 4.5 and Section 4.3.
- The limitations are well discussed. The authors explain that although DP-SBCD shows lower privacy loss than DP-SGD, direct comparisons can be misleading due to different privacy loss assumptions. They also discuss the memory efficiency of the coordinate descent algorithm and the challenges it introduces, such as high variance in mini-batch gradients and the need for large batch sizes to mitigate this.

### Weaknesses
Please refer to the questions.

### Questions
- For the log-Sobolev inequality (LSI) for the distribution of $\theta$ in Lines 309-310 and Theorem 4.5:

1. Is the LSI proven based on Assumption 4.3, or is it an additional assumption?
2. What does the constant $c$ in Line 350 represent, and how is it determined?
3. Given that previous works assumed strongly convex smooth loss functions, how do the authors justify the LSI assumption in this non-convex neural network problem?
4. How does the subproblem viewpoint contribute to making the LSI assumption reasonable in this context?

- What causes the fluctuations in the unseen privacy loss shown by the dashed green and red lines in Figure 1 (right)?

- For the sentence "Otherwise, the privacy loss will increase exponentially" in Lines 78-79?  Are the authors referring to DP-SGD or other methods in this statement? I.e., the DP-SGD leads to exponentially increasing privacy loss in settings that are not strongly convex or smooth?

- For Figure 2 (a):

1. Is there a theoretical explanation for why the curves nearly intersect at K=30?
2. How does this intersection point relate to the convergence of privacy loss discussed in the paper?
3. Can the authors provide a prediction or additional plot showing the behavior of these curves for K>30?
4. Are there any practical implications of this intersection point for choosing the number of epochs in real-world applications?

### Soundness
3

### Presentation
2

### Contribution
3
