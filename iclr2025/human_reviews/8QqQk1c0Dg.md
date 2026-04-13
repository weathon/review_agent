## Human Reviewer 1

### Summary
This paper theoretically analyzed the influence of heavy-tailed gradient noise on the convergence of AdaGrad/Adam (and their delayed version) and their clipping version. The authors found that clipping improves the convergence of AdaGrad/Adam under the heavy-tailed noise, which is validated by some experiments.

### Strengths
1.The authors proved that AdaGrad/Adam (and their delayed version) can have provably bad high-probability convergence if the noise is heavy-tailed.

2.They also derived new high-probability convergence bounds with polylogarithmic dependence on the confidence level for AdaGrad and Adam with clipping and with/without delay for smooth convex/non-convex stochastic optimization with heavy-tailed noise.

3.Some empirical evaluations validated their theoretical analysis.

### Weaknesses
1.The authors stated “Adam can be seen as Clip-SGD with momentum and iteration-dependent clipping level)”. And, the results of Adam/AdaGrad show that  their high-probability complexities don’t have polylogarithmic dependence on the confidence level in the worst case when the noise is heavy-tailed. However, they didn’t explain why the latent clipping brings the negative result (Theorem 1), which is not consistent with their rest results (Theorems 2-4).

2.Some comparisons of results are provided behind Theorems 1 and 4. These comparisons are not clear enough to emphasize the advantages of the results of this paper. Making a table to present all results of this paper and previous work may benefit readers' understanding.

### Questions
1.Line 13: What is the meaning of “for the later ones”? Does “the later one” denote Large Language Models, and why? Do other models not have heavy-tailed gradients？

2.Line 17: Which case do the authors want to state for the phrase “in this case”? 

3.Lines 47-49: What is the meaning of the statement “Adam can be seen as Clip-SGD with momentum and iteration-dependent clipping level)”?

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
2

---

## Human Reviewer 2

### Summary
This paper studies the convergence behavior of AdaGrad/Adam, considering heavy-tailed noise which is both significant in theoretical and empirical aspects. The authors prove AdaGrad/Adam failed in this case. To handle this issue, the authors study AdaGrad/Adam with clipping and derive a high probability convergence bound which owns a polylogarithmic dependence on the confidence level. Finally, they provide some experimental results.

### Strengths
This paper studies the convergence behavior of AdaGrad/Adam when the noise is heavy-tailed, both of which are quite important in the deep learning field. They find the convergence issue of both two algorithms, specifically the polynomial dependence of the confidence level inside the convergence bound. To solve this issue, they consider AdaGrad/Adam with clipping and show a convergence bound with polylogarithm dependence of the confidence level. Finally, some experimental results are provided, showing the superiority of adding clipping over the non-clipping versions.

### Weaknesses
I have the following major concerns.

**On negative results**

The main motivation in this paper comes from the potential failure of AdaGrad/Adam in the heavy-tailed noise case.  However, the main result to prove the failure, Theorem 1, is not convincing. First, the result shows a complexity of Adam/AdaGrad that has inverse-power dependence on $\delta$. However, this bound should require $\beta_2 = 1-1/T$ and $\\|x\_0-x^*\\| \ge \gamma L$ instead of arbitrary $\beta_2$ and $x\_0$. It's then questionable whether another setup of $\beta_2$ and $x\_0$ may achieve success. Second, I think it's not convincing to say Adam/AdaGrad is a failure given the inverse-power dependence on $\delta$ inside the convergence bound. Note that the dominated order in a convergence bound (or complexity) comes from the order of $T$ (or the accuracy $\epsilon$). I see that the complexity still achieves $\Omega(poly(\epsilon^{-1/2}))$ which leads to the convergence.

I suggest the author prove a negative result similar to [Remark 1,1], where they can show that for arbitrary step size and initialization, SGD has a non-convergence issue on a specific problem.

**On results regarding clipping**

First, the author claims that the main goal of incorporating the clipping is to improve the dependence of $\delta$ to polylogarithm order. However, I do not see clearly any polylogarithm order of $\delta$ in Theorem 2, 3, and 4, particularly in the complexity formulas. Second, I do not see the motivation for using a delayed step size. If we have the AdaGrad/Adam with clipping, why do we still need the delayed step-size version? Finally, the polylogarithm order of $\delta$ for AdaGrad with clipping has already been obtained in [2], although with a slightly stronger assumption. I suggest the author claim more on the proof difference with their results.

**Reference**

[1]. Zhang J, Karimireddy S P, Veit A, et al. Why are adaptive methods good for attention models? Advances in Neural Information Processing Systems, 2020, 33: 15383-15393.

[2]. Li S, Liu Y. High Probability Analysis for Non-Convex Stochastic Optimization with Clipping. ECAI 2023. IOS Press, 2023: 1406-1413.

### Questions
Please refer to **Weaknesses**.

### Soundness
2

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
This paper examines the high-probability convergence of adaptive optimizers like AdaGrad and Adam under heavy-tailed noise. Without gradient clipping, these methods can struggle with convergence. The authors show that gradient clipping significantly improves convergence bounds and empirical performance for AdaGrad and Adam, making them more robust to heavy-tailed noise.

### Strengths
The negative example for Adagrad's (actually, Adagrad-Norm) convergence is interesting. This could imply that heavy-tail noise is not handled by "adaptive" methods, which clarifies a misconception in the area. If this point is fully justified (given that the concerns I have below are resolved), I think it is a interesting contribution.

### Weaknesses
1. **Not analyzing Adam, but perhaps a twin of Adagrad**:  This paper does not analyze the original Adam, but Adam with beta2 = 1-1/K. The paper wrote "Therefore, the standard choice of beta2, in theory is, = 1 - 1/K where K is the total number of steps", but this is not the standard choice in theory. For instance, there are recent results proving convergece of Adam for constant beta2 (Zhang et al.'2022, cited in the submitted work). The analyzed algorithm with 1-1/K might be essentially Adagrad (note that for beta2 = 1/1/k, the algorithm becomes Adagrad, but for beta2 = 1 -1/K, it requires more discussion). The convergence properties of Adam and Aadgrad are quite different. 

2. **Analyzed scalar-coefficient clipped version**, instead of regular clipping: The clipped version Algorithm 2 uses a scalar b_t, instead of a vector b_t in the typical adaptive gradient methods. This is because the update of b_t uses the norm of the clipped gradient instead of the gradient vector. This makes the algorithm quite different from the original adaptive gradient methods. 
       The authors renamed the algortithm from '"Adagrad-Norm" to "Adagrad", and uses Adagrad-CW to describe the orginal version, as mentioned in a footnote. But this naming is quite misleading. If the paper analyzed Adagrad-norm, then the title and abstract should reflect it. 
        Another example that renaming Adagrad-norm by Adam is misleading: for the experiments, I cannot tell for sure whether the authors use the original Adam or Adgrad-norm. My guess is the authors used Adagrad-norm for experiments, since the term "Adam" is already renamed. 

3. **Contribution.**  Given the above modifications, the paper actually shows that Adagrad-norm-with-clipping works well while Adagrad-norm-without-clipping works not so well, for the heavy-tail-noise case. Thus the result is not about the orginal Adam. Nevertheless, there is still some chance that such an analysis could shed some light on the relation of clipping and Adam, if the experiments on Adam exhibit similar behavior to Adagrad-norm. However, the experiments are on "Adam", which, I guess, actually means Adagrad-norm in the context of this paper, thus the experiments may not be relevant to practitioners.

### Questions
In the experiments, does "Adam" mean the version of this paper, or the common version in the literature (i.e. the original version by Kingma and Ba)?

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

---

## Human Reviewer 4

### Summary
The authors provide examples to show that the high-probability complexities of Adam/AdaGrad (with momentum) and their delayed versions don’t exist poly logarithmic dependence on the confidence level generally when the gradient noise is heavy-tailed. The authors show that the high-probability complexities of Clip-Adam/AdaGrad and their delayed versions have polylogarithmic dependence on the confidence level under smooth convex and smooth nonconvex assumptions. The authors conducted numerical experiments for synthetic and real-world problems.

### Strengths
1. The authors provide high probability convergence complexity instead of the conventional in-expectation convergence complexity that most previous literature has focused on and such high probability convergence bounds can more accurately reflect the methods’ behavior than in-expectation ones.

2. The author emphasizes the importance of gradient clipping for adaptive algorithms (Adam \ AdaGrad) to deal with heavy- tailed noise through strict high probability convergence complexity analysis.

### Weaknesses
1. The author's statement on the probability convergence results corresponding to different methods is not clear enough, even though these results are similar.
2. The main theoretical results of this paper are based on the assumption of local smoothness of the optimization objective, even in convex cases, which is too strong.

### Questions
1. In the introduction section, you cited some viewpoints from previous literatures to illustrate that Adam and Clip-SGD have similar clipping effects for stochastic gradients. so, "it is natural to conjecture that clipping is not needed in Adam/AdaGrad" . Your theorem 1 emphasizes that Adam/AdaGrad without clipping do not have a high probability convergence complexity with polylogarithmic dependence on \delta even when the variance is bounded, rather than the divergence of Adam when the noise is heavy-tailed?

2. In the discussion section of Theorem 1, you stated that “We also conjecture that for \alpha<2 one can show even worse dependence on ε and δ forAdam/AdaGrad…”. Have similar conjectures been mentioned in previous literatures, or can an informal analysis be provided?

### Soundness
3

### Presentation
2

### Contribution
3

### Rating
5

### Confidence
2