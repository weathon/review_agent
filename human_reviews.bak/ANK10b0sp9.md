# Generalization error bounds for iterative learning algorithms with bounded updates

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 6, 3, 3

## Abstract
This paper explores the generalization characteristics of iterative learning algorithms with bounded updates for non-convex loss functions, employing information-theoretic techniques. Our key contribution is a novel bound for the generalization error of these algorithms with bounded updates. Our approach introduces two main novelties: 1) we reformulate the mutual information as the uncertainty of updates, providing a new perspective, and 2) instead of using the chaining rule of mutual information, we employ a variance decomposition technique to decompose information across iterations, allowing for a simpler surrogate process. We analyze our generalization bound under various settings and demonstrate improved bounds. To bridge the gap between theory and practice, we also examine the previously observed scaling behavior in large language models. Ultimately, our work takes a further step for developing practical generalization theories.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The authors considered the generalization error bound for iterative learning algorithms. It is argued instead of utilizing the chain rule of the mutual information, the generalization bound can be obtained using a variance-based quantity. When the updates are bounded, this will simplify and give a new expression only related to the upper bound on the update and the learning rate. 

The contribution appears to be replacing several quantities, either information theoretic ones or Lipschitz coefficients, by an upper bound on the update norm, therefore providing a new expression for generalization error upper bound.

### Strengths
The strength appears to be the recognition that the bounded updates can be used to replace some relevant quantities in existing bounds. The presentation is in general clear.

### Weaknesses
1. The contribution is minimal. Some results trivially follow those of already known results. A few example cases are below:
 (a) The first theorem 4.4 in fact trivially follows from Xu&Rakinsky 2017, by noticing the expression is basically I(U^{(T)}; S_n|W_0)=I(W_T; S_n|W_0), which is actually greater or equal to I(W_T;S_n), i.e., the bound is Theorem 4.4 is looser. 
 (b) The bound Theorem 4.10 is a very loose bound, essentially assuming the worst-case correlation among updates. This result can easily follow from existing ones using the chain rule of mutual information, by also making the worst-case correlation assumption. 

2. There appear to be technical errors/misunderstandings on some concepts. One particularly troublesome one is around Theorem 4.5. Since h(U^{(T)}|W_0,S_n) is a differential entropy, instead of entropy (i.e., continuous random variables instead of discrete random variables), there is little meaning in its absolute value or its sign, since a simple scaling of the continuous random variable can change the differential entropy value by any amount. The authors appear to misunderstand this difference, and Theorem 4.5 is in a sense meaningless in this form. 

3. The comparisons do not appear convincing. The authors made a different assumption and therefore obtained different bounds. It would appear unfair to argue that the derived bound is tighter than others, as claimed in Section 5.

### Questions
1. For the discussion given in Section 6, are the authors considering estimating the generalization error? The proposed bound appears extremely loose since the worst-case correlation among updates is assumed. I am also not very sure whether numerical experiments are conducted, or if this is just a generic discussion on the expected behavior of the bounds. If numerical experiments are performed, how are the authors able to give order characterization, not just numerical plots?

2. Have the authors considered the bounds given in 
"Tightening mutual information-based bounds on generalization error", Y Bu, S Zou, VV Veeravalli;
"Reasoning about generalization via conditional mutual information", T Steinke, L Zakynthinou. 
Those are more up-to-date information-theoretic approaches, and the relation is worth discussing.

### Soundness
2 fair

### Presentation
3 good

### Contribution
1 poor

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper upper bounds the expected generatization error via the variance of total updates, and the total variance can be decomposed into a sum of variance of each update and their covariances. When the updates are bounded, the expected generalization error of iterative algorithms is then upper bounded by the concentration factors and the learning rates. Surrogate process is considered by unlike the previous results by Neu et. al. that adds Gaussian noises at each update, a Gaussian noise is added at the end of the final updates in the proposed surrogate process. The rate of the proposed generalization error bound is compared to the previously studied testing error rate in LLM.

### Strengths
The technical novelty is moderate, compared to previous mutual-information-based generalization error bound, using variance in the proposed approach though not as tight, allows better decomposition than the chain rule, and thus allows a surrogate process that adding Gaussian noise at the end. 

The connection to LLM practice is meaningful, since the dimension of parameters in this case is relatively high, which hinders the usages of some previous bounds.

### Weaknesses
I have not major concerns on the technicalities and results. The control of \Delta_\sigma is an issue in analyzing the proposed generalization error bound, but inevitable for surrogate process.

### Questions
Why do we consider these choices of \Delta_\sigma in Table 1?
Why is it fair to say h(U^{(T)} | W_0, S_n) \geq 0 in Table 2?
How is \Delta_\sigma determined in Table 3? 

Minor:
Is \E[U^{(t)}] in page 3 and the rest conditioned on W_0?
In Theorem 4.10 and the rest, the last item should be \eta_i?
Eq (*) in Eq (8) should be \geq instead of =, since conditioned on W_T, W_0 and S_n are note independent?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper explores the generalization characteristics of iterative learning algorithms with bounded updates for non-convex loss functions, employing information-theoretic techniques.

### Strengths
This paper explores the generalization characteristics of iterative learning algorithms with bounded updates for non-convex loss functions, employing information-theoretic techniques.

### Weaknesses
* The comparison in 5.4 with other theoretical works seems to be comparing apples to oranges, and I'm unable to see how this paper improves or is better than other methods. The assumptions in different papers seem quite different and hard to compare/

* Section 6 on LLMs seems to be stretching things too much. It's unclear how well real data satisfy assumptions in the paper.

* There are synthetic experiments demonstrating how good the bounds are.

### Questions
How tight is the bound in the paper? Do we know a lower bound?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper proposes a simple variance decomposition technique to evaluate mutual information-based generalization error bounds for iterative learning algorithms with bounded updates. Compared to the previous paper by Neu et al. (2021), this paper uses a simpler surrogate process. Various settings are considered to demonstrate the improvements of the proposed bounds.

### Strengths
The information-theoretic framework for analyzing the generalization error bound is promising and insightful. The proposed technique is straightforward.

### Weaknesses
The major technical contribution of the paper is the variance decomposition technique in Theorem 4.5 and 4.7, which is based on the simple fact that Gaussian distribution achieves maximum entropy under variance constraint. To this end, the contribution is limited. There are multiple ways to further validate the usefulness of the proposed simple technique. Is it possible for the authors to combine their method with the following techniques to enlarge the scope of the paper?


1. The proposed bound is mainly compared with Neu et al. (2021), but a more recent ICLR 2022 paper has proposed a different way to construct the surrogate process, which results in a tighter bound. 

Wang, Ziqiao, and Yongyi Mao. "On the generalization of models trained with SGD: Information-theoretic bounds and implications." arXiv preprint arXiv:2110.03128 (2021).

2. The individual sample MI bound proposed by Bu et al. (2020), where a similar SGLD analysis shows that it is better than the original MI bound by Xu and Raginsky and Pensia et al.

Bu, Yuheng, Shaofeng Zou, and Venugopal V. Veeravalli. "Tightening mutual information-based bounds on generalization error." IEEE Journal on Selected Areas in Information Theory 1, no. 1 (2020): 121-130.


3. The conditional MI bound technique, which is further improved by Wang and Mao (2023)

Steinke, Thomas, and Lydia Zakynthinou. "Reasoning about generalization via conditional mutual information." In Conference on Learning Theory, pp. 3437-3452. PMLR, 2020.

Wang, Ziqiao, and Yongyi Mao. "Tighter Information-Theoretic Generalization Bounds from Supersamples." arXiv preprint arXiv:2302.02432 (2023).

### Questions
The connection to practice in Section 6 seems to be weak. I am not convinced that the rate of the bound matches the empirical estimated converge rate could validate the tightness of the bound. Such a comparison is too rough, and a more direct evaluation of the bound is needed.


Minor comments:
1.	I am pretty sure that in Xu and Raginsky’s original paper, they did not consider SGLD. The first information-theoretic analysis of iterative analysis was by Pensia et al.
2.	For the limitation raised by Haghifam et al. (2023), the author might check the following two recent papers for some recent updates
https://arxiv.org/abs/2310.20102
https://arxiv.org/pdf/2210.09864

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair
