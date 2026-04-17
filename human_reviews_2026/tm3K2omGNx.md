# Private Top-$k$ Selection under Gumbel Differential Privacy Guarantees

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 2, 0

## Abstract
From the perspective of hypothesis testing, $f$-differential privacy ($f$-DP) as a relaxation of differential privacy (DP) possesses numerous desirable properties, the most prominent of which is its lossless characterization of the composition of DP mechanisms.  Within the $f$-DP class, Gaussian differential privacy (GDP), as a canonical family introduced to design Gaussian mechanism, has  gained widespread acceptance. However, Gaussian mechanism is not the optimal option for all scenarios to ensure DP. As a type of extreme value distribution, Gumbel distribution is naturally considered to design private top-$k$ selection algorithms. In this work, a new family in $f$-DPs, named Gumbel differential privacy (GumDP), is developed to parameterize Gumbel mechanism as similar to GDP. And the composition of Gumbel mechanisms is studied. In addition, two important composition properties of the Gumbel mechanism are discovered among different private selection problems. Utilizing these, a novel privacy-preserving top-$k$ selection algorithm with Gumbel mechanism, called the peeling algorithm under oneshot RNM, is presented based on the Report Noisy Min (RNM) and peeling algorithms. Simulations demonstrate that the privacy-utility performance of the proposed private selection algorithm is significantly improved compared to the peeling algorithm under RNM with Laplace or Gaussian mechanism.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work considers joint release of max and argmax values of a set of functions evaluations on some dataset in a differentially private manner, that is, releasing both the index and the value of the function achieving the minimal (maximal) evaluation, and its extension to top-k functions. For a single index, this is typically achieved by first identifying the maximal index in a differentially private manner using the report-noisy-max algorithm, followed by a noisy evaluation of the query in that index. Extension to top-k is achieved using the peeling algorithm which sequentially removes the maximal index.

One possible implementation of the report noisy max algorithm is by adding Gumbel noise to each functions evaluation, and reporting the index of the noisy max. This work considers an alternative algorithm which rather than releasing a separate evaluation of the function in the reported index, simply uses the original noisy evaluation. To analyze this option the authors utilize the f-DP framework by computing the explicit tradeoff function of the Gumbel mechanism.

### Strengths
The work is well motivated, clearly structured, and its results seem plausible.

### Weaknesses
Unfortunately, it seems like there are many incorrect / inaccurate details in the write-up, which - while all might be fixable, make it nearly impossible to evaluate in its current form. The fact this work does not even contain proof outlines for most claims, makes it even harder to evaluate.

1. The CDF of the Gumbel distribution is $e^{-e^{(x-\mu)/\gamma}}$, not $1-e^{-e^{(x-\mu)/\gamma}}$ as stated in the opening of Section 2.1. This is not a simple typo, but seems to affect the analysis of some of the proofs (e.g., Lemmas 1 and 2). It might very well be the case that switching CDF and CCDF in the appropriate places will solve the issue, but it its current form it makes it hard to follow some of the proofs, e.g., Lemma 2 discussed below.
2. If I understand correctly, Lemma 1 is incorrect in its current for, and requires an additional assumption to hold. It's proof relies of the claim (last line of the proof) that $\vert g(\mathcal{D}) - g(\mathcal{D}') \vert \le \Delta$ but in the general case it is in fact $\Delta + \frac{\Delta}{\epsilon}\ln(m)$. The $\ln(m)$ term can be ignored only under the additional assumption that changing a single element can change the evaluation of only a single function.While this assumption indeed holds for some functions such as histograms, it does not hold in general.
3. Lemma 2 is started in a way that is very hard to follow. Assuming I understood correctly, and the claim is that releasing jointly the maximal index and value of the Gumbel mechanism is distributed identically like independent evaluation of the maximal index and value, I fail to understand the proof, specifically the second equality. This might also have to do with the CDF/CCDF issue mentioned before, but regardless at least one of the two terms must take the form of $p(t)$ and the other $1-p(t)$, which I can't find.

Besides these clarity issues, I further fail to understand how the two parts of this work are related to each other. Even if Lemma 2 is correct, it essentially allows reducing the analysis of $M^{*}_{Gum}$ to the composition of the Gumbel noise addition mechanism and the exponential mechanism, which can be done using basic composition. the privacy guarantees of the Gumbel noise addition mechanism can be analyze directly via DP definition.

Furthermore, the It is not clear if the Gumbel mechanism is even a useful tool in most cases. Using corollary 1 it is clear that for somewhat large $\mu$ (say $\approx5$), $\delta(\varepsilon)$ approaches $1$.

Indeed, Figure 3 shows that the advantage of the Gumbel mechanism relative to the Gaussian mechanism is negligible except for extremely small values of $\varepsilon$.

### Questions
Please address the questions above.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces a new top-k selection algorithm based on the Gumbel mechanism under $f$-differential privacy ($f$-DP). The authors proposes a novel notion of privacy termed  $\mu$-Gumbel DP, which is a instantiation of $f$-DP obtained by specifying an appropriate trade-off function. Properties of $\mu$-Gumbel DP in the context of private selection problem are also presented. The authors also demonstrate better privacy-utility trade-off compared with existing methods via simulation experiments.

### Strengths
The paper offers a new perfective on privately solving the top-k selection problem, which is an important and  classic problem in differential privacy. I also find the composition properties presented in the paper interesting.

### Weaknesses
1. The applicability of the composition result shown in Theorem 2 is limited. Theorem 2 holds when all statistics $h_i$ are consistent, which is a rather strong assumption, especially for large $k$. Moreover, does the definition of consistency in Definition 9 need to hold for any neighboring dataset $D$ and $D'$? If that is the case, the practical use of Theorem 2 will be quite limited from my perspective. In addition, it would be helpful to clarify whether the consistency condition must be verified prior to applying the Gumbel mechanism within an algorithm.
2. Lemma 1 looks like a standard post-processing result to me. I am curious if Gumbel DP also satisfies post-processing property. If yes, is it a new property specific to Gumbel DP or property inherited from the general f-DP?
3. The utility metric shown in 3.2 is not entirely convincing. The authors use noise variance to analyze the privacy-utility performance. However, the private-k selection problem has a more common and meaningful metric that is the difference between the output statistics and the true minimum value. The absence of analytical results and comparison suing this metric significant weaken the contribution of this work.

### Questions
All questions are included in "Weaknesses" section.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper considers a noise additive DP mechanism using the Gumbel distributed noise (called Gumbel mechanism in the paper). This is well known in case of top-k selection, as the arg max of n additive Gumbel noise mechanisms with limited sensitivity deterministic parts is equivalent to the exponential mechanism which is already well convered in the literature (see, e.g., [Dong et al., 2022](http://proceedings.mlr.press/v119/dong20a/dong20a.pdf)). The paper gives analytical expressions for the trade-off functions of the Gumbel mechanisms and their compositions.

### Strengths
- Well written, accurate mathematical description of the trade-off functions of the Gumbel noise-additive DP mechanisms and of their compositions.

### Weaknesses
- The contribution remains quite thin in my opinion: it is just the analytical characterization of the trade-off functions of the Gumbel noise-additive mechanisms. The second part, i.e., the top-k selection with Gumbel noise is already well known as it is the exponential mechanism. 
- No real contribution in the second part (top-k part): the paper states existing results like that of [Dong et al., 2022](http://proceedings.mlr.press/v119/dong20a/dong20a.pdf)
- There is no utility analysis of the Gumbel mechanism. It remains unclear why would one use it.

### Questions
The top-k selection is well known and already covered by the existing literature as it is the exponential mechanism. What is really the benefit of using the noise-additive Gumbel mechanism?

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
This paper uses Gumbel instead of the Gaussian mechanism to achieve DP by designing a private top-k selection algorithm. The peeling algorithm is considered for analysis where the queries are perturbed by Gumbel mechanism. The analysis is done with f-DP as it has lossless composition property rather than $(\epsilon,\delta)$-DP. Further, conversion between $(\epsilon,\delta)$- DP and f-DP for the Gumbel mechanism is derived. The authors build on the result of [1], which shows that the exponential mechanism is equivalent to the RNM with Gumbel noise

### Strengths
Please see the detailed review under questions textbox

### Weaknesses
Please see the detailed review under questions textbox

### Questions
The paper seems to have a major technical problem namely the fact that it requires the consistency assumption. Note that the consistency assumption (line 293) state that the k query function are consistent if $sign(h_1 (D' ) − h_1 (D)) = · · · = sign(h_k (D' ) − h_k (D))$ where $sign()$ is the sign function. However this would restrict the possible datasets $D'$ itself and won't hold for any arbitrary neighbouring datasets which will violate the very definition of DP. When any datapoint is replaced, even when the ordering doesn't change, the signs will be different when the replaced point happens to be in bottom k: there will be k-1 zeros in the set of sign of differences and one value will be +1 or -1. And when the ordering changes, this set of signs will be even more diverse. 
Elaborating with an example, if we consider $k=4$, and neighbouring datasets $D=\{7,2,15,6,1,4\}$ and $D’=\{7,2,15,3,1,4\}$, we have $h_1(D)=1$, $h_2(D)=2$, $h_3(D)=4$, $h_4(D)=6$, and $h_1(D’)=1$, $h_2(D’)=2$, $h_3(D’)=3$, $h_4(D’)=4$. Here, $sign(h_1(D)-h_1(D’))=0$, $sign(h_2(D)-h_2(D’))=0$, $sign(h_3(D)-h_3(D’))=1$, $sign(h_4(D)-h_4(D’))=1$.
Even in this simple example, the consistency assumption doesn’t hold. Similarly for any given dataset, we can find an arbitrary replacement of a datapoint and construct the neighbouring dataset for which consistency doesn’t hold. 

Other comments

2) It will be better to state the probability distribution of Gumbel used as Gumbel (minimum) as it is referred to in literature to avoid confusion

3) In Corollary 1 (Page 5), $\epsilon<=0$ is given as the condition in $(\epsilon,\delta)$-DP in order to satisfy $\mu$-GumDP. Isn't $\epsilon$ supposed to be positive according to the definition of $(\epsilon,\delta)$-DP? It appears to be a typographical error.

4) Similarly, in Corollary 2 (Page 6), is $\delta_k(\epsilon)$ guaranteed to take positive values? If not, the domain of operation has to be clearly mentioned.

5) In Appendix A, Step 6, the convexification step that produces the three-segment trade-off $B_{\mu}(\alpha)$ is not clearly explained. Can you clarify how the breakpoints $\alpha_1$ and $\alpha_2$ and the middle section arise from the bi-conjugate construction, i.e., provide a brief derivation or reference for this step (equation 3)?
6) In Appendix D (Page 14), equations (7) and (8) have "n" as the total number of terms in the summation. However, after the transformation of variables as $y_i=\exp(x_i)$, in the subsequent 2 equations, the number of summation terms is written as "k", which should be "n". Please check the consistency of variables being used. Also, will this change the final expressions derived?

### Soundness
1

### Presentation
2

### Contribution
1
