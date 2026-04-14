# A Skewness-Based Criterion for Addressing Heteroscedastic Noise in Causal Discovery

- Decision: Accept (Poster)
- Scores: 6, 10, 5, 6, 6

## Abstract
Real-world data often violates the equal-variance assumption (homoscedasticity), making it essential to account for heteroscedastic noise in causal discovery. In this work, we explore heteroscedastic symmetric noise models (HSNMs), where the effect $Y$ is modeled as $Y = f(X) + \sigma(X)N$, with $X$ as the cause and $N$ as independent noise following a symmetric distribution. We introduce a novel criterion for identifying HSNMs based on the skewness of the score (i.e., the gradient of the log density) of the data distribution. This criterion establishes a computationally tractable measurement that is zero in the causal direction but nonzero in the anticausal direction, enabling the causal direction discovery. We extend this skewness-based criterion to the multivariate setting and propose \texttt{SkewScore}, an algorithm that handles heteroscedastic noise without requiring the extraction of exogenous noise. We also conduct a case study on the robustness of \texttt{SkewScore} in a bivariate model with a latent confounder, providing theoretical insights into its performance. Empirical studies further validate the effectiveness of the proposed method.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
In this paper, the author proposed a novel causal learning algorithm for heteroscedastic symmetric noise models (HSNMs), where the noise term of a variable $X_i$ can be parameterized by $\sigma(pa(X_i))N_{X_i}$ with symmetric noise $N_{X_i}$. Specifically, the author showed that the skewness of the score function (SkewScore) in the $X_i$-coordinate is zero if and only if $X_i$ is a leaf node under certain assumptions. Therefore, the topological order can be identified by iteratively removing the variables with the least SkewScore. Conditional independence tests are then applied to determine the edges. Its validity for the HSNM with latent confounder is also discussed. Extensive experiments showed that the proposed method is effective on HSNMs and is robust to assumption violations.

### Strengths
1. The method is novel and can be potentially applied to causal discovery with latent confounders.
2. The theoretical justification of Assumption 1 seems sound, and the experiments showed that the proposed method works particularly well on sparse HSNMs.
3. The author extensively evaluated the method when the assumptions are violated, e.g. non-symmetric noise distribution, and latent confounder in the multivariate setting, and proved its effectiveness and robustness.

### Weaknesses
1. The author mainly used ER1 to evaluate the methods on HSNMs without latent confounders, which is relatively sparse.
2. I think the justification of Equation (6) was a bit missing. Proposition 5 justified that Assumption 1 is mild, and therefore Equation (5) in the bivariate setting holds. However, there is no such explanation for Equation (6). Can you provide some concrete examples where Equation (6) is reduced to some known results, or simplified by specific function and noise classes?

### Questions
1. Which ER graph model is used in the experiments, $G(d, p)$ or $G(d, m)$? 
2. Can you explain in more detail why $c\neq0$ in general in Remark 6?
3. What is the runtime for the order learning and independent tests separately?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
10

### Rating Number
10

### Confidence
4

### Summary
==Post-rebuttal update==

The authors' replies to my questions show their further insights into the problem and how the work could be deepened.

I also had a rough scan of other reviews and their respective replies, and I hold my original score.

This is an exceptionally well-written paper with valuable insights and high potential impact. Of course, no papers are perfect, but the current rating best describes my evaluation of the paper, i.e., it "should be highlighted at the conference".

==End update==

This paper proposes a method to identify causal directions in heteroscedastic symmetric noise models (HSNMs) using the skewness of the score function, which is the gradient of the log density. The skewness considered here is analogous to standard skewness but is distinct, offering computational advantages. Importantly, the paper proves that the score's skewness is zero only in the causal direction. This idea is later generalized to the multivariate case. The proposed method also shows robustness in the presence of hidden confounders, and a theoretical analysis is provided for an "additive" confounder in the bivariate case. The method demonstrates strong empirical performance.

### Strengths
The paper provides a neat and effective solution to a critical problem. The connection between score skewness and causal direction is both theoretically grounded and practically effective.

The extensions to multivariate cases and the potential applicability under hidden confounding are valuable contributions.

The theoretical analysis appears rigorous (though I haven’t verified every detail).

The paper is clearly written, with helpful elements like the two properties of the score function, the motivation of score skewness by analogy to standard skewness, and well-designed figures.

### Weaknesses
The generality of Assumption 1, which defines the applicable HSNMs, could be explored further, although the paper presents sufficient results for a conference submission.

The method's applicability under hidden confounding might be limited to special confounding structures.

### Questions
### On Proposition 5
Could you elaborate on why, “when $\sigma(x)$ is not constant (i.e., in the heteroscedastic model), $c \neq 0$ in most cases”? More broadly, could you comment on the condition $c=0$? This characterization could be crucial as it defines a class of distributions $p(u, x)$.

Proposition 5 considers $f$ in a finite-dimensional linear space, specifically polynomials. Could neural networks with hidden layers be considered also (since Example 7 doesn’t)?

It’s interesting that $f$ is required to lie in a “large finite-dimensional space.” I would assume that if $f$ were in an infinite-dimensional functional space, the space should instead be “small.” Could you share your thoughts on this apparent contrast?

### Additional Comments
Could similar theoretical results be proven for standard skewness?

Assumption 4: Typically, “regularity” refers to the properties of a well-behaved functional space. In Assumption 4, condition 1) is indeed a regularity condition, but it’s unclear if condition 2) qualifies as such. Additionally, could you explain why condition 2) is considered mild?

Minor: There is a duplicate bibliography entry for Mityagin (2015).

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
2

### Summary
The paper presents a novel approach to uncovering causal relationships in data with heteroscedastic noise, which challenges traditional causal discovery models. The authors propose the use of a skewness-based criterion derived from the score of data distributions to identify causal directions, establishing that this measurement is zero in the causal direction but non-zero in the anti-causal direction.

### Strengths
1. The introduction of the SkewScore, which is leveraged to identify causal structures without requiring the extraction of exogenous noise.
2. A theoretical extension of the criterion to multivariate cases, along with empirical validations showing its robustness, particularly in scenarios involving latent confounders.

### Weaknesses
See questions.

### Questions
1. How is Theorem 1 applied to establish the causal direction between variables $X$ and $Y$ in the experiments? Equations (4) and (5) represent the population mean, while in experiments, we can only obtain the sample mean.
2. The performance of SkewScore in terms of accuracy, as shown in Figure 3, is not consistently superior to other methods. Specifically, SkewScore outperforms others in the case of GP-sig Student's t in Figure 3(a), and in the cases of GP-sig Gaussian and GP-sig Student's t in Figure 3(b). What are the computation times associated with the different methods?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work proposes a causal discovery method for a class of restricted structural causal models with heteroscedastic symmetric noise. The identification of the causal graph is established using a score that quantifies the skewness of the score function. A causal discovery algorithm is then proposed based on the skewness score.

### Strengths
The notation is well-organized and the paper is easy to follow overall.

### Weaknesses
1. Fact 2 is incorrect. Let X follow univariate exponential distribution with parameter $\lambda$, then $E[\frac{d \text{log} p(X)} { dx}] = -\lambda$. Its proof is also incorrect. 

2. The proof of Theorem 1 heavily relies on the assumption of symmetric noise. Without this symmetry, skewness no longer provides useful implications for the causal direction, as the proof of Theorem 1 breaks down. 

3. Definition 1 needs detailed discussions.  As stated, "this measure captures the asymmetry...," it would be helpful to demonstrate that it is zero when the distribution is symmetric.

### Questions
Is it possible to provide identification results such as 'the skew score is larger under the wrong causal direction' without symmetric noise assumption?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper provides a novel causal discovery algorithm that leverages skewness to determine causal direction when heteroscedastic noise (HN) is present.

### Strengths
1. An example in Section 5 provides a clear, practical illustration of the algorithm, connecting theoretical concepts with application. 

2. This paper provides a well-structured and self-contained summary of prior works in causal discovery algorithms based on functional causal models, allowing readers to clearly follow the evolution for learning causal graphs. As a reviewer, this summary enables me to track the advancements in the field and understand the recent progress. 

3. The core idea behind the proposed method is simple and intuitive, based on the asymmetry of score skewness to identify causal directions. This approach is conceptually accessible and computationally efficient solutions. 

4. The experiments provide strong empirical evidences of the paper’s contributions, with a wide range of scenarios that demonstrate the robustness of the proposed framework in heteroscedastic settings.

### Weaknesses
1. The paper provides correctness guarantees only for the two-variable case, where skewness in the score function can reliably distinguish causal directions. However, for settings with more than two variables, there is no formal guarantee of correctness for Algorithm 1. 

2. Current empirical results seem insufficient in showcasing its performance in high-dimensional scenarios. In other words, it remains unclear whether SkewScore scales effectively as the number of variables grows significantly. 

3. While the authors claim robustness in the presence of latent confounders, this robustness appears to be limited to simple bivariate cases. In multivariate settings, SkewScore can still produce incorrect causal graphs when latent confounders are present. For instance, in the causal structure $X_1 \rightarrow X_2 \rightarrow X_3$ with a latent confounder $L$ influencing both $X_1$ and $X_3$ ($X_1 \leftarrow L \rightarrow X_3$), the method would infer the structure $X_1 \rightarrow X_2 \rightarrow X_3$ along with a spurious direct link $X_1 \rightarrow X_3$. This incorrect edge arises because the latent confounder $L$ induces a conditional dependence between $X_1$ and $X_3$, which SkewScore interprets as a direct causal link. As a result, while the method can handle some latent confounding, it does not offer sufficient mechanisms to detect and mitigate confounding in more complex structures, limiting its application in practical scenarios where latent variables are common.

### Questions
1. Is the algorithm scalable?  It would be valuable to discuss if this approach can handle high-dimensional datasets effectively and whether its complexity scales polynomially or exponentially with the number of variables. 

2. Can you discuss the relation between the proposed method with other types of causal discovery algorithms, such as (1) constraints-based algorithms such as PC algorithm etc., and (2) continuous optimization based algorithms such as the one in https://arxiv.org/abs/1803.01422, which provides scalability? 

3. What are the effects of small sample sizes on SkewScore’s performance? It would be great if authors can discuss the convergence guarantee of the algorithm in terms of the size of samples. 

4. Can the proposed algorithm provide confidence intervals or uncertainty quantification in measuring the skewness or determining causal directionality?

### Soundness
4

### Presentation
3

### Contribution
3
