# Navigating Sparsities in High-Dimensional Linear Contextual Bandits

- Decision: Reject
- Scores: 4, 2, 2, 6

## Abstract
High-dimensional linear contextual bandit problems remain a significant challenge due to the curse of dimensionality. Existing methods typically consider either the model parameters to be sparse or the eigenvalues of context covariance matrices to be (approximately) sparse, lacking general applicability due to the rigidity of conventional reward estimators. To overcome this limitation, a powerful pointwise estimator is introduced in this work that adaptively navigates both kinds of sparsity. Based on this pointwise estimator, a novel algorithm, termed HOPE, is proposed. Theoretical analyses demonstrate that HOPE not only achieves improved regret bounds in previously discussed homogeneous settings (i.e., considering only one type of sparsity), but also, for the first time, efficiently handles two new challenging heterogeneous settings (i.e., considering a mixture of two types of sparsity), highlighting its flexibility and generality. Experiments corroborate the superiority of HOPE over existing methods across various scenarios.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies high-dimensional linear contextual bandits, a challenging problem due to the curse of dimensionality.
Since the general high-dimensional case is unsolvable, existing research has introduced additional structural assumptions, typically the sparsity of either the model parameter or the covariance matrix of the context distribution.
Under the sparse-parameter assumption, the LASSO estimator is commonly used, while a recent work employs the RDL estimator for cases with sparse covariance matrices.
However, previous studies can handle only one of these cases at a time and cannot address hybrid/both settings.
To overcome this limitation, this paper adopts the PWE and proposes a policy called HOPE, an EtC-type policy.

### Strengths
* The proposed PWE-based policy can handle both cases: (i) a sparse model parameter and (ii) a sparse covariance matrix of the context distribution. In contrast, the LASSO estimator can be applied only to (i), and the RDL estimator only to (ii). Moreover, the proposed HOPE policy is applicable to both Scenarios 3 and 4.
* The paper provides a potentially tighter regret bound under Scenario 3 by incorporating the effective rank of the covariance matrix.

### Weaknesses
* Unlike previous EtC-type policies for high-dimensional linear contextual bandits—such as those by Hao et al. (2020) and Li et al. (2022), which adopt the LASSO estimator for case (i), and Komiyama & Imaizumi (2024), which adopts the RDL estimator for case (ii)—the proposed HOPE policy appears to require computing the LASSO estimator at every exploitation round with dimension $N+1$. Since the theoretical results (Propositions 1–4) involve the choice of $N$, even at the order of $T^{2/3}$, the proposed method incurs additional computational cost, which is expected to be substantial for large $T$ (as it computes LASSO estimator in $T^{2/3}+1$ dimension at least $T-T_0$ times), compared with existing EtC approaches that compute the estimator only once. Although numerical results are provided (with small $T$), the performance improvement appears marginal. Therefore, a more detailed comparison of computational costs would be valuable.
* It is difficult to compare the results with prior works, particularly for Scenario 2, where certain instance-dependent constants $a$ and $c$ are introduced without explanation. While the authors claim that the corresponding proposition yields an improved regret rate, this is not clear from the current presentation without clarification of the role of $c$.

### Questions
1. In Scenario 4, it appears that the policy knows in advance which arms belong to Part 1 or Part 2. While the authors claim that none of the previous works can handle Scenario 4, one might also consider a simple EtC-type approach that applies the LASSO estimator to Part 1 (where the dimension of $\theta$ is truncated to the size of Part 1) and the RDL estimator to Part 2. Since both estimators exhibit good concentration properties, such an EtC policy may achieve a similar regret bound under a simple worst-case analysis. Could the authors clarify the potential advantages of the proposed approach over this simple alternative, particularly in the context of Scenario 4?

2. Related to Weakness 1: Could the authors provide a detailed runtime comparison between the proposed HOPE policy and existing EtC-type policies?

3. Related to Weakness 2: While $\alpha,a,b$ and $c$ are instance-dependent constants, are there any known bounded relationships or ranges for these parameters? For instance, is $b\in (0,1)$ or are $a$ and $c$ related in a specific way?

4. Could the authors elaborate on Lines 849–850 (“to mitigate … where $i_0$ ...")? It seems that the authors replace one column with $\bar{\zeta}$ (based on the LASSO estimator) to further exploit sparsity in the model parameter. However, the intended meaning of these lines is unclear to me.

---

### Additional comment

Using the same symbol "C" for several universal constants can be confusing, as it obscures how individual lemmas relate to others in the overall argument. For instance, while the authors state that Lemma 4 can be proved by replacing part of the proof of Zhao et al. (2023) with Lemma 5, it remains unclear how the constants $c_l$ and $c_u$ influence $C$. Providing a clearer distinction between constants would improve readability and logical transparency.

---
### Minor comments

1. $M$ is undefined in Table 1: its definition first appears in Section 6.3.
2. typo in Line 855: "$\ne =$" should be corrected.
3. In definition 4, $M^{(i)}_{S_1(i)}$ appears without introduction. It seems to refer $M^{(i)}(S_1(i))$. It would be better to define the notation explicitly or ensure consistency.
4. type in Line 909: the subscript in $\theta$ is missing.

### Soundness
3

### Presentation
2

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
This paper combines the PWE estimator of Zhao et al. (2023) and the ETC algorithm to solve the high-dimensional linear contextual bandit problem with either a sparse parameter or approximately sparse eigenvalues of the covariance matrix of the context. Especially, the proposed method can also handle cases where both sparsities are present.

### Strengths
The paper proposes a novel and unified method of efficiently solving both sparse linear contextual bandits and linear contextual bandits with approximately sparse covariate matrix in eigenvalues.

### Weaknesses
1. The paper lacks motivation explaining why we need a better solution for both sparsities. For instance, I authors could have claimed that it is common for both sparsities to arise in practice, or while having both sparsities at once means that both LASSO and RDL can achieve sublinear regret, they are suboptimal in certain sense.  
To elaborate, while Scenario 3 (both sparsities) must be the new regime of problem instances that this algorithm tackles for the first time, I think there should be more discussion about how the proposed method improves upon using one of the previous methods such as LASSO and RDL. I see that there is improvement over LASSO-ETC when $M \ll \sqrt{s_ 0}$ in Proposition 3, however it is not clear whether that is commonly the case. Similarly, Scenario 4 (mixed sparsities) can be addressed by using both LASSO and RDL depending on the arms, and the advantage of using HOPE is not explained. Without this kind of discussion, it could be slightly misleading to claim that this paper studies the both-sparsity case for the first time, because any algorithm that exploits one of the sparsities would also achieve the same guarantees when both sparsities are present.

2. I don't think comparing the regret bounds with the existing methods is fair when Assumption 1 and additional assumptions in Appendix G are required. For Assumption 1, I am not convinced that it is a standard assumption as the authors claim. Could the authors clarify which works require these assumptions and which don't, among those mainly compared in the paper? I also think assumptions in Appendix G must be included in the main paper for clarity.

3. While the abstract and the introduction give an impression that HOPE automatically adapts to two different types of sparsity, it actually undergoes a different procedure for each case. I think clarification must be made.

4. It is hard to understand what the definition of the "approximately sparse eigenvalue of the covariance matrix" condition is. Is it Definition 5 in Appendix D.6? If so, it is not clear to me what relationship this definition has with the eigenvalues of the covariance matrix.

5. It is hard to understand the procedure of computing the PWE from the main text. What is the role of the "initial estimator" in line 203? It is not used again within Section 4. Some important steps are deferred to Appendix C, and Appendix C.2 is especially hard to follow. For the second information case, I don't understand what $\Gamma_ t^{(i)}$ should be. For the both-information case, it is hard to understand why replacing one of the column vectors is necessary, and what exactly happens when the replacement occurs.

6. The presented experimental results don't seem to add much significance. The difference between the algorithms is quite small, and the nearly linear increase of the cumulative regret suggests that learning may not have happened yet, meaning that all the algorithms require more samples to properly learn the true parameter. In addition, the fact that LinUCB is competitive with other methods suggests that the experiment setting might not be sparse enough. I understand that the main contribution lies in theoretical work and I think this is a minor weakness compared to the ones listed above.

### Questions
1. The proof of Lemma 5 is quite confusing. I can't see where Eqs. (18)-(20) come from and how they imply the following equations in lines 1227-1238. Could the authors clarify each step? In addition, the definition of $\theta_ {Q_ x}$ is misplaced in line 1188.

2. Is Definition 5 the condition the covariance matrix $\Sigma$ must satisfy? If so, for which scenarios? As mentioned in Weaknesses, the relationship between Eq. (13) in Definition 5 and the decaying of the eigenvalues is not clear.

3. In the main text, the noise is assumed to have variance $\sigma$. Is it not necessary for the noise to be $\sigma$-subgaussian?

4. In Appendix G.3, what is $k^* $ in Line 1450?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper aims to unify two distinct notions of sparsity: parameter sparsity in the model itself and contextual sparsity, where the covariance matrix displays sparsity in its eigenvalues. To address this, the authors introduce a new algorithm, HOPE, which they argue outperforms existing methods when only one sparsity type is present, while also serving as a versatile approach capable of handling both sparsity regimes effectively.

### Strengths
The overall setup and research direction are fairly intriguing. While most studies on model parameter sparsity rely on some form of regularity assumption—such as imposing a lower bound on the smallest eigenvalue of the covariance matrix, a compatibility constant, or certain conditions on the context distribution (e.g., compatibility or margin conditions)—this paper distinguishes itself by proposing an algorithm that remains effective even when the covariance matrix exhibits eigenvalue sparsity. More importantly, it claims to function well across both sparsity regimes. Notably, Assumption 1 does not include any constraint on the minimum eigenvalue. If the theoretical results indeed hold without such assumptions, the paper’s contribution would be quite significant.

### Weaknesses
I have strong doubts about the soundness of this paper, especially about eigenvalues and dimensional dependencies.

It seems that the authors have likely received similar criticism before, as evidenced by the lengthy paragraph in Remark 7. However, upon close inspection, I found that the paper still hides potentially dimension-dependent quantities behind some constants.

** Regarding Assumption 2 **

The authors state that the relevant quantity is bounded by a constant $c_1$, and they refer the reader to Appendix G for verification. My reasoning is as follows:

- In Proposition 12, the error (involving $\phi$) is of order $O!\left(\sqrt{\frac{s_0 \log(p)}{\kappa \cdot N}}\right)$. (For my own reference: $d(\cdot, \cdot)$ denotes the prediction error, as defined in Definition 1, Line 883.) If this expression seems suspicious, see Section 2.4 of Bühlmann & Van de Geer (2011), Eq. (2.8).
- In Assumption 5, the constant $C$ likely hides a restricted eigenvalue. The authors may have intended to refer to Section 2.5 rather than Section 2.4 of Bühlmann & Van de Geer (2011), specifically Eq. (2.13), where the minimum signal $a_n$ appears together with the compatibility constant–based error bound. (The cited work uses the $\ell_1$-norm version, while Proposition 12 uses the $\ell_2$-norm version—corresponding to restricted eigenvalue—but in both cases, these are eigenvalue-related quantities that are being treated as constants.)
- Next, let us see how these constants affect the results. According to Proposition 6 on the PWE estimator, its error is proportional to the Lasso estimator’s error $d(\cdot)$. Thus, $\kappa$ (or the compatibility constant) enters inversely into the PWE estimator’s error bound. Tracing through the regret analysis confirms that the regret is indeed $\kappa$-dependent.
- One might argue that this dependence is not an issue if the compatibility constant or restricted eigenvalue is constant. However, that is a special case—in many realistic settings, these quantities are dimension dependent. For example, let $x^{(i)}$ be a $p$-dimensional vector drawn uniformly from the unit sphere. Its restricted eigenvalue or compatibility constant is closely related to the minimum eigenvalue, which is of order $1/p$. Hence, the authors are effectively treating a quantity that could scale with $p$ as a constant, disguising this issue under Assumption 2. For Assumption 2 to hold, one would need an unusually well-conditioned setting, such as $X \sim N(0, I)$. The authors should explicitly clarify this point.

In summary, the claim made in Remark 7—that the results are free from restricted eigenvalue or compatibility constant assumptions—is false. The authors are concealing unfavorable dependencies behind constant notation. This alone should be grounds for rejection. This issue is particularly serious because the authors explicitly state in Related Works, Line 107:

>This work, instead, focuses on the high-dimensional setting… with the feature dimension $p$ at least on the same order as the budget $T$, i.e., $p \gtrsim T$.

In such a setting, $p$-dependence is critically important.

Furthermore, I now question whether Assumption 2 itself is even correct. Shouldn’t it involve $|\hat{\theta} - \theta|_{\Sigma}$? I could not find any such term, $|\hat{\theta} \Sigma \hat{\theta} - \theta \Sigma \theta|$-ish thing in Appendix G, so clarification is needed.


Although I have not examined the sparse covariance part as carefully, even from the sparsity component alone, it is evident that the authors have made serious mathematical misstatements and overstated claims.
Therefore, I firmly believe this paper should be rejected.

### Questions
Please check the weaknesses above. Here are some suggestions:

1) Minimum signal condition: The minimum signal condition is far from trivial. Assumption 5 is too crucial to be buried in the appendix; its presence determines whether the convergence rate is $\sqrt{T}$ or $T^{2/3}$, as seen in Hao et al. (2020) and Jang et al. (2022). This assumption should be explicitly presented in the main text. The same goes for Assumption 4.

2) Compatibility constant assumption: Same as above, it is not trivial. Please add it in your main body. 

3) In Assumption 3, what is $C_1$? While $c_1$ appeared earlier in Assumption 1, this is the first appearance of $C_1$. Is it another constant?

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
1

### Summary
This paper focuses on high-dimensional linear contextual bandit problems. Specifically, the authors consider either sparsity in the model parameters or sparsity in the eigenvalues of the context covariance matrices. To handle both forms of sparsity, a pointwise estimator is proposed. The authors then provide regret bounds for an algorithm based on this estimator in homogeneous and heterogeneous scenarios. Finally, the superior performance of the proposed method is demonstrated through experiments.

### Strengths
- The paper is well-structured and easy to follow. The motivation and contributions are clearly highlighted.
- It is surprising that sublinear regret bounds can be achieved across various scenarios.
- The effectiveness of the proposed method is demonstrated not only theoretically but also empirically.

### Weaknesses
I am not an expert in high-dimensional contextual bandits, so I am not sure whether the provided theoretical results are stronger than those in prior work. Specifically, I wonder whether the assumptions in Propositions 1–4 are the same as or weaker than those in previous studies.

Moreover, I am curious about the relationship between parameter sparsity and eigenvalue sparsity. If the assumption on parameter sparsity holds, does eigenvalue sparsity also follow? Is one of these assumptions stronger than the other?

### Questions
My main concerns and questions are outlined in the Weaknesses section. Additionally, I have the following question:

- In Algorithm 1, which line uses the initial estimator?

### Soundness
3

### Presentation
2

### Contribution
3
