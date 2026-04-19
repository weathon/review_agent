# Convergent Privacy Loss of Noisy-SGD without Convexity and Smoothness

- Decision: Accept (Poster)
- Scores: 8, 6, 5, 8

## Abstract
We study the Differential Privacy (DP) guarantee of hidden-state Noisy-SGD algorithms over a bounded domain. Standard privacy analysis for Noisy-SGD assumes all internal states are revealed, which leads to a divergent R\'enyi DP bound with respect to the number of iterations. Ye & Shokri (2022) and Altschuler & Talwar (2022) proved convergent bounds for smooth (strongly) convex losses, and raise open questions about whether these assumptions can be relaxed. We provide positive answers by proving convergent R\'enyi DP bound for non-convex non-smooth losses, where we show that requiring losses to have H\"older continuous gradient is sufficient. We also provide a strictly better privacy bound compared to state-of-the-art results for smooth strongly convex losses. Our analysis relies on the improvement of shifted divergence analysis in multiple aspects, including forward Wasserstein distance tracking, identifying the optimal shifts allocation, and the  H\"older reduction lemma. Our results further elucidate the benefit of hidden-state analysis for DP and its applicability.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper studies the problem of releasing the last iterate of noisy-SGD with Renyi DP guarantees. It presents:

- Bounds on the Renyi DP guarantee of order $\alpha$ for this problem that are independent of the number of iterations for non-convex, non-smooth functions that have a Holder continuous gradient. 
- Tighter bound for strongly convex, smooth loss functions.
- These bounds hold with the following batching strategies: Full batch, sampling without replacement, and shuffled cyclic.

### Strengths
**New Bounds and Relevance:** The paper studies a crucial problem: finding tight privacy bounds for solutions of non-convex optimization problems. Tight privacy accounting is essential for practical implementations of DP-SGD, particularly in sensitive domains like healthcare. When assumptions about a privacy guarantee are not met (e.g. convexity), the guarantee is inapplicable, forcing the use of  overly conservative bounds that can hinder accuracy. 

**Novel assumption:** The authors introduce an assumption that allows them to reuse previous techniques (shifted divergence) under a milder condition (holder continuous gradient).

**Clarity:** The techniques are clearly explained  for the full batch setting. While I did not check the math exactly, the proofs do follow logically. 

**Improved bounds:** The authors provide examples where their bound improves over prior work, even in the already studies (strongly) convex setting. 

**Broader impact:** The results are valuable for the broader ICLR community by enabling practical implementation of ML with sensitive data. The paper presents interesting techniques (Lemma 3.7) and assumptions (reduction of smoothness to Holder continuous gradient) in a clear and insightful way.

### Weaknesses
Below I state some potential weaknesses of the paper and point to specific questions in the next question that would allow me to understand better. 

1. The authors present a contrived loss function that meets the Holder assumption to demonstrate it is not a vacuous  relaxation from the convex to the non-convex setting. While this function meets the assumption, it questions the applicability of their results in real world settings. (Q1)

2. Theorems 3.1, 3.6, 3.11. Present bounds parameterized by optimal values of $\tau$ and a sequence $\beta$ making it hard to parse and consequently unintuitive. It becomes hard to understand practical implications and compare to prior work, and the claims about the burning period are hard to verify. (Q2)


3. The originality of Lemma 3.5 is questionable since a similar result was introduced before by Chien et al (2024b). While the authors acknowledge this prior work, they argue that it applies to a different context, machine unlearning. However, this is a misleading distinction because machine unlearning literature also tracks unlearning with a DP definition. (Q6)

4. Minor comments:
- For a general reader it is not clear from the paragraph in the introduction why Output perturbation is not a good option. (It becomes more clear in figure 1 but I would suggest having it in the main body).

- The first sentence presents noisy SGD and DP-SGD as equivalent, which is not true, in general. Noisy SGD typically refers to SGD where gradients are noisy, but there is no expectation about privacy guarantees. DP-SGD, should provide DP guarantees which might involve more steps than Noisy-SGD, e.g.,  clipping. 

- Line 96, K having diameter D would suggest the set is bounded but it does not matter for the bounds, D could be infinite and the bound still holds. I would clarify this since it makes the paper stronger :) 

- I suggest proofreading for typos. A few are:
    - Line 106: “differing with one point.” differing by?

    - Fig 2 caption, line 125: “It done by constructing a coupling (G_t, G_t’), resulting a coupled process….”

    - Theorem 3.11 statement “Then the DP-SGD update (1) under without replacement…”

    - Missing parenthesis in Eq. 19.

** Update **
See discussion below addressing these weaknesses.

### Questions
While I believe that the topic of the paper is significant, (1) the originality of ideas is not clear to me and (2) the applicability of the results it introduces could invalidate the progress I believe the paper makes in the area. The following questions are mostly to clarify this.



**Q1** Is there a practical setting where one would use the function exhibiting Holder continuity ? Or is there another function used in a practical setting that meets this assumption?

**Q2** How tractable are the problems in theorem statements 3.1, 3.6, 3.11? I assume it is easy since the plots for different settings are provided but I would suggest:
- Can the authors clarify how the bounds are calculated for the provided plots?
- Provide simplified versions or corollaries of these theorems that give more intuitive bounds? even if they are slightly looser. 
- Provide concrete examples comparing these bounds to prior work for specific parameter settings?

**Q3** From Figure 1 (b) one can assume there is a regime where the presented bounds are better than [AT22]. Is there a closed characterization for this regime? And can we compare with the convergence rate to see if this number of iterations is sufficient? This would prove the presented bounds are more practical than previous ones. 

**Q4** The $\hat{L}$-Lipschitz constant estimated in Figure 2 is necessary to calibrate the noise value in practical deployment of DP-SGD. After reading the methodology in the appendix it is still unclear to me how it is estimated and if this estimate is an upper bound on the real constant. Can the authors comment on how to guarantee that $\hat{L}$ is an upper bound and that this approach would not break privacy guarantees? 

**Q5** In Fig 2. a) what happens as $\lambda \to  0$? The paper only shows values for $\lambda>0.5$… 

**Q6** What are the differences between the introduced Lemma 3.5 and the lemma from Chien et al.? 

**Q7** I found the following related works that have related topics, can the authors comment on them and how they relate to their work? 
1. [https://arxiv.org/abs/2407.06496](https://arxiv.org/abs/2407.06496)
2. [https://arxiv.org/abs/2407.05237](https://arxiv.org/abs/2407.05237)

### Soundness
4

### Presentation
3

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
This paper investigates the convergent privacy loss of noisy SGD within a convex bounded domain. Applying standard composition results indicates that the final privacy loss after $T$ iterations can scale with $\sqrt{T}$. However, if only the last iterate is revealed, two recent works (Ye & Shokri, 2022; Altschuler & Talwar, 2022) demonstrate that running noisy SGD on smooth and (strongly) convex functions can result in convergent privacy loss after a burn-in phase. This paper enhances the shifted divergence analysis, presenting a strictly better privacy bound under the same assumptions, and also provides convergent privacy bounds for non-convex functions with Hölder continuous gradients.

### Strengths
This paper is well-written and well-structured, clearly discussing prior works and their methodologies. The topic of convergent privacy loss in noisy SGD is significant, and this work makes meaningful progress in addressing it. Additionally, the results concerning non-convex functions, which can outperform output perturbation, are particularly intriguing.

### Weaknesses
The novelty and improvements presented are somewhat limited. The Forward Wasserstein distance tracking lemma appears straightforward, and the primary enhancement in the shifted divergence analysis seems to lie in identifying a better allocation of shifts. Regarding the non-convex case, I am concerned that the improvements may not be substantial when compared to output perturbation in practical applications. Hence,  I would not be surprised if other reviewers lean towards rejecting the paper, given the concerns raised.

### Questions
Is there a typo in Equation (11)?
Do any classic neural networks exhibit Hölder continuous gradients? Why is this assumption appropriate?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper investigates the differential privacy (DP) guarantees of Noisy Stochastic Gradient Descent (Noisy-SGD) without relying on traditional assumptions of convexity or smoothness in the loss function, thereby addressing significant limitations in prior research. Traditional DP analysis for Noisy-SGD often assumes convexity and smoothness, restricting its applicability to various real-world machine learning problems. The authors demonstrate that it is possible to achieve convergent privacy loss bounds even when these assumptions are relaxed, requiring only that the loss function has a Hölder continuous gradient. This generalization broadens the applicability of DP-SGD to a wider range of non-smooth, non-convex problems.

#### Main Contributions:
1. **Generalized Privacy Bounds**: Establishes a new privacy bound for Noisy-SGD that applies to non-convex, non-smooth losses under the less restrictive condition of Hölder continuity.
2. **Improved Privacy Bounds for Smooth Convex Losses**: Refines analysis techniques involving forward Wasserstein distance tracking and optimal shift allocation, resulting in improved privacy bounds for cases with smooth, strongly convex losses.

By advancing hidden-state analysis, this work deepens the understanding of DP in machine learning models and introduces a more flexible framework for balancing privacy and utility in Noisy-SGD, making it more suitable for deep learning applications.

### Strengths
The strengths of this paper lie in its innovative approach to privacy loss analysis in Noisy-SGD. Unlike prior work that requires assumptions of smoothness and convexity, this paper successfully establishes privacy bounds under more general conditions, making it applicable to a broader range of non-smooth, non-convex problems. Key strengths include:

1. **Generalization with Hölder Continuity**: The paper introduces the concept of Hölder continuous gradients, allowing the privacy analysis to hold even without the smoothness assumption. This broadens the applicability of Noisy-SGD to various challenging settings.

2. **Tighter Privacy Bounds**: For cases involving smooth and strongly convex losses, the paper presents a stricter privacy bound than prior results, offering enhanced privacy-utility trade-offs.

3. **Refined Analytical Techniques**: By optimizing privacy loss bounds through careful shift allocation and forward Wasserstein distance tracking, the analysis is robust and offers concrete improvements over existing methods.

Overall, this work contributes a significant advancement in privacy-preserving machine learning by providing a more flexible and precise analysis framework that can benefit a wide range of applications.

### Weaknesses
1. **Unclear Structure**: The structure of the paper lacks clarity. The main text includes numerous proofs, while some theorem statements are placed in the appendix, which disrupts the flow and clarity of the overall structure.

2. **Figure Quality**: Figure 2(a) appears somewhat rough and could benefit from refinement to improve visual quality.

3. **Typo in Lemma 2.3**: There is a typo in Lemma 2.3, where the function notation shifts from \( f \) to \( h \), which should be consistent.

4. **Overly Complex Parameters**: The large number of parameters results in overly complex expressions, making it difficult to clearly interpret and compare the results.

5. **Lack of Comparative Analysis**: The paper lacks a comparison with prior results, as well as with the non-private version of the results, which would provide a better context and understanding of its contributions.

### Questions
1. **Comparison with Non-Private and Smooth/Convex Cases**: Could you provide more detail on any existing comparison with non-private results? Additionally, how do these results compare specifically to the smooth and convex cases?

2. **Simplified Upper Bound for Summations**: Is it possible to present a more concise upper bound for the summations in the final results to enhance readability and ease of comparison?

3. **Challenges in Removing Smoothness and Convexity**: While the paper does extend to non-smooth and non-convex scenarios based on the condition of Hölder continuity, the results appear inherently limited by this assumption. I am particularly interested in understanding the main challenges that arise when attempting to remove the smoothness and convexity requirements entirely.

4.** Lower bound and tightness**:  Is the result in the paper tight? In other words, is there a lower bound provided?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper refines the hidden-state privacy analysis of Noisy-SGD algorithm under various loss assumptions from strong-convex to non-convex and from Lipschitz smooth to Holder smooth. Building on the analysis of Feldman et al., 2018, and Altschuler & Talwar, 2022, the paper shows that the shift-reduction can be applied in a more effective way by noting that a better bound on the forward Wasserstein distance is possible rather than relying on the degenerate domain diameter. The paper also notes that the Lipschitz-reduction lemma of Altschuler & Talwar, 2022 can be generalized to a Holder-reduction lemma that applies to maps that satisfy the weaker condition of Holder-continuity. Applying the findings, the paper shows better DP bounds than existing work under strongly convex and smooth losses. For convex and smooth losses, their bounds matches that of Altschuler & Talwar, 2022. Additionally, for non-convex but smooth losses, the paper promises a better DP bound than the best via composition or that of output perturbation (or any Pareto-optimal combination of the two).

### Strengths
- The paper improves upon the state-of-the art DP bounds for Noisy-SGD on strongly-convex and smooth losses.
- The relaxation of assumption on Lipschitz-continuity to Holder-continuity is a step forward towards assumption lean convergent DP bound for Noisy-SGD.
- The paper provides DP bounds under two batch subsampling techniques that are most common.
- The ideas in the paper are well illustrated and presented well for a technical audience.
- The paper claims that under non-convex but smooth losses, their DP bound remains finite with number of iteration, unlike composition based-bounds.

### Weaknesses
- The main results in Theorem 3.1, 3.6 and 3.11 and Theorem aren't provided in a closed-form. This makes them hard to operationalize.
- The computational complexity of the presented bounds should be discussed.
- Full batch setting of strongly-convex and smooth case in Theorem 3.1 isn't compared with the bound in Chourasia et al., under identical assumptions.
- Utility analysis is missing. For (strongly) convex and smooth losses, the improved DP bounds can yield a better utility bounds that should be compared against the known lower-bounds [1].

[1] Bassily, Raef, Adam Smith, and Abhradeep Thakurta. "Private empirical risk minimization: Efficient algorithms and tight error bounds." 2014 IEEE 55th annual symposium on foundations of computer science. IEEE, 2014.

### Questions
- In equations on lines 297-305, the inequalities are under the conditioning that $Z_{1:T-1}=Z_{1:T-1}'$, right? Could the authors clarify how the dependence assumptions of Lemma 3.2 and 3.3 holds for applying the (b) inequality?
- Can the bounds presented be computed exactly or we need numerical approximations to realize them?

### Soundness
4

### Presentation
4

### Contribution
4
