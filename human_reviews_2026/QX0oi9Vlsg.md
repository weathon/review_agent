# Adversarial Bottleneck Method for Vision-Language Large Model Explainability

- Decision: Reject
- Scores: 4, 2, 6, 4

## Abstract
Nowadays CLIP is a leading vision-language model, showing strong functionality, especially in tasks like search engine matching. However, its high performance is often accompanied by the complexity of the decision-making process, making the interpretability of the model a major challenge. Existing XAI methods mainly focus on unimodal settings, with state-of-the-art methods often being attribution algorithms based on adversarial attacks. These methods perform well in unimodal tasks such as image classification. However, expanding these methods to handle cross-modal tasks (such as image-text alignment and cross-modal retrieval) presents several obstacles. For multimodal tasks, the most effective XAI methods currently rely on the bottleneck principle, which limits information flow to analyze model decisions. In this paper, we propose a new approach that integrates adversarial attribution methods with the bottleneck principle. This approach not only interprets multimodal models such as CLIP but also preserves the advantage of unimodal attribution algorithms in precisely identifying key features that influence model decisions within a specific modality. By introducing our model, we can obtain a more robust and broadly applicable representation for vision-language models, further enhancing their transparency and trustworthiness in complex tasks. Comprehensive experiments demonstrate that, compared to state-of-the-art XAI methods, our approach improves the interpretability of text and images by 69.12\% and 19.36\%, respectively. Our code is available at https://anonymous.4open.science/r/ABM-5C28/

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces ABM, a VLM interpretation method that works by fusing adversarial attribution and information bottleneck. ABM improves over M2IB by performing adversarial updates on intermediate-layer bottleneck variables with a sign-based step scaled by the original magnitude and clipped by a constraint, which overcomes the limitation of M2IB including sensitivity to hyper-parameters and difficulty in gradient-based optimization.

### Strengths
1. The idea to integrate adversarial attribution and information bottleneck for VLM explanation is interesting.
2. The authors provide theoretical insights into the proposed method.
3. The evaluation is comprehensive.

### Weaknesses
1. The impact of this paper is small to me. It improves over M2IB, a very specific VLM interpretation technique. And the improvement mainly focuses on the optimization of M2IB.  M2IB is not the SOTA method for CLIP interpretation, with later work such as Grad-ECLIP [1] showing much better performance. 
2. The readability of the math part is poor.  Many symbols are reused to represent different meanings (e.g., $T$ represents both text and the total steps, $g$ appears in both eq.3 and eq.16 with different meanings). And many clarifications are posited far away from the equation (e.g., the explanation in line 315 should be put earlier). This makes the paper very difficult to read. 
3. ABM is built on M2IB, but the paper does not introduce the specific algorithm or derivation of M2IB, only a high-level introduction of M2IB is given on Sec.3.2.1. This makes the paper less self-contained. For example, when introducing eq.11, the KL term seems very strange as it has not been mentioned before. Also, I would recommend the authors to include an algorithm box to better clarify ABM.
4. I do not totally understand the proof of theorem3.1. There are some typos. For example, in eq.16, it states, $\tilde{z}^{t+1}=g(z^t)$, then in 17, why not $I(\tilde{z}^{t+1}, e_{m’})= I(g(z^t), e_{m’})$? Moreover, how can you entirely ignore the higher-order terms when proving the monotonicity?
5. In line 372, the definition of confidence drop is confusing. Shouldn’t a larger drop confidence drop when removing the important features denote better interpretability? The correct definition should be the drop in performance if only the high-attribution parts are kept.

[1] Zhao, Chenyang, et al. "Gradient-based visual explanation for transformer-based clip." ICML 2024.

### Questions
1. Could you compare your method with Grad-ECLIP?

### Soundness
2

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
The authors study the interpretability of vision language models (VLMs) and attempt to improve the existing M2IB method (reviewed in equation 2) by the framework called "Adversarial Attribution Theory". In Sec. 3.2.1. the authors argue that the M2IB method may give untrustworthy results because of the fixed choice of hyperparameter $\beta$ and the hyperparameter choice of the gradient descent method (Lines 216-220). Then, it seems the authors review some calculus in section 3 to propose a new optimization approach, by introducing the extra $C_z$ clipping operator. They present numerical results in section 4.

### Strengths
-The authors' motivation to improve the interpretability of CLIP is sensible.

### Weaknesses
The paper has several flaws in its theoretical claims, technical soundness, and presentation.

-**Incorrect theoretical result in Theorem 3.1**: Theorem 3.1, the paper's only theorem, tries to prove Equations 7 and 8. However, the proof of Equation 7 is incorrect and suffers from a basic mistake. In the Appendix sec A, the authors mention the following equation in their proof (Lines 663–665):

> $$I(\tilde{z}^0, x_m) = H(x_m) − H(z | x_m) = H(x_m)$$

The above equation is wrong and the correct equation is $I(Z;X)= H(X) - H(X|Z)$, where the conditional entropy is mistakenly stated to be $H(Z|X)$ by the authors. This error invalidates the authors’ claim, because they use the wrong equation to claim $H(z | x_m) = 0$. This is while, in the correct form, $H(x_m|z) \neq 0 $ unless $x_m$ is a function of $z$, being exactly the opposite of the authors' statement in front of Equation 13 and disproving their claim in Equation 7.

-**Trivial theoretical claim in Equation 8 of Theorem 3.1**: In addition to the wrong statement in Equation 7, the authors state a trivial result in Equation 8 (second part of Theorem 3.1). If I understood the claim correctly (the presentation is unclear), Equation 8 says that, assuming a zero error for the first-order Taylor series expansion (as supposed in Line 671 and not mentioned in Theorem 3.1), applying the gradient ascent with the clipped gradient in Equation (9) will only increase the function value.

However, I believe this is a trivial statement under the assumption of zero error for the first-order Taylor series expansion. Given the first-order Taylor expansion, the function value will trivially increase locally along any direction with a positive inner product with the gradient direction, and the clipping operation is the projection of the gradient vector on the $\ell_\infty$-norm ball, which will obviously have a positive inner product with the gradient direction. I am wondering in what sense Equation (8) in Theorem 3.1 proves something non-trivial, as this part of the theorem is the only remaining theoretical claim in the draft, excluding the wrong Equation (7).

-**Clarity of the presentation**: I cannot understand the purpose of Equations 3 and 4. Equations 3 and 4 seem to state the fundamental theorem of calculus, that if one integrates the gradient of a function along a path from $z^0$ to $z^T$, the output is the function difference evaluated at $z^0$ and $z^T$. I cannot appreciate why the authors spend more than half a page on this basic fact. In what sense does this discussion support the known optimization iteration (projected gradient ascent using $\ell_\infty$-norm ball) in Equation 5?

-**Presentation issues**: There are several writing errors in the text. Some obvious cases are the misspecified references “(?)” in Lines 182–184 and the wrong in-line position of the $\log$ input in Equation 11.

Also, I could not see how the authors obtained Equation (10) from what they discussed before. $z^{(t)}$ is supposed to follow the update Equation (6) with the mutual information between the CLIP input and output variables. How does this equation lead to Equation (10)? What is the role of the mutual information in getting the update rule in (10) from Equation (6)? As said, it is difficult to follow the discussion after this equation due to the incoherence in writing and notations.

### Questions
1. Can the authors clarify the correctness of Equation (7) and the non-triviality of Equation (8) in Theorem 3.1, which appears to be the main technical claim of this work?

2. How do the authors derive the update rule (10) from Equations (6) and (9)?

3. The authors criticize the M2IB method of Wang et al. (2023) in Equations (1) and (2) for relying on the $\beta$ hyperparameter in Equation (2) and for using gradient ascent to solve (2) (as stated in Lines 210–213). Can the authors clearly explain how their algorithm differs from that of Wang et al. (2023)? Specifically, is the only modification the introduction of the clipping operator in Equations (6) and (9) in place of the vanilla gradient ascent update?

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper are proposed to enhance explanations for large-scale Vision-Language Models (VLMs) like CLIP. It identifies critical flaws in two major categories of existing Explainable AI (XAI) methods when applied to this cross-modal setting. To overcome these limitations, the paper proposes a framework called the Adversarial Bottleneck Method (ABM). ABM synthesizes the strengths of both adversarial attribution and the information bottleneck principle.

### Strengths
The core idea of ABM using an adversarial update process to achieve the goals of the information bottleneck is considerably novel. The method reframes the bottleneck optimization problem, replacing a difficult-to-tune hyperparameter $\beta$ with an iterative optimization process governed by a more intuitive parameter. The theoretical reasoning provided in Theorem 3.1 establishes a theoretical intuition for why this adversarial method works. The quantitative results presented in this work also show sufficient improvements compared to other baselines.

### Weaknesses
1. The paper positions itself as eliminating heuristic hyperparameters, primarily targeting M2IB's parameter $\beta$. However, it introduces its own hyperparameter $T$. The ablation study in Figure 2 shows that the model's performance on images is sensitive to the choice of $T$, also requiring the heuristic and empirical choice of hyperparameter tuning. While $T$ may be more intuitive than $\beta$, it is still a hyperparameter that requires tuning for optimal performance. The claim of eliminating parameters is somehow an overstatement.
2. One question is about the evaluation scheme for test modality. The evaluation for text interpretability in Appendix C is defined as a binary indicator. While the authors follow the evaluation in M2IB, this metric could easily mask the true performance differences between methods or accurately reflect the quality of the explanation. Therefore, the stability of text results across different T values in Table 5 compared to the image modality looks like a result of this binary metric rather than true performance. Could the author provide some discussions about this point?
3. Minor typos: Page 4, Section 3.2.1 "Integrated Gradients (?) or Grad-CAM (?)"

### Questions
Please refer to the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper introduces the Adversarial Bottleneck Method (ABM), a framework for improving explainability in vision-language large models (VLLMs). It provides a bound relating adversarial risk to mutual information between latent codes and input features.

### Strengths
+ the method is theoretical solid. It integrates adversarial robustness and information bottleneck theory.
+ The experiments are comprehensive. The evaluation covers diverse VLLM tasks (captioning, VQA, entailment).

### Weaknesses
- Baseline comparison.
Comparisons focus mainly on M2IB, VIB, and GradIB. It would be informative to include more recent multimodal explainability baselines like BLIP-Explain or ALIGN-Attribution.

### Questions
refer to weakness

### Soundness
3

### Presentation
3

### Contribution
2
