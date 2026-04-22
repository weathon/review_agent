# Pointwise Generalization in Deep Neural Networks

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 6, 4

## Abstract
We address the fundamental question of why deep neural networks generalize by establishing a pointwise generalization theory for fully connected networks. For each trained model, we characterize the hypothesis via a pointwise Riemannian Dimension, derived from the eigenvalues of the \textit{learned feature representations} across layers.   This approach establishes a principled framework for deriving tight, hypothesis-dependent generalization bounds that accurately characterize the rich, nonlinear regime,  systematically upgrading over approaches based on model size, products of norms, and infinite-width linearizations, yielding guarantees that are orders of magnitude tighter in both theory and experiment.  Analytically, we identify the structural properties and mathematical principles that explain the tractability of deep networks. Empirically, the pointwise Riemannian Dimension exhibits substantial feature compression, decreases with increased over-parameterization, and captures the implicit bias of optimizers. Taken together, our results indicate that deep networks are mathematically tractable in practical regimes and that their generalization is sharply explained by pointwise, spectrum-aware complexity.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper gives new generalization bounds to explain the good generalization properties of deep neural networks, a long-standing topic in learning theory. The proposed bounds are called ``pointwise'', meaning that they apply for every hypothesis individually and are also data-dependent. The main technical elements are based on an application of PAC-Bayesian bounds with well-chosen probability distributions. This allows the authors to introduce an effective dimension, which is related to a fractal dimension notion of a data-dependent prior. In the case of fully-connected deep neural networks to a notion of Riemannian dimension, based, in particular, on the feature Gram matrices. Empirical studies support the theoretical findings.

### Strengths
- Understanding the generalization error of modern deep neural networks is an important topic.
- The introduced notions of effective dimensions might have interest on their own.

### Weaknesses
In my opinion, there are three main weaknesses.

 *1. Poor literature review*: The paper pretends to address the long-standing question of generalization and explicitly states "theory has not kept pace". Despite this, there is an important lack of literature review on the rich literature review. Most of the classical references are missed and the introduction barely contains citations. For instance, algorithmic stability or information-theoretic bounds are never mentioned. PAC-Bayes bound are mentioned but with almost no reference except (Dziugaite and Roy, 2017). NTK theory is mentioned but with no reference. The same remark can be made or uniform covers, VC dimension, and product of norms. Moreover, similar notions of intrinsic dimensions already appear in machine learning, but with no reference, see [1] for instance.
 In conclusion, the authors claim that theory has not kept pace on generalization without acknowledging the rich literature on this question.

*2. The proofs seem to contain critical mistakes*, such as:
 - Proof of Lemma 4: The proof is based on classical PAC-Bayes bounds but they cannot be applied here. Indeed, Lemma 9 is only true for a prior distribution $\pi$ that is independent of the data. It can be seen in the proof of thm 2.1 of [Alquier, 2024] (cited for lemma 9): the critical step is to apply Fubini's theorem to switch the prior and the data distribution, it is only possible if \pi does not depend on the data. 
 - Proof of thm 1: First, at line 1030, the notation does not make sense because there is an expectation on z outside the sup, while z is already the integrand of $\mathbb{P} - \mathbb{P}_n$ according to your notation. Even if this is fixed, I am not convinced by the symmetrisation argument at line 1040 because the inside of the supremum is not symmetric in z and z'.

*3. Empirical section:* at line 81 and in the contributions section, the empirical validation section is clearly treated as though it was part of the main paper and not of the appendix. In my opinion, an effort should be made to compress the main paper so that a proper empirical section can be included in the main text.

Regarding, the proofs, please correct me if I am wrong. 

Finally, here are some more minor issues:
 - Line 69: a effective -> an effective
 - The notation at equation 3 is confusing to me because $z$ because $z$ is the integrand on the left-hand side, so it should not appear in my opinion. Maybe something like $\mathbb{P}(\ell(f, \cdot))$ would be better.
 - Line 133 - 134, is a word missing in the sentence?
 - Line 901 - 902: too much space between so and on
 - Equation 15, I think it should be said that the uniform distribution is meant with respect to $\pi$
 - I don't think the notion of prior distribution has been properly defined.
 - Proof of Theorem 1: sometimes the $\otimes$ is missing in the products between distributions, is it a typo?


[1] "Fractal Structure and Generalization Properties of Stochastic Optimization Algorithms" Cameo et al., 2021.

### Questions
- Line 167, it is claimed that the bounds covers a rich class of models. To my understanding, if the point wise dimension grows as $d(f) \varepsilon^{-2}$, it means that the fractal dimension is at most 2, can you comment further on how this is restrictive?
- To my knowledge, eq. (14) is a step to prove eq. (13), why not go directly go to eq. (14)?

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper aims to sharp understand the generalization of neural network. This is done by developing pointwise, spectrum-aware PAC-Bayesian generalization bounds. Their prior is then constructed considering the low-rank features in FFNs. The new bounds reveal that generalization can be captured by finite-scale Riemannian dimension, which also empirically better capture generalization under overparameterization.

### Strengths
- This paper provides deeper understanding by connecting generalization with finite-scale Riemannian dimension, which is then connected to effective dimensions of FFNs. The resulting bounds provide better indicator of generalization compared to existing results and reflects many important generalization behaviours under overparameterization.
- This paper contributes many advancement in PAC-Bayesian theories, eg, single-hypothesis bounds. The bounds are then refined with consideration on exploiting implicit biases like spectral properties of hidden features. 
- This paper involves multiple techniques of novelty, including loss symmetrization, non-perturbative expansion, etc. It also gives new results in pure mathematics.

### Weaknesses
- Weight and data dependence of $\pi$: The **Key Challenge** paragraph has emphasized that $\pi$ cannot rely on $W$. However, the hierarchical construction of $\pi$ involves subspaces subspaces, whose dimension is the effective rank of $G(W)$. It relies not only on weight $W$ but also on data $X$ or $z$. 
- Evidence on tightness is not direct: The bound seems easy to compute and its most complicated part, ie, the Riemannian dimension, has been computed. So why not compute the entire bound.
- Presentation and typos: 
  - Theorem 1 emphasizes prior's data dependence. However, in latter construction of $\pi$, lots of efforts have been put to make $\pi$ independent of $W$ and thus data $z$ (please correct me if my understanding is wrong), making this data dependence useless. Also I found the proof incomplete  for the data dependence of Theorem 1 (See my Question 1 below). I believe it is better to remove the claimed data dependence.
  - Eq.(7) has incomplete parentheses.
  - Unspecified matrix norm in Eq.(9).

### Questions
- Questions on proof details: In the proof of Theorem 1, Line 1040 bounds original losses using symmetrized losses. This step relies on the fact that $\\ell(f ; z′) − \\ell(f ; z)$ has a symmetric distribution. But as far as I am concerned (please correct me if I was wrong), this step not only involves this symmetrically distributed term, but also the $\\log \\frac{1}{\\pi(\\dots)}$ term where $\\pi$ depends on $z$ only and does not have a symmetric distribution. Could the authors provide more details on how this term is handled in this complicated mixture with $\\sup_f$? 
    - I did some calculation and the symmetric distribution seems not sufficient: Let's assume a simplification with $n=1$ so all $\\mathbb{P}\_n$ notations disappear. Then this step can be abstractly seen as whether
        
        $$
            \\begin{aligned}
                \\mathbb{E}_{z, z'}  \\sup_f \\ell(f; z') - \\ell(f; z) - b(f; z) 
                &\\overset{?}{=} \\mathbb{E}_{\xi} \\mathbb{E}_{z, z'}  \\sup_f \xi (\\ell(f; z') - \\ell(f; z)) - b(f; z)\\\\
                &=\\frac{1}{2} \\left(\\mathbb{E}_{z, z'}  \\sup_f \\ell(f; z') - \\ell(f; z) + b(f; z) \\right) + \frac{1}{2} \\left(\\mathbb{E}_{z, z'}  \\sup_f \\ell(f; z) - \\ell(f; z') - b(f; z) \\right)\\\\
                &=\\frac{1}{2} \\left(\\mathbb{E}_{z, z'}  \\sup_f \ell(f; z') - \\ell(f; z) - b(f; z) \\right) + \\frac{1}{2} \\left(\mathbb{E}_{z, z'}  \\sup_f \\ell(f; z') - \\ell(f; z) - b(f; z') \\right),
            \\end{aligned}
        $$

        where $b(\\cdot, \\cdot) \\ge 0$ abstracts the pointwise dimension term.
        It is equivalent to whether

        $$
            \\begin{aligned}
                \\mathbb{E}_{z, z'}  \\sup_f \\ell(f; z') - \\ell(f; z) - b(f; \\red{z}) 
                &\\overset{?}{=} \\mathbb{E}_{z, z'}  \\sup_f \\ell(f; z') - \\ell(f; z) - b(f; \\red{z'})
            \\end{aligned}
        $$

        There are counter-examples for this statement:

        $$
            z, z' \\in \\{-1, +1\\}, f \\in \\{-1, +1\},\\\\
            \\ell(f; z) = f z + 42, b(f; z) = \\begin{cases}
                0 & z = f\\\\
                42 & z \\neq f
            \\end{cases} 
        $$

        Then we have

        $$
            \\begin{aligned}
                \\sup_f \\ell(f; z') - \\ell(f; z) - b(f; z)
                =& \\sup_f f \\times (z' - z) - 42 \\cdot \\mathbb{I}[z \neq f]\\\\
                =& z(z' - z) \\quad\\quad (\\text{$f$ must equal $z$}),
            \\end{aligned}
        $$

        whose expectation is $\\mathbb{E}[-z^2] = -1$.
        On the other hand, we have

        $$
            \\begin{aligned}
                \\sup_f \\ell(f; z') - \\ell(f; z) - b(f; z)
                =& \\sup_f f \\times (z' - z) - 42 \\cdot \\mathbb{I}[z' \neq f]\\\\
                =& z'(z' - z) \\quad\\quad (\\text{$f$ must equal $z'$}),
            \\end{aligned}
        $$

        whose expectation is $\\mathbb{E}[z'^2] = 1$.
        The two sides do not equal and the statement is generally untrue.
        Therefore, the symmetrical distribution of the loss difference term alone seems not enough and I kindly ask whether the step relies on more conditions or the symmetrical distribution is applied in some special way?
    - The conclusion of Theorem 1 also seems a bit unexpected because the data-dependent prior seems comes at little cost. For example, in Theorem 2.4 from Alquier (2024), the data-dependent choice of $\\lambda$ costs a penalty of $\\log \\text{card } \\Lambda$. In contrast, Theorem 1 does not contain any similar penalty. It seems all extra terms compared to Lemma 9 comes from either making the hypothesis deterministic ($\\epsilon$) or from standard procedures (I skimmed the proof and my finding may be imprecise, so please correct me; but at least they do not depend on $\\pi$). As a result, it seems one can arrive at extremely small generalization bounds at least for any deterministic learner: Assume $f = A(z)$ is the output of the deterministic learner. Then we can fix a small $\\epsilon \\sim \\sqrt{1/n}$ and then construct $\\pi$ to exactly cover the ball around $f=A(z)$. Since $f$ only depends on $z$, the loss-induced metric only depends on $z$ as well, and the radius $\epsilon$ of the ball is fixed, such $\\pi$ is a well-defined data dependent prior. However, this $\\pi$ makes the pointwise dimension term $0$ *without* any penalty. In this case, one would have a $\\tilde{O}(1/\\sqrt{n})$ rate, regardless of architecture, optimizer or data properties.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors consider the generalization of fully connected networks at the trained parameters. The proposed generalization bound depends on the Riemannian dimension, which is based on the spectral properties of the feature representations. The proposed complexity measure is tighter compared to related approaches, and the Riemannian dimension exhibits appealing properties that can potentially enhance and explain generalization.

### Strengths
- Addresses an important and timely question about understanding the behavior of neural networks.
- The analysis and technical content appear rigorous and sound. However, I have not checked the theoretical derivations in detail.
- The pointwise dimension and the non-perturbative expansion are interesting contributions.
- The experiments support the theory, showing that the proposed complexity measure is smaller than related metrics.

### Weaknesses
- I believe that the clarity and accessibility could be improved. While this is a technical theoretical work and simplifying further is challenging, the text is somewhat hard to follow. Providing clearer intuition and explanations for the theoretical results would help.
- The paper is dense, as the generalization bound depends on both probabilistic and differential geometric concepts. The exposition of the latter (Sec. 3.2) could benefit from illustrations.
- Some terms may be well-known in the literature, but I believe, should still be defined in the paper to make it self-contained. See questions below for examples.
- The relevance of certain sections is not directly clear. See questions below.
- From the experiments, while the Riemannian dimension appears smaller than related metrics, it does not seem to provide a non-vacuous generalization bound. I may be missing something.

### Questions
1. In Eq. 4, the denominator represents the integral of the prior over a ball centered at $f$?
2. How can a data-dependent prior be considered, and how is it related to Theorem 4?
3. The discussion after Theorem 2 is somewhat unclear, and similarly, the relevance of Sec. 2.2. As regards, Sec. 4.2 is not immediately obvious if the Riemannian dimension is implicitly regularized during training or if the idea is to consider it as an explicit regularizer.
4. In the experiments, even if the bounds are smaller than the compared approaches, they still appear rather vacuous. Am I missing something?
5. The considered architecture does not include bias. Would the analysis change significantly if biases were included?

### Soundness
3

### Presentation
2

### Contribution
3
