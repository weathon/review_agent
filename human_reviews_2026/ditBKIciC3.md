# Beyond Spectra: Eigenvector Overlaps in Loss Geometry

- Decision: Accept (Poster)
- Scores: 8, 8, 6, 2, 4

## Abstract
Local loss geometry in machine learning is inherently a two-operator concept. While a single loss is locally characterized by its Hessian spectrum, practical learning depends on both training and test losses, whose joint geometry is determined not only by their spectra but by the alignment of their eigenspaces. We establish general foundations for this two-loss geometry by deriving a universal local fluctuation law: the expected test-loss increment under small training perturbations is a trace combining train and test spectral data with a precise factor quantifying eigenvector overlap. We further prove a transfer law describing how overlaps transform under noise. As a solvable model, we apply these results to ridge regression under arbitrary covariate shift, where operator-valued free probability yields asymptotically exact overlap decompositions that identify overlaps as the natural quantities for specifying shift, and resolve multiple descent: error peaks are governed by eigenspace misalignment rather than Hessian ill-conditioning alone. We then validate the fluctuation law in multilayer perceptrons, develop scalable estimators for overlap functionals based on subspace iteration and kernel polynomial methods, and apply them to a ResNet-20 trained on CIFAR-10, showing that class imbalance reshapes train–test geometry through induced misalignment. Together, these results establish eigenvector overlaps as the fundamental missing ingredient in local loss geometry, providing both theoretical foundations and practical tools for analyzing generalization in modern neural networks.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The authors first formalize the local fluctuation at a local minimum of a model, and then the authors analyze the eigen space of the Hessian of the training loss landscape during these fluctuations. Finally, the authors relate the training Hessian to the Hessian eigen space of the test loss landscape to explain loss behavior during covariate shift and commonly studied optimization behavior, such as multiple descent.

### Strengths
1. The reviewer believes that the contribution of this paper is strong. Through analyzing the loss landscape between training and testing, the authors explain the behavior of a couple of important scenarios in modern deep learning, such as covariate shifts (commonly happen in applications), and multiple descent (commonly observed during optimization of over-parameterized models).
2. The author not only provides analysis and empirical results with synthetic training/testing environments but also extends their findings to more complex datasets such as CIFAR. 
3. The work is novel.

### Weaknesses
Minor:
1. The related work seems to slightly lag behind, for exammple, regularizers that encourage cross-domain invariance have recent advancements such as [1]. Maybe the related work can also touch on sharpness-aware optimization and other robust optimization techniques that focus on controlling the gradient and the Hessian. It will round out the related work nicely to tie it back to the more application side of machine learning. 
2. A suggestion on notation: in eq 6, a notation $q$ is used, but it was only defined until the page after. Then also a parameter $\alpha = q^{-1}$ is used through out the paper. Perhaps the author can tidy up this notation and stick with $\alpha$

Major:
1. The presentation of the work can be improved. The reviewer sincerely hope that the authors can use the additional page during rebuttal to strengthen/clarify some of the theoretical points and figures in the manuscript. See following for details. 


[1]: Hasan, Ali, Haoming Yang, Yuting Ng, and Vahid Tarokh. "Elliptic Loss Regularization." In The Thirteenth International Conference on Learning Representations.

### Questions
1. Perhaps the progression from eq 1 to eq 2 can be improved with a slight introduction on how eq 2 is achieved? Similarly, this can be improved through the development of eq 5-7 (or refer to the appendix). 
2. On line 156, the authors assumed $\mathbb{E}z = 0$, is this assumption realistic during scenarios such as covariate shift? If the noise $\epsilon$ is caused by covariate shift while the training data is centered, does this mean $\mathbb{E}z \neq 0$?
3. How should one intuitively understand the paragraph from lines 214 to 217. Does this essentially mean that if the noisy part of the train landscape aligns with the more important part of the test landscape, we will see increased loss value during evaluation?
4. In Proposition 1, can the authors explain conceptually what *"X is free from A, B"* means? Does it simply mean independence?
5. Figure 1 a) seems confusing. What is the purpose of the red contour in Figure 1a)? How are they related to the cyan? Figure 1 b) and c) are cleaner and more understandable. 
6. Figure 3 b), does each color of the contour mean a specific ranking of eigenvalue?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors derive a universal local fluctuation law considering the train/test loss Hessian spectra and overlaps of their eigenspaces, and apply it to ridge regression, trying resolve the puzzle of multiple descent.
They also empirically validate their theory with MLPs.

### Strengths
- This paper provides a new perspective on the importance of the eigenspace overlaps.
- It explains the multiple descent phenomenon from this perspective.

### Weaknesses
- As (3) is very important equation in the paper, I'd like to know how to obtain the equation in detail. The explanation is very unclear to me.
    - The approximations of (1) and (2) are not clear. In what sense, they are approximated? (e.g., $O(\\|w-w_0\\|^2), O(\\|\epsilon\\|^2)$)
    - What is the definition of 
        - $J_{\text{train}}(w,\epsilon)$ (Is it $J\_{\text{train}}(w+\epsilon)$?)
        - $H_{\text{train}}$? If it depends on $w$ as written in the paper ($H_{\text{train}}:=d\nabla^2 J_{\text{train}}(w,0)$), then (1) is not quadratic and $\Delta w$ is not just $-H^{-1}_{\text{train}}z$. 
        - $\Delta J_{\text{test}}$ (Is it $J_{\text{test}}(w_0+\Delta w,\epsilon)-J_{\text{test}}(w_0,\epsilon)$? or without $\epsilon$?)
    - The approximations of (1) and (2), but equality in (3). How do we get the exact equality in (3) from (1) and (2)?
- The caption of Fig 2a says "$J_\text{{\color{red}train}},\Delta J_{\text{test}}$ and bias" but in the panel it says "$J_{\text{test}},\Delta J_{\text{test}}$, Bias."
- em dash? (L41, L222, L262, L348, L353, L361, ...)
- errata? $\mu_\Sigma = p_1\delta_{s_1^{\color{red}2}}+p_2\delta_{s_2^2}$
- What is $\tilde\lambda{\color{red}'}$ in (9)?
- errata? (L265) Isn't it $s_1^2,s_2^2=2^0,2^{-4}$? The eigenvalues in Fig 1(b) should be $s_i^2$ as written in (8).

### Questions
see weaknesses

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper studies the use of loss geometry for both train and test sets, specifically the eigenvector overlap between the two, rather than just their spectra, as measure of loss geometry and to predict generalization. It shows that this measure explains covariate shift and multiple descent through a unified lens via simulations on synthetic data. It also develops an efficient estimator to show how this measure can quantify how class imbalance can induce misalignment in train-test loss geometry.

### Strengths
The paper makes interesting contributions to use eigenvalue overlap between train and test Hessians, rather than just their spectra, as measure of loss geometry and generalization, and presents a scalable numerical estimator and empirical validations.

### Weaknesses
The paper is missing an expanded related work section with a detailed discussion on most closely related works. As someone who is not very familiar with the literature on random matrix theory, and eigenspace overlap estimators, it is hard to judge the novelty of this work relative to other works. It would be good to add a more detailed discussion comparing and contrasting the work with most closely related work.

The experimental methodology in Section 3.4 is somewhat strange. It is stated that “a CIFAR-10 trained ResNet-20 was obtained from Chen”, and then “5000 train and test examples were randomly selected to define train and test Hessians”. A more convincing experiment would be to train it from scratch using the selected train set samples. Additionally, why not use imbalanced train set with balanced test set to compare the effect of imbalance?

### Questions
Can authors add details on compute and runtime for the CIFAR-10 results in Section 3.4? Is it possible to use the proposed method for larger models?

Suggestions to improve writing/readability:

1. The paper uses hyphens instead of em dashes in almost every occurrence, and in some cases, it uses en dashes instead of hyphens, please fix.
2. The paper uses $J$ to denote the loss. I suggest using $\mathcal{L}$, or simply $L$, which is more standard. $J$ could be misinterpreted as denoting the Jacobian.
3. In line 157, the expectation terms are missing square brackets.
4. In line 162, the order of $H_{\text{test}}$ and $C_{\text{train}}$ is swapped. 
5. In line 215, it should be ‘significant’.
6. In Eq. (8), it should be $\delta_{s_2}^2$.
7. In Fig. 1, panel (b) is missing the y-axis label, please clarify.
8. In most cases, the word ‘traces’ is used to refer to the solid lines in the plots. I suggest simply using ‘solid lines’. 
9. In most cases the subfigures are referred to as, e.g., Fig. 1a instead of Fig. 1(a), please fix.
10. Fig. 2 caption states “Traces in panel a) correspond to gold and blue lines”. Please clarify.
11. In line 323, the phrase “geometric cartoon” should be rephrased.
12. In line 377, ‘2d’ should be ‘2D’.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper studies how the relationship between the 2nd order shape of the train and test objectives affects generalization. They then use the general results to study several regression settings, including covariate shift. Finally, they extend the results to neural networks.

### Strengths
The results are interesting and provide an additional perspective from which to view generalization. I am not familiar enough with the details of the related literature to know if this is the first work to consider this, but I trust other reviewers will know more. The paper also contains a very large number of results.

### Weaknesses
As written, this paper reads like a physics paper. In order to be published in a computer science / ML venue, I think it needs a bit more exposition describing notation, exactly the approximations being made in Section 3, and what the operators in 3.1.2 represent in the ML setting. 

1. Is $\Delta J_\text{test} = J_\text{test}(w_0 + \Delta w) - J_\text{test}(w_0)?$
2. What is the source of the noise $\epsilon$? Is it sampling noise, label noise, general?
3. It would help to break up Section 3 into Theorem statements.
4. It would help to have prose at the start of Section 3, 3.1 describing what those sections do.
5. Consider separating out preliminaries for notation and results. The authors should be more explicit about things like what the noise is, what exactly $\Delta J_\text{test}$ is, etc.

It seems like a nice paper, but I would recommend either submitting it to a different venue or writing it more in the style of ML theory papers for publication at an ML conference.

### Questions
Feel free to clarify the points above.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper analyzes the alignment between eigenvectors of train and test loss Hessians and its impact on generalization, extending prior works that focused mainly on eigenvalue spectra. A universal local fluctuation law is formulated to demonstrate the predictive role of eigenvector alignment for test loss. The theory is further applied to linear regression under covariate shift and multiple descent scenarios. Experiments extend the framework to neural networks, including MLPs and ResNet-20 trained on CIFAR-10, with a novel algorithm estimating eigensubspace alignment in the bulk of small eigenvalues.

### Strengths
- The paper provides a novel and original analysis revealing the importance of eigenvector alignment between train and test Hessians, which is an interesting and underexplored aspect in the literature.
- A new framework is introduced to evaluate the alignment between eigenvectors associated with the bulk of small eigenvalues.
- The analysis in linear regression provides concrete intuition on covariate shift and multiple descent.

### Weaknesses
-  Although the eigenvector alignment analysis is theoretically interesting, it is unclear how this insight could translate into practical benefits for model training, as test data are not available during training.
-  The experiments involving neural networks in Sections 3.3 and 3.4 lack clear motivation. Section 3.3 partially supports the theory in Section 3.1 but does not provide concrete insights into MLP generalization or learning dynamics. The purpose of the figures at the bottom of Figure 4 is also unclear and needs further explanation. Section 3.4 proposes an interesting scalable estimation method for bulk subspace alignment, yet the analysis does not seem to make it a central element of the argument, which leaves its importance underemphasized. Moreover, the section concludes by examining train–test Hessian misalignment under test-class imbalance, without providing additional insights into generalization.
-  The paper’s organization could be improved for readability. Key equations (e.g., (13)) and concepts (e.g., smoothing kernels in (10)) are relegated to the appendix. Terms such as “error increment” and “bias” in Figure 2(a, c) should be explicitly defined in the main text.

**Minor comments**

- The caption of Figure 4 (referring to $H_{test}$) does not match the main text (referring to $H_{train}$).

### Questions
-	Please refer to the points raised in the weaknesses section.
-	The developed theory resembles the Takeuchi information criterion (TIC). Could the authors discuss potential connections or differences? (see, e.g., Thomas et al., On the interplay between noise and curvature and its effect on optimization and generalization, AISTATS 2020). 
-	How can the analysis be adapted to account for stochastic optimization?
-	How robust is the estimator proposed in Section 3.4? Some ablation studies on its hyperparameters would strengthen the argument.

### Soundness
3

### Presentation
2

### Contribution
2
