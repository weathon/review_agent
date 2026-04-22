# A Law of Data Reconstruction for Random Features (And Beyond)

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 6, 6, 8, 6

## Abstract
Large-scale deep learning models are known to *memorize* parts of the training set. In machine learning theory, memorization is often framed as interpolation or label fitting, and classical results show that this can be achieved when the number of parameters $p$ in the model is larger than the number of training samples $n$. In this work, we consider memorization from the perspective of *data reconstruction*, demonstrating that this can be achieved when $p$ is larger than $dn$, where $d$ is the dimensionality of the data. More specifically, we show that, in the random features model, when $p \gg dn$, the subspace spanned by the training samples in feature space gives sufficient information to identify the individual samples in input space. Our analysis suggests an optimization method to reconstruct the dataset from the model parameters, and we demonstrate that this method performs well on various architectures (random features, two-layer fully-connected and deep residual networks). Our results reveal a *law of data reconstruction*, according to which the entire training dataset can be recovered as $p$ exceeds the threshold $dn$.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper considers a random-features model, and proves that if the number of parameters $p$ of the output layer is larger than the total number of coordinates in the training dataset ($nd$), then the linear subspace spanned by the features of the training examples gives enough information for (approximately) reconstructing training examples. This requires certain assumptions, including some assumptions on the Hermite expansion of the activation function. Inspired by their theoretical results, they suggest a method for reconstruction without knowing the subspace of training features. They demonstrate how their method works in various settings. Also, they show how empirically p=n is the threshold for interpolation while p=nd is the threshold for reconstructability.

### Strengths
The theoretical results (Theorems 1 and 2) are very nice and non-trivial. I think that understanding the theoretical aspects of the ability to reconstruct training data is a challenging and intriguing question, and the paper makes a step in this direction.

Regarding the empirical results, I liked the consistent thresholds of p=n for interpolation vs. p=nd for reconstruction. The authors show that these thresholds appear in various settings, which improves our understanding of this aspect.

### Weaknesses
My main concern is that the implications of the results on reconstruction in the common setting, where the attacker does not know the subspace of training features, are somewhat limited for the following reasons:

First, the theoretical results only consider the case where the attacker knows this subspace, and hence it is unclear whether the attacker can reconstruct training data without this knowledge. The theoretical results imply that it suffices to know the subspace of training features, which is interesting and non-trivial, but it does not allow for reconstruction only from the network’s parameters, as in “common” reconstruction attacks. 

Regarding the reconstruction algorithm that does not assume knowledge of the training features subspace: The proposed method relies on the fact that the output weights are spanned by the features. This observation is similar to Loo et al. (2024), as the authors noted. Thus, this method does not rely on the theoretical results, and it could be argued that they are independent of the theoretical part of the paper. The fact that it is similar to Loo et al. also makes it less novel (although the setting here is different). Having said that, I still think that the empirical results here give convincing evidence regarding the importance of the p=nd threshold. This was not studied in such a systematic way in previous works, and I view it as a valuable observation.

Overall, I think that the theoretical results are intriguing and nontrivial, and that the conceptual contribution regarding the “law of data reconstruction”, which appears both in the theoretical results and in the experiments, is valuable. But given the limitations discussed above, I tend to “weak accept”.

A more minor comment: In the paper, the authors mention that for interpolation, we need more parameters than training samples. A more accurate claim is that the number of parameters times the depth needs to be larger than the number of samples. Or alternatively, that the number of bits in the network’s representation needs to be larger than the number of samples. See the paper “On the Optimal Memorization Power of ReLU Neural Networks” by Vardi, Yehudai, and Shamir for details.

### Questions
The paper only considers the square loss function. However, the empirical part may also work for classification with the logistic or cross-entropy loss. That is, Eq. 11 should still hold for classification (since the max margin predictor is spanned by the inputs). Have the authors verified whether the “law of data reconstruction” also appears in classification?

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
This paper proposed a new law of data reconstruction in overparameterized neural networks. The authors point out that when model parameters contain sufficient information to reconstruct the entire training set. The key finding is that full data reconstruction becomes feasible when the number of parameters 𝑝 greatly exceeds the product of data dimension 𝑑 and sample count 𝑛.

### Strengths
1. Clear statement of the reconstruction law with 2 new theorem.
2. Proposed the first quantitative relationship between model size and data reconstructability.
3. Visualization validation from light-weight models convincingly support the claims.

### Weaknesses
1. Analysis restricted to random features and small networks (even no evidence from the widely-used transformer to show whether it also works on the attention framework), not large-scale modern models.
2. Experiments involve small sample sizes (mainly on 32*32), limiting statistical robustness on larger size image from STL10 or Imagenet.

### Questions
1. Does this theorym could transfer to transformer or diffusion models?
2. Will this thoerm still work when the training datapoints become morem complex with larger size?
3. Will regularization or pruning invalidate the reconstruction law?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper studies data memorization as the ability to reconstruct training inputs from model parameters. It shows that, unlike label memorization (which occurs when $p \gg n$), full input reconstruction requires $p \gg nd$, where $p$ is the number of parameters, $n$ the number of samples, and $d$ the data dimension. The authors prove that in random features models, if the span of the features of $X \in \mathbb{R}^{n\times d}$ lies inside the span of the features of $\hat{X}$, then (up to permutation) $\hat{X}$ must match $X$ with high probability. They also propose a reconstruction loss and show empirically that the same scaling enables recovery in random features, two-layer networks, and deep residual networks.

### Strengths
1. The paper is clearly structured and well written, and it is easy to follow.
2. Data memorization, in the sense of recovering training inputs from a trained model, is an important problem with implications for privacy and model safety. The paper addresses it in a timely way.
3. The theoretical and empirical results are convincing. Although the theory is developed for random feature models with assumptions on the nonlinearity, it is compelling that similar empirical behavior appears in more complex architectures under the same scaling.
4. To my knowledge, both the analysis and the reconstruction algorithm motivated by the theory are novel, and the results are convincing.

### Weaknesses
1. The reconstruction algorithm, although motivated by the theory, still feels somewhat mysterious. In particular, the trained $\theta*$ is only known to be a vector in the span of the features of the training data. Enforcing that $\theta*$ lies in the span of the reconstructed features does not immediately imply that $\varphi(x_i)$ is in $\text{span}(\varphi(\hat{x}_j))$ for all $i$. It would be helpful if the authors could clarify this.
2. The theory appears to require knowing the exact number of training samples $n$. Is this correct? What would happen if $\hat{X}$ had a different number of rows than $X$?
3. In the proof sketch of Theorem 1 (lines 227–229), it seems that the roles of $\hat{x}$ and $x$ might be swapped. Can the authors clarify whether this is a typo?

### Questions
Please refer to the previous section

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes a law of data reconstruction for over-parametrized model in random feature regression showing that the training set can be reconstructed when $p \gg dn$ where, $p$ is parameter count, $d$ is data dimensionality and n the number of samples in the set. They provide elegant theoretical results with empirical validation.

### Strengths
* Clear “law” with an intuitive threshold: The paper isolates a memorable scaling for data reconstruction in random-features regression: once the parameter count satisfies ($p \gg dn$), full input recovery becomes feasible. This makes the results feel elegant and easy to understand.
* Crisp geometric criterion (Theorem 1). If the training feature vectors lie in the span of features generated by candidates, each sample must be close (in input space) to some training sample. This spanning-subspace view is conceptually simple and informative.
* Experiments visualize two thresholds: labels fit near ($p\approx n$), while reconstruction error collapses near ($p\approx dn$). The figures make the phenomenon immediately clear.
* Activation analysis. The parity/sign-ambiguity discussion (e.g., ReLU vs. mixed-parity activations) explains when reconstructions may flip signs and how to avoid it, giving a quick diagnostic practitioners can check.

### Weaknesses
1. Specialized setting and assumptions: The theory is developed for random-feature regression under sub-Gaussian data, Lipschitz activations, and a strong overparameterized regime (e.g., (p) on the order of (nd) with log factors). This is narrower than many trained deep-net scenarios.
2. Partial completeness of theory. The uniqueness/no-duplicates guarantee is proved cleanly for (n=2); a full characterization of global optima (and uniqueness) for general (n) remains open.

### Questions
1. How sensitive is success to initialization and stepsizes? Runs across initalizations, learning rates; report success rates and iterations could help strengthen the paper
2. What is the theoritical challenge in addressing the no duplicates results beyond n=2? 
3. How do results change beyond sub-Gaussian isotropic inputs? Statements/experiments under covariance is not $I$ and heavy-tailed features or discussion on how to potentially handle this in the future would help

### Soundness
2

### Presentation
2

### Contribution
2
