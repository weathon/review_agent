# A universal compression theory for lottery ticket hypothesis and neural scaling laws

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 8, 6, 4

## Abstract
When training large-scale models, the performance typically scales with the number of parameters and the dataset size according to a slow power law. A fundamental theoretical and practical question is whether comparable performance can be achieved with significantly smaller models and substantially less data. In this work, we provide a positive and constructive answer. We prove that a generic permutation-invariant function of $d$ objects can be asymptotically compressed into a function of $\operatorname{polylog} d$ objects with vanishing error, which is proved to be the optimal compression rate. This theorem yields two key implications: (Ia) a large neural network can be compressed to polylogarithmic width while preserving its learning dynamics; (Ib) a large dataset can be compressed to polylogarithmic size while leaving the loss landscape of the corresponding model unchanged. Implication (Ia) directly establishes a proof of the dynamical lottery ticket hypothesis, which states that any ordinary network can be strongly compressed such that the learning dynamics and result remain unchanged. (Ib) shows that a neural scaling law of the form $L\sim d^{-\alpha}$ can be boosted to an arbitrarily fast power law decay, and ultimately to $\exp(-\alpha' \sqrt[m]{d})$.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper studies how neural networks and datasets can be compressed by exploiting permutation symmetries. The authors show that symmetric functions can be represented using fewer variables, which implies that both the model and the data can be reduced to polylogarithmic size without significantly changing the loss. This leads to what the authors call a dynamical lottery ticket hypothesis and stronger scaling laws.

### Strengths
The paper presents an interesting idea: using permutation symmetry to achieve strong compression of both networks and datasets.
The theoretical argument (that symmetric functions can be represented with fewer variables), is promising. The results aim to connect model compression, scaling laws, and the lottery ticket hypothesis in a unified framework.

### Weaknesses
The paper proposes a theoretical link between symmetry, compression, and scaling laws. However, the lack of clear algorithmic formulation and the absence of fair experimental baselines limit its current practical relevance.

The main limitation of the paper is the lack of rigor and clarity. The compression process is described only at a high level. It is not clear how one would actually construct the compressed network or dataset in practice. The paper does not include pseudocode or complexity estimates, making it hard to evaluate the tractability of the proposed methods. 

The experimental comparison is incomplete. The proposed compressed network is compared with both the original network and a random sparse network. However, it is already known that random sparse networks perform poorly, while sparse networks obtained with *Iterative Magnitude Pruning* (IMP, Frankle & Carbin 2019) can match the performance of dense ones. A fair comparison should therefore include IMP or other modern sparse training methods.

The compression-error trade-off is not clearly quantified. The claim that a network with (d) parameters can be reduced to polylogarithmic size should be expressed as a function of the error, and possibly compared to existing theoretical bounds.
Finally, some parts of the theoretical presentation are unclear. The meaning of the function $f$ in Theorem 5 is not explained, and the notation $|f' - f| = \omega(d)$ is confusing, since $ \omega(d)$ can mean any function that grows faster than $d$, but such a bound would be vacuous.

### Questions
- Could you provide a concrete description of the compression algorithm? How are the compressed parameters and datasets obtained from the original ones?
- How does your method compare, both in compression ratio and performance, with Iterative Magnitude Pruning or other sparse training techniques?
- Can you explicitly state the trade-off between compression and approximation error, and how it compares with previous results (e.g. to those for the Strong LTH such as Pensia et al., 2020)?
- What exactly does the function $f$ represent in Theorem 5? You didn't define it. Can you clarify the notation?
- Can you argue that the bound $|f' - f| = \omega(d)$ is not vacuous?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The paper introduces the universal compression theorem as a step towards the dynamical lottery ticket hypothesis (LTH), which claims that in a dense network there exists a subnetwork, which when trained in isolation exhibits the same training dynamics as the original one. The theorem states (informally) that a permutation-invariant function of $d$ variables each of dimensionality $m$ can be asymptotically compressed to a function of $O(\text{polylog } d)$ variables. The authors argue that, because many model / dataset objects are symmetric in parameters / datapoints, these results imply polylog-rate network and dataset compression under the assumptions of the theorem.  Another implication of polylog compression is the scaling law $L \approx L_0 + C d ^{-\alpha}$ changing from power law form to stretched-exponential form  $L \approx L_0 + \exp (- \alpha’  \sqrt[m]{d})$, both for model and dataset size.

### Strengths
1. The paper provides theoretical grantees on asymptotic polylogarithmic compression for symmetrical functions. The authors provide Algorithm 1 for compression of symmetric functions using moment-matching and validate it numerically.
2. An important feature is the universality of the result: the implications of the theorem include both neural networks and datasets.
3. A major practical consequence of the work is the potential speed up guarantees on the power-law scaling laws, which are known to "be slow", i.e. have small power exponentials.
4. Although the main result is theoretical, the authors back each claim with numerical experiments: they show on a synthetic function that compression error drops with in agreement with the theoretical bound (Fig. 2); that training dynamics on a compressed dataset follows training on the full dataset (Fig. 3); training performances of full and compressed models are identical to support dynamical LTH (Fig. 4); and compressing a network or dataset leads to a larger scaling law exponent (Fig. 5). These comprehensive validations neatly complement the theoretical backbone of the paper.

### Weaknesses
1. Further empirical evaluation would strengthen this work, as the authors note.
2. The proposed moment-matching algorithm scales poorly with moment order $k$ and dimension $m$ (via $\binom{m+k}{k}$), which limits immediate practical effects despite the asymptotic guarantees.
3. The theoretical claim of polylogarithmic compression yielding a stretched-exponential scaling $\text{exp} (- \sqrt[m]{d})$ is not supported with evidence. The numerical experiments in Section 6 demonstrate how the scaling laws can be improved only for quadratic compression.

### Questions
1. Can you show an example with the scaling laws of a form $L \approx L_0 + c \text{exp} (- \alpha’ \sqrt[m]{d})$ to illustrate the stretched-exponential regime?
2. In numerical experiments in Section 6 the exponent should have improved by a factor of 2: $C d^{-\alpha} = C (\frac{d’}{16})^{-2 \alpha} =C’ (d’)^{-2\alpha} $. The reported values are close but lower, 1.271 vs $2\alpha = 1.366$ and 0.608 vs $2 \alpha=0.616$. Why does this difference appear? And why is it larger for dataset compression? 
3. Many elements of modern neural networks do not fall under the smoothness assumptions, like ReLU, top-k selections, sparse \ quantized representations. How do you imagine expanding your work around those limitations and how would compression rates be affected?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proves a universal compression theorem, showing that almost any symmetric function of $d$ elements can be compressed to a function with $O({\rm polylog}$ (d)) elements losslessly. The theory leads to two key applications. First is the dynamical lottery ticket hypothesis, proving that large networks can be compressed to polylogarithmic width while preserving their training dynamics. Second is dataset compression, demonstrating that neural scaling laws can be theoretically improved from power-law to stretched-exponential decay.

### Strengths
- The paper delivers a rigorous theoretical result that proves the dynamical lottery ticket hypothesis by showing that large networks can be compressed while preserving their original training dynamics.
- Provides a generalized compression theory with broad applicability across diverse domains (e.g., dataset and model compression), demonstrating strong theoretical versatility and significant potential for cross-domain impact.
- Establishes clear practical advantages, such as improved scaling laws and model compression, that are well grounded in the proposed theoretical framework.

### Weaknesses
- The paper lacks a thorough discussion on the applicability of the proposed theory to complex neural architectures such as Transformer blocks, which integrate linear projections, attention mechanisms, and normalization layers.
- There seems to be a missing reference link to the Appendix at line 190 on page 4 (“Appendix ??”).

### Questions
- The model assumes neuron permutation symmetry. Does the assumption is applicable to complex modules in neural networks, such as Transformer block?
- In experiments such as Figure 3 or 4, how much real computation time does the proposed compression take?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work addresses dataset and neural network compression from a moment-matching perspective. Under certain assumptions, this approach establishes novel compression rates and power laws for these tasks. It also enables the boosting of neural power laws, which describe performance versus dataset size dynamics. A number of low-dimensional experiments are conducted to support the claims.

### Strengths
The work is mathematically sound and easy to follow. The text is clear, supported by a decent and concise background overview. The authors provide both rigorous derivations and intuitive explanations for their theoretical results, and the experiments support their claims across a number of settings.

### Weaknesses
My main criticism revolves around the **curse of dimensionality**, which the authors underaddress several times throughout the paper.

1. Both (9) and (10) have dimensionality-dependent exponents, which explode when $m \to \infty$ given that other constants are fixed. This is later combated by selecting $k > (1 = \sigma^{-1}) m - 1$, which, in turn, explodes $\binom{m+k}{k}$. Through some trickery in Theorem 7 (unfortunately, due to time constraints, I was not able to fully verify the math), the authors miraculously balance these issues by attaining a poly-log compression rate.

    That said, one might expect that substituting $d'$ from (45) into (44) should yield errors which are (asymptotically) under some fixed $\omega$. However, when done numerically for $m=10$, $\rho=0.1$, $\omega=0.1$, and any multiplicative constant in (45), I always get an exploding upper bound on the compression error. Reasonable variations of $\rho$ and $\omega$ do not alleviate the issue, which only worsens as $m$ grows.

2. Since $k$ in Theorem 7 grows with increasing $d$, $f$ is required to be increasingly smooth. While most contemporary NNs are $\infty$-smooth almost everywhere, their numerical smoothness degrades with increasing dimensionality or a decreasing learning rate [1]. In practice, this will take a toll on the derived bounds in terms of asymptotic constants or other parameters (e.g., $\rho$ in (44)). This problem remains unaddressed in the main text.

3. The experimental setups are toy, with the dimensionality being $4-12$ orders of magnitude lower than in real-world tasks. In my opinion, this might lead to the following problems:
    - While showing decent performance in low-dimensional regimes, the proposed compression method might entail overfitting in high-dimensional setups. Stochastic gradient descent (SGD) is known to apply implicit regularization during training [2], thus selecting less overfitting solutions. Your method, however, might "overcompress" a NN/dataset: among all solutions, a non-generalizable one is selected (train error or even dynamics are the same, but test error is not).
    - It is known that some problems in ML have exponential (in dimensionality) sample complexity (e.g., density estimation). Your result, however, suggests that these problems are also log-exponential in dimensionality (Theorem 7 applied to dataset compression) given the train error is preserved. The only logical conclusion I can arrive at is that such compression almost always entails overfitting when considering complex problems.

4. While the authors briefly mention the manifold hypothesis in Section 7, it is not clear how one can use it to improve the method. Moment matching is agnostic to manifolds: i.e., it generally cannot capture such intricate structures. Therefore, another manifold learning strategy must be employed beforehand to decrease the dimensionality. Such a strategy typically requires the full dataset, as manifold learning is usually of exponential sample complexity.

[1] Cohen et al. "Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability". Proc. of ICLR 2021.

[2] Smith et al. "On the Origin of Implicit Regularization in Stochastic Gradient Descent". Proc. of ICLR 2021.

**Minor issues:**

1. Broken reference in line 190: "Appendix ??"

### Questions
1. Can you, please, provide additional experiments (e.g., for high dataset dimensionality or low sampling sizes) proving that your method avoids overfitting?
2. I kindly ask to address my concerns in Weakness 1. In particular, I am interested in the numerical verification of the bounds provided.

### Soundness
3

### Presentation
3

### Contribution
2
