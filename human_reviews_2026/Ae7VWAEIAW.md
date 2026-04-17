# High-Dimensional Analysis of Single-Layer Attention for Sparse-Token Classification

- Decision: Accept (Poster)
- Scores: 8, 6, 4, 8

## Abstract
When and how can an attention mechanism learn to selectively attend to informative tokens, thereby enabling detection of weak, rare, and sparsely located features? We address these questions theoretically in a sparse-token classification model in which positive samples embed a weak signal vector in a randomly chosen subset of tokens, whereas negative samples are pure noise. For a simple single-layer attention classifier, we show that in the long-sequence limit it can, in principle, achieve vanishing test error when the signal strength grows only logarithmically in the sequence length $L$, whereas linear classifiers require $\sqrt{L}$ scaling. Moving from representational power to learnability, we study training at finite $L$ in a high-dimensional regime, where sample size and embedding dimension grow proportionally. We prove that just two gradient updates suffice for the query weight vector of the attention classifier to acquire a nontrivial alignment with the hidden signal, inducing an attention map that selectively amplifies informative tokens. We further derive an exact asymptotic expression for the test error of the trained attention-based classifier, and quantify its capacity---the largest dataset size that is typically perfectly separable---thereby explaining the advantage of adaptive token selection over nonadaptive linear baselines.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This work presents a theoretical analysis of a sparse-token classification task, in the high dimensional scaling limit of number of samples and embedding dimension, under the information theoretic limit of $n/d = \Theta(1)$. For a simple attention model, the precise asymptotics of the test error under an optimization procedure is shown, for a given finite sequence length $L$, as well as a fine-grained analysis of how small the signal-to-noise ratio can be. To complement this, a lower bound is given for linear feedforward models, matching intuitions from previous works.

### Strengths
- The results are very rigorous, and hold under the high-dimensional limit of $n,d$ for a given sequence length, and follow upon a line of works analyzing the fundamental task for attending to sparse token signals.
- The analysis shows that two gradient steps suffice for learning the hidden representation of the query matrix, which matches intuitions for weak recovery of the signal from other high dimensional settings such as multi-index models and such.
- Lower bounds are given for linear models, in the sense that the signal strength requires $\sqrt{L}$ scaling compared to $\log L$ in the attention model.

### Weaknesses
See questions below.

### Questions
- Finite sample/dimensional guarantees are given - what are the bottlenecks to extending the analysis to this setting?
- A natural question is whether or not the learned attention model length generalizes to longer length, as shown in the regression setting of Wang et al. (2024). I suspect that it does up to some reasonable upper bound - could the authors comment on this?

### Soundness
4

### Presentation
4

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
This paper presents theoretical results regarding the learnability and representational capacity of a single attention layer on the task of sparse token classification. Compared to two linear model baselines, the attention model is shown to detect signals exponentially weaker in the limit of large sequence length, and is proved to require only two gradient steps for learning meaning representations that align with the hidden signal. Further, it provides a precise characterization of the test errors achieved by the attention model, highlighting a significant gap of the representation power of linear and the attention models. Overall, it presents solid theoretical evidence for the superior performance of the attention model in the sparse token classification task.

### Strengths
1. From my knowledge, it is a novel theory work that studies attention in the task of sparse token classification. It shows a clear advantage of the attention model in terms of representation capacity and learnability in this task.
2. It extends existing literature beyond square loss to general convex losses.
3. This paper is well-written and clearly presented.

### Weaknesses
1. The attention model considered is rather simplified, without multi-heads and separate K, V states. While this benefits theoretical analysis, there exists a clear gap between this simplified model and real practice.
2. While a minor weakness, the capacity result remains a conjecture rather than full proof.
3. Some experiments, even synthetic ones, could further strengthen the work and provide more evidence for the real-world applicability of the presented results.

### Questions
Please see the weaknesses section.

### Soundness
3

### Presentation
3

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
This paper theoretically proves that a single-layer attention model can detect signals exponentially weaker ($\theta = O(\log L)$) than linear classifiers ($\theta = \Omega(\sqrt{L})$) in sparse-token classification tasks where weak signals are randomly scattered across sequences. 

In a high-dimensional asymptotic regime ($d, n \to \infty$ with $n/d$ fixed), the authors show that just two gradient steps enable the attention mechanism to align with the hidden signal and provide exact characterizations of the resulting test error and model capacity.

### Strengths
The paper tackles a meaningful problem--detecting weak, rare, and sparse signals (e.g., lesions in scans)-- where attention’s adaptive token selection offers clear benefits over fixed pooling.

It establishes an exponential gap in representational power, showing attention detects signals at $\theta = O(\log L)$ vs. linear classifiers needing $\theta = \Omega(\sqrt{L})$, where $L$ is the sequence length. The analysis provides exact asymptotic characterizations of test error and shows that just two gradient steps suffice for meaningful query-signal alignment.

Experiments (Figures 1, 3) align closely with theoretical predictions for test/train errors, confirming the asymptotic analysis in moderately high-dimensional regimes.

### Weaknesses
W1.  The 4-stage training protocol (Section 3, lines 270–291) with data splitting $\mathcal{D} = \mathcal{D}_0 \cup \mathcal{D}_1$, only 2 gradient steps on $q$, then full ERM on $w, b$ does not reflect how transformers are actually trained.  

W2.  The attention model (Eq. 5) is single-layer, single-head with no key/value matrices, feedforward layers, layer norms, or residual connections.  

W3. Theorem 2 proves attention can detect signals at $\theta = O(\log L)$ by showing existence of optimal parameters $(q^* = \tau\xi, w^* = \xi)$, but provides no guarantee that gradient descent actually finds these parameters. 


W4. Lemma 4 defines the clipping threshold $I$ using $R^2 = (2/\lambda) \mathbb{E}[\ell(\epsilon; y)]$, but this expectation depends on the solution $w^\star$ being bounded. The proof that $x^\star = x^\star_{\text{clip}}$ holds with probability $1-\delta$ relies on quantities that themselves depend on the solution’s properties, creating potential circularity that needs explicit resolution.


W5. Conjecture 1  provides the main capacity characterization via expressions in Eq. (15), but the derivation involves an unverified “heuristic step” (line 2356) in taking the $\gamma, \nu \to \infty$ limit. 


W6. The high-dimensional limit takes $d, n \to \infty$ with $\alpha = n/d$ fixed while keeping $L, \theta, R$ finite (Assumption 1). This is opposite to practical transformer settings where sequence length $L$ often far exceeds embedding dimension $d$.

### Questions
Please refer to W1-W6.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces a sparse-token classification task, where positive samples demonstrate a weak signal in a random sparse subset of tokens, while negative tokens are i.i.d. Gaussian noise. The authors prove that by learning to recover the sparse subset, a single-layer of attention with a single learnable query token can achieve vanishing test error with the magnitude of signal only logarithmic in sequence length. On the other hand, the authors consider two baselines, fully vectorized linear classifiers and average-pooled linear classifiers, where they demonstrate that the signal magnitude requires to be at least of order $\sqrt{L}$ for sequence length $L$ for vanishing test error.

The authors additionally study the effect of two gradient descent updates on the attention model, which allows the query token to achieve non-trivial alignment with the signal vector. They further characterize the exact limiting train and test errors of the above algorithms, as well as their capacity to overfit.

### Strengths
While the ability of attention to recover a sparse subset of important tokens has been the subject of many recent theoretical works, this paper presents a strong case by studying a fundamental task in classification. A particularly interesting aspect of the paper is on the technical side to accommodate attention in high-dimensional asymptotic analyses of classification error.

### Weaknesses
* The problem introduced does not really need to use the full power of attention. This setting can be seen as having a fixed query vector $\mathbb{1}\_d$ of all ones and a trainable query projection $\mathrm{diag}(q)$. Recurrent/state-space models could also efficiently achieve vanishing test error by going through the input sequence and memorizing tokens that have a high correlation with $q$. It would be interesting, perhaps for future work, to extend this setting to the case where the number of queries in the attention mechanism would also grow, which would enable attention to outperform even more baselines.

* I think some additional details in the main text could provide valuable intuition to the readers. For example, how do the optimal weights for attention/vectorized/pooled models look like in Section 2? I also thing some details might be missing on the loss, e.g. Corollary 1 could not hold if the loss is constant, since $q$ would be left at $0$.

### Questions
* Isn't average pooling a specific case of the fully vectorized classifier? Would that imply that in Section 2, the optimal error of the pooled classifier is lower bounded by that of the fully vectorized?

* Minor typos:
    * Line 103: negative signals -> negative samples
    * Line 122: I believe the authors meant $CR/\sqrt{L}$, but it looks like $R$ is the exponent of $C$ in that expression.
    * Line 458: the capacity if linear classifiers -> of linear classifiers

### Soundness
3

### Presentation
3

### Contribution
3
