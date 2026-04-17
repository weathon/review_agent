# Understanding Private Learning From Feature Perspective

- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
Differentially private Stochastic Gradient Descent (DP-SGD) has become integral to privacy-preserving machine learning, ensuring robust privacy guarantees in sensitive domains. Despite notable empirical advances leveraging features from non-private, pre-trained models to enhance DP-SGD training, a theoretical understanding of feature dynamics in private learning remains underexplored. This paper presents the first theoretical framework to analyze private training through a feature learning perspective. Building on the multi-patch data structure from prior work, our analysis distinguishes between label-dependent feature signals and label-independent noise—a critical aspect overlooked by existing analyses in the DP community. Employing a two-layer CNN with polynomial ReLU activation, we theoretically characterize both feature signal learning and data noise memorization in private training via noisy gradient descent. Our findings reveal that (1) Effective private signal learning requires a higher signal-to-noise ratio (SNR) compared to non-private training, and (2) When data noise memorization occurs in non-private learning, it will also occur in private learning, leading to poor generalization despite small training loss. Our findings highlight the challenges of private learning and prove the benefit of feature enhancement to improve SNR. Experiments on synthetic and real-world datasets also validate our theoretical findings.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper develops a theoretical framework for analyzing differentially private stochastic gradient descent (DP-SGD) from a feature learning perspective. It distinguishes between label-dependent feature signals and label-independent data noise, using a two-layer CNN with polynomial ReLU activation and a multi-patch data structure. The authors show 
- Effective private learning requires a higher signal-to-noise ratio (SNR) than non-private learning.
- Data noise memorization in non-private settings persists (and may amplify) in private settings.
- Theoretical findings are validated via synthetic and real-data experiments.

### Strengths
- Novel perspective: The paper bridges differential privacy and feature learning theory—two previously disconnected theoretical frameworks. Analyzing DP-SGD dynamics via feature-signal and data-noise decomposition is original and relevant.
- Clear theoretical contributions: The main theorems (Theorem 4.6 and 4.10) rigorously characterize the learning dynamics of signal vs. noise under noisy gradient descent. The scaling conditions involving SNR, ε, and n provide interpretable insights into when private learning succeeds or fails.
- Connection to empirical literature: The theory aligns well with empirical observations from prior work (e.g., Tramer & Boneh, 2020; Tang et al., 2024), reinforcing its credibility and potential impact.
- Readable structure: The introduction and motivation are well-written, and the results are carefully organized into signal-learning and noise-memorization regimes.
- Synthetic experiments: Simple but effective experiments confirm the theoretical intuition—especially the threshold effect of SNR and privacy budget $\varepsilon$.

### Weaknesses
- Please use \cite, \citep, and \citet correctly. Note that the form of \cite can vary between different Latex template. 
- About Theorem 1.1, 4.6, 4.10:
	- Since $\mathbf{x}$ and $\boldsymbol{\xi}$ are random variables, should the SNR—a deterministic quantity—be expressed in terms of its expectation? I might have missed some context, but this notation caused some confusion about the learning setup when reading the discussion around Theorem 1.1.
	- Also, can SNR be approximated or estimated for real datasets in practice?
	- The results describe how, depending on whether SNR is large or small, the CNN captures certain types of information. The conditions use $\tilde{\Omega}(1)$ to represent thresholds on SNR. It would be helpful if the authors could clarify (perhaps I missed this) the exact constants behind these $\tilde{\Omega}(1)$ terms, and whether the corresponding regions overlap or are strictly separated. This clarification would help readers understand what happens in the intermediate regime—i.e., whether both types of information can be captured simultaneously or if a gap remains.
- What does the term feature enhancement specifically mean in the context of this paper?
- There appears to be a typo on line 264.
- The statement \textit{When data noise memorization occurs in standard non-private learning, it will also occur in private learning as long as $\varepsilon \geq \text{SNR}^{1-q}n^{-1}$}: Is this claim empirically justified? Evidence from membership inference and related privacy attacks suggests that differential privacy can mitigate memorization. Moreover, in most realistic settings $\varepsilon \gg n^{-1}$. Please correct me if I am missing some relevant literature. This question also relates to my earlier point about the typical magnitude of SNR in real datasets.

### Questions
Please refer to the weaknesses. I would be happy to raise my score if some of the points were explained.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper considers training a two-layer CNN on a data distribution that disentangles signal and noise, and proves there is a phase change in the training dynamics of Noisy Gradient Descent with respect to the signal-to-noise ratio (SNR). Specifically, while the training error converges, the test error is small when SNR is small, whereas the test error can be large when SNR is large. Compared with non-private gradient descent, Noisy Gradient Descent injects noise during training process and requires stronger SNR to generalize.

### Strengths
There is limited exploration of private learning beyond convex settings, and this work investigates one non-convex setting, showing a privacy cost: training privately requires a higher SNR to generalize than non-private learning. 

The paper provides explicit bounds on training and test error for this setting, which might be an interesting result.

### Weaknesses
* The setting and analysis are very similar to prior work (Cao et al., 2022).

* The sensitivity of the gradient step appears to depend on the data noise $\xi$, which is data-dependent, so the privacy guarantee is unclear.

### Questions
* The distributional assumption requires a specific decomposition of signal and noise. Do the same results hold when $x_2$ is independent of $x_1$ (without the orthogonality/projection structure using $v$)? It also requires $x_1$ to be a linear function of $y$ and do the results extend when $x_1 = f(y)$ for some nonlinear function $f$?

* For the same data distribution, does the SNR-based separation between non-private and private generalization also hold for convex models?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
-  This paper investigates differentially private learning from a feature learning perspective. They analyze why gradient perturbation in DP-SGD is affected by the signal-to-noise (SNR) ratio and investigate data noise memorization as a cause of poor generalization.

### Strengths
-	The paper investigates the poor quality of differentially private learning in terms of feature learning, which has not been actively investigated.
-	The authors provide a theoretical analysis of why DP training is largely affected by noise addition relative to the signal.
-	Based on their observations, the authors argue that a stronger feature signal is needed for private learning and that data noise may cause data noise memorization.

### Weaknesses
Please refer to the Questions section.

### Questions
- The main experimental analysis is based on Figure 2. However, Figure 2 is hard to understand since figures with a similar layout are used repeatedly, even though each figure delivers different results in Section 5. Furthermore, the ‘legend’ is hard to read.

-	Furthermore, can the authors compare the analysis of their results with varying architectures or training methods designed to capture better feature learning, such as [1]?

    [1] Differentially Private Learning Needs Better Features (or Much More Data), ICLR 21

-	The authors mention that experimental results on CIFAR-10 are in the Appendix. However, as far as the reviewer can tell, the reviewer only sees Figure 3 in the Appendix and CAM visualizations. The fact that ‘Higher SNR Improves Accuracy Across Various Privacy Budgets’ is well-known in DPDL, and the CAM visualizations do not seem well aligned with the analysis in the main paper. How does the paper apply its ideas to real-world datasets?

-	How does clipping affect feature learning?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies differentially private learning from a feature-learning perspective.
Using a simplified two-layer CNN model trained with Gaussian-noise gradient updates (NoisyGD), the authors analyze how the signal-to-noise ratio (SNR) governs feature learning and noise memorization. 
They show that private learning requires a higher SNR to capture meaningful signal and provide theoretical insight into why DP training tends to degrade feature quality.
Some synthetic and real-world experiments qualitatively support the theory.

### Strengths
* **Interesting novel perspective on private machine learning**.
You discuss an interesting shift in perspective from utility measures such as risk/accuracy to feature-learning-based analyses.
The analysis of the learning dynamics in terms of SNR is relevant as it more closely captures what the structure of what is learnt is, rather than simply evaluating output accuracy.
Your analysis gives insights on _why_ DP training may degrade feature quality and how this can be counteracted.
The change of focus from "DP hurts accuracy" to "DP changes how features are learnt" is interesting and, I think, can spark future developments in this direction.

* **Technical contributions.**
Within its chosen framework, the paper adapts existing non-private feature-learning analyses to private learning.
The derivations provide useful insights, in particular with respect to noise memorization.
The necessity of higher SNR for private learning is clearly formalized and matches intuition.
In particular, the paper provides a theoretical foundation for the large literature on pre-training and fine-tuning for DP learning, which shows how pre-learnt features can positively impact utility.

* **Paper structure.**
Despite the sometimes heavy notation, the paper is well structured and the flow of arguments is easy to follow.

### Weaknesses
* **Formality of the DP guarantee**.
I have some concerns with respect to the privacy guarantee you provide.
In your analysis, you assume no gradient clipping which, paired with ReLU activation, leads to unbounded $L_2$ sensitivity.
Lemma 4.4 provides a high probability bound on the internal coefficients and, from this, you mention that the gradient sensitivity can therefore be bounded by a term proportional to the noise term (line 285).
However, it remains unclear to me whether the bound on the sensitivity holds only with high probability as well and, regardless, the bound depends on the norm of the noise term which is unbounded and data dependent.
While I do acknowledge that, as you mention in footnote 1, many theoretical investigations do not consider gradient clipping, I feel like further discussions are necessary to be able to formally guarantee DP.
It is not clear whether the result should be interpreted as an analysis of the NoisyGD dynamics, rather than of a provably DP algorithm.
Moreover, it is difficult to follow the $(\epsilon, \delta)$-DP guarantees throughout the proofs.

* **NoisyGD vs DP-SGD.**
In general, I personally find the distinction between NoisyGD and DP-SGD a bit opaque.
If my understanding is correct, the approach you follow is to add Gaussian noise to the full gradient step (Equation 3).
However, DP-SGD generally focuses on per-example, clipped gradients.
Even under the assumption that gradient clipping should be disregarded, further justification/clarification on whether the results only hold for the specific NoisyGD dynamics presented in Equation 3, or for a more general DP-SGD framework, as claimed by the first contribution, are necessary.

* **Empirical evaluation and visualizations**.
The empirical evaluation is limited in scope, as it considers a single dataset and does not discuss possible comparisons with other related work.
Moreover the results in Figure 2 are very difficult to read, due to the legend taking up most of the plot.

* **Assumptions and their intuitive meaning.**
Despite the easy-to-follow structure, I find the exposition to be a bit too dense.
The manuscript could benefit from a more intuitive discussion of how restrictive the assumptions are, and why they matter.
For instance, the use of polynomial ReLUs is not thoroughly justified: this is presented as an improvement over previous work that discusses only linear models, but a further discussion on the benefits and limitations of polynomial ReLUs would be beneficial.

 ### Other comments
* Citations are misused throughout the paper, with in-line citations (`\citet{}`) used instead of parenthetical citations (`\citep{}`).

### Questions
* Do the results presented hold for NoisyGD or for DP-SGD?
* In the proof of Lemma I.2 you choose a specific constant, namely $0.2$. It is unclear to me how this simplifies or otherwise impacts the proof.

### Soundness
2

### Presentation
3

### Contribution
2
