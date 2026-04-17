# Aligning Inductive Bias for Data-Efficient Generalization in State Space Models

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 6, 2

## Abstract
Empty

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work introduces a framework for enhancing the generalization of state-space models via Task-Dependent Initialization. The methods is  motivated by the theory of RKHS induced by the convolutional kernel of SSMs. Experiments show that the method improves the test error when training data is limited.

### Strengths
* The proposed method is based on principled analysis via a RKHS. The theory is intuitive and the derivations seem accurate. Experiments on synthetic datasets well-align with the theory.
* The task studied in the paper is important since high-quality data is limited.
* Most experimental designs are fair (up to a couple of questions outlined in the questions section), and figures are well-made.

### Weaknesses
* There is room for improvements in the presentation of the paper. In particular, the explanation of "why the proposed objective helps" is scattered around. As far as I can tell, the most important intuition is to have the kernel of SSM capture most energy in the kernel of data, but this is only loosely stated in several places without too much emphasis. To improve, Lemma 3.5 and Theorem 3.7 need to be made more precise, especially the "thus improving the generalization ability" part, which is currently hand-wavy while being arguably *the* most important statement of the paper.
* Some assumptions are still too restrictive, e.g., the assumption that training data come from an i.i.d. distribution, and the sampling error $\boldsymbol{\epsilon}^\mu$ follows the zero-mean and uniform-covariance assumption. I also have confusion regarding the role of $\overline{f}$ and $\boldsymbol{\epsilon}$ in the proposed method. Please see questions below.
* While the empirical validation on the long-range arena datasets shows some promise, the results are not strong enough to show the effectiveness of the method. For example,
   * LRA is already considered a limited-data regime. The benefits of TDI emerge only when the data is further significantly limited, which limits the use cases of the method.
   * Only benefits are shown on two tasks, and results on a few tasks in LRA (retrieval and text) are alway missing.
   * I also have some further questions regarding the design of experiments. See below.

### Questions
1. In section 2.3, should I interpret $\overline{f}$ as the ground truth and $\boldsymbol{\epsilon}$ as measurement noises? If so, there are two follow-up questions:
   1. In a classification task, the function $\overline{f}$ is not necessarily well-defined. How would you reconcile this scenario, and how does this impact the further analysis in this paper?
   2. To help "generalization," one should aim to avoid overfitting of the measurement noises $\boldsymbol{\epsilon}$. I don't see this being discussed anywhere else in the paper. Is it hidden somewhere in the theory of section 3?
2. The discussion in section 3 assumes that $L$ is fixed. A key thing about sequence models is that the model should handle sequences of varying lengths. How does the theory work in that case?
3. How is the loss in (4) optimized and how efficient is it?
4. In Figure 5a, why does the TDI method work worse than the default method when no training data is removed (ratio = 1)?
5. In Section 4.2, you mentioned that only the first-layer initialization is changed. Why did you make this choice, and what if other layers are also initialized using TDI?
6. Not critical but just out of curiosity: how easy is it to adapt the TDI method to Mamba, if you have any idea?
7. Comment: $E_g = \mathbb{E}_{\mathcal{D}} [E_g(\mathcal{D})]$ is a terrible recursive notation. Why don't you pull the definition of $E_g(\mathcal{D})$ directly into it? Also, it seems like you never use this notation again in your paper. It would be better not to introduce notations that you will not use to avoid overloading.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies linear time-invariant (LTI) State Space Models (SSMs) using the framework of kernel regression. It introduces a novel SSM-induced kernel to formalize the model’s inductive bias, proving that its spectral decomposition is governed by the SSM’s frequency response (Theorem 3.3). This reveals that SSMs exhibit a frequency-dependent spectral bias, where modes with stronger frequency response are learned faster. Building on this, the authors propose Task-Dependent Initialization (TDI), a fast method that uses power spectrum matching to optimize initial SSM parameters before training, minimizing a loss between the model's and task's power spectra.
Empirical validation on both synthetic data and real-world data are provided to validate the theoretical result. 
It is claimed that TDI enhances generalization and sample efficiency on real-world benchmarks.

### Strengths
* It is novel to study SSM using the kernel regression framework. The SSM-induced kernel is new and interesting. Moreover, a connection between the spetra of the kernel and the SSM frequency response is made theoretically.

* Empirical results are given to validate the theory.

### Weaknesses
* The connection between the SSM-induced kernel and kernel regression framework is not made clearly. The introduction of the problem settings is unclear: what is the input ($x$ or $u$), what is output and what is the target function? The output seems to be 1-d at Section 2.1, $L$-dimensional in Eq.(1) and $d$-dimensional in Line 149. It is very confusing.

* The empirical results are not convincing enough, particularly regarding TDI. While Figure 5 seems to show some advantage of TDI in some cases, the full results in Section B actually show that the reported generalization errors are somehow meaningless: the improvement only happens when the error is even larger than that at initialization. See, for example, Figure B.9. I believe the further results should be provided.

* THe proof is suspicious. Theorem 3.4 claims the equality $s_\rho = |H(\omega_\rho)|$, but from the proof in Section E we only have  asymptotically equivalence. I did not check other details.

### Questions
1. How is SSM-induced kernel related to the kernel regression framework?

2. Can you explain in detail the experiment setting of Figure 2?

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper develops a theoretical framework for improving the data efficiency of linear time-invariant State Space Models. The authors formalize an SSM-induced kernel, showing that its eigen-spectrum acts as the model’s inductive bias and is governed by the SSM’s frequency response. They also propose Task-Dependent Initialization (TDI) and a power-spectrum matching loss to align the model’s inductive bias with the task’s spectral characteristics before training. They showed in experiments that TDI accelerates cumulative power growth and improves generalization, demonstrating a principled way toward data-efficient initialization of SSMs.

### Strengths
1. The paper builds a clear and solid framework for understanding inductive bias in SSMs through an SSM-induced kernel perspective.
2. Focusing on data efficiency is timely and meaningful, given increasing practical constraints on access to large, high-quality datasets.

### Weaknesses
1. The experimental impact seems limited. In Figure 5, the data efficiency gains from TDI are not very consistent across training data ratio, with notable improvement only on Pathfinder.
2. The evaluation scope is narrow: TDI is applied only to the first S4 layer on a small set of benchmarks, with no comparison to other initialization strategies. It’s hard to evaluate the scalability and general applicability of the proposed method.

### Questions
1. Since minimizing the loss can be also viewed as a very efficient pretraining phase, what is the actual computational overhead in practice?
2. Given that the model is essentially multi-layer with nonlinearity and the propagation of frequency structure across layers is not clear, why is aligning only the first-layer frequency response expected to yield global improvement? Is there any analysis of the first-layer representations in the experiments showing that TDI produces a more task-relevant feature space to support the claimed mechanism?
3. The cross-spectrum estimate relies on sufficient data to stabilize. How sensitive is TDI to dataset size, especially in very low-data regimes?

### Soundness
3

### Presentation
3

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
The paper studies the inductive bias of SSMs via an SSM-induced kernel, connecting the kernel's spectrum to the model’s frequency response, and proposes Task-Dependent Initialization (TDI): pre-align the first SSM layer’s spectrum to a task’s cross-power spectrum before standard training. The suggested method leads to lower test loss on several tasks in low data regimes.

### Strengths
The paper provides a clear lens for studying the bias of SSMs through the eigendecomposition of the reproducing kernel.
The theoretical analysis is used to derive a simple algorithm for initializing the first layer of an SSM with empirical results that support the papers claim to some extent - yielding lower test loss in a low data regime on some tasks.

### Weaknesses
The analysis in sections 2 & 3 seem to utilize the properties of a general toeplitz matrix and not one that is induced by SSM kernel, which is a specific parameterization of a convolution kernel - if some of the assumptions made to derive the main result relies on the specific structure of SSMs it is not clear to me from the text.
While I acknowledge that studying the theoretical inductive bias of architectures is important, modern practice typically involve a self-supervision phase that aligns the model parameters with the data structure, thus making the results less impactful. See for example [1] also showing gains in low data regimes by including self supervision.

[1] Never Train from Scratch: Fair Comparison of Long-Sequence Models Requires Data-Driven Priors

### Questions
1. Is the analysis in sections 2 & 3 restricted to SSMs alone and utilize their specific parameterization of the toeplitz matrix or does it apply to any convolution kernel?
2. Can you include the accuracy on various tasks in section 4.2

### Soundness
3

### Presentation
3

### Contribution
2
