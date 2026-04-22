# Data-Efficient Training by Evolved Sampling

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 4, 6, 2

## Abstract
Data selection is designed to accelerate learning with preserved performance. To achieve this, a fundamental thought is to identify informative data samples with significant contributions to the training. In this work, we propose **Evolved Sampling** (**ES**), a simple yet effective framework for *dynamic* sampling along the training process. This method conducts *batch* level data selection based on the dynamics of losses and augmented *loss differences*, which enables flexible *frequency tuning*, and hence significantly reduces the back propagation time with maintained model performance. Due to its conciseness, ES is also readily extensible to incorporate *set* level data selection (to form ES with pruning, **ESWP**) for further accelerations. As a plug-and-play framework, ES(WP) consistently achieves lossless training accelerations across various pre-training and post-training tasks, saving up to nearly 45\% wall-clock time. Our results motivate further investigations on the data efficiency aspect of modern large-scale machine learning.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes to identify informative data samples in order to reduce back propagation time. The proposed method Evolved Sampling (EV), as well as ES with Pruning (ESWP) determines the importance/weights of data samples based on both (zero-order) losses and additional (first-order) loss differences along the training dynamics. Results show lossless training, saving up to 45% wall-clock time.

### Strengths
The paper is original in combining batch-level and set-level sampling.

Results are extensive, with gains in low-resource settings and larger acceleration in LLM fine-tuning beyond CIFAR-10/100.

A Fourier-based analysis explains the observed benefits.

### Weaknesses
Lack of clarity: see Questions on which points are not clear

No variance: it is not clear how the measurements vary and how statistically significant are the results.

Profiling: paper could benefit from supporting claims on computational and memory efficiency in the main text with numbers.

For other Questions please refer to the Questions.

### Questions
1. Please clarify whether the reported results are means over multiple seeds or single runs; in close comparisons, multi-seed averages can be lower and differences may lack statistical significance relative to the baselines.

2. In Table 1, shouldn’t the percentage of samples for BP rather than using the “#” symbol, which suggests a raw count and is confusing?

3. Is convexity a reasonable assumption in the proof, and can the result be established under milder conditions?

4. Please elaborate on computational efficiency at line 278; the claim that overhead is modest because FP uses far fewer FLOPs than BP needs exact FLOP counts and wall-clock timings, since BP is typically about twice the cost of FP and not negligible.

5. For the memory efficiency claim after Eq. 3.1, please provide concrete numbers showing that storing per-sample scores or weights is negligible relative to the data tensor footprint.

6. Since the beta parameters require a grid search, please report the end-to-end wall-clock time to select beta and then train in a practical scenario, or should one use (0.2, 0.8) across datasets every time?

7. Do the beta values interact with other hyperparameters such as the learning rate, or can standard hyperparameters be reused without retuning?

8. Could you provide intuition for why ESWP outperforms ES in Table 3?

9. At line 375, please specify which components are distributed and how.

10. For Table 8, describe how Random is implemented, how the learning rate is handled under Random sampling, and whether the randomness is static or dynamic across iterations.
Please also explain how methods that add selection overhead can surpass Random in efficiency, given that Random should incur the least training overhead.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes "Evolved Sampling (ES)", a dynamic sampling method that employs an exponential moving average scheme to select training samples based on loss values during training. It reduces the number of samples required for backpropagation and shortens the necessary training time.

### Strengths
1. The paper provides a comprehensive set of experiments covering small-scale (CIFAR), large-scale (ImageNet, ViT-Large), NLP (GLUE), and even large language model fine-tuning tasks. This demonstrates the robustness and general applicability of the method.

2. Including thorough ablation studies (e.g., the importance of loss differences, annealing, pruning strategies, and hyperparameter sensitivity ).

### Weaknesses
​1. The paper lacks a systematic comparison with a random sampling baseline. For example, in experimental results such as Table 2 and Table 3, a baseline that purely randomly selects data subsets is absent. 

2. ​Limited novelty, the core innovation of the method essentially introduces an "exponential moving average with historical score" mechanism into dynamic sampling. 


3.  The weight update formula (Equation 3.1) is somewhat heuristic, primarily relying on an exponential moving average mechanism. It lacks a solid theoretical derivation.

4. The assumption in Proposition 2 that "there exists $\theta^{\star}$ such that  $\hat{L}_n(\theta^{*}) = 0$... "  is too strong for me.

### Questions
1. Why is the specific formulation of Eq. (3.1), which incorporates the difference between the current loss and the historical trajectory (as shown in Proposition 3.1), more beneficial than simply using the smoothed historical score si​(t) for weighting?  Can you provide a more detailed comparison, including a simple hyperparameter search for $s_i(t)$， to illustrate the necessity of your method？

2. Even if the stability of sampling is resolved, why is the sampling scheme in Eq. (3.1) helpful for training？

3. How sensitive is the performance of ES(WP) to deviations from the default (β1​,β2​)values on different tasks? Are there any heuristics or correlations between optimal ($\beta_1$, $\beta_2$​)and dataset properties?

4. The validity of Proposition 2.1 under standard training regimes appears questionable. It would be valuable to empirically validate it on synthetic data, and, if feasible, to explore its robustness under relaxed assumptions.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This work proposes to enhance dynamic data pruning by conducting batch-level data selection based on the dynamics of losses and augmented loss differences. It demonstrates an improved lossless saving ratio on the Transformer architectures.

### Strengths
1. The theoretical motivation is reasonable and meaningful. Turning it into a pruning probability captured the key benefits of dynamic data pruning.
2. The method design, probably inspired by the moving average in optimizers, is quite clever and efficient. It captures the losses’ dynamical differences without introducing any memory space dependence on epoch number. Compared to the uncertainty term in UCB, this design is more reasonable and is shown to be effective.
3. On Transformer architectures, the proposed method demonstrates improved data efficiency.
4. This work extends evaluation to more language model tasks and demonstrates the effectiveness.

### Weaknesses
1. The Tab.2 baseline performance of ResNet on the CIFAR datasets seems to be lower than generally reported. So does the saving ratio of the baseline method InfoBatch.
2. As there is no claim on code availability, the current experiment details could be insufficient. See questions.

### Questions
1. For Tab.3 finetuning ViT-Large on ImageNet-1K, which pretraining dataset does ViT-Large use?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a dynamic data sampling framework called Evolved Sampling (ES) and its pruned version (ESWP). The goal is to accelerate model training through data selection while preserving model performance. The core of the method is to evaluate data sample importance based on the dynamics of both (zero-order) losses and (first-order) loss differences.

### Strengths
1. The primary strength of this paper is its extensive experimental validation.
2. "Lossless" Acceleration:  these speed-ups are achieved without a drop in model performance.
3. As a loss-based method, the framework is "much more light-weighted" than gradient-based sampling approaches.

### Weaknesses
1. Proposition 2.1 analyzes loss-weighted gradient descent, which corresponds to the baseline method "Loss" (Eq. 2.3), not the core ES algorithm proposed by the authors (Eq. 3.1).
2. The Fourier analysis in Theorem 3.2 merely concludes that the method dampens high-frequency oscillations. This is an intuitive and well-known property of using an exponential moving average (EMA), which is what the s(t) term represents. This is not a deep insight and does not prove why this specific form of smoothing leads to training acceleration.
3. the paper provides no convergence guarantees for its core proposed algorithm.
4. Limited Novelty: The core idea of the paper is to use a weighted average of the current loss and a smoothed version (EMA) of historical loss to determine sampling weights. As noted in the paper's introduction, dynamic sampling methods based on current losses and historical losses  are both well-studied.
5. The motivation of the paper is overly embellished. The core rationale — namely, the need for stability to counteract loss oscillations — naturally leads to EMA, which is already a standard smoothing technique. However, I do not see sufficient justification for the claimed "introduction of first-order loss differences". Equation (3.1) can essentially be viewed as a simple variant of EMA, while Proposition 3.1 mainly concludes that EMA is equivalent to a first-order loss difference. This logical flow is confusing and somewhat circular.
6. The method introduces two key new hyperparameters, betas, in addition to the pruning ratio r and annealing ratios for ESWP. This presents a significant tuning burden for practitioners, making the "plug-and-play" claim difficult to justify.

### Questions
See Weaknesses.

### Soundness
3

### Presentation
2

### Contribution
2
