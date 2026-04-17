# Larger Datasets Can Be Repeated More: A Theoretical Analysis of Multi-Epoch Scaling in Linear Regression

- Decision: Accept (Poster)
- Scores: 8, 6, 4, 8

## Abstract
While data scaling laws of large language models (LLMs) have been widely examined in the one-pass regime with massive corpora, their form under limited data and repeated epochs remains largely unexplored. This paper presents a theoretical analysis of how a common workaround, training for multiple epochs on the same dataset, reshapes the data scaling laws in linear regression.
    Concretely, we ask: to match the performance of training on a dataset of size $N$ for $K$ epochs, how much larger must a dataset be if the model is trained for only one pass?
    We quantify this using the $\textit{effective reuse rate}$ of the data, $E(K, N)$, which we define as the multiplicative factor by which the dataset must grow under one-pass training to achieve the same test loss as $K$-epoch training.
    Our analysis precisely characterizes the scaling behavior of $E(K, N)$ for SGD in linear regression under either strong convexity or Zipf-distributed data: (1) When $K$ is small, we prove that $E(K, N) \approx K$, indicating that every new epoch yields a linear gain; (2) As $K$ increases, $E(K, N)$ plateaus at a problem-dependent value that grows with $N$ ($\Theta(\log N)$ for the strongly-convex case), implying that larger datasets can be repeated more times before the marginal benefit vanishes.
    These theoretical findings point out a neglected factor in a recent empirical study by [Muennighoff et al. (2023)](https://arxiv.org/abs/2305.16264), which claimed that training LLMs for up to $4$ epochs results in negligible loss differences compared to using fresh data at each step, $\textit{i.e.}$, $E(K, N) \approx K$ for $K \le 4$ in our notation. 
    Supported by further empirical validation with LLMs,
    our results reveal that the maximum $K$ value for which $E(K, N) \approx K$ in fact depends on the data size and distribution, 
    and underscore the need to explicitly model both factors in future studies of scaling laws with data reuse.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper presents a theoretical framework for understanding multi-epoch data reuse in the context of linear regression and its implications for data-scaling laws in large model training. It shows that larger datasets can be repeated more times effectively. Simulation and LLM pretraining experiments confirm the theory’s predictions.

### Strengths
1. Valuable theoretical insights: The discussion of how larger datasets (N) allow more effective reuse is both novel and practically relevant.
2. Solid theoretical foundation under two regimes: The analysis is carefully constructed for both strongly convex and Zipf-distributed settings.
3. The experiment empirically confirms the theory’s predictions.

### Weaknesses
1. Limited discussion of “depends on the data size and distribution.”
While the paper acknowledges that E(K, N) depends on both dataset size and distribution, the explanation remains mostly theoretical. More quantitative or illustrative examples—especially for large-scale LLM pretraining where data heterogeneity and long-tail effects dominate—would make this claim more convincing.

### Questions
1. Theoretically, the paper suggests that a dataset of size N can be effectively “amplified” by a factor of log N through multi-epoch training.
If that interpretation is correct, why do modern LLMs only have 4 with billions of tokens?
Does this discrepancy imply that current LLMs operate beyond the idealized regime assumed in the theory (e.g., due to non-convexity, heavy-tailed data, or curriculum effects)?

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
2

### Summary
This paper presents a theoretical analysis of multi-epoch training for large-scale models, a common practice when high-quality training data is limited. The authors introduce a metric called the "effective reuse rate," $E(K, N)$, to quantify how many additional "fresh" data samples one-pass training would need to match the performance of training for K epochs on a dataset of size N. Through a detailed analysis of Stochastic Gradient Descent (SGD) in linear regression, they demonstrate two key regimes: 1) for a small number of epochs (K), the benefit is nearly linear (E(K, N) ≈ K), meaning each pass is almost as good as seeing new data; 2) as K increases, the benefit saturates. Crucially, they prove that the saturation point itself grows with the dataset size N (e.g., logarithmically or as a power of N). This central finding—"larger datasets can be repeated more"—challenges previous empirical work that suggested the reuse rate was independent of N. The authors validate this theoretical insight with both synthetic data simulations and pre-training experiments on a large language model.

### Strengths
1. The paper provides a principled and rigorous theoretical framework to analyze the widely used but poorly understood practice of multi-epoch training. The introduction of the "effective reuse rate" E(K, N) is a clear and valuable conceptual contribution. The key finding that the benefit of data reuse scales with the dataset size (N) is a significant insight. It provides a concrete guideline for practitioners: one can and should repeat larger datasets more times before expecting diminishing returns. This directly refutes a simpler assumption from prior empirical work, making a clear and important contribution to the field of scaling laws.

2. While the core theoretical results are derived in the simplified setting of linear regression, the authors do an excellent job of validating their main qualitative finding with an actual LLM pre-training experiment (Section 6.3). This strengthens the paper's claims significantly and shows that the core intuition derived from theory holds in a much more complex, real-world scenario.

3. The paper is well-written and structured.

### Weaknesses
1. The primary limitation is the gap between the theoretical setting (linear regression with SGD) and the practical setting of interest (Transformer-based LLMs trained with AdamW). Linear models cannot capture the complex, non-linear feature learning that occurs in deep networks. While the qualitative findings transfer, the exact quantitative predictions (e.g., the saturation point scaling as Θ(log N)) may not hold for Transformers. This is a standard and often necessary simplification in theoretical work, but it's an important caveat.

2. The theory is developed for Mean Squared Error (MSE) loss, which is standard for regression. However, LLMs are almost universally trained using a cross-entropy loss. These two loss functions have different properties, and it's not immediately obvious if the scaling dynamics would be identical. 

3. While the inclusion of LLM experiments is a major strength, their scope is naturally limited by computational cost. The experiments use a 0.3B parameter model. While this provides strong evidence, it does not definitively prove the same scaling behavior would be observed in much larger, state-of-the-art models, where different phenomena might emerge.

### Questions
n/a

### Soundness
4

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
2

### Summary
This paper challenges prior work that assumed the effective reuse rate of data is independent of dataset size. Through rigorous theoretical analysis in linear regression, the authors demonstrate that larger datasets can be trained for more epochs before experiencing diminishing returns. Specifically, they show that the effective reuse rate E(K,N) depends not only on the number of epochs K, but critically on the dataset size N, which is a factor overlooked in previous empirical scaling laws.

### Strengths
1. The paper presents rigorous theoretical analysis with precise characterizations of the effective reuse rate E(K,N) under both strongly convex and Zipf-distributed settings.
2. The central insight that larger datasets can be repeated more times is clearly articulated and challenges existing assumptions in the field.
3. The theoretical predictions are thoroughly validated through two complementary approaches: controlled simulations on synthetic data and large-scale LLM pretraining experiments (up to 200B tokens), both of which strongly support the main hypothesis.

### Weaknesses
1. The overall conclusions are very similar to the previous work "Improved scaling laws in linear regression via data reuse". 
2. And the paper still lacks sufficient practical evidence from LLMs. It is well established that LLM performance differs significantly between large and small models. A more meaningful experiment would be to scale across different model sizes and examine how the effective reuse rate varies with model capacity.

### Questions
1. Could the authors provide a clearer distinction between this work and prior theoretical studies, especially Lin et al. (2025)? While the paper mentions providing "o(1) relative error" versus "Θ(K)" bounds, it would be helpful to understand what new insights or capabilities this precision enables.
2. Could the authors clarify the practical utility of these theoretical findings? Specifically, how should practitioners use the E(K,N) ≈ log(N) saturation result to inform training decisions, given that most modern LLMs train for fewer than 5 epochs?

### Soundness
3

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
This paper studies the question: how large of a dataset is required for one-pass training to match the loss of a dataset of size N trained for K epochs?

They theoretically characterize the scaling behavior for SGD in linear regression in two settings: strong convexity and Zipf-distributed data. In each settings, there are two phases, one phase where K is small and data can be repeated without harm to the performance, and one where K is large and reused data plateaus in usefulness. The point where this phase transition occurs depends on the setting (strongly convex vs Zipf-distributed data) and the data distribution.

In contrast to recent empirical work, their analysis supports a functional form where the number of times you can repeat the dataset grows with the size of the dataset. In other words, the practical takeaway is that larger datasets can be repeated more.

They perform LLM pretraining experiments where they take different size datasets, train them for 100 epochs, extract the loss after varying numbers of epochs, and compare to a 200B dataset trained for one epoch. The experiments validate the small K regime where data reuse doesn't hurt performance significantly, and that the larger datasets can be repeated more.

### Strengths
This is a very nice paper. The core question in the paper is important and well-framed and finding an interesting but tractable theoretical analysis is a valuable contribution. Solving the linear regression problem in both the strongly convex and Zipf distribution settings is valuable and illustrated the dependence on the data distribution exponent. The proof sketch gave nice intuition about the approach and which techniques were used to bound which terms. The LLM experiments give useful validation of the key takeaways (illustrating the small K regime where data reuse is not harmful, and showing that the effective reuse ratio increases with the dataset size).

### Weaknesses
All of the LLM experiments use a constant learning rate schedule with AdamW, rather than some form of learning rate decay (e.g. cosine) as is required for competitive performance in practice. This is a reasonable limitation of a primarily theoretical paper as using a time-horizon-dependent learning rate schedule would require training separate models for every different number of epochs, requiring substantially more compute.

(Similarly, they use the same peak learning rate for all the training runs, and this should likely be tuned for each dataset size and number of epochs, but again this would require substantial compute.)

In particular, there may be an interaction between the learning rate decay and the bias-variance decomposition (i.e. the learning rate decay at the end of training reduces the gradient noise and "reveals" how much the model learned from the repeated data).

To capture the effects of learning rate decay without requiring significantly more compute, one approach would be to load the existing checkpoints (perhaps from a small number of steps before the end of training), then perform linear learning rate decay to zero across a small number of steps. This would produce a "trapezoidal" learning rate schedule for each setting without needing to train a model from scratch for each distinct number of epochs. Then the final decayed losses could be plotted / analyzed as is already done in Figure 2.

### Questions
N/A

### Soundness
3

### Presentation
4

### Contribution
4
