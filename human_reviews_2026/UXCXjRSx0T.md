# Speculative Sampling for Parametric Temporal Point Processes

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 6, 6, 4

## Abstract
Temporal point processes are powerful generative models for event sequences that capture complex dependencies in time-series data. They are commonly specified using autoregressive models that learn the distribution of the next event from the previous events. This makes sampling inherently sequential, limiting efficiency. In this paper, we propose a novel algorithm based on rejection sampling that enables exact sampling of multiple future values from existing TPP models, in parallel, and without requiring any architectural changes or retraining.
Besides theoretical guarantees, our method demonstrates empirical speedups on real-world datasets, bridging the gap between expressive modeling and efficient parallel generation for large-scale TPP applications.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a novel algorithm based on rejection sampling that enables exact sampling of multiple future values from existing TPP models, in parallel, and without requiring any architectural changes or retraining. The authors provided theoretical guarantees, which makes the algorithm solid. Experiments demonstrate the good performance and potential application.

### Strengths
1. The paper is well-organized and the writing is clear.
2. The paper presents a parallel sampling method that accelerates TPP sampling without requiring architectural changes or retraining. It is a novel adaptation of the "speculative sampling" paradigm from language modeling to TPPs.
3. A key strength is the rigorous theoretical foundation, complete with derived rejection constants for distributions like Exponential, Gamma, Log-Normal, and Weibull, as well as a proven error bound that strengthens confidence in the method.

### Weaknesses
1. The algorithm's reliance on computing the rejection constant M via piecewise-linear approximation raises concerns. Key issues include whether the grid is sufficiently dense and if the approximation error for multi-modal distributions could lead to an underestimated M, thereby violating the guarantee of exact sampling.
2. Relying on the first rejected sample to terminate a speculative block can reduce parallelism, particularly for high-variability processes where early rejection is likely.

### Questions
1. The piecewise-linear approximation for estimating the rejection constant M in complex distributions raises several concerns. The tightness of this approximation critically depends on the selection and number of grid points, yet the paper lacks a principled method for this choice (e.g., based on density curvature). An overly loose bound (large M) inflates the rejection rate, hurting efficiency, while an overly tight one risks violating the exact sampling guarantee. It is also unclear if this method remains computationally feasible for highly complex or high-dimensional distributions.
2. A major limitation is the algorithm's dependence on the average accepted length for efficiency. For processes with strong temporal dependencies, the overhead from parallel verification and M computation may not be amortized, leading to performance worse than standard sampling. The paper would be strengthened by exploring more robust strategies for these cases beyond top-k, such as backtracking or partial re-sampling mechanisms.
3. Is the method particularly sensitive to the calibration quality of the underlying model? Have the authors investigated the correlation between model prediction uncertainty and inferred sampling efficiency?
4. Could the authors provide a detailed performance analysis showing what percentage of the total sampling time is spent computing the rejection constant in a typical experimental setup? Is there a critical point below which the total cost of the algorithm exceeds that of traditional autoregressive sampling?
5. Why not compare your approach to a model trained directly for multi-step forecasting? For example HYPRO [1] and Add-and-Thin [2]. Even if the latter has slightly lower sample quality, it can be sampled very quickly. This "quality-speed" trade-off is worth exploring.
6. For datasets with strong, near-deterministic transition rules like Taxi, is it possible to modify the proposal distribution to better capture this structure? Is there any empirical experiment results?

[1] HYPRO: A Hybridly Normalized Probabilistic Model for Long-Horizon Prediction of Event Sequences. NeurIPS 2022

[2] Add and thin: Diffusion for temporal point processes. NeurIPS 2023

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper addresses the efficiency bottleneck of sampling from autoregressive temporal point processes (TPPs). The authors propose a novel speculative sampling algorithm based on rejection sampling that enables efficient generation of multiple future events. The approach can be applied to existing parametric TPP models without requiring any architectural changes or retraining. The core idea is to use a proposal distribution to generate several candidate events at once. These proposed events are then fed back into the model in a single parallel pass to compute their target (true) distribution and decide whether each sample is rejected or accepted. The method is supported by a principled theoretical foundation for computing the necessary rejection constant. Experiments on real-world datasets confirm that this approach provides substantial runtime speedups while maintaining the exact statistical properties of the original model.

### Strengths
1. The work is well-motivated, addressing the critical bottleneck of inefficient sampling in autoregressive Temporal Point Processes (TPPs). Addresses this major limitation could have significant implication for real-time applications.

2. The paper's claims are supported by strong empirical results. The experiments demonstrate substantial efficiency improvements in wall-clock time compared to conventional sequential sampling, as presented in Table 3. Furthermore, the authors validate their claim of exact sampling by showing that the generated samples are statistically indistinguishable from the samples obtained with traditional methods, using metrics like MMD on several benchmark datasets.

3. The proposed sampling algorithm is principled and rests on a solid theoretical foundation. The core of the method is a general technique for computing the necessary rejection constant by constructing piecewise linear upper and lower bounds for the target and proposal densities. The authors also introduce theoretical results on finding rejection constant efficiently.

4. A significant strength of this work is its generality. The proposed algorithm is model-agnostic and can be used as a drop-in addition to many existing parametric TPP models. The paper's theoretical framework is designed to handle a wide variety of distributions.

### Weaknesses
1. A significant limitation of this approach is its reliance on a closed-form, parametric density function. Computing the rejection constant, particularly the general method in Section 3.2, requires the ability to evaluate both the target density and its derivative to find inflection points and construct the necessary linear bounds. This constraint makes the approach unapplicable to many intensity-based or diffusion-based TPP models. 
2. The paper's claim in Section 3.4 that the hidden states $h_{I+j}$ for the proposed events can be processed "in parallel" is questionable and highly model-dependent. If a model generates $h_{I+j}$ autoregressively, they cannot be processed in parallel.
3. It seems that the efficiency is also data-dependent. As the author acknowledges, the Taxi dataset, due to its dataset-specific properties, leads to a large rejection rate and inferior efficiency improvement relative to other datasets.

### Questions
Please see the weakness section.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses the inefficiency of sequential sampling in autoregressive Temporal Point Processes (TPPs)—a critical limitation for real-time applications like finance and social network event modeling—by proposing a novel speculative sampling algorithm based on rejection sampling. TPPs model irregular event sequences (e.g., earthquakes, limit order book messages) but require sequential generation of events, which bottlenecks large-scale applications. The proposed method enables parallel sampling of multiple future events from existing parametric TPP models (no retraining/architectural changes) while guaranteeing exact sampling.

Key Contents.

1. The algorithm uses a pre-trained TPP encoder’s next-event distribution as a proposal to generate multiple future events in parallel. It accepts events until the first divergence between proposal and target distributions (derived by reprocessing proposed events with the encoder), ensuring exact sampling.

2. A universal method to compute rejection constants (critical for rejection sampling) via piecewise linear bounds on CDFs/PDFs.

3. Rigorous proofs confirm exact sampling (via valid rejection sampling) and bounded total variation error (≤δ) for categorical distributions with truncated constants, ensuring statistical correctness.

4. Experiments on 7 real datasets (e.g., Amazon, Earthquake, limit order books) show significant speedups (e.g., ~3.2x for Retweet data) with negligible quality loss (MMD/KL-divergence close to baseline). A financial application (limit order books) achieves average speculative steps of 2.82, validating practical utility.

5. Works with diverse encoders (GRU, Transformer, CNN) and decoders (log-normal mixture, exponential), making it compatible with most existing parametric TPPs.

### Strengths
The paper is easy to follow, and its main idea is clear and straightforward. The paper demonstrates notable originality by addressing a core inefficiency in TPP sampling—sequential generation bottlenecks—through a novel speculative sampling framework that avoids modifying existing models. Unlike prior works (e.g., Gloeckle et al., 2024; Zeng et al., 2023) that require retraining TPPs to predict multiple steps, this method leverages pre-trained encoders’ proposal distributions to generate parallel future events, with acceptance based on target-proposal divergence . A key innovation is the universal rejection constant calculation: using piecewise linear bounds on densities (exploiting convex/concave regions) to handle common TPP distributions (exponential, Gamma) and mixtures—an approach distinct from envelope methods (Gilks & Wild, 1992) that only apply to log-concave cases . This fills a gap for non-retrainable, high-frequency TPP applications (e.g., limit order books) where efficiency and model compatibility are critical.

In addition, the experimental results are substantial and adequate.

### Weaknesses
The main weakness of this paper is its presentation. Some parts are not clear enough. See my comments in "Questions" section.

### Questions
1. In Section 3.4, the renewal sampling could be made more clearly. For example, the authors can add more descriptions or formula to explain the generation from $(\tau_{i+j}, x_{i+j})$ to $(\tau_{i+j+1}, x_{i+j+1})$.

2. In "sample acceptance" of Section 3.4, the author should make it mathematically clear. What is the formula of the target distribution $p^*$ and the formula of the proposal $p$.

3. In Table 1, what does it mean by "commonly used distribution" in TPP models? Does it refer to the distribution for modeling the gap time $ \tau_{i+1} - \tau_i$?

4. In the experimental section, what does "sampling quality" mean? Could author give out the exact mathematical definition of the "quality"?

5. Could the author give some intuitive explanations for why top-k does not affect the sampling quality?

6. In the introduction of TPP, it is more formal to define $\mathcal H_i$ by the filtration of all past events.

### Soundness
3

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
2

### Summary
This paper proposes a speculative sampling scheme for temporal point processes (TPPs), inspired by speculative decoding methods used in large language models. The method aims to accelerate sampling from autoregressive TPP models by generating multiple future events in parallel using rejection sampling, without retraining or architectural modifications. The authors derive rejection constants for common distributions, provide theoretical proofs for bounding the acceptance rate, and report empirical speedups on several datasets.

### Strengths
1. The paper identifies a relevant problem — the inefficiency of sequential sampling in autoregressive TPPs — and attempts to address it using ideas from speculative decoding.

2. The proposed algorithm is conceptually simple and can be implemented without retraining existing models.

3. Theoretical analysis (Section 3) is relatively clear, providing proofs for the bounding procedure.

### Weaknesses
1. Limited Novelty and Conceptual Contribution:

1)The proposed approach is a straightforward adaptation of speculative decoding to temporal point processes. There is no fundamental theoretical or algorithmic innovation beyond applying rejection sampling to a different data modality.

2) The main “novelty” (deriving rejection constants for specific distributions) is technical but not conceptual — such derivations follow standard bounding techniques from rejection sampling literature.

2. Theoretical Claims Are Overstated:

1) The claim of “exact sampling” is questionable: the acceptance rule relies on upper/lower bounds that are themselves approximations (piecewise linear). The proofs (Appendix A) ensure boundedness, not exact equivalence to the target process.

2) The treatment of mixture distributions is hand-wavy: the proof assumes shared grid convexity and linearity, which can break under multimodal densities (e.g., log-normal mixtures).

3) The relationship between the proposed “speculative” method and standard multi-step Monte Carlo sampling is not rigorously differentiated.

### Questions
1. Is the rejection constant estimation stable across distributions with heavy tails (e.g., log-normal with large σ)?

2. Could this approach be extended to continuous-time diffusion TPPs

### Soundness
2

### Presentation
2

### Contribution
2
