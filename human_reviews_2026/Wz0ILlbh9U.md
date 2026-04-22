# Temporal Generalization: A Reality Check

- Avg Score: 7.00
- Decision: Accept (Poster)
- Scores: 8, 8, 6, 6

## Abstract
Machine learning (ML) models often struggle to maintain performance under distribution shifts, leading to inaccurate predictions on unseen future data. In this work, we investigate whether and under what conditions models can achieve such a generalization when relying solely on past data. We explore two primary approaches: convex combinations of past model parameters (parameter interpolation) and explicit extrapolation beyond the convex hull of past parameters (parameter extrapolation). We benchmark several methods within these categories on a diverse set of temporal tasks, including language modeling, news summarization, news tag prediction, academic paper categorization, satellite image-based land use classification over time, and historical yearbook photo gender prediction. Our empirical findings show that none of the evaluated methods consistently outperforms the simple baseline of using the latest available model parameters in all scenarios. In the absence of access to future data or robust assumptions about the underlying data-generating process, these results underscore the inherent difficulties of generalizing and extrapolating to future data and warrant caution when evaluating claims of such generalization.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper deals with the issue of predictions on future data based on past checkpoints, presenting comprehensive benchmarking on various tasks (language modeling on evolving news corpora, news summarization and tag prediction, categorization
of academic papers, land use classification from satellite imagery reflecting changes over several
years, and gender prediction from historical yearbook photos ). Existing benchmarking studies are hampered by unrealistic evaluation settings as they e.g. assume access to future data.
The core hypothesis is that approaches can be broken down into two phenomena: interpolation vs. extrapolation. They adopt a sequential fine-tuning approach.

Their core findings are:
1) No method consistently improves performance over the recent model.
2) Taylor-Series extrapolation underperforms compared to other methods.
3) Continual learning is important.

### Strengths
The experiments presented in the paper are extensive and are clearly guided by research questions, which are answered and discussed well.

### Weaknesses
There are minor parts of the presentation that could be improved, e.g. spell-checking, using the same font for all figures and the same style for all figure captions. Table 1 is also positioned strangely.

### Questions
N/A

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This study asks whether ML models can truly generalize to future, unseen data when you only have access to past model checkpoints—no peeking at future data. The authors benchmark two families of methods that operate purely on historical parameters: (i) parameter interpolation (averaging/EMA and a simple downscaling of the latest model), and (ii) parameter extrapolation (first-order, Taylor-style forecasting of weights). Across language modeling and summarization on monthly news, plus yearly tasks like Yearbook, HuffPost, arXiv, and FMoW, they find no approach reliably beats simply deploying the most recent model; notably, downscaling the latest weights is the only method that doesn’t consistently degrade performance.

### Strengths
The paper introduces a highly realistic experimental framework that mirrors real-world deployment, strictly prohibiting the use of future data.

It delivers a comprehensive empirical evaluation, testing multiple architectures and datasets with different temporal frequencies and scales.

The authors offer transparent and reproducible experimentation, including open-source code and detailed implementation settings.

The study provides valuable negative evidence, demonstrating that intuitive weight-based strategies can fail dramatically, which grounds future work in reality rather than optimism.

Beyond reporting results, it also diagnoses the causes—linking observed failures to theoretical properties such as non-identifiability and parameter-space discontinuity.

### Weaknesses
The investigation remains restricted to parameter-space manipulation, leaving out more adaptive or data-driven temporal methods that could exploit unlabeled future samples.

Some evaluated datasets exhibit limited temporal granularity, reducing the precision with which the authors can analyze temporal drift.

The reliance on historical validation only makes hyperparameter selection brittle and may underestimate the potential of some methods.

While empirically thorough, the paper’s methodological contribution is modest, focusing on benchmarking rather than innovation.

### Questions
N/A

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper studies temporal generalization — the ability of models trained on past data (and only past data) to perform well on future, potentially shifted data. The authors point out that many recent papers implicitly assume that, given a sequence of checkpoints, one can interpolate or extrapolate those weights to obtain parameters that work better on future data, even when the future distribution is unknown. The central claim of the paper is negative but important: under a realistic, “no access to the future” setting with streaming / continual updates, none of the evaluated interpolation (uniform / EMA merging, recent model, norm-downscaling) or extrapolation (first-order Taylor on parameter trajectory) methods consistently beats the trivial baseline of “just use the latest model. The only method that consistently avoids hurting performance is a simple downscaling of the latest model (interpolating toward the origin), which the authors motivate via parameter-norm growth and sharper minima over time.

### Strengths
1. The paper is very explicit about the setting: streaming / continual fine-tuning, no access to future unlabeled or labeled data, and hyperparameters must be chosen from the past. This closes an often-overlooked loophole in earlier temporal/DG works that quietly validated on future data.
2. The experiments span: (i) monthly news language modeling and summarization with T5 (showing clear temporal decay and the salience of downscaling), and (ii) WILDS-Time datasets (Yearbook, HuffPost, arXiv, FMoW), where no single method dominates. This makes the main negative result much more convincing than if it were shown on one dataset.
3. The paper separates parameter interpolation (merging, recent, downscaling) from parameter extrapolation (Taylor step using the most recent difference).
4. The observation that the optimal extrapolation factor 𝛼 often drifts below 1 or even becomes negative (meaning “don’t extrapolate; sometimes go backward”) is a strong piece of evidence that parameter trajectories of large models are not sufficiently smooth or aligned over time to support simple Taylor-based prediction.
5. The finding that downscaling the latest model is the only option that does not reliably hurt, and that this correlates with steadily growing parameter norms under continual learning, is a small but real contribution that practitioners can immediately try.

### Weaknesses
1. The paper explains why interpolation/extrapolation is hard via non-identifiability, non-convexity, and noisy parameter trajectories, but these are largely qualitative or synthetic-illustration arguments; there is no general impossibility statement of the form “under assumptions X, any future-agnostic checkpoint-only method cannot beat the latest model by more than 𝜖”. A lightweight formalization, even under a stylized model, would sharpen the negative result.
2. The paper itself shows that if you don’t do sequential fine-tuning (i.e. if you reinit from a pretrain every time), all methods degrade much more (Fig. 6). That means the key negative result is partly conditional on the positive result that CL keeps consecutive checkpoints close. This is important, but it also means the conclusions are strongest for CL-like pipelines, weaker for other update regimes.
3. The paper does the right thing — tune 𝛼 only on past data — but for some methods (esp. extrapolation) the chosen 𝛼 fluctuates sharply over time (Fig. 5). In practice, this means users would have to re-tune at every step, which undercuts the appeal of “cheap temporal generalization.” A small ablation on “fixed  𝛼 over long horizons” vs “retuned 𝛼 ” would have made this clearer.

### Questions
1. You show that parameter norms steadily increase under CL and that downscaling helps. Did you try layerwise or modulewise downscaling (e.g. scale only attention / only FFN / only classifier) and, if so, was the benefit still present? Intuitively, temporal brittleness might concentrate in the last layers that overfit to very recent tokens / labels.
2. Can you tell whether the sign flip correlates with which dataset was seen last (e.g. news vs scientific vs vision)? In other words, is the problem that parameter trajectories are globally noisy, or that mixing modalities / tasks breaks linearity?
3. Fig. 6 clearly shows CL helps. But you fine-tune from θt−1 on all datasets. What happens if the per-step compute is reduced (e.g. 10% of the usual steps, or LoRA-style adapters only)? Do interpolation/extrapolation fail immediately because checkpoints become less aligned, or do they degrade more gracefully?

### Soundness
3

### Presentation
3

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
Due to inaccurate prediction on unseen future data by Machine Learning (ML) models, the authors investigate under what conditions models can achieve such a generalization when depending on past data. The authors explore two Primary approaches: Parameter Interpolation and Parameter Extrapolation. Furthermore, several methods were benchmarked on a set of temporal tasks, including language modeling, news summarization, tag prediction, academic paper categorization, and land use classification.

### Strengths
1. It presents a large-scale, systematic evaluation across multiple tasks (language modeling, summarization, paper categorization, land use classification, and photo gender prediction) and datasets (NewsRoom, Wilds-Time, arXiv, HuffPost, FMoW, Yearbook).

2. The paper makes a contribution by showing that none of the proposed or existing methods reliably outperform a simple "most recent model" baseline.
3. Through parameter norm analysis and dimensionality reduction (PCA/UMAP), the paper provides intuitive interpretations of why downscaling helps.
4. The authors emphasize reproducibility and maintaining transparent reporting of hyperparameters and evaluation setups.

### Weaknesses
1. The paper primarily evaluates existing approaches (parameter interpolation, extrapolation, downscaling) rather than introducing a new algorithm or theoretical framework.

### Questions
1. What was the data size for each time stamp when using different datasets for different methods? 
2. While the paper includes an Oracle model in Figures 2, its definition and implementation are not clearly explained in the main text.

### Soundness
3

### Presentation
3

### Contribution
3
