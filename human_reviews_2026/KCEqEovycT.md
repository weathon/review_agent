# Diffusion-LLM Provides Ultra-Long-Term Time Series Forecasting with Probabilistic Alignment

- Avg Score: 3.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 2, 4

## Abstract
Time series forecasting is a fundamental task in machine learning. Recently, Large Language Models (LLMs) have gained attention for this task due to their strong generalization capabilities, particularly in recognizing patterns and performing complex reasoning across diverse data modalities. Apart from having the architecture suitable for long-context learning, LLMs are an interesting option also because of their few-shot and zero-shot transfer learning capability, making it possible to use pretrained frozen LLMs directly for time series forecasting. However, challenges remain in adapting LLMs to multimodal tasks: they often lack a calibrated understanding of probabilistic structure in non-text modalities and struggle with aligning heterogeneous representations. To address these limitations, we propose Diffusion-LLM, a novel framework that integrates a conditional diffusion model into an LLM-based forecasting pipeline. This joint setup enables the model to learn the conditional distribution of future time series trajectories while reinforcing semantic alignment in the shared latent space. We evaluate Diffusion-LLM on six standard long-term forecasting benchmarks, including ETT, Weather, and ECL datasets. Our approach consistently outperforms existing LLM-based baselines, achieving substantial gains in ultra-long-term and few-shot forecasting tasks, while demonstrating the effectiveness of distribution-aware regularization for enhancing the robustness and generalization of time series LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes Diffusion-LLM, which integrates a conditional DDPM into LLM-based time series forecasting to improve ultra-long-term predictions. The DDPM models the conditional distribution $p(z_y|z_x)$ in the shared embedding space and acts as a regularizer during training. Experiments on 6 benchmarks show improvements over TimeLLM in ultra-long-horizon (H=1024-2048) and few-shot settings, but demonstrate inconsistent performance on standard long-term forecasting.

### Strengths
S1. Addresses Under-Explored Problem: Ultra-long-term forecasting is an important yet under-explored problem.
S2. Reasonable Methodological Design: Using conditional DDPM to model the distribution in embedding space is conceptually sound.
S3. Reproducibility: Hyperparameters, dataset details, and anonymous code are provided in the Appendix, and easy to implement on the BasicTS benchmark.

### Weaknesses
**W1. Insufficient Experimental Validation**:

The proposed method was only compared with TimeLLM and TS-specific models before 2024. Comparisons with the competitive/SOTA models of the last two years are essential. Furthermore, except TS-specific models, LM-based models e.g GPT4TS/TimeVLM, and diffusion-based models e.g. Diffusion-TS,  LM+diffusion work e.g. LDM4TS are necessary to compare.

**W2. Limited Novelty**:

The proposed method mainly combines existing components (TimeLLM reprogramming + standard DDPM). The technical contribution is more engineering-oriented than methodologically innovative, lacking theoretical justification for the approach.

**W3. Limited improvements:**

For example, Table 3 shows Diffusion-LLM is worse than PatchTST and DLinear on some dataset. The experiments fail to convincingly demonstrate the method's advantages.

**W4. Presentation Problems:**

There is a lot of white space around the figures and formulas, which gives the impression that the manuscript is under-written.

**W5. Missing Efficiency Study:**

The authors claim their method preserves "efficiency" provide no empirical evidence: no training time, GPU memory, or FLOPs comparison despite claiming minimal overhead. Moreover, DDPM is known to be substantially slower than alternatives like DDIM, yet the paper offers no justification for choosing DDPM. The lack of efficiency analysis and the justification for diffusion/LLM model choices makes the efficiency claims unsubstantiated and raises concerns about practical applicability.

**W6. Results on ILI datasets are missing.**

### Questions
Q1: Why does DDPM act as an effective regularizer for LLM-based forecasting? What is the justification for this specific method choice compared to alternatives?

Q2: Why is "trained with DDPM, inferred without" superior to direct LLM training? Is there empirical evidence or ablation study supporting this design choice?

Q3: How does performance vary with λ in Eq. (8)? 

Q4: Where is the ablation study mentioned in Section 5? Are the results missing from the tables/figures?

Q5: Can the method provide uncertainty quantification at inference time, given that both DDPM and LLMs have capacity for probabilistic modeling?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
Proposes Diffusion-LLM, a framework integrating conditional Denoising Diffusion Probabilistic Model (DDPM) into LLM-based time series forecasting for probabilistic alignment and distribution-aware regularization.

### Strengths
- Targets LLM’s time series flaws (poor probabilistic modeling, weak alignment) via DDPM, with gains in ultra-long-term/few-shot scenarios.
- Lightweight DDPM avoids heavy params; inference speed matches baseline LLMs.

### Weaknesses
- Incremental innovation: relies on existing components (LLaMA, TimeLLM’s reprogramming, DDPM) with limited novel design.
- High training compute (A100/H100) but no quantitative training time/energy vs. baselines.
- Underperforms non-LLM baselines (e.g., PatchTST) in standard long-term forecasting.

### Questions
-  What unique designs distinguish it from prior generative-regularized LLMs for other modalities?
- Can training compute (GPU hours/memory) be quantified vs. TimeLLM/non-LLM baselines?
- Why not integrate DDPM into inference (e.g., probabilistic forecasting), and would this improve performance?
- For underperforming datasets (e.g., ETTm2, ECL), do conflicting LLM-DDPM objectives cause issues?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper targets LLM forecasters’ “regress-to-mean” and weak probabilistic alignment, proposing **Diffusion-LLM**: reprogram time series into an LLM token space and attach a **conditional DDPM** that learns $(p(z_y\mid z_x))$ to regularize the LLM with distribution-aware signals. Training minimizes a joint objective $(L_{\text{forecast}}+\lambda L_{\text{ddpm}})$; only the frozen-backbone LLM path is used at inference, keeping latency unchanged. Experiments on several standard benchmarks demonstrate consistent improvements over baselines, especially in ultra-long-term (1024–2048 steps) and few-shot forecasting tasks.

### Strengths
1. The paper is well-structured and easy to follow.
2. The proposed method shows relatively strong performance under the ultra long-term forecasting setting.

### Weaknesses
1. Lines 38–39 claim that many real-world scenarios require predictions far beyond the typical forecasting range. However, the authors do not provide concrete examples of such domains. **It is unclear why ultra long-term forecasting is necessary when sufficient historical data exists, as many practical applications—especially in finance—still focus on single-step or short-horizon predictions.**
2. Lines 57–59 argue that LLMs tend to predict the mean and rely more on nearby points when forecasting long-term. **Yet this issue is common across many deep learning time series models due to their training paradigms, not unique to LLMs.** Hence, the motivation for combining LLMs with diffusion models is not well justified.
3. Following the previous point, if one uses a well-designed time series model such as PatchTST [1] or iTransformer [2] together with the proposed DDPM module, the performance might surpass that of LLM-based models. If so, the role of LLMs here seems limited.
4. Table 1 shows that under standard forecasting horizons, the proposed model performs worse than TimeLLM [3]. This further raises concerns that the method offers little advantage in common forecasting scenarios, which are far more prevalent than ultra long-term ones.
5. Section 5 (lines 432–463) contains only analysis but no experimental results, which is unusual and weakens the empirical evidence.
6. The citation format is used incorrectly throughout. When citations are not part of the sentence structure, the author should use `\citep` instead of `\cite`.

[1] A Time Series is Worth 64 Words: Long-term Forecasting with Transformers

[2] iTransformer: Inverted Transformers Are Effective for Time Series Forecasting

[3] Time-LLM: Time Series Forecasting by Reprogramming Large Language Models

### Questions
See **Weakness**.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
In this paper the authors build upon the work of TimeLLM to study multimodal time series forecasting with existing LLM backbones. Their main contribution is the additional alignment of the encoded target time series into the LLM embedding space by introducing a denoising diffusion probabilistic modeling task between the input and the target time series as a regularization during model training. They empirically show that this added alignment helps with long term forecasting and forecasting in data scarce regimes.

### Strengths
The paper studies an interesting and legit question of time series tokenizer at the presence of the requirement of LLM alignment. The proposed method for aligning the target time series embeddings is clean. Paper writing is good.

### Weaknesses
The foundation of the research is not well established, which makes it hard to assess the significance and the generalization of the proposed work. In particular: (1) It is not clear that the base method, TimeLLM, is a good approach for tacking multimodal time series forecasting. (2) The selected benchmark datasets, which have been pervasive (overused) in time series research, might not be the best multimodal test cases. See questions.

### Questions
1. My most concern is about the benchmarking datasets. Though overly used by time series research, these datasets might not provide enough diversity in terms of the text prompts. In addition, based on the PatchTST performance in Figure 3, they are not essentially multimodal forecasting problems. What's your justification of the expectation that the proposed method can generalize to other real world usecases?

2. The proposed method seems to be not improving or strictly hurting the performance for the long term tasks with full data, (see Table 1, take into consideration the CIs). 

3. How does the trained DDPM stand as a unimodal forecaster?

### Soundness
2

### Presentation
3

### Contribution
4
