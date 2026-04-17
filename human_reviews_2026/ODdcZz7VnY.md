# TradeFM: A Generative Foundation Model for Trade-flow and Market Microstructure

- Decision: Reject
- Scores: 4, 4, 2, 2

## Abstract
Learning generalizable representations from the high-frequency, heterogeneous event streams of financial markets is a significant challenge. We introduce TradeFM, a foundation model that learns the universal dynamics of market microstructure. Pre-trained on billions of equities transactions, TradeFM uses a novel scale-invariant feature representation and a universal tokenization scheme to form a unified representation, enabling generalization without asset-specific calibration. We validate the quality of the learned representations by demonstrating that model-generated rollouts in a closed-loop simulator successfully reproduce canonical stylized facts of financial returns. We robustly evaluate the model’s ability to generalize to temporally and geographically out of sample data, as well as its ability to match real distributions of quantities like log returns and spreads. TradeFM provides a high-fidelity engine for synthetic data generation and downstream agent-based modeling.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces TradeFM, a large-scale generative foundation model designed for cross-asset financial market modeling. It addresses the challenge that traditional models are often asset-specific and fail to generalize across different market microstructures.

### Strengths
This paper is generally well written and easy to follow. However, several design choices appear somewhat ad hoc and lack clear theoretical justification or motivation. Without a stronger conceptual grounding, it is difficult to judge whether these modeling decisions are fundamentally sound or simply work well empirically.

### Weaknesses
1. The dataset used in the study is proprietary and not publicly released, which hinders independent replication and validation of the results.


2. The paper only compares TradeFM with a zero-intelligence random trader baseline. Although classical models such as Hawkes processes and early deep learning order-book models (e.g., DeepLOB) are mentioned, they are not empirically evaluated, weakening the comparative validity of the claims.

3. The paper lacks systematic quantitative evaluation of out-of-distribution generalization — e.g., performance on unseen assets, extreme volatility regimes, or temporal shifts — relying mostly on qualitative or intuitive arguments.


4. The market replay experiments simulate relatively short event sequences (512 trades per rollout). The paper does not examine whether the model can stably reproduce long-term market dynamics across multiple trading days.


5. While the empirical design of TradeFM is sound, the paper does not provide sufficient theoretical grounding for why the chosen feature normalization, tokenization, and model architecture should guarantee cross-asset generalization or preserve market invariants. A more formal discussion linking design choices to market microstructure theory would strengthen the work.

### Questions
1. Could the authors include comparisons with traditional models such as Hawkes process–based methods or deep learning order book models like DeepLOB, to better contextualize the empirical performance of TradeFM?

2. Could the evaluation of generalization be made more rigorous, for example by conducting cross-asset tests, rolling-window experiments, or stress simulations under different market regimes?

3. Is there any plan to release the tokenizer design or dataset, at least partially, to improve reproducibility and facilitate further research on cross-asset generative modeling?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes TradeFM, a decoder-only Transformer for modeling market microstructure as an autoregressive sequence over tokenized trade events. The major contributions include: (i) a novel tokenization that compresses multi-feature trade tuples into a single token, (ii) closed-loop rollouts with a market simulator, and (iii) high-fidelity synthetic data.

### Strengths
- Universal representation learned by TradeFM seems useful in downstream applications.
    
- The proposed approach directly works on raw data, without the need for human expertise.

### Weaknesses
- The introduction of scale-invariant features in Section 5.3  is unclear for readers who are not familiar with this concept. Concrete, self-contained explanations and citations to prior works are needed. Personally, I prefer to involve some simple illustrations or examples to clarify this.
    
- This paper only evaluates the fidelity of synthetic data generation using stylized facts. To evaluate the fidelity of synthetic time series, there are many other metrics, including goodness of fit and experimental settings like “train on synthetic, test on real“ (TSTR) [1].
    
- No ablation study to verify the effectiveness of the proposed tokenization and scale-invariant features.

[1] Yoon et al. "Time-series generative adversarial networks." *Neurips* 2019.

### Questions
- How can we define and interpret the universal representation in this context? Can we use this as a pre-trained embedding in other deep learning models?
    
- What exactly is the pre-training task? Next-token prediction?
    
- Is there a risk of data leakage using the train-test split?
    
- What’s the impact of different setups of bin type and size?
    
- There is only one baseline compared in the experiments. Are there any more recent ones?
    
- What’s the data source? Is it public?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces TradeFM to model financial market microstructure under partial observability. The authors consider scale-invariant features and propose a new tokenization scheme. The model is validated via a closed-loop simulator.

### Strengths
- The research problem is critical and challenging in financial economics.

- Design choices reflect market pragmatism. The model is learned from partial observations to mimic the realistic scenarios, which aligns with how practitioners actually see the market.

### Weaknesses
- Tokenizer calibration on the first 30 days invites regime bias and drift.

- Validation for the quality of synthetic data using stylized facts is not convincing enough. For example, the lack of autocorrelation and heavy tails are easy for many generative models to achieve, but their practical usefulness is limited in financial applications.

- Baselines are limited. A zero-intelligence agent is a weak competitor.

- There is no strong evidence that synthetic data improves downstream tasks.

### Questions
- This work integrates the idea of scale-invariant features in financial data with autoregressive modeling. Since scale invariance implies recurring patterns across time, have you considered modeling market data using a pattern-based scheme?

- What do you observe about how LLMs understand tokenized transaction data?

- Why is the tokenizer calibrated on the specific first 30 days of data?

### Soundness
2

### Presentation
2

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
The paper introduces TradeFM, a generative foundation model for trade-flow and market microstructure, pre-trained on billions of equities transactions over 130 trading days across 8,365 assets to learn a unified representation without asset-specific calibration. It learns from a partially observed market state, uses a scale-invariant feature representation with Exponentially-Weighted Volume-Weighted Average Price (EW-VWAP) mid-price estimation, and applies a universal tokenisation scheme that maps multi-feature events into a composite integer via a mixed radix construction (vocabulary 16,384), conditioned on liquidity, ∆p, and IMP. The architecture is a decoder-only Transformer based on the Llama family with grouped-query-attention (GQA) and rotary positional encoding (RoPE), totalling 251M parameters and achieving a held-out perplexity of 17.85. Evaluation couples TradeFM with a deterministic market simulator in a closed-loop rollout using price-time priority matching, where generated events update the LOB state that is then fed back to the model. Simulations reproduce stylised facts—lack of autocorrelation in returns, volatility clustering (slowly decaying ACF of absolute returns), heavy tails, and aggregational Gaussianity—and outperform a calibrated Zero-Intelligence (ZI) baseline on metrics such as kurtosis. The conditioning signals (liquidity tiers and market-level vs. participant-level indicators) enable controllable generation, supporting high-fidelity synthetic data generation, backtesting, data augmentation for illiquid assets, and multi-agent modelling.

### Strengths
The choice of a good research topic, Foundation Model, is a significant advantage for the paper. A clear formulation of trade-flow modelling under partial observability, including an EW-VWAP mid-price estimator and a scale-invariant feature representation that aims to align assets across liquidity regimes. A universal tokenisation via mixed-radix composite trade tokens (vocab 16,384) plus conditioning signals (liquidity bins, Δp, IMP), which yields a compact tabular embedding pipeline. A closed-loop evaluation with a deterministic price–time-priority simulator; the generated rollouts reproduce key stylised facts (near-zero return ACF, volatility clustering, heavy tails, aggregational Gaussianity). Controllability via conditioning (market- vs participant-level; liquidity tiers) showing intuitive shifts in volume/interarrival variability.

### Weaknesses
(1) Scaling evidence is missing for a foundation model claim. The paper reports a single model size (251M) and a single held-out perplexity (17.85). Please provide reproducible scaling curves across model sizes, data fractions, and compute (e.g., 50M/150M/251M/500M; 25/50/100% tokens), with monotonic trends and power-law fits. Without multi-point fits, the FM positioning is under-supported.

(2) Novelty vs MaRS and strong baselines. The paper positions universality largely at the feature/tokenization level; however, prior work (e.g., MaRS) also trains across heterogeneous assets. 

(3) Spanning 130 trading days from July 2024 to January 2025, across 8,365 unique assets, the data scale is not large, especially compared to MarS and LOBS5 (LOB Bench), because U.S. equity trading volume is concentrated in the front, high-liquidity names. For example, in the LOBSTER data, Apple’s one-year message and order book data, after compression, is 64 GB; even for S&P 500 stocks with low liquidity, the compressed data size is only about 20 MB.

(4) Many passages appear to be AI-generated. However, I did not see a disclaimer in the paper indicating this. I may be mistaken, but the Turnitin-AI rate is 38%, and I can submit this report to the AC if needed.

### Questions
(1) The results are not reproducible, so they do not demonstrate credibility. There are no details, such as code to prove reproducibility. The paper also does not specify what the matching engine is; it only states that price–time priority is used and provides a rules citation. When submitting the paper, you can place the code in the Supplementary Material. If you can provide detailed code during the rebuttal stage, I will run it. I have sufficient GPU resources and LOBSTER data (but it would be best if you provide your original training data as well as the scripts needed for training). If the authors believe data and environment are issues, I can SSH into a server they provide to check, for example, Colab.  I will examine the implementation of the matching engine; moreover, if I can reproduce similar results, I will consider adjusting the score.

(2) I need you to provide the training duration for the 4 Nvidia A10G GPUs; the paper does not disclose how long it ran or how many GPU-hours were used. This does not appear to be sufficient GPU resources for training a foundation model. You trained for a total of 4 epochs, and the paper does not report whether training converged. Based on experience, this number of epochs is unlikely to achieve convergence on tabular data, because tabular data is not text and already has an abstraction.

(3) For reproducibility and to rule out randomness, I need the authors to provide training/test curves across multiple seeds, with error bars. I also need the code so that I can run it myself to verify the repeatability of the experiments; if these materials are provided and I can validate the results, I will consider revising my score.

### Soundness
1

### Presentation
2

### Contribution
2
