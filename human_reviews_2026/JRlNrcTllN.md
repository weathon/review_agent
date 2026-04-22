# CoRA: Boosting Time Series Foundation Models for Multivariate Forecasting through Correlation-aware Adapter

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 8, 2, 8

## Abstract
Most existing Time Series Foundation Models (TSFMs) use channel independent modeling and focus on capturing and generalizing temporal dependencies, while neglecting the correlations among channels or overlook the different aspects of correlations. However, these correlations play a vital role in Multivariate time series forecasting. To address this, we propose a Correlation-aware Adapter (**CoRA**), a lightweight plug-and-play method that requires only fine-tuning with TSFMs and is able to capture different types of correlations, so as to improve forecast performance. Specifically, to reduce complexity, we innovatively decompose the correlation matrix into low-rank Time-Varying and Time-Invariant components. For the Time-Varying component, we further design learnable polynomials to learn dynamic correlations by capturing trends or periodic patterns. To learn positive and negative correlations that appear only among some variables, we introduce a novel dual contrastive learning method that identifies correlations through projection layers, regulated by a Heterogeneous-Partial contrastive loss during training, without introducing additional complexity in the inference stage. Extensive experiments on 10 real-world datasets demonstrate that CoRA improves the state-of-the-art TSFMs in average forecast performance.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces CoRA, a lightweight and plug-and-play module designed to enhance Time-Series Foundation Models (TSFMs) by better modeling inter-channel correlations. CoRA employs a correlation decomposition mechanism combining global time-invariant matrices and learnable polynomial bases, capturing both static and dynamic dependencies. Importantly, CoRA does not require re-pretraining of the base TSFM and can be applied during few-shot fine-tuning, making it a practical adaptation method.

### Strengths
1. Clear Motivation and Positioning:
The paper clearly identifies a key limitation in existing TSFM-based forecasting—insufficient modeling of heterogeneous inter-channel correlations—and proposes a theoretically grounded yet efficient mechanism to address this.

2. Lightweight and Practical Design:
CoRA's plug-in design allows easy integration with existing TSFMs during fine-tuning without requiring re-pretraining. Its parameter efficiency and low inference overhead make it appealing to practitioners seeking post-hoc adaptation over full model retraining.

3. Theoretical Depth and Interpretability:
The proposed correlation decomposition elegantly connects polynomial-based dynamic components with global invariant structures. Appendix provides a theoretical foundation and enhances interpretability of the correlation modeling process.

### Weaknesses
1. Limited Empirical Evaluation Scope:
Table 1 focuses on few-shot MSE benchmarks using standard datasets but lacks evaluation under out-of-distribution shifts, missing channels, or nonstationary regimes—critical aspects for real-world TSFM deployment.
Figure 4 presents relative MSE changes, which obscures absolute performance deltas; including absolute values alongside relative improvements (as in Table 1) would increase transparency.

2. Assumption Boundaries Underexplored:
Theoretical guarantees rely on local stationarity and bounded smoothness assumptions that may not hold in volatile domains (e.g., finance, IoT, environmental data). The paper does not examine edge cases such as abrupt regime shifts or high-noise inputs.

3. Potential Overstatement of Generality:
The paper claims CoRA "captures various correlations with O(N) inference complexity." While this is theoretically sound, real-world multivariate systems may involve nonlinear or hierarchical dependencies that exceed the current decomposition's expressiveness. The claim is reasonable but should avoid implying complete modeling capability.

### Questions
1. Statistical Significance and Robustness:
Are the improvements in Tables 1 and 2 statistically significant? Please report standard deviations, confidence intervals, or results across multiple random seeds to validate robustness.

2. Adaptation in Nonstationary Environments:
How does CoRA behave under regime shifts or nonstationary dynamics where the global time-invariant assumption may break down? Could it be extended to support online updates or adaptive windowing?

3. Scalability to Very High-Dimensional Settings:
For extremely large N (> 500), do global correlation matrices introduce memory or numerical bottlenecks? Any empirical or theoretical evidence on efficiency and convergence at that scale?

4. Extensibility Beyond Forecasting:
Could the proposed correlation decomposition framework generalize to other TSFM-based tasks—such as anomaly detection or temporal classification? What architectural adjustments would be required?

A thorough response and additional evidence to the above points would influence my score adjustment.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
To address the limitation that existing time series foundation models often overlook inter-channel modeling, this paper proposes a lightweight adapter, termed CoRA. CoRA innovatively decomposes the correlation matrix into time-varying and time-invariant components. It leverages learnable polynomials to capture dynamic patterns and introduces a novel dual contrastive learning mechanism to distinguish between positive and negative correlations. Optimized via a heterogeneous-local contrastive loss, CoRA incurs minimal computational overhead during the inference stage. Empirical results demonstrate that CoRA provides an effective solution for correlation modeling in multivariate time series forecasting.

### Strengths
1. The paper is well-motivated, addressing the critical challenge of modeling complex inter-channel dependencies in multivariate time series.
2. The manuscript is well-written, featuring clear and consistent notation throughout.
3. The empirical evaluation demonstrates that CoRA consistently surpasses state-of-the-art baselines across a diverse set of real-world datasets.

### Weaknesses
1. The projection layers are not much different from the related modeling methods in existing methods (such as TSMixer [1]), a more detailed clarification is needed.
2. The set of baselines for comparison is not comprehensive, as it omits recent state-of-the-art foundation models like TimerXL [2].	
3. Discuss and comparison with classical baselines (e.g., PatchTST  [3], Leddam [4], iTransformer [5], Autoformer [6], DLinear [7]) is suggested.

*[1] TSMixer: An All-MLP Architecture for Time Series Forecasting.*

*[2] Timer-XL: Long-Context Transformers for Unified Time Series Forecasting.*

*[3] A Time Series is Worth 64 Words: Long-term Forecasting with Transformers.*

*[4] Revitalizing Multivariate Time Series Forecasting: Learnable Decomposition with Inter-Series Dependencies and Intra-Series Variations Modeling.*

*[5] iTransformer: Inverted Transformers Are Effective for Time Series Forecasting.*

*[6] Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting.*

*[7] Are Transformers Effective for Time Series Forecasting?.*

### Questions
1. How does CoRA, as a plug-and-play module, accommodate the fundamental architectural differences between encoder-only and encoder-decoder models, particularly with respect to their distinct training objectives and inference processes?
2. What is the underlying rationale for incorporating rule-based correlation relationships when the model is already designed to capture these dependencies through a learnable mechanism?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a way to fine-tune multivariate time series foundation models as to better capture correlation between variables. Key ideas are to estimate a correlation matrix between features that has a time-dependent and time-invariant decomposition, and to screen for which variables to actually focus on in terms of positive or negative correlations via a contrastive learning strategy.

### Strengths
- I appreciate that the paper is trying to separately handle different types of correlation (DCorr, HCorr, PCorr)
- I find the idea of using a time-varying and time-invariant decomposition to be valuable, along with the idea of finding pairs of features with strong enough positive or negative correlation
- The experimental results appear to be very impressive (although as pointed out in weaknesses, I would like to see more baselines and some additional experiments)

### Weaknesses
- At least in how it is stated now, it's hard for me to see why Theorem 1 should hold since equation (15) in Theorem 1 disagrees with equation (5) and the decomposition showed earlier in Figure 3.
- In stating theoretical guarantees, I would suggest also providing intuition for why the guarantee should hold (this intuition should be in the main paper and not in an appendix/supplemental material as it helps the reader understand/interpret the guarantee) and whether the proof uses any nontrivial ideas, or if it's largely just based on an existing result. Maybe I'm missing something but Theorem 2 seems to be a known result regarding polynomial approximation using a variant of the mean value theorem? If the result was not previously known, what are the closest known existing theoretical results (which can help provide the reader with point(s) of comparison as to what proof techniques/ideas are novel)? Also can you comment on the extent to which the assumption holds for real data?
- Experiments: It would be helpful including more baselines, especially ones that already account for correlation structure (even if it doesn't do so as comprehensively as the proposed approach) such as UniTS and Moirai (both of which are mentioned in the paper but not actually compared against in the experiments).
- Experiments: Right now Tables 1 and 2 are reported without error bars (such as standard deviation over experimental repeats with different random seeds). Please include error bars as to give a better sense of variability in results.
- Experiments: It would be helpful seeing for the different datasets how the amount of data available for fine-tuning affects the performance of CoRA especially since I'd imagine that if one has somewhat limited data to fine-tune with, then the correlation matrix might be hard to estimate accurately.
- Overall I find that the paper has many exposition issues and typos that altogether make it so that the paper has serious clarity problems. Here are a few issues (this list is not exhaustive):
    - (minor) line ~17: since you're abbreviating "**Cor**relation-aware **A**dapter", shouldn't the abbreviation be "CorA" and not "CoRA" (i.e., why is the "R" capitalized)?
    - line ~53 (last line of page 1 that actually is one line after line 53): the explanation here makes it clear why MLP doesn't handle DCorr but why precisely does it not model HCorr? Can an MLP not figure out positive or negative correlations? I'd imagine that the learned weights would be able to capture this sort of information?
    - line ~122: "LLMs enable" should say "LLMs to enable"
    - line ~153-154: "$L$ length" should say "length $L$"
    - line ~153-154: please clearly define what $t$ is
    - line ~161: "pluged" should say "plugged"
    - line ~185: the comma after "Figure 2" should be a period
    - lines ~205-206: please clearly state what $M$ is (the rank) and perhaps comment on how it is chosen (clearly it seems like we would want $M < N$ but is this a hyperparameter that is tuned later?)
    - Figure 3: it would be helpful to better motivate the decomposition especially since the time-varying and time-invariant information is getting mixed together in the multiplication so that they're not decoupled (also Figure 3 seems to make it seem like $M_t^{corr} = Q_t V Q_t^T$ but then this isn't actually the case according to equation (5) or equation (15) later) -- yet somehow lines ~209-210 suggests that we can estimate $Q_t$ and $V$ separately and the text doesn't sufficiently provide intuition at this point for why this separate estimation should be possible
    - Section 4.1.1: in modeling dynamic regularities, it would be helpful to better motivate why polynomials are particularly well-suited for the task vs alternative approaches (I'd imagine higher order polynomials become more susceptible to noise, for instance; for example, how does the proposed approach compare with a different strategy that amounts to just using a first-order polynomial but having multiple basis functions?)
    - line ~231-232: "i-th" should say "$i$-th"
    - line ~233-234 and also line ~236: inconsistent notation -> "$q^i$" should use a bold version of "$q$" to match equation (1); also I'd suggest stating clearly that "$\odot$" stands for Hadamard product
    - line ~235: "we define the set of $C_{i,t}$" should not use the word "set" here since what is being described is not what the word "set" means in math
    - line ~250-251: "Where" should not be capitalized"
    - line ~261: "L is the size of the input series" -> "L" should be italicized (i.e., written as the math variable $L$)
    - line ~261-262: "$\mathbf{R}\in\mathbb{R}^{N\times N}$ to denote the set of $r$" -> this is not the correct use of the word "set" in math
    - equation (5) does not agree with equation (15) and also does not seem to agree with Figure 3
    - line ~282: "Where" should not be capitalized
    - line ~282: "layerNorm" should be capitalized (i.e., written as "LayerNorm") to match equations (6) and (7)
    - line ~283: $\text{MLP}_{\alpha}=\mathbb{R}^d\rightarrow\mathbb{R}^d$ ---- the equal sign "$=$" should be replaced by a colon "$:$" (a similar math typo shows up for the other MLP defined in this same line)
    - does $\beta$ in equation (7) and in equation (14) actually mean different things?

### Questions
Please see weaknesses (I raised a number of concerns there).

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper proposes CoRA (Correlation-aware Adapter), a lightweight, plug-and-play module designed to enhance existing Time Series Foundation Models (TSFMs) for multivariate forecasting. The central premise is that many TSFMs, while powerful in capturing temporal dependencies, neglect the complex inter-channel correlations that are vital for accurate multivariate prediction in downstream tasks. The main contributions of the paper can be summarized as follows:
1. A Unified Framework for Complex Correlations: The paper provides a structured view by categorizing inter-channel relationships into three types: Dynamic (DCorr), Heterogeneous (HCorr), and Partial (PCorr) correlations.
2. An Efficient and Effective Adapter Design: CoRA is designed as an efficient adapter that only requires fine-tuning. It models dynamic correlations (DCorr) by innovatively decomposing the correlation matrix into low-rank time-varying and time-invariant components.
3. A Novel Contrastive Learning Method (HPCL): The core technical contribution is the HPCL module, which models HCorr by projecting representations into separate positive and negative latent spaces. Within each space, it then applies a guided contrastive loss to cluster strongly correlated channels, a process that elegantly captures PCorr.

### Strengths
This paper proposes CoRA, a novel correlation-aware adapter designed to enhance the multivariate forecasting capabilities of Time Series Foundation Models (TSFMs) on specific downstream tasks.
1. Originality: The paper is the first to propose a unified framework that simultaneously addresses the dynamic, heterogeneous, and partial aspects of inter-channel correlations.
2. Quality: The paper is of high quality, featuring a rigorous methodology, clear theoretical derivations, and practical efficiency.
3. Clarity: The paper is well-written, with a complete logical flow and effective visualizations.
4. Significance: This work is highly significant for the time series domain, particularly for fine-tuning TSFMs on downstream tasks.

### Weaknesses
This paper could be improved in the following areas:
1. Lack of Direct Experimental Validation for Partial Correlation (PCorr): The paper claims to model three types of correlations: DCorr, HCorr, and PCorr. While experiments provide strong evidence for DCorr (e.g., Fig. 7) and HCorr (separation of positive/negative spaces), the validation for PCorr is less direct. The ablation study (Table 2) shows the overall benefit of the HPCL module but does not disentangle the specific contributions of modeling HCorr versus PCorr. The visualization in Fig. 7 implicitly shows PCorr through clustering but fails to highlight or analyze it separately.
2. Omission of Experimental Details: The paper states that the thresholdεis a "learnable" parameter but does not detail how it is optimized. Furthermore, the main results in Table 1 are based solely on a 5% few-shot setting. It is suggested that the authors include experiments conducted with full-dataset fine-tuning.
3. Confusing Notation: The authors use the standard and calligraphic forms of the same letter to represent two different concepts. It is recommended that the authors use distinct symbols and consider adding a notation table to the paper to ensure consistency and clarity.

### Questions
1. Regarding the Explicit Validation of Partial Correlation (PCorr): The paper's core thesis is the unified modeling of DCorr, HCorr, and PCorr. While the evidence for DCorr and HCorr is quite direct, the support for PCorr modeling appears more implicit. Could you provide a more direct piece of evidence to demonstrate that CoRA is effectively learning and leveraging partial correlations?
2. Regarding the Optimization of the Learnable Threshold ε: Could you please clarify the specific mechanism that allows gradients to flow back toεduring the training process?
3. Regarding Performance in Data-Rich Scenarios: The main experiments are conducted in a 5% few-shot setting. However, how does CoRA perform when fine-tuned on the full training dataset (100% data)? Does the performance gap between the baseline TSFM and TSFM+CoRA widen, narrow, or remain the same?

### Soundness
4

### Presentation
4

### Contribution
4
