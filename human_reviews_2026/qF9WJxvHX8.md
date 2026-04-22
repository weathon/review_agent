# Building Massively Multimodal Foundation Models with Interaction-aware Mixture-of-Experts

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 6, 8, 4, 8

## Abstract
Modern applications increasingly involve many heterogeneous input streams, such as clinical sensors, wearable device data, imaging, and text, each with distinct measurement models, sampling rates, and noise characteristics. We define this as massively multimodal setting, where each sensor constitutes a separate modality. As modality counts grow, capturing their complex, time-varying interactions such as delayed physiological cascades between sensors, has becomes essential yet challenging. Mixture-of-Experts (MoE) architectures are naturally suited for this setting since their sparse routing mechanism enables efficient scaling across many modalities. However, existing MoE architectures route tokens based on similarity alone, overlooking the rich temporal dependencies across modalities: this prevents the model from capturing delayed cross-modal effects, leading to suboptimal expert specialization and reduced accuracy.
We propose a framework that explicitly quantifies temporal dependencies between modality pairs across multiple discrete time intervals, defined as delays between an event in one input stream and its manifested effect in another, and uses these to guide MoE routing. A interaction-aware router dispatches tokens to specialized experts based on interaction type. This principled routing enables experts to learn generalizable interaction-processing skills. Experiments across healthcare, activity recognition, and affective computing benchmarks demonstrate substantial performance gains and interpretable routing patterns aligned with domain knowledge.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper presents TIME-MoE, a Temporal Interaction-guided Mixture-of-Experts framework that leverages information-theoretic decomposition (RUS) to guide expert routing.

### Strengths
The paper introduces a novel information-theoretic framework for guiding expert routing in multimodal MoE systems.

### Weaknesses
The proposed RUS estimation and routing involve substantial computational overhead and complex hyperparameter tuning. Scalability to large-scale multimodal models remains to be shown.

### Questions
1. The proposed multi-scale estimator is interesting, but the overall computation involving PID and Sinkhorn alignment still seems heavy, especially for long temporal sequences or high-dimensional inputs. It would be helpful if the authors could comment on the practical computational cost and scalability of TIME-MoE in such settings.

2. The model appears to rely on several threshold and weighting hyperparameters (τ_R, τ_U, τ_S, λ_R, λ_U, λ_S). How sensitive are the results to these choices? Any guidance or heuristics for tuning them would be useful.

3. Most of the experiments are on medium-scale datasets. Have the authors tried (or do they see a clear path to) applying TIME-MoE to larger-scale setups, e.g., vision-language or multimodal LLM tasks?

4. The framework is built on information-theoretic principles, but in practice RUS estimation depends on neural approximations. How stable are these estimations during training, and is there any quantitative sense of their bias or variance?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces TIME-MoE (Temporal Interaction-guided Mixture of Experts), a novel multimodal architecture that integrates temporal interaction dynamics into the MoE routing process. The core idea is to quantify multimodal interactions (redundancy, uniqueness, synergy – RUS) over time, and use these dynamics to guide expert routing. The authors design a RUS-aware router and corresponding auxiliary losses that encourage experts to specialize in particular interaction types. Empirical results on multiple benchmarks (MIMIC-IV, PAMAP2, WESAD, MOSI, Opportunity) show that TIME-MoE outperforms state-of-the-art fusion and MoE-based baselines (e.g., FuseMoE, I2MoE), achieving both higher predictive accuracy and more interpretable expert activation patterns over temporal dimension.

### Strengths
1. Novel theoretical grounding: Extends Partial Information Decomposition (PID) to the temporal domain using directed information, leading to a measurable RUS metric that captures delayed effects.
2. Principled architecture: The RUS-aware router and auxiliary loss design directly encode theoretical insights into model training.
3. Empirical generality: Demonstrates improvements across heterogeneous datasets (clinical, affective, activity, physiological).
4. Interpretability: Expert routing visualizations reveal meaningful alignment with modality interactions (e.g., redundancy between chest and hand motion over the temporal dimension).
5. Comprehensive evaluation: Includes ablations, sensitivity to RUS sequence length, and comparison with multiple baselines.

### Weaknesses
W1. Computing temporal RUS and training the router could be computationally expensive, which may limit scalability for large multimodal models.

### Questions
1. How sensitive is TIME-MoE to errors in RUS estimation, especially when temporal dependencies are weak or noisy?
2. Can the RUS-guided router be learned end-to-end without explicit precomputation of RUS values?
3. Could the same framework generalize to non-temporal multimodal fusion (e.g., static image–text tasks) or to LLM-scale architectures?
4. What is the computational overhead compared to FuseMoE and I2MoE in FLOPs and training time?
5. How does the proposed directed-information-based temporal RUS compare with simpler correlation-based interaction metrics?

### Soundness
3

### Presentation
4

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
This paper presents the study on Temporal Interaction-guided Mixture-of-Experts (TIME-MoE). This is a modified MoE architecture that explicitly leverages time-varying multimodal interactions to guide expert routing. In conventional MoE routers the tokens are navigated based solely on content similarity, while for some tasks and inputs, the temporal interaction between tokens is crucial. TIME-MoE's router is interaction-aware, it considers the redundancy, uniqueness, and synergy (RUS) between modalities over time when deciding which expert processes which token. By using quantified interaction dynamics as a prior, the model encourages experts to specialize in processing particular interaction patterns (redundant, unique, or synergistic information) rather than only learning task-specific features. 
The authors testes their methods on various multimodal benchmarks, mostly connected to the medical domain, and found out that their method outperforms alternative MoEs approaches, and also improves interpretability of the method.

### Strengths
The main strengths of the research are the following:

1. Novel interaction-aware MoE for multimodal data.
The authors integrate an information-theoretic framework with modern deep learning to guide Mixture-of-Experts routing using temporal redundancy, uniqueness, and synergy (RUS). This design yields an MoE that is both temporally and modality-grounded, leading to more specialized and interpretable experts.
2. Unlike prior approaches that rely on static cross-modal correlations, TIME-MoE explicitly models time-lagged multimodal effects, which are especially important in medical and physiological domains.
3. Thorough and diverse evaluation.
The approach is tested on a broad set of multimodal benchmarks — PAMAP2, MIMIC-IV (IHM/LOS), MOSI, WESAD, and Opportunity — covering various modality mix. The authors validate their mechanism through component ablations and routing-pattern analyses, confirming both effectiveness and interpretability.
4. Comprehensive ablation studies.
Each component (redundancy, uniqueness, synergy guidance) is evaluated separately, demonstrating its distinct contribution to the final performance.

### Weaknesses
I would define the following weaknesses:

1. Pairwise-only interaction modeling.
The RUS framework focuses on pairwise modality interactions; higher-order (three-way or more) synergies are not explicitly modeled. While this ensures scalability, it limits the expressiveness of the model.
2. Although the authors claim a τ-fold speedup in RUS computation and improved parameter efficiency, the paper does not provide GPU hours, or memory-usage comparisons. Including such measurements would make the efficiency claims more convincing.
3. Integration with large-scale LLM-MoE frameworks remains unexplored.
Given the rapid development of Large Language Model MoEs (e.g., Mixtral, DeepSeek-MoE, etc.), it would be highly valuable to discuss how TIME-MoE could scale to these architectures. The paper does not explore or speculate on integrating its RUS-guided routing into LLM-scale systems, which could significantly broaden its impact.
4. While Appendix E describes architecture and hyperparameters, and Appendix D details the RUS estimation algorithms, the paper lacks a concise description of the training setup (optimizer, learning rate, batch size, epochs, etc.).

### Questions
1. Please clarify the optimizer, learning-rate schedule, batch size, number of epochs, and gating top-k used during TIME-MoE training. Were modality encoders frozen or fine-tuned?
2.
Is RUS estimation performed as a fully offline pre-processing step (aligned with train/validation/test splits), or is there any joint optimization of the RUS estimator with TIME-MoE training?
3. Did you observe any expert collapse (e.g., one expert processing the majority of tokens)?
4. How sensitive is the final model’s performance to the accuracy of the RUS estimation? If the interaction estimator produces noisy or weak signals, does performance degrade significantly?
5. Could you report wall-clock training times, GPU type/count, and memory usage for both RUS estimation and TIME-MoE training, relative to baseline MoE models (e.g., FuseMoE, I2MoE)?
6. Do you plan to extend TIME-MoE to large-language-model MoE architectures? Such integration could yield interpretable and context-aware routing in large-scale generative models.

If the authors address my questions, I am willing to increase the final score.

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
This work introduces TIME-MOE, which routes tokens in a mixture-of-experts using time-aware interaction cues between modalities. The model estimates redundancy, uniqueness, and synergy (RUS) via a directed-information flavored partial information decomposition, producing sequences over different lags. These sequences condition the router, while auxiliary losses encourage consistent behavior: co-route redundant inputs, diversify for unique inputs, and prefer dedicated synergy experts when cross-modal effects appear.

### Strengths
1)	Temporal interaction modeling: The paper builds on PID with directed information to model time-lagged redundancy, uniqueness, and synergy between modalities. The derivation reads cleanly and fits the needs of multimodal time series.
2)	Architecture: The RUS-aware router plus synergy experts turn the interaction signals into concrete routing decisions. The activation patterns look modality-consistent and more structured than a standard MoE.
3)	Efficient multi-lag estimation: The multi-scale estimator reuses computation across lags, staying close to step-wise estimates while delivering roughly τ-fold speedups. 
4)	Empirical results: Strong empirical results & breadth: TIME-MOE wins on most metrics across 6 tasks, including clinical (MIMIC-IV) and affective datasets (MOSI, WESAD); results averaged over 5 runs.
5)	Reproducibility: Anonymous code link and detailed appendices and hyper params are provided.

### Weaknesses
1)	Estimator guarantees and identifiability: The construction of optimal Q*_{tau} via Sinkhorn-normalized alignment tensors is elegant but lacks formal guarantees about convergence to the PID-consistent minimizer and the induced bias in R/U/S under finite data and high-dimensional encoders. A short theorem or calibration study (e.g., on synthetic systems with known RUS) would make the claims much stronger.
2)	Dependency on label availability and distribution: For sequence-level labels (MIMIC-IV, MOSI), RUS is computed at a single time step while sweeping lags from the sequence end. That’s reasonable, but it could be brittle under class imbalance or label noise. Please analyze sensitivity and discuss how label quality affects RUS and routing.
3)	Scalability and cost: There is qualitative speedup from the multi-scale RUS estimator, but the paper could better quantify end-to-end training cost (GPU-hours) and router overhead vs. a standard MoE for larger expert counts and more modalities.

### Questions
1)	Estimator validation: Could You please provide a synthetic benchmark where ground-truth RUS over lags is known, to calibrate the multi-scale estimator vs. a step-wise computation, including error bars across data sizes? How sensitive are RUS estimates to encoder capacity and the Sinkhorn regularization?
2)	Label choice: For MIMIC-IV and MOSI, have you tried alternative targets (e.g., intermediate pseudo-labels or weak supervision) to compute more informative temporal RUS sequences? Any performance deltas?
3)	Computationals : Please report GPU-hours for TIME-MOE vs. other MoE baselines across datasets, and the incremental cost attributable specifically to RUS estimation and auxiliary losses.

### Soundness
3

### Presentation
3

### Contribution
3
