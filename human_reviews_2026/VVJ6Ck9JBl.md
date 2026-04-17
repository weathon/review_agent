# Aurora: Towards Universal Generative Multimodal Time Series Forecasting

- Decision: Accept (Poster)
- Scores: 6, 8, 6, 4

## Abstract
Cross-domain generalization is very important in Time Series Forecasting because similar historical information may lead to distinct future trends due to the domain-specific characteristics. Recent works focus on building unimodal time series
foundation models and end-to-end multimodal supervised models. Since domain-specific knowledge is often contained in modalities like texts, the former lacks the explicit utilization of them, thus hindering the performance. The latter is tailored for end-to-end scenarios and does not support zero-shot inference for cross-domain scenarios. In this work, we introduce Aurora, the first Multimodal Time Series Foundation Model, which supports multimodal inputs and zero-shot inference. Pretrained on Cross-domain Multimodal Time Series Corpus, Aurora can adaptively extract and focus on key domain knowledge contained in corresponding text or image modalities, thus possessing strong cross-domain generalization capability. Through tokenization, encoding, and distillation, Aurora can extract multimodal domain knowledge as guidance and then utilizes a Modality-Guided Multi-head Self-Attention to inject them into the modeling of temporal representations. In the decoding phase, the multimodal representations are used to generate the conditions and prototypes of future tokens, contributing to a novel Prototype-Guided Flow Matching for generative probabilistic forecasting. Comprehensive experiments on 5 well-recognized benchmarks, including TimeMMD, TSFM-Bench, ProbTS, TFB, and EPF, demonstrate the consistent state-of-the-art performance of Aurora on both unimodal and multimodal scenarios.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes Aurora, a multimodal time series foundation model for zero-shot forecasting. Aurora is pretrained on cross-domain corpus with time series, text descriptions, and endogenous images. The encoder uses token distillation and modality-guided self-attention to fuse multimodal features. The decoder employs prototype-guided flow matching for probabilistic forecasting, where prototypes encode periodicity/trend initialized from text and image features. Experiments on TimeMMD, TSFM-Bench, and ProbTS show SOTA performance on both multimodal and unimodal tasks.

### Strengths
**S1. Novel multimodal foundation model architecture**

Aurora introduces a well-designed multimodal architecture with token distillation and modality-guided self-attention. The modality-guided mechanism (Equations 12-16) explicitly uses cross-modality correlations to adjust temporal attention, enabling domain knowledge injection. The prototype-guided flow matching is innovative, retrieving learned period/trend prototypes as starting points rather than Gaussian noise. Ablations (Table 4) validate both components with 16-70% MSE increases when removed.

**S2. Strong empirical performance across diverse benchmarks**

Aurora achieves SOTA on three benchmarks: rank 1 on MSE/MAE in 31/36 settings on TimeMMD (27% MSE reduction vs. Sundial), 24/28 on TSFM-Bench (15.1% vs. Time-MoE), and 18/23 CRPS/NMAE on ProbTS (21.5% vs. CSDI). The model shows strong zero-shot generalization, outperforming full-shot supervised models on several datasets (Climate, Environment). The 10%-shot results demonstrate data efficiency with 12.8% improvement over GPT4MTS using 90% less training data.

**S3. Comprehensive evaluation coverage**

Experiments span 20 datasets covering multimodal forecasting, unimodal deterministic forecasting, and probabilistic forecasting. The evaluation includes both zero-shot and few-shot scenarios, demonstrating versatility across task types and generalization capabilities.

### Weaknesses
**W1. Insufficient ablation on modality contributions:** Table 4 only ablates entire mechanisms but does not quantify individual modality contributions. The paper lacks experiments comparing time-only, time+text, time+image, and time+text+image performance. This omission makes it impossible to understand what each modality provides. Since images are deterministic transformations of time series (Equations 3-7), readers cannot assess whether images truly add information beyond text or simply provide redundant patterns already in the time series.

**W2. Insufficient ablations on modality-guided attention design:** Variant 1 removes the entire modality-guided mechanism, but the paper lacks finer-grained ablations to validate specific design choices. The Corr matrix computation (Equation 14) multiplies $VAttn$, learnable metric $W$, and $TAttn^T$, but alternative designs (text-only guidance, image-only guidance, additive combination) are not tested. The necessity of the learnable metric W is not validated.

**W3. Confusing experimental setup mixing training regimes:** Table 1 compares 10%-shot Aurora against full-shot baselines, conflating model capability with data efficiency. While this demonstrates Aurora's data efficiency, the comparison is confusing. Zero-shot Aurora outperforms some full-shot models, but readers cannot determine whether advantages come from architecture or simply using less data. A clearer separation of zero-shot comparisons (foundation model vs. foundation model) and few-shot comparisons (Aurora vs. baselines both at 10% data) would eliminate ambiguity.

**W4. No interpretability analysis of learned representations:** The paper provides no visualization or analysis of what Aurora learns. The 1000 learned prototypes in the bank are not visualized or characterized. Whether prototypes cluster into interpretable patterns (linear trends, sinusoids, seasonality) is unknown. Which prototypes are retrieved for different domains is not analyzed. Attention pattern changes with/without modality guidance are not shown. This absence prevents validating that Aurora learns meaningful multimodal representations rather than memorizing dataset-specific patterns.

**W5. Benchmark fragmentation creates unnecessary complexity:** Three separate benchmarks (TimeMMD, TSFM-Bench, ProbTS) use different evaluation protocols, metrics (MSE/MAE vs. CRPS/NMAE), and stride settings. Tables 9-11 in the appendix span many pages with inconsistent experimental configurations. This fragmentation increases complexity and reduces reproducibility. A unified benchmark covering multimodal, deterministic, and probabilistic tasks would streamline evaluation.

**W6. Limited efficiency and scalability analysis:** Table 8 reports 83.5ms inference time, but Figure 5 shows 100 samples are needed for good probabilistic performance. Training time, training cost, and parameter efficiency (accuracy per parameter) are not reported. For a 418M parameter model claiming to be a foundation model, understanding computational requirements and scalability is important for practical deployment.

### Questions
**Q1. Can you provide individual modality ablations?**

Report performance with time-only, time+text, time+image, and time+text+image to quantify each modality's contribution and validate that each adds value.

**Q2. Can you provide finer ablations of modality-guided attention?**

Test text-only guidance, image-only guidance, alternative combination methods, and ablate the learnable metric W to validate design choices in Equations 12-14.

**Q3. Can you visualize and characterize learned prototypes?**

Show visualizations of the 1000 prototypes, analyze whether they form interpretable clusters, and demonstrate which prototypes are retrieved for different domains and patterns.

**Q4. What are typical LLM-generated description examples?**

Provide more examples beyond Figure 7. How do you ensure generated descriptions contain genuine domain knowledge comparable to human expert knowledge? Have you validated description quality through human evaluation or comparison with real domain texts?

**Q5. Can you clarify the zero-shot vs. full-shot comparison?**

In Table 1, is the zero-shot vs. full-shot comparison primarily demonstrative rather than a rigorous benchmark? Can you add 10%-shot baseline results for fair apples-to-apples comparison?

**Q6. Can you document the full pretraining corpus?**

Provide the complete list of pretraining datasets with names, sizes, and explicit confirmation of which evaluation datasets are excluded to verify no data leakage.

**Q7. Would a unified benchmark be feasible?**

Could you propose or adopt a single unified benchmark covering multimodal, deterministic, and probabilistic forecasting to reduce experimental complexity and improve reproducibility?

### Soundness
3

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
5

### Summary
This work makes a timely and impactful contribution: it is, to the best of my knowledge, the first universal generative foundation model that unifies multimodal inputs (text + image + time series) for cross-domain zero-shot forecasting. Prior foundation models (e.g., MOIRAI, Chronos, Sundial) are unimodal; prior multimodal models (e.g., Time-LLM, CALF, GPT4MTS) are supervised and non-generative. Aurora bridges this gap. The prototype-guided flow matching is also a compelling design that provides interpretable inductive bias (trend/periodicity) to the generative process, improving sample efficiency and forecast realism.

### Strengths
- First generative multimodal foundation model enabling cross-domain zero/few-shot time series forecasting.
- Novel cross-modal encoder (modality-guided attention) and prototype-guided decoder design boost generalization and probabilistic prediction quality.
- Extensive experiments across diverse benchmarks (TimeMMD, TSFM-Bench, ProbTS) consistently outperform unimodal/multimodal baselines.

### Weaknesses
- Prototype scalability: The prototype bank contains 1,000 learnable patterns. How does performance vary with the number of prototypes (e.g., M = 100 vs. 1,000 vs. 10,000)? Is there a trade-off between forecast accuracy, training stability, and inference latency? A small ablation would clarify whether this component is over-parameterized.
- The method assumes that text and image modalities are faithful and aligned with the time series. However, in real-world settings: Text may be vague, outdated, or incorrect; and images rely on FFT-based period estimation, which can be unreliable for non-stationary or chaotic series. The paper does not evaluate modality corruption robustness (e.g., masking text, perturbing image tokens), which is critical for practical adoption.
- Can Aurora be extended to other multimodal time series tasks (e.g., anomaly detection)?

### Questions
See the weaknesses

### Soundness
3

### Presentation
4

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
This paper presents Aurora, which is a generative foundation model to unify time series forecasting across diverse domains and modalities. First, it has token distillation and modality alignment modules for cross-modality fusion, then the modality-guided attention will inject textual and visual knowledge into temporal encoding. Lastly, in the  decoding stage, a prototype-guided flow matching will improve generative probabilistic forecasting by using learned trend and periodic prototypes. Through the experiments on TimeMMD, TSFM-Bench, and ProbTS, the authors demonstrate Aurora's good performance in zero-shot and few-shot performance across unimodal and multimodal settings.

### Strengths
1. It really focuses on the limitations of the existing multimodal TS forecaster and foundation models. It introduces a novel multimodal pretraining paradigm with datasets, which is a huge contribution to this field. 
2. The architecture is novel. The modality guided attention bridges domain knowledge with temporal information. The prototype guided flow matching improves the interpretability of probabilistic forecasting.
3. Aurora is evaluated on diverse benchmarks with both deterministic and probabilistic metrics. The results consistently show the effectiveness of the proposed architectures.

### Weaknesses
1. Using LLM to generate a textual description can be problematic. The paper provides no analysis of data leakage or the quality analysis of generated texts for pretraining data. Also, the paper should provide a more rigorous clarification regarding whether the pretraining data has any overlap with the benchmark datasets.
2. Prototypes give more interpretability for this paper, but have no analysis or visualization of learned prototypes. 
3. Missing TTM[1] in related work and baseline comparison. 


[1] Ekambaram, Vijay, et al. "Tiny time mixers (ttms): Fast pre-trained models for enhanced zero/few-shot forecasting of multivariate time series." Advances in Neural Information Processing Systems 37 (2024): 74147-74181.

### Questions
1. Is the use of Flow Matching in the decoder essential, or could other generic multimodal decoding frameworks be applied instead?
2. Could you provide specific examples where your model demonstrates advantages to your claim: cases where similar historical patterns lead to different future trends due to domain-specific characteristics?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors propose Aurora, a multimodal time series foundation model. By leveraging multimodal information fusion, Aurora achieves impressive zero-shot forecasting performance. The paper also introduces a new decoder design. Experimental results demonstrate that the proposed method outperforms baseline models across various datasets.

### Strengths
- The authors propose a powerful zero-shot time series foundation model that effectively integrates multimodal information for prediction.
- The proposed method integrates different modules and achieves remarkable performance in terms of robustness.

### Weaknesses
### **Experiment Setting and Performance**

The proposed method achieves better zero-shot performance than baselines. However, MAE and MSE are greater than 0.5, as seen in PEMS08, Traffic, and Wind. I am not sure whether the metrics are calculated based on normalized results. If so, a high MAE or MSE may not be better than a straight line or random output.


### **Inference time**

The work focuses on few-shot learning and zero-shot learning. However, compared to the training cost, such a large model increases the inference time and requires more GPU resources. I suggest that the author include an experiment to compare the proposed 16G model with iTransformer on ETT, traffic, or weather, in terms of total running time (training time + inference time for a fair comparison). I am curious about whether the training time and inference time of a small model may be quicker than that of a large model. 

### **Baselines**

I noticed the author only chose large foundation models to evaluate few-shot and zero-shot learning. Could the author test small models, e.g., iTransformer and recently advanced methods, on few-shot learning with different budgets, such as 10% and 20%?

### **Full-shot Setting**

Due to the training cost of the small model, the author should also compare their proposed method with unimodal models on full-shot settings, such as PatchTST, iTransformer, TimeXer, or other new advanced methods.

### **Benchmarks**
As a time series forecasting task, prediction is 7.5 times the input length, which may not be useful in daily life. Such as forecasting the future one-week weather, but only based on 16 hours. I think the authors should consider making their work more valuable and closer to everyday life settings. Moreover, the author should add more short-term forecasting datasets, such as M4, PEMS, illness, and EPF datasets.

### Questions
I am willing to adjust the rating based on the authors' feedback, particularly regarding my concerns about baselines, benchmarks, and experiment settings.
Please refer to the weakness.

### Soundness
3

### Presentation
3

### Contribution
2
