# Test-Time Efficient Pretrained Model Portfolios for Time Series Forecasting

- Decision: Accept (Poster)
- Scores: 6, 4, 2, 8

## Abstract
Is bigger always better for time series foundation models? With the question in mind, we explore an alternative to training a single, large monolithic model: building a portfolio of smaller, pretrained forecasting models. By applying ensembling or model selection over these portfolios, we achieve competitive performance on large-scale benchmarks using much fewer parameters. We explore strategies for designing such portfolios and find that collections of specialist models consistently outperform portfolios of independently trained generalists. Remarkably, we demonstrate that post-training a base model is a compute-effective approach for creating sufficiently diverse specialists, and provide evidences that ensembling and model selection are more compute-efficient than test-time fine-tuning.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents Chroma and framework for having portfolio of specialized models originated by fine-tuning a general purpose model for time-series forecasting. At inference time, this frameworks employs a subset of the models based a Greedy Ensemble Weighting algorithm. The paper provides a lot of empirical evaluations ranging from performance and ablation to test-time compute (in terms of FLOPs) and scaling. The results (including statistical analysis and bias-variance tradeoff) suggest the utility of the proposed framework in terms of time saving while maintaining performance.

### Strengths
- The paper is strong in terms of empirical contributions to support claims 
- I also found the paper strong in terms of reproducibility 
- Overall, I think the paper provides insights for the community that could be interesting therefore, I am in favour of acceptance.

### Weaknesses
- I think the paper lacks technical novelty because training a source model and then fine-tuning it for different target tasks has been round in transfer learning (particularly domain adaptation) for a long time. Similarly, the studied selection strategies are also known. Therefore, it seems the main contribution of Chroma is the fine-tuning step as opposed to training from scratch  (comparing figure 1 b and c) but that step according to table 2 did not have an impact on the performance as opposed to the ensembling which makes the technical contributions very limited. 

- This is not a weakness per se but I think it will be beneficial to show that the presented results are not only specific to T5 architecture 

- I think the paper can benefit from more discussion around G.5 results. Particularly to identify for what type of problems this approach works better and where we should train from scratch. Currently, these results are buried in the appendix but a discussion around them can clarify the applicability domain of the presented framework better.

### Questions
- How many times each experiment was repeated? I recognize the paper is quite strong in terms of experiments but providing confidence interval or standard error can make the message stronger specially for key experiments

- Did you perform p-value correction for multiple hypothesis testing? if so, what was the employed method?

### Soundness
3

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
The authors proposed chroma to reframes “bigger is better” for time-series FMs by replacing one large monolith with a portfolio of small specialists. Starting from a generalist base model, the authors fine-tune disjoint subsets to create diverse experts, then use model selection or ensembling at inference. Across standard benchmarks, this portfolio matches large models with far fewer parameters and is more compute-efficient than test-time fine-tuning, suggesting a simple, scalable alternative that may generalize beyond forecasting.

### Strengths
Strengths:
1. The problem is well motivated and of interest in the community.
2. The empirical study is comprehensive. The authors investigate different portfolio design and the combination methods. One thing I personally like is that, from Table 2, the findings of simply using ensemble cannot help improve performance.
3. Technical contributions are somewhat limited. But I feel the problems is still worth investigated from this perspective.

### Weaknesses
Weakness:
1. The proposed ensemble method is a bit hand-wavy. It shows the good performance but doesn't provide insights on why/how it works. Althrough some ablation studies are provided, I still wonder, in general, what kind of ensembling methods can help to improve the performance and any principles behind that. If we have this kind of insights, that would bring this work to a better level. 
2. I am not sure if the conclusion can extend to other model architecture. In this paper, it only uses chronos-bolt which is encoder-decoder architeture. But other time series foundation models use decoder-only (e.g., TimesFM) or maksed-encoder (e.g., Moirai). I think that if only encoder-decoder architecture is considered, maybe it is better to adjust some of the claims and the conclusion to limit the model type.
3. For time series foundation models with MoE, there are some other works. For examples, time-moe and moirai-moe. May also consider has some discussions on these in related works. 

References 

[1] Time-MoE: Billion-Scale Time Series Foundation Models with Mixture of Experts. ICLR 2025

[2] Moirai-MoE: Empowering Time Series Foundation Models with Sparse Mixture of Experts. ICML 2025

### Questions
Questions:
1. post-training helps to reduce training cost but I wonder if it reduces the diversity. Because if we train the specialist models from scratch, the specialist models may learn more on the unique pattern in different subsets. Particularly, if we train from scratch, could we get better performance? Or only similar performance but with significantly more computational cost. 
2. How to determine the validation sets, especially if we don't have enough data? Additionally, when we only want to make a forecast for a single time series instead of a whole evaluation dataset, how we can do the forecast combination at test-time? 
3. I notice that 1K gradient steps seems sufficient for the fine-tuning (i.e., post-training) and further increases this doesn't help. I wonder if this is because of the data amount used in post-training or any other possible reasons? 
4. I wonder if the scaling behavior holds if we further increase the model size to 200M (the size of chronos-bolt-base).

### Soundness
3

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
The paper investigates whether building portfolios of smaller pretrained models can match or surpass large monolithic time-series foundation models (e.g., Chronos-Bolt) under constrained compute budgets.

### Strengths
- Elegant and Compute-Efficient Framework. The proposed portfolio of specialists provides a pragmatic alternative to large models, achieving substantial compute savings. The post-training step is extremely lightweight (only 1K updates), making the approach appealing for practitioners with limited resources.
- Strong Empirical Insights. The work provides valuable empirical findings: ensemble diversity is critical for the forecasting tasks and ensembling yields a favorable accuracy-compute trade-off close to the efficiency frontier.
- Potential for Broader Impact. By reframing scaling from “bigger models” to “smarter test-time computation,” the study could inform future directions in efficient time-series foundation modeling and resource-adaptive forecasting systems.

### Weaknesses
1. **Validation data at test time is impractical and misaligned with typical deployment.** The method assumes the availability of non-trivial validation data at test time to select or weight experts. In realistic forecasting deployments, only inputs (and possibly a short warm-up context) are available. Labeled validation slices are rare or prohibitively small. This creates a distributional and resource gap between the paper’s evaluation protocol and common practice.
2. **Zero-shot and OOD forecasting ability are not validated.** Although the paper positions the approach against “zero-shot” foundation models, the core mechanism, model selection or greedy ensembling using validation losses, consumes task labels and thus departs from zero-shot forecasting. **The zero-shot experiments are highly recommended, especially when the test data have a significant domain shift with the validation data.**
3. **Greedy ensemble selection risks overfitting to the validation window.** The ensemble is built by iteratively adding experts to minimize validation error. Repeated selection against a small, local window is prone to selection bias and variance inflation, particularly with correlated experts. We may need further experiments to show that this method has resistance against overfitting
4. **Scaling-law claims rest on a narrow 1–9M parameter band.** Figure 3 supports “comparable scaling to single models” only in a small-model regime. Extrapolating this trend to ≥100M/1B parameters is not justified without additional anchor points, thus being unconving.
5. **Expert construction is largely manual and metadata-dependent.** Experts are partitioned by frequency/sector categories, and those categories are highly dependent on the manual design. Hence, it is unclear how this manually designed expert construction performs on the unseen domain/tasks.
6. **Experimental gains are modest and sometimes negative.** Across benchmarks, the ensemble often only slightly outperforms the best single expert. On Chronos, the 4M-parameter ensemble even underperforms the top single model.

### Questions
See the weaknesses

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper investigates fine-tuning/post-training of time series foundation models. Rather than using pre-trained TSFMs for zero-shot forecasting directly, they propose to specialize a base model into different groups (can be flexibly defined, but the paper experimented with domains and frequency groupings). Given a group of models, a single forecast can be obtained either by model selection, or model ensembling. Experiments are done to identify the best approach. Results show that an ensemble of small models are on par with much larger models.

### Strengths
This is a very interesting investigation into the capabilities of time series foundation models. Paper is well written and well positioned in the literature. The paper proposes an innovative method to utilize pre trained time series models, and performs extensive experiments, yielding useful insights.

### Weaknesses
* Scaling results are not strong. There are likely too few data points across the scaling axis to reliably trust the results. For frequency specialists, 4m models are consistently stronger than tiny (9m) models. There is no clear indication we should expect the regression to extrapolate. This is also reflected in the main results figure, leading to some confusion when one is reading about scaling behavior but in the main results, 4m is clearly stronger than 9m.
* It is unclear what "active parameters" refers to in this setting. From my reading I understand active parameters for an ensemble of 4m models to be 4m. However, I would think that for model ensembles, the number of active parameters should be ensemble size * model size
* Omits more recent models with much stronger results, e.g. Toto, Tirex, TimesFM 2.5

### Questions
* if model selection/ensembling is selected using the first evaluation window, how are the metrics for this first evaluation window obtained without using the ground truth values? Shouldn't the window before the first evaluation window be used for selecting the model combination method instead?
* Is the same model combination selection method also used for other experiments, e.g. scaling plot?

### Soundness
3

### Presentation
3

### Contribution
4
