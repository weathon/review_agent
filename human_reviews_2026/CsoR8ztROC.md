# TimeRecipe: A Time-Series Forecasting Recipe via Benchmarking Module Level Effectiveness

- Decision: Accept (Poster)
- Scores: 8, 6, 2, 4

## Abstract
Time-series forecasting is an essential task with wide real-world applications across domains. While recent advances in deep learning have enabled time-series forecasting models with accurate predictions, there remains considerable debate over which architectures and design components, such as series decomposition or normalization, are most effective under varying conditions. Existing benchmarks primarily evaluate models at a high level, offering limited insight into why certain designs work better. To mitigate this gap, we propose TIMERECIPE, a unified benchmarking framework that systematically evaluates time-series forecasting methods at the module level. TIMERECIPE conducts over 10,000 experiments to assess the effectiveness of individual components across a diverse range of datasets, forecasting horizons, and task settings. Our results reveal that exhaustive exploration of the design space can yield models that outperform existing state-of-the-art methods and uncover meaningful intuitions linking specific design choices to forecasting scenarios. Furthermore, we release a practical toolkit within TIMERECIPE that recommends suitable model architectures based on these empirical insights.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
TimeRecipe is a benchmarking framework designed to systematically evaluate time-series forecasting methods at the module level, rather than just assessing complete end-to-end models. This paper is well-written and easy to follow and reproduce, providing interesting view for designing/selecting TSF methods/modules.

### Strengths
1. It pioneers a systematic evaluation focused on the effectiveness of individual forecasting model components (modules), addressing a gap left by existing benchmarks that primarily assess entire models.
2. the study is built upon a large-scale experimental setup, encompassing over 10,000 experiments that cover numerous module combinations.
3. this benchmark successfully identifies module combinations that outperform established SOTA models in over 90% of evaluated scenarios , achieving an average error reduction. Crucially, it provides practical insights by correlating module effectiveness with data properties and offers a toolkit for model selection.

### Weaknesses
1. The connections drawn between module effectiveness and data characteristics are based on extensive empirical results and statistical tests, the paper provides limited theoretical analysis or causal explanation.
2. The benchmark focuses solely on predictive accuracy and completely omits analysis of computational efficiency. This oversight limits the benchmark's practical applicability, as users cannot assess the crucial accuracy-efficiency trade-offs needed for real-world deployments, especially in resource-constrained or latency-sensitive scenarios. Incorporating complexity analysis, like in LightCTS, would significantly enhance the framework's value.

### Questions
Please see the weaknesses.

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
4

### Summary
This paper introduces TimeRecipe, a modular benchmarking framework for time-series forecasting that evaluates the effectiveness of common architectural components (e.g., normalization, decomposition, embedding, fusion, feed-forward modeling) across a wide range of forecasting tasks, horizons, and datasets. By systematically exploring over 10,000 combinations, TimeRecipe identifies component-level contributions to performance, offers statistical correlations between data properties and module choices, and supplies a training-free recommendation toolkit for model selection. Extensive results demonstrate that exhaustive module-level exploration can yield architectures that outperform existing state-of-the-art (SOTA) methods and provide practical guidance for real-world forecasting.

### Strengths
1、The paper introduces a new paradigm for time-series forecasting benchmarking by breaking down models into five core modules and systematically benchmarking their combinations. 

2、There are quite a few nice illustrations.

3、 This work focuses on an important problem that could have real-world applications.

4、 The figures and tables used in this work are clear and easy to read.

### Weaknesses
1、While the coverage of LTSF, PEMS, and M4 datasets is excellent, novel datasets introduced (e.g., unemployment forecasting from Time-MMD) are only briefly mentioned and lack rigorous description (see Section 4.2 and Appendix B). For maximal transparency, the properties, preprocessing, and evaluation setup should be as detailed for these new datasets as for the standard ones.

2、While Table 2 and Figure 1 are helpful, many of the empirical summaries require close reading to decipher key findings. It would be helpful if the text more directly guided the reader through takeaways from these visualizations; for example, highlighting trends or “rules” in Table 2, or explicit failure cases identified in the configuration tables.

3、A large combinatorial space is searched, but more discussion of robustness to random seed, hyperparameters, or the presence of multiple optima would be valuable. Are the best configurations stable across resamplings—or is performance sensitive to chance initialization? Furthermore, are error magnitudes meaningful under operational constraints for real-world forecasting tasks?

### Questions
1、How robust are the results to changes in hyperparameters, default training budgets, or random seeds? Are the observed improvements for best configurations consistent under different parameterizations?

2、For the model selection toolkit: How does the training-free strategy compare to stronger meta-learning or AutoML approaches? Is LightGBM sufficient, or do more advanced selectors close the gap further to the global best configurations?

3、Can the authors provide additional clarity or pseudocode for the inter-module dimension handling, especially for embeddings (patch vs. invert) and their compatibility with MLP/Transformer/RNN blocks?

4、Is there a sense in which the best module combinations generalize beyond the datasets tested, or are they highly scenario-dependent? Are any “failure” cases (where TimeRecipe-selected models underperform SOTA) documented, and why do they arise?

### Soundness
3

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
The paper advances a module-centric framework for time-series forecasting. Specifically, the authors decompose canonical architectures into pre-processing, embedding, backbone, fusion strategies, and projection/post-processing. Building on this decomposition, the authors conduct a large-scale benchmarking study that enumerates and evaluates numerous module combinations across diverse tasks and datasets. The study yields statistically grounded mappings from data characteristics, such as trend strength, seasonality, and cross-channel correlation, to appropriate module choices.

### Strengths
1. The paper is well written and easy to understand.
2. The mapping from measured properties to module choices is an interesting idea and worthy of investigation.
3. The training-free selector is a pragmatic contribution that can reduce exploration cost.

### Weaknesses
1. Regularization, optimization, schedulers, and data augmentation are not systematically modularized, though they often rival architecture in impact. The historical-window length is also unclear, despite its significant effect on performance.
2. The choice of datasets and prediction horizons (e.g., 720) has been criticized by researchers as impractical in real-world settings (https://cbergmeir.com/talks/bergmeir2024NeurIPSInvTalk.pdf), which weakens the reliability of the conclusions and suggests the findings may be primarily academic.
3. Some intertwined cross-channel mixers and hybrid designs are only partially represented within the studied design space.

### Questions
1. Do the authors plan to modularize regularization, optimizers, schedulers, and augmentation so that training choices can be benchmarked with the same rigor as architecture?
2. How would this framework integrate with pretrained time-series backbones, for example, by swapping embeddings/fusion while freezing a large backbone, or by using the selector to choose adaptation routes?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper focuses on time series forecasting applications. Building TSFM requires many modules, and testing each module is an important task. The introduction section is written clearly - the 4 outcomes of the study.

### Strengths
The paper has conducted an extensive survey of reusable components and then performed a systematic study. The Table 2 is an output of such extensive study.

### Weaknesses
- Given a paper submitted on learning time series and dynamical systems, I feel the paper is more suitable for the benchmark and dataset track. Thus, I have started looking at the paper from a benchmarking and experimental task perspective. Why are the foundation models not part of this work?  

- Motivation. If I am a developer, how can I consume the outcome of your study? For example, can you provide a case study on how tord- get a leaderboatopping agent on GiftEval? GiftEval is a time series leaderboard, so it shows how much gain I have made using Table 2. 

- There is an alternate thought: I would go to Leaderboard, compile all the models listed there along with their performance, and then generate Table 2. How accurate can it be? In summary, can I generate the same table 2 but without going into the entire time recipe framework? Can you compare and contrast this alternate, more dynamic approach? 

- Hyperparameter tuning, I did not see how you capture the effect of parameter tuning? 

- What are the new datasets that studies bring and then demonstrate the value of learning?

### Questions
Please address all the weak points.

### Soundness
2

### Presentation
2

### Contribution
2
