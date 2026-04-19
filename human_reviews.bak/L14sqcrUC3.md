# TabReD: Analyzing Pitfalls and Filling the Gaps in Tabular Deep Learning Benchmarks

- Decision: Accept (Spotlight)
- Scores: 3, 8, 8, 8

## Abstract
Advances in machine learning research drive progress in real-world applications. 
To ensure this progress, it is important to understand the potential pitfalls on the way from a novel method's success on academic benchmarks to its practical deployment. In this work, we analyze existing tabular deep learning benchmarks and find two common characteristics of tabular data in typical industrial applications that are underrepresented in the datasets usually used for evaluation in the literature.
First, in real-world deployment scenarios, distribution of data often changes over time. To account for this distribution drift, time-based train/test splits should be used in evaluation. However, existing academic tabular datasets often lack timestamp metadata to enable such evaluation.
Second, a considerable portion of datasets in production settings stem from extensive data acquisition and feature engineering pipelines. This can have an impact on the absolute and relative number of predictive, uninformative, and correlated features compared to academic datasets.
In this work, we aim to understand how recent research advances in tabular deep learning transfer to these underrepresented conditions.
To this end, we introduce TabReD -- a collection of eight industry-grade tabular datasets. 
We reassess a large number of tabular ML models and techniques on TabReD. We demonstrate that evaluation on both time-based data splits and richer feature sets leads to different methods ranking, compared to evaluation on random splits and smaller number of features, which are common in academic benchmarks. Furthermore, simple MLP-like architectures and GBDT show the best results on the TabReD datasets, while other methods are less effective in the new setting.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper discusses two major issues within the area of tabular data: time-shifting and rich feature datasets, both of which are often underrepresented. It mentions 100 datasets from academic benchmarks to highlight these problems. The author argues that both issues can lead to decreased performance and remain largely unexplored.

To address these challenges, the author introduces a new framework called TabReD, designed for conducting experiments in this specific context. The experiments utilize eight datasets that represent real-world data. Additionally, the paper compares TabReD to a framework developed by Gorishniy. The results indicate that TabReD can effectively serve as a framework for these experiments.

Furthermore, the findings reveal that the results of the time-shifting experiments exhibit higher variance.

### Strengths
The paper shows experiments on 8 different databases representing real-world scenarios. The experiment results are thoroughly presented and explored, The paper provides up-to-date comparisons with other major papers.
The code provided are very useful.

### Weaknesses
Some terms are missing, and the paper is not self-contained. Take data leakage, for example. The writer cites another paper, which I believe is the writer's paper, but does not give a glimpse of the definition. 

One of the main arguments is data leakage; however, only 11 (10%) were detected among 100 datasets, which is not significant.

The main issues are domain shifting and time shifting. Both areas are not new and have been explored very well. Domain shifting is also called an out-of-domain or out-of-distribution Problem. This problem is not new and is well-known within academic research.

For the second problem, time shifting split is a vague problem. There are reasons why random split is widely used. The main reason is that a random split is expected to have a similar distribution, unlike a shifting split. The result of a model trained in a time-shifting split is expected to vary between experiments; this is just because it came from a different distribution. 

TabRed, we do not need to reinvent the wheel for each problem. Gorishniy's framework is more than enough for the experiment. If your focus is on the above problems, then you can just adjust the Dataloader class from Pytorch. 

While there were some models inside, the comparison was made only to one single framework. The related work mentioned more than one. Some experiments may be helped to pursue reader the importance of TabReD.

The paper has many details, but they are written within the appendix. The main pages lack technical details. To understand the paper completely, the 10 main pages are not sufficient. This is actually a very long paper.

### Questions
What is the definition of data leakage, and why is it a problem?
Similar distribution with training and testing is not necessarily called data leakage.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
Motivated by the current landscape of academic benchmarks for tabular data, this work aimed to explore the limits of such benchmarks and provide an alternative. In detail, the paper proposes eight new datasets as part of a benchmark called TabReD. Moreover, the paper analyzes several prior benchmarks for tabular data and evaluates deep learning vs. classical machine learning methods on TabReD.

The contributions of the paper are:
* Analysing prior tabular benchmarks that focused on IID* tasks. 
* TabReD with datasets originating from Kaggle and newly sourced from the industry, focus on non-IID with a temporal shift. 
* Evaluating classical machine learning, recent deep learning methods, training methodologies, and MLP ensembles on TabReD.
* Highlight the difference between the conclusion from the result on TabReD and one from one prior benchmark.
* Highlight the difference between time-based (no IID assumption) and random-based (IID assumption) splits. 

\* In my wording, I use IID (or iid) to refer to machine learning tasks applied to data from independent and identically distributed random variables. Likewise, this makes the ML task predict by interpolation instead of extrapolation.

### Strengths
The paper tackles the important problem of tabular benchmarking and its limits in academic practice. Tabular predictive modeling for non-iid tasks (such as under a temporal shift) is rarely correctly discussed and largely overlooked in our field. Thus, I see this work as eye-opening to many academics and hopefully inspiring better benchmarking practice in the future. Furthermore, providing reliable and usable benchmarks for this problem type is helpful for future researchers, as well as helping us to determine trends that generalize across task settings (e.g., PLR working well for non-iid and iid data).

The related work section is well-written and sufficiently exhaustive to provide the correct context. It fits the framing provided in this work.

Given the experimental design and datasets, the results look reasonable and, from the code, reproducible. The observation that traditional boosting outperforms recent tabular deep learning approaches matches prior (big) benchmarks. Moreover, the power of ensembling also fits into this picture. Section 5.3 is a highlight among the results and an exciting avenue for meta-studies. Section 5.4 is interesting and useful empirical evidence but not surprising or unexpected.

The experimental design for training, tuning, and running baselines is mostly in line with prior work on deep learning vs. classical machine learning. Thus, it seems to be accepted by the community as sufficient. In other words, while there is a lot of room for improvements (e.g., better search spaces or non-standard preprocessing for filling NaNs), this is, to me, not the core contribution of this work, and improving such points would likely not change the conclusions presented in the work.

### Weaknesses
I detail the weaknesses of this work in subsections, starting with the biggest weakness. 

## Discussion and Framing of non-IID vs. IID Machine Learning 

Prior benchmarks for tabular data that focus on classical machine learning vs. deep learning assume that the data is IID. Thus, the framing of this work seems problematic to me, as it (at least) implies that prior tabular benchmarks desire to determine how good tabular models are for non-IID data. 

I see where one might get this impression from in the prior classical machine learning vs. deep learning benchmarks, as they have little formalized assumptions about their data. Moreover, this is rarely mentioned or specified by method developers that use such benchmarks either. Nevertheless, from my perspective, the core assumption for many algorithms and research in this domain is still IID data. 

Consequently, this work positions itself as a solution to an ill-specified problem. If I am benchmarking for IID data, I might not care about performance for non-IID tasks such as the temporal shift (i.e., I do not see this as a gap that needs to be filled). In general, the need for time splits is a clear indicator that we might need to change methodology away from pure tabular IID-based machine learning and switch to (multivariate) forecasting with (many) exogenous variables. 

The paper crucially lacks a discussion about this perspective of IID vs. non-IID. The authors seem motivated by a real-world application that could have been better solved without pure-tabular IID-based models. To be more precise, the missing discussion is about which machine learning tasks we want to solve. Prior tabular benchmarks evaluate which models are good at interpolation, while this work wants to evaluate which models are good at extrapolation. But somehow, this distinction has not happened, confusing the results and conclusion. 

In summary, this paper lacks a discussion about IID vs. non-IID and should likely even include forecasting algorithms as a baseline (especially for the regression tasks; for classification, it gets more complicated). Moreover, the title and framing of the abstract/introduction could be adjusted such that this work is more about a new, specific benchmark than an alternative or extension of prior benchmarks. The results would fit such a framing even better. 

## Related Benchmarks

While the work correctly identifies the big benchmarks for deep learning vs. classical machine learning (minus too recently published works), it fails to include all big classical machine learning benchmarks (for IID data). This is problematic as several of the claims and results in this paper seem independent of the deep learning debate. 

To illustrate, the paper states "the landscape of existing tabular machine learning benchmarks compared to TabReD" in Table 1, which is categorically wrong. It would be more correct to claim "the landscape of existing tabular deep learning benchmarks compared to TabReD".

For general tabular benchmarking, sets like the AutoML Benchmark Suite [1], or TabRepo [2] are missing. While most of these also exhibit some of the problems detailed in this work, the AutoML Benchmark Suite is better curated than deep learning vs. classical machine learning benchmarks. In comparison, TabRepo shows how the correct search spaces, tuning, meta-learning, and ensembling can boost performance for tabular IID-based modeling. 

Finally, the field of time series forecasting offers many more benchmarks for predictive modeling with non-IID data and (slight) temporal shifts. For example, see the benchmark and datasets used in AutoML or foundation models for time series [3, 4]. 

This criticism is mostly related to framing and careful wording. The author should ensure that their work is clearly differentiated from such prior work or incorporate it into their analysis. 

## Kaggle Data

The paper presents data from Kaggle as the "holy grail" for real-world data and benchmarks. This is an overly positive framing, given that Kaggle has many problems with its data and also hosts synthetic competitions. Specifically, the paragraph about Kaggle in the related work section (Line 113) lacks any critical reflection of data from Kaggle. To give some examples, data from competitions on Kaggle can be duplicates, sourced from UCI or OpenML, purely synthetic data, badly preprocessed, or uploaded with a data leak (see [5] for a related and extended discussion). 

Furthermore, sourcing data from Kaggle competitions and employing the top-performing preprocessing pipelines has two critical problems not discussed in this paper: 1) the pipeline is selected on test performance (leaderboard score) and thus leaking or overfit; 2) we do not have the target labels for the test data. 

\2) is specifically bad if, for example, the data only exhibits a crucial distribution shift in the test (or a temporal shift). Then, determining how good a model is for the training data is not representative of the real-world task that is to be solved by the competition. Likewise, the pipeline selected for preprocessing might not be representative of the training data or comparing models on this data. 
 


# References 
1. Gijsbers, Pieter, et al. "Amlb: an automl benchmark." Journal of Machine Learning Research 25.101 (2024): 1-65.
2. Salinas, David, and Nick Erickson. "TabRepo: A Large Scale Repository of Tabular Model Evaluations and its AutoML Applications." arXiv preprint arXiv:2311.02971 (2023).
3. Shchur, Oleksandr, et al. "AutoGluon–TimeSeries: AutoML for probabilistic time series forecasting." International Conference on Automated Machine Learning. PMLR, 2023.
4. Ansari, Abdul Fatir, et al. "Chronos: Learning the language of time series." arXiv preprint arXiv:2403.07815 (2024).
5. Tschalzev, Andrej, et al. "A Data-Centric Perspective on Evaluating Machine Learning Models for Tabular Data." arXiv preprint arXiv:2407.02112 (2024).

### Questions
* Why does the paper only show 36 instead of 176 datasets for TabZilla? 
* Several interesting aspects could be added to the overview of the new datasets: What year are the new datasets from? What task (classification, regression, forecasting) are they intended for? Which metric is used to measure performance for these datasets? What are the features about? 
* The appendix does not provide enough detailed information about the new datasets and how they were preprocessed. Moreover, it is unclear which feature pipeline was chosen for Kaggle datasets. Such information are important and would be needed for something like dataset cards. 
* Why is there no ensemble of classical machine learning methods? And what kind of ensemble is it for the MLP?
* What is the variance of the scores across seeds? Why is no backtesting for time splits considered (akin to cross-validation)?  
* I seem like the plots in Figure 2 are needlessly complex to read for the point that is made. Maybe replace the metric with the rank to make the message clearer? (also note the arrows for higher/lower is better are pointing in the wrong directions right now)

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper identifies and addresses important gaps between academic tabular ML benchmarks and industrial applications. The key contributions are:
* Literature Cleanup: Analysis of 100 existing benchmark datasets, revealing issues like data leakage and underrepresentation of real-world conditions -- primarily (a) lack of temporal drift, and (b) small feature space.
* New Benchmark: Introduction of TabReD, a new benchmark of 8 industry-grade datasets focusing on temporal drift and feature-rich scenarios. Datasets are sourced from Kaggle (4) and proprietary internal company data (4).
* Evaluations: Comprehensive evaluation showing that most recent tabular DL advances do not transfer to these more realistic conditions, and that gradient boosted trees and simple MLPs tend to perform best

### Strengths
The paper addresses a critical gap between academic benchmarks and industrial applications of tabular ML, with clear evidence from an analysis of 100 existing academic datasets that prior work is limited by (a) data leakage, (b) lack of temporal splits, (c) containing mostly non-tabular data, such as flattened images.
* Very rigorous empirical methodology:
 * Clear criteria for dataset selection
* Statistical significance testing across 15 random seeds for evaluations
* Comprehensive evaluation of multiple model architectures and techniques

### Weaknesses
* No detailed analysis of failure modes for retrieval-based models
   * Possible future extension to important domains — not fully representative (healthcare, science)
   * The authors do explicitly mention this as a specific limitation
* Given the emphasis on industrial applications, it would be nice to have a sense of how impactful these performance shifts are for the downstream tasks, i.e. can you give us a sense of how much 0.01 RMSE matters for cooking time estimation dataset?
* Unclear comparison to prior benchmarks on how your benchmark yields improved model rankings. A central claim of this paper is that prior benchmarks yield model rankings that change once temporal splits are taken into account. Hence, your arguments about XGboost and MLP being surprisingly superior. But you don’t specify how the models that you tested would have been ranked by the prior work benchmarks (e.g. Tabzilla, WildTab, etc.) that lack temporal splits. You get at this a bit in Figure 2 within your own benchmark, but it may help strength the argument to a more detailed comparison between the conclusions of your Table 3 to the rankings that would have been generated by running these models on the prior work instead.

### Questions
* How does the subsampling of larger datasets affect the conclusions? Could you provide a sensitivity analysis?
* How did you determine data leakage issues that weren't reported in prior literature?
* Could you elaborate on the feature engineering process for the new datasets and how it reflects industry practices?
* Why did you choose not to include in TabReD the existing datasets where temporal splits are possible due to the presence of temporal information?
* Provide recommendations as to how TabReD should be used with existing benchmarks.

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The present paper studies the characteristics of academic tabular benchmarks for deep learning models, and makes the following contributions:

- The authors identify several issues that hinder the reliability of such benchmarks for real-world applications. In particular, they show that 1) time-evolving datasets (where the data distribution changes with time) and 2) feature-rich datasets (e.g. stemming from feature engineering pipelines) are under-represented in academic benchmarks, while they are common in industrial scenarios.

- To address these pitfalls, the authors introduce TabReD, a collection of 8 industry-grade tabular datasets from various domains, and show that they exhibit the 2 aforementioned aspects.

- Finally, they evaluate several tabular models on TabRed. Results seem to show that top-performing architectures in academic benchmarks do not transfer well to TabRed, while simpler models (e.g. GBDT or MLP ensembles) remain performant. The authors attribute this observation to the temporal data drift present in the datasets, and the fact that they have large number of correlated/uninformative features.

### Strengths
1) The paper is clear and easy to read
2) The paper has practical implications. Benchmarks are an important aspect of research and limitations of standard ones for tabular learning are demonstrated. The TabReD benchmark can fill this gap by being closer to real-world/industry settings.
3) The claims are overall well supported by experiments and analyses, although several things could be improved (see weakness section):
    - The authors conduct a extensive analysis of 100 datasets used in common tabular learning benchmarks. A significant fraction of them exhibit data quality issues such as leakage, untracability, or a non-tabular format. Many of them are also potentially subject to temporal data drift, but this aspect is overlooked in most benchmarks.
    - The authors provide 8 datasets (including 4 novel ones) from real-world, industrial applications. Figure 3 and 4 confirm that they exhibit temporal data drift, and have large number of features with complex correlation patterns (unlike academic datasets).
    - The benchmark on TabReD includes a wide range of approaches: baselines, top-performing models in academic benchmarks, and training techniques that were also shown to improve accuracy in academic benchmarks. The results seem to show that certain models/techniques which performed better in academic benchmarks actually perform similarly or worse than simple baselines (Fig 1 and Table 3).
    - The authors partly explain these differences with the temporal data drift present in industrial datasets. For this they compare two data splits scenarios (time-based and random, see Figure 2), and show that random splits overestimates model performance (both in mean and std across seeds). They also seem to alter model ranking, although this is not clear.

### Weaknesses
1) The average ranks shown in Table 3 are a bit hard to intepret because the methods are often very close in terms of performance, so it is not clear how much difference there is between the worst and best approaches. Figure 1 partly solves this by showing the average percentage of improvement over a MLP baseline, but doesn't show it for all methods. It would be nice to have this information either in Figure 1 or as another column of Table 3 in order to get a better feeling of how the methods perform.
2) Even if recent tabular DL methods seem to perform worse than simple baselines on TabReD, I am not fully convinced that it is due to 1) temporal data drift or 2) datasets with large number of correlated/uninformative features, as the authors imply. Knowing this is important to allow further research to focus on the right aspects when designing novel methods. It would also motivate the choices the authors made when they designed the TabReD benchmark.
    - For instance, Figure 2 compares the performance of a few methods when doing time-based versus random splits of the data. As expected, performance decreases when using time-based splits, but the impact on the models ranking and relative gap is not clear. For instance, TabR seem to perform poorly in both time-based and random splits. In addition, most of the methods considered in Table 3 are not included. I would like to see a table with aggregated results for most methods, such as the average ranking in both splitting scenarios, and the average percentage decrease in performance when going for random to time-based splitting. This will allow to assess the effect of temporal data drift more easily, and see if the baseline models are indeed more resilient to it.
    - The impact of having datasets with large numbers of correlated/uninformative features is not explored. I would like to see a analysis similar to the one done for splitting strategies. For instance, the authors could apply simple filtering techniques to remove highly correlated and/or uninformative features, and compare the average ranks in the "raw" versus "filtered" scenarios, and show the average percentage of improvement per model.

### Questions
1. The meaning of needed/possible/used in Table 1 wasn't clear with just the caption. It would be good to have a sentence to explain it in the caption.
2. When calculating model rankings in Table 3, the Appendix (line 902) explains that a model B is ranked better than model A if 1) B is in average better (across 15 random initializations), and 2) the gap in average performance is smaller than the standard deviation of model B performance. First, I guess the authors meant larger instead of smaller? Second, it seems that with this definition the ranking isn't transitive. For example, imagine three models: model A with average R2 score = 0.6 (std = 0.1), model B with average R2 score = 0.4 (std = 0.2) and model C with average R2 score = 0.55 (std = 0.2). In this case model A should be ranked higher than model B and ranked the same as model C. However model C should also be ranked the same as model B, so it is confusing. I maybe misunderstood something here, so I would like to have more details on that.

### Soundness
3

### Presentation
3

### Contribution
3
