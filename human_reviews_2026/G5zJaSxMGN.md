# Large-Scale Pretraining Offers Modest Benefits for Tabular Transfer Learning

- Decision: Reject
- Scores: 6, 6, 2, 2

## Abstract
Several recent works seek to train foundation models for tabular prediction by pretraining neural networks on large collections of tabular classification and regression datasets. These tabular foundation models (TFMs) are often reported to outperform non-pretrained baselines when applied to predictive tasks on unseen tables, demonstrating effective tabular transfer learning. In this paper, we show that, in contrast to the positive conclusions of prior works, the perceived performance benefits from large-scale tabular pretraining largely diminish when we aggregate the results across datasets while (i) preserving the performance differences between models in their original scale (e.g., without min-max normalization); and (ii) testing for the statistical significance of these differences. For example, when we replicate the original evaluation setup for TabPFN-v2 on classification tasks, TabPFN-v2 indeed achieves the highest average min-max normalized AUROC, but reaches a statistical tie with CatBoost in 69% of all datasets, while significantly outperforming it in 20.7% of datasets and underperforming it in the remaining 10.3% of datasets. We evaluate seven open-source TFMs on 88 classification and 82 regression datasets in both full-data (i.e., using all training examples) and few-shot settings, and find that existing TFMs only show statistically significant improvements over non-pretrained baselines on small classification datasets, with no consistent gains in other settings. To isolate the impact of tabular pretraining, we also compare three TFMs directly to their non-pretrained counterparts, and find that, in most cases, the performance gains from pretraining are minimal. Our findings suggest that current approaches to large-scale tabular pretraining may offer limited performance benefits, showing room for improvement in methods for effective tabular transfer learning. For standardized and reproducible evaluation of TFMs, we release our evaluation suite as the TFM Evaluation Harness.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper examines the tabular foundation models (TFMs) and their claims to be better than tree-based baselines or the non-pretrained baselines. The authors used comprehensive experiments to show that some of the claims TFM papers made were slightly misleading. The authors used the original scale and tested for statistical significance and found a different conclusion. The paper concluded that TFMs only offer modest performance gain even from large scale pre-training.

### Strengths
### Originality and Significance
1. The main claim of the paper is original and has not been discussed rigorously in the past
2. The paper offers comparison of most popular and recent TFMs which other studies have not done

### Quality and Clarity
1. The paper is well written and very clear throughout
2. The experiments are well designed. The use of bootstrapping for statistical significance is useful.

### Weaknesses
1. The paper has not examined the computational complexity of each method. 
* The overhead of selecting hyper-parameters for baseline methods might be higher than the TFM methods if not considering the pre-training cost of TFMs.
2. There is no study on fine-tuning or other adaptation methods such as feature ensembling used by TabPFN

### Questions
1. Is the validation set passed to TFMs as context?
2. Do the TFM models fail to the baseline on the same datasets?
3. Will there be a rule of thumb for choosing which method to use based on the dataset?

### Soundness
4

### Presentation
4

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
The authors evaluate a range of tabular foundation models with a focus on critically evaluating performance differences between methods, challenging previous papers that showed TFMs significantly outperforming baselines. They argue that their results suggest limited benefit to large-scale tabular pretraining.

### Strengths
- The paper points to a common shortcoming of the tabular modelling literature in that metrics are preferentially chosen that emphasize the difference between models which often don't have large differences in absolute performance.
- Having an independent and critical evaluation of tabular models is a useful contribution.
- A wide range of interesting experiments are provided with applicability to different tabular settings.

### Weaknesses
- Most tabular foundation models under evaluation don't perform large-scale pretraining in a way that is comparable to vision and language models, making the framing of the paper somewhat overstated. TabDPT, XTab, and TP-BERTa don't perform large-scale pretraining at all, only using 123, 202, and 52 tables for pretraining respectively. TabPFNv2 and TabICL were trained at scale but only on synthetic data. CARTE pretraining is on graph relationships rather than tabular data.
- The core point of the paper is rather subtle and it could be presented with more rigour. The authors refer to "the statistical significance of the extent of improvement" and "whether the absolute gains in performance themselves are statistically significant", but this differs from the standard usage of statistical significance - either differences are statistically significant at some level or they aren't. The paper prefers certain approaches for evaluating statistical significance but it's not clear that this invalidates the statistical significance evaluations in existing tabular modelling papers.
- The evaluations deliberately skip preprocessing hyperparameter tuning for in-context learning models where available but perform extensive hyperparameter tuning for other models, which could account for some of the reduced gap in performance. I'm not convinced this is fair, and it wouldn't be applicable to a practitioner seeking the most performant model.

**Minor**

- Typo line 238: "classicial"

### Questions
- Could you clarify how the "best baseline" was selected in results that include it? If this was selected based on the test data that's being shown, then it effectively means model selection on test data, which isn't a statistically fair comparison.
- What is the interpretation of models showing statistically significant differences when min-max scaling is applied but not when it isn't? These both seem like valid comparisons for testing significance, even if differences are more or less emphasized. Similarly, how should we interpret a significant difference being shown by a Wilcoxon signed-rank test but not by this paper's bootstrapping procedure?
- In your "non-tabular" exclusion, how did you treat datasets that originally come from non-tabular data such as images, but used extraction methods to produce numeric summary features?

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
The paper empirically evaluates accuracy and generalisation properties of leading tabular foundation models. Authors curate 88 classification and 82 regression datasets and show that in some settings the FMs do not outperform classic baselines such as XGBoost.

### Strengths
Authors conduct a large and comprehensive empirical comparison, benchmarking leading TFMs and baselines. Multiple generalisation settings are evaluated including full training data and few shot. Some conclusions are important for future work and fair evaluation of TMFs, in particular the difference between statistics significance in the min-max/rank setting vs original space. The evaluation suite is publicly released.

### Weaknesses
Authors make a number of design choices that are not well justified such as z-score normalisation, mean/mode imputation, and removing regression target transforms. Furthermore, internal feature pre-processing steps for leading TFMs are also shut off to make to comparison "fair". I don't see how this makes the comparison fair, if a TFM is pre-trained with these feature transforms then shutting them off during inference will inevitably hurt performance. Instead, I think the same transforms should be made available to all baselines and swept over as part of the hyper-parameter search.

Paper also make a number of strong statements that are not well justified such as "simply scaling pretraining
over a diverse collection of tabular datasets may offer limited performance benefits". From Figure 1b, TabPFN-v2 still has double the win rate over the leading CatBoost baseline *without* re-training where as CatBoost is trained from scratch and heavily tuned on each dataset. I think the generalisation benefits of pre-training are very clear even under this evaluation setting.

Overall, while the paper shows some interesting findings, I think the experimental settings disadvantage TFMs and overly strong conclusions are drawn from the results.

### Questions
Do you have results when feature transforms are turned on (and swept over) for TFMs? These transforms can't be shut off if models were trained with them.

### Soundness
3

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper evaluates seven open-source tabular foundation models (TFMs) on 88 classification and 82 regression datasets, in both full-data and few-shot regimes, to check whether the performance gains reported in prior work actually hold under stricter evaluation. It shows that common practices—min–max normalizing metrics and only doing rank-based tests—can overstate the benefits of tabular pretraining; when performance is kept on the original scale and per-dataset bootstrap significance is applied, TFMs are usually tied with strong tree/NN baselines like CatBoost, only clearly winning on small classification tables.

The authors also compare three TFMs (TabuLa-8B, TP-BERTa, XTab) with their non-pretrained counterparts and find that pretraining mainly helps LLM-style tabular ICL, but brings little or no benefit to other architectures, so the overall gains from large-scale tabular pretraining are modest.

### Strengths
1. Clear problem framing. The paper explicitly asks: “Do reported gains from tabular pretraining really hold up under stricter, per-dataset evaluation?”—a useful and timely question for the community.

2. Evaluation is careful and more realistic. Using original-scale metrics + per-dataset bootstrap CIs + win/tie/loss tallies is stricter than most prior TFM papers and exposes overclaiming.

3. Solid experiments covering sufficient classification and regression datasets.

### Weaknesses
1. Current TFMs are not a faithful proxy for “large-scale pretraining” on tabular data. The paper’s main takeaway is that existing tabular foundation models (TFMs) do not clearly outperform the best overall (highest Elo) baselines. However, this does not necessarily imply that the learning paradigm of large-scale pretraining “fails” on tabular data. A more cautious interpretation is that the current crop of TFMs is not yet able to exploit large-scale pretraining as effectively as models in vision or NLP. This gap may be attributable to architectural and methodological limitations rather than to an inherent ceiling of tabular pretraining itself.

2. The paper acknowledges successes of pretraining in CV and NLP, but it does not sufficiently connect these successes to the fact that those domains benefit from strong, well-understood inductive biases (locality, translation equivariance, token order, etc.). In tabular learning, such inductive biases are notably weaker or dataset-specific, making it harder for current TFMs to turn scale into generalization. For instance, TabPFN-v2 can be viewed as a relatively brute-force design that attends over samples and features via full self-attention without structural pruning, whereas in vision and language we routinely inject structure (e.g. shifted windows in Swin-Transformer, sparse / block attention in many LLMs). The paper does not explore whether better inductive structure—rather than “more data”—would change the outcome.

3. The paper reads more like a performance audit than a path forward. The four findings are largely descriptive—they carefully document that the reported advantages of TFMs shrink when evaluated per-dataset with proper uncertainty estimates. This is valuable, but the paper stops one step early: it does not turn these observations into concrete design guidance on how to build TFMs that actually benefit from scale on tables.

4. In the head-to-head comparison (TabuLa-8B, TP-BERTa, XTab vs. their non-pretrained counterparts), the paper concludes that pretraining mainly helps LLM-style models on in-context tabular tasks, but not other models. However, XTab does not actually support in-context learning in the same sense: it is pretrained across many datasets but still requires fine-tuning, and its featurizers and projection heads are data-specific, with only the transformer backbone being transferable. Putting XTab in the same “ICL-capable TFM” bucket risks conflating “multi-dataset pretraining” with “true ICL (train+query in the prompt)”, and thus weakens the generality of the claim in Finding 4.

### Questions
See weaknesses

### Soundness
2

### Presentation
2

### Contribution
1
