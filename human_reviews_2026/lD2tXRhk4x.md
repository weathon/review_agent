# Temporal Misinformation Detection: Simple Ways to Improve Temporal Generalization and Better Evaluate Language Models

- Avg Score: 2.67
- Decision: Reject
- Scores: 4, 2, 2

## Abstract
Most of the current misinformation detectors display a lack of temporal generalization, despite the increasing reported scores in the literature. This can be attributed to classical machine learning evaluation protocols based on random splits. While widely adopted, these protocols often fail to reflect real-world model performance, a limitation that is particularly critical in misinformation detection, where temporal dynamics play a central role. In this paper, we present a comprehensive analysis of temporal biases across multiple misinformation datasets, with a specific focus on the temporal distribution of labels. We also introduce simple yet effective methods to improve performance in scenarios where temporal generalization is critical for NLP tasks. Our findings show that classical evaluation protocols tend to overestimate model performance in misinformation detection. To address this, we propose FC30, a new dataset, and introduce a general-purpose evaluation metric designed to better assess models under temporal shift and capture potential temporal bias.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper argues for the importance of taking temporal drift into account when studying (and especially evaluating) methods for detecting misinformation.  The paper includes analysis, specifically showing how taking the timestamps of benchmark data into account (train on earlier, test on later examples) leads to reductions in accuracy estimates.  The paper describes a new dataset designed to train on earlier and test on later instances, and introduces new methods based on BERT-style encoder models finetuned on the new dataset, showing how an ensemble of representations trained on temporally organized data and those trained on randomly split data can lead to a performance boost.

More details: 

This paper highlights that current misinformation detection models are not evaluated on their performance in real world settings where they may be used to detect misinformation about data that occurred after the model information cut-off date. The authors define temporal generalization as evaluation under three constraints: the data must be temporally ordered, the test data must be constrained such that the earliest item in the test data is later than the latest item in the train and validation splits, and the earliest item in the test data must also be later than the pretraining date of the evaluated model. The authors evaluate current misinformation benchmarks on their ability to correctly find misinformation under these practical constraints. They find that when using DeBERTaV3-Large, most misinformation detection datasets had a lower accuracy and f1-score in this realistic setting. The authors also propose a metric (LabDrift) for measuring whether a dataset is temporally biased. They create a measure of how well a model could do if they were to classify everything prior to a certain date as one label and everything after as the other label. The ratio between the real ordering of the data time stamps, and a perfectly separated ordering provides the LabDrift score. They find that most datasets have some level of temporal bias in the datasets, and provide a new dataset that has the lowest amount of temporal bias. This new dataset is covered by the three constraints that the authors defined, and covers 30 years of data. The authors further propose Fusion and TempoGen as approaches to improve temporal generalization. Both methods combine models trained on a random train/validation split and a temporally ordered train/validation split, where all validation data is later than all train data. While both approaches require twice the train compute, as two different models need to be trained, fusion uses both models at inference time as well, while TempoGen only requires the random model at inference time. Using an F1-score to compare, Fusion is the best approach across several different models, with TempoGen coming in second but with a lower inference cost.

### Strengths
The claim that temporal ordering of a dataset could lead to better real world outcomes is strong and sensible tests are conducted to support it

The paper tackles an important problem and draws attention to the temporal issue in dataset construction.

The proposed approaches to mitigating temporal bias include two options, one of which increases training cost (but not inference time cost) and the other which is overall more costly but has stronger performance.  The reviewer appreciates the consideration of computational costs, even though that’s not a major focus of the paper.

Experiments show across three encoder models (variants of BERT) that using embeddings learned on a fusion of random split between training and validation, and a temporal split between training and validation, slightly outperforms using only one split or the other.

Each of the given constraints that ensure temporal correctness is well defined and builds off of the other constraints

### Weaknesses
Main technical weaknesses

It is not clear to the reviewer (until reading section 5) what the practical difference is between Constraint TO and Constraint TC. Suggest to clarify more when first introduced

The setup in 3.1, where training/validation/test splits are designed around timestamps, is presented as novel (“it has not been explored in NLP yet” - line 161).  But this has been widely done in papers going back more than a decade at least, especially where the task requires some kind of forecasting.  Closer to this paper’s subject, see Luu et al., 2022, cited in this paper’s introduction.

Section 3.2 experiments don’t give much detail at all, or motivation, for the choice of model.  Was there model selection using the validation set?  How many random seeds for the classical evaluation protocol?  The one dataset where the trend is reversed – MisInfoText – is not discussed at all, though the scores are much higher in the “realistic temporal” evaluation.  Also, why do FC30 numbers go up for F1 and down for accuracy, with the new evaluation setup compared to the old one? 

In section 3, the phrase over evaluated in this context does not make sense. I understand that the meaning is that the model performance on classical evaluation metrics is better than in the practical, temporally ordered setting, but "evaluated" is too overloaded a term here.

The labdrift metric in section 4 is designed to measure the extent to which the timestamps in the data predict the labels.  While the idea is reasonable, one could imagine simpler approaches, based on correlations (point-biserial correlation or Eta; or something nonparametric like Mann-Whitney or Kruskal-Wallis), or based on information theory (e.g., what is the entropy of the label distribution, conditional on time?), or just taking tools that already exist in the literature like the temporal degradation quantity from the Luu et al. paper cited in the introduction.  Regardless, Table 2 might be more valuable if it included an estimate of the distribution over labdrift scores for variations of the datasets in which labels were scrambled across items.  This would give the reader a sense of how much “bias” we would see if the datasets were constructed specifically to avoid it.  Some amount is surely expected due to chance (as we see with the nonzero value for FC 30).

The new dataset sounds exciting, but not much information is given about it.  The relationship to other datasets like the Politifact data that’s compared to is not discussed.  What circumstances led to this dataset’s availability now? What steps have the authors taken to ensure that it doesn’t contaminate the training data for new models?  And why couldn’t the existing benchmarks simply be repurposed under the constraints proposed in section 3.1?  The paper doesn’t say what the labels are or how they compare with other datasets for misinformation research, or other datasets that have been designed to challenge models’ temporal generalization.

Statistical significance tests are used in some places but not others; this inconsistency is not explained

The accuracy metric in the tables is not discussed in the main text, and accuracy and F1 scores often don’t point in the same direction in this paper’s results.  This goes completely unremarked and makes me concerned that all of the findings are metric-dependent

The paper doesn’t really engage with or compare to recent temporal alignment methods like “set the clock” in Zhao et al. (“Set the Clock: Temporal Alignment of Pretrained Language Models,” ACL 2024) or the work of Zhang and Choi (“Mitigating Temporal Misalignment by Discarding Outdated Facts, EMNLP 2023) or “mind the gap” (Lazaridou et al., NeurIPS 2021).  It’s hard to know whether the results here represent the state of the art in mitigating temporal misalignment.

The results are also only evaluated on encoder only models, saying that they are “proficient baselines” but do not describe further what that means, or if the increase in abilities of LLMs since the publication of the citation for that claim (2021) has rendered it null, and does not address the obvious question of whether the same findings would hold up with today's more widely used decoder-only models
The paper would be stronger with a tighter connection between the methods and real world practice

Suggestions to improve writing

There is a mix between active and passive voice in the paper that leads to an occasional implied level of distance from the results (“several observations can be made” vs “we make several observations”)

There is a missing “W” in Q2: “...bias in labels. e propose a…”
the introduction and related works sections are written very informally and have some awkward transitions between tasks (one example being “Many datasets exist (D’Ulizia et al., 2021), allowing for the training of models to detect different aspects of misinformation” what are the datasets?)

In section 1 when defining the research questions, the phrase “temporal test set” is used without definition. Though temporality is defined, and it is clear that this temporal test set is different from a random split, it is not clear what it is.

### Questions
When you concatenate the embeddings and train the two-layer perceptron, do you know how similar the performance of the model is on each item when compared to the individual models? I’d be interested in a per-question evaluation of how often the fusion is the same as the random vs the same as the temporal models

Table 1 reports many statistical significance tests (about 16 by my counting).  What corrections were used to account for multiple tests?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper argues that random-split evaluation inflates misinformation-detection results and that models should be tested under temporal generalization: train on past, validate on a temporally earlier slice, and test strictly on later data (Constraints TO, TC, PT). It introduces LabDrift, a label-drift metric based only on temporally ordered labels; releases FC30, a 36,619-claim fact-checking dataset spanning 1995–2025 with labels aggregated to three classes; and proposes two simple training strategies—Fusion (concat embeddings from random- and temporal-trained encoders plus an MLP) and TempoGen (train the MLP on both embeddings but use only temporal embeddings at inference)—to improve temporal robustness.

### Strengths
1. The TO/TC/PT constraints crisply capture leakage risks from both fine-tuning and pre-training.

2. The paper is easy to follow

### Weaknesses
1. The paper studies an existing problem. The paper does not discuss the difference between existing works such as [1][2].

[1] Jiang B, Tan Z, Nirmal A, et al. Disinformation detection: An evolving challenge in the age of llms[C]//Proceedings of the 2024 siam international conference on data mining (sdm). Society for Industrial and Applied Mathematics, 2024: 427-435.

[2] Jiang B, Zhao C, Tan Z, et al. Catching chameleons: Detecting evolving disinformation generated using large language models[C]//2024 IEEE 6th International Conference on Cognitive Machine Intelligence (CogMI). IEEE, 2024: 197-206.


2. Core experiments use encoders (BERT/DeBERTa/EuroBERT). The case for “LLMs degrade under temporal shift” would be stronger with decoder-only baselines or recent instruction models.

3. Fixing the test to the last 10% may be brittle for smaller corpora/classes (the paper notes PolitiFact’s tiny temporal test). Consider rolling-window or multiple cut points.

4. FC30 pulls from Snopes/PolitiFact; generalization to social-platform posts, multimodal content, or non-English settings is untested.

5. On FC30, some improvements are modest and model-dependent (Table 3). Error analyses (topic drift, named entities, lexical shift) would clarify where methods help or fail.

### Questions
1. Can u discuss in more detail the related/existing works?

2. Can you report variance across multiple temporal cut points (e.g., last 5/10/20%) and rolling-origin evaluation to reduce dependence on a single split?

3. Do results transfer to decoder-only LLMs (frozen features + linear probe) and to multimodal detectors?

4. On FC30, can you provide topic/entity drift analyses and qualitative failure cases for each approach?

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper addresses the problem of temporal generalization in misinformation detection models. The authors argue that standard evaluation protocols, which rely on random data splits, tend to overestimate model performance. This is because models may learn event-specific representations rather than generalizable patterns, a critical flaw in a real-world setting where new misinformation constantly emerges.

### Strengths
1. The paper's clearest strength is its empirical diagnosis of the overestimation problem. Table 1 provides stark, quantitative evidence of how much traditional benchmarks inflate performance, with F1 scores dropping by as much as 40-50 points (e.g., MediaEval, Proppy).

2. The paper introduces FC30, a new dataset with a 30-year span. Its most important feature is the large amount of data from 2020-2025, which finally allows for proper evaluation of models under the "Pre-Training" (PT) constraint.

### Weaknesses
1. The paper's main conclusion about the effectiveness of Fusion and TempoGen is based on FC30. However, the appendix (Table 12) shows these methods are not a general solution. On ISOT, MediaEval, and Proppy, the 'Random' baseline performs best. This lack of generalization suggests the methods may be tuned to the specific properties of FC30 and are not robust. This is the most significant weakness.

2. The central thesis that "models must generalize to future data" is a well-known form of distribution shift. While applying it to misinformation is valid, the paper overstates its novelty. Many recent works on LLM evaluation, robustness, and "out-of-distribution" (OOD) detection have highlighted this exact challenge. The paper fails to differentiate its contribution from this large body of existing work.

3. The results in Table 3 show that the 'Temporal' (T) split baseline consistently performs worse than the 'Random' (R) split baseline. This is counter-intuitive, as one would expect the 'T' split to be better aligned with the temporal test set. The paper does not investigate why this happens. Is the model learning spurious temporal cues from the validation set? Without this analysis, the proposed solution (which just combines R and T) feels like an ad-hoc fix rather than a principled one.

4. Furthermore, I personally believe that the introduction of such benchmarks necessitates measures to prevent benchmark data contamination. Otherwise, such datasets will become obsolete after the next large-scale pre-training.

### Questions
1. The most critical issue is the discrepancy between your main results (Table 3) and your appendix results (Table 12). Why do Fusion and TempoGen fail to generalize, and in some cases, perform worse than the 'Random' baseline on datasets like ISOT, MediaEval, and Proppy? Does this not invalidate the claim that they are effective methods for improving temporal generalization?

2. Could you provide more analysis on why the 'Temporal' split baseline (T) performs so poorly, even worse than the 'Random' (R) baseline? This seems to be a key finding. Does this imply that naively training on "the most recent" data is actually harmful for generalization, perhaps due to overfitting on temporally-specific artifacts in the validation set?

3. Given that the problem of temporal distribution shift is well-established, could you clarify what you see as the primary conceptual contribution for the community, beyond the (valuable) dataset and metric? The proposed methods seem to be straightforward ensembles rather than a new learning paradigm.

### Soundness
2

### Presentation
3

### Contribution
2
