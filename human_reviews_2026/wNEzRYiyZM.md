# Repurposing Foundation Model for Generalizable Medical Time Series Classification

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 8, 4, 6

## Abstract
Medical time series (MedTS) classification suffers from poor generalizability
in real-world deployment due to inter- and intra-dataset heterogeneity, such as varying
numbers of channels, signal lengths, task definitions, and patient characteristics.
% implicit patient characteristics, variable channel configurations, time series lengths, and diagnostic tasks.
To address this, we propose FORMED, a novel framework for repurposing a backbone foundation model, pre-trained on generic time series, to enable highly generalizable MedTS classification on unseen datasets.
FORMED combines the backbone with a novel classifier comprising two components: (1) task-specific channel embeddings and label queries, dynamically sized to match any number of channels and target classes, and (2) a shared decoding attention layer, jointly trained across datasets to capture medical domain knowledge through task-agnostic feature-query interactions. After repurposing, FORMED achieves seamless adaptation to unseen MedTS datasets through lightweight label query training (0.1\% of parameters), eliminating the need for full fine-tuning or architectural redesign.
We evaluate FORMED on 5 diverse MedTS datasets, benchmarking against 11 Task-Specific Models (TSM) and 4 Task-Specific Adaptation (TSA) methods. Our results demonstrate FORMED's dominant performance, achieving up to 35\% absolute improvement in F1-score (on ADFTD dataset) over specialized baselines.
By decoupling domain-invariant representation learning from task-specific adaptation, FORMED establishes a scalable and resource-efficient paradigm for foundation model repurposing in healthcare. This approach prioritizes clinical adaptability over rigid task-centric design, offering a practical pathway for real-world implementation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a novel approach called FORMED for repurposing an existing time series foundation model for multivariate time series classification. Importantly, FORMED is able to accommodate arbitrarily numbers of channels and target classes, and separately captures medical domain knowledge and task-specific knowledge in the model architecture.

### Strengths
- The proposed repurposing strategy is quite clean and elegant
- The experimental results look promising
- I found this paper quite clear and easy to follow

### Weaknesses
My main concern at the moment is that some of the background information provided is inaccurate or is missing references that are relevant. For example:
- Line 116: "To date, no foundation model has been specifically designed for time series classification tasks, let alone MedTS classification[...]" -> I would suggest toning down this sentence especially as I think some authors of various time series foundation models would arguably claim that their model was designed to handle forecasting **and** classification (along with possibly other tasks). For example, classification is considered in the original papers for MOMENT (Goswani et al, ICML 2024), TimesNet (Wu et al, ICLR 2023), Mantis (Feofanov et al, ICML FMSD Workshop 2025), and UniTS (this is already cited in the paper).
- Much of the paper is on basically figuring out a way to combine/mix information across channels for medical time series. However, there actually already is a paper that does precisely this ("Generalized Prompt Tuning: Adapting Frozen Univariate Time Series Foundation Models for Multivariate Healthcare Time Series" by Liu et al, ML4H 2024). From what I can tell, FORMED is more sophisticated than generalized prompt tuning (such as having an additional adapting phase) but even so, it would be helpful to discuss what is different/novel (how information across channels is combined seems similar).
- Line 181: "NOT a simple modification of the output layer" -> I think this is a strong claim that really needs to come with strong empirical justification since I do think that existing authors have been able to get successful classification results by essentially just changing the prediction head and fine-tuning or training from scratch (this is basically what various time series foundation models that are already applied to classification are doing).

Regarding experiments:
- While the paper focuses on using TimesFM as the base foundation model, I do think that it would be valuable seeing how FORMED works when using different base time series foundation models.
- I find that the datasets considered are arguably too similar in structure (they're all ECG or EEG waveform data if I understand correctly). I would be helpful seeing how the results play out with other kinds of medical time series data such as EHR data (e.g., there are ways to preprocess MIMIC for instance to get it to be regularly sampled and to consider classification tasks).
- A selling point early on in the paper is the idea of separately capturing medical domain knowledge and task-specific knowledge (e.g., line 94). I would imagine that these are often actually quite correlated so that they perhaps should not be that decoupled? Do you have thoughts on this? Are there experiments that could be run to somehow verify this knowledge separation in what is encoded?

### Questions
Please see the concerns I raised in "weaknesses".

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The authors introduce FORMED, for utilizing foundational time series models for adaptation to medical time series classification. The authors show improvement in results for classifying EEG and ECG signals.

### Strengths
- The authors target a relevant problem in the medical field on the lack of work for addressing domain nuances around inter- and intra dataset heterogeneity.
- The paper seems sound technically.

### Weaknesses
Avenues for improvement:
- The authors can expand their experiments and ablation study to test more settings such as few-shot learning and include tabular results for easier inference.
- Minor formatting required for page 18.

### Questions
- The experiments focus on high frequency time-series such as EEG, ECG. How well do the authors think their improvements will hold for lower frequency medical time series such as HRV, GSR, etc?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors propose a two-stage parameter-efficient fine-tuning approach for adapting time series forecasting foundation models to medical time series classification problems.

### Strengths
- The paper tackles a relevant concern of adapting public models trained on general data to different domains, particularly addressing the needs of health data.
- The new mechanisms used for fine tuning are promising and interesting ways to extend models to new tasks.

### Weaknesses
- The baselines being compared to are mostly models intended for time series forecasting, not classification. Even within the forecasting domain, SOTA models are not represented, and some of the included baselines are even known to be outperformed by linear models [1]. They are also mostly univariate and therefore not suited for multivariate classification tasks. Medformer is the only current classification model provided. It would also help to have comparisons to strong established baselines like MiniROCKET. (The baselines are also all described as "SOTA", which is not accurate.)
- Existing parameter-efficient fine tuning methods like LoRA (or even end-to-end fine-tuning) could be used for either the Repurposing or Adapting phase rather than the new method the authors propose, but these are not discussed or evaluated against.
- Ablations showing the contributions of different modelling decisions are not provided.
- I have doubts about the design motivation and experimental soundness but would like to discuss further - see Questions.

**Other content issues**

- The "Adaptation of Foundation Models" section of Related Work only has one citation, which seems incomplete.
- The second-last paragraph of the introduction seems to describe the data used in [2] as if it was a contribution of this paper - I think this should be clarified.

[1] Zeng et al. "Are transformers effective for time series forecasting?" AAAI 2023.  
[2] Wang et al. "Medformer: A Multi-Granularity Patching Transformer for Medical Time-Series Classification" NeurIPS 2024.

### Questions
- Why use a zero-shot univariate forecasting model for a trained multivariate classification task? This seems like an awkward decision for a backbone model, and since it's frozen, this mismatch could limit performance.
- Some of the performance differences between the proposed model and baselines seem dubiously large given the differences in modelling decisions. For instance, for ADFTD, it's difficult to see access to the other four pretraining datasets giving FORMED such a massive boost over the other methods, including TimesFM-TSA, without data leakage or a mismatch in experimental procedures. Since code is not available at this time, reproducibility is limited. How would you explain these differences, and are you able to share anonymized code to reproduce them? Ablations could also help clarify this.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes an approach to repurpose the backbone of a time series foundation model for multi-variate medical time series. Authors freeze the backbone and adapt to the target task by learning task-specific channel and label embeddings. Results on real-world datasets show that this approach is able to both transfer information from pre-trained models and generalize to new tasks.

### Strengths
The paper is very well written and easy to follow. The proposed approach is relatively straightforward but principled and well justified. Experiments on real-world datasets show strong generalization with significant gains over leading baselines in both full and few-shot settings.

### Weaknesses
I would have liked to see more ablation studies. Hyper-parameter stability results for channel and label embeddings, stability in the few shot regime (it is hard to learn robust embedding from only a few examples), applicability to other tabular foundation models etc. I also didn't see anything specific to medical time series in the architecture of the proposed approach and generalization to other multi-variate settings could be interesting to test.

### Questions
Do you have additional ablation results on hyper-parameter and learning stability in the few-shot setting? Is there anything in the architecture of FORMED that is specific to medical time series?

### Soundness
3

### Presentation
4

### Contribution
3
