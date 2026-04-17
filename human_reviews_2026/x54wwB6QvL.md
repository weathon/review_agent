# Scaling Laws Revisited: Modeling the Role of Data Quality in Language Model Pretraining

- Decision: Accept (Poster)
- Scores: 4, 6, 8, 6

## Abstract
Scaling laws for language model training traditionally characterize how performance scales with model size and dataset volume. Prior work has explored architecture variants and data treatments such as dataset filtering and noise injection in language model pretraining; however, these studies have not formalized data quality within a principled scaling law. We introduce a dimensionless data-quality parameter Q, and propose a quality-aware scaling law extending the Chinchilla framework to predict loss as a joint function of model size, data volume, and data quality. The law is motivated by an effective-sample-size and information-theoretic view of noisy or redundant corpora, and it admits two practical estimators for Q: (i) a corruption rate proxy and (ii) a deficiency measure. Through synthetic experiments in neural machine translation and autoregressive modeling--where we systematically control data quality via multiple levels of noise injection variation--we show that loss scales predictably with data quality and that higher-quality data can substantially reduce model size and hence compute requirements. Our results demonstrate a sublinear decay of effective data with quality and robustness to moderate data corruption; out-of-sample evaluations further validate the predictive form of the law. Unlike prior empirical analyses, our work establishes an explicit, generalizable law for data quality, offering concrete guidance for balancing data curation effort and model scale in large-scale pretraining.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces an LLM pretraining scaling law that accounts for data quality. This takes the form of calculating a dimensionless data-quality parameter and modifying the standard Chinchilla scaling law. The paper studies two different formulations of data quality and provides experiments on neural machine translation and language modeling that empirically validate the proposed scaling law with synthetically noised datasets.

### Strengths
1. The paper provides a theoretically motivated way to incorporate a dimensionless data quality term into a Chinchilla-style scaling law.
2. The experimental setup is sound, and the experiments show that the proposed scaling law fits LLM training in practice.
3. The proposed scaling laws also enable studying how data quality impacts performance, as with the observation that "models are more robust to moderate corruption than predicted by simple effective sample-size theories from PAC learning or channel-capacity analysis" (lines 374-377).

### Weaknesses
The main weakness of the paper is that it studies limited forms of data quality that do not correspond to the kind of quality that is used in practice. In practice, pretraining data quality is often estimated with classifiers (such as estimating the degree to which a given document resembles manually curated high-quality data) or heuristics (such as removing documents with many repeated n-grams). The current experiments are limited to synthetically noised datasets. Do either of the proposed quality formulations capture this more nuanced form of data quality?

I am willing to raise my score if either 1) further experiments can show that the current formulations of quality do in fact capture more practical notions of quality or 2) more reasoning is provided to show the studied kinds of noise are  practically useful.

### Questions
1. Can you elaborate on the relationship between your scaling law and the  heterogenous quality scaling laws in [1]?

2. The setting of Q=1 (the highest data quality) to recover the standard Chinchilla scaling law (line 250) does not make the most sense to me. I could imagine scenarios where we would want to be able to account for having "higher quality" data than that used for the Chinchilla law.

Small comments:
- Line 289/290: "(Table 2" -> "(Table 2)"
- Is there any meaning behind using $E^+$ in Tables 1, 2 and Section 5.6 instead of just E like in Equation 2?
- Line 355: N is used for noise percent, but doesn't N already mean model size?
- Line 471: "Notably, their findings" -> "Notably, our findings"
- Line 477: "Their experiments on both" -> "Our experiments on both"

[1]: Scaling Laws for Data Filtering -- Data Curation cannot be Compute Agnostic, Goyal et al., CVPR 2024.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper revisits the scaling laws of large language model (LLM) pretraining by introducing an explicit, dimensionless data quality parameter Q, extending the traditional analysis of model size and data volume to a joint framework that incorporates data quality. The authors propose a quality-aware scaling law. They systematically inject synthetic noise and vary data coverage in neural machine translation and causal language modeling tasks. Experimental results show that high-quality data can significantly reduce loss for a given model size, and under high-quality data conditions, smaller models with less computational resources can achieve strong performance.

### Strengths
1. The work innovatively incorporates data quality as a single parameter into the scaling law, theoretically demonstrating a strong correlation between model performance and data quality.
2. Controlled experiments are conducted across multiple experimental settings, validating the practical applicability of the quality-aware scaling law.

### Weaknesses
1. Although synthetic noise offers strong controllability, it may not capture the highly complex and heavy-tailed distribution of low-quality data in real training datasets. How can the current single-parameter quantification method be extended to more realistic data scenarios?

2. The authors use a fixed learning rate to investigate the scaling law. During training, could different learning rates have additional effects on the conclusions regarding the data-quality scaling law?

### Questions
Please refer to the relevant points in the Weaknesses section. If the authors can provide clarification and improvements, I would be very happy to raise my score.

### Soundness
3

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
This paper extends classical LLM scaling laws by incorporating data quality alongside model size and data volume. It introduces a dimensionless quality parameter $Q$ and derives a quality-aware, Chinchilla-style law from effective sample size and information-theoretic arguments. Relative to the original Chinchilla formulation, the data term $D^{\beta}$ is additionally modulated by $Q^{\gamma}$.
The authors validate the approach with controlled experiments in neural machine translation (ParaCrawl) and causal language modeling (C4), varying dataset size and injecting synthetic noise across seven quality levels. To generate noise, they pad 50% of tokens in the NMT setting and replace 50% of tokens with random tokens in the language-modeling setting. Test loss scales predictably under the proposed law.
Empirically, higher-quality data can substitute for larger models or greater compute, and the estimated robustness exponents are sublinear, indicating resilience to moderate corruption. Noise is less detrimental for NMT than for language modeling. The paper also provides iso-loss trade-off contours and out-of-sample validations, offering practical guidance on balancing data-curation effort against model scale—particularly relevant in specialized domains with scarce but high-quality corpora.

### Strengths
Originality
The paper proposes an original method for measuring data quality. To the best of my knowledge, the information-theoretic approach and its assumptions are new.

Quality
The paper presents small-scale but thorough experiments. It includes carefully controlled studies in both CLM and NMT—with nested dataset sizes, seven corruption levels, and single-factor manipulations to isolate quality effects.

Clarity
The text, derivations, and presentation are clear.

Significance
This work can simplify practice in domains with scarce data and limited GPU budgets. It helps practitioners decide whether to invest in data quality/size or compute. It also provides a framework for working with token-level corruption in a more principled way.

### Weaknesses
Estimating Q. It is unclear how to estimate Q in practical settings for new datasets with multiple noise sources.

Scale of experiments. The experiments are small-scale relative to current SOTA setups: the authors use a 133M-parameter model for NMT and a small 8-layer Llama-3 variant (the parameter count is not reported in the paper).

Comparative evaluation. There is no direct comparison of the proposed law’s quality against competing methods. It would be helpful to include an error analysis across alternative approaches using the experimental data, or to explain why such comparisons are not applicable / are out of scope.

Related work referenced.

“Scaling Laws for Data Filtering—Data Curation Cannot Be Compute-Agnostic.”

“Scaling Parameter-Constrained Language Models with Quality Data.”

Additional, not previously mentioned work: https://aclanthology.org/2025.acl-long.1163/

### Questions
It is unclear how Q should be estimated in practical settings. Could you provide a step-by-step guide for practitioners on applying the proposed method to non-standard datasets, including handling multiple noise sources?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors use a definition of data quality to include in the chinchilla scaling laws that consider model size and data size. They motivate this data quality notion and formulation with an effective data size formulation and run experiments with an approximately 140M parameter model.

They train 3 scales with 7 corruption levels for an NMT task and a CLM task. The text corruptions are synthetically induced by either replacing 50% of non-special tokens with pad tokens or randomly swapping 50% of tokens with other random tokens from the vocab. 

The authors show that they can predict test loss with this new scaling plot with additional findings on how effective data size interacts with data quality. 

The paper formulates the problem well, comes up with a good extension to scaling laws that is both simple and effective. 

However the experimental setup can be better presented or even be extended to support the core claim of the paper.

### Strengths
The paper presents data quality scaling laws which is simple and empirically supported.

The presentation is extensive and very good.

The data subsampling scheme in the experiments as well as the high number of Q values in experiments is quite good.

### Weaknesses
The experimental setup doesn't consider scaling the N value(model parameters) the paper works with a fixed model size however the scaling law has all three components and one is actually not incorporated in the experimental setup. Yet the paper makes claims about "smaller models and less compute is needed to achieve strong results" 

The noise injection method is unnatural. A more natural noise injection strategy could be actually replacing a certain % of documents with low quality web documents. This could create a better noise injection method and make results more relevant. 

The choice to swap with pad in NMT and other tokens in CLM, as well as to use GPT type base model for one and LLama base model for the other seems interesting, particularly when keeping variables constant could help better pinpoint the effect of cross lingual redundancy. Now the effect is conflated with different noise injection strategies as well. 

No sensitivity or ci's are presented. Particularly considering the difference between NMT and CLM gamma values a sensitivity analysis on the experimental design could be useful in convincing readers to the results. 

Revisiting Scaling Laws for Language Models: The Role of Data Quality and Training Strategies (Chen et al, 2025) seems to be published online in January but the authors have not cited the work which should really be included even contrasted in their paper. The underlying question is exactly the same.

### Questions
Is there a typo in definition 2? "data deficiency data quality Q(ω)" 

Do you expect your findings to generalize over your experimental setup? Particularly considering the lack of any experiments with model size how confident should we be in the estimates for the scaling laws.

### Soundness
2

### Presentation
3

### Contribution
3
