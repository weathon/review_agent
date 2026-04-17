# LLMs Can Get "Brain Rot"!

- Decision: Reject
- Scores: 2, 8, 4, 4

## Abstract
We propose and test the **LLM Brain Rot Hypothesis**: continual exposure to *junk web text* induces lasting cognitive decline in large language models (LLMs). To causally isolate data quality, we run controlled experiments on real Twitter/X corpora, constructing junk and reversely controlled datasets via two orthogonal operationalizations: **M1** (engagement degree) and **M2** (semantic quality), with matched token scale and training operations across conditions. Contrary to the control group, continual pre-training of 4 LLMs on the junk dataset causes non-trivial declines (Hedges' $g>0.3$) on reasoning, long-context understanding, safety, and inflating ``dark traits'' (e.g., psychopathy, narcissism). The gradual mixtures of junk and control datasets also yield dose-response cognition decay: for example, under M1, ARC-Challenge with Chain Of Thoughts drops $74.9 \rightarrow 57.2$ and RULER-CWE $84.4 \rightarrow 52.3$ as junk ratio rises from $0$ % to $100$ %.

Error forensics reveal several key insights. First, we identify *thought-skipping as the primary lesion*: models increasingly truncate or skip reasoning chains, explaining most of the error growth. Second, partial but incomplete healing is observed: scaling instruction tuning and clean data pre-training improve the declined cognition yet cannot restore baseline capability, suggesting persistent representational drift rather than format mismatch. Finally, we discover that the popularity, a non-semantic metric, of a tweet is a better indicator of the Brain Rot effect than the length in M1. Together, the results provide significant, multi-perspective evidence that *data quality is a causal driver of LLM capability decay*, reframing curation for continual pretraining as a *training-time safety* problem and motivating routine "cognitive health checks" for deployed LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces the term Brain Rot in LLMs, defined as continual pre-training on junk web text inducing lasting cognitive decline in LLMs. The junk data consists of Twitter/X posts that are short, semantically simple, and popular (high engagement), whereas the control data are posts from the same dataset that are longer, less popular, and with more high quality text (as judged by GPT-4o). By pre-training 4 models of varying families and sizes on the junk and controlled datasets and evaluating on a series of benchmarks of cognitively demanding tasks, results demonstrate a decline in performance. Additional experiments trying to mitigate the effect with instruction tuning show it is somewhat effective, but does not fully restore the original model capabilities. Overall, the results suggest that additional caution is needed in curating high quality pre training datasets to avoid "LLM Brain Rot."

### Strengths
The paper is well written and clear. The topic of "Brain Rot" as defined in the paper is original to my knowledge, and the in depth analysis on "Brain Rot" effects for example on reasoning failure modes is interesting and insightful.

### Weaknesses
The main weakness is that the main contribution is neither novel nor realistic. The main conclusion from the paper is that pre-training datasets should be audited to ensure high quality and avoid the supposed cognitive decline. This is neither novel (training/finetuning on bad data has been shown to reduce language modeling capabilities before), and is also not realistic (no pre-training dataset actually contains such high proportions of specifically junky twitter data, and the authors do not justify that this is something being done in the real world). As a result, I feel that this mostly undermines the insight or utility of the work. To actually make this more realistic, for example it would be interesting to do a pre-training dataset audit to see the actual proportion of low quality data, and then re-train the model with this subset removed, and comparing the results. Or, perhaps using the existing setup, it would be more notable if there was some difference noticed by end-users interacting with the models or if model responses were more engaging for users? 

Second, I'm not convinced of the soundness of the dataset construction. The thresholds for popularity and length are seemingly arbitrary. Moreover the inclusion of length as part of the M1 engagement metric is motivated in 3.1 because shorter tweets supposedly boost engagement, but then later in lines 195--6 the authors find no strong correlation between these two factors, which feels contradictory. There are also lacking details on the human evaluation of the GPT-4o classification of high vs low semantic quality.

### Questions
1. Could you please address how the setting of continual training on such junk web data reflects realistic model training in practice? See first paragraph in weaknesses above.
2. How were the thresholds for popularity and length determined? 
3. How many samples were manually annotated by the 3 humans? What specific criteria / instructions were given to the annotators to define low quality? What is the inter-human-annotator agreement, and how does that compare to the GPT-4o?

### Soundness
3

### Presentation
4

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
The paper proposes and validates the LLM Brain Rot Hypothesis, which suggests that exposure to junk data can cause cognitive degradation in large language models. The authors identify fine-grained failure modes such as “thought skipping” and show that popular and short-form data contribute differently to cognitive decline. They further explore post-hoc mitigation strategies, finding that instruction tuning with additional data can partially restore model performance. The persistence of these effects highlights the need for careful data curation during pre-training to prevent long-term cognitive damage in LLMs.

### Strengths
The paper provides strong evidence supporting its main claim regarding model degradation. It is clearly written, and the central hypothesis is both unique and thought-provoking. Moreover, the findings on engagement intervention reveal non-trivial effects on safety and model personality, adding an important dimension to the overall contribution. The result showing that brain rot is persistent against post-hoc tuning is surprising.

### Weaknesses
* Training on lower-quality data is generally known to yield poorer downstream performance [1]. It would be helpful if the authors could more clearly distinguish their specific claim from this broader, well-established observation.

* The presented continual learning framework corresponds to one particular training setup with fixed hyperparameters. It is unclear how sensitive the results are to changes in training conditions (e.g., less data, fewer training steps, or lower learning rates). Discussing the robustness of the findings to such variations would strengthen the paper. This claim is true both for the continual pretraining stage and the instruction tuning stage.

[1] Hoffmann et al., Training Compute-Optimal Large Language Models, 2022.

### Questions
Could the authors provide an analysis of the content diversity within each dataset? Data quality can also be assessed through metrics such as duplication and repetition rates, which often reflect redundancy and lower informational value. It would be interesting to see whether the observed effects correlate with these measures of data diversity.

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
2

### Summary
This paper proposes and validates the LLM Brain Rot Hypothesis: continual pre-training on junk web text leads to significant degradation in large language models’ reasoning, long-context understanding, and safety, while also inducing negative “personality” traits such as narcissism and psychopathy.

### Strengths
This paper draws an analogy from human psychology by introducing the concept of “Brain Rot” into LLMs, proposing and systematically validating the LLM Brain Rot Hypothesis, which offers a novel perspective on the causal relationship between data quality and model capability degradation.

### Weaknesses
1. The experiments rely exclusively on Twitter/X data to construct the junk corpus, lacking cross-platform or cross-lingual validation. The authors should clarify whether the findings generalize to broader Internet contexts; otherwise, the external validity of the conclusions may be limited.

2. This paper employs the TRAIT questionnaire from psychology to evaluate “personality” tendencies of LLMs, but this approach carries methodological concerns. Since LLMs do not possess genuine personalities, the results may conflate prompt sensitivity with representational drift, and the authors need to justify the validity of this evaluation method more rigorously.

3. This study only covers models in the 0.5B–8B parameter range, omitting larger mainstream LLMs. The authors should discuss whether the conclusions extend to 70B-scale models or beyond, and whether larger parameter sizes might provide resilience that alters the manifestation of the Brain Rot effect.

### Questions
see weaknesses

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper examines the effects of continued pretraining (CPT) of language models on “junk web text,” using Twitter corpora as the source data. The authors report a decline in reasoning ability, safety, and ethical alignment, as well as an increase in “dark traits” such as hostility and rigidity.

### Strengths
Strengths:

- The evaluation scope is comprehensive, covering reasoning ability, mechanistic interpretability, personality, and ethical norms. This multifaceted testing approach effectively demonstrates how low-quality data can influence model behaviour.

- The observed degradation in reasoning performance after CPT on low-quality data is a novel and significant empirical contribution.

### Weaknesses
Weaknesses: 
- The overall approach raises conceptual questions. Since modern LLMs are already trained on large-scale open web data that includes Twitter and similar content, it remains unclear whether the addition of “junk” text meaningfully changes exposure. The paper also does not discuss whether filtering such data could improve model performance, or whether similar filtering is already in practice.

- The two metrics used to define “junk” data  (engagement and semantic quality) are only partially convincing. While semantic quality seems reasonable, using engagement (e.g., fewer than 500 interactions) as a proxy for low quality is debatable. Low engagement does not necessarily indicate poor content and may even correspond to high-quality but less viral material. This issue also affects the selection of “control” data.

- Section 3 (M2) claims that data quality aligns with human preferences, but lacks methodological details about the annotation process, such as inter-annotator agreement, the diversity of annotators, or stratified sampling across engagement levels and text lengths.
CPT training was conducted for three epochs. While this is understandable for a small corpus, it may have caused overfitting or catastrophic forgetting. A comparison with one-epoch training would help clarify whether the degradation stems from overexposure or intrinsic data issues.

- The NIAH-MK3 model exhibits lower performance after CPT, yet the underlying reasons are not thoroughly analyzed.

- The paper does not explicitly test for catastrophic forgetting, which is especially important given the multi-epoch training setup.

- The dataset construction and filtering pipeline lacks transparency, particularly regarding how high- and low-engagement examples were balanced, raising the risk of sampling bias.

- Finally, while the paper convincingly shows that CPT on “junk” data harms model behaviour, it offers no discussion of potential mitigation strategies. It remains unclear whether the authors advocate for stricter data filtering, improved data curation, or new training safeguards to prevent such “brain rot.”

### Questions
Please see my concerns in the weaknesses section

### Soundness
3

### Presentation
3

### Contribution
3
