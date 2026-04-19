# How Abilities in Large Language Models are Affected by Supervised Fine-tuning Data Composition

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 5, 3

## Abstract
Large language models (LLMs) with enormous pre-training tokens and parameter amounts emerge abilities including math reasoning, code generation, and instruction following. These abilities are further enhanced by supervised fine-tuning (SFT). The open-source community has studied on ad-hoc SFT for each ability, while proprietary LLMs are versatile for all abilities. It is important to investigate how to unlock them with multiple abilities via SFT. In this study, we specifically focus on the data composition between mathematical reasoning, code generation, and general human-aligning abilities during SFT. From a scaling perspective, we investigate the relationship between model abilities and various factors including data amounts, data composition ratio, model parameters, and SFT strategies. Our experiments reveal that different abilities exhibit different scaling patterns, and larger models generally show superior performance with the same amount of data. Mathematical reasoning and code generation improve as data amounts increase consistently, while the general ability is enhanced with about a thousand samples and improves slowly. We find data composition results in various abilities improvements with low data amounts, while conflicts of abilities with high data amounts. Our experiments further show that composition data amount impacts performance, while the influence of composition ratio is insignificant. Regarding the SFT strategies, we evaluate sequential learning multiple abilities are prone to catastrophic forgetting. Our proposed Dual-stage Mixed Fine-tuning (DMT) strategy learns specialized abilities first and then learns general abilities with a small amount of specialized data to prevent forgetting, offering a promising solution to learn multiple abilities with different scaling patterns.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
- The paper investigates the impact of data composition on the coding, mathematical reasoning, and general instruction-following abilities of LLMs through supervised fine-tuning (SFT).
- It explores the scaling laws of these abilities concerning factors like data amounts, data composition ratios, model parameters, and SFT strategies. It also studies whether these abilities interact with each other.
- A new training strategy, Dual-stage Mixed Fine-tuning (DMT), is proposed to address the issue of ability conflicts and catastrophic forgetting during multi-task learning.

### Strengths
1. This is a good empirical paper with solid experimental results. It makes several important observations that might be helpful for later fine-tuning research. For instance, it is interesting to know that SFT data from different sources appears to benefit each other in low-resource settings, but may act as noise when sufficient data is available.

2. The presentation is clear and easy to follow.

3. Though there's less algorithmic novelty in this paper, the empirical problem it studies is important, and there doesn't seem to be previous work focusing on empirically understanding the compositionality and ability conflicts of LLMs.

### Weaknesses
1. Limited abilities/datasets: while the general problem of data composition and ability conflicts is interesting, the paper only focuses on three abilities---reasoning, coding, and aligning general human intentions---and uses only a single dataset for each ability. This raises the question of whether the observed trends and scaling laws can generalize. 

2. Limited models: similarly, only the LLaMA families are evaluated. It would be better if the paper can test DMT on several other models so that we could know the empirical observations about training strategies can generalize.

3. The current submission does not contain the training and evaluation code. Hopefully this will be released to facilitate reproducibility when the paper is publicly available.

### Questions
1. General setup: What’s the zero-shot performance (i.e., 0 SFT data) of the models on the three datasets? It's good to have these results as a baseline.

2. RQ1: How is the data selection process performed for each training dataset? Is it random? If so, did you sample using multiple seeds and compared results of different subsets? Are they similar? 

3. RQ2: Since the paper mixes the datasets using the same subset ratio instead of the same data amounts, it would be better to indicate the number of data samples for each dataset in the paper. For instance, if one dataset is extremely large and the others are small, will this affect learning the other abilities? Also, it would be interesting to see how the model performs when it is fine-tuned with the same number of data points from each training dataset (rather than same subset ratio).

4. RQ4: The paper discusses sequential training. What is the performance when training with math -> code -> general abilities? Is it different from code -> math -> general abilities?

5. Figure 5 uses the features from the 15th layer of the model. What is the rationale for selecting this particular layer? Would it be better to show results from the beginning or end of the layers?

6. The paper discusses the impact of the distribution gap on acquiring multiple abilities. In the last paragraph of section 4.2, it says that  "*This distribution gap introduces an extra noise during the SFT phrase, while its removal enables the model to better generalize coding and mathematical abilities.*" However, I don’t think better **test-time accuracy** means better generalization ability. It can be that the model learns better how to deal with (or even overfit to) the training distributions (after removing code & math from ShareGPT), and removing the data ensures that **the test distribution is similar to the training distribution**. But in reality, the model can encounter diverse data formats such as the ones removed, and we want to generalize to these diverse data formats rather than only being able to perform well on the data included in ShareGPT.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper studies the impact of different training strategies for supervised fine-tuning of language models for downstream tasks.  Further experiments are conducted to investigate how data composition, amount of data, and data composition ratios impact downstream performance.  Performance is evaluated on a subset of tasks for LLMs including code, symbolic (math) knowledge, and general human alignment.

### Strengths
* This paper presents a thorough evaluation of dataset composition, data size, and data composition ratios on different training strategies for supervised fine-tuning of LLMs.   Based on this, the authors propose a new fine-tuning strategy which is in-turn well-motivated.  

* The authors find some surprising trends in the analysis namely that scaling data amount and data ratios on code in mixed settings have unreliable effects, whereas for other tasks this is not the case.  See Q2 regarding code mixing for training.

### Weaknesses
* All evaluations are done only on a single dataset from chosen domains: math, code, general human,  However, this does not cover different datasets within these domains or across other evaluations including world knowledge, language understanding,  reasoning, fairness, etc.  Many of the claims that are made in the paper such as "different abilities exhibit different scaling curves" and "larger models show better performance with the same data amount generally" feel too strong given the limited evaluations which may be anecdotal to the datasets and specific tasks chosen.  Many of the inconsistency results could also be attributed to differences in the dataset size or the data the model was trained on.  I would like to see more tasks and another dataset comparison before making these large claims as there are many other inconsistencies between the tasks that could lead to differing trends and performance differences beyond just the tasks themselves.

* Results in Table 1 are missing comparisons between datasets (fine-tuning on math and zero-shot or eval on HumanEval).  Results appear to be not much better for DMT compared with existing approaches, and DMT appears to be very particular to these tasks/datasets.  The paper would be much stronger with evaluation on another set of tasks/datasets to show the strategy will generalize.

* The section on scaling laws in LLMs seems a bit out of place in the related work as computing scaling laws is not a primary focus of this work, and there are no conclusions on the amount of data needed based on model size or compute budget for fine-tuning.

### Questions
Q1: There appears to be no reliable trend in the results with DMT and varying k on each of the datasets in Table 5.  What is the intuition for how to select k, and why there is inconsistency.  Intuitively more data should always help.

Q2: Results on code mixing having negative effects with more data seemed to be at odds with some of the results presented in prior work: https://arxiv.org/abs/2305.16264 which showed that adding code could actually help with NL tasks by adding structure.  Clarification on the results and why they're expected is beneficial.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The manuscript studies data composition's effects on supervised fine-tuning in language modeling, especially large language models. The work proposes a new strategy for dataset mixing, incorporated into the pipeline of LLaMA, that demonstrates competitive behavior on general and specialized benchmarks like MT-Bench and (GSM8k, HumanEval) respectively.

### Strengths
1. Straightforward methodology
2. Experimental results are extensive in the aspects that the author care about
3. Experiments are performed on open models, which make the process replicable.

### Weaknesses
1. The analyses are very superficial, as only trends in standard benchmarks are measured and discussed. It is unclear what significance do t-SNE graphs show. The authors did not attempt to arrive at any form of empirical laws for typical scaling law works.
2. The main strategy has no novelty, as it's simply a mixture of different datasets. It seems very straightforward that adjusting the mixture will result in different capabilities on benchmarks, as well as the general rule of thumb that "more data under proper learning set-up results in better performance." The authors did not attempt to give any theoretical analysis of what they have observed, which further weakens the novelty claim.
3. Some wordings in the manuscript are unclear and unfit for an academic context. e.g. data amount, 100 thousand samples.

### Questions
What does it even mean for "scaling" of SFT when it deals with fractional data? I'm failing to see why the problem the author is studying is significant. Typical instruction fine-tuning works deal with a fine-tuning budget of 1E18 - 5E21 FLOPs [1], how would that translate to the amount of compute used in this work (since it's not clearly indicated)? Suppose the actual range of compute budget being studied (e.g. 1/256 "data amount" to 1 "data amount,") is much less than the usual amount in the literature. How would one ascertain that the scaling capabilities would extrapolate?

[1]: Scaling Instruction-Finetuned Language Models

### Soundness
2 fair

### Presentation
2 fair

### Contribution
1 poor
