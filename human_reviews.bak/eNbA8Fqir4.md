# DataMan: Data Manager for Pre-training Large Language Models

- Decision: Accept (Poster)
- Scores: 5, 6, 5, 6, 8

## Abstract
The performance emergence of large language models (LLMs) driven by data scaling laws makes the selection of pre-training data increasingly important. 
However, existing methods rely on limited heuristics and human intuition, lacking comprehensive and clear guidelines.
To address this, we are inspired by *``reverse thinking''* -- prompting LLMs to self-identify which criteria benefit its performance. 
As its pre-training capabilities are related to perplexity (PPL), we derive 14 quality criteria from the causes of text perplexity anomalies and introduce 15 common application domains to support domain mixing.
In this paper, we train a **Data** **Man**ager (**DataMan**) to learn quality ratings and domain recognition from pointwise rating, and use it to annotate a 447B token pre-training corpus with 14 quality ratings and domain type.
Our experiments validate our approach, using DataMan to select 30B tokens to train a 1.3B-parameter language model, demonstrating significant improvements in in-context learning (ICL), perplexity, and instruction-following ability over the state-of-the-art baseline. 
The best-performing model, based on the *Overall Score l=5* surpasses a model trained with 50% more data using uniform sampling. 
We continue pre-training with high-rated, domain-specific data annotated by DataMan to enhance domain-specific ICL performance and thus verify DataMan's domain mixing ability. 
Our findings emphasize the importance of quality ranking, the complementary nature of quality criteria, and their low correlation with perplexity, analyzing misalignment between PPL and ICL performance. 
We also thoroughly analyzed our pre-training dataset, examining its composition, the distribution of quality ratings, and the original document sources.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
2

### Summary
This paper proposes DataMan, a perplexity-driven data management system that annotates the quality and domain of data points, which could be used for many applications such as data selection in the pre-training stage. The authors proposed 14 data quality evaluation metrics and trained a model for data quality based on Qwen-2-1.5B with GPT-4o-turbo annotations. Utilizing the data quality annotations, the authors selected a subset of data and trained a Sheared-Llama-1.3B model from scratch. The model trained with the best overall score demonstrates superior ability in various downstream tasks compared with baseline methods. The authors also conducted various analyses of DataMan and revealed several interesting findings.

### Strengths
1. The proposed system DataMan is overall sound for assessing data quality and this is an important research problem in the current research community.
2. The annotated data is an expensive but valuable resource for the community.
3. The experiments, along with analyses, are solid and demonstrate the great performance of DataMan.

### Weaknesses
1. The presentation of this paper needs significant improvement. I can find typos almost everywhere. I just list a few here. In line 193 "minimizing"; A missing space in line 241; Figure 1 caption is too close to the figure;  line 110 "require"; line 234 incorrect citation format; line 420, "quating";
2. Some claims made in this paper are not well supported. The one that mostly concerns me is that the authors claim "previous works are intuitive and lack generalizability". However, the 14 quality criteria seem also intuitive and manually designed by the authors.

### Questions
1. In lines 231-235, the authors claim that pointwise assessment is better than pairwise one by presenting results in Table 5. I don't quite understand how content in Table 5 could be related to this. Moreover, I don't think handpicked results could really support this claim.
2. It would be better if the authors could write how they picked these 13 quality criteria and all the domain categories. For example, the authors might provide some statistics on the outlier causes mentioned in lines 92-94, and how the iterative refinement is done. Is it done manually? By student or expert?
3. In line 285, what is the value of $k$ in top-k?
4. In lines 201-203, can you explain how the designed 14 quality criteria correspond to the three principles listed here?

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses the need for comprehensive data quality criteria for pre-training large language models (LLMs). It introduces "DataMan," a tool that evaluates text quality and domain types using 13 quality criteria related to LLM performance. The paper demonstrates the effectiveness of these criteria by fine-tuning LLMs on a subset of the SlimPajama corpus, showing performance improvements over state-of-the-art benchmarks. Contributions include the establishment of quality evaluation criteria, the creation of the DataMan tool, and the release of models and annotated datasets to facilitate further research on the relationship between data quality and LLMs.

### Strengths
- The paper offers a comprehensive set of quality criteria for evaluating pre-training data, which helps improve LLM performance.
- Though not comprehensive, the approach of leveraging LLMs to self-identify anomalies in perplexity is an effective method for deriving quality metrics.
- The release of annotated datasets and models is a valuable contribution to the community, promoting further research and development.

### Weaknesses
- The dependence on perplexity as a primary measure for quality assessment may limit the generalizability of the approach. While perplexity is an indicator of understanding, it may not fully capture other essential aspects of data quality, such as creativity or sensitivity. It is also bounded by the knowledge and capabilities of the LLMs that are used to compute perplexity.
- It fixates comparing with Wettig et al.,2024 and lacks comparison with other related work.

### Questions
- How does this method compare with other methods (e.g., deduplication, heuristic-based selection, domain mixture, other LLM quality signals) mentioned in the related work?
- How does this method compare with these large open-source efforts described in the following two papers?
  - Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research [[paper](https://arxiv.org/abs/2402.00159)] [[code](https://github.com/allenai/dolma)]
  - The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale [[paper](https://arxiv.org/abs/2406.17557)] [[code](https://github.com/huggingface/datatrove/)]

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper presents DataMan, a Qwen2-1.5B fine-tuned to perform pointwise evaluation across 14 quality criteria on a 356K dataset annotated by GPT-4-Turbo. The authors then use this model to create the DataPajama dataset from annotating a 447B tokens subset of the SlimPajama dataset. Pre-training from scratch on a subset of DataPajama show improvements compared to other baselines.

### Strengths
- Use LLMs for filtering pretrained data as opposed to using heuristics.
- Promise to release code, models and datasets, which would benefit the research community especially given the cost of obtaining such models and datasets.

### Weaknesses
- **Novelty**: [1] already proposed to train a model to rate and select high-quality pre-training data based on four qualities: "writing style", "required expertise", "facts & trivia", and "educational value". The main differences seem to be on the considered criteria, the model used (Qwen2-1.5B vs Sheared-Llama-1.3B) and using pointwise instead of pairwise ratings.

[1] Wettig et al., QuRating: Selecting High-Quality Data for Training Language Models, 2024.

### Questions
- The message in Section 3.4 is overall confusing: 

i) I'm not convinced that pointwise ratings can capture the same essence as pairwise ratings. Suppose you are evaluating the factuality of an LLM response given a question about Albert Einstein, the LLM gives two answers (A) and (B): both are factually correct but (A) adds a few more details while (B) simply gives the answer. Pointwise rating would not distinguish the subtleties of both responses while pairwise would. To support the opposite view, the authors perform validation through human judgments. However, I'm concerned about the validation process (Table 5), as it appears limited, having been conducted on only 10 samples. 

ii) For prompt validation, the authors claim that human agreement exceeds 95%. Could you clarify the validation process ?

iii) Could you provide more context and an explanation to the equation (L.246-255) ? I can imagine that the error in rating decrease with the loss when training. But, how does it justify the use of pairwise over pointwise ?

- As a parallel work to [1], is there a reason behind choosing Qwen2-1.5B instead of Sheared-Llama-1.3B to train your DataMan ? Plus, the authors in [1] have disclosed their training dataset. This would have enabled a direct comparison between pointwise and pairwise ratings. 
- For the "Sample with Qurating" baseline, could you clarify the sampling process ? My understanding is that you sampled with DataPajama and not the original QuRatedPajama. In this case, is the comparison fair ? A quick look at results in [1] showed lower perplexity.

[1] Wettig et al., QuRating: Selecting High-Quality Data for Training Language Models, 2024.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors propose DataMan, a pipeline for pretraining data quality filtering. First, PPL-thresholded data are used to extract 13 different quality rating criteria. Then, a smaller LM is fine-tuned to annotate documents with scores for each quality rating (and domain.) Finally, the LM is used to produce a higher quality small subset of RedPajama. Training on this subset shows accuracy gains across numerous tasks and significant improvement over baselines.

### Strengths
- The paper uses a novel "reverse thinking" concept which derives quality criteria using documents filtered into high and low perplexity buckets, which avoids pitfalls of criteria based on human intution.
- Introduces 13 detailed criteria for quality, which moves beyond typical heuristic approaches which threshold on perplexity, deduplication, etc. 
- The focus on 13 specific quality metrics and their annotations makes it clear why a particular document was chosen by the model for inclusion in the dataset, thus making the method interpretable.
- DataMan demonstrates a significant performance improvement over numerous heuristic and model-based baselines across multiple metrics: PPL, benchmark accuracy w/ ICL, and instruction following.
- The method performs well against impressive baselines, including statistical approaches like DSIR and models trained with 50% more data.
- Empirical evidence shows benefits of DataMan for domain-specific continued pre-training.
- The presentation is clear.

### Weaknesses
- The authors apply DataMan to a 30B subset of SlimPajama, which is a relatively small dataset. Further work should be done to validate that the method works for large pretraining-scale datasets.
- The SlimPajama dataset has already been filtered for quality with duplicate documents removed. It’s not clear if this influences the analysis of the method or if similar improvements would be seen applying DataMan to unfiltered web data.
- The paper does not discuss the inference FLOPs required to run DataMan. This would be useful to understand the tradeoff between spending more FLOPs on annotation versus pretraining for more tokens.

### Questions
- SlimPajama is already quality-filtered and deduplicated. How does this affect your analysis and results? If DataMan's improvements come mainly from filtering already clean data, the impact on raw web data could be quite different. Could you provide results or analysis on a raw, unfiltered dataset?

- How is the "overall score" calculated from the 13 criteria? Could you provide the exact formula or methodology? Is this determined by the annotating LLM?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper introduces "DataMan," a comprehensive data manager designed for pre-training large language models (LLMs). DataMan aims to improve data quality and selection by introducing 13 well-defined quality criteria and an overall quality score. The approach leverages reverse thinking by prompting LLMs to self-identify the factors that influence performance, focusing on perplexity outliers. The paper conducts experiments on several benchamrks to show how DataMan effectively annotates and selects high-quality training data, resulting in improvements in in-context learning (ICL) and instruction-following tasks. As described in the paper, the implementation of the framework would be released, which could be beneficial for real practices.

### Strengths
1. The motivation of using LLMs to self-select the criterias that are beneficial for model's performance is straightforward and easy to understand. The overall framework makes sense.
2. The description of the method is clear and easy to follow.
3. Data selection for both pre-training, instruction-tuning, and SFT is an important direction for LLM applications, where I think the direction in this paper is interesting and could be beneficial for downstream applications. The experiments show that the proposed method can also improve the model's performances in different benchmarks.

### Weaknesses
1. I think the motivation to use 14 scores is still a little bit unclear. It would be beneficial to have an analysis of why these scores were chosen and how these scores can distinctively differentiate models' abilities when evaluation.

### Questions
1. How do the authors plan to resolve the situation where model itself made unreasonable decisions on criterias? The hallucinations generated during the process could have snowball effects for some training scenarios. One approach that can be tried is using Mixture-of-Experts to reduce the possibility of such situations, but I would like to see authors' insights and if they did any experiments on this direction.
2. Is there a way to mitigate the high computational cost of DataMan’s annotation process, perhaps by leveraging lower-cost models or more efficient sampling techniques?
3. The paper mentions that perplexity and ICL do not align strictly in some cases. Could the authors provide more insights or analysis into the scenarios where this misalignment is most pronounced and how it impacts downstream tasks?
4. I think "vasrtdatasets" at line 26 is a typo?

### Soundness
3

### Presentation
3

### Contribution
3
