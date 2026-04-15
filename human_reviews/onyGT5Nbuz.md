# Efficient Continual Pre-training for Building Domain Specific Large Language Models

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 5, 6, 5

## Abstract
Large language models (LLMs) have demonstrated remarkable open-domain capabilities. Traditionally, LLMs tailored for a domain are trained from scratch to excel at handling domain-specific tasks. In this work, we explore an alternative strategy of continual pre-training as a means to develop domain-specific LLMs. We introduce FinPythia-6.9B, developed through domain-adaptive continual pre-training on the financial domain. Continual pre-trained FinPythia showcases consistent improvements on financial tasks over the original foundational model. We further explore simple but effective data selection strategies for continual pre-training. Our data selection strategies outperforms vanilla continual pre-training’s performance with just 10% of corpus size and cost, without any degradation on open-domain standard tasks. Our work proposes an alternative solution to building domain-specific LLMs from scratch in a cost-effective manner.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper investigates a new approach to developing domain-specific Large Language Models (LLMs) using continual pre-training. A domain-specific LLM, FinPythia-6.9B, was created for the financial sector through domain-adaptive continual pre-training. The results show that FinPythia has improved performance on financial tasks compared to the original base model. Additionally, the paper proposes effective data selection strategies at the embedding level for continual pre-training.

### Strengths
* The paper proves that continual pre-training can facilitate the LLM's performance on domain-specific LLMs.

* The authors use embedding level selection to acquire the essential data samples and show that with 10% data, the pre-training can achieve comparable performance instead of using a large amount of data.

### Weaknesses
* This paper only demonstrates that the proposed pipeline can be efficient with the data selection on one type of LLM, Pythia, which is insufficient to support the claim of efficiency advantages for other types of LLMs, e.g., LLAMA, OPT.

* The comparison with other baseline methods is not fair because more data is used for training in continual pre-training. To illustrate the effectiveness of the continual pre-training, the authors should apply the proposed method to other types of LLMs.

### Questions
* Could you do similar continuous pre-training and ablation studies for LLAMA-7B?

* Could you do the same continuous pre-training for OPT, BLOOM, and GPT-J and report the results on financial tasks?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper explores domain-adaptive pre-training in the finance sector. It introduces a heuristic data selection method based on novelty (perplexity) and diversity (entropy of POS tagging). The results demonstrate its superior performance compared to general language models (LLM).

### Strengths
1. DAPT/DACP is an important and practical problem
2. The proposed data selection method is simple to use and easy to understand

### Weaknesses
In this paper, Domain-adaptive pre-training (DAPT) or DACP is not a novel concept, and the main innovation lies in the proposed data selection method. However, the paper lacks comparisons with other DAPT baselines, which is a significant drawback. For example, some prior works have explored modifications to the DAPT loss or gradient. Notably, [1] addresses continual pre-training but is not compared with in this work, nor are any other mentioned baseline systems in [1].

[1]: Adapting a Language Model While Preserving its General Knowledge, Ke et al., EMNLP 2022

### Questions
See above

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper introduces FinPythia-6.9B, a LLM developed through domain-adaptive continual pre-training on the financial domain. The paper shows that continual pre-training can yield consistent improvements on financial tasks over the original model. It also experiments different data selection strategies for continual pre-training and proposes a data selection strategy based on novelty and diversity measurements. The proposed efficient domain-adaptive continual pre-training technique outperforms vanilla continual pre-training's performance with just10% of corpus size and cost.

### Strengths
1. The paper contributes FinPythia-6.9B a foundation model for financial domain via continual pre-training. FinPythia-6.9B outperforms the original LLM on a series of tasks in the financial domain which showcases the feasibility of building domain-specific LLMs in a cost-effective manner.
2. Improving continual pre-training from the data selection aspect is interesting. This paper conducts extensive experiments on different data selection methods and the gained insights can be useful to the community. Also, the proposed data selection technique is effective based on the experimental results.
3. The paper also curates a large-scale financial corpus.

### Weaknesses
1. The tasks for experimental are mainly classification tasks, which is limited as the LLM is powerful and should be evaluated on more complicated tasks or at least some generation tasks. I know the paper conducts qualitative evaluation on some QA samples. Is there any generation task in financial domain that you can use to systematically evaluate FinPythia-6.9B?
2. This is mainly an empirical paper and does not have solid theoretical supports.
3. ETS gives better result than ETA, but I'm wondering if knowing task data is a reasonable assumption in the real world as a foundation model is mainly designed for multiple tasks. 
4. The paper involves a lot of abbreviations that are quite similar, which makes the paper hard to read. I suggest using the full word "task-specific" and "task-agnostic" and removing the same suffix "DACP" in the abbreviations.

### Questions
1. The paper argues that good data selection can make continual pre-training data-efficient and maintain the performance on general tasks. Speaking of this part, I think [1] needs to be discussed in the paper. That work also studies the continual pre-training problem and considers preserving the model's general ability from a different perspective.
2. I think "building domain-specific LLMs from scratch" in the abstract may confuse the readers as continual pre-training is actually opposed to pre-training from scratch.

[1] Adapting a Language Model While Preserving its General Knowledge, Ke et al., EMNLP 2022

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper explores domain-specific continual pretraining for LLMs through continued pretraining of Pythia models on a newly curated financial dataset. The authors demonstrate that domain-adaptive continual pretraininig (DACP) can improve performance of an LLM on domain-specific tasks. They then explore methods for improving the efficiency of DACP by selecting data according to different metrics, demonstrating efficiency and performance gains using only 10% of the tokens of the original domain-specfic dataset. Finally, the authors show that general (non-domain-specific) performance is degraded less in the 10% trained models compared to the 100% DACP baseline, suggesting that in addition to gains in domain-specific performance and in training efficiency, the proposed methods help the model retain more of its original domain-nonspecific performance.

### Strengths
The paper has an appealing overall structure (DACP vs baseline, modifications of DACP vs DACP on domain-performance, mod vs DACP on general performance). The questions addressed are of immediate topical relevance to researchers and practitioners of LLMs. The fact that the proposed modifications to DACP yield better specific performance and better generalization performance is a nice contribution.

### Weaknesses
The fundamental questions of the paper are valuable to address, but the conclusions drawn from the answers provided by this paper are not as comprehensive or original as one might hope. The demonstration that training a pretrained LLM with continued pretraining on a domain-specific dataset is more efficient than training from scratch is not a surprise, nor a novelty. The results of the proposed efficient-DACP methods are generally positive (though not universally, and the deficiencies are not explained or addressed), but it is unclear to what extent those positive results might generalize to other domain-specific datasets.

The clarity of the writing could be improved. Some sections (introduction of TACP) are ambiguous regarding important details, or just difficult to read (SoftSampling).

The results of the paper are only presented on 10% and 100% of the domains-specific corpus. Why 10% in particular? It would be useful to understand the tradeoffs involved at differing percentages. The paper would be strengthened by showing the performance curves across other percentages as well. Particularly, how does this perform with as little as 1%? At what dataset size does high-quality data fail improve domain specific performance in continual pretraining? This would be very interesting to know.

The results tables (3 and 4) are very busy and somewhat difficult to parse. The stddevs presented in Table 3 are really large, which makes it difficult to be confident in the significance of the results. The relationship between the columns of Table 1 and the rows of Table 3 is unclear. Overall, the results in the tables support the arguments made in the text, but the tables are difficult to understand due to their arrangement and formatting.

Different metrics are proposed for scoring the quality of the domain-specific data, and their performance varies dramatically across datasets. The paper would be made much stronger with analysis of the strengths and weaknesses of the different metrics, and by explanations of their divergent performance, perhaps with examples. As is, it is difficult to asses which metric one should use, and why. For example, ETA-DACP-ppl uses perplexity as a heuristic for novelty, which is assumed to be desirable. In a high-quality dataset, this might be a reasonable assumption, but in a noisy dataset this will likely score highly noisy examples, leading to a noise-enriched low quality sample. This method performs quite poorly compared to the others, suggesting the dataset may be noisy. To what extent are the other methods just avoiding training on the noisy examples in the full dataset?

### Questions
Questions/suggestions:

Why is performance on FiQA SA so bad? Table 1. This is not necessarily an issue but is an outlier compared to the other datasets so should be mentioned/explained.

TACP: Make explicitly clear the similarities/differences between TACP / DACP / generic finetuning while introducing TACP. What is the difference between domain-specificity and task-specificity? Is domain-specific LM data on a topic, and Task-specific is data formatted in a specific format (ie QA style)? From the text, it is unclear how DACP and TACP differ except in the amount of {domain/task}-specific data available, and this distinction isnt made until section 2.4, but should be made in 2.3. Also, is there a citation for TACP?

Soft Sampling (2.4.3): please rewrite this section to make more clear the methodology. The first couple sentences leave ambiguous whether the softness refers to you are probabilistic sampling or uniform sampling with example-loss-weighting, and the sentences that clarify that ambiguity aren't presented til midway through the next paragraph.

Table 1:
Put the columns for the 1B models and 7B models adjacent to each other, so the comparison is easily made. Consider also adding diffs for the FinPythia performance showing the delta compared to Pythia, as the diff is the point of the table.

Table 3:
The stddev of all models, even on the averaged F1 is very high. This makes it hard to know how significant the results are.
Shouldn't the Pythia 1B column in Table 1 correspond to the same evaluation as the Pythia 1B row in Table 3? Please clarify.
Which row does FinPythia 1B from Table 1 correspond to in Table 3? DACP 100%?

Table 4:
HellaSwag isn't mentioned anywhere else in the text of the paper. Please introduce all datasets in use here (probably in 3.1, optionally in 4.2). Also, why not present the Pile test loss as well?
Since the point of the table is to demonstrate low deltas over the baseline Pythia model, please show the actual deltas in the text of the table. This would help the table make its point and significantly improve readability.

Table 2: this takes up a ton of room and doesn't add much. Consider trimming to 2 examples or moving to appendix.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair
