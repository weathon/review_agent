# Revisiting Multilingual Data Mixtures in Language Model Pretraining

- Avg Score: 5.50
- Decision: Reject
- Scores: 4, 4, 8, 6

## Abstract
The impact of different multilingual data mixtures in pretraining large language models (LLMs) has been a topic of ongoing debate, often raising concerns about potential trade-offs between language coverage and model performance (i.e., the curse of multilinguality). In this work, we investigate these assumptions by training 1B and 3B parameter LLMs on diverse multilingual corpora, varying the number
of languages from 25 to 400. Our study challenges common beliefs surrounding multilingual training. First, we find that combining English and multilingual data does not necessarily degrade the in-language performance of either group, provided that languages have a sufficient number of tokens included in the pretraining corpus. Second, we observe that using English as a pivot language (i.e., the language with the highest data proportion) yields benefits across language groups, and contrary to expectations, selecting a pivot language from within a specific group does not consistently improve performance for languages within that family. Lastly, we do not observe a significant "curse of multilinguality" as the number of training languages increases in models at this scale. Our findings suggest that multilingual data, when balanced appropriately, can enhance language model capabilities without compromising performance, even in low-resource settings.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper investigates how multilingual data mixtures affect LLM pretraining, focusing on the "curse of multilinguality." The authors train 1B–3B parameter models with up to 400 languages using the mC4 and FineWeb2 datasets. They test four key assumptions:

1. English hurts multilinguality: Increasing English data does not necessarily degrade non-English performance.

2. Family-specific pivots help: Using English as a pivot language performs as well as or better than family-specific pivots.

3. Curriculum learning helps: The order of language introduction does not significantly reduce interference.

4. Curse of multilinguality: Adding more languages does not inherently hurt performance if data is balanced.

Their conclusion: Multilingual data, when balanced appropriately, can enhance language model
capabilities without compromising performance, even in low-resource settings.

### Strengths
- Comprehensive experimental design: Systematic testing of four major hypotheses at multiple scales (1B and 3B models).

- Relevant and timely topic: Multilingual data efficiency and scaling are crucial for LLM research.

- Clear presentation: Good structure, consistent visualizations, and explicit takeaways per assumption.

- Empirical insights: Provides evidence that challenges widespread assumptions about multilinguality trade-offs.

### Weaknesses
- Questionable assumptions / missing motivation: The origins of “Assumptions 1–4” are not sufficiently grounded in prior literature. For example, a citation of the works that assumed or discussed Assumption 1 is not listed.

- Tokenizer bias as a confounder: The use of the fixed Mistral-Nemo-Base-2407 tokenizer, which primarily includes English tokens, introduces a major confounding factor. Since all languages are tokenized with this fixed vocabulary, tokenization efficiency and representation differ across languages. This could significantly skew the results toward English, especially in Assumptions 1 and 2.

- Missing related work: Recent peer-reviewed work on the same topic, but focused on continual multilingual pretraining (https://arxiv.org/pdf/2504.04152, COLM 2025), is omitted.

- Citations around L304 on the number of languages these models are trained on are needed. The citations are sometimes misformatted (e.g., L572 “Josh Gpt-4 Team”?).

- You can also define the curse of multilinguality for a language or a group of languages: some may benefit when other languages are added, while others degrade. Showing only the overall trend that “the curse” is absent in your setting may not be very informative. You might not have enough training tokens to reach the point where performance degrades and adding more languages is not helpful. It would be useful to highlight some specific languages that are hurt and others that benefit as well (see https://aclanthology.org/2023.acl-long.61.pdf, Table 9).


- Limited findings: Typically, models are trained on many languages. If practitioners are already including multiple languages and with increasing the proportion of English data (aligned with the findings suggestions): how can these results inform or influence current pretraining practices for such models?

### Questions
- It is hard to believe that adding more multilingual data in Assumption 1 (under a fixed total budget) does not significantly improve downstream multilingual performance. If that is indeed the case, how are models expected to improve their performance in multilingual settings, and why does this occur?


- Could retraining with a multilingual tokenizer, or using a separate tokenizer for each experiment, alter the conclusions? How much do tokenizer biases affect results, especially for non-Latin scripts?

- You have two experiments on the imbalance between languages (natural and temperature) in assumption 4, but I still could not figure out whether the total number of tokens for each experiment is 100B. Does this mean that if there are 25 languages, the total is 100B, and if there are 400 languages, it is still 100B? We need experiments for both cases: one where the total size is fixed and one where the total size grows with the number of languages.

- If the order does not matter, what type of curriculum learning helps?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates some questions related to multilingual data mixtures for LLM pre-training by training 1–3B parameter LLMs on multilingual corpora covering 25–400 languages.

The examined assumptions and corresponding findings are:  
1\. More English data comes at the cost of performance in other languages. -> Increasing the amount of English data does not necessarily degrade multilingual performance.

2\. Languages within the same family offer the strongest boost to multilingual generalization. -> English serves as a broadly effective pivotal language but in low-resource settings, typologically similar pivots can be important.

3\. Curriculum-based language introduction mitigates negative interference. -> Curriculum learning does not have an observed practical impact.

4\. Adding more languages to a pretraining mixture reduces performance. -> Performance reduction is due to finite capacity of models rather than the addition of languages.

### Strengths
1. The study investigates a number of different research questions that can be taken into consideration when pre-training large multilingual models.
    
2. The paper studies models at two different parameter sizes, which improves the generality of the findings.
    
3. The study investigates scaling with different numbers of languages, including a larger number of languages.
    
4. Ablation studies are generally well-thought out, such as comparing fixed total and fixed multilingual budget or studying pivot languages from different language families.

### Weaknesses
As a whole, I have concerns regarding the relevance of the studied assumptions and as a result the novelty of the corresponding insights.

Re #1: recent closed-source and open-source models have much improved English and multilingual performance, which indicates that more English data does not come at the cost of performance in other languages (up to a %), contrary to the assumption stated in the paper.

Re #3: curriculum learning has not been used in the pre-training of a state-of-the-art LLM as far as I’m aware, so finding that curriculum learning does not help is not surprising IMO. Studies that demonstrate the circumstances under which curriculum learning can be effective (such as this one: [https://arxiv.org/abs/2506.11300](https://arxiv.org/abs/2506.11300)) seem more surprising in that regard.

Re #4: The finding that the curse of multilinguality relates to model capacity rather than simply adding more languages has already been stated in the original XLM-R paper introducing the term. The XLM-R authors write: “Model capacity […] is constrained due to practical considerations […]. For a fixed size model, the per-language capacity decreases as we increase the number of languages.” This observation is also what motivated and led to the success of adapters for multilingual model adaptation. So I’m unsure of the novelty of the finding given the consensus in prior work.

I think the study would be stronger if you highlighted which findings are novel more clearly and explored the actually novel or practically relevant ones more in-depth. Right now, the findings that are already established or well-known take attention away from the other parts.

For the study of assumption #1, I am missing a comparison to a setting where the model is only trained on English data so that English performance is maximized. This reflects the most common setting in practice where it is key to compete on English performance and multilingual performance is increased as long as it does not decrease English performance. Extending the graphs in Figures 1 and 2 to English Data Proportion 100% would provide this comparison.

A confounding factor in the study of multilingual data mixtures is the tokenizer. A tokenizer with low compression rates for some languages leads to the model seeing less data (as more tokens are necessary to represent each data point) for those languages. So in addition to looking at the relative token budget across languages, it would be good to control for the tokenizer’s compression rate in some experiments. Alternatively, doing similar experiments with another tokenizer (with a markedly different compression distribution) would also shed additional light on this. Here are a few studies on the effects of tokenization on language models: [https://arxiv.org/abs/2012.15613](https://arxiv.org/abs/2012.15613), [https://aclanthology.org/2023.emnlp-main.614/](https://aclanthology.org/2023.emnlp-main.614/)

As has been shown in the past, the behavior of models can change dramatically with larger parameter sizes. The examined assumptions such as the curse of multilinguality or the trade-off between English and multilingual performance are also closely tied to model capacity. Running experiments on a model in the 20–40B range would provide evidence that the provided findings generalize to sizes of models that are more commonly used in practice these days.

The current experimental setting restricts training to a single epoch as far as I’m aware. This doesn’t capture the impact of repeated data in pre-training, which is relevant for under-represented languages where repeating data may be necessary to see a sufficient amounts of tokens. Prior work ([https://arxiv.org/abs/2305.16264](https://arxiv.org/abs/2305.16264)) has shown that in data-constrained settings, repeating data up to 4 times performs similarly to training on equivalent amounts of unique data. It would be useful to understand to what the number of training epochs impacts the paper’s findings.

### Questions
Line 368: What does it mean for performance to remain stable here? Is this based on a weighted or uniform average of performance across languages?

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
4

### Summary
The paper primarily studies the impact of multilingual data mixtures during pretraining under various conditions on downstream performance. It challenges several previously held beliefs about multilingual pretraining, showing that (i) when both pivot languages and less represented languages are present in sufficient quantities (even if their ratios are unequal), the model can perform well on both types of languages, and (ii) the so-called “curse of multilinguality” arises mainly due to limited model capacity and the artificial amplification of low-resource data points.

### Strengths
1. Tackles an important yet relatively understudied topic of multilingual fairness in large language models.
2. The paper is well written. The narrative is clear, and the experiments are comprehensive and carefully designed.

Overall, this is a strong paper that will be of interest to the community. I recommend acceptance, assuming the minor concerns outlined below are addressed during the rebuttal.

### Weaknesses
1. The pivot language experiments are limited to the Cyrillic and Slavic languages. It would strengthen the paper to include other language families to confirm the generality of the results, especially given that one of the paper’s main claimed strengths is its broad coverage of languages.
2. Table 1 could be made stronger by studying varying percentages of English instead of keeping it fixed at 40%. If the claim is that beyond a certain English data threshold, the number of additional languages has minimal effect, then I would expect to see comparisons across different proportions of English data.
3. I am interested to know what Figure 4c would look like on the non-pivot (multilingual) tasks. I think this would be particularly interesting to see as opposed to just the pivot languages because the model is explicitly trained on those anyway.

### Questions
1. In Figures 1 and 2, why is the validation loss for the multilingual model lower than that for the English-only model, even when English constitutes the majority of the data?
2. At the point where the two curves intersect (if they ever) in the unconstrained data setting (Figure 1b), is the multilingual loss still stable?

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
3

### Summary
This paper trains a number of multilingual language models to investigate several previously documented assumptions about how multilingual data mixtures affect a model’s quality. In particular, the paper:
* Trains multilingual models with a fixed amount of non-English data, but while varying the amount of English. It finds that the amount of English does not affect performance in non-English languages.
* Trains multilingual models with English vs. other languages used as “pivot” languages. They find that language family is only important for transfer in lower-resource scenarios.
* Trains multilingual models with multiple curricula (e.g., starting with English, then introducing other pivot languages, then lower-resource languages). This does not seem to significantly affect the performance in lower-resource languages.
* Trains multilingual models with varying numbers of languages. As long as the amount of data per language is kept fixed, performance is not harmed by increasing the number of languages.

### Strengths
The paper provides a “short survey” of several prior results on how to design multilingual data mixtures for training multilingual language models. The paper then provides evidence contrary to important beliefs propagated by these prior papers.

The paper is very well written and organised, with clear assumptions and takeaways from each experiment.

### Weaknesses
While this paper examines several prior assumptions about multilingual data mixtures, each assumption is not necessarily comprehensively examined. Further, the paper does not provide enough experiments to show *why* its results differ from prior works.

### Questions
> Selecting a high-resource pivot language from within a specific family (e.g., Russian for Slavic languages) does not consistently enhance performance across languages in that family

As far as I know, Slavic is not a language family per se, but a branch in the Indo-European language family. It is thus still in the same language family (i.e., Indo-European) as English (which is from the West Germanic branch of the Indo-European family).


> Figures.

The grey background present in many figures makes them harder to read when the paper is printed in black and white. I’d suggest removing it.

>  English Hurts Multilinguality

Related to this result. Wendler et al. (2024) show that large models can leverage one (pivot) language’s circuits when processing other languages. And Schäfer et al. (2024) showed that performance might transfer from a pivot to non-pivot languages when using imbalance language mixtures, but not when training models in a balanced setting.


* Wendler et al. 2024. [Do Llamas Work in English? On the Latent Language of Multilingual Transformers](https://aclanthology.org/2024.acl-long.820/). In: ACL.
* Schäfer et al. 2024. [The Role of Language Imbalance in Cross-lingual Generalisation: Insights from Cloned Language Experiments.](https://arxiv.org/abs/2404.07982). In: arXiv.

### Soundness
4

### Presentation
4

### Contribution
2
