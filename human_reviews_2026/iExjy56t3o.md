# TokSuite: Measuring the Impact of Tokenizer Choice on Language Model Behavior

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Tokenizers provide the fundamental basis through which text is represented and processed by language models (LMs). Despite the importance of tokenization, its role in LM performance and behavior is poorly understood due to the challenge of measuring the impact of tokenization in isolation. To address this need, we present TokSuite, a collection of models and a benchmark that supports research into tokenization's influence on LMs. Specifically, we train fourteen models that use different tokenizers but are otherwise identical---using the same architecture, dataset, training budget, and initialization. Additionally, we curate and release a new benchmark that specifically measures model performance subject to real-world perturbations that are likely to influence tokenization. Put together, TokSuite allows robustly decoupling the influence of a model's tokenizer, supporting a series of novel findings that elucidate the respective benefits and shortcomings of a wide range of popular tokenizers.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces TokSuite, a suite of LLMs to study the effect of tokenizer choice on LLMs's behavior and robustness. In addition to the models the paper also introduces several benchmarks to study the robustness of these models under varying real-world like perturbations. 

Some key results are - 

- Different tokenizers perform in different ways on different perturbations (TokenMonster which is English-only is more robust to some perturbations as compared to multilingual tokenizers)
- Byte tokenizers are more robust to multilingual perturbations and subword fragmentation.
- Tokenizer significantly impacts processing ability of certain types of content such as technical content like LaTeX content due to lossy pre-processing

### Strengths
- The paper is clearly motivated and tests several different off-shelf popularly used tokenizers
- The results on the author created benchmarks show clear effects of tokenizer changes
- The unification of shared token initialization is a nice step and shows attention to detail
- Technical content analyses section is quite interesting as well as relevant

### Weaknesses
- I feel an interesting and important axes is comparing the effect of vocab size within a single tokenization method. For example if over the same corpus and using the same tokenization/pretokenization algorithm, the authors learnt multiple vocabs of varying sizes and saw changes in performance then that would be a more clear indicator of effect of tokenizer vocab size as at the moment comparing across tokenizers does not make a lot of sense given a several times larger tokenizer leads to many more parameters for the model to learn.
- The claim in Lines 84-85 seems to be unsupported as it could be because tokenizers are undertrained for non-English languages and had you learnt the vocab over a corpus which has several of the commonly occurring real world issues with other languages the tokenizer would have tokens to handle those issues more gracefully
- The findings about byte-level models seems fairly obvious to me given that is their entire selling point?
- It is hard to make sense of numbers like how different is 0.04 v/s 0.06 drop in line 359. Does it even matter or is it a big difference?
- The custom benchmarks seem quite small and differences may be in bound of variance?

### Questions
- I'm assuming data order is also same? (Line 54-55)
- Line 197, don't you mean intersection?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper aims to systematically explore the robustness of different tokenizers and their impact on language model performance, attempting to isolate the effects of the tokenizer from other training factors.
In addition, the authors propose the TokSuite Benchmark, designed to evaluate the robustness of various tokenizers.

### Strengths
1.The authors constructed a high-quality TokSuite Benchmark.

2.The authors evaluated 14 different tokenizers and provided several insightful conclusions based on the experimental results.

### Weaknesses
1.In Table 1, the average (Avg.) results across the 14 tokenizers show little variation—except for TokenMonster and Tekken, the remaining 12 tokenizers perform similarly. This raises the question of whether the benchmark is sufficiently sensitive to differentiate between the effectiveness of different tokenizers.

2.The benchmark does not evaluate generation, translation, or code-related tasks. Since the impact of tokenization can vary significantly across task types, the applicability of the conclusions may be limited.

3.Regarding the writing and presentation, the main text is text-heavy. Apart from a single table displaying the experimental results, there is a lack of additional visual aids—such as figures or more comprehensive tables—to help readers better understand the findings.

### Questions
1.In Section 3.3, the authors mention that “training models with different tokenizers under the same token budget means that each model has seen a different collection of text.” Given that the actual text exposed to each model can vary significantly due to tokenizer differences, how might this strategy of aligning only the token budget, but not the text, affect the fairness of tokenizer evaluation?

2.The authors choose LLaMA-1B as the model for evaluation. Could they briefly explain the rationale behind selecting this model? Moreover, since the vocabulary size of a tokenizer is often considered to be most effective when adapted to the model’s size, does using a fixed-size model (LLaMA-1B) for all tokenizers—regardless of their vocabulary scale—potentially suppress the performance of tokenizers that are designed for larger or smaller models?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces TokSuite to explore the impact of tokenizer choice on LM behavior, addressing a critical confounding variable in model evaluation. The authors train 14 identical 1B-parameter models, varying only the tokenizer, using a novel "super vocabulary" for shared initialization. To evaluate them, TokSuite also provides a new 5-language benchmark focused on robustness to real-world perturbations like typos and Unicode styling. The study finds that tokenizer algorithms can be more critical for robustness than vocabulary size. The work also identifies universal failures across all models when handling STEM notation and Unicode formatting.

### Strengths
* This paper conducts an in-depth study of tokenizers in large language models. By keeping all other conditions constant and varying only the tokenizer, the authors train 14 different models and perform a unified comparison.
* The design of the TokSuite benchmark is highly targeted. Instead of using standard, clean evaluations, it focuses on “perturbations” that specifically test tokenizer weaknesses, with particular attention to multilingual, math/STEM, and Unicode formatting scenarios.
* The paper carries out extensive and detailed experiments, revealing many new insights and findings.

### Weaknesses
* Most of the reported results only include the mean values, lacking variance and statistical significance tests. Given that many tokenizers show only small differences in performance, could these results be influenced by random factors?
* In this paper, the authors control the total number of training tokens to be consistent across models, but the total number of training texts varies significantly between datasets. Could this be a major factor contributing to performance differences, such as disparities in language knowledge? Therefore, I believe the authors should include experiments where the total number of training texts is also controlled to ensure greater completeness and fairness of the study.

### Questions
* Could the authors provide further scaling experiments on tokenizers, including variations in model size and data volume? I’m curious whether the characteristics of these tokenizers would change when scaled up.
* Could the authors include experiments where the total number of training texts is controlled to be consistent? This would make it easier to compare the impact of different tokenizers under equivalent knowledge conditions.
* Would it be possible to first train on a superset tokenizer, and then continue pretraining on each specific tokenizer? This could help ensure that the number of texts seen does not differ too much, thereby reducing knowledge discrepancies.
* Could the authors provide more detailed experimental results, including variance and statistical significance tests?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The propose a set of tests to understand the impact of tokenizer choices and train a set of lms to evaluate on the benchmark.

Note that my scores will be set after discussion. So dont take my scores serious please.

### Strengths
The study is systematic and broad.

### Weaknesses
Training new models has the problem that the models are comparetively small and hence one needs to wonder if the findings are transferable

I would have loved to see the byte latent transformer in the study as well as decoding methods that target byte level decoding such as Phan 2025.

### Questions
Generally i think this is a wonderful study here are a few things i would like clarification on


Can you justify the choices of the benchmark for me? Do you know if this is in any sense complete? 

I also wonder about langruages like chinese or thai. During generation one token may span only part of the semantic meaning aka the opposite of the prompt boundary problem. Is there a chnace to get an evaluation in that adresses that.

In appendix a can you clarify how all the different bpe tokenizers differ? They all use the same code to my understanding but with different settings what are the settings?


To summarize my biggest concern. You will have to clarify the scope and hence value of the study a bit better. I think the mean reader will say well 5is does not generalize why would i trust the conclusions for my work.

### Soundness
2

### Presentation
2

### Contribution
2
