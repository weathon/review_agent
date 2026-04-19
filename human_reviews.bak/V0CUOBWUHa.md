# Scaling Sentence Embeddings with Large Language Models

- Decision: Reject
- Scores: 6, 5, 6

## Abstract
Large language models (LLMs) have recently garnered significant interest. With in-context learning, LLMs achieve impressive results in various natural language tasks. However, the application of LLMs to sentence embeddings remains an area of ongoing research. In this work, we propose an in-context learning-based method aimed at improving sentence embeddings performance. Our approach involves adapting the previous prompt-based representation method for autoregressive models, constructing a demonstration set that enables LLMs to perform in-context learning, and scaling up the LLMs to different model sizes. Through extensive experiments, in-context learning enables LLMs to generate high-quality sentence embeddings without any fine-tuning. It helps LLMs achieve performance comparable to current contrastive learning methods. By scaling model size, we find scaling to more than tens of billion parameters harms the performance on semantic textual similarity (STS) tasks. However, the largest model outperforms other counterparts and achieves the new state-of-the-art result on transfer tasks. We also fine-tune LLMs with current contrastive learning approach, and the 2.7B OPT model, incorporating our prompt-based method, surpasses the performance of 4.8B ST5, achieving the new state-of-the-art results on STS tasks.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper works on generating sentence embeddings using large language models. First,  the authors design a specific prompt  to compress the semantic of an input sentence into a single word.  Then, this paper investigates zero-shot,  in-context and fine-tuning settings of sentence embedding learning.  For in-context learning,  this paper proposes a demonstration selection method for inducing good sentence representations.  For fine-tuning, to solve the large memory issue,  the authors use QLoRA to perform contrastive learning.  Empirical results on common sentence embedding evaluation benchmarks with both OPT and LLaMA series models show that the proposed method can match (or even exceed) the performance of pretrained language models (such as BERT).

### Strengths
1. The writing is easy to follow and the idea is well presented. 
2. The proposed prompt, in-context demonstration and fine-tuning method solve the specific issues of scaling large language models for sentence embedding learning. 
3. The experimental results are effective on both the SentEval and Transfer settings compared to BERT-base based contrastive learning method.

### Weaknesses
1. In Table 1,  only the results based on OPT are presented.  Why not also including the results based on LLaMA? 
2. In Table 1,  the best configuration PromptEOL+ICL+ OPT (6.7B) does not show clear advantages than PromptRoBERTa (123M). 
3. For the in-context setting,  why only use one demonstration?   In Table 1,  comparing PromptEOL+ICL + OPT with baselines models  is not fair since the baseline models do not use the development set. 
4. When the model size increases, the performance does always not increase.  Especially, the 13B, 30B, and 60B models do not perform better than smaller models such as 1.3B and 6B models. 
5. The overall method is a little bit heavy. It is worth to discuss whether we should improve the sentence embeddings using large language models.

### Questions
1. Do you also try LLaMA 2? 
2. In Equation 1, why using the last token hidden state as the sentence representation instead of the representation vector of the last generated token using the explicit one word prompt? 
3. For the fine-tuning, do you also try including in-context demonstration for the fine-tuning? 

Minors: 

The citation format is not correct. Please correct all of them.  Try to use the cite command in a correct way.

### Soundness
3 good

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
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes a set of methods leveraging LLMs for sentence embeddings.

* It introduces a prompting strategy with explicit one word limitation, pushing the model to condense as much information as possible into the last hidden representation. This method is an adaptation of PromptBERT's approach for autoregression models.
* It leverages in-context-learning in order to improve the quality of sentence embeddings. To this end, it relies on two approaches: 1) It generates one-word summaries of sentences from the STS training set using GPT-3.5. 2) It leverages entries from the Oxford dictionary. The concatenation of samples from these two sources is then incorporated into the LLM's prompt.
* It leverages fine-tuning with contrastive learning to further improve the quality of sentence embeddings. It does so by leveraging qLORA and training on supervised datasets such as SNLI and MNLI.


The paper's conclusions are as follow:
* The explicit one-word limitation prompt improves the quality of sentence embeddings derived from OPT on STS benchmarks.
* In-context learning and supervised fine-tuning improve the performance on STS benchmarks, allowing the proposed solution to beat the state of the art. However, the resulting embeddings do not transfer as well to other tasks.
* In the paper's proposed setup, the largest base models do not have a clear performance advantage: best results on STS without fine-tuning are obtained with OPT's variants ranging between 1.3 and 6.7B parameters.

### Strengths
* Originality: the proposed PromptEOL is a novel adaptation of the BERT prompting paradigm for sentence representations. The prompting strategy combining GPT-augmented STS sentences and oxford definitions is novel as well, and the use of qLORA to make contrastive fine-tuning feasible for larger models shows creativity in putting together existing solutions.

* Quality: The experiments are well-devised and executed. The proposed methods are simple and beat the state of the art on semantic textual similarity benchmarks.

* Clarity: The paper articulates very clearly its methodology. It is easy to read and describes well the corresponding pre-existing work. It motivates very well the choice of an explicit one-word prompt, the value of in-context learning and the need for quantization in order to fine-tune the largest models in a contrastive learning setup.

* Significance: while the results on STS benchmarks look good, both in-context learning and contrastive fine-tuning do not show incremental value on transfer tasks. The relatively low generalisation capabilities of these methods limit greatly the appeal of such techniques for the average practitioner, as most real-life applications of sentence embeddings are not for semantic textual similarity.

### Weaknesses
* While it is helpful to the reader to see the entire distribution of Spearman correlations, it may be relevant to give more details on how the two sources for ICL data impact the quality of downstream representations. The 1/3-2/3 mix of STS sentences vs Oxford definitions would benefit from an explicit ablation.

* The value of ICL and CSE is demonstrated only for OPT. Indeed, table 1 is missing results that would demonstrate the added value of ICL and CSE on Llama.

* The proposed methods (in-context learning and possibly fine-tuning) are performing worse than simple explicit-one-word-limit prompting on transfer tasks.

* It is not clear what section 5.2 demonstrates:
  * first, the text mentions "in-context learning examples that were obtained from each model on the STS-B development set", while the table caption reads "In-context learning examples used in various model size". The paper states clearly in section 3.2 that the in-context learning examples (1) come from the STS-B training set and (2) are not generated / obtained from the model itself.
  * second, the method used to sample the data from table 4 is not described, and the meaning of the "Improve" column is not clear: does it correspond to the improvement coming from one additional sentence in the prompt? Or from the addition of the 100s of sentences in the ICL prompt? In any case, it seems premature to draw generic conclusions such as "related examples are usually more implicit" from a sample size of 1 from each model. Appending 5-10 random samples of each category in the appendix would give more compelling evidence for this.
  * finally, it is not clear how to relate the findings of section 5.2 to the overall quality of the sentence embeddings introduced by this work.

* Typos and errors:
  * table 6, the first group of data rows should mention (16-bit)
  * table 8, the ordering is not the same between the "Without fine-tuning" and "Fine-tuning on unsupervised datasets" groups of rows.

### Questions
1. It would seem that some experiments have been run only on OPT, while others have been run on OPT and Llama. It would be helpful to have all experiments run on both models to show that the conclusions are robust to the choice of base LLM.

2. How was the data mix for ICL (1/3 STS sentences, 2/3 Oxford definitions) devised? Are they both necessary to achieve good performance? A proper ablation of this setup would be helpful.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper investigates how to better leverage large language models (LLMs) for generating sentence representations, traditionally obtained from smaller encoder-based models like BERT variants. 
It introduces two approaches. 
Initially, it adopts in-context learning, similar to the utilization of LLMs in other tasks. 
Employing the "Explicit One word Limitation (EOL)"—which posits that decoder-based models can produce viable sentence-level representations when prompted to summarize a sentence in a single word—sentence-to-word pair contexts are used to enhance representation derivation. 
Decoder models, without any fine-tuning, showed performance on par with existing contrastive learning approaches. 
Additionally, the authors explored fine-tuning decoder models using the prevalent contrastive learning framework in sentence representation research, employing the parameter-efficient technique known as QLoRA. 
The findings reveal that fine-tuning with contrastive learning notably benefits larger decoder models, surpassing smaller encoder models in both Semantic Textual Similarity (STS) benchmarks and transfer tasks for classification.

### Strengths
- Suggested a variety of plausible methods for utilizing Large Language Models (LLMs) to compute sentence representations.
- Explored both in-context learning and fine-tuning approaches with LLMs, encompassing a broad spectrum of potential applications for these models.
- Introduced a straightforward yet insightful technique for integrating in-context learning into the sentence representation learning paradigm.

### Weaknesses
- While the methods proposed are sound, they consist of previously suggested and widely implemented techniques, which diminishes the novelty aspect of the work.
- Contrary to SimCSE, the in-context learning approach depends on the use of the STS-B dataset, including its training and validation components, which could potentially confer an unfair advantage to the method.
- There appears to be no direct link between the two proposed methods; that is, the approach based on in-context learning and the one utilizing contrastive learning.

### Questions
- I'm curious whether the authors have any insights or hypotheses as to why (much) larger models (over 10B) do not excel as expected in computing sentence representations, which contrasts with their effectiveness in other standard applications.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
