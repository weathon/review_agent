# Seq vs Seq: An Open Suite of Paired Encoders and Decoders

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 6, 6, 4

## Abstract
The large language model (LLM) community focuses almost exclusively on decoder-only language models, since they are easier to use for text generation. However, a large subset of the community still uses encoder-only models for tasks such as classification or retrieval. Previous work has attempted to compare these architectures, but is forced to make comparisons with models that have different numbers of parameters, training techniques, and datasets. We introduce the SOTA open-data Ettin suite of models: paired encoder-only and decoder-only models ranging from 17 million parameters to 1 billion, trained on up to 2 trillion tokens. Using the same recipe for both encoder-only and decoder-only models produces SOTA recipes in both categories for their respective sizes, beating ModernBERT as an encoder and Llama 3.2 and SmolLM2 as decoders. Like previous work, we find that encoder-only models excel at classification and retrieval tasks while decoders excel at generative tasks. However, we show that adapting a decoder model to encoder tasks (and vice versa) through continued training is subpar compared to using only the reverse objective (i.e. a 400M encoder outperforms a 1B decoder on MNLI, and vice versa for generative tasks). We open-source all artifacts of this study including training data, training order segmented by checkpoint, and 200+ checkpoints to allow future work to analyze or extend all aspects of training.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces SEQ vs SEQ (ETTIN Suite), an open-source suite of paired encoder-only and decoder-only language models trained with identical data, architecture, and training recipes. The suite spans models from 17M to 1B parameters, enabling an “apples-to-apples” comparison of encoder and decoder architectures across scaling trends. The authors also provide 200+ checkpoints, token-level training data ordering, and open datasets, making this a valuable resource for community research. Like existing works, encoder-only models outperform decoders on classification and retrieval tasks (e.g., GLUE, MTEB, MS MARCO). Decoder-only models dominate generative tasks (e.g., ARC, HellaSwag, TriviaQA). Adapting models to the opposite objective through continued cross-objective pretraining (e.g., MLM-trained decoders, CLM-trained encoders) is ineffective can not beat the single native baseline. Additionally, the paper includes a case study on gender bias, showing diverging pronoun prediction behaviors between encoder and decoder models.

### Strengths
1. An apple-to-apple suite of encoder and decoder models trained with identical recipes and data. This eliminates confounding factors present in previous studies. 
2. All model weights, checkpoints, and training data orders are released. This enables reproducibility and further analysis.
3. Both encoder and decoder models achieve state-of-the-art performance for their size, outperforming ModernBERT (encoder) and SmolLM2/LLaMA 3.2 1B (decoder) baselines.
4. Demonstrates that continued training of decoders with MLM (or encoders with CLM) fails to match native training, even when using 50B tokens of additional training. 
5. Covers 6 model sizes (17M → 1B). This allows to study scaling trends across both architectures.

### Weaknesses
1. Cross-training adapting decoder into encoder often applies masked language modeling. Is it helpful also add masked next token prediction or unsupervised contrastive learning like [1].
2. The evaluation is centered on GLUE, MTEB, and knowledge-based benchmark. Can you also math or coding benchmarks to see the reasoning abilities of the models.

References:
BehnamGhader et al. LLM2Vec: Large Language Models Are Secretly Powerful Text Encoders. COLM 24

### Questions
1. Why are the results of TQA in Table 4 exceptionally low for some models? 
2. For smaller models (<=70M), adapting encoder to decoder or decoder to encoder seem little gap from native pretraining. If we increase the tokens for adapting phase, would that be helpful? Is there any scaling behavior that can bridge the gap between adapting model performance and native one?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper presents a set of open-source and open-data text models, consisting of encoders (trained with a masked language modeling objective) and decoders (trained as autoregressive language models) of sizes from 17M to 1B parameters.  The main goal is to enable direct comparison of encoders and decoders on a variety of tasks:  classification and retrieval tasks, where encoders tend to be preferred, and generation tasks where causal LMs are typically best.  For each model size the authors train a pair of encoder and decoder trained with the same hyperparameters, as well as an encoder further trained with a causal LM loss and a decoder further trained with MLM.  Whereas previous such comparisons have compared models of varying sizes and training setups, this work allows for more direct comparison.  The main findings are that (1) as expected, encoders outperform decoders of the same size on classification and retrieval tasks and vice versa for generation tasks and (2) continued pre-training with the other loss (e.g. decoders further trained with MLM) does not undo this trend.  As a case study of additional analyses that this model family enables, the authors also study the distribution of gender pronouns for models of different type and size, finding for example that encoders tend to favor more neutral pronouns and that the gender distribution changes with model size.

The main contribution of the paper is the open models themselves.  The findings are interesting and demonstrate the value of the resource, but are not very far-reaching.  I therefore consider this paper to be mainly a "resource" paper.  A natural question is whether the resource rises to the level of a research contribution and therefore is worthy of a conference publication.  I believe the contribution is important and am recommending acceptance.  That being said, the paper could make a more convincing research contribution if it provided more detail and analysis, beyond just the models themselves (see below under "weaknesses"), and for this reason my recommendation is marginal acceptance.

### Strengths
- A large set of open models trained on open data makes it possible to make comparisons that were previously only approximate (e.g. comparing encoders an decoders of different sizes, or of similar sizes but with otherwise very different training conditions).

- The findings are interesting.  It is especially interesting to see that decoders can't be easily "converted" into equally strong encoders via continued pre-training, and vice versa.

- The paper is generally presented well.  It provides good background on existing models and findings and lays out the new work clearly.

### Weaknesses
- The paper should provide more detail about how the hyperparameters were chosen.  For example, it is not obvious to me that using identical hyperparameters for encoders and decoders is the best choice.  Ideally there would be some tuning done and a study of the sensitivity to hyperparameter choices.  Could the results be improved with different choices of hyperparameters?  Could they change enough to modify the findings?  

- The results are given without any analysis into why the findings are as they are, especially the less obvious ones related to continued pre-training and gender pronouns.  I recognize that it may not be easy to do, but I would expect some attempt at understanding the results and studying how they may be affected by aspects of the data distribution, number of training iterations, or other variables.  This is related to the first point above, but has more to do with explanation of the findings.

- A minor weakness:  While the paper is generally well-written, there are quite a few minor grammatical errors throughout.  A thorough proofreading would help.

- Also minor:  Citations should be provided to support the statement "Part of this gap is due to the sentiment within the community that decoders can be adapted for use in tasks that were once predominantly encoder-focused (e.g. classification, embeddings)".

### Questions
- Did you consider including larger models, say up to 3B or 7B?  The work is valuable without it, and to some extent the maximum size is arbitrary.  However, several B parameters is a fairly typical size for research models, and it would be interesting to know whether there is any "phase transition" as model size increases further.

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
Authors opensource pairs of encoder-only and decoder-only models along with training pipeline. They explore and compare the performance in classification, retrieval, and generative task.

### Strengths
1. The shift from decoder to encoder on LLMs is still not scientifically clear. This paper present paired encoder and decoder models for people to study on and has done some initial exploration.
2. The paper is fully opensource in terms of weights and training pipeline.

### Weaknesses
1. There are research on turning decoder-only models to encoders like [1], which authors may need to take into account when claiming “encoder models are better at classification”. The possibility of “with proper finetuning, decoder-only model outperforms encoder-only model on non-generative tasks” cannot be ruled out by the current set of experiments. Therefore some statement could be premature/misleading.
2. Authors’ efforts in opensourcing large-scale encoder models (compared to other encoder-only models) are unprecedented. However, compared to some other alternatives to decoder-only LLMs (e.g. MAMBA, diffusion LLM), the current efforts lack a 7B-scale model, which is (in my opinion) usually considered as the smallest usable model for fair comparison with decoder-only LLMs.
[1] LLM2Vec: Large Language Models Are Secretly Powerful Text Encoders. BehnamGhader et al. COLM 2024
[2] Falcon Mamba: The First Competitive Attention-free 7B Language Model, Zuo et al. Arxiv
[3] Large Language Diffusion Models. Nie et al. NeurIPS 2025.

### Questions
I believe such opensource efforts should be encouraged in the community. I am willing to increase my score if W1 is properly addressed by proper experiments. And would like to hear author's arguments on not conducting experiments at a larger scale for encoder-only models.

### Soundness
2

### Presentation
3

### Contribution
3
