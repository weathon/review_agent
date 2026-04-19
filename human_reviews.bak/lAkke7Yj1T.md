# Paramanu: A Family of Novel Efficient Generative Foundation Language Models for Indian Languages

- Decision: Reject
- Scores: 3, 1, 3, 5

## Abstract
We present PARAMANU (which means "atom" in multiple Indian languages), a family of novel language models for Indian languages.
It is a collection of auto-regressive monolingual, bilingual, and multilingual Indian language models pretrained from scratch, currently covering 10 Indian languages (Assamese, Bangla, Hindi, Konkani, Maithili, Marathi, Odia, Sanskrit, Tamil, Telugu) across 5 scripts (Bangla, Devanagari, Odia, Tamil, Telugu).
The models are pretrained with a context size of 1024 on a single GPU, and are of varying sizes ranging from 13.29\,M to 367.5\,M parameters. We proposed a RoPE embedding scaling method that enables us to pretrain language models from scratch at larger sequence length context size on single GPU without increased GPU memory. We have also developed an efficient and advanced novel tokenizer with least fertility score among existing LLMs for Indian languages using a combination of BPE and Unigram that can also tokenize unseen languages written in the same script or the Roman script. We also proposed language specific tokenization for multilingual models and domain specific tokenization for monolingual language models. In order to avoid the "curse of multi-linguality" in our multilingual "mParamanu" model, we pretrained on comparable corpora by typological grouping using the same script. We proposed and performed pretraining for more than 1 epoch of training for most of our language models. From our results, we observed the language transfer phenomenon from low resource to high resource within languages of the same script and typology. We performed human evaluation of our pretrained models for open end text generation on grammar, coherence, creativity, and factuality metrics for several languages.
Our Paramanu models outperformed standard and multilingual large language models (LLMs) by a large margin in performance despite being smaller in size by 64 to 20 times. We studied the impact of language specific tokenization versus language agnostic tokenization for bilingual language modeling. We also studied the impact of BPE versus Unigram tokenization for Devanagari script languages. We further created instruction-tuning datasets and instruction-tuned our pretrained models on 23,000 instructions in respective languages except Hindi, for which we used 75,000 instructions. Comparison with multilingual LLMs on various commonsense reasoning benchmarks for natural language understanding, natural language inference, and machine reading comprehension shows the advantage of our models. The performance of our Paramanu models leads to the conclusion that high quality generative language models are possible without high amount of compute power (FLOPS) and enormous number of parameters.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
In this paper, the authors present a collection of small-scale auto-regressive, monolingual, bilingual and multilingual pretrained decoder-only LMs for 10 Indic languages covering 5 distinct scripts, supporting a context window of 1024. The authors also present a tokenization scheme combining both unigram tokenization and byte-pair encoding. Additionally, they propose an engineering method to scale down position IDs to allow longer context pre-training than maximum permissible context length on the physical memory.

### Strengths
- The extensive experiments showcase the capabilities of language-specific Paramanu-family LMs on multiple benchmarks (Belebele, XCOPA, MMLU, ARC, XNLI, HellaSwag, etc.) against Large-scale multilingual LMs which are $\geq$ 20 times the parameter count of Paramanu models.

### Weaknesses
- The paper is difficult to follow. At many points in the paper, the authors provide little to no background on the proposed approaches and evaluation metrics. For example, no background is provided for the Fertility metric, which is used to show the effectiveness of the mBharat tokenizer (Figure 2). Similarly, the perplexity metric is vaguely mentioned only in Appendix C.0.1 and is not referenced in the main text. This might make the paper difficult to read for people who are less familiar with this area of work. 

- Instruction tuning details are underspecified/inconsistent. Abstract, Section 1 and Section 3.2 mention that 23K instances are used for instruction tuning Paramanu Models. However, Section 3.9 mentions 27K + 52K instances being used for Paramanu-Hindi 356M and 27K instances being used for instruction tuning Paramanu-Bangla 108M.

- In section 3.6, The heuristic used to set overall vocab size to 1750, 1K vocab size for konkani and 750 vocab size for Maithili is not clear.  A vocab size of 1750 may be too small to draw conclusions from. The two tokenization comparisons should be studied on a range of vocab sizes to ensure generalizability. Moreover, it might be interesting to look at language-specific perplexity scores, to validate how well the model trained using the merged-tokenizer performs for both the languages on their own. Overall, the comparison lacks experimental rigor.

- The proposed tokenization approach simply combines two pre-existing works: byte-pair encoding and unigram tokenization via a set operation. 

- Quantitative figures on language and domain-wise data distribution in the pre-training corpora are missed out from the main text of the paper. In my opinion, pre-training corpora is one of the most important aspects of LM pre-training and should be included in the main text. `[This point is not considered while assigning scores]`

### Questions
- Can the authors provide more information on the Human evaluation conducted in this study. Mainly, how many annotators were involved in the evaluation? Does this work quantitatively measure inter-annotator agreement?

- Can the authors clarify the exact number of instruction tuning instances used for each model. If different models used different numbers of instances, explicitly state and explain this in the paper.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
1

### Rating Number
1

### Confidence
4

### Summary
The authors have developed a set of language models focusing on 10 Indian languages. They have studied various tokenization approaches for these languages. Additionally, they have created instruction datasets for these languages. The authors claim superior performance compared to larger models that are not focused on these languages. They have also performed human evaluation focusing on grammar, coherence, creativity, and factuality.

### Strengths
1. The authors curate a large pre-training dataset for Indian languages which are usually underrepresented in the available pre-training corpus.
2. The authors curate a human written instruction dataset for Bengali language.

### Weaknesses
1. The instruction tuning dataset is created in one language (Bengali) and then translated using Google-Translate to the other languages. This is a significant limitation as the quality of the instruction data is not on-par with human written instruction data for most of the languages.
2. The authors do not clearly explain how they merge Unigram and BPE tokenizers. It is not clear how the authors tokenizing a given text. Are they using BPE decoding algorithm or the Unigram decoding algorithm?
3. The choice to exclude English from pre-training is puzzling as there is a lot of available training data available for English. This is a significant limitation as the trained model cannot deal with source code, scientific journals/articles, medical and other technical domain data. The authors have not explained the motivation behind this decision.
4. Human evaluation is performed on just 4 prompts. This is not enough to make any reliable conclusions on the quality of the models. The authors have not reported inter-annotator consistency for the human evaluations or even how many independent human evaluations were taken per sample. Thus the authors claim that their models are better than existing LLMs is not supported.

### Questions
1. What criteria has been used to chose prompts for human evaluation? How many human annotators were there? How was the inconsistency among annotators resolved?
2. How do you combine a BPE and a Unigram based tokenizer?

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This paper presents monolingual, bilingual, and multilingual Indian language models trained from scratch for 10 Indian languages across 5 scripts. The models are small (ranging 13M to 368M parameters, all trained on a single A100 GPU), but shown to outperform recent open-source LLMs for the target languages. In addition to automatic metrics human evaluation is performed. 

To do so, they present both new data and modelling experiments: 
- an instruction-tuning dataset with 23k instructions, automatically translated with Google Translate
- automatically translate existing English benchmarks such as HellaSwag to evaluate Indic languages
- an adapted RoPE positional embedding
- compare unigram and BPE tokenization (at a specific vocabulary size), and combine language-specific and domain-specific tokenizers to create a tokenizer with improved fertility

### Strengths
Trains a variety of small GPT models for 10 Indic languages which outperform significantly larger public models for the target languages. 
Model artifacts and datasets may be of use to future researchers.

### Weaknesses
Many choices are not experimentally validated. 

The main text directly lists results from tables, instead of providing additional insights or analyses.

References very little past work on Indic languages. See for example the work from AI4Bharat on creating models and corpora for the languages described here such as [IndicLLMSuite](https://arxiv.org/abs/2403.06350), which presents instruction-tuning datasets and public models. Tokenization and vocabulary choice for Indic languages has also been the subject of a fair amount of prior work in for example the recent WAT evaluations which is not referenced. It has typically been found significantly beneficial to transliterate or romanize Indic scripts to create a shared vocabulary - see [RomanSetu](https://arxiv.org/abs/2401.14280)'s related work

### Questions
How were the vocabulary sizes (750 and 1k) per language and domain chosen? Such small vocabulary sizes will not allow sufficient merges to result in specialized vocabulary - were these numbers chosen experimentally, or how were they chosen?

Creating multilingual datasets via automatic translation can introduce errors; was there any evaluation or spot checking of the data quality?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper introduces Paramanu models, designed to improve NLP capabilities for ten Indian languages across five scripts. The authors developed monolingual, bilingual, and multilingual models using novel tokenization techniques, optimized for low-resource settings, and a scalable RoPE embedding method that enables efficient pretraining on a single GPU. Evaluations across benchmarks showed that Paramanu models outperform many larger language models, especially on tasks involving grammar, coherence, creativity, and factuality in Indian languages.

### Strengths
- Paramanu models effectively address the low-resource problem for Indian languages with high accuracy across language-specific NLP tasks.
- The paper introduces an efficient RoPE embedding scaling technique that enables larger context sizes without requiring increased GPU memory.
- The novel tokenization approach combining BPE and Unigram tokenizers improves performance, especially for Indian languages with complex morphology.

### Weaknesses
- The model's performance was primarily evaluated on a limited set of benchmarks, potentially limiting insights into other diverse language tasks.
- Some models may be undertrained, as indicated by perplexity scores, suggesting room for improvement with extended training.
- The approach may require further testing to generalize across Indian languages with unique typological features, beyond the ten languages used.

### Questions
-

### Soundness
3

### Presentation
2

### Contribution
2
