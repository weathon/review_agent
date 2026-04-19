# Visual Grounding Helps Learn Word Meanings in Low-Data Regimes

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 6, 6, 5

## Abstract
Modern neural language models (LMs) are powerful tools for modeling human sentence production and comprehension, and their internal representations are remarkably well-aligned with representations of language in the human brain. But to achieve these results, LMs must be trained in distinctly un-human-like ways---requiring orders of magnitude more language data than children receive during development, and without any of the accompanying grounding in perception, action, or social behavior.
Do models trained more naturalistically---with grounded supervision---exhibit more human-like language learning?
We investigate this question in the context of *word learning*, a key sub-task in language acquisition.
We train a diverse set of LM architectures, with and without auxiliary supervision from image captioning tasks, on datasets of varying scales. We then evaluate these models on a broad set of benchmarks characterizing models' learning of syntactic categories, lexical relations, semantic features, semantic similarity, and alignment with human neural representations. 
We find that visual supervision can indeed improve the efficiency of word learning. However, these improvements are limited: present almost exclusively in the low-data regime, and sometimes canceled out by the inclusion of rich distributional signals from text. The information conveyed by text and images is not redundant---we find that models mainly driven by visual information yield qualitatively different from those mainly driven by word co-occurrences. However, our results suggest that current multi-modal modeling approaches fail to effectively leverage visual information to build more human-like word representations from human-sized datasets.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The study investigates the effect of visual grounding upon language understanding of language models. For this objective, the authors perform the training of vision-language models in three settings, i.e. language-only, visual-language, and visual-word. Experiments show that visual grounding can aid language learning but mostly in the low-data regime.

### Strengths
The paper has the following strengths:

- The motivation is clearly stated and makes intuitive sense.

- The experiments are conducted thoroughly with various benchmarks, datasets, and prototype models.

- The obtained insights are strongly proven by the experiment results.

### Weaknesses
There are some details which can be raised from the paper:

- Even though the authors provide various evidence to substantiate their claim about the visual grounding ability to help language models, such claim is opposite from the findings attained by previous research of visual grounding [1,2,3] to certain extent. The paper lacks a discussion towards these research.

- Because language models haven proven their effectiveness in multiple applications nowadays, the contribution would become more appealing if the paper discusses what is the impact of its observation upon training language models.

[1] Retrieve, Caption, Generate: Visual Grounding for Enhancing Commonsense in Text Generation Models, AAAI 2022.

[2] Language Adaptive Weight Generation for Multi-task Visual Grounding, CVPR 2023. 

[3] Visual Grounding in Video for Unsupervised Word Translation, CVPR 2020.

[4] Vokenization: Improving Language Understanding with Contextualized, Visual-Grounded Supervision, EMNLP 2020.

### Questions
- Why is the capacity of visual grounding for language understanding differently found from previous works?

- What benefit does the insight that visual grounding is beneficial for language understanding in low-data settings could provide for training language models?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper investigates the influence of visual information on word meaning learning of language models. 

The authors train CLIP, GIT, and GPT-2 models in a controlled way and examine the word learning results on various benchmarks such as word similarity and PoS tagging. 

The experimental findings suggest that visual signals bring slight improvements to word learning, especially under low resource regimes. Interestingly, CLIP + Word predictions and language model predictions correlate poorly, indicating that contrastive learning captures a different pattern for word distribution. GIT performs similarly with pure language model, suggesting that it mainly learns from the cross-word distribution, struggling to balance between vision and text.

Further analysis shows that grounded learning relates to concrete words more like humans than abstract words. Additional results on vision encoder variants, Flamingo architecture, and sentence processing tasks further validate the main results of previous experiments.

### Strengths
- Investigating the effect of vision signals on language acquisition is an exciting topic, and the findings of this paper deepen our understanding of vision-aided language modeling. 

- The experimental setup and results are comprehensive, with different types of models trained using the controlled network architecture and dataset, various benchmarks are adopted for evaluation and different angles are analyzed.

### Weaknesses
- The findings are weak in terms of practicality, as they cannot be directly translated into improvements for existing models. 
- (Minor) Some experimental details are missing. See my questions (Q2 and Q3) below. 
- (Minor) The results bullets are somewhat difficult to follow. I suggest the authors organize these findings into a more natural story to improve the reading experience.

### Questions
Q1: What are the implications of these findings for future research? I am not sure if the better concrete word modeling ability in low-resource regimes would be a bonus when we have abundant image-text pairs or single modality data to train models.  I think some previous explorations such as Vokenization [1], Initialization and plug-in fine-tuning [2] and Distillation [3] can be investigated (or discussed accordingly) with the findings in this paper.

Q2: How was the Flamingo model trained? Did you use the same 6-layer Transformer on the CC12M dataset? This seems not possible as it requires a special network for integrating vision information with cross-attention and an interleaved image-text dataset. 

Q3: What representation of Visual + language models (CLIP) was used for word-based tasks? Did you extract the corresponding word representation from the full sentence to perform the downstream tasks?

[1] Vokenization: Improving language understanding with contextualized, visual-grounded supervision, Tan et al,  https://arxiv.org/abs/2010.06775

[2] How much can clip benefit vision-and-language tasks?, Shen et al,  https://arxiv.org/abs/2107.06383

[3] VIDLANKD: Improving Language Understanding via Video-Distilled Knowledge Transfer, Tang et al,  https://arxiv.org/abs/2107.02681

[4] Can Language Models Understand Physical Concepts? Li et al, https://arxiv.org/abs/2305.14057

### Soundness
4 excellent

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
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper presents an analysis of the word learning / understanding capabilities of visually grounded models. The paper is motivated by human word/language learning which generally requires much smaller magnitudes of text data than language models, and thus investigates what differences lie in the word-learning dynamics/capabilities of models under different regimes of exposure to text and visual data (in the form of images) under a variety of evaluations pertaining to word learning. 

Models evaluated have the following key properties: 
* For language-only models a variant of GPT-2 is used. 
* For visually grounded models, the two main families of models evaluated are CLIP and GIT, however Flamingo is also evaluated in one experiment. 
* Models are also evaluated under varying sizes of text input windows, i.e. at the single-word level, at small contexts around target words, as well as at the full sentence level.

To assess word learning capability , a variety of evaluations are used: 
* Word similarity -- how model similarity between words correlates with human similarity judgements. 
* Lexical relation prediction -- training linear probes over model features to predict lexical relations such as synonymy or antonymy. 
* Semantic feature prediction -- linear regression over model features to predict strengths of different features of words. 
*  Part of speech prediction -- predicting part of speech tags from SVMs trained on top of model features. 
* Context-based word-understanding benchmark -- a new benchmark presented with the paper, in which models are tasked with ascribing higher probability to correct contexts over perturbed distractors. 
* Brain-response prediction -- linear regression over model features to predict brain response features to input text. 

The paper finds that for a majority of experimental conditions, the multimodal models are generally either worse or comparable to the language-only model. Other findings include: 
* A more fine-grained analysis in human ranking correlation controlled by word features (e.g. prevalence) finds that multimodal models perform better than language models on concrete words, which appears intuitive. 
* Incorporating greater amounts of language context negatively impacts some multimodal model performance. 
* Fine-tuning visual encoders improves performance. 
* Models trained with smaller amounts of data benefit more from multimodality.

### Strengths
* The paper is compellingly motivated, and the research direction of understanding the word-learning capabilities of multimodal models is an important area that may help inform the development of future models. 

* The paper presents an extensive set of experiments across a wide variety of experimental conditions for analyzing word-learning capabilities of multimodal models. 

* The paper does a good job of summarizing its findings/conclusions gleaned from the experimental results, I found the summaries in the subsection titles and conclusions at the end of each subsection helpful in digesting the results.

### Weaknesses
* I believe this is a minor weakness, but I was left wondering what experimental results would be like across a wider variety of model families /variants. I was surprised to see the relatively lower performance of Flamingo, and wonder how models such as InstructBlip, LLaVA, and CM3 would perform.

### Questions
* For experiments evaluating similarities I'm wondering if any metrics other than the cosine similarity was used? I ask because as I understand it CLIP was trained explicitly to maximize the cosine similarity between matching image/text pairs, so it seems intuitive to me that its similarity judgement capability would be well evaluated under that metric. However, I'm not sure if this is necessarily true of other models, and I wonder if the observed trends might be any different under other metrics such as L2.

### Soundness
2 fair

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
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper investigates the problem of visually grounded word learning, and arrives at the main conclusion that visual grounding mainly help acquire word meanings in low-data regimes. The performance of word acquisition are measured by calculating the similarity between model prediction and human annotations. Other side findings are also presented in the paper.

### Strengths
- Comprehensive evaluation on a wide range of tasks.
- While some figures are hard to read, the paper is generally well-written.

### Weaknesses
- Motivation of the work: the main metric is the similarity between model prediction and human judgments, but there are at least two steps that may result in information loss, and I'm not sure how reliable the conclusions are with the presented approaches:
    1. The authors used the cosine similarity between word representations to measure how the models acquire words, but other useful information that affects model preference may be encoded in the deeper architectures.
    2. There may be disagreement between human annotators. For example, Brysbaert et al. (2014) have the word *can* labelled highly concrete, which can cause quite high disagreement among people.
- Plausibility in terms of data exposure: while [*these (text-only) models are profoundly implausible as models of human cognitive development*] (Page 1), isn't the finetuning CLIP approach similarly implausible? Humans do not pre-train their visual and textual understanding systems on large parallel data; instead, human acquisition of words arguably happens in an incremental way.
- Arbitrary definition of *human-likeliness* (Appendix A1.2). The evaluation metrics, while reflecting the model acquisition of word meanings to some extent and from certain perspectives, do not necessarily reflect the model's ability to learn words. 
- The line plots with multiple lines (Figs. 1B, 2B) are largely imperceptible. It would be better to use different line styles.

### Questions
- For CLIP-based settings, did you use a pretrained CLIP model or use the model architecture/objective with random initialization to train from scratch? If the former, isn't it exposed to many more image-caption pairs than your training data? If the latter, I'd be surprised that 4.3K pairs can lead to a decent performance and would meanwhile suggest the authors rename their models---CLIP is usually used to refer to the pretrained CLIP model.
- Why did you specifically pick CLIP, Flamingo and GIT as the model? There are several models, such as [Kiros et al. (2014)](https://arxiv.org/abs/1411.2539) and its variants, working on learning visually grounded word and sentence representations. In terms of performance on image-caption retrieval or generation, they might not be competitive with recent work, but the task of this paper is to investigate the acquisition of word meanings, and there's no reason to stick to the recent popular models.
- (minor) In the first sentence, you probably wish to use *NLM* instead of *LLM* as the abbreviation of *neural language models*?

--------------------updates after response--------------------------------------------------------

My apologies for not realizing the fact that the authors cannot see reviewers' comments after the rebuttal period, so I'll just post it here. I can see the rationales of the authors. All of them are, to some extent but not completely, convincing to me. The additional experiments are interesting, but I'm not sure if the authors have controlled carefully to ensure the fair comparison (sorry I should've realized and brought this up earlier) between visual + word and visual + context: if both of them are trained for the same number of epochs, visual + word will see #(words) - #(sentences) times more images than visual + context, while seeing the same amount of text.
I also have similar feelings on the child language acquisition claims---yes, the visual + word learning protocol is a reasonable hypothesis of human language acquisition, and this paper presents some evidence to support the hypothesis, but there is still something arguable in terms of experiment settings, and how strong the evidence is. 

I've raised my rating to 5 to show my appreciation of the authors' response; however, I still think this paper needs some substantial work before publication.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair
