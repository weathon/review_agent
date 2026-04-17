# ECHO: Where Multilingual Sentence Embeddings Speak the Same Language

- Decision: Reject
- Scores: 4, 6, 2

## Abstract
Cross-lingual sentence encoders create unified embedding representations of sentences across languages. However, achieving both strong downstream performance and cross-lingual alignment remains a fundamental challenge. Early models relied on contrastive learning, yet were unable to leverage hard negatives to unlock the full benefits of the contrastive paradigm. These contrastive approaches were surpassed by non-contrastive approaches leveraging token-level decoders. This is in contrast with recent generic embedding models that achieve strong results by combining contrastive objectives, large language models (LLMs) initialization, and hard negatives usage.
We introduce ECHO, a novel cross-lingual sentence encoder that bridges this gap by integrating pretrained LLMs in an Encoder-Decoder architecture with contrastive training and hard negatives. Our bottleneck Encoder-Decoder design forces the model to capture essential semantic information in a shared vector space while preserving fine-grained nuances. ECHO achieves half of the error rate of the previous state-of-the-art encoders in cross-lingual similarity search across 200 languages, while showcasing unprecedented cross-lingual transfer on downstream tasks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces a method for mapping text from any language to a language-agnostic embedding that captures the semantics of the sentence. This is done in 3 training stages:
(1) training a seq2seq translation model
(2) tuning the embeddings obtained from stage 1 by minimizing a contrastive loss plus a translation CE loss
(3) repeating stage 2, but adding hard negative examples to the mix.
The method achieves SoTA results on a variety of evluation settings.

### Strengths
- The paper achieves strong results: SoTA in several evals
- Well positioned in prior literature
- Seems useful in practice

### Weaknesses
Although I believe the proposed method is practically useful, I'm not convinced the contribution is major enough for ICLR. It's mostly a combination of previous methods, and I didn't see a key innovation beyond that.
I certainly don't want to disparage the work -- I believe there is much value in sharing learnings from method engineering, and the paper achieves very strong results. I applaud the authors for that. But I don't think ICLR is the ideal place to publish this kind of result.

It seems that only a 1B model was trained. Does performance scale with model size? Or has performance saturated already before reaching 1B parameters? In that case, what's the minimum model size that suffices to achieve satisfactory performance?

Some suggestions for improving the paper:
- I found the paper a bit hard to follow in the beginning, as concepts such as "contrastive loss", "hard negatives", "decoder signals" were used without being introduced intuitively. I think the report could be made more appealing by explaining/defining these concepts early on.
- See my questions below. Answering those could help make the manuscript more clear.

### Questions
- Line 186: "target sentences incorporate task specification, output language information, and data provenance": was the loss for this part of the target sentence masked? It seems that this would be the right thing to do (or how else should the model know which language it will be asked to decode into?), but I didn't find any statement about this.
- Line 145 says, "We focus on X-to-English directions", but then line 182 says "with more than 5 thousand translation directions". Is this correct? If the 200 languages are considered, wouldn't there be 200 X-to-English directions, rather than 5000?
- Line 255: How many negative examples were used positive example?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces ECHO, a novel cross-lingual sentence encoder that repurposes a pretrained Large Language Model  into a bottleneck encoder-decoder architecture. ECHO is trained in three stages: 
(1) sequence-to-sequence pre-training on translation tasks, 
(2) contrastive fine-tuning with in-batch negatives, 
(3) continued contrastive fine-tuning with synthetically generated hard negatives. 
The model is evaluated on multilingual alignment, downstream classification and pair classification tasks, and cross-lingual transfer. 
ECHO achieves state-of-the-art results in multilingual alignment across 200 languages and demonstrates strong cross-lingual transfer capabilities.

### Strengths
1. Give credit to the author to conduct comprehensive evaluation, covering 200 languages and multiple tasks, including multilingual alignment, downstream classification, pair classification, and cross-lingual transfer. The results show consistent improvements over strong baselines (SONAR, MEXMA, LaBSE, mE5large).
2. I like the integration of code and math: The inclusion of code and math data, along with natural language, is innovative. The syntax-aware segmentation of code and the generation of natural language descriptions for code and math expressions add to the modality-agnostic nature of the embeddings.

### Weaknesses
1. I am a little confused about hard negative sampling for Code/Math, the method for code and math ("mine the top 5 negatives over a pool of 200k candidates") is less clear and straigthforward. It would be helpful to specify what model is used for this mining and what the nature of these "hard negatives" are.
2. The paper could be benefited by more error analysis. Providing examples where ECHO fails (e.g., low-resource languages, hard negatives, or code/math snippets) would help understand limitations and guide future work.

### Questions
see weakness part

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes ECHO, a three-stage framework that finetune a encoder-decoder model intitilaized with LLMs for multilingual sentence embeddings. Using parallel corpus, the proposed ECHO finetune the model with translation objective (stage 1), contrastive objective in the encoder output representation space (stage 2) and contrastive learning with hard negative (stage 3). Experiments on cross-lingual retrieval and classfication show emprical improvements compared to the baseline.

### Strengths
1. Multilingual sentence embedding is an important tasks that have many applications. It aligns with the scope of ICLR too.
2. Emprical results show improvements compared to previous embedders.
3. Non-natural languages including math and code are included in the experiments. This is useful to study multilingual embedding in a broader scope.

### Weaknesses
1. The English writing needs improvement. A revising by native speaker or LLMs can be helpful.
2. The organization of this paper is not clear. For example, the motivation and justification for using encoder-decoder framework is not clearly stated. For example, "...they largely overlook cross-lingual transfer..." is not a informative argument to introduce the limitation of previous work, especially the proposed method has significant overlap with baselines such as SONAR [1]. A more structual story-tell can improve this.
3. A main concern is that the novelty of this work seems to be limited. The objectives of machine translation along with contrastive learning has been studied in previous works such as [1] and [2] and etc. Sorely utilzing LLMs as backbone lacks of technical contribution. Hard negative for contrastive learning is also insufficient to be a contribution.
4. In addition to limitation 3, the baselines and proposed ECHO are built on different backbones, and LLaMA3 is more powerful initillay hench the comparision is not convincing enough. More details on the training settings including training data use to show that the baselines and ECHO are comparable will be helpful.
5. As for the experiments settings, only LLaMA3 is used so the emprical results are not model-agnositic which limits the contribution. In addition, LLaMA3 is a decoder LLM and why utilizing it for both encoder and decoder framework is not justified.
6. The ablation study is weak. The three stage training is the core design of ECHO, however, training with only state 2 or stage 3 are not evaluated to show how those stages affect the behavior of the framework. Table 9 in appedix seems not introduce the detailed experimental setting and also missing settings metioned above but just incremental experiments from stage 1 to 1+2 and 1+2+3.


Minor comments:
1. line 30: citations are broken
2. What is the implication of "speak the same language" in the title?

[1] SONAR: sentence-level multimodal and language-agnostic representations.
[2] Enhancing Cross-lingual Sentence Embedding for Low-resource Languages with Word Alignment, NAACL 2024 findings.

### Questions
1. The decoder is designed for probing the sentence embedding, which is useful. However, the focus of this paper is multilingual embedding and the decoder design is not nessasary, for example, we could just fine-tune a encoder-only embedder and train a seperate probing decoder for this purpose. I would appreciate more explanation on this issue.

2. What is the essential difference between ECHO and SONAR? I beleive the answer to this question will highlight the contribution of this work better.

### Soundness
2

### Presentation
1

### Contribution
2
