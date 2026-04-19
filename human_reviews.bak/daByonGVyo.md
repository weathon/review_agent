# Compression via Pre-trained Transformers: A Study on Byte-Level Multimodal Data

- Decision: Reject
- Scores: 5, 6, 6, 3

## Abstract
Foundation models have recently been shown to be strong data compressors. However, when accounting for their excessive parameter count, their compression ratios are actually inferior to standard compression algorithms. Moreover, naively reducing the number of parameters may not necessarily help as it leads to worse predictions and thus weaker compression. In this paper, we conduct a large-scale empirical study to investigate whether there is a sweet spot where competitive compression ratios with pre-trained vanilla transformers are possible. To this end, we train families of models on 165GB of raw byte sequences of either text, image, or audio data (and all possible combinations of the three) and then compress 1GB of out-of-distribution (OOD) data from each modality. We find that relatively small models (i.e., millions of parameters) can outperform standard general-purpose compression algorithms (gzip, LZMA2) and even domain-specific compressors (PNG, JPEG~2000, FLAC) &mdash; even when factoring in parameter count. We achieve, e.g., the lowest compression ratio of 0.49 on OOD audio data (vs. 0.54 for FLAC). To study the impact of model- and dataset scale, we conduct extensive ablations and hyperparameter sweeps, and we investigate the effect of unimodal versus multimodal training. We find that even small models can be trained to perform well on multiple modalities, but, in contrast to previously reported results with large-scale foundation models, transfer to unseen modalities is generally weak.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The paper outline a study of transformers / next-token-prediction for lossless compression of multiple modalities.
It focuses on relatively small models and show good results, across a variety of tasks and hyper-parameters.
The resulting model if able to compete and beat general and domain specific algorithms (FLAC for instance) on the different datasets and modalities.
One limitation with the model/trainset size is the lack of cross-domain compression abilities of these models, as remarked by the authors, and as somewhat expected.

### Strengths
I take this paper at face value. A lot of the conclusions of this paper are somewhat expected. Ability of next-token-prediction transformers to compete (and even beat) classic lossless compression algorithms, but also lack of abilities to move across domains, due to the limit in the parameters count / train-set size.

On the other hands this is also the strength of the paper - a clear claim and a credible set of results to substantiate such claims. Also results that can be seen as impactful.

### Weaknesses
I don't see any particular weakness, beyond the simplicity of this paper claims.
I am lent to ask the question about novelty, I'd like to say that perhaps there is no single paper coming to this conclusion, but there are bits and pieces in the literature pointing at exactly this conclusion. That is why I am not surprised by the results obtained in this work.

### Questions
- it maybe good to compare / scale the flops versus the classic algorithm we are comparing against ... it's possible that transformers are order of magnitude more expensive, almost surely, but would still be a good graph to have
- Figure-5, it's unclear: it seems to show compression ratio going up with model size (for audio) but down with context size (for video) ??
  I am unclear why model-size would at anytime make a consistent degradation in compression

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
4

### Summary
The paper proposes a pretrained Transformer architecture for doing compression on Byte level. They use simple vanilla transformer based architecture to carry this out. The findings are great, and have shown an expected results that transfer to unseen modalities is generally weak for cross modal compression on byte level data. The paper claims to have shown to achieve lowest compression ratio of 0.49 on OOD audio data vs other traditional audio compressors e.g. FLAC. The paper is well written, using a simple straightforward approach and its findings can perhaps lay the foundation to build on further complex architectures as well building byte level compressors that utilizes minimum domain specific knowledge.

### Strengths
strengths of paper on originality, quality, clarity, and significance

== Decent paper. 
I wish the authors would have tweaked and added some novelty or some innovations for the compression than vanilla Transformer architecture. It would have made the paper really strong. 

== I liked the idea of incorporating out-of-domain metrics for each of the modalities in addition to the out of modality benchmark

== The paper is well-written and explains all of the ideas proposed clearly. The experiments are well structured and the expectations are laid out at the very outset.

== The paper is reproducible. 

== There is a good amount of insights from the paper that can be used for further studies or lay the foundation for future work and research directions.

== Simplicity. The paper is fairly simple and easy to read.

### Weaknesses
= Straightforward methodology. In brief, the paper compares vanilla transformers on byte level data for audio, text, images without any prior knowledge. What would have happened if we were to tweak the Transformer with something like Perceievers that also operate on Byte level. 

== It would have been great if the authors would have evaluated atleast on one of the benchmarks the comparison with tokenization methods in text, audio and images. e.g. ENCODEC for audio and Byte-Pair Encoding for text

== These problems are quite different -- compression of 1024 or 8192 for audio at byte level is far simpler than compression of byte level data of text. The authors should have shed light on all of these different aspects, why it is and how it affects the compression ratios. 

== The authors should have run experiment on scale of the number of parameters on experiment similar to Figure 3. What happens if I do one of these cross modal experiments on say a GPT-2 small or large level parameter architecture. Does it help ?  Does the gap narrow down. How does scaling affect the results.  It seems that the authors did have plenty of compute available for carrying out all of these experiments. Currently the scaling experiments are carried out on tiny Transformer models with maximum of 20M parameters. 

== The methodology is straightforward. There could have been some novelty that could have been incorporated into the paper.

### Questions
Here are a few questions that would help me better evaluate the paper. The authors need not do any more experiments but rather simply address 


== What would be the comparison with other advanced architectures such as perceivers. Would it help. If yes, why.  If not why? 

== Why was there no comparison or experiment carried out for vanilla byte level architecture vs a tokenized in anyone of the three domains. 

== Would the current results hold if we were to replace the input in all of the modalities by their respective token level representation. Which domains are likely to be improved the most. 

== In figure 3 why are the compression ratios not getting better with size of the data. One should expect the plot to be similar to that of Figure 4.

### Soundness
2

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper conducts an empirical study to investigate the effectiveness of small pre-trained transformers (millions of parameters) as competitive multimodal compressors for text, image, and audio data. By training on a substantial dataset of 165GB per modality, the authors assess the compression performance on out-of-distribution (OOD) data for each modality. The results presented in the paper show that small transformers can outperform general-purpose and domain-specific compression algorithms, with compression ratios competitive even with state-of-the-art online adaptive transformers. Additionally, the paper explores the impact of multimodal training on compression efficacy, revealing that cross-modal transfer remains weak in these smaller models, contrasting with previously reported results for large-scale language models.

### Strengths
1. I find the paper to be thorough in its investigation of transformer-based compression across different modalities, with detailed experiments on model and context sizes, as well as dataset combinations.
2. The authors demonstrate that small, pre-trained transformers can achieve compression rates comparable to both domain-specific and general-purpose algorithms. Additionally, the paper shows that training on multiple modalities does not significantly impact unimodal performance while producing effective domain-general compressors.

### Weaknesses
1. The evaluation on a fixed 1GB test set limits the scalability of the compression methods. Did the authors explore test set sizes beyond 1GB or consider varying test data sizes? What is the rationale behind taking test data size as 1GB?
2. I noticed that the paper shows limited cross-modal transfer, struggling to compress modalities they weren’t trained on (e.g., training on text and audio did not generalize well to image data). Did the authors try any transfer learning approaches to address this?

### Questions
I have addressed all my questions in the weakness section.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper studies the connection between auto-regressive transformers and data compression. The authors presented an empirical study, investigating the usage of transformer-based models for data compression considering multi-modal data. The authors compared the transformer-based compressors with standard compression techniques while considering compressor sizes in comparison ratio calculations. 
The authors show that small-scale transformer models can achieve better compression ratios than general multi-purpose compressors for in-domain modality data, however when considering out-of-domain modality data, standard compressors achieve superior performance.

### Strengths
1. This paper conducts a comprehensive empirical study on compression performance, especially for small transformer models considering multi-modal data. The paper is clearly written and easy to follow.
2. The presented results are interesting (the fact that small pre-trained transformers can achieve better compression ratios than standard general-purpose compressors) and would be of value to the community. 
3. The authors provide scaling analysis on the performance of the models.

### Weaknesses
The main weakness of this study is its novelty / contribution. Considering the work done in [1,2], the contribution of this study is unclear. I understand that the authors additionally evaluated model performance under the multi-modal setup, however I'm afraid this is not enough for an ICLR publication. 


[1] Bellard, Fabrice. NNCP v2: Lossless data compression with transformer. Technical report, Amarisoft, 2021.
[2] Delétang, Grégoire, et al. "Language modeling is compression." arXiv preprint arXiv:2309.10668 (2023).

### Questions
1. Following the weakness mentioned above, I would like to ask the authors to clarify their novelty/contribution to this research work. 
2. How did you represent the multi-modal data? Representing the raw data can result in extremely large sequences. For example, music data is sampled at 48kHz, so only 10 seconds of music is a sequence length of 480k tokens. Similarly, how did you flatten the images?
3. Can the authors provide more details about model training and evaluation? How long did you train the model? On what sequence length? What was the sequence length during the evaluation? Etc. 

I'm willing to change my score in case I missed something.

### Soundness
3

### Presentation
4

### Contribution
2
