# Triple-S: A Sticker Semantic Similarity Benchmark with General Sticker Encoder

- Decision: Reject
- Scores: 4, 2, 2

## Abstract
Stickers have become a popular form of visual communication, yet understanding their semantic relationships remains challenging due to their highly diverse and symbolic content. In this work, we formally define the Sticker Semantic Similarity task and introduce Triple-S, the first benchmark for this task, consisting of 905 human-annotated positive and negative sticker pairs. Through extensive evaluation, we show that existing pretrained vision and multimodal models struggle to capture nuanced sticker semantics. To address this, we propose the General Sticker Encoder (GSE), a lightweight and versatile model that learns robust sticker embeddings using both Triple-S and additional datasets. GSE achieves superior performance on unseen stickers, and demonstrates strong results on downstream tasks such as emotion classification and sticker-to-sticker retrieval. By releasing both Triple-S and GSE, we provide standardized evaluation tools and robust embeddings, enabling future research in sticker understanding, retrieval, and multimodal content generation. The Triple-S benchmark and GSE have been publicly released and are available here \footnote{https://anonymous.4open.science/r/triple-s-6E65/}

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper introduces the task of sticker semantic similarity examining the challenge of encoding the diverse and symbolic semantics of digital stickers. The main contribution is Triple-S, the first human-annotated benchmark dataset designed specifically for sticker semantic similarity evaluation, with 905 positive and negative sticker pairs. The authors demonstrate that recent vision and general multimodal models mainly attempt to capture the nuanced meaning in stickers. For examining this deficiency, the authors proposes the general sticker encoder for improving the semantic representation and fusion capabilities for this specific visual-symbolic modality.

### Strengths
S1) The definition of sticker semantic similarity addresses notable terms of multimodal communication research, moving beyond literal image understanding to symbolic, expressive content.

S2)  The proposed benchmark can be one of the contributions for future research in this unique domain.

S3)  The work effectively validates the limitations of existing foundation models, demonstrating that their pre-training on natural images or general vision-language data does not translate well to stylized, cartoon-like, and symbolic sticker semantics.

### Weaknesses
W1) A size of 905 annotated pairs is small and limited, raising notable concerns about the potential for overfitting and the generalization capability.

W2) The novelty in the multimodal fusion architecture of the General Sticker Encoder needs clearer articulation. What is new? There is no specific difference between the findings of the current study and implications of prior research.

W2) Sticker semantics are heavily context- and culture-dependent.

### Questions
Q1) Could the authors elaborate on the specific architectural choices in the general sticker encoder for allowing to capture symbolic semantics?

Q2) What is the performance drop when depending on  the visual encoder features versus the fused representation?

Q3) Could the authors test any other methods (e.g. self-supervised learning) for addressing this task?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This work focuses on the sticker semantic similarity task. The authors manually created a benchmark, Triple-S, which consists of 905 sample pairs, and trained a lightweight General Sticker Encoder (GSE) using the training set from the benchmark. Experimental results on both Triple-S and other test datasets show that GSE outperforms CLIP, ViT, and DINOv2.

### Strengths
1. The paper constructs a high-quality sticker similarity benchmark through manual annotation, which is meaningful for the relevant community.

2. The experimental design is comprehensive, demonstrating the effectiveness of the collected data through ablation studies and showcasing the performance advantages of GSE through experiments on multiple test datasets.

3. The overall structure of the paper is clear and easy to understand.

### Weaknesses
1. The scale of the constructed benchmark is relatively small, with only 140 sample pairs used for testing. I am concerned whether such a small-scale test set is sufficient to reflect the model's performance.

2. The methods compared are limited. Most experiments only compare CLIP, DINOv2, and ViT. More advanced CLIP-based methods (e.g., EVA-CLIP, SigLIP) or MLLM-based methods (e.g., UniME-V2, VLM2Vec-V2) need to be compared.

3. The technical contribution of the GSE design is limited, as it only fine-tunes CLIP using the constructed dataset.

### Questions
1. The writing logic could be further improved:
    a. In the second paragraph, the phrase "In addition to existing baselines, providing a feasible evaluation method xxx" is unclear. The terms "baselines" and "evaluation method" are not well-defined, which may cause confusion.
    b. The second paragraph mentions that manually annotated datasets suffer from issues like subjectivity and labor-intensiveness, but the dataset constructed by the authors is also manually annotated, so the issues of subjectivity and labor-intensiveness have not been addressed.

2. What are the differences between GSE and StickerCLIP in terms of methodology?

3. Why is the performance of GSE not reported in Table 3?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper's main goal is to address the challenge of understanding nuanced sticker semantics. To do this, the authors first introduce Triple-S, a new, human-annotated benchmark of 905 sticker pairs, and demonstrate that standard encoders (e.g., CLIP and ViT) and a VLM (ChatGLM-4V-Flash) struggle to do well on it. They then propose the General Sticker Encoder (GSE), a CLIP image encoder fine-tuned on their Triple-S data combined with the MultiChat dataset. The paper evaluates GSE on a range of downstream tasks, showing it outperforms baselines on unseen sticker similarity and retrieval datasets (i.e., WXChallenge, SER30K) and achieves new state-of-the-art performance in emotion classification when its embeddings are integrated into an existing multimodal model.

### Strengths
- The paper formally provides a carefully curated benchmark (Triple-S) designed to test the nuanced understanding of stickers, and benchmarks common vision encoders (CLIP, ViT, DINOv2), effectively demonstrating their limitations in capturing nuanced sticker semantics and justifying the need for a better encoder.

- The proposed General Sticker Encoder (GSE) demonstrates effective transfer learning and outperforms other popular vision encoders. When integrated into a multimodal LLM, it achieves new state-of-the-art results on downstream emotion classification tasks, validating that the embeddings are robust and generalizable

### Weaknesses
- The paper only benchmarks one VLM (ChatGLM-4V-Flash) in Section 5. Given the test set's tiny size (140 pairs), there should be results for stronger commercial or larger open-source VLMs. The paper fails to reveal the state of the art on this task.

- The ablation study (Table 7) shows that training on Triple-S or MultiChat alone gives marginal benefit, but training on both provides a huge boost. There is no analysis or reasoning on why the small-scale training set of Triple-S can bring that much improvement when combined with MultiChat. Furthermore, there should also be discussions to rule out any potential data leakage between its Triple-S training set and the WXChallenge test set.

### Questions
Please see the weaknesses section.

### Soundness
2

### Presentation
3

### Contribution
2
