# Bridging Information Asymmetry in Text-video Retrieval: A Data-centric Approach

- Decision: Accept (Poster)
- Scores: 8, 6, 5

## Abstract
As online video content rapidly grows, the task of text-video retrieval (TVR) becomes increasingly important. A key challenge in TVR is the information asymmetry between video and text: videos are inherently richer in information, while their textual descriptions often capture only fragments of this complexity. This paper introduces a novel, data-centric framework to bridge this gap by enriching textual representations to better match the richness of video content. During training, videos are segmented into event-level clips and captioned to ensure comprehensive coverage. During retrieval, a large language model (LLM) generates semantically diverse queries to capture a broader range of possible matches. To enhance retrieval efficiency, we propose a query selection mechanism that identifies the most relevant and diverse queries, reducing computational cost while improving accuracy. Our method achieves state-of-the-art results across multiple benchmarks, demonstrating the power of data-centric approaches in addressing information asymmetry in TVR. This work paves the way for new research focused on leveraging data to improve cross-modal retrieval.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper investigates the problem of information asymmetry in text video retrieval from a data-centric perspective. Existing works usually use model-centric approaches of carefully designing advanced text-video interaction modules, e.g., text-conditioned video representations, stochastic embedding, etc. This paper proposes a unified text enrichment framework to enhance the textual representation during both training and testing phase. The results consistently show promising improvements over existing methods. Interestingly, the concept of “oracle query” clearly demonstrate the potential of the query generation and selection, possibly opening up a new venue in this domain.

### Strengths
- The paper investigates an under-explored area in the field of text-video retrieval, a data-centric approach to improving the performance by identifying pitfalls in the current training dataset captions and test retrieval process.
- The full method demonstrate remarkable performance in the text-video retrieval task across several benchmarks.
- The experiments with the oracle queries show a huge gap in the current methods, which is novel and interesting. It opens up future works to consider such data-centric approaches for improving performance.
- The paper is well written and easy to follow.

### Weaknesses
- There are some existing works on augmenting the data, e.g., [1], what is the main difference between this work and existing works?
- The oracle query experiment shows huge performance gap. Although the result is inspiring, it is better to provide more explanations and/or investigations about this phenomenon. What contribute to the final performance?
- In the ablation study, the performance gain seems mainly come from the Retrieval Phase Enrichment, while the gain Training Phase Enrichment seems to be marginal.
- The method will inevitably increase the computational cost, which should better be investigated.
- It is mentioned that majority voting over the enriched queries yields the best performance. However, it is not clear how this is implemented. I would suggest include such details, as least in the appendix.

[1] HAVTR: Improving Video-Text Retrieval Through Augmentation Using Large Foundation Models. ECCV 2024.

### Questions
- Will the author open-source the enriched dataset and code?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
Due to the information asymmetry between video and text in text-video retrieval (TVR), this paper introduces a data-centric framework aimed at enriching textual representations to better align with the rich information contained in video content. However, despite efforts to leverage vision-language models (VLMs) and large language models (LLMs), the method lacks significant innovation.

### Strengths
S1.  This paper shows a method of utilizing VLMs and LLMs to enrich textual representations in training and inference stage.
S2. The data-centric method proposed by this paper maybe useful in practice, as this paper provides evidence of its effectiveness.
S3.  The writing is well-structured and clear, making the paper easy to follow.

### Weaknesses
W1. The method does not present substantial innovation. The improvement in model performance appears to be primarily attributed to the inherent capabilities of VLMs and LLMs in visual and textual understanding.
W2. While the paper introduces a query selection mechanism and designs a Farthest Query Sampling (FQS) algorithm, it would benefit from exploring additional query selection algorithms to further validate FQS’s effectiveness.
W3. The comparative experiments with the latest methods are somewhat lacking. Most of the related work cited in the experiments on state-of-the-art (SOTA) methods is limited to publications from 2022.

### Questions
Q1.  Could you include experiments comparing the model’s performance with more recent methods beyond those published in 2022? This would enhance the context of the model's strengths and limitations relative to current advancements.
Q2. How might the proposed framework perform in real-world, large-scale text-video retrieval systems？
Q3.  Could you clarify how this work distinguishes itself from existing methods beyond leveraging VLMs and LLMs? Are there any unique components or methodological innovations that specifically contribute to the improvements in performance?

Below are a few suggestions which could help the authors to refine a better version of this work.

1. This paper addressed the issue of information asymmetry in text-video retrieval from a data-centric perspective, therefore leveraging VLMs for event-level captioning and LLMs for query diversification. While this approach primarily combined existing methods to enhance textual representations, emphasizing the uniqueness of the model structure would better highlight its innovative contributions.
 
2. As the paper mentioned, relevant and diverse queries are expected to be retrieved. After initially constraining the relevance of the generated queries, FQS is applied to iteratively select queries by maximizing the minimum distance between them.
It may be worth considering adding relevance constraints to FQS, such as setting a minimum similarity threshold between query embeddings. Similarity metrics, such as dot product or cosine similarity, could be used for this purpose. Incorporating both distance and similarity into FQS might enhance its performance.
Additionally, FQS could be also compared with methods that minimize the minimum distance between selected queries to further demonstrate its ability to select diverse queries effectively.
Overall, designing alternative algorithms as comparisons would provide a stronger demonstration of FQS’s effectiveness.
 
3. In this paper, Table 1 presents the performance of the latest models. However, Tables 2 and 3 do not include these methods, such as Cap4Video. Including these comparisons would provide a more comprehensive evaluation.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
This work aims to solve the problem of information asymmetry between text descriptions and videos in text-video retrieval,i.e., text often contains only part of the video content. The author uses VLM to generate more captions from videos during training; and uses LLM to expand queries and add more content during inference.

### Strengths
1. The proposed problem of information asymmetry in the text-video retrieval is important and needs to be solved in the development of this task.
2. The paper is well written and easy to follow.

### Weaknesses
1. The proposed method uses VLM and LLM to generate details to the original caption. The captions and queries generated in this way are unreliable and contain a lot of information that is not in the video. In the inference phase, without the addition of oracle information, the newly generated captions and querys are hallucinated by the large language model. The authors need to provide more evidence to support their idea.

2. The work involves utilizing large visual language models and large language models. The required training cost and inference computation are much higher than compared works. It`s not fair to compare directly in experiments. The authors need to prove the performance gain is not from the accumulation of more computation.

3. Line 238 mentions that this work used an image captioner. The designed method does not consider the temporal cues in the video neither. The proposed method is essentially designed for an image-text retrieval task rather than a video-text retrieval task.

4. In line 278, the fact that oracle query can improve model performance does not mean that supplementing the query is useful. Introducing the ground truth query can greatly enhance model performance, regardless of whether the query is supplemented or not.

### Questions
1. Line 290 proposed to ensure the diversity in the supplemented queries. Why are the supplemented queries should be expressed in as many ways?

2. To verify the effectiveness of the proposed method, the authors should provide some examples of the generated captions during training and the expanded queries during inference. It can help to understand the contribution of the work.

3. The approach proposed in Figure 3 requires more illustration on design idea and details; the current version is a bit confusing.

4. The authors need to explain the training efficiency and inference efficiency of the proposed method.

Other questions seen in the Weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2
