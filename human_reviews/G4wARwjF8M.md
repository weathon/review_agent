# SLMRec: Distilling Large Language Models into Small for Sequential Recommendation

- Avg Score: 6.25
- Decision: Accept (Poster)
- Scores: 6, 6, 5, 8

## Abstract
Sequential Recommendation (SR) task involves predicting the next item a user is likely to interact with, given their past interactions. 
The SR models examine the sequence of a user's actions to discern more complex behavioral patterns and temporal dynamics. 
Recent research demonstrates the great impact of LLMs on sequential recommendation systems, either viewing sequential recommendation as language modeling or serving as the backbone for user representation. Although these methods deliver outstanding performance, there is scant evidence of the necessity of a large language model and how large the language model is needed, especially in the sequential recommendation scene. Meanwhile, due to the huge size of LLMs, it is inefficient and impractical to apply a LLM-based model in real-world platforms that often need to process billions of traffic logs daily. In this paper, we explore the influence of LLMs' depth by conducting extensive experiments on large-scale industry datasets. Surprisingly, our motivational experiments reveal that most intermediate layers of LLMs are redundant, indicating that pruning the remaining layers can still maintain strong performance.
Motivated by this insight, we empower small language models for SR, namely SLMRec, which adopt a simple yet effective knowledge distillation method. Moreover, SLMRec is orthogonal to other post-training efficiency techniques, such as quantization and pruning, so that they can be leveraged in combination. Comprehensive experimental results illustrate that the proposed SLMRec model attains the best performance using only 13\% of the parameters found in LLM-based recommendation models while simultaneously achieving up to 6.6x and 8.0x speedups in training and inference time costs, respectively. Besides, we provide a theoretical justification for why small language models can perform comparably to large language models in SR.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper works on improving the efficiency of LLM as a recommender scheme by arguing that the intermediate layer of LLM is redundant. Specifically, the paper suggests a vallina knowledge distillation approach to achieve near LLM-based Recommendations in an efficient way.

### Strengths
1.	The paper addresses an important problem regarding the efficiency of LLM as a recommender scheme and provides a reasonable knowledge distillation approach to solving the problem.
2.	The paper is generally well-written.

### Weaknesses
1.	The knowledge distillation approach is widely used in LLM-related work [1], which raises concerns about its novelty when adopting it in the recommendation.
2.	Although the paper claims that they provide both experimental and theoretical justification for adopting KD to LLMRec. It does not look strong enough. 
    1. In the experimental part, as only 8-layer and 24-layer are discussed, could the author provide a more detailed analysis of the performance differences between the 8-layer, 24-layer, and 32-layer models? 
    2. In terms of the theoretical part, the author suggested that a small model can also learn user representation that is similar to the larger model, while it does not align with the recent work on the emergent ability of LLM that a small model cannot achieve a similar result of the large language model.
Could the author discuss how your findings relate to existing work on emergent abilities in LLMs, and explain why user representation learning may differ from other tasks where emergent abilities have been observed? Also, some supportive experiment is needed to justify this claim.


[1] Xu X, Li M, Tao C, Shen T, Cheng R, Li J, Xu C, Tao D, Zhou T. A survey on knowledge distillation of large language models. arXiv preprint arXiv:2402.13116. 2024 Feb 20.

### Questions
Could the author justify the claim that “an 8-layer E4SRec can obtain nearly as informative user representation as a 24-layer E4SRec” by answering the following question in detail?
1. Provide a more detailed analysis of the performance differences between the 8-layer, 24-layer, and 32-layer models.
2. Explain the apparent performance surge at 32 layers and discuss its implications for your claims.

### Soundness
2

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
This work investigates the use of small language models (SLMs) in the realm of sequential recommendation systems, proposing a model called SLMRec. The authors identify that traditional sequential recommendation (TSR) methods have reached a performance plateau and that existing large language model (LLM) approaches often require significant computational resources due to their large size. The proposed SLMRec aims to maintain competitive performance with a much smaller parameter footprint (less than 1 billion parameters) compared to the over 7 billion parameters typical of LLMs used in recommendations. The authors utilize knowledge distillation techniques to align the representation knowledge between a larger teacher model and the smaller student model, achieving an up to 8x acceleration in both training and inference times.

### Strengths
S1. SLMRec demonstrates significant efficiency improvements, achieving competitive performance while using substantially fewer parameters than larger models.
S2. The model's design allows it to operate within the resource constraints typical of real-world applications, making it more applicable for widespread use in industry.
S3. The paper's approach to knowledge distillation, focusing on feature alignment rather than just output prediction, is a noteworthy contribution that can lead to better representation learning in smaller models.
S4. The authors conduct comprehensive experiments on large-scale datasets, providing empirical evidence of the model's effectiveness across multiple metrics and domains.

### Weaknesses
W1. The implementation of knowledge distillation can introduce complexity in the training process, which may not be straightforward for all practitioners to apply.

### Questions
Please refer to the Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
This work discusses the impact of LLMs in SR, but questions the necessity of their size. The inefficiency of applying LLMs to real-world platforms due to their size is noted. The experiments show that most intermediate layers of LLMs are redundant, leading to the development of a small language model for SR using knowledge distillation. SLMRec achieves strong performance with significantly fewer parameters than LLM-based models, along with notable speedups in training and inference times. It also provides theoretical reasoning for the effectiveness of small language models in SR.

### Strengths
1. The authors explored the relationship between the parameter count of pre-trained LLMs and recommendation performance under the E-LLMRec paradigm, discovering a balance point between performance and parameter count that enhances the practicality of deployment in real-world scenarios.

2. The authors conducted thorough theoretical justifications to demonstrate why small language models can possess learning capabilities for user representations that are comparable to those of large language models.

3. The paper is well-written, and the figures clearly illustrate the overall workflow of SLMRec.

### Weaknesses
1. The motivation is not sufficiently clear. The authors claim that LLMs with fewer layers can achieve recommendation performance comparable to those with more layers, and thus they opt to use large LLMs to distill a small LLM. However, it is unclear why small LLMs cannot be fine-tuned directly. The authors also do not present results in the ablation study showing the fine-tuning of a 4-layer small LLM with CE loss without any distillation operation.

2. Generally speaking, distillation is performed to transfer knowledge from one model to another. However, small LLMs possess certain language capabilities on their own. During the distillation process, how can the teacher model be guided to transfer recommendation-related information to the student model? If only general knowledge is distilled, the parameter limitations of small LLMs imply that they may not be able to learn additional knowledge effectively.

3. Although knowledge distillation can help small LLMs acquire certain recommendation capabilities and reduce deployment costs, simply reducing the number of layers from 8 to 4, while being more efficient than Open-P5 and E4SRec, still renders it too inefficient compared to traditional sequence recommendation methods. This creates a significant gap in terms of practical deployment, weakening the motivation claimed by the authors.

4. The experimental section lacks some of the latest traditional sequence recommendation methods, such as self-supervised approaches (e.g., DuoRec [1], MAERec [2]) and pre-trained methods (e.g., S3-Rec [3]).

5. In Table 2, the experimental results on the Cloth dataset show that E4SRec outperforms SLMRec in HR@1, NDCG@5, and MRR. However, it is marked in purple as having "the second best performance," which seems inconsistent.

[1] Qiu, Ruihong, et al. "Contrastive learning for representation degeneration problem in sequential recommendation." Proceedings of the fifteenth ACM international conference on web search and data mining. 2022.

[2] Ye, Yaowen, Lianghao Xia, and Chao Huang. "Graph masked autoencoder for sequential recommendation." Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval. 2023.

[3] Zhou, Kun, et al. "S3-rec: Self-supervised learning for sequential recommendation with mutual information maximization." Proceedings of the 29th ACM international conference on information & knowledge management. 2020.

### Questions
1. What are the results of fine-tuning small LLMs using only CE loss? (i.e., without employing knowledge distillation methods). Additionally, what would the results be if only both CE loss and MS loss were used? (It appears that MS loss is unrelated to knowledge distillation.)

2. Could the authors include comparisons with other traditional sequence recommendation methods in the Efficiency Comparison section to provide a more comprehensive evaluation of their efficiency?

3. Could the authors provide experimental results for the self-supervised and pre-trained sequence recommendation methods mentioned in the fourth point of the Weakness section?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces SLMREC, a method that enhances small language models for sequential recommendation tasks using knowledge distillation. The authors conduct extensive experiments on large-scale industry datasets, revealing that most intermediate layers in large language models are redundant. SLMREC achieves competitive performance with only 13% of the parameters of LLMs, making it highly efficient and practical for real-world applications. The model is orthogonal to other post-training efficiency techniques, such as quantization and pruning, allowing for combined use to further improve efficiency. Comprehensive experimental results demonstrate that SLMREC outperforms existing state-of-the-art methods in various evaluation metrics.

### Strengths
- The paper reveals that most intermediate layers in large language models are redundant, providing valuable insights into the architecture of these models. 

- The paper introduces a novel method, SLMREC, which effectively enhances small language models for sequential recommendation tasks using knowledge distillation. SLMREC achieves competitive performance using only 13% of the parameters of large language models, making it highly efficient and practical for real-world applications.

- The proposed method is orthogonal to other post-training efficiency techniques like quantization and pruning, allowing for combined use to further enhance efficiency.

### Weaknesses
- The paper does not describe the specific process of knowledge distillation in detail, and lacks a detailed explanation of the key steps in the distillation process (such as temperature adjustment, loss function selection, etc.)

### Questions
- The article mentions the use of multiple supervisory signals. What are these signals? How can we combine them in the training process?

### Soundness
4

### Presentation
4

### Contribution
4
