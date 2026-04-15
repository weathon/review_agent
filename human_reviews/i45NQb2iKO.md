# MM-EMBED: UNIVERSAL MULTIMODAL RETRIEVAL WITH MULTIMODAL LLMS

- Decision: Accept (Poster)
- Scores: 8, 6, 6, 6

## Abstract
State-of-the-art retrieval models typically address a straightforward search scenario, in which retrieval tasks are fixed (e.g., finding a passage to answer a specific question) and only a single modality is supported for both queries and retrieved results. This paper introduces techniques for advancing information retrieval with multimodal large language models (MLLMs), enabling a broader search scenario, termed universal multimodal retrieval, where multiple modalities and diverse retrieval tasks are accommodated. To this end, we first study fine-tuning an MLLM as a bi-encoder retriever on 10 datasets with 16 retrieval tasks. Our empirical results show that the fine-tuned MLLM retriever is capable of understanding challenging queries, composed of both text and image, but it underperforms compared to a smaller CLIP retriever in cross-modal retrieval tasks due to the modality bias exhibited by MLLMs. To address the issue, we propose modality-aware hard negative mining to mitigate the modality bias exhibited by MLLM retrievers. Second, we propose continuously fine-tuning the universal multimodal retriever to enhance its text retrieval capability while preserving multimodal retrieval capability. As a result, our model, MM-Embed, achieves state-of-the-art performance on the multimodal retrieval benchmark M-BEIR, which spans multiple domains and tasks, while also surpassing the state-of-the-art text retrieval model, NV-Embed-v1, on the MTEB retrieval benchmark. Finally, we explore prompting the off-the-shelf MLLMs as zero-shot rerankers to refine the ranking of the candidates from the multimodal retriever. We find that, through prompt-and-reranking, MLLMs can further improve multimodal retrieval when the user queries (e.g., text-image composed queries) are more complex and challenging to understand. These findings also pave the way for advancing universal multimodal retrieval in the future. We release the model weights at: https://huggingface.co/nvidia/MM-Embed.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes a general retrieval method for multimodal retrieval tasks, named UniEmb, based on multimodal large language models. The paper identifies the phenomenon of modality bias in multimodal retrieval and finds that current retrieval methods based on multimodal large language are lacking in text retrieval performance. To address these issues, the authors propose a continued fine-tuning method based on hard negative samples, as well as a strategy that uses multimodal large language for further re-ranking. The proposed method is comprehensively evaluated on two widely adopted datasets, M-BEIR and MTEB, with results showing that UniEmb improves text retrieval performance while maintaining the effectiveness of multimodal retrieval.

### Strengths
1. The paper identifies an interesting issue of modality bias in multimodal retrieval based on multimodal large language models, and it proposes an effective hard sample mining strategy to address this issue while preserving text retrieval capabilities.
2. The paper introduces a zero-shot re-ranking approach using multimodal large language model, which, to some extent, improves multimodal retrieval accuracy.
3. The evaluation is thorough, using widely adopted M-BEIR and MTEB datasets, and the paper provides detailed parameters for reproducibility, lending credibility to the experimental results.

### Weaknesses
1. The proposed methods for hard sample mining and re-ranking using large multimodal models are straightforward and intuitive, and may lack some novelty.
2. There is no direct comparison with similar baseline models, such as E5-V, on the M-BEIR dataset.

### Questions
1. In Table 1, why is the M^rand column for MTEB Text Retrieval empty? Are these models not evaluable on MTEB? I believe that comparing M^rand with M^hard would more effectively demonstrate the modality bias issue and the effectiveness of the proposed method.
2. In Table 4, why does re-ranking show limited improvement for knowledge-related multimodal retrieval datasets like OVEN and InfoSeek? and even a decline in performance on some tasks (e.g., task1 and task3)? How do the authors explain this phenomenon, and could it be related to the modality of the retrieval task?
3. In line 343, the authors mention that MLLM-based retrievers tend to retrieve text over images. Could this be related to the sampling of negative samples during training? If there are more image samples in the dataset, this could lead to a higher proportion of images as negative samples, which could further induce bias. I suggest conducting a statistical analysis on the modality distribution of the data samples.

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
The paper presents techniques for advancing information retrieval with multimodal large language models (MLLMs). It first finetunes MLLM-based retrievers to tackle universal multimodal retrieval tasks and shows that MLLM-based retrievers exhibit modality bias in cross-modal retrieval tasks compared to CLIP-based retrievers. To address the issue, it proposes modality-aware hard negative mining and continual fine-tuning, and the MLLM-based retriever, UniEmb, yields state-of-the-art retrieval accuracy in universal multimodal retrieval tasks while maintaining strong text-to-text retrieval capability. Moreover, it explores to prompt MLLMs as a re-ranker to further boost retrieval accuracy.

### Strengths
1.	The paper is well-motivated to use multimodal LLMs for advancing information retrieval and support broader retrieval scenarios.
2.	The experiments are relatively comprehensive to validate the effectiveness of the proposed method.

### Weaknesses
1.	The paper claims that existing retrieval models typically addressed search scenario where retrieval task are fixed and only a single modality is supported for both queries and retrieved results, but a series of retrieval models for multimodal retrieval are listed in related work.
2.	The paper claims that it is the first work to explore prompting MLLMs fir zero-shot reranking, but as far as I know, the work [a] has used MLLMs for reranking, and the differences should be explained in detail.
3.	Some expressions are confused. For example, in Line 308 of Page 6, “we initialized retriever using the pretrained model”, does the “retriever” refer to the model to be fine-tuned with hard negatives?
4.	It is not easy to understand Figure 1, where the task definition of Task 1-13 is missing. The relationship between the upper left and lower right sub-figure is not clearly illustrated.
5.	How is the performance when using fine-tuned MLLMs for reranking? Is it better than reranking in a zero-shot manner for tasks that involve single-modal queries?
6.	More qualitative cases are suggested to be provided to better illustrate the effects, such as how modality-aware hard negative mining alleviates modality bias. 

[a] Qu L, Li H, Wang T, et al. Unified Text-to-Image Generation and Retrieval[J]. arXiv preprint arXiv:2406.05814, 2024.

### Questions
Please refer to the Weaknesses section and address my concerns.

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
4

### Summary
This paper explores a more practical scenario in information retrieval, namely general multimodal retrieval. The author proposes a new fine-tuning pipeline and re-ranking method to improve the performance of specific retrieval tasks. However, the organization of the experimental results of this paper is rather coarse, and the performance presented makes me concerned about its generalization and robustness. It is recommended that the author further improve the presentation of results and methods. However, it is undeniable that the research topic can bring contribution and inspiration to the multimodal community.

### Strengths
1. The topic explored by the authors is a great one, namely general multimodal retrieval.
2. The method proposed by the author is effective to a certain extent.
3. The authors describe the experiment in detail and I think it is reproducible.

### Weaknesses
1. Unfortunately, the authors are not the first to use MLLM for re-ranking. It is recommended that the author faithfully investigate related research, e.g., [1]. The authors should narrow down the scope of the study, i.e., composed image retrieval.
2. The author's claim in line 184 should be verified experimentally using either very large k or very small k.
3. Line 306: The author's fine-tuned batch size for the two types of models is different. Will this affect the effect of mining negative samples?
4. I personally think that CLIP is a product from 3 years ago, and more advanced models should be used, such as BLIP or an improved version of CLIP. Moreover, the comparative experiment lacks experimental results of some recent methods that solve specific tasks, and it is impossible to quantitatively feel the advantages of the proposed MLLM-based method. The current organization is more like a complete ablation experiment.
5. From the results in Table 4, the full version re-ranking results do not seem to be optimal, which makes it difficult to explain the effectiveness of the method's continuous fine-tuning on some tasks. The improvement margin of its full version seems to be even worse than M^hard, which makes it hard to convince me.
6. From the studied results, the author expects to achieve universal retrieval, but as far as the experimental results (e.g., Table 7) are concerned, the proposed method will reduce the performance in some benchmarks. This makes me concerned about the generalization and robustness of its method.

[1] MERLIN: Multimodal Embedding Refinement via LLM-based Iterative Navigation for Text-Video Retrieval-Rerank Pipeline

### Questions
See weaknes.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
In this paper, the authors focus on developing multimodal LLMs (MLLMs) for universal multimodal retrieval. Instead of directly fine-tuning MLLM-based bi-encoder retrievers with instructions as a guide on multiple multimodal retrieval tasks, the authors propose modality-aware hard negative mining and continual text-to-text retrieval fine-tuning to address the modality bias problem from MLLMs. Furthermore, the authors explore to prompting MLLMs as zero-shot rerankers, which can further boost retrieval accuracy in complex retrieval tasks. Extensive experimental results demonstrate the effectiveness of the proposed method.

### Strengths
1)	This paper explores the use of multimodal LLMs for universal multimodal retrieval, which is an interesting and potential research topic for the retrieval field.
2)	In this paper, the authors explore universal multimodal retrieval from three aspects, including modality-aware hard negative mining, continual text-to-text retrieval fine-tuning, and prompting MLLMs as zero-shot rerankers. These components are designed to improve the effectiveness of MLLM-based universal multimodal retrieval, while maintaining strong text-to-text retrieval capability.
3)	Extensive results demonstrate the effectiveness of the proposed method.

### Weaknesses
1)	The details of the proposed method are not clear and confusing. In sec. 4.1.1, it seems random negatives are denoted as c_i^{\times} symbol, while the generated triplets use c_i^{\times} again, does it mean positive sample or random negative?
2)	In sec. 4.2, the authors prompt LLaVa-Next for reranking, while not introducing the detailed MLLMs adopted in sec. 4.1. Are there two multimodal LLMs needed for the proposed method? 
3)	It will be better to provide the framework figure for the whole method. 
4)	In Table 4, it is not clear which results denote the reranking results, it seems the last row does not outperform the fifth row.
5)	As a retrieval method, the efficiency of retrieval time and memory is as important as effectiveness. The authors should analyze the efficiency of the unimodal retrieval mechanism.

### Questions
please try to address the weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3
