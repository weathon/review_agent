# FreeRet: MLLMs as Training-Free Retrievers

- Decision: Reject
- Scores: 8, 2, 6

## Abstract
Multimodal large language models (MLLMs) are emerging as versatile foundations for mixed-modality retrieval.
Yet, they often require heavy post-hoc training to convert them into contrastive encoders for retrieval.
This work asks: \textit{Can off-the-shelf MLLMs serve as powerful retrievers without additional training?}
We present \textbf{FreeRet}, a plug‑and‑play framework that turns any MLLM into a two‑stage retriever.
FreeRet first derives semantically grounded embeddings directly from the model for fast candidate search, and then exploits its reasoning ability for precise reranking.
The framework contributes three advances: bypassing lexical alignment layers to obtain semantically faithful embeddings, conditioning representation generation with explicit priors, and mitigating framing effect in reranking via neutral choice framing.
On the MMEB and MMEB-V2 benchmarks spanning 46 datasets, FreeRet substantially outperforms models trained on millions of pairs.
Beyond benchmarks, FreeRet is model-agnostic and scales seamlessly across MLLM families and sizes, preserves their generative abilities, supports arbitrary modality combinations, and unifies retrieval, reranking, and generation into end-to-end RAG within a single model.
Our findings demonstrate that pretrained MLLMs, when carefully harnessed, can serve as strong retrieval engines without training, closing a critical gap in their role as generalists.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper presents FreeRet, a two-stage framework that directly uses off-the-shelf MLLMs as "training-free" retrievers: first, use the internal representations of the model for candidate recall, and then use the same model for fine-grained re-ranking. The core idea is to no longer perform contrastive learning or additional supervision, but rather to transform the model's semantic understanding and reasoning abilities into usable retrieval signals through careful design of architecture and prompts. The three key technical points of the authors are as follows: 
- Semantic Representation Extraction: Bypass the last layer MLP before the language head to alleviate "lexicalization pressure", and use more semantically faithful hidden states as embeddings;
- Controlled Representation Generation: Using prompts that incorporate task priors, semantic anchoring, and noise suppression, the generation process of "summarize in one word" is constrained to be a more stable and topic-relevant representation;
- Neutralization rearrangement hint: Rewrite the binary classification relevance judgment into a multiple-select question to alleviate the bias of labels such as Yes/No in pre-trained corpora (which the author refers to as the "LLM framing effect").

On 46 datasets such as MMEB and MMEB-V2, FreeRet significantly outperformed multiple comparison methods trained with millions of samples without any additional training; under larger model scales, it even approached or exceeded the SOTA directly supervised on the benchmark.

### Strengths
The innovation of the paper does not lie in inventing a brand-new architecture, but in proposing three key and mutually coordinated approaches to "how to transform the potential capabilities of existing MLLMs into high-quality retrieval signals": First,  analyze and avoid the lexicalization pressure of the last layer of MLP , supported by probe experiments on intra-layer similarity and alignment with the LM head; Second,  transform the freely generated one-word summary into controlled generation , constrained by concise prompts to align the vocabulary space with the task; Third,  reveal and mitigate the pragmatic bias of re-ranking labels , replacing binary labels with MCQ to obtain more stable judgments. These all belong to the originality of "de-engineering": significantly improving the usability of representation and discrimination without modifying the model or conducting training.

### Weaknesses
1. The measurement of the framing effect is mainly conducted in classification scenarios, and it is recommended  to extend it to VQA, retrieval question answering, and cross-modal attribute matching scenarios  to examine the sensitivity of MCQ copywriting design to different tasks. 

2. The paper discusses engineering strategies under ultra-large candidate scales in the appendix, but  measured overhead figures  (throughput, latency, and video memory usage) are still lacking. Suggestions: 
- Under unified hardware and batch settings, report the end-to-end latency and GPU utilization of recall and re-ranking;
- Presentthe curve of candidate pool size - performance - latencyto quantify the cost - effectiveness inflection point of "controlled reordering candidate count";

3. The author emphasizes the cross-family and cross-task stability and suggests further: 
- Supplementary Error Analysis : Statistically analyze the most easily confused negative example types and error causes (such as lexical bias, modal alignment failure, long-distance dependency loss, etc.) to guide subsequent prompt or structural improvements;
- More experiments on RAG task such as WebQA and MultimodalQA.
- More experiments on retrieval task such as Flickr30K and MS-COCO.

### Questions
1. Although controlled prompts suppress function words, the differences in common word frequencies and morphological structures among different languages are relatively large. Have the authors considered generating within a restricted task vocabulary/category set (e.g., news topic sets, object category sets) to avoid cross-lingual word distribution drift? If run on Chinese or mixed corpora, what impact does lemmatization have on performance?

2. In VQA, cross-modal attribute matching, or retrieval question answering, the  option text  and  explanatory prompt  of MCQ may affect the coupling of the model's "reasoning - judgment". Have the authors systematically compared the effects of different MCQ copywriting (more neutral vs. more fine grain)? Is there an inflated result caused by "overly strong prompts"? 

3. Though the paper mentions that framing the ranking problem as a MCQ is to alleviate the LLM framing effect, it is mentioned in [1] that the responses of LLMs to MCQ are actually unstable. How do the authors view this issue?

ref:

[1] Zheng C, Zhou H, Meng F, et al. Large language models are not robust multiple choice selectors[J]. arXiv preprint arXiv:2309.03882, 2023.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This work aims to convert MLLM into retrievers by training-free methods, where the authors claim that fine-tuning MLLM needs too-much data source and computation, and would show little generalization. Their proposed FreeRet provide three advcances: 1) refine embedding quality by bypassing the final MLP before LM head of MLLM, 2) stabilize the embedding space via controlled summarization prompts, and 3) cast reranking as multiple-choice questions to mitigate LLM framing effect. By combining the FreeRet embedding and reranking models,  FreeRet surpass a range of heavly fine-tuned embedding models on MMEB, which is a unfair comparson. I like the paper many points, such as probing experiments on lexicalization pressure. But I think the experiment part leads to a unscientific framework and findings.

### Strengths
1. The idea of training-free multimodal retriever approach, which is very practical in many low-resource domains and scenarios.
2. The probing of lexicalization pressure in MLLM representation, very well thinking.

### Weaknesses
1. The model performance contribution is mixed, FreeRet is a retreival-stack of embedding+reranking, comparing to solely embedding models is clearly unfair. I think is absolutely a big issue.
2. The research contribution is somehow unclear, we can not know the effectiveness of embedding part of FreeRet.
3. The reranking part (LLM framing effect) is not a scientific or technical contribution IMHO, just a prompt engneering.

### Questions
1. I would suggest authors to show the embedding result on main table 1 & 2.
2. It's would be helpful to understand the contribution by comparing with some zero-shot rerank baselines, such as ones in MM-Embed [1]. 

[1] MM-Embed: Universal Multimodal Retrieval with Multimodal LLMs (Lin et al, ICLR 2025)

### Soundness
1

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes a plug-and-play framework that turns any pretrained multimodal large language model (MLLM) into a two-stage retriever without further training.

### Strengths
- It introduces a fully parameter-free retrieval framework that repurposes any existing MLLM as an effective retriever, avoiding the cost of multimodal fine-tuning.
- The work identifies bottlenecks such as lexicalization pressure in the final MLP, and reveals the LLM framing effect in reranking.
- Unlike fine-tuned retrievers, it retains the original model’s multimodal understanding and generative capabilities.

### Weaknesses
- Limited novelty, although the training-free idea is appealing, the improvements mostly in prompt design.
- The method’s success relies heavily on manually crafted prompts and framing choices, which may not generalize well across unseen domains or languages without careful tuning.
- Running the same large MLLM for both embedding and reranking can be computationally expensive at inference time, yet the paper provides little discussion on latency or resource cost.
- The insights provided by the Probing Experiment are limited. I believe it is expected that the final layer undergoes a transformation of the representation space, so the experiment offers only weak support for the claim of lexicalization pressure. Moreover, in the analysis of Semantic Retention, using only synonym pairs is insufficient to demonstrate that semantic resolution degrades after the LM head. Including distance analyses among synonym, antonym, and random pairs would make the argument more convincing. In addition, I think a deeper analysis and explanation are needed to clarify why lexicalization pressure leads to performance degradation.

### Questions
I suggest that the authors further analyze and investigate the generalization ability of the proposed prompts, in order to verify their robustness and effectiveness in open-ended or noisy retrieval scenarios. In addition, the relationship between lexicalization pressure and model performance requires further investigation.

### Soundness
3

### Presentation
3

### Contribution
3
