# MAR: Medical Asymmetric Retriever for Efficient Chinese Medical Dense Retrieval

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 4, 6

## Abstract
Embedding models are critical for domain-specific information retrieval (IR), particularly in healthcare, where accurate, low-latency access to medical knowledge can enhance clinical decision support and mitigate hallucinations in retrieval-augmented generation (RAG) systems.
However, Chinese medical retrieval remains underdeveloped due to the absence of high-quality medical retrieval benchmark. To address this limitation, we propose a novel high-quality Chinese **Med**ical **T**ext **E**mbedding **B**enchmark (**MedTEB**), which covers three practical tasks close to real-world scenarios: retrieval, reranking, and semantic textual similarity (STS). We introduce comprehensive LLM-based annotation in the construction process to improve the quality of curated datasets. Through evaluating existing powerful general-purpose embedding models on MedTEB, we demonstrate that MedTEB is a challenging domain-specific embedding benchmark to evaluate models' retrieval capabilities on Chinese medical retrieval.
On this foundation, we propose **M**edical **A**symmetric **R**etriever (**MAR**), an asymmetric embedding architecture that decouples query and document encoding: a lightweight encoder handles online queries with minimal latency, while a powerful and offline LLM-based encoder preserves retrieval quality. Optimizing the asymmetric architecture brings to new challenges. We introduce a novel two-stage optimization framework: 1) **query encoder alignment** and 2) **joint fine-tuning**.
Through the novel approach, MAR achieves state-of-the-art (SOTA) performance on MedTEB while maintaining lightweight inference speeds comparable to small-size BERT-style embedding models, leading to an excellent trade-off on accuracy and efficiency and thus offering a practicable and effective solution for real-world Chinese medical retrieval scenarios.
Our code, data and model will be made publicly available to facilitate future research on domain-specific IR.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces MedTEB, a new benchmark for Chinese medical text embeddings covering three tasks—retrieval, reranking, and synonym STS—and proposes MAR (Medical Asymmetric Retriever), an asymmetric dual-encoder framework pairing a lightweight query encoder with a large document encoder.

The authors then design a two-stage optimization scheme (query alignment and joint fine-tuning) and demonstrate state-of-the-art performance on MedTEB while maintaining efficiency comparable to smaller BERT-style models.

### Strengths
1. This paper introduces a comprehensive benchmark (MedTEB) for Chinese medical embeddings with multiple tasks, addressing an underexplored but practically important domain.

2. This paper provides thorough empirical evaluation comparing MAR against a wide range of strong baselines (BGE, GTE, Qwen, etc.), demonstrating consistent efficiency–accuracy improvements.

### Weaknesses
1. The proposed MAR framework is primarily a combination of existing asymmetric retrieval and distillation methods (e.g., KALE, HotelMatch), with only minor architectural differences (removing projection layer, using MSE + InfoNCE). The methodological innovation is minimal.

2. Annotation reliability not verified: LLM-based labeling is claimed to be high-quality, but no inter-LLM agreement or human validation statistics are reported.

3. Improvement margins are modest (≈1.3–1.5%), and the methods rely heavily on proprietary LLMs for annotation. The authors also did not  compare with baselines using more parameters such as Qwen-3-embedding-4b, GTE etc.

4. It would be better to also evaluate over different retrieval-based applications such as chinese medical QA. 

5. Missing references on medical retrievers:

> [1] Xu et al. "BMRetriever: Tuning Large Language Models as Better Biomedical Text Retrievers." Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing. 2024.
> [2] Jin et al. "Medcpt: Contrastive pre-trained transformers with large-scale pubmed search logs for zero-shot biomedical information retrieval." Bioinformatics 39.11 (2023): btad651.
> [3] Li et al. "R2MED: A Benchmark for Reasoning-Driven Medical Retrieval." arXiv preprint arXiv:2505.14558 (2025).

### Questions
1. Can you quantify inter-LLM agreement rates or human verification accuracy to support the claim of “high-quality annotation”?

2. How sensitive is MAR to the choice of base document encoder (e.g., Qwen3 vs. BGE)?

3. How would MAR perform in non-medical Chinese retrieval tasks? Does the alignment generalize beyond the medical corpus?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces MedTEB, a high-quality benchmark for Chinese medical text retrieval, reranking, and synonym understanding, addressing limitations in existing datasets. To enable both strong accuracy and low latency, the authors propose MAR, an asymmetric retriever that combines a lightweight query encoder with a powerful offline document encoder, aligned through a two-stage training strategy. MAR achieves state-of-the-art performance on MedTEB while remaining efficient for real-time medical retrieval and RAG applications.

### Strengths
+ By leveraging the capabilities of LLMs, this paper validates the issues and shortcomings present in current benchmarks, thereby uncovering noise-related problems in the data. 
+ The authors introduce an asymmetric retrieval framework and demonstrate its effectiveness on this benchmark, achieving state-of-the-art performance.
+ The research objectives are clearly defined and effectively addressed.

### Weaknesses
Perhaps due to space constraints, several issues remain unclear to me:  
+ For instance, in which specific aspects do the existing benchmarks fall short, and to what extent are these deficiencies present? Can experiments based solely on LLMs sufficiently support the claim made in the original text: "However, the retrieval tasks suffer from annotation noise and false negatives, leaving only the reranking tasks relatively reliable"?  
+ The description of the benchmark construction process in this paper lacks sufficient detail. It is unclear whether many of the construction details, such as data distribution, adhere to benchmark construction standards.  
+ The proposed model employs an asymmetric structure, yet why are all the baselines symmetric? The selection of baselines does not appear to be comprehensive.

### Questions
See above

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The research seeks to address the challenge in medical information retrieval where existing search systems must tradeoff being fast but inaccurate or accurate but slow, making them impractical for real-time clinical use. This problem is stated as especially acute for Chinese medical text, where testing benchmarks are incomplete and specialized tools are lacking. Researchers propose two contributions: MedTEB, a comprehensive benchmark using real medical queries and multi-AI verification systems, and MAR (Medical Asymmetric Retriever), an innovative two-part search architecture that separates the search process into complementary components.

MARs uses a small "query encoder" to process searches in real-time paired with a large, powerful "document encoder" that pre-processes medical documents offline. This approach—trained through a three-stage process of independent training, alignment, and joint fine-tuning—eliminates the traditional speed-accuracy trade-off. The results claim to demonstrate that MAR achieves the high accuracy of larger models while maintaining the processing speed of smaller ones.

### Strengths
The work seeks to address the tradeoffs between speed, computational cost and accuracy often encountered in document retrieval systems. This is an important (albeit well-established) challenge in this area. The application to Chinese text documents and a curated, benchmark dataset of such documents along with the code is also a clear contribution of this work. Also, and in general, the paper is well-written and technically sound.

### Weaknesses
As noted above, the challenge itself is not novel, per se, nor is the strategy of preprocessing and creating an index offline and using the index at run time (hence my rating). Perhaps more detail on what is special about this specific manifestation of the problem and on the 2-tier solution could increase the novelty of this work. Also, once the dataset and the code has matured and the community has built on it, this work will have much greater impact. The authors should be encouraged to pursue this direction.

### Questions
Could the authors provide a better characterization of the novelty of both the problem and the solution?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses key gaps in Chinese medical dense retrieval: the lack of high-quality benchmarks and the accuracy-efficiency trade-off of embedding models. It proposes two core contributions: (1) MedTEB, a Chinese Medical Text Embedding Benchmark covering 3 new tasks (Retrieval, Reranking, Synonym STS) and 2 existing high-quality datasets, constructed via multi-LLM consensus annotation to reduce noise; (2) MAR, a Medical Asymmetric Retriever with a lightweight online query encoder and a powerful offline document encoder, optimized via a two-stage framework (query alignment + joint fine-tuning) to achieve SOTA performance on MedTEB while maintaining low latency. The authors also commit to open-sourcing the benchmark, models, and code.

### Strengths
1. Fills the gap of reliable Chinese medical retrieval benchmarks: MedTEB solves annotation noise and false negatives in existing datasets (e.g., C-MTEB), providing a rigorous evaluation standard.
2. Practical asymmetric architecture: MAR balances real-time deployment needs (low-latency query encoder) and retrieval quality (powerful document encoder), breaking the accuracy-latency trade-off.
3. Significant domain value: Focuses on underdeveloped Chinese medical retrieval, supporting clinical decision support and medical RAG; open-source resources facilitate follow-up research.

### Weaknesses
1. Insufficient validation of LLM annotation professionalism: No clinical expert sampling verification or inter-annotator agreement metrics (e.g., Cohen’s Kappa) for MedTEB labels, risking "pseudo-professional" errors.
2. Unclear LLM annotation disagreement details: Fails to report the proportion of partially agreed samples or analyze disagreement causes, potentially compromising benchmark comprehensiveness.
3. Ambiguous document encoder pretraining: No information on the proportion of medical data in Qwen3’s pretraining corpus; lacks baseline comparisons (e.g., Llama3-8B) to verify Qwen3’s medical advantages.
4. Incomplete ethical/data compliance: No details on IRB approval for online user queries, specific anonymization steps, or exact public data sources, raising compliance concerns.
5. Lack of comparison with medical RAG methods: No performance comparison with domain-specific methods (e.g., Hyper-RAG), limiting competitiveness assessment.

### Questions
see weaknesses

### Soundness
3

### Presentation
2

### Contribution
4
