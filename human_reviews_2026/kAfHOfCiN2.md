# MHA-RAG: Improving Efficiency, Accuracy, and Consistency by Encoding Exemplars as Soft Prompts

- Avg Score: 3.00
- Decision: Reject
- Scores: 4, 2, 2, 4

## Abstract
Adapting Foundation Models to new domains with limited training data is challenging and computationally expensive. While prior work has demonstrated the effectiveness of using domain-specific exemplars as in-context demonstrations, we investigate whether representing exemplars purely as text is the most efficient, effective, and stable approach. We explore an alternative: representing exemplars as soft prompts with an exemplar order invariant model architecture. To this end, we introduce Multi-Head Attention Retrieval-Augmented Generation (MHA-RAG), a framework with the number of attention heads serving as a simple hyperparameter to control soft prompt-generation across different tasks. Across multiple question-answering benchmarks and model scales, MHA-RAG achieves a 20-point performance gain over standard RAG, while cutting inference costs by a factor of  10X GFLOPs—delivering both higher accuracy and greater efficiency, invariant to exemplar order.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces a method that represents exemplars as soft prompts in an invariant manner before feeding them into large language models (LLMs) for question-answer generation. The approach treats the number of attention heads as a tunable hyperparameter to control soft prompt generation across different tasks. Experiments on diverse QA benchmarks show that the proposed MHA-RAG significantly outperforms standard RAG methods and enhances inference efficiency in domain adaptation settings.

### Strengths
- The paper presents a strong motivation by clearly identifying the limitations of In-Context Learning (ICL) and the challenges of applying in-context retrieval in Retrieval-Augmented Generation (RAG) for domain adaptation. The proposed approach, which replaces hard token exemplars with soft prompts, proves effective not only in improving task performance (QA accuracy) but also in reducing inference time.
- The paper is generally well-written and well-structured. It includes extensive experiments and ablation studies, offering several insightful empirical findings. These include comprehensive comparisons with traditional prompt tuning, LoRA, and the Instance-Dependent Prompt Generation (IDPG) method.

### Weaknesses
- The baseline comparisons are limited, as only xRAG and its variant xRAG-K are included. Given that other related baselines are mentioned, it would strengthen the paper to include comparisons with those methods as well. Furthermore, the work lacks evaluation against other stronger RAG-based approaches (such as adaptive RAG categories) in terms of both task performance and inference cost.
- The proposed method depends on retrieving relevant exemplars from the training set, which may restrict its practicality in domains where no training data is available.
- The experiments are conducted only on relatively small models (up to 4B parameters), leaving the effectiveness of the proposed method on larger-scale LLMs uncertain.

### Questions
In Table 1, it is unclear why Qwen3-4B performs significantly worse than Qwen3-0.6B on the ClinTox dataset. Furthermore, after applying MHA-RAG, Qwen3-4B continues to underperform compared to Qwen3-0.6B. Could the authors clarify the reasons behind this unexpected result?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces Multi-Head Attention Retrieval-Augmented Generation (MHA-RAG), a framework that represents retrieved in-context exemplars as soft prompts using multi-head attention mechanisms. The key innovation is using the number of attention heads as a tunable hyperparameter to control soft-prompt generation, claiming to achieve order-invariance through scaled dot-product attention. The authors evaluate MHA-RAG on molecular property prediction tasks (BACE, BBBP, ClinTox) and biomedical QA (PubMedQA) using small language models (Qwen3-0.6B/4B, Llama3.2-3B-Instruct). Results show an average 20-point performance gain over standard RAG while reducing inference costs by 10× in terms of GFLOPs.

### Strengths
1. The theoretical guarantee of order-invariance through scaled dot-product attention is sound and formally proven. This addresses a real problem in RAG systems where exemplar order can affect performance (Table 2 shows clear variance in baselines).

2. MHA-RAG shows consistent performance gains over RAG and xRAG variants across the molecular property prediction tasks and PubMedQA (Table 1), with an average improvement of 19.66 points.

3. The paper includes useful ablations on number of heads (Figures 3, 5) and value of K (Table 3), showing that performance generally improves with more heads when sufficient context is available.

4. Appendix A.3 (Table 6) provides interesting analysis showing that adding back a small number of text exemplars (c=5) alongside soft prompts can further improve performance, suggesting the soft prompts and text are complementary.

### Weaknesses
1. Missing Baselines: The paper does not compare against crucial recent methods:
* AttentionRAG: Attention-Guided Context Pruning in Retrieval-Augmented Generation, which uses model attention to prune retrievals in a training-free manner.
* EXIT: Context-Aware Extractive Compression for Enhancing Retrieval-Augmented Generation, a Context-aware extractive compression for RAG that preserves contextual dependencies. Reported to outperform both compressed and uncompressed baselines.
* Context Embeddings for Efficient Answer Generation in RAG, which compresses contexts into embeddings for RAG, handles multiple documents, offers tunable compression rates. This is a direct competitor that should be compared.

Without these comparisons, we can't assess whether MHA-RAG represents an improvement over current SOTA or is merely competitive with outdated baselines (xRAG from NeurIPS 2024).


2. The efficiency claims are undermined by missing measurements:
* Only GFLOPs are reported, but additional encoding overhead from sentence embedding model is not counted. Meanwhile, the attention computation over K exemplars with H heads has its own cost. Moreover, there is no comparison of actual inference time on identical hardware.
* The paper doesn't report 1)Peak memory consumption during inference, 2)M emory required to store embedding model + foundation model, and 3) Memory footprint compared to RAG with FlashAttention, which is a common optimization used in applications

### Questions
* Can you provide more comparisons with the recent baselines mentioned in the weakness? 
* Modern RAG systems use 7B+ models. Can you provide results on Llama-3-8B, Mistral-7B, or similar scales to test if the method scale?
* Can you provide more details around the efficiency, including 1) wall-clock latency measurements on identical hardware, 2) peak memory consumption during training and inference, 3) complete FLOPs including encoding overhead, and 4) comparison with FlashAttention-enabled RAG?

ps:
* Table 4 shows LoRA achieving 0.0 effective accuracy on 9 out of 12 settings. This is unreasonable for a well-established method. Can you provide learning curves, loss plots, etc. to explain this? 
* Table 3 shows MHA-RAG performance degrades at K=10 compared to K=5, does it mean the proposed methods may not scale to more examples?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
In this paper, the authors propose multi-head attention retrieval-augmented generation to adapt foundation models to new domains with limited data, thereby addressing 3 challenges for conventional RAG, including high inference costs from long textual contexts, suboptimal performance on out-of-distribution data, and sensitivity to the order of retrieved exemplars. Instead of appending exemplars as text, MHA-RAG encodes them into compact soft prompts using multi-head scaled dot-product attention and ensures order invariance and achieves high compression ratios.

The experiments are conducted on low-data molecular property prediction tasks and  biomedical QA. The results show the improvements in effective accuracy and FLAPs reductions. The authors also conclude by positioning the proposed method as a cost-effective RAG alternative with future work on scaling to longer documents.

### Strengths
* Strong empirical gains in accuracy while reducing educing inference costs
* Multi-head attention addresses the performance fluctuation issue of RAG with true order invariance
* The hyperparameter offers some flexibility to tune trade-offs between effectiveness and efficiency across models and domains

### Weaknesses
* Limited scope of benchmarks that only focus on specific domains in the experiments.
* Lack of robustness analysis for noisy and adversarial exemplars
* Missing discussion on training efficiency

### Questions
* How does MHA-RAG perform on open-ended text generation tasks, such as summarization or creative writing?
* I wonder if the authors can provide some specific examples of failure cases where MHA-RAG produces incorrect outputs.
* The retrieval function could matter. I wonder if the authors have studied on the choice of this.
* What results does MHA-RAG achieve when using foundation models from other families, like GPT or Mistral?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces MHA-RAG, a model that encodes each retrieved document as distinct key and value embeddings. A multi-head attention (MHA) layer is then applied to the query, key, and value embeddings. Each head within the MHA layer learns a unique embedding. These head embeddings are subsequently concatenated to form a fixed-size embedding matrix for any given set of k documents.
The experimental evaluation of MHA-RAG was conducted in domains where the base Large Language Model (LLM) exhibits suboptimal performance, thereby necessitating the integration of retrieved documents. The experimental results align with the theoretical analysis, confirming MHA-RAG's order invariance. Through careful selection of hyperparameters, such as K and H, MHA-RAG demonstrates superior performance compared to baseline models in most evaluated scenarios.

### Strengths
1. The proposed method is clearly presented and easy to follow.
2. Extensive experiments were conducted to prove the effectiveness of the MHA-RAG, and the effects of different hyperparameters were also studied.
3. MHA-RAG is justified as being order-invariant, a characteristic that cannot be guaranteed by the baseline methods.

### Weaknesses
1. Limited Generalization: All datasets utilize pre-defined labels (e.g., "Yes" or "No"), which raises concerns regarding the model's ability to generalize to open-domain Question Answering (QA) tasks.
2. Inconsistent Performance: MHA-RAG fails to consistently outperform the basic RAG baseline (as shown in Table 1), and the reasons for this inconsistency are not explained.
3. Incomplete Comparison (Table 1): Table 1 exclusively presents performances with K=5. This limited scope raises concerns about a fair comparison, as other baseline methods might perform better with a different value of K.
4. Missing Hyperparameter Details (Table 3): While Table 3 studies the effect of varying K, the corresponding value of H used in these experiments is not provided.

### Questions
1. Section **Performance Metrics** (line 273): what is the range of o_{i}?
2. Figure 1 shows that MHA-RAG freeze foundation model and only train the encoder the multi-head attention. Can you clarify in *Training Details* (line 309), what does the `Models` refer to?
3. What is the metric for evaluating PubMedQA dataset in **Performance Metrics**? 
4. What is the number of attention heads in Table 3?

### Soundness
3

### Presentation
3

### Contribution
2
