# ReCalKV: Low-Rank KV Cache Compression via Head Reordering and Offline Calibration

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 6, 4, 4

## Abstract
Large language models (LLMs) have demonstrated remarkable performance, but their long-context reasoning remains constrained by the excessive memory required for the Key-Value (KV) cache. This makes KV cache compression a critical step toward efficient long-context inference. Recent methods have explored low-rank techniques to reduce the hidden size of the KV cache. However, they neglect the distinct roles and varying importance of Keys and Values, leading to significant performance drops under high compression. To address this, we propose ReCalKV, a post-training low-rank KV cache compression approach with tailored strategies for Keys and Values. For Keys, we propose Head-wise Similarity–aware Reordering (HSR), which clusters structurally similar heads into groups, enabling more accurate low-rank approximation via grouped SVD. For Values, we propose Offline Value Calibration (OVC), which efficiently calibrates the value projection matrix using calibration data without training, ensuring an accurate representation of contextual information. Extensive experiments show that ReCalKV consistently outperforms existing low-rank compression methods, achieving high compression ratios with minimal performance loss. We will release all the code and models.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper addresses the problem of high memory and bandwidth overhead caused by the KV cache in LLM during long-context during. This paper proposes ReCalKV, a post-training low-rank KV cache compression method that introduces differentiated strategies. it applies Head-wise Similarity-aware Reordering to cluster structurally similar attention heads before grouped SVD for Key compression, and Offline Value Calibration to recalibrate Value projection matrices using small calibration datasets. 
The experimental results demonstrate that ReCalKV achieves consistent improvements over prior low-rank compression baselines （Palu and LoRC). It maintains competitive perplexity and zero-shot accuracy under 50–70% KV-cache compression ratios on models including LLaMA-2-7B, Mistral-7B, and LongChat-7B. However, all experiments are conducted on relatively older architectures, and no evaluation is reported on Llama-3 or Qwen-3.

### Strengths
1. The paper tackles an important and practical problem—reducing KV-cache memory overhead for long-context LLM inference, which remains a key bottleneck for efficient deployment.
2. The proposed approach is model-agnostic and can be readily applied to various Transformer architectures without retraining, showing potential for integration into large-scale serving systems.

### Weaknesses
1. The reported experimental performance, while better than earlier SVD-based baselines, remains clearly inferior to recent quantization-based methods such as KVQuant and AnTKV, which achieve much lower perplexity under similar even higher compression ratios.
2. ReCalKV still introduces additional computations for restruct KV using low rank kv (compute with R_k and R_v) during each decoding step. I recommend the authors evaluate latency and accuracy on end-to-end tasks such as AIME.
3. The evaluation focuses mainly on outdated models (e.g., LLaMA-2, Mistral-7B) and lacks results on modern architectures like LLaMA-3 or Qwen-3, making it difficult to assess real-world relevance.

### Questions
see weakness

### Soundness
2

### Presentation
2

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
The paper introduces ReCalKV, a post-training framework for compressing the Key-Value (KV) cache in large language models (LLMs) by reducing the hidden dimension via low-rank approximations. It proposes asymmetric strategies: Head-wise Similarity-aware Reordering (HSR) for Keys, which reorders and groups attention heads based on Centered Kernel Alignment (CKA) similarity to enable more accurate grouped Singular Value Decomposition (SVD); and Offline Value Calibration (OVC) for Values, which calibrates the decomposed SVD matrices using a small dataset and fuses the right factor into the output projection to eliminate runtime reconstruction overhead. Extensive experiments on LLaMA and Mistral models demonstrate superior perplexity, zero-shot QA accuracy, and long-context performance compared to baselines like Palu, with minimal degradation (e.g., ~2% relative accuracy drop at 50% compression) and compatibility with quantization for higher ratios.

### Strengths
* K-side uses CKA-guided head reordering + grouped SVD to share low-rank factors among similar heads (greedy pairing; fixed group size), and V-side uses closed-form offline calibration to minimize projection error on a small calibration set.  

* Matrix fusion folds (R_v) into (W_o), eliminating online reconstruction and avoiding extra inference ops; the end-to-end procedure is fully post-training/offline (Algorithm 1).  

* Strong ablations isolate HSR and OVC and show they are complementary at fixed compression (Table 3).  

* Evaluations span multiple model families and tasks, plus quantization compatibility (3–4-bit) demonstrating orthogonality to per-token KV quantization (Table 4).  

* Figures 2–3 make the reordering/grouped-SVD mechanism concrete; Algorithm 1 spells out the pipeline; equations (9–11) specify the fused-inference path.  

* Targets a real deployment bottleneck (KV memory/latency) with demonstrable inference efficiency improvements on long contexts, while remaining compatible with common compression stacks.

### Weaknesses
- Code not provided, therefore it's not reproducible as is.
- While the method is effective, baselines are limited primarily to Palu (G-LRD), lacking comparisons with recent variants like CommonKV or FDC, which could better substantiate SOTA claims (section 4).
- Experiments do not quantify runtime overhead from Key reconstruction post-HSR (Figure 3), despite claims of low cost; real-world latency measurements on diverse hardware would strengthen efficiency arguments (Figure 4). 
- Equations (7) and (8) for OVC appear to have typos in transposes and do not explicitly state assumptions (e.g., whitening) needed for the closed forms.

### Questions
- Please address the items mentioned under Weaknesses. For example, lack of reproducibility.
- In section 3.3 (lines 216-269), the OVC calibration uses equations (7) and (8) with a small dataset X (256 WikiText2 samples; section 4.1). How sensitive is performance to the size and domain of X? Could you provide perplexity results on WikiText2 for LLaMA-2-7B at 50% compression using 128 vs. 512 samples, or a different domain like C4?
- Section 3.2 describes HSR as greedy grouping based on the CKA similarity matrix S (Eq. 5) with a fixed group size (e.g., 4 heads per group when  h=32). Table 3 shows that at 80% compression, HSR+OVC attains 8.48 perplexity on WikiText-2. What is the effect of the HSR group size s on performance?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces ReCalKV, a post-training framework for low-rank KV cache compression that treats Keys and Values separately. It enhances Key approximation  through similarity-based head grouping and decomposition and refines Value through lightweight calibration and fusion. Experiments demonstrate that ReCalKV consistently outperforms prior methods under high compression rates.

### Strengths
1. The paper identifies and analyzes the asymmetric roles of Keys and Values, particularly emphasizing that individual attention heads differ in information content. Using CKA-based head reordering before SVD to minimize approximation error is a well-motivated and conceptually sound idea.

2. Across multiple model families and compression ratios, ReCalKV demonstrates competitive or superior results compared with the main low-rank baseline (Palu). The method maintains high accuracy even under aggressive compression and shows compatibility with quantization.

### Weaknesses
1. The paper mainly compares with low-rank SVD-based approaches such as Palu, but lacks comparisons with other classes of KV cache compression methods (e.g., KIVI, KVQuant, or token eviction approaches).
As a result, the reader cannot fully assess how ReCalKV performs in a broader landscape of KV compression techniques — especially when low-rank compression is not necessarily the only or best strategy.

2. Experiments focus on older LLaMA/Mistral models, with limited evaluation on recent architectures or larger scales. Since the method relies on specific structural properties of attention heads (CKA similarity patterns), it's unclear whether these properties generalize across diverse modern architectures and model scales beyond the tested family.

3. While latency speedups are reported, the computational cost of online head reordering during inference is not quantified separately. The reliance on custom Triton kernels also raises questions about achievability with standard inference frameworks, limiting practical deployment insights.

### Questions
see weakness

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces a novel low-rank KV cache compression method, building on PaLU, a prior work that decomposes the KV projection matrix using SVD to reduce the dimension of the KV cache. The paper makes the following contributions on top of PaLU:
- Reordering the key projection matrix to achieve a better SVD decomposition.
- Refining the low-rank approximation of the value projection matrix using a calibration dataset.

### Strengths
- Strong improvements over PaLU on the evaluated models.
- Comprehensive ablation studies demonstrating the effectiveness of each contribution for both key and value projections.

### Weaknesses
**[W1]** The evaluated models are outdated. I suggest moving the results on the Llama-3.1 model to the main body. This is important because many modern LLM architectures employ GQA, while only the Mistral model from the main results section does so. Validation on multiple models with GQA would strengthen the paper.

**[W2]** Comments on writing:
- *L55–60:* This is not something revealed by your analysis.
- *L60–63:* I cannot find a section describing your analysis of Fisher information.
- The fact that whitening is applied before SVD should be discussed earlier in the Methods section, with more detail.

Minor comments that did not affect the score:
- *L69:* “Offline Calibration Value” --> “Offline Value Calibration”

### Questions
**[Q1]** Is there a reason why offline calibration is applied only to the value projection matrices? Could this also be applied to the key projection matrices?

**[Q2]** How effective is the proposed method in terms of the memory–accuracy trade-off compared to other KV cache compression methods beyond those based on SVD of the projection matrices?

**[Q3]** How would the method perform for reasoning models such as the Qwen3 model family on long generation tasks like AIME or LiveCodeBench? Demonstrating this would highlight the method’s robustness under long-generation scenarios, which are not captured by perplexity or long-context retrieval tasks.

### Soundness
2

### Presentation
2

### Contribution
2
