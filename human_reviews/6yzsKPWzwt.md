# Core Context Aware Attention for Long Context Language Modeling

- Decision: Reject
- Scores: 6, 5, 3

## Abstract
Transformer-based Large Language Models (LLMs) have exhibited remarkable success in various natural language processing tasks primarily attributed to self-attention mechanism, which requires a token to consider all preceding tokens as its context to compute the attention score. However, when the context length L becomes very large (e.g., 32K), more redundant context information will be included w.r.t. any tokens, making the self-attention suffer from two main limitations: 1) The computational and memory complexity scales quadratically w.r.t. L; 2) The presence of redundant context information may hamper the model to capture dependencies among crucial tokens, which may degrade the representation performance. In this paper, we propose a plug-and-play Core Context Aware (CCA) Attention for efficient long-range context modeling, which consists of two components: 1) Globality-pooling attention that divides input tokens into groups and then dynamically merges tokens within each group into one core token based on their significance; 2) Locality-preserved attention that incorporates neighboring tokens into the attention calculation. The two complementary attentions will then be fused to the final attention, maintaining comprehensive modeling ability as the full self-attention. In this way, the core context information w.r.t. a given token will be automatically focused and strengthened, while  the context information in redundant groups will be diminished during the learning process. As a result, the computational and memory complexity will be significantly reduced. More importantly, the CCA-Attention can improve the long-context modeling ability by diminishing the  redundant context information. Extensive experimental results demonstrate that our CCA-Attention significantly outperforms state-of-the-art models in terms of computational efficiency and long-context modeling ability.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces CCA-Attention, which consists of globality-pooling attention and locality-preserving attention. This plug-and-play method reduces computational cost while improving performance in long-context processing.

### Strengths
1. This paper introduces CCA-Attention, which leverages local information and compressed remote information to model long texts. The method is novel and effective.

2. Experiments and ablation studies demonstrate that CCA-Attention achieves strong performance while reducing computational costs during both training and inference.

3. The paper is well-written and clearly explains the details of CCA-Attention.

### Weaknesses
1. The paper only compares CCA-Attention with StreamingLLM and LM-infinite, without including existing methods that retrieve relevant information in long contexts, such as MInference [1] and InfLLM [2]. Additionally, it does not compare the proposed method with models directly trained on longer texts.

2. The evaluation of long-text capabilities is limited to multi-document question answering. It would be beneficial for the authors to evaluate the methods on a wider range of tasks, such as those in the RULER [3] and LongBench [4] benchmarks.

[1] MInference 1.0: Accelerating Pre-filling for Long-Context LLMs via Dynamic Sparse Attention
[2] InfLLM: Training-Free Long-Context Extrapolation for LLMs with an Efficient Context Memory
[3] RULER: What's the Real Context Size of Your Long-Context Language Models?
[4] LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding

### Questions
1. The attention mechanisms are applied to the query of the last token and the key and value of the core tokens. However, during the formation of core tokens, the importance of each token in the group is determined by the query of the last token. Could information that is relevant to the query but less important within the group be overlooked?

2. As discussed in Weaknesses 1 and 2, could the authors provide experiments with more benchmarks and alternative methods? Additional results would further substantiate the effectiveness of the proposed methods.

3. Does the method support FlashAttention? If not, could the authors provide the time and space costs of the method compared to direct inference and training with FlashAttention?

4. The MMLU performance of StreamingLLM and LM-infinite appears unusual. Since the MMLU samples are short, these methods should perform similarly to the original model. Could the authors investigate these results?

Currently, I give a weak reject. My scores will rise if the authors add more experiments and respond to my questions.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This work focuses on reducing the computational and memory complexity of the attention mechanism in Transformer architecture to enable efficient long-range context modeling with additional fine-tuning. The authors highlight the existence of redundant contextual information in attentions and propose Core Contect Aware(CCA) Attention to diminish this redundancy while essuring reachability within the token sequence. The proposed CCA Attention is made up of a globality-pooling attention and a locality-preserved attention, combined through a learnable parameter. The model with CCA can be easily initialized with pre-trained parameters for further fine-tuning and demonstrates consistent improvements across three metrics from different aspects when compared to several baselines.

### Strengths
* Overall, the paper is well-structured. 
* The proposed approach is well-motivated, easy to understand, and supported by detailed equations and illutrative diagrams.
* The discussions on ablations and parameter searches demonstrate the efficacy of the proposed CCA-Attention.

### Weaknesses
A notable concern is that the baselines adopted in this work may be too weak. Two of them (StreamingLLM and LM-Infinite) are training-free approaches aimed at enabling LLMs trained with a finite-length attention window to generalize to infinite sequence length without any fine-tuning, rather than reducing the complexity attention mechanism for efficient long-range context modeling with fine-tuning. More comparisons with models based on other sparse attention or linear attention approximations from prior work are expected.

### Questions
1. How do models based on other sparse attention or linear attention approximations perform under the same training configuration?
	* Additionally, Table 3 reflects the different trends of PPL and MMLU metrics. Althought the PPL of Max Pooling is slighly lower than that of Mean Pooling, its MMLU score is significantly better than the other one's. This raises concerns about whether adopting LoRA+ will decrease LongLoRA's performance as presented in Table 2. 

2. Why is the performance of SinkAttention even worse than that of Vanilla Self-Attention, especially in the aspect of Inference Speed and Inference Memory Usage? Also, what is the performance of LongLoRA in this context?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
The paper introduces a Core Context Aware Attention (CCA-Attention) mechanism for enhancing computational efficiency in long-context language modeling. Traditional self-attention models face inefficiencies with long sequences due to high computational and memory demands. CCA-Attention addresses this by introducing two mechanisms: (1) Globality-pooling attention, which groups and reduces tokens to core representatives, and (2) Locality-preserved attention, which maintains contextual information from neighboring tokens. These components are adaptively fused, reducing redundancy and computational costs while improving long-context understanding. Experimental results demonstrate CCA-Attention’s efficiency and superior performance compared to state-of-the-art models.

### Strengths
1. The paper is well-written and clearly structured.
2. The method demonstrates a significant speed improvement.

### Weaknesses
1. The method evaluates on too few benchmarks; recent long-document evaluation tasks, such as Longbench, Ruler, and Infinitebench, are available. Many papers have indicated that PPL is not an accurate metric.
2. In Table 2, why wasn’t LLaMA-2 used with continued training for comparison?
3. In Table 2, why does the MMLU score initially decrease and then increase as the Training Context Length increases?

### Questions
See weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2
