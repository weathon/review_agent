# APE: Faster and Longer Context-Augmented Generation via Adaptive Parallel Encoding

- Decision: Accept (Poster)
- Scores: 6, 6, 8, 6, 6

## Abstract
Context-augmented generation (CAG) techniques, including RAG and ICL, require the efficient combination of multiple contexts to generate responses to user queries. Directly inputting these contexts as a sequence introduces a considerable computational burden by re-encoding the combined selection of contexts for every request. To address this, we explore the promising potential of parallel encoding to independently pre-compute and cache each context's KV states. This approach enables the direct loading of cached states during inference while accommodating more contexts through position reuse across contexts. However, due to misalignments in attention distribution, directly applying parallel encoding results in a significant performance drop. To enable effective and efficient CAG, we propose Adaptive Parallel Encoding (**APE**), which brings shared prefix, attention temperature, and scaling factor to align the distribution of parallel encoding with sequential encoding. Results on RAG and ICL tasks demonstrate that APE can preserve 98\% and 93\% sequential encoding performance using the same inputs while outperforming parallel encoding by 3.6\% and 7.9\%, respectively. It also scales to many-shot CAG, effectively encoding hundreds of contexts in parallel. Efficiency evaluation shows that APE can achieve an end-to-end 4.5$\times$ speedup by reducing 28$\times$ prefilling time for a 128K-length context. The code is available at 
https://github.com/Infini-AI-Lab/APE.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
APE proposed the adaptive parallel encoding of LLM prompt, to speedup the prefill stage without limited by pretrained context length. APE add sink tokens (shared prefix) and split the context into chunks and perform attention inside of it. During decoding stage, it attends to every previous KV without adjusting the RoPE, so it will not exceed the pretrained context length.

### Strengths
- Nice empirical analysis to show value state can be merged & key states are similar for each context chunk.
- Speeding up prefill stage drastically.

### Weaknesses
1. The methodology might be too similar, with fixed step sparse attention with sink token. When we draw the attention mask of APE, the attention mask itself is extremely similar to the step attention mechanism. However, the difference between APE and APE is that APE reuses the RoPE embeddings rather than just extends them. However, treating the RoPE index is the same as streaming with fixed step sparse attention. Therefore, I think the scientific contribution of methodology is limited. 

2. The performance of the method is mostly proved by empirical results. I do not think these empirical results are weaknesses, but I have concerns about the lack of important comparisons with previous technologies to extend the context window and speed up prefill.

- Lack of comparison with training-free context extension method (Self-Extend: https://github.com/datamllab/LongLM).

- Lack of comparing with pretrained long context LLM. This might not be a big problem, but we need to know what performance is the upper limit if the LLM is already trained in a long context. The LLMs used for experiments are all short context (and maybe a little bit old at this point) models. I am concerned that using a long-context model such as Qwen2 or Llama3.1 that supports 128k tokens with a large-scale GPU cluster might lead to better performance than APE.

- Lack of comparing with techniques to speed up the prefill stage.

--- Some minor improvements ---

- Figure 3 and 4 are quite hard to understand. Can you increase the font size? And also I do not think we need visualized every layers. Can you randomly sample some layers and show them only? (e.g., range(0, 32, 4)).
- In Table 1, there is a typo in Gemma2. Maybe you shifted the columns to the left.
- Tables 2, 4, and 5 might have a too large a size.
- Table 2 should improve the formatting.

### Questions
1. In efficiency analysis, the claim `976x faster` might be a little bit overclaiming because there is no performance (accuracy) evaluation on 512k. Can you add relevant benchmarks? (InfiniteBench: https://github.com/OpenBMB/InfiniteBench/, LOFT: https://github.com/google-deepmind/loft, RULER: https://github.com/NVIDIA/RULER)

2. Table 5: what is the average context length of APE and real-world, end-to-end latency, including retrieval latency? As far as I understand the CRAG paper, the context length is quite limited with a smaller size (2k~4k) due to the speed of retrieval.

3. What do you mean that `query and generation lengths were fixed at 256 tokens` in line 464? Is that mean, the chunk size of APE is 256? As far as I see the latency, I think 256 means the chunk size, which is NOT used in performance evaluation. Do you have any performance evaluation with 256 chunk size? If my understanding is correct, then the reported latency is might be significantly different with real-world scenario.

### Soundness
4

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces Adaptive Parallel Encoding, a method to improve the efficiency of large language models when processing multiple external contexts. APE pre-caches key-value states of contexts separately and enables position reuse during inference. The method consists of three key components: a shared prefix to align initial token distributions, a scaling factor to offset increased attention weights, and lower attention temperature to focus on semantically important tokens. The authors demonstrate that APE achieves a 976× speedup for long context generation while maintaining 93% accuracy compared to sequential encoding.

### Strengths
1. The authors provide thorough empirical analysis to understand the behavior of attention mechanisms and KV states. 
2. The method is practical as it requires no additional training and can be implemented with minimal modifications to existing architectures. 
3. Comprehensive evaluation across multiple tasks (ICL, RAG, long-context understanding)

### Weaknesses
1. The performance evaluation is primarily focused on 8k context length, which feels insufficient given that many open-source LLMs now support context lengths of 128k or more. This restricted evaluation scope makes it difficult to assess the method's scalability to longer contexts. 
2. Compared to sequential encoding, APE introduces a non-negligible performance degradation at the 8k length scale, raising concerns about its effectiveness at longer context lengths.

### Questions
1. How does the computational complexity of APE scale with increasing context length compared to sequential encoding?
2. Why does the method perform particularly poorly on code completion tasks with continuous long contexts?
3. How sensitive is the method to the choice of hyperparameters (scaling factor and attention temperature) across different models and tasks?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper investigates the performance degradation issue in parallel encoding and analyzes the distribution patterns of key and query vectors. This paper analysis reveals that key vector distributions remain relatively similar and query vector distributions can be combined. Based on these findings, they propose APE (Adaptive Parallel Encoding) to mitigate performance losses in LLMs during parallel encoding. Specifically, APE incorporates three key components: shared prompt prefix, position-aware dynamic scaling, and attention temperature adjustment. We evaluated APE on three LLMs (Llama-3-8B, Llama-2-7B, Gemma-2-9B) across multi-document QA and few-shot ICL tasks. Results demonstrate performance improvements compared to full attention and baseline methods, with latency benchmarks showing significant speedup over full attention.

### Strengths
1. The research focuses on a practical and significant challenge, particularly relevant to real-world LLM applications like RAG.
2. The paper provides insightful observations and analysis regarding the feasibility of parallel encoding.

### Weaknesses
1. This paper demonstrates limited evidence for performance superiority over baselines across various tasks. Specifically:
  - The evaluation is restricted to only two task categories from LongBench (multi-document QA and few-shot ICL), which is insufficient to demonstrate the method's effectiveness across diverse scenarios.
  - For multi-document QA tasks, the evaluation is conducted on a limited subset of LongBench, with additional testing only on one RAG dataset in Sec 6.3. This narrow scope of evaluation fails to provide comprehensive evidence of the method's effectiveness. And Table 5 lacks crucial baseline comparisons, making it difficult to assess the relative performance improvements.
  - The evaluation of other task types in Sec 6.2 is inadequate, missing essential baseline comparisons needed for meaningful performance assessment.

2. And this paper lacks sufficient component analysis and validation.
  - The paper lacks comprehensive ablation studies to isolate and validate the contribution of each proposed component. This makes it impossible to determine whether performance improvements stem from specific modules (e.g., attention temperature adjustment) or their combination.
  - There is insufficient analytical evidence demonstrating how the three proposed components effectively address the challenges identified in the earlier sections of the paper. The causal relationship between the proposed solutions and the observed improvements needs stronger empirical support.
  - The absence of detailed component-wise analysis makes it difficult to justify the necessity and effectiveness of each module in the proposed architecture.

### Questions
1. Formatting issue: In Table 1 #413 Gemma-2-9B line has miss alignment.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The papers introduce a simple and effective modification to parallel decoding which can be used a training-free drop-in method to substantially increase efficiency of inference while retaining most of its accuracy.

### Strengths
* The paper evaluates across many different models of different architectures. 
* The method is training-free which enables it be easily tested across many model -- assuming a working implementation.
* Shows significant gains on efficiency and would have an real-world impact on deployed models. For example, enabling Pre-caching contexts.

### Weaknesses
* Paper does not provide implementation of the method -- this can make it easier for reproducibility. Would it be possible to provide that?
* The paper fixes a windows size but shows no empirical or theoretical analysis of choosing different sizes of this window: either from an efficiency or quality perspective. It would be great to have empirical analysis.
* No complexity analysis written out. This could be particularly relevant if we want to vary window above.

### Questions
* Would it be possible to include RULER as an evaluation for long-context task. It has become more standard for LC. Showing the impact of APE on needle in a haystack would be interesting as well.
* Are the tasks where APE fails. For example, Does APE still hold when used for Humaneval or RepoBench?
* What would be the impact of finetuning with examples that had APE applied to then. Would it stop degradation?
* Can you provide an analysis of the impact of changing the "window" size you use for parallel decoding.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper presents Adaptive Parallel Encoding (APE) in order to improve efficiency and performance in language models handling RAG and ICL tasks. APE overcomes limitations in sequential encoding, which requires re-encoding context data and suffers from context window restrictions, by employing parallel encoding with adaptive adjustments. These changes include a shared prefix, scaling factors, and a modified attention temperature, aligning parallel encoding closely with sequential encoding. Experiments show APE achieves a high speedup in long contexts and outperforms slightly other methods in accuracy for long contexts.

### Strengths
- The method sounds intuitive. APE extends the usable context length for language models, overcoming limitations in the context window and enabling efficient handling of much larger inputs without additional training.
- The performance is great esp. in latency
- The analysis in Section 3 looks quite novel and interesting

### Weaknesses
- The performance seems quite sensitive on specific parameters

### Questions
- Will we conclude anything between the task and the choice of hyperparameters?
- The latency improvement in Table 2 would be more interesting if other baselines are included.

### Soundness
3

### Presentation
3

### Contribution
2
