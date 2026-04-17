# Training-Free Multi-Token Prediction via Probing

- Decision: Reject
- Scores: 4, 4, 2, 2

## Abstract
Large Language Models (LLMs) possess latent capabilities for multi-token prediction (MTP), despite being primarily trained for next-token generation. We introduce a simple, **novel, and training-free MTP** method that probes an LLM using **on-the-fly generated mask tokens** derived from the model’s own embedding space. These mask tokens guide the model to predict multiple future tokens in parallel without modifying model weights or relying on external draft models.
To assess the quality of these predictions, we propose a dynamic draft tree construction mechanism based on cumulative token probabilities, coupled with a simple pruning algorithm that removes redundant token paths. Our method supports multiple mask token designs, enabling flexible probing strategies to improve parallel prediction quality.
We also define block complexity—the number of tokens processed per model iteration—as a key setting for controlling compute usage during inference. Under equal block complexity, our approach consistently outperforms existing training-free baselines such as Lookahead decoding and Prompt-lookup-decoding in terms of **block efficiency** (acceptance rate), significantly **reducing number of model calls**. Finally, we provide **theoretical and empirical justification for why our method works**. We show that decoder layers progressively align mask token representations with valid token states, and formalize this connection through a lemma linking cosine similarity to Top-$K$ prediction accuracy, supported by empirical evidence.
We evaluate our method on Spec-Bench using `LLaMA3` and `Qwen3` models, providing both quantitative and qualitative analyses. Our probing-based MTP improves block efficiency by $\sim12\\%$ for `LLaMA3` and $8–12\\%$ for `Qwen3` models and tokens per second by upto $\sim 15-19\\%$ over baselines **without retraining or auxiliary models**.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents a novel, training-free method for multi-token prediction (MTP). The core idea is to probe a large language model using mask tokens generated on-the-fly from the model's own embedding space, with the goal of significantly reducing the number of required model calls for generation. Another component of the proposed system is a dynamic tree expansion mechanism, which adaptively grows the decoding tree based on the predicted tokens. The authors validate their method through experiments on the SpecBench benchmark.

### Strengths
* The finding that an LLM's embedding space can be probed to elicit multi-token prediction is a surprising and interesting discovery.

* This finding leads to a novel, training-free approach that capably supports lossless generation while requiring fewer forward evaluations.

* The development of a dynamic tree construction mechanism with efficient caching represents a meaningful and practical systems-level contribution.

### Weaknesses
* The paper fails to discuss or contextualize itself against related work on masked diffusion LLMs, which represent another promising architecture for multi-token prediction using mask tokens. [1, 2, 3, 4]

* A significant weakness is the lack of explanation for why the proposed probing strategy works. The paper demonstrates that it works, but provides very little intuition or analysis regarding the underlying mechanism that makes this probing possible in a model trained exclusively for next-token prediction.

* The use of "block efficiency" as the primary metric is potentially misleading. While the method may decode 1.5 tokens per forward pass, this metric obfuscates the true wall-clock speedup. The overhead introduced by handling masked inputs and the verification process could diminish or even negate the gains. The authors should report throughput in their comparison against relevant baselines like STAND and LADE.

[1] Sahoo, Subham, et al. "Simple and effective masked diffusion language models." Advances in Neural Information Processing Systems 37 (2024): 130136-130184.

[2] Wu, Chengyue, et al. "Fast-dllm: Training-free acceleration of diffusion llm by enabling kv cache and parallel decoding." arXiv preprint arXiv:2505.22618 (2025).

[3] Hu, Zhanqiu, et al. "Accelerating diffusion language model inference via efficient kv caching and guided diffusion." arXiv preprint arXiv:2505.21467 (2025).

[4] Israel, Daniel, Guy Van den Broeck, and Aditya Grover. "Accelerating Diffusion LLMs via Adaptive Parallel Decoding." arXiv preprint arXiv:2506.00413 (2025).

### Questions
1. Given that the model is trained for next-token prediction, it is surprising that an input embedding can elicit a prediction at the position of the input mask rather than for the next position. Do the authors have any additional insight into the mechanism that enables this probing to work?

2. Speculative decoding can be extended to sampling [5], and most practical applications of LLMs rely on sampling to ensure response diversity. Is there a specific reason the experiments were limited to greedy decoding (as mentioned on line 269)?

3. For the sake of completeness, how does this training-free method compare in performance to existing approaches that require training?

[5] Chen, Charlie, et al. "Accelerating large language model decoding with speculative sampling." arXiv preprint arXiv:2302.01318 (2023).

### Soundness
1

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes a training-free multi-token prediction (MTP) method that probes frozen large language models (LLMs) using dynamically generated mask tokens derived from the model’s own embedding space. These mask tokens enable parallel prediction of multiple future tokens without modifying model weights or using external draft models. The authors introduce a dynamic token tree construction mechanism guided by cumulative probabilities and a lightweight pruning strategy to improve prediction diversity and efficiency. Evaluated on Spec-Bench with LLaMA3 and Qwen3 models, their approach consistently outperforms existing training-free baselines in block efficiency, reducing model calls by up to 40% while maintaining lossless generation.

### Strengths
1. Training-free and plug-and-play: Requires no fine-tuning, auxiliary models, or architectural changes, making it easy to deploy across diverse LLMs.
2. Efficient dynamic decoding: The adaptive token tree expansion and pruning strategy improve block efficiency while respecting compute constraints (block complexity).
3. Strong empirical performance: Achieves consistent gains (8–12% higher block efficiency) over state-of-the-art training-free baselines across multiple tasks and model families.

### Weaknesses
1. A primary limitation is its practical deployability. The "plug-and-play" claim is misleading because the method's core relies on a highly custom "Causal Tree Attention Mask" for simultaneous verification and probing. This non-standard architecture is fundamentally incompatible with the highly-optimized, standard autoregressive mechanisms of SOTA inference frameworks like vLLM (e.g., PagedAttention). Integrating it would require a major re-engineering of the core attention and scheduling kernels, which contradicts the "plug-and-play" promise and is noted by the authors themselves as future work.

2. The method's generalizability is severely limited as all experiments are conducted only under greedy decoding (Temperature=0). This overlooks the predominant use of stochastic sampling (Temperature > 0, Top-P/K) in real-world applications like chatbots to ensure response diversity. The "probing" mechanism is fundamentally optimized to predict the highest-probability token path, which directly conflicts with the goal of sampling. Consequently, the moment a sampler selects a creative, non-greedy token, the entire probed draft tree is likely to be invalidated, causing the method's performance to collapse to that of standard autoregressive generation.

### Questions
- I understand that the authors use Block Efficiency to demonstrate the improvement in inference efficiency. However, I would also like to know about more conventional metrics, such as speedup ratio and mean number of accepted tokens per speculation step, as these would allow readers to make direct comparisons with other methods evaluated on SpecBench. I hope the authors could provide results using these standard metrics as well.

- Please provide the speedup results at temperature = 1, as measured by other speculative decoding (SPD) methods.

- Please adapt your method (e.g., by simplifying the tree structure) to integrate it with vLLM. I would like to see a more practical experimental result, and I hope the authors can provide the original throughput and the improved throughput.

I will adjust my score based on the author's Rebuttal answer

### Soundness
3

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
This paper proposes a training-free MTP method that probes an LLM using on-the-fly generated mask tokens derived from the model’s own embedding space. These mask tokens guide the model to predict multiple future tokens in parallel without modifying model weights or relying on external draft models. They also propose a dynamic token tree to assess the quality of these predictions, and use block efficiency to control compute usage during inference. Experimental results show the effectiveness of the method.

### Strengths
1. This paper is technically sound and easy to understand. 
2. The experimental results show the effectiveness of the proposed method.

### Weaknesses
1. Why do you use block complexity/efficiency rather than tokens/s or mean accept tokens to judge the compute usage during inference when comparing to other training-free methods? It is more reliable to see if the actual speed-up of the proposed method is better than other competitors. 
2. It is far from enough to show the token per second results of your model and base model in Tab.3. It is also far from enough to only validate results on two models. Since it is a training free method, it should be easy to generalize to many different types (Qwen, Deepseek, etc.) and sizes of models (70B). The authors should conduct experiments on more of them. 
3. The paper focuses on generating tree structure and improve the overall MAT to speedup the large language model. However, they conduct experiments without using popular inference frameworks. Here comes a problem that the method may not have such speedup on the popular inference framework such as vLLM. In fact, HuggingFace Transformers framework does not optimize the speed of LLMs very well, which makes the ratio of the latency of tree generation process smaller. When using vLLM framework where the operations in LLMs are optimized very well, the tree generation process will take more time and reduce the speedup. The author should verify their method on such inference frameworks to show that their method is actually useful in reality.

### Questions
See weaknesses above.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes a prompt based probing method for muli-token prediction with LLMs which requires no special training or finetuning. The core of the method is a way to generate special masking tokens that are appended to a prompt in order to stimulate the multi-token prediction.

### Strengths
The paper compares different ways to generate the mask tokens. Further, it provides experiments across the LLaMA and Qwen-3 model families, covering performance and speed comparisons.

### Weaknesses
While the paper presents an interesting approach, several issues limit its clarity and contribution.

Formatting and Presentation: The overall formatting is not clean, which affects readability. For instance, Table 3 extends beyond the document boundary, making it difficult to interpret. Attention to layout and consistency in formatting would improve the paper’s professionalism.
- Notation Inconsistency: The mathematical and symbolic notations are inconsistent throughout the paper. This inconsistency makes it hard to follow the proposed method and understand the exact mechanisms involved. A clearer and standardized notation scheme would help readers grasp the technical details more effectively.
- Missing Related Work and Comparisons: The paper overlooks several important and directly relevant studies, which weakens the contextualization of the proposed method:
a) arXiv:2505.10518: Introduces a prompt-level multi-token prediction (MTP) approach using trained register tokens instead of heuristic masking, offering a more principled token selection strategy.
b) arXiv:2405.18628: Proposes a method that also employs trained prompt tokens and integrates a dynamic tree structure, providing an alternative perspective on structured token prediction.
c) arXiv:2502.09419: Demonstrates that large language models trained for next-token prediction inherently exhibit multi-token prediction capabilities, an insight that the current paper treats as novel.

### Questions
1) How to you compare to the related literature mentioned above
2) In section 3.1 you provide 3 heuristics for mask token injection. Can you provide reasoning why you chose exactly those? Are these the only choices possible?

### Soundness
2

### Presentation
2

### Contribution
2
