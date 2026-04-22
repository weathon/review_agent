# Beyond Speedup - Utilizing KV Cache for Sampling and Reasoning

- Avg Score: 4.80
- Decision: Accept (Poster)
- Scores: 4, 4, 6, 6, 4

## Abstract
KV caches, typically used only to speed up autoregressive decoding, encode contextual information that can be reused for downstream tasks at no extra cost. We propose treating the KV cache as a lightweight representation, eliminating the need to recompute or store full hidden states. Despite being weaker than dedicated embeddings, KV-derived representations are shown to be sufficient for two key applications: (i) Chain-of-Embedding, where they achieve competitive or superior performance on Llama-3.1-8B-Instruct and Qwen2-7B-Instruct; and (ii) Fast/Slow Thinking Switching, where they enable adaptive reasoning on Qwen3-8B and DeepSeek-R1-Distil-Qwen-14B, reducing token generation by up to $5.7\times$ with minimal accuracy loss. Our findings establish KV caches as a free, effective substrate for sampling and reasoning, opening new directions for representation reuse in LLM inference.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work investigates the use of the KV cache as intermediate representation instead of the commonly used hidden states for two reasoning-focused downstream tasks: The first is an adapted Chain-of-Embedding method based on the KV cache that is roughly on par with existing CoE, while reusing the already stored KV cache. The second is a novel reasoning mode meta-model that uses the aggregated KV caches to optionally insert \<think\> / \</think\> tokens to expand / decrease "reasoning" via more token output. On the simpler GSM8K dataset models with a "fast" thinking path already perform on par with more expensive (longer) traces, but on the MATH500 the proposed method shows performance improvements at smaller token budgets in between a "fast" and "slow" thinking mode.

### Strengths
- shows that KV caches contain similar information content compared to hidden states for CoE downstream method
- introduces a KV-cache based thinking mode switch (either ahead of or continuously during generation)
- shows there is optimization potential towards existing dedicated embedding models

### Weaknesses
- using hidden states can be similarly optimized for downstream tasks (e.g. in-place / recursive summation across layers / tokens - if common simple aggregations are used)
- no comparison to a hidden state-based thinking mode switching 
- fails to cite cache tuning methods (Prefix-Tuning [1] / Cartridges [2]) as related work of where the KV cache is actually modified

[1] Li et al. (2021): https://arxiv.org/abs/2101.00190

[2] Eyoboglu et al. (2025): https://arxiv.org/pdf/2506.06266

### Questions
- Can you explain what the CoE is used for in these reasoning benchmarks, i.e. how the extracted embeddings are used towards what kind of goal? This is not clear from the paper.
- How is the fast thinking and slow thinking induced / separated without the KV cache-based control? Can you provide exact prompts?
- Could you add Prefix-Tuning [1] and Cartridges [2] as related work operating on the KV cache (though more in the direction of fine-tuning adapters)

[1] Li et al. (2021): https://arxiv.org/abs/2101.00190

[2] Eyoboglu et al. (2025): https://arxiv.org/pdf/2506.06266

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper repurposes the KV caches in LLMs for purposes beyond attention. The authors explore the use of KV embeddings as text embeddings and as classifier features for fast/slow thinking. Their results indicate that KV cache embeddings contain significant information that can be used in other applications. However, I think the empirical validation is a bit limited.

### Strengths
1. KV caches already take up significant space, so having them perform multiple functions seems well-motivated
2. Demonstrated that KV caches can show promise for other purposes
3. No architectural changes needed

### Weaknesses
1. Fast/slow thinking experiments are lacking. The method was only validated on GSM8k and MATH500, both of which are fairly simple for the reasoning models tested. Thus, it may just be that these models are fairly robust for these relatively easy tasks, so I would recommend testing on more challenging reasoning tasks (AIME, BRUMO, etc)
2. Numbers in Table 3 are reported using the best hyperparameters per task/model combination. This feels too cherry-picked since I didn't find a way to know which hyperparameters to choose before testing.

### Questions
1. I am unfamiliar with the datasets/tasks in Table 1. What is the accuracy for random chance?
2. What is the distribution of the training data difficulty labels detailed in page 8?
3. Can these methods be applied to non-reasoning generative tasks like summarization?
4. Table 3: Do the token averages also contain sequences that hit the generation limit (i.e. incomplete)?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents a novel approach to repurposing the KV cache, a critical component of efficient autoregressive decoding in large language models (LLMs), for downstream tasks beyond just acceleration. The authors demonstrate that the KV cache, despite not being explicitly trained as a general-purpose embedding, nonetheless encodes rich contextual information that can be effectively leveraged for two key applications: (1) Chain-of-Embedding (CoE) and (2) Fast/Slow Thinking Switching.

### Strengths
1. The paper repurposes the KV cache and provides new insights. It also demonstrates the versatility of the KV cache through two practical applications: CoE and Fast/Slow Thinking Switching.

### Weaknesses
1. I would appreciate it if the authors could provide more details on MaxProb, PPL, and Entropy, specifically, what these metrics represent and why they are chosen as baselines in Table 2.
2. From my understanding, CoE explores the model’s representation space through hidden states, whereas the KV cache contains only information derived from the attention heads. Considering that hidden states integrate residual connections, MLP outputs, and attention information, I am curious how KV-CoE effectively tracks representations across layers under this limitation.

### Questions
1. According to the official format, the table caption should appear before the table.
2. Line 239 contains a typo: "is obtained by by averaging"
3. Could the authors please elaborate on the pooling strategy used in the experiments?
4. Is there any accuracy trade-off associated with the improvement in fast and slow thinking when using the KV cache?

### Soundness
2

### Presentation
1

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper explores the application of KV-Cache for two different applications than speeding up decoding: (1) Chain-of-Embedding a method for self-evaluation of LLMs and (2) Fast/Slow Thinking Switching a method that enables adaptive reasoning based on KV-Cache embeddings of the prompt exclusively or the prompt in combination with the reasoning trace. 
In their experiments, this paper demonstrates that (1) outperforms other baselines on self-evaluation tasks and (2) that we can enable adaptive reasoning based on difficulty scores derived from the KV-cache embeddings.

### Strengths
- Clear motivation, well written and interesting research question.
- Strong performance of KV-CoE on self-evaluation tasks.
- The thinking switch experiments demonstrate the practicability of thinking mode switching based on KV-cache embeddings.

### Weaknesses
- Even though the paper is well written, it seems a bit like the combination of two methods into one paper.
- After reading the introduction the reader might think that KV-caches can be used as embeddings, but the result in section 4.1 contradicts this initial claim or limits it to “certain classification tasks”, which are not specified further. I suggest clarifying this confusion in the paper and/or adding additional details on the “certain classification tasks”.
- KV-CoE seems to outperform standard CoE for the Qwen Models (Table 2), but not for the LLama models. It would help the reader to add an explanation (or run the baselines with the correct base model) or mention in the main text that KV-CoE outperforms default CoE.
- Do the reasoning results in Table 3 also transfer to larger models e.g. Qwen3 32B?


Even though the paper could add a few more details or explanations at some places (see questions & weaknesses) the authors convincingly demonstrate the effectiveness of their proposed methods based on embeddings stemming from the KV-Cache. Therefore, I am inclined to recommend acceptance of this paper.

### Questions
- L.176 Formatting issue with Table 1 (whitespace missing)
- Description of the metric used in Table 1 (assume higher is better)
- L.164 The KV Cache embeddings are constructed by averaging over attention heads. Have you also tried concatenating over attention heads instead of averaging? This would counteract limitation 3 in L. 180.
- In addition in Sec. 5.2 the KV embedding is constructed by flattening the head and key/value dimension. This seems to be inconsistent.
- Is it not possible to recompute the embeddings efficiently from the KV cache? How much overhead would it be? Is this really a bottleneck? 
- Why does the KV-CoE operate along the token position and not along the layer index? Have you tried using the layer index as in default CoE?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
I am not very well versed with this literature. So I will focus mostly on clarification questions at this time. I will defer my decision to later time after the rebuttal and other reviewer comments.

In this paper, authors propose a way to construct embeddings from KV cache instead of using hidden spaces to be used for CoE and adaptive reasoning problems. While constructing the embedding, they aggregate across layers (instead of tokens as done in earlier CoE works) and measure path characteristics along token dimension (as opposed to layers done in earlier CoE). The argument in favor of choosing KV caches for embeddings is that they do not incur any additional cost.

### Strengths
1. Using KV Cache for self-evaluation avoids extra memory overhead. 
2. The accuracy improvements are impressive especially for MATH dataset.

### Weaknesses
Writing is a bit messed up. for instance, 

Section 4. makes arguments are not quite clear.
	a. why is min \gamma (x) > 0 enough for this task? how is this class of problems different from a classification problem. One can argue that the same thing holds for classification as well. 
        b. contextual conditionaing argument is not clear to me. Can the author's elaborate.
        c. Efficiency constraint argument is okay but unnecessarily formalized.
        d. I dont understand the discussion on local decision adequacy. How do you determine whether a problem is local / global. for all you know you need to know the solution of the problem to know how hard it is -- knowing the solution is the most global problem in context of LLM generation. 
I wonder if the usage of LLM is quite strong in this section and it needs rework.


I am not sure if state of the art baselines are compared. (see questions)

### Questions
1. How does hidden state based embedding perform as compared to off the shelf embedding? (table 1)
2. How is COE used in practice. Is it a pass@k while choosing the highest ranking pass?
3. Do you have a ablation of what happens when you use CoE with switching layer and token dim (the way you propose with KV)?
4. There were a bunch of baselines mentioned in the related work Chen 2024, Beigi 2024, Zhang 2025 for CoE problem and ASRR, PATS, DOTS for adaptive reasoning problem. Are these baselines relevant to the experiments you ran. It would be illuminating to compare your method against these methods if they are SOTA. 
5. why were CoE results taken from the paper when they belonged to a previous version of model which has very different context windows among other things. ( in table 2).

### Soundness
2

### Presentation
1

### Contribution
2
