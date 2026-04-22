# ICaRus: Identical Cache Reuse for Efficient Multi-Model Inference

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 6, 4, 6

## Abstract
Multi model inference, where multiple task-specialized models collaborate to solve complex real-world problems, has recently emerged as a prominent paradigm, particularly in the development of agentic AI systems. However, in such scenarios, each model must maintain its own Key-Value (KV) cache for the identical prompt, leading to explosive memory consumption.
This explosive growth of KV caches forces LLM serving systems to evict previously stored caches, which in turn introduces significant recomputation overhead whenever the evicted caches are required again. 
Moreover, prefix caching is inherently infeasible across different models, forcing each model to recompute KV cache for the identical prompt, which leads to signficant overhead. To alleviate these issues, we propose Identical Cache Reuse (ICaRus), a novel architecture that allows multiple models to share identical KV caches across all layers. ICaRus is based on the key observation that a decoder-only Transformer can be conceptually decomposed into a logical encoder, which generates KV caches, and a logical decoder, which predicts output tokens from the KV caches. ICaRus fine-tunes only the logical decoder while freezing the logical encoder, enabling multiple models to share an identical KV cache. This eliminates cache memory explosion and unexpected evictions while also allowing cross-model reuse of KV caches for new input tokens, thereby removing redundant recomputation in multi model inference achieving both efficiency and scalability. Moreover, by incorporating lightweight adapters such as LoRA, ICaRus parallelizes KV cache generation and next-token prediction during decoding. ICaRus achieves comparable accuracy to task-specific fine-tuned model across a diverse set of tasks, while allowing multiple specialized models to fully share KV caches. ICaRus achieves up to $11.1\times$ lower P95 latency and $3.8\times$ higher throughput in multi agent scenarios with 8 different models, compared to prior multi model system.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces ICaRus, a novel architecture designed to optimize multi-model inference by enabling full Key-Value (KV) cache sharing across decoder-only Transformer models. The core innovation lies in conceptually decomposing a Transformer into a logical encoder (which generates KV caches) and a logical decoder (which predicts output tokens). By freezing the encoder and fine-tuning only the decoder using lightweight adapters (e.g., LoRA), ICaRus allows multiple specialized models to reuse identical KV caches, significantly reducing memory usage and recomputation overhead. The authors demonstrate that ICaRus achieves comparable accuracy to task-specific fine-tuned models while delivering up to 11.1× lower P95 latency and 3.8× higher throughput in multi-agent workflows.

### Strengths
* The logical encoder-decoder decomposition is an interesting and insightful abstraction that enables cross-model KV cache reuse, a previously unsolved challenge.  
* ICaRus shows substantial improvements in latency and throughput, especially in multi-agent workflows like ReAct and Reflexion, while surprisingly matches or even exceeds the performance of fully fine-tuned models in several benchmarks.

### Weaknesses
* Limited scope of sharing. The proposed approach assumes problems are solved with multiple task-specialized models that **have different weights** and **are finetuned from the same base model**, such an assumption limits its applications. Is such a setting common? To my best knowledge, multi-agent systems typically use the same model for all roles, with only the system prompts customized for each.

* Synthetic evaluation setup. The experimental setting for multi-model inference (Lines 318–320) relies on a synthesized workload that routes multi-turn requests in a round-robin fashion to different models. While this setup enables controlled benchmarking, it may not accurately reflect the dynamic and heterogeneous nature of real-world multi-agent systems, where agent interactions are often asynchronous, task-dependent, and influenced by external stimuli. This raises concerns about the generalizability of the reported latency and throughput improvements.

### Questions
1. In Figure 3, does the decode phase update the **shared** KV Cache? If yes, does another role-specific decoder directly reuse the updated KV or it always requires the logical encoder (Figure 3(a)) to recompute the KV cache?

2. Is ICaRus coupled with LoRA finetuning? Or it is compatible with other finetuning approaches (e.g., full parameter finetuning or prefix tuning)? Which adaptations are needed for other finetuning approaches?

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
3

### Summary
ICaRus introduces “Identical Cache Reuse” to let multiple task-specialized LLMs share the same KV cache for identical prompts, avoiding redundant memory use and recomputation in multi-model inference. It splits a decoder-only Transformer into a frozen logical encoder (that builds KV caches) and a fine-tuned logical decoder (that predicts tokens), so all models reuse one shared cache. Using lightweight adapters (e.g., LoRA), ICaRus achieves similar accuracy to task-specific fine-tuning but cuts P95 latency by up to 11× and boosts throughput up to 3.8× in multi-agent workflows.

### Strengths
1. Innovative idea that changes how multi-model share KV cache.
2. Demonstrates good potential system improvement over previous methods by the new design.
3. Promising path to efficient multi-model inference.

### Weaknesses
1. Unclear accuracy comparison to workflows without any sharing or naive baselines. How does a base model perform if I do not use the new workflow?
2. Training seems to be more complex. How robust is this training method? It will be more convincing to do more experiments on bigger models and more datasets.

### Questions
1. How does the work deal with cases when agent finishes prefill and does some decoding? How to reuse the decode token for that part? Or how does system perform if that part is not used? Especially in long decoding scenarios like coding.

2. How does the accuracy compare with not sharing at all? A simpler workflow without multi-model inference. Not sure if the dataset needs multi-model inference.

### Soundness
2

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
4

### Summary
* The paper proposes a novel method to reuse KV Caches across different models to reduce redundant prefill operations. The method achieves up to 11.1× lower P95 latency and 3.8× higher throughput in agentic workflow across 8 different models.

### Strengths
* The method decouples the decoder-only model into a logical encoder and a token predictor, and freezes the encoder for fine-tuning. This makes the fine-tuning faster and more memory efficient. All LLMs also share the same logical encoder.
* Instead of fine-tuning the entire decoder, lightweight adapters are used for fine-tuning for more efficiency to reduce the overhead of multiple KV accesses.

### Weaknesses
* The paper mentions DroidSpeak as related work, which also performs sharing of the KV Cache across multiple LLMs, but points out that DroidSpeak requires some re-computation of sensitive layers. It would be useful to see a more direct comparison with DroidSpeak as a baseline and compare the cost of fine-tuning adapters in ICaRus vs selective re-computation in DroidSpeak.
* The evaluation only considers a round-robin routing scheme, which deliberately increases the number of models being used in a session. It would be useful to see the evaluation under prefix-aware or KV-aware routing. Do the gains still hold in terms of the latency and throughput if reuse across models is not required?

### Questions
* In the round-robin evaluation setup, how many different models are used? I missed this detail and would encourage the authors to clarify.
* In Figure 2, please clarify the insight as the training loss curves look almost identical and I am missing the takeaway.

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
2

### Summary
The core idea of the work is to attempt to re-use the KV for the prefill phase, compared to the traditional LoRA based approach.The work instead finetunes a decoder style model to handle the sub tasks, using a light weight adapter.

### Strengths
1. I think this is an interesting twist on traditional LoRA by choosing to change just the decoding side. 
2. provides a 11x decent speedup over the existing gpu cache system

### Weaknesses
1. I feel like the optimization provided by Icarus, while very impressive, feels a bit too incremental on existing LoRA systems.

2. From Appendix A2.2, the experiments tested the case where each agent has it's own LoRA adapter. Due to arrival patterns and batching, It is unclear the amount of LoRA used per GPU might not be true in practice. 

3. The main alternatives the paper lists are either recompute or save. It is possible that swap is a valid strategy.

### Questions
1. was swap space turned on for the experiments? Can all the KV cache be evicted/loaded back instead of recomputation?

### Soundness
3

### Presentation
4

### Contribution
3
