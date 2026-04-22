# Mixture of Thoughts: Learning to Aggregate What Experts Think, Not Just What They Say

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 6, 4, 4

## Abstract
Open-source Large Language Models (LLMs) increasingly specialize by domain (e.g., math, code, general reasoning), motivating systems that leverage complementary strengths across models. Prior multi-LLM approaches either (i) route a query to one or a few experts and generate independently, (ii) aggregate outputs from each model via costly multi-turn exchanges, or (iii) fuse weights into a single model—typically requiring architectural homogeneity. We introduce Mixture of Thoughts (MoT), a simple method for latent-level collaboration among heterogeneous experts under a global routing scheme. For each query, a lightweight router selects top-$K$ experts and designates a primary expert; uniformly placed interaction layers project hidden states into a shared latent space where the primary expert performs cross-attention over its active (selected) peers. Pre-trained experts remain frozen; only the router and the lightweight interaction layers are trained with a novel joint training objective that improves both the expert selection and inter-expert collaboration. Across five in-distribution (ID) and three out-of-distribution (OOD) benchmarks, MoT surpasses the current routing and aggregation-based state-of-the-art, Avengers, by $+0.38\%$ and $+2.92\%$, respectively. Further, MoT significantly outperforms the best-performing single model. It achieves this with single-pass inference, runtime comparable to routing baselines, and none of the overheads of iterative aggregation. MoT offers a simple latent-space mechanism for combining heterogeneous LLMs, a practical step toward broader multi-LLM collaboration. Our code is publicly available at \url{https://anonymous.4open.science/r/mot-5B4B/}.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
Authors propose a method to aggregate pretrained LLM decoders, Mixture of Thoughts (MoT). The idea is to let a primary expert “look at” the hidden states of other active experts and use that information thought cross attention. For each query, a router selects the top-K experts and designates a primary expert. The authors insert interaction layers that project each model’s hidden states into a shared latent space; the primary then performs cross-attention over the selected peers’ latents. All experts are frozen; only the router and interaction layers are trained with a joint objective. 

While I like the idea of “merging” pertained models, there are several drawbacks in the proposed implementation as well as in the evaluation setup (see Weaknesses). I think that the marginal improvements on downstream tasks do not justify the overhead and complexity of the proposed approach.

### Strengths
- The paper introduces a new way to enable communication between pretrained LLM decoders by exchanging latent representations rather than final outputs. This latent-level interaction is conceptually interesting and bridges ideas from model ensembles and mixture-of-experts architectures. 
- It is well written and easy to follow.

### Weaknesses
The main drawback is that injecting interaction layers makes inference synchronous across experts. One of the main benefits of using a pool of models is the ability to aggregate their results asynchronously, but this design removes that advantage.

Let us consider two cases:
1. On device inference with limited memory (we can fit only one model)
In prior works such as RouteLLM and Averagers, we can process the prompt, select experts, and then sequentially: load an expert, perform the computation, save the output, and repeat this process k times. In the proposed approach, however, this becomes problematic because intermediate hidden states must be stored to allow the primary expert to attend to them later. This introduces additional memory reads and writes, which can further stress an already memory-bound inference setup.

2. All experts placed on different devices:
In this scenario, interaction layers require a gather operation by the primary expert. While the exchanged hidden states are small and the bandwidth cost is minimal (even over Ethernet I suppose), this design introduces a synchronization point. In heterogeneous setups, where devices or models have varying latencies, these synchronization points could accumulate and lead to noticeable overhead.

I would appreciate seeing a table similar to Table 10 in the Appendix, but focusing on overhead rather than parameter counts. In particular, it would be helpful to include:

- communication overhead when models are placed on different devices with different type of interconnect
- memory overhead
- latency for generation ( authors provide average time across downstream tasks but I would be more interested in latency for prompts of different length)

A table summarizing these measurements would make it much easier to understand the trade-offs.
If the authors can explicitly demonstrate that generation speed is not hindered by the additional communication and provide a detailed analysis of memory and bandwidth requirements (see Questions), the proposed method would be significantly more convincing.

### Questions
Inference is reported to be conducted on an RTX A6000 GPU with 48 GB of memory. The authors state that $k = 3$, which implies that three $7$B models (approximately $21$B active parameters) are used simultaneously. Assuming bfloat16 precision, the model weights alone would require roughly 42GB of memory, leaving very limited space for the rest (KV caches, router, etc). This suggests that inference must have been distributed across multiple devices or involved CPU offloading.
Could the authors please clarify the inference setup, specifically:
- whether experts parallelism was used,
- whether any activation checkpointing, quantization, or CPU offloading was employed 

to fit the models into 48 GB?

### Soundness
2

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
The authors propose Mixture of Thoughts (MOT), a method for latent-level collaboration among heterogeneous experts using a global router. For each query, a lightweight router selects the top-K experts and a primary expert. Interaction layers with adapters and cross-attention are added to each expert, allowing their hidden states to be integrated into the primary expert, while keeping the expert backbones frozen. A joint training objective optimizes both the router and interaction layers.

### Strengths
1. The paper is clearly written and easy to follow.
2. The use of both ID and OOD datasets effectively demonstrates the method's generalizability.
3. An expert loss experiment provides evidence for the method's robustness.

### Weaknesses
1. The experts chosen (from Llama and Mistral families) seem similar. It would be insightful to test experts from more diverse families to assess generalization.
2. In "Removing Primary Expert Cross-Attention," comparing against the optimal performance of all experts with LoRA, rather than just removing cross-attention, could better highlight its impact.
3. The model without auxiliary loss (52.42%) performs lower than ensembles/routers (53-60%). This suggests auxiliary loss is important. Could other ensembles/routers improve with it?

### Questions
1. I'm curious about how the choice of the Top-k parameter influences the accuracy of the proposed method. Could varying Top-k lead to noticeable changes in results, and what trade-offs should practitioners be aware of?
2. The standard deviation is not reported in this paper.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes Mixture of Thoughts (MoT), a novel and efficient framework for collaboratively leveraging multiple specialized open-source large language models (LLMs). Unlike prior approaches that rely on routing, iterative aggregation, or model merging, MoT introduces lightweight routers and cross-attention-based interaction layers in the latent space, enabling selected expert models to exchange information during a single forward pass. With frozen pre-trained experts and only the router and interaction layers trained, MoT achieves state-of-the-art performance across in-distribution and out-of-distribution benchmarks.

### Strengths
1. Innovative Methodology. The paper proposes the Mixture of Thoughts (MoT) framework, introducing a latent-level collaboration mechanism in multi-LLM systems. It enables information fusion across experts through a lightweight interaction layer, serving as a significant extension to existing routing or ensemble approaches.
2. Efficiency and Performance Balance. It achieves state-of-the-art performance with single-pass inference, runtime comparable to routing-based baselines, and avoids the overhead associated with iterative aggregation.
3. Comprehensive Experimental Design. The method is evaluated on five in-domain tasks such as MMLU and GSM8K, and three out-of-domain tasks including MBPP and C-Eval, with comparisons against strong baselines like RouterDC, Avengers, and ZOOTER.
4. Thorough Empirical Analysis. The paper presents ablation studies on the number of interaction layers, insertion positions, number of experts, and expert dropout, demonstrating a degree of robustness.

### Weaknesses
1. Limited Gains in In-domain Settings. MoT is supervised trained on in-domain data using a joint loss to optimize both the router and interaction layers, while the baseline Avengers requires no training. However, the absolute improvements over Avengers are only +0.23% on in-domain tasks and +1.36% on out-of-domain tasks, which appear marginal given the additional training cost.
2. Lack of Scalability Analysis. The paper reports that generation time is comparable to Avengers, but MoT requires simultaneous loading of k LLMs, and decoding involves inter-model communication and memory overhead. Avengers, on the other hand, supports fully parallel inference without communication. The paper does not verify whether MoT maintains linear scalability under common multi-GPU inference setups, where communication cost could become a bottleneck.
3. Weak Theoretical Justification. The paper lacks theoretical analysis explaining why interaction in the latent space leads to performance gains.

### Questions
1. Scalability Issue. Although the authors claim that generation time is comparable to Avengers, MoT involves communication of hidden states between experts, whereas Avengers allows for full parallelization without inter-model communication. When using large models that require multi-GPU setups, communication costs may significantly hinder MoT’s efficiency. The authors should provide a detailed analysis of communication, memory, and other scalability-related overhead to validate MoT’s efficiency in multi-GPU settings.
2. Is the Performance Gain Worth the Training Cost? MoT is supervised trained on in-domain data, while Avengers achieves competitive performance without any training. Given the training overhead and implementation complexity of MoT, do the modest performance improvements justify the cost?

### Soundness
2

### Presentation
2

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
This paper introduces Mixture of Thoughts (MoT), a framework to drive multiple pre-trained LLMs to perform effective collaboration. The core idea is to aggregate their internal "thought" representations through by adding some adapter blocks interleaved with the original transformer blocks. Besides, the paper also introduce a lightweight and learnable router to select a subset of experts for an input query. During inference, the router select a subset of experts with a primary expert, which integrates the representations of other experts and produces the final response. This work conducts experiments on five benchmarks, include both in-distribution and out-of-distribution setups. The results demonstrate that the proposed framework outperforms router-based baselines. Besides, ablation studies validat the key design choices of the MoT architecture.

### Strengths
- The central concept of aggregating latent "thoughts" rather than just outcomes is an intuitive research area of multi-agent systems. It addresses the limitation of cross-model communication, where models/agents typically communicate with text rather than some latent representations.
- The proposed framework consistently surpasses strong router-like baselines on a diverse set of benchmarks, for both in-domain and out-of-domain setups. 
- Ablation studies systematically validate the contributions of key architectural components such as the number of interaction-layer number of depth. Ablation studies support the architectural designs of the proposed framework.

### Weaknesses
- Limited Scope of Baselines: The paper's abstract and related work section astutely categorize prior multi-LLM approaches into three paradigms: (i) routing, (ii) response-level collaboration and (iii) parameter fusion. However, the selected baselines for comparison are predominantly router-based or simple output-ensembling methods (e.g., Voting, RouterDC).  The experimental evaluation would be significantly more comprehensive if it included representative baselines from the other two categories.
- Limited Validation on Thinking-Centric Tasks: The title, "Mixture of Thoughts," sets a high expectation for deep collaboration on complex reasoning. While the chosen benchmarks like GSM8K are valuable, the evaluation would be more convincing if it included tasks that require longer chains of thought (e.g., AIME).
- Lack of Qualitative Analysis on Collaboration Dynamics: Since the paper seems to focus on aggregate the "thoughts", the paper would be strengthened by an analysis of the tought aggregation behaviors usch as cross-expert attention patterns. Such an analysis could provide direct evidence of meaningful collaboration, for instance, by showing that a primary expert dynamically attends to a math-specialized expert's hidden states when solving a math problem.
- Clarity on Training/Inference Implementation: The precise mechanics of the token-by-token generation for non-primary experts could be more explicit in the main text to improve clarity. Could the authors clarify what inputs each expert receives during training? Are all experts given the same input sentences? My current understanding is that the inputs are identical during training but differ during inference—if so, does this create a mismatch between training and inference?

### Questions
- The reported latency for the "Voting" baseline is much higher than one would expect from a single, parallelized forward pass, where latency should be bounded by the slowest expert. Does this implementation involve generating multiple candidate samples from each expert and then voting on the final answers?

- To better isolate the benefits of MoT's heterogeneous collaboration from the general benefits of ensembling, have you considered a baseline where the single strongest expert uses self-consistency by voting on its own multiple sampled results, using a comparable computational budget to MoT?

### Soundness
2

### Presentation
2

### Contribution
3
