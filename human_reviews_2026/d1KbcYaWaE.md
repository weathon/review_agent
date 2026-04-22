# Coupling Attention and Memory: A Dynamic Memory Module for Efficient Adapation with Pretrained LLMs

- Avg Score: 2.67
- Decision: Reject
- Scores: 2, 4, 2

## Abstract
Pretrained large language models (LLMs) are highly capable but still require adaptation for various domains. Existing fine-tuning strategies typically assume either access to all target task data *simultaneously* (e.g., multi-task learning), or a sequential data stream, as in continual learning, where the former tackles the simultaneous task interference issue while the latter focuses on addressing the catastrophic forgetting problem. In this work, we propose a unified approach to address both scenarios. We present DynMem, a unified framework that tackles both settings with a lightweight dynamic memory module built on top of frozen pretrained LLMs. DynMem encodes past examples into a fixed-sized memory bank. We design a novel dynamic update mechanism where new examples and existing memory entries are ranked based on their *accumulated* attention scores, and the lowest-ranked examples are thus pruned to maintain size. To further reduce recency bias, we adopt a new bi-level memory design:  $\mathrm{L_1}$ Memory is actively used by the backbone LLM, while $\mathrm{L_2}$ Memory stores more diverse examples for improved effectiveness at minimal cost. The design also supports more flexible test-time scaling by allowing larger memory banks. We evaluate DynMem under both simultaneous and continual learning settings. Our method consistently outperforms state-of-the-art baselines tailored for each scenario, demonstrating its great potential in mitigating task inference for both simultaneous and sequential learning. In particular, DynMem outperforms the state-of-the-art method in simultaneous adaptation across different models, yet achieves this with approximately 50\% fewer trainable parameters.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a novel framework, DynMem, designed to address both the interference issue in multi-task learning and the catastrophic forgetting problem in continual learning. This framework introduces a bi-level memory module (i.e., L1 for short-term, L2 for long-term) that continuously re-ranks its entries based on attention scores. Through this dynamic memory mechanism, DynMem enables parameter-efficient adaptation and inference.

### Strengths
- This paper is easy to follow.

- Rather than focusing on a single challenge, the authors design a unified and well-structured module that jointly addresses both catastrophic forgetting and interference issue.

- The proposed framework is highly parameter-efficient, as it freezes the backbone LLM and trains only a very small set of additional parameters.

- The experimental setup is rigorous, carefully considering domain shift through diverse evaluation streams.

### Weaknesses
W1. Unclear theoretical justification for some of its key design choices: 
- While the bi-level design memories serves as the structural core of the framework, the current discussion focuses mainly on implementation and empirical validation. A more explicit conceptual rationale for clarifying the underlying principles that this hierarchical organization contributes to tackling the challenges of continual and multi-task learning would make the architectural choice more persuasive and theoretically grounded.

- Although the memory entries are continuously re-ranked based on attention scores, there is no strong evidence that this ranking remains relevant to the current input. It is unclear whether, at the point of significant domain shift, the stored memory might even introduce negative interference rather than helping adaptation.

- While the ablation study highlights the importance of the gated fusion mechanism, the paper does not provide analytical evidence showing that the gating coefficient effectively controls the strength of information injection as claimed.


W2. Hyperparameter Sensitivity:
- One of the core design factors in this work is the periodic memory update mechanism. However, this frequency is likely to be strongly correlated with the dataset size. For example, when training set involves a collection of smaller datasets, a faster update interval would likely be more beneficial, preventing inefficient or outdated entries from persisting in memory. Therefore, analyzing this relationship between dataset characteristics and update frequency is crucial for practical deployment, yet the paper lacks a detailed discussion or empirical study on this sensitivity.

W3. Asymmetric methodological experimental settings:
- The training dataset consists of 1,000 examples, which coincides with the sizes of the L1 and L2 memories, making the experimental setup overly favorable to DynMem. In contrast, other methods such as EMAR are evaluated under asymmetric conditions, for instance by setting the buffer size to 10.


W4. The key limitations identified about previous works — (1) the need to know which adapter to use for each task, and (2) the lack of transferability between adapters — have largely been addressed by methods based on the Mixture of Experts (MoE) framework [1,2]. You should reference these works.
The paper provides little discussion of MoE approaches and lacks a direct comparison with them. Moreover, the baselines used for continual learning tasks are mostly from 2022–2023, which makes them somewhat outdated compared to these references.
[1] Xu Owen He, Mixture of A Million Experts, 2024
[2] Jiazuo Yu, et al., Boosting Continual Learning of Vision-Language Models via Mixture-of-Experts Adapters, 2024


W5. $W$ is sensitive to the order of $M$, but since the order of memory matrix $M$ changes after each update, $W$ must be relearned from scratch every time. In contrast, adapter parameters can be reused without additional training, for a task once trained. In real-world continual scenarios with many recurring tasks, adapters might be more training-efficient.


W6. $M$ is composed of QA pairs, with each row containing one QA pair. Therefore, $M$ selectively references specific samples from the QA dataset, which can make it sample-efficient.
Adapters, on the other hand, usually capture the average distribution over the entire QA dataset, and compress them in the parameters of adapter . So it might not be sample-efficient, but might more precisely comprehend the task.
From this analysis, there comes several concerns.
- DynMem can likely achieve better performance with short training. However, with longer training, adapter-based models might perform better. It would be helpful if Table 1 included results for adapters trained over multiple epochs. (Of course, DynMem’s fast adaptation remains an advantage.)
- Using $M$ may be sensitive to the selected samples. So, when the dataset is highly diverse, a few stored samples may not adequately represent the overall distribution. Benchmarks like PIQA have relatively consistent QA formats, so this approach can work well there. However, for natural, open-ended QA tasks with more diverse distribution (such as ELI5), A few selected samples captured in $M$ would not be sufficient to cover the task.

### Questions
- It would be interesting to include an additional study on the effect of memory update frequency during training.

- It would be helpful to include an algorithm or example illustrating the full training and memory update process for clarity.

-  It would be beneficial to include additional experiments comparing performance when the dataset size exceeds the capacities of the L1 and L2 memories, to assess whether the proposed method remains effective under such conditions.

- Backward transfer is measured as $\sum_{k=1}^{k-1} A_{i,k}$, which means “train on $k-1$ (or <k-1) and test on $k$. This does not seem to capture forgetting, rather it measures how well the model handles unseen task. More right form seems to be $A_{k,i}$.

### Soundness
2

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
This paper introduces DynMem, a bi-level dynamic memory module designed to improve large language model (LLM) adaptation under both continual and simultaneous learning settings. Unlike conventional parameter-efficient fine-tuning (PEFT) methods that either overwrite task-specific adapters or require separate modules for each task, DynMem combines short-term (L1) and long-term (L2) memory banks, effectively balancing knowledge retention and forward transfer. Experimental results across various adaptation benchmarks demonstrate that the proposed method significantly outperforms existing PEFT and continual learning approaches, achieving highly competitive performance.

### Strengths
1. The paper addresses a significant gap by proposing a single framework capable of handling both continual and multi-task adaptation settings, whereas prior PEFT or CL approaches generally support only one paradigm.
2. The L1/L2 memory structure successfully separates short-term relevance from long-term knowledge diversity, leading to strong backward and forward transfer improvements.
3. DynMem consistently demonstrates superior performance across continual, single-task, and multi-task benchmarks.

### Weaknesses
1. The paper lacks a theoretical justification for how the dual-memory structure ensures desirable long-term memory composition while avoiding memory bias accumulation.
2. Despite using randomized task orders, the evaluation still assumes explicit task identities, leaving the method unverified in task-free or real-world continual learning scenarios.
3. The experiments are mainly limited to reasoning-focused NLP benchmarks, making it unclear whether the method generalizes to broader LLM adaptation tasks; more diverse evaluations are needed to demonstrate general applicability.

### Questions
1. Does the paper provide a clear theoretical justification for adopting attention-based retrieval and memory update over alternative feature alignment modeling techniques?
2. When memory size grows large, is there a risk of negative interference or noisy retrieval that may degrade performance?
3. Since memory entries are created from final-layer token representations only, does the model risk losing important contextual or structural information required for accurate retrieval and adaptation?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
Pretrained LLMs still need adaptation. Existing approaches either assume simultaneous access to all task data or a sequential stream. The paper argues these worlds are siloed and seeks a single method that works well for both. A bi‑level memory module (L1 active, L2 reservoir) is attached to a pretrained LLM at the last decoder layer. During training, the model cross‑attends to L1 and fuses with a learned gate; attention scores accumulate into utilities that rank entries for periodic pruning and promotion between L1/L2. In inference, the model performs global retrieval over L1 and L2 to get a small, input‑specific set of top‑K memories, giving scalability.

### Strengths
1. The paper is consistent and well-structured, making it easy to read.
2. The problem of controlling an LLM's memory in continual scenarios is important.

### Weaknesses
1. Methods such as SAPT (Zhao et al., 2024) already offer task-agnostic inference methods, and no comparison was made between the proposed method and this approach.
2. Novelty is incremental relative to prior retrieval‑ and memory‑augmented transformers (RETRO, Memformer, Memorizing Transformers; hierarchical memory; lifelong LM).
3. There are methodological ambiguities, most notably how test‑time queries are formed when memory keys are generated from $(x, y)$.
4. The metrics are standard but could be expanded (AULC, multi‑task synergy), fairness would benefit from matched parameter budgets and stronger baselines, and latency and runtime reporting is currently missing.
5. Abstract claims ≈50% fewer trainable parameters. Yet, Table 3 shows DenseLoRA at 0.06% vs DynMem 0.18%. The statement is, at best, conditional and not universally supported.

### Questions
See above.

### Soundness
2

### Presentation
2

### Contribution
1
