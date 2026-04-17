# Less Gradient, More Speed: Rethinking Pipeline Parallelism for Efficient Fine-Tuning with FluidPipe

- Decision: Reject
- Scores: 2, 2, 8

## Abstract
Fine-tuning large pretrained models often uses pipeline parallelism (PP) to split layers across devices. PP is simple to deploy but requires per-iteration cross-stage gradient exchanges, creating pipeline bubbles and making performance highly sensitive to latency. We introduce FluidPipe, a two-stage pipeline design that replaces these gradient exchanges with local updates guided by an auxiliary head and cross-stage bi-directional distillation. This re-design eliminates iteration-time synchronization while preserving model quality. We develop a cost and communication model explaining when FluidPipe outperforms PP, and validate on BERT-Large and ViT-Large fine-tuning, where FluidPipe achieves up to $3.3\times$ faster training while matching or improving accuracy.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a pipeline-parallelism strategy that approximates exact pipeline training. It incorporates bidirectional knowledge distillation between pipeline stages 1 and 2 and introduces a local loss within each stage to reduce inter-stage communication.

For knowledge distillation, the paper uses two loss terms. For stage 1, it uses the stage-2 logits computed on the same training sample in the previous epoch. These stale stage-2 logits serve as soft labels, and a KL-divergence loss is computed between them and the current stage-1 logits. For stage 2, the logits from stage 1 are forwarded to stage 2, and a KL-divergence loss is computed between the forwarded logits and the logits produced by stage 2. Thus, knowledge distillation between stages 1 and 2 is bidirectional.

For the local loss, blocks in stage 1 are updated using a loss computed between the output of the first block and the ground-truth label. This loss can be computed without waiting for stage 2 to finish its forward pass on the same sample or for gradients to arrive from stage 2. Consequently, the peer-to-peer communication volume between stages due to these gradients can be eliminated.

The method is evaluated primarily for speedup on ViT-Large and BERT-Large. Results show that it achieves speedups up to 3.3x while attaining accuracy comparable to exact pipeline training. Experiments are conducted on two GPUs across two servers for BERT-Large, and on four GPUs across two servers (two GPUs per server) for ViT-Large.

### Strengths
- This paper targets the important problem of reducing transformer training time—a widely used model family—and focuses on pipeline parallelism, which has been studied for several years and is commonly adopted for large transformer models.

- The paper proposes a novel method that omits peer-to-peer communication during backpropagation between pipeline stages by using knowledge distillation and a local-loss update.

- The writing is clear, easy to follow, and understandable.

- The figures are very helpful for understanding the proposed method and the timing of each loss term.

### Weaknesses
Although the paper proposes a novel scheme, I have several concerns about the experimental setup; the experiments are neither thorough nor sufficiently transparent.


- While I understand that not everyone has access to a large number of GPUs, the total number used here (2 for BERT-Large and 4 for ViT) is too small to convincingly demonstrate the method’s effectiveness. With so few GPUs, it is impossible to assess whether the method scales as the number of GPUs increases or how it behaves when combined with other parallelism strategies such as tensor parallelism and data parallelism.

- The paper defers too many important research questions to future work. In particular, evaluating how the method interacts with advanced pipeline schedules such as 1F1B (as in PipeDream) is important because these schemes are widely used. It is therefore essential to test whether the method still achieves speedups when such schedules reduce pipeline bubbles.

- Some experimental settings are not clearly stated. For example, the inter-node bandwidth between servers, the rank used for the low-rank approximation in LoRA, and whether mixed-precision training or FP32 was used are not specified.

- Certain experimental choices appear inconsistent and need justification. Why are only two GPUs used for BERT-Large and four GPUs (across two servers) for ViT? In the 4-GPU ViT setup, the authors use the PP baseline for intra-node scheduling and their proposed approach for inter-node parallelism when parallelizing the model to evaluate the method’s effectiveness. This design choice needs justification, as tensor parallelism is commonly employed within a server. I am also curious whether the proposed method achieves speedups compared to a single server equipped with all GPUs (e.g., one server with two GPUs using tensor parallelism for BERT-Large, and one server with four GPUs using tensor parallelism for ViT).

- I also have concerns about the fairness of the experiments in Table 3. The authors compare the runtime of their method on multiple GPUs to LoRA on a single GPU. For a fair comparison, the LoRA setup should also use multiple GPUs with tensor parallelism.

- The paper does not include a breakdown showing the proportion of time spent on computation versus communication. A bar chart that breaks down runtime and demonstrates that the proposed method reduces peer-to-peer communication time would help substantiate the claim that the speedup comes from the proposed technique.

### Questions
- How does the method interact with advanced pipeline schedules (e.g., 1F1B/PipeDream)? Does it still achieve speedups when pipeline bubbles are reduced?

- What are the exact experimental settings—inter-node bandwidth, LoRA rank, and precision mode (mixed precision vs. FP32)?

- Why are only two GPUs used for BERT-Large and four GPUs (across two servers) for ViT?

- In the 4-GPU ViT setup, why choose PP for intra-node and the proposed method for inter-node parallelism? Given that tensor parallelism is commonly used within a server, why not employ TP intra-node?

- Can you clarify the fairness of the Table 3 comparison? Why is your method evaluated on multiple GPUs while LoRA is on a single GPU, and can you provide results where LoRA also uses multiple GPUs with tensor parallelism?

- How does the method compare against a single-server baseline with all GPUs using tensor parallelism (e.g., 2-GPU TP for BERT-Large; 4-GPU TP for ViT)?

- Can you provide a runtime breakdown (compute vs. communication) to show where the speedup comes from?

- Regarding the soft labels passed from stage 2 to stage 1: since they are produced in the previous epoch, staleness may accumulate due to intervening weight updates. Would reducing this staleness—e.g., by replaying the same mini-batch consecutively within an epoch rather than reusing it in the next epoch—improve accuracy?

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
1. **Core Contribution**
This paper introduces **FluidPipe (FP)**, a novel two-stage pipeline design aimed at solving the inefficiencies of standard **Pipeline Parallelism (PP)** in large-model fine-tuning. Standard PP is highly sensitive to network latency and suffers from "pipeline bubbles" because it requires per-iteration gradient exchanges between stages.
FluidPipe addresses this by:
    - **Eliminating Gradient Synchronization**: Instead of exchanging gradients, it adds an auxiliary head to Stage 1, allowing it to perform its own local updates.
    - **Introducing Bi-Directional Distillation**: Cross-stage feedback is replaced with low-frequency, bi-directional logit distillation (S2 $\to$ S1: once per epoch / S1 $\to$ S2: once per iteration).
2. **Experimental Claims**
The authors claim that in experiments with ViT-Large and BERT-Large, FluidPipe achieves up to a 3.3x training speedup over traditional PP. This effect is reportedly most pronounced in high-latency environments (e.g., 25ms). Furthermore, they claim this speedup is achieved while maintaining or even slightly improving model accuracy compared to PP.

### Strengths
- **Training Speedup (Especially in High-Latency Environments)**
    - The paper's strongest contribution is its effective elimination of "pipeline bubbles," the chronic bottleneck of traditional Pipeline Parallelism (PP).
        - **Elimination of Gradient Synchronization:** FluidPipe avoids per-iteration gradient exchanges by equipping Stage 1 with an auxiliary head to perform local updates.
        - **Minimized Communication Frequency:** Cross-stage feedback is replaced by a once-per-epoch bulk transfer (S2 $\to$ S1) and a per-iteration, one-way transfer (S1 $\to$ S2).
- **Maintained or Improved Model Accuracy**
    - The method doesn't just increase speed; it maintains or even improves the final quality (accuracy) of the training.
        - **Quality Preservation:** FluidPipe performs the same full-model fine-tuning as PP, yet it achieves comparable or better accuracy on most tasks despite the altered training dynamics.
        - **Regularization Effect:** The auxiliary head on Stage 1 and the bi-directional distillation can act as a form of regularization, potentially helping the model converge to a local minimum with better generalization performance.

### Weaknesses
- **Fundamental Scalability and Memory Flaw:** The design is restricted to two stages, with deeper pipelines deferred to “future work,” but this sidesteps a core limitation. Scaling to $N>2$ stages would likely require **each intermediate stage** to cache **per-sample logits** to support bi-directional distillation, creating a cascading storage and communication burden that is never quantified. Even in the 2-stage case, Stage-2 maintains a per-epoch dictionary $\mathcal{P}*2$ of logits—an **unstated but material memory overhead**.
This cost is muted on small-class classification (e.g., $C \le 100$), but would become **prohibitive for token-level LM tasks**, where caching scales as $O(N \times T \times V)$ with the number of samples $N$, sequence length $T$, and vocabulary size $V$. The paper briefly gestures at “streaming” as a mitigation yet provides **no quantitative memory/latency analysis, ablation, or implementation evidence**. This omission is at odds with PP’s primary purpose—**alleviating memory pressure**, not shifting it elsewhere.
- **Unstable Training & Heuristic Design (insufficiently stress-tested):** The method shows high sensitivity to the **split point**, **distillation weights** ($\alpha_1,\alpha_2$), and the **Extra Block**. As reported (Table 6), the mere choice of split can drive accuracy from 92.14% down to 86.63%. The Extra Block frequently “rescues” failing splits, suggesting it works as a **heuristic patch** for a deeper mismatch: features needed for Stage-2 continuation vs. features that are linearly separable for Stage-1’s auxiliary head. The paper does not investigate principled alternatives (e.g., **stop-gradient/detach on the auxiliary branch**, **gradient-surgery or orthogonality regularizers between heads**, or **capacity sweeps** for the auxiliary head). Without such analyses, the training **fragility appears structural**, not incidental.
- **Missing Baselines, Conditional Speedups, and Limited Scale:**
    - **Scheduling baselines absent.** Although the aim is to reduce bubble-induced stalls, there is **no experimental comparison** against **PipeDream, Zero-Bubble PP, BitPipe**, etc., under identical hardware and RTT (Round-Trip Time). For practitioners, these are the **direct alternatives**; omitting them leaves the throughput claims under-substantiated.
    - **2-GPU setup lacks DP/TP baselines.** On BERT-Large (fits in memory), a **2-way Data Parallel (and/or ZeRO-DP) and 2-way Tensor Parallel** baseline is feasible and standard; its absence makes the 2-GPU results hard to interpret as evidence for replacing PP.
    - **Speedups are highly conditional.** FluidPipe still transmits **forward activations** hhh each step ($Stage-1 \to Stage-2$); it only removes **per-step gradient returns**. Thus gains depend on the **ratio of gradient-return volume to activation volume** and the **network RTT**. When activations dominate (e.g., **longer sequences, larger hidden states**) the advantage can shrink; the paper lacks a **cost model + measurement** tying speedups to ($\text{activation size}, \text{batch size}, \text{RTT}$).
    - **Scale and task mismatch.** The motivation centers on very large models (GPT-3/PaLM-class) and sequence tasks, yet experiments remain on **BERT-L / ViT-L** and small-class classification. There is **no evidence** that the method preserves accuracy or yields net gains under **LLM-scale, token-level workloads** where the proposed memory/communication trade-offs are most stressed.

### Questions
- **Scheduling baselines.** Why are key pipeline-scheduling baselines (e.g., GPipe, PipeDream, Zero-Bubble PP/BitPipe) absent under identical hardware and RTT?
- **2-GPU baselines.** In the BERT-Large 2-GPU setting, how does FluidPipe compare (accuracy + wall-clock) to **DP (with ZeRO-1/2)** and **2-way Tensor Parallel**? If memory pressure is the focus, please also include **ZeRO-3**.
- **Scaling to LLMs & LM tasks.** Can you report results on **larger LLMs** and **token-level language modeling tasks**, including accuracy retention and end-to-end time, where distillation storage/communication is most stressed?
- **Memory/communication accounting.** Please provide a quantitative analysis for the per-epoch logits cache $\mathcal{P}_2$ (and its multi-stage generalization) and an empirical evaluation of the proposed **streaming** mitigation (memory footprint, bandwidth, and latency trade-offs).

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper identifies a fundamental bottleneck in Pipeline Parallelism (PP) for fine-tuning large models: the per-iteration exchange of gradients across stage boundaries creates pipeline bubbles and makes performance highly sensitive to communication latency.

To address this, the authors introduce FluidPipe, a novel two-stage pipeline training algorithm that eliminates per-iteration gradient synchronization. The core idea is to decouple the stages by equipping the first stage with an auxiliary task head, allowing both stages to update their parameters locally. Cross-stage guidance is provided through bi-directional distillation at a much lower frequency (once per epoch), replacing fine-grained gradient dependencies with coarse, semantic feedback.

### Strengths
This work makes important contributions to distributed training through its innovative approach to pipeline parallelism. The key strength lies in fundamentally rethinking pipeline dependencies rather than optimizing within existing constraints.

The paper introduces FluidPipe, which eliminates per-iteration gradient synchronization through a novel combination of auxiliary heads and bi-directional distillation. This paradigm shift from fine-grained gradient exchanges to coarse semantic feedback opens new design possibilities.

Technically, the work is rigorous with comprehensive evaluations across ViT and BERT architectures. The ablation studies are particularly insightful, revealing that distillation is optional - the auxiliary head itself enables the performance gains. The cost model provides solid theoretical grounding.

The practical impact is significant, demonstrating up to 3.3× speedup while maintaining accuracy, especially valuable in high-latency scenarios. The method requires no architecture changes, making deployment straightforward.

By challenging a fundamental assumption in pipeline parallelism, this work opens a promising research direction that will likely inspire substantial follow-up work.

### Weaknesses
While the paper presents a compelling approach, several limitations warrant attention.

The most significant concern is the restriction to two-stage pipelines. Though justified for isolating the core mechanism, this narrow scope undermines the paper's broader claim of opening a new path for pipeline algorithms. A proof-of-concept with deeper pipelines would substantially strengthen this claim.

The cost-benefit analysis feels incomplete. The computational overhead of the auxiliary head and distillation losses remains unquantified, while the memory burden of storing epoch-long logits could be prohibitive for large-scale applications. These practical costs deserve explicit discussion.

The treatment of the no-distillation variant (FP-ND), while intriguing, lacks depth. Understanding why FP-ND works well - whether through regularization, feature preservation, or other mechanisms - would transform an empirical finding into a fundamental insight.

Comparisons with state-of-the-art scheduling methods like Zero-Bubble PP are missing. A direct comparison under high latency would better demonstrate FluidPipe's unique value proposition versus scheduling optimizations.

Finally, the method's sensitivity to model partitioning remains unexplored. Testing different split points would reveal robustness limits, particularly when Stage 1 becomes too shallow to support effective learning.

### Questions
Scalability Evidence:
Can you provide any preliminary evidence for extending FluidPipe to deeper pipelines? For instance, have you tested scenarios where intermediate stages employ auxiliary heads? A discussion of potential hierarchical distillation schemes for multi-stage setups would help assess the method's generalizability.

Cost-Benefit Analysis:
What is the actual computational overhead of the auxiliary head and distillation mechanism? A detailed per-iteration time breakdown (computation vs communication) would clarify whether advantages persist in computation-bound environments. Additionally, what is the memory footprint of storing epoch-long logits? This is crucial for practical deployment.

FP-ND Mechanism:
What is the underlying mechanism behind FP-ND's strong performance? We hypothesize the auxiliary head may act as a regularizer. Did you analyze feature evolution (e.g., tracking Stage 1 head accuracy, measuring feature similarity)? Such analysis could elevate FP-ND from an empirical finding to a principled method.

Baseline Comparisons:
How does FluidPipe directly compare with state-of-the-art scheduling methods (e.g., Zero-Bubble PP) under high latency conditions? A head-to-head comparison would better demonstrate whether dependency removal outperforms sophisticated scheduling approaches.

Method Robustness:
How sensitive is performance to model partitioning? Have you tested extreme split scenarios (e.g., very shallow Stage 1)? Also, how critical are the distillation weights? Does FP-ND's success primarily stem from eliminating hyperparameter tuning?

Streaming Evaluation:
Did you empirically evaluate streaming logits transfer? Concrete results would validate whether bulk transfer is indeed optimal or if streaming offers practical advantages.

### Soundness
4

### Presentation
3

### Contribution
4
