# Mirror Speculative Decoding: Breaking the Serial Barrier in LLM Inference

- Avg Score: 5.33
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 6, 6

## Abstract
Speculative decoding accelerates LLM inference with draft lookahead, but its effectiveness is bottlenecked by autoregressive draft generation: larger drafts improve acceptance yet also increase speculation latency overhead, capping speedup. Existing approaches such as Medusa, Hydra, EAGLE partially address draft inefficiency, but ultimately trade acceptance rates for reduced draft latency, or preserve acceptance at the cost of added overheads that limit scaling.

Modern SoCs increasingly integrate heterogeneous accelerators, most commonly GPUs and NPUs with complementary throughput and efficiency characteristics, yet existing approaches are accelerator-agnostic and usually place both draft and target on the same type of device, which leaves cross-accelerator parallelism unused. We introduce Mirror Speculative Decoding (Mirror-SD), which breaks the latency--acceptance tradeoff by launching branch-complete rollouts from early-exit signals in parallel with the target’s suffix and by explicitly mapping computation across heterogeneous accelerators. In this design, the draft speculates forward token continuations for target to verify, while the target speculates correction paths for the draft, creating a bidirectional speculative process. To further reduce draft speculation latency overhead while preserving acceptance semantics, we pair Mirror-SD with speculative streaming (SS) so the draft emits multiple tokens per step. This dual strategy of combining parallel heterogeneous execution and SS pushes speculative decoding closer to its ideal regime of high acceptance with negligible speculation overhead. On SpecBench with server-scale models from 14B to 66B parameters, Mirror-SD consistently delivers realistic end-to-end gains, achieving 2.8$\times$--5.8$\times$ wall-time speedups across diverse tasks representing 30\% average relative improvement over the strongest baseline, EAGLE3. By eliminating serial bottlenecks and exploiting multi-accelerator SoCs, Mirror-SD establishes a practical low-latency regime for large-scale LLM serving. We plan to release code and draft model checkpoints to facilitate reproducibility and further research upon acceptance.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes Mirror Speculative Decoding for early-exit speculative decoding. It trains a bypass draft model, using the topk sampling tokens from middle-layer early exit as input and output k drafted n-grams. The drafting stage runs on NPU, pipelined with target verification on GPU. It also employs speculative streaming to further utilize hardware.

### Strengths
The idea of ‘early-exit topk sampling’ is creative, where topk sampled tokens from the early exit head enabled n-gram of drafting. Although early-exit drafting has been a common idea, directly using sampled tokens rather than early hidden states is new.

The idea of boosting NPU utilization is practically important, yet it has been overlooked in the existing works.

The empirical results is strong, showing that the potential of speculative decoding can be further explored in heterogenous-device pipelining scenario.

### Weaknesses
Main concerns:

According to the paper, the early exit hidden states are directly input to sampling head, without training on the original model. The first tokens of speculation are exactly these topk tokens, meaning that if they are rejected then the whole sequence will also be. Existing works (e.g. LayerSkip) require training on the base model to ensure this accuracy, while this paper does not. Does this mean that the accuracies of early-exit sampling can natively be high without training?

Does the baselines also use device pipeline for acceleration? In Fig3(a), the drafting latency of Mirror-SD does not change with drafting length increasing, while other baselines do. I believe that the latency of Mirror-SD remains the same as the drafting is on NPU. If the baselines also use NPU for pipelining, their latency should also have no change.

Minor concerns:

The proposed method is only evaluated on GPU-NPU systems, while it can also run on GPU-only systems, with a spare of some GPUs for drafting pipelining. Extra results on GPU-only systems can show its potential wider usage.

The paper trains a new drafter rather than using existing models, introducing training overhead. Moreover, whether the acceleration is due to training or pipelining, is not clear.

### Questions
1. How accurate are the early-exit sampled tokens? It will be clarified if you could provide numbers of accuracies.
2. Does the baselines also use device pipline (NPU in your case) for acceleration?
3. How about running the pipeline on GPU-only systems, with a spare of some GPUs for drafting

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
This paper introduces Mirror Speculative Decoding (Mirror-SD), a system-algorithm co-design that enables parallel execution of the draft and target models to accelerate LLM inference. The method is validated  on SpecBench and MT-Bench with 14B–66B models, and obvious speedup is achieved.

### Strengths
- This paper addresses a problem of high practical importance: the fundamental sequential bottleneck in speculative decoding that limits inference latency, a well-known challenge in real-time LLM serving.

- The architectural innovation that enables concurrent draft and target execution, effectively breaking the sequential dependency that has limited prior methods.

- The well-executed latency modeling provides clear theoretical grounding for the performance gains, while the thorough literature survey convincingly establishes the necessity of this optimization and demonstrates strong understanding of the field.

- The experiments show significant and consistent speedups  across multiple model families, scales, and tasks. The comprehensive comparisons against modern baselines and thorough ablation studies provide robust evidence for the method's effectiveness and practicality.

### Weaknesses
- While the parallelization of target-draft execution is effectively demonstrated, the core methodology builds substantially upon the speculative streaming framework. The extension from single-model to dual-model speculation represents a valuable but incremental advancement rather than a fundamental algorithmic shift.

- The empirical validation, though thorough on Apple SoC architectures, leaves open questions regarding broader hardware generalization. Performance analysis on industry-standard platforms (e.g., NVIDIA GPU clusters or TPUs) would significantly strengthen the claim of practical utility across common deployment scenarios.

- The system's practical adoption faces challenges due to the inherent complexity of early-exit coordination and cross-accelerator synchronization. A concrete discussion of integration pathways with production inference frameworks would enhance the work's translational impact and address legitimate deployment concerns.

### Questions
- A quantitative analysis of how performance scales under different compute capacity ratios between the draft and target models would further strengthen the work. Exploring this hardware heterogeneity would help clarify whether the performance gains are robust across imbalanced hardware configurations.

- Extending the evaluation to include mainstream accelerator platforms, such as NVIDIA GPUs or TPUs, would significantly enhance the practical relevance of the proposed method. Demonstrating its effectiveness in these common deployment environments is crucial for assessing its broad impact.

- A discussion on integrating the proposed mechanisms into established inference frameworks like vLLM would be highly valuable. Elaborating on the practical implementation of early-exit coordination and cross-accelerator synchronization would help address potential concerns regarding system complexity and adoption.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces Mirror Speculative Decoding (Mirror-SD), a system-algorithm co-design method to accelerate Large Language Model (LLM) inference. It breaks the traditional serial bottleneck of draft generation in speculative decoding by parallelizing and overlapping the target model suffix computation with the draft model's generation process. It is specifically designed for heterogeneous accelerators (e.g., GPU-NPU setups) and achieves a throughput acceleration of 2.8x to 5.8x while preserving output correctness.

### Strengths
+ **Significant Acceleration:** Achieves 2.8x-5.8x wall-time speedups and a substantial 30% average relative improvement over the SOTA EAGLE3.
+ **Correctness Preserved:** The method guarantees the exact same output quality as the original target model while accelerating inference.
+ **Heterogeneous Hardware Adaptability:** Leverages the heterogeneous accelerators (GPU + NPU) found on modern SoCs, enabling overlapping computation and greatly improving hardware resource utilization.

### Weaknesses
- The analysis mentions the use of a **lightweight** draft model. This means deploying Mirror-SD is not a simple drop-in replacement; it requires managing and maintaining **two models** (target and draft) and potentially training the draft model specifically for optimal acceptance rates within the Mirror-SD framework.
- The evaluated metric only presents end-to-end wall-time speedups (latency). Readers could be curious about other metrics like **energy efficiency**, especially as running two models concurrently on separate devices inherently uses more power than running one sequentially.

### Questions
see weaknesses.

### Soundness
3

### Presentation
4

### Contribution
3
