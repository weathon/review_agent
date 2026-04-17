# Fastcar: Cache Attentive Replay for Fast Auto-Regressive Video Generation on the Edge

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
Auto-regressive (AR) models, initially successful in language generation, have recently shown promise in visual generation tasks due to their superior sampling efficiency.
Unlike image generation, 
video generation requires a substantially larger number of tokens to produce coherent temporal frames, resulting in significant overhead during  decoding. 
We first make specific key observations: (i) MLP modules in the decode phase dominate the inference latency, and (ii) there exists high temporal redundancy in MLP outputs of adjacent frames. 
With the insights, we propose **FastCar** to accelerate the decode phase for the AR video generation  by exploring the temporal redundancy. The Temporal Attention Score (TAS) is proposed to determine whether to apply the replay strategy (i.e., reusing cached MLP outputs from the previous frame to reduce redundant computations)  with detailed theoretical analysis and justification.
Furthermore, we develop a hardware accelerator on FPGA with Dynamic Resource Scheduling  based on TAS to enable better resource utilization and faster inference.
Experimental results demonstrate the effectiveness of our method, which outperforms traditional sparse attention approaches with more than 2.1x decoding speedup and higher energy efficiency on the edge.
Furthermore, by combining FastCar and sparse attention, FastCar can boost the performance of sparse attention with alleviated drifting, demonstrating our unique advantages for high-resolution and long-duration video generation.
Code: https://github.com/shawnricecake/fast-car

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes FASTCAR, an efficient framework for auto-regressive (AR) video generation, based on the insight that MLP modules are the main latency bottleneck and exhibit high temporal redundancy. FASTCAR implements a "cache attentive replay" strategy, using a Temporal Attention Score (TAS) derived from the preceding attention layer at no extra computational cost to determine whether to skip an MLP computation and reuse the cached output from the previous frame. Experimental results demonstrate that this method outperforms sparse attention approaches on VBench, achieving speedup with better generation quality and power efficiency on edge hardware.

### Strengths
1. This paper is well-written and easy to read.
2. The proposed method is simple, efficient and effective.
3. When combined with sliding window attention, FastCAR show the possibility to alleviate drifting caused by global information dropping.

### Weaknesses
1. The hardware design section is not connected to other parts and no experiment results prove its efficiency.
2. The claim that MLPs dominate latency is based on 8 frames short sequences which is relatively short to real-world using cases.
3. Lacks discussion of similar caching approaches, such as DeepCache in diffusion generation.
4. typo: "third" in line 459. 

[1] [CVPR2024] DeepCache: Accelerating Diffusion Models for Free

### Questions
Though QK^T is computed in the attention layer, the result is temporary and on-chip in Flash Attention implementations. How do you calculate TAS efficiently when Flash Attention fusion is used? Does this require kernel modifications and what is the overhead?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes FastCar, a method to speed up autoregressive video generation by avoiding redundant MLP computation across adjacent frames. The authors profile decoding latency and argue that the MLP/FFN block, not attention, is the main bottleneck. They observe that many tokens barely change between consecutive frames, so instead of recomputing the MLP output every time, FastCar “replays” cached MLP outputs from the previous frame for stable tokens. A Temporal Attention Score, derived from cross-frame attention, is used to decide which tokens can be safely replayed, and the paper provides a theoretical argument that high TAS implies similar MLP outputs.

The paper further presents an FPGA-oriented runtime scheduler (DRS) that dynamically assigns only the “needs recompute” tokens to compute cores to keep utilization high. Experiments claim >2× speedup over dense decoding and better quality / less drift than sparse-attention baselines. The high-level idea of reusing features for unchanged regions is conceptually simple and related to prior token-reuse / feature-caching work, but the authors position their novelty in applying it to autoregressive video decoding, tying the gating decision to TAS, and demonstrating a hardware-aware implementation.

### Strengths
The paper performs actual latency profiling and shows that, in autoregressive video decoding, the main bottleneck is the MLP/FFN block rather than attention. This is valuable because most prior acceleration work focuses on attention; here the motivation is concrete and data-driven.

The core mechanism of FastCar — reusing the previous frame’s MLP outputs for tokens that barely change, instead of recomputing them every step — is straightforward and can be applied at inference time without retraining. The per-token, per-layer replay design makes it easy to imagine plugging this into existing AR video generators as a decoding-time optimization.

The method uses a Temporal Attention Score (TAS), derived from cross-frame attention, to decide which tokens can safely skip recomputation. The paper also provides a theoretical argument that high TAS implies similar MLP outputs. This lifts the approach above a pure heuristic and makes the selective replay decision more interpretable.

Beyond the algorithmic idea, the paper presents a hardware-oriented runtime scheduler (DRS) on FPGA that dynamically dispatches only the “needs recompute” tokens to available compute cores, improving utilization and energy efficiency. This gives the work a full-stack feel rather than just a conceptual trick.

The experiments report over 2× decoding speedup while maintaining generation quality and reducing long-horizon drift compared to sparse-attention baselines. This suggests the approach could be especially useful for long-duration or higher-resolution AR video generation.

### Weaknesses
1. The core idea — detect tokens/regions that barely change over time and skip recomputing them by reusing cached features — is not fundamentally new. Similar per-token reuse / feature caching / dynamic skipping ideas have already appeared in video and diffusion generation acceleration. The paper mainly adapts this intuition to autoregressive video decoding, adds a specific gating signal (TAS), and wraps it with an FPGA story. This feels more like an incremental systemization than a genuinely new algorithmic principle.

2. And as written, all results (latency profiling, TAS analysis, replay ablations, quality vs. drift comparisons, and the FPGA/DRS speedup and energy numbers) appear to be reported on a single autoregressive video generation model (vila-u-7b-256)? Could you explain why the evaluation is limited to this one model? In particular: (1) is FastCar tied to architectural details of that specific model (e.g., tokenizer, decoder layout, MLP shape), or should it in principle apply to other AR video generators? (2) did you attempt to run the method on any other AR video backbones or model sizes, and if so, what prevented you from reporting those results? Right now it is difficult to judge how general the proposed approach is versus a case study on one model. 

If I missed any details, please let me know during the rebuttal period.

### Questions
See Weaknesses

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
4

### Summary
This paper proposes FastCar, a system-level framework to accelerate auto-regressive (AR) video generation on edge devices by exploiting temporal redundancy in MLP outputs across adjacent video frames. The core insight is that MLP modules, not attention, dominate decoding latency in AR video models like VILA-U, and their outputs exhibit high similarity between consecutive frames. To leverage this, the authors introduce the Temporal Attention Score (TAS), a lightweight metric derived from attention logits, to decide when to replay (i.e., reuse) cached MLP outputs from the previous frame instead of recomputing them. They further design a custom FPGA-based hardware accelerator with Dynamic Resource Scheduling (DRS) that adapts computation allocation in real time based on TAS-driven replay decisions. Experiments show that FastCar achieves >2.1x decoding speedup, ~45% FLOPs reduction, and higher energy efficiency than sparse attention baselines, while preserving video quality and even mitigating temporal drifting when combined with sparse attention.

### Strengths
1. Strong empirical motivation: The paper provides compelling profiling evidence that MLPs, not attention, are the bottleneck in AR video decoding, which justifies shifting optimization focus away from KV-caching or sparse attention (common in LLMs) toward MLP replay.
Novel and well-motivated algorithmic component: The Temporal Attention Score (TAS) is a simple yet effective proxy for temporal similarity that incurs zero extra compute (as it reuses existing attention logits). The theoretical analysis (Theorems 4.4–4.6) formally links TAS to MLP output similarity, lending credibility to the replay strategy.
2. Hardware-software co-design: The integration of TAS with a custom FPGA accelerator featuring DRS is a notable strength. The DRS mechanism intelligently balances workload across cores in response to dynamic replay patterns, which is essential for real-world deployment.
3. Comprehensive and reproducible evaluation: The authors evaluate across multiple replay ratios, report detailed metrics (PSNR, SSIM, LPIPS, VBench, latency, power, FLOPs), compare against StreamingLLM (a relevant sparse attention baseline), and demonstrate complementarity with sparse attention.
4. Practical relevance: The work directly addresses a critical bottleneck in deploying AR video models on edge devices, making it highly relevant to both systems and generative AI communities.

### Weaknesses
1. Threshold selection is manual: The replay threshold is tuned empirically. The paper shows robustness across thresholds (Fig. 4), but does not propose an adaptive or learned thresholding strategy, which could improve usability in dynamic real-world scenarios.
2. Limited evaluation and comparison: Evaluation limited to VILA-U, comparison limited to StreamingLLM. Yes, VILA-U is the only open-source AR video generation without diffusion model, but it may indicate that the community and business are not interested in such types of models.

### Questions
Generality beyond VILA-U: Is FastCar applicable to other AR video models that may have different MLP/attention ratios or tokenization schemes? What architectural properties are required for TAS to be predictive?

### Soundness
3

### Presentation
3

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
This paper aims to address the high computational cost of auto-regressive (AR) video generation, particularly for deployment on edge devices. The authors argue that in AR video generation, the MLP modules—not the attention modules—dominate the inference latency during the decoding phase. They also observe a high degree of temporal redundancy in the outputs of these MLP modules for adjacent frames.

Based on these findings, they propose FastCar, a framework that accelerates decoding by conditionally reusing cached MLP outputs from the previous frame. The decision to reuse or recompute is governed by a novel metric called the Temporal Attention Score (TAS), which measures the similarity between a token's query and the key of its corresponding token in the prior frame. 

Experimental results on the VILA-U model show that FastCar significantly outperforms sparse attention methods, achieving over a 2.1x speedup when combined with sparse attention, while better preserving generation quality and improving energy efficiency.

### Strengths
1. The paper is well-written. The proposed method is well-illustrated and easy to follow.

2. **The proposed method is simple and efficient:** Using the Temporal Attention Score (TAS)—a metric that is effectively "free" as it is derived from pre-existing attention calculations—to guide the caching strategy is a very clever design choice. This avoids the overhead that often plagues other dynamic execution methods.

3. **Comprehensive empirical evaluation:** The paper presents a robust set of experiments with convincing results. The method demonstrates significant improvements in latency, throughput, and power efficiency. The comparison against sparse attention (StreamingLLM) effectively highlights the superiority of the proposed approach for this problem domain. Furthermore, showing that FastCar can be combined with sparse attention to mitigate its weaknesses (like "drifting") and achieve even greater speedups is a powerful demonstration of its complementary nature.

4. **Practical hardware co-design and validation:** This work goes beyond a purely algorithmic proposal by designing, implementing, and evaluating a hardware accelerator on an FPGA. The development of Dynamic Resource Scheduling (DRS) to handle the dynamic workloads shows a deep understanding of the practical challenges involved in deploying such models. This end-to-end, software-hardware co-design approach greatly increases the credibility and practical value of the research.

### Weaknesses
1. **Limited architectural generalization:** The experiments and analysis are conducted exclusively on the VILA-U model. While the authors correctly note the lack of other open-source AR video models, this raises a question about the generality of the core insight. The MLP-bottleneck observation may be specific to this particular architecture's configuration (e.g., hidden size, MLP expansion factor). The paper would be stronger if it included analysis on other AR models (even from image or language domains) to hypothesize about the broader applicability of the findings.

2. **Static thresholding mechanism:** The replay decision relies on a single, manually-tuned, and globally consistent threshold (`τ`). While effective, this is a relatively simple criterion. A more sophisticated, adaptive thresholding mechanism could potentially unlock a better trade-off between performance and quality. The ablation on "inconsistent thresholds" is a good first step, but since the thresholds were manually set, it doesn't fully explore the potential of adaptive strategies.

### Questions
1. The Temporal Attention Score (TAS) is calculated by averaging scores across all attention heads. Did you investigate the behavior of individual heads? Is it possible that some heads are more specialized in tracking temporal correspondence and could serve as better indicators for the replay decision than a simple average?

### Soundness
2

### Presentation
3

### Contribution
3
