# Optimized Early-Exit Based Speculative Decoding via Pipeline Parallelism

- Decision: Reject
- Scores: 4, 6, 4, 2

## Abstract
Large language models (LLMs) deliver impressive generation quality, but incur very high inference cost for the auto-regressive decoding manner.
Early-exit based speculative decoding (EESD) has emerged to reduce decoding latency.
However, in practice, many approaches struggle to achieve an expected acceleration in the draft-then-verify paradigm even with a well-aligned early-exit head and selected exit position.
Our analysis reveals that EESD only pays off when the vast majority of draft tokens are accepted by the LLM.
Otherwise, the draft cost may overcome the acceleration gain and lead to a negative speedup.
To mitigate this, we propose \textbf{Pipeline-Parallel Speculative Decoding (PPSD)} that fully pipelines the draft and verification work so that no effort is wasted on failed predictions.
It has two key innovations.
Pipeline-Parallel Early-Exit Execution: We design a fine-grained pipeline allocation and execute system, in which early-exit (draft) computations and remaining-layer (verification) computations overlap with minimal blocking waste.
Verify-while-draft Decoding: We interleave drafting and verification per token. 
While the LLM is verifying the current token in its final layers, the early-exit path simultaneously drafts the next token. 
This high parallel scheme keeps all units busy and validates tokens on-the-fly, analogous to pipelining the speculation and verification stages. 
Each token is confirmed as soon as it enters the output, ensuring correctness without stalling.
All these design choices are supported by both theoretical analysis of pipelined throughput and extensive experiments.
Empirical results confirm that PPSD achieves state-of-the-art acceleration in self-speculative LLM inference. 
On diverse benchmarks, PPSD achieves speedup ratios in the range of $2.01\times\sim3.81\times$, which gains almost the optimal acceleration at the fixed acceptance rate and exit position, showcasing its advancement in providing efficiency.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes Pipeline-Parallel Speculative Decoding (PPSD), a self-speculative decoding scheme with pipeline-parallel early-exit execution to parallel the draft and verification process and verify-while-draft to interleave drafting and verification per token. Several experiments demonstrate the effectiveness of the method.

### Strengths
-  The paper identifies a real failure mode of EESD: long draft lengths increase rejection waste with good motivation.
-  The throughput expression in Eq. (6) makes the α–E trade-off explicit and clarifies when speedups are expected.  
-  Results span multiple model sizes (7B→70B) and tasks, with PPSD outperforming EESD variants and Medusa on speed while keeping quality unchanged.

### Weaknesses
-  While verification overlaps drafting, it is not free in wall-clock terms; the claim rests on ideal overlap and balanced stages. The paper could more carefully separate amortized from absolute costs and report stall/bubble statistics.  
-  Eq. (6) highlights dependence on α and E; more sensitivity metrrics (input difficulty, prompt length, temperature, exit-head training regimen) would clarify robustness across workloads.  
- The authors may need to do more research on more recent LLMs (e.g., Qwen 3 and Llama3/4 families).

### Questions
Please refer to weakness.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes Pipeline Parallel Speculative decoding, a pipeline framework for early-exit self-speculative decoding. It involves 2 modifications: 1) Pipeline-Parallel Early-Exit Execution divides layer of the model evenly to create balanced pipeline blocks, and 2) Verify-While-Draft Decoding overlaps final-layer verification with drafting. Experiments on Vicuna-7B/13B and LLaMA-2 13B/70B show 2.0×–3.8× speedups across benchmarks.

### Strengths
The idea of fully pipelining the drafting and verification stages is innovative. By evenly partitioning the model layers into blocks and pipelining them with draft tokens produced from early exits, the method ensures that the worst-case performance is no worse than standard autoregressive (AR) decoding (i.e., when the speculation accuracy equals zero). This design is both theoretically appealing and practically robust, as it guarantees that the inference speed is lower bounded by that of AR decoding.

The theoretical analysis (Eq. 4–7) is convincing and strict, providing principled analysis of the relationship between acceleration and other factors (e.g. speculation length, accuracy).

Ablation studies (Figures 3–5, Table 3) are comprehensive, covering multiple aspects of factors including pipeline granularity, exit position, and exit head design.

### Weaknesses
1.	In my opinion, this paper does not belong to the topic of ‘speculative decoding’, but more to another topic like ‘pipeline decoding’ or something similar. Speculative decoding is typically about utilizing token-level parallelism to verify multiple draft tokens in a single forward. On the contrary, this paper uses device-level parallelism, while in each device the forward is only about 1 token.
2.	In sec4.1, the paper claims that if N/E is too large, then the layers can be further partitioned by S. However, the pipeline efficiency is ultimately dominated by E, that the next token can only be obtained after E layers of draft, and further partitioning cannot make the S (S<E) layers pipelined, since the next token is not available to the 0 layer when S+1-N layers are running.
3.	It is confusing to me that PPSD and EESD use the same early-exit configuration for speculation, but the accuracies of PPSD are much higher. The explanation in sec5.2 that ‘PPSD avoids accumulation of invalid draft tokens’ makes me even more confused, as I did not find out how PPSD does this. Further clarification could make it clearer.
4.	The speculation head is trained by distillation, while the used datasets are not specified. More specification about training details would improve reproductability.

### Questions
1.	Why and how does the pipeline efficiency improve with S (S<E) layers pipelined?
2.	Why the accuracies of PPSD is much higher than baselines, with the same early-exit configuration and head? Where is the invalid token accumulated, and how did PPSD address this issue?
3.	On which dataset is the speculation head trained?

### Soundness
3

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
The paper studies why early‑exit self‑speculative decoding (EESD) often underperforms in wall‑time. The authors argue that draft‑then‑verify wastes compute when many drafted tokens are later rejected. They propose PPSD, which splits the model into pipeline stages and verifies while drafting: while upper layers verify the current token, early layers draft the next one. Specifically, the authors propose pipeline-parallel early-exit exection and verify-while-draft decoding, which can fully overlap the drafting phase and verification phase. The authors also provide a theoretical analysis to proposed methods. Experiments on 3 benchmarks demonstrate the effectiveness of PPSD.

### Strengths
1. The paper is overall well-written. and easy to understand. The figures and tables are intuitive.
2. The authors provide some theoretical analysis of the proposed methods. The throughput formulas (Eq. 4–6) offer intuition on acceptance–depth trade‑offs.
3. The experiments demonstrate the consistent efficiency improvement of PPSD.

### Weaknesses
1. **Clarification of motivation and method.** The motivation and the idea of parallelizing the drafting and the verification are well explored by some related works, especially PEARL [1]. The whole architecture of PPSD is more like the extension of PEARL into EESD. More specifically, in Figure 2, "2 stage earlier" corresponds to "pre-verify" and "3 stage faster" corresponds to "post-verify". The authors should give a more comprehensive comparison between PPSD and PEARL / DSI, including empirical experiments.
2. **Regarding Tensor Parallelism.** While the pipeline parallelism can help overlap the drafting and the verification, tensor parallelism (TP) remains unexplored. Moreover, the computational cost of the additional trained early-exit head is not discussed. If the computational cost of the additional head is negligible, please further illustrate the difference between PPSD and Medusa. (prediction from the early layer and the last layer?) 
3. **Unfair resource accounting.** PPSD relies on multiple workers. Tables compare wall‑time to AR/EESD apparently on fewer devices. Tokens/s without normalizing by **GPU count/bandwidth** is not persuasive.

If the authors clearly address these concerns, I am willing to increase my rating.

### Questions
1. Could you please provide the exact number of GPUs during evaluation?
2. What is the computational cost of the early-exit head? How long does it take for training?
3. Could you please provide a direct comparison between PPSD and TP?
4. As PPSD is still a training-based method, what is its strengths against EAGLE?
5. What is the detailed communication cost during inference?

### Soundness
2

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
This paper introduces Pipeline-Parallel Speculative Decoding (PPSD), an architectural optimization for Early-Exit Speculative Decoding (EESD). PPSD utilizes a "Verify-while-Draft" pipeline parallel execution scheme aimed at eliminating redundant computation caused by speculative failures, thereby boosting LLM inference speed.

### Strengths
The reported speedup figures are numerically competitive under the specific, tested hardware and dataset configurations.

### Weaknesses
1. **Lack of Novelty:** PPSD essentially combines two established concepts: Speculative Decoding and Pipeline Parallelism/Model Parallelism. The core speedup comes from utilizing idle time for draft generation, making it primarily an **engineering optimization** rather than a fundamental breakthrough in decoding algorithms. Given the computational characteristics of EESD, adopting parallelization to fill idle compute time is straightforward.
3. **Missing More Key Ablation Studies:** The paper should provide more ablation experiments to isolate the pure gain from "Pipeline Parallelism" versus gains from "Underlying CUDA/operator Implementation Optimizations." Also, I suggest the authors should analyze the sensitivity of the speedup to different pipeline depths (Stage Count) and communication latency.

### Questions
See weaknesses.

### Soundness
1

### Presentation
4

### Contribution
2
