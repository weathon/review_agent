# Breaking the MoE LLM Trilemma: Dynamic Expert Clustering with Structured Compression

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 8, 4, 2, 4

## Abstract
Mixture-of-Experts (MoE) Large Language Models (LLMs) face a trilemma of load imbalance, parameter redundancy, and communication overhead. We introduce a unified framework based on dynamic expert clustering and structured compression to address these issues cohesively. Our method employs an online clustering procedure that periodically regroups experts using a fused metric of parameter and activation similarity, which stabilizes expert utilization. To our knowledge, this is one of the first frameworks to leverage the semantic embedding capability of the router to dynamically reconfigure the model’s architecture during training for substantial efficiency gains. Within each cluster, we decompose expert weights into a shared base matrix and extremely low-rank residual adapters, achieving up to fivefold parameter reduction per group while preserving specialization. This structure enables a two-stage hierarchical routing strategy: tokens are first assigned to a cluster, then to specific experts within it, drastically reducing the routing search space and the volume of all-to-all communication. Furthermore, a heterogeneous precision scheme, which stores shared bases in FP16 and residual factors in INT4, coupled with dynamic offloading of inactive clusters, reduces peak memory consumption to levels comparable to dense models. Evaluated on GLUE and WikiText-103, our framework matches the quality of standard MoE models while reducing total parameters by approximately 80\%, improving throughput by 10\% to 20\%, and lowering expert load variance by a factor of over three. Our work demonstrates that structural reorganization is a principled path toward scalable, efficient, and memory-effective MoE LLMs.
Code for experiments is available at https://anonymous.4open.science/r/SUBMIT-0001/README.md

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors delineate an optimization trilemma in MoE models of load imbalance,  parameter redundancy, and communication overhead, and propose tackling all three with an online clustering algorithm that periodically groups experts, intra-group parameter saving via a low-rank reparameterization, 2-stage hierarchical routing across and then within expert groups, and finally dynamic offloading and quantization.

### Strengths
1. The main experimental result of maintaining comparable performance at just 20% of the parameter count is remarkable

2. The overall design is comprehensive and well-engineered, with multiple mechanisms appearing meticulously thought through and tuned to work together

3. The hierarchical router is a clever and efficient method to bring meaningful improvements to throughput

### Weaknesses
**Unsubstantiated claims** The authors propose as early as the title that their method deals with load imbalance, yet there are no experimental results on load imbalance with any baselines. The authors only discuss load imbalance in an ablation study where they compare the load balance across their own model with different components ablated away, which is not an appropriate way to demonstrate any improvements to load imbalance. More broadly, the load imbalance angle feels poorly motivated, and I'm not yet convinced intuitively that grouping the experts will necessarily alleviate load imabalance, as the router could just end up favoring one particular group of experts and some subset of experts within that group. The authors should at very least compare load imbalance with the baselines they include (switch and moe-lite using generic load balancing loss), along with techniques that are designed to improve load imbalance, for example expert-choice routing [1] and loss-free load balancing [2].

**Limited experimental validation**. The authors only use two fairly small datasets to validate their claims, and compare with only a generic dense transformer, switch and MoE-Lite. As mentioned above, including expert-choice routing is a highly relevant baseline that offers improvements over generic MoE. The authors also mention StableMoE in their related work but do not compare with it. Additionally, examining the performance of the method at larger model sizes would be useful to validate the scalability of the method. Overall, adding baselines, datasets, and different sizes would help to demonstrate the efficacy of the method, whereas at present the limited experimental results make it hard to critically assess the empirical performance. 

**Limited novelty**.  The authors do acknowledge that there is some crossover with their reparameterization scheme and PEFT, but the proposed base matrix + low rank adaptor used in their expert groupings is used in PEFT-MoE setups [3,4]. The reparamterization scheme is then, in my view, more of an extension of LoRA-MoE beyond pure finetuning. While still mostly novel, the technical contribution is somewhat diminished. (Aside, I would also recommend the authors include a short discussion with the differences of their reparameterization method to PEFT/LoRA-MoE methods).

**Theoretical motivation for the method is poorly written**. The optimization problem referenced in (1) serves almost no purpose as far as I can see, beyond essentially just stating that these three issues (load imbalance, redundancy and dispatch cost) are all bad and we want to minimize them. There's no need to design an optimization problem only to leave it undeveloped and unused. Indeed the coefficients introduced are never mentioned again and the trillemma issues only appear as generic variables.

[1] Mixture of experts with expert choice routing (Zhou et al, NeurIPS 2022) \
[2] Auxiliary loss free load balancing strategy for MoE (Deepseek, 2024) \
[3] Pushing MoE to the limit: extremely parameter efficient moe for instruction tuning (Zadouri, ICLR 2024) \
[4] HydraLoRA: An Asymmetric LoRA Architecture for Efficient Fine-Tuning (Tian, NeurIPS 2024) \

### Questions
As mentioned above, I'd recommend adding to the discussion with existing PEFT-MoE methods. Though I do think your parameter compression approach is mostly new, there is some overlap that I think merits discussion.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
Proposes a unified MoE training framework to tackle three coupled bottlenecks—load imbalance, parameter redundancy, and all-to-all communication.  dynamically clusters experts online using a fused parameter+activation similarity and within each cluster shares a FP16 base matrix and adds very low-rank INT4 residual adapters per expert; It applies hierarchical routing (group → experts) to shrink routing search space and bytes moved

### Strengths
New combination, systemically unified. Online dual-similarity reclustering leveraged during training (using router-anchored activations), coupled with shared-base + ultra-low-rank residuals and hierarchical routing, is an integrated design that targets all three MoE pain points at once. Prior works address these dimensions separately (compression, routing balance, comms libraries); this paper’s co-design and reconfiguration during training are the main novelty levers.

### Weaknesses
Quality parity vs strong MoE - GLUE table shows the method slightly lags Switch-Top2 on MNLI/QQP/SST-2 and GLUE-avg; reconcile this with the “matches quality” claim (report CIs, more tasks).

Stability analysis. Show training stability across recluster intervals, α/β/δ sensitivities, and failure modes (e.g., mode collapse, oscillations).

Calibration of ranks/precision. Justify why r=16 beyond reconstruction error by downstream impact; compare FP8/INT8 variants and per-group vs per-tensor scaling

### Questions
How does GLUE accuracy compare to Switch-Top2 across more tasks and with confidence intervals

At larger expert counts (e.g., E≥128), how do clustering overheads, offload hit-rates, and routing gains scale; any topology-aware grouping results (by GPU placement)? 

What sensitivities did you observe to α (fusion weight), b (EMA), T (reclustering interval), and d (min-gain threshold); any stability safeguards beyond the brief freeze/warm-start?

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
4

### Summary
The paper tackles the MoE “trilemma” (load imbalance, parameter redundancy, and cross-device communication) with a unified framework: (i) **online dual-similarity clustering** that groups experts using a fused metric of parameter and activation similarity; (ii) **intra-group structured compression** that reparameterizes each expert as **a shared group base plus a low-rank residual**; (iii) **two-stage (hierarchical) routing**—first select a group, then select experts within the group. The system is rounded out with **heterogeneous precision** (base in FP16, residual factors quantized) and **NVMe offloading** for long-inactive groups. The authors report large parameter reductions (around ~80%) with comparable quality, 10–20% higher throughput, and ~3× lower load-variance on representative benchmarks.

### Strengths
**1. Insightful related work.** The related-work section is well curated and helps position the contribution clearly within both the compression and MoE-routing literatures, highlighting what prior systems optimize and where gaps remain.

**2. Compelling co-design view.** Treating MoE as a joint algorithm–systems co-design problem is timely. The pipeline—dynamic grouping ($\to$) shared-base + low-rank ($\to$) hierarchical routing—forms a coherent design that aims to reduce compute, memory, and communication together rather than in isolation.

### Weaknesses
**1. Figure-to-text inconsistency.** 

In Figure 1 the low-rank factors are annotated as **(A) in FP16 and (B) in INT4**, which conflicts with the textual description elsewhere (where the base is FP16 and the residual adapters are quantized; or both (A/B) are quantized). Please reconcile the figure and the prose and state the final precision choices per matrix and per layer unambiguously.

Figure 2 color-coding and semantics are unclear. The legend lists a yellow “Inactive expert,” but in the panel the inactive experts appear **gray** and yellow is effectively unused.


**2. Specification of “inactive expert.”**

 The paper defines an expert as *inactive* if it receives zero tokens over a window of ($S_{\text{idle}}$) steps, but the value of ($S_{\text{idle}}$) is never reported. In the reported setting with **32 experts**, standard **router balance** losses should keep activations roughly even; if ($S_{\text{idle}}$) were **256/512 steps**, it seems unlikely that any expert would remain at zero tokens, i.e., *inactive experts may not occur at all*. Please (i) specify ($S_{\text{idle}}$), (ii) report the duration of zero-token windows per expert (and per group), and (iii) inactivity in fewer experts E=8 MoE like Mixture.



**3. Reuse path for the down-projection is unclear.** 

For a single expert the computation expands as
 $
 (y_i = (W^{\text{down}}_{\text{base}} + A^{\text{down}}i B_i^{\text{down}\top}) [\sigma(W^{\text{gate}}{\text{base}}x + A_i^{\text{gate}}B_i^{\text{gate}\top}x) \odot (W^{\text{up}}{\text{base}}x + A_i^{\text{up}}B_i^{\text{up}\top}x)]).
 $
 The paper states that ***“the product involving the base matrix can be efficiently reused for all experts in the group that process the same tokens,”*** which is clear for up/gate (same input (x)). But for down, the input is expert-specific $(h_i)$. Please spell out whether down-base results are actually reused across experts, or whether only the weights are reused (cached) while results are recomputed per $(h_i)$.

**4. Missing citation/novelty vs. D^2-MoE.** 

Section 3.2’s **“shared base with low-rank residuals”** appears conceptually close to the **delta-decomposition** used in **D^2-MoE (Delta Decompression for MoE-based LLMs Compression, ICML 2025)**. The paper neither cites D^2-MoE nor clearly positions its novelty relative to it. Please include ablations isolating dynamic clustering from the base+low-rank factorization to demonstrate incremental gains beyond D^2-MoE; if your method subsumes D^2-MoE under certain settings, state those conditions explicitly.


**5. Baselines lag behind the current literature.** 

Besides MoE-Lite (2023), please include comparisons to more recent compression/merging approaches such as MC-MoE (Mixture Compressor, ICLR 2025), D^2-MoE (Delta Decompression, ICML 2025), and Sub-MoE (2025) to reflect the state of the art.

**6. Model scale is small.** 

Experiments are limited to relatively small settings. To strengthen generality, consider pretrained MoE models such as Mixture-8×7B and Qwen3-30B-A3B, and report both quality and system metrics (throughput, all-to-all bytes, memory) under matched budgets.

---
MC-MoE: Mixture Compressor for Mixture-of-Experts LLMs Gains More ICLR2025 

D^2-MoE: Delta decompression for moe-based llms compression ICML2025 

Sub-MoE: Sub-MoE: Efficient Mixture-of-Expert LLMs Compression via Subspace Expert Merging 2025

### Questions
See weakness.

If these issues are addressed, I would be happy to raise my score.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes a mixture-of-experts layer equipped with a two-stage routing mechanism. That is, each input token is first routed to a group of experts, then within the selected group of experts, it is routed to top matching experts (group members). This method reduces the communication cost by factoring all-to-all routing into two stages. Group assignments are performed via parameter/activation similarity-based clustering of expert networks.

### Strengths
1. On top of the routing mechanism, authors conduct experiments with quantization and offloading that achieves the throughput comparable to dense Transformer while offering performance between standard MoE counterparts and dense Transformer.
2. Group assignments by clustering is a new technique for pre-training stage which seems to be contributing quite a bit to the performance according to Table 3.

### Weaknesses
1. A major weakness for me is that a primary contribution of the paper, two-stage routing, is actually not new. Hierarchical routing was proposed in [1] (not to mention an iconic MoE paper [2]) for the same reason of reducing inter-communication overhead. However, I did not see any discussion of this overlap in the main text.
2. As a result of Weakness 1 above, the remaining contribution of the paper (group assignment by clustering and quantization) is incremental and relatively weaker to stand alone as a conference main track paper (but can be strengthened by rewriting the paper narrative and adding larger-scale experiments which at least can show a minimal sign of scaling laws).

Therefore, although the paper shows promising efficiency gains through experiments, I believe a major revision and larger-scale experiments are needed to fully develop the paper.
___

### References

1. Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, Jeff Dean. Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer. ICLR 2017
2. Michael I. Jordan, Robert A. Jacobs. Hierarchical mixtures of experts and the EM algorithm. Proceedings of 1993 International Conference on Neural Networks (IJCNN-93-Nagoya, Japan), Nagoya, Japan, 1993

### Questions
See Weaknesses.

### Soundness
3

### Presentation
2

### Contribution
1

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper tackles the **MoE trilemma**—load imbalance, parameter redundancy, and communication overhead—via a unified framework that combines: (i) **online expert clustering** using a fused parameter/activation similarity; (ii) **intra-group structured compression** (a shared base plus very low-rank residual adapters per expert); (iii) **hierarchical routing** (group-then-expert) to reduce all-to-all scope; and (iv) **heterogeneous precision with dynamic offloading** to lower memory. On GLUE and WikiText-103, the authors report matching or near-baseline quality with large parameter reductions, modest throughput gains, reduced load variance, and smaller peak memory.

### Strengths
- **System-level perspective on MoE.** The paper provides a structured diagnosis of load imbalance, redundancy, and communication and proposes a unified treatment rather than isolated tweaks.  
- **Cohesive design.** Online dual-similarity clustering aligns experts with similar parameters and activations; shared base + low-rank residuals exploit intra-cluster redundancy; hierarchical routing narrows the routing search space and potentially the all-to-all domain; mixed precision/offloading reduces memory.  
- **Initial empirical promise.** On the reported benchmarks, the framework yields sizable parameter savings and 10–20% throughput improvements with reduced load variance while roughly matching baseline quality in some settings.

### Weaknesses
1. **Experimental scope is limited.** Evaluations focus on GLUE and WikiText-103 with relatively small models. Conclusions may not transfer to pretraining-scale MoEs, long-context inference, instruction-following, or reasoning tasks. The claim of addressing the trilemma feels **overstated** given the narrow evidence.
3. **Tri-lemma not convincingly “solved.”** Reported throughput/memory improvements are helpful but relatively **minor**, and some quality drops persist. The paper does not yet present clear **Pareto dominance** (quality vs throughput vs memory) against strong MoE baselines at scale.
4. **Baselines could be stronger.** Missing comparisons against recent **system-optimized MoE** variants and modern compression baselines adapted to MoE. 
5. **Training and samples required.** Despite being framed as structural/system reorganization, the method **does require training/fitting** during training time: online reclustering, prototype updates, initialization/adaptation of low-rank residuals, and bookkeeping for offloading. The **compute overhead**, stability, and sensitivity to calibration/streaming data are not comprehensively reported.
6. **Novelty is incremental.** The core components—similarity-based clustering, shared bases with low-rank residuals, hierarchical routing, mixed precision/offloading—have precedents; the contribution reads as a reasonable integration rather than a clear methodological leap.

### Questions
1. **Evidence you “solve” the trilemma.** Provide **Pareto frontiers** (quality vs throughput vs memory) against strong MoE baselines at multiple scales. Where does your method strictly dominate, and where is it a trade-off?
2. **Generalization and stability.** How robust is online clustering under domain shift or non-stationary traffic? Any oscillation or collapse modes? Can groups be frozen at convergence without losing gains?

### Soundness
3

### Presentation
2

### Contribution
2
