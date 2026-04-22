# DHKR: Dynamic Hierarchical Knowledge Routing for Efficient Low‑Resource Alignment

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 4, 6

## Abstract
Knowledge-Aligned Domain Shift Tuning (KADA) is a PEFT framework based on the Lottery Hedge Fund Hypothesis (LHFH) to identify and reuse latent knowledge fragments.  
Although KADA bridges knowledge gaps between source and target domains, 
it relies on a fixed set of subnetworks, which limits flexible adaptation and prevents automatic discovery of optimal model capacity.
Existing MoE and dynamic PEFT methods lack a unified mechanism that jointly enables adaptive capacity growth and strong routing stability.

To address these limitations,
DHKR employs a two-level routing mechanism to expand subnetworks hierarchically on demand (domain $\rightarrow$ modality), explore and adjust capacity ($K \times L$) as needed for new domains.
To support dynamic capacity growth, DHKR stabilizes routing via a composite growth trigger—monitoring stagnation, entropy, imbalance, and instability—and multi-level Loss-Free Balancing (LFB).
Ablation studies show that these mechanisms reliably prevent routing instability during growth.
Like KADA, 
DHKR places the Knowledge Steering Layer (KSL) immediately below the LM head and inherits its heritage, enabling efficient parallel routing while keeping the 4‑bit backbone frozen.
Experiments show that DHKR improves calibration (ECE 0.02 vs. KADA 0.13) and lowers training cost (5.67 sec/iter vs. AdaLoRA 15.00 sec/iter), demonstrating both robustness and practical efficiency.
DHKR provides a unified design for dynamic, knowledge-aligned adaptation for knowledge jackpots while maintaining routing and calibration stability.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces Dynamic Hierarchical Knowledge Routing (DHKR), a parameter-efficient fine-tuning framework that extends Knowledge-Aligned Domain Shift Tuning (KADA) by enabling dynamic subnetwork growth for domain adaptation. DHKR uses a two-level hierarchical routing mechanism that expands subnetworks on demand. Experiments on two instruction and QA benchmarks show that DHKR improves adaptability and stability over KADA.

### Strengths
- The paper introduces several relevant components, including dynamic subnetwork growth, hierarchical routing, and routing stabilization mechanisms, to effectively enhance the flexibility and efficiency of the original KADA method.

- The runtime analysis demonstrates improved GPU utilization and lower per-iteration runtime, even when training a substantially larger set of parameters, which highlights the method’s computational efficiency.

### Weaknesses
- The empirical results (e.g., Tables 1 & 2) are relatively weak, as the proposed method achieves only comparable or worse performance than existing baselines. The rationale for using certain metrics, particularly ROUGE, METEOR, and calibration-based measures, is not clearly justified.

- The evaluation scope is limited to only two datasets; broader testing on more diverse and challenging benchmarks (e.g., reasoning tasks) is necessary to substantiate the method’s effectiveness.

- Additional ablation studies are needed to quantify the individual impact of each proposed component (dynamic growth, hierarchical routing, and stability control).

- The related work discussion is underdeveloped and should be expanded to better connect this work to prior studies, especially the extensive literature on routing and expert selection methods.

- Presentation quality can be improved: some references are missing or incorrectly formatted, and result tables (e.g., Tables 1 & 2) could be reformatted or reorganized to emphasize key findings more clearly.

### Questions
Please see weaknesses.

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
4

### Summary
This paper introduces a new framework to improve domain adaptation for large language models, building on parameter-efficient fine-tuning (PEFT) methods such as LoRA and BitFit. This work presents a method, named DHKR, that allows models to grow dynamically and route knowledge hierarchically using PEFT adapters.
- DHKR extends a previous framework KADA, through a two-level subnetwork growth mechanism inspired by the mixture-of-experts architectures. The system organizes subnetworks hierarchically: the first layer handles domain specialization, and the second layer captures modality-specific nuances. This structure allows new subnetworks to be added when a performance bottleneck is detected. Another knowledge steering layer (KSL) operates on top of a frozen, quantized LLM, combining its outputs with the base representation. 
- To stabilize training and prevent catastrophic forgetting, DHKR employs a training objective that combines cross-entropy loss with regularization terms that prevent overgrowth using a Minimum Description Length penalty. Subnetworks grow progressively, initialized through Net2WiderNet duplication with Gaussian perturbations to maintain diversity.

The experiments evaluated DHKR on the Meta-Llama-3-8B-Instruct model using Alpaca and OpenBookQA datasets, comparing KADA with other parameter-efficient fine-tuning methods, including QLoRA, AdaLoRA, and BitFit. DHKR began with a single subnetwork and expanded dynamically as needed. Results showed that while DHKR’s initial performance was slightly lower due to gradual growth, it achieved highly stable accuracy, outperforming KADA. Despite having more trainable parameters, DHKR trained faster per iteration than other methods.

### Strengths
- DHKR achieves a lower Expected Calibration Error compared to KADA (0.02 vs. 0.13), showing that its predictions remain well-calibrated across domains.


- Despite having more trainable parameters, DHKR trains 2–3× faster per iteration than AdaLoRA and BitFit. 


- Its dynamic, two-level subnetwork expansion allows the model to grow capacity.

### Weaknesses
- The problem statement is not clearly defined. It would be better to articulate a precise problem that DHKR is meant to solve, such as inefficiency, instability, or failure mode in KADA. 
- While the authors claim that DHKR trains faster than other PEFT methods despite having more trainable parameters, it is unclear whether this improvement is from algorithmic innovations or from system-level optimizations, like kernel fusion and reduced precision casting. Therefore, it is hard to understand the contribution of this paper. 
- The accuracy of the proposed method is roughly on par with (or below) KADA, depending on epoch. The experiments do not report confidence intervals or significance, making it hard to judge the robustness of the trade-offs.

### Questions
- Were results averaged across multiple runs or random seeds, and are the reported differences statistically significant?
- Can the authors provide insight into what knowledge each hierarchical subnetwork learns as the model grows?

### Soundness
2

### Presentation
2

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
This paper addresses a key limitation in the PEFT method, KADA. The authors identify that KADA's reliance on a static, predefined set of subnetworks limits its ability to adapt to unseen domains. To overcome this, they propose Dynamic Hierarchical Knowledge Routing (DHKR), a novel framework inspired by Mixture-of-Experts (MoE) routing principles. DHKR extends KADA by introducing a two-level subnetwork growth mechanism that dynamically expands both high-level subnetworks (K) and their second-level components (L) on demand. The authors claim this transforms KADA's static mechanism into a self-adaptive, robust system.

### Strengths
- The core contribution is the design of a dynamic, hierarchical PEFT framework. This directly addresses a clear and important limitation in the prior KADA method and offers a more flexible approach to knowledge-aligned domain adaptation.
- The methodological design is quite principled.  The authors have incorporated sophisticated stability mechanisms, namely the composite growth trigger (monitoring stagnation, entropy, imbalance, etc.) and the multi-level Loss-Free Balancing.
- The detailed experiments look good.

### Weaknesses
- The main weakness of DHKR, in my opinion, is its complexity (both hyperparameters and overhead). Regarding that, I had a few questions: How difficult is it to tune this system? The effectiveness seems highly dependent on getting the thresholds for the composite trigger correct. Furthermore, could the authors please quantify the training and inference overhead (e.g., FLOPs, latency, or wall-clock time) of DHKR? How does it compare to the static KADA-5 baseline and QLoRA?
- The main KADA baseline is fixed at K=5 subnetworks. This might not be a fair comparison. It is unclear if DHKR's advantage comes from its dynamism or simply from its ability to grow to a larger total capacity. How does DHKR's performance and final subnetwork count (K, L) compare to a stronger static baseline, such as KADA with K=10 or K=15?
- The composite growth trigger is a central contribution, but its components are not ablated. It would be valuable to see an ablation study that justifies this complex design. For example, what is the performance if growth is triggered only by stagnation or only by usage imbalance?

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
