# RFD-LoRA: Robust Federated Distillation for LoRA Fine-Tuning under Heterogeneous and Adversarial Clients

- Decision: Reject
- Scores: 6, 2, 6

## Abstract
Federated learning (FL) with low-rank adaptation (LoRA) is attractive for efficiency but fragile compared to full-rank FL. We show three fundamental vulnerabilities: (i) aggregation and projection bias, since bilinear averaging of adapters misrepresents the true global update; (ii) adversarial amplification, where low-rank projections can magnify malicious perturbations; and (iii) Jacobian sensitivity, where small adapter changes trigger large gradient variation. Existing methods only mitigate these issues and require identical client ranks, limiting practicality. We propose Robust Federated Distillation for LoRA (RFD-LoRA), the first framework to combine federated distillation with LoRA. By aggregating logits in a shared subspace, RFD-LoRA totally eliminates aggregation and initialization lag while enabling clients with heterogeneous ranks and adapter structures to collaborate seamlessly. To defend against non-IID and adversarial clients, we design three modules: Confidence-Adaptive Temperature (CAT), MMD-based Distillation (MMD-KD), and Disagreement Suppression (DIS). We provide error bounds and show on GLUE benchmarks that RFD-LoRA consistently outperforms prior methods in accuracy and robustness.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes RFD-LoRA, a novel federated distillation framework that eliminates aggregation bias and supports heterogeneous LoRA configurations by operating in logit space. Through theoretical analysis and robust module design, the method achieves strong accuracy and robustness under both non-IID and adversarial client settings.

### Strengths
1. Supports heterogeneous LoRA ranks and structures, making the method realistic for real-world federated systems.

2. Eliminates aggregation and initialization bias by using logit-space distillation rather than bilinear adapter averaging.

3. Strong theoretical foundation, with new error bounds for projection bias and adversarial amplification.

### Weaknesses
1. Public anchor set selection is under-specified: It remains unclear how the choice of reference dataset $\mathcal{D}_{\text{ref}}$ affects performance, task-specific vs. task-agnostic? How sensitive is RFD-LoRA to anchor domain shift?

2. Lack of evaluated datasets: The study focuses only on classification (e.g., GLUE) and does not assess performance on generation tasks common in LLM settings.

3. Logit computation in Equation (4) could be elaborated further. Do clients average logits per sample?

### Questions
see above.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper argues that parameter space aggregation for federated LoRA is intrinsically fragile due to (i) aggregation/projection bias from bilinear LoRA factors, (ii) adversarial amplification that grows with sqrt{d/r}, and (iii) Jacobian sensitivity of the bilinear map BA. It proposes RFD LoRA, a logit space alternative that (a) has clients send logits on a public anchor set, (b) aggregates them with a Median of Means (MoM) rule, and (c) distills the aggregate into a global student using three robustness modules: Confidence Adaptive Temperature (CAT) to bound gradients, MMD based KD to match mean/variance of logits, and Disagreement Suppression (DIS) to down weight high variance anchors.

### Strengths
- Clear problem formulation of LoRA specific vulnerabilities (aggregation/projection bias, adversarial amplification, Jacobian sensitivity).

- Conceptual simplicity of moving to logit space aggregation to support heterogeneous ranks/structures and avoid bilinear averaging issues.

### Weaknesses
- There is a notable inconsistency between Eq. (4) and Algorithm 1 regarding which model parameters clients use each round. Eq. (4) defines client logits using the frozen base model W0, whereas Algorithm 1 describes logits computed from the evolving global model W. This ambiguity makes it unclear whether clients receive updated weights from the server after each round or continue to fine-tune on a fixed W0. 
- Table 2 claims 12 KB per round, “rank independent,” yet the training protocol states that clients upload logits for a public anchor set comprising 10% of the per-task training size. For GLUE tasks, 10% of the training data is thousands of examples; even with 2–3 classes and 32-bit floats, per client per round payload would be orders of magnitude larger than 12 KB. A precise accounting is needed; currently, the claim seems implausible.
- Foundational FD approaches, FedMD and ensemble distillation, already support heterogeneous models and logit space aggregation; they were also motivated by privacy/communication benefits. RFD LoRA takes this paradigm and adapts it specifically to LoRA FL, adding CAT/MMD KD/DIS and MoM aggregation. A quantitative comparison to these FD baselines is needed to substantiate the claimed gains.
-	The paper proposes sharing per-anchor logits but does not analyze potential information leakage through these representations. Prior works on FD have demonstrated susceptibility to membership inference and label-distribution inference when logits over public anchors are shared. The absence of any privacy-preserving mechanism (e.g., noise injection, quantization, or selective sharing) weakens the practicality of deploying the proposed method in privacy-sensitive FL scenarios.
-	Experiments are conducted with only 10 clients on GLUE benchmarks, with no timing, throughput, or end-to-end communication measurements. This limited scale does not convincingly demonstrate the claimed communication and scalability benefits.
-	The paper claims to “enable heterogeneous ranks and adapter structures,” but experiments vary ranks only. Please include a study where clients differ in adapter placements (e.g., Q/V only vs. Q/K/V, or different layers) and scales α to validate “adapter structure” heterogeneity.
-	Tables report means over 5 runs but no standard deviations/confidence intervals. Error bars matter given small absolute gaps (e.g., +1–2% CA/RA).

### Questions
1. Provide a precise formula and numbers for per round bytes plus any compression, sub-sampling, or sparsification. How does this yield 12 KB? If you sub-sample anchors per round, specify the schedule; otherwise, Table 2 appears inaccurate.
2.	Please report wall clock and server-side compute for distillation vs. parameter averaging baselines, and how anchor size affects latency/throughput.
3.	Provide results for FedMD and Ensemble Distillation baselines (with and without robust aggregation) to separate the benefit of “doing FD” from CAT/MMD KD/DIS.
4.	How does MoM compare to geometric median (RFA) in logit space? Any reason MoM was preferred beyond convenience?

### Soundness
3

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
The paper proposes RFD-LoRA, a robust and rank-flexible framework for federated learning with LoRA adapters that overcomes the fragility of conventional LoRA aggregation. Instead of averaging adapter parameters, which introduces aggregation bias, projection errors, and vulnerability to adversarial perturbations, RFD-LoRA performs logit-space distillation across clients, enabling collaboration among heterogeneous LoRA ranks. It further enhances robustness through three modules: CAT for stabilizing gradients, MMD-based Distillation  for aligning logit distributions, and DIS for mitigating non-IID effects.

### Strengths
* The paper’s key strengths lie in its conceptual novelty (moving LoRA aggregation to logit space)
* the authors provide clear diagnostic analysis of LoRA fragility, and its practical robustness framework combining distillation, adaptive temperature scaling, and disagreement suppression. 
* The approach is empirically validated, rank-flexible, and communication-efficient, making it a meaningful step toward robust, efficient federated fine-tuning. 
* Despite its heuristic components, it offers a well-motivated and pragmatically valuable contribution bridging theory and deployment.

### Weaknesses
* Although the paper claims to formalize LoRA fragility and provide theoretical error bounds, its analysis is largely heuristic and lacks rigorous theorems or proofs. The discussions of aggregation bias, projection bias, and adversarial amplification are insightful but remain descriptive, offering qualitative scaling arguments rather than formal guarantees. The work succeeds in identifying the sources of fragility conceptually but does not deliver provable robustness results or explicit analytical bounds supporting its claims.
* The proposed RFD-LoRA algorithm is a creative and practical integration of heuristic modules addressing known LoRA fragilities, but lacks formal grounding. Each component, CAT, MMD-KD, and DIS, is introduced intuitively without being derived from a unified optimization framework or robustness objective. Their combination relies on empirical tuning rather than principled theory. Even the median-of-means step, while statistically justified, is used heuristically in logit space without convergence guarantees. As a result, RFD-LoRA is better characterized as a robust heuristic pipeline than as a theoretically principled federated learning algorithm. It achieves robustness empirically, not provably.

### Questions
Refer to weaknesses

### Soundness
3

### Presentation
3

### Contribution
3
