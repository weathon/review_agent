# CMQuant: A Quantization-Aware Parameter-Efficient Fine-Tuning Framework for 4-Bit Consistency Models

- Decision: Reject
- Scores: 4, 6, 4

## Abstract
Consistency Models (CMs), built on diffusion models, use model state trajectory fitting to reduce iterations required for sample generation. However, they still maintain high per-iteration computational costs and large model parameter sizes, which hinder deployment on resource-constrained devices. Quantization, an effective model compression technique with notable success in Large Language Models, remains largely unexplored for CMs. We observe that the unique characteristics of CMs pose significant obstacles to effective quantization. First, the trajectory fitting errors inherent to CMs accumulate across iterations and are further amplified when quantization errors are involved. Second, CMs training often relies on Low-Rank Adaptation (LoRA), which injects low-rank matrices into specific layers without altering pretrained weights. However, quantization errors not only disrupt this initialization consistency, but also hinder training optimization and impair convergence. In this paper, we propose CMQuant, a novel quantization-aware parameter-efficient fine-tuning framework tailored for CMs. CMQuant introduces three innovations: (1) Trajectory Distillation with Phased Targets (TDPT), which assigns distinct optimization objectives to different stages of trajectory, enables accurate starting points for each stage and thereby minimizes accumulated quantization errors in iterations. (2) Hessian-Guided SVD-Initialized LoRA (HGS-LoRA), which leverages hessian-guided matrix decomposition to initialize LoRA, directing weight updates along quantization-friendly paths and thereby reducing quantization errors. (3) Quantization-Aware Rank Adaptation (QRA), which assigns ranks adaptively based on the degree of variation in activations and weights across different CMs layers. This minimizes the impact of quantization without increasing the total number of LoRA parameters. By integrating quantization into the CMs training, CMQuant achieves the first 4-bit quantization of CMs for both weights and activations. Experiments show that CMQuant outperforms SOTA at least FID↓/PickScore↑/IC↑ of 4.74/11.61/3.01 on FLUX. Furthermore, it improves throughput by 1.71×/3.43× on SDXL/FLUX, with only 27\%/25\% memory footprint.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes CMQuant, a quantized parameter efficient fine-tuning method for consistency models. 
CMQuant consists of three parts: (1) two-step fine-tuning with different objective; (2) activation-induced low-rank approximation to compensate weight quantization error (3) adaptive rank allocation for adapters.

### Strengths
The motivations, problem statement, and method are clear. For example, the observation on weight & activation statistics justifies of need of adaptive rank allocation.

### Weaknesses
- The contribution of the paper is unclear, especially considering recent related works not mentioned in the paper
  - The second contribution of the work, i.e., HGS-LoRA, which is an analytical activation-induced low-rank approximation, has been proposed in other works, for example, [QERA (ICML2025)](https://openreview.net/forum?id=LB5cKhgOTu)  has exactly the same solution, and [Caldera (NeruIPS2024)](https://proceedings.neurips.cc/paper_files/paper/2024/hash/a20e8451ffb07ad25282c21945ad4f19-Abstract-Conference.html) has a similar case where activation error is additionally considered.
  - The last contribution of the work, which relocates lora ranks across layers, has also been explored in other works, e.g., [AdaLoRA (ICLR2023)](https://arxiv.org/abs/2303.10512), [LQ-LoRA (ICLR2024)](https://openreview.net/pdf?id=xw29VvOMmU), etc.
- Unclear experiment design.
  - Table 1 and table 2 seems mixing the comparison of fine-tuning-based methods with post-training quantization methods? For example, in table2, SVDQuant is post-training quantization method, but CMQuant is fine-tuning based.

### Questions
Please answer the following questions in addition to the Weakness section
- Does CMQuant use any techniques/tricks to mitigate quantization error in activations before the fine-tuning starts? My understand is that HGS-LoRA is an weight initialization method that compensate the quantization error of weights.
- What's the number format used in this work? INT4? 
- QLoRA seems a bit outdated as the baseline in Table 1 and Table 2. Could the author compare against newer quantized parameter-efficient fine-tuning methods?
- The claim of "most existing methods focus solely on weight quantization while neglecting activation quantization, leaving the model with substantial computational overhead" may be inaccurate. My understanding is that activation quantization for Transformer is also an activate research area.

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
This paper tackles low-bit quantization of Consistency Models (CMs) and proposes CMQuant, a single-stage training scheme that jointly performs CM distillation and quantization-aware LoRA adaptation. The key idea is a trajectory-aware objective (TDPT) that aligns early steps to a frozen, untrained quantized copy of the CM to anchor the student in the quantization space, and aligns late steps to the FP16 CM to preserve endpoint quality. The parameterization uses HGS-LoRA, which splits weights into quantization-sensitive principal components (frozen and quantized) and a trainable LoRA on the residual subspace, and QRA to adapt LoRA rank per layer under a fixed parameter budget using activation dynamics and spectral cues. Empirically, the method reports the first W4A4 (weights and activations) CM with minimal quality regression and tangible speed/memory gains over PTQ and QAT baselines on SDXL/FLUX-style CMs.

### Strengths
1. Addresses a real gap—W4A4 quantization for CMs with multi-step error compounding.
2. Conceptual insight: using a frozen, untrained quantized teacher early provides a “low-bit anchor” rather than redundant self-copying, improving stability across steps.
3. Strong empirical results: maintains generation quality while enabling low-bit inference (reported W4A4) with measurable throughput and memory benefits.

### Weaknesses
1. The paper does not explicitly state whether LoRA is merged into the base and re-quantized for inference; the W4A4 claim implies it, but the procedure should be made explicit.
2.  QLoRA/QALoRA ranks (r) and parameter budgets are not reported in experiments; fairness may be affected without matching total LoRA capacity/compute.
3. QRA specifics: criteria, thresholds, and sensitivity are heuristic in description; needs ablation vs fixed-r, and analysis of stability/sensitivity to the allocation policy.
4. The speed evaluation primarily compares against the 20-step FP16 diffusion teacher rather than a like-for-like 4-step FP16 CM baseline, conflating fewer-step sampling with low-bit kernel effects and potentially overstating the acceleration attributable to quantization itself.

Minor Weaknesses
a）Inconsistent terminology/acronyms: the paper alternates between “Trajectory Distillation with Phased Targets (TDPT)” and “Trajectory Distillation with Adaptive Targets (TDAT)”
b) The main text cites “Appendix A.4” for the single-step error discussion, but the corresponding section in the appendix is mislabeled as a second “A.3”

### Questions
1.  Do you merge the learned LoRA into the base weights and then re-quantize for deployment? If not, how is W4A4 preserved without a high-precision branch?
2. What LoRA ranks (r) and total parameter budgets were used for QLoRA/QALoRA? Can you report sensitivity to r and ensure matched budgets/compute?
3. HGS-LoRA details: how is the Hessian approximated (e.g., Hutchinson/K-FAC/diag Fisher), and how is the principal subspace/rank k chosen per layer?
4. TDPT schedule: how are early vs late steps defined and weighted? Is the split static or adaptive? Ablate the split point and the loss weights; compare to using only FP teacher vs only quantized teacher.
5. Could you add a direct comparison against an FP16 CM (same architecture, same number of steps/resolution/batch) in terms of latency and throughput to isolate the net gains from low-bit quantization? Also, please clarify whether inference actually uses W4A4 kernels with activation quantization enabled, and specify the exact hardware, libraries/kernels, and batch settings used.

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
2

### Summary
This paper proposes a quantization-aware PEFT framework named CMQuant for CMs in W4A4. The authors identify two main challenges when applying quantization to CMs: (1) quantization errors accumulate and amplify across inference iterations due to the trajectory fitting nature of CMs, and (2) existing LoRA-based training is incompatible with quantization as it disrupts initialization consistency. Three contributions, TDPT, HGS-LoRA and QRA, are proposed to address the problems, and experimental results show improvements in throughput and memory savings.

### Strengths
* Applying quantization to CMs is an interesting and novel problem.

### Weaknesses
* It feels like this work is stitching three pieces together, making novelty a bit weak.
* Writing could be improved. For example, Figures 1 and 2 are not mentioned in the text, making the motivation confusing.
* How do TDPT, HGS-LoRA, and QRA interact and affect the results?
* Table 6 misses comparison with other related works.
* The datasets used are limited - how about evaluation on other datasets (e.g., ImageNet)?

### Questions
* In Figure 1 and Table 1, CMQuant seems to be close to TCD. But TCD is missing in Table 2. Could the authors explain why?
* Could the authors provide justifications for the selection of parameters (e.g., T' and r)? How sensitive are they?

### Soundness
2

### Presentation
2

### Contribution
2
