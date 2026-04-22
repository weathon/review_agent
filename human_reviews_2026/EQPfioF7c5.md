# LoRAtorio: An intrinsic approach to LoRA Skill Composition

- Avg Score: 3.00
- Decision: Reject
- Scores: 2, 2, 4, 4

## Abstract
Low-Rank Adaptation (LoRA) has become a widely adopted technique in text-to-image diffusion models, enabling the personalisation of visual concepts such as characters, styles, and objects. However, existing approaches struggle to effectively compose multiple LoRA adapters, particularly in open-ended settings where the number and nature of required skills are not known in advance. In this work, we present LoRAtorio, a novel train-free framework for multi-LoRA composition that leverages intrinsic model behaviour. Our method is motivated by two key observations: (1) LoRA adapters trained on narrow domains produce unconditioned denoised outputs that diverge from the base model, and (2) when conditioned out-of-distribution, LoRA outputs show behaviour closer to the base model than when conditioned in distribution. In the single LoRA scenario, personalisation and customisation show exceptional performance without catastrophic forgetting; the performance, however, deteriorates quickly as multiple adapters are loaded.
Our method operates in the latent space by dividing it into spatial patches and computing cosine similarity between each patch’s predicted noise and that of the base model. These similarities are used to construct a spatially-aware weight matrix, which guides a weighted aggregation of LoRA outputs. To address domain drift, we further propose a modification to classifier-free guidance that incorporates the base model’s unconditional score into the composition. We extend this formulation to a dynamic module selection setting, enabling inference-time selection of relevant LoRA adapters from a large pool. LoRAtorio achieves state-of-the-art performance, showing up to a 1.3\% improvement in ClipScore and a 72.43\% win rate in GPT-4V pairwise evaluations, and generalises effectively to multiple latent diffusion models. Code will be made available.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper introduces LoRAtorio, a train-free framework for composing multiple LoRA adapters in text-to-image diffusion models. The method leverages intrinsic model behavior by computing patch-wise cosine similarity between LoRA outputs and the base model in latent space, using these similarities to weight contributions during denoising. It also proposes a modification to classifier-free guidance to mitigate domain drift and extends the approach to dynamic module selection for inference-time adaptability. Experiments on the ComposLoRA benchmark and Flux architecture show improvements in CLIPScore, GPT-4V evaluations, and human preference metrics compared to prior state-of-the-art methods.

### Strengths
Novelty: The idea of using intrinsic similarity for inference-time LoRA composition is original and avoids retraining, addressing practical constraints.

Comprehensive Evaluation: Includes automated metrics (CLIPScore), GPT-4V-based evaluation, and human studies across multiple datasets and architectures.

Dynamic Module Selection: Extends beyond static composition, which is relevant for real-world scenarios.

Clear Motivation: Observations about domain drift and latent divergence are well-supported by empirical analysis and visualizations.

Model-Agnostic Design: Demonstrates applicability to both Stable Diffusion and Flux architectures.

### Weaknesses
Computational Overhead: The method scales linearly with the number of LoRAs, making it impractical for large pools in dynamic settings. Latency (61–122s) is significantly higher than simpler baselines like Switch or Merge.

Limited Theoretical Depth: While cosine similarity is justified empirically, the theoretical motivation is relegated to an appendix and lacks rigorous formalism.

Evaluation Bias: Heavy reliance on CLIPScore and GPT-4V pairwise comparisons, which do not fully capture compositional fidelity or semantic correctness. Human evaluation is limited to three experts.

Failure Cases: The paper acknowledges severe failure modes (e.g., nonsensical outputs, concept confusion, duplicate limbs) but does not propose concrete mitigation strategies.

Assumption of LoRA Quality: The approach assumes LoRAs are well-trained and semantically coherent, which is unrealistic in community-driven repositories.

Scalability Concerns: Dynamic selection still requires loading all LoRAs into memory, which is infeasible for large-scale deployments.

Overstated Generalization: Claims of model-agnostic robustness are based on only two architectures; broader applicability remains unproven.

Ethical Section Superficiality: Mentions risks but lacks actionable guidelines or safeguards for misuse.

Generalizability: The methods have not been applied to large language models or multimodal large language models.

### Questions
How does LoRAtorio perform when LoRAs are trained on highly heterogeneous datasets with conflicting semantics?

Can the similarity-based weighting be approximated earlier in the pipeline to reduce computational cost?

How sensitive is the method to patch size and temperature hyperparameters in real-world scenarios?

Why was λ fixed at 0.5 for re-centering? Did you explore adaptive strategies?

Could metadata-driven pre-filtering or clustering of LoRAs improve dynamic selection efficiency?

How does the method handle cases where LoRA adapters introduce adversarial or biased features?

Is there any quantitative analysis of memory footprint for dynamic settings?

Would integrating learned gating (e.g., lightweight attention) outperform intrinsic similarity without full retraining?

### Soundness
2

### Presentation
2

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
This paper presents LoRAtorio, a train-free framework for composing multiple LoRA adapters in text-to-image diffusion models. The authors identify key challenges in multi-LoRA composition—namely, semantic drift and performance degradation as more LoRAs are added—and address these by leveraging intrinsic model behavior rather than external supervision or retraining. Their approach partitions the latent space into spatial patches and computes cosine similarity between LoRA and base model noise predictions, constructing a spatially-aware weight matrix for adaptive skill fusion. They also propose a modification to classifier-free guidance to mitigate domain drift. Furthermore, LoRAtorio supports dynamic module selection at inference time, selecting relevant LoRAs on-the-fly. Experimental results on the ComposLoRA benchmark and additional settings demonstrate that LoRAtorio achieves better performance in both static and dynamic scenarios, outperforming prior works across automated metrics (CLIPScore), GPT-4V evaluations, and human assessments

### Strengths
The paper demonstrates originality by proposing a train-free, intrinsically guided framework for multi-LoRA composition, departing from the reliance on weight merging or learned gating. The quality of the work is evident in the methodology, including spatial patch-based weighting, re-centered guidance, and dynamic module selection. The paper is clearly written, with effective visualizations and thorough empirical support.

### Weaknesses
While the paper presents an innovative and effective approach, there are several notable weaknesses that merit attention. First, the authors do not release their code, which hinders reproducibility and weakens the reliability of the claimed results. Second, the core mechanism—spatial patch-based weighting—raises concerns when dealing with heterogeneous LoRA types. For example, style-oriented LoRAs may introduce global stylistic shifts across all spatial regions, while object-specific LoRAs affect only localized areas. The current similarity-based weighting may fail to harmonize such differences, potentially leading to outputs where object identity is distorted by the base model's style or vice versa, contrary to the intended composition. Third, a key advantage of LoRA is the ability to merge multiple adapters at inference with negligible overhead, but LoRAtorio requires evaluating all adapters independently at each step, increasing inference cost significantly. The paper lacks a discussion or analysis of this added complexity, which could impact its scalability in real-world applications. Addressing these issues would strengthen both the practicality and theoretical grounding of the work.

### Questions
1. Given the complexity of the method, open-sourcing the code would be essential for reproducibility and to support broader adoption in the community.

2. Have the authors evaluated or analyzed the behavior of their patch-based weighting mechanism when composing LoRAs of fundamentally different types—e.g., one encoding global stylistic shifts and another modeling localized objects? In such cases, a patch that diverges from the base model may not necessarily indicate higher relevance. Can the authors provide qualitative examples or ablations showing the composition quality in these mixed scenarios?

3. Relatedly, is there a risk that the current weighting method leads to mismatched compositions—e.g., an object generated with correct shape but styled according to the base model, while the background reflects the intended LoRA style? If so, how might this be mitigated?

4. The proposed method computes conditional and unconditional predictions for each LoRA independently at every denoising step, which appears to scale linearly with the number of adapters. Could the authors clarify the runtime and memory impact of their method compared to weight-merging baselines?

5. In the dynamic module setting, how well does the method scale when the number of candidate LoRAs is very large (e.g., dozens or hundreds)? Is the top-k selection based purely on per-step cosine similarity stable across timesteps, or is there a risk of inconsistency during denoising?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
LoRAtorio is a train-free framework for composing multiple LoRA adapters in text-to-image diffusion models. It leverages the observation that LoRAs deviate from the base model on in-distribution inputs but remain close on out-of-distribution ones, using cosine similarity between LoRA and base outputs in latent space to guide spatially weighted aggregation. The method also introduces a modified classifier-free guidance term to mitigate domain drift and supports dynamic, inference-time selection of relevant adapters. LoRAtorio achieves state-of-the-art performance, improving CLIPScore and human-evaluated visual quality across multiple diffusion architectures.

### Strengths
1. The authors propose spatially-aware similarity metric to use as a proxy for LoRA adapter's confidence, with sound theoretical motivation.
2. The authors extend the task of multi-LoRA composition to a dynamic module selection setting, which is a good, real-world skill composition scenario.

### Weaknesses
1. The first contribution seems to be incremental - MultLFG (2nd best method) proposes "... training-free frequency-aware multi-LoRA merging. The key idea is to decompose LoRA-based noise predictions into frequency subbands and perform adaptive merging based on relevance scores." (https://arxiv.org/pdf/2505.20525), whereas this paper proposes patched cosine distance instead of frequency subbands.
2. The second contribution - re-centering - is, per your results in Table 6a, only better by 0.01 (36.543 with vs 36.532 w/o) CLIPScore, on a limited ablation study (see Weakness 3), which makes me believe it does not improve anything. What are the standard deviations for these results?
3. You compare your method to 8 reference methods in Table 1, just 4 in Table 2 and only 2 in the rest (Table 3, 4, 5c/d). The 2nd best performing method (per Table 1), MultLFG, is only shown once and never mentioned again. Why is that?
4. Ablation study of the proposed method (Table 6) lacks the same detail as, for example, Table 1. It only analyzes 2 or 3 component scenarios (missing 4 and 5), only on one subset (anime), on one backbone (stable-diffusion-v1.5).

### Questions
1. Could you extend the analysis of the weight matrix Ω? For example, I am wondering if taking the most dominant adapter's category per each patch could result in semantic masks appearing. Especially non global categories like character or object may be visible.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper presents LoRAtoria, a novel train-free framework for multi-LoRA composition that leverages intrinsic model behavior. 
The framework consists of two parts: skill composition on the patch level and re-centering of the unconditional noise output.
Also, the paper introduces MultiLoRA composition task with a dynamic LoRA selection

### Strengths
1. The paper is well structured and easy to follow.
2. The proposed approach demonstrates better results with increasing the active LoRA adapters
3. Re-centering of the unconditional noise could be used independently
4. Both UNet and DiT-based models are checked
5. The human and VLM-based evaluations are fully described
6. Extensive appendix

### Weaknesses
1. MultiLoRA composition task with a dynamic LoRA selection probably requires more detailed description as now it lacks motivation (at least some potential use cases)
2.The majority of the comparisons are done using CLIPScore that is a good proxy metric; however, a more extensive human or VLM-based evaluation is suggested
3. Only composition of LoRas for the Character, Style and Background are considered. No compositions with LoRAs for faster inference (e.g., LCM) are checked 
4. see questions

### Questions
1) inconsistent d:
* lines 213-214: "$d$ is the number of pixels per patch"
* lines 226-227 mention upscaling to $H/d \times W/d$
Please use $d^2$ as the number of pixels in patch or upscaling to $\sqrt(d)$ in the blocks description
2) the commonly used number of diffusion steps for SD1.5 is 30-50 steps (towards 30 if DPM++ solver is used); however, in the section 3.1 authors mentioned 100 steps for realistic subset and 200 steps for the anime subset without any further explanation.
3) The human evaluation mentioned only 3 experts. Have you considered running the quality assessment using GPT4v?
4) The results for the Rectified Flow are presented only on FLUX.1-dev checkpoints that is guidance distilled (re-centering couldn't be applied) while Stable Diffusion 3.5 is not checked.
5) Comparison with AutoGuidance is presented in the appendix; however, AutoLoRA(https://arxiv.org/abs/2410.03941) shows that AutoGuidance-ish approach could be combined with CFG for LoRAs
6) I believe that the skill composition process could be better illustrated

### Soundness
3

### Presentation
3

### Contribution
3
