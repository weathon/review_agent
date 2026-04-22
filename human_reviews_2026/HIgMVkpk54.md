# Training-Free Reasoning and Reflection in MLLMs

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
Recent advances in Reasoning LLMs (e.g., DeepSeek-R1 and OpenAI-o1) have showcased impressive reasoning capabilities via reinforcement learning. However, extending these capabilities to Multimodal LLMs (MLLMs) is hampered by the prohibitive costs of retraining and the scarcity of high-quality, verifiable multimodal reasoning datasets. This paper introduces ANO Model (a temporal name for anonymous review), a training-free and R1-like MLLM that imbues off-the-shelf MLLMs with reasoning and reflection abilities, without any gradient updates or extra supervision. Our key insight is to decouple perception and reasoning across MLLM decoder layers. Specifically, we observe that compared to the deeper decoder layers, the shallow decoder layers allocate more attention to visual tokens, while the deeper decoder layers concentrate on textual semantics. This observation motivates a hierarchical weight merging approach that combines a visual-text pretrained MLLM with a reasoning-specialized LLM. To this end, we propose a layer-wise, Taylor-derived closed-form fusion mechanism that integrates reasoning capacity into deep decoder layers while preserving visual grounding in shallow decoder layers. Extensive experiments on challenging multimodal reasoning benchmarks demonstrate the effectiveness of our approach. On the MMMU benchmark, our model ANO-38B achieves an accuracy of 69.2, outperforming the strongest baseline InternVL2.5-38B by +5.3, and even surpasses the proprietary GPT-4o model.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
Built on top of task arithmetic, this paper leverages the finding that shallow decoder layers in MLLMs tend to focus on
visual perception and deeper layers on textual reasoning and proposed layer-wise task vector merging.  The paper derives a closed-form solution to guide layer-wise task vector merging and demonstrated significant model performance improvement on multimodal reasoning tasks.

### Strengths
Quality: principled way of using Taylor expansion to derive a closed-form optimal fusion strategy for each layer.


Significance: Proposes an effective and efficient way of combining non-reasoning MLLMs and reasoning LLMs into powerful MLLMs that have strong multimodal reasoning capabilities.

### Weaknesses
1.  By heavily weighting the text-based reasoning in deeper layers, the proposed method might have weakened native image-based reasoning pathways. The paper could be strengthened by more experiments and analysis on how the proposed method could affect model's image-based reasoning capability, especially on datasets where the prompt format is image.

2.  "shallow decoder layers focus on visual perception and deeper layers on textual reasoning, is identified and leveraged." could be strengthened by testing on more architectures and across model scales if it were to be included as one of the contributions.

### Questions
Could you help clarify what each color in figure 2(a), 4(a) and 5(a) represents?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces ANO, a training-free method for integrating reasoning and reflection capabilities into off-the-shelf Multimodal Large Language Models (MLLMs) by merging them with reasoning-specialized LLMs.  The approach is based on two key insights: (1) the functional specialization of shallow and deep decoder layers in MLLMs (visual perception vs. textual reasoning), and (2) the near-orthogonality of task vectors from vision and reasoning fine-tuning.  The authors propose a layer-wise, Taylor-derived closed-form fusion mechanism that optimally combines task vectors without additional training.

### Strengths
- The derivation of closed-form fusion weights using Taylor expansion and NTK linearization is rigorous and well-motivated.
- Comprehensive experiments across multiple benchmarks and model scales (8B to 38B) show consistent and significant improvements over strong baselines.
- The paper includes thorough ablations to validate the contribution of each component (e.g., modality priors, layer-wise fusion).

### Weaknesses
- The method relies heavily on the orthogonality of task vectors and the NTK linearization assumption, which may not hold universally, especially for smaller models or non-standard architectures. The authors are encouraged to provide results on more mainstream architectures such as LLaVA, Qwen, etc. Have the authors considered the potential negative impact of task vector interference when the orthogonality assumption is violated?  Are there fallback mechanisms?
- The approach is only validated on vision-language tasks.  Its applicability to other modalities (e.g., video, structured data) remains unverified.

### Questions
Please refer to the weaknesses part.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes ANO, a training-agnostic, R1-style multimodal model fusion method. ANO leverages task vectors and second-order Taylor/NTK linearization to derive closed-form fusion weights at every decoder layer. It fits a modality prior from the exponentially decaying visual attention across layers to preserve visual perception in shallow layers while injecting language reasoning and self-reflection capabilities in deeper layers, yielding a strong reasoning MLLM without additional data or training.

### Strengths
The method requires no gradient updates, reinforcement learning, or extra labeled data. Fusion coefficients are given in closed form, avoiding grid search and validation-set tuning. Under NTK linearization and approximate orthogonality of task vectors, a second-order Taylor expansion yields closed-form fusion weights whose dependence only on layer-wise task-vector norms is both concise and interpretable. Empirical results demonstrate successful transfer of R1-like reasoning and self-reflection behaviors.

### Weaknesses
1. A central insight that shallow layers handle perception while deep layers handle reasoning (Fig. 2) has already been articulated in [1], which diminishes the contribution.
2. The paper lacks discussion and comparison with closely related work. Both the proposed method and [2] reformulate differences between the merged model and task-specific models via Taylor approximation (paired with NTK linearization and high-dimensional approximate orthogonality) into a data-free computable objective. In essence, replacing the true loss in a local neighborhood with a low-order approximation. The assumptions (NTK regime, small perturbations, near-orthogonality) are shared. Although this paper uses a second-order expansion, the similarities are substantial; why is [2] not discussed?
3. Missing comparisons against recent model-merging baselines. What happens if the proposed merging method is replaced with alternatives such as DOGE [2], WUDI [3], or OptMerge [4]?
4. Performance on state-of-the-art MLLMs (e.g., Qwen2.5-VL and Qwen3-VL) is not reported.

[1] Bring Reason to Vision: Understanding Perception and Reasoning through Model Merging. ICML 2025.

[2] Modeling Multi-Task Model Merging as Adaptive Projective Gradient Descent. ICML 2025.

[3] Whoever Started the Interference Should End It: Guiding Data-Free Model Merging via Task Vectors. ICML 2025.

[4] Unifying Multimodal Large Language Model Capabilities and Modalities via Model Merging. arXiv2505.

### Questions
N/A

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes a training-free, layer-wise merge of two models: a vision-grounded MLLM and a text-reasoning LLM. Instead of fine-tuning, they combine weights per decoder layer using a closed-form rule derived from a local second-order (Taylor) approximation; shallow layers are biased to keep perception, deeper layers to inject reasoning via an attention-based prior fit from the model’s own layer-wise visual attention. The idea is simple to deploy (no gradients, no grid search) and shows notable gains on multimodal reasoning (e.g., 69.2 on MMMU with the 38B variant, beating strong open baselines and matching/exceeding proprietary systems on that benchmark).

My reservations are about assumptions and scope. The derivation leans on NTK-style linearization and near-orthogonality of task vectors; both can be only approximately true in finite multimodal transformers, and the paper itself shows performance drops when task vectors are less orthogonal. The modality prior is fit on generic images (e.g., MSCOCO) and most wins are on math/diagram-heavy benchmarks, so it’s unclear how robust the scheme is across architectures or domains that don’t share the same attention profile. In short: novel and practical, with strong initial results, but it should have stronger sensitivity checks (when assumptions weaken) and broader evaluations before concluding it generalizes widely.

### Strengths
- The paper replaces heuristic task-arithmetic with a per-layer merger derived from a second-order Taylor approximation under NTK-style linearization, yielding analytic coefficients (their Eq. (13), prior-weighted in Eq. (15)) that require no grid search or supervision.

- The attention-guided prior is fit to the model’s observed decay of visual attention across depth and then used to bias fusion via a simple exponential schedule—tightening the link between empirical signal and architectural choice.

- The authors compute per-layer cosine similarity between vision and reasoning deltas and show near-orthogonality (justifying the decoupled quadratic form), and they further stress-test cases where this assumption weakens, noting degraded performance—usefully delineating the method’s operating regime.

- Using MME, the full method (with the prior) stays close to a perceptual “upper bound” while improving reasoning, and contrasts cleanly with the ablation without the prior—making the impact on perception explicit and measurable.

### Weaknesses
- The closed-form per-layer coefficients rest on NTK-style linearization and an isotropic Hessian surrogate, but the paper provides no error bounds or diagnostics quantifying deviation from these ideals in finite-width, multimodal transformers; optimality can drift when curvature/anisotropy is non-negligible.

- The derivation effectively requires near-orthogonality between the vision and reasoning deltas; when vectors correlate, performance materially degrades, and the method lacks a correlation-aware fusion or mitigation strategy (e.g., whitening/rotation, damping).

- The exponential attention-decay prior is fitted on ~1k MSCOCO images and then applied to math/diagram reasoning; robustness under domain or encoder shifts is not established, making the prior potentially domain-sensitive outside the measured regime.

- Most wins are on MMMU/MMMU-Pro/MathVision/WeMath (text-in-image/diagram heavy). Absent broader benchmarks (document OCR beyond math, chart QA outside math, natural-image long-context VQA, shifted domains), generality remains unproven.

- Results are reported at single settings; there are no sensitivity surfaces versus global fusion strength or prior slope, nor curves versus induced task-vector correlation. Without these, practitioners lack clear guardrails for accuracy–perception trade-offs and failure modes.

### Questions
- Can you quantify how closely the multimodal decoder operates in the NTK-style linear regime assumed for your second-order Taylor derivation

- Your derivation and Eq. (13)/(15) rely on near-orthogonality of task vectors, yet Appendix stress tests show degradation when cosine similarity rises. Can you provide a correlation-aware fusion 

- how stable are the fitted decay parameters across domains (diagrams, charts, UI screenshots) and encoders? Please report per-domain priors and sensitivity of accuracy to the prior slope α.

- How would you fuse non-homologous backbones or differing tokenizer/vision-encoder stacks (e.g., alignment layers/adapters)? Any empirical evidence beyond the specific pairings in Appendix A.4?

### Soundness
2

### Presentation
3

### Contribution
3
