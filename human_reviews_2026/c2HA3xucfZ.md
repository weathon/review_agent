# UpSafe℃: Upcycling for Controllable Safety in Large Language Models

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
Large Language Models (LLMs) have achieved remarkable progress across a wide range of tasks, but remain vulnerable to safety risks such as harmful content generation and jailbreak attacks. Existing safety techniques---including external guardrails, inference-time guidance, and post-training alignment---each face limitations in balancing safety, utility, and controllability. In this work, we propose UpSafe℃, a unified framework for enhancing LLM safety through safety-aware upcycling. Our approach first identifies safety-critical layers and upcycles them into a sparse Mixture-of-Experts (MoE) structure, where the router acts as a soft guardrail that selectively activates original MLPs and added safety experts. We further introduce a two-stage SFT strategy to strengthen safety discrimination while preserving general capabilities. To enable flexible control at inference time, we introduce a safety temperature mechanism, allowing dynamic adjustment of the trade-off between safety and utility. Experiments across multiple benchmarks, base model, and model scales demonstrate that UpSafe℃ achieves robust safety improvements against harmful and jailbreak inputs, while maintaining competitive performance on general tasks. Moreover, analysis shows that safety temperature provides fine-grained inference-time control that achieves the Pareto-optimal frontier between utility and safety. Our results highlight a new direction for LLM safety: moving from static alignment toward dynamic, modular, and inference-aware control.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces UPSAFE°C, a method that improves the safety of pretrained LLMs by upcycling harmful latent concepts into benign ones using residual stream alignment. The approach is efficient, interpretable, and shows strong refusal performance across various base models and refusal datasets.

### Strengths
1. **Strong Empirical Results.** It shows strong performance across both standard Large Language Models (LLMs) and Large Reasoning Models (LRMs).
2. **Practicality.** The method is practical—it requires minimal model modification and preserves both safety and utility.
3. **Detailed Analysis.** The paper provides thorough ablations and layer-wise visualizations, offering clear insights into how and where safety improvements occur.

### Weaknesses
1. **Limited Scope of Evaluated Attacks.** The paper evaluates UPSAFE°C against WildJailbreak, JBB, and StrongReject, but does not test against sophisticated jailbreak methods such as PAIR [1] or ReNeLLM [2]. Evaluation on these stronger attack vectors would better assess robustness.
2. **Weak Baseline Comparisons.** The baselines are limited to vanilla models, SFT-only training, and a single-stage MoE variant. Comparisons against recent defense methods, particularly representation-level [3,4] and reasoning-based [5] safety techniques, would better position the method's effectiveness.

If my concerns are addressed, I’m willing to raise my score.

[1] Chao, Patrick, et al. "Jailbreaking black box large language models in twenty queries." 2025 IEEE Conference on Secure and Trustworthy Machine Learning (SaTML). IEEE, 2025.\
[2] Ding, Peng, et al. "A wolf in sheep's clothing: Generalized nested jailbreak prompts can fool large language models easily."NAACL (2023).\
[3] Zou, Andy, et al. "Improving alignment and robustness with circuit breakers." NeurIPS (2024).\
[4] Yousefpour, Ashkan, et al. "Representation bending for large language model safety." ACL (2025).\
[5] Jeung, Wonje, et al. "SAFEPATH: Preventing Harmful Reasoning in Chain-of-Thought via Early Alignment." NeurIPS (2025).

### Questions
1. The paper identifies safety-critical layers empirically through linear probing but does not explain whether there is a consistent pattern for why safety signals concentrate in specific layers.
2. Some utility metrics increase after applying UPSAFE°C. Could the authors explain why performance improves in these cases? (Table 1)
3. According to Figure 4, accuracy drops significantly as τ increases, while Table 1 shows strong utility performance at τ=0.5. Also, where is the vanila model in Figure 4?

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
4

### Summary
This paper introduces UPSAFE°C, a unified framework designed to enhance the controllable safety of Large Language Models (LLMs) through safety-aware upcycling, addressing the limitations of existing safety techniques in balancing safety, utility, and controllability. The framework first identifies safety-critical layers in pre-trained LLMs via a Safety Sensitivity Score, and upcycles these layers into a sparse Mixture-of-Experts (MoE) structure. In this MoE setup, a router acts as a "soft guardrail" to selectively activate the original Multi-Layer Perceptrons (MLPs, serving as general experts) and additional safety experts copied from the original MLPs. To strengthen safety discrimination while preserving general capabilities, a two-stage Supervised Fine-Tuning (SFT) strategy is employed: the first stage trains safety experts and the router on harmful data, and the second stage trains only the router on a mixed dataset of harmful and benign data to refine its input discrimination. A safety temperature mechanism is further introduced for inference-time control, dynamically adjusting the safety-utility trade-off by biasing router logits and scaling probabilities.
Experiments across multiple benchmarks, model types, and scales demonstrate that UPSAFE°C achieves robust safety improvements while maintaining competitive general performance. Ablation studies confirm the efficacy of the proposed mechanisms. The safety temperature mechanism enables fine-grained control, allowing the model to approximate the Pareto-optimal frontier between safety and utility, marking a shift from static alignment to dynamic, modular, inference-aware LLM safety control.

### Strengths
- Introduces a targeted safety-critical layer scanning mechanism using the SS-Score, which efficiently identifies layers most responsive to safety signals—this avoids unnecessary parameter tuning across all layers and lays a precise foundation for subsequent upcycling.
- The framework’s two-stage SFT strategy effectively specializes safety experts in mitigating harmful content while enabling the router to discriminate between benign and harmful inputs, striking a robust balance between safety enhancement and general capability preservation.
- The novel safety temperature mechanism provides fine-grained, dynamic control over the safety-utility trade-off at inference time, allowing the model to approximate the Pareto-optimal frontier and adapt to diverse safety requirements without retraining.
- Experiments demonstrate strong generalization across model types, scales, and challenging benchmarks, outperforming baselines in both safety improvements.

### Weaknesses
- The evaluation part lacks comparative evaluations against established fine-tuning-based safety enhancement baselines or simple defense techniques like filtering by judgers, which limits the verification of its effectiveness in outperforming widely adopted safety techniques.
- As a practically oriented framework, the manuscript does not provide analyses of application costs (e.g., training/inference time overhead) or assess the additional adaptation efforts required to integrate its MoE-upcycled structure and safety temperature mechanism into modern inference frameworks, leaving uncertainty about its efficiency advantage over simpler safety methods.
- The experimental results include numerous perfect scores (100.00), which are uncommon for models like the Qwen series on standard safety datasets—especially given potential dataset biases. Without detailed explanations for these unusually stable high values (e.g., dataset distribution characteristics or evaluation metric artifacts), the reliability of such results remains unsubstantiated.

### Questions
For questions, please kindly refer to the previously identified weaknesses, which highlight several issues that require further resolution or explanation. As this paper presents a technically solid framework, addressing these unresolved issues would help enhance the study’s rigor and persuasiveness and may potentially contribute to a higher score.

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
4

### Summary
This paper proposes UpSafe, a framework for upcycling large language models (LLMs) to ensure consistent safety alignment across multiple instruction-tuning or post-finetuning stages. The authors observe that safety degradation frequently occurs when models undergo continued training (e.g., domain adaptation, capability enhancement), because existing safety fine-tuning methods (e.g., DPO, SFT) operate in isolation and fail to preserve alignment consistency across evolving model generations.

### Strengths
1. The paper introduces the concept of safety upcycling, which is a creative and underexplored idea—treating alignment as a cumulative inheritance process rather than one-time optimization.

2. The paper is clearly written and well-structured.

3. The proposed method addresses a critical real-world challenge: the loss of safety alignment during continual or multi-stage training—a widely recognized issue in the deployment of foundation models.

### Weaknesses
1. Limited novelty beyond structural adaptation. The core idea—analyzing or enforcing safety alignment at the safety layer—has appeared in prior works. This paper primarily transfers that analysis into a Mixture-of-Experts (MoE) architecture, rather than introducing a fundamentally new alignment principle.

2. Potential trade-off in temperature-based defense for complex reasoning tasks. The proposed temperature-scaling defense is effective in simple safety-sensitive or low-reasoning contexts. Still, it could significantly degrade performance on hard reasoning or mathematical problems, where token-level uncertainty is naturally higher.

### Questions
See Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes UPSAFE°C, a safety-alignment method that upcycles a dense transformer into a sparse MoE by identifying safety-critical layers, duplicating their MLP blocks into safety experts, and introducing a learnable router to gate between general and safety experts. The training is done in two stages: first, fine-tuning the safety experts on harmful data, then training the router on a mixture of harmful and STAR-benign prompts. At inference time, a “safety temperature” controls how much the router favors safety experts versus the general expert, aiming to enable controllable safety–utility trade-offs without retraining. The approach is framed as lightweight and modular, offering adjustable refusal strength at runtime.

### Strengths
The proposed method is conceptually neat: instead of retraining the entire model, it inserts a sparse MoE structure into safety-critical layers and uses routing as a soft guardrail. The idea of tuning a temperature at inference for dynamic safety–utility control is intuitive and flexible. The method is orthogonal to traditional gradient-based alignment methods, potentially allowing fast safety interventions without modifying the main model. It also engages with an increasingly relevant direction—how to build modular, controllable safety mechanisms inside model architectures rather than relying solely on external filtering.

### Weaknesses
1. The evaluation omits several widely used and stronger baselines, including Antidote: Post-fine-tuning Safety Alignment for Large Language Models Against Harmful Fine-tuning Attack, Safe LoRA: the Silver Lining of Reducing Safety Risks when Fine-tuning Large Language Models, Shape it Up! Restoring LLM Safety during Finetuning via STAR-DSS, Safety Alignment Should Be Made More Than Just a Few Tokens Deep, and Lisa: Lazy Safety Alignment for Large Language Models against Harmful Fine-tuning Attack. More importantly, the paper does not compare against a simple guardrail + rejection sampling baseline, which remains standard and surprisingly effective in practice.
2. There is no quantitative discussion of how many parameters are added through expert duplication and router layers. This is crucial because adding multiple safety experts per layer can substantially increase model size and memory footprint, undermining claims of lightweight deployment.
3. Missing capability–safety trade-off evaluation: The safety expert training uses only harmful and STAR benign data, but no general capability data (e.g., math or reasoning tasks). This makes it impossible to assess whether MoE upcycling slows learning or harms utility compared to dense fine-tuning. STAR benign prompts are too narrow, which explains the high over-refusal rate observed — the router learns a conservative boundary that over-activates safety experts.
4. Limited applicability to MoE backbones: The method assumes a dense base model. If the base model is already MoE (e.g., Mixtral, DeepSeek-MoE), it is unclear whether to (a) add new safety experts or (b) repurpose existing ones, and either approach would require router retraining and disrupt expert balancing. The paper does not discuss this scenario, which is critical for modern production-scale models.
5. Aadaptive attacks: The router can be bypassed or manipulated through adaptive prompt strategies. For example: A malicious prompt can be crafted to appear benign at the start, routing to the general expert and bypassing safety. Adversarial fine-tuning could shift router decision boundaries. Mixed benign–harmful inputs could remain in the general expert’s path throughout the rollout.
6. Unclear RLHF training results: The method is only evaluated under SFT. In reinforcement learning settings, router activation is dynamic and not guaranteed to consistently route harmful tokens to safety experts. RL can easily shift router distributions or deactivate safety pathways over training, fundamentally breaking the controllability guarantee.
7. Scalability is unclear: Upcycling requires per-layer safety-critical analysis and expert duplication. This process must be repeated for each new model or checkpoint, unlike simple fine-tuning or guardrail filtering which scale more naturally across deployments.

### Questions
See above

### Soundness
3

### Presentation
3

### Contribution
2
