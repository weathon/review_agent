# Balancing the Experts: Unlocking LoRA-MoE for GRPO via Mechanism-Aware Rewards

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 6

## Abstract
Parameter-efficient Mixture-of-Experts (MoE) architectures, such as LoRA-MoE, enable strong and generalizable fine-tuning. However, a critical problem arises when fine-tuning these architectures with advanced reinforcement learning algorithms such as Group Relative Policy Optimization (GRPO). Traditional supervised techniques are not naturally compatible with
the GRPO  objective, and naive combinations fail to effectively address routing collapse and the underutilization of MoE adapter parameters. To resolve this disconnect, we introduce Routing-Optimized Group Relative Policy Optimization (RO-GRPO), a mechanism-aware framework. It turns internal expert routing statistics collected during training into a direct reward signal, seamlessly integrating routing supervision into the reinforcement fine-tuning (RFT) process. This enables effective optimization of parameter utilization and improves performance on both unimodal and multimodal mathematical reasoning tasks, all without extra training stages. Our work provides the first demonstration that a scalar reward in GRPO can be engineered from a model's own internal mechanics to explicitly guide its optimization, extending alignment from mere behavior tuning to holistic mechanism alignment.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents a technique called RO-GRPO, which maximizes utility of experts of MoE through LLM-RL training. RO-GRPO adds an additional rewards based on routing entropy and expert utilization to explicitly encourage the MoE model to efficiently use experts. The paper introduces two variations, RO-GRPO (Smooth) and RO-GRPO (Relative). Smooth variation schedules weights of additional reward terms while Relative variation computes the reward signal based on improvements over historical baselines, which eliminates the manual scheduling mechanics. In experiments, both variations outperforms a baseline and especially the Relative variation shows strong performance improvements.

### Strengths
- The proposed idea is simple and easy to integrate to existing MoE training. And, the Relative variation eliminates manual balancing to tune performance, which makes this approach very practical.
- The performance improvement is significant especially when increasing the number of experts, which proves its effect.
- Through extensive analysis, how individual changes contribute behavior of MoE training is clarified in the paper.

### Weaknesses
- Although the proposed idea sounds sane, a straightforward alternative approach would be to use the routing entropy and the expert utilization as loss functions, which could directly optimize those metrics. I found the similar description in Appendix E.1, however this seems vital baselines to makes more sense to use the proposed idea. Lack of this analysis keeps my score as it is.
- Analysis of models' outputs with the proposed change should be interesting, but there isn't much observation provided in the paper except Appendix D. Since the proposed model can utilize experts more, it would be interesting to observe different trends in output texts. This analysis will improve the manuscript.

### Questions
- Computing the route reward adds any additional computing overheads?
- Do we need to balance scaling between the main task reward and the route reward? Can we just use the route reward with its raw scale?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies the expert collapse and underutilization problem in reinforcement fine tuning of LORA-MOE architectures. The authors proposed a routing confidence and a load balancing reward. The two rewards are combined and summed with the task reward using a weight curriculum. Then the combined reward is optimized using the GRPO algorithm. Experiments on conducted on several math reasoning benchmarks. The authors show the effectiveness of their method for controlling expert confidence and load balance. It was further shown in ablations that removing the proposed mechanisms degrade the performance of LORA-MOE fine tuning.

### Strengths
* The presentation of the idea is pretty clear. 
* The ability of the proposed method to control entropy and load balance is clearly pretty effective from the plots.

### Weaknesses
* While the method is effective at controlling entropy and load balance, it is not clear that it improves performance for more experts, which I belief is the ultimate goal for this family of methods. In fact from the results tables, 8 experts is almost never the overall best. 
* While integrating these objectives into the reward function seems to unify the training method, it is not clear from the paper that simply adding auxiliary losses to the RL objective is fundamentally flawed or significantly worse.

### Questions
* Have you considered adding auxiliary loss to the simple RL objective as a baseline? Could you discuss this option or show some experiments?
* Although it's likely intuitive, could you explain why applying the combined reward on the entire policy rather than applying the entropy and load balancing reward to only the routers is so effective? And why not the latter option?

### Soundness
3

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
This paper addresses the challenge of applying LoRA-MoE architectures to reinforcement learning fine-tuning RFT, specifically GRPO. The authors state that traditional auxiliary losses used to guide expert routing in supervised settings are incompatible with GRPO's scalar reward framework, leading to routing collapse and poor expert utilization. They propose RO-GRPO, which transforms internal routing statistics into reward signals through two components: a confidence reward encouraging low-entropy routing and a balance reward promoting uniform expert utilization. Two variants are introduced, (i) Smooth (curriculum-based scheduling) and (ii) Relative (binary reward for simultaneous improvement). Experiments on mathematical reasoning tasks using Qwen2.5 models demonstrate improvements in both task performance and routing efficiency across numerous expert counts.

### Strengths
- The paper tackles a meaningful problem at the intersection of parameter-efficient fine-tuning and reinforcement learning, with practical implications for deploying and adapting large models.
- The proposed approach is technically sound, integrating routing supervision directly into the reward signal without requiring architectural modifications or additional training stages.
- The experimental design is comprehensive, covering both unimodal and multimodal mathematical reasoning tasks with consistent methodology across multiple benchmarks.
- The theoretical grounding connecting the approach to constrained optimization and information bottleneck principles adds depth to the empirical contributions.

### Weaknesses
- The scope of evaluation is limited to adapter-based MoE architectures, leaving unclear whether the identified routing pathologies and proposed solutions generalize to full MoE models such as Phi-3.5-MoE or OLMoE.

- The design choice to formulate routing supervision as rewards rather than auxiliary losses is not adequately justified. Given that GRPO already employs KL divergence as a regularization term, it is unclear why routing constraints cannot be incorporated similarly as loss terms.

- The experimental comparisons lack a crucial baseline: a standard LoRA configuration with parameter count matching the LoRA-MoE variants, making it difficult to isolate whether performance gains stem from the routing-aware rewards or simply from increased model capacity.

- Tables 2 reveal several instances where the vanilla GRPO (LoRA) baseline with approximately one-quarter the trainable parameters outperforms RO-GRPO variants, undermining claims of consistent improvement.

### Questions
- Could you provide experimental results on full MoE architectures rather than adapter-based variants? This would clarify whether routing collapse is inherent to MoE structures under GRPO or specific to the LoRA-MoE configuration.

- What happens when you include a GRPO baseline using standard LoRA with trainable parameters matching the LoRA-MoE models (e.g., approximately 127M parameters for the E=8 configuration)?

- Can you provide qualitative analysis or case studies demonstrating what different experts actually learn? Do they specialize in different mathematical reasoning strategies, problem types, or computational roles?

- Why must routing supervision be formulated as a reward signal rather than an auxiliary loss term? Could you compare your approach against a variant that incorporates routing entropy and load balancing as loss terms alongside the GRPO objective?

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
The paper addresses a mismatch when fine-tuning LoRA-MoE models with GRPO: the external task reward is unaware of internal expert routing, leading to routing collapse and poor MoE utilization. The proposed RO-GRPO augments the reward with a mechanism-aware term $R_{\text{routing}}$ added to the task reward $R_{\text{task}}$. $R_{\text{routing}}$ combines (i) routing confidence (mean per-token routing entropy; lower is better) and (ii) load balance (MSE to uniform expert usage; lower is better). Two strategies are evaluated: a smooth curriculum that shifts emphasis from confidence to balance, and a relative scheme that rewards only when both metrics improve over a historical baseline. On uni- and multimodal math reasoning tasks, RO-GRPO improves accuracy and yields healthier expert utilization versus vanilla LoRA-MoE. Ablations indicate both components matter; a shuffled-control (permuting $R_{\text{routing}}$ within a batch) removes gains, supporting causal attribution.

### Strengths
1. Simple, novel, and practical: Converts internal routing statistics into a scalar reward, no architectural changes or auxiliary losses.  
2. Credible evidence: Informative ablations and a strong shuffled-control isolate the effect of routing-aware feedback.  
3. Clear exposition & tangible impact: Clean figures and a persuasive qualitative case link improved routing to better reasoning.

### Weaknesses
1. Parameter-budget confound (major): Baselines compare ~30M LoRA vs RO-GRPO variants up to ~127M trainables; needs budget-matched or FLOPs-normalized comparisons.  
2. Narrow task scope: All benchmarks are math-centric; generality to instruction following, summarization, dialogue, or safety is untested.  
3. Limited algorithmic scope & reporting gaps: Only GRPO is shown; PPO/DPO not demonstrated. Clarify entropy/balance aggregation and sensitivity; consider risks of “confidence” gaming.

### Questions
1. Budget parity: How does RO-GRPO compare to a high-rank LoRA baseline matched to the $E{=}8$ trainables (~127M)? Please report accuracy and routing metrics under parameter/FLOPs parity.  
2. Beyond math: Any preliminary results on instruction-following or summarization, and observations on expert specialization there?  
3. PPO/DPO adaptation: For PPO, can $R_{\text{routing}}$ be used as standard reward shaping without instability (need annealing/curriculum)? For DPO, is reweighting by $\Delta$routing (preferred minus rejected) viable?

### Soundness
3

### Presentation
3

### Contribution
3
