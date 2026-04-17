# MoSE: Decoupled Tuning for Forgetting-Resilient Multi-task Fine-tuning of LLMs

- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
Integrating Low-rank Adaptation (LoRA) and Mixture-of-Expert (MoE) is the mainstream for applying LLMs in multi-task scenarios. Existing works assume that different experts can share common knowledge and hold the specific information dynamically. They employ router to select appropriate experts for different tasks. Despite the achieved progress,most of existing works still face the problem of cataclysmic forgetting of both common and specific information since they tuning the LoRA modules indiscriminately. The learned information in the LoRA modules from previous task might be overwritten by the finetuning of subsequent tasks. To tackle this problem, in this paper, we propose a novel Mixture of Shared and Exclusive Experts framework (MoSE) for better multi-task fine-tuning of LLMs. Different from most existing works, we first separate the LoRA experts into routing experts for task-specific information and shared experts for common knowledge. For routing experts, we develop a feature-wise module to select the most appropriate experts and tuning their parameters entirely. For shared experts, we aim to maintain as much common knowledge as possible.Thus, we design a novel Top-k-selection tuning strategy to selectively finetune certain parameters of shared experts. Then, we adopt expert assignment strategies to mitigate task imbalance and ensure fair expert utilization. Finally, extensive experiments over diverse multi-task scenarios demonstrate the effectiveness of our proposed MoSE.Moreover, MoSE exhibits strong continual learning ability, effectively adapting to new tasks while retaining prior knowledge.Moreover,MoSE exhibits strong continual learning ability, effectively adapting to new tasks while retaining prior knowledge (average 3.3\% and 7.4\% improvement compared with advanced baselines in sequential continual learning).

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces MoSE (Mixture of Shared and Exclusive Experts), a novel parameter-efficient fine-tuning framework for large language models in multi-task and continual learning scenarios. MoSE explicitly separates LoRA experts into shared experts for common knowledge and routing experts for task-specific information. The proposed framework employs three key techniques:
(1) a Feature-wise Linear Modulation (FiLM)-based router for task-aware expert selection,
(2) a Top-k gradient-based sparse update mechanism for stabilizing shared expert learning and mitigating catastrophic forgetting, and
(3) balanced and orthogonal regularization to ensure fair expert utilization and disentanglement between shared and routing experts.
Extensive experiments on GLUE, commonsense reasoning, and continual learning benchmarks (e.g., CSQA, RACE, SciTail) demonstrate consistent gains over strong baselines such as MultiLoRA, MixLoRA, and HydraLoRA.

### Strengths
1. Clear motivation and structure: The paper provides a solid motivation for decoupling shared and task-specific experts, supported by illustrative figures and detailed analysis.
2. Technical soundness: The proposed sparse Top-k update mechanism and orthogonality regularization are well justified and effectively address forgetting and interference.
3. Comprehensive experiments: Evaluation spans multiple benchmarks, backbones (T5, Qwen3-4B/8B), and includes ablations, parameter analysis, and visualization studies.
4. Practical relevance: The method maintains parameter efficiency comparable to existing PEFT methods while substantially improving performance on continual learning settings.

### Weaknesses
1. Insufficient explanation for large performance gains.Table 3 shows surprisingly large improvements over MixLoRA and MultiLoRA (e.g. +42.1% over LoRA in zero-shot transfer). The paper does not analyze why MoSE achieves such strong transfer performance—whether it comes from better routing, reduced interference, or simply more parameters.
2. Limited analysis of forgetting dynamics.Tables 4 and 5 demonstrate smaller forgetting scores, but the paper only provides final metrics. There is no temporal analysis (e.g., accuracy over task sequence) to justify why MoSE retains prior knowledge better than MixLoRA.
3. Missing discussion on computational efficiency. Although the authors emphasize “parameter efficiency,” MoSE introduces additional modules (shared expert, FiLM, orthogonality regularization). Training time, GPU memory, and inference latency are not reported, leaving its practical advantage uncertain.
4. Ablation and sensitivity analysis remain shallow. The ablation in Table 6 removes components one by one but does not explore hyperparameter sensitivity—such as the Top-k ratio, EMA decay, or orthogonality weight β. Without these, it is hard to judge whether the method’s stability depends on careful tuning.

### Questions
1. Can the authors report training cost and inference latency compared with MultiLoRA or HydraLoRA?
2. How sensitive is the final performance to the Top-k selection size or orthogonality weight β?
3. Does MoSE maintain its advantage on larger models (e.g., 14B, 32B) or longer-sequence tasks?

### Soundness
3

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
This paper studies multi-task and continual fine-tuning of LLMs under a parameter-efficient regime. The authors observe that in existing MoE-style or multi-LoRA settings, shared experts can be overwritten by later tasks because routing decisions do not preserve the functional role of experts and because all experts are often updated in a similar way. To address this, the paper proposes MoSE, a decoupled tuning framework. Routing experts are trained in a task-aware way (FiLM-style conditioning and top-k routing) so that routing capacity follows each task. Shared experts are updated in a sparse and importance-driven manner using EMA-based importance scores, together with load-balancing and a mild orthogonality constraint, so that the shared capacity is not easily destroyed by late tasks. Experiments on a set of NLP and reasoning benchmarks, including a sequential setting, show that MoSE improves over multi-LoRA, MixLoRA and other recent parameter-efficient multi-task baselines.

### Strengths
The motivation is concrete. The paper is not only saying that LoRA forgets but identifies the more specific failure case, namely that shared experts in a multi-task MoE style architecture lose their function when later tasks update them in an undifferentiated way. This is a real and under-reported phenomenon in current LLM fine-tuning practice.

The method is coherent. Routing experts receive task-aware conditioning and can move quickly. Shared experts receive sparse and importance-based updates and are kept stable by balancing and orthogonality. This produces a clear story: route what is task specific, update shared parts slowly.

The empirical section covers more than one model size and more than one task family. The paper does not stay only on GLUE-type classification. It also tests commonsense, QA and a sequential scenario where tasks arrive one after another. This makes the claim of forgetting mitigation more credible.

The baselines are better chosen than in many PEFT for CL papers. The paper compares to multi-LoRA style and recent mix-of-LoRA methods that are actually close in design, not only to plain LoRA.

### Weaknesses
The method contains several components at once. There is FiLM-like task conditioning for routing, there is decoupling between routing and shared experts, there is EMA-based sparse updating for shared experts, there is load balancing, and there is an orthogonality term. The paper shows that removing some parts hurts, but the ablation is not strong enough to reveal a minimal necessary core. Right now the contribution looks like a well engineered combination rather than a clearly isolated idea.

The continual learning evidence is still light. The sequential experiments are relatively short and remain in a single modality. There is no run with 8 to 10 tasks and no report with several random seeds and CL metrics such as average accuracy, forgetting and backward transfer. The paper therefore demonstrates that MoSE improves stability, but it does not yet show that it is a reliable CL method under longer and noisier streams.

The cost of the additional bookkeeping is not fully reported. MoSE maintains EMA importance, performs load balancing and applies an orthogonality regularizer. These operations have a cost in memory and in wall clock time. The paper states that the overhead is small, but there is no quantitative table that compares MoSE and the strongest baseline on the same hardware.

The relation to very recent MoE or multi-LoRA works could be made sharper. The paper argues that previous works do not decouple shared and routing updates and therefore overwrite knowledge. This is reasonable, but it would help to show an experiment where a frozen shared expert plus task-routed experts is used as an additional baseline. If such a simple baseline closes part of the gap, then MoSE needs to emphasise what is truly unique in its design.

### Questions
1. Can you provide a full ablation table that has at least these four rows: base multi-LoRA or mix-of-LoRA, base plus task-aware routing, base plus sparse shared-updates, and the full MoSE
2. Can you run a longer sequential setting, with 8 tasks or more, and report average accuracy, forgetting and backward transfer, each with at least 3 seeds
3. Can you report wall clock time per epoch and peak memory for MoSE and for the strongest baseline on the same model and sequence length
4. Can you add a baseline that freezes the shared expert and only allows task-routed experts to adapt, so that we can see whether the EMA based sparse update is essential
5. In the current results the gains are often between 1 and 2 points. Can you add variance across runs to show that these gains are statistically meaningful?

I like the problem, I like the decoupled view of routing experts and shared experts, and I believe the direction is useful for practitioners who need to run many tasks on one LLM. However the current version still looks like a rich combination of several sensible ideas rather than a sharply defined core contribution. The continual learning part in particular needs to be longer and more systematic to justify the forgetting-resilient claim.

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
This paper proposes MoSE, a framework designed to mitigate catastrophic forgetting in multi-task fine-tuning of large language models. The method divides LoRA experts into shared experts and routing experts: routing experts learn task-specific information through feature-wise gating mechanisms with full parameter updates, while shared experts employ a Top-k selection strategy to selectively update only the most important parameters for preserving common knowledge. Further experiment results shows the effectiveness of the proposed method.

### Strengths
1.	The paper is well-organized with clear motivation, comprehensive experimental design, and thorough ablation studies that effectively demonstrate each component's contribution.
2.	The proposed method shows consistent improvements across multiple benchmarks, validating its practical effectiveness in multi-task scenarios.

### Weaknesses
1.	The core contribution primarily combines existing mechanisms (shared experts and routing experts from prior MoE literature) without introducing fundamentally new insights or architectural innovations that advance the field.
2.	While the low-rank FiLM module reduces parameters, it introduces additional forward pass layers (sequential Wdown and Wup projections) that increase computational overhead; critically, the paper only reports final performance metrics without providing training time, GPU utilization, or throughput comparisons, obscuring potential efficiency bottlenecks that could limit practical deployment.
3.	The paper lacks direct experimental comparisons with several relevant continual learning methods mentioned in the related work (such as I-LoRA, CL-LoRA, GainLoRA, and LoRI), making it difficult to comprehensively assess the relative advantages of the proposed method against state-of-the-art continual learning approaches specifically designed to address catastrophic forgetting.

### Questions
1.	Given the additional computational layers introduced by the low-rank FiLM module and the gradient-based sparse update mechanism for shared experts, does MoSE incur significantly higher training latency compared to baseline methods like MultiLoRA and MixLoRA? Quantitative comparisons of wall-clock training time and throughput would be essential to assess practical feasibility.
2.	MoSE has more trainable parameters compared to other methods. Could the reported performance improvements be primarily attributed to this increased parameter budget rather than the proposed architectural innovations?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes MoSE, a MoE-LoRA framework to reduce catastrophic forgetting during multi-task fine-tuning. It separates LoRA experts into a "shared expert" for common knowledge and multiple "routing experts" for task-specific knowledge. The key idea is its Gradient-based Sparse Update (GSU), which only fine-tunes a small subset of the shared expert's parameters based on gradient momentum, thereby preserving existing knowledge. Experiments show MoSE reduces forgetting compared to other multi-task PEFT methods.

### Strengths
1. Important Problem: Addresses the critical challenge of catastrophic forgetting in parameter-efficient multi-task and continual learning.

2. Intuitive Method: The separation of shared and task-specific experts is a logical approach to disentangling knowledge.

### Weaknesses
1. Inadequate Baselines for Core Claim: The paper's central claim is "forgetting-resilience," yet its continual learning experiments (Tables 4, 5) only compare against multi-task methods (MultiLoRA, MixLoRA), not state-of-the-art CL methods. The authors even cite dedicated PEFT-CL baselines in Sec 2.3 but do not benchmark against them. This is a critical omission.

2. Limited Novelty: The architectural idea of separating shared and task-specific modules is well-established. The main contribution, the GSU sparse-update heuristic, is not adequately positioned against related work in sparse training or regularization-based CL (like EWC).

### Questions
Refer to Weaknesses for related questions.

### Soundness
2

### Presentation
2

### Contribution
2
