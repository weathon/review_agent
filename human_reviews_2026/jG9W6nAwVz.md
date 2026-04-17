# TwinVLA: Data-Efficient Bimanual Manipulation with Twin Single-Arm Vision-Language-Action Models

- Decision: Accept (Poster)
- Scores: 6, 4, 4, 4

## Abstract
Vision-language-action models (VLAs) trained on large-scale robotic datasets have demonstrated strong performance on manipulation tasks, including bimanual tasks. However, because most public datasets focus on single-arm demonstrations, adapting VLAs for bimanual tasks typically requires substantial additional bimanual data and fine-tuning. To address this challenge, we introduce TwinVLA, a modular framework that composes two copies of a pretrained single-arm VLA into a coordinated bimanual VLA. Unlike monolithic cross-embodiment models trained on mixtures of single-arm and bimanual data, TwinVLA improves both data efficiency and performance by composing pretrained single-arm policies. Across diverse bimanual tasks in real-world and simulation settings, TwinVLA outperforms a comparably-sized monolithic RDT-1B model without requiring *any* bimanual pretraining. Furthermore, it narrows the gap to state-of-the-art model $\pi_0$, which relies on extensive proprietary bimanual data and compute cost. These results establish our modular composition approach as a data-efficient and scalable path toward high-performance bimanual manipulation, leveraging public single-arm data.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes TwinVLA, a data-efficient framework that finetunes a pre-trained single arm VLA into a bimanual VLA without requiring any bimanual pretraining. The proposed method utilizes a joint attention mechanism of two identical pre-trained single-arm policies to adapt the bimanual action space. The experiment results show that TwinVLA achieves competible performance on downstream tasks compared to several bimanual VLAs that were pre-trained on large-scale bimanual manipulation data.

### Strengths
1. The single-arm policy can be trained to execute bimanual manipulation tasks without requiring any bimanual pre-training;
2. The proposed model can achieve competible performance compared to SOTA bimanual VLAs such as $\pi_0$ and RDT with much less training cost.

### Weaknesses
1. The paper has limited experimentation in real-world scenarios (only 3 tasks) and lacks comparisons with more bimanual VLA baselines;
2. There is a noticeable performance gap between TwinVLA and both $\pi_0$ and RDT in Hard tasks of the RoboTwin2.0 benchmark. More importantly, yet the study omits a detailed analysis of it.

### Questions
The paper indicates that TwinVLA performs poorly on the RoboTwin-Hard tasks. Is there a more in-depth analysis of this critical limitation? Furthermore, the authors also mention that TwinVLA exhibits weak generalization to unseen tasks. Do the aforementioned results reveal inherent weaknesses in this coordination mechanism's out-of-distribution generalization capability? Given that VLAs are expected to demonstrate strong generalization on unseen tasks, does this mean we still cannot avoid relying on large-scale bimanual pre-training data to achieve this goal?

### Soundness
3

### Presentation
3

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
This paper addresses the challenge of data scarcity in bimanual manipulation by proposing TwinVLA, a modular and data-efficient VLA framework. The core idea is to duplicate a VLA pre-trained on abundant single-arm data into two single-arm VLAs coupled via joint attention, with MoE routing, connecting them with a lightweight module for coordination. Experimental results demonstrate that this approach achieves strong performance by fine-tuning on only a small amount of bimanual data.

### Strengths
Modular twin design. Proposes TwinVLA, a clear, practical architecture that couples two single-arm pretrained VLAs via causal joint attention, enabling coordinated bimanual control while preserving per-arm specialization.

Efficiency mechanisms. Integrates MoE routing (and attention re-weighting on shared tokens) to avoid duplicate computation on shared inputs and to stabilize adaptation, yielding a favorable compute/memory profile.

Data efficiency. Achieves strong performance with only a small number of bimanual demonstrations for fine-tuning, offering a realistic path to leverage abundant single-arm corpora without large-scale bimanual pretraining.

### Weaknesses
Architectural advantage not causally established: The paper argues that the "twin structure" is superior to a "monolithic" one by comparing TwinVLA (1.3B) to RDT-1B (1.2B) . However, this comparison is confounded by major differences in their pre-training data and recipes. Without a size-matched monolithic VLA trained on exactly the same 0.5M single-arm pretraining data and the same 50 bimanual demos (with equal token/compute), the observed gains cannot be causally attributed to the twin.

Bimanuality insufficiently demonstrated in real-world tasks: The three real-world tasks appear predominantly sequential—two single-arm subtasks executed in series—rather than requiring concurrent, mutually constraining coordination. In the absence of tasks that necessitate simultaneous bilateral coupling (e.g., coordinated lifting/transport of a single large object), the evidence for learned genuine bimanual synergy remains weak.

### Questions
On the Language Following Evaluation (Sec 5.4): The paper claims TwinVLA's strong performance (80.6% vs. RDT-1B's 55.5%) is due to its architecture better preserving pre-trained language knowledge. How is a simpler confounding variable ruled out: that the baselines' (RDT-1B, Pi0) pre-training data mixtures were simply less effective at teaching these specific color-based instructions in the first place? Was any single-arm evaluation performed on the pre-trained checkpoints to confirm all models started with this capability before bimanual fine-tuning?

Aloha-Sim Easy may be too weakly bimanual. The w/o Joint attn variant (no cross-arm coordination) achieves 69.6%, only 6.2% below full TwinVLA (75.8%), suggesting these tasks may not require strong concurrent, mutually-constraining coordination. We request this 'w/o Joint attn' ablation be performed on the real-world and RoboTwin tasks to see if the coordination module proves more valuable there.

On 'Scratch' TwinVLA (71.2%) outperforming pre-trained RDT-1B (61.6%) on Aloha-Sim Easy: This is a counter-intuitive result, as one would expect the pre-trained RDT-1B to have an advantage. Could the authors analyze whether this finding points to: a) A superior inductive bias of the 'twin structure' itself for low-data bimanual learning, which is strong enough to overcome the lack of pre-training? b) A potential flaw, simplicity in the Aloha-Sim Easy benchmark?

### Soundness
2

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
TwinVLA proposes a modular approach for bimanual robotic manipulation by composing two pretrained single-arm vision-language-action models into a coordinated dual-arm system, avoiding the need for dedicated bimanual datasets. Instead of training a monolithic model on mixed data, it reuses single-arm policies and fuses them for coordinated control, improving data efficiency and generalization across both simulated and real-world settings. Experiments show that this compositional strategy can match or surpass prior large-scale bimanual methods without additional bimanual pretraining, suggesting a scalable and practical path to bimanual manipulation using existing public single-arm data.

### Strengths
- The evaluation is comprehensive and convincing, covering both simulation and real-world settings. The experiments span two simulation benchmarks and physical robot trials, with tasks designed to test both coordinated bimanual manipulation and asymmetric master-assistant roles.

- The method draws inspiration from neuroscience principles, providing an interesting conceptual link between biological motor coordination and modular robotic control.

- The approach significantly reduces VRAM and computational overhead compared to monolithic bimanual models, enabling training and deployment on more affordable hardware and lowering practical adoption barriers.

- The modular policy design offers strong scalability and flexibility, allowing pretrained single-arm expertise to be reused efficiently without requiring specialized or costly bimanual datasets.

### Weaknesses
- The core architecture and data-flow formulation of TwinVLA—arguably the central contribution—would benefit from a more formal and detailed presentation (*i.e.*, with equation and notations) in the main paper. Key notations and explanations are currently placed in the appendix, while the preliminaries and single-arm VLA review receive proportionally more emphasis. Streamlining those background sections could make room for a clearer, more rigorous exposition of the proposed twin-policy mechanism and coordination strategy.

- The real-world evaluation focuses largely on master–assistant settings, where one arm primarily aids the other. Including tasks that require tighter spatial and temporal coordination—such as hand-overs, bilateral grasping, or deformable-object manipulation (e.g., folding cloth by holding both corners)—would strengthen claims about general bimanual capability. These scenarios inherently require richer information exchange between the representations of the two arms, which may challenge the current decoupled design of TwinVLA and provide further insight into its scalability to fully interdependent control regimes.

- While the empirical results demonstrate benefits over monolithic bimanual models, the underlying mechanism for this advantage remains insufficiently explored. In my evaluation setting, this point is particularly relevant: as embodied datasets continue to scale, the performance gap between compositional and monolithic approaches may shift (Table 9, twinVLA vs pi0, btw, it's not a big deal). Consistent with the “bitter lesson,” it would be valuable to articulate whether the observed gains arise from modular inductive biases, improved optimization stability, or data-efficiency effects, rather than relying primarily on current empirical trends that may be influenced by the present limits of bimanual data availability. A deeper discussion here would strengthen confidence in the method’s long-term relevance and theoretical grounding.

I would be inclined to raise my score if the paper organization  are further improved.

### Questions
- L469 the absolute end-effector representation is inherently not frame-agnostic. Given that relative end-effector actions are typically more suitable for embodiment transfer, could the authors clarify why the relative formulation was not adopted here, and elaborate on the rationale behind choosing the absolute representation?

### Soundness
3

### Presentation
2

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
This paper improves bimanual manipulation by combining single-arm pretraining with a Mixture-of-Experts (MoE) mechanism.
This approach enables the model to achieve performance close to state-of-the-art (SOTA) results while using significantly less pretraining data.

### Strengths
The motivation is sound, and the results demonstrate that it is a more efficient training approach.

The experimental results and supplementary materials provide additional details, and the experiments on weight changes in particular highlight the advantages of the proposed method.

### Weaknesses
1. The implementation details are not clearly described. The approach to handling bimanual manipulation appears somewhat similar to previous works, or at least it’s not evident how it fundamentally differs from other methods. More implementation details are needed from the authors to ensure an accurate evaluation.

[1] Anybimanual: Transferring Single-Arm Policy for General Bimanual Manipulation
[2] InterACT: Inter-dependency Aware Action Chunking with Hierarchical Attention Transformers for Bimanual Manipulation

2. It is unclear whether this fine-tuning paradigm, which is specifically designed for bimanual tasks, can be applied to robots with different degrees of freedom. There might be concerns regarding the generalizability of the method.

3. Will the authors conduct evaluations on more challenging tasks, such as tests under varying lighting conditions or complex backgrounds? I would like to understand the current limitations or upper bound of this fine-tuning approach.

### Questions
Please refer to the weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3
