# Task-Aware Mechanism: Hybrid MoE Vision Tower Towards Holistic Video Understanding

- Avg Score: 4.67
- Decision: Reject
- Scores: 4, 6, 4

## Abstract
Does *Comprehending the main idea of a 2-hour movie* and *Counting the birds appearing in a 15-second clip* really warrant the same video processing pipeline? Recent successes of Mixture-of-Experts (MoE) architectures in language modeling have inspired explorations of MoE applications. However, existing MoE models mainly focus on Large Language Models (LLMs) while neglecting Vision Tower (VT) in multimodal models. MoE-LLMs are predominantly designed for capacity scaling, whereas VT contains three fundamentally distinct modules, indicating that directly copying MoE-LLM designs to VT is unlikely to be effective. Inspired by the emerging Task-Aware idea, we argue that MoE-VT architectures should embody the principle of *Right Tool for the Right Job*, providing suitable processing to different tasks. To address this, we propose Task-Aware Mechanism (TAM), a MoE-VT architecture that employs Hybrid Gating Strategy to endow VT with intrinsic Task-Aware ability. To equip the framework with task-aware capabilities, we further introduce a compact Inductor module with only 0.1B parameters, trained on our new dataset TA-116k. With the Inductor, TAM could dynamically determine the appropriate task category, the optimal resolution and number of frames to sample, based on the user query and the length of video. Leveraging TAM, we introduce the TallVA-8B-A7B model, which outperforms current SOTA methods across various benchmarks on comparable LLMs, demonstrating that TAM enables video understanding models to become more holistic on diverse tasks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes Task-Aware Mechanism (TAM), a mixture-of-experts vision tower that enables video-language models to dynamically adapt to different tasks. It introduces a lightweight Inductor module trained on the newly constructed TA-116K dataset, where the Inductor learns to classify user queries and video metadata to predict task types and determine the appropriate frame count and resolution. By integrating the hybrid gating strategy into the TallVA-8B-A7B model, TAM achieves state-of-the-art performance across a wide range of video understanding benchmarks.

### Strengths
1. The paper is well written and easy to follow.

2. The idea of leveraging human expertise across different tasks by training a classifier for hard gating is intuitive and well motivated.

3. The proposed method achieves state-of-the-art performance across diverse video understanding benchmarks, outperforming models with stronger LLMs and demonstrating the effectiveness of task-aware vision processing.

### Weaknesses
1. The major concern is the generalizability of the proposed method to more diverse tasks and environments. The hard-coded strategy of the hard-gating, including the predefined task types and the fixed resolution/frame numbers, may limit its effectiveness and flexibility in handling real-world or out-of-distribution tasks beyond the predefined task types.

2. In practice, even within the same task type, different input videos can vary in difficulty. The proposed inductor considers task awareness only at a coarse granularity, without accounting for sample-wise differences.

3. Efficient and dynamic compute allocation across different samples is a key motivation for adopting routers, yet this work does not provide an in-depth analysis of the achieved efficiency frontier.

4. The experimental results in Tables (c) and (d) show limited accuracy improvement with the routers. It would be desirable to present a detailed performance breakdown to clarify which gains come from architectural improvements and which result from data enhancements.

### Questions
In addition to the questions raised in the weakness section, I would also expect the authors to discuss whether such a hard-coded routing strategy could represent the future direction of LVLM design, either for cloud or edge deployment.

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
This paper proposes Task-Aware Mechanism (TAM), a hybrid Mixture-of-Experts (MoE) architecture applied to the Vision Tower (VT) of video-language models. Unlike prior works that mainly use MoE in LLMs for capacity scaling, TAM focuses on making the VT task-aware. A lightweight Inductor module (0.1B parameters) predicts the task type and dynamically adjusts video resolution and frame count based on the input query and video length. The authors also introduce a new dataset TA-116K for training the Inductor. The resulting model, TallVA-8B-A7B, achieves better performance across multiple video understanding benchmarks compared to state-of-the-art LVLMs with comparable LLM backbones.

### Strengths
1. Well-written and easy to follow: The paper presents strong organization, with clear figures (e.g., Fig. 1 and Fig. 2) illustrating the pipeline and gating mechanisms.
2. Comprehensive experiments: The evaluation spans diverse benchmarks (e.g., MVBench, EgoSchema, LongVideoBench, etc.), with consistent gains even when using smaller LLMs.
3. Reproducibility: Implementation details, dataset composition, and ablation studies are described in depth, which enhances reproducibility.

### Weaknesses
1. Generalizability of the 8 task types: It is unclear whether the eight predefined video understanding categories comprehensively capture the diversity of real-world video tasks. If new task types emerge, will the Inductor or gating modules require retraining from scratch, or can they generalize via few-shot adaptation?

2. Comparison with existing token merging or pruning works. Based on my understanding, the proposed gating mechanism can perform visual token compression before passing them to the LLMs, guided by the information from the input queries. However, some token merging or pruning methods (e.g., https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02577.pdf and https://arxiv.org/abs/2405.16148) also reduce visual tokens, often based on the attention maps from previous layers or additional input cues, and may achieve a similar goal.

3. Scalability beyond 7B-scale models: While experiments are comprehensive, all evaluations rely on 7B-scale LLMs. It would strengthen the paper to show whether the task-aware mechanism scales effectively to larger models or yields diminishing returns.

### Questions
The manuscript needs more careful proofreading, e.g., “leverage video metadata to train a task-aware Hybrid-Gated MoE Vision Tower.” → “leverages,” singular.

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
The paper presents TAM, a task-aware hybrid Mixture-of-Experts (MoE) Vision Tower for video large vision-language models. A small Inductor module predicts the task type and selects frame count and resolution based on the user query, while the vision encoder applies soft-gated MoE and the projector uses hard, resolution-specific experts. The model, TallVA-8B-A7B, trained in a staged manner (LLM-first followed by co-upcycled MoE initialization), achieves notable gains across ten video understanding benchmarks compared with models of similar LLM scale.

### Strengths
- **Well-motivated architecture**. The hybrid gating design—soft MoE for the vision encoder and hard MoE for the projector—aligns gating type with module characteristics, showing moderate but consistent empirical gains.

- **Broad benchmark coverage**. Evaluations span ten diverse video datasets, including long-video and temporal reasoning benchmarks, indicating reasonable generalization.

- **Practical training observation**. The staged “LLM-first” training prevents collapse when adapting to variable visual tokens, a potentially reusable insight for similar multimodal systems

### Weaknesses
- **Cascading error from uncalibrated routing**. The Inductor deterministically sets both frame count and resolution, yet the paper provides no calibration or uncertainty estimate for its predictions. A single misclassification can simultaneously under-sample frames and reduce resolution, compounding errors before the LVLM sees the input. Reporting confidence metrics (e.g., ECE) or fallback routing policies would make the “task-aware” claim more convincing.

- **Weak validation of temporal fidelity.** The method selects task-relevant frames but provides no metric showing these frames capture the decisive temporal evidence. For long videos, missing key moments can undermine performance even if accuracy appears high. Including measures such as hit-rate@k or causal-segment coverage would strengthen the evaluation.

- **No isolation of MoE benefit**. There is no non-MoE query-aware baseline, making it uncertain whether improvements are stemed from dynamic frame/resolution routing or MoE specialization

- **Lack of cost–accuracy tradeoff analysis**. Although FLOPs are reported, the paper omits practical latency, throughput, and memory benchmarks under varying frame budgets. Since routing aims to optimize efficiency, TAM should ideally be evaluated along a Pareto frontier of accuracy versus compute cost.

### Questions
What exact modifications were made to lmms-eval and how do results differ when using the unmodified version?

Routing visualizations imply specialization, but is there causal evidence that experts encode distinct semantics rather than superficial resolution cues? Have the authors tried expert-swap or routing-scramble tests to validate this claim?

Figure 7 is qualitative only. What is the quantitative distribution of load or token assignment across experts? Are there signs of imbalance or dominance that could affect scalability?

Since routing relies on free-form text queries, how robust is the Inductor to paraphrasing, ambiguity, or multilingual inputs? Have confusion matrices or OOD tests been conducted to assess generalization?

### Soundness
2

### Presentation
3

### Contribution
3
