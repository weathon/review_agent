# MetaSpatial: Reinforcing 3D Spatial Reasoning in VLMs for the Metaverse

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 2, 6, 4, 4

## Abstract
We present MetaSpatial, the first reinforcement learning (RL) framework for enhancing 3D spatial reasoning in vision-language models (VLMs), enabling real-time 3D scene layout generation without post-processing. MetaSpatial addresses two key challenges: (i) the need for extensive post-processing, as existing VLMs lack inherent 3D spatial reasoning to generate realistic layouts; and (ii) the inefficiency of supervised fine-tuning (SFT) for layout generation due to scarcity of perfect annotations. Our core contribution is the 3D Spatial Policy Optimization (3D-SPO) algorithm, which incorporates physics-aware modulation into advantage estimates at the object level and trajectory-level reward from a training-only multi-turn refinement pipeline. This design enhances temporal credit assignment and encourages spatially consistent policy learning. Empirical evaluations across models of varying scales demonstrate that MetaSpatial improves spatial coherence, physical plausibility, and formatting stability, leading to more realistic and functionally coherent object placements applicable to metaverse environments.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces a novel RL framework for layout generation from vision-language models. It outperforms several existing VLLMs including both open-weight and API-based and training paradigms over one layout generation benchmark.

### Strengths
1. New training data for layout VLM training and new multi-turn training paradigm for layout reasoning
2. Better performance over existing VLMs on one layout generation benchmark and a spatial reasoning task

### Weaknesses
1. Some basic setup details are missing, what is the detailed setup of the evaluation benchmark for layout generation? The only one evaluation set mentioned in the paper is Open3DVQA and the model results on it are in the appendix
2. Lack of layout generation evaluation tasks. As this paper introduces a novel layout generation framework, it is only evaluated on one layout generation task. To justify the performance, at least one more layout generation task should be tested, e.g., PlanGen-1k
3. Potentially unfair baseline comparisons: LayoutGPT uses GLIGEN for rendering, however the method in the paper uses Blender and claims better performance over LayoutGPT. Additionally, there are more recent open-weight layout generation baselines such as PlanGen and LayoutVLM missing as well.
4. Unconvincing claim of the proposed RL performs better than traditional SFT: intuitively, SFT helps models gain knowledge/behaviours that are missing in their inherent knowledge domain while RL helps to generalize them to OOD tasks, and layout generation is one kind of ability that VLMs do not generally acquire much during pre and post training, thus SFT might be more helpful. In Table 7 SFT alone can outperform the RL alone model in 2 out of 4 metrics, and RL after SFT even largely jeopardizes the Constraint score. This evidence does not support the claim that RL outperforms SFT well.

### Questions
Since you have tested on general spatial reasoning tasks, I’m curious what would the model perform on other spatial reasoning tasks with real-world visual input

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
This paper proposes a novel framework called MetaSpatial, to learn to generate feasible but non-unique 3D scene layouts through RL instead of SFT. Without requiring precise annotations, it directly optimizes spatial feasibility and aesthetic consistency via environmental feedback, thereby reducing or even eliminating the need for heavy differentiable or heuristic post-processing.
The method enables a VLM to learn to directly generate physically plausible and semantically consistent 3D scene layouts, improving its three-dimensional spatial reasoning ability. Through interaction with virtual environments, the model gradually acquires human-designer-like spatial aesthetics and practical layout principles without human annotations, significantly enhancing the usability and visual quality of the final layouts.

### Strengths
1．	The authors correctly formulate the problem as a policy learning task and design a multi-stage optimization framework that iteratively improves spatial understanding ability.
2．	The method relies entirely on interaction feedback without GT coordinates, which aligns with the inherently continuous and multi-solution nature of 3D spatial layout generation. This demonstrates the rationality of using reinforcement learning instead of supervised learning.
3．	The authors conduct comprehensive ablation studies to validate the effectiveness of the three proposed components and their contribution to performance improvement.
4．	The method achieves significant performance gains across different models, showing clear improvement over baseline models.

### Weaknesses
I have the following concerns about this MetaSpatial:
1．	In spatial layout design, the work only consider the xyz positions of assets without explicitly modeling rotation or scale. Would this omission have a significant impact on the generated layouts?
2．	In the rendering stage, if the rendering viewpoint is fixed, could this bias the generation results toward that viewpoint, leading to a non-generalizable 3D spatial understanding?
3．	Although the generated furniture layouts satisfy physical constraints such as collision avoidance, are they functionally feasible-for instance, can people move normally between furniture?
4．	I wonder whether relying solely on GPT-4o as an expert evaluator is sufficient. Should human experts or other models also be involved in the evaluation process?

### Questions
see weakness

### Soundness
4

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
The paper proposes using an RL framework to post-train VLMs for 3D scene layout generation. The key benefit is that RL removes the need for post-processing and does not rely on ground-truth layout labels. In addition to standard format rewards, the method introduces physics-aware rewards and rendering-based rewards to guide spatial consistency. The framework also incorporates both trajectory-level and object-level feedback, which helps address credit assignment issues commonly seen in RL training.

### Strengths
* The paper is clearly written and easy to understand.
* Using RL framework for 3D scene layout is indeed a good fit for the task and is well motivated.
* The added physics and rendering rewards make sense and help guide realistic layout generation.
* The zero-shot tests on spatial reasoning benchmarks are useful to see whether the method really improves general spatial understanding.

### Weaknesses
* The evaluation setting feels limited. The layouts use a small set of assets and the text instructions are not very complex, so it is unclear how well the method scales to more diverse or realistic scenarios.

* There is no human evaluation or diversity analysis. This makes it hard to tell whether the generated layouts are actually preferred or just optimized for the automatic rewards.

* It would be helpful to compare more directly with prior layout generation methods (e.g., LayoutGPT, LayoutVLM) under their setups. I understand many of these methods require post-processing/optimization, while the proposed method provides real-time performance. But this would give readers a clearer sense of relative performance and trade-offs.

* The rendering-based evaluation relies on GPT-4o as the judge, which may not be reliable or scalable. The model could potentially reward-hack the evaluator, and the rendered scenes shown in the paper do not look strong enough to guarantee accurate judgments. It is also unclear whether this evaluation can handle small object placement or fine-grained spatial correctness. Additionally, rendering plus querying GPT-4o introduces latency, which may limit scalability.

### Questions
Please see the weakness section.

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
The paper presents MetaSpatial, a reinforcement learning framework designed to enhance 3D spatial reasoning in vision-language models (VLMs), aimed at generating physically consistent, realistic 3D scene layouts without post-processing. The core contribution is 3D Spatial Policy Optimization (3D-SPO), an RL algorithm that integrates:

- Physics-aware advantage modulation at the object level (masking x,y,z coordinate tokens and adjusting based on collision/constraint ratios),

- Trajectory-level reward aggregation via multi-turn layout refinement, and

- A three-tier reward structure combining format, physics, and rendering-based assessments.

Empirical results show substantial improvement in scene plausibility, formatting accuracy, and physical feasibility over baselines such as LayoutGPT, I-Design, and standard supervised Qwen2.5-VL models. MetaSpatial achieves better GPT-4o-based perceptual scores and lower collision rates. The paper also explores an SFT+RL hybrid strategy, showing efficiency improvements via pseudo-labeling high-reward rollouts.

### Strengths
- Well-structured problem formulation. The authors correctly identify that supervised fine-tuning (SFT) fails for spatial tasks with non-unique solutions and continuous coordinate spaces, positioning RL as a more suitable approach.

- Technically sound algorithmic design. The proposed 3D-SPO builds meaningfully on GRPO by adding object-level physics-aware modulation and trajectory-level reward shaping, addressing sparse or unstable feedback in 3D reasoning.

- Comprehensive reward design. The hybrid reward integrates low-level correctness, physical realism, and high-level perceptual alignment (via GPT-4o). The staged reward weighting strategy is well motivated.

### Weaknesses
- Incremental conceptual novelty. While 3D-SPO is well engineered, it is primarily an adaptation of known ideas (GRPO + physics-informed masking + cumulative reward shaping). The theoretical grounding of the advantage modulation remains heuristic; no convergence or stability analysis is presented.

- Evaluation largely internal. The results rely heavily on the authors’ synthetic dataset and GPT-4o-based evaluation. No human evaluation or cross-dataset validation is performed. It’s unclear how well the model generalizes to unseen real-world 3D scenes.

- Ambiguity in quantitative metrics. The GPT-4o perceptual score and “format correctness” are proprietary or internal metrics with no clear definition of variance or statistical significance. Reported deltas (e.g., +0.2–0.3) may not be statistically meaningful.

- Heavy dependence on GPT-4o for reward and evaluation. The rendering-based reward is computationally expensive and opaque; it conflates aesthetic preference with physical correctness, limiting reproducibility and interpretability.

- Computational cost and scalability. Training requires rendering in Blender and external VLM calls (GPT-4o). The cost and latency make large-scale or real-time deployment questionable.

- Lack of theoretical insight. The paper is strong empirically but weak in analysis, no ablation isolates whether improvements come from multi-turn refinement, reward design, or the masking mechanism.

- Limited generality. The work focuses on interior design scenes; extension to robotics or embodied reasoning tasks is claimed but not demonstrated.

### Questions
- How sensitive is performance to the weighting factors (λ₁, λ₂, λ₃) and decay factor (γ)? Is 3D-SPO robust under different hyperparameter settings?

- Can MetaSpatial generalize to real-world room scans (e.g., ScanNet or Matterport3D) beyond synthetic I-Design data?

- How much of the performance gain stems from the multi-turn training vs. the physics-aware masking?

- Could GPT-4o evaluation bias the model toward aesthetic rather than physical correctness?

### Soundness
3

### Presentation
2

### Contribution
2
