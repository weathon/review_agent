# BideDPO: Conditional Image Generation with Simultaneous Text and Condition Alignment

- Decision: Accept (Poster)
- Scores: 4, 2, 6, 6

## Abstract
Conditional image generation augments text-to-image synthesis with structural, spatial, or stylistic priors and is used in many domains. However, current methods struggle to harmonize guidance from both sources when conflicts arise: 1) input-level conflict, where the semantics of the conditioning image contradict the text prompt, and 2) model-bias conflict, where learned generative biases hinder alignment even when the condition and text are compatible. These scenarios demand nuanced, case-by-case trade-offs that standard supervised fine-tuning struggles to deliver. Preference-based optimization techniques, such as Direct Preference Optimization (DPO), offer a promising solution but remain limited: naive DPO suffers from gradient entanglement between text and condition signals and lacks disentangled, conflict-aware training data for multi-constraint tasks. To overcome these issues, we propose a self-driven, bidirectionally decoupled DPO framework (BideDPO). At its core, our method constructs two disentangled preference pairs for each sample—one for the condition and one for the text—to mitigate gradient entanglement. The influence of these pairs is then managed by an Adaptive Loss Balancing strategy for balanced optimization. To generate these pairs, we introduce an automated data pipeline that iteratively samples from the model and uses vision-language model checks to create disentangled, conflict-aware data. Finally, this entire process is embedded within an iterative optimization strategy that progressively refines both the model and the data. We construct a DualAlign benchmark to evaluate a model’s ability to resolve conflicts between text and condition, and experiments on commonly used modalities show that BideDPO delivers substantial gains in both text success rate (e.g., +35\%) and condition adherence. We also validated the robustness of our approach on the widely used COCO dataset. All models, code, and benchmarks will be released to support future work.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses a crucial and underexplored problem in conditional text-to-image generation: the conflict between guidance from textual prompts and auxiliary conditions (e.g., depth maps, Canny edges). The authors compellingly categorize these conflicts into "Input-Level Conflict" and "Model-Bias Conflict." To resolve these issues, they propose BideDPO, a novel framework based on Direct Preference Optimization (DPO). The core contributions are threefold: 1) A Bidirectionally Decoupled DPO (BideDPO) algorithm that disentangles the optimization signals for text adherence and condition fidelity, combined with an Adaptive Loss Balancing strategy. 2) An automated pipeline to generate conflict-aware, disentangled preference data using a Vision-Language Model (VLM) as a judge. 3) An iterative optimization strategy that progressively refines both the generator and the training data. The authors introduce a new benchmark, DualAlign, to specifically evaluate performance in conflict scenarios. Experiments on state-of-the-art models like FLUX show substantial improvements in both text alignment and condition adherence over baselines like SFT and naive DPO.

### Strengths
1. **Novel and Technically Sound Method:** The core idea of BideDPO is both novel and elegant. Decoupling the preference pairs for text and condition is a clever solution to the "gradient entanglement" problem that plagues naive multi-objective DPO. The mathematical derivation is clear (Sec 3.1, Appendix F), and the Adaptive Loss Balancing (ALB) mechanism using a stop-gradient is a pragmatic and effective way to ensure balanced training without introducing instability.
2. **Innovative and Valuable Data Generation Pipeline:** The lack of suitable preference data for this specific task is a major bottleneck. The proposed automated pipeline (Fig. 3, Algorithm 1) is a significant contribution in itself. Using a VLM to programmatically generate disentangled positive and negative samples for both text and condition is highly innovative. This data-centric approach, especially the iterative refinement loop, is powerful and could be adapted for other multi-objective alignment tasks.

### Weaknesses
1. **Over-reliance on the VLM Oracle:** The entire framework, from data generation to evaluation (Success Ratio, SGMSE/SGF1), heavily relies on the judgments of a single VLM (Qwen2.5-VL-70B). The paper lacks a critical discussion of the potential limitations and biases of this VLM. For example, if the VLM has its own biases (e.g., favoring certain styles or objects), the generator might overfit to these VLM-specific preferences rather than a more general notion of human preference. This is a potential point of failure for the iterative refinement process, as biases could be amplified over time.
2. **Limited Novelty in Data Decoupling Strategy:** The method for generating disentangled preference data, while effective, arguably lacks fundamental novelty. The core strategy of isolating one variable (e.g., text alignment) while holding the other (condition fidelity) constant is a standard analytical technique for factor disentanglement, not a new concept. This approach is analogous to creating modular reward functions in multi-objective reinforcement learning. Consequently, this aspect of the work can be viewed more as a clever and practical engineering solution for applying DPO to a multi-objective problem, rather than a novel advance in preference learning itself.
3. **Stability of Adaptive Loss Balancing (Minor):** The ALB weights are calculated on a per-batch basis from the loss magnitudes. This approach, while simple and effective, could potentially introduce noise into the training process, especially with smaller batch sizes. While the empirical results are strong, a brief discussion on the sensitivity to batch size or potential alternatives (e.g., using an exponential moving average of the losses) could strengthen the paper.

### Questions
1. Regarding the VLM oracle: Could you elaborate on the reliability of Qwen2.5-VL-70B as a preference judge? Have you performed any analysis of its failure modes or systematic biases? What is the risk of the model overfitting to the VLM's specific preferences rather than a general human alignment, especially during the iterative optimization? Did you consider using multiple different VLMs for ensembled judging to improve robustness?
2. Regarding the computational cost: Could you provide an estimate of the computational resources (e.g., GPU type and hours) required to complete one iteration of your framework (i.e., data generation and model optimization)? This would provide valuable context on the practicality of the iterative approach.
3. Regarding the Adaptive Loss Balancing: The ALB weights are computed on a per-batch basis. Did you experiment with or consider using a moving average of the loss values to calculate the weights for a more stable signal? Do you have any insights on how sensitive this mechanism is to the batch size?
4. Regarding the generality of the method: Your method is demonstrated on conditions where fidelity can be measured with clear geometric metrics (MSE for depth, F1 for edges). How do you foresee BideDPO extending to more abstract conditions, such as artistic style from a reference image or high-level scene layouts, where defining a "condition-disentangled pair" and a corresponding error metric is less straightforward?

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
5

### Summary
This paper addresses a critical challenge in conditional image generation: reconciling conflicting guidance from text prompts and structural/ spatial conditioning inputs (e.g., depth maps, Canny edges). The authors first identify two key conflicts hindering model controllability: Input-Level Conflict (semantic mismatch between conditioning images and text) and Model-Bias Conflict (learned generative biases overriding text alignment even when text and condition are compatible). To solve these issues, the paper proposes BideDPO, a bidirectionally decoupled Direct Preference Optimization (DPO) framework.

### Strengths
- Unlike naive DPO (which suffers from gradient entanglement between text and condition signals), BideDPO’s decoupled preference pairs and adaptive loss balancing provide clear, independent optimization signals for each objective. This design effectively addresses the "trade-off dilemma" in multi-constraint generation.
- The automated data pipeline resolves the critical bottleneck of scarce conflict-aware DPO data for conditional generation. By leveraging LLMs for prompt generation and VLMs for quality control, the pipeline enables scalable, self-driven data creation—avoiding labor-intensive manual annotation.
- The DualAlign benchmark is a valuable contribution, as existing benchmarks (e.g., COCO, ImageNet) do not explicitly test text-condition conflict scenarios. It enables standardized evaluation of multi-constraint alignment, which will benefit future research.
- Experiments cover multiple conditioning modalities (depth, Canny, soft edge), include ablation studies for core components (adaptive loss balancing, preference pair disentanglement), and validate robustness on COCO—strengthening the credibility of BideDPO’s effectiveness.

### Weaknesses
- The paper only compares BideDPO against naive DPO and supervised fine-tuning (SFT) on FLUX variants. It fails to include critical SOTA methods in conditional image generation, such as ControlNet++ (Li et al., 2024, which improves conditional control via consistency feedback), LooseControl (Bhat et al., 2024, for generalized depth conditioning), and OmniControlNet (Wang et al., 2024, for multi-modal control). Without these comparisons, BideDPO’s competitiveness in the broader conditional generation landscape remains unproven.
- While the paper mentions DPO variants like RankDPO (Karthik et al., 2024) and SPO (Liang et al., 2024) in related work, it does not compare BideDPO against them. For example, RankDPO handles scalable ranked preferences—relevant to BideDPO’s preference-based optimization—and omitting this comparison obscures BideDPO’s advantages over specialized DPO adaptations for image generation.
- The text SR is measured using Qwen2.5-VL-70B, but the paper provides no justification for choosing this VLM over alternatives (e.g., GPT-4V, Gemini Pro Vision) or validation of Qwen2.5-VL-70B’s reliability in judging text-image alignment. There is also no comparison between automated VLM scores and human annotations—leaving uncertainty about the accuracy of the SR metric.
- All experiments are conducted exclusively on FLUX-family models. The paper does not test BideDPO on other widely used conditional generation frameworks (e.g., Stable Diffusion with ControlNet, MidJourney fine-tuning variants). This limits conclusions about BideDPO’s applicability to different model architectures.
- The iterative optimization strategy (alternating data generation and fine-tuning) likely incurs significant computational costs. The paper provides no analysis of BideDPO’s training time, GPU memory usage, or inference speed compared to baselines—critical for real-world deployment.
- The DualAlign benchmark focuses on structural/ spatial conditions (depth, edges) but excludes other common conditional modalities (e.g., semantic masks, human poses, style references). This limits BideDPO’s validation to a subset of conditional generation scenarios.
- The paper notes performance degradation at the 4th iteration (due to overfitting to self-generated data) but offers no mitigation strategies (e.g., data augmentation, regularization for preference pairs). This limits the practicality of the iterative loop for long-term training.

### Questions
- Why were key SOTA conditional generation methods (e.g., ControlNet++, LooseControl, OmniControlNet) excluded from the baseline comparisons? Would BideDPO maintain its performance advantages when competing with these methods, especially in scenarios with strong structural constraints?
- How does Qwen2.5-VL-70B’s SR judgment correlate with human evaluations? Have you tested other VLMs to confirm that the observed SR improvements are not VLM-specific artifacts?
- The automated data pipeline uses VLMs to filter "high-quality" preference pairs, but what thresholds (e.g., VLM confidence scores) are used to accept/reject samples? How sensitive is BideDPO’s performance to these thresholds?
- Would BideDPO work for non-structural conditional inputs (e.g., style prompts, color palettes, semantic labels)? The current experiments focus on spatial/ structural conditions—no evidence supports generalization to other condition types.
-What strategies could address the overfitting observed in the 4th iterative loop? For example, would integrating external datasets (instead of solely self-generated data) or adding regularization to the preference pair loss prevent performance degradation?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes to alleviate the common issue of conflicts between conditional images and textual prompts in the task of conditional image generation. The authors propose a novel disentangled preference-based optimization technique to alleviate these conflicts.

### Strengths
- The paper introduces a disentangled preference-based optimization technique, which helps mitigate the frequent conflicts between conditional images and text prompts in conditional image generation tasks.
- An automatic Disentangled, Conflict-Aware Preference DPO data pipeline is presented, streamlining the process of handling conflicting conditions.
- The authors construct a DualAlign Benchmark, enabling robust evaluation of a model’s ability to resolve conflicts between visual and textual conditions.

### Weaknesses
- The proposed method is quite straightforward and lacks significant novelty.

- The comparison with state-of-the-art post-training methods is limited; the paper only benchmarks against DPO and SFT.

### Questions
Please refer to my comments in Weaknesses.

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
4

### Summary
This paper introduces BideDPO, a framework for conditional image generation that is specifically designed to resolve conflicts between a user's text prompt and any other conditioning input (like Canny edge). Existing methods struggle to balance these two constraints, especially when they conflict often leading to poor controllability.

### Strengths
1. This paper considers tackling a very practical problem of resolving conflicts between multiple conditionings, which is faced while utilising/controlling majority generative models.
2. The strongest claim is extending DPO to handle competing objectives by mitigating gradient entanglement.
3. The qualitative results shown in Figure 5 are of impressive quality, and prove the effectiveness of the 2 objectives working simultaneously.

### Weaknesses
1. The observed degradation at Iteration 4 suggests the model begins to overfit to the biases and narrow distribution of the data it generates itself? Please provide more intuition for why/why not this may be the case.
2. The main weakness is the unproven quality of the VLM-based preference scoring. If the VLM is not a good proxy for human preference on novel, abstract, or conflicting constraints, then the reported gains may be less strong.

### Questions
1. Could the method scale to more than 2 such conditionings?
2. How reliable is the VLM based check while constructing the preference data given some of these input conditionings/styles are subtle in the image?

### Soundness
2

### Presentation
3

### Contribution
2
