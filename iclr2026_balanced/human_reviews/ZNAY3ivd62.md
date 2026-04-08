## Human Reviewer 1

### Summary
GUI-SPOTLIGHT proposes an iterative "spotlight" mechanism for GUI visual grounding, where a multimodal model invokes cropping, quadrant extraction, and color-guided tools to progressively refine focus on screen regions. It couples multi-turn tool reasoning with a stabilized GSPO-based reinforcement learning pipeline and curated high-resolution GUI data. The approach improves grounding accuracy on ScreenSpot-Pro and UI-Vision while using far fewer samples than prior systems. Contributions include the iterative tool framework, modified multi-tool RL objective, and dataset/ablation insights for GUI grounding.

### Strengths
1. The paper implements a structured, multi-turn GUI grounding model with crop, quadrant, and color-guided tools, paired with a stabilized GSPO-based RL objective, offering a disciplined training-and-tool framework for GUI localization.

2. Empirical results on ScreenSpot-Pro and UI-Vision, along with ablations on tool policies and reward shaping, show consistent accuracy gains and provide useful insights into scaling high-resolution GUI grounding with limited data.

### Weaknesses
1. The core "spotlight" idea and iterative tool chain (crop/quadrant/color) closely follow prior ScreenSpot-Pro iterative narrowing and ScreenSeekeR baselines, and the main novelty lies largely in data curation and retraining rather than a fundamentally new paradigm. A clearer comparison and attribution to SS-Pro’s iterative region-refinement design is needed.

2. Directly comparing scores shows GUI-Spotlight performs far below GTA-1-7B when using the same Qwen2.5-VL-7B backbone; without closing that gap or analyzing failure modes, the method provides limited evidence it advances the state of the art. A detailed head-to-head and explanation of the performance deficit are necessary.

3. The paper evaluates only isolated grounding tasks and lacks validation in the GUI agents tasks (e.g., OSWorld). Demonstrating consistent performance in holistic GUI tasks and multi-step interaction settings is necessary to support claims of practical utility.

### Questions
Apart from the training process, what differences exist between this work and the ScreenSeekeR baseline in ScreenSpot-Pro?

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
4

### Confidence
5

---

## Human Reviewer 2

### Summary
Introduces GUI-Spotlight:
1. A dataset pipelines that filters be high quality using Qwen2.5 (removing low quality instructions, inaccurate bounding boxes and consistency).
    a. This includes introducing a new dataset of 15k high-resolution samples, since existing open-source datasets did not have high resolution images.
2. Introduces a multi-stage fine-tuning setup with SFT and RL. Ablating various RL setups.
3. Outperforms existing models on GUI-specific benchmarks, using few samples.

### Strengths
1. Introduces an interesting new training setup for tool-use in GUI-setups. Ablates various RL algorithms, and shows how they are stable or  unstable.
2. Various ablations on their training steps.
3. Good performance beyond the evaluation set they hillclimbed on.

### Weaknesses
1. Details are unclear about their new proposed dataset:
    a. 15k high resolution samples, but where did they come from? How were the domains selected?
    b. Where did the instructions come from in this crawled dataset?
    c. "For our high-resolution dataset, we enhanced the pipeline with additional functions" -> details?
2. The new dataset sounds very useful, but it wasn't ablated how much of the performance improvement is simply because of the new data pipeline or high quality samples that were crawled outside of the open-source datasets. Hard to estimate the impact of that versus the other proposals (like multi-stage or RL algorithm).

### Questions
1. Can you add more details on the dataset filtering and dataset creation from scratch (e.g. in appendix if no space)?
2. Can you ablate training only on the newly filtered + proposed dataset, ignoring tool calls or multi-stage finetuning?
3. How much does this fine-tuned model decrease in performance on non-GUI tasks? Can we still use it as a "foundation model"? Even if performance in non-gui tasks is bad, would be helpful to report (none of the existing GUI-specific frameworks keep generalization beyond GUI, but it's not being tracked).
4. "After warm-up the model learns to invoke multiple tools but remains under-aligned": Do we know whether we trained for long enough/hyper-parameters are optimal (or "good-enough")? I don't see many experiments around stage 1 (completely removing it, increasing duration, different HPs, different data)
5. How much compute did you use?

### Soundness
3

### Presentation
3

### Contribution
2

### Rating
6

### Confidence
3

---

## Human Reviewer 3

### Summary
This paper proposes GUI-Spotlight, a reinforcement learning framework for GUI grounding that coordinates multiple tools (crop, extract, find_color) to enable multi-step “think-with-image” reasoning. The approach fine-tunes 7B vision-language models with 18.5K examples using a GSPO-based algorithm to stabilize training. Experiments on ScreenSpot-Pro (52.8%) and UI-Vision (23.4%) show that GUI-Spotlight achieves competitive performance compared with larger models.

### Strengths
**1. Careful Experimental Analysis**: The paper conducts extensive ablations on RL algorithms (Section 4.1) and reward design (Section 4.2), providing valuable insights beyond reporting final results. The inclusion of negative results is particularly commendable.

**2. Comprehensive Empirical Results**: Consistent improvements across multiple benchmarks (ScreenSpot-Pro: 52.8%, UI-Vision: 23.4%, OSWorld-G: 62.7%) demonstrate robustness. The method works well under different initializations (Qwen2.5-VL-7B and UI-TARS-1.5-7B).

**3. Practical System Design**: The tool-based iterative refinement pipeline is intuitive and interpretable. The three tools (*crop*, *extract*, *find_color*) are simple and easy to use, and the inference pipeline is clear and implementable.

**4. Rigorous Data Curation**: The multi-stage filtering pipeline, which leverages Qwen2.5-VL-72B for instruction quality checks, bounding box validation, and consistency filtering, ensures high-quality training data.

**5. Training Stability Innovation**: The modified GSPO with auxiliary loss J'(θ) and tool-filtered sampling addresses key challenges in multi-turn RL training and effectively prevents collapse observed in vanilla GRPO/GSPO.

### Weaknesses
**1. Lack of Comparison with Recent Baselines**

The evaluation on ScreenSpot-Pro does not include several stronger 7B baselines that are already reported on the leaderboard, such as GUI-ARP-7B (60.8%), Holo1.5-7B (57.9%), GTA1-7B (55.5%), and GUI-Cursor (56.5%). Although some of these works may be concurrent, it would strengthen the work to demonstrate whether the proposed iterative tool-coordination approach can further improve these stronger base models, helping to isolate the contribution of the methodology versus the initialization choice.

**2. Data Efficiency Claims**

The claim of achieving results with "only 18.5K samples" could benefit from more clarification. The model initializes from UI-TARS-1.5-7B, which was pre-trained on millions of GUI examples, so the 18.5K represents incremental fine-tuning rather than training from scratch. When starting from the non-GUI-specific Qwen2.5-VL-7B baseline, performance drops to 38.7% on ScreenSpot-Pro, which still lags behind other grounding models trained on large-scale datasets. Clarifying the relative contributions of the base model versus the proposed training procedure would better position this work as an incremental improvement built upon strong pre-training.

**3. Insufficient Analysis of RFT Design Choices**

The paper employs a modified GSPO algorithm but lacks detailed comparison with other recent RFT approaches. For instance, UI-AGILE uses only 9K examples (versus 18.5K in this work) and achieves 48.7% on ScreenSpot-Pro when initialized from Qwen2.5-VL, compared to 38.7% reported here with the same baseline. 

**4. Simplistic Tool Design**

The tool set (crop, extract, find_color) is relatively simple. Given that GUIs are not natural images but structured environments with explicit hierarchies, element types, and interaction semantics, incorporating structure-aware approaches (e.g., pretrained GUI element detectors, OCR-based text localization) could potentially improve both efficiency and accuracy. The current design treats GUIs as generic images, which may limit extensibility to more complex tasks.

**5. Presentation Issues**

- **Citation formatting**: Inconsistent use of `\citep` vs. `\citet` throughout. For example, lines 81-90 in Section 2 incorrectly format parenthetical citations as narrative ones (e.g., "...screen coordinates You et al. (2024)" should be "...screen coordinates \citep{You2024}"), creating inconsistent formatting across references.

- **Layout**: Some figures and tables have insufficient spacing, which affects readability. For instance, the gap between Figure 4 and its caption is too small, and in Section 5.2, the spacing below Table 4 is too tight against the heading of Section 5.3. Adjusting the vertical spacing would improve the overall visual clarity.

- **Minor inconsistency**: Mixed use of "GUI-SPOTLIGHT" versus "GUI-Spotlight". Using one form consistently would be clearer.

These issues do not affect the technical content but could be easily addressed in revision to improve clarity and readability.

### Questions
**1. Ablation on Stronger Baselines**

Can the proposed iterative tool-coordination approach be applied to stronger 7B GUI grounding models such as GUI-ARP-7B, Holo1.5-7B, GTA1-7B, and GUI-Cursor-7B? Demonstrating improvements on these baselines would help strengthen the contribution and better distinguish the method from base-model choices.

**2. Comparison with Other RFT Methods**

For example, UI-AGILE achieves 48.7% on ScreenSpot-Pro with only 9K samples using Qwen2.5-VL as the base, compared to 38.7% reported here with 18.5K samples. What accounts for this discrepancy? Are there complementary design choices from UI-AGILE or other RFT methods that could be integrated into GUI-Spotlight?

**3. Tool Usage Ablation**

What is the distribution of tool invocations during inference? How many tool calls are typically needed before producing a final answer, and does this vary across benchmarks or UI complexity levels? An ablation on the usage of individual tools could provide insights into the model’s reasoning patterns and inform future tool design.

**4. Tool Generalization**

Have the authors considered extending the tool set to leverage GUI-specific structure? For example, incorporating GUI element detectors or OCR-based text localization could potentially reduce iteration steps and improve efficiency on more complex interfaces.

**5. Presentation Issues**

Fixing the presentation issues noted in the weaknesses would improve clarity and readability.

### Soundness
3

### Presentation
2

### Contribution
2

### Rating
4

### Confidence
3

---

## Human Reviewer 4

### Summary
This paper mainly focused on the GUI Grounding task. The authors proposed an adaptative iterative focus refinement method to progressively encourage models using tools, e.g., cropping sub-regions, for more fine-grained information and enhance the overall grounding accuracy. A modified Group Sequence Policy Optimization (GSPO) algorithm was proposed to help MLLM adapt to the tool using mechanism. Multiple experiments are presented to show the effectiveness of the proposed method.

### Strengths
The idea of encouraging models to use tools to progressively ground potential GT elements is interesting.

### Weaknesses
1. Table 1 shows three available tool functions and I have some questions
  - extract can be considered a special case of crop. So I wonder why is the extract function included?
  - what is the motivation for including the find_color function? Was it because MLLMs were found to have errors in color recognition? 
  - If there are multiple regions matching the target color, how are the offsets for these multiple regions represented in the return value, and how are the sizes of the corresponding regions indicated? 
2. In Section 3.2.2, Stage 1, it is stated that "collected 2561 multi-turn dialogue trajectories with tool invocations." Please present more detail of how this multi-turn dialogue data incorporating tool usage acquired.
3. The proposed method for element grounding requires inference involving multiple tool calls, which will significantly increase the time and computational overhead for the grounding task. Have the authors statistically analyzed the average number of tool calls required to complete a single grounding task? Could the authors provide a comparison of the time and computational overhead against existing methods? Furthermore, in the context of a GUI Agent, this overhead would severely impact the real-time performance of the agent.
4. Based on the tool use prompt template shown in Appendix A.3, the model's need to call a tool is likely because the input image resolution is too large (considering computational costs, the MLLM performs some degree of compression), and the MLLM expects to zoom in on a local area to see more clearly. For the extract/crop functions, after obtaining the local region, is the sub-region image enlarged before being fed back into the model? If so, to what size is it enlarged? On the other hand, providing only a local region might lead to incomplete context, which is problematic since the grounding for some tasks relies on surrounding context. How does the proposed method mitigate this issue of context loss?
5. For the GUI Grounding task, some current researches are exploring more efficient coordinate-free grounding methods, such as Attention-driven GUI Grounding: Leveraging Pretrained Multimodal Large Language Models without Fine-Tuning (https://arxiv.org/abs/2412.10840) and GUI-Actor: Coordinate-Free Visual Grounding for GUI Agents (https://arxiv.org/abs/2506.03143). GUI-Actor, based on Qwen-2-VL-7B-Instruct, achieved a performance of 40.9\% on ScreenSpot-Pro, while the proposed method, based on Qwen2.5-VL-7B-Instruct, only achieved 38.7\% in Table 3. Furthermore, the inference time and computational overhead of the proposed method might be higher. It is suggested that the authors provide a discussion on this point, explaining the necessity of the proposed multi-turn iterative grounding pipeline.
6. This paper lacks exploration of GUI Agent Systems. GUI grounding is not an isolated task; it is a critical component within a comprehensive GUI Agent System. MLLM-based GUI Agent Systems aim to increase the success rate of complex task completion by improving grounding quality, ideally through an economical, efficient, and unified integration into the MLLM inference process. This paper exclusively explores MLLM-based GUI grounding but fails to investigate the crucial next step: how to seamlessly integrate this capability into a functional GUI Agent System is a critical step.

### Questions
Please find the question in the Weaknesses section.

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
2

### Confidence
5