# UI-Ins: Enhancing GUI Grounding with Multi-Perspective Instruction as Reasoning

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 2

## Abstract
GUI grounding, which maps natural-language instructions to actionable UI elements, is a core capability of GUI agents. Prior works largely treats instructions as a static proxy for user intent, overlooking the impact of instruction diversity and quality on grounding performance. Through a careful investigation of existing grounding datasets, we find a 23.3% flaw rate in their instructions and show that inference-time exploitation of instruction diversity yields up to a substantial 76% relative performance improvement.
In this paper, we introduce the Instruction-as-Reasoning paradigm, treating instructions as dynamic analytical pathways that offer distinct perspectives and enabling the model to select the most effective pathway during reasoning. To achieve this, we propose a two-stage training framework: supervised fine-tuning on synthesized, diverse instructions to instill multi-perspective reasoning, followed by reinforcement learning to optimize pathway selection and composition.
Our resulting models, UI-Ins-7B and UI-Ins-32B, achieve state-of-the-art results on five challenging grounding benchmarks and exhibit emergent reasoning, selectively composing and synthesizing novel instruction pathways at inference. In particular, UI-Ins-32B attains the best grounding accuracy, scoring 87.3% on UI-I2E-Bench, 57.0% on ScreenSpot-Pro, and 84.9% on MMBench-GUI L2. Furthermore, our model demonstrates strong agentic potential, achieving a 74.1% success rate on AndroidWorld using UI-Ins-7B as the executor. Our in-depth analysis reveals additional insights such as how reasoning can be formulated to enhance rather than hinder grounding performance, and how our method mitigates policy collapse in the SFT+RL framework. All code and models are released.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses the GUI grounding task by proposing a novel paradigm called "Instruction as Reasoning" (IR). The authors argue that existing methods treat instructions as static user intent, overlooking the value of instruction diversity. The core contribution is a two-stage training framework: first, using Supervised Fine-Tuning (SFT) on a cleaned and augmented dataset (containing multi-perspective instructions) to teach the model to generate intermediate reasoning paths; second, employing a Reinforcement Learning (RL) stage to optimize the model to autonomously select the best reasoning path for a given task. The authors introduce the UI-Ins-7B and UI-Ins-32B models, achieving SOTA performance on five major benchmarks and demonstrating an emergent ability to combine and generate novel reasoning paths in new scenarios. I think this is a well-motivated and experimentally sound paper.

### Strengths
1. The models achieve SOTA performance on five benchmarks, including MMBench-GUI L2 and UI-I2E-Bench.
2. The "Instruction as Reasoning" (IR) paradigm is novel. A key insight of the paper is that traditional "Free-Form Reasoning" (FFR) may degrade grounding performance, while the proposed structured IR significantly improves it.
3. The proposed two-stage SFT+RL (GRPO) framework is methodologically sound. The SFT stage effectively teaches the model to understand and generate diverse reasoning paths, while the RL stage optimizes the path selection strategy.
4. The experimental evidence supporting the method is convincing. The ablation studies are strong, validating the necessity of both the SFT and RL stages, the importance of the intermediate reasoning step, and the role of IR-SFT in stabilizing RL training.

### Weaknesses
1. **Choice of GRPO**  
   The paper employs GRPO for the RL stage but does not sufficiently justify why GRPO was chosen over other reinforcement learning algorithms. A comparative discussion or empirical evidence supporting this choice would strengthen the methodological soundness.

2. **Definition of "Optimal" Selection**  
   The paper claims that the RL stage learns to select the "optimal" analysis perspective. However, there is insufficient analysis explaining:
   - Under which contexts the model selects specific perspectives  
   - Why these selections are considered "optimal"  
   
   A deeper examination of selection patterns and their correlation with performance would help substantiate this claim.

### Questions
1. **Rationale for Selecting GRPO**  
   Provide more justification for using GRPO in the RL stage. Include comparisons or discussions with other popular RL algorithms to clarify the advantages or unique suitability of GRPO for this task.

2. **Qualitative Analysis of Emergent Reasoning**  
   Expand the qualitative analysis to better illustrate the phenomenon of "emergent reasoning." More detailed case studies are needed to show when and why the model synthesizes multiple perspectives or formulates entirely novel instructional perspectives.

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
4

### Summary
This paper is well-written. It addresses the task of GUI grounding. The authors make two primary observations:

1. Existing GUI grounding datasets suffer from a high flaw rate (a manually verified 23.3%).
2. Current models fail to leverage "instruction diversity," the concept that a single intent can be expressed from multiple perspectives (e.g., by appearance, function, or location). The authors demonstrate that simply exploiting this diversity at inference time can yield up to a 76% relative performance improvement.

Based on these insights, the paper introduces the "Instruction as Reasoning" (IR) paradigm, which treats instructions not as static inputs but as dynamic, selectable "analytical pathways". To implement this, the authors propose a high-fidelity data pipeline to clean existing datasets and augment them with multi-perspective instructions. They then introduce a two-stage training framework. The resulting models, UI-Ins-7B and UI-Ins-32B, achieve new SOTA performance on five challenging benchmarks.

### Strengths
- The paper presents solid empirical foundation. The preliminary analyses in Section 2 are data-driven conclusions from well-designed controlled experiments. The solid investigation provides an irrefutable "why" for the rest of the paper.
- The "Instruction as Reasoning" (IR) paradigm is a key conceptual contribution. It moves beyond generic "think step-by-step" reasoning and proposes a *structured* form of reasoning that is directly derived from the problem's multimodal context.
- The SFT+RL two-stage framework is an elegant solution. It correctly identifies that the model must first learn *how* to perform a new skill (SFT for reasoning generation) before it can learn *when* to apply it (RL for reasoning selection) .
- The method is not just elegant; it is highly effective. It achieves SOTA on five distinct grounding benchmarks, with particularly strong gains on "advanced" and "implicit" instructions, indicating a deeper semantic understanding. The SOTA on AndroidWorld proves that these offline grounding gains translate directly to improved online agent performance.
- A significant finding is the model's emergent ability to synthesize multiple perspectives and generate entirely new analytical angles not seen in the SFT data. This suggests the SFT+RL framework doesn't just teach the model to memorize four perspectives but encourages a more general and flexible reasoning capability.

### Weaknesses
- In 3.2 Multi-Perspective Instruction Augmentation, you mention that you leverage GPT-4.1 (OpenAI, 2025) to generate new instructions from the four fundamental analytical perspectives identified: appearance, functionality, location, and intent. However, you haven’t presented further analysis on the crafted dataset. The paper lacks a systematic comparison between the original dataset and the augmented one to prove the effectiveness.
- This paper only conducts experiments on one online agent benchmark, especially in the context of Android. This might be insufficient to prove the performance of the UI-Ins in dynamic GUI environments

### Questions
- Have you explored any other online agentic benchmark and applied your model on different platforms (e.g., Web/Desktop, etc.)?
- Could you share more details of the rationale behind the design of the four components (appearance, functionality, location, and intent) during the data augmentation? Have you done any ablation studies on different components and evaluated their significance?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper aims to address the lack of diversity in reasoning for GUI grounding. It proposes a data synthesis pipeline and a two-stage training framework. The resulting model achieves strong performance on grounding tasks and improves the accuracy of existing methods in real-world navigation scenarios.

### Strengths
1. This paper offers a novel perspective: instruction and reasoning diversity.
2. The overall presentation of the paper is fairly clear.
3. The experimental section is comprehensive.

### Weaknesses
1. One of the core contributions of the paper is the construction of a data synthesis pipeline. However, the description in Section 3 remains unclear:

    1.1 There is a lack of description of the raw dataset formats. Although Tables 9 and 10 in the Appendix describe the datasets used for training, they do not provide the original dataset details. For example, what is the original size of OSAtlas, and which subset of OSAtlas is used? In addition, the dataset meta information though full pipeline is unclear. For example, how to process the multi-step dataset AgentNet. How much data is remained at each stage.


    1.2 The formalization of the training data synthesis pipeline is missing. Since most grounding datasets consist of a single image paired with multiple element annotations, the statement on Line 215, “For each data instance, the model receives the screenshot with the highlight,” is ambiguous. Does “data instance” refer to one image together with its set of element annotations, or to a single element?

2. The analysis of the online evaluation experiments is relatively weak. While the overall performance improves, it is unclear which specific capabilities are enhanced; more detailed analysis is needed.

### Questions
1. In Line 918, is the Action Space in the SFT training prompt fixed to only one action?

2. The paper states that during the SFT stage, the diversified instruction serves as the reasoning. In the actual SFT training data, is the reasoning directly the diversified instruction? If so, does the instruction then need to be replaced with a standard instruction? The construction process of the data samples is not sufficiently clear.

3. In the data synthesis pipeline, the diversified instruction is used as reasoning. Is the instruction used for training also augmented accordingly?

4. AgentNet is a multi-step trajectory dataset that includes many non-grounding actions. How is this type of dataset converted into the Grounding format?

5. Sampling of SFT data: According to Equation (1), the final reasoning used for training is sampled from four perspectives. Why not use the full dataset directly for training?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper first identifies two types of flaws in current GUI grounding datasets: insufficient prompt diversity and poor prompt quality. The authors conduct preliminary experiments showing that pairing a model with the instruction perspective most suitable for a given task can boost grounding performance even without retraining. They then develop a data-processing framework to filter out unclear instructions and increase perspective diversity, and introduce "Instruction-as-Reasoning," a two-stage training pipeline to further improve grounding. Experiments demonstrate new state-of-the-art results on five major grounding benchmarks.

### Strengths
1. New SOTA performance across five grounding benchmarks
2. Propose improving grounding ability by matching the most suitable instruction type to each task.

### Weaknesses
1. I am doubtful about the contribution of the "Instruction as Reasoning" paradigm. I think this paradigm is essentially just a combination of "instruction augmentation + chain-of-thought generation". Compared to existing multi-perspective instruction augmentations (such as Jedi, UI-E2I-Synth, etc.), does this work possess genuine novelty at the algorithmic level, or is it merely an innovation in framework naming?
2. Also, the paper claims "instructions as dynamic reasoning pathways," but in practice, it primarily employs a two-stage training mechanism of SFT+GRPO. What is the fundamental difference between this mechanism and the typical RLHF/GRPO structure? Is it simply a renaming of "prompt diversity" to "reasoning diversity"?
3. The figure and the table in this paper are not perfect and lack inspiration for the readers. For example, Figure 6 is unclear and should be either condensed or supplemented with more useful information. I am interested in how the model selects optimal strategies across different perspectives during RL, which the figure does not show. Researchers in the field are likely already familiar with the "Advantages" and "Objectives" sections, making them redundant.

### Questions
1. Do all baselines use the same data cleaning and preprocessing? The paper points out that the original data has a 23.3% defect rate. If the baseline model is trained on uncleaned data, while this method uses cleaned and augmented data, the performance improvement may come from the difference in data quality rather than algorithm improvement.
2. When related to "ambiguous & mismatch", did the authors measure human–AI agreement?
3. In Figure 6, if the author could choose a clearer screenshot of the input image to explain, that would be better.

### Soundness
2

### Presentation
2

### Contribution
2
