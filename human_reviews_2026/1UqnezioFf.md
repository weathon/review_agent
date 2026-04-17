# ChartMaster: Boosting MLLMs for Chart Analysis through Data, Perception, and Reasoning Optimization

- Decision: Reject
- Scores: 4, 6, 4, 4

## Abstract
Multimodal Large Language Models (MLLMs) have demonstrated significant potential in understanding visual information, yet they often fall short in the complex domain of chart analysis. 
Existing models often struggle to accurately capture detailed visual elements and to perform efficient multi-step reasoning. 
To address these challenges, we introduce ChartMaster, a holistic framework that systematically advances chart analysis by jointly optimizing data, perception, and reasoning.
Our approach is built on three core innovations.
First, we construct ChartVerse, a large-scale synthetic dataset with diverse chart types, rendering styles, and reasoning levels.
Building on this foundation, we introduce a novel two-stage training paradigm: 
(i) Multi-Negative Direct Preference Optimization (MNDPO), which improves perceptual precision by training models to distinguish correct answers from carefully designed hard negative samples (i.e., plausible but incorrect alternatives); and (ii) Reinforcement Learning with Dynamic Length Reward (DLR), which adapts chain-of-thought reasoning to task complexity, encouraging concise solutions for simple queries and rigorous multi-step reasoning for complex ones. 
Extensive experiments across six benchmarks demonstrate that ChartMaster achieves state-of-the-art performance, surpassing prior chart-domain models and rivaling proprietary systems. These results highlight that coupling diverse data foundations with targeted perceptual and reasoning optimization provides an effective pathway toward robust chart understanding in MLLMs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces ChartMaster, a framework to boost MLLMs for chart analysis through optimizations in data, perception, and reasoning. Key components include: (1) ChartVerse, a 128k-instance synthetic dataset with diverse charts and QA pairs via a decoupled generation pipeline (95% error-free); (2) Multi-Negative Direct Preference Optimization (MNDPO), extending DPO with hard negatives for precise visual extraction; (3) Reinforcement Learning with Dynamic Length Reward (DLR) for adaptive reasoning depth. Trained in two stages, it achieves SOTA on six benchmarks, rivaling proprietary models.

### Strengths
The construction of ChartVerse is a contribution, providing a high-quality, diverse dataset (128k instances) with an LLM code-based synthesis pipeline that ensures better alignment between charts and QA pairs.

The holistic framework addresses key challenges in chart analysis (perception and reasoning) in a structured way, leading to consistent improvements across benchmarks in the main results (Table 1).

The ablation studies provide some insight into hyperparameters, such as the number of negatives in MNDPO (N=1 to 5) and the insensitivity of DLR to its temperature parameter τ.

### Weaknesses
The proposed MNDPO lacks significant novelty, as similar multi-negative extensions of Direct Preference Optimization have been explored in recent works[1], which also leverage multiple negatives for multimodal alignment. This makes MNDPO feel like a minor variant on multi-modal chart understanding tasks.

The Dynamic Length Reward (DLR) mechanism in the RL stage appears to have negligible practical impact. Ablations show it achieves the "best overall performance," but the gains are minimal or absent in terms of accuracy (no performance drop without it, per the text), and it primarily reduces average output length by only around 20 tokens across benchmarks. This reduction is not substantial enough to meaningfully improve efficiency or mitigate overthinking in real-world scenarios, especially given the variable nature of chart queries.

Closed-source models compared are not the latest, as current closed-source models come to GPT-5 or Claude-4.5, although I get that beating these models is not the goal.

Overall, the paper feels incremental rather than transformative. The core ideas rely heavily on established techniques (synthetic data generation, DPO variants, and RL for reasoning), with limited evidence of superior generalization or robustness beyond the reported benchmarks. The claims of rivaling proprietary models are strong but not deeply substantiated against the latest closed-source systems, and the work could benefit from more rigorous comparisons or analyses of failure cases.

[1]Chen et al. On Softmax Direct Preference Optimization for Recommendation. NeurIPS 2025

### Questions
See Cons

In line 356, I think ChartMaster is built upon Qwen2.5-7B-VL, not the Qwen2.5-7B-Instruct model, right? I suppose Qwen2.5-7B-Instruct is more likely referred to as the text model.

### Soundness
3

### Presentation
2

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
This paper introduces ChartMaster, a state-of-the-art chart reasoning model. The authors begin by identifying the limitations of existing Vision-Language Models (VLMs) in chart domains, specifically in visual perception and varying levels of reasoning. To address these challenges, they propose two distinct training techniques:

* **Multi Negative Preference Optimization:** This technique trains the model to accurately differentiate between correct and incorrect data perceived from chart images.
* **Reinforcement Learning with Dynamic Length Reward:** This method enables the model to adjust its response length based on the complexity of the given question.

The authors also present the ChartVerse dataset, which is generated from template YAML files and rendered into chart images. This dataset includes question-answer pairs categorized by defined difficulty levels. By training on this dataset using the above two training techniques, ChartMaster achieves state-of-the-art performance across multiple downstream tasks, including CharXiv, ChartQA, ChartQAPro, and ChartX.

### Strengths
The authors effectively identified two core limitations of Visual-Language Models (VLMs) in chart reasoning: visual perception and the varying degrees of reasoning required. These limitations motivated the development of two well-designed techniques:

* Multi-Negative Preference Optimization: This technique helps the model distinguish between correct and incorrect perceived values from chart images. A notable aspect is the generation of "hard negatives" (e.g., using adjacent values as distractors) which often confuse existing VLMs. I liked this approach and I believe it has very high potential. 


* Reinforcement Learning with Dynamic Length Reward: The authors introduced a carefully designed reward function that encourages the model to adapt its response length based on question difficulty. For simple extraction questions, the reward promotes concise responses, while for complex reasoning questions, it encourages longer, more in-depth thinking.

By combining these two training techniques, the resulting model, ChartMaster, achieved state-of-the-art results on several downstream tasks, including ChartQA, ChartX, ChartQAPro, and CharXiv. I also believe the ChartVerse dataset could be valuable to the chart research community. 

Overall, I believe the proposed solutions are both well motivated and well-designed.

### Weaknesses
* The authors have not presented any evaluation or training experiments to demonstrate the superiority of their ChartVerse dataset compared to existing ones. Therefore, I recommend the following experiment: Fine-tune the same base model using ChartVerse and other publicly available datasets (e.g., ChartReasoner, ChartGemma, TinyChart) and then compare their respective performances.


* The authors claim in lines 186-192 that using YAML file templates is superior to instructing an LLM to generate the underlying code. However, no direct, "apple-to-apple" comparison has been provided to substantiate this. I am concerned that relying on a fixed set of templates could limit the diversity of the generated dataset, ChartVerse. Therefore, I propose the following experiment:
  * Begin with identical seed data. Render this data once using the YAML file technique and again by instructing an LLM (such as Gemini) to perform the rendering. Subsequently, generate QA pairs from each format separately and then fine-tune the base model on each resulting dataset. Finally, compare the performance differences to demonstrate the impact of each approach.



* Generating QA solely from YAML files, without accompanying chart images, could restrict the dataset's visual questions (e.g., "what is the sum of red bars?"). Such questions are crucial for chart reasoning, as the Language Model (LLM) would lack the necessary visual information.

### Questions
Please see weaknesses above.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper presents ChartMaster, a synthetic dataset with 128K examples, containing charts with complex and detailed visual elements across various reasoning levels. Next, the paper introduces MNDPO, a modified version of DPO that leverages multiple negatives to improve models. Finally, the paper uses RL to further enhance performance while maintaining concise reasoning traces. Overall, the paper addresses important challenges and combines various training regimes, but the data synthesis pipeline and several design choices in MNDPO are questionable.

### Strengths
- The paper addresses an important gap in chart analysis, namely imprecise visual perception and limited multi-step reasoning.
- Results across multiple evaluation benchmarks show significant improvements.

### Weaknesses
**Data synthesis pipeline**
- The pipeline is not detailed.
- How are the YAML templates defined in the first place? Are they synthetically curated or retrieved from various sources? If they are synthetically curated, how is visual diversity ensured?
- Which LLMs are used for YAML completion and QA generation? Line 220 mentions “API of several LLMs” but does not specify their names.
- Line 199 states “structured records are converted into Python code.” How is this done?
Some examples of the full pipeline, perhaps a figure highlighting how the charts and QA pairs are generated, would improve clarity.

**MNDPO design choices**
- Why is MNDPO used instead of DPO? 
- Are there experiments showing that MNDPO performs better for chart understanding tasks? Although Table 2 suggests that introducing more negatives helps, this is shown on only one dataset and the gains are not significant.
- It is unclear how hard negatives are generated. Examples of various perturbations applied to charts and texts are needed for better understanding.

### Questions
**Experiments**

- Line 347 mentions “MNDPO is applied using only Level 1 and Level 2 questions.” Why is this the case? What happens if MNDPO is applied to the full dataset—does the performance degrade or improve?
- The paper only uses Qwen-2.5-VL-7B-Instruct as the base model. However, to better demonstrate the effectiveness of the proposed methods, results on at least one additional base model—preferably from a different family—are necessary.
- How is ChartMaster superior to existing chart datasets? Are there any results using MNDPO and RL on other datasets, or any qualitative comparisons?

**Minor**
- Was any human evaluation conducted to verify the difficulty levels?
- It would be good to check for any dataset leakage into test benchmarks, either through image or text semantic similarity.
- Several papers also constrain generation length during DPO. Are there any experiments on constraining generation length in MNDPO?

### Soundness
2

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
This paper introduces ChartMaster, a framework to improve MLLM performance on chart analysis. The framework consists of three main components: (1) ChartVerse, a new large-scale synthetic dataset; (2) Multi-Negative Direct Preference Optimization (MNDPO), a variant of DPO that uses hard negative samples; and (3) Reinforcement Learning with Dynamic Length Reward (DLR), a reward mechanism that encourages concise reasoning for simple queries and multi-step reasoning for complex ones. The authors show that this framework achieves state-of-the-art (SOTA) performance on six chart benchmarks.

### Strengths
1. The framework achieves state-of-the-art (SOTA) performance on six chart benchmarks.
2. Their applied reasoning and perceptual optimization methods help the model to improve performance. 
3. Introduces a new large-scale synthetic dataset for MLLM training.

### Weaknesses
The major limitation of this paper is its limited novelty:

(i) MNDPO is quite a straightforward extension of DPO to multiple negatives. 

(ii) DLR is also a simple heuristic that penalizes deviation from the shortest correct answer. 

(iii) ChartVerse contains synthetic data, which is also a common practice.

### Questions
Justify the novelty of the work. Add further comparisons with existing similar literature and demonstrate how this work is a novel contribution.

### Soundness
3

### Presentation
3

### Contribution
2
