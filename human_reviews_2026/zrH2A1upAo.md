# GuirlVG: Incentivize GUI Visual Grounding via Empirical Exploration on Reinforcement Learning

- Decision: Accept (Poster)
- Scores: 4, 4, 6, 6

## Abstract
Graphical user interface visual grounding (GUI-VG)—a core capability for GUI agents—has primarily relied on supervised fine-tuning (SFT) of multimodal large language models (MLLMs), demanding extensive data curation and significant training costs. However, as MLLMs continue to advance and even cover GUI domains during pretraining, the necessity of exhaustive SFT post-training becomes increasingly questionable. Meanwhile, the recent successes of rule-based reinforcement fine-tuning (RFT) suggest a more efficient alternative. However, despite its promise, the optimal manner of RFT for GUI-VG remains unexplored. To bridge this gap, we introduce GuirlVG, a reinforcement learning–based GUI-VG method built on a systematic empirical study and a novel stabilization technique. Preliminarily, we find that naive application of RFT underperforms the SFT baseline, motivating a deeper exploration of RFT. First, we decompose RFT into its core components and analyze the optimal formulation of each. Second, as part of this exploration, we propose a novel Adversarial KL Factor that dynamically stabilizes training to mitigate reward over-optimization. Third, we further explore the training configurations of RFT to enhance the effectiveness. Extensive experiments show that GuirlVG, with only 5.2K training samples, outperforms SFT methods trained on over 10M samples, achieving a +7.7% improvement on ScreenSpot, a +17.2% improvement on ScreenSpotPro and 91.9% accuracy on ScreenSpotV2.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
this manuscript introduces GuirlVG, a rule-based reinforcement fine-tuning (RFT) method designed to address the high costs and extensive data requirements of traditional supervised fine-tuning (SFT) for graphical user interface visual grounding (GUI-VG) tasks. The core of GuirlVG lies in the systematic empirical study and improvement of Group Relative Policy Optimization (GRPO) to achieve efficient training. Experimental results demonstrate that GuirlVG, using only 5.2K training samples, surpasses SFT methods that use over 10M samples across three major benchmarks. It showed a +7.7% improvement on ScreenSpot, a +17.2% improvement on ScreenSpot-Pro, and achieved a 91.9% accuracy rate on ScreenSpot-V2. These findings confirm that RFT is a more data-efficient and generalizable post-training solution for GUI-VG.

### Strengths
1.Achieves performance far exceeding SFT with significantly fewer training samples.

2.Provides a relatively comprehensive analysis of the impact of various parameters in the GRPO training process on the final performance.

### Weaknesses
1.The paper states its aim is to provide guidance for the design of Reinforcement Fine-Tuning (RFT) in GUI-Visual Grounding (GUI-VG) tasks. However, the final performance of its 7B model shows a significant gap compared to other 7B models on the current screenspot-pro leaderboard(such as SE-GUI-7B, GTA1-7B). This makes it difficult to be convinced of the validity of the paper's conclusions.

2.Although the work explores various parameters within the GRPO training process, it does not introduce any fundamental innovations to the algorithm itself. This significantly diminishes the work's originality.

3.Furthermore, the parameter settings are likely strongly dependent on the specific dataset, model architecture, and model scale used. Therefore, the conclusions drawn in the paper may not be generalizable. More experimental results are needed to verify the generalizability of these findings.

### Questions
please see weakness.

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
3

### Summary
This paper presents GuirlVG, a RL-based framework for graphical user interface visual grounding (GUI-VG). Through a systematic empirical study, the authors identify critical factors influencing RFT success in GUI-VG and introduce key improvements: a Soft Reward Function (SRF), In-Bbox point reward, and a novel Adversarial KL Factor to stabilize training. Using as few as 2K–5.2K samples, GuirlVG outperforms strong SFT baselines, achieving impressive results.

### Strengths
- Comprehensive empirical study

The paper systematically dissects RFT components, including reward design, KL penalty, fine-tuning method, and prompt structure, offering rare empirical clarity in a field often driven by ad hoc innovation.

- Novel stabilization mechanism

The Adversarial KL Factor dynamically scales the KL penalty, effectively mitigating reward over-optimization—a notable technical contribution to GRPO-style RL for multimodal models.

- Strong empirical results

GuirlVG achieves state-of-the-art accuracy on multiple GUI grounding benchmarks with orders-of-magnitude less data, convincingly demonstrating the data efficiency of RFT.

- Clarity and transparency

Each ablation is well-motivated. 

- Practical insight for the community

The findings (e.g., point-based rewards outperforming bbox IoU, LoRA matching full FT, resolution prompting effects) are valuable practical takeaways likely to influence future GUI agent training.

### Weaknesses
The main reason I gave a score of 4 is that the scale of the empirical experiments are not sufficient to make a strong conclusion:

- The experiments are only done on Qwen2.5-VL. While I understand the Qwen-VL series is probably the only modern model architecture choice in the field, more experiments are required to find out if the findings in the paper are universal or model-specific. For example, Finding 5 says "LoRA offers comparable performance to full fine-tuning", but is it the case with Qwen2-VL which is not extensively trained on the task? Even on the Qwen2.5-VL architecture, there are some choices to experiment with, such as UI-TARS.

- Some experiments are too small to draw a conclusion. For example, in Table 6, only group size = {6, 8} and batch size = {1, 4} are tested, with a total of only three runs. The conclusions from such experiments can be very brittle.

### Questions
Please see weaknesses.

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors applied RFT to the GUI-agent which is underexplored. the study step-by-step analyze the intermediate resutls and propose an efficient way that outperforms SFT on the tasks of interest.

### Strengths
1. It is a well-motivated study
2. The experimental results are sufficient to convince its effectiveness 
3. The methodology is efficient, clear, and easy to follow

### Weaknesses
1. What about the performances of other steps? Any indications from those?
2. Could be more ablations on hyperparameters of the config.

### Questions
See weaknesses

### Soundness
2

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
GUIRLVG addresses GUI visual grounding (GUI-VG) by proposing a rule-based RFT framework, challenging the dominant supervised fine-tuning (SFT) paradigm that requires massive labeled data. The paper first finds that naive RFT underperforms SFT, then conducts a systematic empirical study to optimize RFT’s core components. 

Key innovations include:
1. Replaces rigid binary rewards with fractional credit for partial format compliance, reducing training noise.
2. Aligns directly with GUI-VG’s goal (locating actionable points) for better performance than bounding box/IoU-based rewards.
3. Adopts LoRA for fine-tuning (25x faster than full fine-tuning with negligible performance loss), optimal group/batch sizes (6/4), and resolution prompting (withheld during training, added at test time).

Trained on only 2K–5.2K samples, GUIRLVG outperforms SFT baselines trained on 1M–13.58M samples: achieving 88.7% accuracy on ScreenSpot, 91.9% on ScreenSpotV2, and 36.1% on ScreenSpotPro.

### Strengths
1. This paper introduces a novel framework that incorporates Rule-based Reinforcement Fine-Tuning (RFT) into GUI visual grounding for the first time. Its results outperform other Supervised Fine-Tuning (SFT)-based methods, providing a valuable indication for future research directions in visual grounding.

2. The results presented in Section 4 are highly impressive: using only 2K or 5.2K training samples, the framework achieves superior performance compared to previous SFT-based methods that rely on massive datasets and larger model sizes. If these findings can be reproduced, they will undoubtedly reshape the future direction of GUI visual grounding (GUI-VG) research.

3. The research motivation is fully clarified by addressing key questions posed by the authors, while the literature review and preliminary analysis further elaborate on this motivation. The paper follows a clear and logical storyline, with the problem formulation and proposed pipeline progressing in a coherent manner.

### Weaknesses
1. Although the empirical research approach and writing style are acceptable, the theoretical details of the design and calculation processes need more explicit elaboration. 

For instance, in Section 3.2, when proposing the “Soft Reward Function,” a specific mathematical formulation would be preferable to purely natural language descriptions. This issue persists in other methodology subsections. Otherwise, this presentation reads more like an application report, which weakens the theoretical novelty of the proposed pipeline.

2. A critical oversight is the lack of comparison with existing RFT-based GUI-VG methods. 

As mentioned in section 2.1, ”While prior works have proposed various modeling choices for RFT-based GUI-VG, these advances often emphasize reward function novelty or performance improvements without systematic examination of underlying design factors.”  Although,  the authors claim that"differences in data, training, and models would yield limited rigorous conclusions in systematic experiments."  Given that several RFT-based GUI-VG methods (e.g., GUI-R1, VLM-R1, UnivG-R1) share core design foundations (e.g., GRPO-based reinforcement fine-tuning), I think it is not impossible to align those models with GuirlVG’s experimental setup for fair comparison.

Omitting these comparisons undermines the paper’s claim of advancing RFT for GUI-VG, as readers cannot determine whether GuirlVG’s performance gains stem from novel technical contributions or only better hyperparameter tuning or data selection. Additionally, the paper fails to distinguish GuirlVG from these prior RFT methods theoretically: it does not explicitly address how its design choices differ from or improve upon the reward function novelty or modeling choices of existing RFT-based approaches—an essential detail to validate its theoretical novelty.

### Questions
Overall, this is a strong, readable paper with clear motivation and well-structured proposed solutions. Its core strengths lie in a coherent research narrative, impactful empirical findings on RFT for GUI-VG, and compelling results demonstrating superior data efficiency over SFT methods.

However, its theoretical novelty is somewhat weakened by two key omissions: the lack of mathematical details for proposed modules (e.g., the Soft Reward Function/SRF) and insufficient theoretical analysis, evaluation, and comparison with other RFT-based models. The authors should clarify these aspects to strengthen the work’s rigor.

Additionally, including a visual overview of the overall GuirlVG pipeline would greatly benefit readers, enabling them to grasp the end-to-end design more straightforwardly.

I would like to raise my evaluation score if the authors supplement the paper with more theoretical details (or designs) and thorough comparisons with existing RFT-based models.

### Soundness
4

### Presentation
3

### Contribution
4
