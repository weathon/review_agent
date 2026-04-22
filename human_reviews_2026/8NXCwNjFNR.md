# Presenting a Paper is an Art: Self-Improvement Aesthetic Agents for Academic Presentations

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 8, 2, 6

## Abstract
The promotion of academic papers has become an important means of enhancing research visibility. where the appeal of dissemination largely determines its effectiveness.
However, existing automated methods struggle limited storytelling, insufficient aesthetic quality, and constrained self-adjustment, making it difficult to achieve efficient and engaging dissemination. At the heart of those challenges is a simple principle: *there is no way to improve it when you cannot evaluate it right*.
To address this, we introduce **EvoPresent**, a self-improvement agent framework that unifies coherent narratives, aesthetic-aware designs, and realistic presentation delivery via virtual characters. 
Central to EvoPresent is **PresAesth**, a multi-task reinforcement learning (RL) aesthetic model that provides reliable aesthetic scoring, defect adjustment, and comparative feedback, enabling iterative self-improvement even under limited aesthetic training data. 
To systematically evaluate the methods, we introduce **EvoPresent Benchmark**, a comprehensive benchmark comprising: *Presentation Generation Quality*, built on 650 top-tier AI conference papers with multimodal resources (slides, videos  and scripts) to assess both content and design; and *Aesthetic Awareness*, consisting of 2,000 slide pairs with varying aesthetic levels, supporting joint training and evaluation on scoring, defect adjustment, and comparison. Our findings highlight that (i) High-quality feedback is essential for agent self-improvement, while initial capability alone does not guarantee effective self-correction.
(ii) Automated generation pipelines exhibit a trade-off between visual design and content construction. (iii) Multi-task RL training shows stronger generalization in aesthetic awareness tasks.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces EvoPresent, a multi-agent framework designed to automate academic presentations by integrating a self-improvement loop. This loop is driven by PresAesth, a new multi-task RL model trained to evaluate presentation aesthetics and provide feedback. The authors validate their system using the new EvoPresent Benchmark, demonstrating that EvoPresent outperforms current methods and produces presentations with improved narrative coherence and visual design, approaching human-level quality.

### Strengths
EvoPresent goes beyond just slides; it supports multiple formats (videos, scripts, slides). The system aims to automate the entire act of presentation, not just the materials.

PresAesth is not a simple binary classifier. Its strength lies in its multi-task RL foundation, enabling it to provide (1) reliable aesthetic scoring, (2) specific defect identification for adjustment, and (3) comparative feedback. This rich, actionable feedback is far more useful for an agent than a simple score.

The paper introduces a much-needed, systematic benchmark EvoPresent Benchmark, which includes multimodal resources like slides, videos, and scripts.

The authors conduct extensive experiments with detailed analysis, which shows a huge workload with high quality.

### Weaknesses
The framework's evaluation is limited to static slide aesthetics. The Checker Agent and PresAesth model score aspects like layout and visualization, but they do not appear to evaluate the dynamic animation of slide elements. This is a significant omission, as the sequence, timing, and quality of animations are crucial for a coherent narrative and professional delivery. Consequently, the self-improvement loop cannot identify or correct errors in animation logic, such as elements appearing in the wrong order, which limits the final quality of the presentation playback.
 
The paper states in Appendix E that the data was labeled by "approximately 30 volunteers with design backgrounds" and "2-3 annotators". The paper does not report any inter-annotator agreement metrics to validate the consensus among annotators.

The self-improvement loop is focused exclusively on slide creation, while the video generation is treated as a separate, non-iterative task. The paper would be significantly stronger if the iterative optimization process also included feedback on the delivery itself, such as the virtual character's pacing, intonation, or synchronization with the dynamic slide content.

### Questions
For suggestions for authors, please refer to the Weakness part.

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
4

### Summary
This paper introduces EvoPresent, a novel self-improving multi-agent framework designed to automate the creation of high-quality academic presentations. The system employs specialized agents for storyline construction, content enrichment, and visual design, all coordinated within a draft-feedback-refinement loop. A key innovation is PresAesth, a multi-task reinforcement learning model that provides aesthetic scoring, defect adjustment, and comparative feedback, enabling iterative self-improvement even with limited human preference data. The authors also contribute the EvoPresent Benchmark for systematic evaluation. Experiments demonstrate that the framework surpasses existing methods in generating presentations with superior narrative coherence and visual appeal, highlighting the critical role of high-quality aesthetic feedback for effective self-correction.

### Strengths
1. The paper introduces a novel and interesting task, which indeed leaves ample room for further exploration and development.

2. The overall aesthetic quality of the paper is quite good — the figures, charts, and tables are all well-designed and visually appealing.

3. The supplementary materials are sufficient and help present the work in a more comprehensive and detailed manner.

### Weaknesses
1. First of all, the dataset size seems quite limited. For Presentation Generation Quality, there are only about 650 samples, and for Aesthetic Awareness, just around 2,000 slides. That’s quite small for training or evaluating models of this scale.

2. Regarding the checker agent, it’s not clear how the threshold for evaluation is determined. Moreover, since different slides may have very different design intents — some emphasizing clarity, others emphasizing key visual highlights — are these aesthetic thresholds the same across all slides, or adaptive to content type?

3. When it comes to the collaboration among agents, it seems the paper doesn’t discuss what happens when one agent makes an error. How do you detect and handle such errors in real time, or mitigate their impact to improve overall cooperation efficiency?

4. Minor issue: in line 209, it seems that item (ii) is missing.

5. I’m a bit confused about how Generation Quality and Aesthetic Awareness are differentiated. You mentioned that Generation Quality is evaluated using your own trained PresAesth model to obtain aesthetic scores — but how exactly are these scores different from those used in Aesthetic Awareness?

6. During the data annotation process, although you mentioned using multiple annotators, it seems you didn’t assess their inter-annotator agreement (e.g., Fleiss’ Kappa). Given that this task heavily depends on subjective annotation quality, more details about annotator backgrounds, collaboration protocols, and agreement levels would significantly improve credibility. Also, showing a few annotation examples would help demonstrate data reliability.

7. In Figure 13, some human faces are shown without blurring. This potentially raises ethical concerns regarding privacy and data usage.

8. Overall, the task itself is interesting and has an innovative motivation. However, the technical side feels more like assembling multiple existing agents and applying reinforcement learning as a wrapper. There’s little algorithmic novelty, and the data annotation process doesn’t clearly offer a distinctive contribution. The demos look nice, but the practical value remains somewhat unclear, and the work seems limited mainly to the computer science domain.

### Questions
See above

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
4

### Summary
This paper introduces EvoPresent, a multi-agent framework for automatically generating academic presentations from research papers. The system consists of four sequential agents: (1) Storyline Agent for content extraction and narrative construction, (2) Scholar Agent for content enrichment, (3) Design Agent for layout and rendering in HTML format, and (4) Checker Agent for iterative quality improvement. Central to the approach is PresAesth, a multi-task reinforcement learning model based on Qwen-2.5-VL-7B, trained via Group Relative Policy Optimization (GRPO) to perform aesthetic scoring (1-10 scale), defect adjustment, and pairwise comparison. The authors construct the EvoPresent Benchmark with 650 annotated presentations from top AI conferences and 2,000 slide pairs for aesthetic evaluation. Experiments demonstrate that EvoPresent outperforms existing methods (PPTAgent, PresentAgent, Paper2Poster) and end-to-end LLM approaches (GPT-4o, GPT-5, Claude-4-Sonnet) across multiple metrics, achieving aesthetic scores of 8.05/10 (approaching the 8.50 oracle score) and 87.8% accuracy on aesthetic comparisons (versus 77.1% for GPT-4o). The iterative self-improvement mechanism enables the system to reach target quality in fewer iterations compared to baselines.

### Strengths
1、The four-agent architecture is well-structured with clear separation of concerns. The iterative draft-feedback-refinement loop with reversion to best previous version demonstrates thoughtful engineering to prevent quality degradation.

2、Automating academic presentation generation is a relevant task with clear applications in research dissemination.

3、Comprehensive evaluation including quantitative metrics (perplexity, ROUGE-L, layout balance), fine-grained content/design scores, human preference studies, and ablations. The results consistently demonstrate improvements over baselines.

### Weaknesses
1、 While the system integration is solid, the core components rely on standard techniques—sequential agent architectures, GRPO for preference learning, and VLM-based evaluation. The main contribution is engineering integration rather than algorithmic innovation. The multi-task RL formulation is relatively straightforward.

2、The reliance on HTML rendering for "maximal control and flexibility" limits practical utility, as most researchers work with PowerPoint or PDF formats. While conversion is mentioned, the fidelity and editability of converted presentations are not evaluated.

### Questions
see weakness

### Soundness
3

### Presentation
3

### Contribution
3
