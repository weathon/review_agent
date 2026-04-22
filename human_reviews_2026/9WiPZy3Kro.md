# Grounding Computer Use Agents on Human Demonstrations

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
Building reliable computer-use agents requires grounding: accurately connecting natural language instructions to the correct on-screen elements. While large datasets exist for web and mobile interactions, high-quality resources for desktop environments are limited. To address this gap, we introduce GroundCUA, a large-scale desktop grounding dataset built from expert human demonstrations. It covers 87 applications across 12 categories and includes 56K screenshots, with every on-screen element carefully annotated for a total of over 3.56M human-verified annotations. From these demonstrations, we generate diverse instructions that capture a wide range of real-world tasks, providing high-quality data for model training. Using GroundCUA, we develop the GroundNext family of models that map instructions to their target UI elements. At both 3B and 7B scales, GroundNext achieves state-of-the-art results across five benchmarks using supervised fine-tuning, while requiring less than one-tenth the training data of prior work. Reinforcement learning post-training further improves performance, and when evaluated in an agentic setting on the OSWorld benchmark using o3 as planner, GroundNext attains comparable or superior results to models trained with substantially more data. These results demonstrate the critical role of high-quality, expert-driven datasets in advancing general-purpose computer-use agents.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces GROUNDCUA, a large-scale human-expert-annotated dataset for desktop UI grounding. The dataset is comprehensive, covering 56K screenshots from 87 applications with over 3.56M dense annotations. The authors also present GROUNDNEXT, a series of models (3B and 7B) trained on this dataset using a two-stage SFT-RL process. GROUNDNEXT achieves state-of-the-art performances on five grounding benchmarks, significantly outperforming models trained on datasets 10x larger. The paper argues for the superior value of high-quality, dense, expert-driven data over sheer data quantity for this task.

### Strengths
- The GROUNDCUA dataset. It collects grounding samples during human-driven task execution. The inclusion of complex and diverse desktop screen states directly addresses a major gap in existing resources.
- Sample efficiency. This is another valuable feature of the dataset. Training with significantly fewer samples, GROUNDNEXT can outperform models with much more samples.

### Weaknesses
- Although during data collection, synthetic tasks are written by human and conducted by human in real environments, only the grounding samples automatically extracted from each step is collected for training a grounding-only model. Can the authors elaborate more on the reasons and rationales behind such design?
- A highly relevant work on scaling desktop UI data and training a grounding model is lacking discussion in the paper: Aria-UI: Visual Grounding for GUI Instructions.
- Limited RL impact: 66.4->68.4 and 69.2->70.5. Considering that in GTA1, RL training brings very significant improvements, what's the underlying factors that lead to limited RL impact on GROUNDNEXT after SFT? Would it be relevant to the data? Also, it might be valuable to see how the GTA1-style heavy RL training for grounding would work with the GROUNDCUA dataset.
- Is there a timeline for open-sourcing the tasks and trajectories?

### Questions
Please see weaknesses.

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
4

### Summary
The paper introduces GROUNDCUA, a large, expert annotated desktop UI grounding dataset built from human demonstrations across 87 applications and 56K screenshots with over 3.56M element annotations. The authors convert these dense annotations into contextual instruction. Based on this dataset, a series of 3B/7B models are also trained, which are termed GROUNDNEXT. They are trained with supervised fine-tuning, followed by a light RL. On five benchmarks spanning desktop, web, and mobile, GROUNDNEXT achieves the top average SFT performance at both model sizes, with RL offering small but consistent gains.

### Strengths
GROUNDCUA features high quality, human verified supervision data. The elements are hand labeled from expert task demonstrations, giving reliable targets rather than noisy accessibility or synthetic signals.

The dataset contains dense and fine grained coverage of samples, with screens averaging 64 labeled elements to support precise grounding at pixel-level granularity.

The SFT model has demonstrated strong performance with modest data volume. GROUNDNEXT tops SFT baselines across five benchmarks for both 3B and 7B sizes, highlighting data quality over raw scale.

### Weaknesses
As a dataset paper, it would be great that the authors can provide a link to the dataset.

### Questions
Listed above.

### Soundness
4

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
This paper tackles the grounding problem for computer-use agents. Basically helping AI systems correctly identify which UI element to click when given natural language instructions. The main contribution is GROUNDCUA, a massive dataset with 56K desktop screenshots from 87 applications. The dataset coverage is pretty broad, from office software to creative tools to development environments.

### Strengths
GROUNDCUA fills a major gap in desktop grounding data. Most existing datasets focus on web or mobile interfaces, but desktop apps are way more complex with tiny icons and dense layouts. The human expert approach is smart - instead of automated scraping, they had people actually use the software and annotate everything they see.

The results show that data quality beats data quantity. The two-stage approach with supervised fine-tuning plus reinforcement learning is pretty straightforward. No complex reward engineering needed.

Even though the models only see desktop data during training, they transfer surprisingly well to mobile and web interfaces. This suggests that learning good grounding skills on complex desktop environments helps with simpler interfaces too.

### Weaknesses
The paper describes three instruction types but doesn't analyze how they affect model performance differently. The instruction generation relies heavily on prompting Qwen2.5-VL-72B. But there's no discussion of prompt sensitivity or failure cases.

What happens when the LLM generates wrong instructions? The paper mentions using "about 100 templates" for textual elements and "120 templates" for general ones. But it doesn't say which templates work best or how template diversity impacts training.

The discrete reward function with six levels seems somewhat arbitrary. Why these specific thresholds (-0.5, -0.1, 0, 0.1, 0.5)? The paper mentions trying continuous and binary rewards but doesn't provide detailed comparisons.

The RLOO method choice over GRPO is mentioned briefly but lacks analysis. How sensitive is the approach to the group size (n=8) and batch size (64) choices? The modest RL improvements suggest the reward signal might not be optimal. More sophisticated reward formulations could yield better gains.

The error analysis in Appendix E is brief and doesn't systematically categorize failure modes across domains. The paper mentions some errors come from "limited domain knowledge", it would be helpful to explore this.

### Questions
Please see the weakness section above for more detailed analysis.

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
This paper introduces GROUND-CUA, a large-scale, human-annotated grounding dataset, and demonstrates its effectiveness using two-stage, state-of-the-art grounding models evaluated on several desktop benchmarks. The dataset offers diverse applications, detailed annotations, and broad coverage of action types and pixel densities. The paper’s introduction of the concept of phased rewards—which encourages models to gradually move toward the ground truth—is particularly innovative and interesting.

### Strengths
1. A new SOTA performance on grounding tasks: with only one-tenth of the training data compared to previous methods, GROUNDNEXT achieves state-of-the-art performance on desktop benchmarks while also generalizing to OOD categories, demonstrating the effectiveness and huge contribution of this grounding dataset.
2. Action and task types that match real-world computer tasks: demonstrations are collected from real-world user experiences, and the applications span diverse real-world use cases, helping to narrow the gap between synthetic and real-world grounding tasks.
3. The instructions generated for fine-tuning target diverse descriptions — including directness, intent, and location — thereby enhancing the generalization ability of the fine-tuned models.

### Weaknesses
1. Although the authors demonstrate their dataset's effectiveness on the grounding task, they did not show whether this grounding ability transfers to improved performance on computer-use tasks.
2. After the model underwent reinforcement learning on 10K samples, the improvement was limited, casting doubt on the authors' motivation for using the reinforcement learning step.

### Questions
1. Why choose RLOO rather than the normally used GRPO? Are there any preliminary experiments that show RLOO is better?

2. When does $D_{norm}$ in the reward definition exceed 0.5? Does that imply the grounding box is approximately the same size as the screenshot? Are you using the bounding box center as the reference point? Could you provide concrete examples for the different reward types?

3. I would be grateful if the authors could provide some analysis on the dataset’s scaling trend: if we sample datasets of different sizes and perform SFT on the model, will we observe a similar scaling curve in the grounding domain?

4. It would be better if the test results in the main table had confidence intervals.

### Soundness
3

### Presentation
3

### Contribution
3
