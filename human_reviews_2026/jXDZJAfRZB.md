# Seeing Across Views: Benchmarking Spatial Reasoning of Vision-Language Models in Robotic Scenes

- Decision: Accept (Poster)
- Scores: 8, 6, 6, 6

## Abstract
Vision-language models (VLMs) are essential to Embodied AI, enabling robots to perceive, reason, and act in complex environments. They also serve as the foundation for the recent Vision-Language-Action (VLA) models. Yet most evaluations of VLMs focus on single-view settings, leaving their ability to integrate multi-view information underexplored. At the same time, multi-camera setups are increasingly standard in robotic platforms, as they provide complementary perspectives to mitigate occlusion and depth ambiguity. Whether VLMs can effectively leverage such multi-view inputs for robotic reasoning therefore remains an open question. To bridge this gap, we introduce \textbf{MV-RoboBench}, a benchmark specifically designed to evaluate the multi-view spatial reasoning capabilities of VLMs in robotic manipulation. MV-RoboBench consists of 1.7k manually curated QA items across eight subtasks, divided into two primary categories: spatial understanding and robotic execution. We evaluate a diverse set of existing VLMs, including both open-source and closed-source models, along with enhanced versions incorporating Chain-of-Thought (CoT)-inspired techniques. The results show that state-of-the-art models remain far below human performance, underscoring the substantial challenges VLMs face in multi-view robotic perception. Additionally, our analysis uncovers two key findings: (i) spatial intelligence and robotic task execution are positively correlated in multi-view robotic scenarios; and (ii) strong performance on existing general-purpose single-view spatial understanding benchmarks does not reliably translate to success in the robotic spatial tasks assessed by our benchmark. We release MV-RoboBench as an open resource to foster progress in spatially grounded VLMs and VLAs, providing not only data but also a standardized evaluation protocol for multi-view embodied reasoning.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces a novel and valuable benchmark called MV-RoboBench, specifically engineered to test the multi-view spatial reasoning capabilities of Vision-Language Models (VLMs) within the context of robotic manipulation. Diverse existing VLMs, along with enhanced versions incorporating CoT-inspired techniques, are evaluated on the benchmark.

### Strengths
1. The benchmark advances beyond traditional multi-view reasoning tasks by integrating them with explicit robotic execution tasks. The accompanying analysis establishes a strong, positive correlation between spatial intelligence and robotic task execution in these multi-view scenarios, validating the design and highlighting the necessity of integrated capability.
2. MV-RoboBench consists of 1.7k manually curated QA items across eight carefully defined subtasks, ensuring comprehensive coverage and a balanced dataset. The construction process utilized a detailed human-in-the-loop quality review pipeline, including rigorous checks for question-answer alignment and randomization of answer options to prevent model bias.
3. A diverse array of existing state-of-the-art models was thoroughly evaluated, including both open-source and closed-source VLMs, as well as variants incorporating CoT-inspired enhancements. This broad evaluation provides a robust baseline against which models are shown to remain far below human performance.
4. The study includes rich analytical results that explore the relationship between perception and action. A particularly insightful finding is that strong performance on general-purpose single-view spatial benchmarks does not reliably transfer to the demanding, embodied tasks within MV-RoboBench, underscoring the unique challenge of multi-view reasoning in robotics.

### Weaknesses
1. A typo exists in Figure 3, where "AgiWolrd" should be corrected to "AgiWorld".
2. The paper lacks transparency regarding the human effort involved in its construction. It should expose the origin of the hired annotators, the total number of human-hours dedicated to selecting image pairs and building the 1.7K QA items, and the compensation structure for the annotators.
3. Since the benchmark is grounded in robotic manipulation (the domain of VLA models) and utilizes real robotic demonstration data, a VLA-centric evaluation would be highly beneficial, as VLAs are specifically designed for the physical execution and grounding that this benchmark demands, contrasting sharply with the training data distribution of many general-purpose VLMs.

### Questions
1. Given that Action Planning and Step Execution answers rely on normalized directional terms (e.g., leftward, forward), how were these linguistic action sequences generated and validated to be reliable and unbiased? Were these descriptions manually authored by annotators based on visual observation, or were they extracted directly from existing action annotations in the source datasets? What specific measures were implemented to ensure the reliability and eliminate ambiguity or bias in the normalized directional language, especially concerning the coordinate system definition and the consistency of the distractor options?
2. About Trajectory Selection questions, what is the origin of the ground-truth trajectories (e.g., raw robot waypoints from the AgiWorld/BridgeV2 data vs. human-drawn ideal paths)? What methodology was used to create the distractor options to ensure they are visually plausible but fail due to collision or poor geometric alignment, thus demanding multi-view reasoning?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This work introduces MV-RoboBench, a benchmark specifically designed to evaluate the multi-view spatial reasoning capabilities of vision-language models (VLMs) in robotic manipulation. The benchmark contains 1.7k manually curated QA items covering eight subtasks. Evaluation results show that state-of-the-art models still perform far below human levels. The study also explores chain-of-thought (CoT)-inspired enhancements, which produce mixed, model-dependent effects and reveal two main insights:
(1) spatial reasoning ability strongly correlates with robotic execution performance; and
(2) high accuracy on general single-view spatial benchmarks does not reliably transfer to multi-view robotic tasks.

### Strengths
1. The proposed benchmark addresses an important and practical problem in robotics.
2. The annotations for the benchmark are manually collected, which helps ensure their correctness and reliability.
3. This work also investigates the effects of CoT prompting and uncovers two key correlations, between spatial reasoning and robotic execution, and between single- and multi-view understanding, offering interesting insights to the research community.
4. The presentation is easy to follow.

### Weaknesses
1. The paper claims to be the first to integrate spatial and robotic reasoning with synchronized multi-view inputs in robotic manipulation scenarios. However, the previous ERQA benchmark also includes some multi-view spatial reasoning and manipulation questions in its test set. Although such samples are fewer and many ERQA items are single-image based, it nevertheless contains similar tasks, such as cross-view matching, as those in the proposed benchmark. Therefore, the authors should discuss ERQA in the paper.
In addition, another benchmark, MMSI-Bench, also focuses on multi-image spatial understanding. Although it is not primarily designed for manipulation scenarios, it contains samples from datasets like Agibots, which are relevant. The authors should be aware of and discuss this work as well, to provide readers with a more comprehensive understanding of the existing benchmark landscape in this domain.
2. Although the answers in the proposed benchmark are manually crafted, the questions are generated using template-based designs, which may limit the diversity and richness of the dataset.
3. While the benchmark emphasizes multi-view understanding, the paper does not provide a mechanism to ensure that questions cannot be answered by observing only a single image. For example, in Figure 1, the distance judgment and spatial consistency examples appear solvable from one view alone. I recommend conducting another round of quality review to verify each question’s validity and to filter out those that can be answered using only a single image.
4. Regarding the experimental results in Figure 6, the authors use the OmniSpatial benchmark. However, I do not believe OmniSpatial is an appropriate choice for this analysis, as it includes many samples with multiple images, not-real scenes, and non-robotic content. This may distort the analysis of the correlation. A more suitable comparison would be to use the single-view questions from the ERQA benchmark instead.
5. This work did not have an error analysis of the tested models.

I am open to raising the score if my concerns are addressed.

### Questions
1. The model performance on the 3D spatial consistency subset appears quite unusual. In particular, GPT-5-Chat achieves only 4.9%, whereas GPT-5-Mini reaches 72.55%. I am curious about the API identifiers of these two models. This large discrepancy also suggests that there might be an evaluation bug. It would be helpful if the authors could provide some example predictions from these models to illustrate and better understand the observed performance gap.

2. I also wonder whether the authors believe there are potential methods to automatically collect such QA pairs for training purposes. If so, I am curious how fine-tuned models would perform on this benchmark. Moreover, the limited chain-of-thought (CoT) effects observed in this work might be due to the fact that the models have not been exposed to such reasoning patterns during training. Fine-tuning with CoT-style data could potentially enhance their performance further.

### Soundness
3

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
3

### Summary
The paper built a benchmark for evaluating the spacial reasoning ability of VLMs under multi-view setting. The experiments analyzed the how spatial and robotic reasoning relate within multi-view manipulation, and whether spatial intelligence measured in single-view settings can transfer to embodied multi-view tasks. The results revealed that the VLMs have some common shortcoming in certain spacial reasoning scenario, provided a potential direction of VLM developments.

### Strengths
1. The paper has good structure and easy to follow.
2. The author conducted a large scale experiment on multiple modern VLMs, which makes the experimental results and conclusions convicible.
3. The spacial reasoning ability is critical for the development of VLMs. This paper provided a good example of how to evaluate the VLMs and thus might have broad influence to the community.

### Weaknesses
1. In abstract, should not use abbreviation when the the first time "CoT-inspired techniques" appears. Therefore "CoT-inspired techniques" -> "Chain of Thought (CoT)-inspired techniques."
2. Does the image orientation matters? For example, if the head camera view is upside down, can the system still get correct inference? This discussion is necessary to determine the VLMs' generalization on spacial reasoning.
3. The benchmark have multiple-choice questions across eight subtasks, each with exactly one correct answer. However, another important aspect is that whether the system can determine there is no correct choice.

### Questions
See weaknesses

### Soundness
4

### Presentation
4

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
Paper presents a new benchmark for assessing cross-view reasoning-related functionalities of AI models, with a focus on tabletop manipulation scenarios. The benchmark is constructed with a 3-step procedure: 1) collecting multi-view frames (mostly left arm, right arm, and center view) from canonical robotic datasets (AgiBot  World, etc); 2) generating QA pairs by crowdsourcing with human raters and QA templates were adopted; 3) curating the benchmark with human raters to obtain a total of 1.7k QA entries. The dataset covers a broad range of reasoning tasks, primarily categorized into spatial reasoning and robotic task reasoning. The authors have conducted a rich collection of experiments, including general benchmarking on foundation AI models, the effectiveness of CoT-enhanced reasoning, the correlation between the performances of spatial reasoning and robotic task reasoning, and whether performances on single-view reasoning benchmarks could transfer to the proposed multi-view benchmarks.

### Strengths
+The paper is clearly written and easy to follow. The topic (benchmarking of AI models, embodied AI, spatial AI) is relevant to a broad spectrum of ICLR community members.

+The proposed benchmark is constructed and curated in a rigorous fashion, where human raters, instead of automatic pipelines, perform most annotation tasks. Compared to related benchmarks, this alleviate the difficulty of data curation and potentially help with a higher dataset quality.

+I appreciate the authors for arranging such rich collection of tasks. Notably, showing the correlation between spatial reasoning and robotic task reasoning, as well as the transferabiliy from single view to multi-view reasoning, could indeed shed some light into future research.

### Weaknesses
My primary concerns lies with the validity of the "multi-view" setting in the current dataset format and relation to embodied AI tasks. 

-It seems that many of the designed tasks do not nessitate the need for multi-view input. For the tasks illustrated in Figure 1, most tasks seems doable with only one view, except for viewpoint identification and cross-view matching. If this is the case, I would like to see an ablation on using only one view (say the center view) as input and see if there is significant performance drop -- of course the two tasks mentioned above should be excluded. 

-Although there are tasks that indirectly evaluate the capability of robot manipulation (i.e., robot task reasoning), but it would be great to see if there is more direct connection, let's say the AI models control the robot directly. The authors claim this in the abstract "spatial intelligence and robotic task execution are correlated in multi-view robotic scenarios". However, the "robotic task" in the proposed dataset never touches controlling an actual robot but merely works on reasoning aspects of high-level planning. If the authors choose to go with the current setting, I suggest modifying their claim into "reasoning for high-level planning", not robot task executation in general. Otherwise, experiments on maybe fine-tuning these models into low-level control should be necessary.

### Questions
See "Weaknesses".

### Soundness
2

### Presentation
3

### Contribution
2
