# Image Quality Assessment for Embodied AI

- Decision: Accept (Poster)
- Scores: 8, 8, 6, 6

## Abstract
Embodied AI has developed rapidly in recent years, but it is still mainly deployed in laboratories, with various distortions in the Real-world limiting its application. Traditionally, Image Quality Assessment (IQA) methods are applied to predict human preferences for distorted images; however, there is no IQA method to assess the usability of an image in embodied tasks, namely, the perceptual quality for robots. To provide accurate and reliable quality indicators for future embodied scenarios, we first propose the topic: IQA for Embodied AI. Specifically, we (1) based on the Mertonian system and meta-cognitive theory, constructed a perception-cognition-decision-execution pipeline and defined a comprehensive subjective score collection process; (2) established the Embodied-IQA database, containing over 30k reference/distorted image pairs, with more than 5m fine-grained annotations provided by Vision Language Models/Vision Language Action-models/Real-world robots; (3) trained and validated the performance of mainstream IQA methods on Embodied-IQA, demonstrating the need to develop more accurate quality indicators for Embodied AI. We sincerely hope that through evaluation, we can promote the application of Embodied AI under complex distortions in the Real-world.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The paper proposes that image quality assessment methods focus on the ability to predict human ratings for distorted images, which lacks meaning when applied to perceptual quality for robots. The authors propose the topic IQA for Embodied AI, a topic focusing on a robots ability to perform downstream tasks on distorted images. The Embodied-IQA database, a set of reference-distorted image pairs, with corresponding decisions or task executions, where labels are provided by a set of vision language models, with some evaluation against existing datasets and traditional IQMs.

### Strengths
- The topic is very relevant for ICLR and the contributions are significant.
- The dataset is impressive, in the amount of distortions applied to images, the cognition/decision/execution labels, and the breadth of situations covered.
- The dataset analysis is thorough, providing correlations between different aspects of the datasets and evaluations using traditional full-reference and non-reference IQAs.
- I am not aware of any datasets covering the end-to-end pipeline of using distorted images in the field of robots

### Weaknesses
- A discussion on how likely the distortions are to show up in the robotic systems is missing. For example, are changes in chroma often observed in the sensors used in robotic systems? I imagine this is similar to the ISP pipeline where out of focus (blur), sensor noise (noise) and compression are more likely than others.
- It would be interesting to note where certain distortions effect the pipeline - for example spatial distortions are more likely for the robot to miss and object, chroma distortions are more likely to misinterpret which object to act upon ect.

### Questions
- How similar is the topic of Embodied AI to image distortions effect downstream tasks in topics such as image classification, object recognition, ect? I imagine some of the distortions are quite correlated with issues in these topics as discussed in papers such as [1] 
- Do authors expect the above to be captured in their dataset?

[1] Zhou, Yiren, Sibo Song, and Ngai-Man Cheung. "On classification of distorted images with deep convolutional neural networks." 2017 IEEE International conference on acoustics, speech and signal processing (ICASSP). IEEE, 2017.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes a completely new research task: Image Quality Assessment for Embodied AI . The authors' core argument is that traditional human-centric or general machine-centric IQA metrics are not applicable to Embodied AI, because RVS involves a more complex Perception-Cognition-Decision-Execution pipeline. To address this issue, the paper makes three main contributions:

1. Based on the "Mertonian system" theory, it constructs the aforementioned four-step pipeline, theoretically demonstrating the fundamental difference between RVS and MVS/HVS.

2. It constructs a large-scale benchmark dataset named Embodied-IQA. It contains over 30k reference/distorted image pairs and 5 million fine-grained annotations. These annotations are collected separately from VLMs, VLAs, and real-world robots (representing Execution). 

3. The paper benchmarks 15 mainstream IQA methods on the Embodied-IQA dataset. The results show that existing methods perform poorly on this task (e.g., SRCC is much lower than on HVS tasks), which strongly demonstrates the necessity of developing new Embodied-IQA metrics.

### Strengths
1. The paper identifies and clearly defines a completely new, critical, and timely research problem: assessing image quality for Embodied AI. Its theoretical framework based on the "Mertonian system" to differentiate RVS, MVS, and HVS is highly novel and persuasive, laying a solid theoretical foundation for this new field.

2. The paper's main contribution—the Embodied-IQA dataset—is an extremely valuable resource. Its scale and granularity are unprecedented in the IQA field. This dataset will likely drive much research in the coming years.

3. The paper not only provides the dataset but also conducts a comprehensive benchmark of 15 existing IQA methods. The analysis convincingly demonstrates the failure of existing methods on this new task, strongly validating the paper's motivation and the necessity of its contribution.

### Weaknesses
1. The paper defines the VLA "Decision" score as a simple average of errors in three dimensions: Position, Rotation, and State. This metric seems overly simplistic. In real robotics tasks, the importance of these three dimensions can be highly imbalanced (e.g., a minor rotation error could cause catastrophic failure, while a larger position error might still be acceptable).
2. As a benchmark paper, its primary duty is to define the problem and provide data, which it does exceptionally well. However, after proving that 15 existing methods fail, the paper does not attempt to train even a simple new baseline model (e.g., a simple CNN trained from scratch on Embodied-IQA) to set an initial performance bar for future researchers to challenge.
3. While 1,500 real-world robot experiments (for execution validation) is already impressive and costly, it is still a small scale compared to the 5 million VLM/VLA annotations. Although the paper acknowledges this, it leaves in question whether VLA annotations can serve as a high-fidelity proxy for real execution.

### Questions
1. You define the "Decision" score as a simple average of position, rotation, and state errors. How did you account for the unequal importance of these three dimensions in different tasks? For example, in a "turn faucet" task, shouldn't rotation error be weighted much more heavily than position error?
2. You found that the correlation between different VLA models is extremely low, indicating they have vastly different "preferences" for distortion. Your final "Decision" score appears to be an average of these 15 VLAs. Does this imply that an IQA metric performing well on the "average VLA" might perform poorly for a specific VLA?
3. Regarding the correlation between VLA and Execution: You found the SRCC between VLA (Decision) and real-world (Execution) to be ~0.67. In your opinion, is this correlation high enough for the community to confidently use VLA annotations as a proxy for real execution in the future? Or does this gap suggest that VLA annotations themselves still have limitations?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes IQA for Embodied AI, arguing that existing IQA methods of HVS and MVS are insufficient to judge image usability for robots, because robot performance depends not only on perception/cognition (VLM) but also on decision-making/planning (VLA) and real execution.To this end, the paper proposes: 1. A robot intelligence perception-cognition-decision-execution workflow based on Merton's systems perspective; 2. Embodied-IQA, a novel and diverse dataset. Furthermore, experiments demonstrate that embodied artificial intelligence requires more sophisticated IQA metrics.

### Strengths
1. The paper presents a clear motivation and significant innovation. It defines the image quality assessment (IQA) problem in embodied intelligence as “image usability for robots,” transcending traditional frameworks based on human or machine vision systems (HVS/MVS). It innovatively models the robot's “decision-making” and “execution” phases explicitly.
2. The research exhibits high quality. First, its constructed dataset is exemplary in scale (36k+ images, 5m+ labels), breadth (30 distortion types, multiple viewpoints and scenarios), and depth (incorporating 1,500 valuable real-world robotic execution trials). Second, the experimental design is rigorous, employing a comprehensive multi-dimensional scoring system and robust benchmarking protocols. This includes 10 data-split replicates and multiple correlation metrics, ensuring reliable results.
3. The study maintains rigorous structure, with appendices candidly discussing its limitations.
4. This work holds significant value. It not only provides a novel benchmark dataset but also successfully establishes critical empirical links between image quality and robots' actual decision-making and execution capabilities.

### Weaknesses
This study exhibits several critical weaknesses.
1. Methods that evaluate differences based on metrics may inherit and amplify inherent biases and errors within the model and its assessment indicators. Specifically, using metrics like BLEU/ROUGE to measure cognitive comprehension is highly sensitive to phrasing and redundancy, potentially failing to accurately reflect task equivalence. Adopting structured patterns (e.g., action-parameter tuples) or task success classifiers may be more robust alternatives.
2. The scoring design for the decision phase is also inadequate. The study simply averages standardized position, rotation, and gripper state metrics, failing to account for the weighting of critical failure modes across different tasks—for example, a minor rotation error may cause collision in some tasks, while gripper state importance varies by task. The absence of task-specific weighting may undermine label authenticity.
3. The real-world robot execution experiments are limited in scale and diversity. The 1,500 trials are confined to the simplest tasks per image, failing to comprehensively cover all distortion types, intensities, objects, and viewpoints. Thus, the generalizability of these findings to more complex multi-step tasks remains unclear.
4. The low correlation (SRCC) among different VLA models may stem not only from “genuine preference differences” but also from their heterogeneous action spaces or variations in pre- and post-processing workflows.
5. The study leaves a notable gap: while successfully demonstrating the inadequacy of existing IQA baseline models for new tasks, it fails to propose an “embodied IQA model” or architecture specifically designed for robotic usability (e.g., a model capable of predicting multidimensional scores based on task/text conditions).
6. Regarding clarity, the paper exhibits a few ambiguous sentences. The figures are information-dense, and critical conclusions are occasionally buried in supplementary materials, hindering the communication of core findings.

### Questions
1. How often do BLEU/ROUGE/CIDEr disagree with human judgments of task equivalence for VLM outputs? Have you validated a small subset with human raters using task-specific correctness criteria?

2. How were the 15 VLAs harmonized regarding action parameterization (units, reference frames, delta vs absolute commands)? Could inconsistencies reduce inter-model SRCC?

3. Are there results from more powerful VLA models (exceeding 8B parameters) or closed-source baselines to understand the impact of parameter scale on distortion robustness?

4. How are the difficulty levels for the five tasks defined? Are they objective?

### Soundness
3

### Presentation
2

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
1

### Summary
In this paper, the author proposes Image Quality Assessment for Embodied AI, where the quality of perception is not judged by human preferences, but a robot's ability to perform the task after perceiving distorted images. It introduces a database with 36900 reference distorted images.

### Strengths
- The idea of IQA for machines vs humans is established in prior works. This paper reframes the problem for robotics, the stage of how degradation of images affects robot task execution, not just visual recognition.
- The paper was interesting to read, with extensive experiments and detailed analysis.
- The Embodied-IQA database is very large, containing 36,900 image pairs and over 5 million fine-grained annotations, having good scale. It is also annotated along three unique axes, reflecting different visual systems and their real-world performance.

### Weaknesses
- The pipeline assumes vision is the dominant modality, neglecting that in true Embodied AI, perception often must fuse audio, tactile, and temperature cues. Is it possible that temperature might play a role on how bright the image might be?
- The writing is often verbose and repeats technical claims across sections, particularly regarding pipeline design and dataset composition.

### Questions
See above.

### Soundness
3

### Presentation
3

### Contribution
3
