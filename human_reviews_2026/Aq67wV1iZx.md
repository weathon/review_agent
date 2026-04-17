# Right Side Up? Disentangling Orientation Understanding in MLLMs with Fine-grained Multi-axis Perception Tasks

- Decision: Reject
- Scores: 8, 6, 4, 6

## Abstract
Object orientation understanding represents a fundamental challenge in visual perception that underpins critical real-world applications like robotic manipulation and augmented reality. However, current vision-language benchmarks fail to isolate and evaluate this core capability, often conflating it with positional relationships (such as above/below or proximity between objects) and general scene understanding. To address this, we introduce Discriminative Orientation Reasoning Intelligence (DORI), a comprehensive hierarchical benchmark that establishes object orientation perception as a primary evaluation target. DORI rigorously assesses four essential dimensions of object(s) orientation comprehension: frontal alignment, rotational transformations, relative directional relationships, and canonical orientation understanding. DORI provides valuable insights on how existing multi-modal systems process and understand object orientations through carefully curated tasks from 14 sources that spans $67$ object categories across synthetic and real-world scenarios. Our evaluation of $22$ state-of-the-art vision-language models using DORI reveals critical limitations: even the best models achieve only $54.2\%$ accuracy on coarse tasks and $45.0\%$ on granular orientation judgments, with performance deteriorating substantially for tasks requiring reference frame shifts or compound rotations. These findings demonstrate the urgent need for dedicated orientation representation mechanisms in future architectures, as models show a systematic inability to perform precise angular estimations, track orientation changes across multiple viewpoints, and understand compound rotations—suggesting fundamental limitations in their internal 3D spatial representations. As the first diagnostic framework specifically designed for advancing orientation awareness in multimodal systems, DORI offers immediate implications for improving robotic control, 3D scene reconstruction, and human-AI interaction in physical environments

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper introduces a hierarchical benchmark that evaluates MLLM’s ability to understand and reason about orientation. The paper evaluates multiple MLLMs on the benchmarking and shows poor performance.

### Strengths
* The motivation behind the curated dataset is well justified. The curated dataset is diverse and contains common objects that shall be viewed by MLLM during pre-training. 
* The paper has a clear presentation of the dataset and complete evaluation of the popular open-source and closed-source MLLMs.
* This dataset has a potential be improved to 3D. If so, this will be very beneficial for active learning and robotic pre-training, etc..

### Weaknesses
* I have a concern about whether the "counter-clockwise" and "clockwise" are consistently defined. In a 3D setting, when talking about rotation, we always need to specify the direction of the z-axis. But such information is not provided in the dataset.
* I also have a concern about whether "face toward" is well defined. This clearly requires the described object to have a "face" that is visually decidable. If the object is a human, it is simple. But for other objects like tables or sofa, this language may not apply.
* This is not a weakness. Since the tasks proposed in the paper mostly require 3D reasoning, it may make the dataset stronger if 3D point cloud or depth are also provided for those simulated images.

### Questions
No question.

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
3

### Summary
This paper presents the DORI benchmark, developed to specifically evaluate how well Multimodal Large Language Models (MLLMs) understand object orientation. DORI uses a cognitive science-informed approach, assessing orientation perception across multiple facets including how objects face the viewer, how they change with rotation, their orientation relative to other objects or viewpoints, and their typical 'right-side-up' state. The evaluation includes both basic categorical questions and more demanding fine-grained angular questions. It is applied to a substantial amount of real and synthetic images (over 13k images from 14 sources) with structured prompts. Experiments involving 18 MLLMs indicated difficulties in this dataset, particularly in making precise orientation judgments versus simpler classifications. Notable performance declines happen when tasks required understanding rotations or shifts in perspective. The findings suggest current models may lack robust internal mechanisms for representing and reasoning about object orientation.

### Strengths
1. The most noticeable contribution is the dataset collected. The benchmark's hierarchical structure, decomposes orientation related questions into four dimensions, which care frontal alignment, rotational transformations, relative orientation and canonical orientation, is quite inspiring. The inclusion of both coarse and fine-grained questions allows a more comprehensive assessment of model proficiency

2. The paper effectively identifies and addresses a limitation in existing MLLM, which is the lack of ability to assess object orientation understanding, separate from general spatial reasoning. This dataset helped to evaluate more detailed object orientation understanding ability in MLLM.

3. DORI is constructed from a quite diverse set of images (13,652 images from 14 sources), which include both real-world and synthetic data. The evaluation is conducted across 18 different MLLMs, providing a quite holistic benchmark evaluation.

### Weaknesses
1. The presentation of the paper can be improved. Limited examples are provided for the VQA questions involving canonical orientation. I have remaining concerns on these types of questions since canonical orientation or frontal alignment itself might remain inherently ambiguous for certain object types, like symmetric ones. This might introduce noise into the ground truth and evaluation, and I am interested in seeing how they are addressed more detailedly.

2. Another limitation is the absence of empirical validation showing that performance on DORI actually correlates with MLLM capabilities in real-world applications like robotic manipulation or autonomous navigation. The paper claims relevance but doesn't demonstrate a predictive link between benchmark scores and success on applied tasks requiring orientation understanding. 

3. The paper's writing can be improved. Some tables exceed text width. In main experiments tables, using vertical axis to separate different task is recommended. Overall, there are some minor presentation issues in the paper.

### Questions
Please refer to the weakness.

Will this dataset be released?

### Soundness
3

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
5

### Summary
This work introduces DORI, a benchmark designed to evaluate the orientation perception ability of current multimodal large language models (MLLMs). DORI comprises 13,652 images from 14 sources, forming a total of 33,656 samples. It assesses four key aspects of object orientation understanding: frontal alignment, rotational transformations, relative directional relationships, and canonical orientation comprehension. The results show that even the best-performing models achieve only 54.2% accuracy on coarse-level tasks and 33.0% on fine-grained orientation judgments, with performance degrading significantly on tasks involving reference frame shifts or compound rotations.

### Strengths
1. The proposed benchmark addresses an important problem.
2. The writing is generally clear and easy to follow.
3. The related work section is detailed and clearly explains the limitations of existing benchmarks.
4. The proposed benchmark is novel in terms of its practical usability.

### Weaknesses
1. Figure 1 is not very clear and does not effectively convey the definitions of the four task categories. In particular, the examples for rotational transformation and relative orientation appear very similar, making it difficult to distinguish between them.
2. Although the paper cites many works from related fields such as cognitive science to explain how humans understand rotation, the definitions of the four subproblems lack clear logic and structure. The relationships among them are not well articulated, leaving it unclear whether the proposed categorization is both complete and necessary.
3. For some objects, the front face is inherently ambiguous (e.g., a table). Although the paper mentions that specific prompt designs are used to define the front face for tested models, such strategies cannot fully resolve these ambiguities. This raises concerns about the correctness and answerability of certain questions.
4. Based on the above, I suspect that some samples may be ambiguous. However, the paper does not describe any quality control process to ensure dataset accuracy. For a benchmark, it is generally expected that every sample be manually verified to guarantee correctness.
5. While the paper evaluates several state-of-the-art models, it omits important models such as the InternVL series, and for the Qwen family, only the 3B variant is tested without including larger models.
6. Lines 418–419: The observed difference may stem from variations in training data, so this conclusion should not be drawn too hastily.
7. Table 5: Please correct the label from “GPT-4 O” to “GPT-4o.”
8. The paper claims that the proposed systematic approach isolates orientation understanding from scene perception skills and minimizes confounding factors such as object recognition difficulty, scene clutter, linguistic ambiguity, and contextual distractions that affect existing benchmarks. However, no experiments or examples are provided to substantiate these claims.

### Questions
1. In Section 3.1, the definition of the viewing plane is not clearly explained.

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
The paper proposes DORI, a diagnostic benchmark for orientation understanding in MLLMs across four dimensions (frontal alignment, relative orientation, rotational transformation, canonical orientation). 
It uses standardized MCQ prompts (with a Cannot be determined option) and reports that models handle coarse judgments better than granular angles; token-based fusion appears stronger than linear projection. 
LoRA fine-tuning on DORI reportedly transfers to external spatial benchmarks.

### Strengths
The paper cleanly isolates orientation understanding into complementary abilities (frontal, relative, rotational, canonical) and tests them with a coarse–granular design that probes both category and precise angle reasoning. 
The benchmark is well engineered, and broad model coverage exposes consistent weaknesses. 
Findings are actionable, making DORI a practical diagnostic tool for geometry-sensitive applications.

### Weaknesses
- Since prompting is part of the measurement apparatus, please ablate the components to quantify their contribution and ensure models aren’t over-relying on the scaffold rather than vision.

- Ground-truth fidelity and metric design: While synthetic sources yield precise angles, human-annotated natural images can have ambiguous frontality (e.g., symmetric furniture) and unknown camera intrinsics, which may distort a fixed discrete angle taxonomy.

- Architectural claims need stronger controls: The observation that token-based integration > linear projection is compelling but potentially confounded by pretraining data or instruction tuning.

- Human study scale and reporting: Human evaluation covers 30 examples per type with seven experts. This is useful but small.

### Questions
- Have you tried free-form numeric responses (regression-style) and then quantized at evaluation time? Do model rankings persist? Please share results with permuted answer choices and removed examples section to quantify prompt-component effects.

- For the claim that token-based fusion > linear projection, can you provide experiments with the same visual backbone and identical instruction-tuning, changing only the fusion scheme? Any results with feature token counts swept to test capacity vs mechanism? (If it's not possible, that's understandable)

- Would you consider scaling the human study to 300–500 items with crowdworkers + expert adjudication, and report results to better anchor the human–model gap?

### Soundness
2

### Presentation
2

### Contribution
3
