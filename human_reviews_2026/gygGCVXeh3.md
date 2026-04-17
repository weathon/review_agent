# Go Beyond Earth: Understanding Human Actions and Scenes in Microgravity Environments

- Decision: Accept (Poster)
- Scores: 6, 6, 4

## Abstract
Despite substantial progress in video understanding, most existing datasets are limited to Earth’s gravitational conditions. However, microgravity alters human motion, interactions, and visual semantics, revealing a critical gap for real-world vision systems. This presents a challenge for domain-robust video understanding in safety-critical space applications.
To address this, we introduce MicroG-4M, the first benchmark for spatio-temporal and semantic understanding of human activities in microgravity. Constructed from real-world space missions and cinematic simulations, the dataset includes $4{,}759$ clips with $13{,}261$ action annotations covering $50$ actions, $1{,}238$ context-rich captions, and over $7{,}000$ question–answer pairs on astronaut activities and scene understanding. MicroG-4M aims to support three core tasks: fine-grained multi-label action recognition, temporal video captioning, and visual question answering, thereby enabling a comprehensive evaluation of both spatial localization and semantic reasoning in microgravity contexts. We establish baselines using state-of-the-art models. All data, annotations, and code are available at https://github.com/lei-qi-233/MicroG-4M.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a new benchmark on video understanding. 
The benchmark address the novel scenario of astronautic environment in space.
The benchmark contain 5k clips of short videos (3 seconds) with rich annotations. 
This paper then benchmark state-of-the-art models using their dataset, results suggest the evaluated models do not work as good as they are in normal earth-based videos.

### Strengths
This paper addresses a novel scenario for video understanding. 

The paper is well motivated. 

The paper clearly documented the data collection pipeline.

The evaluation setting of baseline methods are also clearly written.

### Weaknesses
The main issue with this paper is that its distinction from the Earth-Video benchmark is not clearly presented. 
The reported performance indicates that the benchmark is difficult, but it is unclear whether any unique aspects of the astronautic videos contribute to this difficulty. 
Currently, the presentation of experiment section 5 feels like “just another video benchmark”, as it does not provide much surprising findings. 


Regarding the baselines of human action recognition task, I’m curious whether the authors have tried VLMs like Gemini 2.5 Pro? I feel advanced VLMs like Gemini 2.5 Pro may be able to answer this. It would be nice to have them.

### Questions
This paper is clearly written, thus I don’t have questions regarding clarity. 

I expect more qualitative results like Figure 1, 3 and 4. 

The videos.zip in the supplementary does not work for me.

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
The paper introduces MicroG-4M, the first benchmark dataset designed for spatio-temporal and semantic understanding of human activities in microgravity environments. It addresses a critical gap in current vision research, as most existing datasets for video captioning and action recognition are recorded on Earth under normal gravity, whereas microgravity significantly alters human motion, interactions, and visual semantics. Constructed from real-world space mission footage and cinematic simulations, MicroG-4M contains 4,759 3 second video clips at 30fps covering 50 actions, 1,238 captions, and over 7,000 question–answer pairs centered on astronaut activities and scene understanding. The dataset supports three core tasks: fine-grained multi-label action recognition, temporal video captioning, and visual question answering.

### Strengths
1.  The contribution of MicroG-4M is highly interesting and original, as it introduces the first large-scale benchmark for understanding human actions, captions, and question answering in microgravity environments. The dataset offers significant potential for advancing research in vision-language modeling, domain adaptation, and embodied AI under extreme physical conditions.

2. The strength of this paper lies in the detailed presentation of dataset statistics and discussion with limitation. The authors provide a clear breakdown of the distribution across broad action types, the number of persons per clip, and fine-grained action frequencies, highlighting important patterns such as the dominance of single-person clips and the long-tail distribution of actions. Additionally, the per-class AP results convincingly demonstrate the value of MicroG-4M for microgravity-specific action recognition, while the high-density, VQA annotations further support rich semantic understanding.

### Weaknesses
1. In the collection methodology section, the author’s writing style is precise, formal, and research-oriented, making it suited for submission in conference. However, it leans toward being dense and information-heavy, which could benefit from slight simplification or the inclusion of visual aids (such as tables or flow diagrams) to enhance clarity and readability.

2. The author does not explicitly define the categories “Object Manipulation,” “Person Interaction,” and “Person Movement.” Instead, it appears that the reader is expected to infer their meaning from common sense or from the constituent fine-grained actions. It would improve clarity if the author briefly described each category with examples. For instance, "Object Manipulation could be defined as actions where a person interacts with objects in the environment, including picking up, carrying, holding, pushing, pulling, operating equipment, or using tools. Examples include carry/hold object, push object, operate spaceship, or using a computer. In microgravity, these actions are particularly important because the dynamics of motion and object handling differ from those on Earth."

3. While the authors state that all captions and QA annotations were created manually by annotators with “domain guidance,” they do not specify what type of guidance was provided or give concrete examples. It would be helpful to clarify whether this guidance included official space agency documents, astronaut manuals, mission reports, or expert review, and to briefly describe what information from these sources was used (e.g., spacecraft layout, standard operating procedures, or typical astronaut activities). Providing such details would improve transparency and help readers better assess the quality and reliability of the annotations.

### Questions
1. For video captioning and VQA, the authors mention using large language models (LLMs) but do not describe the prompts or prompting strategy employed. Providing information about the prompts, including their format, instructions, or examples, would improve reproducibility and allow readers to better understand how LLMs contributed to annotation quality.

2. It is unclear how video input is processed to generate captions. Do the authors first extract keyframes, or do they read video frames sequentially? If keyframes are used, the authors should specify the algorithm or criteria for keyframe selection. Additionally, the caption generation process using Visual-Language Models (VLMs) or Multimodal Large Language Model (MLLM) is not described in detail, clarifying whether captions are generated per frame, per keyframe, or for the entire clip would improve reproducibility.

3. Employing VLMs may introduce computational overhead, especially for high-resolution frames or sequential multi-frame processing. For practical deployment, it would be helpful if the authors reported the hardware configuration (GPU type, CPU, RAM), memory usage, and average execution time per frame or video clip. This information would provide readers with a clearer understanding of the method’s efficiency and scalability.

4. The authors state that each three-second clip contains six QA pairs; however, it is unclear how these questions were generated or selected. It would be helpful to clarify whether the questions were derived directly from the video content or generated based on the captions. The paper should also describe the question selection process, including examples of question types and how balance was maintained across reasoning categories such as temporal reasoning. The author could explain such as Six QA pairs covering different aspects: (1) Identity (“Who is the astronaut?”), (2) Action detail (“What does she do with her hands?”), (3) Body motion (“How do her head and shoulders move?”), (4) Spatial context (“What items can be seen behind?”), (5) Static background recognition (“What remains stationary?”), (6) Unanswerable / implicit reasoning. However, the authors do not indicate whether such systematic consideration guided their question design. A clearer explanation of the QA generation and selection methodology would greatly enhance reproducibility and transparency.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work introduces MicroG-4M, the first large-scale dataset specifically curated for human action recognition and vision-language understanding in microgravity environments. MicroG-4M includes 4,759 clips covering 50 actions, 1,238 context-rich captions, and over 7,000 question–answer pairs on astronaut activities and scene understanding. MicroG-4M aims to support three core tasks: fine-grained multi-label action recognition, temporal video captioning, and visual question answering, thereby enabling a comprehensive evaluation of both spatial localization and semantic reasoning in microgravity contexts.

### Strengths
1.The paper is well-written and easy to understand.

2.This work introduces MicroG-4M dataset, which is a valuable contribution to fill the gap of video understanding benchmarks under microgravity scenarios.

### Weaknesses
1.One concern for this work is the technical contribution. From my point of view, the major contribution of this work comes from its data collection and organization, while the methodological contributions are missing, i.e., there is no specifically designed baselines for video understanding under microgravity scenarios nor novel insights from this work. This decreases the overall contributions of this work.

### Questions
Please refer to the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2
