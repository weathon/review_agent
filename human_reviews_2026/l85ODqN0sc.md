# V2P-Bench: Evaluating Video-Language Understanding with Visual Prompts for Better Human-Model Interaction

- Decision: Accept (Poster)
- Scores: 2, 6, 6, 4

## Abstract
Large Vision-Language Models (LVLMs) have made significant strides in the field of video understanding in recent times. Nevertheless, existing video benchmarks predominantly rely on text prompts for evaluation, which often require complex referential language. To address this limitation, we propose V2P-Bench, a robust and comprehensive benchmark for evaluating the ability of LVLMs to understand Video Visual Prompts in human–model interaction scenarios. V2P-Bench consists of 980 videos and 1172 well-structured high-quality QA pairs, each paired with manually annotated visual prompt frames. The benchmark spans three main tasks and twelve categories, thereby enabling fine-grained, instance-level evaluation. Through an in-depth analysis of current LVLMs, we identify several key findings: 1) Visual prompts are both more model-friendly and user-friendly in interactive scenarios than text prompts, leading to significantly improved model performance and enhanced user experience. 2) Models are reasonably capable of zero-shot understanding of visual prompts, but struggle with spatiotemporal understanding. Even o1 achieves only 71.8%, far below the human expert score of 88.3%, while most open-source models perform below 60%. 3) LVLMs exhibit pervasive hack phenomena in video question answering, which intensify with longer videos and lower frame sampling density, artificially inflating performance scores. We anticipate that V2P-Bench will not only shed light on these challenges but also serve as a foundational tool for advancing human–model interaction. The code and datasets are available at https://github.com/gaotiexinqu/v2p-bench.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper introduces a new video benchmark that tests models’ abilities at visual question answering while allowing for visual prompting in addition to text.  With visual prompting, users are allowed to annotate the video and ask questions that refer to the annotation (eg., what is the person with the box around them doing?).  The benchmark is built on existing video collections and includes a range of different types of questions, ranging from low level questions about, for example, object attributes, to higher level questions, about plot or causal relationships.  It seems that all questions are designed to require identification of specific people or objects that might be difficult to otherwise describe.  A large number of open and closed models are identified on the benchmark, which still demonstrate performance that is below human level.

### Strengths
The evaluations seem thorough.  Visual prompting does not seem to have been studied adequately in video.  The work seems very comprehensive.

### Weaknesses
First, it is not clear how important this problem is.  The questions in the dataset seem designed to benefit from visual prompting.  But how often do users really want to ask questions about people or objects that are difficult to describe?  This might happen often, but there is no evidence given that this is the case.

Second, it is not clear that the results that visual prompting is more effective and efficient than text prompting are truly valid, since the questions developed seem designed to favor visual prompting.  This would be more convincing if these questions arose organically from users attempting to solve real tasks.  

Third, the fact that the benchmark contains high and low level questions does not seem very significant.  Visual prompting is used entirely to make it easier for a user to specify a person or object.  The only real question is how much more effective this is than specifying people or objects with text.  If the model can interpret the visual prompting, failures to answer higher level questions have nothing to do with the prompt, but more to do with higher level reasoning. 

Four, one of the key questions in video understanding is the extent to which models can integrate information from different parts of the video.  Video prompting seems to be primarily used here with questions that are temporally localized, and could perhaps be answered using a single frame of the video.

### Questions
Please provide more details on how the questions were chosen.  How do you ensure that you are not selecting questions that will bias performance in favor of visual prompting?

Experiments in Table 3 are not clearly explained.  How are the text and visual prompts generated?  In Table 3b, what interface is provided for visual prompts?  Is training required?  The supplementary material addresses some of this, but I still found it unclear.  This should be clearly explained in the body of the paper.

### Soundness
2

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
This paper proposes V2P-Bench, a new benchmark addressing the inefficiency of text-based evaluation in video-language understanding. By using visual prompts directly on video frames, it assesses model performance across perception, temporal, and reasoning tasks. Results show that visual prompts improve model accuracy while exposing weaknesses in long video and spatiotemporal reasoning.

### Strengths
This paper introduces the novel idea of using visual prompts for video-language evaluation, providing a more intuitive and human-aligned alternative to text-based prompts. The dataset and tasks are well designed, with careful annotation and broad experimental coverage across major LVLMs. The presentation is clear and well-structured, supported by effective figures and analyses.

### Weaknesses
The main weaknesses of this paper lie in its theoretical and methodological depth. While the idea of visual prompts is novel, the work lacks a stronger theoretical or cognitive grounding to explain why this approach better reflects human interaction. The evaluation remains limited to offline video QA with a relatively small dataset and does not include audio or multimodal signals, which limits its realism and scalability. In addition, the analysis is mostly descriptive, without deeper investigation into why models fail in spatiotemporal reasoning or specific error types.

### Questions
I have several critical questions that the authors should address to clarify the scientific validity of the work.

First, the central claim, that visual prompts are a more human-aligned form of interaction, remains unsubstantiated. Could the authors provide theoretical grounding or empirical evidence (e.g., from cognitive studies or user experiments) rather than relying on intuitive justification?

Second, the benchmark design is still based on offline video QA, which seems inconsistent with the paper’s framing around “human–model interaction.” How do the authors ensure that this static setup meaningfully reflects real interactive understanding?

Third, the error analysis is largely descriptive and lacks mechanism-level insight. Why do models consistently fail on spatiotemporal reasoning tasks, and how can the authors be sure that the evaluation protocol effectively eliminates “hack” behaviors rather than concealing them? 

Addressing these questions is essential to assessing the scientific rigor and actual contribution of the work.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes V2P-Bench, a robust and comprehensive benchmark for evaluating the ability of LVLMs to understand video visual prompts in human–model interaction scenarios. The authors argue that existing video benchmarks primarily rely on text prompts, which often involve complex referential language and, in turn, reduce the accuracy and efficiency of human–model interaction. To address this, they introduce more user-friendly visual prompts. The authors conduct a comprehensive analysis and highlight several key findings, such as performance on spatiotemporal understanding and the prevalence of the hack phenomenon.

### Strengths
1. The paper is well-motivated and studies a timely problem of visual prompt-based evaluation instead of unnecessarily overcomplicated textual prompts. I particularly liked that the authors clearly established this with a human study as well.
2. The benchmark is carefully curated with multiple automated and human checks.
3. The authors conduct comprehensive experiments to study different models, and go beyond standard evaluation to study interesting hack phenomenon and other ablations.

### Weaknesses
While I don't foresee any major weaknesses, I have some questions about experiment design and further experiments that I would like to see before I raise my score further.

### Questions
1. The hack phenomenon is concerning, especially as it increases rates as the video length grows. I am curious if the authors convert their benchmark to an open-ended generation one instead of MCQ-based, would that help in reducing some of this? 

2. I am curious how does the structuring of the visual query itself impact the performance? For instance, what if you have scribbled unstructured bounding boxes instead of perfect rectangles or other more natural markers that are more user-friendly, how does that impact overall performance? 

3. From many of the query examples provided in the paper, it appears that most could be resolved using only a few frames, with the main bottleneck being the grounding of the visual query frame. Could the authors elaborate on this point, perhaps by quantifying it using temporal certificates [1] or frame rate ablations? Specifically, do the models on their benchmark perform better as more visual information is provided, or can the benchmark generally be answered using just a few frames?

[1] EgoSchema: A Diagnostic Benchmark for Very Long-form Video Language Understanding

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces V2P-Bench, a benchmark designed to evaluate large vision-language models (LVLMs) using visual prompts instead of text prompts for video understanding, enabling fine-grained assessment of spatial, temporal, and reasoning abilities. Results show that visual prompts improve both model accuracy and user experience, but current LVLMs still lag behind humans, particularly in spatiotemporal comprehension and robustness.

### Strengths
- Introduces a visual prompt–based benchmark that improves realism in human–model interaction evaluation.
- Provides a comprehensive dataset with multi-level tasks and rigorous annotation ensuring high quality and diversity.
- Offers clear empirical insights into LVLM weaknesses, including hack phenomena and spatiotemporal reasoning gaps.

### Weaknesses
1. The benchmark currently focuses on synthetic QA tasks rather than natural, conversational interactions, which limits its generalization to real-world human–model dialogues. How would you integrate free-form conversational or multimodal dialogue settings to better simulate authentic human–AI interaction? If not, it is really not geberalisable. 

2. The dataset construction process is highly manual and time-consuming, making it difficult to scale to larger domains or real-time applications. How can we incorproate scalability?

3. Several analyses, such as the user experience evaluation and hack behavior detection, rely on qualitative interpretation rather than standardized quantitative frameworks. Cou.d the authors design more rigorous, quantitative evaluation protocols and reproducible metrics to strengthen the reliability and comparability of findings?

4. The benchmark primarily evaluates static performance outcomes but does not assess model adaptability or learning progression over time.

### Questions
See weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
