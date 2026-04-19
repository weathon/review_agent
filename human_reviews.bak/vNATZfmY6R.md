# KiVA: Kid-inspired Visual Analogies for Testing Large Multimodal Models

- Decision: Accept (Poster)
- Scores: 6, 8, 8, 6

## Abstract
This paper investigates visual analogical reasoning in large multimodal models (LMMs) compared to human adults and children. A “visual analogy” is an abstract rule inferred from one image and applied to another. While benchmarks exist for testing visual reasoning in LMMs, they require advanced skills and omit basic visual analogies that even young children can make. Inspired by developmental psychology, we propose a new benchmark of 4,300 visual transformations of everyday objects to test LMMs on visual analogical reasoning and compare them to children (ages three to five) and to adults. We structure the evaluation into three stages: identifying what changed (e.g., color, number, etc.), how it changed (e.g., added one object), and applying the rule to new scenarios. Our findings show that while GPT-o1, GPT-4V, LLaVA-1.5, and MANTIS identify the “what” effectively, they struggle with quantifying the “how” and extrapolating this rule to new objects. In contrast, children and adults exhibit much stronger analogical reasoning at all three stages. Additionally, the strongest tested model, GPT-o1, performs better in tasks involving simple surface-level visual attributes like color and size, correlating with quicker human adult response times. Conversely, more complex tasks such as number, rotation, and reflection, which necessitate extensive cognitive processing and understanding of extrinsic spatial properties in the physical world, present more significant challenges. Altogether, these findings highlight the limitations of training models on data that primarily consists of 2D images and text.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work explores visual analogical reasoning in large multimodal models (LMMs) and compares their performance to that of human adults and children. Inspired by developmental psychology, the authors developed a benchmark, KiVA, with 1,400 visual transformations of everyday objects to test and assess models' capabilities in visual analogical reasoning.

### Strengths
1.KiVA introduces realistic, grounded visual tasks inspired by developmental psychology, aimed at evaluating analogical reasoning in ways similar to early human cognitive development.
2.The authors broke down their evaluation and proposed a 3-stage evaluation procedure to examine the different steps involved in analogical reasoning to determine which steps a model can perform and where it may fail.
3.Results from KiVA demonstrate that state-of-the-art large multimodal models, i.e., GPT-4V OpenAI (2023a), LLaVA-1.5 (Liu et al., 2024) and MANTIS (Jiang et al., 2024), cannot solve visual analogies like humans can. The authors discovered that While LMMs can detect some object transformations, they cannot make extrapolations about those transformations to new objects.

### Weaknesses
1.The KiVA test only include changes like orientation, changes in numbers, changes in size of the objects, and reflection while neglecting other basic transforms common in daily life like stretching(which can also be solved by young children).
2.LMMs today still rely heavily on the language model backbone. In other words their ability to assess the images heavily rely on what information the image encoder can provide to the backbone. Some encoders tend to lose information like size or rotation. Blaming the poor performance of the inference only on the analogy reasoning ability of the models might not be fair.

### Questions
1.I still have questions on why in the 3-stage evaluation procedure, even if the first question is answered incorrectly, the third question can be answered correctly. The authors did not provide an explanation on how this could occur, and what it could mean.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors describe KiVA, a benchmark for evaluating visual analogical reasoning in LMMs, using image pairs of common household objects before and after several different transformations. They test several LMM models as well as children and adults on three different tasks: what changed, how it changed, and applying the rule to a new object. Five transformations are used: color changes, size changes, rotation, reflection, and number changes. They find that each progressive task is more difficult and less consistent for LMMs, and strategies like prompt engineering do not help.

### Strengths
The writing is clear and well-structured. Figures also clearly demonstrate the test tasks and their results. The authors introduce a novel and well-motivated benchmark for studying LMM capabilities. The test is grounded by using real-world objects, and draws inspiration from developmental psychology. The three stages introduced by the authors help clarify where LMMs have shortcomings. The experimental design is rigorous, and validated with human studies of both children and adults. The analysis is detailed and includes both error and consistency patterns. The results reveal important limitations in LMM visual reasoning capabilities that have been previously underexplored.

### Weaknesses
The discussion could be expanded with discussion of why models tend to fail at certain transformations, outside of investigating their consistency. While the paper mentions objects were "handpicked by developmental psychology experts", it doesn't detail the selection criteria or validation process. There's no reported validation that the transformations are equally discriminable across categories. For instance, an example image shows a die face with five dots - almost completely symmetric under 90 degree rotations, one of the allowed transformations. Would the dataset include that difficult a question, or nearly-as-difficult ones with minor visual changes across transformations? Lacking space in the 10-page limit, several experiments are described extraordinarily briefly along with their results, with real methods left to the appendix. One such description - “verbal questions facilitate” -  lacks useful data, only reporting p values without actual quantities of the results.

### Questions
Were there any noticeable patterns in tasks that LMMs failed but humans succeeded, outside of category (e.g. any specific objects?) when the models are consistent? Likewise for humans?

How were outliers handled in your human studies, if at all?

The child participants span a wide age range that covers important developmental milestones relevant to this task. Despite the small number of subjects, do you see correlations with age, especially for your specific test categories?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
Overall this is a strong paper. The contribution of the benchmark is interesting and well-designed, and both adult and child data is provided as baseline for comparison. The paper is well written.

### Strengths
- Breaking down visual analogies into these reasoning steps helps to highlight exactly where humans and models fail
- Presenting both adult and child data on the benchmark questions is valuable. The human studies appear well conducted.
- Various additional common steps are evaluated to improve model performance, which interestingly do not seem to change model performance greatly.
- Examination of model response consistency helps to unpack where model decisions go wrong.

### Weaknesses
1. The rotation task does not appear to assess 3D rotation, which is the main focus of studies of mental rotation from cognitive psychology. As far as I can tell, these rotation tasks could in principle be solved by rotating the image plane (e.g. pixels, monitor, or the participant's head). Since you have 3D objects, why not add real 3D rotations (where a "hidden" part of the object due to self-occlusion is revealed)? This task would further strengthen the challenge of the benchmark.
2. Additional analysis of conditional performance would add further understanding to model performance. For example, line 262 states "If they fail to identify the specific change, any attempt at extrapolation would likely be incorrect". Is that true? I think the authors have data to assess this: is the accuracy on extrapolation different depending whether you condition on specification correctness?
3. Similarly to point (2), unless I have missed it, the authors do not appear to evaluate model performance on specification and extrapolation if the model is first given the answer to the previous step. For example: tell the model that the object increased in size, then instruct the model to apply the same transformation to a new image. Can the models at least do this? If not, this indicates and even more fundamental problem.

### Questions
- I am not sure that the descriptions of the instructions / prompts in the appendix are accurate. Specifically, the instructions in A2 (prompting models and human adults) read the same as the instructions in A4 (prompting models through reflection and self-critique).
- line 168 "three=year=old" should be hyphenated with "-"
- "No Change" was given as an option. Were no-change trials also included?
- "If they fail to specify the change, any extrapolation would likely be incorrect". --> is this true? conditional data?
- line 279 "handpicked by Developmental Psychology experts..." what does this mean? How could we evaluate the veracity of this statement?
- did human participants practice the tasks (with feedback)? I could imagine this could particularly matter for children.
- related work: A recent relevant paper on Bongard problems may also point towards possible ways to improve analogical reasoning: Depeweg, S., Rothkopf, C. A., & Jäkel, F. (2024). Solving Bongard Problems With a Visual Language and Pragmatic Constraints. Cognitive Science, 48(5), e13432. https://doi.org/10.1111/cogs.13432

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
KiVA is a new benchmark for assessing visual analogical reasoning in large multimodal models (LMMs) by comparing their performance to human adults and children. Inspired by developmental psychology, KiVA includes 1,400 visual transformations of everyday objects and tests models on identifying changes, quantifying them, and applying inferred rules to new scenarios. Experiments with models like GPT-4V, LLaVA-1.5, and MANTIS reveal that while LMMs can recognize "what" has changed, they struggle with quantifying "how" it changed and generalizing these rules, especially for complex tasks like rotation and reflection, highlighting a significant gap between LMMs and human reasoning abilities.

### Strengths
1. The dataset, inspired by developmental psychology, is unique in its simplicity, enabling assessments that even young children can complete. Its three-stage structure offers a clear breakdown of different analogical reasoning abilities in LMMs versus humans.
   
2. Extensive experimentation demonstrates specific strengths and weaknesses of LMMs, providing critical insights. For example, while models can recognize "what" changed in an image, they struggle to quantify "how" it changed and to generalize this rule to new objects (e.g., recognizing transformations in rotation or reflection).

### Weaknesses
1. The selection of visual analogy domains, while simple and fundamental, lacks sufficient justification regarding why these specific transformations were chosen over others. Intuitively, additional characteristics—such as edibility, danger, sharpness, and liveliness—are also essential features humans consider. For more complex natural scenes, it’s unclear whether the selected features are more significant than others.  The authors can provide further rationale for choosing these five factors or discuss the broader context of feature selection in visual analogy.

2. The discussion on how to improve LMM performance on these tasks is limited. It’s challenging to determine whether the low performance is due to limitations in analogical reasoning or to information loss during the initial perception stage. I’m curious whether translating the visual information into text would improve LMM performance on the task, as models might process textual representations more effectively. Investigating or discussing whether such an experiment might clarify whether perceptual or reasoning stages primarily limit LMM performance could help.

3. **Typographical errors**: Line 39 is missing a period after "reasoning"; text in Figure 1 is obscured by images, particularly in the percentage labels; Figure 8 is not referenced in the corresponding paragraph.

### Questions
1. Did the authors experiment with other visual analogy domains? If so, what were the results?

2. Could LLMs perform better on these tasks if the image contents were translated into text descriptions? Would textual encoding support LMMs in achieving higher accuracy in detecting “how” changes occurred and extrapolating rules?

### Soundness
3

### Presentation
2

### Contribution
2
