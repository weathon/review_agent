# Gaze-to-text Generation: Beyond Categorical Decoding of Human Attention

- Avg Score: 2.67
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 2

## Abstract
We introduce a novel learning problem: decoding gaze into natural language descriptions of human goals across diverse visual tasks. Unlike prior work, which frames gaze decoding as a classification task over predefined categories, we formulate it as a generative learning problem: training a model to produce free-form descriptions that capture the rich context, nuance, and open-ended nature of human intentions beyond fixed labels. 
    
To this end, we introduce Gazette, the first gaze-to-text decoding framework. Based on multimodal large language models (MLLMs), Gazette learns to decode gaze scanpaths into natural language for goals that may extend beyond categorical labels and require articulation in natural language. To help Gazette filter out individual differences in gaze behavior and learn the goal-specific spatiotemporal dynamics crucial for generating accurate natural language goal descriptions, we propose a novel strategy that leverages the encyclopedic knowledge and reasoning abilities of a large language model to synthesize natural language explanations of goal-directed attentional behavior called think-aloud transcripts. Instruction tuning on these synthetic narratives allows Gazette to achieve state-of-the-art performance in gaze decoding across multiple tasks, demonstrating its generalizability and versatility, thereby enabling gaze to serve as a powerful, non-intrusive cue for inferring human goals and intentions in diverse scenarios.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes Gazette, which takes an image along with a sequence of coordinates (representing the gaze scanpath within the image) as input, and outputs a natural language description of that gaze scanpath. The authors use GPT-4 to generate natural language descriptions as ground truth to train the LLaVA-1.5-7B model, and evaluate the system by measuring the similarity between the model-generated descriptions and the GPT-4 generated ground truth (with metrics like BLEU and LLM-as-a-Judge).

### Strengths
- The idea of converting gaze into text is intuitively reasonable and meaningful.

### Weaknesses
- The experimental objective and design are incorrect. Since the paper claims to be the first to propose a gaze-to-text framework, the focus of the experiments should be on validating the advantages of representing gaze as text, rather than merely evaluating the accuracy of the generated descriptions. Under this premise, the authors should design downstream tasks that require gaze understanding and compare the performance of Gazette with other gaze-analysis algorithms. A VQA dataset that incorporates gaze would be a more appropriate benchmark than textual similarity.  A very straightforward baseline (which is not sufficient but serves as a direct example) would be to use VLMs or object detection algorithms to identify objects at the key points of the gaze scanpath and use the resulting object sequence as an additional input to assist VLMs in performing VQA. In addition, the quality of the GPT-4 generated ground truth is not verified, which undermines the reliability of both the model’s performance and the experimental evaluation.
- The method design is simplistic. Gazette uses a large, closed-source model to generate data and then trains a smaller, open-source model. This is very common and, to some extent, even outdated. The only additional design introduced by Gazette is splitting the output text into coarse-grained and fine-grained descriptions, which is also not a novel idea.
- The paper lacks necessary analytical experiments, especially regarding how gaze scanpath are fed into VLMs. Although many VLMs can handle coordinates within an image, this capability is not effectively validated. The authors should compare the current input method with alternative approaches, including (but not limited to) directly annotating the gaze scanpath on the image.

### Questions
See the weaknesses section. If the first and third issues can be addressed, I would be willing to raise my score.

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
1

### Summary
This work introduces decoding gaze into natural language descriptions of human goals across diverse visual tasks. Unlike prior work, which frames gaze decoding as a classification task over predefined categories, this paper formulates it as a generative learning problem: training a model to produce free-form descriptions that capture the rich context, nuance, and open-ended nature of human intentions beyond fixed labels.

### Strengths
1. This paper introduces the task of unconstrained decoding of goal-directed attention, where top-down goals are expressed in natural language, supporting a wide range of human gaze behaviors.
2. This paper proposes a text-generative MLLM-based framework, an instruction-tuned to decode a scanpath of gaze fixations (during image viewing) to natural language.

### Weaknesses
1. The motivation of this work is not very clear. For example, why is decoding gaze as natural language better than simple labels?
2. The innovation of the proposed model is quite limited. For example, the explanation of gaze seems to derive from the prompting strategy of LLMs.
3. The data used to generate natural language for gaze is not clear.

### Questions
1. How is the data for natural language decoding obtained?
2. Why is decoding gaze as natural language better than simple labels?
3. The use of gaze information in the model is based on prompting?
4. What's the relationship between gaze and visual representation?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes Gaze-to-text Generation, shifting from classifying gaze to generating natural-language descriptions of a user’s intention from their scanpath and the viewed image. The authors introduce Gazette, an MLLM-based framework (built on LLaVA-1.5-7B) that textualizes gaze and outputs a “cognitive context” describing behavior type (search, referral, VQA) and the specific goal or stimulus. They also synthesize think-aloud supervision with GPT-4 from multiple observers’ scanpaths on the same task, teaching the model to focus on task-driven spatiotemporal patterns.

### Strengths
1. Propose a new task Gaze-to-text Generation
2. Generate a dataset of Gaze-to-text Generation using ChatGPT
3. Finetune a LLaVA1.5 model to perform the task.

### Weaknesses
1. The necessity of the proposed new task Gaze-to-text Generation is quetionable. If the gaze points are already there, an additional referring which generates the text will lead to a structure gaze movement. Is natural language really necessary to describe the gaze movement? I think structured representation contains enough information to infer intention or study human behavior.
2. Do we really need to finetune a MLLM to perform this tasks or we can achieve the same performance with assembled set of foundations models off-the-shelf. One example, is using referring models to extract what each gaze point is, then directly summarize with a LLM. This could serve as a strong baseline which is missing in the paper.
3. Lack of technical contribution. This paper only proposes a task with a generated dataset and then finetunes a LLaVA without making any adaptions to LLaVA for the task. All these efforts are largely reuses existing techniques (e.g. using ChatGPT to generate data is proposed in the LLaVA paper and has been widely adopted, LLaVA model is also widely used and there is no technical improvement in this paper to improve LLaVA).

### Questions
Please see Weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2
