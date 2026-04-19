# Fugatto 1: Foundational Generative Audio Transformer Opus 1

- Decision: Accept (Poster)
- Scores: 6, 6, 8

## Abstract
Fugatto is a versatile audio synthesis and transformation model capable of following free-form text instructions with optional audio inputs. While large language models (LLMs) trained with text on a simple next-token prediction objective can learn to infer instructions directly from the data, models trained solely on audio data lack this capacity. This is because audio data does not inherently contain the instructions that were used to generate it. To overcome this challenge, we introduce a specialized dataset generation approach optimized for producing a wide range of audio generation and transformation tasks, ensuring the data reveals meaningful relationships between audio and language. Another challenge lies in achieving compositional abilities -- such as combining, interpolating between, or negating instructions -- using data alone. To address it, we propose ComposableART, an inference-time technique that extends classifier-free guidance to compositional guidance. It enables the seamless and flexible composition of instructions, leading to highly customizable audio outputs outside the training distribution. Our evaluations across a diverse set of tasks demonstrate that Fugatto performs competitively with specialized models, while ComposableART enhances its sonic palette and control over synthesis. Most notably, we highlight our framework's ability to execute emergent sounds and tasks -- sonic phenomena that transcend conventional audio generation -- unlocking new creative possibilities. \href{https://fugatto.github.io/}{Demo Website.}

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces Fugatto, a versatile audio synthesis and transformation model capable of following free-form text instructions with optional audio inputs. The paper focuses on enabling generalist audio models with emergent capabilities, similar to large language models (LLMs), but optimized for audio. Fugatto can handle tasks such as audio generation, transformation, and compositional tasks by using a new dataset generation approach and a technique called Composable Audio Representation Transformation (ComposableART).  This method enhances classifier-free guidance, enabling complex operations like combining, interpolating, and negating instructions.

### Strengths
(1) Generalist Audio Model: Fugatto offers a broad range of audio generation and transformation capabilities, filling the gap between specialist models and generalist models.

(2) ComposableART: The novel technique extends classifier-free guidance to handle compositional tasks, allowing the model to compose instructions in ways that were not seen during training.

(3) Dataset Generation: The paper provides a strategy for generating diverse and dynamic datasets, using LLMs for instruction creation and data augmentation. The authors claims they will release these the dataset and instruction generation code, which is useful for the research community.

### Weaknesses
At this stage, I believe the presentation of this paper is not strong enough. I still have the following questions:

a. How should we understand "unsupervised multitask learning"? In your training tasks, each task has both inputs and labels, such as TTS and TTA. I am not clear on how this relates to unsupervised multitask learning. UniAudio and AudioBox are closely related to your work, and I believe both Fugatto and UniAudio are trained on multiple tasks. Generally, we refer to 'unsupervised learning' when there are no labels, such as in the pre-training of LLMs.

b. Your main framework is based on Flow Matching (FM). How do you control the duration? Since FM is a non-autoregressive model, this could be a challenge. For example, AudioBox or VoiceBox use a phoneme duration predictor. Are you using a duration predictor for TTS? Similarly, for tasks such as singing generation, how is the duration controlled?

c. In Table 1, what qualifies as "large-scale data" in terms of hours? It seems UniAudio uses more than 100,000 hours of audio data. It would be helpful if you could list the number of hours for each model, as this would provide readers with a clearer understanding of what constitutes 'large-scale.'

d. The definition of "Emergent properties": In the context of LLMs, we describe 'emergent properties' as the ability to solve unseen tasks and perform reasoning. However, in this paper, the examples given for emergent properties seem to focus on generating sounds that don’t exist in the real world or the training data. From my perspective, this doesn’t fit the definition of 'emergent properties,' as the model has already learned to understand sound types based on text descriptions. I strongly recommend discussing this point with other reviewers, as it currently seems like a bit of an overstatement.

e. From a high-level perspective, Fugatto follows the multi-task training paradigm used in UniAudio and AudioBox. The authors need to explicitly highlight the advantages Fugatto has over AudioBox. That said, I do agree that the model's performance is impressive.

In conclusion, I agree that building a generalist model is a valuable topic, and this paper demonstrates good performance. However, the authors need to improve the presentation to help readers better understand the contributions.

I am happy to improve my score during the rebuttal stage if the authors solve my concerns.

### Questions
Refer to weakness.

### Soundness
3

### Presentation
2

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
The paper proposes Fugatto 1, a versatile audio synthesis and transformation model, has capable of following free-form text instruction with optimal audio inputs. To address the problem that audio data does not inherently contain the instructions that were used to generate it, the paper introduces a specialized dataset generation approach to enhance the data reveals meaningful relationships between audio and language. Additionally, the paper proposes ComposableART with CFG to enhance the compositional abilities of the model.

### Strengths
The paper proposes Fugatto, a large foundational audio synthesis and transformation model. This paper attempts to address the problem of audio data does not highly correspond to the text-instruction by a data generation strategy and extend CFG to support compositional guidance. The paper provides extensive experiments across different tasks in the audio domain.

### Weaknesses
The paper is a bit difficult to follow, and the proposed methods are trivial incremental. The main contribution of this paper is to train a generalist model with additional enhanced diverse synthesized datasets. I admit that the motivation of exploring a generalist model to benefit downstream tasks is good; however, the paper methodology lacks insights in the domain of audio synthesis and generation. The contribution of this paper for audio domain research is limited. In my opinion, this paper is more suitable for a technical report. 

I feel lost when reading the experiment section, the paper should provide at least a brief introduction for evaluation metrics used in the experiments, experimental details for adapting the proposed method in each single-task, and insights about why to adapt the method to each different single-task.

Even though the paper showcased its applicability in various audio related tasks, however, it is not convincing to me the advantage of the proposed generalist model compared to other specialist models for each single-task. For example, in the TTS experiment, only speech similarity and intelligibility been evaluated, a further analysis, such as MOS study or F0 visualization compared to other methods and GT, is necessary for evaluating speech naturalness and quality to showcase the method can provide more natural speech than existing methods; No model comparison for the SVS experiment; Table 3(b) compares its performance with other specialist models, however, MusicGen and AudioLDM2 have different focuses (music v.s., multi-modality audio generation). The experiment is not a fair comparison and not convincing.

### Questions
1. If I understand correctly, the paper provides an incremental method for building a large foundational audio generation model. What is the technical contribution of this paper?

2. Have you verified the quality of the synthesized new dataset? How do you ensure the synthesized data is high-quality and strongly corresponds to the prompt instruction?

3. Will you release the codes, the new datasets and the pretrained model?

### Soundness
3

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper presents Fugatto, a generalist audio synthesis model capable of handling various audio generation and transformation tasks using text prompts. It introduces a dataset generation method that broadens Fugatto’s capabilities. In addition, ComposableART is proposed which extends classifier-free guidance to allow compositional control over audio generation. The model is evaluated on tasks across different audio domains, showcasing its versatility and competitive performance.

### Strengths
1. Fugatto handles multiple tasks with text and audio inputs.
2. The approach to synthetic instruction generation and data enrichment is well-structured, supporting the model’s generalization across diverse tasks
3. ComposableART enables flexible audio composition, interpolation, and negation of prompts, adding control over generated outputs.

### Weaknesses
1. Techniques such as using LLMs for synthetic instruction generation lack novelty, which may challenge the paper's originality and scientific contribution.

3. The comparison with state-of-the-art specialist models is limited for certain tasks, and the overall impact of ComposableART on performance remains unclear.

### Questions
1. How does ComposableART impact model performance quantitatively on specific tasks? It is recommended that the authors include more subjective and objective comparisons of the proposed system against task-specific models, such as LASS-Net and AudioSep for text-based audio removal.

2. Could the authors clarify which benchmarks or metrics were used to evaluate compositional tasks?

3. As a scientific conference submission, I would personally advise against the use of excessive fancy fonts and varied colors within the paper.

### Soundness
3

### Presentation
3

### Contribution
3
