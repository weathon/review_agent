## Human Reviewer 1

### Summary
This paper proposed AVHBench, a benchmark to assess cross-modal hallucinations in audio-visual LLMs. The benchmark is built on existing datasets VALOR and AudioCaps, and contains about 6k QnA pairs and 1k audio-visual captions across four cross-modal tasks, including audio-driven video hallucination, video-driven audio hallucination, audio-visual matching and audio-visual captioning. The authors design a semi-automatic pipeline for data anntation. Several open-source audio-visual LLMs are evaluated on AVHBench, and the results show that most existing audio-visual LLMs suffer from cross-modal hallucinations. To alleviate this problem, the authors further enhances Video-LLaMA through audio feature alignment and LoRA fine-tuning, proving that audio-visual hallucinations might come from insufficient training on paired audio-visual data.

### Strengths
- This paper proposed a audio-visual hallucination benchmark mainly focusing on cross-modal hallucination evaluation, which is an aspect that has received little attention.

- Some tasks in the benchmark provide new perspectives on the study of multi-modal hallucinations, like *Audio-driven Video Hallucination* and *Video-driven Audio Hallucination*. The inability of the model to distinguish between information from audio or video may be a vital reason that causes multi-modal hallucination.

- The paper is well-written, clear and easy to understand.

### Weaknesses
- The benchmark uses existing datasets VALOR and AudioCaps, which may introduce biases inherent to those datasets, potentially affecting the generalizability and validity of the evaluation.

- In this benchmark, for human speech, the authors seem to have only considered the event of "one person is speaking," instead of taking into account the content of what is being said. The content of speech contains a wealth of information and is very likely to contribute to the hallucinations of audio-visual LLMs. However, the paper seems to overlook this scenario.
    
    For example, consider a scenario where a person in a video is saying to himself "Yesterday I heard a dog barking", but neither the dog nor the barking sound appears in the video or audio. Models are prone to hallucinations in this scenario. 

- It is recommended to evaluate more recent audio-visual LLMs, like Video-LLaMA 2, or video-SALMONN.

- Since the video is so rich in content, long text is required to descirbe the video completely. However, for the "audio-visual captioning" task, only a short caption is provided as the groundtruth. This suggests that the groundtruth caption is likely to contain only the important information in the video and omit the secondary information. However, it is still possible for the model to describe something that is present in the video but is not hallucination. That's why I'm concerned about the correctness of the "audio-visual captioning" task of AVHBench.

### Questions
- After LoRA fine-tuning, the results of the model on AVHBench become significantly better. Does this suggest that the model may just not be able to do the judgement questions, or that it hasn't seen the mismatched audio/video and therefore performs poorly on that test set, rather than the model having a large number of hallucinations?

- Do the authors consider providing Gemini's results on the benchmark?

### Soundness
3

### Presentation
2

### Contribution
3

### Rating
6

### Confidence
5

---

## Human Reviewer 2

### Summary
**Summary of this paper:**

This work proposes a cross-modal hallucination evaluation benchmark called AVHBench, which comprises four different tasks: audio-driven video hallucination, video-driven audio hallucination, audio-visual matching, and audio-visual captioning. Besides, the paper analyzes the presence of cross-modal hallucinations and investigating their potential causes using the proposed benchmark on six recent audio-visual LLMs.

**Strengths:**

1.	This paper introduces the first comprehensive benchmark specifically designed to evaluate the perception and comprehension capabilities of audio-visual LLMs.

2.	Authors include a clear organization of the related literature on multimodal large language models (MLLMs) and hallucinations in MLLMs.

3.	The figures in this paper are well-designed.

**Weakness:**

The datasets used in this paper are relatively limited in terms of scene diversity, which may hinder the benchmark's ability to evaluate the model's performance in a broader range of real-world scenarios.

**Comments, Suggestions And Typos:**

To provide a more comprehensive evaluation of this benchmark, it is recommended to increase the number of evaluation models and diversify the scenarios.

### Strengths
Refer to the summary

### Weaknesses
Refer to the summary

### Questions
Refer to the summary

### Soundness
3

### Presentation
3

### Contribution
2

### Rating
5

### Confidence
4

---

## Human Reviewer 3

### Summary
The paper introduces AVHBench, a novel benchmark designed to evaluate cross-modal hallucinations in audio-visual large language models (LLMs). Addressing a critical gap, AVHBench tests models on their ability to handle complex interactions between audio and visual cues without generating erroneous outputs, known as cross-modal hallucinations. It includes four tasks: audio-driven video hallucination, video-driven audio hallucination, audio-visual matching, and captioning. Using a semi-automated pipeline for dataset curation, AVHBench facilitates robust assessment and enhancement of model accuracy in handling multimodal inputs.

### Strengths
1. This paper is well-written.
2. The motivation for proposing benchmark for audio-visual hallucination is clear.
3. The tasks proposed by the benchmark are valuable, and the semi-automatic solutions designed are also reasonable.

### Weaknesses
As a benchmark paper, more evaluation models are needed. For example, the recent video-salmonn[1], advanced Gemini[2], and unimodal models like Qwen-audio[3,4] for audio, llava-onevision[5] for vision.

1. Sun, Guangzhi, et al. "video-SALMONN: Speech-enhanced audio-visual large language models." arXiv preprint arXiv:2406.15704 (2024).
2. Team, Gemini, et al. "Gemini: a family of highly capable multimodal models." arXiv preprint arXiv:2312.11805 (2023).
3. Chu, Yunfei, et al. "Qwen-audio: Advancing universal audio understanding via unified large-scale audio-language models." arXiv preprint arXiv:2311.07919 (2023).
4. Chu, Yunfei, et al. "Qwen2-audio technical report." arXiv preprint arXiv:2407.10759 (2024).
5. Li B, Zhang Y, Guo D, et al. Llava-onevision: Easy visual task transfer[J]. arXiv preprint arXiv:2408.03326, 2024.

### Questions
see weaknesses.

### Soundness
2

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
3

---

## Human Reviewer 4

### Summary
This article proposes a comprehensive benchmark for evaluating the perceptual and understanding capabilities of audiovisual LLMs, which includes four tasks: Audio-driven Video Hallucination, Video-driven Audio Hallucination, and Audio-visual Matching. It assesses the hallucination phenomena of existing AV-LLMs, as well as the cross-modal matching and reasoning abilities of these models, and provides relevant analyses and conclusions.

### Strengths
1. Well-Written and Accessible: The article is well-written, well-motivated, and easy to follow.
2. Comprehensive Benchmark: The proposed benchmark is comprehensive, containing several complementary dimensions and devised tasks.
3. Valuable Takeaways: The takeaways provide valuable analyses, conclusions, and insights.
4. The paper presents three methods to improve the trustworthiness of multimodal large language models (MLLMs).

### Weaknesses
1. How do the authors ensure the quality of the dataset? Are there any evaluation measures in place?

2. For the tasks of Audio-driven Video Hallucination and Video-driven Audio Hallucination, how do the authors ensure that the visuals or audio contain objects that are either silent or not present? Most events, objects, and sounds in videos are quite singular, with the sound-producing objects being consistent and uniform in both video and audio.

3. How should the presence of ambient sounds, such as wind or rain, which do not correspond to specific "objects" in the visuals, be handled?

4. If there are multiple objects of the same type in the video that do not make sounds, for example, several dogs that are silent while a dog off-screen is barking, or if the audio consists largely of narration while the video shows people, would such cases still be classified as "present in both video and audio"?

5. How is the situation handled when the audio is background music? Additionally, in cases where the samples themselves are inconsistent between video and audio but originate from the same video source, the model may determine them as not matching, while the ground truth indicates they do match. How is this discrepancy addressed?

### Questions
See Weaknesses.

### Soundness
3

### Presentation
4

### Contribution
3

### Rating
8

### Confidence
4

---

## Human Reviewer 5

### Summary
The paper proposes AVHBench, a benchmark for evaluating cross-modal hallucination for audio-visual language models. The paper finds that the current models fall short when it comes to evaluations designed to test hallucination, with a performance close to random guesses. The eval bench comes out of a GPT-4 aided data generation pipeline with human verification. The author fine-tuned their model using the data coming out of the same pipeline without any evaluation and found that they were able to improve the hallucination issue significantly.

### Strengths
Looking into visual-audio-language model cross-modal hallucination seems to be novel. The paper is well-written and clearly motivated.

### Weaknesses
* The synthetic dataset seems to be the important component for both the evaluation set and the training set.  There seem to be several sources of error that the author did not either discuss or give some analysis on:
 1. For Audio-Visual disentanglement: (a) error might come from the visual tagging process (b) The prompt in Table s1 can not distinguish the sound or appearance of multiple instances of the same type of object. (e.g. given two people and the sound of a human talking, it will recognize someone is talking but have no idea whether the one who is talking is in view)
2.  For audio-visual caption generation. Given two unaligned audio and visual captions, likewise, the language model is not guaranteed to be able to capture the correspondence between visual and audio information.
Given the above, I think it's crucial that the author give some error analysis of the proposed pipeline (e.g. from the verification data of manual labour.) 

* Related to the previous point, the proposed pipeline is not able to generate audio-visual captions/questions that require temporal reasoning. Would be good to have some discussion on this.

* Also related to the above point, to show the proposed pipeline is truly useful for generating data to fine-tune the audio-visual model, it would be nice to see some more results on how the fine-tuned model performs on other audio-visual benchmarks.(In Table 4 beyond VAST Captioning dataset)

### Questions
My concern majorly lies in the error analysis and limitation of the data generation pipeline as well as the results on the fine-tuned models.

### Soundness
3

### Presentation
3

### Contribution
2

### Rating
6

### Confidence
3