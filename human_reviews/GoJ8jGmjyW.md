# HowToCaption: Prompting LLMs to Transform Video Annotations at Scale

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 5, 5, 5

## Abstract
Instructional videos are an excellent source for learning multimodal representations by leveraging video-subtitle pairs extracted with automatic speech recognition systems (ASR) from the audio signal in the videos. However, in contrast to human-annotated captions, both speech and subtitles naturally differ from the visual content of the videos and thus provide only noisy supervision for multimodal learning. As a result, large-scale annotation-free web video training data remains sub-optimal for training text-video models. In this work, we propose to leverage the capability of large language models (LLMs) to obtain fine-grained video descriptions aligned with videos. Specifically, we prompt an LLM to create plausible video descriptions based on ASR narrations of the video for a large-scale instructional video dataset. To this end, we introduce a prompting method that is able to take into account a longer text of subtitles, allowing us to capture context beyond a single sentence. To align the captions to the video temporally, we prompt the LLM to generate timestamps for each produced caption based on the subtitles. In this way, we obtain human-style video captions at scale without human supervision. We apply our method to the subtitles of the HowTo100M dataset, creating a new large-scale dataset, HowToCaption. Our evaluation shows that the resulting captions not only significantly improve the performance over many different benchmark datasets for text-video retrieval but also lead to a disentangling of textual narration from the audio, boosting performance in text-video-audio tasks.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
Freely available web videos provide a rich source of multimodal text-video data. However, training on such data presents challenges, primarily due to the limited guidance offered by video subtitles for text-visual learning. 
In this paper, it tackle this issue by harnessing the capabilities of large-language models (LLMs). The paper introduces a novel method called HowToCaption, which entails instructing an LLM to generate detailed video captions based on automatic speech recognition (ASR) subtitles. 
To assess the effectiveness of the proposed HowToCaption method, the authors have curated a comprehensive HowToCaption dataset.This research showcases the potential of large-language models in creating annotation-free, large-scale text-video datasets.

### Strengths
- They propose  a novel dataset HowToCaption with high-quality human-style textual descriptions.

- The paper is easy to understand.

- This paper proposes a HowToCaption method to efficiently leverages recent advances in LLMs and generates high-quality video captions at scale without any human supervision.

### Weaknesses
- Novelty is insufficient.  This paper propose a approach, HowToCaption, prompting an LLM to create detailed video captions based on ASR subtitles.  However, similar methods have been used widely in dataset industry, including Tencent, OpenAI and so on, which could improve the worker's efficiency. Additionally, the process of method is so engineering and not suitable for ICLR.

- Experiment  is insufficient.  Firstly, in ablation study, the authors need to evaluate different speech recognition methods. Also they don't compare the influences of  captions length.  Secondly, Other downstream tasks such as VQA and Video Caption should be performed to prove the efficiency of dataset. the paper only test it on text-video retrieval which is far from enough. 

- Analysis is not enough. The setting of MSVD is not describled clearly.  the difference between HowTo100M and other large scale caption dataset such as WebVid10M  is also not describled. Also, this paper lacks some quantitative analysis  of HowToCaption, such as the distribution of caption length and the . Some settings are not be evaluate such as WebVid2M + CC3M, which is used widely in pre-trained field.

- Writing. In the process of reading, the same name of dataset and method sometimes could make reader confused. Apart from that, there are some grammar errors in paper, such as in third paragraph in 3.3, "first" is wrong.

### Questions
the same as weakness

### Soundness
2 fair

### Presentation
1 poor

### Contribution
1 poor

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This work proposes a framework to improve and collect text descriptions for videos by leveraging the powerful LLMs introduced recently. It designs a prompting mechanism that asks the LLM to rephrase subtitles extracted by automatic speech recognition (ASR) from the audio. To ensure good visual-text alignment, the framework (1) prompts the LLM to output timestamps for the generated sentences and (2) utilizes a vision-language model to filter and realign sentences that are not well aligned. The ablation studies confirm the effectiveness of the prompting mechanism. Furthermore, the experiments highlight (1) the importance of filtering and realignment, and (2) the vision-language model trained on the newly collected dataset outperforms baselines in various benchmarks on the text-to-video(+audio) retrieval task.

### Strengths
+) Combining LLM assures the production of high-quality descriptive sentences for videos. This is an intriguing and novel approach. Moreover, prompting the LLM to assist in text-video alignment is compelling. One could speculate that the commonsense knowledge or reasoning ability inherent in LLMs could greatly enhance alignment results. The qualitative samples depicted in Figure 2 appear impressive. The authors intend to make the dataset and code publicly available.

+) The ablation study in Table 1 clearly demonstrates the design choice behind the proposed prompting mechanism. Yet, it is somewhat surprising that, as per Table 2, long context information only yields marginal improvements in downstream tasks. The analysis of the effects of filtering and alignment also highlights the necessity for robust text-clip alignment annotations.

+) The zero-shot text-to-video retrieval results depict improvements from training BLIP on the newly assembled dataset. Although the progression seems marginal compared to WebVid2M data, Table 5 showcases that BLIP training on the proposed dataset achieves state-of-the-art performance in the text-to-video retrieval task.

+) The zero-shot text-video+audio retrieval outcomes highlight the strength of the proposed dataset, as models trained on it seemingly perform better.

### Weaknesses
-) A primary concern is that the authors solely validate the utility of the proposed dataset for the text-video(+audio) retrieval task. Presumably, the newly collected data could be an great resource for training foundational video-text representations, video captioning models, or video question-answering systems. It's somewhat disappointing to only see results related to retrieval tasks, especially considering BLIP's capability in visual captioning and visual question-answering.

-) While the authors show that filtering/alignment augments the final retrieval performance, a direct quantification of the assembled dataset would be valuable. For instance, what is the alignment accuracy when implementing the proposed filter/alignment technique? This would help subsequent users understand the noise level they might encounter when using the dataset.

-) Merely a few qualitative examples are presented in Figures 1 and 2, with no instances of failure cases. It would be beneficial if the authors included a broader range of generated sentences in the appendix, encompassing both successful and subpar examples. Additionally, given the notorious propensity of LLMs to fabricate or mislead, this work lacks both qualitative examples and quantitative analysis to assess such issues within the proposed dataset.

-) The work lacks qualitative results for the text-video(+audio) retrieval tasks. Specifically, for the text-video+audio task, insight into whether the proposed rephrasing method can effectively circumvent issues related to overly relying on ASR-generated information would be helpful.

### Questions
o) Can the authors present more qualitative examples from the newly introduced dataset, including both exemplary and flawed examples?

o) Could the authors share some qualitative results for the text-video(+audio) retrieval tasks?

o) Would the authors clarify why evaluations were restricted to retrieval tasks using BLIP? Is it feasible to assess the model's performance in visual captioning or question-answering scenarios?

o) The font size in Figure 1 appears too small. Could the authors enlarge it for easier readability?

o) Considering the generality of the proposed framework, which should be applicable to any video-text dataset, have the authors considered employing these techniques across all publicly available benchmarks to train a large-scale video-language model?

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper presents a novel way to construct large-scale video datasets, i.e., prompting an LLM to create both natural and rich video descriptions (based on ASR narrations). In this way, this paper contributes a large-scale video dataset, HowToCaption.

### Strengths
1. The paper is well-written and easy to follow. Different sections are well organized to present the proposed method and the experimental results.

2. Existing large-scale video datasets typically don't include detailed annotations (in the form of dense captioning with both timestamps and captions), since the time and labor costs of annotating temporal segments would be rather expensive. This paper presents a possible solution to this issue by automating the annotation process with pre-trained LLMs.

### Weaknesses
1. This work is more of a prompt engineering than a research paper, since the core components, i.e., the pre-trained Large Language Model and video-language encoder are both borrowed from existing literature. In essence, the technical contribution is a little bit weak.

2. Considering the generated captions include fine-grained timestamp annotations, it would be better to evaluate the proposed method on temporal localization tasks like moment retrieval, instead of text-to-video retrieval only that doesn't require fine-grained modeling.

### Questions
It would be better to evaluate the proposed dataset on more challenging tasks, e.g., dense video captioning or moment retrieval.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes a new dataset HowToCaption by prompting LLMs to modify the existing ASR subtitles in HowTo100M dataset in a human readable form. The newly created dataset is then pretrained on a model and compared against existing datasets and models on zero-shot video retrieval and text-video + audio retrieval.

### Strengths
**Clarity:** 
- The paper is well written and easy to follow.

**Significance:** 
- This paper proposes a new method to reduce the mis-alignment and noise in video-text datasets without the need for human supervision. It can provide a framework for future works to create large datasets without human supervision.
- Results on text-video + audio are promising. This is generally an ignored area of research due to lack of quality datasets. Training on the proposed dataset show its effectiveness.

### Weaknesses
**Unclear advantages of using this dataset on video retrieval task:**
- In Table-5 the authors present a comparison with SOTA models. However, a lot of models are missing in the Table. For example LAVENDER shows 37.8 points and 46.3 points on MSRVTT and MSVD datasets respectively which is more than HowToCaption while trained much smaller data (5.5M). This begs the question why does the community need to use the proposed dataset of 25M as opposed to Vid2.5M + CC3M. 

**Terminology usage:** 
- In section 3.1, the authors mention that the aim is to "generate" captions. However, the term "generate" might be mis-leading as the LLM merely "rephrases" semantically the ASR subtitle within a time-step in a more human-alike sentence and doesn't add any new details. 
- In section 3.3, the authors hint at "fine-grained" captions, I am not fully convinced that LLMs always produces better fine-grained captions. Sometimes, they add new unrelated details to the ASR subtitles. Since no quantifiable measure is provided in the paper, I would suggest the authors avoid this terminology.

### Questions
1. Are the ground truth time-stamps of ASR subtitles provided in the HowTo100M dataset? If not how are they determined?

2. In Section 4 the authors mention that dual-encoder of the BLIP is used as "T-V" model. Is it initialized with pre-trained weights of BLIP? How is it different from frozen-in-time architecture?

3. In Table-3, the authors present metrics of R-10 and MR for comparison which is rather unusual in video retrieval. Is there any specific reason for this? Is it possible to provide R-1 and R-5 scores?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair
