# TVBench: Redesigning Video-Language Evaluation

- Decision: Reject
- Scores: 6, 5, 3, 6

## Abstract
Large language models have demonstrated impressive performance when integrated with vision models even enabling video understanding. However, evaluating these video models presents its own unique challenges, for which several benchmarks have been proposed. In this paper, we show that the currently most used video-language benchmarks can be solved without requiring much temporal reasoning. We identified three main issues in existing datasets: (i) static information from single frames is often sufficient to solve the tasks (ii) the text of the questions and candidate answers is overly informative, allowing models to answer correctly without relying on any visual input (iii) world knowledge alone can answer many of the questions, making the benchmarks a test of knowledge replication rather than visual reasoning. In addition, we found that open-ended question-answering benchmarks for video understanding suffer from similar issues while the automatic evaluation process with LLMs is unreliable, making it an unsuitable alternative. As a solution, we propose TVBench, a novel open-source video multiple-choice question-answering benchmark, and demonstrate through extensive evaluations that it requires a high level of temporal understanding. Surprisingly, we find that most recent state-of-the-art video-language models perform similarly to random performance on TVBench, with only a few models such as Qwen2-VL, and Tarsier clearly surpassing this baseline.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper analyzed issues in the current most popular video-language benchmark (MVBench), and propose a new benchmark that alleviates the issues. Specifically, the authors provide solid evidence showing MVBench is less temporal, less visual, and simple solution of making an open-ended evaluation protocol can't address the problem. The authors then manually design question types and create questions by combining and filtering questions from existing video-language benchmarks. The authors show the resulting dataset is does not suffer from the issues.

### Strengths
- The paper works on an important problem of video-language model evaluation, and identify issues in a widely used benchmark with solid evidences. I believe the impact of the paper will be high.

- I like the analysis and experiments the authors provided for MVBench, the numbers in Table 1 and Table 2 are convincing, and the examples in Figure 2-4 are illustrative. The presentation is well structured and convincing.

- Evaluation on the new benchmark in Table 4 shows text and image only baselines are close to random, and shuffle or reverse the frames drops the performance for all models. This supports that the proposed benchmark emphasizes on temporal information. Table 4 contains results of a wide range of models.

### Weaknesses
- While this paper convincingly shows the proposed benchmark is better than MVBench, the authors did not provide discussion/ evidence whether if it is the final video language evaluation benchmark. For example, the benchmark might be too short / in too limited domains / person centric (I am not raising concerns on these particular issues). Some statistics about the datasets are needed.

- The 10 tasks picked in Table 5 looks a bit arbitrary / artificial to me. It will be helpful if the authors provide more rationale why these tasks are picked.

- It is unclear to my how the question-answers are created. Are then all from existing datasets, or the authors hired raters to filter / verify them?

### Questions
Overall this paper works on an important problem and provided a valid solution (a new benchmark). The analysis are backed up by solid experiments, and I believe this paper will have positive impact on the community. My concerns are mostly on discussions and clarity, and I expect the authors to address them in the rebuttal. My current rating is a weak accept, and I am happy to raise my rating if my concerns are addressed.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper investigates three issues of MVBench: 1) independence of video or video motion, 2) bias in the generated question-answer pairs, and 3) heavy reliance on world knowledge in questions. A significant part of the paper was written to prove and showcase these problems in the MVBench. A new benchmark called TVBench is proposed to mitigate these issues by redesigning the questions and available choices. The new benchmark attempts to prove that with no visual input or just image input, the models will perform like random guesses. Some strong video models also perform so even with full video inputs. The experiment also presents the results of inputting video frames in reverse order or shuffled order to prove that the benchmark questions requires understanding on the true video motion to be answered correctly.

### Strengths
- The paper addresses some critical issues with existing video understanding benchmarks, which is that the correct answers to many questions do not rely on information from the video or video motion. Thus, proposing a new benchmark to resolve these issues are well-motivated.
- The paper explains in detailed examples and some ablation studies to prove that these problems exist widely in MVBench.
- Based solely on the reverse & shuffle order experiment results in Table 4, it seems that TVBench indeed improves some questions' reliance on video inputs and the motion contained in those videos.

### Weaknesses
While I appreciate the authors' great efforts to prove their statements of the issues, I am confused by many details after reading through the paper and am not fully convinced by the quality of the new benchmark.

- Some of the results in the figures and tables, or the way they are presented, can be confusing. In Fig.1 left, the trend seems linear after the models achieve a certain level of performance (>50) on MVBench. In Fig. 1, right, MVBench shows a performance drop in VIdeoChat2 when the video is reversed. How do these results support the claim that MVBench does not measure temporal understanding? What is Table 1 trying to prove? I cannot compare the results of GPT-4o + image inputs with Gemini 1.5 Pro + video input to the conclusion that a single image is sufficient. You should at least fix other variables and leave the input as the one changing to prove that. Besides, even though the results are close, did you prove that the questions answered correctly are the same ones? It's a similar issue in Table 2 that I cannot understand how text-only rows could be compared to video input rows since they are using different models. 

- Since it's a benchmark, it should attempt to document the performance of as many models as possible. A lot of video models are missing, such as Video-LLaVA, mPLUG-Owl, PandaGPT, ImageBind, Video-LLaMa and etc. In addition, the GPT-4 series can accept multiple images, which is essentially the same as video models with video inputs -- they all need to sample a certain number of frames as multiple image inputs. You can also concatenate multiple frames into one image and feed into the GPT-4 series. It doesn't make sense to me to only benchmark GPT-4o with a single frame input. 

- Writing is a big issue in this paper. So many details make it hard to understand the paper without being confused. 
  - In Fig. 1, what is the unit of the axes? 
  - Table 1 is presented but never referred to in the text. If I understand correctly, some "Tab. 2" should refer to Table 1 instead. Please also choose between "Tab 2" and "Tab. 2" so that searching is convenient. 
  - I think the paper shows an excessive amount of bad examples from MVBench, which makes some of these figures unnecessary. While it's good to identify and prove the existence of these problems, more efforts should be spent convincing the readers that the "proposed" benchmark is high-quality and indeed resolves these issues. 
  - I understand that Sec. 4 is trying to show that open-ended qa and evaluation are not reliable, but how does that matter with the main point of this paper? Multiple-choice-based QA and open-ended QA are different settings used in different benchmarks or evaluations. It doesn't convince me that TVBench is high quality by showing the weaknesses of open-ended QA -- they are different settings. 
  - In line 430, *following the model provided in Tab. 5 for each task*, what is *model* in Table 5? There is no *model* column in Table 5 and the appendix is too short to provide enough context. How many templates are you using? Are the templates in Table 5 showing all you are using? How did you collect these templates? If you have hired annotators, how did you ensure the quality of these templates? These are all important details to be included in the paper to convince readers about the quality of TVBench. 
  - This is a minor point, but I don't favor using statistics of **Huggingface downloads** as some sort of evidence in the introduction (lines 37-38). Regardless of whether you are trying to use the number to support MVBench or question its reliability, it's better to appreciate **scientific merits** instead of **popularity metrics** in academic writing.

### Questions
Please see the weaknesses.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
2

### Summary
The paper introduces TVBench, a new video-language benchmark that addresses critical flaws in existing benchmarks like MVBench. The authors identified three problems with current benchmarks such as:
- Single frames are enough
- Question text reveals answers.
- Common knowledge beats video.

The authors demonstrated those problems by showing that both text-only language models and single-frame vision models perform well on existing benchmarks. In contrast, when it comes to TVBench, most state-of-the-art video-language models perform close to random chance.

The benchmark consists of 10 temporal tasks across 2,654 question-answer pairs, ensuring models must understand the sequence and timing of video events to succeed. The authors validated their benchmark by showing that shuffling or reversing video frames significantly impacts performance, unlike previous benchmarks.

### Strengths
- MVBench Analysis: Thorough and systematic identification of MVBench limitations with clear evidence.
- Validation Methods: Creative use of video shuffling/reversal to verify temporal understanding requirements.
- Benchmark difficulty: Evaluation showing most current models fail at true temporal reasoning.

### Weaknesses
- Problem analysis: even though the authors did identify the problems with MVBench, the analysis of other benchmarks is quite limited. The paper states “We conduct a comprehensive analysis of widely used video question-answering benchmarks” while focusing only on MVBench. Several datasets in the relative section can be analyzed similarly and it is still uncertain if all of those datasets also have those problems
- Small amount of dataset examples: there are 10 different tasks within the dataset, yet the paper shows only one example from the whole dataset.
- Task design: Authors state: “Questions should not be answerable using spatial details from a single random frame or multiple frames e.g. after shuffling them.” However, even the only given example from TVBench about scenes in the movie can be solved using two frames. Additionally, tasks in TVBench like Scene transition, Action Antonym, and Moving Direction can be solved with only two frames instead of one. Image LLM evaluation with more frames would be important.
- Benchmark creation details: There are very few details on how the dataset was collected and annotated. In general, given details about the dataset creation are very vague. For example: "Instead of including random, easy negative candidates, we define hard candidates that cannot be discarded without temporal information". How do you generate the hard negative examples?

### Questions
- How the dataset was annotated, how exactly did you come up with the wrong answers? 
- Show more examples from the dataset. Several examples from each of the tasks. The examples can be illustrated similarly to how you did it in Figure 5
- Are those MVBench problems shown in other benchmarks?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces TVBench, a new benchmark for testing video understanding capability of multimodal models. Flaws in widely used existing benchmark (MVBench) are demonstrated, namely, spatial biases, textual biases and reliance on world knowledge. In addition, it's also shown that open-ended benchmarks can contain similar biases. TVBench is constructed from pre-defined templates in order to mitigate these biases and test temporal reasoning capabilities. Supporting experimental results demonstrate that state-of-the-art models struggle on this benchmark. Similarly, text-only or image-text foundation models struggle to beat random chance signifying the difficulty of this benchmark compared with existing benchmark.

### Strengths
The paper is well-written and easy to understand. It tackles an important area of video understanding, i.e. the lack of strong benchmarks that test temporal reasoning in videos. The presentation clearly analyzes drawbacks of existing benchmarks and proposes a new benchmark.
- The QA pairs don't use LLMs in the loop, and thus can avoid many hallucination related issues.
- The performance of SOTA models is very low (Table 4). This indicates the benchmark is indeed difficult.
- Clear contrast with MVBench is demonstrated, especially using text-only and image-only models. This justifies most of the claims in the paper. 
- A significant, and often overlooked issue in open-ended evaluations is pointed out in Section 4. Using closed-source proprietary models whose back-ends may change arbitrarily to score open-ended responses and track our progress on video understanding can be misleading.

### Weaknesses
The main weakness of this work is around experimentation. 
- Human baseline performance is not presented. This is important to judge the quality of the benchmark and the presented results.
- Different models are used in Table 2 to make the claim that MVBench has textual bias. Ideally, the same model (ideally the best model) needs to be presented with text-only and video as inputs to justify the claim.
- Similarly, in Table 4, different models are used to compare different biases (text, image, video) of the model. 

Further Limitations:
- Using standard template QA pairs may limit the range of video understanding being assessed.
- In Figure 2 and the associated text in the paper, it's presented as if detecting the absence of something is an easy task. However, by definition, one must watch the entire video to make sure what we're detecting is indeed absent.

### Questions
- Can we use the same model and ablate text-only, image, video, shuffle, reverse, etc. in Table 4? Ideally Gemini 1.5 pro as it performs the best on this benchmark?
- MVBench is presented as not a great benchmark. However, its performance is also not saturated (best model achieves 67.7 in Table 4). Do the remaining QA pairs satisfy the criteria set in the paper? What is the size of the data? Can we remove the bad examples from MVBench and get a bigger and better dataset than TVBench?

EDIT: Updated score based on the rebuttal.

### Soundness
3

### Presentation
4

### Contribution
3
