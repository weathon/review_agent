# LoCoT2V-Bench: A Benchmark for Long-Form and Complex Text-to-Video Generation

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 6, 4, 2

## Abstract
Recently text-to-video generation has made impressive progress in producing short, high-quality clips, but evaluating long-form outputs remains a major challenge especially when processing complex prompts. Existing benchmarks mostly rely on simplified prompts and focus on low-level metrics, overlooking fine-grained alignment with prompts and abstract dimensions such as narrative coherence and thematic expression. To address these gaps, we propose LoCoT2V-Bench, a benchmark specifically designed for long video generation (LVG) under complex input conditions. Based on various real-world videos, LoCoT2V-Bench introduces a suite of realistic and complex prompts incorporating elements like scene transitions and event dynamics. Moreover, it constructs a multi-dimensional evaluation framework that includes our newly proposed metrics such as event-level alignment, fine-grained temporal consistency, content clarity, and the Human Expectation Realization Degree (HERD) that focuses on more abstract attributes like narrative flow, emotional response, and character development. Using this framework, we conduct a comprehensive evaluation of nine representative LVG models, finding that while current methods perform well on basic visual and temporal aspects, they struggle with inter-event consistency, fine-grained alignment, and high-level thematic adherence, etc. Overall, LoCoT2V-Bench provides a comprehensive and reliable platform for evaluating long-form complex text-to-video generation and highlights critical directions for future method improvement.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces LoCoT2V-Bench, a new benchmark designed specifically for evaluating long video generation under complex input conditions.
The authors argue that existing benchmarks often rely on simplified prompts and focus on low-level metrics, neglecting higher-level dimensions such as narrative coherence and thematic expression. To address this gap, LoCoT2V-Bench presents two main contributions: (1) A suite of longer and more complex prompts derived from real-world videos, incorporating elements like scene transitions and event dynamics. (2) A multi-dimensional evaluation framework that, in addition to traditional metrics, proposes novel dimensions such as event-level alignment, content clarity, and HERD.

### Strengths
**1. Well-motivated and Significant Problem**

As video generation models advance, the evaluation of longer, more complex videos has become a critical challenge. This paper accurately identifies the limitations of existing benchmarks and proposes a solution tailored for long video under complex prompts, which is crucial for advancing the field.

**2. Comprehensive Evaluation Dimensions**

The evaluation is very comprehensive.  It not only covers fundamental metrics like Static Quality, Text-Video Alignment, and Temporal Quality but also innovatively introduces higher-level dimensions like Content Clarity and the Human Expectation Realization Degree. HERD, in particular, is a valuable exploration in video generation evaluation as it attempts to quantify abstract concepts like emotion, narrative, and character development.

**3. Extensive Experiments and In-depth Analysis**

The authors have conducted a thorough evaluation of existing open-source LVG models. The evaluation not only reports overall performance but also delves into deeper exploration (e.g. Sec 4.2~4.4).

### Weaknesses
**1. Limited Scale of the Benchmark**

The benchmark consists of 240 samples distributed across 18 themes. My main concern is that this sample size may be insufficient to draw robust evaluation about model capabilities.

**2. Evaluation Reliability of HERD**

The proposed automated evaluation pipeline, especially the HERD metric, is heavily dependent on multiple third-party LLM/MLLMs. The generation and evaluation process for HERD involves a long chain of model calls. How do the authors evaluate and control the accumulated errors in this chain? How sensitive is the final metric to a potential failure at any stage of this pipeline?

**3. Handling the Errors during Evaluation**

For fine-grained metrics like "event-level temporal consistency," the calculation seems to presuppose that the corresponding events and subjects can be accurately located in the video. I think this assumption is questionable in complex scenarios. Also, there might be multiple subjects in the same video clip, or even in the same video frame. Does subject extraction confuse different subjects in these complex settings? 
It is also unclear how the system robustly handles intermittent subject presence, such as when a subject is occluded or temporarily leaves the frame and then reappears.

### Questions
Please see weaknesses.

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
4

### Summary
This paper presents LoCoT2VBench, a benchmark for evaluating video generative models on long video generation. It presents a prompt suite consisting of 240 samples from 18 themes, derived from real-world videos, and an evaluation framework consisting of a taxonomy of novel metrics.

The paper also evaluates open source video generation methods on these metrics, and does an insightful analysis over key questions derived from them, like the correlation of video content type and evaluation results, relationship between static metrics (like image quality) and the long video ones, and the impact of prompt complexity in the results.

Prompt data, benchmark results, and evaluation code will be released with the paper (the first two having already been published anonymously).

### Strengths
The following are some of the key strengths of this paper:

* The problem this paper focuses on is indeed important, and increasingly so as the video generation models improve
* The taxonomy of metrics to analyze is very well selected, thoughtful, and exhaustive. Definitely a strong point for this work. Particularly the approach for event-level alignment is interesting; but overall many of these metrics are definitely insightful
* The questions stated in the Results & Analysis section are interesting and the analysis well done. Even (perhaps) more intuitive results like "methods tend to perform worse with higher semantic complexity, indicating their difficulty in understanding complex semantics" are made insightful by the quantitative approach.
* This work leverages the strengths of modern Vision-Language Models correctly
* The code being open source will be great for reproducibility, once it's made available
* Good ethics section

### Weaknesses
These are the main weaknesses of this work:

* Primarily, there isn't a comparison against human evaluation, so it's unclear how these metrics compare to reality. This is the main point that, if fixed, would greatly enhance this work.
  * As an example of how this matters, it is claimed that for the Overall Alignment metric, existing benchmarks use CLIP-based computations but for this work an approach based on MLLMs summarization followed by semantic similarity is preferable. But no quantitative comparison between both approaches is made; if there was a measure of human alignment, this claim could be verified.
  * In another point in the paper, it is concluded that "the differences across theme categories are minor, which (...) demonstrates the robustness of our evaluation framework to diverse content types". I don't think this claim is guaranteed without a human baseline.

* Similarly, the benchmark is only applied to the output of generative video models, but never to ground truth data. If the authors applied the benchmark to real videos one could know what the upper bound of quality is, and it would make all analysis and plots more complete.

Other misc weaknesses:

* All baselines are open source; including at least some proprietary baselines would increase the variety and provide some additional insights (e.g. what is the gap, if any, between open and closed source systems with respect to these metrics?)
* The source is not currently available (hopefully it can be made available before the end of the review period)

Other smaller issues regarding clarity:

* Early on, the paper introduces 7 dimensions in the "Evaluation Information Integration" section. Later on, the paper starts describing a more complex taxonomy of metrics split into 5 groups (in the "Evaluation Dimension Suite Construction" section). 2-3 pages later the proper context for the 7 dimensions (HERD) is introduced; this makes the work harder to follow.

* Some phrasing is ambiguous, particularly "existing LVG methods exhibit clear improvements in frame-level static quality" (what is the improvement compared against?). A few typos are also present (I can now recall "adpot")

### Questions
Most questions raised during the review are expressed in the Weaknesses section; a few less important considerations that I'd be interested in hearing about in the comments section of this review:

* Have the authors thought about possible extensions to video + audio?
* What insights (if any) do you derive from these results, in terms of how to improve video models?

### Soundness
2

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
This paper introduces LoCoT2V-Bench, a new benchmark specifically designed to evaluate long-form and complex text-to-video (T2V) generation , aiming to address the limitations of existing benchmarks that focus on short clips and simple prompts. The work constructs a challenging prompt suite of 240 samples with an average length of 236 words , derived from real-world YouTube videos. Its core contribution is a multi-dimensional evaluation framework that, beyond static and temporal quality, introduces novel metrics: Event-level Alignment to assess the accurate sequencing of multiple events; Content Clarity to evaluate narrative logic and thematic coherence using MLLMs ; and the Human Expectation Realization Degree (HERD) to quantify abstract aspects like emotional response and narrative flow . By testing nine representative models , the paper finds that while current methods perform adequately in basic visual fidelity , they struggle significantly with fine-grained event alignment, long-range temporal consistency, and high-level narrative adherence .

### Strengths
1. It introduces LoCoT2V-Bench, the first systematic benchmark specifically designed for evaluating long-form and complex text-to-video (LVG) generation, addressing a critical gap left by existing benchmarks that primarily focus on short clips and simple prompts.
2. It proposes a novel suite of high-level evaluation dimensions to assess abstract narrative aspects often overlooked by traditional metrics. Key innovations includes Human Expectation Realization Degree, Event-level Alignment and Content Clarity.

### Weaknesses
1. Lack of Human Validation: The paper heavily relies on MLLMs to score subjective metrics like HERD but provides no meta-evaluation to correlate these automated scores with actual human judgments, leaving the validity and reliability of these new metrics unconfirmed.
2. Limited Dynamics Assessment: The paper's "Temporal Quality" evaluation focuses primarily on frame-level dynamics (e.g., Motion Smoothness) , while overlooking higher-level dynamics such as the inter-segment and video-level dynamics defined in [1]. This raises the question of whether changing the methods for measuring dynamics would affect the conclusions.
3. Limited Model Scope: The evaluation is restricted to nine open-source models , excluding state-of-the-art (SOTA) closed-source models. This limits the representativeness of the paper's conclusions that current models universally struggle with these complex tasks .


[1] Evaluation of text-to-video generation models: A dynamics perspective.

### Questions
see weakness above. I would raise the score if the author can solve my concerns.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces a new benchmark for long video generation (30–60s). The dataset consists of 240 text prompts, obtained by captioning real videos with SOTA VLMs. The authors also propose an evaluation suite across five categories. Experiments on 9 open models suggest that current models struggle with high-level adherence and temporal consistency.

While the topic is very relevant to the community, I believe the paper needs a thorough verification of the proposed metrics as well as improved writing of the dataset and metrics. Because of these reasons, I do not think the paper is ready for publication at this time.

### Strengths
1. A new prompt set for long video generation, with longer and more complex prompts generated by VLMs and verified by humans.
2. An evaluation suite that includes previously used dimensions (e.g. quality and text alignment) as well as high-level dimensions such as emotional response.
3. Experiments with 9 different models, including analyses of entanglement between different dimensions.

### Weaknesses
1. For a paper presenting a new dataset, none of the actual prompts (and corresponding real videos) are shown in the main paper. This makes it difficult to understand the data creation and the complexity of the prompts from the main paper alone. I only saw 3 prompts in Appendix E.
2. Details and takeaways from the prompt generation are missing. It would be informative to know how each of the stages (raw captioning, self-refine, manual review) change the prompt set, so the community can decide whether and how to build on top of this pipeline.
3. A large portion of the paper is used to introduce a large number of metrics to measure different aspects of generation. Nevertheless, some of the metrics are not clearly defined, entirely referring the reader to other papers (eg. L281). While details can be read in other papers, the main body of this paper should be self-contained, providing the minimum details needed for the reader to understand the methodology.
4. Importantly, there is no correlation analysis for any of the proposed metrics to verify that they measure what the authors intended. This is important not only for newly proposed metrics but also for existing metrics, as they are applied to a different setting (long videos) than initially proposed. This casts doubts on all the results and conclusions from the paper.
5. The number of metrics is also substantial, making it hard to process the different results. While more metrics can capture more nuances, it is important for a benchmark to provide a clear (sub)set of metrics to drive progress in the area. 
6. Related to the point above, some metrics are related to each other (L412-413). Therefore, it should be verified (with human evaluations) whether they actually capture different aspects, or otherwise be removed from the evaluation suite.
7. Some of the new metrics, such as narrative flow and character development seem too ambitious for 30-60s videos. It would be great to have a human study to identify whether (or for which prompts) they are indeed meaningful.
8. It would also be great to measure how the real, ground-truth videos fare with respect to the metrics. This could be seen as some kind of upper-bound, which would increase credibility in the results.
9. Some older yet relevant related work is missing. This includes models such as Phenaki (Villegas et al., ICLR’23), which was the first model for multi-event generation, and benchmarks such as StoryBench (Bugliarello et al., NeurIPS’23) which evaluates multi-prompt generations and shows that metrics like DOVER do not correlate well with human preferences.


---
Villegas et al. Phenaki: Variable Length Video Generation From Open Domain Textual Description. ICLR’23

Bugliarello et al. StoryBench: A Multifaceted Benchmark for Continuous Story Visualization. NeurIPS’23

### Questions
1. It would be nice to define early on in the paper what “long” video generation means (i.e., 30-60s).
2. You should also leave a whitespace between a word and its following citation: word (citation) rather than word(citation).
3. For TVA metrics, why do you compare the generated caption against the raw caption? Current SOTA methods for image–text and video–text alignment employ VQA-based metrics (Wiles et al., ICLR’25).
4. It would also be great (yet optional) if you could evaluate a SOTA model, as examples in Appendix E show high aesthetic scores for videos that are clearly much worse than SOTA, casting doubts on whether the proposed metrics will be able to capture differences in future models.

---
Wiles et al. Revisiting Text-to-Image Evaluation with Gecko: On metrics, prompts, and human ratings. ICLR’25

### Soundness
2

### Presentation
2

### Contribution
2
