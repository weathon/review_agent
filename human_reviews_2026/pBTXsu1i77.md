# VLM-SubtleBench: How Far Are VLMs from Human-Level Subtle Comparative Reasoning?

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 8, 6, 6, 2

## Abstract
The ability to distinguish subtle differences between visually similar images is essential for diverse domains such as industrial anomaly detection, medical imaging, and aerial surveillance. While comparative reasoning benchmarks for vision-language models (VLMs) have recently emerged, they primarily focus on images with large, salient differences and fail to capture the nuanced reasoning required for real-world applications. In this work, we introduce **VLM-SubtleBench**, a benchmark designed to evaluate VLMs on *subtle comparative reasoning*. Our benchmark covers ten difference types—Attribute, State, Emotion, Temporal, Spatial, Existence, Quantity, Quality, Viewpoint, and Action—and curate paired question–image sets reflecting these fine-grained variations. Unlike prior benchmarks restricted to natural image datasets, our benchmark spans diverse domains, including industrial, aerial, and medical imagery. Through extensive evaluation of both proprietary and open-source VLMs, we reveal systematic gaps between model and human performance across difference types and domains, and provide controlled analyses highlighting where VLMs’ reasoning sharply deteriorates. Together, our benchmark and findings establish a foundation for advancing VLMs toward human-level comparative reasoning.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper introduces a new benchmark designed to evaluate whether multimodal LLMs can distinguish subtle differences between similar images (i.e., perform subtle comparative reasoning). The benchmark is comprehensive: it contains 11.7K triplets of image pairs, questions, and answers, covering ten types of differences (e.g., attribute or state) across five image domains, including natural, industrial, and medical. The authors evaluate multiple recent MLLMs, both open-source and proprietary, on this benchmark and show that subtle comparative reasoning remains a significant challenge for current models.

### Strengths
- The benchmark is comprehensive, encompassing a large number of data points as well as diverse visual types and domains.

- The paper identifies the remaining limitations of recent MLLMs, providing valuable insights into where the research community should focus to further improve these models.

- The paper is well-written and easy to follow.

### Weaknesses
- A more detailed analysis is needed. e.g., why do MLLMs struggle more with certain types of comparisons? Why do some models perform better than others?

- The paper currently reports results but lacks discussion or suggestions on how to improve subtle comparative reasoning in MLLMs.

### Questions
- Similar to the study (Table 3) in MLLM-CompBench, what would happen if the models were first asked to analyze two images separately and then compare them using a purely language-based question, instead of being given both images simultaneously?

- Did annotators label all the data? How many annotators were involved in building this benchmark, and what were their backgrounds?

- It seems that only a test set is provided. Do you also have training or validation sets? It would be interesting to see whether fine-tuning on such data could improve performance.

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
The paper introduces VLM-SubtleBench, a benchmark targeting subtle comparative reasoning across 10 difference types (Attribute, State, Emotion, Temporal, Spatial, Existence, Quantity, Quality, Viewpoint, Action) and 6 domains (Natural, Game, Industry, Aerial, Medical, Synthetic), comprising 11.7k (image-pair, question, answer) triplets. It emphasizes minimally different pairs (e.g., pairs with high DINOv3 similarity) and augments MCQ with a captioning track. Systematic studies span open-source and proprietary VLMs, prompt/fusion strategies, and controlled synthetic stress tests. Human accuracy is ~95.5%, whereas the best proprietary model remains well below this, with particular weaknesses in temporal/spatial/viewpoint categories.

### Strengths
- Substantive contribution via task definition & data curation. Clearly formalizing subtle comparative reasoning as an evaluation target and curating a benchmark dataset with transparent collection/validation protocols is, by itself, a meaningful research contribution.

- Breadth + diagnostics. Coverage of 10 difference types and 6 domains with controlled synthetic factors (e.g., brightness deltas, object size, translation, object count) supports failure-mode analysis rather than aggregate scores only.

### Weaknesses
- Data generation dependencies. Some Attribute pairs are created with Gemini-2.5 flash image preview (“nano-banana”) editing; Medical questions are refined by gpt-4o. This can introduce stylistic artifacts or distribution shifts that confound evaluation unless carefully audited. Please quantify any such effects (e.g., edited vs. non-edited subsets).

- The paper identifies notable gaps in temporal/spatial/viewpoint (e.g., stable accuracy requiring ~160 px camera translation in synthetic tests), but a more explicit reporting of difficulty curves on natural data would help establish ecological validity.

### Questions
- Your Figure 5 synthetic-control study probes failure modes by manipulating low-level factors (e.g., brightness/scale/count/translation). Could you extend this with a color-sensitivity axis inspired by VLM’s Eye Examination [1], which reports consistent green insensitivity across VLMs, to refine the Attribute subset? Concretely, consider hue sweeps with a green vs. non-green contrast and ΔE/brightness steps, and report performance as a function of hue as well as factor interactions (color × size/count/translation). This would test whether known perceptual color deficits compound subtle comparative reasoning failures.

- You clearly motivate the benchmark with high-stakes domains (industrial anomaly detection, medical imaging, aerial surveillance). Could you provide evidence of transfer, e.g., correlations between SubtleBench category scores and downstream metrics on representative datasets in those domains, or a small-scale finetuning study showing measurable gains in real applications?

[1] VLM’s Eye Examination: Instruct and Inspect
Visual Competency of Vision Language Models, https://arxiv.org/pdf/2409.14759

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes a benchmark to evaluate how well models are at discerning subtle changes between two images in form of visual question answering and captioning. The dataset covers instances from multiple domains (such as game, industrial, medical, etc) and has ten different types of differences (such as temporal, spatial, emotion, etc). Evaluation results are comprehensive including multiple popular open-source and proprietary models such as GPT-5, o3, Claude-sonnet-4, Gemini-2.4-pro along with different prompting strategies. Human evaluation is also conducted and current results indicate a gap between human performance (avg. performance 95.5) and current frontier models (best 77.8 avg performance).

### Strengths
1. The paper focuses on an interesting problem setup of fine-grained changes between two images, and it is interesting how current frontier models struggle at these tasks.
2. The paper adequately describes the dataset construction process, model evaluation setup, and experimental results. Overall, it is well written.
3. The evaluation process studies multiple factors that can influence model performance -- such as how to combine the two images when feeding images, different prompting strategies and impact on model performance with diff. controllable percentage of change b/w 2 images.

### Weaknesses
1. This task of subtle difference changes b/w two images has been previously explored in works such as Spot-the-Diff [1], Img-Diff [2] and MLLM-CompBench [3] as noted by authors. The primary novelty seems to be expansion to multiple domains, more question types and combination of multiple choice questions and captioning in a single benchmark. In this regard, novelty is a bit limited.

2. There can be further baselines/prompting strategies considered such as:
- Calculating regions of interest from the subtraction of 2 images, and then highlighting these regions in the 2 input images (through simple bounding boxes or masks) and feed them to VLM stating regions of interest are highlighted.
- 2-step reasoning process -- first ask the VLM to describe differences b/w the 2 images with respect to answering the question, and then feed this output in addition to the 2 images and original question.

Relatively minor:
3. It would also be interesting to study whether this task can simply be solved by training models on a mix of synthetic and real samples as the task itself might be out-of-distribution. But I understand this may be out of scope for current paper.

### Questions
Please see weaknesses.

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
3

### Summary
In this paper, the author presented VLM-SubtleBench, a benchmark to evaluate subtle comparative reasoning in VLMs across 10 difference types and 6 visual domains. The benchmark contains more than 11k image-pair QA samples and 1k human-annotated difference captions. In this paper, the authors also claim that existing such benchmarks focus on salient differences and lack domain diversity. They evaluated several current VLMs on the benchmark and found that they struggle with subtle differences compared to human performance.

### Strengths
- In this paper, they introduced an increasingly relevant capability: subtle visual comparison between images, across multiple domains. It contains various difference types (attribute, temporal, viewpoint, etc.) and datasets beyond natural images (industrial, medical, etc.).
- It also has a mix of real data and synthetic setups, to show controlled evaluation capabilities.
- The paper is easy to read, though the figures could use more text to make them clearer.

### Weaknesses
- The dataset seems to be largely a small increment of prior multi-image VLM benchmarks like MLLM-CompBench, ReMI. The claim of novelty in subtlety is only partially convincing.  Subtle differences are defined via embedding cosine similarity (DINOv3), but this does not necessarily guarantee perceptual or semantic subtlety.
- In Figure 3, it can be seen that when the catcher moved, the other player moved as well. As the paper claims to be fine-grained, I would be interested to know how the authors ensured that in the video, only the object/person in question moved, not the other parts.
- Repetition of the same datasets across categories. MVTEC-AD reused across “attribute” and “state” categories. 
- “Overlap” and “subtraction” images are described vaguely. It is unclear to me whether pixel-level overlay vs feature-space subtraction is used. This step needs technical clarity and justification. [line 354]
- No evaluation on two-image native models (e.g., LLaVA Next, Clip).
- What is domain domain-specific performance difference for each model in the benchmark? Can the author provide some insights on: are the methods bad at medical or really good in synthetic for every type?
- Typos: line 242, archange to are

### Questions
- how subtlety was maintained for the types from frames were takes from videos, to be sure that only the object in question moved.
- How do authors ensure repetition of the same datasets across categories represents distinct types [weakness 3]?
- Insights about domain-specific evaluation and evaluating multi-image models on the benchmark [weakness 5].

### Soundness
2

### Presentation
3

### Contribution
2
