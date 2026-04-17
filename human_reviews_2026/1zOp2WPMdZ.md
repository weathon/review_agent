# IWR-Bench: Can LVLMs reconstruct interactive webpage from a user interaction video?

- Decision: Accept (Poster)
- Scores: 4, 6, 4, 6

## Abstract
The webpage-to-code task requires models to understand visual representations of webpages and generate corresponding code.
However, existing benchmarks primarily focus on static screenshot-to-code tasks, thereby overlooking the dynamic interactions fundamental to real-world web applications.
To address this limitation, this paper introduces IWR-Bench, a novel benchmark for evaluating the capabilities of Large Vision-Language Models (LVLMs) in interactive webpage reconstruction from video.
IWR-Bench comprises 113 meticulously curated tasks from 100 real-world websites, with 1,001 actions and featuring diverse interaction complexities (e.g., web games), visual styles, and domains.
Aligning with standard web development practices, each task includes not only user interaction videos but also all crawled static assets (e.g., images, videos).
This benchmark evaluates models on two fundamental challenges: comprehensive multi-modal reasoning to infer interaction logic from video and assets, and advanced code generation to translate this logic into functional code.
An agent-as-a-judge framework with a comprehensive metric system automatically assesses the functional correctness and visual fidelity of generated webpages.
Extensive experiments on 28 LVLMs reveal a significant challenge: the best model achieves an overall score of only 36.35\%, as functional correctness (24.39\% IFS) lags significantly behind visual fidelity (64.25\% VFS).
These results highlight critical limitations in current models' ability to reason about temporal dynamics and synthesize event-driven logic, establishing IWR-Bench as a challenging frontier for vision-language research.
The benchmark and evaluation code will be made publicly available.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper shifts evaluation from static mimicry to working webpage from interaction videos, revealing a wide functionality gap in today’s LVLMs and offering a rigorous, automated protocol to measure progress.

### Strengths
1. Shifts the problem from static screenshot‑to‑code to interactive webpage reconstruction from video;
2. First benchmark to combine videos and ground‑truth action trajectories for web reconstruction, filling the gap between static page benchmarks and general video QA.

### Weaknesses
1. With only 113 tasks, the benchmark is relatively small compared to other vision-language benchmarks;
2. The authors set w=0.5 for VFS and α=0.7 for the Final Score “based on validation studies,” but details are limited in the paper;
3. A single MLLM (Gemini‑2.5‑Pro) is used in both logical asserts verification and providing the visual score, which risks bias and errors.

### Questions
1. Task source: the paper mentions "Experts in web development", diversity balancing, and the three-step annotation. However, there is no any detail, such as concrete criteria and inter-annotator agreement. Would the authors supply more details?
2. VFS: VFS is computed only on checkpoints that are both reached and flagged, so that the missing states never contribute. Models that fail early avoid penalties on hard visual states, with higher VFS. please clarify the metric in details.
3. Weights: VFS blends the scores with w=0.5; the Final Score uses α=0.7. The paper cites a validation study but provides few in‑text sensitivity analyses.
4. Single‑judge bias: The same MLLM (Gemini‑2.5‑Pro) is used to verify logical assertions and provide the high‑level visual score, and this model also appears on the leaderboard. Were the results verified by humans? i am concern its reproducibility and stability. A single MLLM pass determines assertions, which may exhibits sampling variance. The authors should run multiple independent judging passes and aggregate the results.

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
The authors introduce the benchmark IWR-Bench, a benchmark for video to code generation. In this benchmark, the multimodal large language model (MLLM) receives a video of a user interacting with a website, along with anonymized assets, and must generate code that reproduces the same functionality. They also propose a framework for evaluating the generated code using two metrics: the Interaction Functionality Score (IFS) for functional accuracy and the Visual Fidelity Score (VFS) for visual correctness. The overall performance is reported as a weighted combination of these two metrics. The authors evaluate 28 MLLMs, including GPT-5, Sonnet-4, and Gemini-2.5-Pro, showing that current models still struggle with this challenging task.

### Strengths
- The paper defines a novel and interesting task, video-2-code, which is both challenging and potentially useful for practical applications.

- The paper is well-written and provides detailed descriptions of data generation, evaluation metrics, and experimental setup.

### Weaknesses
- Missing citation: the benchmark PairBench [1] investigates when an MLLM can act as a reliable judge and could be used to identify an agent-as-a-judge suitable for this task.

- Minor typos, such as inverted quotation marks.

- The paper does not report the average, maximum, and minimum number of assets per task, which could help analyze how performance varies with the number of input assets. A results table showing model performance grouped by asset count would make this analysis clearer and more informative.

[1] PairBench: Are Vision-Language Models Reliable at Comparing What They See?

### Questions
- What are the resolutions of the videos used in the benchmark?

- How much does it cost to run the benchmark on proprietary models versus open-source models, given that the agent-as-a-LLM is Gemini-2.5-Pro? This would help assess the practicality of the benchmark for everyday use.

### Soundness
4

### Presentation
4

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
* The paper introduces IWR-Bench, a new benchmark for evaluating vision-language models on the task of reconstructing interactive webpages from videos of user interactions. The benchmark contains 113 tasks collected from real websites, covering 1,001 user actions with diverse layouts, domains, and interaction complexities. Each task includes a user interaction video, crawled static assets, action trajectories, and checkpoint screenshots.
* The valuation is performed using an agent-as-judge method. Twenty-eight LVLMs are included for the evaluation: GPT-5, Claude, Gemini, etc and the top-performing model only obtain a score of 36.35.

### Strengths
* The paper provides a comprehensive and well-structured dataset sourced from real-world websites, ensuring realism and diversity across domains, visual styles, and interaction types.
* Extensive empirical study across 28 LVLMs offers valuable baseline results and comparative insights.

### Weaknesses
* The difference between this paper and prior work such as Interaction2Code is not substantial. The main distinction lies in the number of interaction steps: Interaction2Code handles single-step interactions, whereas IWR-Bench introduces multi-step sequences. This difference, while incremental, limits the benchmark’s overall novelty and impact.

* For experiments, as mentioned around Line 090, the challenges of interactive webpage reconstruction lie in two aspects: multi-modal perception and code generation. It would be valuable to include pipeline-based baselines. A natural idea would be to use a vision-language model to describe each step of the interaction video in terms of states and action assertions, and then feed this detailed description as a caption to a code-specialized model to generate the webpage code.

### Questions
* What is the correlation coefficient between the metric results on the benchmark proposed in this paper and the metric results on Interaction2Code?

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
3

### Summary
This work introduces IWR‑Bench, a benchmark that asks multimodal models to watch a video of a real website in use, along with a sheet of actual page assets, and then generate HTML/CSS/JS code that reproduces both the look and the interaction logic shown in the video. To curate IWR‑Bench, the author asks web‑dev experts to propose a large batch of candidate tasks, following which human annotators perform the workflow on the live website and record interaction videos. In parallel, they automatically collect all static assets used on the page, such as images and icons. Finally, 113 tasks are curated in IWR‑Bench. The generated websites are evaluated by replaying the workflow, and an LLM is used to score the interactive functionality and visual fidelity. The experiment results show that the SOTA LLM struggles to achieve reasonable performance. In particular, synthesizing correct interaction logic is the main bottleneck.

### Strengths
1. The task of synthesizing website code directly from user‑interaction video is both novel and interesting, and likely to remain challenging for multimodal LLMs in the near term.

2. Dataset curation is careful and well‑documented, with cross‑annotator verification and asset checks. Therefore, the resource should be useful to the community.

3. It's an interesting finding that LLMs mainly struggle with interactive functionality

### Weaknesses
1. It's understandable that building such a dataset is labor-intensive and time-consuming. But a test set with only 113 tasks might be too small-scale.

2. The evaluation metric might be too binary since a minor defect (e.g., a button a few pixels off that breaks a click) is penalized the same as a significant logic error (totally different UI element). 

3. Although the proposed task is interesting, practical and non‑malicious value propositions are under‑argued. The new task would be more well-motivated if the author explained why reconstructing an existing site.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
