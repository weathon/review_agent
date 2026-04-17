# IV-Bench: A Benchmark for Image-Grounded Video Perception and Reasoning in Multimodal LLMs

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 8

## Abstract
Current benchmarks for Multimodal Large Language Models (MLLMs) predominantly rely on text-only queries, overlooking the essential role of images as visual context for enhancing video comprehension and facilitating natural human-AI interaction. To bridge this gap, we introduce \textbf{IV-Bench}, the first comprehensive benchmark for evaluating MLLMs on Image-Grounded Video Perception and Reasoning. IV-Bench comprises 966 videos paired with 2,560 meticulously annotated image-text queries across 13 tasks (7 perception and 6 reasoning tasks) spanning 5 distinct categories. We extensively evaluate state-of-the-art MLLMs, including open-source models (e.g., InternVL2.5, Qwen2.5-VL) and closed-source models (e.g., GPT-4o, Gemini2.0 series), revealing substantial performance gaps, with the best-performing model achieving only 28.9\% accuracy. Ablation studies demonstrate that incorporating images significantly enhances video understanding and highlight key model design factors influencing performance. Our findings provide valuable insights and guidance for future research. The code and dataset are available at \url{https://github.com/multimodal-art-projection/IV-Bench}.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents IV-Bench, the first large-scale benchmark specifically designed to assess image-grounded video reasoning and understanding in multimodal large language models (MLLMs). IV-Bench comprises 966 videos paired with 2,560 manually annotated image–text queries, covering 13 tasks across 5 diverse domains. Through an extensive evaluation of state-of-the-art MLLMs, the study finds that current models achieve no more than 28.9% accuracy, revealing substantial performance gaps.

### Strengths
1. The paper tackles an important and previously overlooked problem—image-grounded video understanding—by establishing the first comprehensive benchmark in this area, thereby filling a crucial gap in existing multimodal evaluation benchmarks.

2. The paper highlights the limitations of current MLLMs in handling such tasks, offering clear guidance and future directions for advancing MLLM capabilities.

### Weaknesses
The analyses in the paper are not sufficiently deep or insightful. For example, the reported phenomenon of "moderate performance gains with larger models" lacks both theoretical grounding and concrete case studies. Assertions such as "increasing model size primarily enhances memorization and shallow pattern recognition rather than reasoning ability" are not adequately supported by evidence. Similar shortcomings appear throughout other analytical sections. Moreover, the paper fails to propose practical solutions or methodological advances to address the identified challenges.

### Questions
I have no questions.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes a benchmark named IV-Bench that requires the models to perceive and reason the contents from a reference image which is not involved in the input video. The dataset is manually labored and elaborated 7 main findings from evaluating 28 MLLMs.

### Strengths
- The curation of the dataset does not rely on automated annotation tools (e.g., GPT-4V) and ensures high quality by manually annotations.
- Assessing model capability to utilize reference images is a novel motivation. Also, excellent presentation with high overall readability and nice figures.
- Impressively extensive evaluation (28 MLLMs) is reported, providing actionable diagnostics. Also, the performance gap between humans and models is huge.

### Weaknesses
- The main concern is that the findings are not surprising compared to the quality of the dataset. Comparisons across the model capacity, video fps, and resolution are trivial according to the scaling law [1].
- Although it is not necessary, the analysis lacks experiments on meta-data (e.g., subtitles).
- All tasks are based on multi-choice questions, not requiring open-ended questions even for the “reasoning” category.
- Inter-annotator agreement and quality-control rejection rates on both rounds are missing.
- The paper rigorously describes the model performance by model families (e.g., line 349 to 360). I believe that reporting this with averaged gain by model scaling and adding more verifications and allocating more descriptions of author’s findings can make this paper much more stronger.

I will increase the score once current concerns are addressed.

### Questions
- How do RL-trained models perform on this benchmark? Are there any inductive bias observed from visual encoders? Are there any underlying biases or trends from the training set? These points are revealed that these two points matter [2,3].
- Authors analyzed the order of video and image matters. Did the ordering of text inputs change performance?
- I believe that analysis on attention score can somehow verify the finding in Section 4.3 (line 400). How are the attention scores distributed towards image tokens?
- How can the evaluated models have forgetting issues (line 401) in the input sequence? Different from LSTM models that sequentially feed tokens to update hidden state, Transformers architecture processes the input tokens in parallel rather than sequentially. Isn’t the “forgetting” happens only when the input tokens are truncated according to the size of the context window? The size of the context window should be reported to make this claim concrete. In other words, how many previous tokens can the models condition on.

References

[1] Fang et al. MMBench-Video: A long-form multi-shot benchmark for holistic video understanding. NeurIPS D&B Track 2024.

[2] Wang et al. VideoHallucer: Evaluating Intrinsic and Extrinsic Hallucinations in Large Video-Language Models. arxiv:2406.16338

[3] Li et al. Vidhalluc: Evaluating Temporal Hallucinations in Multimodal Large Language Models for Video Understanding. CVPR 2025.

### Soundness
4

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces IV-Bench, the first comprehensive benchmark designed specifically for evaluating Multimodal Large Language Models (MLLMs) on image-grounded video perception and reasoning tasks. The benchmark comprises 966 videos and 2,560 meticulously annotated image-text queries, spanning 13 distinct tasks (7 perception and 6 reasoning) across 5 video categories. The authors conduct an extensive evaluation of 28 state-of-the-art MLLMs, revealing that even the best-performing model (Qwen2.5-VL-72B) achieves only 28.9% overall accuracy, which is significantly lower than human performance (88.8%). Furthermore, through ablation studies, the authors analyze the impact of factors such as the order of image input, number of video frames, and resolution on model performance, providing valuable insights for future model design.

### Strengths
1. IV-Bench is the first benchmark specifically dedicated to image-grounded video understanding tasks. It effectively addresses the limitation of existing video benchmarks that rely solely on text queries, thereby advancing the field of multimodal reasoning evaluation.
2.The benchmark's high quality and validity are ensured through a two-round quality control process, the use of externally sourced images, and diverse task design. The inclusion of "effective distractors" is particularly noteworthy, as it forces models to rely on the image information.
3. The systematic evaluation of 28 models thoroughly exposes the significant shortcomings of current MLLMs on this task. The ablation studies offer practical design recommendations, such as the finding that placing the image after the video frames yields better performance.

### Weaknesses
1.Although covering multiple categories, the total of 966 videos is smaller than some existing video understanding benchmarks (e.g., Video-Bench with 5,917 videos), which might affect the benchmark's broad representativeness.
2.The current work focuses only on the triplet input of image-text-video. It does not explore more complex multimodal grounding signals, such as audio, multiple images, or dynamic image sequences (e.g., GIFs).
3.While emphasizing "image-grounding," the benchmark may not fully account for whether a single static image is sufficient to represent dynamic changes in a video. Some tasks might still lean towards static matching rather than genuine spatio-temporal reasoning.

### Questions
1.	During the construction of IV-Bench, did you consider incorporating dynamic image sequences (e.g., GIFs or short video clips) as the grounding signal to better simulate real-world scenarios where users provide visual context?
2.	Regarding the particularly poor performance on tasks like "Temporal Reasoning," have you conducted further analysis into the root causes? Is it related to the models' ability to understand long videos or their inherent temporal modeling mechanisms?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
IV-Bench is one of the first benchmark for image-grounded video perception and reasoning in MLLMs, featuring 966 videos and 2,560 external image-text queries across 13 diverse tasks. Unlike other benchmarks, all queries use external images as anchors, increasing real-world complexity. Evaluations on 28 leading MLLMs show performance is much lower than humans, especially in reasoning. The paper analyzes model size, reasoning methods, and token allocation, and both dataset and code are open-source.

### Strengths
- This is one of the first systematic benchmark proposal for image-grounded video perception and reasoning, filling a notable gap in existing evaluation ecosystems.

- The benchmark features high data quality and a robust evaluation framework. All images are sourced externally, preventing information leakage and accurately reflecting the difficulty of cross-source visual grounding in complex tasks such as search and retrieval, which strongly enhances the validity of the assessment.

- Experimental analyses are thorough and the findings are insightful for follow-up research. The paper details the effects of image token order, scale, resolution, and frame count on model performance, and clearly points out the extremely limited reasoning capabilities of current MLLMs in image-grounded video scenarios. It also shows model scaling has minimal impact on reasoning, which offers clear direction for future advances.

### Weaknesses
- Some tasks (e.g., Instruction Understanding, Summary) feature only weak connections to the visual “grounding,” where images act in a mainly auxiliary role and do not fully embody the core definition of image grounding.

- The paper lacks granular error analysis. It reports only accuracy without dissecting which sub-tasks exhibit systematic failure (e.g., universal shortcomings in temporal reasoning or attribute change), and does not offer failure case examples. It is suggested to add error type breakdowns (e.g., missed visual cues, text comprehension errors, temporal confusion) and provide typical failed cases versus human labels in the appendix.

- The discussion around the semantic gap and selection logic for external images should be strengthened. Although the necessity and realism of using external images is emphasized, there is no structured description or case analysis of their diversity/difficulty. A quantitative contrast between tasks with highly similar external images (near frame extraction) versus high-gap scenarios should be considered.

### Questions
1. On subjective scoring consistency and ground-truth diversity: For reasoning tasks with subjective or diverse answers, how is human label consistency and answer distribution measured? How are multi-answer questions evaluated?

2. On the irreplaceability of images: Has image replacement testing been conducted across all task categories? If an image is replaced by another with similar semantics but different visuals, does the answer change? If not, how can it be demonstrated that the image truly “grounds” the reasoning process?

3. On prompt strategy: In Table 2, do all models use the optimal prompt order (image after video)? If not, have you considered rerunning the experiments for fairness?

### Soundness
3

### Presentation
3

### Contribution
4
