# MIMIC-Bench: Exploring the User-Like Thinking and Mimicking Capabilities of Multimodal Large Language Models

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 8, 4, 6, 2

## Abstract
The rapid advancement of multimodal large language models (MLLMs) has greatly prompted the video interpretation task, and numerous works have been proposed to explore and benchmark the cognition and basic visual reasoning capabilities of MLLMs.  However, practical applications on social media platforms demand MLLMs that can emulate user-like thinking and behavior when interpreting user-generated videos, which has been rarely studied in current research. To bridge the gap and get closer to general practical artificial intelligence (AI), we first construct **MIMIC-Data**, a large-scale dataset containing 150K+ user-shared videos with corresponding information including captions, tags, comments, *etc.*  Then, we present **MIMIC-Bench**, a large-scale benchmark building upon curated 4,000 user-shared videos from MIMIC-Data, which is designed to evaluate user-like thinking and mimicking capabilities of MLLMs in real-world video contexts.  MIMIC-Bench not only supports user-like thinking challenges including creator intent, user feedback interpretation, *etc.*, but also introduces a novel comment imitation task to assess whether MLLMs can generate human-like responses to video content.  Based on MIMIC-Data and MIMIC-Bench, we develop **MIMIC-Chat**, which integrates spatial and temporal features into a large language model, and finetunes the model to perform user-like thinking and mimicking tasks. Extensive experiments conducted based on 24 existing MLLMs and our MIMIC-Chat model show that current MLLMs exhibit limited capabilities to perform human-like thinking and responses, and MIMIC-Chat performs better to some extent.  We hope MIMIC-Bench can contribute to the advancement of human-aligned video understanding in the multi-modal era. The MIMIC-Data, MIMIC-Bench, and MIMIC-Chat will be released upon the publication.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
To emulate user-like thinking and behaviour, the paper introduces MIMIC-Bench, a large-scale benchmark for evaluating human-like reasoning and social cognition in multimodal large language models (MLLMs). Built on MIMIC-Data—a dataset of over 150K user-generated videos with rich metadata such as titles, tags, and comments—MIMIC-Bench emphasizes how people think, feel, and react to videos, rather than only recognizing visual content.  It includes two main tasks: (1) User-like Thinking, covering creator intent, content attributes, and user interactions; and (2) User-like Mimicking, assessing models’ ability to generate human-like comments. The authors also develop MIMIC-Chat, a multimodal model trained on MIMIC-Data to jointly learn video semantics and human-style reasoning. Experiments on 24 state-of-the-art MLLMs show limited performance, while MIMIC-Chat demonstrates improved human-aligned understanding.

### Strengths
1. The explanation of the problem and experimental protocol is clear, and the figures and tables are well-organized, effectively helping readers understand both the problem and the model architecture.
2. The task is interesting and has clear real-world applications. Given a video, the model can generate human-like emotional responses and categorize the content into relevant tags. Such capabilities are essential for video understanding, predicting audience feedback on movies or online videos, and improving video recommendation systems.
3. The paper presents comprehensive experiments comparing the performance of multiple models on the proposed dataset.
4. For this task, the authors also propose MIMIC-Chat to estimate human responses. Based on the results, it serves as a strong and effective baseline.

### Weaknesses
1.  There are a few formatting issues—for example, in Lines 83–84, some words are incorrectly bolded, and in Lines 261–264, two metrics are presented but only one is highlighted in bold.
2. The evaluation procedure for the User-like Mimicking task is unclear. Is the generated response from each of the 24 MLLMs individually compared against the real user comments?
3. It would be helpful to include further insights and analysis based on the current models’ results—for example, whether the lower performance stems from limited perceptual ability or insufficient common-sense knowledge.

### Questions
None

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents **MIMIC-Bench**, a large-scale benchmark for human-aligned video understanding, together with **MIMIC-Data**, a supporting dataset, and **MIMIC-Chat**, a model fine-tuned for human-like reasoning and comment generation. The benchmark consists of two complementary tracks—User-like Thinking Tasks (structured reasoning based on metadata) and User-like Mimicking Tasks (comment imitation). The authors evaluate 24 multimodal large language models (MLLMs), including both open-source and proprietary systems, and provide human evaluations for reference.

### Strengths
1. The creation of MIMIC-Data and MIMIC-Bench fills a gap in evaluating human-aligned reasoning in video understanding, moving beyond visual recognition to social and cognitive aspects.
2. The benchmark incorporates structured reasoning tasks and a creative comment imitation task, encouraging evaluation of both perceptual and cognitive understanding.
3. The dual-branch spatiotemporal encoding and unified instruction format in MIMIC-Chat demonstrate careful design and solid engineering implementation.
4. Extensive experiments on a large set of baselines and human participants give a comprehensive picture of current model capabilities.

### Weaknesses
1. Since MIMIC-Chat is trained on MIMIC-Data, which shares sources with the benchmark,  there is potential concern regarding content overlap or stylistic memorization.
2. MIMIC-Bench focuses on highly engaging videos (top 2–5%), which might bias the content toward certain social or cultural styles. The work would benefit from analysis or experiments on more diverse or out-of-domain video data.
3. While the Comment Imitation task benefits from human judgment, this can lead to variability in evaluation. Including more information on the annotation process, strategies for mitigating bias, and possible alternative metrics would help improve reproducibility and clarity.

### Questions
1. Could the authors elaborate on whether there is any potential overlap between MIMIC-Data (used for training) and MIMIC-Bench (used for evaluation)? If so, how is such overlap avoided or controlled to ensure fair assessment?

2. It would be helpful if the authors could describe the human annotation process for comment imitation in greater detail—e.g., annotation interface, annotator expertise, and disagreement resolution.

### Soundness
2

### Presentation
2

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
This paper addresses the gap where existing MLLM benchmarks primarily focus on basic visual reasoning and neglect the ability of models to emulate user-like thinking when interpreting videos and their associated social context on platforms like social media. The authors propose a new dataset, a benchmark, and a specialized model. Experiments reveal MLLMs' limited capabilities in human-aligned thinking and responses.

### Strengths
- This paper proposes a fundamentally new evaluation paradigm, shifting the focus from basic visual reasoning to user-like thinking and mimicking capabilities for User-Generated Content
- This paper is well-written and easy to understand.

### Weaknesses
- The evaluation paradigm for the Mimicking Task (Comment Generation) suffers from poor scalability. Any new model introduced to the benchmark would necessitate re-evaluation, making comparative assessment difficult. Furthermore, the dimensions used for evaluation are insufficient.

### Questions
Please refer to weaknesses.

### Soundness
2

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
This paper introduces MIMIC-Data, a large-scale dataset of over 150K user-shared videos with rich contextual metadata such as captions, tags, and comments. Building upon it, the authors present MIMIC-Bench, a benchmark of 4K videos designed to evaluate MLLMs’ ability to mimic human-like thinking and responses in realistic social media contexts. They further develop MIMIC-Chat, a model that integrates spatiotemporal understanding with fine-tuning for user-like reasoning and comment imitation. Experiments on 24 existing MLLMs show limited human-like behavior, while MIMIC-Chat demonstrates improved alignment with user-style cognition and response.

### Strengths
1. The paper nicely fills a research gap by being the first to focus on human-like thinking and imitation of MLLMs in user-generated video contexts. The proposed MIMIC-Data is large-scale and well-curated, offering rich multimodal metadata that will benefit future research. The MIMIC-Bench benchmark is also valuable for assessing real-world user-aligned reasoning.

2. The proposed MIMIC-Chat model is well-motivated — it integrates spatiotemporal features into an LLM and uses instruction tuning to balance structured reasoning with open-ended generation.

3. The experiments are thorough and convincing, comparing 24 state-of-the-art MLLMs (both open and closed-source) and incorporating human annotations as an upper bound, which strengthens the reliability of the results.

### Weaknesses
1. My main concern is that the core contribution — MIMIC-Bench — feels too simple. In Tables 1 and 2, existing models already achieve very high scores, and the benchmark cases shown (e.g., in Fig. 1) lack strong distractors, making the evaluation less convincing.

2. The paper does not report the video length distribution in MIMIC-Data. If short clips dominate, it may fail to assess long-video understanding. Also, the computational cost (e.g., GPU memory, inference speed) of MIMIC-Chat is not discussed, which limits understanding of its practical deployability.

3. The paper does not address potential overfitting — whether MIMIC-Chat overfits MIMIC-Data or whether MIMIC-Data and MIMIC-Bench are too similar. Moreover, results on other benchmarks are missing, leaving generalization unverified.

### Questions
Please refer to the Weaknesses section — the most critical point is to increase the difficulty of MIMIC-Bench to better match the rapidly improving capabilities of current VLMs.

### Soundness
3

### Presentation
2

### Contribution
2
