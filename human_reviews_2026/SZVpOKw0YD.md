# A.I.R.: Enabling Adaptive, Iterative, and Reasoning-based Frame Selection For Video Question Answering

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 4

## Abstract
Effectively applying Vision-Language Models (VLMs) to Video Question Answering (VideoQA) hinges on selecting a concise yet comprehensive set of frames, as processing entire videos is computationally infeasible. However, current frame selection methods face a critical trade-off: approaches relying on lightweight similarity models, such as CLIP, often fail to capture the nuances of complex queries, resulting in inaccurate similarity scores that cannot reflect the authentic query-frame relevance, which further undermines frame selection. Meanwhile, methods that leverage a VLM for deeper analysis achieve higher accuracy but incur prohibitive computational costs. To address these limitations, we propose A.I.R., a training-free approach for Adaptive, Iterative, and Reasoning-based frame selection. We leverage a powerful VLM to perform deep, semantic analysis on complex queries, and this analysis is deployed within a cost-effective iterative loop that processes only a small batch of the most high-potential frames at a time. Extensive experiments on various VideoQA benchmarks demonstrate that our approach outperforms existing frame selection methods, significantly boosts the performance of the foundation VLM, and achieves substantial gains in computational efficiency over other VLM-based techniques.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper proposes A.I.R., a training-free Adaptive, Iterative, and Reasoning-based framework for frame selection in VideoQA. It first adaptively samples query-relevant events based on similarity distributions, then iteratively refines frame selection using VLM reasoning within a cost-effective loop. Experiments show superior efficiency and accuracy across benchmarks.

### Strengths
The paper is original in introducing a reasoning-based, training-free iterative framework for frame selection, addressing the cost–accuracy trade-off in VideoQA. The methodology is solid and well-motivated, combining adaptive sampling with iterative refinement. Results show significant efficiency gains, and the paper is clear and impactful, improving VLM applicability to real-world video understanding.

### Weaknesses
The paper lacks runtime comparisons with existing frame selection methods, needs stronger justification of iterative VLM selection, and does not discuss generalization to other video-language tasks.

### Questions
1. The proposed rule-based frame selection achieves higher recognition accuracy, but the paper does not compare computational efficiency with existing frame selection methods (e.g., frame selection time).

2. The key contribution is the VLM-based iterative frame selection algorithm, yet the correctness of the iterative process requires further clarification. Additionally, it is unclear how performance compares if all frames are selected in a single, non-iterative pass using the same VLM under the maximum budget.

3. Essentially, the paper introduces another key-frame selection method for multimodal (video-language) understanding to improve accuracy. It remains unclear whether this approach generalizes to other video-language tasks, such as video-language retrieval.

### Soundness
4

### Presentation
3

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
This paper presents a training-free framework to improve accuracy in long-form VQA. The method starts with the adaptive initial sampling by using CLIP-based query-frame similarity and Gaussian mixture modeling, followed by an iterative reasoning loop that selectively applies a foundation VLM to high-potential frames. Experiments on several benchmarks show that the proposed method consistently improves accuracy.

### Strengths
The paper is well-written and easy to follow, with clear motivation, framework design, and experimental presentation. 

The three-stage pipeline is also technically coherent and logically structured. The approach is training-free, which is good for integration into various VLMs,

The experiments and ablation studies demonstrate consistent improvements across multiple benchmarks.

### Weaknesses
While the paper is technically solid, its novelty and conceptual contribution are limited. The proposed method offers an effective improvement by integrating various schemes, but the overall contribution is incremental, because it only refines existing query-based frame-selection paradigms rather than offering a fundamental advance in long-video reasoning.

The paper does not include comparisons or discussions with agent-based methods, which represent another important line of work in long-video understanding.

The proposed method remains query-specific and cannot support multi-turn or persistent reasoning across video contexts.

### Questions
In addition to questions in weakness part, the authors are encouraged to address the following issues.

The iterative refinement uses several manually tuned hyperparameters (e.g., γ, α, β), and it's better to discuss their generalization across datasets.

It's suggested to discuss the future extensions, especially its combination with agent-based long-form video understanding systems.

### Soundness
2

### Presentation
3

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
The paper proposes A.I.R., which is a multi-stage frame selection heuristics for long video question answering. The pipeline emphasizes efficiency by minimizing the number of frames fed to the heavy-weight reasoning-based VLM for analysis. The pipeline first computes question-frame similarity scores using a light-weight CLIP-style VLM, and then perform multi-stage selection to determine the frames for the expensive MLLM-powered question answering. The selection process include an initial adaptive sampling step, where scores are modeled as a 2-component GMM (relevant and irrelevant), followed by a loop of interval potential ranking, reasoning-based VLM analysis, early stop mechanism and localized density sampling to add or remove from the candidate frame set until a fixed budget is used. The frames are then input to a MLLM with the question to generate the answer. The method is evaluated on a wide range of models (VILA-1.5, QwenVL-2.5, InternVL-3, LLaVA-OneVision) and on multiple datasets (VideoMME, MLVU, LVB, EgoSchema, NextQA), showcasing its effectiveness in a broad range of settings.

### Strengths
* The paper addresses the topic of long-video understanding, which is of high real-world value and is essential for a broad set of video analysis applications.

* By demonstrating the effectiveness of human-derived heuristics, the paper reveals key limitations of recent LLM-agent-based frame selection pipelines, which often incur much higher costs due to frequent calls to expensive LLMs/MLLMs.

* The paper is very well written, with a clear framework and a good coverage of technical details.

* The proposed method is evaluated against a wide range of recent methods across multiple datasets and shows strong performance, demonstrating its efficiency and effectiveness in a comprehensive way. Ablation experiments also demonstrate the effectiveness of individual components of the proposed method.

### Weaknesses
* The CLIP affinity score is used extensively in multiple stages of the pipeline, which raises questions about how the quality of this score may affect the overall efficiency and accuracy of the pipeline. CLIP models are known to have several limitations, such as a relatively low token number limits of the text encoder (tens to hundreds, versus thousands or more in MLLMs), being less capable at modeling complex relationships in text expressions [1], and being less capable at extracting fine-grained visual information [2]. A low precision or recall may result in excessive frames being sent to reasoning VLM or information loss, potentially impacting the efficiency or quality of the pipeline. A more detailed discussion [e.g., see questions (a, b)] on the issue might clarify the strength and limitations of the proposed approach.

* In Table 2 and 3, it seems that the performance gain of frame selection is closing rapidly as the backbone model quality improves, which raises some concerns about the long-term impact of the work or the frame selection pipelines in general.


[1] Yuksekgonul, Mert, et al. "When and Why Vision-Language Models Behave like Bags-Of-Words, and What to Do About It?." The Eleventh International Conference on Learning Representations.

[2] Xie, Chunyu, et al. "FG-CLIP: Fine-Grained Visual and Textual Alignment." Forty-second International Conference on Machine Learning.

### Questions
* Following weakness 1, have the authors encountered any failure cases of the CLIP text encoder, e.g., questions longer than the max number of tokens or CLIP failing to capture the semantic of some complex expressions?

* Following question 1, if systematic failure cases exist, could the paper include a discussion about how they are mitigated by designs in the proposed pipeline, or how they could possibly be mitigated in future works?

* Following weakness 2, if resources permit, could the paper include more results on larger or closed models to better demonstrate its advantages of being training-free and avoiding large numbers of MLLM calls?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper tackles a core challenge in Video Question Answering (VideoQA) — how to efficiently select a small set of informative video frames that best correspond to a textual query. To overcome this trade-off, the authors propose A.I.R., a training-free framework for Adaptive, Iterative, and Reasoning-based frame selection. A.I.R. adaptively chooses the most relevant frames by combining efficient CLIP-based similarity analysis with targeted reasoning from a VLM. A.I.R. consists of two main stages, including adaptive initial sampling and iterative frame selection.  The method was extensively tested with various foundation VLMs, including InternVL-3, QwenVL-2.5, LLaVA-OneVision, and VILA-1.5, on both long-video and short-video benchmarks.

### Strengths
1. The authors provide a clear problem motivation, starting from the inefficiency of uniform sampling and the limitations of existing CLIP-based and VLM-based frame selection methods. The technical pipeline is presented logically, with consistent terminology and step-by-step exposition of each stage (Adaptive Initial Sampling → Iterative Frame Selection → QA). 
2. This paper shows high-quality figures and conceptual illustrations. Figures such as Figure 1 (problem motivation), Figure 2 (overall pipeline), and Figure 3 (detailed visualization of both Adaptive Sampling and Iterative Selection stages) are visually rich and well-annotated, effectively guiding the reader through the method.
3. The technical part is given with many details. Readers can easily grasp the general idea of this paper.

### Weaknesses
1. Although the proposed A.I.R. framework achieves practical performance improvements, its core design is composed mainly of heuristic strategies rather than a principled methodological innovation. Many components—such as the adaptive thresholding via Gaussian Mixture Models, interval potential ranking, and localized density sampling—are intuitive extensions or empirical engineering choices rather than conceptually new formulations. As a result, the framework feels more like an optimized combination of existing ideas than a grounded advancement. This reliance on heuristics makes it difficult for readers to extract generalizable insights or new theoretical understanding from the paper, which somewhat diminishes its methodological contribution.
2. The proposed A.I.R. framework involves a large number of hyperparameters, which makes the overall method difficult to interpret and reproduce. This abundance of loosely motivated hyperparameters weakens the methodological transparency of A.I.R. and increases the barrier for readers to understand or extend the framework fully.
3. The performance advantage of the proposed A.I.R. framework is not particularly significant, and the experimental section lacks comprehensive comparisons with the most recent state-of-the-art methods. Although the paper reports moderate gains (often around 1–2% improvement on several benchmarks such as Video-MME and MLVU), these margins are relatively small given the additional complexity introduced by the multi-stage adaptive and iterative procedures. Moreover, the experiments primarily compare A.I.R. against earlier CLIP-based or training-free frame selection baselines (e.g., MDP3, BOLT, Frame-Voyager), but do not include newer or stronger contemporary methods that could provide a more convincing assessment of its competitiveness. Consequently, the empirical results do not clearly demonstrate that A.I.R. represents a substantial advancement over current techniques.

### Questions
The paper does not provide sufficient details or rationale regarding the hyperparameter settings used in the A.I.R. framework. How could this issue be addressed to improve the transparency and reproducibility of the proposed method?

### Soundness
3

### Presentation
3

### Contribution
2
