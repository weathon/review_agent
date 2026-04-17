# CaReBench: A Fine-grained Benchmark for Video Captioning and Retrieval

- Decision: Accept (Poster)
- Scores: 4, 6, 10, 6

## Abstract
Video understanding, including video captioning and retrieval, is still a great challenge for video-language models (VLMs). The existing video retrieval and caption benchmarks only include short descriptions, limits their ability of detailed video understanding evaluation. To address this problem, we present CaReBench, a testing benchmark for fine-grained video Captioning and Retrieval with 1,000 high-quality pairs of videos and human-annotated detailed captions. Uniquely, it provides manually separated spatial annotations and temporal annotations for each video. Based on this design, we introduce two evaluation metrics, ReBias and CapST, specifically tailored for video retrieval and video captioning tasks, respectively. These metrics enable a comprehensive investigation into the spatial and temporal biases inherent in VLMs. In addition, to handle both video retrieval and video captioning tasks in a unified framework, we develop a simple baseline based on a Multimodal Language Model (MLLM). By implementing a two-stage Supervised Fine-Tuning (SFT), we fully unlock the potential of MLLM, enabling it not only to generate detailed video descriptions but also to extract video features. Surprisingly, experimental results demonstrate that, compared to the CLIP-based models designed for retrieval and the popular MLLMs skilled in video captioning, our baseline shows competitive performance in both fine-grained video retrieval and video detailed captioning.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper propose CAREBENCH, a novel evaluation benchmark for fine-grained video captioning and retrieval with manually separated spatial annotations and temporal annotations. New evaluation metrics ReBias and CapST are introduced for a deeper investigation into the spatial and temporal biases inherent in VLMs. Comprehensive experiments are conducted with developed unified baseline method CARE for address both tasks.

### Strengths
1.	Fine-grained video captioning and retrieval are worth studying and have certain commonalities. This work proposes a new benchmark for unified evaluation on both tasks.

2.	The experiment result is solid with a newly developed baseline method for both video captioning and retrieval.

3.	Convincing visualization examples are provided to prove the effectiveness of the proposed method.

### Weaknesses
1.	Although video captioning and retrieval tasks have a certain duality, the name of the proposed fine-grained video captioning and retrieval benchmark in this article seems to simply blend these two tasks, appearing to be an improvement and integration of previous work, lacking some innovative new concepts of video understanding.

2.	The developed baseline method is encouraged to be tested on some other related video understanding benchmarks to demonstrate its generalization ability.

### Questions
1.	See weakness.

2.	I appreciate the dedicated efforts on the proposed benchmark and its experiment results. However, I still have some doubts about the positioning of this job as a benchmark or dataset. To begin with, as a benchmark, its targeted video captioning and retrieval are relatively old video understanding tasks, the name of the proposed method should require more innovative new concepts of video understanding. Meanwhile, many compared benchmarks in Table 1 are the test set of a larger video understanding dataset, which includes more training samples for the cross-validation of its effectiveness, but this work only proposes a testing benchmark instead of a complete dataset for fine-grained video captioning and retrieval. In summary, I would give a much higher score if this work proposed a complete dataset with both a training set and a testing benchmark. If not, I would consider raising my score if the author integrates these two tasks to provide convincing and innovative concepts of video understanding.

### Soundness
3

### Presentation
2

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
The paper introduces CaReBench, a fine-grained benchmark for video captioning and retrieval with human-annotated spatial and temporal descriptions, along with two new evaluation metrics, ReBias and CapST, to analyze spatiotemporal bias. It further proposes CARe, a unified two-stage model that jointly handles both tasks, achieving competitive performance compared to CLIP-based and MLLM-based baselines.

### Strengths
1. Proposes CaReBench, a fine-grained benchmark with detailed spatial and temporal annotations and new metrics ReBias and CapST for analyzing spatiotemporal bias.
2. Introduces a unified model CARe that jointly handles video captioning and retrieval through a two-stage training framework.
3. Demonstrates strong experimental results and clear analysis, supported by high-quality human annotations and well-presented methodology.

### Weaknesses
See questions.

### Questions
1. Could the authors clarify the motivation for integrating video captioning and video retrieval into a single unified model? It would be helpful to explain whether these two tasks bring any mutual benefits or complementary effects in practice.
2. Regarding the use of LLMs for NLI judgment, could the authors elaborate on its reliability? For instance, would the evaluation results vary significantly if a different LLM were used? 
3. It would be interesting to know whether the EOL-based embedding extraction method can effectively distinguish long texts that are semantically similar but not identical.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
10

### Rating Number
10

### Confidence
3

### Summary
The paper introduces CAREBENCH, a new benchmark of 1,000 videos with rich human-written captions designed to evaluate fine-grained video understanding. Each video has hierarchical, multi-aspect annotations (overall summary, static object details, dynamic action details, and miscellaneous context). Spatial and temporal descriptions are manually separated to probe how models handle static scene vs. dynamic action information. Based on this data, the authors propose two evaluation metrics: ReBias, which measures a model’s bias toward spatial vs. temporal retrieval, and CapST, an LLM-based captioning score combining recall/precision of objects and events. They also present CARE, a unified video-language model (built on Qwen2-VL) trained via a two-stage fine-tuning (first to improve caption detail, then contrastively for retrieval). Experiments (zero-shot evaluation on CAREBENCH and other datasets) show CARE achieves very strong performance in both detailed captioning and video retrieval, outperforming specialized retrieval models (CLIP variants) and powerful captioning MLLMs. The paper carefully discusses related work, conducts ablations of the two-stage training, and analyzes a consistent spatiotemporal bias (models do much better on spatial cues than temporal actions).

### Strengths
1. CAREBENCH demonstrates notable strengths across key areas. Its originality lies in the novel hierarchical annotation schema—dividing video content into summary, objects, actions, and miscellaneous categories—which addresses a critical gap in video understanding evaluation. The introduction of specialized ReBias and CapST metrics further showcases innovation, leveraging the dataset's structure to examine biases and caption quality.

2. The work exhibits strong methodological rigor through its careful human annotation process involving dual annotators and expert refinement. The two-stage training approach is well-designed, with Stage I focusing on fine-grained caption alignment and Stage II on contrastive retrieval training, a design validated through ablation studies.

3. The paper is clearly written and logically structured, with informative figures and tables that effectively support the technical content. The comprehensive related work discussion demonstrates deep scholarly engagement.

4. In terms of significance, CAREBENCH provides valuable tools that reveal how current VLMs often rely on spatial shortcuts while struggling with dynamic actions. By exposing these biases, the work should stimulate development of more balanced models, with the CARE baseline demonstrating the feasibility of unified modeling approaches.

### Weaknesses
1. The work has several limitations that warrant consideration. While the paper effectively identifies significant spatiotemporal biases in models, it stops at measurement and does not propose methods to mitigate these biases. This focus on benchmarking over developing corrective techniques somewhat limits its immediate practical impact on improving model design.

2. The generalization capability of the proposed CARE model remains uncertain. Its evaluation on standard benchmarks beyond CAREBENCH shows mixed results, particularly on MSR-VTT retrieval where it fails to clearly outperform established baselines. More comprehensive analysis is needed to understand how well the unified approach transfers across diverse domains.

3. The CapST metric relies exclusively on DeepSeek-V3 as the sole LLM judge for evaluating objects and actions, which raises potential concerns about robustness and inherent biases in the assessment. Although some validation against human judgment is provided, a more extensive evaluation using multiple LLMs, varied prompts, or additional human raters would strengthen confidence in the metric's reliability and general applicability.

### Questions
1. How sensitive is the CapST metric to the choice of LLM judge or prompt? Have you validated its consistency using alternative models like GPT-4 or through prompt variations?

2. While CARE shows strong zero-shot retrieval results on MSR-VTT, MSVD, and DiDeMo, could you provide more experimental details? How does it perform on captioning tasks for these datasets, and what are its limitations on out-of-domain videos?

3. Given CAREBENCH's scale of 1,000 videos, do you recommend it primarily for evaluation or also for fine-tuning? What are the risks of overfitting and how might training on it affect model robustness?

4. Will the benchmark be publicly released? Are there licensing restrictions from FineAction, and how will reproducibility be ensured?

5. Given the spatial bias identified by ReBias, have you explored any mitigation strategies, such as data augmentation or balanced training? What approaches would you suggest to reduce this bias?

6. How does CapST handle paraphrases or partial omissions? Could you provide quantitative correlation between CapST scores and human judgments to further validate the metric?

7. Could you elaborate on how conflicts between annotators were resolved during expert refinement? What guidelines ensured consistency, and was any inter-annotator agreement measured?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces CaReBench, a new fine-grained benchmark for video captioning and retrieval, comprising 1,000 videos with human-annotated hierarchical captions. Each caption includes spatial and temporal components, allowing a detailed evaluation of models’ understanding of static objects and dynamic actions.

The authors also propose two novel metrics:
ReBias, for quantifying spatiotemporal bias in retrieval tasks, and
CapST, for evaluating captions with both precision and recall over spatial and temporal elements.

Additionally, the paper presents CARE, a unified baseline model built upon Qwen2-VL, trained via two-stage SFT to handle both captioning and retrieval tasks simultaneously.

### Strengths
1. **Innovative Benchmark Design:**
CAREBENCH introduces a uniquely structured dataset with hierarchical annotations that explicitly separate spatial and temporal descriptions, filling a clear gap in current benchmarks.

2. **New Evaluation Metrics:**
ReBias and CapST are well-motivated and address the shortcomings of existing metrics (e.g., CIDEr, AutoDQ, VDCScore), providing more interpretable and fine-grained evaluations.

3. **Unified Framework:**
The CARE model elegantly unifies retrieval and captioning within one architecture using a two-stage fine-tuning pipeline, demonstrating potential task synergy.

4. **Comprehensive Experiments:**
The paper includes extensive quantitative results across state-of-the-art models and thorough ablation studies on the two-stage fine-tuning process.

### Weaknesses
1. **Reliance on LLM-Based Evaluation:**
CapST depends on an LLM (DeepSeek-V3) as the evaluator, which may introduce bias. The paper lacks inter-rater consistency checks or human alignment experiments.

2. **Theoretical Depth:**
The paper mainly focuses on empirical contributions. The “unified mapping” idea (ϕ: RT×H×W×C → RD) is intriguing but not theoretically explored.

3. **Incomplete Bias Mitigation:**
While ReBias reveals spatiotemporal imbalance, the proposed model still shows clear temporal bias, and the paper does not attempt to mitigate it.

4. **Ablation Detail and Visualization:**
Some analyses, such as the effects of video length, motion complexity, or per-category performance variance, are missing and would add valuable insight.

### Questions
1. How stable are CapST scores across different LLM evaluators (e.g., GPT-4o vs DeepSeek-V3)?

2. Have you explored joint multi-task training (caption + retrieval simultaneously) instead of the sequential two-stage pipeline?

3. What are the main annotation challenges in separating spatial and temporal components, and how is annotator consistency ensured?

4. Given the rapid evolution of multimodal large language models (MLLMs), have you considered adding comparisons with more recent MLLMs (for example, Qwen2.5-VL) to demonstrate how your benchmark and unified model stack up against the latest models?

### Soundness
3

### Presentation
3

### Contribution
3
