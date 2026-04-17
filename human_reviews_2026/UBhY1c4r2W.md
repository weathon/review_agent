# PoSh: Using Scene Graphs to Guide LLMs-as-a-Judge for Detailed Image Descriptions

- Decision: Accept (Poster)
- Scores: 4, 6, 8, 4

## Abstract
While vision-language models (VLMs) have advanced into detailed image description, evaluation remains a challenge. Standard metrics (e.g. CIDEr, SPICE) were designed for short texts and tuned to recognize errors that are now uncommon, such as object misidentification. In contrast, long texts require sensitivity to attribute and relation attachments and scores that localize errors to particular text spans. In this work, we introduce PoSh, a metric for detailed image description that uses scene graphs as structured rubrics to guide LLMs-as-a-Judge, producing aggregate scores grounded in fine-grained errors (e.g. mistakes in compositional understanding). PoSh is replicable, interpretable and a better proxy for human raters than existing metrics (including GPT4o-as-a-Judge). To validate PoSh, we introduce a new dataset, DOCENT. This novel benchmark contains artwork, paired with expert-written references, and model-generated descriptions, augmented with granular and coarse judgments of their quality from art history students. Thus, DOCENT enables evaluating both detailed image description metrics and detailed image description itself in a challenging new domain. We show that PoSh achieves stronger correlations (+0.05 Spearman ρ) with the human judgments in DOCENT than the best open-weight alternatives, is robust to image type (using CapArena, an existing dataset of web imagery) and is a capable reward function, outperforming standard supervised fine-tuning. Then, using PoSh, we characterize the performance of open and closed models in describing the paintings, sketches and statues in DOCENT and find that foundation models struggle to achieve full, error-free coverage of images with rich scene dynamics, establishing a demanding new task to gauge VLM progress. Through both PoSh and DOCENT, we hope to enable advances in important areas such as assistive text generation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper proposes PoSh, a new metric for evaluating fine-grained image descriptions. PoSh extracts scene graphs from both reference and generated descriptions, preserving the object–attribute–relation structure, and uses these graphs as a structured scoring criterion to guide open-source LLMs in performing fine-grained judgments. This enables the identification of mistakes and omissions in generated text.

The authors construct a new benchmark, DOCENT, comprising 1,750 artworks from the National Gallery of Art, expert-written descriptions, outputs from VLMs, 300 fine-grained annotations, and 600 coarse-grained pairwise ratings provided by art history students. Experiments show that PoSh outperforms existing metrics on both DOCENT and CapArena. It can also serve effectively as a reward function in reinforcement learning.

### Strengths
1. PoSh addresses a clear gap in evaluating detailed image descriptions. By using scene graphs as structured rubrics, PoSh enables error localization and produces human-aligned, interpretable scores.
2. Built entirely on open-weight models and public tools, PoSh avoids reliance on proprietary APIs (e.g., GPT-4o), making it accessible and deployable for researchers with limited resources.
3. The paper introduces DOCENT—a novel dataset of expert-written art descriptions paired with granular and coarse human judgments from domain-knowledgeable annotators. This enables evaluation for complex visual domains.

### Weaknesses
1. **Computational overhead may hinder scalability**: The pipeline, comprising scene graph extraction, multi-pass identifier generation, and LLM-based QA, is considerably more complex than standard metrics. The paper does not include ablation studies or timing analyses of individual components. Without such efficiency profiling, it remains unclear whether PoSh offers a favorable trade-off between evaluation quality and computational cost in large-scale settings.
2. **Lack of validation as a data curation tool for MLLM training**: While the paper demonstrates PoSh’s effectiveness as a reinforcement learning reward signal, it does not explore its utility in filtering or ranking training data for MLLM fine-tuning. Comparing models trained on PoSh-filtered data versus those trained with other metrics would better establish PoSh’s practical value in the full model development lifecycle.

### Questions
Do captioning models that incorporate structural priors (e.g., spatial or scene graph inputs) benefit disproportionately under PoSh?
Some recent methods explicitly inject visual relational priors—such as object positions or scene graph structures—into the captioning process. 
1. Since PoSh itself uses scene graphs as evaluation rubrics, such models might receive inflated scores due to structural alignment between generation and evaluation, rather than superior semantic fidelity. Have the authors evaluated PoSh on outputs from graph-aware captioning systems (e.g., SG-LLaVA [1])?
2. I also wonder whether the authors think that incorporating structural priors during generation might be more important than adding them during verification.

[1] Jingyi Wang, Jianzhong Ju, Jian Luan, and Zhidong Deng. LLaVA-SG: Leveraging Scene Graphs as Visual Semantic Expression in Vision-Language Models.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents POSH, a new metric for detailed image description. It computes scores grounded in fine-grained errors by adopting scene graphs to guide LLMs-as-Judge. POSH is interpretable and aligns better with human validation. Additionally, they propose DOCENT, a new dataset of artwork, references, and descriptions. They validate POSH on DOCENT and find it has better correlation with human judgments. By benchmarking open and closed-source models on DOCENT, they identify strengths and weaknesses of VLMs in understanding images.

### Strengths
- They focus on an important aspect of VLM understandings. They focus on the detailed description, and the metric is interpretable. 
- They propose a benchmark with expert-written descriptions and 900 granular & coarse judgments from raters. The manual effort is massive.
- They open-sourced the benchmark and metric, which will benefit the community. 
- They evaluate multiple open-source and closed-source models.

### Weaknesses
- The writing could be better. For example, in the table, they use POSH to denote the finetuned Qwen model with POSH reward, while POSH is a metric in the meantime. This is a bit confusing.
- For the findings of POSH as a reward function, they only experiment with the Qwen2.5-VL-7B model. The findings may not be model-agnostic. 
- It is concerning that POSH works better on their proposed DOCENT benchmark but is adequate on other benchmarks like CapArena.

### Questions
- Could you provide results with other VLMs other than Qwen2.5-VL?
- What is the text and image distribution difference between DOCENT and CapArena? Is there any other caption benchmarks you could use to validate the effectiveness of POSH?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors propose PoSh, a reference-based metric for long-form image captioning. PoSh works by first constructing scene graphs for both the reference and generated captions. The metric then constructs templated questions for both the reference and generated captions' scene graphs and answers these questions with an LLM judge to determine the recall and precision, respectively, of the generated caption. As the metric generates numerous questions for both scene graphs, PoSh can yield both granular and coarse assessments of caption quality.

The authors validate alignment with human judgements on CapArena, a pre-existing benchmark, and DOCENT, a dataset containing artwork with expert-written captions that they introduce. PoSh outperforms baseline metrics for granular identification of mistakes and omissions and is either competitive with or outperforms existing metrics for coarser evaluations. The authors also validate their metric as a reward function for RL training, finding improvement over simple SFT.

### Strengths
The paper has numerous strengths:
- Firstly, the task of long-form image captioning evaluation is an important one as models continuously improve in capabilities. PoSh acts as an important contribution within this space by proposing a straightforward reference-based metric that converts the references and generations into scene graphs and using these to assess precision and recall. Particularly, the use of questions to assess both precision and recall lends the metric interpretability and granularity.
- The method outperforms prior baselines for granular evaluations and is competitive with or better than other metrics for coarse evaluations.
- The authors propose a new benchmark, DOCENT, with the novel domain of visual art. DOCENT is coupled with expert-written reference captions and human judgements for a range of vision-language models. This not only helps the evaluation of the metric but could also be used as a testbed for future metrics or for the generation quality of other vision-language models.
- Additionally evaluating PoSh's performance as a reward model makes the evaluation of the metric very complete. I can imagine the granular nature of PoSh being also used to generate targeted natural language feedback for models.

### Weaknesses
The main weaknesses I can see are:
- PoSh is going to be sensitive to the accuracy of the extracted scene graphs, where there could be errors either during the dependency parsing process or during coreference resolution. Figure 3 marking "painting" as a mistake acts as one example of this. 
- While I think PoSh could act as a strong reward model, it does presume access to detailed reference captions, which is expensive to curate on a large scale. PoSh being reference-based similarly restricts its use for evaluation to dedicated datasets for this purpose.

### Questions
The metric's performance on the various evaluations already gives good signal regarding its quality. The paper would nonetheless be improved through an evaluation of the intermediate components of the metric itself. For instance, how accurate is the scene graph extraction (as measured via precision and recall against reference human-annotated scene graphs)? Alternatively, as this might be more feasible during the rebuttal period, how accurate are the LLM judge's answers?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a new metric for evaluating detailed image descriptions, termed POSH. POSH utilizes scene graphs as structured guidelines to direct LLMs in assessing fine-grained errors in image descriptions, such as compositional correctness. POSH offers a replicable and interpretable evaluation experience, outperforming existing metrics, including GPT4o. Additionally, the authors introduce a new dataset, DOCENT, which contains artwork paired with expert-written descriptions and model-generated descriptions. The dataset also includes annotations from experts, providing a challenging benchmark for evaluating the detailed description of images. Furthermore, the authors proof that POSH can be used as a reward function to achieve better performance than standard sft.

### Strengths
* The paper is well-written and easy to follow.
* Evaluating the image description is indeed a non-trivial task, and the proposed new metric for evaluating detailed descriptions is important for the field of image captioning.
* I agree that a good metric for image-description should be grounded on fine-grained cues, localized on text spans.
* The paper introduced the DOCENT dataset, which includes expert-written descriptions and annotations, and the quality is well controlled.

### Weaknesses
* POSH is reliance on a model to generate the scene graph introduces inaccuracies and errors, which could be a potential bottleneck for its effectiveness.
* The use of scene graphs to evaluate image-text alignment has been discussed in previous papers like [1]; the authors need to clarify the uniqueness of POSH.
* The proposed dataset covers artworks, but in practical applications, images of natural scenes are more common.

[1] Davidsonian Scene Graph: Improving Reliability in Fine-grained Evaluation for Text-to-Image Generation

### Questions
* Why choose artworks as a benchmark? Instead of other more common or more representative domains
* How to ensure repeatability, as there are probabilistic models used (qwen3)?

### Soundness
3

### Presentation
3

### Contribution
3
