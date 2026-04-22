# Understanding the Design Space and Cross-Modality Transfer for Vision-Language Models

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 4, 4, 4

## Abstract
The training of multimodal models involves many design choices, such as the underlying modality-specific tokenizers, fusion mechanisms, and strategies for freezing model layers during different training stages. However, the individual impact of these decisions on downstream multimodal performance remains poorly understood due to the diversity of current practices. In this paper, we systematically investigate how choices in image tokenization, architectural design, and layer-freezing strategies affect the training and cross-modal generalization of vision-language models (VLMs). We systematically explore a design space comprising six image tokenizers, three VLM architectural variants, and various parameter-freezing strategies. To further probe cross-modality transfer, we introduce three new synthetic datasets, which we use to evaluate our pretrained models. Our experiments reveal several key trends. (i) Image tokenizers trained with text-aware objectives are crucial for strong VLM performance, outperforming those trained without such objectives on both in-domain and out-of-domain tasks. (ii) Architectures that explicitly separate modalities such as the Mixture-of-Transformers fusion architecture, along with training recipes that preserve the more general textual knowledge and reasoning of the base language model, generalize well to out-of-domain tasks. (iii) Cross-modality transfer is heavily dependent on representational alignment between the text and images; in our synthetic setting, image-to-text transfer is comparatively strong, whereas there was little text-to-image transfer.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper explores what design choices have a large impact on VLMs' VQA and reasoning performance. 
This paper examines image tokenizers, fusion architectures, and layer-freezing strategies for the design choices. 
The authors train 50+ model variants on a Qwen3-0.6B backbone and evaluate on standard benchmarks plus three new synthetic datasets designed to probe cross-modality transfer.

### Strengths
The paper tackles an interesting question of what steps create good VLMs.

The controlled experimental design effectively isolates individual design choices (tokenizer, architecture, freezing strategy) that are typically confounded in VLM research.

Provides actionable insights for practitioners (e.g., Takeaways 1-8).

### Weaknesses
Limited model size. The authors only use a single sub-1B model. This begs the question of whether the results are still valid when scaling up the models. While I understand the authors may be limited in their available compute, the question they seek to answer requires more computational power. Hence, the authors should instead use their compute to explore questions within their realm. 

Only single runs. No tests of significance.

Unfair Architectural Comparisons. MoT has 400M+ extra parameters vs Joint-Decoder despite "comparable FLOPs." Table 2 does not normalize for parameter count—MoT's advantage when all frozen may simply be due to more capacity. You should add parameter-matched baselines or explicitly report params/FLOPs for each config.

While the authors explore what setup gives the best performance, they do not tackle the why?

There is also no error analysis.

There is no discussion of risks regarding data contamination. There are datasets from 2014, so models like SigLIP 2 have likely seen samples from it.

“Our results show that the Mixture-of-Transformers (MoT) architecture is particularly effective.” Why do you state this? On what grounds? See under Questions for more details.

The reasoning performance metrics should include what random guessing would yield.

The table captions need to be above the tables as per the formatting instructions: “The table number
and title always appear before the table.”
The table captions are also incomplete and missing, for instance, what the bold numbers represent and what can be drawn from the table.
Table formatting is inconsistent. The style of Tables 1 and 2 is very different from that of Tables 3 to 6.

There are some odd hyperlinks, such as SigLIP 2, on lines 217 and 219. Also, why are the references to Sections, Tables, and Figures in red?

Section 2 is missing a subsection on evaluation metrics.

Given the datasets you generate, you should consider (and cite!) https://arxiv.org/abs/2407.06581. Also, you should include the full prompts given to the models.

The findings, while useful, are largely confirmatory (text supervision helps, unfreezing trades off in/out-domain performance) rather than surprising. The cross-modality transfer analysis, though interesting, is hampered by methodological constraints.

Typos etc.:
Line 115: “which contains of 28 transformer layers” should be “which contains 28 transformer layers”
Line 236: +5.3 is missing colour.

### Questions
How do you conclude: “MoT with an unfrozen image tokenizer and frozen language layers delivers the best overall task performance?” 
The joint decoder with everything unfrozen delivers 1.9pp higher in distribution performance with 3.2 pp lower out of distribution performance. However, depending on how you weigh the two metrics, the conclusion changes. Furthermore, if I looked at improvement over the baseline frozen model, then the conclusion changes yet again.
Please create a single table extension of Table 2 where you do not aggregate on the tokenizers. (It should include all tokenizers and not just the subset you used for Table 2).

For the start of Section 4, why do you create your own datasets rather than using existing ones?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper broadly evaluates existing vision-language model approaches, covering image tokenizers, model architecture, and frozen parameter settings.
Each method is evaluated on several vision question answering datasets in terms of several perspectives including in-domain/out-of-domain settings and cross-modalirty transfer from image/text to text/image.
Through experimental evaluations, this paper provides some observations, e.g., the best approach for image tokenizers is to learn with text-alignment objectives.

### Strengths
- Readers easily grasp the claims of this paper.
The experimental evaluation setup is clearly organized. In addition, the conclusion of each experiment is well summarized in takeaways.

- Broadly and fairly evaluating existing technologies provides useful information for practitioners.
This paper provides a comprehensive evaluation of key aspects in handling vision language models, including continuous/discrete image tokenizers and frozen/unfrozen approaches,
from several perspectives including in-domain and out-of-domain scenarios.

- Model architecture and image tokenizers are crucial elements in vision language models. 
If the paper had provided a clearer and more promising research direction, it could have had a significant impact.
However, as noted in Weakness, the claims are not sufficiently supported by the experimental evidence.

### Weaknesses
- In the evaluation, the language backbone consists solely of one model, Qwen3-0.6B. This might cause some bias in experiments although I do not know the impact of language backbone.

- The experimental evaluation mainly serves as a benchmark comparison across existing techniques and concludes without deeper evaluation based on hypotheses or analytical questions.
Since no experiments were designed to substantiate the observations, the takeaways might not go beyond observations.
 Of course, benchmark evaluations are important, but the paper becomes stronger by presenting analytical experiments that go beyond observations,
 identifying open questions that should be addressed as a field, and outlining future directions for vision language models.

- Regarding the above issue, I'm not sure if the conclusion in the takeaways is correct. 
Takeaway 1 claims "Image tokenizers trained with text-alignment objectives are crucial for strong VLM performance.".
Its basis is that image tokenizers trained with text-alignment objectives (AIMv2, SigLIP 2, CLIP) outperform those trained for image reconstruction (TiTok, VAR).
However, in Section 2.2, the distinction between AIMv2, SigLIP 2, CLIP and TiTok, VAR is whether they are continuous tokenizers or discrete tokenizers.
Thus, it is also possible to conclude that continuous tokenizers outperform discrete tokenizers.
Takeaway 5 makes a similar claim regarding cross-modality transfer. However, since Table 3 and Table 1 show similar trends, it is also possible that TiTok and VAR simply have lower performance.

### Questions
If there are any misunderstandings on my part in Weakness, could you point them out?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents a systematic empirical study of the design space for vision-language models (VLMs). The authors investigate three key dimensions: image tokenization, architectural fusion mechanisms, and layer-freezing strategies. By training and evaluating over 50 VLM variants built on a Qwen3-0.6B backbone, the paper derives some takeaways about what design choices lead to better performance on in-domain and out-of-domain tasks.

### Strengths
The paper is exceptionally well-written and organized.

The experimental methodology is comprehensive.

### Weaknesses
1. All experiments are conducted on a 0.6B parameter LLM backbone. The paper's takeaways are presented as general principles for VLM design, but without validation on larger, more capable models, they may simply be artifacts of a low-capacity regime.

2. Several of the main takeaways are well-known, such as takeaway 1 & 2 & 3.

3. I am not sure how to get a certain conclusion of this paper, it seems more like to be a blog & survey.

### Questions
I'm curious what the author's highest priority takeaway is.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents a detailed survey of VLM design axes, focusing on image tokenizers, modality fusion architectures, and LLM backbone freezing strategies by altering many different configurations based on a single model (Qwen3-0.6B). Through a consistent three-stage training pipeline (pretraining, finetuning, and cross-modality reasoning transfer), the authors evaluate performance on both in-domain and out-of-domain benchmarks. They report several interesting findings regarding the impact of image tokenizer design, the effect of thawing the language backbone, and comparisons among modality fusion strategies. These results provide actionable insights, although their exclusive use of a very small sized model leaves some room for doubt about the generalization of the findings.

### Strengths
- The paper conducts a comprehensive and well-structured examination of previously underexplored VLM design axes (such as image tokenizers, fusion architectures, and layer-freezing strategies) within a consistent framework.

- Their findings yield several actionable insights, which benefits future model development, e.g., the benefit of training image tokenizers with text-alignment objectives, the importance of selecting unfreezing strategies based on target tasks, and the effectiveness of the Mixture-of-Transformers (MoT) architecture.

- In addition, the analysis of cross-modality transfer offers a valuable perspective that goes beyond conventional, often superficial, VQA style benchmarking.

### Weaknesses
- All experiments are conducted on a very small 0.6B-parameter model (Qwen3-0.6B), which limits the overall impact of the work. A study of this depth and architectural scope would ideally include experiments on larger models. If training all variants is not feasible, testing a few representative configurations with larger models would help validate the trends observed with the 0.6B model.

- While the paper mentions training "over 50 variants", these are essentially factorial combinations of a few design axes rather than truly distinct model architectures. The framing could better reflect that this is rather an extensive ablation study rather than a broad model comparison.

- Over the years, I have seen several survey papers on the design space of multimodal LLMs (some of these papers are listed below), some of which might already have covered similar architectural and training aspects. These works would be valuable reference points, but the current related work section does not discuss them.

- (Minor) The presentation could be improved. Some results are reported in aggregated form without a clear enumeration of all configurations. For example, the claim of “50+ variants” could be shown more explicitly through a summary table or schematic.

Long, Siqu, et al. "Vision-and-language pretrained models: A survey." arXiv preprint arXiv:2204.07356 (2022).

Du, Yifan, et al. "A survey of vision-language pre-trained models." arXiv preprint arXiv:2202.10936 (2022).

Yin, Shukang, et al. "A survey on multimodal large language models." National Science Review 11.12 (2024): nwae403.

Zhang, Duzhen, et al. "Mm-llms: Recent advances in multimodal large language models." arXiv preprint arXiv:2401.13601 (2024).

Ma, Xiaorui, Haoran Xie, and S. Joe Qin. "Efficiently Integrate Large Language Models with visual perception: A survey from the training paradigm perspective." Information Fusion (2025): 103419.

### Questions
See the above weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3
