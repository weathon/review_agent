# Omni-Weather: A Unified Multimodal Model for Weather Radar Understanding and Generation

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 6, 4, 8

## Abstract
Weather modeling requires both accurate prediction and mechanistic interpretation, yet existing methods treat these goals in isolation, separating generation from understanding. To address this gap, we present Omni-Weather, the first multimodal foundation model that unifies weather generation and understanding within a single architecture. 
Omni-Weather integrates a radar encoder for weather generation tasks, followed by unified processing using a shared self-attention mechanism.
Moreover, we construct a Chain-of-Thought dataset for causal reasoning in weather generation, enabling interpretable outputs and improved perceptual quality.
Extensive experiments show Omni-Weather achieves state-of-the-art performance in both weather generation and understanding.
Our findings further indicate that generative and understanding tasks in the weather domain can mutually enhance each other.
Omni-Weather also demonstrates the feasibility and value of unifying weather generation and understanding.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents Omni-Weather, a multimodal foundation model that unifies weather generation and understanding within a single architecture. Unlike existing models that separately address forecasting or diagnostic reasoning, Omni-Weather integrates radar and text modalities through a shared self-attention backbone and a Chain-of-Thought (CoT) dataset to enable causal reasoning in weather modeling. The model achieves state-of-the-art results on both weather generation (e.g., nowcasting, radar inversion) and understanding (e.g., RadarQA tasks), demonstrating that generative and interpretive capabilities can reinforce each other.

The contributions are:
1. Introduction of the first unified multimodal foundation model for weather that jointly handles generation (forecasting, inversion) and understanding (diagnostic reasoning, QA) tasks within a single framework.  
2. Construction of a weather-specific Chain-of-Thought (CoT) dataset for causal reasoning in generation, improving interpretability and perceptual quality of outputs.  
3. Empirical results showing Omni-Weather surpasses strong baselines (e.g., CasCast, DiffCast, WeatherGFM, RadarQA) in both pixel-level and perceptual metrics, with reasoning further enhancing visual fidelity and explainability.

### Strengths
1. The paper introduces the first unified multimodal foundation model for weather generation and understanding, representing a novel and impactful problem formulation.  
2. The Chain-of-Thought dataset for causal reasoning in weather generation is promising, enabling interpretable forecasting and bridging the gap between prediction and explanation.  
4. The experiments are comprehensive, covering both pixel-level and perceptual evaluations with clear comparisons to strong baselines.  
5. The paper is well-written and clearly structured.
6. The demonstrated mutual benefit between generation and understanding tasks highlights significant scientific insight with implications for broader multimodal foundation model research.

### Weaknesses
1. The claim of a “foundation model for weather” seems overstated, as the model’s scope is limited to a single variable (radar VIL precipitation) rather than encompassing multiple atmospheric variables such as temperature, pressure and wind.  
2. The proposed model only addresses short-range nowcasting (approximately one hour ahead) and is restricted to the SEVIR dataset covering the continental US, limiting its generalization and global applicability.  
3. The Chain-of-Thought (CoT) dataset used for training is entirely LLM-generated, with no human expert validation or meteorological review to ensure physical correctness, as GPT-series models are not fine-tuned as meteorologist experts.
4. The CoT generation pipeline relies on GPT-4o for attribute annotation and GPT-o3 for reasoning synthesis, producing synthetic causal narratives that may not reflect authentic meteorological reasoning. An ablation or qualitative comparison between LLM-generated CoT reasoning and human meteorologist-written reasoning would clear the confound of whether the improvements stem from genuine interpretability or stylistic mimicry of GPT.

### Questions
Please refer to weaknesses.

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
This paper introduces Omni-Weather, a multimodal foundation model designed to address a significant gap in radar modeling: the separation of generation (numerical prediction) and understanding (textual interpretation). The authors propose a single architecture that unifies these two capabilities, arguing that they are mutually beneficial. The model's core contributions are its unified architecture, the introduction of a novel Chain-of-Thought (CoT) dataset for causal reasoning in weather, and its demonstration of strong performance on both task categories.

### Strengths
- A multimodal model is a great direction towards briding the gap between numerical prediction tasks and high-level textual interepretations/analyses.
- The framework is well motivated and is at the forefront of such multimodal models in this weather/radar domain.
- Clearly writing
- Evidence that joint training/multimodality provides complementary supervision signals and better scores in some areas that just a single modality.

### Weaknesses
1. The considered data is exclusively radar. Weather in the title makes it sound overly general. As the authors point out, there would be signifcant challenges in even just extending this framework to more general weather-related tasks/dataset. Thus, I suggest writing OMNI-Radar and replacing most occurrences of weather with radar in the text. Similarly, the term "foundation model" in the title feels premature; this needs to urgently be renamed and the text revised to accurately reflect the true contributions of the work.
2. Lack of clarity/details in some places. For example:
- Unclear how encoders are trained and what their specific designs are (beyond high-level descriptions like "VAE decoder")
- What's high-value retaining/matching?
- eq 3.4 feels very abrupt... did some related sentences go missing?
- $\lambda_t$ is poorly explained/introduced. Multiplying both loss terms in Eq. 3.4 by $\lambda_t$ doesn't make sense. Please correct. Also, please explain how it was tuned (same for $n_t$).
- This claim should be toned down: *"On the radar inversion task, Omni-Weather consistently surpasses both specialized... and generalist... models, achieving higher CSI scores across all thresholds, with gains up to 20% at high-value levels."* given that it's not true for the RMSE metric.
- How's the CRPS computed? How many ensemble members are used?
- Fig. 3: Full prompts should be included in appendix. Same for exact versioning of GPT models used
- I'm confused by the "CFG Setting" (classifier free guidance) paragraph. There's no reference to CFG, not even diffusion, anywhere else... did the authors use it but forgot to mention it in the main text?
3. No discussion of the complexity of the model, especially when compared to the "generation-only" baselines
4. More comprehensive evaluations would be useful. E.g.:
- Human expert evaluation of "understanding" outputs would be really useful and strong contribtuion. Are the explanations at the level of a meteorology expert? How useful are they actually? Are the textual outputs given by the model consistent with the numerical nowcasts (e.g., in fig. 4)? With the current results, it's hard to judge how scientifcally useful the "understanding" part of the model actually is.
- How's the RMSE in Table 2 computed? A more comprehensive ablation (e.g. like the part of table 1 that's about radar nowcasting) would be more useful.
5. While the paper is at the forefront of multimodal modeling for weather/radar, it's not there all by its own. The paper misses some important references and contextualization. In particular, 1) this paper is only *one* of the first multimodal models in this weather/radar domain [1], 2) There's been a benchmark proposed in this space, which includes SEVIR (the only weather dataset used on this paper) [2]. It would have been nice to use it here, but at least it should be discussed.

Minor:
- VIL should be explained before using its abbreviation form.
- cascast is misspelled in Fig. 5

[1] Aquilon: Towards Building Multimodal Weather LLMs; Varambally et al. 2025 (https://openreview.net/forum?id=KVxOwEYAF4)

[2] CLLMate: A Multimodal Benchmark for Weather and Climate Events Forecasting; Li et al. EMNLP 2025 (https://arxiv.org/abs/2409.19058)

### Questions
- Why are so many different encoder/decoder's used? E.g., why are separate single-frame radar and multi-frame radar encoder needed?
- *"In the radar nowcasting task, forecasts exhibit fine-grained storm details with improved spatial coherence"*... I'm not sure how the authors identify "improved spatial coherence" in Fig. 5?
- Is there anything special about the extra Metaquery data that would make it particularly useful or do you think your model would benefit from any other extra, not radar specific, data? Table 5 seems to explore this a bit, but it's unclear what the two possible "gen" datasets are and why adding the 2nd "gen" dataset is so detrimental to performance.
- Why not report CRPS for radar inversion task?

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
This paper proposes OmniWeather, a weather understanding and forecasting model that aims to perform three main tasks: (i) radar inversion (ii) radar understanding (iii) radar nowcasting. The authors finetune the multimodal Bagel on their CoT dataset, and benchmark their method against unimodal models for these three tasks.

### Strengths
1. The authors train a single unified model that can reason across both images and text, and effectively produce interpretable forecasts. As far as I know, this is the first work that combines weather generation and understanding in the same model.
2. The proposed model achieves strong results on all three considered tasks.
3. The ablations are interesting, and shed light on the different steps of the model pipeline.

### Weaknesses
1. My primary concern is that the paper overstates its scope and significance. The definition for a foundation model from [1] is "A foundation model is any model that is trained on broad data (generally using self-supervision at scale) that can be adapted (e.g., fine-tuned) to a wide range of downstream tasks;". While the paper certainly achieves impressive results in unifying different modalities, labeling the model a foundation model feels premature given that it is fine-tuned for only three task types on a limited data regime. I would recommend the authors to soften the claims in the introduction and abstract. I would also recommend a title change that better reflects the scope of the tackled problem. For example, a title with some combination of the words "unified multi-task model for short-range weather understanding and generation".

2. The work is missing several important citations and discussions related to short-range/medium-range weather forecasting, e.g.  GenCast [2], Stormer [3], Pangu-Weather [4], Aurora [5], Prithvi WxC [6]. These are the canonical exemplars readers associate with large-scale weather pretraining/foundation model claims. Even if the focus is nowcasting, the paper should explicitly contrast goals, data scope, and evaluation scales with these systems. The authors should also compare their model against the important now-casting work [7] to better situate progress within the nowcasting literature.


3. From my understanding, the authors use GPT-4o (Appendix A.4) to annotate radar data and identify important phenomenon from the images. I am concerned that this process might be error-prone and introduce mistakes that might propagate into the training process. Do the authors benchmark 4o annotations against a gold standard (for example, expert human)? How reliable is this data annotation process?

The manuscript also needs a precise description of the quality-control (QC) stages—currently “Structure Check, Causal Alignment, and Terminology” are named but not operationalized.


### References
[1] Bommasani, Rishi. "On the opportunities and risks of foundation models." arXiv preprint arXiv:2108.07258 (2021).

[2] Price, Ilan, et al. "Gencast: Diffusion-based ensemble forecasting for medium-range weather." arXiv preprint arXiv:2312.15796 (2023).

[3] Nguyen, Tung, et al. "Scaling transformer neural networks for skillful and reliable medium-range weather forecasting." Advances in Neural Information Processing Systems 37 (2024): 68740-68771.

[4] Bi, Kaifeng, et al. "Pangu-weather: A 3d high-resolution model for fast and accurate global weather forecast." arXiv preprint arXiv:2211.02556 (2022).

[5] Bodnar, Cristian, et al. "A foundation model for the Earth system." Nature (2025): 1-8.

[6] Schmude, Johannes, et al. "Prithvi wxc: Foundation model for weather and climate." arXiv preprint arXiv:2409.13598 (2024).

[7] Ravuri, Suman, et al. "Skilful precipitation nowcasting using deep generative models of radar." Nature 597.7878 (2021): 672-677.

### Questions
Apart from the main issues flagged in the Weaknesses section, I have other minor comments/questions/suggestions.

1. The current description of CoT data annotation and Figure 4 are cluttered and hard to follow. The authors should consider simplifying it, or replacing it with a figure that reads top-to-bottom.
2. Why do the authors use the word "causal"? For example, the prompt in Appendix A.4 asks the model to extract "Temporal causal factor, perceptual causal factor" without any sufficient explanation of what this means. How do we trust that the model knows the true "causal" factors for explaining these weather phenomenon?
3. Lines 193-197 do not add any substantive value in explaining the problem setup and should either be replaced by a more complete mathematical description of the problem setup or omitted entirely.
4. There are insufficient architectural details about the VAE used in the radar inversion task, and these details should be added to the manuscript.
5. The clarity in Figure 3 could be improved. In particular, it is unclear how the tokens from the different modalities are combined in the model architecture.
6. Line 214: modal -> model
7. The authors need to add more details about how many data samples are used for training. While the authors mention that they generate 4000 CoT samples for radar nowcasting and 4,000 CoT annotations for radar inversion, the authors should also clarify the number of samples used from RadarQA, and the general metaquery data.


Overall, I think this is a substantial and promising paper marred by some fixable issues. I would be willing to raise my score if the authors can satisfactorily address my concerns.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper introduces Omni-Weather, a unified multimodal foundation model that brings weather generation and understanding into the same architecture. The authors also create a chain-of-thought dataset tailored for causal reasoning in generation and use it for finetuning and "thinking" inference. They show strong (often SOTA) results across nowcasting, radar inversion, and radar understanding, and provide evidence that training generation and understanding together lets the two enhance one another. Ablations further indicate that mixing scientific and general data boosts performance, especially on deterministic and perceptual metrics.

### Strengths
(1) This paper introduces a multimodal foundation model that unifies weather generation and understanding within one architecture, using modality-specific encoders, and takes a step toward reasoning-capable unified foundation models for weather.

(2) They present experiments and ablations with useful insights, showing how generation and understanding tasks can mutually enhance each other.

(3) They demonstrate strong results across nowcasting, radar inversion, and radar understanding, often matching or exceeding state-of-the-art models.

### Weaknesses
(1) As mentioned in the limitation section by the authors, the model cannot yet adapt to general-domain VAEs.

(2) It would strengthen the paper to include a small human-validation study with weather experts. In particular, having domain experts rate the generated reports/explanations, and comparing those ratings to the LLM-based judge.

(3) Results are centered on SEVIR-style radar nowcasting, satellite-to-radar inversion, and RadarQA understanding, and generalization to other weather tasks is not demonstrated.

### Questions
(1) It is mentioned that there is a quality verification step to produce the final CoT dataset, including causal alignment, structure checks, etc. Is there human/expert validation at any point during the dataset generation or evaluation?

### Soundness
3

### Presentation
3

### Contribution
3
