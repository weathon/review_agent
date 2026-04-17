# Spatial-DISE: A Unified Benchmark for Evaluating Spatial Reasoning in Vision-Language Models

- Decision: Accept (Poster)
- Scores: 4, 4, 4, 4

## Abstract
Spatial reasoning ability is crucial for Vision Language Models (VLMs) to support real-world applications in diverse domains including robotics, augmented reality, and autonomous navigation. Unfortunately, existing benchmarks are inadequate in assessing spatial reasoning ability, especially the \emph{intrinsic-dynamic} spatial reasoning which is a fundamental aspect of human spatial cognition. In this paper, we propose a unified benchmark, \textbf{Spatial-DISE}, based on a cognitively grounded taxonomy that categorizes tasks into four fundamental quadrants: \textbf{I}ntrinsic-\textbf{S}tatic, Intrinsic-\textbf{D}ynamic, \textbf{E}xtrinsic-Static, and Extrinsic-Dynamic spatial reasoning. Moreover, to address the issue of data scarcity, we develop a scalable and automated pipeline to generate diverse and verifiable spatial reasoning questions, resulting in a new \textbf{Spatial-DISE} dataset that includes Spatial-DISE Bench (559 evaluation VQA pairs) and Spatial-DISE-12K (12K+ training VQA pairs).
Our comprehensive evaluation across 32 state-of-the-art VLMs reveals that, current VLMs have a large and consistent gap to human competence, especially on multi-step multi-view spatial reasoning. Spatial-DISE offers a robust framework, valuable dataset, and clear direction for future research toward human-like spatial intelligence. Benchmark, dataset, and code are available at https://shinmohuang.github.io/spatialdise_page/Spatial-DISE .

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces Spatial-DISE, a new benchmark and dataset designed to evaluate the spatial reasoning capabilities of VLMs, along with the data generation pipeline. The authors argue that existing benchmarks are inadequate, often focusing on static scenarios and lacking a systematic, cognitively-grounded framework. To address this, they propose the DISE taxonomy, which categorizes spatial reasoning tasks into four quadrants based on two dimensions: Intrinsic vs. Extrinsic and Static vs. Dynamic. Experiment results show that current models struggle in mental simulation.

### Strengths
1. Comprehensive empirical evaluation. The paper presents an extensive and rigorous evaluation across 28 distinct Vision-Language Models. The selection of models is commendable, covering a wide spectrum of architectures including proprietary APIs, open-source foundation models, and specialized models fine-tuned for reasoning. The experiments with fine-tuning in Section 4.3 are interesting, where performance gains on in-domain tasks come at the cost of generalization to other spatial benchmarks.
2. Cognitively grounded and unified taxonomy, which provides a comprehensive analysis framework in analysing spatial reasoning performance of VLMs. 
3. The paper is well-written, with rich error analysis.

### Weaknesses
1. Fail to disentangle perceptual confounders from reasoning deficits. While the paper attributes model failures primarily to reasoning, as far as I'm concerned, the analysis may not fully account for more fundamental perceptual challenges that act as confounding variables. The evaluation format often presents a complex multi-panel image containing both the question pattern and several graphical options. A model's failure could stem from an inability to correctly understanding this layout which is a visual parsing issue, not a spatial reasoning deficit (which I believe is not really the perceptual error and comprehension error in L447. Please correct me if I'm wrong). This potential confounder weakens several key conclusions:
* The claim about the relationship between static and dynamic reasoning could be affected.
* The "catastrophic forgetting" observed in Section 4.3 could be reinterpreted. The performance drop on other benchmarks might not be a loss of reasoning ability, but rather a failure to generalize to different visual formats and layouts on the perception side. 
2. Concerns in task categorization. While the DISE taxonomy is well-founded, the mapping of specific tasks to its quadrants can be ambiguous. A task may not be "cognitively pure" and could be solvable via multiple reasoning strategies that cross quadrant boundaries. For example, 3D Shape Finding is classified as Intrinsic-Static, implying a logical deduction about a fixed object's properties. However, a viable and intuitive strategy to solve this task involves mentally rotating the cube (an Intrinsic-Dynamic process) to build a complete 3D mental model before identifying the missing face. This ambiguity raises concern regarding the diagnostic clarity of the benchmark. If a model fails this task, it is unclear whether the failure lies in static deduction or dynamic mental simulation, which in turn limits the precision of the paper's conclusions about specific cognitive weaknesses.
3. Lack of deeper analysis on model architecture. The evaluation is comprehensive in its breadth of models but could be deeper in its analysis of architectural influences. The paper groups models by family or purpose (e.g., "reasoning models") but does not delve into how specific architectural choices (e.g., type of vision encoder, cross-attention mechanism, size of the vision model vs. language model) might correlate with performance on the different DISE quadrants. (minor comments that are not factored into my score fyi)

### Questions
See as above in weaknesses. I'm happy to adjust the scores if the author can address the three concerns in the weaknesses. 

Typo: missing . in L433

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper presents Spatial-DISE, a unified benchmark for evaluating spatial reasoning in vision-language models (VLMs). It introduces a cognitively inspired taxonomy, DISE (Intrinsic/Extrinsic × Static/Dynamic), to categorize spatial reasoning tasks. Using a three-stage pipeline combining real-world, synthetic, and human-validated samples, the authors construct a diverse dataset and evaluate 28 VLMs against human performance (76.8%). Results reveal large performance gaps, especially in dynamic and extrinsic reasoning. A fine-tuning study shows that models can improve on Spatial-DISE but suffer from catastrophic forgetting. The paper also offers a cognitive error taxonomy analyzing model failures.

Main Contributions
- A cognitive taxonomy (DISE) for structuring spatial reasoning evaluation.
- A large, human-verified benchmark dataset combining real, synthetic, and controlled samples.
- A comprehensive evaluation of 28 leading VLMs with human baselines.
- Error taxonomy and analysis revealing systematic weaknesses in spatial reasoning.

### Strengths
1. Clear Cognitive Taxonomy
- The proposed DISE framework (Intrinsic/Extrinsic × Static/Dynamic) is cognitively grounded, offering interpretability which is good for benchmark design.
- This taxonomy provides a systematic structure that unifies fragmented spatial reasoning benchmarks.

2. Cognitive Error Analysis
The error taxonomy (Perceptual / Comprehension / Reasoning errors) and the subtypes (Rule Application, Mental Simulation, Holistic–Local Failure) are both novel and psychologically informed.

3. Human Benchmarking
They include 54 human participants to establish a strong baseline.

### Weaknesses
1. Fine-tuning and Generalization Analysis Could Be Deeper
- The fine-tuning experiments (Qwen2.5-VL, SpaceOm) show interesting trends but lack representation analysis — e.g., why forgetting occurs, or what cognitive dimension the gains are concentrated in.

2. Visual Accessibility
- Figures are a bit too dense (figure 1, 2, 3) and may not fully communicate the DISE framework or task intuitions clearly to non-specialists.

### Questions
1. Beyond taxonomy and scale, what new reasoning capabilities does this benchmark test?
2. Is there evidence of cross-domain transfer? For example, does training on Intrinsic tasks improve performance on Extrinsic reasoning, or are these dimensions fully independent?

I think overall it's a good benchmark. I would consider raising my score if the questions and weaknesses are well-addressed.

### Soundness
3

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
3

### Summary
The paper proposes two semi-synthetic datasets (Spatial-DISE Bench and Spatial-DISE-12K) to evaluate spatial reasoning in Vision-Language Models (VLMs), with a special emphasis on intrinsic and dynamic tasks that are missing from previous benchmarks. The datasets are created using a new data generation framework based on the graphics software Blender to generate complex 3D scenarios that require spatial reasoning to solve. The work also offers a comprehensive evaluation of 28 state-of-the-art (SotA) VLMs on the proposed data showing that current models perform much worse than human subjects.

Despite some good contributions (i.e., evaluation study), I am really not sure that creating new datasets and the entire data generation pipeline (which are claimed to be key contributions) was at all needed in the first place. Given the way the problem is framed (see Table 1), one could combine pre-existing benchmarks to obtain all combinations of domains, sources, DISE tasks and large scale, completely bypassing the need to create a new data generation pipeline and new datasets.

The motivation behind the need for spatial reasoning in VLMs is also unconvincing in the current presentation. I am not saying it should not be done, but right now it is unclear why the community should be interested.

Because the above issues pertain to key contributions and the main motivation of the paper, I am leaning towards rejection. I am, however, open to clarifications from the authors in case I am misunderstanding those aforementioned aspects, in which case I would be willing to update my score.

### Strengths
-	Adoption of taxonomy to classify spatial tasks from the cognitive sciences.
-	Comprehensive evaluation of 28 SotA VLMs.
-	Human quality control of created datasets.
-	Human performances (54 participants) collected and included in the evaluation as a baseline.
-	New data generation framework based on Blender that can generate complex spatial reasoning scenarios.

### Weaknesses
-	Several critical parts motivating this work are unclear or questionable, which makes me seriously concerned as to whether this work addresses a genuine gap in the literature (more below).
-	Many important details in the methodology (dataset creation, error analysis) are unclear or missing (more below).
-	Any discussion about potential limitations of the work is completely missing.

### Questions
**Questions to authors**

-	[**critical**] Why was it necessary to create Spatial-DISE datasets? Can’t we combine all existing benchmarks referenced in Table 1 to get a large-scale meta-benchmark that covers all domains and DISE tasks?
-	[**critical**] Why VLMs capability for sophisticated dynamic spatial reasoning is important beyond the fact that humans can do it? Can you give some detailed examples and elaborate on their importance?
-	[**critical**] The last paragraph in Section 2 discusses latest work that also covered most or all DISE tasks, but mentions verifiability as an important distinctive factor. Can you elaborate on the importance of verifiability and why it was so critical that it warranted creating a new data generation pipeline?
-	Why was there a need to create two datasets (Spatial-DISE Bench and Spatial-DISE-12K) and not just one? Do they serve different purposes?
-	Section 3.2 Stage 1 mentions 1180 VQA pairs, but then Spatial-DISE Bench (which is 53% “wild” data) has only 559 samples? Where does the 559 come from?
-	Were all 12,000 generated VQA pairs manually verified by humans?
-	Can the proposed data generation framework generate only 3D scenarios? Can it generate 2D examples?
-	Is the error analysis done with Doubao-1.6-thinking free from any potential mistakes? That is, can the model make mistakes (e.g., miscategorise other models’ errors) and mislead the error analysis?
-	In Section 5, why only a subsample (200) of incorrect responses was used and not 100%?

**Additional feedback**

-	I do not think that claiming the DISE taxonomy as novel is accurate as it was derived from (Maier, 1996; Uttal et al., 2013). Authors should make it clear that this taxonomy has been adopted from previous work. For example, first contribution in line 98 calls it novel.
-	Calling non-synthetic data “wild” seems very uncommon and quite confusing. Using a more common term “real data” or “real-world data” would resolve the confusion. If you keep using an uncommon term, I would suggest to explain it early in the text.
-	Most of the code in the algorithm listings (Algorithms 1-5) is very hard to interpret as the code mostly consists of function names that are not explained anywhere. Are those functions built-in procedures in Blender?
-	The paper uses phrases like “mental rotation”, “mental transformation” and alike throughout. I am not sure the word “mental” is the right choice as it generally refers to the mind, which I do not think VLMs have. Words like “simulation” or “hypothesis” might be more accurate.
-	Verifiability is being mentioned very often throughout the text, but its importance to this work is never properly explained.
-	Line 68: “have limited scopes” -> “are limited in scope”.
-	Line 69: “multi-steps” -> “multi-step”.
-	Line 83: “The Spatial-DISE” -> “Spatial-DISE”.
-	Lines 89-90: mentions “Spatial-DISE Bench” for the first time without explaining what it is.
-	I do not think the first paragraph of Section 3 is at all needed.

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
4

### Summary
This paper introduces Spatial-DISE, a new benchmark for evaluating spatial reasoning in VLMs. The work leverages a $2\times2$ cognitive taxonomy (Intrinsic/Extrinsic vs. Static/Dynamic) to structure the benchmark. A key contribution is a scalable Blender-based pipeline for generating a 12K-pair training dataset and a 559-pair evaluation bench. The authors perform a comprehensive study on 28 VLMs, finding that current models perform poorly, especially on dynamic tasks, and that failures are rooted in reasoning (e.g., rule application, mental simulation) rather than perception.

### Strengths
1. The adoption of the DISE cognitive taxonomy provides a structured and theoretically-grounded framework for evaluating spatial reasoning, moving beyond ad-hoc task collections.

2. The Blender pipeline is a valuable engineering contribution, enabling the verifiable and programmatic generation of complex 3D spatial reasoning tasks, which are difficult to source at scale.

3. The empirical study is extensive, testing 28 SOTA models. The resulting error analysis (Section 5) provides a clear insight: the primary bottleneck for VLMs is cognitive reasoning, not visual perception.

### Weaknesses
1. **Marginal Novelty in a Crowded Field**: The paper's own related work (Table 1) demonstrates this is an extremely crowded and concurrent field (e.g., BSA, OmniSpatial, BLINK, SPACE). I believe it is necessary to demonstrate that this dataset is fundamentally distinct from existing ones and possesses intrinsic value.

2. **Dataset Utility is Questionable**: The paper presents the Spatial-DISE-12K dataset as a major contribution for future training. However, the paper's own finetuning analysis (Section 4.3, Table 3)  demonstrates this dataset may be flawed. While SFT on Qwen2.5-VL-7B improves performance on the in-domain Spatial-DISE benchmark (+23.6pp), the SpaceOm model shows "catastrophic forgetting" on a general benchmark like CVBench (a -32.4pp drop) . This strongly suggests the 12K dataset lacks diversity and induces severe overfitting to the specific generative patterns of the pipeline.

3. **Limited Evaluation**: The model set omits several state-of-the-art commercial VLMs (e.g., GPT o3/5, Google Gemini 2.5 pro), which weakens the headline claim about “current VLMs.” They have more robust and powerful ability.

### Questions
I am curious about the true value of the dataset, as I strongly suspect that it may merely overfit to its own format.

If a base model (e.g., Qwen-VL) could be fine-tuned on this dataset and subsequently demonstrate performance gains on other benchmarks (such as VSI-Bench, OmniSpatial and SPACE), I would be much more inclined to recognize the dataset’s contribution.

### Soundness
2

### Presentation
2

### Contribution
1
