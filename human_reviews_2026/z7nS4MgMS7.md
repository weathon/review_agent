# Social Human Robot Embodied Conversation (SHREC) Dataset: Benchmarking Foundational Models’ Social Reasoning

- Decision: Reject
- Scores: 6, 4, 6, 4, 6

## Abstract
Our work focuses on the social reasoning capabilities of foundational models for real-world human–robot interactions. We introduce the Social Human Robot Embodied Conversation (SHREC) Dataset, a large-scale benchmark of 400 real-world human-robot interaction videos and over 10K annotations, capturing robot social errors, competencies, underlying rationales, and corrections. Unlike prior datasets focused on human–human interactions, the SHREC Dataset uniquely highlights the challenges faced by real-world embodied social AI agents, where robots lack innate social abilities such as emotion understanding, intention tracking, and conversational mechanics. Moreover, current foundational models struggle to recognize these deficits, which manifest as subtle, socially situated failures. To evaluate AI models’ capacity for social reasoning, we define eight benchmark tasks targeting critical areas such as (1) detection of social errors and competencies, (2) identification of underlying social attributes, (3) comprehension of interaction flow, and (4) providing rationale and alternative correct actions. Experiments with state-of-the-art foundational models, alongside human evaluations, reveal substantial performance gaps—underscoring the difficulty and providing directions in developing socially intelligent AI.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces the Social Human Robot Embodied Conversation (SHREC) Dataset, containing 10,353 annotations from 403 interaction videos spanning over 3,500 minutes. With this dataset, the paper developed eight tasks to measure the social reasoning capability of foundational models. Benchmarking results over 17 FMs alongside human evaluations reveal substantial performance gaps, underscoring the difficulty and providing directions in developing socially intelligent AI.

### Strengths
- The paper contributes one of the largest real-world human–social robot interaction benchmark datasets.

- The annotation seems of high quality and valuable to the community to evaluate social intelligence.

- The benchmark is comprehensive, and the analysis is solid and insightful.

### Weaknesses
- The "embodied" conversation seems not well-justified in the paper, which should relate more to the physical body interactions or self-awareness, or egocentric perceptions. "Situated" might be a better term for the scope of the proposed dataset.

- The dataset is curated from three existing sources, with no new data source introduced.

### Questions
- What is the "we observe a significant gap between nascent open-source vision-language models (VLMs) and even text-only LMs," in line 428-429?

- How are nonverbal questions fed into pure text models?

- How many human hours and money are cost for this dataset annotation project?

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
3

### Summary
The paper introduces SHREC, a real-world benchmark of ~400 human–robot interaction videos with10k+ annotations capturing social competencies, social errors, their underlying social attributes(seven categories), and rationales/corrections.

### Strengths
1. Eight well-defined tasks cover a full spectrum of social reasoning, from error and competence detection to rationale and correction generation.
2. The study benchmarks 17 leading LLMs/VLMs and shows a clear human–model performance gap, underscoring the benchmark’s challenge and value.
3. Independent annotations with ~91% agreement and multi-label structure ensure reliability while capturing the richness of real-world social interactions.

### Weaknesses
1. The paper’s so-called “real-world physical AI agents/robots” function more like voice-based chatbots equipped with camera access for visual input(1HZ), with interactions primarily limited to video and audio exchanges, rather than physical manipulation or embodied actions. Such a setup aligns more closely with a multimodal Vision-Language Model dataset rather than a genuinely embodied human-robot interaction dataset, where agents can physically engage with and influence their environment. Consequently, the proposed work appears to represent a multimodal VLM interaction benchmark rather than a truly embodied conversational framework. Similar multimodal datasets are already abundant in existing research, which somewhat limits the novelty of the contribution.
2. Although the paper claims to use FRESCO for high-fidelity face stylization to preserve social signals while protecting identity, Figures 10 and 11 still reveal discernible facial features, raising concerns about the adequacy of anonymization. Such visual cues could potentially allow re-identification, posing privacy risks. Moreover, it remains unclear whether all human participants appearing in the dataset provided informed consent for their images to be included and publicly shared. The absence of explicit ethical review or consent documentation may undermine the dataset’s credibility from an ethical and compliance standpoint.
3. All data are drawn from three prior deployments involving similar social robots and university-aged participants. The limited diversity of settings and demographics may restrict the generalizability of SHREC to other social contexts or robot morphologies.

### Questions
1. “Some scenarios that were too subjective to reach full consensus yielded some annotations in the Disagree category of 8.7%.” Such disagreement introduces additional subjectivity and ambiguity into the dataset, which may undermine the stability and interpretability of evaluation results. 
2. In Table 2, the performance of GPT-4o with few-shot prompting is notably weaker than that of GPT-4o alone, which appears counterintuitive. Typically, few-shot examples are expected to enhance model performance on novel tasks or at least maintain comparable results. This anomalous outcome may indicate potential inconsistencies in the dataset design or task formulation.
3. Reporting Cohen’s κ or Krippendorff’s α could further strengthen dataset's reliability claims.

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
3

### Summary
The paper presents SHREC Dataset. It is a large-scale benchmark designed to evaluate the social reasoning abilities of foundational AI models in real-world human–robot interactions. It includes over 400 videos and 10,000 annotations capturing social errors, competencies, and rationales in tabletop robot conversations. Results show that current models significantly lag behind human performance, highlighting the need for more socially intelligent AI.

### Strengths
1. Very thorough experiments on current language models and vision language models, clearly indicating the usefulness of the dataset.

2. SHREC fills a critical gap by focusing on real-world human–robot interactions, which are underrepresented in existing social reasoning benchmarks that typically center on human–human dialogue.

### Weaknesses
1. Counterintuitively, the human evaluation is not significant high: up and down 70%. The paper explains how the tasks are difficult, but I think there should be some kind of verification of the dataset's quality and interpretability by conducting expert-level human check.

2. It seems from the images shown (without video demos I cannot say for sure) that the videos are not as descriptive as general videos provide. The relatively small gap between language only models and VLMs also indicates this.

### Questions
1. Do you have any means to justify your dataset's quality and interpretability?

2. If a descriptive text on top of the transcript is provided, containing e.g. the actions taken by each human in the video, how will the language only models perform and how will human perform?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper investigates an important topic — enabling LLMs and VLMs to better understand subjective (social) content in videos. The authors define the problem through several sub-categories, collect and annotate the dataset themselves, and evaluate model performance under the designed scenarios. Based on these experiments, they draw conclusions about the models’ capabilities.

### Strengths
1. The problem addressed in this paper is both timely and valuable, with strong relevance to current model training practices and potential benefits for social good.
2. The authors have formulated the problem thoughtfully and designed their approach in a well-structured and deliberate manner.
3. The authors take appropriate care to preserve user privacy by re-generating faces using a generative model rather than using real user data directly.

### Weaknesses
1. There is a typo in Line 96: “n selecting” should be corrected.
2. The presentation quality can be improved, as it is currently difficult to follow. For example, the font size in the figures is too small, and each scenario description contains excessive text. The inclusion of clearer and more visually appealing figures would make the paper much easier to understand.
3. The model selection appears somewhat inconsistent and disorganized. For instance, the comparison table should be divided into two parts: one for open-source models and another for closed-source models. It is also unclear why the authors chose PaliGemma and Gemini-1.5-Flash but did not include Gemini-2.5-Pro. Since the chosen models are not the most recent ones, I am skeptical about the validity of the conclusions regarding the current capabilities of VLMs.

### Questions
1. In Line 99, the authors use “Hz.” Could the authors clarify the difference between “Hz” and “fps” in this context? From my experience, many VLMs also downsample videos to 1 fps for processing.
2. I am curious about how the authors handle ambiguity in interpretation. For example, in Line 145, the paper mentions that crying indicates sadness. However, crying can also occur in positive emotional contexts (e.g., an athlete crying with joy after winning an Olympic final). Do the authors’ methods account for such ambiguous cases?
3. Since the authors use a vision–language (V+L) model, I wonder whether audio information is also incorporated. If so, why not extend the framework to an omni-modal setting?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces SHREC, a new benchmark dataset designed to evaluate social reasoning in embodied AI agents, particularly social robots engaged in human–robot interaction (HRI). The dataset consists of 400+ real-world videos (~3,500 minutes) and 10,000+ human annotations, capturing social errors, competencies, rationales, and corrective actions across verbal and non-verbal channels.

It defines eight tasks probing four key aspects of social reasoning:

1.	Error and Competence Detection

2.	Social Attribute Identification

3.	Interaction Flow Reasoning (Pre/Post-condition inference)

4.	Rationale and Correction Generation

The authors benchmark 17 state-of-the-art LLMs and VLMs (including GPT-4o, Gemini, LLaMA, o1, etc.) and compare them with human baselines. Results show a persistent gap between model and human performance, highlighting social reasoning as a distinct and underexplored capability for foundation models.

### Strengths
1.	SHREC addresses a gap in multimodal social reasoning evaluation. Most prior work (e.g., Social-IQ, SocialIQA, MELD, ATOMIC) involves human–human data, whereas SHREC uniquely targets real-world human–robot interactions, which involve distinctive failure modes like delayed reactions or incorrect affective mirroring.
2.	Real, embodied setting – The dataset uses physically embodied tabletop social robots rather than simulated or text-based scenarios. The focus on embodied social reasoning makes it a valuable step toward practical social AI evaluation.
3.	Comprehensive task suite – The eight tasks cover multiple reasoning levels: detection (Error/Comp./None), attribution (social attributes), sequential reasoning (pre-/post-condition), and prescriptive inference (rationale and correction). This provides both diagnostic (which aspect fails) and generative (what to fix) signals for model improvement.
4.	Systematic benchmarking – Evaluating 17 models under consistent multimodal settings (text-only and vision+language) with human baselines strengthens the empirical value. By evaluating both chain-of-thought (CoT) and few-shot variants, the authors effectively probe how explicit reasoning and contextual examples influence models’ ability to infer social context and intent. 
5.	Attention to privacy and realism – The use of FRESCO for video anonymization while maintaining spatiotemporal cues (e.g., gaze, affect) reflects a well-considered approach to preserving critical social cues while ensuring participant privacy.
6.	The 91% inter-annotator agreement reflects that annotators consistently recognized and categorized nuanced social behaviors, supporting the reliability of the dataset’s labeling process.

### Weaknesses
1. **Unclear modality dependence** – It’s often unclear which tasks rely primarily on textual vs visual cues.
- For example, “Social Error, Competence, None Detection” seems largely text-driven, while “Social Attribute Identification” might require multimodal inputs (e.g., gaze, engagement). However, the paper doesn’t quantify modality contribution or ablate verbal-only vs visual-only performance.
- Similarly, some VLMs (e.g., paligemma, Llava-Next) perform worse than text-only models, which warrants a deeper discussion of why multimodality doesn’t help.

2. **Fine-tuning ambiguity** – It is unclear whether the models are zero-shot, few-shot, or fine-tuned for SHREC tasks.
- While Table 1 shows “few-shot” and “CoT” variants, it’s not stated whether any task-specific fine-tuning was performed or whether prompts were unified across models.
- Clarifying this would affect fairness and reproducibility of comparisons.

  
3. **Task redundancy/confusion** – The difference between Social Error, Competence, None Detection and Social Error Detection tasks is minor and not clearly motivated. The latter seems to be a subset of the former (binary vs ternary classification. It’s unclear if both are needed.

### Questions
- **Task modality**: Which tasks require video (vision) versus text (transcript) input? Could the authors provide per-channel ablation (vision-only vs text-only) to quantify the contribution of each?

- **Fine-tuning**: Were all models evaluated zero-shot, or was there any fine-tuning or prompt adaptation on SHREC? I wonder if finetuning on the dataset helps with any task.

- **Task redundancy**: What distinct insights are gained from both “Social Error, Competence, None Detection” and “Social Error Detection”?

### Soundness
3

### Presentation
3

### Contribution
3
