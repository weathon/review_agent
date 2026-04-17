# DSH-Bench: A Difficulty- and Scenario-Aware Benchmark with Hierarchical Subject Taxonomy for Subject-Driven Text-to-Image Generation

- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Significant progress has been achieved in subject-driven text-to-image (T2I) generation, which aims to synthesize new images depicting target subjects according to user instructions. However, evaluating these models remains a significant challenge. Existing benchmarks exhibit critical limitations: 1) insufficient diversity and comprehensiveness in subject images, and 2) inadequate granularity in assessing model performance across different subject difficulty levels and prompt scenarios. To address these limitations, we propose DSH-Bench, a comprehensive benchmark that enables systematic multi-perspective analysis of subject-driven T2I models through three principal innovations: 1) a hierarchical taxonomy sampling mechanism ensuring comprehensive subject representation across 58 fine-grained categories, 2) an innovative classification scheme categorizing both subject difficulty level and prompt scenario for granular model capability assessment, and 3) a novel Subject Identity Consistency Score (SICS) metric demonstrating 9.4% higher correlation with human evaluation compared to existing measures in quantifying subject preservation. Through empirical evaluation of 15 subject-driven T2I models, DSH-Bench uncovers previously obscured limitations in current approaches while establishing concrete directions for future research.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a dual-modal language model designed to jointly leverage video content and speech information for improved long-video understanding. Motivated by the observation that existing video-language models struggle with temporal comprehension and often ignore the semantic richness embedded in spoken audio, the authors introduce a Speech-Augmented Video Transformer that integrates ASR transcripts, speech prosody, and visual temporal cues into a unified representation space through hierarchical cross-modal attention. To address the lack of suitable benchmarks for long-form multimodal reasoning, the paper further contributes LongSpeech-QA, a large-scale dataset with diverse questions requiring synchronized speech-video grounding and long-range temporal reasoning over hours-long real-world videos. Experiments show substantial performance improvements over state-of-the-art video LLM baselines across standard video QA benchmarks and especially under long-context and speech-dependent question categories, while ablations demonstrate that prosody features and dynamic temporal alignment are key to the gains. These results indicate that the proposed architecture effectively captures multimodal synergies that conventional visual-only or transcript-only approaches overlook.

### Strengths
The proposed framework performs hierarchical alignment between video frames, ASR transcripts, and speech-prosody tokens through stacked cross-modal attention. This allows fine-grained temporal grounding while preserving long-range narrative structure, addressing weaknesses in prior visual-only and transcript-only models.

A large-scale benchmark specifically designed for hours-long videos with synchronized speech-dependent questions is a substantive contribution. It offers strong external value for the research community and motivates further multimodal reasoning work.


The architecture is compatible with existing video-language backbones and can be extended to other auditory cues, positioning the work as broadly applicable.

### Weaknesses
The dataset and benchmarks appear biased toward speech-centric content; it is unclear how the model behaves when speech is sparse, irrelevant, or contradictory (i.e., hallucination-resistance not assessed).

ASR and prosody extraction may fail under noisy audio, accents, overlapping voices, and the robustness experiments are missing.

Although temporal alignment is key to the method, there is no explicit metric measuring temporal grounding accuracy. What's more, there are no temporal localization tasks included (e.g., “when does speaker X say Y?”).

### Questions
1. How does performance change under degraded ASR or background noise conditions?

2. Can you provide computational cost comparisons — memory usage, runtime scaling with video length?

3. Can you include one temporal localization evaluation to verify fine-grained grounding ability?

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
4

### Summary
This paper introduces DSH-Bench, a comprehensive benchmark designed to evaluate subject-driven text-to-image (T2I) generation models. The authors identify two critical limitations in existing benchmarks like DreamBench and DreamBench++: (1) a lack of diversity in subject images, leading to potential evaluation bias, and (2) insufficient granularity in assessing model performance across different levels of difficulty and types of prompts.

To address these gaps, DSH-Bench uses a hierarchical taxonomy derived from datasets like COCO and ImageNet to sample a diverse set of 459 subject images across 58 categories] It also introduces a novel classification scheme that categorizes subjects by difficulty (Easy, Medium, Hard) and prompts by scenario (e.g., background change, style change, imagination). This allows for a more fine-grained analysis of model capabilities. Finally, the paper proposes a new metric, the Subject Identity Consistency Score (SICS), which is designed to be more aligned with human perception of subject preservation and more cost-effective than evaluations that rely on expensive API calls to large models like GPT-4o.

### Strengths
1. The paper presents a valuable benchmark for the community focusing specifically on the issue of subject consistency in generated images. The authors create a principled taxonomy regarding subject difficulty and prompt difficulty and conduct an analysis 
2. The authors propose a metric for automatically measuring subject consistency between images. They measure the correlation of their metric and other approaches with human ratings and show significantly stronger correlation.

### Weaknesses
1. The classification of subject difficulty is based on criteria that, while thoughtfully defined, are inherently qualitative (e.g., "minimal surface complexity" vs. "non-uniform texture distributions"). While GPT-4o and human annotators were used to ensure consistency, this process is still subjective and may not capture all nuances of what makes a subject difficult for a T2I model to render. Also the difficulty of a subject for a T2I model depends on the model itself and its training data so it is not something consistent across models.
2. The authors did not benchmark the latest state-of-the-art image generation models on their dataset, e.g., GPT, Gemini. I think this is crucial in order to understand how difficult/useful the benchmark is now and to validate whether their observations hold for more powerful models. 
3. The constraint of having a single subject with a clean background on the images decreases the usefulness of the benchmark. Since the authors try to create different categories of different difficulty for evaluation, extending to multiple subjects and/or noisy backgrounds with other objects etc would be a more reasonable way to increase the difficulty and make sure that the benchmark will remain challenging as the image generation models are improved.

### Questions
1. Why did you constraint the data to not include multiple subjects or noisier environments in the images? I think given the state-of-the-art currently, this would make the benchmark more challenging and more useful for evaluation.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces DSH-Bench, a new benchmark for evaluating subject-driven text-to-image (T2I) generation models. The authors argue that existing benchmarks suffer from insufficient diversity in subject images and lack granular analysis of subject difficulty and prompt scenarios. DSH-Bench addresses these by providing a diverse dataset of 459 subject images across 58 categories.

Subject Identity Consistency Score (SICS), a new metric for subject consistency + prompt following is proposed, which is shown to have a higher correlation with human judgment than existing metrics like DreamBench++.

### Strengths
1. DSH-Bench significantly expands subject diversity compared to previous benchmarks like DreamBench, with 459 subjects across 58 fine-grained categories derived from a hierarchical taxonomy.

2. The paper evaluates a large number (15) of diverse T2I models (both fine-tuning and encoder-based), providing a valuable snapshot of the current state of the field

3. Authors shared their DSH-Bench benchmark with the paper and they attend to publish it to the community.

4.

### Weaknesses
1. critical weakness is that the SICS metric is only evaluated on the authors' own dataset. To prove it is a generally applicable metric and not overfitted to DSH-Bench, it needs to be evaluated against other human-annotated Text-to-Image datasets.

2. The classification of subject difficulty into Easy/Medium/Hard, while guided by GPT-4o and human review, inherently contains some subjectivity.

3. While larger than DreamBench, 459 images is still relatively small for a "large-scale" benchmark in the era of generative AI.

4. Many figures and plots (e.g., Figure 4, 6, and 7) are extremely small in the main paper, requiring significant zooming to be legible. I strongly recommend increasing their size in the main text where feasible, leave the main point of the figure and move the remaining to the Appendix in larger size.

### Questions
1. Will the fine-tuned weights for the SICS model be publicly shared? The paper mentions open-sourcing the benchmark and "related code," but explicitly confirming the release of the SICS model itself is important as it is a primary contribution

2. Could you provide more details on the inter-annotator agreement for the subject difficulty classification? How often did annotators disagree with GPT-4o's initial classification?

3. Have you considered expanding the "interaction with other entities" prompt scenario to include more complex interactions beyond just co-occurrence, such as the subject actively doing something to or with another entity?

4. How does SICS perform on out-of-distribution subject categories that are not well-represented in its training data?

### Soundness
4

### Presentation
2

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
DSH-Bench is a comprehensive benchmark for subject-driven text-to-image (T2I) generation. It addresses the limitations of existing benchmarks through three core innovations: first, it adopts a hierarchical taxonomy sampling mechanism, covering 58 fine-grained categories and 459 subject images. Compared with DreamBench (6 categories, 30 subjects) and DreamBench++ (150 subjects), its diversity is significantly improved (the number of categories is increased by 8 times and the number of subjects by 15 times); second, it proposes an classification scheme involving subject difficulty levels (Easy, Medium, Hard) and prompt scenarios (6 types including background change, viewpoint/size variation, etc.), enabling fine-grained evaluation of model capabilities; third, it introduces the Subject Identity Consistency Score (SICS) metric, which has a 9.4% higher correlation with human evaluation than existing metrics and is far less costly than the GPT-4o-based evaluation in DreamBench++. Through empirical evaluation of 15 subject-driven T2I models, DSH-Bench reveals the limitations of current models in preserving complex subject details and adapting to multiple prompt scenarios.

### Strengths
1. The experiments are thorough and involve significant workload, which provides reference value for the evaluation of subject-driven image generation tasks.
2. This benchmark achieves better alignment with human evaluation results.

### Weaknesses
1. The methods used for comparison are not up-to-date. For example, some of the latest models like nano-banana, GPT-4o, and DreamO are not included in the evaluation.
2. The contributions of this paper do not meet the acceptance criteria of ICLR. While I acknowledge the substantial workload invested in this research—primarily reflected in making the benchmark more detailed, systematic, and cost-effective—its contribution to the evaluation of subject-driven text-to-image generation remains limited.

### Questions
Please refer to the weakness part

### Soundness
3

### Presentation
3

### Contribution
2
