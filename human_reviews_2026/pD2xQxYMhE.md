# Learning Human-Perceived Fakeness in AI-Generated Videos via Multimodal LLMs

- Decision: Reject
- Scores: 6, 4, 8, 4

## Abstract
Can humans identify AI-generated (fake) videos and provide grounded reasons? While video generation models have advanced rapidly, a critical dimension -- whether humans can detect deepfake traces within a generated video, i.e., spatiotemporal grounded visual artifacts that reveal a video as machine generated -- has been largely overlooked. We introduce DeeptraceReward, the first fine-grained, spatially- and temporally-aware benchmark that annotates human-perceived fake traces for video generation reward. The dataset comprises 4.3K detailed annotations across 3.3K high-quality generated videos. Each annotation provides a natural-language explanation, pinpoints a bounding-box region containing the perceived trace, and marks precise onset and offset timestamps. We consolidate these annotations into 9 major categories of deepfake traces that lead humans to identify a video as AI-generated, and train multimodal language models (LMs) as reward models to mimic human judgments and localizations. On DeeptraceReward, our 7B reward model outperforms GPT-5 by 34.7% on average across fake clue identification, grounding, and explanation. Interestingly, we observe a consistent difficulty gradient: binary fake v.s. real classification is substantially easier than fine-grained deepfake trace detection; within the latter, performance degrades from natural language explanations (easiest), to spatial grounding, to temporal labeling (hardest). By foregrounding human-perceived deepfake traces, DeeptraceReward provides a rigorous testbed and training signal for socially aware and trustworthy video generation.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work introduces DeeptraceReward, a benchmark and modeling framework designed to capture how humans perceive “fakeness” in AI-generated videos. The authors argue that while video generation quality has improved rapidly, little attention has been given to understanding which specific visual or temporal cues make humans recognize a video as machine-generated. To address this, they collect over 4,300 fine-grained annotations across 3,300 generated videos, each including a natural-language explanation, spatial bounding box, and temporal boundaries of the perceived fake trace. From these annotations, they derive nine major categories of deepfake traces and train multimodal large language models (MLLMs) as reward models to mimic human perception. Their 7B-parameter model surpasses GPT-5 by 34.7% on average across three key dimensions: identifying fake clues, localizing them spatially, and explaining them linguistically. Notably, they uncover a clear hierarchy of task difficulty—binary fake vs. real classification is easiest, followed by explanation, spatial grounding, and finally temporal labeling as the hardest. Overall, the paper provides both a novel dataset and a methodology for modeling human-perceived authenticity in AI-generated videos, advancing research on interpretable and socially aware video generation systems.

### Strengths
1. The purpose of DEEPTRACEREWARD is highly in line with the current development needs in video generation field and will be definitely helpful to either improve authenticity of AI generated videos or intelligently identify whether a video is real or AI generated.
2. The paper clearly exposes a limitation of state-of-the-art MLLMs such as GPT-5 and Gemini-2.5-Pro—their inability to spatially and temporally ground fine-grained deepfake traces despite being able to classify videos as fake. The proposed 7B reward model significantly outperforms these baselines, laying a strong foundation for future research in human-aligned video evaluation.
3. The annotation pipeline is well designed and rigorously executed. For instance, the use of LabelBox for structured labeling, meticulous frame-by-frame inspection by annotators, and GPT-4-based standardization of textual explanations all contribute to a reliable and reproducible dataset.
4. The multi-dimensional evaluation metrics effectively demonstrate performance gaps across MLLMs, making the analysis comprehensive and interpretable.

### Weaknesses
1. Lack of detailed explanations about the construction process of prompts used to text-to-video generation, which is an indispensable part in such benchmarks related to AI generated videos.
2. Some strong spatial-temporal grounding baselines (e.g., LLaVA-ST, SpaceVLLM) are missing. Given that the defined task combines deepfake recognition and spatial-temporal localization, inclusion of such models would provide a fairer and more informative comparison.
3. The ablation study omits a “w/o bounding box” condition. Since the paper analyzes the effects of time and explanation labels, the contribution of bounding box annotations should also be examined for completeness.

### Questions
1. Why do SOTA MLLMs (e.g., GPT-5, Gemini-2.5-Pro) perform well on binary real/fake classification but struggle with fine-grained grounding? Without explicit trace localization, how do these models internally judge fakeness? 
2. Why do nearly 40% of the fake videos not have explanation about how they could be identified as fake videos? Does this mean these videos are easy to discern but hard to give corresponding explanations under the defined evaluation framework?
3. Why does training the model w/o explanation do good to metrics like fake accuracy, bounding box distance and time distance? Is there any interpretation for this kind of contrastive effect between “time” and “explanation” labels?
4. When multiple deepfake traces occur at different timestamps in a single video, how did you determine which label and explanation to use for that instance?
5. You mention summarizing nine movement-centric categories after analyzing annotations, yet Figure 3 seems to define tags before annotation. Could you clarify this potential inconsistency?

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
This paper introduces DEEPTRACEREWARD, a benchmark and set of reward models for human-perceived fakeness in AI-generated videos. Instead of only asking “is this video real or fake,” the authors argue that what really matters is whether humans can point to concrete, spatiotemporal traces that reveal the video as machine-generated (distortions, merging/splitting, sudden blur, odd trajectories, disappearing objects, etc.). They collect AI-generated videos from current SOTA T2V systems (Sora, Kling, Pika, MiniMax, Gen-3, Mochi, curated Sora demos) and annotate fine-grained “deepfake trace” instances with: (i) a natural-language explanation, (ii) a bounding-box over time, and (iii) start/end timestamps. They also pair these with an equal number of real videos from LLaVA-Video-178K for balanced training. They then train multimodal LLM-based reward models (on top of VideoLLaMA3 and Qwen2.5-VL) and show that, on their benchmark, all open/closed baselines (GPT-4.1, GPT-5, Gemini 2.5) are good at real/fake classification but bad at localizing and timing the actual fake trace, while their fine-tuned 7B model gets a big jump in overall score and can localize better.

### Strengths
1. This paper is easy to follow.
2. Timely, well-motivated task. Video gen is getting good enough that “why is it fake?” matters; the paper hits that exactly.
3. Fine-grained, human-aligned annotations. Bounding boxes + timestamps + NL reasons across thousands of videos, on current high-end generators, is useful.
4. The motivation is clear and experiments support authors' claim well.

### Weaknesses
1. Data-collection pipeline and training setup are not fundamentally new. The contribution is incremental in methodology aspects.
2. Evaluation partly depends on LLM-as-judge for explanations. A human eval slice would make the claim stronger.
3. Generalization to future generators is unclear. See more details in Questions.
4. Incomplete discussion of most recent related work such as [1,2]

Reference
1. How Far are AI-generated Videos from Simulating the 3D Visual World: A Learned 3D Evaluation Approach. 2025
2. BusterX: MLLM-Powered AI-Generated Video Forgery Detection and Explanation. 2025

### Questions
1. How robust is the model to unseen generators? If tomorrow’s videos come from, say, a Sora-level model with fewer motion artifacts but more material/lighting artifacts, will the trained reward model still localize well, or will it just say “REAL”? Test on latest models like Sora 2, Kling 2.x etc. would help strengthen the claim.
2. What is the intended release format? Are you releasing raw videos from Sora/Kling, etc., or just annotations + extracted features? This matters for reproducibility/licensing.
3. Happy to see more discussion with latest related work in the related area. The artifact detection capabilities of proposed method is similar to capabilities introduced in [1]. Showing whether the model can detect the physical and geometric artifacts from Sora shown in [1] would help define its capability boundary.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper introduces DEEPTRACEREWARD, a dataset designed to identify and analyze human-perceived flaws/anomalies in AI-generated videos. The authors created a dataset of over 4,300 annotations on 3,300 videos, where each inconsistency, or "deepfake trace," is documented with a natural language explanation, a bounding box, and precise start and end times. Using this data, they trained an MLLM as a baseline to act as a reward model that mimics human judgment in detecting these artifacts. This model significantly outperformed state-of-the-art systems like GPT-5 and Gemini 2.5 Pro, and the research revealed a clear hierarchy of difficulty: explaining a flaw is easier than grounding it in visual cues. And localizing where the fake part begins seems to be the hardest problem of all.

### Strengths
- The paper provides a much needed and timely expert annotated video dataset for AI-Generated video detection and reasoning, as it is largely missing in the literature. Most efforts have gone into image reasoning datasets, which has become largely insufficient in today's day and age.
- The scale of the dataset is also sufficient for future models to finetune on.

### Weaknesses
- The paper mentions that low-quality generated videos were filtered out. However, annotations for these low-quality videos are especially essential, because more often than not, the DeepFakes appearing in social media are low-quality. If a model is only trained on high-quality annotated videos, in practical deployment applications, this will result in a significant bias.
- All "real" videos were sourced from a single dataset (LLaVA-Video-178K). This raises a siginificant concern that the model (or any model trained on this data) just might be distinguishing between the sources of the data as a "shortcut" rather than actually grounding it's detection/reasoning on visual cues.  To prevent issues from domain gap, real videos could have been sourced just as easily from other publicly available licensed sources like FF++, CelebDF, etc.
- Since the main contribution of the paper is focused on providing expert annotations for the videos, the missing performance metrics like BLEU, ROUGE-L, CIDEr, etc. is a critical limitation. Without these NLG metrics it is hard to verify how well the trained model is actually reasoning.
- The dataset is also missing face-manipulated videos and is focused on T2V/I2V generators. Face-manipulated videos are nuanced since a large (spatial) chunk of the video is real and the only anomaly is in the face region. It will be extremely helpful to have some videos from this traditional DeepFake literature (for example videos from FF++, CelebDF, etc.).

### Questions
- How was consensus reached among annotators when there were disagreements on labeling an artifact? What was the protocol for resolving subjective ambiguities?
- The results show that temporal localization (predicting the start time of a fake trace) is the most difficult task for all models. What is the authors' hypothesis for why this is the case? Is it a limitation of current model architectures/training paradigms or an issue of ambiguous annotations? For example, a frame-by-frame model should be able to easily tell where the fake frame begins, as long as we assume it is a reasonably good (oracle) Deepfake detector.
- The ablation study indicates that removing supervision for explanations or temporal data can sometimes improve the model's performance on other metrics. Why might training on multiple tasks (e.g., localization and explanation) simultaneously cause this interference?
- A recent paper: PhysicsIQ [1] showed that Sora has the high visual realism and very low physical realism (i.e., following principles of gravity, fluid mechanics, etc.) in it's generated videos. Are the performance of the tested models on the annotated Sora videos available? It will be extremely valuable to see how much of a difference the provided annotations make on detecting or reasoning on Sora videos.
- How do the authors plan to keep the benchmark relevant as video generation technology rapidly evolves and produces new kinds of artifacts? Will it be updated periodically?
- The video data along with the explanations will be made publicly available for at least academic use, correct? And what is the expected timeline for the data release? This is crucial for the community.

[1] Motamed, Saman, et al. "Do generative video models understand physical principles?." arXiv preprint arXiv:2501.09038 (2025).

### Soundness
4

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
This paper addresses a critical and timely challenge in the evaluation of AI-generated video: moving beyond simple binary classification (real vs. fake) to understand why, where, and when humans perceive artifacts in synthetic content. 

I observe following contributions:

- A novel benchmark designed to capture "human-perceived deepfake traces." It comprises 3.3K high-quality generated videos sourced from seven SoTA models (including Sora, Kling, Pika) and 3.3K real videos. It features 4.3K fine-grained expert annotations, each including a natural language explanation, spatial localization (bounding boxes), and temporal localization (timestamps).

- The authors analyze these annotations to develop a taxonomy of nine major categories of movement-centric artifacts (e.g., Object Distortion, Object Merging, Sudden Blurring).

- An extensive evaluation of 13 MLLMs. The results show that while SoTA models may achieve high accuracy in binary classification, they perform poorly (below 37% overall) on the fine-grained grounding and explanation tasks.

- The authors demonstrate the dataset's utility by fine-tuning a 7B model (based on VideoLLaMA 3), which achieves 70.2% overall performance, significantly outperforming the baselines.

- The paper identifies a consistent difficulty gradient: binary classification is easiest, followed by explanation, spatial grounding, and finally, temporal localization (hardest).

### Strengths
As video generation models approach photorealism, the need for interpretable, localized, and explainable detection methods becomes paramount. This motivation makes the work relevant. This work provides a framework for understanding the specific failure modes of current generators.

- Novel and Important Task Formulation: The shift towards interpretable, grounded reasoning for deepfake detection is essential for trustworthy AI and for improving generation models via human feedback. This approach significantly advances the field beyond existing benchmarks like VBench (holistic quality) or DeMamba (binary classification).
- Valuable Dataset Contribution: DEEPTRACEREWARD is a meticulously constructed and highly relevant resource. The effort to collect expert annotations with multimodal grounding (text, space, time) across diverse SOTA generators is commendable. The focus on high-quality, motion-rich videos ensures the benchmark addresses the subtle, temporally dependent artifacts characteristic of cutting-edge models.
- *Insightful Taxonomy of Failures:* The categorization of the nine major artifact types (Sec 2.3, Figs 1 and 6) provides excellent insight into the current limitations of generative models, particularly regarding temporal consistency, physics, and object permanence (e.g., splitting, merging, disappearance).
- Good Empirical Validation and Analysis: The extensive benchmarking clearly highlights the deficiencies of current MLLMs for this task. The significant performance leap of the fine-tuned 7B model (from <37% to 70.2%) strongly validates the quality of the dataset as a supervisory signal. The analysis of the difficulty gradient (L424-430) is an important empirical observation, pinpointing temporal reasoning as a major bottleneck.

### Weaknesses
- Lack of reproducibility: The repeated reference to "GPT 5" and "GPT 4.1" (e.g., Abstract, Table 2, Section 3.1). The citation provided, (Achiam et al., 2023), refers to the original GPT-4 technical report. These names do not align with standard public nomenclature (e.g., GPT-4o, GPT-4 Turbo).

This ambiguity severely undermines the credibility and reproducibility of the baseline comparisons.
The central claim that the authors' 7B model "outperforms GPT-5 by 34.7%" (L025-026) cannot be verified. The authors must specify the exact model identifiers (e.g., API endpoints or version numbers) used in the experiments or even then these baselines are blackboxed (what if the authors were part of an A/B test?)

- Inflated Metric: The Overall score (L380) is an unweighted average of four metrics, including binary accuracy. As the paper's model achieves near-perfect binary classification (99.4%), a task acknowledged as easy (L27), this heavily inflates the overall 70.2% result. This masks the much lower performance on the novel grounding tasks (e.g., BBox IoU of 32.6%) and presents a misleading picture of the model's capabilities.

- Inadequate Temporal Localization: The "Time Distance" metric (L375) only measures the error in the starting timestamp, ignoring the duration of the artifact. A Temporal IoU metric, which considers the overall temporal overlap, is standard in video analysis and required here.

- The paper is heavily framed around providing a "Reward" signal for video generation (Title, Abstract and throughout the paper). However, the experiments focus entirely on video understanding/detection. While the trained model could serve as a reward model (e.g., for RLHF), this utility is hypothesized (L711) but not empirically demonstrated in the paper.

### Questions
- Why does the paper use GPT-4 to post-process and standardize the explanation text (L:213)? Does this not introduce LLM biases / randomness and undermines the premise that the paper is capturing human perception by effectively contaminating the ground truth?

- To add to above question: Please explain how does the Explanation score relying solely on LLM judgment (L357) substitutes for the focus on human perception (As per the research question in intro). Given the contamination of the ground truth by GPT-4, this metric may be evaluating alignment with LLM-standardized text rather than genuine human explanations.

- Who were the LabelBox experts?

- What was the inter-annotator agreement? It can be measured via Cohen's Kappa for categorization and average IoU for localization.

### Soundness
3

### Presentation
3

### Contribution
2
