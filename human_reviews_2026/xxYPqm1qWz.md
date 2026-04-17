# VideoAgentTrek: Computer-Use Pretraining from Unlabeled Videos

- Decision: Accept (Poster)
- Scores: 8, 8, 4, 6

## Abstract
Training computer-use agents requires massive amounts of GUI interaction data, but manually annotating action trajectories at scale is prohibitively expensive. We present VideoAgentTrek, a scalable pipeline that automatically mines training data from publicly available screen-recorded videos, eliminating the need for manual annotation. Our approach addresses a key challenge: raw videos contain implicit demonstrations but lack explicit action labels. To solve this, we develop Video2Action, an inverse dynamics module (IDM) with two components: (1) a video grounding model that detects and localizes GUI actions with precise temporal boundaries, and (2) an action-content recognizer that extracts structured parameters like click coordinates and typed text. Applied to 39,000 YouTube tutorial videos, our pipeline generates 1.52 million interaction steps. We leverage this data through continued pretraining followed by supervised fine-tuning. On OSWorld-Verified, our approach improves task success rates from 9.3% (SFT-only baseline) to 15.8%, a 70% relative improvement. On AgentNetBench, step accuracy increases from 64.1% to 69.3%. Our results demonstrate that passive internet videos can be transformed into high-quality supervision for computer-use agents, providing a scalable alternative to expensive manual annotation.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
Paper introduces a scalable pipeline, VideoAgentTrek to automatically mine training data for use in training of computer-use agents.
The work aims to address the challenge costliness of manually annotating such training data at scale. It does so by exploiting the large amount of screen-recorded tutorials available online and introduces a means of extracting explicit structured action labels from such unlabeled videos.
To achieve this, the authors designed VIDEO2ACTION, a two component inverse dynamics module (IDM) consisting of a video grounding model and a action content recognizer.
The module first performs dense action-event detection to segment clips and assign action labels, then a action parameterization model analyzes these segments to produce structured parameters such as pointer coordinates, typed text etc.
Using 39000 YouTube tutorials, the pipeline generates 1.52M interaction steps for large-scale pretraining.

### Strengths
The scalability of the method, producing over a million structured steps providing a way to generate large amount of annotated data required for training.

Clear establishment of need for large scale data via performance scaling evaluation to support motivation and need for automated scalable data collection. 

Inclusion of cognitive-style reasoning using inner monologue generation process to extract rationale for steps i.e explicit the intent, the local plan, and the expected state change, enhances model interpretability and could improve models reasoning.

### Weaknesses
It could be beneficial to show the performance across more vision-language models to see if the improvements are generalizable.

For the generation of inner monologue while effective, reliance on LLM-generated text raises concerns about consistency and reliability of the rationales. Were there any analysis on the outputs? 
Quantitative human study on a small subset could provide some insight.

In the performance evaluation of Action Event Detection, what is the threshold for temporal overlap to count as a hit?

Minor comments not affecting rating:
Line 266: "ASR" define abbreviations\
line 376 typo "iff"

### Questions
Were the inner monologue used in the training?

In the first stage of action event detection, type a_k is predicted with the timestamps, why train the action parameterization model to again predict the type (line 245 - 247).\
it is mentioned in line 247 "when available, we optionally condition on the detector’s $a_k$ to stabilize type predictions.". How does this affect the performance of the parameterization model?

For the Stage 2 training, since it's stated that training was done on "curated set of clean, human-annotated trajectories" (line 310), were this the ones samples from Open CUA and AGUVIS.
If so how does the performance change if for stage 2, training data was drawn from only VideoAgentTrek generated trajectories and/or using a mix of human annotated and generated.

### Soundness
3

### Presentation
2

### Contribution
4

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper presents VIDEOAGENTREK, a scalable pipeline that automatically extracts GUI-action trajectories from unlabeled screen-capture videos (39 k YouTube tutorials → 1.52 M steps) and uses them for large-scale pre-training of computer-use agents. A learned inverse-dynamics module (VIDEO2ACTION) first localizes actions in time (event detector, F1=0.78) and then infers their parameters (click coordinates, typed text, etc.; 65.8 % human-verified accuracy). Continued pre-training on the mined data followed by supervised fine-tuning improves task success on OSWorld-Verified from 9.3 % (SFT-only) to 15.8 % (+70 % relative) and step accuracy on AgentNetBench from 64.1 % to 69.3 %. The authors release SCREENFILTER and VIDEO2ACTION as open-source tools.

### Strengths
Novel, timely problem: Leveraging the enormous volume of passive screen-capture videos for GUI-agent training is an appealing idea that addresses the current data bottleneck.
End-to-end pipeline: From raw YouTube crawl to executable (screenshot, action, parameters) tuples, the system is fully automated and scales to web size.
Strong empirical gains: Clear, statistically meaningful improvements over a pure SFT baseline on two independent benchmarks, plus positive scaling curves with data volume and test-time compute.

### Weaknesses
See questions.

### Questions
This paper presents a timely and impactful contribution by introducing VIDEOAGENTREK, the first fully-automated pipeline that converts unlabeled, publicly available screen-capture videos into large-scale, training-ready trajectories for GUI agents. By equipping an inverse-dynamics module (VIDEO2ACTION) with dense event detection and parameter extraction, the authors bypass the expensive manual-annotation bottleneck and demonstrate clear downstream gains on both online and offline benchmarks. 
One open question remains: will the complete codebase (SCREENFILTER, VIDEO2ACTION training & inference scripts, data-preparation pipeline) and the processed VideoAgentTrek dataset be publicly released?

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
4

### Summary
The paper introduces VideoAgentTrek, which synthesizes GUI interaction trajectories from actionless videos to replace costly manual annotations. The core is the Video2Action inverse dynamics module: it first localizes events, then infers action parameters, and finally synthesizes a chain-of-thought. Training proceeds in two stages: supervised fine-tuning on the full dataset, followed by fine-tuning on a human-verified subset. Experiments report absolute gains of +6% on OSWorld-Verified and +5% on AgentNetBench. The authors release the ScreenFilter and Video2Action tools but do not release the video dataset or the trajectory dataset.

### Strengths
- The paper is easy to follow. The data processing pipeline, training setup, and results are clearly presented.
- Applying VPT-style ideas to GUI agents is intuitive and scalable for data collection, and the idea is well-executed.
- Clear improvements on two popular benchmarks.
- ScreenFilter and Video2Action are released to support the reproduction of the video annotation pipeline.

### Weaknesses
- While tools are open-sourced, the full video corpus and trajectory annotations are not available, which I believe could significantly increase the contribution of this work.
- Potential data leakage is not quantified. The training data derived from public tutorials may overlap with OSWorld/AgentNetBench tasks, but the paper does not appear to include a rigorous deduplication or leakage analysis.

### Questions
- Tutorial videos often contain a lot of noise (e.g., extraneous/meaningless mouse movements, hotkey usage). Is there any specific mechanism to handle this? For text-input actions, how does the model differentiate between a string-level typing action and individual key presses?
- How are frames with no user action but visible UI changes handled? Does the Video2Action model have an explicit noop action?
- Cursor icons can vary across operating systems and applications. What cursor types are supported by the cursor detection model?
- Do event detection and cursor detection models require a standardized resolution/frame rate? How robust is the system across varying FPS and resolutions?
- Can you provide an ablation on the effect of automatically generated inner monologue?
- Is there a plan to release the full video dataset and extracted trajectories? Public release, especially of high-resolution videos with audio, would substantially strengthen the paper’s contribution. I would increase my score if this were to become available.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes VideoAgentTrek, a scalable pipeline to pretrain computer-use agents from unlabeled, publicly available screen-recorded videos. The key challenge is that raw videos  lack structured action labels (types, timestamps, and parameters). The authors address this by introducing an inverse dynamics module to generate the parameterized action tuples. With the collected dataset and the proposed multi-stage training strategy, the final performance of the GUI Agent achieved SOTA.

### Strengths
1. Interesting and reasonable reframing of GUI action recovery as inverse dynamics from raw videos without (great) manual labels.

2. The filter system leverages cursor detection to automatically focus on GUI-heavy segments; channel-coherence expansion for source discovery is practical and scalable.

3. Clear improvements over a strong SFT-only baseline on both OSWorld-Verified and AgentNetBench, plus analysis showing benefits grow with pretraining scale and planning horizon.

4. Pipeline is well-structured and explained (collection → filtering → detection → parameterization → rationale → training).

5. Practical tools and release plans (if they actually release finally) can catalyze broader community work.

### Weaknesses
1. While Stage 2 SFT mitigates noise (Parameterization accuracy, detection errors, etc), a more systematic study quantifying how parameter errors propagate to agent performance (e.g., via controlled corruption of parameters) would clarify robustness and failure modes.

2. The dense detector shows weaker recall for keyboard/press actions, which can be crucial in many workflows. It would help to analyze how missing non-pointer actions affects task categories (e.g., auth flows, terminal usage) and to explore augmentations (ASR cues, keystroke audio hints) to close this gap. Also, since the author uses the pointer as one of the  video filters, I am very curious whether the trained model would bias to such pattern? Further discussion and analysis would help. 

3. More detailed release of manifests (video IDs, timestamps, filtering decisions) would further improve reproducibility within ToS boundaries.

4.  The pipeline currently focuses on 2D screen recordings. Some OSWorld tasks involve subtle text/element detection under variable themes or require OCR robustness. An ablation with explicit OCR enhancements or higher-res sampling could clarify performance ceilings. Similarly, mobile platforms or non-English UIs are out-of-scope; discussing adaptation strategies would strengthen the broader impact story.

5. In Table 9, the authors provided a cross-dataset comparison, which is good; But I do think there misses some important datasets. I would prefer a more complete comparison as done in AgentNet, Table 2. This will help the readers to localize. 

6. From the Figure, it seems the "test-time scaling" helps a lot, but I cannot find the further details.

### Questions
1. See the weakness.

2. Have you tried ensembling multiple parameter predictors or using confidence-based filtering to drop uncertain steps? How does selectively pruning low-confidence parameterizations affect Stage-1 benefits?

3. Can you quantify how coordinate noise (e.g., ±k pixels) affects downstream success on benchmarks via a controlled perturbation study?
Improving non-pointer actions:

4. For press/type, did you explore fusing ASR transcripts or keystroke sounds to improve recall? Are there simple heuristics (e.g., stable cursor + text change) that boost detection without heavy supervision?


5. Beyond releasing filter tools, can you also share:
- Video ID lists and segment timestamps (without frames), with pass/fail reason codes.
- A small, license-cleared demo subset (e.g., Creative Commons) to enable exact replication of Stage-1 on a miniature scale.

6. Did video-pretraining improve performance uniformly across OSWorld app buckets (calc, chrome, vscode, etc.), or are gains concentrated in certain domains? 

7. Since mined text can include sensitive inputs (typed passwords, emails), how to deal with?

8. Could you incorporate UI element detection (e.g., layout parsing) to normalize parameterization into element-centric actions rather than screen coordinates, improving cross-resolution robustness?

### Soundness
3

### Presentation
3

### Contribution
3
