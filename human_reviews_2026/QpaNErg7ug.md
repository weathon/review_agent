# WearVox: An Egocentric Multichannel Voice Assistant Benchmark for Wearables

- Decision: Accept (Poster)
- Scores: 8, 6, 4, 6

## Abstract
Wearable devices such as AI glasses are transforming voice assistants into always-available, hands-free collaborators that integrate seamlessly with daily life, but they also introduce challenges like egocentric audio affected by motion and noise, rapid micro-interactions, and the need to distinguish device-directed speech from background conversations. Existing benchmarks largely overlook these complexities, focusing instead on clean or generic conversational audio. To bridge this gap, we present WearVox, the first benchmark designed to rigorously evaluate voice assistants in realistic wearable scenarios. WearVox comprises 3,842 multi-channel, egocentric audio recordings collected via AI glasses across five diverse tasks including Search-Grounded QA, Closed-Book QA, Side-Talk Rejection, Tool Calling, and Speech Translation, spanning a wide range of indoor and outdoor environments and acoustic conditions. Each recording is accompanied by rich metadata, enabling nuanced analysis of model performance under real-world constraints. We benchmark leading proprietary and open-source speech Large Language Models (SLLMs) and find that most real-time SLLMs achieve accuracies on WearVox ranging from 29\% to 59\%, with substantial performance degradation on noisy outdoor audio, underscoring the difficulty and realism of the benchmark. Additionally, we conduct a case study with two new SLLMs that perform inference with single-channel and multi-channel audio, demonstrating that multi-channel audio inputs significantly enhance model robustness to environmental noise and improve discrimination between device-directed and background speech. Our results highlight the critical importance of spatial audio cues for context-aware voice assistants and establish WearVox as a comprehensive testbed for advancing wearable voice AI research.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper collects a large set of in-the-wild data of users interacting with expected Voice AI tasks using microphone enabled glasses. The range of tasks tested is Search-Grounded QA, Closed-Book QA, Side-Talk Rejection, Tool Calling, and Speech Translation. For each task, there are multiple possible speakers including the wearer, the conversational partner, and bystanders and speech from each should be handled differently on a contextual basis. The authors first benchmark a range of current audio models on these benchmarks, then do an ablation of their own custom models designed for this data type and find that multi-channel signal processing is key to handling noise and acoustic cues in the data.

### Strengths
- The dataset covers realistic tasks, realistic environments, and realistic challenges for speech models. Despite being primarily focused on this realism, the benchmark is clearly difficult for frontier speech LLMs especially open-weights speech LLMs. It will clearly help measure (and therefore drive) progress.
- The multi-party nature of the environment is a welcome addition to the speech benchmarking space which usually operates on data from relatively clean single-speaker environments. With this, the benchmark will clearly drive models to be more robust to the natural challenges from real multi-party environments (such as distractions and interruptions) that often plague today's speech LLMs.
- The authors then go above and beyond reporting the results of their benchmark by showing an example of a low-level ablation which the benchmark allows them to enable the impact of.

### Weaknesses
- The benchmark seems somewhat likely to incentivize models to optimize for the specific hardware the benchmark was recorded on. This is of course true of any speech benchmark which uses a consistent recording device, but might become increasingly impactful for devices which have specific form factors and layouts for their microphone arrays.

### Questions
- Are there any ablations for the SC and MC Llama on how much more specified MC Llama is to the specific layout of the microphones in these particular glasses? Could you train MC Llama on a simulations of a different Multi-Channel layout and get similarly strong transfer to WearVox? While it makes sense that multi-channel is useful, it's not clear to what degree a single channel model might be more transferrable across devices.

- Are you able to share the participant demographics mentioned in the ethics statement? It says "demographic data was aggregated and used solely for fairness analysis." but I wasn't able to find such analysis and it would be valuable to share.

(Typo on 347. Appedix should be Appendix)

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
2

### Summary
The paper presents WearVox, an egocentric multimodal dataset that integrates head-mounted RGB video, binaural microphones, and body-contact microphones to capture synchronized, real-world audio-visual data for wearable perception tasks. To complement this dataset, the authors propose WearVoxNet, a unified multimodal encoder that jointly processes visual and acoustic signals for tasks such as speaker localization, active speaker detection, audio-visual event detection, and grounding. The overarching motivation is to advance embodied perception research by enabling models that can understand and respond to multisensory inputs from the perspective of a wearable device, bridging the gap between egocentric vision and human-centered audio perception.

### Strengths
(1) WearVox represents a significant and original data contribution. It is, to the best of current knowledge, the first large-scale dataset that combines egocentric RGB video, binaural recordings, and contact microphone signals in naturalistic scenarios. This unique combination allows for studying wearable perception in realistic settings, including interactions and self-generated sounds that traditional datasets cannot capture.
(2) The dataset’s coverage of multiple downstream multimodal tasks—including speaker localization, active speaker detection, audio-visual event detection, and grounding—demonstrates its broad utility. Such diversity supports a range of research directions in speech perception, ambient understanding, and diarization, positioning WearVox as a potentially foundational dataset for wearable multimodal intelligence.
(3) A key technical insight is the inclusion of contact microphones, which capture body-conducted vibrations and help distinguish self-speech from surrounding voices. This modality provides an innovative solution to one of the core challenges in wearable audio—disambiguating the user’s voice from ambient speech—making the dataset especially valuable for applications like personal assistants, AR/VR communication, and cognitive hearing aids.
(4) The data collection emphasizes ecological validity by recording in diverse real-world environments and activities. This diversity strengthens the dataset’s relevance for real-world deployment and ensures it captures the dynamic acoustic and visual variability encountered in wearable scenarios.

### Weaknesses
(1) The WearVoxNet model, while competent, does not introduce fundamentally new architectures or fusion strategies. It primarily builds upon established audio-visual encoder paradigms, combining features through standard cross-attention or concatenation methods. As a result, its novelty lies more in the dataset than in methodological innovation.
(2) The paper does not explore cross-dataset generalization, such as training on WearVox and evaluating on existing benchmarks like Ego4D or AVD. Such experiments would be essential to demonstrate the robustness and transferability of the learned representations beyond the curated data, highlighting whether WearVox-trained models can generalize across wearable domains.
(3) While the contact microphone modality is a central innovation, its contribution is not rigorously analyzed. The paper lacks systematic ablation studies on factors such as sensor placement, environmental interference, and failure modes. A deeper examination of when and how this modality enhances performance would provide greater insight into its reliability and sensitivity across contexts.
(4) The work does not address practical deployment constraints. Wearable devices typically face stringent limitations on power consumption, latency, and computational capacity, yet the paper does not discuss whether WearVoxNet—or any of its variants—can be efficiently adapted for on-device inference or lightweight architectures suitable for real-world use.
(5) The empirical comparisons could be stronger. Benchmarking against state-of-the-art models from recent audio-visual or egocentric perception challenges (such as AVE, Ego4D, or AVD) would better substantiate the claims of high performance and clarify where WearVoxNet stands relative to the broader literature.
(6) Finally, while the contact microphone clearly boosts self-speech detection, it is not clear how it affects other multimodal tasks. A more detailed analysis of modality contribution across different perception scenarios—such as non-self speech, ambient sounds, or silent visual events—would strengthen the interpretability of the findings and clarify when this modality provides genuine cross-task benefits.

### Questions
(1) Would it be possible to include a new set of experiments demonstrating cross-dataset generalization (for instance, training on WearVox and testing on Ego4D or AVD) to validate that the learned audio-visual representations transfer effectively to unseen wearable contexts and real-world data?
(2) Can the authors present a focused study on the contact microphone’s contribution—analyzing placement robustness, ambient noise resilience, and task-wise performance gains—to highlight its distinct sensing value and clarify why it is essential for reliable egocentric audio understanding?

### Soundness
3

### Presentation
2

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
This paper introduces WearVox, a benchmark dataset designed to evaluate voice assistants in realistic wearable scenarios, e.g. with wearable glasses. The authors mention that existing voice assistant benchmarks fail to capture challenges of wearable, egocentric audio such as motion, wind noise, background and side-talk rejection. This paper provides a benchmark dataset of 3,842 recordings captured using AI glasses in multi-channel audio format for five diverse tasks that AI assistants may have to do: Search-Grounded QA, Closed-Book QA, Side-Talk Rejection, Tool Calling, and Speech Translation. The authors study existing voice assistant methods, Speech Large Language Models (SLLMs) as baselines and also introduce specialized new models that handle native multichannel audio better.

### Strengths
1. This paper is a solid contribution to benchmark real-world AI assistant applications in a wearable setting, such data is very hard to find and is expensive to manually curate, script, and collect. It is a first of a kind dataset for this emerging setting in HCI and human-AI interfaces.

1. The baselining is done for a wide range of commercial models, and is done across settings and tasks that matter for this wearable setting.

1. "Side-Talk Rejection" is arguably the single most important and overlooked challenge for an always-on wearable assistant. Evaluating this explicitly is a huge contribution.

1. The paper carefully curates data, using specific angles and distances, and capturing audio across 13 different noise types, the paper has created a dataset that reflects the structured complexity of real-world interactions, not just random noise.

### Weaknesses
1. For the custom trained models, it is not clear if there is data leakage? It is also not clear from the paper what data was used to train these models. Presumably the baseline models were also trained with noise augmentation as is standard with commercial-grade speech models.

1. There doesn't seem to be an easy way to explore the benchmark. Perhaps this is a way to prevent leakage, but it would be great to be able to explore this dataset in an interactive way.

1. It is hard to get a sense of diversity in the dataset. Presumably the annotators who were asked to write multi-turn conversations did a good job, but there are no quantitative evaluations about the _quality_ or diversity of this dataset. Additionally, the limited scale might not capture edge cases that matter like speaker accents.

1. The data was collected using the microphone array from a specific set of AI glasses, and does not capture the diversity in hardware.

### Questions
1. More details on SC/MC WearLlama would be nice, especially the training data and methods. Was there any overlap or fine-tuning done on WearVox? You demonstrated that proprietary models struggled with noise. How did your noise augmentation strategy for Wearllama differ from the standard techniques you assume are used in commercial models?

1. Is there a way to measure the "realness" and diversity of the scripts? Particularly humans are bad at creating entropy, what measures were takes to make sure sure there is enough diversity?

1. Are there plans to make the WearVox dataset, or at least a sample of it, publicly available or browseable? An interactive way to explore the audio and metadata would be invaluable for the research community.

1. You used an LLM to judge the QA and Translation tasks. What steps did you take to validate the LLM-judge itself? For instance, how did its scores correlate with human expert scores on a subset of the data?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
WearVox introduces a wearable-specific benchmark of 3842 egocentric multichannel recordings collected with AI glasses, spanning 5 speech tasks, with each clip having rich environment and position metadata for nuanced analysis. The paper benchmarks open-source and proprietary SLLMs and reports large drops in noisy/outdoor conditions.

### Strengths
- First benchmark aimed squarely at wearables with egocentric, multi-mic audio, diverse indoor/outdoor scenes, and explicit side-talk; prior suites largely miss these factors.
- Both open and proprietary SLLMs; headline finding: most real-time SLLMs land ~29–59% on WearVox, highlighting difficulty.
- Five tasks with clean input/output definitions.
- Support multi-channel processing, for testing the SLLMs.

### Weaknesses
- Some reporting is aggregate. More per-environment/per-distance breakdowns (beyond the figures) would make failure modes easier to act on. 
- "Thinking" boosts scores but increases TTFT substantially, this deserves heavier emphasis for wearables.
- A careful proofreading pass is needed to improve clarity as well as typos.
- No examples to listen.

### Questions
- Will you release a public leaderboard and evaluation server? If so, please provide the planned URL and timeline.
- Please report per-task latency and time-to-first-token (TTFT) for each model—especially in ‘thinking’ mode—so we can judge real-time feasibility on wearables.
- Could you please eleberate more on the MC Wearllama? The description is not clear.

### Soundness
3

### Presentation
3

### Contribution
3
