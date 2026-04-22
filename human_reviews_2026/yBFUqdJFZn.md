# ScaleCUA: Scaling Open-Source Computer Use Agents with Cross-Platform Data

- Avg Score: 6.80
- Decision: Accept (Oral)
- Scores: 6, 10, 6, 6, 6

## Abstract
Vision-Language Models (VLMs) have enabled computer use agents (CUAs) that operate GUIs autonomously, showing great potential, yet progress is limited by the lack of large-scale, open-source computer use data and foundation models. In this work, we introduce ScaleCUA, a step toward scaling open-source CUAs. It offers a large-scale dataset spanning 6 operating systems and 3 task domains, built via a closed-loop pipeline uniting automated agents with human experts. Trained on this scaled-up data, ScaleCUA can operate seamlessly across platforms.  Specifically, it delivers strong gains over baselines (+26.6 on WebArena-Lite-v2, +10.7 on ScreenSpot-Pro) and sets new state-of-the-art results (94.4% on MMBench-GUI L1-Hard, 60.6% on OSWorld-G, 47.4% on WebArena-Lite-v2). These findings underscore the power of data-driven scaling for general-purpose computer use agents. We will release data, models, and code to advance future research.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper presents a dataset and agent models for computer agents. Provides a rich source of open source computer use data that is curated across various platform to fine tune the models. They show that fine tuning with this data curated across various platforms creates a strong model that performs well across various computer use benchmarks.

### Strengths
1. The comprehensive data set and model that the paper proposes have the potential to further the research in this field.
2. Experimental results showing the overall approach and how well it generalizes across various tasks. 
3. Method presented for data curation, which can be further extended to create more data assets

### Weaknesses
no comment

### Questions
Is the fin-tuning used SFT in this case?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
10

### Rating Number
10

### Confidence
3

### Summary
This paper introduces a cross-platform computer use dataset, collected via an interactive data pipeline that
integrates automated agents with human experts. They also developed ScaleCUA, a family of base agent models achieving reported SOTA on several GUI-oriented benchmarks.

### Strengths
- The model achieves impressive results on several challenging benchmarks, including SOTA performance on MMBench-GUI L1-Hard (94.4%).
- The paper presents a comprehensive suite of ablation studies exploring design choices, including data augmentation, trajectory weighting, coordinate formats, resolution impact, inference methods, and data scaling.
- The dataset spans six OSs and multiple GUI domains, broader than previous open datasets (e.g., OSWorld, JEDI, WebArena). Openly releasing all data, models, and code is a significant contribution, particularly within this research area, where datasets are often proprietary.

### Weaknesses
No major weaknesses, some suggestions:
- The paper contains several minor typos and grammatical errors (e.g., "expensively to annotate," capitalization issues) and inconsistencies in terminology. 
- They can add an analysis of the data quality, comparing the current approach and manual collection.

### Questions
How does the model perform when faced with minor software changes (e.g., UI updates, version differences)? Is there a mechanism to adapt to these types of alterations without requiring retraining?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces ScaleCUA, a large-scale dataset with multiple OS and tasks supported and associated agents trained on the dataset. It begins by incorporating a dual-loop data collection pipeline consisting of agent-environment interaction loop and agent-human hybrid data acquisition loop. The authors then train Qwen2.5-VL on the collected dataset with three modes: grounding mode, direct action mode and reasoned action mode. Experiments are conducted on multiple datasets, including screenspot-pro, webarena-lite-v2, mmbench-gui, os-world-g etc. These evaluations cover multiple tasks and demonstrate a comprehensive & promising results for ScaleCUA. ScaleCUA shows the potential to scale up CUA performance with cross-platform data collected from automated agent data collection and human-in-the-loop data collection together.

### Strengths
1. ScaleCUA contributes a large-scale dataset to the GUI Agent community, with 15k weak-semantic trajectories from automated agent data collection and 4k expert-curated trajectories from human-in-the-loop data collection. Notably, the collected data cover 6 OS, including desktop and mobile ones. 
2. ScaleCUA contributes several interesting key insights to the community backed with supporting experiment results, including the finding that models trained with raw coordinates perform better than those trained with normalized coordinates, especially in cross-platform scenarios; reasoned action mode is  always better than direct action mode, the multi-format GUI grounding corpus, etc.
3. This paper is well written, clearly structured, and provides comprehensive details and ablation studies in the appendix for future research.

### Weaknesses
1. For the dual loop, currently it includes two components: agent-environment interaction loop and agent-human hybrid data acquisition loop, and these two components both contribute to the final data collection of ScaleCUA, with different properties, i.e., data collected from agent automated pipeline are of large scale but weak semantics, data collected from human-in-the-loop pipeline are of high quality but high cost. However, ScaleCUA did not comprehensively integrate these two components so that they can benefit each other. This is not a major drawback, but a point that the authors can improve further. The proposed dual loop is promising, but the current implementation of simply merging the two data sources is naive, which has been discussed in prior work。
2. The dataset distribution is extreme, e.g., MacOS data takes only 0.6% of the total data, and grounding takes 97.6% of the total Windows data. As a cross-platform dataset with multi-task support, this distribution is not reasonable enough.

### Questions
1. for Figure 5(a), I observed that the grounding accuracy keeps almost unchanged or dropping when the resolution is increased from 720p to 4k in screenspot-v2, also the grounding accuracy saturates at 1080p in osworld-g. Do these results demonstrate that "Grounding accuracy rises steadily from 720p to 2K, with diminishing returns at 4K." is not a universal finding, but only occuring in screenspot-pro benchmark?
2. It can be observed that scaling data has marginal effects on improving agent performance on WindowsAgentArena benchmark,  does this implicate that increasing train data amount cannot benefit navigation tasks? Can you also share the performance on OSWorld with different training data ratio?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
1

### Summary
The paper tackles the challenge of building computer use agents (CUAs)—AI systems that can autonomously operate GUIs across desktop, mobile, and web platforms. Current progress is limited by the scarcity of high-quality, open computer-use data and the reliance on closed-source models.

To address this, the authors introduce ScaleCUA, a framework that combines a large-scale cross-platform dataset with a family of open-source agent models. Their approach integrates automated exploration with human expert trajectories, creating a diverse and scalable training corpus. The resulting models unify perception, reasoning, and action, and support multiple inference modes for flexible deployment.

### Strengths
1. Introduces a large-scale, open dataset spanning six platforms (Windows, macOS, Linux, Android, iOS, Web) and three task domains (GUI understanding, grounding, and task completion), collected through a hybrid pipeline of automated agents and human experts.

2. Develops a suite of base agent models that unify perception, reasoning, and action, supporting three inference modes (Grounding, Direct Action, Reasoned Action) within a unified action space for seamless cross-platform interaction.

3. Demonstrates strong performance across multiple benchmarks, achieving new state-of-the-art results and showing that scaling diverse GUI-specific data significantly improves general-purpose computer use agents.

### Weaknesses
1. The dual-loop data collection pipeline (agent + human) is presented as a key innovation, but similar hybrid strategies have been explored in prior works such as OS-Genesis (Sun et al., 2024b) and AGUVIS (Xu et al., 2024). The paper risks being perceived as a scale-up rather than a conceptual breakthrough.

2. A large fraction of trajectories come from random-walk or weakly semantic exploration, which may not reflect realistic user goals. This could bias the model toward superficial navigation patterns rather than meaningful task completion.

3. The paper reports strong results but lacks fine-grained ablations. For instance, it is unclear how much each component (weak semantic trajectories, reasoning data, augmentation strategies) contributes to final performance. Similarly, the trade-offs between the three inference modes (Grounding, Direct, Reasoned) are not deeply analyzed.

### Questions
While ScaleCUA spans six platforms, it is unclear how well it generalizes to unseen applications or domains. For example, can a model trained on productivity and web apps adapt to enterprise software or creative tools?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors build a large-scale training dataset for GUI agents, including 6 operating systems and 3 task domains, and develop the ScaleCUA series of models. The authors propose an automatic method to collect trajectories and grounding annotations and build a model that can perform well on GUI understanding, grounding, and planning. The authors benchmark ScaleCUA on a number of tasks to show its efficacy.

### Strengths
1. The authors curate a large-scale dataset for GUI agents. This includes understanding, grounding, and planning data. The data consists of several software applications across diverse platforms. The authors claim to release the data openly, which will be very valuable.
2. The authors conduct a comprehensive set of experiments. They evaluate on several benchmarks across different tasks. They also provide several analyses on data scaling, the effect of screenshot resolutions, and several other design decisions. These provide good insights for future model development. 
3. The authors provide practical guidance for collecting data from various platforms in the Appendix, which is very insightful.

### Weaknesses
1. The data collection introduced by the authors consists of an automatic method where rule-based agents and heuristics are used, and another component that comprises human-created tasks and trajectories. The presentation of this section gives a misleading impression that the trajectories are created by both humans and agents collaboratively, but in actuality it is some trajectories created by agents and others created entirely by humans. I fail to see any new method or innovation here (besides the heuristics that the authors develop for rule-based agents). Could the authors please provide clarification for this?
2. The authors do extensive evaluation, but compare with a limited set of models. Several recent models, like [1], [2], [3] use RL to train their models and show that they can achieve impressive performance using public data. I understand that the ScaleCUA models are only SFTed and not RL finetuned, but since models are also highlighted as one of the contributions of this work, comparing them more extensively would further show the effectiveness of their models.
3. The authors should conduct a more controlled comparison between their data and open source datasets like Jedi. The authors currently compare the benefits of using their data over public data, but it is not clear if the quantity of data used for this comparison is the same or not. The authors could take a fixed random sample of data from different datasets like Jedi, OsAtlas etc., and fine tune the same model to show whether the data collected by the authors provides more gains than already available datasets.

[1] Yang et al. Gta1: Gui test-time scaling agent

[2] Liu et al. Infigui-g1: Advancing gui grounding with adaptive exploration policy optimization

[3] Tang et al. Gui-g2: Gaussian reward modeling for gui grounding.

### Questions
1. On online evaluation tasks, especially OSWorld, ScaleCUA underperforms compared to OpenCUA and a few other models despite being trained on large quantities of data. Could the authors comment on this? This is especially relevant because it raises questions about the effectiveness of the data collection method.

Please see the weakness for more questions and suggestions.

### Soundness
3

### Presentation
3

### Contribution
3
