# ReWatch-R1: Boosting Complex Video Reasoning in Large Vision-Language Models through Agentic Data Synthesis

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 4, 4, 6, 6, 6

## Abstract
While Reinforcement Learning with Verifiable Reward (RLVR) significantly advances image reasoning in Large Vision-Language Models (LVLMs), its application to complex video reasoning remains underdeveloped. This gap stems primarily from a critical data bottleneck: existing datasets lack the challenging, multi-hop questions and high-quality, video-grounded Chain-of-Thought (CoT) data necessary to effectively bootstrap RLVR. To address this, we introduce ReWatch, a large-scale dataset built to foster advanced video reasoning. We propose a novel multi-stage synthesis pipeline to synthesize its three components: ReWatch-Caption, ReWatch-QA, and ReWatch-CoT. A core innovation is our Multi-Agent ReAct framework for CoT synthesis, which simulates a human-like "re-watching" process to generate video-grounded reasoning traces by explicitly modeling information retrieval and verification. Building on this dataset, we develop ReWatch-R1 by post-training a strong baseline LVLM with Supervised Fine-Tuning (SFT) and our RLVR framework. This framework incorporates a novel Observation \& Reasoning (O\&R) reward mechanism that evaluates both the final answer's correctness and the reasoning's alignment with video content, directly penalizing hallucination. Our experiments show that ReWatch-R1 achieves state-of-the-art average performance on five challenging video reasoning benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes ReWatch-R1, a novel approach to improve complex video reasoning in LVLMs by addressing the critical data bottleneck in existing methods. The authors introduce ReWatch, a large-scale dataset synthesized via a multi-stage agentic pipeline that includes temporally dense captions, high-difficulty multi-hop QA pairs, and video-grounded CoT traces generated through a Multi-Agent ReAct framework simulating human-like “re-watching.” They further develop an Observation & Reasoning (O&R) reward mechanism for RL, which jointly evaluates answer correctness and the factual grounding of intermediate reasoning steps.

### Strengths
1. Novel QA curation methods: ReWatch is carefully designed to enforce video dependency and multi-step reasoning through contrastive QA generation and rigorous filtering, effectively eliminating textual shortcuts and hallucination-prone supervision.

2. Innovative O&R reward mechanism: By evaluating both final answers and the fidelity of intermediate observations and reasoning steps, the O&R reward explicitly discourages hallucination and promotes evidence-based reasoning.

### Weaknesses
1. Formula 4 merges captions across each time interval independently, which leads to a loss of referential consistency. For example, the caption for 0–10s might be “A man…”, while that for 10–20s is again “A man…”, even though it refers to the same individual as in the earlier segment. This inconsistency compromises the overall quality of the generated captions.

2. The observation mechanism introduces inference latency. Despite yielding performance gains on certain video reasoning benchmarks, it provides only marginal improvements on most general video understanding benchmarks.

3. The paper lacks comparisons with more recent and advanced baselines, such as VersaVid-R1 and GRPO-CARE.

4. Quantitative analysis is missing: the paper does not present any concrete inference results or case studies from ReWatch-R1.

### Questions
1. Line 207: Why does the original set of 85K QA pairs yield over 170K multiple-choice QA pairs?  

2. From the Chain-of-Thought (CoT) example shown in the bottom-right corner of Figure 2, is retrieving explicit timestamps truly necessary? Could the reasoning path be simplified—for instance, as follows:  
> *"... So, I’ll \<action\> retrieve segments focusing on the man with blonde, curly hair on a jet ski interacting with a passenger \</action\>. \<observation\> The man on the jet ski passes a sandwich to the passenger, who then takes a bite. \</observation\> This directly answers the question. The food item passed was a sandwich..."*

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
3

### Summary
This paper introduces novel datasets about captions, QAs, and CoTs, with a training pipeline to improve complex video reasoning. Additionally, based on the synthesized CoT data, this paper further designs a new reward for RL training. The trained model achieves state-of-the-art performance on several video reasoning benchmarks, outperforming models trained on other open-source datasets.

### Strengths
- This paper finds drawbacks in the current synthesized long CoT datasets, e.g., Video-R1, and synthesizes a quality-improved video reasoning dataset.
- This paper redesigns the reward shaping for video understanding.
- The trained model sets a new SOTA on several benchmarks.

### Weaknesses
- The contribution of this paper mainly lies in the dataset construction. Can you break down the improvement of each part, like adding timestamps into the captions, adding question difficulty, and the proposed reward design?
- The introduction about the observation is not clear. How do you define observation formerly? In common, observation is visual content, while in your settings, the observation is processed captions. Would you consider implementing the thinking-with-image style training?

### Questions
- As shown in Tab.3, why does the trained model show marginal improvement on the most popular, general video question-answering datasets with reasoning?
- How do you distinguish reasoning video benchmarks and general video benchmarks?

### Soundness
3

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
5

### Summary
The paper proposes ReWatch-R1, a framework that enhances complex video reasoning in large vision-language models through agentic data synthesis and verifiable reinforcement learning. It introduces ReWatch, a large-scale, high-quality dataset built via a three-stage pipeline—hierarchical video captioning, high-difficulty QA generation, and multi-agent chain-of-thought synthesis—ensuring strong temporal grounding and reasoning diversity. Furthermore, the authors design an Observation & Reasoning (O&R) reward that jointly evaluates answer correctness and factual grounding of intermediate reasoning steps. Combined, these innovations enable ReWatch-R1 to achieve state-of-the-art performance on multiple challenging video reasoning benchmarks.

### Strengths
1. The paper introduces a novel multi-stage data construction pipeline that enables the creation of ReWatch, a large-scale and high-quality dataset specifically designed for video reasoning. The dataset comprehensively includes caption, QA, and chain-of-thought (CoT) annotations, providing valuable and versatile resources that can benefit future research and community development in multimodal reasoning.
2. The authors offer detailed experimental descriptions and implementation settings, ensuring strong reproducibility and facilitating independent verification and extension of their results.

### Weaknesses
1. The paper lacks qualitative experimental results, such as example responses generated by ReWatch-R1 for specific video reasoning cases. Including such examples would help readers intuitively understand the model’s reasoning style, output quality, and advantages over baseline models.
2. The selection of baseline models is relatively limited, lacking comparisons with stronger closed-source systems such as Gemini 2.5 Pro/Flash and GPT-4o, as well as with recent RL-based approaches like TW-GRPO. Including these baselines would provide a more comprehensive evaluation of ReWatch-R1’s effectiveness and competitiveness.

### Questions
1. Does the proposed Observation & Reasoning (O&R) reward mechanism introduce additional reasoning overhead or inference latency compared to conventional reward designs? It would be helpful if the authors could provide quantitative results or analysis on the computational cost and efficiency trade-offs, as well as discuss potential optimization strategies to mitigate these issues.
2. The paper currently compares ReWatch-R1 with a limited set of baseline models. Have the authors considered evaluating against stronger closed-source models (e.g., Gemini 2.5 Pro/Flash, GPT-4o) or recent RL-based approaches such as TW-GRPO, VersaVid-R1? Such comparisons would help better position ReWatch-R1’s performance and highlight its relative advantages within the broader landscape of contemporary video reasoning models.

### Soundness
3

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
4

### Summary
This paper introduces **ReWatch-R1**, which tackles the challenge of video reasoning from two aspects: 
1. Data. A large synthetic dataset called **ReWatch** is collected, which includes detailed video captions, challenging question-answer pairs, and high-quality, video-grounded reasoning traces (COT data). These are generated using a multi-agent ReAct system. 
2. Model. Authors post-train QwenVL by SFT and GRPO. They introduce a new Observation & Reasoning (O&R) reward, which evaluates the accuracy of video observations and the validity of reasoning process. 
ReWatch-R1 achieves promising performance compared with other 7B models.
Analysis shows that high-quality reasoning data is crucial for RL and RL on "thinking" mode improves the reasoning efficiency.

### Strengths
1. The proposed multi-agent COT data synthesis pipeline is scalable to curate large-scale video-grounding reasoning data.
2. The reward design shows the emphasis on explicit observation and reasoning is beneficial for video reasoning.
3. The data and model design achieves state-of-the-art performance on video reasoning and understanding benchmarks in 7B-scale models. The extensive analysis shows insights on the role and importance of SFT and RL.

### Weaknesses
1. The information source for data synthesis is semantic segmentation and detailed video description from Gemini. The accuracy of this multi-step hierarchical captioning is not validated. Therefore there is no direct quality assessment of the synthetic data.
2. All results are based on a 7B model. The benefits of high-quality COT data and O&R reward mechanism are not validated on larger-scale models.

### Questions
1. Equation (20): the design of non-format rewards lacks motivation. For example, why not use the (weighted) summation of all rewards. The design has no explanation or experimental results.
2. RL on 7B model shows promising improvements. However, the performance still lags behind the larger 32B model. Therefore, whether the same RL on 32B leads to improvement is questionable.

Some typos:
1. L255: "a novel O&R reward mechanism we propos" -> "propose"
2. L771: "Tabale 4" -> "Table 4"

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
3

### Summary
This paper mainly contributes to the following three points:

1). Proposing a novel multi-stage synthesis pipeline to synthesize ReWatch, a large-scale dataset which includes ReWatch-Caption, ReWatch-QA, and ReWatch-CoT three components, for fostering advanced video reasoning. 

2). Proposing a new Observation & Reasoning (O&R) reward for RLVR that improves reasoning by rewarding both final-answer correctness and the factual grounding of intermediate steps in video content.

3). Developing ReWatch-R1 by post-training a strong baseline LVLM with Supervised Fine-Tuning (SFT) and O&R framework, achieves state-of-the-art average performance on five challenging video reasoning benchmarks.

The dataset construction pipeline and the two-stage post-training framework proposed in this paper put the description of video change into the CoT, providing a new idea for the research of video reasoning from the perspectives of dataset and training.

In summary, this paper shows a high level in terms of method description (data construction pipeline,the O&R methods, etc.), experimental setup and writing, but for the situation that pipeline may cause error accumulation, some experimental results are not analyzed (this part will be described in detail in "Weaknesses"), which leads to the lack of analysis completeness. The quality of this paper will be improved after analyzing the missing parts.

### Strengths
This paper proposes a dataset construction pipeline for video reasoning, which is enlightening for the dataset construction method in this field. The detailed description of video in different time steps is introduced based on semantic segmentation, which combines with the video summary to enhance the reasoning ability. In Addition, this paper uses three-layer filtering to screen the data that can best reflect the reasoning ability. The two-stage post-training framework simulates the "Thought-Action-Observation loop", and uses the improved O&R reward for RLVR to enhance the reasoning performance of ReWatch-R1.

Therefore, this paper provides a new idea for the subsequent dataset construction method in this field and the improvement of reasoning performance.

### Weaknesses
The weaknesses of this paper focus on the method, experiment and writing：

In method and experiment：

1). Error accumulation: the stage of semantic segmentation and detailed description generation in the dataset construction pipeline designed in this paper may cause error accumulation. The error caused by segmentation may cause one semantic of the video to be put into multiple segments. The description generation may result in the missing description between multiple time steps, which may result in the description of an item appearing in the previous time step, but the description of the item is missing in the next step, resulting in error accumulation. This paper does not discuss and analyze this situation.

2). In Table 1, the results of ReWatch-R1-SFT and ReWatch-R1+O&R on the CG AV counting are the same, which are not analyzed in the paper.

3). Lack of comparative analysis between the vanilla RLVR and the improved O&R method under the dataset constructed in the paper.

4). In the analysis in Section 4.2 (specifically line 411), the performance degradation caused by Video-R1 replacing ReWatch-CoT does not consider the reason that SFT data and RL data is mismatch, which seems to be somewhat contradictory to the previous sentence "SFT is an independent prerequisite for RL".

5). Also in the analysis in Section 4.2 (specifically 414 lines), it seems that ablation study using different data combinations (e.g. only using ReWatch-Caption or ReWatch-QA or ReWatch-CoT) have not been carried out, and the conclusion that "The Quality of QA data used for RL determines final performance." is somewhat far fetched.

6). In Appendix C2, the performance degradation of 384 frame training is not analyzed compared with 192 frame training.

In writing：

1). In Figure 1, the annotation order of the legend is inconsistent with the display order of each LVLM in the figure.

2). The two marking symbols not appear in Table 3 of appendix C2.

### Questions
1) Why are the number of ReWatch-QA and ReWatch-CoT inconsistent? Shouldn't one QA data correspond to one CoT data?

2) In Section 2.3 (specifically line 247), why does a structured execution trajectory end with A_final instead of observation? (Thought-Action-Observation loop)

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 6

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This submission presents ReWatch-R1, a framework for advancing complex video reasoning in large vision-language models (LVLMs). The core contributions include the creation of the large-scale, multi-stage ReWatch dataset aimed at challenging multi-hop video reasoning, a multi-agent data synthesis pipeline generating temporally precise captions, high-difficulty QAs, and video-grounded CoT traces, and a new Observation & Reasoning (O&R) reward mechanism for RL. By post-training an LVLM backbone with these elements through SFT and RL, the authors demonstrate state-of-the-art performance across five video reasoning benchmarks, alongside ablation studies.

### Strengths
-The topic of video reasoning is of community’s interest and timely.
-The hierarchical segmentation and multi-agent CoT pipeline used to generate the ReWatch dataset is methodically crafted and shown to have yielded data with deeper temporal grounding and higher question complexity.
-The baselines compared are pretty up-to-date, which verifies the state-of-the-art achievement on the benchmarks tested in the paper.

### Weaknesses
-The incremental benefit of the O&R reward mechanism is not fully dissected independently from other RL contributions. In Table 1, the improvement from RL (+O&R) appears relatively modest, and there is limited analysis delineating whether specific reasoning failures or hallucinations are directly ameliorated by the O&R reward in practice.
-The paper leans heavily on LLMs (Gemini, GPT-4.1, Qwen, etc.) for practically all phases - data synthesis, answer verification, reward calculation, and benchmarking. Although this is common in the area, the cumulative propagation of LLM biases and potential “meta-overfitting” is insufficiently analyzed; for example, does repetitive LLM-based data filtering introduce subtle data shortcuts or annotation artifacts? A systematic error/robustness analysis would be welcome.
-There is little systematic exploration of failure modes, such as where ReWatch-R1 still hallucinates, or which reasoning subtypes remain unsolved - as could be illustrated by extensive qualitative error analysis or confusion matrices per reasoning dimension (section 4.2 or appendix). This is important to inform the future community on limits.

### Questions
-Can the authors provide concrete ablation or error-type breakdowns specifically isolating the effect of the O&R reward on hallucination rates or logical errors, perhaps through qualitative examples or confusion matrices? It is unclear if this reward is the main driver for reduced hallucinations, or if SFT data quality dominates.
-Given the extensive use of LLMs at nearly every phase, is there a risk that ReWatch-R1’s strong performance partly reflects an overfit to LLM-specific language or annotation artifacts in the data/reward pipeline?
-How sensitive is the model (and pipeline) to coarse versus fine semantic segmentation in the captioning stage? For example, does segment over-segmentation harm long-term reasoning due to context fragmentation?

### Soundness
3

### Presentation
3

### Contribution
2
