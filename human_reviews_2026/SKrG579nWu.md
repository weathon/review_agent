# MMChat: A Multi-turn Multi-Modal Conversational Benchmark Grounded in Real World User Behaviors

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 4, 6, 2

## Abstract
Large Vision-Language Models (LVLMs) are the foundation of modern AI assistants, which users typically engage in multi-turn conversations involving interrelated questions about images. However, existing benchmarks predominantly evaluate LVLMs in isolated, single-turn settings, creating a significant gap with real-world use. While recent efforts have explored multi-turn evaluation, they often rely on oversimplified, synthetic dialogues that fail to capture the complex, context-dependent nature of real-world user interactions. To address this critical gap, we introduce MMChat, a multi-turn, multimodal benchmark grounded in realistic conversational dynamics. Our benchmark systematically models conversational flows derived from authentic user data, incorporating the diversity, fragmentation, and contextual dependencies observed in practice. We synthesize 1,000 high-quality dialogue turns across 100 conversations, leveraging image-question pairs from SceMQA and interaction patterns from VisionArena-Chat. To evaluate LVLM performance, we employ a comprehensive two-way framework that combines fine-grained subdimension scoring with winrate against human verified ground truth, achieving strong agreement with human judgments. Through extensive evaluation of state-of-the-art LVLMs, we uncover systematic weaknesses in both open-source and proprietary models, particularly in handling fragmented and context-dependent user queries. For example, even an advanced proprietary LVLM like GPT-4o sees its performance score drop from 3.92 to 2.92 on direct opening queries compared to descriptive ones. Furthermore, models struggle with iterative tasks, with GPT-4o scoring just 3.65 in refinement and 3.85 in augmentation. Smaller open-source models perform even worse in these categories, with scores as low as 1.70, highlighting a critical gap in conversational ability.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces MMCHAT, a multi-turn multimodal benchmark designed to address critical gaps in evaluating Large Vision-Language Models (LVLMs) by leveraging real-world user interactions. MMCHAT systematically models dialogue flows derived from authentic user data in VisionArena-Chat, incorporating diverse, fragmented, and context-dependent characteristics of human-AI conversations. The benchmark consists of 100 dialogues totaling 1000 high-quality turns, built upon image-question pairs from SceMQA and interaction patterns from VisionArena-Chat. The paper uncovers weakness in both open-source and proprietary models. The study highlights model weaknesses in handling direct opening queries, iterative refinement, and augmentation tasks, providing insights into challenges like conversational decay and error propagation.

### Strengths
Originality: This paper is the first to explore the dimensions of recollection, expansion, refinement, follow-up, augmentation, and repetition in the field of multi-turn multimodal evaluation. 

Significance: In the field of multi-turn multimodal large language model evaluation, assessments along the dimensions of recollection, expansion, refinement, follow-up, augmentation, and repetition are highly meaningful. For example, they play an important role in advancing embodied intelligence, long-context agents, and personalized interactive systems for multimodal large models.

Clarity: The paper is easy to follow, and the stated motivation and contributions are easy to understand and relatively clear.

### Weaknesses
This paper is not very novel or is an incremental innovation. Reasons are as follows:

1.The identified LVLM failure mode insights are not novel and lack multimodal characteristics. Regarding the inspirations and insights obtained from the evaluation, MMChat arrived at the same conclusions as the multi-turn benchmark (MT-Eval) for large language models, namely Conversational Decay and Error Propagation. However, since this is indeed a multi-turn benchmark for multimodal large language models, evaluations conducted within these established dimensions should produce new insights to guide training.

2.Why then was only the same conclusion as MT-Eval obtained? The reason is that the organizational dimensions of the benchmark are not comprehensive. The paper directly continues using MT-Eval’s testing dimensions “recollection,” “expansion,” “refinement,” and “follow-up,” and only adds “augmentation,” “repetition,” “open-direct,” and “open-descriptive.” These additions are clearly weak and incomplete, still focusing on the detection of language capabilities within these dimensions rather than the detection of visual capabilities. When performing similar work for multimodal large language models, it is more important to emphasize the evaluation of visual information while retaining the original foundations. For example, in the augmentation pattern, beyond supplementary text, could multimodal information supplementation (e.g., sending another image) also be possible? In the recollection pattern, could it apply not just to recalling a single image but also two or even multiple images? I believe each dimension should consider multimodal aspects, aiming for maximum completeness. The current version is far from sufficient.

3.Real-world user behavior is claimed by the authors as a contribution, meaning that other multi-turn multimodal large model evaluation benchmarks are based on artificially synthesized dialogues. However, to my knowledge, ConvBench involved substantial human labor in annotation and dataset creation. Anyway, even if this is considered a contribution, the paper does not demonstrate what benefit this “real-world user behavior” brings to the evaluation. In other words, does real-world user behavior lead to different conclusions compared to artificially synthesized dialogues? The comparison with synthetic dialogues is missing. The performance comparison with previous multi-turn multimodal large model benchmarks (ConvBench, MMCR) is also missing.

4.The evaluation framework experiments are not rigorous enough. Visual grounding is entirely delegated to detection by a closed-source model, without any validation of whether the closed-source model’s visual grounding evaluation is correct, and without any comparison to human evaluation for consistency.

### Questions
In addition to the issues mentioned in the weaknesses, the following details are also missing.

1. Lack of key statistics of MT-Eval, including average/max turns per dialogue, average/max words in prompt, average/max words in response, average/max words per turn.

2. Lack of cost overview in using GPT4 in evaluation framework.

3. Computational resources, including the infer time for each LVLMs, evaluated time, resource consumption.

4. Effectiveness proof for evaluation framework.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces MMCHAT, a multi-turn, image-grounded conversational benchmark for LVLMs. The benchmark is (i) taxonomy-driven, derived from human-labeled real user–AI interactions, and (ii) flow-driven, sampling multi-turn trajectories from an empirically estimated transition model. The authors synthesize 100 conversations (1,000 turns) by pairing category-specific prompts with STEM images from SceMQA, then run a two-way evaluation: fine-grained, category-specific subdimension scores plus a win-rate vs. human-verified ground truth, both judged by GPT-4o with human verification checks and order randomization to reduce position bias. Through extensive evaluation of state-of-theart LVLMs, they have some main findings.

### Strengths
1.	Real-world grounding and principled taxonomy
The benchmark’s turn taxonomy (opening-direct/descriptive; follow-up, expansion, recollection, refinement, augmentation) is derived from human-annotated real dialogs and used to sample transitions that reflect authentic multi-turn distributions, rather than synthetic, fixed templates. This closes a known gap between single-turn evaluation and practical usage. 
2.	Flow-aware data generation with category-specific prompting
Conversations are trajectory-sampled from an empirically estimated transition matrix and generated turn-by-turn with category-specific prompts over a shared image context, yielding coherent, long-range interactions. 
3.	Clear empirical insights that matter in practice
The paper surfaces opening-direct difficulty, iterative-turn weaknesses, and detect two failure mode, decay/propagation phenomena, and per-family scaling trends—useful for model builders and eval designers.

### Weaknesses
1.	Judge and generator circularity / family dependence
GPT-4o is used for taxonomy auto-labeling, turn generation, and as the scoring/win-rate judge (albeit with human verification and order randomization). This creates potential circularity and family bias, especially when evaluating proprietary/open-source systems against GPT-4o-aligned criteria. The paper notes mitigation but does not quantify cross-judge robustness and reliablility (e.g., experment on different judge models to verify or estimate variance). 
2.	Limited scale of curated conversations vs. source corpus
Only 100 conversations / 1000 turns are released, derived from a 230k-conversation source for flow estimation. The paper would be more enough if it show more power analysis, confidence intervals, and stability under re-sampling to show that rankings and category trends are statistically robust at this 100 size.  
3.	Single-image conversations and domain coverage
Each conversation is tied to one image. While this controls variables, it narrows ecological diversity (multi-image, open-world scenes, documents/UI, egocentric video). Can the pipeline be generalized or extended to a second domain or multi-image/video scenarios?
4.	Ground-truth provenance and human verification scope
Ground truths are generated with GPT-4o then “human-verified.” The manuscript summarizes agreement rates, but it seems do not detail annotator scale, qualifications, double-blind checks, or inter-annotator agreement for ground-truth creation itself. This leaves residual uncertainty about truthfulness.

### Questions
See weaknesses.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces MMCHAT, a new multi-turn multimodal conversational benchmark designed to evaluate large vision-language models (LVLMs) under realistic user interaction scenarios. MMCHAT establishes a detailed taxonomy of conversational turn types—such as opening-direct, follow-up, refinement, and augmentation. The benchmark consists of 1,000 human-verified dialogue turns (100 conversations) built upon SceMQA image–question pairs. Evaluation uses both fine-grained multi-dimensional scoring and win-rate comparisons with human responses, showing strong agreement between automated and human judgments.

### Strengths
**Empirically grounded design:** MMCHAT is notably based on real VisionArena-Chat data, lending ecological validity absent in prior synthetic benchmarks.

**Comprehensive taxonomy:** The paper defines clear turn categories and subdimensions, enabling systematic analysis of conversational behaviors.

**Clear paper structure and readability**: The paper is well organized and logically presented, making the methodology and results easy to follow.

### Weaknesses
**Limited dataset scale**: The benchmark includes only 1,000 turns, which may restrict statistical robustness and coverage across domains.

**Dependence on GPT-4o**: Using GPT-4o for both data generation and evaluation introduces potential self-bias despite mitigation efforts.

**Domain narrowness**: The use of SceMQA (STEM-oriented) content constrains conversational diversity and limits generalization to other domains.

### Questions
**1. Dataset Scale and Diversity:**
The benchmark currently contains 1,000 turns across 100 conversations, which seems relatively small in scale and may limit statistical robustness.

**2. Dependence on GPT-4o:**
Since GPT-4o is involved both in data generation and in the evaluation process, how did you ensure that no bias occurred? Have you validated your results using a human-only or model-agnostic judging setup?

**3.Comparison with Prior Multi-turn Benchmarks:**
Both ConvBench and MMDU are multi-turn multimodal benchmarks that also evaluate dialogue understanding over multiple images. Could you clarify in more detail how MMCHAT fundamentally differs from these prior works to substantiate the novelty and contribution of this work?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
MMChat introduces a multi-turn multimodal conversational benchmark grounded in real user–AI interaction data to address the gap between how large vision-language models (LVLMs) are evaluated and how they are actually used. Unlike prior benchmarks that rely on synthetic or single-turn dialogues, MMChat builds a turn interaction taxonomy from the VisionArena-Chat dataset, modeling authentic conversational patterns such as opening-direct, refinement, and augmentation turns. It synthesizes 1,000 dialogue turns across 100 conversations, paired with STEM images from SceMQA, and evaluates models using a fine-grained, category-specific framework combining factual correctness, visual grounding, and tailored subdimensions. 

Extensive evaluation of 10 LVLMs reveals systematic weaknesses in handling direct queries, iterative refinement, and long-context grounding, identifying two key failure modes: Conversational Decay and Error Propagation. MMChat offers a more realistic and challenging evaluation paradigm for developing robust, context-aware multimodal AI systems.

### Strengths
1. The work provides a comprehensive taxonomy of seven interaction types (two opening and five later-turn categories), effectively modeling diverse user behaviors such as refinement, recollection, and augmentation, which are often ignored in previous studies.

2. The paper offers a realistic multi-turn multimodal dataset, constructed from real user–AI interaction distributions rather than purely synthetic scripts. This provides a valuable resource for evaluating LVLMs in settings closer to actual deployment.

### Weaknesses
1. The paper sometimes overstates its contribution. While the benchmark is grounded in real conversational statistics, the authors do not sufficiently position it relative to existing multi-turn multimodal datasets such as MMDU [1] and similar efforts. A more explicit comparison—both conceptually and empirically—would help clarify what is truly novel versus what is incremental, and would strengthen the narrative that this benchmark fills an unmet need in the space.

2. The presentation quality could be improved. Several figures and tables appear visually plain (such as Fig. 1, 2) and could benefit from clearer layout, consistent formatting, and more polished design. In addition, some equations lack alignment and visual structure (such as Line. 247, 248, 249), which affects readability. Combined with occasional clarity issues in the writing, these presentation limitations make the paper more difficult to follow than necessary and may hinder broader accessibility of the work.

3. The contribution scope feels relatively narrow given the ambition of the problem. The benchmark construction is thoughtful, but beyond dataset creation and analysis, there is limited methodological innovation.

[1] MMDU: A Multi-Turn Multi-Image Dialog Understanding Benchmark and Instruction-Tuning Dataset for LVLMs. NeurIPS 2024 Dataset & Benchmark Track

### Questions
1. Could you elaborate on plans to expand the dataset to include more images, domains, and longer conversations, or provide evidence that performance on this subset generalizes to other contexts?

2. Could you report inter-judge agreement between GPT-4o and human evaluators across categories?

### Soundness
2

### Presentation
2

### Contribution
2
