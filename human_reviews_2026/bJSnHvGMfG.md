# MLFM: Multi-Layered Feature Maps for Richer Language Understanding in Zero-Shot Semantic Navigation

- Decision: Reject
- Scores: 6, 4, 4

## Abstract
Recent progress in large vision-language models has driven improvements in
language-based semantic navigation, where an embodied agent must reach a target
object described in natural language. Yet we still lack a clear, language-focused
evaluation framework to test how well agents ground the words in their instruc-
tions. We address this gap by proposing LangNav, an open-vocabulary multi-
object navigation dataset with natural language goal descriptions (e.g. ‘go to
the red short candle on the table’) and corresponding fine-grained linguistic an-
notations (e.g., attributes: color=red, size=short; relations: support=on). These
labels enable systematic evaluation of language understanding. To evaluate on
this setting, we extend multi-object navigation task setting to Language-guided
Multi-Object Navigation (LaMoN), where the agent must find a sequence of goals
specified using language. Furthermore, we propose Multi-Layered Feature Map
(MLFM), a novel method that builds a queryable, multi-layered semantic map
from pretrained vision-language features and proves effective for reasoning over
fine-grained attributes and spatial relations in goal descriptions. Experiments on
LangNav show that MLFM outperforms state-of-the-art zero-shot mapping-based
navigation baselines.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces an open-vocabulary multi-object navigation dataset featuring natural language goal descriptions. It extends the traditional multi-object navigation task to a new setting, where an agent must locate a sequence of goals described in language. The authors further propose a novel approach that constructs a queryable, multi-layered semantic map using pretrained vision-language features, demonstrating its effectiveness in reasoning about fine-grained attributes and spatial relationships expressed in goal descriptions.

### Strengths
Overall, the problem setting is both realistic and practical, as it aligns closely with how users naturally specify sequential goals through language instructions.

### Weaknesses
In this setting, the task involves finding three sequential goals. I am curious about how this number was determined—have you experimented with different numbers of goals or tested the model’s performance under varying sequence lengths? What is the intuition behind choosing three? Additionally, it might be beneficial to provide a training dataset to further support the research community and enable more comprehensive exploration of this task.

### Questions
1. I am wondering whether the paper positions itself as an evaluation benchmark for open-vocabulary navigation, given that it provides only validation and test datasets.
2. In Section 4.2, the paper explains that MLFM+VLM is used to query relationships between objects, while MLFM+RGraph constructs a relational graph. Why isn’t the relationship queried directly from MLFM+RGraph? This seems like a chicken-and-egg problem—does the RGraph get constructed dynamically during the process?

### Soundness
3

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
This paper addresses the lack of language-centric evaluation in embodied semantic navigation by proposing three core contributions: (1) LangNav, an open-vocabulary multi-object navigation dataset with manually validated natural language goal descriptions and fine-grained linguistic annotations (attributes like color/size, spatial relations like "on/near"); (2) LaMoN, a task extending MultiON to require sequential navigation to language-specified goals with varying specificity; (3) MLFM, a multi-layered feature map method that balances 2D map efficiency and 3D height information, with three query variants (vanilla/VLM/RGraph) for fine-grained reasoning. Experiments on LangNav show MLFM+RGraph outperforms SOTA zero-shot baselines, and it generalizes to GOAT-Bench.

### Strengths
1. Manual validation eliminates VLM hallucinations (a major flaw of GOAT-Bench), and fine-grained tags enable dimensional evaluation of language understanding.

2. By requiring sequential language-specified goals without path instructions, LaMoN’s task better simulates real human-robot interaction than step-by-step VLN tasks.

3. The multi-layer map avoids 3D’s high memory cost and 2D’s height information loss.

### Weaknesses
1. Dataset limitations undermine generalizability: (1) Scenes are restricted to synthetic HSSD and semi-real GOAT-Bench. This paper did not test on physical environments (e.g., real apartments), where lighting/occlusions differ sharply from synthetic data. (2) Language is overly simple: no complex structures (coreference, negation) or action directives (e.g., "pick up"), making it irrelevant to real-world tasks like home service.

2. MLFM’s attribute understanding is incomplete: (1) Texture attribute success rate remains 0% (Table 2/3), but the paper only attributes it to feature extractors without proposing solutions or deeper analysis (e.g., fine-tuning CLIP on texture datasets). Moreover, state attribute (e.g., "illuminated mirror") underperforms OneMap-v2 (42.8% vs. 57.1%), but the failure analysis only simply mentions "detector issues" without detailed debugging. Better provide some experiments to discuss the issues mentioned above.

3. RGraph lacks adaptability: (1) Spatial relations are inferred via hand-crafted rules (e.g., Euclidean distance <0.2m = "next to"), not learned from data. This fails in scenes where "near" has different scales (e.g., 0.5m is "near" for a shelf but "far" for a table). (2) Cannot distinguish overlapping relations on the same layer (e.g., "inside" vs. "above" when two objects project to the same layer), with no workaround proposed.

4. Experiment scope is narrow: No comparison to state-of-the-art LLM-driven navigation methods (e.g., GPT-4V + embodied agents, 2024) – it’s unclear if MLFM is competitive with recent approaches. Besides, MLFM’s memory/compute cost vs. 2D/3D maps is unreported, which is critical for real-time robot deployment.

5. Some related and important works are missing citations: [1] NavQ: Learning a Q-Model for Foresighted Vision-and-Language Navigation [2] Learning Active Camera for Multi-Object Navigation [3] Mo-ddn: A coarse-to-fine attribute-based exploration agent for multi-object demand-driven navigation

### Questions
See weakness.

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
5

### Summary
This paper introduces LaMoN, a novel object navigation benchmark designed to address the limitations of existing ObjectNav benchmarks. It achieves this improvement primarily through two innovations: (1) It enriches the diversity of object descriptions by incorporating linguistic tags, allowing for more nuanced object definitions. (2) It integrates spatial relation descriptions for target objects, making the task more aligned with real-world navigation scenarios. These two features effectively narrow the gap between simulated benchmarks and practical applications, while also providing a more comprehensive evaluation platform for object navigation methods. Additionally, the paper proposes MLFM, a modular-based approach tailored for the LaMoN benchmark. Experimental results show that MLFM outperforms multiple baseline methods on ObjectNav tasks.

### Strengths
1. The proposed LaMoN benchmark is well-motivated with clearly presented details. Since most zero-shot object navigation (ZSON) approaches are still evaluated and compared on limited object categories and coarse-grained descriptions, this benchmark offers a superior alternative.
2. The paper presents a novel modular approach, MLFM, which is generalizable to different object navigation benchmarks including LaMoN and GOAT, and achieves competitive results on both.
3. The paper conducts detailed failure analysis and in-depth ablation studies on the impact of different types of image features in constructing the Multi-Layer Feature Map (MLFM) and the map query scheme.

### Weaknesses
1. The paper lacks essential comparisons with state-of-the-art navigation foundation models (e.g., NaViD[1], NaViLA[2], StreamVLN[3]) that are capable of handling general navigation tasks.

2. The contributions and differences between the proposed LaMoN benchmark and recent object navigation benchmarks (e.g., DOZE[4]) are not clearly justified.

3. The performance of MLFM in real-world scenarios remains unclear.

### Questions
1. What is the performance of recent end-to-end navigation foundation models on the LaMoN benchmark?

2. The paper presents the LangNav dataset—does this dataset benefit the training of navigation models on other object navigation benchmarks?

3. Given that obstacle sizes vary across different object types, how to define adaptive horizontal slices and construct a multi-layer feature map suitable for different scenes?

4. What is the decision frequency of the proposed MLFM approach?

### Soundness
3

### Presentation
3

### Contribution
3
