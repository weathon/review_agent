# InPhyRe Discovers: Large Multimodal Models Struggle in Inductive Physical Reasoning

- Decision: Reject
- Scores: 4, 2, 2, 6

## Abstract
Large multimodal models (LMMs) encode universal physical laws observed during training, such as momentum conservation, as parametric knowledge. It allows LMMs to answer physical reasoning queries, such as the outcome of a potential collision event from visual input. However, since parametric knowledge includes only the physical laws seen during training, it is insufficient for reasoning when the inference scenario follows physical laws unseen during training. In contrast, humans can adapt their physical reasoning to unseen physical environments with only a few visual examples. This inductive physical reasoning ability is indispensable for LMMs if they are to replace human agents in safety-critical applications. Despite its importance, existing visual benchmarks evaluate only the parametric knowledge in LMMs, and not inductive physical reasoning. To this end, we propose InPhyRe, the first visual question answering benchmark to measure inductive physical reasoning in LMMs. InPhyRe evaluates LMMs on their ability to predict the outcome of collision events in algorithmically generated synthetic videos. By inspecting over 13 open-source and proprietary LMMs, InPhyRe informs us that (1) LMMs struggle to apply their limited parametric knowledge about universal physical laws to reasoning, (2) inductive physical reasoning in LMMs is weak when inference scenarios obey physical laws unseen during training, and (3) inductive physical reasoning in LMMs suffers from language bias and largely ignores the visual inputs, questioning the trustworthiness of LMMs regarding visual inputs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces INPHYRE, a VQA benchmark designed to evaluate inductive physical reasoning in LMMs. INPHYRE assesses whether LMMs can infer and apply unseen or violated physical laws from demonstration videos. The benchmark uses algorithmically generated synthetic videos of collision events that violate real-world physical laws. Through extensive evaluation of 13 open-source and proprietary LMMs, the authors find that LMMs struggle with inductive physical reasoning, rely heavily on language bias, and often ignore visual inputs.

### Strengths
1. First benchmark to systematically evaluate inductive physical reasoning in LMMs using physically impossible scenarios.

2. This paper's writing is well-organized, clearly written, and richly illustrated.

### Weaknesses
1. Limited Insight: The paper's conclusion that "LMMs struggle with inductive physical reasoning (L68)" may stem from the limited scale of SFT data and potential conflicts between counterfactual scenarios and pre-trained knowledge. Moreover, the observation that "inductive physical reasoning in LMMs suffers from language bias and largely ignores visual inputs" has already been extensively discussed in prior literature.

2. Lack of Closed-Source Model Evaluation: The study does not include evaluations on closed-source models such as GPT-4, Gemini, or Doubao. It is recommended that the authors incorporate results from at least one of these models to enhance the comprehensiveness of their analysis.

3. Narrow Physical Scope: While the benchmark concentrates mainly on collision events and mechanical reasoning, the authors suggest that their findings are likely generalizable. However, key physical domains such as fluid dynamics and thermodynamics remain unexamined, limiting the scope of the study.

### Questions
See weakness.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This paper presents InPhyRe, a benchmark designed to evaluate the inductive physical reasoning of Large Multimodal Models. By constructing synthetic visual scenarios that intentionally violate universal physical laws, InPhyRe provides a method to quantitatively measure a model's ability to infer and apply novel physical principles from in-context visual examples. The benchmark is used to evaluate over LMMs on their parametric knowledge, inductive reasoning, and susceptibility to language bias. The results show that current models struggle significantly on InPhyRe, demonstrating weak inductive capabilities and a strong reliance on language cues over genuine visual understanding.

### Strengths
1. Clear definition and writing. The paper is clearly written and very effectively identifies the current shortcoming of current LLM/VLMs in inductive physical reasoning.
2. The authors conducted comprehensive experiments on curent LMMs.
3. I think the paper overall challenges the in-context viusal reasoning capabilities of LMMs, pushing the community to develop more adversarial and out-of-distribution benchmarks to probe the true limits of these models.

### Weaknesses
1. My biggest concern is the positioning of this paper. The authors select examples that violate physical principles in order to avoid being covered by the LLM's pretrained parametric knowledge. However, I think the model's weak performance in such an environment only indicates that it is dominated by a strong physics prior, not necessarily that its inductive reasoning ability is weak. To the best of my knowledge, experiments related to the violation-of-expectation paradigm in psychology [1] have shown that human infants, when encountering phenomena that appear to violate their expectations, also tend to assume their internal world model is correct, rather than overturning their established mental simulation model based on a few sampled trajectories.

2. There is a huge gap between the benchmark's design and the real world applicability. Real-world scenarios, such as autonomous driving on diverse ground materials mentioned, are still physically consistent and will not feature impossible events like violation of object permanence.

3. I think the dataset design is a bit simple. The dataset only includes basic geometric shapes on a flat surface, meaning the model does not need to handle any real-world visual challenges such as lighting and shadows, occlusion, complex backgrounds, or material reflections. Have the authors considered including more diverse object materials (e.g., soft bodies/cloth) and richer shapes?

4. The paper lacks further experiments to investigate the cause of this phenomenon, such as visualizing specific attention maps or applying linear probing to inner representations. 

[1] Piloto LS, et al. Intuitive physics learning in a deep-learning model inspired by developmental psychology. Nature Human Behaviour, 2022.

### Questions
The authors argue that "inductive physical reasoning, is a hallmark of intelligence that humans develop at a very young age." Previous experiments on inductive and physical reasoning in infants were mostly conducted in real-world environments and did not involve scenarios that violate real-world physics. Could the authors provide further human studies to support this claim?

I'll be glad to increase my score if the concerns above are properly resolved.

### Soundness
1

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper introduces INPHYRE, a synthetic CLEVR‑style benchmark that gives exemplar videos and asks VQA about outcomes of collisions, including rule violations. The authors report three findings. Limited parametric physics knowledge, weak adaptation to unseen rules, and strong language bias.

### Strengths
- Clear motivation about adaptation to out‑of‑distribution physics.
- Simple, reproducible pipeline built on PyBullet and Blender with a scenario taxonomy.
- Broad model sweep and a few ablations on number of exemplars, CoT, and quantization.

### Weaknesses
- Firstness claim is incorrect or overstated. Prior work already probes counterfactual or violated physics and adaptation in VQA or video settings, for example IntPhys, CLEVRER, CoPhy, ComPhy, Physion and Physion++, ContPhy, PhysBench, Physics Context Builders, and context conditioned physics efforts, and related context driven benchmarks.

- Core novelty is thin. This is another templated synthetic collision suite with attribute rules tied to color. The few‑shot prompting setup is standard. It is unclear why the original CLEVRER dataset cannot be used for this task.

- Evaluation design confounds language bias by construction. Exemplars often include QA pairs, while the test uses only a single image frame. This invites template copying and makes the video‑only drop unsurprising. Claims that models ignore vision are therefore not well supported.

- Many questions require motion information that a single frame cannot provide. Penalizing Not enough data may punish reasonable uncertainty.

- No human baseline, no test of option prior balance, limited statistical analysis, and no side‑by‑side with earlier benchmarks under a shared protocol.

Overall, the paper’s empirical sweep is useful, but the novelty and claims are not strong, and key design choices undermine the main conclusions.

### Questions
- What exact boundary makes this benchmark first? If the novelty is exemplar videos plus VQA under rule changes, state that narrowly and revise the claim.

- Why is the query only a first frame? Do results hold if the model sees the full video at test.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper's motivation is that LMMs fail at inductive physical reasoning, which is the human-like ability to infer and apply unseen physical laws from a few examples. The authors introduce INPHYRE, a novel VQA benchmark of synthetic collision videos, to measure this skill as distinct from memorized parametric knowledge. The methodology compares LMM performance on regular scenarios (obeying true physics) against irregular ones that algorithmically violate universal laws (e.g., momentum conservation). Key findings show that LMMs not only possess weak parametric knowledge (struggling to apply laws they can recite) but also demonstrate poor inductive reasoning. The limited reasoning they do show is heavily language-biased, relying on textual cues in examples while largely ignoring the visual data.

### Strengths
- The paper introduces a new benchmark, INPHYRE, which uses physically impossible videos to test if LMMs can learn new physical laws from examples, rather than just repeating what they memorized. A key finding is that LMMs show a strong language bias; they primarily rely on text descriptions in the examples and struggle to learn from the visual-only videos.

- The study shows that LMMs are not only weak at learning new physical laws from examples but also struggle to correctly apply their existing, memorized knowledge of universal physics.

- The paper is well organized with clear logic. Finding 2 and 3 are interesting.

### Weaknesses
- The benchmark uses visually simple scenes with primitive shapes. Whether this failure on simple tasks means LMMs would perform even worse on complex, real-world videos, which are much noisier?

- The finding that LMMs ignore visual input in favor of text points to a serious trustworthiness problem. If models can recite a physical law but fail to apply it visually, does this mean they lack any real world model and are just manipulating text-based facts without genuine understanding?

- Some concurrent work is exploring the use of LMMs to guess world parameters, such as depth[1]. Perhaps the weakness highlighted in Finding 1 is simply because the models were not trained on that specific kind of data. Correct me if I am wrong.

[1] Cai, Z., Yeh, C. F., Xu, H., Liu, Z., Meyer, G., Lei, X., ... & Shi, Y. (2025). DepthLM: Metric Depth From Vision Language Models. arXiv preprint arXiv:2509.25413.

### Questions
See in the weakness.

### Soundness
3

### Presentation
4

### Contribution
3
