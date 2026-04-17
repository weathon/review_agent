# Spatial Mental Modeling from Limited Views

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 2

## Abstract
Can Vision Language Models (VLMs) imagine the full scene from just a few views, like humans do? Humans form spatial mental models, internal representations of unseen space, to reason about layout, perspective, and motion. Our new MindCube benchmark with 21,154 questions across 3,268 images exposes this critical gap, where existing VLMs exhibit near-random performance. Using MindCube, we systematically evaluate how well VLMs build robust spatial mental models through representing positions (cognitive mapping), orientations (perspective-taking), and dynamics (mental simulation for "what-if" movements). We then explore three approaches to help VLMs approximate spatial mental models, including unseen intermediate views, natural language reasoning chains, and cognitive maps. The significant improvement comes from a synergistic approach, "map-then-reason", that jointly trains the model to first generate a cognitive map and then reason upon it. By training models to reason over these internal maps, we boosted accuracy from 37.8% to 60.8% (+23.0%). Adding reinforcement learning pushed performance even further to 70.7% (+32.9%). Our key insight is that such scaffolding of spatial mental models, actively constructing and utilizing internal structured spatial representations with flexible reasoning processes, significantly improves understanding of unobservable space.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces MindCube, a new benchmark for evaluating the ability of VLMs to form "spatial mental models" from limited views. The authors show that existing models perform poorly, then propose a "map-then-reason" approach, which is to train a model to first generate a structured cognitive map and then reason upon it. This method leads to performance improvement through a combination of supervised fine-tuning and reinforcement learning.

### Strengths
1. High-quality benchmark and problem formulation. The proposed benchmark MindCube is well-designed and challenging, which systematically probes the critical VLM weakness of reasoning about unobserved space from multi-views. 
2. Systematic and rigorous experiments: The authors conduct a thorough investigation into different scaffolding methods with different input and output settings, logically progressing from frozen to trained models via SFT and RL. 
3. The "map-then-reason" paradigm yields a huge performance improvement, providing strong evidence for its efficacy.

### Weaknesses
1. Simplified "Cognitive Map" Representation: The paper's "cognitive map" is a practical choice but is a simplification of the richer cognitive science concept with JSON format, which may limit its generalizability towards more complex scenarios. 
2. Some Shaky Conclusions in the Paper: Regarding experiments, some of the claims remain quite shaky. For example, the conclusion that cognitive maps guide reasoning (L315) in frozen models is based on a marginal performance gain ( from 40.48 to 41.43, less than 1%, 10 data points) that is likely not statistically significant, especially considering the performance variance reported across seeds (in Table 14).

### Questions
1. The interesting finding that view interpolation is ineffective is not deeply analyzed. Could the authors elaborate on the reason for this failure? Is it due to the quality of the interpolated views, the increased context length, or is it further evidence that the primary bottleneck is reasoning, not perception?
2. Line 391 states a performance gain of "51.09%" for FF-Rsn, but this number does not appear in Table 4 (which shows 53.52%). Could the authors please clarify the source of this value?

I'm happy to adjust the scores if the author can address the concerns in the weaknesses and questions.

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
4

### Summary
This paper addresses the critical failure of VLMs in forming spatial mental models: the ability to reason about unseen spaces from limited views. To systematically evaluate this gap, the authors introduce the MINDCUBE benchmark, demonstrating that existing models perform near-randomly on such tasks. The paper's key contribution is a synergistic "map-then-reason" approach, which trains a model to first generate an internal cognitive map of an environment and then reason upon it. This method actively constructs and uses an internal spatial representation, especially when refined with reinforcement learning, can significantly boost task accuracy from a baseline of 37.8% to 70.7%.

### Strengths
1. The paper originally frames the issue of spatial reasoning through the cognitive science concept of "spatial mental models" and introduces MINDCUBE, a high-quality benchmark designed to rigorously diagnose this fundamental capability gap in current VLMs.

2. The "map-then-reason" approach is a highly effective solution, significantly outperforming methods that use pre-supplied maps or unstructured reasoning.

3. The study is distinguished by its rigorous and comprehensive experiments, including a critical bottleneck analysis that pinpoints the language reasoning module as the primary weakness, providing a robust foundation for its conclusions and clear direction for future research.

### Weaknesses
1. The proposed cognitive map is a significant oversimplification of the flexible and distorted mental models from cognitive science, potentially teaching the model a brittle form of reasoning that may not generalize to less structured environments.

2. The benchmark's reliance on static images evaluates the one-time synthesis of a mental model but not its crucial ability to dynamically update from a continuous visual stream, which may limit the relevance of the findings for real-world embodied agents that require ongoing spatial awareness.

### Questions
The paper's fine-tuning success is demonstrated exclusively on a single model family, leaving its generalizability as an open question. Have the authors tried other architecturally distinct VLMs?

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
The paper introduces MindCube, a multi-view reasoning benchmark tailored for evaluating whether VLMs can build "spatial mental models" of scenes from partial observations. Using MindCube, authors show that state-of-the-art VLMs perform near chance and struggle to maintain cross-view consistency or reason about occluded objects. The paper also explored three structural scaffolds (view interpolation, free-form reasoning, and cognitive maps) and find a synergistic "map-then-reason" approach yields the largest gains. Finally, they train models with SFT and reinforce with RL, and find that the joint map-then-reason setup with RL boosts accuracy, indicating that constructing and using internal structured maps substantially improves multi-view spatial reasoning.

### Strengths
* The paper addresses an important spatial, cognitive ability for VLMs, which is to understand the coherent 3D structure from partial multi-view observations. It would be essential for VLMs to be integrated into real-world embodied systems.

* The authors propose a novel, well-structured benchmark MindCube tailored for assessing the 3D spatial understanding ability of VLMs.

* The paper delivers a rich set of experiments to evaluate which “cognitive scaffold” best assists VLMs to do accurate spatial reasoning, and how post-training techniques like SFT and RL each enhance the spatial mental modeling process. Moreover, the authors present insights on the behavior of Qwen.2.5-VL on each of these configurations, providing a solid ground for future research on multi-view spatial reasoning.

### Weaknesses
While the paper includes a rich set of experiments, the **experiments setups for different cognitive scaffolds** seems quite naive. That is, the comparisons among different scaffolds may not be fair enough to conclude that "cognitive map" is the best choice among those. I outline some of the main concerns below.

* **View Interpolation:** It seems obvious that simply adding more interpolated views would not lead to great improvements. Expanding the visual context can inflate the token budget and increase hallucinations in VLM reasoning. Given that Qwen2.5-VL-3B is relatively low-capacity model compared to larger Qwen variants and other proprietary models, expecting it to improve solely from more views seems unrealistic. A more promising direction could to add a view selection phase to filter redundant perspectives. Additionally, it would be informative to test whether view interpolation becomes effective for higher-capacity models (e.g., > 7B models or proprietary models).

* **Cognitive Map:** The paper presents cognitive maps as a core contribution, but offers little analysis of **which representation formats** most benefit VLMs. Although the paper introduces an augmented version, it is simply addining more information to the original JSON-like format. This format seems to be highly influenced by the cognitive map prompting technique from Thinking-in-Space [1]. Have the authors tried exporing any variants or alternative for the cognitive map formats, rather than following Thinking-in-Space? A broader ablation over map formats would better justify the proposed design and its effectiveness.

* One of the main claims in the paper is that through SFT and RL Qwen2.5-VL can learn to "generate" a map and "reason" upon it, which leads to notable improvements on MindCube. But recent works in LLM reasoning have shown multiple evidence that the CoT reasoning traces in LLMs may not be logically related to the final output, and in fact they may be separate. In this perspective, I'm curious whether simply generating a map and returning the final output are in fact logically cause-and-effect relationship, or rather they may be mere byproducts similar to the CoT in LLMs [2, 3]. 

---

[1] Thinking in Space: How Multimodal Large Language Models See, Remember, and Recall Spaces, Yang et al., CVPR 2025

[2] Reasoning Models Don't Always Say What They Think, Chen et al., 2025

[3] Reasoning Models Can Be Effective Without Thinking, Ma et al., 2025

### Questions
* I agree that explicit reasoning can help VLM's spatial reasoning. However, it is unclear whether the paper’s "cognitive map" is truly explicit: the JSON-like maps are implicitly produced as token sequences by the LLM and only later decoded into a parseable structure. Would it be an feasible option to use a more explicitly constructed representation (e.g., using external tools to draw sketches as in Visual Sketchpad [1] or ViLaSR [2]), so that intermediate maps are drawn and manipulated directly rather than decoded from textual tokens?

---

[1] Visual Sketchpad: Sketching as a Visual Chain of Thought for Multimodal Language Models, Hu et al., NeurIPS 2024

[2] Reinforcing Spatial Reasoning in Vision-Language Models with Interwoven Thinking and Visual Drawing, Wu et al., NeurIPS 2025

### Soundness
2

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
3

### Summary
This paper focuses on the VLMs’ capabilities in "spatial mental modeling", the ability to imagine environments from a few visual observations. The paper first proposes a benchmark to measure current VLMs’ capabilities, finding that most existing models do not perform well on these tasks. Then, the paper explores methods to improve such capabilities through two approaches: (1) Scaffolds: carefully designed data structures to encourage spatial mental modeling, and (2) Training (SFT and RL). The paper identifies several scaffolds that could benefit spatial mental modeling and observes that combining SFT and RL leads to the best spatial reasoning performance.

### Strengths
1. The paper performs solid and substantial experiments, measuring various models and considering various settings when studying the spatial mental modeling ability.
2. The paper not only proposes a benchmark, but also proposes several methods to overcome the challenges of current VLMs.
3. Spatial cognition is an interesting and important problem, and many works have shown that it could be challenging for VLMs.

### Weaknesses
1. Benchmark Motivation: The benchmark aims to evaluate the ability to imagine a full scene from a few views. To me, this seems indirect, since humans and embodied agents usually understand environments through continuous observations or frames rather than 90-degree viewpoint shifts as shown in the proposed benchmark. This raises two concerns: (1.1) Can you provide real-world examples where a VLM must infer spatial relations from four orthogonal views? (1.2) More importantly, how does the proposed benchmark improve upon existing benchmarks that test MLLMs’ spatial cognition abilities given a video, e.g., [1]? The proposed benchmark seems like a special case of video input.

2. Benchmark Settings: The benchmark includes three settings (rotation, around, among). However, the number of “among” samples is significantly larger than the others (18,204 vs. around 1,000). Why is the dataset so imbalanced?

3. Proposed Scaffolds seem trivial: Section 3 introduces three categories of scaffolds that could potentially improve model performance. Based on Figure 3, these are specially designed input structures. How do they differ from traditional chain-of-thought prompting through the system prompt?

4. Clarity and Consistency: Some parts of the paper could be better organized for clarity and consistency. For example, in Table 2, what is the difference between FF-Rsn and FFR? They both seem to refer to free-form reasoning and should not use two different terms. Additionally, it is unclear what distinguishes Aug-CGMap-In from Aug-CGMap-Out, or Aug-CGMap-Out from Plain-CGMap-Out, etc. These distinctions can only be seen clearly through examples the Appendix, but the main text should describe them clearly and concisely.

5. Training: Based on Figure 4, SFT is conducted for around 55 steps. How large is the training dataset? Why does the training converge so quickly?

6. It is not surprising that SFT and RL on the specific tasks could improve the corresponding performance. The conclusion could be useful but the SFT and RL method does not introduce technical novelty.

[1] Thinking in Space: How Multimodal Large Language Models See, Remember, and Recall Spaces

### Questions
Please see the weaknesses above.

### Soundness
2

### Presentation
1

### Contribution
2
