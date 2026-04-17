# Math Blind: Failures in Diagram Understanding Undermine Reasoning in MLLMs

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
Diagrams represent a form of visual language that encodes abstract concepts and relationships through structured symbols and their spatial arrangements. Unlike natural images, they are inherently symbolic, and entirely artificial.  They thus pose unique challenges for Multimodal Large Language Models (MLLMs) distinct from natural image processing. Recent studies have shown that MLLMs often exhibit flawed reasoning and hallucinations when handling diagram inputs.  We investigate here whether these limitations stem from shortcomings in the models' ability to interpret diagrams themselves. To this end, we develop a diagnostic test suite that isolates perception from reasoning. Our systematic evaluation reveals that MLLMs perform poorly on basic perceptual tasks, e.g., shape classification, object counting, relationship identification, and object grounding, with near-zero accuracy on fine-grained grounding. Further analysis shows that weak diagram perception leads to ``blind faith in text", where models rely on textual shortcuts rather than visual understanding (that is, they are $\textit{Math Blind}$). We hypothesize that enabling models to capture the inherent structural properties of diagrams, represented as graphs of primitives and their interrelationships, is essential for improving diagram understanding. Experiments with 7B and 32B MLLMs validate this assumption, with models trained on such representations achieving a +79\% gain on the grounding task. Crucially, these gains transfer to reasoning, achieving 3–4\% cross-suite improvements on four public benchmarks even without additional chain-of-thought reasoning data. Our findings demonstrate that low-level perception supports faithful high-level reasoning in mathematical MLLMs. We provide both methodological frameworks and empirical evidence to guide future research in this direction. Our project page is at \href{https://vi-ocean.github.io/projects/MATHEMETRIC/index.html}{\color{blue}{viocean/\ourMethod}}.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes a benchmark called MATHEMETRIC that just tests basic visual understanding, like counting shapes or finding relationships, separate from complex reasoning. They found that even top models fail these simple perception tasks, often just trusting the text blindly. Their fix is a new training dataset, GEOMETRIC, that explicitly teaches models the underlying structure of diagrams. When they trained models on this, perception scores shot up with a 79% gain on grounding. More importantly, this new visual skill transferred, making the models 3-4% better at actual high-level reasoning on other public benchmarks.

### Strengths
Primarily, for me, the main strengths of the paper are:

1) Transfer of Learning: The most significant result is the 3-4% improvement on downstream reasoning tasks (Table 2). This demonstrates that fixing the low-level perceptual foundation directly improves high-level reasoning, confirming the paper's central thesis. Achieving this gain without any new reasoning data was a useful finding

2) Isolation of variables: The authors rightly point out that existing benchmarks like MathVista and MathVerse conflate perception and reasoning, making it impossible to diagnose failure. The MATHEMETRIC benchmark is a good diagnostic tool that successfully disentangles these two capabilities.

### Weaknesses
A few weaknesses to point out:

1) The paper's solution dataset is synthetic and focuses on plane geometry. While the authors show some cross-domain generalisation to solid geometry and graphs (Table 2), the gains are far smaller. The "graph-based data construction" is not yet extended to these other domains. This makes the solution feel less general than the diagnosis, although the authors mention this in the limitation.

### Questions
I have a few required clarifications from the authors:

1) The 3-4% gain on MathVerse -  Could the authors provide more qualitative examples in the appendix showing what kinds of reasoning problems the GEOMETRIC-trained models now solve correctly? Are they all problems that hinge on a single, correctly perceived geometric property, or does the improved perception unlock more complex, multi-step reasoning chains?

2) The text-distractor experiment- While it shows that models over-rely on conflicting text. Did the authors perform the inverse experiment: providing a diagram with clear visual cues (e.g., two parallel lines) but no ambiguous or ambiguous textual information? Do models still fail, or can they reason from the visual-only input? This would further strengthen the "blind faith in text" hypothesis.

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
4

### Summary
This paper investigates whether the failures of current Multimodal Large Language Models (MLLMs) in mathematical reasoning stem from limited diagram perception. The authors introduce MATHEMETRIC, a benchmark designed to isolate perceptual abilities (shape classification, object counting, relationship identification, grounding) from reasoning, and a synthetic GEOMETRIC dataset that encodes diagram structure as graphs of shapes and relations. Experiments on 18 MLLMs demonstrate that most models perform poorly on basic perceptual tasks, and that improved perceptual training can yield modest reasoning gains across MathVerse, MathVista, and GeoQA.

### Strengths
1. **Timely and relevant topic**: The paper addresses an important and underexplored aspect of multimodal reasoning: diagram perception, aligned with ongoing interest in evaluating the visual grounding of MLLMs.

2. **Benchmark contribution**: The proposed MATHEMETRIC benchmark provides a systematic framework to separate low-level perception from reasoning, which is valuable for diagnosing model bottlenecks.

3. **Synthetic dataset design**: The GEOMETRIC dataset introduces structured, graph-like representations of diagrams that could facilitate learning of spatial and relational structures.

4. **Comprehensive evaluation**: The study covers a broad set of models and tasks, including both open- and closed-source MLLMs, and provides ablation studies on factors such as distractors, noise, and text–image conflicts.

### Weaknesses
1. **Lack of validation for benchmark annotation quality.**
Since MATHEMETRIC and GEOMETRIC are key contributions, the paper should provide validation or human verification to ensure the accuracy and reliability of the annotated data. Without such validation, it is difficult to assess whether the benchmark truly measures perceptual ability rather than artifacts of synthetic generation.

2. **Unclear causal interpretation of findings.**
The central question "Do current MLLMs genuinely perceive mathematical diagrams?" is not convincingly answered. The reported evidence (e.g., reliance on text, pattern memorization, robustness to distractors) does not establish causal links between perception and reasoning. The argument that stronger perception leads to better reasoning remains correlational, lacking controlled experimental design or counterfactual analysis.

3. **Ambiguity in defining "genuine perception".**
Claims about "blind faith in text" or "Math Blindness" are conceptually interesting but not rigorously operationalized. The study could benefit from clearer definitions or measurable criteria for "genuine diagram perception".

4. **Potential shortcut effects.**
If improved perception data lead to higher reasoning accuracy, it remains unclear whether this reflects better perceptual grounding or merely learning shortcuts in synthetic data. This ambiguity weakens the interpretation of performance improvements as evidence of genuine understanding.

5. **Writing and organization issues.**
The writing could be more concise and polished. Certain sections (e.g., section 3.3) contain excessive detail that obscures the main findings, while others (e.g., motivation and discussion) could better connect empirical results to claims.

### Questions
See Weakness

### Soundness
2

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
This paper investigates a critical but underexplored limitation of Multimodal Large Language Models (MLLMs): their inability to perceive mathematical diagrams correctly, leading to reasoning failures. The authors introduce two main contributions: 1. MATHEMETRIC, a diagnostic benchmark that isolates low-level perception (shape classification, counting, relationship identification, grounding) from high-level reasoning across three domains—plane geometry, solid geometry, and graphs. 2. GEOMETRIC, a structured training dataset encoding diagrams as graphs of geometric primitives with attributes and relationships.

Through evaluations of 18 MLLMs (generic and math-specific), the paper demonstrates that current models fail severely in fine-grained perception (e.g., <20% in grounding) but that training on GEOMETRIC yields substantial perception gains (+79% grounding accuracy) and modest reasoning improvements (+3–4% on MathVerse/MathVista). Finally, the authors argue that perception is a fundamental bottleneck for reasoning and that structure-aware visual pretraining can partially bridge this gap.

### Strengths
The work introduces a novel and valuable perspective on multimodal reasoning by isolating diagram perception from mathematical reasoning, a distinction often blurred in existing benchmarks like MathVista and MathVerse. The proposed “Math Blind” hypothesis is conceptually fresh and empirically grounded, offering a clear diagnostic lens for assessing visual understanding in symbolic domains. The paper is clearly written, visually well-presented, and effectively highlights a long-overlooked challenge in MLLMs—the perception of structured, symbolic diagrams.

### Weaknesses
- The paper operationalizes perception through four basic CV-style tasks (classification, counting, grounding, relationship). This oversimplifies the rich perceptual hierarchy involved in diagram understanding, which may include topological reasoning, implicit constraints (e.g., angle or symmetry inference), and multi-object composition.
- The claimed perception-to-reasoning transfer (+3–4%) is small compared to the massive perception improvement (+79%), weakening the causal claim that “low-level perception is the main bottleneck of reasoning.”
- Recent high-performing multimodal reasoning models such as Vision-R1, MINT-CoT [Chen et al., 2025], and the MathVision benchmark [Wang et al., 2024] are not included.
- The “blind faith in text” phenomenon is intriguing, yet the paper lacks controlled prompts that explicitly modulate text–image priority (e.g., “prefer textual information” vs. “prefer visual cues”). Without this, the analysis remains observational, not causal.

### Questions
1. How would your findings change under interleaved visual token reasoning paradigms (e.g., MINT-CoT [1]) where visual tokens are mixed directly with text during CoT generation? Could these newer paradigms render the perception–reasoning separation less meaningful?

2. When textual and visual information conflict, how do you determine which should be “correct”? Have you tried prompting models with explicit modality preference instructions to measure alignment sensitivity in the relative experiments (“blind faith in text”)?

3. The inclusion of recently released reasoning-enhanced models, such as Vision-R1 (Huang et al., 2025), needs to be added to strengthen the study's central claim regarding perception as the primary bottleneck. 

4. To enhance the evaluation, it is recommended to incorporate advanced multimodal math benchmarks like MathVision [3].


References:
 [1] Chen et al., MINT-CoT: Enabling Interleaved Visual Tokens in Mathematical Chain-of-Thought Reasoning, arXiv:2506.05331, 2025.
 [2] Huang et al., Vision-R1: Incentivizing Reasoning Capability in Multimodal Large Language Models, arXiv:2503.06749, 2025.
 [3] Wang et al., Measuring Multimodal Mathematical Reasoning with MathVision Dataset, NeurIPS 2024.

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
This paper address an important aspect of evaluating MLLMs on visual math problems: Isolating the perception of simple visual concepts from complex mathematical reasoning and problem solving. To this end, the authors present a benchmark that tests basic perception skills like counting, shape recognition, and locating objects, and a training dataset of geometric shapes and relationships to improve the models' perfromance. Their results show while models peform poorly on the benchmark, using the training set their performance improves, and even results in improvements on the other reasoning benchmarks.

### Strengths
- This paper addresses an important challenge in evaluating the performance of the MLLMs on visual understanding and conceptualization
- The benchmark is curated with a high quality, the labels are generated in an accurate way (not like some work that use LLMs for sample generation/annotations)
- The paper also provides a training set for training the models
- The results of training using this set are strong, with high improvements on their proposed bencmark, but more interestingly, improvement on other reasoning benchmarks (MathVerse and MathVista)

### Weaknesses
- The object grounding task (with. example in Figure 3, rightmost) is not. It is of no surprise that the models are performing so poorly on this task, given the difficult nature of finding the exact pixel coordinates from an image, which is at the same time unnecessary for visual perception and reasoning. A better alternative to this task would have been to ask the models to give the object vertex letters (for instance in the same task: Q: Please provide the List the vertex labels of the object: scalene triangle. A: HVI)
- The shape classification task can have a shortcut for models: If they see a vertex label of three letters, it is a triangle, so even without the image they can response correctly if only one answer in the answer choices is a triangle, or with a higher probability than random (0.25) if not all answer choices are triangles. 
- Some of the graphs and figures could improve for clarity. For instance, Figure 1 is hard to compare different models. maybe a radar plot could have been more illustrative. Also the colors of Figure 5 could be chosen better. It's quite hard to follow the light brown and yellow colors.
- While the paper states that the tasks are easy for humans multiple times, there is no human performance baseline provided. Going back to the first point, the grounding task would be impossible for humans only with looking at the images.

I am willing to increase my score if these concerns are addressed.

### Questions
- Could you please give some details on the size and samples of the training set (GeoMetric)?
- How do you make sure that the weakness 2 is not happening?
- Your work resembles a recent work on conceptualisation (Visual Graph Arena: Evaluating Visual Conceptualization of Vision and Multimodal Large Language Models, ICML2025). In that works the authors observe two anomalies in perception, the Easier-Worse and  Middle-Score. Did you observe any similar anomalies in your experiments?
- Do you have an idea to evaluate not only the final answer, but also the models' CoT accuracy on the benchmark? (sometimes the models give the correct answers by chance, with the wrong CoT)

### Soundness
3

### Presentation
3

### Contribution
4
