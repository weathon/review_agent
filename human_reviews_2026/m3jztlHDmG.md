# Mixture of Cognitive Reasoners: Modular Reasoning with Brain-Like Specialization

- Avg Score: 5.20
- Decision: Accept (Poster)
- Scores: 4, 4, 6, 4, 8

## Abstract
Human cognitive behavior arises from the interaction of specialized brain networks dedicated to distinct functions, such as language, logic, and social reasoning. Inspired by this organization, we propose Mixture of Cognitive Reasoners (MiCRo): a modular, transformer-based architecture post-trained with a curriculum that induces functional specialization across experts. Concretely, we partition the layers of a pretrained language model into four expert modules aligned with well-studied cognitive networks in the human brain. MiCRo offers three key advantages over standard language models. (1) The specialized experts are interpretable and causally meaningful---ablating a module causes substantial drops on benchmarks requiring its specialized domain. (2) MiCRo's behavior can be dynamically steered at inference time by routing tokens to particular experts (e.g., favoring social over logical reasoning), enabling fine-grained control over outputs. (3) MiCRo outperforms or matches comparable baselines on both machine-learning reasoning benchmarks (e.g., GSM8K, BBH) and alignment to human behavior (CogBench), while maintaining interpretability. Taken together, cognitively grounded functional specialization yields models that are both more human-like and more human-interpretable.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes Mixture of Cognitive Reasoners (MICRO), a modular transformer architecture inspired by the brain’s functional specialization into distinct cognitive networks (language, logic, social reasoning, world knowledge). The model partitions each transformer block into four “experts” corresponding to these domains, trained through a three-stage curriculum intended to induce brain-like specialization. The authors claim this architecture enhances interpretability, controllability, and alignment with human behavioral benchmarks (COGBENCH), while maintaining competitive reasoning performance on standard NLP tasks (e.g., GSM8K, MATH, MMLU, BBH).

### Strengths
The motivating analogy to cognitive neuroscience is clear and interesting: the authors connect transformer modularity to the brain’s distributed but specialized functional architecture.

The paper is technically ambitious and proposes a relatively clean experimental pipeline (three-stage training) that is easy to reproduce conceptually.

The authors provide a comprehensive empirical evaluation, including behavioral alignment metrics, neuroscience “localizers,” and ablation analyses.

The inclusion of interpretable routing and causal ablation is a meaningful step toward testable hypotheses about functional decomposition in large models.

### Weaknesses
1. Conceptual clarity and motivation.
While the analogy to brain modularity is compelling, the paper does not clearly articulate why such modular specialization is desirable in language models. The claimed benefits—interpretability and controllability—are asserted but not demonstrated. The architecture yields mixed performance gains, suggesting that interpretability alone may not justify the added complexity.

2. Neuroscientific grounding.
The mapping between the four “expert” modules and the purported brain networks is overly categorical and simplified. The cognitive neuroscience literature remains divided on several of these assumptions. For example:
- The supposed separation between language and reasoning networks remains debated;
- The distinction between “logic” and “math” is underdefined, yet the paper treats them as part of a single module.
- Recent work (e.g., Hope Kean et al.) suggests that logical reasoning may rely on a distinct neural network separate from the multiple-demand system invoked here.
As such, the neuroscience framing may be more metaphorical than mechanistic.

3. Data labeling and methodology.
The “MICRO_SFT” dataset is central to inducing specialization, but the criteria for domain labeling are unclear. The dataset was pseudo-labeled using O1 and GPT-4o, but there is little evidence that these models’ judgments correspond to meaningful domain boundaries. No human validation or inter-rater reliability is reported. It remains uncertain whether the apparent “specialization” reflects genuine cognitive decomposition or artifacts of the labeling pipeline.

4. Relevance to cognitive neuroscience.
The authors seem to suggest (implicitly) that MICRO could bridge AI and brain science, but the connection is speculative. The architecture may be inspired by brain modularity, yet it does not provide new neuroscientific insight—no neural data are modeled, and the alignment tests (using functional localizers) are correlational. While ablation studies show that removing experts affects performance, this is a coarse-grained effect and not obviously interpretable at the cognitive level.

### Questions
How do you justify the specific choice of four networks, given the ongoing debates about their boundaries and overlap in the brain?

What validation steps were taken to ensure the O1/GPT-4o pseudo-labels correspond to human-like task domains?

How do you distinguish interpretability (as in mechanistic insight) from mere architectural labeling?

What concrete cognitive-neuroscience hypotheses does MICRO make that could be tested empirically?

Given the small size of the MICROSFT dataset (≈3k examples), how sensitive are your results to its composition or labeling noise?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a method (MiCRo) for post-training a transformer to incorporate specialized expert blocks across four domains: language, logic, social, and world knowledge. Inspired by the functional regions observed in brain networks, the approach involves a three-stage training process: (1) duplicating the original transformer blocks to create four expert blocks, each trained on data labeled at the sentence level according to domain; (2) freezing the network parameters and training routers to direct inputs to the appropriate expert blocks; and (3) conducting large-scale supervised fine-tuning. Experimental results demonstrate that, after training, these expert blocks exhibit distinct and specialized functionalities.

### Strengths
1.	The proposed method is well-motivated by the four modular functional regions in human brain.

2.	The paper is clearly written and easy to follow.

### Weaknesses
1.	The diversity and scale of tested models can be further enriched. The paper test Llama, SmolLM, and OLMO models of up to 3B parameters. I would recommend testing on 7B scale models from other model families, such as Qwen2.5-7B-Instruct. Small models sometimes behave very different from large models, so experiments on 7B scale models would make the results more convincing.

2.	The paper categorizes MMLU as a reasoning benchmark, which is not quite accurate. I would recommend extra experiments on MMLU-Pro (or a subset of it), which contains more reasoning-oriented questions.

3.	In Figure 7 and Table 3, the proposed MiCRo architecture does not outperform the dense model baseline on reasoning tasks, even though it has three times more parameters. While MiCRo-Ablation (which shows the best results when up to one expert is ablated) can sometimes achieve better performance than the dense model baseline, it requires four times more compute at test time. Therefore, the improvement of MiCRo on reasoning tasks is still limited.

### Questions
1.	The paper notes that small MOE models do not show expert specialization like the MiCRo counterpart. Is it due to the limitation of the model scale?

2.	If a question is related to more than one field (e.g., a question that is both related to math and social science), what will be the behavior of the MiCRo model? Are there any experiments on this?

3.	The paper mentions that using a small amount of data in Stage 1 and 2 suffices to elicit expert specialization behavior and that this specialization remains in the large-scale SFT stage. What is the intuition behind this phenomenon?

4.	In some subplots in Figure 4 (e.g., MiCRo-Llama-3B on MMLU_other), ablating any of the logic, social, or world expert can improve performance. It seems weird since it implies all of these experts are detrimental to the task. Are there any explanations on this phenomenon?

5.	The meaning of the marks “*” and “ns” in Figure 7 is not clear. One can guess that it shows the significance of the Welch’s t-tests, but it is better to explain them in the caption.

### Soundness
2

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
3

### Summary
This paper proposes MICRO (Mixture of Cognitive Reasoners), a novel mixture-of-blocks architecture designed to induce functional specialization across different experts. Unlike conventional MoE models that apply expert routing only to FFN layers, MICRO routes tokens through entire Transformer blocks, leading to emergent domain-specific experts (e.g., language, logic, social reasoning). The work is conceptually inspired by cognitive neuroscience and provides interesting empirical evidence for modularity in large language models.

### Strengths
- The paper is thought-provoking and connects cognitive science with model architecture in an elegant way. The results showing emergent specialization patterns and controllable routing behavior are intriguing. Overall, the idea of modeling “cognitive modularity” within Transformers is both fresh and potentially impactful.

- The paper is generally well written, and the experiments are clearly structured. The figures and analyses are helpful for understanding how modularity emerges under the proposed mechanism.

### Weaknesses
- While the modular design is inspired by human cognition, the model enforces top-1 routing per layer, meaning each token is processed by only one expert. In contrast, human reasoning typically involves parallel activation and cooperation among multiple brain regions. This exclusive routing assumption may limit the biological and functional plausibility of the approach.

### Questions
No.

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
3

### Summary
This paper introduces MICRO (Mixture of Cognitive Reasoners), a modular transformer architecture inspired by the functional specialization of human brain networks for language, logic, and social reasoning. The model partitions layers of a pretrained language model into four expert modules, each aligned with distinct cognitive domains, and is post-trained using a curriculum to promote specialization. MICRO offers three main advantages: (1) its expert modules are interpretable and causally meaningful, (2) it allows dynamic control at inference by routing tokens to domain-specific experts, and (3) it matches or exceeds baseline performance on reasoning benchmarks (e.g., GSM8K, BBH) and human-alignment tasks (CogBench). Overall, MICRO demonstrates that cognitively inspired modularity can enhance both the interpretability and human-likeness of large language models.

### Strengths
The novelty of MICRO lies in its cognitively inspired modular design, where a pretrained language model is partitioned into expert modules aligned with human brain networks for language, logic, and social reasoning. This structure enables interpretable, causally meaningful reasoning and dynamic control at inference, while maintaining or improving performance on both reasoning and human-alignment benchmarks.

The research provides:

•	Interpretable modules that provide causally meaningful insights into model behavior.

•	Dynamic control at inference, allowing selective routing to domain-specific experts.

•	Strong empirical performance, matching or exceeding baselines on reasoning and human-alignment benchmarks.

### Weaknesses
There are some shortcomings:

•	Increased model complexity due to multiple expert modules and modular routing.

•	Potential scalability issues for very large models or tasks requiring many cognitive domains.

•	The work makes a valuable contribution and builds effectively on current advances. However, including a discussion of remaining challenges and possible avenues for future research would strengthen the paper and highlight its long-term potential.

### Questions
Please discuss limitations of MICRO and how these limitations may be addressed.

It is unclear what is going on in Figs 4 and 16. Please explain the figures.

What is the key for Figure 5?

Please discuss the possible impact of this research.

Please discuss the generalizability of this research.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposes the Mixture of Cognitive Reasoners, a modular, transformer-based architecture that built based on neuroscience background with individual language, logic, social, and world experts. The training procedure of the model is composed by the following three stages: Training the experts with a small curated dataset to provide basic inductive bias; Training the router using the same dataset, with the experts frozen; Training the whole model with large instruction-tuning dataset. Experiments shows that the model acquires experts specialization, exhibits alignment to human behaviors, and matches dense and non–brain-aligned modular baselines on several tasks.

### Strengths
- The paper is well-written and understandable.
- The neuroscience motivation of the architecture design is very novel.
- The training procedure is novel.
- The experimental results are promising, showing that the proposed Mixture of Cognitive Reasoners model lead to strong interpretability and performance gain over other architectures.

### Weaknesses
- The language, logic, social, and world experts decomposition might not be optimal.

### Questions
I do not have additional questions at this stage.

### Soundness
4

### Presentation
4

### Contribution
3
