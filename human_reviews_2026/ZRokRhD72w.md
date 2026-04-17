# Mini-Omni-Reasoner: Token-Level Thinking-in-Speaking in Large Speech Models

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 6, 2, 2

## Abstract
Reasoning is essential for effective communication and decision-making. While recent advances in large language models (LLMs) and multimodal models (MLLMs) have shown that incorporating explicit reasoning significantly improves understanding and generalization, reasoning in large speech models (LSMs) remains in a nascent stage. Early efforts attempt to transfer the “thinking-before-speaking” paradigm from textual models to speech. However, this sequential formulation introduces notable latency, as spoken responses are delayed until reasoning is fully completed, impairing real-time interaction and communication efficiency. To address this, we propose Mini-Omni-Reasoner, a framework that enables reasoning within speech via a novel “thinking-in-speaking” formulation. Rather than completing reasoning before producing any verbal output, Mini-Omni-Reasoner interleaves silent reasoning tokens with spoken response tokens at the token level. This design allows continuous speech generation while embedding structured internal reasoning, leveraging the model’s high-frequency token processing capability. Although interleaved, local semantic alignment is enforced to ensure that each response token is informed by its preceding reasoning. To support this framework, we introduce SPOKEN-MATH-PROBLEMS-3M, a large-scale dataset tailored for interleaved reasoning and response. The dataset ensures that verbal tokens consistently follow relevant reasoning content, enabling accurate and efficient learning of speech-coupled reasoning. Built on a hierarchical Thinker–Talker architecture, Mini-Omni-Reasoner delivers fluent yet logically grounded spoken responses, maintaining both naturalness and precision. On the Spoken-MQA benchmark, it achieves a +19.1% gain in arithmetic reasoning and +6.4% in contextual understanding, with shorter outputs and zero decoding latency. These results demonstrate that high-quality reasoning and real-time spoken interaction can be effectively unified in a single framework.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces Mini-Omni-Reasoner, a large speech-language model designed under a new “thinking-in-speaking” paradigm. Instead of the conventional thinking-before-speaking strategy that delays spoken responses until reasoning completes, Mini-Omni-Reasoner interleaves reasoning and response tokens during generation. This enables real-time reasoning and fluent spoken responses with near-zero latency. The authors also release a large-scale dataset, Spoken-Math-Problems-3M, and design a 5-stage training pipeline to align reasoning with spoken output. On spoken mathematical reasoning benchmarks (e.g., Spoken-MQA, AddSub, SVAMP), the model significantly outperforms prior LSLMs and performs comparably to or better than larger text-speech models, achieving both shorter response length and improved reasoning accuracy.

### Strengths
1. The “thinking-in-speaking” formulation is clearly motivated by modality constraints and addresses the latency issue in the “thinking-before-speaking” paradigm. This is an interesting conceptual shift with potential implications beyond math tasks.
2. The paper combines algorithmic design (token-level interleaving), dataset construction (Spoken-Math-Problems-3M), and a new model (Mini-Omni-Reasoner). Together they demonstrate an end-to-end system for real-time reasoning in speech.
3. The paper is well-written and easy to follow.

### Weaknesses
1.Although the 2:8 speech-to-reasoning token ratio is justified by hardware throughput, hardcoding this ratio limits adaptability. If deployment hardware or inference speed changes, both model and dataset construction would need to be redone. It would strengthen the work to include an ablation or adaptive mechanism for this ratio. In related LLM work such as Quiet-STaR [1], increasing the number of “thinking” tokens can systematically improve reasoning. I am wondering if this is case for Mini-Omni-Reasoner.
2. The main evaluation is restricted to spoken math reasoning. While this is a clean domain for analysis, it is unclear whether “thinking-in-speaking” generalizes to other reasoning-heavy but natural conversational tasks. The claimed fluency and precision improvements are qualitatively supported, but more quantitative user or perceptual studies would be valuable.
3. The paper claims that detokenization checks guarantee “no overshooting,” but the guarantee seems heuristic—based on alignment checks and empirical verification rather than a theoretical bound.

[1] Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking, Zelikman et al., COLM 2024

### Questions
1. How can the data construction pipeline ensure there is truly no overshooting issue? The paper mentions "We introduce a prompting strategy that defers substantive content in the output stream while frontloading reasoning steps in the internal stream", but this seems heuristic rather than provable.

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
3

### Summary
- This paper introduces Mini-Omni-Reasoner, a framework that enables large speech models to perform reasoning while generating spoken responses through a novel "thinking-in-speaking" paradigm. 

- The motivation for this method is reduce the latency caused by traditional thinking-before-speaking approach. In order to validate the effectiveness of the model, the authors construct a 3M sample dataset and demonstrate improvements on mathematical reasoning benchmarks.

### Strengths
- The motivation for the paper is clear, the author would like to adress the latency issue in speech reasoning.
- The structure of the paper is clear and easy to follow.
- The authors design a complete system regarding dataset construction, and well-designed training pipeline.
- Experiment results stronly demonstrate the effectiveness of the model with 19.1% arithmetic improvement, +6.4% reasoning accuracy improvement and 63% shorter responses than baselines

### Weaknesses
- Limited evaluation scope: Only mathematical reasoning tasks are evaluated. No evidence show this can generalize to other domain, e.g. dialogue, creative tasks.
- Weak ratio justification: 2:8 ratio derived from GPU throughput (100 vs 12.5 tokens/sec), not cognitive or linguistic principles. No perceptual studies justify 20 spoken tokens/sec target. No ablations testing 1:9, 3:7, 4:6, or dynamic ratios

### Questions
1. Does this work for anything besides math problems?
2. What's the actual measured latency vs. thinking-before-speaking?
3. How do you know reasoning tokens actually improve responses (not just generated independently)?
4. Why not test other ratios?
5. Where's the human evaluation?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes Mini-Omni-Reasoner, a model that enables reasoning while speaking by interleaving reasoning tokens with response tokens during training. The model is evaluated on mathematical reasoning tasks. Experimental results show decent improvements in arithmetic reasoning and a significant reduction in response tokens compared to baselines.

### Strengths
- Each step of the training procedure is explained clearly and in sufficient detail.
- The idea is interesting and practical. It could make interaction with the model more convenient by reducing user waiting time after speaking.
- Experimental results include multiple baseline models, making the comparison relatively comprehensive.

### Weaknesses
- Although the main idea is "thinking while speaking", the interleaving order between reasoning and response tokens is reversed -- the model first generates response tokens, then reasoning tokens. In this setup, the reasoning tokens serve as post-hoc justification rather than genuine reasoning, which contradicts the motivation of the paper.
- The numbers of response and reasoning tokens per interleaving step are fixed (2 and 8, respectively). However, different parts of intermediate steps may require varying levels of reasoning. A fixed ratio limits flexibility. A dynamic allocation depending on intermediate reasoning difficulty would be more realistic.
- The method is evaluated only on math datasets without clear justification. It is unclear why the authors did not test on more open-ended reasoning or conversational tasks.
- While the paper claims to reduce latency, the analysis focuses instead on token efficiency (accuracy vs. number of words).
- The approach introduces an additional hyperparameter -- the ratio between reasoning and response tokens -- but the paper does not explore different settings or analyze sensitivity to this ratio.

Overall, the analysis does not fully substantiate the claimed advantages of the proposed method.

### Questions
Why is Qwen2.5-Omni-3B listed as a baseline, while Qwen2.5-Omni-7B and others are referred to as conversational models? What is the distinction between these two categories?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes Mini-Omni-Reasoner, a framework enabling real-time spoken reasoning through a “thinking-in-speaking” paradigm. Instead of finishing internal chain-of-thought before speaking, the model interleaves internal reasoning tokens with spoken tokens, leveraging the gap between model inference speed and audio playback rate. The system is built on the Thinker-Talker architecture and includes a large dataset (Spoken-Math-Problems-3M) to train temporally aligned reasoning and speech traces. Experiments on the Spoken-MQA benchmark show improved mathematical reasoning quality and near-zero latency compared to conventional think-before-speak models.

### Strengths
1. Large-scale dataset contribution.
The paper introduces a 3M-sample spoken math-reasoning dataset with paired reasoning traces. This is a valuable resource and, if released, could meaningfully benefit the community and future research in speech-based reasoning.

2. Clear writing and presentation.
The problem, motivation, and methodology are presented clearly, supported by informative figures and conceptual diagrams that make the technical ideas easy to follow.

3. Comprehensive ablation and analysis.
The experimental section provides a wide range of ablations and analyses, helping to understand the impact of key design decisions.

### Weaknesses
1. Limited task scope and unclear generalization.
The method is evaluated only on spoken mathematical reasoning and a single benchmark (Spoken-MQA). While math is a structured domain, the ability to generalize to broader real-world multimodal tasks remains unclear. The current claims are not strong enough given the effectiveness beyond math has not yet demonstrated.

2. Fixed token-ratio design lacks justification and appears overly hand-crafted. The 2:8 speech-to-reasoning token ratio is motivated by inference-budget considerations, yet the paper does not demonstrate whether this choice is optimal or robust across different tasks and scenarios. It remains unclear whether the ratio should adapt dynamically to task complexity, interaction style, or model behavior. If the ratio must be manually tuned per task or application, this introduces significant human overhead and undermines the generality of the proposed approach. Exploring learned or adaptive strategies would strengthen the contribution.

3. Most important, insufficient evidence that “thinking-in-speaking” is superior. The paper argues that thinking-in-speaking is better than think-before-speaking, but does not provide direct studies comparing the two paradigms under the same conditions.
Additionally, think-before-speaking intuitively provides fuller context and may generalize more robustly across tasks, whereas “thinking-in-speaking” may require careful tuning (e.g., token-ratio) and offers only partial internal reasoning capacity during speech. Stronger empirical comparison is needed to convincingly justify the paradigm shift.

Overall, while the core idea seems fancy, the current experimental evidence does not convincingly validate the proposed paradigm.  It requires stronger experimental support before it can be considered mature for publication.

### Questions
see weakness

### Soundness
3

### Presentation
3

### Contribution
2
