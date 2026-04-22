# On Robustness and Chain-of-Thought Consistency of RL-Finetuned VLMs

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 4, 6

## Abstract
Reinforcement learning (RL) fine-tuning has become a key technique for enhancing large language models (LLMs) on reasoning-intensive tasks, motivating its extension to vision language models (VLMs). While RL-tuned VLMs improve on visual reasoning benchmarks, they remain vulnerable to weak visual grounding, hallucinations, and over-reliance on textual cues. We show that simple, controlled textual perturbations—misleading captions or incorrect chain-of-thought (CoT) traces—cause substantial drops in robustness and confidence, and that these effects are more pronounced when CoT consistency is taken into account across open-source multimodal reasoning models. Entropy-based metrics further show that these perturbations reshape model uncertainty and probability mass on the correct option, exposing model-specific trends in miscalibration. To better understand these vulnerabilities, we further analyze RL fine-tuning dynamics and uncover an accuracy–faithfulness trade-off: fine-tuning raises benchmark accuracy, but can simultaneously erode the reliability of the accompanying CoT and its robustness to contextual shifts.
Although adversarial augmentation improves robustness, it does not by itself prevent faithfulness drift. Incorporating a faithfulness-aware reward can restore alignment between answers and reasoning, but when paired with augmentation, training risks collapsing onto shortcut strategies and robustness remains elusive.
Together, these findings highlight the limitations of accuracy-only evaluations and motivate training and assessment protocols that jointly emphasize correctness, robustness, and the faithfulness of visually grounded reasoning.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper investigates the robustness and chain-of-thought (CoT) consistency of reinforcement-finetuned vision-language models (VLMs). The authors introduce controlled textual perturbations (e.g., Wrong-Caption and Wrong-Think) to test whether these models can resist misleading text and remain visually grounded. Results show that although RL fine-tuning improves benchmark accuracy, it reduces reasoning faithfulness and heightens sensitivity to misleading context. To mitigate these effects, the authors propose data augmentation and faithfulness-based reward methods that enhance both robustness and reasoning coherence.

### Strengths
The paper focuses on an important question—how Reinforcement Finetuning affects the reasoning faithfulness of large vision-language models.

### Weaknesses
1. The overall contribution of this paper is limited. It lacks technical innovation, does not present a solid benchmark, and offers no particularly insightful experimental findings.

    a. The paper primarily analyzes existing RL-trained VLMs through text perturbations. However, such perturbation-based evaluation has been extensively explored in prior works, such as [1].

    b. The benchmark proposed in this paper is mostly an extension of existing datasets with additional annotations of incorrect text or captions. This extension is incremental and does not provide substantial novelty or methodological advancement. Besides, it is not clear whether the additional annotations will be released.

    c. The proposed data augmentation strategy resembles approaches introduced in previous studies, such as [1]. Furthermore, the faithfulness-based reward function is conceptually similar to previously explored *process reward models* used in prior work, such as [2]

[1] Chen et al. PerturboLLaVA: Reducing Multimodal Hallucination with Perturbative Visual Training.

[2] Zhang et al. The Lessons of Developing Process Reward Models in Mathematical Reasoning.


2. The results in Table 1 show only marginal improvements or even degradations. Moreover, the evaluation relies on a single prompt configuration, making it difficult to assess the robustness and reliability of the reported results. To improve experimental validity, multiple prompt settings should be tested to evaluate consistency across prompt variations.


3. The assessment of faithfulness and reasoning trace quality depends solely on a single large-model judge. The absence of human evaluation or inter-annotator validation undermines the reliability of the reported metrics. It remains unclear whether the automated judgments accurately reflect true reasoning quality.

4. The used datasets assess simple spatial patterns and do not reflect complex multimodal reasoning or real-world understanding. It is unclear whether improvements or degradations on such tasks can generalize to more complex real world visual understanding.

5. The implementation details of experiments in Section 3 are unclear. Though the author provide details on how to generate captions and initial think, it is unclear how these annotations are used in training.

### Questions
1. How does your method differ from previous perturbation-based evaluations and previous design on process reward (e.g., [1], [2])?
2. Could you describe more clearly how the “initial think” and caption data are used in training?
3. Are the findings consistent when tested on more complex used, multimodal reasoning datasets, such as MMBench, V* Bench, MME, GQA?

### Soundness
2

### Presentation
1

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
This paper systematically evaluates the robustness and COT faithfulness of RL-tuned vision language models on basic visual reasoning, including counting, identity, and 2D and 3D spatial relations on static images. The authors augment established benchmarks: 3DSRBench, CV-Bench, Spatial-MM, and WhatsUp and add controlled textual perturbations that keep the image unchanged:: Stop-Think, Wrong-Think, and Wrong-Caption. Across five RL-finetuned models derived from Qwen2.5-VL-7B-Instruct, the study measures accuracy, robustness, and CoT consistency.

### Strengths
Focusing on robustness and faithful, visually grounded reasoning, the paper uses simple, controlled textual perturbations to effectively probe modality conflict.
By analyzing training dynamics, the paper indicates an accuracy–faithfulness tradeoff, shows that augmentation improves robustness while faithfulness continues to drift, and finds that adding faithfulness to the reward aligns CoT with answers yet becomes unstable when combined with augmentation, yielding limited robustness gains.

### Weaknesses
The paper mainly reveals the accuracy–faithfulness disconnect and sensitivity to textual perturbations, but does not provide training or inference method that can be readily reused.
The augmentation strategies with wrong-think and wrong-caption yield clear in-distribution improvements, but evidence for transfer across datasets and tasks is limited.
Out-of-distribution performance is under-reported, including results on different data sources and task types
Formatting error: “n Appendix D.1 we show that” seems it should be “In Appendix D.1 we show that.”

### Questions
Please report results on  spatial-relation datasets from different sources, outside the training mix, to quantify performance under distribution shift
Evaluate transfer without changing the training method (e.g., spatial relations, spatial relations → geometric reasoning) to distinguish answer accuracy from visually grounded reasoning.
Report out-of-distribution variants for Wrong-Think and Wrong-Caption separately.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents a systematic empirical study of the robustness and chain-of-thought (CoT) consistency of RL-finetuned vision-language models (VLMs). By introducing controlled textual perturbations—misleading captions and reasoning prefixes—the authors reveal that RL-tuned models are often misled by textual cues and exhibit an accuracy–faithfulness trade-off. They further analyze the effects of adversarial augmentation and faithfulness-aware rewards. The findings provide valuable diagnostic insights into how RL fine-tuning affects multimodal reasoning reliability.

### Strengths
+ Proposes a clear and reproducible textual perturbation framework to probe VLM robustness.

+ Identifies a consistent accuracy–faithfulness trade-off during RL fine-tuning.

+ Covers a wide range of recent RL-based multimodal reasoning models and benchmarks.

+ Analysis is careful, and the findings are both timely and practically relevant.

### Weaknesses
+ The paper remains primarily empirical, without a formal theoretical explanation or principled model-level intervention to mitigate the observed trade-off.

+ The faithfulness-as-reward experiments, though conceptually interesting, are underexplored; their instability and optimization dynamics merit deeper quantitative analysis.

+ The study relies on a single large-language-model judge (Qwen3-32B) to assess reasoning faithfulness, which may introduce evaluation bias; cross-validation with other judgment models would strengthen the claims.

+ While the work discusses potential solutions (e.g., richer reward signals, uncertainty modeling), these remain qualitative suggestions rather than systematically validated interventions.

### Questions
+ Have the authors considered developing a formal metric or theoretical framing for “faithfulness drift”? For instance, could the trade-off between accuracy and CoT consistency be modeled via reward attribution entropy or causal influence metrics?

+ Can the authors provide quantitative evidence—such as variance across seeds, reward gradients, or convergence plots—to substantiate the claim of unstable training dynamics?

+ Have alternative evaluators been tested to confirm that the faithfulness judgments are not artifacts of Qwen3’s inductive biases?

+ Could the authors empirically evaluate one or more of the suggested remedies (e.g., uncertainty-aware reward, contrastive consistency loss) to demonstrate their potential effectiveness?

### Soundness
3

### Presentation
4

### Contribution
3
