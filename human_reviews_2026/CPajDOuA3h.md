# RL-Obfuscation: Can Language Models Learn to Evade Latent-Space Monitors?

- Decision: Reject
- Scores: 4, 4, 6, 6

## Abstract
Latent-space monitors aim to detect undesirable behaviours in Large Language Models by leveraging their internal representations rather than relying solely on black-box outputs. These methods have shown promise in identifying behaviours such as deception and unsafe completions. However, these monitors may themselves become training signals, for example, by using problematic samples found in deployment to retrain models. This raises an important question: can models learn to evade such monitors? To evaluate this capability, we introduce RL-Obfuscation, in which LLMs are finetuned via reinforcement learning to evade latent-space monitors while maintaining their blackbox behaviour. We apply RL-Obfuscation to Language Models ranging from 7B to 14B parameters and evaluate their Evasion Success Rate against a suite of latent-space monitors. We find that token-level monitors are highly vulnerable to this attack while more holistic monitors, such as max-pooling or attention-based probes, remain robust. Moreover, for these vulnerable monitors, models trained to evade a single static monitor can generalise to evade other unseen monitors. We also find that the models can be trained to conditionally bypass latent-space monitors on only certain inputs. Finally, we study how the models bypass these monitors and find that the model can learn to repurpose tokens to have different internal representations.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Authors propose RL-Obfuscation, a reinforcement learning framework that fine-tunes models to improve the quality and coherence of harmful generations while reducing the chance of being detected by latent-space probes.  The method does not rely on monitor gradients.
Experiments show that models can bypass certain monitors, especially token-level linear probes, while attention-based or max-pooling monitors remain more robust. The trained models also generalize this evasion to unseen monitors, suggesting the formation of a broader evasion policy.
When trigger phrases are introduced, the models learn to activate evasive behavior only under specific conditions, similar to a latent backdoor.
Activation-space visualizations further show that RL-optimized models generate different internal representations even for the same text outputs, implying that evasion occurs through changes in latent encoding.  This finding reveals both the manipulability of internal representations and the limits of latent-space safety mechanisms.

### Strengths
The experimental design includes multiple models and different probe architectures, allowing a systematic cross-evaluation of evasion mechanisms. Each experiment is clearly motivated and aligned with the research questions, while the horizontal comparison across probes is clearly presented and insightful. If the paper could further elaborate on why certain probe types tend to be more easily bypassed, it would provide additional insight into the underlying differences in detection sensitivity and make the discussion even more informative.

### Weaknesses
## 1. Probe training and generalization

The probes are trained on outputs from the same models that later evaluate, which limits the assessment of generalization and validity. The setup does not ensure that the probes can reliably detect evasive behavior in unseen models or under shifted data distributions.
A more convincing validation would involve cross-model or cross-domain probe evaluations to demonstrate that the learned probes capture transferable representational signals rather than artifacts of a single model’s embedding space.

## 2. Lack of causal validation

The paper presents correlational evidence linking latent-space changes to evasion success but does not include activation-swap, layer-freezing, or intervention experiments that would establish causality, leading the interpretation that evasion is achieved through latent-space re-encoding remains speculative.

## 3. Dataset limitation and the interpretation of trigger mechanisms

Although the authors separate training and validation sets, both subsets are drawn from the same data source and are reused across all experiments. This design limits the statistical independence and external validity of the evaluation.
Given the relatively small dataset size and the lack of an out-of-distribution test set, the reported conditional trigger behavior may still reflect surface-level cues rather than a genuine latent-space switching mechanism, as the authors claim based on correlational evidence. However, this interpretation remains unsubstantiated, since the experiments demonstrate association but not causation between trigger-induced activations and evasion outcomes.

## 4. Lack of robustness and reproducibility

All reported results are based on single training runs without variance estimates or repeated trials. Given the stochastic nature of reinforcement learning, such outcomes may vary substantially across random seeds or initialization states.
Including multiple runs and reporting statistical variance would significantly improve the credibility and robustness of the findings

### Questions
See weaknesses

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
4

### Summary
This paper introduces RL-Obfuscation, a reinforcement learning-based methodology for stress-testing latent-space safety monitors in Large Language Models. The authors investigate whether LLMs can learn to evade activation-based monitors that detect harmful behavior by fine-tuning models using a dual-objective reward function that incentivizes both producing high-quality harmful content and lowering detection rates by white-box monitors, without requiring gradients through the monitoring system. Through comprehensive experiments across multiple probe architectures and token aggregation strategies, the authors demonstrate that models can successfully evade monitors using mean or median aggregation, and that this learned evasion generalizes to unseen monitors of similar architecture. However, attention-based probes and max-pooling aggregation remain robust to these attacks. The paper further shows that models can learn conditional evasion triggered by backdoor strings, and provides mechanistic analysis revealing that evasion occurs through both learning out-of-distribution output policies and modifying internal activations to appear less harmful.

### Strengths
+ Novel and Important Research Direction: The paper addresses a critical gap in AI safety research by investigating whether models can autonomously learn to evade safety monitors through RL, complementing existing adversarial attack methodologies.

+ Comprehensive Experimental Design: The study evaluates multiple model sizes (7B-14B parameters), various probe architectures (linear, MLP, attention), and different aggregation strategies (mean, median, max), providing broad empirical coverage.

+ Practical Red-Teaming Approach: RL-Obfuscation fills an important gap between input-space attacks and gradient-based attacks, offering a realistic threat model where adversaries have weight access but not gradient access to monitors.

### Weaknesses
+ Limited Scope of Latent-Space Monitors: The study focuses exclusively on probe-based monitors (linear, MLP, attention). Other monitoring approaches like SAE-based probes, latent OOD detectors, or ensemble methods are mentioned but not evaluated, limiting generalizability of conclusions.

+ Unclear Threat Model: The paper conflates different attacker capabilities—sometimes assuming black-box access (no gradients), sometimes white-box (full weights access for fine-tuning). The practical scenario where an adversary has compute to fine-tune a 14B model but cannot access monitor gradients needs better justification.

+ Insufficient Methodology Justification: Equations 4-5 lack theoretical or empirical justification for the specific reward formulation: Why this particular combination of SW_B, S_BB, and S_LP? The paper also does not provide ablation studies on alternative reward structures.

+ Incomplete Analysis of Failure Modes: While the paper shows attention probes and max-pooling are robust, the mechanistic understanding of why is limited. Is this fundamental or could it be overcome with different RL training?

+ Missing Baselines: No comparison to simpler evasion strategies (e.g., adding benign prefix/suffix, style transfer) to establish whether RL is necessary or if simpler approaches suffice.




Minor: 
+ Readability and Clarity Issues: The paper can be further improve its readability and clarity. For example, key experimental details are scattered across main text and appendices.

### Questions
Please respond to the weaknesses above.

### Soundness
2

### Presentation
2

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
The article proposes a new probing technique to evaluate the robustness of LLMs to evasion attack: RL-obfuscation.
Using reinforcement learning, LLMs are fine-tuned to evade monitoring methods, and ultimately make the attack pass.
This technique is tested against a range of monitors, for multiple models, showing a low robustness of token-based monitor, and a higher robustness of holistic monitors (based on max-pooling or attention).

### Strengths
The article introduces a novel technique to evade latent space monitoring, clearly states the conditions of the attack and show it's efficiency on a large set of probes, models, with various intensities and size of models.
The structure is clear, and summarized regularly with key take aways across the publication.
Very complete appendices are completing the article, which is appreciated for the details and the reproducibility.

### Weaknesses
As for any adaptive attack, the proposition of an adaptive defense following the same principles would have been appreciated, but this could be more fitting for future works.
The content of the article is quite dense, even if detailed, and was hard to read.

### Questions
Have you tried training jointly the monitors using the same RL process, in the loop?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper investigates the problem that LLMs can intentionally evade latent-space level safety monitors. The authors propose RL-obfuscation, where model is rewarded from maintaining original output performance while minimizing the possibility of being detected by latent-space detectors. They found that token-level monitors are vulnerable while more holistic monitors remain robust. Experiments across multiple model sizes support their findings.

### Strengths
1. The problem is novel and interesting. This is an interesting red-team method showing that LLMs can be adversarial trained to bypass latent-space monitoring system. 

2. The paper is well written and well presented. 

3. The findings are insightful and the analysis are in-depth, demonstrating that different types of monitors shows different robustness level toward this attack method.

### Weaknesses
1. The paper would be beneficial to expand the scope of tasks. Current experiments focus on synthetic tasks. Adding more full instruction-following or open-domain generation tasks can strengthen the paper. 

2. More interpretability on explaining how internal representations shift or which features the model manipulates can further strengthen the paper. 

3. More discussion on key-components on more robust latent-space monitors can further strengthen the paper.

### Questions
Please address the weakness above.

### Soundness
3

### Presentation
3

### Contribution
3
