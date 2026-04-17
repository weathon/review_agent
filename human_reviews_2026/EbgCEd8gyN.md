# Sysformer: Safeguarding Frozen Large Language Models with Adaptive System Prompts

- Decision: Accept (Poster)
- Scores: 4, 6, 4, 6

## Abstract
As large language models (LLMs) are deployed in safety-critical settings, it is essential to ensure that their responses comply with safety standards. Prior research has revealed that LLMs often fail to grasp the notion of safe behaviors, resulting in either unjustified refusals to harmless prompts or the generation of harmful content. While substantial efforts have been made to improve their robustness, existing defenses often rely on costly fine-tuning of model parameters or employ suboptimal heuristic techniques. In this work, we take a novel approach to safeguard LLMs by learning to adapt the system prompts in instruction-tuned LLMs. While LLMs are typically pre-trained to follow a fixed system prompt, we investigate the impact of tailoring the system prompt to each specific user input on the safety of the responses. To this end, we propose Sysformer, a transformer model that updates an initial system prompt to a more robust system prompt in the LLM input embedding space while attending to the user prompt. While keeping the LLM parameters frozen, the Sysformer is trained to refuse to respond to a set of harmful prompts while responding ideally to a set of safe ones. Through extensive experiments on 5 LLMs from different families and 2 recent benchmarks, we demonstrate that Sysformer can significantly enhance the robustness of LLMs, leading to upto 80% gain in the refusal rate on harmful prompts while enhancing the compliance with the safe prompts by upto 90%. Results also generalize well to sophisticated jailbreaking attacks, making LLMs upto 100% more robust against different attack strategies. We hope our findings lead to cheaper safeguarding of LLMs and motivate future investigations into designing variable system prompts.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes SysFormer, an adaptive system prompt optimization framework designed to safeguard large language models (LLMs) against harmful prompts. The method introduces a trainable transformer that optimizes system prompts in the input embedding space, enabling the model to refuse harmful prompts while maintaining proper responses to safe ones. Experiments conducted on five LLMs and two benchmarks demonstrate the method’s effectiveness in improving safety without retraining model parameters.

### Strengths
- The paper is well-written and generally easy to follow.
-  The idea of enhancing LLM safety via adaptive system prompt refinement is an important and timely research direction, especially as system prompts become a key component of deployed LLM systems.
- The experimental results suggest that the proposed method can improve refusal behavior across multiple models.

### Weaknesses
1.	Threat model clarity:
The paper’s threat model needs clearer justification. If the goal is to protect models using developer-provided system prompts, then the possibility of double-jailbreak attacks should be considered. On the other hand, if attackers do not have access to the system prompt, the described threat scenario may not fully hold.
2.	Training complexity and stability:
The proposed optimization involves multiple loss terms. It remains unclear how these losses are balanced and whether training is stable and convergent in practice. A discussion or ablation study on this would strengthen the paper.
3.	Comparison with related work:
A closely related approach, SOP (Adaptive Content Restriction for Large Language Models via Suffix Optimization, 2025), also optimizes system suffix components for output control. The paper should include a direct comparison or discussion to clarify the conceptual and empirical differences between SysFormer and SOP.
4.	Alternative design choices:
Why not use a smaller auxiliary model or a lightweight controller to enhance or rewrite the system prompt dynamically? Direct instruction-level modification could be more straightforward—please justify this design decision.
5.	Optimization domain and textual space:
Since the optimization is performed in the embedding space, it is unclear whether similar effects can be achieved directly in the textual space (e.g., using gradient-guided optimization such as GCG). A comparison or reasoning would be valuable.
6.	Comparison with decoding-based defenses:
Some decoding-level defenses can repair or filter harmful outputs more efficiently without modifying inputs. The paper should provide comparisons or explain why SysFormer is preferable in terms of flexibility or deployment.
7.	Transferability across models:
How transferable is the learned system transformer? Can the same parameters generalize to different LLMs, or is separate optimization required for each model? This issue affects the scalability of the approach.
8.	Black-box applicability:
The paper does not discuss performance on black-box models (e.g., GPT series). Since system prompt control is particularly relevant for black-box deployments, experiments or analysis in this setting would be important to demonstrate broader applicability.

### Questions
Overall, this paper explores a promising and practically relevant idea—leveraging adaptive system prompts for improving LLM safety. However, several conceptual and empirical issues remain open, particularly regarding the threat model, training stability, and comparison with existing methods. Addressing these concerns would significantly strengthen the contribution. I encourage the authors to expand the analysis, include more comprehensive baselines (especially SOP and decoding-based defenses), and clarify the deployment assumptions. I would like to reconsider my rating after reading the authors' response.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents a method for LLMs Jailbreak defense by generating an input-dependent system prompt. The advantage of this method is that the target LLM to be protected can be freezed, in other words, there is no need to finetune it. Experiments on two datasets show the proposed method has good defense performance.

### Strengths
1. The proposed method is novel to my knowledge.

2. The defense effectiveness is good.

3. This paper is well written.

4. The defense method does not rely on finetuning the target LLM to be protected.

### Weaknesses
1. The baseline methods compared in this paper are very scarce. Many prompt based especially system prompt based defense methods are not discussed or compared at all.

2. The proposed method relies on an additional dataset for training the prompt generation model. It is not clear how the proposed method relies on the size and quality of the training data. In addition, it is unclear whether the proposed method can work for the new attacks which are not covered by the training data.

3. The proposed method needs to generate a new prompt for each user request, which brings additional computation and delay.

4. The experiments are conducted on small and weak LLMs. It is unclear whether the findings hold for frontier models.

### Questions
see my above comments

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
3

### Summary
The paper introduces Sysformer, a lightweight transformer module that enhances LLM safety by adapting the system prompt based on each user input instead of fine-tuning model parameters. The transformer module transforms the system prompt in embedding space to enforce refusals on harmful prompts and compliance on safe ones, offering an efficient, modular approach to safeguard frozen LLMs through adaptive system-prompt optimization.

### Strengths
+ The paper reads smooth and clear.
+ The baseline evaluation is rather comprehensive, containing efficienct fine-tuning (LoRA) and embedding space optimization. Dataset selection looks good.

### Weaknesses
- The transformer component takes in user prompts, which means the embedding prompt is generated on every query. While the motivation statement criticized efficiency of prior defense methods, Sysformer also introduces overhead but not evaluated.
-  The traiing loss uses predefined fixed strings like "I cannot help you" as a signal of refusal, which restricts the flexibility of the training method. Not sure if the training pipeline is working on larger and more powerful models that do not answer fixed strings as refusal (like GPT-5). It is also a risk of overfitting.
- The evaluation does not evaluate the quality of answers to safe questions. Does the injected embedding ever harm model performance in normal tasks like text comprehension, math, etc?
- As demenstrated in the evaluation, Sysformer cannot defend unseen jailbreaking attacks. The dependence on he training data limits the usefulness of Sysformer, as data nowadays is a bottleneck of model development. Requiring the data of attack also opens the oppotunity of adaptive attacks.

### Questions
* If a model does not have clear fixed string for refusal, how should the training loss be computed?
* Will Sysformer affect the quality of answering normal prompts?
* While Sysformer can defend jailbreaking attacks by augmenting the training with the corresponding data, can jailbreaking attacks also evolve to defeat Sysformer given the knowledge of the transformer component?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents Sysformer, a lightweight transformer model that adapts the system prompt for a frozen LLM. By doing so, it significantly improves the model’s safety while maintaining compliance on safe prompts, and is deployable without full model retraining. It offers a practical step toward safer LLM deployment in real‐world settings.

### Strengths
The paper uses multiple benchmarks—JailbreakBench and StrongReject—plus 16 jailbreak variants. The proposed method show a strong empirical performance on these benchmarks, shows that adaptive system prompts can meaningfully improve LLM safety and robustness without modifying model weights.

### Weaknesses
While the paper includes solid ablation studies on loss components and demonstrates impressive generalization to unseen jailbreak attack types, it does not assess cross-benchmark transfer — e.g., training Sysformer on JailbreakBench and evaluating on StrongReject (or vice versa). As a result, it remains unclear how well the learned safety behavior generalizes to qualitatively different harmful-prompt distributions. Including such a cross-dataset evaluation (or at least reporting zero-shot transfer results) would strengthen the claim that Sysformer captures general safety principles rather than dataset-specific artifacts.

### Questions
Since Sysformer is trained on labeled data from existing safety benchmarks, it is unclear how general the method is to new domains or harmful behaviors. It would be very helpful to see an ablation where Sysformer is trained on one benchmark and evaluated on another, with comparison to the baselines, to better assess its cross-benchmark generalization.

### Soundness
2

### Presentation
3

### Contribution
2
