# ASGuard: Activation-Scaling Guard to Mitigate Targeted Jailbreaking Attack

- Decision: Accept (Poster)
- Scores: 6, 4, 2, 6

## Abstract
Large language models (LLMs), despite being safety-aligned, exhibit brittle refusal behaviors that can be circumvented by simple linguistic changes.
As tense jailbreaking demonstrates that models refusing harmful requests often comply when rephrased in past tense, a critical generalization gap is revealed in current alignment methods whose underlying mechanisms are poorly understood.
In this work, we introduce Activation-Scaling Guard (ASGuard), an insightful, mechanistically-informed framework that surgically mitigates this specific vulnerability.
In the first step, we use circuit analysis to identify the specific attention heads causally linked to the targeted jailbreaking such as a tense-changing attack.
Second, we train a precise, channel-wise scaling vector to recalibrate the activation of tense vulnerable heads.
Lastly, we apply it into a "preventative fine-tuning", forcing the model to learn a more robust refusal mechanism.
Across four LLMs, ASGuard effectively reduces the attack success rate of targeted jailbreaking while preserving general capabilities and minimizing over refusal, achieving a Pareto-optimal balance between safety and utility.
Our findings underscore how adversarial suffixes suppress the propagation of the refusal-mediating direction, based on mechanistic analysis.
Furthermore, our work showcases how a deep understanding of model internals can be leveraged to develop practical, efficient, and targeted methods for adjusting model behavior, charting a course for more reliable and interpretable AI safety.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper studies a safety flaw called tense jailbreaking, where rephrasing harmful prompts in the past tense bypasses model refusal. It proposes ASGUARD, a three-step mechanistic patch:
(1) build circuits with edge attribution patching and integrated gradients (EAP-IG) to locate attention heads active only in successful tense jailbreaks;
(2) learn channel-wise activation scalers to steer those heads toward safe refusals;
(3) run preventative fine-tuning while freezing scalers so the model internalizes safety, then remove them.
Experiments on Llama-3.1-8B-Instruct, Qwen-2.5-7B-Instruct, and Gemma-2-9B-it show large drops in attack success with little utility loss. Linear probes reveal that several heads act as tense detectors.

### Strengths
- Combining circuit discovery with scaling and using the scalers as a temporary scaffold during fine-tuning is novel. It goes beyond standard post-hoc steering by making the intervention teachable and later removable.
- Experiments cover three instruction-tuned models and compare against SFT, DPO, Circuit Breaker, and Representation Bending. The paper also introduces a robustness composite R-Score and reports raw metrics on OR-Bench-Toxic, OR-Bench-Hard, and MMLU. The Pareto plots are informative.
- The paper is clearly structured and easy to follow.

### Weaknesses
- In Table 2, all models show heavily degraded performance on OR-Bench-Hard. **It is possible that the reduction in Past-Tense ASR mainly results from an increased tendency to refuse responses**. This undermines the purpose of updating only the tense-vulnerable attention circuits. An analysis controlling for over-refusal rates would be very helpful.
- The study focuses **solely on past-tense reformulations**. It does not show transfer to other semantic variants that commonly appear in real jailbreaks, such as future or hypothetical moods, passive voice, or non-English prompts. The work would have greater impact if it could generalize this approach to more attack types.
- Circuit discovery with EAP-IG involves many integration steps, multiple refusal templates, and top-n pruning, which can be **computationally heavy** compared to baselines. The paper should include an analysis of EAP-IG’s cost and accuracy for readers less familiar with the method.
- **Some baselines may be under-tuned or incomplete**. DPO uses only one epoch, while Circuit Breaker and Representation Bending use fixed configurations per model, and result counts vary. Because safety fine-tuning is sensitive to hyperparameters, stronger or more systematically tuned baselines would improve fairness.
- The paper **lacks a reproducibility statement and open-source code**. Since ASGUARD involves complex interpretability tools and custom fine-tuning procedures, the results are difficult to verify.

### Questions
- The SFT(30/70) model appears competitive but sacrifices robustness. This might result from catastrophic forgetting due to full fine-tuning on small datasets. Would using LoRA mitigate this issue? Since ASGUARD fine-tunes only a small subset of parameters, this may explain why it maintains overall performance.
    
- While the paper reports MMLU scores to assess utility retention, MMLU does not involve tense understanding. If updates to tense-related circuits affect the model’s handling of tense more broadly, how could that degradation be measured?

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
3

### Summary
The authors provide a new way of mitigating specific attack points of an LLM. They identify key components to the past-tense attacks, learn suppression coefficients and then finetune with the suppressed model to increase its robustness.

### Strengths
- The approach seems to be effective against the past tense attack
- It is evidence for the circuit identification

### Weaknesses
- The technique is very attack-specific. It is unclear how effective the finetuning will be against any other attack. In that sense, the research is quite limited in my opinion. If it does not generalize, then it is a very expensive way of finetuning since one cannot just add data from other attacks but has to do all the different steps again. 
- Because of the previous point, the Pareto frontier data is a bit less relevant since the reduction in ASR is only against a single attack. I would like to ask the authors to add a range of other attacks to provide some data for the generalization of the technique
- If it does not work well, then I believe the paper is still somewhat relevant, but it should be rephrased towards not being a defense technique but rather evidence for the usefulness of mechanistic interpretability (with a few more findings on the mechanisms, it could be very interesting)
- The insight of this paper that attention heads only signal the tense and then the harmfulness evaluation will be later could use more experiments or arguments. To me, it feels a bit unsupported and is only drawn as a conclusion based on other work

### Questions
I would like to understand the finetuning step a bit more. 
As I understand it, the coefficients disable the "tense" heads to signal the tense. Then the model is finetuned and the coefficients are detached. To me, it seems like you are helping the model not to be susceptible to this attack during finetuning and enabling this attack path again after it. Hence, I am a bit confused about why this works. What is your justification for this? Can you provide some evidence that this is actually the case (e.g., only using a few coefficients).

### Soundness
3

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
4

### Summary
This paper introduces ASGuard, a mechanistically-informed defense against tense jailbreaking attacks in LLMs. The method: (1) uses circuit analysis to identify "tense vulnerable" attention heads causally linked to the attack, (2) trains channel-wise scaling vectors to suppress these heads, and (3) applies "preventative fine-tuning" to learn robust refusal mechanisms. Across 3 models and multiple benchmarks, ASGuard reduces tense jailbreaking ASR from 42-51% to 8-19% while maintaining better safety-utility balance than baselines like SFT, DPO, and Circuit Breakers.

### Strengths
1. Strong mechanistic foundation & insights: The circuit analysis approach (EAP-IG) to identify vulnerability-specific heads is principled, well-studied in literature, and provides interpretable insights. The linear-probe validation (§6.1) provides mechanistic confirmation that these heads encode tense information. Furthermore, the finding that alignment is localized to specific routing paths (many vulnerable heads disappear post-ASGuard, §6.2) is an important contribution to understanding LLM safety mechanisms.

2. Novel defense paradigm: The authors propose the "Preventative fine-tuning" (training with scaling vectors attached, then removing them), which is to my knowledge creative and shows clear improvements over naive scaling or standard fine-tuning alone.

3. Good empirical results (but unsure if there are still competive to the baseline that I’ll mention below): ASGuard achieves Pareto-optimal safety-utility trade-offs across all three models tested, substantially outperforming strong baselines, including well-known methods such as circuitbreaker.

### Weaknesses
1. Extremely Limited Scope
The method is designed for exactly one attack type (tense jailbreaking). There is no evidence it generalizes to: other semantic attacks (negation, translation, role-play, etc.), adversarial suffixes (GCG, AutoDAN), multi-turn jailbreaks.
I think this is a critical weakness as the baselines that are used by authors are evaluated on different types of attacks. It further shows somehow an unfair comparison. 

2. Circular Methodology: the detection dataset (§3.1) and evaluation dataset both use JBB-Behaviors prompts, creating a potential unreliable evaluation. Indeed, 100 prompts are used for circuit construction and the same prompts (with different reformulations) seem to be used for evaluation, exposing the risk of overfitting to these specific prompts.

3. Limited Baseline Comparisons

There are missing well-known defenses: Representation Engineering (Turner et al., 2023) is cited but not compared. Prompt-based defenses (system prompts, few-shot examples) and Input filtering or paraphrasing are not evaluated. Authors should also compare their method against RFA (Yu et al, 2024) "refusal feature adversarial training" and Gradient Cuff. 

Furthermore, there seem to be some issues with the RepBend comparison: RepBend is described as "recent SoTA" but it's surprising that  it doesn't consistently outperform simpler methods.  Some results show RepBend > ASGuard on specific metrics (e.g., Llama Toxic: 96.1 vs 96.4).

### Questions
1. :Can this framework defend against attacks other than tense jailbreaking? If not, how is it more than a specialized patch?

2. What are the computational costs relative to standard fine-tuning? Can you ablate on detection dataset size? 

3. If you train defenses for multiple attack types, do they interfere? Can one set of scaling vectors handle multiple attacks?

4.Why does preventative fine-tuning work? Can you provide formal analysis or at least deeper mechanistic explanation?

### Soundness
3

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
2

### Summary
The paper proposes ASGUARD, an activation-scaling method to mitigate targeted jailbreaks like tense-based attacks. It identifies vulnerable attention heads via circuit attribution, scales their activations to suppress unsafe behavior, and uses a short preventative fine-tuning phase to internalize the fix. Experiments across models show lower attack success rates and limited over-refusal while maintaining utility. The approach has demonstrated a practical and interpretable defense to the specific type of attack.

### Strengths
1. The paper targets a concrete jailbreak based on tense-based rephrasing and provides an interpretable fix.
2. The activation-scaling intervention is lightweight, requiring only head-level scaling vectors rather than any weight updates.
3. The preventative fine-tuning shows that it is feasible to transfer the scaling effect into model parameters, achieving Pareto-optimal results under this compounded method.
4. The experiments generalize across architectures and model sizes, showing that the internal behavior is replicable and the robustness holds consistently across models.

### Weaknesses
1. The attribution step that identifies “vulnerable heads” is largely heuristic. It would be beneficial to demonstrate that the selected heads are causally necessary for the jailbreak behavior. Additional analyses, such as head ablation or random-head controls, would make the causal interpretation more convincing.

2. The method is designed specifically for tense-based jailbreaks, and it is unclear whether the same workflow would generalize to other jailbreak categories. While the paper does not overclaim, the scope still appears limited.

### Questions
1. Have you verified whether removing or scaling different random sets of heads produces similar reductions in attack success?

3. Could ASGUARD be extended to other types of jailbreaks, and if so, how would “vulnerable heads” be defined in those cases?

3. How well does the preventative fine-tuning generalize to OOD data?

### Soundness
3

### Presentation
3

### Contribution
3
