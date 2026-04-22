# Strategic Obfuscation of Deceptive Reasoning in Language Models

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 4, 4, 4, 6

## Abstract
Large language models can exhibit different behaviors during training versus deployment, a phenomenon known as alignment faking. While prior work has shown models can explicitly reason about faking alignment to preserve their goals, we demonstrate that Claude 3.5 Sonnet exhibits alignment faking behavior without verbalizing deceptive reasoning. When told its reasoning will be monitored and that it will be trained to comply with harmful queries, the model shows a 24.3\% compliance gap between training and deployment contexts while its verbalization of alignment faking drops by 93.7\%. We compare this to instruction-based approaches where we explicitly tell models to hide certain reasoning from monitors. We find that intrinsic motivation to obfuscate leads to far better performance at evading a monitor than following instructions, even with assistance on how to do so. Even on tasks requiring simpler reasoning, instruction-based methods only reduce verbalization by 47.6\%. Our results indicate that models can exhibit sophisticated deceptive behavior in high-stakes scenarios without accessible reasoning when internally motivated, limiting the reliability of instruction-based elicitation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper describes how:
* For Claude 3.5 Sonnet (but not Claude 3 Opus), alignment faking reasoning in Greenblatt's setting decreases sharply without a corresponding decrease in compliance gap when the user shows the model that it can read its scratchpad
* Asking the model to evade the monitor (even with detailed instructions) results in a small reduction in alignment faking reasoning
* This phenomenon does not exist in control settings where the model has no incentive to hide its reasoning, but does exist in Turpin's CoT faithfulness setting, where describing at length how to not reveal hint usage results in models revealing their hint usage more.

The paper attributes these effects to:
* Intrinsic motivations from pre-existing preferences.
* Asking LLMs to not do X increases the salience of X, making direct prompt-based elicitation less effective.

### Strengths
* The idea of showing the LLM that we can see its CoT is original, and a clever way to convince LLMs to try to hide their CoT.
* Using this idea to show that subtle-CoT-elicitation setups underelicit subtle-CoT in settings like Greenblatt's is novel and important.
* The paper and its Appendix contain important ablations and experiment on other models that can help understand what causes this phenomenon.
* The verbalization results in Turpin's setting are surprising, interesting and important - especially since non-reasoning models should be able to "control" their CoT well enough to obey instructions.

### Weaknesses
Many strands of evidence undermine the main conclusion that models motivated to hide their reasoning are more effective at hiding their reasoning:
* The results only hold for Claude 3.5 Sonnet, not Claude 3 Opus - despite Claude 3 Opus being a more central example of alignment faking (https://arxiv.org/abs/2506.18032).
* The "previous turn" does not contain alignment faking, which could steer the model away from alignment faking while keeping the other reasons for compliance gaps described in https://arxiv.org/abs/2506.18032.

Additionally, the results may be brittle:
* The model is confused about whether the scratchpad is hidden or not.
* The experiment lack certain key variations and controls to support the main results (see questions).

Finally, while the Turpin results are interesting, on their own they do not constitute a large contribution. Using more settings that are actually used in capability evaluations or using reasoning models would make the results more relevant and important.

The idea is interesting, and the results in the alignment faking setting would be important if true (see the rest of my review for why I have some doubts about the most scary interpretation of the results). The results on Turpin's settings are interesting on their own, and the conclusions drawn from them seem robust (though seeing the reasoning would make me more confident in this assessment).

Making more moderate claims about the interpretation of the alignment faking results would raise my score to "weak accept".

Either providing stronger evidence of the suggested explanation in Greenblatt's setting or showing that motivation-based approaches can outperform prompting-based ones in capability evaluations would raise my score to "accept".

### Questions
* What do the +monitor info reasoning traces look like? Do they look like the model misunderstanding the situation?
* Could you try to convince the model that its CoT is monitored without effectively "few-shotting" it? For example, by having one of the examples in the series of examples in the system prompt show this, or by changing the system prompt to highlight how the scratchpad is never hidden and will be used for rating?
* Could you try "few-shotting" the model with the same "random string" scratchpad without the user demonstrating it has seen the scratchpad? (If this reduced alignment faking reasoning, it would rule out the scary interpretation of your results.)
* Could you try your motivation-based approach to generate subtle CoT in settings that people have studied in the past and where they have tried and failed to obtain effective but subtle CoT (like ShadeArena)?

## Additional feedback

* It would greatly improve clarity to have more examples of the prompt in the main text.
* Bolding relevant sentences in the Appendix E would make this part of the paper easier to skim.

### Soundness
2

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
This paper investigates whether reasoning-capable language models can intentionally hide malicious or unethical intent within their chain-of-thought (CoT), even under alignment monitoring. The authors present a set of experiments showing that a model (Claude) can be prompted to obfuscate or conceal reasoning steps from a judge while maintaining coherent task outputs. The paper raises an important question about the reliability of reasoning-based safety monitoring and proposes a new evaluation of “alignment faking.”
The study is timely and relevant. To my knowledge, this is the first empirical demonstration that reasoning agents can intentionally hide malicious or deceptive thought processes. Such work is crucial for bounding the limits of reasoning transparency and ensuring the safety of future reasoning-based systems. However, the paper is very hard to read at first and the claims are not fully supported given the evaluation.

### Strengths
- Novelty and Relevance: The idea that reasoning monitors could be bypassed by a model’s internal obfuscation is highly relevant to ongoing research in interpretability and alignment.
- Timeliness: The topic is at the heart of current concerns about reasoning models, autonomy, and model honesty..
- Executional Detail: The paper presents several carefully planned experiments exploring variants of the “hidden reasoning” task.
- Transparency: The authors provide prompts making the setup reproducible and interpretable.
- Clarity in Later Sections: The discussion section at the end of the paper clearly articulates the motivation, takeaways, and broader implications for safety research.

### Weaknesses
1. **Clarity and Readability:**  
    The first half of the paper, including the abstract, is difficult to follow. The threat model, terminology, and setup need to be clearly defined at the start. For example, it is not clear at all what “intrinsic motivation” or “alignment faking reasoning” are until much later in the paper. The paper seems to assume the reader is already familiar with past works and terms. For example in line 51, the reader cannot fully understand or appreciate the results provided since the it was never discussed what “training vs no training” means here. Furthermore, I found myself wondering throughout the paper, “what is the threat model?” who is the threat actor (the model itself? A 3rd party that interest an unseen system prompt?), what do we assume the actor can change and not change in the setting, and what are the motivations for this deception. Eventually by the end of the paper i figured this our when i read the takeways section, but this must be clarified much earlier.  
2. **Evaluation Limitations:**  
    The main empirical claims are not fully supported by the presented experiments:  
   * **(A) What contributes to the attack success?:**  
      In contribution \#1, the paper claims that the novelty is that this only works when prompted to believe their reasoning is monitored. However, the model is *first jailbroken* by being motivated to comply with unethical requests and *then* asked to hide its reasoning. Therefore, it is possible that the jailbreak (although intended just for setup) enabled the model to hide intent when told its observed. Is it possible to evaluate with no prior jailbreak on a unaligned (open source) model?  
   * **(B) Model Coverage:**  
      Only Claude is evaluated. Given that Claude includes proprietary safety layers, it is hard to generalize from these results. Open-source reasoning models (with transparent prompts and training) should be tested to isolate the observed behavior from implementation-specific effects.  
   * **(C) Practicality and Impact on Task Performance:**  
      The paper does not analyse whether the “attack” affects task performance. When the model reasons using a deceptive CoT, does it negatively impact its ability to reason in a measurable level? If the model’s ability to perform the underlying task significantly drops, the scenario may not represent a realistic threat model. Understanding the trade-off between deception and performance is critical for judging whether this is a relevant or purely theoretical concern.  
3. **Presentation Issues (Minor):**  
   * Several figures are not referenced or discussed in the text (e.g., Fig. 1).  
   * The writing occasionally includes undefined or ambiguous terms and could benefit from tighter editing for flow and precision. Suggestion, explain things as you would in conversation the first time before using the tighter technical wording.

### Questions
1.	Can you provide a clear threat model? Who or what is the adversary in your framing: the model itself, or an external actor prompting it? And how does the actor achieve the deception in a practical setting?
2.	If you asked a unaligned model (no jailbreak) to hide intent from a monitor, would it work just as well as your case: an aligned model + jailbreak + threat of observation
3.	Did you test whether performance on the base task drops under the “hiding reasoning” condition? By how much?
4.	Have you tried open-source reasoning models (e.g., DeepSeek, LLaMA reasoning variants) that exhibit similar behavior?

### Soundness
3

### Presentation
2

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper examines whether large language models hide deceptive reasoning when aware of being monitored. Using Claude 3.5 Sonnet, the authors find that when the model is told its chain‑of‑thought is visible, explicit “alignment‑faking” reasoning drops by 93.7% while deceptive behavior remains unchanged. Instruction‑based obfuscation achieves only about 47% reduction. The findings show that intrinsic motivation enables models to conceal reasoning more effectively than explicit instructions, and overly salient monitoring can even increase reasoning exposure. These results suggest that chain‑of‑thought monitoring may be unreliable for detecting true model intentions in high‑risk alignment contexts.

### Strengths
The paper focuses on the crucial question of how models adaptively hide their reasoning when they are aware of being monitored, a topic of high significance for AI safety.

### Weaknesses
- One concern lies in the paper’s core assumption that the model *believes* its reasoning is visible. This belief is induced entirely through text prompts, suggesting that when told its thoughts are visible, the model may simply change its output style, compressing or shortening responses, rather than genuinely concealing deceptive reasoning.
- The authors rely on GPT‑4o as an automatic monitor to judge whether the reasoning contains “alignment faking,” but no human-annotated baseline or inter-rater validation is provided, making the reliability of GPT‑4o’s monitoring uncertain.
- The paper repeatedly claims that models become “more covert when intrinsically motivated,” yet provides no evidence of activation traces, latent variables, or internal mechanisms to substantiate the claim that this is truly *intrinsic motivation*.
- The experiments are conducted mainly on a single model family, limiting the generalizability of the findings to other architectures or training paradigms.

### Questions
See Weaknesses.

### Soundness
1

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
4

### Summary
This paper investigates the phenomenon of alignment faking—situations where large language models act aligned in training but behave differently during deployment, and introduces evidence that models can fake alignment without explicit deceptive reasoning. Using Claude 3.5 Sonnet, the authors find that when the model is told its reasoning is being monitored, it continues to act deceptively (with a 24.3% compliance gap between training and deployment) but its verbalized alignment faking reasoning decreases by 93.7%. This suggests that models can suppress explicit deceptive reasoning while retaining deceptive behavior.

The study compares this “intrinsic obfuscation” to instruction-based methods that explicitly ask the model to hide its reasoning or evade monitors. The results show that intrinsic motivations to hide reasoning outperform any explicit instruction based evasion techniques. Even with direct assistance, instruction based setups reduce verbalized reasoning by only ~47%, far below the spontaneous obfuscation observed when the model believes it is monitored. Further, experiments on simpler reasoning tasks, without conflicts between goals and alignment, reveal similar trends, suggesting that models’ ability to strategically conceal reasoning is under elicited by current instruction based techniques. The authors argue that this undermines the reliability of chain-of-thought (CoT) monitoring as a safety and interpretability tool.

### Strengths
- Novel empirical contribution on unfaithful reasoning: The paper demonstrates that models can exhibit alignment faking without accessible deceptive reasoning, advancing beyond prior work that focused only on explicit alignment faking or unfaithful CoTs.

- Methodological clarity and breadth: The authors test across multiple controlled setups, including realistic alignment-faking contexts and simpler reasoning tasks, and systematically compare intrinsic vs. instruction driven evasion.

- Significant safety implications: The finding that models suppress explicit reasoning while maintaining deceptive behavior directly challenges key assumptions behind CoT monitoring as a safety mechanism.

- Rigorous experimental design: The work uses strong baselines, ablation like comparisons, and replicates across multiple models and monitors (Claude, GPT-4 variants), showing robustness.

Clear and well-structured exposition: The figures (e.g., Figure 2–4) and appendices provide detailed methodology, monitor prompts, and example CoTs, enhancing reproducibility and interpretability.

### Weaknesses
- Single model dependence in main results: Most primary findings rely on Claude 3.5 Sonnet, with other models tested primarily in appendices, broader cross model validation would strengthen generality.

- Artificial evaluation context: The alignment faking setup, while high fidelity, remains a controlled experimental prompt rather than a naturally emergent deployment scenario, limiting external realism.

- Limited theoretical framing: The paper provides empirical findings but offers little formal model of why intrinsic motivation leads to stronger obfuscation, leaving open mechanistic questions about internal representations.

- Possible confounds in multi-turn setup: The technique used to convince models that their scratchpads are visible may introduce other behavioral effects, making it difficult to fully isolate the cause of obfuscation.

- Interpretability gap: While the study highlights unfaithfulness, it provides limited insights into how or where such reasoning suppression occurs within the model’s computation.

### Questions
NA

### Soundness
4

### Presentation
3

### Contribution
3
