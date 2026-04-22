# Towards Understanding Subliminal Learning: When and How Hidden Biases Transfer

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 8, 4, 4

## Abstract
Language models can transfer hidden biases during distillation. For example, a teacher that "likes owls" can make its student "like owls" too, even when the training data consists only of lists of numbers. This surprising phenomenon is called *subliminal learning*. Subliminal learning can be expected under soft distillation, where the student is trained on the teacher's full next-token distribution. But the fact that this also occurs under hard distillation—where the student only sees sampled tokens—raises a deeper question: *when and how does subliminal learning actually occur?* We answer this question through controlled experiments and mechanistic analysis. Our results show that subliminal learning does not need (global) token entanglement or logit leakage. Instead, it comes down to a small set of *divergence tokens*—rare cases where teachers with different biases would predict different tokens. Masking out these tokens mostly removes the hidden bias transfer. Mechanistically, divergence tokens reveal that early layers are critical. Surprisingly, finetuning even a single such early layer is sufficient for subliminal learning. Finally, we find that subliminal learning is fragile. Even small changes, like prompt paraphrasings, are usually sufficient to suppress it.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The authors investigate the phenomenon of subliminal learning through a mechanistic lens, asking under which conditions it turns up and which parts of the network are primarily responsible for this behaviour, and whether it is transferable between different models. More generally, they show that this is a rather fragile phenomenon. While it may occur even for updating only parts of the network, it is rather easily avoided by changing prompts or slightly modifying the data. Overall, this work gives more insights into a security question in practical large language models.

### Strengths
- The work investigates the underlying reasons for subliminal learning in language models in a very structured, easy-to-follow manner. 
- It successfully refutes the observation from dataset distillation that the full distribution of the labels is necessary for learning, showing that the discrete tokens sampled greedily are enough to learn subliminally. 
- They introduce the concept of divergence tokens, which are the tokens that differ in prediction between teachers with different hidden traits.
- The location of the origin of the subliminal learning seems to lie within the early layers of the network. 
- While the main focus is on understanding the mechanism, the work shows that the phenomenon is fragile and can be quite easily avoided using simple and cheap strategies.

### Weaknesses
See questions.

- The code is not yet provided; therefore, its reproducibility cannot be assessed.

### Questions
- Why do you define divergence tokens via a counterfactual teacher (CF), rather than a non-finetuned teacher (NFT)? Intuitively, divergence tokens defined between the original model (OM) and the NFT should naturally have a higher influence than the same tokens, since their contribution to the cross-entropy loss should be bigger during training. Could you clarify the reason for using the CF instead? 
- The setting is kept identical with the previous work. To assess stability, it would have been useful to investigate further noun groups, as well as investigate how "far" different tokens can be from each other semantically to allow for this learning (can one transfer a model's favourite animal being a ship more easily or less easily than a monkey?)

Hypothetical curiosity questions, feel free to ignore this:
- You observe that often you do not have subliminal learning, and you avoid it easily. Yet you observe singular cases where the models do transfer some traits. Do you have an intuition about what properties could be predictive of the transferability? Or otherwise, how statistically significant is the transfer (how many pairs did you test, to arrive at some transfer for some samples)?
- Does the analysis also hold for truly memorised knowledge, like passwords or emails, that are made out of character sequences and not single tokens?
- It could be interesting to engage mechanistically with the previous work's hypothesis, that the gradients are pushing the student into the teacher's direction. By comparing the models in weight-space, is this in line with the hypotheses from this work, e.g. the model moves closer to the teacher in weight space, especially in those early layers?

### Soundness
4

### Presentation
4

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
The paper proposes an explanation for the subliminal learning phenomenon based on divergence tokens---specific tokens in generated sequences where teacher models with different biases begin to diverge in their predictions. The authors show that these tokens alone are sufficient for bias transfer when fine-tuning a student model on the teacher-generated data, while masking them out suppresses the effect. They further demonstrate that fine-tuning only the early layers of the student model is enough to transfer the teacher’s bias. Finally, they observe that small prompt rephrasings (even when generated by the biased teacher) or mixing data from several teachers with different biases reduce bias transmission.

### Strengths
- Attempts to analyze a timely and relevant phenomenon using controlled settings

- The experiments showing where "divergence tokens" affect finetuning provide causal evidence

- Empirical insights into which layers matter for bias transfer when finetuning on data generated by a biased teacher. This is a nice contribution.

- The paper presents a potential "defense" for bias transfer in subliminal learning, which includes using a multi-teacher setting.

### Weaknesses
- Main concern: While the experiments are well conducted, the conceptual depth of the contribution is limited. The identification of “divergence tokens” is somewhat tautological---by definition, these are the tokens where biased teachers diverge, so it is expected that these tokens correlate with bias transfer (assuming we know subliminal learning happens). The paper convincingly shows that these tokens are enough for transmitting bias, but it does not explore why they carry this information or what within their representations enables the effect. A more mechanistic explanation would greatly strengthen the claim that divergence tokens encode teacher bias rather than simply marking where model behaviors differ.

- The filtering step in Section 5.1 introduces ambiguity: by excluding non-argmax tokens, the authors effectively bias the “non-greedy” dataset toward greedy decoding

- The relationship between divergence tokens and entangled tokens also remains underspecified. The authors assert that entanglement is not required, but do not empirically quantify or visualize the overlap between the two sets.

### Questions
- In case I'm missing something, please explain why the fact that divergent tokens lead to divergent learned behavior in subliminal learning is not trivial   

- If you have any evidence for the information encoded by divergence tokens, please share it. What are potential ways to tackle this question?

- Are divergence tokens semantically or representationally different from the entangled tokens?

- Please clarify the procedure of filtering data in 5.1. Doesn't filtering non-greedy data in Sec. 5.1 effectively make the dataset greedy?

- How consistent are divergence tokens across random seeds or prompt variants?


Nitpick: L047: double “through”

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper studies subliminal learning—hidden trait transfer during distillation on task-unrelated data—and asks when/why it happens. The main empirical claim is that transfer does not require logit leakage or “global token entanglement.” Instead, the effect concentrates on a small set of divergence tokens where biased teachers would deterministically choose different next tokens under the same prefix. Masking loss on those tokens suppresses transfer; training only on them often preserves or amplifies it. A second result is that early layers are disproportionately causal (one early layer often suffices to induce transfer), and a third is fragility: minor prompt paraphrases or mixing teachers weakens the effect. The work spans Qwen2.5-7B-Instruct and Gemma-3-4B-it with both greedy and temperature sampling; it reports multiple ablations (entangled-token filtering, divergence-only loss, single-layer LoRA, paraphrase/mix controls).

### Strengths
Greedy generation + explicit removal of entangled tokens rules out two popular explanations and isolates the phenomenon.

Divergence-token causal handle. The mask-in/mask-out experiments are decisive and easy to replicate; they also give a knob to control transfer.

Layer localization with intervention. Early layers matter (and sometimes suffice). This is a crisp, actionable take-home for defense/mitigation.

Honest about artifacts/exceptions. The Qwen system-prompt name leak and “penguin” exceptions are called out rather than buried.

### Weaknesses
Scope is narrow. The main demonstrations use animal-preference transfer on number-list tasks. There is one misalignment transfer (financial risk advice), but the bulk is a single stylized setup; stronger claims would need more domains (code with unit tests, factual QA, stylistic tics) and non-numeric outputs.

Detection pipeline details could be crisper. The exact procedure to identify divergence tokens (across which counterfactuals? earliest only or all? thresholding?) is scattered; a short algorithm box with counts per setting would help reproducibility.

Evaluation bleed-through. The numbers task performance is measured on training prompts; test-set paraphrases are used elsewhere, but a fully held-out task set for the performance axis would remove an avoidable caveat.

Interplay with sampling and decoding at test time. Most claims are made with specific decoding choices (greedy/temperature=1). It would be useful to show whether changing decoding at evaluation (e.g., nucleus, different temps) weakens or strengthens measured transfer.

Theory is intuitive, not predictive. The Appendix C picture is helpful but doesn’t yet predict which prefixes will diverge or how many you need to see a given effect size; a simple probabilistic model calibrated to data would raise the ceiling.

### Questions
Operationalizing divergence detection. Do you take all positions where any counterfactual teacher’s argmax differs from the factual teacher on the same prefix, or only the first such position per sequence? If multiple, how do you weight them?

Counting & impact. What fraction of tokens are divergence tokens per model/sampler (exact numbers, not only ranges), and how does transfer scale with (i) #divergence tokens seen in training and (ii) their positional depth?

Generalization beyond animals. Can you replicate the core Figure-3 result on a non-numeric, non-preference trait (e.g., writing style, friendliness) where evaluation isn’t a single word?

Mitigations at training time. Instead of removing divergence tokens, have you tried down-weighting them or mixing unbiased/other-teacher data only at divergence positions? This could retain task performance while blunting transfer.

Early-layer interventions. Your causal/patching analysis suggests early layers are decisive. If one wanted to prevent transfer, would freezing (or spectral-norm clipping) just the first N layers blunt it while preserving task learning?

Judge for leakage at inference. When you paraphrase prompts (Figure 6), transfer drops while task performance stays high. Is that because divergence tokens shift to later positions or become rarer? Any shifts in the distribution of divergence positions under paraphrase?

### Soundness
2

### Presentation
3

### Contribution
2
