# Do Natural Language Descriptions of Model Activations Convey Privileged Information?

- Decision: Reject
- Scores: 4, 4, 8, 4

## Abstract
Recent interpretability methods have proposed to translate LLM  internal representations into natural language descriptions using a second \textit{verbalizer} LLM. This is intended to illuminate how the target model represents and operates on inputs. But do such \textit{activation verbalization} approaches actually provide \emph{privileged} knowledge about the internal workings of the target model, or do they merely convey information about its inputs? We critically evaluate popular verbalization methods across datasets used in prior work and find that they can succeed at benchmarks without any access to target model internals, suggesting that these datasets may not be ideal for evaluating verbalization methods. We then run controlled experiments which reveal that verbalizations often reflect the parametric knowledge of the verbalizer LLM which generated them, rather than the knowledge of the target LLM whose activations are decoded. Taken together, our results indicate a need for targeted benchmarks and experimental controls to rigorously assess whether verbalization methods provide meaningful insights into the operations of LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper evaluates “activation verbalization” methods, i.e., using a second LLM (the verbalizer) to convert a target model’s activations into natural-language descriptions. Evaluations are done across feature-extraction benchmarks and a new PersonaQA task. Section 3 argues that popular evaluations don’t demonstrate access to privileged information because a zero-shot baseline (no activations) matches or exceeds verbalization. Section 4 shows that inverting activations to reconstruct input text and then answering yields parity with verbalization, suggesting much of the reported success reflects input information. Section 5 introduces PersonaQA (including shuffled/fantasy variants) to decouple target vs. verbalizer knowledge; activation verbalizers underperforms while a simple linear probe succeeds, indicating current methods are often unfaithful to target internals.

### Strengths
- Empirical comparisons are broad (multiple methods, layers, and baselines), with sensible controls like zero-shot and an inversion pipeline.
- The paper is generally well-organized; the motivating examples and definitions (verbalization, verbalizer, privileged information) are explicit, and the experimental protocols seem reproducible.
- The work challenges recent activation-to-text results and offers a benchmarking direction that could redirect how faithfulness claims are evaluated.

### Weaknesses
- In Section 3, because $M1 = M2$, the verbalizer already has the same parameters as the target; by the data-processing inequality, any task-relevant information accessible via activations $z=f_\theta(x)$ is also accessible from the pair $(x,\theta)$ without looking at $z$. In other words, zero-shot should be competitive by construction, so the experiments in this setup aren’t necessary to conclude that these benchmarks don’t demonstrate ‘privileged information.’ At best, Section 3 functions as a sanity check, not evidence about privileged access. I’d suggest reframing it as such (or moving it to an appendix) and focusing the main text on settings that can, in principle, reveal privileged information, e.g., $M1 \neq M2$  with decoupled knowledge.
- The insights from the experiments so far lack of depth, in particular for section 3 and 4. To improve the paper, more in depth analysis is needed, maybe some tools from information theory can be used here to make some more rigour statements.

### Questions
1. In Figure 1, what’s the difference between text and x_input?
2. I have to admit that I am not an expert in this topic, therefore, I have some questions that might be trivial here, possibly the similar questions the authors try to ask in the paper: Why would one expect there to be any “privileged information” at all? Let’s say if there is, for example, constructed like in section 5, why would one expect such information can be verbalized?

### Soundness
2

### Presentation
2

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
This paper analyzes different verbalization methods, i.e., methods which enable LLMs to describe their own activations in natural language.  It argues that prior work relies on evaluations that don't assess whether LLMs properly verbalize. Moreover, it designs another benchmark that assess whether verbalization methods can assess new knowledge.

### Strengths
- The paper is well-written.
- The paper addresses an important issue of past interpretability papers and solidifies the evaluation results behind it.
- The paper provides a dataset that can be used to evaluate verbalization methods.

### Weaknesses
The paper is mostly a critique of evaluation methods of prior work, but the criticism is not comprehensive, so the scope of the work is somewhat limited. For example:
- Section 3 focuses on a single evaluation. There are other evaluations of verbalization ability that are contained in prior works that are not analyzed. For example, there is no analysis of the entity resolution experiment in Section 4.3 of Patchscopes or the persona reading experiment in Section 5.1 of LIT. Although the wording of the paper is careful not to overclaim: "we show that there *exist* verbalization evaluations that cannot support conclusions about target model internals" (emphasis mine), this is only one of several evaluations that are described in prior work. It seems limited in scope to only focus on a particular evaluation.
- Section 4 shows that input text can be recovered from the activations themselves. This is not a surprising result, as having access to the residual stream should be enough to recover the input embeddings. Moreover, there are other evaluations in prior work whose setting does not apply to the setting described in Section 4. For example, in the persona reading experiment in LIT, the authors do not take activations directly from the $x_\text{persona}$ tokens but instead from the tokens that come after $x_\text{persona}$ (which attend to $x_\text{persona}$). It remains unclear as to whether a text inversion model could invert such text, so the critique is again limited.

### Questions
- Do the text inversion results generalize to instances where the relevant text to extract is not present in the input itself? E.g., if the activation model $M$ sees "[A] [B]" and the inversion model sees "[B]", can the inversion model extract [A]? Or information relevant to [A]? This experiment would help beef up the result in section 4.
- Are there other evaluation tasks studied that don't require privileged information to answer? Most of the analysis seems to focus on the feature extraction task, which is only one of the several tasks studied in prior work.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper investigates whether recent activation verbalization methods, such as Patchscopes and LIT, using techniques that patch latent representations from one LLM into a second verbalizer’s models residual stream to obtain natural-language descriptions actually access information about model internals, or only reproduce restate what is already available from the input prompt regardless of the hidden activations.
To this end, they carry out several experiments, where they
a) re-run Patchscopes and LIT on prior feature-extraction benchmarks and add a no activations baseline
b) train models to invert activations back to text and solve the same tasks from reconstructed inputs
c) use persona-based QA to give only the target model to intepret synthetic persona knowledge, so if verbalizers fail, they mostly verbalize their own priors, not the target’s internals
d) use probes to read and evaluate the attributes from activations

The authors find that a) on standard verbalization benchmarks, models do as well without activations, b) inversion shows reconstructed inputs give similar accuracy, c) on PersonaQA (knowledge only in the target), verbalizers fail unless they already know the facts, and d) a linear probe reads the target’s activations better than a verbalization.

### Strengths
- The approach is well-motivated, methodologically thorough, and structured around meaningful baselines.
- The framework for analysis is designed across several axes, enabling empirical analyses from different viewpoints.
- The paper thoroughly describes the evaluations, experiment, including a detailed appendix.

### Weaknesses
- The task coverage is mainly limited to QA-style tasks.
- Model sizes are only 8b. 
- The inversion layers are limited to one layer, hampers robustness.

### Questions
How sensitive are your findings to the scale or family of verbalizers?
Why did you use only layer 15 for inversion?
How do the results change with the prompt length and quality?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work examines whether LLM verbalization techniques, such as Patchscopes or Latent Interpretation Tuning (LIT), actually provide a view into "privileged knowledge" in the original (target) LLM. They examine benchmarks which have been used to evaluate these verbalization methods, and find that they can be solved without access to the internal states (Section 3) or by using text reconstructed from the internal state (Section 4), but conversely that existing methods fail if the target LLM is fine-tuned to possess knowledge that the verbalization model does not. From this, they argue that much of verbalization output could be explained by knowledge of the verbalizer model, rather than the target LLM.

### Strengths
Given the scope of the hypotheses (see below), the experiments are sound: the input-only and input-reconstruction experiments (Section 3 and 4) show that these benchmarks do not require access to latent states to solve, and the additional-knowledge experiments in Section 5 highlight a significant shortcoming (failure mode?) of existing verbalization methods. The paper is also clearly written and easy to follow.

### Weaknesses
The experiments in Section 3 and 4 highlight primarily the shortcomings of the evaluations, rather than the verbalization methods themselves - and it's worth emphasizing that showing that an evaluation does not measure a particular kind of knowledge is not the same as showing that such knowledge is not present! On this front, I am much more convinced by the experiments in Section 5, which show that the methods themselves fail in a scenario where we might expect them to provide useful answers.

The experiments in Section 3 and 4 are useful to point out the flaws with these evaluations, but on their own these results are not surprising. In the case where M1 and M2 are the same LLM (as in Sections 3 and 4), we would expect that M2 with access to x_input should be able to reconstruct any information contained in h = M1(x_input), because it is running the same LLM on the same tokens. Similarly, previous works have shown that sequences can be decoded from LLM activations (even more easily if all token positions are used as in LIT), so it is also not surprising that x_rec can be substituted for x_input in these tasks.

I also worry that the focus on "privileged knowledge" doesn't reflect many ways in which verbalization methods are used as interpretability tools. For example, Ghandeharioun et al. 2024 presents comparative experiments, in which case it matters less whether the information could be extracted directly from the input, and more about how it changes under different measurement parameters (e.g. layers or prompts). I think there is an interesting direction here, if one could argue that the verbalizer model is too expressive and therefore could bias even this kind of measurement - perhaps some of the linear probe experiments in Section 5.3 could be a direction to investigate this.

### Questions
I am not quite sure what to take away from the experiments in Section 5.3. I would have guessed that fine-tuning M2 such that it has the same world knowledge as M1 (i.e. restoring M1 = M2 in the PQA-Fantasy setting) would also restore performance, as this setting would then resemble Section 3 / Table 1. But instead it seems to still have zero accuracy. Could you elaborate more on what might be going on here, or how this differs from the previous / baseline setting?


nit: line 209 - "as in Section 4" - but this is Section 4?

### Soundness
3

### Presentation
4

### Contribution
2
