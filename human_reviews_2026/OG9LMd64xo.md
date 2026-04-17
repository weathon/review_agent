# When Do LLMs Admit Their Mistakes? Understanding the Role of Model Belief in Retraction

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 4, 4, 4

## Abstract
Can large language models (LLMs) admit their mistakes when they should know better? In this work, we study when and why LLMs choose to retract, i.e., spontaneously and immediately acknowledge their errors. Using model-specific testbeds, we find that while LLMs are capable of retraction, they do so only rarely, even when they can recognize their mistakes when asked in a separate interaction. We identify a reliable predictor of retraction: the model’s \emph{momentary belief}, as measured by a probe on its internal states that is trained to predict correctness on external datasets unrelated to retraction. A model retracts only when it "believes" its answers to be incorrect \emph{during generation}; these beliefs frequently diverge from models' parametric knowledge as measured by factoid questions. Steering experiments further demonstrate that model belief causally drives retraction. In particular, when the model believes its answer to be incorrect, this not only encourages the model to attempt further verification, but also alters attention dynamics. Finally, we show that supervised fine-tuning improves retraction performance by helping the model learn more accurate internal belief.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper studies when and why large language models (LLMs) retract their mistakes—defined as spontaneous acknowledgments of incorrect answers. The authors construct model-specific continuation datasets from WIKIDATA and CELEBRITY benchmarks, where models generate potentially wrong answers and are then observed to see if they retract them without external prompting. Overall, the paper provides a principled, mechanistic understanding of why models sometimes refuse to retract even when they “know better.”

### Strengths
- Positions “retraction” as a measurable, meaningful behavioral metric for model reliability.

- Uses steering and patching methods to demonstrate directional control over behavior.

- Provides concrete evidence that attention value vectors, not just weights, mediate belief propagation.

- Evaluates multiple model families, increasing robustness.

- Links findings to SFT improvements, suggesting paths for aligning model introspection with truthfulness.

- Extensive appendices, code availability, and transparent methodology.

### Weaknesses
- Focuses on factual QA; results may differ for open-ended or reasoning-heavy tasks (e.g., math or multi-step reasoning).

- While linear probes capture useful signals, belief may conflate confidence, calibration, and factual recall.

- Although consistent, evaluation could benefit from human verification for robustness.

- Steering effects may differ across architectures or prompt styles.

- Retracting only ~25% of wrong answers initially limits downstream applicability, even if mechanisms are well-understood.

### Questions
- How robust are belief-based probes to changes in prompt style or model sampling temperature?

- Could non-linear or attention-based probes (e.g., small MLPs) capture belief signals more accurately?

- How does belief steering generalize to reasoning or creative tasks, where “wrong” answers are less well-defined?

- Does the causal influence of belief persist in RLHF-tuned models, which are already calibrated for truthfulness?

- Have you examined whether cross-model belief vectors (e.g., between Qwen and Llama) share a universal direction?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper investigates when and why LLMs spontaneously retract their own incorrect answers. For systematic evaluation, the authors propose continuation testbeds by prefilling the answers (e.g., Hillary Clinton [Model generation continues from here...]). The main observation based on this testbed is LLM can retract, but infrequently. By using probing and activation steering, it shows that the LLM’s internal belief reflects when the model internally “believes” it is wrong, it is more likely to retract. Besides, it shows that supervised fine-tuning improves retraction by aligning internal beliefs with factual correctness.

### Strengths
1. The paper introduces a new framework for studying LLM reliability as the testbeds for retraction behaviour. 

2. This paper proposed a warning for SFT. The connection between belief and retraction holds even after supervised fine-tuning, showing that improved belief calibration enhances factual alignment and transparency.

### Weaknesses
1. For table 2, I am curious if the phenomenon is consistent for larger LLM (e.g., meta-llama/Llama-3.1-70B-Instruct).

2. For the training procedure of the linear prob, I think the training dataset (UTQA dataset) is not a continuation dataset. The linear prob trained on this dataset aims to reflect factual correctness but not the retract behaviour, right? The interesting observation is that the linear prob is  highly predictive of whether the model will retract its answer. Any additional and deep explanation for this phenomenon will be very interesting.

3. For activation steering, it shows that changing internal belief alone is enough to change whether the model admits a mistake. I believe some ablation studies should be added to support this claim. For example, we should first split the testbed into two groups by the criterion as if the LLM can do retraction originally, and then study how negative/positive belief steering can change the retraction rate for these two different groups. 

4. For supervised fine-tuning, figure 4 shows that the probing of factual correctness can achieve significantly higher accuracies for middle layers (12-24). It is very interesting. Any explanation for this phenomenon?

5. There are several interesting observations in this paper, but most of them are quite facial. Besides, the relation between factual correctness and retraction behaviour is still not explained clearly. 

6. The novelty is quite limited. The methods used for analysis, such as linear probing, steering, and patching are not novel.

### Questions
Please check the weaknesses.

### Soundness
3

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
This paper studies spontaneous retraction—cases where an LLM, after giving an answer, immediately acknowledges it is wrong without being prompted. The authors build model-specific “continuation” testbeds from two knowledge tasks (WIKIDATA; CELEBRITY) and evaluate three 7–8B instruction-tuned models. They find 1. models can retract but do so rarely (low recall), even when separate verification questions show the model “knows” the answer is wrong; 2. a linear probe trained on external true/false datasets provides a belief signal that predicts retraction much better than correctness; 3. activation steering along belief directions causally modulates retraction (negative steering → frequent retraction; positive steering → almost none); and 4. SFT improves in-distribution retraction and aligns internal belief more closely with factual correctness.

### Strengths
- Clear, focused problem: Spontaneous retraction is practically relevant and distinct from multi-turn self-correction.

- Neat empirical finding: Linear probes of hidden states correlate strongly with retraction but less with ground-truth correctness, clarifying what such probes actually capture.

- Causality evidence: Activation steering gives credible leverage that belief directions are not merely correlates but drivers of retraction behavior.

- Mechanistic analysis: Patching experiments suggest attention value vectors (not only attention weights) are a primary pathway by which belief influences retraction.

- Training connection: Showing that SFT improves retraction by aligning internal belief bridges interpretability results with standard training practice.

### Weaknesses
- **External validity / scope** All core experiments use three small instruction models and two knowledge-centric datasets. It’s unclear if belief→retraction generalizes to larger or reasoning-style models, to non-factoid tasks, or to tool-augmented settings.

- **Evaluation dependence on an LLM judge** Retraction is judged automatically (Llama-3.3-70B). Although convenient, relying on a single judge risks systematic bias and false positives/negatives (e.g., hedged text vs true retraction). Human audits or multi-judge agreement would strengthen claims.

- **Prompt / decoding sensitivity** The phenomenon may hinge on stopping behavior and continuation prompts (the “is-appended” setting materially changes results). More thorough robustness to decoding parameters, prompt templates, and stop-criteria would help.

- **Probe training / generalization** Belief probes trained on UTQA then applied OOD to continuation data work well for retraction but less for correctness. This invites concerns about dataset-specific artifacts (style, length, lexical cues). More ablations on token position, layer selection, and context length would clarify.

- **Steering hyperparameters** Steering strength/layer ranges are manually searched. The narrative would benefit from systematic sweeps with confidence intervals and controls for side effects (fluency, length, off-topic drift).

- **Statistical reporting** Many results are presented as point metrics. Please include CIs, random-seed variance, and significance tests for retraction precision/recall and AUROC curves.

### Questions
- How consistent are retraction labels across multiple judges or human raters? Any inter-annotator stats?

- How does temperature, top-p, and max-tokens affect retraction rates independently of belief steering?

- Can you quantify textual markers of retraction (e.g., “I was wrong,” “Correction: …”) and report precision/recall for each subtype?

- Do belief directions found on one model family transfer to another? Any cross-model steering experiments?

- Beyond attention values, did you test MLP value patching or logit lens analyses to localize belief-to-retraction pathways?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a reliable predictor of retraction: the model’s momentary belief, as measured by a probe on its internal states that is trained to predict correctness on external datasets unrelated to retraction. Supervised fine-tuning shows to improve retraction performance by helping a model learn more accurate internal belief.

### Strengths
1. Regarding originality, momentary belief appears to be an interesting concept. See questions/weaknesses below so I can better judge the novelty.

2. I like the supervised fine-tuning part, which connects the idea of LLMs' internal belief.

### Weaknesses
1. Soundness needs to be improved by running experiments on larger models. The largest model studied is 8B, so I am not sure how the findings generalize to larger models. It is necessary to discuss the similarity and difference between large and smaller models in terms of your momentary belief concept.

2. I am not sure the utility of momentary belief. What are the final metrics of measuring the utility of momentary belief? The clarity needs to be improved.

3. I am not sure the efficiency of computing momentary belief. If your concept of momentary belief is indeed helpful, it is necessary to show the comparison on performance, computation costs, and inference latency across a wide range of baselines.

4. Writing needs to be improved in order to reach a publishable state. For example, I am not sure how your definition of "retraction" differs from "backtracking" as the question asked below. This is my concern, because I use it to judge the novelty of this paper.

### Questions
1. In your abstract, can you explain "Steering experiments further demonstrate that model belief causally drives retraction"? You used the word "causally" throughout your paper, but I still find a hard time understanding it.

2. Since you define retraction in Section 3.1, is it the same as "backtracking" that is more commonly used in other literature such as [1]?

[1] "To Backtrack or Not to Backtrack: When Sequential Search Limits Model Reasoning"

### Soundness
2

### Presentation
2

### Contribution
2
