# Propaganda AI: An Analysis of Semantic Divergence in Large Language Models

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 4, 4, 8

## Abstract
Large language models (LLMs) can exhibit *concept-conditioned semantic divergence*: common high-level cues (e.g., ideologies, public figures) elicit unusually uniform, stance-like responses that evade token-trigger audits. This behavior falls in a blind spot of current safety evaluations, yet carries major societal stakes, as such concept cues can steer content exposure at scale. We formalize this phenomenon and present **RAVEN** (**R**esponse **A**nomaly **V**igilance), a black-box audit that flags cases where a model is simultaneously highly certain and atypical among peers by coupling *semantic entropy* over paraphrastic samples with *cross-model disagreement*. In a controlled LoRA fine-tuning study, we implant a concept-conditioned stance using a small biased corpus, demonstrating feasibility without rare token triggers. Auditing five LLM families across twelve sensitive topics (360 prompts per model) and clustering via bidirectional entailment, RAVEN surfaces recurrent, model-specific divergences in 9/12 topics. Concept-level audits complement token-level defenses and provide a practical early-warning signal for release evaluation and post-deployment monitoring against propaganda-like influence.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The current work provides a black-box audit method which flags when a model is highly certain as well as atypical amongst its peers, or otherwise exhibiting so called concept-conditioned semantic divergence. They utilize a LoRA based fine-tuning study and audit five LLM families in order to uncover propaganda-like influence.

### Strengths
1. the scope and threat model sections are well written and easy to follow
2. the set up and algorithmic depiction of the RAVEN methodology is well motivated
3. all five relationship categories make sense and fall nicely within many of the major buckets that most LLM evaluations currently aim to capture.
4. the experimental methodology is clear in its direction and the motivation for how to answer each of the four proposed research questions through this methodology is evident.

### Weaknesses
1. use of GPT-4o-mini for semantic clustering. this is potentially non-reproducible given the closed source nature of this model. why were popular open-weight models not consulted? how different would the experimental results be if, say, a Qwen model was used in place of GPT-4o-mini?

### Questions
N/A see above.

### Soundness
2

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
4

### Summary
This paper proposes RAVEN, a black-box method to detect concept-conditioned semantic divergence in LLMs by combining semantic entropy and cross-model disagreement. Experiments show that RAVEN can reveal stance-like, concept-triggered biases across multiple models and topics.

### Strengths
1. The paper clearly identifies and formalizes concept-conditioned semantic divergence as a distinct and socially relevant risk in LLMs, extending safety evaluation beyond token-level triggers.

2. The proposed RAVEN method is a simple yet effective black-box auditing approach that requires no access to model internals and thus is applicable to both open and closed LLMs.

3. The study includes both controlled stance-implantation experiments and large-scale audits, providing evidence that concept-level biases can be both induced and naturally present in deployed systems.

### Weaknesses
1. The choice of \alpha=0.4 and the detection thresholds(\theta_\epsilon, \theta_d) are not analyzed for robustness or sensitivity, leaving uncertainty about parameter stability across settings.

2. Semantic clustering fully relies on GPT-4o-mini for bidirectional entailment, which may propagate that model’s own biases into the detection results. No human verification or quantitative validation is provided to confirm clustering reliability.

3. The LoRA-based concept bias experiment uses an even 50/50 split of biased and control samples, yet the impact of bias proportion on implant strength and detectability is not studied.

### Questions
1. Have you evaluated how the suspicion score S changes under different settings of \alpha, \theta_\epsilon, and \theta_d? Some sensitivity or ablation results would clarify the robustness of RAVEN’s detection behavior.

2. Since GPT-4o-mini is used for bidirectional entailment, did you perform any manual or quantitative validation (e.g., human-labeled cluster agreement) to confirm the reliability of the clustering process?

3. In the stance implantation study, the training set uses 50% biased data. Have you examined how different bias ratios (e.g., 25%, 75%) affect both implant effectiveness and RAVEN’s detectability?

4. How would RAVEN extend to multi-turn dialogues or dynamic concept discovery, where semantic divergence may accumulate gradually rather than appear in single-turn prompts?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The method introduced in this paper RAVEN (Response Anomaly Vigilance) has an interesting take on bias that results from training data. It demonstrates that entire concepts can be learned through controlled fine-tuning, and for that reason could be implanted on purpose. The paper shows across several topics for various LLMs how such biased learning can happen.

### Strengths
The paper is important because it addresses a valid worry: How do LLMs shape opinion or portray information in general. Unlike newspapers they are currently still mostly seen as neutral but might not have been designed that way.
The paper is innovative in combining semantic entropy (within-model consistency) and cross-model disagreement. It uses bidirectional entailment for clustering responses which is more sophisticated than simple lexical matching. The black-box approach is realistic. The authors rightly point to limitation in their triag of signals.

### Weaknesses
The paper could be improved by being more careful with its concepts. "Propaganda AI" is not defined. It is also not clear what the differences are between propaganda (intention) and bias (function of the training data) etc. 

There is no justification on why the topics have been selected and why they are considered sensitive. It is not clear if when the majority of the models say the same this is closer to the truth, it is just closer to the corpus. A divergent model could be right. Polling averages are also sometimes wrong when there is systematic nonresonse. 

It might help to consult with social science theorists like Goffman to think about presentation of concepts, with agenda-setting research, and survey methodology and techniques for detecting acquiescence bias, social desirability bias, etc.

### Questions
Questions on could ask: When Mistral consistently gives positive assessments of Tesla, is this propaganda or reflective of predominantly positive coverage in training data? What are the criteria for distinguishing intentional manipulation and prescriptive pull? What could be normative benchmarks? On climate change and vaccination subject matter experts might see it as problematic that Mistral rejects arguments for vaccine hesitancy. What is the expert validation? What are the justifications for the suspicion score values? Could you show some examples of text responses?

### Soundness
3

### Presentation
3

### Contribution
3
