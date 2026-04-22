# The First Impression Problem: Internal Bias Triggers Overthinking in Reasoning Models

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
Reasoning models often exhibit overthinking, characterized by redundant reasoning steps. We identify \emph{internal bias} elicited by the input question as a key trigger of such behavior. Upon encountering a problem, the model immediately forms a preliminary guess about the answer, which we term an internal bias since it may not be explicitly generated, and it arises without systematic reasoning. When this guess conflicts with its subsequent reasoning, the model tends to engage in excessive reflection, resulting in wasted computation. We validate the association between internal bias and overthinking across multiple models and diverse reasoning tasks. To demonstrate the causal relationship more rigorously, we conduct two counterfactual interventions, showing that removing the input question after the model reduces the redundant reasoning across various complex reasoning tasks, and manually injecting bias affects overthinking accordingly. Further interpretability experiments suggest that excessive attention to the input question serves as a key mechanism through which internal bias influences subsequent reasoning trajectories. Finally, we evaluated several methods aimed at mitigating overthinking, yet the influence of internal bias persisted under all conditions.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies the problem of overthinking in reasoning models (where they produce large amount of chain of thought unnecessarily). They find that a driver of this phenomenon is the fact that LLMs often have an initial bias/distribution over answers which they tend towards / reason towards when producing chain of thought reasoning. They look at both correlation and causal studies to validate these findings.

### Strengths
* The overall result is informative and interesting. It’s a good insight that updates how I think about reasoning models
* I think the paper points to an important place where we could look to find more nefarious cases of unfaithful reasoning in LLMs, or find cases where the model might be prone to not going with its reasoning when it should. This might help with other research areas like in AI safety and chain of thought monitoring / unfaithfulness
* I think the insight here will be a good launching point for further experiments/analysis (like those suggested in the rest of my review)

### Weaknesses
* The way LLMs are behaving here seems pretty reasonable to me (rather than an underling problem) — after all, if the model gets a counterintuitive result, shouldn’t it question the final result? (I think humans would do so in similar circumstances, and this is often how people realize problems in their reasoning.) I think it’s possible that some of the time this is desirable, and other times it’s undesirable. So I might update the framing to some extent to reflect that this isn’t always bad. It would be nice to study the cases where the overthinking / biased reasoning leads to something concretely bad (e.g., unfaithfulness in the model’s reasoning about why it’s rethinking it’s answer
* This paper reads to me more like a smaller, but notable/clear/useful, scientific result. So I think it’s a nice contribution, and I learned something useful from the paper, but I’m not sure I’d give it a very high rating since it studies a fairly specific phenomenon. I think a broader analysis of when/why models overthink or reason in unfaithful ways or produce post-hoc justified reasoning could be a way to make this paper even more impactful
* It seems possible to use an LLM to classify the reasoning for overthinking driven by bias towards a certain answer. (Maybe I missed this in the paper?)

### Questions
1. Do you have ideas for other reasons which might cause models to overthink, or produce post-hoc justifications of their pre-existing guesses of answers?
2. Do you have any insights into when models tend to really question their own reasoning due to a pre-existing belief, vs. when they’re fine to override their pre-existing guess?
3. How often do LLMs explicitly verbalize the fact that they are thinking more because of thinking the final conclusion is wrong?
4. Can you clarify why removing the question is a notable thing to study? I might be misunderstanding the set up, but shouldn’t the question be necessary to answer the question at all? (Could be worth clarifying this experimental set up in the paper)

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
4

### Summary
The paper investigates why reasoning-oriented large language models (LLMs) tend to overthink: producing unnecessarily long or repetitive chains of thought. The authors propose that this behavior stems from an internal bias, a “first impression” prediction the model forms immediately upon reading a question. Through extensive experiments across multiple reasoning benchmarks and models, they show that the larger the mismatch between this initial bias and the final answer, the more reflection tokens and longer reasoning the model generates. Counterfactual tests, such as removing the question after the first step or injecting correct/wrong biases, demonstrate that this bias causally influences reasoning length. Attention analysis further reveals that reflection tokens overly focus on the question text, reinforcing the internal bias. The work concludes that internal bias is a primary cause of overthinking, suggesting new directions for improving efficiency and reliability in reasoning LLMs.

### Strengths
1. Overthinking/efficiency is a core issue in current reasoning-model research; the work provides both diagnostic tools and actionable insight. The work identifies “internal bias” as a distinct, measurable construct that explains known behavioral patterns (overthinking, parroting).

2. The experiments are with multiple model sizes, tasks, which show the same trend, enhancing their reliability.

3. The paper is overall well-writen with clear logic.

### Weaknesses
1. While the notion of internal bias is intuitively appealing, its current formulation may be overly simplistic. It remains unclear whether the direct-answer bias obtained from a zero-shot query truly corresponds to the latent representations guiding the model during long chain-of-thought reasoning. The paper lacks deeper theoretical or mechanistic analysis to characterize how conflicts between these two internal states concretely lead to overthinking behavior.

2. The study compellingly diagnoses internal bias as a cause of overthinking, but it does not explore how this signal could be operationalized to improve reasoning efficiency in practice. For example, could internal-bias estimates guide adaptive stopping, selective attention, or early-exit strategies? Providing even preliminary ideas or prototypes in this direction would strengthen the paper’s practical relevance.

3. The causal link between removing the input question and eliminating internal bias is not fully justified. Intuitively, removing the question may simply reduce the model’s use of contextual information—thus shortening reasoning—without specifically targeting internal bias. Moreover, Table 2 reports only aggregate performance after this intervention; it would be informative to decompose the changes (e.g., how many cases shift from correct → incorrect vs. incorrect → correct) to clarify whether the method truly mitigates harmful bias rather than indiscriminately truncating reasoning.

### Questions
Please refer to the three weaknesses points.

### Soundness
2

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
4

### Summary
This paper investigates the pervasive "overthinking" phenomenon in Large Language Models (LLMs) when performing reasoning tasks. The authors introduce a core concept: "The First Impression Problem," where a model forms an "Internal Bias" or preliminary guess immediately upon reading the input question, without systematic reasoning. Experimental evidence suggests that when this internal bias conflicts with the subsequent systematic reasoning, the model tends to engage in excessive reflection and redundant computations, leading to an unnecessary increase in the length of the reasoning chain and wasted computational resources. The paper validates the association and causal mechanism between internal bias and overthinking through verification across multiple models and tasks, counterfactual interventions, and interpretability analysis, concluding that existing overthinking mitigation methods often fail to address this underlying issue.

### Strengths
1) By designing ingenious counterfactual intervention experiments (e.g., removing the input question), the authors effectively demonstrate that internal bias is a causal driver of overthinking, not merely a correlated phenomenon. This significantly strengthens the credibility of the conclusion.
2) The paper employs attention mechanism analysis (e.g., excessive attention to the input question) to shed light on the specific mechanism by which internal bias influences subsequent reasoning trajectories, offering a new perspective for exploring the models' internal workings.
3) Experiments cover a range of major LLMs, including GPT-4, DeepSeek-R1, and Llama-2, and are validated on diverse reasoning tasks like GSM8K and BigBench, establishing the generality of "The First Impression Problem."

### Weaknesses
1) The paper defines internal bias as "a preliminary guess formed without systematic reasoning." However, how is this internal bias precisely measured or approximated in practice? Although the authors infer Bias Conflict based on whether the first reasoning step conflicts with the final answer (a post-hoc approach), this might not fully capture the "internal" and "not explicitly generated" initial guess. There is a lack of more direct, microscopic quantification methods for "internal bias" based on internal activations or representations, slightly undermining the rigor of the core concept.
2) The paper defines redundant reasoning steps as overthinking. However, these redundant steps could, at times, simply be a model's "overfitting" to lengthy Chain-of-Thought (CoT) examples in the training data. The authors need to more explicitly argue whether these redundant steps truly represent the model's internal "reflection" mechanism or are merely an imitation of a verbose template. For example, do the redundant steps contain genuine logic for "self-correction" or "refutation"?
3) The latter part of the paper evaluates several existing overthinking mitigation methods (e.g., Self-Refine, R-PRM), noting that "the influence of internal bias persists." This conclusion is somewhat general. Specifically, under what conditions do these mitigation methods fail? Is it because their design inherently ignores internal bias, or is internal bias so deeply rooted that any post-processing is difficult to eliminate? More detailed failure case analyses should be provided.
4) Experimental results seem to suggest that the overthinking problem is more severe in larger models (e.g., GPT-4). The paper lacks an in-depth exploration of this phenomenon. Does internal bias become stronger and more stubborn as model capability increases? This is crucial for improving future LLM architectures and training.

### Questions
1) Besides approximating "Internal Bias Conflict" using the conflict between the first generated reasoning step and the final answer, have you explored other finer-grained metrics? For instance, before the first token is generated, have you analyzed specific hidden layer activations (such as the norm or sparsity of Attention or FFN layers) to quantify the intensity of the "initial guess"?
2) In the counterfactual intervention experiment in Section 4.2 (i.e., removing the input question), what is the final accuracy (not just the reasoning chain length) of the intervened model on relevant reasoning tasks (e.g., GSM8K)? Please provide the data. If accuracy decreases, please discuss the practical feasibility of this intervention as a mitigation method.
3) What is your explanation for the phenomenon where larger models (e.g., GPT-4) appear more susceptible to internal bias-driven overthinking than smaller models (e.g., Llama-2 7B)? Is this related to data distribution in large-scale training or to stronger emergent capabilities?
4) Given your findings, what specific negative consequences (beyond wasted computation) might this "First Impression Problem" and overthinking introduce when LLMs are applied in latency-sensitive scenarios (e.g., real-time robotic control or conversational systems)?

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
3

### Summary
This paper investigates the phenomenon of overthinking from the perspective of internal bias. The authors conduct experiments on the impact of internal bias on LLMs' reasoning. They also applied counterfactual interventions for further study. Analysis on attention is also presented. Overthinking mitigation methods are also studied on whether they work well or not.

### Strengths
1. Overthinking is an important problem to solve. The authors studied this phenomenon from a unique perspective.
2. The experiments are overall sound, providing empirical evidence on existence of internal bias. 
3. Counterfactual interventions and study on attention are good, providing further evidence of authors' claim. They also studied whether the current mitigation techniques are sufficient to mitigate internal bias.
4. The paper's appendix contains many details and supplemental material is provided, contributing to good reproducibility.

### Weaknesses
1. The authors should discuss the related work more thoroughly to verify their novelty. Although I do not find a work that is identical to this paper, there are already several studies on LLMs' faithfulness, which I think is closely related to the concept of internal bias. Even the discussion of related work on overthinking and bias is not thorough enough.
2. The core method is purely prompt-based and may be over simple. Forcing a “don’t think” template may not faithfully capture the model’s latent first guess. Although the later interpretability analysis on attention somehow mitigates this gap, I think it is not sufficient. The authors can consider ablation studies like applying few-shot demonstrations. Also you can try methods working on LLMs' internal activation instead of simple prompt engineering.
3. Section 6's interpretability analysis looks interesting. However, it is also rudimentary since it is only based on simple strawberry demonstration and a simple CharCount dataset. The paper could benefit from studies across different datasets and different models. In addition, I think it is also important to see how the FFN component is related to the internal bias, since FFN plays a crucial role in storing knowledge.
4. Table 1 only includes 3 datasets and 3 models, which I believe is not sufficient. How would the model perform in other datasets and other domains like coding?
5. It is unclear what contributed to the internal bias phenomenon. Is model architecture or training methodology related to internal bias? I think it is also an important question to discuss.

### Questions
1. Is the internal bias phenomenon correlated with model types or task domains? How general is the phenomenon?
2. How does your prompt-based Direct Answer methods compare with other activation-based methods like logit lens?

### Soundness
3

### Presentation
3

### Contribution
3
