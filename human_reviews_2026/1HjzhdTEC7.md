# Theory-Grounded Evaluation of Human-Like Fallacy Patterns in LLM Reasoning

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 0, 8, 4, 6

## Abstract
We study logical reasoning in language models by asking whether their errors follow established human fallacy patterns. Using the Erotetic Theory of Reasoning (ETR) and its open‑source implementation, PyETR, we programmatically generate 383 formally specified reasoning problems and evaluate 38 models. For each response, we judge logical correctness and, when incorrect, whether it matches an ETR‑predicted fallacy. Two results stand out: (i) as a capability proxy (Chatbot Arena Elo) increases, a larger share of a model’s incorrect answers are ETR‑predicted fallacies ($\rho=0.360, p=0.0265$), while overall correctness on this dataset shows no correlation with capability; (ii) reversing premise order significantly reduces fallacy production for many models, mirroring human order effects. Methodologically, PyETR provides an open‑source pipeline for unbounded, synthetic, contamination‑resistant reasoning tests linked to a cognitive theory, enabling analyses that focus on error composition rather than error rate.

## Human Reviews

## Human Reviewer 1

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
### Summary: *Analyzing LLMs through the Erotetic Theory of Reasoning (ETR)*

The paper analyzes large language models (LLMs) through the lens of the **Erotetic Theory of Reasoning (ETR)** — a framework developed by philosophers and logicians to study how questions relate to one another.

### **Background: Key Idea of ETR**
ETR focuses on how the **answer to one question can resolve or transform another question**.  
For example:

> “If the light is on, then the power is on.”

This statement evokes two questions:
1. Is the light on?
2. Is the power on?

Answering *yes* to the first question naturally resolves the second one, concluding that the power is on.

### **Experimental Setup**
The authors use **PyETR**, a tool that automatically derives ETR structures from statements, to generate prompts composed of:
**theme introduction**, a set of **premises**, and a follow-up **question** asking what logically follows from those premises.
They source their ETR style reasoning problems from Reason and Inquiry text (Koralus, 2022).

### **Evaluating Logical Correctness of Model Response**
To test logical correctness, they use **PySMT**, a logic solver already open-sourced for the community.
The paper measures the logical correctness by setting the criterion that if the **negation** of a model’s conclusion contradicts the premises, the conclusion is considered **logically correct**.

### **Evaluating Human-like Fallacies**
To assess human-like reasoning errors, the authors check whether the model’s answer matches ETR’s prediction of what a human would infer,  but is not logically correct. These cases are labeled as ETR-predicted fallacies. 
Finally, the **fallacy rate** is defined as the proportion of these human-like fallacies among all logically incorrect responses.

### **Key Findings of This Paper**
- There is a **negative correlation** between logical ability and fallacy rate:  
  **Stronger models are more likely to make human-like reasoning mistakes** when they do err.
- Reversing the **order of premises** in the same logical problem often **reduces fallacy production**, showing that LLMs exhibit **order effects** similar to humans.

### Strengths
This paper draws ETR reasoning theory into evaluating LLM behavior, suggesting a viewpoint that logically strong models in terms of reasoning ability are more prone to making human-like fallacy errors. This is evaluated by employing existing open-source implementation of ETR logic solver, PyETR, and existing dataset "Reason and Inquiry text".

### Weaknesses
### **1. Low-Quality Prompt Design**

The prompts generated for evaluation appear to be poorly constructed, making it unclear whether models could interpret or respond to them properly.

**Evidence**: The reported correlation between model performance (as measured by LLM Elo scores) and logical correctness is nearly zero
(r = 0.004, p = 0.981; ρ = −0.04, p = 0.777). This indicates that the dataset may not reliably capture reasoning ability, suggesting that prompt design, rather than model reasoning, could be the limiting factor.
This undermines the paper’s main claim about fallacy trends across model capabilities, since a dataset that fails to differentiate between weak and strong models cannot support such conclusions.

### **2. Lack of Methodological Clarity**

Several crucial details are omitted or insufficiently explained, making it difficult to assess or reproduce the study.

a. The statement “Pre-analysis integrity checks identified 17 items whose fallacy status did not meet the final specification” is ambiguous. The paper never defines what “final specification” refers to or how it was determined.

b. The authors mention creating “12 themes for natural language framings of logical problems,” yet these themes are not listed, exemplified, or referenced in the appendix or main text. Such omissions significantly reduce transparency and reproducibility.

c. The criterion for logical correctness (checking whether the negation of the conclusion contradicts the premises) is introduced without any justification or explanation of the underlying logic principle. While the negation-contradiction test is a valid formal method for entailment checking, it should have been theoretically motivated and explained for readers unfamiliar with formal logic or SMT solvers.

### **3. Conceptual and Logical Weaknesses**

The paper’s contribution is mainly an application of an existing psychological framework (Erotetic Theory of Reasoning, ETR) to LLM outputs, with little novel insight into model-specific reasoning mechanisms. It feels more like a reinterpretation of known model behaviors through an old theoretical lens rather than a discovery of new phenomena.

There is a logical circularity in the argument: the paper assumes that ETR is a valid description of human reasoning and then uses it as both the theoretical predictor and evaluation criterion for LLMs. This conflates explanatory alignment with empirical validation — the models might appear “ETR-like” simply because the evaluation framework enforces it.

The study treats fallacy alignment with ETR as evidence of human-likeness, but it never justifies why human-like fallacies should increase with logical ability. This causal interpretation is speculative and unsupported by formal or cognitive reasoning theory.

### **4. Poor Exposition and Notation Clarity**

Section 2, which introduces ETR, is very difficult to follow. The text uses non-standard and undefined notations without clear definitions or examples. The presentation of PyETR is redundant and unfocused, repeating information available in prior works without connecting it directly to this study’s experimental design.

Moreover, the citation style is inconsistent: citations should be enclosed in parentheses unless grammatically integrated into the sentence. Several sentences lack periods or contain minor grammatical errors, making the text unnecessarily difficult to read. These stylistic and formatting lapses reflect insufficient editorial care, which reduces the paper’s overall readability and professionalism.

### **5. Lack of rigorous experimental design and analysis**
There is no ablation or control analysis demonstrating that the observed “fallacy-blocking” from reversing premises is statistically robust or independent of surface-level linguistic cues. The paper conflates logical validity (a formal property) with psychological plausibility (a behavioral trait), without acknowledging that these are distinct dimensions of reasoning.

### Questions
The questions mainly concern the clarity of the notations, the ETR-style logical expressions, and the undefined logical operators that appear throughout the paper.
For example, what does “p” represent in the reported Pearson/Spearman correlations? I assume it refers to the p-value, but such notation should be explicitly defined, not used without explanation.

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
Note that I have previously reviewed this paper for another conference. The paper is substantially similar, and I copy some of the review here.

This paper shows that language models which are “stronger” (according to Chatbot Arena Elo score) produce more “human-like” errors, specifically those described by the “erotetic theory of reasoning” (ETR), which posits that humans reason by maintaining disjunctive alternatives and filtering these alternatives when taking on new information.

### Strengths
I find the results and analyses to be sound, interesting, and worth sharing. 
* interesting and important result about how models interpret disjunctive normal forms in ways similar to humans
* the use of the ETR model is novel and the analyses are sound

I also appreciate that the authors have addressed my previous concerns, copied here:
* The Chatbot Arena may not be the best proxy for "strong" models, given that human preferences on the arena may be prone to surface features like formatting (which the arena tries to account for) and sycophancy. Recent work has also demonstrated the fallibility of the leaderboard (https://arxiv.org/abs/2504.20879). I think a plausible alternate interpretation of the results is that Chatbot Arena measures (amoong other things) how well a model interprets natural language in the way that humans do. Given that humans interpret disjunctive normal forms in a particular way, the ELO score reflects this. Thus, intead of speaking to patterns in the behavior of models of various "strength", these results might instead be speaking more to what Chatbot Arena Elo is measuring
* I think it’s a bit misleading to call it an inverse scaling law, which most people would I think interpret as indicate worse absolute performance with scaling. Rather, here the result is about the proportion of errors that fit the etoteric pattern
* I may have misunderstood the methods, but I believe that the evaluation set mostly revolves around disjunctive normal forms and slight variations thereof. I think this is a relatively narrow set of behaviors, and the claim in the title may be overreaching given this narrowness

### Weaknesses
No substantial concerns

### Questions
I would also be interested to see the analysis separately for non-reasoning and “reasoning” models (esp those models trained via long-range RL, such as the o-series from OpenAI). And also models that are trained more heavily on e.g. code rather than human language. But this is not so much a concern as a curiosity.

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
4

### Summary
This paper uses the ETR to evaluate whether LLM errors match human fallacy patterns across 383 reasoning problems and 38 models. The key finding is,  higher-capability models (Chatbot Arena Elo score) produce a significantly larger proportion of human-like fallacies (ρ = 0.360, p = 0.0265),  but no improvement in overall correctness. This suggests scaling increases alignment with human error patterns rather than improving logical reasoning accuracy.

### Strengths
1. This paper introduces a mature method for evaluating human fallacies, which has a solid theoretical foundation and provides good reproducibility based on the open-source pyETR implementation.
2. Extensive experimental validation was conducted on a wide range of models (of different sizes and architectures), and the results are quite comprehensive.
3. Insightful experimental designs explored how human thought patterns manifest in large-scale models.

### Weaknesses
1. Experiments indicate that the moderately low correlation weakens the experimental claims put forward in the paper. 
2. The paper lacks theoretical and principled analysis, focusing instead on assessing the similarity between LLMs and humans in reasoning errors and presenting observed experimental results. However, it fails to explore potential causes for this phenomenon or propose solutions to the problem, resulting in limited practical applicability.

### Questions
1. This paper evaluates model performance solely through the relatively comprehensive Elo metric. However, it remains unclear whether the reasoning fallacy test is more inclined toward models capable of generating long reasoning chains, or whether it is more closely related to the model's ability to answer questions in the STEM domain.
2. Could a more comprehensive set of metrics be employed to holistically evaluate the model's performance, encompassing aspects such as answer diversity and reasoning capabilities? Further analysis of the relationship between human-like errors and these capabilities would strengthen the article's conclusions.

### Soundness
2

### Presentation
2

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
The paper looks into whether LLMs generate predictable fallacies like humans using ETR. Using the PyETR framework, authors generate synthetic reasoning problems and evaluate 38 LLMs. Their findings indicate that as models get more capable (with chatbotarena elo as a proxy) the predictability of the inaccurate responses increases (they get more human-like). The paper highlights that more capable models don’t just make fewer errors, they make more human-like ones.

### Strengths
1.  I resonate well with evaluating the human-likeness of reasoning of LLMs - LLMs are trained with trillions of tokens that are human written, and it is expected that human cognitive biases should show in their generations. And in that sence I find the paper quite interesting

2. The paper also brings in the research question of predictability of errors. And shows that more capable model are more predictable as well

3. Authors have done a good job at addressing a bigger audience than the ones familiar with cog-sci. I feel the paper is written well enough for someone not familiar with ETR to read and appreciate this. And I commend this.

That being said, I have concerns too. Please see weaknesses.

### Weaknesses
Weakness: 

1. My first and major concern is the scope of the work. I think the section 4 needs more insights. I feel the scope is too limited in its current form.

2. I think the paper can benefit from some anecdotal analysis, i.e. if the authors could give more insights of the form "the reasoning pattern in model X is like Y, and hence …." instead of a high level correlation metric reported.

### Questions
Please see weakness

### Soundness
2

### Presentation
3

### Contribution
2
