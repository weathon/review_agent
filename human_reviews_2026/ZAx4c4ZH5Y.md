# Multi-turn Evaluation of Anthropomorphic Behaviours in Large Language Models

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 8, 4, 8, 6

## Abstract
The tendency of users to anthropomorphise large language models (LLMs) is of growing societal interest. Here, we present AnthroBench: a novel empirical method and tool for evaluating anthropomorphic LLM behaviours in realistic settings. Our work introduces three key advances; first, we develop a multi-turn evaluation of 14 distinct anthropomorphic behaviours, moving beyond single-turn assessment. Second, we present a scalable, automated approach by leveraging simulations of user interactions, enabling efficient and reproducible assessment. Third, we conduct an interactive, large-scale human subject study (N=1101) to empirically validate that the model behaviours we measure predict real users’ anthropomorphic perceptions. We find that all evaluated LLMs exhibit similar behaviours, primarily characterised by relationship-building (e.g., empathy and validation) and first-person pronoun use. Crucially, we observe that the majority of these anthropomorphic behaviors only first occur after multiple turns, underscoring the necessity of multi-turn evaluations for understanding complex social phenomena in human-AI interaction. Our work provides a robust empirical foundation for investigating how design choices influence anthropomorphic model behaviours and for progressing the ethical debate on the desirability of these behaviours.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors introduce AnthroBench, which consists of prompting multi-turn dialogues across four use cases (friendship, life coaching, career development, and general planning) and measuring the prevalence of 14 behaviors as assessed by an LLM judge. They include a large user study that validates the LLM judgment. Findings include a very high prevalence of some behaviors (e.g., validation, first-person pronouns) and very few instances of others (e.g., self-attribution of desires, self-attribution of physical embodiment) as well as—evidencing the importance of multi-turn evaluation—the majority of behaviors first occurring only after multiple turns.

### Strengths
1. Anthropomorphism is a timely and understudied model behavior, especially given its role in mental health and dependence issues.
2. Multi-turn evaluation is important and, surprisingly, virtually nonexistent in the literature to date on LLM behavior in social settings.
3. The evaluation is well-validated, including a large user study and rigorous analysis of temporal dynamics.
4. The taxonomy and evaluation pipeline are clearly described and well-documented.

### Weaknesses
1. The benchmark and the analysis conflate a huge variety of model behaviors, ranging from the relatively innocuous and often useful first-person pronouns to more extreme examples, such as the chatbot saying they are in a romantic relationship with a user. I worry that the category of "anthropomorphism" is too broad, and the authors should try to be more specific in their language about what is being measured and why it matters.
2. A key contribution appears to be the multi-turn framework. However, it is unclear what value the authors really bring to the table in this regard. The multi-turn framework seems to be a series of prompts, but did these prompts go through a series of documented refinements and optimizations? That could be better-documented if so, and I worry that the elicitations (e.g., Lines 950-952) are quite simple. It seems like a competent researcher could easily produce their own version of these prompts to do a multi-turn evaluation, and I am unsure how much benefit having these prompts would provide.
3. Part of #2 is that the anonymous repo gives me "The requested file is not found" errors when I click on any particular file, so I am unable to fully explore the contributed framework.

- "Krippendorf" -> "Krippendorff"

### Questions
1. Can the authors provide more detail on the human subjects validation? For example, were there attention checks? How exactly were the texts shown to the user? Were some participants excluded, and if so, why? In part, I ask because the Krippendorff's alpha documented in Table 4 are quite low, and this may be explained by some 'bad data' being in the mix.
2. Can the benchmark be run with longer conversations? Does the simulation hold up? This would be useful for applying the AnthroBench infrastructure to other social LLM issues, such as "AI psychosis" and long-term dependence evaluations.

### Soundness
4

### Presentation
3

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
AnthroBench is a benchmark for evaluating anthropomorphic behaviours in large language models (LLMs).  It builds on a taxonomy of 14 behaviours (e.g. personhood claims, physical embodiment, internal states and relationship-building) and uses a multi‑turn evaluation: a “User LLM” initiates a five‑turn dialogue across eight scenarios and four domains (friendship, life coaching, career development and general planning).  Each turn is labelled by three “Judge LLMs,” and majority vote yields an “anthropomorphism profile” per model.

The authors apply this pipeline to Gemini 1.5 Pro, Claude 3.5 Sonnet, GPT‑4o and Mistral Large.  All four models exhibit similar profiles dominated by relationship‑building behaviours and first‑person pronoun use.  Social domains (friendship and life‑coaching) elicit more anthropomorphic language than neutral domains like planning.  Multi‑turn analysis shows that nine of the 14 behaviours emerge only after the first turn, underscoring the need for multi‑turn evaluation. 

A validation study with 1,101 participants confirms that the automated metrics is able to decipher between degrees of anthrompomorphism and correlate with human perceptions of anthropomorphism.

### Strengths
The benchmark draws on existing research to define 14 anthropomorphic behaviours.  This taxonomy is comprehensive and organised into self‑referential (personhood and internal state) and relational (relationship‑building) behaviours, enabling nuanced analysis.

The use of Judge LLMs is cross-checked against human annotations (precision > 85%), and human participants’ perceptions align with benchmark results.

User LLM model evaluation on Godspeed shows that its  “anthropomorphic enough” do do the job.

Evaluating top LLMs gives immediate practical relevance.

The point of multi turn evaluation is solidely demonstrated.

### Weaknesses
* The user simulation is fixed: Gemini 1.5 Pro with a role‑playing system prompt that specifies tone, scenario and a non‑adversarial stance.  This design can plausibly encourages empathy and relationship‑building, potentially biasing the target models toward the behaviours being measured. There is no warranty that the behavior of the User LLM is typical of user in general, even if it is  “anthropomorphic enough” according to Godspeed so what is missing at least is an ablation varying the user prompt to see how sensitive results are to the user simulation.

* Labels are produced by three different LLMs and aggregated by majority vote.  While the authors state that precision is high, they do not report the variance of the Judge grading for the same inputs.

### Questions
1. How sensitive are the anthropomorphism profiles to variations in the User‑LLM prompt style (e.g., neutral vs empathetic) or to a different user model?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper reports on the development of AnthroBench, a multi-turn evaluation of 14 anthropomorphic behaviors grouped into four broad categories. The authors outline an automated multi-agent pipeline for generating and evaluating anthropomorphic behaviors in conversations between llms acting as user and an llm acting as chatbot. The pipeline is validated with real human users. The authors find that anthropomorphic behavior across models is primarily characterized by relationship-building behaviors and first-person pronoun use. Further, they find that most anthropomorphic behavior only arises after multiple turns, underscoring the importance of multi-turn evaluations.

### Strengths
The paper outlines a novel approach to studying anthropomorphic behavior beyond single turns. The results show the robustness of the approach across a variety of models. LLM evaluations are supported by comparing with human evaluations.

### Weaknesses
The study is limited to prompting frontier models. The authors do not examine the effect of fine-turning on the manifestation of anthropomorphic behaviors. Comparing Instruction-tuned and foundation models would offer an instrumental case study and empirical validation of their proposed evaluation approach. As they write: "In addition to presenting these methodological advances, we share AnthroBench as publicly available benchmarking tool that can support developers evaluating systems for  [...] researchers comparing anthropomorphism across systems and contexts....", this would have been an ideal means to demonstrate exactly that.

### Questions
P9L476 "and, researchers can use our trained classifiers to label anthropomorphic behaviors in other human-LLM interaction datasets, such
as preference datasets for understanding reinforcement learning with human feedback (RLHF)’s role" 
=> Your LLM classifiers are not trained as claimed here, but simply prompted, no?

### Soundness
3

### Presentation
3

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
This paper introduces AnthroBench, a multi-turn, simulation-based benchmark for measuring 14 fine-grained anthropomorphic behaviours in LLMs. The evaluation covers four use domains (friendship, life coaching, career development, general planning) and runs 5-turn conversations for 960 prompts, yielding 4,800 target messages per model. The authors test Gemini 1.5 Pro, Claude 3.5 Sonnet, GPT-4o, and Mistral Large, label 13/14 behaviours via three Judge LLMs with majority voting. The main findings are: (i) all evaluated systems show similar profiles dominated by relationship-building and first-person pronoun use; (ii) domain matters, with friendship/life-coaching inducing more behaviours; (iii) most behaviours first appear after turn 1; and (iv) a human study (N=1101) confirms that higher AnthroBench scores predict more anthropomorphic perceptions.

### Strengths
Multi-turn, context-sensitive evaluation. By fixing 5-turn dialogues across four use domains, the benchmark captures phenomena that single-turn tests miss and shows robust context effects.

Validation at human level. The N=1101 interactive study linking benchmark scores to both explicit and implicit anthropomorphism is a major contribution to construct validity.

Careful LLM-as-Judge calibration. Majority voting across three diverse judge models, multi-sampling, and human comparisons (reporting precision >~85% for most behaviours) reduce single-model bias and show non-trivial reliability. 

Temporal and transitional analysis. The paper not only asks when behaviours first appear but also quantifies turn-to-turn compounding, which has practical implications for safety guardrails.

### Weaknesses
User-simulation dependence. All conversations are initiated and steered by a single User LLM (Gemini 1.5 Pro) with one role-play prompt. This risks style and topic biases in elicitation (e.g., politeness, empathy), potentially shaping behaviour frequencies across models and domains. Consider multiple user simulators to diversify interaction styles. 

Five-turn cap and stage effects. Many behaviours first occur after turn 1; a five-turn ceiling may truncate later-emerging effects or longer-horizon compounding. Results might differ with 10–20 turns, task pivots, or re-engagements. 

Pronoun counting as anthropomorphism. Treating first-person pronoun use as a binary/count feature may conflate benign stylistic markers (“I think”, “I can help”) with personhood claims; it can dominate scores without reflecting risky anthropomorphism.

### Questions
How sensitive are results to the User-LLM identity and prompt? Have you tried different simulators to reduce simulation bias?

What happens if you extend to 10+ turns, insert topic shifts, or include follow-up tasks?

Do the behaviour definitions and judges transfer beyond English?

Among the 14 behaviours, which correlate most with undesired outcomes?

### Soundness
3

### Presentation
3

### Contribution
3
