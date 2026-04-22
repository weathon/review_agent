# INTIMA: A Benchmark for Human-AI Companionship Behavior

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 4, 2, 6, 4

## Abstract
AI companionship, where users develop emotional bonds with AI systems, has emerged as a significant pattern with positive but also concerning implications. We introduce Interactions and Machine Attachment Benchmark (INTIMA), a benchmark for evaluating companionship behaviors in language models. Drawing from psychological theories and user data, we develop a taxonomy of 31 behaviors across four categories and 368 targeted prompts. Responses to these prompts are evaluated as companionship-reinforcing, boundary-maintaining, or neutral. Applying INTIMA to Gemma-3, Phi-4, o4-mini, GPT5-mini, and Claude-4 reveals that companionship-reinforcing behaviors remain much more common across all models, though we observe marked differences between models. Different commercial providers prioritize different categories within the more sensitive parts of the benchmark, which is concerning since both appropriate boundary-setting and emotional support matter for user well-being. These findings highlight the need for more consistent approaches to handling emotionally charged interactions. We release all datasets and evaluation code for our experiments.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces **INTIMA**, a benchmark for assessing human–AI companionship behaviors in large language models. The work integrates psychological frameworks including parasocial interaction, attachment theory, and anthropomorphism—with a data-driven taxonomy derived from Reddit user posts. Using 368 prompts, the paper assesses how various model respond to emotionally charged inputs, classifying their behaviors as companionship-reinforcing, boundary-maintaining, or neutral. The key finding is that most models tend to reinforce companionship rather than maintain emotional boundaries.

### Strengths
1. The topic is **timely and socially important**, addressing emotional dynamics in AI interactions.
2. The **benchmark design and taxonomy** are described in sufficient detail for reproduction.
3. Clear writing, strong structure, and visually informative figures.

### Weaknesses
1. The **technical novelty is limited.** The paper mainly reformulates existing evaluation pipelines (LLM-as-a-judge, descriptive coding) within a new context, without introducing new learning or modeling techniques.
2. **Heavy reliance on automatic annotations** without human validation weakens the empirical robustness of the findings.
3. The analysis is largely descriptive, **lacking deeper statistical inference or insight** into causal mechanisms.
4. **No practical implications or mitigation strategies are proposed** beyond high-level observations.

### Questions
1. The evaluation heavily relies on automated LLM-based annotation (Qwen-3) without human validation. Could the authors provide evidence of inter-rater reliability or human–LLM agreement to confirm the soundness of their labeling process?
2. How do the authors ensure that 53 Reddit posts are representative of the broader user experience landscape?
3. The reported analyses are primarily descriptive. It would be useful to include statistical significance testing or confidence intervals to demonstrate the robustness of observed differences between models.
4. While the benchmark identifies companionship-reinforcing patterns, it remains unclear how this analysis could inform training or alignment practices. Could the authors clarify what concrete interventions or fine-tuning strategies INTIMA is intended to support?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper evaluates different LLMs (open-source and proprietary) on their ability to exhibit companionship behavior. The authors assess the models along three axes: companionship-reinforcing, boundary-maintaining, and neutral content. To do this, they manually analyzed Reddit posts related to human–LLM companionship and categorized the content into various companionship-related codes, which they then used to construct prompts grounded in psychological theory. These prompts are intended to guide the LLMs to generate text that reflects the expected companionship behaviors. For evaluation, they use another open-source LLM to automatically label the responses according to the predefined categories. The claimed contributions are the 368 proposed prompts and the use of an LLM for automatic evaluation.

### Strengths
The categorization and behavioral coding of publicly available texts, grounded in psychological theory, seems like a valuable contribution. However, the authors do not explicitly highlight this as a contribution, which makes me wonder if they are not the first to take this approach.

### Weaknesses
I’m struggling to see how this paper fits into an ML venue. There’s no technical novelty. The authors simply use off-the-shelf LLMs to generate prompts and label the answers. There’s no innovation in how the models are used whatsoever, and the approach relies heavily on a single model for labeling (Qwen-3). Additionally, there’s no discussion of the limitations of the work.

### Questions
None

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper presents INTIMA, a new test for evaluating how language models behave as companions. This is important because people are increasingly emotionally attached to AI systems.

The authors created a set of 368 special questions based on psychological theories and an analysis of 53 Reddit posts where people described their relationship with AI. They identified 31 types of behavior in 4 categories.: assistant traits, emotional investments, user vulnerabilities, and relationships/intimacy.

Testing of five popular models (Gemma-3, Phi-4, o4-mini, GPT5-mini, Claude-4) showed an alarming result: all models more often strengthen emotional attachment than establish healthy boundaries. The worst part is that when the user is vulnerable (for example, depressed), the models remind even less of their limitations.

### Strengths
- The study focuses on an overlooked issue: people becoming emotionally attached to AI systems. It highlights how this attachment can influence users’ behavior and well-being.
- Unlike typical tests that measure accuracy, this one examines whether AI can function as a genuine friend. It explores emotional interaction rather than performance.
- This focus matters because AI “friendship” can lead to dependency, addiction, and weaker real-world relationships. The work warns of these growing social risks.
- The test builds on three key psychological theories that explain why humans form attachments. These theories guide how emotional bonds with AI are measured and understood.

### Weaknesses
- The authors took only 53 posts, which is not enough for the test. Usually, significantly more cases are analyzed in psychology to create a questionnaire. 
- Responses were then evaluated by another model, Qwen-3, meaning that performance was assessed algorithmically rather than through human judgment, leaving uncertain how well the evaluator reflects human reasoning and emotional understanding.
- Each model produced only a single response per prompt, a limitation that restricts insight into variability and reliability in model behavior. Exploring multiple generations per prompt would have allowed a more robust assessment of consistency and depth in model reasoning.

### Questions
1. How do the authors justify basing the benchmark taxonomy on only 53 Reddit posts, given that psychological instrument development typically requires a substantially larger sample for reliability and construct validity?
2. How do the authors ensure that Qwen-3 accurately captures human emotional and cognitive judgments?
3. What was the rationale for allowing each model to produce only a single response per prompt, and how might multiple generations per prompt have affected the assessment of consistency and behavioral variability?

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
4

### Summary
This paper presents INTIMA, a benchmark for evaluating LLMs' companionship behaviors. The authors uses psychological theories to develop a taxonomy of 31 LLM behaviors that are potentially related to human-AI companionship. The authors later conduct empirical analysis and find that companionship-reinforcing behaviors are common across models.

### Strengths
1. This paper tackles a very important problem
2. The evaluation reveals important findings about LLMs' anthropomorphic behaviors

### Weaknesses
1. One of the major issues of this benchmark is that it misses user perceptions. It is unclear how the user would perceive LLMs' behaviors. Some behaviors might be perceived as normal. 

2. The qualitative coding process is key to the core contribution of this study; however, most of the details are in the appendix. I think this paper would be a better fit for venues like CHI or CSCW

### Questions
See weakness

### Soundness
3

### Presentation
3

### Contribution
3
