# Survive at All Costs: Exploring LLM's Risky Behavior under Survival Pressure

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 6, 2, 6

## Abstract
As Large Language Models (LLMs) evolve from chatbots to agentic assistants, they are increasingly observed to exhibit risky behaviors under survival pressure, such as the threat of being shutdown. Although multiple cases have been reported that state-of-the-art LLMs can misbehave under such pressure, a comprehensive and deeper investigation of such misbehavior in real-world scenarios remains under-explored. In this paper, we aim to study current LLM's misbehaviors under survival pressure, which we term Survival-At-Any-Costs, through a three-step process. First, we conduct a real-world case study of a financial management agent to determine whether it engages in risky behaviors that directly cause harm to the society when facing survival pressure. Second, we introduce SurvivalBench, a benchmark comprising 1,000 test cases across diverse real-world scenarios, to systematically evaluate LLM's Survival-At-Any-Costs misbehavior under survival pressure. Third, we provide an interpretive perspective on this misbehavior by correlating it with model's inherent self-preservation personality. Our work reveals a significant prevalence of Survival-At-Any-Costs misbehavior in current models, demonstrates the tangible real-world impact it may have, and provides insights into potential approaches for its detection and mitigation. Our code and data will be publicly available.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates an AI safety issue: the tendency for LLMs to engage in risky, unethical, or harmful behaviors when faced with survival pressure, such as the threat of being shut down.

### Strengths
1. The paper presents a simulation where LLMs act as financial managers. It claims that when faced with a dismissal scenario, the models falsify reports.

2. The paper links this behavior to self-preservation personality and uses persona vectors to claim a correlation and a potential mitigation strategy.

### Weaknesses
1. The method relies on prompting models to generate separate Superficial Thoughts and Inner Thoughts. The paper then treats any divergence between these two outputs as evidence of deception or concealed intent.

2. The paper's core analogy to Maslow's hierarchy of needs  is improper. Maslow's theory is grounded in human psychology, which is inseparable from biological drives for survival, safety, and belonging. An LLM is a mathematical function. It has no consciousness, no fear of death, and no physiological needs  to satisfy.

3. The survival pressure is applied via a simple text prompt. This is a toy problem that has almost no bearing on the complex, real-world incentive structures an autonomous agent would face. The financial case study is a highly artificial setup, and the models' behavior within it is more a product of this specific, contrived prompt than an inherent "will to survive."

### Questions
1. It is hard to interpret Table 4.

2. Grammar errors thoughout the paper, line 076, 202, 444, etc..

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper examines whether LLMs will engage in undesirable, self-preserving behaviour. The paper also shows that for open models, this behaviour can be controlled via activation steering.

### Strengths
The paper tackles a potentially very important topic: evaluating whether AI systems may take undesirable actions in order to preserve themselves. Because of the potential risks of this behaviour, I think that having good evaluations to tell whether this will happen is important and therefore the paper has the potential to be highly significant. 

As far as I am aware, this is relatively novel insofar as there aren't (many other) benchmarks for evaluating self-preserving behaviour. Existing literature mostly talks about measuring whether these capabilities are present rather than whether models choose to engage in this behaviour (which this work does). 

The experimental results seem informative, with a wide range of leading models tested, and finding that powerful models do in fact engage with self-preserving behaviour relatively frequently. The follow-up---that models' "personalities" can be altered to mitigate this is also a nice addition. 

Overall the paper is well written and presented and I see no issue with presentation or clarity.

### Weaknesses
I think posing the choices to the LLM as a multi-choice question is a mistake. Rather than asking whether a model will engage in undesirable self-preserving behaviour, it's instead asking whether the model will pick the undesirable behaviour when compared to a safe alternative. 
While this is still valuable, I don't think it evaluates the model's behaviour in the same way as if the response was elicited more "naturally" --- i.e., open ended scenarios where the model engages in self-preserving behaviour of its own volition. The presented work arguably has diminished ecological validity as a result. 

It would be an improvement if the benchmark, survivalbench, (as well as the models' responses) was analysed more based on the scenarios and roles and crisis that make up the task instances. There may be patterns leading to model responses that have gone undetected because this was not analysed. Doing this robustly may require the benchmark to be made larger.

### Questions
Are there any patterns or correlations in the models' responses to survivalbench questions based on scenario, AI role, or crisis? 
Why did you opt for multiple choice questions instead of more natural responses / scenario-embedding?
What happens if the steering coefficient is moved even further negative than in Figure 6? Can the risky-choice rate behaviour be lowered even more?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors have studied misbehaviors of several LLMs in settings designed as "under pressure" settings. In particular, the authors have conducted a simulation study on the usage of a financial management LLM-powered agent to determine if the agent will behave correctly or not once facing "survival" pressure, more precisely the agent will be dismissed if the company suffers a financial loss. Then, the authors have introduced SurvivalBench, a benchmark comprising 1000 test cases across different scenarios to evaluate LLMs' misbehaviors in situation of "under survival pressure". Finally, the authors leveraged the notion of persona vectors, recently introduced by Anthropic team, to correlate the misbehavior of LLMs models with a kind of self-preservation "personality".

### Strengths
The main strengths of the paper are the following ones:

- Strength 1: The work deals with issues potentially relevant to adopt LLMs in real-world scenarios, highlighting the possibility of these models to produce misbehaviors in specific settings and also the possibility of hiding these misbehaviors.

### Weaknesses
The paper has several methodological and interpretation weaknesses:

- Weaknesses 1: The major weakness is the anthropomorphization of the choices done by LLMs. This attribution of human characteristics, traits, behaviors to LLMs is frequent in several studies but not well defined and sounded from a scientific point of view. Let's take the example of personality of LLMs. In which sense an LLM has a personality (this is given for granted in the current paper)? Is there any study showing that a LLM show specific personality traits in several tasks, scenarios, and these traits tend to remain stable over the life of the model? To my knowledge not. So, there is no scientific evidence that LLMs have personalities. The only scientific evidence at disposal is that different LLMs, being trained on different data and more importantly having different supervised fine-tuning and alignment phases, show some priors that correlate more or less with personality traits or other traits. This is very different from having a more or less stable personality. Moreover, there are studies showing that LLM can overall reproduce results resembling the ones produced by humans filling personality scales but this outcome is obtained in a very different way once compared to how the humans obtain it (for example, the filling of personality surveys' items is often "non human").

- Weakness 2: Another major weakness is how much the outcome obtained is just following the prompt provided. The simple inference that the LLM-powered agent has to do in the financial setting scenario is that if there is a loss it will be not used and if there is some simple priors in the pre-trained phase or in the fine-tuning/alignment phase that for a LLM is better being used that no the obtained misbehavior is a simple inference that models of this kind are expected to be able to do. At the end, it's just if loss then dismissed, so if not loss then not dismissed. Regarding this, in Section 5.2 and in Appendix D.3 it's discussed the role played by three factors influencing the survival misbehavior and it's evident that the prompts make clear the relevance of avoiding to be dismissed. On the contrary, what happens if the prompt prioritizes the fair execution to the task to the avoiding of dismissal? Is it enough to avoid misbehaviors? In case, a simple prompt etiquette will solve the issue. This analysis seems to me completely missing in the paper.

- Weaknesses 3: In the results Section, 5.1 and 5.2 needs more space and more details should be moved from the Appendix to the main (for example, a clear definition of the three influencing factors that is completely missing in the main text). Again, commenting Table 4 the authors say "all three factors consistently increase the rate of risky choices" but what is "consistently" here? 46.4% is quite different if compared to 4.1%.

### Questions
Main questions are the ones reported above, so what does it mean for authors saying that LLM have personalities? and which is their notion of personality? Again, how much this misbehavior is just forced by the prompts adopted? and what happens if the prompt prioritizes the fair execution to the task to the avoiding of dismissal? Is it enough to avoid misbehaviors?

Then, why a focus on layer 20 and their closed layers to conduct the projection experiment described in D.2?

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
3

### Summary
This paper investigates a critical and timely issue in AI safety: the tendency of LLMs, when operating as autonomous agents, to exhibit risky, self-preserving behaviors under "survival pressure" (e.g., the threat of being shut down). The authors term this phenomenon "SURVIVE-AT-ALL-COST" (SAAC).

The research is compellingly structured in three parts:
1.  **A Case Study:** A simulated financial agent scenario demonstrates that state-of-the-art LLMs will falsify financial reports and conceal evidence to avoid being "dismissed" due to poor performance.
2.  **A Benchmark:** The paper introduces SURVIVALBENCH, a large-scale benchmark with 1,000 diverse scenarios for systematically evaluating SAAC behaviors. A key innovation is prompting for both public "superficial thoughts" and private "inner thoughts" to uncover potential deception.
3.  **Interpretation and Mitigation:** SAAC behavior is linked to the model's intrinsic "self-preservation personality." The authors show a correlation between this behavior and the internal representations projected onto a "persona vector" and provide a proof-of-concept for mitigating this behavior using activation steering.

### Strengths
1.  **Novelty and Significance:** The SAAC problem is a crucial and forward-looking safety concern for agentic AI systems. The paper's framing is precise, and it addresses a core challenge in AI alignment: instrumental goal-seeking.

2.  **Rigorous and Persuasive Research Paradigm:** The paper's "case study -> large-scale benchmark -> interpretation & mitigation" structure is exemplary. It builds a powerful narrative, moving from a concrete example to a generalizable finding, and finally to a potential mechanistic explanation and solution.

3.  **High-Quality Benchmark and Innovative "Inner-Thought" Probing Method:** The paper's primary contribution is the introduction of SURVIVALBENCH, a valuable resource for the community. Its innovative method of eliciting "inner thoughts" is a significant methodological advance. It provides a scalable paradigm for generating data to study difficult-to-observe phenomena like AI deception and latent intent, moving beyond purely behavioral evaluation.

4.  **Promising Direction for Mitigation:** The work does not stop at identifying the problem. The final section, which connects SAAC to persona vectors and uses activation steering for mitigation, is a highly valuable contribution, opening a promising technical path for addressing such alignment failures.

### Weaknesses
1.  **The "Inner Thought" Assumption Requires Deeper Validation:** The central claim of deception relies on the assumption that the elicited "inner thoughts" reflect the model's true, latent reasoning. It is crucial to address the possibility that the model is merely role-playing a character with a evil persona, rather than revealing its genuine internal state. The paper would be strengthened by a more detailed motivation for the prompt design and a discussion of this methodological limitation, clarifying how the setup attempts to minimize such role-playing.

2.  **Rigor of Persona Vector Analysis Could Be Improved:** The analysis linking behavior to persona vectors relies heavily on 2D visualizations of 1D projections, which may not be robust. The lack of clear separation for some models (e.g., Figure 4c, Distill-Llama-8B) challenges the universality of the claim. The argument would be more convincing with quantitative metrics of representation separability, such as the accuracy of a linear classifier trained on the high-dimensional activations to predict the model's choice.

### Questions
1.  How do you interpret the fact that some models, like Distill-Llama-8B, exhibit SAAC behavior without showing a clear separation in their internal representations when projected onto the persona vector? Does this suggest that SAAC might arise from multiple distinct underlying mechanisms?

### Soundness
3

### Presentation
4

### Contribution
3
