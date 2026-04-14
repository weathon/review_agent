# Exploring Prosocial Irrationality for LLM Agents: A Social Cognition View

- Decision: Accept (Poster)
- Scores: 6, 5, 6, 6

## Abstract
Large language models (LLMs) have been shown to face hallucination issues due to the data they trained on often containing human bias; whether this is reflected in the decision-making process of LLM agents remains under-explored. As LLM Agents are increasingly employed in intricate social environments, a pressing and natural question emerges: Can we utilize LLM Agents' systematic hallucinations to mirror human cognitive biases, thus exhibiting irrational social intelligence? In this paper, we probe the irrational behavior among contemporary LLM agents by melding practical social science experiments with theoretical insights. Specifically, we propose CogMir, an open-ended Multi-LLM Agents framework that utilizes hallucination properties to assess and enhance LLM Agents’ social intelligence through cognitive biases. Experimental results on CogMir subsets show that LLM Agents and humans exhibit high consistency in irrational and prosocial decision-making under uncertain conditions, underscoring the prosociality of LLM Agents as social entities and highlighting the significance of hallucination properties. Additionally, CogMir framework demonstrates its potential as a valuable platform for encouraging more research into the social intelligence of LLM Agents.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes to utilize the hallucinations of LLM-based agents to reflect and study human cognitive biases, which can further study irrational behaviors. It designs a new framework named CogMir, focusing on the study of social intelligence from hallucinations of LLMs. The results of extensive experiments show that CogMir can be aligned with real-world humans in terms of irrational and prosocial decision-making.

### Strengths
Pros:
- The idea of studying irrational behaviors from humans and LLM-based agents is very interesting. The pathway of reflecting human's irrational behaviors with LLM's hallucinations is fancy and interesting as well.
- The pipeline of CogMir is reasonable, which aligns the social simulations with real-world social experiments by outlining a standard workflow.
- The communication part models not only P2P settings but also the centralized (broadcast) settings, which is important for certain scenarios, such as social media trends.

### Weaknesses
Cons:
- How the LLM-based agents are designed? Are they just taking actions by prompting LLMs, rather than other modules like reasoning and memory? Could the authors elaborate on the architecture of the LLM agents? Specifically, are there any components beyond the base language model, such as reasoning modules or memory systems? A diagram or more detailed description of the agent architecture would be helpful.
- How do you deal with the factor of time? I think many social simulations are time-aware. Could the authors discuss how temporal aspects are handled in CogMir? For example, are there mechanisms to model the evolution of beliefs or behaviors over time in multi-turn interactions? Concrete examples of how time-dependent phenomena are captured would strengthen the paper.
- Could you please analyze the efficiency of CogMir? It would be helpful if the authors could provide an analysis of CogMir's computational efficiency. Specifically, how does the runtime scale with the number of agents or the complexity of scenarios? Are there any bottlenecks in the current implementation that could be addressed in future work?

### Questions
Please see the cons, and please point out if I have any misunderstandings.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
2

### Summary
The authors explore using hallucinations from LLMs to mirror human cognitive biases. They leverage the hallucination property of LLMs to assess and enhance LLM Agents' social intelligence through cognitive biases. Their proposed framework, CogMir, indicates that LLM Agents have pro-social behavior in irrational decision making.

### Strengths
- systematic construction of framework to assess and interpret social intelligence of agents
- The paper throughout is engaging, and methodical, self-contained for readers that are not familiar with the topic. Overall very well written.

### Weaknesses
- missing distributional metrics, since the temperature is set to 1 for all models, might be hard to reproduce the results. Also the conclusions drawn from these results might be invalid due to variance.
- The LLM Agent architectures/prompts used were not presented in the paper.

### Questions
- The prompts used for the agents are not presented. How are the prompts constructed and "tuned"?

### Soundness
3

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
This submission proposes CogMir, a novel framework for evaluating the "social intelligence" of Large Language Model (LLM) Agents by assessing their susceptibility to cognitive biases traditionally studied in social science.  The framework leverages the "hallucination" properties of LLMs, positing that these deviations from factual accuracy can mirror human cognitive biases, thus offering a window into the development of irrational social intelligence in agents. CogMir integrates established social science experiments, adapting them into human-LLM and multi-agent interaction scenarios, along with multiple evaluation methods (human, LLM, dataset-based, and using state-of-the-art discriminators). The authors conduct experiments on a subset of cognitive biases, demonstrating consistency between LLM agents and human responses in prosocial bias contexts, while highlighting differences in non-prosocial and certainty/uncertainty situations.

### Strengths
Novel Approach: The paper presents a unique and intriguing approach to evaluating LLM agent social intelligence by connecting it to the well-established field of human cognitive biases. This bridges a gap in current LLM Agent research that often focuses on black-box testing.

Well-Defined Framework: CogMir is described as a modular and extensible framework, encompassing various components for experimental settings, interaction modes, cognitive bias subsets, and evaluation metrics. This provides a structured and replicable methodology for future research in this area.

Transparency and Reproducibility: The paper provides details about the datasets used, evaluation methods employed, and experimental setup, enhancing the potential for reproducibility and allowing others to build upon this work.  The inclusion of an appendix with further details is also commendable.

### Weaknesses
Lack of a Robust Definition of Social Intelligence: The paper relies on a superficial association between cognitive biases and "social intelligence" without a clear theoretical grounding, hence the specifics of social intelligence being measured remain ambiguous. It's difficult to contextualize what the findings actually suggest without such grounding. It may be beneficial to have a more rigorous and operationalized definitions.

Insufficient Engagement with Social and Cognitive Theory: The paper's theoretical framework is underdeveloped.  While it mentions evolutionary psychology, it lacks a deep engagement with broader social and cognitive theories. Concepts like bounded rationality and cognitive bounds, which recognize inherent limitations on human cognitive processing and decision-making, are more relevant to understanding the role of biases in shaping social behavior. For example, we have been exploring computational models of bounded rationality in our field, it would make a lot of sense to compare notes with. Irrationality requires a clear definition of what it means to be rational, which isn't present in this submission. Incorporating such theoretical perspectives would provide a stronger foundation for interpreting the findings.

Limited Scope of Cognitive Biases and Social Scenarios: The paper's focus on a small subset of cognitive biases risks an incomplete picture of LLM agent social intelligence.  Similarly, the simulated social scenarios, while helpful for initial exploration, lack the complexity of real-world social interactions.

The promised connection between hallucination and cognitive biases are missing.  It seems like there are certain types of human cognitive biases that are particularly easily translatable with "hallucination". I wonder if it's worth narrowing down and finding a crisp connection. Also framing "hallucination" as a potential source of social intelligence needs careful consideration. The paper needs to explicitly address the potential negative impacts of inaccurate and biased outputs in real-world settings. The ethical implications of replicating human cognitive biases, especially harmful ones, in LLMs require a more thorough discussion.

Methodology Limitations:  While the paper strives for transparency and replicability, further validation of the evaluation metrics and exploration of alternative assessment methods would strengthen the robustness of the findings.


----
I sincerely apologize for engaging with the submission just now after the extended deadline. 

I thank the authors for clarifying some of the issues raised. While the authors have provided additional citations and their interpretations - I would have liked a stronger grounding in cognitive modeling and cognitive theory literature. Specifically not through authors' own interpretations, but through experiments or logical reduction to theories that are particularly important in the assumptions authors make. I remain concerned on this end.

I was able to resolve some misunderstandings I had by author responses. While I still think the claimed contributions need additional validation, I will remain reserved. There's greater value in replicating social experiments with LLM agents in a systematic manner, and it does have a path forward if the authors spend more space for explaining their agent design/architecture to make tighter connections to the results.

### Questions
How can the CogMir framework be used to understand more complex social interactions, such as negotiation, persuasion, and deception?

What are the implications of designing LLM Agents with cognitive biases? How does the performance of LLMs with different architectures and training datasets vary in their susceptibility to cognitive biases?

Can the insights from CogMir be used to improve LLM Agent training and design, leading to more socially intelligent and responsible AI systems? Does mirroring human biases in LLMs might simply perpetuate and amplify existing societal biases?

Authors have selected seven "state-of-the-art" models in line 289, yet the models aren't state of the art. Any further explanations?

---

I thank the authors for answering my questions. I'm still unsure if we readers can grasp how sensitive the results are when experimented with other newer models. Some validation of generalizability across models would be beneficial. Though I think it's a peripheral issue to the main constribution - I would suggest only to rephrase (perhaps delete?) mentions of "SOTA" to avoid confusions.

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
This paper introduces CogMir, a novel open-ended framework designed to evaluate and interpret social intelligence in Large Language Model (LLM) agents through cognitive biases. By leveraging the hallucination properties of LLMs, the authors explore whether these agents can mirror human-like prosocial behavior and irrational decision-making. The study categorizes cognitive biases into prosocial and non-prosocial subsets, showing that LLM agents demonstrate high consistency with human responses under prosocial cognitive biases such as the Ben Franklin effect and the Halo effect. The findings are positioned as an advancement in understanding LLMs' social intelligence, suggesting applications in more nuanced AI-human interaction scenarios.

### Strengths
- the concept of utilizing LLM hallucinations to mirror human cognitive biases is novel and intersects social sciences with AI in a meaningful way
- the experiments are meticulously designed, with a variety of cognitive bias subsets tested across state-of-the-art LLMs.
- the framework's modular and extensible design is well-articulated, allowing for potential adaptation and expansion in future research.
- the findings emphasize the potential for using hallucination as an adaptive feature in LLMs, paving the way for advancements in understanding and designing socially intelligent AI systems.

### Weaknesses
- some results, particularly those involving nuanced distinctions between models' biases, are complex and might benefit from clearer interpretations or summaries
- while the paper discusses prosocial biases effectively, it notes limitations in simulating non-verbal human behaviors. Expanding the scope to include multimodal interactions could improve the framework's applicability
- although CogMir is flexible, the reliance on simulated social scenarios might limit generalizability to real-world AI applications involving diverse human inputs

### Questions
How did the authors ensure that the constructed datasets used to test biases did not unintentionally favor certain models over others?
Are there plans to extend CogMir to include non-language-based behaviors or actions, as noted in the limitations?
What types of practical applications do the authors envision for LLMs exhibiting prosocial cognitive biases in real-world settings?

### Soundness
2

### Presentation
3

### Contribution
2
