# Are LLM Agents Behaviorally Coherent? Latent Profiles for Social Simulation

- Decision: Reject
- Scores: 2, 2, 2, 0

## Abstract
The impressive capabilities of Large Language Models (LLMs) have fueled the
notion that synthetic agents can serve as substitutes for real participants in human-
subject research. To evaluate this claim, prior research has largely focused on
whether LLM-generated survey responses align with those produced by human
respondents whom the LLMs are prompted to represent. In contrast, we address
a more fundamental question: Do agents maintain internal consistency, retaining
similar behaviors when examined under different experimental settings? To this
end, we develop a study designed to (a) reveal the agent’s internal state and (b)
examine agent behavior in a conversational setting. This design enables us to
explore a set of behavioral hypotheses to assess whether an agent’s conversational
behavior is consistent with what we would expect from its revealed internal state.
Our findings show significant internal inconsistencies in LLMs across model families
and at differing model sizes. Most importantly, we find that, although agents may
generate responses matching those of their human counterparts, they fail to be
internally consistent, representing a critical gap in their capabilities to accurately
substitute for real participants in human-subject research.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
In this paper, the authors attempt to test the "internal consistency" of the so-called LLM agent. To do this, the authors hand-selected a series of LLM agent profile combinations and selected a number of topics with varying degrees of controversy, had the LLM agents engage in conversations with each other in order to discuss these topics, and observed the performance of the LLM agents during the experiment as well as the responses of the LLM agents to the questions that were specifically designed to assess their "internal state" in order to summarize the conclusions of the experiment. The entire experiment ultimately gave the conclusion that the LLM agent could not replace human participants in human-subject research.

### Strengths
This paper is concerned with whether LLM agents can replace human participants in social simulations and tries to give an answer through a series of experimental designs.

### Weaknesses
# The Critical weakness
This paper does not give a clear definition of the LLM agent it discusses. Since LLM is designed as a system that complements text based on the context of the input, the behavioral performance of the LLM agent will depend entirely on the design of its prompts. Therefore, the conclusions given by all the experiments in this paper can only represent the situation of the LLM agent they are designed for and are not generalizable.

# Other More Intuitive weaknesses
1. This paper does not use the ICLR 2026 LaTeX template.
2. The main text does not contain any description of the technical points such as the design of the experimental framework, the design of the LLM agents, or the selection of topics. Please note that the appendices are not required reading and the text should be self contained.
3. The formatting of Table 1 is incorrect; the top horizontal line is missing.
4. Given the relatively short length of the LLM agent dialogues presented in this paper (API call fees should not be an issue), the decision to use small-sized LLMs for testing means that the evaluation conclusions cannot accurately represent the current state of LLM capability development.

### Questions
Please follow the weaknesses.

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper explores the notion of behavioral coherence in large language model-based agents, defined as the internal consistency between their latent attributes (for example, preferences and openness) and their observable conversational behavior. The authors design pairwise interaction experiments across nine topics to examine whether agents with similar preferences achieve higher agreement than those with opposing views. The paper reports that large language model agents often show surface-level agreement yet internal inconsistency across conditions, suggesting a lack of stable behavioral coherence.

### Strengths
* The topic of examining internal behavioral consistency in large language model agents is timely and relevant to the broader discussion of using language model agents in social simulation.

* The experimental setup is clearly described, and the paper is easy to follow.

* The attempt to move beyond external alignment toward internal consistency evaluation is interesting.

### Weaknesses
1. The definition of behavioral coherence is not rigorously derived from social science or cognitive theory. It is instead motivated by intuitive expectations such as “agents with similar preferences should agree more”. These expectations are plausible but remain heuristic rather than theoretically or empirically justified. Without a rigorous formulation, the concept of behavioral coherence risks being circular and untestable.

2. The conclusion that language model agents lack internal behavioral coherence is not well supported. The observed pattern of “surface agreement but internal contradiction” could have many alternative explanations, including the conversational format, prompt framing, or model uncertainty. Moreover, humans themselves often show context-dependent inconsistency, so it is unclear whether these results truly reflect a structural limitation of large language model agents.

3. The study does not include any human data or behavioral benchmark to anchor its conclusions. Without comparing model patterns to real human interactions, it is impossible to determine whether the observed inconsistencies are meaningful or simply reflect normal human variability. 

4. The methodology mainly depends on prompt-based simulation and descriptive statistical analysis. The contribution is therefore conceptual rather than technical, and the insights are mostly intuitive and qualitative.

### Questions
* How is “behavioral coherence” theoretically grounded beyond intuitive expectations, such as the idea that agents with similar preferences should agree more?

* Could the authors include or discuss a human baseline to support the findings?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper asks whether LLM agents behave consistently enough to truly stand in for humans in social research. Instead of just checking if their answers sound human, the authors test whether agents act in line with their own stated preferences and openness during conversations. They find that while LLMs often appear reasonable, they’re internally inconsistent: rarely disagreeing even when they should, favoring positive views, and being overly influenced by topic sensitivity. These issues show up across models and sizes, suggesting a broader limitation. The authors argue that true behavioral coherence, not just surface realism, is essential before treating LLMs as human substitutes.

### Strengths
•	The introduction is well written.

•	The overall flow of the paper is clear.

•	I appreciate the fact that the authors tried to back the claims by statistical tests, which is often missing in ML papers.

### Weaknesses
•	Throughout the paper, the authors seem to confuse “sentiment” with “stance”, which are orthogonal concepts. This is an NLP 101 issue. Briefly here:  a positive stance (e.g., “I support the mayor candidate”) can be expressed with either a positive sentiment (“I support him because I love his character”) or a negative sentiment (“I support him because I hate the current mayor”.), and same for negative stance. Based on the context, what the authors want to say is “stance” rather than “sentiment”. Using “sentiment” to refer to “stance” causes confusion.

•	Section 2. Missing literature. Previous studies have shown that demographic information is usually insufficient to align LLM’s response with humans - the value it provides is little to none, and that one should instead include latent belief to ensure alignment [1]. Given this, you want to justify why you only include demographic information.


•	Section 3.1. You should formally define what “U” stands for.

•	Section 3.2. You should provide measures when using LLM as a judge, e.g., validated through human annotation.


•	Section 4.1 “We establish the other end of the spectrum, the disagreement side, by assuming it to be the inverse of agreement as shown in Table 1.” -> You didn’t justify why this assumption makes sense. The assumption is critical in calculating the expected probability in Table 1, Figure 4, and throughout this section.

•	Account for model inherent bias? That may explain why “shared dislike consistently yields lower agreement”.


•	Figure 5 is hard to understand. The label on the right (0,1,2,3) should be better explained.

•	Finding #4 bears more interpretation and analysis. You should elaborate what roles the contentiousness level actually plays. The current writing only describes Figure 6 verbatim.


•	Section 4.2. You should elaborate how you control the openness level. 


•	Section 4.3 and Table 2. Based on the statistical test, the findings you listed in section 4.1 are mostly unfounded, except the two surface-level findings. In that case, I think the majority of the findings and conclusions are not justified by the statistical tests.

References


[1] Chuang, Y. S., Nirunwiroj, K., Studdiford, Z., Goyal, A., Frigo, V., Yang, S., ... & Rogers, T. (2024, November). Beyond Demographics: Aligning Role-playing LLM-based Agents Using Human Belief Networks. In Findings of the Association for Computational Linguistics: EMNLP 2024 (pp. 14010-14026).

### Questions
•	How do you decide the distribution over the demographic space? How sensitive is your result to this distribution?
•	How do you decide the “controversy level”?
•	Section 4.3. Table 2. Are you concerned about the power of the statistical tests being too low? Did you do a priori any power analysis?

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 4

### Rating
0

### Rating Number
0

### Confidence
3

### Summary
N/A

### Strengths
N/A

### Weaknesses
N/A

### Questions
N/A

### Soundness
1

### Presentation
1

### Contribution
1
