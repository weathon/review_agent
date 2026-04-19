# Large Language Models Assume People are More Rational than We Really are

- Decision: Accept (Poster)
- Scores: 8, 6, 6, 5

## Abstract
In order for AI systems to communicate effectively with people, they must understand how we make decisions. However, people's decisions are not always rational, so the implicit internal models of human decision-making in Large Language Models (LLMs) must account for this. Previous empirical evidence seems to suggest that these implicit models are accurate --- LLMs offer believable proxies of human behavior, acting how we expect humans would in everyday interactions. However, by comparing LLM behavior and predictions to a large dataset of human decisions, we find that this is actually not the case: when both simulating and predicting people's choices, a suite of cutting-edge LLMs (GPT-4o \& 4-Turbo, Llama-3-8B \& 70B, Claude 3 Opus) assume that people are more rational than we really are. Specifically, these models deviate from human behavior and align more closely with a classic model of rational choice --- expected value theory. Interestingly, people also tend to assume that other people are rational when interpreting their behavior. As a consequence, when we compare the inferences that LLMs and people draw from the decisions of others using another psychological dataset, we find that these inferences are highly correlated. Thus, the implicit decision-making models of LLMs appear to be aligned with the human expectation that other people will act rationally, rather than with how people actually act.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper compares the implicit decision-making process of LLMs with the human decision making process and finds that LLMs assume that people are more rational than they are, which is in line with what humans expect from other humans.

The paper uses two test suites: (1) Forward Modeling, which is a task in which the LLM needs to predict how likely people would select gamble A over gamble B, (2) Inverse Modeling, in which the LLM needs to predict people’s preferences from a list of choices. The authors compare the outputs of several LLMs with reported human answers to come to their conclusions.

The authors argue that the results have important implications for alignment, namely that alignment should be divided into two cases: alignment with human expectations, and alignment with human behavior.

### Strengths
* **Topic is important.** It is important to get insight in the decision process of LLMs, which has important consequences for how trustworthy we deem their outputs. This paper approaches this question from a CogSci angle.
* **Paper invites follow-up work.** The paper sets the stage for follow-up work on alignment. What it means to align well with humans, depends on whether one wants to align to human expectations, or human behavior, according to the authors.
* **Experiments are mostly solid.** The authors conduct experiments over a wide variety of popular LLMs, and compare the results. This gives insights in how the results generalize over different LLMs. Moreover, the experimental setup is strongly grounded in well-known experiments in Cognitive Science.

### Weaknesses
* **Unclear what the effect of the temperature setting is.** The authors briefly mention their choices for different temperature settings, but not in a lot of depth. Currently it is unclear how robust the findings are against different temperature settings (which would affect the outcomes). See questions below.
* **Unclear what the effect of different prompting strategies is.** The authors report the zero-shot prompt used, and share some of their considerations for designing the prompt. However, as LLMs can be sensitive to the actual prompt used, it would be good to understand how robust the findings are when different prompts are used, beyond the zero-shot vs. CoT. See questions below.

### Questions
Based on the weakness above:

* What does the CoT prompt look like? Only the zero-shot prompt is reported in the appendix.
* What is the effect of changing the temperature? Especially, how do the most popular temperature settings (when known) compare to temperature settings that are less common to use?
* What is the effect of changing the prompt? Did the authors try many different prompts to arrive at their results, or do they think the results are stable if one changes the prompt?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper investigates how LLMs perceive and predict human decision-making. The authors claim that LLMs (e.g., GPT-4 and Llama-3) tend to assume more rationality than people exhibit in reality. They adapt existing design of psychology experiments regarding human decisions, and compared human behavior with LLM predictions in two decision-making tasks: (1) predicting binary decision for gambling and (2) pairwise ranking the decisions regarding how much it tells about the human preference of an object. The experiments show that LLMs are more correlated with rational decisions than actual human behavior. The authors also claim this discrepancy between LLM rational assumptions and actual human behavior may have implications for alignment in AI systems, as well as for designing models that either simulate human behavior or align with human expectations.

### Strengths
- The authors explore a very interesting direction and implication in the current LLM alignment research. The current LLM alignment may actually limit the models’ ability to understand genuine human behavior.
- The authors link their work nicely to existing well-grounded cognitive science studies, which will be helpful to other readers who are not familiar with this area.

### Weaknesses
- While I also believe LLMs struggle to model irrationality, I’m not entirely convinced that the current experiment design fully supports this claim.
    - In the forward modeling section, the binary choice questions have an objectively optimal answer. Why should we conclude that the models are “assuming humans are more rational” simply because their answers are closer to this optimal? A simpler interpretation could be that the models are simply solving problems optimally, regardless of the prompt’s content.
    - The fact that the forward modeling part relies on binary choice questions amplified my concern. For instance, consider a set of binary math questions that are challenging for humans, where we ask the model to predict which answer a person might choose. If the model consistently picks the correct answer more often than humans, would it be valid to conclude that the model assumes humans are more rational than they actually are? Or is the model simply aiming to answer correctly?
- The forward modeling results in Table 1 & 2 indicate that models are more aligned with human choices than with maximum expected values in the zero-shot setting, but become more aligned with maximum expected values only when using CoT. Why, then, does this lead to the conclusion that “LLMs assume people are more rational than we really are”?

### Questions
- How do you measure the correlation for task 2 and 3 in the forward modeling task?
- Could there be circular inconsistencies in the pairwise ranking for the inverse modeling task? For instance, if the model ranks decision A > B, B > C, but then C > A, how is this resolved?
- Why not use “average human participant” for the prompts in Table 5 in the Appendix?
- Why did you decide to use “you” and refer this to human simulation? “What would an (average) human participant do?” sounds better?
- In Table 11 in the Appendix, where is Bag 2 in Choice 2? Is it referring to the Bag 2 from above?

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
3

### Summary
Mainly the paper prompts LLMs (Llama3, Claude Opus, GPT-4) to predict the human choices in two games (predicting gambling choices and inferring preferences from choices). They find that LLMs poorly predict human behavior in both the zero-shot and COT settings—in the latter, the error is explained well by the models predicting the human behavior to be more rational than it is.

### Strengths
- applies research from the psychology field
- good discussion of alignment

### Weaknesses
see questions

### Questions
- Why set the temperature to 0.7 for some experiments?
- If I'm reading correctly, the poor predictions in the zero-shot setting aren't explained well by the model incorrectly modeling humans as rational actors? What's your alternative explanation?

> We found that GPT-4 Turbo could not provide a valid output ranking 47 choices at once. Thus, for all LLMs, we obtained rankings by asking the LLM to perform pairwise comparisons across 47choose2 pairs of decisions. Pairwise outputs were limited to {stronger, weaker, tie}, and were aggregated across decisions to form a ranking. Ties were discouraged to capture small differences between decisions

Why even permit the "tie" answer?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper studies how advanced LLMs make predictions of human decisions, and reveal that frontier LMs assume that people are more rational than we really are. By using established datasets capturing human decisions, this study show that LLMs appear to be aligned with the human expectation that other people will act rationally (much like expected value theory), rather than with how people actually act.

### Strengths
- The application of psychometric data of human for analyzing LLMs behaviors reveals very interesting insights between the gap of LM vs. human decision-making.
- The study design and the analysis are rigorous.
- The insights of "While we have focused on using theories and paradigms from psychology to analyze LLMs, there may also be opportunities to use LLMs to refine existing theories about people." is thought-provoking.

### Weaknesses
- One might argue that these psychology tests used in this work appear to be quite simplistic to inform practical development of next generation LMs.

### Questions
- What are some other potential human decision-making models that can also be tested on LLMs, in addition to forward and inverse modeling?
- How are the insights gathered from this study inform the development of LMs applications, or inform better design of the next generation of LMs or other AIs?
- Would you imagine an ideal LMs in the future should align perfectly with humans on these psychology data? As we know that the conclusion of human data very much depend on who these people are. How do you imagine the different participant pool will impact the impact of the results of this study?

### Soundness
3

### Presentation
3

### Contribution
3
