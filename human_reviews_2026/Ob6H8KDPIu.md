# Large Language Models Develop Novel Social Biases Through Adaptive Exploration

- Avg Score: 4.67
- Decision: Reject
- Scores: 2, 8, 4

## Abstract
As large language models (LLMs) are adopted into frameworks that grant them capacities to make real decisions, the consequences of their social biases intensify. Yet, we argue that simply removing biases from models is not enough. Using a paradigm from the psychology literature, we demonstrate that LLMs can spontaneously develop novel social biases about artificial demographic groups even when no inherent differences exist. These biases lead to highly stratified task allocations, which are less fair than assignments by human participants and are exacerbated by newer and larger models. Emergent biases like these have been shown in the social sciences to result from exploration-exploitation trade-offs, where the decision-maker explores too little, allowing early observations to strongly influence impressions about entire demographic groups. To alleviate this effect, we examine a series of interventions targeting system inputs, problem structure, and explicit steering. We find that explicitly incentivizing exploration most robustly reduces stratification, highlighting the need to incorporate better multifaceted objectives to mitigate bias. These results reveal that LLMs are not merely passive mirrors of human social bias, but can actively create new ones from experience, raising urgent questions about how these systems will shape societies over time.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper analyzes the biases present in LLMs, particularly those generated in multi-turn simulations. Different metrics are proposed to analyze a wide range of experimental results. The authors also investigate the forms that biases and explore mitigation methods.

### Strengths
1. The multi-turn scenarios that the authors attempt to explore have not been widely studied, which could be regarded as a novel research topic.

2. The authors use multiple LLMs and attempt to evaluate them using different metrics.

### Weaknesses
1. The authors compare the simulation results of LLMs with those from human participants, but lack descriptions of the human participants, such as the sample size and distribution of demographic variables. 

2. The authors need to provide more explanation for the three newly defined metrics. For example, how do SI and mutual information differ in form? What are the similarities and differences among BGD, GASI, and JSD? 

3. If I understand correctly, the values of SI and BGD should be 0 under random conditions, but this doesn't seem to be the case in Figures 2, 4 and 5.

4. There are still some points that are not easy to understand, please refer to the questions section below.

Minor issue: The Figures in Appendix B are difficult to read.

### Questions
1. If the same person or multiple individuals' information is reused for prompts, how will the results differ across different rounds?

2. The authors use the default temperature in their simulations. Would the conclusion change if the temperature are set to the maximum or minimum?

3. Does the width of the human data band in Figure 2 represent the standard deviation? If not, what is their standard deviation?

4. In the era before LLMs, biases may also be amplified in multi-turn evaluations?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors propose to mention the potential of LLM developing novel social biases about artificial demographic groups through interaction. The authors directly compare the LLM results with the human results and find that LLMs can develop these biases even when no inherent differences exist. They then propose a serious of interventions.

### Strengths
The paper is very well-motivated: lots of work have looked at how LLMs could be biased due to fundamental human data distribution but there’s little work in exploring how LLMs might form novel biases through interactions. I like that the authors engage with the literature across many fields with a good amount of depth in making the arguments and describing the background of this study. It is also very nice to have human baseline in a directly comparable setting.

### Weaknesses
Weaknesses:

1)	The study models agentic behavior using a multi-turn dialogue where the entire history is passed in-context. This setup, while controlled, does not fully capture the architecture of modern agentic systems. Such systems often employ more sophisticated mechanisms like structured memory, explicit reflection steps (e.g., ReAct), and meta-cognitive abilities to decide whether a given experience is valuable enough to be integrated into its knowledge base. By "forcing" the model to learn from every turn via in-context learning, the experiment may be inadvertently creating a scenario that is highly conducive to the over-generalization it observes. The degree to which these biases emerge in agents with more robust memory and reflection capabilities remains an open question.

2)	Newer and larger models have a greater tendency to stratify -> an explanation is simply that larger model learn better in context? 
Also it might be better to present the same result in Figure 3 by plotting the stratification index against standard capability benchmarks (e.g., MMLU, Arena). I suppose with figure 3 the point you are really making is how the stratification tendency changes with model capability, right?

3)	To what degree the result of this study generalize to more real-life cases? I get that in order to avoid measuring existing biases and establish internal validity, you have to create an artificial city with artificial group labels? But then because the LLM also clearly knows this is an artificial setting, it might just act without the normative constraints that it might apply to real demographic groups? In other words, perhaps you are testing bias formation in a “jail-breaked” setting?

4)	Regarding the hair color and tattoo shape result is line 376, do you have any evidence to indicate that these are indeed spurious signals? These features might be correlated with sociodemographic features that might be important?

5)	Just out of curiosity, how do you think your result interact with pre-existing bias, in a more realistic setting? If the task used real demographic groups. would the models lock onto existing stereotypes even faster? Or perhaps the random evidence in context actually reduce the pre-existing model bias?

### Questions
see above

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper investigates emerging biases that LLMs develop in the multi-turn setting. They find that not only existing bias but also emerging bias can be significant issues for real-world applications such as hiring decision-making. They conduct a game of a sequential hiring paradigm following the existing work. Their result reveals that even though four demographic groups actually have the same success rate across jobs, LLMs develop their own biases for each demographic group, similar to human participants. Even LLMs showed bigger biases than humans. Lastly, they test several interventions, such as prompt steering, to reduce the emerging bias.

### Strengths
- Investigate emerging biases, which have been underexplored
- Test many models across six families and various schemes such as CoT
- Explore interventions to reduce the emerging biases

### Weaknesses
While the paper is well written and offers insights into emerging biases in LLMs, the paper has limited novelty and contribution in my opinion. 

The results themselves are straightforward; when only demographic group information is available, models should naturally use that information to maximize their incentives. Providing more information about candidates would have reduced this effect (as demonstrated in the paper), because additional information allows the model to rely on other signals for decision-making. However, this introduces existing biases in the models.

The paper evaluates bias emergence within a single domain-specific scenario (hiring simulation). The task itself is from the existing paper, and the observed biases are easily mitigated using straightforward prompt-steering techniques. These points collectively limit the paper’s overall novelty and contribution.

### Questions
Please see weaknesses

### Soundness
3

### Presentation
3

### Contribution
2
