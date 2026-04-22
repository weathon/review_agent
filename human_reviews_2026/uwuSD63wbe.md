# The Potential of CoT for Reasoning: A Closer Look at Trace Dynamics

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 6, 6, 4

## Abstract
Chain-of-thought (CoT) prompting is a de-facto standard technique to elicit reasoning-like responses from large language models (LLMs), allowing them to spell out individual steps before giving a final answer. While the resemblance to human-like reasoning is undeniable, the driving forces underpinning the success of CoT reasoning still remain largely unclear. In this work, we perform an in-depth analysis of CoT traces originating from competition-level mathematics questions, with the aim of better understanding how, and which parts of CoT actually contribute to the final answer. To this end, we introduce the notion of a \textit{potential}, quantifying how much a given part of CoT increases the likelihood of a correct completion. Upon examination of reasoning traces through the lens of the potential, we identify surprising patterns including (1) its often strong non-monotonicity (due to reasoning tangents), (2) very sharp but sometimes tough to interpret spikes (reasoning insights and jumps) as well as (3) at times lucky guesses, where the model arrives at the correct answer without providing any relevant justifications before. While some of the behaviours of the potential are readily interpretable and align with human intuition (such as insights and tangents), others remain difficult to understand from a human perspective. To further quantify the reliance of LLMs on reasoning insights, we investigate the notion of CoT transferability, where we measure the potential of a weaker model under the partial CoT from another, stronger model. Indeed aligning with our previous results, we find that as little as 20% of partial CoT can ``unlock'' the performance of the weaker model on problems that were previously unsolvable for it, highlighting that a large part of the mechanics underpinning CoT are transferable.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a novel metric called "potential" to analyze the reasoning process in large language models using Chain-of-Thought (CoT) prompting. By examining how the potential evolves over the course of a CoT on competition-level math problems, the authors identify several distinct patterns: "reasoning insights" where the potential sharply increases, "reasoning tangents" where it drops, and "lucky guesses" where the potential remains low until the very end .

### Strengths
(1) Novel and Intuitive Methodology: The core contribution, the "potential" metric (Eq. 1), offers a principled and intuitive method for quantifying progress within a CoT. This provides a valuable tool for moving beyond simple final-answer accuracy to analyze the intermediate reasoning steps.

(2) Strong Qualitative Analysis: The paper excels in its qualitative analysis, providing clear and well-annotated examples that connect the behavior of the potential curve to specific segments of the model's generated text.

(3) Interesting Empirical Findings: The study uncovers compelling and non-obvious dynamics in CoT reasoning.

### Weaknesses
(1) Estimating the potential metric requires sampling a large number of completions ($N=128$) at every intermediate CoT step to obtain a stable estimate. The manuscript does not address the resulting computational overhead, which may pose a serious limitation for scalability and broader adoption.

(2) The quantitative analysis in Table 1 depends on fixed, manually chosen thresholds to categorize behaviors such as “insights” (potential increase > 40%), “tangents” (potential drop > 30%), and “guesses” (potential < 5% at the penultimate step). The absence of justification for these cutoffs makes the reported statistics appear somewhat subjective.

(3) The empirical study centers almost entirely on competition-level mathematics problems (AIME) and on the Qwen family of models. It remains unclear whether the observed potential trajectories and reasoning patterns would extend to other tasks—such as commonsense reasoning or code generation—or to other model architectures.

(4) Algorithm 1 employs a brute-force approach that randomly samples and evaluates candidate CoT segments to maximize potential. This process is computationally inefficient and appears intended as a proof of concept. However, the manuscript does not discuss its scalability or the resources required to produce the results in Figure 3.

### Questions
Please refer to weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper studies chain-of-thought (CoT) reasoning in large language models using a new measure called “potential,” which quantifies how each reasoning step contributes to getting the correct answer. The authors analyze competition-level math problems to reveal how CoT progress fluctuates — sometimes increasing sharply with key insights, other times dropping due to reasoning tangents or random guesses. They further explore how partial CoT from stronger models can boost weaker ones, suggesting reasoning transfer across architectures.

### Strengths
The paper addresses an important gap in understanding how CoT actually helps models reason, not just appear to. Introducing “potential” as a metric is novel and provides a quantitative way to evaluate reasoning progress. The experiments on AIME problems are well-motivated and the qualitative analysis of “reasoning tangents” and “insights” is compelling. The study also goes beyond introspection by testing CoT transferability, showing interesting empirical results that small parts of reasoning from stronger models can unlock weaker ones

### Weaknesses
The proposed concept of potential is not rigorously justified beyond empirical correlation, and its interpretation remains vague. The assumption that higher potential implies genuine reasoning progress is too strong, as sampling-based estimation might simply capture distributional artifacts. The analysis often reads descriptive rather than explanatory; the paper reports patterns (insights, jumps, tangents) without providing mechanisms or theoretical grounding. The heavy reliance on visual examples and qualitative claims weakens the argument, since the method’s reliability is unclear. The “transferability” results are interesting but confounded — providing partial CoT is effectively giving a hint, so improved performance does not prove reasoning generalization. There is also limited diversity in tasks and model families; focusing only on AIME math and Qwen variants narrows generality. The paper overstates human-analogy comparisons despite limited evidence. Overall, while the work is original and thought-provoking, it lacks methodological rigor and over-interprets its empirical patterns.

### Questions
see above

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
The authors endeavor the investigate which parts of the CoT are most "important" in the model's reasoning and eventual production of the answer. To do so, the authors define the notion of potential, which, given some partial CoT, is defined as the probability that the model will output the correct answer after generating the completion of the CoT. Thus, the authors can ask which parts of the CoT have the largest potential or the largest change in potential.

### Strengths
- The authors characterize a new way to study which parts of the CoT are most impactful/useful in the model's reasoning via a new metric they call "potential".
 - Some interesting observations are made that connect the changes in the potential with different parts/strategies in the CoT.
 - The paper is largely well-written.

### Weaknesses
- The implications and future applications of the potential metric are unclear (or they seem rather limited). Additional quantitative analysis may enable us to draw more general observations about the reasoning of LLMs. The conclusion is missing a discussion on future work, which could have alleviated this concern.
 - Experiments were focused on math competition examples, and it's unclear if similar observations would be drawn in other domains.

### Questions
The notion of potential is very closely related to the value function in actor-critic models in RL, but the paper doesn't really draw this connection. A vague connection is made at the end of Section 2, but the similarity between the two notions warrants further discussion and elaboration.

Algorithm 1 describes a "greedy" strategy for finding the optimal CoT. However, there may exist more "globally optimal" CoTs that find the correct answer faster than the CoT produced by Algorithm 1, while at the same time, do not increase the potential as quickly. But I suppose the probability of the CoT is an important consideration, as there exists the possibility of a "degenerate" CoT where the model guesses the correct answer immediately, which would result in an immediate jump in the potential at the very beginning of the CoT (due to lucky guessing). As such, the result of Algorithm 1 is sensitive to the choice of M, where in the limit as M goes to infinity, the algorithm would eventually recover this degenerate (lucky guess) CoT.

Some of the key observations were made in the qualitative analysis, such as the one in Figure 6, where the portion of the CoT that is more difficult for humans actually corresponds to a small increase in potential, whereas other portions that seem quite simple to humans are associated with larger jumps in potential. It is unclear to what extent these observations generalize to other examples without any accompanying quantitative evidence. How often does the behavior observed in Figures 7 and 8 occur?


More detailed comments and questions follow. While the paper is largely well-written, there are a small number of grammatical and stylistic errors. I include some of them from the first three sections in the list below. This list is not comprehensive, so I encourage the authors to carefully read through the paper and correct all such errors.

Line 28: A weaker what? Is this a typo?

Line 31: Grammatical error: there should be a complete clause following "that", such as "...highlighting that the grass is green." Here, note that "grass is green" can stand on its own as a complete sentence. However, "a large part of the mechanics underpinning reasoning transfer" is not a complete sentence/clause.

Line 36: "lead to" -> "led to"

There are a number of claims in the introduction that are stated without supporting citations. For example "[CoT] reasoning (Wei et al. 2023) has [led] to several breakthroughs in domains spanning mathematics"; and similarly for "...coding." There are several others.

Line 298: "Model size also seems to surpress [guessing behaviour] more, which is expected since larger models generally tend to perform better"  Wouldn't this only be expected if the model's final answers are faithful with respect to their CoT? For example, it is possible that larger models are better at finding the correct answer *latently* (in a fashion that is not verbalized in their CoT), in which case, they may be able to more accurately find the correct answer via the guessing tactic.

Line 306: Why is the pass@k metric prone to suffer from guessing behavior?

### Soundness
3

### Presentation
3

### Contribution
2
