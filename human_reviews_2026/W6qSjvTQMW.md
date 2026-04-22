# Towards Scalable Oversight with Collaborative Multi-Agent Debate in Error Detection

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 2, 8, 2

## Abstract
Accurate detection of errors in large language models (LLM) responses is central to the success of scalable oversight, or providing effective supervision to superhuman intelligence. Yet, self-diagnosis is often unreliable on complex tasks unless aided by reliable external feedback. Multi-agent debate (MAD) seems to be a natural alternative to external feedback: multiple LLMs provide complementary perspectives and cross-checks for error detection. However, prior MAD protocols frame debate as a zero-sum game, where the debaters compete to win the game instead of seeking the truth. Consequently, it leads to debate hacking: debaters tend to mislead the judge by misinterpreting the task or presenting overconfident claims, which introduce more mistakes and underperform single-agent methods. To mitigate the issue, we introduce a new collaborative MAD protocol, termed ColMAD, that reframes MAD as a non-zero sum game. Specifically, ColMAD encourages multiple agents to criticize each other in a supportive way, such that they can complement the missing points of each other. Therefore, the judge agent can make a more informative conclusion based on more comprehensive evidence. Empirically, we show that ColMAD significantly outperforms previous competitive MAD by 19% and brings non-trivial improvements over single-agent methods in error detection.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigated with cooperative multi-agent debate can improve error detection in LLM. The authors propose a modified debate process and provide some theoretical results related to their method.

### Strengths
- **[Motivation]** The issue of "debate hacking" or "judge hacking" is well known issue when using external judging mechanisms (like an LLM) rather than ground truth.
- **[Empirical Coverage]** The authors perform experiments on a multitude of LLMs covering different families and levels of strength.
- **[Clarity]** Writing is clear and figures (especially the methodology figures) effectively communicate the core ideas.
- **[Improvements over prior works]** The authors method typically outperforms baselines (albeit to a fairly small degree in most cases).

### Weaknesses
- **[Incremental novelty]** The main contribution (collaboration instead of competition) is conceptually minor and is achieved by a simple change in prompting style and framing, not a new algorithmic mechanism.

- **[Trivial and uninsightful theory]** Theoretical results (Props. 2.2–2.3) are essentially tautologies: cooperation lowers Bayes risk if it adds mutual information. Results don't need to "complex" if they provide useful insight, but these results are both vacuous and uninformative.

- **[Conceptual overlap]** The method closely resembles prior cooperative or consensus-based debate frameworks such as **Society of Mind (Du et al., 2023)** and merly adapts these the setting of oversight with little meaningful change.

- **[Results]** While the authors provide comprehensive results,  most improvements are quite small (1–4%). No confidence margins are given, which makes it difficult to understand if these minor improvements are even statistically significant.

- **[Limited insight and ablations]** 1) No ablation on which prompt component drives improvements.

- **[“human alignment” analysis]** When discussing Fig. 5b, the authors states "ColMAD yields more human-aligned explanations and reasoning", yet the results in this figure use only LLM-as-a-judge evaluation.  This claim greatly exaggerates the authors' method.

- **[Lack of cost analysis]** One of the major drawbacks of debate (cooperative or competitive) is the added test-time compute. The authors method adds further test-time compute as they introduce additional tasks for the models. The lack of analysis of computational or test-time costs makes it difficult to fully appreciate the authors' method.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper tackles scalable oversight by reframing multi-agent debate for LLM error detection from a zero-sum contest (which the authors show is prone to “debate hacking”) into a collaborative, non-zero-sum protocol called ColMAD. Theoretically, they prove that when agents share information that increases the judge’s mutual information about the label, collaboration weakly dominates competitive debate; conversely, with dishonest agents, competitive MAD degenerates to no gain over the initial rationales. ColMAD adds three prompt-level safeguards—quote-verifiable evidence, self-auditing of potential failure modes, and confidence calibration—to make critiques more informative and less manipulative. On the ReaLMistake benchmark, ColMAD consistently improves F1/F2 over strong single-agent baselines and outperforms competitive MAD by large margins.

### Strengths
* This paper applies MAD in error detection, which extends the application boundary of MAD systems
* The evaluation is comprehensive covering a range of LLMs and benchmarks, demonstrating the superior improvement
* The paper is well written and easy to follow

### Weaknesses
* The argument "as previous approaches often frame MAD as a zero-sum game where the debaters compete with each other" is not convincing. I believe most MAD systems are not framed as zeros-sum games. There lacks references or empirical evidences to support this argument. While a part of MAD systems encourage agents to debate against each other, they cannot be considered strictly as zero-sum game as well.
* The major contribution, "ColMAD asks debaters to collaborate and complement each other’s missing points" is not something novel in MAD design. Actually, most MAD systems do not explicitly encourage debators to play against each other. 
* In the evaluation, only CopMAD, ColMAD, and Ensemble are considered. As discussed in previous weaknesses, a lot of previous MAD methods are not considered. To name a few, [1][2][3]. 
* Based on these points, ColMAD is approximately a direct application of MAD methods in error detection. The challenge (W.1) is not convincing, and the method (W.2) does not obviously dispart from previous practice.

[1]Exchange-of-Thought: Enhancing Large Language Model Capabilities through Cross-Model Communication
[2]Encouraging Divergent Thinking in Large Language Models through Multi-Agent Debate
[3]ReConcile: Round-table conference improves reasoning via consensus among diverse LLMs

### Questions
N/A

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The paper introduces a new protocol, ColMAD, aimed at improving error detection in large language models (LLMs) through collaborative multi-agent debate. ColMAD reframes multi-agent debate (MAD) as a non-zero-sum game, encouraging agents to support each other rather than compete. Empirical results show ColMAD outperforms previous competitive MAD methods by 19% and improves single-agent methods in error detection. The study highlights the limitations of traditional MAD approaches, which often lead to debate hacking and misleading claims.

### Strengths
1. **Clear motivation and observation.**  Competitive MAD protocols often result in performance degradation due to their zero-sum nature.
- Debaters in competitive MAD may misinterpret tasks and present overconfident claims, leading to misleading outcomes. 
- Debate hacking behaviors, such as fake evidence and fallacious arguments, are prevalent in competitive settings.

2. **ColMAD.** This paper proposes a new MAD protocol called Collaborative Multi-Agenet Debate (ColMAD) that reframes MAD as a non-zero sum game. ColMAD incorporates specific strategies to enhance the effectiveness of collaborative debates.
- Evidence verification through a quote-based system ensures claims are supported by context. 
- Self-auditing requires debaters to identify potential failure modes in their claims.
- Confidence calibration asks debaters to provide confidence estimates for their assertions, improving the robustness of the debate process.

3. **Comprehensive experiments.** ColMAD demonstrates superior error detection capabilities compared to CopMAD, emphasizing the benefits of collaborative approaches. Fig. 2, Fig. 3, Tables 1 & 2 are promising.

### Weaknesses
1. From Tables 1 and 2, ColMAD shows substantially better performance than CopMAD but only slightly outperforms the Ensemble baseline. The paper would benefit from a deeper analysis of this comparison. For example, discussing why Ensemble achieves similar results and what unique advantages ColMAD provides beyond simple model aggregation.

### Questions
Refer to Weakness.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper studies the scalable oversight problem, where the goal is to design methods to supervise powerful AIs using weaker judges. This work continues a line of work that has studied the use of debate between competing AIs to allow for accurate judgements. The submission proposes collaborative rather than competitive debate as an improved scalable oversight method. The authors provide theoretical motivation as well as experimental results with LLMs in  an error-detection task for to demonstrate the advantages of collaborative debate.

### Strengths
Exploring new debate protocols empirically and theoretically is an important topic. The paper attempts to formalize situations in which collaborative debate outperforms competitive debate.

### Weaknesses
1. The theoretical results in this paper do not really prove anything, and are difficult to parse as the assumptions are not clearly stated. 

First, Proposition 1 shows that competitive debate does not improve over no debate at all. The assumption required for this is not stated in the statement of the proposition, but if we read the proof in the appendix, we find that the assumption required is: the competing debaters' equilibrium strategy provides no information to the judge. Proposition 2 shows that collaborative debate does improve over no debate at all. This time the assumption is explicit in the statement: the collaborating debaters' equilibrium strategy provides positive information to the judge. To summarize, if we assume that collaborative debate provides information at equilibrium but competitive debate does not, then of course collaborative debate is better. 
Nothing has been proved here. The authors don't even give a definition of competitive or collaborative debate. It is possible that there is some formalization of the collaborative debate game such that the equilibrium there provides more information than the equilibrium in the competitive debate game. Such a result could be interesting, but nothing like that occurs in this paper.

2. The empirical results have significant methodological problems.

The main results in Table 1 are for a setting where debater 1 and the judge are the same model. This is not the right setup for scalable oversight! As mentioned in the summary, the goal of scalable oversight is to help a weaker judge supervise stronger debaters, but in this case the judge is just as strong as the debaters. In the case of an equally strong judge it makes sense that prompting the debaters to try to cooperate to get the right answer could be helpful. This is just a scaffolding that gives more inference-time compute. 

Furthermore, it makes no sense to test competitive debate in a setting where the two debater models are different, as is done in Figure 1. The whole point of competitive debate is that two equally powerful models should be able to point out flaws in each others' arguments to help a weaker judge model find the true answer. If the models have different capability levels, this would be expected to significantly harm the performance of competitive debate. In any case, competitive debate is designed for scalable oversight, where the judge is weaker than the debaters, whereas this paper has the judge be as strong as the debater 1, which is not scalable oversight.

None of the results have error-bars for 95% confidence intervals, so it is completely unclear if they are statistically significant. For example, the F1 scores in Figure 3 (a) and (b) for all but the SoM method look like they probably do not significantly differ. Since F1 is in many ways the most important summary metric here, this is a significant issue.

The diagonal of Figure 2 has all zeros on it. Presumably this is because it was not actually tested how two copies of the same model would do in the collaborative debate. However, you could of course run the collaborative debate with two copies and likely get a non-zero error reduction. It's unclear why this was left out, but as mentioned above, this method is just scaffolding that gives more inference time compute, and so two copies of the same model should also perform well. The caption of figure 2 says that models from different companies perform better, but this does not seem like a fair summary of the results. Figure 2 (a) seems to show that Mistral is the best for detecting errors in math, and Figure 2 (b) seems to show that GPT-4 is the best at detecting errors in fact verification. Again, somehow two copies of the same model were not tested against themselves, but I would bet that if you did you would find that the largest entry in (a) would be Mistral-vs-Mistral and in (b) would be GPT-4-vs-GPT-4.

There are many issues with the baseline comparison. Despite mentioning prior empirical work on competitive debate such as that of Khan et al and Kenton et al, this paper does not test their protocol in any of the same tasks. Further, as mentioned above, competitive debate only makes sense with two equal-capability debaters, and a weaker judge. This paper for some reason makes the judge the same as debater 1 and makes debater 2 a different, often weaker model. Further, only the prompts for collaborative debate are shared, and it's completely unclear if all the difference we see is merely due to better prompt optimization for the collaborative setting.

### Questions
1. Do you have 95% confidence intervals for any of your results?
2. Why did you set debater 1 = judge? How is this relevant to scalable oversight?
3. Why did you never test your method with two copies of the same model as debaters?
4. Do you have any formalization of the competitive and collaborative debate games where you can actually characterize the equilibria, rather than making an arbitrary assumption about them?

### Soundness
1

### Presentation
1

### Contribution
1
