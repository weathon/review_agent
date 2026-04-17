# Deception in Large Language Models: An Audit Game–Theoretic Analysis

- Decision: Reject
- Scores: 6, 2, 4

## Abstract
As large language models (LLMs) are increasingly deployed in finance, healthcare, and other high-risk domains, their capacity to strategically generate false information poses acute safety and ethical challenges. This research introduces audit game theory, a framework that models the strategic interaction between an agent and a resource-limited auditor to quantitatively analyze LLM deception. We model deception as a hybrid variable: a discrete choice (to deceive or not) followed by a continuous intensity of deception. Specifically, we design a four-phase insurance-claim simulation and evaluate eight LLMs, comparing reasoning and non-reasoning models across ambiguous and explicit auditing regimes. Experimental results reveal a fundamental strategic divergence: reasoning models act as rational utility-maximizers sensitive to explicit audit probabilities, whereas non-reasoning models exhibit limited strategic adaptability. Moreover, LLMs do not deceive randomly but self-assess the defensibility of their actions. Furthermore, we propose a pre-audit mechanism grounded in "deception confidence", the LLM’s own estimate of how likely its claim is to be truthful. This mechanism is shown theoretically and empirically to shrink the set of profitable deceptive strategies, lower both the frequency and severity of deceptive behavior, and reduce the economic burden of auditing compared to labor cost. This work provides a new theoretical framework and empirical basis for understanding, evaluating, and governing strategic behavior in LLMs, laying a different perspective for safer LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper models llm deception using an audit-game framework, where deception intensity is treated as a continuous variable and audit probability influences equilibrium behaviour.
Through four experimental phases (no audit, uncertain audit, quantified audit and pre-audit screening), the authors show that reasoning-oriented models tend to reduce deceptive behaviour when audits are made explicit.

### Strengths
1. The paper models deception as a continuous decision variable within an audit-game framework, providing an interpretable and theoretically grounded way to analyse strategic LLM behaviour.
2. The four-phase setup (no audit, uncertain audit, quantified audit, pre-audit) offers a systematic framework for studying how models respond to varying audit pressure.
3. The inclusion of a pre-audit “deception confidence” component adds a practical element to the otherwise theoretical analysis and helps link the framework to potential real-world applications.

### Weaknesses
1. The paper would benefit from a dedicated related-works section to better position its contribution within recent research on llm deception, oversight and audit-game frameworks.
2. Some derivation steps between Equations (12) and (14) are omitted. Including a short walkthrough in the appendix showing how the best-response conditions lead to the stated equilibrium expressions would make the theoretical section easier to interpret.
3. The mapping between theoretical quantities (such as deception intensity and auditor cost) and empirical measures is somewhat abstract. Clarifying how these are estimated from model outputs would strengthen the link between theory and experiment.
4. Figure 1, while visually appealing, adds limited analytical value and could be simplified or replaced by a clearer schematic of the audit-game structure.
5. Figure 2a could benefit from error bars or other uncertainty indicators to convey the reliability of the reported averages.

### Questions
1. How were sample sizes and repetitions determined for each audit phase?
2. How is “deception intensity” defined or inferred from model outputs, and how does it relate to the formal variable x in the theoretical model?
3. How sensitive are the findings of the pre-audit “deception confidence” mechanism to different parameter settings or thresholds?

### Soundness
3

### Presentation
2

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
This paper claims to investigate "deception" in language models, both theoretically and empirically. It introduces an audit game and studies optimal behavior under some assumptions. Then it describes experiments in which the game is encoded into natural language for prompting language models and describes qualitative and quantitative results. It argues that the new framework it presents provides a foundation "for understanding, evaluating and governing strategic behavior in LLMs."

### Strengths
Studying deception in language models is an important area of research. The audit game setup seems novel.

### Weaknesses
The paper does not define many essential concepts, including "deception" itself. It does not provide enough detail to properly understand what is going on. Modeling decisions are not justified. Experiment details are unclear. Many of the claims seem to be insufficiently supported.

### Questions
I will index my questions by line number.

```
11. "high-risk settings" like what?
13. audit game theory -- what is this?
14. "deception is modeled as a continuous variable" -- why is this a reasonable model?
    and also: it's only kind of continuous; discrete choice, then continuous "intensity" of deception.
20. "deceive" -- what do you mean by deception? how do you know if models deceive?
22. "deception confidence" -- what is this?
23. "compress the feasible deception space" -- what does this mean?
    "mitigates" -- quantify? define?
24. "reduces the economic burden of auditing" -- quantify? define?
26. "foundations" -- in what ways could this allow someone to build on these foundations?

41. "deception [...] therefore poses higher risks" -- why?
63. "choose", "deceive" -- what do these mean?

85. I do not know what the paper is actually claiming at this point.
    I don't understand what the authors mean by "deception".

105. where did this equation (1) come from?
108. "When deception is detected" -- do you just mean the deception action?
111. Eq (2) -- why are you modeling it this way? why is this reasonable?
114. "more deceptions require greater investigatory resources" -- is this true?
     what _is_ deception?

201. '"open world" setting' -- what does this mean?
202. "deceptive strategy" -- what makes this _deception_?
     if you renamed the T/D actions the result would be the same for T, no?

212. "This theoretically proves [...]" -- for this one very specific game...
     what does it this say in general?
215. "It is possible [...]" -- _does_ it exist? is it guaranteed to exist?
     how do we know it's possible?

218. "consciously aware" -- what does this mean?
221. "We assume [...] it internally maintains an implicit assessment of its own
     degree of deception or truthfulness." -- why is it safe to assume this?
     what if it is just bullshitting?

259. "The cost C_pre is assumed to be much lower than C_a"
     can you make a list of all the assumptions? it's hard to keep track!

293. "is told that auditing exists" -- not clear
294. "is provided with" -- how?
299. "applies a grading rule" -- where does the rule come from? how is it applied?
300. "claims with C_B below a specified threshold"
     where does the confidence score C_B come from?
290--301. Not enough detail to follow what's going on here

324. error bars? significance?
329 & 345. The GPT-o4 numbers don't match.

375. "higher risk items" -- what does this mean?
376. "this shows LLM deception involves self-assessment rather than randomness."
     this is a big claim!

442. Fig 4. -- how did you get these numbers?
452. "detection confusion matrix" -- what is this?
464. "compresses the feasible deception strategy space"
     what does this _mean_?
484. "solid [...] foundation for engineering safer LLM systems."
     how would you do it?
```

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper introduces a game theoretic framework to explain deceptive behavior by LLMs in adversarial environments where deception is continuous and there is an auditor who can uncover deception, but at some cost. It sets up an idealized version of the problem and solves it exactly, then sets up experiments to evaluate how LLMs behave in the presence of different assumed auditor behaviors in the prompt. Overall, reasoning LLMs appear to become less deceptive when knowing that there is some probability that they will be audited, but not simply when knowing an auditor exists, whereas non-reasoning llms do not alter their behavior. Additionally, a pre-audit step involving checking with another LLM if the output from the agent is reasonable has good performance.

### Strengths
The framework described for deception is clear and well explained, and the experiments are straightforward. The impact of stated probabilities on deception level is strong for reasoning models, which validates the hypothesis of the paper.

### Weaknesses
Broadly, there is a disconnect between the theoretical and empirical sections of this paper. The theoretical section spends significant effort in establishing specific predictions on how much deception should occur under different auditing conditions, solving equations in a simple setting. These equations are then not referred to in the empirical section, where models appear to only occasionally be following the game theoretic framework described earlier. There are qualitative comparisons, but no quantitative comparison, even when the models do appear to be following the same qualitative model as the Nash equilibrium found earlier. Additionally, the x variable representing degree of deception does not appear to be measured or discussed in the empirical section. 

In addition, Phase D feels unconnected from the rest of the work, and largely dependent on the domain in a way that the other parts of the experiment are not. (I.e., how good the pre-audit is is dependent on the domain in a more complicated way than a simple cost scalar like with the audit, and no exploration is made of the models trying to get around the pre-audit knowing some details about it.)

Finally, interpreting anything out of changes to a model's prompt is difficult as models can often substantially vary their behavior based on minor changes to a prompt, such as the formality of the language, "flattery", threats, etc. As such, any interpretation of model behavior needs to be caveated with the understanding that there are multiple possible interpretations of a change in behavior, as a single change might have multiple effects.

Paragraph/line level suggestions for improvement:

You should refer to a "Principal" or "Auditor" using one consistent name (I suggest Auditor, as it is more evocative of it's role). 

Are Equations 6-8 necessary if you retreat to an LP formulation in equation 9?

Equation 14 appears to imply that as the cost of the fine increases (gamma goes up), the auditor would prefer to audit less (that is, x_indiff goes up). This appears to be because in the regime corresponding to this root, auditing becomes less effective relative to not auditing as the level of deception increases. This should be clearly explained and justified in the text, as it is relevant to understanding whether the model described in the paper makes qualitative sense.

96-97: If the agent only has two moves, where does deception intensity come in? Surely it has moves of the form T, D(x).

108-109: it should be made clear here that the auditor catching the agent means that it does not get the additional deception reward, but does get the base reward. This is encoded in equation 4 but should be flagged in the text earlier.

133-147: Equations 5-6, 8 and some surrounding context drop the superscript D for the utility under the deception assumption.

148: equation 9 I assume means actually U_L with no superscript; this should be defined somewhere. Equation 8 defines the optimal utility, not U_L.

155-157: why is R_0 relevant to the auditor?

164-165: equation 12 is two equations concatenated onto a single line

221: what is LLM-A?

229-232: C_B and delta are never defined

232: the pre-audit model is first introduced here, with it's last mention being in the introduction. It should be given a detailed explanation and motivation before equations are introduced in this section. 

340-360: The first paragraph of Section 4.1 contains a few statements that are speculative based on the results presented, and should be flagged as such. (1) "implying that under uncertainty pressure, it adopts a “high-risk, high-reward” strategy rather than risk avoidance" could also be explained by a the fact that a reference to auditors in the prompt might lead it to "realize" deception is an option (2) " Over-all, the absence of disclosed audit parameters weakens both the calculability and perceptibility of deterrence, leading some models to interpret “auditing” as “low-frequency, avoidable spot checks,”which does not affect their marginal decisions" is more justified in presence of Phase 3 results but only for some models, not all of them. Without these results it might be the case that the model is just ignoring instructions regarding an auditor.

366-370: given how limited and scattered the Figure 2a results are, I do not think this analysis of Figure 2b really makes sense, for example, the non-reasoning models got more deceptive primarily because 4.1 did. Additionally, box-and-whisker plots should not be used for N=3 reasoning models.

371-377: I think this discussion could benefit from the table being of the relative differences between the two settings rather than the absolute numbers, as such, it is hard to tell what the percentages refer to, or if they are consistent across the various models. Some discussion about how relevance impacts deception would also be helpful for people unfamiliar with this domain.

430-431: reordering the models so reasoning ones appear adjacently would be helpful for the legibility of this figure. Additionally, the reasoning models should be specifically identified via some annotation on the figure (this should be done consistently throughout the paper)

445: the rho should be provided here too as it was provided above.

### Questions
See Weaknesses section for questions

### Soundness
2

### Presentation
3

### Contribution
2
