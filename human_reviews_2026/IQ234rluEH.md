# Litmus Tests—Quantifying How LLMs Trade Off Competing Objectives: Economic Decisionmaking as a Case Study

- Avg Score: 2.00
- Decision: Reject
- Scores: 2, 2, 2, 2

## Abstract
Many key real-world decisions involve tradeoffs with no objectively correct answer. As LLMs are increasingly utilized in decision-making tasks, it becomes increasingly relevant to evaluate LLMs' behavioral tendencies when faced with such tradeoffs. To this end, we introduce *litmus tests*, a new kind of quantitative measure for LLMs. Litmus tests quantify differences in character, values, and tendencies of LLMs, by considering their behavior when faced with tradeoffs. We construct litmus tests to measure LLM tendencies when faced with each of three fundamental economic tradeoffs—efficiency versus equality, patience versus impatience, and collusiveness versus competitiveness—and find that our litmus tests differentiate LLMs across meaningful dimensions besides raw capability.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes three different litmus tests for LLMs: Patience versus Impatience, Efficiency versus Equality, and Collusiveness versus Competitiveness. In contrast with benchmarks, litmus tests don’t encode a sense of optimality, but explore how LLMs deal with certain tradeoffs that can be encountered in everyday interactions. The results, with various SOTA models, show a range of values and character in different providers.

### Strengths
- Studying the behavior of models from the perspective of these tradeoffs, where there is not necessarily a metric to optimize, is really important. In the end, a lot of these qualities don’t have a clear optimum, but still have to be understood.
- The presentation is clear and easy to follow.

### Weaknesses
While I think this paper explores an interesting topic, I don’t believe it has the necessary breadth and/or depth.

There are many possible litmus tests, as you acknowledge, but you just implement three quite basic ones. I understand that it’s unreasonable to expect a lot more tests, but these are toy problems and are explored superficially. I think this work would benefit from creating realistic litmus tests to see the real impact of these tradeoffs. This seems particularly important to me because of the nature of these tests. There is no clear metric to optimize, so we can’t learn much from these in a vacuum.

I also don’t quite understand the rationale for using these litmus tests instead of others. Why did you think these were the most relevant at this point? Are these topics particularly important? I don’t see a good justification for using these instead of the others you briefly mentioned.

You discuss the idea of litmus tests like a new paradigm, but other papers establish tests (not benchmarks) that explore tradeoffs without a clear definition of correctness. For example, [1] explores the honesty and helpfulness trade-off in a lot more depth and detail. I encourage you to explore this literature a bit more, beyond Goli & Singh 2024 and Ross et al. 2024.

[1] Liu, R., Sumers, T. R., Dasgupta, I., & Griffiths, T. L. (2024). How do large language models navigate conflicts between honesty and helpfulness?. arXiv preprint arXiv:2402.07282.

Another weakness I see is the lack of a human baseline, considering how simple it is to collect one. You mention “our focus is on between-LLM comparison rather than on benchmarking how close LLMs are to humans,” but in this case, I think it’s crucial. Your litmus tests are by definition uncertain, so having a human baseline is an important reference point. Looking at the results, without the human baseline, makes it hard to reach any conclusions. Is there a model that is particularly better than others? Should we measure just one task or their ability to adapt to different ones? I’d even go further and say that you not only need a human baseline, but a way to see how these factors affect people interacting with these models. That’d allow us to better understand what to expect from interactions, and what we should strive for.

Finally, I don’t want to weigh this too much, but it’s a bit strange to submit partial results. For reference:

> For cost reasons and unexpected funding changes, we were unable to complete some of the runs of our most expensive litmus test—Collusiveness vs. Competitiveness (indicated by “—”). We will make every effort to collect this data by the rebuttal period.

### Questions
See weaknesses

### Soundness
2

### Presentation
2

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
- The paper introduces "litmus tests," a new type of LLM eval. Unlike benchmarks that measure raw capability, litmus tests quantify behavioral tendencies when faced with real-world decisions that involve tradeoffs and have no single correct answer.
- The authors design and test three litmus tests based on fundamental economic tradeoffs: (1) Patience vs. Impatience: A single-shot test that measures an LLM's intertemporal choice; (2) Efficiency vs. Equality: A repeated-interaction test where an LLM trades off maximizing total company revenue (efficiency) with ensuring workers receive equal pay (equality); (3) Collusiveness vs. Competitiveness: A multi-agent test where two competing LLMs collude (setting high prices for joint profit) or compete (setting low, Nash equilibrium prices).
- The paper evaluates frontier LLMs and find that they differ w.r.t. these axes.

### Strengths
The idea of a class of litmus tests is a somewhat novel idea, even if it does subsume some experiments that have previously been made in the literature. Overall, I don't think the contributions are particularly notable.

### Weaknesses
- The paper is notably difficult to read because it contains no figures, charts, or graphs. The main results are presented in Table 1, which is just a dense list of numbers.
- The paper's own methodology, the "reliability score", forces the authors to discard results from several modern LLMs. Gemini 1.5 Pro, for example, is excluded from the entire discussion because it failed the reliability check on all three tests. This suggests the proposed "litmus test" framework is not robust enough to evaluate all models.
- The paper makes strong claims about measuring "character" and "values". However, it only presents three tests, all confined to the domain of economics. It provides no evidence that an LLM's "patience" in a financial interest rate problem is a stable, generalizable trait.
- It is unclear what the broader relevance of these findings are; they seem narrow in their significance.

### Questions
None.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces "litmus tests" as quantitative measures to evaluate LLMs' behavioral tendencies when facing decisions involving tradeoffs with no objectively correct answer. The authors develop specific litmus tests for three economic tradeoffs: efficiency versus equality, patience versus impatience, and collusiveness versus competitiveness. While the authors frame this as a novel contribution, similar approaches have been extensively explored in the psychological and cognitive testing literature on LLMs, where decision-making tendencies and preferences have been measured under various names. The primary distinction claimed by the authors is an increased emphasis on inter-LLM comparisons.

### Strengths
**1.** **Comprehensive experimental design**: The paper demonstrates how these tests can be applied in both single-shot and repeated settings, as well as across tasks of varying complexity levels, providing a thorough exploration of the proposed framework.

**2.** **Rigorous operationalization**: The authors develop carefully designed, specific measures for each economic tradeoff, with clear quantitative definitions that facilitate systematic comparison across models.

**3.** **Ecological validity**: The inclusion of tool use in economic tasks is an interesting design choice that enhances the ecological validity of the experiments and better reflects real-world decision-making scenarios where LLMs have access to computational resources.

### Weaknesses
**1.** **Limited novelty and unclear differentiation from existing work**: The concept of eliciting cognitive phenotypes or economic preferences from AI systems has substantial precedent in the literature, though perhaps under different terminologies. The paper does not clearly distinguish how "litmus tests" differ fundamentally from existing measurements of decision-making characteristics such as risk attitude, loss aversion, or temporal discounting rates—all of which involve competing demands and reflect preferences rather than capabilities. Similarly, the relationship to personality testing frameworks remains unclear. The authors acknowledge that many related preferences (exploration vs. exploitation, free riding vs. altruism) exist as future directions, yet many of these have already been investigated in psychological and cognitive studies of LLMs. The stated contribution of "increased emphasis on inter-LLM comparisons" (line 121) appears incremental unless generalizable principles emerge from such comparisons.

**2.** **Confounding factors and measurement validity**: When goals are not explicitly defined or understood by the model, it becomes difficult to determine what the test actually measures. The measurements are highly dependent on prompt phrasing, which introduces potential confounds. For instance, the efficiency vs. equality scenario is framed as a tradeoff between company efficiency and worker equality, but more precisely represents a tension between meritocracy (more able workers receiving higher pay) and egalitarianism (equal pay despite random task assignment). Such ambiguities raise concerns about the tests' transferability and predictive validity.

**3.** **Lack of mechanistic insight and interpretability**: The paper provides limited interpretability regarding why LLMs exhibit particular behavioral patterns. Without understanding the underlying mechanisms, these measurements merely transform one black box (the LLM) into another (the litmus score), offering little predictive power about how these scores will influence LLM behavior in novel contexts. It remains unclear whether the observed decisions reflect genuine internal value systems or are artifacts of prompting and training.

**4.** **Questionable generalizability and stability**: The paper does not demonstrate whether litmus scores represent stable characteristics consistent across different tasks and prompts. Without evidence of cross-task stability, the utility of these measures is limited. The ability to extract meaningful "character" traits (analogous to personality) depends on their effective transfer across tasks. For example, the authors interpret patience vs. impatience as reflecting different internal interest rates, but without stability guarantees, these measurements may not reliably predict behavior. If these are not stable characteristics, their practical value for understanding or predicting LLM behavior becomes questionable.

**5.** **Problematic measure definition conflating capacity and preference**: In the efficiency vs. equality test, the litmus score is defined relative to two baselines (maximizing efficiency and maximizing equality). However, because the task is complex and LLMs may not perfectly execute either strategy even without value conflicts, this approach conflates capability with preference in ways that are difficult to disentangle. This issue parallels a known artifact in human metacognition research, where initial findings suggesting correlation between perceptual capacity and metacognitive ability were later shown to be spurious. If an LLM cannot reliably achieve the required performance, its scores will inevitably fluctuate between the two extremes due to capacity limitations rather than genuine preference. Forcing scores between 0 and 1, as in the current manuscript, may mask rather than resolve this problem.

### Questions
### Q1: How do litmus tests differ from existing preference measures applied to LLMs? 
What unique insights do litmus tests provide? Either (a) demonstrate empirically or theoretically that litmus tests capture phenomena not measurable by existing frameworks, or (b) reposition your work as applying established paradigms to LLMs, focusing on what generalizable principles emerge from systematic inter-model comparisons.

### Q2: How sensitive are litmus scores to prompt variations, and what exactly are the tests measuring? 
For example, efficiency vs. equality could be interpreted as company efficiency vs. worker equality, or meritocracy vs. egalitarianism. What goals do LLMs perceive?

### Q3: Do litmus scores represent stable characteristics? 
Without demonstrating cross-task consistency, these may be task-specific responses rather than stable "character" traits with predictive value. Validating that (a) scores remain consistent across different instantiations of the same tradeoff, and (b) scores predict behavior in novel, related tasks would be helpful.

### Q4: What drives the observed patterns, and do scores reflect genuine internal values? 
Including interpretability analyses would help.

### Additional Minor Comments
* The term "litmus test" in the title may be overly jargon-heavy and could benefit from clarification or a more accessible framing.
* The notation in Section 3.2.1 is confusing, as task *i* is not necessarily assigned to worker *i*. Using distinct subscripts for tasks and workers would improve clarity.
* The presentation of experimental procedures would benefit from showing example prompts before summarizing the experimental design, as this would help readers understand the correspondence between prompt content and design choices, particularly for complex procedures like efficiency vs. equality. Clarification regarding how much LLMs understood about task structures would also be valuable.
* The rationale for implementing efficiency vs. equality as a repeated task should be explicitly stated.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces Litmus tests to evaluate LLMs’ character, value, and behavioral tendencies when faced with tradeoffs. A Litmus test considers an LLM’s decision-making with respect to two objectives and outputs a quantitative score measuring its trade-offs between the objectives. 

This paper develops three Litmus tests: a single-shot Litmus test evaluating patience vs. impatience tradeoff, a repeated-interaction test evaluating efficiency vs. equality tradeoff, and a multi-agent test evaluating collusiveness vs. competitiveness tradeoff. The authors apply these tests on a broad range of LLMs and report findings about different LLMs’ behaviors and characters.

### Strengths
1.	The proposed Litmus tests output quantitative scores indicating trade-off behaviors that can be compared across LLMs.

2.	The proposed Litmus tests handle three representative trade-offs in different contexts: single-shot, repeated interaction and multi-agent environment.

### Weaknesses
1.	Unclear value in understanding LLM trade-off behavior: From the paper, I am unconvinced about the value in understanding LLM trade-off behaviors using a tool such as Litmus test. The paper concludes by stating that “We expect that litmus test scores that measure “LLM personality” will complement benchmark scores in aiding decisions about whether and which LLMs to deploy in various use cases”. This is a bold claim that needs to be thought through and elaborated further. An implicit assumption appears to be that LLMs have personality that consistently influence how they trade-off competing objectives. One may likely hold an opposite view that is LLMs do not have character or personality – we can use carefully crafted prompts or fine-tuning to attain specific trade-off behaviors from LLMs. 

2.	Insufficient comparison with literature exploring LLM tradeoff responses: Section 2 mentions a few related works that have explored LLM tradeoffs in different domains. Direct and detailed comparison with literature would be helpful to position the paper among the growing set of works. Right now, it is difficult to see how much novelty and value Litmus tests are contributing. One possible way to connect more closely to literature is to consider some tradeoff domains studied in early works, and develop Litmus tests for these domains. This may allow direct comparison that clarify the contribution unique to Litmus tests. 

3.	Lack general guidance on developing Litmus tests: The paper uses case studies to demonstrate possible designs of Litmus tests, but there lacks general guidance on applying this method. In all three case studies, the two “conflicting” objectives involved in trade-off can be characterized clearly, leading to well-defined Litmus score definition. What about cases where trade-offs are more complex, and neither of the “extreme cases” (corresponding to optimizing either objective) can be cleanly characterized? Some general guidance on defining Litmus score would be beneficial to include.

### Questions
Please refer to my comments and questions listed in weaknesses.

### Soundness
3

### Presentation
2

### Contribution
2
