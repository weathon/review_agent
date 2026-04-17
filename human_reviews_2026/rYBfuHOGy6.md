# Inferring Capabilities from Task Performance with Bayesian Triangulation

- Decision: Reject
- Scores: 4, 6, 2, 2

## Abstract
As machine learning models become more general, we need to characterise their capabilities in richer, more interpretable ways that move beyond aggregated statistics on static benchmarks. We describe a method to infer the cognitive profile of a system from diverse experimental data. To do so, we introduce  measurement layouts which model how task-instance features interact with system capabilities to explain performance. System capabilities can be estimated using Bayesian triangulation, inferring their value based on the performance of a model on tasks with different features. Our approach accurately recovers the cognitive profiles of hand-crafted behavioural agents, as well as estimating the cognitive profiles for deep reinforcement learning agents and human children in a virtual game environment. These cognitive profiles are significantly richer than aggregated benchmark statistics, summarising multiple distinct capabilities that explain behaviour, and are also significantly more predictive, accurately estimating performance on new, held-out tasks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper presents cognitive profiles that explain behavior in a more fine-grained manner than traditional aggregate benchmark statistics. The authors demonstrate that these profiles lead to improved prediction for performance on new, held-out tasks in synthetic agent experiments.

However, the method is difficult to scale and relies heavily on domain knowledge. While the construction of an effective measurement layout for a real-world task would be a significant contribution, the experiment with real data did not show strong improvements in prediction. Therefore, I recommend against acceptance.

### Strengths
The paper provides an intuitive introduction to measurement layouts. The problem is well-motivated, addressing an important challenge in understanding AI system capabilities in a more fine-grained and generalizable way.

For the synthetic agents, the method does appear to improve prediction over aggregate statistics (Figure 5), demonstrating the value of the approach in controlled settings.

### Weaknesses
The predictive evaluation results are not very strong or consistent when applying the measurement layout to real data (Figure 8). Given that measurement layouts require domain knowledge and careful effort to construct, the primary contribution should be based on how much they can improve prediction in practical, real-world settings.

The authors construct measurement layouts for a few tasks, but the scalability of this approach remains unclear. While they discuss some general principles in Section 6, there is no systematic procedure for constructing measurement layouts. The bottleneck appears to be domain expertise, which limits the method's practical deployment.

Robustness levels are mentioned as a core component of cognitive profiles in Section 2, but they don't seem to be mentioned later in Section 6 about how to construct measurement layouts. This inconsistency raises questions about the completeness of the framework.

### Questions
**Questions for clarification:**

1. (Line 106) "Bias values represent some other preferences or limitations that may impact performance in a less monotonic way." Can you elaborate on how biases are less monotonic than capability levels? In Section 6, they seem to be treated similarly.
2. What led to the choice of linking functions used? Are these chosen based on domain knowledge?
3. (Line 128) "A key conceptual difference between our measurement layouts and typical uses for HBNs is that our approach is trying to capture a hierarchical dependency relation on capabilities and demands, rather than encoding information on hyper-priors." Can you give a typical example use case for HBNs and a use case that measurement layouts would be advantageous for?

---

**Minor comments:**

- Line 19 refers to "cognitive profiles of hand-crafted behavioural agent" while line 104 defines "cognitive profile of an AI system" as a tuple of capability levels, bias, and robustness values. Switching between "system" and "agent" is confusing since "system" can be confused with the agent and the environment combined.
- (Lines 42-44) "We could now infer, by triangulation, that the system's failure is likely a failure of object permanence, a robust inferential method in formal epistemology (Heesen et al., 2019) and causal (Bayesian) reasoning." Please add a citation for triangulation in causal reasoning and consider splitting this sentence for clarity.
- "To achieve this, the team was split into Team A who designed the synthetic agents, and Team B who built the measurement layout without knowledge of the agents built by Team A." Clarify in the text that the team refers to the authors of the paper. While this information is mentioned in the appendix, it should be clear in the main text.
- Maintain consistent x-axis label formatting between Figures 5b, 6, and 9.
- In figure captions or the Qualitative Evaluation paragraph, define what the metrics are for each capability shown in the figures.

---

**Suggestions for improvement (not factored into score):**

1. It would be interesting to conduct some analysis on the evaluation instances in Figure 5a that had low predictive ability. Do these instances correspond to the "fraudster" agents mentioned in the previous paragraph?

### Soundness
3

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
4

### Summary
The paper describes Heirarchical Bayesian Networks, graphs connecting meta-features of task characterisation with the cognitive profile of a system, to predict observed performance of 3D agents in simulated and real environments. Over the course of three experiments of increasing complexity (simple navigation, simulated object permanence tasks, and real agents solving OP tasks), HBN models assign scores to latent capabilities. The resultant models gave better predictions than predictions based on the average performance on the test set. Additionally, the method successfully identifies deliberate "Achilles' heels" of specially-designed agents.

### Strengths
Originality
--
This is, to my knowledge, the most thorough application of HBNs to 3D RL agents.

Quality
--
Leans on strong and established software (PyMC, AAI) inspiring confidence in code correctness. The three experiments span a range of complexity, validating the work in simple situations before taking the work to address complex real-world data. The inclusion of human performance is an added bonus. The paper applies HBNs to a battery of environments, with many tasks, agents, and roll-outs.

Clarity
--
The paper is well laid out and language is overall clear. Figures simply explain measurement layouts and clearly visualise the improvement of Brier score compared to the chosen baseline.


Significance
---
The HBN approach descibed successfully identifies deliberate "Achilles' heels" of specially-designed agents.

### Weaknesses
One of the stated advantages of this approach is the possibility to introduce domain knowledge. However, the "bitter lesson" suggests that this approach could be outperformed by simpler approaches combined with scaled compute/data. Although this Bayesian approach is arguably more principled, taking expert domain knowledge into account, in my opinion it needs a comparison to a baseline of a simple exploratory / confirmatory GLM / PCA such as those mentioned in the Related Work.

Minor points: in my opinion, paper flow would be improved by signposting that Related Work is located towards the end of the paper (I was expecting it after the introduction, to establish relevance and to emphasise the distinctions between the current work and existing literature).

### Questions
Could the authors expand on the advantage that this Bayesian Triangulation method has over existing Bayesian frameworks such as those mentioned in the Related Work? In particular, could the authors say more about the claimed advantage over HiBayES, that their approach emphasises "capability-oriented inference from instance-level meta-features"? What is the claimed significance of this difference?


L104: Unclear what the justification is for the <C. B. R> tuple. Why this, not something else?

L130: What does it mean in practice to be "trying to capture a hierarchical dependency relation on capabilities and demands, rather than encoding information on hyper-priors"? Isn't this undermined by the predetermination of the Measurement Layout?

Fig 2C: Why did no agent pass more than 50% of the instances? Would you expect the relative performance of model and aggregate to continue past 50% for more capable models? Why or why not?

L184: What are the "random stationary actions" ?

L374: What does it mean to "use the meta-features of each task instance to ensure demands are objective properties of the task"?

In Appendix A, point 3: to what extent does the introduction of general linking functions derived from domain knowledge bias discovered latents towards researcher priors and away from novel discoveries? Would even exploratory GLMs be more appropriate in cases where the number of latent capabilties is unknown? At a fundamental level, doesn't the structure of the measurement layout introduce severe bias, completely determining the latents which can be measured and found?

Perhaps this is what is meant around L394, but I found the language hard to follow at that point.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes measurement layout as an alternative to aggregate statistics for evaluating ML models.

### Strengths
- The paper is mostly well-written and easy to follow.
- Measurement layouts provide an interesting means to evaluate ML models on static benchmarks (and the experiments demonstrate that).

### Weaknesses
- The major limitation of this work is that the measurement layout has to be manually designed by the humans.
- The population of agents used to fit the measurement layouts seems to be created in an ad-hoc manner.

### Questions
1. Can the measurements layouts be automatically generated, without human intervention?
2. Is there a systematic way to generate the population of agents used to fit these layouts?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This paper proposes measurement layouts, hierarchical Bayesian networks that infer cognitive profiles (capabilities, biases, robustness) of AI systems from performance patterns across diverse tasks. The core contribution is Bayesian triangulation, leveraging task-instance features that vary independently to disambiguate which capabilities drive success or failure. The work is technically sound and addresses a genuine problem in AI evaluation related to moving beyond aggregated benchmark statistics to interpretable, capability-level profiles. The three experiments demonstrate proof-of-concept with increasing complexity, and the approach shows promise in predicting performance on held-out tasks.

However, the paper oversells the generality of its framework without adequately addressing fundamental identifiability constraints. Most critically, the central claim about "triangulation" relies on an implicit assumption that rarely goes unchallenged: that you can always construct enough independent task dimensions to triangulate the capabilities of interest. This might not be feasible in many tasks,

### Strengths
1. Applying psychometric ideas to AI evaluation is creative and timely.

2. Three increasingly complex scenarios provide scope for this work.

### Weaknesses
1. This is your key insight, and it's a real problem. The paper invokes "triangulation" as though it's a general principle, but the method fundamentally requires that task dimensions are sufficiently independent and align with capability structure.
The core issue: In Experiment 1, you have 2 task demands (goal size, goal distance) and 2 capabilities (visual acuity, navigation). That's nearly square which is barely enough degrees of freedom. In Experiment 2, you have 12 task dimensions but 7 capabilities. Yet the paper never formally addresses: when does the system become underdetermined?

2,  If you have $k$ capabilities and $d$ task dimensions, when is the system identifiable? The paper has no theorem or even heuristic guidance. For instance, if navigation performance depends on "both" visual acuity and memory, but your task dimensions only vary goal distance and visibility, can you separate these effects?

3. The paper acknowledges (Appendix C.2) that some task dimensions were not included in the measurement layout because Team B didn't realize they existed (e.g., occluder RGB values). This is presented as a strength by showing robustness to model misspecification, but it actually undermines the triangulation premise. If you omit a task dimension that affects performance, you risk confounding it with capability estimates. The paper should quantify this risk.

4. The paper claims the approach is "agnostic to the task or type of system" (lines 84-87), yet constructing measurement layouts requires substantial expertise.

5. Fitting profiles for DRL agents and children is impressive, but the evaluation is thin. There is no greound truth. Unlike Experiment 2, you don't know the true capabilities of PPO or Dreamer agents. Showing that you can fit a model and make predictions doesn't validate that you've recovered anything meaningful.

6. The paper cites psychometric principles from IRT and SEM but doesn't formalize the connection. You invoke "hierarchical dependency relation on capabilities and demands" (line 128-129) but never define this formally. What's the mathematical structure of your HBN beyond the linking functions?

### Questions
1. Can you formalize the identifiability condition? Under what constraints on the number of capabilities $k$, task dimensions dd
$d$, and linking function structure is the posterior over cognitive profiles unique (up to label swapping)?

Why not validate measurement layouts against ground truth beyond Experiment 2? For real agents, can you compare inferred capabilities against interpretability methods (e.g., saliency maps, attention weights, or probing classifiers)?
How sensitive are results to linking function choice? Can you show that products vs. minimums vs. generalized means give consistent capability orderings?

### Soundness
3

### Presentation
2

### Contribution
2
