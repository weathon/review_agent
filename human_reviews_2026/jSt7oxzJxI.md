# Benchmarking and Enhancing Rational Preference Utilization for Personalized Assistants: A Pragmatic View

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 6

## Abstract
Large language model (LLM)-powered assistants have recently integrated memory mechanisms that record user preferences, leading to more personalized and user-aligned responses.
However, the dual effects of personalization remain underexplored, and its adverse consequences are especially salient in real-world applications.
To address this gap, we propose Rational Personalization Acts, which reformulates memory utilization as a problem of pragmatic intent reasoning.
Building on this perspective, we develop **RPEval**, a benchmark comprising a personalized intent reasoning dataset and a multi-granularity evaluation protocol.
RPEval not only reveals the widespread phenomenon of irrational personalization in existing LLMs, but also, through a novel error pattern analysis, illustrates how irrational personalization can undermine user experience.
Finally, we introduce RP-Reasoner, which treats memory utilization as a pragmatic reasoning process, enabling the selective integration of personalized information. Experimental results demonstrate that our method significantly outperforms carefully designed baselines on \textsc{RPEval}, and resolves 80\% of the bad cases observed in a large-scale commercial personalized assistant, highlighting the potential of pragmatic reasoning to mitigate irrational personalization. Our benchmark is publicly available at \url{https://anonymous.4open.science/r/RPEval-E4B0}.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work addresses the question of how LLMs should incorporate user-specific memories, arguing that naively applying personalization can lead to poor outputs.  To study this problem, the authors introduce a synthetic benchmark of underspecified user queries labeled according to intent.  The dataset is constructed through multi-stage LLM generation, and evaluated primarily using an LLM judge.  To help the LLM address this query underspecification problem, the authors proposed a method, RP-Reasoner, a heuristic prompting scheme inspired by Rational Personalization Acts that combines a query likelihood term with an intent prior to decide the appropriate personalization level based on the inferred user intent.  Experiments suggest that this method outperforms simpler prompting baselines.

### Strengths
The paper is well-motivated and timely.  LLM personalization is an area of great interest to the community, and the authors rightly point out that existing approaches are largely naive, and better algorithms are needed for true contextual personalization.

### Weaknesses
My major concern is that the paper is organized around the idea of RPAs, but I am not sure what new insights are gained from taking this viewpoint.  The core claim, that personalized systems should not blindly apply stored preferences, but instead should infer intent and use context to decide whether/what to personalize, is a long-standing theme in recommender systems and LLM personalization.  For example, here is a survey paper on context-aware recommender systems from 2011 with over 3K citations: https://ojs.aaai.org/aimagazine/index.php/aimagazine/article/view/2364.  I think L2 personalization is always taken for granted as the goal.

Also, I find that the description of “memory utilization” does not really align with the task or proposed method; this is more about properly conditioning on a persona.  There are many interesting questions around how to use a constantly evolving memory store of past user interactions (from this and other users), but this work does not grapple with that.

I was excited when I read the first sentence of the motivation, that “This work centers on the duality of personalization, particularly the potential risks.”  However, I don’t think the issues addressed in this paper are truly risks, they’re just cases of bad personalization.  When I think of the risks of personalization, I think of addiction, sycophancy, and a range of other unhealthy feedback loops and phenomena.  It would have been interesting to see the paper focus on some of these real risks.

I am unconvinced that the LLM judge has been thoroughly validated.  Figure 3c is underexplained, and from what I can tell agreement levels are not that high.  What check or significance test is done here?  I looked at examples in the appendix, and I actually disagreed with the filter bubble example (why can’t the parent want the child to have a small serving of protein with their vegetables?), so I am worried about how well evaluation might function here.

I am also unconvinced that baselines are very competitive.  How was the CoT prompt optimized?  It seems like a carefully prompted reasoning model should be competitive at this.  Overall, it is hard to draw strong conclusions from the experiments; while the proposed method performs best, it is somewhat unsurprising given that it is bespoke to the unique benchmark created by the authors.  I am not sure of the broad applicability of this method.

### Questions
- What significance testing was done w.r.t. agreement between LLM judge and human annotions?
 - How was the CoT prompt optimized?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper tackles over-personalization in LLM assistants—when stored preferences get applied even when they shouldn't be. The authors propose RPA (a pragmatic framework), RPEval (a benchmark with 8K samples), and RP-Reasoner (a Bayesian reasoning method). Results show current LLMs struggle badly at deciding when to ignore preferences, with RP-Reasoner bringing ~35% improvement.

### Strengths
1. Important problem: Over-personalization is a real issue that hasn't gotten enough attention. The sleep music example in Figure 1 perfectly illustrates why this matters.
2. Well-designed benchmark: The preference inversion strategy is clever—generating queries first, then creating preferences that should/shouldn't apply. The error taxonomy (FB, RII, UPB, etc.) is also useful for understanding failure modes.
3. Strong empirical results: 80% resolution on real commercial system bad cases is impressive and shows practical value.
4. Interesting finding about model scale: The counterintuitive result that stronger models (GPT-5) can be worse at ignoring irrelevant preferences is worth highlighting.

### Weaknesses
1. Heavy GPT-4 dependency: The whole dataset comes from GPT-4.1 generation. This feels circular when you're then evaluating GPT-4.1/GPT-5 on it. How do you know the benchmark doesn't just measure "how well does model X mimic GPT-4's personalization decisions"? Would've been better to ground this in real user interaction data.
2. Subjectivity issue not fully resolved: The paper acknowledges when to apply preferences is subjective, but doesn't provide inter-annotator agreement scores. Also, only ~1K samples get human annotation—what about the other 7K?
3. Still far from human performance: On Single.ALL, best result is 0.77 vs 0.95 for humans. What's causing this gap? The paper doesn't dig into what RP-Reasoner still gets wrong.

### Questions
1. What's the inter-annotator agreement? How were disagreements handled?
2. Computational cost: how much slower/expensive is RP-Reasoner vs baselines?
3. The "Ignore" intent is hardest for models—why? Is there something fundamental about LLMs that makes them reluctant to discard context?
4. How would this work with actual user behavior data instead of synthetic scenarios?

### Soundness
3

### Presentation
2

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
**Summary:**

The paper addresses the problem of *over-personalization* in LLM-based assistants and seeks a *rational equilibrium*, conceptually a Pareto-optimal balance, between personalization and generalization. It frames personalization as a **multi-objective reasoning task**, proposing the **Rational Personalization Acts (RPA)** framework, the **RPEVAL** benchmark, and the **RP-Reasoner** model, which performs pragmatic inference to decide *when and how* to use memory.

### Strengths
### **Strengths**

* **Timely and Well-Motivated Problem Setting.**
  The paper addresses an important and underexplored challenge in LLM-based personalized assistants—how to balance personalization and generalization by reasoning about *when and how* to apply user memory. The framing of personalization as a multi-objective pragmatic reasoning problem is both novel and relevant to current trends in LLM alignment.

* **High-Quality Benchmark (RPEVAL).**
  The **RPEVAL** This dataset can serve as a reusable diagnostic tool for evaluating rational memory utilization across future LLMs.

* **Effective and Practical Solution (RP-REASONER).**
  The proposed **RP-REASONER** model demonstrates large and consistent gains over strong baselines—improving intent prediction accuracy by roughly **35%** and reducing error severity by **26%**. Moreover, the finding that it resolves **≈80% of bad cases** in a real commercial assistant underscores its potential practical value and real-world applicability.

### Weaknesses
**Weaknesses:**

1. **Oversimplification of intent categories**
   The three-way scheme {Ignore, Support, Dominate} is a substantial simplification of real user needs. In practice, intentions are often multi-faceted, evolving, and context-dependent (e.g., conflicting or partially overlapping preferences).

2. **Unclear baseline motivation (Vanilla, Reminder, CoT)**
   The paper does not clearly justify why only Vanilla, Reminder, and CoT are used as baselines. It would strengthen the work to explain why **more advanced training-free approaches** and **stronger prompting ensembles** (e.g., self-consistency, self-refine/verify) are omitted. Without this rationale, it remains unclear whether the reported gains derive from the proposed pragmatic reasoning itself or from a limited baseline set.

3. **Conceptual overlap with prior work**
   The problem formulation is closely related to [1], which likewise argues that LLMs should not naively trust historical personalization and must continually detect and adapt to shifting user preferences. Both are **training-free, inference-time** frameworks that dynamically correct misalignment between user preferences and model behavior. The paper should explicitly compare and position its contribution relative to [1].

Reference
[1] Unlearning Misalignment for Personalized LLM Adaptation via Instance-Response-Dependent Discrepancies (TMLR 2025).

### Questions
Question 1 (Baselines & Rationale)
Your baselines focus on prompting (Vanilla/Reminder/CoT). Why were advanced training-free approaches like [1] not included, and how would RP-Reasoner compare to stronger prompting ensembles (e.g., self-consistency [2], self-refine [3])?

Question 2 (Comparison & Positioning)
Rational Preference Utilization performs inference-time pragmatic reasoning in intent space to regulate memory usage. This is conceptually similar to [1], which performs training-free, inference-time discrepancy unlearning via probabilistic marginalization in response space.
Could you include a comparison with [1] and discuss where RP-Reasoner is preferable (e.g., interpretability, latency, robustness to stale or contradictory memories) and where [1] is stronger?

References

[1] Unlearning Misalignment for Personalized LLM Adaptation via Instance-Response-Dependent Discrepancies (TMLR 2025).

[2] Self-Consistency Improves Chain of Thought Reasoning in Language Models.

[3] Self-Refine: Iterative Refinement with Self-Feedback.

### Soundness
3

### Presentation
4

### Contribution
3
