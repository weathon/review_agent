# Can Past Experience Help LLMs Reason Faster?

- Decision: Reject
- Scores: 4, 8, 4, 2, 2

## Abstract
Allocating more compute to large language models (LLMs) reasoning has generally been demonstrated to improve their effectiveness, but also results in increased inference time. In contrast, humans can
perform tasks faster and better with increased experience and exposure. Hence, this paper aims to investigate
the question: Can LLMs also become faster at reasoning through recurrent exposure on relevant tasks, and if
so, how can it be achieved? To address these questions, we first formalize the problem setting of LLM reasoning
speedup systematically in the dimensions of task similarity and compute budget calculation. We then propose
SpeedupLLM, a theoretically guaranteed framework to implement and benchmark such reasoning speedup
behaviour based on adaptive compute allocation and memory mechanisms. We further conduct comprehensive
experiments to benchmark such behaviour across different reasoning tasks, question similarity levels, memory methods, and
reasoning methods. Results show that LLMs can generally reason faster with past experience, achieving up to a
56\% reduction in compute cost when equipped with appropriate memory and reasoning methods.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper aims to study how to speedup LLM reasoning through both test-time compute allocation and memory mechanism. The authors empirically investigated diffrent test-time scaling methods and memory mechanisms, showing that past experience (i.e., memory) could speed up reasoning and it is dependent on the test-time scaling method and task similarity.

### Strengths
- Considering the synergy between test-time scaling and memory mechanism is interesting.
- The studied test-time scaling methods and memory mechanisms are comprehensive.

### Weaknesses
- The presentation of this paper could be further improved to highlight its significance and strengthen its clarity. For example, it is hard to understand whether the authors try to propose a new algorithm or just analyze/study the speedup problem. In addtion, the combination of adaptive test-time allocation and memory mechanism is not well-justified. From the title itself, it seems the focus of this paper should be investigating the memory mechanism. It might be clearer if the authors could first show speedup LLM reasoning through test-time scaling is dependent on memory mechanism.
- The theorems are straightforward and not very meaningful. Especially, the assumptions (e.g., the probability that the best response exceeds the quality threshold is non-decreasing with t) in both theorem 1 and theorem 2 are unrealistic. If consider Best-of-N case with independent sampling, the probability remains the same, which would give a trivial result. Besides, the assumption in theorem 2 assumes that including more memory (e.g., in-context) is always good, but in reality, this is often violated, as the model will eventually suffer from longer context.
- The task similarity is not technically formulated well, and it is unclear how task similarity could be leveraged in the proposed framework, as the experimental discussions are only for different task similarity levels. I would expect that, given some task similarity information, the framework could optimize either the compute allocation or improve the memory mechanism.

### Questions
See weaknesses above.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper investigates the effect of repeated exposure to similar questions on reasoning speed in LLMs. They focus on adaptive compute-budget allocation methods, where the cost of reasoning is reduced by early stopping. Additionally, they look at several methods of incorporating previous questions in memory, such as SFT, in-context learning, and reflection. 

They propose a theoretical framework to unify several different types of adaptive test-time scaling methods and memory mechanisms. They show that if the memory mechanism does not decrease answer quality, then SpeedUpLLM achieves non-increasing budget while maintaining answer quality. Finally, they benchmark the method across tasks, question similarity, memory mechanisms, and test-time scaling mechanisms.

### Strengths
The paper explicitly outlines its contribution (p2). It is well-situated in prior work on test-time scaling and memory methods. When the theoretical framework is outlined, the authors specifically describe how various common test-time scaling methods can be made adaptive, which makes the description concrete. 

The experiments are extensive and show clear trends across the different axes that are evaluated. Further, the results are broken into several ‘findings’ sections, where the authors discuss the implications and significance of each observation in detail. I also appreciated how comparisons were made to results in human cognition.

### Weaknesses
I don’t see any significant weaknesses with this work.


My only suggestion is that Figures 1 and 3 are difficult to parse. Particularly with Figure 1, it is not easy to compare across the different variables by looking at it. Additional figures like Figure 5 for scaling types or question similarity would be useful.

### Questions
The theoretical analysis requires that memory ‘does not degrade model performance’. Do you think that methods like ‘Reflect’ didn’t scale as well in the long horizon for this reason, or is it because the reflections don’t have as much capacity to hold information?

I would expect the LMs to be completely accurate on the S1 type question, since the answer to the identical question is included in memory (I guess specifically for the in-context mechanism). Why is this not the case?

Typos:

Theorem 2 – …the probability [that] a satisfying answer appears…

### Soundness
4

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
3

### Summary
The paper studies whether LLMs can “reason faster with experience.” It proposes SPEEDUPLLM, combining (i) adaptive compute allocation (early stopping under a score threshold τ) and (ii) a memory mechanism (SFT or text memories such as in-context and reflections). Theoretical results (Thm. 1–2) claim that as answer quality improves with experience, expected compute decreases; and that expanding relevant memory does not hurt the probability of producing a satisfactory answer within the first k candidates. Experiments across four task types report up to 56% compute reduction with comparable or higher accuracy when questions are similar.

### Strengths
1. Timely problem and clean framing of “reasoning speedup” by similarity levels (S1–S4) and compute budget definitions per decoding paradigm. 

2. Unifies several test-time scaling methods under a simple early-stop view; clear to implement and evaluate. 


3. Broad empirical sweep (tasks, scaling methods, memory variants) with some actionable insights (episodic memories > reflections; stronger gains at higher similarity; negative correlation between compute and error). 

4. Public code link and sensible implementation details (models, scorer, temps).

### Weaknesses
Incremental significance – The main insight—that prior exposure and adaptive stopping can reduce compute—is intuitive and somewhat expected. The theoretical contribution would be more valuable if the assumptions were clearly formalized and the empirical validation extended to natural sequential data.

Theorem 1 clarity – The proof is difficult to follow. The assumption that cost(R) increases with |R| is never stated, nor is it clear whether |R⁽ᵗ⁾| is constant across t. These conditions are critical for the result and should be explicitly stated and justified.

Theorem 2 assumptions – The argument from expectation to higher probability is not rigorous. The theorem should be restated directly under a stochastic-dominance premise, with independence and DAG-monotonicity listed as explicit assumptions. The current proof is not convincing.

Modeling opacity – The theoretical framing (Eqs. 3–5) is overly compact and hard to interpret. It would help to provide small illustrative examples showing what cost, score, and adaptive mean in practice—e.g., that adaptive allocation simply finds a prefix of R⁽ᵗ⁾ whose score exceeds τ. Without such grounding, the modeling contribution feels abstract and under-explained.

τ selection and calibration – The method for choosing τ is unclear. Is it tuned per task, globally, or adaptively? Results should show sensitivity of both speedup and accuracy to τ and to the judge model used for scoring.

Memory degradation – The paper observes performance drops at low similarity but does not quantify when memory becomes harmful. Please report a retrieval-similarity threshold (e.g., embedding cosine or PRF) where memory is suppressed, and show an ablation.

Limited generalization – The evaluation is largely synthetic (similarity levels S1–S4). It remains unclear whether the findings hold on realistic, temporally ordered workloads such as near-duplicate QA streams under distribution drift.

### Questions
1. Theorem 1 details: i don't quite fully follow the proof. The assumption that cost(R) is increasing when |R| increases is never mentioned before. Also are you assuming |R^{t}| is a constant across t? Please state everything clearly. 

2. Theorem 2 assumptions: Can you restate Thm. 2 with the stochastic-dominance premise directly (rather than mean-score), and list independence/DAG-monotonicity as explicit assumptions in the theorem body? I am not following the proof either. 

3. Overall I think there is a value for modeling contribution in this paper. But it is too short for me to digest and appreciate. Can you provide examples/illustrations for exactly what cost, score, adaptive are etc. In particular, i feel the adaptive compute budget allocation is simply finding a prefix of R^{(t)} such that the score is large enough. The current formulations (3) - (5) make it overly complex.


4. τ selection & calibration: How is τ chosen across tasks/methods? Is it per-task tuned on a held-out stream? Show sensitivity of speedup/accuracy to τ and to the judge model. 

5. Memory harms: You note degradation at low similarity; can you quantify a retrieval-similarity threshold (e.g., via embedding cosine or PRF) below which memory is suppressed, and show that policy ablation? 

6. Generalization beyond synthetic similarity: Can you add a real, temporally ordered workload (e.g., near-duplicate QA logs) to validate speedup under distribution drift?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper investigates whether LLMs can reason faster after repeated exposure to similar tasks. The authors formalize this as a reasoning speedup problem, defined by decreasing compute budgets across question sequences with varying similarity. They propose SpeedupLLM, a unified framework integrating adaptive compute budget allocation and memory mechanisms, and provide theoretical guarantees showing non-increasing compute costs with non-decreasing answer quality. Experiments demonstrate that LLMs can achieve measurable reduction in compute cost with memory assistance.

### Strengths
* The paper introduces and formalizes the new concept of reasoning speedup by linking compute allocation with question similarity, providing clear operational definitions and measurable evaluation metrics.
* SpeedupLLM elegantly integrates adaptive compute allocation and memory mechanisms, supported by Theorems 1–2.
* The empirical study is comprehensive, which spans four common reasoning tasks (math, code, commonsense, logic), four scaling strategies (Best-of-N, DFS, Self-Refine, Long CoT), and multiple memory types.

### Weaknesses
Despite that the paper presents a novel concept of reasoning speedup, the presentation is a main weakness, as stated in the following:
* The motivation (“Can LLMs reason faster through past experience?”) is novel, but the introduction drifts between human analogy, test-time scaling, and memory mechanisms without a clear logical bridge.
* The methodology section is dense with equations and symbol-heavy formulations, but the conceptual flow isn’t intuitive. For example, Section 3.2 mixes adaptive compute allocation and memory mechanisms before giving readers a clear intuition for why these are the right components. The theoretical results (Theorems 1–2) are not well contextualized. Readers must infer the meaning of “non-increasing compute budget” in practical terms.
* Figures (especially Fig. 1–3) are visually complex and lack concise takeaways in captions.

Overall, a clearer structure separating motivation, intuition, and formal results would greatly improve readability and impact. If the authors would improve the presentation, I will consider raising my score. Besides the presentation, there are also some minor weaknesses:
* The question similarity levels are manually designed, potentially oversimplifying real-world semantic overlaps. Using automated metrics (e.g., embedding cosine similarity) would reduce subjectivity.
* The theoretical guarantee assumes non-degrading memory, which may not hold when question similarity is low (e.g., S4 cases).

### Questions
Since there is a mainstream using RL to imrove LLM reasoning, can the proposed framework integrate RL to autonomously optimize memory retention and compute budgets over time?

### Soundness
2

### Presentation
1

### Contribution
3

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper explores if LLMs can reason faster via past experience. LLMs need more compute for better reasoning but suffer longer inference time, and existing LLMs lack experience leverage and adaptive compute allocation. It formalizes the problem by defining question similarity (S1-S4) and computing budget, then proposes SPEEDUPLLM, a theoretically guaranteed framework using adaptive compute allocation and memory mechanisms (parametric like SFT, textual like in-context).  Experiments on multiple tasks with 4 test-time scaling methods and 5 memory methods show LLMs can reason faster, with up to 56% compute cost reduction. Higher question similarity and episodic memory boost speedup, and faster reasoning correlates with higher accuracy.

### Strengths
- The paper has a clear motivation and is well-written.
- It is novel that the paper analyzes reasoning efficiency from the perspective of similar data.

### Weaknesses
- In fact, under the condition of providing similar questions and their reference answers, the increase in the model's reasoning efficiency is reasonable and predictable. If identical questions and their standard answers are provided to the model, LLMs only need to restate the standard answers without any reasoning. Furthermore, we need additional and greater costs to retrieve similar questions, or even to synthesize similar questions and their standard answers. What do you think the practical utility of such evaluation and analysis is for actually improving reasoning efficiency?
- Following up on the previous question, I believe some additional analysis and exploration may be necessary. For example: 1) Similar Q&As are not obtained through synthesis, but retrieved from actual datasets (such as cross-datasets), which is more in line with real-world scenarios (though it seems the effect of improving reasoning efficiency is already weak for S4 Setting). 2) Reference answers are not provided when similar questions are given. After all, similar questions are easy to obtain (e.g., through synthesis), but reference answers are difficult to acquire (synthesis also requires significant costs).

### Questions
See Weaknesses

### Soundness
2

### Presentation
3

### Contribution
2
