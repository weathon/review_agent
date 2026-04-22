# The Disparate Impacts of Speculative Decoding

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
The practice of speculative decoding, whereby inference is probabilistically supported by a smaller, cheaper, `drafter' model, has become a standard technique for systematically reducing the decoding time of large language models. This paper conducts an analysis of speculative decoding through the lens of its potential disparate speed-up rates across tasks.  Crucially, the paper shows that speed-up gained from speculative decoding is not uniformly distributed across tasks, consistently diminishing for under-fit, and often underrepresented tasks.
To better understand this phenomenon, we derive an analysis to quantify this observed ``unfairness'' and draw attention to the factors that motivate such disparate speed-ups to emerge. Further, guided by these insights, the paper proposes a mitigation strategy designed to reduce speed-up disparities and validates the approach across several model pairs, revealing up to a 76.7\% improvement in our fairness metric.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper investigates fairness issues in speculative decoding, where the inference speed-up achieved varies substantially across populations such as languages. Underrepresented or low-resource settings tend to experience slower decoding, revealing a form of computational disparity. The authors formalize this inequality using a divergence-based metric and introduce a mitigation method called stochastic corrective drafter finetuning (s-CDF). s-CDF is a straightforward approach that minimizes discrepancies. Empirical results demonstrate that the method effectively reduces these disparities, leading to fairer and more balanced acceleration across tasks.

### Strengths
* Novel framing: Reinterprets an efficiency-focused technique through a fairness perspective, exposing a blind spot in how LLM serving practices can create uneven user experiences.
* Theoretical grounding: Offers a formal connection between divergence, acceptance rate, and speed-up, framing fairness within speculative decoding in a principled way.
* Practical mitigation strategies: Demonstrates multiple, progressively stronger methods to reduce disparities, from simple adjustments like temperature tuning or data balancing to the more targeted optimization via s-CDF.

### Weaknesses
* The analysis is tightly bound to the Speculative Decoding, particularly with the drafter–verifier setup, making it unclear how well the ideas extend to other efficiency methods beyond speculative decoding.
* Some findings feel quite obvious. For instance, that higher acceptance rates lead to faster decoding, or that language representation in pretraining influences task performance.
* The proposed fairness metric, based on cross-entropy variance, may not directly correspond to user-perceived latency or practical deployment fairness.

### Questions
* How does speed-up unfairness manifest in real-world latency, cost differences, or overall user experience?
* Could this framework be extended to other acceleration methods, such as early exiting or mixture-of-experts inference?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper points out that the speedup performance of speculative decoding is uneven across different tasks and languages, with slower speedups on tasks with low fit and insufficient representativeness (such as low-resource languages). The authors quantify this unfairness, revealing its correlation with the fit of the draft model, and propose a randomized correction fine-tuning strategy that improves the fairness metric by an average of 12%.

### Strengths
- The paper elucidates this from three points: (i) why speed-up is monotone in acceptance, (ii) how speed-up is consequently induced by drafter-verifier fitness, and (iii) how cross-entropy misfit provides an optimizable surrogate for speed-up unfairness with provable implications for acceleration. The structure is very clear.

- The paper's definition of unfairness is very clear and reasonable.

### Weaknesses
- The main experiment only compared English and Japanese, which is too limited in terms of language compatibility. It is recommended to conduct experiments using 3-4 languages.

- In the experiment, the small and large models were from the same series, such as Qwen2.5-0.5B and Qwen2.5-3B. There is a lack of comparison for cases where the small and large models are not from the same series.

### Questions
- This paper primarily focuses on inequalities in efficiency/time, but in practical use, users are more concerned with the effectiveness. Does inequality exist in terms of effectiveness?

- If the verifier is larger, such as Qwen3-32B, will the results be different, considering that the verifier's parameter size is likely larger in real-world applications?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper aims to explore the different amount of inference speed up realized by the speculative decoding on different tasks/languages. Towards this, the paper first empirically demonstrates the existence of such a disparate latency reduction across languages and identify the correlation between the poor latency reduction and final model accuracy. The paper then develops a theoretical framework to study this phenomenon (which the paper refers to as unfairness in speculative decoding) and connects the acceptance rate for draft tokens to per-task divergence between the drafter & verifier distributions as well as the task fitness of the drafter model. The paper propose a solution by minimizing the variance in the per-task divergences between the drafter & verifier model.

### Strengths
- The paper focuses on a rather unexplored yet important topic of disparate impacts of speculative decoding across different tasks/languages.
- The paper successfully demonstrates the existence of poorer latency reduction for rarer languages.
- The aims to develop a rigorous theoretical framework to understand and tackle the unfairness in latency reductions via speculative decoding across different tasks/languages.

### Weaknesses
- The novelty and significance of theoretical contributions in the paper appears to be somewhat limited. Theorem 1 looks like a straightforward corollary of the existing results about speculative decoding in the literature (e.g., see Leviathan et al.). Also, it's not clear why the authors presented the lower bound on $S_T$ in the form of $K_T$ or $D_T$. What is the downside of keeping the lower bound in the form of total variation distance, i.e., the first inequality in Eq. (5)? Similarly, Theorem 2 is just a triangle inequality. Could the authors clarify where the assumption $r_p < r_q$ is used in the proof of Theorem 2.
- The paper is missing important baselines. The paper has identified that rarer languages benefit less from speculative decoding. This would naturally suggest that one should train drafter model in such a manner that its performance is relatively more uniform across different tasks/languages. Admittedly, the authors have considered data balancing during the drafter training in Section 8.3. However, the exploration is not comprehensive enough. There is a vast literature on long-tail learning which proposed custom loss functions to ensure good model performance for both rare vs. popular subpopulations (tasks/languages). The authors should consider expanding their empirical section by studying the effect of some of the techniques from this literature. Furthermore, how would techniques like DistillSpec affect the unfairness is not studies and compared with the proposed s-CDF method.

### Questions
See weaknesses section above. The reviewer has a few additional questions:

- Lines 266-269 discuss dropping a terms from $\nabla\_{\theta}\mathcal{U}$. Did you empirical verify the utility of dropping this term in your submission?
- In Line 431, the authors mention ``We see on average, a 20\% reduction...a 12\% decrease in unfairness $\mathcal{U}$.`` Are these results documented (in the form of figures or tables) in the submission?
- Potential typo in Line 128 - `... **tb** section 6 and 8`.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Speculative decoding speeds up sampling of a large (verifier) LLM by having a smaller (drafter) model generate an output sequence, which is then verified by the verifier model. Key to the latency reduction is that the verification step over sequence prefixes can be done in parallel. This paper makes an observation that the speed-up gained is not uniform across all tasks. For instance, the speed-up achieved on Japanese language tasks is less compared to the speed-up on English. The paper makes a connection between the proportion of data from a task (e.g., a language) and the speed-up achieved from speculative decoding. Briefly and roughly, rare tasks (as measured by their presence in the training set) tend to yield less speed-up. The paper also proposes a gradient-based mitigation strategy (Sec 7).

### Strengths
While speculative decoding has gathered much attention as of late, to my knowledge, this work is one of the first that attempts to formally connect the occurrence probability of a task to the per-task speed-up from speculative decoding. The research direction is original. The paper is mathematically precisely written, and is easy to read overall. Sections are also organized appropriately. Connecting the task speed-up to the cross entropy between the two models on data of that task gives interesting insights (Theorem 1). Though the results only rely on basic inequalities, there is value from the precision in writing, and good explanation.

### Weaknesses
While the paper is very well written and the theoretical results are interesting, the paper misses the following important points.

1. The paper lacks baselines in experiments. I see only one baseline method which is “Data balancing” (L417). This is understandable since the paper tackles a new sub-area. However, I think a few “ablative” baselines would provide more insights.

2. The abstract and the formulation consider generic notion of tasks. However, experiments consider only one definition, which is that one task is one natural language. Investigating alternate notions of tasks would provide more insights.

More specific questions are given in Questions. Overall, I find the research direction and theoretical results promising.  It is just that the empirical results do not quite provide sufficient evidence.

### Questions
**Questions**:

**Q1**:  Are there other simple baseline methods that can be compared to? By baseline, I mean a method that attempts to provide more uniform speculative decoding speed-up to all tasks. Currently there is only one baseline (Data Balancing). Please correct me if I’m wrong.


**Q2**:  Are there other notions of tasks to show the generality of the theoretical results and the proposed unfairness mitigation approach? Currently, in experiments, one task is defined as a data subset that has text from one language. It would be interesting to consider tasks that have overlapping support. That is, a prefix $s \\in \\mathcal{V}^*$  (per L137, Sec 5) can belong to more than one task (with non-trivial probabilities). 


**Q3**:  In the paragraph after Theorem 2 at line 209, we have

> drafter fitness $(1- r_q) \\uparrow \\implies \\alpha_T \\uparrow$

Could you please elaborate on this implication? I don’t quite see why $(1- r_q)$ increasing would imply $\\alpha_T$ increasing. I may be missing something trivial.

**Q4**: Is it possible to have a frequent task $T$ (i.e., a significant portion in the training set) that has a large $D_T$ (i.e., large discrepancy between $q$ and $p$ on the task $T$)? Could you please give an example? In that case, what are the effects of the proposed procedure on the speed-up on task $T$?


**Q5**: Related question. What happens to the speed-up gained on task $T$ if it is rare in the training set of $q$ (drafter), and frequent in the training set of $p$ (verifier)? And vice versa? This question is not about the proposed mitigation strategy. The setting is that $q$ and $p$ are given (where $q$ may not have been trained to align with $p$). Does the non-uniformity of tasks in the training of $q$ (or $p$) have more negative effects on the speed-up gain? 


**Q6**:  I understand that the proposed mitigation strategy will trade-off speed-up gains across tasks. I don’t think this point is sufficiently elaborated. **This is an important point.** Take the example considered in the paper where there are two tasks: English (dominant) and Japanese (rare). After using the proposed procedure, what happens to the speed-up on the English task? Looking at the proposed gradient in Sec 7, L258, weighting across $m$ tasks is uniform $1/m$. For a dominant task whose base occurrence probability is above $1/m$, it will be weighted down. Is this correct?


**Q7**: Is it correct that the proposed fairness mitigation strategy does not change the overall quality (across all tasks)? This is because the procedure freezes $p$ (verifier), and only fine-tunes $q$, and speculative decoding guarantees that the sample generated will follow $p$, regardless of $q$. Is this correct? If so, it may be worth highlighting, say, in Sec 8 (experiments). It may be trivial to those working on speculative decoding but it is helpful for non-specialist readers.


I very much hope all the above questions can be addressed. 


----------

**Minor issues** (easily fixable):

* Duplicate references to Leviathan et al., 2023.

* L215: preforms -> perform

### Soundness
2

### Presentation
4

### Contribution
3
