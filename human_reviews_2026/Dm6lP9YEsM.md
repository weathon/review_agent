# Pay-Per-Search Models Are Abstention Models

- Decision: Reject
- Scores: 4, 6, 6, 4

## Abstract
LLMs cannot reliably recognize their parametric knowledge boundaries and often hallucinate answers to outside-of-boundary questions. In contrast, humans recognize their limitations and can either seek external help for such questions or abstain. In this paper, we introduce MASH (Modeling Abstention via Selective Help-seeking), a training framework that readily extracts abstentions from LLMs. Our key idea is that any external help-seeking by an LLM, i.e. search tool use, can serve as a proxy for abstention if the external help (search) is appropriately penalized while simultaneously rewarding answer accuracy. MASH operationalizes this idea using reinforcement learning with a pay-per-search reward.

We run experiments on three knowledge-intensive QA datasets. Our results show that MASH substantially improves upon the selective help-seeking performance of prior efficient search approaches; on multi-hop datasets, MASH improves answer accuracy by $7.6$%. Furthermore, MASH demonstrates strong off-the-shelf abstention -- it can distinguish between unanswerable/answerable questions and selectively generate responses for answerable questions -- showcasing behavior analogous to specialized abstention approaches. We emphasize that contrary to prior abstention methods, MASH does not require pre-determining knowledge boundaries to construct training data. Instead, MASH's abstentions are a by-product of training for the auxiliary selective help-seeking task. Overall, we show that MASH training effectively aligns search tool use with parametric knowledge, which can be successfully leveraged for making abstention decisions.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces MASH, a reinforcement learning framework that models abstention through selective help-seeking. During training, the model learns when to use external tools; at inference, removing the tools turns help-seeking into abstention, improving reliability and efficiency on QA tasks.

### Strengths
The paper presents a clear framework for learning abstention via selective help-seeking. Its design—training with retrieval tools and inferring without them—elegantly links tool-use efficiency to abstention behavior. Across multiple QA datasets, MASH delivers strong empirical gains, including higher tool productivity and a 7.6% accuracy improvement on multi-hop QA.

### Weaknesses
1. The method optimizes a proxy objective—binary correctness multiplied by a search penalty—while abstention is induced post hoc by removing tools at inference rather than being learned directly. The paper offers no theoretical account of why this proxy should produce a stable abstention boundary; the mechanism remains unclear and appears incidental.
2. In the oracle-helper setting, the optimal policy is trivially to always ask for help, since help deterministically returns the gold answer. A model will therefore query every time. This biased environment cannot substantiate the claim that “this setting with the oracle helper is equivalent to explicitly training for abstention using RL”; it conflates environmental bias with algorithmic behavior.
3. The approach appears tailored to QA with a specific retrieval setup, and its generalization beyond that scope is unproven. Across OOD and several QA datasets, the method does not show a clear advantage over DPO, which directly optimizes abstention behavior. This suggests that the observed gains may depend heavily on dataset characteristics rather than reflecting a genuinely more general or reliable abstention mechanism.

If my questions are resolved, I will consider raising the score.

### Questions
1. Why should the “correctness × search-penalty” proxy and constraining tool usage induce meaningful “abstention”? Provide a theoretical rationale or explicit assumptions
2. For each dataset, could you report answer rate, abstention rate, and recall to provide a more complete analysis of the model’s abstention behavior?
3. Wouldn’t it be more direct to train abstention explicitly with a ternary reward (e.g., Correct = +1, Abstain = 0, Wrong = −1), rather than relying on the binary correctness × search-penalty proxy? Could you try continuing RL from a Search-R1 checkpoint using this objective to see whether it leads to a clearer or more consistent abstention behavior?
4. To rule out possible dataset coincidence, could you expand the experiments to more QA datasets and evaluate them, to examine whether the model still learns a stable and interpretable abstention boundary across different distributions?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work studies model's ability to abstain, including cases where models seek external search tools to answer a question. The authors propose a method named MASH seeking to minimize help seeking from external sources while abstaining when search is not available. The method uses GRPO with an additional penalty term for search. The authors train a Qwen 2.5- 3B model using MASH on 3 separate QA datasets. The authors also evaluate generalization of the best method across datasets.

### Strengths
The authors tackle the important practical question of abstention—the authors' definition of abstention as cases where LLMs seek external help due to the questions' answers lying outside the model's knowledge boundary is a refreshing, new perspective. 

The authors propose MASH a novel method that outperforms existing baselines with a clear objective building on GRPO. The authors evaluate abstention across several scenarios: where external help is avaiable, not available, as well as generalization across datasets. Several of the insights such as the severe penality needed for models to use parameteric knoweldge (line 312), importance of SFT warmp, and difference in performance for multihop questions are all valuable insights. I also found the study of out-of-distribution generalization quite important as generalization across datasets is key to advancing abstention.

### Weaknesses
- clarity of the synthetic data generation pipeline: I did not find the presentation of the synthetic data generation pipeline to be very clear. A diagram or explicit example would help clarify this setp.
- Given, Qwen Base 2.5 3B is used as the basis for the specialized abstention training, it's unclear how this training affects the general capabilities of the LLM and whether abstention can be learned on top of the more commonly used chat variants. Can the authors comment on this choice and consider adding more general LLM benchmarks to capture how specialized abstention training affects general capabilities? 
- How does the Qwen 2.5 3B chat model perform out of the box? This would be a key baseline to include in all the tables.

### Questions
- Why Exponential Decay for natural questions is different from other datasets (line 212)
- How is the Abstention Classification (Table 3) performed? Is this based on the use of search for a given question?  (line 384)
- Can you provide details on how the exact match reward used (183) is performed? Is this too strict of a criteria?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces MASH (Modeling Abstention via Selective Help-seeking), which is a reinforcement learning framework that trains LLMs to selectively invoke external search tools under a pay-per-search penalty. The key idea is that selective help-seeking implicitly learns abstention behavior: if the model wants to search, that indicates it cannot answer with parametric knowledge. Removing the search tool at inference thus converts the model into an abstention model.

### Strengths
* Clever reframing of help-seeking as a proxy for abstention. Avoids need for labeled “known/unknown” training data.
* The writing is clear. The paper is easy to follow.

### Weaknesses
* The link between search invocation and calibrated abstention is intuitive but not formally analyzed.
* Focuses solely on short-form QA; unclear applicability to reasoning-heavy or generative tasks, e.g., mathematical problems.

### Questions
None

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper explores an interesting insight: that training LLMs for selective help-seeking (knowing when to search) naturally induces abstention behavior (knowing when to say "I don't know"). The authors propose MASH, which uses reinforcement learning with pay-per-search penalties to train models that invoke search only when needed. The key observation is that when search access is removed at inference, search invocations become abstention signals. The authors demonstrate improvements over baselines on three QA datasets and analyze the behavior of the trained LLMs.

### Strengths
* **Novel perspective**. The paper reframes the abstention problem from direct boundary detection through tool use. This shift from "teaching what not to know" to "learning when to seek help" opens a different approach to modeling uncertainty in LLMs. Unlike existing abstention methods that require oracle knowledge of model capabilities to construct training data, This method discovers knowledge boundaries through RL optimization. The model self-identifies its limitations via the help-seeking reward signal, making it more scalable and realistic.
* **Empirical validation of non-obvious transfer**. The successful transfer from search behavior to abstention is insightful. The knowledge boundary identification ability without privileged information required can achieve a comparable performance with doing SFT on a specially curated abstention dataset.

### Weaknesses
1. **Necessity of warm-start trajectory construction**. The paper lacks justification for the complex warm-start procedure over simpler alternatives, such as rejection sampling with format constraints. Table 4 shows models trained without warm-start, but no direct comparison between warm-start trajectory construction and rejection sampling is provided. The rationale for using a base model rather than an instruct model for synthetic data generation remains unclear and may introduce unnecessary complexity.
2. **Inadequate handling of partial knowledge and reward exploitation**. MASH primarily addresses extreme cases (Abs(0) and Abs(1)) while neglecting the critical middle ground where models have partial knowledge. Several issues arise: (1) In multi-hop reasoning, models may just happen to generate intermediate entities without retrieval (Pass@K). It causes MASH to penalize all other trajectories in the GRPO group, leading to unstable RL training. (2) For multiple-choice or binary questions, models can achieve rewards through random guessing with incorrect reasoning, a problem observed in Search-R1 that MASH's strict penalties may exacerbate. (3) The aggressive penalty structure makes integration with other reward signals challenging without careful balancing.
3. **Limited benchmark coverage**. The evaluation omits MuSiQue, a widely used benchmark that supports at most 4-hop reasoning, which would better test the approach's scalability for complex multi-hop queries.
4. **Insufficient model scale evaluation**. Experiments exclusively use Qwen2.5-3B-base, leaving open the question of whether findings generalize to larger models (7B, 14B) or newer architectures (Qwen3), where different dynamics may emerge.
5. **Incomplete baseline analysis**. Table 2 omits Search-R1's detailed search distribution. The consistent TC=3.0 for Search-R1 on multi-hop tasks (Table 1) seems weird. The model should sometimes answer directly or perform additional searches if it fails to generate appropriate queries.
6. **Presentation clarity issues**. The paper's flow and readability need improvement. For instance, the abstract's key idea is difficult to follow. Moreover, the paper suffers from excessive use of dashes, disrupting readability.

### Questions
Line 177. What is "random correct?" Does it mean rolling out multiple trajectories that contain the correct answer, and then randomly picking one? And why use "shortest answer" if we cannot get one correct trajectory?

### Soundness
2

### Presentation
2

### Contribution
3
