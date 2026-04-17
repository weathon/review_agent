# Learning to Route LLMs from Bandit Feedback: One Policy, Many Trade-offs

- Decision: Reject
- Scores: 4, 4, 2

## Abstract
Efficient use of large language models (LLMs) is critical for deployment at scale: without adaptive routing, systems either overpay for strong models or risk poor performance from weaker ones. Selecting the right LLM for each query is fundamentally an online decision problem: models differ in strengths, prices fluctuate, and users value performance and cost differently. Yet most routers are trained offline with labels for all candidate models, an assumption that breaks in deployment, where only the outcome of the chosen model is observed. We bridge this gap with a bandit-feedback routing approach that trains under the same partial-feedback restriction as deployment, while supporting preference-tunable inference: operators can dial the performance–cost trade-off at test time without retraining. Framed as a contextual bandit over prompt features and a user preference vector, our method simulates an online feedback setting during training and adapts its routing decisions to each new prompt, rather than depending on full-information offline supervision. Comprehensive experiments on RouterBench show that our method consistently outperforms strong offline routers, including GraphRouter and RouterDC, in terms of performance and cost, and generalizes robustly for unseen tasks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses the problem of selecting the best large language model (LLM) for each query under real-world constraints of partial feedback and varying user priorities. It proposes BaRP (Bandit-feedback Routing with Preferences), a routing policy trained as a contextual bandit that observes only the chosen model’s outcome (bandit feedback) and conditions on a user-specified accuracy-vs-cost preference vector. This approach bridges the gap between offline training (which assumes full information) and deployment (which provides only bandit feedback), allowing the router to learn without requiring labels from all models and to dial the performance–cost trade-off at test time without retraining. Experiments show that BaRP consistently outperforms prior offline-trained routers by significant margins (≥12% relative) and even surpasses always using the largest model (by ~2–4%), while generalizing robustly to unseen tasks.

### Strengths
- The paper tackles two key challenges together, learning from partial (bandit) feedback and allowing tunable cost–accuracy trade-offs, in one framework, effectively addressing limitations that kept prior routers from being deployed adaptively. This yields a practical solution for real-world LLM orchestration.

- Empirical performance: BaRP demonstrates consistent outperformance over baselines, including prior learned routers (e.g. RouterDC, GraphRouter) and trivial policies (always using the largest or smallest model). 

- Flexible inference capability: A strength is the model’s ability to adapt to different user preference settings on the fly. The single learned policy can be “dialed” between high-accuracy vs. low-cost operating points without retraining.

### Weaknesses
- Limited evaluation scope: The experimental evaluation, while thorough on the provided benchmarks, omits some recently introduced routing benchmarks. In particular, the paper does not evaluate on the SPROUT dataset from the CARROT router work (https://arxiv.org/pdf/2502.03261), which is designed to test cost-aware routing across a broad range of queries and models.

- Missing comparison with concurrent work: The authors do not discuss or compare against a relevant concurrent approach in the literature, specifically, a NeurIPS 2025 paper (https://openreview.net/pdf?id=Rx0JIC41LJ) that also addresses a very similar LLM routing under partial feedback with cost-aware exploration strategies.  

- Lack of theoretical insight: The contribution is primarily empirical, without accompanying theoretical guarantees. Unlike some recent works that provide formal analyses, this paper does not analyze convergence or regret for its bandit training procedure.

### Questions
My biggest question is: How is it possible for BaRP to outperform full-feedback routers like RouterDC and GraphRouter, despite only receiving bandit feedback during training? This seems counterintuitive, since those baselines have access to strictly more supervision. I encourage the authors to investigate this further and clarify what enables BaRP to achieve such results.

In addition:


- Evaluation on CARROT/SPROUT: Could the authors evaluate BaRP on the recently introduced SPROUT dataset from the CARROT router. This dataset provides a wide spectrum of query difficulties and up-to-date model pool (used alongside RouterBench in the CARROT paper). Testing BaRP on SPROUT would help confirm that its performance gains hold under a broader range of conditions and with the latest models.

- Comparison with Cost-Aware Exploration: There is a concurrent NeurIPS 2025 work that also learns routing policies from observational (bandit) feedback with user cost preferences. How does BaRP differ from this approach in terms of methodology and results? It would strengthen the paper to include a discussion or empirical comparison with that method to clarify the advantages or trade-offs of your solution.

- Ablation on cost scaling: Could the authors analyze the impact of the calibrated cost scaling trick used during training? An ablation study (e.g., training the policy without cost scaling) would be valuable to demonstrate how much this component contributes to stability or performance. Clarifying whether the policy’s effectiveness is sensitive to this scaling (or other hyperparameters like the entropy regularization weight) would help others understand and reproduce the approach.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces BaRP (Bandit-feedback Routing with Preferences), a framework for efficiently selecting the right large language model per query under the online learing setting. Unlike offline routers that require full supervision, BaRP learns from bandit feedback and adapts routing decisions based on the historical observations only. Experiments show that it outperforms both strong offline routers and individual LLMs across multiple tasks.

### Strengths
1. This paper studies a very interesting problem, and I believe LLM routing is a useful topic for real-world implementation and has great potentials.
2. The presentation of this work is mostly clear.
3. The experimental results are significant and clear to highlight the advantages of BaRP over some baselines.

### Weaknesses
1. Part of the presentation can be improved. For example, it is not clear to me how the preference encoder looks and how it is trained. I feel this encoder is co-trained with the decision head, but from Algorithm 1 pseudo-code, it seems that the preference encoder is pre-defined.
2. I feel it is better to show some online evaluation metrics, i.e. the model's performance during the online learning training phase. Right now the model is evaluated on the testing dataset after being fully trained on the training dataset, which I feel it is the same as the traditional offline training works in terms of implementation.
3. From the section 4.5, it seems that the linear decision head can already yield similar performance compared with more advanced models such as the bilinear and MLP. However, from the Section 4.6, it seems that the linear bandit algorithm cannot yield stable performance, and the author claims that this is due to the non-linear pattern. Can you explain this phenomenon to me? Since I feel those LinTS, LinUCB, epsilon-greedy algorithms are the real classic bandit algorithms with theoretical guarantees.

### Questions
1. How do you choose the value of tau in your experiments and in practice? I am still now quite sure the intuition of this hyperparameter, since that may underestimate the cost of some ultra long responses.
2.  I may overlook, what is the meaning of the testing scores in your results, are they the real accuracies of the LLM responses? If so, it is quite surprising to see that sometimes the proposed BaRP algorithms can clearly outperforms the largest LLM. So, it is better to do some qualitative analysis on this result. Since the proposed method take the cost of the LLM into consideration as part of the training process, it is interesting to see that the obtained router is better than solely using the largest LLM on the accuracy.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The authors formulate LLM routing as a contextual bandit problem, where context $s_t = (x_t, w_t)$ is a query and cost/accuracy preference vector. The authors introduce an architecture for the policy $\pi_{\theta}(a|s)$ that outputs a probability distribution over models. The policy is fit on an entropy-regularized loss using REINFORCE, and results are compared to two prior works on Routerbench.

### Strengths
* The authors provide a partial evaluation on OOD performance of their router, which is an important consideration and is under discussed in the routing literature.
* The authors include a thorough ablation study on both decision heard architecture and the learning algorithm used.

### Weaknesses
The main weakness is insufficient comparison to prior work and evaluation restricted to routerbench.
* The authors comparing their method to GraphRouter (Feng et al., 2025) and RouterDC (Chen et al., 2024), claiming that their method improves both on data requirements and flexibility. Prior predictive routers (RORF [https://www.notdiamond.ai/blog/rorf], CARRROT [Somerstep et al. 2025], and Routerbench [Hu et al. 2024.]) are easily adaptable to the online setting (one can simply train the predictors online) and readily allow for tuning the preference vector $w$. In fact, StageRoute (Li and Li 2025.) comes with theoretical guarantees in the online setting. Additionally, routerDC is designed for settings where multiple LLMs perform well on a query, but in routerbench performance is relatively dominated by GPT4.
* Lack of routing benchmark variety. Prior work (e.g. CARROT, or Causal routing [Tsiourvas et al. 2025.]) evaluate routers on several data sets, such as SPROUT, and OpenLLMlleaderboard v2. Given that the main contribution needs to be improved routing performance, a more robust evaluation is in order.

### Questions
* Can the authors compare to a complete set of prior work on routing, and do so on more datasets?
* For the out of distribution test, can the authors also present the the cost/accuracy trade off numbers? Of course routing to the largest LLM is always best here so if their method simply selects this model more often it will perform better. Without more context I am not sure what to take from this.

### Soundness
2

### Presentation
3

### Contribution
2
