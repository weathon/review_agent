# BALROG: Contextual Bandits meets Active Learning for Online Generative Model Selection

- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
The rapid proliferation of open-platform text-to-image generative models has made prompt-wise model selection essential for producing high-quality and semantically accurate images, yet it remains a challenging problem. Existing approaches, including contextual bandit algorithms, often converge slowly and fail to exploit the semantic relationships across prompts. We introduce BALROG, a non-parametric, neighbor-based bandit framework that directly addresses these issues by transferring information across similar prompts to speed up convergence and improve generalization. By leveraging similarities between prompts, BALROG achieves faster learning and comes with strong theoretical guarantees through a sub-linear regret bound. In addition, we incorporate an active learning strategy that selectively queries ground-truth model rankings on ambiguous prompts, where ambiguity is quantified by the gap between the estimated rewards of the top two candidate models. This simple yet effective uncertainty measure substantially improves convergence and robustness. Extensive experiments on four datasets with six image generative models show that BALROG reduces regret by up to 60% compared to state-of-the-art baselines, enabling more accurate prompt-wise model selection in practice.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes BALROG, a non-parametric, neighbor-based contextual bandit framework for online generative model selection. The key idea is to exploit similarities between prompts so that information can be transferred across semantically related queries, speeding up convergence. The method integrates a lightweight active-learning component that queries human feedback when model rankings are uncertain. Experiments across several text-to-image datasets and six generative models report up to 60 % regret reduction over existing bandit baselines, supported by a sub-linear theoretical regret bound.

### Strengths
1.Addresses a practically relevant and under-explored problem: online model selection for generative models.

2.The non-parametric neighbor transfer idea adds useful inductive bias without requiring deep model retraining.

3.Incorporates an interpretable active-learning strategy to reduce annotation cost.

4.Provides both theoretical analysis (regret bound) and empirical validation across several datasets.

### Weaknesses
1.Motivation and methodological clarity.
The motivation for adopting a non-parametric, neighbor-based framework is not sufficiently articulated. While the idea of transferring information across similar prompts is intuitive, the paper would benefit from clearer justification or empirical evidence demonstrating when this approach substantially outperforms parametric alternatives. Concrete examples or ablation results could help clarify the design rationale.

2.Limited evaluation scope.
The experimental evaluation is conducted on a relatively small number of datasets and prompt sets, which may not capture the true diversity of user intents or prompt semantics in open-domain scenarios. This restricts the generalizability of the claims regarding real-world model selection.

3.Insufficient experimental rigor.
It is unclear whether all baselines were tuned equivalently. The paper also lacks ablation studies on key design choices such as the number of neighbors and the reward-gap threshold, making it difficult to assess the robustness and sensitivity of the proposed framework.

4.Evaluation methodology and interpretability.
Figure 1 appears to suggest that CLIPScore differences alone can effectively rank models, raising doubts about the incremental benefit of BALROG beyond such baselines. Moreover, since both OtB and OPR rely on CLIPScore, these metrics may not fully capture user-perceived or aesthetic quality. Including qualitative visualizations or limited human evaluations would help substantiate the practical value of the method.

### Questions
Could BALROG be applied to multimodal tasks (e.g., text-to-audio)?
What is the computational complexity of maintaining neighbor graphs online?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces BALROG, a contextual bandit algorithm with an integrated active learning mechanism for online selection among text-to-image generative models.
The method extends prior bandit approaches (e.g., PAK-UCB) by replacing parametric reward estimation with a non-parametric k-nearest-neighbor scheme and by adding an active querying rule that requests full feedback when the top-two model estimates are close.

The paper provides a theoretical poly-logarithmic regret bound under smoothness and margin assumptions, and presents experiments on four prompt datasets and six diffusion models.
Empirically, BALROG shows improved CLIPScore-based performance over existing bandit baselines.

While the problem setting is interesting, the overall contribution is incremental—the algorithm mainly adapts existing kNN-UCB and active-learning ideas to the generative model selection context, and the practical impact and scalability of the approach remain unclear.

### Strengths
This paper introduces BALROG, a contextual bandit algorithm with an integrated active learning mechanism for online selection among text-to-image generative models.
The method extends prior bandit approaches (e.g., PAK-UCB) by replacing parametric reward estimation with a non-parametric k-nearest-neighbor scheme and by adding an active querying rule that requests full feedback when the top-two model estimates are close.

The paper provides a theoretical poly-logarithmic regret bound under smoothness and margin assumptions, and presents experiments on four prompt datasets and six diffusion models.
Empirically, BALROG shows improved CLIPScore-based performance over existing bandit baselines.

### Weaknesses
The main limitation of this work lies in its lack of conceptual novelty and unclear practical significance.
The proposed BALROG framework essentially combines two well-known ideas—non-parametric contextual bandits (e.g., kNN-UCB, Reeve et al., 2018) and active learning based on uncertainty sampling (Settles, 2009)—and applies them to the relatively superficial task of selecting among a small set of pretrained text-to-image models.
This setup makes the contribution appear mostly an adaptation rather than a new algorithmic insight.

From an application standpoint, the problem definition itself is questionable in utility: choosing between existing generative models for each prompt does not necessarily advance the understanding or improvement of generative modeling. In practice, such “model routing” could be implemented with a simple ensemble or ranking heuristic without the need for a bandit formulation.
As a result, the proposed method risks being trivial in motivation, offering little theoretical or empirical insight beyond existing contextual-bandit literature.

### Questions
None

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
3

### Summary
This paper introduces BALROG, a non-parametric, neighbor-based contextual bandit framework designed for the online model selection problem in text-to-image generative models. BALROG addresses these limitations by transferring information between similar prompts. It also utilize active learning method to enhance the algorithm performance. Also, this paper provides strong theoretical guarantees for BALROG algorithm, which including upper regret bound derivation as well as time and space complexity derivation.

### Strengths
- BALROG employs a non-parametric k-Nearest Neighbor approach for reward estimation. The core innovation is the incorporation of a finite-budget active learning strategy.The proposed Delta rule is a simple yet highly effective measure of uncertainty, and the utilization of active learning is quite impressive.
- The math derivation of this paper is clear and concise. The question formulation as well as the pseudocode part were written clearly. 
- With many text-to-image models coexisting, automatically selecting the best model for each prompt is a very practical problem – the paper clearly illustrates this practical scenario and gives a solution BALROG.

### Weaknesses
- Lack of ablation study is the biggest problem for this paper. First, the algorithm involves many hyperparameters. If these hyperparameters are fixed, whether there is a significant impact on the performance of the algorithm should be shown in the paper. Secondly, the paper mentions that active learning can "reduce ambiguity, accelerate the convergence of neighborhood estimates, and uncover correlations between models." I hope there will be ablation experiments to illustrate this.
- The experiments focus mainly on contextual bandit baselines. Missing are more LLM-based adaptive routing or meta-learning approaches that could serve as stronger recent baselines.
- It is unclear how sensitive BALROG’s performance is to the choice of reward metric. CLIPScore mainly focus on prompt-image alignment, but it cannot measure the quality of image itself. 
- Insufficient sensitivity explanation for active query policy thresholds. Delta, UCB-threshold, variance-threshold, and other strategies all require thresholds. Although there is a section in the appendix on threshold selection (B.2), the text should report more clearly on the sensitivity analysis of thresholds.
- Insufficient reporting of runtime/GPU costs in the experiment. Given that the method actually performs a large number of model inferences (especially in the query step), it is recommended to report the actual GPU hours or the total computing resources per experiment to facilitate the review and judgment of the method's feasibility in the real world (the recommended inference steps for each model are given in the appendix, but the overall computational cost statistics are missing).
- Minor problem, 148-149, $\mu$ is a redundant definition. Better remove this.

### Questions
- How does the algorithm perform when the embedding space is noisy or high-dimensional?
- Does the performance depend heavily on δ, or is it robust? A sensitivity analysis could clarify stability.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes BALROG, a contextual bandit framework for online selection of text-to-image generative models. The method addresses the challenge of selecting the best generative model for each input prompt by leveraging semantic similarities between prompts and strategically querying full model feedback when uncertainty is high. BALROG achieves faster convergence and improved generalization compared to existing bandit approaches.

### Strengths
1. The paper is clearly written and well organized. The motivation for prompt-wise selection is compelling and illustrated visually.
2. It creatively combines non-parametric neighbor-based bandits (kNN-UCB) with budgeted active learning, using a simple but effective top-two gap trigger to allocate full-feedback queries at decision boundaries. This synthesis removes key limitations of prior contextual bandits that assume fixed parametric form or neglect cross-prompt structure.
3. Theoretical analysis is nontrivial and well-structured, with careful assumptions and supporting lemmas. The improvement over passive non-parametric baselines is clearly articulated.

### Weaknesses
1. Tha paper’s comparison of active learning strategies is primarily focused on different triggers within its own framework (Delta, UCB-threshold, etc.) and a single external baseline. This is insufficient to fully establish the superiority of the proposed “on-the-fence” querying approach. A critical missing strategy would be a simple, nonadaptive budget allocation strategy. For instance, distributing the query budget uniformly at random across the entire horizon could serve as a powerful sanity check. If such a simple strategy performs comparably, it might suggest that the benefit comes more from having access to full feedback itself, rather than the timing of the queries.
2. The geometric uncertainty weight, which is set to log(t) in practice. The paper dose not provide a strong theoretical or empirical rationale for this specific choice over other non-decreasing functions (e.g., a constant, sqrt(log(t)), or t). It is unclear how robust the algorithm’s performance is to this hyperparameter.
3. The paper exclusively uses one type of embedding and one metric. However, the quality and the geometric properties of text embeddings can vary significantly. Similarly, other distance metrics(e.g. L2 distance) could alter the neighborhood structure. It is unclear whether the strong results are specific to the CLIP embedding and CLIPScore or genuinely robust to the choice of representation and metric.

### Questions
1. Could the authors provide a comparison against a simple yet important baseline, such as allocating the active query budget uniformly at random across the entire horizon? How does your proposed method perform against such a non-adaptive strategy?
2. Could you provide a more detailed justification for the setting phi(t)=log(t)? How sensitive is BALROG’s performance to different phi(t)?
3. The paper’s results are based solely on CLIP embeddings and cosine distance. Have you experimented with other text embedding models(e.g., BLIP) or alternative distance metrics?

### Soundness
2

### Presentation
3

### Contribution
3
