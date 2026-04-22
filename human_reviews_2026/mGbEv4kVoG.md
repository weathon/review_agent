# Why Ask One When You Can Ask $k$? Learning-to-Defer to the Top-$k$ Experts

- Avg Score: 5.20
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6, 2

## Abstract
Existing _Learning-to-Defer_ (L2D) frameworks are limited to _single-expert deferral_, forcing each query to rely on only one expert and preventing the use of collective expertise. We introduce the first framework for _Top-$k$ Learning-to-Defer_, which allocates queries to the $k$ most cost-effective entities. Our formulation unifies and strictly generalizes prior approaches, including the _one-stage_ and _two-stage_ regimes, _selective prediction_, and classical cascades. In particular, it recovers the usual Top-1 deferral rule as a special case while enabling principled collaboration with multiple experts when $k>1$. We further propose _Top-$k(x)$ Learning-to-Defer_, an adaptive variant that learns the optimal number of experts per query based on input difficulty, expert quality, and consultation cost. To enable practical learning, we develop a novel surrogate loss that is Bayes-consistent, $\mathcal{H}_h$-consistent in the one-stage setting, and $(\mathcal{H}_r,\mathcal{H}_g)$-consistent in the two-stage setting. Crucially, this surrogate is independent of $k$, allowing a single policy to be learned once and deployed flexibly across $k$. Experiments across both regimes show that Top-$k$ and Top-$k(x)$ deliver superior accuracy–cost trade-offs, opening a new direction for multi-expert deferral in L2D.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes Top - $k$ Learning-to-Defer (L2D), a novel framework that allows a model to defer each query to a set of $k$ experts instead of a single expert. This generalizes all prior L2D approaches, which were limited to one expert (“Top-1”) deferrals. The authors formulate a unified approach covering both major paradigms of L2D: in the one-stage regime, the predictor and deferral policy are learned jointly in one model, and in the two-stage regime, experts and a main predictor are trained offline and a separate routing function learns to allocate queries. Crucially, the framework recovers the traditional Top-1 deferral as a special case while enabling principled collaboration with multiple experts when $k>1$.  In addition, the paper introduces Top-$k(x)$ L2D, an adaptive extension where the system learns to choose a query-specific number of experts (up to $|A|$, the total pool) based on factors like input difficulty, expert accuracy, and consultation cost. To support both fixed-$k$ and adaptive deferral, the authors develop a $k$-independent surrogate loss that facilitates practical training.

### Strengths
- Innovative Multi-Expert Deferral Framework: This is the first work to allow deferring to multiple experts in a Learning-to-Defer setting, addressing a real-world gap. The idea is original and expands the scope of L2D beyond the restrictive one-expert paradigm. By formulating Top-$k$ deferral, the paper enables more robust decision-making (aggregating complementary expertise) which is crucial for complex tasks. This multi-expert allocation capability is a direct answer to the stated limitation of prior work (that relying on only one expert can amplify bias or error). The framework cleanly unifies both one-stage and two-stage L2D, showing broad applicability across different training setups. It also strictly generalizes previous approaches: for example, the authors show that setting $k=1$ recovers the standard L2D solutions in both regimes, and even classical model cascades and selective prediction emerge as special cases of their formulation. 

- Theoretical Foundations: The paper provides theoretical analysis and guarantees. The introduction of a $k$-independent surrogate loss function is a technical strength. It is proven to be Bayes-consistent, meaning that as the amount of training data grows, the learned deferral policy will converge to the Bayes-optimal top-$k$ strategy (the one that minimizes true expected cost). Furthermore, the surrogate is shown to maintain H-consistency within restricted hypothesis classes providing bounds on the excess risk when using finite-capacity models. These results extend known consistency bounds from $k=1$ to the multi-expert scenario. The theoretical development is non-trivial and well-executed.

- Empirical Effectiveness: The proposed methods show empirical advantages over prior approaches. Across multiple benchmarks, Top-$k$ deferral improves the trade-off between prediction accuracy and consultation cost. Notably, Top-$k(x)$ (adaptive) is very effective: it achieves comparable or better accuracy with significantly lower average cost by varying the number of experts per query. For instance, in one experiment the adaptive policy reached the same RMSE as a fixed $k$ policy while querying ~1.3 fewer experts on average (saving ~25% of the budget). This demonstrates efficient use of experts. The results also confirmed that even a fixed Top-$k$ (with an appropriate $k$) outperforms the traditional Top-1 deferral baseline in accuracy, under the same cost budget. Overall, the authors back up their claims with comprehensive experiments in both one-stage (e.g. CIFAR-10 image classification with learned deferral) and two-stage (e.g. regression with a separate main model and experts) scenarios.

### Weaknesses
- Complexity of Training & Implementation: One potential weakness is the added complexity of the proposed approach, especially the adaptive Top-$k(x)$ mechanism. The framework introduces an additional model (the cardinality function $k_\theta(x)$) that must be learned alongside the main deferral policy. This effectively means training two interconnected components: one to score and select experts, and another to decide how many to select. The paper defers some of the training details to the appendix (Algorithms 1 and 2 outline separate procedures for the policy and the cardinality function), but it’s not entirely clear from the main text whether these components are trained jointly or sequentially, and how hyperparameters (especially the regularization $\lambda$ in $\ell_{\text{card}}$) are chosen. The need to tune $\lambda$ to balance accuracy vs. cost could require expert knowledge or cross-validation on a utility metric, which adds complexity. In practice, implementing Top-$k(x)$ may be more involved than standard L2D due to this bi-level optimization (learn policy given a fixed $k$, and learn the $k$-selection function), a point the authors could clarify further. 

- Evaluation Scope – Lack of Real-World Expert Data: The experiments, while thorough on synthetic and benchmark data, do not include a real-world multi-expert scenario, which slightly limits the demonstration of the framework’s practical value. All evaluations seem to use either simulated experts (e.g., classifiers with predetermined accuracy on subsets of classes) or machine models as stand-in “experts.” While this is common in L2D research (since obtaining multiple human experts’ responses can be difficult), it would have been insightful to see at least one case with actual human experts or a real ensemble to validate that Top-$k$ deferral works outside the lab setting.

- Assumptions for Theoretical Guarantees: The consistency guarantees, while impressive, rely on certain assumptions that may not hold in practice. For example, the two-stage consistency assumes that the Bayes-optimal cost can be approached arbitrarily well by the model classes. Essentially this is a realizability or universal approximation assumption for the hypothesis spaces. If the chosen model architectures are misspecified or capacity-limited, there could be a minimizability gap where the learned policy cannot reach the true optimum. The authors do acknowledge this, but in practice one never knows if one’s model is rich enough. Thus, the Bayes-consistency is a bit theoretical, at finite sample sizes with imperfect models, one might not achieve the optimal deferral policy. The paper does not discuss the finite-sample performance beyond asymptotic consistency; for instance, how fast is the convergence or how the excess risk scales with $k$. Some empirical calibration of the surrogate (like comparing surrogate risk vs true deferral risk on validation data) could help validate that the loss is working as intended in finite data regimes.

### Questions
- Training Strategy for $k$-Selection: Could you elaborate on how the deferral policy $\pi(x,\cdot)$ and the cardinality function $k_\theta(x)$ are trained in practice? Was the $k$-adaptive model trained in an end-to-end fashion, or did you first learn a Top-$|A|$ policy and then learn the cardinality predictor (or vice versa)? 

- Decision Rule for Multiple Experts’ Outputs: In a deployment, once the Top-$k$ experts are consulted for a query, how is their collective response used to make a final prediction? In your experiments, what rule did you use to decide the final output or correctness from multiple predictions? For example, for CIFAR-10 with $k=2$ experts selected, did you consider the query correct if either expert (or the model) got the label right (i.e. at least one correct prediction in the set), or did you require agreement/majority? Different aggregation strategies (e.g., majority vote vs. oracle-style “at least one correct”) can yield different accuracy metrics. It would clarify your evaluation if you specify the aggregation/decision procedure for multiple expert opinions. 

- Sensitivity to Cost Hyperparameters ($\alpha,\beta,\lambda$): The framework introduces cost parameters $\alpha_j$ and $\beta_j$ for each expert (measuring error penalty vs. consultation cost), as well as a regularization weight $\lambda$ in the adaptive loss. How sensitive are your results to the choice of these values? In practice, one might not know the exact cost of consulting an expert or might have to guess a trade-off parameter. Did you perform any analysis or ablation where you vary $\beta_j$ or $\lambda$ to see how the deferral behavior changes? For instance, if all $\beta_j$ are set to 0 (ignoring cost), does the policy simply always choose all experts (maximizing accuracy)? And conversely, if $\beta_j$ are very high, does it revert to always using the model (to save cost)?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
Learning-to-defer (L2D) deals with utilizing expert advice for queries. Existing approaches only consider using a single expert, and this paper proposes a loss that works for multiple experts and satisfies desirable consistency objectives. Further, the author shows that prior setups are special cases of the novel approach. Experiments are provided to justify the theoretical results.

### Strengths
- The paper is well motivated and contextualized. For instance, it is good that the authors mention why the naive extension of the top-1 selection doesn't work (line 216).
- Extensive experiments are provided, and description of setup is very detailed.

### Weaknesses
- Some of the notation is kind of cumbersome, eg the superscripted down and up-arrows. Also, hat seems to be used for many things: argmax (eg line 115), experts, and predictions (lines 134-135). It would be better to use something like asterisk for argmax, eg $h^* = \arg\max h_i$
- It looks like the proof of Lemma A.6 is incorrect/incomplete, as eq 15 is directly assuming the implication.

### Questions
Overall, the presentation could use significant work. For instance:
- Can $\mathcal{Y}$ be defined at the beginning of the preliminaries? More generally, please provide a more exhaustive discussion of notation, eg. either in the preliminaries, or as a table in the appendix.
- What is the definition of consistency, calibration? It would be good to add a section in the appendix for additional preliminaries to make all of this self-contained.

Additional:
- Is top-k(x) also consistent (as in Theorem 4.7)?
- In Figure 5, is there intuition why accuracy for top-k isn't monotonic increasing (in budget) whereas top-k(x) is?
- wording in line 400 is a little unclear: minimizability gap... vanishes... if the hypothesis set is as rich as the set of all measurable functions

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
2

### Summary
This paper studies the Learning-to-Defer (L2D) problem, where a model may defer prediction to external experts. While existing L2D frameworks only allow deferral to a single expert, this work introduces a more general framework that supports deferring to the top-$k$ most cost-effective experts. The authors further develop an adaptive extension, Top-$k(x)$, which selects the optimal number of experts per input based on expert quality, instance difficulty, and consultation cost.

To support this generalization, the paper proposes a surrogate loss that is Bayes-consistent and class-consistent in both one-stage and two-stage L2D settings. Importantly, this surrogate is independent of $k$, meaning that a single model can be trained once and deployed for arbitrary $k$. The authors show that this framework strictly subsumes prior approaches, including selective prediction and Top-1 L2D, and they also unify cascade-style inference under the same formulation. Experimental results suggest that Top-$k$ and Top-$k(x)$ achieve better accuracy–cost trade-offs compared to existing single-expert L2D methods.

### Strengths
- The paper introduces a novel and compelling extension of the L2D framework to multi-expert settings, which appears natural and highly relevant in domains requiring collective expert judgment.
- The theoretical development is rigorous, with proofs of Bayes-consistency and class-consistency that generalize known guarantees from Top-1 to Top-$k$ and adaptive settings.
- The unifying perspective on existing L2D, top-$k$ classification, selective prediction, and cascaded models is conceptually valuable and may stimulate further research in this direction.
- Experimental results demonstrate noticeable improvements over prior approaches, particularly under constrained consultation budgets.

### Weaknesses
- While the theoretical contributions are strong, empirical evaluation is somewhat limited. The experiments focus on standard benchmarks, and it remains unclear how well the proposed framework performs in practical multi-expert environments with heterogeneous expert behavior or human-in-the-loop settings.
- The implementation details and computational overhead of training with multi-expert routing are not discussed in depth. In practice, the cost of querying multiple experts and potential resource constraints may affect feasibility.
- The method assumes that expert predictions are readily available. It would be valuable to discuss applicability when expert labels are costly or sparse.

### Questions
- Can the authors comment on how the framework would behave when expert accuracy varies significantly across different regions of the input space? Is there theoretical or empirical justification that the learned top-$k$ policy will avoid consulting redundant experts?
- How sensitive is the adaptive Top-$k(x)$ approach to the choice of regularization on consultation cost? Are there guidelines for selecting these parameters in practice?
- In real-world settings where experts may be humans, costs and response times are not fixed. Could the authors discuss whether their framework can incorporate stochastic or dynamic consultation costs?

### Soundness
3

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
This paper studies the multi-expert Learning-to-Defer (L2D) problem and extends the classical Top-1 deferral framework to a more general Top-k deferral setting, where the system may consult multiple entities (base model and experts) simultaneously. The authors formulate the Top-k deferral loss, derive the Bayes-optimal Top-k selection rule, and propose the first Bayes-consistent surrogate for Top-k L2D that is independent of k. This allows a single learned scoring function to support any fixed k, as well as an adaptive variant Top-k(x) that chooses a sample-specific number of queried entities. The proposed framework unifies prior one-stage and two-stage L2D results as a special case of k=1, and the authors further validate the approach with extensive experiments demonstrating the benefits of consulting multiple entities and of adapting k to the input.

### Strengths
1. The proposed Top-k formulation strictly generalizes prior Top-1 L2D methods, and the theoretical results provably recover previous consistency guarantees when k=1. This provides a clean unification of existing L2D theory under a more general and expressive decision space.

2. Theoretical guarantees are established for both one-stage and two-stage regimes, which is non-trivial. The surrogate achieves Bayes-consistency in both settings, thereby offering a comprehensive and unified theoretical foundation for multi-expert deferral.

3. The empirical results are thorough and convincingly support the claims. The experiments show that Top-k L2D improves over Top-1 and random querying, and that the adaptive Top-k(x) strategy yields further improvements by selecting the appropriate number of experts per input under a given budget constraint.

### Weaknesses
1. The paper does not provide theoretical justification for why consulting and aggregating multiple entities should improve final prediction quality. The theory only covers the selection stage, while the aggregation step remains ad-hoc and intuition-based, leaving the core motivation for Top-k partially unsupported from a theoretical perspective.

2. The paper assumes that all experts are deterministic functions of the input $X$ (line 108). 
In contrast, the original multi-expert deferral work [1] does not impose this constraint. 
This assumption appears restrictive, since an expert depending only on $X$ cannot outperform the most probable label, 
which may lead the Bayes-optimal deferral strategy to never defer. While this setting is indeed non-trivial under the zero extra cost or $\mathcal{H}$-consistency framework, 
extending the analysis to stochastic experts would significantly strengthen the contribution and practical relevance of this work.

3. In line 171, [2] is described as being related to $\mathcal{H}$-consistency. However, [2] mainly focuses on Bayes consistency and surrogate regret bounds.
It is suggested to clarify this distinction to avoid confusion and to better position the related concepts.

4. Several recent advances in learning to defer [3] and closely related work on classification with rejection [4, 5] are not cited. Including these developments would provide a more complete and up-to-date literature review.

[1].  Rajeev Verma, Daniel Barrejon, and Eric Nalisnick. Learning to defer to multiple experts: Consistent surrogate losses, confidence calibration, and conformal ensembles. AISTATS 2023.

[2]. Hussein Mozannar and David Sontag. Consistent estimators for learning to defer to an expert. ICML 2020.

[3]. Zixi Wei, Yuzhou Cao, and Lei Feng. Exploiting human-ai dependence for learning to defer. ICML 2024.

[4]. Yuzhou Cao, Tianchi Cai, Lei Feng, Lihong Gu, Jinjie Gu, Bo An, Gang Niu, and Masashi Sugiyama. Generalizing consistent multi-class classification with rejection to be compatible with arbitrary losses. NeurIPS 2022

[5]. Corinna Cortes, Giulia DeSalvo, and Mehryar Mohri. Theory and algorithms for learning with rejection in binary classification. Annals of Mathematics and Artificial Intelligence, 92(2), 2024, 277–315.

### Questions
Please refer to the weaknesses listed above. I would be glad to revise my rating if the authors address these points in a future version of the paper.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper considers a setting where, given a feature, a learner makes predictions or defers to a set of experts. They formulate the problem abstractly, as learning a model for selecting k assets, where each asset can either be a label or an expert. The hypotheses selected by the learner score each of these assets, and the loss incurred by the learner is a function of the top-k highest scoring assets. The specific loss function they consider is a sum over the individual losses of each of the selected assets. The individual loss is a cost-sensitive loss combining a predictive loss for the label selected by the model (or expert), and possibly a consultation loss for deferring to an expert. 

They describe how their setting generalizes previous work in what they call the 1-stage and 2-stage settings. They use ideas from the literature to bound the cost sensitive loss with a surrogate, and prove statistical consistency. They state a version of the problem where k need not be fixed. Finally, they conclude with empirical experiments that compare two methods for fixed k and adaptive k.

### Strengths
Strength: I like that the paper unifies the 1-stage and 2-stage settings in a single framework that considers experts and labels to simply be decisions ("assets" in the language of the paper) that need to be scored by some model.

### Weaknesses
Weakness 1. The theoretical portion of the paper considers a loss function that sums the individual losses of each asset. As a modeling assumption, this has a number of problems. In settings where there is a good expert, or when the model is good at identifying labels without experts, the model is forced to select a number of sub-optimal assets by the setting, and is strictly penalized for doing so. The aggregation also does not allow the learner to meaningfully improve its prediction (e.g. by taking a major vote). I believe the authors mitigate this somewhat by introducing the adaptive-k setting and by considering more interesting aggregation strategies in their experiments. However, it does call into question the utility of the theory around a somewhat degenerate loss function. 

Weakness 2: The surrogate loss optimized by the authors is independent of k (Corollary 4.4), and recovers the top-1 solution from previous literature. Thus, the authors' algorithm learns a hypothesis for the top-1 problem, and uses the k highest scores found by that hypothesis to pick its assets. In light of Weakness #1, the most interesting aspect of this problem over top-1 is to learn a model whose scores preserve rank over suboptimal assets, which this solution expressly avoids modeling.  

Weakness 3: The experiments primarily compare the paper's methods against each other. I am not convinced that there are not more adequate baselines. Cortez et al. 2024, which the authors cite, for example, seems very relevant. 

Weakness 4: The paper casts and re-casts the problem a number of times before finally arriving at the formulation in lines 200-208 and 226-240. I think the presentation can be improved by presenting this more succinctly. In my opinion, the observation that this generalizes prior work need not take up more than a paragraph or two.

### Questions
Why is Lemma 4.2 stated as a Lemma, when it seems more like a definition? The proof of the Lemma does not seem to help resolve it for me. The proof seems to state that the uniform top-k loss upper bounds the standard deferral loss (which #1 seems obvious, and #2 is not in the statement of Lemma 4.2). It is also interspersed with asides about special cases, which are not proving anything.

### Soundness
3

### Presentation
2

### Contribution
2
