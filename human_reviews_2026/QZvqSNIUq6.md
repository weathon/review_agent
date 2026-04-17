# Journey to the Centre of Cluster: Harnessing Interior Nodes for A/B Testing under Network Interference

- Decision: Accept (Poster)
- Scores: 2, 4, 6, 4

## Abstract
A/B testing on platforms often faces challenges from network interference, where a unit's outcome depends not only on its own treatment but also on the treatments of its network neighbors. To address this, cluster-level randomization has become standard, enabling the use of network-aware estimators. These estimators typically trim the data to retain only a subset of informative units, achieving low bias under suitable conditions but often suffering from high variance. 
In this paper, we first demonstrate that the interior nodes—units whose neighbors all lie within the same cluster—constitute the vast majority of the post-trimming subpopulation. In light of this, we propose directly averaging over the interior nodes to construct the mean-in-interior (MII) estimator, which circumvents the delicate reweighting required by existing network-aware estimators and substantially reduces variance in classical settings. However, we show that interior nodes are often not representative of the full population, particularly in terms of network-dependent covariates, leading to notable bias. 
We then augment the MII estimator with a counterfactual predictor trained on the entire network, allowing us to adjust for covariate distribution shifts between the interior nodes and full population.
By rearranging the expression, we reveal that our augmented MII estimator embodies an analytical form of the point estimator within prediction-powered inference framework~\citep{angelopoulos2023prediction,angelopoulos2023ppi++}. This insight motivates a semi-supervised lens, wherein interior nodes are treated as labeled data subject to selection bias. Extensive and challenging simulation studies demonstrate the outstanding performance of our augmented MII estimator across various settings.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper addresses the challenge of conducting A/B tests on networks where interference between connected units violates standard causal assumptions. The authors propose the Mean-in-Interior (MII) estimator, which leverages interior nodes lying within the same cluster and *eliminates the inefficiency* introduced by the cluster-relevant weights used in the previous CAE estimator. Recognizing that interior nodes may not represent the overall population, the paper further introduces the Augmented MII (AMII) estimator, which employs a GNN-based counterfactual predictor to correct for selection bias. Theoretical analysis implies that AMII has an intuitively smaller bias than MII estimator in specific additive model, with another interpretation based on semi-supervised adjustment provided in terms of this method. Extensive simulations on semi-synthetic social network data demonstrate that AMII generally achieves lower MSE than comparative methods, establishing it as an efficient and practical approach for causal estimation under network interference.

### Strengths
The paper introduces the Mean-in-Interior (MII) and Augmented MII (AMII) estimators, fulfilling the gaps in GATE estimation based on interior nodes, and draws the attention on possible incomplete representativeness of the interior nodes to the entire cluster.

### Weaknesses
The paper features several notable weaknesses. While the authors attempt to convey dense ideas through multiple key intuitions and assertions, few are elaborated with sufficient rigor to enable non-expert readers to fully grasp them. The theoretical justification for the advantages of the MII and AMII estimators remains limited. Moreover, the rationale for viewing the MII estimator as an improvement over the CAE estimator is insufficient and not theoretically supported. Additionally, Section 3, which introduces the MII estimator, and Section 4, which discusses covariate adjustment, appear insufficiently connected in their conceptual development. Please see Questions for details.

### Questions
1. **On the novelty of the MII estimator**  
   Overall, I think the novelty of the proposed MII estimator could be better explained. From what I can tell, it’s basically a difference-in-means estimator applied to interior nodes, which is a pretty classical idea in causal inference. Also, the idea of leveraging interior nodes has been seen in earlier work. It would help if the authors could clarify the genuine improvement on MII estimator here in rigorous theories.

2. **On theoretical justification for the advantage of MII over CAE**  
   In lines 250–262, the paper argues that MII improves efficiency over the CAE estimator by using better cluster-relevant weights. But CAE’s weighting scheme is rooted in **cluster randomization**, which doesn’t seem to be part of the main assumptions (2.1 and 3.1). So, it feels like the two estimators might be targeting different experimental designs, making the motivation a bit insufficient. Also, there doesn’t seem to be a clear theoretical guarantee backing this claim—some clarification would be great.

3. **On clarity of statements**  
   A few parts could use clearer explanations or more precise wording:  
   - *Line 48:* The paper says that leveraging the known graph structure is *much more appealing*, but it doesn’t really argue why.  
   - *Line 251:* I didn’t quite follow why the model *helps build intuition* that variance analysis is more complex than bias analysis—or why it’s necessary to emphasize that intuition here.  
   - *Lines 315–316:* The statement *they may struggle … a comparative advantage* is confusing. It’s not clear why it “struggles,” or why an $f$ -model that doesn’t estimate the interference function $h$ should be considered an advantage. Since any model $f$ can adapt, this doesn’t seem to tie closely to the covariate adjustment goal. 
   - *Line 275-276:* *We argue that ... can be violated.* It does not provide any explanation not state exactly which assumptions are violated. 
   - *Line 281-282:* *Most estimators without a regression component tend to rely heavily on interior nodes*, there have been a number of estimators on exposure effects that do not rely on interior nodes as well, thus an arbitrary assertion.
   - The inconsistent names of $h$ as both ''transformation function'' (line 255) and ''interference function'' (line 315).

4. **On covariate adjustment and coherence**  
   Covariate adjustment can be applied to pretty much any estimator, not just MII. So it would be fairer to compare **AMII** with the augmented versions of all other estimators too. Also, Sections 3 and 4 don’t quite feel conceptually aligned—cluster-related weights and covariate adjustment seem like two separate issues, and it might help to connect them more clearly.

5. **On Assumption 4.1 and model training**  
   Assumption 4.1 says that $f$ is trained in a regression form, but in line 290 the paper says $f$ is trained using a GNN. How does that assumption hold in this case? A short explanation would make this part clearer.

6. **On experiments and ablation studies**  
   The experimental section feels a bit thin. It would really help to include a few ablation studies, for example:
   - What happens if covariates actually affect the outcome?
   - What if the interior nodes aren’t very representative of the full cluster?  
   Seeing these variations would strengthen the experimental support for MII.

### Soundness
2

### Presentation
1

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
This paper proposes two methods for estimating the global average treatment effect (GATE) when the effect of a unit may depend on the treatment assignments of nearby units, with proximity defined by a graph structure over the units.

The first proposed method, Mean-In-Interior (MII) estimator, averages the outcomes of the interior nodes receiving the same treatment as that of each unit. The authors show that the MII estimator is consistent (Theorem 3.1) under the Neighborhood Interference Assumption (NIA; Assumption 2.1) and the assumption about the interior nodes (Assumption 3.1).
The authors also claim that the MII estimator reduces the variance compared to the recently proposed CAE estimator (Liu et al., 2024).

The second method, called Augmented MII (AMII), corrects the bias of the MII estimator due to the focus on the interior points, using a counterfactual predictor $f$, which predicts the expected outcome given specified treatment assignments of all units (as well as covariates and the graph). Theorem 4.1 presents the bias of this estimator under Assumption 4.1 about the model.

Finally, the paper presents experiments using a synthetic dataset with several specifications of the parameters of the data generating process. The results shown in Figure 2 demonstrate that the MII estimator has lower variance compared to the previous methods, and the AMII reduces the bias leading to reduced MSEs.

### Strengths
- The task of estimation of GATE in the presence of interference is practically relevant and challenging.

- The proposed methods are developed based on interesting ideas.

- Theorems 3.1 and 4.1 provide soundness and strength of the proposed methods.

I set the Contribution score 3: good because the direction of this work looks great.

### Weaknesses
- The proof of Theorem 4.1 (Appendix B.2) uses Eq. (24), which is only introduced "to illustrate the idea behind the AMII estimator". (The statement of Theorem 4.1 does not mention this.)

- I would need (at least) some more details about the Eq. (21) in the proof of Theorem 3.1 to verify the proof.

- It is unclear to me what Eq. (13) is illustrating.

- The AMII estimator looks like doubly robust/debiased estimators proposed in the standard treatment effect estimation literature, but there is no reference or discussion on this.

- The statement of Assumption 4.1 is not clear to me.

- The proposed methods are shown to work well on one dataset, which is a little underwhelming.

- The authors claim that the distributions in Figure 3 "exhibit clear discrepancies". However, it seems like a subjective observation and the criterion of this claim is not clear.

### Questions
Major concerns:
- Does the statement of each theorem include all of its required assumptions? For example, "treatment is assigned at the cluster level, meaning that all units within the same cluster receive the same treatment" (l.168): It is not clear if this is an assumption in the theory. Does Theorem 4.1 only needs Assumption 4.1? Please also see the Weaknesses section.

- Is Assumption 4.1 a condition about the true regression function, or the function used in the estimator? Is the function $g$ known to the estimator?

- Could you discuss the contributions in relation to doubly robust/debiased estimators?

- l. 227, "the MII estimator exhibits substantially lower variance": Why does the MII have lower variance compared to the CAE?

- I suppose the proposed estimators depends on the clusters through $\text{Int}_k$. Do the estimators fail if the clusters are not correct?


Minor issues:
- l.466, "In the absence of the interaction term and 2-hop interference, we find that MII is nearly unbiased": How did the author draw this observation?

- Is the difference between the distribution in Figure 3 statistically significant?

- Punctuation is missing after the equations.

- l.475, "increases exponentially": Isn't it polynomial in $p$?

- l.396, what is the difference between $\text{deg}_i$ and $c_i$?

### Soundness
2

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
5

### Summary
The paper proposes a new estimator for the global average treatment effect (also known as the total treatment effect) in the presence of network interference. The work falls under the framework of cluster randomization, where conventional methods (e.g. difference in means or Horvitz-Thompson) suffer from either bias due to inter-cluster interference or large variance due to randomization. Specifically, the authors propose to focus on the interior units whose potential neighbors are all within the same cluster in order to reduce the variance. Furthermore, the bias from sub-sampling is corrected (or partially corrected when mis-specified) by a counterfactual predictor. Under certain assumptions, the bias bound is carefully elaborated, and more simulations are added for model misspecification.

### Strengths
- **originality**: the proposed method focusing on interior units is new and novel.  
- **quality**: the paper is well-written with solid theoretical results and simulations. 
- **clarity**: the paper is clear and well-explained assumptions 
- **significance**: the significance is relatively fair (see weakness part).

### Weaknesses
Although the paper provides a thorough analysis of the bias of the proposed method, the variance analysis part of the MII estimator is missing, which makes it hard to conduct a real test of the causal effects. More detailed questions are left to the questions part.

### Questions
1. How can we use the proposed method to do hypothesis testing on GATE? Is there any direct way that we can estimate the variance of the MII estimator along with the asymptotical normality property? 
2. I'm a bit confused about the asymptotic scheme the assumption 3.1 is working on. Are we considering a sequence of networks with an incremental number of clusters? It could be more helpful if the authors can show that the assumption 3.1 is easy to satisfy for common random network models. The $o_p(K^{-1})$ bound for the maximum deviation does not sound trivial to me. My intuition for the bound is around $O_p(K^{-1/2}\log K)$ for Erdos-Renyi random networks or similar. 
3. The bound in Theorem 3.1 does provide a convergence rate related to either cluster size or number of clusters. It could be refined to provide more insights on designing an experiment.

### Soundness
4

### Presentation
4

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
The paper addresses the challenge of estimating the Global Average Treatment Effect (GATE) in the presence of network interference, a scenario where a unit’s outcome is influenced not only by its own treatment but also by the treatments of its neighbors. The authors focus on cluster-level randomization and identify limitations in existing estimators and propose new mean-in-interior estimator and augmented mean-in-interior to address the limitations in existing estimators for estimating GATE. Simulation studies are conducted to assess and validate the effectiveness of the proposed estimators and compare with existing estimators across a range of network and treatment settings.

### Strengths
1.  The paper addresses the important problem of estimating the average treatment effect under network interference, which has practical significance for causal inference in many real-world applications.

2. The motivation to extend existing estimands is well-articulated, and the comparisons with prior work are thoroughly examined.
3. Viewing interior nodes as "biased labeled data" and approaching the problem from a semi-supervised learning angle is conceptually compelling.
4. The core idea is clearly explained and easy to follow.

### Weaknesses
1. The proposed estimators rely on strong parametric assumptions, which may limit their robustness and applicability in real-world settings. Moreover, the practical implementation of these estimators could be challenging, particularly in large-scale or heterogeneous networks where model assumptions may not hold or be verifiable.
2. The impact of network interference on the proposed estimators is not thoroughly addressed. In particular, the paper does not fully analyze how varying levels or structures of interference may affect estimator performance, which raises concerns about their robustness in complex or sparse networks.

### Questions
1. The nodes within each cluster are partitioned into interior and boundary nodes. In sparse networks, the number of interior nodes may be relatively small, which could adversely impact the performance and stability of the proposed estimators. Could the authors elaborate on how the estimators behave under such conditions, and whether any adjustments are needed to ensure reliable performance in sparse or low-density network settings?
2. The paper proposes mitigating selection bias in the Mean-in-Interior estimator by training a counterfactual predictor over the entire graph. Could the authors elaborate on the rationale behind this approach?
3. In the proposed approach, boundary nodes are excluded and only interior nodes are retained for estimating the global average treatment effect under network interference. However, interference is inherently present in GATE since it captures both direct and indirect effects propagated through the network. How do the authors account for or mitigate the influence of network interference on the estimation process, given that boundary nodes—where interference effects may be most pronounced—are omitted from the analysis?

### Soundness
2

### Presentation
3

### Contribution
2
