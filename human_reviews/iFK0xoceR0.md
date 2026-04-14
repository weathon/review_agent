# Provably Robust Explainable Graph Neural Networks against Graph Perturbation Attacks

- Decision: Accept (Poster)
- Scores: 6, 5, 8, 6

## Abstract
Explaining Graph Neural Network (XGNN) has gained growing attention to facilitate the trust of using GNNs, which is the mainstream method to learn graph data. Despite their growing attention, Existing XGNNs focus on improving the explanation performance, and its robustness under attacks is largely unexplored. We noticed that an adversary can slightly perturb the graph structure such that the explanation result of XGNNs is largely changed. Such vulnerability of XGNNs could cause serious issues particularly in safety/security-critical applications. In this paper, we take the first step to study the robustness of XGNN against graph perturbation attacks, and propose XGNNCert, the first provably robust XGNN. Particularly, our XGNNCert can provably ensure the explanation result for a graph under the worst-case graph perturbation attack is close to that without the attack, while not affecting the GNN prediction, when the number of perturbed edges is bounded. Evaluation results on multiple graph datasets and GNN explainers show the effectiveness of XGNNCert.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work first studies the robustness of Explaining Graph Neural Network (XGNN) under graph perturbation attack and proposes an explainable defense mechanism. By generating hybrid subgraphs in which most of the designed voting classifier and voting explainer will give the correct prediction, this method avoids the XGNN performance degradation on both GNN's prediction and explanation under a worst-case attack. At the same time, the author provides a theoretical guarantee of the proposed defense's robustness.

### Strengths
1. The writing is well-structured, and the contributions are clearly stated.

2. The author provides a detailed explanation of the attack, the defense, and the theoretical guarantee.

### Weaknesses
1. The author claimed their method can be applied to any GNN classifier and explainer but did not show any results on different models. Only one type of classifier and explainer is used in the experiment.

2. A lot of space is spent on parameter analysis of the proposed method, which may be combined as an ablation study.

3. There is an explanation accuracy gap between XGNNCert and Baselines on datasets Benzene and FC in Table 1, where XGNNCert performs quite worse.

4. The text in Figure 2 may need to be more visible.

### Questions
1. The author mentions the attacker can input a perturbed graph into the drug analysis tool. How can the attacker access the data in the realistic application?

2. The author mentions that both the GNN classifier and the GNN explainer are essential for making correct explanations. Can these two models be combined into one model to defend against the graph perturbation attack rather than ensuring both model's accuracy?
 
3. While the author emphasizes the robustness benefits of XGNNCert, can the author show more results against different attacks? For example, shorten the content of 4.2.1 and 4.2.2 and expand 4.2.3 by showing more results on more GNN models and comparing the proposed method with more baselines.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper proposes a certified defense method to mitigate the risk of an attacker who aims to mislead the GNN explanation. In order to achieve this goal, the calculate the maximum perturbation size to perturb in order to alter the prediction (explanation) of a graph.

### Strengths
1. The paper is well written. 
2. The authors provide a reasonable motivation for the study (such as giving an motivating example in the introduction). 
3. The proposed method empirically efficacy to obtain certified robustness and it would not hurt the original performance. 
4. The method is potential to be applied for various GNN models.

### Weaknesses
Although the authors give a good motivation example (in the introduction section), I am still not fully convinced by the importance of the proposed study. For example, if one GNN builder aims to explain or understand their GNN model's prediction via GNN explainers, will there still be a high risk that the graph data is perturbed by an attacker? An perturbation operation could possibly easily identified by the GNN builder. 

Besides, I am also wondering the tightness of the proposed certified robustness. In Figure 5, the authors only provide the calculated perturbation bounds. How close are them to the real minimum perturbation budget?

### Questions
Plz see the weakness.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper introduces a new architecture for explainable graph learning that allows to additionally derive a certificate regarding the set of explanations (edges) w.r.t. edge perturbations. This is achieved by splitting the graphs into subgraphs and derive an ensemble-based explainer. In particular, the well-known robustness certification strategy for majority voting classifiers is extended to the set case. The effectiveness of the method is demonstrated on several graph classification datasets.

### Strengths
* In general, the paper is well-written and nice to follow. Next to the well-written main parts, I want to additionally highlight the very good and thorough overview of related work and contextualization of this work in it given in Section 5.
* With a certifiably robust XGNN the paper seems to tackle a relevant problem given the brittleness of XGNNs to adversarial attacks.
* Extending robustness certification (especially w.r.t. ensembles) to explainable sets is non-trivial, conceptually, as well as w.r.t. the proof strategy.
* Through appropriate experiments, the effectiveness of the method to provide a certain robustness to structure changes is shown. Furthermore, the experiments nicely show the tradeoffs between classic XGNN methods and XGNNCert.

### Weaknesses
1. Currently, the experiments are not reproducible as experimental details are missing about $(i)$ the training strategy (even though Line 312 links to Appendix B for this, Appendix B does not include any experimental details), and $(ii)$ architectural details and hyperparameters of the used base classification and explainer GNNs. This is made even more critical as $(1)$ there is no code submitted for the review or indication of releasing code upon acceptance, and  $(2)$ no reproducibility statement provided. 
2. The $(M_\lambda, \lambda)$ robustness definition together with the remark in Line 148 is currently confusing. In particular, for any $\lambda$, there can be many associated $M_\lambda$ such that the network is still $(M_\lambda, \lambda)$ robust. Thus, $M_\lambda$ is not uniquely indexed by $\lambda$. However, by indexing $M$ with $\lambda$, stating that $M_\lambda$ is the certified perturbation size, and explicitly remarking it depends on $\lambda$, it gives the impression as if it should be a unique function of $\lambda$ and in the first read, it was unclear to me if the authors therefore actually mean the maximal $M_\lambda$ for a particular $\lambda$. I think the confusion could be resolved by removing the index $\lambda$ (it is evident that $\lambda$ and $M$ have a certain connection and tradeoff) and optimally, also introduce the concept of a maximal $M$ after the definition with a different letter (e.g., $M^*$ as used in Thm. 3, or now $M_\lambda$, as the the maximal $M$ is uniquely determined by a given $\lambda$). 
3. The goal of the work stated in lines 141-144 does not align with the contribution of the paper. In particular, the paper designs a concrete XGNN architecture based on ensembles to be able to (for the first time) derive robustness certificates. However, Eq. 1 defines a very different goal, namely optimizing over function spaces defining the XGNN and GNN classifiers, to find those functions leading to the highest certified robustness. However, the optimization over function spaces to maximize certified robustness is never mentioned or tackled in the subsequent paper. My recommendation is to remove Eq. 1, as the problem tackled by the paper is already very well motivated and formulated before providing this very different goal description. Otherwise, questions such as "what function spaces do we optimize over", or "do we have convergence guarantees", or similar, need to be asked and answered. As a third alternative, one could not state it as the goal of the current work, but a general goal the whole field/literature tries to answer/converge to.
4. Generating subgraph indexes via the proposed hash mapping has first been proposed and done by Xia et al. (2024), I would expect to make clear somewhere that their hashing is used, potentially just by stating this and citing Xia et al. (2024) in the beginning of the second paragraph in Section 3.1 (e.g., "We use the hash function (e.g., MD5) to generate the subraph indexes as done in Xia et al. (2024)."). Furthermore, regarding majority voting-based certified defenses, the work of Wang et al. (2022) already divides the input data into non-overlapping parts (in contrast to what the paper mentions in lines 526/527). Even though this work is qualitatively different, I expect the work to be cited and the statement corrected when this is explicitly discussed in lines 526/527.
5. The method is not permutation invariant. The authors discuss this in Appendix C, which however, is not linked in the main draft (please link to it). GNNs are usually developed with the goal to be permutation invariant (Hamilton 2020), and to me, it feels odd if an XGNN explanation is dependent on the concrete node order. In particular, node-order sensitivity can improve expressivity (and generalization) in the face of non-discriminatory node features (Huang et al. 2022), but I do not see that this property is particularly relevant for the developed XGNNCert as all used datasets seem to have discriminatory node features (nor any argument given that the concrete usage of permutation sensitivity improves expressivity). Addressing the permutation invariance is not critical for me to recommend acceptance, but I would like to at least see an experiment showing how brittle or stable the performance (classification and explanation-wise) of XGNNCert is to different permutations of the adjacency matrix. 
6. From the sentence at line 93-94, I first had the impression that the method can be applied to node classification tasks. However, Appendix C (again not linked) states that this is not possible without adaptation. I recommend to make this clear in the main draft when introducing the background or method and link to the discussion in Appendix C.
7. In Section 3.3 the authors mention multiple times a statement similar to the one found in Line 276: "$\bar{\mathcal{E}}_M$ [are] the edges in $\bar{\mathcal{E}}$ with top-$M$ scores in $n^\gamma$". Until reading and understanding the proof it was unclear to me if such a statement means either $(i)$ the edges in $\bar{\mathcal{E}}_M$ that are among the top-$M$ edges in $n^\gamma$ given all edges (i.e., $|\bar{\mathcal{E}}_M|\le M$); or $(ii)$ $\bar{\mathcal{E}}_M$ are the top-$M$ scored edges in $\bar{\mathcal{E}}$, i.e. $|\bar{\mathcal{E}}_M|=M$. Making this clear before providing Thm. 3 would greatly improve readability of Section 3.3 and the Thm. 3 in particular.
8. Figure 4 and 5 feel repetitive to Figure 3. Maybe, it could be more interesting to show the impact of $p$ and $\gamma$ also in the main draft (i.e., Figures similar to Fig. 9 & 10). Also note that in a first read, I was overwhelmed by Line 431 linking to Figures 3-11 all at once and it was not clear to me, if I should look at that figures now and how to find all the in-between figures.

I think the paper has the potential to be a good contribution to the ICLR community. If the reproducibility weakness is addressed, I will raise my rating to a borderline score. If additionally my other weaknesses and questions are appropriately addressed, I am considering changing my score to accept (8).

### Questions
1. How do the hyperparameters $p$ and $h$ effect prediction accuracy? I really like Tables 1 & 2 and think having a similar investigation for $p$ and $h$ in the Appendix would be interesting.
2. Is certification possible if an edge in the ground truth explanation set is perturbed? In particular, I'm not very convinced of the stealthiness argument in line 102, as a defender would not have access to the ground truth graph, but only the perturbed version (and of course, does not have access to the ground truth explanatory edges, as this would make the method otherwise obsolete). My thinking would be yes, as from a defender perspective the $\mathcal{E}_k$ are already derived from the perturbed graph and the $\lambda$ guarantee holding for any graph reachable through $M^*$ edge changes.
3. Where does the -1 in the robustness guarantee come from? Shouldn't from the proof $M \le \lfloor \frac{n_y-n_b+\mathbb{I}(y<b)}{2}\rfloor$ suffice? What am I missing?
4. Regarding the experiments performed in Table 6, what is the instability of XGNNCert under the same experimental setup?
5. In the related work, can the authors include one or two sentences about certification of explainable AI in a non-graph context? This would be interesting to the reader, as one could ask the question of what approaches are already out there and if these are directly applicable to the graph domain. 

Minor
* Figure 1: "Origianl Prediction" -> "Original Prediction" (2x)
* Line 12: "learn graph data" -> "learn on graph data"
* Line 13: "its robustness" -> "their robustness"
* Line 18: "XGNN" -> "XGNNs"
* Line 53: It could help to first define a certified defense before using the term.
* Line 265: Writing $|x|$ to denote the number of elements in $x$ is confusing, as it is not written or indicated that $x$ is a set. 
* Line 278/279: "with" -> "within"
* Line 354: $M^*$@$\lambda$ <- is this intended?
* Line 363/364: "GNN explaniner" -> "GNN explainer"


**References**  
Hamilton, "Graph Representation Learning", McGill University 2020  
Huang et al., "Going Deeper into Permutation-Sensitive Graph Neural Networks", ICML 2022  
Wang et al., "Improved Certified Defenses against Data Poisoning with (Deterministic) Finite Aggregation", ICML 2022  
Xia et al., "GNNCert: Deterministic Certification of Graph Neural Networks against Adversarial Perturbations", ICLR 2024

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
This paper proposes a novel certified defense framework for GNN explanations, named XGNNCert. In particular, XGNNCert generates hybrid subgraphs using subgraphs of the target graph and subgraphs of the complete graph. Based on the hybrid subgraphs, XGNNCert adopts majority voting based GNN classifier and explainer as the backbone. The authors derive a theoretical guarantee on the certified robustness of GNN classifier and explainer and provide the formula to compute the perturbation budget. Experimental evaluations on various datasets demonstrate that XGNNCert maintains the explanation accuracy and guarantees the preservation of critical explanatory edges under adversarial perturbations.

### Strengths
- Recent works reveal that GNN explanation is vulnerable to adversarial attacks. This paper is the first to propose a certified defense framework against the attacks.
- The certification has a deterministic guarantee, which makes XGNNCert reliable in real-world scenarios.
- The experimental results show that XGNNCert achieves desirable certification compared with other empirical defense methods.
- The paper is well-organized and the writing is easy to follow.

### Weaknesses
- The certification relies on the majority voting mechanism. These can lead to the global information of the target graph being corrupted. For example, in experimental results (Table 1 and Table 2), the performance of XGNNCert in FC dataset decreases consistently compared with the original GNN.
- In hybrid subgraph generation, how to ensure that the size of $\{\mathcal{E}^i\}$ is uniform? If sizes of subgraphs differ distinctly from each other, it would be better to determine the parameter $p$ more carefully, such as being dependent on each subgraph.
- XGNNCert requires to generate subgraphs of the complete graph. It can be difficult to scale due to the extra memory cost. This part seems not to be included in the complexity analysis.
- XGNNCert poses challenges on hyperparameter tuning in real-world implementation. The performance can be sensitive to some hyperparameters such as $T$, $p$, and $\gamma$, while all of them should locate in a specific range.

### Questions
Please see Weaknesses

### Soundness
3

### Presentation
3

### Contribution
3
