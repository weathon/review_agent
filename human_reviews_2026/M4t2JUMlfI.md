# Neural Collapse in Multi-Task Learning

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 6, 8, 6, 6

## Abstract
Neural collapse (NC) plays a key role in understanding deep neural networks. However, existing empirical and theoretical studies of NC focus on one single task. This paper studies neural collapse in multi-task learning. We consider two standard feature-based multi-task learning scenarios: Single-Source Multi-Task Classification (SSMTC) and Multi-Source Multi-Task Classification (MSMTC). Interestingly, we find that the task-specific linear classifier and features converge to the Simplex Equiangular Tight Frame (ETF) in the setting of MSMTC. In the setting of SSMTC, task-specific linear classifier converges to the task-specific ETF and these task-specific ETFs are mutually orthogonal. Moreover, the shared features across tasks converge to the scaled sum of the weight vectors associated with the task-specific labels in each task's classifier. We also provide the theoretical guarantee for our empirical findings. Through detailed analysis, we uncover the mechanism of MTL where each task learns task-specific latent features that together form the shared features. Moreover, we reveal an inductive bias in MTL that task correlation reconfigures the geometry of task-specific classifiers and promotes alignment among the features learned by each task.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The manuscript discusses the phenomenon of Neural Collapse in the setting of multi-task learning. Specifically, it states that in two specific forms of multi-task learning with shared backbone and task-specific classification heads, adopting the unconstrained features model, neural collapse emerges. Besides theoretical results, also experiments on real datasets are provided.

### Strengths
+ theoretical insights into multi-task learning are a valuable addition to the neural collapse literature
+ the extensive experiments do a good job of confirming that NC does emerge in practice

### Weaknesses
While generally of high quality, I find the work to have some shortcomings in the presentation of the work and the strength of the scientific contribution.

* Presentation

For a predominantly theoretical paper, I found the presentation in several parts insufficiently precise or clear. For example, the main concepts SSMTC-NC (Section 3.1) and MSMTC-NC (Section 3.2) are introduces as the phenomenon that some quantities converge to some constants (often 0) in final phase of network training. However, the associated theorem state that any *global optimizer* of the training objectives *satisfies* SSMTC-NC (Theorem 3.1) and MSMTC-NC (Theorem 3.2), so there is no aspect of optimization or convergence. 
The theorems are phrased without the constraints on the regularization strength that are usually required to prevent trivial solutions. The proofs (lines 764 and 943, respectively) correctly contain the statements that too strong regularization will lead to trivial instead of collapsed solutions.

The description of (theoretical) results would have benefited from giving more insights how the results are relevant and related, e.g. by discussing at least some border cases. 
1) For $T=1$, all results should reduce to the known ones for (balanced, ...) NC with cross-entropy loss. This seems to be the case, but it would be good to explicitly state it.  
2) For SSMTC with $T>1$, nothing seems to prevent that multiple tasks are *identical*. Take, e.g., $y_i^1=y_i^2=\cdots=y_i^T$ for all $i\in[N]$. In that case, the multi-task problem would reduce to identical copies of a single-task problem. $W^1=W^2=\cdots=W^T$ should be a globally optimal solution, and $H$ only needs to be of rank $K$ (which minimizes regularization cost). However this seems incompatible with NC3? The answer might be probably related to Theorem 5.1, where NC3 becomes Correlated-NC3, but the relation of assumptions is not clear to me. Potentially, more answers could be found in the proofs in the appendix, but given their repetitive nature, the manuscript's heavy notation and the too short time that ICLR provides for reviewing this year, I was not able to fully parse the proofs.
3) For MSMTC with $T>1$, each tasks has its own data and only the backbone is shared. But in the UFM, each task has its own  unconstrained features, which remove the central element of "hard parameter sharing" MTL. With independent features, the problem seems to become completely decoupled into independent classification tasks. Since we know that individual UFMs exhibit NC, what is there left to prove? And why would the result still be interesting?


* Scientific contribution

The submissions makes some conceptual, theoretical and experimental contributions. I do see that the conceptual contribution of establishing NC for MTL exist, but I do not find it particular strong or surprising, given the many settings in which final-layer NC has already been proven. The manuscript would benefit from a stronger discussion what benefit the community has from knowing that NC holds in the MTL-UFM setup. The theoretical contribution lies mainly in generalized definitions of NC, and the proofs of the main theorems. The latter mostly follow standard steps from single-task cross-entropy NC. There are new aspects, but e.g. I didn't notice any step in the proofs that I would expect to have broader impact outside of the scope of the work. The experiments are extensive and appear well done. I would encourage the authors to release their source code to allow for reproducibility. The results confirm collapse behavior for real data in many settings, but (as in previous studies) the difference between numeric training of real networks and properties of the global-optimizer in a UFM is quite big. Presumably, the solutions found numerically are not actual global minimizers, but how do they related? Some interpolating experiments could help to reduce this gap (e.g. numeric optimization of the UFM). 


* Minor comments:
- reference (Yu et al, NeurIPS, 2020) is duplicate
- notation such as "5e-3" should better be $5\cdot 10^{-3}$
- line 724: "if we minus a vector/scalar" should be "if we subtract a vector/scalar"
- I found the line of argumentation in the proofs not always ideal to follow. You provide a number of inequalities, and extract conditions saying e.g. "The conditions under which equality holds are as follows:" (line 767). Usually, this would be a way to establish *sufficient* conditions, but for your results of the type "all global minimizers have a certain property", you need also *necessary* conditions. These follow e.g. from the "only if" part of Lemma A.2 (line 701), but such subtleties can easily get lost.

### Questions
I have some question that I hope will help me to understand the work better, in particular its contribution. This list includes the relevant question from the "weakness" section above:

1) For $T=1$, am I right that you recover the same NC properties as in single-task learning?

2) For Theorem 3.1, is it possible that tasks are identical? If yes, why wouldn't the features and weight matrices become identical, violating NC3? 

3) Am I right that the UFM variant of MSMTC decouples the optimization tasks, so the "parameter sharing" aspect of MTL disappears? If yes, what insights into MTL do we still obtain?

4) is there any other (hidden) conditions beyond the bounds on the regularization strength requires for Theorems 3.1 and Theorem 3.2 to hold?

5) how is the setting of Theorem 5.1 not compatible with the assumptions of Theorem 3.1?

6) given the many prior results on NC, how will the community benefit from your result that NC emerges (also) in MTL

7) would you say that your proofs contain any steps of independent interest, i.e. besides their role in proving your theorems?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper analyses Neural Collapse (NC) properties in Multi-Task Learning networks with single and
multiple sources (SSMTC and MSMTC). In the MSMTC scenario, the learned representations and task-specific classifier weight vectors converge to a unified simplex ETF geometry similar to STL. In the SSMTC scenario, each task's weights converges to a separate simplex ETF, and these are mutually orthogonal across tasks. They reveal some interesting properties concerning the
link between the task-specific classifiers as well as the relationship of the shared features and the task-
specific classifiers. Furthermore, they provide theoretical guarantees and extensive empirical validation across different network architectures, data sets and training conditions. Overall, this work extends NC analysis beyond the single-task case and offers valuable new, theoretically grounded insights for MTL.

### Strengths
The manuscript provides a non-trivial theoretical extension of Neural Collapse to multi-task learning. It is proven that any global optimum in both the SSMTC and MSMTC scenarios must satisfy analogous NC properties.

Extensive empirical evaluation is conducted, using multiple architectures, datasets, and multi-task loss weighting strategies. All of these choices are well-grounded in recent literature.

The findings reveal novel insights into geometric structures in how MTL networks organize their features. These insights deepen our understanding of how tasks interact in a shared model. This is important in a literature dominated by heuristic optimization (weighting) strategies without sound theoretical backdrop. 

The paper shows that the degree of task relatedness can shape the learned feature geometry, which helps in treating tasks as heterogeneous and understanding when and why MTL works.

### Weaknesses
The discussion of practical implications is fairly limited. A better connection to practical MTL challenges could be made. For instance, it is unclear how these collaps properties might help improve training procedures. Linking the insights from this paper to issues such as task imbalance, negative task transfer and MTL architecture design would be very valuable.

### Questions
Do the observed NC structures also manifest on hold-out data? Can you say something about how NC relates to generalization in this context? In STL, NC has been linked to improved generalization. Any insight into how NC on training data translates to performance or structure on test data would be appreciated.

In SSMTC, the orthogonality between task-specific classifier subspaces seems elegant. In reality, deep features are typically shared across tasks. Do you view this orthogonality as a theoretical idealization, or do you think partial orthogonality might actually be beneficial in practice (e.g., for reducing task interference)?

Do any of the differences observed between MTL and STL settings suggest concrete benefits of training tasks simultaneously (as done here) versus training the same tasks independently on separate models?

The theoretical proofs rely on the Unconstrained Feature Model (UFM) and the existence of a global optimum. In practice, deep networks rarely reach global optima. How sensitive are your predicted NC structures to this? For instance, does partial or approximate collapse occur under suboptimal convergence?

Your linear combination view seems to suggest that the shared representation in MTL is fundamentally additive. Do you agree with this?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work formalizes two settings, SSMTC and MSMTC, under the unconstrained feature model and prove that any global minimizer exhibits NC geometry extended to multi-task, introducing new NC measurement, SNC1–SNC5 for SSMTC and MNC1–MNC4 for MSMTC, to quantify convergence to the predicted NC geometry. Experiment is verified over multiple backbone and datasets. To my knowledge, the formulations should be new, and the contribution over decorrelation across tasks & correlation-aware alignment is also supposed to be new.

### Strengths
- The overall work is well written. The setup and background sufficiently familiarize readers with the problem.
- I briefly reviewed the appendix proofs and believe the main claims and high-level theoretical results are sound. Some clarity issues remain; please see my questions.
- The experiments are extensive across different setups and backbones and are presented under multiple MOO weighting strategies. I believe most of the claims are verified.
- As noted in the summary, the theoretical contribution is new and the derivations appear solid (I did not spot any major errors). However, the technical novelty is somewhat limited, as the work reads as an extension of single-task neural collapse rather than a new paradigm.

---

Overall, I believe this is a new contribution in theory towards multi-task NC, even though it might be somehow incremental to view from technical theoretical contribution. My current assessment is boardline; I will re-evaluate it based on the rebuttal. Still, please address the concerns in weakness section.

### Weaknesses
- The method assumes balanced labels. I suggest to add discussion of how label imbalance and skewness would affect your modules and final results, through some specific quantitative analysis.

- Following from the previous point, although the balanced-label assumption aligns with prior work, the datasets used are relatively standard and small-to-medium for probing NC geometry. The effect on large, real-world datasets remains unclear and is therefore a limitation. Please consider adding at least one larger split to demonstrate robustness for your readers. For the medium-size experiment in Appendix D.4, report mean & std in the curve across multiple seeds. I also suggest moving the real-dataset experiments on ImageNet and CelebA from the appendix into the main paper, replacing some synthesized-data experiments.

- Several constraints are hard to interpret in terms of realistic, practical impact, and I have concerns about the UFM setup. See Questions for details.

- The clarity of writing and presentation should be improved. See Questions for details.

### Questions
- For thm3.1/3.2. All the NC results only sound given nontrivial $\lambda_{H}\lambda_{W}$ less than the threshold, but as you derived in line 764, when $\lambda_{H}\lambda_{W}$ is bigger than the threshold, the global minimizer is simply a trivial solution, making the NC properties no longer hold. How should I interpret this in terms of actual applicability? I suggest to add several real examples of $N$ and $K$ from your actual dataset to justify to what extent such condition would hold.

- For the whole UFM setup, I am concerned that the actual network you trained is not optimizing the UFM objective you analyzed, which should be a deep shared feature extractor plus per-task heads with CE. The UFM analysis in multi-task form breaks the coupling between tasks, which seems like a toy model.


Presentations:

- Please properly use \citet for better readability, e.g., line 181, line 182-183, and all the citations from Appendix E.

- Please consider to add all eq ref in your proof. Some of them are not added, and it reads somehow inconvenient, e.g., line 954 ref to inequality m. 

- Please consider add a table to present all the metrics in section 4.3.

- typo. Line 114, SSTMC-NC

### Soundness
3

### Presentation
2

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
This paper investigates the phenomenon of neural collapse (NC) in multi-task learning (MTL), extending the existing understanding of NC beyond single-task settings. The authors propose two novel MTL-specific NC formulations—SSMTC-NC and MSMTC-NC—and provide both theoretical guarantees under the Unconstrained Feature Model (UFM) and extensive empirical validation across multiple datasets and architectures. They further explore how task correlation influences the geometry of task-specific classifiers and shared feature representations.

### Strengths
1.The paper is the first to systematically extend neural collapse to multi-task learning, introducing two well-defined geometric configurations (SSMTC-NC and MSMTC-NC).

2.The authors provide rigorous proofs showing that both SSMTC-NC and MSMTC-NC emerge as global optima under the UFM setting.

3.Experiments are conducted across a wide range of datasets (e.g., Multi-MNIST, CIFAR100-Split, CelebA) and architectures (ResNet, VGG), with additional studies on task weighting strategies and task correlation.

### Weaknesses
1. Most theoretical results assume balanced class and task distributions, which may not hold in practical MTL scenarios.

2. Key figures (e.g., Figure 1, 2) do not include confidence intervals or measures of variability, which could mislead readers about the robustness of results.

### Questions
1.How would the NC phenomena change in highly imbalanced MTL settings, both in terms of class and task distributions?

2.Have the authors considered evaluating the proposed NC metrics in few-shot or cross-domain MTL scenarios to assess generalizability?

### Soundness
3

### Presentation
3

### Contribution
3
