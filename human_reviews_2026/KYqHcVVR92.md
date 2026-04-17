# Fast SDP certification of neural networks : towards large multi-class datasets

- Decision: Reject
- Scores: 4, 4, 2, 6

## Abstract
We present a new quadratic model for the certification problem in adversarial robustness, which simultaneously accounts for all possible target classes. Building on this model, we propose a novel semidefinite programming (SDP) relaxation for incomplete verification. A key advantage of our approach is that it certifies robustness in a single optimization, avoiding the need for a separate resolution per class. This yields a significant computational speed-up and enables scalability to large datasets with many classes. To further gain in efficiency, we also propose an effective pruning strategy of active neurons, thus reducing the problem dimensionality and accelerating convergence.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes a single (untargeted) quadratic certification model by introducing selector binaries $\beta_j$ so that one optimization certifies robustness across all possible targets, and then derives an SDP relaxation to solve it.  To improve scaling, the authors use chordal decompositions, pruning of stable active (and inactive) neurons and compensating for the missing quadratic couplings via McCormick-based cuts. Experiments on MNIST fully connected feed-forward networks and a 67-class composite benchmark suggest higher certified rates and/or faster runtime with respect to other target-based SDP baselines.

### Strengths
Considered problem is relevant, as robust DNN-driven systems are crucial for successful deployment. The pruning of *active* neurons (not only inactive ones) and the chordal decomposition are sensible choices to reduce PSD block sizes in practice.

### Weaknesses
Modeling disjunction via binary variables is a relatively standard and well-known approach. The contribution lies more in the SDP instantiation and pruning/cut design, but exhaustive ablation studies illustrating the clear impact of adding specific individual features to the SDP relaxation are missing.

### Major points

- Experimental results are very limited. Only relatively small feed-forward networks are used (up to 1800 hidden neurons). Tightness of the bound and/or scalability of the method with respect to different sizes of perturbation regions is not assessed.
- Presentation: The paper is difficult to follow because key ideas are scattered, notation is overloaded and inconsistent (e.g., superscripts can refer to vectors or specific neurons, $u$ can mean both untargeted and unstable, and many others), with many typos and language issues (e.g., equality in lines 336–337 is mathematically unsound). Explanations for understanding experimental results only come after a few paragraphs or subsections, etc. Overall, a thorough rewrite is necessary before the contributions can be clearly and fairly assessed.

### Minor points

- For most of the references, `\citep{}` should be used to facilitate reading.
- Be consistent in writing. For instance, formulations *ReLU*, *relu*, and *Relu* all appear in the manuscript. Same with *semi-definite*, *semidefinite*, and *semi definite*. The dataset is sometimes denoted $\mathcal{D}$ and sometimes $\mathcal{X}$. Introducing $\mathcal{K}$ is probably not necessary.
- Figure 1 is never referred to in the main text.
- Property 3 is trivial and already explained in words before being formally written down. I suggest removing it. The same can be said about Theorem 1.

### Questions
1. Why consider a manufactured dataset with 67 classes rather than a real multiclass dataset (e.g., CIFAR-100 with small convolutional networks)?

2. Wouldn't an explicit proof (even in the Appendix) of Proposition help readers understand the pruning method and, consequently, the structure of the relaxation?

3. What do the percentages in Figure 2 represent?

4. Why not formulating the untargeted objective for other approaches as well and comparing them under this common objective? This would isolate the impact of each modeling choice.

5. For targeted approaches from the literature, how is the target index chosen? Are you solving for all possible targets? Please clarify.

6. For your proposed relaxations, is it possible to analyze loss in bound tightness as a function of, say, attack magnitude, network depth, or fraction of pruned neurons? This analysis could strengthen the paper.

Finally, Your approach selects $\lfloor pn_k\rfloor$ inter-layer RLT cuts per neuron using the top magnitudes $|W^{j}_{k+1}|$,
 computed with $n_k$.  

Why is this preferable to selecting $\lfloor p n_k^{u}\rfloor$? What is the sensitivity of certification rate and runtime w.r.t. $p$? Is there an empirical threshold $p_{\min}$ below which tightness and performance degrade notably?

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
This paper aims to accelerate incomplete verification of adversarial robustness by (i) casting an untargeted single quadratic objective that accounts for all classes, (ii) deriving a semidefinite relaxation augmented with tightening cuts, and (iii) introducing a pruning rule for provably stable-active neurons to reduce computational load.

### Strengths
(1) The move from per-class targeted certification to a single untargeted quadratic program with class-selection binaries $\beta_j$ is clear. Theorem 1 shows equivalence to minimizing over targets, justifying the single-SDP approach.
(2) Tight, well-motivated relaxation: The SDP includes chordal decomposition, McCormick envelopes for $\beta \times z$, and two new coupling inequalities ((26)–(27)) tailored to class selection; these are technically sound choices for tightening.
(3) The paper prunes stable active ReLU units and replace them with linear expressions, then compensates the missing quadratic terms with linear bounds. This is practical contribution with qualification in Proposition 2. 
(4) The results in Figure 2 strongly demonstrate the scalability with respect to the number of classes.

### Weaknesses
(1) RLT cuts are formed by multiplying valid linear inequalities to obtain quadratic ones; as the number of RLT constraints grows, the candidate set can blow up combinatorially, risking intractability. The paper chooses a cut percentage  $p$ by heuristic per architecture. Please report certificate rate and runtime as functions of  $p∈(0,10,30,60,100)$% and identify the “sweet-spot” $p^*$. 

(2) Only fully connected networks are evaluated. Extensions to CNNs/ResNets (where SDP relaxations are common) are missing. Even a small CNN/ResNet ablation would clarify whether the method’s gains persist under weight sharing and deeper topologies.

(3) Baseline coverage is weak. Beyond SDP, include LP relaxations, $β$-CROWN, and other established incomplete verifiers, under matched perturbation radii. Please sweep multiple $ϵ∈(1/255,2/255,4/255,8/255)$ and report certified accuracy–runtime curves.

(4) Section 5.2 claims that $\beta-$CROWN  is unable to tighten the bounds to eliminate all possible target classes as the number of neurons increases, but no experiment demonstrates this trend. Provide plots of (i) average margin lower bound vs. width/depth, and (ii) fraction of eliminated classes vs. network size.

(5) Table 2 shows fewer certified cases with pruning, but it remains unclear why. Please quantify: (i) certified fraction vs. number of unstable ReLUs, (ii) average bound gap (lower–upper) before/after pruning, and (iii) failure mode taxonomy (e.g., timeout, solver infeasibility, loose pre-activation bounds). Wider validation across architectures/datasets is encouraged.

### Questions
(1) How does certification rate trade off with the RLT percentage when total time is capped? Do you have empirical heuristics for tuning $p$ (e.g., based on instability rate, Lipschitz proxies, or pre-activation interval widths)?

(2) Can you quantify the relaxation gap introduced by the pruning rules (Eqs. 28–31)? Under what conditions does this relaxation change the certificate (i.e., converts a would-be certificate into a failure)? Are these rules implicated in the observed failures?

(3) Extend Fig. 2 with $β$-CROWN-based pruning curves vs. class count to show how complexity and certified accuracy scale. 

(4) How does the approach perform on CIFAR-10? Include the same ablations (RLT percentage, pruning on/off, baseline comparators, and $\epsilon$) for a like-for-like comparison. 

Overall, widening the evaluation (architectures + baselines + $ϵ$) and adding the ablations above would substantively strengthen the paper’s empirical support and clarify where the proposed method is most beneficial.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
While most neural network verifiers rely on linear relaxations of the networks to be verified, alternatives based on semidefinite relaxations provide tighter bounds, enabling more properties to be verified. However, Semidefinite Programming (SDP)-based verifiers usually need to solve one optimisation problem for each incorrect class, leading to high costs and slow verification. This work proposes an alternative formulation of the neural network verification problem as a quadratic program which verifies the robustness of a neural network across all classes simultaneously. The authors then relax this formulation to an SDP and introduce additional cuts based on McCormick envelopes. They further propose a method for pruning stably active neurons in a verification problem and, since the proposed method would break the chordal decomposition of the SDP, suggest a relaxation of the pruned problem which preserves the chordal structure. The proposed method is evaluated on a number of standard benchmarks and is shown to outperform previous verification approaches in terms of both runtime as well as the number of verified instances.

### Strengths
- Neural network verification is an important research topic
- The requirement to solve a number of separate optimisation problems is a major issue in SDP-based verification and tackling this is an important contribution.
- The experimental evaluation shows a significantly improved performance compared to the baselines that were evaluated

### Weaknesses
- To me, the biggest weakness of this work is the empirical evaluation. The results look impressive, but the authors use outdated baselines and do not compare against newer verification approaches. The settings used for e.g. $\beta$-CROWN are also questionable.
  - The experimental evaluation on $\beta$-CROWN seems unfair since the verifier is only run for very short time budgets (2-5 seconds) while the authors' proposed method is run for up to ~1600 seconds. Since $\beta$-CROWN is a complete verifier, a fair comparison would enable its branching and run it for the same time budget as $SDP_u$. The fact that the numbers reported here are not representative is also evidenced by comparing them to e.g. those reported by [6] where the gap between $\beta$-CROWN and the SDP verifiers is significantly smaller than the gap reported in this work.
  - $\beta$-CROWN has been improved by the introduction of general cutting planes generated by MILP solvers in [3] and branch-and-bound-inferred cutting planes in [4]. The method proposed by the authors should be evaluated against these newer works to benchmark how well it actually performs and not against the older $\beta$-CROWN. The authors conduct their experiments on very small models, I understand that this is standard in SDP-based verification so I do not hold this against them. However, it does seem quite likely that a MILP solver would scale to these sizes and would therefore be able to produce effective cutting planes in GCP-CROWN.
  - The authors do not compare their approach to newer SDP-based verifiers such as [6, 7, 8, 9], I know that at least for [8] the code has been made available so a comparison would be easy.
  - Given the small size of the networks which benchmarks are run on, I do wonder whether improved MILP verifiers or hybrid ones such as [2] would perform on these benchmarks
  - The paper's primary motivation is scaling to "large multi-class datasets", which also implies convolutional neural networks (CNNs). However, all experiments are conducted on small, fully-connected networks. The paper would be much stronger if it included at least a preliminary study on a small CNN.
- The related work section is missing multiple important references
  - The work cites early MILP verification approaches, but omits multiple later works on MILP-based verification which significantly improve upon the early (naive) approaches such as [1, 2]
  - All of the state-of-the-art work on incomplete verification is missing, including for example GCP-CROWN [3], BICCOS [4] and Marabou 2.0 [5]
  - A number of recent works on SDP-based verification are missing [6, 7, 8, 9]
- The removal of stably active neurons in a neural network verification context has previously been proposed by [10]. The approach proposed by the authors in this paper seems more general than the previously proposed method, but I think the previous method should still be cited. The previous method applies to fewer neurons but would not break the chordal sparsity pattern and would therefore not require the additional relaxation by constraints (28-31), so it would be interesting to see a performance comparison between the two
- The authors should provide some details on the networks that they train in Section 5.4 in terms of size. Being able to verify a neural network with 67 classes using SDP verification is great, but I wonder what the size and architecture of that network is.

### Minor weaknesses and typos
- Line 336: The authors say that "unmodeled quadratic terms appear in Constraint (5)". It should be made clearer that this is because of the removal of active neurons from the network which leads to new cross-layer dependencies, at the moment this is a bit difficult to understand.
- Line 108: Incomplete verifiers are derived into a wide variety of approaches --> Incomplete verifiers are **divided** into a wide variety of approaches
- Line 138: $W_K^j$, the $j^{th}$ row of matrix $W_K^j$ --> $W_K^j$, the $j^{th}$ row of matrix $W_K$
- Line 210: The triangular constraint (14) introduced in Ehlers (2017) tighten --> The triangular constraint (14) introduced in Ehlers (2017) tighten**s**
- Line 267: the DNN satisfy Property 2. --> the DNN satisf**ies** Property 2.
- Line 308: leverage the specific structure certification problem --> leverage the specific structure **of the** certification problem
- Line 357: where coefficient of $C_{k−1}$ are a linear combination --> where **the** coefficient**s** $C_{k−1}$ are a linear combination

### References
[1] Botoeva, E., Kouvaros, P., Kronqvist, J., Lomuscio, A. & Misener, R. (2020) Efficient Verification of ReLU-Based Neural Networks via Dependency Analysis. In: Proceedings of the AAAI Conference on Artificial Intelligence. 3 April 2020 pp. 3291–3299. doi:10.1609/aaai.v34i04.5729.

[2] Liao, Y., Genest, B., Meel, K. & Aryaman, S. (2025) Solution-aware vs global ReLU selection: partial MILP strikes back for DNN verification. doi:10.48550/arXiv.2507.23197.

[3] Zhang, H., Wang, S., Xu, K., Li, L., Li, B., Jana, S., Hsieh, C.-J. & Kolter, J.Z. (2022) General Cutting Planes for Bound-Propagation-Based Neural Network Verification. doi:10.48550/arXiv.2208.05740.

[4] Zhou, D., Brix, C., Hanasusanto, G.A. & Zhang, H. (2024) Scalable Neural Network Verification with Branch-and-bound Inferred Cutting Planes. Advances in Neural Information Processing Systems. 37, 29324–29353.

[5] Wu, H., Isac, O., Zeljić, A., Tagomori, T., Daggitt, M., Kokke, W., Refaeli, I., Amir, G., Julian, K., Bassan, S., Huang, P., Lahav, O., Wu, M., Zhang, M., Komendantskaya, E., Katz, G. & Barrett, C. (2024) Marabou 2.0: A Versatile Formal Analyzer of Neural Networks. In: Computer Aided Verification: 36th International Conference, CAV 2024, Montreal, QC, Canada, July 24–27, 2024, Proceedings, Part II. 25 July 2024 Berlin, Heidelberg, Springer-Verlag. pp. 249–264. doi:10.1007/978-3-031-65630-9_13.

[6] Lan, J., Brückner, B. & Lomuscio, A. (2023) A Semidefinite Relaxation Based Branch-and-Bound Method for Tight Neural Network Verification. Proceedings of the AAAI Conference on Artificial Intelligence. 37 (12), 14946–14954. doi:10.1609/aaai.v37i12.26745.

[7] Lan, J., Zheng, Y. & Lomuscio, A. (2023) Iteratively Enhanced Semidefinite Relaxations for Efficient Neural Network Verification. Proceedings of the AAAI Conference on Artificial Intelligence. 37 (12), 14937–14945. doi:10.1609/aaai.v37i12.26744.

[8] Chiu, H.-M. & Zhang, R.Y. (2023) Tight Certification of Adversarially Trained Neural Networks via Nonconvex Low-Rank Semidefinite Relaxations. In: Proceedings of the 40th International Conference on Machine Learning. 3 July 2023 PMLR. pp. 5631–5660. https://proceedings.mlr.press/v202/chiu23a.html.

[9] Ueda, R., Sato, T., Kobayashi, K. & Nakata, K. (2025) Interior-Point Vanishing Problem in Semidefinite Relaxations for Neural Network Verification. doi:10.48550/arXiv.2506.10269.

[10] Serra, T., Kumar, A. & Ramalingam, S. (2020) Lossless Compression of Deep Neural Networks. In: E. Hebrard & N. Musliu (eds.). Integration of Constraint Programming, Artificial Intelligence, and Operations Research. 2020 Cham, Springer International Publishing. pp. 417–430. doi:10.1007/978-3-030-58942-4_27.

### Questions
- How does the pruning method proposed by the authors compare to that proposed by Serra et al. in [10]?
- Why is $\beta$-CROWN only run for 2-5 seconds in the empirical evaluation?
- How does the method proposed by the authors compare to newer SDP-based as well as other NN verification works?
- What architecture is used for the self-trained models in Section 5.4?

### Soundness
2

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
Paper develops an SDP relaxation of neural network verification methods intended to verify a large number of classification/prediction criteria by considering the criteria holistically as a single problem rather than individual problems.

### Strengths
1.	Advances the state of practice in relaxation-based approaches for NN verification based on SDP relaxations.
2.	Increases the scale of NN’s were SDP relaxations can be used.

### Weaknesses
1.	The literature review is generally restricted to a single paper from 2022 and the rest being older than that.  There is quite a bit of more recent work, esp. https://arxiv.org/abs/2506.06665, which has very much improved the state of the art of relaxations to SDP for NN verification.  Oddly, this paper appears in the reference list, but is not referred to in the text. So, I am a bit concerned that this and other recent work is not compared against.
2.	Some parts of the model are unclear.  For example, it is not clear whether or not the beta (binary) variables are relaxed or not.  From the notation, I think the beta variables remain binary.  However, my understanding is that Mosek does not support mixed integer SDPs.  Either way, I have a couple questions below.

### Questions
1.	Are the binary variables relaxed or binary in the final relaxation?
a.	If relaxed, I would expect the relaxation to be weaker than the enumeration of all classes using SDP with the cuts that are specific to each class.  (because the optimization can assign fractional values to beta).  Is this the case? Do you have CPU and solution quality comparisons between your model enumerated per class vs. your one large model?  I am not completely sure if SDP_t is exactly this or not.
b.	If not relaxed, is the main value of the combined model that it can take advantage of the branching strategies and pruning capabilities of the mixed integer solver, and not have to do a complete enumeration of all classes (e.g., the search tree is essentially one layer below the root node, with each leaf essentially corresponding to one verification class and many leaves pruned and never evaluated)?

### Soundness
2

### Presentation
2

### Contribution
3
