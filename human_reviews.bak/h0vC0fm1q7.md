# Sensitivity Verification for Additive Decision Tree Ensembles

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 8, 6

## Abstract
Tree ensemble models, such as Gradient Boosted Decision Trees (GBDTs) and random forests, are widely popular models for a variety of machine learning tasks. The power of these models comes from the ensemble of decision trees, which makes analysis of such models significantly harder than for single trees. As a result, recent work has focused on developing exact and approximate techniques for questions such as robustness verification, fairness and explainability for such models of tree ensembles.

In this paper, we focus on a specific problem of feature sensitivity for additive decision tree ensembles and build a formal verification framework for a parametrized variant of it, where we also take into account the confidence of the tree ensemble in its output. We start by showing theoretical (NP-)hardness of the problem and explain how it relates to other verification problems. Next, we provide a novel encoding of the problem using pseudo-Boolean constraints. Based on this encoding, we develop a tunable algorithm to perform sensitivity analysis, which can trade off precision for running time. We implement our algorithm and study its performance on a suite of GBDT benchmarks from the literature. Our experiments show the practical utility of our approach and its improved performance compared to existing approaches.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
In the context of interpretable machine learning, this paper investigates the problem of "feature sensitivity" in tree ensembles. Specifically, given a classifier $ c $ and a set of features $ F $, the goal is to determine whether there exists an instance $ x $ such that the classification of $x$ by $c$ changes when only the values of the features in $ F $ are modified. The paper establishes that this problem is NP-complete for gradient-boosted trees, even when $ F $ consists of a single feature. The authors also explore an extension of this problem that considers the confidence associated with class changes, which is also shown to be NP-complete. To deal with this source of complexity, the authors propose a Pseudo-Boolean encoding for the feature sensitivity task, which they empirically validate on several benchmarks.

### Strengths
**Novelty:**
Arguably, the main strength of this paper lies the novelty of the complexity results for deciding feature sensitivity when the predictive model is a binary-class gradient boosted tree. Notably, the fact that the problem remains NP-hard even in the case where the sensitivity analysis is reduced to a single feature is intriguing.

### Weaknesses
**Clarity:** 
The scope of the paper is quite challenging to define. As indicated in the title, summary, and introduction, it seems that the authors aim to examine feature sensitivity for “tree ensembles,” which is a broad category that includes several model classes, such as gradient-boosted trees (GBTs) and random forests (RFs). This is further illustrated in Section 2, where the authors attempt to provide a general definition of tree ensembles (Lines 130-136). However, upon reviewing the technical sections of the study (Sections 3-5), it becomes clear that the results are exclusively related to GBTs. This focus is reiterated toward the end of the introduction (starting at Line 65), where the authors clarify their exclusive emphasis on GBTs.

Therefore, it is essential for the authors to establish a clear consensus on the scope of the paper in the revised version. If they intend to concentrate solely on GBTs, this specificity should be highlighted throughout the paper, beginning with the title and summary. The definition of tree ensembles in Section 2 should also be replaced with a definition of GBTs. Conversely, if the authors believe that their theoretical results could be easily extended to other tree ensemble classes (e.g., random forests), then the proofs in Section 4 should be modified accordingly (see Question 2).

At present, the definition of tree ensembles in Section 2 is somewhat unclear. If we denote by $c(x)$ the sum of the decisions $T_i(x)$ made by each of the $m$ decision trees $T_i$, then what class in $\mathcal{F}$ is predicted by $c$ on $x$? This is evident for GBTs when $\mathcal{F} = \\{-1,+1\\}$, using the sigmoid function $\sigma$, but it remains ambiguous for other classes of tree ensembles. If the authors’ goal is to address various classes of ensemble models, it is important to provide a comprehensive definition of “tree ensemble” that can be instantiated for both GBTs and random forests.

**Significance:**
The main theoretical contribution of this paper lies in Theorem 2, which establishes that single feature sensitivity is NP-hard for GBTs. Other NP-hardness results presented in this study stem from this case. While this finding is commendable, I remain unsure if it is sufficient for ICRL. To enhance the theoretical significance of this study, it would be beneficial to investigate whether this problem and its variants are W[1]-hard or if some of them are fixed-parameter tractable (FPT), with the parameter of interest being the maximum depth of the trees (See Question 1). Establishing hardness results for W[1] would indeed bolster the justification for the constraint programming approach discussed in Section 4.

### Questions
(1) Can the NP-hardness of single feature sensitivity (Theorem 2) or subset feature sensitivity (Theorem 1) for GBTs be extended to W[1]-hardness? 

(2) Alternatively, can these hardness results for GBTs be extended to random forests?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper addresses the sensitivity verification problem of GBDTs by formalizing a pseudo-Boolean encoding. The authors claim their approach can achieve significant performance gains over existing sensitivity verification tools, specifically for GBDTs. However, the paper lacks a comprehensive presentation, robust mathematical formulation, and systematic experiments. As such, it would need major revisions to be accepted.

# Discussion phase
After a deep discussion phase during which the authors addressed all my weaknesses and improved their work, including a better result, I increased my overall score to 6/10. Specifically, I've changed only the scores while leaving the original review; I could also change the review if that's mandatory.

### Strengths
- **S1:** The paper addresses an important issue aimed at verifying the sensitive features of GBDTs.
- **S2:** Mathematical proofs, which I would have expected to have a better presentation, but I appreciate the effort.
- **S3:** The authors present an approach using pseudo-Boolean encoding, arguing that it is more suitable than other methods.

### Weaknesses
- **W1:** The notation is inconsistent and confuses readers, especially those who are unfamiliar with the topic. For example, input examples are denoted by $x$ and $x'$ in Def. 3.1, but then they are $x_1$ and $x_2$ in Def. 3.2; furthermore, in the proof of Th. 2 instances are symbolized $a$ and $b$ (a very unhappy notation). Similarly, trees are defined as $T$ in Section 2 but then symbolized with $D$ in the proof of Th. 1.
- **W2:** Certain mathematical expressions and definitions are unclear or incorrect. For example, $F = f$ instead of $F = \\{f\\}$ detracts from the overall comprehensibility. As another example, the set minus ($\setminus$) operation is properly used in Def. 3.1 but then in Section 4 is not used; that is, the authors use minus ($-$). A further example is when the authors use $\mathcal F = F \bigcup f$, which, again, is not a proper mathematical definition; it should perhaps be $\mathcal F = F \cup \\{ f \\}$.
- **W3:** The proofs suffer from inconsistent symbols, poor spacing, and cluttered notation, making the logical progression of arguments hard to follow. For example, while reading the proof of Th. 1, I had to stop and skip such proof because the notations became heavy, and my overall cognitive overload intensified. To give a precise idea of how it felt: in $c'(x) = -n$, $n$ is the same $n$ as the number of decision trees in the ensemble (line 205)? What is the $\Longrightarrow$ in the notation? What is $\wedge$? Etc. As you can see, there are too many doubts that the reader has at this point. I assumed the proof was correct, even though I hadn't thoroughly reviewed it. The problem with this way of writing the proof is that the authors mixed textual explanations with mathematical ones in an unpleased way. This point is very critical.
- **W4:** The experimental results lack clarity; terms like "maximum, minimum, and average" time are used ambiguously without explaining what they represent or if multiple runs were performed.
- **W5:** Despite the general title, the paper focuses solely on GBDTs, excluding other ensemble methods like random forests without justification. If that's not the case, I expected to see experiments with random forests as well.
- **W6:** The paper contains grammatical errors, inconsistent abbreviations (e.g., "wrt" vs. "w.r.t."), and informal language, which detracts from the overall presentation.

### Questions
- **Q1:** Could you ensure consistent notation throughout, especially in proofs? E.g., $x$ and $x'$ vs $x_1$ and $x_2$.
- **Q2:** Why did you limit the work to GBDTs? Would your framework extend to other ensemble methods, such as random forests?
- **Q3:** Could you clarify the connection between sensitivity and $(p)$-sensitivity, especially after Definition 3.2? The special-case/general-case distinction you imply needs a clearer mathematical basis, and it is *not* easy to see (i.e., one uses the sigmoid function $\sigma$ and the other does not).
- **Q4:** Why do you use the sigmoid function $\sigma$ in Def. 3.2, and then decide to drop such "complexity" in Section 4 when you present the encoding? I would either drop it from the definition or introduce it in the encoding; either way would be acceptable, but please avoid the asymmetric treatment.
- **Q5:** What does "maximum, minimum, and average time" mean in your experiments? Were these metrics based on repeated runs, and if so, could you clarify the settings?

### Soundness
3

### Presentation
3

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
The paper puts forward an approach for formal verification of feature sensitivity. Feature sensitivity is a characteristic of a model whereby the model changes its output depending on values of such features. In domains such as fairness checking for feature sensitivity is one of the ways in which fairness checks are meaningfully encoded.

The paper answers some complexity problems on answering features sensitivity, leaves some open. Crucially it provides a novel and efficient way based on pseudo-Boolean encodings to solve the verification problem. Some experimental results showing SoA performance are provided.

Overall this is a solid paper, arguably with some weaknesses, that advances the SoA of the area.

Post-rebuttal note: The authors answered by questions and provided additional material (developed post-submission) to solve some of the questions left unanswered in the paper. On the other hand I also noticed some presentation issues that I think can be fixed. I think the final answer on scalability on the point are raised remains unresolved; so it is still not 100% clear to me the extent to which the novel encoding is responsible for the gain. I do accept the authors point in as far as the comparison against SMT goes. 
Overall I am still mildly positive on the paper.

### Strengths
The paper is very well presented (some very minor typos that can be fixed). I cannot judge the soundness of the encoding but it seems reasonable. I read the complexity proofs and appear correct (even if not surprising). In terms of advancement on SoA I judge this to be in line with the standards expected at ICLR, even if perhaps not as a top paper (see weaknesses below).

Most importantly solving model valuation for ensembles remains a key question in a variety of domains and this paper presents a principled and noteworthy contribution to the challenge. While the NP-hardness results may not be surprising they fill a gap in the knowledge in the area and the encoding presented achieves SoA performance.

### Weaknesses
While the NP hardness won't be a surprise to most, not all the theoretical questions are answered in the paper with the obvious gap being trees of depth 2 and 3. I feel this has a considerable implication even in terms of architectural suggestions from this study. Has this question been answered by the authors in the meantime? To me these corner cases are actually the most interesting ones. Finding the the case for 3 is polynomial would be a very interesting result

The reasons as to whether the approach appears to scale considerably better than present SoA were not clear to me in the paper. Do the authors have an explanation? If so, could they provide it? The reason for asking is that in principle it could even be that their implementation is just more efficient and it does not have so much to do with the encoding.

### Questions
See questions on the weaknesses above.

### Soundness
4

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This submission considers the problem of sensitivity for tree ensemble classifiers, i.e. the question of determining if, for a selection \\(F\\) of features, there exists a pair of inputs that agree on all the features outside of \\(F\\) but get classified differently by the tree ensemble. This work also develops a quantitative version of sensitivity (Def 3.2), parametrised by a number \\(p\in[0,\frac{1}{2}]\\), that additionally lower-bounds the "distance" between the two classifications. 

The main theoretical contributions consist in proving that 
- the (non-quantitative) sensitivity problem is NP-hard (Thm 1)
- the sensitivity problem restricted to singleton feature sets F={f} is also NP-hard (Thm 2)

The main practical contribution is to show that the problem of establishing \\(p\\)-sensitivity can be encoded as the satisfiability of a system of pseudo-boolean constraints (Thm 3). Experimental evidence shows that this approach scales much better than SOTA methods.

### Strengths
The paper is very well-written, with a good overview of the literature and easy-to-follow examples, proofs and discussions. In particular the proofs are easily understandable even by someone with very limited knowledge of computational complexity theory such as myself.

The main practical novelty of this paper -- using a pseudo-boolean satisfiability encoding -- is credibly shown to be superior to the SOTA methods in a short but convincing experimental section.

### Weaknesses
Theorems 1 and 2 correspond to two extremes of the sensitivity problem: the first considers sensitivity w.r.t. the full set of features, the second considers only singleton subsets of features. It feels like the full story would be a result showing that sensitivity w.r.t. subsets of features of size \\(N\\) is NP-hard, for all \\\(N\\). Similarly, the story is slightly incomplete as in does not cover trees of depth 2 or 3 (as mentioned by the author(s)). The paper would be nicer (and stronger) if these questions were solved. too 

Here are some minor comments/corrections
* l69: when THE number
* l83: whether...whether
* l104: some OF these
* l117: of A \\(d\\)-dimensional
* l157: When \\(F=\\{f\\}\\)
* l168: two-tree
* l194: Def 3.2 is not really a special case of Def 3.1 in the sense that Def 3.1 does not correspond to a particular choice of \\(p\\). As it stands, Def 3.1 is equivalent to \\(\exists p.\\)Def 3.2 holds. To make Def 3.1 a special case of Def 3.2 I would change it to have \\(p\geq 0\\) and \\(\sigma(c(x_1))\geq 0.5+p, \sigma(c(x_2))< 0.5+p\\) (or the other way round). Then Def 3.1 would correspond to the case \\(p=0\\).
* In the proofs of thm 1 and thm 2, why not simply take \\(X_{k+1}=\\{-1,1\\}\\) and \\(X_0=\\{-1,1\\}\\)?
* l275: \\(F\cup\\{f\\}\\) (not \bigcup and singleton)
* l289: I don't understand the first sentence here. Which problem requires three variables \\(x_1,x_2,x\\)?
* l307 and l313: use \setminus for the set difference \\(\mathcal{F}\setminus F\\). Also, if you'd used \\(x,x'\\) or \\(x,y\\) instead of \\(x_1,x_2\\), you wouldn't have problems with double subscripts.
* l310: \log instead of log
* eq (2): shouldn't \\(j+1\\) in the RHS simply be \\(j\\)?
* l327-328: Incomplete sentence
* l328: Since THE root
* Thm 3: use \eqref in the conjunction, in order to get \\((1)\wedge\ldots\wedge(6)\\). Parentheses = equations. Conversely don't use (1) and (2) to refer to the two parts of the theorem, instead use 1. and 2., or better (i) and (ii).
* l394: mine -> us
* l420: p-sensitive -> \\(p\\)-sensitive

### Questions
Do you think it is the case that sensitivity w.r.t. subsets of features of size \\(N\\) is NP-hard, for all \\\(N\\)? If yes, do you have any intuition about whether proving this is substantially harder than your existing proofs?

Same question for trees of depth 2 or 3.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses the feature sensitivity problem for ensembles of decision trees, with a focus on Gradient Boosted Decision Trees (GBDT). The authors demonstrate that this problem is NP-complete through a reduction from the SAT problem. They then propose an algorithm to solve the p-sensitivity problem by encoding it as pseudo-Boolean constraints. Finally, experimental results are provided, comparing the proposed approach to some existing methods.

### Strengths
The paper is well-written, with a clear structure that makes it easy to follow. This clarity enhances the readability and understanding of the material.

The problem is well motivated.

The theoretical contributions are interesting and their proofs appear to be correct.

### Weaknesses
The initial result on NP-hardness of sensitivity problem for all features seems redundant after establishing NP-completeness for a single feature. 

My main issue with this paper is with the experiments. The comparison with VERITAS seems a bit unfair. By limiting VERITAS to the same runtime as SENSPB, the comparison may not reflect its full potential. Allowing VERITAS more runtime might yield better results, or it could have already produced sufficiently good results within its runtime. A separate table detailing the number of instances each method solved and the time taken would provide additional insights.

**Minor Comments/Typos**
- Line 068: "just on" → "just one"
- Line 079: Rewrite this sentence for clarity: “Can we use other forms of powerful reasoning, Pseudo-Boolean solvers, that have shown to be effective in other problems (Mexi et al., 2023), for the sensitivity problem?”
- Line 162: "given set of sensitive features F" → "given set of features F"
- Caption of Table 1: "results than mine" → "results than our method"

### Questions
Is there a reason why the NP-hardness of the all feature problem was included in the paper? If it can be useful in some cases, some explaination here can be beneficial.

It would be nice to know why the results are specific to GBDT and why they could/couldn't be directly extended to other ensemble models, such as random forests.

### Soundness
2

### Presentation
3

### Contribution
2
