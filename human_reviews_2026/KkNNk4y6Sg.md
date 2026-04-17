# Fully Differentiable Temporal FO Rule Learning

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 6, 6, 2

## Abstract
We propose a novel differentiable neural architecture for learning first-order temporal logic rules enriched with metric
operators. Leveraging differentiable immediate consequence operators over data, we extend the approach to temporal data
by learning both the predicates and the temporal intervals in which they hold. Among the strengths of our model are its
support of existential literals in rule bodies to express eventualities within an interval and its seamless applicability to
data over both discrete and dense time intervals. Notably, our model can effectively capture temporal dependencies without
reifying all possible timestamps and produces a linear number of rules in the size of the training set, which has a benign
effect on model complexity and scalability. We explore different use cases and show in experiments the benefits of our
approach, highlighting its potential as a scalable solution for interpretable metric temporal rules over data.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes a method to learn rules in an end-to-end manner. This paper explores different use cases and demonstrate its benefits.

### Strengths
1. Background information and preliminaries are provided to help readers know this domain better.

2. Experimental details are provided.

Overall, this paper is far from reaching ICLR publication standards.

### Weaknesses
1. The contribution is, in general, not clear. In the introduction, the core contribution is not clearly delivered (basically one paragraph with several bullet points). The core contribution and methodology are not clear in many points. See questions below for details.

The contribution #1 seems to be weak as it can only learn two specific rules. Also, the "somewhere" statement looks confusing as it does not establish the significance in the given research fields. This looks like a specific technical point without obvious benefits for the overall rule learning framework.

2. Many presentation issues exist. The overall method is not clear to the reviewer. Many methodological concerns exist.

(1) No sub-sections are provided. The overall paper looks pretty dense and hard to digest.

(2) Fig. 1 is pretty unclear. No descriptions are provided, and links to the methodology body are vague. 

(3) The reviewer does not find any MLP layers or other common neural network layers. Therefore, it's hard to claim the model as a "neural network". It seems that the proposed method is merely an optimization problem over a set of pre-defined logical operations. It has nothing to do with neural networks. 

(4) The preprocessing procedure is very, very confusing. It should be presented with algorithm boxes along with examples. Otherwise, it's hard to follow the dense textual descriptions. Meanwhile, the preprocessing steps do not provide rationales behind, and these model ingredients are not ablated well.

3. The results are much worse than baselines.

### Questions
1. In the introduction, how to define a model as "fully differentiable"? It's unclear because this is the core contribution. Many previous works can produce "human-readable rules", so how can to differentiate from those works, and how do fully differentiable architectures generate explicit rules?

2. What's the number of parameters used in this model?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper presents a differentiable architecture for learning (modal-logic-inspired) temporal first-order rules with metric operators (e.g., properties must hold within a given time interval). A crucial part of the architecture is the use of immediate-consequence operator(s) over data by Walega et al. (2021), which enables the learning of minimal (Herbrand) models using fixpoint semantics. Unfortunately, the paper fails to deliver on many of its promises. From the abstract, the "model [$\ldots$] produces a linear number of rules in the size of the training set", but the paper does not comprehensively analyze this aspect; also, "its seamless applicability to data over both discrete and dense time intervals", but the paper's focus is only on discrete time intervals, while the continuous/dense scenario is only discussed in the last section as future outlook ("our ongoing work aims to support dense timelines [$\ldots$]"). I was genuinely eager of learning more about the method, but the current presentation does not help in this endeavor; specifically, the mathematics are flawed, and the paper's adopted unconventional strategy of not having a guiding example to stress the reader's understanding or to "correct the flawed mathematics" severely hampers the accessibility of this work. Moreover, regarding the experiments, I was unable to fully grasp the setting and the claims; for example, in the cyber-physical scenario, how long the time horizon is, what types of intervals it uses (I assume they were discrete), and how many queries were generated remain a mystery. In summary, even though I have had a strong interest in learning something new, the paper failed to meet my expectations, so I will recommend rejection.

### Strengths
(S1) The paper investigates an interesting problem. 

(S2) The exploitation of the fixpoint semantics adds foundations to the work.

### Weaknesses
(W1) The abstract should be rewritten substantially, as the paper fails to deliver such promises.

(W2) No complete guiding example. One piece of example at lines 356--359 is unclear. 

(W3) Figure 1, which should add substantial information/context to the paper, is underdeveloped. 

(W4) Too many mathematical flaws and presentation inconsistencies in the background and method sections (and beyond). For example:
  - line 036: The triple $(s,r,o)$ is undefined.
  - lines 124--128: $\mathcal P$ is the set of *ground* atoms with the target predicate/atom, and are called positive examples; one should similarly define $\mathcal N$ (the negative examples). I'm pointing this out because from the current presentation, the target atom $h$ is never used. (*My takeaway:* For context, this is the setting of learning from entailments.)
  - lines 134--139: The head atom $\mathbf v$ is never used/defined. Instead, the presentation uses $\mathbf v_h$. Also, "The vector $\mathbf v_h$ encodes the Boolean values of the body atoms of $r_k$ under $\theta$ [$\ldots$]", but what is the relation between $\mathbf v_h$ and $r_k$? Perhaps it should be $\pi_h$ instead of $r_k$ since $r_k \in \pi_h$.
  - line 177: In $p_{i_j} (e_{k_j}, e_{s_j}) @ t_j$ do you really need all those subscripts? I'm asking because $i$ is never used. Also, $k$ and $s$ (either with subscripted $j$ or not) are not defined.
  - lines 177 & 368: Why not use the same notation for (temporal) facts? 
  - lines 191, 304 & 308: Is there any semantical difference between $\times$ and $\cdot$? I would say not. 
  - line 199: "[$\ldots$] $(e_i, e_j)$ has the highest co-occurring score [$\ldots$]". What is a co-occurring score at this point of the presentation? 
  - line 206: The symbols $m$ and $n$ have been already used previously. Please avoid symbol clashing as much as possible. 
  - line 243: What exactly is max_int? Is it the maximum integer that can be stored on a particular hardware? Or is it a constant/hyperparameter that depends on something that I don't see?
  - line 256: I want to stress out the "unhappy" mathematical notation $n^h_r \times 3 n^h$. Is this really the best possible notation that you can come up with? In general, I would advise against these kinds of notations. The entire paper overburdens the reader's cognitive load with subscripts and superscripts.
  - lines 259--260: "[$\ldots$] always-metric atoms [$\ldots$] eventually-metric atoms [$\ldots$]", but always- and eventually-metric atoms have never been defined as such; I mean, these terms are introduced here for the first time. Given my background knowledge, I assume those are the metric atoms that use only (future/past) metric diamonds and boxes, respectively. 
  - line 275: "[$\ldots$] using a sigmoid function [$\ldots$]". Which sigmoid function? Does it have a name or definition?
  - line 277: $\mathbf E^{in}$ is undefined. 
  - lines 280 & 284: The symbol $\odot$ is undefined. It is, however, defined later at line 284. This, again, is unconventional formal writing. 
  - line 286: "[$\ldots$] punctual matrix [$\ldots$]". What is a punctual matrix? As a reader with a PhD in mathematics, I assumed it was $\mathbf P$.
  - lines 273--300: $\mathbf A^i$ for all $i \in \\{1,\ldots, 7\\}$ is an overkill. By the way, at line 283, you use $\mathbf A_3$, while later you use $\mathbf A^3$. 
  - line 298: Following the definition, $\mathbf T$'s dimension is $n^h_r \times 3 n^h$ (line 256), and thus $\tilde{\mathbf T}$'s too (line 272). So, by my count, $\tilde{\mathbf T}[:, 0:n^h] || \tilde{\mathbf T}[:, n^h:2n^h] || \tilde{\mathbf T}[:, 2n^h:3n^h]$, where $||$ is the matrix concatenation operator, has dimension $n^h_r \times (3n^h + 3)$. Therefore, the definition of $\mathbf B$ is wrong. Moreover, again, $\mathbf T^{in}$ is undefined.
  - line 316: The paper uses $p^k$ for the penalty. There are at least two issues here: (1) $p$ has already been used for the atoms, and (2) $k$ has already been used as well. 
  - line 349: Both $\mathbf S_{i,j}$ and $\mathbf E_{i,j}$ are undefined. 
  - lines 360--364: $D,t \models H$ is undefined. Moreover, the equation's definition of $w(h,r)$ has at least three issues: (i) $h$ is never used in the right-hand part of the equation, (ii) $sc(\cdot)$ is undefined, and (iii) $hc(\cdot)$ is undefined as well. 
  - line 370: Once more, $c_{max}$ is undefined. 
  - line 410: $r[D]$ is undefined. 
  - line 419: "[$\ldots$] $R(a,b)@t$, where $t_{max}$ is the maximum timestamp in $D$ [$\ldots$]". What is the role of $t_{max}$ here?

As can be noted, the above (non-comprehensive) list of inconsistencies and flaws severely detracts from the overall understandability of the core proposal. And again, I was genuinely eager to understand the proposal, but even with that in mind, I was unable to put the pieces together. As a general rule, I would advise doing a thorough proofread before submission.

(W5) Now for the experiments. This part feels weak and underdeveloped. What are the "sampled variants of MTLearn"? I assume that some examples have been sampled to produce the results in Table 1. If so, have you used different random seeds for the sampling and aggregated the results in Table 1? If not, why? What are ICEWS14 and ICEWS05-15? I assume those are datasets, given the context in Table 1. Do such datasets have a bibliographic reference? What are the characteristics of these datasets? I'm asking because the abstract claims the extraction of a linear number of rules (in terms of the dimensionality of the training dataset(s)). As noted earlier, the cyber-physical scenario requires further explanation (see the Summary). Finally, what is a "strong version of preconditions" (line 450)? 

(W6) I've looked at the Appendix, which needs significant improvement, as with the main text. I'm not a huge fan of using bullet points/enumerations for listing just one item (e.g., lines 652--654, line 673, and line 680). It is clear to me that the Appendix hadn't been read before submission. Although notable, the ablation study is weak; I would have expected, besides ablating against eventually or not, to see how other components of the architecture perform while enabling/disabling those parts.

Minors:
- line 456 uses $:-$ while it should be $\leftarrow$ 
- Please fix the naming inconsistency of MTLearn vs MT-Learn (line 468).

### Questions
See above.

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces MT-Diff-Learn, a novel, fully differentiable architecture for learning first-order temporal logic rules expressed in DatalogMTL. DatalogMTL extends Datalog with metric temporal operators (e.g., "always", "eventually") annotated with explicit time intervals. The framework aims to address limitations in existing temporal rule learning methods, specifically their limited expressiveness and scalability issues often caused by massive rule generation or the need to reify timestamps. The methodology involves a pipeline: data preparation (sliding windows, heuristic-based lifting of ground facts, PMI filtering), a neural network that learns weights and interval boundaries using differentiable approximations of interval containment (for $\Box$) and overlap (for $\diamond$), and a final rule extraction phase.

### Strengths
•	MT-Diff-Learn is a fully differentiable framework for learning DatalogMTL rules. The inclusion of mechanisms to learn eventuality conditions ($\diamond$, $\bullet$) significantly increases the expressiveness compared to prior temporal rule learning methods, allowing the model to capture temporal uncertainty.

•	The approach demonstrates excellent scalability by generating several rules linear in the size of the training set. Empirically, it produces orders of magnitude fewer rules than MTLearn. This vastly improves interpretability.

•	By treating intervals as first-class citizens and learning their bounds directly, the model avoids the quadratic blow-up associated with reifying every possible timestamp, a common bottleneck in other approaches.

•	The synthetic scenarios (Cyber-Physical and Action Description) effectively demonstrate the utility of the eventuality operators. The ablation study confirms their importance, showing a sharp performance drop when they are disabled.

### Weaknesses
•	The lifting process is complex and relies heavily on heuristics (overlap scores, processing order, variable assignment constraints). The robustness of the system to these choices is unclear, and the impact of associated hyperparameters (window size $l$, breadth $b$, depth $d$) is not analyzed. The generalizability of this lifting process warrants further investigation.

•	On standard tKG benchmarks (ICEWS), MT-Diff-Learn significantly lags behind SOTA embedding-based methods. While the focus is on interpretability, this gap may limit adoption where accuracy is paramount. The paper should better discuss this trade-off.

•	Key parts of the methodology are difficult to parse. The lifting procedure (W1) is confusing. Furthermore, the process of translating the learned intervals back into DatalogMTL syntax is unclear. The example provided (transforming [2, 5] into a conjunction of past and future operators) seems overly complicated and requires clarification regarding how semantics are preserved relative to the window center.

•	The abstract claims applicability to data over dense time intervals. However, the semantics defined in Section 2 are explicitly over integers ($\mathbb{Z}$), and dense time is only mentioned as future work. 

•	While the output model is succinct, the training process involves complex tensor operations. The input tensors (Start/End) have a dimension representing the maximum number of disjoint intervals an atom might hold in a window. The paper does not detail the practical implications of this on memory usage and training time if this value is large.

### Questions
The lifting process is intricate and heuristic-driven. How sensitive are the final results to these heuristics (e.g., the ordering strategy)? Could you provide a small, concrete example illustrating how the variable assignment rules (1-4) operate across two overlapping windows?

Could the authors clarify the interval translation process in L354-359? Assuming a window size of 8 (center at 4), how exactly does a learned interval [2, 5] translate into "■[0,2] \gamma (X,Y) ∧ □[0,1] \gamma (X,Y)"? aka, 2 time units in the past and 1 time unit in the future. The interval relative to the center seems to be [-2, 1]. 

How does the memory consumption scale with the max_int parameter? Could this become a bottleneck for datasets with complex temporal patterns?

Could you clarify the discrepancy between the abstract's claim of applicability to dense time and the discrete-time semantics defined in Section 2?

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
2

### Summary
The paper presents MT-Diff-Learn, a fully differentiable neural architecture for learning First-Order Temporal Logic (FOTL) rules enhanced with metric operators. The model extends differentiable immediate consequence operators, common in differentiable logic programming, to the temporal setting. This extension enables the system to simultaneously learn predicates, the logical structure of rules, and the associated precise metric temporal intervals. The resulting approach is claimed to handle data over both discrete and dense time, capturing temporal dependencies efficiently without reifying all timestamps. The authors state that this leads to a rule complexity that scales linearly with the training data size, positioning the model as a mechanism for discovering complex, interpretable temporal rules.

### Strengths
The primary strength of this work is the integration of metric temporal operators within a fully differentiable rule learning framework. This capability is notable, as it allows the approach to discover rules containing complex temporal patterns over dense time, addressing a limitation of methods restricted to discrete time steps or simple precedence relations. The ablation study effectively isolates the contribution of the eventuality operators, demonstrating a substantial performance gain (e.g., in MRR), which supports the model's design for capturing system dynamics where events occur with uncertainty within a temporal window. Furthermore, the claim of generating a linear number of rules relative to the training data size suggests beneficial characteristics for scalability and model complexity in applications involving large, complex datasets.

### Weaknesses
The experimental section could be strengthened by including comparisons against more diverse state-of-the-art neuro-symbolic or temporal rule mining baseline approaches, beyond standard non-temporal or limited temporal inductive logic programming methods, to fully contextualize the proposed model's performance in the broader field of temporal sequence modeling.

### Questions
How robust is the learning of the continuous metric interval bounds to noise in the training data, and what regularization or loss terms (if any) are specifically implemented to prevent interval collapse or explosion during gradient descent?

Given the focus on interpretability, can the authors provide a more detailed analysis of the learned rules—perhaps a qualitative summary or examples from the different use cases—to illustrate how MT-Diff-Learn discovers non-obvious or complex temporal relationships that purely sequential models might miss?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This paper is a technical contribution within a dense existing project.

It builds on [Gao et al. (2024)](https://www.sciencedirect.com/science/article/pii/S0004370224000444).  That paper is not easy to skim.  Its [2022 version](https://www.ijcai.org/proceedings/2022/417) is not much easier, but it is at least shorter and has a video that sort of walks through an example (see also the [appendices](https://arxiv.org/pdf/2204.13570)).  My *rough* understanding of that prior work after wrestling with it for a while is as follows (the authors can correct me if I'm wrong): 
* The goal is to find a logic program that predicts the truth values of some given positive and negative atoms given a set of background fact atoms.  
* This is done by optimizing a soft Datalog program and then somehow extracting a hard Datalog program from it.  
* A soft program is similar to a hard program except that it replaces AND, OR with fuzzy differentiable versions.
* The learning objective for the soft program is the log-loss on the fuzzy predictions, plus several regularizers.
* Crucially, the set of non-ground terms that can appear in the program is bounded in advance since the maximum number of predicates and variables are given as hyperparameters, the arity of each predicate is always 1 or 2, and the program contains no constants.  
* For *each* non-ground term, a matrix is learned whose rows correspond to the rules with that non-ground term as head.  Each row has entries in [0,1] and specify the degree to which each of the non-ground terms participate in the fuzzy AND for that rule's body.  (The number of rows is a hyperparameter, and there is one column for each non-ground term.)

**The present paper extends that method to [DatalogMTL](https://arxiv.org/abs/2201.04596),** or rather continues an extension already begun by [MTLearn](https://proceedings.kr.org/2024/90/kr2024-0090-wang-et-al.pdf).  

DatalogMTL is a version of Datalog in which each predicate implicitly has an additional argument Time, so that it can be true at some times and not others.  A rule essentially has the form property(Time, Arg1, Arg2) :- ..., where each subgoal has the form (∀T ∈ [Time+a,Time+b]) property(T,X,Y) or (∃T ∈ [Time+a,Time+b]) property(T,X,Y).  These additional levels of quantification will complicate learning.

In the learning method here, the provided positive facts are fully ground, including the Time argument.  However, in contrast to the prior work, there are no negative facts or background facts provided.  (At least, this is what I conclude from Fig. 1.)  Thus, the goal may be to compress the provided set of positive facts by writing a short program that predicts some of them from others. That would make it a kind of association rule mining method (traditional example: "if you buy beer, you're also likely to buy chips").

### Strengths
The authors appear to have worked hard.  I think they themselves probably understand what they're doing, even if it is hard for a reader to understand from this presentation.

### Weaknesses
I apologize that I was not able to follow the paper well.  (I thought it would help to go back to the prior work, but it didn't.)

The paper is a bit like reading someone's half-documented code.  The paper says "We do this, then we do that, then we do that."  But it is often not clear what the goal is or why those steps are appropriate.  This is particularly a problem in section 3.  It also makes it difficult to adjust for lapses in the exposition.  To take some early examples: "target atom $h$" isn't defined at line 126 and it's not clear where it comes from or why it's needed.  At line 132, it's not clear where the threshold comes from: before relaxation, is it meant to be the number of negative literals in the rule?  At line 202, what happens if *multiple* atoms $p(X_i,X_j)$ were "already considered in the previous windows"?

Even as a low-level description of what's done, the paper doesn't quite hang together.  To take one example: Line 190 defines the "overlap score" $O(w)$ and says that it is "used to determine processing order," but it's not clear what "processing order" refers to, and neither $O(w)$ nor the overlap score is ever mentioned again!  (I suppose it's the same as the "co-occurring score" in the next paragraph.)  Furthermore, the definition of $O(w)$ contains unbound variables $i,j$.  (I suppose it should have been named $O_{i,j}(w)$.)

### Questions
What is the formal learning problem here?  
* Is it a machine learning problem where there is a true set of temporal facts over intervals, the answers to some queries are observed at random, and the estimated DatalogMTL program should accurately predict the answers to other queries? 
* Is it a statistics problem where a true parameter is a DatalogMTL program, the answers to some queries are observed at random, and the true program should be recovered in the limit?  If so, is the program in fact identifiable?
* Is it a computational problem of finding a small description of a set of positive facts (and if so, how is it ensured that the description does not predict false facts, since negative examples are not provided)?  
* Is it an association rule mining problem as mentioned earlier?  If so, what is the success criterion?

What does "consecutive occurrences" mean at line 181?  You talk about @t, @t+1, and @t+2.  I had assumed that time was continuous, since you referred early on to "intervals"; did you actually intend for time to be discrete?

It's not clear to me whether learning temporal logic programs is an important problem.  Can you make a case for it?  Line 417 discusses a "temporal link prediction" task, but of course there are many cleaner ways to solve such a task, such as training a Transformer with temporal positional embeddings.  (And based on Table 1, it looks like your method wasn't competitive with such alternatives.  You suggest it might be more explainable, but it's hard to evaluate the quality of the explanation from the paper, especially given its size: 40K rules according to line 425.)  Lines 432 and 449 seem to discuss temporal modeling tasks, but these could be handled by generative methods such as the neural Hawkes process or its variants.  (Here maybe you do have an explainability advantage, with 80 rules at line 460 on this synthetic dataset.  But as an alternative, perhaps one could train a neural generative model and then extract explanatory patterns from it post hoc.)

I am not able to follow the details of the method as presented.  On first principles, I would have expected a method similar to Gao et al. (2024), which you present as your starting point, but where the matrix columns had names like (∀T ∈ [Time+a,Time+b]) property(T,X,Y) or (∃T ∈ [Time+a,Time+b]) property(T,X,Y).  These truth conditions of such a column would be softened, in part by fuzzing the edges of the interval [Time+a,Time+b].  Thus, you could improve a, b by following their gradient.  Why didn't you do it this way?  

Instead, you used some heuristics involving windows and PMI to find rules.  What happens if you make the window size too large or too small?  Does this change the number of rules you find and their specificity, so that you might underpredict or overpredict positive facts?  What are the simplest examples where your heuristics would fail?

### Soundness
2

### Presentation
1

### Contribution
2
