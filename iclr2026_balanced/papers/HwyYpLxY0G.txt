# ALIGNED TEXTUAL SCORING RULES

**Anonymous authors**
Paper under double-blind review


ABSTRACT


Scoring rules elicit probabilistic predictions from a strategic agent by scoring
the prediction against a ground truth state. A scoring rule is _proper_ if, from the
agent’s perspective, reporting the true belief maximizes the expected score. With
the development of language models, Wu & Hartline (2024) proposes a reduction
from textual information elicitation to the numerical (i.e. probabilistic) information
elicitation problem, which achieves provable properness for textual elicitation.
However, not all proper scoring rules are well aligned with human preference over
text. Our paper designs the Aligned Scoring rule (ASR) for text by optimizing and
minimizing the mean squared error between a proper scoring rule and a reference
score (e.g. human score). Our experiments show that our ASR outperforms previous
methods in aligning with human preference while maintaining properness.


1 INTRODUCTION


The theory of proper scoring rules is well established for elicitation of numerical information, such
as the probability of a random state (McCarthy, 1956; Savage, 1971), the mean of a distribution
(Abernethy & Frongillo, 2012), and is widely used in practice (Danz et al., 2022; Hossain & Okui,
2013; Mobius et al., 2022).¨ Proper scoring rules score the quality of a probabilistic prediction by
comparing to the ground truth random state. By scoring a strategic agent, proper scoring rules are
mechanisms that incentivize truthful prediction. Information Elicitation is an important area of
research that has recent practical importance due to the reliance of data-driven algorithms and AI
systems on high-quality input.


For example, in peer grading, students report predictions about their peers’ homework correctness
(the random state). The instructor spot-checks homework submissions and reveals the ground truth
correctness. The student’s prediction is then scored in comparison to the ground truth. A scoring
rule is _proper_ (a.k.a. truthful) if, from the peer’s perspective, truthfully reporting her belief about the
correctness maximizes expected score.


The recent development of large language models (LLM) has enabled the evaluation of textual
information. Textual reports can encode richer information than numerical predictions. For peer
grading, answering open-ended review questions facilitates the students’ learning process better than
checking pre-specified numerical rubrics. One approach to incentivize high-quality textual review
from students is to score peer reviews by querying LLM to compare student reviews with the ground
truth instructor review. Moreover, studies on language-model-generated evaluation systems, i.e.,
LLM-as-Judge (Zheng et al., 2023; Fu et al., 2024), have demonstrated that language models often
align closely with human judgments when scoring text quality.


Language-model-generated evaluations offer scalability but, unfortunately, lack provable guarantees
such as truthfulness, leaving them vulnerable to strategic manipulation. For example, when language
models score peer reviews, fabricated comments may receive a higher expected score (Wu & Hartline,
2024). To address this issue, Wu & Hartline (2024) propose a reduction from textual elicitation
problem to numerical elicitation problem. Wu & Hartline (2024) views a language model as an
oracle accepting _summarization_ and _question-answering_ queries, where summarization identifies a
scoring rubric with states for elicitation, and question-answering maps text to numerical reports and
states. By implementing any numerical scoring rule over the identified space of rubrics, the scoring
mechanism inherits provable properness when the language oracle is perfect and achieves adversarial
robustness when the language oracle has errors. However, the scoring rules might not be aligned with
preferences.


1


The goal of our paper is to align a provably proper textual proper scoring rule with preferences, e.g.
human preferences. With the reduction framework in Wu & Hartline (2024), we optimize proper
scoring rules to align with an exogenously given score that reflects a preference or a scoring rubric.
For the peer grading application, we align the scoring rule to two reference scores: 1) the instructor
score of peer reviews, and 2) the LLM-Judge score, by quering LLM to compare a peer review with
the ground truth. While neither of these reference scores are proper, our optimization framework
converts the reference scores into a proper score.


Our Aligned Scoring Rule (ASR) is simple, provably truthful, and interpretable. We minimize the
Mean Squared Error (MSE) of ASR with the reference score. We optimize over the space of separate
scoring rules, which applies a single-dimensional scoring rule to each summary point and averages
across single-dimensional scores. The hypothesis space induces a convex optimization problem with
efficient algorithms. The separate scoring rules allow us to interpret and identify the important rubric
points from reference scores, by the convexity of each single-dimensional scoring rule.


We evaluate our Aligned Scoring Rule (ASR) on peer grading datasets. Results show that ASR fits
the reference scores effectively and outperforms baselines. We first present the result of a linear
regression that predicts the reference scores from ASR. The regression is nearly the identity function,
showing our ASR aligns with reference scores. Then we present the MSE and the Pearson correlation
between ASR and the reference score, in comparison with baseline methods including the best
constant score and the method in Wu & Hartline (2024). Our ASR outperforms baseline methods in
both metrics. Finally, we show the interpretability of ASR by a case demonstration in the appendix,
where ASR identifies reasonably important and non-important rubric points for scoring.


1.1 RELATED WORK


**Textual Elicitation** Several recent papers design scoring mechanisms to elicit textual information
from language models. Kimpara et al. (2023) models LLM as a distribution that generates independent
and identical (i.i.d.) textual samples. The paper designs a scoring rule that scores the distribution
with access to samples, to incentivize a truthful report of the distribution, while our work directly
scores the quality of a text. Lu et al. (2024) designs truthful peer prediction mechanisms that score
text without ground truth, by comparing the textual report of multiple peers. Wu & Hartline (2024)
designs proper scoring rules that score text with ground truth. The main goal of Lu et al. (2024); Wu
& Hartline (2024) is truthfulness (a.k.a. properness), which does not consider optimization. On the
contrary, our work optimizes over the space of proper scoring rules for alignment.


**Grading with LLMs** Recent work studies the use of LLMs in grading textual reports from students.
Kwiatkowski et al. (2019) studies grading via similarity between the vector embedding of the student
report and ground truth. They show that the vector embedding approach works well for simple binary
questions, but not for multiple-choice and more complex questions. Schneider et al. (2023) prompts
a language model to compare student reports to ground truth, which is shown to have low Pearson
correlation with instructor scores. Instead of directly prompting, our approach identifies scoring
rubrics and optimizes for alignment while maintaining properness, thus having more favorable results.


**Automated Mechanism Design and Differentiable Economics** Automated mechanism design
(AMD) is the use of computational techniques to search for good mechanisms on specific problem
instances. The earliest works in this area use linear programming (Conitzer & Sandholm, 2003a;b;
Sandholm et al., 2007; Conitzer & Sandholm, 2004); others frame the problem in terms of learning
theory, where the goal is to choose a high-performing mechanism from some class given access to
samples from the type distribution (Roughgarden & Schrijvers, 2016; Morgenstern & Roughgarden,
2016; 2015; Balcan et al., 2008; Feldman et al., 2014; Hsu et al., 2016; Balcan et al., 2016; 2018b;a). A
body of work sometimes called “differentiable economics” applies the tools of modern deep learning
to learn good mechanisms, either using neural networks as general function approximators (Dutting¨
et al., 2024), or using specially-designed architectures which guarantee strategyproofness in singleagent (Shen et al., 2019; Dutting et al., 2024; Curry et al., 2024) and multi-agent settings (Curry¨
et al., 2022; Duan et al., 2023; Wang et al., 2024). Like early work on AMD, we solve a convex
optimization to minimize expected loss from few samples. With more training data, applying
differentiable economics’ flexible function approximators is a promising future work.


2


**Optimization of Scoring Rules** There is an extensive literature that characterizes proper scoring
rules for numerical elicitation (McCarthy, 1956; Savage, 1971). Recently, a line of literature works
on the optimization of scoring rules subject to normalization constraints such as boundedness. Li
et al. (2022) optimizes to incentivize a binary effort in peer grading, where a peer either exerts effort
to refine her posterior belief or not. As a generalization, Hartline et al. (2023) considers incentivizing
a multi-dimensional effort. Our paper adopts the computation framework of the optimal scoring rule
in Li et al. (2022). Additionally, Neyman et al. (2021) incentivizes sequential and discrete effort,
Papireddygari & Waggoner (2022) connects proper scoring rules to contract theory, Chen & Yu
(2021) considers robust scoring rule design that relaxes the knowledge of the prior of the designer,
and Chen et al. (2023) designs optimal scoring rules in the online setting where the information
structure and the cost of signals are unknown.


2 PRELIMINARIES


This section introduces the preliminaries of information elicitation and scoring rules we use.


2.1 NUMERICAL ELICITATION


The goal of the principal (mechanism designer) is to elicit numerical reports on the quality over _n_
explicit rubric points, represented by states _**θ**_ = ( _θ_ 1 _, . . ., θn_ ) where each _θi_ _∈_ [0 _,_ 1]. The state space
is Θ = [0 _,_ 1] _[n]_ . For example, in peer grading, the rubric consists of Statement Correctness, Proof
Correctness, and Clarity. A state being 1 means the highest quality on that rubric point. The agent
holds a multi-dimensional belief _q_ _∈_ ∆([0 _,_ 1] _[n]_ ) over the _n_ states. The principal asks the agent to
report the marginal means _**r**_ = ( _r_ 1 _, . . ., rn_ ) from the report space _R_ = [0 _,_ 1] _[n]_ .


The agent is scored by a scoring rule _S_ : _R ×_ Θ _→_ [0 _,_ 1] comparing the reported marginal means _**r**_
and the realized state _**θ**_ . A scoring rule is _proper_ if the expected score is maximized when the agent
reports the true marginal means of the state. From the agent’s subjective perspective, the scoring rule
incentivizes the agent to truthfully report the believed marginal means to maximize expected score.

**Definition 2.1** (Properness) **.** A scoring rule is _proper_ for eliciting the marginal means, if for any
belief distribution _q_ _∈_ ∆([0 _,_ 1] _[n]_ ) with mean _**µ**_ _q_, and any deviation report _**r**_ _∈_ [0 _,_ 1] _[n]_,


**E** _**θ**_ _∼q_ [ _S_ ( _**µ**_ _q,_ _**θ**_ )] _≥_ **E** _**θ**_ _∼q_ [ _S_ ( _**r**_ _,_ _**θ**_ )] _._

A scoring rule is _ϵ_ - _approximately_ proper if for any belief distribution _q_ _∈_ ∆([0 _,_ 1] _[n]_ ) with mean _**µ**_ _q_,
and any deviation report _**r**_ _∈_ [0 _,_ 1] _[n]_,


**E** _**θ**_ _∼q_ [ _S_ ( _**µ**_ _q,_ _**θ**_ )] _≥_ **E** _**θ**_ _∼q_ [ _S_ ( _**r**_ _,_ _**θ**_ )] _−_ _ϵ._


Before reporting the belief, the agent holds a prior belief with marginal means _**p**_ _∈_ [0 _,_ 1] _[n]_, the
empirical frequency of the ground truth in samples. The agent learns and refines the belief by
receiving a signal _s_ _∈_ _S_ correlated with the ground truth state. The signal generation follows an
information structure, a joint distribution ∆(Θ _× S_ ) over the state space and the signal space. Upon
receiving the signal, the agent Bayesian updates to a posterior belief _q_ _∈_ ∆([0 _,_ 1] _[n]_ ).


2.2 TEXTUAL ELICITATION


Text conveys implicit information rather than explicitly listed rubric points in numerical elicitation.
Textual ground truth indicates a set of _m_ summary points. The reported summary points can be
represented by an _m_ -dimensional binary vector _**θ**_ = ( _θ_ 1 _, . . ., θm_ ), where _θi_ _∈{_ 0 _,_ 1 _}_ for each _i_ .
State _θi_ = 1 or 0 means “agree” or “disagree” on the corresponding point. For example, in a peer
review of an induction homework in an algorithm class, the summary points in the textual ground
truth review contain _θ_ 1 the correctness of the hypothesis, _θ_ 2 the base case, and _θ_ 3 _, θ_ 4 two details
about some particular induction step. A reported text can express uncertainty on each state, e.g. “the
base case is likely correct” as 70% probability that _θ_ 2 = 1 for base case.


In our peer grading dataset, we observe that textual reports either express a state being 0 or 1, or have
no information. Thus, we restrict our attention to proper scoring rules with report space _ri_ = _{_ 0 _,_ 1 _, ⊥}_
for each _i_ . We write _pi_ as the empirical frequency of _θi_ = 1 in our dataset. Assumption 2.2 interprets
an uncertain report of _⊥_ as the prior _pi_ .


3


A single-dimensional scoring rule for know-it-or-not reports can be characterized by nine values:
_S_ ( _r, θ_ ) for _r_ _∈{_ 0 _,_ 1 _, ⊥}_ and _θ_ _∈{_ 0 _,_ 1 _}_ . The definition of properness simplifies to Definition 2.5.
A V-shaped scoring rule is a special case of a single-dimensional scoring rule for know-it-or-not
reports, where the score of reporting _⊥_ is fixed at [1] 2 [.] [Figure 2 presents a graphical illustration of such]

a scoring rule.


4


**Assumption 2.2** (Know-it-or-not) **.** In the peer grading dataset, the agent’s posterior belief distribution
_qi_ is either 0, 1, or the prior _pi_ .


Assumption 2.2 restricts the space of proper scoring rules to scoring rules for report space _R_ =
_{_ 0 _,_ 1 _, ⊥}_ .

**Definition 2.3** (Scoring Rules for Know-it-or-not Reports) **.** Given the prior distributions _**p**_, a scoring
rule _S_ _**p**_ : _{_ 0 _,_ 1 _, ⊥}_ _[m]_ _× {_ 0 _,_ 1 _, ⊥}_ _[m]_ _→_ [0 _,_ 1] for know-it-or-not reports is proper if there exists a
proper scoring rule _S_ : [0 _,_ 1] _[m]_ _× {_ 0 _,_ 1 _}_ _[m]_ _→_ [0 _,_ 1], such that


_Sp_ ( _**r**_ _,_ _**θ**_ ) = _S_ (˜ _r_ _**p**_ ( _**r**_ ) _,_ _**θ**_ ) _,_


where ˜ _r_ _**p**_ maps a report to the probabilistic belief, particularly, _⊥_ to the prior:

_r_ ˜ _**p**_ ( _ri_ ) =              - _prii_ else, whenif _ri_ _∈{ r_ 0 _i,_ 1= _} ⊥._


A scoring rule for multi-dimensional summary points can be defined from single-dimensional scoring
rules and multi-dimensional aggregations.


1


_S_ (0 _,_ 0)

_S_ ( _⊥,_ 0)


_S_ (1 _,_ 0)

0

0 prior _p_ 1


state; belief


1


_S_ (0 _,_ 0)

1 _/_ 2

_S_ (1 _,_ 0)


0

0 prior _p_ 1


state; belief


Figure 1: The V-shaped scoring rule, the optimal
scoring rule in Li et al. (2022). Once fixing a report, the expected score is a linear line in both the
realized state and the mean of the ground truth distribution. Reporting the prior always gets a score
of [1] _/_ 2 (the dotted line). The V-shaped upper envelope of the two linear lines forms the expected
score of a truthful agent.


Figure 2: An example of a single-dimensional
scoring rule for know-it-or-not reports. Each report in the ternary space corresponds to a linear
line. The scoring rule can be depicted by three
linear lines. Properness requires that, when the
belief (or, equivalently, the ground truth) is _r_, the
line with the highest expected score is on the line
corresponding to report _r_ .


**Single-Dimensional** **Scoring** **Rule** We introduce the V-shaped scoring rule and the singledimensional scoring rule for know-it-or-not reports here.


The V-shaped scoring rule is introduced in Li et al. (2022) as the optimal scoring rule that incentivizes
a binary effort, when the agent can choose to exert effort and update her belief from prior to posterior.
Wu & Hartline (2024) tests aggregations over the V-shaped scoring rule. The V-shaped scoring rule
partitions the report space into a ternary space: a report higher than the prior mean, lower than the
prior mean, or the prior mean _p_ . Figure 1 depicts a V-shaped scoring rule with _p <_ [1] 2 [.]

**Definition** **2.4** (V-shaped Scoring Rule) **.** Given prior mean _p_ _∈_ [0 _,_ 1], a V-shaped scoring rule
_S_ : [0 _,_ 1] _×_ [0 _,_ 1] _→_ [0 _,_ [1] 2 []][ is defined by]


2 _[·]_ 1 _−p_ if _r_ _> p_

12 else


1
2 _[−]_ 2 [1]


_Sp_ ( _r, θ_ ) =









2 _[−]_ 2 _[·]_ 1 _−p_ if _r_ _< p_

12 [+] [1] 2 _[·]_ _[θ]_ 1 _−_ _[−]_ _p_ _[p]_ if _r_ _> p_


[1] 2 _[·]_ _[θ]_ 1 _−_ _[−]_ _p_ _[p]_

[1] _[θ][−][p]_

2 _[·]_ 1 _−p_


When _p ∈_ ( 2 [1] _[,]_ [ 1]][, the score is symmetric, i.e.] _[ S][p]_ [(] _[r, θ]_ [) =] _[ S]_ [1] _[−][p]_ [(1] _[ −]_ _[r,]_ [ 1] _[ −]_ _[θ]_ [)][.]


The max-over-separate scoring rule scores an agent by the dimension on which the agent has the
highest expected score. It can be implemented by asking the agent to pick her favorite dimension
and score on that dimension. Wu & Hartline (2024) tests the max-over-separate V-shaped scoring
rule (MV), the optimal scoring rule in the multi-dimensional report. We will compare our Aligned
Scoring Rule with the MV scoring rule.


**Definition 2.8** (Max-Over-Separate) **.** Given scoring rules _S_ 1 _, . . ., Sm_, a max-over-separate scoring
rule is
_S_ ( _**r**_ _,_ _**θ**_ ) = _Si_ ( _ri, θi_ ) _,_ where _i_ = arg max **E** _θi′_ [ _Si′_ ( _ri′, θi′_ )] _._
_i_ _[′]_


3 ALIGNED SCORING RULE: ALGORITHM


In this section, we present our design of Aligned Scoring Rule (ASR), which reduces textual elicitation
to numerical elicitation and optimizes for human alignment in peer grading. Section 3.1 list the
provable properness guarantees of the reduction from Wu & Hartline (2024). Section 3.2 describes
our optimization method for alignment.


Following Wu & Hartline (2024), we model the language model as an oracle accepting _Summarization_
and _Question-Answering_ queries, which are fundamental natural language processing tasks (Bar-Haim
et al., 2020; Clark et al., 2019; Rajpurkar et al., 2016). The Summarization oracle outputs a list of
summary points from a list of texts. The Question-Answering oracle identifies whether a text agrees
or disagrees with a summary point.


**Summarization** _OS_, summarizes a list of textual report into summary points.


**Input** A list of texts T1 _, . . .,_ T _N_ .
**Output** A list [t1 _, . . .,_ t _m_ ] of all summary points from texts.


**Question-Answering** _OA_ determines whether a text agrees or disagrees with a summary point, or is
not applicable.


**Input** One text T and a summary point t.
**Output** Output “disagree” 0, “agree” 1, or “NA” _⊥_ .


We describe Elicitation [GPT] from Wu & Hartline (2024) here. Following Assumption 2.2, we map
a report _⊥_ to the prior report, the empirical frequency of a summary point. The clustered nature
of the peer grading application enables the identification of the empirical frequency. The dataset is
partitioned in advance into clusters. Each cluster contains _N_ peer grading tasks, where the homework
submission are all from the same assignment, thus applicable to the same set of grading rubrics.


5


**Definition 2.5.** With prior _p_, a single-dimensional scoring rule for know-it-or-not reports is proper if


_S_ ( _θ, θ_ ) _≥_ _S_ ( _r, θ_ ) _,_ _∀θ_ _∈{_ 0 _,_ 1 _}, ∀r_ _∈{_ 0 _,_ 1 _, ⊥}_
**E** _θ∼p_ [ _S_ ( _⊥, θ_ )] _≥_ **E** _θ∼p_ [ _S_ ( _r, θ_ )] _,_ _∀r_ _∈{_ 0 _,_ 1 _, ⊥}_


**Multi-Dimensional Aggregations** A multi-dimensional aggregation operates over single dimensional scoring rules and preserves properness.


**Definition 2.6.** Given single dimensional scoring rules _S_ 1 _, . . ., Sm_ where each _Si_ : [0 _,_ 1] _×_ [0 _,_ 1] _→_

[0 _,_ 1], a multi-dimensional scoring rule _S_ : [0 _,_ 1] _[m]_ _×_ [0 _,_ 1] _[m]_ _→_ [0 _,_ 1] is aggregated from _S_ 1 _, . . ., Sm_
if 1) _S_ is proper, and 2) there exists aggregation function _A_ such that


_S_ ( _r_ 1 _, . . ., rn_ ; _·_ ) = _A_         - _S_ 1( _r_ 1; _·_ ) _, . . ., Sn_ ( _rn_ ; _·_ )� _._


We introduce two aggregations, the separate aggregation and the max-over-separate (M) aggregation.


We optimize over the space of separate scoring rules (Li et al., 2022). Wu & Hartline (2024) also
tests the averaged V-shaped scoring rule (AV).


**Definition** **2.7.** Given scoring rules _S_ 1 _, . . ., Sm_, a separate scoring rule is the weighed average
_S_ = [�] _i∈_ [ _m_ ] _[w][i][S][i]_ [, with weights] _[ w]_ [1] _[, . . ., w][m]_ [ such that][ �] _i∈_ [ _m_ ] _[w][i]_ [= 1][.]


_i∈_ [ _m_ ] _[w][i][S][i]_ [, with weights] _[ w]_ [1] _[, . . ., w][m]_ [ such that][ �]


_i∈_ [ _m_ ] _[w][i]_ [= 1][.]


When the language oracle _OA_ is non-inverting on the report side, Elicitation [GPT] is proper.

**Definition** **3.1** (Non-inverting _OA_ ) **.** The question-answering oracle for know-it-or-not beliefs is
non-inverting if the probability of inverting the report is strictly less than [1] 2 [, i.e.][ Pr[ˆ] _[r][i]_ [=] _[ r][i][|]_ [R][]] _[ ≤]_ [1] 2

for any _i_ and any R.

**Theorem 3.2** (Wu & Hartline 2024) **.** _If the question-answering oracle for know-it-or-not beliefs is_
_non-inverting for reports, Elicitation_ _[GPT]_ _is proper._


Without assumptions on the language oracle’s error, the reduction above has adversarial robustness.


**Theorem 3.3** (Wu & Hartline 2024) **.** _If the agent has no information, the highest expected score she_
_can achieve is at most by saying ⊥_ _(i.e. “I don’t know”)._


3.2 OPTIMIZATION FOR ALIGNMENT


While Elicitation [GPT] presents a framework for reducing textual elicitation to numerical elicitation,
not all proper scoring rules align well with the instructor preferences. Thus, our Aligned Scoring
Rule (ASR) optimizes over a space of separate scoring rules and selects the one that aligns best with
the reference score, i.e., the instructor score of a peer review. Our optimization framework follows
the computation of optimal scoring rule in Li et al. (2022). Our Aligned scoring rule can be viewed
as a truthful proxy of the instructor score.


Fixing summary points _{_ t1 _, . . .,_ t _m}_ and prior _**p**_, our optimization objective minimizes the mean
squared error (MSE) between Elicitation [GPT] score and the reference score (e.g. instructor score). Our
optimization problem is shown in Program 1 with _s_ normalized to [0 _,_ 1].


                     min **E** ( _**r**_ _,_ _**θ**_ _,s_ ) ( _S_ ( _**r**_ _,_ _**θ**_ ) _−_ _s_ ) [2][�] (1)
_{S}i∈_ [ _m_ ]


s.t. _S_ is proper
_S_ ( _·, ·_ ) _∈_ [0 _,_ 1]


We optimize over the space of separate scoring rules, the sum of single-dimensional proper scoring
rules _{Si}i∈_ [ _m_ ] for know-it-or-not reports. A separate scoring rule is simple and interpretable, where
the convexity of single-dimensional scores can identify the importance of each dimension. Program


1Note that the ground truth may have _⊥_ in our implementation. In such a case, we score the student by the
expected score where the binary state is drawn from the prior.


6


**Input** _N_ ground truth reviews _{_ I1 _, . . .,_ I _N_ _}_ on submissions to the same homework assignment;
one reported review R _k_ on the _k_ th submission; and a proper scoring rule _S_ for know-it-or-know
beliefs. We will write the identified states and reports by the language oracle as _**θ**_ [ˆ] and _**r**_ ˆ, respectively.


**Algorithm (Elicitation** **[GPT]** **)**


- (Summarization) Summarize instructor reviews into points. _{_ t1 _, . . .,_ t _m}_ = _OS_ ( _{_ I _i}i∈_ [ _N_ ]).


- (Question-Answering) Map truth I _i_ to state space. For each instructor review _j_ _∈_ [ _N_ ] and each
summary point _i ∈_ [ _m_ ], _θi_ _[j]_ [=] _[ O][A]_ [(][I] _[j][,]_ [ t] _[i]_ [)][.] [Calculate the prior of each state] _[ p][i]_ [=] _N_ [1]  - _j_ _[θ]_ _i_ _[j]_ [.]

- (Question-Answering) Map the review to report space. For each point _i ∈_ [ _m_ ], ˆ _ri_ = _OA_ (R _k,_ t _i_ ).

- Apply proper scoring rule for know-it-or-not reports. Output _S_ _**p**_ ( _**r**_ _,_ _**θ**_ _[k]_ ) [1] .


3.1 PROVABLE PROPERNESS


We list the provable property of the reduction here, including the case that the language oracle makes
errors and adversarial robustness.

The correctness of summarization _OS_ does not affect the truthfulness of Elicitation [GPT] . To see this,
even when _OS_ misidentifies the summary points, Elicitation [GPT] is still proper as long as _OA_ correctly
identifies the numerical states and reports corresponding to the summaries. We assume _OA_ is perfect
on the ground truth side, as the ground truth reviews often clearly state opinions on summary points.


Our optimization problem with separate scoring rules is convex. Note that this formulation may not
be convex for other spaces of multi-dimensional scoring rules, e.g. max-over-separate scoring rules.

**Corollary 3.4.** _Optimization problem 2 is convex._


To see Corollary 3.4, note that for each dimension, we have six variables: _Si_ ( _ri, θi_ ) for _ri_ _∈{_ 0 _,_ 1 _, ⊥}_
and _θi_ _∈{_ 0 _,_ 1 _}_ . Both our objective and constraints are convex in the variables. Since optimization
problem 2 is convex, we optimize with the gradient descent algorithm over samples.


4 IMPLEMENTATION OF LANGUAGE ORACLES


We describe our implementation of the language oracle here.


4.1 SUMMARIZATION ORACLE


The implementation of the summarization oracle includes three steps: summarizing instructor reviews,
preparing negative/positive statement pairs from reviews, and clustering negative/positive statement
pairs. Note that instead of directly clustering summary statements by similar meanings, for each
statement from the reviews, we concatenate the statement with another of the opposite meaning to
prepare a pair of negative/positive statements. The negative/positive statement pairs improve the
robustness of LLM clustering. When each summary point consists of negative/positive statement
pairs, the semantic meaning of each state can be viewed as neutral, avoiding opposite statements
being identified as different states for elicitation.


**Input** A list of _N_ instructor reviews [I1 _, . . .,_ I _N_ ].


**Output** A list [t _j_ ] _j∈m_ of summary points from reviews.


**Implementation** We provide a toy prompt with each step below. The real prompts we use are listed
in Appendix A.


- Summarize each instructor review into summary points.
_**Toy prompt**_ _:_ _Carefully read the entire review comment._ _Extract all evaluative statements from the_
_review._ _These should be comments that assess the quality, strengths, weaknesses, and suggestions._
_Ignore purely descriptive statements._ _Create an indexed list of these evaluative statements._

- Transform each statement into negative/positive pairs.
_**Toy prompt**_ _:_ _You are tasked with creating opposite evaluative statements for a given list of evaluative_
_statements._ _For_ _each_ _statement_ _provided,_ _you_ _need_ _to_ _create_ _a_ _new_ _statement_ _that_ _has_ _the_ _same_
_content but expresses the opposite emotion or sentiment._

- Cluster the negative/positive pairs of summary points. The semantic meaning of each cluster is
identified as the dimension for elicitation, [t _j_ ] _j∈_ [ _m_ ].
_**Toy prompt**_ _:_ _You will be given a list of opinion pairs, each with a positive and corresponding negative_
_opinion._ _Your task is to analyze these pairs and cluster them based on similarity._


7


2 shows the simplified optimization problem for separate scoring rules. The properness constraint
follows properness for know-it-or-not reports in Definition 2.5.





2 []





min **E** ( _**r**_ _,_ _**θ**_ _,s_ )
_{Si}i∈_ [ _m_ ]


 [�]





_Si_ ( _ri, θi_ ) _−_ _s_

_i∈_ [ _m_ ]


 (2)


s.t. for any dimension _i,_ (Properness)
for any _ri_ _∈{_ 0 _,_ 1 _, ⊥}_
_Si_ ( _θi, θi_ ) _≥_ _Si_ ( _ri, θi_ ) _, ∀θi_ _∈{_ 0 _,_ 1 _}_
**E** _θi∼pi_ [ _Si_ ( _⊥, θi_ )] _≥_ **E** _θi∼pi_ [ _Si_ ( _ri, θi_ )]

 
(Boundedness)
_i∈_ [ _m_ ] _[S][i]_ [(] _[r][i][, θ][i]_ [)] _[ ∈]_ [[0] _[,]_ [ 1]] _[,][ ∀]_ _**[r]**_ _[,]_ _**[ θ]**_


4.2 QUESTION-ANSWERING ORACLE


We directly query LLM to identify whether a review R is positive or negative for a summary point t.


**Input** One review R and a summary point t.


**Output** Positive (1), negative (0), or NA ( _⊥_ ).


**Implementation** We provide an toy prompt below. The real prompt we use are listed in Appendix A.


_**Toy prompt**_ _:_ _Your task is to infer which of the given positive/negative opinions is correct based on the_
_provided review comment._ _For each opinion pair, read and understand both the positive and negative_
_opinions._ _Conclude whether the review supports the positive, the negative, or neither opinion._


5 EMPIRICAL EVALUATION


We describe our dataset and evaluation metric in Section 5.1, our reference scores used for alignment
in Section 5.2, and our experimental results in Section 5.3. We depict the Aligned Scoring Rule
(ASR) for one example homework assignment in Appendix C.


5.1 DATASET AND EVALUATION METRIC


**Dataset** We present results from peer grading data in two undergraduate algorithm classes. Our
dataset includes 22 assignments in total. [2] Each assignment has 6 to 8 homework submissions. Each
homework submission has one instructor review (i.e. ground truth) and 6 to 8 peer reviews. Each
peer review has an instructor score in [0 _,_ 10].


**Metric** We report the _Mean Squared Error_, the _Pearson correlation coefficient_, and the _Spearman_
_rank correlation coefficient_ of our ASR compared with reference scores.


- MSE quantifies the average magnitude of prediction errors.


- Pearson correlation assesses the strength of the linear relationship between predicted scores and
reference scores, capturing whether the model correctly preserves the relative ordering.


- Spearman rank correlation assesses the correlation between two ranks.


5.2 REFERENCE SCORE


We optimize for alignment with two reference scores, the Instructor Score and the LLM-Judge Score.


**Instructor Score** Instructor score (i.e., human preference) from our dataset.


**LLM-Judge Score** We query a language model to grade the peer review against the instructor review
based on a given peer review scoring rubric.


There is a high correlation between the Instructor Score and LLM-Judge score. Figure 3 presents
the empirical joint distribution of Instructor Score and LLM-Judge Score for all data, with a Pearson
correlation of 0.5540. The results show that LLM-Judge score can serve as a substitute for the costly
and noisy instructor score, improving the scalability and the robustness of the peer grading system,
which is consistent with previous studies of the LLM-as-Judge method, e.g., Zheng et al., 2023;
Hackl et al., 2023, etc.


Note, the instructor and LLM-judge reference scores are not proper and therefore might encourage
peer reviewers to engage in strategic behavior like guessing or adding irrelevant statements (Wu &
Hartline, 2024). Our method of aligning a proper scoring rule to these references can be viewed as
converting these non-proper scores into proper ones.


2Algorithm Class 1: 276 reviews by 23 peers on 89 submissions across 12 assignments. Algorithm Class 2:
240 reviews by 24 peers on 59 submissions across 10 assignments.


8


Table 1: Comparison with baselines.


**Comparison with Baselines** Our Aligned Scoring Rule is compared against the following two
baselines which are all truthful:


1. **Best Constant Score (** _S_ **const)** . This method outputs the best constant score for all reviews, which is
the mean of the reference scores _s_ in the training data _D_ . The constant score is weakly truthful.

_S_ const( _r_ T _, θ_ T) =            
( _r,θ,s_ ) _∈D_ _[s/][|][D][|][.]_


2. **Non-aligned ElicitationGPT (EGPT)** . We compare with the Elicitation [GPT] in Wu & Hartline (2024),
which is not aligned to a reference, particularly, the averaged V-shaped scoring rule (AV) and the
max-over-separate V-shaped scoring rule (MV). In Wu & Hartline (2024), the AV scoring rule is
shown to align the best with instructor score. Note that the max-over-separate scoring rule is not in
our hypothesis space of separate scoring rules, and does not induce a convex optimization problem. [3]


The performance of scores is evaluated along three metrics: MSE, the Pearson correlation coefficient,
and the Spearman rank correlation coefficient. Our ASR aligns best with the reference on all metrics.


3We evaluate Spearman correlation differently from Wu & Hartline (2024). They evaluate the ranking of the
same student’s averaged scores over all peer reviews in a class, because the Elicitation [GPT] scores are not in the
same scale as reference scores. We evaluate each individual peer review’s ranking, as our score is aligned.


9


Figure 3: Joint distribution (instructor
score vs. LLM-Judge score).


5.3 EXPERIMENTAL RESULTS


Figure 4: Reference Scores vs. ASR. Left: instructor score
vs. ASR aligned with instructor score. Right: LLM-Judge
score vs. ASR aligned with LLM-Judge score. The green
line represents the linear regression fitting reference score
from ASR, which is nearly the identity function in both plots.


We present our experimental results in this section. First, we show that a linear regression fitting the
reference score from our ASR results in a nearly-identity linear fit. We then present the MSE and the
correlation coefficients and compare with baselines. We use the Gemini-2.5 series models for the
LLM-Judge and the language oracles and provide experimental details in Appendix A. We also tested
the performance of GPT-4.1 as the LLM-Judge, with the results detailed in Appendix B.


**Nearly-Identity Linear Fit** The first criterion for evaluating our approach is to examine whether
our ASR can effectively fit the original reference scores. Figure 4 illustrates the joint empirical
distribution of the ASR scores and the reference scores, with a regression line predicting the reference
score _s_ from the ASR score _S_ . The parameters of linear regression align closely with _s_ = _S_ .


(a) Reference: Instructor Score

Method SquaredLoss PearsonCorr SpearmanCorr
ASR 1.730 0.717 0.622
Constant 3.741 N/A N/A
EGPT(AV) 9.541 0.294 0.301
EGPT(MV) 18.360 0.213 0.207


(b) Reference: LLM-Judge Score

Method SquaredLoss PearsonCorr SpearmanCorr
ASR 2.003 0.705 0.658
Constant 4.136 N/A N/A
EGPT(AV) 7.053 0.328 0.338
EGPT(MV) 17.069 0.246 0.226


REFERENCES


Jacob D Abernethy and Rafael M Frongillo. A characterization of scoring rules for linear properties.
In _Conference on Learning Theory_, pp. 27–1, 2012.


Maria-Florina Balcan, Avrim Blum, Jason D Hartline, and Yishay Mansour. Reducing mechanism
design to algorithm design via machine learning. _Journal of Computer and System Sciences_, 74(8):
1245–1270, 2008.


Maria-Florina Balcan, Travis Dick, and Ellen Vitercik. Dispersion for Data-Driven Algorithm Design,
Online Learning, and Private Optimization. In _2018 IEEE 59th Annual Symposium on Foundations_
_of Computer Science (FOCS)_, pp. 603–614, October 2018a. doi: 10.1109/FOCS.2018.00064.


Maria-Florina Balcan, Tuomas Sandholm, and Ellen Vitercik. A General Theory of Sample Complexity for Multi-Item Profit Maximization. In _Proceedings of the 2018 ACM Conference on Economics_
_and Computation_, pp. 173–174, Ithaca NY USA, June 2018b. ACM. ISBN 978-1-4503-5829-3.
doi: 10.1145/3219166.3219217.


Maria-Florina F. Balcan, Tuomas Sandholm, and Ellen Vitercik. Sample complexity of automated
mechanism design. _Advances in Neural Information Processing Systems_, 29, 2016.


Roy Bar-Haim, Lilach Eden, Roni Friedman, Yoav Kantor, Dan Lahav, and Noam Slonim.
From arguments to key points: Towards automatic argument summarization. In Dan Jurafsky, Joyce Chai, Natalie Schluter, and Joel Tetreault (eds.), _Proceedings_ _of_ _the_ _58th_ _An-_
_nual_ _Meeting_ _of_ _the_ _Association_ _for_ _Computational_ _Linguistics_, pp. 4029–4039, Online, July
2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.acl-main.371. URL
[https://aclanthology.org/2020.acl-main.371.](https://aclanthology.org/2020.acl-main.371)


Siyu Chen, Jibang Wu, Yifan Wu, and Zhuoran Yang. Learning to incentivize information acquisition:
Proper scoring rules meet principal-agent model. In _International Conference on Machine Learning_,
pp. 5194–5218. PMLR, 2023.


Yiling Chen and Fang-Yi Yu. Optimal scoring rule design. _arXiv preprint arXiv:2107.07420_, 2021.


Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and Kristina
Toutanova. Boolq: Exploring the surprising difficulty of natural yes/no questions. In _Proceedings_
_of_ _the_ _2019_ _Conference_ _of_ _the_ _North_ _American_ _Chapter_ _of_ _the_ _Association_ _for_ _Computational_
_Linguistics:_ _Human Language Technologies, Volume 1 (Long and Short Papers)_, pp. 2924–2936,
2019.


Vincent Conitzer and Tuomas Sandholm. Automated mechanism design: Complexity results stemming from the single-agent setting. In _Proceedings_ _of_ _the_ _5th_ _International_ _Conference_ _on_
_Electronic Commerce_, ICEC ’03, pp. 17–24, New York, NY, USA, September 2003a. Association
for Computing Machinery. ISBN 978-1-58113-788-0. doi: 10.1145/948005.948008.


Vincent Conitzer and Tuomas Sandholm. Automated mechanism design for a self-interested designer.
In _Proceedings of the 4th ACM Conference on Electronic Commerce_, EC ’03, pp. 232–233, New
York, NY, USA, June 2003b. Association for Computing Machinery. ISBN 978-1-58113-679-1.
doi: 10.1145/779928.779974.


Vincent Conitzer and Tuomas Sandholm. Self-interested automated mechanism design and implications for optimal combinatorial auctions. In _Proceedings of the 5th ACM Conference on Electronic_
_Commerce_, EC ’04, pp. 132–141, New York, NY, USA, May 2004. Association for Computing
Machinery. ISBN 978-1-58113-771-2. doi: 10.1145/988772.988793.


Michael Curry, Tuomas Sandholm, and John Dickerson. Differentiable Economics for Randomized
Affine Maximizer Auctions. In _International Joint Conference on Artificial Intelligence (IJCAI)_,
2022.


Michael Curry, Vinzenz Thoma, Darshan Chakrabarti, Stephen McAleer, Christian Kroer, Tuomas
Sandholm, Niao He, and Sven Seuken. Automated Design of Affine Maximizer Mechanisms in
Dynamic Settings. _Proceedings of the AAAI Conference on Artificial Intelligence_, 38(9):9626–9635,
March 2024. ISSN 2374-3468. doi: 10.1609/aaai.v38i9.28819.


10


David Danz, Lise Vesterlund, and Alistair J Wilson. Belief elicitation and behavioral incentive
compatibility. _American Economic Review_, 112(9):2851–2883, 2022.


Zhijian Duan, Haoran Sun, Yurong Chen, and Xiaotie Deng. A Scalable Neural Network for DSIC
Affine Maximizer Auction Design. _Advances_ _in_ _Neural_ _Information_ _Processing_ _Systems_, 36:
56169–56185, December 2023.


Paul Dutting, Zhe Feng, Harikrishna Narasimhan, David C Parkes, and Sai Srivatsa Ravindranath.¨
Optimal auctions through deep learning: Advances in differentiable economics. _Journal of the_
_ACM_, 71(1):1–53, 2024.


Michal Feldman, Nick Gravin, and Brendan Lucier. Combinatorial auctions via posted prices.
In _Proceedings_ _of_ _the_ _twenty-sixth_ _annual_ _ACM-SIAM_ _symposium_ _on_ _Discrete_ _algorithms_, pp.
123–135. SIAM, 2014.


Jinlan Fu, See Kiong Ng, Zhengbao Jiang, and Pengfei Liu. Gptscore: Evaluate as you desire.
In _Proceedings_ _of_ _the_ _2024_ _Conference_ _of_ _the_ _North_ _American_ _Chapter_ _of_ _the_ _Association_ _for_
_Computational Linguistics:_ _Human Language Technologies (Volume 1:_ _Long Papers)_, pp. 6556–
6576, 2024.


Veronika Hackl, Alexandra Elena Muller,¨ Michael Granitzer, and Maximilian Sailer. Is gpt-4 a
reliable rater? evaluating consistency in gpt-4’s text ratings. In _Frontiers in Education_, volume 8,
pp. 1272229. Frontiers Media SA, 2023.


Jason D Hartline, Liren Shan, Yingkai Li, and Yifan Wu. Optimal scoring rules for multi-dimensional
effort. In _The Thirty Sixth Annual Conference on Learning Theory_, pp. 2624–2650. PMLR, 2023.


Tanjim Hossain and Ryo Okui. The binarized scoring rule. _Review_ _of_ _Economic_ _Studies_, 80(3):
984–1001, 2013.


Justin Hsu, Jamie Morgenstern, Ryan Rogers, Aaron Roth, and Rakesh Vohra. Do prices coordinate
markets? In _Proceedings of the forty-eighth annual ACM symposium on Theory of Computing_, pp.
440–453, 2016.


Dhamma Kimpara, Rafael Frongillo, and Bo Waggoner. Proper losses for discrete generative models.
In _International Conference on Machine Learning_, pp. 17015–17040. PMLR, 2023.


Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris
Alberti, Danielle Epstein, Illia Polosukhin, Matthew Kelcey, Jacob Devlin, Kenton Lee, Kristina N.
Toutanova, Llion Jones, Ming-Wei Chang, Andrew Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov.
Natural questions: a benchmark for question answering research. _Transactions of the Association_
_of Computational Linguistics_, 2019.


Yingkai Li, Jason D Hartline, Liren Shan, and Yifan Wu. Optimization of scoring rules. In
_Proceedings of the 23rd ACM Conference on Economics and Computation_, pp. 988–989, 2022.


Yuxuan Lu, Shengwei Xu, Yichi Zhang, Yuqing Kong, and Grant Schoenebeck. Eliciting informative
text evaluations with large language models. _the_ _25th_ _ACM_ _Conference_ _on_ _Economics_ _and_
_Computation_, 2024.


John McCarthy. Measures of the value of information. _Proceedings of the National Academy of_
_Sciences of the United States of America_, 42(9):654, 1956.


Markus M Mobius, Muriel Niederle, Paul Niehaus, and Tanya S Rosenblat. Managing self-confidence:¨
Theory and experimental evidence. _Management Science_, 68(11):7793–7817, 2022.


Jamie Morgenstern and Tim Roughgarden. Learning Simple Auctions. In _Conference on Learning_
_Theory_, pp. 1298–1318. PMLR, June 2016.


Jamie H Morgenstern and Tim Roughgarden. On the pseudo-dimension of nearly optimal auctions.
In _Advances in Neural Information Processing Systems_, 2015.


Eric Neyman, Georgy Noarov, and S Matthew Weinberg. Binary scoring rules that incentivize
precision. In _Proceedings_ _of_ _the_ _22nd_ _ACM_ _Conference_ _on_ _Economics_ _and_ _Computation_, pp.
718–733, 2021.


11


Maneesha Papireddygari and Bo Waggoner. Contracts with information acquisition, via scoring rules.
In _Proceedings of the 23rd ACM Conference on Economics and Computation_, pp. 703–704, 2022.


Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. SQuAD: 100,000+ questions
for machine comprehension of text. In Jian Su, Kevin Duh, and Xavier Carreras (eds.), _Proceedings_
_of the 2016 Conference on Empirical Methods in Natural Language Processing_, pp. 2383–2392,
Austin, Texas, November 2016. Association for Computational Linguistics. doi: 10.18653/v1/
D16-1264. [URL https://aclanthology.org/D16-1264.](https://aclanthology.org/D16-1264)


Tim Roughgarden and Okke Schrijvers. Ironing in the Dark. In _Proceedings_ _of_ _the_ _2016_ _ACM_
_Conference on Economics and Computation_, EC ’16, pp. 1–18, New York, NY, USA, July 2016.
Association for Computing Machinery. ISBN 978-1-4503-3936-0. doi: 10.1145/2940716.2940723.


Tuomas W Sandholm, Vincent Conitzer, and Craig Boutilier. Automated Design of Multistage
Mechanisms. In _International Joint Conference on Artificial Intelligence (IJCAI)_, 2007.


Leonard J Savage. Elicitation of personal probabilities and expectations. _Journal of the American_
_Statistical Association_, 66(336):783–801, 1971.


Johannes Schneider, Bernd Schenk, Christina Niklaus, and Michaelis Vlachos. Towards llm-based
autograding for short textual answers. _arXiv preprint arXiv:2309.11508_, 2023.


Weiran Shen, Pingzhong Tang, and Song Zuo. Automated Mechanism Design via Neural Networks.
In _Proceedings_ _of_ _the_ _18th_ _International_ _Conference_ _on_ _Autonomous_ _Agents_ _and_ _MultiAgent_
_Systems (AAMAS)_, 2019.


Tonghan Wang, Yanchen Jiang, and David C. Parkes. GemNet: Menu-Based, Strategy-Proof MultiBidder Auctions Through Deep Learning. In _Proceedings_ _of_ _the_ _2024_ _ACM_ _Conference_ _on_
_Economics and Computation_, 2024.


Yifan Wu and Jason Hartline. Elicitationgpt: Text elicitation mechanisms via language models.
_working paper_, 2024.


Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang,
Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. Judging llm-as-a-judge with mt-bench and
chatbot arena. _Advances in Neural Information Processing Systems_, 36:46595–46623, 2023.


12


A IMPLEMENTATION DETAILS


In this section, we provide a detailed description of how we implement our methods and conduct the
experiments, including the prompts and other parameters for LLM calls, the numerical solution to the
convex optimization problem, as well as the pre/post-processing of human feedback.


A.1 LLM CALLS


We use the gemini-2.5 series models as the LLM oracles in our experiments. Specifically, we experiment with gemini-2.5-flash-preview-04-17 for all tasks other than clustering the negative/potitive
pairs. For clustering, we employed gemini-2.5-pro-preview-05-06 due to its proficiency in handling
long contexts. While calling LLMs, we set the temperature to 0, the “thinking” feature disabled, and
maximum output token 8192. Next, we will provide a detailed description of each prompt used.


A.1.1 SUMMARIZATION ORACLE


The implementation of the summarization oracle includes three steps: summarizing instructor review,
preparing negative/positive statement pairs from reviews, and clustering negative/positive statement
pairs.


**Summarizing Instructor Review**

You are an AI assistant specializing in analyzing assignment reviews. Your task is to extract all evaluative points from a given review
comment.

_<_ review comment _>_ REVIEW ~~C~~ OMMENT _<_ /review ~~c~~ omment _>_

Please follow these steps to analyze the review comment:

1. Carefully read the entire review comment.

2. Extract all evaluative statements from the review. These should be comments that assess the quality, strengths, weaknesses, and
suggestions. Ignore purely descriptive or meaningless statements. Ignore statements purely about specific scores and ratings.

3. Create an indexed list of these evaluative statements. Each entry should be a single sentence in a single line containing a distinct
evaluation from the review.

   - You should clearly convey the sentiment behind an evaluative statement.

4. After creating the indexed list. Split and Rewrite each evaluative statement into several abstract and concise statements, abandoning
the specific expression.

   - Make your entry abstract and concise.

   - Always use ”part A / B / C” in the output to refer parts, even if the input says ”part a / b / c” or ”part 1 / 2 / 3”.

   - If an evaluative statement contains comments on multiple distinct aspects, they need to be listed as multiple entries.

Example: ”I like the overall idea, but authors need to revise the presentation and experiments” have 3 different aspects, ”The overall
idea is good”, ”The presentation need revision”, and ”The experiments need revision”.

Example: ”Part A is correct and part B is wrong” have 2 different aspects, ”Part A is correct”, and ”Part B is wrong”.

   - Ignore the unimportant positive parts of negative statements and the unimportant negative parts of positive statements.

   - Each new entry inherits the index of the original entry, even if there are duplicate indexes.

Your output should be structured as follows:

_<_ numbered entries _>_ [List your numbered entries here, one per line] _<_ /numbered ~~e~~ ntries _>_

_<_ rewrited ~~e~~ ntries _>_ [Rewrite each entry into an abstract and concise statement] _<_ /rewrited ~~e~~ ntries _>_


13


**Preparing Negative/Positive Statement Pair**

You are tasked with creating opposite evaluative statements for a given list of evaluative statements. For each statement provided, you
need to create a new statement that has the same content but expresses the opposite emotion or sentiment.

In addition, you also need to output whether the sentiment of the original statement is positive or negative.

Guidelines for creating opposite evaluative statements:

1. Maintain the same subject matter and key elements of the original statement.

2. Change the emotional tone or sentiment to its opposite (e.g., positive to negative, approval to disapproval).

3. Use similar language structure when possible, but modify words to reflect the opposite sentiment.

4. Ensure the new statement is coherent and makes sense in isolation.

5. Make the new statement as concise as possible.

Here is the list of evaluative statements:

_<_ evaluative ~~s~~ tatements _>_

EVALUATIVE ~~S~~ TATEMENTS

_<_ /evaluative ~~s~~ tatements _>_

For each statement in the list, create an opposite version following the guidelines above. Present your results in the following format:

_<_ result ~~1~~ _>_

_<_ original _>_ [Original evaluative statement] _<_ /original _>_

_<_ sentiment _>_ [Sentiment of the original evaluative statement] _<_ /sentiment _>_

_<_ opposite _>_ [Your created opposite evaluative statement] _<_ /opposite _>_

_<_ /result ~~1~~ _>_

_<_ result ~~2~~ _>_

...

_<_ /result ~~2~~ _>_

...

Ensure that each opposite statement accurately reflects a reversal of sentiment while maintaining the core content of the original
statement.


14


**Clustering Statement Pairs**

You will be given a list of opinion pairs, where each pair consists of a positive opinion and its corresponding negative opinion. Your
task is to analyze these pairs and cluster them based on similarity. Follow these steps:

1. First, read the list of opinion pairs provided:

_<_ opinion ~~p~~ airs _>_ OPINION ~~P~~ AIRS _<_ /opinion ~~p~~ airs _>_

2. Next, cluster the unique pairs based on their similarity in topic or theme in _<_ clustering _>_ tag. Pairs in the same cluster should
address roughly the same aspects of the subject matter. Follow these steps:

1) You need to first draft a set of cluster descriptions in the _<_ draft _>_ tag. Each cluster description must be specific:

   - You should cluster opinion pairs discussing different parts in different clusters.

   - The description should clearly indicate the target of evaluation, avoiding terms like ”overall” or ”assignment” and instead using ”the
proof,” ”part A,” or ”the answer.”

   - The description should clearly specify the evaluation criteria, avoiding terms like ”quality” and instead using ”correctness,” ”clarity,”
or ”detail.”

2) Then, based on these descriptions, analyze the following aspects in the _<_ analysis _>_ tag:

   - Splitting and merging clusters: Merge clusters that are redundant. Split clusters that contain more than one parts or aspects.

   - New clusters: Look for opinions that are not covered by any existing cluster. Create a new cluster when at least two opinions fit it,
and ignore any lone opinion that cannot be grouped.

   - Specificity check: Ensure each cluster description includes specific evaluation criteria, rather than vague terms.

   - Limit the number of clusters: Ensure the total number of clusters is between 10 and 12.

3) After completing this analysis, redefine the cluster descriptions based on your findings and repeat the entire process.

4) Perform this iteration a total of four times, wrapping the results of each iteration inside _<_ epoch ~~i~~ _>_ tags, where i represents the
iteration number.

You should follow this output format:

_<_ clustering _>_

_<_ epoch ~~1~~ _>_

_<_ draft _>_ [Your draft cluster descriptions] _<_ /draft _>_

_<_ analysis _>_ [Your analysis here] _<_ /analysis _>_

_<_ /epoch ~~1~~ _>_

_<_ epoch ~~2~~ _>_

...

_<_ /epoch ~~2~~ _>_

...

_<_ /clustering _>_

3. Complete your final cluster descriptions. For each cluster, generate an opinion pair as the cluster representative.

   - Ensure the opinion pair discusses exactly the core idea of the cluster description.

   - The opinion pair should be brief and omit details.

   - Do not use ”need” or ”need not” in your opinion pair. Instead, express what was done or what was failed to be done.

   - Ensure the positive opinion and the negative opinion present exact opposing views.

   - It is not necessary to summarize all content. Focus only on evaluating the most important aspect, and avoid using ”and” to connect
different aspects.

   - Avoid using extreme words such as ”excellent” and ”awful.”

You should follow this output format:

_<_ clusters _>_

_<_ cluster ~~1~~ _>_

_<_ description _>_ [The description of the cluster] _<_ /description _>_

_<_ representative _>_ [Positive opinion] [Negative opinion] _<_ /representative _>_

_<_ /cluster ~~1~~ _>_

_<_ cluster ~~2~~ _>_

...

_<_ /cluster ~~2~~ _>_

...

_<_ /clusters _>_


A.1.2 QUESTION-ANSWERING ORACLE


We directly query LLM to identify whether the review R is positive or negative for the summary point
t.


**Input** One review R and a summary point t.


**Output** Positive (1), negative (0), or NA ( _⊥_ ).


15


**Question-Answering Oracle**

You are an AI assistant specializing in analyzing assignment reviews. Your task is to infer which of the given positive/negative
opinions is correct based on the provided review comment. You will be given two inputs:

_<_ review comment _>_ REVIEW ~~C~~ OMMENT _<_ /review ~~c~~ omment _>_

_<_ opinion ~~p~~ airs _>_ OPINION ~~P~~ AIRS _<_ /opinion ~~p~~ airs _>_

The review comment is the text of the review that you need to analyze. The opinion pairs consist of several lines, each containing a
positive evaluation and its corresponding negative evaluation.

For each opinion pair, follow these steps to analyze and conclude in _<_ result _>_ tag:

1. Reprint the index of the opinion pair in _<_ index _>_ tag.

2. Copy the text of the opinion pair in _<_ opinion ~~p~~ air _>_ tag.

3. Carefully read and understand both the positive and negative opinions.

4. List all possibly relevant statements in the comment one by one in the _<_ statements _>_ tag. For each relevant statement, determine
whether it supports the positive opinion, the negative opinion, or neither, and specify whether the support is explicit or partial.

- Focus on the original meaning of the statement and avoid speculation as much as possible.

Example: The correctness of the assignment refers to the accuracy of the final answer and does not include the reasoning process.

Example: The correctness of the proof / claim does not affect the correctness of the answer.

Example: The wrong proof / answer / reasoning does not affect clarity.

5. Apply the following rules to determine the final conclusion in the _<_ rubric _>_ tag:

- If only one direction is supported, classify as that direction, even if it is only partially supported.

- If their are conflicts, classify as the direction with stronger support.

- If no statement is relevant to the opinion pair, classify as ”Neither”. Avoid selecting ”Neither” whenever possible.

- At the end of the rubric, explicitly state you choose ”Positive”, ”Negative”, or ”Neither”.

6. Restate your choice of whether the review supports the positive, the negative, or neither in the _<_ conclusion _>_ tag.

- Only contain ”Positive”, ”Negative”, or ”Neither” in the tag! Do not use words like ”Correct”, ”Incorrect”, ”Clear”, ”Unclear”.

Present your analysis and conclusion for each opinion pair in the following format:

_<_ result _>_

_<_ index _>_ [The index of the input opinion pair here] _<_ /index _>_

_<_ opinion ~~p~~ air _>_ [Copy the input opinion pair here] _<_ /opinion ~~p~~ air _>_

_<_ statements _>_

Statement: [Statement 1]

Analysis: [Analysis for Statement 1]

Statement: [Statement 2]

Analysis: [Analysis for Statement 2]

...

_<_ /statement _>_

_<_ rubric _>_ [Apply the rubric here] _<_ /rubric _>_

_<_ conclusion _>_ [Positive / Negative / Neither] _<_ /conclusion _>_

_<_ /result _>_

_<_ result _>_ ... _<_ /result _>_

...


16


A.1.3 LLM SCORE


**LLM Score**

You are an AI assistant specializing in educational assessment. Your task is to evaluate a peer review of a course assignment by
comparing it to an instructor’s review of the same assignment. You will analyze the alignment between the two reviews and assign a
score from 0 to 10.

First, you will be given the instructor’s review first and then the peer review to be evaluated.

To evaluate the peer review, follow these steps:

1. Identify the points in the instructor’s review in the _<_ evaluation ~~p~~ rocess _>_ tag. Express the same aspect across different parts as
separate points. For each point in the instructor’s review:

1) Reprint the text of this point from the instructor’s review.

2) Judge whether the content of this point is subjective or objective.

   - Objective content includes factual assessments, such as the correctness of the assignment or proofs.   - Subjective content includes
aspects like clarity or style.

3) Identify the importance of this point:

   - Give more weight to critical elements like the correctness of the assignment or proofs.

   - Consider subjective elements and minor discrepancies less impactful on the overall score.

4) Extract all relevant text of this point from the peer review.

5) Assess the following aspects:

a. Content: Does the peer review cover the same main topics of this key point? b. Accuracy: Are the peer reviewer’s observations
and critiques accurate when compared to the instructor’s key point? c. Depth: Does the peer review provide an appropriate level of
detail and insight?

6) Judge the overall quality of the peer review on this point.

2. According to your evaluation, offer a comprehensive assessment of this peer review in the _<_ assessment _>_ tag, supported by
justification.

   - highlighting the alignments or misalignments between the peer review and the instructor’s review.

   - Taking into account both the importance of each key point and the degree of alignment.

3. After the assessment, first provide your reasoning, then assign a score from 0 to 10 based on the rubric, enclosed in the _<_ scoring _>_
tag.

   - 0-1: Totally wrong or meaningless review: The review is irrelevant, incoherent, or shows a complete misunderstanding of the
material.

   - 2-3: Poor review: The review demonstrates significant factual inaccuracies or fails to address essential key points.

   - 4-6: Somewhat valuable review: The review contains clear errors or omissions, but partially aligns with the instructor’s review on
some important points.

   - 7-9: Good review: The review largely aligns with the instructor’s review on key points, with only minor inaccuracies or omissions.

   - 10: Exceptional review: The review is highly consistent with the instructor’s on both content and reasoning, with minimal flaws.

4. Output your final score again in the _<_ final score _>_ tag, with only the number.

Present your final evaluation in the following format:

_<_ evaluation ~~p~~ rocess _>_ Point 1: [Description]

   - Instructor’s review: [Reprint text of this point from the instructor’s review]

   - Objective/subjective: [Reasoning first to judge whether the content of this point is subjective or objective]

   - Importance: [Reasoning first to identify the importance of this point]

   - Peer review: [Extract all relevant text of this point from the peer review]

   - Assessment: [Assess the content, accuracy, and depth in detail]

   - Quality: [Judge the quality of the peer review in relation to this point]

Point 2: [Description] ... _<_ /evaluation process _>_

_<_ assessment _>_ [Your comprehensive assessment of this peer review] _<_ /assessment _><_ scoring _>_ [Your reasoning and the score for
the peer review based on the rubric] _<_ /scoring _><_ final ~~s~~ core _>_ [Output the final score] _<_ /final ~~s~~ core _>_

Here is your input:

_<_ instructor ~~r~~ eview _>_ INSTRUCTOR ~~R~~ EVIEW _<_ /instructor ~~r~~ eview _>_

_<_ peer review _>_ PEER ~~R~~ EVIEW _<_ /peer review _>_


B ADDITIONAL RESULTS


This section presents experimental results that are omitted from the main text.


B.1 LLM-JUDGE SCORES USING GPT


In our primary experiments, we obtain LLM-judge scores by querying the gemini-2.5-flash-preview04-17 model to assess each peer review against its corresponding instructor review, according to a
predefined scoring rubric.


17


Figure 6 presents the same linear regression fitting the reference score from our ASR. The regression
line remains almost identical.


C CASE DEMONSTRATION


We present an example of ASR in this section. Figure 7 visualizes the single-dimensional scoring
rules. The homework assignment is on asymptotic analysis and is divided into three parts _A, B, C_,
each corresponding to the asymptotic relationship between two functions. For each dimension, we
plot the V-shape scoring rule for this dimension.


From the plot, we can observe the dimensions that are not important for scoring, where the scoring
line is almost linear, meaning the score does not depend on the report but only on the state. For
example, we observe that the dimensions for clarity are less important, e.g., “part A details are clear”
and “submission well-structured”.


We also identify important dimensions, where the two linear scoring lines form a more strongly
convex function. We observe that summary points on details related to overall correctness are more
important, e.g., “Algorithm logic is correct”, “solution omits details”, Dim 4 “Part B is correct”, and
“Part A is sound”.


In general, we observe that our ASR when learning Instructor Score assign more convex V-shape
scoring rule to the content that is commonly considered to be more important.


Figure 7: The visualization of ASR on one assignment in the algorithm class using instructor score
as the reference. The score of _r_ = _⊥_ for each dimension has been shifted to zero.


18


To evaluate the robustness of this approach, we repeated the procedure using GPT-4.1 with the same
prompt, thereby constructing a GPT-based LLM-judge. The resulting scores are shown in Figure 5.
LLM-Judge with GPT shows a lower consistency with the instructor score.


Figure 5: Joint distribution (instructor
score vs. LLM-Judge score using GPT4.1).


Figure 6: Reference Scores vs. ASR. Left: instructor score
vs. ASR aligned with instructor score. Right: LLM-Judge
score vs. ASR aligned with LLM-Judge score. The green
line represents the linear regression fitting reference score
from ASR, which is nearly the identity function in both plots.