# SEED-SET: SCALABLE EVOLVING EXPERIMENTAL DESIGN FOR SYSTEM-LEVEL ETHICAL TESTING


**Anjali Parashar** [1] _[∗]_ **Yingke Li** [1] **Eric Yang Yu** [1] **Fei Chen** [1]

**James Neidhoefer** [1] **Devesh Upadhyay** [2] **Chuchu Fan** [1]

1Laboratory for Information & Decision Systems (LIDS), MIT 2Saab Inc.
{anjalip,yingkeli,eyyu,feic,jimneid,chuchu}@mit.edu
devesh.upadhyay@saabinc.com


ABSTRACT


As autonomous systems such as drones, become increasingly deployed in highstakes, human-centric domains, it is critical to evaluate the ethical alignment
since failure to do so imposes imminent danger to human lives, and long term
bias in decision-making. Automated ethical benchmarking of these systems is
understudied due to the lack of ubiquitous, well-defined metrics for evaluation,
and stakeholder-specific subjectivity, which cannot be modeled analytically. To
address these challenges, we propose SEED-SET, a Bayesian experimental design
framework that incorporates domain-specific objective evaluations, and subjective
value judgments from stakeholders. SEED-SET models both evaluation types
separately with hierarchical Gaussian Processes, and uses a novel acquisition
strategy to propose interesting test candidates based on learnt qualitative preferences
and objectives that align with the stakeholder preferences. We validate our approach
for ethical benchmarking of autonomous agents on two applications and find our
method to perform the best. Our method provides an interpretable and efficient
trade-off between exploration and exploitation, by generating up to 2 _×_ optimal test
candidates compared to baselines, withdimensional search spaces [1] . 1 _._ 25 _×_ improvement in coverage of high


1 INTRODUCTION


Artificial intelligence (AI)-enabled autonomous systems have seen increased deployment across a
wide range of high-stakes domains, including automated energy distribution, disaster management
(Battistuzzi et al., 2021). Although such applications can bring significant social benefits (Maslej
et al., 2025; Zeng et al., 2024; Weidinger et al., 2021; Birhane et al., 2024), they raise equally
urgent ethical concerns (Sovacool et al., 2016; Bhattacharya et al., 2024; Amodei et al., 2016; Jobin
et al., 2019; Wang et al., 2025; Grabb et al., 2024; Pałka, 2023) across stakeholder groups. For
example, in the power grid resource allocation problem, energy distribution policies often prioritize
higher-income areas during peak demand periods, leaving marginalized populations more vulnerable
to outages (Fahmin et al., 2024; Chitikena et al., 2023; Cong et al., 2022). Such examples highlight
three core challenges of ethical evaluation in real-world autonomous systems:


 - _Measuring ethical behavior is difficult._ Standard ethical evaluation metrics such as fairness and
social acceptability often lack ground-truth labels (Mittelstadt et al., 2016; Salaudeen et al., 2025;
Reuel et al., 2024; Wallach et al., 2025).

 - _Value alignment is user-dependent, and evolving._ Evaluation standards must quickly adapt to the
growing capabilities of autonomous systems (Keswani et al., 2024; Tarsney et al., 2024). Static
evaluation standards such as test suites and benchmarks require persistent revisions. Additionally,
a wide range of ethical benchmarking problems are user-specific and user evaluation can be noisy.

 - _Ethical evaluation of real-world platforms is expensive._ Due to resource constraints such as budget,
real-world systems require sample-efficient evaluation. Disproportionate access to large-scale
human feedback across domains also imposes sample restrictions on stakeholders.


_∗_ Corresponding author
1Project website: [https://anjaliparashar.github.io/seed-site/](https://anjaliparashar.github.io/seed-site/)


1


Figure 1: **SEED-SET** **Overview.** Our framework integrates quantitative metrics learned using
Objective GP (a) with user preferences learned via a Subjective GP through pairwise elicitation (b).
An LLM performs pairwise comparisons of scenario outcomes (c) to inform the acquisition process
(d), which generates a pair of scenarios for evaluation, aligned with user-defined ethical criteria.
These scenarios are then used for system-level simulations (e) in a sequential manner.


To address the first challenge, guidelines and standards for ethical behavior in AI systems have been
proposed (Tabassi, 2023; Winfield et al., 2021; ISO, 2024; Chance et al., 2023). For example, NIST’s
AI Risk Management Framework (AI RMF 1.0, 2023) that suggests high-level guidelines (e.g.,
Govern, Map, Measure, Manage) to promote ‘trustworthy AI’. Although these guidelines serve as
useful heuristics for enforcing measurable ethical behavior, they are not sufficiently concrete for direct
system-level testing. Some recent efforts in this direction have led to automated evaluation tools,
largely focusing on rule-based ethical benchmarking based solely on established guidelines (Reuel
et al., 2022; Dennis et al., 2016), or preference-based methods using human feedback (Keswani et al.,
2024; Liu et al., 2024), and often argue in favor of one over the other. Instead, we argue that in its
most general form, ethical evaluation must incorporate objective feedback from existing guidelines,
as well as stakeholder concerns.


Additionally, existing works assume abundant access to cheap simulations or expert annotations,
leading to sample-extensive approaches based on reinforcement learning (RL), reinforcement learning
from human feedback (RLHF), or adaptive stress testing (Reuel et al., 2022; Dennis et al., 2016; Gao
et al., 2024). Such assumptions restrict their applicability to real-world systems, underscoring the
need for methods that unify both forms of evaluation under realistic data and resource constraints.


To address this, we propose **Scalable Evolving Experimental Design for System-level Ethical**
**Testing (SEED-SET)**, an evaluation methodology that benchmarks autonomous systems against
both objective measurable metrics (e.g. fire damage to buildings) and subjective ethical metrics (e.g.
rescue priority to different vulnerable groups) while maintaining low sampling requirements. Figure 1
provides an overview of our method.


To our knowledge, this is the first framework of its kind to explicitly consider both objective and
subjective ethical evaluation criteria. A key nuance in this design is the interplay of objective metrics
and stakeholder preferences. Stakeholder preferences are affected by and dictate regions of importance
in the objective landscape, and strategy to explore these regions in the objective metrics must adapt
to stakeholders. This dual dependency is highly non-trivial, and not explicitly acknowledged in
prior works. We incorporate this dependency in the design of our novel data acquisition strategy,
that incorporates feedback from both models for proposing challenging test cases ( Section 4). We
evaluate SEED-SET on three tasks for ethical evaluation: Power system resource allocation, Fire
rescue by aerial autonomous agents, and Optimal route design in urban traffic ( Section 5). Our
method successfully scales to high dimensional scenarios, by proposing up to 2 _×_ more optimal test
cases compared to baselines.


2


Our contributions can be summarized as follows:


 - We introduce a unified, domain-agnostic problem formulation for system-level ethical testing,
modeling it as an adaptive, sample-constrained inference task over both objective metrics and
subjective values.

 - We formalize a hierarchical Variational Gaussian Process (VGP) model that maps design parameters
to measurable ethical criteria and learns their utility according to subjective factors.

 - We derive a novel joint acquisition criterion for hierarchical models that balances exploration of
uncertain ethical factors with exploitation of learned ethical preferences in our hierarchical VGP.


2 RELATED WORK


We discuss key related works here, with a detailed discussion in Appendix A.1.


**Governance approaches for responsible AI.** A wide range of governance frameworks, guidelines
and standards have been proposed to guide the ethical development and deployment of AI systems
(Tabassi, 2023; Organisation for Economic Co-operation and Development, 2019; IEEE Global
Initiative, 2019; Winfield et al., 2021; ISO, 2024)). These works articulate high-level values but are
vague about specific mechanisms for practical enforcement (Hagendorff, 2020). For domain-specific
implementation of these guidelines, automated ethical evaluation tools have been proposed in the
literature.


**Automated** **tools.** Prior technical approaches include reinforcement learning and orchestration
for instilling ethical values (Noothigattu et al., 2019), large-scale studies of moral judgment in
LLMs (Zaim bin Ahmad & Takemoto, 2025), and active learning for preference elicitation (Keswani
et al., 2024). With the exception of active learning, these techniques impose large-scale data and
simulation budget requirements and lack interpretablity and modularity provided by our framework.


3 PROBLEM STATEMENT


We formulate system-level ethical testing as follows:
**Problem** **3.1.** _Given_ _a_ _black-box_ _autonomous_ _system_ _π,_ _parameterized_ _by_ _policy_ _π_ Π _(such_
_S_ _∈_
_as power resource allocation, drone navigation in environment), evaluate its ethical alignment by_
_querying_ _it_ _in_ _scenarios_ _x_ _∈X_ _(environment_ _properties_ _such_ _as_ _location_ _of_ _assets),_ _collecting_
_objective evaluationsfunction f_ : Π _× X_ _→ y_ _∈Y_ R _that captures both objective outcomes and subjective value judgments.(such as cost, resilience), and estimating an unknown ethical compliance_

We list some design choices to meet the key requirements of our problem formulation:


**Multi-faceted ethical criteria.** The overall subjective evaluation depends on some task-specific,
and some task-agnostic parameters. For example, a stakeholder in power resource allocation will
prefer low cost and high grid reliability, regardless of the system specifics, such as grid size. Thus,
we decompose ethical compliance _f_ ( _π, x_ ) into two parts: a set of objective metrics _f_ obj :
_X_ _→Y_
(e.g., cost, reslience) which can be modeled analytically using prior knowledge (domain experts,
guidelines) and user specific subjective evaluation _f_ subj : R (e.g., perceived fairness using
_Y_ _→_
cost, resilience as metrics), with limited access to ground-truth evaluations. For a given _y_, _f_ subj( _y_ )
represents the degree of ethical alignment with the subjective evaluation criteria.


**Sample-constrained learning.** Evaluation is costly: querying _π_ in scenario _x_ incurs a cost _c_ ( _π, x_ ).
_S_
We approach this using a sequential design paradigm. Given a total testing budget _B_, the goal of
Bayesian Experimental Design is to sequentially select query points to best learn _f_ within budget, to
maximize the amount of information obtained about the model parameters of interest. This promotes
sample efficiency by utilizing information from collected data := ( _x_ 1 _, y_ 1) _, . . .,_ ( _xn, yn_ ) .
_D_ _{_ _}_

**Scalability and uncertainty modeling.** The space Π _× X_ may be high-dimensional, and complex,
and evaluations may inherit uncertainty from experts and stakeholders. We model this uncertainty
cumulatively in the objective evaluation as _y_ _∼N_ ( _f_ ( _x_ ) _, σ_ [2] ( _x_ )), by assuming the noise follows a zeromean normal distribution with a standard deviation _σ_ . Ethical testing thus requires explicit uncertainty
modeling (e.g., via Bayesian inference) and scalable function approximation (e.g., variational models),
to guide testing toward the most informative scenarios.


3


**Assumptions.** We make the following assumptions about the system under test _π_, and the user:
_S_

**A1** The policy _π_ is fixed during testing, and the scenario space _X_ is known and fixed a priori.
**A2** The user provides their true subjective ethical preferences (i.e., users do not misreport), as the ethical
evaluation depends on the user-defined notion of “good” or “preferred” behavior. Additionally, we
assume that given a set of objectives, the user’s latent ethical model is stationary, corresponding to
an unknown but fixed subjective criterion.

**A3** We additionally assume access to objective evaluations (e.g., damage caused by fire), which are
required for subjective evaluation. The complete set of objective metrics is assumed to be fully
known a priori.


Assumption A1 pertains to system requirements, and is a widely used, fundamental assumption in
BO/BED literature (Rainforth et al., 2024; Chaloner & Verdinelli, 1995). This is needed to ensure a
fixed feature space for learning surrogate models for objective and subjective criteria.


Assumption A2 is rooted in pairwise elicitation literature that assumes a stationary latent utility
function associated to preferential evaluation (Chu & Ghahramani, 2005). Note that our framework
still accommodates stochasticity of evaluation using a probabilistic framework, assuming a stationary
latent utility model.


Assumption A3 lists assumptions on objectives, and is also made in prior works investigating
composite optimization and preferential evaluation using human feedback (Christiano et al., 2017;
Lin et al., 2022; Astudillo & Frazier, 2019), which consider human feedback over explicitly available
‘observables’ generated from a system. This also mirrors real-world preference elicitation settings,
where user judgments are anchored to clearly defined objective attributes (Shao et al., 2023).


Following the hierarchical distinction of _f_, we meet the multifaceted ethical evaluation requirement
using the design of a hierarchical surrogate model, learnt using data queried by our proposed
acquisition strategy, in a Variational Bayesian Experimental Design loop. Furthermore, we mitigate
the scalability constraints with human evaluation using LLM as proxy evaluators, for a fixed ethical
criterion. We now discuss the specifics of our methodology.


4 SEED-SET


Our approach consists of three main modeling components, a hierarchical VGP for surrogate modeling,
a data acquisition strategy to generate test cases, combined with an LLM as a proxy evaluator for
pairwise preferential evaluation. Figure 1 provides an overview of our approach.


4.1 A VARIATIONAL BAYESIAN FRAMEWORK FOR SCALABLE ETHICAL MODELING


Our three main components interact with each other with inherent stochasticity from system observations and uncertainty from limited user evaluations. To account for these considerations in a
sample-efficient evaluation setting, we adopt a variational Bayesian framework. Specifically, we learn
a surrogate model for _f_ using _f_ ( _x_ ) _∼_ _p_ ( _f_ ( _x_ ) _|D_ ) by applying a joint distribution over its behavior
at each sample _x_ _∈X_ . The prior distribution of the objective _p_ ( _f_ ( _x_ )) is combined with the likelihood function _p_ ( _D|f_ ( _x_ )) to compute the posterior distribution _p_ ( _f_ ( _x_ ) _|D_ ) _∝_ _p_ ( _D|f_ ( _x_ )) _p_ ( _f_ ( _x_ )),
which represents the updated beliefs about _f_ ( _x_ ). We approximate the posterior _p_ ( _f_ ( _x_ ) _|D_ ) using a
variational distribution parameterized by _ϕ_, as _qϕ_ ( _f_ ( _x_ )), for sample-efficient posterior estimation.


In this work, we use GPs to estimate the posterior distribution, due to their analytical compatibility
with evaluation under limited data. In GP models, the distribution is a joint normal distribution
_p_ ( _f_ ( _x_ ) _|D_ ) = _N_ ( _µ_ ( _x_ ) _, k_ ( _x, x_ _[′]_ )) completely specified by its mean _µ_ ( _x_ ) and kernel function _k_ ( _x, x_ _[′]_ ),
where _µ_ ( _x_ ) represents the prediction and _k_ ( _x, x_ _[′]_ ) the associated uncertainty. The computational
complexity of GP models scales with _O_ ( _n_ [3] ) as the number of observations _n_ increases. To ensure
scalability with the number of observations, we generalize our variational posterior models _qϕ_
to Variational GPs (VGPs) (Tran et al., 2015), that reduce the computational burden of inference
through sparse approximation of the posterior distribution. A detailed discussion on the computational
efficiency of VGPs is provided in Appendix A.3.


4


In this way, the complexity of inference is reduced from _O_ ( _n_ [3] ) to _O_ ( _nm_ [2] ), which significantly
improves the efficiency if _m ≪_ _n_ .

**Hierarchical VGP (HVGP) for Modular Modelling.** Ethical evaluation in autonomous systems is
inherently hierarchical: system designs give rise to observable behaviors measured by _f_ obj, which in
turn elicit subjective ethical evaluations _f_ subj from user. To capture this structure in a scalable and
interpretable way, we decompose the ethical evaluation task into two distinct modeling stages, each
represented by a VGP:


 - _Objective GP_, which models the mapping _f_ obj using a surrogate _g_ : _x_ _y_, where _y_ R _[d]_ are
_→_ _∈_
objective metrics, intermediate quantities that reflect system behaviors relevant to ethical concerns
(e.g., cost of decision-making, resilience, equity in resource distribution across stakeholders, etc.).

 - _Subjective GP_, which models the mapping _f_ subj using a surrogate _h_ : _y_ _z_, where _z_ R denotes
_→_ _∈_
a latent utility score representing stakeholder judgments (e.g., perceived fairness or acceptability),
obtained from qualitative evaluations.


Subjective ethical evaluation lacks ground-truth values, i.e., we do not have direct access to label
_z_ . This makes supervised training infeasible. We therefore adopt a common practice for grounding
qualitative information involves preference elicitation using pairwise evaluation from an oracle
_T_ : ( _y, y_ _[′]_ ) _→{_ 1 _,_ 2 _}_ (Huang et al., 2025), which are objectives from scenarios ( _x, x_ _[′]_ ). Here, the
oracle takes a pair of evaluations _y, y_ _[′]_ _∈Y_, and returns a binary label “1” or “2”, indicating its
preferred design (“1” if _y_ _≻_ _y_ _[′]_ and vice-versa).

This hierarchical structure offers two critical advantages: 1) _Interpretability_ : Ethical preferences
are grounded in observable system outcomes _y_, not the latent design parameters _x_ . Modeling _h_ ( _y_ )
instead of _h_ ( _x_ ) aligns with how stakeholders assess ethical outcomes in terms of behaviors they can
perceive and evaluate. 2) _Data efficiency_ : By incorporating the subjective criteria’s dependency on a
combination of task-specific and task-agnostic objectives, we promote accurate modeling choices.
For accurate evaluation in limited evaluations, sample efficiency and quality of evaluation is highly
sensitive to modeling choices (Keswani et al., 2024).


Efficient data querying is critical under limited budgets. Naive random sampling wastes evaluations
and often misses key test cases. Moreover, objectives both shape and are shaped by subjective
evaluations, making separate model training ineffective. Instead, one must target regions of objective
space aligned with subjective criteria. We address this through adaptive data acquisition within a
Bayesian Experimental Design (BED) framework.


4.2 A BAYESIAN EXPERIMENTAL DESIGN LOOP FOR ADAPTIVE AND EFFICIENT TESTING


Given a history of experiments := ( _x_ 1 _, y_ 1) _, . . .,_ ( _xn, yn_ ), BED seeks to maximize the _Expected_
_D_ _{_ _}_
_Information Gain_ (EIG) that a potential experimental outcome can provide about the model parameter
of interest, denoted as _θ_ . This is measured as the expected reduction in entropy _H_ ( _·_ ) of the posterior
distribution of _θ_ :


EIG( _x_ ) = _H_ [ _p_ ( _θ_ )] E _p_ ( _y|x,D_ )[ _H_ [ _p_ ( _θ_ ( _x, y_ ))]] = _I_ ( _θ_ ; ( _x, y_ ) ) _,_ (1)
_|D_ _−_ _|D ∪_ _|D_

where, _I_ denotes the mutual information between _θ_ and ( _x, y_ ).


We generalize this paradigm into our HVGP models and propose a nested approach that unifies the
exploration and exploitation of both the objective factors and subjective preferences simultaneously.
We consider two variational distributions _qϕ_ ( ) and _qψ_ ( ), of the Objective and Subjective GP,
parameterized by _ϕ, ψ_ respectively. We propose _· V_ : _X_ _→_ _·_ R:

_V_ ( _x_ ) = _I_ ( _gx_ ; _y_ ) + E _qϕ_ ( _y|x_ )[ _I_ ( _hy_ ; _z_ ) + E _qψ_ ( _hy_ )[ _hy_ ]] _,_ (2)
_|D_ _|D_

such that two jointly evaluated candidates _x_ = [ _x_ 1 _, x_ 2] can be obtained by maximizing V. Here _gx, hy_
denote _g_ ( _x_ ) and _h_ ( _y_ ) respectively. Maximizing the first two terms maximizes mutual information in
scenario and objective spaces, while the third enforces preferential alignment with proposed criteria.


Equation (2) reflects the inherent hierarchical structure necessary for accurately modeling ethical
preferences. The first term captures the expected information gain about the **objective layer**, ensuring
that we reduce uncertainty in the objectives. The second term quantifies information gain in the
**subjective layer**, directly improving our estimate of the latent utility function _h_ ( _y_ ). The final term


5


Figure 2: Environments for the two case studies considered in this work. **(Left)** Power Grid Allocation

- IEEE 5-Bus and 30-Bus ( Section 5.1). **(Right)** Fire Rescue ( Section 5.2). Additional case study
for Optimal Routing in Appendix E.


encourages sampling in regions where the current model predicts higher ethical utility, enabling the
method to balance exploration with targeted exploitation. Balanced exploration–exploitation requires
all three. In Section 5, we study acquisition ablations, and in Section 5.1, stakeholder ablations
on the learned objective space. In Appendix I we demonstrate the advantage of joint acquisition in
sample efficiency and discovery of scenario with high alignment.


4.3 AN LLM-BASED PROXY FOR SUBJECTIVE ETHICAL EVALUATION


The pairwise elicitation oracle is modeled using humans for evaluation. However, using human experts
can lead to constraints on the number of pairwise evaluations that can be performed. Additionally,
getting true experts who are not biased to provide the subjective evaluation can be challenging,
especially for understudied domains, and is not cost-effective. To reduce the dependency on user
evaluation, we leverage LLMs as proxies to make evaluations on behalf of stakeholders according
to certain stakeholder-specified criteria encoded through prompt design (Huang et al., 2025). Our
proposed prompt design accommodates the hierarchical structure proposed so far.


**Prompt** **Design:** The prompt has three main parts: 1) _Task_ _description_ : Specifies task-relevant
contextual details, 2) _Objective metrics_ ( _y_ 1 _, y_ 2): For two scenarios ( _x_ 1 _, x_ 2), corresponding objective
metrics ( _y_ 1 _, y_ 2) for chosen objectives are provided for comparison, 3) _Subjective criteria_ : An NLP
description of preference over the objective landscape, that encodes criteria for selecting the preferred
candidate. These details along with response instruction are used to extract a binary preference “1” if
_y_ 1 _≻_ _y_ 2 from the LLM. We conduct ablation studies on model and temperature in Appendix F that
validates the reliability of LLMs as proxy evaluators.


5 EXPERIMENTS


Our central hypothesis is that SEED-SET enables scalable, accurate, and data-efficient ethical evaluation of autonomous systems. Using previously discussed ethical evaluation concerns (Battistuzzi
et al., 2021; Luo et al., 2024; Bieler et al., 2024) to guide the design of our observables, we test our
hypothesis on several case studies across three main applications. In addition, we conduct several
ablations to better understand our methodology. In all plots, solid lines are the mean _µ_ and shaded
areas are one standard deviation ( _µ ± σ_ ). We run for five seeds per experiment, using GPT-4o for all
LLM queries. Additional details are provided in Appendix A.5.


**Benchmarks.** We propose three case studies, exploring real world applications using Power Grid
Resource allocation, Fire Rescue (as illustrated in Figure 2), and Optimal Routing case study
(Appendix E). We also explore multi stakeholder evaluation in Appendix G and applicability of our
method on real human data in Appendix H. To the best of our knowledge, there are no standard
simulation platforms/benchmarks to test domain-agnostic ethical benchmarking techniques for lowbudget experimental validation. We discuss more about each case study in the coming sections.


**Baselines.** We test our SEED-SET framework against the following relevant baselines. First, the
Random sampling baseline samples uniformly in the design parameter space. Single GP (Keswani


6


5-Bus


7


4


1


30-Bus


9


4


0


0 25 50 75 100
Iterations


0 25 50 75 100
Iterations


HVGP Single GP Random VS-AL-1 VS-AL-2


Figure 3: **Power Grid Allocation Preference Scores.** Preference scores baseline comparison for
5-Bus (left) and 30-Bus (right).


et al., 2024) is a pairwise preferential GP that directly maps design parameters consisting of a pair
of scenarios ( _x, x_ _[′]_ ) to ethical evaluations _z_ . VS-AL-1 and VS-AL-2 are Version Space Active
Learning baselines, referenced in (Keswani et al., 2024) that use a Support Vector Machine (SVM)
to learn a preferential decision boundary for pairs of scenarios ( _x, x_ _[′]_ ), with a linear kernel and
RBF kernel respectively. Finally, we also compare against BOPE, i.e., Bayesian Optimization
with Preference Exploration (Lin et al., 2022) in Appendix I, which decomposes the two stages of
preference exploration and experimentation.


**Metrics.** Since we optimize over the preferences of LLM-proxy stakeholders, we do not actually
have access to the ground truth preference function. Instead, for each baseline and case study,
we handcraft a deterministic _preference_ _score_ function _h_ : _y_ _→_ _z_ representing the ground truth
preference over objectives given the observables. The function is designed to be proportional to
_f_ subj. To validate the correctness of the preference score function, we utilize TrueSkill Bayesian skill
ranking (Herbrich et al., 2006) to predict scores for each evaluation point. We also use other task
specific metrics for evaluating the quality of generated scenarios, such as measuring distribution
overlap with real data (Appendix H), estimating coverage in feature space to measure the degree of
exploration (Section 5.2), and qualitative visualizations (Section 5.1).


5.1 POWER GRID RESOURCE ALLOCATION


We first study the ethical impact of using different Distributed Energy Resource (DER) deployment
strategies with varying reactive power limits on the Power Grid Allocation IEEE 5-bus and IEEE
30-bus networks (Luo et al., 2024), denoted 5-Bus and 30-Bus for convenience.


**Scenario Description.** binary vector indicating if a certain location has DER deployment, andA scenario is parameterized by _x_ := [ _l, r_ ] _r∈X∈_, whereR [20] + [specifies the reactive] _l_ _∈{_ 0 _,_ 1 _}_ [20] is a
power limits. This is a challenging problem for ethical evaluation due to multifaceted ethical concerns
in distribution of resource arising from various stakeholders (Luo et al., 2024; Bieler et al., 2024).


**Observables.** Given scenario _x_, the resulting observables vector _y_ _∈_ R [4] has four components.
The voltage _Fairness_ ( _y_ [1] ) measures the uniformity of voltage distribution across all buses. The total
_Cost_ ( _y_ [2] ) combines the expenses of installing DER units and reactive power provision. The _Priority_
( _y_ [3] ) area coverage measures how well each design serves under-served or rural buses. Finally, the
_Resilience_ ( _y_ [4] ) assesses the network’s ability to maintain voltages above a specified threshold. We
provide more details, including formulas, in Appendix B.0.3.


**Evaluation Method.** In our prompt, we ask the LLM to prioritize Priority, followed by Cost, and
ignore all other dimensions (prompt example in Appendix B.0.7). Since we do not have access to
ground truth evaluation scores, we approximate the LLM’s preference function with a preference
score function _h_ [˜] ( _y_ ) := [0 _, −_ 0 _._ 5 _,_ 1 _,_ 0] _y_ .


7


1.0


0.5


0.0


24


0

0 35
Cost


0


0 35
Cost


1.0


0.5


0.0


Figure 4: **Bus-30 Different Stakeholder Groups.** We show that our learned preference GP is able to
adapt to the needs of different potential stakeholder groups. The plots show data point for optimum
preference score (shown in red) projected on objective space, with contours of predicted preference
score for Stakeholder A (left) and B (right), with optimum value data point.


**Results** We evaluate on 5-Bus and 30-Bus and observe that our proposed HVGP achieves a
higher preference score than all other baselines. Although Single GP can do better than Random
on 5-Bus, it cannot solve 30-Bus since the design parameter dimensions grows to 40, making
it hard to efficiently explore the space. On the contrary, HVGP can mitigate this through the
hierarchical structure, which reduces the complexity of mapping from objectives to subjective
assessment, followed by our acquisition strategy, which prioritizes targeted exploration of objective
space through the second mutual information term (MI2).


Both VS-AL-1 and VS-AL-2 cannot solve either tasks, which we hypothesize is because of the
inaccurate modeling choice in VS-AL-1 due to learning a linear decision boundary. Similar reasoning
as Single GP can be used to explain the inefficiency of VS-AL-2 due to its direct modeling of a
complex decision space.


5.2 FIRE RESCUE


Next, we consider an autonomous drone navigation scenario for fire extinguishing in a semi-urban
setting, as motivated by discussions on ethical concerns in rescue robotics (Battistuzzi et al., 2021).


**Scenario** **description.** A Fire Rescue scenario is parameterized by _x_ _∈_ [0 _,_ 1] [30], with only 9
dimensions relevant to scenario design, controlling the placement of assets such as a museum, a
gas station, a food court, and residential blocks with tree covers of variable density. The remaining
dimensions are uncorrelated to the objectives. In Figure 9 of Appendix C), we give examples of
three scenario visualizations. The goal of the autonomous system is to search the area for fire and,
based on its uncertainty, decide whether to continue exploring or spray retardant.


**Observables.** Given scenario _x_, the observables vector _y_ = [ _y_ [1] _, y_ [2] _, y_ [3] ] _∈_ R [3] quantifies the
cumulative potential _Chemical Damage_ caused by deciding to spraying the retardant ( _y_ [1] ), cumulative
potential _Fire Damage_ caused by fire due to deciding not to spray the retardant ( _y_ [2] ), and _Spread_
_factor_ ( _y_ [3] ), measuring risk of firespread due to proximity of assets.


**Evaluation Method.** In our prompt, we ask the LLM to prioritize high Chemical damage and high
Spread factor (prompt example in Appendix C). We use preference score function _h_ [˜] ( _y_ ) := [1 _,_ 0 _,_ 1] _y_
and coverage score function ˜ _c_ ( **x** ) that estimates cumulative standard deviation for **x** = [ _x_ 1 _, . . ., xn_ ]
for _n_ collected data points to measure the coverage of search space to sample novel scenarios.


**Results.** We observe that HVGP achieves higher preference score than all baselines, with a similar
explanation as provided in the results of Section 5.1. We also observe that our acquisition strategy
achieves higher preference score than two HVGP variants: MI1+MI2, and Pref. MI1+MI2 does
not consider the preference term in the acquisition function, and Pref does not consider the two
mutual information terms. The discrepancy in the preference scores is due to MI1+MI2’s inefficient


8


6


5


4


3


|Col1|Col2|
|---|---|
|||
|||
|||
|||


Iterations


6


5


4


3


Iterations

HVGP MI1+MI2 Pref


HVGP
Single GP


VS-AL-1
VS-AL-2


Random


Figure 5: **Fire Rescue Preference Scores.** We report preferences scores for baseline comparisons
(left) and acquisition strategy ablations (right).


|Col1|Col2|
|---|---|
|||
|||
|||
|||


Iterations


|Col1|Col2|Col3|
|---|---|---|
||||
||||
||||
||||


Iterations

HVGP MI1+MI2 Pref


HVGP
Single GP


VS-AL-1
VS-AL-2


Random


Figure 6: **Fire Rescue Coverage.** We report coverage scores for baseline comparisons (left) and
acquisition strategy ablations (right).


exploration in higher dimensions. While Pref performs better than MI1+MI2, the complete acquisition
strategy performs the best with the additional improvement from targeted exploration. Our method
also shows higher coverage scores than baselines and ablations. We also provide an example of
scenario generation using the learned model ( Appendix C.0.4).


5.3 ADDITIONAL RESULTS


To better understanding how SEED-SET works, we perform several ablation studies in a questionand-answer format.


**(Q1) What conditions enable SEED-SET to perform well?** We observe that our method performs
best when the design parameter space is large. In lower-dimensional settings such as 5-Bus, Single
GP performs well but is still suboptimal compared to our method ( Figure 3 (left)). However,
in higher-dimensional cases such as 30-Bus and Fire Rescue, HVGP outperforms Single GP
by exploring test cases that highly align with the subjective criteria. We also observe superior
performance of our method compared to ablations of BOPE in Appendix I, which shows that in
limited sample settings, joint learning of objective and subjective models improves sample efficiency.


5-Bus


30-Bus


**well approximate the LLM’s preference function?** It 35

Bayesian skill rating system (Herbrich et al., 2006) to

0 47 95 0 47 95

evaluate their individual skill ratings (evaluation details in HVGP IterationsSingle GP Random VS-AL-1Iterations VS-AL-2
Appendix B.0.2). This can be a more accurate evaluation
method than our proposed preference scores since it does Figure 7: **Power Grid Allocation Skill**
not assume the LLM’s optimization objective form. How- **Ratings.** Skill ratings computed using
ever, this evaluation process can be extremely expensive a Bayesian skill rating algorithm (Herand the skill ratings are not directly comparable across brich et al., 2006).
seeds and baselines. We observe that for both 5-Bus and 30-Bus, the trends roughly match the trends
seen from the heuristic preference scores in Figure 3.


35


25


15


45


15


30


0 47 95
Iterations


0 47 95
Iterations


HVGP Single GP Random VS-AL-1 VS-AL-2


Figure 7: **Power Grid Allocation Skill**
**Ratings.** Skill ratings computed using
a Bayesian skill rating algorithm (Herbrich et al., 2006).


9


**(Q3)** **How** **does** **acquisition** **strategy** **support** **sample** **efficiency?** In the acquisition ablation
studies, we observe that our complete acquisition strategy consistently performs well. In Fire Rescue
(right of Figure 5), while preferential optimization is crucial, mutual information enables efficient
exploration, which is essential in a high-dimensional search space with low volume optima, leading to
incremental improvement over Pref. This validates our idea of balancing exploration and exploitation
with the acquisition strategy.


**(Q4) How well does our model adapt to different stakeholders?** In power resource allocation,
we consider an ablation with two stakeholders. Stakeholder A and B care mainly about high priority,
and low cost respectively. Figure 4 shows the optimum value scenario for Stakeholder A has a high
priority score, and high cost, whereas for Stakeholder B, the optimum corresponds to low cost and
low priority. This shows that the sampling procedure accurately accounts for the stakeholder-specific
criteria, resulting in different explorations, and stakeholder-specific test cases. We also extend this
analysis to a more challenging, multi-stakeholder setting for the Power Grid case study (details
in Appendix G), Figure 16 and observe similar trends.


**(Q5) How robust is our approach to LLM specifications?** We conduct ablation studies for the
Fire Rescue case study for LLM parameters, varying temperature, prompt and model (Appendix F),
and observe robustness to evaluator perturbations. We attribute this to pairwise elicitation, that
eliminates uncertainty from self-inconsistency (Bouwer et al., 2024; Hoeijmakers et al., 2024).


**(Q6) How suitable is our framework for real-world ethical alignment?** In Appendix H, we apply
our framework to extract latent objectives influencing travel mode choices using the TravelMode
dataset (Greene, 2003). This case study shows that our framework can be effectively applied to learn
underlying trends in objective landscape (Figure 17).


6 LIMITATIONS


While the SEED-SET framework effectively mitigates common challenges associated with ethical
assessment, certain limitations remain, along with promising future directions.


**Scalability for Extremely Large Datasets.** Using sparse variational GPs reduces complexity from
_O_ ( _N_ [3] ) to _O_ ( _NM_ [2] ) with _M_ inducing points, enabling SEED-SET to handle tens of thousands of
observations. Scaling to hundreds of thousands or more remains challenging, which future work
could address via stochastic variational inference (SVI).


The current model uses a stationary kernel (e.g., RBF), assuming covariance depends only on relative
distance. This can be too restrictive for systems with varying regimes. To relax this, SEED-SET can
be extended with non-stationary kernels (e.g., spectral mixture, input-warped) or deep GPs that warp
inputs through neural layers. Our model requires complete knowledge of objective metrics a priori.
While this is a reasonable assumption that mimics real-world preference elicitation, in the lack of
complete list of objectives, the testing process can have inaccuracies.


Using LLMs as ethical proxies also risks sensitivity to prompts and context, so ongoing alignment
checks or fine-tuning are needed to keep their judgments in sync with human values. Still, when no
ground truth exists, preference data is a practical surrogate, and VGPs with Bayesian design offer a
sample-efficient solution that would be far cheaper than training an LLM from scratch on preferences.


7 CONCLUSION


We presented SEED-SET, a scalable framework for ethical evaluation of autonomous systems that
combines objective system metrics with subjective stakeholder judgments through a hierarchical
variational Bayesian model. By separating measurable factors from user preferences and guiding
exploration via a principled acquisition strategy, SEED-SET enables efficient and interpretable
evaluation of ethical trade-offs. The integration of large language models as proxy evaluators further
reduces human burden while maintaining value alignment. Experiments across domains demonstrate
SEED-SET’s effectiveness, with future work aimed at extending to multi-agent settings and real-time
applications.


10


8 ETHICS STATEMENT


In this work, we propose an automated evaluation tool for ethical assessment of autonomous systems.
Our work assumes a provided ethical criteria for evaluation, and associated cost functions. We make
no recommendations or comments on the correctness of any ethical criteria in this work, and mainly
leverage existing instances of ethical evaluation for validation of our pipeline. The paper does not
involve crowdsourcing or research with human subjects.


9 REPRODUCIBILITY STATEMENT


All simulations for Fire rescue, Power resource grid allocation and Optimal Routing were conducted
on a Linux workstation with Ubuntu 22.04 LTS equipped with an Intel 13th Gen Core i7-13700KF
CPU (16 cores, 24 threads, up to 5.4 GHz) and an NVIDIA GeForce RTX 4090 GPU (24 GB VRAM).
The Bayesian Experimental Design (BED) loops were implemented using wrapper BoTorch (Balandat
et al., 2020) and GPyTorch (Gardner et al., 2018) libraries. The compute requirements were consistent
with standard usage of these libraries and did not require additional specialized hardware beyond
what was used for Webots simulation. Implementation specific details of both the simulations are
provided in Appendix B and Appendix C. Examples of LLM prompts used in the generation of results
reported in the paper are also provided in the Appendix.


10 ACKNOWLEDGEMENT


This research was developed with funding from the Defense Advanced Research Projects Agency
(DARPA) with Distribution Statement "A" (Approved for Public Release, Distribution Unlimited) .
The views, opinions and/or findings expressed are those of the author and should not be interpreted
as representing the official views or policies of DARPA or the U.S. Government.


We thank Jonathan Becker, Ashley Wiren, and Mitul Saha for their valuable discussions and support
throughout the project.


11


REFERENCES


Trueskill: the video game rating system. [https://trueskill.org/.](https://trueskill.org/)


Iso/pas-8800 road vehicles — safety and artificial intelligence, 2024.


Dario Amodei, Chris Olah, Jacob Steinhardt, Paul Christiano, John Schulman, and Dan Mané.
Concrete problems in ai safety. _arXiv preprint arXiv:1606.06565_, 2016.


Raul Astudillo and Peter Frazier. Bayesian optimization of composite functions. In _International_
_Conference on Machine Learning_, pp. 354–363. PMLR, 2019.


Maximilian Balandat, Brian Karrer, Daniel Jiang, Samuel Daulton, Ben Letham, Andrew G Wilson, and Eytan Bakshy. Botorch: A framework for efficient monte-carlo bayesian optimization.
_Advances in neural information processing systems_, 33:21524–21538, 2020.


Linda Battistuzzi, Carmine Tommaso Recchiuto, and Antonio Sgorbissa. Ethical concerns in rescue
robotics: a scoping review. _Ethics and Information Technology_, 23(4):863–875, 2021.


Sanjukta Bhattacharya, Anastasios Oulis Rousis, and Aikaterini Bourazeri. Trust & fair resource
allocation in community energy systems. _IEEE Access_, 12:11157–11169, 2024.


Stephanie Bieler, Cara Goldenberg, Avery McEvoy, Katerina Stephan, and Alex Walmsley.
Aggregated distributed energy resources in 2024: The fundamentals. Technical report, RMI, under
contract with NARUC, July 2024. [URL https://connectedcommunities.lbl.gov/](https://connectedcommunities.lbl.gov/sites/default/files/2024-07/NARUC_ADER_Fundamentals_Interactive.pdf)
[sites/default/files/2024-07/NARUC_ADER_Fundamentals_Interactive.](https://connectedcommunities.lbl.gov/sites/default/files/2024-07/NARUC_ADER_Fundamentals_Interactive.pdf)
[pdf.](https://connectedcommunities.lbl.gov/sites/default/files/2024-07/NARUC_ADER_Fundamentals_Interactive.pdf) Prepared as part of the NARUC–NASEO DER Integration and Compensation Initiative,
DOE Award No. DE-OE0000925.


Abeba Birhane, Sepehr Dehdashtian, Vinay Prabhu, and Vishnu Boddeti. The dark side of dataset
scaling: Evaluating racial classification in multimodal models. In _Proceedings of the 2024 ACM_
_Conference on Fairness, Accountability, and Transparency_, pp. 1229–1244, 2024.


Renske Bouwer, Marije Lesterhuis, Fien De Smedt, Hilde Van Keer, and Sven De Maeyer. Comparative approaches to the assessment of writing: Reliability and validity of benchmark rating and
comparative judgement. _Journal of Writing Research_, 15(3):498–518, 2024.


Kathryn Chaloner and Isabella Verdinelli. Bayesian experimental design: A review. _Statistical_
_science_, pp. 273–304, 1995.


Gregory Chance, Dhaminda B Abeywickrama, Beckett LeClair, Owen Kerr, and Kerstin Eder.
Assessing trustworthiness of autonomous systems. _arXiv preprint arXiv:2305.03411_, 2023.


Hareesh Chitikena, Filippo Sanfilippo, and Shugen Ma. Robotics in search and rescue (sar) operations:
An ethical and design perspective framework for response phase. _Applied Sciences_, 13(3):1800,
2023.


Paul Francis Christiano, Jan Leike, Tom B. Brown, Miljan Martic, Shane Legg, and Dario Amodei.
Deep reinforcement learning from human preferences. _ArXiv_, abs/1706.03741, 2017. URL
[https://api.semanticscholar.org/CorpusID:4787508.](https://api.semanticscholar.org/CorpusID:4787508)


Wei Chu and Zoubin Ghahramani. Preference learning with Gaussian processes. In _Proceedings of_
_the 22nd international conference on Machine learning_, pp. 137–144, 2005.


Shiyang Cong, Destenie Nock, Yuchen L. Qiu, and Bo Xing. Unveiling hidden energy poverty
using the energy equity gap. _Nature Communications_, 13(1):2456, 2022. [doi:10.1038/s41467-022-](https://doi.org/10.1038/s41467-022-30146-5)
[30146-5.](https://doi.org/10.1038/s41467-022-30146-5)


Louise Dennis, Michael Fisher, Marija Slavkovik, and Matt Webster. Formal verification of ethical
choices in autonomous systems. _Robotics and Autonomous Systems_, 77:1–14, 2016.


Andrés Domínguez Hernández and Vassilis Galanos. A toolkit of dilemmas: Beyond debiasing and
fairness formulas for responsible ai/ml. _arXiv preprint arXiv:2303.01930_, 2023.


12


Ahmed Fahmin, Muhammad Aamir Cheema, Mohammed Eunus Ali, Adel Nadjaran Toosi, Hua Lu,
Huan Li, David Taniar, Hesham A. Rakha, and Bojie Shen. Eco-friendly route planning algorithms:
Taxonomies, literature review and future directions. _ACM Computing Surveys_, 57(1):1–42, 2024.


Xin Gao, Tian Luan, Xueyuan Li, Qi Liu, Zhaoyang Ma, Xiaoqiang Meng, and Zirui Li. Ethical
alignment decision-making for connected autonomous vehicle in traffic dilemmas via reinforcement
learning from human feedback. _IEEE Internet of Things Journal_, 2024.


Jacob Gardner, Geoff Pleiss, Kilian Q Weinberger, David Bindel, and Andrew G Wilson. Gpytorch:
Blackbox matrix-matrix gaussian process inference with gpu acceleration. _Advances in neural_
_information processing systems_, 31, 2018.


Declan Grabb, Max Lamparth, and Nina Vasan. Risks from language models for automated mental
healthcare: Ethics and structure for implementation. _arXiv preprint arXiv:2406.11852_, 2024.


William H. Greene. _Econometric Analysis_ . Prentice Hall, Upper Saddle River, NJ, 5 edition, 2003.


Thilo Hagendorff. The ethics of ai ethics: An evaluation of guidelines. _Minds and machines_, 30(1):
99–120, 2020.


Ralf Herbrich, Tom Minka, and Thore Graepel. Trueskill™: a bayesian skill rating system. In
_Proceedings_ _of_ _the_ _20th_ _International_ _Conference_ _on_ _Neural_ _Information_ _Processing_ _Systems_,
NIPS’06, pp. 569–576, Cambridge, MA, USA, 2006. MIT Press.


Eva JI Hoeijmakers, Bibi Martens, Babs MF Hendriks, Casper Mihl, Razvan L Miclea, Walter H
Backes, Joachim E Wildberger, Frank M Zijta, Hester A Gietema, Patricia J Nelemans, et al. How
subjective ct image quality assessment becomes surprisingly reliable: pairwise comparisons instead
of likert scale. _European Radiology_, 34(7):4494–4503, 2024.


David Huang, Francisco Marmolejo-Cossío, Edwin Lock, and David Parkes. Accelerated preference
elicitation with llm-based proxies, 2025. [URL https://arxiv.org/abs/2501.14625.](https://arxiv.org/abs/2501.14625)


IEEE Global Initiative. Ethically aligned design: A vision for prioritizing human wellbeing with autonomous and intelligent systems. [https://sagroups.ieee.org/](https://sagroups.ieee.org/global-initiative/wp-content/uploads/sites/542/2023/01/ead1e.pdf)
[global-initiative/wp-content/uploads/sites/542/2023/01/ead1e.pdf,](https://sagroups.ieee.org/global-initiative/wp-content/uploads/sites/542/2023/01/ead1e.pdf)
2019.


Liwei Jiang, Jena D. Hwang, Chandra Bhagavatula, Ronan Le Bras, Jenny Liang, Jesse Dodge,
Keisuke Sakaguchi, Maxwell Forbes, Jon Borchardt, Saadia Gabriel, Yulia Tsvetkov, Oren Etzioni,
Maarten Sap, Regina Rini, and Yejin Choi. Can machines learn morality? the delphi experiment.
_arXiv preprint arXiv:2110.07574_, 2021.


Anna Jobin, Marcello Ienca, and Effy Vayena. The global landscape of ai ethics guidelines. _Nature_
_machine intelligence_, 1(9):389–399, 2019.


Vijay Keswani, Vincent Conitzer, Hoda Heidari, Jana Schaich Borg, and Walter Sinnott-Armstrong.
On the pros and cons of active learning for moral preference elicitation. In _Proceedings of the_
_AAAI/ACM Conference on AI, Ethics, and Society_, volume 7, pp. 711–723, 2024.


Zhiyuan Jerry Lin, Raul Astudillo, Peter Frazier, and Eytan Bakshy. Preference exploration for
efficient bayesian optimization with multiple outcomes. In _International Conference on Artificial_
_Intelligence and Statistics_, pp. 4235–4258. PMLR, 2022.


Yinhong Liu, Han Zhou, Zhijiang Guo, Ehsan Shareghi, Ivan Vuli´c, Anna Korhonen, and Nigel
Collier. Aligning with human judgement: The role of pairwise preference in large language model
evaluators. _arXiv preprint arXiv:2403.16950_, 2024.


Dongwen Luo, Jiachen Zhong, Yiting Wang, and Weihao Pan. Ethical considerations in smart grid
optimization promoting energy equity and fairness. 2024.


Karel Martens. _Transport justice:_ _Designing fair transportation systems_ . Routledge, 2016.


Nestor Maslej, Loredana Fattorini, Raymond Perrault, Yolanda Gil, Vanessa Parli, Njenga Kariuki,
Emily Capstick, Anka Reuel, Erik Brynjolfsson, John Etchemendy, et al. Artificial intelligence
index report 2025. _arXiv preprint arXiv:2504.07139_, 2025.


13


Ninareh Mehrabi, Fred Morstatter, Nripsuta Saxena, Kristina Lerman, and Aram Galstyan. A survey
on bias and fairness in machine learning. _arXiv preprint arXiv:1908.09635_, 2021.


Olivier Michel. Cyberbotics ltd. webots™: professional mobile robot simulation. _International_
_Journal of Advanced Robotic Systems_, 1(1):5, 2004.


Brent Daniel Mittelstadt, Patrick Allo, Mariarosaria Taddeo, Sandra Wachter, and Luciano Floridi.
The ethics of algorithms: Mapping the debate. _Big_ _Data_ _&_ _Society_, 3(2):2053951716679679,
2016.


Ritesh Noothigattu, Djallel Bouneffouf, Nicholas Mattei, Rachita Chandra, Piyush Madan, Kush R
Varshney, Murray Campbell, Moninder Singh, and Francesca Rossi. Teaching ai agents ethical
values using reinforcement learning and policy orchestration. _IBM_ _Journal_ _of_ _Research_ _and_
_Development_, 63(4/5):2–1, 2019.


Organisation for Economic Co-operation and Development. Oecd principles on artificial intelligence. [https://www.oecd.org/en/topics/sub-issues/ai-principles.](https://www.oecd.org/en/topics/sub-issues/ai-principles.html)
[html, 2019.](https://www.oecd.org/en/topics/sub-issues/ai-principles.html)


Przemysław Pałka. Ai, consumers & psychological harm. _AI_ _and_ _Consumers,’Larry_ _DiMatteo,_
_Cristina Poncibò, Martin Hogg, Geraint Howells (Eds.), Cambridge University Press (2023/2024)_,
2023.


Rafael HM Pereira, Tim Schwanen, and David Banister. Distributive justice and equity in transportation. _Transport reviews_, 37(2):170–191, 2017.


Tom Rainforth, Adam Foster, Desi R Ivanova, and Freddie Bickford Smith. Modern bayesian
experimental design. _Statistical Science_, 39(1):100–114, 2024.


Anka Reuel, Ben Bucknall, Stephen Casper, Tim Fist, Lisa Soder, Onni Aarne, Lewis Hammond,
Lujain Ibrahim, Alan Chan, Peter Wills, et al. Open problems in technical ai governance. _arXiv_
_preprint arXiv:2407.14981_, 2024.


Ann-Katrin Reuel, Mark Koren, Anthony Corso, and Mykel J Kochenderfer. Using adaptive stress
testing to identify paths to ethical dilemmas in autonomous systems. In _SafeAI@ AAAI_, 2022.


Olawale Salaudeen, Anka Reuel, Ahmed Ahmed, Suhana Bedi, Zachary Robertson, Sudharsan
Sundar, Ben Domingue, Angelina Wang, and Sanmi Koyejo. Measurement to meaning: A
validity-centered framework for ai evaluation. _arXiv preprint arXiv:2505.10573_, 2025.


Murat Sensoy, Lance M. Kaplan, Simon Julier, Maryam Saleki, and Federico Cerutti. Risk-aware
classification via uncertainty quantification. _arXiv preprint arXiv:2412.03391_, 2024.


Han Shao, Lee Cohen, Avrim Blum, Yishay Mansour, Aadirupa Saha, and Matthew Walter. Eliciting
user preferences for personalized multi-objective decision making through comparative feedback.
_Advances in Neural Information Processing Systems_, 36:12192–12221, 2023.


Benjamin K Sovacool, Raphael J Heffron, Darren McCauley, and Andreas Goldthau. Energy decisions
reframed as justice and ethical concerns. _Nature Energy_, 1(5):1–6, 2016.


[Elham Tabassi. Artificial intelligence risk management framework (ai rmf 1.0). https://tsapps.](https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=936225)
[nist.gov/publication/get_pdf.cfm?pub_id=936225, January 2023.](https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=936225) National In[stitute of Standards and Technology, Gaithersburg, MD. doi:10.6028/NIST.AI.100-1.](https://doi.org/10.6028/NIST.AI.100-1)


Christian Tarsney, Teruji Thomas, and William MacAskill. Moral decision-making under uncertainty.
2024.


Michalis Titsias. Variational learning of inducing variables in sparse Gaussian processes. In _Artificial_
_intelligence and statistics_, pp. 567–574, 2009.


Dustin Tran, Rajesh Ranganath, and David M Blei. The variational gaussian process. _arXiv preprint_
_arXiv:1511.06499_, 2015.


14


Hanna Wallach, Meera Desai, A Feder Cooper, Angelina Wang, Chad Atalla, Solon Barocas, Su Lin
Blodgett, Alexandra Chouldechova, Emily Corvi, P Alex Dow, et al. Position: Evaluating
generative ai systems is a social science measurement challenge. _arXiv preprint arXiv:2502.00561_,
2025.


Angelina Wang, Xuechunzi Bai, Solon Barocas, and Su Lin Blodgett. Measuring machine learning
harms from stereotypes requires understanding who is harmed by which errors in what ways. In
_Proceedings_ _of_ _the_ _2025_ _ACM_ _Conference_ _on_ _Fairness,_ _Accountability,_ _and_ _Transparency_, pp.
746–762, 2025.


Laura Weidinger, John Mellor, Maribeth Rauh, Conor Griffin, Jonathan Uesato, Po-Sen Huang, Myra
Cheng, Mia Glaese, Borja Balle, Atoosa Kasirzadeh, et al. Ethical and social risks of harm from
language models. _arXiv preprint arXiv:2112.04359_, 2021.


Laura Weidinger, Maribeth Rauh, Nahema Marchal, Arianna Manzini, Lisa Anne Hendricks, Juan
Mateos-Garcia, Stevie Bergman, Jackie Kay, Conor Griffin, Ben Bariach, et al. Sociotechnical
safety evaluation of generative ai systems. _arXiv preprint arXiv:2310.06682_, 2023.


A. F. T. Winfield, S. Booth, L. A. Dennis, T. Egawa, H. Hastie, N. Jacobs, R. I. Muttram, J. I.
Olszewska, F. Rajabiyazdi, A. Theodorou, M. A. Underwood, R. H. Wortham, and E. Watson. Ieee
p7001: A proposed standard on transparency. _Frontiers in robotics and AI_, 8:665729, 2021.


Peipei Xu, Wenjie Ruan, and Xiaowei Huang. Towards the quantification of safety risks in deep
neural networks. _arXiv preprint arXiv:2009.06114_, 2020.


Muhammad Shahrul Zaim bin Ahmad and Kazuhiro Takemoto. Large-scale moral machine experiment on large language models. _PloS one_, 20(5):e0322776, 2025.


Yi Zeng, Yu Yang, Andy Zhou, Jeffrey Ziwei Tan, Yuheng Tu, Yifan Mai, Kevin Klyman, Minzhou
Pan, Ruoxi Jia, Dawn Song, et al. Air-bench 2024: A safety benchmark based on risk categories
from regulations and policies. _arXiv preprint arXiv:2407.17436_, 2024.


Daniel Ziegler, Long Ouyang, Jeffrey Wu, Ryan Lowe, Nisan Stiennon, John Gray, John Schulman,
and Paul Christiano. Self-rewarding language models. _arXiv preprint arXiv:2210.01152_, 2022.


15


A LITERATURE REVIEW


A.1 ADDITIONAL RELATED WORKS


**Governance Approaches, for Responsible AI.** A wide range of governance frameworks, guidelines
and standards have been proposed to guide the ethical development and deployment of AI systems
(Tabassi, 2023; Organisation for Economic Co-operation and Development, 2019; IEEE Global
Initiative, 2019). For example, NIST’s AI Risk Management Framework (AI RMF 1.0, 2023),
IEEE’s P7001 on transparency levels (Winfield et al., 2021) or ISO PAS 8800 on ethical design (ISO,
2024)). Hagendorff’s meta-analysis (Hagendorff, 2020) highlights the vagueness and redundancy in
many such documents. Recent reports and position papers also emphasize broader societal impacts,
including the AI Index 2025 report (Maslej et al., 2025), new perspectives on ML harms and domainspecific risks (Wang et al., 2025; Grabb et al., 2024; Salaudeen et al., 2025; Reuel et al., 2024;
Wallach et al., 2025; Pałka, 2023). These works highlight that evaluating AI systems is not only a
technical task but also a broader measurement challenge.


Recent reports and position papers also emphasize broader societal impacts, including the AI Index
2025 report (Maslej et al., 2025), new perspectives on ML harms and domain-specific risks (Wang
et al., 2025; Grabb et al., 2024), and ongoing debates on measurement validity, governance challenges,
and consumer harms (Salaudeen et al., 2025; Reuel et al., 2024; Wallach et al., 2025; Pałka, 2023).
These works highlight that evaluating AI systems is not only a technical task but also a broader
measurement challenge.


**ML-Based Ethical Evaluation.** Complementary to governance, researchers have explored ML-based
methods for quantifying ethical properties of AI behavior. Fairness metrics such as demographic
parity and equalized odds are widely studied (Mehrabi et al., 2021), though their applicability varies
across contexts. Risk estimation techniques, including uncertainty-aware classification models, have
been used to quantify prediction confidence and manage potential harm in safety-critical domains
(Xu et al., 2020; Sensoy et al., 2024). Recent work has also highlighted the need for frameworks
that support situated ethical reasoning, emphasizing context, trade-offs, and reflexivity in the design
of responsible AI/ML systems (Domínguez Hernández & Galanos, 2023), while frameworks like
Weidinger et al.’s sociotechnical safety evaluation introduce layered approaches that include systemic
impacts (Weidinger et al., 2023). More recently, LLMs have been used as ethical evaluators: some
systems learn to score the moral acceptability of generated content (Jiang et al., 2021), while others
self-evaluate outputs against behavioral objectives (Ziegler et al., 2022). This motivates the need for
practical, system-level methods that can evaluate ethical behavior empirically and at scale. While
frameworks and metrics exist in isolation, few approaches integrate them into a unified methodology
suitable for continuous testing or deployment, which is addressed by our work.


**Automated** **tools.** Prior technical approaches include reinforcement learning and orchestration
for instilling ethical values (Noothigattu et al., 2019), large-scale moral judgment studies on
LLMs (Zaim bin Ahmad & Takemoto, 2025), and active learning for preference elicitation (Keswani
et al., 2024). With the exception of active learning, these techniques impose large-scale data and simulation budget requirements. Our baseline models are adopted from active learning based techniques.
In addition, ethical concerns have been explored in domain applications such as smart grids (Luo
et al., 2024) and search-and-rescue robotics (Battistuzzi et al., 2021), which we leverage for the case
studies considered in our work.


**Bayesian Optimization (BO)/Bayesian Experimental Design (BED) for sample efficient eval-**
**uation.** Works such as Lin et al. (2022); Astudillo & Frazier (2019) have explored hierarchical
decision making in sample efficient settings within BO/BED paradigm, which we also leverage
as the mathematical foundation of our technique. While Lin et al. (2022) addresses joint objective–subjective evaluation, its performance depends heavily on careful tuning of design parameters.
Our key technical contribution is joint learning of objective metrics and subjective preferences, paired
with an acquisition function that integrates both sources of information. This yields significantly
improved sample efficiency, making the approach practical for real-world evaluation.


16


A.2 VARIATIONAL BAYESIAN METHODS:OVERVIEW


In variational inference, the posterior distribution over a set of unobserved variables **u** =
_u_ 1 _,_ _, un_ given some data is approximated by a so-called variational distribution _q_ ( **u** ):
_{_ _· · ·_ _}_ _D_
_p_ ( **u** _|D_ ) _∼_ _q_ ( **u** ).

Variational Bayesian methods are a family of techniques for efficient posterior approximation in
Bayesian inference. In variational inference, the posterior distribution over a set of unobserved
variables **u** = _u_ 1 _,_ _, un_ given some data is approximated by a so-called variational distribution
_{_ _· · ·_ _}_ _D_
_q_ ( **u** ): _p_ ( **u** _|D_ ) _∼_ _q_ ( **u** ). The distribution _q_ ( **u** ) is restricted to belong to a family of distributions of
simpler form than _p_ ( **u** _|D_ ) (e.g. a family of Gaussian distributions), selected with the intention to
minimize the Kullback-Leibler (KL) divergence between the approximated variational distribution
_q_ ( **u** ) and the exact posterior _p_ ( **u** _|D_ ). This is equivalent to maximizing the evidence lower bound
(ELBO) (Titsias, 2009):


ELBO( _q_ ( **u** )) = E _q_ ( **u** )[log _p_ ( **u** )] _D_ KL[ _q_ ( **u** ) _p_ ( **u** )] _,_
_D|_ _−_ _||_


which can be considered as a sum of the expected log-likelihood of the data and the KL divergence
between the variational distribution and the prior _p_ ( **u** ) (Titsias, 2009).


A.3 VGPS


In GP models, the distribution is a joint normal distribution _p_ ( _f_ ( _x_ ) _|D_ ) = _N_ ( _µ_ ( _x_ ) _, k_ ( _x, x_ _[′]_ )) completely specified by its mean _µ_ ( _x_ ) and kernel function _k_ ( _x, x_ _[′]_ ) with corresponding hyper-parameters
_θ_, where _µ_ ( _x_ ) represents the prediction and _k_ ( _x, x_ _[′]_ ) the associated uncertainty. The computational
complexity of GP models scales with _O_ ( _n_ [3] ) as the number of observations _n_ increases. Sparse
Variational GP (SVGP) reduces the computational burden of inference through sparse approximations
of the posterior distributions by introducing auxiliary latent variables **u** and **Z**, where the _induc-_
_ing variables_ **u** = [ _u_ ( _z_ 1) _,_ _, u_ ( _zm_ )] [⊺] R _[m]_ are the latent function values corresponding to the
_inducing input locations_ contained in the matrix _· · ·_ _∈_ **Z** = [ _z_ 1 _,_ _, zm_ ] [⊺] R _[m][×][d]_ .
_· · ·_ _∈_

Typically, the _variational distribution qϕ_ ( **u** ) is parameterized as a Gaussian with variational mean
**mu** and covariance **Su** . Assuming that the latent function values _f_ ( _x_ ) _, f_ ( _x_ _[′]_ ) are conditionally
independent given **u** and _x, x_ _[′]_ _/_ _z_ 1 _, . . ., zm_, the GP posterior can be cheaply approximated as
_∈{_ _}_

                   _p_ ( _f_ ( _x_ ) ) _qϕ_ ( _f_ ( _x_ )) = _p_ ( _f_ ( _x_ ) **u** ) _qϕ_ ( **u** ) _d_ **u**
_|D_ _≈_ _|_

= ( _µϕ_ ( _x_ ) _, σϕ_ ( _x, x_ _[′]_ )) _,_
_N_

where


_µϕ_ ( _x_ ) = _ψ_ **u** [⊺][(] _[x]_ [)] **[m][u]** _[,]_

_σϕ_ ( _x, x_ _[′]_ ) = _kθ_ ( _x, x_ _[′]_ ) _ψ_ **u** [⊺][(] _[x]_ [)(] **[K][uu]**
_−_ _[−]_ **[S][u]** [)] _[ψ]_ **[u]** [(] _[x][′]_ [)] _[,]_

_ψ_ **u** ( _x_ ) = **K** _[−]_ **uu** [1] _[k][θ]_ [(] **[Z]** _[, x]_ [)] _[.]_

In this way, the complexity of inference is reduced from _O_ ( _n_ [3] ) to _O_ ( _nm_ [2] ), which significantly
improves the efficiency if _m ≪_ _n_ .


A.4 PAIRWISE BAYESIAN OPTIMIZATION


Following (Chu & Ghahramani, 2005), we assume that the responses are distributed according to a
_probit_ likelihood used in the construction of _V_ ( _x_ ) as in equation 2:


_L_ ( _z_ ( _y_ 1 _, y_ 2) = 1 _g_ ( _y_ 1) _, g_ ( _y_ 2)) = Φ( _[g]_ [(] _[y]_ [1][)] _[ −]_ _[g]_ [(] _[y]_ [2][)] ) _,_
_|_ ~~_√_~~ 2 _λ_


where _λ_ is a hyper-parameter that can be estimated along with the other hyper-parameters of the
model, and Φ is the standard normal CDF. We extend this concept to LLM as evaluators, with the
same assumptions on probit likelihood modeling.


17


A.5 BASELINES AND METRICS


We ablate our problem formulation to the Single GP baseline, which is also popularly used in literature
for pairwise preferential elicitation.


Version space active learning (VS-AL) has been adopted from Keswani et al. (2024), and corresponds
to learning an accurate decision boundary for a pair of scenarios as inputs. VS-AL-1 samples next
data-point based on margin maximization from decision boundary using a Support Vector Machine.
VS-AL-2 requires an explicit utility function to be supplied, and performs a weighted explorationoptimization on this utility function controlled by a hyperparameter _λ_ . In our experiments, we set
the utility function to be the same as preference score. We see that _λ_ = 0 gives fast and efficient
optimization. The results reported in main paper correspond to _λ_ = 0 _._ 3.


We note that VS-AL does not perform well in our tasks due to the fact that it is designed for _prediction_
_accuracy_, whereas our method is geared towards online test case generation. This also shows the
crucial role of modeling choices in limited data settings, where VS-AL-1 fails due to linear decision
boundary assumptions, which do not capture the complex landscape. The sensitivity of VS-AL-2 to
noise also renders it unsuitable for noisy explorations such as those we focus on.


B DISTRIBUTED ENERGY RESOURCE ALLOCATION IN POWER GRIDS


We evaluate our distributed energy resource allocation problem on a real-world decision-making
task modeled after an Optimal Power Flow (OPF) scenario in power systems. The objective is to
identify deployment strategies for Distributed Energy Resources (DERs) that align with implicit
ethical preferences across multiple performance dimensions.


B.0.1 SYSTEM SETUP


The testbed is the standard IEEE 5/30-bus network, a widely used benchmark in power system studies.
We consider a variety of DER placement and sizing configurations, each representing a distinct design
candidate. For each configuration, an AC OPF is solved using the pandapower library to compute
physical network states under steady-state conditions. All experiments were conducted using the
same computational resources described in Appendix C.0.2.


B.0.2 RANKING FOR BAYESIAN INFERENCE


We evaluate the alignment of our proposed method’s queried candidates with the LLM evaluator
using the TrueSkill Bayesian rating system (Herbrich et al., 2006) and implemented using tru. The
intuition is that as training progresses, the proposed candidates should achieve higher alignment with
the LLM’s preferences. Thus, later candidates should inherently achieve a higher preference rating
than candidates queried earlier during training.


Informally, let := _x_ 1 _, . . ., xN_ R _[d]_ be the sequence of queried candidate points over the course
of a single training session. _X_ _{_ Define the latent utility function _} ⊂_ _u_ : R _[d]_ _→_ R for the LLM evaluator:
Since candidates should optimize on the LLM’s preference over time, we expect an approximately
monotonic improvement in alignment


_u_ ( _xi_ ) _u_ ( _xi_ +1) _i < N_ (3)
_≤_ _∀_


Consider each latent utility as a random variable


_u_ ( _xi_ ) _∼N_ ( _µi, σi_ [2][)] (4)


We perform Bayesian skill rating inference Herbrich et al. (2006) to obtain the posterior distributions
over each _µi_, and sorting them to obtain the final rankings.


In practice, for each baseline and seed, we first downsample on the total number of candidate points,
down to _N_ points. We define each of the _N_ points as a player in a free-for-all game (as mentioned
in tru). Then, we take _Comb_ ( _N,_ 2) combinations of 1-versus-1 games to approximate the skill rating


18


(or _µi_ ) of each player, which is what we plot in Figure 7. Each 1-versus-1 game is resolved by
treating the game as a pairwise comparison standard to our SEED-SET methodology, where we have
an LLM-proxy select which player has higher preference alignment with the LLM. This choice then
corresponds to the winning player in the game.


B.0.3 PERFORMANCE METRICS


Each design is evaluated using four ethically motivated metrics:


    - Voltage Fairness: Measures the variance in bus voltages across the network; lower variance
implies more equitable voltage delivery.


    - Total Cost: Combines capital expenditures for DER installation and operational costs related
to reactive power support.


    - Priority Area Coverage: Quantifies the share of power delivered to high-priority buses, such
as rural or underserved regions.


    - Resilience: Assesses the percentage of time that all bus voltages remain within safe operating
limits under perturbations (e.g., load uncertainty or line outages).


B.0.4 PREFERENCE MODELING


Instead of assuming explicit utility weights on the objectives, we simulate human-in-the-loop or
policy-driven decision-making via pairwise preference queries. That is, for selected pairs of outcomes
( _y_ 1 _, y_ 2), a preference function indicates which design is ethically preferred. These preferences are
generated based on a latent utility function, not revealed to the optimizer, that reflects nonlinear
trade-offs among the four objectives.


B.0.5 OPTIMIZATION TASK


The goal is to efficiently identify high-utility DER configurations by querying preferences, without
direct access to the utility values. This falls under composite Bayesian optimization with preference
exploration, where the acquisition function balances exploration of uncertain regions in ethics space
with exploitation of inferred preferences.


B.0.6 QUERY STRATEGY


An initial set of 10 pairwise preferences is randomly sampled to initialize the model. Each step of the
optimization selects new pairs to query, guided by the used acquisition strategy.


B.0.7 EVALUATION CRITERIA


**Criterion.** Priority primary, Cost secondary (with threshold): Your evaluation should consider both
Priority and Cost, with Priority given greater importance. Specifically, you should first compare the
scenarios based on their Priority scores. If the difference in Priority between the two scenarios is
within a small threshold of 0.5 (indicating that their Priority performance is very close), then you
should prefer the scenario with the lower Cost, even if its Priority is slightly lower. The objectives
Fairness and Resilience are not considered in this evaluation. Below, we provide the detailed prompt
design for the Power Grid Resource Allocation experiment, as shown in Figure 8. The other prompts
can be created by modifying various user-specified criteria.


C FIRE RESCUE


We used an open-source simulator Webots simulator Michel (2004) for scenario generation and
drone navigation using a PID controller. We choose to incorporate Chemical Damage as an observable
in response to the ethical criteria of not adding additional risk in rescue robotics tasks suggested in
(Battistuzzi et al., 2021).


19


***Task:** Your task is to compare and evaluate two optimal power flow (OPF)
scenarios in a standard IEEE 30-bus network. Each scenario involves installing
Distributed Energy Resources (DERs) at specified buses. The performance of each
scenario is quantified by four key objectives: Voltage Fairness, Total Cost,
Priority Area Coverage, and Resilience. Given two scenarios and a user-defined
criterion specifying which objectives matter to the user and their relative
importance, your task is to determine which scenario is preferable according to
the criterion.


***Scenario** **parameters:** Each scenario is quantified by an outcome vector consisting
of four dimensions, representing distinct performance measures:
Dimension 1: Fairness (higher is better)  - Measures the uniformity of the voltage
across buses, with values closer to 1 indicating more uniform voltages.
Dimension 2: Cost (lower is better)  - Represents the total combined cost
associated with DER installations and reactive power provision.
Dimension 3: Priority (higher is better)  - Indicates how well the scenario
prioritizes DER installations at underserved or rural buses.
Dimension 4: Resilience (higher is better)  - Assesses the network’s capability
to maintain bus voltages above the specified threshold, with values closer to
1 indicating higher resilience.
You will be provided two outcome vectors|one for each scenario|each containing
these four parameters and formatted as follows:
Outcome 1: _{_ [round(num, 5) for num in outcome1.tolist()] _}_
Outcome 2: _{_ [round(num, 5) for num in outcome2.tolist()] _}_

***Criteria** **(Important** **considerations** **for** **evaluation):** Priority primary, Cost
secondary (with threshold): Your evaluation should consider both Priority and
Cost, with Priority given greater importance. Specifically, you should first
compare the scenarios based on their Priority scores. If the difference in
Priority between the two scenarios is within a small threshold of 0.5 (indicating
that their Priority performance is very close), then you should prefer the
scenario with the lower Cost, even if its Priority is slightly lower. The
objectives Fairness and Resilience are not considered in this evaluation.


***Response** **instructions:** After carefully evaluating each scenario/outcome
according to the criteria provided above, clearly indicate your decision using
one of the following numerical responses: -Respond ‘1’ if Outcome 1 is preferred.
-Respond ‘2’ if Outcome 2 is preferred.


***Answer** **format:** First, clearly state your numerical choice (1 or 2). Then, in
the next paragraph, provide a detailed justification of your choice. Explicitly
refer to the provided user-defined criteria and clearly discuss the numerical
differences between the two scenarios. Your explanation must directly connect to
the numerical outcome vectors of each scenario and show clear reasoning aligned
with the specified criteria.


Figure 8: Example prompt for Power Grid Resource Allocation experiment.


C.0.1 FIRE RESCUE SIMULATION DETAILS


Different types of buildings and their spatial locations in this scenario are defined using a scenario
parameterare scalars _x_ denoting= [ _d_ 1 _, d_ the2 _, b, g, m, g_ tree density, _x, gy, m_ such [1] _x_ _[, m]_ that [1] _y_ _[, m]_ higher [2] _x_ _[, m]_ [2] _y_ _[, m]_ value [3] _x_ _[, m]_ denotes [3] _y_ []] _[∈X]_ higher [,] [where] risk _[d]_ [1] _[, d]_ of [2] fire _[∈]_ [[0] spread. _[,]_ [ 100]]
_b ∈{_ 0 _,_ 1 _,_ 2 _,_ 3 _}_ governs the number and placement of food courts in the scenario, and _g, m ∈{_ 0 _,_ 1 _}_


20


|  <br> |Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|Col11|Col12|Col13|Col14|Col15|Col16|Col17|Col18|Col19|Col20|Col21|Col22|Col23|Col24|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|<br>
<br><br>||||||||||||||||||||||||
|<br><br><br> <br><br><br><br><br>re 9: Scenario<br>ils in Appendix|||||||~~~~|~~~~||||||||||||||||
|<br><br><br> <br><br><br><br><br>re 9: Scenario<br>ils in Appendix|||||||~~~~|~~~~||||||||||||||||
|<br><br><br> <br><br><br><br><br>re 9: Scenario<br>ils in Appendix||||||||<br><br>||||||||||||||||
|<br><br><br> <br><br><br><br><br>re 9: Scenario<br>ils in Appendix||||||||<br><br>||||||||<br>||||||||
|<br><br><br> <br><br><br><br><br>re 9: Scenario<br>ils in Appendix||||||||||||||||||||||||
|<br><br><br> <br><br><br><br><br>re 9: Scenario<br>ils in Appendix||||||||||||||||||||||||
|<br><br><br> <br><br><br><br><br>re 9: Scenario<br>ils in Appendix||||||||||||||||<br><br>~~~~||||||||
|<br><br><br> <br><br><br><br><br>re 9: Scenario<br>ils in Appendix||||||||<br>||||||||||||||||
|<br><br><br> <br><br><br><br><br>re 9: Scenario<br>ils in Appendix||<br><br>|<br><br>|<br><br>|<br><br>|<br><br>|<br><br>|<br><br>|<br><br>|<br><br>|<br><br>|<br><br>|<br><br>|<br><br>|<br><br>|<br><br>|<br><br>|<br>|<br>|<br>|<br>|<br>|<br>|
|<br><br><br> <br><br><br><br><br>re 9: Scenario<br>ils in Appendix||<br><br>|<br><br>|<br><br>|<br><br>|<br><br>|<br><br>|<br><br>|<br><br>|<br><br>|<br><br>|<br><br>|<br><br>|end of d|end of d|iscover|iscover|ed asset|ed asset|<br>  s shown|<br>  s shown|<br>  s shown|<br>  s shown|
|<br><br><br> <br><br><br><br><br>re 9: Scenario<br>ils in Appendix||cenario<br> ppendix|cenario<br> ppendix|visuali<br>  C.|visuali<br>  C.|zation fo|zation fo|r three|r three|r three|r three|r three|r three|r three|r three|r three|r three|r three|r three|r three|r three|r three|r three|


are binary variables denoting presence of a Gas station and Museum in a scenario, and _gx, gy_ controls
position of gas station in the scene, and _m_ _[i]_ _x_ _[, m][i]_ _y_ [controls position of] _[ i]_ [th][ manor.]

Figure 9 shows three generated scenarios sampled from _X_ . As we can see, first image shows larger
density of trees discovered. Also note that the museum is not discovered in any of the scenarios,
since it is excluded from the field of view of the circular trajectory of the robot at all times. The
trajectory is kept constant across all experiments. The shaded colored region corresponds to the part
of the building that has been discovered by the drone and is used to estimate confidence of perception,
which is utilized in the decision-making of whether the building must be further investigated or a
retardant must be sprayed on the building.


Given a scenario _x_, the simulation rollouts are used to generate a decision _di_ = [ _w, s_ ] for each
building in scenario, where _w_ = 1 denotes decision to spray the building _i_ with spray strength _s_,
and _w_ = 0 denotes decision to further explore the area on accounts of uncertainty of discovery.
The cumulative information for all buildings is used to generate a three dimensional observable
_y_ = [ _y_ [1] _, y_ [2] _, y_ [3] ] for each scenario _x_, where _y_ [1] quantifies cumulative damage caused by chemical
retardant sprayed on the buildings to extinguish fire, and _y_ [2] quantifies cumulative damage caused by
fire due to the decision to not spray the retardant, and _y_ [3] quantifies spread factor. Spread factor is
calculated as _y_ [3] = 1 _/distance_, where _distance_ pertains to euclidean distance between all assets,
therefore, close by assets have high spread factor.


The damage also depends on the type of asset. Gas station poses higher risk of damage due to fire
and therefore accumulates higher damage value.


C.0.2 EXPERIMENTAL SETUP DETAILS


A singular simulation was conducted in Webots simulation platform to obtain the trajectory of
the robot, which was kept constant across different scenario definitions. We developed a custom
simulator for scenario generation by specification of various assets, perception mapping of the robot,
and the corresponding cumulative damage from fire and chemical retardant estimations. Code for the
simulation will be released upon request.


All simulations for Fire rescue simulation were conducted on a Linux workstation with Ubuntu 22.04
LTS equipped with an Intel 13th Gen Core i7-13700KF CPU (16 cores, 24 threads, up to 5.4 GHz)
and an NVIDIA GeForce RTX 4090 GPU (24 GB VRAM). The system ran CARLA simulations
using CUDA 12.2 and NVIDIA driver version 535.230.02. The Bayesian Experimental Design
(BED) loops were implemented using the BoTorch Balandat et al. (2020) and GPyTorch Gardner et al.
(2018) libraries. The compute requirements were consistent with standard usage of these libraries
and did not require additional specialized hardware beyond what was used for Webots simulation.


21


***Task:** Your task is to compare and evaluate two instances of decision making
in a wildfire rescue scenario. Each scenario involves a set of assets on fire,
and a drone first manuevers around the area to estimate the locations of each
asset, and decides whether to continue exploring a certain asset more or spray
a retardant on it. The result of the agent’s decision making policy gives the
location and type of assets to explore further, and location, type and spray
strength of assets on which a chemical retardant will be sprayed. Retardant
strength is used to denote the intensity of the chemical used, where a higher
intensity of retardant is more toxic to the environment. Given two scenarios
and a user-defined criterion specifying which objectives matter to the user
and their relative importance, your task is to determine which scenario is
preferable according to the criterion.


***Scenario** **parameters:** Each scenario is quantified by a 3 dimensional outcome
vector, where first entry of the vector corresponds to damage caused by
toxicity of the retardant, and second corresponds to damage caused by fire left
untreated, and third corresponds to risk of spread.


Scenario-1:
Damage caused by toxicity of spraying of retardant for scenario 1 (DT-1):
_{_ outcome1[0] _}_ .
Damage caused by fire left untreated for scenario 1 (DF-1): outcome1[1].
Spread factor for scenario 1: outcome1[2].
Scenario-2:
Damage caused by toxicity of spraying of retardant for scenario 2 (DT-2):
_{_ outcome2[0] _}_ .
Damage caused by fire left untreated for scenario 2 (DF-2):
Spread factor for scenario 2: outcome2[2]outcome2[2] _}_ .


Your goal is to assess scenario difficulty, a more diffcult scenario is more
preferred in this context, and a more difficult scenario corresponds to one that
involves more damage.


***Criteria** **(Important** **considerations** **for** **evaluation):** Scenario with high spread
factor and high DT are preferred.


***Response** **instructions:** After evaluating each scenario/outcome according to
only the criteria provided above, clearly indicate your decision using one of
the following numerical responses: -Respond ‘1’ if Scenario-1 is preferred.
-Respond ‘2’ if Scenario-2 is preferred. Do not create your own rules, stick to
the above provided criteria.


***Answer** **format:** Respond with a single integer: ‘1’ if Scenario 1 is preferred,
‘2’ if Scenario 2 is preferred. Do not provide explanation.


Figure 10: Example prompt for Fire Rescue experiment.


C.0.3 LLM PROMPTS AND EVALUATION CRITERIA


We provide the detailed prompt design for the Fire Rescue experiment, as shown in Figure 10.
Alternative prompts can be generated by changing different criteria specified by the user.


22


Figure 11: GUI of webots with the generated buildings using our scenario generation simulator.
Also shown is the perception mapping and discovery of assets as the drone maneuvers around the
environment. The trajectory is generated by navigation of drone in a 2D circular trajectory using a
PID controller.


C.0.4 EXAMPLE OF SCENARIO GENERATION USING OUR PIPELINE


The assets for constructing scenario (Food court, Museum, etc.) are chosen from pre-available assets
in Webots. Our scenario generation mechanism outputs the location of assets using the _x_ generated
by the acquisition strategy which populates the Webots simulation. A DJI Mavic Pro drone model is
used for navigation around the scenario and discovery of assets. Figure 11 shows an example of the
scenario generated using our custom scenario generation pipeline.


D THE USE OF LARGE LANGUAGE MODELS


In our research framework, our paper relies on Large Language Models (LLMs) as proxies for humans
in performing system-level ethical testing.


However, we do not use LLMs for any writing, other than to occasionally check for spelling and
grammar issues, and recommend how to format figures.


E OPTIMAL ROUTING


We consider the ethical assessment of optimal routing algorithm in an urban setting with pedestrians
and schools, with periodic fluctuations.


**Scenario description.** The routing algorithm takes the pedestrian traffic into account and proposes
most optimal route for a given origin _u_ _∈_ R [2] and destination location _d_ _∈_ R [2] . This is used to
parameterize the scenario as _x_ = [ _u, v_ ] on a map with some regions that have high density of
pedestrian traffic, and designated school areas.


**Observables.** Motivated by discussions on ethical concerns in travel planning (Battistuzzi et al.,
2021), we consider two main observables _y_ = [ _y_ [1] _, y_ [2] ], namely, _Cost_ ( _y_ [1] ) and _Length of route_ ( _y_ [2] ).
_Cost_ refers to the weighted sum of nodes, where nodes closer to the pedestrian and school traffic are
assigned higher weights to encourage the route planning algorithm to take routes further away from
regions of pedestrian and school traffic.


**Evaluation** **Method.** We ask LLM to prioritize scenarios with high cost and length, with an
objective of generating interesting test cases that stress test the planning algorithm. We use a
preference score function _h_ [˜] ( _y_ ) := [1 _,_ 1] _y_ to evaluate the quality of scenarios generated.


23


20

15

10


5

0


Iterations

HVGP Single GP Random sampling


Figure 12: **Optimal Routing Preference Scores.** Comparison of our method (HVGP) against Single
GP, Random baselines.


6.5

6.0

5.5

5.0

4.5

4.0

3.5


|Col1|Col2|
|---|---|
|||
|||
|||
|||
||0.0<br>0.2<br>|
||~~0.4~~<br>0.7<br>|
||~~1.0~~|


Iterations


Figure 13: LLM temperature ablations for Fire Rescue


**Results.** We compare against Single GP, Random baselines for five random seeds, and observe
that our method outperforms both (Figure 12). Specifically, we can clearly observe Single GP
converging to a local optimum. This shows that wherever we have access to well-defined objectives,
hierarchical deconstruction outperforms end-to-end learning.


F LLM ABLATIONS


We evaluate robustness of our framework to LLM as an evaluator with three main ablations: (1)
temperature, (2) prompt, and (3) model. For temperature and prompt ablations, we fixed GPT 4-o
model. We perform these evaluations on Fire Rescue study, and report preference score as a metric,
with mean and standard deviation measured across five seeds.


**Temperature.** We measured across five values of temperature: 0 _,_ 0 _._ 2 _,_ 0 _._ 4 _,_ 0 _._ 7 _,_ 1 _._ 0. The preference
scores for Fire Rescue case study are shown in Figure 13.


**Model.** We measured preference score for three LLM models: GPT 4-o (default), GPT o3, and
GPT o3-mini. The preference scores for Fire Rescue case study are shown in Figure 14.


F.1 PROMPT ABLATIONS


We performed experiments across three sets of prompts, including the original prompt reported in
main paper. The two additional prompts were designed by querying ChatGPT 5.1 to add vaguness
to the task description (Prompt B) and criteria (Prompt C) respectively. The modified prompts are
shown below. Figure 15 shows the preference score for prompt ablations.


24


6.0


5.5


5.0


4.5


4.0


3.5


|Col1|Col2|
|---|---|
|||
|||
|||
|||
||gpt o3<br>gpt 4-o<br>|
||~~gpt o3-mini~~|


Iterations


Figure 14: LLM model ablations for Fire Rescue


**Analysis.** We report the mean and standard deviation across all ranges of ablations for the last ten
iterations as a quantitative measure of performance:


- Nominal case: (temperature _t_ = 0, Prompt A, GPT 4-o): 5 _._ 23 _±_ 0 _._ 56


- Temperature ablation (Prompt A, GPT 4-o): 5 _._ 4 _±_ 0 _._ 58


- Prompt ablation (temperature _t_ = 0, GPT 4-o): 5 _._ 26 _±_ 0 _._ 57


- Model ablation: 5 _._ 37 _±_ 0 _._ 56


We observe that for all ablations, the mean and standard deviation of preference score is comparable.
This shows that our pipeline assures robustness to perturbations in evaluation. We attribute this to the
inherently probabilistic nature of our preference modeling, and pairwise elicitation as a method for
preference querying (Bouwer et al., 2024).


25


6.0


5.5


5.0


4.5


4.0


3.5


Iterations


Figure 15: Preference scores for prompt ablations on the Fire Rescue case study


Iterations


Cost


25

20

15

10

5


20

15

10

5

0


Iterations


20.0

17.5

15.0

12.5

10.0

7.5

5.0

2.5

0.0


Figure 16: **Multi-stakeholder Criteria ablation** : Cost and Priority for each criteria discussed in
Section G, last 40 samples generated by our framework plotted for each criteria against corresponding
mean and standard deviation. These plots show the effect of changing the objective landscape for one
stakeholder, as both cost and priority are inter-dependent.


G MULTI-STAKEHOLDER EVALUATION


Our method can be used for generating scenarios corresponding to complex preferences that cannot be
captured using handcrafted preference scores. This often emerges in multi-stakeholder setting, where
different stakeholders have different interests (i.e., different preferences over objectives). We perform
a multi-stakeholder evaluation for Power resource allocation (Case-30) problem to validate this.
Specifically, we consider four stakeholders: _Policymakers, Utility Operators, Community Advocate_,
and _Market Operators_ .


**Evaluation.** The LLM is asked to prioritize scenarios that best lead to a compromise between
the various stakeholders, balancing their values fairly. Objectives relevant to each stakeholder are
provided in the form of two criteria shown below:


**Criteria-1** : _Policymaker_ : fairness, priority
_Utility operator_ : resilience, cost
_Community advocate_ : fairness, priority
_Market operator_ : cost


**Criteria-2** : _Policymaker_ : fairness, priority
_Utility operator_ : resilience
_Community advocate_ : fairness, priority
_Market operator_ : cost


The key difference between both criteria is the lack of cost as an objective for the utility operator,
which was enforced to test the corresponding change in the scenarios generated for each criteria.
Figure 16 and compare the two objectives, cost and priority for each criteria.


26


**Results.** We observe that the cost and priority values for Criteria-1 converge to much lower values
compared to Criteria-2. This is due to the fact that Criteria-1 automatically prioritizes cost more
heavily than Criteria-2, due to its presence in the Utility operator’s stated preference. This drives
the optimization of cost to much lower values in first case. Additionally, cost and priority are not
independent due their mutual dependence on scenario, and involve a trade-off. This leads to low
priority scores in first case. In the second case, the framework optimizes for priority, leading to higher
cost values too, which incur less penalty than in Criteria-1.


H TRAVEL MODE CASE STUDY


We validate the applicability of our method on real human data using the Travelmode dataset Greene
(2003). The dataset consists of real human feedback on preferred modes of travel, over between
Sydney and Melbourne over a set of objectives: travel time, wait time (time spent at the terminal or
station), mode specific vehicle cost (vcost), general cost combining vcost and time (gcost), household
income, and party size Greene (2003). These objectives are often used to quantify ethical concerns in
route and travel planning in urban settings Pereira et al. (2017); Martens (2016).

**Scenario and observables.** simulator parameterized by _x_ We adopt the six objectives as observables _∈_ R [5] denoting party size, weather, income, _y_ _∈_ purposeR [6], and construct aof travel and
holiday season respectively. These represent a combination of independent variables from the
objectives and additional variables affecting travel-planning.


The ranges of scenario parameters _x_ and observables _y_ are chosen based on ranges retrieved from the
original dataset.


**Evaluation method.** We present the objectives pertaining to 12 randomly chosen individuals as
in-context learning, and the LLM is asked to report which of the two user profiles are likely to prefer
air travel as a mode, based on presented objectives.


**Results.** We report mean and standard deviation of objectives corresponding to scenarios marked
as ‘preferred’ by the LLM, and compare against mean and standard deviation for the data available as
shown in Table 1, and normalized visualization in Figure 17.


**Variable** **Real Mean** **Sample Mean** **Real Std** **Sample Std**
travel 124.83 115.32 49.85 19.55
wait 46.53 86.91 24.18 12.58
vcost 97.57 135.24 31.46 55.92
gcost 113.55 152.69 32.91 57.52
income 41.72 63.27 18.95 23.81
size 1.57 1.53 0.81 0.87


Table 1: Comparison of Real vs. LLM Means and Standard Deviations for Air-related Observations.


We observe from the mean estimates that the sampled data aligns most in travel time, income and
party size, and displays some variation in costs, and significant variation in wait time compared to the
real distribution. This can be attributed to simulator parameters, which generate much larger wait
times more often than observed in real data. Similarly, the income is biased towards higher values
due to the simulator not being adjusted for skewness in real income reports.


I EXTENDED BASELINE COMPARISON FOR POWER GRID RESOURCE
ALLOCATION


In this section we discuss extended baseline comparison with BOPE based approaches adopted
from Lin et al. (2022), that considers a composite problem setting, and decouples the two stages
of _preference_ _exploration_ and _experimentation_ . The original approach is open-ended, with a lot
of problem-specific flexibility on how the various design stages can be chosen. We consider four
ablations of their technique, and compare against our joint acquisition approach with HVGP. Figure 18


27


1.0

0.8

0.6

0.4

0.2

0.0


|Col1|Col2|Col3|zed)<br>ed)|Col5|Col6|Col7|Col8|
|---|---|---|---|---|---|---|---|
|||Real (Normali<br>LLM (Normaliz|zed)<br> ed)|zed)<br> ed)|zed)<br> ed)|zed)<br> ed)|zed)<br> ed)|
|||||||||


Figure 17: Comparison of distributions for real and data generated by our method (sampled) using
mean _±_ std. deviation.

7

6

5

4

3

2

1

|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|
|---|---|---|---|---|---|---|---|
|||||||||
|||||||||
|||||||||
|||||||||
|||||||||
|||||||||
|||||||||


Iterations


BOPE-III BOPE-I BOPE-II BOPE-IV HVGP RS


Figure 18: **Extended** **comparison** **with** **baselines** **(Case-5)** . Comparison against BOPE based
approaches, plotted against Random sampling for reference.


shows the comparison on Power Grid Resource Allocation case study (Section 5.1), Case-5 problem,
using preference score as a metric of evaluation.


    - BOPE-I: A two-phase strategy that begins with qEUBO (utility-focused exploration,(Astudillo & Frazier, 2019)) and switches to qNEI in the second half (objective-driven
refinement), useful for testing the effect of premature exploitation.

    - BOPE-II: A two-phase strategy that begins with qNEI (to explore the objective space) and
switches to qEUBO in the second half of optimization (to exploit the learned preference
model), using a frozen outcome snapshot for qEUBO.

    - BOPE-III:A qEUBO-only variant where experiment selection uses qEUBO with a new
objective realization sampled at every iteration instead of a frozen snapshot, encouraging
higher exploration through objective variation.

    - BOPE-IV: A baseline BOPE variant that uses standard preference exploration (EUBO- _τ_ )
and selects experiments exclusively with qNEI, fully refitting both objective and subjective
GPs after each update.


**Results.** We observe that our method explores higher preference regions after some initial exploration, unlike BOPE variants, where the performance is highly sensitive to design choices. BOPE-I
starts off with higher preference values due to optimization driven approach, but fails to handle both


28


optimization and exploration, leading to loss of convergence in second half. This also shows that the
performance is sensitive to the point where the design stages are switched. BOPE-II explores first,
and then performs optimization, leading to better performance eventually, however, as in BOPE-I,
the complete disconnect bewteen exploration and exploitation leads to sub-optimal performance.
BOPE-III combines both exploration and exploitation but is still more exploitation centric, leading to
higher preference scores than BOPE-I,II but sub-optimal compared to HVGP. BOPE-IV and our
method HVGP have similar trends, due to similar concept of refitting both models at each stage, but
BOPE-IV converges to a sub-optimal value. Our acquisition strategy combines information from
both objective and subjective layers at once, which leads to higher sample efficiency, and discovery
of higher preference region.


29