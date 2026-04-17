000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 047 048 049 050 051 052 053

# The Minimal Search Space For Conditional Causal Bandits

Anonymous authors Paper under double-blind review

## Abstract

Causal knowledge can be used to support decision-making problems. This has been recognized in the causal bandits literature, where a causal (multi-armed) bandit is characterized by a causal graphical model and a target variable. The arms are then interventions on the causal model, and rewards are samples of the target variable. Causal bandits were originally studied with a focus on hard interventions.

We focus instead on cases where the arms are *conditional interventions*, which more accurately model many real-world decision-making problems by allowing the value of the intervened variable to be chosen based on the observed values of other variables. This paper presents a graphical characterization of the minimal set of nodes guaranteed to contain the optimal conditional intervention, which maximizes the expected reward. We then propose an efficient algorithm with a time complexity of O(|V | + |E|) to identify this minimal set of nodes. We prove that the graphical characterization and the proposed algorithm are correct. Finally, we empirically demonstrate that our algorithm significantly prunes the search space and substantially accelerates convergence rates when integrated into standard multi-armed bandit algorithms.

## 1 Introduction

Lattimore et al. (2016) introduced a class of problems termed *causal bandit* problems, where actions are interventions on a causal model, and rewards are samples of a chosen reward variable Y belonging to the causal model. They focus on hard interventions, where the intervened variables are set to specific values, without considering the values of any other variables. We will refer to this as a hardintervention causal bandit problem. They propose a best-arm identification algorithm that utilizes observations of the non-intervened variables in the causal model to accelerate learning of the best arm as compared to standard multi-armed bandit (MAB) algorithms. Causal bandits have applications across a broad range of domains, particularly in scenarios requiring the selection of an intervention on a causal system. These include computational advertising and context recommendation (Bottou et al., 2013; Zhao et al., 2022), biochemical and gene interaction networks (Meinshausen et al., 2016; Basharin, 1959), epidemiology (Joffe et al., 2012), and drug discovery (Michoel & Zhang, 2023). Most of the work in causal bandits (see Section 7) focuses on developing MAB algorithms which incorporate knowledge about the causal graph. Lee & Bareinboim (2018), in contrast, use the fact that the causal graph is known not to develop yet another MAB algorithm, but to reduce the set of nodes (*i.e.* variables) of the causal graph on which hard interventions should be examined, thereby reducing the search space for hard-intervention causal bandit problems. This reduction of the search space significantly improves and scales the applicability of existing causal MAB algorithms. It is recognized in the MAB literature that, for many if not most applications, actions are taken in a context, that is, with available information (Lattimore & Szepesvári, 2020; Agarwal et al., 2014; Dudik et al., 2011; Jagerman et al., 2020; Langford & Zhang, 2007). *E.g.*, content recommendation based on the user's demographic characteristics, such as age, gender, nationality and occupation.

Similarly, in causality, conditional interventions - where a variable X is set to a value g(ZX) through some function g after observing a set of variables (a context) ZX - are more realistic than hard or soft1interventions in many real-world scenarios. Conditional interventions were first introduced in Pearl (1994) based on the argument that "In general, interventions may involve complex policies in 1In a soft intervention, the intervened variable keeps its direct causes (Peters et al., 2017).

1 054 055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082 083 084 085 086 087 088 089 090 091 092 093 094 095 096 097 098 099 100 101 102 103 104 105 106 107 which a variable X is made to respond in a specified way to some set ZX of other variables." Shpitser
& Pearl (2012) motivate their interest in conditional interventions by providing the concrete example of a doctor selecting treatments based on observed symptoms and medical test results ZX to improve the patient's health condition. The doctor performs interventions of the form do(Xi = xi), but "the specific values of the treatment variables are not known in advance, but instead depend on symptoms and test results performed 'on the fly' via policy functions gi" (Shpitser & Pearl, 2012). Formally, this is denoted do(Xi = g(ZXi)). See the paragraph on conditional interventions in Section 2 for further motivation and details about ZX
Novelty and contributions: This work, like that of Lee & Bareinboim (2018), leverages the causal graph to reduce the search space of the MAB problem, thereby accelerating MAB algorithms applied to it and effectively serving as a pre-processing step for (causal) MAB problems. While Lee & Bareinboim (2018) study causal bandits with multi-node hard interventions in the presence of latent confounders, we focus on single-node conditional interventions under the assumption of no latent confounders. As discussed in Section 2, restricting to single-node interventions in fact makes the problem more challenging, as does considering conditional rather than hard interventions. Therefore, our work addresses a fundamentally different and non-comparable problem from that of Lee & Bareinboim (2018). Because the single-node intervention problem without latent confounders is already highly non-trivial, we leave latent confounders to future work, making our study a necessary step toward the general case. The setting we study remains widely applicable - for instance, to the examples discussed in Section 2. Explicitly, our work is novel because we consider the case where (i) the arms are *conditional interventions* (which generalize both hard and soft interventions); and (ii) the interventions are *single-node interventions*. This is the first time the minimal search space for a causal bandits problem with non-hard interventions is fully characterized. Such a characterization has also not been done for single-node interventions (of any kind). Our contributions are as follows: (a) we establish a graphical characterization of the minimal set of nodes guaranteed to contain the optimal node on which to perform a conditional intervention; and (b) we propose an algorithm which finds this set, given only the causal graph, with a time complexity of O(|V| + |E|), that is, linear in the number of nodes and edges of the causal graph. As a supplementary result, we also show that, perhaps surprisingly, the exact same minimal set would hold for the optimization problem of selecting an atomic (i.e. single-node and hard) intervention in a deterministic causal model. We provide proofs for the graphical characterization and correctness of the algorithm, as well as experiments that assess the fraction of the search space that can be expected to be pruned using our method, in both randomly generated and real-world graphs, and demonstrate, using well-known real-world models, that our intervention selection can significantly improve a classical MAB algorithm. Note that, if the true causal graph is unknown and instead a family of candidate graphs is available, the C4 algorithm can simply be applied to each candidate graph, and the results combined by taking the union of the resulting minimal search spaces. All proofs of the results presented in the paper can be found in the appendix. The code repository with the experiments can be found in the supplementary material.

## 2 Preliminaries

Graphs and causal models We will make use of Directed Acyclic Graphs (DAGs). The main concepts of DAGs and notation used in this paper are reviewed in Appendix A. Furthermore, we operate within the Pearlian graphical framework of causality, where causal systems are modeled using Structural Causal Models (SCMs) (Peters et al., 2017; Pearl, 2009). An SCM C is a tuple
(V, N, F, pN), where V = (V1*, . . . , V*n) and N = (NV1
, . . . , NVn
) are vectors of random variables.

The exogenous variables are pairwise independent, and are distributed according to the noise distribution pN, while each endogenous variable Viis a deterministic function fViof its noise variable NViand a (possibly empty) set of other endogenous variables Pa(Vi), called the parents of Vi. The Vi and NViare called *endogenous* and exogenous (or *noise*) variables, respectively. RV denotes the range of the random variable V . F is a set of functions fVi: RPa(Vi) × RNVi → RVi, termed structural assignments. The endogenous variables together with F characterize a DAG called the causal graph GC := (V, E) of C, whose edge set is E = {(*P, X*) : X ∈ V, P ∈ Pa(X) \ {X}}.

We denote by C(G) the set of SCMs whose causal graph is G. Having an SCM allows us to model interventions: intervening on a variable changes its structural assignment fX to a new one, say ˜fX.

This intervention is then denoted do(fX = ˜fX). In the simplest type of interventions, called atomic interventions, a variable X is set to a chosen value x, thus replacing the structural assignment fX
108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 of X with a constant function setting it to x. Such an intervention is denoted do(X = x), and the SCM resulting from performing this intervention is denoted C
do(X=x). The joint distribution over the endogenous variables resulting from the atomic intervention do(X = x) is denoted p do(X=x)and called the *post-intervention distribution* for this intervention. Each realization n ∈ RN of the noise variables will be called a *unit*. A *deterministic SCM* is an SCM for which the noise distribution is a point mass distribution with all its mass on some (known) unit n ∈ RN. Finally, nodes are denoted by upper case letters, sets of nodes by boldface letters, and variable values by lower case letters. We will make use of the fact that the structural assignments of the ancestors of an endogenous variable X
(including its own structural assignment) can be composed to express X as a function ¯fX(n) of the vector n of exogenous variables values. We call this2the *unrolled assignment* of X. Conditional interventions Given an SCM C = (V, N, F, pN) with causal graph G, X ∈ V, ZX ⊆ V \ {X}, and a (any) function g : RZX → RX (which we call a *policy for* X), the conditional intervention on X given ZX *for the policy* g, denoted do(X = g(ZX)), is the intervention where the value of X is determined by that of ZX through g (Pearl, 2009). The precise conditioning set ZX
for each X is pre-determined by the specific problem or application, or by the practitioner. In order to systematically study conditional interventions, we will need to make some assumptions of what nodes can reasonably be in ZX, *i.e.* what variables can we expect to have knowledge of at the time of applying the policy g to intervene on X. As noted in Pearl (1994; 2009), the nodes in ZX cannot be descendants of X in G. Hence, ZX ⊆ V \ De(X). On the other hand, all (proper) ancestors of X are realized before X. Since we will be dealing with the case with no latent variables, we can assume that all ancestors of X are observed, and can be used by a policy g to set X to a value g(ZX). Thus, we assume3that An(X)\{X} ⊆ ZX. We will then focus on the case where for each X, the conditioning set ZX, chosen by the practitioner, obeys the inclusion relations An(X) \ {X} ⊆ ZX ⊆ V \ De(X).

Furthermore, we focus on cases where the context that is available for an intervention is also available for later interventions. As an example, consider the case where a traffic controller needs to intervene on the delay Di,s of a train i at a train station s (for example by forcing it to wait for 5 extra minutes before departing). Clearly, all delays Di
′,s′ of all train/station pairs affecting Di,s have already been observed, and can therefore be used when selecting Di,s. As another example, similar to the one used in Pearl (1994) when first introducing conditional interventions, consider the situation where a doctor must decide, over a period of three weeks, whether and when to intervene on the weight, blood pressure or renal blood flow of a patient, in order to improve the patient's kidney function. The goal is to maximize kidney function (variable Kidney3) at the end of the third week. Due to side-effects, the patient can only be prescribed medication for one week. The causal graph for this situation can be found in Figure 4, Appendix B. Notice that at the time of intervening on a node Xi, the doctor can use information about all measurements made until then. For instance, all the data available when performing an intervention on the renal flow on week 1 (node RenalFlow1) will also be available when intervening on the renal flow on week 2 (node RenalFlow2). Mathematically, this last assumption can be written W ∈ An(X) ⇒ ZW ⊆ ZX. We then say that ZX is an observable conditioning set for X.

Conditional causal bandits Recall that a MAB problem consists of an agent pulling an arm a ∈ A
at each round t, resulting in a reward sample Yt from an unknown distribution associated to the pulled arm (Lattimore & Szepesvári, 2020). We denote the mean reward for arm a by µa and the mean reward for the best arm by µ
∗ = maxa∈A µa. The objective is to maximize the total reward obtained over all T rounds. Equivalently, this can be framed as minimizing the cumulative regret RegT = T µ∗ −PT
t=1 E[Yt]. We now introduce a novel type of (causal) MAB problem. Consider the setting where the bandit's reward is a (endogenous) variable Y in an SCM C = (V, N, F, pN), and the arms are the conditional interventions do(X = g(ZX)), where X ∈ V \ {Y }. Furthermore, the agent has knowledge of the causal graph G of C, but not of the structural assignments F or the noise distribution pN. We call this a *single-node conditional-intervention causal bandit*, or simply conditional causal bandit. The reward distribution for arm do(X = g(ZX)) is the post-intervention distribution p do(X=g(ZX))
Y, and is unknown to the agent, since it has no knowledge of F. Notice that selecting an arm can be subdivided in (i) choosing a node X to intervene on; and (ii) choosing a policy g, i.e. choosing a value to set X to given the observed variables ZX. *We do not impose any* 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 restrictions on the function g. The conditioning sets ZX are *specified in advance*, as described in the paragraph on conditional interventions above. In this paper, we find the minimal set of nodes that need to be considered by the agent in step (i). The value of X chosen in step (ii) can be selected by an MAB algorithm. As stressed in Section 1, the novelty of our problem lies in the fact that we deal with *conditional* interventions that are *single-node*. Both of these characteristics of our problem complicate the analysis. Unsurprisingly, searching over conditional interventions is more complicated than over hard or soft interventions. Perhaps more unexpectedly, single-node interventions also make a search for a minimal search space more involved. Indeed, if one allows for interventions on arbitrary sets, one simply needs to intervene on all the parents Pa(Y ) of Y (Lee & Bareinboim, 2018). Since in our case the agent cannot do this whenever |Pa(Y )| > 1, the minimal search space will, as we will see, be complex even without unobserved confounding. That said, the assumption that there is no unobserved confounding is a limitation of this paper, and a natural next step for future work (see Section 7).

## 3 Conditional-Intervention Superiority

In this section, we will define a preorder ⪰
cY of "conditional-intervention superiority" on nodes of an SCM. If X⪰
c Y W, then W can never be a better node than X to intervene on with a conditional intervention4. We will then show that, perhaps surprisingly, this relation is equivalent to another superiority relation, defined in terms of atomic interventions in a deterministic SCM. Definition 1 (Conditional-Intervention Superiority). Let X, W, Y *be nodes of a DAG* G. X is conditional-intervention superior to W relative to Y in G*, denoted* X⪰
c Y W, if for all SCM with causal graph G there is a policy g for X such that for every observable conditioning sets ZX and ZW for X and W and all policies h for W,

$$\mathbb{E}_{\mathbf{n}}{\bar{f}}_{Y}^{d o(X=g(\mathbf{Z}_{X}))}(\mathbf{n})\geq\mathbb{E}_{\mathbf{n}}{\bar{f}}_{Y}^{d o(W=h(\mathbf{Z}_{W}))}(\mathbf{n}).$$

A similar relation can be defined for atomic interventions in deterministic SCMs, where the vector N of exogenous variables is fixed to a *known* value n (see Section 2). Definition 2 (Deterministic Atomic-Intervention Superiority). Let X, W, Y *be nodes of a DAG* G. X is deterministically atomic-intervention superior to W relative to Y *, denoted* X ⪰
det,a Y W,
if for every SCM C with causal graph G and every unit n there is x ∈ RX such that no atomic intervention on W results in a larger Y than the value of Y resulting from setting X = x*. That is,* for all (C, n) ∈ C(G) × RN:

$$(1)$$
$$\exists x\in R_{X}\colon\forall w\in R_{w},\;{\bar{f}}_{Y}^{d o(X=x)}(\mathbf{n})\geq{\bar{f}}_{Y}^{d o(W=w)}(\mathbf{n}).$$
$$(2)$$
Y(n). (2)
We extend Definitions 1 and 2 for sets of nodes in the obvious way: X is superior to W if every node in W is inferior to some node in X. Definition 3. Let now X,W *be sets of nodes of* G. X is conditional-intervention superior *(respectively* deterministic atomic intervention superior) to W*, also denoted* X⪰
cY W *(respectively* X ⪰
det,a Y W),
if ∀W ∈ W, ∃X ∈ X *such that* X⪰
cY W *(respectively* X ⪰
det,a Y W).

The two relations ⪰
cY, ⪰
det,a Yactually coincide (both for nodes and sets of nodes).

Proposition 4 (Conditional vs Atomic superiority). Let X, W, Y be nodes in a DAG G*. Then* X is conditional-intervention superior to W relative to Y in G if and only if X *is deterministic* atomic-intervention superior to W relative to Y in G*. That is,* X⪰
c Y W ⇔ X ⪰
det,a Y W.

Since these two relations are equivalent, we henceforth refer simply to interventional superiority without further specification, and use the symbol ⪰Y when distinguishing them is not necessary. We will use Proposition 4 to simplify our problem. Since deterministic atomic interventions are easier to reason about, we use them in formulating proposals for the minimal search space and in our proofs.

216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269

![4_image_0.png](4_image_0.png)

![4_image_1.png](4_image_1.png)

![4_image_3.png](4_image_3.png)

![4_image_2.png](4_image_2.png)

## 4 Graphical Characterization Of The Mgiss

Goal Our aim is to develop a method to identify, based on a causal graph G, the smallest set of nodes that are "worth testing" when attempting to maximize Y by performing one single-node intervention. Specifically, regardless of the structural causal model C associated with G, we want to ensure that the optimal intervention can be discovered within this selected set of nodes. We define this set as follows: Definition 5 (GISS and mGISS). Let G be a DAG with set of nodes V. A globally interventionally superior set (GISS) of G relative to Y , is a subset U of V \ {Y } satisfying U ⪰Y (V \ {Y }) \ U. A
minimal globally interventionally superior set (mGISS) is a GISS which is minimal with respect to set inclusion. This set is unique, so that we can talk of the minimal globally interventionally superior set. Proposition 6 (Uniqueness of the mGISS). Let G be a DAG and Y a node of G with at least one parent. The minimal globally interventionally superior set of G relative to Y *is unique. We denote it* by mGISSY (G). Intuition Since the value of Y is completely determined by the values of its parents A1*, . . . , A*m, along with the fixed value nY of a noise variable that cannot be intervened upon (see Definition 2), we aim to induce the parents to acquire the combination of values (a
∗1*, . . . , a*∗m) that maximizes Y when NY = nY . If this is not possible to achieve using a single intervention, we aim to obtain the best combination possible. Clearly, the parents of Y themselves need to tested by bandit algorithms: there may be one parent on which Y is highly dependent, in such a way that there is a value of that parent which will maximize Y . In the particular case where Y has a single parent A, that node is the only node worth intervening on, since all other nodes can only influence Y through A. Indeed, if a
∗ ∈ RA
is the value of A which maximizes Y , it is not necessary to try to find an intervention on ancestors of A which results in A = a
∗: just set A = a
∗ directly (Figure 1c). If Y has two or more parents, it is possible that a single intervention on one of the Ai does not yield the best possible outcome. Instead, a better configuration (potentially even the ideal case (a
∗1
, . . . , a∗m)) may be achieved by intervening on a common ancestor of some or all of the Ai (Figure 1a). Notice that X0 is also a common ancestor of A1, A2, but one is never better off intervening on X0 than on X1. This seems to indicate that testing interventions on, for instance, all lowest common ancestors (LCAs, see Appendix A) of the parents of Y , and only them, is necessary. While this works in Figure 1a, it fails for a graph such as Figure 1d, where X needs to be tested and yet it is not in LCA(A1, A2) = {A1}. This suggests that we need to define a stricter notion of common ancestor to make progress in characterizing mGISSY (G). Definition 7 (Lowest Strict Common Ancestors of a Pair of Nodes). The node V ∈ V *is a* strict common ancestor of X, Y ∈ V if V is a common ancestor of X, Y from which both X and Y can be reached from V *with paths* V 99K X and V 99K Y not containing Y and X, respectively. The set of strict common ancestors of X, Y *is denoted* SCA(X, Y ). Furthermore, V *is a* lowest strict common ancestor of X, Y ∈ V if V *is a minimal element of* SCA(X, Y ) with respect to the ancestor partial order ≼. The set of lowest strict common ancestors of X, Y *is denoted* LSCA(*X, Y* ).

Definition 8 (Lowest Strict Common Ancestors of a Set). Let U ⊆ V and V ∈ V \ U. The node V is a lowest strict common ancestor of U *if it is a lowest strict common ancestor of some pair of nodes* 270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323 Our heuristic argument so far suggests that we need to test the parents of Y and their LSCAs. However, there are additional nodes that must be considered: the reasoning for testing the lowest strict common ancestors of the parents can be repeated. For instance, in Figure 1b, the best possible configuration of the Ai may be achieved by intervening on Z. Such an intervention could result in a combination of values of X1 and A2 that leads to the best possible combinations of A1 and A2. This suggests that the mGISSY (G) should be determined by recursively finding all the LSCAs of the parents of Y , then the LSCAs of that set, and so on, ultimately resulting in what we call the "LSCA closure of the parents of Y ", denoted L∞(Pa(Y )). In the remainder of this section, we formally define L∞(Pa(Y )), find a simple graphical characterization for it, and prove that it indeed equals mGISSY (G).

Definition 9 (LSCA closure). For every i ∈ N we define the ith*-order LSCA set* L
i(U) of U ⊆ V as follows:
L
0(U) := U, and L
i(U) := LSCA(L
i−1(U)) ∪ Li−1(U). (4)
The LSCA closure L∞(U) of U *is given by*5 L
∞(U) := L
k
∗(U), *where* k
∗ = min{i ∈ N: L
i(U) = L
i+1(U)}. (5)
Example 10. Consider the graph in Figure 1b and set U = {A1, A2}. Then, L
0(U) =
{A1, A2},L
1(U) = {X1, A1, A2},L
2(U) = L
3(U) = {Z, X1, A1, A2} = L∞(U).

We will introduce the notion of "Λ-structures" (Figure 2a), which provides an alternative, elegant, simple graphical characterization of L∞(Pa(Y )). It will also be instrumental in the proofs of the main results of this paper.

Definition 11 (Λ-structure). Let V, A, B ∈ V*. Furthermore, let* πA : V 99K A, πB : V 99K B be paths. The tuple (V, πA, πB) *is a* Λ-structure over (A, B) if πA and πB only intersect at V . Now, let U,W ⊆ V. The node V *is said to* form a Λ-structure over (U,W) *if there are nodes* U ∈ U
and W ∈ W, and paths πU : V 99K U, πW : V 99K W such that (V, πU , πW ) is a Λ*-structure over*
(U, W). Denote by Λ(U,W) the set of all nodes forming a Λ*-structure over* (U,W).

Notice that, if V ∈ U ∩ W, then trivially V ∈ Λ(U,W): just take the trivial paths π = π
′ = (V ).

Theorem 12 (Simple Graphical Characterization of LSCA Closure). A node V ∈ V *is in the LSCA*
closure L∞(U) of U ⊆ V if and only if V forms a Λ*-structure over* (U, U). I.e. L∞(U) = Λ(U, U).

![5_image_1.png](5_image_1.png)

(a) A Λ-structure over (U, U). Theorem 12 states that the LSCA closure L
∞(U) of a set U is the set of all such structures.

![5_image_0.png](5_image_0.png)

## Figure 2

We are now ready for the main result of this paper. Theorem 13 (Superiority of the LSCA Closure). Let G be a causal graph and Y a node of G *with at* least one parent. Then, the LSCA closure L∞(Pa(Y )) of the parents of Y *is the minimal globally* interventionally superior set mGISSY (G) of G *relative to* Y . We emphasize that, due to Proposition 4, this graphical characterization of the mGISSY (G) is valid both for conditional interventions in a probabilistic causal model as for atomic interventions in a deterministic causal model (i.e. a causal model with known n).

5Notice that the existence of the k
∗is guaranteed, since by construction L
i(U) ⊆ Li+1(U) ⊆ V for all i ∈ N and V is finite. U, U′in U*. The set of lowest strict common ancestors is denoted* LSCA(U)*. That is,*
LSCA(U) := {V ∈ V \ U: ∃U, U′ ∈ U *s.t.* V ∈ LSCA(*U, U*′)}. (3)

## 5 Algorithm To Find The Minimal Globally Interventionally Superior Set

1: **input:** DAG G = (V, E), set of nodes U ⊆ V
2: **output:** The closure L∞(U)
3: S ← U ▷ initialize closure 4: c[V ] ← V for V ∈ U ▷ initalize connectors 5: for V ∈ V\U in reverse topological order do 6: C ← {c[V
′] : V
′ ∈ Ch(V ) ∩ An(U)}
7: if |C| = 1 **then** 8: c[V ] ← X where C = {X} 9: else if |C| > 1 **then**
10: c[V ] ← V , S ← S ∪ {V } ▷ V is added to closure 11: **return** S
324 325 326 327 328 329 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377 Crucially, Lemma 15 implies that V ∈ L∞(U) ⇔ c[V ] = V . Intuitively, if all children of V have the same connector X (*i.e.* C = {X}), then V can only influence U via X, making X interventionally superior to V , and thus V /∈ L∞(U). On the other hand, if V 's children have multiple connectors (*i.e.*
|C| > 1), then interventions on V can influence all those connectors, so V is a potentially worthwhile candidate for intervention, and thus V ∈ L∞(U). This establishes correctness of C4, which finds all nodes satisfying c[V ] = V in linear time. Theorem 16. C4 correctly computes L∞(U), and runs in O(|V| + |E|) *time.*

## 6 Experimental Results

We evaluate C4 on both random and real graphs. Additionally, we examine the impact of our method on the cumulative regret of a bandit algorithm. Search space reduction in random graphs We applied the C4 algorithm to randomly generated DAGs using the ErdoS-Rényi model for ˝ N graphs and probability p (Erdos & Rényi, 1959) adapted ˝
to DAG-generation6. We generated 1000 graphs using 20, 100, 300, and 500 nodes, and varying the expected (total) degree of nodes from 2 to 11 in steps of 3. For each graph G, we set the target Y to be the node with the most ancestors, used C4 to compute L∞(Pa(Y )) = mGISSY (G), and calculated the fraction of nodes in An(Y ) \ {Y } that remain in mGISSY (G). The results revealed that, for a given number of nodes, graphs with lower expected degrees benefit more from our method (*i.e.* their mGISSY (G) correspond to smaller fractions of An(Y ) \ {Y }). Furthermore, for a fixed expected degree, our method is more effective for higher numbers of nodes. For example, for graphs with 500 Lemma 15 illuminates the connector's relation to L∞(U): c[V ] "connects" V to L∞(U) in that it is the first node in L∞(U) in any path from V to L∞(U). Thus, c[V ] mediates all influence that V exerts over L∞(U).

Lemma 15. Let G = (V, E) *be a DAG,* U ⊆ V, V ∈ An(U). c[V ] *is the unique node s.t. a path* πc[V ]: V 99K c[V ] exists where πc[V ] ∩ L∞(U) = {c[V ]} (if V *is its own connector, the path is* trivial). This is equivalent to: for every node X ∈ L∞(U) *and path* πX : V 99K X, c[V ] *is the* maximal element of πX ∩ L∞(U) *w.r.t. the ancestor partial order* ≼.

The Closure Computation via Children with Multiple Connectors (C4) Algorithm (Algorithm 1)
computes the closure L∞(U) in O(|V| + |E|) time, using *connectors* (illustrated in Figure 2b):
Definition 14 (Connector). Let G = (V, E) *be a DAG,* U ⊆ V, V ∈ An(U)*. The* U-connector c[V ] of V in G *is defined recursively. Let* C = {c[V
′] : V
′ ∈ Ch(V ) ∩ An(U)} be the set of V 's children's connectors. If V ∈ U*, then* c[V ] := V . If V /∈ U*: if* |C| = 1 and C = {X} *then* c[V ] := X*, otherwise* c[V ] := V . Algorithm 1 C4 378

![7_image_0.png](7_image_0.png) 379 380 381 382 383 384 385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431 nodes, the mGISS retained, on average, 17%, 29%, 62% and 77% of the nodes, for expected degrees of 2, 5, 8 and 11, respectively. Moreover, graphs with an expected degree of 5 saw these numbers decrease from 70% at 20 nodes to 47%, 35% and 29% for 100, 300 and 500 nodes, respectively. The complete results are presented in Figure 5 (Appendix H). These results are not surprising: if the average degree is small compared to the number of nodes, the edge density is small, in which case we expect fewer Λ-structures to form over Pa(Y). Graphs modeling real-world systems tend to have low average degrees, as can be seen in the graphs from the popular Bayesian network repository bnlearn. Therefore, we expect our method to be especially effective in those graphs. We test this below. Search space reduction in real-world graphs We tested our method in most graphs from the bnlearn repository7, as well as on a graph representing the causal relationships between train delays in a segment of the railway system of the Netherlands (see Appendix H). For each graph, we set Y to be the node with most ancestors8. The results are presented in the bar plot of Figure 6
(Appendix H). This confirmed that realistic models with larger graphs tended to benefit more from our method, with a reduction of over 90% of the search space for some of the largest models. Notice also that these models indeed have relatively small average degrees, all below 4.0. From this, we conclude that we can expect our method to be useful when reducing the search space of conditional causal bandit tasks in real-world causal models, especially when they are large. Impact on conditional intervention bandits We present empirical evidence that restricting the node search space to the mGISS allows a straightforward UCB-based9algorithm (which we call CondIntUCB) for conditional causal bandits to converge more rapidly to better nodes. As explained in Section 2, on each round the algorithm must (i) choose which node X to intervene on; and (ii) choose the value for X, given its conditioning set ZX
10. Choice (i) employs UCB over nodes, while choice (ii) utilizes a UCB instance specific to the conditioning set value. In other words, for each realization of ZX (each context) there is a UCB. This is identical to what is described in Lattimore &
Szepesvári (2020, §18.1) for contextual bandits with one bandit per context. The cumulative regret11is computed with respect to node choice, since we want to see how our node selection method affects the quality of node choice by CondIntUCB. We use 4 real-world datasets from the bnlearn repository, and again choose the node of each dataset with the most ancestors as the target8. These datasets were selected because their graphical structures are non-trivial12and both An(Y ) and mGISSY (G) are sufficiently small to allow experimentation with our setup. For each dataset, we run CondIntUCB up to 500 times and plot the two average cumulative regret curves along with their standard deviations, corresponding to using all nodes (brute-force) and the mGISS nodes (Figure 3). The total number of 7All which can be imported in Python using the library pgmpy. 8We also require Y to have more than one parent, to avoid the trivial case with |mGISSY (G)| = 1. 9The Upper Confidence Bound (UCB) algorithm is widely used. See *e.g.* Lattimore & Szepesvári (2020).

10For simplicity, we use the smallest observable conditioning set ZX = An(X) \ {X} (see Section 2).

11For the computation of regret, we use the estimated best arm, defined as the arm that most runs concluded to be the best at the end of training.

12In contrast, the cancer dataset, for example, only has nodes whose mGISS is either all of the node's ancestors or a single node.

432 433 434 435 436 437 438 439 440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485 rounds is chosen as to observe (near) convergence. These results show that cumulative regret curves can be significantly improved—meaning that better nodes are selected earlier for applying conditional interventions—if the search space over nodes is pruned using our C4 algorithm.

## 7 Related Work And Conclusion

Recent works in "contextual causal bandits" address interventions that account for context, bearing resemblance to our problem. However, our problem remains distinct. In a K-arm contextual bandit problem, each round is associated with a context that determines the reward distributions of the K arms. The agent uses the context to select one of the K arms. A general approach to solving such problems is to maintain a separate standard bandit algorithm for each context. More efficient solutions typically rely on assumptions about relationships between contexts (Lattimore & Szepesvári, 2020). In contrast, a conditional causal bandit problem involves, in each round, an intervened node X and an observed context that is a sample of ZX. This context determines the reward distributions of the K = |RX| possible atomic interventions on X, and the agent chooses among these according to a policy. Thus, a conditional causal bandit problem can be interpreted as a collection of contextual bandits, one for each node X in a causal graph. In particular, conditional causal bandits are not simply particular cases of contextual bandits. In this paper, we leverage the structure of the causal graph to eliminate certain nodes, i.e., to exclude some of these contextual bandits from consideration. In Madhavan et al. (2024), the term "contexts" is used in a very different way to the one used in our paper, actually referring to different graphs as opposed to different variable values. Subramanian & Ravindran (2022; 2024) tackle the scenario in which an intervention is performed, with knowledge of a given set of context variables, on a *pre-chosen* variable X that has an edge into Y (and no other outgoing edges). This approach can be understood as selecting a conditional intervention for a predefined node from a very simple graph. In contrast, in our setting we need to choose what variable to intervene on to begin with, and there are no restrictions on the causal graph.

As mentioned in Section 1, Lattimore et al. (2016) introduced the original causal bandit problems, which involve hard interventions in causal models. Subsequent works (Sen et al., 2017; Yabe et al.,
2018; Lu et al., 2020; Nair et al., 2021; Sawarni et al., 2023; Maiti et al., 2022; Feng & Chen, 2023) proposed algorithms for variants of causal bandits with both hard and soft interventions, budget constraints, and unobserved confounders. All of the works described above proposed algorithms which aim at accelerating learning by utilizing knowledge of the causal model. As explained in Section 1, this contrasts with our work, which, like the work of Lee & Bareinboim (2018; 2019), uses knowledge of the causal graph to find a minimal search space (over the nodes) for causal bandits. And, while the latter focus on multi-node, hard interventions, we focus on single-node, conditional interventions. The work of Lee & Bareinboim (2020) presents an interesting connection to our work. Given a causal graph, they study the sets of pairs (node, context(node)) (referred to as "scopes") that may correspond to an optimal (multi-node) intervention policy where each node X in a scope is intervened on according to a policy πX(X | context(X)). This is a challenging problem, and they do not provide a full characterization of these optimal scopes, instead deriving a set of rules that can be used to compare certain pairs of scopes. In this paper, we instead address the single-node intervention case, and assume that the problem sets the conditioning set ZX (context) to use and impose only minimal restrictions on what ZX can be, focusing instead on choosing the nodes that can yield the best results. To conclude, in this paper we introduced the conditional causal bandit problem, where the agent only has knowledge of the causal graph G, the arms are conditional interventions, and the reward variable belongs to G. The theoretical contributions include a rigorous, simple graphical characterization of the minimal set of nodes which is guaranteed to contain the node with the optimal conditional intervention, and the C4 algorithm, which computes this set in linear time. Empirical results validate that our approach substantially prunes the search space in both real-world and sparse randomlygenerated graphs. Furthermore, integrating mGISS with a UCB-based conditional bandits algorithm showcased improved cumulative regret curves. While Lee & Bareinboim (2020) consider multi-node interventions, it would be interesting in future work to adapt their ideas to the single-node case to identify the smallest ZX sets for which the best policy can still be found. Addressing latent confounding would also require substantially more research and is thus left as future work. On the practical side, instead of combining C4 with the simple CondIntUCB, one could replace CondIntUCB
486 487 488 489 490 491 492 493 494 495 496 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539 All experiments and results described in Section 6 can be reproduced using the code in the repository submitted alongside the paper. The experiments are simple to run, and instructions are included in the repository itself. All theoretical results are proved in the appendix.

## References

Alekh Agarwal, Daniel Hsu, Satyen Kale, John Langford, Lihong Li, and Robert Schapire. Taming the monster: A fast and simple algorithm for contextual bandits. In *International Conference on* Machine Learning, pp. 1638–1646. PMLR, 2014.

Georgij P Basharin. On a statistical estimate for the entropy of a sequence of independent random variables. *Theory of Probability & Its Applications*, 4(3):333–336, 1959.

Michael A Bender, Martin Farach-Colton, Giridhar Pemmasani, Steven Skiena, and Pavel Sumazin.

Lowest common ancestors in trees and directed acyclic graphs. *Journal of Algorithms*, 57(2): 75–94, 2005.

Léon Bottou, Jonas Peters, Joaquin Quiñonero-Candela, Denis X Charles, D Max Chickering, Elon Portugaly, Dipankar Ray, Patrice Simard, and Ed Snelson. Counterfactual reasoning and learning systems: The example of computational advertising. *Journal of Machine Learning Research*, 14 (11), 2013.

Miroslav Dudik, Daniel Hsu, Satyen Kale, Nikos Karampatziakis, John Langford, Lev Reyzin, and Tong Zhang. Efficient optimal learning for contextual bandits. *arXiv preprint arXiv:1106.2369*,
2011.

P. Erdos and A. Rényi. On random graphs. i. ˝ *Publicationes Mathematicae*, 6(3–4):290–297, 1959. Shi Feng and Wei Chen. Combinatorial causal bandits. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, pp. 7550–7558, 2023.

Rolf Jagerman, Ilya Markov, and Maarten De Rijke. Safe exploration for optimizing contextual bandits. *ACM Transactions on Information Systems (TOIS)*, 38(3):1–23, 2020.

Michael Joffe, Manoj Gambhir, Marc Chadeau-Hyam, and Paolo Vineis. Causal diagrams in systems epidemiology. *Emerging themes in epidemiology*, 9:1–18, 2012.

John Langford and Tong Zhang. The epoch-greedy algorithm for multi-armed bandits with side information. *Advances in Neural Information Processing Systems*, 20, 2007.

Finnian Lattimore, Tor Lattimore, and Mark D Reid. Causal bandits: Learning good interventions via causal inference. *Advances in Neural Information Processing Systems*, 29, 2016.

Tor Lattimore and Csaba Szepesvári. *Bandit algorithms*. Cambridge University Press, 2020.

Sanghack Lee and Elias Bareinboim. Structural causal bandits: Where to intervene? Advances in Neural Information Processing Systems, 31, 2018.

Sanghack Lee and Elias Bareinboim. Structural causal bandits with non-manipulable variables. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 33, pp. 4164–4172, 2019.

Sanghack Lee and Elias Bareinboim. Characterizing optimal mixed policies: Where to intervene and what to observe. *Advances in Neural Information Processing Systems*, 33:8565–8576, 2020.

with any other conditional bandit algorithm that leverages the model's causal structure. As discussed earlier in this section, no such algorithm currently exists. Nevertheless, we expect that combining C4 with any future algorithm for causal bandits with conditional interventions will be advantageous, as it reduces the number of arms that need to be considered.

## Reproducibility Statement

Yangyi Lu, Amirhossein Meisami, Ambuj Tewari, and William Yan. Regret analysis of bandit problems with causal background knowledge. In *Conference on Uncertainty in Artificial Intelligence*, pp. 141–150. PMLR, 2020.

Rahul Madhavan, Aurghya Maiti, Gaurav Sinha, and Siddharth Barman. Causal contextual bandits with adaptive context. *arXiv preprint arXiv:2405.18626*, 2024.

540 541 542 543 544 545 546 547 548 549 550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593 Aurghya Maiti, Vineet Nair, and Gaurav Sinha. A causal bandit approach to learning good atomic interventions in presence of unobserved confounders. In *Uncertainty in Artificial Intelligence*, pp.

1328–1338. PMLR, 2022.

Nicolai Meinshausen, Alain Hauser, Joris M Mooij, Jonas Peters, Philip Versteeg, and Peter Bühlmann.

Methods for causal inference from gene perturbation experiments and validation. *Proceedings of* the National Academy of Sciences, 113(27):7361–7368, 2016.

Tom Michoel and Jitao David Zhang. Causal inference in drug discovery and development. Drug discovery today, 28(10):103737, 2023.

Vineet Nair, Vishakha Patil, and Gaurav Sinha. Budgeted and non-budgeted causal bandits. In International Conference on Artificial Intelligence and Statistics, pp. 2017–2025. PMLR, 2021.

Judea Pearl. A probabilistic calculus of actions. In *Uncertainty in Artificial Intelligence*, pp. 454–462.

Elsevier, 1994.

Judea Pearl. *Causality*. Cambridge university press, 2009. Jonas Peters, Dominik Janzing, and Bernhard Schölkopf. Elements of causal inference: Foundations and learning algorithms. The MIT Press, 2017.

Ayush Sawarni, Rahul Madhavan, Gaurav Sinha, and Siddharth Barman. Learning good interventions in causal graphs via covering. In *Uncertainty in Artificial Intelligence*, pp. 1827–1836. PMLR, 2023.

Rajat Sen, Karthikeyan Shanmugam, Alexandros G Dimakis, and Sanjay Shakkottai. Identifying best interventions through online importance sampling. In International Conference on Machine Learning, pp. 3057–3066. PMLR, 2017.

Ilya Shpitser and Judea Pearl. Identification of conditional interventional distributions. arXiv preprint arXiv:1206.6876, 2012.

Chandrasekar Subramanian and Balaraman Ravindran. Causal contextual bandits with targeted interventions. In *International Conference on Learning Representations*, 2022.

Chandrasekar Subramanian and Balaraman Ravindran. Causal contextual bandits with one-shot data integration. *Frontiers in Artificial Intelligence*, 7:1346700, 2024.

Akihiro Yabe, Daisuke Hatano, Hanna Sumita, Shinji Ito, Naonori Kakimura, Takuro Fukunaga, and Ken-ichi Kawarabayashi. Causal bandits with propagating inference. In International Conference on Machine Learning, pp. 5512–5520. PMLR, 2018.

Yan Zhao, Mitchell Goodman, Sameer Kanase, Shenghe Xu, Yannick Kimmel, Brent Payne, Saad Khan, and Patricia Grao. Mitigating targeting bias in content recommendation with causal bandits'. In *Proc. ACM Conference on Recommender Systems Workshop on Multi-Objective Recommender* Systems, Seattle, WA., 2022.

## A Directed Acyclic Graphs 594

595 596 597 598 599 600 601 602 603 604 605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647 All graphs in this paper are directed acyclic graphs (DAGs). Every path is assumed to be directed. A path π in a graph G = (V, E) is a tuple of nodes such that each node X in the path has an outgoing arrow from X to the next node in the tuple13. For X ∈ V, we denote by Pa(X), Ch(X),
De(X) and An(X) the sets of parents, children, descendants and ancestors of X, respectively. We denote by π : X 99K Y a path starting at node X and ending at node Y , and ˚π denotes the path formed by the inner nodes of π. By abuse of notation, we often perform set operations such as π1 ∩ π2 between paths, which implicitly means that these operations are performed on the sets of nodes belonging to the paths. Tuples with a single node are also considered to be paths, and are said to be *trivial*. Also, if B ∈ π : X 99K Y , then the paths π| Z : Z 99K Y and π|Z : X 99K Z are the paths resulting from removing from π all nodes before and after Z, respectively. Every node is an ancestor of itself, so that the relation ≼ defined by X ≼ Y ⇐⇒ Y ∈ An(X) is a partial order. Given a set U of nodes, we denote by max≼[U] the set of maximal elements of U with respect to ≼. We call this the *ancestor partial order*. If there is a non-trivial path from X to Y ,
then Y is said to be *reachable* from X. The set of common ancestors of nodes X and Y is denoted CA(*X, Y* ) = An(X) ∩ An(Y ) = {Z ∈ V : Z ≼ X ∧ Z ≼ Y }. Finally, the *degree* of a node in a DAG is the sum of the incoming and outgoing arrows of that node. We also make use of a lesser-known graph theory concept, relevant for this paper: the "lowest common ancestors" of nodes (X, Y ). These are common ancestors that don't reach any other common ancestors, intuitively making them the "closest" to (*X, Y* ). Definition 17 (Lowest Common Ancestors in a DAG (Bender et al., 2005)). Let X, Y *be nodes of a* DAG G = (V, E). A lowest common ancestor (LCA) of X and Y *is a minimal element of* CA(*X, Y* )
with respect to the ancestor partial order ≼. The set of all lowest common ancestors of X and Y is denoted LCA(*X, Y* ). For example, in Figure 1a, LCA(A1, A2) = {X1}, whereas in Figure 1b, LCA(A1, A2) = {A1}.

## B The Kidney Function Example

Recall the kidney function example discussed in Section 2. The variables WeightN, BPN and RenalFlowN are the weight, blood pressure, and renal blood flow of the patient at the end of week N (equivalently, at the start of week N + 1). All are measured at the end of each week. The doctor can intervene on one of these variables *using the measured values as context for the intervention*, in order to optimize the kidney function of the patient at the end of week 3 (Kidney3). We model this situation with the causal graph depicted in Figure 4. Making use of Theorem 13, we see that the minimal set of nodes which needs to be tested in this case is {RenalFlow2, Weight2, Weight1, Weight0}.

![11_image_0.png](11_image_0.png)

Figure 4: Causal graph for the kidney function example from Section 2. The doctor can intervene on any node X except Kidney3, making use of the measured variables ZX until (and including) that week, thus including in particular An(X) \ {X}).