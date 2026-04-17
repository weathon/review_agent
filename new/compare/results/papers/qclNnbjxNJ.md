# Characterization And Learning Of Causal Graphs With Latent Confounders And Post- Treatment Selection From Interventional Data

Gongxu Luo1, Loka Li1, Guangyi Chen1,2, Haoyue Dai2**, Kun Zhang**1,2 1 Mohamed bin Zayed University of Artificial Intelligence, 2 Carnegie Mellon University
{gongxu.luo, kun.zhang}@mbzuai.ac.ae

## Abstract

Interventional causal discovery seeks to identify causal relations by leveraging distributional changes introduced by interventions, even in the presence of latent confounders. Beyond the spurious dependencies induced by latent confounders, we highlight a common yet often overlooked challenge in the problem due to posttreatment selection, in which samples are selectively included in datasets after interventions. This fundamental challenge widely exists in biological studies; for example, in gene expression analysis, both observational and interventional samples are retained only if they meet quality control criteria (e.g., highly active cells). Neglecting post-treatment selection may introduce spurious dependencies and distributional changes under interventions, which can mimic causal responses, thereby distorting causal discovery results and challenging existing causal formulations. To address this, we introduce a novel causal formulation that explicitly models post-treatment selection and reveals how its differential reactions to interventions can distinguish causal relations from selection patterns, allowing us to go beyond traditional equivalence classes toward the underlying true causal structure. We then characterize its Markov properties and propose a Fine-grained Interventional equivalence class, named FI-Markov equivalence, represented by a new graphical diagram, F-PAG. Finally, we develop a provably sound and complete algorithm, F-FCI, to identify causal relations, latent confounders, and post-treatment selection up to FI-Markov equivalence, using both observational and interventional data. Experimental results on synthetic and real-world datasets demonstrate that our method recovers causal relations despite the presence of both selection and latent confounders.

## 1 Introduction

Causal discovery from interventional (and observational) data, often referred to as interventional causal discovery, aims to identify causal relations by exploiting distributional changes induced by interventions (Spirtes et al., 2000; Pearl, 2000). Despite progress in interventional causal discovery in handling latent confounders, pre-treatment selection (Dai et al., 2025), and biological constraints (Luo et al., 2025), we highlight a common yet often overlooked problem, post-treatment selection, which refers to the selective inclusion of samples after interventions (Heckman, 1978).

For example, in gene perturbation studies, only perturbed cells (intervention) that pass the quality control (selection) are profiled (Norman et al., 2019). In Clinical Trial Per-Protocol Analysis, only participants completing over 80% of scheduled visits (e.g., up to week 12) are included in the final analysis (Detry & Lewis, 2014). Failure to account for post-treatment selection introduces spurious dependencies and intervention-driven distributional changes that mimic causal responses, thereby leading to incorrect statistical inference and challenging existing interventional causal formulations.

Specifically, existing interventional formulations neither distinguish causal relations from posttreatment selection nor detect where the selection is present. Mainstream frameworks identify causal relations and characterize interventional Markov equivalence classes (ECs) on DAGs by exploiting a cross-intervention pattern: after interventions on the cause, marginal distribution p(effect) changes, and conditional distribution p(effect|cause) remains (Tian & Pearl, 2001; Hauser & Buhlmann, 2012; ¨ 2015). When latent confounders are present, variations in p(effect|cause) are further utilized to 1

![1_image_0.png](1_image_0.png)

Figure 1: Motivation examples. (a) & (b) exhibit same dependence with tails from X1 and arrowheads into X2, regardless of direct causation; (c) & (d) exhibit same dependence with tails on both X1 and X2, regardless of direct selection. Existing methods cannot distinguish these cases, whereas ours can.

characterize the interventional ECs involving latent confounders (Eaton & Murphy, 2007; Ghassami et al., 2017; Kocaoglu et al., 2019; Zhou et al., 2025). However, post-treatment selection is nonidentifiable within these frameworks because it yields the same pattern, variant p(effect) and invariant p(effect|cause) before and after the intervention, as causal relations. For example, under posttreatment selection, Figure 1(a) exhibits the same pattern (variant p(X2) and invariant p(X2 | X1) after intervening on X1) as (b). Subsequently, current frameworks place (a) and (b) in the same EC (same representation), regardless of whether a direct causal link exists between X1 and X2, thereby failing to identify causal relations from post-treatment selection. An analogous non-identifiability arises for direct selection, as illustrated in Figure 1(c) and (d). This representational gap challenges existing frameworks and motivates a new formulation that explicitly models post-treatment selection. In this paper, we examine the causal structure among intervened variables in the general setting involving latent confounders and selection bias, explicitly handling post-treatment selection without imposing graphical or parametric assumptions. First, we demonstrate how causal relations, latent confounders, and post-treatment selection differ in structural symmetries (e.g., selection structure with both tails on endpoints, while causation is not) and distributions (variability and invariance) after intervention, which are characterized by conditional independence (CI) patterns. Second, building on these observations, we propose a Fine-grained Interventional equivalence class (e.g.,
distinguishing Figure 1(a) from (b), and (c) from (d)), named FI-Markov equivalence, and provide detailed characterizations. In graphical representation, partial ancestral graph (PAG) edges encode ECs with a broad range of possible structures and thus prevent the unique graphical representation. To obtain a more concise and expressive graphical representation for FI-Markov equivalence, we introduce F-PAG, an extension of the PAG diagram that incorporates novel edge types. Third, we present a sound and complete algorithm F-FCI for recovering the FI-Markov equivalence class. Contributions. In this paper, we focus on a fundamental yet largely overlooked problem, the posttreatment selection that lies beyond the scope of existing interventional causal discovery frameworks. First, we introduce a causal formulation that models post-treatment selection in the presence of latent confounders, and we define the novel FI-Markov equivalence and F-PAG accordingly. Second, building on this formulation, we develop a new algorithm F-FCI, which integrates intervention-based CI patterns with tailored orientation rules. Theoretically, we prove its soundness and completeness. Third, we validate our approach on both synthetic and real-world datasets, demonstrating its effectiveness. Collectively, these contributions provide a principled framework for distinguishing post-treatment selection from true causal relations, thereby broadening the scope of interventional causal discovery.

## 2 Preliminaries And Motivation

In this section, we first introduce the graphical causal model that involves both latent confounders and selection bias (§ 2.1). We then review the standard paradigm for interventional causal discovery (details in Appendix C) and demonstrate why it fails to handle post-treatment selection (§ 2.2).

## 2.1 Graphical Causal Model With Latent Confounders And Selection Bias

We begin with the general problem setup: a DAG with latent confounders and selection bias. Let the DAG G on vertices with index [N] := {1, · · · , N} encode the structure of the underlying causal model where vertices correspond to observed random variables X = (Xi)
N
i=1. For any subset A ⊂ [N], let XA := (Xi)i∈A and by convention X∅ ≡ 0. Apart from the observed ones, L = {Li}
R i=1 accounts for the confounders that affect X but remain unobserved (latent confounders),
and the exogenous selection variable S = {Si}
T i=1 generally represent both *pre-treatment selection*
(preferential inclusion of samples before intervention) and *post-treatment selection* (arising after intervention) (Heckman, 1978; Elwert & Winship, 2014). In this paper, we specialize in

![2_image_0.png](2_image_0.png)

post-treatment selection and assume selection works on at least two observed variables. Throughout, analyses are conducted conditional on S = 1 (i.e., within the selected sample).

To represent the general graph with latent confounders and selection bias, the ancestral graph is defined by a mixed graph without direct and indirect cycles (detailed in Definition 8). To investigate the learnability of graphical models and the information-theoretic limits of the CI test on observational data, the Markov properties of ancestral graphs are examined. Analogous to the d-separation (Definition 7) criterion used for DAGs, the m-separation (Definition 9) blocks the paths in ancestral graphs. Under the pairwise Markov property, in which the absence of edges reflects conditional independence, the ancestral graph that satisfies this property, a.k.a. maximality assumption (Definition 10), is the Maximal Ancestral Graph (MAG). Given that the global, local, and pairwise Markov properties enable the recovery of causal structures via CI tests, under the representation of the MAG diagram, different graphs that entail the same m-separation form the Markov equivalence class (Definition 11).

## 2.2 Limitations Of Existing Interventional Causal Discovery Paradigm Under Post-Treatment Selection

Interventional causal discovery aims to learn the structure of G from data collected under multiple
(hard and soft) intervention settings, each with an *intervention target* I ⊂ [N], meaning variables XI
are intervened on. Let I = {I
(0), I (1)*, . . . , I* (K)} denote the collection of intervention targets across K interventions, and {p
(0), p(1)*, . . . , p*(K)} indicate the corresponding *interventional distributions* over X. We assume throughout I
(0) = ∅, i.e., the pure observational data is available.

Hard interventions remove all incoming edges to the vertices in the intervention target I
(k)in G while all other edges remain. Soft interventions do not break any arrows incident on the intervention target; instead, they only change the conditional distribution (Eberhardt & Scheines, 2007). Rather than analyzing each interventional setting separately, a more effective approach is to exploit changes and invariances across settings: intervening on a cause alters the marginal distribution p(effect), while the conditional p(effect|cause) remains invariant. Conversely, intervening on an effect leaves p(cause)
unchanged, but p(cause|effect) changes (Hoover, 1990; Tian & Pearl, 2001). Such invariance is exploited in the *invariance causal inference framework* (Meinshausen et al., 2016; Ghassami et al., 2017) and has been extended to settings with latent confounders (Jaber et al., 2020). To formally exploit such invariance analysis and model "the action of changing targets", Newey &
Powell (2003); Korb et al. (2004) introduce the *augmented DAG*, denoted by aug(G, I), which, as shown in Figure 2(a), extends the original G by adding exogenous binary vertices ψ = {ψI(k) }
K k=1 as *intervention indicators*, each pointing to its target I
(k). Whether the k-th intervention alters a marginal density, i.e., p
(0)(XA) ̸= p
(k)(XA) or equivalently p(XA | ψI
(k) = 0) ̸= p(XA | ψI
(k) = 1), is then nonparametrically represented by the CI relation ψI
(k) ̸⊥⊥ XA, and graphically represented by the *d-separation* ψI
(k) ̸⊥⊥d XA in aug(G, I), where ⊥⊥d denotes d-separation and
̸⊥⊥d d-connection. Moreover, latent confounders can also be incorporated into augmented DAGs, as shown in Figure 2(b). The invariance in marginal distributions (p
(0)(X2) = p
(1)(X2) represented by ψ1 ⊥⊥ X2) and variability in conditional distributions (p
(0)(X2|X1) ̸= p
(1)(X2|X1) represented by ψ1 ̸⊥⊥ X2|X1) after intervention help distinguish latent confounders from causal relations. Causal discovery algorithms like PC (Spirtes et al., 2000) and FCI (Spirtes et al., 2000; Zhang, 2008b) have been applied in this context (Zhang, 2008a; Huang et al., 2020; Magliacane et al., 2016; Kocaoglu et al., 2019), and *augmented MAG* have been developed as the corresponding graphical representation. Building on the established framework of interventional causal discovery, when selection appears after intervention, the post-treatment selection induces changes in the marginal distribution p(effect) while keeping invariant in the conditional distribution p(effect|cause), as illustrated in Figure 2(c): p
(1)(X2 | S = 1) ̸= p
(0)(X2 | S = 1) and p
(1)(X2 | X1, S = 1) = p
(0)(X2 | X1, S = 1) with examples in (d) and (e). Although post-treatment selection can be represented within the augmented framework, its invariant and variant characteristics are indistinguishable from those of causal relations, rendering it non-identifiable as discussed in Figure 1. This motivates a new formulation that models post-treatment selection and identifies true causal relations.

## 3 New Formulation: Intervention Meets Post-Treatment Selection

Based on the exploration of the interventional causal discovery paradigm, in this section, we extend the paradigm to design a new formulation for post-treatment selection in the presence of latent confounders (§ 3.1), characterize the Markov properties (§ 3.2), and provide the graphical criteria for determining whether two augmented DAGs are Markov equivalent given the same interventions (§ 3.3).

## 3.1 Modeling Post-Treatment Selection

The first step is to model the post-treatment selection explicitly. Since the variant and invariant characteristics are consistent with the Markov assumption, post-treatment selection can be naturally modeled within the augmented DAG (see § 2.2) by adding a selection variable S. Accordingly, we adopt the augmented DAG to coherently unify observational and interventional data by introducing an intervention indicator ψ. Under this model, the joint distribution over the observed variables X in the k-th intervention, conditioning on post-treatment selection denoted by p
(k)
s (X), factorizes as

$$p_{*}^{(k)}(X)=\prod_{\{i|\{i\}\subset I^{(k)}\}}p^{(k)}(X_{i}|\hat{X}_{pa_{\mathcal{G}}(i)},S=1)\prod_{\{j|\{j\}\subset I^{(k)}}}p^{(0)}(X_{j}|\hat{X}_{pa_{\mathcal{G}}(j)},S=1),\tag{1}$$

where XˆpaG (i) ⊂ X ∪ L indicates the parents of Xi, and S = 1 indicates the presence of post-treatment selection. To represent the details of the structural causal model, the graph involving S and L is represented by the augmented DAG, which is redefined as follows (details in Definition 12). Definition 1 (**Augmented DAG**). For a DAG G over X ∪ L ∪ S and intervention target I ⊂ [N],
the augmented graph AugI(G) is a DAG with vertices ψ ∪ X ∪ L ∪ S ∪ ϵ, where: ψ = {ψI
(k) }
K k=1 is a set of exogenous binary indicators for the representation of marginal changes between two environments (observation-intervention or intervention-intervention), pointing to the corresponding intervened variable XI
(k) ; ϵ is exogenous noise for variables, whether observed or hidden. 

An illustration of the augmented DAG is shown in Figure 3, depicting the data generation process. In AugI(G), only X and ψ are measurable, forming the basis of the representation of different environments, such as observational data p(X|ψ = 0, S = 1) and interventional data p(X|ψ = 1, S = 1). S = 1 is conditioned on, meaning all individuals, whether observed or intervened, are selected at the outset. Moreover, following the diagram of MAG for marginalized representation (Zhang, 2008b), each Augmented DAG can also be formally represented by the corresponding Augmented MAG.

\# $ \# % &
! 1  2  3 "
Figure 3: Illustration of a structural causal model (SCM) represented by an augmented DAG.

## 3.2 Characterizing Markov Properties

Our ultimate goal is to learn the causal structure from both observational and interventional data.

On the rationale of modeling post-treatment selection and marginal changes between observational and interventional data using the Augmented DAG AugI(G) in §3.1, the standard Markov properties, i.e., the global Markov property (formulated via d-separation) and the local Markov property (each node is conditionally independent of its non-descendants given its parents), hold exactly as they do in conventional DAGs. These properties provide the theoretical foundation for using CI tests in structure learning and offer an information upper bound for the CI implementations.

![4_image_0.png](4_image_0.png)

By leveraging the Markov properties, CI tests can recover causal structure from data. In particular, three classes of statistical signals are informative: 1) **Interventional distribution changes:** Variability in marginal observational and interventional distributions manifests as conditional dependencies between the intervention indicator ψ and affected variables X. For example, in Figure 4(a), ψ1 ̸⊥⊥ X2 indicates that perturbing X1 propagates a distributional change to X2. 2) **Invariant relations:**
Equality of conditional distributions across observational and interventional data signals invariance.

Specifically, ψ1 ⊥⊥ X2 | X1 indicates the invariance of p
(0)(X2 | X1) = p
(1)(X2 | X1), I
(1) = {1}.

3) **Structural symmetries:** Certain structures exhibit characteristic symmetry in their CI patterns.

For instance, a symmetric selection shown in Figure 4(e) yields ψ1 ⊥⊥ X2 | X1, ψ2 ⊥⊥ X1 | X2, ψ1 ̸⊥⊥ X2, ψ2 ̸⊥⊥ X1. Below, we formally define these relations implied by the model.

Theorem 1 (CI and invariance implementation). *For positive interventional distributions* p
(k)(X) and observational distribution p
(0)(X) generated from the DAG G *in the presence of latent confounders* L and selection S *with intervention targets* {I
(k)}k∈{0}∪[K], let {AugI
(k) (G)}k∈{0}∪[K] be the corresponding augmented DAGs. For any disjoint A, B, C ∈ [N], the following statement holds:
- For any k ∈ {0} ∪ [K], if XA ⊥⊥d XB|{XC , S} holds in AugI
(k) (G), then XA ⊥⊥ XB|{XC , S}
in p
(k).

- For any k ∈ [K]*, if* ψI
(k) ⊥⊥d XA|XB holds in AugI
(k) (G)*, then* p
(k)(XA|XB) = p
(0)(XA|XB).

- For any k ∈ [K]*, if* ψI
(k) ̸⊥⊥d XA | ∅ holds in AugI
(k) (G)*, then* p
(k)(XA) ̸= p
(0)(XA).

Remark 1. ψI
(k) generally marginalizes changes between different environments. The difference between the two interventions on the same I
(k)also follows the statements in Theorem 1, where the hard-hard intervention changes the causal diagram, providing additional information that is only used to identify the structures of unblocked paths. Theorem 1 shows that invariance and variability in marginal and conditional distributions are implied by graphical conditions, namely d-separation among ψ ∪ X|S = 1 in augmented DAGs. Previous studies leveraged this statistical information to distinguish causal effects from associations induced by latent confounders. However, it is known that selection bias also introduces spurious dependencies:
Lemma 1 (Additional dependencies induced by selections). For any DAG on X[N] ∪ L ∪ S*, targets* I ∈ [N], and disjoint A, B, C ∈ [N], if XA ⊥⊥d XB|XC , S holds in the augmented DAG AugI(G), then XA ⊥⊥d XB|XC *holds in the original DAG. The reverse is not necessarily true.* With the characterization of global and local Markov properties ready, differences in graphical properties, such as asymmetry, captured in CI patterns as shown in Figure 4, help distinguish different structures. Specifically, Figure 4(i) shows that (a) and (e) exhibit different CI patterns, with (e) being symmetric. We further observe that although (a) and (b) share the same CI patterns between X1 and X2 regardless of whether a direct causal link exists in (b), they differ in their underlying causal structures. This is due to the Y-structure at X2 forming an unblocked path, which exhibits the same characteristics as causation. Beyond focusing only on cause and effect targets, hard interventions on X3 open the path by blocking the selection effect on the latent confounder L. The variation in two different hard interventions on X3 can be modeled by ψ3 for representation. Then, ψ3 ̸⊥⊥ X2 allows us to distinguish case (b) and assess whether there is a direct causal link between X1 and X2.

Similarly, direct selection can be identified in the same way for (e) and (f). This, in turn, goes beyond traditional ECs and can identify concrete causal structures at the DAG level.

## 3.3 Fi-Markov Equivalence

In § 3.2, we characterized the Markov properties implied by the true model of the data. Now, to identify the true model from data, in this section, we shall understand to what extent the true model is *identifiable*, as different models may share identical CI implications, namely, being Markov equivalent. To characterize the equivalence class under the general setting with MAG representation, the Markov equivalence with corresponding m-separation on observational data is discussed. However, with the help of interventional data, we propose a novel Fine-grained Interventional Markov Equivalence, named FI-Markov equivalence (Definition 2), and characterize the EC under the augmented DAG framework based on the Markov properties. Different from the graphical representation of the structural causal model, the representation of learned ECs is only over observations X. We subsequently extend the Partial Ancestral Graph (PAG) framework (Definition 4) with novel edges for the representation of FI-Markov ECs, which are more informative compared with PAG. Because the Markov property allows us to distinguish structures that existing formulations cannot, for instance, whether a direct causal link exists in Figure 4(b), we define a new FI-Markov equivalence under the augmented DAG framework for a more precise structural representation. Two different augmented DAGs with the same intervention targets can entail the same CI patterns in the data. Formally, Definition 2 (FI**-Markov equivalence**). Two Augmented DAGs, AugI(G1) and AugI(G2), are FI- Markov equivalent with the same intervention targets I, if and only if they have the same d-separation (the same skeleton and v-structure in the description of the corresponding MAG representation) among X[N]\I, and the same CI patterns between ψ and any intervened variable Xi ∈ XSI.

## 3.3.1 Graphical Criteria For Fi-Markov Equivalence

When learning the EC based on Markov properties, causal discovery methods only recover whether the unblocked paths have a tail or an arrowhead at each endpoint over X[N], not the full structure with L and S (Kocaoglu et al., 2017). These unblocked paths are named inducing paths, defined as follows:
Definition 3 (**Inducing path**). In augmented DAGs, Xi, Xj are any two vertices, and L, S are disjoint sets of vertices not containing Xi, Xj . A path p between Xi, Xj is called an inducing path relative to ⟨*L, S*⟩ if every non-endpoint vertex on p is either in L or a collider, and every collider on p is an ancestor of either Xi, Xj , or a member of S.

Example. In Figure 4(b), the path between X1 and X2 is the inducing path with unblocked X3.

To characterize the ECs via graphical features over X[N], we still need to borrow the MAG representation for marginalization. Following the procedure of MAG construction introduced in (Zhang, 2008b),
every augmented DAG corresponds to an augmented MAG M(AugI (G)) in graphical representation. Then, we show the rules to construct M(AugI (G)) by presenting the following lemmas. Lemma 2 (When are two variables dependent in observational data?). For any i, j ∈ [N], Xi and Xj are adjacent in M(AugI (G)), if and only if Xi and Xj have at least one inducing path in AugI (G).

The adjacencies in Lemma 2 capture all dependencies induced by inducing paths, forming the foundation for constructing the skeleton. Then, interventional data help further identify the structures:
Lemma 3 (When does intervention always alter marginal distribution?). For any *i, j* ∈ [N], i ∈ I, Xi and Xj are adjacent in M(AugI (G)) with a tail at Xi*, if and only if every inducing path between* Xi and Xj begins with a tail in AugI(G), i.e., Xiis the ancestor of Xj *or is ancestrally selected.*
With Lemma 3, the variant marginal distribution characterizes the ECs of inducing paths starting with a tail. The variability in the conditional distribution is utilized to characterize the ECs as follows:
Lemma 4 (When does intervention always alter conditional distribution?). For any i, j ∈ [N], i ∈ I, Xi and Xj are adjacent in M(AugI (G)) with an arrowhead at Xi, if and only if every inducing path between Xi and Xj begins with an arrowhead in AugI(G), i.e., Xi*is a descendant of* Xj or L.

With the foundational graphical criteria that are consistent with the MAG construction ready, the graphical criteria for the FI-Markov equivalence learned from data are as follows:
Theorem 2 (Graphical criteria for FI-Markov equivalence). Two augmented DAGs AugI(G1) =
(ψ ∪ X ∪ L ∪ S, E) and AugI(G2) = (ψ ∪ X ∪ L
∗ ∪ S
∗, E∗) are FI-Markov equivalent for a set of

![6_image_0.png](6_image_0.png)

3.3.2 F-PAG: GRAPHICAL REPRESENTATION FOR FI-MARKOV EQUIVALENCE
Based on the Markov properties of the augmented DAG in the presence of latent confounders and post-treatment selection, the CI patterns precisely characterize when two such distributions are Markov equivalent in Section 3.3.1. This alignment with our learning objective allows us to recover the EC directly from data. For the graphical representation of the ECs, we follow the conventional approach of using the Partial Ancestral Graph (PAG), defined as follows: Definition 4 (**Partial ancestral graph**). Let [M] be the Markov equivalence class of a MAG M. A partial ancestral graph (PAG) for [M] is a graph P with possibly six kinds of edges: , →,↔ , ◦ , ◦ ◦, ◦→, such that (1) P has the same adjacencies as M does; (2) every non-circle mark in P is an invariant mark in [M]. If it is furthermore true that (3) every circle in P corresponds to a variant (indeterminate) mark in [M], P is called the maximally informative PAG. Although PAG is a general framework for the graphical representation of DAGs under selection bias and latent confounders, it is designed for ECs, which are maximally informative in CI relations from observational data (v-structure induced independence). This limitation usually results in dependencies induced by broad inducing paths represented by ◦ ◦. With the discussed characterizations of Markov properties in Section 3.2, PAG is too broad to represent FI**-Markov equivalence**. For example, in Figure 5(b) and (c), the presence or absence of a causal link between X1 and X2 results in the same PAG. However, Figure 5(c) can be distinguished with interventional data as discussed in Figure 4
(b). To describe the reduced FI-Markov equivalence, we proposed the F-PAG defined as follows: Definition 5 (F**-Partial Ancestral Graph**). A F-Partial Ancestral Graph, denoted as Gp, is a graphical representation derived from a DAG with latent and selection variables. It captures conditional independence relationships and consists of four types of marks (tail , arrowhead >, square □, and circle ◦), and eight types of edges ( 
▲→,
▲
, , →,↔, □ , □ □, □→, ◦ , ◦ ◦, ◦→).

The mark □ denotes a node with at least one tail and at least one arrowhead, 
▲→ (Figure 5(c)) and
▲
(Figure 5(e)) represent inducing paths that have the same CI patterns with →, , but without a direct causal link and selection separately in between. These two types of inducing paths can be identified only through the graphical conditions involving **Type I** inducing nodes defined as follows: Definition 6 (**Inducing Node**). In an F-PAG, the nodes are referred to as inducing nodes if and only if the non-endpoint nodes on the inducing path are characterized either by an incoming arrowhead into a square (→ □ **Type I**) or by adjacent two squares (□□ **Type II**).

For example, in Figure 5(b), non-endpoint node X3 is a **Type I** inducing node. With the advanced graphical representation in place, FI-Markov equivalence can be expressed more clearly, allowing us to distinguish whether the observed dependence arises from genuine causal relations (a) and (b), direct selection (d), or from equivalent inducing paths (c) and (e) as shown in Figure 5.

## 4 Algorithm: F-Fci

In this section, we propose a novel Algorithm 1, named Fine-graind FCI (F-FCI). Using Markov properties of the augmented DAG in Section 3, F-FCI learns causal relations, latent confounders, Algorithm 1: F-FCI: Algorithm for learning F-PAG
Input: Observational and interventional data {p
(k)}
K
k=0 over X[N] with interventional targets I.

Output: A fine-grained partially ancestral graph (F-PAG Gp) over vertices X.

Step 1: Get skeleton from pure observational data. G
(0)
p ← FCIske(p
(0))).

Step 2: Get F-PAG orientation over XSI **from interventional data. for** 1 ≤ *i < j* ≤ K do Step 2.1: Capture CI patterns between ψ and XSI. foreach condition set C ⊆ {Xn : Xn ∈ *AllP aths*(G
(0)
p , XI(i) , XI(j) )} \ {XI(i) , XI(j) } do CIs = { CI (ψI
(i) , XI
(j) | C), CI (ψI
(i) , XI
(j) | XI
(i) , C), CI (ψI
(j) , XI
(i) | C),
CI (ψI
(j) , XI
(i) | XI
(j) , C)}, If no more paths can be blocked **then** break; Step 2.2: Orient the skeleton between XI
(i) and XI
(j) .

if CIs == (̸⊥⊥, ⊥⊥, ⊥⊥, ̸⊥⊥) **then** Orient XI
(i) → XI
(j)
if CIs == (⊥⊥, ̸⊥⊥, ⊥⊥, ̸⊥⊥) **then** Orient XI
(i) ↔ XI
(j)
if CIs == (̸⊥⊥, ̸⊥⊥, ⊥⊥, ̸⊥⊥) **then** Orient XI
(i) ◦→ XI
(j)
if CIs == (̸⊥⊥, ⊥⊥, ̸⊥⊥, ⊥⊥) **then** Orient XI
(i) XI
(j)
if CIs == (̸⊥⊥, ⊥⊥, ̸⊥⊥, ̸⊥⊥) **then** Orient XI
(i) □XI
(j)
if CIs == (̸⊥⊥, ̸⊥⊥, ̸⊥⊥, ̸⊥⊥) **then** Orient XI
(i) □ □XI
(j)
Step 2.3: Refine the orientation. Identify causal relations in between for XI
(i) ◦→ XI
(j) .

foreach *inducing path between the node pairs with interventional data (*XI
(i) , XI
(j) )
marked with → or from *Step 2.2* do Detect if the path has non-endpoints vertex and **Type I** inducing nodes.

If ∃ **Type I** inducing node Xn with XI
(i) → Xn□ XI
(j) **then** CI (ψn, XI
(i) ).

If ψn ⊥⊥ XI
(i) **then** update XI
(i) XI
(j) to XI
(i)
▲ XI
(j) .

If ∃ **Type I** inducing node Xn with XI
(i) □Xn ↔ XI
(j) **then** CI (ψn, XI
(j) ).

If ψn ⊥⊥ XI
(j) **then** update XI
(i) → XI
(j) to XI
(i)
▲→ XI
(j) .

Step 2.4: Update the F**-PAG.** G
(k)
p ← G(k−1)
p Step 3: Get F**-PAG orientation over** X[N]/SI. Apply the orientation rules of FCI among X[N]/SI and apply the invariance rules in Theorem 1 (the see-see, *do-see*, and *do-do* rules in (Kocaoglu et al., 2019)) between XSI and X[N]/SI in G
(K)
p . Update Gp ← G(K)
p return Gp and post-treatment selection up to the FI-Markov equivalence class, from both observational and interventional data. We assume faithfulness, i.e., no CIs beyond those implied by the graph. The first step is to recover the undirected skeleton from observational data p
(0), since it yields the sparsest graph encoding all inducing paths. We then adopt the standard constraint-based skeleton discovery procedure (e.g., as in FCI) under our general graphical assumptions to obtain this skeleton. Step 2 consists of four sub-steps. **Step 2.1** captures CI patterns reflecting marginalized changes between observational p
(0) and interventional data {p
(k)}
K
k=1. In **Step 2.2**, we then leverage the captured CI patterns to orient edges incident to the intervened variables XSI, using the orientation rules summarized in Figure 4. In particular, the rule ◦→ uses ◦ instead of □ because the existence of an inducing path beginning with a tail in between is uncertain when based solely on the marginalized changes between observation and intervention. For example, the structure ψ1 → X1 ← L → X2. When X1 is under selection, one observes ψ1 ̸⊥⊥ X2 regardless of whether X1 is conditioned on. Then, this structure cannot be distinguished from the structure of the latent with causal relation in Figure 4(d), represented by ◦→. To go beyond CI patterns and identify real structure, in **Step 2.3**, F-FCI firstly addresses the uncertainty of ◦→ by blocking selection on latent confounders via marginalized changes from two hard interventions. Then, **Type I** inducing nodes along inducing paths between intervened variables are detected to disambiguate cases that CI patterns of endpoints alone cannot distinguish. This procedure is valid for inducing paths containing more than one non-endpoint vertex and including at least one Type I inducing node. For example, graph (b) in Figure 4 share the identical CI patterns regardless of whether a direct or indirect causal link exists between X1 and X2. By utilizing changes of hard intervention on the **Type I** inducing node X3, we can test for ψ3 ⊥⊥ X2 | S to determine whether a true causal relation exists. Likewise, direct selection (Figure 4(f)) becomes identifiable under the same rationale. Specialized edge marks 
▲→ and ▲are established to represent the inducing paths in Figure 5.

![8_image_0.png](8_image_0.png)

By explicitly orienting edges between intervened node pairs, the core contribution of our algorithm, we subsequently apply the standard FCI orientation rules and rules of invariance to all remaining edges: those between intervened and unintervened nodes, as well as those among unintervened nodes in **Step 3**. The extra orientations recovered from interventional data furnish richer structural information than v-structures identified from only observational data.

Theorem 3 (Soundness of F-FCI). Let Gp be the output F-PAG of Algorithm 1 with oracle CI tests on multi-distribution data {p
(k)}
K
k=0 given by (G, I). Gp *is consistent with augmented DAG*
AugI(G) *in arrowhead, tails, square, and structures of paths* 
▲→,
▲
*among intervened variables.*
Theorem 4 (Completeness of F-FCI). Let Gˆp *be the output of Algorithm 1 with oracle CI tests on* multi-distribution data {p
(k)}
K
k=0 given by (G, I). Each type of substructures represented by tail, arrowhead, square, 
▲→, and ▲*between a pair of intervened nodes in the corresponding augmented* DAG of Gˆp *can be identified by different types of CI patterns.*

## 5 Experiments

In this section, we present empirical studies on simulations and real-world data to demonstrate that F-FCI identifies causal relations in the presence of latent confounders and post-treatment selection.

## 5.1 Simulations

We conduct simulations to validate the effectiveness of our proposed F-FCI. We compare F-FCI against strong baselines in interventional causal discovery, including GIES (Hauser & Buhlmann, ¨
2012), IGSP (Wang et al., 2017), UT-IGSP (Squires et al., 2020), JCI-GSP (Mooij et al., 2020), FCI with interventional data (FCI-interven) (Kocaoglu et al., 2019), and CDIS (Dai et al., 2025). Following the data-generating procedure in Definition 1, we begin by randomly sampling Erdos–R ¨ enyi ´
graphs with an average degree of 2 as the ground truth DAG for {Xi}
N
i=1. We then randomly generate 2 or 3 selection variables, each with two randomly chosen parents from {Xi}
N
i=1, and 2 or 3 latent confounders, each with two randomly chosen children. Finally, we simulate general SEMs Xi =
f(XˆpaG (i)) + ϵi, ϵiis sampled from *Unif*([0, 2] ∪ [2, 4]), and select samples with Pfs(Xi) that fall within a predefined interval, where f and fs are randomly drawn from linear, square, sin and tanh.

To evaluate the effectiveness of F-FCI 1in identifying causal relations despite the presence of latent confounders and post-treatment selection, we report the main Precision and Structural Hamming Distance (SHD) metrics compared with baseline methods in Figure 6 (the F1-score and recall are given in Figure 10 in Appendix D). The experimental results demonstrate that F-FCI outperforms baselines with an average precision of over 5% in most configurations and lower SHD. These observations validate the effectiveness of F-FCI in identifying true causal relations, whereas baselines may infer spurious ones induced by latent confounders and post-treatment selection. Moreover, the robustness of F**-FCI under different noise levels** is evaluated in Figure 12, the **scalability** is evaluated in Figure 11, and **its ability to distinguish post-treatment selection** is assessed in Table 1.

1A Python implementation of F-FCI is available at https://github.com/GongxuLuo/F-FCI.

## 5.2 Real-World Applications

We evaluate the gene regulatory networks (GRNs) of genes using large-scale single-cell gene perturbation data collected from Human Lung Epithelial Cells (HLEC), i.e., Norman datasets (Norman et al., 2019). We report both the regulatory (causal) links and the spurious dependencies induced by post-treatment selection, as identified by F-FCI in Figure 13, and detailed analysis can be found in Appendix D.3. Experimental results are evaluated using prior knowledge provided by Enrichr, a tool that compiles extensive libraries from enrichment experiments (Kuleshov et al., 2016; Xie et al., 2021).

## 6 Conclusion And Limitations

We introduce a fundamental yet underexplored challenge for causal discovery: post-treatment selection, particularly the often-overlooked quality control constraint that shapes dependencies. We show why existing models fail to handle such bias, propose a new formulation to model post-treatment selection, establish criteria for a novel fine-grained interventional Markov equivalence, and define a corresponding graphical representation. Building on this formulation, we develop a sound and complete algorithm, named F-FCI, that uncovers causal relations and post-treatment selection.

Empirical analyses on synthetic and large-scale real-world datasets demonstrate the effectiveness of F-FCI in accurate and robust causal discovery. The identification of direct causal links and selection structures depends critically on the presence of **Type I** inducing nodes. One future direction is how to identify the causal structure along inducing paths composed solely of **Type II** inducing nodes. In addition, as discussed in (Luo et al., 2025), biological constraints filter out cells and introduce extra dependencies; another extension can be how to distinguish biological constraints from post-treatment selection.

## Acknowledgements

We would also like to acknowledge the support from NSF Award No. 2229881, AI Institute for Societal Decision Making (AI-SDM), the National Institutes of Health (NIH) under Contract R01HL159805, and grants from Quris AI, Florin Court Capital, MBZUAI-WIS Joint Program, and the Al Deira Causal Education project.

## References

Ayesha R Ali, Thomas S Richardson, Peter L Spirtes, and Jiji Zhang. Towards characterizing markov equivalence classes for directed acyclic graphs with latent variables. arXiv preprint arXiv:1207.1365, 2012.

Steen A Andersson, David Madigan, and Michael D Perlman. A characterization of markov equivalence classes for acyclic digraphs. *The Annals of Statistics*, 25(2):505–541, 1997.

Laura E Brown, Ioannis Tsamardinos, and Constantin F Aliferis. A comparison of novel and state-ofthe-art polynomial bayesian network learning algorithms. In *AAAI*, volume 2005, pp. 739–745, 2005.

Fuyuan Cao, Yunxia Wang, Kui Yu, and Jiye Liang. Causal discovery from unknown interventional datasets over overlapping variable sets. *IEEE Transactions on Knowledge and Data Engineering*, 2024.

Chandler Squires. causaldag*: creation, manipulation, and learning of causal models*, 2018. URL
https://github.com/uhlerlab/causaldag.

Edward Y Chen, Christopher M Tan, Yan Kou, Qiaonan Duan, Zichen Wang, Gabriela Vaz Meirelles, Neil R Clark, and Avi Ma'ayan. Enrichr: interactive and collaborative html5 gene list enrichment analysis tool. *BMC bioinformatics*, 14(1):1–14, 2013.

David Maxwell Chickering. Optimal structure identification with greedy search. *Journal of machine* learning research, 3(Nov):507–554, 2002.

Tom Claassen and Tom Heskes. Causal discovery in multiple models from different experiments.

Advances in Neural Information Processing Systems, 23, 2010.

Emily Clough and Tanya Barrett. The gene expression omnibus database. In Statistical Genomics:
Methods and Protocols, pp. 93–110. Springer, 2016.

ENCODE Project Consortium et al. An integrated encyclopedia of dna elements in the human genome. *Nature*, 489(7414):57, 2012.

Gregory F Cooper and Changwon Yoo. Causal discovery from a mixture of experimental and observational data. *arXiv preprint arXiv:1301.6686*, 1999.

Justine Creff and Arnaud Besson. Functional versatility of the cdk inhibitor p57kip2. Frontiers in cell and developmental biology, 8:584590, 2020.

Haoyue Dai, Ignavier Ng, Jianle Sun, Zeyu Tang, Gongxu Luo, Xinshuai Dong, Peter Spirtes, and Kun Zhang. When selection meets intervention: Additional complexities in causal discovery. arXiv preprint arXiv:2503.07302, 2025.

Michelle A Detry and Roger J Lewis. The intention-to-treat principle: how to assess the true effect of choosing a medical treatment. *Jama*, 312(1):85–86, 2014.

Daniel Eaton and Kevin Murphy. Exact bayesian structure learning from uncertain interventions. In Artificial intelligence and statistics, pp. 107–114. PMLR, 2007.

Frederick Eberhardt and Richard Scheines. Interventions and causal inference. *Philosophy of science*,
74(5):981–995, 2007.

Wafik S El-Deiry, Takashi Tokino, Victor E Velculescu, Daniel B Levy, Ramon Parsons, Jeffrey M
Trent, David Lin, W Edward Mercer, Kenneth W Kinzler, and Bert Vogelstein. Waf1, a potential mediator of p53 tumor suppression. *Cell*, 75(4):817–825, 1993.

Felix Elwert and Christopher Winship. Endogenous selection bias: The problem of conditioning on a collider variable. *Annual review of sociology*, 40(1):31–53, 2014.

Nir Friedman, Michal Linial, Iftach Nachman, and Dana Pe'er. Using Bayesian networks to analyze expression data. In Proceedings of the fourth Annual International Conference on Computational Molecular Biology, pp. 127–135, 2000.

AmirEmad Ghassami, Saber Salehkaleybar, Negar Kiyavash, and Kun Zhang. Learning causal structures using regression invariance. *Advances in Neural Information Processing Systems*, 30, 2017.

Guoqiang Han, Manman Cui, Pengbo Lu, Tiantian Zhang, Rong Yin, Jin Hu, Jihua Chai, Jing Wang, Kexin Gao, Weidong Liu, et al. Selective translation of nuclear mitochondrial respiratory proteins reprograms succinate metabolism in aml development and chemoresistance. *Cell Stem Cell*, 31
(12):1777–1793, 2024.

J Wade Harper, Guy R Adami, Nan Wei, Khandan Keyomarsi, and Stephen J Elledge. The p21 cdk-interacting protein cip1 is a potent inhibitor of g1 cyclin-dependent kinases. *Cell*, 75(4): 805–816, 1993.

Alain Hauser and Peter Buhlmann. Characterization and greedy learning of interventional markov ¨
equivalence classes of directed acyclic graphs. *The Journal of Machine Learning Research*, 13(1): 2409–2464, 2012.

Alain Hauser and Peter Buhlmann. Jointly interventional and observational data: estimation of ¨
interventional markov equivalence classes of directed acyclic graphs. *Journal of the Royal* Statistical Society Series B: Statistical Methodology, 77(1):291–318, 2015.

James J. Heckman. Sample selection bias as a specification error. *Econometrica*, pp. 153, Dec 1978.

doi: 10.2307/1912352.

Kevin D Hoover. The logic of causal inference: Econometrics and the conditional analysis of causation. *Economics & Philosophy*, 6(2):207–234, 1990.

Patrik Hoyer, Dominik Janzing, Joris M Mooij, Jonas Peters, and Bernhard Scholkopf. Nonlinear ¨
causal discovery with additive noise models. *Advances in neural information processing systems*, 21, 2008.

Biwei Huang, Kun Zhang, Jiji Zhang, Joseph Ramsey, Ruben Sanchez-Romero, Clark Glymour, and Bernhard Scholkopf. Causal discovery from heterogeneous/nonstationary data. ¨ *Journal of* Machine Learning Research, 21(89):1–53, 2020.

Antti Hyttinen, Patrik O Hoyer, Frederick Eberhardt, and Matti Jarvisalo. Discovering cyclic causal models with latent variables: A general sat-based procedure. *arXiv preprint arXiv:1309.6836*, 2013.

Amin Jaber, Murat Kocaoglu, Karthikeyan Shanmugam, and Elias Bareinboim. Causal discovery from soft interventions with unknown targets: Characterization and learning. *Advances in neural* information processing systems, 33:9551–9561, 2020.

Alexandra B Keenan, Denis Torre, Alexander Lachmann, Ariel K Leong, Megan L Wojciechowicz, Vivian Utti, Kathleen M Jagodnik, Eryk Kropiwnicki, Zichen Wang, and Avi Ma'ayan. Chea3: transcription factor enrichment analysis by orthogonal omics integration. *Nucleic acids research*,
47(W1):W212–W224, 2019.

Murat Kocaoglu, Alex Dimakis, and Sriram Vishwanath. Cost-optimal learning of causal graphs. In International Conference on Machine Learning, pp. 1875–1884. PMLR, 2017.

Murat Kocaoglu, Amin Jaber, Karthikeyan Shanmugam, and Elias Bareinboim. Characterization and learning of causal graphs with latent variables from soft interventions. Advances in Neural Information Processing Systems, 32, 2019.

Kevin B Korb, Lucas R Hope, Ann E Nicholson, and Karl Axnick. Varieties of causal intervention.

In *PRICAI 2004: Trends in Artificial Intelligence: 8th Pacific Rim International Conference on* Artificial Intelligence, Auckland, New Zealand, August 9-13, 2004. Proceedings 8, pp. 322–331.

Springer, 2004.

Maxim V Kuleshov, Matthew R Jones, Andrew D Rouillard, Nicolas F Fernandez, Qiaonan Duan, Zichen Wang, Simon Koplev, Sherry L Jenkins, Kathleen M Jagodnik, Alexander Lachmann, et al. Enrichr: a comprehensive gene set enrichment analysis web server 2016 update. *Nucleic acids* research, 44(W1):W90–W97, 2016.

Alexander Lachmann, Denis Torre, Alexandra B Keenan, Kathleen M Jagodnik, Hoyjin J Lee, Lily Wang, Moshe C Silverstein, and Avi Ma'ayan. Massive mining of publicly available rna-seq data from human and mouse. *Nature communications*, 9(1):1366, 2018.

Adam Li, Amin Jaber, and Elias Bareinboim. Causal discovery from observational and interventional data across multiple environments. *Advances in Neural Information Processing Systems*, 36:
16942–16956, 2023.

Loka Li, Haoyue Dai, Hanin Al Ghothani, Biwei Huang, Jiji Zhang, Shahar Harel, Isaac Bentwich, Guangyi Chen, and Kun Zhang. On causal discovery in the presence of deterministic relations. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024a.

Loka Li, Ignavier Ng, Gongxu Luo, Biwei Huang, Guangyi Chen, Tongliang Liu, Bin Gu, and Kun Zhang. Federated causal discovery from heterogeneous data. *arXiv preprint arXiv:2402.13241*, 2024b.

Gongxu Luo, Haoyue Dai, Loka Li, Chengqian Gao, Boyang Sun, and Kun Zhang. Gene regulatory network inference in the presence of selection bias and latent confounders. Advances in Neural Information Processing Systems, 2025.

Sara Magliacane, Tom Claassen, and Joris M Mooij. Ancestral causal inference. Advances in Neural Information Processing Systems, 29, 2016.

Shuhei Matsuoka, Michael C Edwards, Chang Bai, Susan Parker, Pumin Zhang, Antonio Baldini, J Wade Harper, and Stephen J Elledge. p57kip2, a structurally distinct member of the p21cip1 cdk inhibitor family, is a candidate tumor suppressor gene. *Genes & development*, 9(6):650–662, 1995.