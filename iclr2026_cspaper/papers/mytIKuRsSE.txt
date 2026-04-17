# Learning With Dual-Level Noisy Correspon- Dence For Multi-Modal Entity Alignment

Haobin Li1 Yijie Lin1 Peng Hu1 Mouxing Yang1∗ **Xi Peng**1,2∗
College of Computer Science, Sichuan University, China.1 National Key Laboratory of Fundamental Algorithms and Models for Engineering Numerical Simulation, Sichuan University, China.2
{haobinli.gm, linyijie.gm, penghu.ml, yangmouxing, pengx.gm}@gmail.com

## Abstract

Multi-modal entity alignment (MMEA) aims to identify equivalent entities across heterogeneous multi-modal knowledge graphs (MMKGs), where each entity is described by attributes from various modalities. Existing methods typically assume that both intra-entity and inter-graph correspondences are faultless, which is often violated in real-world MMKGs due to the reliance on expert annotations. In this paper, we reveal and study a highly practical yet under-explored problem in MMEA, termed Dual-level Noisy Correspondence (DNC). DNC refers to misalignments in both intra-entity (entity-attribute) and inter-graph (entity-entity and attributeattribute) correspondences. To address the DNC problem, we propose a robust MMEA framework termed RULE. RULE first estimates the reliability of both intra-entity and inter-graph correspondences via a dedicated two-fold principle. Leveraging the estimated reliabilities, RULE mitigates the negative impact of intraentity noise during attribute fusion and prevents overfitting to noisy inter-graph correspondences during inter-graph discrepancy elimination. Beyond the trainingtime designs, RULE further incorporates a correspondence reasoning module that uncovers the underlying attribute-attribute connection across graphs, guaranteeing more accurate equivalent entity identification. Extensive experiments on five benchmarks verify the effectiveness of our method against DNC compared with seven state-of-the-art methods. Code is available at https://github.com/XLearning- SCU/2026-ICLR-RULE.

## 1 Introduction

Multi-Modal Entity Alignment (Liu et al., 2021; Li et al., 2023) (MMEA) aims to identify equivalent entities across different Multi-modal Knowledge Graphs (MMKGs) (Liu et al., 2019; Zhu et al.,
2022), where each entity is associated with attributes of various modalities (*e.g.*, structural triples and images). Due to the heterogeneity of attributes from different modalities and graphs from different sources (e.g., Wikidata (Vrandeciˇ c & Krötzsch, 2014) and YAGO (Suchanek et al., 2007)), ´ the key challenge of MMEA is to learn a comprehensive representation for each entity with its respective attributes while eliminating the cross-graph discrepancy. To this end, existing methods usually conduct multi-modal fusion for attributes within the same entity based on the intra-entity correspondences (i.e., entity-attribute pairs), while performing cross-graph alignment by resorting to the inter-graph correspondences (i.e., entity-entity pairs and attribute-attribute pairs). Despite significant efforts in intra-entity attribute fusion (Chen et al., 2023a; Huang et al., 2024a) and inter-graph discrepancy elimination (Xu et al., 2023; Guo et al., 2021), existing MMEA methods heavily rely on the assumption of faultless intra-entity and inter-graph correspondences. However, as shown in Fig. 1(a), the assumption is daunting and even impossible to satisfy, leading to the Noisy Correspondence (NC) problem at dual levels. On the one hand, as the MMKG construction requires expert knowledge, it is inevitable to wrongly associate some entities with irrelevant attributes, resulting in intra-entity NC. For instance, image of "Elvis Tsui" is incorrectly associated with entity "Jason Momoa" because of the visual resemblance. On the other hand, due to the inherent complexities in attribute and entity association, accurately associating all the inter-graph entities and
∗Corresponding author.

1

![1_image_0.png](1_image_0.png)

their corresponding attributes is impractical, leading to inter-graph NC. For example, movie entity
"Mr. & Mrs. Smith" is mistakenly associated with real-life couple "Will Smith and Mrs. Smith".

According to the statistics in Appendix B, real-world benchmarks always contain numerous NC (e.g., over 50% in ICEWS benchmarks). As shown in Fig. 1(b), NC would not only undermine the fusion of within-entity attributes but also mislead the inter-graph alignment, both of which significantly degrade the performance. Based on the above observations, we reveal a new problem for MMEA, termed Dual-level Noisy Correspondence (DNC). To conquer the DNC problem, we propose a novel method, dubbed dually RobUst LEarning (RULE), for achieving robust MMEA against DNC. Specifically, RULE first estimates the reliability of both the intra-entity and inter-graph correspondences by resorting to a dedicatedly-designed two-fold principle and then divides the entity-attribute, entity-entity, and attribute-attribute pairs into different groups. Based on the estimated reliabilities and division results, RULE alleviates the negative impact of intra-entity NC during intra-entity attribute fusion, while preventing the model from overfitting the inter-graph NC during inter-graph discrepancy elimination. Beyond the training-time designs, RULE further incorporates a novel correspondence reasoning module to enhance the test-time robustness. In brief, this module performs deep reasoning to uncover the underlying attribute-attribute connections across graphs, thus preventing seemingly dissimilar but inherently identical attributes from being neglected (as shown in Fig. 1(c)) and guaranteeing more accurate equivalent entity identification during inference. In summary, the major contributions and novelties of this work are given as follows.

- We reveal and study a novel and practical problem in MMEA, termed Dual-level Noisy Correspondence (DNC). In brief, DNC refers to the noisy correspondence rooted in the intra-entity (entity-attribute) pairs and inter-graph (entity-entity, attribute-attribute) pairs. We empirically demonstrate that DNC not only undermines multi-modal attribute fusion but also misleads the inter-graph alignment, leading to significant performance degradation for existing MMEA methods.

- To achieve robust MMEA against the DNC problem, we propose a novel method termed RULE,
which estimates the reliability of both the intra-entity and inter-graph correspondences with a dedicatedly-designed two-fold principle and accordingly mitigates the negative impact of DNC during the multi-modal attribute fusion and inter-graph alignment processes.

- During inference, RULE employs a novel correspondence reasoning module to uncover inherentlyidentical attributes and accordingly achieve more precise cross-graph equivalent entity identification. To the best of our knowledge, this could be one of the first methods to enhance test-time robustness for the MMEA task.

## 2 Method

In this section, we introduce the proposed RULE for tackling the DNC problem. In Section 2.1, we present the formal definition of the MMEA task and the DNC problem. In Section 2.2, we elaborate on the two-fold principle for the reliability estimation and pair division. In Section 2.3-2.4, we introduce the robust attribute fusion and robust discrepancy elimination modules. In Section 2.5, we design a test-time correspondence reasoning module to uncover underlying connections between inter-graph attributes, facilitating the equivalent entity identification.

## 2.1 Problem Formulation

Given two heterogeneous multi-modal knowledge graphs (MMKGs), denoted as G =
{xi, {x m i
}Mm=1}
N
i=1 and G˜ = {x˜j , {x˜
m j
}M˜m=1}
N˜
j=1, where xi and x˜j are entities in G and G˜, respectively. Each entity xi ∈ G is associated with M attribute-specific attributes {x m i}M
m=1, such as structured triples, textual descriptions, and images. Within a single graph, the association between an entity and its attributes is captured by entity-attribute pairs (xi, xm i, hm i), where h m i ∈ {0, 1} is a binary indicator, h m i = 1 indicates the valid intra-entity correspondence, and h m i = 0 denotes no correspondence between xi and x m i. Across graphs, intergraph correspondences govern the alignment of both the entity-entity pairs and attribute-attribute pairs. To be specific, the entity-entity pair is represented by (xi, x˜j , yij ), where the correspondence yij = 1 if xi and x˜j refer to the same real-world concepts, and yij = 0 otherwise. Similarly, the attribute-attribute pair is denoted by (x m i
, x˜
m j
, ym ij ), where the correspondence y m ij = 1 *i.f.f* both attributes are linked to correct entities (*i.e.*, h m i = 1 & h˜m j = 1) and the corresponding entities xi and x˜j are aligned (i.e., yij = 1). In other words, once the inter-graph entities are associated, their corresponding attributes could be treated as matched.

Given a query entity xi ∈ G, the goal of multi-modal entity alignment is to identify its equivalent entity x˜j from the other G˜ such that yij = 1. To this end, existing approaches typically follow a two-stage pipeline: i) intra-entity attribute fusion: for each entity xi, attribute representations are first extracted using attribute-specific encoders z m i = f m(x m i), and then aggregated to form a unified entity representation zi; ii) inter-graph discrepancy elimination: based on the fused entity representations zi and z˜j , contrastive learning (Chen et al., 2020) is employed to mitigate the inter-graph discrepancy.

However, in practice, this pipeline assumes that both the intra-entity correspondences (*i.e.*, entityattribute h m i) and inter-graph correspondences (*i.e.*, entity-entity yij and attribute-attribute y m ij ) are perfectly labeled. However, due to annotation errors, such an assumption is often violated, leading to the DNC challenge. As discussed in Introduction, the DNC problem would undermine the inter-graph and intra-entity learning, leading to remarkable performance degradation.

## 2.2 Reliability Estimation And Pair Division

To facilitate robust inter-graph discrepancy elimination and intra-entity attribute fusion, we first estimate the reliability of both the intra-entity and inter-graph correspondences by resorting to a two-fold principle, *i.e.*, uncertainty and consensus. Without loss of generality, in the following, we take the inter-graph entity-entity correspondence as a showcase to elaborate on the process of correspondence reliability estimation. For a given entity xi, the reliability wi between xi and its associated counterpart x˜j (yij = 1) is estimated using the following principle:
wi = (1 − ui) γ + ci(1 − γ), (1)
where γ is the balanced hyper-parameters (fixed as 0.5 for simplicity, see Appendix G.10 for more choices), ui and ci denote the uncertainty and consensus for the correspondence and will be detailed in the following sections.

## 2.2.1 Uncertainty Modeling

For a given entity, uncertainty in this work refers to whether its correspondence is trustworthy or not, which could serve as the principle to identify NC. According to the Dempster-Shafer Theory (Shafer, 1992), uncertainty could be quantified by evidence, which measures how the data support the

![3_image_0.png](3_image_0.png)

association between a query and a candidate. Specifically, the more evidences the entity accumulates, the lower uncertainty it embraces. Formally, evidence of the entity pairs (xi, x˜j ) is defined as eij = exp (tanh (sij/τ )), (2)
where sij = zi· z˜j denotes the dot product between the entity representation zi and z˜j , τ is the temperature, and the evidence vector for xiis ei = [ei1; ei2; *· · ·* ; eiN˜ ]. Following Subjective Logic (Sensoy et al., 2018), we associate the evidence vector ei with the parameters of the Dirichlet distribution αi = [αi1, αi2, · · · , αiN˜ ], where αij = eij + 1.

Definition 1. **Uncertainty.** For a given entity xi*, the uncertainty and the corresponding belief mass*

are defined as_  $$u_{i}=\frac{\tilde{N}}{Q_{i}}\;and\;b_{ij}=\frac{e_{ij}}{Q_{i}}=\frac{\alpha_{ij}-1}{Q_{i}},$$  _where $Q_{i}=\sum_{j}^{\tilde{N}}\left(e_{ij}+1\right)=\sum_{j}^{\tilde{N}}\alpha_{ij}\;and\;u_{i}+\sum_{j}^{\tilde{N}}b_{ij}=1$._
$$(3)$$

, (3)
The Qi denotes the Dirichlet distribution strength, and the belief mass assignment bi =
[bi1; bi2; · · · ; biN˜ ], *i.e.*, subjective opinion, corresponds to the Dirichlet distribution with parameters αi. Such a formulation encourages the mismatched entity-entity pairs to yield limited evidence, as the given entity fails to associate with any entity in the other MMKG, resulting in high uncertainty.

## 2.2.2 Consensus Modeling

Although the formulated uncertainty would help to identify noisy correspondence, we observe that a low uncertainty does not necessarily indicate a correct correspondence. Formally, Theorem 1. A low uncertainty ui does not necessarily imply that the highest belief is assigned to the annotated correspondence yi*, i.e.,*
zi *with low* ui ̸⇒ arg max bi = arg max yi. (4)
The Proof is placed in Appendix E. Here, yi = [yi1; yi2; *· · ·* ; yiN˜ ] is a one-hot vector indicating the inter-graph entity-entity correspondence of entity xi. Such a theorem highlights that uncertainty is insufficient to determine whether the belief is concentrated on the annotated correspondence. Therefore, we propose the consensus principle as follows. Definition 2. **Consensus.** For a given entity xi*, the consensus is defined as* ci = max(0, si· yi), (5)
where si = [si1, si2, · · · , siN ] *denotes the similarity vector,* max(0, ·) ensures the consensus is non-negative.

$$(4)$$
$$y_{i}=\arg\operatorname*{max}y_{i}.$$
$$({\boldsymbol{5}})$$
$\mathcal{C}_{ij}$
$\mathbf{\hat{i}}\,=\,\mathbf{\hat{j}}$
$$y_{i}\,\},$$

Intuitively, a low consensus ciindicates that the given correspondence is unreliable, thus serving as another principle to identify noisy correspondence. However, during inference, the annotated correspondence yiin Eq. 5 is unavailable. To remedy this, we propose to estimate the correct correspondence through a greedy strategy based on marginal contribution. Here, we begin with a definition of marginal contribution. Definition 3. For a given entity xi, the marginal contribution of its m*-th attribute is defined as*
∆ = v(π ∪ {m}) − v(π), (6)
where v(·) indicates the value function, π ⊆ Π \ {m} denotes a subset π of attributes excluding the m-th one, Π *is the complete set of available attributes.* In the implementation, we define the value function as v(π) = max 1 |π| Pj∈π s j i and v(π ∪
{m}) = max 1 |π|+1 Pj∈π∪{m}s j i
, where *| · |* denotes the number of attributes. Inspired by Shannon's principle that "the essence of information is to eliminate uncertainty", we expect that the informative attributes would contribute to establishing reliable correspondence for the entity-attribute pairs. Thus, Assumption 1. For a given entity xi*, if* x m iis correctly associated with xi, then ∆ ≥ 0*. Conversely,*
if x m iis irrelevant to xi*, then* ∆ < 0.

Assumption 1 provides a feasible way to estimate the correct correspondence. Specifically, incorporating attributes until the marginal contribution no longer improves, and the established subset π would help to indicate a reliable correspondence. To implement this, we adopt the following greedy strategy, π
∗ = π0 ∪ {m ∈ (Π \ π0) | v(π0 ∪ {m}) − v(π0) > 0} , (7)
where π0 denotes the initial subset with |π0| =M
2 + 1when M ≥ 3. See more details in Appendix F.3. With the selected subset π
∗, the estimated correspondence is finally given as yi =
one-hot(arg max( 1 |π∗| Pm∈π∗ s m i
)), where one-hot denotes the vector conversion.

## 2.2.3 Pair Division

With the formulated uncertainty and consensus, we could further identify the inter-graph NC. Specifically, we propose to divide the inter-graph pairs with yij = 1 into three portions: noisy portion with high uncertainty SU = {xi, x˜j | ui > βu}, noisy portion with low consensus SI = {xi, x˜j | ui ≤ βu and ci < βc} and clean portion SC = {xi, x˜j | ui ≤ βu and ci ≥ βc}. The thresholds βu and βc are determined in a self-adaptive manner via βu = min(u T P, 1 − β), βc = max(β, cT P ), (8)
where u T P = maxi∈STP ui, c T P = mini∈STP ci, and β indicates the threshold hyperparameter. Here, S
TP = {i| arg max(si) = arg max(yi)} denotes the set of true positive pairs. With the above pair division, the inter-graph pairs could be divided into SU , SI , and SC , which are further used for inter-graph discrepancy elimination.

## 2.3 Robust Inter-Graph Discrepancy Elimination

With the established reliability and pair division results, we could obtain three subsets: SU , SI , and SC . Since the pairs in SU exhibit high uncertainty, they are considered as unreliable and are excluded from the discrepancy elimination. As discussed in Section 2.2.2, inter-graph pairs with low consensus do not necessarily indicate correct matches, thus the pairs in SI cannot be regarded as reliable. Accordingly, we propose a novel Dually Robust Learning (DRL) that employs tailored strategies for the three subsets, thereby achieving robustness against inter-graph noisy correspondence. Formally, the overall objective is defined as L = LDR + λLReg, (9)
where LDR and LReg denotes the dually robust loss and regularization loss, λ denotes the trade-off parameter. Specifically, the dually robust loss and regularization loss are given by,

$$\mathcal{L}_{DR}\!=\!\mathcal{L}_{DR}(\alpha_{i},\hat{\mathbf{y}}_{i})\!+\!\sum_{m=1}^{M}\mathcal{L}_{DR}(\alpha_{i}^{m},\hat{\mathbf{y}}_{i}^{m}),\quad\mathcal{L}_{Reg}\!=\!\mathcal{L}_{Reg}(\alpha_{i},\hat{\mathbf{y}}_{i})\!+\!\sum_{m=1}^{M}\mathcal{L}_{Reg}(\alpha_{i}^{m},\hat{\mathbf{y}}_{i}^{m}),\tag{10}$$

where αm iand yˆ
m iare the Dirichlet parameter and refined correspondence for x m i. More specifically, for the given entity xi, the dually robust loss is defined as

$${\mathcal{L}}_{D R}(\mathbf{\alpha}_{i},{\hat{\mathbf{y}}}_{i})=\mathbb{I}\left(i\notin{\mathcal{S}}_{U}\right)\int\|{\hat{\mathbf{y}}}_{i}-\mathbf{p}_{i}\|_{2}^{2}\ D(\mathbf{p}_{i}\mid\mathbf{\alpha}_{i})\ d\mathbf{p}_{i},$$
2 D(pi| αi) dpi, (11)
where D(pi| αi) denotes the density function of the Dirichlet distribution over the query probability pi = [pi1, pi2, · · · , piN˜ ], and I(·) indicates an indicator function evaluating to 1 *i.f.f* the condition is satisfied. The refined correspondence yˆiis defined as follows,

$$(11)$$

$$\hat{\mathbf{y}}_{i}=\begin{cases}\mathbf{y}_{i},&\text{if}i\in\mathcal{S}_{C}\\ c_{i}\mathbf{y}_{i}+(1-c_{i})\operatorname{Softmax}(\mathbf{s}_{i}),&\text{if}i\in\mathcal{S}_{I}\end{cases}.\tag{1}$$
$$(12)$$
$$(13)$$

Such behavior enhances robustness against inter-graph noisy correspondences for the following reasons. On the one hand, the upper bound of query probability is proportional to Qi (Theorem 2), thus preventing over-optimization when the accumulate Qiis limited. On the other hand, excluding high-uncertainty correspondences in SU and refining the low-consensus correspondences in SI would prevent erroneous optimization caused by NC. Although the proposed dually robust loss in Eq. 11 could encourage higher evidence for inter-graph pairs with reliable correspondence, it is unable to guarantee that unassociated inter-graph pairs generate limited evidence. To achieve this, a Kullback-Leibler (KL) divergence term is adopted to penalize the evidence of the unassociated inter-graph pairs, *i.e.*,
LReg(αi, yˆi) = KL [D (pi| α˜i) ∥ D (pi| 1)] (13)

$${\mathcal{L}}_{\mathrm{Reg}}(\mathbf{\alpha}_{i},{\hat{\mathbf{y}}}_{i})=\mathrm{KL}\left[D\left(\mathbf{p}_{i}\mid{\hat{\mathbf{\alpha}}}_{i}\right)\,\parallel D\left(\mathbf{p}_{i}\mid\mathbf{1}\right)\right]$$

where 1 ∈ R
N˜is a N˜-dimensional vector of ones, α˜i = yˆi + (1 − yˆi) ⊙ αi denotes the Dirichlet parameters which help to penalize the evidence of unassociated correspondence, Γ(·) and ψ(·) are the gamma and digamma function, respectively.

## 2.4 Robust Intra-Entity Attribute Fusion

As discussed in Section 2.1, inter-graph attribute associations emerge as the by-product of establishing entity-attribute and entity-entity correspondences. Therefore, for correctly paired entities, the attributeattribute correspondence is incorrect, *i.f.f*, the corresponding entity-attribute correspondence is wrongly established. Thus, the inter-graph reliability w
m
icould be employed to identify unreliable
intra-entity attributes and weaken the emphasis on them during attribute fusion. Specifically, for a given entity xi, we employ the following Dually Robust Fusion (DRF) module to obtain the integrated representation,
$$z_{i}=\oplus_{m\in M}\left(w_{i}^{m}\cdot z_{i}^{m}\right),$$

## I), (14)
Where ⊕ Indicates The Concatenation Operator. Such Behavior Achieves Robustness Against Noisy
Entity-Attribute Pairs By Fusing The Multi-Modal Attributes With Adaptive Weights. In Other Words, Attributes With Higher Reliability Are Emphasized, While Those With Lower Reliability Are Weakened. 2.5 Test-Time Correspondence Reasoning

As discussed in the Introduction, the seemingly similar attributes might hinder the identification of equivalent entities. To solve the problem, we propose Test-time correspondence Reasoning (TTR) module, which uncovers the underlying attribute-attribute connections across graphs, thus improving the equivalent entity identification during inference. Specifically, the refined entity-entity similarity scores are given by,

$$\hat{\mathbf{s}}_{i}=\sum_{m\in M}\hat{w}_{i}^{m}\cdot\hat{\mathbf{s}}_{i}^{m},\tag{1}$$

where sˆ
m
irepresents the similarity scores of the m-th attribute output by the MLLM and wˆ
m
idenotes
the corresponding reliability weight. Such behavior could mitigate the negative impact of intraentity NC, which might undermine attribute fusion during test time. More specifically, we employ Chain-of-Thought (CoT) to guide the MLLM toward step-by-step reasoning. Mathematically,
$$\hat{\mathbf{s}}_{i}^{m}=\mathrm{Softmax}\left(\oplus_{j\in\mathcal{T}_{i}^{m}}\left(\mathrm{{\small{Cov}}}\left[x_{i}^{m},\hat{x}_{j}^{m},\mathbf{s}_{i}^{m}\right]\right)\right),$$
 , (16)
where T
m
idenotes the set of correspondences with the highest similarity in prior results s
m
i, CoT
indicates the reasoning process. Although a feasible solution is to prompt the MLLM with simple
$$(14)$$

$$(15)$$
$$(16)^{\frac{1}{2}}$$

Table 1: Comparisons with state-of-the-art methods on Non-name benchmarks under DNC setting.

"Inherent DNC" refers to the setting without any additional injected noise. H@k indicates the top-k retrieval accuracy while MRR denotes the mean reciprocal rank. The best and second best results are marked in **bold** and underline.

| Inherent DNC 20% DNC 50% DNC   |
|--------------------------------|

Setting Method ICEWS-WIKI ICEWS-YAGO DBP15K ZH-EN DBP15K JA-EN DBP15K FR-EN Avg.

H@1 H@5 MRR H@1 H@5 MRR H@1 H@5 MRR H@1 H@5 MRR H@1 H@5 MRR H@1

EVA 29.6 40.7 35.1 8.0 13.7 11.1 70.7 86.8 77.9 73.6 89.5 80.6 74.3 90.5 81.4 51.2 MCLEA 43.2 63.1 52.4 30.1 47.7 38.8 76.6 90.8 83.0 77.8 92.0 84.1 78.7 92.7 84.9 61.3 XGEA 49.8 61.5 55.5 35.5 46.7 41.2 81.1 93.0 86.3 82.6 94.3 87.8 83.1 94.7 88.3 66.4 MEAformer 53.5 70.1 61.3 35.0 51.2 42.8 82.4 93.5 87.3 81.9 94.2 87.3 82.1 94.4 87.5 67.0 UMAEA 51.2 70.0 59.9 32.4 49.4 40.6 79.1 93.2 85.3 79.6 93.9 85.8 81.2 95.0 87.3 64.7 PMF 52.6 67.9 59.9 38.3 53.2 45.4 83.9 94.6 88.9 83.9 94.9 89.0 84.4 95.3 89.6 68.6 HHEA 49.0 64.6 56.4 37.5 50.4 43.8 48.7 62.5 55.5 49.9 60.6 55.4 52.8 63.6 58.2 47.6 Ours 64.2 76.7 70.0 48.8 60.5 54.6 85.6 94.8 89.7 85.2 95.4 89.6 85.1 95.4 89.6 **73.8** EVA 15.2 21.6 18.4 0.2 0.4 0.4 51.0 70.2 59.7 54.5 73.4 63.1 53.4 73.8 62.6 34.9 MCLEA 34.5 53.6 43.5 24.6 40.4 32.5 69.9 85.7 77.0 70.1 85.6 77.2 70.7 87.3 78.1 54.0 XGEA 40.4 48.4 44.6 22.6 27.6 25.7 76.3 90.7 82.7 76.6 91.1 83.0 76.9 91.2 83.7 58.6 MEAformer 50.8 67.5 58.4 35.9 50.7 43.0 77.7 90.6 83.4 77.8 90.9 83.6 78.0 91.5 84.0 64.0 UMAEA 48.4 64.6 56.1 31.1 46.5 38.6 74.5 89.6 81.3 73.6 89.4 80.7 74.3 89.9 81.1 60.4 PMF 45.4 60.6 52.6 36.2 49.9 42.7 76.7 90.2 82.7 76.5 89.9 82.5 77.1 90.7 83.2 62.4 HHEA 47.8 61.8 54.4 37.4 49.5 43.3 48.7 58.8 53.8 49.0 58.7 54.0 52.5 61.7 57.1 47.1 Ours 62.4 75.1 68.5 48.3 59.5 53.9 81.1 92.0 86.0 80.5 92.2 85.6 80.5 92.2 85.8 **70.6** EVA 0.5 0.8 0.9 0.0 0.1 0.2 17.2 30.5 23.6 18.3 32.0 24.8 14.0 27.2 20.3 10.0 MCLEA 24.5 39.9 31.9 17.4 31.1 24.1 55.2 72.1 63.1 54.0 70.4 61.5 54.6 70.9 62.0 41.1 XGEA 39.5 47.0 43.4 23.7 27.8 26.3 67.9 83.6 74.9 68.0 83.8 75.0 68.0 83.9 75.1 53.4 MEAformer 42.4 58.8 50.1 30.6 45.0 37.5 68.1 83.7 75.1 62.9 80.3 70.8 65.8 82.6 73.4 54.0 UMAEA 37.8 55.0 46.0 25.4 40.0 32.5 64.8 82.1 72.7 58.1 78.5 67.2 61.8 80.9 70.3 49.6 PMF 35.1 48.8 41.8 29.6 42.4 35.8 67.1 82.6 74.2 65.6 80.7 72.5 66.1 81.5 73.1 52.7 HHEA 43.9 57.7 50.4 34.3 46.2 40.2 45.5 55.2 50.3 46.4 55.4 51.2 50.1 59.1 54.7 44.1 Ours 58.2 69.7 63.6 46.9 57.4 52.0 73.4 85.9 79.2 71.8 84.9 77.8 71.4 84.8 77.5 **64.3**

instructions such as "Identify the similarities between these attributes.", such vanilla prompts fail to fully activate the deep reasoning capabilities of MLLM. In contrast, the proposed CoT-based reasoning would enable the MLLM to leverage prior results and detailed steps for reasoning, preventing deviations from the prior knowledge while facilitating the mining of underlying connections. See Appendix F.5 and Appendix I for more details. Finally, the joint similarity score could be derived as s joint i = si + sˆi and the identified equivalent entity is given by arg max s joint i.

## 3 Experiments

In this section, we conduct extensive experiments on five widely-used MMEA datasets to validate the effectiveness of the proposed RULE. Due to space limitation, we present more experiments in Appendix G.

## 3.1 Implementation Details And Experimental Settings

Our method contains two networks, the attribute-specific encoders f m and the test-time correspondence reasoning module. Specifically, we first utilize a pre-trained CLIP model (Radford et al., 2021) to extract features from visual and textual attributes. After that, we employ the attribute-specific encoders to obtain the latent embeddings following (Huang et al., 2024a; Xu et al., 2023). For the test-time correspondence module, we use Qwen2.5-VL-72B-Instruct (Bai et al., 2025) as default to facilitate the test-time correspondence reasoning module (Section 2.5). Regarding hyperparameters, we set the trade-off parameter λ in Eq. 9, the threshold β in Eq. 8 are fixed as 1e
−4, 0.3 for all the experiments, respectively. The temperature τ in Eq. 2 is set to 0.07 following (Chen et al., 2020). We evaluate our method on five benchmark datasets: ICEWS-WIKI (Jiang et al., 2024), ICEWS-
YAGO, DBP15KZH-EN (Liu et al., 2021), DBP15KJA-EN, and DBP15KFR-EN. Details of the dataset and evaluation metric are provided in Appendix F.1 and F.4. As discuss in Introduction, the MMEA benchmarks including ICEWS are always contaminated by DNC which denoted as "Inherent DNC"
in the paper. To further evaluate the robustness toward DNC, we manually inject noise to conduct more comprehensive evaluations by following the widely-adopted strategies in the noisy correspondence/label learning community (Natarajan et al., 2013; Huang et al., 2021). Specifically, the artificial noise are injected in the following three aspects: i) *entity-entity NC*: one entity in an aligned entity pair is randomly replaced with a different entity; ii) *entity-attribute NC*: a visual or textual attribute is randomly reassigned to a different entity; iii) *attribute-attribute NC*: visual attributes are perturbed

Setting Method ICEWS-WIKI ICEWS-YAGO DBP15KZH-EN DBP15KJA-EN DBP15KFR-EN Avg.

H@1 H@5 MRR H@1 H@5 MRR H@1 H@5 MRR H@1 H@5 MRR H@1 H@5 MRR H@1

EVA 90.7 95.7 93.0 86.5 94.0 89.8 89.8 96.6 92.8 94.8 98.8 96.5 98.7 99.8 99.2 92.1 MCLEA 93.8 98.3 95.9 92.1 97.7 94.6 94.5 98.6 96.4 97.8 99.7 98.7 99.2 99.9 99.5 95.5 XGEA 83.5 94.4 88.6 93.9 97.3 95.8 91.4 97.4 94.1 94.3 98.0 96.0 97.3 99.3 98.2 92.1 MEAformer 95.9 98.8 97.2 93.8 97.9 95.7 96.7 99.0 97.7 98.8 99.8 99.3 99.6 **100.0** 99.8 97.0 UMAEA 94.8 98.7 96.6 92.8 97.9 95.1 95.4 98.9 97.0 98.2 99.7 98.9 99.4 99.9 99.6 96.1 PMF 94.9 98.4 96.5 92.8 97.7 95.0 96.3 99.1 97.6 98.5 99.7 99.1 99.5 **100.0** 99.7 96.4 HHEA 89.9 95.5 92.5 89.7 95.2 92.2 68.1 78.8 73.2 77.0 86.0 81.1 85.8 92.2 88.7 82.1 Ours 98.9 99.2 99.1 97.6 98.8 98.2 98.3 99.5 98.8 99.3 99.9 99.6 99.8 100.0 99.9 **98.8** EVA 67.4 76.2 71.6 17.9 21.4 19.7 64.2 78.9 70.8 72.6 85.7 78.5 88.0 95.2 91.3 62.0 MCLEA 89.0 95.2 91.8 88.8 95.8 92.0 91.5 97.0 94.0 95.6 98.8 97.0 97.8 99.6 98.6 92.5 XGEA 56.1 67.3 61.7 60.1 71.4 65.5 89.5 96.4 92.6 92.4 98.2 95.0 96.6 98.9 97.6 78.9

MEAformer 93.8 97.6 95.6 91.8 97.2 94.3 95.5 98.5 96.8 98.3 99.6 98.9 99.4 99.9 99.7 95.7

UMAEA 90.3 96.5 93.1 86.8 95.1 90.5 94.1 98.2 95.9 97.2 99.4 98.2 98.8 99.9 99.3 93.5 PMF 92.2 96.9 94.3 90.9 96.3 93.4 94.8 98.1 96.3 97.6 99.3 98.3 99.2 99.9 99.5 94.9 HHEA 87.6 93.8 90.5 89.3 94.6 92.1 66.1 75.9 70.8 72.6 81.9 77.0 83.5 90.2 86.6 79.8 Ours 98.3 98.9 98.6 97.5 98.7 98.1 97.6 99.1 98.3 99.1 99.9 99.5 99.8 100.0 99.9 **98.5** EVA 2.7 3.8 3.4 0.0 0.1 0.2 17.5 31.7 24.2 18.4 33.2 25.2 15.3 30.5 22.4 10.8 MCLEA 78.9 88.3 83.2 75.9 88.1 81.5 84.5 91.7 87.8 88.7 94.7 91.4 93.5 97.5 95.4 84.3 XGEA 50.3 60.3 55.3 34.8 44.5 39.8 71.3 86.4 78.0 70.1 85.5 77.0 88.7 95.9 91.9 63.0

MEAformer 91.9 96.7 94.1 91.9 96.8 94.1 93.4 97.3 95.2 97.3 99.1 98.1 99.1 99.9 99.5 94.7

UMAEA 87.0 94.4 90.4 85.7 93.9 89.4 91.4 96.7 93.8 95.9 98.8 97.2 98.1 99.6 98.8 91.6 PMF 86.9 93.9 90.0 87.6 94.4 90.7 92.2 96.5 94.2 96.1 98.8 97.3 98.6 99.6 99.1 92.3 HHEA 86.2 92.8 89.2 84.2 92.1 87.8 56.8 71.3 63.7 70.5 82.2 75.9 76.9 86.1 81.1 74.9 Ours 97.7 98.3 98.0 97.0 98.2 97.6 96.3 98.1 97.2 98.7 99.7 99.1 99.7 100.0 99.8 **97.9**

| Inherent DNC 20% DNC 50% DNC   |
|--------------------------------|

![7_image_0.png](7_image_0.png)

with Gaussian noise, while textual attributes are corrupted via random character replacements. The artificial noise levels are set as 20% and 50% in our experiments, which represents the proportion of corrupted E-E/E-A/A-A pairs.

## 3.2 Comparisons With State-Of-The-Arts

In this section, we compare our method RULE with seven state-of-the-art MMEA methods under the Dual-level Noisy Correspondence setting, including EVA (Liu et al., 2021), MCLEA (Lin et al., 2022b), XGEA (Xu et al., 2023), MEAformer (Chen et al., 2023a), UMAEA (Chen et al., 2023b), PMF (Huang et al., 2024a), and HHEA (Jiang et al., 2024). Following (Chen et al., 2023a; Huang et al., 2024a; Xu et al., 2023), we conduct experiments under two widely-adopted evaluation protocols:
Non-name setting denotes all attributes except for the entity name are used, while All-attributes setting includes all available modalities. For fair comparisons, we adopt the same backbone (*i.e.*,
CLIP) for all baselines and our method. For more results on different backbones, please refer to Appendix G.11. As shown in Tables 1-2, we could have the following conclusions: i) existing methods face substantial performance degradation as noise increases, highlighting their vulnerability to noisy correspondences. In contrast, RULE outperforms all baselines across different datasets and noise settings, demonstrating superior robustness against DNC; ii) even without any manually-injected noise, RULE still achieves performance gains compared to existing methods, as the real-world MMEA datasets contain a considerable number of DNC. To further verify the effectiveness of RULE, we conduct experiments under the manually-injected noise ratio from 0.0 to 0.7. As shown in Fig. 3 (a), RULE not only achieves higher performance across all noise levels but also exhibits significantly slower performance degradation, which further confirms the robustness of RULE against DNC.

![8_image_0.png](8_image_0.png)

Stage Setting Non-name **All-attributes**

H@1 H@5 MRR H@1 H@5 MRR

Train

w/o DRL 31.6 45.9 38.6 82.3 90.4 86.0 w/o DRF 50.4 66.2 57.6 93.4 97.4 95.2 Only Unc. 53.5 67.8 60.2 93.6 97.4 95.4 Only Cons. 48.3 60.3 54.3 87.7 93.2 90.4

Testw/o DRF 52.4 66.2 59.0 95.1 97.9 96.3

w/o TTR 56.5 68.6 62.3 94.0 97.7 95.7

MLLM Enhance 56.6 69.0 62.4 97.6 98.2 97.9

Both Default 58.2 69.7 63.6 97.7 98.3 98.0

![8_image_1.png](8_image_1.png)

## 3.3 Analysis And Ablation Study

In this section, we conduct analysis and ablation studies on the ICEWS-WIKI dataset.

Analysis Studies on Uncertainty and Consensus. As discussed in Section 2.2, the estimated reliability plays a key role in identifying DNC. To better understand its behavior, we visualize the reliability distribution of all training entity pairs. As shown in Fig. 3(b), clean pairs are concentrated on the right side of the plot (indicating high reliability), while noisy pairs are predominantly on the left (indicating low reliability). This confirms that the proposed reliability serves an effective indicator for distinguishing clean and noisy pairs. To further explore how the proposed uncertainty and consensus behave under noise, we construct subsets SU and SI by injecting synthetic noise and randomly shuffling the name attributes of the raw set SC . As illustrated in Fig. 4, uncertainty and consensus principles successfully separate the three subsets, which supports the design of our tailored loss strategies in Eq. 11. Effectiveness of Robust Fusion. To qualitatively study the effectiveness of RULE in handling entityattribute noise, we visualize the reliability in Eq. 1 during the fusion process. As shown in Fig.5, correctly associated attributes are assigned high reliability scores, while noisy or irrelevant attributes receive significantly lower scores. This behavior confirms that RULE effectively suppresses the influence of unreliable attributes during fusion, thereby enhancing robustness against entity-attribute noise. Ablation studies. To verify the effectiveness of each component in our framework, we conduct ablation experiments on the modules involved in both training and test-time phases. According to the results in Table 3, one could have the following conclusions. First, during training phase, both the "Only Unc." variant (which applies the uncertainty-guided loss in Eq. 18) and the "Only Cons." variant (which uses a consensus-based MSE loss) outperform the baseline "w/o DRL" (which uses only a standard MSE loss). This demonstrates the effectiveness of our proposed Dually Robust Learning mechanism in handling noisy correspondence. Second, during the test phase, the TTR module significantly improves alignment performance by uncovering latent semantic connections.

In particular, the comparison between the "MLLM Enhance" (which only uses rethinking scores in Eq. 16) and "w/o TTR" settings shows that combining rethinking scores with prior similarity scores leads to complementary effects, resulting in improved robustness and accuracy. Third, the Dually Robust Fusion (DRF) module effectively mitigates the influence of intra-entity NC. Its inclusion enhances performance in both the training and testing stages.

## 4 Conclusion

In this paper, we study a new problem in MMEA, *i.e.*, Dual-level Noisy Correspondence, which refers to the wrongly annotated intra-entity and inter-graph correspondences. To solve this problem, the proposed methods estimate the reliability of both the intra-entity and inter-graph correspondences and alleviate the negative impact of NC during the inter-graph discrepancy elimination and intra-entity attribution fusion. Beyond the training-time design, we employ a novel correspondence reasoning module to guarantee more accurate equivalent entity identification during inference. We believe this work might remarkably enrich the learning paradigm with noisy correspondence by simultaneously considering the noise across both training-time and test-time.

## Author Contributions

All authors contributed significantly to this work. Xi Peng and Mouxing Yang conceived the study, designed the RULE algorithm, refined the manuscript, and supervised the project. Haobin Li codesigned and implemented the RULE algorithm, conducted the baseline evaluations, and drafted the manuscript. Yijie Lin and Peng Hu analyzed the experimental results and contributed to the formulation of the manuscript. All authors reviewed and approved the final version.

## Acknowledgments

This work was supported in part by NSFC under Grant 624B2099; in part by the Fundamental Research Funds for the Central Universities under Grant CJ202303; in part by Sichuan Science and Technology Planning Project under Grant 24NSFTD0130; and in part by Fundamental and Interdisciplinary Disciplines Breakthrough Plan of the Ministry of Education of China under Grant JYB2025XDXM610.

## References

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2.5-vl technical report. *arXiv preprint arXiv:2502.13923*, 2025.

Cynthia Brame. Active learning. *Vanderbilt University Center for Teaching*, 2016. Liyi Chen, Ying Sun, Shengzhe Zhang, Yuyang Ye, Wei Wu, and Hui Xiong. Tackling uncertain correspondences for multi-modal entity alignment. *NeurIS*, 2024.

Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations. In *ICML*, 2020.

Zhuo Chen, Jiaoyan Chen, Wen Zhang, Lingbing Guo, Yin Fang, Yufeng Huang, Yichi Zhang, Yuxia Geng, Jeff Z Pan, Wenting Song, et al. Meaformer: Multi-modal entity alignment transformer for meta modality hybrid. In *ACM Multimedia*, 2023a.

Zhuo Chen, Lingbing Guo, Yin Fang, Yichi Zhang, Jiaoyan Chen, Jeff Z Pan, Yangning Li, Huajun Chen, and Wen Zhang. Rethinking uncertainly missing and ambiguous visual modality in multimodal entity alignment. In ISWC, 2023b.

Yunpeng Gong, Liqing Huang, and Lifei Chen. Eliminate deviation with deviation for data augmentation and a general multi-modal data learning method. *arXiv preprint arXiv:2101.08533*, 2021.

Yunpeng Gong, Liqing Huang, and Lifei Chen. Person re-identification method based on color attack and joint defence. In *CVPR, 2022*, pp. 4313–4322, 2022.

Yunpeng Gong, Zhun Zhong, Yansong Qu, Zhiming Luo, Rongrong Ji, and Min Jiang. Crossmodality perturbation synergy attack for person re-identification. Advances in Neural Information Processing Systems, 37:23352–23377, 2024.

Hao Guo, Jiuyang Tang, Weixin Zeng, Xiang Zhao, and Li Liu. Multi-modal entity alignment in hyperbolic space. *Neurocomputing*, 2021.

Zongbo Han, Changqing Zhang, Huazhu Fu, and Joey Tianyi Zhou. Trusted multi-view classification with dynamic evidential fusion. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 2022.

Sida Huang, Hongyuan Zhang, and Xuelong Li. Enhance vision-language alignment with noise. In AAAI, 2025.

Yani Huang, Xuefeng Zhang, Richong Zhang, Junfan Chen, and Jaein Kim. Progressively modality freezing for multi-modal entity alignment. *arXiv preprint arXiv:2407.16168*, 2024a.

Zhenyu Huang, Guocheng Niu, Xiao Liu, Wenbiao Ding, Xinyan Xiao, Hua Wu, and Xi Peng.

Learning with noisy correspondence for cross-modal matching. In *NeurIPS*, 2021.

Zhenyu Huang, Mouxing Yang, Xinyan Xiao, Peng Hu, and Xi Peng. Noise-robust vision-language pre-training with positive-negative learning. *IEEE Transactions on Pattern Analysis and Machine* Intelligence, 2024b.

Xuhui Jiang, Chengjin Xu, Yinghan Shen, Yuanzhuo Wang, Fenglong Su, Zhichao Shi, Fei Sun, Zixuan Li, Jian Guo, and Huawei Shen. Toward practical entity alignment method design: Insights from new highly heterogeneous knowledge graph datasets. In WWW, 2024.

Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In *ICML*, 2022.

Qian Li, Shu Guo, Yangyifei Luo, Cheng Ji, Lihong Wang, Jiawei Sheng, and Jianxin Li. Attributeconsistent knowledge graph representation learning for multi-modal entity alignment. In WWW, 2023.

Jia-Qi Lin, Man-Sheng Chen, Chang-Dong Wang, and Haizhang Zhang. A tensor approach for uncoupled multiview clustering. *IEEE Transactions on Cybernetics*, 2022a.

Jia-Qi Lin, Man-Sheng Chen, Xi-Ran Zhu, Chang-Dong Wang, and Haizhang Zhang. Dual information enhanced multiview attributed graph clustering. IEEE Transactions on Neural Networks and Learning Systems, 2024.

Yijie Lin, Mouxing Yang, Jun Yu, Peng Hu, Changqing Zhang, and Xi Peng. Graph matching with bi-level noisy correspondence. In ICCV, 2023.

Zhenxi Lin, Ziheng Zhang, Meng Wang, Yinghui Shi, Xian Wu, and Yefeng Zheng. Multi-modal contrastive representation learning for entity alignment. *arXiv preprint arXiv:2209.00891*, 2022b.

Fangyu Liu, Muhao Chen, Dan Roth, and Nigel Collier. Visual pivoting for (unsupervised) entity alignment. In AAAI, 2021.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In *NeurIPS*,
2023.

Ye Liu, Hui Li, Alberto Garcia-Duran, Mathias Niepert, Daniel Onoro-Rubio, and David S Rosenblum.

Mmkg: Multi-modal knowledge graphs. In *ESWC*, 2019.

Nagarajan Natarajan, Inderjit S Dhillon, Pradeep K Ravikumar, and Ambuj Tewari. Learning with noisy labels. In *NeurIPS*, 2013.

Heiko Paulheim. Knowledge graph refinement: A survey of approaches and evaluation methods.

Semantic web, 2016.

Shichao Pei, Lu Yu, Guoxian Yu, and Xiangliang Zhang. Rea: Robust cross-lingual entity alignment between knowledge graphs. In KDD, 2020.

Alessandro Piscopo and Elena Simperl. Who models the world? collaborative ontology creation and user roles in wikidata. *Proceedings of the ACM on Human-Computer Interaction*, 2(CSCW):1–18, 2018.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In *ICML*, 2021.

Murat Sensoy, Lance Kaplan, and Melih Kandemir. Evidential deep learning to quantify classification uncertainty. In *NeurIPS*, 2018.

Glenn Shafer. Dempster-shafer theory. *Encyclopedia of Artificial Intelligence*, 1992. Fabian M Suchanek, Gjergji Kasneci, and Gerhard Weikum. Yago: A core of semantic knowledge.

In WWW, 2007.

Denny Vrandeciˇ c and Markus Krötzsch. Wikidata: A free collaborative knowledgebase. ´ Communications of the ACM, 2014.

Yuanyi Wang, Haifeng Sun, Jiabo Wang, Jingyu Wang, Wei Tang, Qi Qi, Shaoling Sun, and Jianxin Liao. Towards semantic consistency: Dirichlet energy driven robust multi-modal entity alignment.

In *IEEE ICDE*. IEEE, 2024.

Baogui Xu, Chengjin Xu, and Bing Su. Cross-modal graph attention network for entity alignment. In ACM Multimedia, 2023.

Baogui Xu, Yafei Lu, Bing Su, and Xiaoran Yan. Position-aware active learning for multi-modal entity alignment. In *ICASSP*. IEEE, 2024.

Mouxing Yang, Yunfan Li, Zhenyu Huang, Zitao Liu, Peng Hu, and Xi Peng. Partially view-aligned representation learning with noise-robust contrastive loss. In *CVPR*, 2021.

Mouxing Yang, Zhenyu Huang, Peng Hu, Taihao Li, Jiancheng Lv, and Xi Peng. Learning with twin noisy labels for visible-infrared person re-identification. In *CVPR*, 2022.

Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In *ICCV*, 2023.

Xiangru Zhu, Zhixu Li, Xiaodan Wang, Xueyao Jiang, Penglei Sun, Xuwu Wang, Yanghua Xiao, and Nicholas Jing Yuan. Multi-modal knowledge graph construction and application: A survey. *IEEE*
Transactions on Knowledge and Data Engineering, 2022.