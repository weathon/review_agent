# ConRep4CO: Contrastive Representation Learning of Combinatorial Optimization Instances across Types

- Decision: Accept (Poster)
- Scores: 4, 4, 6, 4

## Abstract
Considerable efforts have been devoted to machine learning (ML) for combinatorial optimization (CO) problems, especially on graphs. Compared to the active and well-established research for representation learning of text and vision, etc., it remains under-studied for the representation learning of CO problems, especially across different types. In this paper, we try to fill this gap (especially for NP-complete (NPC) problems, as they, in fact, can be reduced to one another). Our so-called ConRep4CO framework, performs contrastive learning by first transforming CO instances in various original forms into the form of Boolean satisfiability (SAT). This scheme is readily doable, especially for NPC problems, including those practical graph decision problems (GDPs) which are inherently related to their NP-hard optimization versions. Specifically, each positive pair of instances for contrasting consists of an instance in its original form and its corresponding transformed SAT form, while the negative samples are other instances not in correspondence. Extensive experiments on seven GDPs (most of which are NPC) show that ConRep4CO significantly improves the representation quality and generalizability to problem scale. Furthermore, we conduct extensive experiments on NP-hard optimization versions of the GDPs, including MVC, MIS, MC and MDS. The results show that introducing ConRep4CO can yield performance improvements of 61.27%, 32.20%, 36.46%, and 45.29% in objective value gaps compared to problem-specific baselines, highlighting the potential of ConRep4CO as a unified pre-training paradigm for CO problems.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper considers a contrastive learning based approach to graph Combinatorial Optimization (CO) problems. The approach is based 
on associating each CO problem with a graph decision problem. It then associates each GDP with a Boolean Satisfiability problem (SAT) and considers the corresponding SAT graph. The authors then develop models for each of the GDPs and one unified model for the SAT graphs. During training, they use a contrastive loss that purportedly aligns the GDP and SAT modalities they also use a warm-start training technique.

Experimentally, the authors consider seven GDPs include k-clique, k-coloring and k-independent sets. They that adding in the ConRep4CO framework (which appears to refer to their contrastive loss between the SAT and the GDP) improves performaces verses an ablated model. The performance is quantified both on the performance of the representation for solving the GDP and the underlying CO problem, although the performance increase is substantially more modest in the former case (which makes it hard to gain intuition about the larger performance boost in the CO problem).

### Strengths
Method does seem to improve performance in most cases

### Weaknesses
The method is not clearly explained. More specifically, more background should be given on how a GDP may be related and how the GDP is related to a SAT or a SAT graph. Furthermore, it is not clear how the model actually operates. The overall architecture is not well-explained in the text and in my summary I am making educated guesses based on Figure 2, which is not a substitute for actually explaining the model.

Methodological contributions seem incremental, although this is hard to assess to the general lack of clear explanations as noted above 

Additionally, there insufficient discussion of computational complexity / run time. This makes it difficult to weigh the increase in model performance against increased cost. Notably, in CO problems, this is particularly essential. All of these problems are exactly solvable, e.g, by using GUROBI solvers, so the entire name of the game is developing fast, approximate solutions.

The code is not available. This is particularly problematic because a) the method is motivated primarily by experimental results and b) there is not enough detail given on the model in order to be able to reproduce these experiments.

Minor:

The first two sentences of the abstract seem to contradict one another

"texts" should be "text" in line 42

Notational inconsistency G_i^j vs G_j in the start of section 3.3.1

The relative contributions of the SAT backbone and GNN backbone are hard to parse in Table 1

### Questions
what is the computational complexity of the proposed model?

how does performance compare to non message-passing based approached, e.g., reinforcement learning

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes ConRep4CO, a contrastive pre-training framework for learning representations across different combinatorial optimization (CO) problems. The key idea is to transform graph decision problems (GDPs) into SAT form and use contrastive learning to align representations. Positive pairs consist of a GDP instance and its SAT transformation; negative pairs are non-corresponding instances. The method shows improvements on 7 GDPs and 4 downstream optimization problems (MVC, MIS, MC, MDS).

### Strengths
- The use of SAT as a common modality to bridge different CO problems is creative and theoretically grounded. The motivation that NPC problems can be reduced to one another through SAT is compelling and elegantly exploited.
- Extensive experiments across 7 GDPs with multiple architectures (GCN, GraphSAGE, PGN, GraphGPS), different difficulty levels; demonstrates generalization to both larger problems and unseen problem types.
- The downstream results on MVC, MIS, MC, and MDS show substantial improvements
- Ablations on cross-domain information transfer, different loss functions, and sensitivity analysis provide good insight into what drives the performance gains.

### Weaknesses
- Experiments use very small graphs. Performance gains decrease substantially with scale. Unclear how this scales.
- No empirical comparison to GOAL, UniCO methods despite discussing them. Downstream comparisons only include problem-specific baselines.
- Warm-start strategy mentioned as important but not ablated
- False negatives issue: Modified loss without false negatives shows comparable performance, questioning what contrastive loss actually learns
- Temperature differs between difficulty levels without justification or sensitivity analysis

### Questions
- Pre-training uses very small graphs (5-25 vertices) while downstream performance degrades substantially with scale. Real-world CO problems typically involve thousands to tens of thousands of nodes. To be practically useful, would the approach require pre-training on much larger problem instances? Can you provide a guess of the time and ressources needed to train on graphs with 10,000+ vertices, and maintain similar performance improvements at that scale?
- Can you add empirical comparisons to GOAL and UniCO ? These seem like important missing baselines. 
- How sensitive is performance to warm-start? What happens without it?
- SAT Model Failure: Why does SAT model show 50% accuracy (random) on many hard instances while graph models transfer reasonably?
- Temperature Selection: Why does temperature differ between easy/medium datasets? How was this chosen?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces ConRep4CO, a framework that leverages contrastive pretraining to generate representations for combinatorial optimization. The core innovation lies in its method of unifying diverse NP-complete problems by converting them into a Boolean satisfiability format; these SAT-derived versions serve as positive counterparts to their original instances within the contrastive learning objective. Empirical evaluation across seven graph-based decision problems and four optimization tasks indicates that the framework not only enhances performance but also exhibits robust generalization, positioning it as a viable paradigm for unified representation learning in the CO domain.

### Strengths
1. The core novelty of this work lies in its use of Boolean satisfiability as a unifying representational bridge, enabling knowledge transfer across distinct combinatorial optimization problems.

2. The experiments are sufficient, rigorously evaluating the framework across various graph decision problem categories, architectural backbones, and downstream solver methodologies.

3. The results demonstrate observable empirical improvements, confirming that the method enhances both the quality of the learned representations and the end-to-end performance of the solvers that utilize them.

### Weaknesses
1. While the results are compelling, the paper would benefit from a deeper mechanistic analysis of the SAT-based alignment process. A more thorough investigation into why this specific form of contrastive learning fosters such effective cross-domain generalization would strengthen the theoretical contribution.

2. It seems that the framework is explicitly designed for problems that can be efficiently reduced to SAT (primarily NP-complete problems). I am not sure whether I missed anything. But its applicability to problems outside this class, or to CO problems with complex, non-logical constraints (e.g., certain scheduling or routing constraints), is unclear and not explored.

3. The practical scalability of ConRep4CO warrants further discussion, as the computational complexity associated with the CNF transformation and the requirement to train multiple models could present significant bottlenecks for larger, real-world instances.

### Questions
1. The paper justifies the use of SAT through its universal expressiveness (reducibility). However, a deeper theoretical or visualized explanation of its representational advantage would be beneficial. For instance, does the SAT form act as an information bottleneck, stripping away problem-specific noise to reveal a common core, or does it function as a canonical encoder that explicitly captures fundamental combinatorial structures shared across different GDPs?

2.  To ensure reproducibility and provide deeper insight, could the authors elaborate on the training details? Specifically, clarifying the objective and impact of the warm-start phase, and explaining how the balance (relative weighting) between the two loss components was determined or tuned would be highly valuable.

3. The premise of the framework is that diversity in pre-training tasks fosters generalization. Was this correlation investigated? Specifically, does performance on downstream tasks improve when the model is pre-trained on a broader variety of problem types? Furthermore, which specific design elements (e.g., the specific contrastive pairing, the SAT transformation itself) are identified as most critical for enabling this effective cross-domain knowledge transfer?

4. How does the method scale with an increasing number of pre-training problem types? Have you considered or experimented with other unifying formalisms beyond SAT (e.g., Integer Linear Programming) that might offer different trade-offs in terms of representation compactness or transformation complexity?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces ConRep4CO, a novel training framework that leverages contrastive learning to significantly enhance the learned representations of graph decision problems. Experimental results demonstrate that ConRep4CO's contrastive learning techniques indeed improve model performance. Furthermore, the approach shows promising extensibility, maintaining performance improvement when applied to unseen combinatorial optimization problems.

### Strengths
-	The illustration of the training process is clear and easy to follow.
-	The experimental evaluations are comprehensive.

### Weaknesses
-	The definitions of some symbols are missing. What are confidence intervals? What is ER in Table 4? What is RB used in Table 5?
-	It is unclear why the models are only trained on the easy and medium problems without considering the hard problems (Tables 1 and 2).
-	Some numeric calculations in Table 4 are incorrect. For instance, GCNN+ConRep4CO gets 55.17 OBJ score, while the optimal score is 54.62. The gap_{abs} score should be 55.17-54.62=0.55, not 0.57, which is listed in the table. In fact, I find that all the gap_{abs} scores of GCNN+ConRep4CO are incorrect.
-	The improvement introduced by training the model across multiple domains (Table 6) seems to be very small. The difference between the single domain setting and the multiple domain setting is within the standard deviation.

### Questions
1.	What are the definitions of confidence intervals, ER, and RB?
2.	Why are models only trained on the easy and medium problems (Tables 1 and 2)?

### Soundness
2

### Presentation
2

### Contribution
2
