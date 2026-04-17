# URS: A Unified Neural Routing Solver for Cross-Problem Zero-Shot Generalization

- Decision: Reject
- Scores: 4, 6, 4, 4

## Abstract
Multi-task neural routing solvers have emerged as a promising paradigm for their ability to solve multiple vehicle routing problems (VRPs) using a single model. However, existing neural solvers typically rely on predefined problem constraints or require per-problem fine-tuning, which substantially limits their zero-shot generalization ability to unseen VRP variants. To address this critical bottleneck, we propose URS, a unified neural routing solver capable of zero-shot generalization across a wide range of unseen VRPs using a single model without any fine-tuning. The key component of URS is the unified data representation (UDR), which replaces problem enumeration with data unification, thereby broadening the problem coverage and reducing reliance on domain expertise. In addition, we propose a Mixed Bias Module (MBM) to efficiently learn the geometric and relational biases inherent in various problems. On top of the proposed UDR, we further develop a parameter generator that adaptively adjusts the decoder and bias weights of MBM to enhance zero-shot generalization. Moreover, we propose an LLM-driven constraint satisfaction mechanism, which translates raw problem descriptions into executable stepwise masking functions to ensure solution feasibility. Extensive experiments demonstrate that URS can consistently produce high-quality solutions for more than 100 distinct VRP variants composed of 10 different constraints without any fine-tuning, which includes more than 90 unseen variants. To the best of our knowledge, URS is the first neural solver capable of handling over 100 VRP variants with a single model.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces URS, a novel unified RL routing approach that is able to find high-quality solutions for over 100 VRP variants, of which 90 are done through zero-shot generalization. URS is able to achieve this through four novel proposed components. The first is a unified data representation, whereby each routing problem shares a unified representation. Secondly, URS introduces a Mixed Bias Module (MBM), an attention mechanism that learns the geometric and relational priors, such as distance and the relation between nodes. Thirdly, they utilize a parameter generator that adjusts the weights and biases of the MBM for each specific routing problem. Lastly, URS uses an LLM that generates a masking function for each routing variant, whereby it checks the mask against a predefined checker function to see if it is correct.

### Strengths
• The unified representation is well motivated, and the ablation study in Table 4 shows the cost of removing attributes, such as problem state and node-type indicator.

• URS is able to handle over 100 VRP variants, whereby 96 variants are unseen through training, and URS is able to achieve this through zero-shot learning. The results also indicate that URS is either able to achieve similar or outperform existing baselines on the seen VRP variants, and outperforms all baselines on the unseen variants.

• While the main results are URS using the AM model, the ablation study shows that URS is also able to work fine with POMO or ReLD, which both report a worse gap; however, the difference is only around 1%. This signifies that URS is not strongly dependent on the model used.

### Weaknesses
•	The most glaring weakness of this paper is the LLM masking. The paper explains that an LLM is used to create a masking function, whereby it iterates through prompting by checking if the masking function satisfies a checker function. The main issue is that the checker function needs to be predefined, and without it, URS would not function. However, if you need to predifine the checker function, why not define the masking yourself? The LLM feels unnecessary to me and also raises questions about the zero-shot capabilities of URS, if the checker function is predefined. This limitation is also not mentioned in the limitations section.

•	Section 3.2 about the model architecture is difficult to understand, and it is unclear why MBM, the WEIGHT($\lambda$), and BIAS($\lambda$) are needed. Especially, WEIGHT($\lambda$) and BIAS($\lambda$) are unclear to me, given that the ablation study does not include them. The paper would improve if it included a better explanation of why they are included and how they improve the performance of URS.

### Questions
•	Would it be possible to do the masking completely automatically, whereby the checker function is also written by an LLM?

•	How important is the LLM masking for URS to function? Is it needed for each VRP variant or only some?

•	How many iterations are needed on average for the LLM to find a valid masking?

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
URS proposes a unified neural solver along with a unified data representation (UDR) scheme for 100+ VRP variants using a single model which possess strong zero-shot generalization capability to other unseen VRP variants, prominently elevating the performance and expanding solvability scope of NCO model to a wider range of VRPs especially on current rarely-studied VRP variants (over 52 variants). Since generalization and adaptability on a wider range of VRP tasks has been an very promising and necessary direction in this field, the proposed framework in the paper offers a solid foundation for wider real-world application of NCO solver in the field of vehicle routing tasks.

### Strengths
1) Realistic wider application: Solving 100+ VRP variants with one model is genuinely ambitious. Most prior work handles maybe 10-20 variants with a pre-defined basket of problem tag or problem-specific adapters, so this is a real step forward for broader deployment that requires less training and expert knowledge.

2) Concrete representation of VRPs: The design of UDR is an interesting thought and makes sense as shown by many experiments-nstead of hardcoding problem types, you define a shared feature space and let different problems "activate" different features. With concrete and thorough consideration of problem settings (e.g.  multi-hot representation λ is simple and effective), UDR feels like the right level of abstraction. The combination of mixed bias module, UDR-based parameter generator and LLM-driven constraints-satisfication checker generation also ensures practicability and effectiveness of the unified network.

3) Comprehensive experiments and ablations: Testing on 96 unseen variants (including 52 brand new ones) is thorough. The ablations clearly show each component matters - removing node-type indicators or the relation matrix hurts performance where you'd expect. While the performance degradation of URS compared to URS-STL remains at an acceptable range,  the generalization capability on unseen problems is well-established in tables 10-11. I appreciate seeing results on both seen and unseen problems.

### Weaknesses
Reproducibility and Missing Details: Several implementation details remain unclear, which raises concerns about reproducibility. For instance, it is not specified which large language model (LLM) was used in Section 4, nor which evolutionary algorithm was employed for evolving the mask generator and constraint checker. Additionally, the number of iterations and the typical runtime for the evolutionary process are not reported. Finally, the absence of a GitHub or code repository link further limits the reproducibility of the reported results.

limitation of the proposed method is not dicussed.

### Questions
1.	On the difficulty of Duration-Limited and Open-Route problems:
What factors make Duration-Limit (L) and Open-Route variants more challenging for URS? As shown in Table 10, URS consistently underperforms baselines on problems with Duration Limit and Time Window constraints—for instance, OCVRPB (9.35% vs. 5.36% for ReLD-MTL) and CVRPLTW (2.67% vs. 1.17% for ReLD-MTL). Similarly, Open-Route combinations also exhibit weaker performance. These variants appear to share specific constraint patterns that might expose weaknesses in URS’s design. Could the authors analyze why these variants are harder and discuss potential architectural modifications that could better handle such constraints?
2.	On the choice of training problems:
The paper reports training on 11 specific problem types. How would the performance change if the model were trained on a broader set (e.g., 20 problems) or on a different selection of the 11? A justification of the problem selection criteria, along with a brief sensitivity analysis, would strengthen the empirical validity of the results.
3.	On the single-task version (URS-STL):
Could the authors clarify what the single-task variant (URS-STL) entails? Does it refer to training on a single problem type using the same hyperparameters and training setup as URS, or does it involve any task-specific adaptation? A clearer explanation would help interpret the reported results.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper addresses to learn solvers for vehicle routing problems with various
combinations of 10 properties and constraints (capacity, open route, backhaul,
backhaul and priority, duration limit, time windows, multi-depot, price collecting,
asymmetric distances, pickup and delivery).  The authors propose to 
represent all these problems as a graph of customer nodes where each nodes
has features for all the 10 problems, set to 0, if the problem instance does
not feature this property (called "unified data representation"). An attention-based
model is used to predict the next node to add to the current tour. 
Attentions are  biased by the distance between nodes (in both directions) and
a problem-specific relation between them (e.g., a priority constraint in 
pickup and delivery problems) as well as a learnable factor depending
on an indicator for different problem properties and constraints (called 
"multi-hot problem representation"). To mask off infeasible candidates,
an LLM is tasked with writing the code to produce the masks based on
a natural language description of the constraints. In experiments it is shown
that the proposed approach for TSP type problems works better than 
the existing multitask model GOAL-MTL, while for VRP type problems
it is on par with ReLD-MTL on the problem variants it has been trained
for. On the unseen problem variants, it performs better, and it is shown
to solve many combinations other multitask solvers cannot tackle.

### Strengths
- s1. common representation of 10 different VRP properties and constraints.
- s2. ample experiments on over 100 combinations

### Weaknesses
- w1. in the evaluation, it is not clear what percentage of feasible solutions the proposed
  model achieves.
- w2. the effort required to write the constraint checkers for the 10 different VRP properties
  and constraints seems exaggerated, and thus the approach to let an LLM write them questionable.
- w3. a comparison with published results in the literature is unclear, esp. for problem-specific
  solvers.

more details:

w1. in the evaluation, it is not clear what percentage of feasible solutions the proposed
model achieves.
- As the code for the masks is written by an LLM, there is no guarantee that the constraints
  actually are met in the end. Did you evaluate how many of the solutions actually are 
  feasible?

w2. the effort required to write the constraint checkers for the 10 different VRP properties
and constraints seems exaggerated, and thus the approach to let an LLM write them questionable.
- writing 10 constraint checkers seems not so much effort. You talk about constraint
  checkers for the combinations of those constraints. Are there combinations of constraints
  that pose difficulties? I.e., are not just done by checking constraint 1 and 2 individually?

w3. a comparison with published results in the literature is unclear, esp. for problem-specific
solvers.
- For example, for VRP type problems, only CVRP is compared with problem specific solvers such
  as POMO, and here POMO wins by a good margin.
- Are there other VRP type problems with established neural solvers where URS gets close?
- Can you reproduce some experiments on the established methods such as Sym-NCO or BQ
  and compare your method on their evaluation scenarios?



smaller points:
- p1. on p.3, "a feasible solution is a tour" is only valid for TSP type problems, and then m is n.
- p2. the claim to look at "more than 100 distinct VRP variants" is slightly misguiding, as all these 
  variants are different combinations of 10 properties and constraints.
- p3. on p.4 "multi-hot type tensor": usually called "binary vector".
- the relation matrix R (eq. 4) comes out of the blue, it was not introduced in sec. 3.1. But as it is 
  part of the common representation of  the 10 different problem aspects, it should be mentioned 
  in sec. 3.1.

### Questions
- q1. Can you report how many of the tours the LLM authored masking mechanism
  lets through, actually are feasible?
- q2. Are there combinations of constraints that pose difficulties? I.e., are not just done 
  by checking constraint 1 and 2 individually?
- q3. Are there VRP type problems beyond CVRP with established neural solvers where 
  URS gets close?

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
4

### Summary
This paper proposes URS to enhance the zero-shot generalization of neural routing solvers across various unseen VRPs. It incorporates four components: unified data representation (UDR) to broaden problem coverage; mixed bias module (MBM) to learn the relational biases in various problems; a parameter generator to adjust model parameters for generalization enhancement; and  an LLM-driven constraint satisfaction mechanism to engender stepwise masking function for solution feasibility. Experimental results show the effectiveness of the proposed method across multiple variants.

### Strengths
1. The paper proposes a comprehensive framework to address the zero-shot cross-problem generalization issue for VRPs, allowing fully zero-shot inference without retraining.
2. The combination of UDR, MBM, and conditional hypernetwork forms a coherent and principled framework. The paper is well-written and easy to follow.
3. Experimental results show that the proposed method can efficiently solve over 100 problem variants, 90 of which are unseen during training.

### Weaknesses
1. The framework is well-constructed, but its components (UDR, MBM, conditional generator) mainly combine known ideas such as feature unification and attention biasing similar to existing works. The novelty lies more in integration and scale than in new learning principles.

2. The strong zero-shot results may largely stem from the LLM-based constraint module, yet its impact is not clearly isolated. The paper also lacks analysis of mask reliability, prompt sensitivity, or consistency across different LLM settings.

3. Experiments are limited to VRPs ( ≤1000-node), leaving scalability to larger or more realistic problems (>2000 nodes) uncertain.

### Questions
Please refer to the weakness part.

### Soundness
2

### Presentation
3

### Contribution
2
