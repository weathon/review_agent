# Architect of the Bits World: Masked Autoregressive Modeling for Circuit Generation Guided by Truth Table

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 4, 6, 6

## Abstract
Logic synthesis, a critical stage in electronic design automation (EDA), optimizes gate-level circuits to minimize power consumption and area occupancy in integrated circuits (ICs). 
Traditional logic synthesis tools rely on human-designed heuristics, often yielding suboptimal results. 
Although differentiable architecture search (DAS) has shown promise in generating circuits from truth tables, it faces challenges such as high computational complexity, convergence to local optima, and extensive hyperparameter tuning.
Consequently, we propose a novel approach integrating conditional generative models with DAS for circuit generation.
Our approach first introduces CircuitVQ, a circuit tokenizer trained based on our Circuit AutoEncoder
We then develop CircuitAR, a masked autoregressive model leveraging CircuitVQ as the tokenizer.
CircuitAR can generate preliminary circuit structures from truth tables, which guide DAS in producing functionally equivalent circuits. 
Notably, we observe the scalability and emergent capability in generating complex circuit structures of our CircuitAR models.
Extensive experiments also show the superior performance of our method.
This research bridges the gap between probabilistic generative models and precise circuit generation, offering a robust solution for logic synthesis.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes Architect of the Bits World, a framework that combines a conditional generative model with differentiable architecture search (DAS) for logic-circuit generation. A VQ-based tokenizer (CircuitVQ) converts circuits into discrete tokens, and a masked autoregressive Transformer (CircuitAR) generates circuit topologies conditioned on truth tables. The generated circuits serve as initialization priors for DAS, which guarantees final functional correctness. The authors introduce a proxy metric (Bits Distance) to measure conditional generation quality and demonstrate that larger models (0.3B–2B params) reduce DAS search steps and NAND counts on IWLS benchmarks, showing promising scaling and efficiency.

### Strengths
1. Clear motivation and practical pipeline combining generation and optimization.

2. Solid empirical study showing efficiency and scaling trends.

3. Comprehensive ablations and architectural details supporting reproducibility.

### Weaknesses
1. Limited evaluation on large IWLS circuits; scalability and generalization remain unproven.

2. BitsD is only heuristically justified as a proxy for final metrics.

3. Lack of analysis on DAS convergence and potential failure cases.

### Questions
1. (**Major**) Could you provide results on larger or full IWLS circuits, including functional accuracy and final area/NAND count after DAS? Please report success rate and scaling behavior with circuit size.

2. What is the empirical correlation between Bits Distance and post-DAS metrics (NAND count, search steps, area)?

3. How consistent is the claim of “100% correctness guaranteed by DAS”? Please provide convergence thresholds, number of restarts, and any failure cases.

4. Could you include training/inference cost statistics (GPU hours, DAS runtime per circuit) to assess practical efficiency?

### Soundness
3

### Presentation
3

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
This paper proposes a two-stage approach to logic circuit generation from truth tables. The authors introduce Bits Distance—the MAE between an untrained DAS model (initialized by CircuitAR) and the target truth table—as a proxy for conditional generation quality.
Empirically, on selected IWLS’22 tasks, the method claims 100% functional accuracy while reducing DAS optimization steps by ~45% on average vs. CircuitNN and search space / #NAND relative to baselines. The paper also presents scaling curves (300M→2B parameters) for BitsD and an “emergent capability” plot suggesting deeper yet more NAND efficient circuits at 2B parameters. The Limitations section acknowledges compute constraints.

### Strengths
• A clear and interesting integration of masked autoregressive generation with DAS. The idea of using a generative prior to prune/guide the continuous search is sensible and novel in this space. 

• The training/evaluation pipeline is well illustrated, and the DAG projection routine is explicitly provided. 

• Empirical signal: Across ten IWLS’22 circuits, the approach reports substantial reductions in DAS steps and non trivial #NAND savings while keeping functional correctness. Scaling experiments and an ablation on using CircuitAR derived edge probabilities provide some evidence that larger models and the proposed initialization matter.

### Weaknesses
•	The main claims rely on 10 IWLS’22 circuits (Table 1–2, p.8). This is a small sample of the broader IWLS/EPFL/ISCAS landscape. 

•	There is no comparison to industrial strength flows (e.g., ABC/Yosys scripts such as resyn2) in terms of final area/depth/delay under a standard cell library—even as a reference. The paper frames #NAND and “search space” but not mapped area/timing—which is what designers ultimately care about.  

•	BitsD is measured as MAE of an untrained CircuitNN initialized by CircuitAR vs. truth table outputs. While intuitively appealing as a pre DAS proxy, the paper does not provide correlation analyses between BitsD and the final outcomes designers care about (e.g., DAS steps, final #NAND/area, or success under time limits). 

•	CircuitAR conditions on full truth tables via cross attention, but the paper does not describe how T is represented (row major? compressed? learned embeddings?). The training data caps PI/PO at ≤15, which is small compared to real designs; beyond this, truth table representations become exponential in PI. The method’s practical deployment arguably hinges on decomposition, but the paper does not implement or evaluate a decomposition→recombination pipeline. 

•	The “phase transition” at 2B parameters is based on four model sizes and two metrics (depth/#NAND), without uncertainty bands or tests for statistical significance. Also, Figures 4–6 (p.7) assert power law like improvements in BitsD, but again no error bars or replicates are shown. 

•	While Appendix B gives hyperparameters and hardware, the paper does not clearly state that code, models, and the circuit dataset will be released. Given the complexity and compute, artifact availability is essential for verification.

### Questions
1.	BitsD validation: Can you show correlation plots between BitsD and (a) DAS steps, (b) final #NAND / mapped area, (c) success under fixed time budgets across a large set? Does BitsD remain predictive when node budgets and idle gate ratios vary? 

2.	Truth table representation: How is T encoded for cross attention in CircuitAR?   

3.	Broader benchmarks: Can you include ABC/Yosys and report mapped area/timing under a standard cell library? Even if not directly comparable, this gives a reality check on hardware relevance.   

4.	Statistical rigor: Please provide error bars (multiple seeds) for Tables 1–2 and Figures 4–7; clarify stopping criteria and early stopping policies for DAS.   

5.	Search space definition: How exactly is the important metric, “Search Space” in Table 2, computed? A precise definition would help the community reproduce and interpret the gains.   

6.	Artifacts: Will you release code, data gen scripts, and pretrained models?

### Soundness
3

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
This paper presents an autoregressive approach to generate (raw) circuits from truth tables, which can serve as an initialization of differentiable architecture search to generate valid and exactly equivalent circuits. Firstly, an autoencoder model "CircuitVQ"  is trained to achieve interconversion between a circuit and discrete tokens. Then, an autoregressive model "CircuitAR" is trained to generate tokens from truth table. During the inference stage, given a truth table, the CircuitAR model transform it into tokens, and the decoder of CircuitVQ further transform the tokens into a reconstructed circuit (represented by an adjacency matrix). The matrix is further processed in a greedy way to make it valid (no cycles, etc), and serve as an initialization of CircuitNN for an exactly equivalent circuit. Experimental result shows that the performance of CircuitAR can scale with more parameters and tokens, and the proposed pipeline performs better than baseline approaches.

### Strengths
- The "truth table - discrete tokens - adjacency matrix" pipeline is novel and technically solid.
- The result shows that the performance of CircuitAR can scale with more parameters and tokens, which looks promising.

### Weaknesses
- The circuit size scalability of the proposed approach is questionable. No experimental result is shown about the circuit size scalability. The paper mentioned "a tenfold increase in circuit complexity compared to prior works", but it is not clear how this conclusion is drawn, and how the "circuit complexity" is measured (number of gates/PIs/POs?). It looks like all the circuits selected in table 1 and 2 are pretty small.
- The number of sample circuits for evaluation is very small. In table 1 and 2, only ten circuits are included. Even if there is a scalability issue, the test circuits can be extended by extracting sub-circuits from large benchmark circuits.

### Questions
- In line 161, $\ell_2(\cdot)$ seems to represent a $\ell_2$-normalized vector, rather than the $\ell_2$-norm of a vector. Maybe this can be mentioned in an explicit way to avoid confusion.

### Soundness
3

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
2

### Summary
This work presents an approach for circuit generation that combines conditional generative modeling with differentiable architecture search (DAS). The CircuitAR model produces initial circuits from truth tables which is optimized by DAS. The CircuitAR graph autoregressive model is trained with masked autoregressive training conditioned on truth tables that define the circuit functionality.

### Strengths
* Novel modeling approach of circuit generation from truth table using a masked autoregressive graph model.

### Weaknesses
* The primary utility of this work is reducing DAS search steps. This is a weaker utility compared to improving the quality of circuit optimization (fewer nodes or lower area/delay/power).
* Quality comparison versus classical logic optimization algorithms such as resyn2 is not presented. Prior work such as ShortCircuit presents such comparisons.

### Questions
* How does the overall logic optimization system with this approach compare to classic logic optimization algorithms such as resyn2, for various circuit sizes?

### Soundness
3

### Presentation
4

### Contribution
3
