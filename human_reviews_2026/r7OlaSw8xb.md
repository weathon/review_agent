# MCCE: A Framework for Multi-LLM Collaborative Co-Evolution

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
Multi-objective discrete optimization problems, such as molecular design, pose significant challenges due to their vast and unstructured combinatorial spaces. Traditional evolutionary algorithms often get trapped in local optima, while expert knowledge can provide crucial guidance for accelerating convergence. Large language models (LLMs) offer powerful priors and reasoning ability, making them natural optimizers when expert knowledge matters. However, closed-source LLMs, though strong in exploration, cannot update their parameters and thus cannot internalize experience. Conversely, smaller open models can be continually fine-tuned but lack broad knowledge and reasoning strength. We introduce Multi-LLM Collaborative Co-evolution (MCCE), a hybrid framework that unites a frozen closed-source LLM with a lightweight trainable model. The system maintains a trajectory memory of past search processes; the small model is progressively refined via reinforcement learning, with the two models jointly supporting and complementing each other in global exploration. Unlike model distillation, this process enhances the capabilities of both models through mutual inspiration. Experiments on multi-objective drug design benchmarks show that MCCE achieves state-of-the-art Pareto front quality and consistently outperforms baselines. These results highlight a new paradigm for enabling continual evolution in hybrid LLM systems, combining knowledge-driven exploration with experience-driven learning.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes MCCE, a framework that unites a frozen, closed-source LLM (for global exploration) with a smaller, trainable open-source LLM (for local adaptation) in a collaborative co-evolution setup. The two models alternate between generating and refining candidate solutions, with the small model updated using DPO based on “breakthrough” trajectories. The system aims to combine reasoning strength and adaptability, achieving improved performance on multi-objective molecular design tasks such as drug discovery.

### Strengths
This paper addresses an emerging direction in hybrid LLM collaboration by demonstrating the strengths of large frozen models and fine-tunable smaller models.

Strong results on multi-objective optimization benchmarks with quantitative evidence (e.g., hypervolume and diversity metrics).

### Weaknesses
1. Key elements such as the feedback exchange protocol, update frequency, and data flow between models are only briefly described. Without algorithmic pseudocode or concrete update rules, reproduction is difficult.

2. Evaluation is limited to a single domain (molecular design). Either broader multi-domain testing (e.g., combinatorial optimization or symbolic reasoning) or re-framing the title and introduction to emphasize domain specificity would make the contribution more accurate and credible.

3. Claims that MCCE constitutes a “general framework for collaborative reasoning” are not substantiated by the experiments, which focus solely on molecular tasks. The paper would benefit from tempering these claims or providing stronger cross-domain evidence.

4. There is no detailed analysis isolating the contributions of each design component—e.g., DPO fine-tuning, trajectory selection, or mutual feedback.

5. The paper does not discuss computational cost, scalability, or stability of the co-evolution loop, which are critical for practical adoption.

6. Missing references on multi-agent collaboration:

[1] Collabllm: From passive responders to active collaborators.

[2] From LLM-anation to LLM-orchestrator: Coordinating Small Models for Data Labeling 

[3] Collab-RAG: Boosting Retrieval-Augmented Generation for Complex Question Answering via White-Box and Black-Box LLM Collaboration

[4] Many Heads Are Better Than One: Improved Scientific Idea Generation by A LLM-Based Multi-Agent System

### Questions
1. Could you elaborate on the exact information exchange mechanism between the large and small models—does the frozen model adapt its generation strategy based on feedback, or is the communication one-way?

2. What is the update schedule between the large and small models? Are updates synchronous after each generation cycle or asynchronously buffered?

3. Have you tested MCCE in non-molecular domains (e.g., code synthesis, symbolic reasoning, or text generation) to assess generalizability?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes MCCE, a hybrid optimization framework that couples a frozen, closed‑source LLM (for global exploration) with a lightweight, trainable local LLM (for experience‑driven exploitation). The system keeps a trajectory memory and periodically fine‑tunes the local model using a Direct Preference Optimization (DPO) scheme. A key design is a similarity‑based data synthesis procedure that forms stable preference pairs for DPO by filtering generated molecules using global similarity statistics and score quantiles. Core contributions claimed: (i) a collaborative co‑evolution framework that lets a frozen API LLM and a trainable local LLM “mutually inspire” each other; (ii) an experience‑driven learning paradigm via DPO with similarity‑aware triplet construction; (iii) empirical gains on a five‑objective drug design benchmark, extending prior three‑objective setups.

### Strengths
1. Timely hybrid design that leverages a frozen API LLM for wide exploration and a trainable local LLM for learned exploitation.  

2. Similarity‑aware triplet construction with global statistics and progressive windows

### Weaknesses
1. sim(c,q) between a molecule and a prompt is not formally defined; the text proposes fingerprint‑based metrics but those are molecule‑to‑molecule.  

2. Lack of detail about the multi‑objective selection operator undermines interpretability of diversity and HV results. 

3. Add GFlowNet numbers and standard EA baselines under identical objectives; include recent LLM‑EA baselines (e.g., MoLLEO/ExLLM reproductions with your five‑objective setup).

4. Report API token counts, calls per generation, and cost vs. HV curves. Consider a budgeted setting where total API calls are capped; show MCCE’s advantage under realistic constraints.

### Questions
see above

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes Multi-LLM Collaborative Co-evolution (MCCE), a hybrid framework for multi-objective discrete optimization. The core idea is to combine a powerful, but frozen, closed-source LLM (e.g., GPT-4) with a smaller, trainable, open-source LLM. The frozen LLM acts as a global explorer, while the local model is progressively fine-tuned on "breakthrough" search trajectories to internalize experience and perform more targeted, experience-driven learning. Furthermore, a more stable, similarity based version of DPO is presented for RL based training.

### Strengths
● Paper is well written and easy to understand - great use of diagrams and figures ● SOTA results on the multi-objective showing the benefit of co-evolving LLMs vs closed-source LLMs alone 
● Similarity based DPO is a well thought out method for avoiding training instability and ensuring training on structural meaningful pairs

### Weaknesses
● Limited comparison to prior work. the benchmarking is done against the closed source LLM and the trainable model, however, the method is not compared against methods such as MoLLEO. 
● The paper restricts its experiments to molecular design and fails to show the benefit of co-evolving LLMs in other discrete optimization domains. 
● Hyperparameters could be ablated to study the effect of values such as alpha or the intervals for similarity.

### Questions
● The paper states the operator alternates between the frozen and local LLMs. Is this split 50/50? Is it fixed or adaptable? 
● How crucial is the Tanimoto similarity metric for similarity based DPO? Have you explored alternative, simpler, or non-domain-specific similarity functions (e.g., embedding-based similarity)?

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
3

### Summary
This paper studies multi-objective molecular optimization with LLMs. Classic evolutionary algorithms often converge prematurely and lose diversity; single-LLM optimizers can stagnate and, if frozen, cannot absorb experience. The proposed MCCE framework pairs a powerful frozen closed-source LLM (broad exploration) with a trainable lightweight local LLM (experience-driven adaptation). The two alternate as generators in an evolutionary loop; “breakthrough” trajectories are logged and used to continually refine the local model so the pair co-evolves rather than simply distilling one into the other. For training the local model, the authors compare SFT and RL, finding SFT induces catastrophic forgetting (hurting uniqueness) and RL is unstable with scalar rewards. They instead adopt DPO with a similarity-based preference construction that forms (prompt, preferred vs. rejected) pairs from structurally comparable molecules, improving stability and data efficiency. Empirically, MCCE achieves state-of-the-art hypervolume and consistently outperforms single-model and co-evolution baselines using SFT or RL. DPO-based parameter training is key for long-horizon gains; both the local and frozen components benefit (fitness/diversity and exploration), and score distributions shift upward after co-evolution.

### Strengths
1. Combining a frozen, high-capacity API model for exploration with a trainable local model for exploitation/learning is well-motivated and practically appealing.

2. The DPO + similarity-based pair construction is a neat way to stabilize preference learning without expensive curated labels.

### Weaknesses
1. The paper claims they have provide the code however, I do not find the link to the code. 

2. How is the synthesizability metric?

3. What is the training cost?

### Questions
Please see the Weaknesses section.

### Soundness
3

### Presentation
3

### Contribution
3
