# Continual Learning Knowledge Graph Embeddings for Dynamic Knowledge Graphs

- Avg Score: 4.75
- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 6, 3, 5

## Abstract
Knowledge graphs (KG) have shown great power in representing the facts for numerous downstream applications. Notice that the KGs are usually evolving and growing with the development of the real world, due to the change of old knowledge and the emergence of new knowledge, thus the study of dynamic knowledge graphs attracts a new wave. However, conventional work mainly pays attention to learning new knowledge based on existing knowledge while neglecting new knowledge and old knowledge should contribute to each other. Under this circumstance, they cannot tackle the following two challenges: (C1) transfer the knowledge from the old to the new without retraining the entire KG; (C2) alleviate the catastrophic forgetting of old knowledge with new knowledge. To address these issues, we revisit the embedding paradigm for dynamic knowledge graphs and propose a new method termed \textbf{C}ontinual \textbf{L}earning \textbf{K}nowledge \textbf{G}raph \textbf{E}mbeddings (\textbf{CLKGE}).  In this paper, we establish a new framework, allowing new and old knowledge to be gained from each other. Specifically, to tackle the (C1), we leverage continual learning to conduct the knowledge transfer and obtain new knowledge based on the old knowledge graph. In the face of (C2), we utilize the energy-based model, learning an energy manifold for the knowledge representations and aligning new knowledge and old knowledge such that their energy on the manifold is minimized, hence can alleviate catastrophic forgetting with the assistance of new knowledge. On top of this, we propose a theoretical guarantee that our model can converge to the optimal solution for the dynamic knowledge graphs. Moreover, we conduct extensive experimental results demonstrating that CLKGE achieves state-of-the-art performance compared with the embedding baselines.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The authors propose the CLKGE method to address the incremental update and catastrophic forgetting problems in the representation learning of dynamic knowledge graphs. The proposed method is provided with a physical interpretation and a proof of convergence. Experimental results show that CLKGE can outperform existing methods.

### Strengths
1.The authors propose a unified framework to achieve the incremental update of embeddings and alleviating the catastrophic forgetting for dynamic knowledge graph representation learning.
2.The technical design of the proposed method seems reasonable and the experimental results demonstrate its effectiveness.
3.The convergence analysis of CLKGE is provided.

### Weaknesses
1. The effectiveness of g in alleviating the distribution gaps needs more explanations, maybe more mathematical proof.
2. Eq (3) seems very like a GNN aggregation, more analysis should be provided.
3. The parameter sensitivity of the proposed model should be studied.

### Questions
None

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes CLKGE for the task of dynamic knowledge graphs embeddings, which can allow new knowledge and old knowledge to gain from each other. In particular, to transfer knowledge from the existing to the novel without necessitating the retraining of the entire knowledge graph, this paper employs continual learning for knowledge transfer. To mitigate the problem of catastrophic forgetting when incorporating new knowledge alongside the old, the paper utilizes the brachistochrone curve to model the associations. The extensive experimental results presented in the study demonstrate that CLKGE attains state-of-the-art performance.

### Strengths
1. This paper has a clear structure and strong logical coherence. The language used in the paper is fluent, and the tables and figures in the experimental section are clear and complete. 
2. The authors propose a unified framework to achieve the incremental update of embeddings and alleviating the catastrophic forgetting for dynamic knowledge graph representation learning.
3. This paper provides comprehensive mathematical proofs for the proposed model.

### Weaknesses
1.	This paper introduces the Energy-based Model, but I am unclear about the necessity of this introduction. I do not understand why the paper did not opt for a purely mathematical optimization approach to address the problem, and instead introduced the Energy-based Model, which is a physical model.
2.	All experiments in this paper are conducted based on FB15K-237. It might be beneficial to include additional datasets to enhance the generalizability and credibility of the conclusions.

### Questions
1.	Can you add more datasets?
2.	Can you explain the necessity of introducing the Energy-based Model?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper studies the representation of dynamic knowledge graphs and tries to combine continual learning with dynamic graphs. The authors attempt to solve two challenges: 1) transfer the old knowledge to new input, and 2) alleviate the catastrophic forgetting problem. Therefore, they propose a method to conduct continual learning and utilize the energy-based model to align new knowledge and old knowledge such that their energy is minimized. The experiments are conducted on one knowledge graph.

### Strengths
1. The paper provides a comprehensive literature review.
2. Many related works have been compared in the experiments.

### Weaknesses
1. The definition of dynamic knowledge graphs is vague and inaccurate. The authors restrict that the dynamic knowledge graphs only add new entities, relations, and triples during evolution. However, some old entities, relations, and triples would be removed in dynamic knowledge graphs. In addition, the authors assume that at each snapshot, the added entities, relations, and triples are all new, this scenario is hard to find in the real world in which most cases are that some new entities and old relations are linked and new relations and old entities are linked. This paper focuses on the dynamic knowledge graphs but only considers a very rare scenario. 

2. It is hard to find out how embedding transfer learns the representation of new knowledge.

3. The motivation for using EBM is not clarified. The benefits of EBM are not explained.

4. The writing needs further improvement. The methodology part only introduces what are the designs without clear explanations.

5. The setting of continual learning and the setting of dynamic graphs are not the same. The authors should at least discuss the differences.

### Questions
1. The authors claim that only the new entities and relations are added to the existing knowledge graph, it is not clear why Eq. (3) makes use of the previously learned representation $r_{i-1}$, $h_{i-1}$, and $t_{i-1}$ and how to obtain these representation before seeing the new entities and relations? What will happen if a new entity is linked to another new entity by a new relation?

2. No definition for the function $f$ in Eq. (5).

3. It is not clear why the EBMs are effective in controlling the representational shift and why the energy of new knowledge and old knowledge should be minimized.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The author tries to address two challenges in dynamic KG: knowledge transfer and knowledge retention. The proposed method included two components - a embedding evolution module based on embedding functions, and a knowledge retention method based on energy-based model.

### Strengths
S1. The author proposed to leverage EBM method on KG embedding learning. It is a new direction that worth further investigating.

S2. Despite minor typos, the math is sound throughout the paper. It is also good to see the theoretical proof on the convergence of the training. 

S3. The experiment is thorough and provides insight into the performance of the CLKGE. It included important baselines on KG embedding methods.

### Weaknesses
W1. Lack of related works regarding current research on temporal knowledge graphs. The setting is close enough to be included. For example, recently Xu et. al (https://arxiv.org/abs/2305.07912) leverages large language model to tackle the temporal knowledge graph setting. Jung et. al (https://arxiv.org/abs/2012.10595) considers the relative displacement timestamps and uses an attention network to model it. 

W2. Section 3.1 is very confusing… The two steps described are not clear. 1) How does old knowledge embedding update the knowledge representations? 2) How do you learn new knowledge with old entities? And why does that transfer from old to new?

W3. The ablation study result is concerning. I am more interested to see the full results on different dataset. The transfer method might have more negative impact on some of the dataset.

### Questions
Q1. Typo in definition of CL? why not italicize e_{i-1}?
Q2. What does sum_{i-1}_{j=1} mean? where does i start?
Q3. Can you provide more intuition behind your formulation when presenting the section 3.1? 
Q4. Experiments: Can you give definition of FWT and BWT?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
3 good
