# A Formal Comparison Between Chain-of-Thought and Latent Thought

- Avg Score: 2.67
- Decision: Reject
- Scores: 2, 2, 4

## Abstract
Chain-of-Thought (CoT) elicits reasoning in large language models by explicitly generating intermediate steps in natural language. In contrast, Latent Thought in looped models operates directly in the continuous latent space, enabling computation beyond discrete linguistic representations. While both approaches exploit iterative computation, their comparative capabilities remain underexplored. In this work, we present a formal analysis showing that Latent Thought in looped Transformers enables parallel computation, which is more efficient than the inherently sequential process of CoT. In contrast, CoT leverages stochastic decoding to approximate solutions to problems where exact computation is intractable. These separations suggest the tasks for which depth-driven recursion is more suitable, thereby offering practical guidance for choosing between reasoning paradigms.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper compares two reasoning paradigms in large language models — Chain-of-Thought (CoT) and Latent Thought implemented via Looped Transformers. It provides a formal theoretical analysis showing that Looped Transformers can perform efficient parallel computation by operating in latent space, while CoT reasoning, being sequential, excels in stochastic or approximate problem solving.

### Strengths
he paper makes a valuable theoretical contribution by precisely linking CoT and Looped Transformers to formal complexity classes, providing a solid mathematical foundation to compare reasoning in latent versus linguistic space.

### Weaknesses
The main limitation lies in lack of conceptual clarity and practical interpretability. The theoretical analysis is rigorous, the motivation and intuition for non-specialist readers however are weak — the connection between abstract complexity classes (e.g., TCk, ACk) and practical model behaviors remains opaque. The formalism is heavy and often repeats known results under new notation rather than offering fresh insights. 

Most importantly!!!!1 the experimental validation is **very very narrow and superficial**, relying on small-scale, synthetic tasks that do not convincingly demonstrate real-world implications. The comparison between “loops” and “steps” lacks fairness in terms of computational cost and hardware efficiency, which undermines the practical value of the claimed separation. The assumptions about “log-precision” and “polynomial embedding size” are idealized and may not hold in realistic settings. 

Last, 
while the paper claims to reveal “complementary strengths,” it fails to discuss how these insights can inform model design or prompt engineering, leaving the contribution mostly theoretical and of limited immediate impact.

### Questions
see above

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper investigates the concept of latent thought and attempts to provide a formal analysis demonstrating that latent thought emerges in Looped Transformers. It seeks to explore the distinction between Chain-of-Thought (CoT) reasoning and latent thought within this framework. The authors approach the problem by formalizing deterministic computations as graph evaluations and conclude that latent thought facilitates efficient parallel computation, whereas CoT supports randomized approximation.

Overall, the paper suffers from unclear writing and lacks a well-defined objective and motivation. The theoretical analysis is not logically structured, making it difficult to follow. The experimental design is also weak, providing unconvincing results and offering limited insights.

### Strengths
1. The explored problem of investigating the gap between Chain-of-Thought (CoT) and latent thought is somewhat useful for developing effective methods of enhancing latent CoT reasoning.

### Weaknesses
1. The paper writing is unclear, lacking a well-defined objective and motivation. 
2. The theoretical analysis is not logically structured, making it difficult to follow. 
3. The experimental design is weak, providing unconvincing results and offering limited insights.

### Questions
1. Beyond the Looped Transformer architecture, can the proposed theoretical analysis be generalized to other latent CoT structures?
2. The theoretical analysis is poorly presented. What are the connections among the proposed theorems and lemmas?
3. How does the experimental design answer the proposed research question in the introduction?

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents a formal comparison between Chain-of-Thought (CoT) reasoning and Latent Thought in Looped Transformers (Looped TFs). Through theoretical analysis and controlled experiments, the authors demonstrate that Looped TFs enable efficient parallel computation in deterministic settings, whereas CoT excels in stochastic approximation tasks. The work establishes asymptotic separations between the two paradigms, suggesting when each reasoning strategy is more suitable.

### Strengths
- **Clear theoretical framing:** The paper provides a rigorous and formal comparison between CoT and Looped TFs, filling an underexplored gap in reasoning analysis.  
- **Insightful comparative analysis:** The asymptotic and empirical results jointly clarify when each reasoning paradigm is advantageous.  
- **Strong motivation:** The study addresses an important and timely question in understanding reasoning mechanisms in LLMs.

### Weaknesses
- **Simplifying assumptions:** The theoretical setting (deterministic vs. stochastic) abstracts away several practical factors that may affect real model behavior.  
- **Clarity issues:** Some formal sections (e.g., proof sketches and separation definitions) could use more intuitive explanation or visualization.

### Questions
- **Assumption Robustness:**  The theoretical analysis relies on several strong assumptions (e.g., polynomial-size graphs, log-precision). How sensitive are the conclusions to these assumptions? Would relaxing them change the separation results?
- **Empirical Validation:**  The experiments (e.g., DAG and DNF counting) are highly synthetic. Can the authors demonstrate results on realistic reasoning benchmarks to confirm the claimed separation holds empirically?

### Soundness
3

### Presentation
3

### Contribution
2
