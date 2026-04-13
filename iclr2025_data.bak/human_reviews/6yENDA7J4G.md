## Human Reviewer 1

### Summary
This paper takes an early step to leverage foundation models for solving Mixed Integer Linear Programming (MILP), which plays an important role in real-world applications. Specifically, it studies three important tasks, including two typical tasks integrality gap prediction and learning to branch, and one proposed task of aligning MILP instances with natural language descriptions. Compared with previous works, it emphasizes generalization performance across problem classes, and proposes an LLM-based data augmentation framework named MILP-Evolve to generate diverse problem classes for the training of models. Experimental results demonstrate that the proposed MILP-Evolve can generate diverse data, and improve the overall performance of pretrained models on the three studied tasks.

### Strengths
1. Even that there have been a variety of works on leveraging LLMs to solve complex decision-making problems, to the best of my knowledge, this is the first work that focus on LLM-based training data augmentation in the field of learning to solve MILP. In experiments, the proposed MILP-Evolve show great capacity to generate diverse problems and improve the generalization performance of the trained models.
2. This paper is well-written, with rich technical details of the proposed MILP-Evolve. I am convinced that such a new open-source and powerful data argumentation method can benefit the community of learning to solve MILP.

### Weaknesses
1. As the idea of the proposed MILP-Evolve, which prompts the LLMs to generate diverse data under an evolution framework, is straightforward and not new, I am concerned that the technical insights of this paper are limited.

### Questions
1. What about the cost of data generation in the experiments? As the running of such a LLM-based data generation process may be very expensive, will the generated problem classes be collected and open-sourced together? 

2. How do you envision the practical applications of the newly proposed task of aligning MILP instances with natural language descriptions in the MILP solving process? Can you discuss more on it?

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
3

---

## Human Reviewer 2

### Summary
This paper introduces MILP-Evolve, a novel LLM-based evolutionary framework designed to generate a large and diverse set of MILP instances. This paper evaluates the proposed method on three key learning tasks relevant to MILP: integrality gap prediction, learning to branch, and a new task of aligning MILP instances with natural language descriptions.

### Strengths
1. The proposed data generation method is validated on three MILP-related learning tasks.
2. The paper is well-structured and presented clearly.

### Weaknesses
1. The methodological contribution is somewhat limited, primarily offering a data augmentation approach that employs LLMs to generate diverse MILP instances.
2. There is a mismatch between the content and title of the paper. The title “Towards Foundation Models for Mixed Integer Linear Programming” suggests a broader scope, while the paper mainly discusses a data generation method for MILP.

### Questions
1. The fairness of experiments comparing the proposed data generation method with others is unclear. The paper mentions a 7:1:2 split of generated MILP problem classes into training, validation, and test subsets, raising concerns that the test data distribution may resemble that of the training data. In contrast, other methods likely produce differently distributed training data, which could skew comparisons and inflate the performance of MILP-Evolve.
2. For the task of aligning MILP instances with natural language descriptions, what are the specific formats for both the MILP instances and the textual descriptions? What are the sources of these instances and descriptions? Will all elements in each set be matched one-to-one, and does this task hold practical significance?
3. How should Figure 1b be interpreted?
4. The meaning and context of the “Mean” baseline used in the experiment is unclear.

### Soundness
3

### Presentation
4

### Contribution
2

### Rating
6

### Confidence
4

---

## Human Reviewer 3

### Summary
The paper explores the potential of foundation models for Mixed Integer Linear Programming (MILP), introducing a novel framework called MILP-Evolve that leverages large language models (LLMs) to generate diverse MILP instances. The authors apply this framework to three distinct tasks: (1) integrality gap prediction, (2) learning to branch, and (3) a new task of aligning MILP instances with natural language descriptions. While promising empirical results are shown, especially in generalizing across different MILP classes, some aspects—particularly the Language-MILP Contrastive Learning task—require further clarification regarding its practical significance and feasibility.

### Strengths
- **Diversity of MILP Generation**: The MILP-Evolve framework introduces an innovative approach to generating diverse MILP problem classes, which has the potential to enhance generalization in ML-based MILP solvers.
- **Empirical Performance**: The paper demonstrates strong performance improvements on the integrality gap prediction and learning to branch tasks, providing evidence that the proposed approach can generalize to unseen MILP classes.
- **Extensive Experimental Work**: The paper presents a substantial amount of experimental results across various tasks, demonstrating the significant effort in evaluating the proposed methods. The authors cover a wide range of experiments, which showcases the robustness of their approach.

### Weaknesses
1. **Practical Application of Language-MILP Contrastive Learning**: The Language-MILP Contrastive Learning task is positioned as a way to assist non-experts in understanding and formulating MILPs. However, the generated natural language descriptions tend to emphasize technical details (e.g., linear constraints, integer variables), and it is not entirely clear how this helps users grasp the real-world significance of MILP problems. It would be helpful if the authors could provide more clarification on how aligning these mathematical descriptions with natural language assists in bridging the gap between abstract optimization models and their practical applications. Including more concrete examples or case studies could further reinforce this task’s practical relevance.

2. **Language Quality**: The natural language descriptions used in the Language-MILP Contrastive Learning task are generated by LLMs from solver code (e.g., SCIP, Pyomo, Gurobi). There may be some concerns regarding the quality of these descriptions. If the language samples were curated by human experts, this task could capture valuable domain-specific insights. However, relying solely on LLM-generated descriptions raises questions about the meaningfulness of the alignment. It might be worth considering how this task compares to having an LLM directly interpret a new MILP, as it is not immediately clear whether the proposed approach would outperform such a method. Clarifying the value added by this task would strengthen the paper.

3. **Disconnect Between Tasks**: While the paper introduces multiple tasks, the connection between them could be better articulated. For instance, the relationship between the Language-MILP Contrastive Learning task and the Multi-Class Learning task is not immediately clear, which might make the paper seem somewhat disjointed. A clearer explanation of how these tasks fit together within the broader scope of MILP optimization, particularly how Language-MILP Contrastive Learning complements the other optimization tasks, would improve the cohesion of the work.

4. **Comparative Experiments**: The comparison between the proposed method and works like ACM-MILP in the experiments might benefit from some adjustments. The current experimental setup involves:
   - Using problems generated by MILP-Evolve based on 8 seed MILP classes to create a large number of new problem types.
   - Using problems generated by ACM-MILP, which also learns and generates problems based on the same 8 seed MILP classes.

   Both sets of problems are then used as training data for a downstream ML-based MILP optimization framework, and the models are tested on other MILP classes generated by MILP-Evolve. However, existing MILP generation frameworks typically aim to enhance performance within a specific type of MILP class. Therefore, I suggest the following alternative comparative experiments:

   **Experiment 1:** Compare the models trained on problems generated by:
   - MILP-Evolve, which generates a large number of new problem types based on the 8 seed MILP classes.
   - ACM-MILP, which learns and generates problems based on the same 8 seed MILP classes.

   Then, test the trained MILP optimization frameworks on the same 8 seed MILP classes used by both MILP-Evolve and ACM-MILP. This would allow for a more direct comparison within the shared MILP classes.

   **Experiment 2:** Select a set of MILP problems generated by MILP-Evolve or from MIPLIB as seed MILP classes. Compare the models trained on problems generated by:
   - Problems generated by MILP-Evolve based on the selected seed MILP classes.
   - Problems generated by ACM-MILP based on the same selected seed MILP classes.

   After training, test both frameworks on the selected seed MILP classes to directly compare their performance.

   These alternative experimental designs would provide a more balanced comparison, as they ensure that both approaches have access to similar training data. This could help avoid potential biases in the current experimental setup, where ACM-MILP might be disadvantaged by the absence of instances from the test problem classes in its training set.

5. **Overclaim in Contribution**: The paper states that it achieves “Substantial Multi-Class Learning Gains Across All Tasks,” but the results presented primarily focus on integrality gap prediction and learning to branch. Since there are no substantial results or experiments demonstrating gains in the Language-MILP task, this claim could be seen as somewhat overstated. The authors could either provide additional results for the Language-MILP task or rephrase the contribution to more accurately reflect the scope of the work.

6. **Impact of Seed Class Selection**: The choice of seed classes in the MILP-Evolve framework likely has an important influence on the distribution and diversity of the generated MILP classes. However, the paper does not delve deeply into this aspect or provide an experimental evaluation of how seed class selection affects the generated instances. Including an analysis of how different seed classes influence the diversity and quality of the generated MILP instances, and whether certain seed classes lead to better generalization in the optimization tasks, would help strengthen the paper’s claims regarding the versatility of MILP-Evolve.

### Questions
1. Could the authors further clarify the real-world impact of the Language-MILP task? Specifically, how does aligning natural language descriptions with MILPs help non-experts understand and solve optimization problems?
2. Given that the language samples are generated by LLMs from solver code, how do the authors ensure the quality of these samples? Would human-generated descriptions lead to better learning outcomes in this task?
3. How does the Language-MILP Contrastive Learning task connect to the other tasks in the paper, such as integrality gap prediction and learning to branch? Could the authors provide more insights into the overall coherence of the tasks?

### Soundness
2

### Presentation
2

### Contribution
2

### Rating
6

### Confidence
4

---

## Human Reviewer 4

### Summary
This paper considers a novel dataset generation method for learning to solve mixed integer linear programming (MILP), leveraging the large language model (LLM). Given an input MILP instance, this method combines the evolution algorithm and parameter search to compute diversified new instances. The authors consider three tasks: (1) predicting the integrality gap. (2) learning to branch and (3) Aligning MILP problems with natural language to help non-experts.

The authors then tested their method on a dataset called SEED, gathered from the recent popular deep learning for MILP papers. The results showed that their method outperformed all other baselines. Moreover, the attention used on the variables can further improve the transferability.

### Strengths
1. Employing LLM to generate diversified MILP instances is novel and helpful for training a foundation model for MILP. The entire MILP space is too huge, so datasets created by humans can only cover a part of it. So, leveraging the power of LLM is a good direction.

2. The authors' commitment to open-source the entire framework is valuable for the entire community.

### Weaknesses
1. The dataset test seems to be not that "unseen." You mentioned that you collected MILP problems from eight classes. But you randomly split them after the augmentation. Then, the trained model still learned from all these eight classes. So it would be great if you only use six classes for training, 1 for validation, and 1 for testing. Then this can further show the power of your method.

### Questions
1. See the weaknesses (1)

2. One more interesting experiment is to fix the number of training data for your method and the baselines. To be more specific, let N be the number of instances of SEED. Then, we randomly take N / 10 data and use Evolve to generate N instances and call them dataset B. Then, training directly on SEED and this B can further show the power of your model.

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
3