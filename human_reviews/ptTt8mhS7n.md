# In-Context Transfer Learning: Demonstration Synthesis by Transferring Similar Tasks

- Decision: Reject
- Scores: 5, 3, 6, 6

## Abstract
In-context learning (ICL) is an effective approach to help large language models (LLMs) adapt to various tasks by providing demonstrations of the target task. Considering the high cost of labeling demonstrations, many methods propose synthesizing demonstrations from scratch using LLMs. However, the quality of the demonstrations synthesized from scratch is limited by the capabilities and knowledge of LLMs. To address this, inspired by transfer learning, we propose In-Context Transfer Learning (ICTL), which synthesizes target task demonstrations by transferring labeled demonstrations from similar source tasks. ICTL consists of two steps: source sampling and target transfer. First, we define an optimization objective, which minimizes transfer error to sample source demonstrations similar to the target task.  Then, we employ LLMs to transfer the sampled source demonstrations to match the definition and format of the target task. Experiments on Super-NI show that ICTL outperforms synthesis from scratch by 2.0% on average, demonstrating the effectiveness of our method.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper introduces In-Context Transfer Learning (ICTL), a method to enhance in-context learning by synthesizing examples for new target tasks through the transfer of demonstrations from similar tasks. Unlike conventional synthesis methods constrained by the limitations of LLMs, ICTL leverages labeled demonstrations from related source tasks. The approach involves selecting source tasks closely aligned with the target task and then refining these examples through LLMs to match the target. Experimental results on the Super-NI dataset show that ICTL achieves a 2% average performance improvement over other baselines.

### Strengths
- ICTL combines ICL and Transfer Learning, presenting a simple yet effective method for ICL across different tasks, where  target demonstrations are strictly limited.

- ICTL achieves consistent improvement on the Super-NI dataset, with a 2% average performance gain over other baselines.

- The paper is written in a clear manner, making it easy to understand the proposed method and its applications.

### Weaknesses
- While the proposed ICTL combines the concepts of Transfer Learning and ICL, its contribution is not clearly specified compared to prior research involving advanced demonstration generation in zero-shot ICL settings and advanced retrieval methodologies for ICL with RAG. 

[1] SELF-ICL: Zero-Shot In-Context Learning with Self-Generated Demonstrations (EMNLP 2023)

[2] Demonstration Augmentation for Zero-shot In-context Learning (ACL 2024)

[3] Bridging Distribution Gap via Semantic Rewriting with LLMs to Enhance OOD Robustness (ACL 2024)
   
- The selected baselines used in the experiments mostly correspond to ablations of the proposed method. For a fairer evaluation, it is necessary to directly compare ICTL with similar approaches discussed in the related work. Specifically, the authors could compare with methodologies that rely on LLMs to generate demonstrations for ICL; e.g., [1, 2] or those that modify source task formats to enhance transfer learning; e.g., [3]. These comparisons could state the contributions and novelty of the proposed ICTL clearly. 

- While restructuring the retrieval process based on a Transfer Learning objective is a positive step, the generation of new target demonstrations relies only on LLM prompting and self-verification, which can be insufficient to fundamentally improve task performance. Recent studies such as [4] claim that LLMs are limited in their ability to self-correct reasoning errors. To position target demonstration generation as a significant contribution, the paper needs to further enhance this aspect beyond sampling by proposing novel approaches to optimize the generation process.

[4] LARGE LANGUAGE MODELS CANNOT SELF-CORRECT REASONING YET (ICLR 2024)

- ICTL involves multiple steps, including sampling, transferring, and verifying demonstrations, which may increase computational demands compared to simpler in-context learning approaches. Considering the use of high-cost and high-quality models like GPT-4o and Llama3.1-8b, along with the extra inference required for demonstration generation and self-verification within ICTL, the 2%
average performance improvement seems less significant to justify the increased cost.

- If there are few or no closely matching source tasks, the quality of transferred demonstrations may suffer, impacting the model’s ability to generalize to the target task.

- The Super-NI dataset used to demonstrate performance improvement is somewhat limited. In Section 1, the authors claim that their approach aims to enhance performance on tasks where the LLM lacks prior knowledge. However, it is hard to argue that the tasks or knowledge in the Super-NI dataset are entirely absent from GPT-4o or Llama3.1-8b-Instruct models. To provide more convincing validation, the authors should either use a more up-to-date benchmark or design tasks in environments like Alfworld [5] for additional testing.

[5] ALFWorld: Aligning Text and Embodied Environments for Interactive Learning (ICLR 2021)

### Questions
- The experiments in the paper focus solely on final task performance, with no analysis of generated target demonstration quality. It would strengthen the paper’s contributions if more analysis was presented. Could the authors provide example target demonstrations to illustrate the model’s limitations and how they were improved? 

- How does ICTL perform when there were no closely related source tasks available? Experimental results on how ICTL performs when there are limited or no closely related source tasks available would be beneficial.

- How does the computational cost of ICTL compare to traditional ICL?

- Could the authors compare ICTL with recent works on learning-based retrievers in LLM RAG approaches instead of Direct?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper proposes a method to improve ICL (in-context learning) by integrating principles from transfer learning. Specifically, it finds similar demonstrations from another task, using distributional similarity between tasks, to copy into the ICL prompt. It finds very modest increases in ROUGE score on the Super-NI benchmark.

### Strengths
1. The proposed idea seems reasonable at a high level.

2. Results show some promise of proposed method.

### Weaknesses
I strongly believe this paper, in its current form, does not meet the threshold for ICLR acceptance.

1. It is a bit hard to determine novelty of this work. Authors cite the previous main ICL paradigm of generating synthetic examples from a single human-provided example, but do not compare to prior methods that combine transfer learning, or similarity of examples, to do in-context learning. For instance, Dr.ICL (Demonstration-Retrieved ICL; Luo et al 2023), seems to be retrieving demonstrations that are similar to the task at hand. Thus, Luo et al seems relevant but was not mentioned or compared against.

2. Much of the paper language needs editing. There are issues with writing in most paragraphs.

3. Figure 2 is not particularly informative after Figure 1. Additionally, the examples in Figures 1 and 2 do not make much sense. For instance, in Figure 1, most of the “Definitions” boxes do not make sense (they are not well defined sentences), and the “Question” and “Output” pairs do not make sense either--the output is “Yes” when the question is not a yes/no question. Therefore, it confuses the reader about how the proposed transfer step would help ICL.

4. There are serious coherence issues in Section 3: Method. In Section 3.1: “...We define similarity as, if the sampling scale is N, the N demonstrations most similar to the target task are called similar...” Isn’t this a recursive definition? How are the most similar N demonstrations computed? Variables in 3.1.1 are not well defined, which is a serious issue as this is the only section in the paper with math. What is a hypothesis? What is a task error? What does the “distribution for each task” mean? How are tasks defined statistically/numerically? Where is this task error term used in the method?

5. Evaluations: Table 1 does not seem to show Super-NI to be a challenging task. There is not much of a difference, especially with GPT-4o, on zero-shot and proposed method (68.7 vs 73.1 ROUGE score), as acknowledged by authors. The gap is 52.0 vs 60.3 with Llama3.1-8b. Given the small gap, I would recommend authors to choose a more challenging task where ICL is much more crucial to the model’s success, so that any differences are less likely to be due to noise.

6. Table 2 has a serious typo/weird result. Why is “Source Sample” row, “Rouge” column listed as 43.7 (-3.5), even though 43.7 should be a decrease of -16.6, instead of -3.5? The fact that this number is so low compared to the other entries in the column makes it seem like it is a typo.

7. ROUGE is not necessarily a good metric, depending on the task and dataset answer length. A better one would be expert human evaluations (preferably pairwise evaluations between the methods). If authors do not have the resources for human eval, there are better automated metrics available.

8. All of the results discussion about Figure 3, from Sections 4.4.1 through 4.4.4, lack insight, provide no analysis, and are just restating the graph trends in many words. The redundancy of information throughout this paper is prevalent in other sections as well, not just these sections.

9. Section 4.4.1: “Even transferring using one source demonstration can also effectively improve the performance of the target task. This is because: (i) Even using one single source demonstration, we can also synthesize a large amount demonstrations [sic]  ... (ii) ... even without source demonstrations, LLMs can still synthesize demonstrations...” This is a tangled bunch of sentences that has many issues. First, (ii) seems to contradict the premise of using one source demonstration, as written at the beginning of the section and in (i). Second, (i) and (ii) seem to both talk about the same reason--demonstrations can be synthesized.

10. I encourage the authors to improve their paper by cutting down on a lot of redundancy. A 5-page paper undershooting the page limit is better than a 9-page paper that repeats much of the information multiple times.

### Questions
1. What are some examples of source tasks? Where are the 128 or 512 demonstrations (mentioned in section 4.1.5) from? What are source-target task pair examples that were ranked most similar to each other?

2. How many tokens is the average Super-NI answer in your evaluation dataset?

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes In-Context Transfer Learning (ICTL), a framework to enhance demonstration synthesis for in-context learning (ICL) by transferring labeled demonstrations from similar tasks. The motivation is to overcome the limitations of synthesizing demonstrations from scratch using large language models (LLMs). The framework aims to address cost and efficiency issues in generating high-quality task demonstrations by leveraging transfer learning. ICTL has two steps: (1) source sampling: select source tasks similar to the target task by minimizing transfer error. (2) target transfer: transfer these selected demonstrations to the target task’s format using LLMs (transferring, verify, sample). Experiments on the Super-NI dataset show that ICTL achieves 2.0% improvements on average compared to synthesizing demonstrations from scratch.

### Strengths
1. The paper is well written and easy to understand. It tackles an important problem of enhancing the generalibility of LLMs.
2. The proposed in-context transfer learning is simple yet produce reasonable improvements over generating demonstrations from scratch. The ablation studies and sensitivity analysis of paramters are beneficial. Evaluations on the Super-NI demonstrate the method’s cross-task performance. The paper also provides a code release for reproducibility, which is beneficial to the community.

### Weaknesses
I have two major concerns:

1. The performance improvements seem marginal. The 2% performance improvement over baseline method (generating target demonstartions from scratch) is marginal and may not justify the added complexity of ICTL. The procedure of sampling, transferring, verifying, and re-sampling introduces computational overhead. It would be great to include the computation cost comparison between different approaches in Table 1. Moreover, from the analysis of different parameters in Figure 3, seems the performance can drop signficiantly (more than 2%) if the parameters are not selected properly. This let me suspect the effectiveness of the proposed framework in more general real-world scenarios. It would be interesting to see more results on other tasks. 

2. The proposed framework follows the standard transfer learning pipeline. The theoretical analysis is also straight-forward extension. In general, I feel the technical contribution is limited while there are some specific designs for LLM domain.

### Questions
While the paper provides an interesting application of transfer learning to in-context learning, I have the concern of the incremental improvement and relatively complex framework. The contribution is not sufficiently novel, and the results are a bit underwhelming. I would be happy if the authors can provide more compelling evidence of ICTL’s practical benefits (e.g., compare the computational cost in Table 1, provide additional evaluation on other tasks).

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces In-Context Transfer Learning (ICTL), a method designed to improve demonstration synthesis for in-context learning (ICL) by leveraging labeled examples from similar tasks. 
Unlike conventional approaches that generate task demonstrations from scratch using LLMs, ICTL employs a transfer learning-inspired process consisting of two steps: Source sampling and target transfer. 
In the source sampling phase, relevant source tasks are filtered based on task definitions, and demonstrations are subsequently selected using a simulated annealing approach that minimizes transfer error.
During the target transfer phase, LLMs are used to adapt these selected demonstrations to synthesize examples suitable for the target task.
Experimental results on the Super-NI dataset indicate that ICTL outperforms traditional demonstration synthesis methods, effectively addressing the limitations of LLMs in producing high-quality demonstrations from scratch.

### Strengths
1. The synthesis of demonstrations from similar tasks is an interesting and novel approach to enhancing the quality of in-context learning.
1. The paper provides an optimization objective for sampling demonstrations, aimed at minimizing the task error bound.
1. The experimental results demonstrate improvements over traditional direct demonstration synthesis methods, showing the effectiveness of the proposed approach.

### Weaknesses
1. The proof of Equation 3 (Appendix A) is not clearly explained. At the end of the proof (lines 868-869), the statement "by substituting Theorem 1 into Equation 5, we can derive Equation 3" skips too many intermediate steps, which makes the derivation ambiguous. While Theorem 1 and Equation 5 are related to finding the target task $\hat{\mu}_T$, Equation 3 is focused on determining the source task $\hat{\mu}_S$. This discrepancy is confusing and should be clarified further to convincingly demonstrate that Equation 3 is a reasonable objective.

1. The proposed approach is primarily applicable to benchmarks that provide explicit task definitions, which restricts its applicability to other benchmarks. This limitation confines the evaluation to the Super-NI dataset, and it would be beneficial to explore how this method could generalize to other settings.

1. The sampling process involving simulated annealing lacks sufficient clarity. Given that simulated annealing typically involves multiple iterations, it raises concerns about efficiency. Further details on how the sampling algorithm is implemented, and any measures taken to ensure efficiency, would be helpful.

Other Issues:
1. Rouge -> RougeL

### Questions
1. In Equation 2, $\mu$ represents a data distribution of a task, while $x$ is a representation vector of a task definition. Since these two elements are in different spaces, how is the Wasserstein distance between them computed?

1. Lines 206-207 mention that task selection is ranked by the Wasserstein distance between the embedding vectors of task definitions. How is the Wasserstein distance calculated between two vectors? Typically, the Wasserstein distance is used to compare distributions rather than individual points.

1. How does the proposed source task filtering compare to retrieval-based approaches, such as those using Contriever or BERT?

1. What are the "perturbations" used in simulated annealing during the sampling of source demonstrations?

1. What kind of score function is employed in the simulated annealing process?

### Soundness
2

### Presentation
3

### Contribution
3
