# Fill in the Blank: Exploring and Enhancing LLM Capabilities for Backward Reasoning in Math Word Problems

- Decision: Reject
- Scores: 3, 8, 5, 6

## Abstract
While forward reasoning (i.e., find the answer given the question) has been explored extensively in the recent literature, backward reasoning is relatively unexplored. We examine the backward reasoning capabilities of LLMs on Math Word Problems (MWPs): given a mathematical question and its answer, with some details omitted from the question, can LLMs effectively retrieve the missing information? 

In this paper, we formally define the backward reasoning task on math word problems and modify three datasets to evaluate this task: GSM8k, SVAMP and MultiArith. Our findings show a significant drop in the accuracy of models on backward reasoning compared to forward reasoning across four SOTA LLMs (GPT4, GPT3.5, PaLM-2, and LLaMa). Utilizing the specific format of this task, we propose three novel techniques that improve performance: Rephrase reformulates the given problem into a forward reasoning problem, PAL-Tools combines the idea of Program-Aided LLMs to produce a set of equations that can be solved by an external solver, and Check your Work exploits the availability of natural verifier of high accuracy in the forward direction, interleaving solving and verification steps. Finally, realizing that each of our base methods correctly solves a different set of problems, we propose a novel Bayesian formulation for creating an ensemble over these base methods aided by a verifier to further boost the accuracy by a significant margin. Extensive experimentation demonstrates that our techniques successively improve the performance of LLMs on the backward reasoning task, with the final ensemble-based method resulting in a substantial performance gain compared to the raw LLMs with standard prompting techniques such as chain-of-thought.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper provides insight into the relatively unexplored area of backward reasoning in Math Word Problems (MWPs). The authors formally define the task of backward reasoning to derive missing information from given answers and incomplete questions.
The authors modify three datasets to evaluate this task. The experiments show that multiple Large Language Models (LLMs) showed a significant accuracy drop in backward reasoning compared to forward reasoning. 
The authors propose three basic prompt methods as improvements, namely “Rephrasing”, “PAL-Tools” and “Reprompting and Verification”. The authors further propose one ensemble-based method via the use of a verifier. 
Through extensive experimentation, the authors demonstrate that their techniques substantially enhance LLM performance on the backward reasoning task, especially the ensemble-based method further boost the accuracy by a significant margin.

### Strengths
1. The paper explores the relatively understudied area of backward reasoning in mathematical word problems (MWPs) and formalizes the task of backward reasoning.
2. The research methodology is rigorous. The authors used a variety of state-of-the-art prompt techniques for reverse reasoning with LLM, testing them to ensure a thorough evaluation.
3. The paper is systematically structured and clearly distinguishes between problem definition, methodology, experimentation and analysis.

### Weaknesses
1. The motivation seems to be Ambiguous. The paper's definition of backward reasoning essentially frames it as a fill-in-the-blank task, rather than a genuine backward reasoning or a broader sense of causal reasoning. In reference [1], a similar task is merely a sub-task in backward verification for verifying forward reasoning. The authors have repurposed it as a new backward reasoning task, which seems redundant and lacks research value. The paper's discussion on the practical application scenarios of this task is insufficient, making it challenging to discern its real-world significance. Moreover, the rationale behind comparing the difficulty levels of backward and forward reasoning remains unexplained.
2. Lack of Methodological Novelty. The three primary methods presented are essentially repurposed from forward reasoning techniques that have been previously introduced and widely applied in other works. The authors have essentially transformed backward reasoning into forward reasoning, without designing specific methods tailored to the unique characteristics of backward reasoning, thereby undermining the essence of studying backward reasoning. Specifically:
-----The “Rephrasing” method aligns closely with the “Condition Mask Verification” method from reference [1]. This should have been treated as a baseline rather than a novel approach due to its lack of originality.
-----The “PAL-Tools” method is fundamentally the same as the method introduced in reference [2], with the authors merely employing the SymPy library for solving for ‘x’. This approach lacks innovation.
-----The “Reprompting and Verification” method is essentially an inversion of a method from reference [3] used to verify the correctness of forward reasoning, which seems like a forced novelty.
3. While the paper presents a range of experimental results, the depth of analysis behind these results is lacking. For instance, the paper doesn't delve deep into the differences between various prompting techniques and why certain techniques are more effective in specific scenarios. The comparative analysis mostly revolves around forward reasoning methods, lacking a direct comparison with potential backward reasoning techniques, making it difficult for readers to gauge the true advantages of the proposed methods.
4. The authors' modifications to the datasets GSM8k, SVAMP, and MultiArith are minimal, merely replacing numbers in questions with blanks. However, they claim to have created “new datasets”, which seems to exaggerate their contribution.
5. The paper contains many grammatical errors. One is the missing period at the end of the paragraph “In order to establish … on this task”, and the second is the incorrect punctuation in the paragraph “A forward or the typical … numeric value of the blank” with the misplaced colon in ’half.’.

[1]	Weng, Y., “Large Language Models are Better Reasoners with Self-Verification”, <i>arXiv e-prints</i>, 2022. doi:10.48550/arXiv.2212.09561.
[2]	Luyu Gao and Aman Madaan and Shuyan Zhou and Uri Alon and Pengfei Liu and Yiming Yang and Jamie Callan and Graham Neubig (2022)PAL: Program-aided Language Models International Conference on Machine Learning abs/2211.10435
[3]	Aman Madaan and Niket Tandon and Prakhar Gupta and Skyler Hallinan and Luyu Gao and Sarah Wiegreffe and Uri Alon and Nouha Dziri and Shrimai Prabhumoye and Yiming Yang and S. Welleck and Bodhisattwa Prasad Majumder and Shashank Gupta and A. Yazdanbakhsh and Peter Clark (2023)Self-Refine: Iterative Refinement with Self-Feedback arXiv.org abs/2303.17651

### Questions
1. What is the research significance and practical implications of the proposed backward reasoning task? Specifically:
---a) Why have you chosen to define backward reasoning as a fill-in-the-blank task and study the capabilities of LLMs on this particular task?
---b) What motivated the exploration of the relative difficulty between backward and forward reasoning?
---c) Could you elaborate on potential applications of this task in sectors like education, industry, or other domains?
2. I found it challenging to understand the precise mechanism of the “Ensembling” method. Can you offer a more intuitive explanation, especially in the context of Figure 2 in your paper? Moreover:
---a) How did you arrive at the decision to use a holdout set of 100 examples from the datasets?
---b)What criteria or methods were used to select these examples for the holdout set? Is there a specific scientific basis for this choice?
3. Would it be possible to design ablation studies to elucidate why the “Ensembling” method performs better? Can you shed light on the effectiveness of each module and whether the method seamlessly integrates the backward reasoning capabilities of the three basic methods?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper presents a the backward reasoning task on math word problem solving. The authors conduct extensive experiments on three datasets. The empirical results show the effectiveness of the proposed method. The paper is well written and the solution is clear.

### Strengths
1. The authors conduct extensive experiments on three datasets. The empirical results show the effectiveness of the proposed method. 
2. The paper is well written and the solution is clear.

### Weaknesses
1. The paper explores the LLM capabilities for backward reasoning, it is not clear what the complexity is compared to the traditional math word problem solving models, such as Graph2Tree, GTS etc. 
2. The implementation details are not clearly described. For example, GPU and memory size, codes and datasets.

### Questions
1. The paper explores the LLM capabilities for backward reasoning, it is not clear what the complexity is compared to the traditional math word problem solving models, such as Graph2Tree, GTS etc. 
2. The implementation details are not clearly described. For example, GPU and memory size, codes and datasets.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes a new problem called "backward reasoning", namely in a math word problem, given some conditions and the final answer, it aims to infer a missing condition. The authors propose three strategies, including rephrase, code-aid tool and verification, with a final ensemble stage to solve the problem. The proposed approach is then evaluated on three math world problem benchmarks, and several ablation studies are done to further understand the role of each design in the proposed approach.

### Strengths
- Three strategies, including rephrase, tool and verification, are proposed to solve the ``backward reasoning'' problem.

- Extensive experiments have been done to demonstrate the effectiveness of the proposed approach.

- The paper is very easy to follow and organized well.

### Weaknesses
-  The motivation for the proposed ``backward reasoning'' problem is not very clear to me. In practice, if we want to know a condition, it is natural to just rephrase the question and make {given conditions, final answer} as conditions and the missing condition as a question. In other words, I do not really see the essential of defining a so-called "backward reasoning".

- I have non-trivial concerns about the novelty and contribution of this paper. The "rephrase" techniques of the three proposed solutions that use x to replace the missing condition are not novel, which is already proposed in the advanced chain-of-thoughts methods [1]. Also, I do not see the uniqueness of the proposed ``PAL-Tools'', it is basically a special case of PAL [2]. Similar novelty concerns to the proposed ensemble stage as well, since the ensemble is definitely not a new technique and in most cases, it can improve the performance.

- The ``backward reasoning'' problem and proposed approach are only evaluated on math benchmarks. However, if we target for uncovering a missing condition, it is essential to do evaluations on other types of reasoning tasks, such as commonsense and symbolic.


[1] Fu et al, complexity-based prompting for multi-step reasoning.  
[2] Gao et al. PAL: Program-aided Language Models

### Questions
Please check the weakness section, my concerns and questions are pretty much there.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper studied the task of "backward reasoning" in math word problems, to be specific, they "mask"  the numeric token with an underscore. Then they adopt several prompting methods to solve the problem as a forward reasoning problem. The methods include the proposed one, reprompt and verification / check your work. Finally, they propose a Bayesin ensembling technique to ensemble the prompting results.

### Strengths
1. Propose the task of backward reasoning.
1. Propose a re-prompt and verification method, similar to self-refine technique to iteratively finding the correct answer as prediction.
2. Propose the Bayesian ensembling technique, to obtain superior performance in Table 4.

### Weaknesses
1. Only the verifier with Bayesian ensembling gives some novelty and some insights to the community. But they did not expand more analysis except Table 5 (it should be Table 5 rather than Figure 5 I guess)
2. The other content seems more engineering efforts than scientific efforts. I think the dataset is also really important, which should be described (with more details) in the main paper rather than appendix. 
3. Not enough experiments to compare, for example, to show that the ensembling is better, we probably need to compare with other ensembling techniques. 
4. I think the paper should focus more on the task itself, rather than propose something and just put some short results.

### Questions
1. What's the difference between the reprompt and verification and check your work, are they the same thing? Why we have two names for the same method?
2. How is the proposed ensemble compared with other vanilla ensembling methods? For example, if I just do majority voting.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair
