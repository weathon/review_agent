# Helix: Evolutionary Reinforcement Learning for Open-Ended Scientific Problem Solving

- Decision: Accept (Poster)
- Scores: 4, 4, 8, 4

## Abstract
Large language models (LLMs) with reasoning abilities have demonstrated 
growing promise for tackling complex scientific problems. Yet such tasks are inherently domain-specific, unbounded and open-ended, demanding exploration across vast and flexible solution spaces. Existing approaches, whether purely learning-based or reliant on carefully designed workflows, often suffer from limited exploration efficiency and poor generalization. 
To overcome these challenges, we present **HELIX**---a 
**H**ierarchical **E**volutionary reinforcement **L**earning framework with **I**n-context e**X**periences. HELIX introduces two key novelties: (i) a diverse yet high-quality pool of candidate solutions that broadens exploration through in-context learning, and (ii) reinforcement learning for iterative policy refinement that progressively elevates solution quality. This synergy enables the discovery of more advanced solutions. On the circle packing task, HELIX achieves state-of-the-art result with a sum of radii of 2.63598308 using only a 14B model. Across standard machine learning benchmarks, HELIX further surpasses GPT-4o with a carefully engineered pipeline, delivering an average F1 improvement of 5.95 points on the Adult and Bank Marketing datasets.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a unified framework that integrates reinforcement learning (via GRPO), evolutionary search (via NSGA-II), and in-context learning to enable large language models to autonomously tackle complex, unbounded scientific problems. HELIX maintains a diverse population of candidate solutions, embedding and filtering them for both reward and semantic diversity, while reinforcement learning updates the policy to progressively improve reasoning and exploration efficiency. Through iterative feedback loops, the model learns to build upon previous high-quality solutions, balancing exploration and exploitation. Evaluated on 19 tasks across five domains—including machine learning, physics simulation, circle packing, function minimization, and symbolic regression—HELIX consistently outperforms strong baselines, including GPT-4o and task-specific algorithms, and achieves state-of-the-art results such as a new world record in the circle-packing benchmark.

### Strengths
- **\[S1] Important and timely problem.** The paper tackles the open-ended scientific problem-solving capability of LLMs, an increasingly important direction with potential impact across AI-driven discovery, optimization, and reasoning tasks.

- **\[S2] Thoughtful integration of multiple paradigms.** The proposed HELIX framework effectively combines reinforcement learning (GRPO-style policy optimization), evolutionary search (NSGA-II–based diversity–reward balancing), and in-context learning into a single hierarchical system. While each component is known, their coordination into an iterative, feedback-driven process is well-motivated and technically coherent.

- **\[S3] Comprehensive and diverse experimental evaluation.** The authors evaluate HELIX on 19 tasks spanning five domains—including machine learning, physics simulation, symbolic regression, and geometric optimization—demonstrating broad applicability and consistent improvements over strong baselines such as GPT-4o and AlphaEvolve-style systems.

### Weaknesses
See “Questions” below.

### Questions
- **\[Q1]** The proposed framework mainly combines existing components—GRPO-style reinforcement learning, NSGA-II–based evolutionary selection, and in-context prompting. Could the authors clarify what specific technical or methodological challenges arise when integrating these elements, and how HELIX overcomes them? In particular, what makes this combination more than an engineering effort or straightforward composition of known techniques?

- **\[Q2]** The paper motivates HELIX as balancing exploration and exploitation through the interplay of RL, evolution, and in-context learning. However, no formal theoretical analysis or guarantees are provided. Could the authors clarify whether there is any theoretical grounding (e.g., convergence, stability, or optimality analysis) supporting this hybrid framework, or is it intended purely as an empirical system-level contribution?

- **\[Q3]** While HELIX shows consistent empirical improvements over baselines across multiple benchmarks, many of these tasks (e.g., UCI datasets, classical optimization functions, small-scale physics simulations) are relatively well-studied and limited in scale. Could the authors clarify how to interpret the significance of these improvements—do they demonstrate genuine scientific discovery capabilities, or primarily reflect engineering and optimization gains within benchmark settings? In other words, how do these results translate to more challenging, real-world, or open-ended scientific problems?

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
2

### Summary
The paper proposes HELIX, a framework to learn an LLM policy that iteratively produces diverse yet high quality solutions for open-ended problems.

### Strengths
- The paper studies a setting of current interest to the community: open-ended problem solving.
- The approach proposed in the paper, HELIX, is novel and interesting (though some details are unclear). 
- The experiments show the efficacy of the proposed method on interesting scientific reasoning tasks.

### Weaknesses
- My primary concern with the paper is with the description of the algorithm. It's unclear how NSGA-II is integrated into the pipeline. For instance, in Figure 2, the solutions selected via NSGA-II are passed to the RL algorithm as feedback. How does that work? 
- While the experiments are interesting, on many tasks, the Direct Prompt baseline works pretty well already. Additionally, on some tasks, Open Evolve performs worse than Direct Prompt. The authors should use more challenging domains to evaluate their method.

### Questions
See weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces HELIX, a hierarchical evolutionary reinforcement learning framework that combines GRPO-based policy optimization, multi-objective evolutionary search (via NSGA-II), and in-context learning for solving open-ended scientific problems. The authors argue that HELIX enables LLMs to balance exploration and exploitation while iteratively improving solution quality. The method is tested on 19 diverse tasks (physics simulations, machine learning, symbolic regression, circle packing, and mathematical optimization) and achieves superior results on 16 tasks compared to strong baselines, including GPT-4o.

### Strengths
- Novel combination of reinforcement learning, evolutionary algorithms, and in-context prompting, addressing key limitations of existing approaches (e.g., entropy collapse, lack of diversity).

- Strong empirical results across a wide range of domains, including physical design and scientific optimization, demonstrating the generality of the method.

- Clear framework design and well-articulated motivation connecting HELIX’s components to the nature of open-ended scientific discovery.

- Ablation and scaling analyses provide convincing evidence that both diversity-aware evolution and reinforcement learning contribute meaningfully to performance.

### Weaknesses
- While HELIX shows impressive performance across various scientific tasks, the paper provides limited discussion on the sensitivity of the framework to its key hyperparameters. In particular, the reward normalization constants (e.g., denominators used in Eq. 6–10 for physics tasks) and NSGA-II parameters (e.g., population size, crowding distance, KNN-based diversity metric) appear to play a crucial role in balancing exploration and exploitation, yet no analysis is offered regarding their influence on stability or convergence. This raises concerns about how robust HELIX remains under different hyperparameter settings.

### Questions
- How sensitive is HELIX to the reward normalization constants and NSGA-II hyperparameters?

### Soundness
4

### Presentation
4

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
This paper introduces HELIX, a hybrid method that enhances the reasoning capabilities of LLMs by combining reinforcement learning (RL) with verifiable rewards, evolutionary algorithms, and in-context learning. The authors argue that the RL fine-tuning (GRPO) phase increases domain-agnostic adaptation, the evolutionary nature promotes the solution diversity, and the in-context learning improves the solution quality using previous diverse high-quality solutions. This method is validated on several scientific and engineering tasks, showing improvement over baselines.

### Strengths
1. This work effectively combines the known strategies for reasoning with LLMs, the RL fine-tuning, evolutionary algorithms, and in-context learning. To my knowledge, this combination is new.
2. The proposed method shows strong empirical results compared to baselines across many scientific problems, showing its practicality.
3. The source code is provided.

### Weaknesses
1. (Clarity) One of my main concerns is the clarity of the paper. While this work is understandable and easy-to-read at a high level, there is some ambiguity at low levels and many details seem to be missing. I left many questions to clarify some points that I couldn't understand; see the questions below. Additionally, I believe providing a detailed formalised algorithm and LLM prompts used for each experiment would be valuable. 
2. (Method) I believe GRPO is an on-policy RL algorithm, but the algorithm seems to train it with off-policy samples from the previous steps (since $(q,a) \sim \mathcal{D}$ where $\mathcal{D}$ is previously discovered solutions). Please provide any rationale for this choice, or correct me if I'm wrong.
3. (Evaluation) Some necessary evaluation settings seem to be missing, especially the standard deviation or confidence interval for each experiment. I assume that the proposed method consumes significantly more LLM calls than the current baselines (direct prompt, open evolve, task-specific methods), as it requires fine-tuning. Therefore, I believe that the fine-tuning method, e.g., vanilla GRPO, should also be included as a main baseline for Table 1.

(Minor)  
4. Line 90: Please add the reference for GRPO  
5. Line 92: Using "NSGA-II" without any explanation makes readers confused; a short explanation of it (at least that it is an evolutionary algorithm) would be helpful.  
6. Line 115-117: The related works are only roughly introduced. Providing more details regarding them would be valuable, e.g., how KL-Conv addressed the limitation, what the memory-less RL is, and why it cannot leverage previous solutions.

### Questions
1. Line 158, Eq. (2): does this objective guarantee Eq. (1)?
2. Line 158, Eq. (2): why you take average over $s_t\sim \mathcal{P}$, not $\sim \mathcal{D}$? 
3. Figure 2: What is the lineage tree? An explanation seems to be needed for it.
4. Line 211, Eq. (4): a) What is the formal definition of $q$? b) What is the formal definition of $\hat{A}$? c) Can you define the MDP for your GRPO fine-tuning?
5. Line 245, Eq. (5): a) Is this diversity measure new to this work, or borrowed from other works? b) It seems to require the pairwise distance calculation. Isn't it computationally heavy?
6. Is the LLM finetuned to be used for different kinds of tasks? For example, for the Machine Learning benchmark, is a single LLM used for all the subtasks (Adult Income, Bank Marketing, and Boston Housing), or do we need to finetune an LLM for each subtask? 
7. How to generate the initial solutions?
8. Doesn't HELIX suffer from a similar limitation as the evolutionary-algorithm-based approaches, designing problem-specific prompts for each task? If I understand correctly, HELIX needs prompts to combine previous solutions to generate a new one, just like the EAs need prompts to mix (crossover/mutation) solutions. How did you compare the complexity of designing the prompt between your algorithm and EAs?
9. Line 253, what does "nondominated sorting procedure" mean?
10. Line 256, what do "a crowding distance measure" and the "objective space" mean?
11. Is "Evolutionary Search" in the title of section 3.2 indicating the in-context learning with previous solutions ("ancestral states" in Line 220)?

---

### LLM usage disclosure
I did not use an LLM for my review, but I used grammar check software.

### Soundness
3

### Presentation
1

### Contribution
2
