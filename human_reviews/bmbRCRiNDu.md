# Dualformer: Controllable Fast and Slow Thinking by Learning with Randomized Reasoning Traces

- Decision: Accept (Poster)
- Scores: 6, 5, 3, 5

## Abstract
In cognition theory, human thinking is governed by two systems: the fast and intuitive System 1 and the slower but more deliberative System 2. Analogously, Large Language Models (LLMs) can operate in two reasoning modes: outputting only the solutions (\emph{fast mode}) or both the reasoning chain and the final solution (\emph{slow mode}).  We present \dualformer, a single Transformer model that seamlessly integrates both the fast and slow reasoning modes by training on randomized reasoning traces, where different parts of the traces are strategically dropped during training. At inference time, \dualformer can be easily configured to execute in either fast or slow mode, or automatically decide which mode to engage (\emph{auto mode}).  It outperforms baselines in both performance and computational efficiency across all three modes: \textbf{(1)} in slow mode, \dualformer achieves $97.6\%$ optimal rate on unseen $30 \times 30$ maze tasks, surpassing the \searchformer baseline (93.3\%) trained on data with complete reasoning traces, with $45.5\%$ fewer reasoning steps; \textbf{(2)} in fast mode, \dualformer achieves $80\%$ optimal rate,  significantly outperforming the Solution-Only model trained on solution-only data,  which has an optimal rate of only 30\%;  \textbf{(3)} in auto mode, \dualformer achieves $96.6\%$ optimal rate with $59.9\%$ fewer steps than \searchformer.  For math reasoning problems, our techniques have also achieved improved performance with LLM fine-tuning, demonstrating its generalization beyond task-specific models. We open source our code at https://github.com/facebookresearch/dualformer.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
- The paper proposes the question: Can we integrate both the fast and the slow modes into Transformer-based reasoning agents, similar to how humans possess two distinct thinking systems, and let them complement each other?
- Main contributions: A framework for training Dualfromer, a Transformer that solves reasoning and planning tasks inspired by the “System 1 & System 2” thesis by Kahneman in _Thinking, Fast and Slow_. 
- One of the tasks is a maze-like environment where the model is trained to find the shortest path between two cells. The model can either output the solutions and reasoning chain or only the final solution. In “auto mode,” it can decide which of both strategies to engage in. 
- Being able to engage in these different modes is an improvement over current models in the cluster, noticeably Searchformer, which is only able to engage in outputting reasoning chain & solution, potentially requiring lots of inference time. The new model improves here by being trained on randomly sampled reasoning traces and dropping traces with various strategies. 
- They also continue evaluating their new framework by training a Dualformer to solve the Aug-MATH benchmark, where the model outperforms finetuned LLMs.

### Strengths
- Considering system 1 & 2 while working with transformers isn’t particularly novel, I thought the way it was included while training the transformer on reasoning traces interesting. Beyond that, the paper provided good motivation for why the auto mode strategy could be more efficient than current models. 
- I found the results on the MATH dataset fascinating and was surprised to see a “more real-world” application of this strategy.

### Weaknesses
- While I could understand the performance gains of the new model based on some results being higher in Table 5.1-5.3, I found it very hard to make sense of which models exceed where and why. E.g. I understood that “Auto mode” should, in theory, be less exploratory than “Slow mode”, why is the SWC for 30x30 mazes higher? (0.993>0.989). This is, of course, a nitpick, and I don’t expect the reason here to be super relevant, but I would have, in general, appreciated a better overview of the results. Every score improvement has a blue box, which ultimately doesn’t carry a lot of information value (_if everything is highlighted, nothing is_). It could easily be that I don’t understand a fundamental consideration of this yet for them to be all evaluation indistinction, so I am excited to hear your response. 
- I have a similar, though slightly less confusing, nitpick with the results in the MATH benchmark. I appreciate this inclusion of a finetuned Mistral and Llama, though I would have wished to better understand how good this is compared to other sota models.

### Questions
- Why did you not test auto mode on the MATH benchmark?
- How did the training differ between Dualformer and finetuned MATH models?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
Inspired by slow and fast thinking modes in human cognition theory, this paper introduces Dualformer by training models on reasoning traces with randomized dropping, which allows the model to learn efficient reasoning shortcuts. The performance outperforms baselines on a maze environment and MATH dataset.

### Strengths
1. The paper is well written and easy to follow. 
2. The idea of integrating fast and slow reasoning modes from human cognition theory is novel.

### Weaknesses
1. The motivation is not super convince to me, why LLM needs to integrate both the fast and slow reasoning modes are not clear, it’s not convinced that human takes benefit from the these two modes then LLM should also follow these two modes given the fact that human brain and LLM are running with two distinct systems.
2. The mathematical foundation of the methodology requires more rigorous discussion. Specifically, it's important to explain why dropping random intermediate steps can lead to better performance. This contrasts with findings in recent reasoning papers, for example [1, 2, 3] where learning longer reasoning steps involving (1) different reasoning strategies; (2) self-refinement; (3) backtrack, etc, led to better performance since it provides the model with more flexible function mapping, which is easier to fit compared to direct query-to-solution mapping. Importantly, a section discussing the foundation of the proposed method is necessary.
- [1] Qin, Yiwei, et al. "O1 Replication Journey: A Strategic Progress Report--Part 1." arXiv preprint arXiv:2410.18982 (2024).
- [2] Kumar, Aviral, et al. "Training language models to self-correct via reinforcement learning." arXiv preprint arXiv:2409.12917 (2024).
- [3] Gandhi, Kanishk, et al. "Stream of Search (SoS): Learning to Search in Language." arXiv preprint arXiv:2404.03683 (2024).
3. A major limitation is that the toy example is too simplistic for understanding the method's effectiveness in real reasoning tasks, though it serves to illustrate the problem scope, and there is one table showing results on MATH. To demonstrate that the model truly learns better solution strategies rather than overfitting on synthetic data, one idea is evaluating in out-of-distribution environments. For example, the model could be finetuned on the maze dataset but evaluated on MATH or GSM8K - problems that could benefit from better thinking/reasoning strategies. 
4. Moreover, in Table 5.6, the differences between the baseline and the method proposed in this paper are marginal. This raises questions about whether the results stem from hyperparameter selection, especially since finetuning in the MATH domain uses the same hyperparameters despite different response lengths. Additionally, the evaluation is conducted with only one seed, and the model uses temperature=0.7 and top-p=0.9 for better diversity. While this approach is valuable for studying whether the model can generate diverse solutions, it introduces uncertainty when evaluating the method's performance.

### Questions
1. In section 5.2 (Structured Trace Dropping and Randomization), the MATH dataset is constructed by randomly dropping intermediate reasoning steps from the solution. However, in the examples shown in Answers to Q1 and Q2, the model doesn't skip steps but rather presents solutions more concisely. Does this truly demonstrate learned slow and fast thinking strategies, or is it simply a more compact problem-solving approach? And where does the mismatch between the training demo and inference response come from?
2. Following Limitations 3 and 4, the toy example's limitations seems also stem from the generalizability in approach. In the maze example, the procedure first creates a search path using A*, and the model attempts to mimic this search procedure with random dropout. However, in math problems, the baseline provides a direct path with multiple steps, which is then subjected to random step dropping. This raises questions about where the equivalent of the A* algorithm is in MATH problems and how Structured Trace Dropping is implemented on MATH or other general tasks. The methodology appears to be using different approaches for maze and math problems without clear justification for their equivalence.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
The paper proposes **Dualformer**, a new Transformer model designed to integrate both fast and slow reasoning modes, inspired by the dual process theory in human cognition. The model is trained on data with randomized reasoning traces, allowing it to dynamically choose between quick, solution-only outputs (fast mode) and detailed, step-by-step reasoning (slow mode).

### Strengths
The idea of combining fast and slow reasoning modes within a single model is novel and inspired by human cognition, which is a strong conceptual foundation.

### Weaknesses
1. **Reproducibility**: The paper does not provide code or sufficient details about the implementation, which raises concerns about reproducibility.

2. **Comparison**: There is no comparison with similar methods such as SwiftSage[1], which should be cited and discussed to contextualize the contributions of this work.

3. **Experimental Scope**: The experimental tasks (maze navigation and Sokoban) are relatively simple and may not fully demonstrate the model's potential in more complex, real-world scenarios. More challenging and diverse benchmarks would strengthen the paper.

### Reference:

- [1] Lin, Bill Yuchen, et al. "Swiftsage: A generative agent with fast and slow thinking for complex interactive tasks." Advances in Neural Information Processing Systems 36 (2024).

### Questions
1. **Code Availability**: Will the authors release the code to ensure reproducibility? If not, can they provide more detailed pseudocode or implementation guidelines?

2. **Comparison with SwiftSage**: The authors should include a comparison with SwiftSage and discuss the differences and similarities. How does Dualformer improve upon or differ from SwiftSage?

3. **Complex Scenarios**: Can the authors extend their experiments to more complex tasks such as strategic games (e.g., Werewolf) or combinatorial optimization problems? This would better demonstrate the model's versatility and robustness.

4. **Theoretical Justification**: While the empirical results are promising, a theoretical analysis of why and how the randomization of reasoning traces improves performance would strengthen the paper. Can the authors provide more theoretical insights or hypotheses?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper aims to endow a transformer model with both slow and fast reasoning modes using only a simple data recipe. The proposed model -- Dualformer -- is trained using randomized trace dropping, which enables the model to switch between fast (solution-only) and slow (step-by-step reasoning) processing without any architectural modifications. Dualformer is evaluated on maze and Sokoban pathfinding tasks. The model is evaluated on maze and Sokoban pathfinding tasks and is also applied to fine-tune large language models (LLMs) for solving math problems. Experimental results demonstrate that Dualformer achieves superior performance compared to baseline methods.

### Strengths
In this work, the authors leverage a purely data-driven approach (randomized trace dropping) to enable controllable dual-mode reasoning in a Transformer model.  This approach is computationally efficient and reduces the need for complex architectural changes. In the pathfinding tasks, Dualformer outperforms baseline models in both fast and slow modes, both in terms of the number of optimal solutions found and the diversity of these solutions.. Additionally, the approach proves effective in improving performance on math problem-solving tasks. The methodology, experiments, and results are clearly presented, making the paper easy to follow and understand.

### Weaknesses
It is unclear whether the chosen tasks (mazes, Sokoban) encapsulate fast versus slow thinking as humans would apply them. In particular, simply outputting a solution without explicitly listing the reasoning steps does not necessarily capture fast cognitive processing.

It would be valuable to have more extensive analysis of where and why Dualformer’s fast mode fails. Specifically, examining how task complexity—such as increased wall density or structured obstacle layouts—impacts performance could reveal important limitations.

More investigation into the auto mode, specifically on how the model determines which mode to activate under varying maze difficulties, would add depth. Understanding the criteria or patterns that lead to automatic mode-switching would be very useful.

### Questions
- What are the specific failure modes in fast mode?
- Does fast mode performance degrade as maze complexity increases (e.g., with a higher density of wall cells or more structured wall arrangements)?
- How would Dualformer perform on a rectangular maze instead of a square one at test time?
- In the example mazes shown, it appears relatively easy for a human to find the optimal path. How does fast mode perform on more challenging mazes where a human might need more time to find a path?
- For the dropping strategies, how were the initial three sets of probabilities selected?
- The experiment with OpenAI o1 models seems to use them in a zero-shot manner. Have you tried using few-shot approaches and testing on smaller mazes?
- Could you elaborate on how the randomized dropping strategies mimic shortcuts in human cognitive processes?

### Soundness
2

### Presentation
3

### Contribution
2
