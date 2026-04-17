# Beyond Magic Words: Sharpness-Aware Prompt Evolving for Robust Large Language Models with TARE

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
The performance of Large Language Models (LLMs) hinges on carefully engineered prompts. However, prevailing prompt optimization methods, ranging from heuristic edits and reinforcement learning to evolutionary search, primarily target point-wise accuracy. They seldom enforce paraphrase invariance or searching stability, and therefore cannot remedy this brittleness in practice. Automated prompt search remains brittle: small, semantically preserving paraphrases often cause large performance swings. We identify this brittleness as the **textual sharpness** of the **prompt landscape**. In this work, we provide the first formal treatment of textual sharpness in the discrete, semantic space of prompts, together with an operational robustness criterion over a semantic neighborhood; the design is black-box or API-only, requiring no gradients to update the model's parameters. Then we introduce **TARE** (Textual Sharpness-Aware Evolving), a derivative-free framework that alternates between an inner, sampling-based adversarial search that stresses a prompt with hard paraphrases and an outer, robust selection that prefers candidates whose neighborhoods remain strong. We further propose **ATARE**, which learns anisotropic weights to shape the semantic neighborhood and adapts its radius over time to balance exploration and fidelity. Diverse tasks evaluate our methods, whose design for minimizing textual sharpness gap leads to prompts that preserve accuracy under paraphrasing, outperforming accuracy-only prompt search while remaining computationally practical. The code is available for anonymous access at https://anonymous.4open.science/r/ATARE_TARE/.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper works on finding robust prompt landscapes with inspiration from the works on Sharpness-Aware Minimization (SAM). The idea behind SAM is to optimize a model on a continuous loss landscape and find a minimum where the loss function is flat via a minmax objective. Akin to this idea this paper tries to formulate the problem on a discrete space where they use an LLM as a sampler for potential neighbour data point samplers. Based on these neighborhoods they propose TARE to find adversarial potential prompts and use those prompts as a guidance to generate better neighborhood prompt candidates. This sampling is done iteratively akin to min max optimization to find the robust prompt landscape. Further they also proposed a componentwise weighted method for optimization in the form of ATARE.

### Strengths
1. Experiments were performed across multiple language families and for the sake of generalizability different tasks were used. Experiments were also performed on black box models. 

2. The ablation on the multiple components of the problem statement performed in the final results. 

3. Ablation was done on weaker and stronger oracles. 

4. Comparisons performed with point wise benchmarks

### Weaknesses
1. Problem formulation is hard to follow and not well defined. Notations are not explained and abruptly introduced which makes following through harder. The overall methodology can be simplified in the writing. Over use of notations that are not well defined makes it harder to follow through. 

2. While sharpness neighborhoods can be defined by edit distance or paraphrase distance the sampling is not guaranteed to be within these bounds. Is this accounted for in the sampling via filtering

### Questions
See weaknesses

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
4

### Summary
To find the optimal prompt that is robust to the input perturbation and has better generalization, the authors propose a new method called TARE. More specifically, it formulates the prompt optimization problem as a min-max optimization, aiming to find the optimal prompt that minimizes the maximum empirical risk incurred by the prompts within its sharpness neighborhood. To solve the optimization problem, TARE formalizes the definition of sharpness in the discrete semantic space of prompts. To capture heterogeneous sensitivity across the semantic components of a prompt, the authors further adapted the TARE to ATARE by learning the weights of the components to reflect the heterogeneity. Experimental results show that TARE or ATARE can outperform the evaluated baselines over multiple settings.

### Strengths
The strengths of the paper are listed as follows.
1. It is great that the authors provide open-source codes to reproduce their experimental results.
2. The proposed method is reasonable. The prompt that falls in a smooth landscape usually has a better generalization and is robust to the perturbation in the input.
3. The min-max formulation of the problem TARE aims to solve is concise and elegant.

### Weaknesses
The weaknesses of the paper are listed as follows.    
1. The authors did not provide a dedicated related work section in the main paper. Considering that prompt optimization is a research area with extensive existing work, it is necessary to have a section for comparing the differences between TARE and existing works in the main paper.    
2. The authors only compare their methods with two prompt optimization works. Some important baselines are missing, for instance, APE [1], APO[2], and EvoPrompt [3].    
3. The readability of the methodology section should be improved. First, Figure 2 contains too much information to illustrate the overview of TARE framework. Although it seems to be very comprehensive, incorporating too many details into a single figure can make the readers hard to catch the key information. Second, the authors introduce too many notations and formulas to describe their methods, where some of them are unnecessary and increase the difficulty of understanding the key ideas of TARE.   
4. Some important technical details are not clear.      
a. For the inner adversarial search, how to implement the generator oracle and how to define the perturbation range $\rho$ over a given prompt in the context of the prompt optimization problem. Please give more concrete details instead of the abstract formulation.     
b. In equation (4), how to calculate the $\Delta(p, p’)$?     
c. In ATARE, how to define the components of a given prompt?     
d. For the outer robustness update, how to implement the optimizer oracle and how to measure the semantic distance and budget?     
e. It is unclear how the landscape information can be leveraged by the LATO. It would be better if some examples could be provided.

### Questions
Please refer to the weaknesses section for my questions.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper introduces TARE, a black-box framework that improves the robustness of large language model prompts by explicitly minimizing textual sharpness—the sensitivity of prompt performance to small paraphrases. It defines a semantic neighborhood around each prompt and optimizes for stability within that region through an inner adversarial search and an outer robustness update, with ATARE further adapting neighborhood shape and size. Experiments across reasoning benchmarks show that TARE and ATARE yield more stable and accurate prompts than existing optimization methods like TextGrad and Revolve

### Strengths
1. The paper identifies an important and underexplored issue of brittleness under paraphrasing. It formalizes it as textual sharpness, bridging ideas from Sharpness-Aware Minimization to discrete text optimization.
2. The proposed algorithms are effective and model-agnostic, achieving consistent robustness improvements across tasks and architectures without requiring gradient access.

### Weaknesses
1. While the method defines textual neighborhoods semantically, the paper does not deeply analyze how the neighborhood metric affects robustness. It remains unclear whether the choice of embedding space or paraphrase generator significantly biases outcomes.
2. The study focuses on single-turn prompts. Extending sharpness-aware evolution to multi-turn or retrieval-augmented contexts is not explored.

### Questions
n/a

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper tackles an important yet under-explored problem in prompt optimization: the brittleness of prompts to semantically-preserving paraphrases. The authors formalize this as "textual sharpness" in the prompt landscape and propose TARE (Textual Sharpness-Aware Evolving), a framework that performs inner adversarial search over semantic neighborhoods and outer robust selection. They extend this to ATARE, which learns anisotropic weights for different prompt components. Experiments across reasoning tasks (Object Counting, Temporal Sequences, Tracking Shuffled Objects, GSM8K) demonstrate consistent improvements over baselines like TextGrad and Revolve.The core contribution is adapting the SAM (Sharpness-Aware Minimization) principle from continuous parameter space to discrete, semantic prompt space—a novel and well-motivated direction. I appreciate that they optimize the entire chain and that robustness truly matters for trusting LLM outputs in practice.

### Strengths
Novel problem formulation: Textual sharpness is an intuitive and important lens for understanding prompt brittleness. The formal treatment (Section 2) bridges optimization theory and discrete NLP in a principled way.

Comprehensive framework: TARE is well-designed with clear components (inner adversarial search, LATO optimizer, outer robust validation). The anisotropic extension (ATARE) demonstrates thoughtful attention to heterogeneous sensitivity across prompt components.
Strong empirical results: Consistent improvements across multiple tasks and models. I particularly appreciate the ablation study (Figure 3) showing each component matters, and the resilience analysis (Figure 4) demonstrating graceful degradation.

Practical relevance: The work addresses a real pain point in prompt engineering. Robustness to paraphrasing is crucial for building trustworthy LLM systems.

Thoughtful experimental design: Testing across diverse backbones (GPT-3.5, Gemini, Llama) and using two universal engines (GPT-4o, Claude 3.5 Sonnet) for fair comparison shows rigor.

### Weaknesses
1. Limited task diversity: All evaluation tasks are verifiable reasoning problems with clear right/wrong answers (counting, math, object tracking). This raises a key question: What about open-ended generation tasks where diversity matters? For creative writing, summarization, or tasks requiring pluralism (e.g., generating diverse perspectives on a political issue), optimizing for robust, flat neighborhoods might suppress desirable output diversity. I worry this approach could kill the creative variance we sometimes want from LLMs.

2. Missing analysis on reasoning depth: You test on reasoning tasks, but there's no analysis of how the method performs across different reasoning complexities. Does TARE work equally well for simple arithmetic vs. multi-hop reasoning? Given the recent release of models with explicit reasoning traces (o1-style "thinking" models like Qwen with QwQ or Mistral's reasoning variants), it would be very valuable to test whether optimizing prompts for these models behaves differently. Do the optimized prompts transfer across reasoning-capable and standard models?

3. Transferability not explored: A critical practical question is prompt transferability. If I optimize a prompt for one task/model using TARE, does it transfer to related tasks or different models? This would greatly affect the practical utility—if every prompt needs task-specific optimization, the compute cost becomes prohibitive.

4. Optimizer architecture choices under-explored: You use GPT-4o and Claude 3.5 Sonnet as universal optimizers. But there's no analysis of whether certain optimizers work better for certain target models. Is there a "universal optimal optimizer"? Or should the optimizer match the target model's family? For instance, does using a Claude optimizer work better when optimizing Claude prompts vs. Llama prompts? This could have practical implications for cost and efficiency.

5. Semantic neighborhood definition: While you provide formal definitions (Equations 3-5), the practical instantiation of "semantic dissimilarity" dₜₑₓₜ and how you ensure perturbations truly preserve semantics while revealing sharpness could be explained more clearly. What prevents the generator from creating trivial or overly conservative perturbations?

### Questions
Diversity vs. Robustness Trade-off: Have you tested this on any open-ended generation tasks? I'm genuinely curious whether there's a fundamental tension between robustness (flat neighborhoods) and diversity (exploring semantic space). For tasks where we want multiple valid outputs, does TARE hurt performance?

Reasoning Patterns: Do you observe any patterns in how TARE performs on different reasoning types? For example, does it help more on tasks requiring arithmetic precision vs. logical deduction? Understanding this could guide when to apply the method.

Thinking Models: Given the rise of o1-style reasoning models (Qwen QwQ, Mistral's reasoning variants), have you considered testing TARE on these? Do prompts optimized for standard models transfer to thinking models, or do the additional reasoning tokens fundamentally change the prompt landscape?

Cross-Task Transfer: If I optimize a prompt for GSM8K using TARE, does it work well on other math datasets (e.g., MATH, SVAMPs)? How much does the optimized prompt overfit to the specific task distribution?

Optimizer Analysis: In Table 1, you report results for GPT-4o and Claude 3.5 Sonnet backbones. But is there an analysis of which optimizer works best for which target model? Or is the optimizer choice model-agnostic? It would be interesting to see if using the target model itself as the optimizer (when possible) yields better results.

Computational Cost: While you mention computational practicality, what is the wall-clock time and API cost comparison between TARE/ATARE and baselines? For practitioners, this matters a lot.

Failure Cases: Are there scenarios where TARE makes prompts worse? Understanding failure modes would help users know when to apply the method.

### Soundness
3

### Presentation
3

### Contribution
3
