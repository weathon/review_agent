# AbstRaL: Augmenting LLMs' Reasoning by Reinforcing Abstract Thinking

- Decision: Accept (Poster)
- Scores: 8, 8, 2, 2

## Abstract
Recent studies have shown that large language models (LLMs), especially smaller ones, often lack robustness in grade school math (GSM)  reasoning. In particular, they tend to experience performance drops when faced with distribution shifts, such as changes to numerical or nominal variables, or insertions of distracting clauses. A possible strategy to address this involves generating synthetic data to further "instantiate" reasoning problems on potential variations. In this work, we instead focus on the strategy of ``abstracting'' reasoning problems. This not only helps counteract distribution shifts but also facilitates the connection to symbolic tools for deriving solutions. Focusing on GSM, we find that this abstraction process is better acquired through reinforcement learning (RL) than just supervised fine-tuning, which often fails to produce faithful abstractions. Our method, AbstRaL---which promotes abstract reasoning in LLMs using RL on granular abstraction data---significantly mitigates performance degradation on recent GSM perturbation benchmarks. Besides, improving GSM robustness via AbstRaL is shown to also implicitly benefit LLMs' capabilities on OOD mathematical and general reasoning tasks, indicating that abstract thinking broadly enables better generalizability.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces a novel reasoning technique that abstracts mathematical problems *before* solving them. The abstraction process goes from recognizing input variables to generating an abstraction that can be solved using external math tools or by an LLM tailored for the task. The authors show that this strategy improves the performance of many LLMs on two variants of the GSM dataset.

### Strengths
- The idea of uncovering the underlying abstraction of a math problem and using this abstraction to solve it (rather than the potentially noisy original text) is elegant.
- The abstraction pipeline can be implemented with relatively simple components, which means this direction is likely to used in practice.
- The results are consistently better than multiple strong baselines, for nine LLMs and two variants of the GSM dataset.

### Weaknesses
- While I think the contributions of this paper are sufficient for a conference publication, the authors focus on a single style of math problems since both datasets are spinoffs of the GSM dataset. The paper's claims would be stronger if at least another math dataset was used.
- In the same vein, previous work has consistently shown that mathematical reasoning problems are sensitive to noise in the problem formulation. How sensitive is this method to noise? See, for example, this new dataset for this investigation: https://ymingl.com/GSMDC/
- While I understand that math reasoning is the focus here, which, again, is sufficient for a conference paper, the authors should discuss how this work would be adapted for other types of reasoning scenarios.

### Questions
- What is the accuracy of each step of abstraction process summarized in Section 3?
- Eq 1: a better name for the reward formula introduced here would be "symbolic similarity reward" since it's computed as (1 - distance).
- Are the results shown in Table 1 statistically significantly better than the next strongest baseline in Table 1?
- Typo in Abstract: "we instead focuses"

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper presents an interesting technique of leveraging symbolic abstract thinking as a way to improve the performance of LLM's ability to solve math problems. Instead of directly direct solutions from surface form, LLMs recognize input parameters, symbolic formula to compute the answer, and then the answer is derived by symbolic execution. 

The paper presents a data generation pipeline, that (1) rewrites instance into parametric representation, (2) sample CoT, and (3) rewrite CoT into abstract CoT and use symbolic derivation to calculate final answers. This way, the model can be trained with first SFT and then RL to perform abstract reasoning.

Results show the performance of Abstral model on par and better than other reasoning based approaches. Ablation shows the effectiveness of RL, contexts and the reward design.

### Strengths
- This paper presents a quite interesting approach for math reasoning. Instead of directly reason on the text space, it leverages formula inference to derive the right result. This effectively turn math reasoning into a programming task (i.e., extract parameter, write a program, and run the program to get result) and assemble results with response.

- The dataset construction leverages existing CoT to construct generalizable abstract trace.

- Evaluation is pretty solid, especially with ablation with SFT vs RL's contribution to the final performance.

### Weaknesses
- the proposed abstraction composition is quite complicated, especially given the rewriting + retrieval (I personally think it's better called extraction as I was confused with document retrieval at first glance). Will a simpler way to construct reasoning, i.e, ask the model to directly generate formulas on top of basic COT (e.g., in a separate section) achieve the similar performance? Especially considering directly generate a small snippet of code for computation and ask LLM to assemble back to response. Is this just a format / style issue, or some more fundamental difference?

- r_symbolic seems quite important. But it's not clear reasons behind difference of results without no r_symbolic, some qualitative analysis compare results with and without r_symbolic would be helpful. I think r_symbolic is important kinda hint there may not be sufficient SFT for guiding formula generation.

### Questions
check the section above.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces AbstRaL, a framework designed to enhance the reasoning robustness of LLMs, particularly for GSM problems. Instead of relying on extensive data augmentation ("instantiation"), the authors propose to explicitly teach LLMs to first generate an abstract representation of a problem. This abstraction, being invariant to surface-form variations, is intended to make the model more robust to OOD shifts like changes in numbers, names, or the addition of distractors. The core of the method involves a new, fine-grained data format called GranulAR and a RL scheme built on top of SFT. The RL component uses novel, model-free rewards based on final answer correctness and the symbolic similarity of the generated abstraction to a gold reference, aiming to improve the faithfulness of the abstract reasoning process. The authors demonstrate that AbstRaL significantly improves performance on GSM robustness benchmarks (GSM-Symbolic, GSM-Plus) and shows some generalization benefits to other OOD tasks.

### Strengths
1. The central idea of moving from "instantiation" to "abstraction" is a strong conceptual contribution. Explicitly teaching the model a reasoning schema that is invariant to surface details is a promising direction for improving OOD robustness.

2.  The proposed AbstRaL framework is well-thought-out. The creation of the GranulAR data format, which blends socratic decomposition with symbolic abstraction, is a clever way to make the task more tractable for LLMs. The use of reinforcement learning with a custom symbolic distance reward (`rsymbolic`) to enforce faithfulness is a significant and novel part of the method.

### Weaknesses
1.  The framework's dependence on a vastly more powerful model (Llama-3.3-70B) for critical data generation steps is a major weakness. Both the initial "Condition Recognition" and the subsequent "Abstract Reasoning Chain Rewriting" are performed by this moel. This raises the concern that the improvements seen in smaller models are largely due to knowledge distillation from a superior model, rather than the inherent power of the AbstRaL learning scheme itself. 

2. The method is evaluated exclusively on grade school math problems, where the underlying abstractions (arithmetic operations) are simple and well-defined. It is highly uncertain how this framework would translate to more complex domains with ambiguous or non-mathematical abstractions, such as legal reasoning, commonsense planning, or scientific hypothesis generation. The paper does not discuss these limitations or provide a path toward generalization.

3.   The primary comparison is between AbstRaL (SFT + RL) and baselines that are either prompting-based (CoT-8S) or SFT-only (CoA, AoT). The `w/o RL` ablation in Table 2 shows a massive performance drop, suggesting that the GranulAR data format with SFT alone is not very effective. A more direct and fair comparison would be between "AbstRaL-SFT" and other SFT-based abstraction methods. Furthermore, for the stronger Qwen2.5-Math-7B model, AbstRaL underperforms CoT-RL on the original, in-distribution data (91.0 vs 96.0), indicating a potential trade-off where improving OOD robustness comes at the cost of ID performance.

4.  The paper claims that AbstRaL "implicitly benefit more general reasoning capabilities." However, the results in Table 3 show that the improvements on general reasoning benchmarks like MMLU are marginal at best. The most significant gains are on other math-related datasets (MATH, AIME24), which is less surprising and does not fully support the claim of broad generalizability.

### Questions
1.  How can you be sure that the performance gains are not primarily a result of distilling its superior reasoning capabilities into the smaller models? Have you tried using a weaker model as the oracle to see how that affects the final performance?

2.  The `w/o RL` ablation shows that SFT on GranulAR data performs poorly, often worse than simple CoT prompting. Does this imply that the proposed abstraction format is too difficult for models to learn via SFT alone and that the complex RL setup is a necessary crutch? 

3.  Regarding the scalability to other domains: What do you foresee as the main challenges in applying AbstRaL to problems where the "abstraction" is not a simple mathematical formula? For example, in a commonsense reasoning task, what would the "conditions" C and the "abstraction" A look like, and how would they be reliably extracted?

4.  For the strongest models, AbstRaL seems to trade in-distribution performance for OOD robustness (e.g., Table 1, Qwen2.5-Math-7B, Origin 100). Do you consider this an inherent trade-off of the method? Why do you think this performance degradation on the original task occurs?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
Authors introduce AbstRaL (Abstraction Reinforcement Learning), a framework that uses abstract intermediate representations to improving the robustness of LLMs in math word problems. Authors propose teaching models to abstract reasoning problems—that is, to represent problems symbolically and reason at a higher level of abstraction. AbstRaL integrates RL with a new Granularly-decomposed Abstract Reasoning (GranulAR) dataset that blends CoT reasoning with symbolic abstraction. The method encourages LLMs to generate faithful abstract representations that can be solved via symbolic tools such as equation solvers. Experiments are done two LLMs mainly and results on new benchmarks (GSM-Symbolic, GSM-Plus) show improved robustness under numerical, contextual, and distractor perturbations.

### Strengths
1. Combination of learning abstractions using an SFT-RL framework seems interesting.
2. Though authors have used only two main LLMs, the ablation studies are done htoughtfully.
3. Using the socratic version of GSM8k to rewrite the abstract intermediate steps (/decomposed questions) is interesting. Therefore, makes the data somewhat valuable.

### Weaknesses
The paper suffers largely on novelty, lack of sufficient experimentation, broad claims which are untrue.

Novelty: Neither decomposition nor abstraction is new in symbolic math, math word problem solving. There are too many papers, with too many variations and ideas. Authors also do not specify any contributions. So, I am not even sure if this is a central claim.

Lack of sufficient Experimentations: Only two small LMs are chosen to support very broad claims. Even the variations of GSM chosen are only two. There are many variants: GSM-Hard, i-GSM, GSM8k_MORE. And this is just GSM8k. There are so many fantastic and important datasets out there. 

Broad claims: See below.

Others:
1. Limited comparison to earlier equation-prediction approaches: Prior work (e.g., Mitra et al. 2018; Agarwal et al. 2021) also translated math word problems into equations for symbolic solving (more below)
2. The generation of GranulAR data relies heavily on a large teacher model (Llama-3.3-70B). Whats the effect of using such a model here. What if we use smaller models.
3. While the method checks answer correctness, it is unclear how the necessary and sufficient quality of abstract decompositions is ensured or whether abstraction errors propagate through training.

### Questions
L77: There are other work which used to convert the math word problems to equations and then use tools to solve it. Why is this different?

For example, Agarwal et al. ICLR 2021 MathAI showed that it was fundamentally difficult for Transformers to perform numeric addition, multiplication. So, using symbols in place and postponing the calculation worked much better. Other work such as Mitra et al. 2018 and many, also predicted equations from math words as an intermediate step. This was the norm in pre-LLM era.

L201: The abstract answer $\mathcal{Y}_\mathcal{A}$ seems to be generated by LLMs. It involves a crucial step of decomposition which needs to have a necessary and sufficiency check? Was it done by any means?

L250: This will miss any equivalent expressions, lets say which is a simplification or complicated version of A.   Can't we look at AST matches?

L268: These types of claims and statements are very generic. Many LLMs are trained code corpora, sees a lot of different structured languages. Why should it not be inherently incapable of such abstraction?

L375: Doesn't it show that the step-by-step decomposition is helping more? Based on the drops in Table 2, thats what is implying. Am I missing anything here?

L401: Its interesting that removing the abstraction does not help much with distractors. But these are small models (only two). Would it affect larger ones as well? Is it possible to show this on large models (gpt-oss size).

### Soundness
2

### Presentation
2

### Contribution
2
