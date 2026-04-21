# Divide and Translate: Compositional First-Order Logic Translation and Verification for Complex Logical Reasoning

- Avg Score: 5.67
- Decision: Accept (Poster)
- Scores: 6, 6, 5

## Abstract
Complex logical reasoning tasks require a long sequence of reasoning, which a large language model (LLM) with chain-of-thought prompting still falls short. To alleviate this issue, neurosymbolic approaches incorporate a symbolic solver. Specifically, an LLM only translates a natural language problem into a satisfiability (SAT) problem that consists of first-order logic formulas, and a sound symbolic solver returns a mathematically correct solution. However, we discover that LLMs have difficulties to capture complex logical semantics hidden in the natural language during translation. To resolve this limitation, we propose a Compositional First-Order Logic Translation. An LLM first parses a natural language sentence into newly defined logical dependency structures that consist of an atomic subsentence and its dependents, then sequentially translate the parsed subsentences. Since multiple logical dependency structures and sequential translations are possible for a single sentence, we also introduce two Verification algorithms to ensure more reliable results. We utilize an SAT solver to rigorously compare semantics of generated first-order logic formulas and select the most probable one. We evaluate the proposed method, dubbed CLOVER, on seven logical reasoning benchmarks and show that it outperforms the previous neurosymbolic approaches and achieves new state-of-the-art results.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
Introduces a new algorithm, CLOVER, for solving logic questions in natural language, specifically by addressing the challenges in parsing natural language into the correct first-order logic so that an SAT solver can determine the answer.  To do this, the paper proposes translating the question into smaller pieces that accumulate based on how each logic unit in natural language relates to other logic units until the resulting sentence is the final sentence that needs to be tested.  Each accumulation from the previous step, including the final sentence, is translated into first-order logic, then the paper introduces two novel verification methods that check if the translations are valid and if there are contradictions in the final translation. The paper shows that with this accumulation strategy with their two verifiers, their method can outperform all baselines (including Logic-LM, a baseline that similarly translates sentences into FOL for SAT solvers) on common logic datasets like AR-LSAT, FOLIO, and ProofWriter.

### Strengths
- The results are promising; the authors do a fantastic job of motivating their new algorithm CLOVER by showing common failures of previous methods like Logic-LM, then show that their method fixes many of these errors (leading to the performance boost reported in Table 1).

- The method here is pretty novel. Breaking down sentences into atoms isn't too novel, but I haven't seen someone decode them all individually and progressively (tapping into the auto-regressive natural of LMs) to improve the performance of the translation.  The verification algorithms seem pretty intuitive (although they are described complexly), but again, despite being intuitive, I think they are fairly novel as well.

### Weaknesses
- The ablations in Table 3 need to be explained more clearly and then discussed.  What is "Is Clover?" and why is the simplest ablation (no clover, direct translation, no verification / essentially none of the new things introduced in this paper) outperforming Logic-LM on AR-LSAT by 10.9% already from Table 1? Does this mean that your direct translation prompt already improves over complex algorithms like Logic-LM? If so, this deflates the papers impact, so it should be addressed (it's possible I am missing something, but others will catch this too, so it's best to explain it away.)

- I believe the paper would benefit greatly from expanding on the models being evaluated; right now, only GPT-4o and GPT-4o-mini are evaluated.  Showing that CLOVER consistently outperforms the baseline methods across model classes would improve the impact of this work.

- (minor point) There is no discussion of inference-time compute costs for CLOVER vs. the other baselines.  I imagine the inference cost is significantly higher, but I am unsure how much.  Is this negligible compared to Logic-LM?  Is there a way to compare CLOVER with baselines that use the equivalent amount of compute during inference?  I think much of this point could be explained away with a textual justification (i.e., this isn't possible, or the compute costs are nearly equivalent, etc.), but I do think it should be mentioned.

- (minor point) Clarity in section 3 could be improved. I would use the example in Figure 2 to clearly define each variable mentioned in the text to help readers follow your algorithm.  For instance, defining with x^prep, T, phi_k, the mapping NL(phi), etc., with values from Figure 2 would help readers follow significantly. This could also be done in Figure 2 if you mark which parts of it are which variables.  The text gets very dense with variables that are derived from other variables quickly; having these concrete instantiations really helps.

### Questions
- I'm curious why the execution rate increases when using CLOVER.  As I read the methods section, it looked like CLOVER primarily helps with execution accuracy, but I didn't see much about how it would help repair/fix/generate better code for the SAT solver.  

- It's reported that "CLOVER’s errors are primarily caused by preprocessing and other errors, which takes 78.6% of the total errors", do you have examples of this? Is this an error in the accumulation stage?  I think the paper does a great job of explaining where Logic-LM fails and why CLOVER is needed, but I think expanding on CLOVER errors is just as important to show where researchers can look next.

- How much of the performance gain seen in CLOVER is due to a higher execution rate (runnable code)? I think expanding on how the metrics in Table 2 are computed would be helpful. For example is `Execution Acc = (correct_and_executable_programs / all)` or `Execution Acc = (correct_and_executable_programs / executable_programs)`.  The latter, I think, helps distinguish if you are generating better executable problems or if you are only improving the execution rate (which maybe there is a simple fix to Logic-LM to help it create better executable problems)?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
Authors propose a novel method of using LLMs to translate natural language descriptions into a set of first-order logical forms. This novel method decomposes this challenging task into two steps. The first step is to translate a long and complex sentence into a number of short sentences, the second step is to translate each short sentence into simple first-order logical forms and the connections between/among these short sentences into corresponding logical connectors. Experiments on seven benchmark datasets greatly outperform current SOTA level.

### Strengths
It is reasonable to improve the translation quality by decomposing a complex sentence into several shorter sentences. Using SAT solvers certainly improve the quality.

### Weaknesses
Not all natural language sentences can be translated to first-order logic forms. Authors did not discuss what sentences cannot be translated. 

Authors use symbolic SAT solver in evaluating and selecting correct first-order logical forms.  This limits the method only for the case where SAT solvers work. 

Theoretically, the meaning of natural language is not logical formula. This work is valued within fixed benchmark datasets. 
 
The formalism of the paper is not easy to read.

### Questions
1. line 115: "To save computational cost, we compare each one of logically equivalent formulas". You probably mean to "compare each logically equivalent formula". How can this save computational cost?

2. Line 149: how to read this formula in natural language? 

3. What is the output for the sentence "A barber shaves all who do not shave themselves."? 

4. How are "Declarations" created? 

5. How to decide a sentence not fit for your system? (or how to decide an unintended input sentence?)

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper introduces CLOVER, an approach designed to enhance the translation of natural language logical problems into logical code, thereby improving the performance of language models on logical reasoning benchmarks. CLOVER achieves this by compositional translation of natural language into first-order logic and verification of logical semantics. The method involves parsing natural language sentences into logical dependency structures, translating these into first-order logic formulas, and employing verification algorithms to ensure accuracy. The authors demonstrate CLOVER's effectiveness on seven logical reasoning benchmarks, showing it outperforms previous neurosymbolic approaches.

### Strengths
- The paper presents a new approach by breaking down compositional translation into smaller steps and combining it with verification for logical reasoning tasks, leading to improved performance in neurosymbolic translation.
- The experimental results are robust using GPT4-o, showing improvements over other methods across multiple benchmarks.
- The authors propose two SAT-based first-order logic verification algorithms for selecting a sample from LLMs' logical code generations.

### Weaknesses
- The approach is primarily applicable to problems that can be represented in a SAT solver, limiting its generalizability to other reasoning datasets, such as those involving mathematical equations or visual components, e.g., MATH dataset.
- The core idea of breaking down tasks into subtasks and using multiple samples and tests (e.g., verification, self-reflection, deterministic tests) to select the best generation is not novel.
- The paper lacks comparison with chain-of-thought (CoT) based methods designed to improve implicit reasoning of language models, as in "reliable reasoning beyond natural language". These methods help the model extract information that is implied but not directly stated by interleaving natural language comments with logical code, and can alleviate the translation bottlenecks identified.
- The paper only reports results using one language model, making it unclear if the method would improve performance across different models and weakening the experimental results.

### Questions
1. Is it possible to extend CLOVER to improve performance on tasks that involve reasoning with data formats beyond natural language, such as mathematical equations or visual reasoning tasks?
2. Can the authors provide more insights into how CLOVER compares with CoT-based methods designed for improving implicit reasoning of LLMs?
3. Why not test CLOVER on a wider range of language models to assess its generalizability?

### Soundness
3

### Presentation
2

### Contribution
2
