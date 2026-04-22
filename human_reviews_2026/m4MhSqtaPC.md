# Enhancing Neural Decompilation with Code-aware Fine-Tuning and Inference-time Refinement

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Language models (LMs) hold promise for automating binary decompilation by translating low-level assembly into high-level source code, helping code security analysis. However, current LM-based methods struggle with the complex control and data flows in real-world programs. 
We present TuneRDEC, a code structure-aware
LM-based approach to binary decompilation. TuneRDEC combines a task-specific LM, fine-tuned on domain-specific data, with a general-purpose LM trained on generic corpora. The task-specific LM uses abstract syntax tree (AST) analysis to assign higher loss weights to key language constructs, such as loops, conditionals, and pointers, prioritizing their importance during training. At inference time, TuneRDEC employs an iterative self-refinement process guided by compiler feedback and test-driven prompts, leveraging the general-purpose LM to improve the decompilation output.
We evaluate our approach on the HumanEval-Decompile dataset. The results show significant improvements in code readability, functional correctness, and robustness compared to state-of-the-art neural decompilation methods and an industry-strength decompiler.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces TuneRDEC, a language model–based framework designed to enhance binary decompilation by translating low-level assembly into readable and functionally accurate source code. Traditional LM-based decompilers often fail to handle the complex control and data flows present in real-world binaries. TuneRDEC addresses this limitation through a code structure-aware design that integrates a task-specific LM, fine-tuned on domain-relevant data, with a general-purpose LM trained on broader code corpora. The task-specific model leverages abstract syntax tree (AST) analysis to emphasize critical constructs such as loops, conditionals, and pointers during training. During inference, an iterative self-refinement process guided by compiler feedback and test-based prompting further improves code quality. Experiments on the HumanEval-Decompile dataset demonstrate that TuneRDEC achieves substantial gains in code readability, functional correctness, and robustness compared to both existing neural decompilers and an industry-grade tool.

### Strengths
The trained model enhances decompilation results by enforcing structural constraints on the generated code. When combined with a general-purpose model, this approach achieves over 90% accuracy, demonstrating strong performance improvements.

### Weaknesses
The token weighting scheme is tuned on the evaluation set, but an ablation study is needed to assess the impact of different weighting configurations—for example, assigning high weights to individual categories or testing various combinations. Moreover, since the evaluation is conducted solely on the HumanEval dataset, which contains only 160 cases, the reported benefits of the weighted loss may not be statistically reliable and could be influenced by random fluctuations.

Regarding the motivation for training a new decompiler, the performance improvement over prior decompilers is marginal (about 5 points in execution rate) compared to the much larger gains (around 40 points) attributed to the general-purpose LM. The paper should justify why a new model is necessary rather than directly feeding the outputs from existing tools (e.g., IDA, Ghidra, or other LLM-based decompilers) into GPT for refinement. Ablation experiments should be conducted under identical refinement conditions to compare this approach directly.

Furthermore, the contribution of the self-refinement process is unclear when compared to DEGPT [1] and DECLLM [2], both of which also use GPT-based iterative refinement guided by feedback from tests or other external signals. 

Finally, relying solely on HumanEval, a synthetic and small dataset, limits the validity of the results. Evaluations on more realistic benchmarks—such as DecompileBench [3] or similar datasets—are necessary to substantiate the method’s practical effectiveness.

[1] Hu, Peiwei, Ruigang Liang, and Kai Chen. "Degpt: Optimizing decompiler output with llm."
[2] Wong, Wai Kin, et al. "DecLLM: LLM-Augmented Recompilable Decompilation for Enabling Programmatic Use of Decompiled Code."
[3] Gao, Zeyu, et al. "DecompileBench: A Comprehensive Benchmark for Evaluating Decompilers in Real-World Scenarios."

### Questions
The code and model are not open-sourced, which significantly limits the reproducibility and fairness of the paper’s evaluation.

### Soundness
2

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
4

### Summary
The paper proposes TUNERDEC, a new system to translate compiled binary code (assembly) back into readable C. 

Traditional decompilers produce hard-to-read-code and doesn't always match the original logic. Neural/LM-based approaches are a promising alternative, but struggle with complex control flow, etc.

TUNERDEC uses two LMs: a task-specific LM (FT CodeLama trained to translate C-assembly pairs), and a general LM (Deepseek) that iterates on the initial translation using compiler feedback and test results.

### Strengths
- Strong empirical results: perfect compilation rates, very high re-executability rates. Qualitatively, the code is more readable and faithful to the original input compared to other approaches. The eval compares with Ghidra, LLM4Decompile and other LM baselines.
- The figures are very clear and help to understand the paper.
- Interesting innovative design: combines both specialized LMs and general LMs with reasoning. AST-based weighting is a well-motivated idea.
- The training data preprocessing is well-detailed and sensible.

### Weaknesses
The main doubt of this paper is the evaluation. Re-compilabilty and re-executability are interesting metrics, but they don't imply correctness. Also, the HumanEval decompile appears to be almost saturated, any more challenging benchmarks you could evaluate on? The Leetcode evaluation in the appendix is interesting, but the kind of functions in Leetcode might be similar in spirit to the ones in HumanEval.

Relatedly, and very importantly, the paper doesn't mention decontamination efforts.

Small issues:

- Missing citations: https://squareslab.github.io/materials/KatzRNN2018.pdf, https://arxiv.org/abs/2309.14396
- decopmiler -> decompiler
- Table 3 is hard to read

### Questions
Does the input assembly contain symbols?

How was data overlap with HumanEval/ExeBench controlled concretely?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes an iterative LLM-based approach to decompilation called TuneRDEC.

Section 2 provides a motivational example, illustrating how the approach manages to decompile a binary for a recursive function to compute Fibonacci numbers.

Section 3 provides an overview of the approach, distinguishing a decompiler and a feedback loop to improve the decompiled code, using separate LLMs.

Section 4 explains the decompiler: It starts from CodeLlama, which is finetuned on (assembly code, source code) pairs. It uses a token scheme that gives higher weights to certain language constructs (loops, conditionals, ...) in the AST, in order to support reconstructing control flow logic. The training data uses ExeBench as starting point, using TF-IDF for deduplication and compilation at various optimization levels. The fine-tuning approach uses LoRA.

Section 5 covers the iterative LLM-based (DeepSeek) refinement process, which takes compilation or test execution errors, and prompts the LLM to revise the inferred C source code to fix the problems.

Section 6 discusses the evaluation setup. TuneRDEC is compared against various baselines, including classical Ghidra, three plain LLMs (DeepSeek, ChatGPT 4o, and CodeLlama), the dedicated LLM4Decompile-End, and just the own TuneRDEC decompiler without iterative refinement. Furthermore, DeepSeek and TunderDec itself are applied with three refinement iterations. Results are compared by measuing recompiliability, reexecutability, edit similarity, and CFG match accuracy. The evaluation datasets come from HumanEval-Decompile, compiled with four optimization flags.

Section 7 covers the results, which identify TuneRDEC as the winner in all cases. The section attributes the improvements to the "code-aware weighted token scheme, which helps the decompiler better capture code structure and generate more accurate decompiled output."

### Strengths
- The paper is very well written, including accessible examples, and thanks to that is easy to follow
- The paper addresses a relevant and challenging problem
- The experiments indicate TuneRDEC beats established decompilation approaches.
- The appendix includes an interesting analysis trying to assess CFG similarity in a nuanced way.

### Weaknesses
- The iterative refinement process is extremely simple, and not a novel contribution (as implicitly acknowledged in the related work section) yet prominently mentioned in the title and as third contribution
- The HumanEval-Decompile dataset is a very unrealistic dataset to use.
- Limitations of the approach and the evaluation are not discussed (such as scalability or level of realism of the code in the evaluation, and the reliance on availability of test cases)
- It is not clear why DecLLM is not used as a benchmark
- While the appendix seems promising, section 7 does a shallow job only in analyzing the reasons underlying the findings in table 3.

### Questions
- Why isn't DecLLM used as a benchmark?
- Why doesn't table 3 iclude CFG full match accuracy? Or phrased differently: Why does section 6.3 explain CFG full match accuracy if it isn't used?
- In Table 3, how can the IR version of DeepSeek for O2 recompiliability be lower than the non IR version? (84.8 vs 88.4)

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents TUNERDEC, a neural framework for binary-to-C decompilation that combines task-specific and general-purpose language models. The approach fine-tunes a 7B CodeLlama model using a code-aware weighted token scheme based on AST analysis, assigning higher loss weights to critical programming constructs (loops, conditionals, pointers). During inference, TUNERDEC employs iterative refinement using DeepSeek-V3/R1 with compiler feedback and test-driven prompts to improve decompilation quality. Evaluated on HumanEval-Decompile across four GCC optimization levels, TUNERDEC achieves 100% re-compilability, 91.6% re-executability, and 62.6% edit similarity, outperforming baselines including Ghidra, ChatGPT-4o, DeepSeek-V3, and LLM4Decompile.

### Strengths
- The code aware weighted token scheme represents a novel adaptation of AST-based loss weighting for decompilation, prioritizing critical language constructs during training. The combination of domain-specific fine-tuning with general-purpose LM refinement is a reasonable architectural choice, though not groundbreaking.
- The experimental results demonstrate clear improvements over baselines, with TUNERDEC achieving 91.6% re-executability compared to 47.7% for LLM4Decompile. The ablation study (TUNERDEC-w/o-IR vs TUNERDEC) effectively demonstrates the value of iterative refinement, showing a 38.2% absolute gain. The evaluation includes multiple optimization levels (O0-O3), which is important for assessing robustness to compiler transformations and the model shows improved performance over harder optimization levels.
- The work addresses an important problem in software security and reverse engineering. The 91.6% re-executability represents meaningful progress over prior neural decompilers. The demonstration that iterative refinement with general-purpose LMs can substantially improve decompilation quality has practical implications for tool development.

### Weaknesses
- The code-aware weighting scheme lacks rigorous justification. Why are loops weighted at 2.0 and conditionals at 1.8? 
- Only one benchmark (HumanEval-Decompile, 164 programs) is evaluated. No assessment on other source languages (C++, Rust) or other benchmarks.
- No analysis of failure cases. What characterizes the 8.4% that fail re-executability despite 100% re-compilability?
- Atleast the ablation of code-aware weighted loss vs default loss function should be part of the main paper
- What are the evaluation scores absent for models with IR configs for other models apart from Deepseek V3 ?
- Its a bit impractical to use a combination of a light weight model assisted with a LLM like Deepseek V3.

### Questions
- Refer to weaknesses
- The presentation of the paper needs some work : you should focus more on evaluation results and ablations, those are more interesting to prove your points about weighted loss function, iterative refinement.

### Soundness
3

### Presentation
3

### Contribution
3
