# SK2Decompile: LLM-based Two-Phase Binary Decompilation from Skeleton to Skin

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 8, 6, 4, 6

## Abstract
Large Language Models (LLMs) have emerged as a promising approach for binary decompilation. However, the existing LLM-based decompilers still are somewhat limited in effectively presenting a program's source-level structure with its original identifiers.
To mitigate this, we introduce SK2Decompile, a novel two-phase approach to decompile from the skeleton (semantic structure) to the skin (identifier) of programs. Specifically, we first apply a Structure Recovery model to translate a program's binary code to an Intermediate Representation (IR) as deriving the program's "skeleton", i.e., preserving control flow and data structures while obfuscating all identifiers with generic placeholders. We also apply reinforcement learning to reward the model for producing program structures that adhere to the syntactic and semantic rules expected by compilers. Second, we apply an Identifier Naming model to produce meaningful identifiers which reflect actual program semantics as deriving the program's "skin". We train the Identifier Naming model with a separate reinforcement learning objective that rewards the semantic similarity between its predictions and the reference code. Such a two-phase decompilation process facilitates advancing the correctness and readability of decompilation independently.
Our evaluations indicate that SK2Decompile, significantly outperforms the SOTA baselines, achieving 21.6% average re-executability rate gain over GPT-5-mini on the HumanEval dataset and 29.4% average R2I improvement over Idioms on the GitHub2025 benchmark.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
SK2Decompile proposes a two-phase LLM-based decompilation framework that separately reconstructs:

1. Program structure (skeleton) via Structure Recovery from binary to IR.

2. Identifier semantics (skin) via Identifier Naming from IR to human-readable source code.

Both phases are fine-tuned with reinforcement learning (RL):

1. Structure Recovery RL uses compiler feedback and placeholder accuracy.

2. Identifier Naming RL uses semantic cosine similarity between generated and reference code embeddings.

### Strengths
Originality

* Two-phase “skeleton→skin” formulation reduces a monolithic task to two focused ones. 
* IR defined as obfuscated source, motivated by the Information Bottleneck (compression vs relevance). 
* Phase-specific RL: compiler/placeholder Jaccard for structure; embedding cosine for naming/readability. 

Quality

* Clear IR generation algorithm and training details (SFT→GRPO RL).
* Strong, multi-metric benchmarks (re-executability, R2I, GPT-Judge) across diverse suites. 
* SOTA gains (e.g., +21.6% re-exec on HumanEval; +29.4% R2I on GitHub2025) with solid ablations isolating where gains come from. 

 Clarity

* Effective figures (motivation, framework, case study) and precise math/rewards. 
* Reproducibility statement and artifact release. 

Significance

* Practical impact: higher re-executability/readability; paradigm likely transferable to other program-recovery tasks.

### Weaknesses
Experimental coverage

1. Architecture/compiler diversity is narrow. All binaries are x86/Linux; no ARM/MIPS, MSVC/Windows, or ICC variants. This limits generalization. Action: add cross-architecture (ARM64) and cross-compiler (MSVC) evaluations; report deltas by ISA/ABI and OS. 
2. Language scope is mostly C/C-like. No results for Rust/Go/C++ templates where type systems and naming semantics differ. Action: evaluate at least one non-C family (Rust, Go) and modern C++ (templates/RAII) to test Identifier Naming generality. 

Baselines & fairness

No head-to-head with classic de-compilers. Action: add side-by-side comparisons (functional pass rate, R2I, blinded human readability) using Ghidra/Hex-Rays outputs to contextualize LLM gains.

### Questions
- RL reward design & potential leakage:
You “provide the compiler with the header of the ground-truth IR” to check compilability and grant reward.
• Can you quantify how much this leaks structure/type information compared to a realistic setting?
• Please add an ablation with no ground truth headers and report re-executability/R2I deltas.

- RL stability, sensitivity, and cost:
You apply GRPO on 50k samples due to compute limits but report no convergence/variance details.
• Please provide training curves (reward, validation re-exec/R2I) and GPU hours, and batch/length constraints.
• Any reward hacking observed?

- Naming reward via a single embedding model: Identifier Naming uses cosine similarity with qwen-embedding-0.6B
• How sensitive are results to the embedding choice? Please compare against alternative code/text-code embeddings and report correlation with identifier-level F1 and human ratings.

### Soundness
3

### Presentation
3

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
The paper tackles binary decompilation. Traditional tools like Ghidra or IDA produce code that works but is hard to read. LLM-based decompilers make code readable but often wrong.

SK2Decompile proposes a 2-phase approach:
- Structure Recovery (“skeleton”): translate binary into IR that keeps control flow but with placeholder symbols. 
- Identifier Naming (“skin”): fills in meaningful names for the symbols.

Both phases use RL. The reward of the phase 1 is IR compilability + structural corectness. Phase 2 is rewarded if the predicted identifiers are semantically similar)

The method improves both functional correctness and readability independently. Results show large gains over prior systems (GPT-5-mini, LLM4Decompile).

### Strengths
- Reproducibility effort.
- Clever and clear conceptual decomposition in 2 different parts. Well-motivated.
- Interesting and well-principled use of the information bottleneck principle.
- Comprehensive evaluations, and ablation study. Different optimization levels evaluated, diverse benchmarks.
- Very strong quantitative results.

### Weaknesses
- The 2-level decomposition of decompilation is not novel. The particular implementation is. I don't think the paper itself makes a good job at clarifying these 2 levels. E.g., https://arxiv.org/abs/2103.12801,
- Limited scope in languages and targets.
- Algorithm 1 could better specify how “names to preserve” FP are determined automatically.
- Minor: the paper could greatly improve in clarity with the help of some diagrams.

Nits: 
- some citations are incorrectly formatted. 
- Incorrect spacing in some footnote marks.
- “find out it achieves” → “find that it achieves.”
-  “we analogize … to the human body” -> “we draw an analogy.”
- Inconsistencies in capitalization of “Re-Executability” vs “Re-executability.”

### Questions
Do you have any intuition on how GPT-Judge works for decompilation? What criteria does the model seem to be using?

How robust is SK2Decompile to compiler obfuscations or optimizations beyond O3?

Can the same IR concept be used cross-language (e.g., C++, Rust)? And with other assembly yargets?

How much of the gain comes from RL versus just having a cleaner IR target?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper introduces SK2Decompile, a two phase binary decompilation strategy for binary lifting. The first phase focuses on lifting to the "skeleton" of the code, at the IR level, followed by the second phase which aims to provide structured recovery of meaningful identifiers in the actual program semantics. The authors employ both language models and RL for their approach, and the tool is evaluated on multiple benchmarks, and compared to other LLM lifters.

### Strengths
+ Important, established problem, with lots of headroom for innovation
+ The 2-phase approach is quite interesting, and I appreciated that contribution

### Weaknesses
- The baseline is not strong: none of the proposed benchmarks are "realistic" in their complexity nor size, which makes me question the feasibility of this technique to generalize beyond the studied benchmarks.
- There are no rule-based tools being compared. In particular, McToll is a well respected tool for decompilation, and I'd be very curious to see how it performs relative to SK2Decompile. This is important, because while LLMs are useful tools, rule-based tools should really be considered state-of-the-art here (as that is what they are replacing), not just other LLM lifters.
- No true analysis on what fails and what doesn't fail in the decompilation process (some explainability). The ablation study helps explain where some of the performance improvements arise, but does not give practical description of what programs failed to lift, and why that was the case.

### Questions
While the idea is good, the baseline for evaluation is quite underwhelming to this reviewer. Instead of a "basic" benchmark suite such as HumanEval and others, how would such a tool perform on a larger and more realistic codebase, such as BringUpBench?

Additionally, modern rule-based lifters actually perform quite well on "simple" benchmarks such as HE, but struggle against more complex benchmarks. This is true for McToll and a few others. That said, McToll does surprisingly well HumanEval-like benchmarks, which to me indicates that should be the best baseline available, not GPT-5-Mini. If SK2Decompile (or other LLM-based lifters for that matter) do not match McToll level accuracy, then their contribution is questionable.

On the flip side, BringUpBench is also challenging from a rule-based approach: how does SK2Decompile (and other lifters) perform on such a benchmark suite?

To summarize:
- The idea is quite interesting (splitting the decompilation into two phases); however, the evaluation is not well executed given the comparison points (there was no rule-based technique), and the benchmarks are simple.
- If the authors are able to generate results on BringUpBench-level complexity with their tool, I would increase my score. 
- How does this tool fare compared to rule-based tools, such as McToll, when it comes to re-compilation. 

I will say that the R2I and GPT-Judge results are interesting; but practically speaking the recompilation is what is very interesting to this reviewer, as that opens the door for lots of future directions in binary modifications and optimizations.

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
4

### Summary
This paper proposes SK2Decompile, an LLM-based binary decompilation approach.

Section 2 discusses existing decompilation approaches, and provides a motivational example.

Section 3 proposes SK2Decompile. It consists of (1) structure recovery yielding an intermediate representation (IR); and (2) identifier naming. Each phase uses a model that has been subjected to supervised fine-tuning and reinforcement learning. The design of the intermediate representation is cast as an "Information Bottleneck" optimization task. The IR produced traverses an abstract syntax tree to determine names and types of identifiers. The identifier naming process employs sequence-to-sequence training followed by reinforcement learning to aligh the outputs with compiler preference and type constraints.

Section 4 covers the evaluation. Exebench is used as training data. The evaluation compares against three other decompilers: GPT5-mini, LLM4Decompile, and Idioms. The evaluation benchmarks used are HumanEval and MBPP. The evaluation metrics used include re-executability rate, the relative readability metric for decompiled code R2I, and GPT-Judge, a score between 1 and 5 reflecting the meaningfulness of variable names. The results mark SK2Decompile as winner in almost all cases. An abblation study illustrates the need for all components (src, ir, and rl). Lastly, the naming benefits are illustrated on a dedicated case.

### Strengths
- LLM4Decompile and Idioms are adequate baselines to compare against
- The results show good performance
- The abblation illustrates the need of the various components
- Meaningful ethics statement

### Weaknesses
- Unclear why re-executability results are limited to just two benchmarks
- HumanEval is a dataset that is not representative for actual decompilation problems -- the only justification can be that it is used in earlier studies.
- I would expect a feedback loop trying to fix errors could improve reproducibility. This could be separately discussed
- The 'case study' is not a case study in a methodological meaning of the word. Is a useful illustration, but not a critical qualitative assessment of the capabilities of the approach proposed.
- I would like to see a more critical (qualitative) assessments of cases not properly handled by SK2Decompile, in order to identify challenges the research community should focus on next

### Questions
- Why are there no re-executability results for exebench? Doesn't it come with a test suite? I don't understand the remark about stripping blocking this on p.6.
- Nova (and other models) are easily dismissed as they do not "provide details about their data preprocessing approaches or do not release their models" -- but I'm not convinced by this -- what is the best you can do?

### Soundness
3

### Presentation
3

### Contribution
3
