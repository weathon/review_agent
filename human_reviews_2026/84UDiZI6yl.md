# Improving Code Translation Correctness and Efficiency with Multi-Perspective Exploration and Difference-Aware Selection

- Decision: Reject
- Scores: 2, 2, 4, 8, 6

## Abstract
While large language models (LLMs) have greatly advanced the functional correctness of automated code translation systems, the runtime efficiency of translated programs has received comparatively little attention.
With the waning of Moore’s law, runtime efficiency has become as critical as functional correctness in evaluating program quality.
Our preliminary study reveals that LLM-translated programs often run slower than human-written ones, and this issue cannot be remedied through prompt engineering alone.
Therefore, our work proposes SwiftTrans, a code translation framework comprising two key stages:
(1) Multi-Perspective Exploration, where MpTranslator leverages parallel in-context learning (ICL) to generate diverse translation candidates;
and
(2) Difference-Aware Selection, where DiffSelector identifies the optimal candidate by explicitly comparing differences between translations.
We further introduce Hierarchical Guidance for MpTranslator and Ordinal Guidance for DiffSelector, enabling LLMs to better adapt to these two core components.
To evaluate the runtime efficiency of programs, we extend existing benchmarks, CodeNet and F2SBench, with efficiency-critical test cases and maximum runtime constraints on translated programs.
We also introduce SwiftBench, a new benchmark designed to evaluate whether translation models can improve the efficiency of programs when the source code exhibits inefficiencies.
Experimental results across all three benchmarks show that SwiftTrans achieves consistent improvements in both correctness and efficiency.
Notably, SwiftTrans built on Qwen2.5-7B surpasses current state-of-the-art models such as GPT-5 and training-based F2STrans.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This work introduces SwiftTrans, a new framework that improves both the correctness and runtime efficiency of code translated by LLMs. Unlike most existing code translation techniques that focus only on producing functionally correct code, SwiftTrans ensures the efficiency of translated code. The main contributions in SwiftTrans are 1) Multi-Perspective Exploration and 2) Difference-Aware Selection. To support evaluation of SwiftTrans, the authors extend existing benchmarks with efficiency-focused tests and introduce a new dataset called SwiftBench, which includes programs with intentional inefficiencies. Experiments show that SwiftTrans significantly outperforms strong baselines (including GPT-5 and F2STrans) using much smaller open-source models like Qwen2.5-7B.

### Strengths
- Joint approach for improving translation correctness and efficiency
- A new benchmark SwiftBench
- Improve performance of open-source models

### Weaknesses
- Reliance on weak benchmark. e.g., CodeNet.
- Efficiency only measured by runtime. What about memory usage? What about structural efficiency (recursive vs non-recursive)?
- Concerns on intentional inefficiencies 
- Tool runtime

### Questions
- CodeNet is known to be weakly tested. Available tests (in most cases only 1 test) are not rigorous. Therefore, relying on existing tests is not reliable to me. In SwiftBench, what is the code coverage and branch coverage of tests?

- What is the runtime of your tool? How fast is generating all those ICL examples?

- How realistic are your intentional inefficiencies in SwiftBench? How would you convince someone with scientific support?

- Recent work (https://arxiv.org/pdf/2412.14234, https://dl.acm.org/doi/10.1145/3729287, https://dl.acm.org/doi/10.1145/3729379, https://dl.acm.org/doi/10.1145/3729315) focus on repository-level code translation. Why the authors did not evaluate on benchmarks involving repository-level projects?

- What is the real-life implication of your work? It is hard to accept this works when not evaluated on complex benchmarks.

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes SwiftTrans, a code translation framework combining multi-perspective generation (MpTranslator) and difference-aware selection (DiffSelector) to improve correctness and runtime efficiency. Experiments show improvements over prior methods, and a new benchmark SwiftBench is introduced.

### Strengths
- Novel framework combining multi-perspective generation and difference-aware selection.

- Systematic approach to evaluating efficiency, with a new benchmark (SwiftBench).

- Results demonstrate improvement in both correctness and runtime performance.

### Weaknesses
- Efficiency is measured solely by runtime, which is a narrow metric, while other important factors such as memory usage and overall computational cost are not considered.
- SwiftTrans improves translation quality, but this comes at the expense of substantial computational overhead.
- Even with bubble selection, generating and evaluating multiple candidates remains far more expensive than a single-shot translation.
- The DiffSelector component relies on invoking an LLM as a judge, which is inherently costly in terms of computation.

### Questions
- The benchmark does not specify how many candidate translations are generated (i.e., pass@k).
- A comprehensive comparison of runtime, memory usage, and pass@k should be provided.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents **SwiftTrans**, a code translation framework that enhances both **functional correctness** and **runtime efficiency**. Its core innovation is a two-stage pipeline: a **Multi-Perspective Translator**, fine-tuned with **Hierarchical Guidance** to generate diverse candidates, and a **Difference-Aware Selector** that identifies the optimal translation. Evaluated on extended benchmarks and a new efficiency-focused benchmark, **SwiftBench**, SwiftTrans built on models like Qwen2.5-7B outperforms state-of-the-art baselines, including GPT-5.

### Strengths
1. This paper is well-motivated since the efficiency of code translation is practical and matters a lot in our real life.
2. This paper’s ideas are easy to follow, holistic, and pretty effective.
3. This paper contribute a new benchmark *SWIFTBENCH*, which takes account for the efficieny for code translation.

### Weaknesses
### **Multi-Perspective Exploration**
The idea of leveraging parallel ICL to generate diverse candidates is effective but not novel. I’m curious how well does the model perform with the direct use of ICL only (prompt) compared to your Hierarchical Guidance (training). I think this can be an important ablation study.

### **Difference-aware Selection**
I believe the motivation and ablation study for the Difference-aware Selection is insufficient. Firstly, given the extensive context windows of modern LLMs (e.g., Qwen2.5-3B supports 32k tokens), it’s reasonable to consider evaluating all `m` candidates **simultaneously** within a single, long context (or a batch of candidates). In this case the efficiency will be dramastically improve. Secondly, the selector relies on the `diff(tgt1, tgt2)` operation. This asymmetric presentation may introduce an inherent bias since `tgt1` presents full code but `tgt2`  only shows a partial (the diff with respect to `tgt1`). Therefore, the model might develop a preference for `tgt1` or `tgt2`. It’s recommanded to quantify the consistency of the selector.

Overall, I think the main weakness is the lack of comprehensive ablation study to prove the efficiency of individual parts of the framework (since the overall framework is complex).

### Questions
1. Can SWIFTTRANS be a purely inference-time framework (no training needed)? If yes, how does it perform? 
2. How do you ensure the quality of SWIFTBench? Do you have a more comprehensive comparison of SWIFTBench with CodeNet and F2SBench since I think their sources are similar.
3. What's the efficiency of the whole framework? Do not have a longer inference time compared to F2STrans?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper starts from the observation that hand-crafted translations are faster on average than the translations produced by LLM translators--even if those translations are correct (in terms of i/o-equivalence). It then presents a novel code-to-code translation approach with the goal of producing translations that are not only correct but also efficient.

To this end the authors employ a generator and a selector. The generator produces translations conditioned on the source language code and sets of exemplars of efficient code translations. The selector is prompted with a diff of two possible translations and picks the better one, considering both correctness and efficiency. The selector can be employed on an arbitrarily large set of candidate translations by performing pairwise comparisons in a bubblesort like manner. Both generator and selector are finetuned for this task.
To test this setup the authors extend CodeNet and F2SBench with new tests that increase the input size. In addition, a new dataset is created from Codeforces examples that intentionally introduces inefficient elements into the source language. The generator-selector approach yields significant improvements on all three datasets, to the point where small-scale LLMs with this approach outperform the much larger GPT5.

### Strengths
Exploring secondary characteristics of translation quality, such as efficiency, is an important research direction, and becomes more important as the state-of-the-art for correctly translating code improves. The authors do a good job quantifying this motivation with their preliminary study.

Furthermore, the presented approach yields impressive results (especially in comparison to GPT5) and significantly furthers the state-of-the-art.

Methodologically the paper is primarily a well-executed application of existing ideas; In-context learning, task finetuning, and LLM-as-a-judge--are all well established. However, there is value in how exactly these components are implemented. All parts of the setup appear well-motivated (e.g. to avoid diversity collapse) and the authors thoroughly ablate each component.
The augmented versions of CodeNet and F2SBench, as well as SwiftBench, provide a meaningful point of comparison for evaluation and are certainly of value for future work in this area.

Overall, the presentation quality is high; the paper presents a clear thread from motivation to method to evaluation. All figures are well done and aid understanding.

### Weaknesses
The paper leaves out some important details:

1. F2STrans was presented as a method to improve code style in code-to-code translations. What are the details of how it was applied to efficient translation?
2. How were the inefficiencies in SwiftBench created?
3. What is an "efficiency-critical" testcase? Is this just a larger input size?

The paper's language is in parts a bit too vague and too overdramatic. A more nuanced and grounded register, as well as a focus on the facts, would be more appropriate.

Examples of this include:
-	"[...] runtime efficiency has become as critical as functional correctness in evaluating program quality." (p. 1, ll. 14-16)
-	"efficiency-critical"
-	"code collected on online platforms (e.g., Codeforces)" (p. 4 ll. 169-170) As far as I understand only data from Codeforces is used.
-	"Multi-Perspective" I would argue different sets of demonstrations don't really constitute different perspectives.
Furthermore, I see two minor conceptual problems:
1. SwiftBench tests for a slightly different problem then the rest of the paper. The paper is motivated by the fact that an efficient implementation in the source language could be translated into something that is correct, but not efficient in the target language. SwiftBench evaluates a scenario, where the code in the source language is already inefficient. This is a different, and I believe less relevant, setting.
2. While it appears empirically successful, I fail to see why the number of demonstrations in the MpTranslator should be trained to correlate with target code efficiency.

The related work section misses the following important research directions in code to code translation:
-	Translations on intermediary representations of the code (e.g. [1], [2])
-	Rule based approaches (e.g. [3]) and neuro-symbolic hybrids (e.g. [4])

A final minor concern is the mention of the update frequency of SwiftBench (p.5 ll. 61-63) This is an unverifiable statement about the future and is of no value to the reader.

[1] Szafraniec et al., Code Translation with Compiler Representations

[2] Macedo et al., InterTrans: Leveraging Transitive Intermediate Translations to Enhance LLM-based Code Translation

[3] Galois, C2Rust URL https://www.galois.com/articles/c2rust

[4] Nitin et al., C2SaferRust: Transforming C Projects into Safer Rust with NeuroSymbolic Techniques

### Questions
- How was F2STrans applied to this setting?
- What are the details of the "efficiency-critical" test cases?
- How were the inefficiencies in SwiftBench created?
- Do you think this approach remains relevant as the baseline translation models become better?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work tackled the challenge of generating not only functional correct but also efficient code. They propose two modules: MpTranslator, a multi-perspective exploration technique for generating diverse translation candidates, and DiffSelector, a comparison framework to find the best candidate. These modules contribute to SWiftTrans, a code translation framework focusing on correctness and efficiency. For Multi-Perspective exploration, given a source code, an in-context learning phase that generated multiple sets of demonstrations for a given library for parallel translation. To help LLMs learn better in code translation, given multiple demonstrations, authors provide hierarchical data construction to filter the set of correct programs measured by test cases, and these programs have to be consistently 10% speedup from multiple rounds of code generation. The DiffSelect module leverages a difference-aware approach for candidate selection to avoid too similar target codes. The authors optimized the comparison process by applying the Bubble sort algorithm. Next, another round of optimization was performed to rank candidates based on the proposed judge loss function. In this evaluation, this work constructs SwiftBench, a dataset derived from the CodeNet dataset that includes more information about efficiency-critical test cases and a baseline execution time for the target code. By Computational Accuracy and Execution Time, the authors demonstrate that their proposed models perform best with Qwen2.5-7 B and outperform the existing well-known LLM, GPT-5, in code translation.

### Strengths
- The proposed problem is important.
- The design selections for two modules MpTranslator and DiffSelector are sound.
- This work proposes a rigorous process of evaluation with experiments.

### Weaknesses
- In terms of the paper written, there is a concept that was very unclear to me: “demonstration”. Also, there is lack of examples of Library C (mentioned in L152). Currently, I understand the demonstrations as the set of generated code by open LLMs for hierarchical data construction.
- In section 2.2.2, the authors mentioned in Line 227 that DiffSelector needs to rank incorrect translation code. However, in the earlier process of this framework, hierarchical data construction, functional incorrect codes were eliminated (Line 175-176), which is contradictory. 
- Experiments lack configurations of building SwiftTrans over other open LLMs besides Qwen. Although it doesn’t mean the paper is invalid, a brief explanation about the decision to choose Qwen for optimization is needed.

### Questions
- Authors should clarify the definition of demonstrations and the demonstration library in the camera-ready version.
- An explanation of the reasons for choosing Qwen.
- Can this work be extended to ensure the generated code follows other perspectives besides functional correctness (such as coding style, code readability defined in [1]?

1.CodeUltraFeedback: An LLM-as-a-Judge Dataset for Aligning Large Language Models to Coding Preferences. Martin Weyssow, Aton Kamanda, Xin Zhou, Houari Sahraoui

### Soundness
3

### Presentation
2

### Contribution
3
