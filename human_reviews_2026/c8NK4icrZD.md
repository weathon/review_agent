# Code-driven Number Sequence Calculation: Enhancing the Inductive Reasoning Abilities of Large Language Models

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 6, 4, 6, 2, 8

## Abstract
Large language models (LLMs) make remarkable progress in reasoning tasks.
Among different reasoning modes, inductive reasoning, due to its better alignment with human learning, attracts increasing interest.
However, research on inductive reasoning faces certain challenges.
First, existing inductive data mostly focuses on superficial regularities while lacking more complex internal patterns.
Second, current works merely prompt LLMs or finetune on simple prompt–response pairs, but do not provide precise thinking processes nor implement difficulty control. 
Unlike previous work, we address these challenges by introducing \textit{CodeSeq}, a synthetic post-training dataset built from number sequences. 
We package number sequences into algorithmic problems to discover their general terms, defining a general term generation (GTG) task correspondingly. 
Our pipeline generates supervised finetuning data by reflecting on failed test cases and incorporating iterative corrections, thereby teaching LLMs to learn autonomous case generation and self-checking.
Additionally, it leverages reinforcement learning with a novel Case-Synergy Solvability Scaling Reward based on both solvability, estimated from the problem pass rate, and the success rate of self-directed case generation, enabling models to learn more effectively from both successes and failures.
Experimental results show that the models trained with \textit{CodeSeq} improve on various reasoning tasks and can preserve the models’ OOD performance.
Our code and data are available at: https://anonymous.4open.science/r/CodeSeq2-1ABE.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper addresses the deficiency of LLMs in inductive reasoning by introducing CodeSeq, a synthetic post-training dataset built from number sequences. The authors package number sequences into algorithmic problems to discover their general terms, defining a General Term Generation (GTG) task. Their pipeline generates supervised fine-tuning data through case-based reflection and iterative corrections, teaching LLMs autonomous case generation and self-checking. For reinforcement learning, they propose a Case-Synergy Solvability Scaling Reward (CSSR) that combines problem solvability (estimated from pass rate) and self-directed case generation success.

### Strengths
- First work to systematically use number sequences for LLM inductive reasoning training
- Comprehensive experiments on two base models (LLaMA3-8B, Qwen2.5-7B) with multiple random seeds
- Clear motivation for using number sequences and code-based representation
- Demonstrates substantial in-domain improvements

### Weaknesses
- No theoretical or empirical comparison showing why number sequences are fundamentally better for inductive reasoning
- Only 200 test samples for GTG evaluation - may not be statistically robust
- Data contamination risk: Number sequences from OEIS might overlap with pretraining data

### Questions
- What is the theoretical or empirical evidence that number sequences are superior to List Functions or ARC for inductive reasoning?
- How do you ensure number sequences from OEIS haven't been seen during pretraining?
- Why only 200 test samples for GTG? This seems insufficient for robust evaluation.
- The OOD improvements are very small. Are these statistically significant?

### Soundness
3

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
2

### Summary
This paper addresses the challenge of enhancing inductive reasoning capabilities in large language models (LLMs) by introducing CodeSeq, a synthetic post-training dataset built from number sequences. The authors argue that existing inductive reasoning data focuses on superficial regularities and lacks complex internal patterns. To address this, they package number sequences into algorithmic problems aimed at discovering general terms, defining a General Term Generation (GTG) task. The pipeline generates supervised finetuning (SFT) data through case-based reflection on failed test cases with iterative corrections, teaching LLMs autonomous case generation and self-checking. Additionally, they employ reinforcement learning (RL) with a novel Case-Synergy Solvability Scaling Reward (CSSR) that considers both problem solvability (estimated from pass rates) and success rates of self-directed case generation. Experimental results demonstrate that models trained with CodeSeq improve on various reasoning tasks while preserving out-of-domain (OOD) performance.

### Strengths
- Novel approach of using number sequences as a source for inductive reasoning training data, which provides more complex internal patterns compared to existing datasets like List Functions or ARC
- Comprehensive experimental evaluation across 7 benchmarks (1 in-domain, 3 close-domain, 3 out-of-domain)
- Clear problem motivation highlighting two key challenges in inductive reasoning research

### Weaknesses
- The paper equates inductive reasoning primarily with number sequence pattern recognition, which is a narrow interpretation of inductive reasoning
- No discussion of potential data leakage or contamination from web-scraped number sequences
- No analysis of failure cases or error patterns to understand what the model still cannot solve

### Questions
- What percentage of problems required multiple reflection rounds, and what is the distribution?
- Have you tested on longer sequences or sequences with more complex patterns?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a data pipeline called 'CodeSeq' that aims to enhance the reasoning abilities of LLMs by training them to infer general rules from number sequences. Therefore they transform number sequences into algorithmic problems expressed in code, enabling models to generate and test general terms through executable programs rather than abstract formulas. CodeSeq consists of three stages: (1) Sequence Algorithmization: converts curated number sequences into solvable algorithmic problems; (2) Case-based Reflection Injection: creates supervised finetuning data by iteratively refining code through test-based self-correction; and (3) Solvability-Estimated Selection: constructs reinforcement learning data using a new reward function prioritizes challenging problems and successful self-directed case generation. Experiments on open models such as  LLaMA3-8B and Qwen2.5-7B show that this appraoch improves reasoning and code reasoning performance without harming out-of-domain general abilities. Smaller models trained on CodeSeq achieve results comparable to much larger models.

### Strengths
Data pipeline
* The pipeline is conceptually thorough and the two-agent system (working and guiding LLMs) gives a built-in mechanism for problem validation, which improves data reliability.
* The sandbox testing ensures verification of code correctness.
* The CSSR reward is well designed and balances solvability and autonomous case-generation.

Experiments 
* The experiments include ID, close-domain, and OOD tasks and hence give a comprehensive evaluation of reasoning improvements.
* Results show that CodeSeq improves inductive reasoning without sacrificing general reasoning.
* Multiple ablation variants help isolating which components matter, showing the importance of both reflection and solvability-based rewards.
* They further explore scaling effects (reflection rounds, pattern diversity & offer insights into data complexity and diminishing returns.
* The use of multiple seeds, multiple backbones, and standardized benchmarks (HumanEval, MMLU, etc.) adds credibility and reproducibility.

### Weaknesses
Data pipeline
* The section lacks quantitative examples.
* Can you please explain these manually written rules and give examples for them: "We first manually write rules to remove sequences with insufficient information, such as those without the calculation process or evolve from other sequences (requiring additional webpage links for references)."
* The reliance on LLM agents for data verification could introduce systematic errors > was this analysed? Where human evaluators invovled in verifying the success rate of the pipeline? 
* The LLM-based reflection and validation process might be  expensive, pls discuss cost factors and tradeoffs between having a human-in-the-loop data gen approach vs purely synthetic with LLMs.
* There is no analysis of failure cases, e.g. algorithmic problems or code solutions being incorrect while passing validation.
* The paper provides limited evidence on the quality or representativeness of the final dataset.
* How reproducible is the pipeline? Given that it depends on specific prompt-engineering choices and models?

Experiments 
* Eval focuses mostly on open-source models, specialized reasoning models (e.g. for Math or coding) would strengthen claims.
* Since number sequences are scraped from public sources, overlap between training and evaluation sequences might occur.
* While results show models improve, they don't qualitatively show with examples how model's reasoning chains improved before and after.

### Questions
See section weaknesses above

### Soundness
2

### Presentation
2

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
This work focuses on improving inductive reasoning capability in LLMs. The authors first package number sequences into algorithmic problems to discover their general terms, defining a general term generation (GTG) task that can be represented by code solutions. Then, the authors propose a synthetic data pipeline, forming a post-training dataset that consists of SFT and RL data for solving number sequence algorithmic problems.

### Strengths
1. This work enhances inductive reasoning ability in LLMs, which is an unexplored yet interesting direction. The General Term Generation (GTG) task is well-motivated and addresses a critical gap in inductive reasoning benchmarks.

2. Packaging number sequences as algorithmic problems with code solutions is novel, and the number sequence data synthetic pipeline is methodologically sound.

3. The synthetic data CodeSeq is proven effective for various reasoning tasks and achieves good performance on the proposed GTG task.

### Weaknesses
### Main weakness

- One concern is the scope between sequence data and inductive reasoning definition: Framing inductive reasoning as general term generation in number sequences is reasonable as a starting point but risks oversimplification. It is debatable to what extent sequence prediction captures "true" inductive reasoning as done in broader sentries (*e.g.*, relational, visual, or linguistic domains).

- Limited evaluation benchmarks. The improvements in inductive reasoning ability are not benchmarked on existing public datasets but only evaluated on a single self-crafted GTG subset. I strongly suggest the authors provide results on CodeARC benchmark [1] (Wei et al., 2025) to further validate the effectiveness of the proposal method.

- Unclear costs. The data construction pipeline involves iterative generation and reflection using APIs of superior LLMs. I wonder if it's costly to generate such a dataset. I suggest the authors can clearly report relevant costs and time efficiency.


### Minor weakness
- Several relevant recent works and discussions [1][2] on inductive reasoning in LLMs are missing from the related work.

- Although the authors claim "no security concerns" in the ethics statement, it would be helpful to discuss about known risks of LLMs in the context of inductive reasoning, such as bias, deception, or producing wrong but convincing answers.

---

[1] CodeARC: Benchmarking Reasoning Capabilities of LLM Agents for Inductive Program Synthesis. arXiv preprint:2503.23145

[2] Patterns Over Principles: The Fragility of Inductive Reasoning in LLMs under Noisy Observations. arXiv preprint:2502.16169

### Questions
1. What guarantees does the use of code unit tests offer against "shortcut" code solutions that pass curated test cases but fail in other general cases? Has any formal error analysis been performed?

2. In reward function CSSR, the authors combine the metric of $ \frac{N_{tc}}{N_c} $. This seems to require the model to have the ability to generate test data in CoT itself. But after checking the $CodeSeq_{RL}$ training data, I don't see prompts that guide the model to do this. Could the author explain this formula?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper presents a framework and dataset to finetune models on inductive reasoning tasks, where the training data is based on number sequences. Number sequences are transformed into algorithmic problems, and the models are trained to produce code that provides a general term for the number sequence. A dataset for SFT and RL-based finetuning is constructed in several steps, letting a larger LM reason about the problem and make several attempts. Experiments show that finetuning with this dataset can improve the performance of small models significantly, sometimes even outperforming closed-source models.

### Strengths
1. The paper presents an interesting approach to convert simple number sequences into a large synthetic dataset with a variety of algorithmic problems.
2. There are extensive ablation studies that justify the design choices and chosen parameters.
3. The experiments show large performance improvements on related benchmarks.

### Weaknesses
1. The methodological novelty of the framework is limited, since it mainly consists of distilling knowledge from LLMs into smaller models, by letting the larger models construct algorithmic problems and solve them. Standard finetuning methods are used. 
2. In general, the contribution is unclear. In the abstract, the contribution is framed as a dataset, but usually a dataset is proposed as a benchmark or for a specific real-world application, and not as a toy problem for finetuning. There are some nice contributions, such as defining a solvability score for problems and weighting the RL reward by the solvability score. If the main contribution are these methodological adaptations, it would be required to train on other datasets to show whether they consistently improve performance. If the main contribution is a dataset of toy problems, it would be necessary to show that this dataset is more useful for finetuning (or complementary) to other datasets. 
3. The performance on out-of-domain problems only shows very limited improvements. However, if the models do not achieve good performance on general inductive reasoning tasks, it is questionable whether this method is useful.
4. The need for this dataset is motivated by saying that other related datasets such as ARC “fail to form complex internal patterns”. However, Table 3 shows that ARC is a lot harder for LLMs than the proposed dataset.

### Questions
1. I’m very confused about the notion of “publishing a RL dataset”. To the best of my knowledge, GRPO (and other common RL algorithms) are trained on-policy, so that the data is only generated during training. The reward is computed for the generated answer and then the model is updated. So how is it possible to construct CodeSeqRL dataset and train multiple models with it?
2. Why not compare to a model fine-tuned with another dataset, e.g. the train split of the Humaneval or MBPP dataset? This would show whether the proposed dataset brings an advantage compared to existing datasets.

### Soundness
3

### Presentation
2

### Contribution
1

---

## Human Reviewer 6

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper introduces **CodeSeq**, a novel synthetic *post-training* framework aimed at improving **inductive reasoning** capabilities in large language models (LLMs). Instead of relying on text-only reasoning data, CodeSeq leverages **number sequences** a domain that naturally encodes inductive structure and transforms them into *algorithmic reasoning problems* solvable by code.

The pipeline has three main stages:

1. **Sequence Algorithmization** – Number sequences are automatically converted into *code-driven tasks*, where the model must infer the generating rule and implement it as executable code. This enables **unit-test-based supervision**, providing objective correctness signals rather than fuzzy linguistic ones.

2. **Case-based Reflection Injection (SFT phase)** – Using two collaborating LLM agents (“working” and “guiding”), the system generates iterative *reflections*: failed test cases, diagnostic reasoning, and code revisions. These reflections are incorporated into supervised fine-tuning data, teaching the model to reason through *error detection and correction*.

3. **Solvability-estimated Reinforcement Learning (RL phase)** – The authors introduce a new reward, **CSSR (Case-Synergy Solvability Scaling Reward)**, which combines (a) the *difficulty* of a problem (estimated by its empirical solvability rate) and (b) the *quality* of the model’s generated test cases. This encourages exploration of harder inductive patterns while reinforcing effective self-checking behavior.

Experiments on LLaMA-3-8B and Qwen-2.5-7B show significant improvements on the proposed **General Term Generation (GTG)** benchmark (up to +42.7 points), as well as *transfer gains* on standard code reasoning datasets such as **HumanEval**, **MBPP**, and **LiveCodeBench**. Notably, performance on unrelated tasks (MMLU, BBH, Gaokao-Bench) remains stable, suggesting that the method enhances reasoning without harming broad language competence.

### Strengths
**Originality**

The paper is conceptually original in how it reimagines *inductive reasoning* training for LLMs through **code-driven number sequences** rather than conventional text or mathematical proofs. By operationalizing reasoning as a *code synthesis problem with verifiable test cases*, it bridges the gap between symbolic algorithm induction and neural post-training. The **Case-based Reflection Injection** stage adds further novelty teaching the model to reason through *error diagnosis and correction* in a structured manner. The introduction of the **CSSR (Case-Synergy Solvability Scaling Reward)** is also a creative step that makes reinforcement learning sensitive to task difficulty and model-generated validation quality. Together, these ideas represent a thoughtful blend of synthetic data generation, reflective supervision, and difficulty-aware RL.

**Quality**

The paper demonstrates methodological rigor and careful experimental design. The proposed framework is implemented systematically from automatic task generation to reflection-driven SFT and a clear RL finetuning phase. The experiments span two model families (LLaMA3 and Qwen2.5), multiple scales, and both *in-domain* (GTG) and *transfer* benchmarks (HumanEval, MBPP, LiveCodeBench). Ablations are included for reflection data, CSSR, and solvability thresholds, showing how each component contributes. The results are consistent and reproducible across settings, with transparent reporting of data volumes, model configurations, and cost. Importantly, the method demonstrates transferability rather than overfitting to the synthetic domain, implying strong generalization.

**Clarity**

The paper is clearly written and well-structured, with intuitive diagrams illustrating the CodeSeq pipeline. Each component ie sequence algorithmization, reflection injection, and RL is explained with sufficient algorithmic detail and examples. Mathematical notation for CSSR is compact yet interpretable, and pseudocode clarifies the data generation and training processes. The visual examples of reflection traces effectively communicate the iterative reasoning process the model learns. Even technically complex sections (like solvability estimation and reflection-guided data curation) are presented in an accessible way, supporting broad comprehension.

**Significance**

This work holds high significance for both *practical LLM improvement* and *conceptual progress* in reasoning research. Practically, it provides a scalable synthetic data pipeline that improves reasoning without relying on scarce human annotations. Conceptually, it advances the field’s understanding of how to align LLMs toward **self-verifying, case-based reasoning**, a crucial step beyond rote pattern prediction. The demonstrated transfer to open-ended code tasks shows the potential for CodeSeq to strengthen reasoning skills that generalize across domains. It also offers a reproducible testbed (the GTG benchmark) that can stimulate further research on inductive reasoning evaluation.

### Weaknesses
**1. Limited reasoning scope (narrow inductive domain)**

While number sequences offer a clean testbed for inductive reasoning, they represent a *narrow and low-entropy domain*. The model’s improved performance on these structured tasks may not necessarily translate to the kind of **open-domain or abstract reasoning** that motivates the paper. The inductive structure of integer sequences is fundamentally algorithmic and deterministic, which may inflate the appearance of “reasoning” without requiring conceptual generalization.

*Actionable insight:* Extend the pipeline to other symbolic or relational domains, e.g., **graph induction**, **analogy puzzles**, or **patterned logical expressions,** to test if the reflection mechanism generalizes beyond numeric patterns.


**2. Dependence on LLM-generated data introduces bias and noise**

The dataset is created using “working” and “guiding” LLM agents to generate both tasks and reflections. This **LLM-in-the-loop data generation** can embed the biases, style, and error distributions of the base models, producing self-reinforcing artifacts. There’s limited analysis of the diversity, correctness, or quality assurance of the generated reflections. The paper briefly mentions filtering invalid problems, but offers no systematic verification or human evaluation of data quality.

*Actionable insight:* Include a **human-audited sample analysis** (e.g., 100 random problems) quantifying correctness, novelty, and reflection coherence. Report statistics on how often guiding-agent feedback leads to genuinely correct fixes versus superficial edits.


**3. Evaluation contamination and reproducibility concerns**

The GTG benchmark used to demonstrate performance is internally generated by the authors from similar sources as the training data. The paper provides no detailed description of **data leakage prevention** or overlap detection between the training set and GTG test set. This raises potential issues of data contamination that could exaggerate model gains. Additionally, only aggregate metrics are presented; qualitative error analysis is minimal.

*Actionable insight:* Release scripts for overlap analysis or disclose specific measures (e.g., hash-based deduplication, source isolation). Include a **qualitative failure analysis** showing examples where models produce plausible but wrong rules, revealing what kind of reasoning errors persist.


**4. CSSR reward lacks theoretical justification and sensitivity analysis**

The proposed CSSR reward combines problem solvability and case success rate using a logarithmic scaling. While empirically useful, its **functional form is heuristic**, and its parameters (e.g., solvability cutoff 0.46, smoothing constant ε) are fixed without justification or robustness checks. The ablations show CSSR helps, but there’s no exploration of *why* or how sensitive results are to these values.

*Actionable insight:* Perform a **parameter sweep** on CSSR’s components or report confidence intervals to assess robustness. Provide a short theoretical or empirical rationale explaining why log-scaling solvability yields better exploration.


**5. Lack of deeper interpretive analysis of “reasoning” behavior**

Although the paper improves quantitative benchmarks, it doesn’t analyze *how* the model’s reasoning changes after CodeSeq training. There’s no introspection into whether the model actually learns iterative hypothesis testing or simply overfits to reflection patterns seen during SFT. The presented examples of reflection are anecdotal and do not evaluate correctness or consistency across reasoning steps.

*Actionable insight:* Include **behavioral or process-level evaluation**, such as reasoning trace consistency metrics or correlation between intermediate test-case success and final correctness. Even a small manual annotation of 50 reflections would reveal if the model internalizes a genuine error-checking habit or merely imitates the reflection style.


**6. Limited transparency and dataset accessibility**

Despite being a synthetic framework, the paper does not release datasets or reflection traces for reproducibility. The appendix mentions dataset statistics but omits exact sources, sample data, or implementation details for the reflection generator.

*Actionable insight:* Provide partial dataset releases or at least **10–20 illustrative full examples** of SFT and RL training data in the appendix. Releasing the GTG benchmark publicly would also enhance credibility and encourage further work on code-based induction.

### Questions
1. **On dataset overlap and contamination control**

   * Could you clarify how you ensured *no overlap* between the generated training data (from sequence algorithmization) and the **GTG test set**?
   * Were any hash-based deduplication, source separation, or semantic similarity filtering steps used to prevent data leakage?
   * A clear statement of contamination control would significantly strengthen the credibility of your evaluation, since both the training and test data derive from number-sequence sources.

2. **On the representativeness and quality of reflections**

   * How often do the *guiding-agent reflections* actually lead to *corrected solutions* versus superficial edits?
   * Did you perform any manual or automated quality check of reflection traces (e.g., proportion of cases where the fix was logically valid)?
   * Could you share statistics or examples showing the *distribution of reflection depth*—how many reasoning iterations typically occur before convergence?

3. **On CSSR reward design and robustness**

   * The CSSR reward function is defined heuristically as a log-scaled function of solvability combined with case-success rate. Could you provide an intuition or empirical justification for this formulation?
   * Why was the *solvability threshold* set to 0.46, and how sensitive are results to this value?
   * Have you tried linear or exponential alternatives to the log scaling, and what were the effects?
   * A short sensitivity or ablation study would help confirm that CSSR’s performance isn’t overly dependent on hand-tuned hyperparameters.

4. **On reasoning behavior and interpretability**

   * Do the models trained under CodeSeq show measurable changes in reasoning style? For instance, are their *reflection traces* or *self-debugging steps* more coherent or consistent compared to baseline models?
   * It would be helpful if you could include an *error typology* or a qualitative analysis of reflection chains, identifying whether models truly engage in hypothesis testing or merely imitate reflection syntax.
   * Could you provide examples of failure cases where the model’s reasoning remains brittle and discuss what these reveal about current limitations?

5. **On generalization beyond number sequences**

   * While number sequences are a good proxy for inductive structure, do you believe the CodeSeq pipeline generalizes to *non-numeric domains* such as symbolic reasoning, causal inference, or analogical tasks?
   * What modifications would be necessary for CodeSeq to operate on **graph patterns, textual analogies, or logic puzzles**?
   * If possible, please discuss one concrete obstacle or preliminary result showing whether the framework extends beyond algorithmic sequence induction.

6. **On data transparency and reproducibility**

   * Could you release at least a subset of the generated dataset—say, 100–200 SFT examples and reflection traces—to allow community validation and replication?
   * Will the **GTG benchmark** be made public, and if so, how do you plan to handle potential overlap with future training data in open checkpoints?
   * Publicly releasing even partial resources would substantially improve the paper’s impact and trustworthiness.

7. **On evaluation metrics and cross-domain effects**

   * Have you examined whether the CodeSeq-trained models demonstrate improved reasoning *process fidelity* (e.g., fewer self-contradictions, more accurate intermediate reasoning steps) rather than only higher end-task accuracy?
   * Additionally, could you expand on how you ensured that post-training on CodeSeq did not degrade performance on general-language reasoning benchmarks (e.g., MMLU, BBH)? Was any drop in specific reasoning categories observed?


**Overall suggestion**

Clarifications on dataset independence, reflection validity, and CSSR robustness would directly address the main limitations of the current version. Furthermore, presenting qualitative examples of reasoning progression and exploring possible extensions to other inductive domains would meaningfully strengthen the paper’s claim of advancing *true reasoning* in LLMs.

### Soundness
3

### Presentation
4

### Contribution
4
