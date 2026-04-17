# MATH-Beyond: A Benchmark for RL to Expand Beyond the Base Model

- Decision: Accept (Poster)
- Scores: 2, 4, 4, 6

## Abstract
With the advent of DeepSeek-R1, a new wave of reinforcement learning (RL) methods has emerged that seem to unlock stronger mathematical reasoning. However, a closer look at the open-source ecosystem reveals a critical limitation: with sufficiently many draws (e.g., $\texttt{pass@1024}$), existing base models already solve nearly all questions on widely used math benchmarks such as MATH-500 and AIME 2024. This suggests that the RL fine-tuning methods prevalent in the LLM reasoning literature largely sharpen existing solution modes rather than discovering entirely new ones. Such sharpening stands in contrast to the broader promise of RL: to foster exploration and to acquire new skills. To move beyond this plateau, we introduce MATH-Beyond (MATH-B), a benchmark deliberately constructed to defeat common open-source models of up to 8B parameters even under large sampling budgets. Improving performance on our benchmark via RL requires methods that learn to reason in ways that go beyond base model capabilities in repeated sampling. Since the problems are drawn from subsets of DAPO-Math-17K and DeepScaleR datasets, they remain topically equivalent to standard high-school math. Validating our premise, RL fine-tuned models such as Nemotron-Research-Reasoning-Qwen-1.5B and DeepScaleR-1.5B-Preview perform poorly on MATH-B at $\texttt{pass@1024}$, showing how existing approaches fall short on tackling harder instances. We hope MATH-B will catalyze exploration-driven RL approaches that elicit deeper reasoning capabilities.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces MATH-B, a benchmark designed to evaluate whether reasoning models fine-tuned with RL can genuinely “expand” beyond the capabilities of their base models, aiming to push progress past the current plateau. The benchmark is constructed by filtering existing datasets, DAPO-Math-17K and DeepScaleR, to retain problems unsolved by several open-source models even under a large sampling budget (pass@1024). The authors define an “Expansion Rate” metric, effectively pass@k on a zero-baseline set, and evaluate several RL-trained and SFT models. They find that most RL methods yield only modest (<10%) improvements, whereas long-chain-of-thought (CoT) distillation achieves substantially higher (>50%) gains. The authors argue that MATH-B can serve as a diagnostic tool to guide future exploration-driven RL research.

### Strengths
1. The limits of RL-fine-tuned reasoning models is an important topic, especially given saturation on common benchmarks.

2. The paper is reasonably well written and clearly explains its data-filtering pipeline and evaluation setup. The figures and tables (e.g., dataset construction flow, pass@k comparisons) are well organized.

### Weaknesses
1. The dataset is almost entirely derived from existing public datasets (DAPO-Math-17K, DeepScaleR), using common filtering, for example, deduplication, sampling, verification, etc. The resulting benchmark is a subset of known problems already curated and verified in prior works. The paper does not introduce new problem formulations, annotation schemes, or evaluation methodologies that go beyond what prior datasets or analyses already offer, especially given the availability of broader, harder, and more rigorously verified benchmarks such as MathArena [3], UQ [4] and more.

2. The motivation and methodology closely mirror those of DAPO [1] and DeepScaleR [2], both of which already include difficult math subsets, verification studies, and RL vs. SFT comparisons. These prior works went further by introducing RL frameworks, conducting ablations, and analyzing exploration effects

3. The evaluation simply confirms what is already known, that current RL methods (e.g., GRPO, R1-style) yield limited improvement over base models, especially on problems where base model can barely solve.

4. The authors claim MATH-B encourages “exploration-driven RL,” and can "move beyond the current plateau", but provide no evidences show that training with the dataset help model learn new reasoning strategies. Thus, its value as a benchmark remains speculative.

[1] Yu, Qiying, et al. "Dapo: An open-source llm reinforcement learning system at scale." arXiv preprint arXiv:2503.14476 (2025).

[2] Luo, Michael, et al. DeepScaleR: Surpassing O1-Preview with a 1.5B Model by Scaling RL. 2025, pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2.

[3] Balunović, Mislav, et al. "Matharena: Evaluating llms on uncontaminated math competitions." arXiv preprint arXiv:2505.23281 (2025).

[4] Nie F, Liu K Z, Wang Z, et al. UQ: Assessing Language Models on Unsolved Questions[J]. arXiv preprint arXiv:2508.17580, 2025.

### Questions
1. In Figure 3, the authors show that the problems models find challenging are not necessarily those that humans typically struggle with. Why does this mismatch occur, and what specific failure modes of LLMs lead to zero-success rates across multiple models?

2. For problems outside MATH-B that models solve with very low probability, is success mainly due to genuine reasoning and correct problem-solving, or does it mostly arise from randomness, where the reasoning is incorrect but the final answer happens to match?

3. Given the reliance on GPT-5 for verification, how robust are the correctness checks? Did the model ever mis-verify solutions?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
1.Improving performance is explored on the proposed benchmark via RL requires methods that learn to reason in ways that go beyond base model capabilities in repeated sampling. Since the problems are drawn from subsets of DAPO-Math-17K and DeepScaleR datasets, they remain topically equivalent to standard high-school math.

### Strengths
1.The dataset is hard for 1.5b llm to follow, RL fine-tuned models such as Nemotron-Research-Reasoning-Qwen-1.5B and DeepScaleR-1.5B-Preview perform poorly on MATH-B at pass@1024.

### Weaknesses
1.The proposed dataset does not show the goodness on the performance rising after using the dataset for RL training.
2.The difficulity and other flag is not given for the dataset such as AIME24 and AIME25.
3.The quality of the proposed dataset from the LLM-based algorithm is not double-check on human experts.
4.These is no proof to show a variety of LLMs improve their reasoning ability on the application of the proposed dataset.
5.There is no comparison with the proposed dataset and other famous reasoning math dataset such as MATH-0500, AIME24, AIME25.

### Questions
1.Why does the proposed dataset fail to demonstrate performance gains when used for reinforcement learning (RL) training?
2.Why are difficulty levels and other metadata flags not provided for datasets such as AIME24 and AIME25?
3.Why is the quality of the proposed dataset, generated via an LLM-based algorithm, not validated through human expert review?
4.Why is there no evidence showing that a diverse set of LLMs improve their reasoning abilities when trained on the proposed dataset?
5.Why is there no comparative analysis between the proposed dataset and established reasoning math datasets such as MATH-500, AIME24, and AIME25?

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
3

### Summary
This paper introduces MATH-Beyond (MATH-B), a diagnostic benchmark designed to evaluate whether post-training methods genuinely expand mathematical reasoning capabilities or merely refine existing ones. Observing that base models with large sampling budgets (pass@1024) already solve nearly all problems on widely used benchmarks, the authors construct MATH-B to challenge open-source models up to 8B parameters even under large sampling budgets. MATH-B serves as a "zero-baseline" diagnostic tool to catalyze research into exploration-driven RL approaches that achieve genuine capability expansion.

### Strengths
1. This paper addresses an important problem in post-training research, evaluating whether models genuinely explore new reasoning paths versus merely optimizing existing solution modes. This distinction between capability expansion and capability refinement is crucial for understanding the true progress in mathematical reasoning.
2. This presents a structured approach to dataset construction using automated methods, which could potentially improve efficiency compared to pure manual annotation.
3. The paper provides extensive experimental validation across multiple RL fine-tuned models and large sampling budgets to demonstrate the benchmark's effectiveness. The empirical results convincingly show that existing post-training methods struggle on MATH-B, validating the benchmark's design as a diagnostic tool.
4. The paper is well-written with clear motivation and logical flow.

### Weaknesses
1. This paper uses models with <8B parameters for data filtering may introduce model-specific biases. The filtering model's capability ceiling and training data preferences could propagate into the dataset, potentially affecting the benchmark's reliability and fairness across different model architectures and training paradigms.
2. The paper only evaluates models ≤8B parameters, leaving the benchmark's effectiveness for larger models (>8B, especially >70B) unverified. Without validation on state-of-the-art large models, it remains unclear whether MATH-B maintains discriminative power or exhibits ceiling effects for models that mainstream research focuses on, potentially limiting the benchmark's practical utility.
3. Evaluating the same or similar models that were used for data filtering may introduce circular dependencies and bias.

### Questions
1. How do you ensure that MATH-B problems require genuine reasoning exploration rather than reflecting the specific failure modes of the filtering models? 
2. What are the results on large models (>8B), and does the benchmark maintain discriminative power at scale?

### Soundness
2

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
The paper introduces MATH-Beyond (MATH-B), a set of small mathematical benchmarks designed to exclusively measure the Expansion Rate of post-trained LLMs, which is the percentage of problems that post-trained models can solve but base models do not even at high sampling budget (1024). The central motivation is to address the debate that current RL methods primarily sharpen existing base model skills rather than discover entirely new reasoning capabilities (Expansion).

### Strengths
The paper is a meaningful contribution to the community debate regarding whether RL-trained models truly discover new reasoning skills. These benchmark can provide good signal on this, as well as how to improve current methods.

The paper clearly defines the metric and is very detailed on their data filtering pipeline which makes readers more confident about the correctness of their benchmarks.

### Weaknesses
The resulting datasets are small enough that the authors could possibly annotate themselves, and perhaps provide further qualitative insights on these questions. In particular, why are these problems hard for the models but their GPT5-rated difficulties are lower (which the author calls  human-perceived difficulty).

I believe there are questions in the pre-filtered dataset that are rated as more difficulty (see the Omni-MATH paper, which dataset is also used here), yet somehow they end up solvable by some models, which confuses me. Could it be evidence of data leakage?

The number of benchmarked models is quite small, only 4 pairs, and all models are smaller than 8B. How do the benchmarks hold up for larger models? This would enhance the benchmark's longevity.

The authors should discuss the RL post-training data used for the post-trained models in section 4. Is there train-test leak? For example, in the pair r1-1.5b + DeepScaleR, does the training data contain some same problems as in your proposed test datasets? Similarly, the authors should discuss the SFT training data.

I'm willing to raise my score once the weaknesses are addressed.

### Questions
Please see my weaknesses section.

Could the author comment on the expansion rate of the 2 deepseek-r1-distill models, since they are finetuned from a teacher model as well? This is related to your paragraph lines 406 - 411. Does it make sense to include them in the base model list?

I don't understand the division into base and supplementary models and what it is supposed to show.

### Soundness
3

### Presentation
3

### Contribution
3
