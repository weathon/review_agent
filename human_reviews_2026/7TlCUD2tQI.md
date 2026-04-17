# Augmenting Industrial Maintenance with LLMs: A Benchmark, Analysis, and Generalization Study

- Decision: Reject
- Scores: 2, 4, 6, 4

## Abstract
Monitoring the life cycle of complex industrial systems often relies on expertly curated temporal conditions derived from sensor data, a process that requires significant time investment and deep domain expertise. We explore the potential of utilizing Large Language Models (LLMs) to generate context-aware and accurate recommendations for maintenance based on their ability to reason and generalize on temporal sensor conditions. To this end, we formulate a novel pipeline that systematically converts human-authored symbolic conditions into a multiple-choice question answer (MCQA) dataset. We apply our pipeline by creating DiagnosticIQ, a 6,000+ MCQA dataset covering 16 different types of physical assets that represent real-world maintenance use cases. We assess 15 state-of-the-art large language models (LLMs) with this dataset and create a leaderboard for the maintenance action recommendation task. Furthermore, we evaluate and demonstrate the practical utility of DiagnosticIQ in two key aspects. First, as a knowledge base to enhance maintenance action recommendations, and secondly, as a fine-tuning resource to fine-tune a specialized LLM that generalizes across previously unseen assets to facilitate the rule creation process.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
The paper introduces DiagnosticIQ, a benchmark for evaluating whether LLMs can recommend maintenance actions from symbolic, time-persistent sensor conditions used in industrial monitoring. The dataset contains 6,690 MCQs drawn from 120 rules across about 16 asset types, with several variants: DiagnosticIQPro, DiagnosticIQPert, DiagnosticIQRationale, and DiagnosticIQVerbose. The authors evaluate 15 LLMs in a zero-shot setting and report a leaderboard. Macro accuracy is highest for Claude-3-7-Sonnet, and most models drop sharply on the Pro split with larger answer choices. They also present a small human study assessing model rationales, a cross-asset fine-tuning study with SFT and GRPO, and detailed analyses by asset and by question type.

### Strengths
- The paper targets a real gap: connecting anomaly rules to actionable maintenance guidance at scale.
- The rules-to-MCQ formulation is well motivated by real maintenance workflows.
- The benchmark construction pipeline is transparent and reproducible: condition trees are converted to disjunctive normal form, and question types are systematically constructed.
- It covers diverse evaluation axes, including robustness to prompt perturbations, per-asset performance, question-type differences, and cross-asset transfer with SFT and GRPO.

### Weaknesses
- Many models on the leaderboard seem outdated and inconsistent. There are more recent closed-source models for Gemini and OpenAI's reasoning models than the ones listed on the benchmark. I suggest updating the leaderboard with more recent model versions. Also, Qwen2.5 is tested for zero-shot but Qwen3-8B is tested in the generalization study, which look like inconsistent choices of models.
- The generalization section uses three 8B models and shows inconsistent gains for GRPO versus SFT across splits. If the authors plan to keep these results, at least some discussion on why this happens would help readers decide what to use in the future.

### Questions
- It would be great if the authors could add the meaning of * in the caption of Table 1 so that readers do not need to look for its meaning.
- The embedding model (all-mpnet-base-v2) is used for creating incorrect options. Is it plausible to use that embedding model? Why not use LLMs (which would have more industrial knowledge) for constructing the negative options?
- The claim "For many enterprise customers, smaller language models will be key, as they provide a practical way to embed domain-specific knowledge directly into the model" seems partially correct as a poor model with smaller parameters would not be preferable to a large quantized model with better capability. Could you elaborate more on this? Similarly, transfer learning is indeed important, but I do not see much of an advantage in fine-tuning over using larger, general models without fine-tuning if anomaly detection is a very important use case for LLMs.
- "Transfer learning between different shows" seems like a typo.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents DiagnosticIQ, a benchmark for evaluating large language models in industrial maintenance. It converts symbolic diagnostic rules into multiple-choice QA tasks to test reasoning, cross-equipment transfer, and maintenance recommendation. The work highlights the gap between current LLM capabilities and real-world industrial reasoning needs.

### Strengths
- The paper addresses an emerging but underexplored area: benchmarking large language models for industrial maintenance tasks. This direction is highly relevant to practical applications in Industry 4.0 and intelligent manufacturing.

- Writing is good and easy to follow and understand.

- The authors propose a well-structured pipeline that converts symbolic diagnostic rules into multiple-choice QA tasks (MCQA). This symbolic-to-language transformation is technically neat and represents a creative way to evaluate LLMs on reasoning grounded in real industrial knowledge.

### Weaknesses
- The manuscript does not cite recent related benchmarks such as CAMB (“A Comprehensive Industrial LLM Benchmark on Civil Aviation Maintenance”) and Wind‑Turbine Maintenance Logs Benchmark (“A Comparative Benchmark of Large Language Models for Labelling Wind Turbine Maintenance Logs”). A clearer comparison with these works, including differences in task types, domain scope, modality coverage, and benchmark construction process, is needed to better highlight the novelty of the current benchmark.

- Although the authors aim for broad industrial coverage, the dataset is restricted to a limited set of device types. The transferability to other equipment categories or industrial domains is not demonstrated.

- While the authors provide code and claim reproducibility, my attempt to run the provided implementation faced issues (e.g., missing configuration files, unclear dependencies). The benchmark currently lacks full reproducibility, which limits its value as a standardized community resource.

### Questions
See in weakness

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper presents a deterministic rule-to-question pipeline that transforms expert-authored industrial maintenance rules into multiple-choice question–answer (MCQA) format.
Applied to 120 real maintenance rules accumulated over seven years, the pipeline yields DiagnosticIQ, a benchmark of 6.7k MCQA instances (plus several variants) spanning 16 industrial asset types.
Fifteen state-of-the-art LLMs are evaluated in zero-shot mode, producing the first leaderboard for “maintenance-action recommendation.”

### Strengths
1. Rules, conditions, and actions originate from seven years of subject-matter-expert (SME) curation, giving the dataset strong industrial validity and relevance. 

2. The rule-to-MCQA conversion algorithm (§3, Alg. 1) is clearly specified, with formal DNF conversion, rule rewriting (RRSim), and interpretable α/β parameters controlling diversification.

3. Visuals and tables are well-organized: e.g., Table 1 highlights the steep accuracy drop from DiagnosticIQ to its harder +Pro variant, while Fig. 3 clarifies asset imbalance motivating macro-accuracy metrics.

### Weaknesses
1. About 58 % of items concern air-handling units (AHUs) (Fig. 3), yet overall accuracy (Table 1) remains the primary metric. While macro-accuracy is reported, several analyses aggregate raw accuracy, potentially overstating performance on dominant asset classes.

2. The macro-accuracy equation (p. 6) omits the denominator $|D_a|$ under the outer summation, causing a dimensional mismatch.
In §3.2.3, the claim that larger α/β “increase question count but reduce diversity” lacks quantitative backing.

3. Several recent benchmarks with strong thematic overlap are omitted: MME-Industry (Yi et al., 2025) – cross-industry multimodal evaluation. PHM-Bench (Yang et al., 2025) – maintenance and health-management tasks.

4. The rules originate from a commercial monitoring system, yet the paper omits discussion of confidentiality, potential misuse, or licensing constraints, which are critical for public release.

### Questions
1. How were action labels deduplicated into the 193-item observation set? Was synonym merging performed manually or algorithmically?

2. What steps are in place to ensure IP compliance and anonymization when releasing SME-derived rules?

3. How do we verify that the data truly reflects realistic maintenance reasoning, rather than just faithfully encoding the rule templates?

### Soundness
3

### Presentation
2

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
This paper introduces DiagnosticIQ, a large-scale benchmark and dataset for evaluating LLMs in industrial maintenance action recommendation. It proposes a rule-to-MCQA pipeline that systematically transforms symbolic, expert-authored maintenance rules into multiple-choice QA datasets, encompassing over 6,600 validated questions across 16 asset types. The authors benchmark 15 LLMs and analyze reasoning, generalization, and robustness, releasing variants such as DiagnosticIQPro (10-option), Pert (perturbed), Verbose (NL conditions), and Rationale (explanation-based). The work also includes fine-tuning (SFT/GRPO) and deployment experiments (MAReE engine), showing that LLMs can partially generalize and reason about sensor-based maintenance tasks.

### Strengths
The paper presents the first standardized benchmark for LLMs in industrial maintenance—a domain rarely addressed in LLM evaluation. The deterministic symbolic-to-MCQA pipeline is well-motivated and rigorously described, ensuring reproducibility and logical consistency. The authors benchmark 15 state-of-the-art LLMs with clear comparisons on reasoning, generalization, and robustness, producing actionable insights (e.g., domain sensitivity across assets, compositional reasoning gap). The integration of the dataset into a real-world recommendation engine (MAReE) is commendable, bridging benchmark analysis with deployable use cases. Dataset variants are thoughtfully designed to probe distinct reasoning dimensions (formatting, rationale, perturbation), increasing diagnostic value beyond simple accuracy.

### Weaknesses
Despite the industrial framing, the dataset is dominated by AHU-related rules (≈58%), with only 10+ asset types, limiting claims of cross-domain generalization. The analysis focuses mostly on macro accuracy, with little discussion on statistical significance or variance across model families and seeds. No comparison against non-LLM baselines (e.g., rule-based or symbolic expert systems) to contextualize the LLM performance gains. The symbolic-to-natural-language conversion step and question templates are discussed but not quantitatively ablated (e.g., contribution of DNF conversion vs. text formatting).  While informative, the leaderboard lacks qualitative error analysis or failure categorization, making it unclear why models fail (e.g., semantic confusion vs. numerical reasoning).

### Questions
How consistent is the rule-to-MCQA generation pipeline across asset types with fundamentally different sensor modalities? 

Could the authors provide examples of incorrect reasoning patterns observed in LLMs (e.g., conflating conditions vs. missing causal links)?

How was expert validation performed—was inter-annotator agreement measured among SMEs?

For the fine-tuning experiments, how is overlap between training and test rules prevented beyond asset-based stratification?

Did the authors consider incorporating numerical reasoning evaluation (e.g., comparing thresholds or temporal trends) explicitly in the benchmark?

### Soundness
3

### Presentation
3

### Contribution
2
