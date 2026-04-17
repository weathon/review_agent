# From Amateur to Master: Infusing Knowledge into LLMs via Automated Curriculum Learning

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 2

## Abstract
Large Language Models (LLMs) excel at general tasks but underperform in specialized domains like economics and psychology, which require deep, principled understanding. To address this, we introduce ACER (Automated Curriculum-Enhanced Regimen) that transforms generalist models into domain experts without sacrificing their broad capabilities. ACER first synthesizes a comprehensive, textbook-style curriculum by generating a table of contents for a subject and then creating question-answer (QA) pairs guided by Bloom’s taxonomy. This ensures systematic topic coverage and progressively increasing difficulty. The resulting synthetic corpus is used for continual pretraining with an interleaved curriculum schedule, aligning learning across both content and cognitive dimensions.

Experiments with Llama 3.2 (1B and 3B) show significant gains in specialized MMLU subsets. In challenging domains like microeconomics, where baselines struggle, ACER boosts accuracy by 5 percentage points. Across all target domains, we observe a consistent macro-average improvement of 3 percentage points. Notably, ACER not only prevents catastrophic forgetting but also facilitates positive cross-domain knowledge transfer, improving performance on non-target domains by 0.7 points. Beyond MMLU, ACER enhances performance on knowledge-intensive benchmarks like ARC and GPQA by over 2 absolute points, while maintaining stable performance on general reasoning tasks. Our results demonstrate that ACER offers a scalable and effective recipe for closing critical domain gaps in LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes ACER (Automated Curriculum-Enhanced Regimen), a framework for infusing domain-specific knowledge into large language models (LLMs) through automated generation of textbook-style curricula and Bloom’s taxonomy–guided question–answer (QA) pairs. The synthetic data is used for continual pretraining under various curriculum scheduling strategies, including cognitive progression (from textbooks to easy/hard QA) and persona-based content ordering (high school → researcher). Experiments on Llama 3.2 (1B and 3B) show consistent gains (~3 percentage points macro-average) on five MMLU subdomains where the base models underperform relative to larger teachers (e.g., microeconomics, econometrics). The authors also report modest improvements on non-target MMLU domains (+0.7 points), as well as gains on knowledge-intensive benchmarks like ARC (Clark et al., 2018) and GPQA (Rein et al., 2023), without degradation on general reasoning tasks (e.g., GSM8K, HellaSwag).

### Strengths
1. Well-motivated problem: The paper addresses a genuine limitation of current LLMs—their shallow understanding of specialized domains—highlighted by consistent performance gaps on benchmarks like MMLU (Hendrycks et al., 2021).
2. Systematic synthesis pipeline: ACER’s multi-stage generation process (domain detailing → outline → textbook → QA pairs) is pedagogically grounded and scalable, drawing on established educational frameworks like Bloom’s taxonomy (Bloom et al., 1956).
3. Comprehensive evaluation: The authors evaluate across multiple benchmarks (MMLU, ARC, GPQA, AGIEval, GSM8K, HellaSwag) and include ablations over curriculum scheduling strategies.
4. Reproducibility: Training details, data mixing ratios, decontamination procedures, and prompt templates are thoroughly documented in the appendix.

### Weaknesses
1. Lack of comparison to strong synthetic data baselines: The paper does not compare ACER against recent, high-impact synthetic data methods, like Phi-4 (Abdin et al., 2024), which uses multi-agent, multi-stage pipelines to generate diverse, high-quality textbook-like data. Without such comparisons, it is unclear whether gains stem from curriculum structure or simply from high-quality synthetic data.
2. Marginal and unstable gains from curriculum scheduling: While the “Flat” baseline (random mixing of books and QA) already yields +2.5 points, the best curriculum (Cog+Con) adds only ~0.5 additional points. More concerning, the Interleaved schedule—inspired by Lee et al. (2024)—performs worse than Flat, particularly in mathematics and statistics. This undermines the central claim that structured sequencing is beneficial, and suggests the gains may be schedule-sensitive rather than robust.
3. Heavy reliance on a powerful external model for data generation: The synthetic corpus is generated using Gemini 2.0 Flash, a proprietary, state-of-the-art model. The paper provides no ablation using weaker or open-source generators (e.g., Llama 3 itself). This raises concerns about generalizability: ACER may essentially be a form of knowledge distillation from a stronger teacher, not a self-contained curriculum learning method.
4. Limited model scale and cherry-picked domains: Evaluation is restricted to 1B/3B models. Larger models (e.g., 7B/8B) may already encode sufficient domain knowledge, rendering ACER’s marginal gains irrelevant at scale. Moreover, the five target domains are selected based on the largest student–teacher gaps in MMLU—a reasonable heuristic, but one that risks selection bias; the method’s efficacy on domains with smaller gaps remains untested.
5. Overstated claims about catastrophic forgetting: While non-target MMLU performance improves slightly (+0.7), the 1B model shows degradation on GSM8K (Strict EM: 0.0667 → 0.0591). Although small, this suggests potential capacity interference. The paper lacks more rigorous forgetting metrics (e.g., loss on original pretraining data).

### Questions
see weakness

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The authors introduce ACER, a framework that synthesizes textbook-style corpora alongside complementary exam-style question-answer pairs. They also design curriculum learning methods that align both cognitive and content dimensions.

### Strengths
The paper identifies that state-of-the-art LLMs struggle in specialized domains requiring deep, principled understanding.

### Weaknesses
- The paper only compares the effectiveness of different curriculum schedules within their own method in Table 1, and lacks comparisons against other baseline approaches in the same domain (e.g., those referenced in lines 53–55).

- The ACER synthesis seems to rely on Gemini 2.0 Flash as the LLM teacher, but this is not explicitly stated in Section 3. Moreover, the paper omits presenting Gemini 2.0 Flash’s performance on the evaluation benchmarks, which is important as it is the teacher model.

- The ACER simulations lack input from external domain knowledge, as the authors’ emphasis on the need for “principled domain expertise.”

- The paper does not include results from larger open-source or proprietary models, even as reference results.

### Questions
- Please add results of strong data-curriculum baselines (e.g., self-instruct,and works cited in lines 53–55).

- State explicitly which teacher LLM is used for ACER synthesis (it seems to be Gemini 2.0 Flash).

- Report the teacher's and other LLMs' scores on all benchmarks

- Describe quality controls for factual/conceptual soundness of the synthesized corpus.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper describes a new method for improving domain-expertise in LLMs. Unlike prior work where synthetic instruction data or domain-specific pretraining lacked structured progression, the authors propose ACER (Automated Curriculum-Enhanced Regimen), which automatically generates textbook-style content plus question-answer pairs following Bloom’s taxonomy, and uses a curriculum-aware schedule (cognitive difficulty + persona audience progression) to continually pretrain a general LLM. The proposed method is evaluated on subsets of the MMLU benchmark (five niche domains), as well as ARC, GPQA and other tasks. Experiment results show 3 percentage point macro-average improvement in target domains while preserving general capabilities and achieving 0.7 point gains in non-target domains.

### Strengths
1. The paper presents a systematic synthetic corpus generation pipeline that is well-motivated and clearly described.
2. The curriculum scheduling is a reasonable design choice, enabling ablations that highlight the value of ordering in continual pretraining.
3. Empirical results demonstrate improvements in both target niche domains and stability on general capability benchmarks.

### Weaknesses
1. The impact of the synthesis book corpus generation pipeline is not sufficiently discussed. The experiments centered around using the same pipeline under different curriculum schedules. It's unclear how big a role the generation pipleline plays in the overall performance improvement.
2. Limited insight was revealed and discussed among the different curriculum schedules, e.g. what makes some schedule works better than the others.
3. The performance improvement in some domains are rather limited and may well fell within the range of variance, e.g. Econ, psych, and Macro_nt.

### Questions
1. What's the impact of the pipeline design for the synthesis book corpus generation?
2. Any insight on how sensitive are the results to the quality of the synthetic textbook content, e.g. if generated from a weaker model?

### Soundness
2

### Presentation
3

### Contribution
1
