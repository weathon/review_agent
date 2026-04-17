# GneissWeb: Preparing High Quality Data for LLMs at Scale

- Decision: Accept (Poster)
- Scores: 6, 4, 6

## Abstract
Data quantity and quality play a vital role in determining the performance of Large Language Models (LLMs). High-quality data, in particular, can significantly boost the LLM's ability to generalize on a wide range of downstream tasks. In this paper, we introduce **GneissWeb**, a large dataset of around 10 trillion tokens that caters to the data quality and quantity requirements of training LLMs. Our GneissWeb recipe that produced the dataset consists of sharded exact sub-string deduplication and a judiciously constructed ensemble of quality filters. 
GneissWeb goes beyond simple model-based quality filtering used in recent datasets by designing an ensemble of filters incorporating novel quality filters. Novel components enable us to achieve a favorable trade-off between data quality and quantity, producing models that outperform models trained on state-of-the-art open large datasets (5+ trillion tokens). We show that models trained using GneissWeb outperform those trained on FineWeb-V1.1.0 by 2.73 percentage points in terms of average scores on a set of 11 commonly used benchmarks (both zero-shot and few-shot) for pre-training dataset evaluation. When the evaluation set is extended to 20 benchmarks (both zero-shot and few-shot), models trained using GneissWeb still achieve a 1.75 percentage points gain over those trained on FineWeb-V1.1.0.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This method uses an ensemble of quality filters and exact deduplication to improve the pretraining data recipe. This paper works, demonstrates consistent improvement across scales. The evaluations were diverse and rigorous.

### Strengths
1. Powerful empirical results across the benchmarks. I really like the scaling plots. Tell a very clear story.
2. The mixing of multiple quality filters is a nice paradigm. 
3. The evaluations were particularly rigorous. 
4. Good statistical rigor.

### Weaknesses
None really. Small things:
1. Could have discussed the computational cost of experiments.
2. Starting from FineWeb is good, but maybe could have ablated different sources. 
3. Math/code performance is ok, but not stellar.

### Questions
N/A

### Soundness
3

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
The paper presents GneissWeb, a 10T-token web dataset and an associated recipe for stage-1 LLM pretraining. Starting from FineWeb-V1.1.0, the authors add (1) sharded exact substring deduplication, (2) an ensemble of quality filters that includes two fastText classifiers, a category-aware readability filter (McAlpine-EFLAW), and a category-aware extreme-tokenized-documents filter (based on tokens/char and tokens/byte), and (3) lightweight category classifiers to apply category-specific thresholds. Across 1.4B/3B/7B Llama-style ablation models trained on 350B tokens, GneissWeb yields consistent gains over FineWeb and FineWeb-Edu-Score-2. It also reports 27% lower FLOPs to reach a 64% high-signal score at 7B. Stage-2 continuation from GneissWeb retains an advantage, and a Bloom-filter (~28 GB) is proposed to reproduce an approximation of the selected subset.

### Strengths
- The experimental results show clear improvements at multiple scales. Consistent gains at 1.4B/3B/7B on both high-signal and extended suites.
- This paper introduces a practical and scalable pipeline. Sharded exact substring dedup and ensemble filtering implemented with an open Data Prep Kit. The Bloom-filter reproduction path is pragmatic for large-scale users.
- Reporting FLOP reductions to a fixed quality target can be helpful.

### Weaknesses
- While each component is tested separately, the interactions among filters (and the precise contribution of threshold tuning) are not clearly separated. There is also a risk of overfitting the filtering thresholds to the evaluation sets.
- It seems that the pipeline leans on readability heuristics and tokenization extremes. There is limited analysis of the false-positive and false-negative rates and their downstream semantic impact (e.g., removal of niche yet valuable domains).
- Many of the data filtering methods in this paper are based on existing approaches, but the authors do not clearly explain their specific improvements or the details of their ensemble methods. This makes the paper’s contribution less clear.

### Questions
Please refer to the Weaknesses

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The work introduces a 10T token LLM pretraining dataset called GneissWeb. In addition to traditional filters and deduplication, their data curation recipe includes new quality filters based on readability, tokenization, and category (domain). Using this dataset, the authors train models at 1.5B, 3B, and 7B scales, and show improvement in average performance across common benchmarks when compared to other contemporary open datasets.

### Strengths
The paper claims novelty in terms of several new filtering metrics being applied. These metrics are supported in the context of their training setup with ablations comparing performance against reasonable baselines. The authors are transparent about their methods and clarify all important details about their setup for data curation and training. The paper also touches on fairness/bias, training efficiency, and downstream LLM performance.

### Weaknesses
The ablations and comparisons against FineWeb appear thorough and convincing as a case for how the new specific filtering approach improves on the FineWeb recipe. However, the paper leaves unclear where this stands overall as a contribution to pretraining dataset construction.

One significant claim from the paper is that other competing filtering methods (like DC-LM Baseline) are more aggressive in quality filtering, and thus GneissWeb has the advantage of creating a larger pretraining dataset. However, this claim does not appear to be backed up by indicators of how GneissWeb would compare to these other methods given an equivalent starting dataset.

The paper also introduces new rule-based filters that show improvements in data quality. The effect of these filters seems underexplored, e.g., how do their tunable thresholds scale? How sensitive is performance to the thresholds of these parameters?

### Questions
1. How does GneissWeb compare to other recent open models trained at a similar scale? The focus of the baselines is FineWeb; are there other similar open datasets that should be compared against?
2. The paper seems to focus on cutting as little as possible during data curation, pointing out more aggressive filtering methods as a negative. However, the training experiments are all on a small subset of the dataset. Is there evidence that the performance would continue to scale at the same rate to the size of the full dataset?
3. On a related note, do you find any evidence of performance gains from making the filters more restrictive (i.e. changing the thresholds to filter more data)? Do the proposed filters continue to work well with stronger thresholds?

### Soundness
4

### Presentation
3

### Contribution
3
