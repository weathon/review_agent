# Penalizing Length: Uncovering Systematic Bias in Quality Estimation Metrics

- Decision: Reject
- Scores: 2, 8, 4, 8

## Abstract
Quality Estimation (QE) metrics are vital in machine translation for reference-free evaluation and as a reward signal in tasks like reinforcement learning. However, the prevalence and impact of length bias in QE have been underexplored. Through a systematic study of top-performing regression-based and LLM-as-a-Judge QE metrics across 10 diverse language pairs, we reveal two critical length biases: First, QE metrics consistently over-predict errors with increasing translation length, even for high-quality, error-free texts. Second, they exhibit a preference for shorter translations when multiple candidates are available for the same source text. These inherent length biases risk unfairly penalizing longer, correct translations and can lead to sub-optimal decision-making in applications such as QE reranking and QE guided reinforcement learning. To mitigate this, we propose two strategies: (a) applying length normalization during model training, and (b) incorporating reference texts during evaluation. Both approaches were found to effectively reduce the identified length bias.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
In this work, the authors address the issue of length bias in quality estimation (QE) metrics, where such metrics tend to favor shorter translations and penalize correct but longer ones.

To mitigate this problem, the authors propose two approaches:

(i) normalizing gold scores by translation length during metric training, and

(ii) incorporating reference translations when they are available.

To analyze length bias, the authors conduct two complementary experiments. First, they artificially increase translation length by concatenating correct translations across multiple segments and observe how QE metric scores change. Second, they generate translations of varying lengths for the same source sentence—by prompting a large language model to maintain accuracy and fluency similar to a reference translation—and then compare metric scores across different length buckets.

Overall, the results demonstrate that the proposed methods can effectively reduce length bias to some extent.

### Strengths
•	The paper addresses an important issue in QE metrics, as length bias can affect decision-making in downstream applications.

•	Several thoughtful analyses are presented to highlight and motivate the problem of length bias in QE metrics.

•	The use of multiple QE metrics helps demonstrate the generalizability of the issue.

•	The proposed solution of training QE metrics on length-normalized scores is simple and intuitive.

### Weaknesses
•	No meta-evaluation is performed to validate the effectiveness of the proposed solution, making it difficult to assess its reliability or whether it introduces unintended artifacts. The authors could consider reviewing recent work on meta-evaluation, e.g., https://www2.statmt.org/wmt24/pdf/2024.wmt-1.35.pdf and https://www2.statmt.org/wmt24/pdf/2024.wmt-1.2.pdf

•	The analyses assume that translations are error-free, which is not always the case. According to the WMT findings paper (https://www2.statmt.org/wmt24/pdf/2024.wmt-1.1.pdf)), reference translations are not always preferred, so some observed bias could be due to errors in the translations themselves. A small-scale human evaluation could help validate this aspect. For example: collecting MQM or DA scores for each individual segment as well as concatenated segments and check whether the length bias is also present from human perspective.

•	The proposed approach of training metrics with normalized scores lacks sufficient training details. Providing information on the training data and hyperparameters would improve reproducibility.

•	It is unclear why the second proposed solution—using reference translations to mitigate length bias—is considered a contribution. This approach does not address the issue in reference-free QE models, which are the main focus of the paper.

### Questions
N/A

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
the paper explores lenght bias in quality estimation
it shows that longer segments are penalised by quality current estimation metrics, both autoregressive as well as LLM-based

two possibilities for reducing the effect are proposed: normalisation of scores over the segment length, and using reference translations as additional input

### Strengths
the paper describes an interesting observation regarding evaluating long segments without references

the analysis is sound and the findings are useful

the paper is clearly written

### Weaknesses
no major weaknesses

a few small details could be improved (see Questions)

### Questions
Related work should be placed after introduction, not at the end

what are tokens exactly? sub-word units? 



3.2  the described MQM score is not normalised over the text length (one major error brings -5 points, no matter whether it is one error in a segment of 10 words or one error in a segment of 100 words) -- could it have some influence to the observed length bias? 

it seems that this is precisely the idea for mitigation described in 4.2 => it could be discussed clearly already in 3.2

### Soundness
4

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
Interesting research for MT QE.
-  The study addresses an underexplored yet critical issue—length bias in QE metrics, which are vital for reference-free MT evaluation and RL tasks. By focusing on a practical vulnerability ignored by prior research, it offers meaningful insights into metric reliability, making the research highly engaging and relevant.
-  It systematically uncovers the "length penalty" bias: QE metrics spuriously overpredict errors for longer but error-free translations, verified via controlled experiments of concatenating high-quality segments, ensuring rigorous and convincing findings.
-  The research identifies another distinct bias—QE metrics disproportionately favor shorter candidates when multiple translations of the same source exist. Complemented by effective mitigation strategies (length normalization, reference supplementation), it provides actionable solutions for more equitable evaluations.

### Strengths
Good research problems.
- The research targets an underexplored yet practically consequential issue—length bias in QE metrics—filling a critical gap in existing literature and addressing a vulnerability that misguides downstream MT applications like reranking and reinforcement learning.
- It adopts a systematic and rigorous experimental design, with two complementary experiments (concatenating error-free segments, comparing multi-candidate translations) across 10 language pairs, ensuring reliable and generalizable findings.
- The study not only uncovers length biases but also proposes two actionable mitigation strategies (length normalization in training, reference supplementation in evaluation) with validated effectiveness, providing direct value for improving QE metric robustness.

### Weaknesses
Not deep analysis in this research.
- It lacks error analysis of high-quality translations; the study uses "error-free" segments but fails to elaborate on error types/distributions that might interact with length, limiting insights into how bias interacts with actual translation quality.
- There is no analysis of the impact of evaluation model size—neither regression-based models nor LLM-as-a-Judge approaches are examined for whether model scale correlates with the severity of length bias.
- The deep-rooted causes of length penalty remain unexplored; the study identifies the bias but does not investigate why QE metrics conflate length with quality (e.g., training data biases, model architectural limitations).
- It neglects differential analysis across QE metric types: the research groups regression-based and LLM-as-a-Judge models together, offering no insights into whether one category exhibits stronger length bias or responds better to mitigation strategies.

### Questions
- Differerent translation errors analysis will be helpful.
- Different model and model size analysis will be helpful.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The paper investigates length bias in machine translation Quality Estimation (QE) models, which assess translation quality without references. The authors find that both traditional regression-based models like MetricX and COMET and large language model judges such as AutoMQM tend to give lower scores to longer translations, even when they are error-free.

Using data from ten language pairs in the WMT24++ dataset, they show that QE scores consistently decline with text length, and that models often prefer shorter outputs when comparing translations of equal quality. They claim that this effect stems from a lack of long, clean examples in the training data.

To reduce this bias, the authors propose two methods: training models to predict error density instead of raw errors and combining QE outputs with reference-based metrics when available. They find that these methods lessen the bias but do not remove it completely.

Overall, the paper provides a good empirical investigation of length bias in quality estimation. It shows that this bias can affect evaluation results and downstream processes such as reranking and RLHF. The proposed methods offer practical approaches to reduce the problem and improve the robustness of QE models.

### Strengths
- The paper addresses a relevant and underexplored problem: length bias in quality estimation metrics and it proposes a clear and effective solution to mitigate this issue.

- The experiments cover a wide range of QE architectures, including both regression-based and LLM-based models. The evaluation spans ten diverse language pairs, showing that the bias is systematic and cross-lingual.

- The proposed method is supported by strong empirical evidence, demonstrating its practical effectiveness.

### Weaknesses
- The paper lacks human evaluation results, which would strengthen the findings and provide a clearer link between metric predictions and human judgment. I think this is a minor point as the paper evaluates using multiple automated metrics

- The authors cite related work addressing similar issues, but they do not include direct comparisons or benchmarks against those existing approaches. Some relevant works from outside the MT community - (https://arxiv.org/pdf/2407.01085v5, https://arxiv.org/pdf/2406.17744)

### Questions
check weaknesses

### Soundness
3

### Presentation
3

### Contribution
3
