# When Scale is Fixed: Revisiting Pre-training Indicators for LLM Fine-tuning Performance

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 6, 4, 2

## Abstract
While scaling laws tell us that metrics like perplexity effectively indicate how a model performs as it grows, we still don't fully grasp its predictive power at a fixed size. This lack of clarity makes it challenging to conduct effective ablation studies on smaller models, for example, when trying out various pre-training objectives. Since a primary application for these pre-trained models is supervised fine-tuning (SFT) on specific data or tasks, it's crucial for our ablation studies to connect this post-SFT performance back to the initial pre-training choices. This helps us conduct more effective pre-training research.

To study this problem, we first construct a dataset using 50 1B parameter LLM variants with systematically varied pre-training configurations, e.g., objectives or data, and evaluate them on diverse downstream tasks after supervised fine-tuning (SFT). We demonstrate that the conventional perplexity is a highly misleading indicator in this scenario. To address this gap, we formulate the task of selecting pre-training checkpoints to maximize downstream fine-tuning performance as a pairwise classification problem: predicting which of two LLMs, differing in their pre-training, will perform better after SFT. We introduce novel unsupervised and supervised proxy metrics derived from pre-training that successfully reduce the relative performance prediction error rate by over 50% when comparing with existing methods. Despite the inherent complexity of this task, we demonstrate the practical utility of our proposed proxies in specific scenarios, paving the way for more efficient design of pre-training schemes optimized for various downstream tasks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper tackles an underexplored problem: predicting fine-tuning performance of large language models (LLMs) when model scale is fixed. Prior work largely focuses on scaling laws—how performance changes with size or compute—but this paper isolates pre-training effects (data, objectives, filtering) at a constant parameter count (1B). The authors cast the task as pairwise prediction: given two pre-trained models, which will perform better after supervised fine-tuning (SFT)? Using 50 systematically varied 1B models, they evaluate existing pre-training indicators (e.g., perplexity) and propose new unsupervised and supervised proxies. The main contribution is a Learning-to-Compare (LTC) classifier that integrates multiple pre-training metrics and cuts prediction error by >50% relative to perplexity.

### Strengths
1. The paper makes a clear conceptual departure from scaling-law analyses by focusing on within-scale predictive indicators of fine-tuning performance. This framing is both original and practically significant, as it directly addresses the common real-world scenario in which multiple pre-training checkpoints of identical size must be compared efficiently.
2. The study’s formulation of the evaluation problem as pairwise classification over more than 1,000 model pairs is statistically sound and well-justified for the task of model ranking.
3. The experimental results compellingly demonstrate that conventional perplexity is a poor predictor of downstream fine-tuning performance within a fixed model scale (accuracy ≤ 0.33).

### Weaknesses
1.  Details of exp setup are missing. What LLM are used in this work? What are their parameter settings? If I understand correctly, all results are on 1B-parameter models. Without even partial scaling evidence (e.g., 3B subset), it’s uncertain if the same patterns hold at practical large-model scales (7B–70B).
2. The paper claims systematic variations in data composition and filtering but doesn’t analyze which factor contributes most to proxy predictability.
3. Although cross-validation and 20 random splits are used, the dataset of 50 models is small.
4. The feature-importance plot shows Kshot-RAG dominates, but it’s unclear whether the LTC gains come mostly from simple nonlinear reweighting. An ablation of proxy subsets could clarify if supervised fusion is truly synergistic or just re-scaling.

### Questions
1. Provide sensitivity analysis isolating pre-training objective vs. data composition.
2. Include scaling-up validation on at least 3B or 7B checkpoints, even partial.
3. Detail the statistical control to prevent leakage or overfitting in LTC training.
4. Clarify proxy combination mechanisms and perform ablation across subsets.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper studies whether pre-training indicators can predict fine-tuning performance when model scale is held constant. While perplexity is widely used as a progress indicator during pre-training and is known to correlate with downstream performance across scaling laws, its predictive power at a fixed model size remains unclear. The authors systematically pre-train 50 variants of a 1B-parameter language model, varying pre-training objectives, data mixture weighting, and tagging. They then fine-tune all variants across three downstream categories. Results show that standard perplexity correlates poorly with fine-tuning outcomes. In contrast, proxies such as span-corruption perplexity and k-shot performance better predict downstream results.

### Strengths
This paper addresses an important and underexplored question: whether pre-training indicators can predict fine-tuning performance when model scale is fixed. The authors formulate this as a pairwise prediction problem and conduct a systematic empirical study across 50 variants of a 1B-parameter LLM. The experimental results provide evidence that conventional perplexity is an unreliable predictor for post-SFT performance, while span corruption perplexity and few-shot evaluation metrics offer significantly better predictive power. The proposed supervised Learning-to-Compare framework further integrates multiple proxies to enhance prediction accuracy and demonstrates cross-task generalization.

### Weaknesses
1. While the experiments on 1B-parameter models are internally consistent, it would strengthen the paper to include one or two experiments verifying whether the observed trends extrapolate to larger, more realistic LLMs.

2. The evaluation currently covers only three SFT tasks (CMS, RAG, CBQA); it is unclear whether the proposed proxies generalize to broader SFT tasks.

3. The paper reports pairwise prediction accuracy as the main evaluation metric; it would be helpful to also present ranking-based measures.

### Questions
Please refer to weaknesses.

### Soundness
2

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
This paper investigates how to predict the fine-tuning (SFT) performance of large language models when model size and compute are fixed. Using 50 one-billion-parameter LLMs trained with different pre-training objectives and data mixtures, the authors show that standard perplexity fails to predict post-SFT performance. They introduce alternative proxies, like span-corruption perplexity and few-shot (k-shot) scores, and a learning-to-compare (LTC) framework that combines these signals via supervised classification. The LTC approach improves pairwise prediction accuracy by over 50% versus perplexity and reliably identifies the best checkpoints, offering a practical method for efficient model selection during pre-training.

### Strengths
1. The paper conducts comprehensive evaluation using multiple SFT tasks and metrics (CMS, RAG, CBQA). It tests both unsupervised proxies and supervised meta-predictors. It also provides systematic experiment design that include 50 model variants with consistent size and compute budgets.
2. The paper offers insightful empirical findings, showing strong evidence that perplexity is not reliable for fixed-size model selection.
3. I also like the pairwise evaluation formulation, which is simple and interpretable.

### Weaknesses
1. Experiments confined to 1B-parameter models; unclear if findings hold for larger LLMs (10B–70B) which is commonly used nowadays. While I understand this could because of the computation cost, a small-scale study would be interesting in this parameter size range.
2. The paper provides largely empirical insights and there are limited theoretical or mechanistic analysis on why certain proxies work better. It will be great if theoretical reasoning or model can be provided to shed some light.
3. The “learning-to-compare” framework is a straightforward application of supervised ranking. While it’s effective, it’s not conceptually novel.

### Questions
See weakness.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper addresses the problem of predicting which of two checkpoints will perform better after supervised fine-tuning (SFT). It evaluates the discrimination accuracy of two types of proxies: perplexity on the Pile and few-shot performance. These proxies are then used as features to train a classifier, which achieves higher classification accuracy than any single proxy.

### Strengths
* The problem addressed is both interesting and important.
* The experimental setup is described in detail, and the data used for analysis are provided in Table 6. This transparency increases confidence in the replicability of the results.
* The presentation is generally clear and easy to follow.

### Weaknesses
A core claim of the paper is that perplexity is a misleading indicator of performance after SFT. I believe that not enough evidence is put forward to substantiate this claim. First, only perplexity on the Pile dev set is considered. Authors should explore at least a few other PPL indicators; for example, PPL on a test set drawn iid from DC-0, PPL on the CC split of SlimPajama, PPL on the Wiki split of SlimPajama, and mean PPL on the train sets of the benchmarks considered. Relative to training models, these experiments should be rather cheap to run.

Secondly, the results imply that higher PPL is indicative of better performance (e.g., see Table 1 top row, this rule would lead to 100% - 33% = 67% accuracy for SFT-CMS). This is highly surprising and extremely counterintuitive (if it holds beyond Pile dev), so much so that it deserves additional explanation of (1) why this might arise in the considered setup and (2) whether we can we expect similar results for more realistic setups. For example, the architecture search community compares architectures almost exclusively based on PPL evaluations, should we sound the alarm that they are hill climbing in the diametrically opposite direction? I am inclined to think not.

The work also has a serious flaw in the analysis: it does not account for statistical significance of either (1) differences in benchmark performance, or (2) differences in the pre-training proxy indicators. For example, in Table 6 SFT-CMS, model 18 has a mean accuracy of 72.98, whereas model 19 has a mean accuracy of 72.94. These two SFT models should be considered equivalent. Authors should repeat their analysis taking into account statistical significance in model performance and the individual proxies.

Authors find that the most accurate proxy is Kshot-RAG. Looking at Table 6, I would argue that this is simply because it is the one that gives the most resolution in ranking models (e.g., and not because Kshot-RAG is otherwise particularly indicative of downstream performance). I encourage authors to make the following simple analysis: among the 12 benchmarks considered, take the top 4 that give the most resolution in ranking models (e.g., those benchmarks with largest mean difference in accuracy between contiguous model ranks relative to mean accuracy across all ranks). Then check how good of an indicator mean performance across these 4 models is.

### Questions
* For QA benchmarks, are models evaluated (1) by choosing the answer choice with lowest PPL given the question, (2) MMLU-style (e.g., A, B, C, D), or (3) by generating from the model and using some form of answer matching? I would recommend the former as this should give the most resolution in the benchmark evals.

### Soundness
1

### Presentation
2

### Contribution
1
