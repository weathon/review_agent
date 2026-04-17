# How NOT to benchmark your SITE metric: Beyond Static Leaderboards and Towards Realistic Evaluation.

- Decision: Accept (Poster)
- Scores: 6, 4, 8, 6

## Abstract
Transferability estimation metrics are used to find a high-performing pre-trained model for a given target task without fine-tuning models and without access to the source dataset. Despite the growing interest in developing such metrics, the benchmarks used to measure their progress have gone largely unexamined. In this work, we empirically show the shortcomings of widely used benchmark setups to evaluate transferability estimation metrics. We argue that the benchmarks on which these metrics are evaluated are fundamentally flawed. We empirically demonstrate that their unrealistic model spaces and static performance hierarchies artificially inflate the perceived performance of existing metrics, to the point where simple, dataset-agnostic heuristics can outperform sophisticated methods. Our analysis reveals a critical disconnect between current evaluation protocols and the complexities of real-world model selection. To address this, we provide concrete recommendations for constructing more robust and realistic benchmarks to guide future research in a more meaningful direction.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper discusses the evaluation of metrics for Source Independent Transferability Estimation. These metrics attempt to predict pretrained models' downstream performance after finetuning on a given dataset. The paper raises three points of criticism aimed at current evaluation practices based on a small set of models and datasets: First, the set of included models is often unrealistic and dominated by differently sized versions of a small set of architectures. Evaluations results are heavily dependent on the set of included models. Second, a heuristic static model ranking outperforms all evaluated methods. Third, current evaluation metrics focus on rank correlations, partially masking failures to predict the magnitude of performance differences.

### Strengths
- The "static ranking" experiment relatively convincingly demonstrates that the current evaluation setup for SITE metrics is flawed

### Weaknesses
- The paper could be strenghtened by producing an actual setup that mitigates the discussed issues.
- The writing could be improved in various places. 
   - For example, the discussion mentions a bunch of datasets without citations 
   - Some claims in the paper are also unsupported (by either citations or arguments). Example: "In addition, SDTE metrics are not reliable when the discrepancy between the source and target datasets is very high"
    - As a minor nitpick, the paper could also benefit some additional proofreading in terms of grammar/sentence structure. For example: "For purpose of this study we examine the following metrics on standard benchmark"

### Questions
How exactly was the static ranking chosen? The paper says "based on the model size and an alternation of ResNet and DenseNet model families", but there seem to be two densenets followed by two resnets at ranks 4-7.

### Soundness
3

### Presentation
2

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
The authors propose a deeper look into the problems in the assumptions made for evaluating transferability estimation. They show how the the status quo benchmark framework (datasets, models, evaluation protocols) is faulty will 3 concrete emperical critiques and make concrete suggestions for good practises that future works can focus on.

### Strengths
1. Well-motivated problem statement: shifting from SDTE to SITE is timely and important as we know less and less about the true distribution of training data. Also, with scale, distribution matching will only get more expensive.

2. Probing into the correlation between the difference in magnitude of accuracy and tranferability is a critical analysis the paper did quite well. With more fine-grained studies, it can be quite insightful to practioners.

### Weaknesses
My main issues with this paper revolve around the lack of extensive experimentation: to make the strong claim that a form of benchmarking is "fundamentally flawed" requires significantly more analysis than currently provided. 

1. Even though the paper prescribes including ViT, ConvNeXt, MLP-Mixer etc, the core experiments to determine the core claim does not extend beyond the standard set of models (ResNet, DenseNet, etc). Though intuitive, we need emperical evidence that the three critiques will hold true with a diverse pool of models.

2. Related works such as [1] have identified similar problems, such as "bias toward high-capacity models" and the unrealistic model search space, in addition to other relevant problems this work has not addressed such as the lack of fully labelled target datasets in real-world scenarios. They propose solutions such as adding self-supervised models and evaluating transferability estimation on model weights before and after optimisation using the Wasserstein Distance(even though their analysis focusses on SDTE, both these solutions are applicable to SITE). Existing literature already extends the scope of this work by providing concrete solutions to fix the evaluation setup, calling the novelty of this work into question.

3. Finetuning needs to be more vigorous: Small changes in seed, optimizer, number of epochs, etc can produce statistically significant variance in model performance. The paper does not but should report independent fine-tuning sweeps and checkpoints and report mean &plusmn; standard deviation to check for this sensitivity. Also, the same seeds should be applied to all models and model families.

4. I disagree with the claim that larger models naturally outperform their smaller counterpart. Recently, [2] has shown that sample-level benchmarking showcases interesting scenarios where older versions of the same model or smaller models of the same family outperform their newer iteration/larger size counterpart for specific subsets of the evaluation datapool. This brings into question whether the static ranking based on model size is robust to more specific data subsets.

5. The NLP results feel like a footnote in the appendix. It should be as rigorous as the image classification experiments. In general, for a paper claiming transferability estimation metrics are flawed and that the insights are applicable to other domains and tasks, it is in the scope of the work to demonstrate that. Related works such as [3] do show analysis on medical image classification.


Minor points:
1. L137: using the same notation n for (T_n) and data (x_n, y_n).

[1] Kazemi et al, Benchmarking Transferability: A Framework for Fair and Robust Evaluation, 2025

[2] Ghosh et al, ONEBench to Test Them All: Sample-Level Benchmarking Over Open-Ended Capabilities, ACL 2025

[3] Chaves et al, Back to the Basics on Predicting Transfer Performance, 2024

### Questions
1. How does the fidelity affect top models? Practitioners generally care about the top performing models, so I would be curious to see how $\Delta$Acc - $\Delta$T correlation for model pairs where at least one of the models is in the top-k or the bottom-k. This would tell us if fidelity fails in the current evaluation setup for important models.

2. Re-iterating from W4, does the static leaderboard still outperform SITE metrics for granular sample-level datapoints?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper provides an empirically-backed critique of existing evaluations for SITE (source-independent transferability estimation). This problem concerns how to estimate which of a pool of models will perform best on a target domain after finetuning. The paper makes several well-received points: existing evaluations tend to use overly simple pools of models and datasets. As a result, naive baselines like a hand-crafted static ranking of models can outperform highly sophisticated SITE methods. Popular ranking metrics also tend to understate how poorly SITE scores reflect actual differences in finetuned model accuracies (on an absolute scale, not just in terms of rank ordering). The paper offers several constructive recommendations for how to improve evaluation in the space. I believe the strongest point is that evaluations must capture a meaningful degree of variance in finetuned model performance across different domains, i.e. rank orderings of finetuned models on target datasets should show a meaningful degree of variance that SITE metrics should then explain.

### Strengths
- Very important: The paper is well written and very clear.
- Very important: The paper’s experiments get straight to the point and effectively undermine the validity an existing, popular approach to SITE benchmarking.
- Very important: The recommendations are fairly actionable and should push work on this problem in the right direction.

### Weaknesses
- Of some importance: I am not deeply familiar with work on SITE and so a small concern of mine is that this paper is strawmanning what is a small and not overly influential line of research. Yes, I agree that these papers are being published in top venues. But is this really the mainstream way that people think about capturing low-dimensional properties of AI’s adaptability to downstream tasks? Forgive me for comparing this problem area to LLM evaluations, but I think what people tend to put stock in these days is a small set of benchmark scores for an AI system. A set of 3-4 benchmark scores is taken as gospel for which model is most “intelligent”, and therefore going to perform best when tailored to a specific downstream task. Sure, there is clever ML work going on to understand properties of vision model representations and how they relate to downstream performance. But this is not the main thing I have in mind when I think of how people assess broad transferability of AI systems.
- Of minor importance: I had a small quibble with this point: “practitioners are not interested in knowing whether they should use a ”bigger vs. smaller” ResNet, but which architecture performs best under a fixed size budget.” I think this point is stated a little more elegantly in the recommendations section: “To isolate architectural inductive bias, models should be matched on computational budgets (e.g., parameter counts or FLOPs).” In reality, practitioners care about a lot of things, including size, training speed, inference speed, ease of access to models, interoperability with common training and deployment libraries, etc. I think the claim in the recommendations section helps point out what kind of model pool would lend support to scientific conclusions about predicting downstream adaptability of different model architectures. Still more consideration might go into constructing a model pool that reflected all the various axes of variance that practitioners care about (which is more than just performance vs. size).

### Questions
- please feel free to comment on the noted minor weaknesses/concerns above
- typo: as two scorings with induce the same order can differ in their evaluation merely due to calibration

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper argues that the standard evaluation protocol for Source-Independent Transferability Estimation (SITE) is misspecified and overstates progress. Unlike Agostinelli et al., this work targets the protocol rather than metric design.

The authors show three failures:

- **1. Unrealistic model space.** Common SITE pools are dominated by scaled variants of a few CNN families (chiefly ResNet and DenseNet). Removing the largest models sharply drops Kendall’s τ_w across methods, exposing brittleness and a “bigger-is-better” effect that drives rankings.

- **2. Static leaderboard.** A trivial, dataset-agnostic ranking -- argely by model size and family -- matches or beats SITE metrics across datasets. The same few large models (e.g., ResNet-152) repeatedly top the charts; a fixed size-aware ranker attains higher τ_w than all evaluated SITE metrics.

- **3. No fidelity checks.** SITE scores are not tested for decision fidelity -- whether score gaps map to meaningful accuracy gains. Pairwise ∆score correlates weakly with pairwise ∆accuracy, making score differences hard to interpret for cost–benefit trade-offs.

The paper ends with best-practice recommendations and a checklist for better evaluation.

### Strengths
- **Timely and important.** Evaluation design choices meaningfully shape this subfield; focusing on protocol is valuable in its own right.

All the three findings seem correct and fatal:
- Ablations show over-represented large models inflate results (Fig. 2).
- The static-ranker baseline dominates (Table 1) and should be fairly easy to reproduce.
- The ∆score vs ∆accuracy study addresses an often-ignored need and shows that score gaps frequently fail to predict accuracy gains (Fig. 4; App. B).

It's correct and likely impactful on the field of source-independent transferability estimation.

### Weaknesses
**[Critical] Could the paper implement the fixes recommended instead of best practices?**

The critiques are actionable, yet the paper stops short of fixing it. Best-practices section seems unsuitable as it reads as advice masquerading as work that is not done.

- If the pool over-indexes ResNet/DenseNet, please propose and release a diversified model space (include ViTs and other modern families). Alternatively consider a two-stage aggregation: Compute SITE within each architecture family, then aggregate fairly across families. But a new benchmark, if proposed, instead of simply this critique could be widely adapted.

- If a static ranker “solves” current datasets, expand the dataset suite (e.g., include harder, real-world transfer targets such as those discussed in *Does Progress on ImageNet Transfer to Real-World Datasets?* by Fang et al.). Leaderboard suites from *Benchmark suites instead of leaderboards for evaluating AI fairness* by Wang et al. might also be valuable.

- For fidelity, propose an actionable fidelity metric grounded in decision quality (∆score vs ∆accuracy study already seemed promising). To provide some straws in the wind -- social-choice theory offers rank-aggregation tools with known axioms and trade-offs; some classical criteria may fit SITE’s needs.

If fixed, the resulting benchmark could be potentially quite valueable for the next generation of SITE algorithms.

**[Major] Central findings #1 and #2 already seems well-known.**

- The strong performance of dataset-agnostic rankings (by size or ImageNet top-1) echoes established findings -- e.g., *Do Better ImageNet Models Transfer Better?* (Kornblith et al., CVPR’19) and *Do ImageNet Classifiers Generalize to ImageNet?* (Recht et al., ICML’19). Could the work clarify what is new beyond confirming known transfer trends.
- Lots of similar work: *Back to the Basics on Predicting Transfer Performance*, *Scalable Diverse Model Selection for Accessible Transfer Learning* (relevant to finding #1) are not cited or differentiated from. Could the authors detail the core differences to past work?

**[Minor] Reporting and metrics.**

- How large are the uncertainty bands (confidence intervals and significance tests) for τ_w to indicate whether observed differences are meaningful.
- The work seemed to need more polishing to concisely state the message. Please confirm whether my understanding and critiques are correct, I had to read this multiple times to understand.

### Questions
Please address weakness 1 and 2.

Overall, a well-argued, well-executed critique of current SITE evaluation with compelling evidence -- solid work in my opinion. The work would be much stronger if it shipped a concrete repaired benchmark (diversified model pool, harder datasets, and a fidelity metric), rather than "best-practices" guidance.

### Soundness
4

### Presentation
2

### Contribution
4
