# Adaptive Testing for LLM Evaluation: A Psychometric Alternative to Static Benchmarks

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 6, 2, 4

## Abstract
Evaluating large language models (LLMs) typically requires thousands of benchmark items, making the process expensive, slow, and increasingly impractical at scale. Existing evaluation protocols rely on average accuracy over fixed item sets, treating all items as equally informative despite substantial variation in difficulty and discrimination. We introduce ATLAS, an adaptive testing framework based on Item Response Theory (IRT) that estimates model ability using Fisher information–guided item selection. ATLAS reduces the number of required items by up to 90\% while maintaining measurement precision. For instance, it matches whole-bank ability estimates using only 41 items (0.157 MAE) on HellaSwag (5,600 items). We further reconstruct accuracy from ATLAS's ability estimates and find that reconstructed accuracies closely match raw accuracies across all five benchmarks, indicating that ability $\theta$ preserves the global performance structure. At the same time, $\theta$ provides finer discrimination within accuracy-equivalent models: among more than 3000 evaluated models, 23--31\% shift by more than 10 rank positions, and models with identical accuracies receive meaningfully different ability estimates. Code and calibrated item banks available at https://anonymous.4open.science/r/ATLAS-3210/README.md.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors:

- propose and motivate an IRT framework for adaptively sampling items to use to benchmark language model (LM) capabilities
- evaluate their algorithm on 5 benchmarks

### Strengths
- Clear, concise, accurate title
- Important problem - accurately evaluating language models' capabilities at specific tasks is good
- The paper is well written and easy to follow. (Note: I believe some details are omitted, which I pointed out under Questions)

### Weaknesses
1. The goal when evaluating language models is “How good is model X on task Y?” Here, the primary metrics of interest is MAE between an IRT estimate on a subset of data and the corresponding IRT estimate on all the data. Thus, MAE is a proxy metric that doesn’t really capture what we care about.

2. When considering efficiency, the real concern for practitioners is that evaluating models requires paying for accelerators (GPUs, TPUs, whatever) to run these models.  Something like “Selection Time (s)” is not a real consideration; if it takes me 60 seconds to select the next item, I could have spent those 60 seconds running inference on the models rather than choosing the next point. My guess is that, when controlling for chip-seconds, evaluating random items provides better performance than pausing for 10-75 seconds per item (Table 2) to choose the next item.

3. Key experimental results are weak. Specifically, in Table 1, when evaluating how good ATLAS is, let’s first consider a reasonable null distribution. There are 7 algorithms and 3 of them are ATLAS. If all of them are equally good up to randomness (e.g., from the scores, from sampling, from model training, etc.), then we expect ATLAS to score best in 3/7 and MetaBench to score best in 2/7. It looks like ATLAS scores best in 3/5 benchmarks and MetaBench scores best in 2/5 benchmarks. **Table 1 looks to me like compelling evidence that these algorithms are basically equal.**

4. Methodologically, it is unclear how finicky this methodology is or how well it works generally. The method has many degrees of freedom to play around with (e.g., which data to exclude Section 3.2, what SE threshold to use, how to set $\tau$, etc.), leaving it ripe for unfair comparison with baselines. I also don’t see evidence of preregistration, which would preclude such post-hoc favourable treatment.

### Questions
## Title

- Solid! Thank you for a clear and concise title

## Introduction

- Line 051-052: Where are the citations for the benchmarks? WinoGrande, TruthfulQA, etc.
- Line 053: Where is the citation for MetaBench?

## Section 3 Methodology

- Lines 141-143: “Models with extreme scores (below 0.1st percentile) are excluded to
prevent parameter estimation instability, as IRT’s sigmoidal functions become under-constrained at
the boundaries.” Are models with extremely high scores (e.g., *above* 0.1st percentile) similarly excluded for the same reason?
- Lines 171-175: Can you please add an Appendix explaining this in more detail? I can’t quite follow. What exactly does “calibrate” mean in this context? What are “linking anchors”? 

## Section 4 Experiments

- Line 274: Why is the metric of interest (MAE) defined between $\hat{\theta}_{\ell}$ and $\hat{\theta}_{\ell}^{whole}$ instead of something real/tangible, such as average score on the benchmark? This MAE metric seems more focused on “Does the IRT approach yield a consistent estimate on a subset of items as it does on the full set?”, which is a proxy metric that doesn’t seem important. What we want to know is: how good is model X at task Y?
- Why is lowered “Test Overlap Rate” considered good?
- Table 1: nit: Please state what bold and underline and dashed underline means in the caption.
- Table 1: When evaluating how good ATLAS is, let’s first consider a reasonable null distribution. There are 7 algorithms and 3 of them are ATLAS. If all of them are equally good up to randomness (e.g., from the scores, from sampling, from model training, etc.), then we expect ATLAS to score best in 3/7 and MetaBench to score best in 2/7. It looks like ATLAS scores best in 3/5 benchmarks and MetaBench scores best in 2/5 benchmarks. **Table 1 looks to me like compelling evidence that these algorithms are roughly equal.**
- Table 1: Presumably, the MAEs are averaged over multiple items and/or models? If so, where are the notions of uncertainty e.g., standard errors, confidence intervals?
- Table 1: I personally prefer visualizations over tables. You could easily and helpfully visualize this as a pointplot https://seaborn.pydata.org/generated/seaborn.pointplot.html
- Table 2: To clarify, what does “fast selection times” measure exactly? Is this the time to select the next item? Are we presuming that we’ve already evaluated all LMs on all items?
- Lines 359-365: The argument that small test overlap rate prevents test set contamination seems like nonsense. If I pretrain a model directly on the test set, why does the subset of items chosen for evaluation prevent the model from scoring above its true ability? 

## Appendices

- Line 688: Why is the definition of MAE defined twice, once here and once below on line 759? Same with Average Item Exposure Rate.
- Line 753: Why does Appendix A Evaluation Metric Definitions come after Appendix D Data Processing Details (line 727)?
- Line 755: The reference to the Section is missing.
- Line 786: The reference to the Section is missing.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces ATLAS, an LLM evaluation method based on item response theory (IRT) and computerized adaptive testing (CAT). Specifically, the authors fit 3PL IRT models to LLM responses from the Open LLM Leaderboard and, during evaluation, use the IRT parameters to select items dynamically via Fisher information. In their experiments, they show that this leads to more precise ability estimates compared to recent static evaluation methods based on IRT (TinyBenchmarks, MetaBench), plus several other advantages.

**NB:** ATLAS is almost identical to Fluid Benchmarking, a method proposed in a recent [COLM paper](https://openreview.net/forum?id=mxcCg9YRqj): the COLM paper also fits IRT models to the Open LLM Leaderboard and uses the IRT parameters to conduct CAT, selecting items dynamically via Fisher information, exactly as the current paper does. There are minor differences (e.g., ATLAS uses a 3PL IRT model while Fluid Benchmarking uses a 2PL IRT model), but otherwise the methods are the same. Since the COLM paper was published on July 7th, it is contemporaneous work according to the [ICLR rules](https://iclr.cc/Conferences/2025/FAQ), and I will not hold it against the authors that they did not mention it in their paper. However, given the strong similarities, I highly recommend that the authors add a discussion.

### Strengths
ATLAS draws upon several decades of research in psychometrics and shows that the methods developed in that field can be fruitfully applied in the context of LLM evaluation. I liked it that the authors thought very carefully about how best to adapt IRT/CAT to the LLM domain (e.g., by using common-person calibration). The experimental setup is also sound, and the authors convincingly show that ATLAS offers advantages for LLM evaluation (but see my concerns below).

### Weaknesses
There are currently several weaknesses that undermine the contribution of the paper. If the authors address them, I will consider raising my score.

- The experimental section misses key details. Specifically, it is unclear whether the LLMs used for evaluation were already used for fitting the IRT models (which would be problematic). Further, if there _was_ a clear train-test split, it is unclear how it was determined. This limits the credibility of the reported results.

- For measuring precision, the authors solely examine how well different methods recover _ability_ as estimated on the full benchmark. However, TinyBenchmarks and MetaBench (and most other methods from the efficient evaluation literature) aim to recover _accuracy_ on the full benchmark. One reason for this is that most practitioners are interested in getting an estimate of the final accuracy, not ability. Of course, there is an intrinsic tension between the two (which the authors nicely demonstrate in the paper), and I agree in principle that ability should be preferred over accuracy. Still, this should be explicitly addressed in the paper, especially since the item sets from TinyBenchmarks and MetaBench were optimized against accuracy, meaning that the current comparison in the paper is note entirely fair. Thus, I think the authors should add an analysis of how well ATLAS recovers full-benchmark accuracy and compare against the same baselines.

- The discussion on contamination is not convincing and seems to be based on wrong assumptions about contamination and LLM evaluation. Contamination happens when a model is _trained_ on items from a benchmark's test set, not when it is _evaluated_ on them during pretraining, so I do not see how CAT (a method for evaluation, not training) would offer any advantage compared to static testing. For example, the authors write that "[e]ven smaller banks maintain low exposure rates, [...] making systematic memorization during pretraining practically impossible" (363-365), but this does not make any sense since unlike humans, evaluation does not cause contamination with LLMs. In other words, even if a model has "seen" all items of a benchmark multiple times as part of evaluations during pretraining, this will not allow it to memorize any of those items.

### Questions
- Will you release your code? What programming languages/packages did you use?
- I was surprised that you excluded MMLU from your analysis, despite the fact that it is also part of the Open LLM Leaderboard, and TinyBenchmarks and MetaBench also examine it, so it would have been very easy for you to include it in your experiments. MMLU is also the most widely used out of all the Open LLM Leaderboard benchmarks. Can you comment on your rationale here?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes ATLAS, an adaptive evaluation framework that selects a subset of the benchmark for evaluation to make running evaluation cheaper. The authors propose a three parameter logistic IRT model for fitting the probability of a correct response, and then adaptively selects the most informative items using fisher information. Using ATLAS, the authors are able to prune down benchmarks by over 90% on tasks like HellaSwag.

### Strengths
- The authors tackle an important problem of making evaluations cheaper for large language models, given a lot of benchmarks are used to track a given model's capabilities.
- The proposed methodology is clear by formalizing evaluation as a latent-ability measurement with a three-parameter logistic model.
- The results are nice compared to other baselines like TinyBenchmarks with huge reduction in size of the evaluation sets.

### Weaknesses
- Code and calibrated items are missing in the provided link.
- There are some inherent issues with IRT framing and using reduced sizes for evaluation of language models. See [1]
- Inconsistent claims and results: the main text of the paper mentions good fits with RMSEA $\leq 0.05$ but Table 4 reports otherwise.
- Current framework is only applicable to MCQ tasks, but MCQ benchmarks have many inherent problems [2]. Generalizability to many modern evals which are free-form like math, coding, reasoning, etc. is not clear.

[1]: Quantifying Variance in Evaluation Benchmarks, Madaan et al., 2024

[2]: Answer Matching Outperforms Multiple Choice for Language Model Evaluation, Chandak et al., 2025

### Questions
I have asked most of my questions in the weaknesses section above.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes an alternative to static benchmark evaluation of LLMs using ideas from computerized adaptive testing. The method they propose is called ATLAS. They fit a 3PL IRT model to open LLM leaderboard (a collection of models, benchmarks, and binary responses); one model per benchmark. In doing so, they follow a pretty standard idea of using the Fisher information of theta (the ability score of an item) based on the fit IRT model to guide the informative item selection process. They evaluate this technique against baseline efficient LM benchmarking papers like tinyBenchmarks and MetaBench as well as against a random subsampling baseline; evaluation criteria is based on ability to accurately produce the same assessment but on fewer examples.

### Strengths
The core ideas here are interesting and can motivate new research in improvements to how we perform LLM evaluation. It’s nice to pull in techniques from other fields like psychometrics and CAT to see how we can improve our field. The experimental results showing efficiency versus baselines are good.

### Weaknesses
I want to be upfront: There is concurrent work at COLM 2025 called Fluid Language Model Benchmarking that is highly similar (also inspiration from psychometrics & CAT, also fitting IRT models on Open LLM leaderboard, using the Fisher information for item selection, also baselining against tinyBenchmarks and MetaBench). I did not penalize this work for overlap with concurrent work as the COLM paper was published around the same time this paper was submitted.

That being said, there are some issues:

**On claims that ATLAS improves data contamination**
First, this paper makes a big point about “data contamination” and how this efficient LM benchmarking strategy can mitigate data contamination. Indeed, data contamination in our field is an issue, but this paper is suggesting that efficient LM benchmarking is actually a way to mitigate this issue (by simply revealing less data to model developers in the process). I don’t buy this argument at all.

For example, L037 suggests static benchmarks are easier to leak into “pretraining corpora”. The problem is, this is not solved by the proposed method. For example, let’s consider how this paper performs experiments by applying ATLAS to Open LLM leaderboard data. All of that is actually reusing benchmark data that’s already public & thus already revealed to model developers. If the research community is to adopt ideas like ATLAS for benchmarking but ultimately still rely on existing public test examples, then the contamination problem is not about efficient methods like ATLAS, it’s about public vs private test sets.

Ok, so now let’s consider the argument that efficient evaluation methods like ATLAS, while studied/demonstrated reusing static benchmarking data, would be deployed on new, private test sets. And thus, the fact that we make use of fewer testing examples helps prevent contamination.

There are a couple issues with this problem. First, it makes sense in computerized adaptive testing for humans. Humans, when seeing test examples, immediately learn from those observed test examples. And thus, any time you test humans, contamination happens. Language models don’t inherently learn from examples as we test them. Language model state is captured in its weights at the end of training and no amount of evaluation will update those weights unless the model developer explicitly decides to include test set examples into the training data.  So again, adaptive testing techniques like ATLAS aren’t actually addressing the contamination issue in machine learning, which is very much about (improper) practices of model developers.

Now then finally, let’s consider – maybe one can argue that adaptive testing methods like ATLAS, by virtue of them withholding test examples, work well against adversarial model developers who are incentivized of including any test examples in the training data. Again, I don’t buy this argument. In the course of a normal model development cycle, let’s say fitting scaling laws or performing data mixing ablations for pretraining, model developers will train hundreds or even thousands of language models, each of which will have to undergo some ATLAS evaluation; what is the likelihood that “data contamination” is actually being addressed in this scenario? Even revealing a small percentage of the benchmark instances per model, at the large experimentation scale that naturally happens in model development, we’re likely exhausting and thus “contaminating” full static benchmark collections (or what this paper calls our “banks”) rapidly. Then the way to solve this problem is not about adaptive testing, but it’s about scalable generation of new test examples to keep up with how quickly we’re exhausting our “banks”. 

Overall, I find the emphasis on how ATLAS and adaptive testing improves “data contamination” is incorrect and detracts from the merits of this paper. 

**Unfair baseline comparisons**
The baseline comparison uses MAE as the target metric against the “full bank theta”. That means that the evaluation setup is assuming there exists some ground truth theta that can be estimated on the full benchmark and efficient benchmarking techniques are evaluated on their ability to approximate it with fewer examples. The problem with this is, the baselines tinyBenchmark and MetaBench weren’t designed with IRT in mind; the proposed method ATLAS was. So it is unfair to define an IRT criteria (approximate full bank theta) and show that your IRT method is better than non-IRT methods. The correct evaluation would have been to show that the proposed ATLAS method can efficiently approximate the actual benchmark evaluation metric (accuracy).

### Questions
I think this paper would be in very publishable state if can address those weaknesses above: Remove the claims about data contamination and add experimental results showing ability to reconstruct "accuracy" instead of MAE against full bank theta. Is this something palatable to the authors?

### Soundness
2

### Presentation
3

### Contribution
3
