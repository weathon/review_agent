# Train-before-Test Harmonizes Language Model Rankings

- Avg Score: 7.00
- Decision: Accept (Oral)
- Scores: 6, 8, 8, 6

## Abstract
Existing language model benchmarks provide contradictory model rankings, even for benchmarks that aim to capture similar skills. This dilemma of conflicting rankings hampers model selection, clouds model comparisons, and adds confusion to a growing ecosystem of competing models. In this paper, we take a different perspective on model comparison: instead of relying on out-of-the-box performance via direct evaluation, we compare model potential by providing each model with identical benchmark-specific fine-tuning before evaluation. We call this approach train-before-test. Our primary contribution is a comprehensive empirical evaluation of model potential across 24 benchmarks and 61 models. First, we demonstrate that model potential rankings obtained through train-before-test exhibit remarkable consistency across all benchmarks. Whereas traditional rankings demonstrate little external validity under direct evaluation, they enjoy a significant degree of external validity when applying train-before-test: model potential rankings transfer gracefully from one benchmark to another. Second, train-before-test restores the connection between perplexity and downstream task performance, lost under direct evaluation. Remarkably, even pre-finetuning perplexity of a base model predicts post-finetuning downstream performance, suggesting that ranking consistency reflects inherent model potential rather than fine-tuning artifacts. Finally, train-before-test reduces the model-score matrix to essentially rank one, indicating that model potential is dominated by one latent factor, uncovered by train-before-test. While direct evaluation remains useful for assessing deployment-ready performance, train-before-test provides a complementary lens for understanding achievable performance of models after adaptation.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper studies how LLM performance varies across tasks. It proposes that instead of directly evaluating LLMs on downstream tasks, LLMs should first be finetuned on the training set of the downstream task (a paradigm they call "train-before-test"). The paper shows via a large-scale empirical evaluation that model rankings after train-before-test are much more correlated with each other across benchmarks than the out-of-the box performances are.

### Strengths
1. The paper provides a comprehensive empirical evaluation to show that a) model rankings change after finetuning and b) model ranks are much more correlated across benchmark tasks after finetuning.
2. The paper shows that pre-finetuning perplexity model rankings correlate with post-finetuning benchmark performance model rankings for base models but not for instruction-tuned models.
3. The paper shows that after fine-tuning, the model-benchmark score matrix becomes nearly rank 1.
4. The paper is quite well-written.

### Weaknesses
I see no major weaknesses of this paper. My biggest concern is its difference from [1], which is currently mentioned a few times in the paper. Lines 154-157 state that the paper introduces the "train-before-test" paradigm, but [1] already proposes to finetune on task-relevant data before evaluation (Section 2.1 of [1]). Can you more thoroughly articulate the differences between your paper and [1]? As it stands, I think this paper provides valuable insights about the stability of model rankings across tasks after finetuning, but the magnitude of this paper's contribution is smaller in light of [1].

[1]: "Training on the Test Task Confounds Evaluation and Emergence", Dominguez-Olmedo et al, ICLR 2025.

### Questions
1. "Model potential" does not feel like the right term for finetuned performance. To me, the natural meaning of "potential" would apply *before* finetuning instead of afterwards. The phrase is currently used to mean performance after finetuning, so I'd suggest thinking of a different term.

Small comments:
- Line 46: "\citet{Kaplan et al. (2020)}" -> "\citep{Kaplan et al. (2020)}"
- Line 52: I'd avoid "regret-free", since it's a technical term in online learning for sublinear regret, which is not what's being referred to here.
- Line 310: "\texttt{Wiki}pedia" -> "\texttt{Wikipedia}" and "\texttt{Stack}Exchange" -> "\texttt{StackExchange}"

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This work is follow-up to a previous ICLR 2025 paper proposing “training on the test set” as a better means of evaluating language models; this prior work showed that properties like “emergence” arise due to language models being tested on benchmarks that don’t take into account whether the model has been adapted to the downstream task; with “training on test set”, phenomena like emergence disappear. This work builds on this idea by showing further benefits of “training on test set”:

This work shows that after performing “training on test”, model rankings between similar benchmarks (e.g. benchmarks both testing QA) become more similar. Intuitively, one interprets this as – If you have multiple instantiations of benchmarks that all seem to test a common capability (e.g. medical knowledge vs math vs ..), then evaluating on those benchmarks in standard way can lead to very different evaluation results due to benchmark-specific properties (or if the models have trained in a manner arbitrarily to favor one benchmark but not the other), but with “train on test”, one is evaluating “potential” of the model on the underlying capability and not these other task-specific factors. In a similar analysis, this work shows that after performing “training on test”, model rankings between perplexity evaluations and downstream tasks are also more aligned.

### Strengths
This is a good paper and merits publication at ICLR. While it is a follow-up to a previous ICLR paper, it adds net new contributions, notably showing how “train on test” fits into the existing language model benchmarking paradigm, which the previous work did not show explicitly. This work also shows interesting finding about how to unlock correlation between perplexity and downstream benchmark, which has long been a frustrating topic in language modeling research. Experiments are sensible, varied in scope of models used in experimentation (though would’ve preferred to have seen more modern fully open models like Olmo or SmolLM; Pythia is outdated & not reproducible because not all data is public).

### Weaknesses
I think the work is good, so not too many critiques. I would say first, the heat map figures for rank correlation are hard to read (Fig 3, 4, 5). Maybe use a different color scheme, consider annotating where we should pay attention, and make caption a bit more self contained so we can read the Figure + grasp the finding instead of having to cross reference it with the body text.  

Given the chosen models, I would’ve liked to have seen more findings that take advantage of: (1) comparing models of similar size across families, (2) comparing models within same family of different sizes. Aside from a Qwen analysis, this paper didn’t really dive into any of that, which is a shame.

### Questions
Even after this fine tuning, there is still not perfect correlation between (i) pairs of benchmarks measuring similar underlying capabilities, and (2) perplexity and downstream benchmarks. In fact, the best these rank correlations reach is somewhere around 0.7-0.8. That means there is something else irreducible that causes model rankings to be different. I would love to see authors provide a bit of discussion as to whether this is due to (a) intrinsic benchmark noise (an interesting reference would be Signal to Noise 2025 paper by Heineman et al though I understand it was published after this submission), (b) some differences in the underlying potential that these benchmark pairs are capturing, (c) some benchmark specific properties that even finetuning isn’t helping with, or something else? Just want to understand what is “left” between this work & maximal rank correlation. 

Can you explain more the finding in Sec 3.4? In short, it seems that running PCA before train-then-test shows a single main principal component that is driving most of the variation in benchmark scores; and that this component is highly representative of pretraining compute. After train-then-test, it looks like, while the PC1 explains more variation, the conclusion about pretraining compute being the primary driver is still the same, is it not? I’m not understanding what net new learning we are getting here or how to interpret the significance of increasing the amount of variation that is explained by PC1.

Similarly, I’m not sure what is the takeaway from Qwen PC1’s analysis.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This work introduces a way compare model ranking based on performance of models after they are fine tuned on train split for a task. they say that this approach measures "model potential". The approach is dubbed "train-before-test". Through extensive experimental results, the paper shows that train-before-test stabilizes rankings of models: whereas before (under traditional protocol, which they dub "direct evaluation") rankings of models would change depending on benchmark, now rankings transfer between one model and the others. Overall, the paper represents an interesting contribution, and would benefit from being presented at ICLR.

### Strengths
- The paper is generally well motivated, easy to follow, and well written.
- Experimental setup is mostly sound (see weaknesses below), and the paper studies models across many LM families. 
- I appreciate the framing of model potential as a mean to pick model to best adapt to a task. paper would benefit from stating the goal of train-before-test even more explicitly in the abstract: the presented technique is NOT an intrinsic evaluation of models as finished products, but as starting point for fine tuning.

### Weaknesses
- One limitation of this approach is that it does not provide an estimate of the magnitude of improvement. This could have been achieved by either proposing a way to average score, or use rankings to determine if any two models statistically different. Given the focus on practitioners using this method to choose models for downstream applications, providing a single, easy to interpret number is crucial. 
- The paper lacks other comparison with other techniques that could be used to improve rankings. For example, the paper evaluates all models using shots, which gives puts models, especially those that are not instruction tuned at a major disadvantage [\(Gu et al., 2024\)](https://arxiv.org/abs/2406.08446). The test split of some of these benchmarks is also very small, and ranking stability would likely improve by adding training samples to the test set (see Fig 9 of [Heineman et al., 2025](https://arxiv.org/abs/2508.13144)) . 
- I would caution against comparing with perplexity, as decontamination cannot be ensured for all models used.

### Questions
N/A

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
Disclosure: I reviewed an earlier version of this paper for NeurIPS. I had significant concerns with that draft, but the revised version is substantially improved.

The paper addresses a common problem in LLM benchmarking: benchmarks often result in contradictory model rankings, even if they are supposed to measure the same underlying capability (e.g., reasoning). The authors offer a new perspective on this problem: they suggest to focus on model potential after finetuning rather than model performance under direction evaluation. Thus, they propose to add a step of task-specific finetuning before evaluation, an approach they call _train-before-test_. In their experiments, the authors show that train-before-test leads to more consistent model rankings across benchmarks, and even increases the correlation with perplexity-based model rankings.

### Strengths
The problem addressed by the paper is important. The experiments conducted by the authors are very extensive, comprising a large set of LLMs and benchmarks. The paper is well written and easy to follow.

Compared with the NeurIPS submission, the revised framing in terms of model potential is persuasive and clarifies the significance of the experiments and results. I also liked the added sections connecting the work to the scaling-laws literature.

### Weaknesses
- The method proposed by the authors only works in the narrow setting of LLM evaluation _where subsequent task-specific finetuning is guaranteed_. As a result, train-before-test provides little evidence about performance in typical deployment without such fine-tuning. The authors note this in the discussion, but because deployment without task-specific fine-tuning is far more common, this is a substantial limitation and should be stated upfront &mdash; in both the abstract and the introduction &mdash; to avoid readers over-generalizing the scope of the paper.

- The central results of the paper are expected given established facts about the connection between (i) model size/pretraining compute and (ii) downstream performance after task-specific finetuning. The main empirical take-away (supported by the PCA analysis in section 3.4) is that larger/more pretrained models are consistently better across downstream tasks than smaller/less pretrained ones _after_ task-specific finetuning, but not _before_. However, this appears to follow directly from prior work:
  - After task-specific finetuning, model size/pretraining compute (in combination with task-specific features) strongly predict downstream task performance (e.g., [Lin et al., 2024](https://arxiv.org/abs/2402.02314); [Zhang et al., 2024](https://arxiv.org/abs/2402.17193)).
  - Under direct evaluation (i.e., without finetuning), model size/pretraining compute are much less predictive of downstream task performance (e.g., [Magnusson et al., 2024](https://arxiv.org/abs/2312.10523); [Lourie et al., 2025](https://arxiv.org/abs/2507.00885)).

   The paper would be stronger if this connection were made explicit. Given that the authors already refer to the scaling-laws literature, adding this would be relatively easy.

### Questions
You write in lines 97 to 99: "Perplexity benchmarks used to be popular, but fell out of fashion because of the apparent disconnect between perplexity and downstream task performance" &mdash; can you clarify what specifically you base this claim on? In contemporary LLM development, perplexity remains a primary evaluation metric, despite all its limitations.

### Soundness
3

### Presentation
4

### Contribution
3
