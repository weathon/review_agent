# Tabular Data: Is Deep Learning All You Need?

- Avg Score: 3.50
- Decision: Reject
- Scores: 8, 2, 2, 2

## Abstract
Tabular data represent one of the most prevalent data formats in applied machine learning, largely because they accommodate a broad spectrum of real-world problems.  
Existing literature has studied many of the shortcomings of neural architectures on tabular data and has repeatedly confirmed the scalability and robustness of gradient-boosted decision trees across varied datasets. However, recent deep learning models have not been subjected to a comprehensive evaluation under conditions that allow for a fair comparison with existing classical approaches. This situation motivates an investigation into whether recent deep-learning paradigms outperform classical ML methods on tabular data. 
Our survey fills this gap by benchmarking twenty state-of-the-art methods, spanning neural networks, classical ML and AutoML techniques. Our empirical results over 68 diverse classification datasets from a well-established benchmark indicate a paradigm shift, where Deep Learning methods outperform classical approaches.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This is a rigorous, large-scale empirical study asking whether modern deep learning now consistently outperforms classical methods on tabular classification. The authors benchmark 17 methods (foundation models, dataset-specific NNs, GBDTs, and AutoML) across 68 OpenMLCC18 datasets with nested 10-fold CV, model-based HPO (Optuna/TPE; up to 100 trials or 23h), and a refit-after-HPO protocol. Headline results: deep learning—especially in-context foundation models (TabICL, TabPFNv2) and the MLP ensemble TabM—dominates GBDTs overall; refitting after HPO improves ranks and can reshuffle winners; and DL wins across data-regimes, including “small data,” contrary to long-standing folklore. Code is released for reproducibility.

### Strengths
The nested CV + model-based HPO + refitting design is stronger than prior surveys relying on random search and no refit. The authors explicitly motivate these choices and quantify their impact.  

Inclusion of both in-context tabular FMs and strong MLP ensembles (TabM), alongside AutoML, makes conclusions relevant to practice in 2025.  

Rank distributions, one-vs-one win-rates, CD diagrams, and regime plots tell a consistent story (TabICL/TabPFNv2 ≳ TabM ≳ CatBoost/XGBoost).  

Actionable insights: (i) Refitting after HPO improves results and may change leaderboards; (ii) HPO helps several methods materially (e.g., XGBoost, XTab), with detailed hyperparameter importance analyses.  

Contrary to the common belief that trees rule small tables, TabICL/TabPFNv2 (and TabM) are highly competitive, often surpassing CatBoost/LightGBM.

### Weaknesses
Many plots focus on ranks/win-rates. Please complement with absolute ROC-AUC deltas (mean/median ± CIs) and per-dataset paired tests to quantify practical margins.  

You note ~8M evaluations; please add wall-clock/energy summaries and cost-normalized leaderboards (e.g., AUC per hour) to contextualize results for practitioners.

Some models use bespoke preprocessing; batch size is heuristic due to memory. A short ablation on preprocessing/batch sensitivity for a few methods would strengthen fairness claims.

### Questions
Add tables with median ΔAUC vs top GBDT per dataset family (small/medium/large), with 95% CIs.  
Provide cost-performance plots (AUC vs GPU-hours) and a practitioner-oriented “best under X hours” guide.  
Include a compact regression panel or, at minimum, temper the title/claims to “classification.”

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper presents a survey and benchmarking study of ML and deep learning methods on a diverse collection of tabular datasets. It also offers useful insights into the role of hyperparameter optimization and the cost-performance trade-offs between different model families. However, the main weakness is its novelty: the recent TabArena benchmark [1] already provides an extensive, reproducible, and **continuously maintained** benchmarking effort. This raises questions about how the present work advances the state of knowledge beyond TabArena.



References:
[1] Erickson, Nick, Lennart Purucker, Andrej Tschalzev, David Holzmüller, Prateek Mutalik Desai, David Salinas, and Frank Hutter. "Tabarena: A living benchmark for machine learning on tabular data." arXiv preprint arXiv:2506.16791 (2025).

### Strengths
- The manuscript addresses a central challenge in tabular learning e.g. fairly benchmarking ML and DL methods on tabular datasets
- It provides valuable insights for practitioners, particularly regarding training strategies and hyperparameter sensitivity.
- The paper is clearly written and easy to follow.

### Weaknesses
The primary weakness is limited novelty. The TabArena work [1] already provides: (1) a large-scale, reproducible benchmarking ecosystem, (2) a live leaderboard that can be continuously updated, (3) a carefully curated dataset collection, and (4) strong baselines with advanced evaluation protocols. Moreover, TabArena reports similar empirical conclusions—for example, that DL methods can outperform classical methods in certain regimes. As a result, it is unclear what unique contribution this paper makes beyond prior work.

### Questions
I have one main question: What are advantages of the presented work over the TabArena? 

Also, If the intention is to study a simpler or more controlled benchmarking setting than TabArena, this should be explicitly justified, along with the scientific insight that such a restriction reveals.

I'm ready to update my rating depending on the authors response.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper performs an extensive and thourough comparison of the recent tabural ML approaches. The authors demonstrate that for classification tasks DNN-based methods outperform GBDTs on the benchmark of classification problems.

### Strengths
The fact that refitting (after performing hyperparameter optimization) is more beneficial for GBDTs than for DNNs is interesting and new to me, at least I have never met it in the literature.

### Weaknesses
(1) The paper investigates only classification problems and it is not clear from the title/abstract/contributions bullet list. The final claim in the abstract can be misleading, for instance, "deep learning methods outperform classical approaches" can be false for regression, see the recent TabArena leaderboard for regression.

(2) The chosen wording can also be misleading. For instance, the claim "nonfinetuned foundation models outperform fine-tuned ones" can be unclear, since rigorously speaking, meta-learned foundational models can be finetuned (e.g. see "On finetuning tabular foundation models", Rubachev et al.) and then the authors' claim is false.

(3) The authors do not include the recent non-parametric models (TabR, ModernNCA) in their evaluation, despite these models being strong players in the field. For instance, TabICL paper demonstrated that TabR and ModernNCA can outperform meta-learned models. I believe that in such kind of papers all the recent models should be used.

(4) In my opinion, most of the news from this submission are already known by the people in the field. For instance, TabM paper already reported the advantage of DNNs over GBDT, as well as TabICL and TabArena papers.

### Questions
I do not have any specific questions for the rebuttal, if the authors will address by concerns from the weaknesses section, I will appreciate that.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper re-examines the question of deep learning vs decision trees on tabular data. It uses the OpenML-CC18 benchmark suite and proposes it's own experimental protocol (with main difference from prior work being refitting). The paper compares AutoML system (in autogluon), in-context learning (ICL) based tabular foundation models (in TabPFN(v1, v2) and TabICL), parametirc transformer-based foundation models that require finetuning (XTab, TPBERTa, CARTE), GBDT and neural network baselines (in TabM, RealMLP, FT-Transformer, SAINT, ResNet and MLP). The paper core message seems to be the highlight of paradigm shift, where present-day tabular DL models outperform GBDTs. Mostly through TabM and ICL-based foundation models. The paper also makes multiple finer observations like ICL-based foundation models being superior to finetuning-based parametric ones or looking at the dataset size win-rate profiles and model hyperparameter sensitivity.

### Strengths
The first (broader) strength and argument for the paper is in it's ovearall message and survey-ish nature. The field of Tabular Deep Learning did progress in recent years and the paper manages to convey this message. I think it may be important for the broader DL community to know about the subfield advances.

Another strong aspect (much less important in my view) is in more subtle findings that are novel:
- RealMLP performance seems to be much less good compared to it's resulst on TabArena - this is interesting and warrants at least some investigation? Does the setup difference matter this much?
- XTab and TPBERTa lacking behind ICL-based foundation models - this result is important to set the record straight (but may also warrant some explanation of why this is the case, compared to the success in the original papers)
- An observation that small datasets is where the ICL-based foundation models have more wins
- Demonstration that refitting provides some improvements

### Weaknesses
The core weakness of this work is in lacking the field context. By lacking context I mean that the paper for it's main goal (that seems to be conveying the message of progress in DL for tabular data), misses a bit on where the field is actually at.

First, I think that recent focus on dataset quality in benchmarks brought up in recent work ([Erickson et al.](https://arxiv.org/abs/2506.16791), [Rubachev et al.](https://arxiv.org/abs/2406.19380), [Tschalzev et al.](https://arxiv.org/abs/2503.09159) should be discussed and taken into account in a new "benchmark"/"revisiting the state of X" paper.

Second, It is very hard to discount the existence of TabArena (([Erickson et al.](https://arxiv.org/abs/2506.16791)), which is not discussed in the current paper. In my view TabArena is a step in the right direction for the field in general. There is a certain blueprint to revisiting DL for tabular data publications (e.g. [Tabzilla](https://arxiv.org/abs/2305.02997), [TALENT](https://arxiv.org/abs/2407.00956), [MultiTab](https://arxiv.org/abs/2505.14312), and I'm probably missing some more, but you see the point) which is roughly take some dataset suite, take some recent models, compare. I believe that it is wastefull to reinvent the evaluation over and over again each year. From my point of view the field is slowly growing past that blueprint: more downstream task relevant benchmarks being introduced (like TabReD [Rubachev et al.](https://arxiv.org/abs/2406.19380) - covering Industry ML use-cases or [Barkov et al.](https://arxiv.org/abs/2508.09888) covering digital soil mapping applications, and I think we should have much more of that kind of work). With all the more specific benchmarks, TabArena stands for the general MMLU/GLUE-like proxy for researcher fast iteration.

Furthermore, the present paper study misses some important baselines in ModernNCA, TabR and the more recent LimiX tabular foundation model. In contrast, TabArena has the first two (covering a paradigm important in modern tabular deep learning), and the LimiX (which is recent and may be hard to ask to put in an ICLR submission) is on the way there https://github.com/autogluon/tabarena/pull/208

To circle back and summarize my argument: I think that the paper may be important to share with the broader DL community at ICLR, but as I outlined above, it feels a bit disconnected from many recent developments in the field of tabular deep learning. 

There are other strenghts that are related to more specific findings, but these lack deeper analysis and provide limited insight or little to no-explanation.

### Questions
I am very skeptical that it is possible to address the core weakness outlined above, but I believe that I can be swayed by deeper analysis, specifically in two of the following areas:

- Why does RealMLP performance differ in your protocol and in tabarena?
- Why do non-ICL foundation models fail?
- A  more  in-depth study of dataset-size and ICL models dominance.

### Soundness
3

### Presentation
3

### Contribution
1
