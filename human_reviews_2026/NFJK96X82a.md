# Rethinking Reward Models for Multi-Domain Test-Time Scaling

- Avg Score: 6.00
- Decision: Reject
- Scores: 6, 6, 4, 8

## Abstract
The reliability of large language models (LLMs) during test-time scaling is often assessed with *external verifiers* or *reward models* that distinguish correct reasoning from flawed logic. Prior work generally assumes that process reward models (PRMs), which score every intermediate reasoning step, outperform outcome reward models (ORMs) that assess only the final answer. This view is based mainly on evidence from narrow, math-adjacent domains. We present the first unified evaluation of four reward model variants, discriminative ORM and PRM (dORM, dPRM) and generative ORM and PRM (gORM, gPRM), across 14 diverse domains. Contrary to conventional wisdom, we find that (i) dORM performs on par with dPRM, (ii) gPRM is not competitive, and (iii) overall, gORM is the most robust, yielding significant and consistent gains across every tested domain. We attribute this to PRM-style stepwise scoring, which inherits label noise from LLM auto-labeling and has difficulty evaluating long reasoning trajectories, including those involving self-correcting reasoning. Our theoretical analysis shows that step-wise aggregation compounds errors as reasoning length grows, and our empirical observations confirm this effect. These findings challenge the prevailing assumption that fine-grained supervision is always better and support generative outcome verification for multi-domain deployment. We publicly release our code, datasets, and checkpoints at this [anonymous repository](https://anonymous.4open.science/r/iclr2026-5078-7744) to facilitate future research in multi-domain settings.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper compares four types of external verifiers: dPRM, dORM, gPRM, and gORM. (d for discriminative, and g for generative)
Findings are divided by domain and cot-type (short or long) and summarized on the last page of the paper. 

In general, the paper suggests ORM-type verifiers to show the best (most robust performance), and to prefer gORM over dORM when compute allows. They hypothesize this is likely because PRMs are likely to accumulate errors or are not a suitable match with the current long-reasoning paradigm, where models can self-recover from errors. 

I have mixed feelings about this paper. While, after reading it, I think some of the results are obvious, I understand it's an important part of science for someone to conduct a well-controlled experiment to document and prove such obviousness.

### Strengths
While the paper is not of the type that proposes a new method or dataset, it aims to broaden our understanding of external verifiers. And might be a good reference material for those in the industry who have to choose which verifier to use in certain scenarios.

The presentation is nice and I would say fancy.

### Weaknesses
(1) I think we can summarize scenarios where we need external verifiers into two cases. (a) at test-time: where we want to boost the performance of a fixed language model. The paper may be helpful to practitioners who need to choose a generator-verifier combination. One thing that worries me is that the paper highly relies on empirical findings, and I'm not sure whether the same rules will apply to new datasets, domains, training datasets, etc. (b) at train-time: where we use external verifiers to guide training. For instance, see Figure 4 of [1], which provides an experiment on whether their analysis on verifiers can be applied to PPO-style training. The paper lacks analysis in this direction. In a nutshell, while this may appear a bit off-topic for the authors, I think the findings are heavily reliant on empirical observations on the test-time side and lack exploration in the train-time side.


(2) The paper emphasizes that the limitations of past works are being narrowed down to math only. However, I think this paper still lacks exploration in non-math domains. The dataset used to train the verifiers for MMLU-Pro is automatically generated via Llama-3.1-8/70B. Evaluation is done only on MMLU-Pro, a very standardized MCQA benchmark. Especially with the diverse evaluation datasets flooding recently, it would have been better for the authors to collect a diverse set of benchmarks (even if they had to sample it).

(3) An increasing usage of external verifiers is in training where we the community now aims for harder and harder datasets. It would have been interesting to include the results in harder datasets.

### Questions
See weakness.

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper challenges the belief that process reward models (PRMs) are superior to outcome reward models (ORMs). While PRMs excel in narrow math domains, this study conducts the first unified evaluation of four reward model variants (dORM, dPRM, gORM, gPRM) across diverse domains. The key finding is that Generative ORM (gORM) is the most robust and delivers significant and consistent gains across every tested domain. In this multi-domain setting, dORM performs on par with dPRM, and gPRM is not competitive. The authors demonstrate that PRMs fail because their stepwise error aggregation compounds errors as reasoning length grows. PRMs also struggle to evaluate long reasoning chains that involve self-correction ("aha" moments), a common feature in complex, non-math domains.

### Strengths
1. This paper directly challenges the widely held belief that Process Reward Models (PRMs) are superior to Outcome Reward Models (ORMs). It correctly identifies that this consensus was formed from studies on narrow, math-adjacent domains. By conducting the first unified evaluation of four model variants (dORM, dPRM, gORM, gPRM) across diverse domains, it provides a more generalizable finding: gORM (Generative ORM) is the most robust and yields significant and consistent gains across every tested domain.

1. The authors don't just present what happens, but why. It offers a theoretical analysis showing that the log-error lower bound for PRMs (dPRM and gPRM) grows linearly with the length of the reasoning chain ($T$). In contrast, the error bound for ORMs is independent of $T$. This provides a formal explanation for why PRMs fail on the longer, more complex reasoning chains found in multi-domain settings.

1. This paper diagnoses specific, practical failure modes of PRMs that dPRM is highly sensitive to label noise, which is prevalent in multi-domain datasets auto-labeled by LLMs. It also finds that gPRM suffers from a severe shift in its CoT-length distribution because its strict consensus filtering removes too many long CoTs, fatally mismatching its training data to the test set.

### Weaknesses
1. The study's evaluations are confined to tasks with verifiable outcomes (i.e., math and multiple-choice questions). The authors also explicitly state that these findings may not generalize to open-ended generation tasks like dialogue or summarization.

1. The authors do not perform exhaustive tuning, as mentioned in the appendix. This may have led to an unfair comparison. The poor performance of PRM-style models (dPRM, gPRM) on this new, noisy, multi-domain task might be because their inherited hyperparameters were not optimal, while the hyperparameters inherited by the ORM-style models just happened to be more robust.

1. The paper strongly demonstrates that gPRM fails because its strict consensus filtering mechanism leads to a severe lack of long CoT samples in its training data, thereby creating a length distribution shift. This weakness might be caused by gPRM's data collection pipeline, not by the concept of gPRM itself. A gPRM that adopted a different data collection strategy (e.g., less strict filtering or sampling specifically for long CoTs) might still be very competitive.

### Questions
1. The dPRM's failure is attributed to its sensitivity to label noise from the single Llama-3.1-70B annotator. Is this a fundamental flaw of dPRM, or an artifact of this specific annotator's noise profile? If regenerating the process labels with a more advanced model, can the dPRM's performance be recovered with higher-quality process labels?

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
4

### Summary
This paper evaluates several variants of RMs on math and general domains with controlled experiments. By unifying the setups of several previous works, the authors found out that math and multi domain settings might lead to different trends among those RM variants. Then intuitive understandings of these differences were proposed, namely error accumulation along CoT trajectories, label noise, and length distribution shift. Results suggest that PRM might not be always better than ORMs especially in multidomain setup, and provide practical guidelines.

### Strengths
* The paper presents controlled experiments on RM variants to address the debate around them on math and multi domain setup.
* The paper is well written, with clear narratives and good figures. The main message is clearly conveyed with experiments, followed by in-depth analysis.

### Weaknesses
* For math, process reward labels come from PRM 800k, which is typically believed to have good quality. For MMLU-pro, they are fully synthetic, from Llama 3 70b. Potentially the degradation of label quality on MMLU-pro can lead to the result that PRM underperforms ORM. In contrast, the ORM labels of both PRM 800k and MMLU pro should be good. The difference in label quality itself might undermine the validity of the conclusions   
* Although MMLU pro is different from math, they are still both tasks with verifiable rewards. It is still unclear how ORM and PRM compare in a more open ended setup.
* Weak theoretical results. It is well known that PRM error accumulates as the CoT length grows. In this sense, the results are of almost no significance. Besides, this point alone does not establish the advantage of ORM over PRM.

### Questions
* The paper shows gPRM is robust to label noise in the math domain but fails in the multi-domain setting due to a distribution shift. Is this shift purely an artifact of consensus filtering on noisy labels? In other words, would gPRM still fail in a multi-domain setting even if it were trained on perfectly clean, human-annotated process labels (like PRM 800k)?
* The paper states CoTs account for 15.3% of ProcessBench. What is the prevalence of "aha" CoTs in MMLU-Pro? Is it significantly higher, and could this (in addition to CoT length) be a primary factor of the PRM performance collapse?

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
This paper provides a unified evaluation of four reward model variants for test-time scaling—discriminative and generative outcome reward models (dORM, gORM) and process reward models (dPRM, gPRM)—across 14 domains. While prior work favored PRMs (step-level supervision) over ORMs (final-answer supervision), the authors find that this assumption breaks down in multi-domain reasoning. Specifically, gORM consistently outperforms all others, dORM ≈ dPRM, and gPRM underperforms despite prior success in math. The paper attributes these results to compounded stepwise errors and label noise in long or noisy CoTs. Theoretical analysis establishes that PRM log-error grows linearly with reasoning length, while ORM error remains bounded. Empirical results (e.g., Figs. 5–8, 12, 15) confirm this, and the authors provide practical guidelines for model selection.

### Strengths
- This paper is very well written with lots of careful analysis. I appreciate all the figures and they clearly conveys the results. 
- The paper compares four reward model variants across 14 diverse domains (law, biology, philosophy, etc.), using multiple backbones and datasets, and the study the quite diverse and comprehensive.

### Weaknesses
- Limited task diversity in form: All benchmarks are multiple-choice or verifiable; generalization to open-ended reasoning or generation remains untested.

### Questions
n/a

### Soundness
3

### Presentation
4

### Contribution
3
