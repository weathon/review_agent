# Locally Adaptive Multi-Objective Learning

- Decision: Reject
- Scores: 4, 4, 4, 2

## Abstract
We consider the general problem of learning a predictor that satisfies multiple objectives of interest simultaneously. We work in an online setting where the data distribution can change arbitrarily over time. Here, multi-objective learning captures many common targets such as online calibration, regret, and multiaccuracy. In the online setting, common approaches to this problem that minimize the set of objectives over the entire time horizon can fail to adapt to distribution shifts. Previous work has tried to alleviate this problem by incorporating additional objectives that target local guarantees over contiguous subintervals. However, empirical evaluations of the performance of this proposal in practice are sparse. In this article, we consider an alternative procedure that achieves local adaptivity by replacing one part of the multi-objective learning method with an adaptive online algorithm. Empirical evaluations on datasets from energy forecasting and algorithmic fairness show that our proposed method improves upon existing proposals and achieves unbiased predictions over subgroups, while remaining robust under distribution shift.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper investigates online multi-objective learning, aiming to learn a predictor that satisfies multiple objectivtes of interest simultaneously. The authors work in an online setting where the
data distribution can change arbitrarily over time. In the online setting, existing approaches to this problem that minimize the set of objectives over the entire time horizon can fail to adapt to distribution shifts. This paper proposes algorithms that guarantee small error for all objectives
over any local time interval of a given width.

### Strengths
The topic studied in this paper is interesting, and the authors have provided a thorough discussion of the related work.

### Weaknesses
I have outlined the main weaknesses of this paper in the Question section.

### Questions
According to the proposed Algorithm 1, I believe that this work does not actually address changing environments as claimed by the authors. In other words, referring to Algorithm 1 as an adaptive algorithm is not appropriate. Specifically, the authors describe the notion of strong adaptivity in the paper. It is worth noting that a strongly adaptive algorithm (Daniely et al., 2015) does not require prior knowledge of interval information and can automatically adapt to all unknown intervals.

However, the proposed Algorithm 1 requires interval information to set $\eta$. In my view, this is unreasonable, as it implies that once the environment changes, the authors would need to manually adjust the step size of the algorithm. 

Please clarify this point.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes a locally adaptive algorithm for online multi-objective learning, aiming to ensure small error for all objectives over any local time interval, rather than minimizing a time-averaged global objective. The algorithm builds on exponential-weights updates and Fixed Share ideas to handle temporal adaptivity. The work focuses primarily on the multiaccuracy problem, where the goal is to learn unbiased predictors under covariate shift, while preserving performance relative to a base predictor through an added regret objective. The authors also discuss extensions to multicalibration, multi-group learning, and omniprediction. Empirical results on electricity demand forecasting (GEFCom2014-L) and COMPAS recidivism prediction suggest improved “local adaptivity” compared to non-adaptive baselines. The authors claim their work bridges a gap between theoretical multi-objective learning and empirical application.

### Strengths
1. The motivation is strong as local adaptivity under distribution shift is a meaningful and contemporary topic, especially for fairness and online prediction settings. 
2. Its attempt to bring online multiobjective theory closer to empirical evaluation is welcome in a literature often dominated by proofs. 
3. The paper is well-written and logically structured, with clear definitions, algorithms, and theorems that make the technical content easy to follow.

### Weaknesses
1. The main algorithm is only a minor variation of existing methods such as the exponential-weights and Fixed-Share frameworks from prior work (e.g., Lee et al., 2022; Gradu et al., 2023). The theoretical results follow known patterns without introducing new proof techniques or bounds. 
2. The claimed “multi-objective” setup is misleading since all objectives depend on the same residual term. It can hardly be generalized to multi-objective setting with competing objectives. 
3. The experiments are limited to two small datasets, with mostly qualitative plots and no statistical analysis or ablations on key hyperparameters. 
4. The connection between “local adaptivity” and true robustness to arbitrary distribution shift is not theoretically justified.

### Questions
1. In line 121 $\ell:[0,1]\times X \times Y \rightarrow [-1,1]$ takes $p_t$ as the first input, while in line 119, it says $p_t\in Y=[a,b]$. Then why it is  [0,1] here? And, in Definition 1, it says $L = \{\ell:[0,1]\times X \times Y \rightarrow [0,1]\}$ why the output is $[0,1]$ rather than $[-1,1]$ now? And is Definition 1 a standard definition for online multi-objective learning? 
2.  Is Definition 1 a standard definition for online multi-objective learning? And can you provide justification or references for Defs 2 and 3 as well? 
3. How sensitive are results to the interval width and exploration rate? 
4. Why the objectives can be simply renormalized to [−1,1] rather than balanced by variance or importance? 
5. Can the algorithm handle noncontiguous or abrupt distribution shifts, and how does it compare to other adaptive fairness methods?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper considers online minimax multiobjective optimization, and proposes an algorithm that offers locally adaptive regret guarantees (which hold not only marginally but also over windows of some size). The paper develops theoretical guarantees for local adaptivity, that combine the insights from Lee et al (2022) with control-inspired adaptive methods of Gradu et al (2023). It then tests the proposed approach on two datasets, a fairness one and an electric one, and finds a substantial empirical advantage of the proposed method over the previous algorithm, i.a. when it is geared towards optimizing for multiaccuracy + regret.

### Strengths
The proposed method appears to be an empirically advantageous one compared to prior work, naturally iterating on the original work of Lee et al. The algorithm remains clean and interpretable. Several practical tweaks, such as adding in regret constraints and using the step size heuristic from the adaptive conformal literature, appear to be fruitful at enhancing performance.

Furthermore, some of the empirical findings (i.e. the faster convergence, to multi accuracy, of multiaccuracy-only approaches compared to multicalibration ones) add to the recently accumulated evidence on the theory/empirical fronts that there are interesting separations between multiaccuracy and multicalibration notions. 

As another point to put this work into perspective, the multivalidity / multicalibration literature has long been developed with mostly theoretical guarantees in mind, and a variety of initially developed methods have suffered from various performance issues (such as overfitting in the batch case, and slow empirical convergence in the online case), which have only recently received attention from the empirical angle. These recent works have focused on the batch case, primarily, and so the development of an empirically performant online method is a welcome and timely development from that broader viewpoint.

### Weaknesses
The proposed method, while a rigorous one and an apparent improvement over the state of the art, appears to be a somewhat modest development on the theory front. I have read the proofs (and found them to be correct), but have not found that many surprising insights. In a way, it is natural with respect to the online multiobjective framework that there are improvements of the adaptive regret style (indeed, these can just be enforced via separate constraints, ignoring the worse theoretical/empirical guarantees that this would obtain); what would have been more exciting for me to see would be strong dynamic regret results: indeed, short of the well-known blackbox adaptive <-> dynamic regret conversions, it’s not quite clear to me how to naturally embed dynamic guarantees into an explicitly minimax-solving framework like the one studied here.

Moreover, I think the experiments could also be expanded on in a fruitful manner, as I was left with some questions on the “practical” front; here are a few points. First, note that multicalibration is separated from multiaccuracy via the tunable parameter m (number of buckets), such that when there are no buckets we get multiaccuracy. So in terms of the multiaccuracy/multicalibration separation, it would be very useful to show some plots where less and less multicalibration constraints are imposed, and seeing how that reflects on the obtained multiaccuracy convergence rates; seeing this at least for small bucket counts (2, 3, 5, etc) would be informative. Second, while the learning rate choice as in Gibbs and Candes is a solid one, carefully ablating against it would be important to see to what extent that choice matters compared to the locally adaptive updates themselves. Third, given the partly theoretical nature of the main contribution, it would be good to see some synthetic experiments besides real-world ones, e.g. exploring the speed of data drift impacting the new and the old algorithms.

### Questions
First, please see the empirical questions above. Second, I have not found this in the paper in long/explicit form, but it would be good, exposition-wise, to explain to the readers how local adaptivity can be obtained in the existing original framework via separate constraints, namely, by just instantiating constraints (that come and go every length-of-window many rounds) for each of the currently active windows; explaining and perhaps even showcasing via a quick experiment why that would be less practical.

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
The paper presents adaptive algorithms for multi-objective learning to guarantee controlled error across a set of objectives over local time intervals in sequential problems. The paper also does experimental studies on electric load forecasting and compas dataset, and compares with prior work from Lee et al. (2022) and other strong adaptive variants proposed in the literature.

### Strengths
The paper has some merit as it attempts to initiate empirical studies on this line of work. However, I do have some concerns as noted below.

### Weaknesses
For me it is not clear what is the main contribution of the paper. The introduction says: the majority of algorithms proposed in the literature are not adaptive, but then then around lines 247 mentions strong adaptivity has been previously studied before, and the related extensions have been studied for multi-calibration. In experimental section, comparison against strong adaptivity is done, but I'd expect some more insightful resolution on why that does not work in practice as the experiment section claim. Maybe this needs to be studied a bit more. 

Additionally, I'm bit confused about the setting of the paper. I assume the related work in this line assumes an adversarial form of nature, i.e. there is no distribution, and the outcome is determined by an adversarial nature. In that sense, what does it mean to have distribution shift. The example given at the beginning of Section 2.2, I believe the online algorithms proposed in the literature would be able to cope with it, although there'd be some delay for $t > T/2$. I get the motivation of the work that vanishing error (multi-accuracy or multi-calibration error) in the limit does not say much about error in local intervals, which makes up for a good motivation for the paper, but distribution shift argument is something I'd appreciate clarification about. 

My reading of the paper says that the paper aims to provide experimental evidence or is intended as a practical study of multi-objective algorithms in the online setting with adaptiveness as a major focus, however the current setup seems lacking as I could not learn why prior algorithms (multi-calibration and strong adaptive variant of it) does not work, and what does the algorithm proposed in the paper contributes that gives better performance. I feel like this requires more study.

My preliminary impression of the paper is that while the paper has good potential as a rather insightful contribution in this literature, the paper requires more thorough work (experimental and maybe theoretical to compare against prior work) to add value to the literature. And I hope authors will consider this.

### Questions
See above

### Soundness
2

### Presentation
2

### Contribution
2
