# Beyond Match Maximization and Fairness: Retention-Optimized Two-Sided Matching

- Decision: Accept (Poster)
- Scores: 4, 6, 4, 6

## Abstract
On two-sided matching platforms such as online dating and recruiting, recommendation algorithms often aim to maximize the total number of matches. However, this objective creates an imbalance, where some users receive far too many matches while many others receive very few and eventually abandon the platform. Retaining users is crucial for many platforms, such as those that depend heavily on subscriptions. Some may use fairness objectives to solve the problem of match maximization. However, fairness in itself is not the ultimate objective for many platforms, as users do not suddenly reward the platform simply because exposure is equalized. In practice, where user retention is often the ultimate goal, casually relying on fairness will leave the optimization of retention up to luck.

In this work, instead of maximizing matches or axiomatically defining fairness, we formally define the new problem setting of maximizing user retention in two-sided matching platforms. To this end, we introduce a dynamic learning-to-rank (LTR) algorithm called Matching for Retention (MRet). Unlike conventional algorithms for two-sided matching, our approach models user retention by learning personalized retention curves from each user’s profile and interaction history. Based on these curves, MRet dynamically adapts recommendations by jointly considering the retention gains of both the user receiving recommendations and those who are being recommended, so that limited matching opportunities can be allocated where they most improve overall retention. Naturally but importantly, empirical evaluations on synthetic and real-world datasets from a major online dating platform show that MRet achieves higher user retention, since conventional methods optimize matches or fairness rather than retention.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
In this paper, the authors studied a new paradigm for two-sided matching platforms that optimizes user retention rather than traditional objectives like match maximization or fairness. The authors proposed MRet (Matching for Retention), a dynamic learning-to-rank algorithm that learns personalized retention functions for each user. MRet then dynamically recommends users by maximizing the total expected retention gains for both the receiving and recommended users. The authors conducted numerical experiments on both synthetic and real-world data from a major online dating platform, demonstrating the efficacy of their method.

### Strengths
- I think the model studied in this paper is fairly novel and has practical relevance, as user retention is indeed aligned with real-world business metrics.
- The theoretical results appear sound, where the authors derive an nice relaxation of an NP-hard objective via concavity assumptions. 
- The authors conducted numerical experiments on both synthetic and real-world data and provided very comprehensive analysis. In particular, the discussion provided nice insights into how fairness-centric methods differ from retention-centric approaches.

### Weaknesses
- The framework assumes access to the match probability $r(x,y)$, whereas in practice this must be estimated online. Although the paper acknowledges this limitation, it does not empirically assess robustness to estimation errors. Such errors could substantially degrade performance, and their impact deserves further investigation.
- The real-world validation is relatively limited, as the experiment uses a sampled subset of 1k × 1k users with imputed match probabilities rather than a fully deployed system. This setup demonstrates feasibility but does not provide evidence of live or large-scale performance.
- In this paper, fairness is presented primarily as a comparison to retention, but in practice I think it remains an important consideration for many platforms. Could fairness be incorporated alongside retention as a secondary objective within this framework?

### Questions
Related to the real-world experiments, I also have a conceptual question: on dating platforms, user retention and user satisfaction may not always align. For instance, if matching is highly effective, satisfied users may form lasting relationships and naturally stop using the platform. In such cases, is retention truly the right optimization objective? It would be valuable for the authors to discuss how platforms might balance retention with user success or satisfaction, and whether MRet can accommodate such trade-offs.

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
3

### Summary
This paper departs from the classic goals of two-sided matching (e.g., stable matchings, utility or welfare maximization, fairness constraints) by proposing a new objective: user‐retention (or platform‐driven “stickiness”) in a two-sided matchmaking setting (e.g., online dating, ride-sharing, job matching). The authors argue that match count or immediate welfare may not correlate strongly with long‐term retention, and instead propose modeling a personalized retention curve for each match pair (or each agent × partner type) that estimates the probability of continued engagement over time. They then formulate a matching/allocating algorithm, MRet, which assigns scarce match opportunities not simply to maximize immediate matching or fairness but to maximize expected retention gain subject to capacity constraints. Empirically, they validate the retention‐curve learning (from observational data) and test MRet on a large real‐world dataset (from a dating-platform) plus synthetic simulations. The results suggest that MRet can increase overall retention by focusing matches where retention impact is large, and still provide acceptable fairness outcomes (e.g., representation across groups).

### Strengths
1-  Many matching/recommender systems (dating, jobs, rides, tutors) care about retention, not just initial matching. Framing retention‐maximization in two‐sided matching is novel in the ML/optimization space.

2- Learning per‐agent or per‐pair retention probability over time is a creative step that bridges predictive modeling with matching allocation.

3- The shift from maximizing number of matches/fairness to maximizing expected retention under capacity constraints is clean and actionable.

4- Real‐world data from a large-scale platform, showing retention lift and reasonably good fairness outcomes, is a strong edge for an applied paper.

5- The authors do not ignore fairness; they include group‐level analyses and discuss retention trade-offs across demographic slices.

### Weaknesses
1- The paper lacks formal results on the complexity of the retention‐objective matching problem, or approximation bounds/algorithms for large‐scale graphs. If the objective is non-linear or non-additive, this becomes non-trivial. The submission would be stronger if it clearly stated whether the problem is polynomially solvable (e.g., weighted bipartite matching) or NP‐hard and if so, provided a provably efficient approximation.

2- Learning retention curves is central. However, the paper provides limited diagnostics of how accurate those curves are (prediction error, cross‐validation, time‐shift evaluation). Also, the domain dependence (single platform dataset) raises questions: how does the method generalize to new user cohorts, or across platforms?

3- While fairness is discussed, the core algorithm (MRet) optimizes retention. If certain groups inherently have lower retention predictions, the algorithm may preferentially allocate to higher‐retention groups, exacerbating representation/fairness issues. More explicit trade‐off analysis (worst‐group retention, disparity curves) would strengthen the fairness dimension.

4- The evaluation would benefit from stronger comparative baselines beyond standard matching: for example, match‐volume maximization with fairness constraints, heuristic retention prioritization without modeling, and analyses of small vs. large capacity constraints. Also, ablation studies (e.g., retention modeling on/off, fairness constraints on/off) would help attribute gains.

5- It is unclear how the approach scales to many‐to‐many matching or large quotas, or domains where retention is less measurable or delayed. Some simulation/transfer experiments would help.

### Questions
1- Is the allocation problem solved as a linear assignment with weights = expected retention gains, or is it more complex (e.g., non-additive, dependent pairs)? If the latter, what solver is used, and what is the worst‐case runtime?

2- What features are used to predict retention? How was the model validated (train/dev/test splits, cross‐validation, hold‐out platform cohort)? What is the typical prediction error (e.g., RMSE, calibration by bins)?

3- How sensitive are results to the capacity constraint (e.g., limited matches per period)? Did you vary the quota regime and show retention gains scale?

4- Given that retention predictions may correlate with demographic/time-on‐platform features, how do you ensure that optimizing retention doesn’t disproportionately disadvantage historically marginalised groups? Could you offer fairness constraints (min‐quota, parity) and show corresponding retention/representation curves?

5- If user cohorts shift or retention distributions change over time (e.g., seasonality, platform evolution), how robust is the learned retention model? Did you evaluate transfer across time windows or hold‐back users?

6- Have you applied MRet beyond the dating platform domain (e.g., job-matching, ride‐sharing) or simulated different domains to test generality?

7- Did you compare to simpler heuristics (e.g., match first, then sort by predicted retention gain greedily) rather than full model + allocation? And what is the incremental benefit of the full pipeline?

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
This paper reframes online two-sided matching around user retention rather than match count or fairness. Under the assumption that per-user retention as a function of cumulative matches is monotone and concave, the authors apply a Jensen/linear lower bound to derive an additive, argsortable score that accounts for both receiver- and candidate-side marginal retention gains. Experiments on synthetic data and a small real dataset report retention improvements.

### Strengths
- Elevates long-term user retention over proxy objectives like match count or exposure parity.
- An interesting setting that explicitly models both receiver and candidate utilities to enhance long-term retention.
- Clean derivation and standard components make the method easy to code and to reproduce end-to-end.

### Weaknesses
1. **Motivation/positioning is unclear. Fairness appears as a “solution” without a precise link to retention.**

The paper treats fairness methods as peers to retention optimization, but never clarifies the key retention concepts (e.g., “satisfaction thresholds,” “marginal retention slope,” “early low-match churn”) nor how these relate to fairness (guardrail vs competing target).

2. **Related work is late and under-covered, blurring the contribution boundary.**

Retention optimization and reciprocal-fairness are direct neighbors; coverage is thin and appears too late. The following are some examples:
> - Wang Y, Sun P, Ma W, et al. Intersectional two-sided fairness in recommendation[C]//Proceedings of the ACM Web Conference 2024. 2024: 3609-3620.
> - Biswas A, Patro G K, Ganguly N, et al. Toward fair recommendation in two-sided platforms[J]. ACM Transactions on the Web (TWEB), 2021, 16(2): 1-34.

3. **Target misaligned. Examples/experiments focus on popularity bias, but the primary comparator optimizes disparity**

Most setups reveal a ‘rich-get-richer’ dynamic, yet the fairness baseline optimizes exposure disparity rather than marginal retention for low-match users and does not directly address popularity bias. This misalignment structurally favors MRet.

4. **Baselines are limited and their selection lacks rationale.**

Several cited works are not used as baselines; Max Match/FairCo’s origin/implementation/tuning is under-specified.

5. **Real-world setting is narrow and assumption-heavy**

Small sample sizes(1000 x 1000), heavy reliance on missing-value imputation, coarse retention labels (e.g., next-month login), and no off-policy or online evidence.

### Questions
1. In dating, a “successful match” often triggers intentional departure. Did you distinguish success-exit from disengagement churn?
2. Do you model popularity as time-varying $p_u(t)$ (recency, seasonality, inactivity decay)? Please provide dynamics of popularity over time and test whether your method adapts when popularity drifts.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This manuscript proposes a novel problem setting focused on maximizing user retention in two-sided matching platforms, along with a dynamic learning-to-rank algorithm (MRet) that models personalized retention curves to optimize recommendations. The work addresses a critical gap in existing research—where match maximization causes user imbalance and fairness objectives fail to directly align with platform sustainability—and validates the approach through rigorous synthetic and real-world experiments on a large-scale online dating platform. The paper is well-structured, logically coherent, and its core contribution directly responds to practical needs of two-sided platforms dependent on user retention.

### Strengths
1. The paper breaks away from the dominant paradigms of match maximization and axiomatic fairness, instead centering on user retention as the core objective—an issue directly tied to platform revenue and sustainability (e.g., subscription-based models). This choice is well-justified by empirical evidence (Figure 1) showing that users with fewer matches have significantly higher churn rates, addressing a real pain point ignored by prior work.
2. MRet’s focus on dual retention gains (both the recommending and recommended users) is a key innovation. Unlike conventional methods that optimize for one-sided utility, this two-sided consideration aligns with the reciprocal nature of matching platforms. The use of personalized retention curves, derived from user profiles and interaction history, ensures recommendations are tailored to individual retention needs rather than generic fairness or match probability.
3. Experiments not only demonstrate higher retention rates but also explore edge cases (e.g., varying popularity skew, sparse match data, different exposure probabilities), proving MRet’s robustness. The analysis of match distribution (Figure 5) further clarifies why fairness-based methods fail—they ignore individual satisfactory match counts—strengthening the paper’s core argument.

### Weaknesses
1. The paper assumes match probabilities (r(x,y)) are either known or estimated upstream, but provides no guidance on how to integrate this estimation into the MRet framework. In practice, inaccurate match probability estimates could degrade retention performance, and the authors should discuss how MRet interacts with common estimation methods (e.g., collaborative filtering) or mitigate estimation errors.
2. Although the paper mentions potential applications in recruitment, experiments are limited to online dating. Job matching has distinct characteristics (e.g., asymmetric priorities between employers and job seekers, longer decision cycles) that may affect retention dynamics. Validating MRet on a recruitment dataset would strengthen its generalizability.
3. The paper focuses on retention rates as the key metric but does not address whether MRet impacts user satisfaction beyond continued login. For example, do users receiving "retention-optimized" matches report higher engagement quality (e.g., longer conversations, successful relationships in dating)? Supplementing with qualitative or secondary engagement metrics would enrich the evaluation.

### Questions
Note: Here are some questions I have after reading the paper. Since I am not very familiar with this field, the authors may choose to answer selectively, focusing on issues related to the paper’s contributions. I will also take into account comments from other reviewers when forming my final evaluation.

1. The retention function f(x,m) is assumed to be concave, but the paper notes MRet performs well on non-concave functions in real-world data. What specific properties of non-concave retention functions does MRet leverage to maintain effectiveness, and are there scenarios where non-concavity would significantly degrade performance?
2. How does MRet handle users with limited interaction history (e.g., new users) who lack sufficient data to train personalized retention curves? Is a fallback strategy (e.g., cluster-based retention curves) used, and how does this affect overall retention?
3. The paper mentions platform business models may prioritize one user side (e.g., paying male users in dating). How would MRet be modified to incorporate weighted retention objectives (e.g., 60% weight on male retention, 40% on female), and what trade-offs would arise from such prioritization?
4. In the real-world experiment, missing match probabilities are imputed using ALS. How sensitive is MRet’s performance to imputation errors? Would alternative imputation methods (e.g., matrix factorization with side information) yield better retention results?

### Soundness
3

### Presentation
3

### Contribution
3
