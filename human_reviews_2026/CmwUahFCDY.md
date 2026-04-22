# CANMI: Causal Discovery under Nonstationary Missingness Mechanisms

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 6, 4, 4

## Abstract
Causal discovery from time series data is a typical and fundamental problem across various domains. In real-world scenarios, these data often have missing values occurring under different mechanisms, which limits the applicability of most existing approaches, especially when the missing values do not occur randomly due to the influence of other variables. This challenge is further exacerbated when missingness mechanisms also depend on nonstationarity in time series data. In this paper, we propose CANMI, a novel constraint-based approach designed for CAusal discovery under Nonstationary MIssingness mechanisms. Our proposed method can recover the causal structure using only observed data with different missingness mechanisms, including missing not at random (MNAR). Furthermore, we prove the identifiability of the direct causes of missingness and reveal a formula for recovering the data distribution from nonstationary data with missing values. Extensive experiments on both synthetic and real-world datasets demonstrated that our proposed model outperforms state-of-the-art approaches for causal discovery across various evaluation metrics even under substantial missingness. Our source codes are available at https://anonymous.4open.science/r/CANMI-0CDD.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper addresses the challenging problem of causal discovery from time series data with missing values, where the missingness mechanism is nonstationary. They provide theoretical guarantees for identifiability and distribution recovery, and demonstrate empirical performance on synthetic and real-world datasets, claiming superiority over existing baselines.

### Strengths
Strengths
- The paper tackles a practical nonstationary missingness issue, which is important in many real-world applications.

- The experiments on synthetic and real-world datasets suggest that the promising results of CANMI algorithm under the nonstationary missing data.

### Weaknesses
Weaknesses

- The main issue is that the contribution of this work is somewhat limited. The methodological and theoretical foundation of the work appears to be a relatively straightforward extension of MVPC [1]. Moreover, the main contribution of the nonstationary missingness mechanism is simply solved by introducing a time index, which, however, is only a direct application of [2].

- Furthermore, the definition of nonstationary that is used in this work is not formally defined. One typical type of nonstationary data should be defined on time series data, with, for example, different lengths of time lag and the Granger causality. This vagueness makes it difficult to assess the generalizability and boundaries of the proposed approach.

- The method's reliance on using time as a proxy variable is not sufficiently justified. The paper lacks a clear discussion of the assumptions and conditions under which a simple time index can adequately capture complex, underlying nonstationary processes. For example, how the time index models the time series data with a time lag.

[1] Ruibo Tu, Cheng Zhang, P Ackermann, H Kjellström, and Kun Zhang. Causal discovery in the presence of missing data. In Proceedings of the 22nd International Conference on Artificial Intelligence and Statistics, pp. 1762–1770. PMLR, 2019.

[2] Biwei Huang, Kun Zhang, Jiji Zhang, J Ramsey, Ruben Sanchez-Romero, C Glymour, and B Scholkopf. Causal discovery from heterogeneous/NOnstationary data. Journal of machine learning research, 21(89):89:1–89:53, 2020.

### Questions
See the weaknesses above.

### Soundness
2

### Presentation
2

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
This paper tackles the challenging problem of causal discovery from time series data in the presence of nonstationary missingness mechanisms, including MNAR (Missing Not At Random) cases. The authors propose a constraint-based approach that extends the framework of Tu et al. (2019) by incorporating time information T to handle nonstationarity. The experimental results demonstrate the effectiveness of the proposed method.

### Strengths
1. The paper addresses an important and challenging problem—causal discovery under nonstationary and MNAR settings in time series data. This topic is of broad interest to both the causal discovery and time-series communities.

2. The authors provide a good review of related literature.

3. The paper is well-organized and clearly written.

### Weaknesses
1. The theoretical analysis focuses mainly on the non-missing case (R=0), i.e., when no missing values are present. Proposition 3 appears to closely resemble the result in Qiao et al. (2024), so the novelty of the theoretical part may be limited.

2. Assumption 1 ensures that nonstationarity can be addressed by including T as a variable. It would be helpful to discuss what happens if this assumption is violated—e.g., whether the method becomes biased or fails to identify correct causal directions.

3. The faithfulness assumption is made on the distribution P, rather than on the missing-data distribution. In principle, faithfulness should be discussed with respect to the augmented distribution including missingness indicators. If my understanding is incorrect, clarification would be appreciated.

4. The formal statements of Proposition 2 and Proposition 3 are somewhat unclear in the context of missing data. It would strengthen the paper to explicitly formulate these results using the missingness graph (m-graph) representation and clarify their implications.

Reference:

[1]. Identification of causal structure in the presence of missing data with additive noise model. AAAI 2024

### Questions
See Weaknesses.

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
3

### Summary
This paper tackles an important and underexplored problem in causal discovery—handling nonstationary missingness mechanisms (i.e., missingness depending on both other variables and time variation). The authors propose CANMI, a constraint-based algorithm capable of identifying causal structures using only partially observed time series data, even under MNAR settings. The work provides theoretical guarantees (identifiability, recoverability, and order independence) and validates the method on both synthetic and real-world fMRI datasets.

### Strengths
The problem stated in the paper is interesting and important. The paper provides theoretical guarantees about identifiability.

### Weaknesses
The overall presentation of the paper requires substantial improvement. For instance, the paper currently lacks a **problem setup section**, which should clearly introduce the problem formulation and provide concrete examples of scenarios that fit the proposed framework. In addition, Section 3.1 asserts that nonstationarity and missingness can induce spurious causal relations; it would be clearer and more rigorous to formalize this claim as a **proposition** and illustrate it with specific examples. Furthermore, the **role of Step 1** in the algorithm remains unclear. Once ( P(V) ) can be recovered, the causal structure can, in principle, be identified. According to Theorem 1, it is sufficient to determine the **parent set of each missingness indicator**, suggesting that Steps 2, 4, and 5 alone may suffice for causal recovery.

### Questions
See the weakness above.

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper proposes a constraint-based algorithm for discovering causal relations in time-series data that handles nonstationary mechanisms and randomly missing data by introducing missingness indicators and employing importance resampling. The authors claim it is the first method to jointly address these two challenges in this setting. Experiments on synthetic and real-world datasets show consistently higher accuracy than baseline methods, and the reported runtime is comparable to prior work.

### Strengths
1. The paper is complete, providing both theoretical guarantees and a series of experiments under various settings.

2. Empirically, the algorithm shows higher accuracy than baseline methods over various settings.

3. The work tackles two practical challenges: nonstationarity and missing data, both common in real-world settings.

4. The use of joint-distribution reconstruction and importance resampling to reduce the false positive rate under nonstationarity and missingness is interesting and novel.

### Weaknesses
1. The paper’s main novelty lies in Theorem 1 (Step 4), whereas Propositions 1–3 and Steps 1–3 are relatively standard under Assumption 5 after introducing missing indicators. The central challenge in time series, exacerbated by nonstationarity and missing data, is spurious (false-positive) edges. Therefore, the parts on distribution reconstruction and importance resampling should be explained more clearly and given greater emphasis. For example, Step 2 is not directly useful for causal discovery; it serves as a prerequisite for Step 4 to remove spurious edges. Additionally, is the output of Step 1 used in later steps? (See the Questions section for further points about Step 1.) Given this, an ablation study with and without Step 4 would be valuable, since Step 4 appears to be the key step.

2. The rationale for the chosen baselines is unclear, since many of them are not designed for time series. Why use LiNGAM instead of VAR-LiNGAM, NOTEARS instead of DYNOTEARS, and why omit PCMCI, which is widely used as a time-series causal discovery baseline?

3. While the proposed algorithm can identify variables with time-invariant mechanisms, can it also identify variables whose mechanisms are time-variant correctly? Step 1 is titled Detecting changing causal mechanisms, but it discusses only invariance. Could edges between variables and $T$ also arise from spurious correlations? If the method can detect variables with changing mechanisms, it would be helpful to report the accuracy for this sub-task.

4. It would be better to include an official computational analysis.

Other comment:

Consider adding an assumption that nonstationarity changes effect sizes only, while edge presence remains fixed.

### Questions
1. What is the difference between Step 1 and CD-NOD? Can CD-NOD be directly extended to the missing-data setting by combining distribution reconstruction and importance resampling?

2. Beyond nonstationarity and missing data, spurious edges can occur even in stationary time series without missing data due to autocorrelation. Can Step 4 also address it?

3. After Step 4, are all obtained edges retested on the modified datasets? If spurious edges arise between observed variables and missingness indicators in earlier steps, and hence the indicator’s parent set is Eq.3 is a superset of the true parents, what is the impact?

4. Which conditional independence tests are used in the proposed algorithm, given that $R$ is binary while other variables may be continuous? 

5. For the baselines, do you use all $N$ samples or only complete cases after listwise deletion?

6. The learned causal graph is a summary graph that includes only lag 0 effects. How is the comparison performed when a baseline returns a graph with time lags? What alignment and scoring procedure do you use across lags? Are any baselines used that do not allow contemporaneous edges?

### Soundness
3

### Presentation
2

### Contribution
3
