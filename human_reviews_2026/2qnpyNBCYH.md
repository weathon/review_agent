# Dynamic Dual-Feedback Conformal Inference for Time series Forecasting

- Decision: Reject
- Scores: 4, 4, 6, 2

## Abstract
Quantifying uncertainty in time series forecasting is particularly demanding because sequential data exhibit temporal dependence and are prone to distributional changes. Conformal inference has emerged as a powerful uncertainty quantification approach for evaluating the reliability of predictive models through the construction of prediction sets. Recent advances have introduced online conformal methods that adaptively adjust prediction thresholds through feedback mechanisms. However, the existing feedback mechanism typically relies solely on miscoverage indicators (actual feedback) — whether the true label falls within the interval at each time step — while overlooking the empirical prediction threshold (estimated feedback) that is derived from the oracle conformal method. In this paper, we propose $\textit{Dynamic Dual-feedback Conformal Inference}$ (DDCI), which incorporates a dual-feedback mechanism consisting of $\textit{actual feedback}$ and $\textit{estimated feedback}$. The former drives the primary adjustment of the intervals based on true observations, while the latter dampens excessive expansions or contractions by leveraging empirical thresholds from conformal inference during updates. By balancing these two signals, DDCI achieves more stable and narrower prediction intervals in sequential settings while preserving the coverage validity.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces Dynamic Dual-feedback Conformal Inference (DDCI), a new online conformal prediction method for time series forecasting. DDCI enhances standard adaptive conformal inference by incorporating two feedback signals: actual coverage error and an estimated threshold from conformal inference. This dual-feedback design stabilizes the update process, producing narrower prediction intervals while maintaining target coverage, as validated theoretically and empirically across multiple datasets and forecasting models.

### Strengths
1. The idea of incorporating estimated feedback from conformal inference into the online update process is well-motivated. 
 2. The mathematical proofs are easy to follow and the notations are clear.
 3. The experiment section is well-presented.

### Weaknesses
1. A recurring issue in this line of work, including the present paper, is the treatment of parameters. While this work introduces an additional parameter, a more thoughtful approach to parameter selection would have strengthened the contribution.
 2. Although DDCI shows significant improvements on stock datasets, the gains on the Delhi temperature and electricity demand datasets are less pronounced. This suggests that the method may be more suited to certain types of non-stationarity, and a deeper analysis of its limitations across different time series characteristics would be beneficial.
 3. See questions.

### Questions
1. Although it is reasonable to utilize the estimated threshold to stablize and counterbalance the intervals, is there any theoretical intuition on why choosing the estimated feedback in the form of (5), or theoretical results on the impact of added feedback terms compared to OGD? For example, why not simply take the estimated feedback function as $-|x|/2B*tanh(cx)$? 
2. The experimental results of ECI with AR model seems consistently outperforms DDCI (except for Microsoft data). How do the authors comment about this?
3. Since the main contribution lies in the estimated threshold part, can you provide a more specific way on how to choose $\epsilon$?

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
This paper proposes a new online conformal prediction method by adding an "Estimated Feedback" to the ECI method, which incorporates more information from historical data. The experiment results show the superiority of the proposed method over the existing baselines.

### Strengths
The estimated feedback is new in the online CP, which helps construct more stable and potentially shorter prediction intervals.

### Weaknesses
**1. The new method introduces more tuning parameters.**

Compared with ECI, there is an additional parameter $\epsilon$ in the update rule. According to Figure 3, the size is very sensitive to the choice of $\epsilon$. However, choosing $\epsilon$ is challenging in the online task.

**2. About the role of the estimated feedback**.

The estimated feedback has an opposite sign to the actual feedback, intending to smooth the update. To achieve this goal, we can choose any positive sequence to replace $h(c e_t^* \ )$, e.g., $h(c e_t)$. Why do we use $e_t^*\ $? 

 Also, the statement in Lines 213-214 seems incorrect. Even though $q_t^* \ $ is the true $1-\alpha$ quantile of  $s_t$, the difference $e_t^* = |s_t - q_t^*|$ can be very large. 

Overall, the role of estimated feedback is not well discussed, which makes it like a tuning trick. I suggest the authors provide a fundamental explanation based on a working data model.

### Questions
1. The actual feedback in (4) is not consistent with that in ECI.

2. What if we only use the estimated feedback $h(c e_t^* \ )$ to replace the actual feedback in ECI.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes Dynamic Dual-feedback Conformal Inference (DDCI), a new online conformal prediction method for time series forecasting. DDCI implements the control-theory concept of dual feedback, which incorporates two feedback signals: (1) error / miscoverage - normalized and squashed error magnitude, and (2) a stabilizing signal derived from empirical quantiles of previous scores.

### Strengths
- The paper is clearly written and well organized. I understood the method and the experiment well. 

- The dual-feedback idea is nice and a nice fit for dynamic control of coverage. Some adaptability is sacrificed for stability, but the stability of quantile updates results in more efficient intervals. 

- The tasks used in the experiment section is diverse and the baseline selection is very up-to-date. DDCI achieves coverage and sharpness on all datasets, verifying the theory.

- I think the experiment section is overall quite thoughtful. The natural extension to weighted quantiles (DDCI-Nex) is reasonable and well-explored, and the sensitivity analysis on epsilon is explained clearly.

### Weaknesses
The flip side of spending 3.5 pages on comprehensive experiments is the lack of depth on theoretical discussions. As the dual-feedback concept is clearly inspired by control theory (fast and slow feedback loops), the paper would benefit for more detailed analysis on adaptation speed, coverage stability, computational cost, etc explicitly compared to ACI (pure P control) and Conformal PID. Specifically: 
 
- How is dual-feedback related to PID's three terms (P, I, D)? Could DDCI be expressed as a PID controller with specific tuning?
- Are there advantages to dual-feedback over PID, or vice versa? Under what situations would PID control outperform DDCI and vice versa?
- Is there a specific setup where you can prove that the DDCI has smaller interval than other methods? by what magnitude? what does this magnitude depend on? 
- Can these concepts guide the setting of the hyperparameters such as $\epsilon$?

### Questions
See weakness.

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
The work proposes a conformal method for time series forecasting. The proposed method employs a dual-feedback system, not previously used, to obtain conformal regions that are more efficient than existing baselines.

### Strengths
The update rule to adapt the conformal set provides a flexible mechanism. The experiments demonstrate the efficacy of the proposed methods in the chosen settings.

### Weaknesses
The presentation can be improved. Many times, the terminology just sounds a bit odd, or there is repetition of ideas already mentioned.

The novelty seems limited over the existing methods. 

See "Questions*" below for more

### Questions
1. Line 037: Conformal Inference was introduced before the 2005 book cited here. Consider rephrasing the sentence or citing the earlier work from Vovk.

2. Line 072: Upper Limitation? Maybe the authors meant an upper bound here. 

3. The first paragraph in the Related Work section seems unnecessary; the work is a conformal work, and the mentioned uncertainty methods appear not to be used as baselines. 

4. Line 101-103: Rather than intuitively defining conformal prediction, it would be better to write it down formally as a definition later. Note that the section is about conformal inference under nonexchangeability, but the description starts with traditional conformal prediction.

5. Line 119: The term residual-aware seems odd. Did the authors mean residual-controlling or something?

6. How dependent is the performance on the squashing function? It appears the work only explored tanh function,

7. The target confidence is set as high as 0.9, which is okay. However, it is necessary to demonstrate the performance of the proposed method with varying significance levels. A calibration curve may be helpful in determining if the proposed method works effectively in all cases.

8. For all the experiments, only single-time statistics are provided. It would be helpful to have the experiments repeated across multiple splits or seeds and the standard deviation reported to get a clearer picture of the method's performance. 

9. For many results, where DDCI performs better, there is no clear indication if the difference in width is sufficient in comparison to the deviations in reported coverage to know if DDCI is indeed a better method. Other metrics, such as Winkler Score, might also be helpful to understand the method better.

### Soundness
2

### Presentation
2

### Contribution
2
