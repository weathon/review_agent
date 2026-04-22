# Point-wise Anomaly Detection via Fold-bifurcation ODE

- Avg Score: 4.67
- Decision: Accept (Poster)
- Scores: 6, 6, 2

## Abstract
Anomaly detection in time series is essential for applications from industrial monitoring to financial risk management. Recent methods --- including forecasting error models, representation learning, augmentation, and weak-label learning --- have achieved strong results for specific anomaly types such as sudden point or gradual collective anomalies. While many prior works report window-level metrics that may mask errors, several recent methods evaluate at the point level as well. Our goal is to use a stricter point-wise protocol to make masking effects explicit. We introduce FOLD (Point-wise Anomaly Detection via fold-bifurcation), a framework that reframes detection as tracking a system’s proximity to a critical transition. FOLD extracts stress signals from a forecasting model and integrates them with a fold-bifurcation inspired ODE to produce the risk state, flagging anomalies once it crosses a threshold calibrated on normal data. This requires no anomaly labels and no additional detector training, enabling a parameter-free and efficient detection process. By modeling anomalies as stress accumulation toward a tipping point, FOLD naturally aligns with point-wise detection, providing a unifying and interpretable perspective that complements type-specific methods. Experiments on 40 benchmarks against 34 state-of-the-art baselines show that FOLD achieves competitive or superior performance, with particular strength under strict point-wise evaluation.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper presents FREE (Free point-wise Anomaly Detection via fold-bifurcation), a framework for time-series anomaly detection that targets strict point-wise evaluation rather than coarse window-level metrics. FREE computes stress signals from a forecasting model and feeds them into a fold-bifurcation–inspired ODE to track a system’s risk state; anomalies are flagged when this state crosses a threshold calibrated on normal data.

### Strengths
FREE, grounded in fold-bifurcation dynamics, recasts anomaly detection as a point-wise decision process driven by the progression from stress to critical transition, providing strong theoretical footing and inherent interpretability. By integrating forecast-derived stress signals into a bifurcation equation, it captures the buildup of gradual pressures while remaining sensitive to sudden tipping events. Extensive multi-benchmark evaluation demonstrates strong performance—particularly under strict point-wise settings—achieving a balanced mix of accuracy and efficiency and underscoring its practical value.

### Weaknesses
1.The central assumption—that many real-world failures stem from gradually accumulating stress leading the system toward a critical transition rather than isolated spikes—seems reasonable, but the paper would benefit from additional concrete scenarios to broaden and clarify its practical applicability.

2.For the fold-bifurcation–inspired ODE modeling, the linkage between theory and specific application domains is underdeveloped; the discussion remains largely theory-driven without fully mapping the formulation to domain variables, constraints, and operational workflows.

3.On visualization, it’s hard to see—at a glance—where the method outperforms alternatives. Clear comparative plots would help, ideally accompanied by visualized uncertainty assessments to show reliability and confidence.

4.Regarding thresholding, the confidence-interval analysis feels insufficient. The large differences in preset thresholds—are they driven by dataset standards? Could thresholds be informed by priors or domain knowledge to improve calibration stability?

5.The efficiency results are compelling, but a deeper analysis is needed to understand the sources of improvement relative to other methods.

### Questions
My primary concerns center on actionable guidance from theory to practice, the method’s novelty relative to comparable baselines, a thorough efficiency analysis, and a clear discussion of threshold selection and robustness.

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
This paper introduces a point-wise anomaly detection method grounded in fold-bifurcation dynamics. The approach is fully unsupervised, requiring neither anomaly labels nor detector training.

### Strengths
The paper develops an intersting view of anomaly detection where by anomaly is seen as the cause of accumulating stress within the system.  Stress signals are extracted from time series data and integrated using a fold-bifurcation ODE.  These signals accumulate over time till they reach a tipping point and an anomalous condition is raised.

The proposed method is reported to address a key shortcoming of prior approaches that show strong results on window-based metrics but often fail to generalize under point-wise evaluations.

The proposed scheme is evaluated on 9 benchmarks and compared against 10 baselines, and the results suggest that the proposed scheme achieves good performance under strict point-wise evaluation.

The premise of this work appears sound given the work in (Scheffer et al. 2009) that shows that variance increases as a system approaches a critical transition.

### Weaknesses
Please state the paper’s novelty relative to the early-warning systems you cite---what does this scheme enable that prior EWS methods do not?

Perhaps I missed it, but it seems that z(t) is computed independently for each feature?  Is that so?  If it is true then how does this method captures subtle interactions between multiple variables present in the time series?  I couldn't following Figure 3, for example, since I was confused to see a single z(t) curve.  Shouldn't there be one for each variable?

### Questions
- Can you please discuss control parameter r in Eq. 1 and Eq. 2 and why is it appropriate to capture it as a time-varying stress signal S(t) in Eq. 3. 

- In what sense is FREE better than early warning systems that monitor Eigenvalues trends?  Why cannot we use early warning systems for anomaly detection?

- Perhaps it is obvious to those working with these ODEs, it isn't immediately obvious to me how will one determine that risk trajectory z(t) has left its stable basin?

- In Sec 3.2, the forecasting model f_{\theta} predicts the next H elements?

- Purturbed sequence contains X except those that belong to patch i?  Is this is what we mean by purturbed sequence?

### Soundness
3

### Presentation
3

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
The paper proposed a method for point anomaly detection in time series, which is an important task in time series domain. The proposed method employs forecasting model and provide masked signal to produce a sensitivity score, which is then utilized with the fold bifurcation theory to get anomaly scores. The method seems to outperform some baselines across 9 datasets in F1 score.

### Strengths
The paper has the following strong points:
1. The proposed method is novel, and well motivated. Particularly, the paper does a good job in motivating the need for the fold bifurcation theory and it's usage for anomaly detection.
2. The proposed method is theoretically motivated and well formulated.
3. Some experimental evidence has been provided.
4. The paper is well written.

### Weaknesses
However, the paper has the following weaknesses.
1. Does the proposed method work only for point anomalies? The author(s) say "However, most approaches are evaluated under coarse window-level settings, which can mask their limitations in the stricter point-wise anomaly detection scenario." This is not entirely true. There are recent papers with multiple types of anomaly detection capabilities. Example include: [1], [2], and many more. 
2. The evaluation metric F1 score has been establised as a biased metric in recent literature [1]. Why does the author not use mathematically proven good metrics like VUS-PR defined in [1]?
3. The evaluation should be done on well establised anomaly detection benchamarks such as in [1], and should be compared with recent SOTA methods on more datasets.
4. Since the proposed method does patching, masking etc., what is the runtime complexity of the method? In terms for both theoretical and empirical evidence. Since anomaly detection is needed in realtime in most usecases, runtime is a very important factor.
5. The authors say "many real-world failures arise not from isolated spikes but from the gradual accumulation of stress that drives a system toward a critical transition." This is not true in several situation when external factors create anomalies, or even periodic anomalies. 
6. How does the method perform on multivariate time series? 

[1] Liu, Qinghua, and John Paparrizos. "The elephant in the room: Towards a reliable time-series anomaly detection benchmark." Advances in Neural Information Processing Systems 37 (2024): 108231-108261.
[2] Ekambaram, V., Kumar, S., Jati, A., Mukherjee, S., Sakai, T., Dayama, P., ... & Kalagnanam, J. (2025). TSPulse: Dual Space Tiny Pre-Trained Models for Rapid Time-Series Analysis. arXiv preprint arXiv:2505.13033.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2
