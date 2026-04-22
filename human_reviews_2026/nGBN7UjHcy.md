# Beyond Accuracy: Are Time Series Foundation Models Well-Calibrated?

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 4, 8, 4

## Abstract
The recent development of foundation models for time series data has generated considerable interest in using such models across a variety of applications. Although foundation  models achieve state-of-the-art predictive performance, their calibration properties remain relatively underexplored, despite the fact that calibration can be critical for many practical applications. In this paper, we investigate the calibration-related properties of five recent time series foundation models and two competitive baselines. We perform a series of systematic evaluations assessing model calibration (i.e., over- or under-confidence), effects of varying prediction heads, and calibration under long-term autoregressive forecasting. We find that time series foundation models are consistently better calibrated than baseline models and tend not to be either systematically over- or under-confident, in contrast to the overconfidence often seen in other deep learning models.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper investigates the calibration of time series foundation models. The study analyzes five foundation models and two non-pretrained baseline models, evaluating their calibration performance, the impact of different prediction heads, and calibration under long-term autoregressive forecasting. The findings reveal that time series foundation models consistently demonstrate superior calibration compared to baseline models, exhibiting neither systematic over-confidence nor under-confidence.

### Strengths
- The paper is well written and easy to follow.
- The problem is interesting: calibration is an important problem, yet not commonly discussed by the community.
- The evaluation metrics set in the paper are comprehensive, and a series of very interesting conclusions have been drawn.

### Weaknesses
- This is a highly experimental paper with substantial experimental content. But it lacks a deep analysis. For example, in Section 4.4, the author's experimental results indicate that "the predictions with a shorter forecast horizon $p$ have poorer calibration", yet no thorough analysis is provided. It would be interesting to formally know how the "horizon length" affects the prediction performance and why.

### Questions
- In Equation 3, it appears that a curly brace is missing to enclose all the content after the $\sum_{s \in S}$.
- In Section 4.3, the authors concluded that the calibration performance of the Gaussian distribution is inferior to that of other distributions. Could you provide a more in-depth explanation for this phenomenon? Generally, errors are widely assumed to follow a Gaussian distribution. But why is the calibration performance of the Gaussian distribution the poorest?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper targets an underexplored but crucial aspect of time-series foundation models (TS FMs): calibration. It conducts a systematic evaluation of five recent TS FMs against two strong baselines, examining (i) overall calibration and confidence (over/under), (ii) the effect of different prediction heads, and (iii) calibration behavior under long-horizon autoregressive forecasting.

### Strengths
1. The authors investigate the calibration properties in time series foundation models. The topic is of interest in the community.
2. The experiments are comprehensive and confirm that time series foudnation models are well-calibrated.

### Weaknesses
1. Although it is an interesting topic and well motivated, but the manualscript does not provide technicial contributions. The main finding is that time series foundation models are well-calibrated compared with traditional and machine learning / deep learning models. It would be better that if we have some contributions on how to further improve the calibration ability.

### Questions
1. For different prediction heads, it looks like they are similar in terms of calibration error except Gaussian prediction heads. But I wonder how the forecasting performance changes with different prediction heads. Because I feel if the calibration errors are similar, then we should choose the one with better forecasting performance when training a time series foundation model.

### Soundness
2

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
This paper presents the first systematic empirical study examining the calibration properties of time series foundation models (TSFMs). Using appropriate metrics and datasets, the authors first evaluate the forecasting calibration levels of TSFMs, finding that they outperform traditional statistical models and exhibit better calibration. Furthermore, the paper investigates the effect of prediction head design on calibration, revealing that Gaussian heads tend to be under-confident in most cases and are outperformed by alternative head types. The study also explores the impact of forecasting horizon length on calibration, showing that for long-term autoregressive forecasting, increasing the horizon length and using the trajectory AR method leads to better-calibrated forecasts.

### Strengths
1. Appropriate experiment setting: The work identifies the shortcomings of commonly used metrics (CRPS, WQL, MSIS) that conflate calibration and sharpness. By prioritizing PCE and CCE, the paper provides a clearer measure of true probabilistic calibration.

2. Novel insights: The study offers meaningful analysis and insights in each experiment. In particular, it systematically investigates the calibration properties of TSFMs, and explores how factors such as prediction head design and forecasting horizon influence calibration—topics that have not been examined in prior work

3. Clear organization: The paper is well-structured, with logically arranged sections and a coherent flow, making it easy for readers to follow the results and key takeaways.

### Weaknesses
1. Section 4.3 and 4.4 lack clear motivation for investigating the influential factors affecting calibration. Why did the authors choose to focus on the impact of prediction head design and forecasting horizon, rather than other factors such as context length or data characteristics ?

2. The investigation of the calibration properties of TSFMs remains at the validation stage and does not further explore the underlying reasons why TSFMs tend to be well-calibrated, whereas traditional statistical models are often overconfident. A deeper analysis regarding the mechanisms behind this difference would make the work more insightful.

3. While the finding of general insensitivity to the prediction head form is valuable, the scope of prediction head selection is limited. Recently, more novel designs like flow-matching prediction head [1] could also offer new insights into calibration behavior.

[1] Liu Y, Qin G, Shi Z, et al. Sundial: A family of highly capable time series foundation models. arXiv preprint arXiv:2502.00816, 2025.

### Questions
1. Do inherent data characteristics such as randomness and autocorrelation influence the calibration evaluation of TSFMs?

2. The paper finds that TSFMs are not systematically over- or under-confident, which contrasts with observations in image and text domains. What is the possible reason for this difference?

3. A low calibration error does not correspond to a low forecasting error, as seen in the Heart Beat and Patents datasets. What causes this discrepancy between the metrics?

4. What could be the underlying reason that all autoregressive TSFMs are consistently overconfident in long-term forecasting? Whether are there measures could be taken during training to mitigate this bias?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper performs an empirical study on whether time series foundation models are calibrated. They select 5 TSFMs, Chronos-bolt, Moirai 2.0, YingLong, TimesFM and Tirex, as well as N-BEATS and ARIMA as baselines. Then, paper evaluates these models on 6 datasets, focusing on calibration metrics, specifically using Probabilistic Calibration Error, Scaled Interval Width (to evaluate sharpness), and Centered Calibration Error. Experiments show that TSFMs have better calibration than baselines, and a few other insights are presented.

### Strengths
The paper explores an understudied aspect of time series foundation models. It investigates a wide range of recent models across 6 different datasets. Ideas are well presented and the relevant experiments have been performed.

### Weaknesses
* Given that the field has been mostly motivated by accuracy metrics, it would be good to give more motivation in the introduction about why researchers should be interested in calibration.
* It would be good to perform a larger scale investigation, for example computing these metrics for GIFT-eval, to get a more comprehensive understanding.
* It would be good to add in mixture of gaussians and mixture of student-ts for Fig 4, given that Toto has been mentioned several times in the paper, and Toto uses mixture of student-t.
* Section 4.4 is quite confusing, I am unclear why forecasts are more calibrated as horizon length increases. Fig 8 in the appendix also seems to contain contradictory findings. Also, why are tirex and yinglong straight lines?

### Questions
See weaknesses section

### Soundness
3

### Presentation
2

### Contribution
2
