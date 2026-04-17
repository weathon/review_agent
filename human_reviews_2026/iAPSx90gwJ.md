# UNDERSTANDING TRANSFORMERS FOR TIME SERIES FORECASTING: A CASE STUDY ON MOIRAI

- Decision: Accept (Poster)
- Scores: 6, 6, 4

## Abstract
We give a comprehensive theoretical analysis of transformers as time series pre-
diction models, with a focus on MOIRAI (Woo et al., 2024). We study its ap-
proximation and generalization capabilities. First, we demonstrate that there exist
transformers that fit an autoregressive model on input univariate time series via
gradient descent. We then analyze MOIRAI, one of the state-of-the-art multivariate
time series prediction models capable of modeling arbitrary number of covariates.
We prove that MOIRAI is capable of automatically fitting autoregressive models
with an arbitrary number of covariates, offering insights into its design and em-
pirical success. For generalization, we establish learning bounds for pretraining
when the data satisfies Dobrushin’s condition. Experiments support our theoretical
findings, highlighting the efficacy of using transformers for time series forecasting.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper develops a theoretical lens on transformer models for time-series forecasting, with a focus on MOIRAI. It claims three main contributions: (i) existence proofs that transformers can implement (via in context learning -ICL-) least-squares AR regression on univariate series and—via MOIRAI’s “any-variate” mechanism—extend to multivariate AR with an arbitrary number of covariates; (ii) generalization bounds for pretraining under Dobrushin’s condition, yielding error decaying on the order of (1/$\sqrt{nT}$); and (iii) experiments (synthetic + some real-world) showing prediction error decreases with longer lookbacks, consistent with the theory. The motivation is to replace architectural heuristics with a principled understanding of why transformer time-series foundation models (particularly MOIRAI) work well and handle variable covariate counts.

### Strengths
In my understanding, this paper draws a new connection between transformer ICL and AR regression for time series, arguing MOIRAI’s any-variate attention and concatenation scheme enables automatic AR dimensionality selection across variable covariate sets. This is a useful theoretical complement to recent empirical TSFMs. The pretraining generalization analysis under non-IID dependence (via Dobrushin) is also timely, given TSFM pretraining on heterogeneous corpora. The positioning against prior ICL theory (largely fixed-dimensional, next-token settings) and time-series FM work is appropriate; the contribution is incremental on the ICL side but significant for time-series where clear theory is scarce.

- Bridges ICL theory to time-series AR regression; explains MOIRAI’s ability to handle arbitrary covariate counts.
- Non-IID pretraining generalization bound under Dobrushin is relevant to TSFM.
- Clear limitations section; transparent about ReLU vs softmax and AR scope.

### Weaknesses
The empirical section aims to validate theoretical trends rather than chase SOTA: synthetic AR data confirms error decreases with longer context and that pretrained MOIRAI adapts across (d,q); there’s a limited real-world section in the appendix. I would like to see stronger baselines beyond LSR (e.g., well-tuned ARIMA/ETS, simple RNN/Temporal-Conv) and ablation isolating any-variate bias terms’ role in practice. Also, reporting statistical variability (multiple seeds) and calibration (since pretraining uses MSE) would strengthen claims.

- Empirical findings can be extended: limited real data, modest baseline coverage, lacking variance/calibration analysis.
- The Dobrushin assumption is nontrivial; guidance on when it holds in common TS domains would help external validity.
- Several results rest on formatting assumptions and constructed sequences; practical robustness across typical TS preprocessing is less clear.

### Questions
- Can you characterize classes of TS (e.g., ARMA, VAR with certain stability) that satisfy Dobrushin and provide diagnostics to check it in practice?


- How sensitive are your AR-via-ICL constructions to positional encoding choices and patching (MOIRAI uses patching by default)? Can you provide ablations?


- Is it possible to add stronger empirical baselines and calibration (e.g., CRPS or PIT histograms) on real datasets to complement theory?


- Beyond AR, can your constructions cover state-space or seasonal/trend components? Even partial results would broaden scope.


- Where does any-variate bias (u₁, u₂) matter most? An ablation isolating those terms would be informative.

### Soundness
3

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
4

### Summary
This work provides a clean theory for why transformer-style time series foundation models, by focusing on the MOIRAI work. They're showing a transformer can learn to perform least-squares autoregression on univariate series via in-context gradient descent. The authors then prove MOIRAI's any-variate encoding and any-variate attention let the model infer an AR predictor with an arbitrary number of covariates, explaining its strong zero-shot and few-shot behavior across domains.

### Strengths
I found the originality, quality, clarity and significance of the work very well. The paper gives a fresh, concrete theory for TS transformers by showing a transformer can do least-squares AR via in-context gradient descent and by explaining how MOIRAI’s any-variate attention builds per-variate lag histories to support multivariate AR, extending prior ICL views to time series with unknown lags. MOIRAI is a perfect paper in my view, and I am happy that a team is working on it. The technical core is strong, and the theory is then stress-tested on real data. The paper also defines the attention variants and the MOIRAI transformer cleanly and keeps the flow from setup easy to follow. Finally, the authors unify why MOIRAI-style FMs work in forecasting, give a generalization story for non-IID series, and show practical gains over tuned AR baselines on ETTh1/ETTm1, so the results matter both for theory and for building stronger pretrained TS models.

### Weaknesses
While I really enjoyed reading the methodology and approaches that the authors followed, I'd like to raise some weaknesses that came to my mind, and I'm looking forward to a nice discussion on these with the authors. 

(1) From my understanding, the theory and constructions use a ReLU-based attention, and not softmax, so I'm curious to know if the authors are willing to either extend the proofs or add tests showing the key claims still hold with softmax MOIRAI. 

(2) Another point is that the pretraining bounds hinge on Dobrushin's condition, so I was wondering if the authors could report a practical check for this on ETTh1/ETTm1 or at least share their thoughts on it.

(3) Additionally, I found the real-data study a bit narrow because only ETTh1/ETTm1 and comparisons only to AT models were discussed, so maybe adding another domain and strong time series foundation model baselines would make the empirical case much stronger. Do the authors have any thoughts on it? Would be good to know.

(4) Finally, the loss uses output clipping (ClipBx), and theory/results depend on qmax and dmax, so I was thinking maybe a short sensitivity study and guidance on choosing these would improve usability.

### Questions
I'd encourage the authors to check the Weaknesses part first, and here I have two more questions.

(1) Your theory assumes d≤dmax and q≤qmax with qmax·dmax tied to the hidden size D. How should we expect the model to behave when test-time d or q exceed these limits, and can we anticipate meaningful degradation vs sharp failure from the construction? Would be good to know the answer to it

(2) Any-variate attention uses a block matrix U with fixed block size T, so I was wondering how the framework handles irregular sampling or missing timestamps across covariates where a clean block structure does not exist and what time encoding is implied in that case? 

Thanks again for your hard work.

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
2

### Summary
This paper provides a theoretical analysis of how transformers can perform well for time series forecasting, focusing particularly on the set of time series modeled by AR processes. First, the authors show that transformers with linear attention can approximate AR regression on univariate, as well as multivariate case (based on the any-variate attention in MOIRAI). Second, the authors use Dobrushin's condition to show a generalization bound for why pretraining on diverse datasets works. Finally, the authors ran experiments on synethetic AR data by first pretraining on generated AR data, and then compare the performances on in-distribution AR processes and out-of-distribution AR processes against least squares regression.

### Strengths
The paper is mostly easy to follow. It extends the ideas in Bai et al. to show that a transformer can approximate any AR model, and also shows the generalization ability of time series pretraining. It also conducts a few controlled experiments to validate the findings.

### Weaknesses
1. Writing. I can follow the paper, but there are quite a few typos here and there, and even in the title “Time Seires” should be “Time Series”,
- I think the following description of iTransformer might be a bit misleading: “iTransformer (Liu et al., 2023) propose to use a pooling technique to reshape arbitrary number of covariates into a unified size.”
- Line 298-299: “.” instead of “,”. For each time series, we encode it with any-variate encoding into an input matrix denoted asH ∈ RD×N , 1 We define each pretraining sample as zj := (Hj , yj ), where yj = x1T j . 
- This seems to be an overstatement. I believe there are other transformer models, or TSFMs that can handle an arbitrary number of covariates. “Note that in the multi-variate case, we only focus on MOIRAI as it is the only transformer-based model that is compatible with arbitrary number of covariates.”
- Move Figure 1 closer to the experiment section.
- Line 143, w* = (w1, ..., wj ) \in R^{qd} should be  (w1, ..., wd )?
2. The proof on approximating any AR model with a transformer seems to be a straightforward application of the proof in Bai et al., as the main difference is the construction of the feature and label pairs. However, I am not very familiar with related literatures. The authors can correct me if I am wrong.

### Questions
1. I don’t think the setting in Appendix F6 EVALUATION ON REAL-WORLD DATASETS is appropriate. The window size seems to be too small for ETTm1 and ETTh1 to cover a whole period of time series. This might be the reason that MSE increases along with window size in Figure 3 left, which is counter-intuitive.
2. Line 80: "We impose periodic boundary conditions for the negative index, i.e., x−1 = xT." Why do we need this? It seems unconventional.
3. A recent paper "Why Do Transformers Fail to Forecast Time Series In-Context?" (https://arxiv.org/pdf/2510.09776) argues that a simple linear model has direct access to the full history of the time series, while an LSA-based model must compress this entire history into the fixed-dimensions in the Q/K/V matrices. Based on this observation, they show that for any AR(p) process, an optimally parameterized LSA model cannot achieve a lower expected MSE than the classical optimal linear predictor. Can you compare your analysis with theirs, and share your thoughts on why your findings are different?
4. Section 4 is a bit difficult to follow. Consider adding/moving more insights and intuitions to somewhere earlier in the paper.

### Soundness
3

### Presentation
2

### Contribution
3
