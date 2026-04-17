# Why Do Transformers Fail to Forecast Time Series In-Context?

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 2, 4

## Abstract
Time series forecasting (TSF) remains a challenging and largely unsolved problem in machine learning, despite significant recent efforts leveraging Large Language Models (LLMs), which predominantly rely on Transformer architectures.
Empirical evidence consistently shows that even powerful Transformers often fail to outperform much simpler models, e.g., linear models, on TSF tasks; however, a rigorous theoretical understanding of this phenomenon remains limited.
In this paper, we provide a theoretical analysis of Transformers' limitations for TSF through the lens of In-Context Learning (ICL) theory.
Specifically, under AR($p$) data, we establish that: (1) Linear Self-Attention (LSA) models $\textit{cannot}$ achieve lower expected MSE than classical linear models for in-context forecasting; (2) as the context length approaches to infinity, LSA asymptotically recovers the optimal linear predictor; and (3) under Chain-of-Thought (CoT) style inference, predictions collapse to the mean exponentially. 
We empirically validate these findings through carefully designed experiments.
Our theory not only sheds light on several previously underexplored phenomena but also offers practical insights for designing more effective forecasting architectures.
We hope our work encourages the broader research community to revisit the fundamental theoretical limitations of TSF and to critically evaluate the direct application of increasingly sophisticated architectures without deeper scrutiny.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper studies why Transformers, modeled via Linear Self-Attention (LSA), underperform simple linear predictors in time series forecasting. Under AR(p) processes, the authors prove that (1) one-layer LSA operates on a coarser σ-algebra than the raw context and thus cannot beat optimal linear regression; (2) there exists a strict finite-sample excess-risk gap relative to linear predictors, with an explicit O(1/n) rate; (3) even though LSA asymptotically matches linear predictors under teacher forcing, Chain-of-Thought rollouts collapse to the mean exponentially. Experiments on synthetic AR data corroborate the theory: OLS consistently outperforms LSA under TF, and CoT rollouts of both collapse, with LSA failing earlier.

### Strengths
1. This work formalizes LSA’s representational limits for AR(p), giving a strict Schur-complement gap and an explicit 1/n rate—a precise and practically interpretable negative result.
2. Connects ICL theory to TSF and explains why simple linear models dominate on AR-like data, aligning with extensive empirical observations
3. Highlights that in TSF, attention’s learned compression may hide local lag structure; suggests focusing on architectures beyond self-attention or leveraging MLPs/Softmax differently.

### Weaknesses
1. The Process focus on stable AR(p) / linear-stationary processes; many real TSF tasks exhibit regime shifts, seasonality with exogenous inputs, or nonlinearities where conclusions may partially change
2. Experiments are on synthetic AR data with Gaussian noise; lack of diverse real-world datasets may limit external validity of empirical claims
3. Main results center on LSA-only Transformers; standard Softmax attention, multi-head interactions, and strong MLP blocks are not theoretically covered. Many studies/observations have discussed the weaknesses of transformers in TSF, and shown that simple linear structures can perform better. This paper primarily provides a theoretical explanation, so a more comprehensive analysis would make the work more solid.

### Questions
I’m not quite familiar with the theoretical analysis of TSF (models), and compared with the many TSF papers focused on experiments and applications, theoretical ones are relatively rare. It’s hard to be convinced "why transformers perform poorly on real-world TSF" under the theoretical analysis (although related works have demonstrated this through extensive experiments and observations). However, has the author considered the gap between theory and practice, and how to address this concern effectively?

### Soundness
2

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
4

### Summary
This paper studies the reason behind the failure of transformers models to surpass simpler methods, such as linear baselines, on time series forecasting. It adopts a theoretical perspective using in-context learning and show that under AR(p) data, linear self-attention models cannot reach the performance of linear models for in-context forecasting. Further results show that increasing the context window up to infinitely closes the gap asymptotically and that chain-of-thought inference make the predictions collapse (to the mean). Numerical experiments are made to confirm these findings.

### Strengths
- The subject tackled is of great interest: many simpler and even linear baselines are strong competitors to more advanced and heavy models. Better understanding why could enable the development of better models.
- The paper is well-written and the theoretical analysis seems well-conducted (I did not check all the proofs in appendix).
- I believe some of the proofs technique could be of independent interest when studying transformers and large language models which is a nice tool to have in the submission.

### Weaknesses
I list below what I believe are weaknesses but I would be happy to be corrected if I misunderstood some parts of the work.

- My main issue is about the positioning of the paper that claims to identify why transformers fail for time series forecasting in-context. It should be noted that many of the time series forecasting models that are transformer-based do not rely on in-context learning and are not autoregressive  but rather encoder-only models (Moirai, PatchTST, SAMformer, Informer, Autoformer, etc.). As such, I do not understand why the paper focus on autoregressive transformers to forecast in-context. Could the authors elaborate on that?
- Connected to the weakness above, the motivation to look at autoregressive transformers in-context is not clear to me. Further studying chain-of-thought for time series forecasting can be interesting but the motivation of findings why transformers fail in time series forecasting is not related to that. More prior works on in-context and chain-of-thought with transformers for time series forecasting should be discussed to convince the reader that this is an issue in practice.
- The main claim is to better understand why complex and heavy transformer models fail for time series forecasting compared to linear baselines, yet the authors consider an attention-only model **with linear attention**. I believe this is an oversimplification, which I do understand from a theoretical perspective to be able to conduct meaningful derivations, however I do not find it convincing to explain the issues of highly non-linear deep transformer based models. Notably, without such structure, the analysis is likely to be disconnected from practical models and no experiments is conducted on them to verify whether the identified issues are really the cause of transformers failure.
- Several mentions in the paper are made regarding how the contributions are important to better understand the failures of transformers yet as explained above, the problem studied is not related to how transformer models are used in time series forecasting. While the technical results are interesting, the phrasing and conclusion from the results are misleading.

Overall, the theoretical analysis is well conducted but the motivation of the paper is unconvincing and the problem formulation is not well motivated. The reasons to look at such AR(p) models for very simplified transformers with linear attention does not seem connected to better understanding failure modes of transformers for time series forecasting or it was not enough explained in the paper. This is the reason why I lean towards rejecting the paper its current state.

### Questions
- I believe [1] would be interesting to discuss since the authors propose a lightweight transformer model SOTA (at the time) and studies in detail why other models fail compared to linear baselines. They identify the optimization to be the cause and show that a small transformer with sharpness-aware optimizer improved significantly over competitors. Could the authors discuss this work from the perspective of identifying failure of transformers?
- [2] is cited in line 59 with other transformers for time series work while it studies in detail ICL for autoregressive models from a theoretical perspective (likely to be applied on text, not time series). Could the authors elaborate on that?

*References*

[1]  Ilbert et al. SAMformer: Unlocking the Potential of Transformers in Time Series Forecasting with Sharpness-Aware Minimization and Channel Wise Attention. ICML 2024

[2] Sander et al. How do Transformers Perform In-Context Autoregressive Learning? ICML 2024

### Soundness
2

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
3

### Summary
This paper analyzes why Transformers struggle with time series forecasting (TSF) when used in an in-context learning setup.
It focuses on Linear Self-Attention (LSA) applied to autoregressive (AR(p)) data and proves three key results: 1) LSA can’t beat linear regression (LR) — there’s a provable gap in performance that only disappears asymptotically. 2) As the context grows, LSA converges to the linear predictor, but never surpasses it. 3) During iterative (chain-of-thought) rollout, predictions collapse to the mean, with errors compounding exponentially. Synthetic experiments on AR data confirm these findings: LSA tracks the process but always performs slightly worse than OLS.

### Strengths
1. The paper delivers a clear and rigorous theoretical explanation of why attention-only Transformers struggle on linear time series forecasting tasks. Its main strength lies in the originality of deriving a strict finite-sample performance gap and an explicit convergence rate, offering a good perspective on the representational limits of LSA. 

2. The proofs are technically sound and well-structured, extending beyond Gaussian settings with careful mathematical reasoning. The writing is clear and organized, with concise takeaways and effective figures that support the theoretical results. 

3. The work makes a timely and meaningful contribution by grounding empirical observations, where simple linear models outperform Transformers, in solid theoretical analysis, providing valuable insights for both researchers and practitioners in time series modeling.

### Weaknesses
1. The title/abstract claim that “Transformers fail” is broader than what is proved: the strict finite‑sample gap is shown for single‑layer LSA on univariate AR(p) with a Hankel + label‑slot input. Depth results are non‑strict, and there is no theory for Softmax attention or attention+MLP as a full Transformer block. 

2. The model structure removes the usual MLP blocks and prediction head in Transformers. This isolates an restricted ICL mechanism with linear attention but diverges from standard practice where a head and residual pathways to output prediction head exist, limiting external validity. Ablations with a simple linear head are missing. 

3. The theory and experiments rely on a specific Hankel format that pre-packages lags and bypasses the need for positional encoding. It remains unclear whether the conclusions hold for standard tokenized time series sequences with positional embeddings. If not, linking these results to previous findings from works such as DLinear and claiming that linear models outperform Transformers would be an overstatement, especially since models that take single time-series inputs, such as PatchTST, still show superior performance compared with linear models.

4. Experiments are only on synthetic AR(p). There is no validation on multivariate (VAR), nonlinear/regime‑switching, seasonal, or exogenous‑covariate settings, nor on real datasets. The claimed 1/n excess‑risk rate is not empirically verified. Moreover, the paper compares to OLS with known order p, but does not show fairness to the unknown‑structure case, e.g. when p is not given.

5. Calling free‑running multi‑step rollout "Chain‑of‑Thought" (CoT) is potentially misleading for readers who associate CoT with reasoning steps in language tasks. “Iterative rollout prediction” would be clearer?

### Questions
1. Maybe it is a better choice to narrow the title/abstract to LSA‑only on univariate linear autoregressive processes? If not, additional results might be needed to justify the broader “Transformers fail” claim.

2. How do results change if you add a linear prediction head or MLP layer? Does LSA‑only still exhibit the same finite‑sample gap when a head can directly read the lag features?

3. Do your conclusions extend to VAR, piecewise‑linear, or mildly nonlinear processes? A lemma, counterexample, or small empirical study would clarify applicability.

4. Could you provide a partial lower bound or a linearized argument for Softmax attention, and discuss how incorporating MLPs might modify the feature-span argument?

If the authors' response addresses my questions and concerns mentioned above, I am willing to raise my score.

### Soundness
2

### Presentation
3

### Contribution
2
