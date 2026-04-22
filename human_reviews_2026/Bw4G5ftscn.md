# Beyond Extrapolation: Knowledge Utilization with Bidirectional Inference for Time Series Forecasting

- Avg Score: 2.67
- Decision: Withdrawn (Treated as Reject)
- Scores: 0, 2, 6

## Abstract
Time-series forecasting is critical in application areas such as energy, transportation, and public health.
Most existing forecasters, however, are designed primarily around unidirectional inference from \textbf{history} to \textbf{target}. 
While this formulation has achieved strong performance in many practical scenarios, it focuses solely on the history–target link and leaves unused the structured information in how trajectories continue after the target, even though such post-target behaviour can provide a valuable inductive bias for forecasting.
In a typical time series, each training example naturally forms a chain of three segments: ``\textbf{history} (model input), \textbf{target} (ground-truth output), \textbf{post-target continuation}''. 
In this work, we explicitly use the third segment as a source of auxiliary features and propose KUP-BI (Knowledge Utilization Paradigm with a Bidirectionally Inspired Auxiliary Stream), a simple non-parametric mechanism that distils continuation-style information from a train-only historical library and injects it into standard forecasting backbones.
For each training chain, we extract an equal-length history window and post-target continuation window, apply a simple ratio-style operator that encodes how the continuation changes relative to its history, and store the resulting transformation together with its history in the library.
Given a current input window, we  extract similar historical segments from this library, aggregate their associated transformations, and apply the aggregated transformation to the current input to obtain a deterministic continuation-style auxiliary feature that summarises how similar histories tended to evolve in the training data.
The input and auxiliary streams are encoded separately and fused through a lightweight feature-level gating module. 
This design does not introduce information beyond what is already contained in the training trajectories, but provides a structured inductive bias that helps backbones exploit typical continuation patterns rather than relying solely on parametric extrapolation. 
Across six benchmarks and several state-of-the-art models, KUP-BI consistently improves forecasting performance with small additional overhead.

## Human Reviews

## Human Reviewer 1

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
This paper proposes using what they call a approximate future prior in addition to the predictive forward model for forecasting.
The paper motivates the problem, goes through a number of derivations of the method they use and then demonstrates the method on a number of testbeds, comparing with standard MSE training methods, but not particularly looking towards state of the art approaches, or considering any of the normal regularisation methods that one would reasonably employ in these settings for the size of forecasts being considered. They focus on long prediction length, but do not consider autoregressive unrolling, or consider methods that are more targeted at long horizon distributional forecasts (e.g. probabilistic basis function approaches).

### Strengths
I am uncertain of the strengths of the paper.

### Weaknesses
The motivation for this paper is somewhat puzzling to me. In a forecasting model P(future|past), the prior is already included in the forecast distribution, in that it moves from the unconditional prior P(future) to the conditional posterior P(future|past), that captures the whole future distribution. Combining on an additional prior component to this breaks the rules of probability. Hence the whole process seems somewhat dubious. I see nothing in the motivation or the problem description that either relates to this or addresses this. The writing of this paper seems somewhat lacking in understanding of the probabilistic foundations of forecasting altogether. 

The formulation itself seems fundamentally broken, and is certainly unmotivated. The authors define R as the division of two matrices, without defining what is meant by that. The motivation for this division (assumed component-wise) is also problematic, and not provided and seems entirely arbitrary. The paper introduce an undefined query vector X_q, and do not explain the purpose here: it appears to be to search histories to find nearby examples that can be used to build a distribution for the future, but this is not a future prior, as it is history dependent, and there is no loss function element. The paper then goes on with what, to my reading, seem like fairly meaningless calculations to produce something that is called a future prior generation, but really seems broken to me. I think what the paper is actually trying to do is some crude form of backoff, akin to that in n-gram language models, but this neither clear nor properly formulated.

The rest of the paper rests on all this, and sadly I think this sits on very shaky ground.

If the authors really believe there is merit in this work they need to found it properly in a proper probabilistic framework, and explain each step of the approach, why each step is done, what impact each step has on the probabilistic formulation of the problem and how that all complies with the rules of probability. This should start with the foundations. As it stands it is a list of apparently-arbitrary and potentially ill-motivated set of computations. That the authors might be able to run some experiments that enable them to bold a column in a table, but that does not, in itself, demonstrate that the approach has any merit. At the end of the paper, I am left with no insight, no change in understanding that I did not have before, and little understanding of the point or process of the procedure they follow.

Fundamentally IMO the premise is wrong: "most existing forecasting models rely solely on unidirectional inference from history to future. While effective in stable scenarios, this paradigm lacks explicit structural constraints about the future, which makes it challenging to stabilize predictions under complex dynamics." - forecasting models rely on inference from history to the future because the history is the only place where data is given - there is no inferential information from the future as it has not happened yet. But the claim that the "paradigm lacks explicit structural constraints about the future" is patently false - that is what the forecasting model does - it is predicting the future evolution, either as a joint distribution over the future states, or as an unrolling of the next-step predictions via the chain rule. These future predictions have an implicit prior already (simply sum out the conditional distribution over the histories to get the prior). These future predictions then already incorporate the "structural constraints about the future". The argument might be that current approaches do that badly (I am not sure they do, and methods such as classifier-free guidance tackle this significantly), but then that needs demonstrating, reasoning about and a clearly formulated, and mathematically justified fix applied. This is not happening here.

### Questions
I do not believe there are answers to any questions that would really clarify or fix this paper for me sufficiently beyond a fundamental rewrite. However the basic questions hold: what is the context, what is the definition of the problem space, what is the current failure-mode that you see in this problem space? In what way and why do current state-of-the-art approaches fail to address this, and how have you demonstrated this? What is the fundamental methodologically grounded insight that you bring to this that allows you to overcome this problem? How is this implemented? How have you reliably demonstrated that this really does do what you expected it to and complies with the methodological expectations in practice?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper presents a novel algorithm of using future priors developed independently to improve time series forecasting by leveraging both the historical values and the future priors. The paper shows improvements in the forecasts based on this methodology across standard benchmark datasets within the time series foundation models community using three forecasting models, namely, PatchTST, DLinear and TimesNet.

### Strengths
Paper presents a novel algorithm by reframing the forecasting problem as  an interpolation problem rather than the conventional extrapolation problem to improve time series forecasting using future priors across a wide range of benchmark datasets. 

Algorithm also allows for any other future prior to be utilized and plugged into the algorithm. 
Significant gains in performance have been shown for DLinear algorithm and modest gains for PatchTST and TimesNet.

### Weaknesses
It is not clear what are the advantages of this method as compared to existing methods mentioned in the literature review by the authors in Section 2.2. 

I think a comparison with state of the art methods is missing. I think comparing the algorithm mentioned in this paper with RAFT model in Section 2.2 or other models in Section 2.2 will illustrate the benefits of this work better (if any). 

State of the art models can be improved. TTM and Chronos are probably more standard models employed right now as compared to PatchTST.

### Questions
As stated in the weaknesses, I think the authors can expand on this section to make the applicability of this method more clear with reference to prior art in Section 2.2.

1. Where would this method work better than the existing methods? 
2. Are there any computational benefits compared to the existing methods?   
3. How does this method compare with other state-of-the-art methods like RAFT?

### Soundness
3

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
This paper introduces KUP-BI (Knowledge Utilization Paradigm with Bidirectional Inference), a framework that augments standard unidirectional time-series forecasting models with an auxiliary future-prior stream. Instead of predicting solely from historical data (past → future), KUP-BI retrieves approximate future priors from a library of historical patterns (constructed as history–target–future chains). These priors represent structural cues about plausible future trajectories and are fused with the current input’s representation via a lightweight gated module. The design reframes forecasting from a purely extrapolative problem into an interpolation-like one, using approximate structural knowledge from correlated historical patterns to stabilize predictions.
Empirical results on six datasets (ETTh1/2, ETTm1/2, Exchange, and ILI) with three backbones (DLinear, TimesNet, PatchTST) show consistent improvements in MSE/MAE, particularly for lightweight models (e.g., DLinear). The authors also provide theoretical motivation (an interpolation vs. extrapolation error bound under Lipschitz continuity) and detailed ablations on retrieval quality, fusion strategies, and hyperparameters.

### Strengths
The conceptual novelty lies in explicitly incorporating an estimated “future prior”—learned from training histories that resemble the current input—as a structural anchor to guide prediction. This moves beyond existing retrieval-augmented or exogenous-enhanced methods, which either concatenate retrieved outputs or use external signals. The bidirectional reasoning idea—combining historical and prospective cues—is fresh and supported by a solid theoretical argument that interpolation is less error-prone than extrapolation for Lipschitz functions.

While related to retrieval-augmented forecasting (RAFT, TS-RAG), future prior modeling as a separate learnable stream with a ratio-based representation (rather than directly fusing retrieved outputs) appears original. The method’s simplicity and plug-and-play nature increase its potential impact—especially since it can attach to diverse backbones with negligible overhead.

- The ratio-style operator ($R = (F-H)/(H+\epsilon,\text{sign}(H))$) is clearly defined, and the retrieval and softmax-weighted fusion mechanisms are mathematically specified.


- The Lipschitz interpolation theorem is formally derived in Appendix A and conceptually motivates the approach. While idealized, it reinforces the intuition that adding a bounded-fidelity future anchor reduces variance.


- The gated fusion module and harmonic residual design (convex combination controlled by α and per-channel gate γ) are reasonable and stabilizing.


- The experiments seem to be methodically conducted with non-leaking retrieval libraries (constructed only from training data).

### Weaknesses
I believe 

- the retrieval-based prior depends on correlation measures that may fail under phase shifts or abrupt regime changes. Although the authors acknowledge this, no quantitative robustness analysis is included.

- the “ratio operator” assumes relative amplitude continuity between history and future, which might not hold in nonstationary series (financial or event-driven).


While results are statistically consistent, the improvements are modest (1–7% MSE reduction). It would help to report statistical significance or confidence intervals. Also, the approach may implicitly leak periodic information if future segments overlap with nearby training windows—clarifying how window offsets are handled would strengthen reproducibility, given that code is not shared yet.

### Questions
- Can you quantify retrieval precision (e.g., correlation between estimated and true future segments) to better understand when the prior helps?


- How does KUP-BI perform on non-periodic or abrupt-shift datasets (e.g., stock tick data)?


- Could a learned retriever (e.g., embedding-based) outperform correlation-based matching?


- The ratio operator assumes elementwise alignment—could phase-shift alignment or dynamic time warping improve the prior’s fidelity?


- Can you include some runtime comparisons (ms per batch) to substantiate claims of negligible overhead?

### Soundness
3

### Presentation
2

### Contribution
2
