# DiffKANformer: Diffusion KAN Transformer for General Time Series Analysis

- Decision: Reject
- Scores: 4, 6, 6, 6, 0

## Abstract
Time series analysis tasks such as forecasting, imputation, anomaly detection, and classification are crucial for applications spanning climate science, financial domain, retail, and cloud infrastructure. We present DiffKANformer, a conditional diffusion model that integrates Kolmogorov-Arnold Networks (KAN) for feature projection and a Diffusion KAN Transformer architecture for denoising, specifically engineered for time series analysis. DiffKANformer introduces two key innovations: (i) a KAN-based projection mechanism in the forward diffusion process that captures complex correlation between features, and (ii) a Diffusion KAN Transformer architecture that effectively models complex long-term dependencies through adaptive univariate functions. Our model achieves superior performance across four fundamental time series analysis tasks, significantly outperforming existing prominent models in forecasting (eight datasets), imputation (six datasets), classification (ten datasets) and anomaly detection (five datasets). Comprehensive ablation studies across all tasks validate the utility of each DiffKANformer component, demonstrating the model's robustness in diverse time series challenges.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces DiffKANformer, a unified framework for time series analysis that combines conditional diffusion models, Kolmogorov-Arnold Networks (KANs) for feature projection in the diffusion process, and a novel Transformer backbone where the MLP blocks are replaced by KANs. DiffKANformer aims to address limitations of prior diffusion-based and Transformer-based models. Namely, it tries to address the limited expressivity in modeling complex non-linear/periodic dependencies and lack of generality across time series tasks by incorporating KANs to enhance both feature projection and denoising.

The method is positioned as the first to demonstrate state-of-the-art results across forecasting, imputation, classification, and anomaly detection benchmarks. The claim is supported by extensive experiments, ablations, and mathematical derivations.

### Strengths
The paper attempts a unified model that can handle forecasting, imputation, classification, and anomaly detection. This is a unified solution across tasks. 

The model is grounded in a principled, variational framework, with mathematical rigorousness.

Integrating KANs for both data projection and as a replacement for Transformer MLPs is a creative architectural step, and, according to the ablation tables,  provides clear empirical benefits over standard MLPs in both main and auxiliary tasks.

### Weaknesses
The paper is promising and technically interesting, but I have concerns about positioning, clarity, and a few technical presentation issues.

While the paper shows superior performance on a broad suite of benchmarks, not all of the most recent diffusion-based transformers and unified models are included as baselines (e.g. TS-Diffusion, DifFormer etc.) The current baseline selection is robust but misses direct one-to-one evaluations with the latest approaches. 

The paper introduces non-Markovian elements in its trainable forward process. However, while the losses are derived, there are insufficient examples or discussions of how the non-Markovian structure concretely manifests in the actual sampling/generation of time series compared to traditional DDPMs. How does training/inference complexity scale, and are there pathological behaviors in terms of stability or convergence as $T$ increases?

The KL divergence expressions are presented, but it’s not obvious how estimators are computed in practice, especially when the KAN projection is “learned” during the forward process on each mini-batch. Are there subtleties in gradient computation/backpropagation not addressed by standard PyTorch/TF autodiff pipelines?

**The novelty is moderate**. The KAN‑parameterized forward diffusion is a novel and appealing idea; KAN‑DiT is an **incremental** (yet useful) architectural tweak.

### Questions
Can the authors clarify how DiffKANformer differs in principle and practice from recently proposed unified diffusion models (e.g., TS-Diffusion)?

Are there technical or empirical distinctions that uniquely favor KAN-based projections and denoisers? 

Given that the forward process is non-Markovian due to KAN projections, what are the implications for sampling complexity, memory efficiency, and convergence/stability in longer time series or higher $T$ values?

Are there any implementation challenges or numerical instabilities when backpropagating through the KAN-parameterized forward process, compared to standard DiT/MLP architectures?

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
This paper proposes DiffKANformer, a unified diffusion-based framework that integrates Kolmogorov-Arnold Networks (KAN) into Transformer-based diffusion models for general time series analysis. The model introduces a KAN projection in the forward diffusion process to capture complex nonlinear dependencies among time series variables and replaces the standard MLP in the Transformer block with KAN network as a Diffusion KAN Transformer. The approach is evaluated across multiple tasks; forecasting, imputation, classification, and anomaly detection. It demonstrates consistent improvements over state-of-the-art baselines across 29 benchmark datasets.

### Strengths
1. Unified framework for diverse tasks : The model provides a single, cohesive architecture that can handle forecasting, imputation, classification and anomaly detection. This unified approach enhances the model's applicability and practical impact.

2. Clear motivation for KAN integration: The motivation to replace ReLU-based MLPs with spline-based KANs is well-articulated. The authors convincingly argue that KANs capture richer nonlinear relationships and better represent high-frequency temporal components.

3. Comprehensive experimental evaluation: The paper conducts large-scale experiments on a broad set of datasets and includes systematic ablations (e.g. with/without KAN projection), which demonstrate robustness and generality of the method.

### Weaknesses
1. Although Appendix D provides detailed derivations for the forward posterior, loss and variational objective (Eqs. 14~15) but do not provide the proof that the KAN projection yields a tighter variational bound or improved diffusion stability. The forward KL term and prior distribution are motivated heuristically; no convergence or bound analysis is offered.

2. The architectural modifications substitute the transformer's MLP with a KAN block and incorporate adaLN for conditional scaling. This is a well-executed engineering refinement, but the conceptual advance is incremental rather than groundbreaking. 

3. The paper reports runtime and memory overheads but does not analyze asymptotic scaling with respect to sequence length L or diffusion steps T. Large scale deployment implications remain unclear.

4. There is no quantitative interpretation of the learned spline bases or frequency-domain analysis of the KAN functions. Without such analysis, the mechanism of improvement is still opaque.

5. Since some results of extensive baselines are cited from prior works rather than retrained under identical conditions, the comparison fairness remains slightly uneven.

### Questions
1. Appendix D defines $q_\phi(x_t|x,c)$ and $q_\phi(x_{t-1}|x_t,x,c)$. Are the KAN parameters $\phi$ shared across t or re-estimated per t? If shared, how does this affect flexibility; if not, how is stability maintained?

2. Section 3.1-3.4 apply the same diffusion backbone to forecasting, imputation and classification with different conditional masks. Are the conditioning strategies implemented within the same noise schedule or adjusted per task?

3. Appendix K measures runtime empirically, but for more formal confirmation, can you provide theoretical computational complexity in T and L?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes DiffKANformer, a conditional diffusion model integrating KAN for feature projection and a Diffusion KAN Transformer architecture for denoising in time series analysis. The method introduces a learnable KAN-based forward process and replaces MLP blocks in the denoiser with KAN layers. Extensive experiments across four major tasks (forecasting, imputation, classification, anomaly detection) and 29 datasets show consistent improvements over strong diffusion-based baselines. Theoretical formulation is clear and self-consistent, though mainly architectural rather than conceptual.

### Strengths
1.Clear motivation and complete methodological formulation.

2.Comprehensive experiments across multiple time series tasks and datasets.

3.Empirically demonstrates consistent improvements and stability.

4.Writing quality and reproducibility are both high.

### Weaknesses
1. The “theory” part (Sections 2.1–2.2) mainly restates the diffusion formulation with a KAN projection. There is no derivation or discussion showing why this design improves optimization or likelihood. The method is clearly written but not theoretically grounded.

2. Runtime results are briefly reported in Appendix Tables 14–15, but no corresponding FLOPs or parameter counts are given, and the discussion is not integrated into the main text.
As a result, the computational trade-offs of KAN remain unclear.

3. KAN layers reduce parameters but add spline computations, which likely increase runtime — yet this trade-off is not analyzed.

4. Recent diffusion foundations (e.g., TimeEdit 2024, Latent DiT 2024) are not compared.

5. The “unified framework” is mostly architectural; each task head is still trained separately.

Overall, the work feels more suitable for research exploration than industrial use, since scalability and efficiency remain unverified.

### Questions
1.Clarify theory in Sections 2.1–2.2.
Explain how the KAN projection affects the diffusion loss or variance schedule,
and show that the formulation recovers DDPM when KAN = Id and c = 0.
A short appendix derivation would make the theoretical part more convincing.

2.Add computational cost analysis.
Include a small table comparing DiffKANformer, DiT, and CnDiff in terms of parameters, FLOPs, and inference time.
This would clarify whether KAN’s higher runtime cost is justified by the accuracy gains.

3. Provide one targeted ablation.
For example, fix or freeze the KAN projection (or replace it with a linear map) to confirm that the gain truly comes from the learnable forward process.
This single ablation would strongly support the main claim.

4. (Optional) Briefly discuss possible runtime optimizations, such as precomputing spline bases or kernel fusion, to make KAN-based diffusion models more practical.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a diffusion-based time series framework that learns the forward (prediction) process via KAN-projection (combining the KANϕ(x, t) term and a learnable pre-component) and replaces the DiT MLP with a KAN block (DiT-KAN) under adaLN conditions.
This approach is evaluated across four tasks (prediction, interpolation, classification, anomaly detection) and multiple datasets, consistently reporting superior performance or SOTA-level results.
Key contributions include learnable forward diffusion tailored to temporal structure, architectural transition to KAN within DiT, and robustness demonstrated through extensive multi-task evidence.

### Strengths
- Combining learnable forward diffusion (KAN-projection) with the DiT-KAN backbone for time series.
- Extensive experiments across multiple datasets and ablation studies (e.g., KAN vs. MLP) highlight architectural advantages, particularly in classification/AD.
- High-level motivations and design choices are well-explained, and the intuitiveness of KAN-projection is supported by correlation analysis.

### Weaknesses
- For classification/AD, the combination method (weighting, schedule) of $L_{rec}$, $L_{diff}$, and class loss is unclear. An explicit formula, $λ$ value, and selection criteria should be provided.
- It appears only random point masking was used. Since block omissions, channel drops, MNAR, and irregular sampling are common in practice, it would be better to include these or discuss limitations.
- A comparison of parameters/FLOPs/memory/latency between DiT-KAN and DiT-MLP is needed, and discussing long sequence scalability and alternatives would be beneficial.

### Questions
- Were all baselines tuned with the same look-back grid, optimizer schedule, and early-stopping rules? If not, quantify the discrepancy and its impact.
- Any observed training instabilities from the learnable forward process? What mitigations helped?
- Sensitivity to spline order/knots and their interaction with diffusion steps T? Are gains robust across ranges?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 5

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
DiffKANformer is proposed as a conditional diffusion model for general time series analysis, integrating Kolmogorov-Arnold Networks (KANs) into both the forward diffusion process and the denoising architecture. The method introduces a KAN-based projection to capture complex feature correlations and replaces MLPs in the Diffusion Transformer (DiT) with KANs to better model long-term temporal dependencies through adaptive univariate functions. The authors evaluate the model across four core tasks—forecasting (8 datasets), imputation (6 datasets), classification (10 UEA datasets), and anomaly detection (5 benchmarks)—reporting state-of-the-art performance. Ablation studies are provided to justify the utility of KAN projection and the Diffusion KAN Transformer block, and the framework claims to be the first diffusion-based model to unify and excel in all four tasks.

### Strengths
1. The paper presents extensive empirical validation across a remarkably wide range of time series tasks and datasets, which is rare and commendable in diffusion-based time series literature.

2. Replacing MLPs with KANs in the DiT architecture is a conceptually interesting direction, especially given recent theoretical arguments about KANs’ superior function approximation for structured data.

### Weaknesses
1. The theoretical justification for the proposed KAN-based forward process is shallow; the derivation in Appendix D assumes a non-Markovian forward process but fails to rigorously analyze how this affects the tightness of the ELBO or the stability of training—critical omissions for a method claiming to “reduce the gap between true NLL and its variational approximation.”

2. The claimed “first unified diffusion model for all four tasks” is misleading; CSDI, TimeGrad, and CnDiff already handle multiple tasks (e.g., forecasting and imputation), and the classification/anomaly detection setups are trivial adaptations using off-the-shelf reconstruction or representation heads, which is not architectural innovations.

3. The ablation in Table 6 conflates the effect of KAN projection with the Diffusion KAN Transformer; no experiment isolates KAN projection alone with a standard DiT backbone, making the contribution attribution ambiguous.

4. Implementation details reveal that condition networks differ per task (dense layer vs. transformer), yet this architectural inconsistency is never discussed nor controlled in ablation, raising concerns that performance gains stem from task-specific design rather than the core DiffKANformer framework.

5. Despite claiming superior efficiency due to fewer parameters (0.5M), Table 14 shows DiffKANformer has significantly higher training time than CnDiff and mr-Diff on ETTh1, contradicting the efficiency narrative and suggesting immature KAN implementation, not architectural advantage.

6. The forward process introduces a learnable condition c and KAN projection, but c is never defined operationally, whether it’s a learned embedding, input statistic, or task-specific encoding remains ambiguous, rendering reproducibility questionable.

### Questions
Please address the concerns raised in Weaknesses 1–6.

### Soundness
1

### Presentation
1

### Contribution
1
