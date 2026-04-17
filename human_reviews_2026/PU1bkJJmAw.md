# RippleNet: Learning Causal Maritime Dynamics for Forecasting Warning-Induced Ripple Effects

- Decision: Reject
- Scores: 2, 6, 8, 4

## Abstract
Maritime transportation networks, including cargo vessels, tankers, and passenger ships, are critical to global trade but remain highly vulnerable to disruptions such as extreme weather or security alerts. These events often trigger ripple effects, with cascading impacts extending far beyond the initial warning zones. Traditional spatio-temporal forecasting methods struggle to capture these dynamics due to their reliance on correlations rather than causal reasoning, particularly in maritime contexts. To address this challenge, we propose RippleNet, a novel causal spatio-temporal framework that explicitly models causal dependencies to predict port-to-port flow disruptions under warning-induced ripple effects. RippleNet comprises three key components: (i) a neural deconfounding module that employs causal adjustment techniques to disentangle genuine causal effects from spurious correlations, addressing confounding factors that arise when warnings simultaneously affect multiple maritime operational aspects, (ii) a continuous-time ODE module that simulates the propagation of disruptions across vessel networks, and (iii) LLM-generated warning vectors that quantify the multidimensional operational impacts of various warning types. Experiments on maritime flow datasets from East Asia and Northwest Europe show that RippleNet significantly outperforms state-of-the-art baselines under warning scenarios, while offering interpretable causal insights into heterogeneous vessel flow behavior.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes RippleNet, a causal spatio-temporal framework for forecasting warning-induced ripple effects in maritime networks. It combines a neural deconfounder, an ODE-based propagation module, and LLM-generated warning vectors. Experiments on real-world datasets show improved forecasting performance over existing baselines.

### Strengths
1. This work has strong potential impact in the domain of maritime logistics and global trade. The proposed framework, if validated and deployed, could significantly improve the understanding and management of warning-induced disruptions in international shipping routes.

2. The figures are clear and well-organized, and the inclusion of map-based visualizations greatly enhances interpretability.

### Weaknesses
1. The reviewer’s primary concern is that the paper’s contributions to the representation learning and machine learning community appear limited. The work emphasizes domain-specific applications (maritime forecasting) rather than advancing general ML methodology. It may be more suitable for a journal, conference, or workshop focused on maritime operations or applied geospatial modeling.

2. The paper lacks any discussion or quantitative analysis regarding computational efficiency, including model size, parameter count, training speed, and GPU memory usage. As the proposed model involves an ODE-based causal propagation module, its computational overhead relative to other baseline models should be reported for a fair assessment.

3. The hyperparameter settings of baseline models (e.g., STGCN, GraphWaveNet, PDFormer, CaST) are not described. It is unclear whether they were tuned on validation sets or directly adopted from the respective papers, which limits the reproducibility of the reported comparisons.

4. The paper’s reliance on LLM-generated warning vectors raises concerns about reproducibility, transparency, and robustness. While Appendix A.2 provides detailed prompt templates and a rationality analysis, the use of a proprietary LLM (ChatGPT-4o) introduces potential subjectivity and temporal variability in outputs. The validation is primarily qualitative; no quantitative reliability assessment (e.g., inter-model consistency or expert annotation agreement) is provided. Without rigorous verification, there remains a risk that hallucinations or latent biases in the LLM outputs could propagate into the causal modeling pipeline, undermining its interpretive validity.

5. The sensitivity and ablation analyses are limited in scope. Figure 5 only examines two hyperparameters (λ and α). A more comprehensive robustness analysis—considering solver tolerances, LLM threshold values, and state vector dimensionality—would strengthen the empirical section. Similarly, the ablation study isolates only top-level modules, leaving finer-grained dependencies (e.g., within the LLM vector representation) unexplored.

6. Theoretical validation of the causal adjustment component remains underdeveloped. Although the equations are sound and conceptually aligned with back-door adjustment, there is no formal discussion of identifiability, convergence, or conditions under which the deconfounder reliably works. Such analysis would enhance the methodological rigor of the paper.

### Questions
1. Could the proposed framework be extended or tested on additional regions, such as American or African maritime networks, to demonstrate cross-regional generalizability?

2. Is RippleNet applicable beyond the maritime transportation domain, for example, to air traffic, railway, or supply chain networks affected by external disruptions?

3. How sensitive is the model’s performance to the quality and consistency of the LLM-generated warning vectors? Have the authors compared outputs across different LLMs (e.g., Claude, Gemini) or evaluated agreement with expert annotations?

4. Could the authors provide quantitative details on model efficiency, such as parameter count, runtime, or GPU memory consumption, to contextualize the cost-performance tradeoff against baseline models?

5. Have the authors considered introducing formal causal validation metrics, such as Average Causal Effect estimation or counterfactual consistency tests, to strengthen the causal claims?

6. Would a more extensive hyperparameter and solver sensitivity analysis, for example, varying the ODE tolerance or hidden dimensionality, alter the observed performance trends?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces RippleNet, a causal spatio-temporal forecasting framework designed to model and predict warning-induced disruptions (“ripple effects”) in maritime transportation networks. The authors identify a fundamental limitation of existing traffic-forecasting methods — their dependence on correlations rather than causal mechanisms — which leads to failure under anomalous warning scenarios (e.g., typhoons, security alerts).

RippleNet integrates three key components:
1. Neural Deconfounder Block — performs causal adjustment via neural back-door control to disentangle genuine effects of warnings from spurious correlations.
2. Continuous-Time ODE Propagation Module — models how disruptions spread temporally and spatially across port networks.
3. LLM-Generated Warning Vectors — quantifies multidimensional warning impacts from textual maritime bulletins using large-language-model prompts.

Experiments on East Asia and Northwest Europe maritime flow datasets demonstrate consistent gains (up to 19.6% MAE improvement over strong baselines such as PDFormer and STAEformer). Ablation and case studies support the contribution of each module and show interpretable causal propagation patterns.

### Strengths
- Originality: Innovative combination of causal inference, ODE modeling, and LLM-based treatment quantification; new causal formulation tailored for maritime ripple-effect prediction.


- Quality: Strong empirical validation with diverse baselines, ablations, and case studies; interpretable visualizations of causal effects (Fig. 7).


- Clarity: Comprehensive methodological exposition, including formal causal graph and mathematical derivations.


- Significance: Provides a blueprint for applying causal deep learning to real-world network resilience problems beyond transportation (e.g., logistics, climate-impact forecasting).


- Reproducibility: Code and datasets are described in detail, supporting transparency.

### Weaknesses
- Causal Validity of LLM Inputs: The reliance on LLM-generated binary treatment vectors introduces potential noise and bias; authors should analyze sensitivity to prompting or threshold choices.


- Generalization Beyond Maritime Domain: While experiments are strong, additional non-maritime datasets (e.g., air-traffic or logistics networks) could demonstrate broader applicability.


- Complexity vs. Interpretability: RippleNet’s multi-module architecture may challenge operational deployment; authors could discuss computational cost and parameter efficiency.


- Ablation Depth: The ablation focuses on module removal; further analysis on causal disentanglement metrics or counterfactual validation would reinforce claims.


- Minor Clarity Issues: Dense notation (especially in §4) and lengthy references may obscure key insights for readers unfamiliar with causal ODEs.

- Typographic Consistency: The manuscript occasionally uses incorrect opening quotation marks (e.g., ”ripple effect”, ”Nanmadol”), where the closing quote glyph is mistakenly used at the start. While minor, this distracts from an otherwise polished presentation and should be corrected.

### Questions
1. How sensitive is performance to the binarization thresholds (τᵢ) and the choice of LLM prompt design?


2. Could the authors provide quantitative evidence that the learned deconfounder truly captures causal rather than correlational signals (e.g., intervention or counterfactual tests)?


3. What are the computational costs of the ODE solver, and how does it scale with network size?


4. Could RippleNet be adapted for continuous (non-binary) treatment variables or multi-modal warning inputs (e.g., satellite imagery, textual logs)?


5. How do the authors plan to address domain shift when applying to unseen maritime regions or future warning types?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors propose a novel causal spatio-temporal framework that explicitly models causal dependencies to predict port-to-port flow disruptions under warning-induced ripple effects. The framework leverages knowledge about warnings (e.g., extreme weather or security alerts) to learn from them and lead to better outcomes. Through several experiments, the authors show that the framework effectively learns causal relationships and that such causal understanding of the context leads to better outcomes than those obtained by correlation-based models. The authors test their framework on two real-world datasets, achieving SOTA performance.

### Strengths
The authors propose a novel causal spatio-temporal framework that explicitly models causal dependencies to predict port-to-port flow disruptions under warning-induced ripple effects. They compare their approach against multiple baseline models, surpassing them in performance and achieving SOTA results. They conduct ablation experiments to understand how specific components of the proposed framework contribute to overall performance. Furthermore, they perform hyperparameter analysis and illustrate the usefulness of the proposed framework on a use case. We consider the manuscript to be clearly written and well articulated. The outcomes are likely to have a big impact, as they can directly influence decision-making on logistics worldwide. In addition, the authors created two datasets for assessing the framework. While grounded in publicly available datasets, releasing them would constitute an additional contribution.

### Weaknesses
While we consider the work to be solid, we would like to highlight some improvement opportunities:

- The framework requires that warning information be translated into vectors to learn the causal relationships between events and how they translate into outcomes. This translation is done by an LLM, and while a case is described to assess the quality of the outcome, it remains unclear how well it generalizes across many scenarios. Furthermore, it is not clear on what grounds the LLM assigns the scores and whether these are consistent for the same scenario over time, and how robust they are, e.g., to how information is presented to them (a frequent issue as reported by e.g., Leidinger, Alina, Robert Van Rooij, and Ekaterina Shutova. "The language of prompting: What linguistic properties make a prompt successful?." arXiv preprint arXiv:2311.01967 (2023).). No insights were provided on the prompts used to generate these scores or on comparisons across different LLMs.

 - The authors describe a module tasked with controlling for confounding. Nevertheless, little detail is given on how causal relationships are modeled, the embeddings used to model relevant confounders' information, and how they assess the correctness of the information learned.

### Questions
1- "This diversity creates challenges for neural networks requiring structured numerical inputs. We address this through large language model-based quantification using domain-specific prompts" -> (i) How reliable are the LLM-generated scores? (ii) In what information are the LLM-generated uncertainty scores grounded? (iii) The authors report average values in Fig. 6a: did they run them several times as to understand consistency and distribution of values? what was the magnitude of the standard deviations? (iv) What kind of models are being used for this purpose? (v) Could this be replaced with an alternative method that, based on the information reported, would provide deterministic estimations for each of the vector values? (vi) Did the authors' study on whether the LLM-generated scores for spatial and duration impact were accurate based on the LLM assessment and the historical data limited to the rationality analysis reported in the Appendix regarding Fig. 6a? (vii) Did the authors cross-check the assessments were accurate ensuring the LLM did not have knowledge about those specific past events (having been trained on such data and therefore creating the illusion of accurate estimation - looking into the future)?, (viii) What prompts did the authors use to extract the information?, (ix) How did the authors test for the robustness of the prompt results to e.g., how the same information reported in different terms (linguistic variations) and order affect the outcomes?, (x) Did the authors consider multiple LLM models and which performed best?

 2- "The threshold values are set [...] based on empirical percentiles and domain knowledge." -> How do the authors ensure proper matching between LLM-generated scores and domain knowledge to arrive at relevant score thresholds?

 3- "and a publicly available maritime warning dataset" -> While the authors provide details regarding the dataset used for East Asia, we could not find details about the one used for Northwest Europe. We encourage the authors to provide a reference to the dataset they used for Northwest Europe.
 
 4- The authors mention the deconfounding block "controls for confounding by conditioning on the sufficient set of confounding mediators M = {X, V} and spatial-temporal embeddings" and that "the confounding mediators M are modeled through dedicated encoders". We would appreciate it if the authors could provide some insights into (i) how are causal relationships modeled or the authors consider they are implicitly modeled based on the warning vectors and model learning on how these correlate with the outcomes?, (ii) how accurate is to model the confounding mediators with a fixed set of embedding types?, (iii) what kind of embeddings are used in each case?, (iv) are the embeddings modeled using separately trained networks? In such a case, we would appreciate some insights about them.

 5- Table 5: (i) what do the different colour codes mean? (e.g., orange, red), (ii) what is the meaning of the underlined and bolded results? 

 6- Figure 6: We encourage the authors to use a monochromatic scale for their heatmaps to make it more friendly toward color-blind people.

 7- We encourage the authors to test whether the difference in performance noticed among the best models is statistically significant.

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
3

### Summary
The paper proposes RippleNet, a causal spatio-temporal framework for forecasting maritime flow disruptions triggered by warnings (weather, security, etc.). Key pieces are: (i) a neural deconfounder intended to implement back-door adjustment between warning vectors and future flows, (ii) a continuous-time ODE module to simulate ripple propagation across the route graph, and (iii) LLM-generated warning vectors that quantify eight impact dimensions and are then binarized.

### Strengths
The ripple-effect phenomenon is well illustrated. The combination of a causal adjustment block with an ODE propagation layer is interesting and plausibly well-suited to maritime dynamics.

### Weaknesses
1. It’s unclear which variables are true confounders versus mediators or effects, so conditioning on them may introduce bias. The paper doesn’t clearly spell out the causal assumptions or show diagnostics that the “deconfounding” actually worked.
2. What is the intervention, exactly? The “treatment” is an LLM-derived warning score, not the original real-world event or action. It’s unclear what real intervention corresponds to changing that score, so the estimand is ambiguous.
3. Continuous warning scores (severity 0–100) are thresholded into on/off flags, which discards useful intensity information. There’s no sensitivity analysis for the chosen thresholds, and a continuous version isn’t evaluated.
4. Many comparison models appear not to use the same warning features, which can unfairly advantage the proposed method. Missing are baselines that take the identical inputs and simple causal baselines for reference.
5. Train/validation/test splits aren’t fully described (risk of temporal or spatial leakage). Results lack confidence intervals or significance tests, and there’s little breakdown by warning periods, subgraph density, or vessel type.
6. The distance metric over latitude/longitude is simplistic for the globe; distance and adjacency may be “double-counted.” The paper also doesn’t clearly define how “effect magnitude” is computed for interpretability plots.
7. Prompts and inference settings aren’t fully pinned down, and outputs don’t appear to be cached—raising determinism and reproducibility concerns (and potential leakage from model prior knowledge).
8. Training and inference time overhead of the ODE component isn’t compared against simpler attention-only baselines, so operational cost is unclear.

### Questions
1. The paper aims to estimate P(Y∣do(B)) (Eq. 3, p. 5) by conditioning on M={X,V} and E. However, it is not fully clear whether X (historical flows) and V (adjacency/context) are pure pre-treatment confounders for the B→Y relation or mediators/colliders along paths from the raw event  A to Y. Figure 4 (p. 4) labels them as “mediates/backdoor/confounds” simultaneously, which is conceptually inconsistent. If any part of M is post-treatment w.r.t. B, conditioning introduces post-treatment bias. Please formalize the DAG and explicitly state ignorability/positivity/stable-unit assumptions.

2. The “neural back-door adjustment” is implemented via learned gates rather than an explicit balancing/weighting or outcome-regression with diagnostics. How do you verify that the learned representation achieves conditional exchangeability in practice (e.g., balance checks, negative controls, sensitivity analysis)?

3. You define the LLM-derived, 8-dimensional “warning vector” B (Fig. 3, p. 4) as the treatment, but it is derived from the event A using an external model. Conceptually, B is then a mediator/measurement of A, not an exogenous action. What does do(B) mean if in the real world we intervene on A (e.g., warnings released/not released)? This matters for the causal estimand and whether the back-door applies. A front-door or measurement-error perspective might be more principled in this case.

4. The eight LLM scores in [0,100] are discretized to binary B∈{0,1}^8 (p. 15). This loses intensity information and can worsen positivity. Please evaluate (i) continuous or ordinal treatments (no thresholding) and (ii) threshold-sensitivity plots for τ_i​ (the appendix lists fixed τ on p. 15).

5. Table 1 (p. 7) compares against strong STGNNs, but it appears most baselines do not consume warning features (or LLM vectors). This favors RippleNet. Please add baselines with warning inputs: e.g., GraphWaveNet/GMAN/PDFormer augmented with the same warning vectors (binary and continuous). Also include simple causal baselines (e.g., TARNet/CFR-Net-style two-head regressors with warnings as treatments).

6. It is unclear how the train/val/test temporal splits are constructed and whether there is leakage via seasonality or adjacency (pp. 6–7; Appx p. 13). Please clarify splits and report per-segment metrics: “warning” vs “no-warning” periods, sparse vs dense subgraphs, and by vessel type.

6. Report uncertainty (std across 5 runs is mentioned but not shown as CIs), and run significance tests on Table 1.

7. Provide the computational cost and wall-clock overhead of ODE solvers versus attention-only baselines.

8. Distance ∥pi​−pj​∥2​ (Eqs. 7–8, p. 6) on lat/long can distort true geodesic distances. Please switch to great-circle distance or note a projection.

9. Rij​(t) and ρij​(t) both use adjacency and distance; discuss double-counting and identifiability of their contributions.

10. Define how “causal effect magnitude” is computed for Fig. 7 (p. 9)—gradient-based, ablation, or interventional difference?

### Soundness
3

### Presentation
3

### Contribution
3
