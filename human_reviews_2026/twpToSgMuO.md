# Are Time-Series Foundation Models Deployment-Ready? A Systematic Study of Adversarial Robustness Across Domains

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 2, 4, 4

## Abstract
Time-Series Foundation Models (TSFMs) are rapidly transitioning from research prototypes to core components of critical decision-making systems, driven by their impressive zero-shot forecasting capabilities. However, as their deployment surges, a critical blind spot remains: their fragility under adversarial attacks. This lack of scrutiny poses severe risks, particularly as TSFMs enter high-stakes environments vulnerable to manipulation. We present a systematic, diagnostic study arguing that for TSFMs, robustness is not merely a secondary metric but a prerequisite for trustworthy deployment comparable to accuracy. Our evaluation framework, explicitly tailored to the unique constraints of time series, incorporates normalized, sparsity-aware perturbation budgets and unified scale-invariant metrics across white-box and black-box settings. Across six representative TSFMs, we demonstrate that current architectures are alarmingly brittle: even small perturbations can reliably steer forecasts toward specific failure modes, such as trend flips and malicious drifts.
We uncover TSFM-specific vulnerability patterns, including horizon-proximal brittleness, increased susceptibility with longer context windows, and weak cross-model transfer that points to model-specific failure modes rather than generic distortions.
Finally, we show that simple adversarial fine-tuning offers a cost-effective path to substantial robustness gains, even with out-of-domain data. This work bridges the gap between TSFM capabilities and safety constraints, offering essential guidance for hardening the next generation of forecasting systems.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents a comprehensive evaluation of the adversarial robustness of Time-Series Foundation Models (TSFMs). Despite their growing adoption in high-stakes applications, the security of TSFMs under adversarial conditions remains underexplored. This work addresses that gap through a unified framework that evaluates various threat models, attack methods, and robustness metrics.

### Strengths
- The paper addresses the underexplored adversarial robustness of TSFMs, which are increasingly used in critical real-world applications.
- The study presents a unified framework covering diverse threat models, attack types, and metrics, offering a broad and systematic analysis of TSFM vulnerabilities.
- The evaluation provides actionable insights for model selection and defense in adversarial settings.

### Weaknesses
- **Outdated Attack and Defense Techniques**:
  Although the evaluation spans different goals, capabilities, and knowledge levels, the choice of attack and defense methods appears outdated and insufficient. The paper still relies on traditional computer vision attacks, with the most recent being SimBA from 2019. Similarly, the defense strategies used are no longer state-of-the-art. I suggest incorporating more recent adversarial attacks and defenses specifically tailored to time-series data.
  
- **Lack of Time-Series–Specific Insights and Misleading Claims**:
  The results echo patterns already well-documented in the computer vision literature, with few observations that highlight the unique characteristics of time-series forecasting. Some statements also appear overstated or even potentially incorrect:
  
  1. For instance, the claim “Failures are model-specific, with limited cross-model transfer” is not surprising, especially since the transfer attacks are seemingly conducted by directly applying white-box perturbations to other models, without any established transfer-enhancing strategies. The paper misses the opportunity to explore whether classic transferability techniques are effective for time-series models.
  2. Additionally, using the same $\epsilon$ across datasets is problematic because time-series data can have vastly different scales. This makes cross-dataset comparisons unfair and weakens the conclusion that certain datasets are more vulnerable. A per-dataset normalization or scale-aware perturbation constraint would be more appropriate.
- **Ambiguous Writing and Visualization**:
  Several parts of the paper are unclear or confusing, making it difficult to assess the actual threat. For example, in **Figure 1**, key parameters (like $\epsilon$) are not specified. The perturbation visually appears to be much larger than $\epsilon$ = 1 used in **Table 2**, suggesting inconsistencies between figures and main experiments. **Figure 3** is similarly vague—only a range of [0.25, 5] is mentioned for the budget, but no specific $\epsilon$ values are plotted. I recommend the authors improve the clarity and specificity of all experimental settings and avoid vague or potentially misleading descriptions.

### Questions
- What is q in Figures 1 and 3? This variable is never defined in the main text, which leaves the reader confused.
  
- It is unclear whether the values of $\epsilon$ {0.25, 0.5, 0.75, 1} and r {0.25, 0.5, 0.75, 1} in Table 2 are paired (i.e., (0.25, 0.25), (0.5, 0.5), etc.) or fully cross-matched. If the latter, then averaging all results may obscure the effect of low perturbation regimes, since the strongest ($\epsilon$ = 1, r = 1) combinations dominate the average. Please consider presenting disaggregated results for different perturbation levels to clarify how performance degrades under small perturbations.
  
- What is the perceptual or practical impact of different perturbation magnitudes? Since Figure 1 and Figure 3 lack explicit parameter annotations, the perturbations—especially in Figure 3(b)—appear visually large (perhaps >1000 in value). I suggest incorporating explicit metrics for perturbation imperceptibility to better support claims of realistic risk.
  
- The phrase “the maximum change per step” is likely incorrect, since the method appears to apply a single-step change. Please remove “per step” for accuracy.
  
- The current black-box attack is purely query-based. Have the authors considered traditional transfer-enhancing methods (e.g., input diversity, ensemble gradients, surrogate fine-tuning)? It would be valuable to assess whether such techniques, commonly used in vision, also work for time-series models.
  
- The true risk posed by these attacks remains unclear. While the RED score may indicate high vulnerability, the robustness curves suggest that small perturbations cause minimal absolute error. Would such small distortions result in meaningful or observable consequences in downstream applications? The paper would benefit from a clearer discussion of what constitutes a “successful” or threatening perturbation in real-world settings.

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper presents an empirical study on the adversarial robustness of multiple Time-Series Foundation Models (TSFMs)—TimesFM, TimeMoE, UniTS, Moirai, Chronos, and TabPFN-TS—under white-/black-box and targeted/untargeted attacks.
It proposes a unified evaluation framework with the Relative Error Deviation (REDE) metric, a mixed-norm perturbation budget (ℓ₀ + ℓ∞), and two defenses (inference-time smoothing and latent adversarial training LAT).
Experiments cover six models and eight datasets, showing that minor perturbations can cause large forecast errors and that LAT improves worst-case robustness.

### Strengths
1. The unified threat model (goal / capability / knowledge) and hybrid-norm constraint are technically well-defined and well-motivated for time-series data.

2. The work systematically examines model-specific failure modes, horizon sensitivity, and context-length effects.

3. Defense results are quantitatively compelling. In-domain LAT improves worst-case NMAE up to 10× under PGD and generalizes well out-of-domain.

4. Reproducibility statement and released code enhance reliability.

### Weaknesses
**1. Over-claimed novelty and limited contribution boundary**

The paper repeatedly claims to be **“the first large-scale, systematic robustness evaluation of TSFMs.”**
However, two peer-reviewed works have already addressed adversarial robustness of TSFMs directly:

**Adversarial Vulnerabilities in Large Language Models for Time Series Forecasting - AISTATS 2025**
Performs a systematic, cross-model and cross-dataset robustness analysis including TSFM such as TimeGPT, demonstrating that small, structured perturbations cause significant and controllable forecast distortions.

**Temporally Sparse Attack for Fooling LLMs in Time Series Forecasting - ICML 2025 workshop**
Introduces a cardinality-constrained optimization attack that manipulates only ≈ 10 % of time steps while severely degrading forecasts of LLM-based TSFMs (including TimeGPT), directly exposing their adversarial weaknesses.

Because both prior studies explicitly involve TSFMs and directly reveal their adversarial vulnerabilities, the core finding of this submission is no longer novel.
The main difference lies in the model family extension (from LLM-TSFMs to other pretrained forecasting models), which constitutes an incremental replication, not a conceptual breakthrough.
Consequently, the repeated “first large-scale, systematic robustness evaluation of TSFMs” statements appear over-claimed and should be toned down to avoid misleading readers.

**2. Limited methodological innovation**

The two proposed defenses—moving-average smoothing and LAT—are straightforward adaptations of known techniques with modest empirical tuning.
No comparison is provided against input-space adversarial training, noise-based defenses, or detection mechanisms, limiting methodological novelty.

**3. Evaluation gaps**

The attack suite omits universal or adaptive baselines (e.g., AutoAttack-style ensembles).

No comparison to traditional forecasting models (LSTM, TCN, Informer), leaving unclear whether TSFMs are uniquely fragile or simply representative of general deep forecasting vulnerabilities.

Mechanistic insights remain descriptive: while horizon-boundary and context-length sensitivities are observed, there are no controlled ablations (e.g., disabling patchification or varying decoder type) to establish causality.

### Questions
1. Compare LAT with input-space adversarial fine-tuning under equal training cost and report clean-accuracy trade-offs.

2. Have you tested universal or adaptive attacks? It will be intresting to see these attacks' performance on TSFM.

3. Can you conduct controlled ablations (e.g., without patchification or using alternative decoding heads) to validate causal explanations for observed vulnerabilities?

4. Add non-foundation baselines to clarify whether fragility is specific to TSFMs.

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
5

### Summary
This submission presents an adversarial attack framework for zero-shot time series forecasting models, designed to assess the robustness of time-series foundation models. Two types of attacks are introduced: a white-box attack based on the Fast Gradient Sign Method (FGSM) and a black-box attack employing zero-order optimization. To improve model robustness, the study further proposes two defense strategies: a filter-based preprocessing defense and an adversarial training–based defense.

### Strengths
The research topic is important and timely. The vulnerability and robustness of time series foundation models remain underexplored.

The experimental design is comprehensive. Six representative models are evaluated across eight diverse datasets, providing convincing evidence to support the study’s findings.

### Weaknesses
The primary weakness of this submission lies in its limited technical novelty. The manuscript mainly applies existing adversarial attack and defense techniques to zero-shot time series forecasting models, without introducing fundamentally new methodologies. Consequently, the proposed attacks can largely be mitigated by existing defense mechanisms.

More specifically:

1. **Relation to Prior Work**: The submission does not clearly articulate its relationship or distinction from prior studies. For example:
  [1] introduced FGSM-based white-box attacks and proposed filter-based and adversarial fine-tuning defenses for time series forecasting;[2] presented a zero-order optimization (SPSA)-based black-box attack for forecasting models; and [3] developed targeted, gradient-free black-box attacks specifically for zero-shot, LLM-based time series forecasting models. 

2. **Novelty and Significance**: The reliance on established adversarial attack and defense methods reduces the overall contribution of the paper. The presented framework does not demonstrate a substantial methodological advancement beyond existing literature.

3. **Insights on Foundation Models**: The work does not sufficiently uncover new challenges or unique vulnerability patterns specific to time series foundation models under adversarial conditions.

4. **Comparative Robustness Analysis**: A direct robustness comparison between zero-shot time series foundation models and conventional forecasting models is missing. Including such baselines would strengthen the evaluation and clarify whether foundation models exhibit distinct robustness characteristics.

**Reference**

[1] Liu, Linbo, et al. "Robust Multivariate Time-Series Forecasting: Adversarial Attacks and Defense Mechanisms." ICLR (2023).

[2] Zhu, Lyuyi, et al. "Adversarial diffusion attacks on graph-based traffic prediction models." IEEE Internet of Things Journal 11.1 (2023): 1481-1495.

[3] Liu, Fuqiang, et al. "Adversarial Vulnerabilities in Large Language Models for Time Series Forecasting." International Conference on Artificial Intelligence and Statistics. PMLR, 2025.

### Questions
This submission applies existing adversarial attack methods to evaluate the vulnerabilities of time series foundation models and employs established adversarial defenses to mitigate these attacks. Incorporating new insights or methodological innovations would substantially enhance the significance and contribution of the work. For example, it remains unclear what new attack designs are specifically tailored to the characteristics of time series foundation models, and what unique vulnerability patterns these models exhibit beyond those already observed in conventional time series forecasting models.

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper systematically evaluates TSFMs under a unified adversarial framework spanning goals (untargeted/targeted), capabilities (hybrid ℓ₀/ℓ∞ budget), and knowledge (white-box/black-box), showing that even small, structured perturbations can reliably steer forecasts (e.g., flips, drifts, scaling) across domains. TSFMs show pervasive vulnerabilities (e.g., TimesFM particularly sensitive under PGD); adversarial examples often don’t transfer well across models (model-specific failure modes); points near the forecast horizon are most vulnerable; longer contexts improve clean accuracy but amplify attack impact.

### Strengths
1. Clear, comprehensive threat modeling & eval setup: Covers white-box (PGD) and black-box (SimBA/ZOO), targeted and untargeted goals, with unified robustness metrics across six TSFMs and eight datasets.

2. Finds pervasive but model-specific vulnerabilities and quantifies factors that modulate attack success (context length, attack location, model size).

### Weaknesses
1. Some robustness signals may reflect gradient obfuscation: MoE-style models appear PGD-resistant, but single-step and query-based attacks still work.

2. Technical contribution seems to be limited. I don't like to use this argument for paper review but vulnerability to adversarial attacks are well-known in the entire ML community.

### Questions
This is a pretty comprehensive study but the technical contribution seems to be limited. Defenses using smoothing to me is still vulnerable to adaptive attacks, and the LAT is also not new.

### Soundness
3

### Presentation
3

### Contribution
2
