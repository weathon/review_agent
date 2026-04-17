# Pre-training Epidemic Time Series Forecasters with Compartmental Prototypes

- Decision: Reject
- Scores: 6, 4, 2, 4

## Abstract
Accurate epidemic forecasting is crucial for outbreak preparedness, but existing data-driven models are often brittle. Typically trained on a single pathogen, they struggle with data scarcity during new outbreaks and fail under distribution shifts caused by viral evolution or interventions. However, decades of surveillance data from diverse diseases offer an untapped source of transferable knowledge. 
To leverage the collective lessons from history, we propose CAPE, the first open-source pre-trained model for epidemic forecasting.
Unlike existing time series foundation models that overlook epidemiological challenges, CAPE models epidemic dynamics as mixtures of latent population states, termed compartmental prototypes. It discovers a flexible dictionary of compartment prototypes directly from surveillance data, enabling each outbreak to be expressed as a time-varying mixture that links observed infections to latent population states. To promote robust generalization, CAPE combines self-supervised pre-training objectives with lightweight epidemic-aware regularizers that align the learned prototypes with epidemiological semantics. On a comprehensive benchmark spanning 17 diseases and 50+ regions, CAPE significantly outperforms strong baselines in zero-shot, few-shot, and full-shot forecasting. This work represents a principled step toward pre-trained epidemic models that are both transferable and epidemiologically grounded.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces CAPE (Compartmental Prototype Epidemic forecaster), a novel pre-trained model for epidemic time series forecasting. CAPE encodes each outbreak as a mixture of latent compartmental prototypes, capturing hidden epidemiological states (like susceptibility or immunity) and allowing adaptation across diseases and geographies.

### Strengths
The introduction of compartmental prototypes provides a flexible and epidemiologically grounded representation of disease progression.

The paper includes robust experiments, ablations, simulation studies, and analysis of distribution shift and representation quality, and CAPE shows reasonable improvements beyond baselines.

### Weaknesses
While compartments are learned, their interpretability (in terms of epidemiological meaning) is limited and mostly illustrated by t-SNE and CMD

The optimization involves EM-like updates with contrastive and reconstruction losses, which may hinder scalability in real-time applications.

### Questions
Can the authors provide concrete examples of how the learned compartments align with known epidemiological states (e.g., infectious vs. latent)?

Given the iterative EM-style updates and contrastive training, how feasible is this model for real-time epidemic response?

How would CAPE perform on novel disease dynamics that are not well captured in the training distribution (e.g., diseases with complex zoonotic spillovers)?

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
5

### Summary
This work presents a representation learning approach for epidemiological time-series modeling, aiming to improve forecasting performance using observational data. The authors introduce CAPE, an algorithm designed to estimate R0 within an SEIR framework and to enable data augmentation strategies that enhance generalization. Experiments are conducted on diverse and extensive datasets, and the results demonstrate strong performance in zero-shot, few-shot, and transfer settings across multiple benchmarks.

### Strengths
1. The problem formulation is well-founded and accurately captures the key challenges in epidemiological modeling.
2. The proposed algorithm for estimating R0 is convincing, supported by a sound theoretical justification. The experiments are comprehensive, with diverse benchmarks and analyses that demonstrate strong generalization.

### Weaknesses
1. The work does not provide strong evidence that it effectively addresses non-stationarity in time-series forecasting. The method appears reliable only when the forecasting horizon is shorter than or comparable to the historical window, which does not reflect the harder non-stationary setting.
2. The approach is heavily dependent on the conventional SEIR model and its simulations. If the method relies on such domain-specific priors, it raises the question of why more principled physics-informed models or efficient simulation strategies were not explored. The added complexity of CAPE is not clearly justified.
3. The set of baselines is limited. Several established forecasting methods that incorporate physics-informed modeling or advanced representation learning are missing from the comparison.
4. The paper claims to perform representation learning, but does not provide any analysis demonstrating the quality or usefulness of the learned representations. There is no assessment of feature extraction, embedding structure, or interpretability.

### Questions
1. Forecasting horizon remains a key issue. Non-stationarity becomes significant when the prediction window exceeds the length of the historical context. The current experiments use much shorter horizons, making the task more stationary and therefore easier than real-world scenarios. 
2. Variation in time-series length during pretraining is not well addressed. In practice, sequences have different lengths, and during evaluation the forecasting horizon must be aligned for inference. The paper provides little clarification on how these mismatches are handled.
3. The representation learning algorithm shows limited advantage. If the learned dynamics are mostly linear (from SEIR ordinary differential equations), simple mix-up could already generate millions of synthetic series for pretraining, which is already proposed. It is unclear why CAPE is necessary or what unique advantage it provides beyond existing mix-up methods. It is also necessary to compare with the work Wang, D., Zhang, S., & Wang, L. (2021, May). Deep epidemiological modeling by black-box knowledge distillation: an accurate deep learning model for COVID-19. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 35, No. 17, pp. 15424-15430).

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper aims to learn an epidemiological foundation model for predicting the spread of infectious diseases. It proposes a pre-training framework for epidemic forecasting models that integrates mechanistic guidance through the Next Generation Matrix (NGM) and a contrastive learning objective to improve generalization across diseases and regions.

### Strengths
(1) The adaptation of the Next Generation Matrix (NGM) method to improve the deep learning model looks interesting.

(2) There has been a non-trivial effort to engage with the literature in mathematical epidemiology, which is commendable.

(3) Paper presentation is good.

### Weaknesses
(1) The paper presents ideas and theoretical expressions that have already appeared in prior work but fails to cite the corresponding sources, which could mislead a less careful reader into thinking they are novel contributions. For example, the proof of the lower and upper bounds of the Next Generation Matrix (NGM) method is not original – it is a standard linear-algebra result applied to the classical NGM formulation of R_0. However, since the proof is provided without any citation, readers may assume it is the authors’ contribution. What appears to be new in this paper is how those bounds are used as a differentiable regularizer for R_0 derived from the model’s learned latent compartments, but this distinction should be made explicit.

(2) A similar issue arises with the idea of using neural networks to map to compartments in an epidemiological model, which has been explored in several previous studies [1]–[4], all of which are omitted from citation. In particular, the monotonicity loss introduced by the authors is strikingly similar to that proposed in [2], yet no reference is provided.

(3) I believe the only genuine technical contribution of this paper lies in the integration of the NGM framework. The remaining components are minor adaptations of well-known techniques that offer limited new insight: for instance, the use of contrastive learning guided by the mechanistic model to determine positive and negative samples. Unfortunately, the ablation results indicate that both the NGM component and the contrastive learning loss contribute only marginal improvements. Overall, this paper can be seen as a nice synthesis of several existing approaches, presenting ideas that may appear novel but are, in fact, extensions of prior work — often without the proper citations that would clarify this connection.

(4) The experimental setup appears to have potential flaws. In epidemic forecasting, it is standard practice to evaluate model performance in an online forecasting setting to better reflect real-world conditions. However, the authors apply this setup only in the appendix (Section A.8), while all main results rely on an offline evaluation. This raises concerns about the practical utility of the proposed method, particularly since ARIMA outperforms it in short-term forecasting (1 to 4 weeks), the most common use case in real-world applications (see [5]). It is also unclear why the authors chose to focus on 1- to 16-week forecasting horizons for the main experiments, while 1-4 is the most relevant.

[1] Wang, L., Adiga, A., Chen, J., Sadilek, A., Venkatramanan, S. and Marathe, M., 2022, June. Causalgnn: Causal-based graph neural networks for spatio-temporal epidemic forecasting. In Proceedings of the AAAI conference on artificial intelligence (Vol. 36, No. 11, pp. 12191-12199).

[2] Rodríguez, A., Cui, J., Ramakrishnan, N., Adhikari, B. and Prakash, B.A., 2023, June. Einns: epidemiologically-informed neural networks. In Proceedings of the AAAI conference on artificial intelligence (Vol. 37, No. 12, pp. 14453-14460).

[3] Kargas, N., Qian, C., Sidiropoulos, N.D., Xiao, C., Glass, L.M. and Sun, J., 2021, May. Stelar: Spatio-temporal tensor factorization with latent epidemiological regularization. In Proceedings of the AAAI conference on artificial intelligence (Vol. 35, No. 6, pp. 4830-4837).

[4] Gao, J., Sharma, R., Qian, C., Glass, L.M., Spaeder, J., Romberg, J., Sun, J. and Xiao, C., 2021. STAN: spatio-temporal attention network for pandemic prediction using real-world evidence. Journal of the American Medical Informatics Association, 28(4), pp.733-743.

[5] Cramer, E.Y., Ray, E.L., Lopez, V.K., Bracher, J., Brennen, A., Castro Rivadeneira, A.J., Gerding, A., Gneiting, T., House, K.H., Huang, Y. and Jayawardena, D., 2022. Evaluation of individual and ensemble probabilistic forecasts of COVID-19 mortality in the United States. Proceedings of the National Academy of Sciences, 119(15), p.e2113561119.

### Questions
Following weakness (2): The manuscript references several deep learning approaches for epidemic forecasting ([6]–[8]); however, it remains unclear why the authors consider it unnecessary to include direct comparisons with these methods (e.g., in their main results, table 1). In particular, for [6], the authors state that corresponding results are provided in the appendix, yet these results are not present. Furthermore, the omission of works [1]–[4], which address closely related ideas, further limits the completeness of the evaluation. The authors are encouraged to clarify the rationale for excluding these baselines from conceptual discussion and/or empirical comparison.

Following weakness (2): The papers CausalGNN [1] and EINNs [2] present ideas that appear closely related to those introduced in the current manuscript. In those works, the neural network predicts the population in each compartment of a mechanistic model, and the authors of EINNs also introduce a monotonicity loss similar to the one described in your Equation (4). Could the authors elaborate on the key differences between their proposed model and these approaches? Additionally, it would be important to clarify why this prior work was omitted from the discussion of related literature.

[6] Adhikari, B., Xu, X., Ramakrishnan, N., and Prakash, B. A. Epideep: Exploiting embeddings for epidemic forecasting. In Proceedings of the 25th ACM SIGKDD international conference on knowledge discovery & data mining, pp. 577–586, 2019.

[7] Panagopoulos, G., Nikolentzos, G., and Vazirgiannis, M. Transfer graph neural networks for pandemic forecasting. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, pp. 4838–4845, 2021.

[8] Kamarthi, H. and Prakash, B. A. Pems: Pre-trained epidmic time-series models. arXiv preprint arXiv:2311.07841, 2023.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes a pre-trained framework for epidemic forecasting that learns transferable epidemic dynamics from diverse historical disease data. CAPE models outbreaks as mixtures of latent population states that it discovers directly from surveillance data. They use self-supervised objectives: masked time-series reconstruction and contrastive learning, and epidemic-aware regularizers. CAPE captures robust, interpretable representations that generalize across diseases and regions. Experiments on 17 diseases across 50+ regions show that CAPE outperforms baseline time-series and epidemic models in zero-shot, few-shot forecasting.

### Strengths
1. The motivation for a pre-trained model for epidemic forecasting is important
2. Epidemic regularized used in pre-trained models in novel
3, Results on provided set of diseases is significant and are compared against without pre-trained models

### Weaknesses
1. Comparison with SOTA epidemic models are lacking (https://arxiv.org/abs/2207.09370 has a few examples)
2. The benchmarks can be expanded with more datasets from wide range of diseases. Eg: see datasets such as (https://www.tycho.pitt.edu/)
3. Are the datasets evaluated on real-time forecasting setup? What is the training validation split. Real-time forecasting setup is more realistic for ongoing epidemics where such models are more useful.
4. Can the model provide probabilistic forecasts? This is very crucial for real-world deployment.

### Questions
See Weaknesses

### Soundness
2

### Presentation
3

### Contribution
2
