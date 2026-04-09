# Direct Review Excerpts: Weakness Patterns Applicable to SFT Paper

## Weakness 1: Limited Benchmark Evaluation & Generalization Concerns

**From Linear Complexity Attention Paper (lXRDQsiP2v.md):**
> "Experiments on the Long Range Arena are not exactly like-for-like, as the ToST hyper-params are tuned individually for each task -- to my complete surprise. See Table 7 in line 972 in appendix. So, I'm not sure what the takeaway is from Table 3 -- if the method needs tuning for each specific task that definitely takes quite a bit away from its appeal. Transformers are amazing (in part) because they require little tuning from task to task."

**From DEPT Pre-training Paper (vf5aUZT0Fz.md):**
> "The multi-domain data is almost only in English. But for the multilingual data, the data of each language should also contain various domains. Therefore there are confounding variables. A natural question would be whether the model can generalize to the same domains across different languages."

---

## Weakness 2: Insufficient Analysis of Failure Modes & Domain Specificity

**From Domain Adaptation Paper (ijwYWoChN9.md):**
> "The foundational hypothesis that 'PLMs encapsulate multiple pieces of knowledge as subnetworks' (Lines 38-40) lacks supporting references or verification experiments. Furthermore, the approach of representing domain gaps by differences in model parameters between source and target domains is not sufficiently justified. Although empirical results support DST's effectiveness, the Introduction lacks a clear causal rationale for these core design choices."

**From GUI Agent Paper (n9PDaFNi8t.md):**
> "There could be more analyses on the data and model scaling... It remains unclear why certain tasks perform better or worse on web/desktop/mobile and different operating systems. Do such differences attribute to model, data, or task setting? And what are the corresponding data and compute needed to improve on a specific setting of interest."

---

## Weakness 3: Memorization vs. Generalization Not Clearly Distinguished

**From DPO Generalization Paper (bGkPZtisSm.md):**
> "The paper has inconsistent descriptions about what component performs the prediction... The generalization guarantee applies to the implicit reward model, but what we actually use is the LLM. Actually, there could not exist such a guarantee for π_θ, creating a disconnect between theory and practice."

**From Time Series Shift Invariance Paper (nibeaHUEJx.md):**
> "Most evaluations, including the ablation studies, focus on downstream tasks, which are interesting and practically relevant but do not provide clear insight into what the algorithm is doing at the shift-invariance level. They also add confounding factors."

**From Continual Learning Paper (EDJ7cPZk7V.md):**
> "The observed correlation between example learning speed and catastrophic forgetting is empirical, with no theoretical analysis provided, hence of limited significance. Empirical analysis provided to establish the correlation is not sufficient. For example, learning dynamics depend on various factors such as learning rate, network architecture, optimizer, regularization etc."

---

## Weakness 4: Hyperparameter Sensitivity & Reproducibility Issues

**From Transformer Architecture Paper (lXRDQsiP2v.md):**
> "Experiments on the Long Range Arena are not exactly like-for-like, as the ToST hyper-params are tuned individually for each task... if the method needs tuning for each specific task that definitely takes quite a bit away from its appeal. Transformers are amazing (in part) because they require little tuning from task to task."

**From Continual Learning Paper (EDJ7cPZk7V.md):**
> "The algorithm may rely on selecting hyperparameters (e.g. s and q) for removing the slowest and fastest examples. And it might be unclear how that parameter varies across different datasets. If choosing a hyperparameter repetitive experiments, then it may defeat the premise of continual learning."

---

## Weakness 5: Incomplete Ablation Studies & Missing Baselines

**From Domain Adaptation Paper (ijwYWoChN9.md):**
> "L_KDL is not ablated to show its usefulness in this work. Some code or pseudocode would strengthen knowing how the KSL/KDM is actually implemented."

**From DEPT Paper (vf5aUZT0Fz.md):**
> "An important additional baseline would be models trained on individual data sets. This would give insights into the advantages/disadvantages of model averaging."

**From Adapter Paper (uJqKf24HGN.md):**
> "More rigorous experiment results may be needed to illustrate ZeroFT's effectiveness. Can you provide a schematic or computational rule of forward computation and backpropagation during training to compare ControlNet and UniCon?"

---

## Weakness 6: Limited Scope & Scalability Questions

**From Transformer Architecture Paper (lXRDQsiP2v.md):**
> "Approximately tripling the number of parameters yields similar performance to GPT2-Base. Looking at Table 4 (right) this would mean that the running time of the ToST is considerably higher than that of GPT-2 when matched for performance -- not exactly re-assuring."

**From DPO Generalization Paper (bGkPZtisSm.md):**
> "Llama 2-7B is relatively old, experiments on 3/3.1/3.2 would be better."

**From GUI Agent Paper (n9PDaFNi8t.md):**
> "The contribution and novelty limited to the main claim of serving as an open-source alternative to commercial VLMs. This work does not address GUI specific challenges. In fine-tuned models, there is often a trade-off between image quality and controllability."

---

## Weakness 7: Theoretical Justification or Mechanistic Understanding Missing

**From Domain Adaptation Paper (ijwYWoChN9.md):**
> "The foundational hypothesis that 'PLMs encapsulate multiple pieces of knowledge as subnetworks' lacks supporting references or verification experiments... Although empirical results support DST's effectiveness, the Introduction lacks a clear causal rationale for these core design choices."

**From Time Series Paper (nibeaHUEJx.md):**
> "While the shift consistency captures something in this direction, it still requires training the classifier to assign class probabilities... Could one not directly measure the variance or consistency of the output angles (or phase-shifted latent representations) across the shifted samples to get clearer insights into what is happening here?"

**From Continual Learning Paper (EDJ7cPZk7V.md):**
> "No explanation or intuition is provided as to why medium learning speed items are the most useful for populating memory. It would be good if the authors provided a rationale beyond the empirical results."

---

## Weakness 8: OOD Evaluation & Generalization Gaps Not Rigorously Measured

**From Time Series Shift Invariance Paper (nibeaHUEJx.md):**
> "A natural follow-up question is how well the guidance network performs in OOD settings where the ground truth shifts are known but the time series were not part of the training data. Since the learned latent representations are still just time series, a simple illustration of the approach could be to take in a number of shifted time series and visualize the resulting latent representation."

**From Model Merging Paper (OZVTqoli2N.md):**
> "Since IN-R, C-100 and CUB are very much id-distribution w.r.t pre-training on ImageNet, I wonder whether simple fine-tuning of the final classification layer, which can be a metric-based classifier with no forgetting, can be sufficient to achieve good performance?"

**From Domain Shift Paper (ijwYWoChN9.md):**
> "The LLM experiments are not compared to few-shot/zero-shot prompting despite these models being able to perform in-context learning. The LLM experiments need baselines to compare to."

---

## Weakness 9: Insufficient Error Analysis & Edge Case Handling

**From Fairness Robustness Paper (lW0ZndAimF.md):**
> "Empirical analysis is somewhat limited both in domain and the datasets used... It's hard to argue only based on the provided analysis that the results will extend to larger models, other vision datasets, or non-vision classification tasks."

**From Time Series Paper (nibeaHUEJx.md):**
> "Some of the results indicate improvement when averaged over multiple runs (how many runs are these results averaged over?). But in some of those cases based on the standard deviations computed there is a fair overlap between prior methods and your method, it's possible that the advantage in those cases is not so clear when there is a large variance."

**From Continual Learning Paper (EDJ7cPZk7V.md):**
> "Although the authors show that the results achieved are better than other alternatives, the benefit is only marginal. Often even less than the standard deviation. During the analysis, the difference is often at most 2%, between all removal combinations slowest/quickest."

---

## Weakness 10: Practical Applicability & Computational Trade-Offs Not Addressed

**From Domain Adaptation Paper (ijwYWoChN9.md):**
> "Although the KSL is smaller compared to the size of the model, it must have some sort of slow-down associated with it since it appears as an additional layer with an additional step across K subcomponents. What is the speed reduction in using this method?"

**From Adapter Paper (uJqKf24HGN.md):**
> "During inference, will UniCon be similar even worse in latency and memory due to the more complex structure? This also seems to explain why UniCon performs better because the block structure is more complex."

**From GUI Agent Paper (n9PDaFNi8t.md):**
> "The released data and model would be a good contribution to the community, looking forward to seeing it being released."

---

## Weakness 11: Benchmark Design Artifacts & Limited Task Diversity

**From Continual Learning Paper (EDJ7cPZk7V.md):**
> "The paper only explores ResNet and its smaller variants for the analysis. For other architectures such as transformers, VGG net, etc do the same conclusions stand?"

**From DEPT Paper (vf5aUZT0Fz.md):**
> "No downstream tasks in natural language understanding or generation are evaluated on the resulting models. But such further evaluation is important. The data sources are not always clear given a dataset."

**From Continual Learning Paper (EDJ7cPZk7V.md):**
> "There is quite a bit of variation across the datasets and experimental conditions, such as buffer size, in terms of the relative performance of different percentages of the too-small and too-fast sets that should be excluded. There is no analysis of this, which begs the question of how to set these hyperparameters in a new setting."

---

## Cross-Cutting Theme: Limited Baseline Comparisons

**From Model Merging Paper (OZVTqoli2N.md):**
> "This works connects to several existing works in the realm of model merging... These works are not cited in the current version, maybe because these connections were not obvious to the authors."

**From Domain Shift Paper (ijwYWoChN9.md):**
> "The description of the Query Fixer module is relatively brief. The authors could consider adding a detailed algorithm for correcting incorrect SQL. Additionally, I suggest including a separate limitations section to discuss the potential challenges for application."

---

## Cross-Cutting Theme: Statistical Rigor

**From Continual Learning Paper (EDJ7cPZk7V.md):**
> "The experiments cover a range of datasets and settings for the size of the buffer of replayed examples... Most of the experiments focus on a sequence consisting of just a pair of tasks, but there are some results with a more extensive set of tasks."

**From Fairness Paper (lW0ZndAimF.md):**
> "It would be great to also see some analysis for the hyper-parameters α and γ for a better understanding of the method. How optimal is 0.3 for α? What is the effect of changing it? I would appreciate if the authors can provide some intuition behind the effect of these two hyper-parameters."

