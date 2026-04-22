# Simulator‑Based Synthetic ECGs for Self-Supervised Pretraining

- Avg Score: 3.00
- Decision: Reject
- Scores: 4, 6, 2, 0

## Abstract
Medical data remain scarce and sensitive despite rich domain knowledge. 
We ask whether knowledge-driven parametric ECG simulators can supply scalable self-supervised pretraining signals without using patient records during pretraining.
We use two established simulator-based ECG generators to synthesize $10$-s, $500$-Hz lead~II signals for pretraining a Transformer encoder with masked autoencoding, and compare it to pretraining on real PTB-XL ECGs and on VAE/GAN-generated ECGs.
We then fine-tune on 26 abnormal-ECG classification tasks across PTB-XL, G12EC, and CPSC2018 and benchmark against five strong supervised baselines. 
The Transformer pretrained on simulator-based synthetic ECG (SimECG) performs comparably to real-data pretraining and outperforms supervised baselines on 24 of 26 tasks, yielding a mean 5.49% relative F1 improvement over the strongest baseline. 
Under reduced labeled-data budgets and across patient demographics, it largely preserves the advantages of real-data pretraining and maintains competitive performance across all 12 single-lead configurations.
Crucially, simulator pretraining still avoids any exposure to patient data during pretraining.
These results indicate that knowledge-driven synthetic ECG corpora can provide practical, privacy-enhancing initialization for downstream ECG models in data-limited regimes.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
In their submission, the authors investigate the prospects of using synthetic single-lead ECG samples for pretraining ECG classification models through self-supervised pretraining (masked auto-encoding, but also other approaches are investigated). They consider different sources of synthetic models from knowledge-based simulator ECGs to ECG generated via generative models. They probe the pretrained models through finetuning on different binary classification tasks (normal vs. x) and find indications for robust improvements through self-supervised pretraining in comparison to training from scratch. The authors provide insights into lead-dependence and fairness aspects by considering different patient subgroups.

### Strengths
* The authors cover a comprehensive set of experimental conditions, ranging from two different categories of synthetic samples with two approaches each, over different self-supervised learning algorithms to three downstream tasks covering several conditions each along with a diverse set of supervised baselines.
* The authors identify settings (MAE with pretraining on SimECG) that show robust performance across several datasets
* The authors present ablation studies to underline the robustness of their findings (varying ECG leads, fairness evaluation)

### Weaknesses
* The motivation of using ECG as a test-case for other domains does not seem convincing since the insights from this work are highly ECG-specific. And in the ECG domain there is no shortage of publicly available unlabeled data, as the authors acknowledge themselves, while not even mentioning HEEDB as the currently largest ECG dataset with more than 10M samples as well as CODE-15. Nevertheless, an approach as proposed by the authors could be interesting for other domains where also unlabeled data is very scarce such as invasive data e.g. intracranial EEGs etc.
* The related work misses essentially all works works on ECG foundation models, such as [1][2] and many more. While this just misrepresents the research landscape, the omission of essentially all modern diffusion-based generative models for ECG data, see e.g. [3],[4] among many others,  is a more severe issue as it also impacts the choice of generative models for the later experiments. This raises doubts about the quality of the generated samples and questions the conclusion about the superiority of knowledge-based approaches.
* The authors consider a very special setup (single-lead input) and investigate artificial normal vs. diseases settings. The problem with such binary setups is that if they are not carefully balanced according to other covariates that it will often be very simple for the model to distinguish cases based on characteristics that have nothing to do with the actual disease label under consideration.
* The authors provide only relative comparisons to own baselines. From my point of view it would raise the impact of their contribution considerably if they would show results on established benchmarks, be it single-lead or 12-lead, where benchmarking results are known.

[1] Li, J., Aguirre, A. D., Junior, V. M., Jin, J., Liu, C., Zhong, L., ... & Hong, S. (2025). An Electrocardiogram Foundation Model Built on over 10 Million Recordings. NEJM AI, 2(7), AIoa2401033.
[2] McKeen, K., Masood, S., Toma, A., Rubin, B., & Wang, B. (2024). Ecg-fm: An open electrocardiogram foundation model. arXiv preprint arXiv:2408.05178.
[3] Lai, Y., Liu, B., Guan, X., Zhao, Q., Li, H., & Hong, S. (2025). ECGTwin: Personalized ECG Generation Using Controllable Diffusion Model. arXiv preprint arXiv:2508.02720.
[4] Alcaraz, J. M. L., & Strodthoff, N. (2023). Diffusion-based conditional ECG generation with structured state space models. Computers in biology and medicine, 163, 107115.

### Questions
* Are there any reasons that prevent the authors from considering multi-lead (potentially knowledge-based ECGs are not available for this setting?) and more realistic i.e. non-binary evaluation modes?
* Could the authors comment on their choice of the baseline architecture for the self-supervised pretraining as a transformer model? At least in their supervised experiments it performs poorly in comparison to other model architectures. This gets overcompensated through pretraining, but a different architecture might profit even more from self-supervised pretraining?
* In 5.2, the authors claim to use a per sample standardization. I cannot image that this is really the optimal setting since certain diseases rely on absolute scales e.g. R-amplitudes in LVH as a very simple example. This information gets largely lost through per sample standardization. Did the authors experiment with other setups?

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
5

### Summary
This paper investigates whether simulator-generated ECG signals can serve as effective pretraining data for self-supervised learning and subsequently benefit downstream ECG classification tasks.

### Strengths
I find this exploratory study valuable. Although it does not introduce new model architectures, it provides meaningful empirical insights and implications that may inspire future research directions.

### Weaknesses
1. However, given that the contribution lies primarily in empirical exploration rather than methodological novelty, I believe this work may be more suitable for a journal submission rather than a conference. Conference page limits may restrict the authors from fully presenting the depth of their analyses and findings.
2. According to Table 1, when downstream training data are limited, the model pretrained on simulator-generated signals exhibits inferior performance; yet, as the amount of downstream data increases, its performance converges toward that of models pretrained on real ECG signals. This trend appears intuitive: with increased training data for fine-tuning, downstream performance depends more on the classification data itself than on the pretraining source. A similar phenomenon is evident in Figure 2, where the scratch model—without any pretraining—achieves comparable performance when sufficient labeled data are available. These observations may weaken the practical impact of the claim regarding the advantages of simulator-based self-supervised pretraining.
3. In fact, a more significant contribution of this study could lie in its comparison with deep generative models such as GAN- and VAE-based ECG synthesis. Unfortunately, this part is relegated to the appendix and is not sufficiently highlighted. Moreover, the appendix presents mixed outcomes—simulator-based models outperform generative models in some cases and underperform in others—without synthesizing clear or generalizable conclusions. The authors are encouraged to extract higher-level insights rather than simply listing experimental results.
4. Simulator-generated signals may lack patient-specific variability. By contrast, deep generative models trained on large-scale real ECG datasets can produce signals with richer pathological diversity and patient-specific morphology. Therefore, similar to the analyses in Figures 1 and 2, the authors should include comparative experiments between simulator-based pretraining and various deep generative models under varying downstream data sizes. Additionally, comparisons with diffusion-based ECG generation methods—such as Biomedically Informed ECG Synthesis: Customizing Cardiac Cycle Phases with Diffusion Models (Y. Lin et al., IEEE BIBM 2024)—should be incorporated to improve completeness and relevance.

### Questions
none

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
The authors present an empirical comparison that shows that pre-training on synthetic ECG data generated through simulators can benefit the downstream performance of transformer-based models compared with approaches without pre-training or pre-training only on (smaller) real-world datasets.

### Strengths
- The paper is easy to follow and delivers one main message
- overall the approach is to some degree convincing

### Weaknesses
- While the idea overall shows convincing results, the main message is limited in that pre-training on synthetic data helps. A more in-depth analysis on the implications, clinical need and improvement, and similar aspects would be interested. In its current form the insight of the manuscript are limited.
- Your results indicate that pre-training on PTB-XL already gives the best performance. PTB-XL is a rather small-scale dataset. Thus pre-training on a synthetic dataset 50x the size of PTB-XL seems excessive if in many settings medium to large real-world datasets are available.
- A computational comparison on the pre-training budget on real-data and synthetic data is not presented. This it is unclear how the gains of your approach would be under the same computational conditions. 
- many information double and scattered. The manuscript reads at times repetitive. Example: Caption table 1 and text right below describes essentially the same.
- in the implications you mention that pre-training size can be scaled more easily in the simulator-based approach. However, you do not show that more pre-training on synthetic data could improve performance. I would expect some kind of plataeu because of missing real-world nuisances.

### Questions
- Some references to ECG-based pretraining methods are missing, e.g. [1] and others.
- please double check writing. E.g. missing blankspace in line 056, 067, and others
- line 139. How exactly do you parameterize the simulator. You say "with parameters randomly sampled from predefined distributions", which parameters do you choose or how do you choose the distributions?
- section 3: It is unclear to me whether you focus your whole development on lead II only or why you are extracting lead II only here. Can you clarify?
- section 5: for each of the 4 axes, how do you define the metrics, e.g. performance, data efficiency and robustness?
- Table 1:
  - Unclear abbreviations: e.g. DGM as VAE or as GAN
  - Can you provide std of the metrics over multiple runs? Meaning are the results significant
- Table 2:
  - Can you provide std of the metrics over multiple runs? Meaning are the results significant
  - the comparison is somewhat flawed since you compare non-pretrained models with a pretrained model. If you only compare the non-pretrained models, then the GRU actually performs best. A rigorous comparison would compare on pretraining in the same way.
- Figure 1: 
  - font size too small 
  - what is the x-axis? (it is number of pretraining samples but please make this explicity)
- line 311: "degrades more gracefully " what does that mean?
- why do you focus on single lead classification instead of 12-lead which often the standard in clinical settings?


[1] Kiyasseh, Dani, Tingting Zhu, and David A. Clifton. "Clocs: Contrastive learning of cardiac signals across space, time, and patients. "International Conference on Machine Learning.

### Soundness
3

### Presentation
1

### Contribution
1

---

## Human Reviewer 4

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
This paper presents a case study on how knowledge driven ECG simulators compare with real data for pretraining. The paper is a simple case-study, and as such, does not provide any contribution in either data generation or pretraining approaches. There is no theoretical and algorithmic contributions to science of signal generation and pretraining. For these reasons, the paper should be rejected.

### Strengths
None noted.

### Weaknesses
Lack of originality and novelty.

### Questions
None

### Soundness
2

### Presentation
2

### Contribution
1
