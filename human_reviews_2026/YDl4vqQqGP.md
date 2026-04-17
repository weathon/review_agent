# MambaSL: Exploring Single-Layer Mamba for Time Series Classification

- Decision: Accept (Poster)
- Scores: 8, 6, 4

## Abstract
Despite recent advances in state space models (SSMs) such as Mamba across various sequence domains, research on their standalone capacity for time series classification (TSC) has remained limited. 
We propose MambaSL, a framework that minimally redesigns the selective SSM and projection layers of a single-layer Mamba, guided by four TSC-specific hypotheses.
To address benchmarking limitations—restricted configurations, partial University of East Anglia (UEA) dataset coverage, and insufficiently reproducible setups—we re-evaluate 20 strong baselines across all 30 UEA datasets under a unified protocol. 
As a result, MambaSL achieves state-of-the-art performance with statistically significant average improvements, while ensuring reproducibility via public checkpoints for all evaluated models.
Together with visualizations, these results demonstrate the potential of Mamba-based architectures as a TSC backbone.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper introduces MambaSL, a single-layer Mamba architecture for time series classification. The authors walk through the Mamba architecture in detail, explaining the individual components and the intuition behind how these components related to time series classification. They then propose their method which makes use of these observations in four succinct hypotheses that they use to build MambaSL. They then demonstrate state-of-the-art results across the UEA dataset, a common time series classification dataset, and examine their hypotheses through detailed ablations.

### Strengths
- The authors walk through the Mamba architecture in incredible detail, explaining the components that are necessary to understand their newly proposed architecture. This is greatly appreciated especially for readers with limited knowledge of SSMs. 
- The architecture choices are well-motivated and explained. It’s greatly appreciated that authors are succinct and organized in their description of augmentations to the Mamba architecture, and the ablation results support their decisions. 
- The work puts incredible emphasis on reproducibility and goes to great lengths to discuss the differences in reported vs. optimized accuracy for baseline methods. I found this incredibly rigorous and greatly appreciated, boosting my confidence in the MambaSL performance given their careful considerations of baselines. In addition, the number of baselines included is quite substantial and covers a wide range of methods.
- This work is overall very principled and novel, with authors clearly laying out the augmentations made on top of the Mamba architecture that are suitable for time series. I think this paves the way for more exploration in this space, and it lays forth a foundation for rigorous and proper benchmarking and principled development of time series models.

### Weaknesses
- The experiments could use errors bars for presentation of results across datasets. Many of the performances are very close to each other on Figure 4, and it’s unclear whether the difference between MambaSL is statistically significant from other methods. In addition, error bars are needed in ablations to understand significance of differences in the dataset.
- The model seems to not transfer well to variable-length settings, an assumption made implicitly in Hypothesis 1 where the k value is chosen based on the length of the dataset. Can authors comment on the ability of the model in variable-length settings?
- The UEA dataset, while extensive, contains many small and curated datasets that are not representative of many real-world time series datasets. Did the authors test the method on other datasets for other challenging tasks? The work could use some demonstration on a frontier dataset, such as one released recently that represents a challenging real-world task.

### Questions
- Did authors test MambaSL on datasets with very long samples? One benefit of the Mamba architecture and other SSMs are the ability to capture long contexts; this could be a beneficial demonstration but is also not required.
- The UMAP showing comparisons of results across datasets is very cool! I’d love to see this UMAP reduction done along the dataset dimension as well to understand which datasets on which MambaSL performs well compared to non-DL vs. DL methods. This is not required at all but could also be an interesting analysis.
- The authors focus on time series classification in this case, but could this be extended to forecasting?
- Further, does the work easily extend to irregular time series, where time points are collected in irregular intervals. Can the architecture make use of the irregularity in observations when making predictions?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes MambaSL, a single‑layer Mamba architecture tailored to time‑series classification (TSC). Four hypotheses drive minimal but targeted changes: H1 scales the input Conv1D kernel with sequence length; H2 modularizes time (in)variance of the SSM parameters; H3 disables the skip (D) connection; H4 introduces a multi‑head adaptive pooling readout. The overall block is in Figure 2 (p.5); ablation evidence is in Table 1 (p.8) and Table 2 (p.8–9). On the full 30‑dataset UEA archive, the method attains the top average accuracy and rank among 21 models.

### Strengths
+ Thorough ablations, including eight TI/TV combinations (Table 2) and multiple pooling alternatives (Table 1). 
+ Excellent commitment to reproducibility across all 30 UEA datasets, with promises of public code, checkpoints, and full logs.
+ The re-evaluation of TSF-origin models, showing they were previously underestimated, is an important finding. The analysis of H2 (time variance) is the most interesting part, showing that for TSC, simpler LTI systems can be better than LTV.

### Weaknesses
+ The novelty of H1–H4 is moderate; each component is a small change rather than a new architectural principle.
+ The paper's own ablation study (Table 1) shows that H1 (scaling kernel size) is not clearly supported by the average accuracy metric.
+ The paper's title and focus on a "single-layer" model  is not well-justified. Why is one layer sufficient? The paper is missing a ablation study on the effect of model depth (i.e., stacking MambaSL layers).

### Questions
+ Could you provide a statistical test (e.g., a Wilcoxon signed-rank test) comparing the performance of MambaSL (with H1) directly against the "w/o H1" variant, or should H1 be re-framed in light of this evidence?
+ Did you try multi‑layer (2–3 layers) MambaSL to confirm single‑layer sufficiency? How does MambaSL perform on variable‑length sequences at test time (beyond the pooling stage)?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces MambaSL, a framework that minimally modifies the selective state space models and projection layers of a single-layer Mamba architecture for time series classification. Experimental evaluations conducted on 30 datasets from the UEA benchmark demonstrate that MambaSL consistently outperforms 20 competitive baseline methods.

### Strengths
1. This paper applies the Mamba architecture to time series classification tasks and achieves superior performance compared to Transformer, MLP, and CNN models on multivariate time series classification benchmarks. This demonstrates the strong potential of Mamba for time series classification modeling.  
2. The authors provide open-source code for the proposed model, along with implementations of the baseline methods, ensuring the reproducibility and transparency of the experimental results.

### Weaknesses
1. The paper’s title, introduction, and model design focus primarily on time series classification without providing an in-depth discussion of how variable relationships are modeled in multivariate time series. In this case, the experiments are limited to the 30 datasets from the UEA multivariate time series classification archive, while excluding the 128 univariate datasets. This omission weakens the paper’s motivation and makes it difficult to assess whether the proposed method achieves state-of-the-art performance under univariate settings.  

2. Although the application of Mamba to time series classification is commendable, the paper does not clearly explain how the proposed MambaSL framework captures intrinsic temporal properties—such as temporal dependencies, inter-variable relationships, and short- or long-term sequence characteristics. The introduction and contribution sections also fail to clarify how MambaSL learns discriminative feature patterns beneficial for classification.  

3. The model section contains extensive background on basic Mamba concepts, making it difficult for readers unfamiliar with Mamba to follow. At the same time, readers experienced in time series classification may still find it unclear how the model effectively learns task-relevant features for classification.  

4. Despite the considerable experimental effort and the inclusion of reproducible scripts, all deep learning results in the paper suffer from a **test data leakage issue**. As shown in the provided code (`MambaSL/exp/exp_classification.py`, lines 22–25), the test set is incorrectly used as the validation set:

   ```python
   self.train_data, self.train_loader = self._get_data(flag='TRAIN')
   if self.args.is_training:
       self.vali_data, self.vali_loader = self._get_data(flag='TEST')
   self.test_data, self.test_loader = self._get_data(flag='TEST')

This indicates that the test set is used for model validation and hyperparameter selection, resulting in a significant evaluation bias.

5. In Appendix A, the authors mention that the experimental framework is based on the **Time-Series Library (TSLib)** (Wu et al., 2023). While TSLib is widely recognized for fair benchmarking in forecasting tasks, prior research ([1], Section 4.4 “Leaky Baselines”) has shown that TSLib introduces **test data leakage** in classification experiments. Furthermore, several non-deep learning baselines (e.g., Rocket, HC2, Hydra) do not rely on validation sets during training and for model selection. Consequently, using the maximum test performance as the final evaluation metric for MambaSL constitutes an **unfair comparison** with these baselines.

**Reference**

[1] *TOTEM: Tokenized Time Series Embeddings for General Time Series Analysis*, TMLR, 2024.

### Questions
1. In univariate time series classification, the original **InceptionTime** paper evaluates models based on the checkpoint with the lowest training loss, and the same evaluation strategy is adopted in **[2]**. Under a comparable setting, how does **MambaSL** perform relative to **HC2** and **MultiRocket+Hydra** on the 30 UEA multivariate time series datasets?  

2. On the 128 UCR univariate time series datasets, following the **InceptionTime evaluation protocol**—where the model with the lowest training loss is used for testing—how does **MambaSL** compare in classification performance with **HC2** and **MultiRocket+Hydra**?  

3. Beyond the evaluation setups in **InceptionTime** and **[2]**, some studies (e.g., **TSLANet**) adopt a different experimental protocol that uses **20% of the training set as a validation subset** for model selection on both UCR and UEA datasets. Under this widely used setting, how does **MambaSL** perform compared with **HC2**, **MultiRocket+Hydra**, and other competitive baselines?  

4. Compared to 20 baseline methods, what are the advantages and disadvantages of the proposed **MambaSL** method in terms of runtime?

**Reference**

[2] *Inherently Interpretable Time Series Classification via Multiple Instance Learning*, ICLR, 2024.

### Soundness
3

### Presentation
2

### Contribution
2
