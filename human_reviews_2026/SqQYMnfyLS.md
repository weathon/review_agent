# Latent Space Structuring for Conditional Tabular Data Generation on Imbalanced Datasets

- Decision: Reject
- Scores: 2, 4, 2, 4

## Abstract
Generating synthetic tabular data under severe class imbalance is essential for domains where rare but high-impact events drive decision-making. However, most generative models either overlook minority groups or fail to produce samples that are useful for downstream learning. We introduce CTTVAE, a Conditional Transformer-based Tabular Variational Autoencoder equipped with two complementary mechanisms: (i) a class-aware triplet margin loss that restructures the latent space for sharper intra-class compactness and inter-class separation, and (ii) a training-by-sampling strategy that adaptively increases exposure to underrepresented groups. Together, these components form CTTVAE+TBS, a framework that consistently yields more representative and utility-aligned samples without destabilizing training. Across six real-world benchmarks, CTTVAE+TBS achieves the strongest downstream utility on minority classes, often surpassing models trained on the original imbalanced data while maintaining competitive fidelity and bridging the gap for privacy for interpolation-based sampling methods and deep generative methods. Ablation studies further confirm that both latent structuring and targeted sampling contribute to these gains. By explicitly prioritizing downstream performance in rare categories, CTTVAE+TBS provides a robust and interpretable solution for conditional tabular data generation, with direct applicability to industries such as healthcare, fraud detection, and predictive maintenance where even small gains in minority cases can be critical.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes CTTVAE+TBS which is a transformer-based tabular VAE that can perform synthetic data generation for imbalanced tabular data. It has 2 key components: CTTVAE (class-aware triplet loss) and TBS (training by sampling). The authors find on the 6 real world datasets that was tested, the proposed method is able to yield a higher F1 score (high utility) while maintaining fidelity (measured by Wasserstein and JSD) and also preserve privacy (measured by DCR and NNDR).

### Strengths
1. The proposed architecture combined with upsampling has not been done before
2. The results look great in terms of all 3 aspects of tabular synthetic data generated
3. The paper is well written and the proposed concepts are presented clearly

### Weaknesses
1. Only 6 datasets are used. It is lot fewer than other past works, which makes the claim less strong.
2. Diffusion method such as TabDDPM and tabular foundation model based methods such as TabPFGen and TabEBM are not present in the results

### Questions
1. It would be great to show an intuitive 2D dataset and show how the proposed method is able to outperform other baseline methods

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes CTTVAE+TBS, a conditional transformer-based tabular variational autoencoder designed to generate high-quality synthetic tabular data in the presence of severe class imbalance. CTTVAE combines a class-aware triplet margin loss (which enforces compact and separable latent representations) with a training-by-sampling (TBS) strategy that adaptively increases exposure to underrepresented classes. This dual mechanism enables the model to produce synthetic data that better supports downstream tasks, particularly for minority categories, without sacrificing fidelity or privacy.

### Strengths
Generating synthetic datasets under severe class imbalance is a practically important problem with broad real-world relevance. The paper is well written and clearly organized, with careful optimization of baseline models and comprehensive details of the experimental setup and results provided in the appendix.

### Weaknesses
My main concern with the paper relate to its empirical evaluation. The paper compares CTTVAE only against GAN- and VAE-based models, which are now relatively weak baselines compared to modern diffusion-based approaches. The paper justify this by stating that diffusion models such as TabDDPM require significantly greater computational resources and are typically reported at the dataset level, making alignment with their evaluation setup impractical. However, this reasoning is not fully convincing. In practice, TabDDPM is not substantially more computationally demanding than models like TVAE or CTGAN. Indeed, when Wang and Nguyen (2025) introduced TTVAE, they included direct comparisons with both TabDDPM and TabSyn. It should therefore be feasible to run diffusion-based baselines on the same train/test splits, using the same number of replications, and to evaluate them with the same utility, fidelity, and privacy metrics—ensuring that hyperparameters are appropriately tuned (e.g., via Optuna). Including these stronger baselines would make the empirical analysis much more compelling.

Additionally, while line 56 indicates that CTTVAE is compared against two classical oversampling baselines, and Figure 4 in the Appendix lists both SMOTE and SMOTENC, the paper only reports results for SMOTE. The paper needs to clarify this point.  

Other aspects of the paper that warrant improvement include:

1. The experiments are replicated only three times. Increasing the number of replications (e.g., to ten) would improve the statistical robustness of the results.

2. The paper evaluates only six datasets in its baseline comparisons. Expanding the number of benchmark datasets would strengthen the empirical evidence—for example, the TTVAE paper includes sixteen benchmarks.

3. In line 58, the paper claims to compare CTTVAE with five state-of-the-art baselines, but this statement should be moderated since models such as TVAE and CTGAN are no longer considered state-of-the-art.

4. Tables 3 and 4 should include standard deviations, and the main text should explicitly reference that per-dataset results are available in Tables 8–13 of the Appendix, which should likewise report standard deviations.

Minor issues:

The caption of Table 2 states that the table report results for both majority and minority groups, but only the minority group results are reported. (The paper might also want to point out that the results for both minority and majority classes are presented in Table 7 in the Appendix).

In line 123, the paper should cite Wang and Nguyen (2025) rather than Badaro et al. (2023).

The captions of Tables 3, 4, and 5 should define the abbreviations “Maj.” and “Min.” for clarity. 

##########################################

Overall, this is paper that could be of relevance to the ICLR community. However, in its current form, I am inclined to recommend rejection due to significant limitations in the experimental evaluation—particularly the absence of comparisons with diffusion-based models. That said, I would be open to revisiting my assessment if the paper address these issues and provide evidence that the advantages of CTTVAE persist when evaluated against stronger diffusion-based baselines.

### Questions
Could you clarify why aligning the evaluations of diffusion models with the paper’s experimental setup is considered impractical?

Can you clarify why the paper did not pursue comparisons against SMOTENC?

The paper notes that DCR and NNDR are reported at the 5th percentile following Zhao et al. (2021), but the computation procedure is unclear. Is the 5th percentile taken over the distances between each synthetic record and all real records, or is DCR first computed per synthetic record and then the 5th percentile reported across all synthetic samples?

The paper omits details about the transformer architecture. Does this imply that the same configuration as in the TTVAE reference is used?

### Soundness
2

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
3

### Summary
This paper proposes CTTVAE+TBS, a conditional transformer-based variational autoencoder framework designed for tabular data generation under severe class imbalance. The model introduces two main components: a triplet-margin loss that structures the latent space to enforce class separability, and a training-by-sampling strategy that probabilistically adjusts batch composition to mitigate class frequency bias during training. Together, these mechanisms aim to improve minority-class representation and enhance the downstream utility of generated data while maintaining fidelity and privacy. Experimental results across multiple real-world datasets show that CTTVAE+TBS achieves stronger minority-class F1 scores and better privacy metrics than existing tabular data generators such as SMOTE, CTGAN, and TTVAE, though its overall utility gains are sometimes inconsistent.

### Strengths
The paper is well written and clearly structured, making the overall methodology and experimental design easy to follow.

### Weaknesses
1. The paper lacks a clear explanation of how 'balanced datasets' are constructed in experiments, and it is also unclear how many minority samples were actually added or generated to compose the balanced dataset used for experiments.


2. In several datasets, models trained on the original imbalanced data outperform those trained on oversampled data, raising questions about the true benefit of the proposed oversampling process. If the primary motivation is to generate standalone synthetic data rather than to improve classifier performance, the paper should include a direct evaluation of the quality, representativeness, and practical utility of the generated samples themselves.


3. There exists diffusion-based oversampling methods, such as Sos (Score-based Oversampling for Tabular Data) [1], tackle the same imbalance problem through score-based generative modeling. The paper would benefit from a comparison or discussion that clarifies how CTTVAE+TBS differs from or complements these diffusion-based approaches.


[1] Kim, J., Lee, C., Shin, Y., Park, S., Kim, M., Park, N., & Cho, J. (2022, August). Sos: Score-based oversampling for tabular data. In Proceedings of the 28th ACM SIGKDD conference on knowledge discovery and data mining (pp. 762-772).

### Questions
1. What do you think might be the reason why oversampling sometimes performs worse than using the original data?

2. Could the authors clarify how the balanced dataset used in the experiments was composed?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors of the paper aim to tackle the problem of conditional generation of imbalanced data within tabular data, regime. They introduce a Conditional Transformer-based Tabular Variational Autoencoder (CTTVAE), an extension of TTVAE augmented by an additional loss, namely a class-aware triplet margin loss and a training-by-sampling (TBS) strategy that adaptively increases exposure to underrepresented groups. Through this adaptation they intend to restructure the latent space so that it better encapsulates the intra-class compactness and inter-class separation and to mitigate representation bias  when categorical features exhibit strong imbalance.
At the experiments section they explore the performance of their method against state of the arts methods, across six real-world benchmarks.

### Strengths
The authors of the paper introduce CTTVAE+TBS through adding a triplet loss to TTVAE and TBS. By doing this, they aim to conditionally generate imbalanced data, a task that is frequently overseen. They add a triplet margin loss to the TTVAEs loss function (that replaces KL divergence term, with Maximum Mean Discrepancy (MMD) penalty between the aggregated posterior q(z) and the Gaussian prior p(z)) in order to encourage embeddings of the same class to lie closer together than those of different classes. 
Moreover they use a variant of the TBS concept, at which sampling is guided solely by a categorical variable as opposed to sampling over all discrete columns.

### Weaknesses
The method, albeit interesting, does not seem to significantly outperform existing methods. Actually SMOTE outperforms CTTVAE+TBS across all distance metrics. 

three soft comments: 
(1) there is a sum at equation (4) that is not explained (why, over what);
(2) at equation 6, it (most likely) should be $\hat{z}_i$;
(3) at Figure 1., above the circled set with the blue and red dots, it should probably be (z,h) instead of just z;

### Questions
I would like to ask the authors if they could please :

1) comment on the sensitivity of the method to hyperparemeters $\alpha$ and $\beta$ (Loss CTTVAE, page 4);
2) please explain what is the role of $u_r \sim \mathcal{U}(0,1)$, in equation 6.;
3) comment on the sensitivity and performance of the method with respect to the number of classes;
4) comment on the outperformance of SMOTE across all distance metrics (Table 3);
5) add the absolute difference between correlation matrices of SMOTE as well (Figure 3);
6) compare their results with [1], a diffusion model based method that also generates (and imputes) tabular data and also employ conditional generation


[1] Jolicoeur-Martineau, Alexia, Kilian Fatras, and Tal Kachman. "Generating and imputing tabular data via diffusion and flow-based gradient-boosted trees." International conference on artificial intelligence and statistics. PMLR, 2024.

### Soundness
3

### Presentation
3

### Contribution
2
