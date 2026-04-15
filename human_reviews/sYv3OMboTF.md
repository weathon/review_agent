# ExcelFormer: Making Neural Network Excel in Small Tabular Data Prediction

- Decision: Reject
- Scores: 5, 5, 5

## Abstract
Data organized in tables are omnipresent in real-world applications. Despite their strong performance on large-scale datasets, deep neural networks (DNNs) perform inferior on small-scale tabular data, which hinders the wider adoption of DNNs across domains. In this paper, we propose a holistic framework comprising a novel neural network architecture called ExcelFormer and two data augmentation approaches, which achieves high-precision prediction for supervised classification and regression tasks, particularly on small-scale tabular datasets. The core component of ExcelFormer is a novel "semi-permeable attention" coupled with a special initialization, which explicitly diminishes the impacts of uninformative features, thereby improving data-efficiency. The methodology insight behind two tabular data augmentation approaches, Feat-Mix and Hid-Mix, is to increase the training samples in a way accommodating the inherent irregularities of data patterns. Comprehensive experiments on diverse small-scale tabular datasets show that, our ExcelFormer consistently and substantially outperforms previous works, with no noticeable dataset type preference. Remarkably, we find the superiority of ExcelFormer extends to large datasets as well.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
**The paper studies** machine learning problems on small (<1K objects) tabular datasets (e.g. classification, regression, etc.) with additional experiments on larger (up to 500K objects) datasets.

**The main contribution of the paper** is ExcelFormer -- a deep learning scheme (Transformer-like architecture + custom training recipe) with the following new elements compared to the vanilla Transformer:
- (architecture) custom attention
- (architecture) custom feed-forward block
- (architecture) custom feature embeddings
- (architecture) custom prediction head
- (training) custom initialization
- (training) two custom augmentations

**The main claim:** *"EXCELFORMER consistently and substantially outperforms previous works"*

### Strengths
- The story is mostly easy to follow (also, I like the main illustration!).
- The research direction (designing better tabular deep learning architectures and augmentations) is important.
- The experimental part includes many datasets.
- I like the idea of using feature importances (1) to guide the attention between features and (2) to guide one of the two proposed data augmentations.

### Weaknesses
(1) (major) **Many *orthogonal* changes (listed in the summary above) are proposed *at once*.** It makes it difficuilt to attribute the observed results to any single element, which I believe to be important in the research context, especially for this genre of papers. I believe that the elements should be introduced either in isolation or step-by-step, but not at once (unfortunately, ablating each of the elements using the *final* architecture does not addresses the issue). Also, in my opinion, each of the elements should be compared against existing alternatives (i.e. the proposed augmentations VS existing augmentations, the proposed embeddings VS the existing embeddings, etc.).

Overall, modifying the well-established Transformer architecture in six(!) different aspects (listed in the summary), most of which has dedicated research subfields looks like an extremely ambitious goal to me. And I respect that, however, it makes it extremely hard to properly introduce and analyse each of the elements.

(2) (major) In my opinion, **the storyline around rotation invariance should be extended with specific analysis/experiments/results. Purely intuitive guidance may not be enough to drive the design decisions.** There are multiple places where the *formal* term "rotation invariance" is used in *informal* ways. For example, the paper uses terms like "more/nearly non-rotationally invariant". Overall, there is nothing wrong with relying on intuition, but after a certain threshold, there is a risk of coming to wrong conclusions.

A potential solution is to design a dedicated experiment that will quantify rotation invariance of any ML model. Then, some of the proposed elements can be motivated as a way to reduce the invariance according to the designed experiment. Again, this should be done *when introducing the elements*, not with the final architecture (as in Figure 5).

(3) Unfortunately, in my opinion, **the novelty is limited.** Some of the proposed modifications (listed in the summary above) are technically new, however, from the same technical perspective, they remain similar to the existing alternatives.

(4) In my opinion, sharing code, starting from the review stage, is important for this kind of studies. I wish I had an opportunity to have a look at the code to review the experimental setup and implementation details.

(5) Instead of Paragraph 2 of Section 1, I recommend writing only ~2-3 high level sentences and then referring to Section 5.4 of "Why do tree-based models still outperform deep learning on tabular data?" by Grinsztajn et al.

(6) I recommend proof-reading the paper for English style, vocabulary and grammar issues.

### Questions
How exactly is the mutual information computed for the continuous features and for regression labels?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The paper proposes a novel transformer architecture with two main suggestions:
- SEA: A semi-permeable attention block that masks the similarity scores from less informative features to the more informative ones, effectively blocking the transfer of information/representation from the less informative one.
- Interaction Attenuated Initialization: A rescaling of the variance of the weight initialization that in turn reduces the impact of SEA, which restricts feature interaction in the initial stages, making the proposed transformer architecture more non-rotationally invariant.

The authors then additionally propose two augmentation methods:
Hidden-Mix and Feat-Mix. One works by augmenting the data on the embedding space, while the other one works on the feature space by using the feature importance.

The authors combine the different components and one of the suggested augmentation methods at a time to yield an architecture that surpasses the baselines in 96 small-scale datasets and 21 large-scale datasets. The method outperforms the baselines without hyperparameter tuning and with hyperparameter tuning.

### Strengths
- The paper has a good structure.
- The authors propose quite a few interesting additions. The additions are ablated individually and the authors additionally show that the algorithm is more non-rotational invariant compared to the other transformer baseline.
- Experiments are extensive, a large number of datasets is considered and all the major baselines are included.

### Weaknesses
- The paper can be written better, typos exist here and there throughout the manuscript. (I will list a few of them in the questions section)
- The work should be self-contained and the "mutual information" should be described.
- In table 1, an interesting investigation would be how ExcelFormer would behave without any data augmentation (compared to the rest, not the ablation that is given) or how FTT would perform with the proposed augmentation approaches. I am additionally surprised that CatBoost performs worse compared to XGBoost consistently.
- No multi-class classification problems in the 96 datasets for the small-scale tabular datasets and only 4 datasets in the 21 large-scale datasets. 4 datasets in 117 datasets is an underrepresentation. 
- Regarding the evaluation metrics, why would the authors use AUC for binary classification and ACC for multi-class classification? The latter would not be a good metric for imbalanced datasets.
- An ablation is given when mixup is used as data augmentation, however, I would also prefer to see cutmix usage as an ablation.
- Without code release, I find it difficult to trust the results, as unfortunately there exist a plethora of recent DL architectures that claim state-of-the-art performance (TabNet, Node, Saint, etc) [1][2][3] only to be debunked later on [4]. It is necessary to validate the proper setup of the baseline algorithms and to verify the results of the method.

[1] Arik, Sercan Ö., and Tomas Pfister. "Tabnet: Attentive interpretable tabular learning." Proceedings of the AAAI conference on artificial intelligence. Vol. 35. No. 8. 2021.

[2] Popov, Sergei, Stanislav Morozov, and Artem Babenko. "Neural Oblivious Decision Ensembles for Deep Learning on Tabular Data." International Conference on Learning Representations. 2019.

[3] Somepalli, Gowthami, et al. "SAINT: Improved Neural Networks for Tabular Data via Row Attention and Contrastive Pre-Training." NeurIPS 2022 First Table Representation Workshop. 2022.

[4] Shwartz-Ziv, Ravid, and Amitai Armon. "Tabular data: Deep learning is not all you need." Information Fusion 81 (2022): 84-90.

### Questions
- This presents a significant challenge and bottleneck in the broader adoption of neural networks on tasks involve tables. -> that involve tables*
- DNNs’ leanring procedure -> Learning *
- Section 2.1, the mask is defined as M, however, Equation 2 and the follow-up text continue with W
- Section 5.1, TabFPN -> TapPFN 
- Section 5.2, indicating that applies hyperparameter finetuning onto EXCELFORMER can yield -> applying*

- Can you present the results where no augmentation is performed for ExcelFormer to analyze how it would perform against the other baselines? Can you provide the results where the proposed augmentation is applied to FTT?
- Can you include more multi-class classification problems in the used benchmarks and provide results?
- It would be interesting to see an ablation of the proposed augmentation methods against maybe cutmix or cutout to observe the overall improvement.

- **Is the EXCELFORMER more non-rotationally invariant and more noise insensitive?** In my perspective, an interesting addition would be to include the plain architecture of ExcelFormer in the investigation, then with every suggestion included one at a time (SEA, IAI), then both. This would show how the architecture gets more non-rotationally invariant as the different components are added compared to the beginning. Comparing against FTT is interesting, but it does not separate the impact of the overall differences in the architecture vs (SEA, IAI).
- I would urge the authors to provide the code to reproduce the results.

I am open to increasing my score if my concerns are addressed.

**Rebuttal Reply**
_____________________________________________________________________________________

I would like to thank the authors for their extensive reply. Below are my answers:

- **Regarding why previous SOTA has been debunked?**
I thank the authors for the explanation, although, I was of the same opinion initially.

**There are a few issues in the rebuttal from the authors, which I would like to point out:**

-  In Table 8 of the revised paper, with default hyperparameters, XGBoost in binary classification achieves a performance of 0 (failure case I would assume) which is surprising. Did the method fail since it does not have a competitive strategy for encoding categorical features like CatBoost? was one-hot encoding used? 
- Typo on Table 8, the Excelformer with HID-MIX is highlighted instead of FEAT-MIX.

- The authors write that they use the XGBoost, CatBoost implementations/search spaces from (Gorishniy et al., 2021), however, the cited work compares transformers against GBDT methods, so the setup might be biased. I would suggest to use the implementations from papers that advocate that GBDT methods outperform DL methods.

- The goal of hyperparameter optimization (HPO) is to find better hyperparameter configurations compared to the default ones, while it seems that with HPO for the experiment related to multi-class classification in the majority of cases a method's performance drops, which again is surprising. This is not consistent with Table 3, where default hyperparameters have a worse performance. 

- The aforementioned issue is additionally concerning in the case of Table 2, where the performance of CatBoost and XGBoost is given only with tuned hyperparameters. I would have liked to see the performance comparison where the aforementioned methods only use default hyperparameters. As an example in Table 8, CatBoost has a very strong performance with default hyperparameters. 

- In Figure 5, it seems that FTT performs better with IAI and SPA. I could not easily find in what subset of datasets the ablation was done or if it was done on all datasets (in the latter case, the authors could further improve their preprocessing/backbone). This point does not take any novelty away from the components that the authors have proposed in the paper.

- It would be nice to run on already existing benchmarks, that are used in the community [1][2]. Although to be fair, the authors do consider an extensive amount of datasets, compared to previous works.

- I would strongly disagree with the use of accuracy for multi-class classification problems. Not only is it not consistent with binary classification problems, but it additionally does not capture the performance with imbalanced datasets.
- nRMSE is confusing in the provided results since higher values in magnitude are better. How are the authors calculating nRMSE? It seems more like a distance to the worst possible value.
- As a last note, the code should also include the baselines, to verify that they have been run properly.

[1] Salinas, D., & Erickson, N. (2023). TabRepo: A Large Scale Repository of Tabular Model Evaluations and its AutoML Applications. ArXiv, abs/2311.02971.

[2] Gijsbers, P., Bueno, M. L., Coors, S., LeDell, E., Poirier, S., Thomas, J., ... & Vanschoren, J. (2022). Amlb: an automl benchmark. arXiv preprint arXiv:2207.12560.


Lastly, I would again like to thank the authors for their extensive reply. I believe the work is stronger with the updated results, as such I am raising my score. Unfortunately, I would need a few more clarifications on the issues I raised to recommend acceptance.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes a few modifications to the transformer
  architecture for small tabular data problems. The modifications are
  motivated by (1) the lack of rotational invariance in GBDTs and by (2) the
  efficacy of data augmentations in mainstream DL domains.

  To address (1), the authors propose:
  - semi-permeable attention (regular self-attention is masked such that more important features do not interact with less important features in self-attention)
  - interaction-attenuated initialization (initializing weights in semi-permeable attention with small values)

  To address (2) authors propose two variations of mixup tailored for tabular data problems:
  - Feat-Mix: swapping a random subset of features in two samples and mixing the labels taking feature's MI with the target into account
  - Hid-Mix: mixing channels after feature embedding and mixing labels proportionally, as in mixup

  With those changes, the proposed ExcelFormer outperforms deep-learning
  baselines (both traditional and more recent transformer-based models)
  and GBDTs in terms of average rank on 96 small (< 10000 samples) tabular problems.

### Strengths
- The paper is clearly and nicely written (both in overall structure and details in the technical details regarding proposed methods).
- It builds upon previous observations in its domain (tabular data) and proposes interesting "domain-specific" solutions to previously stated challenges/points for potential growth. For example, a large portion of the paper concerns with rotational invariance or the lack thereof as an inductive bias and.
- It obtains decent empirical results by integrating said solutions to the transformer architecture.

### Weaknesses
My concerns boil down to two things:

(1) **Using reduced rank as the main and only metric of model performance**. There are multiple problems I see with this approach to reporting the results, which keep me from agreeing with the ExcelFormer performance claims:
   - On this particular set of datasets DL models already perform on-par with GBDTs in terms of average rank (see FT-Transformer avg. rank), thus win over GBDT
   - The degree of improvement (in terms of the task metrics) is not quantifiable from the average rank. Did the ExcelFormer improved upon vanilla FT-Transformer by 10%, 50% in terms of AUC, neg. RMSE, ACC? The magnitude of the improvement is also important.

See also `[1]` regarding issues with comparing average ranks of multiple algorithms across multiple datasets.

I see that you provide all the results (albeit without standard deviations) for all models from Table 1, but this full table from the appendix is on the other side of the spectrum – too large to make generalizable conclusions. A more "zoomed in" view on performance would be very helpful. For example, you could provide metrics for DL baselines, GBDTs and ExcelFormer variants on datasets which were initially "won" by GBDT, but the changes introduced in ExcelFormer turned this around (I assume here that ExcelFormer is in essence a Transformer with potentially important domain-specific tweaks, comparison with MLP and FT-Transformer should be enough for a conclusion).

(2) **Limited ablations and comparisons to baselines**. The paper proposes a few architectural tweaks for a base transformer model: SPA instead of MHSA, IAI initialization in attention, new FFN block, new nonlinearity. With SPA and IAI highlighted as the more important ones. But the section with the ablation is rather short and lacking details regarding the setup, reporting only average rank performance. Could you provide a more detailed ablation and comparison to the vanilla transformer. For example:
- Transformer (no SPA, IAI, fancy embeddings and GLUs in FFN)
- Transformer + SPA
- Transformer + IAI
- Transformer + SPA + IAI
- ExcelFormer

A subset of datasets with metrics instead of ranks would be enough (see point 1).

For a second contribution - novel data augmentations, I believe they could be compared with baselines from pertaining on tabular data `[2,3,4]`, where resampling from marginal distributions for a set of columns was shown to be a decent augmentation. The results for the simplest possible setup (like MLP with all features linearly embedded – MLP-LR from `[5]`) with different augmentation strategies:
- Resample Augmentation
- Feat-Mix
- Hid-Mix
- Feat-Mix + Hid-Mix

would greatly improve the understanding of the efficacy of the proposed augmentations for tabular data.

In SPA and Feat-Mix, ExcelFormer uses mutual information. Could you discuss how different ways of estimating mutual information compare? It seems like a significant detail, but there are no mentions of this in the ablations or the experimental setup.

**References**:
- `[1]` Benavoli, Alessio, Giorgio Corani, and Francesca Mangili. "Should we really use post-hoc tests based on mean-ranks?." The Journal of Machine Learning Research 17.1 (2016): 152-161.
- `[2]` Bahri, Dara, et al. "Scarf: Self-supervised contrastive learning using random feature corruption." arXiv preprint arXiv:2106.15147 (2021).
- `[3]` Yoon, Jinsung, et al. "Vime: Extending the success of self-and semi-supervised learning to tabular domain." Advances in Neural Information Processing Systems 33 (2020): 11033-11043.
- `[4]` Rubachev, Ivan, et al. "Revisiting pretraining objectives for tabular deep learning." arXiv preprint arXiv:2207.03208 (2022).
- `[5]` Gorishniy, Yury, Ivan Rubachev, and Artem Babenko. "On embeddings for numerical features in tabular deep learning." Advances in Neural Information Processing Systems 35 (2022): 24991-25004.

### Questions
Technical details I'd like to clarify:
  - Could you provide details on how you compute mutual information, used in proposed augmentation and the attention module?
  - Could you provide more info on how ablations were run? You compare ablated variants to the fully tuned baseline, are the ablated variations also tuned?
  - How long were the models trained for? Was early stopping used during training? How the number of steps compare across deep models?
  - How ranks were calculated?

Other remarks:
- In the figure 3 hid-mix is called hidden-mix (only in the figure and nowhere else)
- The table with various datasets aggregations looks redundant in its current form. Not much interesting there besides TabPFN comparison. Not sure why grouping by classification vs regression and the number of continuous/categorical features should in differentiate general purpose methods. The results on the aggregated benchmark tell basically the same story: ExcelFormer is better than the baseline in terms of average rank. This space could be used to expand and address weaknesses (more ablations, more metrics).

Overall, I like the paper, and find the proposed architectural tweaks very interesting and important for the field.

I'm open to raise the score if my two concerns are addressed:

1. Results on multiple **challenging for DL datasets** where ExcelFormer significantly outperforms the DL competitors are demonstrated (not in ranks, but in raw metrics improved)
2. Comparisons for augmentations and ablations for SPA and IAI are presented (preferably on the datasets from point 1).

Looking forward to the discussion.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
