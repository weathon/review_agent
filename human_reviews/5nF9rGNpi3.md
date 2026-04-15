# From Fake to Real: Pretraining on Balanced Synthetic Images to Prevent Bias

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 3, 5, 6

## Abstract
Visual recognition models are prone to learning spurious correlations induced by a biased training set where certain conditions $B$ (\eg, Indoors) are over-represented in certain classes $Y$ (\eg, Big Dogs). Synthetic data from generative models offers a promising direction to mitigate this issue by augmenting underrepresented conditions in the real dataset. However, this introduces another potential source of bias from generative model artifacts in the synthetic data. Indeed, as we will show, prior work uses synthetic data to resolve the model's bias toward $B$, but it doesn't correct the models' bias toward the pair $(B, G)$ where $G$ denotes whether the sample is real or synthetic. Thus, the model could simply learn signals based on the pair $(B, G)$ (\eg, Synthetic Indoors) to make predictions about $Y$ (\eg, Big Dogs). To address this issue, we propose a two-step training pipeline that we call From Fake to Real (FFR). The first step of FFR pre-trains a model on balanced synthetic data to learn robust representations across subgroups. In the second step, FFR fine-tunes the model on real data using ERM or common loss-based bias mitigation methods. By training on real and synthetic data separately, FFR avoids the issue of bias toward signals from the pair $(B, G)$. In other words, synthetic data in the first step provides effective unbiased representations that boosts performance in the second step. Indeed, our analysis of high bias setting (99.9\%) shows that FFR improves performance over the state-of-the-art by 7-14\% over three datasets (CelebA, UTK-Face, and SpuCO Animals).

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper's objective is to improve the generative model-based de-biasing approach. The rationale behind this effort is rooted in the idea that the model's capacity to distinguish between 'real' and 'fake' images can be seen as another form of bias concept. To achieve this, the paper introduces a two-step training pipeline: initially, pre-training using balanced synthetic data, and subsequently, fine-tuning with real data.

### Strengths
- The rationale is commendable.
- The performance has shown a remarkable improvement.

### Weaknesses
- The approach is rather simplistic, which may affect the significance of its contribution.
- While there is some theoretical support for the notion that the model could learn to distinguish 'real or fake,' a quantitative analysis is necessary. This could involve assessing the model's performance on a discriminator, for instance.
- The notations employed in the script do not adhere to a professional standard, leading to potential confusion in interpretation. E.g., the notation $P_D(Y|B) \neq P(B)$ is unclear because, in practice, most conditional and marginal probabilities differ, making it challenging to discern the author's intention. Additionally, using $Y$ to represent labels is not accurate, as in a classification task, it typically employs a vector notation, such as $\textbf{y}$.
- The overall presentation would benefit from being more concise and streamlined.

### Questions
- How is the balancing of synthetic data achieved? Does this approach assume prior knowledge of the specific biases present in the dataset?
- It seems that the phenomenon of 'catastrophic forgetting' [1] in continual learning could potentially occur when transitioning from step 1 to step 2 if all model parameters are fine-tuned. Could you please clarify which parameters are trainable at each step?

[1] Kirkpatrick, James, et al. "Overcoming catastrophic forgetting in neural networks." Proceedings of the national academy of sciences 114.13 (2017): 3521-3526.

### Soundness
2 fair

### Presentation
1 poor

### Contribution
1 poor

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The paper first identifies a problem in previous works using synthetic images for bias mitigation—introducing a new source of bias due to the artifacts in synthesized images. To address this problem, the paper proposes From Fake to Real (FFR), a two-stage bias mitigation method. In the first stage, the model is pretrained on the balanced synthetic dataset. In the second stage, the model is finetuned on real data. Since FFR separates the synthetic and real data in two training stages, the identified problem is addressed. In the experiments, FFR outperforms existing methods in high-bias settings on three datasets.

### Strengths
1. The paper is well-motivated. The paper identifies the problem of previous works using synthetic datasets for debiasing—introducing a new bias source of synthetic artifacts, which is a very good motivation for the proposed method to address the problem.
2. The paper also presents a theoretical analysis (Theorem 1) of using synthetic datasets for debiasing in previous works.
3. The proposed method, two-stage training, is simple yet effective.
4. The paper is well-written and easy to follow.
5. The paper provides code for better transparency and reproducibility.

### Weaknesses
### Major Concerns

**Missing Comparison Methods**:
* The paper fails to add comparison methods to address the problem of the newly introduced synthetic artifact bias—using more bias groups in debiasing methods that use group labels: such as GroupDRO (Sagawa et al, 2020a), SUBG [1], DFR (Kirichenko et al, 2023), and Domain Independent (Wang et al., 2020c). While the authors argue that “in order to account for the new source of bias from $(B, G)$ where $G = \{ Real, Synthetic \}$, this approach doubles the number of bias groups ($| (B, G) | = | B || G | = 2 | B |$) which increases the optimization difficulty, reducing performance,” no experimental results are presented to support this claim. In fact, in a multi-shortcut (i.e., multi-bias) benchmark, Li et al. [2] show that the aforementioned four methods can successfully mitigate multiple biases simultaneously (see Table 5 in [2]).
* The paper also fails to compare with methods that infer group (or environment) labels, including LfF [3], George [4], EIIL [5], JTT [6], PGI [7], DebiAN [8], and CnC [9]. By inferring group labels instead of using the predefined one, these methods have the opportunity to mitigate the newly introduced artifact bias (or even other unknown biases).


**High bias settings are unrealistic**: I agree with the authors that the proposed FFR achieves better performance than existing approaches in the high-bias settings (e.g., 99.9% bias ratio). However, I think such extreme bias ratios (95% to 99.9% bias ratios in Figure 3) are too artificial to be found in real-world data. Thus, I encourage the authors to add experiments on real-world data without manipulating the bias ratios.

**Ablation study on the ordering of two stages**: The paper lacks a critical ablation study on the ordering of the two stages in FFR. Why not train on real data first and then train on the synthetic data? I think this alternative is more feasible—most computer vision models are pretrained on real-world data and further finetuning these models on balanced synthetic data is more practical.

**Biases in Stable Diffusion**    The paper uses Stable Diffusion (SD) v1.4 (“implementation details” on page 5) to generate the synthetic images. However, using SD to generate training has many problems:
* First, there is no guarantee that SD will faithfully follow the text prompts for image generation. In fact, SD and many other text-to-image models often struggle with compositional text prompts, e.g., spatial relationships [10] or attribute composition [11]. Thus, it is questionable whether SD can faithfully generate images by following the prompt template of “A photo of { bias } { class }” (“implementation details” on page 5).
* Even assuming that SD can faithfully follow the text prompt, new biases that are not controlled (e.g., explicitly specified) by the text prompts may also be introduced. For example, when generating female smiling images for CelebA HQ dataset, SD may generate stereotypical long-hair female images, leading to newly introduced hair length bias. Not mitigating the newly introduced biases has serious consequences—the amplification of the biases that are not mitigated in multi-bias scenarios [2].

### Minor Concerns
* The paper fails to discuss many works on “uncovering spurious correlations” (Section 2, Page 3). First, as mentioned above, many debiasing methods [3-9] that infer group labels are not discussed. Second, methods directly focused on bias detection are not discussed, such as methods based on clustering [12-14] or classification [15], generative models [16-18], explainability methods [19], and concepts [20,21].
* In the code, I found that the experiments use ImageNet pretrained weights to initialize the model. I wonder if this is consistent with the claim that the first stage of the FFR method “learns unbiased representations” for initialization.
* On the UTK-Face dataset, the authors “use age as the bias attribute and gender as the target attribute” (“Datasets” in Section 4, page 5). However, I don’t think this is reasonable because the age attribute is shown to be more difficult to learn than gender in two papers (see Figure 3 in [22] and Table 11 in (Ramaswamy et al., CVPR, 2021)). As shown in [3], the bias can only be learned if and only if the bias attribute is easier to learn than the target attribute.
* Regarding the synthetic datasets generated by SDv1.4, will they be released for reproducibility purposes?

### Minor Comments

* Section 4.3: Figure 5 reports our … -> Figure 4 reports our
* All quotation marks in the paper are misused (e.g., ”spurious” in Section 3, ”Big Dogs Indoors”, ”Small Dogs outdoors”, and ”toes” in Section 4.4). Please use ``’’ for quotation marks in LaTeX.
* The reference to “Kai Xiao, Logan Engstrom, Andrew Ilyas, and Aleksander Madry. Noise or signal: The role of image backgrounds in object recognition. ArXiv preprint arXiv:2006.09994, 2020.” (page 12) is outdated. The paper was accepted to ICLR 2021 [23].

### References

[1] Badr Youbi Idrissi, Martin Arjovsky, Mohammad Pezeshki, and David Lopez-Paz, “Simple data balancing achieves competitive worst-group-accuracy,” in CLeaR, 2022.

[2] Zhiheng Li, Ivan Evtimov, Albert Gordo, Caner Hazirbas, Tal Hassner, Cristian Canton Ferrer, Chenliang Xu, and Mark Ibrahim, “A Whac-A-Mole Dilemma: Shortcuts Come in Multiples Where Mitigating One Amplifies Others,” in CVPR, 2023.

[3] Junhyun Nam, Hyuntak Cha, Sungsoo Ahn, Jaeho Lee, and Jinwoo Shin, “Learning from Failure: Training Debiased Classiﬁer from Biased Classiﬁer,” in NeurIPS, 2020.

[4] Nimit S. Sohoni, Jared A. Dunnmon, Geoffrey Angus, Albert Gu, and Christopher Ré, “No Subclass Left Behind: Fine-Grained Robustness in Coarse-Grained Classification Problems,” in NeurIPS, 2020.

[5] Elliot Creager, Joern-Henrik Jacobsen, and Richard Zemel, “Environment Inference for Invariant Learning,” in ICML, 2021.

[6] Evan Z. Liu, Behzad Haghgoo, Annie S. Chen, Aditi Raghunathan, Pang Wei Koh, Shiori Sagawa, Percy Liang, and Chelsea Finn, “Just Train Twice: Improving Group Robustness without Training Group Information,” in ICML, 2021.

[7] Faruk Ahmed, Yoshua Bengio, Harm van Seijen, and Aaron Courville, “Systematic generalisation with group invariant predictions,” in ICLR, 2021.

[8] Zhiheng Li, Anthony Hoogs, and Chenliang Xu, “Discover and Mitigate Unknown Biases with Debiasing Alternate Networks,” in ECCV, 2022.

[9] Michael Zhang, Nimit S. Sohoni, Hongyang R. Zhang, Chelsea Finn, and Christopher Re, “Correct-N-Contrast: a Contrastive Approach for Improving Robustness to Spurious Correlations,” in ICML, 2022.

[10] Tejas Gokhale, Hamid Palangi, Besmira Nushi, Vibhav Vineet, Eric Horvitz, Ece Kamar, Chitta Baral, and Yezhou Yang, “Benchmarking Spatial Relationships in Text-to-Image Generation.” arXiv, 2022.

[11] “T2I-CompBench: A Comprehensive Benchmark for Open-world Compositional Text-to-image Generation,” NeurIPS Datasets and Benchmarks Track, 2023.

[12] Arvindkumar Krishnakumar, Viraj Prabhu, Sruthi Sudhakar, and Judy Hoffman, “UDIS: Unsupervised Discovery of Bias in Deep Visual Recognition Models,” in BMVC, 2021.

[13] Sabri Eyuboglu, Maya Varma, Khaled Kamal Saab, Jean-Benoit Delbrouck, Christopher Lee-Messer, Jared Dunnmon, James Zou, and Christopher Re, “Domino: Discovering Systematic Errors with Cross-Modal Embeddings,” in ICLR, 2022.

[14] Gregory Plumb, Nari Johnson, Angel Cabrera, and Ameet Talwalkar, “Towards a More Rigorous Science of Blindspot Discovery in Image Classification Models,” TMLR, 2023

[15] Saachi Jain, Hannah Lawrence, Ankur Moitra, and Aleksander Madry, “Distilling Model Failures as Directions in Latent Space,” in ICLR, 2023.

[16] Zhiheng Li and Chenliang Xu, “Discover the Unknown Biased Attribute of an Image Classifier,” in ICCV, 2021.

[17] Oran Lang, Yossi Gandelsman, Michal Yarom, Yoav Wald, Gal Elidan, Avinatan Hassidim, William T. Freeman, Phillip Isola, et al., “Explaining in Style: Training a GAN to explain a classifier in StyleSpace,” in ICCV, 2021.

[18] Jan Hendrik Metzen, Robin Hutmacher, N. Grace Hua, Valentyn Boreiko, and Dan Zhang, “Identification of Systematic Errors of Image Classifiers on Rare Subgroups,” in ICCV, 2023.

[19] Sahil Singla and Soheil Feizi, “Salient ImageNet: How to discover spurious features in Deep Learning?,” in ICLR, 2022.

[20] Shirley Wu, Mert Yuksekgonul, Linjun Zhang, and James Zou, “Discover and Cure: Concept-aware Mitigation of Spurious Correlation,” in ICML, 2023.

[21] Abubakar Abid, Mert Yuksekgonul, and James Zou, “Meaningfully Debugging Model Mistakes using Conceptual Counterfactual Explanations,” in ICML, 2022.

[22] Luca Scimeca, Seong Joon Oh, Sanghyuk Chun, Michael Poli, and Sangdoo Yun, “Which Shortcut Cues Will DNNs Choose? A Study from the Parameter-Space Perspective,” in ICLR, 2022.

[23] Kai Yuanqing Xiao, Logan Engstrom, Andrew Ilyas, and Aleksander Madry, “Noise or Signal: The Role of Image Backgrounds in Object Recognition,” in ICLR, 2021.

### Questions
I expect the authors to address my concerns listed in the weaknesses section:

* Adding more comparison methods.
* Adding experiments on real-world datasets with real spurious correlations, such as ImageNet background challenge [23] and ImageNet-W [2].
* Adding the ablation study on the ordering of two stages.
* Discussing the bias problem caused by SD.

### Soundness
2 fair

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
This work considers the problem of bias mitigation on image classification tasks and tackles it using balanced synthetic data. More specifically, authors claim naively combining real data with balanced synthetic might not be an efficient way to deal with bias due spurious correlations because the model might exploit potentially existing differences between the real and synthetic distributions when making predictions. To address this issue, authors then propose From Fake to Real (FFR), a two step training framework for bias mitigation. In FFR, a model is trained with balanced synthetic data only, followed by training with samples from the original real distribution. Authors used an off-the-shelf diffusion model to generate balanced synthetic data and compared FFR with other approaches in the literature that considered training a model in a single step by combining synthetic and real data. Additionally, authors combined the proposed training framework (as well as the baselines) with techniques to mitigate bias based such as GroupDRO. The empirical evaluation considered a single architecture (ResNet-50) and three datasets. Overall, results showed that FFR is more effective at bias mitigation as per two metrics, worst group accuracy and balanced accuracy.

### Strengths
- The manuscript is well-written
- The work is well-motivated and the authors contextualized their contributions with respect to prior work
- Contributions are potentially relevant as the work touches on a topic of utmost importance for the machine learning community
- The proposed technique is intuitive and seems relatively easy to implement/reproduce in case one assumes access to the same generative model used in the manuscript 
- The empirical evaluation takes into account multiple baselines and datasets

### Weaknesses
- FFR, the central contribution of this work, is based on the assumption that a model can differentiate between real and synthetic images and is likely to leverage correlations that are not stable between these two sources of data to make predictions, which could lead to biased predictions in the real dataset. However, there is no discussion or result (theoretical or empirical) in the manuscript to support this rather strong claim. (See the next section for questions related to this point)

- The authors consider a two-step training approach, but only report and discuss in the manuscript the performance of the second step, making it hard to fully understand whether (and to what extent) the model is successful at prediction after being trained with balanced synthetic data only. Moreover, the manuscript does not report/investigate whether models are encoding biases or not after training with balanced synthetic data. 

- This contribution heavily relies on the use of synthetic datasets, however, there is no discussion regarding the quality of the generated data.  (See the next section for questions related to this point)

- No error bars or confidence intervals were reported in the results and it seems that a single training was performed for each model, making to hard to evaluate how robust the finding from the experimental evaluation are.

- It is not clear from the manuscript whether all approaches trained with synthetic data had access to the same amount of synthetic instances at training time. If not, the observed conclusions from the experiment could have been confounded by the different number samples in each training dataset.

- The authors used an off-the-shelf generative model to try to approximate the real data distribution, but it seems to me that training specialized models for each dataset could yield higher quality data while also helping to mitigate bias due to unbalanced subgroups [1].  

[1] Ktena et al., Generative models improve fairness of medical classifiers under distribution shifts, 2023.

### Questions
- What if the considered model class does not capture the artifacts that make real and synthetic distinguishable? How would the improvements obtained with FFR reported in the manuscript would be impacted in this case? 

- How would the performance of the approach be impacted by low-quality generated data? More specifically, what would happen in case the generative is not able to generate some classes in the original dataset with good quality?

- Why were no augmentations used to train the ResNet-50s? Could this cripple the ERM+No synthetic data baseline, thus rendering the comparison unfair?

- The authors mentioned in the limitation section that the employed datasets are quite small. Given that and the fact that no augmentations were used, could the improvements observed on top of the baseline that does not have access to synthetic data be simply due to the fact the training data is larger after synthetic data is added?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper presents a method to tackle spurious correlations present in training data, that translate in subpar performance when evaluation tasks do not show these spurious correlations. To do so, the paper proposes using a large-scaled pretrained generative image model to sample for the inbalanced classes, which allows to produce a balanced dataset that does not show the spurious correlation. Despite this, the paper argues (and shows through experiments), that the approach of either training with only generative samples or with the naive combination of synthetic and real is suboptimal, as the model can exploit the shortcut solution of distinguish between real and fake data to gain information about the training task. To overcome this, the paper proposes a two stage method of 1) first training with synthetic data and then 2) finetuning with real data, which results in improved performance when compared to alternatives.

### Strengths
- The motivation is sound and is supported by simple but complete theory.
- The method is simple and relatively easy to apply, although it requires large-scale generative image models, sampling from them and a two stage training process.
- Quantitative results show good performance when compared to simple alternatives, showing that when using synthetic data to rebalance the dataset, the two stage method improves performance significantly.
- Qualitative results (Figure 5) illustrate that the proposed method is more able to ignore spurious correlations than alternatives, as the saliency maps are concentrated in the classification task at hand.

### Weaknesses
- The main issue of the paper is that it ignores the (large-scale) datasources used to train the generative model, and how they can be leveraged as simple baselines for the proposed method, by either finetuning a pretrained classification model (like OpenCLIP [1]) or retrieving images from these datasets to reduce the spurious correlations. Alternatively, a fair comparison would be to use a generative model trained only on the datasets being evaluated.

- In this regard, the model does not compare against using a pretrained backbone model on the same or similar large-scale datasources (LAION) and then testing it zero-shot, finetuning it, or learning a shallow classifier on top. This is a simple baseline with similar trade-offs as using a large scale generative model in terms of off-the-shelf availability, but has the advantage of not requiring sampling nor a two stage method. It may be the case that large-scale classification models are already able to disentangle and ignore the spurious correlations. Does a pretrained model on LAION-5B for classification, and then lightly finetuned for these tasks perform better or worse than the proposed full method? 

- As pointed out in the limitations, the experiments are performed on a "toyish" set of tasks with small amounts of data and simple biases, while the method uses large-scale data. These simple biases showcased may not be as prevalent in the training dataset used for the generative model (LAION-5B), as LAION will likely have much greater variation. For example, it is likely that the smiling bias in people is more prevalent in CelebA HQ than in LAION, and a model trained on CelebA HQ is likely to pick it up because CelebA doesn't have much variation. Although how to best retrieve and use this data to tackle bias is out of the scope of the paper, it would be good to have simple baselines, like retrieving all or a subset of {class} images from LAION and training the first stage with that.

- The experiment does not show results for FFR without rebalancing, which is a simpler setting, as one doesn't need to know the {bias} attributes that cause the spurious correlations to generate images. It has been shown that just randomization can tackle domain missmatch and alleviate spurious correlations [2], making models perform well during evaluation, and this effect is not considered. Given that the generative model is powerful, it is possible that it is introducing extra variation (e.g. lots of different poses, background...), that biases the model to ignore the spurious correlation. To test that the hypothesis that *rebalancing* is the main driver of improved performance and not *extra random diversity* gained by the usage of large scale generative models, it would be good to compare the proposed rebalanced sampling method for the prompts to random sampling (i.e. instead of A photo of {bias} {class}, A photo of {class}, or A photo of {random word} {class}).

Despite these concerns, which if addressed, would make for a stronger paper, I believe the paper as it stands adds value and will be of interest to the community.


[1] https://github.com/mlfoundations/open_clip
[2] https://arxiv.org/abs/1710.06425

### Questions
See weaknesses.

Typo: Uncovering Spurious Correlations. In our work, we are interested *in* mitigating

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair
