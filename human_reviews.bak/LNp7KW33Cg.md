# Stabilized Neural Dynamics for Behavioral Decoding via Hierarchical Domain Adaptation

- Decision: Reject
- Scores: 5, 5, 5, 5

## Abstract
Brain-Computer Interfaces (BCI) have demonstrated significant potential in neural rehabilitation. However, the variability of non-stationary neural signals often leads to instabilities of behavioral decoding, posing critical obstacles to chronic applications. Domain adaptation technique offers a promising solution. Nonetheless, the existing direct adaptation within latent spaces could result in feature deviations. Therefore, developing a stable and efficient alignment framework is crucial for neural decoders.
In this work, we find that dynamical latent features can be extracted from neural dynamics utilizing causal architectures. 
We also demonstrate that the process of self-consistent alignment can generate more stable latent features.
Based on these insights, we propose a novel hierarchical domain adaptation (HDA) method for the alignment of dynamical latent features.
Using Lyapunov theory, we further analytically validate the stability of dynamical features, which experimentally exhibit significant enhancements across various datasets.
Our HDA approach effectively addresses the challenge of non-stationary neural signals, thereby potentially improving the reliability of BCIs.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
Authors propose a neural alignment approach via hierarchical domain adaptation (HDA) in which they first align neural observations across Day 0 and Day K activity, followed by aligning their semantic latent activity. They show that HDA can perform alignment across various days of recordings better than baseline approaches.

### Strengths
Authors show that HDA can improve behavior decoding performance on Day K activity over baseline methods including single-session models of neural activity, e.g., CEBRA and vanilla LSTM, and other alignment approaches such as CycleGAN. Authors also provide a theoretical foundation based on Lyapunov theory for their 2-stage alignment approach.

### Weaknesses
- Major points:
  - It seems like the $E_\gamma$ is trained both on source and target domain data, rather than first training on Day 0 and aligning on Day K activity. However, I find this strategy counterintuitive for a neural alignment task. Shouldn’t authors first train their dynamic network $E_\gamma$ and decoder network $C_\eta$ on the source data initially? If the authors train their network every day from scratch with the new day’s data and source data, what is the advantage of the proposed framework over training completely a new model on new day’s data (as shown in Table 1 columns HDA and retrain)? The only advantage I can think of is the possibility of achieving better behavior decoding performance on Day K activity when the model is trained with extra data from Day 0, which does not seem to be the case. I understand that NoMAD also requires some information about Day 0 data to perform alignment such as latent factor distributions of Day 0 data, but such distributions can be computed across whole Day 0 data and stored to be used during alignment (which would not affect the alignment stage as latent factors for Day 0 data is obtained through frozen network during alignment). Even though this is not the case for ADAN, authors state that having a percentage of the Day 0 data for alignment (such as 50%, see https://github.com/limblab/adversarial_BCI/blob/main/ADAN_aligner.ipynb) is enough for the alignment stage after Day 0 training. I suspect the authors could have a similar strategy for their approach as they do not have any Day 0 training stage. I believe that this is particularly important since a BMI can be trained offline using a large dataset, but needing to have access to this large dataset during the alignment stage could limit the applicability of the decoder alignment. If the authors can, I think it is crucial to perform an ablation study on the amount of Day 0 data required for the alignment stage. 
  - It seems like NoMAD performance does not change across different days, which contradicts the findings shown in the NoMAD paper. Beyond that, I am concerned about NoMAD’s performance on Day 0, which is basically LFADS as no alignment takes place for Day 0 and it has shown great promise across multiple neural datasets (see https://eval.ai/web/challenges/challenge-page/1256/leaderboard/3184). I am surprised that LFADS, i.e., NoMAD Day 0, is performing worse than vanilla **unidirectional** LSTMs. I appreciate that the authors implemented NoMAD themselves as the public repository is not available, but authors can compare against ADAN, whose repository is publicly available, and verify if ADAN will follow a performance pattern. 
  - In line with my previous comment on LFADS, CEBRA performance on Day 0 also seems contradictory with prior findings as it again performs worse than vanilla LSTM. What positive/negative sampling strategy did the authors use while training CEBRA models? 

- Minor points:
  - NDT and STNDT do not aim to build representations that are generalized (unlike claimed in L125-126) across sessions and animals, as they are trained for individual sessions.
  - Regarding the claim about ‘independence of decomposed subspaces’ for VAEs in L272-273, are the authors explicitly learning covariances between different dimensions of latent factors and pushing them towards zero with KL divergence, or treating each latent dimension separately as commonly done in VAEs? If latter, I suspect the model is essentially learning disentangled latent variables, rather, the model tries to push individual latent dimension variances to 1, and learned nothing explicitly about their covariances.

### Questions
- Is $\tilde{Z}_d^T$ in L286 a typo and correspond to $\tilde{Z}_O^T$? 
- In L190-192, the authors mention aligning latent features spaces can be more challenging since spatiotemporal dependencies are more intricate, but for low dimensional latent factors (which is the case for authors as in L225), wouldn’t it be simple as they would capture less-noisy underlying patterns resulting in noisy spiking observations?
- Figure 6 is missing heatmap legends.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The paper created a hierarchical architecture for the alignment of the dynamical latent features. GAN is used for for the raw signal to match the distribution of the the source signal, semantic latent has been decomposed. The representation has been disentangled to get the domain-invariant features. The authors did experiments to show the decoding accuracy and ablation study on the each loss terms.

### Strengths
The paper uses simple networks to retrieve the neural representations and latents and fair experiments to compare with the existing models. The experiment results are clearly explained. The formulas are well notations. The frame of the paper is well presented.

### Weaknesses
The paper seems combining several things together, but kind of redundant work. First, GAN is used for matching the distribution, then a VAE is used for retrieving the latent. There is no need to use GAN since GAN and VAE do the same thing to approach the distribution, and for the latent decomposition in the same level, that does not change anything. The whole algorithm training process, need to alternately train several networks which seems not very efficient.

### Questions
One page 9, in the part Evaluation of Each Loss Term, the authors says from Table 3, all the loss terms are necessary. For the coefficient of the three loss terms, should they be given together as a tuple for one R square value? But in the table, I wander how the R square value is obtained giving only one parameter, but the other two are not given. Besides, to make a fire comparison, for example, setting the sum of the three coefficient to be one, and either fix the values or give a schedule for that to obtain the weights or necessaries? Also, from Table 7, $\lambda_b$ is relative small that could be omitted compared with the other two, which is a contradiction with the previous claim.

For the distribution divergence, can you give a reason why you choose the chi-square instead of KL divergence?

For the Lyapunov stability, since you are using the LSTM network, can you generalize it for complex network for rich representation?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The authors develop a model that stabilizes neural dynamics across recording sessions from multiple days. The authors first explain why this is an issue in real-life scenarios, and then develop a model that significantly improves decoding performance when transferred from one day to new days.

### Strengths
I think the authors do a good job explaining what types of methods exist to tackle this problem, from models that perform alignment on the neural manifold to methods that align data in the original data space. Moreover, the authors compare their model against good baselines, and the results are extremely encouraging. I have some comments about the results, but if those are addressed the contribution can be very big, given the importance of the problem the authors try to solve in this paper. I think the paper is relatively original, but will be helped if the authors improve its clarity.

### Weaknesses
Major weakness(es): \
I believe there are a few issues that together form a large reason why I would not currently recommend acceptance of this paper. First, the authors should improve the structure of the paper, while reading it I felt like the authors jumped back and forth between topics in the paper, e.g, the introduction and related work have a lot of overlap, but read disjointed. Additionally, I think the authors do not explain their intuition behind certain model choices enough, it may not be clear to readers why the authors are decomposing their dynamical latent space for example. Similarly, the theoretical verification of dynamical feature stability does not seem to be one of the main contributions of the paper, so I would move this section to the Appendix and instead focus more on how the authors are measuring stability, which is new and original for papers that try to stabilize neural dynamics across sessions. In terms of writing clarity I would lastly recommend that the authors improve their mathematical notation, the authors often do not use colloquial names but math symbols when referring to components in their model which is confusing, and the math symbols themselves are sometimes also confusing, e.g. the authors use $\mathbf{Z}, \mathbf{\tilde{Z}}, \mathbf{\tilde{Z}}_d^T, \mathbf{Z^S}, \mathbf{\tilde{Z}}_y^S $, and more all in a short section of text (L277-312). Additionally, the authors use $ \cdot^T $ as notation (e.g. $ X^T$, which is confusing because the T is normally used as a transpose operator. I think this paper could be improved tremendously by moving much of the math to the appendix (also because some of the equations are adapted from previous work, e.g. equations 1, 3, 4 are adapted from work the authors cite), and focusing on motivations for components, see [1] Section 3.3 for some tips on improvements. An intuitive explanation of many of the model’s components will greatly improve the readability of the paper and although it is good to adapt math from previous work and use it in your paper, in its current form the notation/equations reduce the clarity of the work.

Minor weaknesses:
1. The authors mention that their model is causal and thus better for real-time BCI applications. However, their model requires training data from a future day to be applicable to that day, so I don’t see how this would lead to real-time application of the current model. Specifically, the authors mention on Lines 380-382 that they use 20% of the sessions of the day that they want to transfer to as a test set. This means that without first training on the 80% of the unlabeled data on that day, the model cannot do any decoding yet. This in itself is not an issue, but one of the claims of the paper in the introduction is that “… we have the potential to meet the real-time operational requirements of BCIs”, which is a claim I would weaken given that atleast at the start of the day, the model requires training data to align first. 
2. The authors do not mention or compare to relevant work in neural population decoding [2]
3. L271-272, the authors should cite disentanglement works, e.g. [3, 4, 5] for the independence claim.
4. The authors should explain their intuition for using Chi-squared divergence more, not every reader may be familiar with Chi-squared divergence. Additionally, the authors should explain why they believe it is more accurate in measuring distribution discrepancies, since although Figure 7B (as mentioned on Line 199) does show that the author’s alignment method performs better, it is not immediately clear why this is the case and how other divergence methods compare to Chi-squared divergence.
5. Instead of using training time the authors should also (and more importantly) report the number of FLOPs since training time is very dependent on differences in hardware.
6. The authors do not compare their results with other methods for Figure 6.


Spelling/grammar:
- L144/145: “… over time as an UDA…” -> a UDA
- L175-177: “The aligned neural signals are then serve as...” -> are then provided
- L192: “This identification may help for aligning…” -> may help to align
- L245-246: “… can provide the dynamical system with an more efficient…” -> a more efficient

[1] Lipton, Z. C., & Steinhardt, J. (2019). Troubling Trends in Machine Learning Scholarship: Some ML papers suffer from flaws that could mislead the public and stymie future research. Queue, 17(1), 45-77. \
[2] Azabou, M., Arora, V., Ganesh, V., Mao, X., Nachimuthu, S., Mendelson, M., ... & Dyer, E. (2024). A unified, scalable framework for neural population decoding. Advances in Neural Information Processing Systems, 36. \
[3] Higgins, I., Matthey, L., Pal, A., Burgess, C. P., Glorot, X., Botvinick, M. M., ... & Lerchner, A. (2017). beta-vae: Learning basic visual concepts with a constrained variational framework. ICLR (Poster), 3. \
[4] Burgess, C. P., Higgins, I., Pal, A., Matthey, L., Watters, N., Desjardins, G., & Lerchner, A. (2018). Understanding disentangling in $\beta $-VAE. arXiv preprint arXiv:1804.03599. \
[5] Higgins, I., Amos, D., Pfau, D., Racaniere, S., Matthey, L., Rezende, D., & Lerchner, A. (2018). Towards a definition of disentangled representations. arXiv preprint arXiv:1812.02230.

### Questions
1. On line 348-349 the authors claim that pre-alignment helps minimize $ || x_i(t) - x_j(t)|| $, I am just wondering why that’s true. It seems to me that pre-alignment helps make the data more similar to the original training data, but there is no guarantee that in the original training data $ ||x_i(t) - x_j(t)|| $ is smaller than in the data for a new day.
2. The authors perform alignment between the label subspace from day 0 to a new day (see Figure 1). In that case: do the authors assume that the same activities are performed in the same order on day 0 and a new day? This seems to be a limiting assumption if that is the case. Comparatively, from reading NoMad, the authors assume that the distribution of all activities on one day are similar to all activities on another day, which is a less restrictive assumption. Can you clarify what assumption is accurate for your model?
3. The authors hypothesize in section 3.2.3 that drifts of dynamical latent features in the target domain primarily stem from latent variables that are loosely connected to observable behavioral variables. Can the authors expand on how they test this hypothesis.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
Summary: In this paper, the authors propose a method named HDA for the domain adaptation of high-dimensional neural observation data by performing the alignment in both the raw neural space and the latent space. It also links the connection to the Lyapunov exponent theory. In the emprical evalution part, the decoding results of HDA outperforms other baseline methods and manifest robustness.

### Strengths
1. How to resolve the domain turnover issue across days is a crucial task in neuroscience, which the paper is working on.
2. The paper as a whole is solid and all the components i.e., hierarchical alignment, causal architecture, used are sounded helpful for the stabilization task.
3. The experiments part are abundant and have a wide comparison across baselines and datasets.

### Weaknesses
1. The combination of the adversarial training and the hierarchical adaptation sounds tedious and a bit meaningless to me. If the proposed hierarchical domain adaptation is powerful enough, why do we still need the adversarial learning component. Thus, the contribution of the hierarchical adaptation is a bit weakened.
2. The theoretical verification is not strong enough since Lyapunov theory is actually a basic concept in control systems. Do you have more detailed and in-depth theory analysis on this part?
3. As shown in Line 277, the constraints added to the algorithm seems too many and no ablation study in the experiemtal part is provided.

### Questions
1. For the ERDiff baseline method [1], the reported results appear to differ from those originally presented. It also seems that CEBRA [2] is not a paper for the neural alignment task. Do you have a detailed re-run of ERDiff and CEBRA? 

[1] Extraction and Recovery of Spatio-Temporal Structure in Latent Dynamics Alignment with Diffusion Models. NeurIPS 2023

[2] Learnable latent embeddings for joint behavioural and neural analysis. Nature 2023.

### Soundness
3

### Presentation
2

### Contribution
2
