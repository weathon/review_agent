# Pre-Training Under Infinite Compute

Konwoo Kim∞, Suhas Kotha∞**, Percy Liang, Tatsunori Hashimoto**
Stanford University

## Abstract

Since compute grows much faster than web text available for language model pre-training, we ask how one should approach pre-training under fixed data and no compute constraints. We first show that existing data-constrained approaches of increasing epoch count and parameter count overfit, and we improve upon such recipes by tuning regularization, finding that the optimal weight decay is 30× larger than standard practice. Since our regularized recipe monotonically decreases loss following a power law in parameter count, we estimate its best possible performance via the **asymptote** of its scaling law rather than the performance at a fixed compute budget. We then identify that ensembling independently trained models achieves a significantly lower loss asymptote than the regularized recipe. Our best intervention combining epoching, regularization, parameter scaling, and ensemble scaling achieves an asymptote at 200M tokens using 5.17× less data than our baseline, and our data scaling laws predict that this improvement persists at higher token budgets.

We find that our data efficiency gains can be realized at smaller parameter counts as we can distill an ensemble into a student model that is 8× smaller and retains 83% of the ensembling benefit. Finally, our interventions designed for validation loss generalize to downstream benchmarks, achieving a 9% improvement for pretraining evals. Our results show that simple algorithmic improvements can enable significantly more data-efficient pre-training in a compute-rich future.

## 1 Introduction

Language model pre-training has historically been studied under compute constraints at training (Kaplan et al., 2020; Hoffmann et al., 2022) and inference (Brown et al., 2024; Snell et al., 2024)
while assuming access to unlimited web text. However, web data grows by 1.03× per year, whereas compute spent on pre-training grows by 4× per year (Villalobos et al., 2024; Sevilla and Roldán, 2024). In anticipation of a regime where compute vastly exceeds data, we ask:
How should one approach pre-training under fixed data and no compute constraints?

To establish a baseline, we fix a seed training corpus of 200M tokens of web text and evaluate a standard recipe following existing data-constrained approaches of repeating data (Muennighoff et al., 2023) and increasing parameter count (Kaplan et al., 2020) (Section 2). We find that either too many epochs or too many parameters results in the loss eventually increasing due to overfitting. This bounds the performance improvements we can get from tuning this recipe, even if we were willing to spend more compute in exchange for a better model. We instead get predictable monotone scaling in parameter count by considering a *regularized recipe* (Section 3). Currently, regularization used in pre-training is often adopted from existing recipes, defaulting to a weight decay of 0.1 from Brown et al. (2020). We find this amount to be inadequate for preventing overfitting under data constraints as the optimal weight decay is 30× larger than standard practice for our most over-parameterized models. After jointly tuning weight decay, learning rate, and epoch count at each parameter count N, loss closely follows a power law in N for parameter-to-token ratios 140× larger than Chinchilla, as shown in Figure 1.

Normally, we would compare two recipes by evaluating performance at different train or inference compute budgets (Hoffmann et al., 2022; Snell et al., 2024). However, this does not reflect our
∞Equal contribution.

1

![1_image_0.png](1_image_0.png)

interest in the best possible performance under fixed data and no compute constraints. Since the loss of the regularized recipe continues to decrease as N increases, we are interested in the limit of the loss as N → ∞. More generally, we propose evaluating monotone scaling recipes by the **asymptote** of their scaling law (e.g. 3.43 for the regularized recipe as seen in Figure 1). By preferring recipes with lower loss asymptotes, we can train better models at sufficiently high compute budgets. Though taking the parameter count to infinity is one possible limit under infinite compute, we ask if we can design recipes with even lower asymptotes. We consider an alternative *ensembling recipe* where we average the logits of K independently trained models of the same size (Section 4). The ensembling recipe achieves a lower asymptote as K → ∞ compared to the regularized recipe as N → ∞ (Figure 1). At sufficiently high parameter counts, it is better to train multiple smaller models instead of a single larger model. We further show that ensembling and parameter scaling compose, achieving a lower asymptote when following the *joint scaling recipe* of taking both K, N → ∞. Since our previous experiments were on the scale of 200M tokens, we study how our recipes scale across higher token counts and find that the asymptotes themselves follow a scaling law (Section 5). Our estimates indicate that the joint scaling recipe achieves its 200M asymptote with 5.17× less data than the standard recipe. Importantly, extrapolation of our data scaling laws indicates that the data efficiency improvements will persist at higher token counts. Though the asymptotes of our recipes benefit the most from large parameter counts, we find that distillation (Hinton et al., 2015; Kim and Rush, 2016) allows us to retain most of the loss improvements without increasing inference parameter count. Distilling an 8-ensemble into a single 300M
model retains 83% of the ensembling loss improvement over the best regularized 300M model and outperforms the asymptote of the regularized recipe. We also find that self-distilling a 300M teacher into a student of the same size reduces loss, improving data efficiency without ever explicitly training a model of higher parameter count. Finally, we confirm that improvements on validation loss translate to improvements on downstream benchmarks (Section 7). Ensembles with better validation loss perform better on downstream benchmarks, with our best ensemble outperforming our best unregularized model by 9% on average over PIQA, SciQ, and ARC Easy (standard benchmarks for models at our scale (Thrush et al., 2025)).

## 2 Standard Pre-Training

Historically, pre-training has focused on training the best possible models subject to compute or parameter constraints. Under train compute constraints, recipes like Chinchilla recommend jointly scaling data and model size with 20× more tokens than parameters (Kaplan et al., 2020; Hoffmann et al., 2022). Under parameter constraints for cheaper inference, current practice opts for over-training language models relative to Chinchilla with token counts 2000× larger than parameter counts or distilling from preexisting larger models (Gadre et al., 2024; Grattafiori et al., 2024; Sardana et al.,
2025; Busbridge et al., 2025). Prior works prescribe such scaling recipes by always training on fresh data. In this paper, we instead study data-constrained pre-training, where we cannot jointly scale data and model size. We analyze the purest form of the problem by lifting all other constraints (including compute) besides data. We formalize standard pre-training as a training routine A that accepts arguments such as token count D, parameter count N, epoch count E to produce a model M with loss L(M). Unspecified arguments are passed through hyperparameter tuple H. Our data-constrained pre-training objective becomes L
∗D = minH L(A (D, H)).

We construct a controlled pre-training environment with a limited amount of web data from DCLM (Li et al., 2025). Since our algorithms spend more compute than Chinchilla scaling at a fixed data budget, we default to 200M tokens and test whether our findings hold across higher token counts in Section 5. For evaluation, we defer to loss on a held-out i.i.d. validation set which is shown to correlate with downstream capabilities in Section 7 and prior work (Chen et al., 2025; Thrush et al., 2025; Gadre et al., 2024). To best represent standard practice, we follow a standard auto-regressive recipe (full details in Appendix B).

## 2.1 Evaluating Existing Data-Constrained Recipes

Since the amount of fresh data is limited, we build a standard recipe of increasing repetition count (Muennighoff et al., 2023) and parameter count (Kaplan et al., 2020). Since there is unlimited compute, we depart from compute-efficient practice by training models that are larger relative to the token count, defaulting to 300M parameter models for 200M tokens. We first increase the epoch count E at a fixed parameter count, taking E× more training compute. Figure 2 (left) shows for high E, overfitting occurs and loss increases. These findings contradict the functional form of the decay-based scaling law in Muennighoff et al. (2023), which posits that loss monotonically decreases in E. Their work acknowledges this discrepancy and removes most overfit runs from their scaling law (see their Appendix D). Since increasing epoch count arbitrarily hurts loss, we turn to increasing parameter count. To establish a competitive baseline, we jointly tune epoch count and learning rate for each parameter

![2_image_1.png](2_image_1.png)

![2_image_0.png](2_image_0.png)

Figure 2: **Evaluating standard recipe of epoching and parameter scaling for 200M tokens.** Left: Though repeating the data lowers the loss, too many repetitions results in overfitting for 300M models. Right: We try increasing parameter count, tuning the epoch count at each parameter count. We similarly find that loss starts increasing. Moreover, increasing the parameter count 10× improves the loss by less than 0.1. count (Appendix C.1). We find minimal improvement in loss with higher model size, with our 1.4B model performing worse than our 600M model. This is consistent with the single-pass findings of Kaplan et al. (2020), Figure 9 which show that increasing parameter count eventually starts increasing loss for fixed data. It is likely that both higher epoch and parameter count result in overfitting the train set, detailed in Appendix C.5.

## 3 Regularized Parameter Scaling

We show that to get the best performance from these over-parameterized, epoched models, it is critical to regularize pre-training with much higher weight decay than standard practice. To jointly tune weight decay, learning rate, and epoch count, we perform an extensive search for "locally optimal" hyperparameters using a coordinate descent algorithm inspired by Wen et al. (2025) (details in Appendix C.1). We find that over-parametrized models need much higher weight decay, over 30× larger than the standard practice of 0.1 (Figure 3, right table). With this tuning, loss follows monotone scaling in parameter count for models up to 140× larger than Chinchilla as shown in Figure 3. This agrees with theory for over-parameterized regression that predicts that even when loss does not monotonically decrease due to double descent, the loss will monotonically decrease when regularization is optimally tuned (Advani and Ganguli, 2016; Nakkiran et al., 2021; Canatar et al., 2021; Simon et al., 2024). In Appendix C.2, we show how locally-optimal tuning is critical to achieve monotone scaling. To capture how increasing parameter count improves loss, we fit a power law with an asymptote as LˆD,N := AD
NαD + ED where we fit free variables AD, αD, ED. Our fit across four parameter counts results in Lˆ200M,N =0.05 N1.02 + 3.431. The exponent of 1.02 for parameter scaling is high given that Chinchilla finds a parameter scaling exponent of 0.34. This suggests that when we better leverage the data, there is faster improvement from larger models.

Our monotone scaling law differs from compute-optimal prescriptions where increasing N can hurt performance due to training on less data. We characterize our best possible performance unconstrained by compute as limN→∞ LˆD,N e.g. the asymptote ED. The asymptote for the regularized recipe law predicts that the best possible model achieves loss 3.432.

## 4 Ensemble Scaling

![3_Image_0.Png](3_Image_0.Png)

| Tuned H       | 150M   | 300M   | 600M   | 1.4B   |
|---------------|--------|--------|--------|--------|
| Learning rate | 3e-3   | 3e-3   | 1e-3   | 1e-3   |
| Epoch count   | 16     | 16     | 8      | 8      |
| Weight decay  | 0.8    | 1.6    | 3.2    | 3.2    |

Figure 3: **Power law scaling from jointly tuning** regularization. We compare the standard recipe in Figure 2 (red line) to our regularized recipe that jointly tunes learning rate, epoch count, and weight decay (purple line). After regularization, the loss decreases proportional to ≈1N
. The power law predicts that as N → ∞, the best model achieves 3.43 loss.

The regularized recipe offers a straightforward way to improve performance by taking N → ∞. Can different training algorithms better leverage the data under infinite compute? In this section, we consider ensembling (Dietterich, 2000): independently train K models and average their logits for generation (Section 4.1). In Section 4.2, we show how ensembling can outperform parameter scaling at fixed parameter counts and under the limit as total parameter count approaches infinity. In Section 4.3, we construct our best recipe composing regularized parameter scaling and ensemble scaling by taking the limit as both N, K → ∞.

## 4.1 Defining Ensembles

The ensembling pre-training algorithm E accepts a pre-training algorithm A, trains K members that

![4_image_0.png](4_image_0.png)

are identical up to random seed Zi controlling data order and model initialization, and returns a model that averages the logits of the K members. See Appendix D.1 for a full formal definition. The number of FLOPs needed to generate from or evaluate an ensemble is simply the sum of the costs for all members. Since the number of FLOPs in a forward pass is approximately linear in parameter count (Kaplan et al., 2020; Hoffmann et al., 2022), we will consider an ensemble's total parameter count as NK when comparing it to standard pre-training.

Figure 4: **Comparing scaling parameter count**
vs scaling ensemble member count. Instead of scaling the parameter count of a single model, we can train an ensemble of smaller models and scale the number of ensemble members (resulting in NK total parameters for K ensemble members). Scaling up member count K can similarly be fit by a power law with exponent approximately 1. Importantly, this law achieves a better asymptote than scaling N.

We compare the regularized and ensembling recipes under the best regularized hyperparameters from Section 3. In Figure 4, we find that the ensembling recipe's excess loss decreases close to a rate of 1 K
, similar to how the regularized recipe's excess loss decreases at a rate close to 1N
. Under infinite compute, the ensembling recipe's (N = 300M, K → ∞) asymptote is 3.34, which is lower than the regularized recipe's (N → ∞, K = 1) asymptote of 3.43. Thus, for large N, it is better to train multiple small models instead of a single large model. In fact, even the K = 3 ensemble outperforms the regularized recipe's asymptote. Why does ensembling improve over parameter scaling? Allen-Zhu and Li (2023) shows that ensembling helps when the data can be well-classified with one of many features but is best classified using all such features. Under this "multi-view" structure, they find that a single model is only learns one feature, whereas each ensemble member learns different features. 3

## 4.3 Joint Scaling Recipe Composing Parameter And Ensemble Scaling

Although the ensembling recipe outperforms parameter scaling, we can compose both by taking the number of members and the size of each member to infinity (N, K → ∞). To estimate the best possible loss of a joint scaling recipe, we take two limits:

$$\hat{\mathcal{L}}_{D}=\operatorname*{lim}_{N\to\infty}\operatorname*{lim}_{K\to\infty}\operatorname*{min}_{H}{\mathcal{L}}\left({\mathcal{E}}_{\mathcal{A}}\left(D,N,K,H\right)\right)$$

As long as minH L(EA (*D, N, K, H*)) monotonically decreases in N and K when fixing the other variable, the value does not depend on the order of the limits. We choose this order as it results in the most convenient hyperparameter tuning (Appendix D.6). For the inner limit, we cannot fully find locally optimal hyperparameters due to experimental constraints. Instead, we use the heuristic of taking the optimal regularized hyperparameters with 2× epochs and 0.5× weight decay (Appendix D.4). In Figure 5, we show how we take this double limit. Our final estimate for the joint scaling recipe's loss is 3.17, which is much better than the regularized and unregularized losses of 3.43 and the 3.75.

3In Appendix D.2, we find optimally tuned ensembles match this intuition. Slightly overfitting each ensemble member beats an ensemble using the best regularized hyperparameters.

![5_image_0.png](5_image_0.png)

## 5 Scaling The Seed Token Count Under Infinite Compute

Do our loss improvements at 200M tokens generalize to larger scales? In Sections 5.1 and 5.2, we

![5_image_1.png](5_image_1.png) first measure the best possible loss of our recipes at higher token counts up to 1.6B tokens. We contextualize the loss improvement and **data efficiency** of a recipe by interpolating how much data the standard recipe would need to match performance. In Section 5.3, we fit data-scaling laws to extrapolate how our recipes would perform at even higher token counts. 5.1 DATA SCALING LAWS FOR SINGLE MODEL RECIPES
As shown in Section 2, the standard recipe overfits and does not admit a monotone scaling law.

Instead, we search for the best parameter count and hyperparameters at each of our four data scales (Appendix E.1). Given these four estimates of the best loss at each token count (Figure 6, right), we fit a data scaling power law shown as the red line using LˆD := A
Dα + E.

We characterize the best possible loss of the regularized recipe by estimating limN→∞ minH L(A (*D, N, H*)) as shown in Section 3. Since we have to compute asymptotes to build the points for the data scaling law, we follow a two step procedure shown in Figure 6.

Measuring data efficiency. We measure the data efficiency between two recipes at a fixed token count D. We first compute the effective data D′that A1 would need to match A2. After interpolating D′ via the data scaling law of A1, we report the data efficiency as D′
D
. This metric characterizes the regularized recipe asymptote as 2.29× more data efficient than the standard recipe at 200M tokens. Even without any extrapolation of the asymptote, the best 1.4B model at 200M tokens is 2.09× more data efficient than our baseline.

## 5.2 Data Scaling Laws For Ensembles

We repeat the above procedure for ensembles by following Section 4.3 and estimating

![6_image_0.png](6_image_0.png)

limN→∞ limK→∞ minH L(EA (*D, N, K, H*)) for each seed token count D. We visualize the three step procedure in Figure 7. At 200M tokens, the asymptote of the joint scaling recipe is 5.17× more data efficient than the standard recipe. Without taking asymptotes, our best ensemble of five 1.4B models is itself 3.75× more data efficient.

## 5.3 Data Scaling Analysis

Although the data scaling laws are expected to be noisy, they predict that all recipes decay at a similar rate with exponents between 0.23 and 0.24 and asymptotes between 1.89 and 1.96. Asymptotic statistics suggests that the asymptotes are equal if the algorithms achieve Bayes-optimal error under infinite data and compute, in which case their loss would be the entropy of text (Shannon, 1951; Van der Vaart, 2000). When the asymptote E and exponent α of the laws are the same for two algorithms, there is a constant data efficiency improvement at all token counts determined by the numerators A1, A2, equal to (A2/A1)
1 α . Our preliminary analysis suggests that our data efficiency wins will not disappear across all data scales even if they perform similarly under infinite data.

## 6 Data Efficiency Under Parameter Constraints

The asymptotes of the regularized and ensembling recipes rely on arbitrarily high parameter models We study whether large models are necessary for data efficiency, either for the final model or for training. In Section 6.1, we distill an 8-ensemble of 300M members into a 300M student, preserving 83% of the loss improvement with an 8× smaller final model. In Section 6.2, we show self-distilling a 300M model into a student of the same size outperforms the teacher, removing the need for large parameter counts at training.

## 6.1 Reducing Final Parameter Count Via Ensemble Distillation

Even if our best scaling recipe helps in the limit as N, K → ∞, can it help train models that are small relative to D? It is known that better large models can improve the performance of smaller models through knowledge distillation (Hinton et al., 2015; Yang et al., 2025; Team et al., 2025a).

Since we are not bound by train compute, we can first pre-train a data-efficient teacher M′ on D tokens using our existing recipes. Then, we sample from M′ unconditionally (i.e. with no prompt) to generate D′tokens. We train our distilled student model M from scratch on the mixture of D and D′(Kim and Rush, 2016).

Figure 8: Ensemble distillation and selfdistillation. We can compress our data efficiency gains into smaller models through distillation. Distilling an 8-ensemble teacher into a 300M student retains most of the loss improvement (pink star) and outperforms the regularized asymptote. Selfdistillation with a 300M teacher and 300M student
(green star) is surprisingly effective, matching the asymptote of the regularized recipe without increasing parameter count at training.

![7_image_0.png](7_image_0.png)

## 6.2 Reducing Train Parameter Count Via Self-Distillation

Is it possible to train a data-efficient model without high parameter count at train time as well? We consider this question for self-distillation where the teacher and student are of the same size and architecture. Many recent papers discuss how training a new student model on model generations can result in model collapse (Shumailov et al., 2024; Gerstgrasser et al., 2024; Dohmatob et al., 2024; Taori and Hashimoto, 2022).

On the contrary, by mixing together the D real tokens and D′synthetic tokens, we avoid collapse and can train a fresh student that vastly *outperforms* its teacher. In Figure 8, we show how using a 300M model as a teacher (blue point) results in a 300M student model (green star) that outperforms the best regularized 300M model (purple point). Why does self-distillation help?Allen-Zhu and Li (2023) provide theory interpreting self-distillation as implicitly ensembling the teacher and freshly initialized student.

## 7 Downstream Tasks

![7_Image_1.Png](7_Image_1.Png)

Figure 9: **Performance of pre-trained models on downstream tasks.** We have thus far been using validation loss (left) to seperate whether models are better pre-trained models or not. We evaluate the same models and ensembles on downstream benchmarks (right). Models with lower validation loss have lower average error across downstream benchmarks. Although validation loss is known to correlate with capabilities of interest (Chen et al., 2025; Thrush et al., 2025; Gadre et al., 2024), we further test our models' general capabilities using downstream benchmarks. For evaluations that are informative for models at our scale, we take all of the accuracybased benchmarks from Thrush et al. (2025), namely PIQA (Bisk et al., 2019), SciQ (Welbl et al., 2017), and ARC Easy (Clark et al., 2018). Notably, we did not evaluate on any benchmarks until the end of the project after we selected the best recipes following validation loss, making these benchmarks a strong test of generalization. In Figure 9, we show the validation loss (left) and downstream benchmark error (right) of our models for 200M tokens. Without regularization, the standard recipe does not benefit much from parameter scaling. Regularization (purple points) makes downstream accuracy scale smoothly with diminishing returns, similar to validation loss. Ensemble error mirrors loss with increasing N and K improving performance. Overall, our best ensemble outperforms our best unregularized model by over 9% on average and our best distilled model outperforms the unregularized 300M model by 7%. See Appendix G for a full breakdown of results.

## 8 Related Work

We cite additional related work on over-parametrized maching learning, distillation algorithms, synthetic data, and classical data-constrained deep learning in Appendix J. Scaling laws. Much of the success of language model pre-training was built upon scaling laws which accurately predict performance at a given resource budget (Hestness et al., 2017; 2019; Rosenfeld et al., 2019; Henighan et al., 2020; Kaplan et al., 2020; Sorscher et al., 2023; Hoffmann et al., 2022; Ruan et al., 2024; Cortes et al., 1993). Past work has studied scaling laws under constraints such as data and compute (Muennighoff et al., 2023; Goyal et al., 2024), hardware precision (Kumar et al., 2024), parameter count (Sardana et al., 2025; Springer et al., 2025; Gadre et al., 2024), and test-time compute (Brown et al., 2024; Snell et al., 2024). We show that past work (Muennighoff et al., 2023)
does not account for over-fitting, fix this via regularization, and propose asymptote estimation as a new metric. Ensembling. Ensembling (Dietterich, 2000) is known to boost performance across settings including uncertainty estimation (Lakshminarayanan et al., 2017), image classification (Huang et al., 2017; Garipov et al., 2018), and reinforcement learning (van Hasselt et al., 2015). Deep ensembles are shown to follow power laws (Lobacheva et al., 2021) and not believed to outperform parameter scaling in certain theoretical models (Vyas et al., 2023; Ruben et al., 2024). We show how ensembling can be adopted for pre-training and build scaling laws to characterize loss. See Appendix D.5 for discussion on related alternatives. Distillation. Distillation spends compute to produce strong models with lower inference costs (Hinton et al., 2015) which we show with sequence knowledge distillation (Kim and Rush, 2016). For self-distillation, there is recent work showing how training on self-generated inputs can be harmful (Shumailov et al., 2024; Dohmatob et al., 2024; Taori and Hashimoto, 2022). Gerstgrasser et al. (2024) suggests that training on self-generated data can be helpful in limited scenarios, though their comparisons are neither compute-matched nor data-matched. The success of self-distillation aligns with prior evidence from data-constrained deep learning (Mobahi et al., 2020; Zhang et al., 2019). Notably, Allen-Zhu and Li (2023) show how self-distillation can be viewed as implicitly performing ensembling and distillation. Modern data-constrained pre-training. There are several recent works which study data-efficient pre-training and show the benefit of epoching (Muennighoff et al., 2023), rephrased synthetic data (Maini et al., 2024; Yang et al., 2024; DatologyAI et al., 2025; Ruan et al., 2024), diffusion language models (Prabhudesai et al., 2025; Ni et al., 2025), and energy-based models (Gladstone et al., 2025). These recent works do not aggressively regularize for optimal epoching nor build scaling laws to estimate infinite compute performance.

## 9 Discussion

The success of classical ideas from data-constrained deep learning like regularization and ensembling suggests that there is free lunch on the table, encouraging us to revisit pre-training design decisions. We are also excited by methods that can better leverage extra compute for performance, in line with The Bitter Lesson (Sutton, 2019). We hope that evaluating scaling recipes via their asymptotes inspires more data-efficient algorithms for the future.

## 10 Acknowledgements

We thank Steven Cao, Sam Park, Jacob Mitchell Springer, Kaiyue Wen, Yu Sun, Nathan Hu, Meena Jagadeesan, Luke Bailey, Neil Band, Sally Zhu, Ben Spector, and Audrey Xie for their helpful discussions or feedback on the paper draft. This work is a part of the Marin Project and the compute is supported by the Google TPU Research Cloud (TRC). TH was supported by a grant by HAI, DSO labs, gifts from Open Philanthropy, Amazon, Schmidt Sciences, the Tianqiao and Chrissy Chen Foundation and a grant under the NSF CAREER IIS-2338866, ONR N00014-24-1-2609, and DARPA Cooperative Agreement HR00112520013. PL was supported by DARPA Cooperative Agreement HR00112520013. This work does not necessarily reflect the position or policy of the government and no official endorsement should be inferred.

## 11 Ethics

We hope that our work may be applied to settings beyond pre-training to improve data efficiency. We acknowledge our work may increase the amount of compute used for language model pre-training. We believe most other harms specific to our work apply to general language modeling research.

## 12 Reproducibility

We open-source all of our runs on WandB and our code on Github.

## References

M. Advani and S. Ganguli. Statistical mechanics of optimal convex inference in high dimensions.

Phys. Rev. X, 6:031034, Aug 2016. doi: 10.1103/PhysRevX.6.031034. URL https://link. aps.org/doi/10.1103/PhysRevX.6.031034.

R. Agarwal, N. Vieillard, Y. Zhou, P. Stanczyk, S. Ramos, M. Geist, and O. Bachem. On-policy distillation of language models: Learning from self-generated mistakes, 2024. URL https:
//arxiv.org/abs/2306.13649.

S. K. Ainsworth, J. Hayase, and S. Srinivasa. Git re-basin: Merging models modulo permutation symmetries, 2023. URL https://arxiv.org/abs/2209.04836.

Z. Allen-Zhu and Y. Li. Towards understanding ensemble, knowledge distillation and self-distillation in deep learning, 2023. URL https://arxiv.org/abs/2012.09816.

Z. Allen-Zhu and Y. Li. Physics of language models: Part 3.1, knowledge storage and extraction, 2024. URL https://arxiv.org/abs/2309.14316.

A. Amini, S. Gabriel, P. Lin, R. Koncel-Kedziorski, Y. Choi, and H. Hajishirzi. Mathqa: Towards interpretable math word problem solving with operation-based formalisms, 2019. URL https:
//arxiv.org/abs/1905.13319.

M. Belkin, D. Hsu, S. Ma, and S. Mandal. Reconciling modern machine-learning practice and the classical bias–variance trade-off. *Proceedings of the National Academy of Sciences*, 116 (32):15849–15854, July 2019. ISSN 1091-6490. doi: 10.1073/pnas.1903070116. URL http:
//dx.doi.org/10.1073/pnas.1903070116.

T. Besiroglu, E. Erdil, M. Barnett, and J. You. Chinchilla scaling: A replication attempt, 2024. URL
https://arxiv.org/abs/2404.10102.

Y. Bisk, R. Zellers, R. L. Bras, J. Gao, and Y. Choi. Piqa: Reasoning about physical commonsense in natural language, 2019. URL https://arxiv.org/abs/1911.11641.

B. Brown, J. Juravsky, R. Ehrlich, R. Clark, Q. V. Le, C. Ré, and A. Mirhoseini. Large language monkeys: Scaling inference compute with repeated sampling, 2024. URL https://arxiv.

org/abs/2407.21787.

T. B. Brown, B. Mann, N. Ryder, M. Subbiah, J. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell, S. Agarwal, A. Herbert-Voss, G. Krueger, T. Henighan, R. Child, A. Ramesh, D. M. Ziegler, J. Wu, C. Winter, C. Hesse, M. Chen, E. Sigler, M. Litwin, S. Gray, B. Chess, J. Clark, C. Berner, S. McCandlish, A. Radford, I. Sutskever, and D. Amodei. Language models are few-shot learners, 2020. URL https://arxiv.org/abs/2005.14165.

D. Busbridge, A. Shidani, F. Weers, J. Ramapuram, E. Littwin, and R. Webb. Distillation scaling laws, 2025. URL https://arxiv.org/abs/2502.08606.

A. Canatar, B. Bordelon, and C. Pehlevan. Spectral bias and task-model alignment explain generalization in kernel regression and infinitely wide neural networks. Nature Communications, 12(1), May 2021. ISSN 2041-1723. doi: 10.1038/s41467-021-23103-1. URL http://dx.doi.org/10.1038/s41467-021-23103-1.

Y. Chen, B. Huang, Y. Gao, Z. Wang, J. Yang, and H. Ji. Scaling laws for predicting downstream performance in llms, 2025. URL https://arxiv.org/abs/2410.08527.

P. Clark, I. Cowhey, O. Etzioni, T. Khot, A. Sabharwal, C. Schoenick, and O. Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge, 2018. URL https:
//arxiv.org/abs/1803.05457.

K. Cobbe, V. Kosaraju, M. Bavarian, M. Chen, H. Jun, L. Kaiser, M. Plappert, J. Tworek, J. Hilton, R. Nakano, C. Hesse, and J. Schulman. Training verifiers to solve math word problems, 2021. URL https://arxiv.org/abs/2110.14168.

C. Cortes, L. D. Jackel, S. Solla, V. Vapnik, and J. Denker. Learning curves: Asymptotic values and rate of convergence. In J. Cowan, G. Tesauro, and J. Alspector, editors, *Advances in Neural Information Processing Systems*, volume 6. Morgan-Kaufmann, 1993. URL https://proceedings.neurips.cc/paper_files/paper/1993/
file/1aa48fc4880bb0c9b8a3bf979d3b917e-Paper.pdf.

F. D'Angelo, M. Andriushchenko, A. Varre, and N. Flammarion. Why do we need weight decay in modern deep learning?, 2024. URL https://arxiv.org/abs/2310.04415.

DatologyAI, :, P. Maini, V. Dorna, P. Doshi, A. Carranza, F. Pan, J. Urbanek, P. Burstein, A. Fang, A. Deng, A. Abbas, B. Larsen, C. Blakeney, C. Bannur, C. Baek, D. Teh, D. Schwab, H. Mongstad, H. Yin, J. Wills, K. Mentzer, L. Merrick, R. Monti, R. Adiga, S. Joshi, S. Das, Z. Wang, B. Gaza, A. Morcos, and M. Leavitt. Beyondweb: Lessons from scaling synthetic data for trillion-scale pretraining, 2025. URL https://arxiv.org/abs/2508.10975.

J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei. Imagenet: A large-scale hierarchical image database. In *2009 IEEE Conference on Computer Vision and Pattern Recognition*, pages 248–255, 2009. doi: 10.1109/CVPR.2009.5206848.

T. G. Dietterich. Ensemble methods in machine learning. In Proceedings of the First International Workshop on Multiple Classifier Systems, MCS '00, page 1–15, Berlin, Heidelberg, 2000. Springer-
Verlag. ISBN 3540677046.

E. Dohmatob, Y. Feng, A. Subramonian, and J. Kempe. Strong model collapse, 2024. URL
https://arxiv.org/abs/2410.04840.

K. Everett, L. Xiao, M. Wortsman, A. A. Alemi, R. Novak, P. J. Liu, I. Gur, J. Sohl-Dickstein, L. P.

Kaelbling, J. Lee, and J. Pennington. Scaling exponents across parameterizations and optimizers, 2024. URL https://arxiv.org/abs/2407.05872.

S. Y. Gadre, G. Smyrnis, V. Shankar, S. Gururangan, M. Wortsman, R. Shao, J. Mercat, A. Fang, J. Li, S. Keh, R. Xin, M. Nezhurina, I. Vasiljevic, J. Jitsev, L. Soldaini, A. G. Dimakis, G. Ilharco, P. W. Koh, S. Song, T. Kollar, Y. Carmon, A. Dave, R. Heckel, N. Muennighoff, and L. Schmidt.

Language models scale reliably with over-training and on downstream tasks, 2024. URL https:
//arxiv.org/abs/2403.08540.

Y. Gal and Z. Ghahramani. A theoretically grounded application of dropout in recurrent neural networks, 2016. URL https://arxiv.org/abs/1512.05287.

L. Gao, J. Tow, B. Abbasi, S. Biderman, S. Black, A. DiPofi, C. Foster, L. Golding, J. Hsu, A. Le Noac'h, H. Li, K. McDonell, N. Muennighoff, C. Ociepa, J. Phang, L. Reynolds, H. Schoelkopf, A. Skowron, L. Sutawika, E. Tang, A. Thite, B. Wang, K. Wang, and A. Zou.

The language model evaluation harness, 07 2024. URL https://zenodo.org/records/ 12608602.

T. Garipov, P. Izmailov, D. Podoprikhin, D. Vetrov, and A. G. Wilson. Loss surfaces, mode connectivity, and fast ensembling of dnns, 2018. URL https://arxiv.org/abs/1802.10026.

M. Gerstgrasser, R. Schaeffer, A. Dey, R. Rafailov, H. Sleight, J. Hughes, T. Korbak, R. Agrawal, D. Pai, A. Gromov, D. A. Roberts, D. Yang, D. L. Donoho, and S. Koyejo. Is model collapse inevitable? breaking the curse of recursion by accumulating real and synthetic data, 2024. URL https://arxiv.org/abs/2404.01413.

A. Gladstone, G. Nanduru, M. M. Islam, P. Han, H. Ha, A. Chadha, Y. Du, H. Ji, J. Li, and T. Iqbal. Energy-based transformers are scalable learners and thinkers, 2025. URL https:
//arxiv.org/abs/2507.02092.

S. Goyal, P. Maini, Z. C. Lipton, A. Raghunathan, and J. Z. Kolter. Scaling laws for data filtering - data curation cannot be compute agnostic, 2024. URL https://arxiv.org/abs/2404.07177.

S. Goyal, D. Lopez-Paz, and K. Ahuja. Distilled pretraining: A modern lens of data, in-context learning and test-time scaling, 2025. URL https://arxiv.org/abs/2509.01649.