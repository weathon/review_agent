# Beyond Interpretability: The Gains of Feature Monosemanticity on Model Robustness

- Decision: Accept (Poster)
- Scores: 6, 5, 6, 6, 6

## Abstract
Deep learning models often suffer from a lack of interpretability due to \emph{polysemanticity}, where individual neurons are activated by multiple unrelated semantics, resulting in unclear attributions of model behavior. Recent advances in \emph{monosemanticity}, where neurons correspond to consistent and distinct semantics, have significantly improved interpretability but are commonly believed to compromise accuracy. In this work, we challenge the prevailing belief of the accuracy-interpretability tradeoff, showing that monosemantic features not only enhance interpretability but also bring concrete gains in model performance of {\color{black} robustness-related tasks}. Across multiple robust learning scenarios—including input and label noise, few-shot learning, and out-of-domain generalization—our results show that models leveraging monosemantic features significantly outperform those relying on polysemantic features. Furthermore, we provide empirical and theoretical understandings on the robustness gains of feature monosemanticity. Our preliminary analysis suggests that monosemanticity, by promoting better separation of feature representations, leads to more robust decision boundaries {\color{black} under noise}. This diverse evidence highlights the \textbf{generality} of monosemanticity in improving model robustness. As a first step in this new direction, we embark on exploring the learning benefits of monosemanticity beyond interpretability, supporting the long-standing hypothesis of linking interpretability and robustness. Code is available at \url{https://github.com/PKU-ML/Monosemanticity-Robustness}.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
In this work the authors show that inducing models to learn more monosemantic features (as opposed to neurons that learn multiple concepts, also known as polysemantic features) not only improves the interpretability of the model decisions, but it can also lead to more robust models. 

The authors show results both in the vision and language domain. For the vision experiments they adopt two different techniques in order to induce monosemanticity: Non-Negative Contrastive Learning  (NCL, Wang et al 2024) which is a simple variation of SimCLR and it is supposed to create monosemantic representations, and Sparse Auto Encoders (SAE, Gao et al 2024) as a post-training intervention that produces embedding with higher monosemanticity. The authors use ReseNet-18 trained in a self-supervised fashion on CIFAR-100 and Imagenet-100. The authors show that when using either NCL to train the model or SAE (on the embedding of a pre trained SimCLR model) the accuracy of both models are comparable to that of the original model (that is usually polysemantic). However, when the dataset includes higher level of label noise, distribution shift (gaussian or uniform noise, real-data distribution shift), or the dataset is much smaller, the model that rely on monosemantic features remain more robust than the original model.

The authors also show some empirical analysis where they find that when training with a 90% of label noise, the dimension activated by the class with lower accuracy is polysemantic (i.e., is also activated by other classes), whereas the class with the higher dimension is monosemantic. Given this they conclude that noisy classifiers prefer monosemantic feature in practice.

For the language model the authors propose to modify the known Low Rank Adaptation mechanism (LoRA) by adding non-negative transformations (ReLU) that force the output of the LoRA layers to be more sparse (which is assumed to also induce monosemanticity). The authors fine-tune a Llama2 model on two different datasets (SST2 and Dolly) using the traditional LoRA and their proposed MonoLoRA, and show that while both models achieve comparable downstream task performance, MonoLoRA better preserves the model alignment (measured by the ShieldGemma and Beavertail alignment scores). In the case of the SST2 dataset the alignment becomes even better than that of the original model.

Lastly, the authors present an analysis using the super-position model (Elhage 2022b) that validates the results found on the vision models: in absence of noise models with polysemantic features show better performance, however, in the presence of noise the models with monosemantic features become more accurate than their polysemantic counterpart. Using a toy task (binary classification with a simple 2 dimensional embedding) the authors also show that when the label noise rate is grater than 25% the model with mono semantic features show a better linear separability between the two classes, again validating the empirical results found in the vision experiments on CIFAR-100 and ImageNet-100.

### Strengths
The paper investigate an aspect of monosemanticity that I am not aware that has been investigated before. 

Among all the experiments presented I think the real-data distribution shift and the low percentage of training data represent important and realistic scenarios where monosemanticity could play an interesting role.

The paper is easy to read and I particularly liked the introduction where all the main findings are nicely summarized.

### Weaknesses
The main weakness I see in this work is related to the experimental setting. The methods used to induced monosemanticity are already known (with the exception of MonoLora) so I would have expected a more thorough evaluation. In particular these are the main points I would like to discuss with the authors: potentially unrealistic scenarios (e.g., label noise >= 25%),  some concerns with the quality of the empirical evaluation (e.g., baselines not matching state of the art results, lack of quantification of monosemanticity, lack of error bars, lack of evaluation of LLM capabilities). Addressing these points could improve the soundness and the contribution of this work. Let me elaborate more in detail each of those points.

- Potentially unrealistic scenarios. The experiments on labels noise explore a range of noise that goes from 0 to 90% which seems a very large and unrealistic amount of noise. I would consider exploring more in details (and with multiple runs) what happens in a more realistic range. The work from MIT “Pervasive Label Errors in Test Sets Destabilize Machine Learning Benchmarks” could be a useful guidance to know what is a realistic range. From that paper it seems the full ImageNet has about 3.4% of label errors, while CIFAR-100 has about 5.85% (from their table 1). The worst case is QuickDraw with 10.12% error rate. The average error rate from that table is 3.32%. Given those results, perhaps focusing on the range 0-10 would be more realistic and informative than 0-90.  I believe this paper would be very interesting (if results holds) if more space was dedicated to real-data distribution shifts (all imagenet test bed as mentioned below), training with low amount of data, and fine-tuning LLMs. All the experiments with label noise and other kind of noise would be interesting only if results showed that monosemanticity can help in realistic scenarios (e.g., label noise between 0% to 10%).

- Baselines not matching state of the art results. Even with a small scale model such as ResNet-18 (as opposed to state of the art work on self-supervised learning that use ResNet-50 - see one of my “minor” comments below about scaling to larger models) the results previously published by Wang et al. 2024 (table 3) show accuracies that are higher than the ones reported in this work for CIFAR-100 and ImageNet-100 (both for the original model and the NLC ones). Specifically: Wang et al. 2024 reports for CIFAR-100
Contrastive learning (with linear probing) 58.6% vs current work 54.5%
Non Contrastive learning (with linear probing) 59.7% vs current work 52.8%
Similar discrepancies can be found for ImageNet-100 with linear probing, and with both CIfar-100 and ImageNet-100 with full fine tuning using 100% of the datasets.
My concerns with these non-matching results are that on the one hand Wang et al. had already shown that NCL is better than traditional contrastive learning even in the original setting, and additionally, it undermine the credibility of the results shown. I would be great to comment on these discrepancies.

- Lack of quantification of “monosemanticity”. The paper rely on models with monosemantic features however the level of monosemanticity is never measured and compared with counterpart models. It would be great to quantitively show that indeed the model that are being produced do contain more monosemantic features. While NCL and SEA were already published and they might have shown their ability to increase monosemanticity, it would still be good to double check this effect. This is even more important with the proposed MonoLoRA since I suppose that no one has ever checked if in this case sparsity also leads to monosemanticity. Additionally an analysis of monosemanticity could help understanding more deeply the differences between NCL and SEA beyond accuracy (which of the two actually leads to more monosemantic representations?).

- None of the experiments show any error bar. It is always good to have multiple runs

- I found the explanation of the LLM experiments lacking necessary details in order to be understood properly. Without reading Appendix A3 (and other still missing details) this section is not self-explanatory. I would encourage rewriting this section with more details. For example, talk about SST2 and Dolly datasets: what kind of tasks are they for? On which metrics are they evaluated? What are the ShieldGemma-9B and Beaverstails-7B alignment scores? 

- Additionally, related to the LLM evaluation, the authors mention overfitting as one of the issue with fine-tuning. If overfitting is the concern, I think that it would be good to show that the model has not lost some of its original capabilities (if they are still of interest). Usually this is shown by empirically measuring the change in perplexity, the MMLU benchmark and the common sense reasoning tasks. The current experiments only show the change in “alignment” (which should be better defined) but not other changes that could be affected by overfitting. 


What follows are some suggestions on how to improve the writing, the clarity, and overall the quality of the manuscripts. While there are some questions in there they are not necessarily meant to be answered in the rebuttal but rather to be answered in a new version of the paper. None of the comments below are of major concern, however, addressing them would make this work stronger. Please focus the rebuttal on the questions related to the main concerns.

Writing:
- Since Wang et al. had already shown that there is no trade off to be paid (in their paper NCL works better than standard SimCLR) I would put the emphasis on showing that monosemanticity could be useful also in terms of robustness. Talking about the trade off seems in contradiction with Wang et al.’s results. 
- Some statements in the abstract should be revised to avoid creating false expectations. For example the authors state “Our preliminary analysis suggests that monosemanticity, by promoting better separation of feature representations, leads to more robust decision boundaries.” This statement seems to be always “true” but from the results presented it seems true only in specific scenarios (high level of noise etc…).
- “More details can be found in Appendix A3“. When referring to the appendix consider offering more info about which details can be found in the appendix so the reader can decide if they want to read it or not. Are there more results? More background on the experiments? More info on the hyperparameters etc…
- Figure 4. The pictures in this figure refer to the model with induced monosemanticity or the original model? It could be interesting to compare this result with the “other” model.
- Figure 4. What does it mean “top activated” samples? What is the value for the highest accuracy group and the lowest accuracy group? Is the issue mono vs poly semanticity of could the issue simply be that the classes that perform poorly have not been properly learnt (that could be indicated by a low magnitude activation?).
- I would invite the authors to think how much the super-position section (both empirical and “theoretical”) add to the paper. I don’t find it particularly illuminating because it is still based on a very specific set of choices (hidden dimension size, task, etc…). Maybe this space could be used to explain more in details the LLM settings, or present experiments on larger scale, etc.. and this section could be moved to the appendix.
- In the introduction “Pretrained LLMs need to be carefully finetuned on small-scale language data for different purposes, e.g., instruction following and certain abilities (e.g., reasoning), while avoiding conflicting and forgetting.” What does “conflicting” means in this context? This should be clarified.
- All the experiments in 3.2.1 could be obtained on the NLC trained backbone or on the original SimCLR backbone. It is unclear what backbone was used both for the standard fine-tuning and the NNT fine-tuning. There are 4 possible combinations (standard SimCLR backbone with standard fine-tuning, standard SimCLR backbone with NNT, NCL backbone with standard fine-tuning and NCL backbone with NNT fine-tuning) it would be great to explicitly say which are being evaluated.
- Writing: I am assuming that the experiments in 3.2 are different than 3.1 because in 3.2 the full backbone model is fine-tuned while in 3.1 the backbone is frozen. However, this should be mentioned more precisely. Currently the paper says “Unlike linear probing we train classifiers on the pretrained representations without clipping the gradient of encoders”. However, clipping the gradient does not necessarily mean freezing the encoder. I would clarify this aspect.
- Suggestion related to Figure 2:  I’d suggest to include (maybe in the appendix) an example of a figure perturbed at the different levels used in the experiments so that the reader can appreciate better if that level of noise is realistic and in which scenario this degradation could happen.
- Page 5 mentions CIFAR-10 but according to the previous explanation I believe this should be CIFAR-100 (Typo)?
- The feature map analyzed in the case of the vision experiments are those before the logits? This detail was never explicitly mentioned (or I might have missed it).
- Throughout the paper there are some “concepts” that are never formally defined. For example, the authors often talk about a neuron being activated by single or multi set of concepts. What does it mean to be activated?

Other aspects (not critical but would improve the work):
- Another potential mismatch is with the SAE definition. In equation (2) the authors define the output for the encoder as  $z(x) = topK(W_{enc}(f(x)-b_{pre}) +b_{enc})$ and the output of the AE as $\hat{f}(x) = W_{dec}z(z) + b_{pre}$. However, Gao et al. uses a slightly different definition of the topK function which does not include the term $b_{enc}$, specifically, $z(x) = topK(W_{enc}(f(x)-b_{pre}))$. Is this a typo, are the two formulations equivalent, or is this an acltual difference compared to Gao et al. (in which case it would be great to motivate the change)?
- I believe the experiments with real-world distribution shift are very interesting and relevant. However the authors only reports results on 2 shifts (sketch and stylized). On the one hand the justification about why only these two out of the existing ones in the cited literature would be necessary. On the other hand it would be even better to include all of them and leverage advances in this field and extend the results to all imaginet-testbed shifts from Taori et al 2020 (“Measuring Robustness to Natural Distribution Shifts in Image Classification”). If authors could show a significant gain in robustness with respect to these distribution shifts the paper would be much stronger since some of these shifts are more realistic in my opinion than a 90% noise in the labels.
- Small scale experiments. It is often the case that when moving from a small scale setting (small model and small dataset) to a large scale many of the advantages observed are unfortunately lost. The authors use ResNet-18 with CIFAR-100 and ImageNet-100 but state of the art in self supervised learning is usually working with ResNet-50 and full ImageNet. I am not sure if the same results will transfer also to larger scale settings. For example, notice how in Figure 3(b) the gap between the accuracy of the polysemantic and the monosemantic models is reducing as you go towards 100% of training data. Had the authors used ResNet50 and full imageNet would you still observe such a difference? Perhaps this indicates that monosemantic features are very important in small scale settings but not in large ones?  This experiment is not crucial but without it I would assume that monosemantic is only useful in a low scale regime. If the authors want to extend the influence of monosemanticity on large scale setting this should be shown.
- I am not sure I would consider the “theoretical understanding” provided in section 4.3 actually theoretical. It seems to me that is a toy model with concrete values for different choices that can be analyzed manually. It has value but I would consider this still an empirical analysis. 
- From the appendix the authors mentioned that the number of activated features for SAE was 256. How was this choice made and how robust are these models to the choice of this parameter?
- Why topK and not JumpRelu (Lieberum et al.)? It would be good to motivate this choice.
- In general the appendix contain most of the hyper-parameters used but I could not find the value of the Temperatures.

### Questions
On the unrealistic scenarios, could the authors:
1. Provide detailed results for the 0-10% noise range, with multiple runs and error bars.
2. Discuss whether the benefits of monosemanticity are still observed in this more realistic regime.
3. Consider emphasizing and expanding the experiments on real-world distribution shift experiments, as these may be more practically relevant.

On the discrepancies with Wang et al. 2024 results for CIFAR-100 and ImageNet-100, could the authors:
1. Provide a detailed explanation for the discrepancies.
2. Verify their implementation of NCL to ensure it matches the original method.
3. If possible, reproduce the exact settings from Wang et al. to confirm whether the discrepancy persists.

On the quantification of “monosemanticity”, could the authors:
1. Measure and report a metric of monosemanticity (e.g., semantic consistency as used in Wang et al.) for all models and methods, including MonoLoRA.
2. Provide a comparative analysis of monosemanticity levels between NCL, SAE, and standard models.
3. Investigate the relationship between sparsity and monosemanticity in MonoLoRA.

On the lack of error bars. It is important to repeat the experiments (especially when stochastic processes such sampling are involved) and provide mean and standard deviation. Are the results shown statistically significant?

On the lack of evaluation of LLMs: would it make sense to show that the LLM abilities are better preserved with monosemanticity by evaluating the MMLU benchmark?

Further questions
- In the abstract the authors called these analysis “Preliminary” could you clarify which of the results you consider preliminary and what you would need to consider them definitive? 
- The end of the intro reads “Theoretically, as a preliminary step, we compare polysemantic and monosemantic features under a toy model proposed in Elhage et al. (2022b). The theory suggests that because monosemantic features have better separation of features, they are less prone to overfitting to noise, leading to more robust decision boundaries compared to polysemantic features.”. Is this what the theoretical analysis concluded? If so, shouldn’t the benefit be evident also in noise-free scenarios? I believe that what it is shown in section 4.3 matches this statement only when in presence of a large amount of noise (>=25%). Or have I misunderstood what the authors mean?

### Soundness
3

### Presentation
2

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
The work tackles the problem of polysemantic neurone, i.e., when in the last layer of a neural network individual neurones are activated by multiple concepts. The authors state to break the accuracy-interpretability tradeoff and show that monosemantic neurones can provide more robust models as well as gain of accuracy. The authors also present theoretical calculations connected to monosemantic and polysemantic features.

### Strengths
The observations made in the manuscript regarding the monosemanticity seem original.
Quality of the research cannot be disputed.

### Weaknesses
1. Reading and understanding the manuscript relatively well requires a specific set/baggage of knowledge to have been acquired, because the manuscript is full of branch-specific references and acronyms. This makes the manuscript difficultly accessible for general public.

Perhaps as a possible improvement, the author(s) could prepare more intuitive writing for the body of the manuscript but shift (more difficult to understand) acronymed details to appendix.

2. I do not find a clear methodological contribution, rather an empirical finding.

3. What authors call Theorems 4.1 and 4.2 are (mathematically speaking) examples, not theorems.

### Questions
I have counted 4 data sets (for different tasks), is it really enough to make any general statement? If yes, the authors should provide an argument why? E.g., can the performance differences from Table 1 be called statistically significant? If yes, with which p-value(s)?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The  authors  propose  that  monosemanticity,  where  neurons  represent  distinct  and  consistent  semantics,  can  improve  interpretability  without  compromising  accuracy.  The  authors  demonstrate  that  models  using  monosemantic  features  perform  better  in  three  robust  learning  scenarios.  The  study  provides  empirical  and  theoretical  evidence  that  monosemanticity  leads  to  more  robust  decision  boundaries,  suggesting  a link  between  interpretability  and  robustness.

### Strengths
1. The paper offers a valuable and insightful perspective on the benefits of feature monosemanticity.

2. The study presents a robust and diverse set of evidence and experiments from various learning scenarios.

3. The paper is well-organized, with a clear and logical structure that makes it easy to follow.

### Weaknesses
1. The paper utilizes existing sparse and non-negative constraints in three tasks, which are well-researched areas. While the application of these constraints to monosemanticity is interesting, the overall novelty is somewhat limited.

2. The paper does not provide a thorough explanation for why non-negative constraints are generalized across all three tasks. It is unclear whether the same form of non-negative constraints is suitable for all tasks or if specific forms are needed for different scenarios. More detailed discussion and additional experiments would help to clarify this point and strengthen the argument.

3. Abbreviations such as NCL and SAE, introduced in line 69, should be defined at their first occurrence. It well be helpful for readers to follow.

### Questions
See weakness.

### Soundness
2

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper challenges the "accuracy-interpretability" tradeoff in deep learning models by considering the robustness of features across a range of approaches to learning. Experimental results validate that learned monosemantic features are more robust to noisy data and produces better performance when finetuning. The paper extends a toy model from Elhage et al. to suggest an explanation for this phenomenon.

### Strengths
1. The application of robustness as a performance metric to feature semanticity is a useful contribution.
2. The choice of baseline and methods to produce monosemanticity are well chosen.
3. The toy model provides a clear demonstration of why monosemantic features fare better.

### Weaknesses
1. The paper spends little time contextualising the "widely accepted belief" that there is an accuracy-interpretability tradeoff.
2. It isn't immediately clear why MonoLoRA promotes feature monosemanticity. A clearer explanation of the mathematics would be helpful.
3. The theoretical explanation is limited to label noise only, rather than finetuning scenarios. It would be good to address it at least in passing, or as a possible avenue of further work.

### Questions
1. What is the principled reason for distinguishing between few shot and LLM finetuning?
2. Should we expect a difference in outcomes between adding Gaussian noise and adding uniform noise?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper aims to investigate the role of monosemanticity of model features on model robustness. Monosemanticity of model features is characterised by individual dimensions of the model embedding layer being activated with single latent natural concepts (vs. several, which corresponds to polysemanticity of model features). Robustness is investigated empirically in linear probing experiments on CIFAR-100 and ImageNet-100 with respect to label noise, gaussian input noise, uniform input noise, and sketch and stylised distribution shifts. Furthermore, it is investigated empirically in few-shot finetuning on ImageNet-100 as well as on LoRA training for LLMs. 
To analyse the effect of monosemanticity on robustness, models trained to exhibit monosemantic features (e.g. through sparse auto encoders) are compared to vanilla trained models (representing polysemantic feature models). In all comparisons where robustness is investigated as described, the models trained to exhibit monosemantic features perform better.
Lastly, the paper illustrates benefits of monosemanticity for model robustness on toy examples as well.

### Strengths
- The paper is investigating an interesting and relevant research question: is monosemanticity beneficial beyond interpretability, and for model robustness in particular?
- The paper is overall well written
- The paper has a good range of experiment beds (but individual experiments should be enriched, see below)

### Weaknesses
- Main weakness: At no point in Section 3 does the paper measure how ‘monosemantic’ the trained models are (according to their definition in line 114). So any claim of ‘monosemanticity -> robustness’ cannot be derived from the experiments in Section 3, as we do not know how monosemantic the models really are. Is there actually a quantifiable and significant difference in the monosemanticity of the models in Table 1, Figure 2, or Figure 3, or Table 2 (here, sparsity is given as a proxy for monosemanticity, but how close of a proxy it is is unclear)
- The definition of the polysemantic feature $v = x_1 - x_2$ in the theoretical toy example is specific rather than general, and insights derived from it will be accordingly so. For a more general validity of the results derived here, it should be of the form $w_1 x_1 + w_2 x_2 + b$ I believe. 
- Figure 3a and 3b: it is possible that the cause for lower training and better validation accuracy is simply the regularisation and not the resulting monosemanticity. To understand this better, including a different regularisation that does not promote semanticity would be beneficial (e.g. early stopping or L2 regularisation, or even something like a negative-cross entropy). In any case monosemanticity of final models needs to be evaluated to draw conclusions here (see first weakness)
- Even if the other weaknesses were addressed, the paper would slightly overclaim, in particular in saying ‘feature monosematnicity to bring clear gains in model accuracy’ (l 100 and similar at other parts). As can be seen in the ‘0’ column of Table 1, monosemanticity is still at odds with model accuracy on clean iid accuracy (even though only slightly). Authors should specify accordingly that monosematnicity can bring gains in model robustness, not in general accuracy (as in some cases as the above it does not). 
- For the vision model results, only a single backbone (ResNet18) is used in experiments. To give the claims and findings greater generality, including results of the same experiment suite but with a ViT backbone would be beneficial. 

- Minor: typo l 242: Cifar-10 should be Cifar-100; typo l 257: ‘if there IS only a SMALL amount of …’, 
- Minor: align references in l 120 and l 45

### Questions
-

### Soundness
3

### Presentation
4

### Contribution
4
