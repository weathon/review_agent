# Flow-Guided Neural Operator For Self- Supervised Learning On Time Series Data

- Decision: Reject
- Scores: 2, 4, 4, 2

## Abstract
Self-supervised learning (SSL) is a powerful paradigm for learning from unlabeled time-series data. However, popular methods such as masked autoencoders (MAEs) rely on reconstructing inputs from a fixed, predetermined masking ratio. 
Instead of this static design, we propose treating the corruption level as a new degree of freedom for representation learning, enhancing flexibility and performance.
To achieve this, we introduce the Flow-Guided Neural Operator (FGNO), a novel framework combining operator learning with flow matching for SSL training.
By leveraging Short-Time Fourier Transform (STFT) to enable computation under different time resolutions, our approach effectively learns mappings in functional spaces.
We extract a rich hierarchy of features by tapping into different network layers ($l$) and flow times ($s$) that apply varying strengths of noise to the input data. 
This enables the extraction of versatile representations, from low-level patterns to high-level semantics, using a single model adaptable to specific tasks.
Unlike prior generative SSL methods that use noisy inputs during inference, we propose using clean inputs for representation extraction while learning representations with noise; this eliminates randomness and boosts accuracy.
We evaluate FGNO across three biomedical domains, where it consistently outperforms established baselines. Our method yields up to 35\% AUROC gains in neural signal decoding (BrainTreeBank), 16\% RMSE reductions in skin temperature prediction (DREAMT), and over 20\% improvement in accuracy and macro-F1 on SleepEDF under low-data regimes. These results highlight FGNO’s robustness to data scarcity and its superior capacity to learn expressive representations for diverse time-series applications.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces the Flow Guided Neural Operator (FGNO) as a framework for SSL training on 1-dimensional time series.
The paper's goal is to introduce a novel model architecture and measure its performance on downstream classification tasks.

The architecture is simple and elegant: the 1D signal is first vectorized through a short-time Fourier transform and then encoded through a transformer architecture, conditioned sinusoidally with a noise parameter, tailored for each task.

The architecture learns in a self-supervised manner according to a flow-matching objective.
This contrasts with the other existing types of approaches for time-series, based on masked reconstruction (termed masked auto-encoder), next-step predictions, and quantizer approaches (both are referenced through the Chronos paper but not described in the paper). The learning objective and exploration seem interesting and additionally offer a novel inference scheme.

Indeed, an important contribution of this architecture is to allow fine-tuning of the model behavior on independent tasks by varying two hyperparameters: the noise (s) and the layer (l), at which the representations are extracted for classifications.
In opposite, the previous approach could only select a certain layer (l) of representations.

The paper is well structured, original, and clear.

### Strengths
Originality:
The paper combines previously known methods into a novel architecture. Notably, they take inspiration from generative SSL methods and adapt them by using clean inputs at test time, solely conditioning on the noise value rather than sampling from noise.

Quality:
The paper explores four different datasets for evaluations.

Clarity:
The architecture is well described. 

Significance:
The author makes a significant improvement compared to one other evaluated model (Chronos).

### Weaknesses
Major methodological issues:
- It is unclear if the (s,t) parameters are selected using the test data, which would be very problematic, or if the author decomposed the datasets into (train, validation, test) folds. Exact folds and test datasets processing are missing from the methods and appendix.
- It is unclear which data the author used for training the models: similar to test datasets, these dataset folds and preprocessing should be described more extensively.
- Little information is provided on comparative baselines: masked-auto-encoder and Chronos architecture. It is not clear if the authors trained these baselines on the same data as their FGNO model.
- The last section of the paper seems to present conflicting evidence in the figure compared to the text. In the figure, the performance of the FGNO model decreases more than the performance of the Chronos model relative to their baseline value (at a downsampling factor of 1). This does not support the claim "FGNO shows remarkable stability across resolutions".

Moderate methodological issues:
- Two separate claims are made on two non-overlapping sets of datasets.
In principle, it would reinforce the claim to also evaluate how FGNO performs on BrainTree Bank and DREAMT with a change in % of training data; otherwise, we don't know whether the authors decided to use other datasets for this evaluation because the models did better on these datasets in this case. Alternatively, the author should stress why a change in % of training data is better evaluated in these two other datasets.

- Compared to previous papers, the paper is relatively sparse in evaluations.  For example, in the BrainBert paper (Wang et al 2023), which the author extensively cites, the model is evaluated on: Sentence onset, Speech/Nonspeech, Pitch Volume, and Task Avg classification scores. Since this paper touches on a very similar approach, there is no a priori reason to exclude these evaluations.

- Simple baselines. The paper uses two complex baselines: MAE and Chronos. But the quality of these baselines is hard to evaluate without a clear reference to a simpler baseline. For example, previous papers have used Linear Baseline on the STFT features, or simple deep NN on the STFT features (Wang et al 2023).  

Minor methodological issues:
- The training details are too short and not very informative.
- No analysis of the model's internal representations is done, such that it remains unclear what the model has exactly learned. Previous papers, for example, in the case of SSL on speech, Baevski et al 2021 have done so. While this could be the scope of future work, the author should touch on whether their architecture allows for similar exploration compared to previous architectures.
- No loss plots of the model training are provided. Although not mandatory, this would help in understanding and replicating the training of the model.

Minor writing issues:
- Line 362: Fig 2 should be Fig 3
- The MAE abbreviation is used several times for different meanings.
- The term "semantic" is used several times throughout the paper, but I would argue that the term "global" better captures the meaning the author would like to convey. Indeed, a very short temporal event could have a semantic relevance (take a sudden seizure event), which is likely captured in the middle-layer; in contrast, a long-lasting change captured by the deeper layer could have little semantic relevance. I would suggest avoiding using "semantic" and focusing on "global" which the author themselves used line 125.

### Questions
- How does the FGNO model compare to simple, supervised baselines (Linear, and deep res-net, deep convnets)?

- What does the author mean by "decoder" in their Figure 1?  They seem to evaluate the model on variable prediction (binary or regression), but the variable seems to refer to a reconstruction of the input signal.

- Did the author gain any novel understanding from the biological signal by applying FGNO compared to other models?

- I have not understood why authors refer to the function space operator? The architecture is indeed a transformer model, which does not make it directly act over function spaces but instead acts on the vectorized inputs. My understanding is that this makes the model presentation more complex than it should be and brings little insight into its behavior. I might be wrong here.

- Why did the author select these specific four datasets for evaluation? What are the specific difficulties these dataset offers that the model can deal with, compared to previous datasets?

- Would it be possible for the authors to provide training details (time, amount of data, number of GPU ect...)?

- Would it be possible for the authors to provide details on dataset preprocessing?

### Soundness
1

### Presentation
2

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
The paper presents a self-supervised method for time-series data based on flow-matching pre-training and short-term Fourier transform (STFT). The method is evaluated on multiple tasks on medical data, with good results at low-label settings.

### Strengths
* The motivation is clear.
* The presentation is good and well organized.
* The method is interesting and shows good results in multiple tasks (although not as good as the initial claims).

### Weaknesses
* The authors claim that FGNO consistently outperforms established baselines. These comments need to be moderated, as for example Table 2 shows that FGNO performs poorly for Epilepsy 100% labeled data.

* Results from Table 2 with 100% of labeled data are taken from [Emadeldeen et al. 2021] (TS-TCC: https://arxiv.org/abs/2106.14112). This should be clarified explicitly. I would also like to know if results from the 5% part were re-run or copied from another source.

* In Table 2: the performance reduction for FGNO between 100% and 5% labeled data is very small. All other methods show large reductions in performance. This difference is mentioned in the text, but it might merit more analysis or a deeper discussion. It could be related to running the method in different conditions to the baselines (e.g. multiple runs vs single run).

* The paper mentions that the method remains computationally efficient during probing, but there are no details of training and inference times, and no comparison to baselines. Presumably training time of Chronos is very high given the large amounts of data used, but how does inference compare? How does FGNO runtime compares to other baselines?

* There are no details about the baseline models (implementations used, multiple runs or single run, copied results vs re-run, settings, etc.). This is especially true for the for the robustness under data scarcity part (5% of labels in table 2).

* Missing information of STFT parameters (window size W, hop H).

* Making the source code available would be beneficial for reproducibility.

### Questions
* Table 2 with 5% data: were these results re-run or copied from another source? Were there multiple runs or a single run?

* How about baselines in other tables? (re-run? multiple runs? settings? implementations used?)

* How does the runtime compare with other methods?

* Please add information about STFT parameters.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces a self-supervised pretraining approach for time series data that relies on a flow matching objective. Univariate time series segments are represented using the short-time Fourier transform, and fed to a Transformer along with a denoising schedule parameter, used as conditioning. The model is then trained to predict the velocity of the denoising process at given schedule values. At inference time, both the schedule dimension and the model depth can be varied to find the optimal coarseness for a given downstream task. This approach is evaluated on biomedical downstream tasks based on intracortical and surface EEG, blood volume pulse and accelerometers time series. Results suggest the proposed approach outperforms existing baselines, including a time series foundation model, across the four tasks and in low labeled data regimes.

### Strengths
Originality:
* The combination of a flow matching objective for time series representation learning, along with the use of an STFT input representation, appears novel.

Clarity:
* The paper is well written and easy to follow despite some important methodological information missing.

Significance
* The results suggest that NFGO outperforms strong time series foundation model baselines. This kind of approach could be an interesting alternative to masked autoencoding and autoregressive approaches that are commonly applied on time series data to pretrain foundation models.

### Weaknesses
1. The lack of variability estimates makes it hard to compare performance with the baseline models. For instance, the three models reported in Table 1 are very close to each other on sleep classification. 
2. There are multiple missing critical methodological details, notably about datasets preparation and splitting (Q1-4).
3. The set of reported baseline models is limited, especially in Table 1. It is unclear why different baselines were used in Tables 1 and 2. There is also missing information about how the different baselines were pretrained/trained and evaluated, for instance MAE, CPC, SimCLR and the supervised baselines. Finally, especially given there are only four datasets, it would be informative to also report for each task the results obtained by domain/dataset-specific approaches as a point of reference (as is already done for the DREAMT dataset in Section 4.2).

### Questions
1. If I understand correctly, each NGFO model is pretrained on the downstream dataset it is going to be evaluated with, without access to labels. It is then reused in a supervised linear probing fashion using the labels. How are the datasets split for pretraining, supervised training, model selection, and final evaluation?  In the case of multivariate datasets, were all channels considered separately, and was the splitting done chronologically? Are the baselines (e.g. MAE, CPC, SimCLR, etc.) also pretrained then linear probed/finetuned using the same scheme? Does Chronos solely use existing external model checkpoints?
2. How were the different datasets preprocessed (assuming other steps were involved to get from the raw data to the STFT)? For instance, the windowing strategy will dramatically impact the number of examples available for training and can introduce leakage if done improperly.
3. Are the final performances reported for NGFO in e.g. Tables 1 and 2 obtained on the same set used for selecting the optimal depth and time, i.e. what is the difference between the validation set of Equation 7 and the final evaluation set? If so, these results are likely to be overfitted as compared to those of the baselines. To make baselines more comparable, a similar layer-wise evaluation of representations could be carried out for Chronos and the other baseline models (i.e. is another layer’s representation better than the last one?).
4. How were the 5% of labels selected for the experiment of Table 2? On biomedical tasks, a realistic scenario would be to keep 5% of subjects (or recordings) rather than randomly sample 5% of the training examples uniformly. If the latter strategy was used, this could explain the relatively similar performance obtained by the different methods on such a small percentage of the original training examples. 
5. Analysis of Figure 4: while FGNO achieves higher performance across the range of tested downsampling factors, the baselines actually do appear more stable; for instance, Chronos is consistently around 60% AUROC across all factors. FGNO, on the other hand, saw a decrease of around 10%. This contradicts the following claim: “Unlike FGNO, Chronos’s performance is more directly tied to the sampling rate of the input sequence.” (also repeated at line 480). I’d argue this is not true based on this figure, and if anything, this analysis suggests that while FGNO performs better than the baselines, it is more affected by changing input sampling frequencies than e.g. Chronos. Reporting variability would also help clarify this trend.
6. How does the proposed method compare to the baselines in terms of computational complexity?
7. How were the hyperparameters of the transformer selected, and does the approach scale with the quantity of training data/the number of parameters? Results on skin temperature regression (Figure 2) suggest that the representations of deeper layers are better for this downstream task, suggesting that deeper models might be beneficial.

Notes:
* The abstract says the model is evaluated across three biomedical domains, but I believe this should be four.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes an SSL approach for time series modelling. Input is first mapped into a short-time fourier space and varying strengths of noise are then applied at different network layers and flow times. The model is then pre-trained in a self-supervised fashion using a flow matching objective. Once pre-trained, representation at one of the layers is chosen for a supervised objective trained for the target task with the rest of the backbone frozen.

### Strengths
The paper is well written and easy to follow. Authors propose an interesting approach to use adaptive noise and representation layer for SSL. Empirical results on real-world datasets show that the proposed approach can outperform strong baselines. Authors also demonstrate that different settings for layer and noise level are needed to achieve top performance across datasets.

### Weaknesses
I have major concerns with the experiments section. First, authors use vary outdated baselines. In Table 2 the most recent baseline is almost four years old. Numerous time series methods have been proposed recently both foundation and task specific. GIFT-Eval is a popular benchmark for these methods (https://huggingface.co/spaces/Salesforce/GIFT-Eval) and many recent leading methods are listed there. I think a thorough comparison against leading recent baselines is needed to confirm the performance of the proposed method. Second, there is very little analysis of complexity, runtime or other aspects of FGNO. Equation 7 can be computationally expensive as one would have to sweep over many pairwise settings especially for deep models with many layers. There is also a risk of overfitting if validation set if probed repeatedly to select a fine grained setting. However, authors do not provide any runtime analysis or strategies to mitigate this complexity. I believe that the experimental section needs a major revision before this paper can be accepted for publication.

### Questions
Do you have runtime complexity to select optimal (l, s) pair in equation 7? Also, how stable is that hyper parameter landscape?

### Soundness
3

### Presentation
3

### Contribution
2
