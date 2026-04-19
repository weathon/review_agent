# NewTime: Numerically Multi-Scaled Embedding for Large-Scale Time Series Pretraining

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 5, 5, 3

## Abstract
Recent research on time-series self-supervised models shows great promise in learning semantic representations. However, it has been limited to small-scale datasets, e.g., thousands of temporal sequences. In this work, we make key technical contributions that are tailored to the numerical properties of time-series data and allow the model to scale to large datasets, e.g., millions of temporal sequences. We adopt the Transformer architecture by first partitioning the input into non-overlapping windows. Each window is then characterized by its normalized shape and two scalar values denoting the mean and standard deviation within each window. To embed scalar values that may possess arbitrary numerical scales to high-dimensional vectors, we propose a numerically multi-scaled embedding module enumerating all possible scales for the scalar values. The model undergoes pretraining using the proposed numerically multi-scaled embedding with a simple contrastive objective on a large-scale dataset containing over a million sequences. We study its transfer performance on a number of univariate and multivariate classification benchmarks. Our method exhibits remarkable improvement against previous representation learning approaches and establishes the new state of the art, even compared with domain-specific non-learning-based methods.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
In this paper, the authors propose a novel method for time series classification using large-scale time series pretraining. To handle time series from different domains with different scales, they propose a numerically multi-scaled embedding (NME). By combining NME with transformer, they pretain the model with a simple contrastive objective over a million sequences. After fine-tune the pretrained model, the proposed method beats the state-of-the-art methods on several univariate and multivariate classification tasks.

### Strengths
1. The proposed NME solves the scale problems arising from different problem domains which paves ways for building large-scale pretrain datasets from different domains and facilitate transfer learning across different domains.
2. A large-scale dataset was built with over one million time series from different domains.
3. Experimental validation looks solid.

### Weaknesses
1. Clarity of the proposed method could be improved. For example, if the authors can present a holistic view of the proposed method (including NME, transformer, and pretraining / fine tuning stages ), then it will help the audience to understand it more easily.
2. Novelty of the proposed NME is limited. Seems the authors choose scales manually based on experience or observations of existing time series data. Are there any automatic ways we can detect scales from time series and build scale embeddings accordingly.

### Questions
1. Page 1, instance normalization needs citation. Does NME fall in the category of the instance normalization?
2. It’s not clear to me what the contrastive loss function is in the context of time series. The authors are encouraged to illustrate it which may help audience to understand how the pretrain works.
3. For time series from different domains, it is hard to determine a universal window length which works for all time series and contains data of single scale. The authors may wanna clarify that how they resolve this problem.
4. Connected with the question above, it’s not clear to me how to split the time series and build a large-scale pretrain dataset.
5. Page 6, the authors mentioned that “datasets containing excessively lengthy sequences are exclude”. Why we would like to exclude these length sequences?
6. Table 1, the high performance of the proposed method looks suspicious. For example, the 100% accuracy of the proposed method on the EMG dataset makes me wondering whether there is data leakage in the pretraining stage. They authors are encouraged to double check their experimental settings.

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper offers a fresh approach to addressing the scaling challenges in time series pre-training. The core motivation behind this research is evident, and the proposed NME method offers some insights. However, the overall contributions appear somewhat one-dimensional and lack the significance to make this work distinct. Several facets of this study could benefit from further elaboration, and both the presentation and experiments have potential for enhancement.

### Strengths
1. The research motivation is convincingly presented with compelling evidence (e.g., Fig. 1). I concur with the authors regarding the scaling challenges inherent in time series pre-training and have observed recent research addressing this issue.
2. The introduced NME method offers a novel approach, allowing for finer-grained normalizations on time series patches beyond the traditional z-score.
3. Although the experiments could benefit from further refinement, it is evident that the NME results in notable performance improvements within the classification protocols in Tab. 4

### Weaknesses
1. The overall contributions (i.e., Fig. 3b and the discsusion below in page 5) appear somewhat one-dimensional and lack the significance to make this work distinct. The essence of the proposed NME appears to be a straightforward (and a brute-force) rescaling by considering all potential factors (i.e., k).

2. Several claims are very arbitrary and not well supported by the evidence:
- The statement "The dilemma between normalization for effective network optimization and high variation of data scales" found in the third paragraph of the introduction is ambiguous. I'd like clarity on how the authors characterize this dilemma and why a sequence of "normalization-optimization-denormalization" isn't relevant here.
- The claims "... data within each window has a simple structure that can be easily modeled at a single scale" and "Instead, we may assume that data within each window has a single scale of variation given the window size is small" are arbitrary. For example, with longer time series patches, the time series within a patch might still encounter the scaling challenges depicted in Fig. 1, making this assumtion fail to generalize well.
- Was there any reference to PatchTST? I couldn't find the acknowledgment in this statement "We follow the tokenization process in the Vision Transformer (Dosovitskiy et al., 2020) by splitting the time series sequence into non-overlapping windows."

3. Some technical designs are not well-motivated or clearly discussed:
- The reason for using BYOL isn't well-justified. BYOL emphasizes positive-only contrastive learning via the distillation, yet I don't observe a strong correlation between its primary features and this work. Why not consider other well-recognized frameworks, like SimCLR?
- What would be the complexity of NME when enumerating all possible scales and ensemble the embeddings across scales?

4. The experiments need furhther improvements to give more robust analysis of the proposed method over existing research.
- Testing on a variety of SSL frameworks and time series tasks (e.g., forecasting) would provide a more thorough evaluation of the proposal.

### Questions
See above.

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
This paper revisits the task of time series self-supervised models for classification under different settings. In particular, the authors argue that one of the main challenges for this task is that time series do not have an a-priori bounded range of values, contrary to the case of images (which are described through RGB values between 0 and 255) and text (which can be described through a finite dictionary or set of keys).

The authors propose a new normalization scheme based on non-overlapping windows for which statistics and the normalized values are processed and provided to a standard transformer architecture. With this, the authors show systematically that the proposed approach performs better and even closes the gap with models that heavily rely on domain knowledge.

### Strengths
1. The paper is well written and it is easy to follow.
2. The proposed approach is reasonable in the sense that partitioning time series into windows of suitable length, and computing the corresponding statistics, might provide an interesting set of opportunities for a new model.
3. The evaluations are extensive and ablations provide an interesting view of what are the parameter values that perform better.
4. The analysis of the key contribution of the paper, namely the function related to the propose normalization, is clear and makes the contribution understandable.

### Weaknesses
1. it is rather unclear how the proposed approach can be applied for time series forecasting. As the authors have cited several papers that focused on it, it remains as an open question how the proposed normalization strategy can be extended to time series forecasting.
2. It is unclear why the performance drops for the case of multivariate time series.
3. There is few insights into how robust is the proposed approach to time series with outliers. The authors do motivate the paper to some degree based on this, but then there is no analysis on how the model behaves specifically with those cases.
4.  There is no analysis on complexity of time execution of the proposed approach.
5. It is unclear if the authors have done cross-validation to choose the optimal parameters for window length and embedding dimension. The authors do present results in Table 8 of the appendix, but it is unclear if this values where chosen through cross-validation or based on the test set.
6. There is no code available.

### Questions
1. How are small time series handled ? For instance, the authors suggest to use a window size of 16. What happens if there is a time series with less than 16 values? Can one expect a single window to perform well on this? In case this has been implicitly done in the current experiments, can you please point me to an example of this?
2. How can missing values be handled in the proposed model? How robust is the proposed model towards time series that are sparse?
3. Do we have a notion of how heavily noisy time series can challenge the proposed approach?
4. What is the complexity / time execution of the proposed approach?
5. In equations 1 and 2, it is mentioned that $w$ and $b$ are initialized with a Gaussian Distribution. Since $w$ and $b$ are vectors, are these Gaussian distributions multivariate, i.e $\mathcal{N} (0_n, \sigma^2_w I_n ), \mathcal{N} (0_n, \sigma^2_b I_n ) $?
6. I appreciate the effort on providing insights in section 3.3. I think the authors have done a great job here. I would very much suggest to add a Lemma or Observation that formalizes the analysis provided. In this way the authors consolidate the insights, provides rigour, and makes it easier to cite in the future. 

I would draft the results as follows.

Let $\gamma, \beta, w, b\in\mathbb{R}$. Then 
$$\lim_{x\to \pm \infty} LN(FC(x)) = \pm \gamma \frac{w}{\sigma_w} + \beta $$

In particular, for $\gamma=1$, $\beta=0$ we have 
$$\lim_{x\to \pm \infty} LN(FC(x)) = \pm \frac{w}{\sigma_w} $$

The draft of the proof would potentially be something along the following lines:

$$
\lim_{x\to \pm \infty} LN(FC(x)) = \lim_{x\to \pm \infty} \gamma \Bigg( \frac{x \cdot w + k \cdot b}{\sqrt{x^2 \sigma_w^2 + k^2 \sigma_b^2}} \Bigg) + \beta = \lim_{x\to \pm \infty} \gamma \Bigg( \frac{ w + \frac{k \cdot b}{x}}{\sqrt{\sigma_w^2 + \frac{k^2 \sigma_b^2}{x^2}}} \Bigg) + \beta = \pm \gamma \frac{w}{\sigma_w} + \beta
$$

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper presents the NewTime model, which is a Transformer-based architecture specifically crafted for the purpose of learning representations for time series data that have vastly different amplitudes. Authors use the representation for the classification downstream task.

### Strengths
**Originality:** The authors appear to tackle a significant problem related to current unsupervised time series representation learning methods.

**Quality and clarity:** Overall, the paper is well presented, although there are certain confusing parts, such as the concept of scale (see below).

**Significance:** Undoubtedly, the development of a universal representation learning method for time series would be a noteworthy contribution to the field. However, additional evidence is required to substantiate these claims.

### Weaknesses
* There is much more to time series analysis than just classification. If authors claim they are doing representation learning the resulting representation should be useful for other types of tasks commonly performed on time series.

* Seems the niche that this paper aims to fill is to cleverly deal with the vastly different amplitudes (not scale) in time series. While authors mention why common normalization methods are not suitable, they fail to acknowledge that the [Wavelet Scattering Spectra](https://arxiv.org/abs/2204.10177) representation of time series is indeed invariant to amplitude, hence, has the "scale" (read amplitude) invariance that authors are after. This omission is especially surprising because the authors intend to assert that their method surpasses domain-specific non-learning-based methods and they fail to adequately review the current state of the art non-learnable methods in this domain.

* The term "scale" is used in a misleading manner throughout the paper. In the context of time series, scale is typically associated with the frequency of the signal (see for example the wavelet transform). However, in this paper, scale appears to be more closely related to the amplitude of the signal. This discrepancy is confusing and demands clarification. The authors should avoid the usage of the term "scale" or make it clear that they are using it in a different sense.

* Related to the previous point, a significant challenge in time series analysis lies in effectively handling the multi-scale (i.e., containing various temporal resolutions, rather than being related to amplitude variations) nature of certain time series. I do not see any justification for how this method is capable of effectively addressing such challenging signals without incorporating any specific inductive bias designed for handling multi-scale characteristics.

### Questions
* Are the baseline self-supervised learning methods pretrained on the same datasets? If not, how do authors know the improvement in downstream classification result is not due to the difference in pretraining data?

* "Due to great domain expertise being engineered into the state-of-the-art methods, only one deep learning method InceptionTime (Ismail Fawaz et al., 2020) is able to rank in the top 10 of the leaderboard" This seems to imply embedding domain knowledge into the design of a deep model is not a good idea. I am not sure the majority of the scientific machine learning community would agree with this.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair
