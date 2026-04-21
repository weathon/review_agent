# Leveraging Diffusion Transformers for Stock Factor Augmentation in Financial Markets

- Avg Score: 4.75
- Decision: Reject
- Scores: 6, 5, 3, 5

## Abstract
Data scarcity poses a significant challenge in training machine learning models for stock forecasting, often leading to low signal-to-noise ratio (SNR) and data homogeneity that degrade model performance. To address these issues, we introduce DiffsFormer, a novel approach utilizing artificial intelligence-generated samples (AIGS) with a Transformer-based Diffusion Model. Initially trained on a large-scale source domain with conditional guidance to capture global joint distribution, DiffsFormer augments training by editing existing samples for specific downstream tasks, allowing control over the deviation of generated data from the target domain. We evaluate DiffsFormer on the CSI300 and CSI800 datasets using eight commonly used machine learning models, achieving relative improvements of 7.3\% and 22.1\% in annualized return ratio, respectively. Extensive experiments provide insights into DiffsFormer's functionality and its components, illustrating their role in mitigating data scarcity and enhancing model performance. Our findings demonstrate the potential of AIGS and DiffsFormer in addressing data limitations in stock forecasting, with the ability to generate realistic stock factors and control the editing process. These results validate our approach and contribute to a deeper understanding of its underlying mechanisms.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents a diffusion transformer-based approach for performing data augmentation in financial markets, addressing the problem of low signal-to-noise ratio and data homogeneity, which can reduce model performance. The authors demonstrate that the proposed approach can improve the performance of various models, leading to improved returns.

### Strengths
- The paper is well-written, with appropriate motivation provided by the authors.

- Improvements in returns and Weighted-IC are observed by applying the proposed approach.

- Extensive evaluations are provided, demonstrating improvements across a wide range of architectures.

### Weaknesses
- Lack of comparison with other time-series augmentation approaches proposed in the literature. This is perhaps the most important concern. Even though, in my experience, performing meaningful augmentation on financial data is indeed very tricky, this should be demonstrated by evaluating a number of baseline approaches. Authors are encouraged to compare with recent related approaches, such as:

Kollovieh, Marcel, et al. "Predict, refine, synthesize: Self-guiding diffusion models for probabilistic time series forecasting." Advances in Neural Information Processing Systems 36 (2024).


Seyfi, Ali, Jean-Francois Rajotte, and Raymond Ng. "Generating multivariate time series with COmmon Source CoordInated GAN (COSCI-GAN)." Advances in neural information processing systems 35 (2022): 32777-32788.

Wiese, Magnus, et al. "Quant GANs: deep generation of financial time series." Quantitative Finance 20.9 (2020): 1419-1440.

Xia, Haochong, et al. "Market-GAN: Adding Control to Financial Market Data Generation with Semantic Context." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 38. No. 14. 2024.


- Traditional forecasting metrics (RMSE, MAPE, MAE) are not evaluated. Although measuring returns and demonstrating improvements are important, they do not provide the full picture. There is a complex interplay between the investment strategy and the actual forecasting capabilities of a model. The use of Weighted-IC does not fully address these concerns. Authors could include a table showing RMSE, MAPE, and MAE results alongside the existing metrics. This would provide a more comprehensive evaluation of the model's forecasting capabilities.

- The models are evaluated only on forecasting tasks. What about classification (e.g., direction prediction) and DRL approaches? Is the generated data expected to be useful in these cases as well? Authors could include experiments demonstrating the method's performance on classification/DRL tasks or approrately discuss in the limitations section whether they expect their approach to generalize to other types of financial prediction tasks and why/why not.

- Lack of qualitative results to demonstrate the ability to condition the generation process. It would be useful to provide some qualitative results to more intuitively understand the effect of the proposed approach. Authors could include a figure showing different timeseries to demonstrate the effects of the proposed approach.

### Questions
Please comment on the weaknesses noted.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper introduces a method, called DiffsFormer, using a diffusion model to generate stock data, addressing the overfitting issue in stock prediction tasks that arises from low signal-to-noise ratios and data homogeneity. Two techniques are used besides the standard diffusion model. The first one is to train the model on a large dataset, called source domain, with a large number of diffusion steps, then get augmented data points starting from a smaller dataset, called target domain, with a smaller number fiffusion steps. The second one is the predictor-free guidence technique to generate factors given the labels and other information, which is from (Ho & Salimans, 2022). Experments on CSI300 and CSI500 were conducted to show the advantage of the proposed method.

### Strengths
The concept of applying a diffusion model to generate stock data is interesting, especially regarding its potential to alleviate homogeneity within the original dataset. The experiments are well-designed, demonstrating the impact of data augmentation using the diffusion model on various prediction methods in financial metrics, and assessing the quality of the generated data through metrics measuring data fidelity and diversity. The results also show an improvement in return ratios with the diffusion model compared to previous data augmentation methods on stock data, displaying the potential for improving stock prediction methods using generative models.

### Weaknesses
1. The presentation of the paper is poor. It's hard to understand the methods and experiments. Some examples are listed below.

The figures are not referenced in order, i.e. some later figures are referenced first. Besides, some figures (Figs. 1 and 3) are not referenced at all, which is quite confusing. 

The phrase “average number of stocks experiencing significant price drops” used to illustrate data homogeneity is confusing, and it is difficult to see how stocks within the same industry exhibit similar behavior based on the corresponding figure. 

Which model was used to generate the results in Fig. 1? What does Fig. 1 exactly tell? Additionally, it shows data betwen Oct 2023 to Aug 2024, which does not correspond to the stated test sample period. 

It's strange to claim in L237 "Since the target domain is a subset of the source domain... " considering that the relationship between the two domains hasn't been specified so far. 

What is a data point like, a 158 dimensional factor vector for one day, or a sequence of factors for 8 days, say a 8*158 matrix? 

In experiments, the source domains and target domains should be clearly staed in the main text instead of in Appendix because they are important information for understanding the results. How much more data was generated to augment the original dataset in experiments? 

In L237, what does $\hat x_{T'}^{(t)}$ stand for? And in this line, why does only the last term in the x sequence have a hat symbol? 

It's confusing to call the generation of a data point with the DM method as "editing" an existing data point. After reading the paper for several times I finally understood that this word is used in a metaphor. This word has caused much confusion to me. 

Line 265: I didn't understand the following sentence for a long time: Since our labels are continuous rather than discrete, we refer to this mechanism as “predictor-free guidance.” After reading the reference (Ho, Salimans, 2022) I realized that this name follows the "classifier-free guidance" in that paper. But the present paper didn't make it clear. 

What's CSIS in Table 4?  

The metric values of Excess Return (ER) in Tables 1 and 2 are the same as the Return Ration (RR) in Tables 8 and 9. Please explain.  

The authors are suggested to create a clear roadmap of the paper structure early on, ensure all figures are properly referenced and explained in order, and provide more explicit definitions of key concepts and variables used throughout.

2. The major technical contribution is training DM in the source domain with T diffusion steps, then get augmented data points in the target domain with T' diffusion steps, where T'<<T. I don't find experimental results directly supporting the advantage of this technique. It's suggested the authors show the results with different T'. This would demonstrate the impact and optimal setting of this key parameter. BTW, what values were used for T and T' for reporting the results in the paper? 

3. The results are not as good as stated in abstract. 

Tables 1&2 report weighted IC. It is true that weighted IC are more related to returns than IC and RankIC, but it is a more direct measurement of prediction accuracy since the prediction models are trained for predicting all stocks. Thus, it’s puzzling that many methods show lower IC after applying data augmentation with the diffusion model (Tables 8&9). 

Many comparison results in Tables 1, 2, 8 and 9 may not be reliable, considering the large standard devision compared with the minor difference in average values. For instance, in table 2, on average Ours exceeds Original by 0.012, while the STDs of the two methods are 0.038 and 0.022. Though Ours improves Original by 11.96%, the difference may not be statistically significant. Many results like this are present in these tables. It's suggested to perform significance  test (e.g., t-tests or ANOVA) to report the comparison results. 

The implementation of a “top30drop30” strategy for measuring excess returns raises concerns, as such strategies typically incur significant transaction costs in the A-share market. In our experience, such a strategy could never get positive returns on average. Please justify the choice of this strategy, provide results with transaction costs included, or implement and compare results using alternative strategies like the top-K approach.

### Questions
From Appendix E, it appears that IC and RankIC are actually worse for many prediction methods after applying the diffusion data augmentation. While it is true that accurately modeling tail stocks has little contribution to excess returns, it seems strange that so many methods exhibit lower IC and RankIC, given that the model is optimized for predicting all stocks. Please provide a more detailed analysis or explanation of why the proposed method improves excess returns while potentially decreasing overall prediction accuracy as measured by IC and RankIC.

How does T' influence the performance of the method? 

Regarding the calculation of excess returns, were transaction costs included?

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
The paper proposes a framework for increasing data by augmentation due to shortage of data in stock market prediction task. The authors propose to incorporate some target domain information while increase variation in augmented source domain data. Authors did so by, what they call, ‘editing’, which is essentially using diffusion models’ inherent properties to start from noisy version of target domain data. Authors also incorporate conditional information to simulate data by industry/sector.

### Strengths
The problem tackled in the paper is quite attractive indeed. Stock market prediction is one of the most financially lucrative applications of ML.

### Weaknesses
Note #1: I was assigned as an emergency reviewer and hence had limited time reading the paper. My assessment is based on evaluating the overall idea.

Note #2: I have little expertise on the specific application domain (stock market prediction) but have exposure to general generative modelling.

The paper, although tackles a lucrative problem, is in no way, novel. The paper mostly uses existing and well-known ideas and applies to a specific problem domain. Also, the writing quality is quite below the bar.

1. The (lack of) novelty is a big factor in my assessment of the paper. The paper uses Diffusion Models, which are not particularly known to be good for sequential data. There is not much discussion about why diffusion model or transformer architecture was chosen. The specific concept of ‘editing’ that was presented in the paper is quite a well-known idea called [SDEdit](https://arxiv.org/abs/2108.01073) which exists for quite a while. The authors did not cite or discuss it.
2. The time efficiency improvement proposed in section 3.2 is rather unnecessary and can be incorrect. The training complexity does not depend on T enhance it makes no difference to reduce the range of time during training. It can in fact lead to a poor model if one does not train it along the entire time horizon.
3. The idea of guidance in diffusion is also well known and the authors only seem to have used it as a black-box.

Moreover, the writing quality of the paper is below the bar. Technical concepts of diffusion models are not very well written or in some cases misleading or incorrect. The specifics of the application that is the variables and other quantities related to stock market prediction isn't very well explained.  The terms like stock factors in return ratio are used throughout the introduction section even though they are defined in section 2. This hinders the reading experience. The terms like SNR and data homogeneity are vaguely defined only in text. They should have been properly defined in mathematical terms.

Lastly, I feel like this paper being very specific in its application should be submitted in a domain specific conference and perhaps not a very good fit for ICLR.

### Questions
No further questions please see the witness section.

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper presents a novel approach, DiffsFormer, to address data scarcity in stock forecasting by leveraging diffusion-based data augmentation combined with transformer models. This method aims to mitigate the common challenges of low signal-to-noise ratio (SNR) and data homogeneity in stock data, thus improving predictive model performance.

### Strengths
+ The novel use of diffusion models for augmenting stock forecasting data is somehow technically sound.
+ Clear and robust experimental validation on real-world financial datasets.
+ Good writing and organization.

### Weaknesses
+ The approach uses transfer learning to train the diffusion model on a large source domain (i.e., broader stock market data) and applies it to a smaller target domain (e.g., CSI300 or CSI800). However, this raises the question of how domain differences (e.g., market structure, regulations, or trading behavior) may affect the generalizability of the learned knowledge. How does the model handle potential domain shift between source and target domains, particularly when the characteristics of the two datasets are fundamentally different (e.g., emerging vs. developed markets)?
+  The paper discusses the trade-off between data fidelity (keeping the data close to the original domain) and diversity (introducing variability in the data). Is there any risk of generating overfitted data that aligns too closely with the source domain but does not generalize well to the target domain? Could the approach lead to a loss of diversity in the generated data?
+ While excess return is a critical performance metric in financial markets, it may not fully capture the model’s stability or generalization ability, which are also important for real-world applications.
+ Diffusion models, especially when combined with complex transformers, are inherently difficult to interpret.  Could the authors explore feature importance or other interpretability techniques (e.g., SHAP, LIME) to provide insights into which factors most influence the model's decisions?
+ The paper compares DiffsFormer against multiple baseline models (e.g., LSTM, GRU, Transformer) and shows significant improvements. However, it is unclear if the comparison includes the latest state-of-the-art models in the domain of stock forecasting?
+  Are there alternative augmentation strategies, like generative adversarial networks (GANs), that could also serve as a competitive baseline?

### Questions
Check Weaknesses.

### Soundness
2

### Presentation
3

### Contribution
3
