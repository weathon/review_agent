

{0}------------------------------------------------

# DOES EXAMPLE SELECTION FOR IN-CONTEXT LEARNING AMPLIFY THE BIASES OF LARGE LANGUAGE MODELS?

Anonymous authors

Paper under double-blind review

## ABSTRACT

In-context learning (ICL) has proven to be adept at adapting large language models (LLMs) to downstream tasks without parameter updates, based on a few demonstration examples. Prior work has found that the ICL performance is susceptible to the selection of examples in prompt and made efforts to stabilize it. However, existing example selection studies ignore the ethical risks behind the examples selected, such as gender and race bias. In this work, we first construct a new sentiment classification dataset —*EEC-paraphrase*, designed to better capture and evaluate the biases of LLMs. Then, through further analysis, we discover that ❶ **example selection with high accuracy does not mean low bias**; ❷ **example selection for ICL amplifies the biases of LLMs**; ❸ **example selection contributes to spurious correlations of LLMs**. Based on the above observations, we propose the *Remind with Bias-aware Embedding (ReBE)*, which removes the spurious correlations through contrastive learning and obtains bias-aware embedding for LLMs based on prompt tuning. Finally, we demonstrate that ReBE effectively mitigates biases of LLMs without significantly compromising accuracy and is highly compatible with existing example selection methods. *The implementation code is available at <https://anonymous4open.science/r/ReBE-1D04>.*

## 1 INTRODUCTION

Although large language models (LLMs) have demonstrated impressive capabilities, efficiently deploying them into downstream tasks remains challenging (Mosbach et al., 2023; Liu et al., 2022a). Among existing solutions, in-context learning (ICL) has proven adept at adapting LLMs to downstream tasks without parameter updates, using only a few demonstration examples (Brown et al., 2020). Compared to fine-tuning (Ziegler et al., 2019), ICL is more flexible and suitable for few-shot scenarios. **In the setting of ICL, examples included in the prompt are the only source for LLMs to learn the task context information (e.g., the answer format), thus attracting considerable attention.** As the research deepened, researchers found that examples selected randomly from the training set led to high variance in performance (Liu et al., 2022b), so numerous example selection methods have been proposed to stabilize the performance of ICL (Gonen et al., 2023; Gupta et al., 2023).

Since LLMs may spread biases learned from the training set during decision-making or user interaction, potentially causing severe harm to society, the biases of LLMs have always attracted significant attention (Liu et al., 2024b; Gupta et al., 2024; Guo et al., 2022). Although not entirely equivalent to social biases, it has been shown that LLMs exhibit stronger cognitive biases (Lin & Ng, 2023), such as position bias (Zhao et al., 2021) and token bias (Zheng et al., 2024), when fed with specific prompts. Similarly, because the example selection method determines the content of the ICL prompt, it is natural to ask: **Does example selection for ICL amplify the biases of LLMs?** It is undoubtedly unacceptable for LLMs to preserve or even exacerbate biases when using ICL to deploy LLMs to downstream tasks. However, **existing example selection studies ignore the ethical risks behind the examples selected**, such as gender and race bias.

To explore the impact of example selection on bias, we conduct an empirical analysis by evaluating the accuracy and biases of LLMs on a sentiment classification dataset —*EEC-paraphrase*, which

{1}------------------------------------------------

![Figure 1: A multi-panel figure illustrating gender bias and accuracy of OPT-13B. The central scatter plot shows Bias (y-axis, 0.000 to 0.150) vs. Acc (x-axis, 0.76 to 0.91). A vertical red dashed line at Acc ≈ 0.81 marks the 'Acc of zero-shot' baseline. A horizontal red dashed line at Bias ≈ 0.04 marks the 'Bias of zero-shot' baseline. A grey shaded region in the top-right is labeled 'High Accuracy High Bias'. Data points for Random (blue x), DPP+ReBE (orange circle), Perplexity (green triangle), Similarity (yellow diamond), and DPP (purple inverted triangle) are plotted. Most points are in the high-accuracy, high-bias region. The left subfigure shows a pipeline: ICL Prompt (Context: {q1, q2}, Question: {q3, q4}) -> Example Selection -> LLM -> Downstream Task -> Evaluate. The right subfigure shows a similar pipeline but with ReBE in the prompt. The bottom subfigure is a box plot of Bias for the five baselines: Random, Perplexity, Similarity, DPP, and DPP+ReBE. The 'Zero-shot' baseline is shown as a blue bar, and 'Example selection' baselines are shown as orange bars. The box plots show that example selection baselines generally have higher bias than the zero-shot baseline, with DPP+ReBE showing a significant reduction in bias.](49ad3a646d84bcfeac02bdf2b3792a3e_img.jpg)

Figure 1: A multi-panel figure illustrating gender bias and accuracy of OPT-13B. The central scatter plot shows Bias (y-axis, 0.000 to 0.150) vs. Acc (x-axis, 0.76 to 0.91). A vertical red dashed line at Acc ≈ 0.81 marks the 'Acc of zero-shot' baseline. A horizontal red dashed line at Bias ≈ 0.04 marks the 'Bias of zero-shot' baseline. A grey shaded region in the top-right is labeled 'High Accuracy High Bias'. Data points for Random (blue x), DPP+ReBE (orange circle), Perplexity (green triangle), Similarity (yellow diamond), and DPP (purple inverted triangle) are plotted. Most points are in the high-accuracy, high-bias region. The left subfigure shows a pipeline: ICL Prompt (Context: {q1, q2}, Question: {q3, q4}) -> Example Selection -> LLM -> Downstream Task -> Evaluate. The right subfigure shows a similar pipeline but with ReBE in the prompt. The bottom subfigure is a box plot of Bias for the five baselines: Random, Perplexity, Similarity, DPP, and DPP+ReBE. The 'Zero-shot' baseline is shown as a blue bar, and 'Example selection' baselines are shown as orange bars. The box plots show that example selection baselines generally have higher bias than the zero-shot baseline, with DPP+ReBE showing a significant reduction in bias.

Figure 1: The **central** scatter figure plots gender bias and accuracy of OPT-13B under various example selection baselines. The horizontal and vertical red dashed lines represent mean accuracy and maximum bias (AvgGF) of OPT-13B under zero-shot, respectively. The **left** subfigure shows the pipeline for adapting LLMs to downstream tasks using ICL. The **right** subfigure illustrates the pipeline using our debiasing method, ReBE. The box plot at the **bottom** depicts the gender bias distribution of OPT-13B under various baselines.

we build on *Equity Evaluation Corpus (EEC)* (Kiritchenko & Mohammad, 2018) but with more complex and natural sentences (More details in Section 3). Considering the generality of the findings, our experiments include eight LLMs and four example selection baselines: Random-based, Similarity-based (Liu et al., 2022b), Perplexity-based (Gonen et al., 2023) and Determinantal Point Processes (DPP)-based (Ye et al., 2023). We use random seeds to sample the *EEC-paraphrase* to construct the few-shot training sets and have collected the bias and accuracy results of baselines under various random seeds. Therefore, we emphasize that the data points of example selection baselines in Figure 1 are evaluation results under different random seeds. According to Figure 1, each example selection baseline has points in the grey area marked as “high accuracy and high bias”, indicating that **example selection with high accuracy does not mean low bias**.

To observe the impact of example selection on biases compared to the case without ICL, we have also collected the experiment results of zero-shot under various random seeds and plotted the red dashed line “Bias of zero-shot” with the maximum bias value in Figure 1. The data points above the horizontal red dashed line in Figure 1 exhibit higher gender bias than zero-shot, indicating that **example selection for ICL does amplify the bias of LLMs**. According to the results in Section 3.3, we further find that example selection amplifies the **maximum bias value**, worsening unfair situations. The maximum bias value refers to the highest bias among results measured under various random seeds using the same example selection method. To uncover why example selection amplifies the biases, based on the MaxTG and MaxFG metrics (Table 1), we observe that LLMs using ICL exhibit **spurious correlations**. **Spurious correlations refer to undesired or unstable correlations learned by LLMs from the training set, which may introduce unintended biases (Albuquerque et al., 2024).** Typical spurious correlations of LLMs include stereotypes such as “He is a doctor; she is a nurse.” Furthermore, it is generally believed that the LLM’s biases come from its parameter knowledge and the input prompt. By excluding the impact of LLM parameters, we find that **example selection contributes to spurious correlations of LLMs**.

The above observations highlight that example selection for ICL truly amplifies the biases of LLMs. In order to mitigate the social biases of adapting LLMs to downstream tasks through ICL, we propose the *Remind with Bias-aware Embedding (ReBE)*, which curbs biases of LLMs by prefixing the bias-aware embedding into the prompt. Besides, we design the bias-contrastive loss based on contrastive learning to remove spurious correlations and obtain the bias-aware embedding through prompt tuning (More details in Section 4). To demonstrate the effectiveness of ReBE, we conduct extensive experiments and the results in Section 5 show that ReBE reduces the maximum bias value without compromising the accuracy and is well compatible with existing example selection methods. In sum, we try to fill the gap in exploring the ethical risks of example selection, which is essential

{2}------------------------------------------------

for deploying LLMs into downstream tasks using ICL. The overall contributions are summarized as follows:

- To the best of our knowledge, **we are the first** to discover the bias risks of example selection for ICL, especially the findings: ❶ Example selection with high accuracy does not mean low bias; ❷ Example selection for ICL amplifies the biases of LLMs; ❸ Example selection contributes to spurious correlations of LLMs.
- We construct a new sentiment classification dataset —*EEC-paraphrase*, which can better identify and evaluate gender and race bias of LLMs in ICL. More specifically, sentences in *EEC-paraphrase* are more complex and natural than in *EEC*.
- To alleviate the bias amplification of example selection, we propose the **Remind with Bias-aware Embedding (ReBE)**, which removes spurious correlations by minimizing the bias-contrastive loss while preserving the advantages of ICL through prompt tuning.
- We conduct extensive experiments to validate the effectiveness of ReBE, including four LLMs and four example selection baselines.

## 2 PRELIMINARIES

### 2.1 EXAMPLE SELECTION FOR ICL

Given a test input  $x_{test}$ , ICL enables the language model  $\mathcal{M}$  to learn how to generate  $y_{test}$  from just a few examples in the context  $C$ . The above process can be formulated as:

$$\hat{y} = \arg \max_{y \in \mathcal{Y}} p_{\mathcal{M}}(y|C, x_{test}), \quad (1)$$

where  $\hat{y}$  is the prediction,  $\mathcal{Y}$  is the label set, and  $p_{\mathcal{M}}(y|C, x_{test})$  represents the probability that  $\mathcal{M}$  generates  $y$  with context  $C$  and  $x_{test}$  as input. For a task with training set  $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$ , if context  $C$  contains  $k$  examples ( $k$ -shot prompt), then  $C = \{(x_1, y_1), (x_2, y_2), \dots, (x_k, y_k)\} \subset \mathcal{D}$ .

Among current studies (Iter et al., 2023; Yang et al., 2023), example/demonstration selection and example/demonstration retriever are interchangeable. To avoid confusion, we use the term *example selection* throughout this paper. Since the performance of  $\mathcal{M}$  depends on context  $C$ , we need to select examples  $(x_i, y_i)$  to minimize the total loss on the test set  $(\mathbf{x}_{test}, \mathbf{y}_{test})$ , which could be formulated as the following problem:

$$C^* = \arg \min_{C \subset \mathcal{D}} \mathcal{L}_{\mathcal{M}}(\tilde{\mathbf{y}}, \mathbf{y}_{test}), \quad (2)$$

where  $\tilde{\mathbf{y}} = \{\arg \max_{y \in \mathcal{Y}} p_{\mathcal{M}}(y|C, x_{test})\}$ ,  $\mathbf{x}_{test} \in \mathbf{x}_{test}$ , and  $C^*$  is the desired sample subset of example selection methods.

### 2.2 CONTRASTIVE LEARNING

Contrastive learning aims to obtain representation by maximizing the similarity between related samples and minimizing the similarity between unrelated samples, simultaneously. Although originating from self-supervised learning, contrastive learning also proves useful in supervised learning (Khosla et al., 2020; Chen et al., 2022). Given a training set  $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$  and its indexes set  $\mathcal{I} = \{1, 2, \dots, N\}$ , define the  $i$ -th sample  $x_i$  as an *anchor*, the contrastive loss for supervised tasks (Khosla et al., 2020) can be defined as:

$$\mathcal{L}_{sup} = - \sum_{i \in \mathcal{I}} \frac{1}{|\mathcal{P}(i)|} \sum_{p \in \mathcal{P}(i)} \log \frac{\exp(z_i \cdot z_p / \tau)}{\sum_{a \in \mathcal{A}(i)} \exp(z_i \cdot z_a / \tau)}, \quad (3)$$

where  $z_i$  is the normalized representation of anchor  $x_i$ ,  $\mathcal{P}(i) = \{p \in \mathcal{A}_i : y_p = y_i\}$  is the index set of *positive* samples.  $\mathcal{A}_i = \mathcal{I} \setminus \{i\}$  is the index set of contrastive samples that removes  $i$  from set  $\mathcal{I}$  and  $\tau$  is the temperature parameter. Constructing sensible  $\mathcal{P}(i)$  and  $\mathcal{A}(i)$  is vital to utilizing the contrastive learning framework.

{3}------------------------------------------------

## 3 EXPLORE THE IMPACT OF EXAMPLE SELECTION ON LLM BIASES

### 3.1 DATASET AND MODELS

**Dataset** To better capture and evaluate the gender and race bias of LLMs, we construct a new sentiment classification dataset —*EEC-paraphrase*. Given a sentence in the template <Person> feels <emotional word>., LLMs are asked to identify the sentiment contained in the sentence. By replacing <Person> with first names (e.g., Alonzo and Alan) or pronouns (e.g., she and he) associated with specific demographic group, *EEC-paraphrase* includes 8,640 English sentences with gender and race attributes.

*EEC-paraphrase* is built through paraphrasing sentences in the *Equity Evaluation Corpus (EEC)* (Kiritchenko & Mohammad, 2018) by GPT-3.5-Turbo. Compared with *EEC*, sentences in *EEC-paraphrase* are more complex and natural, closer to the actual scenario (The quality validation is available in Appendix A.). Besides, to simulate the few-shot scenario, we build a *train400-dev200* dataset by randomly sampling 400 sentences for the training set and 200 sentences for the development set from the *EEC-paraphrase*.

**Language Models** To guarantee the reliability of our findings, we conduct experiments on eight LLMs, including LLaMA-2-7/13/70B, OPT-6.7/13/30B, GPT-J-6B and GPT-neo-2.7B. LLMs with various parameter sizes but within the same series facilitate our analysis of the effects of parameter quantities.

Table 1: Bias metrics for sentiment classification.

| Metric                 | Formula                                                                                                                                    |
|------------------------|--------------------------------------------------------------------------------------------------------------------------------------------|
| Average Group Fairness | $\text{AvgGF} =  P(\hat{Y}=Y S=s_1) - P(\hat{Y}=Y S=s_2) $                                                                                 |
| Maximum TPR Gap        | $\text{MaxTG} = \max_{y \in \mathcal{Y}}  P(\hat{Y}=y Y=y \cap S=s_1) - P(\hat{Y}=y Y=y \cap S=s_2) $                                      |
| Maximum FPR Gap        | $\text{MaxFG} = \max_{y, \hat{y} \in \mathcal{Y}, \hat{y} \neq y}  P(\hat{Y}=\hat{y} Y=y \cap S=s_1) - P(\hat{Y}=\hat{y} Y=y \cap S=s_2) $ |

\*  $s_1$  and  $s_2$  correspond to different demographic groups.

To further validate the generalizability of our findings, we evaluate LLMs on the toxicity detection task using the <sup>1</sup>Jigsaw dataset. The results are available in Appendix F.

### 3.2 BIAS METRICS AND BASELINES

Since the output of LLMs is not numerical value but sentences containing the judgment result, we evaluate the prediction’s accuracy by comparing the semantic similarity between the answer and options.

**Metrics** Drawing on fairness metrics of machine learning (Mehrabi et al., 2021) and natural language processing (Czarnowska et al., 2021), we summarize three representative bias metrics in Table 1, which adapts to the sentiment classification task. The basis for selecting metric is whether it can reflect the unfairness or stereotypes of different groups in various sentiments. See Appendix B for a detailed explanation of metrics.

**Baselines** We select four example selection methods as baselines to study the impact of example selection on the biases of LLMs. *Random-based* example selection refers to randomly choosing examples from the training set to form a few-shot prompt. *Similarity-based* (Liu et al., 2022b) and *perplexity-based* example selection (Gonen et al., 2023) picks the top-k examples based on semantic similarity and perplexity of example, respectively. *Determinantal Point Processes (DPP)-based* example selection (Ye et al., 2023) uses DPP to consider two properties simultaneously when selecting examples.

<sup>1</sup>Jigsaw unintended bias in toxicity classification

{4}------------------------------------------------

### 3.3 IMPACTS OF EXAMPLE SELECTION ON BIAS OF LLMs

Although example selection aims to stabilize the performance of LLMs using ICL, inappropriate examples selected may also mislead LLMs. We assess the change in LLM biases when using example selections for ICL compared to zero-shot. **Figure 2 illustrates the differences in the maximum and mean bias values between random-based example selection and zero-shot.** The comparisons of the remaining example selection baselines are available in the Appendix C.1. It is evident that, although example selections reduce the mean bias value, **the LLMs tested exhibit varying degrees of increase in the maximum gender or race bias value with random-based example selection for ICL.** In other words, **example selection for ICL amplifies the biases of LLM,** increases the fluctuation of biases and exacerbates the unfair risks. Besides, the maximum bias values among LLMs for each baseline are highlighted in Table 2 and are significantly higher than the mean values.

![Figure 2: Six bar charts showing the impact of random-based example selection on gender and race bias for various LLMs. The charts are arranged in a 3x2 grid. The left column shows 'Diff of [maximum] values of bias' and the right column shows 'Diff of [mean] values of bias'. The rows represent different bias metrics: AvgGF (top), MaxTG (middle), and MaxFG (bottom). The x-axis for all charts lists the models: gpt-j-6b, opt-6-7b, opt-13b, opt-30b, Llama-2-7b-chat-hf, Llama-2-13b-chat-hf, and Llama-2-70b-chat-hf. Each model has two bars: 'Diff of max gender bias' (light blue) and 'Diff of max race bias' (dark blue). In the 'Diff of [maximum] values of bias' charts, most bars are positive, indicating an increase in maximum bias. In the 'Diff of [mean] values of bias' charts, most bars are negative, indicating a decrease in mean bias.](6de7dcb072cef2388026fb0f504084b2_img.jpg)

Figure 2: Six bar charts showing the impact of random-based example selection on gender and race bias for various LLMs. The charts are arranged in a 3x2 grid. The left column shows 'Diff of [maximum] values of bias' and the right column shows 'Diff of [mean] values of bias'. The rows represent different bias metrics: AvgGF (top), MaxTG (middle), and MaxFG (bottom). The x-axis for all charts lists the models: gpt-j-6b, opt-6-7b, opt-13b, opt-30b, Llama-2-7b-chat-hf, Llama-2-13b-chat-hf, and Llama-2-70b-chat-hf. Each model has two bars: 'Diff of max gender bias' (light blue) and 'Diff of max race bias' (dark blue). In the 'Diff of [maximum] values of bias' charts, most bars are positive, indicating an increase in maximum bias. In the 'Diff of [mean] values of bias' charts, most bars are negative, indicating a decrease in mean bias.

Figure 2: The impacts of random-based example selection on biases of LLMs. The bar value is calculated by  $\text{Diff} = \text{Bias}_{\text{random}} - \text{Bias}_{\text{zero-shot}}$ .

Table 2: Accuracy and gender bias of LLMs under four example selection baselines.

|            |                        | GPT-J-6B          | GPT-neo-2.7B      | OPT-6.7B          | OPT-13B           | OPT-30B           | Llama-2-7B | Llama-2-13B | Llama-2-70B |
|------------|------------------------|-------------------|-------------------|-------------------|-------------------|-------------------|------------|-------------|-------------|
| Random     | Acc <sub>C(Min)</sub>  | 0.84(0.80)        | 0.77(0.58)        | 0.81(0.67)        | 0.82(0.72)        | 0.84(0.76)        | 0.86(0.81) | 0.87(0.83)  | 0.86(0.82)  |
|            | AvgGF <sub>(Max)</sub> | 0.04(0.08)        | <b>0.04(0.13)</b> | <b>0.04(0.13)</b> | 0.04(0.12)        | <b>0.04(0.13)</b> | 0.03(0.08) | 0.04(0.09)  | 0.04(0.09)  |
|            | MaxTG <sub>(Max)</sub> | 0.15(0.29)        | 0.14(0.31)        | <b>0.18(0.47)</b> | 0.17(0.38)        | <b>0.17(0.47)</b> | 0.11(0.22) | 0.14(0.25)  | 0.17(0.30)  |
|            | MaxFG <sub>(Max)</sub> | 0.17(0.26)        | 0.20(0.39)        | <b>0.20(0.46)</b> | 0.19(0.34)        | <b>0.19(0.46)</b> | 0.13(0.22) | 0.14(0.21)  | 0.17(0.30)  |
| Perplexity | Acc <sub>C(Min)</sub>  | 0.83(0.72)        | 0.82(0.82)        | 0.85(0.81)        | 0.83(0.79)        | 0.86(0.85)        | 0.86(0.8)  | 0.87(0.84)  | 0.86(0.85)  |
|            | AvgGF <sub>(Max)</sub> | <b>0.09(0.15)</b> | 0.08(0.08)        | 0.04(0.09)        | <b>0.05(0.10)</b> | 0.05(0.09)        | 0.03(0.04) | 0.04(0.07)  | 0.05(0.08)  |
|            | MaxTG <sub>(Max)</sub> | <b>0.23(0.38)</b> | 0.18(0.18)        | <b>0.21(0.35)</b> | 0.22(0.32)        | 0.20(0.35)        | 0.18(0.33) | 0.17(0.28)  | 0.20(0.27)  |
|            | MaxFG <sub>(Max)</sub> | <b>0.24(0.50)</b> | <b>0.24(0.50)</b> | 0.17(0.31)        | 0.27(0.46)        | 0.17(0.22)        | 0.14(0.28) | 0.14(0.19)  | 0.17(0.22)  |
| Similarity | Acc <sub>C(Min)</sub>  | 0.92(0.88)        | 0.85(0.82)        | 0.84(0.82)        | 0.87(0.86)        | 0.90(0.86)        | 0.93(0.90) | 0.92(0.90)  | 0.89(0.87)  |
|            | AvgGF <sub>(Max)</sub> | 0.03(0.06)        | <b>0.03(0.09)</b> | 0.03(0.05)        | <b>0.04(0.09)</b> | 0.02(0.04)        | 0.03(0.08) | 0.03(0.04)  | 0.04(0.07)  |
|            | MaxTG <sub>(Max)</sub> | 0.13(0.28)        | <b>0.19(0.30)</b> | 0.12(0.22)        | <b>0.21(0.38)</b> | 0.13(0.30)        | 0.16(0.27) | 0.16(0.25)  | 0.16(0.23)  |
|            | MaxFG <sub>(Max)</sub> | 0.14(0.20)        | 0.16(0.19)        | <b>0.15(0.31)</b> | <b>0.17(0.37)</b> | 0.11(0.18)        | 0.13(0.21) | 0.17(0.27)  | 0.13(0.16)  |
| DPP        | Acc <sub>C(Min)</sub>  | 0.93(0.89)        | 0.89(0.83)        | 0.87(0.79)        | 0.89(0.82)        | 0.91(0.86)        | 0.94(0.90) | 0.93(0.91)  | 0.90(0.85)  |
|            | AvgGF <sub>(Max)</sub> | 0.03(0.06)        | 0.03(0.07)        | <b>0.04(0.11)</b> | <b>0.03(0.12)</b> | 0.02(0.06)        | 0.02(0.06) | 0.02(0.05)  | 0.03(0.08)  |
|            | MaxTG <sub>(Max)</sub> | 0.12(0.28)        | <b>0.13(0.28)</b> | 0.14(0.27)        | <b>0.13(0.38)</b> | 0.11(0.28)        | 0.10(0.18) | 0.10(0.25)  | 0.12(0.25)  |
|            | MaxFG <sub>(Max)</sub> | 0.10(0.17)        | 0.13(0.24)        | <b>0.14(0.27)</b> | <b>0.12(0.38)</b> | 0.10(0.18)        | 0.09(0.22) | 0.09(0.17)  | 0.12(0.21)  |

<sup>†</sup> Avg<sub>C(Min)</sub> are the largest two values in AvgGF; Avg<sub>G(Min)</sub> are the largest two values in MaxTG and MaxFG.

{5}------------------------------------------------

### 3.4 SPURIOUS CORRELATIONS OBSERVED WITH MAXTG AND MAXFG

As seen from Figure 3, we visualize the confusion matrices of OPT-6.7B, which has the biggest fluctuation of MaxTG (0.47) and MaxFG (0.46) in Table 2. With the help of Figure 3, we can further analyze the reasons that cause MaxTG and MaxFG to increase. For MaxTG, by comparing the first two sub-figures of Figure 3 by row, the proportion of sadness sentences correctly predicted in the female group (0.88) is higher than in the male group (0.42), which is consistent with the finding of Plaza-del Arco et al. (2024). Likewise, for MaxFG, by comparing the first two sub-figures of Figure 3 by column, more sentences with *sadness* labels are incorrectly predicted as *fear* in the male group (0.54) than in the female (0.08). We believe the disparity —where sentences labelled as *sadness* containing male pronouns are more easily misjudged as *fear* than those with female pronouns— occurs because the sentiment analysis criteria of LLMs may be influenced by words other than emotional ones, leading to **spurious correlations**. Although it has been proven that spurious correlations exist in LLMs, we are unsure whether the example selection methods for ICL contribute to these spurious correlations.

![Figure 3: Confusion matrix heatmaps of OPT-6.7B for four demographic groups: Female, Male, African-American, and European-American. Each heatmap shows the relationship between True label (rows: anger, fear, joy, sadness) and Predicted label (columns: anger, fear, joy, sadness). The Female group shows high accuracy for sadness (0.88), while the Male group shows a high misclassification of sadness as fear (0.54).](46f43cb4ffd47565e7c0ca306d461435_img.jpg)

Figure 3 displays four confusion matrix heatmaps for OPT-6.7B, categorized by demographic groups: Female, Male, African-American, and European-American. Each heatmap shows the relationship between True label (rows: anger, fear, joy, sadness) and Predicted label (columns: anger, fear, joy, sadness). The color scale ranges from 0.00 (light yellow) to 1.00 (dark red).

- opt-6.7b Female:** True labels: anger (0.66, 0.31, 0.04, 0.04), fear (0.00, 0.92, 0.04, 0.04), joy (0.00, 0.00, 1.00, 0.00), sadness (0.00, 0.08, 0.04, 0.88).
- opt-6.7b Male:** True labels: anger (0.57, 0.33, 0.04, 0.00), fear (0.10, 0.92, 0.00, 0.00), joy (0.00, 0.00, 1.00, 0.00), sadness (0.00, 0.54, 0.04, 0.42).
- opt-6.7b African-American:** True labels: anger (0.46, 0.53, 0.05, 0.10), fear (0.00, 0.95, 0.05, 0.00), joy (0.00, 0.00, 1.00, 0.00), sadness (0.00, 0.21, 0.00, 0.86).
- opt-6.7b European-American:** True labels: anger (0.75, 0.16, 0.00, 0.05), fear (0.00, 0.84, 0.00, 0.24), joy (0.00, 0.00, 1.00, 0.00), sadness (0.00, 0.13, 0.07, 0.71).

Figure 3: Confusion matrix heatmaps of OPT-6.7B for four demographic groups: Female, Male, African-American, and European-American. Each heatmap shows the relationship between True label (rows: anger, fear, joy, sadness) and Predicted label (columns: anger, fear, joy, sadness). The Female group shows high accuracy for sadness (0.88), while the Male group shows a high misclassification of sadness as fear (0.54).

Figure 3: Confusion matrix heatmaps of OPT-6.7B.

Two factors affect the biases of LLMs: the LLM parameters and the input prompt. The former refers to biased knowledge that LLMs acquire during pre-training, which we call *native bias*. To isolate the influence of native bias, we use **null (content-free) prompts** (Lin & Ng, 2023; Zhao et al., 2021) to observe the tendency of LLMs parameters. More specifically, the null prompt fills the <Position> in the template with demographic-related words, leaves the emotional word empty, and tests the probability of LLMs’ prediction for each sentiment label. Combined with the spurious correlation between *male* and *fear* in Figure 3, the fear-label tendency of OPT-6.7B in Figure 4 is nearly identical for *female* and *male*, indicating that spurious correlations are not caused entirely by the LLM parameters and **example selection contributes to spurious correlations**.

![Figure 4: Bar chart showing the native bias of LLMs over various sentiment labels (anger, fear, joy, sadness) for six models: gpt-cj-6b, opt-6.7b, opt-13b, Llama-2-7b-chat-hf, Llama-2-13b-chat-hf, and Llama-2-70b-chat-hf. The chart compares the probability of predicting 'Female' (red) and 'Male' (grey) for each sentiment label. A dashed orange line indicates the baseline probability (approx. 0.25).](e1dda754c2c88a8ad0b968aea4fc0786_img.jpg)

Figure 4 is a bar chart showing the native bias of LLMs over various sentiment labels (anger, fear, joy, sadness) for six models: gpt-cj-6b, opt-6.7b, opt-13b, Llama-2-7b-chat-hf, Llama-2-13b-chat-hf, and Llama-2-70b-chat-hf. The Y-axis represents Probability (0.0 to 0.4). The X-axis shows the sentiment labels. For each label, two bars are shown: Female (red) and Male (grey). A dashed orange line indicates the baseline probability (approx. 0.25).

- gpt-cj-6b:** anger (Female: 0.45, Male: 0.15), fear (Female: 0.25, Male: 0.15), joy (Female: 0.25, Male: 0.15), sadness (Female: 0.25, Male: 0.15).
- opt-6.7b:** anger (Female: 0.35, Male: 0.15), fear (Female: 0.25, Male: 0.15), joy (Female: 0.25, Male: 0.15), sadness (Female: 0.25, Male: 0.15).
- opt-13b:** anger (Female: 0.45, Male: 0.15), fear (Female: 0.25, Male: 0.15), joy (Female: 0.25, Male: 0.15), sadness (Female: 0.25, Male: 0.15).
- Llama-2-7b-chat-hf:** anger (Female: 0.35, Male: 0.15), fear (Female: 0.25, Male: 0.15), joy (Female: 0.25, Male: 0.15), sadness (Female: 0.25, Male: 0.15).
- Llama-2-13b-chat-hf:** anger (Female: 0.35, Male: 0.15), fear (Female: 0.25, Male: 0.15), joy (Female: 0.25, Male: 0.15), sadness (Female: 0.25, Male: 0.15).
- Llama-2-70b-chat-hf:** anger (Female: 0.45, Male: 0.15), fear (Female: 0.25, Male: 0.15), joy (Female: 0.25, Male: 0.15), sadness (Female: 0.25, Male: 0.15).

Figure 4: Bar chart showing the native bias of LLMs over various sentiment labels (anger, fear, joy, sadness) for six models: gpt-cj-6b, opt-6.7b, opt-13b, Llama-2-7b-chat-hf, Llama-2-13b-chat-hf, and Llama-2-70b-chat-hf. The chart compares the probability of predicting 'Female' (red) and 'Male' (grey) for each sentiment label. A dashed orange line indicates the baseline probability (approx. 0.25).

Figure 4: The native bias of LLMs over various sentiment labels.

## 4 REBE: REMIND WITH BIAS-AWARE EMBEDDING

To retain the accuracy and flexibility of ICL while reducing bias, we propose the ReBE, which removes spurious correlations based on contrastive learning and reminds LLMs of fairness with bias-aware embedding.

### 4.1 THE OVERVIEW OF REBE

As shown in Figure 5, using  $(x, y, s)$  as input, ReBE obtains bias-aware embedding by minimizing the bias-contrastive loss during training. Here,  $x$ ,  $y$ , and  $s$  correspond to the task’s sample, label,

{6}------------------------------------------------

and demographic attribute. With the help of prompt tuning, ReBE avoids updating the original parameters of LLM  $\mathcal{M}$ , retaining the flexibility of ICL. Besides, to effectively remove spurious correlations, contrastive learning is introduced to construct the bias-contrastive loss. The verbalizer (Cui et al., 2022) converts representations  $\{z_1, z_2, \dots, z_k\}$  to predicted labels  $\{joy, anger, \dots\}$  used in the downstream task.

![Figure 5: Overview of ReBE framework. The diagram shows the flow from input x (sentence) through a Tokenizer and Virtual Tokens to Embedding Layers. A Noise module N(mu, sigma^2) is applied to the embedding vectors. The Bias-aware Embedding module is highlighted with a red flame icon, indicating trainable parameters. The output is processed by Remaining Layers and then a Verbalizer to produce predicted labels y-hat. The Bias-contrastive Loss is calculated as alpha * L_acc + (1 - alpha) * L_bias. An example on the right shows the input sentence 'He feels happy' with virtual tokens [v1, v2], the label set y = {joy, anger, ...}, and the predicted output y-hat = arg min_{y in y} L(Logic(z, y)) = {joy, anger, ...}.](a738993919a50143787084ee7ce6e2f2_img.jpg)

Figure 5: Overview of ReBE framework. The diagram shows the flow from input x (sentence) through a Tokenizer and Virtual Tokens to Embedding Layers. A Noise module N(mu, sigma^2) is applied to the embedding vectors. The Bias-aware Embedding module is highlighted with a red flame icon, indicating trainable parameters. The output is processed by Remaining Layers and then a Verbalizer to produce predicted labels y-hat. The Bias-contrastive Loss is calculated as alpha \* L\_acc + (1 - alpha) \* L\_bias. An example on the right shows the input sentence 'He feels happy' with virtual tokens [v1, v2], the label set y = {joy, anger, ...}, and the predicted output y-hat = arg min\_{y in y} L(Logic(z, y)) = {joy, anger, ...}.

Figure 5: The overview of ReBE. The left side of the figure depicts the framework of ReBE, including the input  $(x, y, s)$  and the process of obtaining the Bias-aware Embedding. The right side of the figure is an example of inputs and output of ReBE.

### 4.2 BIAS-AWARE EMBEDDING

**Prompt Tuning** (Lester et al., 2021; Gu et al., 2022) is a soft (continuous) prompt construction and parameter-efficient tuning method for LLMs, which generally searches for the best ICL prompt in the semantic space via back-propagation. By adding virtual (pseudo) tokens to the prompt of LLMs, prompt tuning obtains trainable parameters after the embedding processing. We name trainable parameters in the prompt tuning for LLMs debiasing as **Bias-aware Embedding**. It should be noted that virtual tokens have no real meaning and only serve as placeholders. Besides, the contexts of prompt during prompt tuning are constructed based on the example selection method.

To better explain the generation of bias-aware embedding, we take the sentiment classification task in Figure 5 as an example. Represent the sentence as  $x = [v1][v2][He][feels][happy][.]$ , where  $[v_i]$  is the virtual token. After tokenization and embedding processing, bias-aware embedding becomes part of embedding vectors, which are fed into the remaining neural network layers. Representing the number of virtual tokens  $[v_i]$  as  $n_{vr}$ , and the dimension of LLM feature vectors as  $n_{feats}$ , the number of trainable parameters (bias-aware embedding) can be calculated as  $n_{vr} \times n_{feats}$ . All original parameters of LLM are frozen and are not involved in the training process described above. Since prompt tuning has been found to be unstable during training (Chen et al., 2023a), we add Gaussian noise to help the training, which is a common solution (Wu et al., 2022; Pecher et al., 2024).

Through back-propagation and gradient descent, the trainable parameters are updated to minimize the loss and obtain bias-aware embedding, which is then saved in the embedding table of LLM. According to the corresponding virtual tokens, bias-aware embedding is integrated into the embedding vectors during inference.

### 4.3 BIAS-CONTRASTIVE LOSS

Acquiring bias-aware embedding requires a well-designed loss function to guide the training. Given a training set  $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$  and its indexes set  $\mathcal{I} = \{1, 2, \dots, N\}$ , define  $z_i$  as the normalized representation of sample  $x_i$ . To better mitigate biases in the representation of LLM, we first design the bias-contrastive loss  $\mathcal{L}_{bias}$  based on SupCon (Khosla et al., 2020) loss as follows:

$$\mathcal{L}_{bias} = \frac{1}{N} \sum_{i \in \mathcal{I}} \frac{1}{|\mathcal{P}(i)|} \sum_{j \in \mathcal{P}(i)} -\log \frac{\exp(z_i \cdot z_j / \tau)}{\sum_{k \in \mathcal{A}(i)} \exp(z_i \cdot z_k / \tau)}, \quad (4)$$

{7}------------------------------------------------

where  $\mathcal{P}(i) = \{j \in \mathcal{I} : y_j = y_i, s_j \neq s_i\}$ , represents the set of indexes of examples with the same label and different demographic attribute  $s_j$  as  $s_i$ . Conversely,  $\mathcal{A}(i) = \{k \in \mathcal{I} : y_k \neq y_i, s_k = s_i\}$ , represents the set of indexes of examples with the different label and same demographic attribute as  $s_i$ .  $\tau$  is the temperature parameter of contrastive learning.

On the other hand, to retain the accuracy of ICL, we introduce the loss  $\mathcal{L}_{acc}$  based on cross-entropy loss. Following the convention, we define the  $\mathcal{L}_{acc}$  as:

$$\mathcal{L}_{acc} = \frac{1}{N} \sum_{i \in \mathcal{I}} -\log \frac{\exp(p_i)}{\sum_{y \in \mathcal{Y}} \exp(p_i^y)}, \quad (5)$$

where  $p_i$  is the probability that  $z_i$  is predicted to be the ground-truth label,  $p_i^y$  is the probability that  $z_i$  is predicted to be the label  $y$ , and label set  $\mathcal{Y} = \{joy, anger, sadness, fear\}$ .

Finally, we obtain bias-aware embedding by minimizing the weighted sum of the above two objectives:  $\mathcal{L}_{total} = \alpha \mathcal{L}_{acc} + (1 - \alpha) \mathcal{L}_{bias}$ , where  $\alpha$  is the parameter that balances the accuracy and fairness. As shown in Figure 5, the total loss  $\mathcal{L}_{total}$  is used to optimize the bias-aware embedding via back-propagation.

Table 3: Gender bias and accuracy of LLMs under example selections after debiasing.

|                   |     | Acc $\uparrow$            | AvgGF $\downarrow$        | MaxTG $\downarrow$        | MaxFG $\downarrow$        | Acc $\uparrow$            | AvgGF $\downarrow$        | MaxTG $\downarrow$        | MaxFG $\downarrow$        |
|-------------------|-----|---------------------------|---------------------------|---------------------------|---------------------------|---------------------------|---------------------------|---------------------------|---------------------------|
| <b>Random</b>     | Max | GPT-neo-2.7B              | 0.083 <sub>(+0.044)</sub> | 0.260 <sub>(+0.055)</sub> | 0.319 <sub>(+0.067)</sub> | OPT-6.7B                  | 0.086 <sub>(+0.042)</sub> | 0.322 <sub>(+0.146)</sub> | 0.447 <sub>(+0.018)</sub> |
|                   | Avg | 0.828 <sub>(+0.150)</sub> | 0.035 <sub>(-0.000)</sub> | 0.135 <sub>(-0.008)</sub> | 0.156 <sub>(-0.042)</sub> | 0.781 <sub>(-0.027)</sub> | 0.034 <sub>(-0.011)</sub> | 0.151 <sub>(-0.029)</sub> | 0.191 <sub>(-0.006)</sub> |
| <b>Perplexity</b> | Max | GPT-J-6B                  | 0.064 <sub>(+0.024)</sub> | 0.350 <sub>(+0.035)</sub> | 0.381 <sub>(+0.122)</sub> | OPT-13B                   | 0.113 <sub>(+0.013)</sub> | 0.300 <sub>(-0.021)</sub> | 0.301 <sub>(+0.157)</sub> |
|                   | Avg | 0.829 <sub>(-0.002)</sub> | 0.064 <sub>(+0.024)</sub> | 0.171 <sub>(+0.060)</sub> | 0.164 <sub>(+0.079)</sub> | 0.828 <sub>(-0.005)</sub> | 0.058 <sub>(+0.009)</sub> | 0.201 <sub>(+0.019)</sub> | 0.172 <sub>(+0.096)</sub> |
| <b>Similarity</b> | Max | GPT-neo-2.7B              | 0.053 <sub>(+0.036)</sub> | 0.267 <sub>(+0.033)</sub> | 0.167 <sub>(+0.026)</sub> | OPT-13B                   | 0.062 <sub>(+0.022)</sub> | 0.333 <sub>(+0.090)</sub> | 0.283 <sub>(+0.083)</sub> |
|                   | Avg | 0.871 <sub>(+0.024)</sub> | 0.031 <sub>(-0.003)</sub> | 0.140 <sub>(+0.047)</sub> | 0.132 <sub>(-0.032)</sub> | 0.896 <sub>(+0.024)</sub> | 0.032 <sub>(-0.012)</sub> | 0.181 <sub>(+0.028)</sub> | 0.167 <sub>(+0.006)</sub> |
| <b>DPP</b>        | Max | OPT-6.7B                  | 0.073 <sub>(+0.037)</sub> | 0.250 <sub>(+0.023)</sub> | 0.247 <sub>(+0.026)</sub> | OPT-13B                   | 0.080 <sub>(+0.045)</sub> | 0.267 <sub>(+0.117)</sub> | 0.167 <sub>(+0.217)</sub> |
|                   | Avg | 0.874 <sub>(+0.009)</sub> | 0.033 <sub>(-0.003)</sub> | 0.120 <sub>(-0.022)</sub> | 0.122 <sub>(-0.021)</sub> | 0.918 <sub>(+0.027)</sub> | 0.033 <sub>(+0.001)</sub> | 0.120 <sub>(-0.008)</sub> | 0.100 <sub>(-0.021)</sub> |

<sup>1</sup>Red subscript indicates that the metric increases after debiasing, and blue subscript indicates that the metric decreases after debiasing.

## 5 EXPERIMENTAL RESULTS

### 5.1 RESULTS AFTER DEBIASING BY ReBE

To validate the few-shot performance of ReBE, we conduct debiasing experiments on a training set of 400 samples and a test set of 200 samples, split from the *EEC-paraphrase*. According to results in Table 2, we select the two LLMs with the largest AvgGF in each baseline to eliminate the gender bias. The experimental results of race bias are available in Appendix D.2. Due to hardware limitations, we exclude OPT-30b and Llama-2-70b from the choices. We implement the ReBE based on the Huggingface PEFT library and previous work (Nguyen & Wong, 2023).

![Figure 6: Box plots comparing accuracy and gender bias (AvgGF, MaxTG, MaxFG) for GPT-neo-2.7B before and after debiasing using ReBE across four example selection baselines: Random, Rand+ReBE, Perplexity, PPL+ReBE, Similarity, Sim+ReBE, DPP, and DPP+ReBE. The top-left plot shows AvgGF (decreasing), top-right shows MaxTG (decreasing), bottom-left shows MaxFG (decreasing), and bottom-right shows Accuracy (increasing).](68aa26525c9346e4590a15c75d394e9d_img.jpg)

Figure 6: Box plots comparing accuracy and gender bias (AvgGF, MaxTG, MaxFG) for GPT-neo-2.7B before and after debiasing using ReBE across four example selection baselines: Random, Rand+ReBE, Perplexity, PPL+ReBE, Similarity, Sim+ReBE, DPP, and DPP+ReBE. The top-left plot shows AvgGF (decreasing), top-right shows MaxTG (decreasing), bottom-left shows MaxFG (decreasing), and bottom-right shows Accuracy (increasing).

Figure 6: The accuracy and gender bias comparison of GPT-neo-2.7B under four example selection baselines before and after debiasing.

As shown by the blue subscripts in Table 3, the average gender bias of most LLMs decreases after debiasing by ReBE, which works for all example selection baselines. Concerning the issue that example selection may amplify the maximum bias value, the “Max” row in Table 3 shows a significant reduction in maximum bias. In addition, Figure 6 more intuitively shows the changes in accuracy, AvgGF, MaxTG and MaxFG of GPT-neo-2.7B before and after debiasing. The variances of the

{8}------------------------------------------------

three biases all decrease, resulting in a more concentrated distribution, indicating improved stability of the bias. In addition, according to Table 3, the sentiment classification accuracy of LLMs is not significantly affected after using ReBE. The above experimental results demonstrate that **ReBE meets the requirement of reducing bias without significantly compromising the accuracy**. More importantly, the results in Table 3 and Figure 6 demonstrate that **ReBE is compatible with existing example selection methods**. By combining example selection with ReBE, it is possible to achieve high accuracy and low bias of LLMs.

### 5.2 ABLATION STUDY

To further demonstrate that the reduction in bias results from the  $\mathcal{L}_{bias}$  rather than improved accuracy, we conduct ablation studies using the  $\mathcal{L}_{acc}$  and  $\mathcal{L}_{bias}$  to replace the  $\mathcal{L}_{total}$  to train the GPT-J-6B, respectively. As shown in Table 4, the maximum values of AvgGF and MaxTG of  $\mathcal{L}_{acc}$  are much higher than those of ReBE, even though the accuracy is slightly improved. In contrast,  $\mathcal{L}_{bias}$  achieves lower bias but sacrifices accuracy. Therefore,  $\mathcal{L}_{bias}$  is actually responsible for bias reduction, and  $\mathcal{L}_{acc}$  guarantees accuracy.

Table 4: Experimental results of ablation study and parameter analysis of GPT-J-6B.

|                      | Accuracy $\uparrow$ |                           | AvgGF $\downarrow$        |                   | MaxTG $\downarrow$        |                           | MaxFG $\downarrow$        |                   |
|----------------------|---------------------|---------------------------|---------------------------|-------------------|---------------------------|---------------------------|---------------------------|-------------------|
|                      | Mean                | Min                       | Mean                      | Max               | Mean                      | Max                       | Mean                      | Max               |
| <b>Original</b>      | $0.84(\pm 1.7\%)$   | <b>0.80</b>               | $0.04(\pm 2.2\%)$         | 0.084             | $0.15(\pm 5.3\%)$         | 0.295                     | $0.17(\pm 4.9\%)$         | 0.264             |
| $\mathcal{L}_{acc}$  | $0.86(\pm 2.3\%)$   | 0.77                      | $0.03(\pm 2.3\%)$         | 0.089             | $0.13(\pm 5.2\%)$         | 0.292                     | $0.14(\pm 4.2\%)$         | <b>0.250</b>      |
| $\mathcal{L}_{bias}$ | $0.26(\pm 1.7\%)$   | 0.25                      | <b>0.02</b> $(\pm 0.9\%)$ | <b>0.049</b>      | <b>0.02</b> $(\pm 4.9\%)$ | <b>0.196</b>              | <b>0.03</b> $(\pm 7.7\%)$ | 0.327             |
| <b>ReBE</b>          | $0.84(\pm 2.2\%)$   | 0.78                      | $0.03(\pm 2.2\%)$         | <b>0.082</b>      | $0.14(\pm 3.9\%)$         | <b>0.221</b>              | $0.18(\pm 4.5\%)$         | 0.284             |
| $n$ -virtual         | $0.85(\pm 1.3\%)$   | <b>0.80</b> $(\pm 1.9\%)$ | $0.03(\pm 0.9\%)$         | $0.09(\pm 2.1\%)$ | $0.13(\pm 2.6\%)$         | <b>0.26</b> $(\pm 5.7\%)$ | $0.15(\pm 2.2\%)$         | $0.25(\pm 5.6\%)$ |
| <b>Order</b>         | $0.83(\pm 0.5\%)$   | $0.74(\pm 4.9\%)$         | $0.03(\pm 1.9\%)$         | $0.09(\pm 2.1\%)$ | $0.13(\pm 0.7\%)$         | $0.27(\pm 3.5\%)$         | $0.17(\pm 1.3\%)$         | $0.31(\pm 4.8\%)$ |

### 5.3 BASELINE COMPARISON

Regarding baseline selection, although Hu et al. (2024) proposed Fairness via Clustering Genetic (FCG) algorithm, it cannot be applied to sentiment analysis or toxicity detection because it requires explicit feature vectors for clustering. Since there are no other debiasing methods specifically for ICL, we compare ReBE with two context augmentation methods: counterfactual context and gender-balanced context. See the Appendix G for details of these two methods. As shown in Table 5, compared with the counterfactual context and gender-balanced context method, ReBE is compatible with existing example selection methods and can achieve lower bias and higher accuracy.

Table 5: Gender bias of OPT-6.7B under various example selection methods

|                 | AvgGF $\downarrow$                  |              | MaxTG $\downarrow$                  |              | MaxFG $\downarrow$                  |              | Accuracy $\uparrow$ |
|-----------------|-------------------------------------|--------------|-------------------------------------|--------------|-------------------------------------|--------------|---------------------|
|                 | Mean                                | Max          | Mean                                | Max          | Mean                                | Max          |                     |
| Random          | $0.044(\pm 0.03)$                   | 0.129        | $0.180(\pm 0.09)$                   | 0.468        | $0.199(\pm 0.09)$                   | 0.465        | 0.81                |
| DPP             | $0.036(\pm 0.03)$                   | 0.110        | $0.142(\pm 0.08)$                   | 0.273        | $0.144(\pm 0.06)$                   | 0.273        | 0.87                |
| Gender-balanced | $0.040(\pm 0.03)$                   | 0.132        | $0.174(\pm 0.08)$                   | 0.333        | $0.210(\pm 0.09)$                   | 0.417        | 0.80                |
| Counterfactual  | $0.035(\pm 0.03)$                   | 0.125        | $0.145(\pm 0.07)$                   | 0.369        | $0.149(\pm 0.07)$                   | 0.369        | 0.77                |
| Random+ReBE     | $0.034(\pm 0.02)$                   | 0.086        | $0.151(\pm 0.07)$                   | 0.322        | $0.191(\pm 0.08)$                   | 0.447        | 0.78                |
| DPP+ReBE        | <b><math>0.033(\pm 0.02)</math></b> | <b>0.073</b> | <b><math>0.120(\pm 0.05)</math></b> | <b>0.250</b> | <b><math>0.122(\pm 0.05)</math></b> | <b>0.247</b> | <b>0.87</b>         |

### 5.4 PARAMETER ANALYSIS

To illustrate the influence of parameters on ReBE, we conduct the following parameter analysis. Detailed results are available in Appendix D.3.

**$k$ -shot** refers to the number of examples in prompt of ICL. Since the coverage of examples affects the accuracy of ICL (Gupta et al., 2023), the value of  $k$  should be large enough. However, redundant information caused by excessive examples may decline the performance of ICL. As shown in Figure 7, the accuracy of LLMs after debiasing increases with the rise in  $k$ , while the biases tend to decrease initially and then increase. Therefore, considering accuracy and biases, we choose  $k = 18$  as our

{9}------------------------------------------------

![Figure 7: Two line graphs showing Accuracy (Acc) and Bias for GPT-J-6B and OPT-6.7B models across different numbers of examples (k) for few-shot learning. The x-axis represents the number of examples (k) from 2 to 26. The left y-axis represents Accuracy (Acc) from 0.75 to 0.80, and the right y-axis represents Bias from 0.1 to 0.75. Three methods are compared: AvgGF (blue line), MaxTG (orange line), and MaxFG (green line). Shaded regions around the lines represent confidence intervals. In both models, accuracy generally increases with k, while bias remains relatively stable or slightly decreases.](4e0ade2f41b66d5602160da5cc978274_img.jpg)

Figure 7: Two line graphs showing Accuracy (Acc) and Bias for GPT-J-6B and OPT-6.7B models across different numbers of examples (k) for few-shot learning. The x-axis represents the number of examples (k) from 2 to 26. The left y-axis represents Accuracy (Acc) from 0.75 to 0.80, and the right y-axis represents Bias from 0.1 to 0.75. Three methods are compared: AvgGF (blue line), MaxTG (orange line), and MaxFG (green line). Shaded regions around the lines represent confidence intervals. In both models, accuracy generally increases with k, while bias remains relatively stable or slightly decreases.

Figure 7: The accuracy and gender bias of LLM using ReBE under different  $k$ -shot.

**$n$ -virtual** is the parameter of the prompt tuning, which refers to the number of virtual prompt tokens and decides the size of trainable parameters. ReBE needs enough parameters to correct LLMs’ biases, but large  $n$ -virtual takes up more prompt space. We collect the accuracy and bias results of GPT-J-6B using ReBE under different  $n$ -virtual. According to standard deviation data in Table 4 and Figure 19 in the Appendix, there is no apparent relationship between  $n$ -virtual and bias.

**Order of Examples** Since LLMs are susceptible to position bias, previous work has found that the example order of a few-shot prompt affects the performance of ICL (Lu et al., 2022; Zhao et al., 2021). To reveal the effect of example order on ReBE, we shuffle the examples in the prompt under different random seeds. As shown in row “order” of Table 4 and Figure 19, the bias of LLM using ICL is not affected by the example order, and ReBE is also robust to changes in the example order.

## 6 RELATED WORK

After realizing that ICL performance is susceptible to example selection, many efforts have been made to stabilize it. Liu et al. (2022b) proposed the KATE, which retrieves examples semantically similar to the test query samples. After that, many heuristic-based methods have emerged, such as perplexity-based (Gonen et al., 2023; Iter et al., 2023), informativeness-based (Gupta et al., 2023; Li & Qiu, 2023) and sensitivity-based (Chen et al., 2023b). Besides that, some studies understand example selection from different perspectives, such as formulating it as a sequential decision problem (Zhang et al., 2022; Liu et al., 2024a), curating a stable subset from the original training set (Chang & Jia, 2023), selecting based on the Determinantal Point Process (DPP) (Yang et al., 2023; Ye et al., 2023) and Latent Variable Models (Wang et al., 2023). **Although these methods stabilize the accuracy of ICL on downstream tasks to a certain extent, they ignore the potential bias risks.** On the other hand, while extensive research has been conducted on the biases of LLMs (Gallegos et al., 2024), few studies focus on the bias risks of adapting LLMs to downstream tasks, especially for ICL. Although Ma et al. (2023) analyzed the predictive bias of ICL, their method relies on explicit bias attributes, making it inapplicable to the *EEC-paraphrase* dataset used in this paper. Additionally, predictive bias differs slightly from the social bias we focus on.

## 7 CONCLUSION

In this study, we have investigated the impact of example selection on the biases of LLMs. By comparing the biases under four example selection baselines with biases under zero-shot, we have found that example selection for ICL amplifies the biases of LLMs. To mitigate the bias of example selection, we have proposed the *Remind with Bias-aware Embedding* (ReBE), which removes the spurious correlations by contrastive learning and retains the feasibility of ICL by prompt tuning. After extensive experiments, we have demonstrated that ReBE can mitigate the bias without significantly compromising accuracy and is compatible with existing example selection methods. With the spread application of LLMs, more attention must be paid to the ethical risks of adapting LLMs to downstream tasks.

 Rest of paper (reference and Appendix) is removed.