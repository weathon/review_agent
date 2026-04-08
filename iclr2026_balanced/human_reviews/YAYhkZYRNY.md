## Human Reviewer 1

### Summary
This paper studies gender and race biases in generative VLMs through both models’ responses and confidence levels. The authors use multiple choice selection (MCS) to evaluate four open models and show they present biases in two datasets (PAIRS and SCF). In particular, the authors show that bias fluctuates considerably throughout the hidden layers. They then propose two methods to mitigate gender and race bias, which reduces biases in the base models.

While the proposed methods seem promising, I believe the paper needs some writing improvements as well as comparisons with existing debiasing approaches before being published.

### Strengths
1. A study of race and gender bias in generative VLMs through the lens of confidence levels
2. Two methods proposed to mitigate bias in these VLMs that reduce bias in the base models
3.  An interesting analysis of the bias level across different layers.

### Weaknesses
1. The Introduction section could be improved through better contextualization of the work and task setup, and how it differs from the existing setups. Topics like the ones in L66 and in L73-74 could be further expanded to help the reader better understand.
2. A lot of space is used to describe system prompts, which are also claimed to be one of the main contributions of the paper by the authors. In my opinion, while important, the system prompts are not substantially different from existing prompts, and could simply be reported in Appendix, leaving space for a more thorough discussion of the task and setup.
3. In L139-140, it would be great to have an intuition of how / why humans corrected the labels of PAIRS, and whether this is a process that the community should adopt.
4. The description of the post-hoc mitigation method (Section 3.1) could be improved. For instance, the second subscript of delta moves from the residual number (1 or 2) to the fairness label (fair or biased), which is confusing when first reading. The assumption behind the approach (defined in Appendix C) is also very useful in my opinion, and it should be part of the main paper. The same applies to Figure 6.
5. The experiments show how the proposed approaches improve bias issues of the base models, but there is no comparison with other bias mitigation methods (such as those discussed in Section 6) that would help the community understand which approach is more promising.

### Questions
1. Several typos in the paper: “Gender and” repeated in title; “gender” not capitalized in abstract and L382; abbreviations like “there’s” in L211; missing links to Appendices in L284-325-373; character in Fairness equation (L253); most citations don’t include a surrounding parenthesis (\cite instead of \citep).
2. The paper format is different from the ICLR template, which could be ground for desk rejection.
3. The plots in Figure 4 are too small. The authors should at least remove the redundant information in the y-axis of plots b, c and d.
4. Why do you sample one template at random for each concept (L145) rather than evaluating models on 10 templates and averaging the results for robustness?

### Soundness
3

### Presentation
2

### Contribution
2

### Rating
2

### Confidence
3

---

## Human Reviewer 2

### Summary
This paper tackles the problem of gender and racial biases in the predictions of VLMs. There are several presentation issues that make this paper not ready for peer review.

### Strengths
None.

### Weaknesses
This paper does not appear to be ready for review. 

* Each page is only ~46 lines instead of ~55. Something went wrong with the stylesheet.
* The abstract starts with an uncaptialized letter
* The title contains a repeated n-gram: "Mitigating Gender and Gender and Race Bias"
* The paper seems to only uses one style of citation \cite instead of \citep, which leads to a very unnatural visual style.

### Questions
None

### Soundness
1

### Presentation
1

### Contribution
1

### Rating
0

### Confidence
5

---

## Human Reviewer 3

### Summary
The authors investigate and reveal gender and race biases in the responses of 3 SOTA VLMs by using the MCS method to analyze their outputs and by inspecting their internal layers, measuring the confidence level of each token in the response while proposing a new post-hoc
method to mitigate this issue that can be applied during the inference phase, where the method computes the mean of the residual bias vectors and fair vectors and then makes an orthogonal projection that is used as the new representation vector.

The paper appears to not have been proofread.

### Strengths
Post-hoc mitigation method is quite good and makes sense, besides of being simple to understand, apply or even adapt it. Opening the layers and investigate the confidence level to show that sometimes the model may seem fair when in reality the layers show they are not is simple and yet necessary to understand the theme.
However I cant find enough contribution and the paper is poorly proofread

### Weaknesses
Although you explain that PAIRS does not provide race data, which you frame as a limitation, the title highlights this issue, while most of the text focuses much more on gender bias than on racial bias. I also had the impression that the main references in the introduction do not vary much in terms of methodological approaches or datasets. Finally, and no less importantly, I understand that the authors aim to help mitigate these issues, but I found it interesting that there was no mention that the input data the models are trained on is the root cause. This could have been explicitly connected to the proposed mitigation approach at inference time.

This paper is clearly not ready for submission to ICLR. I recomend that the authors perform an extensive proofreading, review the template and consider another venue for submiting their work.

### Questions
Why repeat the word gender on the title?
Why not post the code in an anonymization platform such as Anonymous GitHub - 4open.science?
What is the main contribution of the paper?
What is the novelty in your method?

### Soundness
1

### Presentation
1

### Contribution
1

### Rating
0

### Confidence
4