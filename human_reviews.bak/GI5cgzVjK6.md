# LLM Cascade with Multi-Objective Optimal Consideration

- Decision: Reject
- Scores: 3, 3, 5, 3

## Abstract
Large Language Models (LLMs) have demonstrated exceptional capabilities in understanding and generating natural language. However, their high deployment costs often pose a barrier to practical applications, especially. Cascading local and server models offers a promising solution to this challenge. While existing studies on LLM cascades have primarily focused on the performance-cost trade-off, real-world scenarios often involve more complex requirements. This paper introduces a novel LLM Cascade strategy with Multi-Objective Optimization, enabling LLM cascades to consider additional objectives (e.g., privacy) and better align with the specific demands of real-world applications while maintaining their original cascading abilities. Extensive experiments on three benchmarks validate the effectiveness and superiority of our approach.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
The authors of this paper studied the LLM routing problem, where a large and a small LLM are deployed, and the goal is to route the input query to the large LLM only when needed. The practical value of this routing problem is to balance cost and quality, while the authors added another criterion, privacy, into this paper. 

In general, the studied problem should be relevant and useful in practice, but the proposed solution does not exhibit necessary technical novelty and the reported experiment results are not supportive. See the detailed comments below.

### Strengths
The paper provides relatively rich discussions regarding the related work, and the description of the studied problem is easy to comprehend.

### Weaknesses
The idea is rather straightforward: in addition to generating the response, the small model is also trained to predict the outcome of the considered aspects, e.g., whether the query is privacy sensitive, or whether the response is of good quality. Then the deferral module uses the prediction logits to decide if the query should be routed to the large model.

To certain degree, the title is a bit misleading: what the proposed solution addresses is not a multi-objective optimization problem per say, since the routing is actually rule based. And the training of the small LLM is also not addressing a multi-objective optimization problem, but a multi-task learning problem. 

The experiment results also look a bit weak: the call rate is very high, especially for the prompt-based and instruction tuning based solutions. And even for the best loss tuning method, more than 80% call rate seems do not really save the cost in practice. Moreover, if I understood correctly, when the query is privacy sensitive, even if the small model’s performance could be bad, we should still not send the query to the large model. 100% call rate in prompt and instruction tuning based methods seem a complete failure? Besides, all the reported the results are from the authors’ proposed methods, shall we also compare some baselines, like those discussed in the introduction and related work?

### Questions
How should we set the weights in Eq(1)? And how does the setting of them affect the performance in the reported experiments?

In Figure 3, it seems accuracy is not topped when call rate is 100%. Why is that? In this case it seems very hard to decide when the large or the small LLM would provide the better accuracy?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
In this submission, the authors propose a multi-objective method to consider more real-world requirements for LLM deployment.

### Strengths
1) It is great that the authors consider more real-world requirements for LLM deployment, such as privacy, beyond just performance-cost trade-off. The proposed method has great potentials. 

2) Authors design and conduct experiments from several aspect, analyzing the introduced privacy factor.

### Weaknesses
1) The writing of this submission can be carefully double-checked and be largely improved. 
For example, in Figure 1, the caption describe the opposite contents with subfigures, making readers confusing. And it would be better to have clear-segmented paragraphs, and each one discusses one specific thing, instead of having many different things in a single yet very long paragraph (for example, on the page 2). 
When I read the submission, I always feel that sentences close to each other lacks logic, causing difficulty in understanding the submission.

2) Latency should be discussed, and if possible, experimentally tested. 
Latency is very important for LLM deployment. It is great that authors realize more requirements should be included, and latency should be one of these requirements. Could the authors please discuss how to incorporate it into the proposed multi-objective optimization? In this way, it will be more convincing as the proposed method considers more than one factor (i.e., only privacy).

3) Experiments can be more real. 
The experiments are well designed. However, they are conducted on pseudo scenarios. It will become more solid if the authors can give some experiments for real cases as the motivation of this work is for real-world deployment.

### Questions
Please see above Weaknesses.

### Soundness
2

### Presentation
1

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
This paper presents a multi-objective LLM cascade framework. Traditional LLM cascade systems generally employ smaller, local models for initial processing, only escalating to larger server models if necessary, primarily based on performance-cost considerations. Here, the authors propose a novel approach that integrates additional objectives, such as privacy, into cascade decision-making.
The methodology involves training and training-free techniques, including prompt engineering, instruction tuning, and a custom loss-tuning mechanism that aligns with multi-objective goals. Experimental results, conducted on benchmarks like GSM8K and MedQSum, show that training-based methods outperform prompt engineering, achieving higher accuracy and safe call rates while controlling privacy leakage.

### Strengths
- The key strength of this work is its approach to incorporating multi-objective considerations, including privacy, into the LLM cascading framework. 
- The paper explores a mix of training-based and training-free techniques—like prompting, instruction tuning, and loss tuning. Practitioners can choose the methods that best fit their computational resources and deployment constraints.

### Weaknesses
* This paper builds on the work of Wang et al. (2024), which introduced cascade-aware training, i.e., a technique where the smaller model is fine-tuned with an understanding of its role within the cascade and the capabilities of the larger supporting model, achieved through knowledge distillation. Although Wang et al. (2024) is cited, a more thorough comparison between the two approaches would strengthen the analysis. Based on my understanding, this paper extends Wang et al.'s cascade-aware framework by incorporating privacy as an additional weighted component in the loss function for local training, i.e., the first summation term in Eqn (1) in the paper. 

* The technical writing could be improved for clarity. I only understood this paper’s loss-tuning approach after reading Wang et al. (2024). Clearly positioning this paper within the context of existing work would greatly aid readers in understanding its contributions.

* Appropriately assigning weights to different objectives and determining the threshold in the multi-objective cascade system is very difficult, thus, limiting its usefulness in practice. 

    *  As more objectives are added, it becomes nearly impossible to interpret trade-offs. Moving from a curve in two dimensions (performance vs cost) to higher-dimensional trade-off surfaces complicates understanding. 
    * It will be helpful to include a 3D surface plot or a contour plot that visualizes trade-offs between performance, cost, and privacy, with contour lines showing constant privacy values across the performance-cost plane. This helps illustrate how adjustments in performance and cost impact privacy.
    * The paper does not provide a concrete, practical method for tuning weights and thresholds to achieve the desired balance between objectives. The absence of guidelines means that applying this framework in practice can be a trial-and-error process, which is time-intensive and resource-consuming.

Reference
- Congchao Wang, Sean Augenstein, Keith Rush, Wittawat Jitkrittum, Harikrishna Narasimhan, Ankit Singh Rawat, Aditya Krishna Menon, and Alec Go. Cascade-aware training of language models. arXiv preprint arXiv:2406.00060, 2024

### Questions
See above

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
The paper aims to improve the existing cascading based LLM inference by adding objectives beyond accuracy. The paper starts off by making the point that existing cascading literature for the most part focuses on accuracy and uncertainty while in real world, many more objectives like privacy might be relevant. The paper proposes a simple multi objective loss function where different objectives are weighed using different coefficients. The paper shows experiments with three real world datasets showing how the privacy and task performance are balanced.

### Strengths
1. Privacy-aware inference is an important problem.
2. The paper provides extensive background on LLM cascades and quite a bit of detail on the related work.

### Weaknesses
While the paper tackles an interesting problem, it has three fundamental issues which mean it is not quite ready for publication yet.

First, the writing and presentation of the paper needs to be massively improved. The paper provides too much information when it is not needed, and worse, too little information where it is actually needed. For instance, the intro spends a lot of time on details which are irrelevant to the core contribution of the paper, e.g., the discussion of model confidence in line 71. However, in line 185, the introduction of distillation loss is very abrupt and comes without context. It is not clear to the readers how the models are being distilled, e.g., what data is being used for distillation? Up until this section, distillation was not really mentioned. This results in a situation where the reader cannot grasp the core setup of the models, e.g., training and fine-tuning details.

Second, from what this reviewer could understand, the only contribution of the paper is to turn the regular accuracy oriented (cascade-based) training into multi-objective training. Multi-objective training of ML models (e.g., the type shown in Eq. 1) is not new and has been considered in areas like fairness beforehand, see for instance, [here](https://dl.acm.org/doi/10.1145/3278721.3278779) and [here](https://arxiv.org/abs/2106.12639). So the technical novelty of the paper does not seem sufficient for ICLR.

Third, the paper could do a better job at motivating the setup considered here. It does describe what precisely the privacy concerns in the datasets considered here are. What kind of personal information does the GSM8K dataset contain? How sensitive is this information? Who are the stakeholders and what are their concerns? What is the privacy threat model? Why is the token-based leakage a good solution? The paper provides the reader to Appendix C2 for more details, which contains no examples showing privacy leakage and creates more questions than answers. For instance, what was the p-value computed on?

### Questions
1. Line 189: On “optionally” adding the distillation loss, when is it actually added?

2. Eq 1: When applying the method proposed here on a new dataset, how should one select the weight?

3. Line 141: Why does each of the objective generate a new response? Shouldn’t all objectives be covered in a single response?

### Soundness
1

### Presentation
1

### Contribution
1
