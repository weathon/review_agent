# LoCA: Location-Aware Cosine Adaptation for Parameter-Efficient Fine-Tuning

- Decision: Accept (Poster)
- Scores: 6, 6, 5, 6, 6, 6

## Abstract
Low-rank adaptation (LoRA) has become a prevalent method for adapting pre-trained large language models to downstream tasks. However, the simple low-rank decomposition form may constrain the optimization flexibility. To address this limitation, we introduce Location-aware Cosine Adaptation (LoCA), a novel frequency-domain parameter-efficient fine-tuning method based on inverse Discrete Cosine Transform (iDCT) with selective locations of learnable components. We begin with a comprehensive theoretical comparison between frequency-domain and low-rank decompositions for fine-tuning pre-trained large models. Our analysis reveals that frequency-domain decomposition with carefully selected frequency components can surpass the expressivity of traditional low-rank-based methods. Furthermore, we demonstrate that iDCT offers a more efficient implementation compared to inverse Discrete Fourier Transform (iDFT), allowing for better selection and tuning of frequency components while maintaining equivalent expressivity to the optimal iDFT-based adaptation. By employing finite-difference approximation to estimate gradients for discrete locations of learnable coefficients on the DCT spectrum, LoCA dynamically selects the most informative frequency components during training. Experiments on diverse language and vision fine-tuning tasks demonstrate that LoCA offers enhanced parameter efficiency while maintains computational feasibility comparable to low-rank-based methods.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes LoCA, a method for fine-tuning pre-trained models using frequency-domain adaptation via the inverse Discrete Cosine Transform (iDCT). LoCA focuses on selecting key frequency components to improve the expressivity and efficiency of model adaptation. The theoretical analysis argues that iDCT-based adaptation can match or exceed the effectiveness of low-rank methods. However, the empirical gains over existing methods like LoRA are marginal, especially in vision tasks. LoCA’s added complexity, due to finite-difference approximations and alternating optimization, may not be fully justified by these modest improvements, potentially limiting its practical appeal.

### Strengths
This paper introduces LoCA, a frequency-based approach to parameter-efficient fine-tuning that selectively optimizes specific frequency components using the Discrete Cosine Transform. By focusing on significant frequencies within weight matrices, LoCA aims to reduce the parameter count needed for fine-tuning while maintaining model expressivity. This selective frequency adaptation presents a practical alternative to spatial-domain methods like LoRA, providing a new angle on efficient model tuning. The paper’s theoretical framework, including empirical spectral density and the Central Limit Theorem to analyze expressivity, helps ground LoCA's approach in established statistical methods, adding quality to the work.

### Weaknesses
●  This paper appears to provide inadequate empirical support for its theoretical claims. A central claim of the paper is that randomly selected frequencies for FourierFT yield lower expressivity than LoRA; however, this claim lacks direct experimental validation, which is critical to substantiate the theoretical conclusions. For instance, Figure 3 shows mixed results for FourierFT on the FGVC task, while Figure 4 in *Parameter-Efficient Fine-Tuning with Discrete Fourier Transform* by Gao et al. (2024) (arXiv:2405.03003) presents empirical evidence that contradicts this claim by showing that FourierFT achieves higher accuracy than LoRA across multiple GLUE benchmarks. Additionally, Section C.2 on Expressive Ability in the FourierFT paper’s supplementary material further supports FourierFT’s superior expressivity. The paper also lacks empirical evaluations for selective FourierFT and LoCA, which would further validate the claims made.

●  The paper omits a comparison with a highly representative spatial-domain PEFT method, VeRA, which focuses on lightweight adaptations in the spatial domain and would serve as a useful benchmark for LoCA's performance.

●  The design of LoCA introduces significant parameter overhead due to the individual optimization of frequency component locations and coefficients for each layer. For example, in a model with \( L = 32 \) layers (e.g., LLaMA-2 7B), LoCA’s parameter count is approximately 2.82 times that of FourierFT, raising concerns about the scalability and efficiency of LoCA for large-scale models.

●  The theoretical framework assumes asymptotic normality of weight updates, enabling the use of the Central Limit Theorem and empirical spectral density for analyzing expressivity. However, this assumption relies on i.i.d. updates, which may not hold in the context of gradient-driven, correlated weight adjustments inherent in LoRA and LoCA. Given the limited and targeted nature of LoCA’s updates, the cumulative adjustments may lack the “sum of many independent adjustments” necessary for CLT to apply reliably. This assumption weakens the robustness of the theoretical claims, as the actual distribution of weight updates is likely far from normal in practical implementations. 

●  LoCA’s dynamic selection of high-magnitude frequency components across epochs may introduce instability during convergence, as the selection of significant frequencies may shift due to changing gradients. This could impact the model’s ability to achieve stable and consistent updates over time. Furthermore, by focusing solely on high-magnitude frequencies, LoCA risks omitting task-relevant information in lower-magnitude components, potentially limiting its adaptability in tasks requiring finer-grained details.

●  The method also relies on finite-difference approximation to estimate location gradients, which introduces additional computational and memory costs. This overhead may significantly increase CUDA memory requirements, particularly in high-dimensional models or when frequent updates are necessary.

### Questions
● Q1: LoRA may not be the most parameter-efficient approach among spatial-domain PEFT methods. The work by Kopiczko et al. (2023) in "VeRA: Vector-based Random Matrix Adaptation" (arXiv:2310.11454) demonstrates a more parameter-efficient and lightweight alternative, focusing on diagonal matrix adaptations to achieve efficient adaptation without the need for frequency-based transformations. Could the authors clarify whether their proof and theoretical framework apply to VeRA? Additionally, this paper lacks a comparative analysis of VeRA, both theoretically and experimentally. Would the established proof also support an evaluation of DoRA’s expressivity?

● Q2: For each layer in LoCA, both frequency component locations and coefficients are optimized individually. This approach appears to introduce a higher number of parameters compared to FourierFT, which selects $n$ locations randomly and shares these locations across all layers. Specifically, FourierFT’s parameter count is:

$2n + n* L = n (L + 2)$

where $L$ represents the number of layers in the pre-trained model.

In contrast, LoCA introduces $2n$ parameters for each layer’s and locations,  and $n$ for each layer’s coefficients, resulting in a total parameter count of:

$3n \times L$

This yields a parameter ratio between LoCA and FourierFT of:

$\frac{3L}{L + 2}$


For example, with LLaMA-2 7B where $L = 32$, LoCA’s parameter count is approximately 2.82 times that of FourierFT. This raises concerns about parameter efficiency, especially in large models. To clarify whether the additional parameters in LoCA yield proportional benefits, could the authors provide empirical comparisons across various model sizes and tasks, measuring both fine-tuning performance and resource usage (e.g., memory and compute requirements)? Specific metrics, such as performance improvements relative to parameter increase and scaling efficiency on different benchmarks, would help assess whether gains in expressivity or accuracy justify the increased parameter cost.

● Q3: In lines 191-199, the authors claim that randomly selecting frequencies for FourierFT yields the lowest expressivity, performing worse than LoRA; however, this claim lacks experimental support. For instance, Figure 3 in this paper shows mixed results for FourierFT on the FGVC task, whereas Figure 4 in Gao et al. (2024), *"Parameter-Efficient Fine-Tuning with Discrete Fourier Transform"* (arXiv:2405.03003), presents contrasting findings, particularly on the QQP task in GLUE. In Gao et al., FourierFT consistently outperforms LoRA across GLUE benchmarks, achieving higher accuracy with minimal random spectrum updates and fixed locations across layers. Furthermore, Section C.2 on Expressive Ability in the FourierFT paper’s supplementary material reinforces FourierFT’s superior expressivity over LoRA. Could the authors provide empirical comparisons to clarify these discrepancies, ideally across multiple model sizes and tasks, with metrics on fine-tuning performance and resource usage (e.g., memory and computational requirements)? Demonstrating whether the increased parameter count in LoCA yields proportional performance benefits would strengthen the case for its efficiency.

● Q4: Additionally, the paper lacks empirical evaluations comparing selective FourierFT and LoCA, which would be valuable in validating the theoretical claims. For instance, in line 191, the statement that $W_F(3)$ can outperform \$W_F(2)$ would benefit from empirical results to illustrate how these specific configurations impact performance. Further analyses using different selection strategies within FourierFT would also help substantiate the expressivity claims and clarify the mixed findings observed.

● Q5: The proof assumes asymptotic normality of incremental weight updates, enabling statistical analysis of expressivity via the Central Limit Theorem and empirical spectral density. However, in LoRA, only a subset of weights is updated through low-rank reparameterization, while frequency-based methods like LoCA further restrict updates to high-amplitude frequency components. Given that these updates are gradient-driven and thus correlated, the i.i.d. assumption essential for CLT may not strictly hold. With limited, targeted updates, the cumulative effect lacks the "sum of many independent adjustments" necessary to ensure asymptotic normality. Could the authors provide further justification for assuming convergence to normality under selective updating, and clarify how potential deviations from i.i.d. behavior may impact expressivity comparisons?  It would be helpful if the authors could conduct specific analyses or empirical tests, such as quantifying deviations from normality in the weight updates or performing sensitivity analyses to assess the impact of non-normality on expressivity. 

● Q6: In the alternating optimization strategy, the method first optimizes the coefficients of selected frequency components $\alpha$, before refining their locations for $B_a$ steps. Then, with $\alpha$ fixed, it optimizes the locations $l$ for $B_l$ steps, and finally, the procedure fixes the locations and optimizes only the coefficients $\alpha$ until convergence.

Could the authors clarify the rationale behind this specific order of coefficient-first optimization and its impact on stability and convergence? While this separate optimization approach might simplify the process, it may not fully capture the interactions between coefficients and locations, potentially limiting optimality. Have the authors explored an alternative order—optimizing locations first and then coefficients—and could they provide insights on how this might affect convergence and final performance?

In the ablation study (lines 480-485), the authors present several variant comparisons, yet they do not include an analysis of this alternative pipeline. Additionally, how are the parameters $B_l$ and $B_s$ selected—is their choice task-specific? From Table 4, it appears that the V5 variant achieves relatively better results, but this is not consistent with the description of the alternative policy in lines 284-292 and the algorithm in lines 945-963. Could the authors clarify these inconsistencies and provide further justification for the selected optimization order and parameter settings? 


● Q7: Could the authors clarify how LoCA ensures stable convergence given the dynamic selection of specific magnitude(e.g., high-magnitude) frequency components in $\Delta W$ across epochs? Specifically, as top-ranked frequencies may shift due to gradient changes, how does LoCA maintain consistency in updates to avoid potential instability in the training process? Additionally, could the authors explain how the specific frequency components selected in LoCA—whether high or low frequencies—consistently contribute to model performance across tasks? Is there a risk that focusing solely on high-magnitude components could lead to loss of task-relevant information in lower-magnitude frequencies, which may carry finer-grained details?


● Q8: Could the authors clarify the computational and memory overhead associated with estimating location gradients using finite-difference approximation? Specifically, does this approach increase CUDA memory requirements significantly, and if so, how does it impact the overall efficiency of LoCA? Additionally, an analysis of the trade-offs between accuracy and resource usage in this approximation method would be valuable to understand its practical feasibility.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper introduces Location-aware Cosine Adaptation (LoCA), a novel method for fine-tuning large language models (LLMs) and vision models in a parameter-efficient manner. LoCA is based on the inverse Discrete Cosine Transform (iDCT) and optimizes the locations of learnable frequency components. It addresses limitations of previous low-rank adaptation methods by providing greater optimization flexibility and expressivity. Theoretical analysis and empirical observations support the superiority of LoCA over traditional low-rank methods and iDFT-based methods. LoCA dynamically selects the most informative frequency components during training, leading to enhanced parameter efficiency and computational feasibility. The method demonstrates state-of-the-art performance on diverse language and vision tasks with fewer parameters. The introduction of the paper contextualizes the need for parameter-efficient fine-tuning methods due to the prohibitive costs of fully fine-tuning increasingly large models, and LoCA is presented as a solution that maintains performance while reducing trainable parameters.

### Strengths
1. The concept of applying low-rank adaptation within the Fourier domain is intriguing, and it implicitly suggests a method of tuning that utilizes all available parameters.

2. The theoretical results appear to be novel and have captured the interest of the reviewer.

3. The proposed method delivers strong performance benefits while maintaining an exceptionally low parameter cost.

### Weaknesses
The reviewer, not being an expert in this area, has not identified any major weaknesses. However, with some background in empirically tuning LLMs and ViTs, the reviewer would like to inquire further about the experimental setup.

1. There lack some benchmarks and baselines. 

2. Common advantages of the PEFT method include reduced computation and memory costs. The paper's contribution would be strengthened if the authors included these aspects in their analysis.

### Questions
I will keep my positive score if the authors address Question 1. Other questions require much more experiment time and are quite minor to improve the paper.

1. MT-bench is considered an unstable benchmark. It is strongly recommended that the authors utilize the MathInstruct Dataset instead, which is more stable and generally requires a higher level of expressive power.

2. For fine-tuning Roberta, typical benchmarks include RTE, BoolQ, SST-2, WSC, WIC, MultiRC, SQuAD, CB, COPA, DROP, GSM8K, and ReCoRD. Could the authors consider adding any benchmarks that are currently missing?

3. COLA, ReLoRA, and DoRA represent typical LoRA variants. It would be beneficial if the authors could include any of these variants that are not already covered.

4. In Figure 3, it appears that the performance gain may continue to increase with a larger value of 'r.' Could the authors extend the range of 'r' to determine the optimal value that yields the best performance?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
2

### Summary
The paper titled "LoCA: Location-Aware Cosine Adaptation for Parameter-Efficient Fine-Tuning" introduces a novel parameter-efficient fine-tuning (PEFT) method called Location-Aware Cosine Adaptation (LoCA). This method is designed to adapt pre-trained large language models (LLMs) to downstream tasks with improved optimization flexibility and parameter efficiency. LoCA is based on the inverse Discrete Cosine Transform (iDCT) and selectively tunes learnable components at specific locations

### Strengths
1. LoCA introduces a novel approach for parameter-efficient fine-tuning in the frequency domain through the inverse Discrete Cosine Transform (iDCT) and selective learning of frequency components. This method demonstrates the potential to surpass traditional low-rank decomposition techniques both theoretically and empirically, which is of significant value for resource-constrained environments and the deployment of large-scale models. Furthermore, your work provides a comprehensive theoretical analysis comparing frequency domain methods with low-rank decomposition approaches, which is meaningful.
2. The methodology section of the paper is rigorous, and the experiments cover multiple domains, including natural language processing and computer vision. The paper offers comparisons with existing techniques, such as LoRA and FourierFT, which help readers understand the performance and efficiency of LoCA. Additionally, the in-depth theoretical analysis provides a solid foundation for frequency domain parameter-efficient fine-tuning methods.
3. The writing of the paper is clear and logically structured, with a coherent flow from the introduction to the methodology, experimental results, and conclusions. In particular, the detailed explanation of how LoCA operates, including the application of the inverse Discrete Cosine Transform and the alternating optimization strategy, enhances the reader's understanding of the relevant work.

### Weaknesses
1. The paper contains a limited amount of content related to RELATED WORK, with insufficient coverage of the existing literature in the field.
2. While the experimental results are convincing, the paper could further expand the experimental section to include the verification of LoCA's performance on more datasets. Additionally, a more in-depth analysis of LoCA's performance on different model scales and tasks of varying complexity would help to further demonstrate its applicability and robustness.

### Questions
1.The paper primarily compares LoCA with LoRA-related fine-tuning techniques. Has consideration been given to performance comparisons with other fine-tuning methods such as prompt learning and adapter tuning?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces a novel parameter-efficient fine-tuning method, Location-Aware Cosine Adaptation (LoCA), that leverages inverse Discrete Cosine Transform (iDCT) for selectively optimizing frequency-domain components in pre-trained language and vision models. LoCA aims to surpass traditional low-rank adaptations by dynamically choosing informative frequency components, thus balancing parameter efficiency and performance.

### Strengths
1. The paper provides a rigorous theoretical comparison between frequency-domain and low-rank adaptation methods, filling a gap in the literature on expressivity and optimization constraints.
2.  LoCA’s use of iDCT with dynamic selection of frequency components represents a creative improvement over conventional low-rank methods, particularly for parameter efficiency.

### Weaknesses
Overall it's a good paper, and I will raise my score if the authors could address my concerns.
1. LoCA introduces a computationally complex process with alternating optimization steps and central difference approximation, which could pose practical challenges.
2. How does LoCA handle potential noise in frequency component selection, and are there measures in place to stabilize the optimization process?

### Questions
See weakness.

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
3

### Summary
The paper introduces Location-aware Cosine Adaptation (LoCA), a novel frequency-domain parameter-efficient fine-tuning method for pre-trained LLMs. By leveraging the inverse Discrete Cosine Transform (iDCT) as well as selectively learning components in the frequency domain, LoCA addresses the constraints of the naive low-rank adaptation (LoRA) method. 
In a word, LoCA enhances expressiveness while maintaining computational efficiency.

### Strengths
1.As emphasized by the authors, their iDFT-based variants has managed to outperform  the expressivity of previous low-rank-based methods.

2.Overall, the presentation is clear, supported by rigorous mathematical derivations and extensive experimental results.

### Weaknesses
1.Some baseline experimental results differ significantly from those in related papers, which may indicate carelessness in the experimental process. Also, more ablation experiments are needed to increase confidence.

2.For most datasets, LoCA doesn't show a clear advantage over FourierFT in terms of reducing parameter budget and improving accuracy.

Please see the questions section for more details.

### Questions
1.Why are the accuracy rates of the baseline methods on the Stanford Cars and FGVC datasets more than 5% higher than those reported in related papers? I mainly compared the experimental results from the FourierFT paper(https://arxiv.org/pdf/2405.03003) and yours, and found that the differences are small on other datasets, but the results on the Stanford Cars and FGVC datasets are significantly beyond normal error margins. I am unsure whether this is due to errors caused by carelessness in the experimental process, or if you used different ViT models compared to theirs. Specifically, the experimental results on the Stanford Cars and FGVC datasets are emphasized in your work, and it is crucial to ensure the precision of these results.

2.Why are there so few ablation experiments for FourierFT fine-tuning on ViT? As the most competitive counterpart, additional experimental results for FourierFT 239K and FourierFT 480K after fine-tuning on ViT could be included. After all, LoCA presents results for two different parameter budgets, while FourierFT only provides results for the smallest parameter budget for comparison, which does not meet the fairness requirements of an ablation study.

3.What are the differences between LoCA and other methods in terms of Memory Cost and Training Time? You may use charts to illustrate these differences explicitly.

4.Why does LoCA not show a significant advantage over FourierFT , on fine-tuning various foundation models such as RoBERTa, LLaMA, and ViT, in terms of reducing parameter budget and improving accuracy? Does this suggest that, while your work is strongly interpretable, it may have limited practical value?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 6

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes a novel low-rank adaptation approach for fine-tuning transformer-based pretrained models by deriving weights from parameters learned in the frequency domain using the iDCT operation. Compared to the similar existing method FourierFT, the approach, in theory, promises better reconstruction of the oracle update matrix. Empirically, the results do improve over the baseline FourierFT approach in most cases, indicating effectiveness of the approach. The paper also provides a method to learn locations in the frequency domain where coefficients are required, which is novel and interesting in the context of PEFT.

### Strengths
+ The paper takes an existing idea - learning PEFT parameters in the frequency domain and reconstructing the weight matrix from those learned parameters - and performs an in-depth analysis of the approach. Based on the obtained insights, it presents a new method that is theoretically better than existing approaches in approximating the ideal fine-tuning matrix and shows quantitative improvements on the base method, FourierFT, in most cases. 
+ The paper includes intuitive theoretical statements, backed by mathematical proofs, which is good to see in a PEFT paper since existing methods often lack theoretical insights and are heuristic-based. 
+ The paper is also well written and intuitive to follow, with rigorous experiments. 
+ I especially liked the approach presented for learning discrete locations to be optimized, which I believe is  a novel contribution.

### Weaknesses
- The paper starts with some terminology that are not elaborated: “optimization space” and “flexible optimization”. These terms are not defined precisely anywhere, nor is their link to the theory or empirical results clear. It would be better to ground the explanation to well-defined terms that are used in the analysis.

- According to the reference (Yohai & Maronna, 1979), the initial assumption, the equation in Sec 2, L99, is true only under certain conditions. 
This variant of the equation holds true only when $\psi$ is monotone and X to have full rank. However, this issue is not addressed anywhere in the paper. For eg., the matrix X according to the paper is the input matrix. In practical implementation, the dimension X is m*n where (m<n), i.e. for X to be full rank, we need rank(X) = m. This is not stated or supported in the paper.

- The theory presented is highly domain-specific: it does not translate to more general PEFT methods such as VeRA or DoRA, and requires significant theoretical adaptations to allow for comparisons with arbitrary low-rank methods. 

- The theory also does not *always* agree with practice - there are certain cases where LoRA and FourierFT perform better. This indicates come confounding factors, yet no discussion on this has been included. I do not see this as a reason to reject however, as this is commonly seen in this area, but would appreciate a discussion on the same.

- In the case of ViT, I would have liked to see comparisons of the proposed approach with FourierFT having the same number of trainable parameters, as done for NLU and IFT tasks.

### Questions
Please see the weaknesses above. A few additional questions are below:

* Could a discussion on certain edge cases where the theory does not hold be provided? More precisely, would it be possible to find situations where the assumptions made in theory do not hold well, resulting in a breakdown of expected results?

* Additionally, could results for ViT be provided in situations where the number of parameters are the same as parameters in FourierFT, for a more intuitive comparison?

* I would also like to see results in the Natural Language Generation task - particularly for the GPT-2 training performed by FourierFT, as it would indicate the effectiveness of the method when ported to a Conv1D based implementation of the MLP.

### Soundness
3

### Presentation
2

### Contribution
2
