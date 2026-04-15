# FAN: Fourier Analysis Networks

- Decision: Reject
- Scores: 5, 3, 6, 3

## Abstract
Despite the remarkable success achieved by neural networks, particularly those represented by MLP and Transformer, we reveal that they exhibit potential flaws in the modeling and reasoning of periodicity, i.e., they exhibit satisfactory performance within the domain of training period, but struggle to generalize to out of the domain (OOD). The inherent cause lies in the way that they tend to memorize the periodic data rather than genuinely understand the underlying principles of periodicity. In fact, periodicity is essential to various forms of reasoning and generalization, underpinning predictability across natural and engineered systems through recurring patterns in observations. In this paper, we propose FAN, a novel network architecture based on Fourier Analysis, which empowers the ability to efficiently model and reason about periodic phenomena, meanwhile maintaining general-purpose ability. By introducing Fourier Series, periodicity is naturally integrated into the structure and computational processes of FAN. On this basis, FAN is defined following two core principles: 1) its periodicity modeling capability scales with network depth and 2) the periodicity modeling available throughout the network, thus achieving more effective expression and prediction of periodic patterns. FAN can seamlessly replace MLP in various model architectures with fewer parameters and FLOPs, becoming a promising substitute to traditional MLP. Through extensive experiments, we demonstrate the superiority of FAN in periodicity modeling tasks, and the effectiveness and generalizability of FAN across a range of real-world tasks, including symbolic formula representation, time series forecasting, language modeling, and image recognition.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
+ This paper introduces Fourier Analysis Networks (FAN), a class of learning models with an inductive bias for periodic data.

+ The authors show that FAN match the performance of KAN and MLP, two similar architectures, on a variety of symbolic formula regression tasks. 

+ The authors introduce a variety of baseline models, and they show that their method is competitive on general problems like time series forecasting and language modelling.

### Strengths
+ I think the paper is clearly written and makes a good-faith effort to benchmark, matching the design of benchmarks used in prior works like KAN. I appreciate that the authors benchmark on a wide variety of time series and language tasks. I particularly like the benchmarks on long-horizon forecasting datasets, like ETTh, which has a dominant periodic component that could potentially benefit from a model with an inductive bias for periodicity

+ The authors have included some interesting ablations and generalizations, including a Transformer architecture with FAN replacing the MLP layers, allowing them to directly process language data.

+ I very much appreciate that controlled for model size, by ensuring that all models shown in Table 2 are roughly the same size.

### Weaknesses
**Scope**

I read this paper as arguing that FAN represents a good general-purpose model for any task. The authors perform experiments showing that it exceeds the performance of KAN on general fitting tasks like fitting non-periodic symbolic functions or language modelling.
But, if this is really the case, why would this be true? What about the Fourier basis compared to the b-spline in KAN or sigmoids in MLP would allow it to perform better? I feel like this paper is missing either a theoretical justification, or a set of ablations or other experiments that probe the trained FAN models, that can help establish intuition for why this would work.

If the authors’ argument is instead that FAN has an inductive bias for periodic data, and so it should be best-in-class for a particular type of problem (like forecasting time series with mixed periodicity), then the experiments should be narrower and focus on establishing precedence on this problem. In this case, the model would need to be compared against simpler methods, including a simple Fourier transform forecast model (this is included in the darts library).

Either of the approaches above would make sense to me, but right now the paper seems to be arguing the need for a model with strong inductive biases for time series, and then doing experiments that seem to argue that this model is general-purpose,


**Claims**

+ I don’t completely buy that models like the Transformer can’t model purely periodic functions. Did the authors fairly tune the baselines here? Is this a known problem in the literature? This seems like it would be a widely-known and reported problem, which would preclude any prior progress on datasets like ETTh, and so I am skeptical of this claim. The papers the authors cite in support of this argument in the related work section ((Silvescu, 1999; Liu, 2013; Parascandolo et al., 2016; Uteuliyeva et al., 2020) only show periodic modifications to architectures; none make broader claims about periodic functions being inaccessible to learning models.

+ Why would this work on language data? The authors seem to be arguing that periodic functions are a more expressive general function class than the b-splines of KAN or linear -> activation structure of MLPs, since language itself has no periodic component.


**Novelty**

+ This is my primary justification for my score. I don’t necessarily think that modifying the architecture of KANs and MLP with a different transcendental function is conceptually novel, and the current text doesn’t provide any further mechanistic analysis of trained models 

+ For a higher score I would either need a motivating application, perhaps a SciML problem where the Fourier function class is needed, or I would need some theoretical or empirical insight into why FAN works better than other methods as a general approximator.

### Questions
+ Why hasn't the issue with periodic functions been more widely studied? Are there prior works along these lines?

+ Why do the FAN Transformers work on the language task, given how different this is from the area where Fourier approaches are normally applied?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This paper introduces FAN (Fourier Analysis Network), a new neural network architecture designed to better model periodic patterns. Unlike traditional neural networks, which often struggle to generalize periodic data and tend to memorize it instead, FAN uses Fourier Series to directly incorporate periodicity into the model’s structure. This approach allows FAN to more effectively learn and generalize periodic functions, especially in cases where the data goes beyond the training range. The authors show that FAN can replace multi-layer perceptrons (MLPs) in many tasks, achieving similar or better results with fewer parameters and computational needs. Experiments demonstrate FAN’s strengths across tasks like time series forecasting, symbolic formula representation, and language modeling. FAN outperforms models such as MLPs, KAN, and Transformers, highlighting its effectiveness in both periodic and non-periodic contexts. Additionally, FAN achieves competitive or superior results in real-world applications, showing potential as a foundational model component.

### Strengths
The paper is well-written, and I was able to quickly understand both the core concepts and the practical implementation. The numerous graphs effectively illustrate the methodology and results, enhancing the clarity and impact of the work.

### Weaknesses
1.	When discussing the advantages of predicting Fourier coefficients, consider citing the theorem that emphasizes the significance of Fourier coefficient singularity. Additionally, referencing the foundational property that smooth functions can often be approximated well with a finite number of Fourier coefficients would strengthen the argument that predicting the first ￼ coefficients is typically sufficient.
	2.	The syntax and structure are somewhat confusing. Specifically, the distinction between ￼ and ￼ is unclear upon initial reading and could benefit from a more explicit explanation.
	3.	In demonstrating results on known functions, it is worth noting that functions with a finite number of Fourier coefficients (finite in the frequency domain but infinite in the time domain) will predictably perform better when analyzed in the frequency domain. Consequently, comparing this to models that do not use the frequency domain is not particularly informative. Instead, consider comparing with frequency-based models such as FRETS, FEDformer, and other similar approaches. Furthermore, adding examples with different types of functions would provide a more comprehensive evaluation, such as:
	•	Periodic functions that include both low and high frequencies (e.g., ￼).
	•	Non-periodic functions.
	4.	When discussing other real-world datasets, the choice of datasets could be expanded. The omission of non-periodic datasets, such as exchange rates and electricity usage, is particularly noticeable, and including them would enhance the comprehensiveness of your analysis.
	5.	In the results section, comparing your model solely with non-frequency models may be less meaningful, as the use of FFT in time series prediction is not a novel approach, and several previous models have integrated FFT with MLPs. A more relevant comparison would be to other FFT-based models to effectively demonstrate the advantage of your approach, particularly in terms of the integration between FFT coefficients and linear modeling.
	6.	If you are presenting the FLOP count, it would be beneficial to include a runtime graph. My intuition suggests that the MLP model may have a higher runtime, and a runtime graph would provide a clearer perspective on computational efficiency.
	7.	Including a link to the code within the abstract would be valuable for readers interested in exploring the implementation.

### Questions
1.	Clarify the distinction between your approach and directly learning the coefficients.
	•	Suggestion: Conduct an ablation study to quantify the improvement in results when incorporating the linear component.
	2.	Explain the effect of the parameter ￼ on model performance.
	•	Recommendation: Provide details on how ￼ influences the results.

### Soundness
1

### Presentation
2

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
This paper proposes a new network architecture FAN based on Fourier analysis, which improves the ability of efficient modeling and reasoning of periodic phenomena. By introducing Fourier series, periodicity is naturally integrated into the structure and calculation process of neural network, and more accurate expression and prediction of periodicity pattern are realized. FAN can seamlessly replace MLP in a variety of models while using fewer parameters and FLOPs. Through numerous experiments, the paper demonstrates the effectiveness of FAN in modeling and reasoning periodic functions, as well as its superiority and generalization ability for a range of real world tasks including symbolic formula representation, time series prediction, and language modeling.

### Strengths
1. This paper cleverly integrates Fourier analysis into neural network architecture, including both periodic and aperiodic parts, which can seamlessly replace MLP layer.
2. FAN uses fewer parameters and FLOPs than traditional MLPS and is more computationally efficient.
3. This model is more interpretable than the traditional black box model, such as MLP.

### Weaknesses
1. FLOPs analysis is given in this paper, but the comparison of actual training time is lacking
2. Will the result of a deeper FAN yield a better fit? Will overfitting results be produced? It is recommended to test the performance changes of layers 3-20 at different depths
3. This article compares performance with traditional MLP, KAN and transformer. Whether to compare and discuss with deep learning frameworks that actually deal with time series data, such as LSTM, RNN

### Questions
1. The paper lacks theoretical justification for why FAN enhances generalization capability; 
2. They don't explain how FAN avoids overfitting, such as through some model selection methods.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
The paper addresses the critical challenge of periodicity modeling in neural networks, proposing the Fourier Analysis Network (FAN) to overcome inherent limitations of MLPs and Transformers in handling periodic data. The attempt to leverage Fourier Series for improving periodicity modeling is a novel approach, and integrating this into neural network design could potentially enhance tasks like time series forecasting and symbolic formula representation.

### Strengths
The paper proposes a novel approach based on Fourier Analysis, which is a well-established mathematical tool for modeling periodic functions. The integration of Fourier Series into the network design could potentially address the limitations of traditional architectures like MLP and Transformer in periodicity modeling.

The writing is generally clear and the figures, such as Figure 1 and others, are well-presented. They help in visualizing the performance of FAN relative to other models.

The experiments cover a variety of tasks, including symbolic formula representation, time series forecasting, and language modeling, which demonstrates the general applicability of the proposed architecture.

### Weaknesses
Insufficient Theoretical Justification:
The paper lacks a rigorous theoretical foundation that convincingly demonstrates the superiority of FAN over existing methods. While the integration of Fourier Series is discussed, the connection to the universal approximation theorem and how FAN mitigates spectral bias needs clearer elaboration.

Inadequate Experimental Validation:
The experiments provided are not sufficient to substantiate the claims. The datasets used for evaluation, especially in time series forecasting and symbolic formula representation, do not comprehensively cover real-world complexities. Additionally, key experimental details such as hyperparameter tuning, model initialization, and evaluation metrics are missing or inadequately described.

Limited Comparisons with Baselines:
The paper claims FAN outperforms MLPs, KANs, and Transformers, yet the comparisons lack depth. There is insufficient analysis of why FAN performs better, and the results for non-periodic tasks suggest that FAN’s benefits might be overstated.  It lacks comparisons with other neural networks that also leverage Fourier analysis. The absence of this comparison makes it difficult to fully evaluate the effectiveness of FAN, and this omission raises questions about the extent of its claimed advantages.

Unclear Practical Utility:
While the theoretical motivation is apparent, the practical implications are less clear. The paper does not provide compelling use cases or applications where FAN’s periodicity modeling significantly outperforms traditional methods in practical scenarios.

### Questions
1. Lack of detailed explanation about parameter efficiency:
The abstract claims that FAN can substitute MLP with fewer parameters. However, the theoretical justification and experimental evidence supporting this claim are not sufficiently detailed in the main text. It would be beneficial to include a comprehensive analysis comparing the parameter count and FLOPs of FAN versus MLP in various tasks.

2. Doubtful results in symbolic formula representation:
In Figure 5, the performance of KAN (Liu et al., 2024) was not replicated accurately. As KAN has shown strong results in symbolic formula representation in previous works, the omission of this accuracy raises concerns. I suggest the authors provide a more in-depth discussion of why FAN outperforms KAN in their experiments, or at least explain the differences in experimental settings that may have led to this discrepancy.

3. Practical Use Cases:
Can the practical utility of Fourier Approximation Networks (FAN) be demonstrated with real-world applications beyond synthetic and controlled experiments? How does FAN improve performance on industry-relevant tasks?

4. Clarification of Contributions:
What aspects of FAN are novel compared to existing Fourier-based neural networks? Can the authors clarify potential overlaps with prior work and better situate FAN within the context of ongoing research?

### Soundness
2

### Presentation
2

### Contribution
1
