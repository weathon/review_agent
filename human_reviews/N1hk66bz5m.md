# Query and Response Augmentation Cannot Help Out-of-domain Math Reasoning Generalization

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 5, 3, 5

## Abstract
In math reasoning with large language models (LLMs), fine-tuning data augmentation by query evolution and diverse reasoning paths is empirically verified effective, profoundly narrowing the gap between open-sourced LLMs and cutting-edge proprietary LLMs. 
In this paper, we conduct an investigation for such data augmentation in math reasoning and are intended to answer: 
(1) What strategies of data augmentation are more effective; 
(2) What is the scaling relationship between the amount of augmented data and model performance; and 
(3) Can data augmentation incentivize generalization to out-of-domain mathematical reasoning tasks?
To this end, we create a new dataset, AugGSM8K, by complicating and diversifying the queries from GSM8K and sampling multiple reasoning paths.
We obtained a series of LLMs called MuggleMath by fine-tuning on subsets of AugGSM8K. MuggleMath substantially achieves new state-of-the-art on GSM8K (from 54\% to 68.4\% at the scale of 7B, and from 63.9\% to 74.0\% at the scale of 13B).
A log-linear relationship is presented between MuggleMath’s performance and the amount of augmented data. 
We also find that MuggleMath is weak in out-of-domain math reasoning generalization to MATH. 
This is attributed to the differences in query distribution between AugGSM8K and MATH which suggest that augmentation on a single benchmark could not help with overall math reasoning performance.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper presented data augmentation method for LLM and studied augmentation is log linear to the amount of data added in the training. It found that augmentation didn't help on OOD case. Overall very interesting study and provided insights on data augmentation method.

### Strengths
- The experiments are comprehensive. The authors conducted lots of experiments to study data augmentation impact. 
- The analysis is well-organized and reasoning is sound

### Weaknesses
- Writing can be improved. Figure 1 caption has duplicated `LLaMA-2-7B`.

### Questions
In the paper, it shows that even on wrong problem, data augmentation still helps. It would be interesting to know if we have some perturbation on the test, will there be still any improvement?

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
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper presents a study on data augmentation in math reasoning, investigating the effectiveness of different strategies and the scaling relationship between the amount of augmented data and model performance. The authors created a new dataset, AugGSM8K, and obtained a series of LLMs called MuggleMath that achieved new state-of-the-art on GSM8K.

### Strengths
The authors conducted an investigation of different data augmentation strategies for math reasoning, including query augmentation, response augmentation, and both query and response augmentation. They evaluated the effectiveness of these strategies on the AugGSM8K dataset and analyzed the scaling relationship between the amount of augmented data and model performance.

The authors achieved new state-of-the-art results on the GSM8K dataset with their LLMs called MuggleMath. They fine-tuned MuggleMath on subsets of the AugGSM8K dataset. The authors' investigation of data augmentation strategies and their analysis of the scaling relationship between the amount of augmented data and model performance can help inform future research in this area and contribute to the development of more robust and generalizable models for math reasoning.

### Weaknesses
The authors did not compare their approach with other state-of-the-art approaches for math reasoning, which makes it difficult to assess the relative effectiveness of their approach. The authors generated multiple reasoning paths for each augmented problem, but did not provide a detailed analysis of the impact of different reasoning paths on model performance. A more in-depth analysis of reasoning paths could help identify which types of reasoning paths are most effective for different types of problems.

The impact of data augmentation on model interpretability was not thoroughly analyzed in the paper. The effects of data augmentation on the models' ability to provide explanations or justifications for their reasoning remain unclear.

Furthermore, the authors did not conduct a comprehensive review of the computational efficiency of their approach. Although they mentioned using proprietary models (GPT-3.5 and GPT-4) to implement five types of mathematical problem augmentation methods based on human experience in creating variations of mathematical problems, these models are known to be computationally intensive and demand significant computational resources. The computational cost of their approach could potentially limit its scalability to larger datasets or practical applications. A more thorough analysis of the computational efficiency of their approach could help identify potential bottlenecks and inform the development of more efficient models for mathematical reasoning.

### Questions
The authors did not juxtapose their approach with other cutting-edge methods for mathematical reasoning, making it difficult to gauge the relative efficacy of their solution. They generated numerous reasoning paths for each augmented problem, yet they did not provide an in-depth analysis of how different reasoning paths affect model performance. A deeper exploration of these reasoning paths could potentially determine the most effective paths for various problem types.

The influence of data augmentation on model interpretability was not adequately scrutinized in the paper. The implications of data augmentation on the capacity of models to furnish explanations or justifications for their reasoning are still ambiguous.

Moreover, the authors did not undertake a comprehensive assessment of the computational efficiency of their methodology. They mentioned the use of proprietary models (GPT-3.5 and GPT-4) to apply five types of mathematical problem augmentation methods, reflecting human experience in creating problem variations. These models are renowned for their computational intensity and substantial resource requirements. The computational expense of their methodology could potentially restrict its scalability to larger datasets or practical applications. A more exhaustive analysis of the computational efficiency of their approach could aid in identifying potential bottlenecks and guide the creation of more efficient models for mathematical reasoning.

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper studies data augmentation for mathematical reasoning by LLMs. In particular, the authors derive several augmentations for one of the standard benchmarks, GSM8k, by prompting GPT 3.5 and GPT 4 to generate more diverse queries and responses. It is then shown that fine-tuning on that augmented version of GSM8k brings improvements for the original GSM8k but does not really work for MATH, another benchmarking dataset for math reasoning.

### Strengths
**S1.** The authors created an augmented version of GSM8k that might be of some value for the benchmarking community.

### Weaknesses
**W1.** The paper lacks novelty and scientific contributions. The main message of the paper can be condensed to “Data augmentation works for IID setups but does not work in OOD setups” - this is well-known in the ML theory and I do not see what new contributions this paper brings to this problem. GSM8k and MATH are different enough to assume by default that augmentations of one dataset are unlikely to bring benefits for another. 

**W2.** The paper mostly cites LLM papers from 2021-2023 as if years of works on the ML theory of augmentation and their generalization capabilities do not exist. This is quite narrow-minded and leads to reinventing the wheel but now under the LLM umbrella. I would expect a more rigorous study of why augmentations do not help on MATH (and other math reasoning datasets if available) backed by the theory instead of hand-wavy conclusions “apply augmentations on all datasets” and “improve pre-traininig” offered by the authors.

**W3.** What is the value of the “scaling laws” for the augmented GSM8k (where overall sizes are 4K-100K data points)? The authors mention that the derived log-linear lines are unlikely to transfer to other datasets, so the lines merely show that bigger LLMs bring better performance, but there is no deeper implication of that.

### Questions
Q1. What are the "hidden representations of problems encoded by LLaMA2-7B" used for t-SNE given the causal architecture of Llama?

### Soundness
3 good

### Presentation
2 fair

### Contribution
1 poor

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
In this article the authors investigate the influence of a data augmentation technique on the performance of large language models for mathematical reasoning tasks. They augment the GSM8K dataset by prompting proprietary LLMs to 1) create variations of queries, changing one of 5 aspects of the query and 2) create variations of the responses. The authors derive scaling laws for model performance based on the amount of augmented data. They find a linear relationship between test set and training set accuracy and furthermore investigate the effect of multitask training with augmented data on performance on the MATH dataset, finding no evidence for improvement due to data augmentation on GSM8K.

### Strengths
-	The authors created AugGSM8K a LLM based augmentation of queries, that vary the original queries along 5 aspects.
-	The article contains a wide range of experiments, and the trained models achieve state of the art performance among open-source approaches.
-	The authors replicated (from earlier research) log-linear scaling laws of model performance on amounts of augmented queries and linear relationship between test set accuracy and training set accuracy for the chosen augmentation method.

### Weaknesses
-	The contributions of the paper are limited in the light of recent research. Similar scaling laws have been shown for similar augmentation techniques for the same datasets. The analysis of answer augmentation and query augmentation as well as their combination has also been explored in  Yu et al. 2023 "MetaMath: Bootstrap Your Own Mathematical Questions for Large Language Models" which has been cited in this article.
-	The evaluation on Out-of-Domain Generalization lacks evidence to support the strong claim in the title of the paper. Furthermore, the authors seem to be agnostic of previous research on Out-of-Domain generalization its well known challenges, and existing approaches. The title is also in so far misleading as it suggests there is no generalization effect between domains of mathematics (e.g., algebra vs calculus).
-	It is not clear how the reasoning paths have been sampled. A Case study on the prompting methods would improve the exposition.
- The writing of the paper could be improved.

### Questions
-	According to Fig. 5 there is an overlap between the distribution of MATH and GSM8K in the embedding space. Does the proposed augmentation method improve performance on this subset of MATH?
-	What is the generalization to other subsets of MATH (as listed in fig. 5)?
-	How do you employ human expertise and knowledge in mathematical problems for query augmentation (see paragraph 3.2)?

### Soundness
3 good

### Presentation
2 fair

### Contribution
1 poor
