# Bridging the Knowledge-Prediction Gap in LLMs on Multiple-Choice Questions

- Decision: Reject
- Scores: 2, 2, 6, 2

## Abstract
Large Language Models (LLMs) often fail on multiple-choice questions (MCQs) despite demonstrating correct knowledge in other contexts, such as free-form generation. To investigate the mechanism underlying this knowledge-prediction gap and alleviate it, we conduct a probing analysis on binary-choice questions and find that residual streams in certain layers contain a subspace spanned by two important bases: a knowledge basis that encodes the probability of the ground-truth answer and a prediction basis that encodes the probability of the answer choice predicted by the model.
We observe that incorrect predictions arise from a misalignment of the model's hidden states along these two bases.
Hence, we introduce KAPPA (Knowledge-Aligned Prediction through Projection-based Adjustment), an inference-time intervention that transforms hidden states to align the prediction coordinate with the knowledge coordinate.
Experiments on binary-choice reformulations of Big-Bench-Hard show that KAPPA substantially improves accuracy and consistently outperforms baselines.
KAPPA's benefit further extends to general MCQs, precisely mitigating the knowledge-predictino gap.
Our work provides a new geometric understanding of the knowledge-prediction gap and offers a practical method for better aligning model behavior with its latent knowledge.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work proposes a method for intervening on LLM inference by identifying separate basis vectors for an LLM's internal knowledge, and its task prediction. The work suggests that a major source of error during reasoning is not in the knowledge domain but the prediction. The LLM knows the necessary information, but somehow still does not make the correct prediction. The method aims to align these vectors so that prediction is more representative of its knowledge.

### Strengths
The paper is well laid out. The motivation behind the method in general, as well as many of its specific design choices, is clear. Experimental results are understandable and reasonably in-depth, with the exception of the free-form results.

### Weaknesses
1. Experimental results demonstrate that the method consistently underperforms several alternatives in the multiple-choice setting.
2. Some unsupported claims are made during the analysis of the main-table results. l. 308: "these probes serve approximate upper bounds for each model’s achievable performance without any external information". This phrase makes several key claims which seem problematic. 
(a) "approximate upper bounds": the probe is clearly not the upper bound, since fine-tuning outperforms (b) "without external information": presumably this condition is provided to exclude fine-tuning from the claim, but I'm not sure this makes sense. "External information" is not defined, and its not clear how fine-tuning introduces more information than training probes does: they both leverage information from their training set.
3. While results look better for free-form (since the probe is not applicable anymore), the results presented here are unclear. Which dataset is being tested on? Wouldn't it make more sense to test long-form QA tasks (e.g. summarization) in order to evaluate free-form response?
4. Justification for binarization of the datasets is never provided.

### Questions
1. l.393 "KAPPA’s success is not merely an artifact of manipulating binary choices but stems from an enhancement of the model’s ability to express its knowledge, indicating its potential to improve performance across a broader range of generative tasks". What do you mean by "manipulating binary choices"? This claim doesn't really connect with your previous argumentation that the fact that models will correctly answer in free-form and then mis-classify in a MCQA format, indicating prediction failure.

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper raises the issue of language models failing to answer question even when the correct information is contained within them. Authors apply probe analysis to the residual streams of transformer LLMs and discover the gap between subspaces representing the models actual knowledge and the answer it outputs for binary-choice questions; they show that linear probes from certain layers can achieve performance significantly higher than that of the model itself. Then, they propose an intervention technique that aligns the residual stream representations with the `knowledge' subspace while maximizing the preserved information.

Finally, authors apply their intervention techniques to two language models against competitive methods (activation steering, LoRA finetuning, etc.) and test them on several tasks, including binary question answering, free-form question answering, and general text generation. Results show that the proposed method noticeably improves the model's performance.

### Strengths
- A promising insight into the geometry of the model's "knowledge" in its internal representations.

- An interesting intervention method with well-developed mathematical background.

### Weaknesses
- The choice of title is somewhat misleading. Multiple-Choice Questions are explicitly stated in the title and abstract; however, the proposed method uses exclusively Binary choice questions to learn the transformation. Also, no evaluation on MCQA (questions with more than 2 options; for example, common MMLU) was reported in this paper. Can learning of the Knowledge-Aligned Prediction transformation be extended to general MCQA setup (e.g., 4-option questions)?

- Limited evaluation. Only two LLMs are explored, both of the same size (LLaMA2-7B and Qwen2.5-7B). It would be interesting to see how observed phenomena depend on the scale of the model; are there any qualitive changes. Also, some more recent models would've been much appreciated (LLaMA2 isn't a bad model, per se, but modern models usually perform much better on the reported benchmarks).

- Certain important details of the experimental setup are omitted.

	*) What is the size of train/validation/test subsets? Linear probes have quite a number of features (4096); what regularization was used to prevent overfitting (if any)?

 	*) What is the class balance in the Binary version of the datasets that were created/used in the experiments?

	*) For experiments on vicuna-eval (Section 3.4) could you provide more details (i.e., number of tasks per category, number of restarts for each prompt)? As of now, it is a little confusing. Reported overall performance doesn't seem to be an average of the performances for each category (46.9% vs. 45.2% for LoRA), and, at the same time, neither percentage of the number of tasks in the benchmark (80) is an integer.

- Reproducibility could not be assessed, because code wasn't provided.

### Questions
1. See Weaknesses

2. Please, make the fontsize on all figures larger.
3.  Please clarify in the caption for Table 2 whether you use -Binary versions of the datasets or not.
4. Could you please specify what model was used in the LLM-as-a-Judge setup for experiments with free-form generation?
5. Do you have any insights on how the proposed intervention affects the performance in the case of CoT or few-shot promptings?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces a method to address the knowledge-prediction gap in Large Language Models (LLMs) on multiple-choice questions, called KAPPA. The authors identify a two-dimensional subspace within LLM hidden states spanned by: (1) a knowledge basis encoding the probability of the correct answer, and (2) a prediction basis encoding the probability of the model's predicted answer. KAPPA intervenes at inference time by geometrically realigning these coordinates to make predictions more faithful to internal knowledge. The method improves LLMs performance across several datasets and extends to free-form generation.

### Strengths
- The paper provides a clear intuitive framework for understanding the knowledge-prediction gap through a two-dimensional subspace representation;
- The authors conduct extensive experiments including cross-dataset transfer, free-form generation, and general capability assessment;
- The paper is well-written and the main idea is easy to understand.

### Weaknesses
- The current method only handles binary-choice questions and don't consider more general MCQ scenarios with 4+ options.
- While the geometric framework is intuitive, the paper doesn't deeply explain why this particular misalignment occurs or what causes it during training. Understanding the root cause could lead to better solutions.
- The paper does not cite "Listening to the Wise Few: Select-and-Copy Attention Heads for Multiple-Choice QA" (Tulchinskii et al., 2024), which analyzed - prior to this work - the phenomenon that LLMs can encode correct knowledge internally yet fail to express it in MCQ predictions. This oversight misrepresents the novelty of the core observation and suggests incomplete literature review. 
- Experiments focus on only two 7B parameter models (LLaMA-2 and Qwen-2.5). I've worked with LLaMA 2 and observed clear differences in its performance on MCQ tasks compared with newer models. A newer models need to be included into this research. Also it would be good to check the models of more various sizes (less than 7B, more than 7B).

### Questions
- How does the method scale to multiple-choice questions with >2 options? Is there any barrier to try these scenarios?
- Have you investigated what properties of training data or procedures lead to this knowledge-prediction misalignment?
- Could this approach be combined with training-time interventions to prevent the gap from forming initially?
- How does performance vary with model scale? Do larger models exhibit smaller knowledge-prediction gaps?

### Soundness
3

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
3

### Summary
The authors introduce a new probing technique that dynamically updates the residual stream of an LLM. Then show that it can be used to improve outcomes on multiple choice benchmarks

### Strengths
The authors show that their method improves LLM performance on some benchmarks, and that the intervention appears to generalize slightly. I think the steering intervention as a correction in a learned plane is neat and should be explored more. The authors do a good job of exploring uses for their steering methods and are precise with describing its impact and limitations.

## originality

The steering vector contributions are novel as is the KAPPA technique, but both are directly building on a large body of existing techinques

 ## quality

The analysis and methods look sound, but the tests are limited to a few models and I still have questions about specific parts of the experiments

 ## clarity

I found this readable, but I found determining what exactly happened and what the impact was to be difficult. For example to me table 1 is of major importance for evaluating the results, and understanding if KAPPA generalizes. Showing each entry as the difference from the baseline would make reading it much easier.

## significance

I'm not convinced this method will work in general, so cannot say this is very significant, despite being intriguing

### Weaknesses
I don't think the method is as powerful as the authors claim. This feels like a different type of fine tuning (maybe a simple LORA), so the small boost in performance makes sense as you're "fine-tuning" general purpose models to answer questions better. I don't think it shows the claimed Knowledge dimension. I'm also skeptical of the claims this will generalize, either improving other benchmarks or work for more complex tasks (binary - -> multiple- choice for example), as they will require significantly more complex subspace. As an example multi-calibration is a very powerful technique that allows for similar strong claims about improving results via post processing, but the techniques are hamper if the number of classes that are being calibrated across get above certain thresholds. If I'm wrong about this please let me know.

### Questions
Could you explain more clearly what the "prediction basis" is? The description "encodes the probability of the model’s own choice" sounds  like a tautology. More generally I find the anthropomorphic language used in the paper to describe the models makes understanding more difficult.

In the intro line 47 you make the claim "We find that in the residual streams of certain layers, there exists a subspace spanned by two functionally distinct bases", can you explain how this proves there exists this subspace. And do you mean it exists in the LLM? The probes? or as a platonic form? 

Can this be interpreted as a very low rank LoRA?

### Soundness
2

### Presentation
2

### Contribution
2
