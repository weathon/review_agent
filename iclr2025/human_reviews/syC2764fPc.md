## Human Reviewer 1

### Summary
The authors propose a new method for adapting LLMs for time-series analysis tasks. They test on both time-series generation (forecasting) and time-series classification. Their paper proposes the use of Dual-Scale Context-Alignment GNNs which are used to perform context alignment of the time-series and text tokens/patches.
They combine this strategy with example prompts which they are calling (Demonstration Examples Prompt Technique) and show that using DECA and the dual scale context alignment improves the performance of LLM’s compared to existing LLM benchmarks for time-series analysis.

### Strengths
1. Their method of context alignment seems new and interesting
2. The course and fine-grained context alignments are logical and fit well in combination with patching which is a very popular embedding strategy for time-series data
3. They several different experiments using different tasks and experiment settings to show their model performs at a high level consistently

### Weaknesses
1. This paper focuses on enhancing LLM’s for time-series analysis and it takes a pretrained GPT model and compares it to other LLM-based approaches. This means this approach is ultimately a pretrained model approach (with the LLM being a pretrained model). The authors need to be comparing to other pretrained models and not just LLM based ones. Since its not being used for text generation-based tasks there’s no reason to limit this evaluation to only LLM based approaches and there are many pretrained models which follow the same experimental method.
2. The proposed method hinges heavily on the use of GNN’s and while they do run an ablation study, from what I understand they don’t compare other network architectures for performing the context alignment. Given that the GNN is a major part of the proposed implementation, an ablation showing other networks as course and fine-grained context aligners is necessary. For example, Linear layers, CNNs and self-attention could be a good start.
3. I would like more details on the training, their method does add a layer of complexity in adapting LLM’s for time-series analysis and the amount of added compute for finetuning these models (time and device) should be clearly stated.

### Questions
1. While your methodology considers the strengths of GNN’s as context aligners, I would like to see the choice of GNN’s to be validated experimentally as opposed to other options
2. In my opinion the terminology is unnecessarily complicated, and it takes away from the impact of the paper. One key example is “VCA w/o DSCA-GNNs” used in table 1. It seems to me that this is simply just prompting since there are no context aligners used. It may be more clear if you denote VCA as context alignment without DSCA-GNNs and mark in table 1 VCA + DSCA-GNNs. 
 3. What is the difference between “(DECA) DEMONSTRATION EXAMPLES BASED CONTEXT-ALIGNMENT” and few-shot prompting as in NLP. It seems like this term was invented to enhance the complexity of the paper but at the cost of reader understanding, especially if that reader has an NLP background.
4. How does the prompt effect the context alignment? Are there some prompts that harm context alignment? Since prompting is a key component of this paper, it would be useful to know how their new method performs with different prompt types.
5. Since DECA involves using examples for context alignment a fair comparison against other baselines would be the VCA with the GNN as a context aligner. I think that should be included beside DECA in the results chart

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
4

---

## Human Reviewer 2

### Summary
This paper introduces Context-Alignment, a novel paradigm for enhancing the capabilities of LLMs in time series tasks. The authors argue that leveraging LLMs' strengths in natural language processing, particularly their understanding of linguistic logic and structure, is key to improving their performance on time series data. They propose a Dual-Scale Context-Alignment Graph Neural Networks (DSCA-GNNs) framework that aligns time series data with linguistic components, enabling structural and logical alignment. This framework is used to develop Demonstration Examples based Context-Alignment (DECA), which integrates seamlessly into pre-trained LLMs to enhance their awareness of logic and structure. Extensive experiments across various time series tasks, including forecasting and classification, demonstrate DECA's effectiveness, especially in few-shot and zero-shot scenarios, highlighting the importance of context alignment in activating and enhancing LLMs' potential in time series applications.

### Strengths
The paper strengthens the representation of time series data through GNNs, addressing a gap in prior research on applying LLMs to time series tasks. The method shows promising results and could inspire future work in this area.

### Weaknesses
- The paper's incorporation of GNNs is a functional approach, yet it overlooks a comparison of modeling time consumption, which is a critical aspect of efficiency that should be measured against baselines that do not employ this method.
- Additionally, while the paper presents a generalized embedding technique, further validation across a broader range of time series scenarios is needed to establish its robustness. Testing the method in other contexts, such as time series anomaly detection and imputation, would strengthen the claims of its effectiveness.

### Questions
I have raised several concerns within the weaknesses. Please address the issues I've mentioned there.

### Soundness
2

### Presentation
2

### Contribution
3

### Rating
6

### Confidence
2

---

## Human Reviewer 3

### Summary
This paper aims at addressing the alignment of Time series not just on the token level as done by previous works but at a level that enable LLMs to contextualize and comprehend TS in the same manner as it does for natural language. To this aim, the authors proposed Dual-Scale Context-Alignment GNNs that achieves context level alignment comprising structural alignment and logical alignment thereby activating LLMs’ potential capabilities in time series tasks.

### Strengths
-  The paper addresses an important question of context alignment which promises better use of LLMs in the time series domain.
- The idea of using GNNs to introduce context alignment brings novelty to this work.

### Weaknesses
- Since the authors use GNNs for context alignment, a befitting diagram showing the nodes and edges would have made it easier to follow the text.
- Some typos here and there in the paper.
- Datasets not described clearly in the experimental setup.

### Questions
- I am not completely sure if averaging the error metrics across different prediction lengths is the best way to report and compare the results.
- Is there a justification as to why this averaging approach is adopted?

### Soundness
3

### Presentation
2

### Contribution
3

### Rating
6

### Confidence
2

---

## Human Reviewer 4

### Summary
This work aims at leveraging and improving LLMs for time series tasks. Specifically, the authors propose context alignment, a technique that utilizes dual-scale GNNs in addition to the basic LLM architectures that helps LLM comprehend time series data. Few-shot prompting techniques in regular LLMs are also used in the time series design. Through various experiments on different time series benchmarks, the authors show the performance advantage of the proposed method over baseline models.

### Strengths
This paper is tackling an interesting problem of leveraging and improving pretrained LLMs to do time series tasks. The authors explored a diverse set of benchmarks and baselines and showed a superior performance of the proposed method.

### Weaknesses
This work motivates and explains the advantage of the proposed method by LLMs' deep understanding of linguistic logic and structure rather than superficial embedding processing. However, such explanations lack support from experiments. More analysis can be included on the LLM side. For example, does an inferior LLM basic architecture (either old design or small model sizes) or a badly trained LLM (undertrained or untrained) lead to a bad time series performance?


This work freezes most of the LLM structure and tunes dual-scale context-alignment GNNs between Transformer layers. There seems to be lacking analysis on the effect of fully tuning the attention and feed-forward layers as well, and the subsequent necessity of the dual-scale context-alignment GNNs in that case. 


One of the interesting attributes of LLMs is the scaling effect (e.g., on model sizes, training data sizes, few-shot prompting amounts). The scaling aspect of LLMs and the proposed DECA seems to be unknown in the context of time series.


The writing can be improved for clarity, e.g., by providing qualitative examples besides key equations, especially in the few-shot prompting part, and adding an algorithm table.

### Questions
N/A

### Soundness
3

### Presentation
2

### Contribution
3

### Rating
6

### Confidence
2